#!/usr/bin/env python3
"""Drive one measured region for rocprofv3 --selected-regions (Phase 0.5 / 0.7).

Loads the model exactly as bench/run_bench.py does (model_init.init with the server's
arguments), runs warmups outside the region, then brackets only the measured work with
roctxProfilerResume(0) / roctxProfilerPause(0). Every generator step inside the region is
wrapped in a roctx range "step" so the analyzer can split the trace per token.

Modes:
  decode   prompt of --ctx random tokens; region = the next --steps decode steps
  prefill  region = one --ctx-token prefill (1 new token)
  both     prefill region, then decode region (two regions, one load)

    rocprofv3 --selected-regions --kernel-trace --hip-trace --marker-trace --memory-copy-trace \
        --output-format csv -d OUT -- python bench/profile_region.py -m MODEL --mode decode

Without rocprofv3 the roctx calls are no-ops, so the script also runs standalone.
"""

import argparse
import ctypes
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

# Under rocprofv3, importing triton segfaults: libtriton.so statically links LLVM, and its
# static initializers (PassBuilder.cpp's DebugCounter registry) bind to the LLVM that the
# injected rocprofiler tool already loaded, then free() a foreign pointer. Loading libtriton
# first with RTLD_DEEPBIND makes it resolve against its own LLVM. Verified 2026-09-27 with
# triton-rocm 3.8.0+gitc01b6774 / rocprofv3 1.3.5: JIT kernel compiles, runs, and is traced.
_fl = sys.getdlopenflags()
sys.setdlopenflags(_fl | os.RTLD_DEEPBIND)
import triton._C.libtriton  # noqa: E402,F401
sys.setdlopenflags(_fl)

import torch  # noqa: E402
from exllamav3 import model_init, Generator, Job  # noqa: E402
from exllamav3.generator.sampler import ArgmaxSampler  # noqa: E402


def load_roctx():
    import glob
    cands = glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "_rocm_sdk_core", "lib",
                                   "librocprofiler-sdk-roctx.so.1"))
    for c in cands + ["librocprofiler-sdk-roctx.so.1"]:
        try:
            lib = ctypes.CDLL(c)
            lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
            return lib
        except OSError:
            continue
    return None


RTX = load_roctx()


def resume():
    torch.cuda.synchronize()
    if RTX:
        RTX.roctxProfilerResume(0)


def pause():
    torch.cuda.synchronize()
    if RTX:
        RTX.roctxProfilerPause(0)


def push(name):
    if RTX:
        RTX.roctxRangePushA(name.encode())


def pop():
    if RTX:
        RTX.roctxRangePop()


def rand_ids(tokenizer, n, seed):
    g = torch.Generator().manual_seed(seed)
    v = tokenizer.actual_vocab_size
    return torch.randint(int(v * 0.05), int(v * 0.95), (1, n), dtype=torch.long, generator=g)


def run_to_end(gen):
    while gen.num_remaining_jobs():
        gen.iterate()


def prefill_region(gen, tok, ctx, seed):
    gen.enqueue(Job(input_ids=rand_ids(tok, ctx, seed), max_new_tokens=1))
    resume()
    push("prefill")
    t0 = time.perf_counter()
    while gen.num_remaining_jobs():
        push("step")
        gen.iterate()
        pop()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    pop()
    pause()
    print(f" -- prefill region: {ctx} tokens in {dt * 1000:.1f} ms ({ctx / dt:.1f} t/s wall)", flush=True)


def decode_region(gen, tok, ctx, steps, seed, sampler):
    # A step can emit several tokens under MTP / draft decoding (up to ndt + 1): size the job
    # so it outlasts the region, or the region's tail steps are idle generator iterations
    # (the job finished) that dilute every per-step figure. The untraced remainder runs off
    # in run_to_end.
    gen.enqueue(Job(input_ids=rand_ids(tok, ctx, seed), max_new_tokens=steps * 8 + 8, sampler=sampler))
    # run until the first token is out (prefill done), untraced
    first = False
    while not first:
        for r in gen.iterate():
            if r["stage"] == "streaming":
                first = True
    resume()
    push("decode")
    t0, n = time.perf_counter(), 0
    for _ in range(steps):
        push("step")
        for r in gen.iterate():
            if r["stage"] == "streaming":
                n += r.get("token_ids", torch.empty(1, 1)).shape[-1] if "token_ids" in r else 1
        pop()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    pop()
    pause()
    run_to_end(gen)
    print(f" -- decode region: {steps} steps, ~{n} tokens in {dt * 1000:.1f} ms "
          f"({dt / steps * 1000:.2f} ms/step, {n / dt:.2f} t/s wall)", flush=True)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-cs", "--cache_size", type=int, default=65536)
    ap.add_argument("--mode", choices=["decode", "prefill", "both"], default="decode")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("-ndt", type=int, default=2)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra model_init args, as run_bench.py (e.g. --extra -ngr)")
    a = ap.parse_args()

    ia = ["-m", a.model_dir, "-cs", str(a.cache_size)] + (["--mtp", "-ndt", str(a.ndt)] if a.mtp else [])
    ia += a.extra
    p = argparse.ArgumentParser()
    model_init.add_args(p, cache=True, add_sampling_args=False, add_draft_model_args=True,
                        default_autosplit_max_batch_size=4)
    ia = p.parse_args(ia)
    model, config, cache, tok, dm, _dc, dcache = model_init.init(ia)
    gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dcache,
                    num_draft_tokens=ia.num_draft_tokens)
    print(f" -- roctx {'loaded' if RTX else 'NOT loaded (standalone)'}", flush=True)

    # warmups outside any region: graph capture, Triton JIT, autotune
    gen.enqueue(Job(input_ids=rand_ids(tok, a.ctx, 1), max_new_tokens=16))
    run_to_end(gen)
    sampler = ArgmaxSampler() if a.mtp else None
    if a.mode in ("prefill", "both"):
        prefill_region(gen, tok, a.ctx, 2)
    if a.mode in ("decode", "both"):
        decode_region(gen, tok, a.ctx, a.steps, 3, sampler)
    sys.stdout.flush()
    # libc exit(), not os._exit(): rocprofv3 writes its output from a C atexit handler, which
    # os._exit() skips (the trace came back empty). exit() still skips Python finalization,
    # where the post-load native teardown segfault lives.
    ctypes.CDLL(None).exit(0)


if __name__ == "__main__":
    main()
