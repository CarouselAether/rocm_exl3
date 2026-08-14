#!/usr/bin/env python3
"""Where does decode time actually go?

bench_compute.py shows the quantized linears reach ~76% of achievable bandwidth
at bsz=1, yet whole-model decode lands far below what that per-layer rate
implies. The missing time is somewhere other than the EXL3 GEMMs, and guessing
which module is responsible has a poor track record on this port.

The first question is not "which kernel is slow" but "is the GPU even busy":

  - total kernel time ~= wall time  -> GPU-bound; look at the kernel table
  - total kernel time <<  wall time -> the GPU is idle waiting on the host, and
                                       no kernel optimisation will help

Reports both, then the top kernels by GPU time.

    rocm_tools/profile_decode.py -m /path/to/model
    rocm_tools/profile_decode.py -m /path/to/model -n 32 -k 25
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.profiler import profile, ProfilerActivity
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job


def decode_n(generator, ids, n):
    job = Job(input_ids=ids, max_new_tokens=n)
    generator.enqueue(job)
    res = None
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r["stage"] == "streaming" and r.get("eos", False):
                res = r
    return res


# The profiled job's prompt must be ONE token. The Job runs its prompt's
# prefill inside the same iterate() loop as decode, so a profiler around the
# job counts prefill kernels in the "decode" table -- bleed-through that has
# caused two wrong localizations (cooperative kernels blamed for Gemma decode;
# exl3_moe read as a ~23 ms per-token decode cost on DS4 when it is one
# prefill call per MoE layer per chunk). Starting the Kineto session
# mid-generation captures nothing on ROCm, so the window cannot simply open
# after prefill; a 1-token prompt makes "prefill" a single bsz-1 forward with
# the same kernel mix as a decode step (~1/n of the window, same shapes).


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-n", "--new_tokens", type=int, default=32)
    ap.add_argument("-k", "--top", type=int, default=20)
    args = ap.parse_args()

    print(f" -- loading {os.path.basename(args.model_dir.rstrip('/'))}", flush=True)
    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    rng = torch.Generator().manual_seed(7)
    vocab = tokenizer.actual_vocab_size
    mk = lambda: torch.randint(int(vocab * .05), int(vocab * .95), (1, 64),
                               dtype=torch.long, generator=rng)

    # Warm up: first pass builds the BC-attn graphs and runs any autotuning, which
    # would otherwise dominate a short profile.
    decode_n(generator, mk(), 8)

    one = torch.randint(int(vocab * .05), int(vocab * .95), (1, 1),
                        dtype=torch.long, generator=rng)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        res = decode_n(generator, one, args.new_tokens)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n = res["new_tokens"]
    ka = prof.key_averages()
    gpu_us = sum(getattr(e, "self_device_time_total", 0) or 0 for e in ka)
    gpu_s = gpu_us / 1e6

    print(f"\n=== decode profile: {n} tokens ===\n")
    print(f"  wall time            {wall:8.3f} s   ({n / wall:.2f} tok/s)")
    print(f"  total GPU kernel time{gpu_s:8.3f} s")
    print(f"  GPU busy             {gpu_s / wall:8.1%} of wall")
    idle = wall - gpu_s
    print(f"  GPU idle             {idle:8.3f} s   ({idle / wall:.1%})")
    print(f"\n  -> {'GPU-bound; see kernel table' if gpu_s / wall > 0.8 else 'HOST-BOUND: the GPU is waiting on the CPU'}")

    print(f"\n=== top {args.top} kernels by GPU time ===\n")
    rows = sorted(ka, key=lambda e: -(getattr(e, "self_device_time_total", 0) or 0))
    print(f"  {'kernel':<58} {'GPU ms':>9} {'%':>6} {'calls':>7}")
    for e in rows[:args.top]:
        t = (getattr(e, "self_device_time_total", 0) or 0) / 1e3
        if t <= 0:
            continue
        name = e.key if len(e.key) <= 57 else e.key[:54] + "..."
        print(f"  {name:<58} {t:9.1f} {t / (gpu_s * 1e3) if gpu_s else 0:6.1%} {e.count:7}")

    print(f"\n=== top 10 host-side ops by CPU time ===\n")
    rows = sorted(ka, key=lambda e: -(getattr(e, "self_cpu_time_total", 0) or 0))
    print(f"  {'op':<58} {'CPU ms':>9} {'calls':>7}")
    for e in rows[:10]:
        t = (getattr(e, "self_cpu_time_total", 0) or 0) / 1e3
        if t <= 0:
            continue
        name = e.key if len(e.key) <= 57 else e.key[:54] + "..."
        print(f"  {name:<58} {t:9.1f} {e.count:7}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
