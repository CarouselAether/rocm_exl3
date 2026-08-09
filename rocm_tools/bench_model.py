#!/usr/bin/env python3
"""End-to-end prefill and token-generation throughput for a loaded model.

Reports what a user actually experiences, as opposed to the kernel-level numbers
from bench_moe.py / bench_prefill_tiles.py / bench_decode_splits.py.

Measurement conventions follow bench_moe.py, for the reasons in
gfx1151-hardware-facts: the benchmark noise floor on this hardware is ~4.7%
spread (1.6% stdev) and the first run reads high, so every figure is a median of
repeats with the first discarded, and any spread above 5% is flagged rather than
silently reported.

Two properties of this hardware set what is worth measuring:

  - prefill is (slightly) compute-bound, so it is reported per prompt length;
  - token generation is memory-bound, so it is reported at bsz=1, which is the
    case that matters and the one with the least parallelism to hide latency.

Prompts are built from random token ids, regenerated per repeat. Reusing one
prompt would let the generator's prefix cache serve later repeats (visible as
cached_tokens > 0) and turn the prefill measurement into a no-op.

    rocm_tools/bench_model.py -m /path/to/model
    rocm_tools/bench_model.py -m /path/to/model -p 512 2048 8192 -n 128 -r 5

Exits via os._exit() on purpose: any process here that has loaded a model
segfaults in native teardown after all work completes (see RDNA_NOTES.md), which
would otherwise mask the exit status of the benchmark itself.
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job

NOISE_FLOOR = 0.05


def rand_prompt(tokenizer, n_tokens, rng):
    """A prompt of exactly n_tokens ids, distinct per call so the prefix cache misses."""
    vocab = tokenizer.actual_vocab_size
    lo, hi = int(vocab * 0.05), int(vocab * 0.95)
    ids = torch.randint(lo, hi, (1, n_tokens), dtype=torch.long, generator=rng)
    return ids


def run_job(generator, ids, max_new_tokens):
    job = Job(input_ids=ids, max_new_tokens=max_new_tokens)
    generator.enqueue(job)
    result = None
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r["stage"] == "streaming" and r.get("eos", False):
                result = r
    return result


def summarize(samples):
    med = statistics.median(samples)
    spread = (max(samples) - min(samples)) / med if med else 0.0
    return med, spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-l", "--label", default=None)
    ap.add_argument("-p", "--prefill", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("-n", "--new_tokens", type=int, default=128)
    ap.add_argument("-r", "--repeats", type=int, default=4)
    ap.add_argument("-c", "--cache_size", type=int, default=None)
    args = ap.parse_args()

    label = args.label or os.path.basename(args.model_dir.rstrip("/"))
    max_prefill = max(args.prefill)
    cache_size = args.cache_size or (max_prefill + args.new_tokens + 256)
    # Floored at 4096: model.load() runs a dummy forward at max_chunk_size and asserts the
    # cache can hold it, so a cache sized only for a short prefill test fails during load.
    cache_size = max(cache_size, 4096)
    cache_size = -(-cache_size // 256) * 256      # Cache asserts a multiple of PAGE_SIZE

    print(f" -- loading {label}", flush=True)
    t0 = time.time()
    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=cache_size)
    model.load(progressbar=False)
    load_s = time.time() - t0
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    print(f" -- loaded in {load_s:.1f}s, cache {cache_size} tokens", flush=True)

    rng = torch.Generator().manual_seed(1234)
    rows = []

    # ---- prefill -------------------------------------------------------------
    for n in args.prefill:
        if n + args.new_tokens > cache_size:
            print(f" -- skipping prefill {n}: exceeds cache")
            continue
        samples, cached_seen = [], 0
        for r in range(args.repeats):
            ids = rand_prompt(tokenizer, n, rng)
            res = run_job(generator, ids, 1)
            if res is None:
                continue
            cached_seen = max(cached_seen, res.get("cached_tokens", 0))
            if r == 0:
                continue                      # first run reads high
            samples.append(res["prompt_tokens"] / (res["time_prefill"] + 1e-10))
        if not samples:
            continue
        med, spread = summarize(samples)
        rows.append(("prefill", n, med, spread, cached_seen))

    # ---- token generation ----------------------------------------------------
    samples = []
    for r in range(args.repeats):
        ids = rand_prompt(tokenizer, 128, rng)
        res = run_job(generator, ids, args.new_tokens)
        if res is None:
            continue
        if r == 0:
            continue
        samples.append(res["new_tokens"] / (res["time_generate"] + 1e-10))
    if samples:
        med, spread = summarize(samples)
        rows.append(("decode", args.new_tokens, med, spread, 0))

    # ---- report --------------------------------------------------------------
    print(f"\n=== {label} ===")
    print(f"  median of {args.repeats - 1} timed runs, first discarded\n")
    print(f"  {'phase':8} {'tokens':>7}  {'tok/s':>10}  {'spread':>7}")
    for phase, n, med, spread, cached in rows:
        flag = "  <- exceeds noise floor, rerun" if spread > NOISE_FLOOR else ""
        warn = "  <- PREFIX CACHE HIT, invalid" if cached else ""
        print(f"  {phase:8} {n:7}  {med:10.1f}  {spread:6.1%}{flag}{warn}")
    print(f"\n  RESULT {label}: " + " | ".join(
        f"{p}{n}={m:.1f}t/s" for p, n, m, _, _ in rows))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
