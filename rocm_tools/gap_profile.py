"""GPU idle-gap accounting for decode: the direct evidence for/against graphs.

Graph replay can reclaim exactly two things: (a) GPU idle gaps between kernels
caused by per-launch submission latency, and (b) host CPU time spent in launch
calls. Neither shows in aggregate busy%. This tool exports a Kineto chrome
trace of N decode steps (profile_decode.py's method: 1-token prompt, warmup
first so BC graphs/autotune don't pollute the window), then walks the GPU
timeline: union of kernel intervals vs wall span, per-gap histogram, and the
largest gaps with their neighboring kernels so big host stalls can be told
apart from per-launch overhead. Launch-shaped gaps (< 50us) are the ONLY part
graphs can win back on the device side.
"""

import argparse
import json
import os
import statistics
import sys
import time

REPO = os.environ.get("EXL3_REPO", "/home/carousel/Desktop/exlproject/rocm_exl3_714")
sys.path.insert(0, REPO)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-n", "--new_tokens", type=int, default=32)
    args = ap.parse_args()

    label = "graphs-ON" if os.environ.get("EXL3_ROCM_HIP_GRAPHS") == "1" else "graphs-OFF"
    print(f" -- loading ({label})", flush=True)
    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    rng = torch.Generator().manual_seed(7)
    vocab = tokenizer.actual_vocab_size
    warm = torch.randint(int(vocab * .05), int(vocab * .95), (1, 64),
                         dtype=torch.long, generator=rng)
    decode_n(generator, warm, 8)

    one = torch.randint(int(vocab * .05), int(vocab * .95), (1, 1),
                        dtype=torch.long, generator=rng)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        res = decode_n(generator, one, args.new_tokens)
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    n = res["new_tokens"]

    trace_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              f"gap_trace_{label}.json")
    prof.export_chrome_trace(trace_path)
    evs = json.load(open(trace_path)).get("traceEvents", [])

    GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}
    kern = [(e["ts"], e["ts"] + e["dur"], e.get("name", "?"))
            for e in evs
            if e.get("ph") == "X" and e.get("cat") in GPU_CATS and e.get("dur", 0) > 0]
    kern.sort()
    if not kern:
        print("NO GPU EVENTS IN TRACE"); os._exit(1)

    # Union coverage of GPU busy intervals (streams merged: idle means NO
    # engine was running anything)
    span0, span1 = kern[0][0], max(e[1] for e in kern)
    span = span1 - span0
    busy = 0
    gaps = []          # (gap_us, prev_kernel, next_kernel)
    cur_s, cur_e, cur_name = kern[0][0], kern[0][1], kern[0][2]
    prev_name = cur_name
    for s, e, name in kern[1:]:
        if s <= cur_e:                      # overlaps/extends current busy block
            if e > cur_e:
                cur_e, prev_name = e, name
        else:
            gaps.append((s - cur_e, prev_name, name))
            busy += cur_e - cur_s
            cur_s, cur_e, prev_name = s, e, name
    busy += cur_e - cur_s

    launch_gaps = [g for g in gaps if g[0] < 50]
    big_gaps = [g for g in gaps if g[0] >= 50]
    lg_sum = sum(g[0] for g in launch_gaps)
    bg_sum = sum(g[0] for g in big_gaps)

    print(f"\n=== {label}: {n} decode tokens ===")
    print(f"  wall (host)          {wall*1e3:9.1f} ms  ({n/wall:.1f} tok/s)")
    print(f"  GPU span             {span/1e3:9.1f} ms")
    print(f"  GPU busy (union)     {busy/1e3:9.1f} ms  ({busy/span:.1%} of span)")
    print(f"  kernel launches      {len(kern):9d}   ({len(kern)/n:.0f}/token)")
    print(f"  inter-kernel gaps    {len(gaps):9d}")
    print(f"  launch-shaped (<50us){len(launch_gaps):9d}   sum {lg_sum/1e3:8.2f} ms "
          f"({lg_sum/span:.2%} of span)  <- graphs' max device-side win")
    if launch_gaps:
        vals = [g[0] for g in launch_gaps]
        print(f"      median {statistics.median(vals):.1f} us   mean {statistics.mean(vals):.1f} us")
    print(f"  big gaps (>=50us)    {len(big_gaps):9d}   sum {bg_sum/1e3:8.2f} ms "
          f"({bg_sum/span:.2%} of span)  <- host logic, not launch overhead")
    print(f"\n  largest gaps (us) and neighbors:")
    for g, a, b in sorted(big_gaps, reverse=True)[:6]:
        print(f"    {g:9.0f}  after {a[:40]:<40} before {b[:40]}")
    print(f"\n  per token: busy {busy/n/1e3:.2f} ms | launch-gap {lg_sum/n/1e3:.3f} ms "
          f"| big-gap {bg_sum/n/1e3:.3f} ms", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
