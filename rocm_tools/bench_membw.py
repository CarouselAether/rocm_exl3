#!/usr/bin/env python3
"""Measure achievable GPU memory bandwidth, and whether buffer size changes it.

Motivation: on Strix Halo the BIOS carves out a dedicated VRAM aperture and
exposes the rest of system RAM to the GPU as GTT. Both are the same physical
LPDDR5X, but they are not necessarily reached at the same rate, and torch
reports the *GTT* pool as `total_memory` -- so a machine with a 512 MiB carveout
still advertises ~110 GiB of "VRAM" and gives no hint that models are landing in
GTT.

Check the split before trusting any throughput number:

    rocm-smi --showmeminfo all | grep -E 'VRAM Total Memory|GTT Total Memory'

This sweeps buffer sizes from well inside a small carveout to well outside it.
A flat curve means the distinction does not matter on this machine. A step down
past the carveout size means large allocations are being served more slowly, and
model throughput is bounded by the slower figure rather than the headline
LPDDR5X number.

Reported as read+write bytes moved / elapsed, median of repeats with the first
discarded, per the ~4.7% noise floor in gfx1151-hardware-facts.

    rocm_tools/bench_membw.py
    rocm_tools/bench_membw.py -s 64 256 1024 8192 -r 5
"""

import argparse
import os
import statistics
import sys
import time

import torch

NOISE_FLOOR = 0.05


def time_copy(nbytes, repeats):
    n = nbytes // 2                      # fp16 elements
    try:
        src = torch.empty(n, dtype=torch.half, device="cuda:0")
        dst = torch.empty(n, dtype=torch.half, device="cuda:0")
    except torch.OutOfMemoryError:
        return None
    src.normal_()
    samples = []
    for r in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dst.copy_(src)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if r == 0:
            continue                     # first run reads high
        samples.append(2 * nbytes / dt / 1e9)   # read + write
    del src, dst
    torch.cuda.empty_cache()
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--sizes_mib", type=int, nargs="+",
                    default=[16, 64, 128, 256, 512, 1024, 4096, 8192, 16384])
    ap.add_argument("-r", "--repeats", type=int, default=6)
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"  device: {p.name}   torch total_memory: {p.total_memory / 2**30:.1f} GiB")
    print(f"\n  d2d copy bandwidth, median of {args.repeats - 1} timed runs, first discarded\n")
    print(f"  {'buffer':>10}  {'GB/s':>9}  {'spread':>7}")

    for mib in args.sizes_mib:
        samples = time_copy(mib * 2**20, args.repeats)
        if not samples:
            print(f"  {mib:8} MiB   (out of memory)")
            continue
        med = statistics.median(samples)
        spread = (max(samples) - min(samples)) / med if med else 0.0
        flag = "  <- exceeds noise floor" if spread > NOISE_FLOOR else ""
        print(f"  {mib:8} MiB  {med:9.1f}  {spread:6.1%}{flag}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
