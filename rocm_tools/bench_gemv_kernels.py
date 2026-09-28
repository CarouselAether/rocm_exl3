#!/usr/bin/env python3
"""Per-kernel timing of the EXL3 decode GEMVs through the real extension entry
points (ext.exl3_gemm / ext.exl3_mgemm), on DS4-Flash 2.04bpw decode shapes.

Synthetic trellises (the kernels' cost does not depend on the values), ROT
rotating copies of each weight set so the working set is > 4x the 32 MiB
Infinity Cache (DRAM-resident, RDNA_NOTES "Hardware"). Each call is the whole
dispatch the model issues: input rotation + dot kernel + fused output epilogue
(and the weighted reduce for the routed down projection). Reports us/call,
GB/s of trellis and G weights/s, median of REPS timed loops.

    python rocm_tools/bench_gemv_kernels.py                       # current env
    python rocm_tools/bench_gemv_kernels.py --ab                  # EXL3_ROCM_GEMV_TILES=0 vs 1
    python rocm_tools/bench_gemv_kernels.py --json out.json

Env switches are re-read per call by the extension, so --ab measures both
cores in one process and one build.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3.ext import exllamav3_ext as ext

DEV = torch.device("cuda:0")
CACHE_BYTES = 160 << 20

# name, kind, k, n, K, nmat (slots), m, fp32, mul1, weighted
SHAPES = [
    ("routed gate/up K2 4096->2048 x6", "multi", 4096, 2048, 2, 6, 1, False, True, False),
    ("routed gate/up K2 m=3",           "multi", 4096, 2048, 2, 6, 3, False, True, False),
    ("routed down K2 2048->4096 x6 w",  "down",  2048, 4096, 2, 6, 1, True,  True, True),
    ("wo_a K4 4096->1024 x8",           "multi", 4096, 1024, 4, 8, 1, False, True, False),
    ("wo_a K4 m=3",                     "multi", 4096, 1024, 4, 8, 3, False, True, False),
    ("shared gate/up K4 4096->2048 x2", "multi", 4096, 2048, 4, 2, 1, True,  True, False),
    ("wo_b K4 8192->4096",              "single", 8192, 4096, 4, 1, 1, True,  True, False),
    ("wo_b K4 m=3",                     "single", 8192, 4096, 4, 1, 3, True,  True, False),
    ("wq_b K4 1024->32768",             "single", 1024, 32768, 4, 1, 1, False, True, False),
    ("wq_b K4 m=3",                     "single", 1024, 32768, 4, 1, 3, False, True, False),
    ("shared down K4 2048->4096",       "single", 2048, 4096, 4, 1, 1, True,  True, False),
    ("head K6 4096->129280",            "single", 4096, 129280, 6, 1, 1, False, True, False),
]


def trellis(k, n, K, g):
    t = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype=torch.int32, generator=g)
    return t.to(torch.int16).to(DEV)


def vec(n, g, scale=0.5):
    return (torch.randn((n,), generator=g, dtype=torch.float32) * scale).half().to(DEV)


def build(shape, g):
    name, kind, k, n, K, nmat, m, fp32, mul1, weighted = shape
    set_bytes = k * n * K // 8 * nmat
    rot = max(2, min(64, CACHE_BYTES // set_bytes + 1))
    ctype = torch.float if fp32 else torch.half
    calls = []
    if kind == "single":
        A = (torch.randn((m, k), generator=g) * 0.25).half().to(DEV)
        A_had = torch.empty_like(A)
        C = torch.empty((m, n), dtype=ctype, device=DEV)
        for _ in range(rot):
            B, suh, svh = trellis(k, n, K, g), vec(k, g), vec(n, g)
            calls.append(lambda B=B, suh=suh, svh=svh: ext.exl3_gemm(A, B, C, suh, A_had, svh, -1, False, mul1, 0))
    else:
        e = nmat
        bszm_in = e if kind == "down" else 1
        A = (torch.randn((bszm_in, m, k), generator=g) * 0.25).half().to(DEV)
        A_had = torch.empty((e, m, k), dtype=torch.half, device=DEV)
        C = torch.empty((e, m, n), dtype=ctype, device=DEV)
        idx = torch.arange(e, device=DEV).view(1, e)
        w = (torch.rand((1, e), generator=g) / e).half().to(DEV) if weighted else None
        for _ in range(rot):
            Bs = [trellis(k, n, K, g) for _ in range(e)]
            suhs = [vec(k, g) for _ in range(e)]
            svhs = [vec(n, g) for _ in range(e)]
            keep = (Bs, suhs, svhs)
            Bt = torch.tensor([b.data_ptr() for b in Bs], dtype=torch.long, device=DEV)
            suht = torch.tensor([s.data_ptr() for s in suhs], dtype=torch.long, device=DEV)
            svht = torch.tensor([s.data_ptr() for s in svhs], dtype=torch.long, device=DEV)
            calls.append(lambda Bt=Bt, suht=suht, svht=svht, keep=keep:
                         ext.exl3_mgemm(A, Bt, C, suht, A_had, svht, idx, w, K, -1, False, mul1,
                                        -1, -1, 0, 1, None, None))
    return calls, set_bytes, k * n * nmat, C


def time_calls(calls, reps):
    for c in calls:
        c()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        e0.record()
        for c in calls:
            c()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / len(calls))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", action="store_true", help="EXL3_ROCM_GEMV_TILES=0 vs 1 in one process")
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    modes = [("off", "0"), ("on", "1")] if args.ab else [("cur", None)]
    g = torch.Generator(device="cpu").manual_seed(0)
    rows = []
    hdr = f"{'shape':36s}" + "".join(f" {m + ' us':>9s} {m + ' GB/s':>9s}" for m, _ in modes)
    if args.ab:
        hdr += f" {'on Gw/s':>9s} {'speedup':>8s}"
    print(hdr)
    for shape in SHAPES:
        if args.filter and args.filter not in shape[0]:
            continue
        calls, nbytes, nweights, _ = build(shape, g)
        row = {"shape": shape[0], "bytes": nbytes, "weights": nweights}
        line = f"{shape[0]:36s}"
        for mname, val in modes:
            if val is not None:
                os.environ["EXL3_ROCM_GEMV_TILES"] = val
            us = time_calls(calls, args.reps)
            row[mname + "_us"] = us
            line += f" {us:9.1f} {nbytes / us / 1e3:9.1f}"
        if args.ab:
            line += f" {nweights / row['on_us'] / 1e3:9.1f} {row['off_us'] / row['on_us']:7.2f}x"
        print(line)
        rows.append(row)
        del calls
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
