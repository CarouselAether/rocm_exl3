#!/usr/bin/env python3
"""ext.hgemm timing: WMMA GEMM backend (rocm/wmma_gemm_rdna.hip) vs hipBLAS.

    rocm_tools/bench_wmma_gemm.py                       # DS4 shape matrix, fp32 + fp16 out
    rocm_tools/bench_wmma_gemm.py -m 1792 -nk 4096,8192 # one shape

Both columns are ext.hgemm in the same process on the same buffers; the hipBLAS column
sets EXL3_ROCM_WMMA_GEMM=0 (read per call). B rotates over several buffers so large-K
shapes are not served from the 32 MiB Infinity Cache. Times are the mean of --iters
calls after warm-up, measured with HIP events.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3.ext import exllamav3_ext as ext

MS = [9, 64, 160, 255, 256, 1023, 1792]
NK = [(4096, 8192), (4096, 2048), (1024, 4096), (2048, 4096), (512, 4096), (256, 4096), (32768, 1024)]


def time_ms(fn, iters, rotate):
    for i in range(5):
        fn(i % rotate)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing = True)
    t1 = torch.cuda.Event(enable_timing = True)
    t0.record()
    for i in range(iters):
        fn(i % rotate)
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", type = int, nargs = "+", default = MS)
    ap.add_argument("-nk", type = str, nargs = "+", default = None, help = "N,K pairs")
    ap.add_argument("-i", "--iters", type = int, default = 30)
    ap.add_argument("-r", "--rotate", type = int, default = 4)
    ap.add_argument("--dtypes", type = str, default = "fp32,fp16")
    args = ap.parse_args()
    nks = [tuple(int(x) for x in s.split(",")) for s in args.nk] if args.nk else NK
    dtypes = [{"fp32": torch.float, "fp16": torch.half}[d] for d in args.dtypes.split(",")]

    prop = torch.cuda.get_device_properties(0)
    print(f"device {prop.name} {getattr(prop, 'gcnArchName', '?')}  torch {torch.__version__}")
    print(f"{'out':>4} {'M':>5} {'N':>6} {'K':>6} {'hipBLAS ms':>11} {'WMMA ms':>9} {'speedup':>8} {'WMMA TF':>8}")
    rows = []
    for (n, k) in nks:
        bs = [(torch.randn((k, n), device = "cuda") / k ** 0.5).half() for _ in range(args.rotate)]
        for m in args.m:
            a = (torch.randn((m, k), device = "cuda") * 0.5).half()
            for dt in dtypes:
                c = torch.empty((m, n), dtype = dt, device = "cuda")
                fn = lambda i: ext.hgemm(a, bs[i], c)
                os.environ["EXL3_ROCM_WMMA_GEMM"] = "0"
                t_h = time_ms(fn, args.iters, args.rotate)
                os.environ["EXL3_ROCM_WMMA_GEMM"] = "1"
                t_w = time_ms(fn, args.iters, args.rotate)
                tf = 2.0 * m * n * k / (t_w * 1e-3) / 1e12
                name = "fp32" if dt == torch.float else "fp16"
                print(f"{name:>4} {m:>5} {n:>6} {k:>6} {t_h:>11.3f} {t_w:>9.3f} {t_h / t_w:>7.2f}x {tf:>8.1f}", flush = True)
                rows.append((name, m, n, k, t_h, t_w))
            del a
        del bs
    os.environ.pop("EXL3_ROCM_WMMA_GEMM", None)
    for name in ("fp32", "fp16"):
        sel = [r for r in rows if r[0] == name]
        if sel:
            th = sum(r[4] for r in sel)
            tw = sum(r[5] for r in sel)
            slower = [f"{r[1]}x{r[2]}x{r[3]}" for r in sel if r[5] > r[4] * 1.02]
            print(f"{name}: total {th:.2f} ms hipBLAS -> {tw:.2f} ms WMMA ({th / tw:.2f}x); "
                  f"slower (>2%): {', '.join(slower) if slower else 'none'}")


if __name__ == "__main__":
    main()
