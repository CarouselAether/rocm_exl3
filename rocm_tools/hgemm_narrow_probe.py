#!/usr/bin/env python3
"""Narrow fp16 GEMMs: rocBLAS vs hipBLASLt vs the port's own GEMV kernels, per (m, N).

rocBLAS's legacy kernels answer a (1 x 3072) @ (3072 x 48) fp16 product with a
single 128x128 tile workgroup walking K serially (~63-74 us on gfx1151 for a
295 KB matrix, ~50x its bandwidth bound). That is BC_Attention's headwise gate
on Laguna, once per layer per token. ROCBLAS_USE_HIPBLASLT=1 fixes the narrow
shapes but costs prefill 13-17% on this part (hipBLASLt loses at m >= 128 and
at wide N), so rocm/hgemm_rdna.hip runs m <= 8, N <= 256 on its own split-K
GEMV kernels and leaves the rest on rocBLAS.

    rocm_tools/hgemm_narrow_probe.py [-k 3072] [-n 16 ... 2048] [-m 1 2 4 8] [--check]
    ROCBLAS_USE_HIPBLASLT=1 rocm_tools/hgemm_narrow_probe.py    # the hipBLASLt alternative
    EXL3_RDNA_HGEMM_NARROW_N=0 rocm_tools/hgemm_narrow_probe.py # ext on plain rocBLAS

For every (m, N) it times torch.matmul (fp16 through the library torch uses),
an fp32 GEMV recipe (torch.mv / broadcast multiply + sum; the ATen path this
port briefly shipped and withdrew), and ext.hgemm itself, which is the path the
model takes. --check also compares ext.hgemm against an fp32 reference at every
shape (fp16 and fp32 outputs) and fails loudly on a mismatch. Run this on a new
GPU or runtime before assuming either library's behaviour carries.

Working sets rotate across several weight buffers so the numbers are not
Infinity-Cache numbers (RDNA_NOTES: "There is a 32 MiB Infinity Cache").
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def timeit(fn, iters, rotate):
    for _ in range(10):
        fn(0)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(iters):
        fn(i % rotate)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--size_k", type = int, default = 3072)
    ap.add_argument("-n", "--sizes_n", type = int, nargs = "+",
                    default = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048])
    ap.add_argument("-m", "--sizes_m", type = int, nargs = "+", default = [1, 2, 4, 8])
    ap.add_argument("-r", "--rotate", type = int, default = 8, help = "distinct weight buffers per shape")
    ap.add_argument("-i", "--iters", type = int, default = 200)
    ap.add_argument("--check", action = "store_true", help = "verify ext.hgemm against fp32 at every shape")
    args = ap.parse_args()

    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}   torch {torch.__version__}   hip {torch.version.hip}")
    try:
        from exllamav3.ext import exllamav3_ext as ext
        have_ext = True
    except Exception as e:
        print(f"(ext not importable: {e}; ext.hgemm column skipped)")
        have_ext = False
    print(f"ROCBLAS_USE_HIPBLASLT={os.environ.get('ROCBLAS_USE_HIPBLASLT', '(unset)')}   EXL3_RDNA_HGEMM_NARROW_N={os.environ.get('EXL3_RDNA_HGEMM_NARROW_N', '(default 256)')}   check={args.check}")
    print()
    K = args.size_k
    best = {}
    hdr = f"{'m':>2} {'N':>5} {'GEMM fp16 us':>13} {'GEMV fp32 us':>13} {'ratio':>6} {'ext.hgemm us':>13}"
    print(hdr)
    for m in args.sizes_m:
        for N in args.sizes_n:
            ws = [torch.randn(K, N, device = dev, dtype = torch.half) for _ in range(args.rotate)]
            xs = [torch.randn(m, K, device = dev, dtype = torch.half) for _ in range(args.rotate)]
            c16 = torch.empty(m, N, device = dev, dtype = torch.half)

            def gemm(i):
                return torch.matmul(xs[i], ws[i])

            if m == 1:
                def gemv(i):
                    return torch.mv(ws[i].t().contiguous().float(), xs[i][0].float()).half()
            else:
                def gemv(i):
                    return (xs[i].float().unsqueeze(1) * ws[i].t().contiguous().float()).sum(-1).half()

            if args.check and have_ext:
                ref = (xs[0].float() @ ws[0].float())
                worst = 0.0
                for out_dtype in (torch.half, torch.float):
                    co = torch.empty(m, N, device = dev, dtype = out_dtype)
                    ext.hgemm(xs[0], ws[0], co)
                    torch.cuda.synchronize()
                    rel = ((co.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
                    worst = max(worst, rel)
                assert worst < 2e-2, f"ext.hgemm mismatch at m={m} N={N}: max rel err {worst:.3e}"
            t_gemm = timeit(gemm, args.iters, args.rotate)
            t_gemv = timeit(gemv, args.iters, args.rotate)
            t_ext = timeit(lambda i: ext.hgemm(xs[i], ws[i], c16), args.iters, args.rotate) if have_ext else float("nan")
            ratio = t_gemm / t_gemv
            if ratio > 1.10:
                best[m] = N
            print(f"{m:>2} {N:>5} {t_gemm:>13.1f} {t_gemv:>13.1f} {ratio:>6.2f} {t_ext:>13.1f}", flush = True)
        print()

    print("largest N where the fp32 GEMV still beats the fp16 GEMM by >10%:")
    for m in args.sizes_m:
        print(f"  m={m}: N <= {best.get(m, 0)}")
    rec = max((best.get(m, 0) for m in args.sizes_m), default = 0)
    print(f"\nGEMV recipe still worth it on this stack: {'yes, up to N=' + str(rec) if rec else 'no'}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
