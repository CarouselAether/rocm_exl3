#!/usr/bin/env python3
"""Measure the WMMA GEMM backend's configs against hipBLAS on this GPU and write the local
selection table exllamav3/exllamav3_ext/rocm/wmma_gemm_tuned_<arch>.json.

    python3 rocm_tools/gen_wmma_gemm_table.py --tune   # every config compiled for every arch
    <rebuild the extension>
    rocm_tools/tune_wmma_gemm.py                        # ~45 min on gfx1151; extends an
                                                        # existing table, so a re-run with
                                                        # new -m/-n/-k only measures those
    python3 rocm_tools/gen_wmma_gemm_table.py           # fold the results into the table
    <rebuild>

Why: rocm_wmma_gemm's shipped tables only hold shapes with M, N >= 1024. Its selection rule
(closest K, then closest M,N) extrapolates those big tiles down to prefill's small-M expert
and chunk-tail GEMMs, where a 4-16 workgroup grid walks all of K serially and loses to
hipBLAS (measured 0.4-0.8x at M <= 256, N <= 1024), while a smaller tile wins 2-3x there.

For every (M, N, K) on the grid below: hipBLAS fp32- and fp16-output time
(EXL3_ROCM_WMMA_GEMM=0), every compiled config's fp32-output time (EXL3_ROCM_WMMA_GEMM_CFG=i),
then the fp32-best config's fp16-output time. The entry records the best config and route
flags: fp32/fp16 output is routed to WMMA only where it beat hipBLAS by > --margin.
Results are appended to a .partial file as they come, so an interrupted run resumes.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3.ext import exllamav3_ext as ext

HERE = os.path.dirname(os.path.abspath(__file__))
ROCM = os.path.join(os.path.dirname(HERE), "exllamav3", "exllamav3_ext", "rocm")
ARCH_BIT = {"gfx1151": 1, "gfx1100": 2, "gfx1101": 2}
TABLE_ARCH = {"gfx1151": "gfx1151", "gfx1100": "gfx1100", "gfx1101": "gfx1100"}
KEYS = ("warps_m", "warps_n", "warp_tile_m", "warp_tile_n", "k_slices", "single_buffer", "swizzle", "bits")

GRID_M = [16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
GRID_N = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
GRID_K = [512, 1024, 2048, 4096, 8192]


def time_ms(fn, budget_ms = 40.0, max_iters = 20):
    fn(0)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing = True)
    t1 = torch.cuda.Event(enable_timing = True)
    t0.record()
    fn(1)
    t1.record()
    torch.cuda.synchronize()
    est = max(t0.elapsed_time(t1), 1e-3)
    iters = int(max(3, min(max_iters, budget_ms / est)))
    t0.record()
    for i in range(iters):
        fn(i)
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def compiled_configs(bit):
    src = open(os.path.join(ROCM, "wmma_gemm_table_rdna.hip.h")).read()
    cfgs = [tuple(int(x) for x in m.group(1).split(","))
            for m in re.finditer(r"\{ ([\d, ]+), exl3_wmma_launch_c\d+ \}", src)]
    return [(i, c[:8]) for i, c in enumerate(cfgs) if c[8] & bit]


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margin", type = float, default = 0.03, help = "required win over hipBLAS to route")
    ap.add_argument("-m", type = int, nargs = "+", default = GRID_M)
    ap.add_argument("-n", type = int, nargs = "+", default = GRID_N)
    ap.add_argument("-k", type = int, nargs = "+", default = GRID_K)
    ap.add_argument("-o", "--out", default = None)
    args = ap.parse_args()

    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ARCH_BIT:
        raise SystemExit(f"{arch}: no WMMA GEMM table for this arch")
    out = args.out or os.path.join(ROCM, f"wmma_gemm_tuned_{TABLE_ARCH[arch]}.json")
    partial = out + ".partial"
    cfgs = compiled_configs(ARCH_BIT[arch])
    print(f"{arch}: {len(cfgs)} compiled configs; writing {out}", flush = True)
    if len(cfgs) < 12:
        print("  (few configs: was the extension built from gen_wmma_gemm_table.py --tune?)")

    done = {}
    if os.path.exists(out):   # extend an existing table (e.g. a finer M grid) rather than redo it
        for e in json.load(open(out))["configurations"]:
            done[(e["range"]["M"], e["range"]["N"], e["range"]["K"])] = e
    if os.path.exists(partial):
        for line in open(partial):
            e = json.loads(line)
            done[(e["range"]["M"], e["range"]["N"], e["range"]["K"])] = e
    pf = open(partial, "a")

    for k in args.k:
        for n in args.n:
            rot = 2 if k * n >= 64 * 1024 * 1024 else 4
            bs = [(torch.randn((k, n), device = "cuda") / k ** 0.5).half() for _ in range(rot)]
            for m in args.m:
                if (m, n, k) in done:
                    continue
                a = (torch.randn((m, k), device = "cuda") * 0.5).half()
                c32 = torch.empty((m, n), dtype = torch.float, device = "cuda")
                c16 = torch.empty((m, n), dtype = torch.half, device = "cuda")
                f32 = lambda i: ext.hgemm(a, bs[i % rot], c32)
                f16 = lambda i: ext.hgemm(a, bs[i % rot], c16)
                os.environ["EXL3_ROCM_WMMA_GEMM"] = "0"
                h32 = time_ms(f32)
                h16 = time_ms(f16)
                os.environ["EXL3_ROCM_WMMA_GEMM"] = "2"
                res = []
                for i, c in cfgs:
                    if k % (c[4] * 16):
                        continue
                    os.environ["EXL3_ROCM_WMMA_GEMM_CFG"] = str(i)
                    res.append((time_ms(f32), i, c))
                res.sort()
                w32, bi, bc = res[0]
                os.environ["EXL3_ROCM_WMMA_GEMM_CFG"] = str(bi)
                w16 = time_ms(f16)
                os.environ.pop("EXL3_ROCM_WMMA_GEMM_CFG", None)
                os.environ.pop("EXL3_ROCM_WMMA_GEMM", None)
                e = {
                    "range": {"M": m, "N": n, "K": k},
                    "layout": {"A": "row_major", "B": "row_major", "C": "row_major"},
                    "config": dict(zip(KEYS, bc)),
                    "route": {"fp32": w32 < h32 * (1 - args.margin), "fp16": w16 < h16 * (1 - args.margin)},
                    "ms": {"hipblas_fp32": round(h32, 4), "wmma_fp32": round(w32, 4),
                           "hipblas_fp16": round(h16, 4), "wmma_fp16": round(w16, 4)},
                }
                done[(m, n, k)] = e
                pf.write(json.dumps(e) + "\n")
                pf.flush()
                print(f"{m:>5} {n:>6} {k:>5}  fp32 {h32:8.3f} -> {w32:8.3f} ({h32 / w32:5.2f}x)  "
                      f"fp16 {h16:8.3f} -> {w16:8.3f} ({h16 / w16:5.2f}x)  cfg {bc}", flush = True)
                del a, c32, c16
            del bs
            torch.cuda.empty_cache()

    entries = [done[key] for key in sorted(done)]
    with open(out, "w") as f:
        json.dump({"arch": TABLE_ARCH[arch], "device": torch.cuda.get_device_properties(0).name,
                   "torch": torch.__version__, "margin": args.margin,
                   "note": "written by rocm_tools/tune_wmma_gemm.py; consumed by gen_wmma_gemm_table.py",
                   "configurations": entries}, f, indent = 1)
    pf.close()
    os.remove(partial)
    n32 = sum(e["route"]["fp32"] for e in entries)
    n16 = sum(e["route"]["fp16"] for e in entries)
    print(f"wrote {out}: {len(entries)} shapes, fp32 routed {n32}, fp16 routed {n16}")


if __name__ == "__main__":
    main()
