#!/usr/bin/env python3
"""GEMV vs the cooperative GEMM at m == 1, per shape, on real quantized weights.

Decode should route to the GEMV; on v1.4.1 almost none of it does (see
RDNA_NOTES, "Decode lost the GEMV path"). Before changing the routing it is worth
knowing what the routing *should* be, because the answer is not uniformly "GEMV":
a first pass in-situ measured the fp32-output GEMV as slower than the tile GEMM
it would replace, while the fp16 form was 2.2x faster. That comparison was across
different layers, so it could not separate "fp32 output is slow" from "that layer
has an unfavourable shape".

This drives ext.exl3_gemm directly, at m == 1, on real LinearEXL3 tensors, with
EXL3_GEMV toggled -- same weights, same shape, same call, only the path differs.
Run it twice:

    EXL3_GEMV=1 rocm_tools/bench_gemv_vs_gemm.py -m /path/to/model   # GEMV
    EXL3_GEMV=0 rocm_tools/bench_gemv_vs_gemm.py -m /path/to/model   # GEMM

No graph is captured here, so the EXL3_RDNA_GEMV_GRAPH guard does not interfere;
that is the point -- this measures what the guard is costing.

GEMV parallelism comes only from size_n (one warp per 16-wide output tile, no
split-K), so the block count is reported alongside: with 20 WGPs on gfx1151, a
shape yielding fewer than ~20 blocks cannot fill the part no matter how good the
kernel is, and should route to the GEMM regardless.

Medians of repeats, first discarded, per the ~4.7% noise floor.
"""

import argparse
import collections
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3 import Config, Model, Cache
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3


def gemv_blocks(size_n, size_k):
    nt, kb = size_n // 16, size_k // 16
    w = 16 if nt <= 32 else (8 if (nt <= 96 and kb <= 256) else 4)
    return (nt + w - 1) // w, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-r", "--repeats", type=int, default=6)
    args = ap.parse_args()

    mode = os.environ.get("EXL3_GEMV", "(unset -> GEMV enabled)")
    print(f"  EXL3_GEMV={mode}")

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)

    lins = []

    def walk(mod, d=0):
        if d > 10:
            return
        i = getattr(mod, "inner", None)
        if isinstance(i, LinearEXL3):
            lins.append(i)
        for ch in (getattr(mod, "modules", None) or []):
            if isinstance(ch, LinearEXL3):
                lins.append(ch)
            walk(ch, d + 1)

    walk(model)

    # One representative per distinct shape
    by_shape = {}
    for l in lins:
        by_shape.setdefault((l.in_features, l.out_features), l)

    dev = lins[0].trellis.device
    print(f"\n  {'in':>7} {'out':>7} {'K':>3} {'blocks':>7} {'fp16 us':>9} {'fp32 us':>9} {'fp32/fp16':>10}")

    for (K_in, N_out), l in sorted(by_shape.items(), key=lambda x: -x[0][1]):
        blocks, warps = gemv_blocks(N_out, K_in)
        A = torch.randn(1, 1, K_in, dtype=torch.half, device=dev)
        A_had = torch.empty_like(A)
        suh = getattr(l, "suh", None)
        svh = getattr(l, "svh", None)
        row = []
        for c_fp32 in (False, True):
            C = torch.empty(1, 1, N_out,
                            dtype=torch.float if c_fp32 else torch.half, device=dev)
            samples = []
            for r in range(args.repeats):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                ext.exl3_gemm(A, l.trellis, C, suh, A_had, svh, -1,
                              bool(getattr(l, "mcg", 0)), bool(getattr(l, "mul1", 0)), 0)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                if r:
                    samples.append(dt * 1e6)
            row.append(statistics.median(samples))
        ratio = row[1] / row[0] if row[0] else 0
        print(f"  {K_in:7} {N_out:7} {l.K:3} {blocks:7} {row[0]:9.1f} {row[1]:9.1f} {ratio:9.2f}x")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
