#!/usr/bin/env python3
"""Is token generation on this part bandwidth-bound, compute-bound, or neither?

bench_membw.py measures ~206 GB/s of achievable d2d bandwidth on gfx1151, which
invites the conclusion that decode "should" run at weight_bytes / 206 GB/s. That
inference has a hole in it: decode at bsz=1 is a *skinny* GEMV, and a skinny GEMV
may not be able to saturate bandwidth at all, no matter how it is written. Against
an unreachable ceiling any kernel looks bad.

So this measures three things and only then draws a conclusion:

  1. peak fp16 GEMM (square, large)     -- the compute roofline, TFLOP/s
  2. fp16 GEMV/skinny GEMM at decode M  -- the *reachable* bandwidth ceiling for
                                           decode's actual shape, unquantized
  3. the real quantized LinearEXL3 at the same M -- what the port achieves

(2) is the control that makes (3) interpretable. If (2) is far below 206 GB/s,
then decode is limited by the shape rather than by the EXL3 kernels, and the
headroom implied by comparing against 206 GB/s does not exist. If (2) is near
206 GB/s but (3) is far below it, the gap is real and belongs to trellis dequant.

Weight bytes for (3) are the packed trellis bytes actually read, not the
reconstructed fp16 size, so the two are directly comparable as bytes moved.

    rocm_tools/bench_compute.py -m /path/to/model
    rocm_tools/bench_compute.py -m /path/to/model -b 1 4 16 64

Medians of repeats with the first discarded, per the ~4.7% noise floor.
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.modules.quant.exl3 import LinearEXL3

NOISE_FLOOR = 0.05


def timed(fn, repeats):
    samples = []
    for r in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if r == 0:
            continue
        samples.append(dt)
    med = statistics.median(samples)
    spread = (max(samples) - min(samples)) / med if med else 0.0
    return med, spread


def flag(spread):
    return "  <- exceeds noise floor" if spread > NOISE_FLOOR else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-b", "--batches", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("-r", "--repeats", type=int, default=6)
    args = ap.parse_args()

    dev = "cuda:0"

    # ---- 1. compute roofline -------------------------------------------------
    print("\n=== peak fp16 GEMM (compute roofline) ===\n")
    print(f"  {'shape':>16}  {'TFLOP/s':>9}  {'spread':>7}")
    for n in (2048, 4096, 8192):
        a = torch.randn(n, n, dtype=torch.half, device=dev)
        b = torch.randn(n, n, dtype=torch.half, device=dev)
        med, spread = timed(lambda: torch.matmul(a, b), args.repeats)
        tflops = 2 * n ** 3 / med / 1e12
        print(f"  {n}x{n}x{n:<6}  {tflops:9.1f}  {spread:6.1%}{flag(spread)}")
        del a, b
    torch.cuda.empty_cache()

    # ---- load a real model for the quantized path ----------------------------
    print(f"\n -- loading {os.path.basename(args.model_dir.rstrip('/'))}", flush=True)
    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)

    # LinearEXL3 hangs off Linear.inner, which is NOT part of the .modules tree --
    # walking .modules alone finds nothing.
    def walk(mod, depth=0):
        if depth > 10:
            return None
        inner = getattr(mod, "inner", None)
        if isinstance(inner, LinearEXL3):
            return inner
        for c in (getattr(mod, "modules", None) or []):
            if isinstance(c, LinearEXL3):
                return c
            r = walk(c, depth + 1)
            if r is not None:
                return r
        return None

    # Collect MANY distinct linears, not one. gfx1151 has a 32 MB Infinity Cache, and a
    # single layer's trellis (tens of MiB) goes cache-resident after the first iteration
    # of a repeat loop -- which measures cache bandwidth and flatters the kernel by ~3x.
    # Cycling over a working set several times the cache size keeps the reads in DRAM.
    def walk_all(mod, out, depth=0):
        if depth > 10:
            return
        inner = getattr(mod, "inner", None)
        if isinstance(inner, LinearEXL3):
            out.append(inner)
        for c in (getattr(mod, "modules", None) or []):
            if isinstance(c, LinearEXL3):
                out.append(c)
            walk_all(c, out, depth + 1)

    all_lins = []
    walk_all(model, all_lins)

    lin = walk(model)
    if lin is None:
        print(" !! no LinearEXL3 found; cannot measure the quantized path")
        sys.stdout.flush()
        os._exit(1)

    K_in, N_out = lin.in_features, lin.out_features
    packed_bytes = lin.trellis.numel() * lin.trellis.element_size()
    fp16_bytes = K_in * N_out * 2
    print(f" -- linear: in={K_in} out={N_out}  packed={packed_bytes/2**20:.1f} MiB"
          f"  (fp16 equivalent {fp16_bytes/2**20:.1f} MiB)")

    # ---- 2 + 3: reachable ceiling vs achieved --------------------------------
    w = torch.randn(K_in, N_out, dtype=torch.half, device=dev)

    print(f"\n=== decode-shape GEMV: reachable ceiling vs EXL3 ===\n")
    print(f"  {'bsz':>4}  {'fp16 GB/s':>10}  {'exl3 GB/s':>10}  {'exl3/fp16':>10}  {'spread':>7}")
    for bsz in args.batches:
        x = torch.randn(bsz, K_in, dtype=torch.half, device=dev)
        med_f, sp_f = timed(lambda: torch.matmul(x, w), args.repeats)
        bw_f = fp16_bytes / med_f / 1e9

        xin = x.unsqueeze(0) if x.dim() == 2 else x
        try:
            med_e, sp_e = timed(lambda: lin.forward(xin, {}), args.repeats)
        except Exception:
            med_e, sp_e = timed(lambda: lin.forward(x, {}), args.repeats)
        bw_e = packed_bytes / med_e / 1e9

        sp = max(sp_f, sp_e)
        print(f"  {bsz:4}  {bw_f:10.1f}  {bw_e:10.1f}  {bw_e/bw_f:9.2f}x  {sp:6.1%}{flag(sp)}")
        del x

    print("\n  fp16 GB/s = weight bytes / time for an unquantized matmul of the same shape")
    print("  exl3 GB/s = packed trellis bytes / time for the real quantized layer")
    print("  NOTE: single-layer figures above are cache-flattered -- see the sweep below")

    # ---- cache-defeating: cycle a working set several times the 32 MB Infinity Cache ----
    same = [l for l in all_lins
            if l.in_features == K_in and l.out_features == N_out][:64]
    if len(same) >= 4:
        print(f"\n=== same shape, {len(same)} distinct layers cycled (DRAM-resident) ===\n")
        ws = sum(l.trellis.numel() * l.trellis.element_size() for l in same)
        print(f"  working set {ws / 2**20:.0f} MiB across {len(same)} layers"
              f"  (Infinity Cache is 32 MiB)\n")
        print(f"  {'bsz':>4}  {'exl3 GB/s':>10}  {'vs 1-layer':>11}  {'spread':>7}")
        for bsz in args.batches:
            x = torch.randn(bsz, K_in, dtype=torch.half, device=dev)
            xin = x.unsqueeze(0)

            def run_all():
                for l in same:
                    l.forward(xin, {})

            med, sp = timed(run_all, max(3, args.repeats // 2))
            bw = ws / med / 1e9
            print(f"  {bsz:4}  {bw:10.1f}  {'':>11}  {sp:6.1%}{flag(sp)}")
            del x
    else:
        print("\n  (not enough same-shape layers to build a cache-defeating working set)")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
