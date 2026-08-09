#!/usr/bin/env python3
"""Drive exl3_mgemm at m == 1 directly, outside any graph.

NOTE (2026-08-08): exl3_mgemm at m == 1 now routes to the plain-launch
multi-matrix GEMV (rocm/quant/exl3_mgemv_rdna.hip) unless a shape or grid is
forced. So `-s -1` measures the mgemv path, while `-s 4` (or any forced shape)
pins the cooperative kernel this file was originally written to study -- the
two together are the A/B. EXL3_MGEMV=0 disables the routing entirely.

exl3_mgemm is ~42-48% of MoE decode GPU time and sits at roughly 30% of the
achievable memory roofline, but it is hard to study in situ: at bsz <= MAX_BSZN
the MoE path runs it from inside a captured graph (C++ BC_BlockSparseMLP), and
rocprofv3's counter collection cannot serialise graph-launched kernels -- the
queue-sync watchdog fires and no counters come out.

This reaches the same kernel from Python instead, reusing a real module's
MultiLinear pointer tables (ptrs_trellis / ptrs_suh / ptrs_svh) so the weights,
bit width and codebook are exactly the shipped ones. No graph is involved, so
rocprofv3 can profile it:

    rocm_tools/bench_mgemm.py -m /path/to/model
    rocprofv3 --pmc OccupancyPercent MemUnitBusy FETCH_SIZE \
        --kernel-include-regex exl3_mgemm -d out -o mg --output-format csv \
        -- python rocm_tools/bench_mgemm.py -m /path/to/model --profile

Note gfx1151 collects at most THREE counters per pass; a fourth fails with
"Request exceeds the capabilities of the hardware to collect". Profiling a
PyTorch process also requires that torch's bundled librocprofiler-sdk.so and
librocprofiler-register.so not shadow the system ones -- see RDNA_NOTES.

--experts sweeps how many matrices one call covers, which is the axis that
distinguishes "each expert is too small to fill the GPU" from "the kernel is
inefficient regardless".
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3 import Config, Model, Cache
from exllamav3.ext import exllamav3_ext as ext


def find_moe(model):
    found = []

    def walk(mod, d=0):
        if d > 10:
            return
        if hasattr(mod, "multi_gate") and getattr(mod, "multi_gate", None) is not None:
            found.append(mod)
        for c in (getattr(mod, "modules", None) or []):
            walk(c, d + 1)

    walk(model)
    return found[0] if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-e", "--experts", type=int, nargs="+", default=None)
    ap.add_argument("-r", "--repeats", type=int, default=8)
    ap.add_argument("--profile", action="store_true",
                    help="fixed small loop for rocprofv3; skips timing output")
    ap.add_argument("--sms", type=int, nargs="+", default=[0],
                    help="force_num_sms values (0 = default). The default cooperative grid is "
                         "sized to multi_processor_count, which reports WGPs (20) not CUs (40) "
                         "on gfx1151, giving ~12%% occupancy.")
    ap.add_argument("-s", "--shapes", type=int, nargs="+", default=[-1],
                    help="force_shape_idx values to sweep (-1 = let the selector choose). "
                         "The selector picks a 512-wide N tile, which at m == 1 buys no reuse "
                         "but costs the accumulator registers that cap occupancy at ~12%%.")
    args = ap.parse_args()

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)

    moe = find_moe(model)
    if moe is None:
        print(" !! no MoE block with multi_gate found")
        sys.stdout.flush()
        os._exit(1)

    ml = moe.multi_gate
    H = ml.linears[0].in_features if getattr(ml, "linears", None) else None
    I = ml.linears[0].out_features if getattr(ml, "linears", None) else None
    n_exp = len(ml.linears) if getattr(ml, "linears", None) else 0
    topk = getattr(config, "num_experts_per_tok", 8) or 8
    print(f"  experts={n_exp}  in={H}  out={I}  K={ml.K}  topk={topk}")

    dev = ml.ptrs_trellis.device
    sweep = args.experts or [1, 2, 4, topk, min(16, n_exp), min(32, n_exp)]
    sweep = sorted({e for e in sweep if 0 < e <= n_exp})

    print(f"\n  {'experts':>8} {'shape':>6} {'sms':>5} {'us':>9} {'MiB':>8} {'GB/s':>9} {'spread':>7}")
    combos = [(e, s, n) for e in sweep for s in args.shapes for n in args.sms]
    for e, shape_idx, nsms in combos:
        A = torch.randn(1, 1, H, dtype=torch.half, device=dev)
        C = torch.empty(e, 1, I, dtype=torch.half, device=dev)
        A_had = torch.empty(e, 1, H, dtype=torch.half, device=dev)
        idx = torch.arange(e, dtype=torch.long, device=dev).view(1, e)

        def call():
            ext.exl3_mgemm(A, ml.ptrs_trellis, C, ml.ptrs_suh, A_had, ml.ptrs_svh,
                           idx, None, ml.K, shape_idx, ml.mcg, ml.mul1, -1, -1, nsms, 1, None, None)

        # Coverage check before timing anything: fill C with NaN and require the
        # kernel to overwrite every element. A tile shape that does not divide
        # size_n leaves the tail columns untouched, and the resulting "speedup"
        # is an artifact of skipped work -- a forced 384-wide tile at N = 1024
        # measured 19% faster that way and cost a day. The ext now refuses such
        # shapes outright; this catches any future gap from the output side.
        try:
            C.fill_(float("nan"))
            call()
            torch.cuda.synchronize()
        except RuntimeError as err:
            reason = str(err).splitlines()[0]
            print(f"  {e:8} {shape_idx:6} {nsms:5}  rejected: {reason}")
            continue
        nan_cols = torch.isnan(C).view(-1, I).any(dim=0).sum().item()
        if nan_cols:
            print(f"  {e:8} {shape_idx:6} {nsms:5}  !! UNCOVERED OUTPUT: {nan_cols}/{I} "
                  f"columns never written -- timings would be meaningless, skipping")
            continue

        if args.profile:
            for _ in range(20):
                call()
            torch.cuda.synchronize()
            continue

        samples = []
        for r in range(args.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            call()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if r:
                samples.append(dt)
        med = statistics.median(samples)
        spread = (max(samples) - min(samples)) / med if med else 0
        by = e * H * I * ml.K / 8          # packed trellis bytes actually read
        flag = "  <- noisy" if spread > 0.05 else ""
        # flush=True: an oversubscribed --sms value aborts the process from a GPU
        # assert on the *next* combo, and buffered rows would die with it
        print(f"  {e:8} {shape_idx:6} {nsms:5} {med*1e6:9.1f} {by/2**20:8.1f} {by/med/1e9:9.1f} {spread:6.1%}{flag}",
              flush=True)

    if args.profile:
        # Must NOT os._exit() here: rocprofv3 writes its CSV from exit hooks, and a
        # hard exit skips them, producing a clean-looking run with no output file.
        # The teardown segfault (see RDNA_NOTES) happens after the flush, so the
        # counters survive it.
        print("  (profile mode: loops done)")
        sys.stdout.flush()
        return
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
