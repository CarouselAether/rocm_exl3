#!/usr/bin/env python3
"""GatedResidual decode mix (ext.gr_mix + ext.hc_apply) micro-benchmark and bit-identity check.

Qwen3.8-Flash-Next shapes by default: H = 4 streams, D = 2560, low rank 320, M = 324 fn rows
(site form, with the inject rows) -- 97 sites per decoded token. Weights are rotated over
--sets independent copies (default 12 x 13 MB > the 32 MB MALL) so every call streams its
weights from DRAM, as in the model.

    gr_mix_bench.py [--R 1] [--iters 200] [--save out.pt | --check ref.pt]

Kernel switches are environment variables read once per process (EXL3_ROCM_GR_DOTS,
EXL3_ROCM_GR_DPP), so an on/off A/B is two runs: --save in one, --check in the other.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

# rocprofv3-safe: load libtriton against its own LLVM first (see bench/profile_region.py)
try:
    _fl = sys.getdlopenflags()
    sys.setdlopenflags(_fl | os.RTLD_DEEPBIND)
    import triton._C.libtriton  # noqa: E402,F401
    sys.setdlopenflags(_fl)
except ImportError:
    pass

import torch  # noqa: E402
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, nargs="*", default=[1, 3])
    ap.add_argument("--D", type=int, default=2560)
    ap.add_argument("--LR", type=int, default=320)
    ap.add_argument("--sets", type=int, default=12)
    ap.add_argument("--iters", type=int, default=240)
    ap.add_argument("--final", action="store_true", help="final-mixer form (no post / inject rows)")
    ap.add_argument("--save", default=None)
    ap.add_argument("--check", default=None)
    ap.add_argument("--prefill", type=int, nargs="*", default=None,
                    help="instead: check / time torch.ops.exl3_rocm.gr_gate_mean against the torch "
                         "expression it replaces, at these row counts (e.g. 2048 256 33)")
    a = ap.parse_args()
    if a.prefill:
        return prefill(a)

    dev = torch.device("cuda:0")
    H, D, LR = 4, a.D, a.LR
    M = LR + (0 if a.final else H)
    g = torch.Generator(device = "cpu").manual_seed(1)
    sets = []
    for _ in range(a.sets):
        fn = (torch.randn((M, H * D), generator = g) * 0.02).half().to(dev)
        upt = (torch.randn((H, D // 4, LR, 4), generator = g) * 0.05).half().to(dev)
        w = (1.0 + torch.randn((H * D,), generator = g) * 0.1).half().to(dev)
        sets.append((fn, upt, w))

    results = {}
    for R in a.R:
        streams = (torch.randn((R, H, D), generator = g) * 3.0).float().to(dev)
        dots = torch.empty((R, M + 1, H), dtype = torch.float, device = dev)
        post = None if a.final else torch.empty((R, H), dtype = torch.float, device = dev)
        mixed = torch.empty((R, D), dtype = torch.half, device = dev)
        outs = []
        for fn, upt, w in sets:
            ext.gr_mix(streams, fn, upt, w, 1e-6, dots, post, mixed)
            outs.append((dots.clone(), None if post is None else post.clone(), mixed.clone()))
        results[R] = outs
        torch.cuda.synchronize()
        # timing: rotate over the weight sets
        ev0, ev1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        for _ in range(20):
            fn, upt, w = sets[_ % a.sets]
            ext.gr_mix(streams, fn, upt, w, 1e-6, dots, post, mixed)
        ev0.record()
        for i in range(a.iters):
            fn, upt, w = sets[i % a.sets]
            ext.gr_mix(streams, fn, upt, w, 1e-6, dots, post, mixed)
        ev1.record()
        torch.cuda.synchronize()
        us = ev0.elapsed_time(ev1) * 1000 / a.iters
        mb = (M * H * D * 2 + H * D * LR * 2) / 1e6
        print(f"R={R}  gr_mix {us:7.2f} us/call  ({mb:.1f} MB weights, {mb * 1e6 / us / 1e3:.0f} GB/s)")

    if a.save:
        torch.save({k: [[t.cpu() if t is not None else None for t in o] for o in v] for k, v in results.items()}, a.save)
        print(f"saved {a.save}")
    if a.check:
        ref = torch.load(a.check)
        ok = True
        for R, outs in results.items():
            for s, (o, r) in enumerate(zip(outs, ref[R])):
                for name, x, y in zip(("dots", "post", "mixed"), o, r):
                    if x is None and y is None:
                        continue
                    if not torch.equal(x.cpu(), y):
                        ok = False
                        d = (x.cpu().float() - y.float()).abs().max().item()
                        print(f"MISMATCH R={R} set={s} {name} max|diff|={d:.3g}")
        print("CHECK PASS (bit-identical)" if ok else "CHECK FAIL")
        sys.exit(0 if ok else 1)


def prefill(a):
    dev = torch.device("cuda:0")
    H, D = 4, a.D
    g0 = torch.Generator(device = "cpu").manual_seed(2)
    ok = True
    for R in a.prefill:
        g = (torch.randn((R, H * D), generator = g0) * 4.0).half().to(dev)
        g[0, :8] = torch.tensor([0.0, -0.0, 65504, -65504, 17.0, -17.0, 88.0, -88.0]).half()
        normed = (torch.randn((R * H, D), generator = g0) * 2.0).half().to(dev)
        ref = (torch.sigmoid(g.float()).view(R, H, D) * normed.float().view(R, H, D)).mean(dim = -2).half()
        out = torch.empty((R, D), dtype = torch.half, device = dev)
        torch.ops.exl3_rocm.gr_gate_mean(g.view(R, H, D), normed.view(R, H, D), out)
        same = torch.equal(out, ref)
        ok &= same
        n = 20
        ev = [torch.cuda.Event(enable_timing = True) for _ in range(4)]
        ev[0].record()
        for _ in range(n):
            ref = (torch.sigmoid(g.float()).view(R, H, D) * normed.float().view(R, H, D)).mean(dim = -2).half()
        ev[1].record()
        for _ in range(n):
            torch.ops.exl3_rocm.gr_gate_mean(g.view(R, H, D), normed.view(R, H, D), out)
        ev[2].record()
        torch.cuda.synchronize()
        tt, tk = ev[0].elapsed_time(ev[1]) / n * 1000, ev[1].elapsed_time(ev[2]) / n * 1000
        d = (out.float() - ref.float()).abs().max().item()
        print(f"R={R:5d}  torch {tt:8.1f} us  fused {tk:8.1f} us  {'bit-identical' if same else f'MISMATCH max|diff| {d:.3g}'}")
    print("CHECK PASS" if ok else "CHECK FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
