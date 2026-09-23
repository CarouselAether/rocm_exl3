#!/usr/bin/env python3
"""Bitwise A/B of the mgemv path (rocm/quant/exl3_mgemv_rdna.hip) against itself
across a code change.

mgemv_check.py compares mgemv to the cooperative kernel with a tolerance, which
cannot tell a refactor that changed nothing from one that changed a few ulp.
This records the mgemv path's outputs on real model weights BEFORE a change and
compares them bit for bit AFTER it -- the acceptance bar for any change whose
arithmetic is claimed to be identical (the launch-count fusions of 2026-09).

    python rocm_tools/mgemv_bitwise.py -m /path/to/model --save    ref.pt   # old build
    python rocm_tools/mgemv_bitwise.py -m /path/to/model --compare ref.pt   # new build

Covers the same routing configurations as mgemv_check.py (plain, pack-all,
pack-half, no-idx, weights + 2-token reduce), both C dtypes, for gate (shared
input) and down (per-slot inputs + routing weights). Unwritten rows hold NaN in
both runs and compare equal by bit pattern. Exits nonzero on any difference.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext


def find_moe(model):
    found = []

    def walk(mod, d=0):
        if d > 10:
            return
        if getattr(mod, "multi_gate", None) is not None:
            found.append(mod)
        for c in (getattr(mod, "modules", None) or []):
            walk(c, d + 1)

    walk(model)
    return found[0] if found else None


def run(ml, A, C, A_had, idx, weights, min_index, max_index, num_tokens):
    ext.exl3_mgemm(A, ml.ptrs_trellis, C, ml.ptrs_suh, A_had, ml.ptrs_svh,
                   idx, weights, ml.K, -1, ml.mcg, ml.mul1,
                   min_index, max_index, 0, num_tokens, None, None)


def cases(moe):
    """Yield (name, fn) where fn() returns the output tensor of one config."""
    for label, attr, fan_in_shared, use_w in (
        ("gate", "multi_gate", True, False),
        ("down", "multi_down", False, True),
    ):
        ml = getattr(moe, attr, None)
        if ml is None:
            continue
        H = ml.linears[0].in_features
        I = ml.linears[0].out_features
        n_exp = len(ml.linears)
        dev = ml.ptrs_trellis.device
        e = min(10, n_exp)
        g = torch.Generator(device="cpu").manual_seed(1)
        idx = torch.randperm(n_exp, generator=g)[:e].view(1, e).to(dev)
        w = None
        if use_w:
            w = (torch.rand(1, e, generator=g) + 0.1).half().to(dev)
            w /= w.sum()

        def make(name, idx_, w_, mi, ma, nt, fp32, shared, e_):
            def fn():
                torch.manual_seed(0)
                A = torch.randn(1 if shared else e_, 1, H, dtype=torch.half, device=dev) * 0.5
                C = torch.full((e_, 1, I), float("nan"),
                               dtype=torch.float if fp32 else torch.half, device=dev)
                A_had = torch.empty(e_, 1, H, dtype=torch.half, device=dev)
                run(ml, A, C, A_had, idx_, w_, mi, ma, nt)
                torch.cuda.synchronize()
                return C
            return f"{label}/{name}", fn

        yield make("fp16C/no-idx", None, None, -1, -1, 1, False, fan_in_shared, 2)
        for fp32 in (False, True):
            sfx = "fp32C" if fp32 else "fp16C"
            yield make(f"{sfx}/plain", idx, w, -1, -1, 1, fp32, fan_in_shared, e)
            yield make(f"{sfx}/pack-all", idx, w, 0, n_exp, 1, fp32, fan_in_shared, e)
            yield make(f"{sfx}/pack-half", idx, w, n_exp // 2, n_exp, 1, fp32, fan_in_shared, e)
        if use_w:
            idx2 = torch.cat([idx, idx], dim=1)
            w2 = torch.cat([w, w], dim=1)
            yield make("fp16C/2tok", idx2, w2, -1, -1, 2, False, False, 2 * e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("--save", metavar="FILE")
    ap.add_argument("--compare", metavar="FILE")
    args = ap.parse_args()
    if bool(args.save) == bool(args.compare):
        ap.error("exactly one of --save / --compare")

    os.environ["EXL3_MGEMV"] = "1"
    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    model.load(progressbar=False)
    moe = find_moe(model)
    if moe is None:
        print(" !! no MoE block with multi_gate found")
        sys.stdout.flush()
        os._exit(1)

    ref = torch.load(args.compare) if args.compare else {}
    out = {}
    ok = True
    for name, fn in cases(moe):
        C = fn().cpu()
        out[name] = C
        if args.compare:
            R = ref.get(name)
            if R is None:
                print(f"  ??   {name}: not in reference")
                ok = False
                continue
            same = (R.shape == C.shape and R.dtype == C.dtype and
                    torch.equal(R.view(torch.int32 if C.dtype == torch.float else torch.int16),
                                C.view(torch.int32 if C.dtype == torch.float else torch.int16)))
            if same:
                print(f"  ok   {name}: bit-identical ({tuple(C.shape)})")
            else:
                ok = False
                m = ~(torch.isnan(R) & torch.isnan(C))
                d = (R.float() - C.float()).abs()
                d = torch.where(m, d, torch.zeros_like(d))
                n_diff = int((d > 0).sum().item())
                print(f"  FAIL {name}: {n_diff} elements differ, max abs {d.max().item():.3e}")
        else:
            print(f"  saved {name} ({tuple(C.shape)}, {C.dtype})")

    if args.save:
        torch.save(out, args.save)
        print(f"reference written to {args.save}")
    else:
        print("BITWISE " + ("PASS" if ok else "FAIL"))
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
