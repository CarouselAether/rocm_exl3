#!/usr/bin/env python3
"""Validate the plain-launch multi-matrix GEMV against the cooperative exl3_mgemm.

The mgemv path (rocm/quant/exl3_mgemv_rdna.hip) replaces the cooperative
exl3_mgemm at m == 1. Both are driven here on the SAME real-model weights and
inputs, toggled via EXL3_MGEMV (re-read by the ext on every call), and compared
elementwise. The cooperative kernel is the validated reference; agreement on
real trellis weights across every routing configuration is the acceptance bar.
Note this checks mgemv against mgemm, not against fp32 ground truth -- for
absolute MoE correctness questions use moe_ref32.py, per its header.

Covered per matrix (gate: fan-in shared input, no weights; down: per-slot
inputs, routing weights + reduce):

  - plain routing (indices, min_index = -1)
  - expert-range packing (min_index >= 0), including a range that filters
  - routing weights + the num_tokens grouped reduction
  - fp16 and fp32 C

    python rocm_tools/mgemv_check.py -m /path/to/model

Exits nonzero on any mismatch beyond tolerance. Tolerances are loose-ish
(fp16 accumulation orders differ legitimately between the two paths); a LAYOUT
bug shows up as gross error on most of a row, not as a tolerance miss.
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
        if hasattr(mod, "multi_gate") and getattr(mod, "multi_gate", None) is not None:
            found.append(mod)
        for c in (getattr(mod, "modules", None) or []):
            walk(c, d + 1)

    walk(model)
    return found[0] if found else None


def run_mgemm(ml, A, C, A_had, idx, weights, min_index, max_index, num_tokens):
    ext.exl3_mgemm(A, ml.ptrs_trellis, C, ml.ptrs_suh, A_had, ml.ptrs_svh,
                   idx, weights, ml.K, -1, ml.mcg, ml.mul1,
                   min_index, max_index, 0, num_tokens, None, None)


def check_masked_2tok(name, ml, H, I, idx2, w2, min_index, max_index, tol):
    """Masked range filtering at num_tokens=2 vs the unfiltered -1-skip path."""
    dev = ml.ptrs_trellis.device
    e2 = idx2.shape[1]
    torch.manual_seed(0)
    A = torch.randn(e2, 1, H, dtype=torch.half, device=dev) * 0.5

    # Reference indices: pre-rebased, out-of-range picks -> -1
    keep = (idx2 >= min_index) & (idx2 < max_index)
    idx_ref = torch.where(keep, idx2 - min_index, torch.full_like(idx2, -1))

    os.environ["EXL3_MGEMV"] = "0"
    outs = []
    for run_idx, mi, ma in ((idx_ref, -1, -1), (idx2, min_index, max_index)):
        C = torch.full((e2, 1, I), float("nan"), dtype=torch.float, device=dev)
        A_had = torch.empty(e2, 1, H, dtype=torch.half, device=dev)
        run_mgemm(ml, A, C, A_had, run_idx, w2, mi, ma, 2)
        torch.cuda.synchronize()
        # Only the two reduced token rows are defined output
        outs.append(C[:2].float())
    ref, got = outs
    if torch.isnan(got).any() or torch.isnan(ref).any():
        print(f"  FAIL {name}: NaN in reduced token rows")
        return False
    denom = ref.abs().clamp_min(1.0)
    err = ((got - ref).abs() / denom).max().item()
    good = err <= tol
    print(f"  {'ok  ' if good else 'FAIL'} {name}: tokens=2 max_rel_err={err:.3e} (tol {tol:.0e})")
    return good


def compare(name, ml, H, I, idx, weights, min_index, max_index, num_tokens,
            fp32, fan_in_shared, tol, e=None):
    dev = ml.ptrs_trellis.device
    e = idx.shape[1] if idx is not None else e
    torch.manual_seed(0)
    A = torch.randn(1 if fan_in_shared else e, 1, H, dtype=torch.half, device=dev) * 0.5
    ctype = torch.float if fp32 else torch.half

    outs = {}
    for mode in ("0", "1"):
        os.environ["EXL3_MGEMV"] = mode
        C = torch.full((e, 1, I), float("nan"), dtype=ctype, device=dev)
        A_had = torch.empty(e, 1, H, dtype=torch.half, device=dev)
        run_mgemm(ml, A, C, A_had, idx, weights, min_index, max_index, num_tokens)
        torch.cuda.synchronize()
        outs[mode] = C.float()

    # Rows the kernels legitimately never write (filtered experts under packing,
    # slots past the packed count) hold NaN in both paths; compare only rows
    # that the reference wrote.
    ref, got = outs["0"], outs["1"]
    ref_written = ~torch.isnan(ref).all(dim=2)          # (e, 1) row mask
    got_written = ~torch.isnan(got).all(dim=2)
    if not torch.equal(ref_written, got_written):
        print(f"  FAIL {name}: written-row sets differ "
              f"(mgemm {ref_written.sum().item()}, mgemv {got_written.sum().item()})")
        return False

    mask = ref_written.unsqueeze(2).expand_as(ref)
    r, g = ref[mask], got[mask]
    if torch.isnan(g).any():
        print(f"  FAIL {name}: NaN inside written rows on mgemv path")
        return False
    denom = r.abs().clamp_min(1.0)
    err = ((g - r).abs() / denom).max().item()
    ok = err <= tol
    rows = int(ref_written.sum().item())
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: rows={rows} max_rel_err={err:.3e} (tol {tol:.0e})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    args = ap.parse_args()

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    model.load(progressbar=False)

    moe = find_moe(model)
    if moe is None:
        print(" !! no MoE block with multi_gate found")
        sys.stdout.flush()
        os._exit(1)

    ok = True
    for label, attr, fan_in_shared, use_w in (
        ("gate", "multi_gate", True, False),
        ("down", "multi_down", False, True),
    ):
        ml = getattr(moe, attr, None)
        if ml is None:
            print(f"  -- no {attr}, skipping")
            continue
        H = ml.linears[0].in_features
        I = ml.linears[0].out_features
        n_exp = len(ml.linears)
        dev = ml.ptrs_trellis.device
        print(f"{label}: {H}->{I}, {n_exp} experts, K={ml.K}")

        e = min(10, n_exp)
        g = torch.Generator(device="cpu").manual_seed(1)
        idx = torch.randperm(n_exp, generator=g)[:e].view(1, e).to(dev)
        w = None
        if use_w:
            w = (torch.rand(1, e, generator=g) + 0.1).half().to(dev)
            w /= w.sum()

        # fp16 tolerance: both paths accumulate the dot in fp32 but round the
        # matmul result to fp16 before the output rotation re-reads it (plain
        # path) vs. keeping it in registers (cooperative path), so a few ulp of
        # divergence per element is legitimate.
        # No-indices form: B_list indexed directly by slot, the shape the dense
        # models' fused gate/up mgemm uses (bszm small, indices absent). Uses
        # the first few experts as stand-in matrices.
        ok &= compare(f"{label}/fp16C/no-idx", ml, H, I, None, None, -1, -1, 1,
                      False, fan_in_shared, 2e-2, e=2)

        for fp32, tol in ((False, 2e-2), (True, 2e-2)):
            sfx = "fp32C" if fp32 else "fp16C"
            ok &= compare(f"{label}/{sfx}/plain", ml, H, I, idx, w, -1, -1, 1,
                          fp32, fan_in_shared, tol)
            ok &= compare(f"{label}/{sfx}/pack-all", ml, H, I, idx, w, 0, n_exp, 1,
                          fp32, fan_in_shared, tol)
            half_range = n_exp // 2
            ok &= compare(f"{label}/{sfx}/pack-half", ml, H, I, idx, w, half_range, n_exp, 1,
                          fp32, fan_in_shared, tol)

        # Grouped reduction (down only: needs weights). Two tokens' worth of
        # slots, per-slot inputs, stride = e.
        if use_w:
            idx2 = torch.cat([idx, idx], dim=1)
            w2 = torch.cat([w, w], dim=1)
            ok &= compare(f"{label}/fp16C/2tok", ml, H, I, idx2, w2, -1, -1, 2,
                          False, False, 2e-2)

            # v1.4.4 masked grouped reduction: num_tokens > 1 WITH range filtering.
            # The mgemv fast path declines this combination, so an EXL3_MGEMV A/B is
            # vacuous (both runs are the cooperative kernel). Cross-check instead
            # against the validated unfiltered path: filtering [half, n_exp) with
            # rebase is identical to pre-shifting the indices by half and marking
            # out-of-range picks -1 (negative indices skip the slot; the v1.4.4
            # reduction guard must then skip their stale scratch too). Same experts,
            # same per-slot input rows, same weights, same summation order -> fp32
            # outputs should agree to rounding. Coop-only on both sides.
            ok &= check_masked_2tok(f"{label}/fp32C/2tok+pack-half", ml, H, I,
                                    idx2, w2, n_exp // 2, n_exp, 1e-5)

    print("PASS" if ok else "FAIL")
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
