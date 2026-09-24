#!/usr/bin/env python3
"""Validate the multi-row GEMV (m = 2..8, rocm/quant/exl3_gemv_multirow_rdna.hip)
against the m == 1 GEMV paths, row by row, bit for bit.

Row r of an m-row call runs the same per-row fdot2 chain, the same split-K
reduction order and the same rotation helpers as a separate m == 1 call on
that row, so it must be bit-identical to it -- for every output width the
m == 1 path itself splits K on (n / 16 <= 2048). Above that the m == 1 path
uses one warp per tile and the multi-row path still splits K, so those
shapes (lm_head scale) are checked against the cooperative GEMM with a
tolerance instead. Every case also reports its error against the cooperative
kernel (EXL3_GEMV_MAX_M=1 routes there; the switch is re-read per call), the
independent reference, so a bitwise match cannot hide a shared wrong result.

    python rocm_tools/multirow_check.py            # synthetic trellises, no model

Covers the single-matrix entry (ext.exl3_gemm) and the multi-matrix entry
(ext.exl3_mgemm: broadcast and per-slot inputs, indices, expert-range
packing, per-matrix width/output lists) at m in {2, 3, 5, 8}, K in {2, 4, 6},
both C dtypes. Exits nonzero on any failure.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3.ext import exllamav3_ext as ext

DEV = torch.device("cuda:0")
TOL = 2e-2


def trellis(k, n, K, g):
    t = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype=torch.int32, generator=g)
    return t.to(torch.int16).to(DEV)


def vec(n, g, scale=0.5):
    return (torch.randn((n,), generator=g, dtype=torch.float32) * scale).half().to(DEV)


def same_bits(a, b):
    it = torch.int32 if a.dtype == torch.float else torch.int16
    return a.shape == b.shape and torch.equal(a.view(it), b.view(it))


def rel_err(got, ref):
    m = ~torch.isnan(ref)
    d = (got.float() - ref.float()).abs()[m]
    return (d / ref.float().abs()[m].clamp_min(1.0)).max().item() if d.numel() else 0.0


def single(m, k, n, K, fp32, mcg, g):
    B = trellis(k, n, K, g)
    A = (torch.randn((m, k), generator=g, dtype=torch.float32) * 0.25).half().to(DEV)
    suh, svh = vec(k, g), vec(n, g)
    ctype = torch.float if fp32 else torch.half

    os.environ["EXL3_GEMV_MAX_M"] = "8"
    C = torch.full((m, n), float("nan"), dtype=ctype, device=DEV)
    ext.exl3_gemm(A, B, C, suh, torch.empty_like(A), svh, -1, mcg, False, 0)
    ref = torch.full((m, n), float("nan"), dtype=ctype, device=DEV)
    for r in range(m):
        ext.exl3_gemm(A[r:r + 1], B, ref[r:r + 1], suh, torch.empty((1, k), dtype=torch.half, device=DEV),
                      svh, -1, mcg, False, 0)
    os.environ["EXL3_GEMV_MAX_M"] = "1"
    coop = torch.full((m, n), float("nan"), dtype=ctype, device=DEV)
    ext.exl3_gemm(A, B, coop, suh, torch.empty_like(A), svh, -1, mcg, False, 0)
    torch.cuda.synchronize()

    name = f"single m={m} {k}->{n} K={K} {'fp32' if fp32 else 'fp16'}{' mcg' if mcg else ''}"
    e_coop, e_m1 = rel_err(C, coop), rel_err(ref, coop)
    if torch.isnan(C).any():
        print(f"  FAIL {name}: NaN in output"); return False
    if n // 16 <= 2048:
        # The m == 1 GEMV's own error against the cooperative kernel bounds what
        # fp16 rounding-order differences look like on this shape
        bitwise = same_bits(C, ref)
        ok = bitwise and e_coop <= max(TOL, e_m1 * 1.001)
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {'bit-identical to m=1' if bitwise else 'DIFFERS from m=1'}; "
              f"vs coop {e_coop:.2e} (m=1 path vs coop {e_m1:.2e})")
    else:
        ok = e_coop <= max(TOL, e_m1 * 1.001)
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: single-warp m=1 shape; vs coop {e_coop:.2e} (m=1 path vs coop {e_m1:.2e})")
    return ok


def multi(m, k, n, K, fp32, nmat, e, bszm_in_is_1, packing, lists, g):
    Bs = [trellis(k, n if not lists else (n if i % 2 == 0 else n // 2), K, g) for i in range(nmat)]
    suhs = [vec(k, g) for _ in range(nmat)]
    widths = [b.shape[1] * 16 for b in Bs]
    svhs = [vec(w, g) for w in widths]
    Bt = torch.tensor([b.data_ptr() for b in Bs], dtype=torch.long, device=DEV)
    suht = torch.tensor([s.data_ptr() for s in suhs], dtype=torch.long, device=DEV)
    svht = torch.tensor([s.data_ptr() for s in svhs], dtype=torch.long, device=DEV)
    ctype = torch.float if fp32 else torch.half

    idx = torch.randperm(nmat, generator=g)[:e].view(1, e).to(DEV)
    mi, ma = (nmat // 2, nmat) if packing else (-1, -1)
    bszm_in = 1 if bszm_in_is_1 else e
    A = (torch.randn((bszm_in, m, k), generator=g, dtype=torch.float32) * 0.25).half().to(DEV)

    def call(Ain, rows, max_m):
        os.environ["EXL3_GEMV_MAX_M"] = str(max_m)
        A_had = torch.empty((e, rows, k), dtype=torch.half, device=DEV)
        if lists:
            outs = [torch.full((rows, widths[i]), float("nan"), dtype=ctype, device=DEV) for i in range(nmat)]
            C = torch.full((e, rows, n), float("nan"), dtype=ctype, device=DEV)   # dtype/max width only
            size_n_list = torch.tensor(widths, dtype=torch.int, device=DEV)
            c_ptrs = torch.tensor([o.data_ptr() for o in outs], dtype=torch.long, device=DEV)
            ext.exl3_mgemm(Ain, Bt, C, suht, A_had, svht, idx, None, K, -1, 0, 0, mi, ma, 0, 1,
                           size_n_list, c_ptrs)
            torch.cuda.synchronize()
            return outs
        C = torch.full((e, rows, n), float("nan"), dtype=ctype, device=DEV)
        ext.exl3_mgemm(Ain, Bt, C, suht, A_had, svht, idx, None, K, -1, 0, 0, mi, ma, 0, 1, None, None)
        torch.cuda.synchronize()
        return C

    got = call(A, m, 8)
    coop = call(A, m, 1)
    name = (f"multi m={m} {k}->{n} K={K} {'fp32' if fp32 else 'fp16'} "
            f"{'bcast' if bszm_in_is_1 else 'per-slot'}{' pack-half' if packing else ''}{' lists' if lists else ''}")
    # Per-row m == 1 reference. The row slice must be made contiguous: the
    # binding reads slot j at A + j * rows * k, and a strided view of one row
    # keeps the m-row slot stride.
    bitwise = True
    if lists:
        refs = [torch.full_like(o, float("nan")) for o in got]
        for r in range(m):
            ref = call(A[:, r:r + 1, :].contiguous(), 1, 8)
            for i in range(nmat):
                refs[i][r:r + 1] = ref[i]
                if torch.isnan(ref[i]).all():
                    continue
                if not same_bits(got[i][r:r + 1], ref[i]):
                    bitwise = False
        e_coop = max(rel_err(g_, c_) for g_, c_ in zip(got, coop))
        e_m1 = max(rel_err(g_, c_) for g_, c_ in zip(refs, coop))
    else:
        refs = torch.full_like(got, float("nan"))
        for r in range(m):
            ref = call(A[:, r:r + 1, :].contiguous(), 1, 8)
            refs[:, r:r + 1] = ref
            if not same_bits(got[:, r:r + 1], ref):
                bitwise = False
        written = ~torch.isnan(coop).all(dim=2)
        if not torch.equal(written, ~torch.isnan(got).all(dim=2)):
            print(f"  FAIL {name}: written-row sets differ from coop"); return False
        e_coop = rel_err(got[written], coop[written])
        e_m1 = rel_err(refs[written], coop[written])
    # The cooperative kernel is the independent reference; the m == 1 GEMV's own
    # error against it bounds what fp16 rounding-order differences look like
    ok = bitwise and e_coop <= max(TOL, e_m1 * 1.001)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {'bit-identical to m=1 per row' if bitwise else 'DIFFERS from m=1'}; "
          f"vs coop {e_coop:.2e} (m=1 path vs coop {e_m1:.2e})")
    return ok


def main():
    g = torch.Generator(device="cpu").manual_seed(0)
    ok = True
    print("single-matrix (ext.exl3_gemm)")
    for (k, n) in ((3072, 1024), (3072, 12288), (12288, 3072), (1024, 3072)):
        for m in (2, 3, 5, 8):
            for K in (2, 4, 6):
                for fp32 in (False, True):
                    ok &= single(m, k, n, K, fp32, False, g)
    ok &= single(3, 3072, 3072, 4, False, True, g)
    ok &= single(3, 2048, 100352, 4, False, False, g)   # lm_head scale: single-warp m=1 shape
    print("multi-matrix (ext.exl3_mgemm)")
    for m in (2, 3, 8):
        for fp32 in (False, True):
            ok &= multi(m, 3072, 1024, 4, fp32, 16, 10, True, False, False, g)
            ok &= multi(m, 1024, 3072, 4, fp32, 16, 10, False, False, False, g)
            ok &= multi(m, 3072, 1024, 4, fp32, 16, 10, False, True, False, g)
            ok &= multi(m, 3072, 1024, 4, fp32, 4, 4, True, False, True, g)
    ok &= multi(5, 3072, 1024, 6, False, 16, 10, True, False, False, g)
    print("PASS" if ok else "FAIL")
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
