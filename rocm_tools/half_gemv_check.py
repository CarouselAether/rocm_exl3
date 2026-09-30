#!/usr/bin/env python3
"""Half-integer EXL3 bitrates (1.5 / 2.5 / 3.5 bpw) on the RDNA GEMV fast paths: correctness.

EXL3_ROCM_HALF_GEMV (default on) routes half-rate tensors at m = 1..8 to the multi-row GEMV
(exl3_gemv_multirow_rdna.hip) with the tiles core's dq8_half decoder, and rocm_py sends half-rate
MoE layers through torch.ops.exl3_rocm.moe_decode. This checks, on random packed trellises
(random bits are valid for any codebook), in one process (the switches are read per call):

  1. exl3_gemm rows 1..8 vs reconstruct + fp32 matmul (the frac_check.py reference)
  2. the same calls, tiles core vs direct core (EXL3_ROCM_GEMV_TILES=0): BIT-IDENTICAL
     (same fdot2 chains; the direct core decodes with exl3_dq_rdna.hip.h's dq8_half verbatim,
     so this pins the fast decoder to upstream's bit arithmetic)
  3. row r of an m-row call == the m == 1 call on that row: BIT-IDENTICAL (MTP verify rows)
  4. exl3_mgemm, 8 experts, m = 1: unweighted, and weighted with a 1- and 3-token grouped
     reduce, vs the per-expert reference; and vs EXL3_ROCM_HALF_GEMV=0 (cooperative mgemm)
  5. moe_decode (half K) vs an fp32 reference of the routed block, bsz 1 and 3; and
     bsz-3 rows vs three bsz-1 calls: BIT-IDENTICAL
Integer K 2 runs through the same checks as a control. Exits nonzero on a failure.

    python rocm_tools/half_gemv_check.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.modules.quant.exl3_lib.quantize import codebook_mul1_mult

dev = "cuda:0"
torch.manual_seed(0)
fails = 0
TOL = 5e-3


def report(name, ok, detail):
    global fails
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {detail}", flush = True)
    if not ok:
        fails += 1


def make(K, k, n, key):
    trellis = torch.randint(-32768, 32767, (k // 16, n // 16, int(16 * K)), dtype = torch.int16, device = dev)
    suh = (torch.randint(0, 2, (k,), device = dev) * 2 - 1).half()
    svh = (torch.randint(0, 2, (n,), device = dev) * 2 - 1).half()
    mul1 = torch.tensor(codebook_mul1_mult, dtype = torch.uint32).view(torch.int)
    return LinearEXL3(None, k, n, suh = suh, svh = svh, trellis = trellis, mul1 = mul1, key = key)


class env:
    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kw}
        for k, v in self.kw.items():
            os.environ[k] = str(v)

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def bits_eq(a, b):
    return torch.equal(a.view(torch.int32) if a.dtype == torch.float else a.view(torch.int16),
                       b.view(torch.int32) if b.dtype == torch.float else b.view(torch.int16))


def ptrs(ts):
    return torch.tensor([t.data_ptr() for t in ts], dtype = torch.long, device = dev)


def mgemm(A, lins, C, A_had, idx, w, num_tokens):
    ext.exl3_mgemm(A, ptrs([l.trellis for l in lins]), C, ptrs([l.suh for l in lins]), A_had,
                   ptrs([l.svh for l in lins]), idx, w, lins[0].K, -1, False, True, -1, -1, 0, num_tokens,
                   None, None)


def ref_lin(lin, x):
    return lin.forward(x.view(1, -1, x.shape[-1]), {"reconstruct": True}, out_dtype = torch.float).view(-1, lin.out_features)


def check_gemm(K, k, n):
    lin = make(K, k, n, f"g{K}")
    for rows in (1, 2, 3, 4, 8):
        x = torch.randn((1, rows, k), dtype = torch.half, device = dev)
        y_ref = lin.forward(x, {"reconstruct": True}, out_dtype = torch.float)
        y = lin.forward(x, {}, out_dtype = torch.float)
        with env(EXL3_ROCM_GEMV_TILES = 0):
            y_direct = lin.forward(x, {}, out_dtype = torch.float)
        with env(EXL3_ROCM_HALF_GEMV = 0):
            y_off = lin.forward(x, {}, out_dtype = torch.float)
        r = rel(y, y_ref)
        report(f"exl3_gemm {k}->{n} rows {rows}: vs reconstruct", r < TOL,
               f"rel {r:.2e} (switch off / cooperative: {rel(y_off, y_ref):.2e})")
        report(f"exl3_gemm {k}->{n} rows {rows}: tiles == direct core", bits_eq(y, y_direct),
               "bit-identical" if bits_eq(y, y_direct) else f"max diff {(y - y_direct).abs().max().item():.3e}")
        if rows > 1:
            y1 = torch.cat([lin.forward(x[:, r:r + 1], {}, out_dtype = torch.float) for r in range(rows)], dim = 1)
            report(f"exl3_gemm {k}->{n} rows {rows}: row r == m 1 call", bits_eq(y, y1),
                   "bit-identical" if bits_eq(y, y1) else f"max diff {(y - y1).abs().max().item():.3e}")
        # fp16 output
        yh = lin.forward(x, {}, out_dtype = torch.half)
        r = rel(yh, y_ref)
        report(f"exl3_gemm {k}->{n} rows {rows}: fp16 C vs reconstruct", r < TOL, f"rel {r:.2e}")


def check_mgemm(K, k, n, E = 16, top_k = 8):
    lins = [make(K, k, n, f"m{K}_{e}") for e in range(E)]
    for bsz in (1, 3):
        S = bsz * top_k
        sel = torch.stack([torch.randperm(E, device = dev)[:top_k] for _ in range(bsz)]).view(1, S)
        w = torch.rand((1, S), device = dev).half()
        # unweighted, shared input per token (the gate/up form, input broadcast at bsz 1)
        x = torch.randn((bsz, k), dtype = torch.half, device = dev)
        A = x.view(1, 1, k) if bsz == 1 else x.repeat_interleave(top_k, 0).view(S, 1, k).contiguous()
        C = torch.full((S, 1, n), float("nan"), dtype = torch.float, device = dev)
        A_had = torch.empty((S, 1, k), dtype = torch.half, device = dev)
        mgemm(A, lins, C, A_had, sel, None, bsz)
        with env(EXL3_ROCM_HALF_GEMV = 0):
            C_off = torch.full_like(C, float("nan"))
            mgemm(A, lins, C_off, A_had, sel, None, bsz)
        ref = torch.stack([ref_lin(lins[sel[0, j].item()], x[j // top_k]) for j in range(S)]).view(S, 1, n)
        r = rel(C, ref)
        report(f"exl3_mgemm {k}->{n} bsz {bsz} unweighted: vs reconstruct", r < TOL,
               f"rel {r:.2e} (switch off: {rel(C_off, ref):.2e}, on vs off {rel(C, C_off):.2e})")
        # weighted, per-slot inputs, grouped reduce per token (the down form)
        A = torch.randn((S, 1, k), dtype = torch.half, device = dev)
        C = torch.full((S, 1, n), float("nan"), dtype = torch.float, device = dev)
        mgemm(A, lins, C, A_had, sel, w, bsz)
        with env(EXL3_ROCM_HALF_GEMV = 0):
            C_off = torch.full_like(C, float("nan"))
            mgemm(A, lins, C_off, A_had, sel, w, bsz)
        ref = torch.zeros((bsz, n), dtype = torch.float, device = dev)
        for j in range(S):
            ref[j // top_k] += w[0, j].float() * ref_lin(lins[sel[0, j].item()], A[j]).view(n)
        got, got_off = C.view(S, n)[:bsz], C_off.view(S, n)[:bsz]
        r = rel(got, ref)
        report(f"exl3_mgemm {k}->{n} bsz {bsz} weighted reduce: vs reconstruct", r < TOL,
               f"rel {r:.2e} (switch off: {rel(got_off, ref):.2e}, on vs off {rel(got, got_off):.2e})")


def check_moe_decode(K, Hi = 4096, I = 2048, E = 16, top_k = 8):
    if not hasattr(torch.ops, "exl3_rocm") or not hasattr(torch.ops.exl3_rocm, "moe_decode"):
        report("moe_decode", False, "op missing")
        return
    g = [make(K, Hi, I, f"dg{K}_{e}") for e in range(E)]
    u = [make(K, Hi, I, f"du{K}_{e}") for e in range(E)]
    d = [make(K, I, Hi, f"dd{K}_{e}") for e in range(E)]
    Kc = int(K) if float(K).is_integer() else 16 + int(K)
    gu_t, gu_s, gu_v = ptrs([l.trellis for l in g + u]), ptrs([l.suh for l in g + u]), ptrs([l.svh for l in g + u])
    d_t, d_s, d_v = ptrs([l.trellis for l in d]), ptrs([l.suh for l in d]), ptrs([l.svh for l in d])
    rows = 8 * top_k

    def run(y, sel, w):
        yh = torch.empty((2 * rows, Hi), dtype = torch.half, device = dev)
        gu = torch.empty((2 * rows, I), dtype = torch.half, device = dev)
        a_had = torch.empty((rows, I), dtype = torch.half, device = dev)
        out = torch.full((rows, Hi), float("nan"), dtype = torch.float, device = dev)
        torch.ops.exl3_rocm.moe_decode(y, sel, w, gu_t, gu_s, gu_v, d_t, d_s, d_v, yh, gu, a_had, out,
                                       Kc, 2, Kc, 2, 0.0)
        return out[:y.shape[0]].clone()

    outs = {}
    for bsz in (1, 3):
        y = torch.randn((bsz, Hi), dtype = torch.half, device = dev) * 0.5
        sel = torch.stack([torch.randperm(E, device = dev)[:top_k] for _ in range(bsz)]).contiguous()
        w = torch.softmax(torch.randn((bsz, top_k), device = dev), dim = -1).half()
        out = run(y, sel, w)
        ref = torch.zeros((bsz, Hi), dtype = torch.float, device = dev)
        for t in range(bsz):
            for j in range(top_k):
                e = sel[t, j].item()
                gg = ref_lin(g[e], y[t]).half()
                uu = ref_lin(u[e], y[t]).half()
                a = (F.silu(gg.float()) * uu.float()).half()
                ref[t] += w[t, j].float() * ref_lin(d[e], a).view(Hi)
        r = rel(out, ref)
        report(f"moe_decode K {K} bsz {bsz}: vs fp32 reference", r < 1e-2, f"rel {r:.2e}")
        outs[bsz] = (y, sel, w, out)
    y, sel, w, out3 = outs[3]
    out1 = torch.cat([run(y[t:t + 1].contiguous(), sel[t:t + 1].contiguous(), w[t:t + 1].contiguous()) for t in range(3)])
    report(f"moe_decode K {K}: bsz 3 rows == bsz 1 calls", bits_eq(out3, out1),
           "bit-identical" if bits_eq(out3, out1) else f"max diff {(out3 - out1).abs().max().item():.3e}")


for K in (2.5, 1.5, 3.5, 2):
    print(f"K = {K}", flush = True)
    check_gemm(K, 4096, 2048)
    check_gemm(K, 2048, 4096)
    check_mgemm(K, 4096, 2048)
    check_mgemm(K, 2048, 4096)
    check_moe_decode(K)

print("HALF GEMV CHECK " + ("PASS" if not fails else f"FAIL ({fails})"))
sys.stdout.flush()
os._exit(1 if fails else 0)
