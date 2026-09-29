#!/usr/bin/env python3
"""Fractional (half-integer) EXL3 bitrates on RDNA: kernel paths against references (v1.5.3).

Upstream v1.5.3 added 1.5 / 2.5 / 3.5 bpw tensors (mul1 codebook, 16 * K + 8 uint16 per tile). On
ROCm they run through:
  - reconstruct_rdna.hip (regenerated verbatim from upstream: dq8_half in exl3_dq_rdna.hip.h)
  - the cooperative WMMA GEMM with half_k instances (comp_units_rdna/exl3_comp_unit_h*.hip),
    at every row count: the RDNA GEMV / multi-row / mgemv fast paths decline half_k
  - exl3_mgemm's cooperative kernel (half_k instances) for MultiLinear / MoE decode routes

Checks, on random packed trellises (random bits are valid for any codebook):
  1. reconstruct vs an independent decode: upstream frac.cu unpack_trellis_frac -> ext.decode,
     compared per 16x16 tile as sorted value sets (the two use different in-tile orders)
  2. LinearEXL3 kernel path (exl3_gemm) vs reconstruct + fp32 matmul, rows 1..32
  3. exl3_mgemm (one matrix through the pointer-table API) vs the same reference
Integer K 2 / 3 / 4 run as controls. Exits nonzero on a failure.

    python rocm_tools/frac_check.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.modules.quant.exl3_lib.quantize import codebook_mul1_mult

dev = "cuda:0"
torch.manual_seed(0)
fails = 0


def tile_u16(K):
    return int(16 * K)   # 16 * K for integer K, 16 * K + 8 = 16 * (K + 0.5) for half-integer


def make(K, k, n):
    trellis = torch.randint(-32768, 32767, (k // 16, n // 16, tile_u16(K)), dtype = torch.int16, device = dev)
    suh = (torch.randint(0, 2, (k,), device = dev) * 2 - 1).half()
    svh = (torch.randint(0, 2, (n,), device = dev) * 2 - 1).half()
    mul1 = torch.tensor(codebook_mul1_mult, dtype = torch.uint32).view(torch.int)
    return LinearEXL3(None, k, n, suh = suh, svh = svh, trellis = trellis, mul1 = mul1, key = f"frac{K}")


def report(name, ok, detail):
    global fails
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails += 1


k, n = 4096, 2048
for K in (1.5, 2.5, 3.5, 2, 3, 4):
    lin = make(K, k, n)
    assert lin.K == K
    print(f"K = {K}  (tile {tile_u16(K)} uint16, frac {lin.frac})")

    # 1. reconstruct vs unpack_trellis_frac + decode (half-integer only; integer K has no frac unpack)
    w = torch.empty((k, n), dtype = torch.half, device = dev)
    ext.reconstruct(w, lin.trellis, lin.K, False, True)
    if lin.frac is not None:
        ka, mask = lin.frac
        idx = torch.empty((k // 16, n // 16, 256), dtype = torch.int16, device = dev)
        ext.unpack_trellis_frac(idx, lin.trellis, ka, mask)
        vals = torch.empty((k // 16 * n // 16, 256), dtype = torch.half, device = dev)
        ext.decode(idx.view(-1, 256), vals, False, True)
        wt = w.view(k // 16, 16, n // 16, 16).permute(0, 2, 1, 3).reshape(-1, 256)
        same = torch.equal(torch.sort(wt.float(), dim = 1).values, torch.sort(vals.float(), dim = 1).values)
        report("reconstruct vs unpack_trellis_frac + decode (per-tile value sets)", same,
               "identical" if same else "differ")

    # Reference: y = ((x * suh) H) W (H) * svh, via the module's own reconstruct + hgemm path
    for rows in (1, 2, 4, 8, 16, 32):
        x = torch.randn((1, rows, k), dtype = torch.half, device = dev)
        y_ref = lin.forward(x, {"reconstruct": True}, out_dtype = torch.float)
        y = lin.forward(x, {}, out_dtype = torch.float)
        rel = ((y - y_ref).norm() / y_ref.norm()).item()
        report(f"exl3_gemm rows {rows:2d} vs reconstruct", rel < 5e-3, f"rel err {rel:.2e}")

    # 3. exl3_mgemm, one matrix, m = 1 (the MoE / MultiLinear decode entry)
    x = torch.randn((1, k), dtype = torch.half, device = dev)
    y_ref = lin.forward(x.view(1, 1, k), {"reconstruct": True}, out_dtype = torch.float).view(1, n)
    ptr = lambda t: torch.tensor([t.data_ptr()], dtype = torch.long, device = dev)
    C = torch.empty((1, 1, n), dtype = torch.float, device = dev)
    A_had = torch.empty((1, 1, k), dtype = torch.half, device = dev)
    ext.exl3_mgemm(x.view(1, 1, k), ptr(lin.trellis), C, ptr(lin.suh), A_had, ptr(lin.svh),
                   None, None, lin.K, -1, False, True, -1, -1, 0)
    rel = ((C.view(1, n) - y_ref).norm() / y_ref.norm()).item()
    report("exl3_mgemm m 1 vs reconstruct", rel < 5e-3, f"rel err {rel:.2e}")

print("FRAC CHECK " + ("PASS" if not fails else f"FAIL ({fails})"))
sys.stdout.flush()
os._exit(1 if fails else 0)
