"""End-to-end numerical check of the shipped EXL3 linear path, no model needed.

    python rocm_tools/exl3_stack_check.py

Compares ext.exl3_gemm (the ATen entry point the model actually calls, including
host dispatch, shape selection and the autotuner) against the reconstruct+hgemm
reference that LinearEXL3.reconstruct_hgemm uses. Both consume the same trellis,
so any disagreement is in the fused path.

This covers ground the C++ harnesses cannot. gemm_check.hip validates
exl3_gemm_kernel_inner with a plain launch; gemm_coop_check.hip validates the
cooperative wrapper against a hand-built decomposition. NEITHER goes through
exl3_gemm_gr -- the shape selector, the autotuner, the graph-arg recording, or
the ATen-level argument marshalling. A model producing garbage while both of
those pass points exactly here.

Note the reference is not independent of everything: it shares the codebook with
the fused path (both decode the same trellis). What it does isolate is the
matmul, the Hadamard application and the dispatch around them.
"""

import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3.ext import exllamav3_ext as ext


def check(size_m, size_k, size_n, K, mcg=False, mul1=False, c_fp32=False, seed=0):
    dev = torch.device("cuda:0")
    g = torch.Generator(device="cpu").manual_seed(seed)

    # Random trellis. reconstruct and the GEMM both just decode these bits, so
    # they do not need to come from a real quantization to agree.
    # int16 is signed; generate in int32 then view down so the full 16-bit
    # pattern space is covered (the codebook decodes raw bits, not a value range)
    trellis = torch.randint(-32768, 32767, (size_k // 16, size_n // 16, 16 * K),
                            dtype=torch.int32, generator=g).to(torch.int16).to(dev)

    A = (torch.randn((size_m, size_k), generator=g, dtype=torch.float32) * 0.25).half().to(dev)
    # suh/svh are sign-flip * scale vectors
    suh = (torch.randn((size_k,), generator=g, dtype=torch.float32) * 0.5).half().to(dev)
    svh = (torch.randn((size_n,), generator=g, dtype=torch.float32) * 0.5).half().to(dev)

    out_dtype = torch.float if c_fp32 else torch.half

    # --- fused path: exactly what LinearEXL3 -> BC_LinearEXL3::run does ---
    C_fused = torch.empty((size_m, size_n), dtype=out_dtype, device=dev)
    A_had = torch.empty_like(A)
    ext.exl3_gemm(A, trellis, C_fused, suh, A_had, svh, -1, mcg, mul1, 0)
    torch.cuda.synchronize()

    # --- reference: reconstruct + hadamard + hgemm (reconstruct_hgemm) ---
    xh = torch.empty_like(A)
    ext.had_r_128(A, xh, suh, None, 1.0)
    w = torch.empty((size_k, size_n), dtype=torch.half, device=dev)
    ext.reconstruct(w, trellis, K, mcg, mul1)
    y = torch.empty((size_m, size_n), dtype=torch.half, device=dev)
    ext.hgemm(xh, w, y)
    # svh is applied as the output hadamard by the fused kernel
    y_ref = torch.empty_like(y)
    ext.had_r_128(y, y_ref, None, svh, 1.0)
    torch.cuda.synchronize()

    a = C_fused.float()
    b = y_ref.float()
    denom = b.abs().mean().item() + 1e-6
    err = (a - b).abs().mean().item()
    rel = err / denom
    nan = bool(torch.isnan(a).any() or torch.isinf(a).any())
    status = "NaN/Inf" if nan else ("ok  " if rel < 0.05 else "MISMATCH")
    print(f"  m={size_m:<4} k={size_k:<5} n={size_n:<5} K={K} "
          f"{'mcg' if mcg else ('mul1' if mul1 else 'cb0 ')} "
          f"{'fp32' if c_fp32 else 'fp16'}  {status}  rel_err={rel:.4f}  "
          f"|fused|={a.abs().mean():.4f} |ref|={b.abs().mean():.4f}")
    return (not nan) and rel < 0.05


if __name__ == "__main__":
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    from exllamav3.rocm_py import describe
    print("ROCm patches:"); print(describe())
    print()
    ok = True
    # mcg (cb1) first: that is what the GLM model uses
    for m in (1, 4, 16, 64):
        ok &= check(m, 2048, 2048, 4, mcg=True)
    print()
    for K in (2, 3, 4, 5, 6, 8):
        ok &= check(16, 2048, 2048, K, mcg=True)
    print()
    ok &= check(16, 2048, 2048, 4, mcg=False, mul1=False)
    ok &= check(16, 2048, 2048, 4, mul1=True)
    ok &= check(16, 2048, 2048, 4, mcg=True, c_fp32=True)
    print()
    print("PASS" if ok else "FAIL -- fused exl3_gemm disagrees with reconstruct+hgemm")
    sys.exit(0 if ok else 1)
