// Shim: `#include <mma.h>` (CUDA's nvcuda::wmma C++ API header).
//
// v1.5.3's det_gemm.cuh includes it but never uses nvcuda::wmma -- its int8 MMA,
// cp.async and ldmatrix are inline PTX behind `__CUDA_ARCH__ >= 800` / `>= 750`
// guards, which hip_compat.hip.h's device-pass `__CUDA_ARCH__ 1` compiles out.
// What remains usable on ROCm from det_gemm.cuh is the portable part: the
// FMA-only transcendentals (exp_det, log_det, softplus_det, sigmoid_det) and the
// int8 hi/lo weight quantizer (det_quant_split / det_quant16), which is what the
// RDNA siblings take from it.
//
// Nothing that would EXECUTE the compiled-out PTX paths is reachable on ROCm:
// routing_gemm.cu and hc_mix_tiled.cu (the int8 tensor-core kernels) are in
// ROCM_EXCLUDE, replaced by rocm/routing_gemm_rdna.hip and
// rocm/hc_mix_tiled_rdna.hip, which decline / raise. An empty header is the
// honest shim: a future use of nvcuda::wmma fails to compile instead of
// silently building.
#pragma once
