#pragma once

#include <ATen/Tensor.h>

// RDNA 3.5 GEMV (single-token matmul) for EXL3 quantized weights.
// Ported from the working ROCm fork — complete with bits/cb/c_fp32 dispatch,
// SUH pre-Hadamard, SVH post-Hadamard, and dynamic wave selection.
// Restricted to size_m == 1; the cooperative exl3_gemm is used for larger M.

void exl3_gemv_rdna
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    bool mcg,
    bool mul1
);
