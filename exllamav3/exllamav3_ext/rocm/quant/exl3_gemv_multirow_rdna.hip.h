#pragma once

// =============================================================================
// Multi-row GEMV for RDNA, m = 2..8 -- host interface
// =============================================================================
//
// See exl3_gemv_multirow_rdna.hip for the design. Both entry points return
// false, having launched nothing, when the call must fall through to the
// cooperative GEMM / mgemm.

#include <cuda_fp16.h>
#include <cstdint>

class Graph;

// EXL3_GEMV_MAX_M (default 8, clamp 1..8): the largest m the multi-row GEMV
// takes; 1 switches the path off. Re-read per call.
int exl3_gemv_max_m();

// Allocates the per-device parameter block outside capture (hipMalloc would
// invalidate an active capture). Called from the m == 1 paths' non-graph
// sites, which every BC module runs eagerly before it captures.
void exl3_gemv_multirow_prewarm(int device);

// Single matrix: the exl3_gemm contract (A m x K, B trellis, C m x N, suh,
// A_had scratch of m x K, svh). Records the six GP_gemm_* sites when graph.
bool exl3_gemv_multirow_try_launch
(
    const half* A_ptr,
    const uint16_t* B_ptr,
    void* C_ptr,
    const half* suh_ptr,
    half* A_had_ptr,
    const half* svh_ptr,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    int device,
    cudaStream_t stream,
    Graph* graph
);

// Multi matrix: the exl3_mgemm contract at m > 1 without routing weights
// (A bszm_in x m x K, C bszm_out x m x N, A_had bszm x m x K, per-matrix
// pointer tables, optional indices / expert-range packing / per-matrix width
// and output lists). Records the four GP_mgemm_* sites when graph.
bool exl3_mgemv_multirow_try_launch
(
    const half* A_ptr,
    const uintptr_t* B_ptr_ptr,
    void* C_ptr,
    const uintptr_t* suh_ptr_ptr,
    half* A_had_ptr,
    const uintptr_t* svh_ptr_ptr,
    const int64_t* indices_ptr,
    const half* weights_ptr,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    int bszm_in,
    int bszm_out,
    int min_index,
    int max_index,
    int num_tokens,
    const int* size_n_list,
    void** c_list,
    int device,
    cudaStream_t stream,
    Graph* graph
);
