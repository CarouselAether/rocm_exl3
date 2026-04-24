// =============================================================================
// RDNA 3.5 Support for EXL3
// =============================================================================
//
// Target: gfx1150/gfx1151 (RDNA 3.5), Wave32
//
// This file provides:
// - Vector types (Vec, FragB, WmmaFrag{A,B,C}, half16_t, float8_t)
// - rdna_wmma::{load_matrix_a, load_matrix_b, mma_sync, store_matrix_c*,
//   load_accumulate_c*} — native WMMA 16x16x16 FP16→FP32 wrappers
// - FSHF_IMM / BFE16_IMM / bfe64 — bitfield helpers used by dequant
// - mem_fence() — wave-local s_waitcnt, paired with __syncthreads in the
//   cooperative GEMM inner loop
//
// WMMA argument order and C-fragment store layout are validated by
// tests/wmma_smoke2.cpp on ROCm 7.2.1 against the RDNA 3.5 ISA
// (wmma_smoke2 checks A*B vs B*A and transposed-store patterns).
//
// =============================================================================

#ifndef RDNA35_UNIFIED_HIP
#define RDNA35_UNIFIED_HIP

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// =============================================================================
// Vector Types
// =============================================================================

template <typename T, int N>
struct Vec
{
    T data[N];
    __device__ __forceinline__ T& operator[](int i) { return data[i]; }
    __device__ __forceinline__ const T& operator[](int i) const { return data[i]; }
};

// FragB holds 4 half values as 2 half2 — matches dequantization output
using FragB = Vec<half2, 2>;

// =============================================================================
// WMMA Fragment Types (native 16x16x16)
// =============================================================================

typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef float    float8_t  __attribute__((ext_vector_type(8)));

struct WmmaFragA {
    half16_t data;
    __device__ __forceinline__ void clear() {
        #pragma unroll
        for (int i = 0; i < 16; i++)
            ((_Float16*)&data)[i] = (_Float16)0.0f;
    }
};

struct WmmaFragB {
    half16_t data;
    __device__ __forceinline__ void clear() {
        #pragma unroll
        for (int i = 0; i < 16; i++)
            ((_Float16*)&data)[i] = (_Float16)0.0f;
    }
};

struct WmmaFragC {
    float8_t data;

    __device__ __forceinline__ void clear() {
        #pragma unroll
        for (int i = 0; i < 8; i++)
            ((float*)&data)[i] = 0.0f;
    }

    __device__ __forceinline__ float& operator[](int i) {
        return ((float*)&data)[i];
    }
    __device__ __forceinline__ const float& operator[](int i) const {
        return ((const float*)&data)[i];
    }
};

// =============================================================================
// rdna_wmma Namespace — WMMA Operations for GEMM Kernel
// =============================================================================
//
// Native RDNA 3.5 WMMA: 16x16x16 FP16 → FP32, computes C += A × B.
//
// Builtin signature (B precedes A):
//   __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(B, A, C) → C
//
// Wave32 lane mapping (validated by tests/wmma_smoke2.cpp):
//   A fragment: lane L holds row (L % 16), all 16 columns
//   B fragment: lane L holds column (L % 16), all 16 rows
//   C fragment: row = lane % 16; col_base = (lane >= 16) ? 1 : 0;
//               each lane stores 8 floats at columns [i*2 + col_base] for i in 0..7
//
// =============================================================================

namespace rdna_wmma {

// Load A fragment from row-major matrix.
// Lane L loads row (L % 16), WMMA_K=16 columns starting at the pointer.
// Caller bakes the sub-K column offset into A, so we always load 16 consecutive
// halves — one GLOBAL_LOAD_DWORDX8 instead of 16 scalar loads.
// Alignment: LDS tile starts are >= 32-byte aligned; row*stride offsets are
// multiples of WMMA_K (16 halves = 32 bytes), so 32-byte alignment holds.
__device__ __forceinline__ void load_matrix_a(
    WmmaFragA& frag,
    const half* A,
    int stride)  // stride = K (columns per row in the tile)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    // half == _Float16 on AMD; single vector load replaces 16 scalar loads
    frag.data = *((const half16_t*)(A + row * stride));
}

// Load B fragment from row-major matrix.
// B is [K, N], lane L loads column (L % 16) — strided access, no vector load possible.
__device__ __forceinline__ void load_matrix_b(
    WmmaFragB& frag,
    const half* B,
    int stride)  // stride = N (number of columns in B)
{
    int lane = threadIdx.x & 31;
    int col = lane % 16;
    // Direct cast: half == _Float16 on AMD, avoids f16→f32→f16 round-trip
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        ((_Float16*)&frag.data)[k] = ((_Float16*)B)[k * stride + col];
    }
}

// Matrix multiply-accumulate
__device__ __forceinline__ void mma_sync(
    WmmaFragC& c,
    const WmmaFragA& a,
    const WmmaFragB& b)
{
    c.data = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b.data, a.data, c.data);
}

// Load and accumulate C (FP32) — no bounds checking
__device__ __forceinline__ void load_accumulate_c(
    WmmaFragC& frag,
    const float* C,
    int stride)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    const float* row_ptr = C + row * stride;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int col = i * 2 + col_base;
        frag[i] += row_ptr[col];
    }
}

// Load and accumulate C (FP16 → FP32) — no bounds checking
__device__ __forceinline__ void load_accumulate_c_half(
    WmmaFragC& frag,
    const half* C,
    int stride)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    const half* row_ptr = C + row * stride;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int col = i * 2 + col_base;
        frag[i] += __half2float(row_ptr[col]);
    }
}

// Load and accumulate C (FP32) — with bounds checking
__device__ __forceinline__ void load_accumulate_c_checked(
    WmmaFragC& frag,
    const float* C,
    int stride,
    int valid_rows)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    if (row < valid_rows) {
        const float* row_ptr = C + row * stride;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int col = i * 2 + col_base;
            frag[i] += row_ptr[col];
        }
    }
}

// Load and accumulate C (FP16 → FP32) — with bounds checking
__device__ __forceinline__ void load_accumulate_c_half_checked(
    WmmaFragC& frag,
    const half* C,
    int stride,
    int valid_rows)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    if (row < valid_rows) {
        const half* row_ptr = C + row * stride;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int col = i * 2 + col_base;
            frag[i] += __half2float(row_ptr[col]);
        }
    }
}

// Store C fragment with bounds checking (FP32)
__device__ __forceinline__ void store_matrix_c_checked(
    float* C,
    const WmmaFragC& frag,
    int stride,
    int valid_rows,
    int valid_cols)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    if (row < valid_rows) {
        float* row_ptr = C + row * stride;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int col = i * 2 + col_base;
            if (col < valid_cols) {
                row_ptr[col] = frag[i];
            }
        }
    }
}

// Store C fragment with bounds checking (FP32 → FP16)
__device__ __forceinline__ void store_matrix_c_half_checked(
    half* C,
    const WmmaFragC& frag,
    int stride,
    int valid_rows,
    int valid_cols)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    if (row < valid_rows) {
        half* row_ptr = C + row * stride;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int col = i * 2 + col_base;
            if (col < valid_cols) {
                row_ptr[col] = __float2half(frag[i]);
            }
        }
    }
}

// Store C fragment (FP32, no bounds checking)
__device__ __forceinline__ void store_matrix_c(
    float* C,
    const WmmaFragC& frag,
    int stride)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    float* row_ptr = C + row * stride;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int col = i * 2 + col_base;
        row_ptr[col] = frag[i];
    }
}

// Store C fragment (FP32 → FP16, no bounds checking)
__device__ __forceinline__ void store_matrix_c_half(
    half* C,
    const WmmaFragC& frag,
    int stride)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    half* row_ptr = C + row * stride;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int col = i * 2 + col_base;
        row_ptr[col] = __float2half(frag[i]);
    }
}

} // namespace rdna_wmma

// =============================================================================
// Bitfield and Shift Macros (used by quant/exl3_dq_rdna.cuh)
// =============================================================================

// Funnel shift with immediate shift count
#define FSHF_IMM(dest, lo, hi, shift) \
    do { \
        uint64_t _merged = ((uint64_t)(hi) << 32) | (uint64_t)(lo); \
        dest = (uint32_t)(_merged >> ((shift) & 63)); \
    } while(0)

// Bit field extract: 16 bits starting at 'start'
#define BFE16_IMM(dest, src, start) \
    dest = ((src) >> (start)) & 0xffff

// 64-bit bit field extract
__device__ __forceinline__ uint32_t bfe64(uint32_t lo, uint32_t hi, int start, int len)
{
    uint64_t merged = ((uint64_t)hi << 32) | (uint64_t)lo;
    return (uint32_t)((merged >> start) & ((1ULL << len) - 1));
}

// =============================================================================
// Memory Fence
// =============================================================================
//
// Wave-local s_waitcnt(0) — forces outstanding VMEM/LDS operations for this
// wave to retire. Always pair with __syncthreads() when cross-thread
// visibility is required (the block barrier provides the actual
// synchronization; s_waitcnt just ensures the wave's loads/stores have
// committed before the barrier is reached). See exl3_gemm_inner_rdna.cuh for
// the (mem_fence, __syncthreads) pattern used in the cooperative GEMM loop.
//
// Cross-CU atomic synchronization uses __threadfence() (device scope) in
// barrier_release_rdna, not this function.

__device__ __forceinline__ void mem_fence()
{
    __builtin_amdgcn_s_waitcnt(0);
}

// =============================================================================
// Utility Macros
// =============================================================================

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef CEIL_DIVIDE
#define CEIL_DIVIDE(a, b) (((a) + (b) - 1) / (b))
#endif

#endif // RDNA35_UNIFIED_HIP
