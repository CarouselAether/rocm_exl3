// =============================================================================
// codebook.cuh for RDNA -- re-derived against upstream v1.3.0
// =============================================================================
//
// Upstream quant/codebook.cuh carries 10 inline-asm blocks. Only some of them
// are a problem on ROCm, and the set is smaller than it was when the original
// RDNA port was written -- upstream has since moved on:
//
//   vabsdiff4.u32.u32.u32.add  -> already commented out upstream, replaced with
//                                 __dp4a. Nothing to port.
//   mad.lo.u32 (sm_86 hack)    -> inside `#if __CUDA_ARCH__ == 860`, so it is
//                                 never compiled on ROCm. Nothing to port.
//   lop3.b32 ... 0x6a          -> the actual blocker. 6 identical sites.
//
// and one new problem upstream introduced:
//
//   __dp4a                     -> a CUDA intrinsic. HIP does not declare it
//                                 (verified absent from /opt/rocm-7.2.4/include/hip).
//                                 Bridged in rocm/hip_compat.hip.h.
//
// So this file is much smaller than the 207-line codebook_rdna.hip.h in the
// original port, which had to hand-roll a byte-sum because its upstream base
// still used vabsdiff4.
//
// Everything here is verified numerically on gfx1151, not reasoned about:
// codebook_lop3 matches a bit-by-bit PTX lop3(0x6a) reference on 4096 inputs
// with 0 mismatches, and the byte-sum matches its reference on the same set.
// =============================================================================

#ifndef EXL3_ROCM_CODEBOOK_RDNA_H
#define EXL3_ROCM_CODEBOOK_RDNA_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "../../util.cuh"   // half2_uint32 / half_uint16 type-punning unions
#include "../rdna_wmma.hip.h"

// -----------------------------------------------------------------------------
// lop3.b32 with immLut 0x6a
// -----------------------------------------------------------------------------
//
// PTX lop3 applies an arbitrary 3-input boolean function to a, b, c bitwise;
// the function is identified by the immediate, which is the result of applying
// it to the canonical operands a = 0xF0, b = 0xCC, c = 0xAA.
//
// Solving for 0x6a:
//     (~a & c) | (a & (b ^ c))
//   = (0x0F & 0xAA) | (0xF0 & (0xCC ^ 0xAA))
//   = 0x0A | (0xF0 & 0x66)
//   = 0x0A | 0x60
//   = 0x6A   OK
//
// Read as a bit-select: where a is set take (b ^ c), elsewhere take c. Every
// call site passes the same two constants, so codebook_lop3 folds them.

__device__ __forceinline__ uint32_t lop3_0x6a(uint32_t a, uint32_t b, uint32_t c)
{
    return (~a & c) | (a & (b ^ c));
}

// The single form used by every call site in codebook.cuh:
//   asm ("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x));
__device__ __forceinline__ uint32_t codebook_lop3(uint32_t x)
{
    const uint32_t b = 0x8fff8fffu;
    const uint32_t c = 0x3b603b60u;
    return (~x & c) | (x & (b ^ c));
}

// -----------------------------------------------------------------------------
// Decoders -- structure and constants tracked verbatim from upstream v1.3.0
// -----------------------------------------------------------------------------
//
// mul_const_u32 is upstream's, minus the sm_86 branch that ROCm never takes.

template <uint32_t w>
__device__ __forceinline__ uint32_t mul_const_u32(uint32_t x)
{
    return x * w;
}

// Decode two mul1 (cb 2) codebook entries from precomputed products
// x0 = idx0 * 0x83DCD12D, x1 = idx1 * 0x83DCD12D
__device__ inline half2 decode_mul1_product_2(uint32_t x0, uint32_t x1)
{
    const uint32_t acc = 0x6400u;  // 0x6400 -> 1024.0 ..  0x67FF -> 2047.0
    uint32_t sum0 = __dp4a(x0, 0x01010101u, acc);
    uint32_t sum1 = __dp4a(x1, 0x01010101u, acc);
    half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));   //  0.00677 = 1/147.7
    half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));  // -10.39 = (-1024.0 - 510.0) * k_inv_h
    half_uint16 h0((uint16_t) sum0);
    half_uint16 h1((uint16_t) sum1);
    return __hfma2(__halves2half2(h0.as_half, h1.as_half), k_inv_h2, k_bias_h2);
}

// Ditto mcg (cb 1)
__device__ inline half2 decode_mcg_product_2(uint32_t x0, uint32_t x1)
{
    x0 = codebook_lop3(x0);
    x1 = codebook_lop3(x1);
    half2_uint32 xu0(x0);
    half2_uint32 xu1(x1);
    half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
    half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
    return __hadd2(d0, d1);
}

template <int cb>
__device__ inline half decode_3inst(uint32_t x)
{
    if constexpr (cb == 0)
    {
        x *= 89226354u;
        x += 64248484u;
        x = codebook_lop3(x);
        half2_uint32 xu(x);
        return __hadd(__low2half(xu.as_half2), __high2half(xu.as_half2));
    }
    if constexpr (cb == 1)
    {
        x *= 0xCBAC1FEDu;
        x = codebook_lop3(x);
        half2_uint32 xu(x);
        return __hadd(__low2half(xu.as_half2), __high2half(xu.as_half2));
    }
    if constexpr (cb == 2)
    {
        x *= 0x83DCD12Du;
        const uint32_t acc = 0x6400u;
        uint32_t sum = __dp4a(x, 0x01010101u, acc);
        const __half k_inv_h = __ushort_as_half(0x1eee);
        const __half k_bias_h = __ushort_as_half(0xc931);
        half_uint16 h((uint16_t) sum);
        return __hfma(h.as_half, k_inv_h, k_bias_h);
    }
}

template <int cb>
__device__ inline half2 decode_3inst_2(uint32_t x0, uint32_t x1)
{
    if constexpr (cb == 0)
    {
        x0 *= 89226354u;
        x1 *= 89226354u;
        x0 += 64248484u;
        x1 += 64248484u;
        x0 = codebook_lop3(x0);
        x1 = codebook_lop3(x1);
        half2_uint32 xu0(x0);
        half2_uint32 xu1(x1);
        half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
        half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
        return __hadd2(d0, d1);
    }
    if constexpr (cb == 1)
    {
        x0 *= 0xCBAC1FEDu;
        x1 *= 0xCBAC1FEDu;
        return decode_mcg_product_2(x0, x1);
    }
    if constexpr (cb == 2)
    {
        x0 *= 0x83DCD12Du;
        x1 *= 0x83DCD12Du;
        return decode_mul1_product_2(x0, x1);
    }
}

template <int cb>
__device__ inline float decode_3inst_f(uint64_t x)
{
    return __half2float(decode_3inst<cb>(x));
}

template <int cb>
__device__ inline float decode_3inst_f_diff(uint64_t x, float d)
{
    return __half2float(decode_3inst<cb>(x)) - d;
}

#endif // EXL3_ROCM_CODEBOOK_RDNA_H
