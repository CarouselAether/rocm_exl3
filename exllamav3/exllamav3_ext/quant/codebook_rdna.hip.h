#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Include util.cuh for half2_uint32 and half_uint16 unions
#include "../util.cuh"

// ============================================================================
// RDNA 3.5 Codebook Decoding
// ============================================================================
//
// Ported from CUDA codebook.cuh
//
// Key changes:
// - lop3.b32 replaced with equivalent AND/OR/XOR operations
// - vabsdiff4 replaced with v_sad_u8 or byte-wise computation
// - Type punning unions come from util.cuh
// ============================================================================

// ============================================================================
// lop3.b32 emulation
// ============================================================================
//
// CUDA lop3.b32 performs: result = LUT[a_bit, b_bit, c_bit] for each bit
// The LUT is encoded in the immediate value (0x6a = 0b01101010)
//
// Truth table for LUT 0x6a:
//   a b c | result
//   0 0 0 | 0
//   0 0 1 | 1
//   0 1 0 | 0
//   0 1 1 | 1
//   1 0 0 | 0
//   1 0 1 | 1
//   1 1 0 | 1
//   1 1 1 | 0
//
// This is: (a XOR b) XOR c, or equivalently: a ^ b ^ c
// But wait - 0x6a specifically is: (a & ~b) | (~a & c) | (b & c) ... let's verify
//
// Actually, for the specific case in the codebook:
// lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a
// 
// With LUT 0x6a = 0b01101010:
// result_bit = LUT[(x_bit << 2) | (0x8fff8fff_bit << 1) | 0x3b603b60_bit]
//
// Simplification: This is equivalent to:
// result = (x & 0x3b603b60) | (~x & 0x8fff8fff & 0x3b603b60) | (x & 0x8fff8fff & ~0x3b603b60)
// 
// Let's just compute it directly based on the truth table

__device__ __forceinline__ uint32_t lop3_0x6a(uint32_t a, uint32_t b, uint32_t c)
{
    // LUT 0x6a = 0b01101010
    // Bit i of result = LUT[ (a_i << 2) | (b_i << 1) | c_i ]
    //
    // Analyzing 0x6a:
    // Index 0 (000): 0
    // Index 1 (001): 1  -> c & ~b & ~a
    // Index 2 (010): 0
    // Index 3 (011): 1  -> c & b & ~a
    // Index 4 (100): 0
    // Index 5 (101): 1  -> c & ~b & a
    // Index 6 (110): 1  -> ~c & b & a
    // Index 7 (111): 0
    //
    // Result = (c & ~a) | (a & b & ~c) = c ^ (a & (b ^ c))
    // Or more directly: (~a & c) | (a & (b ^ c))
    
    return (~a & c) | (a & (b ^ c));
}

// Specific implementation for the codebook's usage pattern
__device__ __forceinline__ uint32_t codebook_lop3(uint32_t x)
{
    // lop3.b32 x, x, 0x8fff8fff, 0x3b603b60, 0x6a
    // a = x, b = 0x8fff8fff, c = 0x3b603b60
    
    const uint32_t b = 0x8fff8fff;
    const uint32_t c = 0x3b603b60;
    
    return (~x & c) | (x & (b ^ c));
}

// ============================================================================
// Byte-sum-add: equivalent to NVIDIA's vabsdiff4(x, 0, acc) — i.e. sum the 4
// bytes of x and add acc. Hot path: EXL3 codebook cb==2 decoder (2-bit layers).
//
// Implemented via V_DOT4_U32_U8 (RDNA 3.5 ISA table p. 67, VOP3P dot-product
// instruction): dot(x, 0x01010101, acc) = x.b0 + x.b1 + x.b2 + x.b3 + acc
// in a single instruction. Replaces the 4 byte-extracts + 4 adds the generic
// C version would need. Exposed via clang builtin __builtin_amdgcn_udot4;
// the final `bool clamp` arg is false (no saturation).
//
// The asymmetric vabsdiff4 variant (src1 != 0) is not used anywhere in the
// RDNA build path (EXL3 cb==2 always passes src1=0) so it's omitted.
// ============================================================================

__device__ __forceinline__ uint32_t sum_bytes_add(uint32_t src0, uint32_t acc)
{
    return __builtin_amdgcn_udot4(src0, 0x01010101u, acc, false);
}

// ============================================================================
// Multiply by constant (no special handling needed on RDNA)
// ============================================================================

template <uint32_t w>
__device__ __forceinline__ uint32_t mul_const_u32(uint32_t x)
{
    return x * w;
}

// ============================================================================
// Decode Functions
// ============================================================================

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
        x = mul_const_u32<0xCBAC1FEDu>(x);
        x = codebook_lop3(x);
        half2_uint32 xu(x);
        return __hadd(__low2half(xu.as_half2), __high2half(xu.as_half2));
    }
    if constexpr (cb == 2)
    {
        x *= 0x83DCD12Du;
        uint32_t sum = sum_bytes_add(x, 0x6400u);  // acc = 0x6400 -> 1024.0
        const half k_inv_h = __ushort_as_half(0x1eee);   //  0.00677 = 1/147.7
        const half k_bias_h = __ushort_as_half(0xc931);  // -10.39
        half_uint16 h((uint16_t)sum);
        return __hfma(h.as_half, k_inv_h, k_bias_h);
    }
    
    // Default fallback
    return __float2half(0.0f);
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
        x0 = mul_const_u32<0xCBAC1FEDu>(x0);
        x1 = mul_const_u32<0xCBAC1FEDu>(x1);
        x0 = codebook_lop3(x0);
        x1 = codebook_lop3(x1);
        half2_uint32 xu0(x0);
        half2_uint32 xu1(x1);
        half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
        half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
        return __hadd2(d0, d1);
    }
    if constexpr (cb == 2)
    {
        x0 *= 0x83DCD12Du;
        x1 *= 0x83DCD12Du;
        uint32_t sum0 = sum_bytes_add(x0, 0x6400u);
        uint32_t sum1 = sum_bytes_add(x1, 0x6400u);
        half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));
        half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
        half_uint16 h0((uint16_t)sum0);
        half_uint16 h1((uint16_t)sum1);
        return __hfma2(__halves2half2(h0.as_half, h1.as_half), k_inv_h2, k_bias_h2);
    }
    
    // Default fallback
    return __float2half2_rn(0.0f);
}

template <int cb>
__device__ inline float decode_3inst_f(uint64_t x)
{
    return __half2float(decode_3inst<cb>((uint32_t)x));
}

template <int cb>
__device__ inline float decode_3inst_f_diff(uint64_t x, float d)
{
    return __half2float(decode_3inst<cb>((uint32_t)x)) - d;
}
