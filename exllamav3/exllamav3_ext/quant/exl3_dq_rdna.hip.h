#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "../rdna_wmma.hip"
#include "codebook_rdna.hip.h"

// ============================================================================
// Dequantization for RDNA 3.5
// ============================================================================

__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift)
{
    uint64_t merged = ((uint64_t)a << 32) | (uint64_t)b;
    return (uint32_t)(merged >> shift);
}

template <int bits, int cb>
__device__ __forceinline__ half dq(const uint32_t* ptr, int t_offset)
{
    int b0 = t_offset * bits + bits - 16 + 256 * bits;
    int b1 = b0 + 16;
    int i0 = b0 / 32;
    int i1 = (b1 - 1) / 32;
    int s0 = (i1 + 1) * 32 - b1;

    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i1 % (bits * 256 / 32)];

    // FIX: Must use 0xffff mask, not bit-width specific mask
    uint32_t w0 = fshift(b, a, s0) & 0xffff;
    return decode_3inst<cb>(w0);
}

template <int bits, int cb>
__device__ __forceinline__ half2 dq2(const uint32_t* ptr, int t_offset)
{
    int b0 = t_offset * bits + bits - 16 + 256 * bits;
    int b1 = b0 + 16;
    int i0 = b0 / 32;
    int i1 = (b1 - 1) / 32;
    int s0 = (i1 + 1) * 32 - b1;

    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i1 % (bits * 256 / 32)];
    
    // FIX: Must use 0xffff mask, not bit-width specific mask
    uint32_t w1 = fshift(b, a, s0) & 0xffff;
    uint32_t w0 = fshift(b, a, s0 + bits) & 0xffff;
    return decode_3inst_2<cb>(w0, w1);
}

template <int bits, int cb>
__device__ __forceinline__ void dq4(const uint32_t* ptr, int t_offset, FragB& frag)
{
    int b0 = (t_offset + 257) * bits - 16;
    int b1 = b0 + 3 * bits;
    int b2 = b1 + 16;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;

    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i2 % (bits * 256 / 32)];
    
    // FIX: Must use 0xffff mask, not bit-width specific mask
    uint32_t w3 = fshift(b, a, s2) & 0xffff;
    uint32_t w2 = fshift(b, a, s2 + bits) & 0xffff;
    uint32_t w1 = fshift(b, a, s2 + bits * 2) & 0xffff;
    uint32_t w0 = fshift(b, a, s2 + bits * 3) & 0xffff;
    
    half2 d0d1 = decode_3inst_2<cb>(w0, w1);
    half2 d2d3 = decode_3inst_2<cb>(w2, w3);
    frag[0] = d0d1;
    frag[1] = d2d3;
}

template <int bits, int cb>
__device__ __forceinline__ void dq2x2(const uint32_t* ptr, int t_offset, FragB& frag)
{
    #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        int b0 = (t_offset + 2 * i + 257) * bits - 16;
        int b1 = b0 + 1 * bits;
        int b2 = b1 + 16;
        int i0 = b0 / 32;
        int i2 = (b2 - 1) / 32;
        int s2 = (i2 + 1) * 32 - b2;

        uint32_t a = ptr[i0 % (bits * 256 / 32)];
        uint32_t b = ptr[i2 % (bits * 256 / 32)];
        
        // FIX: Must use 0xffff mask, not bit-width specific mask
        uint32_t w1 = fshift(b, a, s2) & 0xffff;
        uint32_t w0 = fshift(b, a, s2 + bits) & 0xffff;
        
        half2 d0d1 = decode_3inst_2<cb>(w0, w1);
        frag[i] = d0d1;
    }
}

template <int bits, int cb, int align>
__device__ __forceinline__ void dq8(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    int b1 = (t_offset + 257) * bits;
    int b0 = b1 - 16;
    int b2 = b1 + bits * 7;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;

    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i2 % (bits * 256 / 32)];
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    
    if constexpr (align == 1)
    {
        w7 = fshift(b, a, s2);
        w6 = fshift(b, a, s2 + bits);
        w5 = fshift(b, a, s2 + bits * 2);
        w4 = fshift(b, a, s2 + bits * 3);
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = fshift(b, a, s2 + bits * 5);
        w1 = fshift(b, a, s2 + bits * 6);
        w0 = fshift(b, a, s2 + bits * 7);
    }
    if constexpr (align == 2)
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = fshift(b, a, s2 + bits * 2);
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = fshift(b, a, s2 + bits * 6);
        w0 = w1 >> bits;
    }
    if constexpr (align == 4)
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
    }
    if constexpr (align == 8)
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = w4 >> bits;
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
    }
    
    // FIX: Must use 0xffff mask, not bit-width specific mask
    half2 d0d1 = decode_3inst_2<cb>(w0 & 0xffff, w1 & 0xffff);
    half2 d2d3 = decode_3inst_2<cb>(w2 & 0xffff, w3 & 0xffff);
    half2 d4d5 = decode_3inst_2<cb>(w4 & 0xffff, w5 & 0xffff);
    half2 d6d7 = decode_3inst_2<cb>(w6 & 0xffff, w7 & 0xffff);
    frag0[0] = d0d1;
    frag0[1] = d2d3;
    frag1[0] = d4d5;
    frag1[1] = d6d7;
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_4bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, s, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 3;
    i0 = (i1 + 31) & 31;
    a = ptr[i0];
    b = ptr[i1];
    
    FSHF_IMM(s, b, a, 20);
    
    // FIX: Must use 0xffff mask, not bit-width specific mask
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);  w6 &= 0xffff;
    BFE16_IMM(w5, b, 8);  w5 &= 0xffff;
    BFE16_IMM(w4, b, 12); w4 &= 0xffff;
    BFE16_IMM(w3, b, 16); w3 &= 0xffff;
    
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);  w1 &= 0xffff;
    BFE16_IMM(w0, s, 8);  w0 &= 0xffff;
    
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_2bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 4;
    i0 = (i1 + 15) & 15;
    a = ptr[i0];
    b = ptr[i1];
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    
    // FIX: Must use 0xffff mask, not bit-width specific mask
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);  w6 &= 0xffff;
    BFE16_IMM(w5, b, 4);  w5 &= 0xffff;
    BFE16_IMM(w4, b, 6);  w4 &= 0xffff;
    BFE16_IMM(w3, b, 8);  w3 &= 0xffff;
    BFE16_IMM(w2, b, 10); w2 &= 0xffff;
    BFE16_IMM(w1, b, 12); w1 &= 0xffff;
    BFE16_IMM(w0, b, 14); w0 &= 0xffff;
    
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_1bit(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 5;
    i0 = (i1 + 7) & 7;
    a = ptr[i0];
    b = ptr[i1];
    b = fshift(b, a, ((~t_offset) & 24));
    
    // FIX: Must use 0xffff mask, not bit-width specific mask
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 1);  w6 &= 0xffff;
    BFE16_IMM(w5, b, 2);  w5 &= 0xffff;
    BFE16_IMM(w4, b, 3);  w4 &= 0xffff;
    BFE16_IMM(w3, b, 4);  w3 &= 0xffff;
    BFE16_IMM(w2, b, 5);  w2 &= 0xffff;
    BFE16_IMM(w1, b, 6);  w1 &= 0xffff;
    BFE16_IMM(w0, b, 7);  w0 &= 0xffff;
    
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_4bits_bfe64(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    int i1 = t_offset / 8;
    int i0 = (i1 + 31) % 32;
    uint32_t a = ptr[i0];
    uint32_t b = ptr[i1];
    
    // bfe64 extracts 16-bit values correctly - no additional masking needed
    uint32_t w7 = bfe64(b, a, 0, 16);
    uint32_t w6 = bfe64(b, a, 4, 16);
    uint32_t w5 = bfe64(b, a, 8, 16);
    uint32_t w4 = bfe64(b, a, 12, 16);
    uint32_t w3 = bfe64(b, a, 16, 16);
    uint32_t w2 = bfe64(b, a, 20, 16);
    uint32_t w1 = bfe64(b, a, 24, 16);
    uint32_t w0 = bfe64(b, a, 28, 16);

    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int bits, int cb>
__device__ __forceinline__ void dq_dispatch(const uint32_t* ptr, int idx, FragB& frag0, FragB& frag1)
{
    if constexpr (bits == 1)
    {
        dq8_aligned_1bit<cb>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 2)
    {
        dq8_aligned_2bits<cb>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 3)
    {
        dq8<bits, cb, 4>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 4)
    {
        dq8_aligned_4bits<cb>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 5)
    {
        dq4<bits, cb>(ptr, idx, frag0);
        dq4<bits, cb>(ptr, idx + 4, frag1);
    }
    else if constexpr (bits == 6)
    {
        dq4<bits, cb>(ptr, idx, frag0);
        dq4<bits, cb>(ptr, idx + 4, frag1);
    }
    else if constexpr (bits == 7)
    {
        dq2x2<bits, cb>(ptr, idx, frag0);
        dq2x2<bits, cb>(ptr, idx + 4, frag1);
    }
    else if constexpr (bits == 8)
    {
        dq4<bits, cb>(ptr, idx, frag0);
        dq4<bits, cb>(ptr, idx + 4, frag1);
    }
}
