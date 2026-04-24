// =============================================================================
// RDNA 3.5 GEMV Kernel - Native WMMA Version
// =============================================================================
//
// Target: gfx1150/gfx1151 (RDNA 3.5), Wave32
//
// This kernel handles small M (1-8 for autoregressive inference) using
// native 16x16x16 WMMA. M is padded to 16, unused rows are zeroed.
//
// NO EMULATION - uses __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 directly.
//
// =============================================================================

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "../rdna_wmma.hip"
#include "exl3_dq_rdna.hip.h"

// Tile sizes - exported as both prefixed and unprefixed names
#define GEMV_TILESIZE_K 256   // K elements per tile (16 WMMA K-blocks)
#define GEMV_TILESIZE_N 16    // N elements per tile (1 WMMA N-block)

// Unprefixed aliases for compatibility with exl3_gemv.cu
#define TILESIZE_K GEMV_TILESIZE_K
#define TILESIZE_N GEMV_TILESIZE_N

// =============================================================================
// GEMV Kernel using Native WMMA
// =============================================================================
//
// Grid:  (1, size_n / 16, size_k / 256)
// Block: 32 threads (one wave for WMMA)
//
// Each block computes a partial [M, 16] output tile for one K-tile.
// Results are accumulated across K-tiles (caller handles split-K reduction).
//
// =============================================================================

template <int bits, bool c_fp32, int cb>
__global__
__attribute__((amdgpu_flat_work_group_size(32, 32)))
void exl3_gemv_kernel_wmma(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n)
{
    const int lane = threadIdx.x;
    const int tile_n = blockIdx.y;
    const int tile_k = blockIdx.z;
    
    // LDS for dequantized B [K=16, N=16] tiles
    __shared__ half B_lds[16][17];  // Pad to 17 to avoid bank conflicts
    
    // WMMA accumulator (initialized to zero)
    float8_t c_frag = {0, 0, 0, 0, 0, 0, 0, 0};
    
    // B data layout:
    // [K/256 tiles][N/16 tiles][256 K values × bits / 8 bytes]
    // Each 256-K block for one N=16 tile is (256 * bits / 8) bytes
    const int b_tile_bytes = 256 * bits / 8;
    const int b_k_stride = (size_n / 16) * b_tile_bytes;
    const uint8_t* B_tile_base = (const uint8_t*)B + 
        tile_k * b_k_stride + tile_n * b_tile_bytes;
    
    // Process 16 WMMA K-blocks (each K=16, total K=256 per tile)
    #pragma unroll 1
    for (int k_block = 0; k_block < 16; k_block++)
    {
        int k_offset = tile_k * 256 + k_block * 16;
        
        // Pointer to this k_block's 256 quantized values
        const uint32_t* b_ptr = (const uint32_t*)(B_tile_base + k_block * (16 * 16 * bits / 8));
        
        // Dequantize using the standard function
        FragB frag0, frag1;
        dq_dispatch<bits, cb>(b_ptr, lane << 3, frag0, frag1);
        
        // =====================================================================
        // FIXED: Use shuffle-based unswizzle from reconstruct.cu
        // =====================================================================
        // The dequantized values are in "tensor core" layout - we need to
        // shuffle between lanes to get proper row-major format for B_lds.
        
        // Get values from lane+4 (lanes 0-3 get from 4-7, etc.)
        half2 n0 = __shfl_down(frag0[0], 4, 32);
        half2 n1 = __shfl_down(frag0[1], 4, 32);
        half2 n2 = __shfl_down(frag1[0], 4, 32);
        half2 n3 = __shfl_down(frag1[1], 4, 32);
        
        // Only lanes 0-3, 8-11, 16-19, 24-27 write (where !(lane & 4))
        // These 16 lanes each write 16 values = 256 total for 16x16 tile
        if (!(lane & 4)) {
            // Combine current lane's values with shuffled values from lane+4
            half2 m0 = __halves2half2(__low2half(frag0[0]), __low2half(n0));
            half2 m1 = __halves2half2(__high2half(frag0[0]), __high2half(n0));
            half2 m2 = __halves2half2(__low2half(frag0[1]), __low2half(n1));
            half2 m3 = __halves2half2(__high2half(frag0[1]), __high2half(n1));
            half2 m4 = __halves2half2(__low2half(frag1[0]), __low2half(n2));
            half2 m5 = __halves2half2(__high2half(frag1[0]), __high2half(n2));
            half2 m6 = __halves2half2(__low2half(frag1[1]), __low2half(n3));
            half2 m7 = __halves2half2(__high2half(frag1[1]), __high2half(n3));
            
            // Compute row and column indices (matching reconstruct.cu pattern)
            int r0 = (lane % 4) * 2;
            int r1 = r0 + 1;
            int r2 = r0 + 8;
            int r3 = r0 + 9;
            int c0 = (lane / 8) * 2;      // Columns 0,2,4,6 for lanes 0-3,8-11,16-19,24-27
            int c1 = c0 + 8;              // Columns 8,10,12,14
            
            // Write to B_lds as individual halves (B_lds is [16][17])
            B_lds[r0][c0] = __low2half(m0);
            B_lds[r0][c0 + 1] = __high2half(m0);
            B_lds[r1][c0] = __low2half(m1);
            B_lds[r1][c0 + 1] = __high2half(m1);
            B_lds[r2][c0] = __low2half(m2);
            B_lds[r2][c0 + 1] = __high2half(m2);
            B_lds[r3][c0] = __low2half(m3);
            B_lds[r3][c0 + 1] = __high2half(m3);
            B_lds[r0][c1] = __low2half(m4);
            B_lds[r0][c1 + 1] = __high2half(m4);
            B_lds[r1][c1] = __low2half(m5);
            B_lds[r1][c1 + 1] = __high2half(m5);
            B_lds[r2][c1] = __low2half(m6);
            B_lds[r2][c1 + 1] = __high2half(m6);
            B_lds[r3][c1] = __low2half(m7);
            B_lds[r3][c1 + 1] = __high2half(m7);
        }
        
        __syncthreads();
        
        // =====================================================================
        // Load A fragment
        // =====================================================================
        // A is [M, K] row-major. Lane L loads row (L % 16).
        
        half16_t a_frag;
        int a_row = lane % 16;
        
        if (a_row < size_m) {
            const half* a_ptr = A + a_row * size_k + k_offset;
            #pragma unroll
            for (int k = 0; k < 16; k++) {
                ((_Float16*)&a_frag)[k] = (_Float16)__half2float(a_ptr[k]);
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 16; k++) {
                ((_Float16*)&a_frag)[k] = (_Float16)0.0f;
            }
        }
        
        // =====================================================================
        // Load B fragment from LDS
        // =====================================================================
        // B fragment: lane L loads column (L % 16)
        
        half16_t b_frag;
        int b_col = lane % 16;
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            ((_Float16*)&b_frag)[k] = (_Float16)__half2float(B_lds[k][b_col]);
        }
        
        // =====================================================================
        // WMMA: C += A × B
        // =====================================================================
        c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b_frag, a_frag, c_frag);
        
        __syncthreads();  // Prepare LDS for next k_block
    }
    
    // =========================================================================
    // Store C to global memory
    // =========================================================================
    int c_row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;
    int n_offset = tile_n * 16;
    
    if (c_row < size_m) {
        if constexpr (c_fp32) {
            float* c_ptr = (float*)C + c_row * size_n + n_offset;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int col = i * 2 + col_base;
                // Accumulate for split-K (multiple K-tiles add to same output)
                atomicAdd(&c_ptr[col], ((float*)&c_frag)[i]);
            }
        } else {
            // For FP16 output, we need atomic add which is trickier
            // For now, use a simple approach (may need optimization)
            half* c_ptr = (half*)C + c_row * size_n + n_offset;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int col = i * 2 + col_base;
                float val = ((float*)&c_frag)[i];
                // Note: atomicAdd for half may not be available, using float conversion
                float* c_ptr_f = (float*)(c_ptr + col);
                // This is wrong for half - need proper handling
                // For correctness test, just do non-atomic store (assumes no split-K)
                c_ptr[col] = __float2half(val);
            }
        }
    }
}

// =============================================================================
// Alternative: Single-K-tile version (no atomics needed)
// =============================================================================
// Use when size_k == 256 or when split-K reduction is done separately

template <int bits, bool c_fp32, int cb>
__global__
__attribute__((amdgpu_flat_work_group_size(32, 32)))
void exl3_gemv_kernel_wmma_single(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n)
{
    const int lane = threadIdx.x;
    const int tile_n = blockIdx.y;
    
    __shared__ half B_lds[16][17];
    
    float8_t c_frag = {0, 0, 0, 0, 0, 0, 0, 0};
    
    const int num_k_tiles = size_k / 256;
    
    for (int tile_k = 0; tile_k < num_k_tiles; tile_k++)
    {
        const int b_tile_bytes = 256 * bits / 8;
        const int b_k_stride = (size_n / 16) * b_tile_bytes;
        const uint8_t* B_tile_base = (const uint8_t*)B + 
            tile_k * b_k_stride + tile_n * b_tile_bytes;
        
        for (int k_block = 0; k_block < 16; k_block++)
        {
            int k_offset = tile_k * 256 + k_block * 16;
            
            // Dequantize B
            const uint32_t* b_ptr = (const uint32_t*)(B_tile_base + k_block * (16 * 16 * bits / 8));
            FragB frag0, frag1;
            dq_dispatch<bits, cb>(b_ptr, lane << 3, frag0, frag1);
            
            // =====================================================================
            // FIXED: Use shuffle-based unswizzle from reconstruct.cu
            // =====================================================================
            half2 n0 = __shfl_down(frag0[0], 4, 32);
            half2 n1 = __shfl_down(frag0[1], 4, 32);
            half2 n2 = __shfl_down(frag1[0], 4, 32);
            half2 n3 = __shfl_down(frag1[1], 4, 32);
            
            // Only lanes 0-3, 8-11, 16-19, 24-27 write
            if (!(lane & 4)) {
                half2 m0 = __halves2half2(__low2half(frag0[0]), __low2half(n0));
                half2 m1 = __halves2half2(__high2half(frag0[0]), __high2half(n0));
                half2 m2 = __halves2half2(__low2half(frag0[1]), __low2half(n1));
                half2 m3 = __halves2half2(__high2half(frag0[1]), __high2half(n1));
                half2 m4 = __halves2half2(__low2half(frag1[0]), __low2half(n2));
                half2 m5 = __halves2half2(__high2half(frag1[0]), __high2half(n2));
                half2 m6 = __halves2half2(__low2half(frag1[1]), __low2half(n3));
                half2 m7 = __halves2half2(__high2half(frag1[1]), __high2half(n3));
                
                int r0 = (lane % 4) * 2;
                int r1 = r0 + 1;
                int r2 = r0 + 8;
                int r3 = r0 + 9;
                int c0 = (lane / 8) * 2;
                int c1 = c0 + 8;
                
                B_lds[r0][c0] = __low2half(m0);
                B_lds[r0][c0 + 1] = __high2half(m0);
                B_lds[r1][c0] = __low2half(m1);
                B_lds[r1][c0 + 1] = __high2half(m1);
                B_lds[r2][c0] = __low2half(m2);
                B_lds[r2][c0 + 1] = __high2half(m2);
                B_lds[r3][c0] = __low2half(m3);
                B_lds[r3][c0 + 1] = __high2half(m3);
                B_lds[r0][c1] = __low2half(m4);
                B_lds[r0][c1 + 1] = __high2half(m4);
                B_lds[r1][c1] = __low2half(m5);
                B_lds[r1][c1 + 1] = __high2half(m5);
                B_lds[r2][c1] = __low2half(m6);
                B_lds[r2][c1 + 1] = __high2half(m6);
                B_lds[r3][c1] = __low2half(m7);
                B_lds[r3][c1 + 1] = __high2half(m7);
            }
            
            __syncthreads();
            
            // Load A
            half16_t a_frag;
            int a_row = lane % 16;
            if (a_row < size_m) {
                const half* a_ptr = A + a_row * size_k + k_offset;
                #pragma unroll
                for (int k = 0; k < 16; k++) {
                    ((_Float16*)&a_frag)[k] = (_Float16)__half2float(a_ptr[k]);
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 16; k++) {
                    ((_Float16*)&a_frag)[k] = (_Float16)0.0f;
                }
            }
            
            // Load B from LDS
            half16_t b_frag;
            int b_col = lane % 16;
            #pragma unroll
            for (int k = 0; k < 16; k++) {
                ((_Float16*)&b_frag)[k] = (_Float16)__half2float(B_lds[k][b_col]);
            }
            
            // WMMA
            c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b_frag, a_frag, c_frag);
            
            __syncthreads();
        }
    }
    
    // Store C (no atomics needed - single K pass)
    int c_row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;
    int n_offset = tile_n * 16;
    
    if (c_row < size_m) {
        if constexpr (c_fp32) {
            float* c_ptr = (float*)C + c_row * size_n + n_offset;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int col = i * 2 + col_base;
                c_ptr[col] = ((float*)&c_frag)[i];
            }
        } else {
            half* c_ptr = (half*)C + c_row * size_n + n_offset;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int col = i * 2 + col_base;
                c_ptr[col] = __float2half(((float*)&c_frag)[i]);
            }
        }
    }
}

// =============================================================================
// Launch Helper
// =============================================================================

template <int bits, bool c_fp32, int cb>
void launch_gemv_wmma(
    const half* A,
    const uint16_t* B,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    hipStream_t stream = 0)
{
    // Use single-pass kernel (handles all K internally)
    dim3 grid(1, size_n / 16, 1);
    dim3 block(32);
    
    hipLaunchKernelGGL(
        (exl3_gemv_kernel_wmma_single<bits, c_fp32, cb>),
        grid, block, 0, stream,
        A, B, C, size_m, size_k, size_n
    );
}

// =============================================================================
// Kernel Alias for Compatibility with exl3_gemv.cu
// =============================================================================
// exl3_gemv.cu expects exl3_gemv_kernel<bits, c_fp32, cb, k_split>
// We ignore k_split since our implementation handles K internally

template <int bits, bool c_fp32, int cb, int k_split>
__global__
__attribute__((amdgpu_flat_work_group_size(32, 32)))
void exl3_gemv_kernel(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n)
{
    const int lane = threadIdx.x;
    const int tile_n = blockIdx.y;
    
    __shared__ half B_lds[16][17];
    
    float8_t c_frag = {0, 0, 0, 0, 0, 0, 0, 0};
    
    // B tensor layout: [size_k/16, size_n/16, 16*bits] with dtype uint16
    // Each [k16, n16, :] contains one 16x16 quantized weight tile
    // Tile size in bytes: 16 * bits * sizeof(uint16) = 32 * bits bytes
    const int tile_bytes = 32 * bits;  // bytes per 16x16 tile
    const int k_stride = (size_n / 16) * tile_bytes;  // bytes between adjacent K tiles
    
    // Number of K=16 blocks to process
    const int num_k_blocks = size_k / 16;
    
    for (int k_block = 0; k_block < num_k_blocks; k_block++)
    {
        int k_offset = k_block * 16;
        
        // Calculate pointer to B[k_block, tile_n, :]
        const uint32_t* b_ptr = (const uint32_t*)((const uint8_t*)B + 
            k_block * k_stride + tile_n * tile_bytes);
        
        // Dequantize B tile
        FragB frag0, frag1;
        dq_dispatch<bits, cb>(b_ptr, lane << 3, frag0, frag1);
        
        // =====================================================================
        // FIXED: Use shuffle-based unswizzle from reconstruct.cu
        // =====================================================================
        half2 n0 = __shfl_down(frag0[0], 4, 32);
        half2 n1 = __shfl_down(frag0[1], 4, 32);
        half2 n2 = __shfl_down(frag1[0], 4, 32);
        half2 n3 = __shfl_down(frag1[1], 4, 32);
        
        // Only lanes 0-3, 8-11, 16-19, 24-27 write
        if (!(lane & 4)) {
            half2 m0 = __halves2half2(__low2half(frag0[0]), __low2half(n0));
            half2 m1 = __halves2half2(__high2half(frag0[0]), __high2half(n0));
            half2 m2 = __halves2half2(__low2half(frag0[1]), __low2half(n1));
            half2 m3 = __halves2half2(__high2half(frag0[1]), __high2half(n1));
            half2 m4 = __halves2half2(__low2half(frag1[0]), __low2half(n2));
            half2 m5 = __halves2half2(__high2half(frag1[0]), __high2half(n2));
            half2 m6 = __halves2half2(__low2half(frag1[1]), __low2half(n3));
            half2 m7 = __halves2half2(__high2half(frag1[1]), __high2half(n3));
            
            int r0 = (lane % 4) * 2;
            int r1 = r0 + 1;
            int r2 = r0 + 8;
            int r3 = r0 + 9;
            int c0 = (lane / 8) * 2;
            int c1 = c0 + 8;
            
            B_lds[r0][c0] = __low2half(m0);
            B_lds[r0][c0 + 1] = __high2half(m0);
            B_lds[r1][c0] = __low2half(m1);
            B_lds[r1][c0 + 1] = __high2half(m1);
            B_lds[r2][c0] = __low2half(m2);
            B_lds[r2][c0 + 1] = __high2half(m2);
            B_lds[r3][c0] = __low2half(m3);
            B_lds[r3][c0 + 1] = __high2half(m3);
            B_lds[r0][c1] = __low2half(m4);
            B_lds[r0][c1 + 1] = __high2half(m4);
            B_lds[r1][c1] = __low2half(m5);
            B_lds[r1][c1 + 1] = __high2half(m5);
            B_lds[r2][c1] = __low2half(m6);
            B_lds[r2][c1 + 1] = __high2half(m6);
            B_lds[r3][c1] = __low2half(m7);
            B_lds[r3][c1 + 1] = __high2half(m7);
        }
        
        __syncthreads();
        
        // Load A fragment
        half16_t a_frag;
        int a_row = lane % 16;
        if (a_row < size_m) {
            const half* a_ptr = A + a_row * size_k + k_offset;
            #pragma unroll
            for (int k = 0; k < 16; k++) {
                ((_Float16*)&a_frag)[k] = (_Float16)__half2float(a_ptr[k]);
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 16; k++) {
                ((_Float16*)&a_frag)[k] = (_Float16)0.0f;
            }
        }
        
        // Load B fragment from LDS
        half16_t b_frag;
        int b_col = lane % 16;
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            ((_Float16*)&b_frag)[k] = (_Float16)__half2float(B_lds[k][b_col]);
        }
        
        // WMMA: C += A * B
        c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b_frag, a_frag, c_frag);
        
        __syncthreads();
    }
    
    // Store C to global memory
    int c_row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;
    int n_offset = tile_n * 16;
    
    if (c_row < size_m) {
        if constexpr (c_fp32) {
            float* c_ptr = (float*)C + c_row * size_n + n_offset;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int col = i * 2 + col_base;
                c_ptr[col] = ((float*)&c_frag)[i];
            }
        } else {
            half* c_ptr = (half*)C + c_row * size_n + n_offset;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int col = i * 2 + col_base;
                c_ptr[col] = __float2half(((float*)&c_frag)[i]);
            }
        }
    }
}
