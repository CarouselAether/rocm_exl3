#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../rdna_wmma.hip"
#include "exl3_dq_rdna.hip.h"

// ============================================================================
// Constants & Macros
// ============================================================================
#define EXL3_GEMM_THREADS 256
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define SMEM_MAX_RDNA (64 * 1024)

#define EXL3_GEMM_T_ARGS_RDNA \
    const int bits, const bool c_fp32, const int cb, \
    const int TILESIZE_M, const int TILESIZE_K, const int TILESIZE_N, \
    const int SH_STAGES, const int FRAG_STAGES

__device__ __forceinline__ void barrier_acquire_rdna(int* lock, int stage) {
    if (threadIdx.x == 0) {
        while (atomicAdd(lock, 0) != stage) {
            // Spin
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void barrier_release_rdna(int* lock, int val, bool reset) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if (reset) {
            atomicExch(lock, 0);
            return;
        }
        __threadfence();
        atomicAdd(lock, val);
    }
}

// ============================================================================
// Main GEMM Kernel Inner
// ============================================================================
template<EXL3_GEMM_T_ARGS_RDNA>
__device__ __forceinline__
void exl3_gemm_kernel_inner_rdna(
    const half* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C,
    const int size_m, const int size_k, const int size_n, int* __restrict__ locks)
{
    constexpr int TILEBLOCKS_M = TILESIZE_M / WMMA_M;
    constexpr int TILEBLOCKS_K = TILESIZE_K / WMMA_K;
    constexpr int TILEBLOCKS_N = TILESIZE_N / WMMA_N;
    constexpr int NUM_WARPS = EXL3_GEMM_THREADS / 32;
    constexpr int FRAGS_N_PER_WARP = TILEBLOCKS_N / NUM_WARPS;
    constexpr int FRAGS_M = TILEBLOCKS_M;
    
    constexpr int sh_a_stage_size = TILESIZE_M * TILESIZE_K;
    constexpr int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    constexpr int sh_c_size = TILEBLOCKS_K * 8 * EXL3_GEMM_THREADS;
    constexpr int sh_b_dq_size = NUM_WARPS * 16 * 17;

    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*)(sh_a + SH_STAGES * sh_a_stage_size);
    float* sh_c = (float*)(sh_b + SH_STAGES * sh_b_stage_size);
    half* sh_b_dq = (half*)(sh_c + sh_c_size);

    const int t = threadIdx.x % EXL3_GEMM_THREADS;
    const int sub_k = threadIdx.x / EXL3_GEMM_THREADS;
    const int warp_id = t / 32;
    const int lane_id = t % 32;

    const int tiles_k = size_k / TILESIZE_K;
    const int tiles_n = size_n / TILESIZE_N;
    const int blocks_n = tiles_n * TILEBLOCKS_N;
    const int num_slices = gridDim.x;
    const int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;
    const int slice_end = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
    
    auto index_k = [&](int slice_i) { return slice_i % tiles_k; };
    auto index_n = [&](int slice_i) { return slice_i / tiles_k; };
    const int slice_m = 0;
    const int max_m = min(size_m - slice_m * TILESIZE_M, TILESIZE_M);

    WmmaFragA frag_a[FRAG_STAGES][FRAGS_M];
    WmmaFragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
    WmmaFragC frag_c[FRAGS_M][FRAGS_N_PER_WARP];

    auto clear_frag_c = [&]() {
        #pragma unroll
        for (int m = 0; m < FRAGS_M; m++)
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; n++)
                frag_c[m][n].clear();
    };

    // State setup
    int slice0_k = index_k(slice_beg);
    int slice0_n = index_n(slice_beg);
    int slice0_iters = slice_end - slice_beg;
    
    const half* gl_a_ptr = A + slice_m * TILESIZE_M * size_k + slice0_k * TILESIZE_K;
    const uint16_t* gl_b_ptr = B + slice0_k * blocks_n * TILEBLOCKS_K * 256 / 16 * bits
                                 + slice0_n * TILEBLOCKS_N * 256 / 16 * bits;
    
    half* sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;
    uint16_t* sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

    int slice1_k = slice0_k;
    int slice1_n = slice0_n;
    int slice1_iters = slice0_iters;
    half* sh1_a_ptr = sh0_a_ptr;
    uint16_t* sh1_b_ptr = sh0_b_ptr;

    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int slice2_n = slice0_n;
    int slice2_iters = slice0_iters;

    half* gl_c_ptr_16 = ((half*)C) + slice_m * TILESIZE_M * size_n + slice2_n * TILESIZE_N;
    float* gl_c_ptr_32 = ((float*)C) + slice_m * TILESIZE_M * size_n + slice2_n * TILESIZE_N;

    // Advance functions
    auto advance0 = [&]() {
        slice0_k++; slice0_iters--;
        int stage = slice0_iters % SH_STAGES;
        sh0_a_ptr = sh_a + stage * sh_a_stage_size;
        sh0_b_ptr = sh_b + stage * sh_b_stage_size;
        if (slice0_k >= tiles_k) {
            slice0_k = 0; slice0_n++;
            gl_a_ptr = A + slice_m * TILESIZE_M * size_k;
            gl_b_ptr = B + slice0_n * TILEBLOCKS_N * 256 / 16 * bits;
        } else {
            gl_a_ptr += TILESIZE_K;
            gl_b_ptr += blocks_n * TILEBLOCKS_K * 256 / 16 * bits;
        }
    };
    auto advance1 = [&]() {
        slice1_k++; slice1_iters--;
        int stage = slice1_iters % SH_STAGES;
        sh1_a_ptr = sh_a + stage * sh_a_stage_size;
        sh1_b_ptr = sh_b + stage * sh_b_stage_size;
        if (slice1_k >= tiles_k) { slice1_k = 0; slice1_n++; }
    };
    auto advance2 = [&]() {
        slice2_k++; slice2_iters--;
        if (slice2_k >= tiles_k) {
            slice2_k = 0; slice2_k0 = 0; slice2_n++;
            if constexpr (c_fp32) gl_c_ptr_32 += TILESIZE_N; else gl_c_ptr_16 += TILESIZE_N;
        }
    };

    // Load Gl -> Sh
    auto async_load_gl = [&]() {
        if (sub_k) { mem_fence(); return; }
        if (slice0_iters) {
            constexpr int load_a_iters = (TILESIZE_M * TILESIZE_K / 8 + EXL3_GEMM_THREADS - 1) / EXL3_GEMM_THREADS;
            #pragma unroll
            for (int i = 0; i < load_a_iters; i++) {
                int idx = i * EXL3_GEMM_THREADS + t;
                if (idx < TILESIZE_M * TILESIZE_K / 8) {
                    int m = idx / (TILESIZE_K / 8); int k = idx % (TILESIZE_K / 8);
                    if (m < max_m) {
                        const uint4* src = (const uint4*)(gl_a_ptr + m * size_k) + k;
                        uint4* dst = (uint4*)(sh0_a_ptr + m * TILESIZE_K) + k;
                        *dst = *src;
                    }
                }
            }
            constexpr int load_b_iters = (sh_b_stage_size / 8 + EXL3_GEMM_THREADS - 1) / EXL3_GEMM_THREADS;
            #pragma unroll
            for (int i = 0; i < load_b_iters; i++) {
                int idx = i * EXL3_GEMM_THREADS + t;
                if (idx < sh_b_stage_size / 8) {
                    const uint4* src = (const uint4*)gl_b_ptr + idx;
                    uint4* dst = (uint4*)sh0_b_ptr + idx;
                    *dst = *src;
                }
            }
            advance0();
        }
        mem_fence();
    };

    // =========================================================================
    // Load Sh -> Reg  (FIXED: with shuffle-based unswizzle for B)
    // =========================================================================
    auto load_frags = [&](int buf) {
        if (!slice1_iters) return;
        
        // Load A fragments (unchanged)
        #pragma unroll
        for (int m = 0; m < FRAGS_M; m++) {
            const half* a_ptr = sh1_a_ptr + m * WMMA_M * TILESIZE_K + sub_k * WMMA_K;
            rdna_wmma::load_matrix_a(frag_a[buf][m], a_ptr, TILESIZE_K);
        }
        
        // Load B fragments with FIXED shuffle-based unswizzle
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; n++) {
            int n_idx = warp_id * FRAGS_N_PER_WARP + n;
            half* B_lds = sh_b_dq + warp_id * 16 * 17;
            const uint32_t* b_quant = (const uint32_t*)(sh1_b_ptr + (sub_k * TILEBLOCKS_N + n_idx) * 256 / 16 * bits);
            
            // Dequantize
            FragB frag0, frag1;
            dq_dispatch<bits, cb>(b_quant, lane_id << 3, frag0, frag1);
            
            // =====================================================================
            // FIX: Shuffle-based unswizzle (matching reconstruct.cu)
            // =====================================================================
            // Get values from lane+4
            half2 n0 = __shfl_down(frag0[0], 4, 32);
            half2 n1 = __shfl_down(frag0[1], 4, 32);
            half2 n2 = __shfl_down(frag1[0], 4, 32);
            half2 n3 = __shfl_down(frag1[1], 4, 32);
            
            // Only lanes where !(lane_id & 4) write (lanes 0-3, 8-11, 16-19, 24-27)
            // These 16 lanes each write 16 values = 256 total for 16x16 tile
            if (!(lane_id & 4))
            {
                // Combine values from current lane and lane+4
                half2 m0 = __halves2half2(__low2half(frag0[0]), __low2half(n0));
                half2 m1 = __halves2half2(__high2half(frag0[0]), __high2half(n0));
                half2 m2 = __halves2half2(__low2half(frag0[1]), __low2half(n1));
                half2 m3 = __halves2half2(__high2half(frag0[1]), __high2half(n1));
                half2 m4 = __halves2half2(__low2half(frag1[0]), __low2half(n2));
                half2 m5 = __halves2half2(__high2half(frag1[0]), __high2half(n2));
                half2 m6 = __halves2half2(__low2half(frag1[1]), __low2half(n3));
                half2 m7 = __halves2half2(__high2half(frag1[1]), __high2half(n3));
                
                // Compute row/col indices for [K=16][N=16] with stride 17
                int r0 = (lane_id % 4) * 2;
                int r1 = r0 + 1;
                int r2 = r0 + 8;
                int r3 = r0 + 9;
                int c0 = (lane_id / 8) * 2;   // Columns 0,2,4,6 for different lane groups
                int c1 = c0 + 8;               // Columns 8,10,12,14
                
                // Write to B_lds[row][col] with row-major layout, stride 17
                // This is B[K][N] layout that load_matrix_b expects
                B_lds[r0 * 17 + c0] = __low2half(m0);
                B_lds[r0 * 17 + c0 + 1] = __high2half(m0);
                B_lds[r1 * 17 + c0] = __low2half(m1);
                B_lds[r1 * 17 + c0 + 1] = __high2half(m1);
                B_lds[r2 * 17 + c0] = __low2half(m2);
                B_lds[r2 * 17 + c0 + 1] = __high2half(m2);
                B_lds[r3 * 17 + c0] = __low2half(m3);
                B_lds[r3 * 17 + c0 + 1] = __high2half(m3);
                B_lds[r0 * 17 + c1] = __low2half(m4);
                B_lds[r0 * 17 + c1 + 1] = __high2half(m4);
                B_lds[r1 * 17 + c1] = __low2half(m5);
                B_lds[r1 * 17 + c1 + 1] = __high2half(m5);
                B_lds[r2 * 17 + c1] = __low2half(m6);
                B_lds[r2 * 17 + c1 + 1] = __high2half(m6);
                B_lds[r3 * 17 + c1] = __low2half(m7);
                B_lds[r3 * 17 + c1 + 1] = __high2half(m7);
            }
            // =====================================================================
            
            __syncthreads();
            
            rdna_wmma::load_matrix_b(frag_b[buf][n], B_lds, 17);
            __syncthreads();
        }
        advance1();
    };

    auto matmul = [&](int buf) {
        #pragma unroll
        for (int m = 0; m < FRAGS_M; m++) {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; n++) {
                rdna_wmma::mma_sync(frag_c[m][n], frag_a[buf][m], frag_b[buf][n]);
            }
        }
    };
    
	auto threadblock_reduce = [&]() {
	    if constexpr (TILEBLOCKS_K > 1) {
		#pragma unroll
		for (int m = 0; m < FRAGS_M; m++) {
		    #pragma unroll
		    for (int n = 0; n < FRAGS_N_PER_WARP; n++) {
		        // Serialize the reduction across sub_k values
		        for (int src_k = 1; src_k < TILEBLOCKS_K; src_k++) {
			    if (sub_k == src_k) {
			        float* sh_red = sh_c + (sub_k * EXL3_GEMM_THREADS + t) * 8;
			        #pragma unroll
			        for (int i = 0; i < 8; i++) 
				    sh_red[i] = frag_c[m][n][i];
			    }
			    __syncthreads();
			    
			    if (sub_k == 0) {
			        float* sh_red = sh_c + (src_k * EXL3_GEMM_THREADS + t) * 8;
			        #pragma unroll
			        for (int i = 0; i < 8; i++) 
				    frag_c[m][n][i] += sh_red[i];
			    }
			    __syncthreads();
		        }
		    }
		}
	    }
	};

    auto reduce = [&]() {
        threadblock_reduce();
        int lock_i = tiles_k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[slice_m * blocks_n + slice2_n];
        barrier_acquire_rdna(lock, lock_i);
        bool first = (lock_i == 0);
        bool last = (lock_i + lock_d == tiles_k);
        int n0 = warp_id * FRAGS_N_PER_WARP;
        
        // =================================================================
        // Bounds-checked load_accumulate for M < TILESIZE_M
        // =================================================================
        if (!sub_k && !first) {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; n++) {
                #pragma unroll
                for (int m = 0; m < FRAGS_M; m++) {
                    // Calculate valid rows for this M fragment
                    int frag_valid_rows = min(WMMA_M, max_m - m * WMMA_M);
                    if (frag_valid_rows > 0) {
                        if constexpr (c_fp32) 
                            rdna_wmma::load_accumulate_c_checked(
                                frag_c[m][n], 
                                gl_c_ptr_32 + m * WMMA_M * size_n + (n0 + n) * WMMA_N, 
                                size_n,
                                frag_valid_rows);
                        else 
                            rdna_wmma::load_accumulate_c_half_checked(
                                frag_c[m][n], 
                                gl_c_ptr_16 + m * WMMA_M * size_n + (n0 + n) * WMMA_N, 
                                size_n,
                                frag_valid_rows);
                    }
                }
            }
        }
        // =================================================================
        
        if (!sub_k) {
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; n++) {
                #pragma unroll
                for (int m = 0; m < FRAGS_M; m++) {
                    if constexpr (c_fp32) rdna_wmma::store_matrix_c_checked(gl_c_ptr_32 + m * WMMA_M * size_n + (n0 + n) * WMMA_N, frag_c[m][n], size_n, min(WMMA_M, max_m - m * WMMA_M), WMMA_N);
                    else rdna_wmma::store_matrix_c_half_checked(gl_c_ptr_16 + m * WMMA_M * size_n + (n0 + n) * WMMA_N, frag_c[m][n], size_n, min(WMMA_M, max_m - m * WMMA_M), WMMA_N);
                }
            }
        }
        barrier_release_rdna(lock, lock_d, last);
        clear_frag_c();
    };

    auto wait_stage = [&]() { mem_fence(); __syncthreads(); };

    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; i++) async_load_gl();
    wait_stage();
    clear_frag_c();
    if constexpr (FRAG_STAGES > 1) load_frags(0);

    #define FSTAGE(_load, _mul) \
        async_load_gl(); wait_stage(); load_frags(_load); matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); if (!slice2_iters) break;

    if constexpr (FRAG_STAGES == 1) { while (true) { FSTAGE(0, 0); } }
    else if constexpr (FRAG_STAGES == 2) { while (true) { FSTAGE(1, 0); FSTAGE(0, 1); } }
    else if constexpr (FRAG_STAGES == 3) { while (true) { FSTAGE(1, 0); FSTAGE(2, 1); FSTAGE(0, 2); } }
    else if constexpr (FRAG_STAGES == 4) { while (true) { FSTAGE(1, 0); FSTAGE(2, 1); FSTAGE(3, 2); FSTAGE(0, 3); } }
    else if constexpr (FRAG_STAGES == 5) { while (true) { FSTAGE(1, 0); FSTAGE(2, 1); FSTAGE(3, 2); FSTAGE(4, 3); FSTAGE(0, 4); } }
    #undef FSTAGE
}

template<EXL3_GEMM_T_ARGS_RDNA>
__device__ __forceinline__
void exl3_gemm_kernel_inner(
    const half* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C,
    const int size_m, const int size_k, const int size_n, int* __restrict__ locks)
{
    exl3_gemm_kernel_inner_rdna<bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES>
        (A, B, C, size_m, size_k, size_n, locks);
}
