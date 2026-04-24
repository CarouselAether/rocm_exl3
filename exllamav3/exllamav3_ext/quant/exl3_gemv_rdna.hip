// =============================================================================
// RDNA 3.5 GEMV - Optimized with Dynamic Wave Selection
// =============================================================================
//
// Ported verbatim from the working ROCm fork (functional on gfx1151). Only
// adaptation: uses hip_compat.cuh masquerade (at::cuda::OptionalCUDAGuard,
// cudaStream_t, at::cuda::getCurrentCUDAStream) so it fits the restart's
// single-compile convention without diverging from the rest of the tree.
//
// Wave selection (based on working-fork benchmarks):
//   N-tiles <= 32:                       16 waves (512 threads)
//   N-tiles <= 96, K-blocks <= 256:       8 waves (256 threads)
//   Otherwise:                            4 waves (128 threads)
//
// Handles:
//   - bits: 1..8 (full EXL3 range)
//   - cb:   0 (default) / 1 (mcg) / 2 (mul1)
//   - c_fp32 output
//   - SUH pre-Hadamard + SVH post-Hadamard
//
// Why this file exists: the cooperative exl3_gemm kernel uses ~64 KB LDS per
// workgroup, which stalls grid.sync() on RDNA 3/3.5 WGP-paired blocks when
// size_m == 1. Routing single-token inference here avoids the issue.
// =============================================================================

#include "../hip_compat.hip.h"
#include "exl3_gemv_rdna.hip.h"
#include "../util.h"
#include "../util.cuh"
#include "../rdna_wmma.hip"
#include "exl3_dq_rdna.hip.h"
#include "hadamard_inner.cuh"

#include <ATen/Tensor.h>

// =============================================================================
// Dot-Product GEMV Kernel - Templated on WARPS_PER_BLOCK
// =============================================================================

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK>
__global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_dot_kernel(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_k,
    const int size_n)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // Each warp handles one N-tile (16 outputs)
    const int tile_n = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int n_tiles = size_n / 16;

    if (tile_n >= n_tiles) return;

    // Dynamic shared memory - allocated based on WARPS_PER_BLOCK
    extern __shared__ char shared_mem[];

    // Layout: [WARPS_PER_BLOCK][16 * SH_STRIDE] for dequantized B
    //         [WARPS_PER_BLOCK][16 * 8] for quantized B (max 8 bits)
    //
    // SH_STRIDE=18: pads the 16-col tile to 18 halves (36 bytes = 9 DWORDs)
    // per row so that write-pattern banks spread across all 32 LDS banks.
    // Previously 17 → adjacent active-lane groups (0-3 vs 16-19, 8-11 vs
    // 24-27) collided on banks 2,3,10,11 etc, producing ~24% LDS stall time
    // per PMC. 9 is coprime with 32 so (row*9 + col/2) mod 32 gives full
    // coverage.
    constexpr int SH_STRIDE = 18;

    half* sh_b_dq = (half*)shared_mem;
    uint16_t* sh_b_quant = (uint16_t*)(sh_b_dq + WARPS_PER_BLOCK * 16 * SH_STRIDE);

    half* my_sh_b = sh_b_dq + warp_id * 16 * SH_STRIDE;
    uint16_t* my_sh_b_quant = sh_b_quant + warp_id * 16 * 8;

    // Each lane 0-15 accumulates one output element
    float accum = 0.0f;

    const int num_k_tiles = size_k / 16;
    const int tile_elements = 16 * bits;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++)
    {
        const int k_offset = k_tile * 16;

        // =====================================================================
        // Step 1: Load quantized B tile
        // =====================================================================
        const uint16_t* gl_b = B + (k_tile * n_tiles + tile_n) * tile_elements;

        #pragma unroll
        for (int i = lane; i < tile_elements; i += 32)
            my_sh_b_quant[i] = gl_b[i];

        __syncwarp();

        // =====================================================================
        // Step 2: Dequantize
        // =====================================================================
        const uint32_t* b_quant = (const uint32_t*)my_sh_b_quant;

        FragB frag0, frag1;
        dq_dispatch<bits, cb>(b_quant, lane << 3, frag0, frag1);

        // Unswizzle (shuffle by 4 within warp + combine low/high halves)
        uint32_t v0 = *reinterpret_cast<uint32_t*>(&frag0[0]);
        uint32_t v1 = *reinterpret_cast<uint32_t*>(&frag0[1]);
        uint32_t v2 = *reinterpret_cast<uint32_t*>(&frag1[0]);
        uint32_t v3 = *reinterpret_cast<uint32_t*>(&frag1[1]);

        uint32_t s0 = __shfl_down(v0, 4, 32);
        uint32_t s1 = __shfl_down(v1, 4, 32);
        uint32_t s2 = __shfl_down(v2, 4, 32);
        uint32_t s3 = __shfl_down(v3, 4, 32);

        half2 n0 = *reinterpret_cast<half2*>(&s0);
        half2 n1 = *reinterpret_cast<half2*>(&s1);
        half2 n2 = *reinterpret_cast<half2*>(&s2);
        half2 n3 = *reinterpret_cast<half2*>(&s3);

        if (!(lane & 4))
        {
            // VGPR-pressure reduction: the previous implementation built 8
            // intermediate half2's (m0..m7) and then extracted their halves
            // to store. Each __halves2half2(X, Y) followed by __low/high2half
            // just gives back X and Y, so the m_i's were round-trip no-ops
            // consuming VGPRs. Storing the halves directly removes 8 VGPRs of
            // unnecessary live state per lane. Writes are also reordered so
            // each (frag, n) pair is fully consumed before the next, giving
            // the compiler a cleaner signal about which values can die early.
            const int r0 = (lane % 4) * 2;
            const int r1 = r0 + 1;
            const int r2 = r0 + 8;
            const int r3 = r0 + 9;
            const int c0 = (lane / 8) * 2;
            const int c1 = c0 + 8;

            #define B_IDX(row, col) ((row) * SH_STRIDE + (col))

            // Group 1: frag0[0] + n0  →  (r0, c0), (r1, c0) at cols c0, c0+1
            my_sh_b[B_IDX(r0, c0)]     = __low2half (frag0[0]);
            my_sh_b[B_IDX(r0, c0 + 1)] = __low2half (n0);
            my_sh_b[B_IDX(r1, c0)]     = __high2half(frag0[0]);
            my_sh_b[B_IDX(r1, c0 + 1)] = __high2half(n0);

            // Group 2: frag0[1] + n1  →  (r2, c0), (r3, c0)
            my_sh_b[B_IDX(r2, c0)]     = __low2half (frag0[1]);
            my_sh_b[B_IDX(r2, c0 + 1)] = __low2half (n1);
            my_sh_b[B_IDX(r3, c0)]     = __high2half(frag0[1]);
            my_sh_b[B_IDX(r3, c0 + 1)] = __high2half(n1);

            // Group 3: frag1[0] + n2  →  (r0, c1), (r1, c1)
            my_sh_b[B_IDX(r0, c1)]     = __low2half (frag1[0]);
            my_sh_b[B_IDX(r0, c1 + 1)] = __low2half (n2);
            my_sh_b[B_IDX(r1, c1)]     = __high2half(frag1[0]);
            my_sh_b[B_IDX(r1, c1 + 1)] = __high2half(n2);

            // Group 4: frag1[1] + n3  →  (r2, c1), (r3, c1)
            my_sh_b[B_IDX(r2, c1)]     = __low2half (frag1[1]);
            my_sh_b[B_IDX(r2, c1 + 1)] = __low2half (n3);
            my_sh_b[B_IDX(r3, c1)]     = __high2half(frag1[1]);
            my_sh_b[B_IDX(r3, c1 + 1)] = __high2half(n3);

            #undef B_IDX
        }

        __syncwarp();

        // =====================================================================
        // Step 3: Dot product — lane L computes output column L
        //
        // Optimized with V_DOT2_F32_F16 (RDNA 3.5 ISA packed-math op,
        // VOP3P, 1 cycle): __builtin_amdgcn_fdot2(a, b, c, clamp) computes
        //   c + a.x * b.x + a.y * b.y   (all fp16 inputs, fp32 accumulate)
        // This halves the instruction count of the inner loop: 8 dot2 ops
        // instead of 16 f16→f32 converts + 16 FMAs per lane per k-tile.
        //
        // A is contiguous in k → single half2 aligned load per pair.
        // B is strided (rows 18 halves apart in LDS) → pack two scalar LDS
        // reads into a half2 manually. All lanes read A[k_offset + k] for
        // the SAME k — the compiler lifts that to a scalar broadcast load,
        // which is materially faster than a per-lane vector load. Splitting
        // K across lanes 0-15/16-31 breaks this broadcast and regresses
        // kernel time, so we stay with the 16-active-lane design.
        // =====================================================================
        if (lane < 16)
        {
            #pragma unroll
            for (int k = 0; k < 16; k += 2)
            {
                half2 a2 = *reinterpret_cast<const half2*>(&A[k_offset + k]);
                half2 b2 = __halves2half2(
                    my_sh_b[k * SH_STRIDE + lane],
                    my_sh_b[(k + 1) * SH_STRIDE + lane]
                );
                accum = __builtin_amdgcn_fdot2(a2, b2, accum, false);
            }
        }

        __syncwarp();
    }

    // =========================================================================
    // Step 4: Write output
    // =========================================================================
    if (lane < 16)
    {
        const int out_idx = tile_n * 16 + lane;
        if constexpr (c_fp32)
            ((float*)C)[out_idx] = accum;
        else
            ((half*)C)[out_idx] = __float2half(accum);
    }
}

// =============================================================================
// Hadamard helper kernels (size_m == 1 forms)
// =============================================================================

__global__
__launch_bounds__(256)
void rdna_gemv_hadamard_in_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ scales,
    const int size)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int total_warps = size / 128;

    if (warp_id < total_warps)
    {
        int offset = warp_id * 128;
        had_hf_r_128_inner(
            input + offset,
            output + offset,
            scales + offset,
            nullptr,
            0.088388347648f);
    }
}

__global__
__launch_bounds__(256)
void rdna_gemv_hadamard_out_half_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ scales,
    const int size)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int total_warps = size / 128;

    if (warp_id < total_warps)
    {
        int offset = warp_id * 128;
        had_hf_r_128_inner(
            input + offset,
            output + offset,
            nullptr,
            scales + offset,
            0.088388347648f);
    }
}

__global__
__launch_bounds__(256)
void rdna_gemv_hadamard_out_float_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const half* __restrict__ scales,
    const int size)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int total_warps = size / 128;

    if (warp_id < total_warps)
    {
        int offset = warp_id * 128;
        had_ff_r_128_inner(
            input + offset,
            output + offset,
            nullptr,
            scales + offset,
            0.088388347648f);
    }
}

// =============================================================================
// Dispatch — bits × cb × c_fp32 × dynamic wave count
// =============================================================================

void exl3_gemv_rdna(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    bool mcg,
    bool mul1)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);

    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);

    int size_m = 1;
    for (int d = 0; d < A.dim() - 1; ++d)
        size_m *= A.size(d);

    int size_k = A.size(A.dim() - 1);
    int size_n = B.size(1) * 16;
    int bits = B.size(2) / 16;

    TORCH_CHECK(size_m == 1, "exl3_gemv_rdna requires size_m == 1, got ", size_m);
    TORCH_CHECK(bits >= 1 && bits <= 8, "bits must be 1-8, got ", bits);
    TORCH_CHECK(size_k % 128 == 0, "size_k must be multiple of 128, got ", size_k);
    TORCH_CHECK(size_n % 128 == 0, "size_n must be multiple of 128, got ", size_n);

    int cb = mcg ? 1 : (mul1 ? 2 : 0);

    const half* A_ptr = (const half*)A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*)B.data_ptr();
    void* C_ptr = (void*)C.data_ptr();

    // =========================================================================
    // Input Hadamard Transform (SUH)
    // =========================================================================
    const half* gemv_input = A_ptr;

    if (suh.has_value() && A_had.has_value())
    {
        const half* suh_ptr = (const half*)suh.value().data_ptr();
        half* A_had_ptr = (half*)A_had.value().data_ptr();

        int warps_needed = size_k / 128;
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / 32;
        int blocks = (warps_needed + warps_per_block - 1) / warps_per_block;

        hipLaunchKernelGGL(
            rdna_gemv_hadamard_in_kernel,
            dim3(blocks), dim3(threads_per_block), 0, stream,
            A_ptr, A_had_ptr, suh_ptr, size_k);

        gemv_input = A_had_ptr;
    }

    // =========================================================================
    // Dynamic Wave Selection based on benchmarks in working fork
    // =========================================================================
    int n_tiles = size_n / 16;
    int k_blocks = size_k / 16;

    int warps_per_block;
    if (n_tiles <= 32)
        warps_per_block = 16;
    else if (n_tiles <= 96 && k_blocks <= 256)
        warps_per_block = 8;
    else
        warps_per_block = 4;

    int num_blocks = (n_tiles + warps_per_block - 1) / warps_per_block;

    // Shared memory: dequantized B (padded stride 18, conflict-free per PMC
    // analysis — see SH_STRIDE comment in the kernel) + quantized B per warp
    size_t shared_mem_size =
        warps_per_block * (16 * 18 * sizeof(half) + 16 * 8 * sizeof(uint16_t));

    // =========================================================================
    // Kernel Launch Macros
    // =========================================================================
    #define LAUNCH_GEMV_WAVES(bits_val, codebook, warps) \
        if (c_fp32) { \
            hipLaunchKernelGGL( \
                (exl3_gemv_dot_kernel<bits_val, true, codebook, warps>), \
                dim3(num_blocks), dim3(warps * 32), shared_mem_size, stream, \
                gemv_input, B_ptr, C_ptr, size_k, size_n); \
        } else { \
            hipLaunchKernelGGL( \
                (exl3_gemv_dot_kernel<bits_val, false, codebook, warps>), \
                dim3(num_blocks), dim3(warps * 32), shared_mem_size, stream, \
                gemv_input, B_ptr, C_ptr, size_k, size_n); \
        }

    #define LAUNCH_GEMV_CB(bits_val, codebook) \
        switch (warps_per_block) { \
            case 4:  LAUNCH_GEMV_WAVES(bits_val, codebook, 4);  break; \
            case 8:  LAUNCH_GEMV_WAVES(bits_val, codebook, 8);  break; \
            case 16: LAUNCH_GEMV_WAVES(bits_val, codebook, 16); break; \
        }

    #define LAUNCH_GEMV(bits_val) \
        switch (cb) { \
            case 0: LAUNCH_GEMV_CB(bits_val, 0); break; \
            case 1: LAUNCH_GEMV_CB(bits_val, 1); break; \
            case 2: LAUNCH_GEMV_CB(bits_val, 2); break; \
        }

    switch (bits) {
        case 1: LAUNCH_GEMV(1); break;
        case 2: LAUNCH_GEMV(2); break;
        case 3: LAUNCH_GEMV(3); break;
        case 4: LAUNCH_GEMV(4); break;
        case 5: LAUNCH_GEMV(5); break;
        case 6: LAUNCH_GEMV(6); break;
        case 7: LAUNCH_GEMV(7); break;
        case 8: LAUNCH_GEMV(8); break;
    }

    #undef LAUNCH_GEMV
    #undef LAUNCH_GEMV_CB
    #undef LAUNCH_GEMV_WAVES

    // =========================================================================
    // Output Hadamard Transform (SVH)
    // =========================================================================
    if (svh.has_value())
    {
        const half* svh_ptr = (const half*)svh.value().data_ptr();

        int warps_needed = size_n / 128;
        int threads_per_block = 256;
        int warps_per_block_had = threads_per_block / 32;
        int blocks = (warps_needed + warps_per_block_had - 1) / warps_per_block_had;

        if (c_fp32) {
            hipLaunchKernelGGL(
                rdna_gemv_hadamard_out_float_kernel,
                dim3(blocks), dim3(threads_per_block), 0, stream,
                (const float*)C_ptr, (float*)C_ptr, svh_ptr, size_n);
        } else {
            hipLaunchKernelGGL(
                rdna_gemv_hadamard_out_half_kernel,
                dim3(blocks), dim3(threads_per_block), 0, stream,
                (const half*)C_ptr, (half*)C_ptr, svh_ptr, size_n);
        }
    }

    cuda_check(cudaPeekAtLastError());
}
