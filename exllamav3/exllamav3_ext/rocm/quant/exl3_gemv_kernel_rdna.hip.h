#pragma once

// =============================================================================
// RDNA 3.5 GEMV kernel -- fdot2 dot-product form
// =============================================================================
//
// Carried forward from the working ROCm fork
// (rocm_exl3_legacy/.../quant/exl3_gemv_rdna.hip, where kernel and host dispatch
// lived in one file). The kernel body is unchanged apart from the hadamard
// helpers, which are re-pointed at upstream v1.3.0's templated
// had_*_r_128_inner signature (see below). Every comment about measured
// behaviour is from the fork and is not re-derivable from upstream.
//
// This is NOT a port of upstream's quant/exl3_gemv_kernel.cuh. That kernel is
// built on m16n8k16 mma.sync (locally defined as mma_ab_h, which is why grepping
// for ptx.cuh's named wrappers missed it) plus cp_async and a cooperative
// grid.sync. It has no RDNA equivalent, and the mechanical include-swapped copy
// that used to sit at this path did not compile.
//
// The fork also carries a WMMA GEMV at quant/exl3_gemv_kernel_rdna.hip.h (three
// kernels: a split-K form with atomics whose fp16 store its own comments mark as
// wrong, a k % 256 single-pass form, and exl3_gemv_kernel<bits, c_fp32, cb,
// k_split>). It is **superseded as the m == 1 path**: only quant/exl3_gemv.cu
// includes it, that entry point is a hardcoded bits=4 / cb=0 / K_SPLIT=1 stub,
// and nothing on the ROCm path ever called it -- exl3_gemv_rdna handled every
// m == 1 matmul.
//
// It is not worthless, though, and it is the reason this file does not claim
// m > 1. The third kernel pads M to 16 and would cover 2 <= m <= 8, which is
// upstream's envelope and which this kernel cannot reach. It has never been
// numerically validated on this hardware and it stages through a stride-17 LDS
// tile (see SH_STRIDE below for why that is the wrong stride). Evaluating it
// needs its own correctness harness; until then m > 1 falls through to the
// cooperative GEMM.
//
// Shape: one warp per 16-wide output tile, 16 active lanes each accumulating one
// output element in fp32. B is staged quantized through LDS, dequantized in
// registers, unswizzled to row-major in LDS, then consumed by V_DOT2_F32_F16.
//
// Handles bits 1-8, cb 0 (default) / 1 (mcg) / 2 (mul1), fp16 or fp32 C, and
// m == 1 only. Larger m falls through to the cooperative GEMM.
// =============================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "exl3_dq_rdna.hip.h"
#include "../../quant/hadamard_inner.cuh"

// -----------------------------------------------------------------------------
// LDS geometry -- ONE definition, used by the kernel and by the host launch
// -----------------------------------------------------------------------------
// SH_STRIDE = 18: pads the 16-column tile to 18 halves (36 bytes = 9 dwords) per
// row so the write pattern spreads across all 32 LDS banks. At 17 the adjacent
// active-lane groups (0-3 vs 16-19, 8-11 vs 24-27) collide on banks 2,3,10,11
// etc, measured at ~24% LDS stall time by PMC. 9 is coprime with 32, so
// (row * 9 + col / 2) mod 32 covers every bank.
//
// The quantized staging area is sized for the widest bitwidth (8) rather than
// per-instantiation, so one host-side figure covers every kernel.
//
// Keep the host size and the kernel's indexing derived from these constants and
// nothing else. An LDS figure that drifted out of step in the GEMM presented as
// a kernel bug, not as a launch bug, and cost real time.
#define EXL3_GEMV_SH_STRIDE 18
#define EXL3_GEMV_SH_QUANT_U16 (16 * 8)

static inline size_t exl3_gemv_smem_bytes(int warps_per_block)
{
    return (size_t) warps_per_block *
           (16 * EXL3_GEMV_SH_STRIDE * sizeof(half) +
            EXL3_GEMV_SH_QUANT_U16 * sizeof(uint16_t));
}

// =============================================================================
// Dot-product tile loop -- shared between the single-matrix GEMV kernel below
// and the multi-matrix (expert-batched) GEMV in exl3_mgemv_rdna.hip
// =============================================================================
//
// Extracted verbatim from exl3_gemv_dot_kernel; the kernel wrappers own the
// grid/tile mapping and the output store, this owns everything per-warp. All 32
// lanes must enter (the unswizzle round trip uses the full warp); the return
// value is the accumulated dot product, meaningful for lanes 0-15 only.

template <int bits, int cb>
__device__ __forceinline__ float exl3_gemv_dot_tile
(
    const half* __restrict__ A,        // rotated input, [size_k]
    const uint16_t* __restrict__ B,    // quantized trellis for one matrix
    const int size_k,
    const int n_tiles,                 // size_n / 16 for THIS matrix
    const int tile_n,                  // this warp's N-tile
    const int lane,
    half* my_sh_b,                     // per-warp staging, 16 * EXL3_GEMV_SH_STRIDE halves
    uint16_t* my_sh_b_quant,           // per-warp staging, EXL3_GEMV_SH_QUANT_U16 u16
    const int kb_begin,                // k16-tile range for THIS warp; (0, size_k/16)
    const int kb_end                   //   for the whole-K single-warp form
)
{
    constexpr int SH_STRIDE = EXL3_GEMV_SH_STRIDE;

    // Each lane 0-15 accumulates one output element
    float accum = 0.0f;

    const int tile_elements = 16 * bits;

    for (int k_tile = kb_begin; k_tile < kb_end; k_tile++)
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
        const uint32_t* b_quant = (const uint32_t*) my_sh_b_quant;

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

            // Group 1: frag0[0] + n0  ->  (r0, c0), (r1, c0) at cols c0, c0+1
            my_sh_b[B_IDX(r0, c0)]     = __low2half (frag0[0]);
            my_sh_b[B_IDX(r0, c0 + 1)] = __low2half (n0);
            my_sh_b[B_IDX(r1, c0)]     = __high2half(frag0[0]);
            my_sh_b[B_IDX(r1, c0 + 1)] = __high2half(n0);

            // Group 2: frag0[1] + n1  ->  (r2, c0), (r3, c0)
            my_sh_b[B_IDX(r2, c0)]     = __low2half (frag0[1]);
            my_sh_b[B_IDX(r2, c0 + 1)] = __low2half (n1);
            my_sh_b[B_IDX(r3, c0)]     = __high2half(frag0[1]);
            my_sh_b[B_IDX(r3, c0 + 1)] = __high2half(n1);

            // Group 3: frag1[0] + n2  ->  (r0, c1), (r1, c1)
            my_sh_b[B_IDX(r0, c1)]     = __low2half (frag1[0]);
            my_sh_b[B_IDX(r0, c1 + 1)] = __low2half (n2);
            my_sh_b[B_IDX(r1, c1)]     = __high2half(frag1[0]);
            my_sh_b[B_IDX(r1, c1 + 1)] = __high2half(n2);

            // Group 4: frag1[1] + n3  ->  (r2, c1), (r3, c1)
            my_sh_b[B_IDX(r2, c1)]     = __low2half (frag1[1]);
            my_sh_b[B_IDX(r2, c1 + 1)] = __low2half (n3);
            my_sh_b[B_IDX(r3, c1)]     = __high2half(frag1[1]);
            my_sh_b[B_IDX(r3, c1 + 1)] = __high2half(n3);

            #undef B_IDX
        }

        __syncwarp();

        // =====================================================================
        // Step 3: Dot product -- lane L computes output column L
        //
        // Optimized with V_DOT2_F32_F16 (RDNA 3.5 ISA packed-math op,
        // VOP3P, 1 cycle): __builtin_amdgcn_fdot2(a, b, c, clamp) computes
        //   c + a.x * b.x + a.y * b.y   (all fp16 inputs, fp32 accumulate)
        // This halves the instruction count of the inner loop: 8 dot2 ops
        // instead of 16 f16->f32 converts + 16 FMAs per lane per k-tile.
        //
        // A is contiguous in k -> single half2 aligned load per pair.
        // B is strided (rows 18 halves apart in LDS) -> pack two scalar LDS
        // reads into a half2 manually. All lanes read A[k_offset + k] for
        // the SAME k -- the compiler lifts that to a scalar broadcast load,
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

    return accum;
}

// =============================================================================
// Barrier-free dot-tile core -- accumulates in dq's native fragment layout
// =============================================================================
//
// The LDS core above spends most of each k-iteration on the unswizzle round
// trip: stage quantized -> dq -> __shfl_down -> LDS scatter -> __syncwarp ->
// LDS gather -> dot, with lanes 16-31 idle in the dot phase. This core deletes
// all of it by never leaving dq's output layout. Derivation, re-verified
// against the unswizzle above (and matching the workspace m1_dot sketch):
//
//   dq_dispatch<bits,cb>(b_tile, L << 3, frag0, frag1) hands lane L the 16x16
//   (K x N) tile elements
//     frag0[0] = ( B[r0  ][cA], B[r0+1][cA] )      r0 = (L % 4) * 2
//     frag0[1] = ( B[r0+8][cA], B[r0+9][cA] )      cA = (L / 8) * 2 + ((L >> 2) & 1)
//     frag1[0] = ( B[r0  ][cB], B[r0+1][cB] )      cB = cA + 8
//     frag1[1] = ( B[r0+8][cB], B[r0+9][cB] )
//
//   (The unswizzle writes lane L's frag0[0] to column (L/8)*2 when L&4 == 0 and
//   routes it through __shfl_down(,4) to column (L/8)*2+1 when L&4 == 4 --
//   i.e. source lane S holds column (S/8)*2 + ((S>>2)&1). Rows follow (S%4)*2
//   because S%4 == (S+4)%4.)
//
//   So column cA is held by exactly the lane quad {4*cA .. 4*cA+3}, whose four
//   lanes cover rows {0..7} x {+0,+8} between them, and every lane does useful
//   work. Two fdot2 per fragment pair against A rows (r0, r0+1) and (r0+8,
//   r0+9), a 2-hop __shfl_xor quad reduction ONCE at the end of the k-range
//   (not per tile), and a broadcast remap so lanes 0-15 return columns 0-15 --
//   the same contract as the LDS core, so every kernel wrapper takes either.
//
// B is read directly from global: a tile is 32*bits contiguous bytes, the warp
// collectively touches every byte exactly once, and L0 serves the overlapping
// lane reads. A is read per-lane (two half2 loads); quads repeat the same 32
// bytes and hit L0. No LDS, no barriers, no idle lanes.
//
// Selected at runtime by the kernels' trailing lds_core argument (see
// exl3_gemv_lds_core() -- EXL3_GEMV_LDS=1 pins the LDS core). All 32 lanes
// must enter (the reduction shuffles use the full warp).

template <int bits, int cb>
__device__ __forceinline__ float exl3_gemv_dot_tile_direct
(
    const half* __restrict__ A,        // rotated input, [size_k]
    const uint16_t* __restrict__ B,    // quantized trellis for one matrix
    const int n_tiles,                 // size_n / 16 for THIS matrix
    const int tile_n,                  // this warp's N-tile
    const int lane,
    const int kb_begin,                // k16-tile range for THIS warp
    const int kb_end
)
{
    constexpr int tile_elements = 16 * bits;

    const int r0 = (lane & 3) * 2;

    float accA = 0.0f;                 // column cA
    float accB = 0.0f;                 // column cB = cA + 8

    for (int k_tile = kb_begin; k_tile < kb_end; k_tile++)
    {
        const uint32_t* b_ptr = (const uint32_t*)
            (B + (k_tile * n_tiles + tile_n) * tile_elements);

        FragB frag0, frag1;
        dq_dispatch<bits, cb>(b_ptr, lane << 3, frag0, frag1);

        const half2* a2 = (const half2*) (A + k_tile * 16);
        half2 a01 = a2[r0 >> 1];             // (A[r0],   A[r0+1])
        half2 a89 = a2[(r0 >> 1) + 4];       // (A[r0+8], A[r0+9])

        accA = __builtin_amdgcn_fdot2(a01, frag0[0], accA, false);
        accA = __builtin_amdgcn_fdot2(a89, frag0[1], accA, false);
        accB = __builtin_amdgcn_fdot2(a01, frag1[0], accB, false);
        accB = __builtin_amdgcn_fdot2(a89, frag1[1], accB, false);
    }

    // Quad reduction: after two xor hops every lane of quad g holds the full
    // sum for columns g (accA) and g+8 (accB)
    accA += __shfl_xor(accA, 1, 32);
    accA += __shfl_xor(accA, 2, 32);
    accB += __shfl_xor(accB, 1, 32);
    accB += __shfl_xor(accB, 2, 32);

    // Remap to the LDS core's contract: lane l (0-15) returns column l.
    // Column c < 8 lives in quad 4c (accA); column c >= 8 in quad 4*(c-8)
    // (accB). Lanes 16-31 return a defined but meaningless value, as before.
    float vA = __shfl(accA, (lane & 7) * 4, 32);
    float vB = __shfl(accB, (lane & 7) * 4, 32);
    return (lane & 8) ? vB : vA;
}

// Defined in exl3_gemv_rdna.hip; EXL3_GEMV_LDS=1 pins the LDS core in every
// GEMV form (the A/B and kill switch for the barrier-free core). Re-read per
// call. Kernels take the result as their trailing lds_core argument -- runtime
// rather than a template split so the instantiation count stays put; the LDS
// high-water mark is unchanged (smem is passed identically in both modes) and
// LDS was never the occupancy limiter for these kernels.
bool exl3_gemv_lds_core();

// =============================================================================
// Dot-product GEMV kernel -- templated on WARPS_PER_BLOCK
// =============================================================================

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK>
__global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_dot_kernel
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_k,
    const int size_n,
    const bool lds_core
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // Each warp handles one N-tile (16 outputs)
    const int tile_n = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int n_tiles = size_n / 16;

    if (tile_n >= n_tiles) return;

    // Dynamic shared memory -- layout mirrors exl3_gemv_smem_bytes() above
    extern __shared__ char shared_mem[];

    constexpr int SH_STRIDE = EXL3_GEMV_SH_STRIDE;

    half* sh_b_dq = (half*) shared_mem;
    uint16_t* sh_b_quant = (uint16_t*) (sh_b_dq + WARPS_PER_BLOCK * 16 * SH_STRIDE);

    half* my_sh_b = sh_b_dq + warp_id * 16 * SH_STRIDE;
    uint16_t* my_sh_b_quant = sh_b_quant + warp_id * EXL3_GEMV_SH_QUANT_U16;

    float accum = lds_core
        ? exl3_gemv_dot_tile<bits, cb>
          (
              A, B, size_k, n_tiles, tile_n, lane, my_sh_b, my_sh_b_quant,
              0, size_k / 16
          )
        : exl3_gemv_dot_tile_direct<bits, cb>
          (
              A, B, n_tiles, tile_n, lane, 0, size_k / 16
          );

    // =========================================================================
    // Step 4: Write output
    // =========================================================================
    if (lane < 16)
    {
        const int out_idx = tile_n * 16 + lane;
        if constexpr (c_fp32)
            ((float*) C)[out_idx] = accum;
        else
            ((half*) C)[out_idx] = __float2half(accum);
    }
}

// =============================================================================
// In-block split-K form -- one block per N-tile, warps share the K range
// =============================================================================
//
// The single-warp form above gives a matmul only size_n/16 warps of
// parallelism, which starves narrow outputs: 3072->1024 is 64 warps in 8
// blocks on a part with 80 SIMDs. llama.cpp's mmvq solves the same problem on
// this hardware with one wave per output row and lanes splitting K; the EXL3
// trellis decodes in 16x16 tiles so one row per wave is off the table, but the
// K split transplants: all WARPS_PER_BLOCK warps of a block work the SAME
// N-tile on disjoint contiguous k16 ranges, then reduce through 16 floats of
// LDS per warp. Total waves multiply by WARPS_PER_BLOCK with zero extra
// global traffic (the ranges are disjoint), no atomics, no workspace, no
// cooperative launch. Wide outputs that already saturate the device keep the
// single-warp form -- see EXL3_GEMV_SPLITK_MAX_TILES at the launch sites.
//
// Per-warp staging (my_sh_b / my_sh_b_quant) is the same carve as the
// single-warp form; sh_red is WARPS_PER_BLOCK * 16 floats appended after it
// (exl3_gemv_smem_bytes_splitk). Every thread of the block must enter (the
// reduction has a __syncthreads); the return value is the full dot product,
// meaningful for warp 0 lanes 0-15 only.

static inline size_t exl3_gemv_smem_bytes_splitk(int warps_per_block)
{
    return exl3_gemv_smem_bytes(warps_per_block) +
           (size_t) warps_per_block * 16 * sizeof(float);
}

template <int bits, int cb, int WARPS_PER_BLOCK>
__device__ __forceinline__ float exl3_gemv_dot_tile_splitk
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    const int size_k,
    const int n_tiles,
    const int tile_n,
    const int warp_id,
    const int lane,
    half* my_sh_b,
    uint16_t* my_sh_b_quant,
    float* sh_red,                     // WARPS_PER_BLOCK * 16 floats
    const bool lds_core
)
{
    const int num_k_tiles = size_k / 16;
    const int chunk = (num_k_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int kb0 = warp_id * chunk;
    const int kb1 = kb0 + chunk < num_k_tiles ? kb0 + chunk : num_k_tiles;

    float accum = 0.0f;
    if (kb0 < kb1)
        accum = lds_core
            ? exl3_gemv_dot_tile<bits, cb>
              (
                  A, B, size_k, n_tiles, tile_n, lane, my_sh_b, my_sh_b_quant,
                  kb0, kb1
              )
            : exl3_gemv_dot_tile_direct<bits, cb>
              (
                  A, B, n_tiles, tile_n, lane, kb0, kb1
              );

    if (lane < 16) sh_red[warp_id * 16 + lane] = accum;
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0 && lane < 16)
    {
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w)
            total += sh_red[w * 16 + lane];
    }
    return total;
}

// Above this many N-tiles the single-warp form is kept; below it, split-K
// multiplies the wave count by WARPS_PER_BLOCK. 512 was tuned against the LDS
// core; the barrier-free core shifted the balance — DRAM-resident sweep
// (gemv_check GEMV_SWEEP=1, 2026-08-08): split-K wins at 1344 tiles by +29%
// (208 vs 162 GB/s on 5376->21504) and only reaches parity at 16384 tiles
// (lm_head, 227 GB/s both forms). 2048 flips every in-model projection to
// split-K and leaves lm_head-scale outputs on the single-warp form, which
// avoids 16k-block grids for no measured cost. Re-sweep if the core changes.
#define EXL3_GEMV_SPLITK_MAX_TILES 2048

// The split-K wave count is no longer a fixed constant: all three split-K
// sites call exl3_gemv_splitk_warps() (exl3_gemv_rdna.hip), which picks 4, 8
// or 16 from (k_tiles, blocks = n_tiles x bszm) per the 2026-08-13 sweep, and
// EXL3_GEMV_SPLITK_WARPS (env) forces one count everywhere for model-level A/B.

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK>
__global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_dot_kernel_splitk
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_k,
    const int size_n,
    const bool lds_core
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // One block per N-tile; the block's warps split K
    const int tile_n = blockIdx.x;
    const int n_tiles = size_n / 16;

    extern __shared__ char shared_mem[];
    constexpr int SH_STRIDE = EXL3_GEMV_SH_STRIDE;
    half* sh_b_dq = (half*) shared_mem;
    uint16_t* sh_b_quant = (uint16_t*) (sh_b_dq + WARPS_PER_BLOCK * 16 * SH_STRIDE);
    float* sh_red = (float*) (sh_b_quant + WARPS_PER_BLOCK * EXL3_GEMV_SH_QUANT_U16);
    half* my_sh_b = sh_b_dq + warp_id * 16 * SH_STRIDE;
    uint16_t* my_sh_b_quant = sh_b_quant + warp_id * EXL3_GEMV_SH_QUANT_U16;

    float accum = exl3_gemv_dot_tile_splitk<bits, cb, WARPS_PER_BLOCK>
    (
        A, B, size_k, n_tiles, tile_n, warp_id, lane, my_sh_b, my_sh_b_quant,
        sh_red, lds_core
    );

    if (warp_id == 0 && lane < 16)
    {
        const int out_idx = tile_n * 16 + lane;
        if constexpr (c_fp32)
            ((float*) C)[out_idx] = accum;
        else
            ((half*) C)[out_idx] = __float2half(accum);
    }
}

// =============================================================================
// Hadamard helper kernels (m == 1 forms)
// =============================================================================
//
// The cooperative GEMM folds SUH/SVH into the matmul kernel around grid.sync().
// This path is a plain launch, so the transforms are separate kernels either
// side of it -- as in the fork.
//
// Upstream v1.3.0 changed the inner helpers' signature: pre/post scaling is now
// a template pair with a single scale pointer, where v0.0.29 (what the fork was
// written against) took both scale pointers as runtime args. The calls below are
// updated accordingly; the arithmetic is identical.
//
// Note the helpers index the scale array as ((half4*) scale)[blockIdx.y * 32 + t].
// These grids are 1-D, so blockIdx.y == 0 and the per-warp offset has to be
// folded into the pointer -- which is what upstream's own GEMM call sites do
// (suh + (this_warp * 128) % size_k).

// static: this header is included by two RDC TUs since the multi-matrix GEMV
// landed (exl3_gemv_rdna.hip and exl3_mgemv_rdna.hip); non-template __global__
// definitions would collide at device link. Each TU owning a private copy is
// harmless -- only exl3_gemv_rdna.hip launches these three.
static __global__
__launch_bounds__(256)
void exl3_gemv_rdna_had_in_kernel
(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ scales,
    const int size
)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int total_warps = size / 128;

    if (warp_id < total_warps)
    {
        int offset = warp_id * 128;
        had_hf_r_128_inner<true, false>
        (
            input + offset,
            output + offset,
            scales + offset,
            0.088388347648f  // 1/sqrt(128)
        );
    }
}

static __global__
__launch_bounds__(256)
void exl3_gemv_rdna_had_out_half_kernel
(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ scales,
    const int size
)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int total_warps = size / 128;

    if (warp_id < total_warps)
    {
        int offset = warp_id * 128;
        had_hf_r_128_inner<false, true>
        (
            input + offset,
            output + offset,
            scales + offset,
            0.088388347648f  // 1/sqrt(128)
        );
    }
}

static __global__
__launch_bounds__(256)
void exl3_gemv_rdna_had_out_float_kernel
(
    const float* __restrict__ input,
    float* __restrict__ output,
    const half* __restrict__ scales,
    const int size
)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int total_warps = size / 128;

    if (warp_id < total_warps)
    {
        int offset = warp_id * 128;
        had_ff_r_128_inner<false, true>
        (
            input + offset,
            output + offset,
            scales + offset,
            0.088388347648f  // 1/sqrt(128)
        );
    }
}
