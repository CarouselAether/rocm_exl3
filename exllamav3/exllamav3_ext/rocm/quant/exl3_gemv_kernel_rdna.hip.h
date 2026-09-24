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

// -----------------------------------------------------------------------------
// Slot resolution, shared by the multi-matrix kernels (exl3_mgemv_rdna.hip)
// and the multi-row path (exl3_gemv_multirow_rdna.hip)
// -----------------------------------------------------------------------------
// Returns the matrix index for slot j and, through orig_pos, the position in
// the ORIGINAL indices/weights arrays that slot j came from (equal to j when no
// packing is active; the cooperative kernel packs weights alongside indices, so
// weights are always addressed by packed slot once packing has happened --
// which for this path means "the j-th valid original position").
// Returns -1 when slot j has no matrix (fewer than j+1 indices in range).
__device__ __forceinline__ int exl3_mgemv_mat_index
(
    const int64_t* __restrict__ indices,
    int j,
    int bszm,
    int min_index,
    int max_index,
    int* orig_pos
)
{
    if (!indices)
    {
        *orig_pos = j;
        return j;
    }
    if (min_index < 0)
    {
        *orig_pos = j;
        return (int) indices[j];
    }
    int seen = 0;
    for (int i = 0; i < bszm; ++i)
    {
        int idx = (int) indices[i];
        if (idx >= min_index && idx < max_index)
        {
            if (seen == j)
            {
                *orig_pos = i;
                return idx - min_index;
            }
            seen++;
        }
    }
    return -1;
}

// Packed slot count: bszm when no packing, else the number of in-range indices
__device__ __forceinline__ int exl3_mgemv_packed_count
(
    const int64_t* __restrict__ indices,
    int bszm,
    int min_index,
    int max_index
)
{
    if (!indices || min_index < 0) return bszm;
    int seen = 0;
    for (int i = 0; i < bszm; ++i)
    {
        int idx = (int) indices[i];
        if (idx >= min_index && idx < max_index) seen++;
    }
    return seen;
}

// =============================================================================
// Launch-count fusion: rotation prologue and rotation/reduction epilogue
// =============================================================================
//
// Shared by the multi-matrix kernels (exl3_mgemv_rdna.hip, where the design
// is written up under "Launch-count fusion") and the single-matrix graph
// path (exl3_gemv_rdna.hip). Every helper reproduces the standalone
// rotation kernels' arithmetic bit for bit, so a fused launch is
// indistinguishable from the three-kernel form it replaces
// (rocm_tools/mgemv_bitwise.py is the gate).
//
// EXL3_GEMV_FUSE_OUT=0 (exl3_gemv_fuse_out_enabled, exl3_gemv_rdna.hip)
// switches both paths back to the separate output kernels.

// =============================================================================
// Input rotation, fused into the dot kernels' prologue
// =============================================================================
// Bit-for-bit the arithmetic of had_hf_r_128_inner<true, false> (the SUH-scaled
// 128-point Hadamard the standalone had_in kernels used to run), with
// two mechanical differences: the scale is indexed from the caller's pre-offset
// pointer instead of blockIdx.y (these kernels use blockIdx.y for the expert
// slot; the old kernel's grid was 1-D so its blockIdx.y term was zero), and
// the result goes to LDS. One warp rotates one 128-block; all 32 lanes enter.

__device__ __forceinline__ void exl3_gemv_had_in_128
(
    const half* __restrict__ input_ptr,   // 128 halves of A
    half* output_ptr,                     // 128 halves of LDS
    const half* __restrict__ scale,       // 128 halves of suh
    const int lane
)
{
    half4 v = ((const half4*) input_ptr)[lane];

    half4 scales = ((const half4*) scale)[lane];
    v.x = __hmul2(v.x, scales.x);
    v.y = __hmul2(v.y, scales.y);

    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;

    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    const float r_scale = 0.088388347648f;  // 1/sqrt(128)
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

    ((half4*) output_ptr)[lane] = v;
}

// Block-cooperative rotation of one expert's full input into sh_a: the block's
// warps take 128-blocks round-robin. Every thread of the block must enter; the
// caller owns the __syncthreads that publishes sh_a.
template <int WARPS_PER_BLOCK>
__device__ __forceinline__ void exl3_gemv_rotate_in
(
    const half* __restrict__ A_j,
    const half* __restrict__ suh,
    half* sh_a,
    const int size_k,
    const int warp_id,
    const int lane
)
{
    const int k_segs = size_k / 128;
    for (int seg = warp_id; seg < k_segs; seg += WARPS_PER_BLOCK)
        exl3_gemv_had_in_128(A_j + seg * 128, sh_a + seg * 128, suh + seg * 128, lane);
}

// Dynamic shared memory: the single-matrix carve (per-warp B staging, plus the
// split-K reduction slots) followed by the rotated input, size_k halves. Both
// base carves are multiples of 8 bytes, which the half4 stores need.
static inline size_t exl3_gemv_smem_bytes_fused(int warps_per_block, bool splitk, int size_k)
{
    size_t base = splitk ? exl3_gemv_smem_bytes_splitk(warps_per_block)
                         : exl3_gemv_smem_bytes(warps_per_block);
    return base + (size_t) size_k * sizeof(half);
}

// =============================================================================
// Output rotation, fused into the dot kernels' epilogue
// =============================================================================
// Bit-for-bit had_hf_r_128_inner<false, true> / had_ff_r_128_inner<false, true>
// (the SVH-scaled 128-point output Hadamard the had_out kernels run),
// in place, with the scale indexed by lane from the caller's pre-offset
// pointer. One warp rotates one 128-block; all 32 lanes enter.

__device__ __forceinline__ void exl3_gemv_had_out_128_h
(
    half* io,                             // 128 halves of C, in place
    const half* __restrict__ scale,       // 128 halves of svh
    const float r_scale,                  // 1/sqrt(128) * routing weight
    const int lane
)
{
    half4 v = ((half4*) io)[lane];

    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;

    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

    half4 scales = ((const half4*) scale)[lane];
    v.x = __hmul2(v.x, scales.x);
    v.y = __hmul2(v.y, scales.y);

    ((half4*) io)[lane] = v;
}

__device__ __forceinline__ void exl3_gemv_had_out_128_f
(
    float* io,                            // 128 floats of C, in place
    const half* __restrict__ scale,       // 128 halves of svh
    const float r_scale,                  // 1/sqrt(128) * routing weight
    const int lane
)
{
    float4 v = ((float4*) io)[lane];

    float v0 = v.x;
    float v1 = v.y;
    float v2 = v.z;
    float v3 = v.w;
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;

    shuffle_had_f2x32(v.x, v.y, lane);
    shuffle_had_f2x32(v.z, v.w, lane);
    v.x *= r_scale;
    v.y *= r_scale;
    v.z *= r_scale;
    v.w *= r_scale;

    half4 scales = ((const half4*) scale)[lane];
    v.x *= __low2float(scales.x);
    v.y *= __high2float(scales.x);
    v.z *= __low2float(scales.y);
    v.w *= __high2float(scales.y);

    ((float4*) io)[lane] = v;
}

// The fused epilogue proper (see exl3_mgemv_rdna.hip, "Launch-count fusion",
// for the design). Entered by the warp holding one N-tile's 16
// outputs in accum (lanes 0-15); all 32 lanes enter. Stores the tile, arrives
// at its segment's counter, and -- as the last of the segment's 8 tiles --
// rotates the segment in place; then, when the call has routing weights,
// arrives at the segment's reduction counter and -- as the last of the packed
// slots -- runs the grouped weighted sum for the segment's 128 columns, four
// per lane, each column in exl3_mgemv_reduce_kernel's order. Single-matrix
// callers pass red_counter = nullptr. The fences are
// wave-uniform (outside every lane predicate) so they cover all lanes' stores.

template <bool c_fp32>
__device__ __forceinline__ void exl3_gemv_fused_epilogue
(
    const float accum,
    void* Cb,                    // output base for this matrix (C, or c_list[mat])
    const int64_t row_off,       // j * size_n, or 0 for list outputs
    const int size_n,            // row stride of C (reduction only)
    const int tile_n,
    const int lane,
    const half* __restrict__ svh,
    const float scale,           // 1/sqrt(128) * routing weight
    int* seg_counter,            // this (slot, segment)'s arrival counter
    int* red_counter,            // this segment's rotated-slot counter, or nullptr
    const int red_target,        // packed slot count (read only with red_counter)
    const int num_tokens
)
{
    // 1. Store this tile, exactly as the unfused form does
    if (lane < 16)
    {
        const int64_t out_idx = row_off + tile_n * 16 + lane;
        if constexpr (c_fp32)
            ((float*) Cb)[out_idx] = accum;
        else
            ((half*) Cb)[out_idx] = __float2half(accum);
    }

    // 2. Arrive at the segment
    __threadfence();
    int old = 0;
    if (lane == 0) old = atomicAdd(seg_counter, 1);
    old = __shfl(old, 0, 32);
    if (old != 7) return;

    // 3. Last of the 8 tiles: acquire, reset, rotate in place
    __threadfence();
    if (lane == 0) *seg_counter = 0;
    const int seg = tile_n / 8;
    const int64_t seg_off = row_off + (int64_t) seg * 128;
    if constexpr (c_fp32)
        exl3_gemv_had_out_128_f(((float*) Cb) + seg_off, svh + seg * 128, scale, lane);
    else
        exl3_gemv_had_out_128_h(((half*) Cb) + seg_off, svh + seg * 128, scale, lane);

    if (!red_counter) return;

    // 4. Arrive at the segment's reduction
    __threadfence();
    if (lane == 0) old = atomicAdd(red_counter, 1);
    old = __shfl(old, 0, 32);
    if (old != red_target - 1) return;

    // 5. Last rotated slot: acquire, reset, grouped weighted sum. Four columns
    //    per lane with the row loop outside, so a row's four loads are in
    //    flight together (this runs in the kernel's tail, where latency is
    //    exposed); each column's chain is still kernel 3's exact order, and
    //    the in-place write of row t follows every read of that column for t.
    __threadfence();
    if (lane == 0) *red_counter = 0;
    const int stride = red_target / num_tokens;
    const int col0 = seg * 128 + lane;
    for (int t = 0; t < num_tokens; ++t)
    {
        if constexpr (c_fp32)
        {
            const float* C_ = ((const float*) Cb) + (int64_t) t * stride * size_n + col0;
            float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int jj = 0; jj < stride; ++jj)
            {
                #pragma unroll
                for (int c = 0; c < 4; ++c) sum[c] += C_[c * 32];
                C_ += size_n;
            }
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                ((float*) Cb)[(int64_t) t * size_n + col0 + c * 32] = sum[c];
        }
        else
        {
            const half* C_ = ((const half*) Cb) + (int64_t) t * stride * size_n + col0;
            half sum[4] = {};
            for (int jj = 0; jj < stride; ++jj)
            {
                #pragma unroll
                for (int c = 0; c < 4; ++c) sum[c] = __hadd(sum[c], C_[c * 32]);
                C_ += size_n;
            }
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                ((half*) Cb)[(int64_t) t * size_n + col0 + c * 32] = sum[c];
        }
    }
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
