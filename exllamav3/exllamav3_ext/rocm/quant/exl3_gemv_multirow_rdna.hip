// =============================================================================
// Multi-row GEMV for RDNA, m = 2..8
// =============================================================================
//
// The GEMV fast paths take m == 1 only; at m = 2..8 -- speculative-decoding
// verify steps (MTP, DFlash, n-gram drafts) -- every dense linear used to fall
// to the cooperative tile GEMM, which pads the rows to a 16-row tile and walks
// the weights at a fraction of the GEMV's bandwidth. Measured 2026-09-23
// (profile_decode -ndt 2): the dense linears of an m = 3 verify step cost 4x
// (Qwen3.8-Flash-Next) / 4.4x (DS4) their m = 1 cost and were the largest
// kernel class of the step, which is why MTP was a net loss on both models
// (RDNA_NOTES, "Where the m = 3 verify step goes").
//
// This path reads the weights once for up to 8 rows. The dot core is the
// barrier-free m == 1 core (exl3_gemv_dot_tile_direct) with a row tile M in
// {2, 4, 8} chosen as the smallest >= m: one B dequant per k-tile feeds M
// pairs of fdot2 accumulators, and each row's chain is exactly the m == 1
// chain, so row r of an m-row call is bit-identical to a separate m == 1 call
// on that row (rocm_tools/multirow_check.py is the gate). Rows past m are
// computed on row m - 1's data and never stored.
//
// Structure is the pre-fusion three-kernel form, deliberately. The fused
// m == 1 kernels rotate the input into LDS in their prologue, which does not
// extend to m rows on a 64 KB-per-workgroup part (8 x 12288 halves is 196 KB,
// and the rotation is redundant across a matrix's N-tile blocks). So:
//
//   1. exl3_gemv_mr_had_in_{single,multi}   rotates the m rows of every slab
//      into A_had (global; L2-resident thereafter) and hosts the graph patch
//      sites, republishing the patchable pointers the dot kernel needs through
//      a per-device Exl3GemvMrParams block (the pattern of exl3_mgemv_rdna.hip)
//   2. exl3_gemv_mr_dot_{single,multi}      split-K GEMV, one block per
//      (slot, N-tile), M rows per warp; the fused output epilogue (last-
//      arriving warp per 128-wide segment, exl3_gemv_kernel_rdna.hip.h)
//      rotates the m row segments in place
//
// Two launches per call at m > 1 is irrelevant: the verify step is GPU-bound
// at 100+ ms. A part with more LDS could rotate the rows in-kernel (the
// helpers are shared), gated on a runtime LDS budget; not done here.
//
// Routing: exl3_gemm_gr and exl3_mgemm_gr call the two try_launch entry
// points for 2 <= m <= EXL3_GEMV_MAX_M (default 8; 1 switches the path off)
// before the cooperative kernels, in and out of graph capture. The
// multi-matrix form declines routing weights (the weighted reduce is the
// per-token MoE route's, always m == 1) and sliced mode.
// =============================================================================

#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../../util.h"
#include "../../util.cuh"
#include "../../graph.cuh"
#include "exl3_mgemv_rdna.hip.h"
#include "exl3_gemv_kernel_rdna.hip.h"
#include "exl3_gemv_multirow_rdna.hip.h"

#include <cstdlib>
#include <mutex>

int exl3_gemv_max_m()
{
    const char* env = std::getenv("EXL3_GEMV_MAX_M");
    if (!env) return 8;
    int v = atoi(env);
    return v < 1 ? 1 : (v > 8 ? 8 : v);
}

// m == 1 goes through this structure too, with a row tile of 1 -- rows read
// from L2, no rotated row in LDS -- ahead of the fused m == 1 kernels
// (EXL3_GEMV_MR_M1=0 restores those). Measured 2026-09-23: at m = 3 this
// kernel ran below the fused m == 1 kernel per call on the same shapes (71
// vs 78 us), which pointed at the LDS prologue's occupancy cost, and the A/B
// confirmed it -- Laguna's gate/up kernel 343 -> 319 ms per 32 tokens, total
// kernel time back to the pre-fusion figure, decode Laguna 23.5 -> 23.6,
// Qwen3.8 23.8 -> 24.4, DS4 16.2 -> 17.0 t/s, bit-identical logits
// (decode_bitwise). Two launches instead of one, but the launch was never
// the cost (RDNA_NOTES, "Launch-count fusion: landed"). The fused kernels
// keep the weighted-reduce MoE down projection (this path declines weights)
// and lm_head-scale widths (the m == 1 single-warp form; leaving them there
// keeps the two routes bit-identical where both split K).
static bool exl3_gemv_mr_m1_enabled()
{
    const char* env = std::getenv("EXL3_GEMV_MR_M1");
    if (!env) return true;
    return atoi(env) != 0;
}

static inline int exl3_gemv_mr_min_m(int n_tiles)
{
    return (exl3_gemv_mr_m1_enabled() && n_tiles <= EXL3_GEMV_SPLITK_MAX_TILES) ? 1 : 2;
}

// -----------------------------------------------------------------------------
// Per-device parameter block: the patchable pointers the dot kernels read
// (republished by the rotation kernels, which host the graph sites) and the
// fused epilogue's arrival counters, [slot][segment], self-resetting.
// -----------------------------------------------------------------------------
struct Exl3GemvMrParams
{
    const uint16_t* B;      // single-matrix
    void* C;
    half* A_had;
    const half* svh;
    void* Cm;               // multi-matrix
    const int64_t* indices;
    const half* weights;
    int seg_counters[EXL3_MGEMV_SEG_COUNTERS];
};

static Exl3GemvMrParams* exl3_gemv_mr_param_block(int device, bool allow_alloc)
{
    static std::mutex mtx;
    static Exl3GemvMrParams* blocks[64] = {};
    if (device < 0 || device >= 64) return nullptr;
    std::lock_guard<std::mutex> lock(mtx);
    if (!blocks[device] && allow_alloc)
    {
        void* p = nullptr;
        if (cudaMalloc(&p, sizeof(Exl3GemvMrParams)) != cudaSuccess)
        {
            (void) cudaGetLastError();
            return nullptr;
        }
        if (cudaMemset(p, 0, sizeof(Exl3GemvMrParams)) != cudaSuccess)
        {
            (void) cudaGetLastError();
            cudaFree(p);
            return nullptr;
        }
        blocks[device] = (Exl3GemvMrParams*) p;
    }
    return blocks[device];
}

void exl3_gemv_multirow_prewarm(int device)
{
    (void) exl3_gemv_mr_param_block(device, true);
}

// =============================================================================
// Multi-row dot core: exl3_gemv_dot_tile_direct with M rows per warp
// =============================================================================
// Same B path, same per-row fdot2 chain and the same quad reduction as the
// m == 1 core, so each row is bit-identical to it. A is the slab's rotated
// rows, lda halves apart. out[r] is row r's column-lane value (lanes 0-15).

template <int bits, int cb, int M>
__device__ __forceinline__ void exl3_gemv_dot_tile_direct_mr
(
    const half* __restrict__ A,
    const int lda,
    const uint16_t* __restrict__ B,
    const int n_tiles,
    const int tile_n,
    const int lane,
    const int kb_begin,
    const int kb_end,
    float* out
)
{
    constexpr int tile_elements = 16 * bits;
    const int r0 = (lane & 3) * 2;

    float accA[M], accB[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) { accA[r] = 0.0f; accB[r] = 0.0f; }

    for (int k_tile = kb_begin; k_tile < kb_end; k_tile++)
    {
        const uint32_t* b_ptr = (const uint32_t*)
            (B + (k_tile * n_tiles + tile_n) * tile_elements);

        FragB frag0, frag1;
        dq_dispatch<bits, cb>(b_ptr, lane << 3, frag0, frag1);

        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            const half2* a2 = (const half2*) (A + r * lda + k_tile * 16);
            half2 a01 = a2[r0 >> 1];
            half2 a89 = a2[(r0 >> 1) + 4];
            accA[r] = __builtin_amdgcn_fdot2(a01, frag0[0], accA[r], false);
            accA[r] = __builtin_amdgcn_fdot2(a89, frag0[1], accA[r], false);
            accB[r] = __builtin_amdgcn_fdot2(a01, frag1[0], accB[r], false);
            accB[r] = __builtin_amdgcn_fdot2(a89, frag1[1], accB[r], false);
        }
    }

    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        accA[r] += __shfl_xor(accA[r], 1, 32);
        accA[r] += __shfl_xor(accA[r], 2, 32);
        accB[r] += __shfl_xor(accB[r], 1, 32);
        accB[r] += __shfl_xor(accB[r], 2, 32);
        float vA = __shfl(accA[r], (lane & 7) * 4, 32);
        float vB = __shfl(accB[r], (lane & 7) * 4, 32);
        out[r] = (lane & 8) ? vB : vA;
    }
}

// Split-K form: the m == 1 chunking and reduction order (exl3_gemv_dot_tile_
// splitk), M rows wide. sh_red holds WARPS_PER_BLOCK * M * 16 floats. Every
// thread of the block enters; out is meaningful for warp 0, lanes 0-15.
template <int bits, int cb, int WARPS_PER_BLOCK, int M>
__device__ __forceinline__ void exl3_gemv_dot_tile_splitk_mr
(
    const half* __restrict__ A,
    const int lda,
    const uint16_t* __restrict__ B,
    const int size_k,
    const int n_tiles,
    const int tile_n,
    const int warp_id,
    const int lane,
    float* sh_red,
    float* out
)
{
    const int num_k_tiles = size_k / 16;
    const int chunk = (num_k_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int kb0 = warp_id * chunk;
    const int kb1 = kb0 + chunk < num_k_tiles ? kb0 + chunk : num_k_tiles;

    float acc[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) acc[r] = 0.0f;
    if (kb0 < kb1)
        exl3_gemv_dot_tile_direct_mr<bits, cb, M>(A, lda, B, n_tiles, tile_n, lane, kb0, kb1, acc);

    if (lane < 16)
    {
        #pragma unroll
        for (int r = 0; r < M; ++r) sh_red[(warp_id * M + r) * 16 + lane] = acc[r];
    }
    __syncthreads();

    if (warp_id == 0 && lane < 16)
    {
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w)
                total += sh_red[(w * M + r) * 16 + lane];
            out[r] = total;
        }
    }
}

static inline size_t exl3_gemv_mr_smem_bytes(int warps, int M)
{
    return (size_t) warps * M * 16 * sizeof(float);
}

// =============================================================================
// Multi-row fused output epilogue: the m == 1 handshake, m row segments
// =============================================================================
// Entered by the warp holding the tile's M rows in acc (lanes 0-15); stores
// rows < rows_valid, arrives once per tile, and as the last of the segment's
// 8 tiles rotates each valid row's segment in place with the shared helper --
// per row the arithmetic of the m == 1 epilogue exactly.

template <bool c_fp32, int M>
__device__ __forceinline__ void exl3_gemv_fused_epilogue_mr
(
    const float* acc,
    void* Cb,
    const int64_t row_off,       // slab base within Cb (j * m * n, or 0 for list outputs)
    const int ldc,               // row stride of C
    const int rows_valid,        // m (<= M)
    const int tile_n,
    const int lane,
    const half* __restrict__ svh,
    int* seg_counter
)
{
    if (lane < 16)
    {
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            if (r >= rows_valid) break;
            const int64_t out_idx = row_off + (int64_t) r * ldc + tile_n * 16 + lane;
            if constexpr (c_fp32)
                ((float*) Cb)[out_idx] = acc[r];
            else
                ((half*) Cb)[out_idx] = __float2half(acc[r]);
        }
    }

    __threadfence();
    int old = 0;
    if (lane == 0) old = atomicAdd(seg_counter, 1);
    old = __shfl(old, 0, 32);
    if (old != 7) return;

    __threadfence();
    if (lane == 0) *seg_counter = 0;
    const int seg = tile_n / 8;
    const float r_scale = 0.088388347648f;  // 1/sqrt(128)
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        if (r >= rows_valid) break;
        const int64_t seg_off = row_off + (int64_t) r * ldc + (int64_t) seg * 128;
        if constexpr (c_fp32)
            exl3_gemv_had_out_128_f(((float*) Cb) + seg_off, svh + seg * 128, r_scale, lane);
        else
            exl3_gemv_had_out_128_h(((half*) Cb) + seg_off, svh + seg * 128, r_scale, lane);
    }
}

// =============================================================================
// Kernel 1: input rotation, m rows per slab, and the graph patch sites
// =============================================================================
// One warp per (slab, row, 128-block). The single-matrix form hosts the six
// GP_gemm_* sites (argument order 0-5 load-bearing; B, C, svh are patch hosts
// only) and the multi-matrix form the four GP_mgemm_* sites (0-3; weights is a
// patch host only -- the path declines weighted calls).

static __global__
__launch_bounds__(256)
void exl3_gemv_mr_had_in_single
(
    const half* __restrict__ A,        // 0: GP_gemm_A
    const uint16_t* __restrict__ B,    // 1: GP_gemm_B_trellis (patch host only)
    void* __restrict__ C,              // 2: GP_gemm_C (patch host only)
    const half* __restrict__ suh,      // 3: GP_gemm_B_suh
    half* __restrict__ A_had,          // 4: GP_gemm_A_had
    const half* __restrict__ svh,      // 5: GP_gemm_B_svh (patch host only)
    Exl3GemvMrParams* __restrict__ pb,
    const int size_k,
    const int size_m
)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        pb->B = B;
        pb->C = C;
        pb->A_had = A_had;
        pb->svh = svh;
    }
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    const int k_segs = size_k / 128;
    if (warp_id >= size_m * k_segs) return;
    const int row = warp_id / k_segs;
    const int seg = warp_id % k_segs;
    const int64_t off = (int64_t) row * size_k + seg * 128;
    exl3_gemv_had_in_128(A + off, A_had + off, suh + seg * 128, lane);
}

static __global__
__launch_bounds__(256)
void exl3_gemv_mr_had_in_multi
(
    const half* __restrict__ A,          // 0: GP_mgemm_A
    void* __restrict__ C,                // 1: GP_mgemm_C (patch host only)
    const int64_t* __restrict__ indices, // 2: GP_mgemm_indices
    const half* __restrict__ weights,    // 3: GP_mgemm_weights (patch host only)
    Exl3GemvMrParams* __restrict__ pb,
    const uintptr_t* __restrict__ suh_list,
    half* __restrict__ A_had,
    const int size_k,
    const int size_m,
    const int bszm_in,
    const int bszm,
    const int min_index,
    const int max_index
)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        pb->Cm = C;
        pb->indices = indices;
        pb->weights = weights;
    }
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    const int k_segs = size_k / 128;
    if (warp_id >= bszm * size_m * k_segs) return;
    const int j   = warp_id / (size_m * k_segs);
    const int row = (warp_id / k_segs) % size_m;
    const int seg = warp_id % k_segs;

    int orig_pos;
    int mat_index = exl3_mgemv_mat_index(indices, j, bszm, min_index, max_index, &orig_pos);
    if (mat_index < 0) return;

    const half* suh = (const half*) suh_list[mat_index];
    // A slab j (or the broadcast input), row `row`; A_had slabs follow packed slot numbering
    const half* A_src = (bszm_in == 1 ? A : A + (int64_t) j * size_m * size_k)
                        + (int64_t) row * size_k + seg * 128;
    half* dst = A_had + ((int64_t) j * size_m + row) * size_k + seg * 128;
    exl3_gemv_had_in_128(A_src, dst, suh + seg * 128, lane);
}

// =============================================================================
// Kernel 2: the split-K GEMV, M rows per warp, fused output epilogue
// =============================================================================
// Grid (n_tiles, bszm); one block per (slot, N-tile). Both forms share the body.

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK, int M>
__device__ __forceinline__ void exl3_gemv_mr_body
(
    const half* __restrict__ A,     // this slab's rotated rows
    const int lda,
    const uint16_t* __restrict__ B,
    const int size_k,
    const int n_tiles_j,
    const int tile_n,
    void* Cb,
    const int64_t row_off,
    const int ldc,
    const int rows_valid,
    const half* __restrict__ svh,
    int* seg_counter,
    const int warp_id,
    const int lane
)
{
    extern __shared__ char shared_mem[];
    float* sh_red = (float*) shared_mem;

    float acc[M];
    exl3_gemv_dot_tile_splitk_mr<bits, cb, WARPS_PER_BLOCK, M>
    (
        A, lda, B, size_k, n_tiles_j, tile_n, warp_id, lane, sh_red, acc
    );
    if (warp_id != 0) return;
    exl3_gemv_fused_epilogue_mr<c_fp32, M>
    (
        acc, Cb, row_off, ldc, rows_valid, tile_n, lane, svh, seg_counter
    );
}

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK, int M>
static __global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_mr_dot_single
(
    Exl3GemvMrParams* __restrict__ pb,
    const int size_k,
    const int size_n,
    const int size_m
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tile_n = blockIdx.x;
    const int n_tiles = size_n / 16;
    exl3_gemv_mr_body<bits, c_fp32, cb, WARPS_PER_BLOCK, M>
    (
        pb->A_had, size_k, pb->B, size_k, n_tiles, tile_n,
        pb->C, 0, size_n, size_m, pb->svh,
        pb->seg_counters + tile_n / 8, warp_id, lane
    );
}

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK, int M>
static __global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_mr_dot_multi
(
    Exl3GemvMrParams* __restrict__ pb,
    const uintptr_t* __restrict__ B_list,
    const uintptr_t* __restrict__ svh_list,
    const half* __restrict__ A_had,
    const int size_k,
    const int size_n,
    const int size_m,
    const int bszm,
    const int min_index,
    const int max_index,
    const int* __restrict__ size_n_list,  // per-matrix widths; size_n is the max
    void* const* __restrict__ c_list      // per-matrix output bases (m rows, stride n_j)
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int j = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int n_tiles = size_n / 16;

    int orig_pos;
    int mat_index = exl3_mgemv_mat_index(pb->indices, j, bszm, min_index, max_index, &orig_pos);
    if (mat_index < 0) return;   // block-uniform
    const int n_j = size_n_list ? size_n_list[mat_index] : size_n;
    const int n_tiles_j = n_j / 16;
    if (tile_n >= n_tiles_j) return;   // block-uniform

    void* Cb = c_list ? c_list[mat_index] : pb->Cm;
    const int64_t row_off = c_list ? 0 : (int64_t) j * size_m * size_n;
    const int ldc = c_list ? n_j : size_n;
    const int n_segs = size_n / 128;
    exl3_gemv_mr_body<bits, c_fp32, cb, WARPS_PER_BLOCK, M>
    (
        A_had + (int64_t) j * size_m * size_k, size_k,
        (const uint16_t*) B_list[mat_index], size_k, n_tiles_j, tile_n,
        Cb, row_off, ldc, size_m, (const half*) svh_list[mat_index],
        pb->seg_counters + j * n_segs + tile_n / 8, warp_id, lane
    );
}

// =============================================================================
// Host dispatch
// =============================================================================

static inline int exl3_gemv_mr_row_tile(int size_m)
{
    return size_m <= 1 ? 1 : (size_m <= 2 ? 2 : (size_m <= 4 ? 4 : 8));
}

// Split-K wave count: the m == 1 rule (shape-aware; EXL3_GEMV_SPLITK_WARPS
// overrides), which keeps row r of an m-row call bit-identical to the m == 1
// split-K result for the same shape. Above the m == 1 single-warp threshold
// (lm_head-scale outputs) the m == 1 path uses one warp per tile; here the
// block count alone saturates the device, so the narrowest split is used.
static inline int exl3_gemv_mr_warps(int size_k, int n_tiles, int bszm)
{
    if (n_tiles > EXL3_GEMV_SPLITK_MAX_TILES) return 4;
    return exl3_gemv_splitk_warps(size_k / 16, n_tiles, bszm);
}

#define LAUNCH_MR_M(bits_val, codebook, warps_val, m_tile, kernel_name, ...) \
    if (c_fp32) { \
        hipLaunchKernelGGL((kernel_name<bits_val, true, codebook, warps_val, m_tile>), \
            grid, block, smem, stream, __VA_ARGS__); \
    } else { \
        hipLaunchKernelGGL((kernel_name<bits_val, false, codebook, warps_val, m_tile>), \
            grid, block, smem, stream, __VA_ARGS__); \
    }
#define LAUNCH_MR_W(bits_val, codebook, warps_val, kernel_name, ...) \
    switch (m_tile) { \
        case 1: LAUNCH_MR_M(bits_val, codebook, warps_val, 1, kernel_name, __VA_ARGS__) break; \
        case 2: LAUNCH_MR_M(bits_val, codebook, warps_val, 2, kernel_name, __VA_ARGS__) break; \
        case 4: LAUNCH_MR_M(bits_val, codebook, warps_val, 4, kernel_name, __VA_ARGS__) break; \
        case 8: LAUNCH_MR_M(bits_val, codebook, warps_val, 8, kernel_name, __VA_ARGS__) break; \
    }
#define LAUNCH_MR_CB(bits_val, codebook, kernel_name, ...) \
    switch (warps) { \
        case 4:  LAUNCH_MR_W(bits_val, codebook, 4,  kernel_name, __VA_ARGS__) break; \
        case 8:  LAUNCH_MR_W(bits_val, codebook, 8,  kernel_name, __VA_ARGS__) break; \
        case 16: LAUNCH_MR_W(bits_val, codebook, 16, kernel_name, __VA_ARGS__) break; \
    }
#define LAUNCH_MR_K(bits_val, kernel_name, ...) \
    switch (cb) { \
        case 0: LAUNCH_MR_CB(bits_val, 0, kernel_name, __VA_ARGS__); break; \
        case 1: LAUNCH_MR_CB(bits_val, 1, kernel_name, __VA_ARGS__); break; \
        case 2: LAUNCH_MR_CB(bits_val, 2, kernel_name, __VA_ARGS__); break; \
    }
#define LAUNCH_MR(kernel_name, ...) \
    switch (K) { \
        case 1: LAUNCH_MR_K(1, kernel_name, __VA_ARGS__); break; \
        case 2: LAUNCH_MR_K(2, kernel_name, __VA_ARGS__); break; \
        case 3: LAUNCH_MR_K(3, kernel_name, __VA_ARGS__); break; \
        case 4: LAUNCH_MR_K(4, kernel_name, __VA_ARGS__); break; \
        case 5: LAUNCH_MR_K(5, kernel_name, __VA_ARGS__); break; \
        case 6: LAUNCH_MR_K(6, kernel_name, __VA_ARGS__); break; \
        case 7: LAUNCH_MR_K(7, kernel_name, __VA_ARGS__); break; \
        case 8: LAUNCH_MR_K(8, kernel_name, __VA_ARGS__); break; \
    }

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
)
{
    if (size_m < exl3_gemv_mr_min_m(size_n / 16) || size_m > exl3_gemv_max_m()) return false;
    if (K < 1 || K > 8) return false;
    if (size_k % 128 || size_n % 128) return false;
    if (!suh_ptr || !A_had_ptr || !svh_ptr) return false;
    if (size_n / 128 > EXL3_MGEMV_SEG_COUNTERS) return false;
    Exl3GemvMrParams* pb = exl3_gemv_mr_param_block(device, graph == nullptr);
    if (!pb) return false;

    // 1. Rotation + patch sites
    {
        int blocks = CEIL_DIVIDE(size_m * (size_k / 128) * 32, 256);
        hipLaunchKernelGGL
        (
            exl3_gemv_mr_had_in_single,
            dim3(blocks), dim3(256), 0, stream,
            A_ptr, B_ptr, C_ptr, suh_ptr, A_had_ptr, svh_ptr, pb, size_k, size_m
        );
        if (graph)
        {
            void* k = (void*) exl3_gemv_mr_had_in_single;
            graph->record_param(k, GP_gemm_A, 0);
            graph->record_param(k, GP_gemm_B_trellis, 1);
            graph->record_param(k, GP_gemm_C, 2);
            graph->record_param(k, GP_gemm_B_suh, 3);
            graph->record_param(k, GP_gemm_A_had, 4);
            graph->record_param(k, GP_gemm_B_svh, 5);
            graph->record_param(k, GP_end, 0);
        }
    }

    // 2. GEMV, M rows per warp, fused output epilogue
    {
        const int n_tiles = size_n / 16;
        const int warps = exl3_gemv_mr_warps(size_k, n_tiles, 1);
        const int m_tile = exl3_gemv_mr_row_tile(size_m);
        dim3 grid(n_tiles, 1);
        dim3 block(warps * 32);
        size_t smem = exl3_gemv_mr_smem_bytes(warps, m_tile);
        LAUNCH_MR(exl3_gemv_mr_dot_single, pb, size_k, size_n, size_m)
    }
    return true;
}

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
)
{
    if (size_m < exl3_gemv_mr_min_m(size_n / 16) || size_m > exl3_gemv_max_m()) return false;
    if (K < 1 || K > 8) return false;
    if (size_k % 128 || size_n % 128) return false;
    if (weights_ptr || num_tokens != 1) return false;
    if ((size_n_list != nullptr) != (c_list != nullptr)) return false;
    if (size_n_list && min_index >= 0) return false;
    if (!suh_ptr_ptr || !A_had_ptr || !svh_ptr_ptr) return false;
    int bszm = MAX(bszm_in, bszm_out);
    if (bszm < 1 || bszm > 128) return false;
    if (bszm * (size_n / 128) > EXL3_MGEMV_SEG_COUNTERS) return false;
    Exl3GemvMrParams* pb = exl3_gemv_mr_param_block(device, graph == nullptr);
    if (!pb) return false;

    // 1. Rotation + patch sites
    {
        int blocks = CEIL_DIVIDE(bszm * size_m * (size_k / 128) * 32, 256);
        hipLaunchKernelGGL
        (
            exl3_gemv_mr_had_in_multi,
            dim3(blocks), dim3(256), 0, stream,
            A_ptr, C_ptr, indices_ptr, weights_ptr, pb, suh_ptr_ptr, A_had_ptr,
            size_k, size_m, bszm_in, bszm, min_index, max_index
        );
        if (graph)
        {
            void* k = (void*) exl3_gemv_mr_had_in_multi;
            graph->record_param(k, GP_mgemm_A, 0);
            graph->record_param(k, GP_mgemm_C, 1);
            graph->record_param(k, GP_mgemm_indices, 2);
            graph->record_param(k, GP_mgemm_weights, 3);
            graph->record_param(k, GP_end, 0);
        }
    }

    // 2. GEMV
    {
        const int n_tiles = size_n / 16;
        const int warps = exl3_gemv_mr_warps(size_k, n_tiles, bszm);
        const int m_tile = exl3_gemv_mr_row_tile(size_m);
        dim3 grid(n_tiles, bszm);
        dim3 block(warps * 32);
        size_t smem = exl3_gemv_mr_smem_bytes(warps, m_tile);
        LAUNCH_MR(exl3_gemv_mr_dot_multi, pb, B_ptr_ptr, svh_ptr_ptr, A_had_ptr,
                  size_k, size_n, size_m, bszm, min_index, max_index, size_n_list, c_list)
    }
    return true;
}

#undef LAUNCH_MR
#undef LAUNCH_MR_K
#undef LAUNCH_MR_CB
#undef LAUNCH_MR_W
#undef LAUNCH_MR_M
