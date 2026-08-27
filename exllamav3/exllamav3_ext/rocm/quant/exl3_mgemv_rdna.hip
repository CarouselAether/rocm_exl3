// =============================================================================
// Multi-matrix (expert-batched) GEMV for RDNA at m == 1
// =============================================================================
//
// Work item 4d of the decode investigation. exl3_mgemm's cooperative kernel is
// structurally capped at ~12% occupancy at m == 1: grid.sync() requires every
// block co-resident, which limits the grid to ~20 workgroups on gfx1151
// regardless of how much parallel work exists (measured; see RDNA_NOTES).
// The plain-launch GEMV reaches the memory roofline on the same hardware, but
// it is single-matrix -- and a single 3072->1024 expert yields only 8 blocks,
// which loses to the tile GEMM (0.58x, measured).
//
// The multi-matrix form dissolves that trade-off: bszm experts x (size_n / 16)
// tiles gives 640-1920 warps for the shapes that matter (Laguna gate/up/down at
// top-10), oversubscribing all 80 SIMDs without any cooperative launch. Four
// plain kernels replace the one cooperative one:
//
//   1. exl3_mgemv_had_in_kernel   per-expert input rotation (SUH), and the
//                                 graph-patch prologue -- see below
//   2. exl3_mgemv_dot_kernel      the GEMV proper, one warp per (expert, tile);
//                                 per-warp core shared verbatim with the
//                                 single-matrix kernel (exl3_gemv_dot_tile)
//   3. exl3_mgemv_had_out_kernel  per-expert output rotation (SVH), folding in
//                                 the routing weight exactly as the cooperative
//                                 kernel does (scale * weights[j])
//   4. exl3_mgemv_reduce_kernel   the weighted-sum epilogue, mirroring the
//                                 cooperative kernel's num_tokens grouping
//
// Graph capture. Upstream records four patchable parameters per mgemm launch
// (A, C, indices, weights -- add_graph_args in exl3_gemm_rdna.hip), and
// Graph::launch() patches ONE site per caller-supplied parameter, walking sites
// and params in lockstep. A multi-kernel path therefore cannot record the same
// parameter on several kernels and expect all copies patched. Instead all four
// sites are recorded on kernel 1, which takes all four as ordinary arguments
// (C and weights exist there only to host the patch site) and republishes the
// ones the downstream kernels need through a per-device Exl3MgemvParams block.
// Downstream kernels read C/indices/weights from that block, so a patched
// prologue re-propagates automatically on every replay. Site order (A, C,
// indices, weights, GP_end) matches the cooperative path, which is what the
// upstream callers' params subsequences are written against.
//
// Index packing (min_index >= 0). The cooperative kernel packs indices within
// [min_index, max_index) into a device global and processes only the packed
// slots; C rows, A_had slabs and the reduce stride all follow PACKED slot
// numbering. BC_BlockSparseMLP passes a real expert range at decode, so this
// path must reproduce that or it would never run. There is no grid.sync() here
// to broadcast a packed list, so each warp/block derives its own view with an
// O(bszm) scan of the indices tensor (bszm <= 128, cached, negligible next to
// the 128-wide hadamard or a full-K dot). Semantics are identical: slot j means
// "the j-th index inside the range", weights follow their original position,
// and the reduce stride is the packed count.
//
// Per-matrix width/output lists (size_n_list / c_list). DS4's bc_dsa fan and
// fan2 sites project ONE input through several matrices of differing widths in
// a single call: size_n is then only the max width, matrix mat_index is
// size_n_list[mat_index] wide, and its output goes to its own tensor at
// c_list[mat_index] (row 0 -- this whole path is m == 1) instead of C's j-th
// row. The entry point enforces no-weights/no-packing/single-token for list
// calls, so the reduce epilogue never runs with lists. Both list pointers are
// device arrays whose contents the kernels read per launch -- the same
// indirection B_list already uses -- so list calls are graph-stable without
// new patch sites. The grid covers the widest matrix; warps (or split-K
// blocks, which are tile-uniform) beyond a narrower matrix's width exit early,
// and the dot core sees the matrix's own n_tiles because B is packed at that
// width.
// =============================================================================

#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../../util.h"
#include "../../util.cuh"
#include "../../graph.cuh"
#include "exl3_mgemv_rdna.hip.h"
#include "exl3_gemv_kernel_rdna.hip.h"

#include <cstdlib>
#include <mutex>

// -----------------------------------------------------------------------------
// Kill switch: EXL3_MGEMV = 0 sends every call back to the cooperative
// exl3_mgemm. Re-read on every call so a single process can A/B the two paths
// (bench tooling toggles os.environ between calls).
// -----------------------------------------------------------------------------
static bool exl3_mgemv_enabled()
{
    const char* env = std::getenv("EXL3_MGEMV");
    if (!env) return true;
    return atoi(env) != 0;
}

// -----------------------------------------------------------------------------
// Per-device parameter block
// -----------------------------------------------------------------------------
// cudaMalloc'd lazily. The first mgemm call of a BC module runs eagerly (the
// graph records only on the second pass), so allocation normally happens
// outside capture; cudaMalloc is stream-less and remains legal if a first call
// ever does arrive mid-capture.
// allow_alloc must be false while a graph is capturing: hipMalloc synchronizes
// the device, which invalidates an active stream capture. The eager first pass
// every BC module runs before capturing (graph == nullptr there) is where
// allocation happens; a capture arriving on a cold device declines to the
// cooperative kernel rather than allocating.
static Exl3MgemvParams* exl3_mgemv_param_block(int device, bool allow_alloc)
{
    static std::mutex mtx;
    static Exl3MgemvParams* blocks[64] = {};
    if (device < 0 || device >= 64) return nullptr;
    std::lock_guard<std::mutex> lock(mtx);
    if (!blocks[device] && allow_alloc)
    {
        void* p = nullptr;
        if (cudaMalloc(&p, sizeof(Exl3MgemvParams)) != cudaSuccess)
        {
            (void) cudaGetLastError();
            return nullptr;
        }
        blocks[device] = (Exl3MgemvParams*) p;
    }
    return blocks[device];
}

// -----------------------------------------------------------------------------
// Slot resolution, shared by all four kernels
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
// Kernel 1: prologue + per-expert input rotation
// =============================================================================
// Argument order 0-3 is load-bearing: these are the graph patch sites, recorded
// as (A, C, indices, weights) to match the cooperative path's site order. C and
// weights are not used by this kernel's own math.

__global__
__launch_bounds__(256)
void exl3_mgemv_had_in_kernel
(
    const half* __restrict__ A,          // 0: GP_mgemm_A
    void* __restrict__ C,                // 1: GP_mgemm_C (patch host only)
    const int64_t* __restrict__ indices, // 2: GP_mgemm_indices
    const half* __restrict__ weights,    // 3: GP_mgemm_weights (patch host only)
    Exl3MgemvParams* __restrict__ pb,
    const uintptr_t* __restrict__ suh_list,
    half* __restrict__ A_had,
    const int size_k,
    const int bszm_in,
    const int bszm,
    const int min_index,
    const int max_index
)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        pb->C = C;
        pb->indices = indices;
        pb->weights = weights;
    }

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int k_warps = size_k / 128;
    if (warp_id >= bszm * k_warps) return;

    int j   = warp_id / k_warps;
    int seg = warp_id % k_warps;

    int orig_pos;
    int mat_index = exl3_mgemv_mat_index(indices, j, bszm, min_index, max_index, &orig_pos);
    if (mat_index < 0) return;

    const half* suh = (const half*) suh_list[mat_index];
    const half* A_j = (bszm_in == 1 ? A : A + (int64_t) j * size_k) + seg * 128;
    half* A_had_j   = A_had + (int64_t) j * size_k + seg * 128;

    had_hf_r_128_inner<true, false>
    (
        A_j,
        A_had_j,
        suh + seg * 128,
        0.088388347648f  // 1/sqrt(128)
    );
}

// =============================================================================
// Kernel 2: the GEMV proper
// =============================================================================
// Grid (ceil(n_tiles / WARPS_PER_BLOCK), bszm); one warp per (expert, N-tile).
// The per-warp body is exl3_gemv_dot_tile, shared with the single-matrix GEMV.

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK>
__global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_mgemv_dot_kernel
(
    const Exl3MgemvParams* __restrict__ pb,
    const uintptr_t* __restrict__ B_list,
    const half* __restrict__ A_had,
    const int size_k,
    const int size_n,
    const int bszm,
    const int min_index,
    const int max_index,
    const int* __restrict__ size_n_list,  // per-matrix widths; size_n is the max
    void* const* __restrict__ c_list,     // per-matrix output bases (row 0; m == 1)
    const bool lds_core
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int j = blockIdx.y;

    const int tile_n = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int n_tiles = size_n / 16;
    if (tile_n >= n_tiles) return;

    const int64_t* indices = pb->indices;
    int orig_pos;
    int mat_index = exl3_mgemv_mat_index(indices, j, bszm, min_index, max_index, &orig_pos);
    if (mat_index < 0) return;

    // Per-matrix width (the cooperative kernel's n_j): the grid covers the
    // widest matrix, warps beyond a narrower one exit here. B is packed at the
    // matrix's own width, so the dot core must see n_tiles_j, not the max.
    const int n_tiles_j = size_n_list ? size_n_list[mat_index] / 16 : n_tiles;
    if (tile_n >= n_tiles_j) return;

    const uint16_t* B = (const uint16_t*) B_list[mat_index];
    const half* A_j = A_had + (int64_t) j * size_k;

    // Dynamic shared memory -- same carve as the single-matrix kernel
    extern __shared__ char shared_mem[];
    constexpr int SH_STRIDE = EXL3_GEMV_SH_STRIDE;
    half* sh_b_dq = (half*) shared_mem;
    uint16_t* sh_b_quant = (uint16_t*) (sh_b_dq + WARPS_PER_BLOCK * 16 * SH_STRIDE);
    half* my_sh_b = sh_b_dq + warp_id * 16 * SH_STRIDE;
    uint16_t* my_sh_b_quant = sh_b_quant + warp_id * EXL3_GEMV_SH_QUANT_U16;

    float accum = lds_core
        ? exl3_gemv_dot_tile<bits, cb>
          (
              A_j, B, size_k, n_tiles_j, tile_n, lane, my_sh_b, my_sh_b_quant,
              0, size_k / 16
          )
        : exl3_gemv_dot_tile_direct<bits, cb>
          (
              A_j, B, n_tiles_j, tile_n, lane, 0, size_k / 16
          );

    if (lane < 16)
    {
        if (c_list)
        {
            // Each matrix owns its output tensor; no j stride (m == 1)
            const int out_idx = tile_n * 16 + lane;
            if constexpr (c_fp32)
                ((float*) c_list[mat_index])[out_idx] = accum;
            else
                ((half*) c_list[mat_index])[out_idx] = __float2half(accum);
        }
        else
        {
            const int64_t out_idx = (int64_t) j * size_n + tile_n * 16 + lane;
            if constexpr (c_fp32)
                ((float*) pb->C)[out_idx] = accum;
            else
                ((half*) pb->C)[out_idx] = __float2half(accum);
        }
    }
}

// Split-K form of kernel 2: one block per (expert, N-tile), warps share K.
// The expert shapes this path exists for are exactly the narrow-output ones
// the single-warp form starves on (64 tiles at 1024 wide), so in practice
// this is the form MoE decode takes; the threshold keeps parity with the
// single-matrix sites.
template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK>
__global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_mgemv_dot_kernel_splitk
(
    const Exl3MgemvParams* __restrict__ pb,
    const uintptr_t* __restrict__ B_list,
    const half* __restrict__ A_had,
    const int size_k,
    const int size_n,
    const int bszm,
    const int min_index,
    const int max_index,
    const int* __restrict__ size_n_list,  // per-matrix widths; size_n is the max
    void* const* __restrict__ c_list,     // per-matrix output bases (row 0; m == 1)
    const bool lds_core
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int j = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int n_tiles = size_n / 16;

    const int64_t* indices = pb->indices;
    int orig_pos;
    int mat_index = exl3_mgemv_mat_index(indices, j, bszm, min_index, max_index, &orig_pos);
    if (mat_index < 0) return;   // whole block exits together; no sync hazard

    // Per-matrix width: tile_n and mat_index are block-uniform, so the whole
    // block exits together here too and the split-K __syncthreads is safe
    const int n_tiles_j = size_n_list ? size_n_list[mat_index] / 16 : n_tiles;
    if (tile_n >= n_tiles_j) return;

    const uint16_t* B = (const uint16_t*) B_list[mat_index];
    const half* A_j = A_had + (int64_t) j * size_k;

    extern __shared__ char shared_mem[];
    constexpr int SH_STRIDE = EXL3_GEMV_SH_STRIDE;
    half* sh_b_dq = (half*) shared_mem;
    uint16_t* sh_b_quant = (uint16_t*) (sh_b_dq + WARPS_PER_BLOCK * 16 * SH_STRIDE);
    float* sh_red = (float*) (sh_b_quant + WARPS_PER_BLOCK * EXL3_GEMV_SH_QUANT_U16);
    half* my_sh_b = sh_b_dq + warp_id * 16 * SH_STRIDE;
    uint16_t* my_sh_b_quant = sh_b_quant + warp_id * EXL3_GEMV_SH_QUANT_U16;

    float accum = exl3_gemv_dot_tile_splitk<bits, cb, WARPS_PER_BLOCK>
    (
        A_j, B, size_k, n_tiles_j, tile_n, warp_id, lane, my_sh_b, my_sh_b_quant,
        sh_red, lds_core
    );

    if (warp_id == 0 && lane < 16)
    {
        if (c_list)
        {
            const int out_idx = tile_n * 16 + lane;
            if constexpr (c_fp32)
                ((float*) c_list[mat_index])[out_idx] = accum;
            else
                ((half*) c_list[mat_index])[out_idx] = __float2half(accum);
        }
        else
        {
            const int64_t out_idx = (int64_t) j * size_n + tile_n * 16 + lane;
            if constexpr (c_fp32)
                ((float*) pb->C)[out_idx] = accum;
            else
                ((half*) pb->C)[out_idx] = __float2half(accum);
        }
    }
}

// =============================================================================
// Kernel 3: per-expert output rotation, weight folded into the scale
// =============================================================================

template <bool c_fp32>
__global__
__launch_bounds__(256)
void exl3_mgemv_had_out_kernel
(
    const Exl3MgemvParams* __restrict__ pb,
    const uintptr_t* __restrict__ svh_list,
    const int size_n,
    const int bszm,
    const int min_index,
    const int max_index,
    const int* __restrict__ size_n_list,  // per-matrix widths; size_n is the max
    void* const* __restrict__ c_list      // per-matrix output bases (row 0; m == 1)
)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int n_warps = size_n / 128;
    if (warp_id >= bszm * n_warps) return;

    int j   = warp_id / n_warps;
    int seg = warp_id % n_warps;

    const int64_t* indices = pb->indices;
    int orig_pos;
    int mat_index = exl3_mgemv_mat_index(indices, j, bszm, min_index, max_index, &orig_pos);
    if (mat_index < 0) return;

    // Per-matrix width: warps beyond a narrower matrix's rotation blocks exit.
    // EXL3 trellis padding guarantees n_j % 128 == 0 (the cooperative kernel's
    // output rotation depends on the same).
    const int n_j = size_n_list ? size_n_list[mat_index] : size_n;
    if (seg * 128 >= n_j) return;

    const half* svh = (const half*) svh_list[mat_index];

    float scale = 0.088388347648f;  // 1/sqrt(128)
    if (pb->weights) scale *= __half2float(pb->weights[orig_pos]);

    void* Cb = c_list ? c_list[mat_index] : pb->C;
    int64_t offset = (c_list ? 0 : (int64_t) j * size_n) + seg * 128;

    if constexpr (c_fp32)
        had_ff_r_128_inner<false, true>
        (
            ((const float*) Cb) + offset,
            ((float*) Cb) + offset,
            svh + seg * 128,
            scale
        );
    else
        had_hf_r_128_inner<false, true>
        (
            ((const half*) Cb) + offset,
            ((half*) Cb) + offset,
            svh + seg * 128,
            scale
        );
}

// =============================================================================
// Kernel 4: weighted-sum epilogue
// =============================================================================
// Mirrors the cooperative kernel's reduction (m == 1 form): each of the
// num_tokens groups of (packed / num_tokens) contiguous slots sums into its own
// output row. Accumulation dtype matches the cooperative kernel (half chain for
// fp16 C) so the two paths stay numerically comparable. One thread owns a
// column across every t, preserving the increasing-t ordering the in-place
// overwrite depends on.

template <bool c_fp32>
__global__
__launch_bounds__(256)
void exl3_mgemv_reduce_kernel
(
    const Exl3MgemvParams* __restrict__ pb,
    const int size_n,
    const int num_tokens,
    const int bszm,
    const int min_index,
    const int max_index
)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= size_n) return;

    int packed = exl3_mgemv_packed_count(pb->indices, bszm, min_index, max_index);
    int stride = packed / num_tokens;

    for (int t = 0; t < num_tokens; ++t)
    {
        if constexpr (c_fp32)
        {
            float* C_ = ((float*) pb->C) + (int64_t) t * stride * size_n + col;
            float sum = 0.0f;
            for (int jj = 0; jj < stride; ++jj)
            {
                sum += *C_;
                C_ += size_n;
            }
            ((float*) pb->C)[(int64_t) t * size_n + col] = sum;
        }
        else
        {
            half* C_ = ((half*) pb->C) + (int64_t) t * stride * size_n + col;
            half sum = {};
            for (int jj = 0; jj < stride; ++jj)
            {
                sum = __hadd(sum, *C_);
                C_ += size_n;
            }
            ((half*) pb->C)[(int64_t) t * size_n + col] = sum;
        }
    }
}

// =============================================================================
// Host dispatch
// =============================================================================

// One warp per (expert, 16-wide tile); 8 warps per block. The per-expert tile
// counts this path exists for (64 at 1024-wide, 192 at 3072-wide) are multiples
// of 8, so no partial blocks in practice; partial blocks are handled anyway.
#define EXL3_MGEMV_WARPS 8

bool exl3_mgemv_try_launch
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
    // Hard constraints first, then the kill switch. m == 1 and the 128-multiple
    // dims are kernel requirements (hadamard blocks). Per-matrix width/output
    // lists (DS4's bc_dsa fan sites) are supported under the same constraints
    // the entry point enforces for them -- no weights, no packing, single
    // token; size_n is then the max width, per-matrix widths come from the
    // device-side list and are %16/%128 by trellis padding. Both lists arrive
    // together or not at all; a half-supplied pair falls through.
    if (size_m != 1) return false;
    if (K < 1 || K > 8) return false;
    if (size_k % 128 || size_n % 128) return false;
    if ((size_n_list != nullptr) != (c_list != nullptr)) return false;
    if (size_n_list && (weights_ptr || num_tokens != 1 || min_index >= 0)) return false;
    // v1.4.4: upstream made num_tokens > 1 legal WITH range filtering by switching the
    // cooperative kernel to position-preserving masking (out-of-range slots -> -1 in
    // place). This path still packs (compacts), which collapses the per-token slot runs
    // the grouped reduce below divides by (stride = packed / num_tokens) -- wrong
    // grouping, and packed need not even divide. Decline and let the cooperative
    // kernel handle it; only TP-sharded / CPU-split expert maps hit this combination.
    if (num_tokens > 1 && min_index >= 0) return false;
    if (!exl3_mgemv_enabled()) return false;

    // 128 = MAX_INDICES, the cooperative kernel's packing-array bound; kept for
    // parity even though this path scans rather than packs
    int bszm = MAX(bszm_in, bszm_out);
    if (bszm < 1 || bszm > 128) return false;
    if (num_tokens < 1) return false;

    Exl3MgemvParams* pb = exl3_mgemv_param_block(device, graph == nullptr);
    if (!pb) return false;

    // -------------------------------------------------------------------------
    // 1. Prologue + input rotation
    // -------------------------------------------------------------------------
    {
        int k_warps = size_k / 128;
        int blocks = CEIL_DIVIDE(bszm * k_warps * 32, 256);
        hipLaunchKernelGGL
        (
            exl3_mgemv_had_in_kernel,
            dim3(blocks), dim3(256), 0, stream,
            A_ptr, C_ptr, indices_ptr, weights_ptr, pb,
            suh_ptr_ptr, A_had_ptr, size_k, bszm_in, bszm, min_index, max_index
        );
        if (graph)
        {
            void* k = (void*) exl3_mgemv_had_in_kernel;
            graph->record_param(k, GP_mgemm_A, 0);
            graph->record_param(k, GP_mgemm_C, 1);
            graph->record_param(k, GP_mgemm_indices, 2);
            graph->record_param(k, GP_mgemm_weights, 3);
            graph->record_param(k, GP_end, 0);
        }
    }

    // -------------------------------------------------------------------------
    // 2. GEMV -- split-K when the per-expert output is narrow (same rule as
    //    the single-matrix sites), which for MoE decode it always is
    // -------------------------------------------------------------------------
    {
        int n_tiles = size_n / 16;
        bool splitk = exl3_gemv_splitk_enabled() && n_tiles <= EXL3_GEMV_SPLITK_MAX_TILES;
        const bool lds_core = exl3_gemv_lds_core();

        // The split-K wave count is shape-aware (see exl3_gemv_splitk_warps);
        // the grid here is n_tiles x bszm blocks, so bszm enters the rule.
        // The single-warp form keeps the fixed EXL3_MGEMV_WARPS.
        int warps = splitk ? exl3_gemv_splitk_warps(size_k / 16, n_tiles, bszm)
                           : EXL3_MGEMV_WARPS;

        dim3 grid(splitk ? n_tiles : CEIL_DIVIDE(n_tiles, warps), bszm);
        dim3 block(warps * 32);
        size_t smem = splitk ? exl3_gemv_smem_bytes_splitk(warps)
                             : exl3_gemv_smem_bytes(warps);

        #define LAUNCH_MGEMV_FP(bits_val, codebook, warps_val, kernel_name) \
            if (c_fp32) { \
                hipLaunchKernelGGL( \
                    (kernel_name<bits_val, true, codebook, warps_val>), \
                    grid, block, smem, stream, \
                    pb, B_ptr_ptr, A_had_ptr, size_k, size_n, bszm, min_index, max_index, \
                    size_n_list, c_list, lds_core); \
            } else { \
                hipLaunchKernelGGL( \
                    (kernel_name<bits_val, false, codebook, warps_val>), \
                    grid, block, smem, stream, \
                    pb, B_ptr_ptr, A_had_ptr, size_k, size_n, bszm, min_index, max_index, \
                    size_n_list, c_list, lds_core); \
            }

        #define LAUNCH_MGEMV_CB(bits_val, codebook) \
            if (splitk) { \
                switch (warps) { \
                    case 4:  LAUNCH_MGEMV_FP(bits_val, codebook, 4,  exl3_mgemv_dot_kernel_splitk) break; \
                    case 8:  LAUNCH_MGEMV_FP(bits_val, codebook, 8,  exl3_mgemv_dot_kernel_splitk) break; \
                    case 16: LAUNCH_MGEMV_FP(bits_val, codebook, 16, exl3_mgemv_dot_kernel_splitk) break; \
                } \
            } else { \
                LAUNCH_MGEMV_FP(bits_val, codebook, EXL3_MGEMV_WARPS, exl3_mgemv_dot_kernel) \
            }

        #define LAUNCH_MGEMV(bits_val) \
            switch (cb) { \
                case 0: LAUNCH_MGEMV_CB(bits_val, 0); break; \
                case 1: LAUNCH_MGEMV_CB(bits_val, 1); break; \
                case 2: LAUNCH_MGEMV_CB(bits_val, 2); break; \
            }

        switch (K) {
            case 1: LAUNCH_MGEMV(1); break;
            case 2: LAUNCH_MGEMV(2); break;
            case 3: LAUNCH_MGEMV(3); break;
            case 4: LAUNCH_MGEMV(4); break;
            case 5: LAUNCH_MGEMV(5); break;
            case 6: LAUNCH_MGEMV(6); break;
            case 7: LAUNCH_MGEMV(7); break;
            case 8: LAUNCH_MGEMV(8); break;
        }

        #undef LAUNCH_MGEMV
        #undef LAUNCH_MGEMV_CB
        #undef LAUNCH_MGEMV_FP
    }

    // -------------------------------------------------------------------------
    // 3. Output rotation (+ routing weight)
    // -------------------------------------------------------------------------
    {
        int n_warps = size_n / 128;
        int blocks = CEIL_DIVIDE(bszm * n_warps * 32, 256);
        if (c_fp32)
            hipLaunchKernelGGL
            (
                exl3_mgemv_had_out_kernel<true>,
                dim3(blocks), dim3(256), 0, stream,
                pb, svh_ptr_ptr, size_n, bszm, min_index, max_index,
                size_n_list, c_list
            );
        else
            hipLaunchKernelGGL
            (
                exl3_mgemv_had_out_kernel<false>,
                dim3(blocks), dim3(256), 0, stream,
                pb, svh_ptr_ptr, size_n, bszm, min_index, max_index,
                size_n_list, c_list
            );
    }

    // -------------------------------------------------------------------------
    // 4. Weighted-sum epilogue (only when the call has routing weights, exactly
    //    like the cooperative kernel's `if (B_weights)` epilogue; whether the
    //    argument is present is fixed per call site, so this is graph-stable)
    // -------------------------------------------------------------------------
    if (weights_ptr)
    {
        int blocks = CEIL_DIVIDE(size_n, 256);
        if (c_fp32)
            hipLaunchKernelGGL
            (
                exl3_mgemv_reduce_kernel<true>,
                dim3(blocks), dim3(256), 0, stream,
                pb, size_n, num_tokens, bszm, min_index, max_index
            );
        else
            hipLaunchKernelGGL
            (
                exl3_mgemv_reduce_kernel<false>,
                dim3(blocks), dim3(256), 0, stream,
                pb, size_n, num_tokens, bszm, min_index, max_index
            );
    }

    return true;
}
