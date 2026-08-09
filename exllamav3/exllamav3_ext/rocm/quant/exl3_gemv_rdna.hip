// =============================================================================
// exl3_gemv host dispatch for RDNA
// =============================================================================
//
// Provides upstream v1.3.0's quant/exl3_gemv.cu interface -- exl3_gemv_try_launch
// and exl3_gemv -- on top of the fork's fdot2 GEMV kernel. The kernel is carried
// forward; the dispatch around it is new, because the interface changed shape
// between the fork's base (v0.0.29) and v1.3.0.
//
// What upstream's version does that this one does not, and why:
//
// - Cooperative launch. Upstream's GEMV kernel fuses the input Hadamard, the
//   matmul and the output Hadamard into one cooperative kernel separated by
//   grid.sync(). The RDNA kernel is a plain launch with the transforms as
//   separate kernels either side, so this is three launches, not one.
//
// - Occupancy-driven grid sizing and the narrow/wide config pair. Those exist to
//   size a cooperative grid; with a plain launch the grid is just one warp per
//   n-tile. Wave count per block is chosen instead, from the fork's measurements.
//
// - The shape heuristic. Upstream's envelope (K == 4 or cb != 0, n <= 8192 bands,
//   co-residency thresholds) was measured on Ampere against upstream's own tuned
//   GEMM. None of it transfers. See exl3_gemv_rdna_eligible() below for what
//   replaces it and what is still unmeasured.
//
// - m up to 8. This kernel is m == 1 only; see the note on the WMMA kernel in
//   exl3_gemv_kernel_rdna.hip.h.
//
// Bits 1-8 and all three codebooks are supported, which is wider than upstream's
// GEMV path (K 2-4, and K == 4 only for cb 0).
// =============================================================================

#include <cuda_fp16.h>
#include "../../quant/exl3_gemv.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../../util.h"
#include "../../util.cuh"
#include "../../graph.cuh"
#include "exl3_gemv_kernel_rdna.hip.h"
#include "exl3_mgemv_rdna.hip.h"

#include <cstdlib>
#include <mutex>

// Defined in the graph-captured GEMV section at the bottom of this file;
// called from the non-graph dispatch above it.
static void exl3_gemv_graph_prewarm(int device);

// Env: EXL3_GEMV = 0 disables the path (every eligible call falls through to
// the cooperative GEMM); 1 (default) routes shape-aware per
// exl3_gemv_rdna_pays(); 2 takes every eligible call regardless of the
// measured envelope -- upstream's mode-2 semantics, kept for per-shape A/B
// measurement (bench_gemv_vs_gemm). Re-read on every call so one process can
// toggle it.
static int exl3_gemv_env_mode()
{
    const char* env = std::getenv("EXL3_GEMV");
    if (!env) return 1;
    return atoi(env);
}

// -----------------------------------------------------------------------------
// Eligibility
// -----------------------------------------------------------------------------
// These are hard constraints of the kernel, not a performance envelope:
//
// - m == 1: the kernel has no M loop; A is indexed as a bare k-vector.
// - bits 1-8: the full EXL3 range, all covered by dq_dispatch.
// - k % 128, n % 128: the kernel itself only needs multiples of 16, but the
//   Hadamard helpers work on 128-element blocks. Upstream applies the same two
//   constraints, so nothing eligible upstream is rejected here for shape.
//
// Deliberately NOT a constraint: has_su_sv. Upstream's kernel implements the
// transforms internally and so requires all three tensors; here they are
// separate kernels that are simply skipped when absent, exactly as the fork did.
//
// What is unmeasured: whether this beats the cooperative GEMM at m == 1 now that
// EXL3_RDNA_SMEM lets the GEMM be co-resident. The fork routed every m == 1 call
// here because the GEMM deadlocked, which is no longer the reason. Taking every
// eligible call preserves the fork's validated behaviour; EXL3_GEMV=0 measures
// the alternative.
static bool exl3_gemv_rdna_eligible
(
    int size_m, int size_k, int size_n, int K,
    const half* suh_ptr, const half* A_had_ptr
)
{
    if (size_m != 1) return false;
    if (K < 1 || K > 8) return false;
    if (size_k % 128 || size_n % 128) return false;

    // suh without A_had: upstream's "Must supply A_had with suh" check is
    // commented out in exl3_gemm.cu, so this reaches the kernel, where upstream
    // dereferences the null A_had and faults. This path would instead skip the
    // transform and return a plausible but silently untransformed result, which
    // is strictly worse. Decline and let the GEMM handle it, so the two backends
    // fail the same way on the same malformed call.
    if (suh_ptr && !A_had_ptr) return false;

    return true;
}

// -----------------------------------------------------------------------------
// Wave selection
// -----------------------------------------------------------------------------
// From the fork's benchmarks on gfx1151. One warp computes one 16-wide output
// tile regardless; this only sets how many share a block, which trades launch
// granularity against per-block LDS.
static int exl3_gemv_rdna_warps(int n_tiles, int k_blocks)
{
    if (n_tiles <= 32) return 16;
    if (n_tiles <= 96 && k_blocks <= 256) return 8;
    return 4;
}

// -----------------------------------------------------------------------------
// Launch
// -----------------------------------------------------------------------------

static void exl3_gemv_rdna_launch
(
    const half* A_ptr,
    const uint16_t* B_ptr,
    void* C_ptr,
    int size_k,
    int size_n,
    int bits,
    int cb,
    bool c_fp32,
    const half* suh_ptr,
    half* A_had_ptr,
    const half* svh_ptr,
    cudaStream_t stream
)
{
    // =========================================================================
    // Input Hadamard transform (SUH)
    // =========================================================================
    const half* gemv_input = A_ptr;

    if (suh_ptr && A_had_ptr)
    {
        int warps_needed = size_k / 128;
        int threads_per_block = 256;
        int blocks = (warps_needed + threads_per_block / 32 - 1) / (threads_per_block / 32);

        hipLaunchKernelGGL
        (
            exl3_gemv_rdna_had_in_kernel,
            dim3(blocks), dim3(threads_per_block), 0, stream,
            A_ptr, A_had_ptr, suh_ptr, size_k
        );

        gemv_input = A_had_ptr;
    }

    // =========================================================================
    // GEMV
    // =========================================================================
    int n_tiles = size_n / 16;
    int k_blocks = size_k / 16;

    int warps_per_block = exl3_gemv_rdna_warps(n_tiles, k_blocks);
    int num_blocks = (n_tiles + warps_per_block - 1) / warps_per_block;

    // Single source for the LDS figure -- see exl3_gemv_kernel_rdna.hip.h
    size_t shared_mem_size = exl3_gemv_smem_bytes(warps_per_block);

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
    // Output Hadamard transform (SVH), in place on C
    // =========================================================================
    if (svh_ptr)
    {
        int warps_needed = size_n / 128;
        int threads_per_block = 256;
        int blocks = (warps_needed + threads_per_block / 32 - 1) / (threads_per_block / 32);

        if (c_fp32)
        {
            hipLaunchKernelGGL
            (
                exl3_gemv_rdna_had_out_float_kernel,
                dim3(blocks), dim3(threads_per_block), 0, stream,
                (const float*) C_ptr, (float*) C_ptr, svh_ptr, size_n
            );
        }
        else
        {
            hipLaunchKernelGGL
            (
                exl3_gemv_rdna_had_out_half_kernel,
                dim3(blocks), dim3(threads_per_block), 0, stream,
                (const half*) C_ptr, (half*) C_ptr, svh_ptr, size_n
            );
        }
    }
}

// -----------------------------------------------------------------------------
// exl3_gemv_try_launch
// -----------------------------------------------------------------------------
// kernel_args is exl3_gemm's argument array, and its layout is part of the
// contract (exl3_gemm.cu builds it immediately above the call):
//
//   0 A   1 B   2 C   3 size_m   4 size_k   5 size_n   6 locks
//   7 suh   8 A_had   9 svh
//
// Each entry points at the caller's own copy of the argument, so the values are
// read back by dereferencing. locks is unused here: this path has no cross-block
// reduction.
//
// *launched_kernel is set to nullptr on success rather than to a kernel pointer.
// It exists so the caller can record graph parameter offsets, and those offsets
// (7, 8, 9) do not exist on a 5-argument kernel -- patching them would write past
// the node's parameter array. The caller must therefore not take this path while
// capturing; exl3_gemm_rdna.hip enforces that with an explicit !graph guard, as
// the fork did at the same call site. nullptr here means a caller that ignores
// that rule fails loudly in Graph::record() instead of corrupting a node.
bool exl3_gemv_try_launch
(
    void** kernel_args,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    bool has_su_sv,
    int device,
    cudaStream_t stream,
    void** launched_kernel,
    bool force
)
{
    // Pre-warm the graph path's parameter block while we are guaranteed to be
    // outside capture (this function is only routed to when !graph). See
    // exl3_gemv_graph_param_block for why allocation cannot happen mid-capture.
    exl3_gemv_graph_prewarm(device);

    int mode = exl3_gemv_env_mode();
    if (!force && mode == 0) return false;
    // Shape-aware envelope (see exl3_gemv_rdna_pays): below the measured
    // profitability threshold the cooperative GEMM is faster, so decline
    // unless forced or in take-everything mode.
    if (!force && mode != 2 && !exl3_gemv_rdna_pays(size_k, size_n)) return false;

    const half*     A_ptr     = *(const half* const*)     kernel_args[0];
    const uint16_t* B_ptr     = *(const uint16_t* const*) kernel_args[1];
    void*           C_ptr     = *(void* const*)           kernel_args[2];
    const half*     suh_ptr   = *(const half* const*)     kernel_args[7];
    half*           A_had_ptr = *(half* const*)           kernel_args[8];
    const half*     svh_ptr   = *(const half* const*)     kernel_args[9];

    if (!exl3_gemv_rdna_eligible(size_m, size_k, size_n, K, suh_ptr, A_had_ptr)) return false;

    exl3_gemv_rdna_launch
    (
        A_ptr, B_ptr, C_ptr, size_k, size_n, K, cb, c_fp32,
        suh_ptr, A_had_ptr, svh_ptr, stream
    );

    if (launched_kernel) *launched_kernel = nullptr;
    return true;
}

// -----------------------------------------------------------------------------
// exl3_gemv -- direct entry point (testing), bound as exl3_gemv
// -----------------------------------------------------------------------------

void exl3_gemv
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    bool mcg,
    bool mul1
)
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
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")

    const half* suh_ptr = (const half*) OPTPTR(suh);
    half* A_had_ptr = (half*) OPTPTR(A_had);
    const half* svh_ptr = (const half*) OPTPTR(svh);
    TORCH_CHECK(suh_ptr && A_had_ptr && svh_ptr, "exl3_gemv requires suh, A_had and svh");

    int size_m = 1;
    int dim = A.dim();
    for (int d = 0; d < dim - 1; ++d) size_m *= A.size(d);
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;
    int K = B.size(2) / 16;

    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    TORCH_CHECK(exl3_gemv_rdna_eligible(size_m, size_k, size_n, K, suh_ptr, A_had_ptr),
        "exl3_gemv: call is not eligible for the RDNA GEMV kernel "
        "(requires size_m == 1, 1 <= K <= 8, size_k % 128 == 0, size_n % 128 == 0)");

    exl3_gemv_rdna_launch
    (
        (const half*) A.data_ptr(),
        (const uint16_t*) B.data_ptr(),
        (void*) C.data_ptr(),
        size_k, size_n, K, cb, c_fp32,
        suh_ptr, A_had_ptr, svh_ptr,
        stream
    );

    cuda_check(cudaPeekAtLastError());
}

// =============================================================================
// Graph-captured GEMV -- work item 4c
// =============================================================================
//
// The plain-launch GEMV above is declined while a graph is capturing, because
// exl3_gemm's six recorded parameter sites (A, B_trellis, C, suh, A_had, svh)
// have no home on a 5-argument kernel. This section gives them one, using the
// pattern proven by the multi-matrix path (exl3_mgemv_rdna.hip): a prologue
// kernel that takes all six as ordinary arguments -- so it is the single node
// Graph::launch() patches -- performs the input rotation, and republishes the
// four pointers the downstream kernels need (B, C, A_had, svh) through a
// per-device Exl3GemvGraphParams block. Graph::launch() patches ONE site per
// caller params entry, so spreading the six GP ids across three kernels (the
// original plan for this work item) would leave the later copies stale; the
// republish is what makes a multi-kernel node sequence patchable at all.
//
// This path requires all three transform tensors (suh, A_had, svh): with them
// present the kernel count is fixed at three, so the site group always lands
// on the prologue. EXL3 quant linears always carry su/sv, so nothing real is
// excluded. Kill switch: EXL3_GEMV_GRAPH=0 (the routing site also honours
// EXL3_GEMV=0/1/2 exactly like the non-graph path).

static bool exl3_gemv_graph_enabled()
{
    const char* env = std::getenv("EXL3_GEMV_GRAPH");
    if (!env) return true;
    return atoi(env) != 0;
}

// allow_alloc must be false while a graph is capturing: hipMalloc synchronizes
// the device, which INVALIDATES an active stream capture ("operation failed due
// to a previous error during capture" on the next call). The eager first pass
// every BC module runs before its capture is where allocation happens -- the
// non-graph try_launch below pre-warms unconditionally -- and a capture that
// somehow arrives on a cold device declines to the cooperative kernel instead.
static Exl3GemvGraphParams* exl3_gemv_graph_param_block(int device, bool allow_alloc)
{
    static std::mutex mtx;
    static Exl3GemvGraphParams* blocks[64] = {};
    if (device < 0 || device >= 64) return nullptr;
    std::lock_guard<std::mutex> lock(mtx);
    if (!blocks[device] && allow_alloc)
    {
        void* p = nullptr;
        if (cudaMalloc(&p, sizeof(Exl3GemvGraphParams)) != cudaSuccess)
        {
            (void) cudaGetLastError();
            return nullptr;
        }
        blocks[device] = (Exl3GemvGraphParams*) p;
    }
    return blocks[device];
}

static void exl3_gemv_graph_prewarm(int device)
{
    (void) exl3_gemv_graph_param_block(device, true);
}

// Prologue: argument order 0-5 is load-bearing -- these indices are the graph
// patch sites, recorded in the cooperative kernel's order so the upstream
// callers' params subsequences match. B, C and svh take no part in this
// kernel's own math.
static __global__
__launch_bounds__(256)
void exl3_gemv_graph_had_in_kernel
(
    const half* __restrict__ A,        // 0: GP_gemm_A
    const uint16_t* __restrict__ B,    // 1: GP_gemm_B_trellis (patch host only)
    void* __restrict__ C,              // 2: GP_gemm_C (patch host only)
    const half* __restrict__ suh,      // 3: GP_gemm_B_suh
    half* __restrict__ A_had,          // 4: GP_gemm_A_had
    const half* __restrict__ svh,      // 5: GP_gemm_B_svh (patch host only)
    Exl3GemvGraphParams* __restrict__ pb,
    const int size_k
)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        pb->B = B;
        pb->C = C;
        pb->A_had = A_had;
        pb->svh = svh;
    }

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= size_k / 128) return;
    int offset = warp_id * 128;
    had_hf_r_128_inner<true, false>
    (
        A + offset,
        A_had + offset,
        suh + offset,
        0.088388347648f  // 1/sqrt(128)
    );
}

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK>
static __global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_graph_dot_kernel
(
    const Exl3GemvGraphParams* __restrict__ pb,
    const int size_k,
    const int size_n
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tile_n = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int n_tiles = size_n / 16;
    if (tile_n >= n_tiles) return;

    extern __shared__ char shared_mem[];
    constexpr int SH_STRIDE = EXL3_GEMV_SH_STRIDE;
    half* sh_b_dq = (half*) shared_mem;
    uint16_t* sh_b_quant = (uint16_t*) (sh_b_dq + WARPS_PER_BLOCK * 16 * SH_STRIDE);
    half* my_sh_b = sh_b_dq + warp_id * 16 * SH_STRIDE;
    uint16_t* my_sh_b_quant = sh_b_quant + warp_id * EXL3_GEMV_SH_QUANT_U16;

    float accum = exl3_gemv_dot_tile<bits, cb>
    (
        pb->A_had, pb->B, size_k, n_tiles, tile_n, lane, my_sh_b, my_sh_b_quant
    );

    if (lane < 16)
    {
        const int out_idx = tile_n * 16 + lane;
        if constexpr (c_fp32)
            ((float*) pb->C)[out_idx] = accum;
        else
            ((half*) pb->C)[out_idx] = __float2half(accum);
    }
}

template <bool c_fp32>
static __global__
__launch_bounds__(256)
void exl3_gemv_graph_had_out_kernel
(
    const Exl3GemvGraphParams* __restrict__ pb,
    const int size_n
)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= size_n / 128) return;
    int offset = warp_id * 128;
    if constexpr (c_fp32)
        had_ff_r_128_inner<false, true>
        (
            ((const float*) pb->C) + offset,
            ((float*) pb->C) + offset,
            pb->svh + offset,
            0.088388347648f  // 1/sqrt(128)
        );
    else
        had_hf_r_128_inner<false, true>
        (
            ((const half*) pb->C) + offset,
            ((half*) pb->C) + offset,
            pb->svh + offset,
            0.088388347648f  // 1/sqrt(128)
        );
}

bool exl3_gemv_graph_try_launch
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
    if (size_m != 1) return false;
    if (K < 1 || K > 8) return false;
    if (size_k % 128 || size_n % 128) return false;
    if (!suh_ptr || !A_had_ptr || !svh_ptr) return false;

    int mode = exl3_gemv_env_mode();
    if (mode == 0) return false;
    if (mode != 2 && !exl3_gemv_rdna_pays(size_k, size_n)) return false;
    if (!exl3_gemv_graph_enabled()) return false;

    // Never allocate here: this function only runs during capture, where
    // hipMalloc would invalidate the graph. Cold block -> cooperative kernel.
    Exl3GemvGraphParams* pb = exl3_gemv_graph_param_block(device, false);
    if (!pb) return false;

    // 1. Prologue + input rotation
    {
        int blocks = CEIL_DIVIDE((size_k / 128) * 32, 256);
        hipLaunchKernelGGL
        (
            exl3_gemv_graph_had_in_kernel,
            dim3(blocks), dim3(256), 0, stream,
            A_ptr, B_ptr, C_ptr, suh_ptr, A_had_ptr, svh_ptr, pb, size_k
        );
        if (graph)
        {
            void* k = (void*) exl3_gemv_graph_had_in_kernel;
            graph->record_param(k, GP_gemm_A, 0);
            graph->record_param(k, GP_gemm_B_trellis, 1);
            graph->record_param(k, GP_gemm_C, 2);
            graph->record_param(k, GP_gemm_B_suh, 3);
            graph->record_param(k, GP_gemm_A_had, 4);
            graph->record_param(k, GP_gemm_B_svh, 5);
            graph->record_param(k, GP_end, 0);
        }
    }

    // 2. GEMV -- same wave selection as the non-graph path
    {
        int n_tiles = size_n / 16;
        int k_blocks = size_k / 16;
        int warps_per_block = exl3_gemv_rdna_warps(n_tiles, k_blocks);
        int num_blocks = (n_tiles + warps_per_block - 1) / warps_per_block;
        size_t smem = exl3_gemv_smem_bytes(warps_per_block);

        #define LAUNCH_GGEMV_WAVES(bits_val, codebook, warps) \
            if (c_fp32) { \
                hipLaunchKernelGGL( \
                    (exl3_gemv_graph_dot_kernel<bits_val, true, codebook, warps>), \
                    dim3(num_blocks), dim3(warps * 32), smem, stream, \
                    pb, size_k, size_n); \
            } else { \
                hipLaunchKernelGGL( \
                    (exl3_gemv_graph_dot_kernel<bits_val, false, codebook, warps>), \
                    dim3(num_blocks), dim3(warps * 32), smem, stream, \
                    pb, size_k, size_n); \
            }

        #define LAUNCH_GGEMV_CB(bits_val, codebook) \
            switch (warps_per_block) { \
                case 4:  LAUNCH_GGEMV_WAVES(bits_val, codebook, 4);  break; \
                case 8:  LAUNCH_GGEMV_WAVES(bits_val, codebook, 8);  break; \
                case 16: LAUNCH_GGEMV_WAVES(bits_val, codebook, 16); break; \
            }

        #define LAUNCH_GGEMV(bits_val) \
            switch (cb) { \
                case 0: LAUNCH_GGEMV_CB(bits_val, 0); break; \
                case 1: LAUNCH_GGEMV_CB(bits_val, 1); break; \
                case 2: LAUNCH_GGEMV_CB(bits_val, 2); break; \
            }

        switch (K) {
            case 1: LAUNCH_GGEMV(1); break;
            case 2: LAUNCH_GGEMV(2); break;
            case 3: LAUNCH_GGEMV(3); break;
            case 4: LAUNCH_GGEMV(4); break;
            case 5: LAUNCH_GGEMV(5); break;
            case 6: LAUNCH_GGEMV(6); break;
            case 7: LAUNCH_GGEMV(7); break;
            case 8: LAUNCH_GGEMV(8); break;
        }

        #undef LAUNCH_GGEMV
        #undef LAUNCH_GGEMV_CB
        #undef LAUNCH_GGEMV_WAVES
    }

    // 3. Output rotation
    {
        int blocks = CEIL_DIVIDE((size_n / 128) * 32, 256);
        if (c_fp32)
            hipLaunchKernelGGL
            (
                exl3_gemv_graph_had_out_kernel<true>,
                dim3(blocks), dim3(256), 0, stream,
                pb, size_n
            );
        else
            hipLaunchKernelGGL
            (
                exl3_gemv_graph_had_out_kernel<false>,
                dim3(blocks), dim3(256), 0, stream,
                pb, size_n
            );
    }

    return true;
}
