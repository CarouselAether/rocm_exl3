// =============================================================================
// exl3_gemm host dispatch for RDNA -- generated from upstream v1.4.1
// quant/exl3_gemm.cu. Include swaps, plus three behavioural changes to the
// launch (EXL3_RDNA_SMEM, EXL3_RDNA_GEMV_GRAPH, EXL3_RDNA_AUTOTUNE_GRAPH,
// each documented at its site). Nothing else.
//
// Synced to v1.4.1 by three-way merge (base v1.3.0, ours the sibling, theirs
// v1.4.1) rather than regeneration, because the autotune-graph guard re-indents
// a block upstream still owns and a sed-style rewrite cannot express that.
// v1.4.1's additions here are the per-matrix output width/pointer lists
// (size_n_list / c_ptrs) threaded into mgemm, the A_had scratch size check, and
// the autotune-hash clamp; all merged clean.
//
// EXL3_RDNA_SMEM covers the four launch-configuration sites. The four autotuner
// sites still pass upstream's SMEM_MAX, unchanged from before this rebase.
//
// A sibling is required because exl3_gemm.cu includes exl3_gemm_kernel.cuh,
// which reaches ptx.cuh through exl3_gemm_inner.cuh by relative quoted include.
// =============================================================================

#include <cuda_fp16.h>
#include "../../quant/exl3_gemm.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include <cstdlib>
#include "../../util.h"
#include "../../util.cuh"
#include "exl3_gemm_kernel_rdna.hip.h"
#include "exl3_kernel_map_rdna.hip.h"
#include "exl3_mgemv_rdna.hip.h"
#include "../../quant/exl3_devctx.cuh"
#include "../../quant/exl3_gemv.cuh"
#include "../../quant/exl3_gemv_int8.cuh"
#include "../../quant/coop_autotune.cuh"
#include <set>
#include <vector>


// -----------------------------------------------------------------------------
// EXL3_RDNA_SMEM: launch with the shape's actual LDS requirement
// -----------------------------------------------------------------------------
// Upstream passes SMEM_MAX as the dynamic shared-memory size for every launch.
// On a 90 KB part that is harmless slack. On RDNA it is not: LDS is 64 KB, so
// requesting SMEM_MAX reserves the entire workgroup allocation and pins
// residency at one block per WGP no matter how small the tile actually is.
//
// These kernels are launched cooperatively, and a grid.sync() deadlocks if the
// grid exceeds what can be co-resident -- which is exactly the failure the
// original RDNA port hit and worked around by routing size_m == 1 away from
// this kernel entirely ("uses ~64KB LDS, hits RDNA WGP block-per-pair limits,
// grid.sync stalls").
//
// Passing the real figure fixes the cause rather than the symptom. Measured:
// the N=128 shape needs ~19 KB with the hadamard output path, so three blocks
// fit per WGP instead of one. exl3_gemm_smem_bytes() in the RDNA kernel map
// mirrors the kernel's own allocation and is the single source for it.
#define EXL3_RDNA_SMEM(shape_idx) exl3_gemm_smem_bytes(K, (shape_idx))

// -----------------------------------------------------------------------------
// EXL3_RDNA_SMS_MULT: cooperative grid width in CUs rather than WGPs
// -----------------------------------------------------------------------------
// `get_num_sms()` returns cudaDeviceProp::multiProcessorCount, which on RDNA is
// the **WGP** count -- 20 on a 40-CU gfx1151 -- while an NVIDIA SM is the unit
// that actually hosts a workgroup. The cooperative grid is sized from it
// (`block_grid(num_sms, 1, concurrency)` below), so a part that can hold one
// workgroup per CU is asked for half that many.
//
// Measured on Laguna MoE decode, exl3_mgemm at m == 1 with 10 experts: the grid
// comes out (2, 1, 10) = 20 workgroups of 256 threads = 160 waves. gfx1151 has 80
// SIMD32 (20 WGPs -> 40 CUs -> 2 SIMDs each) and RDNA3 allows 16 wave32 per SIMD,
// so that is 2/16 = 12.5% -- and rocprofv3 measures OccupancyPercent 11.7-12.2%
// with MemUnitBusy only 26-32%. Occupancy is identical at VGPR 144, 248 and 256, so it is the grid
// width that pins it, not register pressure. Cooperative launch cannot
// oversubscribe -- grid.sync() requires every block co-resident -- so the kernel
// has no way to recover the missing memory-level parallelism.
//
// This is deliberately an env knob defaulting to 1 (no change) rather than a
// silent 2: an oversubscribed cooperative grid is *refused* by the runtime, and
// how many workgroups are genuinely co-resident depends on the shape's LDS and
// VGPR use. Measure per shape before changing the default.
//
//   EXL3_RDNA_SMS_MULT=1  (default) upstream behaviour, grid sized in WGPs
//   EXL3_RDNA_SMS_MULT=2            one workgroup per CU
static int exl3_rdna_sms_mult()
{
    static int cached = -1;
    if (cached < 0)
    {
        const char* env = std::getenv("EXL3_RDNA_SMS_MULT");
        int v = env ? atoi(env) : 1;
        cached = (v >= 1 && v <= 8) ? v : 1;
    }
    return cached;
}

int exl3_gemm_tilesize_k_g[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n_g[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim_g[] = {EXL3_GEMM_BLOCKDIM};

/*
EXL3 matmul, A @ B -> C

- A: row-major A tensor, shape (m, k), dtype float16, contiguous
- B: EXL3-quantized B tensor, shape (k//16, n//16, 16*K), dtype uint16
- C: empty row-major C tensor, shape (m, n), dtype float16 or float32, contiguous. Does not need to be zero-initialized
- suh: optional, packed input scales/flips, shape (k//16), dtype float16
- A_had: required if suh given, may be reference to A, temporary storage for input transform, size and dtype as A
- svh: optional, packed output scales/flips, shape (n//16), dtype float16

limitations:
- k % 16 == 0
- n % 128 == 0
*/

std::set<void*> kernel_attr_set[MAX_DEVICES] = {};

uint64_t roundup_pow2(uint64_t x)
{
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	x |= x >> 32;
    return x + 1;
}

uint64_t gemm_autotune_hash
(
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int device,
    int cc,
    int max_num_sms,
    int cb
)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&] (uint64_t v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) MIN(roundup_pow2(size_m), 16));
    mix((uint64_t) size_k);
    mix((uint64_t) size_n);
    mix((uint64_t) K);
    mix(c_fp32 ? 1ull : 0ull);
    mix((uint64_t) device);
    mix((uint64_t) cc);
    mix((uint64_t) max_num_sms);
    mix((uint64_t) cb);
    return h;
}

uint64_t mgemm_autotune_hash
(
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int device,
    int cc,
    int max_num_sms,
    int cb,
    int bszm_in,
    int bszm_out
)
{
    uint64_t h = gemm_autotune_hash(size_m, size_k, size_n, K, c_fp32, device, cc, max_num_sms, cb);
    auto mix = [&] (uint64_t v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) MIN(bszm_in, 24));
    mix((uint64_t) MIN(bszm_out, 24));
    return h;
}

int exl3_gemm_gr
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    // TORCH_CHECK_SHAPES(A, 0, C, 0, 1);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);

    // Get SU, optionally
    const half* suh_ptr = (const half*) OPTPTR(suh);
    half* A_had_ptr = nullptr;
    if (suh_ptr)
    {
        // TORCH_CHECK_SHAPES(suh.value(), 0, A, 1, 1);
        A_had_ptr = (half*) OPTPTR(A_had);
        // TORCH_CHECK(A_had_ptr, "Must supply A_had with suh");
        // TORCH_CHECK_SHAPES_FULL(A_had.value(), A);
    }

    // Get SV, optionally
    const half* svh_ptr = (const half*) OPTPTR(svh);
    // if (svh_ptr)
        // TORCH_CHECK_SHAPES(svh.value(), 0, B, 1, 16);

    // Device properties
    int device;
    cudaGetDevice(&device);
    int num_sms = force_num_sms ? force_num_sms : DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);

    // Dispatch
    int K = B.size(2) / 16;
    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();

    int size_m = 1;
    int dim = A.dim();
    for (int d = 0; d < dim - 1; ++d) size_m *= A.size(d);
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;

    // Select kernel
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    // Experimental fused int8-activation GEMV path (EXL3_INT8_GEMV=1) for mul1 tensors. Rows are
    // processed as successive GEMV launches, so this is only sensible for small m (the reconstruct
    // threshold keeps m <= 144 in practice). Not graph-capturable yet; graphed callers fall through
    // to the regular kernel.
    if (mul1 && exl3_gemv_int8_enabled())
    {
        if (exl3_gemv_int8(A, B, C, suh, A_had, svh, stream, graph))
            return 0;
    }

    int block_dim;
    int shape_idx;
    fp_exl3_gemm_kernel kernel;

    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr
    };

    auto add_graph_args = [&](void* kernel_ptr)
    {
        if (graph)
        {
            graph->record_param(kernel_ptr, GP_gemm_A, 0);
            graph->record_param(kernel_ptr, GP_gemm_B_trellis, 1);
            graph->record_param(kernel_ptr, GP_gemm_C, 2);
            graph->record_param(kernel_ptr, GP_gemm_B_suh, 7);
            graph->record_param(kernel_ptr, GP_gemm_A_had, 8);
            graph->record_param(kernel_ptr, GP_gemm_B_svh, 9);
            graph->record_param(kernel_ptr, GP_end, 0);
        }
    };

    // -------------------------------------------------------------------------
    // GEMV routing at m == 1, in and out of graph capture
    // -------------------------------------------------------------------------
    // Out of capture: the fork-carried plain path below (exl3_gemv_try_launch),
    // which records no graph sites (*launched_kernel = nullptr contract).
    //
    // Under capture: the fork's `!graph` guard used to decline everything --
    // correct then (the 5-argument kernel has no home for add_graph_args'
    // offsets 7/8/9, and in the fork's era decode was not captured), but decode
    // on v1.4.1 is ~100% captured, so the guard silently cost the whole GEMV
    // path at decode. exl3_gemv_graph_try_launch is the capture-safe form: a
    // prologue kernel hosts all six GP_gemm_* patch sites and republishes the
    // downstream pointers through a device parameter block (the pattern proven
    // by exl3_mgemv_rdna.hip; Graph::launch() patches one site per caller
    // params entry, so multi-kernel paths must concentrate their sites on one
    // node). Both routes share the measured profitability rule
    // (exl3_gemv_rdna_pays) and the EXL3_GEMV env modes.
    if (force_shape_idx <= 0 && force_num_sms <= 0)
    {
        if (!graph)
        {
            void* gemv_kernel = nullptr;
            if (exl3_gemv_try_launch
            (
                kernelArgs, size_m, size_k, size_n, K, cb, c_fp32,
                suh_ptr && A_had_ptr && svh_ptr,
                device, stream, &gemv_kernel, false
            ))
            {
                add_graph_args(gemv_kernel);
                cuda_check(cudaPeekAtLastError());
                return 90;
            }
        }
        else if (exl3_gemv_graph_try_launch
        (
            A_ptr, B_ptr, C_ptr, suh_ptr, (half*) A_had_ptr, svh_ptr,
            size_m, size_k, size_n, K, cb, c_fp32,
            device, stream, graph
        ))
        {
            cuda_check(cudaPeekAtLastError());
            return 91;
        }
    }

    bool autotune = force_shape_idx <= 0 && force_num_sms <= 0;
    if (autotune)
    {
        uint64_t autotune_key = gemm_autotune_hash(MAX(size_m, 2), size_k, size_n, K, c_fp32, device, cc, num_sms, cb);
        CoopAutotuneLaunch tuned;
        if (CoopKernelAutotuner::launch_locked(autotune_key, kernelArgs, SMEM_MAX, stream, &tuned))
        {
            add_graph_args((void*) tuned.kernel);
            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
        // -------------------------------------------------------------------------
        // EXL3_RDNA_AUTOTUNE_GRAPH: never measure while a graph is capturing
        // -------------------------------------------------------------------------
        // The autotuner times candidates, which means cudaStreamSynchronize at
        // coop_autotune.cu:445. That is illegal mid-capture and aborts with
        // "operation not permitted when stream is capturing".
        //
        // Upstream already guards the *mgemm* autotuner this way (see the
        // `if (!graph)` in exl3_mgemm_gr below) but not this one, because on the
        // CUDA path the cache is warm by the time anything is captured -- the
        // launch_locked fast path above hits and the measuring code is never
        // reached. On this backend the first capture arrived with a cold cache.
        //
        // launch_locked stays available during capture; only the measuring is
        // skipped. A capture with a cold cache falls through to the plain
        // select_exl3_gemm_kernel path below, which is a single launch with no
        // timing and no sync, and the autotuner will measure it later on a
        // non-captured call.
        if (!graph)
        {
            std::vector<CoopAutotuneCandidate> candidates;
            for (int candidate_shape_idx = 1; candidate_shape_idx <= EXL3_GEMM_NUM_SHAPES; ++candidate_shape_idx)
            {
                if (!exl3_gemm_shape_compat(candidate_shape_idx, size_m, size_k, size_n, K)) continue;

                fp_exl3_gemm_kernel candidate_kernel = get_gemm_kernel_ptr(K, candidate_shape_idx, c_fp32, cb);
                if (!candidate_kernel) continue;

                int tilesize_k = exl3_gemm_tilesize_k_g[candidate_shape_idx];
                int tilesize_n = exl3_gemm_tilesize_n_g[candidate_shape_idx];
                int max_slices = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
                int max_candidate_sms = MAX(MIN(max_slices, num_sms), 1);

                candidates.push_back
                ({
                    (void*) candidate_kernel,
                    exl3_gemm_blockdim_g[candidate_shape_idx],
                    max_candidate_sms,
                    1,
                    max_candidate_sms,
                    candidate_shape_idx
                });
            }
            TORCH_CHECK(!candidates.empty(), "exl3_gemm autotune: no compatible kernel shapes");

            tuned = CoopKernelAutotuner::launch(autotune_key, candidates, kernelArgs, SMEM_MAX, stream, (size_t) size_k * size_n);
            add_graph_args((void*) tuned.kernel);
            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
    }

    kernel = select_exl3_gemm_kernel
    (
        cc, size_m, size_k, size_n, K, c_fp32,
        force_shape_idx, &block_dim, &shape_idx,
        &num_sms, cb
    );
    if (!kernel) return 0;

    // Launch
    if (kernel_attr_set[device].find((void*) kernel) == kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, EXL3_RDNA_SMEM(shape_idx));
        kernel_attr_set[device].insert((void*) kernel);
        cuda_check(cudaPeekAtLastError());
    }
    cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        num_sms,
        block_dim,
        kernelArgs,
        EXL3_RDNA_SMEM(shape_idx),
        stream
    );
    add_graph_args((void*) kernel);

    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}

int exl3_gemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms
)
{
    return exl3_gemm_gr
    (
        A,
        B,
        C,
        suh,
        A_had,
        svh,
        force_shape_idx,
        mcg,
        mul1,
        force_num_sms,
        nullptr
    );
}

/*
EXL3 batched/multi-matrix matmul.

This is not a conventional batched A @ B. B, suh and svh are CUDA int64
tensors containing device addresses (one address per quantized matrix), rather
than the matrix data themselves. Entry q of each table describes one linear:

    B[q]   -> EXL3 trellis, logically (k / 16, n / 16, 16 * K) uint16
    suh[q] -> packed input scales/flips, logically (k / 16) float16
    svh[q] -> packed output scales/flips, logically (n / 16) float16

A is contiguous float16 [a_batches, m, k], C is contiguous float16 or
float32 [c_batches, m, n], and A_had is float16 scratch with room for every
active matrix. The kernel applies the input Hadamard transform into A_had,
performs the selected EXL3 matmul, then applies the output transform.

The active matrix/output slot j selects q = indices[j] when indices is given,
or q = j otherwise. This supports the following modes:

- Multiple inputs and outputs: A[j] @ B[q] -> C[j].
- One input, multiple outputs: when a_batches == 1, A[0] is broadcast and
  transformed separately for each selected B[q], producing C[j]. This is used
  for e.g. fused gate/up projections and MoE expert fan-out.
- Indexed matrices: indices is a contiguous int64 [*, num_indices] tensor;
  the kernel reads its first num_indices entries as q values. Negative indices
  skip that slot.
- Weighted MoE reduction: weights is a float16 tensor parallel to indices.
  Each transformed result is multiplied by weights[j], then all active C[j]
  are summed into C[0]. C therefore also serves as per-expert scratch; only
  C[0] is the reduced result.
- Expert-range filtering: with min_index >= 0, selections outside
  [min_index, max_index) are removed and retained indices are rebased by
  min_index. This allows B/suh/svh to be local pointer tables for an expert
  shard. At num_tokens == 1 the retained indices (and their weights) are
  compacted; at num_tokens > 1 out-of-range slots are instead masked to -1 in
  place, preserving the per-token slot groups the final reduction depends on
  (and, with bszm_in > 1, the slot -> input-row correspondence).

Without weights, every active C[j] is a separate output. The active slot count
is max(a_batches, c_batches), capped to num_indices when indices is present.

Limitations: k must be divisible by 16 and n by 128. Range filtering supports
at most 128 slots (the kernel's index-compaction capacity).
*/

int exl3_mgemm_gr
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    int K,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int min_index,
    int max_index,
    int force_num_sms,
    Graph* graph,
    int num_tokens,
    const c10::optional<at::Tensor>& size_n_list,
    const c10::optional<at::Tensor>& c_ptrs
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    // num_tokens > 1 with expert-range filtering (min_index >= 0) uses position-preserving
    // masking in the kernel (out-of-range slots marked -1 in place) instead of index
    // compaction, so the grouped reduction's fixed per-token slot runs stay intact

    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kLong);
    TORCH_CHECK_DTYPE(suh, kLong);
    TORCH_CHECK_DTYPE(svh, kLong);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);
    TORCH_CHECK_DIM(A, 3);
    TORCH_CHECK_DIM(B, 1);
    TORCH_CHECK_DIM(suh, 1);
    TORCH_CHECK_DIM(svh, 1);
    TORCH_CHECK_DIM(C, 3);

    TORCH_CHECK_SHAPES(A, 1, C, 1, 1);
    TORCH_CHECK_SHAPES(B, 0, suh, 0, 1);
    TORCH_CHECK_SHAPES(B, 0, svh, 0, 1);

    int bsz = A.size(1);
    int bszm_in = A.size(0);
    int bszm_out = C.size(0);

    // Per-matrix output widths/pointers (uniform-width callers pass neither): C then only
    // provides the dtype and the max width (locks/shape sizing); outputs go to c_ptrs
    const int* size_n_list_ptr = nullptr;
    void** c_list_ptr = nullptr;
    if (size_n_list)
    {
        TORCH_CHECK(c_ptrs, "exl3_mgemm: size_n_list requires c_ptrs");
        TORCH_CHECK_DTYPE(size_n_list.value(), kInt);
        TORCH_CHECK_DTYPE(c_ptrs.value(), kLong);
        TORCH_CHECK(num_tokens == 1 && min_index < 0 && !weights,
                    "exl3_mgemm: per-matrix widths incompatible with multi-token/filtering/weights");
        size_n_list_ptr = (const int*) size_n_list.value().data_ptr();
        c_list_ptr = (void**) c_ptrs.value().data_ptr();
        bszm_out = (int) c_ptrs.value().size(0);
    }
    int bszm = MAX(bszm_in, bszm_out);

    // The kernel writes one hadamard-transformed input slab PER MATRIX (A_had + j * m * k);
    // an undersized scratch is silent OOB corruption (found the hard way)
    TORCH_CHECK(A_had.numel() >= (int64_t) bszm * A.size(1) * A.size(2),
                "exl3_mgemm: A_had must hold bszm * m * k elements");

    const int64_t* indices_ptr = (const int64_t*) OPTPTR(indices);
    const half* weights_ptr = (const half*) OPTPTR(weights);

    if (indices)
    {
        TORCH_CHECK_DIM(indices.value(), 2);
        int num_indices = indices.value().size(1);
        TORCH_CHECK(num_indices <= bszm_in || num_indices <= bszm_out, "mgemm: too many indices for tensor batch");
        if (bszm_in > num_indices) bszm_in = num_indices;
        if (bszm_out > num_indices) bszm_out = num_indices;
    }

    if (weights)
    {
        TORCH_CHECK_DIM(weights.value(), 2);
    }

    int size_m = A.size(1);
    int size_k = A.size(2);
    int size_n = C.size(2);

    // Device properties
    int device;
    cudaGetDevice(&device);
    // EXL3_RDNA_SMS_MULT: see the note at the top -- get_num_sms() counts WGPs, not CUs
    int total_sms = DevCtx::instance().get_num_sms(device) * exl3_rdna_sms_mult();
    int num_sms = force_num_sms ? force_num_sms : total_sms;
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);

    // Dispatch
    const half* A_ptr = (const half*) A.data_ptr();
    const uintptr_t* B_ptr_ptr = (const uintptr_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();
    const half* A_had_ptr = (const half*) A_had.data_ptr();
    const uintptr_t* suh_ptr_ptr = (const uintptr_t*) suh.data_ptr();
    const uintptr_t* svh_ptr_ptr = (const uintptr_t*) svh.data_ptr();

    // Select kernel
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    // m == 1 decode: route to the plain-launch multi-matrix GEMV, which
    // oversubscribes where the cooperative grid cannot (see
    // exl3_mgemv_rdna.hip). Diagnostic overrides pin the cooperative path so
    // shape/grid sweeps keep measuring the kernel they name; EXL3_MGEMV=0 is
    // the runtime kill switch. Falls through when ineligible.
    if (size_m == 1 && force_shape_idx <= 0 && force_num_sms <= 0)
    {
        if (exl3_mgemv_try_launch
        (
            A_ptr, B_ptr_ptr, C_ptr, suh_ptr_ptr, (half*) A_had_ptr, svh_ptr_ptr,
            indices_ptr, weights_ptr,
            size_m, size_k, size_n, K, cb, c_fp32,
            bszm_in, bszm_out, min_index, max_index, num_tokens,
            size_n_list_ptr, c_list_ptr,
            device, stream, graph
        ))
        {
            cuda_check(cudaPeekAtLastError());
            return 0;
        }
    }

    int shape_idx;
    int block_dim;
    fp_exl3_mgemm_kernel kernel;
    int concurrency;

    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr_ptr,
        (void*)& indices_ptr,
        (void*)& weights_ptr,
        (void*)& bszm_in,
        (void*)& bszm_out,
        (void*)& min_index,
        (void*)& max_index,
        (void*)& num_tokens,
        (void*)& size_n_list_ptr,
        (void*)& c_list_ptr
    };

    auto add_graph_args = [&](void* kernel_ptr)
    {
        if (graph)
        {
            graph->record_param(kernel_ptr, GP_mgemm_A, 0);
            graph->record_param(kernel_ptr, GP_mgemm_C, 2);
            graph->record_param(kernel_ptr, GP_mgemm_indices, 10);
            graph->record_param(kernel_ptr, GP_mgemm_weights, 11);
            graph->record_param(kernel_ptr, GP_end, 0);
        }
    };

    bool autotune = force_shape_idx <= 0 && force_num_sms <= 0;
    if (autotune)
    {
        uint64_t autotune_key = mgemm_autotune_hash
        (
            size_m, size_k, size_n, K, c_fp32, device, cc, total_sms, cb, bszm_in, bszm_out
        );

        CoopAutotuneLaunch tuned;
        if (CoopKernelAutotuner::launch_locked(autotune_key, kernelArgs, SMEM_MAX, stream, &tuned))
        {
            add_graph_args((void*) tuned.kernel);
            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
        if (!graph)
        {
            std::vector<CoopAutotuneCandidate> candidates;
            for (int candidate_shape_idx = 1; candidate_shape_idx <= EXL3_GEMM_NUM_SHAPES; ++candidate_shape_idx)
            {
                if (!exl3_gemm_shape_compat(candidate_shape_idx, size_m, size_k, size_n, K)) continue;

                fp_exl3_mgemm_kernel candidate_kernel = get_mgemm_kernel_ptr(K, candidate_shape_idx, c_fp32, cb);
                if (!candidate_kernel) continue;

                int tilesize_k = exl3_gemm_tilesize_k_g[candidate_shape_idx];
                int tilesize_n = exl3_gemm_tilesize_n_g[candidate_shape_idx];
                int max_slices = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
                int max_candidate_sms = MAX(MIN(max_slices, total_sms), 1);

                candidates.push_back
                ({
                    (void*) candidate_kernel,
                    exl3_gemm_blockdim_g[candidate_shape_idx],
                    max_candidate_sms,
                    bszm,
                    total_sms,
                    candidate_shape_idx
                });
            }
            TORCH_CHECK(!candidates.empty(), "exl3_mgemm autotune: no compatible kernel shapes");

            tuned = CoopKernelAutotuner::launch(autotune_key, candidates, kernelArgs, SMEM_MAX, stream, (size_t) size_k * size_n * bszm);
            add_graph_args((void*) tuned.kernel);

            // DBGI10(size_m, size_k, size_n, K, bszm_in, bszm_out, tuned.tag, tuned.block_dim, tuned.num_sms, tuned.concurrency);

            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
    }

    kernel = select_exl3_mgemm_kernel
    (
        cc, size_m, size_k, size_n, K, c_fp32,
        force_shape_idx, &block_dim, &shape_idx,
        &num_sms, cb, bszm_in, bszm_out
    );
    int tilesize_k = exl3_gemm_tilesize_k_g[shape_idx];
    int tilesize_n = exl3_gemm_tilesize_n_g[shape_idx];
    int tiles = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
    if (force_num_sms > 0)
    {
        // Diagnostic override: honoured exactly as given, unlike the sized
        // grid below. An oversubscribed cooperative grid is refused by the
        // runtime with "too many blocks in cooperative launch", which is the
        // informative failure mode a sweep wants. (This branch used to be
        // dead: `num_sms = tiles` overwrote the forced value, so sweeping
        // --sms changed nothing and looked like evidence the grid size did
        // not matter.)
        num_sms = force_num_sms;
        concurrency = MAX(MIN(total_sms / num_sms, bszm), 1);
    }
    else
    {
        num_sms = tiles;
        if (num_sms * bszm > total_sms) num_sms = MAX(total_sms / bszm, 1);
        if (num_sms <= total_sms && tiles / num_sms > 48) num_sms = MIN(total_sms, num_sms * 2);
        concurrency = MIN(total_sms / num_sms, bszm);
    }

    // DBGI10(size_m, size_k, size_n, K, bszm_in, bszm_out, shape_idx, block_dim, num_sms, concurrency);

    // Launch bigger grid if possible
    dim3 block_grid(num_sms, 1, concurrency);

    // Launch
    if (kernel_attr_set[device].find((void*) kernel) == kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, EXL3_RDNA_SMEM(shape_idx));
        kernel_attr_set[device].insert((void*) kernel);
    }

    cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        block_grid,
        block_dim,
        kernelArgs,
        EXL3_RDNA_SMEM(shape_idx),
        stream
    );
    add_graph_args((void*) kernel);

    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}

int exl3_mgemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    int K,
    int force_shape_idx,
    uint32_t mcg_mult,
    uint32_t mul1_mult,
    int min_index,
    int max_index,
    int force_num_sms,
    int num_tokens,
    const c10::optional<at::Tensor>& size_n_list,
    const c10::optional<at::Tensor>& c_ptrs
)
{
    return exl3_mgemm_gr
    (
        A,
        B,
        C,
        suh,
        A_had,
        svh,
        indices,
        weights,
        K,
        force_shape_idx,
        mcg_mult,
        mul1_mult,
        min_index,
        max_index,
        force_num_sms,
        nullptr,
        num_tokens,
        size_n_list,
        c_ptrs
    );
}
