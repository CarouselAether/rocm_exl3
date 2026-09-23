// =============================================================================
// hgemm_rdna.hip -- hgemm sibling for ROCm: narrow-output GEMV kernel at m <= 8
// =============================================================================
//
// Generated from hgemm.cu (which stays byte-identical to upstream; setup.py
// excludes it from the ROCm build in favor of this file). The diff against
// upstream is confined to the two kernels and hgemm_narrow_try() below plus its
// one call site at the top of hgemm_gemmex_impl(); everything else is a verbatim
// copy so the file can be regenerated mechanically on rebase and audited by diff.
//
// WHY (measured 2026-09-23, gfx1151, rocm_tools/hgemm_narrow_probe.py):
// BC_Attention runs the headwise attention gate through hgemm_gr once per layer
// per token: a (1 x 3072) @ (3072 x 48) fp16 product. Through hipblasGemmEx,
// rocBLAS's legacy kernels answer it with ONE 128x128 tile workgroup walking K
// serially: ~63-74 us for a 295 KB matrix, ~50x its bandwidth bound, and 48
// layers made it 15% of Laguna's decode GPU time. Wide products are fine on the
// same path (N = 12288 at m == 1 streams at ~200 GB/s), so the pathology is
// narrow N, and it is rocBLAS's kernel selection: ROCBLAS_USE_HIPBLASLT=1 fixes
// the narrow shapes (10-22 us) but costs prefill 13-17% (hipBLASLt loses to
// rocBLAS at m >= 128 and at wide N on this part), and there is no per-call
// backend switch. A plain HIP GEMV for the one bad regime is the fix that
// leaves everything else on rocBLAS: two kernel launches, split-K partials
// into the DevCtx workspace then a reduce into C, no allocations -- so it is
// also safe inside HIP graph capture, where the ATen-based attempt could not
// go and where the captured gate node otherwise stays on the slow kernel.
//
// Regime: m <= HGEMM_NARROW_MAX_M (8) and N <= HGEMM_NARROW_MAX_N (256), any K,
// fp16 or fp32 C. EXL3_RDNA_HGEMM_NARROW_N overrides the N cutoff (0 disables).
// =============================================================================

#include <cuda_fp16.h>
#include "hgemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"
#include <limits>

/*

Row-major matmul using cuBLAS, a @ b -> c
- if c is float16, operation is float16 @ float16 -> float16 (float16 accumulate)
- if c is float32, operation is float16 @ float16 -> float32 (float32 accumulate)
*/

using bfloat16 = __nv_bfloat16;


// ---------------------------------------------------------------------------
// ROCm sibling: narrow-output GEMV (see file header)
// ---------------------------------------------------------------------------
#define HGEMM_NARROW_MAX_M 8
#define HGEMM_NARROW_MAX_N 256
#define HGEMM_NARROW_THREADS 256
#define HGEMM_NARROW_MAX_SPLITS 64
#define HGEMM_NARROW_MIN_K_PER_SPLIT 64

// Stage 1, one block per K-split: thread t owns column (t % n) and row-lane (t / n);
// the lanes stride over the split's rows and are reduced through LDS. A is tiny (m x K)
// and every thread of a lane group reads the same element, a broadcast load.
__global__ void hgemm_narrow_partial_kernel
(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ ws,
    int size_m,
    int size_k,
    int size_n,
    int k_per_split
)
{
    __shared__ float red[HGEMM_NARROW_MAX_M][HGEMM_NARROW_THREADS];
    const int split = blockIdx.x;
    const int k0 = split * k_per_split;
    const int k1 = min(size_k, k0 + k_per_split);
    const int lanes = HGEMM_NARROW_THREADS / size_n;
    const int t = threadIdx.x;
    const int col = t % size_n;
    const int lane = t / size_n;

    float acc[HGEMM_NARROW_MAX_M];
    #pragma unroll
    for (int r = 0; r < HGEMM_NARROW_MAX_M; ++r) acc[r] = 0.0f;

    if (lane < lanes)
    {
        for (int k = k0 + lane; k < k1; k += lanes)
        {
            float bv = __half2float(b[(int64_t) k * size_n + col]);
            #pragma unroll
            for (int r = 0; r < HGEMM_NARROW_MAX_M; ++r)
                if (r < size_m) acc[r] += __half2float(a[(int64_t) r * size_k + k]) * bv;
        }
    }
    #pragma unroll
    for (int r = 0; r < HGEMM_NARROW_MAX_M; ++r) red[r][t] = acc[r];
    __syncthreads();

    if (t < size_n)
    {
        for (int r = 0; r < size_m; ++r)
        {
            float s = 0.0f;
            for (int l = 0; l < lanes; ++l) s += red[r][l * size_n + t];
            ws[((int64_t) split * size_m + r) * size_n + t] = s;
        }
    }
}

__device__ __forceinline__ void hgemm_narrow_store(half* c, float v) { *c = __float2half(v); }
__device__ __forceinline__ void hgemm_narrow_store(float* c, float v) { *c = v; }

// Stage 2: sum the split partials and write C with its row stride, in its dtype
template <typename T>
__global__ void hgemm_narrow_reduce_kernel
(
    const float* __restrict__ ws,
    T* __restrict__ c,
    int size_m,
    int size_n,
    int splits,
    int64_t c_stride_m
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size_m * size_n) return;
    const int r = i / size_n;
    const int col = i % size_n;
    float s = 0.0f;
    for (int sp = 0; sp < splits; ++sp) s += ws[((int64_t) sp * size_m + r) * size_n + col];
    hgemm_narrow_store(c + (int64_t) r * c_stride_m + col, s);
}

static int hgemm_narrow_n_cutoff()
{
    static int v = -1;
    if (v < 0)
    {
        const char* e = getenv("EXL3_RDNA_HGEMM_NARROW_N");
        v = (e && *e) ? atoi(e) : HGEMM_NARROW_MAX_N;
        if (v < 0) v = 0;
        if (v > HGEMM_NARROW_MAX_N) v = HGEMM_NARROW_MAX_N;
    }
    return v;
}

static bool hgemm_narrow_try
(
    const half* a_ptr,
    const half* b_ptr,
    void* c_ptr,
    bool output_fp32,
    int size_m,
    int size_k,
    int size_n,
    int64_t c_stride_m,
    int device,
    cudaStream_t stream
)
{
    int cutoff = hgemm_narrow_n_cutoff();
    if (cutoff <= 0 || size_m < 1 || size_m > HGEMM_NARROW_MAX_M || size_n < 1 || size_n > cutoff) return false;
    if (size_k < 1) return false;

    int k_per_split = MAX(HGEMM_NARROW_MIN_K_PER_SPLIT, CEIL_DIVIDE(size_k, HGEMM_NARROW_MAX_SPLITS));
    int splits = CEIL_DIVIDE(size_k, k_per_split);
    float* ws = (float*) DevCtx::instance().get_ws(device);
    if (!ws || (size_t) splits * size_m * size_n * sizeof(float) > (size_t) WORKSPACE_SIZE) return false;

    hgemm_narrow_partial_kernel<<<splits, HGEMM_NARROW_THREADS, 0, stream>>>
        (a_ptr, b_ptr, ws, size_m, size_k, size_n, k_per_split);

    int outputs = size_m * size_n;
    int blocks = CEIL_DIVIDE(outputs, HGEMM_NARROW_THREADS);
    if (output_fp32)
        hgemm_narrow_reduce_kernel<float><<<blocks, HGEMM_NARROW_THREADS, 0, stream>>>
            (ws, (float*) c_ptr, size_m, size_n, splits, c_stride_m);
    else
        hgemm_narrow_reduce_kernel<half><<<blocks, HGEMM_NARROW_THREADS, 0, stream>>>
            (ws, (half*) c_ptr, size_m, size_n, splits, c_stride_m);
    cuda_check(cudaPeekAtLastError());
    return true;
}

static void hgemm_gemmex_impl
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    cudaStream_t stream
)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());

    bool output_fp32 = c.dtype() == at::kFloat;
    bool output_fp16 = c.dtype() == at::kHalf;

    TORCH_CHECK(output_fp32 || output_fp16, "c must be float32 or float16");

    // Check shapes of a,b,c are compatible
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK(c.dim() >= 2, "c must have at least 2 dimensions");
    TORCH_CHECK_SHAPES(a, -1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, -1, 1);
    TORCH_CHECK(c.stride(-1) == 1, "c must have contiguous columns");

    const half* a_ptr = (const half*) a.data_ptr();
    const half* b_ptr = (const half*) b.data_ptr();

    int size_k = a.size(-1);
    int size_m = a.numel() / size_k;
    int size_n = b.size(-1);
    int64_t c_stride_m = c.stride(-2);
    TORCH_CHECK(c_stride_m >= size_n, "c row stride is too small");
    TORCH_CHECK(c_stride_m <= std::numeric_limits<int>::max(), "c row stride is too large");

    // ROCm sibling: narrow outputs at small m run the GEMV kernels above (see file header)
    if (hgemm_narrow_try(a_ptr, b_ptr, c.data_ptr(), output_fp32, size_m, size_k, size_n, c_stride_m,
                         a.device().index(), stream))
        return;

    // Set cuBLAS modes and workspace
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    cudaDataType_t c_type = output_fp32 ? CUDA_R_32F : CUDA_R_16F;
    auto r = cublasGemmEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, b_ptr, CUDA_R_16F, size_n,
                 a_ptr, CUDA_R_16F, size_k,
        &beta_,  c.data_ptr(), c_type, (int) c_stride_m,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    hgemm_gemmex_impl(a, b, c, stream);

    if (graph) graph->need_cublas = true;
}

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
)
{
    hgemm_gr(a, b, c, nullptr);
}

/*
Strided-batched row-major matmul, a[b] @ w[b] -> c[b] for b in [0, B), fp16 inputs with fp32
accumulation (same cuBLAS setup as hgemm). a: [B, m, k], w: [B, k, n], c: [B, m, n], all
contiguous; c fp16 or fp32. Used by the batched expert reconstruct path (moe_batch_recon.py).
*/
void hgemm_batched
(
    at::Tensor a,
    at::Tensor w,
    at::Tensor c
)
{
    // Reconstruct-path GEMM: the fp16-accumulator kernel where it pays (GeForce), else cuBLAS
    if (hgemm_f16acc_try(a, w, c)) return;

    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    bool output_fp32 = c.dtype() == at::kFloat;
    TORCH_CHECK(output_fp32 || c.dtype() == at::kHalf, "hgemm_batched: c must be float32 or float16");
    TORCH_CHECK_DIM(a, 3);
    TORCH_CHECK_DIM(w, 3);
    TORCH_CHECK_DIM(c, 3);
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && c.is_contiguous(), "hgemm_batched: tensors must be contiguous");
    TORCH_CHECK_SHAPES(a, 0, w, 0, 1);
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
    TORCH_CHECK_SHAPES(a, 2, w, 1, 1);
    TORCH_CHECK_SHAPES(a, 1, c, 1, 1);
    TORCH_CHECK_SHAPES(w, 2, c, 2, 1);

    int batch = a.size(0);
    int size_m = a.size(1);
    int size_k = a.size(2);
    int size_n = w.size(2);
    if (!batch || !size_m || !size_n || !size_k) return;

    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    auto r = cublasGemmStridedBatchedEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, w.data_ptr(), CUDA_R_16F, size_n, (long long) size_k * size_n,
                 a.data_ptr(), CUDA_R_16F, size_k, (long long) size_m * size_k,
        &beta_,  c.data_ptr(), output_fp32 ? CUDA_R_32F : CUDA_R_16F, size_n, (long long) size_m * size_n,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}
