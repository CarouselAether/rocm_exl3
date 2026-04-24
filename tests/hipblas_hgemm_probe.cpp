// Standalone probe: does hipblasHgemm produce correct fp16 matmul on gfx1151
// under ROCm 7.2.1? Compares against a host-side fp32-accumulated reference.
//
// This is independent of exllamav3 — if hipblasHgemm fails here, the issue is
// in the hipBLAS library itself (Tensile kernels), not in our invocation.
//
// Build:
//   hipcc -o /tmp/hipblas_hgemm_probe tests/hipblas_hgemm_probe.cpp \
//     -lhipblas -std=c++17
// Run:
//   /tmp/hipblas_hgemm_probe

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <random>

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        printf("HIP error %s at line %d: %s\n", #cmd, __LINE__, hipGetErrorString(e)); \
        return 1; \
    } \
} while(0)

#define HIPBLAS_CHECK(cmd) do { \
    hipblasStatus_t s = (cmd); \
    if (s != HIPBLAS_STATUS_SUCCESS) { \
        printf("hipBLAS error %s at line %d: status %d\n", #cmd, __LINE__, (int)s); \
        return 1; \
    } \
} while(0)

// Run one hgemm and report correctness
static int probe(int M, int K, int N, const char* label)
{
    printf("\n=== Probe %s: M=%d K=%d N=%d ===\n", label, M, K, N);

    std::vector<_Float16> h_A(M * K);
    std::vector<_Float16> h_B(K * N);
    std::vector<_Float16> h_C(M * N, (_Float16)0);

    // Deterministic fill — small magnitudes so fp32 accumulator stays in range
    // but large enough to expose precision issues in wrong-kernel output.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& x : h_A) x = (_Float16)dist(rng);
    for (auto& x : h_B) x = (_Float16)dist(rng);

    // Host reference: C = A @ B, fp32 accumulate, fp16 store
    std::vector<float> h_ref(M * N, 0.f);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += (float)h_A[m*K + k] * (float)h_B[k*N + n];
            h_ref[m*N + n] = acc;
        }

    // Device buffers
    _Float16 *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    HIP_CHECK(hipMalloc(&d_A, M*K*sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_B, K*N*sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_C, M*N*sizeof(_Float16)));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), M*K*sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), K*N*sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, M*N*sizeof(_Float16)));

    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));

    _Float16 alpha = (_Float16)1.0f;
    _Float16 beta  = (_Float16)0.0f;

    // Matching exllamav3's hgemm.cu transposition trick:
    // Row-major A(M,K), B(K,N), C(M,N) computed via column-major hipBLAS call
    //   hipblasHgemm(h, N_op, N_op, N, M, K, &α, B, N, A, K, &β, C, N)
    // Effectively computes C^T (N,M col-major) = B^T * A^T  → same as row-major C = A*B
    HIPBLAS_CHECK(hipblasHgemm(
        handle,
        HIPBLAS_OP_N, HIPBLAS_OP_N,
        N, M, K,
        (const hipblasHalf*)&alpha,
        (const hipblasHalf*)d_B, N,
        (const hipblasHalf*)d_A, K,
        (const hipblasHalf*)&beta,
        (hipblasHalf*)d_C, N
    ));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, M*N*sizeof(_Float16), hipMemcpyDeviceToHost));

    // Tolerances: fp16 output with fp32 accumulate → relative error ~1e-2
    // against our fp32 reference. Use generous absolute threshold scaled by K.
    float tol_abs = 0.05f * sqrtf((float)K);
    float max_abs_err = 0.f;
    float max_rel_err = 0.f;
    int bad = 0;
    int zeros = 0;
    float sum_got = 0.f, sum_ref = 0.f;
    for (int i = 0; i < M*N; i++) {
        float got = (float)h_C[i];
        float ref = h_ref[i];
        float err = fabsf(got - ref);
        if (err > max_abs_err) max_abs_err = err;
        float denom = fmaxf(fabsf(ref), 1e-6f);
        float rel = err / denom;
        if (rel > max_rel_err) max_rel_err = rel;
        if (err > tol_abs) bad++;
        if (got == 0.f) zeros++;
        sum_got += got; sum_ref += ref;
    }

    printf("  tolerance (abs): %.4f\n", tol_abs);
    printf("  max abs error:   %.4f\n", max_abs_err);
    printf("  max rel error:   %.4f\n", max_rel_err);
    printf("  out-of-tol:      %d / %d (%.1f%%)\n", bad, M*N, 100.0f*bad/(M*N));
    printf("  zero cells:      %d / %d\n", zeros, M*N);
    printf("  sum_got=%.4f sum_ref=%.4f\n", sum_got, sum_ref);
    printf("  sample C[0][0]: got=%.4f ref=%.4f\n", (float)h_C[0], h_ref[0]);
    printf("  sample C[0][1]: got=%.4f ref=%.4f\n", (float)h_C[1], h_ref[1]);
    printf("  sample C[M-1][N-1]: got=%.4f ref=%.4f\n",
           (float)h_C[M*N-1], h_ref[M*N-1]);

    int retcode = 0;
    if (zeros > M*N / 4) {
        printf("  ** FAIL: >25%% of output is exactly zero (kernel didn't write or wrote zeros)\n");
        retcode = 2;
    } else if (bad > M*N / 100) {
        printf("  ** FAIL: >1%% of outputs exceed tolerance (likely wrong Tensile kernel)\n");
        retcode = 3;
    } else {
        printf("  PASS\n");
    }

    hipblasDestroy(handle);
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    return retcode;
}

int main()
{
    int dev;
    HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    printf("Device: %s (gcnArch=%s)\n", prop.name, prop.gcnArchName);

    int fails = 0;

    // Small — if Tensile can't even do 32x64x128, it's catastrophic
    fails |= probe(32, 128, 64, "small");

    // Medium — typical attention head projection size
    fails |= probe(128, 512, 1024, "medium");

    // Large — closer to real LLM FFN shapes (but bounded so test finishes fast)
    fails |= probe(256, 2048, 2048, "large");

    // Non-multiple-of-16 — exposes padding / tile-alignment bugs
    fails |= probe(17, 257, 129, "non-aligned");

    if (fails) {
        printf("\n=== OVERALL: hipblasHgemm produces wrong output on this device ===\n");
        return 1;
    } else {
        printf("\n=== OVERALL: hipblasHgemm works correctly. Tensile is fine. ===\n");
        return 0;
    }
}
