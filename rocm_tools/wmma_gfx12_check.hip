// Contributed by choruzo (issue #1), validated on an R9700 (gfx1201). Also passes on gfx1151
// (exercises the unchanged gfx11 mma_sync there).
//
// Check rdna_wmma::mma_sync on gfx12 against a CPU reference, using the shipped header.
// Covers: plain C = A*B, accumulation onto a prior C (C layout round-trip), and a
// K = 64 chain of four mma_sync calls on the same accumulator (the GEMM inner-loop shape).
// Non-symmetric, non-uniform inputs so operand-order and transpose bugs are distinguishable.
//
//   hipcc -o wmma_gfx12_check wmma_gfx12_check.hip -std=c++17 --offload-arch=gfx1201

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include "../exllamav3/exllamav3_ext/rocm/rdna_wmma.hip.h"   // build from rocm_tools/

#define CHK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); return 2; } } while (0)

// C[16x16] = prior + sum over kb of A[:, kb*16..] * B[kb*16.., :]; A is 16 x K (row-major), B is K x 16.
__global__ void k_chain(const half* A, const half* B, const float* prior, float* C, int K, int use_prior)
{
    WmmaFragA fa; WmmaFragB fb; WmmaFragC fc;
    fc.clear();
    if (use_prior) rdna_wmma::load_accumulate_c(fc, prior, 16);
    for (int kb = 0; kb < K / 16; kb++)
    {
        rdna_wmma::load_matrix_a(fa, A + kb * 16, K);
        rdna_wmma::load_matrix_b(fb, B + kb * 16 * 16, 16);
        rdna_wmma::mma_sync(fc, fa, fb);
    }
    rdna_wmma::store_matrix_c(C, fc, 16);
}

static int run(int K, int use_prior, const char* name)
{
    std::vector<half> A(16 * K), B(K * 16);
    std::vector<float> Af(16 * K), Bf(K * 16), P(256), C(256), R(256);
    for (int i = 0; i < 16; i++) for (int k = 0; k < K; k++)
    { float v = 0.01f * (i * 7 + k * 3 % 11) - 0.3f + 0.001f * i * k; A[i * K + k] = __float2half(v); Af[i * K + k] = __half2float(A[i * K + k]); }
    for (int k = 0; k < K; k++) for (int j = 0; j < 16; j++)
    { float v = 0.02f * ((k * 5 + j * 13) % 17) - 0.15f + 0.003f * j; B[k * 16 + j] = __float2half(v); Bf[k * 16 + j] = __half2float(B[k * 16 + j]); }
    for (int i = 0; i < 256; i++) P[i] = use_prior ? 0.5f * i - 17.0f : 0.0f;

    for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++)
    {
        double s = P[i * 16 + j];
        for (int k = 0; k < K; k++) s += (double) Af[i * K + k] * Bf[k * 16 + j];
        R[i * 16 + j] = (float) s;
    }

    half *dA, *dB; float *dP, *dC;
    CHK(hipMalloc(&dA, A.size() * 2)); CHK(hipMalloc(&dB, B.size() * 2));
    CHK(hipMalloc(&dP, 1024)); CHK(hipMalloc(&dC, 1024));
    CHK(hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice));
    CHK(hipMemcpy(dB, B.data(), B.size() * 2, hipMemcpyHostToDevice));
    CHK(hipMemcpy(dP, P.data(), 1024, hipMemcpyHostToDevice));
    CHK(hipMemset(dC, 0, 1024));
    k_chain<<<1, 32>>>(dA, dB, dP, dC, K, use_prior);
    CHK(hipGetLastError());
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(C.data(), dC, 1024, hipMemcpyDeviceToHost));
    hipFree(dA); hipFree(dB); hipFree(dP); hipFree(dC);

    int bad = 0; double maxerr = 0;
    for (int i = 0; i < 256; i++)
    {
        double e = fabs(C[i] - R[i]);
        maxerr = fmax(maxerr, e);
        if (e > 1e-3 + 1e-3 * fabs(R[i])) { if (bad < 4) printf("   mismatch r=%d c=%d got %f ref %f\n", i / 16, i % 16, C[i], R[i]); bad++; }
    }
    printf("%-28s K=%-3d %s  bad=%d/256  max_abs_err=%.2e\n", name, K, bad ? "FAIL" : "PASS", bad, maxerr);
    return bad ? 1 : 0;
}

int main()
{
    hipDeviceProp_t p; CHK(hipGetDeviceProperties(&p, 0));
    printf("device: %s (%s)\n", p.name, p.gcnArchName);
    int f = 0;
    f += run(16, 0, "C = A*B");
    f += run(16, 1, "C = prior + A*B");
    f += run(64, 0, "chain 4x mma_sync");
    f += run(64, 1, "prior + chain 4x mma_sync");
    printf(f ? "RESULT: FAIL\n" : "RESULT: ALL PASS\n");
    return f ? 1 : 0;
}
