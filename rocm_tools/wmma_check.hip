// Validate rocm/rdna_wmma.hip.h against a CPU reference.
//
// This compiles the *actual shipped header*, not a copy of its logic, so it
// catches drift between the header and whatever a smoke test happens to
// replicate. Inputs are deliberately non-symmetric and non-uniform: with
// A[i][j] != A[j][i] and A != B, an operand-order bug and a transposed-store
// bug produce different, detectable failures. A symmetric test matrix would
// pass with either bug present.
//
//   hipcc -o /tmp/wmma_check rocm_tools/wmma_check.hip \
//         -std=c++17 --offload-arch=gfx1151 && /tmp/wmma_check
//
// Re-run after any toolchain bump. The HIP docs cover MFMA (CDNA) but not WMMA
// (RDNA), so the ISA PDF plus this check are the only authority for the layout.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>

#include "../exllamav3/exllamav3_ext/rocm/rdna_wmma.hip.h"

#define M 16
#define N 16
#define K 16

__global__ void k_gemm_f32(const half* A, const half* B, float* C)
{
    WmmaFragA fa; WmmaFragB fb; WmmaFragC fc;
    fc.clear();
    rdna_wmma::load_matrix_a(fa, A, K);
    rdna_wmma::load_matrix_b(fb, B, N);
    rdna_wmma::mma_sync(fc, fa, fb);
    rdna_wmma::store_matrix_c(C, fc, N);
}

__global__ void k_gemm_half(const half* A, const half* B, half* C)
{
    WmmaFragA fa; WmmaFragB fb; WmmaFragC fc;
    fc.clear();
    rdna_wmma::load_matrix_a(fa, A, K);
    rdna_wmma::load_matrix_b(fb, B, N);
    rdna_wmma::mma_sync(fc, fa, fb);
    rdna_wmma::store_matrix_c_half(C, fc, N);
}

// C := prior + A*B, exercising load_accumulate_c
__global__ void k_gemm_acc(const half* A, const half* B, const float* prior, float* C)
{
    WmmaFragA fa; WmmaFragB fb; WmmaFragC fc;
    fc.clear();
    rdna_wmma::load_matrix_a(fa, A, K);
    rdna_wmma::load_matrix_b(fb, B, N);
    rdna_wmma::mma_sync(fc, fa, fb);
    rdna_wmma::load_accumulate_c(fc, prior, N);
    rdna_wmma::store_matrix_c(C, fc, N);
}

// Bounds-checked store: rows >= valid_rows and cols >= valid_cols must be untouched
__global__ void k_gemm_checked(const half* A, const half* B, float* C, int vr, int vc)
{
    WmmaFragA fa; WmmaFragB fb; WmmaFragC fc;
    fc.clear();
    rdna_wmma::load_matrix_a(fa, A, K);
    rdna_wmma::load_matrix_b(fb, B, N);
    rdna_wmma::mma_sync(fc, fa, fb);
    rdna_wmma::store_matrix_c_checked(C, fc, N, vr, vc);
}

// bf16 -> fp32. Reuses WmmaFragC and store_matrix_c, which is the point:
// the bf16 variant differs only in the operand type.
__global__ void k_gemm_bf16(const __hip_bfloat16* A, const __hip_bfloat16* B, float* C)
{
    WmmaFragA_bf16 fa; WmmaFragB_bf16 fb; WmmaFragC fc;
    fc.clear();
    rdna_wmma::load_matrix_a_bf16(fa, A, K);
    rdna_wmma::load_matrix_b_bf16(fb, B, N);
    rdna_wmma::mma_sync_bf16(fc, fa, fb);
    rdna_wmma::store_matrix_c(C, fc, N);
}

template <bool OPSEL>
__global__ void k_gemm_f16acc(const half* A, const half* B, half* C)
{
    WmmaFragA fa; WmmaFragB fb; WmmaFragC_f16 fc;
    fc.clear();
    rdna_wmma::load_matrix_a(fa, A, K);
    rdna_wmma::load_matrix_b(fb, B, N);
    rdna_wmma::mma_sync_f16<OPSEL>(fc, fa, fb);
    rdna_wmma::store_matrix_c_f16<OPSEL>(C, fc, N);
}

// Two independent accumulators sharing one fragment: A*B at opsel=0 and
// B*A at opsel=1. If the halves interfered, neither result would survive.
// This is the property that makes the fp16 form worth having at all.
__global__ void k_gemm_f16_dual(const half* A, const half* B, half* C0, half* C1)
{
    WmmaFragA fa, fa2; WmmaFragB fb, fb2; WmmaFragC_f16 fc;
    fc.clear();
    rdna_wmma::load_matrix_a(fa, A, K);
    rdna_wmma::load_matrix_b(fb, B, N);
    rdna_wmma::load_matrix_a(fa2, B, K);   // swapped operands for a different result
    rdna_wmma::load_matrix_b(fb2, A, N);
    rdna_wmma::mma_sync_f16<false>(fc, fa, fb);
    rdna_wmma::mma_sync_f16<true >(fc, fa2, fb2);
    rdna_wmma::store_matrix_c_f16<false>(C0, fc, N);
    rdna_wmma::store_matrix_c_f16<true >(C1, fc, N);
}

// int8 path. Templated on the signedness so the test covers the mixed-sign
// combination EXL3 actually needs (unsigned weights x signed activations),
// not just the symmetric ones -- a swapped sign-flag pairing is invisible when
// both operands share a signedness.
template <bool SA, bool SB>
__global__ void k_gemm_i8(const int8_t* A, const int8_t* B, int* C)
{
    WmmaFragA_i8 fa; WmmaFragB_i8 fb; WmmaFragC_i32 fc;
    fc.clear();
    rdna_wmma::load_matrix_a_i8(fa, A, K);
    rdna_wmma::load_matrix_b_i8(fb, B, N);
    rdna_wmma::mma_sync_i8<SA, SB>(fc, fa, fb);
    rdna_wmma::store_matrix_c_i32(C, fc, N);
}

static int fails = 0;
static void report(const char* name, int bad, int total)
{
    printf("  %-38s %s  (%d/%d wrong)\n", name, bad ? "FAIL" : "ok", bad, total);
    if (bad) fails++;
}

int main()
{
    half hA[M * K], hB[K * N], hCh[M * N];
    float hC[M * N], ref[M * N], prior[M * N];

    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            hA[i * K + j] = __float2half(0.5f * i - 0.25f * j + 0.125f * ((i * j) % 7));
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            hB[i * N + j] = __float2half(-0.375f * i + 0.5f * j + 0.0625f * ((i + 2 * j) % 5));
    for (int i = 0; i < M * N; i++) prior[i] = 0.03125f * (i % 11) - 0.125f;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float s = 0.f;
            for (int k = 0; k < K; k++)
                s += __half2float(hA[i * K + k]) * __half2float(hB[k * N + j]);
            ref[i * N + j] = s;
        }

    half *dA, *dB, *dCh; float *dC, *dPrior;
    (void) hipMalloc(&dA, sizeof(hA));      (void) hipMalloc(&dB, sizeof(hB));
    (void) hipMalloc(&dC, sizeof(hC));      (void) hipMalloc(&dCh, sizeof(hCh));
    (void) hipMalloc(&dPrior, sizeof(prior));
    (void) hipMemcpy(dA, hA, sizeof(hA), hipMemcpyHostToDevice);
    (void) hipMemcpy(dB, hB, sizeof(hB), hipMemcpyHostToDevice);
    (void) hipMemcpy(dPrior, prior, sizeof(prior), hipMemcpyHostToDevice);

    printf("rdna_wmma.hip.h against CPU reference, gfx1151 wave32\n");

    // 1. f32 store
    hipLaunchKernelGGL(k_gemm_f32, dim3(1), dim3(32), 0, 0, dA, dB, dC);
    (void) hipMemcpy(hC, dC, sizeof(hC), hipMemcpyDeviceToHost);
    int bad = 0, transposed = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            if (fabsf(hC[i * N + j] - ref[i * N + j]) > 0.06f) bad++;
            if (i != j && fabsf(hC[i * N + j] - ref[j * N + i]) < 0.06f) transposed++;
        }
    report("load_a/load_b/mma_sync/store_c", bad, M * N);
    report("orientation (not transposed)", transposed ? 1 : 0, 1);

    // 2. fp16 store
    hipLaunchKernelGGL(k_gemm_half, dim3(1), dim3(32), 0, 0, dA, dB, dCh);
    (void) hipMemcpy(hCh, dCh, sizeof(hCh), hipMemcpyDeviceToHost);
    bad = 0;
    for (int i = 0; i < M * N; i++)
        if (fabsf(__half2float(hCh[i]) - ref[i]) > 0.5f) bad++;   // fp16 store tolerance
    report("store_matrix_c_half", bad, M * N);

    // 3. accumulate
    hipLaunchKernelGGL(k_gemm_acc, dim3(1), dim3(32), 0, 0, dA, dB, dPrior, dC);
    (void) hipMemcpy(hC, dC, sizeof(hC), hipMemcpyDeviceToHost);
    bad = 0;
    for (int i = 0; i < M * N; i++)
        if (fabsf(hC[i] - (ref[i] + prior[i])) > 0.06f) bad++;
    report("load_accumulate_c", bad, M * N);

    // 4. bounds-checked store leaves the out-of-range region alone
    const int vr = 9, vc = 5;
    for (int i = 0; i < M * N; i++) hC[i] = -1234.f;
    (void) hipMemcpy(dC, hC, sizeof(hC), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k_gemm_checked, dim3(1), dim3(32), 0, 0, dA, dB, dC, vr, vc);
    (void) hipMemcpy(hC, dC, sizeof(hC), hipMemcpyDeviceToHost);
    int bad_in = 0, bad_out = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            bool inside = (i < vr && j < vc);
            float v = hC[i * N + j];
            if (inside  && fabsf(v - ref[i * N + j]) > 0.06f) bad_in++;
            if (!inside && v != -1234.f)                      bad_out++;
        }
    report("store_matrix_c_checked (in range)", bad_in, vr * vc);
    report("store_matrix_c_checked (guard)", bad_out, M * N - vr * vc);

    // 5. fp16 accumulate. Tolerance is relative: the accumulation itself runs in
    // fp16, so this is legitimately less accurate than the f32 form.
    auto tol_f16 = [](float want) { return 0.004f * fabsf(want) + 0.02f; };

    hipLaunchKernelGGL((k_gemm_f16acc<false>), dim3(1), dim3(32), 0, 0, dA, dB, dCh);
    (void) hipMemcpy(hCh, dCh, sizeof(hCh), hipMemcpyDeviceToHost);
    bad = 0;
    for (int i = 0; i < M * N; i++)
        if (fabsf(__half2float(hCh[i]) - ref[i]) > tol_f16(ref[i])) bad++;
    report("mma_sync_f16 opsel=0", bad, M * N);

    hipLaunchKernelGGL((k_gemm_f16acc<true>), dim3(1), dim3(32), 0, 0, dA, dB, dCh);
    (void) hipMemcpy(hCh, dCh, sizeof(hCh), hipMemcpyDeviceToHost);
    bad = 0;
    for (int i = 0; i < M * N; i++)
        if (fabsf(__half2float(hCh[i]) - ref[i]) > tol_f16(ref[i])) bad++;
    report("mma_sync_f16 opsel=1", bad, M * N);

    // dual accumulator: opsel=0 holds A*B, opsel=1 holds B*A, in one fragment
    half hC0[M * N], hC1[M * N];
    half *dC0, *dC1;
    (void) hipMalloc(&dC0, sizeof(hC0)); (void) hipMalloc(&dC1, sizeof(hC1));
    hipLaunchKernelGGL(k_gemm_f16_dual, dim3(1), dim3(32), 0, 0, dA, dB, dC0, dC1);
    (void) hipMemcpy(hC0, dC0, sizeof(hC0), hipMemcpyDeviceToHost);
    (void) hipMemcpy(hC1, dC1, sizeof(hC1), hipMemcpyDeviceToHost);
    float refBA[M * N];
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float s = 0.f;
            for (int k = 0; k < K; k++)
                s += __half2float(hB[i * K + k]) * __half2float(hA[k * N + j]);
            refBA[i * N + j] = s;
        }
    int bad0 = 0, bad1 = 0;
    for (int i = 0; i < M * N; i++)
    {
        if (fabsf(__half2float(hC0[i]) - ref[i])   > tol_f16(ref[i]))   bad0++;
        if (fabsf(__half2float(hC1[i]) - refBA[i]) > tol_f16(refBA[i])) bad1++;
    }
    report("dual accumulator, opsel=0 half", bad0, M * N);
    report("dual accumulator, opsel=1 half", bad1, M * N);

    // 6. bf16 -> fp32. Reference is computed from the bf16-rounded values, so
    // this measures the mma, not the input rounding. bf16 has 8 mantissa bits,
    // hence a wider relative tolerance than fp16.
    auto f2bf = [](float f) -> unsigned short {
        unsigned int u; __builtin_memcpy(&u, &f, 4);
        unsigned int r = (u >> 16) & 1u;
        return (unsigned short)((u + 0x7fffu + r) >> 16);
    };
    auto bf2f = [](unsigned short b) -> float {
        unsigned int u = (unsigned int) b << 16; float f;
        __builtin_memcpy(&f, &u, 4); return f;
    };
    unsigned short hAb[M * K], hBb[K * N];
    float refb[M * N];
    for (int i = 0; i < M * K; i++) hAb[i] = f2bf(__half2float(hA[i]));
    for (int i = 0; i < K * N; i++) hBb[i] = f2bf(__half2float(hB[i]));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float s = 0.f;
            for (int k = 0; k < K; k++) s += bf2f(hAb[i * K + k]) * bf2f(hBb[k * N + j]);
            refb[i * N + j] = s;
        }
    __hip_bfloat16 *dAb, *dBb;
    (void) hipMalloc(&dAb, sizeof(hAb)); (void) hipMalloc(&dBb, sizeof(hBb));
    (void) hipMemcpy(dAb, hAb, sizeof(hAb), hipMemcpyHostToDevice);
    (void) hipMemcpy(dBb, hBb, sizeof(hBb), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k_gemm_bf16, dim3(1), dim3(32), 0, 0, dAb, dBb, dC);
    (void) hipMemcpy(hC, dC, sizeof(hC), hipMemcpyDeviceToHost);
    bad = 0; transposed = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float want = refb[i * N + j];
            if (fabsf(hC[i * N + j] - want) > 0.01f * fabsf(want) + 0.02f) bad++;
            if (i != j && fabsf(hC[i * N + j] - refb[j * N + i]) < 0.02f) transposed++;
        }
    report("mma_sync_bf16 -> fp32 (reuses WmmaFragC)", bad, M * N);
    report("bf16 orientation (not transposed)", transposed ? 1 : 0, 1);

    // 7. int8 tensor-core path, all four signedness combinations.
    // Byte values deliberately exceed 127 so signed and unsigned readings differ;
    // if they did not, a swapped sign flag would pass.
    uint8_t hAi[M * K], hBi[K * N];
    int hCi[M * N];
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++) hAi[i * K + j] = (uint8_t)((i * 13 + j * 7) % 256);
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++) hBi[i * N + j] = (uint8_t)((i * 11 + j * 17) % 256);

    int8_t *dAi, *dBi; int* dCi;
    (void) hipMalloc(&dAi, sizeof(hAi)); (void) hipMalloc(&dBi, sizeof(hBi));
    (void) hipMalloc(&dCi, sizeof(hCi));
    (void) hipMemcpy(dAi, hAi, sizeof(hAi), hipMemcpyHostToDevice);
    (void) hipMemcpy(dBi, hBi, sizeof(hBi), hipMemcpyHostToDevice);

    auto check_i8 = [&](const char* label, bool sa, bool sb)
    {
        (void) hipMemcpy(hCi, dCi, sizeof(hCi), hipMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                long s = 0;
                for (int k = 0; k < K; k++)
                {
                    int a = sa ? (int)(int8_t)hAi[i * K + k] : (int)hAi[i * K + k];
                    int b = sb ? (int)(int8_t)hBi[k * N + j] : (int)hBi[k * N + j];
                    s += (long)a * b;
                }
                if (hCi[i * N + j] != (int)s) bad++;
            }
        report(label, bad, M * N);
    };

    hipLaunchKernelGGL((k_gemm_i8<true, true>),   dim3(1), dim3(32), 0, 0, dAi, dBi, dCi);
    check_i8("i8 mma  signed x signed", true, true);
    hipLaunchKernelGGL((k_gemm_i8<false, false>), dim3(1), dim3(32), 0, 0, dAi, dBi, dCi);
    check_i8("i8 mma  unsigned x unsigned", false, false);
    hipLaunchKernelGGL((k_gemm_i8<false, true>),  dim3(1), dim3(32), 0, 0, dAi, dBi, dCi);
    check_i8("i8 mma  A unsigned x B signed", false, true);
    hipLaunchKernelGGL((k_gemm_i8<true, false>),  dim3(1), dim3(32), 0, 0, dAi, dBi, dCi);
    check_i8("i8 mma  A signed x B unsigned", true, false);

    printf("%s\n", fails ? "FAILED" : "all checks passed");
    return fails ? 1 : 0;
}
