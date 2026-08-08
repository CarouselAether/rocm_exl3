// Numeric validation of exl3_gemm_inner_rdna against a CPU reference.
//
//   hipcc -o /tmp/gemm_check rocm_tools/gemm_check.hip -std=c++17 \
//         --offload-arch=gfx1151 -O3 -I../exllamav3/exllamav3_ext \
//         -include ../exllamav3/exllamav3_ext/rocm/hip_compat.hip.h ... && /tmp/gemm_check
//
// The reference weight matrix is produced with upstream's own reconstruct
// mapping (quant/reconstruct.cu, carried into rocm/ with only include lines
// changed), NOT with the GEMM's B path. That matters: the GEMM transposes the
// dequantized fragments across lanes and stages them through LDS to reach the
// WMMA B layout, so using its own transpose as the reference would be circular
// and blind to exactly the bug most likely to exist.
//
// What this covers: the WMMA matmul, fp32 vs fp16 accumulation, the split-K
// global reduction across threadblocks, and the output C layout. What it does
// not cover: the hadamard output path (shmem_out_had), which needs post_scale
// data and is exercised at a higher level.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#include "../exllamav3/exllamav3_ext/rocm/quant/exl3_kernel_map_rdna.hip.h"
#include "../exllamav3/exllamav3_ext/rocm/quant/exl3_gemm_inner_rdna.hip.h"

// ---------------------------------------------------------------------------
// Reference dequant: upstream reconstruct's fragment -> tile mapping.
// Produces W[k][n] as half, given packed B in [k_block][n_block][16*bits] form.
// ---------------------------------------------------------------------------
template <int bits, int cb>
__global__ __launch_bounds__(256)
void ref_reconstruct(half* __restrict__ W, const uint16_t* __restrict__ Bq, int blocks_n)
{
    constexpr int packed_size = 256 * bits / 16;

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int k = blockIdx.y;
    int n = blockIdx.x * 8;

    __shared__ uint32_t s_packed[8][packed_size / 2];
    Bq += (size_t)(k * blocks_n + n) * packed_size;

    if (t < packed_size) ((int4*) s_packed)[t] = ((const int4*) Bq)[t];
    __syncthreads();

    FragB frag[2];
    dq_dispatch<bits, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    __shared__ half2 tile[16][8][8];

    half2 n0 = __shfl_down(frag[0][0], 4, 32);
    half2 n1 = __shfl_down(frag[0][1], 4, 32);
    half2 n2 = __shfl_down(frag[1][0], 4, 32);
    half2 n3 = __shfl_down(frag[1][1], 4, 32);

    if (!(lane_id & 4))
    {
        half2 m0 = __halves2half2(__low2half (frag[0][0]), __low2half (n0));
        half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
        half2 m2 = __halves2half2(__low2half (frag[0][1]), __low2half (n1));
        half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
        half2 m4 = __halves2half2(__low2half (frag[1][0]), __low2half (n2));
        half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
        half2 m6 = __halves2half2(__low2half (frag[1][1]), __low2half (n3));
        half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
        int r0 = (lane_id % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
        int c0 = lane_id / 8, c1 = c0 + 4;
        tile[r0][warp_id][c0] = m0;  tile[r1][warp_id][c0] = m1;
        tile[r2][warp_id][c0] = m2;  tile[r3][warp_id][c0] = m3;
        tile[r0][warp_id][c1] = m4;  tile[r1][warp_id][c1] = m5;
        tile[r2][warp_id][c1] = m6;  tile[r3][warp_id][c1] = m7;
    }
    __syncthreads();

    // Row (k*16+r) of W holds blocks_n*16 halves = blocks_n*2 int4. Each block
    // of 8 n-blocks contributes 16 int4, indexed by c. Mirrors upstream
    // reconstruct.cu's final store exactly.
    int r = t / 16, c = t % 16;
    const int4* tile_i4 = (const int4*) tile;
    int4* out = ((int4*) W) + (size_t)(k * 16 + r) * (blocks_n * 2) + n * 2 + c;
    *out = tile_i4[t];
}

// ---------------------------------------------------------------------------
template <int bits, bool c_fp32, int cb, int TM, int TK, int TN, int SH, int FR>
__global__ __launch_bounds__(256)
void gemm_launch(const half* A, const uint16_t* B, void* C,
                 int m, int k, int n, int* locks)
{
    exl3_gemm_kernel_inner<bits, c_fp32, cb, TM, TK, TN, SH, FR, false>
        (A, B, C, m, k, n, locks, nullptr);
}

static int g_fail = 0;

template <int bits, int cb, int TN, int SH>
void run_case(int size_m, int size_k, int size_n, int num_slices)
{
    constexpr int TM = 16, TK = 16, FR = 3;
    constexpr int TB_K = TK / 16, TB_N = TN / 16;
    constexpr int NUM_WARPS = 256 / 32;
    constexpr int FPW = TB_N / NUM_WARPS;

    const int k_blocks = size_k / 16;
    const int blocks_n = size_n / 16;
    const int packed_per_block = 256 * bits / 16;
    const size_t bq_elems = (size_t) k_blocks * blocks_n * packed_per_block;

    // random packed weights
    std::vector<uint16_t> hBq(bq_elems);
    for (size_t i = 0; i < bq_elems; i++) hBq[i] = (uint16_t)(rand() & 0xffff);

    std::vector<half> hA((size_t) size_m * size_k);
    for (size_t i = 0; i < hA.size(); i++)
        hA[i] = __float2half(((float)(rand() % 2001) - 1000.f) / 4000.f);

    uint16_t* dBq; half* dA; half* dW; void* dC; int* dLocks;
    (void) hipMalloc(&dBq, bq_elems * 2);
    (void) hipMalloc(&dA, hA.size() * 2);
    (void) hipMalloc(&dW, (size_t) size_k * size_n * 2);
    (void) hipMalloc(&dC, (size_t) size_m * size_n * 4);
    (void) hipMalloc(&dLocks, (size_t) blocks_n * 4);
    (void) hipMemcpy(dBq, hBq.data(), bq_elems * 2, hipMemcpyHostToDevice);
    (void) hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice);
    (void) hipMemset(dLocks, 0, (size_t) blocks_n * 4);
    (void) hipMemset(dC, 0, (size_t) size_m * size_n * 4);

    // reference weights via upstream's reconstruct mapping
    hipLaunchKernelGGL((ref_reconstruct<bits, cb>), dim3(blocks_n / 8, k_blocks), dim3(256), 0, 0,
                       dW, dBq, blocks_n);
    (void) hipDeviceSynchronize();
    std::vector<half> hW((size_t) size_k * size_n);
    (void) hipMemcpy(hW.data(), dW, hW.size() * 2, hipMemcpyDeviceToHost);

    // CPU reference GEMM in fp32
    std::vector<float> ref((size_t) size_m * size_n, 0.f);
    for (int i = 0; i < size_m; i++)
        for (int kk = 0; kk < size_k; kk++)
        {
            float a = __half2float(hA[(size_t) i * size_k + kk]);
            if (a == 0.f) continue;
            const half* wrow = &hW[(size_t) kk * size_n];
            float* orow = &ref[(size_t) i * size_n];
            for (int j = 0; j < size_n; j++) orow[j] += a * __half2float(wrow[j]);
        }

    // dynamic LDS, matching the kernel's own arithmetic
    // Must match exl3_gemm_inner_rdna.hip.h exactly. Getting this wrong
    // under-allocates dynamic LDS and corrupts the tail of the last warp's
    // staging tile -- which reads as a kernel bug. It is not; it is this
    // arithmetic drifting from the kernel's.
    constexpr int SH_A_STRIDE   = TK + 8;
    constexpr int SH_B_DQ_STRIDE = EXL3_GEMM_SH_B_DQ_STRIDE;
    size_t smem = 2 * SH * (TM * SH_A_STRIDE)
                + 2 * SH * (TB_K * TB_N * 256 / 16 * bits)
                + 2 * (NUM_WARPS * 16 * SH_B_DQ_STRIDE);

    hipLaunchKernelGGL((gemm_launch<bits, true, cb, TM, TK, TN, SH, FR>),
                       dim3(num_slices), dim3(256 * TB_K), smem, 0,
                       dA, dBq, dC, size_m, size_k, size_n, dLocks);
    hipError_t err = hipDeviceSynchronize();

    std::vector<float> got((size_t) size_m * size_n);
    (void) hipMemcpy(got.data(), dC, got.size() * 4, hipMemcpyDeviceToHost);

    double max_rel = 0.0, rms_ref = 0.0, rms_err = 0.0;
    int bad = 0;
    for (size_t i = 0; i < ref.size(); i++)
    {
        double d = fabs(got[i] - ref[i]);
        double denom = fabs(ref[i]) + 1e-3;
        if (d / denom > max_rel) max_rel = d / denom;
        rms_ref += (double) ref[i] * ref[i];
        rms_err += d * d;
        if (d > 0.02 * denom + 0.05) bad++;
    }
    rms_ref = sqrt(rms_ref / ref.size());
    rms_err = sqrt(rms_err / ref.size());

    printf("  bits=%d cb=%d N=%-4d m=%-3d k=%-5d n=%-5d slices=%-3d %s  "
           "err/RMS=%.5f  worst_rel=%.4f  outliers=%d\n",
           bits, cb, TN, size_m, size_k, size_n, num_slices,
           err == hipSuccess ? "ok  " : hipGetErrorString(err),
           rms_ref > 0 ? rms_err / rms_ref : 0.0, max_rel, bad);
    if (err != hipSuccess || bad) g_fail++;

    (void) hipFree(dBq); (void) hipFree(dA); (void) hipFree(dW);
    (void) hipFree(dC);  (void) hipFree(dLocks);
}


// ---------------------------------------------------------------------------
// Timing. Shapes are compared at one size that every tile width divides, so the
// comparison is not confounded by ragged edges. Repeats are reported as best
// and median: the measured noise floor on this part is ~4.7% spread with the
// first run reading high, so a single timing cannot support a claim under ~5%.
// ---------------------------------------------------------------------------
template <int bits, int cb, int TN, int SH, bool CFP32>
void run_bench(int size_m, int size_k, int size_n, int num_slices, int reps)
{
    constexpr int TM = 16, TK = 16, FR = 3;
    constexpr int TB_K = TK / 16, TB_N = TN / 16;
    constexpr int NUM_WARPS = 256 / 32;

    const int k_blocks = size_k / 16;
    const int blocks_n = size_n / 16;
    const size_t bq_elems = (size_t) k_blocks * blocks_n * (256 * bits / 16);

    uint16_t* dBq; half* dA; void* dC; int* dLocks;
    (void) hipMalloc(&dBq, bq_elems * 2);
    (void) hipMalloc(&dA, (size_t) size_m * size_k * 2);
    (void) hipMalloc(&dC, (size_t) size_m * size_n * 4);
    (void) hipMalloc(&dLocks, (size_t) blocks_n * 4);
    (void) hipMemset(dBq, 0x5a, bq_elems * 2);
    (void) hipMemset(dA, 0x11, (size_t) size_m * size_k * 2);

    // Must match exl3_gemm_inner_rdna.hip.h exactly. Getting this wrong
    // under-allocates dynamic LDS and corrupts the tail of the last warp's
    // staging tile -- which reads as a kernel bug. It is not; it is this
    // arithmetic drifting from the kernel's.
    constexpr int SH_A_STRIDE   = TK + 8;
    constexpr int SH_B_DQ_STRIDE = EXL3_GEMM_SH_B_DQ_STRIDE;
    size_t smem = 2 * SH * (TM * SH_A_STRIDE)
                + 2 * SH * (TB_K * TB_N * 256 / 16 * bits)
                + 2 * (NUM_WARPS * 16 * SH_B_DQ_STRIDE);

    auto once = [&]() {
        (void) hipMemset(dLocks, 0, (size_t) blocks_n * 4);
        (void) hipMemset(dC, 0, (size_t) size_m * size_n * 4);
        hipLaunchKernelGGL((gemm_launch<bits, CFP32, cb, TM, TK, TN, SH, FR>),
                           dim3(num_slices), dim3(256 * TB_K), smem, 0,
                           dA, dBq, dC, size_m, size_k, size_n, dLocks);
    };

    once(); (void) hipDeviceSynchronize();          // warm up, discarded

    std::vector<double> ms;
    hipEvent_t e0, e1; (void) hipEventCreate(&e0); (void) hipEventCreate(&e1);
    for (int r = 0; r < reps; r++)
    {
        (void) hipEventRecord(e0);
        once();
        (void) hipEventRecord(e1);
        (void) hipEventSynchronize(e1);
        float t = 0.f; (void) hipEventElapsedTime(&t, e0, e1);
        ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    double best = ms.front(), med = ms[ms.size() / 2];
    double spread = (ms.back() - ms.front()) / med * 100.0;
    double gflop = 2.0 * size_m * size_k * size_n / 1e9;

    printf("  N=%-4d %s slices=%-3d  best=%7.3f ms  med=%7.3f ms  "
           "spread=%5.1f%%  %6.1f GFLOP/s\n",
           TN, CFP32 ? "fp32-out" : "fp16-out", num_slices, best, med, spread, gflop / (best / 1000.0));

    (void) hipFree(dBq); (void) hipFree(dA); (void) hipFree(dC); (void) hipFree(dLocks);
}

int main()
{
    srand(1234);
    printf("exl3_gemm_inner_rdna vs CPU reference (fp32 accumulate)\n");
    printf("reference weights via upstream reconstruct mapping\n\n");

    run_case<4, 0, 128, 4>(16, 512, 256, 1);
    run_case<4, 0, 128, 4>(16, 1024, 512, 4);
    run_case<4, 0, 256, 3>(16, 1024, 512, 4);
    run_case<2, 0, 128, 4>(16, 512, 256, 2);
    run_case<8, 0, 128, 4>(16, 512, 256, 2);
    run_case<4, 1, 128, 4>(16, 512, 256, 1);
    run_case<4, 2, 128, 4>(16, 512, 256, 1);
    run_case<4, 0, 384, 3>(16, 1024, 768, 4);

    printf("\n%s\n", g_fail ? "FAILED" : "all cases passed");

    // n = 3072 is divisible by 128/256/384/512, so no shape pays an edge penalty
    printf("\ntiming, m=16 k=4096 n=3072, bits=4 cb=0, 9 reps\n");
    for (int slices : {20})
    {
        printf(" fp32 output (no spill):\n");
        run_bench<4, 0, 256, 3, true >(16, 4096, 3072, slices, 9);
        run_bench<4, 0, 384, 3, true >(16, 4096, 3072, slices, 9);
        run_bench<4, 0, 512, 3, true >(16, 4096, 3072, slices, 9);
        printf(" fp16 output (384 spills 48 VGPRs, 512 spills 0):\n");
        run_bench<4, 0, 256, 3, false>(16, 4096, 3072, slices, 9);
        run_bench<4, 0, 384, 3, false>(16, 4096, 3072, slices, 9);
        run_bench<4, 0, 512, 3, false>(16, 4096, 3072, slices, 9);
    }
    return g_fail ? 1 : 0;
}
