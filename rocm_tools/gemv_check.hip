// Numeric validation of exl3_gemv_dot_kernel (rocm/quant/exl3_gemv_kernel_rdna.hip.h)
// against a CPU reference, plus a bandwidth measurement at decode shapes.
//
//   rocm_tools/build_check.sh gemv_check && /tmp/gemv_check
//
// Same discipline as gemm_check.hip: the reference weight matrix comes from
// upstream's own reconstruct mapping (quant/reconstruct.cu), NOT from the GEMV's
// B path. The GEMV dequantizes to a swizzled fragment layout and then unswizzles
// through LDS with a shuffle-by-4; using that same transform to build the
// reference would be circular and blind to the bug most likely to exist.
//
// What this covers: the fdot2 dot product, the LDS staging and unswizzle at all
// three wave counts, every bitwidth 1-8, all three codebooks, and both output
// dtypes. What it does not cover: the SUH/SVH hadamard kernels either side of
// it, which need scale data and are exercised at a higher level. Note that
// dispatch (exl3_gemv_try_launch) is host code and is not exercised here at all.
//
// The LDS size comes from exl3_gemv_smem_bytes() in the kernel header -- the
// same function the host dispatch calls. That is deliberate: in the GEMM the
// equivalent figure was duplicated in three places, drifted, and presented as a
// kernel bug at FRAGS_N_PER_WARP > 1.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#include "../exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_kernel_rdna.hip.h"

// ---------------------------------------------------------------------------
// Reference dequant: upstream reconstruct's fragment -> tile mapping.
// Produces W[k][n] as half, given packed B in [k_block][n_block][16*bits] form.
// Lifted verbatim from gemm_check.hip so both harnesses check against the same
// independent reference.
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

    int r = t / 16, c = t % 16;
    const int4* tile_i4 = (const int4*) tile;
    int4* out = ((int4*) W) + (size_t)(k * 16 + r) * (blocks_n * 2) + n * 2 + c;
    *out = tile_i4[t];
}

static int g_fail = 0;

// ---------------------------------------------------------------------------
// Correctness
// ---------------------------------------------------------------------------

template <int bits, int cb, int WARPS, bool CFP32, bool SPLITK = false>
void run_case(int size_k, int size_n)
{
    const int k_blocks = size_k / 16;
    const int blocks_n = size_n / 16;
    const int packed_per_block = 256 * bits / 16;
    const size_t bq_elems = (size_t) k_blocks * blocks_n * packed_per_block;

    // Random packed weights. Non-symmetric input so an orientation bug cannot
    // cancel out.
    std::vector<uint16_t> hBq(bq_elems);
    for (size_t i = 0; i < bq_elems; i++) hBq[i] = (uint16_t)(rand() & 0xffff);

    std::vector<half> hA((size_t) size_k);
    for (size_t i = 0; i < hA.size(); i++)
        hA[i] = __float2half(((float)(rand() % 2001) - 1000.f) / 4000.f);

    uint16_t* dBq; half* dA; half* dW; void* dC;
    (void) hipMalloc(&dBq, bq_elems * 2);
    (void) hipMalloc(&dA, hA.size() * 2);
    (void) hipMalloc(&dW, (size_t) size_k * size_n * 2);
    (void) hipMalloc(&dC, (size_t) size_n * 4);
    (void) hipMemcpy(dBq, hBq.data(), bq_elems * 2, hipMemcpyHostToDevice);
    (void) hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice);
    (void) hipMemset(dC, 0, (size_t) size_n * 4);

    // Reference weights via upstream's reconstruct mapping
    hipLaunchKernelGGL((ref_reconstruct<bits, cb>), dim3(blocks_n / 8, k_blocks), dim3(256), 0, 0,
                       dW, dBq, blocks_n);
    (void) hipDeviceSynchronize();
    std::vector<half> hW((size_t) size_k * size_n);
    (void) hipMemcpy(hW.data(), dW, hW.size() * 2, hipMemcpyDeviceToHost);

    // CPU reference GEMV in fp32
    std::vector<float> ref((size_t) size_n, 0.f);
    for (int kk = 0; kk < size_k; kk++)
    {
        float a = __half2float(hA[kk]);
        if (a == 0.f) continue;
        const half* wrow = &hW[(size_t) kk * size_n];
        for (int j = 0; j < size_n; j++) ref[j] += a * __half2float(wrow[j]);
    }

    const int n_tiles = size_n / 16;
    const int num_blocks = SPLITK ? n_tiles : (n_tiles + WARPS - 1) / WARPS;
    const size_t smem = SPLITK ? exl3_gemv_smem_bytes_splitk(WARPS)
                               : exl3_gemv_smem_bytes(WARPS);

    // Both dot-tile cores: core=0 is the barrier-free direct core (the runtime
    // default), core=1 the LDS core (EXL3_GEMV_LDS=1). Same reference, same
    // gates -- the direct core's whole risk is its layout claims, which show
    // up here as gross error, not as a tolerance miss.
    for (int core = 0; core < 2; core++)
    {
    const bool lds_core = core != 0;
    (void) hipMemset(dC, 0, (size_t) size_n * 4);

    if (SPLITK)
        hipLaunchKernelGGL((exl3_gemv_dot_kernel_splitk<bits, CFP32, cb, WARPS>),
                           dim3(num_blocks), dim3(WARPS * 32), smem, 0,
                           dA, dBq, dC, size_k, size_n, lds_core);
    else
        hipLaunchKernelGGL((exl3_gemv_dot_kernel<bits, CFP32, cb, WARPS>),
                           dim3(num_blocks), dim3(WARPS * 32), smem, 0,
                           dA, dBq, dC, size_k, size_n, lds_core);
    hipError_t err = hipDeviceSynchronize();

    std::vector<float> got((size_t) size_n);
    if (CFP32)
    {
        (void) hipMemcpy(got.data(), dC, got.size() * 4, hipMemcpyDeviceToHost);
    }
    else
    {
        std::vector<half> got_h((size_t) size_n);
        (void) hipMemcpy(got_h.data(), dC, got_h.size() * 2, hipMemcpyDeviceToHost);
        for (size_t i = 0; i < got.size(); i++) got[i] = __half2float(got_h[i]);
    }

    double max_rel = 0.0, rms_ref = 0.0, rms_err = 0.0;
    int bad = 0;
    // fp16 output carries ~2^-11 relative quantization on the store, so it gets
    // a looser outlier gate than the fp32 path. The RMS ratio is reported for
    // both and is the number to watch.
    const double rel_gate = CFP32 ? 0.02 : 0.01;
    const double abs_gate = CFP32 ? 0.05 : 0.05;
    for (size_t i = 0; i < ref.size(); i++)
    {
        double d = fabs(got[i] - ref[i]);
        double denom = fabs(ref[i]) + 1e-3;
        if (d / denom > max_rel) max_rel = d / denom;
        rms_ref += (double) ref[i] * ref[i];
        rms_err += d * d;
        if (d > rel_gate * denom + abs_gate) bad++;
    }
    rms_ref = sqrt(rms_ref / ref.size());
    rms_err = sqrt(rms_err / ref.size());

    printf("  bits=%d cb=%d waves=%-2d %s %s %s k=%-5d n=%-5d %s  "
           "err/RMS=%.5f  worst_rel=%.4f  outliers=%d\n",
           bits, cb, WARPS, SPLITK ? "splitk" : "single",
           lds_core ? "lds " : "dir ",
           CFP32 ? "fp32" : "fp16", size_k, size_n,
           err == hipSuccess ? "ok  " : hipGetErrorString(err),
           rms_ref > 0 ? rms_err / rms_ref : 0.0, max_rel, bad);
    if (err != hipSuccess || bad) g_fail++;
    }

    (void) hipFree(dBq); (void) hipFree(dA); (void) hipFree(dW); (void) hipFree(dC);
}

// ---------------------------------------------------------------------------
// Bandwidth. Decode is memory-bound on this part, so the meaningful figure is
// how fast B streams, not GFLOP/s. B is k*n*bits/8 bytes and is read exactly
// once per call. Repeats are reported best and median: the measured noise floor
// is ~4.7% spread with the first run reading high, so a single timing cannot
// support a claim under ~5%.
//
// Achievable DRAM bandwidth on this part has never been measured, so these
// numbers are a floor for the kernel, not a fraction of peak.
// ---------------------------------------------------------------------------

template <int bits, int cb, int WARPS>
void run_bench(int size_k, int size_n, int reps, bool lds_core = false)
{
    const int k_blocks = size_k / 16;
    const int blocks_n = size_n / 16;
    const size_t bq_elems = (size_t) k_blocks * blocks_n * (256 * bits / 16);

    uint16_t* dBq; half* dA; void* dC;
    (void) hipMalloc(&dBq, bq_elems * 2);
    (void) hipMalloc(&dA, (size_t) size_k * 2);
    (void) hipMalloc(&dC, (size_t) size_n * 4);
    (void) hipMemset(dBq, 0x5a, bq_elems * 2);
    (void) hipMemset(dA, 0x11, (size_t) size_k * 2);

    const int n_tiles = size_n / 16;
    const int num_blocks = (n_tiles + WARPS - 1) / WARPS;
    const size_t smem = exl3_gemv_smem_bytes(WARPS);

    auto once = [&]() {
        hipLaunchKernelGGL((exl3_gemv_dot_kernel<bits, true, cb, WARPS>),
                           dim3(num_blocks), dim3(WARPS * 32), smem, 0,
                           dA, dBq, dC, size_k, size_n, lds_core);
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
    double b_gb = (double) size_k * size_n * bits / 8.0 / 1e9;

    printf("  bits=%d waves=%-2d %s k=%-5d n=%-5d  best=%7.3f ms  med=%7.3f ms  "
           "spread=%5.1f%%  B-read %6.1f GB/s\n",
           bits, WARPS, lds_core ? "lds" : "dir",
           size_k, size_n, best, med, spread, b_gb / (best / 1000.0));

    (void) hipFree(dBq); (void) hipFree(dA); (void) hipFree(dC);
}

// ---------------------------------------------------------------------------
// Launch-geometry sweep on cache-defeating working sets. The bandwidth section
// above re-reads ONE B buffer per repeat, so any shape near or under the
// 32 MiB Infinity Cache reads high (measured up to +50% on a 33 MB shape).
// Here NB distinct B buffers rotate across repeats, so steady-state traffic is
// DRAM. Sweeps the single-warp form at 4/8/16 waves and the in-block split-K
// form at 4/8/16 warps, both cores, for one (bits, k, n) shape.
// ---------------------------------------------------------------------------

template <int bits, int cb, int WARPS, bool SPLITK>
void sweep_one(int size_k, int size_n, int reps, bool lds_core)
{
    constexpr int NB = 4;
    const int k_blocks = size_k / 16;
    const int n_tiles = size_n / 16;
    const size_t bq_elems = (size_t) k_blocks * n_tiles * (256 * bits / 16);

    uint16_t* dBq[NB]; half* dA; void* dC;
    for (int i = 0; i < NB; i++)
    {
        (void) hipMalloc(&dBq[i], bq_elems * 2);
        (void) hipMemset(dBq[i], 0x5a ^ i, bq_elems * 2);
    }
    (void) hipMalloc(&dA, (size_t) size_k * 2);
    (void) hipMalloc(&dC, (size_t) size_n * 4);
    (void) hipMemset(dA, 0x11, (size_t) size_k * 2);

    const int num_blocks = SPLITK ? n_tiles : (n_tiles + WARPS - 1) / WARPS;
    const size_t smem = SPLITK ? exl3_gemv_smem_bytes_splitk(WARPS)
                               : exl3_gemv_smem_bytes(WARPS);

    auto once = [&](int i) {
        if constexpr (SPLITK)
            hipLaunchKernelGGL((exl3_gemv_dot_kernel_splitk<bits, true, cb, WARPS>),
                               dim3(num_blocks), dim3(WARPS * 32), smem, 0,
                               dA, dBq[i], dC, size_k, size_n, lds_core);
        else
            hipLaunchKernelGGL((exl3_gemv_dot_kernel<bits, true, cb, WARPS>),
                               dim3(num_blocks), dim3(WARPS * 32), smem, 0,
                               dA, dBq[i], dC, size_k, size_n, lds_core);
    };

    once(0); once(1); (void) hipDeviceSynchronize();   // warm up, discarded

    std::vector<double> ms;
    hipEvent_t e0, e1; (void) hipEventCreate(&e0); (void) hipEventCreate(&e1);
    for (int r = 0; r < reps; r++)
    {
        (void) hipEventRecord(e0);
        once(r % NB);
        (void) hipEventRecord(e1);
        (void) hipEventSynchronize(e1);
        float t = 0.f; (void) hipEventElapsedTime(&t, e0, e1);
        ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    double med = ms[ms.size() / 2];
    double spread = (ms.back() - ms.front()) / med * 100.0;
    double b_gb = (double) size_k * size_n * bits / 8.0 / 1e9;

    printf("  bits=%d k=%-5d n=%-5d %s %-6s waves=%-2d  med=%7.3f ms  "
           "spread=%5.1f%%  %6.1f GB/s\n",
           bits, size_k, size_n, lds_core ? "lds" : "dir",
           SPLITK ? "splitk" : "single", WARPS, med, spread, b_gb / (med / 1000.0));

    for (int i = 0; i < NB; i++) (void) hipFree(dBq[i]);
    (void) hipFree(dA); (void) hipFree(dC);
}

template <int bits, int cb>
void sweep_shape(int size_k, int size_n, int reps)
{
    sweep_one<bits, cb, 4,  false>(size_k, size_n, reps, false);
    sweep_one<bits, cb, 8,  false>(size_k, size_n, reps, false);
    sweep_one<bits, cb, 16, false>(size_k, size_n, reps, false);
    sweep_one<bits, cb, 4,  true >(size_k, size_n, reps, false);
    sweep_one<bits, cb, 8,  true >(size_k, size_n, reps, false);
    sweep_one<bits, cb, 16, true >(size_k, size_n, reps, false);
    sweep_one<bits, cb, 8,  false>(size_k, size_n, reps, true);   // lds reference
    sweep_one<bits, cb, 8,  true >(size_k, size_n, reps, true);
    printf("\n");
}

int main()
{
    srand(1234);

    if (getenv("GEMV_SWEEP"))
    {
        printf("launch-geometry sweep, DRAM-resident (4 rotating B buffers), fp32 out\n\n");
        printf(" Gemma-4-31B shapes, 6 bpw:\n");
        sweep_shape<6, 1>(5376, 21504, 12);   // gate/up (n_tiles 1344, above splitk cap)
        sweep_shape<6, 1>(21504, 5376, 12);   // down    (336 tiles, splitk today)
        sweep_shape<6, 1>(5376, 8192, 12);    // q       (512 tiles, splitk boundary)
        sweep_shape<6, 1>(8192, 5376, 12);    // o
        printf(" Laguna expert shapes, 4 bpw:\n");
        sweep_shape<4, 1>(3072, 1024, 12);    // expert down-proj slice
        sweep_shape<4, 1>(1024, 3072, 12);    // expert gate/up slice
        printf(" DeepSeek-V4-Flash expert shapes, 2 bpw (mul1):\n");
        sweep_shape<2, 2>(4096, 2048, 12);    // expert gate/up (k_tiles 256, n_tiles 128)
        sweep_shape<2, 2>(2048, 4096, 12);    // expert down    (k_tiles 128, n_tiles 256)
        printf(" lm_head scale (does split-K still win at 16384 tiles?):\n");
        sweep_shape<6, 1>(5376, 262144, 8);
        return 0;
    }
    printf("exl3_gemv_dot_kernel vs CPU reference (fp32 accumulate)\n");
    printf("reference weights via upstream reconstruct mapping\n\n");

    printf(" bitwidths, cb 0, fp32 out:\n");
    run_case<1, 0, 8, true>(512, 256);
    run_case<2, 0, 8, true>(512, 256);
    run_case<3, 0, 8, true>(512, 256);
    run_case<4, 0, 8, true>(512, 256);
    run_case<5, 0, 8, true>(512, 256);
    run_case<6, 0, 8, true>(512, 256);
    run_case<7, 0, 8, true>(512, 256);
    run_case<8, 0, 8, true>(512, 256);

    printf("\n codebooks (mcg, mul1):\n");
    run_case<4, 1, 8, true>(512, 256);
    run_case<4, 2, 8, true>(512, 256);
    run_case<2, 1, 8, true>(512, 256);
    run_case<2, 2, 8, true>(512, 256);

    printf("\n wave counts -- 16 and 4 are the two the host selector can pick\n"
           " besides 8, and the LDS layout is per-warp, so each needs its own case:\n");
    run_case<4, 0, 4,  true>(512, 256);
    run_case<4, 0, 16, true>(512, 256);
    run_case<4, 0, 16, true>(512, 128);   // n_tiles=8 < WARPS: partial block, tail warps exit early

    printf("\n split-K at every wave count -- the in-block reduce and the k-chunk\n"
           " division depend on WARPS, so 4 and 16 need their own cases now that\n"
           " exl3_gemv_splitk_warps() can pick them:\n");
    run_case<4, 0, 4,  true,  true>(4096, 1024);
    run_case<4, 0, 8,  true,  true>(4096, 1024);
    run_case<4, 0, 16, true,  true>(4096, 1024);
    run_case<4, 0, 16, false, true>(4096, 1024);
    run_case<2, 2, 4,  true,  true>(2048, 4096);   // DS4 expert down shape/cb
    run_case<2, 2, 16, true,  true>(4096, 2048);   // DS4 expert gate/up shape/cb
    run_case<6, 1, 16, true,  true>(2064, 256);    // k_tiles=129, odd chunk split

    printf("\n fp16 output:\n");
    run_case<4, 0, 8, false>(512, 256);
    run_case<2, 0, 8, false>(512, 256);

    printf("\n larger shapes:\n");
    run_case<4, 0, 4, true>(4096, 1024);
    run_case<3, 0, 8, true>(2048, 512);
    run_case<8, 0, 4, true>(1024, 2048);

    printf("\n%s\n", g_fail ? "FAILED" : "all cases passed");

    // Decode shapes: k x n as they appear in attention projections and FFN.
    // Wave count follows the host selector: n_tiles <= 32 -> 16, n_tiles <= 96
    // and k_blocks <= 256 -> 8, else 4.
    printf("\nbandwidth, 4 bpw, 9 reps (dir = barrier-free core, lds = LDS core)\n");
    run_bench<4, 0, 16>(4096, 512,  9);    // n_tiles 32
    run_bench<4, 0, 16>(4096, 512,  9, true);
    run_bench<4, 0, 8 >(4096, 1536, 9);    // n_tiles 96, k_blocks 256
    run_bench<4, 0, 8 >(4096, 1536, 9, true);
    run_bench<4, 0, 4 >(4096, 4096, 9);
    run_bench<4, 0, 4 >(4096, 4096, 9, true);
    run_bench<4, 0, 4 >(4096, 14336, 9);   // FFN width
    run_bench<4, 0, 4 >(4096, 14336, 9, true);
    printf("\nsame shapes, 2 bpw (decode-bound end of the range)\n");
    run_bench<2, 0, 4 >(4096, 4096, 9);
    run_bench<2, 0, 4 >(4096, 4096, 9, true);
    run_bench<2, 0, 4 >(4096, 14336, 9);
    run_bench<2, 0, 4 >(4096, 14336, 9, true);

    // k-scaling at fixed n, to separate kernel time from launch overhead. Every
    // warp owns one n-tile and walks all of k serially, so kernel time should be
    // proportional to k and independent of n until the grid stops filling the
    // device. If these rows are flat, the figures above are measuring the
    // launch, not the kernel.
    printf("\nk-scaling at n=4096, 4 bpw (kernel time vs launch overhead)\n");
    run_bench<4, 0, 4 >(512,  4096, 9);
    run_bench<4, 0, 4 >(1024, 4096, 9);
    run_bench<4, 0, 4 >(2048, 4096, 9);
    run_bench<4, 0, 4 >(4096, 4096, 9);

    printf("\nn-scaling at k=4096, 4 bpw (grid fill)\n");
    run_bench<4, 0, 4 >(4096, 1024, 9);
    run_bench<4, 0, 4 >(4096, 2048, 9);
    run_bench<4, 0, 4 >(4096, 8192, 9);

    // The host selector picks waves by n_tiles: <= 32 -> 16, <= 96 (and
    // k_blocks <= 256) -> 8, else 4. More waves per block means FEWER blocks for
    // the same work -- at n = 512 that is 2 blocks on a 20-WGP part. Since the
    // kernel is k-latency-bound at these widths, block count is what fills the
    // device, so the selector is worth re-testing directly rather than
    // inheriting. Same total work in every row of each group.
    printf("\nwave-count selector check, k=4096, 4 bpw (same work, different block count)\n");
    printf("  n=512  (n_tiles 32 -> selector picks 16):\n");
    run_bench<4, 0, 16>(4096, 512, 9);   // 2 blocks
    run_bench<4, 0, 8 >(4096, 512, 9);   // 4 blocks
    run_bench<4, 0, 4 >(4096, 512, 9);   // 8 blocks
    printf("  n=1536 (n_tiles 96 -> selector picks 8):\n");
    run_bench<4, 0, 16>(4096, 1536, 9);
    run_bench<4, 0, 8 >(4096, 1536, 9);
    run_bench<4, 0, 4 >(4096, 1536, 9);

    return g_fail ? 1 : 0;
}
