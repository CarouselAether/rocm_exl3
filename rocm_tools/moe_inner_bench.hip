// Standalone A/B of the fused-MoE GEMM mainloops on one expert-GEMM shape, no torch:
//
//   SRC=moe_inner_bench rocm_tools/build_gemv_tiles_bench.sh && /tmp/moe_inner_bench [rows...]
//
// Runs the old mainloop (exl3_gemm_kernel_inner, the shared RDNA GEMM inner, exactly
// as exl3_moe_kernel instantiates it: 16-row tiles, TILESIZE_K 32, N 256, 3 stages)
// and the pipelined mainloop (rocm/quant/exl3_moe_inner_rdna.hip.h, 16/32/64-row
// tiles) over E expert matrices in sequence inside one launch of G blocks, the way
// one expert group of exl3_moe does (stream-K slices, fp16 partials through the lock
// protocol). E matrices of K=2 trellis cycle > 4x the 32 MiB Infinity Cache, so
// the rate is DRAM-resident.
//
// For each row count it checks the pipelined output BIT FOR BIT against the old one
// and prints us per matrix and GB/s of trellis per group of G blocks (DS4: two
// groups of 10 share the GPU, so the whole-GPU rate is ~2x the G=10 figure when
// both groups stream).
//
// Env: MIB_E (matrices, default 64), MIB_G (blocks, default 10), MIB_K/N (4096/2048),
//      MIB_REP (timed launches, default 5), MIB_CONC (groups run side by side, default 2)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include "../exllamav3/exllamav3_ext/rocm/quant/exl3_kernel_map_rdna.hip.h"
#include "../exllamav3/exllamav3_ext/quant/exl3_moe_common.cuh"
#include "../exllamav3/exllamav3_ext/rocm/quant/exl3_gemm_inner_rdna.hip.h"
#include "../exllamav3/exllamav3_ext/rocm/quant/exl3_moe_shape_rdna.hip.h"
#include "../exllamav3/exllamav3_ext/rocm/quant/exl3_moe_inner_rdna.hip.h"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

#ifndef MIB_WPE
#define MIB_WPE
#endif
constexpr int BITS = 2;
constexpr int CB = 2;
constexpr int NT = 256;

// mode 0: old inner (16-row tiles looped); 1: pipelined (64/32/16 selection as the kernel)
template <int MODE>
__global__ __launch_bounds__(512) MIB_WPE void mib_kernel
(
    const half* A, const uint16_t* const* B, half* C, int E, int rows, int size_k, int size_n, int* locks,
    int c_stride
)
{
    // blockIdx.z = concurrent group: each group does every E-th matrix offset by z
    const int g = blockIdx.z;
    locks += g * 64;
    for (int e = g; e < E; e += gridDim.z)
    {
        const half* in = A;
        half* out = C + (size_t) e * c_stride;
        int size_m = rows;
        while (size_m > 0)
        {
            int tm;
            if constexpr (MODE == 0)
            {
                exl3_gemm_kernel_inner<BITS, false, CB, 16, 32, NT, MOE_SH_STAGES, MOE_FRAG_STAGES, false>
                    (in, B[e], out, MIN(size_m, 16), size_k, size_n, locks, nullptr);
                tm = 16;
            }
            else
            {
                if (size_m > 48)      { moe_pipe::moe_gemm_pipe<BITS, CB, 4, 32, NT>(in, B[e], out, MIN(size_m, 64), size_k, size_n, locks); tm = 64; }
                else if (size_m > 32) { moe_pipe::moe_gemm_pipe<BITS, CB, 3, 32, NT>(in, B[e], out, MIN(size_m, 48), size_k, size_n, locks); tm = 48; }
                else if (size_m > 16) { moe_pipe::moe_gemm_pipe<BITS, CB, 2, 32, NT>(in, B[e], out, MIN(size_m, 32), size_k, size_n, locks); tm = 32; }
                else                  { moe_pipe::moe_gemm_pipe<BITS, CB, 1, 32, NT>(in, B[e], out, MIN(size_m, 16), size_k, size_n, locks); tm = 16; }
            }
            in += tm * size_k;
            out += tm * size_n;
            size_m -= tm;
        }
    }
}

static int envi(const char* n, int d) { const char* e = getenv(n); return e ? atoi(e) : d; }

int main(int argc, char** argv)
{
    const int E = envi("MIB_E", 64), G = envi("MIB_G", 10), K = envi("MIB_K", 4096), N = envi("MIB_N", 2048);
    const int REP = envi("MIB_REP", 5), CONC = envi("MIB_CONC", 2);
    std::vector<int> rows_list;
    for (int i = 1; i < argc; ++i) rows_list.push_back(atoi(argv[i]));
    if (rows_list.empty()) rows_list = {6, 12, 23, 42, 61};

    const size_t mat_u16 = (size_t) K * N * BITS / 16;
    std::vector<uint16_t> hb(mat_u16);
    uint16_t** dB_list = new uint16_t*[E];
    uint32_t s = 12345;
    for (int e = 0; e < E; ++e)
    {
        for (auto& v : hb) { s = s * 1664525u + 1013904223u; v = (uint16_t) (s >> 16); }
        CK(hipMalloc(&dB_list[e], mat_u16 * 2));
        CK(hipMemcpy(dB_list[e], hb.data(), mat_u16 * 2, hipMemcpyHostToDevice));
    }
    uint16_t** dB; CK(hipMalloc(&dB, E * sizeof(void*)));
    CK(hipMemcpy(dB, dB_list, E * sizeof(void*), hipMemcpyHostToDevice));

    const int MAXR = 128;
    std::vector<half> ha((size_t) MAXR * K);
    for (auto& v : ha) { s = s * 1664525u + 1013904223u; v = __float2half(((int) (s >> 16) % 2001 - 1000) / 1000.0f); }
    half* dA; CK(hipMalloc(&dA, ha.size() * 2)); CK(hipMemcpy(dA, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
    const int c_stride = MAXR * N;
    half *dC0, *dC1; CK(hipMalloc(&dC0, (size_t) E * c_stride * 2)); CK(hipMalloc(&dC1, (size_t) E * c_stride * 2));
    int* dL; CK(hipMalloc(&dL, 4096 * 4)); CK(hipMemset(dL, 0, 4096 * 4));

    CK(hipFuncSetAttribute((const void*) mib_kernel<0>, hipFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
    CK(hipFuncSetAttribute((const void*) mib_kernel<1>, hipFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));

    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    printf("MoE inner bench: K=%d N=%d bits=%d cb=%d E=%d G=%d x %d groups, DB=%d  (%.1f MiB trellis)\n",
           K, N, BITS, CB, E, G, CONC, EXL3_MOE_PIPE_DB, E * mat_u16 * 2 / 1048576.0);
    dim3 grid(G, 1, CONC);
    for (int rows : rows_list)
    {
        float ms[2];
        for (int mode = 0; mode < 2; ++mode)
        {
            half* dC = mode ? dC1 : dC0;
            CK(hipMemset(dC, 0, (size_t) E * c_stride * 2));
            auto launch = [&] ()
            {
                if (mode == 0) mib_kernel<0><<<grid, 512, SMEM_MAX>>>(dA, dB, dC, E, rows, K, N, dL, c_stride);
                else           mib_kernel<1><<<grid, 512, moe_pipe::smem_launch_bytes<32, NT>()>>>(dA, dB, dC, E, rows, K, N, dL, c_stride);
            };
            launch(); CK(hipDeviceSynchronize());
            std::vector<float> t;
            for (int r = 0; r < REP; ++r)
            {
                CK(hipEventRecord(e0)); launch(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float x; CK(hipEventElapsedTime(&x, e0, e1)); t.push_back(x);
            }
            std::sort(t.begin(), t.end());
            ms[mode] = t[t.size() / 2];
        }
        // bitwise compare (rows < size_m only)
        std::vector<uint16_t> h0((size_t) E * c_stride), h1((size_t) E * c_stride);
        CK(hipMemcpy(h0.data(), dC0, h0.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(h1.data(), dC1, h1.size() * 2, hipMemcpyDeviceToHost));
        size_t diff = 0;
        for (int e = 0; e < E; ++e)
            for (size_t i = 0; i < (size_t) rows * N; ++i)
                diff += h0[(size_t) e * c_stride + i] != h1[(size_t) e * c_stride + i];
        const double bytes = (double) E * mat_u16 * 2;
        printf("  rows=%-3d  old %8.3f ms (%6.1f GB/s)   pipe %8.3f ms (%6.1f GB/s)   x%.2f   %s (%zu diff)\n",
               rows, ms[0], bytes / ms[0] / 1e6, ms[1], bytes / ms[1] / 1e6, ms[0] / ms[1],
               diff ? "MISMATCH" : "bit-identical", diff);
    }
    return 0;
}
