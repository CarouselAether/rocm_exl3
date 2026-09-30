// Standalone A/B of the EXL3 GEMV dot cores on decode shapes: the barrier-free
// direct core (exl3_gemv_dot_tile_direct_mr, copied verbatim from
// rocm/quant/exl3_gemv_multirow_rdna.hip) against the tiles core
// (rocm/quant/exl3_gemv_tiles_rdna.hip.h). No torch, builds in ~1 minute:
//
//   rocm_tools/build_gemv_tiles_bench.sh && /tmp/gemv_tiles_bench [filter]
//
// Every case runs the same split-K wrapper (the multi-row kernel's chunking
// and reduction order) over a (N-tile, matrix) grid, with a matrix pointer
// table like exl3_gemv_mr_dot_multi, and
//   - compares the two cores' outputs BIT FOR BIT (they must be identical:
//     same fdot2 chains, same order), and
//   - times each over ROT rotating copies of the weight set (> 4x the 32 MiB
//     Infinity Cache, so the figure is DRAM-resident; RDNA_NOTES "Hardware")
//     and reports us/call, GB/s of trellis and G weights/s.
// The output epilogue (fused rotation) is not included: it is identical for
// both cores and small. A_LDS cases stage the input in LDS first, as the
// weighted mgemv down projection does.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>

#include "../exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_kernel_rdna.hip.h"
#include "../exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_tiles_rdna.hip.h"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

// ---- baseline: verbatim copy of the multi-row direct core ----------------------
template <int bits, int cb, int M>
__device__ __forceinline__ void base_dot_tile_direct_mr
(
    const half* __restrict__ A, const int lda, const uint16_t* __restrict__ B,
    const int n_tiles, const int tile_n, const int lane, const int kb_begin, const int kb_end, float* out
)
{
    constexpr int tile_elements = 16 * bits;
    const int r0 = (lane & 3) * 2;
    float accA[M], accB[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) { accA[r] = 0.0f; accB[r] = 0.0f; }
    for (int k_tile = kb_begin; k_tile < kb_end; k_tile++)
    {
        const uint32_t* b_ptr = (const uint32_t*) (B + (k_tile * n_tiles + tile_n) * tile_elements);
        FragB frag0, frag1;
        dq_dispatch<bits, cb>(b_ptr, lane << 3, frag0, frag1);
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            const half2* a2 = (const half2*) (A + r * lda + k_tile * 16);
            half2 a01 = a2[r0 >> 1];
            half2 a89 = a2[(r0 >> 1) + 4];
            accA[r] = __builtin_amdgcn_fdot2(a01, frag0[0], accA[r], false);
            accA[r] = __builtin_amdgcn_fdot2(a89, frag0[1], accA[r], false);
            accB[r] = __builtin_amdgcn_fdot2(a01, frag1[0], accB[r], false);
            accB[r] = __builtin_amdgcn_fdot2(a89, frag1[1], accB[r], false);
        }
    }
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        accA[r] += __shfl_xor(accA[r], 1, 32);
        accA[r] += __shfl_xor(accA[r], 2, 32);
        accB[r] += __shfl_xor(accB[r], 1, 32);
        accB[r] += __shfl_xor(accB[r], 2, 32);
        float vA = __shfl(accA[r], (lane & 7) * 4, 32);
        float vB = __shfl(accB[r], (lane & 7) * 4, 32);
        out[r] = (lane & 8) ? vB : vA;
    }
}

// ---- kernel: split-K over a (tile, matrix) grid, the multi-row wrapper's order --
// MODE 0 = baseline core; MODE >= 1 = tiles core with (U, T) from mode_ut()
__host__ __device__ constexpr int mode_un(int m) { return m == 0 ? 1 : (m == 1 ? 1 : m == 2 ? 2 : m == 3 ? 4 : m == 4 ? 2 : m == 5 ? 4 : m == 6 ? 1 : 2); }
__host__ __device__ constexpr int mode_tn(int m) { return m <= 3 ? 1 : (m <= 6 ? 2 : 4); }

template <int bits, int cb, int W, int M, int MODE, bool A_LDS>
__global__ __launch_bounds__(W * 32)
void bench_kernel
(
    const half* __restrict__ A,           // [nmat][M][K] (or broadcast when a_stride == 0)
    const uintptr_t* __restrict__ B_list,
    float* __restrict__ C,                // [nmat][M][N]
    const int size_k, const int size_n, const int a_stride
)
{
    constexpr int U = mode_un(MODE), T = mode_tn(MODE);
    extern __shared__ char shared_mem[];
    float* sh_red = (float*) shared_mem;
    half* sh_a = (half*) (sh_red + W * M * T * 16);

    const int lane = threadIdx.x % 32;
    const int warp_id = MODE ? exl3_uni(threadIdx.x / 32) : threadIdx.x / 32;
    const int j = blockIdx.y;
    const int tile_n = blockIdx.x * T;
    const int n_tiles = size_n / 16;
    const uint16_t* B = (const uint16_t*) B_list[j];
    if (MODE) B = exl3_uni_ptr(B);
    const half* Aj = A + (size_t) j * a_stride;

    if constexpr (A_LDS)
    {
        for (int i = threadIdx.x; i < M * size_k / 8; i += W * 32)
            ((uint4*) sh_a)[i] = ((const uint4*) Aj)[i];
        __syncthreads();
        Aj = sh_a;
    }

    const int num_k_tiles = size_k / 16;
    const int chunk = (num_k_tiles + W - 1) / W;
    const int kb0 = warp_id * chunk;
    const int kb1 = kb0 + chunk < num_k_tiles ? kb0 + chunk : num_k_tiles;

    float acc[T * M];
    #pragma unroll
    for (int r = 0; r < T * M; ++r) acc[r] = 0.0f;
    if (kb0 < kb1)
    {
        if constexpr (MODE == 0)
            base_dot_tile_direct_mr<bits, cb, M>(Aj, size_k, B, n_tiles, tile_n, lane, kb0, kb1, acc);
        else
            exl3_gemv_dot_tile_tiles<bits, cb, M, A_LDS, U, T>(Aj, size_k, B, n_tiles, tile_n, lane, kb0, kb1, acc);
    }
    if (lane < 16)
    {
        #pragma unroll
        for (int r = 0; r < T * M; ++r) sh_red[(warp_id * T * M + r) * 16 + lane] = acc[r];
    }
    __syncthreads();
    if (warp_id == 0 && lane < 16)
    {
        #pragma unroll
        for (int t = 0; t < T; ++t)
            #pragma unroll
            for (int r = 0; r < M; ++r)
            {
                float total = 0.0f;
                #pragma unroll
                for (int w = 0; w < W; ++w) total += sh_red[(w * T * M + t * M + r) * 16 + lane];
                C[((size_t) j * M + r) * size_n + (tile_n + t) * 16 + lane] = total;
            }
    }
}

struct Case
{
    const char* name;
    int bits, cb, k, n, nmat, W, M, a_lds;
};

// DS4-Flash 2.04bpw decode shapes (decode trace grids), plus the
// Qwen3.8 / Gemma codebook-bitrate combinations
static Case cases[] = {
    {"ds4 routed gate/up K2 4096->2048 x6",  2, 2, 4096,  2048, 6, 8, 1, 0},
    {"ds4 routed gate/up K2 m3",             2, 2, 4096,  2048, 6, 8, 4, 0},
    {"ds4 routed down K2 2048->4096 x6 lds", 2, 2, 2048,  4096, 6, 4, 1, 1},
    {"ds4 wo_a K4 4096->1024 x8",            4, 2, 4096,  1024, 8, 8, 1, 0},
    {"ds4 wo_a K4 m3",                       4, 2, 4096,  1024, 8, 8, 4, 0},
    {"ds4 wo_b K4 8192->4096",               4, 2, 8192,  4096, 1, 8, 1, 0},
    {"ds4 wq_b K4 1024->32768",              4, 2, 1024, 32768, 1, 4, 1, 0},
    {"ds4 wq_b K4 m3",                       4, 2, 1024, 32768, 1, 4, 4, 0},
    {"ds4 shared gate/up K4 4096->2048 x2",  4, 2, 4096,  2048, 2, 8, 1, 0},
    {"ds4 head K6 4096->129280 1w",          6, 2, 4096, 129280, 1, 1, 1, 0},
    {"qwen-like K4 2048->512 x10",           4, 2, 2048,   512, 10, 4, 1, 0},
    {"gemma K6 mcg 5376->8192",              6, 1, 5376,  8192, 1, 8, 1, 0},
    {"K3 mul1 4096->4096",                   3, 2, 4096,  4096, 1, 8, 1, 0},
    {"K5 mul1 4096->4096",                   5, 2, 4096,  4096, 1, 8, 1, 0},
    {"K1 mul1 4096->4096",                   1, 2, 4096,  4096, 1, 8, 1, 0},
    {"K2 3inst 4096->4096",                  2, 0, 4096,  4096, 1, 8, 1, 0},
    {"K4 mcg 4096->4096",                    4, 1, 4096,  4096, 1, 8, 1, 0},
    {"K7 3inst 4096->4096",                  7, 0, 4096,  4096, 1, 8, 1, 0},
    {"K8 mcg 4096->4096",                    8, 1, 4096,  4096, 1, 8, 1, 0},
};

typedef void (*kfn)(const half*, const uintptr_t*, float*, int, int, int);

template <int bits, int cb, int W, int M, int MODE, bool L>
kfn pick4() { return bench_kernel<bits, cb, W, M, MODE, L>; }

template <int bits, int cb, int MODE, bool L>
kfn pickWM(int W, int M)
{
    #define PM(w) \
        if (M == 1) return pick4<bits, cb, w, 1, MODE, L>(); \
        if (M == 4) return pick4<bits, cb, w, 4, MODE, L>();
    if (W == 1) { PM(1) }
    if (W == 4) { PM(4) }
    if (W == 8) { PM(8) }
    #undef PM
    return nullptr;
}

template <int MODE, bool L>
kfn pickBC(int bits, int cb, int W, int M)
{
    #define PB(b) \
        if (bits == b) { if (cb == 0) return pickWM<b, 0, MODE, L>(W, M); \
                         if (cb == 1) return pickWM<b, 1, MODE, L>(W, M); \
                         if (cb == 2) return pickWM<b, 2, MODE, L>(W, M); }
    PB(1) PB(2) PB(3) PB(4) PB(5) PB(6) PB(7) PB(8)
    #undef PB
    return nullptr;
}

kfn pick(int mode, bool lds, int bits, int cb, int W, int M)
{
    #define PMODE(md) if (mode == md) return lds ? pickBC<md, true>(bits, cb, W, M) : pickBC<md, false>(bits, cb, W, M);
    PMODE(0) PMODE(1) PMODE(2) PMODE(3) PMODE(4) PMODE(5) PMODE(6) PMODE(7)
    #undef PMODE
    return nullptr;
}

int main(int argc, char** argv)
{
    const char* filt = argc > 1 ? argv[1] : nullptr;
    const int reps = getenv("REPS") ? atoi(getenv("REPS")) : 30;
    // modes: 0 = direct core (baseline), 1..7 = tiles core, (U, T) per mode_un / mode_tn
    const char* ms = getenv("MODES") ? getenv("MODES") : "01234567";
    std::vector<int> modes;
    for (const char* q = ms; *q; ++q) modes.push_back(*q - '0');
    srand(1234);
    bool all_ok = true;
    printf("%-40s", "case");
    for (int md : modes) { if (md) printf("   U%dT%d", mode_un(md), mode_tn(md)); else printf("  base us"); }
    printf("   base GB/s  best GB/s  best Gw/s  speedup  bitwise\n");
    for (auto& c : cases)
    {
        if (filt && !strstr(c.name, filt)) continue;
        const size_t mat_u16 = (size_t) (c.k / 16) * (c.n / 16) * 16 * c.bits;
        const size_t set_bytes = mat_u16 * 2 * c.nmat;
        int rot = (int) std::max<size_t>(2, (160ull << 20) / set_bytes + 1);
        rot = std::min(rot, 64);

        std::vector<uint16_t> hB(mat_u16 * c.nmat);
        for (auto& v : hB) v = (uint16_t) (rand() & 0xffff);
        uint16_t* dB; CK(hipMalloc(&dB, set_bytes * rot));
        for (int r = 0; r < rot; ++r) CK(hipMemcpy((char*) dB + set_bytes * r, hB.data(), set_bytes, hipMemcpyHostToDevice));

        std::vector<uintptr_t> hL((size_t) rot * c.nmat);
        for (int r = 0; r < rot; ++r)
            for (int j = 0; j < c.nmat; ++j)
                hL[(size_t) r * c.nmat + j] = (uintptr_t) ((char*) dB + set_bytes * r + mat_u16 * 2 * j);
        uintptr_t* dL; CK(hipMalloc(&dL, hL.size() * sizeof(uintptr_t)));
        CK(hipMemcpy(dL, hL.data(), hL.size() * sizeof(uintptr_t), hipMemcpyHostToDevice));

        const int a_stride = c.M * c.k;
        std::vector<half> hA((size_t) c.nmat * a_stride);
        for (auto& v : hA) v = __float2half(((rand() % 2001) - 1000) / 1000.0f);
        half* dA; CK(hipMalloc(&dA, hA.size() * 2));
        CK(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));

        const size_t c_elems = (size_t) c.nmat * c.M * c.n;
        float* dC; CK(hipMalloc(&dC, c_elems * 4));


        std::vector<uint32_t> ref(c_elems), out(c_elems);
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        std::vector<float> us;
        std::string bw;
        bool ok = true;
        size_t ndiff = 0;
        for (size_t mi = 0; mi < modes.size(); ++mi)
        {
            kfn f = pick(modes[mi], c.a_lds, c.bits, c.cb, c.W, c.M);
            const int T = mode_tn(modes[mi]);
            dim3 grid(c.n / 16 / T, c.nmat), block(c.W * 32);
            size_t smem = (size_t) c.W * c.M * T * 16 * 4 + (c.a_lds ? (size_t) c.M * c.k * 2 : 0);
            if (!f) { us.push_back(0); continue; }
            CK(hipMemset(dC, 0x7f + (int) mi, c_elems * 4));
            hipLaunchKernelGGL(f, grid, block, smem, 0, dA, dL, dC, c.k, c.n, a_stride);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(mi == 0 ? ref.data() : out.data(), dC, c_elems * 4, hipMemcpyDeviceToHost));
            if (mi > 0)
                for (size_t i = 0; i < c_elems; ++i) if (ref[i] != out[i]) { ok = false; ndiff++; }
            std::vector<float> t;
            for (int rep = 0; rep < reps; ++rep)
            {
                CK(hipEventRecord(e0));
                for (int r = 0; r < rot; ++r)
                    hipLaunchKernelGGL(f, grid, block, smem, 0, dA, dL + (size_t) r * c.nmat, dC, c.k, c.n, a_stride);
                CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1));
                t.push_back(ms * 1000.0f / rot);
            }
            std::sort(t.begin(), t.end());
            us.push_back(t[t.size() / 2]);
        }
        all_ok &= ok;
        double bytes = (double) set_bytes, weights = (double) c.k * c.n * c.nmat;
        float best = 1e30f;
        for (size_t mi = 1; mi < us.size(); ++mi) if (us[mi] > 0) best = std::min(best, us[mi]);
        printf("%-40s", c.name);
        for (float u : us) printf(" %8.1f", u);
        printf("   %9.1f  %9.1f  %9.1f  %6.2fx  %s", bytes / us[0] / 1e3, bytes / best / 1e3, weights / best / 1e3,
               us[0] / best, modes.size() < 2 ? "-" : ok ? "IDENTICAL" : "DIFF");
        if (!ok) printf(" (%zu)", ndiff);
        printf("\n");
        hipFree(dB); hipFree(dL); hipFree(dA); hipFree(dC);
        hipEventDestroy(e0); hipEventDestroy(e1);
    }
    printf(all_ok ? "ALL IDENTICAL\n" : "MISMATCH\n");
    return all_ok ? 0 : 1;
}
