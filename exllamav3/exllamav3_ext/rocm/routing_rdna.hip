// =============================================================================
// routing.cu for RDNA -- the router GEMV differs
// =============================================================================
//
// Generated from upstream v1.5.0 routing.cu (which stays byte-identical to
// upstream; setup.py excludes it from the ROCm build in favor of this file).
// The diff against upstream is confined to the block marked "ROCm sibling"
// (routing_gemv_rdna_kernel + its switch) and the RDNA branch at the top of
// routing_gemv(); everything else is a verbatim copy so the file can be
// regenerated mechanically on rebase and audited by diff. Same exported
// symbols. Why: see the comment above routing_gemv_rdna_kernel.
// EXL3_ROCM_ROUTER_GEMV=0 restores upstream behaviour in the same build.
//
// v1.5.3 sync (three-way merge, base v1.5.0): upstream's deterministic router
// math (exp_det / softplus_det / __fdiv_rn / __fsqrt_rn from det_gemm.cuh, for
// cross-architecture TP agreement) and the gate_i8 / gate_sb arguments that
// select the int8 routing_gemm_det path at bsz > 1. That path declines on ROCm
// (rocm/routing_gemm_rdna.hip), so bsz 2..8 keeps the RDNA router GEMV and
// larger calls hgemm, as before. EXL3_ROCM_ROUTER_DET=0 puts the score
// activations and top-k softmax back on the v1.5.0 fast-math forms
// (__expf, log1pf, sqrtf, IEEE-default division) -- a bisect handle for the
// only v1.5.3 numerics change on DS4's decode path; see RDNA_NOTES.
// =============================================================================

#include <cstdlib>
#include <cuda_fp16.h>
#include "routing.cuh"
#include "det_gemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "reduction.cuh"
#include "hgemm.cuh"

#define MAX_NUM_EXPERTS 512
#define MAX_K 32

using bfloat16 = __nv_bfloat16;

// ROCm sibling: EXL3_ROCM_ROUTER_DET (default 1 = v1.5.3 upstream math). A device global rather
// than a template argument so the dozen top-k kernels keep one instance each; written once per
// device by routing_det_sync() only when the switch is 0 (the default needs no write).
__device__ int g_routing_det_rdna = 1;

static void routing_det_sync(cudaStream_t stream)
{
    static const int det = []
    {
        const char* env = std::getenv("EXL3_ROCM_ROUTER_DET");
        return (!env || atoi(env) != 0) ? 1 : 0;
    }();
    if (det) return;
    static bool done[64] = {};
    int device;
    cudaGetDevice(&device);
    if (done[device]) return;
    static const int zero = 0;
    cuda_check(hipMemcpyToSymbolAsync(HIP_SYMBOL(g_routing_det_rdna), &zero, sizeof(int), 0, cudaMemcpyHostToDevice, stream));
    done[device] = true;
}

__device__ __forceinline__ float routing_exp(float x)
{
    return g_routing_det_rdna ? exp_det(x) : expf(x);
}

// Deterministic (FMA-only exp, correctly rounded division): the selection scores must agree
// bit for bit across tensor-parallel ranks of different architectures
__device__ __forceinline__
float sigmoid_stable_hf(float xf)
{
    if (!g_routing_det_rdna)
    {
        float ez = __expf(-fabsf(xf));
        float base = ez / (1.0f + ez);
        return (xf >= 0.0f) ? 1.0f - base : base;
    }
    float ez = exp_det(-fabsf(xf));
    float base = __fdiv_rn(ez, __fadd_rn(1.0f, ez));
    return (xf >= 0.0f) ? 1.0f - base : base;
}

// Score activations for the nogroup top-k kernels. Both are strictly increasing, so sorting
// by the raw logit when there is no selection bias remains valid for either
#define ROUTING_ACT_SIGMOID 0
#define ROUTING_ACT_SQRTSP 1

template <int ACT>
__device__ __forceinline__
float routing_act(float xf)
{
    if constexpr (ACT == ROUTING_ACT_SQRTSP)
    {
        // sqrt(softplus(x)), matching torch F.softplus(beta = 1, threshold = 20)
        if (!g_routing_det_rdna)
        {
            float sp = xf > 20.0f ? xf : log1pf(__expf(xf));
            return sqrtf(sp);
        }
        return __fsqrt_rn(softplus_det(xf));
    }
    else
        return sigmoid_stable_hf(xf);
}


__device__ __forceinline__
void warp_reduce_best_f32(float& key, float& payload, int& idx)
{
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(USE_ROCM)
        // Monotonic unsigned encoding of the float key, hardware max-reduce, then fetch the
        // winner's values from the lowest tied lane
        unsigned int ku = __float_as_uint(key);
        ku = (ku & 0x80000000u) ? ~ku : (ku | 0x80000000u);
        unsigned int m = __reduce_max_sync(0xffffffffu, ku);
        int src = __ffs(__ballot_sync(0xffffffffu, ku == m)) - 1;
        key = __shfl_sync(0xffffffffu, key, src);
        payload = __shfl_sync(0xffffffffu, payload, src);
        idx = __shfl_sync(0xffffffffu, idx, src);
    #else
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other_key = __shfl_down_sync(0xffffffffu, key, offset);
            float other_payload = __shfl_down_sync(0xffffffffu, payload, offset);
            int other_idx = __shfl_down_sync(0xffffffffu, idx, offset);
            if (other_key > key)
            {
                key = other_key;
                payload = other_payload;
                idx = other_idx;
            }
        }

        key = __shfl_sync(0xffffffffu, key, 0);
        payload = __shfl_sync(0xffffffffu, payload, 0);
        idx = __shfl_sync(0xffffffffu, idx, 0);
    #endif
}


// Top-K selection by one warp over num_candidates (<= MAX_NUM_EXPERTS) keys in shared memory.
// Each lane keeps its strided share of the keys in registers (at most MAX_NUM_EXPERTS / 32),
// every round picks the lane-local best, the warp reduces the round's winner and the owning
// lane retires it. Lane k ends up holding the round-k winner (payload, position). Ties: lowest
// position within a lane, lowest lane across the warp. One pass of K rounds replaces the
// per-warp pre-selection plus multi-warp merge stages: each stage cost K block-wide rounds, and
// for K > 16 a stage that keeps K of every 32 candidates could not shrink the field at all
__device__ __forceinline__
void warp_topk_shared
(
    const float* sh_key,
    const float* sh_payload,
    const int num_candidates,
    const int K,
    float& out_payload,
    int& out_idx
)
{
    constexpr int MAXC = MAX_NUM_EXPERTS / 32;
    int lane_id = threadIdx.x % 32;
    float ck[MAXC];
    #pragma unroll
    for (int j = 0; j < MAXC; ++j)
    {
        int pos = lane_id + 32 * j;
        ck[j] = pos < num_candidates ? sh_key[pos] : -1.0e30f;
    }
    out_payload = 0.0f;
    out_idx = -1;
    for (int k = 0; k < K; ++k)
    {
        float best_key = -1.0e30f;
        int best_j = -1;
        #pragma unroll
        for (int j = 0; j < MAXC; ++j)
            if (ck[j] > best_key) { best_key = ck[j]; best_j = j; }
        int my_pos = best_j >= 0 ? lane_id + 32 * best_j : -1;
        float best_payload = my_pos >= 0 ? (sh_payload ? sh_payload[my_pos] : best_key) : 0.0f;
        int best_idx = my_pos;
        warp_reduce_best_f32(best_key, best_payload, best_idx);
        if (my_pos >= 0 && my_pos == best_idx)
        {
            // Retire the winner (unrolled select: no dynamic register indexing)
            #pragma unroll
            for (int j = 0; j < MAXC; ++j)
                if (j == best_j) ck[j] = -1.0e30f;
        }
        if (lane_id == k)
        {
            out_payload = best_payload;
            out_idx = best_idx;
        }
    }
}


__device__ __forceinline__
void warp_radixsort_posf16(half& key, int& idx, int* src_lane_map)
{
    unsigned int lane_id = threadIdx.x % 32;
    const unsigned int active = 0xffffffffu;

    unsigned int ku = __half_as_ushort(key);

    #pragma unroll
    for (int bit = 0; bit < 15; ++bit)
    {
        unsigned int b = (ku >> bit) & 1;
        unsigned int ones = __ballot_sync(active, b);
        unsigned int zeros = active ^ ones;
        int nzeros = __popc(zeros);

        unsigned int below = (1 << lane_id) - 1;
        int r0 = __popc(zeros & below);
        int r1 = __popc(ones & below);

        int dest = b ? (nzeros + r1) : r0;
        int myrank = __popc(active & below);

        src_lane_map[dest] = lane_id;
        __syncwarp(active);
        int src = src_lane_map[myrank];

        ku = __shfl_sync(active, ku, src);
        idx = __shfl_sync(active, idx, src);
    }
    key = __ushort_as_half(ku);
}


__device__ __forceinline__
void warp_radixsort_posf32_pl(float& key, float& payload, int& idx, int* src_lane_map)
{
    unsigned int lane_id = threadIdx.x % 32;
    const unsigned int active = 0xffffffffu;

    unsigned int ku = __float_as_uint(key);

    #pragma unroll
    for (int bit = 0; bit < 31; ++bit)
    {
        unsigned int b = (ku >> bit) & 1u;
        unsigned int ones = __ballot_sync(active, b);
        unsigned int zeros = active ^ ones;
        int nzeros = __popc(zeros);

        unsigned int below = (1u << lane_id) - 1u;
        int r0 = __popc(zeros & below);
        int r1 = __popc(ones & below);

        int dest = b ? (nzeros + r1) : r0;
        int myrank = __popc(active & below);

        src_lane_map[dest] = lane_id;
        __syncwarp(active);
        int src = src_lane_map[myrank];

        ku = __shfl_sync(active, ku, src);
        payload = __shfl_sync(active, payload, src);
        idx = __shfl_sync(active, idx, src);
    }
    key = __uint_as_float(ku);
}


// Single-token router gemv on a transposed gate copy: scores = x @ gate_t.T. One warp per
// expert; cheaper than a cublas call at this size
#define RGEMV_WARPS 8

__global__ __launch_bounds__(RGEMV_WARPS * 32)
void routing_gemv_kernel
(
    const half* __restrict__ x,         // (k)
    const half* __restrict__ gate_t,    // (E, k)
    half* __restrict__ scores,          // (E)
    const int k,
    const int E
)
{
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * RGEMV_WARPS + warp;
    if (row >= E) return;

    const half2* x2 = (const half2*) x;
    const half2* w2 = (const half2*) (gate_t + (size_t) row * k);

    float sum = 0.0f;
    for (int j = lane; j < k / 2; j += 32)
    {
        float2 xf = __half22float2(x2[j]);
        float2 wf = __half22float2(w2[j]);
        sum = fmaf(xf.x, wf.x, sum);
        sum = fmaf(xf.y, wf.y, sum);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);

    if (lane == 0)
        scores[row] = __float2half_rn(sum);
}

// -----------------------------------------------------------------------------
// ROCm sibling: router GEMV with loads in flight, and m = 2..8 rows (RDNA)
// -----------------------------------------------------------------------------
// The kernel above is upstream's, kept verbatim for EXL3_ROCM_ROUTER_GEMV=0. On
// gfx1151 it runs DS4's router (E = 256, k = 4096, 2 MB of fp16 per layer) in
// 38 us per call, ~55 GB/s against a ~10 us roofline (PROFILE.md section 4 and
// the c6c19ca decode profile: 1.6 ms/token): a grid of E / 8 = 32 blocks covers
// 32 of 40 CUs with 2 waves per SIMD, and each wave has one 128-byte load in
// flight per loop trip, so the kernel is latency-bound.
//
// routing_gemv_rdna_kernel keeps upstream's arithmetic exactly -- lane l owns
// the half2 columns j = l (mod 32) and accumulates them in increasing j with the
// same two fmaf per column, then the same __shfl_down tree -- so every score is
// bit-identical to the kernel above. What changes is scheduling: RG_U column
// loads per lane are issued together before the chain consumes them (RG_U x 128
// bytes in flight per wave), and the block is RG_WARPS = 2 waves so the E-wave
// grid spreads over every CU. With M > 1 the kernel reads the weights once for
// M token rows (m = 2..8, the MTP / draft verify step), each row with the m == 1
// chain; upstream sends m > 1 to hgemm (two split-K launches on this build).
#define RG_WARPS 2
#define RG_U 16

template <int M>
__global__ __launch_bounds__(RG_WARPS * 32)
void routing_gemv_rdna_kernel
(
    const half* __restrict__ x,         // (m, k)
    const half* __restrict__ gate_t,    // (E, k)
    half* __restrict__ scores,          // (m, E)
    const int k,
    const int E,
    const int m
)
{
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * RG_WARPS + warp;
    if (row >= E) return;

    const int k2 = k / 2;
    const half2* x2 = (const half2*) x;
    const half2* w2 = (const half2*) (gate_t + (size_t) row * k);

    float sum[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) sum[r] = 0.0f;

    int j0 = lane;
    for (; j0 + 32 * (RG_U - 1) < k2; j0 += 32 * RG_U)
    {
        half2 w[RG_U];
        #pragma unroll
        for (int u = 0; u < RG_U; ++u) w[u] = w2[j0 + 32 * u];
        #pragma unroll
        for (int u = 0; u < RG_U; ++u)
        {
            float2 wf = __half22float2(w[u]);
            #pragma unroll
            for (int r = 0; r < M; ++r)
            {
                if (r >= m) break;
                float2 xf = __half22float2(x2[(size_t) r * k2 + j0 + 32 * u]);
                sum[r] = fmaf(xf.x, wf.x, sum[r]);
                sum[r] = fmaf(xf.y, wf.y, sum[r]);
            }
        }
    }
    for (int j = j0; j < k2; j += 32)
    {
        float2 wf = __half22float2(w2[j]);
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            if (r >= m) break;
            float2 xf = __half22float2(x2[(size_t) r * k2 + j]);
            sum[r] = fmaf(xf.x, wf.x, sum[r]);
            sum[r] = fmaf(xf.y, wf.y, sum[r]);
        }
    }

    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        if (r >= m) break;
        float s = sum[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down_sync(0xffffffffu, s, offset);
        if (lane == 0)
            scores[(size_t) r * E + row] = __float2half_rn(s);
    }
}

// m == 1 form with 16-byte weight loads. The kernel above issues one dword load
// per lane per column step -- 512 K load instructions for DS4's 2 MB router,
// which measured 22 us (95 GB/s): address/TA-rate bound, not latency. Here each
// lane loads 16 bytes (4 adjacent half2 columns) of a 128-column block, the wave
// writes the block to its own LDS slice, and each lane reads back ITS columns
// (l, l + 32, l + 64, l + 96) -- so the chain is still upstream's, column for
// column, and the result bit-identical. x is staged once per block in LDS.
// Whole 128-column blocks go through LDS and a tail (k / 2 % 128 columns) is
// read per lane directly. Needs k % 8 == 0, 16-byte aligned rows and the LDS
// budget (x plus U blocks per wave), all checked by the host; otherwise the
// dword-load kernel above runs. EXL3_ROCM_ROUTER_U = 2 / 4 / 8 / 16 (default 8)
// picks the blocks in flight per wave. DS4 router (E 256, k 4096), rocprofv3,
// GEMV alone: U2 20.5, U4 19.7, U8 13.2 us (the dword kernel: 22.3 us).
#define RGW_WARPS 4

// Stages x into LDS (every thread of the block must enter), then returns this
// warp's row dot product in lane 0 (the shfl_down tree). Rows >= E return 0.
template <int U>
__device__ __forceinline__ float routing_gemv_rdna_w16_row
(
    const half* __restrict__ x,
    const half* __restrict__ gate_t,
    const int k,
    const int row,
    const bool valid,
    uint32_t* rg_sh
)
{
    const int k2 = k / 2;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    uint32_t* xs = rg_sh;
    uint32_t* wt = rg_sh + k2 + warp * U * 128;

    for (int i = threadIdx.x; i < k2 / 4; i += RGW_WARPS * 32)
        ((uint4*) xs)[i] = ((const uint4*) x)[i];
    __syncthreads();
    if (!valid) return 0.0f;

    const uint4* w4 = (const uint4*) (gate_t + (size_t) row * k);
    const half2* xs2 = (const half2*) xs;
    const half2* wt2 = (const half2*) wt;
    const int nb = k2 / 128;
    float sum = 0.0f;

    int b0 = 0;
    for (; b0 + U <= nb; b0 += U)
    {
        uint4 v[U];
        #pragma unroll
        for (int u = 0; u < U; ++u) v[u] = w4[(b0 + u) * 32 + lane];
        #pragma unroll
        for (int u = 0; u < U; ++u) ((uint4*) (wt + u * 128))[lane] = v[u];
        __syncwarp();
        #pragma unroll
        for (int u = 0; u < U; ++u)
        {
            #pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                float2 wf = __half22float2(wt2[u * 128 + q * 32 + lane]);
                float2 xf = __half22float2(xs2[(b0 + u) * 128 + q * 32 + lane]);
                sum = fmaf(xf.x, wf.x, sum);
                sum = fmaf(xf.y, wf.y, sum);
            }
        }
        __syncwarp();
    }
    for (; b0 < nb; ++b0)
    {
        ((uint4*) wt)[lane] = w4[b0 * 32 + lane];
        __syncwarp();
        #pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            float2 wf = __half22float2(wt2[q * 32 + lane]);
            float2 xf = __half22float2(xs2[b0 * 128 + q * 32 + lane]);
            sum = fmaf(xf.x, wf.x, sum);
            sum = fmaf(xf.y, wf.y, sum);
        }
        __syncwarp();
    }
    const half2* w2 = (const half2*) (gate_t + (size_t) row * k);
    for (int j = nb * 128 + lane; j < k2; j += 32)
    {
        float2 wf = __half22float2(w2[j]);
        float2 xf = __half22float2(xs2[j]);
        sum = fmaf(xf.x, wf.x, sum);
        sum = fmaf(xf.y, wf.y, sum);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    return sum;
}

template <int U>
__global__ __launch_bounds__(RGW_WARPS * 32)
void routing_gemv_rdna_w16_kernel
(
    const half* __restrict__ x,         // (k)
    const half* __restrict__ gate_t,    // (E, k)
    half* __restrict__ scores,          // (E)
    const int k,
    const int E
)
{
    extern __shared__ uint32_t rg_sh[];
    const int row = blockIdx.x * RGW_WARPS + threadIdx.x / 32;
    float sum = routing_gemv_rdna_w16_row<U>(x, gate_t, k, row, row < E, rg_sh);
    if (row < E && threadIdx.x % 32 == 0)
        scores[row] = __float2half_rn(sum);
}

static int routing_gemv_rdna_u()
{
    static const int u = []
    {
        const char* env = std::getenv("EXL3_ROCM_ROUTER_U");
        int v = env ? atoi(env) : 8;
        return (v == 2 || v == 4 || v == 8 || v == 16) ? v : 8;
    }();
    return u;
}

// EXL3_ROCM_ROUTER_GEMV=0 restores upstream's routing_gemv (the kernel above at
// m == 1, hgemm at m > 1). Read once per process.
static bool routing_gemv_rdna_enabled()
{
    static const bool on = []
    {
        const char* env = std::getenv("EXL3_ROCM_ROUTER_GEMV");
        return !env || atoi(env) != 0;
    }();
    return on;
}

void routing_gemv
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    const c10::optional<at::Tensor>& gate_t,
    const c10::optional<at::Tensor>& gate_i8,
    const c10::optional<at::Tensor>& gate_sb,
    at::Tensor& scores,
    cudaStream_t stream
)
{
    // Single rows take the fixed-order FMA GEMV (exact on every architecture, and the fastest);
    // with the int8 gate available every other row count takes the deterministic int8
    // tensor-core projection (routing_gemm.cu), so tensor-parallel ranks of any sm_80+
    // architecture routing on identical streams select identical experts (the int8 kernels
    // need cp.async and mma.m16n8k32, both sm_80+). Pre-Ampere devices fall back to cuBLAS,
    // which is device-dependent — uniform per-arch fleets still agree with each other
    int k = hidden.size(-1);
    int E = scores.size(-1);
    bool bsz1 = hidden.numel() == k;

    // RDNA: the m = 1..8 router GEMV (see above)
    if (routing_gemv_rdna_enabled() && gate_t.has_value() && !(k & 1) && hidden.is_contiguous())
    {
        int m = (int) (hidden.numel() / k);
        if (m >= 1 && m <= 8 && scores.numel() == (int64_t) m * E)
        {
            dim3 grid(CEIL_DIVIDE(E, RG_WARPS));
            dim3 block(RG_WARPS * 32);
            const half* xp = (const half*) hidden.data_ptr();
            const half* wp = (const half*) gate_t.value().data_ptr();
            half* sp = (half*) scores.data_ptr();
            if (m == 1 && !(k & 7) && !(((uintptr_t) xp | (uintptr_t) wp) & 15)
                && (size_t) (k / 2 + RGW_WARPS * routing_gemv_rdna_u() * 128) * 4 <= 48 * 1024)
            {
                const int U = routing_gemv_rdna_u();
                dim3 grid_w(CEIL_DIVIDE(E, RGW_WARPS));
                dim3 block_w(RGW_WARPS * 32);
                size_t smem = (size_t) (k / 2 + RGW_WARPS * U * 128) * 4;
                if (U == 2)       routing_gemv_rdna_w16_kernel<2><<<grid_w, block_w, smem, stream>>>(xp, wp, sp, k, E);
                else if (U == 4)  routing_gemv_rdna_w16_kernel<4><<<grid_w, block_w, smem, stream>>>(xp, wp, sp, k, E);
                else if (U == 16) routing_gemv_rdna_w16_kernel<16><<<grid_w, block_w, smem, stream>>>(xp, wp, sp, k, E);
                else              routing_gemv_rdna_w16_kernel<8><<<grid_w, block_w, smem, stream>>>(xp, wp, sp, k, E);
            }
            else if (m == 1) routing_gemv_rdna_kernel<1><<<grid, block, 0, stream>>>(xp, wp, sp, k, E, m);
            else if (m <= 2) routing_gemv_rdna_kernel<2><<<grid, block, 0, stream>>>(xp, wp, sp, k, E, m);
            else if (m <= 4) routing_gemv_rdna_kernel<4><<<grid, block, 0, stream>>>(xp, wp, sp, k, E, m);
            else             routing_gemv_rdna_kernel<8><<<grid, block, 0, stream>>>(xp, wp, sp, k, E, m);
            return;
        }
    }

    if (!bsz1 && gate_i8.has_value() && gate_sb.has_value() && routing_gemm_det_fits(hidden, gate_i8.value(), gate_sb.value(), scores))
    {
        routing_gemm_det_(hidden, gate_i8.value(), gate_sb.value(), scores, stream);
    }
    else if (bsz1 && gate_t.has_value() && !(k & 1))
    {
        routing_gemv_kernel<<<CEIL_DIVIDE(E, RGEMV_WARPS), RGEMV_WARPS * 32, 0, stream>>>
        (
            (const half*) hidden.data_ptr(),
            (const half*) gate_t.value().data_ptr(),
            (half*) scores.data_ptr(),
            k, E
        );
    }
    else
    {
        hgemm(hidden, gate, scores);
    }
}


template <int ACT>
__launch_bounds__(MAX_NUM_EXPERTS)
__global__ void routing_ds3_nogroup_topk_kernel
(
    const half* __restrict__ scores,
    const half* __restrict__ bias,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const float scaling_factor,
    const int num_experts,
    const int K,
    const int bsz
)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    bool mask = t < num_experts;

    scores += num_experts * row;
    topk_indices += K * row;
    topk_weights += K * row;

    extern __shared__ unsigned char sh[];
    float* sh_key = reinterpret_cast<float*>(sh);
    float* sh_payload = sh_key + num_experts;

    // Selection key (biased activation, or the raw logit without a bias: the activations are
    // monotonic) and the output payload for every expert
    if (mask)
    {
        float logit = __half2float(scores[t]);
        float act = bias ? routing_act<ACT>(logit) : 0.0f;
        sh_key[t] = bias ? act + __half2float(bias[t]) : logit;
        sh_payload[t] = bias ? act : logit;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float o;
        int out_idx;
        warp_topk_shared(sh_key, sh_payload, num_experts, K, o, out_idx);
        if (lane_id < K && !bias)
            o = routing_act<ACT>(o);

        float sum = warp_reduce_sum_first_k(o, K) + 1e-20f;
        if (lane_id < K)
        {
            topk_indices[lane_id] = (int64_t) out_idx;
            topk_weights[lane_id] = __float2half_rn(o * scaling_factor / sum);
        }
    }
}


template <int ACT>
__launch_bounds__(MAX_NUM_EXPERTS)
__global__ void routing_ds3_nogroup_kernel
(
    const half* __restrict__ scores,
    const half* __restrict__ bias,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const float scaling_factor,
    const int num_experts,
    const int K,
    const int bsz
)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int num_warps = CEIL_DIVIDE(num_experts, 32);
    bool mask = t < num_experts;

    scores += num_experts * row;
    topk_indices += K * row;
    topk_weights += K * row;

    extern __shared__ unsigned char sh[];
    int K_ = K + (K & 1);
    float* sh_v = reinterpret_cast<float*>(sh);
    float* sh_o = reinterpret_cast<float*>(sh_v + K_ * num_warps);
    int* sh_idx = reinterpret_cast<int*>(sh_o + K_ * num_warps);
    int* perm = reinterpret_cast<int*>(sh_idx + K_ * num_warps);
    float* reduce = reinterpret_cast<float*>(perm + 32 * num_warps);

    // Input activation
    int idx = mask ? t : -1;  // output index
    float v = mask ? routing_act<ACT>(__half2float(scores[t])) : 0.0f;  // sort key
    float o = v;  // output weight

    // Add bias and shift sigmoid(logits) to be non-negative before radix sort
    if (bias)
    {
        v += mask ? __half2float(bias[t]) : 1e30;

        float minv = v;
        for (int offset = 32 >> 1; offset > 0; offset >>= 1)
            minv = fminf(minv, __shfl_down_sync(0xffffffff, minv, offset));
        if (lane_id == 0)
            reduce[warp_id] = minv;

        __syncthreads();

        if (warp_id == 0)
        {
            minv = lane_id < num_warps ? reduce[lane_id] : 1e30;
            for (int offset = 32 >> 1; offset > 0; offset >>= 1)
                minv = fminf(minv, __shfl_down_sync(0xffffffff, minv, offset));
            if (lane_id == 0)
                reduce[0] = minv;
        }

        __syncthreads();

        v -= reduce[0];
        if (!mask) v = 0.0f;
    }

    // Sort by v
    warp_radixsort_posf32_pl(v, o, idx, perm + warp_id * 32);

    while (num_warps > 1)
    {
        if (warp_id < num_warps && lane_id >= (32 - K))
        {
            int kpos = (32 - 1) - lane_id;
            sh_v[warp_id * K + kpos] = v;
            sh_o[warp_id * K + kpos] = o;
            sh_idx[warp_id * K + kpos] = idx;
        }
        __syncthreads();

        int num_experts_k = K * num_warps;
        num_warps = CEIL_DIVIDE(num_experts_k, 32);

        if (warp_id < num_warps)
        {
            if (t < num_experts_k && mask)
            {
                v = sh_v[t];
                o = sh_o[t];
                idx = sh_idx[t];
            }
            else
            {
                v = 0.0f;
                o = 0.0f;
                idx = -1;
            }
            warp_radixsort_posf32_pl(v, o, idx, perm + warp_id * 32);
        }
        __syncthreads();
    }

    // Normalize output in warp 0 lanes 32-K .. K, store result
    if (warp_id == 0)
    {
        float sum = warp_reduce_sum_last_k(o, K) + 1e-20;
        o *= scaling_factor / sum;

        if (lane_id >= (32 - K))
        {
            int kpos = (32 - 1) - lane_id;
            topk_indices[kpos] = (int64_t) idx;
            topk_weights[kpos] = __float2half_rn(o);
        }
    }
}


__launch_bounds__(MAX_NUM_EXPERTS)
__global__ void routing_std_topk_kernel
(
    const half* __restrict__ scores,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const bfloat16* __restrict__ per_expert_scale,
    const half* __restrict__ bias,
    int num_experts,
    int K,
    int bsz
)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int num_warps = CEIL_DIVIDE(num_experts, 32);

    scores += num_experts * row;
    topk_indices += K * row;
    topk_weights += K * row;

    extern __shared__ unsigned char sh[];
    float* sh_key = reinterpret_cast<float*>(sh);
    float* max_red = sh_key + num_experts;

    bool mask = t < num_experts;
    float logit = mask ? __half2float(scores[t]) : -1.0e30f;
    // Router bias (gpt-oss): biased logits drive both the top-k selection and the softmax
    if (bias && mask)
        logit += __half2float(bias[t]);
    if (mask)
        sh_key[t] = logit;

    // Max logit for a stable softmax
    float max_logit = logit;
    max_logit = warp_reduce_max_f(max_logit);
    max_logit = __shfl_sync(0xffffffffu, max_logit, 0);
    if (num_warps > 1)
    {
        if (lane_id == 0) max_red[warp_id] = max_logit;
        __syncthreads();
        max_logit = lane_id < num_warps ? max_red[lane_id] : -1.0e30f;
        max_logit = warp_reduce_max_f(max_logit);
        max_logit = __shfl_sync(0xffffffffu, max_logit, 0);
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float e;
        int out_idx;
        warp_topk_shared(sh_key, nullptr, num_experts, K, e, out_idx);
        e = lane_id < K ? routing_exp(e - max_logit) : 0.0f;
        float sum = warp_reduce_sum_first_k(e, K) + 1e-20f;
        e /= sum;

        if (lane_id < K)
        {
            if (per_expert_scale)
                e *= __bfloat162float(per_expert_scale[out_idx]);
            topk_indices[lane_id] = (int64_t) out_idx;
            topk_weights[lane_id] = __float2half_rn(e);
        }
    }
}


__launch_bounds__(MAX_NUM_EXPERTS)
__global__ void routing_std_kernel
(
    const half* __restrict__ scores,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const bfloat16* __restrict__ per_expert_scale,
    int num_experts,
    int K,
    int bsz
)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int num_warps = CEIL_DIVIDE(num_experts, 32);

    scores += num_experts * row;
    topk_indices += K * row;
    topk_weights += K * row;

    extern __shared__ unsigned char sh[];
    int K_ = K + (K & 1);
    half* sh_v = reinterpret_cast<half*>(sh);
    int* sh_idx = reinterpret_cast<int*>(sh_v + K_ * num_warps);
    int* perm = reinterpret_cast<int*>(sh_idx + K_ * num_warps);
    half* max_red = sh_v;

    // Get max logit, shift prior to sorting so int order matches float order (same sign). Also
    // stabilizes softmax at output

    half max_logit = t < num_experts ? scores[t] : __ushort_as_half(0xfbff);
    max_logit = warp_reduce_max_h(max_logit);
    max_logit = __shfl_sync(0xffffffffu, max_logit, 0);

    if (num_warps > 1)
    {
        max_red[warp_id] = max_logit;
        __syncthreads();
        max_logit = lane_id < num_warps ? max_red[lane_id] : __ushort_as_half(0xfbff);
        max_logit = warp_reduce_max_h(max_logit);
        max_logit = __shfl_sync(0xffffffffu, max_logit, 0);
        // max_red aliases sh_v: no warp may start writing its top-K into sh_v below until every
        // warp has finished reading the reduction slots
        __syncthreads();
    }

    // Input logit, shifted

    int idx = t < num_experts ? t : -1;  // output index
    half v = t < num_experts ? __hsub(scores[t], max_logit) : __ushort_as_half(0xfbff);

    // Sort by v

    warp_radixsort_posf16(v, idx, perm + warp_id * 32);

    while (num_warps > 1)
    {
        if (warp_id < num_warps && lane_id < K)
        {
            int kpos = lane_id;
            sh_v[warp_id * K + kpos] = v;
            sh_idx[warp_id * K + kpos] = idx;
        }
        __syncthreads();

        int num_experts_k = K * num_warps;
        num_warps = CEIL_DIVIDE(num_experts_k, 32);

        if (warp_id < num_warps)
        {
            if (t < num_experts_k)
            {
                v = sh_v[t];
                idx = sh_idx[t];
            }
            else
            {
                v = __ushort_as_half(0xfbff);
                idx = -1;
            }
            warp_radixsort_posf16(v, idx, perm + warp_id * 32);
        }
        __syncthreads();
    }

    // Normalize output in first K lanes, store result

    if (warp_id == 0)
    {

        float e = routing_exp(__half2float(v));
        float sum = warp_reduce_sum_first_k(e, K) + 1e-20;
        e /= sum;

        if (lane_id < K)
        {
            if (per_expert_scale)
                e *= __bfloat162float(per_expert_scale[idx]);
            int kpos = lane_id;
            topk_indices[kpos] = (int64_t) idx;
            topk_weights[kpos] = __float2half_rn(e);
        }
    }
}


// -----------------------------------------------------------------------------
// ROCm sibling: router GEMV + top-k in one launch (m == 1)
// -----------------------------------------------------------------------------
// routing_ds3_nogroup at bsz 1 is two dependent launches, the router GEMV and a
// one-block top-k over its 256 scores (4 us of kernel plus a ~2.5 us dispatch
// gap, 40 times per DS4 token). Here the GEMV blocks store their scores, then
// the last block to arrive (release fence + counter, acquire fence; the counter
// self-resets, so no memset and nothing to patch under graph capture) runs
// routing_ds3_nogroup_topk_kernel's body verbatim on the stored scores: same
// keys, same payloads, same warp_topk_shared, same weight normalization, so
// indices and weights are bit-identical to the two-launch form. Wait-free (the
// classic last-block pattern), so it cannot hang. EXL3_ROCM_ROUTER_FUSE=0 keeps
// two launches.
__device__ unsigned int g_routing_fused_counter[64];

template <int ACT, int U>
__global__ __launch_bounds__(RGW_WARPS * 32)
void routing_ds3_nogroup_fused_kernel
(
    const half* __restrict__ x,         // (k)
    const half* __restrict__ gate_t,    // (E, k)
    half* __restrict__ scores,          // (E)
    const half* __restrict__ bias,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const float scaling_factor,
    const int k,
    const int num_experts,
    const int K,
    unsigned int* __restrict__ counter
)
{
    extern __shared__ uint32_t rg_sh[];
    __shared__ int sh_last;
    const int t = threadIdx.x;
    const int lane_id = t % 32;
    const int warp_id = t / 32;
    const int row = blockIdx.x * RGW_WARPS + warp_id;

    float sum = routing_gemv_rdna_w16_row<U>(x, gate_t, k, row, row < num_experts, rg_sh);
    if (row < num_experts && lane_id == 0)
        scores[row] = __float2half_rn(sum);

    // Arrive: every thread's stores released before thread 0 counts the block
    __threadfence();
    __syncthreads();
    if (t == 0)
    {
        unsigned int old = atomicAdd(counter, 1u);
        sh_last = (old == gridDim.x - 1);
    }
    __syncthreads();
    if (!sh_last) return;
    __threadfence();
    if (t == 0) *counter = 0u;

    // routing_ds3_nogroup_topk_kernel, row 0, over the whole block's threads
    float* sh_key = reinterpret_cast<float*>(rg_sh);
    float* sh_payload = sh_key + num_experts;
    for (int e = t; e < num_experts; e += RGW_WARPS * 32)
    {
        float logit = __half2float(scores[e]);
        float act = bias ? routing_act<ACT>(logit) : 0.0f;
        sh_key[e] = bias ? act + __half2float(bias[e]) : logit;
        sh_payload[e] = bias ? act : logit;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float o;
        int out_idx;
        warp_topk_shared(sh_key, sh_payload, num_experts, K, o, out_idx);
        if (lane_id < K && !bias)
            o = routing_act<ACT>(o);

        float s = warp_reduce_sum_first_k(o, K) + 1e-20f;
        if (lane_id < K)
        {
            topk_indices[lane_id] = (int64_t) out_idx;
            topk_weights[lane_id] = __float2half_rn(o * scaling_factor / s);
        }
    }
}

static bool routing_fuse_enabled()
{
    static const bool on = []
    {
        const char* env = std::getenv("EXL3_ROCM_ROUTER_FUSE");
        return !env || atoi(env) != 0;
    }();
    return on;
}

// Returns true when the fused launch ran (bsz 1, transposed gate, shapes/alignment fit)
static bool routing_ds3_nogroup_fused_try
(
    const at::Tensor& hidden,
    const c10::optional<at::Tensor>& gate_t,
    at::Tensor& scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& topk_indices,
    at::Tensor& topk_weights,
    const float scaling_factor,
    const int act_fn,
    cudaStream_t stream
)
{
    if (!routing_gemv_rdna_enabled() || !routing_fuse_enabled()) return false;
    if (!gate_t.has_value() || !hidden.is_contiguous()) return false;
    const int k = hidden.size(-1);
    const int E = scores.size(-1);
    if (hidden.numel() != k || scores.numel() != E || topk_indices.size(0) != 1) return false;
    if (hidden.dtype() != at::kHalf || gate_t.value().dtype() != at::kHalf) return false;
    if (scores.dtype() != at::kHalf || topk_indices.dtype() != at::kLong || topk_weights.dtype() != at::kHalf) return false;
    if (bias.has_value() && bias.value().dtype() != at::kHalf) return false;
    const int K = topk_indices.size(1);
    if (E > MAX_NUM_EXPERTS || K > MAX_K || K > E || K > 32) return false;
    if (k & 7) return false;
    const half* xp = (const half*) hidden.data_ptr();
    const half* wp = (const half*) gate_t.value().data_ptr();
    if (((uintptr_t) xp | (uintptr_t) wp) & 15) return false;
    if (k / 2 < 2 * E) return false;   // top-k keys reuse the x staging area
    const int U = routing_gemv_rdna_u();
    size_t smem = (size_t) (k / 2 + RGW_WARPS * U * 128) * 4;
    if (smem > 48 * 1024) return false;
    int device = hidden.get_device();
    if (device < 0 || device >= 64) return false;
    // Symbol address resolved once per device (device globals are per device)
    static unsigned int* counter_addr[64] = {};
    unsigned int* counters = counter_addr[device];
    if (!counters)
    {
        if (hipGetSymbolAddress((void**) &counters, HIP_SYMBOL(g_routing_fused_counter)) != hipSuccess)
        {
            (void) hipGetLastError();
            return false;
        }
        counter_addr[device] = counters;
    }

    dim3 grid(CEIL_DIVIDE(E, RGW_WARPS));
    dim3 block(RGW_WARPS * 32);
    #define RG_FUSED_ARGS xp, wp, (half*) scores.data_ptr(), (const half*) OPTPTR(bias), \
        (int64_t*) topk_indices.data_ptr(), (half*) topk_weights.data_ptr(), scaling_factor, \
        k, E, K, counters
    #define RG_FUSED_U(ACT) \
        if (U == 2)       routing_ds3_nogroup_fused_kernel<ACT, 2><<<grid, block, smem, stream>>>(RG_FUSED_ARGS); \
        else if (U == 4)  routing_ds3_nogroup_fused_kernel<ACT, 4><<<grid, block, smem, stream>>>(RG_FUSED_ARGS); \
        else if (U == 16) routing_ds3_nogroup_fused_kernel<ACT, 16><<<grid, block, smem, stream>>>(RG_FUSED_ARGS); \
        else              routing_ds3_nogroup_fused_kernel<ACT, 8><<<grid, block, smem, stream>>>(RG_FUSED_ARGS);
    if (act_fn == ROUTING_ACT_SQRTSP) { RG_FUSED_U(ROUTING_ACT_SQRTSP) }
    else                              { RG_FUSED_U(ROUTING_ACT_SIGMOID) }
    #undef RG_FUSED_U
    #undef RG_FUSED_ARGS
    cuda_check(cudaPeekAtLastError());
    return true;
}

/*
DS3 routing for n_group == 1, topk_group

hidden: Input hidden states, float16, shape (..., hidden_dim)
gate: Router gate matrix, float16, shape (hidden_dim, num_experts)
scores: Output routing logits buffer, float16, shape (bsz, num_experts)
bias: Pre-topk bias, float16, shape (1, num_experts)
topk_indices: int64, shape (bsz, k)
topk_weights: float16, shape (bsz, k)
routed_scaling_factor: float32
act_fn: score activation, ROUTING_ACT_SIGMOID (DS3/dots) or ROUTING_ACT_SQRTSP (DSv4)
*/

void routing_ds3_nogroup
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    at::Tensor scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const float scaling_factor,
    const c10::optional<at::Tensor>& gate_t,
    const int act_fn,
    const c10::optional<at::Tensor>& gate_i8,
    const c10::optional<at::Tensor>& gate_sb
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    routing_det_sync(stream);

    // RDNA: bsz 1 router GEMV + top-k in one launch (see above)
    if (routing_ds3_nogroup_fused_try(hidden, gate_t, scores, bias, topk_indices, topk_weights,
                                      scaling_factor, act_fn, stream))
        return;

    routing_gemv(hidden, gate, gate_t, gate_i8, gate_sb, scores, stream);

    TORCH_CHECK_DTYPE(hidden, kHalf);
    TORCH_CHECK_DTYPE(gate, kHalf);
    TORCH_CHECK_SHAPES_OPT(bias, 0, scores, 1, 1);
    TORCH_CHECK_SHAPES(scores, 0, topk_indices, 0, 1);
    TORCH_CHECK_SHAPES(scores, 0, topk_weights, 0, 1);
    TORCH_CHECK_SHAPES(hidden, -1, gate, 0, 1);
    TORCH_CHECK_SHAPES(gate, 1, scores, -1, 1);
    TORCH_CHECK_SHAPES(topk_indices, 1, topk_weights, 1, 1);
    TORCH_CHECK_DTYPE(scores, kHalf);
    TORCH_CHECK_DTYPE_OPT(bias, kHalf);
    TORCH_CHECK_DTYPE(topk_indices, kLong);
    TORCH_CHECK_DTYPE(topk_weights, kHalf);

    int bsz = scores.size(0);
    int num_experts = scores.size(1);
    int K = topk_indices.size(1);

    TORCH_CHECK(num_experts <= MAX_NUM_EXPERTS, "Too many experts");
    TORCH_CHECK(K <= MAX_K, "Too many experts per token");
    TORCH_CHECK(K <= num_experts, "K cannot exceed number of experts");

    int num_warps = CEIL_DIVIDE(num_experts, 32);
    int num_threads = num_warps * 32;

    // The iterative top-K kernel beats the radix-sort kernel at every measured size
    size_t shmem = num_experts * 2 * sizeof(float);
    auto kernel = act_fn == ROUTING_ACT_SQRTSP ?
        routing_ds3_nogroup_topk_kernel<ROUTING_ACT_SQRTSP> :
        routing_ds3_nogroup_topk_kernel<ROUTING_ACT_SIGMOID>;
    kernel<<<bsz, num_threads, shmem, stream>>>
    (
        (const half*) scores.data_ptr(),
        (const half*) OPTPTR(bias),
        (int64_t*) topk_indices.data_ptr(),
        (half*) topk_weights.data_ptr(),
        scaling_factor,
        num_experts,
        K,
        bsz
    );
    cuda_check(cudaPeekAtLastError());
}


void routing_ds3_nogroup_logits
(
    at::Tensor scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const float scaling_factor,
    const bool use_topk,
    const int act_fn
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    routing_det_sync(stream);

    TORCH_CHECK_SHAPES_OPT(bias, 0, scores, 1, 1);
    TORCH_CHECK_SHAPES(scores, 0, topk_indices, 0, 1);
    TORCH_CHECK_SHAPES(scores, 0, topk_weights, 0, 1);
    TORCH_CHECK_SHAPES(topk_indices, 1, topk_weights, 1, 1);
    TORCH_CHECK_DTYPE(scores, kHalf);
    TORCH_CHECK_DTYPE_OPT(bias, kHalf);
    TORCH_CHECK_DTYPE(topk_indices, kLong);
    TORCH_CHECK_DTYPE(topk_weights, kHalf);

    int bsz = scores.size(0);
    int num_experts = scores.size(1);
    int K = topk_indices.size(1);

    TORCH_CHECK(num_experts <= MAX_NUM_EXPERTS, "Too many experts");
    TORCH_CHECK(K <= MAX_K, "Too many experts per token");
    TORCH_CHECK(K <= num_experts, "K cannot exceed number of experts");

    int num_warps = CEIL_DIVIDE(num_experts, 32);
    int num_threads = num_warps * 32;

    // The radix-sort variant merges K of every 32 candidates per stage, which stops converging
    // past K = 16 (the iterative kernel is the one the model paths use)
    TORCH_CHECK(use_topk || K <= 16, "routing_ds3_nogroup_logits: radix-sort kernel supports K <= 16");
    if (use_topk)
    {
        size_t shmem = num_experts * 2 * sizeof(float);
        auto kernel = act_fn == ROUTING_ACT_SQRTSP ?
            routing_ds3_nogroup_topk_kernel<ROUTING_ACT_SQRTSP> :
            routing_ds3_nogroup_topk_kernel<ROUTING_ACT_SIGMOID>;
        kernel<<<bsz, num_threads, shmem, stream>>>
        (
            (const half*) scores.data_ptr(),
            (const half*) OPTPTR(bias),
            (int64_t*) topk_indices.data_ptr(),
            (half*) topk_weights.data_ptr(),
            scaling_factor,
            num_experts,
            K,
            bsz
        );
    }
    else
    {
        int K_ = K + (K & 1);
        size_t shmem = num_warps * K_ * (2 * sizeof(float) + sizeof(int))
                     + num_threads * sizeof(int)
                     + num_warps * sizeof(float);
        auto kernel = act_fn == ROUTING_ACT_SQRTSP ?
            routing_ds3_nogroup_kernel<ROUTING_ACT_SQRTSP> :
            routing_ds3_nogroup_kernel<ROUTING_ACT_SIGMOID>;
        kernel<<<bsz, num_threads, shmem, stream>>>
        (
            (const half*) scores.data_ptr(),
            (const half*) OPTPTR(bias),
            (int64_t*) topk_indices.data_ptr(),
            (half*) topk_weights.data_ptr(),
            scaling_factor,
            num_experts,
            K,
            bsz
        );
    }

    cuda_check(cudaPeekAtLastError());
}


/*
Routing weights for externally-selected experts (e.g. DSv4 hash-MoE, where selection is a
frozen token-id -> experts table): scores = hidden @ gate, then weights = normalized
activated scores gathered at `selected`, times scaling_factor. Selection order is preserved.

hidden: Input hidden states, float16, shape (bsz, hidden_dim)
gate: Router gate matrix, float16, shape (hidden_dim, num_experts)
scores: Output routing logits buffer, float16, shape (bsz, num_experts)
selected: Expert indices, int64, shape (bsz, k)
weights: Output, float16, shape (bsz, k)
*/

// Dynamic expert placement (CPU expert split): one pass over the selected-expert ids that
// records hit counts, translates router ids to physical slots inplace (the bsz-1 routing
// statics are read by captured graphs at baked addresses), and emits the worker-side
// selection with GPU-resident picks replaced by the -1 sentinel. Replaces a 4-5 launch
// torch chain per MoE layer per step, which measurably outweighed the placement benefit on
// 40-layer models

__global__ void moe_split_map_kernel
(
    int64_t* __restrict__ sel,           // (n,) router ids in, physical slots out
    const int64_t* __restrict__ map,     // (E,) router id -> physical slot
    float* __restrict__ hist,            // (E,) hit counts (router space)
    int64_t* __restrict__ sel_cpu,       // (n,) out: worker slot if CPU-resident else -1
    const int n,
    const int first_cpu_slot
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t r = sel[i];
    atomicAdd(&hist[r], 1.0f);
    int64_t p = map[r];
    sel_cpu[i] = p >= first_cpu_slot ? p - first_cpu_slot : -1;
    sel[i] = p;
}

void moe_split_map
(
    at::Tensor sel,
    const at::Tensor& map,
    at::Tensor hist,
    at::Tensor sel_cpu,
    const int64_t first_cpu_slot
)
{
    const at::cuda::OptionalCUDAGuard device_guard(sel.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    routing_det_sync(stream);
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(map, kLong);
    TORCH_CHECK_DTYPE(hist, kFloat);
    TORCH_CHECK_DTYPE(sel_cpu, kLong);
    int n = (int) sel.numel();
    TORCH_CHECK(sel_cpu.numel() >= n, "moe_split_map: sel_cpu too small");
    int threads = n < 1024 ? n : 1024;
    int blocks = (n + threads - 1) / threads;
    moe_split_map_kernel<<<blocks, threads, 0, stream>>>
    (
        (int64_t*) sel.data_ptr(),
        (const int64_t*) map.data_ptr(),
        (float*) hist.data_ptr(),
        (int64_t*) sel_cpu.data_ptr(),
        n, (int) first_cpu_slot
    );
    cuda_check(cudaPeekAtLastError());
}

// Fused issue for the CPU expert split (decode-size jobs): one launch replaces the map/
// translate kernel, the int32 cast, the zero-pad and the three D2H memcpys of the two-phase
// submit path. The kernel writes the staged inputs straight into the device-mapped pinned
// slot with zero-copy stores, and skips the activation/weight payload entirely (no PCIe
// traffic) when no selected expert is CPU-resident (common case with a hot/cold-swapped
// tail). dev_count[slot] tells the collect kernel whether the worker produced anything. With
// `map`, also translates sel to physical slots in place and bumps the hit histogram (the
// dynamic-placement duties of moe_split_map)
__global__ void moe_split_issue_kernel
(
    int64_t* __restrict__ sel,           // (rows * topk,) router ids; physical slots out (map mode)
    const int64_t* __restrict__ map,     // (E,) or null (static split: ids are physical already)
    float* __restrict__ hist,            // (E,) or null
    const half* __restrict__ y,          // (rows, h_) device, contiguous
    const half* __restrict__ w,          // (rows, topk) device, contiguous
    int32_t* __restrict__ h_sel,         // slot sel, mapped host (rows, topk)
    half* __restrict__ h_x,              // slot x, mapped host (rows, hi)
    half* __restrict__ h_w,              // slot w, mapped host (rows, topk)
    int32_t* __restrict__ dev_count,     // (num_slots,) device
    const int n,                         // rows * topk
    const int rows,
    const int h_,
    const int hi,
    const int first_cpu,
    const int slot_idx
)
{
    __shared__ int s_any;
    const int t = threadIdx.x;
    if (t == 0) s_any = 0;
    __syncthreads();

    int any = 0;
    for (int i = t; i < n; i += blockDim.x)
    {
        int64_t p = sel[i];
        if (map)
        {
            atomicAdd(&hist[p], 1.0f);
            p = map[p];
            sel[i] = p;
        }
        int32_t lc = p >= first_cpu ? (int32_t)(p - first_cpu) : -1;
        h_sel[i] = lc;
        any |= (lc >= 0);
    }
    if (any) atomicOr(&s_any, 1);
    __syncthreads();
    if (t == 0) dev_count[slot_idx] = s_any;
    if (!s_any)
    {
        __threadfence_system();
        return;
    }

    for (int i = t; i < n; i += blockDim.x)
        h_w[i] = w[i];

    // Activations, zero-padded to the quantized input width. Vectorized when both widths
    // are 8-half aligned (they are for every real model dim); PCIe write-combining wants
    // the widest coalesced stores it can get
    if ((h_ & 7) == 0 && (hi & 7) == 0)
    {
        const int h8 = h_ >> 3, hi8 = hi >> 3;
        const int4* y4 = reinterpret_cast<const int4*>(y);
        int4* x4 = reinterpret_cast<int4*>(h_x);
        const int4 z = {0, 0, 0, 0};
        const int total = rows * hi8;
        for (int i = t; i < total; i += blockDim.x)
        {
            const int row = i / hi8;
            const int c = i - row * hi8;
            x4[i] = c < h8 ? y4[row * h8 + c] : z;
        }
    }
    else
    {
        const int total = rows * hi;
        for (int i = t; i < total; i += blockDim.x)
        {
            const int row = i / hi;
            const int c = i - row * hi;
            h_x[i] = c < h_ ? y[row * h_ + c] : __float2half_rn(0.0f);
        }
    }
    __threadfence_system();
}

void moe_split_issue
(
    at::Tensor sel,
    const c10::optional<at::Tensor>& map,
    const c10::optional<at::Tensor>& hist,
    const at::Tensor& y,
    const at::Tensor& w,
    int64_t h_sel_ptr,
    int64_t h_x_ptr,
    int64_t h_w_ptr,
    at::Tensor dev_count,
    const int64_t slot_idx,
    const int64_t hi,
    const int64_t first_cpu
)
{
    const at::cuda::OptionalCUDAGuard device_guard(sel.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    routing_det_sync(stream);
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(dev_count, kInt);
    TORCH_CHECK(y.is_contiguous() && w.is_contiguous() && sel.is_contiguous(),
                "moe_split_issue: inputs must be contiguous");
    TORCH_CHECK(map.has_value() == hist.has_value(), "moe_split_issue: map requires hist");
    int n = (int) sel.numel();
    int rows = (int) y.size(0);
    int h_ = (int) y.size(1);
    moe_split_issue_kernel<<<1, 1024, 0, stream>>>
    (
        (int64_t*) sel.data_ptr(),
        map ? (const int64_t*) map.value().data_ptr() : nullptr,
        hist ? (float*) hist.value().data_ptr() : nullptr,
        (const half*) y.data_ptr(),
        (const half*) w.data_ptr(),
        reinterpret_cast<int32_t*>(h_sel_ptr),
        reinterpret_cast<half*>(h_x_ptr),
        reinterpret_cast<half*>(h_w_ptr),
        (int32_t*) dev_count.data_ptr(),
        n, rows, h_, (int) hi, (int) first_cpu, (int) slot_idx
    );
    cuda_check(cudaPeekAtLastError());
}

// Fused collect for the split path: fold the worker's partial into the routed sum straight
// from the pinned slot (replacing the H2D memcpy, the width trim and the separate add), or
// do nothing at all -- no PCIe reads -- when the issue kernel recorded an empty job. Loads
// go through volatile so no stale line can be served for a slot's previous tenant
__global__ void moe_split_collect_add_kernel
(
    float* __restrict__ final_out,       // (rows, h_) device, contiguous
    const float* __restrict__ h_out,     // slot out, mapped host (rows, ho)
    const int32_t* __restrict__ dev_count,
    const int total,                     // rows * h_
    const int h_,
    const int ho,
    const int slot_idx
)
{
    if (dev_count[slot_idx] == 0) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int row = i / h_;
    const int c = i - row * h_;
    const volatile float* vo = h_out;
    final_out[i] += vo[row * ho + c];
}

void moe_split_collect_add
(
    at::Tensor final_out,
    int64_t h_out_ptr,
    const at::Tensor& dev_count,
    const int64_t slot_idx,
    const int64_t ho
)
{
    const at::cuda::OptionalCUDAGuard device_guard(final_out.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    routing_det_sync(stream);
    TORCH_CHECK_DTYPE(final_out, kFloat);
    TORCH_CHECK_DTYPE(dev_count, kInt);
    TORCH_CHECK(final_out.is_contiguous(), "moe_split_collect_add: output must be contiguous");
    int rows = (int) final_out.size(0);
    int h_ = (int) final_out.size(1);
    int total = rows * h_;
    int threads = total < 1024 ? total : 1024;
    int blocks = (total + threads - 1) / threads;
    moe_split_collect_add_kernel<<<blocks, threads, 0, stream>>>
    (
        (float*) final_out.data_ptr(),
        reinterpret_cast<const float*>(h_out_ptr),
        (const int32_t*) dev_count.data_ptr(),
        total, h_, (int) ho, (int) slot_idx
    );
    cuda_check(cudaPeekAtLastError());
}

__global__ void routing_sel_norm_kernel
(
    const half* __restrict__ scores,
    const int64_t* __restrict__ selected,
    half* __restrict__ weights,
    const float scaling_factor,
    const int num_experts,
    const int K,
    const int act_fn
)
{
    int row = blockIdx.x;
    int lane = threadIdx.x;

    float o = 0.0f;
    if (lane < K)
    {
        int e = (int) selected[row * K + lane];
        float logit = __half2float(scores[row * num_experts + e]);
        o = act_fn == ROUTING_ACT_SQRTSP ?
            routing_act<ROUTING_ACT_SQRTSP>(logit) :
            routing_act<ROUTING_ACT_SIGMOID>(logit);
    }

    float sum = warp_reduce_sum_first_k(o, K) + 1e-20f;
    if (lane < K)
        weights[row * K + lane] = __float2half_rn(o * scaling_factor / sum);
}

void routing_sel_norm
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    at::Tensor scores,
    const at::Tensor& selected,
    at::Tensor weights,
    const float scaling_factor,
    const c10::optional<at::Tensor>& gate_t,
    const int act_fn,
    const c10::optional<at::Tensor>& gate_i8,
    const c10::optional<at::Tensor>& gate_sb
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    routing_det_sync(stream);

    routing_gemv(hidden, gate, gate_t, gate_i8, gate_sb, scores, stream);

    TORCH_CHECK_DTYPE(hidden, kHalf);
    TORCH_CHECK_DTYPE(gate, kHalf);
    TORCH_CHECK_DTYPE(scores, kHalf);
    TORCH_CHECK_DTYPE(selected, kLong);
    TORCH_CHECK_DTYPE(weights, kHalf);
    TORCH_CHECK_SHAPES(hidden, -1, gate, 0, 1);
    TORCH_CHECK_SHAPES(gate, 1, scores, -1, 1);
    TORCH_CHECK_SHAPES(scores, 0, selected, 0, 1);
    TORCH_CHECK_SHAPES(scores, 0, weights, 0, 1);
    TORCH_CHECK_SHAPES(selected, 1, weights, 1, 1);

    int bsz = scores.size(0);
    int num_experts = scores.size(1);
    int K = selected.size(1);
    TORCH_CHECK(K <= 32, "routing_sel_norm: K > 32");

    routing_sel_norm_kernel<<<bsz, 32, 0, stream>>>
    (
        (const half*) scores.data_ptr(),
        (const int64_t*) selected.data_ptr(),
        (half*) weights.data_ptr(),
        scaling_factor,
        num_experts,
        K,
        act_fn
    );
    cuda_check(cudaPeekAtLastError());
}

/*
Standard softmax routing

hidden: Input hidden states, float16, shape (..., hidden_dim)
gate: Router gate matrix, float16, shape (hidden_dim, num_experts)
scores: Output routing logits buffer, float16, shape (bsz, num_experts)
topk_indices: int64, shape (bsz, k)
topk_weights: float16, shape (bsz, k)
*/

void routing_std
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    at::Tensor scores,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const c10::optional<at::Tensor>& per_expert_scale,
    const c10::optional<at::Tensor>& gate_t,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& gate_i8,
    const c10::optional<at::Tensor>& gate_sb
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    routing_det_sync(stream);

    routing_gemv(hidden, gate, gate_t, gate_i8, gate_sb, scores, stream);

    TORCH_CHECK_DTYPE(hidden, kHalf);
    TORCH_CHECK_DTYPE(gate, kHalf);
    TORCH_CHECK_SHAPES(scores, 0, topk_indices, 0, 1);
    TORCH_CHECK_SHAPES(scores, 0, topk_weights, 0, 1);
    TORCH_CHECK_SHAPES(hidden, -1, gate, 0, 1);
    TORCH_CHECK_SHAPES(gate, 1, scores, -1, 1);
    TORCH_CHECK_SHAPES(topk_indices, 1, topk_weights, 1, 1);
    TORCH_CHECK_SHAPES_OPT(per_expert_scale, 0, scores, 1, 1);
    TORCH_CHECK_DTYPE(scores, kHalf);
    TORCH_CHECK_DTYPE(topk_indices, kLong);
    TORCH_CHECK_DTYPE(topk_weights, kHalf);
    TORCH_CHECK_DTYPE_OPT(per_expert_scale, kBFloat16);

    int bsz = scores.size(0);
    int num_experts = scores.size(1);
    int K = topk_indices.size(1);

    TORCH_CHECK(num_experts <= MAX_NUM_EXPERTS, "Too many experts");
    TORCH_CHECK(K <= MAX_K, "Too many experts per token");
    TORCH_CHECK(K <= num_experts, "K cannot exceed number of experts");

    int num_warps = CEIL_DIVIDE(num_experts, 32);
    int num_threads = num_warps * 32;
    size_t shmem = (num_experts + num_warps) * sizeof(float);

    TORCH_CHECK_DTYPE_OPT(bias, kHalf);
    routing_std_topk_kernel<<<bsz, num_threads, shmem, stream>>>
    (
        (const half*) scores.data_ptr(),
        (int64_t*) topk_indices.data_ptr(),
        (half*) topk_weights.data_ptr(),
        (const bfloat16*) OPTPTR(per_expert_scale),
        (const half*) OPTPTR(bias),
        num_experts,
        K,
        bsz
    );
    cuda_check(cudaPeekAtLastError());
}


void routing_std_logits
(
    at::Tensor scores,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const c10::optional<at::Tensor>& per_expert_scale,
    const bool use_topk
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    routing_det_sync(stream);

    TORCH_CHECK_SHAPES(scores, 0, topk_indices, 0, 1);
    TORCH_CHECK_SHAPES(scores, 0, topk_weights, 0, 1);
    TORCH_CHECK_SHAPES(topk_indices, 1, topk_weights, 1, 1);
    TORCH_CHECK_SHAPES_OPT(per_expert_scale, 0, scores, 1, 1);
    TORCH_CHECK_DTYPE(scores, kHalf);
    TORCH_CHECK_DTYPE(topk_indices, kLong);
    TORCH_CHECK_DTYPE(topk_weights, kHalf);
    TORCH_CHECK_DTYPE_OPT(per_expert_scale, kBFloat16);

    int bsz = scores.size(0);
    int num_experts = scores.size(1);
    int K = topk_indices.size(1);

    TORCH_CHECK(num_experts <= MAX_NUM_EXPERTS, "Too many experts");
    TORCH_CHECK(K <= MAX_K, "Too many experts per token");
    TORCH_CHECK(K <= num_experts, "K cannot exceed number of experts");

    int num_warps = CEIL_DIVIDE(num_experts, 32);
    int num_threads = num_warps * 32;

    TORCH_CHECK(use_topk || K <= 16, "routing_std_logits: radix-sort kernel supports K <= 16");
    if (use_topk)
    {
        size_t shmem = (num_experts + num_warps) * sizeof(float);
        routing_std_topk_kernel<<<bsz, num_threads, shmem, stream>>>
        (
            (const half*) scores.data_ptr(),
            (int64_t*) topk_indices.data_ptr(),
            (half*) topk_weights.data_ptr(),
            (const bfloat16*) OPTPTR(per_expert_scale),
            nullptr,
            num_experts,
            K,
            bsz
        );
    }
    else
    {
        int K_ = K + (K & 1);
        size_t shmem = num_warps * K_ * (sizeof(float) + sizeof(int))
                     + num_threads * sizeof(int)
                     + num_warps * sizeof(float);
        routing_std_kernel<<<bsz, num_threads, shmem, stream>>>
        (
            (const half*) scores.data_ptr(),
            (int64_t*) topk_indices.data_ptr(),
            (half*) topk_weights.data_ptr(),
            (const bfloat16*) OPTPTR(per_expert_scale),
            num_experts,
            K,
            bsz
        );
    }

    cuda_check(cudaPeekAtLastError());
}
