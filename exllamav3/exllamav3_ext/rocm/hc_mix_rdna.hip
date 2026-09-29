// =============================================================================
// hc_mix.cu for RDNA -- cross-lane reductions on DPP instead of ds_bpermute
// =============================================================================
//
// Generated from upstream v1.5.0 hc_mix.cu (which stays byte-identical to
// upstream; setup.py excludes it from the ROCm build in favor of this file).
// The diff is confined to the helpers marked "ROCm sibling" and their call
// sites in hc_mix_partials_kernel (the M + 1 accumulator warp sums) and
// hc_mix_finalize_kernel (the sinkhorn butterflies); everything else is a
// verbatim copy, same exported symbols.
//
// WHY (DS4 decode, rocprofv3, c6c19ca + decode-leftovers, 86 sites per token):
// __shfl_down / __shfl_xor lower to ds_bpermute_b32 (an LDS round trip, tens of
// cycles each). The partials kernel reduces 25 accumulators with 5 shuffles
// each (100 ds_bpermute per wave at the end of an otherwise one-iteration
// kernel: 8.4 us), and the finalize kernel's 20-iteration sinkhorn is a serial
// chain of 4 shuffles + 2 divides per iteration on one warp (its critical
// path: 5.3 us). Every one of those shuffles moves a lane value within a row
// of 16 lanes (or row 1 -> row 0 for the offset-16 step), which DPP does in
// the VALU: row_shl:n for __shfl_down by n < 16, v_permlanex16 for 16,
// quad_perm for xor 1 / 2, row_xmask for xor 4 / 8. The value each lane
// consumes, and the order of every add, is unchanged -- only lanes whose
// results upstream never reads (a shfl_down source past the row) see a
// different value -- so post / comb / collapsed are bit-identical.
// EXL3_ROCM_HC_DPP=0 restores the shuffles (same build, runtime switch).
//
// v1.5.3 sync (three-way merge, base v1.5.0): upstream's new GatedResidual
// decode pair (gr_dots_kernel2 / gr_finalize_kernel2) merged in verbatim and
// reachable with EXL3_ROCM_GR_DOTS=0 (see gr_mix); the default stays on the
// RDNA rows dots, bit-identical to v1.5.0. hc_mix_tiled.cu (upstream's tiled
// int8 prefill mix) is a separate, declining sibling: rocm/hc_mix_tiled_rdna.hip.
// =============================================================================

#include <cstdlib>
#include <torch/library.h>
#include <cuda_fp16.h>
#include "hc_mix.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "graph.cuh"

/*

Fused mHC HyperConnection mix() kernel

mix(streams (R, H, D) fp32) -> post (R, H), comb (R, H, H), collapsed (R, D):

  flat = rmsnorm_unweighted(streams.flatten)          (row of H * D values)
  mixv = flat @ fn.T                                  (M = 2H + H^2 outputs)
  pre  = sigmoid(mixv[0:H]  * s0 + base[0:H]) + eps
  post = 2 sigmoid(mixv[H:2H] * s1 + base[H:2H])
  comb = sinkhorn(softmax(mixv[2H:] * s2 + base[2H:]))   (iters alternating row/col norms)
  collapsed = sum_h pre[h] * streams[h]

Two launches, no grid-wide sync, deterministic (fixed reduction order, no atomics):
  K1 partials: grid (chunksA, R); each block reduces its column chunk to M + 1 partials.
  K2 finalize: grid (chunksC, R); EVERY block re-reduces the tiny partial matrix and
    derives rmr + pre redundantly (removes the cross-block dependency), then streams its
    chunk of collapsed; the chunk-0 block also runs the sinkhorn on H^2 lanes of warp 0
    (row sums: shfl_xor 1|2, col sums: shfl_xor 4|8 for H = 4) and writes post/comb.

*/

// Partials blocks are small: at R = 1 the grid is the only parallelism, so favor many
// blocks (row_len / (4 * 64) chunks) over wide ones; the M + 1 block reduce also shrinks
#define NUM_THREADS 256
#define NUM_THREADS_A 64

__device__ __forceinline__ float sigmoidf_(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

// -----------------------------------------------------------------------------
// ROCm sibling: DPP cross-lane moves (see the file header)
// -----------------------------------------------------------------------------
template <int CTRL>
__device__ __forceinline__ float hc_dpp(float v)
{
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xf, 0xf, false));
}

// Lane i of row 0 receives lane i + 16 (identity select across the two rows)
__device__ __forceinline__ float hc_permlanex16(float v)
{
    return __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
}

// xor 16: row 0 lane i <-> row 1 lane i (v_permlanex16 reads the other row in both rows)
__device__ __forceinline__ float hc_permlanex16_bfly(float v)
{
    return __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
}

#define HC_DPP_ROW_SHL(n) (0x100 + (n))
#define HC_DPP_QUAD_XOR1  0xB1        // quad_perm [1, 0, 3, 2]
#define HC_DPP_QUAD_XOR2  0x4E        // quad_perm [2, 3, 0, 1]
#define HC_DPP_ROW_XMASK(n) (0x160 + (n))

// __shfl_down tree (offsets 16, 8, 4, 2, 1) as upstream writes it; lane 0's
// result is the same sum in the same order
__device__ __forceinline__ float hc_warp_sum_lane0(float v)
{
    v += hc_permlanex16(v);
    v += hc_dpp<HC_DPP_ROW_SHL(8)>(v);
    v += hc_dpp<HC_DPP_ROW_SHL(4)>(v);
    v += hc_dpp<HC_DPP_ROW_SHL(2)>(v);
    v += hc_dpp<HC_DPP_ROW_SHL(1)>(v);
    return v;
}

// __shfl_xor(v, o) for o in {1, 2, 4, 8}, within a row of 16
template <int O>
__device__ __forceinline__ float hc_xor16(float v)
{
    if constexpr (O == 1) return hc_dpp<HC_DPP_QUAD_XOR1>(v);
    else if constexpr (O == 2) return hc_dpp<HC_DPP_QUAD_XOR2>(v);
    else return hc_dpp<HC_DPP_ROW_XMASK(O)>(v);
}

static bool hc_dpp_enabled()
{
    static const bool on = []
    {
        const char* env = std::getenv("EXL3_ROCM_HC_DPP");
        return !env || atoi(env) != 0;
    }();
    return on;
}

template <int H, int M_, typename FN_T, bool DPP = false>
__global__ __launch_bounds__(NUM_THREADS_A)
void hc_mix_partials_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const FN_T* __restrict__ fn,         // (M, H * D) float, or half (opt-in, halves traffic)
    float* __restrict__ partials,        // (R, chunksA, M + 1)
    const int row_len,
    const int chunk_cols                 // multiple of 4 * NUM_THREADS_A
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, row_len);

    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int row_len4 = row_len / 4;

    float acc[M + 1];
    #pragma unroll
    for (int k = 0; k <= M; ++k) acc[k] = 0.0f;

    for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS_A)
    {
        float4 s = s4[c];
        acc[M] = fmaf(s.x, s.x, acc[M]);
        acc[M] = fmaf(s.y, s.y, acc[M]);
        acc[M] = fmaf(s.z, s.z, acc[M]);
        acc[M] = fmaf(s.w, s.w, acc[M]);
        #pragma unroll
        for (int j = 0; j < M; ++j)
        {
            float4 w;
            if constexpr (std::is_same_v<FN_T, half>)
            {
                // Single vectorized 8-byte load (two half2 loads would double the LDGs)
                int2 pk = ((const int2*) fn)[(size_t) j * row_len4 + c];
                float2 lo = __half22float2(*(const half2*) &pk.x);
                float2 hi = __half22float2(*(const half2*) &pk.y);
                w = make_float4(lo.x, lo.y, hi.x, hi.y);
            }
            else
                w = ((const float4*) fn)[(size_t) j * row_len4 + c];
            float d = fmaf(s.x, w.x, fmaf(s.y, w.y, fmaf(s.z, w.z, s.w * w.w)));
            acc[j] += d;
        }
    }

    // Block reduce M + 1 lanes' accumulators
    __shared__ float red[NUM_THREADS_A / 32][M + 1];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    #pragma unroll
    for (int k = 0; k <= M; ++k)
    {
        float v = acc[k];
        if constexpr (DPP)
            v = hc_warp_sum_lane0(v);   // ROCm sibling: same tree on DPP
        else
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, offset);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();

    if (warp == 0)
    {
        float* out = partials + ((size_t) r * gridDim.x + blockIdx.x) * (M + 1);
        for (int k = lane; k <= M; k += 32)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_THREADS_A / 32; ++w)
                v += red[w][k];
            out[k] = v;
        }
    }
}

// ROCm sibling: arguments of the rms_norm folded into the finalize (NORM = true)
struct HcNormArgs
{
    const void* w;          // (D) bf16 or half weight, or null (unweighted)
    bool w_bf16;
    half* y;                // (R, D) normed output
    float eps;
    float constant_bias;
    float constant_scale;
};

template <int H, int M_, bool HEAD, bool HALF_OUT, bool DPP = false, bool NORM = false>
__global__ __launch_bounds__(NUM_THREADS)
void hc_mix_finalize_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const float* __restrict__ partials,  // (R, chunksA, M + 1)
    const float* __restrict__ base,      // (M)
    const float* __restrict__ scale,     // (3)
    float* __restrict__ post,            // (R, H)
    float* __restrict__ comb,            // (R, H, H)
    void* __restrict__ collapsed,        // (R, D) float or half
    const int D,
    const int chunksA,
    const int chunk_cols_c,              // multiple of 4
    const float rms_eps,
    const float hc_eps,
    const int sinkhorn_iters,
    const HcNormArgs norm = {}
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int row_len = H * D;

    // Re-reduce this row's partials (tiny, L2-resident). Every block does this to avoid
    // cross-block dependencies. Warp-split over the chunk axis: the serial loop sits
    // at the head of the kernel's critical path, so with many partials chunks (small
    // NUM_THREADS_A blocks) a single-thread-per-quantity loop is too long
    __shared__ float mix_s[M + 1];
    __shared__ float pre_s[H];
    __shared__ float red_s[NUM_THREADS / 32][M + 1];
    {
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        if (lane <= M)
        {
            const float* p = partials + (size_t) r * chunksA * (M + 1) + lane;
            float v = 0.0f;
            for (int i = warp; i < chunksA; i += NUM_THREADS / 32)
                v += p[(size_t) i * (M + 1)];
            red_s[warp][lane] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x <= M)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_THREADS / 32; ++w)
            v += red_s[w][threadIdx.x];
        mix_s[threadIdx.x] = v;
    }
    __syncthreads();
    float rmr = rsqrtf(mix_s[M] / (float) row_len + rms_eps);
    if (threadIdx.x < H)
        pre_s[threadIdx.x] = sigmoidf_(fmaf(mix_s[threadIdx.x] * rmr, scale[0], base[threadIdx.x])) + hc_eps;
    __syncthreads();

    // Sinkhorn + post/comb writes: chunk-0 block only, one lane per comb element. Runs in
    // a dedicated warp, concurrent with the other warps' phase C: the ~20-iteration
    // normalization is a serial shfl/div latency chain that only needs the M + 1 reduced
    // scalars, so at small R it sets the kernel's critical path; don't stack phase C
    // work in front of it.
    //
    // Lane l = i * H + j; row sums reduce over j (xor 1..H/2), col sums over i (xor H..).
    const bool sink_warp = !HEAD && blockIdx.x == 0 && threadIdx.x < 32;
    if (sink_warp)
    {
        if (threadIdx.x < H)
            post[(size_t) r * H + threadIdx.x] =
                2.0f * sigmoidf_(fmaf(mix_s[H + threadIdx.x] * rmr, scale[1], base[H + threadIdx.x]));

        if (threadIdx.x < H * H)
        {
            const unsigned mask = (H * H == 32) ? 0xffffffffu : ((1u << (H * H)) - 1u);
            float v = fmaf(mix_s[2 * H + threadIdx.x] * rmr, scale[2], base[2 * H + threadIdx.x]);

            if constexpr (DPP && H == 4)
            {
                // ROCm sibling: the same butterflies on DPP (H * H = 16 lanes, one row)
                float m = v;
                m = fmaxf(m, hc_xor16<1>(m));
                m = fmaxf(m, hc_xor16<2>(m));
                v = __expf(v - m);
                float s = v;
                s += hc_xor16<1>(s);
                s += hc_xor16<2>(s);
                v = __fdividef(v, s) + hc_eps;

                float cs = v;
                cs += hc_xor16<4>(cs);
                cs += hc_xor16<8>(cs);
                v = __fdividef(v, cs + hc_eps);
                for (int it = 0; it < sinkhorn_iters - 1; ++it)
                {
                    float rs = v;
                    rs += hc_xor16<1>(rs);
                    rs += hc_xor16<2>(rs);
                    v = __fdividef(v, rs + hc_eps);
                    cs = v;
                    cs += hc_xor16<4>(cs);
                    cs += hc_xor16<8>(cs);
                    v = __fdividef(v, cs + hc_eps);
                }
                comb[(size_t) r * H * H + threadIdx.x] = v;
                (void) mask;
            }
            else
            {
            // softmax over rows
            float m = v;
            #pragma unroll
            for (int o = 1; o < H; o <<= 1) m = fmaxf(m, __shfl_xor_sync(mask, m, o));
            v = __expf(v - m);
            float s = v;
            #pragma unroll
            for (int o = 1; o < H; o <<= 1) s += __shfl_xor_sync(mask, s, o);
            v = __fdividef(v, s) + hc_eps;

            // column normalize, then (iters - 1) x (row, column)
            float cs = v;
            #pragma unroll
            for (int o = H; o < H * H; o <<= 1) cs += __shfl_xor_sync(mask, cs, o);
            v = __fdividef(v, cs + hc_eps);
            for (int it = 0; it < sinkhorn_iters - 1; ++it)
            {
                float rs = v;
                #pragma unroll
                for (int o = 1; o < H; o <<= 1) rs += __shfl_xor_sync(mask, rs, o);
                v = __fdividef(v, rs + hc_eps);
                cs = v;
                #pragma unroll
                for (int o = H; o < H * H; o <<= 1) cs += __shfl_xor_sync(mask, cs, o);
                v = __fdividef(v, cs + hc_eps);
            }
            comb[(size_t) r * H * H + threadIdx.x] = v;
            }
        }
        if constexpr (!NORM) return;
    }

    // Phase C: collapsed chunk, weighted sum over the H stream rows. In the sinkhorn
    // block the first warp is excluded, so the remaining threads re-cover its lanes
    float pre_r[H];
    #pragma unroll
    for (int h = 0; h < H; ++h) pre_r[h] = pre_s[h];

    const bool shrunk = !HEAD && blockIdx.x == 0;
    const int tid = shrunk ? threadIdx.x - 32 : threadIdx.x;
    const int nth = shrunk ? NUM_THREADS - 32 : NUM_THREADS;
    const int c0 = blockIdx.x * chunk_cols_c;
    const int c1 = min(c0 + chunk_cols_c, D);
    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int D4 = D / 4;
    // NORM: the whole row in this block (grid.x == 1), kept in LDS for the norm below
    __shared__ half2 sh_norm_x[NORM ? 2048 : 1];
    if constexpr (NORM && HALF_OUT)
    {
        // One block streams the whole row: issue every stream load of this thread first
        // (<= 5 columns: D / 4 <= 1024 over >= 224 threads), then the same arithmetic
        constexpr int NIT = 5;
        float4 sv[NIT][H];
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
        {
            const int c = c0 / 4 + tid + it * nth;
            if (!sink_warp && c < c1 / 4)
            {
                #pragma unroll
                for (int h = 0; h < H; ++h) sv[it][h] = s4[(size_t) h * D4 + c];
            }
        }
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
        {
            const int c = c0 / 4 + tid + it * nth;
            if (sink_warp || c >= c1 / 4) continue;
            float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                o.x = fmaf(pre_r[h], sv[it][h].x, o.x);
                o.y = fmaf(pre_r[h], sv[it][h].y, o.y);
                o.z = fmaf(pre_r[h], sv[it][h].z, o.z);
                o.w = fmaf(pre_r[h], sv[it][h].w, o.w);
            }
            half2* out2 = (half2*) ((half*) collapsed + (size_t) r * D);
            half2 o01 = __floats2half2_rn(o.x, o.y);
            half2 o23 = __floats2half2_rn(o.z, o.w);
            out2[c * 2] = o01;
            out2[c * 2 + 1] = o23;
            sh_norm_x[c * 2] = o01;
            sh_norm_x[c * 2 + 1] = o23;
        }
    }
    else
    for (int c = c0 / 4 + tid; c < c1 / 4; c += nth)
    {
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            o.x = fmaf(pre_r[h], s.x, o.x);
            o.y = fmaf(pre_r[h], s.y, o.y);
            o.z = fmaf(pre_r[h], s.z, o.z);
            o.w = fmaf(pre_r[h], s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) collapsed + (size_t) r * D);
            half2 o01 = __floats2half2_rn(o.x, o.y);
            half2 o23 = __floats2half2_rn(o.z, o.w);
            out2[c * 2] = o01;
            out2[c * 2 + 1] = o23;
            if constexpr (NORM)
            {
                sh_norm_x[c * 2] = o01;
                sh_norm_x[c * 2 + 1] = o23;
            }
        }
        else
            ((float4*) ((float*) collapsed + (size_t) r * D))[c] = o;
    }

    if constexpr (NORM && HALF_OUT)
    {
        // ROCm sibling: rms_norm_kernel<RES_NONE, half, half, W, half> on this row. In NORM
        // mode the host launches one finalize block per row, so the collapsed row (the
        // same half values stored to `collapsed`) is all in this block's LDS. Replays the
        // norm kernel's single-pass form exactly: virtual thread t < T = min(1024,
        // ceil(D / 4 / 32) * 32) owns float4 column t, sum_sq4 from 0, reduce_dyn's two
        // xor-butterfly stages (warp, then the T / 32 warp sums), rmf, then (x * w) * rmf
        // per element and the same half rounding -- so the output equals the separate
        // rms_norm launch bit for bit. The host takes this path only for D / 4 <= 1024.
        __shared__ float sh_sums[32];
        __syncthreads();

        const int columns = D / 4;
        const int T = min(1024, ((columns + 31) / 32) * 32);
        const int n_vw = T / 32;
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        const half4* xin = (const half4*) sh_norm_x;

        auto read_x = [&] (int col) -> float4
        {
            half4 h4 = xin[col];
            float4 f4;
            f4.x = CLAMP_FP16(LOW_TO_FLOAT(h4.x));
            f4.y = CLAMP_FP16(HIGH_TO_FLOAT(h4.x));
            f4.z = CLAMP_FP16(LOW_TO_FLOAT(h4.y));
            f4.w = CLAMP_FP16(HIGH_TO_FLOAT(h4.y));
            return f4;
        };
        auto bfly = [&] (float v) -> float
        {
            if constexpr (DPP)
            {
                v += hc_permlanex16_bfly(v);
                v += hc_xor16<8>(v);
                v += hc_xor16<4>(v);
                v += hc_xor16<2>(v);
                v += hc_xor16<1>(v);
            }
            else
            {
                for (int offset = 16; offset > 0; offset /= 2) v += __shfl_xor_sync(0xffffffff, v, offset);
            }
            return v;
        };

        for (int vw = warp; vw < n_vw; vw += NUM_THREADS / 32)
        {
            const int col = vw * 32 + lane;
            float sum = 0.0f;
            if (col < columns)
            {
                float4 x4 = read_x(col);
                sum = fma(x4.x, x4.x, sum);
                sum = fma(x4.y, x4.y, sum);
                sum = fma(x4.z, x4.z, sum);
                sum = fma(x4.w, x4.w, sum);
            }
            sum = bfly(sum);
            if (lane == 0) sh_sums[vw] = sum;
        }
        __syncthreads();
        float total;
        if (n_vw == 1)
            total = sh_sums[0];
        else
            total = bfly(lane < n_vw ? sh_sums[lane] : 0.0f);
        const float rmf = rsqrtf(total / (float) D + norm.eps) * norm.constant_scale;

        half4* yout = (half4*) (norm.y + (size_t) r * D);
        for (int col = threadIdx.x; col < columns; col += NUM_THREADS)
        {
            float4 x4 = read_x(col);
            if (norm.w)
            {
                float4 w4;
                if (norm.w_bf16)
                {
                    bfloat164 b4;
                    READ64(b4, ((const bfloat164*) norm.w) + col);
                    w4.x = __bfloat162float(__low2bfloat16(b4.x));
                    w4.y = __bfloat162float(__high2bfloat16(b4.x));
                    w4.z = __bfloat162float(__low2bfloat16(b4.y));
                    w4.w = __bfloat162float(__high2bfloat16(b4.y));
                }
                else
                {
                    half4 h4;
                    READ64(h4, ((const half4*) norm.w) + col);
                    w4.x = LOW_TO_FLOAT(h4.x);
                    w4.y = HIGH_TO_FLOAT(h4.x);
                    w4.z = LOW_TO_FLOAT(h4.y);
                    w4.w = HIGH_TO_FLOAT(h4.y);
                }
                if (norm.constant_bias != 0.0f)
                {
                    w4.x += norm.constant_bias;
                    w4.y += norm.constant_bias;
                    w4.z += norm.constant_bias;
                    w4.w += norm.constant_bias;
                }
                x4.x = x4.x * w4.x * rmf;
                x4.y = x4.y * w4.y * rmf;
                x4.z = x4.z * w4.z * rmf;
                x4.w = x4.w * w4.w * rmf;
            }
            else
            {
                x4.x = x4.x * rmf;
                x4.y = x4.y * rmf;
                x4.z = x4.z * rmf;
                x4.w = x4.w * rmf;
            }
            half4 h4
            (
                __halves2half2(__float2half_rn(x4.x), __float2half_rn(x4.y)),
                __halves2half2(__float2half_rn(x4.z), __float2half_rn(x4.w))
            );
            WRITE64(yout + col, h4);
        }
    }
}


/*

GatedResidual (Qwen4Exp, the low-rank elementwise cousin of mHC) fused mix, decode form:

  mix(streams (R, H, D) fp32) -> post (R, H), mixed (R, D):
    normed[h] = rmsnorm(streams[h]) * w[h]           (per-STREAM norm, weighted, w incl +1)
    dots      = cat(down, inject) @ normed.flatten   (M = LR + H outputs, LR = low rank ~320)
    t         = silu(dots[:LR] / H)
    post      = 2 sigmoid(dots[LR:] / H)
    g[h, d]   = sigmoid(up[h * D + d, :] @ t)        (per-CHANNEL gate through the low rank)
    mixed[d]  = mean_h g[h, d] * normed[h, d]

  Same two-launch, no-grid-sync, deterministic shape as hc_mix, restructured for the wide low
  rank (LR >> mHC's M = 24, so per-thread accumulator arrays don't fit):
    K1 gr_dots: one block per (fn row | sum-of-squares), computing per-STREAM partial dots
      against the RAW streams -- by linearity the per-stream rms scale applies in K2, and the
      norm WEIGHT is folded into the fn rows at load time.
    K2 gr_finalize: every block re-derives rmr / t redundantly from the K1 output (the mHC
      finalize pattern), then streams its chunk of mixed, evaluating the per-channel up-gate
      inline -- upT is laid out (LR, H * D) so the serial rank loop reads coalesced and stays
      L2-resident at decode R. NOT for large R (the untiled up/down reads defeat the L2);
      the python side runs a plain half-GEMM path for prefill.

  apply_ is hc_apply with no comb: x[h] += post[h] * y.

*/

#define GR_THREADS_A 128

template <int H>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const half* __restrict__ fn,         // (M, H * D) half, norm weight folded in
    float* __restrict__ dots,            // (R, M + 1, H): per-stream dots, row M = sum sq
    const int M,
    const int D
)
{
    const int r = blockIdx.y;
    const int j = blockIdx.x;            // fn row, or M for the sum-of-squares row
    const int D4 = D / 4;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);

    __shared__ float red[H][GR_THREADS_A / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float a = 0.0f;
        if (j < M)
        {
            // 16-byte fn loads (8 halves) against two 16-byte stream quads
            const int4* f8 = (const int4*) (fn + ((size_t) j * H + h) * D);
            for (int c = threadIdx.x; c < D4 / 2; c += GR_THREADS_A)
            {
                float4 s0 = s4[(size_t) h * D4 + 2 * c];
                float4 s1 = s4[(size_t) h * D4 + 2 * c + 1];
                int4 pk = f8[c];
                half4 w0 = *(half4*) &pk.x;
                half4 w1 = *(half4*) &pk.z;
                a = fmaf(s0.x, LOW_TO_FLOAT(w0.x), a);
                a = fmaf(s0.y, HIGH_TO_FLOAT(w0.x), a);
                a = fmaf(s0.z, LOW_TO_FLOAT(w0.y), a);
                a = fmaf(s0.w, HIGH_TO_FLOAT(w0.y), a);
                a = fmaf(s1.x, LOW_TO_FLOAT(w1.x), a);
                a = fmaf(s1.y, HIGH_TO_FLOAT(w1.x), a);
                a = fmaf(s1.z, LOW_TO_FLOAT(w1.y), a);
                a = fmaf(s1.w, HIGH_TO_FLOAT(w1.y), a);
            }
        }
        else
        {
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                float4 s = s4[(size_t) h * D4 + c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, offset);
        if (lane == 0) red[h][warp] = a;
    }
    __syncthreads();
    if (threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < GR_THREADS_A / 32; ++w)
            v += red[threadIdx.x][w];
        dots[((size_t) r * (M + 1) + j) * H + threadIdx.x] = v;
    }
}

template <int H, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void gr_finalize_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const float* __restrict__ dots,      // (R, M + 1, H) from gr_dots
    const half* __restrict__ upt,        // (H, D / 4, LR, 4) half: up repacked lane-contiguous
    const half* __restrict__ w,          // (H * D) half norm weight (incl +1)
    float* __restrict__ post,            // (R, H) or nullptr (final-mixer form)
    void* __restrict__ mixed,            // (R, D) half or float
    const int D,
    const int LR,                        // low rank; M = LR + (post ? H : 0)
    const int chunk_cols,                // multiple of 4
    const float rms_eps
)
{
    const int r = blockIdx.y;
    const int M = LR + (post ? H : 0);
    const float* dr = dots + (size_t) r * (M + 1) * H;

    // Redundant per-block head derivation (no cross-block dependency): rmr from the sumsq
    // row, then the silu'd low-rank activations into shared memory
    __shared__ float rmr_s[H];
    extern __shared__ float t_s[];
    if (threadIdx.x < H)
        rmr_s[threadIdx.x] = rsqrtf(dr[(size_t) M * H + threadIdx.x] / (float) D + rms_eps);
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int i = threadIdx.x; i < LR; i += NUM_THREADS)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) i * H + h], v);
        v *= inv_h;
        t_s[i] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) (LR + threadIdx.x) * H + h], v);
        post[(size_t) r * H + threadIdx.x] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    // Streamed mixed chunk: one WARP per column quad, the LR-long up-gate dots split across
    // the lanes (lane l covers ranks l, l+32, ...) and shfl-reduced -- at decode R the total
    // column count is small (R * D / 4), so per-thread columns would leave the GPU nearly
    // idle with each thread serializing the rank loop
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
    for (int c = c0 / 4 + warp; c < c1 / 4; c += NUM_THREADS / 32)
    {
        float4 g[H];
        #pragma unroll
        for (int h = 0; h < H; ++h) g[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = lane; i < LR; i += 32)
        {
            float ti = t_s[i];
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                // upx layout (H, D / 4, LR, 4): consecutive lanes (consecutive i) read
                // consecutive 8-byte quads, so the rank loop is fully coalesced
                half4 u = *(const half4*) (upt + ((((size_t) h * (D / 4) + c) * LR + i) * 4));
                g[h].x = fmaf(ti, LOW_TO_FLOAT(u.x), g[h].x);
                g[h].y = fmaf(ti, HIGH_TO_FLOAT(u.x), g[h].y);
                g[h].z = fmaf(ti, LOW_TO_FLOAT(u.y), g[h].z);
                g[h].w = fmaf(ti, HIGH_TO_FLOAT(u.y), g[h].w);
            }
        }
        #pragma unroll
        for (int h = 0; h < H; ++h)
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                g[h].x += __shfl_xor_sync(0xffffffffu, g[h].x, offset);
                g[h].y += __shfl_xor_sync(0xffffffffu, g[h].y, offset);
                g[h].z += __shfl_xor_sync(0xffffffffu, g[h].z, offset);
                g[h].w += __shfl_xor_sync(0xffffffffu, g[h].w, offset);
            }
        if (lane != 0) continue;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            half4 wq = *(const half4*) (w + (size_t) h * D + 4 * c);
            float coef = rmr_s[h] * inv_h;
            o.x = fmaf(sigmoidf_(g[h].x) * coef * LOW_TO_FLOAT(wq.x),  s.x, o.x);
            o.y = fmaf(sigmoidf_(g[h].y) * coef * HIGH_TO_FLOAT(wq.x), s.y, o.y);
            o.z = fmaf(sigmoidf_(g[h].z) * coef * LOW_TO_FLOAT(wq.y),  s.z, o.z);
            o.w = fmaf(sigmoidf_(g[h].w) * coef * HIGH_TO_FLOAT(wq.y), s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
    }
}


// -----------------------------------------------------------------------------
// ROCm sibling: GatedResidual decode mix, restructured (Qwen3.8, 97 sites per token)
// -----------------------------------------------------------------------------
// rocprofv3 on Qwen3.8-Flash-Next decode (b22b243, ctx 512): gr_dots_kernel 60.8 us and
// gr_finalize_kernel 37.5 us per site, 9.5 ms of a 34.8 ms busy token (27%) -- for 6.6 MB of
// fp16 weights each, i.e. ~110 and ~175 GB/s against ~210 achievable.
//
// gr_dots_rows_kernel: gr_dots_kernel with RB fn rows per block instead of one. The thread ->
// column map (8-half column c = tid + 128 k), each per-stream fmaf chain, the __shfl_down
// tree (or its DPP twin, lane 0 bit-identical) and the 4-warp sum are unchanged, so every
// dot is bit-identical; what changes is that a block loads its stream quads ONCE into
// registers and then streams RB fn rows with all of a row's loads in flight (the one-row
// kernel had 325 blocks each re-reading the 40 KB stream stack and ~2.5 dependent loop
// trips per stream). The sum-of-squares row runs in its own block, verbatim.
//
// gr_finalize_kernel stays as it is: it already runs at ~180 GB/s, and a variant that put
// the up-gate loads ahead of the prologue and moved the butterflies to DPP measured slower
// (36.5 -> 37.9 us; RDNA_NOTES "Qwen3.8 decode").
//
// EXL3_ROCM_GR_DOTS=0 restores the one-row kernel (runtime switch, same build);
// EXL3_ROCM_GR_RB picks rows per block (2 / 4 / 8, default 4).

template <bool DPP>
__device__ __forceinline__ float gr_warp_sum_lane0(float a)
{
    if constexpr (DPP) return hc_warp_sum_lane0(a);
    for (int offset = 16; offset > 0; offset >>= 1)
        a += __shfl_down_sync(0xffffffffu, a, offset);
    return a;
}

template <int H, int RB, int NI, bool DPP>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_rows_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const half* __restrict__ fn,         // (M, H * D) half, norm weight folded in
    float* __restrict__ dots,            // (R, M + 1, H)
    const int M,
    const int D
)
{
    const int r = blockIdx.y;
    const int D4 = D / 4;
    const int D8 = D / 8;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    __shared__ float red[RB][H][GR_THREADS_A / 32];

    const int nrb = (M + RB - 1) / RB;
    if (blockIdx.x == nrb)
    {
        // sum-of-squares row: gr_dots_kernel's j == M branch
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float a = 0.0f;
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                float4 s = s4[(size_t) h * D4 + c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
            a = gr_warp_sum_lane0<DPP>(a);
            if (lane == 0) red[0][h][warp] = a;
        }
        __syncthreads();
        if (threadIdx.x < H)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < GR_THREADS_A / 32; ++w)
                v += red[0][threadIdx.x][w];
            dots[((size_t) r * (M + 1) + M) * H + threadIdx.x] = v;
        }
        return;
    }

    const int j0 = blockIdx.x * RB;

    // this thread's stream values, every stream: 8 columns at c = tid + 128 k
    float4 sa[H][NI], sb[H][NI];
    #pragma unroll
    for (int h = 0; h < H; ++h)
        #pragma unroll
        for (int k = 0; k < NI; ++k)
        {
            const int c = threadIdx.x + k * GR_THREADS_A;
            if (c < D8)
            {
                sa[h][k] = s4[(size_t) h * D4 + 2 * c];
                sb[h][k] = s4[(size_t) h * D4 + 2 * c + 1];
            }
        }

    #pragma unroll
    for (int rr = 0; rr < RB; ++rr)
    {
        const int j = j0 + rr;
        if (j >= M) break;                                   // block-uniform
        int4 f[H][NI];
        #pragma unroll
        for (int h = 0; h < H; ++h)
            #pragma unroll
            for (int k = 0; k < NI; ++k)
            {
                const int c = threadIdx.x + k * GR_THREADS_A;
                if (c < D8) f[h][k] = ((const int4*) (fn + ((size_t) j * H + h) * D))[c];
            }
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float a = 0.0f;
            #pragma unroll
            for (int k = 0; k < NI; ++k)
            {
                const int c = threadIdx.x + k * GR_THREADS_A;
                if (c < D8)
                {
                    const float4 s0 = sa[h][k];
                    const float4 s1 = sb[h][k];
                    half4 w0 = *(half4*) &f[h][k].x;
                    half4 w1 = *(half4*) &f[h][k].z;
                    a = fmaf(s0.x, LOW_TO_FLOAT(w0.x), a);
                    a = fmaf(s0.y, HIGH_TO_FLOAT(w0.x), a);
                    a = fmaf(s0.z, LOW_TO_FLOAT(w0.y), a);
                    a = fmaf(s0.w, HIGH_TO_FLOAT(w0.y), a);
                    a = fmaf(s1.x, LOW_TO_FLOAT(w1.x), a);
                    a = fmaf(s1.y, HIGH_TO_FLOAT(w1.x), a);
                    a = fmaf(s1.z, LOW_TO_FLOAT(w1.y), a);
                    a = fmaf(s1.w, HIGH_TO_FLOAT(w1.y), a);
                }
            }
            a = gr_warp_sum_lane0<DPP>(a);
            if (lane == 0) red[rr][h][warp] = a;
        }
    }
    __syncthreads();
    if (threadIdx.x < RB * H)
    {
        const int rr = threadIdx.x / H;
        const int h = threadIdx.x % H;
        const int j = j0 + rr;
        if (j < M)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < GR_THREADS_A / 32; ++w)
                v += red[rr][h][w];
            dots[((size_t) r * (M + 1) + j) * H + h] = v;
        }
    }
}

static int gr_env_int(const char* name, int def)
{
    const char* env = std::getenv(name);
    return env ? atoi(env) : def;
}


// ---------------------------------------------------------------------------------------------
// Decode mix, bytes-in-flight version (benchmarks/gr_mix_decode): the fn table is read once per
// site for all rows, with each lane's chunks of GR_J rows issued together, one warp per stream;
// the finalize issues all its 16-byte up loads before use and runs rows in groups. Cold on a PRO
// 6000: 16.1 -> 14.9 us at one row, 33 -> 26 at eight. Shape-templated on the chunks per lane
// (D / 256) and rank iterations (LR / 64); other shapes take the generic kernels above.
#define GR_THREADS_A2 128           // 4 warps: one per stream
#define GR_THREADS_C 128            // 4 warps: one column quad each
#define GR_RB 4                     // rows per accumulator pass
#define GR_MAX_R 32
#define GR_J 2                      // fn rows per dots block (share the stream loads)

// Phase A: dots[r, j, h] = <streams[r, h, :], fn[j, h, :]> for every row r (row M: sum of squares).
// One block per fn row j, one warp per stream h: each lane holds its 10 x 16-byte chunks of the
// fn row in registers (issued together: the whole 20 KB row is in flight per block), then runs
// the rows in groups of GR_RB against the L2-resident streams. The table is read once per site
// regardless of R.
template <int H, int NCH>
__device__ __forceinline__ void gr_dots_block
(
    const float* __restrict__ streams, const half* __restrict__ fn, float* __restrict__ dots,
    const int M, const int D, const int R, const int block
)
{
    const int j0 = block * GR_J;
    const int h = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int D8 = D / 8;                // chunks per stream row
    if (j0 < M)
    {
        int4 f[GR_J][NCH];
        #pragma unroll
        for (int jj = 0; jj < GR_J; ++jj)
        {
            const int4* f8 = (const int4*) (fn + ((size_t) min(j0 + jj, M - 1) * H + h) * D);
            #pragma unroll
            for (int k = 0; k < NCH; ++k)
            {
                const int c = lane + k * 32;
                f[jj][k] = (c < D8) ? f8[c] : make_int4(0, 0, 0, 0);
            }
        }
        for (int r0 = 0; r0 < R; r0 += GR_RB)
        {
            float acc[GR_RB][GR_J];
            #pragma unroll
            for (int q = 0; q < GR_RB; ++q)
                #pragma unroll
                for (int jj = 0; jj < GR_J; ++jj) acc[q][jj] = 0.0f;
            #pragma unroll
            for (int k = 0; k < NCH; ++k)
            {
                const int c = lane + k * 32;
                const bool ok = c < D8;
                #pragma unroll
                for (int q = 0; q < GR_RB; ++q)
                {
                    const int r = r0 + q;
                    const float4* s4 = (const float4*) (streams + ((size_t) min(r, R - 1) * H + h) * D + (ok ? c : 0) * 8);
                    float4 s0 = s4[0], s1 = s4[1];
                    #pragma unroll
                    for (int jj = 0; jj < GR_J; ++jj)
                    {
                        const half2* w2 = (const half2*) &f[jj][k];
                        float2 w0 = __half22float2(w2[0]), w1 = __half22float2(w2[1]), w2_ = __half22float2(w2[2]), w3 = __half22float2(w2[3]);
                        float a = acc[q][jj];
                        a = fmaf(s0.x, w0.x, a); a = fmaf(s0.y, w0.y, a); a = fmaf(s0.z, w1.x, a); a = fmaf(s0.w, w1.y, a);
                        a = fmaf(s1.x, w2_.x, a); a = fmaf(s1.y, w2_.y, a); a = fmaf(s1.z, w3.x, a); a = fmaf(s1.w, w3.y, a);
                        acc[q][jj] = a;
                    }
                }
            }
            #pragma unroll
            for (int q = 0; q < GR_RB; ++q)
                #pragma unroll
                for (int jj = 0; jj < GR_J; ++jj)
                {
                    float a = acc[q][jj];
                    for (int offset = 16; offset > 0; offset >>= 1) a += __shfl_down_sync(0xffffffffu, a, offset);
                    if (lane == 0 && r0 + q < R && j0 + jj < M) dots[((size_t) (r0 + q) * (M + 1) + j0 + jj) * H + h] = a;
                }
        }
    }
    else
    {
        // sum of squares per (row, stream)
        for (int r = 0; r < R; ++r)
        {
            const float4* s4 = (const float4*) (streams + ((size_t) r * H + h) * D);
            float a = 0.0f;
            for (int c = lane; c < D / 4; c += 32)
            {
                float4 s = s4[c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
            for (int offset = 16; offset > 0; offset >>= 1) a += __shfl_down_sync(0xffffffffu, a, offset);
            if (lane == 0) dots[((size_t) r * (M + 1) + M) * H + h] = a;
        }
    }
}

// Phase B: rmr and the silu'd latent per row (redundantly per block, into shared memory), then
// one warp per column quad: the up-gate dots over the low rank with 16-byte loads (two ranks per
// lane, LR / 64 unrolled iterations, all four streams issued together), rows in groups of GR_RB
// re-reading the (L2-resident) up columns; epilogue: sigmoid gates, stream mean, half/float out.
template <int H, bool HALF_OUT, int NIT>
__device__ __forceinline__ void gr_finalize_block
(
    const float* __restrict__ streams, const float* __restrict__ dots, const half* __restrict__ upt,
    const half* __restrict__ w, float* __restrict__ post, void* __restrict__ mixed,
    const int D, const int LR, const int R, const float rms_eps, const int block
)
{
    const int M = LR + (post ? H : 0);
    extern __shared__ float t_s[];       // (R, LR) then rmr (R, H)
    float* rmr_s = t_s + (size_t) R * LR;
    const int tid = threadIdx.x;
    if (tid < R * H)
    {
        const int r = tid / H, h = tid % H;
        rmr_s[tid] = rsqrtf(dots[((size_t) r * (M + 1) + M) * H + h] / (float) D + rms_eps);
    }
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int idx = tid; idx < R * LR; idx += GR_THREADS_C)
    {
        const int r = idx / LR, i = idx % LR;
        const float* dr = dots + ((size_t) r * (M + 1) + i) * H;
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h) v = fmaf(rmr_s[r * H + h], dr[h], v);
        v *= inv_h;
        t_s[idx] = v * sigmoidf_(v);
    }
    if (post && block == 0 && tid < R * H)
    {
        const int r = tid / H, h = tid % H;
        const float* dr = dots + ((size_t) r * (M + 1) + LR + h) * H;
        float v = 0.0f;
        #pragma unroll
        for (int hh = 0; hh < H; ++hh) v = fmaf(rmr_s[r * H + hh], dr[hh], v);
        post[tid] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    const int lane = tid % 32, warp = tid / 32;
    const int c = block * (GR_THREADS_C / 32) + warp;           // column quad
    if (c >= D / 4) return;
    const int D4 = D / 4;
    for (int r0 = 0; r0 < R; r0 += GR_RB)
    {
        float4 g[GR_RB][H];
        #pragma unroll
        for (int q = 0; q < GR_RB; ++q)
            #pragma unroll
            for (int h = 0; h < H; ++h) g[q][h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int4 u4[NIT][H];
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
        {
            const int i0 = it * 64 + lane * 2;
            #pragma unroll
            for (int h = 0; h < H; ++h)
                u4[it][h] = (i0 < LR) ? *(const int4*) (upt + ((((size_t) h * D4 + c) * LR + i0) * 4)) : make_int4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
        {
            const int i0 = it * 64 + lane * 2;              // ranks i0, i0 + 1
            #pragma unroll
            for (int q = 0; q < GR_RB; ++q)
            {
                const int r = r0 + q;
                if (r >= R) continue;
                const float t0 = i0 < LR ? t_s[(size_t) r * LR + i0] : 0.0f;
                const float t1 = i0 + 1 < LR ? t_s[(size_t) r * LR + i0 + 1] : 0.0f;
                #pragma unroll
                for (int h = 0; h < H; ++h)
                {
                    const half2* u2 = (const half2*) &u4[it][h];
                    float2 a0 = __half22float2(u2[0]), a1 = __half22float2(u2[1]);   // rank i0: 4 columns
                    float2 b0 = __half22float2(u2[2]), b1 = __half22float2(u2[3]);   // rank i0 + 1
                    g[q][h].x = fmaf(t0, a0.x, fmaf(t1, b0.x, g[q][h].x));
                    g[q][h].y = fmaf(t0, a0.y, fmaf(t1, b0.y, g[q][h].y));
                    g[q][h].z = fmaf(t0, a1.x, fmaf(t1, b1.x, g[q][h].z));
                    g[q][h].w = fmaf(t0, a1.y, fmaf(t1, b1.y, g[q][h].w));
                }
            }
        }
        #pragma unroll
        for (int q = 0; q < GR_RB; ++q)
            #pragma unroll
            for (int h = 0; h < H; ++h)
                for (int offset = 16; offset > 0; offset >>= 1)
                {
                    g[q][h].x += __shfl_xor_sync(0xffffffffu, g[q][h].x, offset);
                    g[q][h].y += __shfl_xor_sync(0xffffffffu, g[q][h].y, offset);
                    g[q][h].z += __shfl_xor_sync(0xffffffffu, g[q][h].z, offset);
                    g[q][h].w += __shfl_xor_sync(0xffffffffu, g[q][h].w, offset);
                }
        if (lane != 0) continue;
        #pragma unroll
        for (int q = 0; q < GR_RB; ++q)
        {
            const int r = r0 + q;
            if (r >= R) continue;
            const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
            float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                float4 sv = s4[(size_t) h * D4 + c];
                const half2* wq = (const half2*) (w + (size_t) h * D + 4 * c);
                float2 w0 = __half22float2(wq[0]), w1 = __half22float2(wq[1]);
                const float coef = rmr_s[r * H + h] * inv_h;
                o.x = fmaf(sigmoidf_(g[q][h].x) * coef * w0.x, sv.x, o.x);
                o.y = fmaf(sigmoidf_(g[q][h].y) * coef * w0.y, sv.y, o.y);
                o.z = fmaf(sigmoidf_(g[q][h].z) * coef * w1.x, sv.z, o.z);
                o.w = fmaf(sigmoidf_(g[q][h].w) * coef * w1.y, sv.w, o.w);
            }
            if (HALF_OUT)
            {
                half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
                out2[c * 2] = __floats2half2_rn(o.x, o.y);
                out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
            }
            else
                ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
        }
    }
}


template <int H, int NCH>
__global__ __launch_bounds__(GR_THREADS_A2)
void gr_dots_kernel2(const float* __restrict__ streams, const half* __restrict__ fn, float* __restrict__ dots, const int M, const int D, const int R)
{
    gr_dots_block<H, NCH>(streams, fn, dots, M, D, R, blockIdx.x);
}
template <int H, bool HALF_OUT, int NIT>
__global__ __launch_bounds__(GR_THREADS_C)
void gr_finalize_kernel2(const float* __restrict__ streams, const float* __restrict__ dots, const half* __restrict__ upt, const half* __restrict__ w,
                         float* __restrict__ post, void* __restrict__ mixed, const int D, const int LR, const int R, const float rms_eps)
{
    gr_finalize_block<H, HALF_OUT, NIT>(streams, dots, upt, w, post, mixed, D, LR, R, rms_eps, blockIdx.x);
}

// Residual update for one sublayer site: x[h, d] <- post[h] * y[d] + sum_h' comb[h', h] *
// x[h', d]. Without comb (GatedResidual): x[h, d] <- post[h] * y[d] + x[h, d]. Pure per-column
// mix of the H stream rows, so it runs in place: each thread loads all H values of its columns
// into registers before writing any back.

template <int H, typename Y_T, bool HAS_COMB>
__global__ __launch_bounds__(NUM_THREADS)
void hc_apply_kernel
(
    float* __restrict__ x,               // (R, H, D), updated in place
    const Y_T* __restrict__ y,           // (R, D) float or half
    const float* __restrict__ post,      // (R, H)
    const float* __restrict__ comb,      // (R, H, H), or null
    const int D,
    const int chunk_cols                 // multiple of 4
)
{
    const int r = blockIdx.y;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;

    float post_r[H];
    float comb_r[H][H];
    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        post_r[h] = __ldg(post + (size_t) r * H + h);
        if (HAS_COMB)
        {
            #pragma unroll
            for (int g = 0; g < H; ++g)
                comb_r[h][g] = __ldg(comb + ((size_t) r * H + h) * H + g);
        }
    }

    float4* x4 = (float4*) (x + (size_t) r * H * D);
    for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS)
    {
        float4 xv[H];
        #pragma unroll
        for (int h = 0; h < H; ++h)
            xv[h] = x4[(size_t) h * D4 + c];

        float4 yv;
        if constexpr (std::is_same_v<Y_T, half>)
        {
            half2 y01 = ((const half2*) (y + (size_t) r * D))[c * 2];
            half2 y23 = ((const half2*) (y + (size_t) r * D))[c * 2 + 1];
            float2 lo = __half22float2(y01);
            float2 hi = __half22float2(y23);
            yv = make_float4(lo.x, lo.y, hi.x, hi.y);
        }
        else
            yv = ((const float4*) (y + (size_t) r * D))[c];

        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 o;
            o.x = post_r[h] * yv.x;
            o.y = post_r[h] * yv.y;
            o.z = post_r[h] * yv.z;
            o.w = post_r[h] * yv.w;
            if (HAS_COMB)
            {
                #pragma unroll
                for (int g = 0; g < H; ++g)
                {
                    o.x = fmaf(comb_r[g][h], xv[g].x, o.x);
                    o.y = fmaf(comb_r[g][h], xv[g].y, o.y);
                    o.z = fmaf(comb_r[g][h], xv[g].z, o.z);
                    o.w = fmaf(comb_r[g][h], xv[g].w, o.w);
                }
            }
            else
            {
                o.x += xv[h].x;
                o.y += xv[h].y;
                o.z += xv[h].z;
                o.w += xv[h].w;
            }
            x4[(size_t) h * D4 + c] = o;
        }
    }
}

// Shared launch logic. mode: fn rows M = 2H + H^2 (mix) or H (head)

// ROCm sibling: a deferred hc_apply (x <- post * y + comb^T x on the same streams)
struct HcPendingApply
{
    const void* y;        // (R, D)
    bool y_half;
    const float* post;    // (R, H)
    const float* comb;    // (R, H, H)
};

static void hc_apply_launch(const at::Tensor& x, const HcPendingApply* pend, cudaStream_t stream);

// hc_apply_kernel's arithmetic for stream g, then hc_mix_partials_kernel's for the chunk
// of stream g this group of NUM_THREADS_A threads owns. Block = H groups over the SAME
// column range of the H streams, so every stream value the apply needs is in the block:
// the old values go through LDS before anyone writes, then each group stores its stream
// in place -- no other block touches these columns. Chunk g * cps + blockIdx.x gets
// exactly the per-thread accumulation sequence and the two-warp reduce of the unfused
// partials kernel (same chunk_cols, same thread -> column map), so partials, and with
// them post / comb / collapsed, are bit-identical to hc_apply + hc_mix.
template <int H, int M_, typename FN_T>
__global__ __launch_bounds__(H * NUM_THREADS_A, 1)
void hc_apply_partials_kernel
(
    float* __restrict__ streams,         // (R, H * D), updated in place
    const float* __restrict__ y,         // (R, D)
    const float* __restrict__ post,      // (R, H)
    const float* __restrict__ comb,      // (R, H, H)
    const FN_T* __restrict__ fn,         // (M, H * D)
    float* __restrict__ partials,        // (R, chunksA, M + 1)
    const int row_len,
    const int chunk_cols,
    const int n_chunks_a
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int g = threadIdx.x / NUM_THREADS_A;
    const int t = threadIdx.x % NUM_THREADS_A;
    const int D = row_len / H;
    const int D4 = D / 4;
    const int cps = n_chunks_a / H;
    const int chunk = g * cps + blockIdx.x;
    const int c0 = chunk * chunk_cols;
    const int c1 = min(c0 + chunk_cols, row_len);

    float post_r[H];
    float comb_r[H][H];
    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        post_r[h] = __ldg(post + (size_t) r * H + h);
        #pragma unroll
        for (int q = 0; q < H; ++q)
            comb_r[h][q] = __ldg(comb + ((size_t) r * H + h) * H + q);
    }

    float4* s4 = (float4*) (streams + (size_t) r * row_len);
    const float4* y4 = (const float4*) (y + (size_t) r * D);
    const int row_len4 = row_len / 4;

    __shared__ float4 xs[H][NUM_THREADS_A];

    float acc[M + 1];
    #pragma unroll
    for (int k = 0; k <= M; ++k) acc[k] = 0.0f;

    // Iteration i: every group at in-stream float4 column (c0 - g * D) / 4 + t + i * 64
    const int n_it = (chunk_cols / 4 + NUM_THREADS_A - 1) / NUM_THREADS_A;
    for (int i = 0; i < n_it; ++i)
    {
        const int c = c0 / 4 + t + i * NUM_THREADS_A;     // flattened float4 index
        const bool valid = c < c1 / 4;
        const int cd = c - g * D4;                        // column within stream g
        // fn rows first: independent of the apply, so all M loads are in flight
        // together with the stream load (as in the unfused kernel)
        float4 wv[M];
        if (valid)
        {
            #pragma unroll
            for (int j = 0; j < M; ++j)
            {
                if constexpr (std::is_same_v<FN_T, half>)
                {
                    int2 pk = ((const int2*) fn)[(size_t) j * row_len4 + c];
                    float2 lo = __half22float2(*(const half2*) &pk.x);
                    float2 hi = __half22float2(*(const half2*) &pk.y);
                    wv[j] = make_float4(lo.x, lo.y, hi.x, hi.y);
                }
                else
                    wv[j] = ((const float4*) fn)[(size_t) j * row_len4 + c];
            }
            xs[g][t] = s4[c];
        }
        __syncthreads();
        if (valid)
        {
            float4 xv[H];
            #pragma unroll
            for (int h = 0; h < H; ++h) xv[h] = xs[h][t];
            float4 yv = y4[cd];
            float4 o;
            o.x = post_r[g] * yv.x;
            o.y = post_r[g] * yv.y;
            o.z = post_r[g] * yv.z;
            o.w = post_r[g] * yv.w;
            #pragma unroll
            for (int q = 0; q < H; ++q)
            {
                o.x = fmaf(comb_r[q][g], xv[q].x, o.x);
                o.y = fmaf(comb_r[q][g], xv[q].y, o.y);
                o.z = fmaf(comb_r[q][g], xv[q].z, o.z);
                o.w = fmaf(comb_r[q][g], xv[q].w, o.w);
            }
            s4[c] = o;

            float4 sv = o;
            acc[M] = fmaf(sv.x, sv.x, acc[M]);
            acc[M] = fmaf(sv.y, sv.y, acc[M]);
            acc[M] = fmaf(sv.z, sv.z, acc[M]);
            acc[M] = fmaf(sv.w, sv.w, acc[M]);
            #pragma unroll
            for (int j = 0; j < M; ++j)
            {
                const float4 w = wv[j];
                float d = fmaf(sv.x, w.x, fmaf(sv.y, w.y, fmaf(sv.z, w.z, sv.w * w.w)));
                acc[j] += d;
            }
        }
        __syncthreads();
    }

    // The unfused kernel's reduce, per group (its two warps)
    __shared__ float red[H][NUM_THREADS_A / 32][M + 1];
    const int lane = t % 32;
    const int warp = t / 32;
    #pragma unroll
    for (int k = 0; k <= M; ++k)
    {
        float v = hc_warp_sum_lane0(acc[k]);
        if (lane == 0) red[g][warp][k] = v;
    }
    __syncthreads();

    if (warp == 0)
    {
        float* out = partials + ((size_t) r * n_chunks_a + chunk) * (M + 1);
        for (int k = lane; k <= M; k += 32)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_THREADS_A / 32; ++w)
                v += red[g][w][k];
            out[k] = v;
        }
    }
}

static void hc_mix_launch
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    float rms_eps,
    float hc_eps,
    int sinkhorn_iters,
    at::Tensor& partials,
    at::Tensor* post,
    at::Tensor* comb,
    at::Tensor& collapsed,
    Graph* graph,
    const HcPendingApply* pend = nullptr,  // ROCm sibling: fold a preceding hc_apply in (see below)
    const HcNormArgs* nrm = nullptr        // ROCm sibling: fold the following rms_norm in
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK(streams.is_contiguous() && fn.is_contiguous(), "hc_mix: contiguous inputs required");
    int R = streams.size(0);
    int H = streams.size(1);
    int D = streams.size(2);
    int row_len = H * D;
    int M = fn.size(0);
    bool head = M == H;
    TORCH_CHECK(H == 4, "hc_mix: H = 4 only");
    TORCH_CHECK(head || M == 2 * H + H * H, "hc_mix: fn rows must be H or 2H + H^2");
    TORCH_CHECK(fn.size(1) == row_len && D % 4 == 0, "hc_mix: dims");
    bool fn_half = fn.dtype() == at::kHalf;
    if (!fn_half) TORCH_CHECK_DTYPE(fn, kFloat);
    TORCH_CHECK_DTYPE(base, kFloat);
    TORCH_CHECK_DTYPE(scale, kFloat);

    int n_chunks_a = partials.size(1);
    int chunk_cols = ((row_len / n_chunks_a + 4 * NUM_THREADS_A - 1) / (4 * NUM_THREADS_A)) * (4 * NUM_THREADS_A);
    n_chunks_a = (row_len + chunk_cols - 1) / chunk_cols;
    TORCH_CHECK(n_chunks_a <= partials.size(1), "hc_mix: partials workspace too small");

    int chunks_c = std::min(32, std::max(1, 256 / R));
    int chunk_cols_c = ((D / chunks_c + 4 * NUM_THREADS - 1) / (4 * NUM_THREADS)) * (4 * NUM_THREADS);
    int n_chunks_c = (D + chunk_cols_c - 1) / chunk_cols_c;

    bool half_out = collapsed.dtype() == at::kHalf;

    dim3 grid_a(n_chunks_a, R);
    dim3 grid_c(n_chunks_c, R);
    #define ARGS_A(FN_T) \
        (const float*) streams.data_ptr(), (const FN_T*) fn.data_ptr(), \
        (float*) partials.data_ptr(), row_len, chunk_cols
    #define ARGS_C(POST, COMB) \
        (const float*) streams.data_ptr(), (const float*) partials.data_ptr(), \
        (const float*) base.data_ptr(), (const float*) scale.data_ptr(), \
        POST, COMB, collapsed.data_ptr(), \
        D, n_chunks_a, chunk_cols_c, rms_eps, hc_eps, sinkhorn_iters
    // ROCm sibling: DPP template flag (EXL3_ROCM_HC_DPP, default on)
    const bool dpp = hc_dpp_enabled();
    if (nrm) TORCH_CHECK(!head && half_out && D <= 4096 && R <= 32, "hc_mix: folded norm needs the decode form (D <= 4096)");
    // Folded norm: one finalize block per row covers all D columns (chunk_cols_c = D)
    #define HC_FNORM(POST, COMB) \
        { \
            const int chunk_cols_c_n = ((D + 3) / 4) * 4; \
            (void) chunk_cols_c_n; \
            if (dpp) hc_mix_finalize_kernel<4, 24, false, true, true, true><<<dim3(1, R), NUM_THREADS, 0, stream>>>(ARGS_CN(POST, COMB), *nrm); \
            else     hc_mix_finalize_kernel<4, 24, false, true, false, true><<<dim3(1, R), NUM_THREADS, 0, stream>>>(ARGS_CN(POST, COMB), *nrm); \
        }
    #define ARGS_CN(POST, COMB) \
        (const float*) streams.data_ptr(), (const float*) partials.data_ptr(), \
        (const float*) base.data_ptr(), (const float*) scale.data_ptr(), \
        POST, COMB, collapsed.data_ptr(), \
        D, n_chunks_a, chunk_cols_c_n, rms_eps, hc_eps, sinkhorn_iters

    // ROCm sibling: a pending hc_apply on these streams, fused into the partials kernel
    // when each stream is a whole number of partials chunks (always at decode R);
    // otherwise run upstream's hc_apply first -- same result either way
    if (pend)
    {
        const bool fusable = !head && (D % chunk_cols) == 0 && n_chunks_a % H == 0
                             && pend->y_half == false;
        if (fusable)
        {
            dim3 grid_f(n_chunks_a / H, R);
            #define ARGS_F(FN_T) \
                (float*) streams.data_ptr(), (const float*) pend->y, pend->post, pend->comb, \
                (const FN_T*) fn.data_ptr(), (float*) partials.data_ptr(), row_len, chunk_cols, n_chunks_a
            if (fn_half) hc_apply_partials_kernel<4, 24, half><<<grid_f, 4 * NUM_THREADS_A, 0, stream>>>(ARGS_F(half));
            else         hc_apply_partials_kernel<4, 24, float><<<grid_f, 4 * NUM_THREADS_A, 0, stream>>>(ARGS_F(float));
            #undef ARGS_F
            cuda_check(cudaPeekAtLastError());
            float* post_p = (float*) post->data_ptr();
            float* comb_p = (float*) comb->data_ptr();
            #define HC_F2(HO_) \
                if (dpp) hc_mix_finalize_kernel<4, 24, false, HO_, true><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(post_p, comb_p)); \
                else     hc_mix_finalize_kernel<4, 24, false, HO_, false><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(post_p, comb_p));
            if (nrm)           { HC_FNORM(post_p, comb_p) }
            else if (half_out) { HC_F2(true) }
            else               { HC_F2(false) }
            #undef HC_F2
            cuda_check(cudaPeekAtLastError());
            return;
        }
        hc_apply_launch(streams, pend, stream);
    }
    #define HC_P(M_, FN_T) \
        if (dpp) hc_mix_partials_kernel<4, M_, FN_T, true><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(FN_T)); \
        else     hc_mix_partials_kernel<4, M_, FN_T, false><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(FN_T));
    #define HC_F(M_, HEAD_, HO_, POST, COMB) \
        if (dpp) hc_mix_finalize_kernel<4, M_, HEAD_, HO_, true><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(POST, COMB)); \
        else     hc_mix_finalize_kernel<4, M_, HEAD_, HO_, false><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(POST, COMB));
    if (!head)
    {
        if (fn_half) { HC_P(24, half) }
        else         { HC_P(24, float) }
        cuda_check(cudaPeekAtLastError());
        float* post_p = (float*) post->data_ptr();
        float* comb_p = (float*) comb->data_ptr();
        if (nrm)           { HC_FNORM(post_p, comb_p) }
        else if (half_out) { HC_F(24, false, true, post_p, comb_p) }
        else               { HC_F(24, false, false, post_p, comb_p) }
    }
    else
    {
        if (fn_half) { HC_P(4, half) }
        else         { HC_P(4, float) }
        cuda_check(cudaPeekAtLastError());
        if (half_out) { HC_F(4, true, true, nullptr, nullptr) }
        else          { HC_F(4, true, false, nullptr, nullptr) }
    }
    #undef HC_P
    #undef HC_F
    #undef HC_FNORM
    #undef ARGS_CN
    #undef ARGS_A
    #undef ARGS_C
    cuda_check(cudaPeekAtLastError());
}

int hc_mix_num_chunks(int R, int row_len)
{
    int chunks_a = std::min(128, std::max(1, 512 / std::max(R, 1)));
    int chunk_cols = ((row_len / chunks_a + 4 * NUM_THREADS_A - 1) / (4 * NUM_THREADS_A)) * (4 * NUM_THREADS_A);
    return (row_len + chunk_cols - 1) / chunk_cols;
}

void hc_mix
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (2H + H^2, H * D) float
    const at::Tensor& base,              // (2H + H^2) float
    const at::Tensor& scale,             // (3) float
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,                 // (R, chunks, M + 1) float workspace
    at::Tensor post,                     // (R, H) float out
    at::Tensor comb,                     // (R, H, H) float out
    at::Tensor collapsed                 // (R, D) float or half out
)
{
    hc_mix_launch
    (
        streams,
        fn,
        base,
        scale,
        (float) rms_eps,
        (float) hc_eps,
        (int) sinkhorn_iters,
        partials,
        &post,
        &comb,
        collapsed,
        nullptr
    );
}

void hc_head
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (H, H * D) float
    const at::Tensor& base,              // (H) float
    const at::Tensor& scale,             // (1) float
    double rms_eps,
    double hc_eps,
    at::Tensor partials,                 // (R, chunks, H + 1) float workspace
    at::Tensor collapsed                 // (R, D) float or half out
)
{
    hc_mix_launch
    (
        streams,
        fn,
        base,
        scale,
        (float) rms_eps,
        (float) hc_eps,
        0,
        partials,
        nullptr,
        nullptr,
        collapsed,
        nullptr
    );
}

void hc_apply
(
    at::Tensor x,                        // (R, H, D) float, updated IN PLACE
    const at::Tensor& y,                 // (R, D) float or half
    const at::Tensor& post,              // (R, H) float
    const c10::optional<at::Tensor>& comb   // (R, H, H) float, or none (x[h] += post[h] * y)
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(x, kFloat);
    TORCH_CHECK_DTYPE(post, kFloat);
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && post.is_contiguous(), "hc_apply: contiguous inputs required");
    int R = x.size(0);
    int H = x.size(1);
    int D = x.size(2);
    TORCH_CHECK(H == 4, "hc_apply: H = 4 only");
    TORCH_CHECK(D % 4 == 0, "hc_apply: dims");
    TORCH_CHECK(y.size(0) == R && y.size(-1) == D, "hc_apply: y shape");
    TORCH_CHECK(post.size(0) == R, "hc_apply: gate shapes");
    const float* comb_p = nullptr;
    if (comb)
    {
        TORCH_CHECK_DTYPE(comb.value(), kFloat);
        TORCH_CHECK(comb.value().is_contiguous() && comb.value().size(0) == R, "hc_apply: comb shape");
        comb_p = (const float*) comb.value().data_ptr();
    }

    int chunks_c = std::min(32, std::max(1, 256 / R));
    int chunk_cols = ((D / chunks_c + 4 * NUM_THREADS - 1)
                      / (4 * NUM_THREADS)) * (4 * NUM_THREADS);
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;

    dim3 grid(n_chunks, R);
    #define ARGS(Y_T) \
        (float*) x.data_ptr(), (const Y_T*) y.data_ptr(), \
        (const float*) post.data_ptr(), comb_p, D, chunk_cols
    if (y.dtype() == at::kHalf)
    {
        if (comb_p) hc_apply_kernel<4, half, true><<<grid, NUM_THREADS, 0, stream>>>(ARGS(half));
        else        hc_apply_kernel<4, half, false><<<grid, NUM_THREADS, 0, stream>>>(ARGS(half));
    }
    else
    {
        if (comb_p) hc_apply_kernel<4, float, true><<<grid, NUM_THREADS, 0, stream>>>(ARGS(float));
        else        hc_apply_kernel<4, float, false><<<grid, NUM_THREADS, 0, stream>>>(ARGS(float));
    }
    #undef ARGS
    cuda_check(cudaPeekAtLastError());
}

void gr_mix
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (M, H * D) half: cat(down, inject) * w
    const at::Tensor& upt,               // (H, D / 4, LR, 4) half: up repacked lane-contiguous
    const at::Tensor& w,                 // (H * D) half norm weight (incl +1)
    double rms_eps,
    at::Tensor dots,                     // (R, M + 1, H) float workspace
    c10::optional<at::Tensor> post,      // (R, H) float out, or none (final-mixer form)
    at::Tensor mixed                     // (R, D) half or float out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK_DTYPE(fn, kHalf);
    TORCH_CHECK_DTYPE(upt, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(dots, kFloat);
    TORCH_CHECK(streams.is_contiguous() && fn.is_contiguous() && upt.is_contiguous() &&
                w.is_contiguous() && dots.is_contiguous(), "gr_mix: contiguous inputs required");
    int R = streams.size(0);
    int H = streams.size(1);
    int D = streams.size(2);
    int M = fn.size(0);
    TORCH_CHECK(upt.dim() == 4 && upt.size(0) == H && upt.size(1) == D / 4 && upt.size(3) == 4,
                "gr_mix: upt must be the (H, D / 4, LR, 4) repacked layout");
    int LR = upt.size(2);
    TORCH_CHECK(H == 4, "gr_mix: H = 4 only");
    TORCH_CHECK(D % 8 == 0, "gr_mix: dims");
    TORCH_CHECK(M == LR + (post ? H : 0), "gr_mix: fn rows must be LR (+ H with post)");
    TORCH_CHECK(fn.size(1) == H * D && w.numel() == H * D, "gr_mix: dims");
    TORCH_CHECK(dots.size(0) == R && dots.size(1) == M + 1 && dots.size(2) == H, "gr_mix: dots shape");

    // ROCm sibling: RB-rows-per-block dots (bit-identical; see gr_dots_rows_kernel)
    static const bool gr_dots_on = gr_env_int("EXL3_ROCM_GR_DOTS", 1) != 0;

    // v1.5.3: upstream added a shape-templated decode pair (gr_dots_kernel2 / gr_finalize_kernel2,
    // "bytes-in-flight") ahead of the generic kernels. Its dot reduction order differs from the
    // generic kernel's, so it is NOT bit-identical to v1.5.0. EXL3_ROCM_GR_DOTS=1 (default) keeps
    // the RDNA rows dots + generic finalize (bit-identical to v1.5.0 / perf/stack); =0 takes
    // upstream v1.5.3's dispatch verbatim (fast pair where eligible, else the generic kernels).
    if (!gr_dots_on)
    {
        const int nch = D / 256, nit = (LR + 63) / 64;
        const bool fast = R <= GR_MAX_R && D % 256 == 0 && (nch == 10 || nch == 16 || nch == 20) && (nit == 5 || nit == 8 || nit == 16) && LR % 2 == 0;
        if (fast)
        {
            const float* s_p = (const float*) streams.data_ptr(); const half* fn_p = (const half*) fn.data_ptr(); float* dots_p = (float*) dots.data_ptr();
            const int grid_a2 = (M + GR_J - 1) / GR_J + 1;
            switch (nch)
            {
                case 10: gr_dots_kernel2<4, 10><<<grid_a2, GR_THREADS_A2, 0, stream>>>(s_p, fn_p, dots_p, M, D, R); break;
                case 16: gr_dots_kernel2<4, 16><<<grid_a2, GR_THREADS_A2, 0, stream>>>(s_p, fn_p, dots_p, M, D, R); break;
                case 20: gr_dots_kernel2<4, 20><<<grid_a2, GR_THREADS_A2, 0, stream>>>(s_p, fn_p, dots_p, M, D, R); break;
            }
            cuda_check(cudaPeekAtLastError());
            const int grid_c2 = D / 4 / (GR_THREADS_C / 32);
            const int smem2 = (R * LR + R * 4) * sizeof(float);
            float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
            #define ARGS2 \
                (const float*) streams.data_ptr(), (const float*) dots.data_ptr(), (const half*) upt.data_ptr(), (const half*) w.data_ptr(), \
                post_p, mixed.data_ptr(), D, LR, R, (float) rms_eps
            const bool hout = mixed.dtype() == at::kHalf;
            switch (nit)
            {
                case 5:  if (hout) gr_finalize_kernel2<4, true, 5><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); else gr_finalize_kernel2<4, false, 5><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); break;
                case 8:  if (hout) gr_finalize_kernel2<4, true, 8><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); else gr_finalize_kernel2<4, false, 8><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); break;
                case 16: if (hout) gr_finalize_kernel2<4, true, 16><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); else gr_finalize_kernel2<4, false, 16><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); break;
            }
            #undef ARGS2
            cuda_check(cudaPeekAtLastError());
            return;
        }
    }

    static const int gr_rb = gr_env_int("EXL3_ROCM_GR_RB", 4);
    const bool dpp = hc_dpp_enabled();
    const int ni = (D / 8 + GR_THREADS_A - 1) / GR_THREADS_A;
    bool launched = false;
    if (gr_dots_on && ni >= 1 && ni <= 3 && (gr_rb == 2 || gr_rb == 4 || gr_rb == 8))
    {
        dim3 grid_r((M + gr_rb - 1) / gr_rb + 1, R);
        #define GR_DOTS_ARGS (const float*) streams.data_ptr(), (const half*) fn.data_ptr(), (float*) dots.data_ptr(), M, D
        #define GR_DOTS_NI(RB_, DPP_) \
            if (ni == 1) gr_dots_rows_kernel<4, RB_, 1, DPP_><<<grid_r, GR_THREADS_A, 0, stream>>>(GR_DOTS_ARGS); \
            else if (ni == 2) gr_dots_rows_kernel<4, RB_, 2, DPP_><<<grid_r, GR_THREADS_A, 0, stream>>>(GR_DOTS_ARGS); \
            else gr_dots_rows_kernel<4, RB_, 3, DPP_><<<grid_r, GR_THREADS_A, 0, stream>>>(GR_DOTS_ARGS);
        #define GR_DOTS_RB(DPP_) \
            if (gr_rb == 2) { GR_DOTS_NI(2, DPP_) } \
            else if (gr_rb == 4) { GR_DOTS_NI(4, DPP_) } \
            else { GR_DOTS_NI(8, DPP_) }
        if (dpp) { GR_DOTS_RB(true) } else { GR_DOTS_RB(false) }
        #undef GR_DOTS_RB
        #undef GR_DOTS_NI
        #undef GR_DOTS_ARGS
        launched = true;
    }
    if (!launched)
    {
        dim3 grid_a(M + 1, R);
        gr_dots_kernel<4><<<grid_a, GR_THREADS_A, 0, stream>>>
        (
            (const float*) streams.data_ptr(), (const half*) fn.data_ptr(),
            (float*) dots.data_ptr(), M, D
        );
    }
    cuda_check(cudaPeekAtLastError());

    // Phase C is warp-per-column-quad: chunk at warp granularity (4 * NUM_THREADS / 32
    // columns) so small R still fills the device
    const int gran = 4 * (NUM_THREADS / 32);
    int chunks_c = std::max(1, std::min((D + gran - 1) / gran, 512 / R));
    int chunk_cols = ((D / chunks_c + gran - 1) / gran) * gran;
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;
    dim3 grid_c(n_chunks, R);
    int smem = LR * sizeof(float);
    float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
    #define ARGS \
        (const float*) streams.data_ptr(), (const float*) dots.data_ptr(), \
        (const half*) upt.data_ptr(), (const half*) w.data_ptr(), \
        post_p, mixed.data_ptr(), D, LR, chunk_cols, (float) rms_eps
    const bool half_out = mixed.dtype() == at::kHalf;
    if (half_out)
        gr_finalize_kernel<4, true><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS);
    else
        gr_finalize_kernel<4, false><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS);
    #undef ARGS
    cuda_check(cudaPeekAtLastError());
}


// =============================================================================
// ROCm sibling: hc_apply folded into the next site's hc_mix (exl3_rocm::hc_apply_mix)
// =============================================================================
// DS4 decode runs 86 hyper-connection sites per token, each apply_ (one hc_apply
// launch) immediately followed by the next site's mix (partials + finalize) on the
// same streams. rocm_py defers the apply and hands it to the mix, which runs
// hc_apply_partials_kernel in place of the partials kernel: one launch less per site,
// bit-identical (see the kernel). Anything not fusable runs the unfused pair.

static void hc_apply_launch(const at::Tensor& x, const HcPendingApply* pend, cudaStream_t stream)
{
    int R = x.size(0);
    int D = x.size(2);
    int chunks_c = std::min(32, std::max(1, 256 / R));
    int chunk_cols = ((D / chunks_c + 4 * NUM_THREADS - 1)
                      / (4 * NUM_THREADS)) * (4 * NUM_THREADS);
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;
    dim3 grid(n_chunks, R);
    if (pend->y_half)
        hc_apply_kernel<4, half, true><<<grid, NUM_THREADS, 0, stream>>>
            ((float*) x.data_ptr(), (const half*) pend->y, pend->post, pend->comb, D, chunk_cols);
    else
        hc_apply_kernel<4, float, true><<<grid, NUM_THREADS, 0, stream>>>
            ((float*) x.data_ptr(), (const float*) pend->y, pend->post, pend->comb, D, chunk_cols);
    cuda_check(cudaPeekAtLastError());
}

// x: (R, H, D) float streams. y / post_a / comb_a: the pending apply, or all none.
// norm_y: output of the folded rms_norm (R, D) half, or none (norm_w may be none for an
// unweighted norm). collapsed is always written.
void exl3_rocm_hc_mix
(
    at::Tensor x,
    const c10::optional<at::Tensor>& y,
    const c10::optional<at::Tensor>& post_a,
    const c10::optional<at::Tensor>& comb_a,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,
    at::Tensor post,
    at::Tensor comb,
    at::Tensor collapsed,
    const c10::optional<at::Tensor>& norm_w,
    const c10::optional<at::Tensor>& norm_y,
    double norm_eps,
    double norm_bias,
    double norm_scale
)
{
    TORCH_CHECK_DTYPE(x, kFloat);
    TORCH_CHECK(x.dim() == 3 && x.size(1) == 4 && x.is_contiguous(), "hc_mix_rdna: x must be (R, 4, D) contiguous");
    const int R = x.size(0);
    const int D = x.size(2);

    HcPendingApply pend {};
    const HcPendingApply* pend_p = nullptr;
    if (y.has_value())
    {
        TORCH_CHECK(post_a.has_value() && comb_a.has_value(), "hc_mix_rdna: pending apply needs y, post, comb");
        TORCH_CHECK_DTYPE(post_a.value(), kFloat);
        TORCH_CHECK_DTYPE(comb_a.value(), kFloat);
        TORCH_CHECK(y->is_contiguous() && post_a->is_contiguous() && comb_a->is_contiguous(), "hc_mix_rdna: contiguous inputs required");
        TORCH_CHECK(y->size(0) == R && y->size(-1) == D && post_a->size(0) == R && comb_a->size(0) == R, "hc_mix_rdna: shapes");
        const bool y_half = y->dtype() == at::kHalf;
        if (!y_half) TORCH_CHECK_DTYPE(y.value(), kFloat);
        pend = { y->data_ptr(), y_half, (const float*) post_a->data_ptr(), (const float*) comb_a->data_ptr() };
        pend_p = &pend;
    }

    HcNormArgs nrm {};
    const HcNormArgs* nrm_p = nullptr;
    if (norm_y.has_value())
    {
        TORCH_CHECK_DTYPE(norm_y.value(), kHalf);
        TORCH_CHECK(norm_y->is_contiguous() && norm_y->numel() == (int64_t) R * D, "hc_mix_rdna: norm_y shape");
        bool w_bf16 = false;
        const void* w = nullptr;
        if (norm_w.has_value())
        {
            TORCH_CHECK(norm_w->numel() == D && norm_w->is_contiguous(), "hc_mix_rdna: norm weight shape");
            w_bf16 = norm_w->dtype() == at::kBFloat16;
            if (!w_bf16) TORCH_CHECK_DTYPE(norm_w.value(), kHalf);
            w = norm_w->data_ptr();
        }
        nrm = { w, w_bf16, (half*) norm_y->data_ptr(), (float) norm_eps, (float) norm_bias, (float) norm_scale };
        nrm_p = &nrm;
    }

    hc_mix_launch(x, fn, base, scale, (float) rms_eps, (float) hc_eps, (int) sinkhorn_iters,
                  partials, &post, &comb, collapsed, nullptr, pend_p, nrm_p);
}

// =============================================================================
// ROCm sibling: GatedResidual prefill gate-mean in one pass (exl3_rocm::gr_gate_mean)
// =============================================================================
// The prefill form of GatedResidual._mix (R > 32 rows) ends in
//   mixed = (sigmoid(g.float()).view(R, H, D) * normed.float().view(R, H, D)).mean(-2).half()
// which torch runs as 2 upcast copies, a sigmoid, a multiply and a mean over fp32 (R, H * D)
// temporaries: ~1.6 ms per site at R = 2048 (rocprofv3, Qwen3.8 pp2048, 95 sites per
// forward). This kernel reads g and normed once (half) and writes mixed, with torch's
// arithmetic per element: sigmoid as 1 / (1 + expf(-x)), the product in fp32, the mean as
// ((0 + p0 + p1) + p2) + p3 times 1 / H (the reduce kernel's accumulation order for a
// 4-long strided reduction), RNE to half. Checked bitwise against the torch expression by
// rocm_tools/gr_mix_bench.py --prefill. EXL3_ROCM_GR_PREFILL=0 (rocm_py) keeps torch.

template <int H>
__global__ __launch_bounds__(256)
void gr_gate_mean_kernel
(
    const half* __restrict__ g,          // (R, H, D)
    const half* __restrict__ normed,     // (R, H, D)
    half* __restrict__ mixed,            // (R, D)
    const int R,
    const int D
)
{
    // torch rounds the product before the sum: no fma contraction anywhere in here
    #pragma clang fp contract(off)
    const int D4 = D / 4;
    const size_t n = (size_t) R * D4;
    const float inv_h = 1.0f / (float) H;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (size_t) gridDim.x * blockDim.x)
    {
        const size_t r = idx / D4;
        const int c = (int) (idx - r * D4);
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            const size_t off = ((r * H + h) * (size_t) D4 + c) * 4;
            const half4 gv = *(const half4*) (g + off);
            const half4 nv = *(const half4*) (normed + off);
            // plain operators under contract(off): the HIP __fmul_rn / __fadd_rn helpers are
            // header inlines whose fmul / fadd keep the TU's contract flag and still fused
            const float p0 = (1.0f / (1.0f + expf(-LOW_TO_FLOAT(gv.x)))) * LOW_TO_FLOAT(nv.x);
            const float p1 = (1.0f / (1.0f + expf(-HIGH_TO_FLOAT(gv.x)))) * HIGH_TO_FLOAT(nv.x);
            const float p2 = (1.0f / (1.0f + expf(-LOW_TO_FLOAT(gv.y)))) * LOW_TO_FLOAT(nv.y);
            const float p3 = (1.0f / (1.0f + expf(-HIGH_TO_FLOAT(gv.y)))) * HIGH_TO_FLOAT(nv.y);
            acc.x = acc.x + p0;
            acc.y = acc.y + p1;
            acc.z = acc.z + p2;
            acc.w = acc.w + p3;
        }
        half2* out2 = (half2*) (mixed + r * D);
        out2[c * 2] = __floats2half2_rn(acc.x * inv_h, acc.y * inv_h);
        out2[c * 2 + 1] = __floats2half2_rn(acc.z * inv_h, acc.w * inv_h);
    }
}

void exl3_rocm_gr_gate_mean(const at::Tensor& g, const at::Tensor& normed, at::Tensor mixed)
{
    const at::cuda::OptionalCUDAGuard device_guard(g.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(g, kHalf);
    TORCH_CHECK_DTYPE(normed, kHalf);
    TORCH_CHECK_DTYPE(mixed, kHalf);
    TORCH_CHECK(g.is_contiguous() && normed.is_contiguous() && mixed.is_contiguous(), "gr_gate_mean: contiguous inputs required");
    TORCH_CHECK(g.dim() == 3 && g.size(1) == 4 && normed.sizes() == g.sizes(), "gr_gate_mean: g / normed must be (R, 4, D)");
    const int R = g.size(0);
    const int D = g.size(2);
    TORCH_CHECK(D % 4 == 0 && mixed.numel() == (int64_t) R * D, "gr_gate_mean: dims");
    const int64_t n = (int64_t) R * (D / 4);
    const int blocks = (int) std::min<int64_t>((n + 255) / 256, 65535);
    if (n == 0) return;
    gr_gate_mean_kernel<4><<<blocks, 256, 0, stream>>>
        ((const half*) g.data_ptr(), (const half*) normed.data_ptr(), (half*) mixed.data_ptr(), R, D);
    cuda_check(cudaPeekAtLastError());
}

TORCH_LIBRARY_FRAGMENT(exl3_rocm, m)
{
    m.def("hc_mix_rdna(Tensor(a!) x, Tensor? y, Tensor? post_a, Tensor? comb_a, Tensor fn, Tensor base, "
          "Tensor scale, float rms_eps, float hc_eps, int sinkhorn_iters, Tensor(b!) partials, "
          "Tensor(c!) post, Tensor(d!) comb, Tensor(e!) collapsed, Tensor? norm_w, Tensor(f!)? norm_y, "
          "float norm_eps, float norm_bias, float norm_scale) -> ()");
    m.def("gr_gate_mean(Tensor g, Tensor normed, Tensor(a!) mixed) -> ()");
}

TORCH_LIBRARY_IMPL(exl3_rocm, CUDA, m)
{
    m.impl("hc_mix_rdna", &exl3_rocm_hc_mix);
    m.impl("gr_gate_mean", &exl3_rocm_gr_gate_mean);
}
