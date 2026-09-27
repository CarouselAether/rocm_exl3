// WMMA layout + regression gate for rocm/rdna_wmma.hip.h (gfx11, wave32).
//
// Everything on the device goes through the production helpers in the shipped
// header (load_matrix_a/b*, mma_sync*, load_accumulate_c*, store_matrix_c*);
// no intrinsic is re-implemented here. See docs/RDNA_WMMA.md section 7 for the
// gaps this closes and section 8 for why fp16/bf16 cases are not IEEE-exact.
//
// Every gate case is checked two ways:
//
//  1. MATH (layout/semantics). Expected C is computed on the host in double.
//     - int8 cases and cases with no rounding by construction (C identity with
//       A = 0, untouched opsel halves) are compared with ==.
//     - f16/bf16-input cases get a per-cell bound, because gfx11 WMMA does NOT
//       accumulate like IEEE fp32 (measured: products aligned to the largest
//       exponent and truncated, and a mixed-sign bias of ~1 ulp of the largest
//       product; see docs/RDNA_WMMA.md section 8). The bound is, per 16-deep K
//       step s, 16 * ulp_f32(M_s), with M_s the largest magnitude among the
//       incoming C, every product and every exact partial sum of that step;
//       plus ulp_f16(result) per step for fp16 accumulate, plus ulp_f16(want)
//       for an fp16 store. Observed errors are ~4 ulp at most, so a layout bug
//       (errors of order the values themselves) cannot hide in the bound.
//       Max error is reported in units of ulp_f32(max intermediate magnitude).
//     - On a failure, the fraction of cells matching the TRANSPOSED expectation
//       is printed (operand-order / C-layout swap signature).
//  2. GOLDEN (regression). Every case's full output buffer (including guard
//     cells) is compared bit for bit against rocm_tools/wmma_gate.golden,
//     written by `wmma_gate --record`. This gates "no change from the recorded
//     hardware + toolchain + header behaviour". Re-record only deliberately.
//
// Input design (checked by host code; a violation is a TEST-DESIGN error,
// exit 2): small non-symmetric integers, some power-of-two scaled, so the
// double reference is exact and every expected value is representable in the
// output type; every 16x16 reference tile differs from its transpose in >= 64
// cells. Padding columns of A and B hold NaN (int8: 0x77), so a loader that
// reads past its row poisons the result. Output buffers are pre-filled with a
// NaN sentinel; cells the helpers must not write are checked bitwise.
//
// Build:  rocm_tools/build_wmma_gate.sh             (see that file for env)
// Run:    <out_dir>/wmma_gate [--golden PATH]        exit 0 = PASS, 1 = FAIL,
//         <out_dir>/wmma_gate --record [--golden P]  2 = setup/design error
// Design self-check without a GPU (host C++ only, no HIP runtime touched):
//         rocm_tools/build_wmma_gate.sh --design-check
//
// INFO cases (lane-half divergence, int8 clamp saturation, fp32 accumulator
// characterization) record behaviour the code does not rely on; they never
// affect the exit code and are not part of the golden file.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <set>

// K tile of the LDS-geometry case (MoE's TILESIZE_K); shared by host and device.
#define GATE_TK 32

#ifndef WMMA_GATE_DESIGN_CHECK
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include "../exllamav3/exllamav3_ext/rocm/rdna_wmma.hip.h"
#endif

// =============================================================================
// Device kernels (one wave unless stated). All loads/stores/mmas are the
// production helpers.
// =============================================================================

#ifndef WMMA_GATE_DESIGN_CHECK

// gfx12 has no WMMA implementation in the header: mma_sync compiles to
// __builtin_trap() there, and the bf16 / f16-acc / int8 wrappers are left
// unguarded on purpose so that instantiating them fails to compile. The f32
// kernels below are therefore left alone (they build the documented trap path),
// and the other variants are excluded on gfx12 by this guard. The host refuses
// to run on anything that is not gfx11 regardless.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define GATE_NO_GFX11_WMMA 1
#endif

// f32 accumulate, chained over ksteps 16-deep K slices. A is [16, K] with row
// stride sa, B is [K, 16] with row stride sb. Optional prior is added with
// load_accumulate_c BEFORE the mma chain, so the hardware accumulates onto
// values placed by the helper's layout.
__global__ void k_f32_chain(const half* A, int sa, const half* B, int sb, int ksteps,
                            const float* prior, int sp, float* C, int sc)
{
    WmmaFragA fa; WmmaFragB fb; WmmaFragC fc;
    fc.clear();
    if (prior) rdna_wmma::load_accumulate_c(fc, prior, sp);
    for (int s = 0; s < ksteps; s++)
    {
        rdna_wmma::load_matrix_a(fa, A + s * 16, sa);
        rdna_wmma::load_matrix_b(fb, B + s * 16 * sb, sb);
        rdna_wmma::mma_sync(fc, fa, fb);
    }
    rdna_wmma::store_matrix_c(C, fc, sc);
}

// Production geometry (exl3_gemm_kernel_inner, MoE's TILESIZE_K = 32):
// A staged in LDS with SH_A_STRIDE = TK + 8 halves and read at column
// offset sub_k * 16 [GI:113, 361-362]; B staged in a [16 k][16 n] LDS tile with
// stride 18 halves (EXL3_GEMM_SH_B_DQ_STRIDE [KM:102]) and read by load_matrix_b
// [GI:438-440]. Two K tiles of 32 -> four chained mmas.
#define GATE_SH_A_STRIDE (GATE_TK + 8)
#define GATE_SH_B_STRIDE 18
__global__ void k_f32_lds(const half* A, int sa, const half* B, int sb, int ktiles,
                          float* C, int sc)
{
    __shared__ __attribute__((aligned(32))) half shA[16 * GATE_SH_A_STRIDE];
    __shared__ __attribute__((aligned(32))) half shB[GATE_TK / 16][16 * GATE_SH_B_STRIDE];
    const int t = threadIdx.x;

    WmmaFragA fa; WmmaFragB fb; WmmaFragC fc;
    fc.clear();
    for (int kt = 0; kt < ktiles; kt++)
    {
        for (int i = t; i < 16 * GATE_TK; i += 32)
        {
            int r = i / GATE_TK, c = i % GATE_TK;
            shA[r * GATE_SH_A_STRIDE + c] = A[r * sa + kt * GATE_TK + c];
        }
        for (int i = t; i < GATE_TK * 16; i += 32)
        {
            int k = i / 16, n = i % 16;
            shB[k / 16][(k % 16) * GATE_SH_B_STRIDE + n] = B[(kt * GATE_TK + k) * sb + n];
        }
        __syncthreads();
        for (int sub_k = 0; sub_k < GATE_TK / 16; sub_k++)
        {
            rdna_wmma::load_matrix_a(fa, shA + sub_k * 16, GATE_SH_A_STRIDE);
            rdna_wmma::load_matrix_b(fb, shB[sub_k], GATE_SH_B_STRIDE);
            rdna_wmma::mma_sync(fc, fa, fb);
        }
        __syncthreads();
    }
    rdna_wmma::store_matrix_c(C, fc, sc);
}

// 8 waves (256 threads, the production sub_k group size). Wave w computes
// row block w x column block (3w mod 8) of a [128, K] x [K, 128] product, so
// every wave's helpers see threadIdx.x > 31 and a distinct tile.
__global__ void k_f32_multiwave(const half* A, int sa, const half* B, int sb, int ksteps,
                                float* C, int sc)
{
    const int w = threadIdx.x / 32;
    const int cb = (w * 3) % 8;
    const half* Aw = A + w * 16 * sa;
    const half* Bw = B + cb * 16;
    float* Cw = C + w * 16 * sc + cb * 16;

    WmmaFragA fa; WmmaFragB fb; WmmaFragC fc;
    fc.clear();
    for (int s = 0; s < ksteps; s++)
    {
        rdna_wmma::load_matrix_a(fa, Aw + s * 16, sa);
        rdna_wmma::load_matrix_b(fb, Bw + s * 16 * sb, sb);
        rdna_wmma::mma_sync(fc, fa, fb);
    }
    rdna_wmma::store_matrix_c(Cw, fc, sc);
}

// Store variants and bounds-checked helpers, K = 16.
//   outFull  : fp32, unchecked -- shows which rows load_accumulate_c_checked touched
//   outChk   : fp32, store_matrix_c_checked(vr, vc)
//   outH     : fp16, store_matrix_c_half
//   outHChk  : fp16, store_matrix_c_half_checked(vr, vc)
__global__ void k_f32_stores(const half* A, int sa, const half* B, int sb,
                             const float* prior, int sp, int vr, int vc,
                             float* outFull, float* outChk, half* outH, half* outHChk, int sc)
{
    WmmaFragA fa; WmmaFragB fb; WmmaFragC fc;
    fc.clear();
    rdna_wmma::load_accumulate_c_checked(fc, prior, sp, vr);
    rdna_wmma::load_matrix_a(fa, A, sa);
    rdna_wmma::load_matrix_b(fb, B, sb);
    rdna_wmma::mma_sync(fc, fa, fb);
    rdna_wmma::store_matrix_c(outFull, fc, sc);
    rdna_wmma::store_matrix_c_checked(outChk, fc, sc, vr, vc);
    rdna_wmma::store_matrix_c_half(outH, fc, sc);
    rdna_wmma::store_matrix_c_half_checked(outHChk, fc, sc, vr, vc);
}

// fp16 prior through load_accumulate_c_half and its checked sibling.
__global__ void k_f32_half_prior(const half* A, int sa, const half* B, int sb,
                                 const half* priorH, int sp, int vr,
                                 float* out, float* outChk, int sc)
{
    WmmaFragA fa; WmmaFragB fb; WmmaFragC fc, fc2;
    fc.clear(); fc2.clear();
    rdna_wmma::load_accumulate_c_half(fc, priorH, sp);
    rdna_wmma::load_accumulate_c_half_checked(fc2, priorH, sp, vr);
    rdna_wmma::load_matrix_a(fa, A, sa);
    rdna_wmma::load_matrix_b(fb, B, sb);
    rdna_wmma::mma_sync(fc, fa, fb);
    rdna_wmma::mma_sync(fc2, fa, fb);
    rdna_wmma::store_matrix_c(out, fc, sc);
    rdna_wmma::store_matrix_c(outChk, fc2, sc);
}

// C-layout round trip: C0 placed by load_accumulate_c(_half), mma with A = 0
// (loaded by load_matrix_a from a zero buffer), stored. Output must equal C0:
// the hardware's C-in and D-out layouts agree and the helpers' mapping is a
// bijection on the tile.
__global__ void k_c_identity(const half* Z, int sa, const half* B, int sb,
                             const float* C0, const half* C0h, int sp,
                             float* out, half* outH, int sc)
{
    WmmaFragA fa; WmmaFragB fb; WmmaFragC fc, fc2;
    rdna_wmma::load_matrix_a(fa, Z, sa);
    rdna_wmma::load_matrix_b(fb, B, sb);
    fc.clear();
    rdna_wmma::load_accumulate_c(fc, C0, sp);
    rdna_wmma::mma_sync(fc, fa, fb);
    rdna_wmma::store_matrix_c(out, fc, sc);
    fc2.clear();
    rdna_wmma::load_accumulate_c_half(fc2, C0h, sp);
    rdna_wmma::mma_sync(fc2, fa, fb);
    rdna_wmma::store_matrix_c_half(outH, fc2, sc);
}

// fp16 accumulate, chained. Also stores the OTHER opsel half, which must still
// be the zeros from clear().
template <bool OPSEL>
__global__ void k_f16acc_chain(const half* A, int sa, const half* B, int sb, int ksteps,
                               half* C, half* Cother, int sc)
{
#ifndef GATE_NO_GFX11_WMMA
    WmmaFragA fa; WmmaFragB fb; WmmaFragC_f16 fc;
    fc.clear();
    for (int s = 0; s < ksteps; s++)
    {
        rdna_wmma::load_matrix_a(fa, A + s * 16, sa);
        rdna_wmma::load_matrix_b(fb, B + s * 16 * sb, sb);
        rdna_wmma::mma_sync_f16<OPSEL>(fc, fa, fb);
    }
    rdna_wmma::store_matrix_c_f16<OPSEL>(C, fc, sc);
    rdna_wmma::store_matrix_c_f16<!OPSEL>(Cother, fc, sc);
#else
    __builtin_trap();
#endif
}

// Dual accumulator in one fragment, plus clear_half:
//   opsel=1 <- A1*B1 ; opsel=0 <- A*B (2 chained K steps)
//   store both          -> o0a (=A*B), o1a (=A1*B1)
//   clear_half<0>; opsel=0 <- A2*B2
//   store both          -> o0b (=A2*B2), o1b (=A1*B1, untouched)
__global__ void k_f16acc_dual(const half* A, const half* B, const half* A1, const half* B1,
                              const half* A2, const half* B2, int sa, int sb,
                              half* o0a, half* o1a, half* o0b, half* o1b, int sc)
{
#ifndef GATE_NO_GFX11_WMMA
    WmmaFragA fa; WmmaFragB fb; WmmaFragC_f16 fc;
    fc.clear();
    rdna_wmma::load_matrix_a(fa, A1, sa);
    rdna_wmma::load_matrix_b(fb, B1, sb);
    rdna_wmma::mma_sync_f16<true>(fc, fa, fb);
    for (int s = 0; s < 2; s++)
    {
        rdna_wmma::load_matrix_a(fa, A + s * 16, sa);
        rdna_wmma::load_matrix_b(fb, B + s * 16 * sb, sb);
        rdna_wmma::mma_sync_f16<false>(fc, fa, fb);
    }
    rdna_wmma::store_matrix_c_f16<false>(o0a, fc, sc);
    rdna_wmma::store_matrix_c_f16<true >(o1a, fc, sc);
    fc.clear_half<false>();
    rdna_wmma::load_matrix_a(fa, A2, sa);
    rdna_wmma::load_matrix_b(fb, B2, sb);
    rdna_wmma::mma_sync_f16<false>(fc, fa, fb);
    rdna_wmma::store_matrix_c_f16<false>(o0b, fc, sc);
    rdna_wmma::store_matrix_c_f16<true >(o1b, fc, sc);
#else
    __builtin_trap();
#endif
}

__global__ void k_f16acc_checked(const half* A, int sa, const half* B, int sb,
                                 half* out, int sc, int vr, int vc)
{
#ifndef GATE_NO_GFX11_WMMA
    WmmaFragA fa; WmmaFragB fb; WmmaFragC_f16 fc;
    fc.clear();
    rdna_wmma::load_matrix_a(fa, A, sa);
    rdna_wmma::load_matrix_b(fb, B, sb);
    rdna_wmma::mma_sync_f16<true>(fc, fa, fb);
    rdna_wmma::store_matrix_c_f16_checked<true>(out, fc, sc, vr, vc);
#else
    __builtin_trap();
#endif
}

__global__ void k_bf16_chain(const __hip_bfloat16* A, int sa, const __hip_bfloat16* B, int sb,
                             int ksteps, float* C, int sc)
{
#ifndef GATE_NO_GFX11_WMMA
    WmmaFragA_bf16 fa; WmmaFragB_bf16 fb; WmmaFragC fc;
    fc.clear();
    for (int s = 0; s < ksteps; s++)
    {
        rdna_wmma::load_matrix_a_bf16(fa, A + s * 16, sa);
        rdna_wmma::load_matrix_b_bf16(fb, B + s * 16 * sb, sb);
        rdna_wmma::mma_sync_bf16(fc, fa, fb);
    }
    rdna_wmma::store_matrix_c(C, fc, sc);
#else
    __builtin_trap();
#endif
}

template <bool SA, bool SB>
__global__ void k_i8_chain(const int8_t* A, int sa, const int8_t* B, int sb, int ksteps,
                           int* C, int sc)
{
#ifndef GATE_NO_GFX11_WMMA
    WmmaFragA_i8 fa; WmmaFragB_i8 fb; WmmaFragC_i32 fc;
    fc.clear();
    for (int s = 0; s < ksteps; s++)
    {
        rdna_wmma::load_matrix_a_i8(fa, A + s * 16, sa);
        rdna_wmma::load_matrix_b_i8(fb, B + s * 16 * sb, sb);
        rdna_wmma::mma_sync_i8<SA, SB>(fc, fa, fb);
    }
    rdna_wmma::store_matrix_c_i32(C, fc, sc);
#else
    __builtin_trap();
#endif
}

__global__ void k_i8_checked(const int8_t* A, int sa, const int8_t* B, int sb,
                             int* C, int sc, int vr, int vc)
{
#ifndef GATE_NO_GFX11_WMMA
    WmmaFragA_i8 fa; WmmaFragB_i8 fb; WmmaFragC_i32 fc;
    fc.clear();
    rdna_wmma::load_matrix_a_i8(fa, A, sa);
    rdna_wmma::load_matrix_b_i8(fb, B, sb);
    rdna_wmma::mma_sync_i8<false, true>(fc, fa, fb);
    rdna_wmma::store_matrix_c_i32_checked(C, fc, sc, vr, vc);
#else
    __builtin_trap();
#endif
}

// INFO ONLY. Lanes 16-31 load operand data that differs from lanes 0-15
// (mode 0: A, mode 1: B). The header always replicates (Rule 2.1); what the
// hardware does otherwise is unknown. The helper call is uniform; only the
// base pointer differs per lane half.
__global__ void k_info_halfdiverge(const half* A, const half* A2, int sa,
                                   const half* B, const half* B2, int sb,
                                   int mode, float* C, int sc)
{
    const bool hi = (threadIdx.x & 31) >= 16;
    WmmaFragA fa; WmmaFragB fb; WmmaFragC fc;
    fc.clear();
    rdna_wmma::load_matrix_a(fa, (mode == 0 && hi) ? A2 : A, sa);
    rdna_wmma::load_matrix_b(fb, (mode == 1 && hi) ? B2 : B, sb);
    rdna_wmma::mma_sync(fc, fa, fb);
    rdna_wmma::store_matrix_c(C, fc, sc);
}

// INFO ONLY. int8 clamp=true, never exercised in production. Accumulator is
// seeded near INT_MAX (via operator[], there is no i32 load_accumulate helper)
// and the product pushes it past; records clamp and no-clamp results.
__global__ void k_info_i8_clamp(const int8_t* A, int sa, const int8_t* B, int sb, int seed,
                                int* Cc, int* Cn, int sc)
{
#ifndef GATE_NO_GFX11_WMMA
    WmmaFragA_i8 fa; WmmaFragB_i8 fb; WmmaFragC_i32 fc, fn;
    for (int i = 0; i < 8; i++) { fc[i] = seed; fn[i] = seed; }
    rdna_wmma::load_matrix_a_i8(fa, A, sa);
    rdna_wmma::load_matrix_b_i8(fb, B, sb);
    rdna_wmma::mma_sync_i8<false, false, true >(fc, fa, fb);
    rdna_wmma::mma_sync_i8<false, false, false>(fn, fa, fb);
    rdna_wmma::store_matrix_c_i32(Cc, fc, sc);
    rdna_wmma::store_matrix_c_i32(Cn, fn, sc);
#else
    __builtin_trap();
#endif
}

#endif // !WMMA_GATE_DESIGN_CHECK

// =============================================================================
// Host: number formats
// =============================================================================

static int g_design_errors = 0;
static void design_error(const char* where, const char* what)
{
    printf("  TEST-DESIGN ERROR in %s: %s\n", where, what);
    g_design_errors++;
}

static bool f16_exact(double v)
{
    if (v == 0.0) return true;
    double a = fabs(v);
    if (!(a >= 6.103515625e-05 && a <= 65504.0)) return false;   // normals only
    int e; double m = frexp(a, &e);                                // a = m * 2^e, m in [0.5, 1)
    double scaled = ldexp(m, 11);                                  // 11 significant bits
    return scaled == floor(scaled);
}
static uint16_t f16_enc(double v)                                  // v must be f16_exact
{
    if (v == 0.0) return 0;
    uint16_t s = v < 0 ? 0x8000 : 0;
    double a = fabs(v);
    int e; double m = frexp(a, &e);                                // a = (2m) * 2^(e-1)
    int E = e - 1 + 15;
    int mant = (int) ldexp(2.0 * m - 1.0, 10);
    return (uint16_t)(s | (E << 10) | mant);
}
static double f16_dec(uint16_t h)
{
    int s = h >> 15, E = (h >> 10) & 31, m = h & 1023;
    double v;
    if (E == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    if (E == 0) v = ldexp((double) m, -24);
    else        v = ldexp(1.0 + m / 1024.0, E - 15);
    return s ? -v : v;
}
static bool f32_exact(double v) { return (double)(float) v == v; }
static bool bf16_exact(double v)
{
    if (!f32_exact(v)) return false;
    float f = (float) v; uint32_t u; memcpy(&u, &f, 4);
    return (u & 0xffffu) == 0;
}
static uint16_t bf16_enc(double v)
{
    float f = (float) v; uint32_t u; memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}
// ulp of |x| in fp32 / fp16 (normal range; 0 for x == 0 in fp32).
static double ulp32(double x)
{
    x = fabs(x); if (x == 0.0) return 0.0;
    int e; frexp(x, &e); return ldexp(1.0, e - 1 - 23);
}
static double ulp16(double x)
{
    x = fabs(x); if (x < ldexp(1.0, -14)) return ldexp(1.0, -24);
    int e; frexp(x, &e); return ldexp(1.0, e - 1 - 10);
}

static const uint16_t PAD_F16  = 0x7e00;       // fp16 qNaN in A/B padding
static const uint16_t PAD_BF16 = 0x7fc0;       // bf16 qNaN
static const uint8_t  PAD_I8   = 0x77;
static const uint32_t SENT_F32 = 0x7fa5a5a5u;  // NaN with payload
static const uint16_t SENT_F16 = 0x7d5a;       // NaN with payload
static const uint32_t SENT_I32 = 0x5a5a5a5au;

// =============================================================================
// Host: matrices, references, bounds
// =============================================================================

struct Mat   // dense logical matrix, row-major, values in double
{
    int rows, cols;
    std::vector<double> v;
    Mat(int r = 0, int c = 0) : rows(r), cols(c), v((size_t) r * c, 0.0) {}
    double& at(int r, int c) { return v[(size_t) r * cols + c]; }
    double  at(int r, int c) const { return v[(size_t) r * cols + c]; }
};

// Deterministic, non-symmetric small integers in [-(span/2), span/2].
static double patv(int i, int j, int seed, int span)
{
    int x = (i * (3 + 2 * seed) + j * (7 + seed) + ((i * j + seed) % 5) + seed) % span;
    return (double)(x - span / 2);
}

template <class F> static Mat make(int r, int c, F f)
{
    Mat m(r, c);
    for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) m.at(i, j) = f(i, j);
    return m;
}

enum Acc { ACC_EXACT, ACC_F32, ACC_F16 };

// Reference result plus per-cell error bound and intermediate magnitude.
struct Ref
{
    Mat C, tol, mag;
};

// C = prior + A*B in double (exact: inputs are small dyadic numbers).
// Per 16-deep K step s the hardware bound is 16 * ulp_f32(M_s), M_s = max of
// |C in|, |each product|, |each exact partial sum| (in index order) and |C out|;
// fp16 accumulate adds ulp_f16(C out) per step (the C fragment is fp16).
// ACC_EXACT (int8) has zero tolerance. The granularity check (every term a
// multiple of g, sum|terms|/g < 2^acc_bits) keeps the reference exact and every
// expected value representable in the output type.
static Ref ref_gemm(const char* where, const Mat& A, const Mat& B, const Mat* prior,
                    double g, int acc_bits, Acc acc)
{
    Ref R{ Mat(A.rows, B.cols), Mat(A.rows, B.cols), Mat(A.rows, B.cols) };
    const double lim = ldexp(1.0, acc_bits);
    bool ok = true;
    if (A.cols % 16) design_error(where, "K not a multiple of 16");
    for (int i = 0; i < A.rows; i++)
        for (int j = 0; j < B.cols; j++)
        {
            double c = prior ? prior->at(i, j) : 0.0, sabs = fabs(c), tol = 0.0, mag = fabs(c);
            if (fmod(c, g) != 0.0) ok = false;
            for (int k0 = 0; k0 < A.cols; k0 += 16)
            {
                double M = fabs(c);
                for (int k = k0; k < k0 + 16; k++)
                {
                    double t = A.at(i, k) * B.at(k, j);
                    if (fmod(t, g) != 0.0) ok = false;
                    c += t; sabs += fabs(t);
                    M = fmax(M, fmax(fabs(t), fabs(c)));
                }
                if (acc != ACC_EXACT) tol += 16.0 * ulp32(M);
                if (acc == ACC_F16)   tol += ulp16(c);
                mag = fmax(mag, M);
            }
            if (sabs / g >= lim) ok = false;
            R.C.at(i, j) = c; R.tol.at(i, j) = tol; R.mag.at(i, j) = mag;
        }
    if (!ok) design_error(where, "reference not exact / not representable (granularity check)");
    return R;
}

// Exact expectation (no rounding by construction).
static Ref exact_ref(const Mat& C)
{
    Ref R{ C, Mat(C.rows, C.cols), Mat(C.rows, C.cols) };
    for (size_t i = 0; i < C.v.size(); i++) R.mag.v[i] = fabs(C.v[i]);
    return R;
}

// Rows < vr from a, the rest from b.
static Ref mix_rows(const Ref& a, const Ref& b, int vr)
{
    Ref R = b;
    for (int r = 0; r < vr; r++) for (int c = 0; c < a.C.cols; c++)
    {
        R.C.at(r, c) = a.C.at(r, c); R.tol.at(r, c) = a.tol.at(r, c); R.mag.at(r, c) = a.mag.at(r, c);
    }
    return R;
}

// Every 16x16 tile with origin (r0, c0) must differ from its transpose in at
// least 64 cells, or a transposed / operand-swapped result could pass.
static void need_asym(const char* where, const Mat& C, int r0 = 0, int c0 = 0)
{
    int d = 0;
    for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++)
        if (C.at(r0 + i, c0 + j) != C.at(r0 + j, c0 + i)) d++;
    if (d < 64) design_error(where, "reference tile is too close to symmetric");
}

template <class F> static void need_all(const char* where, const Mat& m, F pred, const char* what)
{
    for (double x : m.v) if (!pred(x)) { design_error(where, what); return; }
}

// Pack into a padded row-major buffer; padding gets the poison value.
static std::vector<uint16_t> pack_f16(const Mat& m, int stride)
{
    std::vector<uint16_t> o((size_t) m.rows * stride, PAD_F16);
    for (int i = 0; i < m.rows; i++) for (int j = 0; j < m.cols; j++)
    {
        double x = m.at(i, j);
        uint16_t e = f16_enc(x);
        if (!f16_exact(x) || f16_dec(e) != x) { design_error("pack_f16", "input not fp16-exact"); return o; }
        o[(size_t) i * stride + j] = e;
    }
    return o;
}
static std::vector<uint16_t> pack_bf16(const Mat& m, int stride)
{
    std::vector<uint16_t> o((size_t) m.rows * stride, PAD_BF16);
    for (int i = 0; i < m.rows; i++) for (int j = 0; j < m.cols; j++)
    {
        if (!bf16_exact(m.at(i, j))) { design_error("pack_bf16", "input not bf16-exact"); return o; }
        o[(size_t) i * stride + j] = bf16_enc(m.at(i, j));
    }
    return o;
}
static std::vector<float> pack_f32(const Mat& m, int stride)
{
    float pad; memcpy(&pad, &SENT_F32, 4);
    std::vector<float> o((size_t) m.rows * stride, pad);
    for (int i = 0; i < m.rows; i++) for (int j = 0; j < m.cols; j++)
    {
        if (!f32_exact(m.at(i, j))) { design_error("pack_f32", "input not fp32-exact"); return o; }
        o[(size_t) i * stride + j] = (float) m.at(i, j);
    }
    return o;
}
// int8: m holds the raw byte values 0..255; the sign reading is the mma's job.
static std::vector<uint8_t> pack_u8(const Mat& m, int stride)
{
    std::vector<uint8_t> o((size_t) m.rows * stride, PAD_I8);
    for (int i = 0; i < m.rows; i++) for (int j = 0; j < m.cols; j++)
        o[(size_t) i * stride + j] = (uint8_t) m.at(i, j);
    return o;
}
static Mat bytes_as(const Mat& m, bool is_signed)
{
    Mat o = m;
    for (double& x : o.v) x = is_signed ? (double)(int8_t)(uint8_t) x : x;
    return o;
}

// =============================================================================
// Host: device buffers (no-ops in design-check mode)
// =============================================================================

struct DevBuf
{
    size_t bytes = 0;
    void* d = nullptr;
    std::vector<unsigned char> h;

    template <class T> void upload(const std::vector<T>& v)
    {
        bytes = v.size() * sizeof(T);
        h.assign((const unsigned char*) v.data(), (const unsigned char*) v.data() + bytes);
#ifndef WMMA_GATE_DESIGN_CHECK
        if (hipMalloc(&d, bytes) != hipSuccess ||
            hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice) != hipSuccess)
        { printf("hipMalloc/hipMemcpy failed\nWMMA GATE: FAIL (setup)\n"); exit(2); }
#endif
    }
    template <class T> void fill(size_t n, T sentinel) { upload(std::vector<T>(n, sentinel)); }
    template <class T> std::vector<T> get()
    {
#ifndef WMMA_GATE_DESIGN_CHECK
        if (hipMemcpy(h.data(), d, bytes, hipMemcpyDeviceToHost) != hipSuccess)
        { printf("hipMemcpy D2H failed\nWMMA GATE: FAIL (setup)\n"); exit(2); }
#endif
        std::vector<T> o(bytes / sizeof(T));
        memcpy(o.data(), h.data(), bytes);
        return o;
    }
    template <class T> T* p() const { return (T*) d; }
    ~DevBuf()
    {
#ifndef WMMA_GATE_DESIGN_CHECK
        if (d) (void) hipFree(d);
#endif
    }
};

#ifdef WMMA_GATE_DESIGN_CHECK
#define LAUNCH(nthreads, kern, ...) do {} while (0)
#else
static void sync_or_die(const char* kname)
{
    hipError_t e = hipGetLastError();
    if (e == hipSuccess) e = hipDeviceSynchronize();
    if (e != hipSuccess)
    {
        printf("  kernel %s failed: %s\nWMMA GATE: FAIL (launch)\n", kname, hipGetErrorString(e));
        exit(2);
    }
}
#define LAUNCH(nthreads, kern, ...) do { \
        hipLaunchKernelGGL(kern, dim3(1), dim3(nthreads), 0, 0, __VA_ARGS__); \
        sync_or_die(#kern); } while (0)
#endif

// =============================================================================
// Host: golden file
// =============================================================================
//
// Text, deterministic:
//   # comment lines
//   arch <gcnArchName>
//   case <n_elems> <elem_bytes> <case name to end of line>
//   <hex words, 8 per line>          (n_elems words of 2*elem_bytes hex digits)

struct GoldenCase { int elem_bytes; std::vector<uint32_t> w; };
static std::map<std::string, GoldenCase> g_golden, g_record;
static std::vector<std::string> g_record_order;
static std::string g_golden_arch, g_arch;
static bool g_golden_loaded = false, g_record_mode = false;

[[maybe_unused]] static bool golden_load(const std::string& path)
{
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    char line[512];
    std::string cur; GoldenCase* gc = nullptr; size_t want_n = 0;
    while (fgets(line, sizeof(line), f))
    {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (!strncmp(line, "arch ", 5))
        {
            g_golden_arch = line + 5;
            while (!g_golden_arch.empty() && (g_golden_arch.back() == '\n' || g_golden_arch.back() == ' '))
                g_golden_arch.pop_back();
            continue;
        }
        if (!strncmp(line, "case ", 5))
        {
            int n = 0, eb = 0; char name[400] = {0};
            if (sscanf(line + 5, "%d %d %399[^\n]", &n, &eb, name) != 3) { fclose(f); return false; }
            cur = name; gc = &g_golden[cur]; gc->elem_bytes = eb; gc->w.clear(); want_n = (size_t) n;
            gc->w.reserve(want_n);
            continue;
        }
        if (!gc) { fclose(f); return false; }
        char* p = line;
        while (*p)
        {
            while (*p == ' ') p++;
            if (*p == '\n' || !*p) break;
            char* end; unsigned long v = strtoul(p, &end, 16);
            if (end == p) { fclose(f); return false; }
            gc->w.push_back((uint32_t) v); p = end;
        }
        if (gc->w.size() > want_n) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

static bool golden_write(const std::string& path)
{
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;
    fprintf(f, "# wmma_gate golden: raw output bits of every gate case (full buffer incl. guard cells).\n");
    fprintf(f, "# Written by `wmma_gate --record`. Re-record only deliberately: after a reviewed\n");
    fprintf(f, "# toolchain / header change, or after changing the gate's inputs.\n");
    fprintf(f, "arch %s\n", g_arch.c_str());
    for (const std::string& name : g_record_order)
    {
        const GoldenCase& gc = g_record[name];
        fprintf(f, "case %zu %d %s\n", gc.w.size(), gc.elem_bytes, name.c_str());
        for (size_t i = 0; i < gc.w.size(); i++)
        {
            if (gc.elem_bytes == 2) fprintf(f, "%04x", gc.w[i]); else fprintf(f, "%08x", gc.w[i]);
            fputc((i % 8 == 7 || i + 1 == gc.w.size()) ? '\n' : ' ', f);
        }
    }
    fclose(f);
    return true;
}

// =============================================================================
// Host: expected-output description and checks
// =============================================================================

enum OutT { O_F32, O_F16, O_I32 };

struct Want   // over the whole output buffer: written cells have a value, the rest the sentinel
{
    OutT t; int rows, stride;
    std::vector<double> v, tol, mag;
    std::vector<char> w;
    Want(OutT t_, int rows_, int stride_)
        : t(t_), rows(rows_), stride(stride_), v((size_t) rows_ * stride_, 0.0),
          tol(v.size(), 0.0), mag(v.size(), 0.0), w(v.size(), 0) {}
    void set(int r, int c, double x, double tl = 0.0, double mg = 0.0)
    {
        size_t i = (size_t) r * stride + c;
        v[i] = x; tol[i] = tl; mag[i] = mg; w[i] = 1;
    }
    // f16_store: the fp32 result is rounded to fp16 on store -> + ulp_f16(want).
    void tile(const Ref& R, bool f16_store = false, int r0 = 0, int c0 = 0,
              int vr = 1 << 30, int vc = 1 << 30)
    {
        for (int i = 0; i < R.C.rows; i++) for (int j = 0; j < R.C.cols; j++)
            if (i < vr && j < vc)
            {
                double x = R.C.at(i, j), tl = R.tol.at(i, j);
                if (f16_store && tl > 0.0) tl += ulp16(x);
                set(r0 + i, c0 + j, x, tl, R.mag.at(i, j));
            }
    }
    size_t cells() const { return v.size(); }
};

static int g_gate_total = 0, g_gate_fail = 0, g_math_fail = 0, g_golden_fail = 0;
static bool g_design = false;

static void read_out(DevBuf& b, OutT t, std::vector<double>& val, std::vector<uint32_t>& bits)
{
    size_t n;
    if (t == O_F32)
    {
        auto x = b.get<uint32_t>(); n = x.size(); val.resize(n); bits = x;
        for (size_t i = 0; i < n; i++) { float f; memcpy(&f, &x[i], 4); val[i] = f; }
    }
    else if (t == O_F16)
    {
        auto x = b.get<uint16_t>(); n = x.size(); val.resize(n); bits.resize(n);
        for (size_t i = 0; i < n; i++) { val[i] = f16_dec(x[i]); bits[i] = x[i]; }
    }
    else
    {
        auto x = b.get<int32_t>(); n = x.size(); val.resize(n); bits.resize(n);
        for (size_t i = 0; i < n; i++) { val[i] = x[i]; bits[i] = (uint32_t) x[i]; }
    }
}

// One gate case: MATH check (exact or bounded) + GOLDEN check (bitwise).
// tdiag: on a math failure, report how many cells of the 16x16 tile at (0, 0)
// match the TRANSPOSED expectation.
static void gate(const char* name, DevBuf& out, const Want& want, bool tdiag = true)
{
    g_gate_total++;
    for (size_t i = 0; i < want.cells(); i++)
    {
        if (!want.w[i]) continue;
        double x = want.v[i];
        bool ok = want.t == O_F32 ? f32_exact(x)
                : want.t == O_F16 ? f16_exact(x)
                : (x == floor(x) && fabs(x) < 2147483648.0);
        if (!ok) { design_error(name, "expected value not exactly representable in output type"); break; }
    }
    static std::set<std::string> seen;
    if (!seen.insert(name).second) design_error(name, "duplicate case name");
    if (g_design) { printf("  %-54s design ok\n", name); return; }

    std::vector<double> got; std::vector<uint32_t> bits;
    read_out(out, want.t, got, bits);
    const uint32_t sent = want.t == O_F32 ? SENT_F32 : want.t == O_F16 ? SENT_F16 : SENT_I32;

    // ---- math
    int bad = 0, clobber = 0, written = 0, inexact = 0, shown = 0;
    bool exact_case = true;
    double max_ulp = 0.0;
    for (size_t i = 0; i < want.cells(); i++)
    {
        if (want.w[i])
        {
            written++;
            if (want.tol[i] > 0.0) exact_case = false;
            double e = fabs(got[i] - want.v[i]);
            if (!(e == 0.0)) inexact++;
            if (!(e <= want.tol[i]))   // NaN-safe
            {
                bad++;
                if (shown++ < 4)
                    printf("      mismatch r=%d c=%d got=%.9g want=%.9g bound=%.3g\n",
                           (int)(i / want.stride), (int)(i % want.stride), got[i], want.v[i], want.tol[i]);
            }
            else if (e > 0.0)
            {
                double u = ulp32(want.mag[i]);
                max_ulp = fmax(max_ulp, u > 0.0 ? e / u : INFINITY);
            }
        }
        else if (bits[i] != sent)
        {
            clobber++;
            if (shown++ < 4)
                printf("      clobbered r=%d c=%d (must stay sentinel) got=%.9g\n",
                       (int)(i / want.stride), (int)(i % want.stride), got[i]);
        }
    }
    bool math_ok = bad == 0 && clobber == 0;
    if (!math_ok && tdiag)
    {
        int tr = 0, cand = 0;
        for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++)
        {
            size_t a = (size_t) i * want.stride + j, b = (size_t) j * want.stride + i;
            if (i == j || !want.w[a] || !want.w[b]) continue;
            if (fabs(want.v[a] - want.v[b]) <= want.tol[a]) continue;   // indistinguishable cell
            cand++;
            if (fabs(got[a] - want.v[b]) <= want.tol[a]) tr++;
        }
        printf("      diagnostic: %d/%d distinguishable cells match the TRANSPOSED expectation%s\n",
               tr, cand, (cand && tr * 2 > cand) ? "  <-- operand order / C layout swapped" : "");
    }

    // ---- golden
    const int eb = want.t == O_F16 ? 2 : 4;
    std::string gstat;
    bool golden_ok = true;
    if (g_record_mode)
    {
        g_record[name] = GoldenCase{ eb, bits };
        g_record_order.push_back(name);
        gstat = "recorded";
    }
    else if (!g_golden_loaded)            { golden_ok = false; gstat = "NO GOLDEN FILE"; }
    else if (g_golden_arch != g_arch)     { golden_ok = false; gstat = "ARCH MISMATCH"; }
    else
    {
        auto it = g_golden.find(name);
        if (it == g_golden.end())                                    { golden_ok = false; gstat = "MISSING"; }
        else if (it->second.elem_bytes != eb || it->second.w.size() != bits.size())
                                                                      { golden_ok = false; gstat = "SHAPE DIFFERS"; }
        else
        {
            int nd = 0;
            for (size_t i = 0; i < bits.size(); i++) nd += bits[i] != it->second.w[i];
            golden_ok = nd == 0;
            char b[64]; snprintf(b, sizeof(b), nd ? "%d words DIFFER" : "bit-exact", nd);
            gstat = b;
        }
    }

    bool pass = math_ok && golden_ok;
    char mdesc[96];
    if (exact_case) snprintf(mdesc, sizeof(mdesc), "exact %d/%d bad", bad, written);
    else            snprintf(mdesc, sizeof(mdesc), "%d/%d bad, %d inexact, max %.2f ulp", bad, written, inexact, max_ulp);
    printf("  %-54s %s  math[%s%s] golden[%s]\n", name, pass ? "PASS" : "FAIL", mdesc,
           clobber ? ", GUARD CLOBBERED" : "", gstat.c_str());
    if (!math_ok) g_math_fail++;
    if (!golden_ok) g_golden_fail++;
    if (!pass) g_gate_fail++;
}

// =============================================================================
// main
// =============================================================================

static double p2(int e) { return ldexp(1.0, e); }

int main(int argc, char** argv)
{
    // Golden path: next to this source file unless overridden.
    std::string golden_path;
    {
        std::string src = __FILE__;
        size_t sl = src.find_last_of('/');
        golden_path = (sl == std::string::npos ? std::string(".") : src.substr(0, sl)) + "/wmma_gate.golden";
    }
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--record")) g_record_mode = true;
        else if (!strcmp(argv[i], "--golden") && i + 1 < argc) golden_path = argv[++i];
        else { printf("usage: %s [--record] [--golden PATH]\n", argv[0]); return 2; }
    }

#ifdef WMMA_GATE_DESIGN_CHECK
    g_design = true;
    printf("rdna_wmma.hip.h gate -- DESIGN CHECK (host only, no GPU)\n");
#else
    {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, 0) != hipSuccess)
        { printf("no HIP device\nWMMA GATE: FAIL (setup)\n"); return 2; }
        g_arch = prop.gcnArchName;
        printf("rdna_wmma.hip.h gate on %s (%s)\n", prop.gcnArchName, prop.name);
        if (strncmp(prop.gcnArchName, "gfx11", 5) != 0)
        {
            printf("  device is not gfx11: the header has no WMMA implementation for it "
                   "(gfx12: mma_sync traps, other variants are compiled out here)\n"
                   "WMMA GATE: FAIL (unsupported arch)\n");
            return 1;
        }
        if (g_record_mode)
            printf("mode: RECORD -> %s\n", golden_path.c_str());
        else
        {
            g_golden_loaded = golden_load(golden_path);
            printf("mode: compare against %s (%s%s%s)\n", golden_path.c_str(),
                   g_golden_loaded ? "loaded, recorded on " : "NOT FOUND -- run with --record",
                   g_golden_loaded ? g_golden_arch.c_str() : "", "");
        }
    }
#endif

    // ------------------------------------------------------------------
    // Problem P: f32 accumulate, K = 64, power-of-two-scaled integers.
    //   A[i][k] in [-4,4] * {1/4,1/2,1,2} by row, B[k][j] in [-4,4] * {1/2,1,2}
    //   by column. Padded strides: A 64+8 (TILESIZE_K+8 analogue), B 16+8, C 16+4.
    // ------------------------------------------------------------------
    const int PK = 64, PSA = PK + 8, PSB = 16 + 8, PSC = 16 + 4;
    Mat PA = make(16, PK, [](int i, int k) { return patv(i, k, 0, 9) * p2((i % 4) - 2); });
    Mat PB = make(PK, 16, [](int k, int j) { return patv(k, j, 1, 9) * p2((j % 3) - 1); });
    Mat Pprior = make(16, 16, [](int r, int c) { return patv(r, c, 5, 31) * 0.5; });
    Ref PC  = ref_gemm("P", PA, PB, nullptr, 0.125, 24, ACC_F32);
    Ref PCp = ref_gemm("P+prior", PA, PB, &Pprior, 0.125, 24, ACC_F32);
    need_asym("P", PC.C); need_asym("P+prior", PCp.C);

    DevBuf dPA, dPB, dPprior;
    dPA.upload(pack_f16(PA, PSA));
    dPB.upload(pack_f16(PB, PSB));
    dPprior.upload(pack_f32(Pprior, PSC));

    printf("f32 accumulate (mma_sync):\n");
    {
        DevBuf out; out.fill<uint32_t>(16 * PSC, SENT_F32);
        LAUNCH(32, k_f32_chain, dPA.p<half>(), PSA, dPB.p<half>(), PSB, PK / 16,
               (const float*) nullptr, 0, out.p<float>(), PSC);
        Want w(O_F32, 16, PSC); w.tile(PC);
        gate("f32 K=64 chained x4, padded A/B/C strides", out, w);
    }
    {
        DevBuf out; out.fill<uint32_t>(16 * PSC, SENT_F32);
        LAUNCH(32, k_f32_chain, dPA.p<half>(), PSA, dPB.p<half>(), PSB, PK / 16,
               dPprior.p<float>(), PSC, out.p<float>(), PSC);
        Want w(O_F32, 16, PSC); w.tile(PCp);
        gate("f32 load_accumulate_c prior + K=64 chain", out, w);
    }
    {
        DevBuf out; out.fill<uint32_t>(16 * PSC, SENT_F32);
        LAUNCH(32, k_f32_lds, dPA.p<half>(), PSA, dPB.p<half>(), PSB, PK / GATE_TK,
               out.p<float>(), PSC);
        Want w(O_F32, 16, PSC); w.tile(PC);
        gate("f32 LDS geometry: A stride TK+8, B stride 18, sub_k", out, w);
    }

    // ------------------------------------------------------------------
    // Multi-wave: 256 threads, [128 x 32] x [32 x 128], wave w -> tile (w, 3w%8).
    // ------------------------------------------------------------------
    {
        const int MK = 32, MSA = MK + 8, MSB = 128 + 8, MSC = 128 + 4;
        Mat MA = make(128, MK, [](int i, int k) { return patv(i, k, 2, 9); });
        Mat MB = make(MK, 128, [](int k, int j) { return patv(k, j, 3, 7); });
        Ref MC = ref_gemm("multiwave", MA, MB, nullptr, 1.0, 24, ACC_F32);
        DevBuf dA, dB, out;
        dA.upload(pack_f16(MA, MSA));
        dB.upload(pack_f16(MB, MSB));
        out.fill<uint32_t>((size_t) 128 * MSC, SENT_F32);
        LAUNCH(256, k_f32_multiwave, dA.p<half>(), MSA, dB.p<half>(), MSB, MK / 16,
               out.p<float>(), MSC);
        Want w(O_F32, 128, MSC);
        for (int wv = 0; wv < 8; wv++)
        {
            int cb = (wv * 3) % 8;
            need_asym("multiwave", MC.C, wv * 16, cb * 16);
            for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++)
            {
                int r = wv * 16 + i, c = cb * 16 + j;
                w.set(r, c, MC.C.at(r, c), MC.tol.at(r, c), MC.mag.at(r, c));
            }
        }
        gate("f32 8 waves x distinct tiles (threadIdx.x > 31)", out, w, false);
    }

    // ------------------------------------------------------------------
    // Problem S: small integers, K = 16 (every result fp16-representable).
    //   A, B in [-4,4]; prior in [-50,50]; |prior + A*B| <= 306.
    //   Strides: A 16+8, B 18 (the B_lds stride), C/prior 16+4.
    // ------------------------------------------------------------------
    const int SSA = 16 + 8, SSB = 18, SSC = 16 + 4, VR = 11, VC = 7;
    Mat SA_ = make(16, 16, [](int i, int k) { return patv(i, k, 4, 9); });
    Mat SB_ = make(16, 16, [](int k, int j) { return patv(k, j, 6, 9); });
    Mat Sprior = make(16, 16, [](int r, int c) { return patv(r, c, 7, 101); });
    Ref SC  = ref_gemm("S", SA_, SB_, nullptr, 1.0, 11, ACC_F32);
    Ref SCp = ref_gemm("S+prior", SA_, SB_, &Sprior, 1.0, 11, ACC_F32);
    need_asym("S", SC.C); need_asym("S+prior", SCp.C);
    Ref Smixed = mix_rows(SCp, SC, VR);   // load_accumulate_*_checked: prior only on rows < VR
    DevBuf dSA, dSB, dSprior, dSpriorH;
    dSA.upload(pack_f16(SA_, SSA));
    dSB.upload(pack_f16(SB_, SSB));
    dSprior.upload(pack_f32(Sprior, SSC));
    dSpriorH.upload(pack_f16(Sprior, SSC));

    printf("store / accumulate helpers (K=16, vr=%d vc=%d):\n", VR, VC);
    {
        DevBuf oFull, oChk, oH, oHChk;
        oFull.fill<uint32_t>(16 * SSC, SENT_F32);
        oChk .fill<uint32_t>(16 * SSC, SENT_F32);
        oH   .fill<uint16_t>(16 * SSC, SENT_F16);
        oHChk.fill<uint16_t>(16 * SSC, SENT_F16);
        LAUNCH(32, k_f32_stores, dSA.p<half>(), SSA, dSB.p<half>(), SSB,
               dSprior.p<float>(), SSC, VR, VC,
               oFull.p<float>(), oChk.p<float>(), oH.p<half>(), oHChk.p<half>(), SSC);
        Want wFull(O_F32, 16, SSC); wFull.tile(Smixed);
        gate("load_accumulate_c_checked rows<vr only", oFull, wFull);
        Want wChk(O_F32, 16, SSC); wChk.tile(Smixed, false, 0, 0, VR, VC);
        gate("store_matrix_c_checked (in range, guard intact)", oChk, wChk, false);
        Want wH(O_F16, 16, SSC); wH.tile(Smixed, true);
        gate("store_matrix_c_half", oH, wH);
        Want wHChk(O_F16, 16, SSC); wHChk.tile(Smixed, true, 0, 0, VR, VC);
        gate("store_matrix_c_half_checked", oHChk, wHChk, false);
    }
    {
        DevBuf o, oChk;
        o   .fill<uint32_t>(16 * SSC, SENT_F32);
        oChk.fill<uint32_t>(16 * SSC, SENT_F32);
        LAUNCH(32, k_f32_half_prior, dSA.p<half>(), SSA, dSB.p<half>(), SSB,
               dSpriorH.p<half>(), SSC, VR, o.p<float>(), oChk.p<float>(), SSC);
        Want w(O_F32, 16, SSC); w.tile(SCp);
        gate("load_accumulate_c_half prior + mma", o, w);
        Want w2(O_F32, 16, SSC); w2.tile(Smixed);
        gate("load_accumulate_c_half_checked rows<vr only", oChk, w2);
    }

    // ------------------------------------------------------------------
    // C-layout round trip: C0 -> load_accumulate -> mma(A=0) -> store == C0.
    // Exact by construction (every product is +-0; measured bit-exact).
    // ------------------------------------------------------------------
    printf("C-layout round trip (A = 0):\n");
    {
        Mat Z(16, 16);
        Mat C0  = make(16, 16, [](int r, int c) { return (r % 2 ? -1.0 : 1.0) * (r * 1000 + c * 3 + 0.5); });
        Mat C0h = make(16, 16, [](int r, int c) { return (r % 2 ? -1.0 : 1.0) * (r * 64 + c * 2 + 1); });
        need_asym("identity", C0); need_asym("identity half", C0h);
        need_all("identity half", C0h, f16_exact, "C0h not fp16-exact");
        need_all("identity", C0, f32_exact, "C0 not fp32-exact");
        DevBuf dZ, dC0, dC0h, o, oH;
        dZ.upload(pack_f16(Z, SSA));
        dC0.upload(pack_f32(C0, SSC));
        dC0h.upload(pack_f16(C0h, SSC));
        o .fill<uint32_t>(16 * SSC, SENT_F32);
        oH.fill<uint16_t>(16 * SSC, SENT_F16);
        LAUNCH(32, k_c_identity, dZ.p<half>(), SSA, dSB.p<half>(), SSB,
               dC0.p<float>(), dC0h.p<half>(), SSC, o.p<float>(), oH.p<half>(), SSC);
        Want w(O_F32, 16, SSC); w.tile(exact_ref(C0));
        gate("f32 C identity: load_accumulate_c/mma/store_c", o, w);
        Want wh(O_F16, 16, SSC); wh.tile(exact_ref(C0h));
        gate("f16 C identity: load_accumulate_c_half/mma/store_c_half", oH, wh);
    }

    // ------------------------------------------------------------------
    // fp16 accumulate (v_wmma_f16_16x16x16_f16), K = 32 chained x2.
    // ------------------------------------------------------------------
    printf("fp16 accumulate (mma_sync_f16):\n");
    {
        const int FK = 32, FSA = FK + 8, FSB = 16 + 8, FSC = 16 + 4;
        Mat FA  = make(16, FK, [](int i, int k) { return patv(i, k, 8, 7); });
        Mat FB  = make(FK, 16, [](int k, int j) { return patv(k, j, 9, 7); });
        Mat FA1 = make(16, 16, [](int i, int k) { return patv(i, k, 10, 7); });
        Mat FB1 = make(16, 16, [](int k, int j) { return patv(k, j, 11, 7); });
        Mat FA2 = make(16, 16, [](int i, int k) { return patv(i, k, 12, 7); });
        Mat FB2 = make(16, 16, [](int k, int j) { return patv(k, j, 13, 7); });
        Ref FC  = ref_gemm("F",  FA,  FB,  nullptr, 1.0, 11, ACC_F16);
        Ref FC1 = ref_gemm("F1", FA1, FB1, nullptr, 1.0, 11, ACC_F16);
        Ref FC2 = ref_gemm("F2", FA2, FB2, nullptr, 1.0, 11, ACC_F16);
        need_asym("F", FC.C); need_asym("F1", FC1.C); need_asym("F2", FC2.C);
        // The three dual-accumulator results must be pairwise distinguishable.
        int d01 = 0, d02 = 0, d12 = 0;
        for (int i = 0; i < 256; i++)
        {
            d01 += FC.C.v[i] != FC1.C.v[i]; d02 += FC.C.v[i] != FC2.C.v[i]; d12 += FC1.C.v[i] != FC2.C.v[i];
        }
        if (d01 < 128 || d02 < 128 || d12 < 128) design_error("F dual", "results too similar");
        Ref zero = exact_ref(Mat(16, 16));

        DevBuf dA, dB, dA1, dB1, dA2, dB2;
        dA.upload(pack_f16(FA, FSA));   dB.upload(pack_f16(FB, FSB));
        dA1.upload(pack_f16(FA1, FSA)); dB1.upload(pack_f16(FB1, FSB));
        dA2.upload(pack_f16(FA2, FSA)); dB2.upload(pack_f16(FB2, FSB));

        {
            DevBuf o, oo; o.fill<uint16_t>(16 * FSC, SENT_F16); oo.fill<uint16_t>(16 * FSC, SENT_F16);
            LAUNCH(32, (k_f16acc_chain<false>), dA.p<half>(), FSA, dB.p<half>(), FSB, FK / 16,
                   o.p<half>(), oo.p<half>(), FSC);
            Want w(O_F16, 16, FSC); w.tile(FC);
            gate("f16acc opsel=0, K=32 chained x2", o, w);
            Want wz(O_F16, 16, FSC); wz.tile(zero);
            gate("f16acc opsel=0 leaves opsel=1 slots zero", oo, wz, false);
        }
        {
            DevBuf o, oo; o.fill<uint16_t>(16 * FSC, SENT_F16); oo.fill<uint16_t>(16 * FSC, SENT_F16);
            LAUNCH(32, (k_f16acc_chain<true>), dA.p<half>(), FSA, dB.p<half>(), FSB, FK / 16,
                   o.p<half>(), oo.p<half>(), FSC);
            Want w(O_F16, 16, FSC); w.tile(FC);
            gate("f16acc opsel=1, K=32 chained x2", o, w);
            Want wz(O_F16, 16, FSC); wz.tile(zero);
            gate("f16acc opsel=1 leaves opsel=0 slots zero", oo, wz, false);
        }
        {
            DevBuf o0a, o1a, o0b, o1b;
            o0a.fill<uint16_t>(16 * FSC, SENT_F16); o1a.fill<uint16_t>(16 * FSC, SENT_F16);
            o0b.fill<uint16_t>(16 * FSC, SENT_F16); o1b.fill<uint16_t>(16 * FSC, SENT_F16);
            LAUNCH(32, k_f16acc_dual, dA.p<half>(), dB.p<half>(), dA1.p<half>(), dB1.p<half>(),
                   dA2.p<half>(), dB2.p<half>(), FSA, FSB,
                   o0a.p<half>(), o1a.p<half>(), o0b.p<half>(), o1b.p<half>(), FSC);
            Want w0a(O_F16, 16, FSC); w0a.tile(FC);
            Want w1a(O_F16, 16, FSC); w1a.tile(FC1);
            Want w0b(O_F16, 16, FSC); w0b.tile(FC2);
            Want w1b(O_F16, 16, FSC); w1b.tile(FC1);
            gate("f16acc dual: opsel=0 half = A*B (chained)", o0a, w0a);
            gate("f16acc dual: opsel=1 half = A1*B1", o1a, w1a);
            gate("f16acc dual: clear_half<0> + A2*B2 at opsel=0", o0b, w0b);
            gate("f16acc dual: opsel=1 half survives clear_half<0>", o1b, w1b);
        }
        {
            DevBuf o; o.fill<uint16_t>(16 * FSC, SENT_F16);
            LAUNCH(32, k_f16acc_checked, dA1.p<half>(), FSA, dB1.p<half>(), FSB,
                   o.p<half>(), FSC, VR, VC);
            Want w(O_F16, 16, FSC); w.tile(FC1, false, 0, 0, VR, VC);
            gate("store_matrix_c_f16_checked<1>", o, w, false);
        }
    }

    // ------------------------------------------------------------------
    // bf16 -> fp32. Inputs are small integers * {1/2,1,2}: exact in bf16.
    // A stride 32+16 keeps the 32-byte A load aligned (misalignment was only
    // ever measured for the f16 loader).
    // ------------------------------------------------------------------
    printf("bf16 (mma_sync_bf16):\n");
    {
        const int BK = 32, BSA = BK + 16, BSB = 16 + 8, BSC = 16 + 4;
        Mat BA = make(16, BK, [](int i, int k) { return patv(i, k, 14, 9) * p2((i % 3) - 1); });
        Mat BB = make(BK, 16, [](int k, int j) { return patv(k, j, 15, 9) * p2((j % 2) - 1); });
        need_all("bf16 A", BA, bf16_exact, "A not bf16-exact");
        need_all("bf16 B", BB, bf16_exact, "B not bf16-exact");
        Ref BC = ref_gemm("bf16", BA, BB, nullptr, 0.25, 24, ACC_F32);
        need_asym("bf16", BC.C);
        DevBuf dA, dB, o;
        dA.upload(pack_bf16(BA, BSA));
        dB.upload(pack_bf16(BB, BSB));
        o.fill<uint32_t>(16 * BSC, SENT_F32);
        LAUNCH(32, k_bf16_chain, dA.p<__hip_bfloat16>(), BSA, dB.p<__hip_bfloat16>(), BSB,
               BK / 16, o.p<float>(), BSC);
        Want w(O_F32, 16, BSC); w.tile(BC);
        gate("bf16 K=32 chained x2, padded strides", o, w);
    }

    // ------------------------------------------------------------------
    // int8 -> int32, all four signedness combinations, K = 64 chained x4.
    // Exact (integer accumulate). Bytes span 0..255 so signed and unsigned
    // readings differ. A stride 64+16 bytes (16-byte aligned), B stride 16+8.
    // ------------------------------------------------------------------
    printf("int8 (mma_sync_i8):\n");
    {
        const int IK = 64, ISA = IK + 16, ISB = 16 + 8, ISC = 16 + 4;
        Mat IA = make(16, IK, [](int i, int k) { return (double)((i * 13 + k * 7 + 3) % 256); });
        Mat IB = make(IK, 16, [](int k, int j) { return (double)((k * 11 + j * 17 + 5) % 256); });
        DevBuf dA, dB;
        dA.upload(pack_u8(IA, ISA));
        dB.upload(pack_u8(IB, ISB));

        struct Combo { bool sa, sb; const char* name; };
        const Combo combos[4] = {
            { true,  true,  "i8 K=64 chained x4: A signed   x B signed"   },
            { false, false, "i8 K=64 chained x4: A unsigned x B unsigned" },
            { false, true,  "i8 K=64 chained x4: A unsigned x B signed"   },
            { true,  false, "i8 K=64 chained x4: A signed   x B unsigned" },
        };
        for (int c = 0; c < 4; c++)
        {
            Ref ref = ref_gemm(combos[c].name, bytes_as(IA, combos[c].sa), bytes_as(IB, combos[c].sb),
                               nullptr, 1.0, 31, ACC_EXACT);
            need_asym(combos[c].name, ref.C);
            DevBuf o; o.fill<uint32_t>(16 * ISC, SENT_I32);
            int8_t* pa = dA.p<int8_t>(); int8_t* pb = dB.p<int8_t>(); int* po = o.p<int>();
            (void) pa; (void) pb; (void) po;
            switch (c)
            {
                case 0: LAUNCH(32, (k_i8_chain<true,  true >), pa, ISA, pb, ISB, IK / 16, po, ISC); break;
                case 1: LAUNCH(32, (k_i8_chain<false, false>), pa, ISA, pb, ISB, IK / 16, po, ISC); break;
                case 2: LAUNCH(32, (k_i8_chain<false, true >), pa, ISA, pb, ISB, IK / 16, po, ISC); break;
                case 3: LAUNCH(32, (k_i8_chain<true,  false>), pa, ISA, pb, ISB, IK / 16, po, ISC); break;
            }
            Want w(O_I32, 16, ISC); w.tile(ref);
            gate(combos[c].name, o, w);
        }
        {
            Mat IA16 = make(16, 16, [&](int i, int k) { return IA.at(i, k); });
            Mat IB16 = make(16, 16, [&](int k, int j) { return IB.at(k, j); });
            Ref ref = ref_gemm("i8 checked", bytes_as(IA16, false), bytes_as(IB16, true), nullptr, 1.0, 31, ACC_EXACT);
            DevBuf o; o.fill<uint32_t>(16 * ISC, SENT_I32);
            LAUNCH(32, k_i8_checked, dA.p<int8_t>(), ISA, dB.p<int8_t>(), ISB, o.p<int>(), ISC, VR, VC);
            Want w(O_I32, 16, ISC); w.tile(ref, false, 0, 0, VR, VC);
            gate("store_matrix_c_i32_checked (u8 x s8)", o, w, false);
        }
    }

    // ------------------------------------------------------------------
    // INFO cases: never affect the exit code, not in the golden file.
    // ------------------------------------------------------------------
    printf("INFO (does not affect the gate result):\n");

    // fp32 accumulator characterization. 16 probes on the diagonal of one
    // K=16 tile (probe p = A row p . B column p), each chosen so the exact sum
    // IS representable in fp32 -- any difference is the accumulator, not the
    // final rounding. Inputs are fp16 normals.
    struct Probe { const char* what; std::vector<double> a, b; };
    const double t8 = p2(-8), t7 = p2(-7), t12 = p2(-12);
    std::vector<Probe> probes = {
        { "9 - 9",                          {3, -3},  {3, 3} },
        { "1 - 1",                          {1, -1},  {1, 1} },
        { "1024 - 1024",                    {32, -32}, {32, 32} },
        { "4 - 1",                          {4, -1},  {1, 1} },
        { "8 - 1",                          {8, -1},  {1, 1} },
        { "9 - 9 + 9 - 9",                  {3, -3, 3, -3}, {3, 3, 3, 3} },
        { "-1 - 1 (no mixed sign)",         {-1, -1}, {1, 1} },
        { "1024 + 2^-13 (1 term)",          {1024, p2(-7)}, {1, p2(-6)} },
        { "1024 + 8 x 2^-16",               {1024, t8, t8, t8, t8, t8, t8, t8, t8}, {1, t8, t8, t8, t8, t8, t8, t8, t8} },
        { "1024 + 14 x 2^-14",              {1024, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7},
                                            {1, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7} },
        { "1024 + 4x2^-12 + 4x2^-13 + 6x2^-14",
                                            {1024, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7},
                                            {1, p2(-5), p2(-5), p2(-5), p2(-5), p2(-6), p2(-6), p2(-6), p2(-6),
                                             t7, t7, t7, t7, t7, t7} },
        { "1024 + 15 x 2^-12",              {1024, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7, t7},
                                            {1, p2(-5), p2(-5), p2(-5), p2(-5), p2(-5), p2(-5), p2(-5), p2(-5),
                                             p2(-5), p2(-5), p2(-5), p2(-5), p2(-5), p2(-5), p2(-5)} },
        { "-1024 + 8 x 2^-16",              {-1024, t8, t8, t8, t8, t8, t8, t8, t8}, {1, t8, t8, t8, t8, t8, t8, t8, t8} },
        { "1 + 14 x 2^-24",                 {1, t12, t12, t12, t12, t12, t12, t12, t12, t12, t12, t12, t12, t12, t12},
                                            {1, t12, t12, t12, t12, t12, t12, t12, t12, t12, t12, t12, t12, t12, t12} },
        { "16 x (1*1) control",             std::vector<double>(16, 1.0), std::vector<double>(16, 1.0) },
        { "3*3 control",                    {3}, {3} },
    };
    if (probes.size() != 16) design_error("accumulator probe", "need exactly 16 probes");
    {
        Mat QA(16, 16), QB(16, 16), exact(16, 16);
        for (int p = 0; p < 16; p++)
        {
            const Probe& q = probes[p];
            double s = 0;
            for (size_t k = 0; k < q.a.size(); k++) { QA.at(p, (int) k) = q.a[k]; QB.at((int) k, p) = q.b[k]; s += q.a[k] * q.b[k]; }
            exact.at(p, p) = s;
            if (!f32_exact(s)) design_error(q.what, "probe sum not representable in fp32");
        }
        DevBuf dA, dB, o;
        dA.upload(pack_f16(QA, 16)); dB.upload(pack_f16(QB, 16));
        o.fill<uint32_t>(16 * 16, SENT_F32);
        LAUNCH(32, k_f32_chain, dA.p<half>(), 16, dB.p<half>(), 16, 1,
               (const float*) nullptr, 0, o.p<float>(), 16);
        if (g_design) printf("  INFO accumulator probe: design ok (%zu probes)\n", probes.size());
        else
        {
            auto raw = o.get<float>();
            printf("  INFO fp32 accumulator probe (K=16, one mma, exact sum representable; Emax = exponent of largest product):\n");
            for (int p = 0; p < 16; p++)
            {
                const Probe& q = probes[p];
                double mx = 0; for (size_t k = 0; k < q.a.size(); k++) mx = fmax(mx, fabs(q.a[k] * q.b[k]));
                double got = raw[p * 16 + p], ex = exact.at(p, p), d = got - ex;
                int e; frexp(mx, &e);
                printf("    %-38s exact=%-16.10g got=%-16.10g diff=%-11.4g = %+g x 2^(Emax-23)\n",
                       q.what, ex, got, d, d / ldexp(1.0, (e - 1) - 23));
            }
        }
    }
#ifndef WMMA_GATE_DESIGN_CHECK
    {
        // Lane-half divergence. Lanes 0-15 hold S.A / S.B, lanes 16-31 hold a
        // different A2 (mode 0) or B2 (mode 1). Report which product each
        // output cell matches (within the gate bound), split by column parity
        // (lanes 16-31 own the odd C columns). Semantics UNKNOWN; recorded only.
        Mat A2 = make(16, 16, [](int i, int k) { return patv(i, k, 20, 9); });
        Mat B2 = make(16, 16, [](int k, int j) { return patv(k, j, 21, 9); });
        Ref CA2 = ref_gemm("info A2", A2, SB_, nullptr, 1.0, 24, ACC_F32);
        Ref CB2 = ref_gemm("info B2", SA_, B2, nullptr, 1.0, 24, ACC_F32);
        DevBuf dA2, dB2;
        dA2.upload(pack_f16(A2, SSA));
        dB2.upload(pack_f16(B2, SSB));
        for (int mode = 0; mode < 2; mode++)
        {
            DevBuf o; o.fill<uint32_t>(16 * SSC, SENT_F32);
            LAUNCH(32, k_info_halfdiverge, dSA.p<half>(), dA2.p<half>(), SSA,
                   dSB.p<half>(), dB2.p<half>(), SSB, mode, o.p<float>(), SSC);
            auto raw = o.get<float>();
            const Ref& alt = mode == 0 ? CA2 : CB2;
            int lo[2] = {0, 0}, hi[2] = {0, 0}, other[2] = {0, 0};
            for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++)
            {
                double g = raw[r * SSC + c]; int par = c & 1;
                if (fabs(g - SC.C.at(r, c)) <= SC.tol.at(r, c)) lo[par]++;
                else if (fabs(g - alt.C.at(r, c)) <= alt.tol.at(r, c)) hi[par]++;
                else other[par]++;
            }
            printf("  INFO lanes 16-31 load different %s: even cols {=lanes0-15 data %d, =lanes16-31 data %d, other %d}, "
                   "odd cols {%d, %d, %d}\n", mode == 0 ? "A" : "B",
                   lo[0], hi[0], other[0], lo[1], hi[1], other[1]);
        }
    }
    {
        // int8 clamp: 0xFF x 0xFF unsigned = 65025 per term, 16 terms per cell =
        // 1040400 added to a seed of INT32_MAX - 1000.
        const int seed = 2147483647 - 1000;
        Mat ones = make(16, 16, [](int, int) { return 255.0; });
        DevBuf dA, dB, oc, on;
        dA.upload(pack_u8(ones, 16)); dB.upload(pack_u8(ones, 16));
        oc.fill<uint32_t>(16 * 16, SENT_I32); on.fill<uint32_t>(16 * 16, SENT_I32);
        LAUNCH(32, k_info_i8_clamp, dA.p<int8_t>(), 16, dB.p<int8_t>(), 16, seed,
               oc.p<int>(), on.p<int>(), 16);
        auto rc = oc.get<int32_t>(); auto rn = on.get<int32_t>();
        const long long exact = (long long) seed + 16LL * 65025;
        const int wrapped = (int)(uint32_t)(uint64_t) exact;
        int sat = 0, wrp = 0, nwrp = 0;
        for (int i = 0; i < 256; i++)
        {
            sat += rc[i] == 2147483647;
            wrp += rc[i] == wrapped;
            nwrp += rn[i] == wrapped;
        }
        printf("  INFO i8 clamp=true: %d/256 saturated to INT32_MAX, %d/256 wrapped; "
               "clamp=false: %d/256 wrapped (exact %lld, wrapped %d)\n",
               sat, wrp, nwrp, exact, wrapped);
    }
#endif

    if (g_design_errors)
    {
        printf("%d test-design error(s): the gate's inputs are not valid\n", g_design_errors);
        printf("WMMA GATE: FAIL (test design)\n");
        return 2;
    }
    if (g_design)
    {
        printf("design check: %d gate cases, inputs/references valid\n", g_gate_total);
        return 0;
    }
    printf("%d/%d gate cases passed (math failures %d, golden failures %d)\n",
           g_gate_total - g_gate_fail, g_gate_total, g_math_fail, g_record_mode ? 0 : g_golden_fail);
    if (g_record_mode)
    {
        if (g_math_fail)
        {
            printf("refusing to record a golden file while the math check fails\nWMMA GATE: FAIL\n");
            return 1;
        }
        if (!golden_write(golden_path))
        {
            printf("cannot write %s\nWMMA GATE: FAIL (setup)\n", golden_path.c_str());
            return 2;
        }
        printf("recorded %zu cases to %s\n", g_record_order.size(), golden_path.c_str());
    }
    printf("WMMA GATE: %s\n", g_gate_fail ? "FAIL" : "PASS");
    return g_gate_fail ? 1 : 0;
}
