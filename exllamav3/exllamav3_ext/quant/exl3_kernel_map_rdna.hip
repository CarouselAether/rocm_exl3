#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <c10/hip/HIPGuard.h>
#include <ATen/hip/HIPContext.h>

#include "../util.h"
#include "../util.cuh"
#include "../rdna_wmma.hip"

#include "exl3_kernel_map_rdna.hip.h"
#include "exl3_devctx.cuh"

#include <map>
#include <mutex>

#include "comp_units_rdna/exl3_comp_unit_1_rdna.hip.h"
#include "comp_units_rdna/exl3_comp_unit_2_rdna.hip.h"
#include "comp_units_rdna/exl3_comp_unit_3_rdna.hip.h"
#include "comp_units_rdna/exl3_comp_unit_4_rdna.hip.h"
#include "comp_units_rdna/exl3_comp_unit_5_rdna.hip.h"
#include "comp_units_rdna/exl3_comp_unit_6_rdna.hip.h"
#include "comp_units_rdna/exl3_comp_unit_7_rdna.hip.h"
#include "comp_units_rdna/exl3_comp_unit_8_rdna.hip.h"

// RDNA 3.5 has 64KB LDS (not 96KB like some other architectures)
#ifndef SMEM_MAX_RDNA35
#define SMEM_MAX_RDNA35 65536
#endif

#ifndef SMEM_MAX
#define SMEM_MAX SMEM_MAX_RDNA35
#endif

// ============================================================================
// GLOBALS
// ============================================================================
std::map<uint64_t, TResult> _tuning_cache = {};
static std::mutex _tuning_mutex;

// Storage for the shared-memory staging arrays declared extern in
// exl3_gemm_kernel_rdna.cuh. Defining them in a single TU avoids the
// multiple-definition problem that the CUDA path side-steps via its own
// non-extern header.
__device__ int64_t v_indices[128];
__device__ half    v_weights[128];
__device__ int     bszm_sync;

int exl3_gemm_num_kernel_shapes() { return EXL3_GEMM_NUM_SHAPES; }

int exl3_gemm_tilesize_k[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim[]   = {EXL3_GEMM_BLOCKDIM};

// ============================================================================
// SHARED-MEMORY MODEL
// ============================================================================
// Must match exl3_gemm_inner_rdna.cuh's dynamic shared layout.
//
// CRITICAL: All shapes now use:
// - TILE_K = 16 (no more K=32)
// - threads = 256 (no more 512)
// - SH_STAGES = 3-4 (reduced where needed)
static inline size_t exl3_gemm_smem_bytes(int bits, int shape_idx)
{
    const int TILE_M = 16;
    const int TILE_K = 16;     // Fixed to 16 for all RDNA 3.5 shapes
    
    const int threads = exl3_gemm_blockdim[shape_idx]; // Always 256
    const int tile_n  = exl3_gemm_tilesize_n[shape_idx];

    // Shape-specific SH_STAGES
    // Shape 1: 4 stages (small tiles can afford it)
    // Shapes 2-4: 3 stages (reduced to save memory)
    int sh_stages = (shape_idx == 1) ? 4 : 3;

    const int tileblocks_k = TILE_K / 16; // Always 1
    const int tileblocks_n = tile_n / 16;
    const int num_warps = threads / 32;

    // sh_a: half
    size_t a_bytes = sizeof(half) * (size_t)(2 * sh_stages * TILE_M * TILE_K);

    // sh_b: uint16_t
    size_t b_elems = (size_t)(2 * sh_stages * tileblocks_k * tileblocks_n * (256 / 16) * bits);
    size_t b_bytes = sizeof(uint16_t) * b_elems;

    // sh_c: int
    size_t c_bytes = sizeof(int) * (size_t)(tileblocks_k * 8 * threads);

    // sh_dq: uint16_t
    size_t dq_bytes = sizeof(uint16_t) * (size_t)(num_warps * 16 * 17);

    return a_bytes + b_bytes + c_bytes + dq_bytes;
}

bool exl3_gemm_shape_compat(int shape_idx, int size_m, int size_k, int size_n, int bits)
{
    (void)size_m;

    // Compatibility is primarily divisibility + shared-memory fit.
    int tk = exl3_gemm_tilesize_k[shape_idx];
    int tn = exl3_gemm_tilesize_n[shape_idx];
    if ((size_k % tk) != 0) return false;
    if ((size_n % tn) != 0) return false;

    // Reject shapes that cannot fit LDS for this bits mode.
    size_t smem = exl3_gemm_smem_bytes(bits, shape_idx);
    if (smem > (size_t)SMEM_MAX) return false;
    
    return true;
}

static inline bool is_compat_shape(int shape_idx, int size_k_eff, int size_n_eff, int bits)
{
    int tk = exl3_gemm_tilesize_k[shape_idx];
    int tn = exl3_gemm_tilesize_n[shape_idx];
    
    // Check divisibility
    if ((size_k_eff % tk) != 0) return false;
    if ((size_n_eff % tn) != 0) return false;
    
    // Check shared memory limit
    size_t smem = exl3_gemm_smem_bytes(bits, shape_idx);
    return smem <= (size_t)SMEM_MAX;
}

// ============================================================================
// Kernel Pointer Helpers
// ============================================================================

static fp_exl3_gemm_kernel get_gemm_kernel_ptr(int bits, int shape_idx, bool c_fp32, int cb)
{
    int kernel_idx = shape_idx + (EXL3_GEMM_NUM_SHAPES + 1) * cb;

    if (c_fp32)
    {
        switch (bits)
        {
            case 1: return tfp_exl3_gemm_kernel_fp32_b1[kernel_idx];
            case 2: return tfp_exl3_gemm_kernel_fp32_b2[kernel_idx];
            case 3: return tfp_exl3_gemm_kernel_fp32_b3[kernel_idx];
            case 4: return tfp_exl3_gemm_kernel_fp32_b4[kernel_idx];
            case 5: return tfp_exl3_gemm_kernel_fp32_b5[kernel_idx];
            case 6: return tfp_exl3_gemm_kernel_fp32_b6[kernel_idx];
            case 7: return tfp_exl3_gemm_kernel_fp32_b7[kernel_idx];
            case 8: return tfp_exl3_gemm_kernel_fp32_b8[kernel_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM bits");
        }
    }
    else
    {
        switch (bits)
        {
            case 1: return tfp_exl3_gemm_kernel_fp16_b1[kernel_idx];
            case 2: return tfp_exl3_gemm_kernel_fp16_b2[kernel_idx];
            case 3: return tfp_exl3_gemm_kernel_fp16_b3[kernel_idx];
            case 4: return tfp_exl3_gemm_kernel_fp16_b4[kernel_idx];
            case 5: return tfp_exl3_gemm_kernel_fp16_b5[kernel_idx];
            case 6: return tfp_exl3_gemm_kernel_fp16_b6[kernel_idx];
            case 7: return tfp_exl3_gemm_kernel_fp16_b7[kernel_idx];
            case 8: return tfp_exl3_gemm_kernel_fp16_b8[kernel_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM bits");
        }
    }
    return nullptr;
}

static fp_exl3_mgemm_kernel get_mgemm_kernel_ptr(int bits, int shape_idx, bool c_fp32, int cb)
{
    int kernel_idx = shape_idx + (EXL3_GEMM_NUM_SHAPES + 1) * cb;

    if (c_fp32)
    {
        switch (bits)
        {
            case 1: return tfp_exl3_mgemm_kernel_fp32_b1[kernel_idx];
            case 2: return tfp_exl3_mgemm_kernel_fp32_b2[kernel_idx];
            case 3: return tfp_exl3_mgemm_kernel_fp32_b3[kernel_idx];
            case 4: return tfp_exl3_mgemm_kernel_fp32_b4[kernel_idx];
            case 5: return tfp_exl3_mgemm_kernel_fp32_b5[kernel_idx];
            case 6: return tfp_exl3_mgemm_kernel_fp32_b6[kernel_idx];
            case 7: return tfp_exl3_mgemm_kernel_fp32_b7[kernel_idx];
            case 8: return tfp_exl3_mgemm_kernel_fp32_b8[kernel_idx];
            default: TORCH_CHECK(false, "No kernel for MGEMM bits");
        }
    }
    else
    {
        switch (bits)
        {
            case 1: return tfp_exl3_mgemm_kernel_fp16_b1[kernel_idx];
            case 2: return tfp_exl3_mgemm_kernel_fp16_b2[kernel_idx];
            case 3: return tfp_exl3_mgemm_kernel_fp16_b3[kernel_idx];
            case 4: return tfp_exl3_mgemm_kernel_fp16_b4[kernel_idx];
            case 5: return tfp_exl3_mgemm_kernel_fp16_b5[kernel_idx];
            case 6: return tfp_exl3_mgemm_kernel_fp16_b6[kernel_idx];
            case 7: return tfp_exl3_mgemm_kernel_fp16_b7[kernel_idx];
            case 8: return tfp_exl3_mgemm_kernel_fp16_b8[kernel_idx];
            default: TORCH_CHECK(false, "No kernel for MGEMM bits");
        }
    }
    return nullptr;
}

// ============================================================================
// OCCUPANCY-AWARE SHAPE SELECTION
// ============================================================================

// Cache occupancy per (bits, shape, c_fp32). -1 means unknown.
static int g_occ_cache[9][EXL3_GEMM_NUM_SHAPES + 1][2] = {};
static bool g_occ_init = false;

static inline void ensure_occ_cache_init()
{
    if (g_occ_init) return;
    for (int b = 0; b <= 8; ++b)
        for (int s = 0; s <= EXL3_GEMM_NUM_SHAPES; ++s)
            for (int f = 0; f < 2; ++f)
                g_occ_cache[b][s][f] = -1;
    g_occ_init = true;
}

static inline int occ_blocks_per_cu(int bits, int shape_idx, bool c_fp32)
{
    ensure_occ_cache_init();
    int fp = c_fp32 ? 1 : 0;
    int &cached = g_occ_cache[bits][shape_idx][fp];
    if (cached >= 0) return cached;

    // Use cb=0 as representative (other cb variants are similar enough for selection).
    fp_exl3_gemm_kernel k = get_gemm_kernel_ptr(bits, shape_idx, c_fp32, /*cb=*/0);
    if (!k)
    {
        cached = 0;
        return cached;
    }

    size_t smem = exl3_gemm_smem_bytes(bits, shape_idx);
    if (smem > (size_t)SMEM_MAX)
    {
        cached = 0;
        return cached;
    }

    int blocks = 0;
    hipError_t err = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks,
        (const void*)k,
        exl3_gemm_blockdim[shape_idx],
        smem
    );
    if (err != hipSuccess) blocks = 0;
    cached = blocks;
    return cached;
}

// ============================================================================
// RDNA 3.5 SHAPE SELECTION STRATEGY
// ============================================================================
// With 64KB limit and uniform 256-thread blocks, we prioritize:
// 1. Compatibility (divisibility + fits in 64KB for given bit precision)
// 2. Larger N tiles for better memory efficiency (within bit budget)
// 3. Higher occupancy (but all shapes now have similar occupancy)
// 4. Enough parallelism to keep all WGPs busy
//
// Bit-aware selection:
// - 1-4 bit: Can use Shape 4 (N=384) for maximum efficiency
// - 5-6 bit: Limited to Shape 3 (N=256) or smaller
// - 7-8 bit: Limited to Shape 2 (N=192) or smaller
// ============================================================================

static int select_shape_core(
    int cc,
    int size_m,
    int size_k_eff,
    int size_n_eff,
    int bits,
    bool multi,
    bool c_fp32)
{
    (void)cc;
    (void)size_m;

    // Use DevCtx CU count
    int device = 0;
    hipGetDevice(&device);
    int cu_count = DevCtx::instance().get_num_sms(device);
    if (cu_count <= 0) cu_count = 1;

    int best_shape = 1;
    long long best_score = LLONG_MIN;

    // Try shapes in descending order (prefer larger tiles)
    for (int shape = EXL3_GEMM_NUM_SHAPES; shape >= 1; --shape)
    {
        if (!is_compat_shape(shape, size_k_eff, size_n_eff, bits)) continue;
        
        size_t smem = exl3_gemm_smem_bytes(bits, shape);
        if (smem > (size_t)SMEM_MAX) continue;

        int blocks_per_cu = occ_blocks_per_cu(bits, shape, c_fp32);
        if (blocks_per_cu <= 0) continue;

        int tn = exl3_gemm_tilesize_n[shape];
        long long slices = (long long)(size_k_eff / 16) * (long long)(size_n_eff / tn);
        if (slices < 1) slices = 1;

        long long max_resident = (long long)cu_count * (long long)blocks_per_cu;
        long long underfill = (max_resident > slices) ? (max_resident - slices) : 0;

        // Score components for RDNA 3.5:
        // - Strongly prefer larger N tiles (most important for memory efficiency)
        // - Moderate preference for higher occupancy (all shapes similar now)
        // - Ensure enough work for all CUs
        // - Light penalty for underfill
        long long score = 0;
        score += (long long)tn * 200;                 // Tile size is king
        score += (long long)blocks_per_cu * 10000;    // Occupancy matters
        score += (long long)std::min(slices, max_resident) * 10;
        score -= underfill * (multi ? 30 : 10);

        // Bit-specific bonuses
        if (bits <= 4) {
            // Low bits can afford large tiles - encourage Shape 4
            if (shape == 4) score += 50000;
        } else if (bits <= 6) {
            // Medium bits - encourage Shape 3
            if (shape == 3) score += 30000;
        } else {
            // High bits - encourage Shape 2 or 1
            if (shape <= 2) score += 20000;
        }

        if (score > best_score)
        {
            best_score = score;
            best_shape = shape;
        }
    }

    return best_shape;
}

// Internal helper: optionally scale dims for MGEMM batching.
static int select_gemm_shape_scaled(int cc, int size_m, int size_k, int size_n, int bits, bool multi, bool c_fp32, int bszm_in, int bszm_out)
{
    int size_k_eff = size_k * bszm_in;
    int size_n_eff = size_n * bszm_out;
    return select_shape_core(cc, size_m, size_k_eff, size_n_eff, bits, multi, c_fp32);
}

// Public heuristic (matches exl3_kernel_map.cuh)
int select_gemm_shape(int cc, int size_m, int size_k, int size_n, int bits, bool multi)
{
    // c_fp32 is unknown at this call site; assume fp16 for shape selection.
    return select_shape_core(cc, size_m, size_k, size_n, bits, multi, /*c_fp32=*/false);
}

// ============================================================================
// LEGACY WRAPPERS
// ============================================================================

fp_exl3_gemm_kernel select_exl3_gemm_kernel(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int bits,
    bool c_fp32,
    int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* out_num_sms,
    int cb)
{
    int shape_idx = force_shape_idx > 0 ? force_shape_idx : select_shape_core(cc, size_m, size_k, size_n, bits, /*multi=*/false, c_fp32);

    TORCH_CHECK(shape_idx > 0, "exl3_gemm: no compatible kernel");
    if (out_shape_idx) *out_shape_idx = shape_idx;
    if (out_block_dim) *out_block_dim = exl3_gemm_blockdim[shape_idx];

    if (out_num_sms)
    {
        // Clamp requested SMS to the number of slices.
        int tn = exl3_gemm_tilesize_n[shape_idx];
        long long slices = (long long)(size_k / 16) * (long long)(size_n / tn);
        int max_slices = (int)std::max(1LL, slices);
        *out_num_sms = MAX(MIN(max_slices, *out_num_sms), 1);
    }

    return get_gemm_kernel_ptr(bits, shape_idx, c_fp32, cb);
}

fp_exl3_mgemm_kernel select_exl3_mgemm_kernel(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int bits,
    bool c_fp32,
    int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* out_num_sms,
    int cb,
    int bszm_in,
    int bszm_out)
{
    int shape_idx = force_shape_idx > 0
        ? force_shape_idx
        : select_gemm_shape_scaled(cc, size_m, size_k, size_n, bits, /*multi=*/true, c_fp32, bszm_in, bszm_out);

    TORCH_CHECK(shape_idx > 0, "exl3_mgemm: no compatible kernel");
    if (out_shape_idx) *out_shape_idx = shape_idx;
    if (out_block_dim) *out_block_dim = exl3_gemm_blockdim[shape_idx];

    if (out_num_sms)
    {
        int tn = exl3_gemm_tilesize_n[shape_idx];
        long long slices = (long long)(size_k / 16) * (long long)(size_n / tn);
        int max_slices = (int)std::max(1LL, slices);
        *out_num_sms = MAX(MIN(max_slices, *out_num_sms), 1);
    }

    return get_mgemm_kernel_ptr(bits, shape_idx, c_fp32, cb);
}

// ============================================================================
// SELECTOR (Cache + Returns GEMM+MGEMM)
// ============================================================================

static TResult g_force_tr;

TResult* select_exl3_gemm_mgemm_kernel_new(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int bits,
    bool c_fp32,
    int force_shape_idx,
    int force_num_sms,
    int cb)
{
    // Forced path (benchmarking)
    if (force_shape_idx > 0)
    {
        TORCH_CHECK(force_num_sms > 0, "Must supply force_shape_idx and force_num_sms together");
        g_force_tr.kernel = get_gemm_kernel_ptr(bits, force_shape_idx, c_fp32, cb);
        g_force_tr.mkernel = get_mgemm_kernel_ptr(bits, force_shape_idx, c_fp32, cb);
        g_force_tr.shape_idx = force_shape_idx;
        g_force_tr.num_sms = force_num_sms;
        g_force_tr.block_dim = exl3_gemm_blockdim[force_shape_idx];
        return &g_force_tr;
    }
    TORCH_CHECK(force_num_sms == 0, "Must supply force_shape_idx and force_num_sms together.");

    // Cache key: size_k, size_n, cc, bits, cb, c_fp32
    uint64_t key = (((uint64_t)(uint32_t)size_k) << 40) |
                   (((uint64_t)(uint32_t)size_n) << 16) |
                   (((uint64_t)(uint8_t)cc)      <<  8) |
                   (((uint64_t)(uint8_t)bits)    <<  4) |
                   ((uint64_t)(uint8_t)cb        <<  1) |
                   (c_fp32 ? 1ull : 0ull);

    std::lock_guard<std::mutex> lock(_tuning_mutex);
    auto it = _tuning_cache.find(key);
    if (it != _tuning_cache.end()) return &(it->second);

    int device = 0;
    hipGetDevice(&device);
    int cu_count = DevCtx::instance().get_num_sms(device);
    if (cu_count <= 0) cu_count = 1;

    int shape_idx = select_shape_core(cc, size_m, size_k, size_n, bits, /*multi=*/false, c_fp32);

    int tn = exl3_gemm_tilesize_n[shape_idx];
    long long slices = (long long)(size_k / 16) * (long long)(size_n / tn);
    int max_slices = (int)std::max(1LL, slices);

    // Choose a reasonable default num_sms: enough to fill the device given occupancy,
    // but not exceeding slices.
    int blocks_per_cu = occ_blocks_per_cu(bits, shape_idx, c_fp32);
    int max_fill = MAX(1, cu_count * MAX(1, blocks_per_cu));
    int num_sms = MIN(max_slices, max_fill);

    TResult tr;
    tr.kernel = get_gemm_kernel_ptr(bits, shape_idx, c_fp32, cb);
    tr.mkernel = get_mgemm_kernel_ptr(bits, shape_idx, c_fp32, cb);
    tr.shape_idx = shape_idx;
    tr.num_sms = MAX(1, num_sms);
    tr.block_dim = exl3_gemm_blockdim[shape_idx];

    _tuning_cache.emplace(key, tr);
    return &(_tuning_cache.find(key)->second);
}
