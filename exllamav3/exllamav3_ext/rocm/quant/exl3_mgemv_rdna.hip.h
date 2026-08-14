#pragma once

// =============================================================================
// Multi-matrix (expert-batched) GEMV for RDNA at m == 1 -- host interface
// =============================================================================
//
// See exl3_mgemv_rdna.hip for the design. This header exists so
// exl3_gemm_rdna.hip (the exl3_mgemm entry point) can route to the path
// without pulling in the kernel definitions.

#include <cuda_fp16.h>
#include <cstdint>

class Graph;

// Device-side parameter block, one per device. The prologue kernel receives the
// graph-patchable pointers (C, indices, weights) as ordinary kernel arguments
// -- making it the single node whose parameters Graph::launch() patches -- and
// republishes them here for the downstream kernels, whose own argument copies
// would otherwise go stale after a patch. In-stream ordering makes the handoff
// safe: the prologue finishes before any consumer launches, and sequential
// calls sharing the block cannot interleave on one stream. Two streams mutating
// one device concurrently would race on this block, but they already race on
// DevCtx's shared `locks` buffer in the cooperative path, so this adds no new
// constraint.
struct Exl3MgemvParams
{
    void* C;
    const int64_t* indices;
    const half* weights;
};

// Single-matrix analogue for the graph-captured exl3_gemm path, implemented in
// exl3_gemv_rdna.hip beside the non-graph dispatch. Same prologue-republish
// trick with the six GP_gemm_* sites; see the comment there.
struct Exl3GemvGraphParams
{
    const uint16_t* B;
    void* C;
    half* A_had;
    const half* svh;
};

bool exl3_gemv_graph_try_launch
(
    const half* A_ptr,
    const uint16_t* B_ptr,
    void* C_ptr,
    const half* suh_ptr,
    half* A_had_ptr,
    const half* svh_ptr,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    int device,
    cudaStream_t stream,
    Graph* graph
);

// Defined in exl3_gemv_rdna.hip; the EXL3_GEMV_SPLITK kill switch, re-read per
// call, shared by every split-K launch site.
bool exl3_gemv_splitk_enabled();

// Defined in exl3_gemv_rdna.hip; shape-aware split-K wave count (4, 8 or 16),
// shared by the plain, graph and mgemv split-K sites. bszm is the grid's
// matrix-batch factor (1 for the single-matrix sites); EXL3_GEMV_SPLITK_WARPS
// forces one count everywhere.
int exl3_gemv_splitk_warps(int k_tiles, int n_tiles, int bszm);

// Shape-aware GEMV profitability rule, shared by the graph and non-graph
// routing sites. With the in-block split-K form (2026-08-08) the GEMV wins
// every measured shape, so the rule is currently "always" and EXL3_GEMV modes
// 1 and 2 are equivalent; the hook stays for the day a losing shape appears.
// Measured on Laguna/gfx1151, µs GEMV(split-K) vs cooperative GEMM at m == 1:
//
//   3072->100352  3.65x   3072->12288  2.21x   3072->9216  2.04x
//   3072->6144    2.11x   9216->3072   1.98x   6144->3072  1.84x
//   12288->3072   2.15x   1024->3072   1.87x   3072->1024  1.45x
//
// (Pre-split-K, the single-warp form lost below n = 3072 -- 3072->1024 was
// 0.58x with only 8 blocks -- which is why this rule used to gate on n.)
// EXL3_GEMV=0 disables the path entirely.
static inline bool exl3_gemv_rdna_pays(int size_k, int size_n)
{
    (void) size_k;
    (void) size_n;
    return true;
}

// Launches the multi-matrix GEMV pipeline (rotate + dot + rotate + reduce) if
// the call is eligible, returning true. Returns false -- having launched
// nothing -- when the call must fall through to the cooperative exl3_mgemm.
bool exl3_mgemv_try_launch
(
    const half* A_ptr,
    const uintptr_t* B_ptr_ptr,
    void* C_ptr,
    const uintptr_t* suh_ptr_ptr,
    half* A_had_ptr,
    const uintptr_t* svh_ptr_ptr,
    const int64_t* indices_ptr,
    const half* weights_ptr,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    int bszm_in,
    int bszm_out,
    int min_index,
    int max_index,
    int num_tokens,
    const int* size_n_list,   // per-matrix output widths (device), or nullptr
    void** c_list,            // per-matrix output pointers (device), or nullptr
    int device,
    cudaStream_t stream,
    Graph* graph
);
