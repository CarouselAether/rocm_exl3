// =============================================================================
// exl3_gemv_int8 for RDNA -- NOT PORTED, path disabled
// =============================================================================
//
// This file exists so the ROCm build links. It is not a port: it defines
// upstream v1.3.0's quant/exl3_gemv_int8.cu interface with the path
// unconditionally off, so every caller falls through to the fp16 kernels.
//
// Upstream's quant/exl3_gemv_int8.cu cannot be built with the shim. It carries
// inline PTX (a dp4a.u32.s32 in quant/exl3_gemv_int8_kernel.cuh, one of the
// three PTX-bearing files in the tree) and cp_async at three sites. Probed
// directly it fails with "invalid output constraint '=f' in asm".
//
// There is no fork equivalent to carry forward -- the fused int8-activation GEMV
// is new in v1.3.0, so this is the one piece of the quant path that has to be
// written rather than ported. What it needs, in rough order:
//
//   - dp4a.u32.s32 -> sudot4(false, a, true, b, c, false). NOT sdot4: gfx1151
//     lacks dot1-insts, so sdot4 fails to compile and reads like "no int8
//     support". It has dot8-insts. Watch the sign-flag pairing -- each flag
//     describes the vector that follows it.
//   - Alternatively the native i32_16x16x16_iu8 WMMA, whose wrapper
//     (mma_sync_i8<signed_a, signed_b> in rocm/rdna_wmma.hip.h) is implemented
//     and validated by rocm_tools/wmma_check.hip. That hides the sign-flag trap.
//   - 3 cp_async sites.
//   - The cooperative launch structure, which is the same shape as the GEMM's
//     and so subject to the same LDS sizing issue -- see EXL3_RDNA_SMEM in
//     exl3_gemm_rdna.hip.
//
// Behaviour of the stubs below is the correct disabled semantics, not an
// approximation:
//
//   exl3_gemv_int8_enabled() -> false
//       exl3_gemm checks this before calling exl3_gemv_int8 at all, so the mul1
//       fast path is skipped and mul1 tensors run the regular kernel.
//
//   exl3_gemv_int8() -> false
//       "not handled, fall through to the regular kernel". Also correct on its
//       own if some future caller skips the enabled() check.
//
//   exl3_gemv_int8_max_k() -> 0
//       This one is bound to Python and read by Config.use_mgemm, which computes
//       K_thr = max_k + 1 and fuses when K >= K_thr. Returning 0 gives K_thr = 1,
//       so every K fuses -- which is what should happen when there is no int8
//       GEMV to unfuse *into*. Returning upstream's per-arch value here would
//       make the Python side unfuse MGEMM calls in favour of a path that does
//       not exist.
//
// When the real port lands, this file is replaced, not amended.
// =============================================================================

#include <cuda_fp16.h>
#include "../../quant/exl3_gemv_int8.cuh"

bool exl3_gemv_int8_enabled()
{
    return false;
}

int exl3_gemv_int8_max_k(int device)
{
    return 0;
}

bool exl3_gemv_int8
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    cudaStream_t stream,
    Graph* graph
)
{
    return false;
}
