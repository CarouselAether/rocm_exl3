// =============================================================================
// exl3_moe_coop.cu for RDNA -- disabled stub
// =============================================================================
//
// Upstream v1.5.0 replaced BC_BlockSparseMLP::run_bszN's three exl3_mgemm graph
// launches with exl3_moe_coop, a fused decode-shaped MoE kernel
// (quant/exl3_moe_coop_kernel.cuh) built on exl3_gemv_kernel.cuh -- the PTX
// mma.sync / cp.async GEMV that already has no RDNA sibling (the RDNA GEMV is a
// dot-product kernel with a different device API, see exl3_gemv_kernel_rdna.hip.h).
// Porting it means re-deriving the kernel, not swapping includes; that has not
// been done.
//
// This file keeps the build linking. libtorch/blocksparse_mlp.cpp (compiled
// unmodified through the shim) constructs the parameter block at load and calls
// exl3_moe_coop_run from run_bszN:
//
//   exl3_moe_coop_prepare  -> returns an empty parameter block (no validation)
//   exl3_moe_coop_run      -> error
//   exl3_moe_coop_launch   -> error
//   exl3_moe_coop          -> error (the test entry point)
//
// run_bszN is never reached on ROCm by default: exllamav3/rocm_py steers
// block_sparse_mlp.forward away from the bsz <= MAX_BSZN route
// (EXL3_ROCM_MOE_BSZN), so bsz 1..8 MoE decode runs the fused exl3_moe kernel
// instead. If it is reached, the error below names the switch.
//
// setup.py's ROCM_EXCLUDE drops quant/exl3_moe_coop.cu (and quant/comp_units/,
// which holds its instances); building both is a duplicate symbol link failure.
// =============================================================================

#include <cuda_fp16.h>
#include "../../quant/exl3_moe_coop.cuh"
#include "../../util.h"

static const char* const kNoCoop =
    "exl3_moe_coop: the fused MoE decode kernel is CUDA-only (built on the PTX GEMV) and is not "
    "ported to ROCm. This route is disabled by default on ROCm (exllamav3/rocm_py, "
    "EXL3_ROCM_MOE_BSZN); unset EXL3_ROCM_MOE_BSZN to use the fused exl3_moe kernel instead.";

void exl3_moe_coop_launch(const MoeCoopParams& p, int K_gu, int K_d, int cb, int device, cudaStream_t stream)
{
    (void) p; (void) K_gu; (void) K_d; (void) cb; (void) device; (void) stream;
    TORCH_CHECK(false, kNoCoop);
}

MoeCoopParams exl3_moe_coop_prepare
(
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    int Kg, int Ku, int Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_gate_w,
    int& K_gu, int& K_d, int& cb
)
{
    // BC_BlockSparseMLP's constructor calls this unconditionally at load. Nothing is
    // validated or retained: the block is never launched on ROCm.
    (void) Hi; (void) g_trellis; (void) g_suh; (void) g_svh; (void) u_trellis; (void) u_suh; (void) u_svh;
    (void) d_trellis; (void) d_suh; (void) d_svh; (void) g_bias; (void) u_bias; (void) d_bias;
    (void) mcg; (void) mul1; (void) act; (void) act_limit; (void) gated;
    (void) had_g; (void) had_u; (void) gu_g; (void) gu_u; (void) act_out; (void) d_out; (void) ctr; (void) out;
    (void) sh_gate_w;
    K_gu = Kg > 0 ? Kg : Ku;
    K_d = Kd;
    cb = mul1 ? 2 : (mcg ? 1 : 0);
    MoeCoopParams p{};
    return p;
}

void exl3_moe_coop_run
(
    MoeCoopParams p, int K_gu, int K_d, int cb,
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    const c10::optional<at::Tensor>& sh_out
)
{
    (void) p; (void) K_gu; (void) K_d; (void) cb; (void) x; (void) sel; (void) rw; (void) sh_out;
    TORCH_CHECK(false, kNoCoop);
}

void exl3_moe_coop
(
    const at::Tensor& x,
    const at::Tensor& sel,
    const at::Tensor& rw,
    int min_expert,
    int max_expert,
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    int Kg, int Ku, int Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_out,
    const c10::optional<at::Tensor>& sh_gate_w
)
{
    (void) x; (void) sel; (void) rw; (void) min_expert; (void) max_expert; (void) Hi;
    (void) g_trellis; (void) g_suh; (void) g_svh; (void) u_trellis; (void) u_suh; (void) u_svh;
    (void) d_trellis; (void) d_suh; (void) d_svh; (void) g_bias; (void) u_bias; (void) d_bias;
    (void) Kg; (void) Ku; (void) Kd; (void) mcg; (void) mul1; (void) act; (void) act_limit; (void) gated;
    (void) had_g; (void) had_u; (void) gu_g; (void) gu_u; (void) act_out; (void) d_out; (void) ctr; (void) out;
    (void) sh_out; (void) sh_gate_w;
    TORCH_CHECK(false, kNoCoop);
}
