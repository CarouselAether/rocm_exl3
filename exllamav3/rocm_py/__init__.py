"""ROCm/RDNA Python-side overrides for exllamav3.

The C++ side keeps upstream sources byte-identical and puts every ROCm-specific
change in ``exllamav3_ext/rocm/`` as a sibling file (see that directory's
README). This package is the same idea for Python: instead of editing upstream
modules in place, the divergences live here and are applied as monkeypatches at
import time, from a single hook at the end of ``exllamav3/__init__.py``.

Why not edit the modules directly? The original ROCm fork did, across five
files, and every upstream rebase then had to re-derive which edits were ROCm
workarounds and which were upstream changes. Keeping them here means
``git diff`` against upstream stays empty for the shared code, and each
divergence carries its own reason and its own off switch.

A patch that cannot be applied reports "!! FAILED" in describe() rather than
being swallowed. A bisect handle that silently does nothing is worse than no
handle at all -- it makes a live kernel look disabled.

Nothing here runs unless ``torch.version.hip`` is set, so a CUDA build imports
this module and does nothing.

Environment switches (all default to the safe value for this backend):

  EXL3_ROCM_PATCH=0        disable every patch below
  EXL3_ROCM_MGEMM=0        distrust exl3_mgemm on RDNA: disables MultiLinear
                           fusion *and* the bsz-1 MoE mgemm routes. Default on
                           since 2026-08-07 (see the note at the patch)
  EXL3_ROCM_MOE_DISABLE=1  route block-sparse MoE through the dense per-expert
                           path instead of the fused kernel (NOT advised -- see
                           the note at the patch itself)
  EXL3_ROCM_RDNA4_FUSED_MOE=1  on gfx120x, do not steer MoE off the fused
                           kernel (whose WMMA traps on gfx12); for a future
                           gfx12 WMMA port

  Bisect handles -- slow, for localising a numerics fault, never to leave on:

  EXL3_ROCM_MOE_TORCH=1    MoE expert compute in pure torch
  EXL3_ROCM_ROUTING_TORCH=1  expert routing in pure torch (routing_ds3)
  EXL3_ROCM_FORCE_TORCH=1  every EXL3 Linear via reconstruct + at::mm, taking
                           exl3_gemm and exl3_gemv out of the model

  Added at the v1.5.0 sync (2026-09-20). Each keeps a v1.4.4-validated path as
  the default and makes the new upstream path opt-in until it has been run on
  RDNA:

  EXL3_ROCM_MOE_BSZN=1     let bsz <= MAX_BSZN MoE decode take upstream's
                           BC_BlockSparseMLP.run_bszN route. On ROCm that route
                           is the unported exl3_moe_coop kernel and raises;
                           default off reroutes to the fused exl3_moe kernel
  EXL3_ROCM_QKV_SLICE=1    enable the one-launch sliced Q/K/V bundle
                           (SlicedMultiLinear, exl3_mgemm sliced mode). Ported
                           into the WMMA kernels, unvalidated on RDNA
  EXL3_ROCM_BATCH_RECON=1  enable the batched expert-reconstruct prefill tier
                           (reconstruct_*_batch + hgemm_batched). Ported
                           mechanically, unvalidated on RDNA
  EXL3_ROCM_MOE_MTILE=1    let Python split fused-MoE launches into 16/32/64-row
                           tiers. Pointless on RDNA, which only builds the
                           16-row instance and runs every tier through it

These are bisect handles, not permanent policy -- turn one on, run a prompt, see
whether the output degrades. Each one's justification is a measurement recorded
at the patch, not an inherited assumption; a guard whose reason has gone stale
is a guard that should be retested and deleted.
"""

from __future__ import annotations
import os


def _env_on(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip() not in ("", "0", "false", "False")


def is_rocm() -> bool:
    try:
        import torch
        return getattr(torch.version, "hip", None) is not None
    except Exception:
        return False


_applied = False
_applied_list: list[str] = []


def apply() -> list[str]:
    """Apply the ROCm patches. Idempotent; returns the list applied.

    Returns the same list on repeat calls rather than an empty one -- callers
    use this for reporting, and recomputing would make an already-patched
    process look unpatched.
    """
    global _applied, _applied_list
    if _applied:
        return _applied_list
    if not is_rocm() or not _env_on("EXL3_ROCM_PATCH", True):
        _applied = True
        return _applied_list
    _applied = True

    applied: list[str] = []

    # ------------------------------------------------------------------
    # arch_list: hipcc takes PYTORCH_ROCM_ARCH, not TORCH_CUDA_ARCH_LIST
    # ------------------------------------------------------------------
    # Setting TORCH_CUDA_ARCH_LIST on a ROCm build makes the JIT path pass
    # NVIDIA arch flags to hipcc. Harmless for a precompiled extension, wrong
    # for a source build.
    try:
        from ..util import arch_list as _al
        _orig_set_arch = _al.maybe_set_arch_list_env

        def _noop_arch_list(*args, **kwargs):
            return None

        _al.maybe_set_arch_list_env = _noop_arch_list
        applied.append("arch_list.maybe_set_arch_list_env -> no-op")
    except Exception as e:
        applied.append(f"!! FAILED arch_list: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Triton kernel binaries: "hsaco" on AMD, "cubin" on NVIDIA
    # ------------------------------------------------------------------
    # attention_fn/bc_attn.py's _compile_kernel does ck.asm["cubin"], which
    # KeyErrors on ROCm -- triton.backends.amd emits GPUTarget(backend='hip') and
    # names the code object "hsaco". The ext-side loader is already portable
    # (cuModuleLoadData maps to hipModuleLoadData, which takes an hsaco).
    #
    # Patched at triton.compile rather than at _compile_kernel because bc_mla.py
    # does `from .bc_attn import _compile_kernel` and so holds its own reference:
    # rebinding bc_attn's module attribute would fix one caller and miss the other.
    # bc_attn imports triton *inside* the function and calls triton.compile off the
    # module, so a single alias here covers every call site with no upstream code
    # duplicated.
    #
    # Aliasing rather than renaming: anything that legitimately wants "hsaco" still
    # finds it.
    try:
        import triton as _triton

        _orig_triton_compile = _triton.compile

        def _compile_alias_hsaco(*a, **kw):
            ck = _orig_triton_compile(*a, **kw)
            try:
                asm = ck.asm
                if "cubin" not in asm and "hsaco" in asm:
                    asm["cubin"] = asm["hsaco"]
            except Exception:
                pass
            return ck

        _triton.compile = _compile_alias_hsaco
        applied.append("triton.compile: alias asm['hsaco'] -> asm['cubin']")
    except ModuleNotFoundError:
        applied.append("triton not installed -- hsaco alias skipped")
    except Exception as e:
        applied.append(f"!! FAILED triton hsaco alias: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # DSA split kernel: RDNA tile/wave retune (spill + queue-stall fix)
    # ------------------------------------------------------------------
    # _dsa_attn_split_kernel at upstream's CUDA tuning (BLOCK_H=16, num_warps=4)
    # compiles on gfx1151 at the 256-VGPR ceiling with ~2050 VGPR spills and
    # 5332 B/work-item scratch -- the ONLY scratch user in the whole decode
    # stream. Every layer then alternates it against scratch-free kernels, and
    # the hardware queue pays a ~100-500us scratch-reconfiguration stall per
    # dispatch: ~41x/token = ~5 ms/token on DeepSeek-V4-Flash, measured
    # device-side (same stream, host parked in hipDeviceSynchronize, graphs-
    # immune -- see RDNA_NOTES "DS4 per-layer stall" and rocm_tools/
    # gap_profile.py). BLOCK_H=8 + num_warps=8 cuts spills to 438 and scratch
    # to 1756 B: the stalls collapse (1369 -> 45 big gaps / 31 tokens) and
    # decode goes 15.6 -> 17.8 t/s (+14%). Sweep of 16 variants in the notes;
    # H4/w16 and BLOCK_N=16 shapes spill less still but bench worse (14.9-17.0).
    #
    # BLOCK_H is a bc_dsa module constant, so it can be retuned here; the
    # split-kernel warps are an inline argument, so wrap _compile_kernel keyed
    # on the kernel NAME -- and rebind the wrapper in every module that did
    # `from .bc_attn import _compile_kernel` (bc_dsa, bc_mla), per the aliasing
    # note above. bc_mla's DSA-on-MLA path (GLM 5.2) hardcodes a local
    # BLOCK_H=16 inside _configure, so it gets only the warps half of the fix
    # (~1428 spills); untestable here regardless -- no model fits.
    # EXL3_ROCM_DSA_TUNE=0 restores upstream tuning.
    if _env_on("EXL3_ROCM_DSA_TUNE", True):
        try:
            from ..modules.attention_fn import bc_attn as _bca
            from ..modules.attention_fn import bc_dsa as _bcd
            from ..modules.attention_fn import bc_mla as _bcm

            _bcd.BLOCK_H = 8
            _orig_compile_kernel = _bca._compile_kernel

            def _compile_kernel_rdna(device, fn, signature, constexprs, num_warps, num_stages):
                if fn.__name__ == "_dsa_attn_split_kernel":
                    num_warps = 8
                return _orig_compile_kernel(device, fn, signature, constexprs, num_warps, num_stages)

            _bca._compile_kernel = _compile_kernel_rdna
            _bcd._compile_kernel = _compile_kernel_rdna
            _bcm._compile_kernel = _compile_kernel_rdna
            applied.append("DSA split kernel retuned for RDNA (BLOCK_H=8, num_warps=8; spills 2050->438)")
        except Exception as e:
            applied.append(f"!! FAILED DSA retune patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # MultiLinear (mgemm) fusion
    # ------------------------------------------------------------------
    # attn.py fuses K/V (and Q/G) into one MultiLinear, and mlp.py fuses
    # gate/up. Both dispatch to exl3_mgemm, whose cooperative grid is
    # dim3(num_sms, 1, concurrency) -- the shape that gets REFUSED outright when
    # it exceeds co-residency (verified: "too many blocks in cooperative
    # launch"). exl3_gemm is validated on this hardware; exl3_mgemm is not.
    #
    # RETIRED 2026-08-07 -- default is now OFF (i.e. mgemm ENABLED). Set
    # EXL3_ROCM_MGEMM=0 to restore the guard.
    #
    # The NaNs measured earlier the same day were not mgemm's. They were two
    # separate defects that have since been fixed:
    #   - hip_compat's __syncwarp mapped to a bare wave_barrier(), dropping the
    #     shared-memory ordering half of CUDA's contract
    #   - threadblock_reduce() in exl3_gemm_inner_rdna.hip.h read a different
    #     sh_c address than it wrote, off the end of the LDS block
    # With both fixed, GLM-4.6V generates coherent text through the mgemm paths,
    # while the guarded route degenerates into repetition. The guard is now the
    # thing producing bad output, so it is off by default.
    #
    # This is the second time this guard's stated reason turned out to be wrong
    # (it was inherited from the fork as "cooperative launch gets refused", then
    # re-justified as "kernel NaNs"). Retest before ever re-enabling it.
    if not _env_on("EXL3_ROCM_MGEMM", True):
        try:
            from ..modules import multilinear as _ml

            class _DisabledMultiLinear:
                """Sentinel that never constructs, so callers keep their None path."""
                def __new__(cls, *args, **kwargs):
                    return None

            from ..modules import attn as _attn
            from ..modules import mlp as _mlp
            _attn.MultiLinear = _DisabledMultiLinear
            _mlp.MultiLinear = _DisabledMultiLinear
            applied.append("MultiLinear fusion disabled (exl3_mgemm unvalidated on RDNA)")
        except Exception as e:
            applied.append(f"!! FAILED MultiLinear patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # bsz-1 MoE decode: off exl3_mgemm, onto the fused exl3_moe kernel
    # ------------------------------------------------------------------
    # The MultiLinear patch above only covers attn.py and mlp.py. BlockSparseMLP
    # builds its own MultiLinears and reaches exl3_mgemm by two further routes,
    # both of which fire at bsz == 1 -- i.e. every decode step of an MoE model:
    #
    #   block_sparse_mlp.py:1222  bszn_eligible -> self.bc.run_bszN(), whose
    #                             BC_BlockSparseMLP::run_bszN_gr is three
    #                             exl3_mgemm_gr calls (gate/up/down) captured
    #                             into a CUDA graph
    #   block_sparse_mlp.py:1319  the else fallback -- the same three calls,
    #                             ungraphed
    #
    # Observed on GLM-4.6V decode: the graph route dies with "Graph update
    # failed" (graph.cu:170) and then segfaults; the first sampled token is "!",
    # the argmax of a garbage logit row. Since mgemm is NaN on this hardware
    # (see above), fixing the graph bookkeeping would only buy a clean path to a
    # wrong answer, so both routes are closed rather than repaired.
    #
    # The escape is branch 1057, whose fourth clause is the only one a bsz == 1
    # call can satisfy: `not (support_quant_paths or bszn_eligible)`. That branch
    # runs ext.exl3_moe -- the fused kernel that prefills all 46 layers finite.
    # So clear exactly those two, and nothing else:
    #
    #   - is_quantized stays True. Forcing it False was the previous bug: it
    #     does not skip MoE, it reroutes to a dense path that cannot handle
    #     quantized weights and emits all-NaN.
    #   - Patch after load_local returns, so multi_gate/up/down and
    #     fused_mode_buffers are already built (all four are computed inside
    #     load_local, gated on support_quant_paths *at load time*).
    #     exl3_moe dereferences all of them.
    #
    # Keyed off EXL3_ROCM_MGEMM because it is the same kernel and the same
    # measurement; one switch should not lie about covering half the routes.
    #
    # RETIRED 2026-08-07 alongside the MultiLinear guard above, and for a sharper
    # reason: this reroute is now measurably WORSE than what it replaced. With the
    # guard on, GLM-4.6V decode degenerates into repetition; with it off (decode via
    # bc.run_bszN -> exl3_mgemm) the same model is coherent. At retirement the fused
    # exl3_moe path this patch forces was also numerically wrong; its two split-K
    # defects were fixed 2026-08-08 (see RDNA_NOTES.md, "exl3_gemm_inner_rdna.hip.h")
    # and it now matches an fp32 reference as closely as the per-expert path. The
    # reroute stays retired anyway: mgemm decode is correct and faster.
    if not _env_on("EXL3_ROCM_MGEMM", True):
        try:
            from ..modules import block_sparse_mlp as _bsq
            _bsq_cls = _bsq.BlockSparseMLP
            _orig_bsq_load = _bsq_cls.load_local

            def _load_no_mgemm_decode(self, *args, **kwargs):
                r = _orig_bsq_load(self, *args, **kwargs)
                self.support_quant_paths = False
                self.bc = None
                return r

            _bsq_cls.load_local = _load_no_mgemm_decode
            applied.append("bsz-1 MoE decode -> fused exl3_moe (exl3_mgemm NaNs on RDNA)")
        except Exception as e:
            applied.append(f"!! FAILED MoE bsz-1 patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # EXL3_ROCM_MOE_TORCH=1 -- bisect handle: MoE compute in pure torch
    # ------------------------------------------------------------------
    # Clearing fused_mode_buffers sets min_rows = 0 in the branch-1057 loop, so no
    # expert is skipped as "already claimed by the fused kernel" and every one falls
    # through to the Torch path -- self.ups[i].forward(), i.e. the per-expert exl3
    # Linear, which is the same GEMM/GEMV a dense model exercises correctly (verified
    # 2026-08-07: Gemma-4-31b generates coherent text on this build).
    #
    # Routing still runs ahead of this, so it isolates the MoE *compute* kernel alone:
    #   coherent -> exl3_moe is the fault
    #   garbage  -> the fault is upstream of it (routing, attention, RoPE, norms, or
    #               glm4v_moe architecture support), and MoE is exonerated
    #
    # Slow by construction (46 layers x top-8 experts of small matmuls per token).
    # Fine for a one-token prompt; not a mode to leave on.
    if _env_on("EXL3_ROCM_MOE_TORCH", False):
        try:
            from ..modules import block_sparse_mlp as _bst
            _bst_cls = _bst.BlockSparseMLP
            _orig_bst_load = _bst_cls.load_local

            def _load_torch_moe(self, *args, **kwargs):
                r = _orig_bst_load(self, *args, **kwargs)
                self.fused_mode_buffers = None
                return r

            _bst_cls.load_local = _load_torch_moe
            applied.append("MoE compute forced to torch path (EXL3_ROCM_MOE_TORCH bisect)")
        except Exception as e:
            applied.append(f"!! FAILED MoE torch patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # EXL3_ROCM_ROUTING_TORCH=1 -- bisect handle: expert routing in pure torch
    # ------------------------------------------------------------------
    # GLM-4.6V (and dots) set router_type="dots", so routing_dots runs
    # ext.routing_ds3_nogroup at *every* batch size -- a kernel built on
    # routing.cu's warp_radixsort_posf32_pl, which passes scores between lanes
    # through LDS. Dense models have no router at all, which is consistent with
    # Gemma-4-31b generating coherent text on this same build while GLM does not.
    #
    # routing_ds3 in the same module is a pure-torch implementation of the same
    # computation. GLM's config is n_group=1, topk_group=1, which collapses its
    # group mask to all-ones -- i.e. exactly the "nogroup" case the kernel
    # implements -- so it is a semantically equivalent drop-in, not an approximation.
    #
    #   coherent -> ext.routing_ds3_nogroup is the fault
    #   garbage  -> routing is exonerated and the fault is elsewhere in the
    #               glm4v_moe path (attention, RoPE, norms, architecture support)
    if _env_on("EXL3_ROCM_ROUTING_TORCH", False):
        try:
            from ..modules import block_sparse_mlp as _bsr
            _bsr_cls = _bsr.BlockSparseMLP
            _orig_bsr_load = _bsr_cls.load_local

            def _load_torch_routing(self, *args, **kwargs):
                r = _orig_bsr_load(self, *args, **kwargs)
                if getattr(self, "routing_fn", None) is _bsr.routing_dots:
                    self.routing_fn = _bsr.routing_ds3
                return r

            _bsr_cls.load_local = _load_torch_routing
            applied.append("expert routing forced to torch routing_ds3 (EXL3_ROCM_ROUTING_TORCH bisect)")
        except Exception as e:
            applied.append(f"!! FAILED routing torch patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # EXL3_ROCM_FORCE_TORCH=1 -- bisect handle: every exl3 Linear via reconstruct
    # ------------------------------------------------------------------
    # The current-tree equivalent of the fork's EXLLAMAV3_FORCE_TORCH_MODE. The fork
    # patched exl3.py directly (`bsz > 32 or FORCE_TORCH_MODE`); upstream restructured
    # that forward, so the same effect is achieved here by forcing params["reconstruct"].
    #
    # Upstream's default is rows <= AUTO_RECONSTRUCT_THRESHOLD (144) -> exl3 GEMM/GEMV
    # kernel, otherwise reconstruct + hgemm. Note the consequence: a short prompt runs
    # *prefill* through the quant kernels too, so "prefill is clean" was never evidence
    # that prefill used a different path from decode.
    #
    # Forcing it takes exl3_gemm and exl3_gemv out of the model entirely, leaving
    # dequant (reconstruct) + at::mm:
    #   coherent -> the fault is in exl3_gemm/exl3_gemv on this model's shapes
    #   garbage  -> those are exonerated; dequant, routing, attention or arch remain
    if _env_on("EXL3_ROCM_FORCE_TORCH", False):
        try:
            from ..modules.quant import exl3 as _x3
            _x3_cls = _x3.LinearEXL3
            _orig_x3_fwd = _x3_cls.forward

            def _forward_force_reconstruct(self, x, params, out_dtype = None):
                if not params.get("reconstruct"):
                    params = dict(params)
                    params["reconstruct"] = True
                return _orig_x3_fwd(self, x, params, out_dtype)

            _x3_cls.forward = _forward_force_reconstruct
            applied.append("all exl3 Linears forced through reconstruct+hgemm (EXL3_ROCM_FORCE_TORCH bisect)")
        except Exception as e:
            applied.append(f"!! FAILED force-torch patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Fused block-sparse MoE
    # ------------------------------------------------------------------
    # exl3_moe launches a NON-cooperative grid whose blocks must all be
    # co-resident for its group barriers, with nothing enforcing that. It has
    # also never been numerically validated. Forcing is_quantized False routes
    # MoE layers through the dense per-expert path.
    # Default OFF (i.e. fused MoE stays ENABLED). Measured 2026-08-07: forcing
    # is_quantized=False on EXL3-quantized tensors does not skip MoE, it reroutes
    # to a dense per-expert path that cannot handle quantized weights, and the
    # first MoE layer emits all-NaN. The fused kernel, by contrast, runs clean
    # through all 46 layers of GLM-4.6V. The fork carried this guard from an
    # older version; it is actively harmful here.
    if _env_on("EXL3_ROCM_MOE_DISABLE", False):
        try:
            from ..modules import block_sparse_mlp as _bs
            _cls = _bs.BlockSparseMLP
            # load_local is where is_quantized is computed (from the exl3 tensor
            # count), not load -- patching the wrong one silently does nothing.
            _orig_load = _cls.load_local

            def _load_no_fused_moe(self, *args, **kwargs):
                r = _orig_load(self, *args, **kwargs)
                self.is_quantized = False
                return r

            _cls.load_local = _load_no_fused_moe
            applied.append("fused block-sparse MoE disabled (exl3_moe unvalidated on RDNA)")
        except Exception as e:
            applied.append(f"!! FAILED MoE patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # v1.5.0: MoE decode for bsz <= MAX_BSZN -- keep off the cooperative kernel
    # ------------------------------------------------------------------
    # Upstream v1.5.0 replaced BC_BlockSparseMLP.run_bszN's three exl3_mgemm
    # graph launches with exl3_moe_coop, a fused decode kernel built on
    # exl3_gemv_kernel.cuh (PTX mma + cp.async). It has no RDNA sibling; the
    # ROCm build links a stub that raises if reached (rocm/quant/
    # exl3_moe_coop_rdna.hip). block_sparse_mlp.forward takes that route
    # whenever `self.bc is not None and bsz <= MAX_BSZN`, and its else-branch
    # asserts bszn_eligible, so the dispatch has to be steered from outside:
    #
    #   - MAX_BSZN is read from the module globals inside forward. It is set to
    #     0 for the duration of each call (and restored), which makes
    #     bszn_eligible false without touching the load-time buffer sizing
    #     that also reads it.
    #   - f_threshold is set to 1 after load_local, so every batch size takes
    #     the fused / per-expert branch instead of the bszN else-branch.
    #
    # self.bc is kept: the per-expert graph and dequant paths
    # (run_single_expert, run_single_expert_dq) still use it, unlike the
    # retired v1.4.4 patch above which set it to None.
    #
    # Net effect: bsz 1..8 MoE decode runs the fused exl3_moe kernel, which
    # matched an fp32 reference on this port after the 2026-08-08 split-K fixes
    # (RDNA_NOTES.md). Correct, and slower than the v1.4.4 mgemm decode route
    # that upstream removed; porting exl3_moe_coop is the item that gets it
    # back. EXL3_ROCM_MOE_BSZN=1 restores upstream dispatch for that day.
    if not _env_on("EXL3_ROCM_MOE_BSZN", False):
        try:
            from ..modules import block_sparse_mlp as _bsn
            _bsn_cls = _bsn.BlockSparseMLP
            _orig_bsn_load = _bsn_cls.load_local
            _orig_bsn_forward = _bsn_cls.forward

            def _load_no_bszn(self, *args, **kwargs):
                r = _orig_bsn_load(self, *args, **kwargs)
                self.f_threshold = 1
                return r

            def _forward_no_bszn(self, *args, **kwargs):
                saved = _bsn.MAX_BSZN
                _bsn.MAX_BSZN = 0
                try:
                    return _orig_bsn_forward(self, *args, **kwargs)
                finally:
                    _bsn.MAX_BSZN = saved

            _bsn_cls.load_local = _load_no_bszn
            _bsn_cls.forward = _forward_no_bszn
            applied.append("MoE bsz<=MAX_BSZN decode -> fused exl3_moe (exl3_moe_coop not ported to RDNA)")
        except Exception as e:
            applied.append(f"!! FAILED MoE bszN patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # RDNA4 (gfx120x): fused MoE kernel unavailable -- per-expert fallback
    # ------------------------------------------------------------------
    # The fused exl3_moe kernel's WMMA sits on gfx11 intrinsics that have no
    # gfx12 encoding; rdna_wmma.hip.h compiles a __builtin_trap() there so the
    # comp units build, and this steer keeps the trap unreachable: clearing
    # fused_mode_buffers after load_local sets min_rows = 0 in the fused
    # branch, so every expert falls through to the per-expert exl3 Linear
    # path -- the same GEMM/GEMV kernels dense models run (all of which
    # compile and select for gfx120x; verified by GPU_ARCH=gfx1201
    # hipcc_probe --all, 2026-08-28). Correct, slower on MoE models, dense
    # models unaffected. UNVALIDATED ON REAL RDNA4 HARDWARE -- compile-level
    # fix only; numerics need a gfx120x tester.
    # EXL3_ROCM_RDNA4_FUSED_MOE=1 skips this steer (future gfx12 WMMA port).
    if not _env_on("EXL3_ROCM_RDNA4_FUSED_MOE", False):
        try:
            import torch as _t
            _is_gfx12 = _t.cuda.is_available() and any(
                "gfx120" in _t.cuda.get_device_properties(i).gcnArchName
                for i in range(_t.cuda.device_count()))
            if _is_gfx12:
                from ..modules import block_sparse_mlp as _bs4
                _bs4_cls = _bs4.BlockSparseMLP
                _orig_bs4_load = _bs4_cls.load_local

                def _load_no_fused_gfx12(self, *args, **kwargs):
                    r = _orig_bs4_load(self, *args, **kwargs)
                    self.fused_mode_buffers = None
                    return r

                _bs4_cls.load_local = _load_no_fused_gfx12
                applied.append("RDNA4: fused MoE -> per-expert path (gfx11 WMMA has no gfx12 encoding)")
        except Exception as e:
            applied.append(f"!! FAILED RDNA4 MoE fallback: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # v1.5.0: sliced Q/K/V bundle -- opt-in until validated
    # ------------------------------------------------------------------
    # attn.py, sliding_attn.py and gated_delta_net.py bundle every attention
    # projection into ONE exl3_mgemm launch of equal-width column slices
    # (SlicedMultiLinear; the C++ BC attention step takes the same tables).
    # The sliced mode -- per-source input Hadamard, strided B/C rows -- is
    # ported into exl3_gemm_kernel_rdna.hip.h / exl3_gemm_inner_rdna.hip.h but
    # has not been run on RDNA. Each module gates it on its own module-level
    # `_qkv_slice_enable` (upstream env EXL3_QKV_SLICE), read at load time, so
    # clearing that flag here leaves the v1.4.4 pairwise bundles in charge.
    # Numerically both routes compute the same projections.
    if not _env_on("EXL3_ROCM_QKV_SLICE", False):
        try:
            from ..modules import attn as _sa, sliding_attn as _ss, gated_delta_net as _sg
            n = 0
            for _m in (_sa, _ss, _sg):
                if hasattr(_m, "_qkv_slice_enable"):
                    _m._qkv_slice_enable = False
                    n += 1
            applied.append(f"sliced Q/K/V bundle (SlicedMultiLinear) off in {n} modules (mgemm sliced mode unvalidated on RDNA)")
        except Exception as e:
            applied.append(f"!! FAILED QKV slice patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # v1.5.0: batched expert reconstruct tier and MoE row-tile tiers -- opt-in
    # ------------------------------------------------------------------
    # BATCH_RECON (default on upstream) dequantizes heavy experts in batches
    # through reconstruct_had_batch / hgemm_batched (moe_batch_recon.py). Both
    # kernels are mechanical batch wrappers of code that runs here, but the
    # tier is unexercised on RDNA, so the v1.4.4 per-expert loop stays default.
    #
    # MTILE makes Python issue up to three fused-MoE launches per layer, one per
    # 16/32/64-row tier. exl3_moe_rdna.hip runs every tier through the 16-row
    # instance (the only one built), so the split only adds launches.
    # Both flags are module globals read at load / call time.
    try:
        from ..modules import block_sparse_mlp as _bst2
        if not _env_on("EXL3_ROCM_BATCH_RECON", False):
            _bst2.BATCH_RECON = False
            applied.append("batched expert reconstruct tier off (unvalidated on RDNA; EXL3_ROCM_BATCH_RECON=1)")
        if not _env_on("EXL3_ROCM_MOE_MTILE", False):
            _bst2.MTILE = False
            applied.append("fused-MoE row-tile tiers off (only the 16-row instance exists on RDNA)")
    except Exception as e:
        applied.append(f"!! FAILED batch-recon/mtile patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # HIP graphs: report the effective state (the decision is made in C++)
    # ------------------------------------------------------------------
    # rocm/graph_rdna.hip gates capture/replay on hipRuntimeGetVersion of the
    # runtime actually loaded: off below 7.14 (system 7.2.x: capture hangs,
    # replays corrupt), on from 7.14 / ROCm 10 (validated). This block only
    # mirrors that decision into describe() so a log shows which mode a run
    # used; EXL3_ROCM_HIP_GRAPHS=1/0 is the override, read by both sides.
    try:
        import torch as _torch
        _hip = str(getattr(_torch.version, "hip", "") or "")
        _mm = tuple(int(x) for x in _hip.split(".")[:2]) if _hip else (0, 0)
        _ov = os.environ.get("EXL3_ROCM_HIP_GRAPHS", "").strip()
        _on = (_ov == "1") if _ov else (_mm >= (7, 14))
        applied.append(
            f"HIP graphs {'ON' if _on else 'OFF (eager passthrough)'}: HIP runtime {_hip or '?'}"
            + (", EXL3_ROCM_HIP_GRAPHS override" if _ov else ", runtime gate >= 7.14")
        )
    except Exception as e:
        applied.append(f"!! FAILED HIP graphs state report: {type(e).__name__}: {e}")

    globals()['_applied_list'] = applied
    return applied


def describe() -> str:
    return "\n".join(f"  - {p}" for p in apply()) or "  (no ROCm patches active)"
