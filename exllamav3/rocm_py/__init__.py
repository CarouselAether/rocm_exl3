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
  EXL3_ROCM_DSA_DECODE=0   DeepSeek-V4 decode attention back on the (retuned)
                           upstream split kernel instead of the MQA kernel in
                           dsa_decode_rdna.py (see the note at the patch)
  EXL3_ROCM_DSA_PREFILL=0  DeepSeek-V4 prefill attention back on the upstream
                           one-shot kernel instead of the MQA kernel in
                           dsa_prefill_rdna.py (see the note at the patch)
  EXL3_ROCM_MOE_PIPE=0     fused MoE prefill kernel on its old mainloop (the
                           shared exl3_gemm inner, 16-row tiles, one block per
                           WGP) instead of the pipelined one; read per call by
                           exl3_moe_rdna.hip. Also keeps the fused-row cap at 128
  EXL3_ROCM_MOE_FUSED_ROWS=N  fused-MoE row cap with the pipelined mainloop
                           (default 512; upstream EXL3_MOE_FUSED_ROWS wins)
  EXL3_ROCM_MOE_BPS=1      pipelined MoE kernel at one block per WGP (default:
                           two when the runtime occupancy query allows it)
  EXL3_ROCM_MOE_GROUP=N    blocks per expert group for the fused MoE kernel
                           (default MOE_SMS_PER_EXPERT = 8); set before load

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
                           default off steers it (see the next switch)
  EXL3_ROCM_MOE_MGEMM_ROUTE=0  steer bsz <= MAX_BSZN MoE decode to the fused
                           exl3_moe kernel (a 16-row tile GEMM padding one
                           useful row: ~half the decode speed). Default on =
                           the restored v1.4.4 per-token exl3_mgemm route,
                           which lands on the mgemv fast path
  EXL3_ROCM_QKV_SLICE=1    enable the one-launch sliced Q/K/V bundle
                           (SlicedMultiLinear, exl3_mgemm sliced mode). Ported
                           into the WMMA kernels, unvalidated on RDNA
  EXL3_ROCM_BATCH_RECON=1  enable the batched expert-reconstruct prefill tier
                           (reconstruct_*_batch + hgemm_batched). Ported
                           mechanically, unvalidated on RDNA
  EXL3_ROCM_MOE_MTILE=1    let Python split fused-MoE launches into 16/32/64-row
                           tiers. Pointless on RDNA, whose kernel picks the row
                           tile per expert itself
  EXL3_ROCM_MOE_BATCH=0    MoE decode back on the per-token mgemm loop (the
                           batched route runs every token of a bsz <= 8 call
                           in one set of launches)
  EXL3_ROCM_MOE_FUSED=0    batched MoE decode on four mgemm-route launches
                           instead of torch.ops.exl3_rocm.moe_decode
  EXL3_ROCM_HC_FUSE=0      mHC apply_ runs as its own launch instead of inside
                           the next site's mix
  EXL3_ROCM_HC_NORM=0      the RMSNorm after each mHC mix runs as its own launch
  EXL3_ROCM_GR_PREFILL=0   GatedResidual (Qwen3.8) prefill gate-mean back on torch ops
  EXL3_ROCM_PREFILL_HD256=0  paged prefill attention at head_dim 256 on upstream's tile
  EXL3_ROCM_BC_BUFOPS=0    graphed GQA decode attention kernels compiled without the
                           buffer-op / alignment attributes the Triton JIT would add
  (C++ side, same build: EXL3_ROCM_ROUTER_GEMV=0, EXL3_ROCM_ROUTER_FUSE=0,
  EXL3_ROCM_MR_WEIGHTED=0, EXL3_ROCM_HC_DPP=0 -- see RDNA_NOTES "Decode leftovers")

  Added at the v1.5.3 sync (2026-09-29):

  EXL3_ROCM_PREFILL_HD128=0  paged prefill attention at head_dim <= 128 on v1.5.3's
                           4-warp tile instead of the RDNA 8-warp one
  EXL3_ROCM_ROUTER_I8=1    build upstream's int8 router tables (unused on ROCm)
  EXL3_ROCM_GR_FUSED_R=N   GatedResidual fused-decode row bound (default 32, the
                           v1.5.0 value; upstream lowered it to 8 for a tiled int8
                           path that does not exist on ROCm)
  EXL3_ROCM_SMEM_LIMIT=0   leave attention_fn/smem.py's per-device budget on its
                           CUDA default (torch on ROCm has no opt-in property, so
                           it would read 96 KB for a 64 KB part)
  (C++ side: EXL3_ROCM_ROUTER_DET=0 router activations on v1.5.0 fast math;
  EXL3_ROCM_GR_DOTS=0 now selects upstream v1.5.3's GatedResidual decode pair)

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
    # v1.5.3 attention_fn/smem.py: the per-device shared-memory budget
    # ------------------------------------------------------------------
    # smem.smem_limit() reads torch's shared_memory_per_block_optin and, when it is missing,
    # guesses 64 KB below compute capability 8 and 96 KB above. torch on ROCm has no opt-in
    # property and reports gfx major 11, so every RDNA part would read 96 KB -- over the 64 KB
    # an RDNA workgroup can allocate -- and the config ladders (paged prefill / decode, DSA,
    # MLA) and bc_attn's BCKernelTooLarge gate would vet footprints the device cannot launch.
    # shared_memory_per_block (hipDeviceProp.sharedMemPerBlock, 64 KB on gfx1151) is the real
    # limit; seed smem._limit with it for every device (EXL3_TRITON_SMEM_LIMIT still caps it).
    # Before v1.5.3 there were no ladders, so the stock tiles ran as-is; they fit 64 KB, so the
    # ladders keep picking them and nothing changes unless a tile does not fit.
    # EXL3_ROCM_SMEM_LIMIT=0 leaves upstream's guess.
    if _env_on("EXL3_ROCM_SMEM_LIMIT", True):
        try:
            import torch as _tsm
            from ..modules.attention_fn import smem as _smm
            _seeded = []
            if _tsm.cuda.is_available():
                for _i in range(_tsm.cuda.device_count()):
                    _p = _tsm.cuda.get_device_properties(_i)
                    _lim = getattr(_p, "shared_memory_per_block_optin", 0) or getattr(_p, "shared_memory_per_block", 0)
                    if _lim:
                        if _smm._env_limit:
                            _lim = min(_lim, _smm._env_limit)
                        _smm._limit[_i] = _lim
                        _seeded.append(_lim)
            applied.append("attention smem budget from sharedMemPerBlock: "
                           + (", ".join(f"{v // 1024} KB" for v in _seeded) or "no devices"))
        except Exception as e:
            applied.append(f"!! FAILED smem budget patch: {type(e).__name__}: {e}")

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
    # DSA decode: MQA-specialized split kernel (rocm_py/dsa_decode_rdna.py)
    # ------------------------------------------------------------------
    # Even retuned, the upstream split kernel spills (~700 VGPRs, 2.2 KB
    # scratch/lane) and costs 160-225 us per call at ANY context -- 16% of
    # DS4 decode (PROFILE.md §4). Root cause: a loop-invariant q tile is kept
    # resident as the WMMA A operand (replicated across half-waves on RDNA3,
    # 16 x 512 per warp = 256 VGPRs), next to a BLOCK_H x 576 fp32
    # accumulator. The replacement gives each program all 64 heads of the
    # single KV head and one 128-column block of the output, streams the score
    # reduction over D, and writes the same workspace for the same combine:
    # 0 spills, ~22 us at ctx 512. The C++ launch is unchanged (BLOCK_H /
    # N_SPLITS set the grid). Layered on top of EXL3_ROCM_DSA_TUNE, which
    # still governs anything this kernel declines (Q_SPLIT / OUT_LATENT:
    # GLM-5.2's DSA-on-MLA). EXL3_ROCM_DSA_DECODE=0 restores the retuned
    # upstream kernel; EXL3_ROCM_DSA_DECODE_{SPLITS,BLOCK_H,HP,BLOCK_N,
    # BLOCK_W,KC,WARPS} override the tuning (sweep: rocm_tools/bench_dsa_decode.py).
    if _env_on("EXL3_ROCM_DSA_DECODE", True):
        try:
            from . import dsa_decode_rdna as _dsad
            applied.append(_dsad.install())
        except Exception as e:
            applied.append(f"!! FAILED DSA decode kernel patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # DSA prefill: MQA-specialized one-shot kernel (rocm_py/dsa_prefill_rdna.py)
    # ------------------------------------------------------------------
    # The upstream one-shot _dsa_attn_kernel (every DS4 prefill chunk) holds
    # a 32 x 576 fp32 accumulator next to a resident 32 x 512 q tile: 256 VGPR,
    # ~2.9K spills, ~5.2 KB scratch per lane, ~33% of pp2048 GPU time. Same
    # cure as the decode kernel: all 64 heads per program, score reduction
    # streamed over D in a runtime loop (q re-read per chunk), 16 warps so
    # the 64 x 512 accumulator is 64 VGPRs. dsa_attn looks the kernel up as a
    # module global, so a launch proxy routes eligible calls (DS4 shapes;
    # not Q_SPLIT / OUT_LATENT, i.e. GLM-5.2 keeps the old kernel) with their
    # own grid. EXL3_ROCM_DSA_PREFILL=0 restores the upstream kernel;
    # EXL3_ROCM_DSA_PREFILL_{HP,BD,KC,BLOCK_N,BLOCK_W,WARPS,KSTAGES} override
    # the tiling (sweep: rocm_tools/bench_dsa_prefill.py).
    if _env_on("EXL3_ROCM_DSA_PREFILL", True):
        try:
            from . import dsa_prefill_rdna as _dsap
            applied.append(_dsap.install())
        except Exception as e:
            applied.append(f"!! FAILED DSA prefill kernel patch: {type(e).__name__}: {e}")

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
    # v1.5.0: MoE decode for bsz <= MAX_BSZN -- the mgemm/GEMV route, restored
    # ------------------------------------------------------------------
    # Upstream v1.5.0 replaced BC_BlockSparseMLP.run_bszN's three exl3_mgemm
    # graph launches with exl3_moe_coop, a fused decode kernel built on
    # exl3_gemv_kernel.cuh (PTX mma + cp.async). It has no RDNA sibling; the
    # ROCm build links a stub that raises if reached (rocm/quant/
    # exl3_moe_coop_rdna.hip). block_sparse_mlp.forward takes that route
    # whenever `self.bc is not None and bsz <= MAX_BSZN`, and its else-branch
    # asserts bszn_eligible, so the dispatch has to be steered from outside.
    #
    # Two steers exist. The default restores upstream's own v1.4.4 route,
    # verbatim: per token, gate and up through exl3_mgemm (indices = that
    # token's experts), the activation, then down through exl3_mgemm with the
    # routing weights, which reduces the top-k expert outputs into out_d[0].
    # That row is the token's routed sum and is copied to out_bszn[i], where
    # the branch reads it back. Every call is num_tokens == 1, so on RDNA each
    # lands on the mgemv fast path (exl3_mgemv_rdna.hip: barrier-free dot
    # core, 120-240 GB/s). Measured 2026-09-22 on Laguna-S-2.1 4bpw
    # (bench_model.py): the fused steer decodes at 10.4 t/s, v1.4.4 on this
    # route at 20.8 -- the fused exl3_moe is a 16-row WMMA tile GEMM that pads
    # one useful row to sixteen and runs at a flat ~53 GB/s (RDNA_NOTES.md,
    # "Token generation was NOT memory-bound as first shipped").
    #
    # Mechanics: forward is wrapped so that, for the duration of the call,
    # self.bc is a proxy whose run_bszN is the Python loop below; every other
    # attribute forwards to the real BC_BlockSparseMLP, so bszn_eligible still
    # sees a non-None bc and the per-expert graph paths (bsz > MAX_BSZN) still
    # reach the C++ object. Shared experts: upstream's kernel merges them in-
    # kernel and the forward tail skips them while self.bc_sh_exp is True, so
    # bc_sh_exp is forced False after load_local and the tail's Python
    # shared-expert path runs instead, as it does for every non-bszN tier.
    # Expert-range shards pass cfg.min_expert / max_expert exactly as
    # run_bszN does (num_tokens == 1 compacts out-of-range picks).
    #
    # EXL3_ROCM_MOE_MGEMM_ROUTE=0 selects the older steer: MAX_BSZN is zeroed
    # for the call and f_threshold set to 1, so bsz 1..8 runs the fused
    # exl3_moe kernel (correct per the 2026-08-08 fp32 comparison, half the
    # decode speed). EXL3_ROCM_MOE_BSZN=1 leaves upstream dispatch alone
    # (raises on ROCm) for the day exl3_moe_coop is ported.
    if not _env_on("EXL3_ROCM_MOE_BSZN", False):
        try:
            import torch
            from ..modules import block_sparse_mlp as _bsn
            from ..ext import exllamav3_ext as _ext
            _bsn_cls = _bsn.BlockSparseMLP
            _orig_bsn_load = _bsn_cls.load_local
            _orig_bsn_forward = _bsn_cls.forward

            if _env_on("EXL3_ROCM_MOE_MGEMM_ROUTE", True):

                # EXL3_ROCM_MOE_BATCH (default on, 2026-09-28; RDNA_NOTES "Decode
                # leftovers"): every token of the call in one set of launches instead of
                # the per-token loop below. The slots are flattened (token-major,
                # S = bsz * top_k): gate and up run once over S slots (the input rows
                # replicated per slot by one copy at bsz > 1; broadcast at bsz 1), the
                # activation over the S valid rows only (not all MAX_BSZN * top_k), and
                # down once with num_tokens = bsz, whose grouped weighted reduce leaves
                # token t's routed sum in out_d row t. out_bszn is then a view of those
                # rows (when the down width is the hidden width), so the per-token
                # copy_ goes too. At bsz 1 that is 5 launches per layer instead of 7 and
                # the same arithmetic (bit-identical); at bsz 3 (MTP verify) 6 instead of
                # 21. num_tokens is also passed to gate/up (no reduce there): the RDNA
                # GEMV sizes its split-K from the per-token slot count, so each slot keeps
                # the m == 1 reduction order and verify rows stay bit-identical to plain
                # decode. Expert-range shards (min_expert >= 0) keep the loop.
                _moe_batch = _env_on("EXL3_ROCM_MOE_BATCH", True)

                # EXL3_ROCM_MOE_FUSED (default on, 2026-09-28): the same block through
                # torch.ops.exl3_rocm.moe_decode (rocm/quant/exl3_gemv_multirow_rdna.hip):
                # gate and up as one 2S-slot GEMV, the activation folded into the down
                # projection's input rotation -- 4 launches instead of 7 (bsz 1) / 8
                # (bsz > 1), same arithmetic. Set up at load for gated SILU experts with
                # matching gate/up K and codebook and fp16 intermediates; anything else
                # stays on the mgemm route above.
                _moe_fused = _env_on("EXL3_ROCM_MOE_FUSED", True)

                def _mgemm_bszN_batched(mod, y, selected_experts, routing_weights):
                    cfg = mod.experts_cfg
                    bsz = y.shape[0]
                    top_k = selected_experts.shape[-1]
                    S = bsz * top_k
                    fz = mod._rocm_fused
                    if fz is not None and S <= fz["max_s"] and y.is_contiguous():
                        torch.ops.exl3_rocm.moe_decode(
                            y, selected_experts, routing_weights,
                            fz["gu_trellis"], fz["gu_suh"], fz["gu_svh"],
                            mod.multi_down.ptrs_trellis, mod.multi_down.ptrs_suh, mod.multi_down.ptrs_svh,
                            fz["yh"], fz["gu"], cfg.interm_a, cfg.out_d,
                            fz["K_gu"], fz["cb_gu"], fz["K_d"], fz["cb_d"], fz["act_limit"])
                        if not mod._rocm_out_alias:
                            width = cfg.out_bszn.shape[-1]
                            cfg.out_bszn[:bsz].copy_(cfg.out_d.view(cfg.out_d.shape[0], -1)[:bsz, :width])
                        return
                    mg, mu, md = mod.multi_gate, mod.multi_up, mod.multi_down
                    sel = selected_experts.view(1, S)
                    w = routing_weights.view(1, S)
                    if bsz == 1:
                        A = y.view(1, 1, -1)
                    else:
                        Hi = y.shape[-1]
                        A = mod._rocm_arep[:S]
                        A.view(bsz, top_k, Hi).copy_(y.view(bsz, 1, Hi).expand(bsz, top_k, Hi))
                    ig, iu, ia = cfg.interm_g[:S], cfg.interm_u[:S], cfg.interm_a[:S]
                    if mod.gated:
                        _ext.exl3_mgemm(
                            A, mg.ptrs_trellis, ig, mg.ptrs_suh, cfg.yh, mg.ptrs_svh,
                            sel, None, mg.K, -1, mg.mcg, mg.mul1, -1, -1, 0, bsz, None, None)
                    _ext.exl3_mgemm(
                        A, mu.ptrs_trellis, iu, mu.ptrs_suh, cfg.yh, mu.ptrs_svh,
                        sel, None, mu.K, -1, mu.mcg, mu.mul1, -1, -1, 0, bsz, None, None)
                    mod.activation_fn_call(ig if mod.gated else iu, iu, ia, mod.act_limit)
                    _ext.exl3_mgemm(
                        ia, md.ptrs_trellis, cfg.out_d, md.ptrs_suh, cfg.interm_g, md.ptrs_svh,
                        sel, w, md.K, -1, md.mcg, md.mul1, -1, -1, 0, bsz, None, None)
                    if not mod._rocm_out_alias:
                        width = cfg.out_bszn.shape[-1]
                        cfg.out_bszn[:bsz].copy_(cfg.out_d.view(cfg.out_d.shape[0], -1)[:bsz, :width])

                def _mgemm_bszN(mod, y, selected_experts, routing_weights):
                    cfg = mod.experts_cfg
                    bsz = y.shape[0]
                    mine, maxe = cfg.min_expert, cfg.max_expert
                    if _moe_batch and mine < 0:
                        return _mgemm_bszN_batched(mod, y, selected_experts, routing_weights)
                    A = y.unsqueeze(1).unsqueeze(1)          # (bsz, 1, 1, Hi)
                    sel = selected_experts.unsqueeze(1)      # (bsz, 1, top_k)
                    w = routing_weights.unsqueeze(1)         # (bsz, 1, top_k)
                    width = cfg.out_bszn.shape[-1]
                    out_row = cfg.out_d[0].view(-1)[:width]  # routed sum lands in row 0
                    mg, mu, md = mod.multi_gate, mod.multi_up, mod.multi_down
                    for i in range(bsz):
                        if mod.gated:
                            _ext.exl3_mgemm(
                                A[i], mg.ptrs_trellis, cfg.interm_g, mg.ptrs_suh, cfg.yh, mg.ptrs_svh,
                                sel[i], None, mg.K, -1, mg.mcg, mg.mul1, mine, maxe, 0, 1, None, None)
                        _ext.exl3_mgemm(
                            A[i], mu.ptrs_trellis, cfg.interm_u, mu.ptrs_suh, cfg.yh, mu.ptrs_svh,
                            sel[i], None, mu.K, -1, mu.mcg, mu.mul1, mine, maxe, 0, 1, None, None)
                        act_g = cfg.interm_g if mod.gated else cfg.interm_u
                        mod.activation_fn_call(act_g, cfg.interm_u, cfg.interm_a, mod.act_limit)
                        # A_had must not alias A (the autotuner relaunches on the first call);
                        # the gate buffer is free after the activation
                        _ext.exl3_mgemm(
                            cfg.interm_a, md.ptrs_trellis, cfg.out_d, md.ptrs_suh, cfg.interm_g, md.ptrs_svh,
                            sel[i], w[i], md.K, -1, md.mcg, md.mul1, mine, maxe, 0, 1, None, None)
                        cfg.out_bszn[i].copy_(out_row)

                class _BCProxy:
                    __slots__ = ("_bc", "_mod")

                    def __init__(self, bc, mod):
                        self._bc = bc
                        self._mod = mod

                    def __getattr__(self, name):
                        return getattr(self._bc, name)

                    def run_bszN(self, y, selected_experts, routing_weights):
                        _mgemm_bszN(self._mod, y, selected_experts, routing_weights)

                def _load_mgemm_route(self, *args, **kwargs):
                    r = _orig_bsn_load(self, *args, **kwargs)
                    self.bc_sh_exp = False
                    self._rocm_out_alias = False
                    self._rocm_fused = None
                    cfg = getattr(self, "experts_cfg", None)
                    # (min_expert >= 0 keeps the per-token loop, whose copy_ into
                    # out_bszn[i] would land on the next token's scratch rows)
                    if _moe_batch and cfg is not None and cfg.out_bszn is not None \
                            and cfg.min_expert < 0:
                        import torch
                        from ..util.tensor import g_tensor_cache
                        rows = cfg.interm_g.shape[0]
                        Hi = self.multi_up.in_features if self.multi_up is not None else None
                        if Hi is not None:
                            self._rocm_arep = g_tensor_cache.get(
                                self.device, (rows, 1, Hi), torch.half, "rocm_moe_arep")
                        # Routed sums land in out_d rows 0..bsz-1: read them in place when
                        # the rows are exactly the hidden width
                        od = cfg.out_d.view(cfg.out_d.shape[0], -1)
                        H = cfg.out_bszn.shape[-1]
                        if od.shape[-1] == H and od.dtype == cfg.out_bszn.dtype \
                                and od.shape[0] >= cfg.out_bszn.shape[0]:
                            cfg.out_bszn = od[:cfg.out_bszn.shape[0]]
                            self._rocm_out_alias = True
                        self._rocm_fused = _moe_fused_setup(self, cfg, rows, Hi) if _moe_fused else None
                    return r

                def _moe_fused_setup(self, cfg, rows, Hi):
                    import torch
                    from ..util.tensor import g_tensor_cache
                    mg, mu, md = self.multi_gate, self.multi_up, self.multi_down
                    if not (self.gated and mg is not None and mu is not None and md is not None):
                        return None
                    if self.activation_fn_call is not _ext.silu_mul:
                        return None
                    if not hasattr(torch.ops, "exl3_rocm") or not hasattr(torch.ops.exl3_rocm, "moe_decode"):
                        return None
                    if mg.K != mu.K or bool(mg.mcg) != bool(mu.mcg) or bool(mg.mul1) != bool(mu.mul1):
                        return None
                    # v1.5.3 half-integer bitrates (LinearEXL3.K 1.5 / 2.5 / 3.5): the multi-row
                    # GEMV cores decode integer K only; those layers stay on the mgemm route,
                    # whose exl3_mgemm runs them on the cooperative kernel
                    if not all(float(ml.K).is_integer() for ml in (mg, mu, md)):
                        return None
                    if any(getattr(l.inner, "bias", None) is not None for ml in (mg, mu, md) for l in ml.linears):
                        return None
                    I = cfg.interm_a.shape[-1]
                    if cfg.interm_a.dtype != torch.half or cfg.interm_g.dtype != torch.half:
                        return None
                    if cfg.out_d.dtype != torch.float or Hi % 128 or I % 128 or cfg.out_d.shape[-1] % 128:
                        return None
                    cbk = lambda ml: 2 if ml.mul1 else (1 if ml.mcg else 0)
                    # Slot bound: the scratch (2 * rows gate|up slots) and the op's arrival
                    # counters (exl3_gemv_multirow_rdna.hip: 2S <= 128, 2S * I/128 and S * Ho/128
                    # within EXL3_MGEMV_SEG_COUNTERS = 128 * 256). Larger calls -- Qwen3.8's
                    # top_k 10 at 7..8 rows, reached by v1.5.3's model.warmup "rows 8" pass and
                    # by 7..8-token prefill chunks -- take the mgemm route instead of raising
                    seg = 128 * 256
                    max_s = min(rows, 64, seg // (2 * (I // 128)), seg // (cfg.out_d.shape[-1] // 128))
                    return {
                        "rows": 2 * rows,
                        "max_s": max_s,
                        "gu_trellis": torch.cat([mg.ptrs_trellis, mu.ptrs_trellis]).contiguous(),
                        "gu_suh": torch.cat([mg.ptrs_suh, mu.ptrs_suh]).contiguous(),
                        "gu_svh": torch.cat([mg.ptrs_svh, mu.ptrs_svh]).contiguous(),
                        "yh": g_tensor_cache.get(self.device, (2 * rows, Hi), torch.half, "rocm_moe_yh2"),
                        "gu": g_tensor_cache.get(self.device, (2 * rows, I), torch.half, "rocm_moe_gu"),
                        "K_gu": int(mg.K), "cb_gu": cbk(mg), "K_d": int(md.K), "cb_d": cbk(md),
                        "act_limit": float(self.act_limit or 0.0),
                    }

                def _forward_mgemm_route(self, *args, **kwargs):
                    bc = self.bc
                    # Only modules set up by _load_mgemm_route (which sets _rocm_out_alias) take
                    # the proxy; anything else -- e.g. upstream's test_moe_shared_schedule, which
                    # drives forward on a stand-in object with a fake bc -- runs upstream verbatim
                    if bc is None or not hasattr(self, "_rocm_out_alias"):
                        return _orig_bsn_forward(self, *args, **kwargs)
                    self.bc = _BCProxy(bc, self)
                    try:
                        return _orig_bsn_forward(self, *args, **kwargs)
                    finally:
                        self.bc = bc

                # v1.5.3 BC_BlockSparseMLP sh_coop: the shared expert as a one-expert exl3_moe_coop
                # launch inside run_bszN. Never run here (run_bszN is the proxy above and the
                # tail's Python shared expert runs, bc_sh_exp False), and exl3_moe_coop is a stub
                # on ROCm; off, so the constructor does not build its parameter block and scratch
                if hasattr(_bsn, "_moe_shared_coop"):
                    _bsn._moe_shared_coop = False
                _bsn_cls.load_local = _load_mgemm_route
                _bsn_cls.forward = _forward_mgemm_route
                applied.append("MoE bsz<=MAX_BSZN decode -> per-token exl3_mgemm route (v1.4.4's; mgemv fast path; exl3_moe_coop not ported)")

            else:

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
                applied.append("MoE bsz<=MAX_BSZN decode -> fused exl3_moe (EXL3_ROCM_MOE_MGEMM_ROUTE=0)")
        except Exception as e:
            applied.append(f"!! FAILED MoE bszN patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Hyper-connections: apply_ folded into the next site's mix (2026-09-28)
    # ------------------------------------------------------------------
    # DS4 runs 86 mHC sites per decoded token, each HyperConnection.apply_
    # (one hc_apply launch, in place on the residual streams) directly followed
    # by the next site's mix (partials + finalize) on the same stream tensor --
    # attn site -> the block's mlp site, mlp site -> the next block's attn site.
    # EXL3_ROCM_HC_FUSE (default on) defers the apply at decode-class row counts
    # (R <= 32) and hands it to that mix, which runs torch.ops.exl3_rocm.
    # hc_mix_rdna: the apply fused into the partials kernel (rocm/hc_mix_rdna.hip),
    # one launch less per site, bit-identical to apply + mix.
    #
    # EXL3_ROCM_HC_NORM (default on): the RMSNorm a TransformerBlock runs on the
    # collapsed stream right after each mix (attn_norm / mlp_norm) is folded into
    # the mix's finalize kernel -- its last-arriving block per row replays the norm
    # kernel's arithmetic bit for bit -- and the norm's own forward then returns
    # that pre-normed buffer untouched (recognised by its data pointer, one-shot).
    # Another launch less per site.
    #
    # The deferred apply is flushed (run as the plain hc_apply) before anything
    # else can see the streams: any other HyperConnection call, HyperHead.forward
    # (the final collapse), and prepare_for_device when the tensor would change
    # device. Nothing is deferred while states are being exported or during
    # conversion (quant_preserve / capture), and a mix on any tensor other than
    # the pending one flushes first.
    if _env_on("EXL3_ROCM_HC_FUSE", True):
        try:
            import torch
            from ..modules import hyperconnections as _hcm
            from ..modules import module as _modm
            from ..ext import exllamav3_ext as _ext
            from ..util.tensor import g_tensor_cache as _gtc
            _ = torch.ops.exl3_rocm.hc_mix_rdna   # raises if the build lacks the op
            from ..modules import transformer as _trm
            from ..modules import rmsnorm as _rmsm
            _hc_norm_on = _env_on("EXL3_ROCM_HC_NORM", True)

            _HC = _hcm.HyperConnection
            _orig_hc_apply = _HC.apply_
            _orig_hc_mix = _HC.mix
            _orig_head_fwd = _hcm.HyperHead.forward
            _orig_prep = _modm.Module.prepare_for_device
            _hc_pend = [None]   # (x, y, post, comb) of the deferred apply

            def _hc_flush():
                p = _hc_pend[0]
                if p is None:
                    return
                _hc_pend[0] = None
                x, y, post, comb = p
                b, s_, H, D = x.shape
                R = b * s_
                _ext.hc_apply(x.view(R, H, D), y.view(R, D), post.view(R, H), comb.view(R, H, H))

            def _hc_apply_rdna(self, x, y, post, comb, params):
                _hc_flush()
                b, s_, H, D = x.shape
                if b * s_ <= 32 and H == 4 and D % 4 == 0 \
                        and "quant_preserve" not in params and "capture" not in params \
                        and not params.get("export_state_layers") \
                        and x.dtype == torch.float and x.is_contiguous() \
                        and y.dtype == torch.float and y.is_contiguous() \
                        and post.dtype == torch.float and post.is_contiguous() and comb.is_contiguous():
                    _hc_pend[0] = (x, y, post, comb)
                    return x
                return _orig_hc_apply(self, x, y, post, comb, params)

            def _hc_mix_rdna(self, streams, params):
                p = _hc_pend[0]
                b, s_, H, D = streams.shape
                R = b * s_
                fusable = self.hc_mult == 4 and R <= 32 and streams.dtype == torch.float \
                    and D % 4 == 0 and streams.is_contiguous()
                if p is not None and not (fusable and p[0] is streams):
                    _hc_flush()
                    p = None
                nrm = getattr(self, "_rocm_norm", None) if fusable else None
                if nrm is not None and (D // 4 > 1024 or (nrm.weight is not None and (
                        nrm.weight.dtype not in (torch.bfloat16, torch.half) or nrm.weight.numel() != D))):
                    nrm = None
                if p is None and nrm is None:
                    return _orig_hc_mix(self, streams, params)
                _hc_pend[0] = None
                chunks = _ext.hc_mix_num_chunks(R, H * D)
                M1 = 2 * H + H * H + 1
                dev = streams.device
                partials = _gtc.get_bucketed(dev, R * chunks * M1, torch.float, "hc_mix_partials").view(R, chunks, M1)
                post = _gtc.get_bucketed(dev, R * H, torch.float, "hc_post").view(R, H)
                comb = _gtc.get_bucketed(dev, R * H * H, torch.float, "hc_comb").view(R, H, H)
                collapsed = _gtc.get_bucketed(dev, R * D, torch.half, "hc_coll").view(R, D)
                if self.fn_h is None:
                    self.fn_h = self.fn.half()
                if p is not None:
                    _, y, post_a, comb_a = p
                    y, post_a, comb_a = y.view(R, D), post_a.view(R, H), comb_a.view(R, H, H)
                else:
                    y = post_a = comb_a = None
                if nrm is not None:
                    normed = _gtc.get_bucketed(dev, R * D, torch.half, "hc_normed").view(R, D)
                    torch.ops.exl3_rocm.hc_mix_rdna(
                        streams.view(R, H, D), y, post_a, comb_a,
                        self.fn_h, self.base, self.scale, self.rms_eps, self.hc_eps, self.sinkhorn_iters,
                        partials, post, comb, collapsed,
                        nrm.weight, normed, nrm.rms_norm_eps, nrm.constant_bias, nrm.constant_scale)
                    nrm._rocm_prenormed = normed.data_ptr()
                    out = normed
                else:
                    torch.ops.exl3_rocm.hc_mix_rdna(
                        streams.view(R, H, D), y, post_a, comb_a,
                        self.fn_h, self.base, self.scale, self.rms_eps, self.hc_eps, self.sinkhorn_iters,
                        partials, post, comb, collapsed,
                        None, None, 0.0, 0.0, 1.0)
                    out = collapsed
                return post.view(b, s_, H), comb.view(b, s_, H, H), out.view(b, s_, D)

            def _hc_head_fwd_rdna(self, x, params, out_dtype = None):
                _hc_flush()
                return _orig_head_fwd(self, x, params, out_dtype)

            def _prep_rdna(self, x, params):
                if _hc_pend[0] is not None and x.device != self.device:
                    _hc_flush()
                return _orig_prep(self, x, params)

            # Norm fold: link each block's HC site to the RMSNorm it runs next, and let
            # that norm pass the pre-normed buffer through
            if _hc_norm_on:
                _orig_tb_init = _trm.TransformerBlock.__init__
                _orig_rms_fwd = _rmsm.RMSNorm.forward
                _rmsm.RMSNorm._rocm_prenormed = None

                def _norm_foldable(n):
                    return isinstance(n, _rmsm.RMSNorm) and not n.span_heads \
                        and getattr(n, "groups", 1) == 1

                def _tb_init_rdna(self, *args, **kwargs):
                    _orig_tb_init(self, *args, **kwargs)
                    for hc, nm in ((self.attn_hc, self.attn_norm), (self.mlp_hc, self.mlp_norm)):
                        if isinstance(hc, _HC) and _norm_foldable(nm):
                            hc._rocm_norm = nm

                def _rms_fwd_rdna(self, x, params, out_dtype = None, residual = None, residual_in = None):
                    pn = self._rocm_prenormed
                    if pn is not None:
                        self._rocm_prenormed = None
                        if pn == x.data_ptr() and residual is None and residual_in is None \
                                and x.dtype == torch.half and (out_dtype or self.out_dtype or torch.half) == torch.half \
                                and (self.weight is None or self.weight.dtype in (torch.bfloat16, torch.half)):
                            if self.key in params.get("export_state_norm_keys", ()):
                                states = params.get("export_states")
                                if states is None:
                                    states = params["export_states"] = []
                                states.append(x.half())
                            return x
                    return _orig_rms_fwd(self, x, params, out_dtype, residual, residual_in)

                _trm.TransformerBlock.__init__ = _tb_init_rdna
                _rmsm.RMSNorm.forward = _rms_fwd_rdna
                applied.append("mHC mix + following RMSNorm in one finalize at R <= 32 (EXL3_ROCM_HC_NORM)")

            _HC.apply_ = _hc_apply_rdna
            _HC.mix = _hc_mix_rdna
            _hcm.HyperHead.forward = _hc_head_fwd_rdna
            _modm.Module.prepare_for_device = _prep_rdna
            applied.append("mHC apply_ folded into the next site's mix at R <= 32 (EXL3_ROCM_HC_FUSE)")
        except Exception as e:
            applied.append(f"!! FAILED HC fuse patch: {type(e).__name__}: {e}")

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
    # 16/32/64-row tier. On RDNA the fused kernel picks its own row tile per
    # expert (pipelined mainloop: 16/32/48/64 rows; old mainloop: 16 only), so
    # the split only adds launches.
    #
    # Fused-row cap: experts with more rows than TEMP_ROWS_FUSED (upstream 128)
    # leave the fused kernel for the reconstruct + hgemm tier. The pipelined
    # mainloop (EXL3_ROCM_MOE_PIPE, default on) runs up to 64 rows per B decode,
    # so it takes hot experts cheaper than reconstructing them: 512 measured DS4
    # pp2048 343.8 -> 371.2 t/s (+8%), pp512 +2% (RDNA_NOTES "Pipelined MoE
    # mainloop"). Upstream's EXL3_MOE_FUSED_ROWS, when set, wins; with the old
    # mainloop (EXL3_ROCM_MOE_PIPE=0) the upstream 128 stays.
    # All three are module globals read at load / call time.
    try:
        from ..modules import block_sparse_mlp as _bst2
        if not _env_on("EXL3_ROCM_BATCH_RECON", False):
            _bst2.BATCH_RECON = False
            applied.append("batched expert reconstruct tier off (unvalidated on RDNA; EXL3_ROCM_BATCH_RECON=1)")
        if not _env_on("EXL3_ROCM_MOE_MTILE", False):
            _bst2.MTILE = False
            applied.append("fused-MoE row-tile tiers off (the RDNA kernel tiles rows itself)")
        if _env_on("EXL3_ROCM_MOE_PIPE", True) and "EXL3_MOE_FUSED_ROWS" not in os.environ:
            _bst2.TEMP_ROWS_FUSED = int(os.environ.get("EXL3_ROCM_MOE_FUSED_ROWS", 512))
            applied.append(f"fused-MoE row cap {_bst2.TEMP_ROWS_FUSED} (pipelined mainloop; EXL3_ROCM_MOE_FUSED_ROWS / EXL3_MOE_FUSED_ROWS)")
    except Exception as e:
        applied.append(f"!! FAILED batch-recon/mtile patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Graphed GQA decode attention: AMD buffer-op specialization for AOT kernels
    # ------------------------------------------------------------------
    # bc_attn._compile_kernel builds the graphed attention kernels ahead of time
    # from a hand-written signature. The Triton JIT on AMD specializes every tensor
    # argument whose storage is < 2 GB as tt.pointer_range = 32 (buffer loads with
    # 32-bit offsets off an SGPR base) and every 16-byte-aligned pointer as
    # tt.divisibility = 16; the AOT signature carries neither, so the graphed
    # _paged_attn_decode_split_kernel compiles to 64-bit per-lane address math,
    # 256 VGPRs + 660 B scratch: 91.5 us per call on Qwen3.8's full-attention
    # layers at ctx 512 (24 q / 2 kv heads, hd 256), where the JIT-launched same
    # kernel takes 13.5 us. Adding both attributes: 45 -> 14 us (q_len 1), 73 -> 23 us
    # (q_len 3, MTP verify) in isolation (RDNA_NOTES "Qwen3.8"). With the attributes
    # the kernels also compile scratch-free at 8 warps / 1 stage (4 / 2 still
    # spilled): in-model ctx 512 91.5 -> 31.8 us, QSA sparse at 8K 117 -> 31.9 us,
    # MTP verify (q_len 3) 163 -> 62 us. EXL3_ROCM_BC_ATTN_WARPS / _STAGES override.
    # Scope: only the GQA decode split kernels below (paged split + QSA sparse
    # split), only when this layer's K/V cache storages are < 2 GB (checked in
    # BCAttn._configure); the pointers declared 16-byte aligned are the slot's
    # static buffers (q / o / partials from g_tensor_cache, offset 0) and the
    # per-layer caches. The attributes change addressing only; the warp count
    # changes the dot layout (see RDNA_NOTES for the bitwise result). DS4's DSA /
    # MLA kernels are not in the list.
    # EXL3_ROCM_BC_BUFOPS=0 restores the plain AOT signature.
    if _env_on("EXL3_ROCM_BC_BUFOPS", True):
        try:
            import torch
            from ..modules.attention_fn import bc_attn as _bca
            from ..ext import exllamav3_ext as _ext
            _prev_ck = _bca._compile_kernel
            _BUFOPS_ALIGNED = {
                "_paged_attn_decode_split_kernel":
                    ("q", "k_cache", "v_cache", "out", "partial_o", "partial_ml"),
                "_qsa_sparse_split_kernel":
                    ("q", "k_cache", "v_cache", "partial_o", "partial_ml"),
            }
            _bufops_state = {"ok": False}
            _bufops_cache = {}

            def _compile_kernel_bufops(device, fn, signature, constexprs, num_warps, num_stages):
                name = getattr(fn, "__name__", None)
                if name not in _BUFOPS_ALIGNED or not _bufops_state["ok"]:
                    return _prev_ck(device, fn, signature, constexprs, num_warps, num_stages)
                # 8 warps / 1 stage: 0 scratch for both kernels (4 / 2 kept 276-524 B);
                # in-model sweep 4/2, 8/2, 8/1, 16/1, 4/1, 2/2 in RDNA_NOTES
                num_warps = int(os.environ.get("EXL3_ROCM_BC_ATTN_WARPS", 8))
                num_stages = int(os.environ.get("EXL3_ROCM_BC_ATTN_STAGES", 1))
                key = (device.index, name, tuple(sorted(constexprs.items())), num_warps,
                       num_stages, tuple(sorted(signature.items())))
                k = _bufops_cache.get(key)
                if k is None:
                    import triton
                    from triton.compiler import ASTSource
                    attrs, sig = {}, {}
                    for n, ty in signature.items():
                        t = ty[:-3] if isinstance(ty, str) and ty.endswith(":16") else ty
                        sig[n] = t
                        a = []
                        if t != ty or n in _BUFOPS_ALIGNED[name]:
                            a.append(["tt.divisibility", 16])
                        if isinstance(t, str) and t.startswith("*"):
                            a.append(["tt.pointer_range", 32])
                        if a:
                            attrs[(fn.arg_names.index(n),)] = a
                    with torch.cuda.device(device):
                        src = ASTSource(fn = fn, signature = sig, constexprs = constexprs, attrs = attrs)
                        ck = triton.compile(src, options = {"num_warps": num_warps, "num_stages": num_stages})
                        # v1.5.3: same shared-memory gate as upstream's _compile_kernel (the
                        # caller catches BCKernelTooLarge and takes the eager path)
                        _lim = _bca.smem_limit(device)
                        if ck.metadata.shared > _lim:
                            raise _bca.BCKernelTooLarge(
                                f"{name}: {ck.metadata.shared} B of shared memory exceeds the device's {_lim} B")
                        k = _ext.TritonKernel(ck.asm["cubin"], ck.metadata.name,
                                              ck.metadata.num_warps, ck.metadata.shared)
                    _bufops_cache[key] = k
                return k

            _orig_bca_configure = _bca.BCAttn._configure

            def _bca_configure_rdna(self, *a, **kw):
                lim = 2**31 - 1
                ts = [t for t in (self.cache_k, self.cache_v, getattr(self, "k_scales", None),
                                  getattr(self, "v_scales", None)) if isinstance(t, torch.Tensor)]
                _bufops_state["ok"] = all(t.untyped_storage().nbytes() <= lim for t in ts)
                try:
                    return _orig_bca_configure(self, *a, **kw)
                finally:
                    _bufops_state["ok"] = False

            _bca._compile_kernel = _compile_kernel_bufops
            _bca.BCAttn._configure = _bca_configure_rdna
            applied.append("graphed GQA decode attention: buffer-op / alignment specialization (EXL3_ROCM_BC_BUFOPS)")
        except Exception as e:
            applied.append(f"!! FAILED BC bufops patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Router int8 tables (v1.5.3 _gate_t): not built on ROCm
    # ------------------------------------------------------------------
    # block_sparse_mlp_routing._gate_t now builds, next to the transposed gate, an int8 hi/lo
    # copy + row scales (ext.det_quant_weight) for the deterministic multi-row router GEMM
    # (routing_gemm.cu). That GEMM is sm_80 PTX; on ROCm rocm/routing_gemm_rdna.hip declines it
    # (routing_gemm_det_fits -> False), so the tables would be dead weight: 2 * E * K bytes per
    # MoE layer (DS4 ~90 MB, Qwen3.8 ~126 MB) plus a quantize launch per router at first use.
    # This keeps _gate_t to the transposed gate; with gate_i8 None the ext routing takes its
    # non-det path, which is what it takes on ROCm anyway -- same kernels, same results.
    # EXL3_ROCM_ROUTER_I8=1 restores upstream's _gate_t.
    #
    # (Retired here: EXL3_ROCM_ROUTER_STD_MR. It passed gate_t to ext.routing_std at bsz 2..8 so
    # the RDNA multi-row router GEMV served MTP verify rows. v1.5.3's routing_std passes gate_t at
    # every bsz itself, so upstream now reaches the same kernel with no hook.)
    if not _env_on("EXL3_ROCM_ROUTER_I8", False):
        try:
            from ..modules import block_sparse_mlp_routing as _bsr
            _orig_gate_t = _bsr._gate_t

            def _gate_t_rdna(cfg):
                if cfg.gate_tensor_t is None:
                    cfg.gate_tensor_t = cfg.gate_tensor.T.contiguous()
                return cfg.gate_tensor_t

            _bsr._gate_t = _gate_t_rdna
            applied.append("router int8 tables not built (det router GEMM is sm_80 PTX, declined on ROCm; EXL3_ROCM_ROUTER_I8)")
        except Exception as e:
            applied.append(f"!! FAILED router int8 patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Paged prefill attention tiles on RDNA: head_dim 256 and head_dim <= 128
    # ------------------------------------------------------------------
    # head_dim 256 (EXL3_ROCM_PREFILL_HD256): paged_attn_triton_prefill's head_dim-256 tile
    # (block_m 64, block_n 32, 8 warps, 2 stages at v1.5.0; 4 warps from v1.5.3) spills on
    # gfx1151 (256 VGPR + 772 B scratch). One stage frees the
    # registers; at q_len >= 128 a 128-row tile (still 16 rows per warp, the
    # rule noted at the config) halves the K/V re-reads. Qwen3.8 shapes (24 q / 2 kv
    # heads, fp16 cache), us: q 1792 fresh 5770 -> 2917; q 255 after 1792 1621 -> 884;
    # q 2048 after 2048 20017 -> 8043; q 2048 after 16384 121711 -> 39755;
    # q 64 after 4096 1098 -> 612 (64-row tile). Error vs an fp32 reference unchanged
    # (1.15e-3 / 4.2e-5). Scope: head_dim 129..256, fp16 cache (qc None), no explicit
    # tile from the caller. DS4 (DSA prefill, D 512) never reaches this; Gemma's
    # 256-dim layers do (PPL gate). EXL3_ROCM_PREFILL_HD256=0 restores upstream.
    #
    # head_dim <= 128 (EXL3_ROCM_PREFILL_HD128, v1.5.3): upstream moved this tile from
    # (128, 32, 8, 2) -- the value the ROCm fork had selected explicitly via an in-file
    # _is_rocm edit to triton_paged.py (gfx1151: BN 32 over 64, 13.06 vs 11.67 TFLOP/s at
    # q_len 2048; rocm_tools/bench_prefill_tiles.py) -- to 4 warps (issue #384: whole 16-row MMA
    # tiles per warp on NVIDIA) and, because ROCm's get_device_capability() reports gfx major 11
    # and so trips the Blackwell test, 3 stages: (128, 32, 4, 3). That is 32 rows per warp, off
    # the RDNA rule (block_m / num_warps == 16; off-ratio configs measured up to 4x slower), and
    # unmeasured here. The in-file edit is gone (triton_paged.py is upstream's verbatim); this
    # restores the fork's (128, 8 warps, 2 stages) for fp16 and quantized caches alike (block_n
    # stays the upstream pick: 32, or the quantized-cache width). Not reached by DS4 / Qwen3.8 /
    # Gemma (head_dim 512 / 256 / 256+512). EXL3_ROCM_PREFILL_HD128=0 takes v1.5.3's tile.
    _pf_hd256 = _env_on("EXL3_ROCM_PREFILL_HD256", True)
    _pf_hd128 = _env_on("EXL3_ROCM_PREFILL_HD128", True)
    if _pf_hd256 or _pf_hd128:
        try:
            import triton
            from ..modules.attention_fn import triton_paged as _tpm
            from ..modules import sliding_attn as _swm
            _orig_prefill = _tpm.paged_attn_triton_prefill

            def _prefill_rdna(*a, **kw):
                q = kw.get("q", a[0] if a else None)
                if q is not None and kw.get("block_m") is None \
                        and kw.get("block_n") is None and kw.get("num_warps") is None \
                        and kw.get("num_stages") is None and q.dim() == 4:
                    hd_pad = triton.next_power_of_2(q.shape[-1])
                    if _pf_hd256 and kw.get("qc") is None and 128 < hd_pad <= 256:
                        kw["block_m"] = 128 if q.shape[1] >= 128 else 64
                        kw["block_n"] = 32
                        kw["num_warps"] = 8
                        kw["num_stages"] = 1
                    elif _pf_hd128 and hd_pad <= 128:
                        kw["block_m"] = 128
                        kw["num_warps"] = 8
                        kw["num_stages"] = 2
                return _orig_prefill(*a, **kw)

            _tpm.paged_attn_triton_prefill = _prefill_rdna
            _swm.paged_attn_triton_prefill = _prefill_rdna
            if _pf_hd256:
                applied.append("paged prefill attention, head_dim 256: 128/64 x 32 tile, 1 stage (EXL3_ROCM_PREFILL_HD256)")
            if _pf_hd128:
                applied.append("paged prefill attention, head_dim <= 128: 128-row tile, 8 warps, 2 stages (EXL3_ROCM_PREFILL_HD128)")
        except Exception as e:
            applied.append(f"!! FAILED paged prefill tile patch: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # GatedResidual (Qwen3.8) prefill mix: gate-mean in one kernel
    # ------------------------------------------------------------------
    # GatedResidual._mix at R > FUSED_MAX_R (prefill chunks) ends in
    #   (sigmoid(g.float()).view(R, H, D) * normed.float().view(R, H, D)).mean(-2).half()
    # -- two fp32 upcasts, a sigmoid, a multiply and a mean over (R, H * D) fp32
    # temporaries, ~1.6 ms per site at R = 2048 on gfx1151 (Qwen3.8 pp2048 trace,
    # b22b243: 95 sites per forward, ~11% of the forward). torch.ops.exl3_rocm.
    # gr_gate_mean (rocm/hc_mix_rdna.hip) does it in one pass over the two half
    # tensors with the same per-element arithmetic, bit-identical to the torch
    # expression (rocm_tools/gr_mix_bench.py --prefill). Everything before it
    # (norm, the two GEMMs, silu, post) is the upstream code verbatim.
    # EXL3_ROCM_GR_PREFILL=0 restores the torch expression.
    #
    # v1.5.3: upstream's _mix gained a third, tiled int8 path (hc_mix_tiled.cu) between the
    # fused decode pair and this cuBLAS one; it is gated off under torch.version.hip, so on ROCm
    # the cuBLAS branch still serves every R > FUSED_MAX_R. proj_h is now zero-padded to a
    # multiple of 64 rows (for the tiled kernel), so the projection reads proj_h[:proj_m] as
    # upstream's branch does. A module that did select the tiled path is left to upstream.
    #
    # FUSED_MAX_R: upstream lowered the fused decode pair's bound from 32 to 8 because the tiled
    # int8 path beats it above 8 rows on NVIDIA. Without that path (ROCm), rows 9..32 would drop
    # to this cuBLAS branch instead; EXL3_ROCM_GR_FUSED_R (default 32, the v1.5.0 bound) keeps
    # them on the fused pair (sized for them: GR_MAX_R 32). Only batched decode reaches 9..32
    # rows (MTP verify is 1 + ndt).
    try:
        from ..modules import hyperconnections as _hcm0
        _gr_r = int(os.environ.get("EXL3_ROCM_GR_FUSED_R", 32))
        if _gr_r != _hcm0.GatedResidual.FUSED_MAX_R:
            _hcm0.GatedResidual.FUSED_MAX_R = _gr_r
            applied.append(f"GatedResidual fused decode pair up to {_gr_r} rows (v1.5.0 bound; EXL3_ROCM_GR_FUSED_R)")
    except Exception as e:
        applied.append(f"!! FAILED GatedResidual row-bound patch: {type(e).__name__}: {e}")

    if _env_on("EXL3_ROCM_GR_PREFILL", True):
        try:
            import torch
            import torch.nn.functional as _F
            from ..modules import hyperconnections as _hcm
            from ..ext import exllamav3_ext as _ext
            _ = torch.ops.exl3_rocm.gr_gate_mean   # raises if the build lacks the op
            _GR = _hcm.GatedResidual
            _orig_gr_mix = _GR._mix

            def _gr_mix_rdna(self, streams, cached = True):
                H, Dh = self.hc_mult, self.hidden_size
                R = streams.shape[0] * streams.shape[1]
                if R <= self.FUSED_MAX_R or H != 4 or Dh % 4 != 0 or not streams.is_cuda \
                        or getattr(self, "tiled", False) or self.proj_h is None:
                    return _orig_gr_mix(self, streams, cached)
                s3 = streams.reshape(R, H, Dh)
                if s3.dtype != torch.float:
                    s3 = s3.float()
                if not s3.is_contiguous():
                    s3 = s3.contiguous()
                dev = s3.device
                post = torch.empty((R, H), dtype = torch.float, device = dev) \
                    if self.use_combine else None
                normed = torch.empty((R * H, Dh), dtype = torch.half, device = dev)
                _ext.rms_norm(s3.view(R * H, Dh), self.w_h, normed,
                              self.rms_eps, 0.0, 1.0, False, False, H)
                dm = torch.matmul(normed.view(R, H * Dh), self.proj_h[: self.proj_m].t())
                t = _F.silu(dm[:, : self.rank] / H)
                if self.use_combine:
                    post.copy_(2.0 * torch.sigmoid(dm[:, self.rank :].float() / H))
                g = torch.matmul(t, self.up_h.t())
                mixed = torch.empty((R, Dh), dtype = torch.half, device = dev)
                torch.ops.exl3_rocm.gr_gate_mean(g.view(R, H, Dh), normed.view(R, H, Dh), mixed)
                return post, mixed

            _GR._mix = _gr_mix_rdna
            applied.append("GatedResidual prefill mix: gate-mean fused (EXL3_ROCM_GR_PREFILL)")
        except Exception as e:
            applied.append(f"!! FAILED GatedResidual prefill patch: {type(e).__name__}: {e}")

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
