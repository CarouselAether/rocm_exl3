# RDNA notes

Reference for maintaining and rebasing the ROCm/RDNA port: what each ROCm-specific
piece is, why it exists, the measurements that justify it, the switches that control
it, the known traps, and how to verify it. A kernel comment that says
`see RDNA_NOTES "X"` points at the heading or phrase X in this file.

The complete dated engineering log (every measurement, sweep, per-run table and dead
end behind these notes) lives outside the repo, in the maintainer's project notes.

As of 2026-09-30 (upstream v1.5.3 + the perf stack). Everything here was measured on
gfx1151 (Strix Halo, RDNA 3.5, wave32) on system ROCm 7.2.4 unless a section says
otherwise (ROCm 10.0 / HIP 7.15 wheels on the `rocm-10` line), not inferred from
documentation. Each item has a plausible-looking wrong answer, which is why it is
written down.

## Switches

C++ switches are read per call (an in-process A/B works); `rocm_py` switches are read at
import / load. Graph-captured kernels bake the switches in at capture time.

| switch | default | effect |
|---|---|---|
| `EXL3_GEMV` | on | `=0` disables the RDNA GEMV fast path in `exl3_gemm` |
| `EXL3_MGEMV` | on | `=0` disables the multi-matrix GEMV in `exl3_mgemm` (cooperative kernel instead); the weighted multi-row path honours it too |
| `EXL3_GEMV_GRAPH` | on | `=0` disables the graph-captured GEMV path |
| `EXL3_GEMV_SPLITK` | on | `=0` disables in-block split-K |
| `EXL3_GEMV_SPLITK_WARPS` | shape rule | `4`/`8`/`16` forces one split-K wave count everywhere |
| `EXL3_GEMV_LDS` | 0 | `=1` pins the old LDS dot core |
| `EXL3_ROCM_GEMV_TILES` | on | `=0` restores the direct dot core instead of the tiles core |
| `EXL3_GEMV_TILES_T` | per-arch table | `1`/`2` forces N-tiles per wave in the tiles core |
| `EXL3_GEMV_FUSE_OUT` | on | `=0` restores separate had_out / reduce kernels |
| `EXL3_GEMV_MAX_M` | 8 | largest m (1..8) on the multi-row GEMV; `1` switches it off |
| `EXL3_GEMV_MR_M1` | on | m == 1 through the multi-row structure; `=0` restores the fused LDS-prologue kernels |
| `EXL3_ROCM_MR_WEIGHTED` | on | weighted MoE down projection on the multi-row path |
| `EXL3_ROCM_HALF_GEMV` | on | half-integer bitrates on the multi-row GEMV and `moe_decode` |
| `EXL3_ROCM_HALF_MOE_PIPE` | on | uniform half-rate MoE layers on the pipelined mainloop |
| `EXL3_ROCM_MOE_PIPE` | on | pipelined MoE mainloop; `=0` restores the shared `exl3_gemm_kernel_inner` |
| `EXL3_ROCM_MOE_BPS` | auto | `=1` forces one MoE block per WGP (same-grid bitwise comparisons) |
| `EXL3_ROCM_MOE_GROUP` | 8 | MoE blocks per expert group (upstream's value) |
| `EXL3_ROCM_MOE_FUSED_ROWS` | 512 | fused-row cap while the pipe is on (upstream `EXL3_MOE_FUSED_ROWS` wins) |
| `EXL3_RDNA_MOE_TILESIZE_K` | 32 | `=16` forces the single-K MoE path (first bisect step if fused MoE regresses) |
| `EXL3_ROCM_MOE_MGEMM_ROUTE` | on | per-token mgemm MoE decode route; `=0` restores the fused `exl3_moe` steer |
| `EXL3_ROCM_MOE_BSZN` | 0 | `=1` leaves upstream bszN dispatch alone (raises in the `exl3_moe_coop` stub) |
| `EXL3_ROCM_MOE_BATCH` | on | MoE decode runs all bsz <= 8 tokens in one set of launches |
| `EXL3_ROCM_MOE_FUSED` | on | `torch.ops.exl3_rocm.moe_decode`, 4 launches per MoE layer |
| `EXL3_ROCM_RDNA4_FUSED_MOE` | 0 | `=1` bypasses the gfx12 per-expert MoE steer |
| `EXL3_ROCM_BATCH_RECON` | 0 | `=1` enables v1.5.0's batched expert reconstruct tier |
| `EXL3_ROCM_QKV_SLICE` | 0 | `=1` enables v1.5.0's sliced one-launch Q/K/V mgemm |
| `EXL3_ROCM_ROUTER_GEMV` | on | router GEMV for m = 1..8 (`=0` also drops the fuse) |
| `EXL3_ROCM_ROUTER_FUSE` | on | router + top-k in one launch |
| `EXL3_ROCM_ROUTER_U` | 8 | router blocks in flight per wave (2/4/8/16) |
| `EXL3_ROCM_ROUTER_STD_MR` | on | std softmax routing at 2..8 rows on the multi-row router GEMV, not hgemm |
| `EXL3_ROCM_ROUTER_DET` | on | upstream v1.5.3 deterministic router math; `=0` = v1.5.0 fast math |
| `EXL3_ROCM_ROUTER_I8` | 0 | `=1` lets `_gate_t` build the (unused on ROCm) int8 router tables |
| `EXL3_ROCM_HC_DPP` / `_HC_FUSE` / `_HC_NORM` | on | mHC: DPP cross-lane ops / apply folded into the next mix / RMSNorm folded into the finalize |
| `EXL3_ROCM_GR_DOTS` | on | GatedResidual rows-dots kernel; `=0` selects upstream v1.5.3's dispatch |
| `EXL3_ROCM_GR_RB` | 4 | GatedResidual rows per block (2/4/8) |
| `EXL3_ROCM_GR_PREFILL` | on | fused GatedResidual prefill gate-mean |
| `EXL3_ROCM_GR_FUSED_R` | 32 | GatedResidual fused-pair row limit (upstream lowered its own to 8) |
| `EXL3_ROCM_DSA_TUNE` | on | DSA split kernel at BLOCK_H 8 / 8 warps; `=0` upstream tuning |
| `EXL3_ROCM_DSA_DECODE` | on | DSA decode MQA kernel; `=0` restores the retuned upstream split kernel |
| `EXL3_ROCM_DSA_DECODE_{SPLITS,BLOCK_H,HP,BLOCK_N,BLOCK_W,KC,KSTAGES,WARPS}` | production | override the decode tiling |
| `EXL3_ROCM_DSA_PREFILL` | on | DSA prefill MQA kernel; `=0` restores the upstream one-shot kernel |
| `EXL3_ROCM_DSA_PREFILL_{HP,BD,KC,BLOCK_N,BLOCK_W,WARPS,KSTAGES}` | production | override the prefill tiling |
| `EXL3_DSA_SWEEP` | -- | `tile`/`splits` sweep mode of `bench_dsa_decode.py` |
| `EXL3_ROCM_BC_BUFOPS` | on | bc_attn GQA decode split kernels compiled with AMD buffer-op attributes |
| `EXL3_ROCM_BC_ATTN_WARPS` / `_STAGES` | 8 / 1 | warps / stages for those kernels |
| `EXL3_ROCM_PREFILL_HD128` | on | narrow-kv paged prefill tile (moved out of `triton_paged.py`) |
| `EXL3_ROCM_PREFILL_HD256` | on | paged prefill tile for head_dim 256 |
| `EXL3_ROCM_MLA_LDS_FIT` | on | MLA unfold / MHA prefill tiles fitted to 64 KB LDS |
| `EXL3_ROCM_SMEM_LIMIT` | on | seed `smem.smem_limit()` from `shared_memory_per_block` (64 KB) |
| `EXL3_ROCM_WMMA_GEMM` | 1 | `0` off, `1` table route flags decide, `2` every eligible call |
| `EXL3_ROCM_WMMA_GEMM_F16` | on | `=0` keeps fp16-output hgemm on hipBLAS |
| `EXL3_ROCM_WMMA_GEMM_MIN_M` | 9 | smallest m on the WMMA GEMM backend |
| `EXL3_ROCM_WMMA_GEMM_CFG` / `_TRACE` | -- / 0 | pin one config / log one line per m > 8 call |
| `EXL3_RDNA_HGEMM_NARROW_N` | 256 | narrow-N split-K hgemm limit (m <= 8); `0` disables |
| `EXL3_RDNA_SMS_MULT` | 1 | multiplies the cooperative grid (experiment; `2` is refused for heavy shapes) |
| `EXL3_ROCM_HIP_GRAPHS` | runtime gate | `1`/`0` forces HIP graph capture; default on for HIP >= 7.14, off below |
| `EXL3_NGRAM_LOCK_HEADROOM_GB` | 8 | memory headroom the `-ngl` n-gram lock keeps |
| `EXL3_GDN_PROJ_FP32`, `EXL3_GDN_CONV_TOKEN_MAJOR` | upstream | upstream v1.5.3 GDN prefill numerics (`=1` / `=0` restore v1.5.0) |

Build-time: `-DEXL3_MOE_PIPE_PROF` (MoE phase timers), `-DEXL3_HALF_U_T2/_T1` (half-rate
U sweep). Legacy: `EXL3_RDNA_GEMV_GRAPH` (see "Decode lost the GEMV path").

## WMMA

All four WMMA variants exist on gfx1151 and all take operand order **`(B, A, C)`**:
`f32_16x16x16_f16`, `f16_16x16x16_f16`, `i32_16x16x16_iu8`, `f32_16x16x16_bf16`.
Wrong operand order produces a transposed result, not a crash. One fragment layout:

| fragment | mapping |
|---|---|
| A | lane holds row `L % 16`, all 16 columns |
| B | lane holds column `L % 16`, all 16 rows |
| C | `row = L % 16`, `col_base = (L >= 16) ? 1 : 0`, element `i` at column `i*2 + col_base` |

- bf16 accumulates to **fp32 with the same C layout**, reusing `WmmaFragC` and helpers.
- fp16-accumulate packs into every other half-slot selected by `opsel` (a
  template/immediate) and preserves the other half: two accumulators per fragment.
  Single-accumulator fp16 saves no registers (8 VGPRs, same as fp32), so fp32
  accumulate is strictly better unless the `opsel` packing is used.
- **int8 sign-flag trap:** the builtin is `(s0, v0, s1, v1, C, clamp)` and each flag
  pairs with the vector that *follows* it, so under `(B, A, C)` the first flag
  describes **B**. `mma_sync_i8<signed_a, signed_b>` in `rdna_wmma.hip.h` hides this.
- A/B are **replicated across the two half-waves**: a resident 16 x 512 A panel is 256
  VGPRs per warp by itself (root cause of the DSA kernels' spills).
- The f32 accumulator rounds each 16-deep step toward zero (see "WMMA GEMM backend").

HIP's documentation covers MFMA (CDNA), not WMMA; the RDNA ISA PDF is the only
authority. `rocm_tools/wmma_check.hip` and `wmma_gate.hip` check the shipped header.

**RDNA4 (gfx1200/1201).** The gfx11 WMMA intrinsics have no gfx12 encoding.
`rdna_wmma::mma_sync` compiles to `__builtin_trap()` under `__gfx1200__/__gfx1201__`
(loud, never silently wrong), and rocm_py clears `fused_mode_buffers` on gfx120x so MoE
runs the per-expert path and the trap is unreachable (`EXL3_ROCM_RDNA4_FUSED_MOE=1`
bypasses). Compile-verified only (`GPU_ARCH=gfx1201 hipcc_probe.sh --all`); **no RDNA4
hardware has run this port.** A real port needs half-size fragments and new lane maps.

## Hardware

- **int8 dot product is `__builtin_amdgcn_sudot4`, never `sdot4`.** gfx1151 lacks
  `dot1-insts`, so `sdot4` fails to compile and reads like "no int8 support". It has
  `dot8-insts`; `udot4` and `fdot2` are available, `udot2` is not.
- **`hipDeviceProp_t` reports `major = 11, minor = 5`**, so upstream's
  `prop.major >= 10` Blackwell test (`exl3_devctx.cu:39`) classifies RDNA as
  Blackwell. `exl3_kernel_map_rdna.hip` ignores `cc` for that reason; the same trap
  recurs in v1.5.3's `routing_gemm_det_fits()`.
- **LDS is 64 KB per workgroup**, against the ~90–100 KB CUDA shape tables assume. The
  dynamic LDS base is 32-byte aligned; misaligned vector loads split rather than fault.
  torch on ROCm has no `shared_memory_per_block_optin`.
- **`multi_processor_count` reports WGPs, not CUs** — 20 on a 40-CU part (20 WGPs -> 40
  CUs -> 80 SIMD32, 16 wave32 per SIMD). Upstream's `2 * sms` Triton decode-split
  target is therefore half what it assumes on NVIDIA; doubling it was tried and
  reverted (split counts 5..12 within ~2% at ctx 1-4K, `bench_decode_splits.py`).
- **`__funnelshift_r` is native**, matches PTX `shf.r.wrap.b32` (shift `& 31`), one
  `v_alignbit_b32`. A hand-rolled uint64 version with `& 63` is wrong for shifts >= 32.
- **VALU rates** (instr/SIMD/clk, wave32): add/bfe/perm/alignbit/lshl_or, `v_sad_u8`,
  `v_sad_hi_u8`, `v_mad_u32_u16`, `v_pk_mul_lo_u16`, `v_mul_u32_u24`, `v_pk_fma_f16`,
  `v_dot2_f32_f16` ~0.92-0.96; **`v_dot4_u32_u8`/`v_dot8_u32_u4` 0.49 (half)**;
  **`v_mul_lo_u32`/`v_mul_hi_u32` 0.24 (quarter)**. The GEMV tiles core is built on this.
- **Prefill is slightly compute-bound**, ~9.8 TFLOP/s at the model level (Gemma-4-31B)
  against a peak fp16 GEMM of 24-32 TFLOP/s.
- **Achievable memory bandwidth is ~206 GB/s** (`bench_membw.py`), ~80% of 256
  theoretical, flat from 64 MiB to 16 GiB. No VRAM-vs-GTT cliff: the 512 MiB "VRAM"
  aperture is a legacy carveout and torch's `total_memory` is the GTT pool. Pure
  streamed reads (DRAM-resident GEMV sweep, `GEMV_SWEEP=1 gemv_check`) sustain **226
  GB/s** — the roofline for weight-streaming kernels.
- **There is a 32 MiB Infinity Cache, and it will flatter any kernel benchmark whose
  working set fits.** 653 GB/s at 16 MiB against 213 GB/s at 64 MiB. Timing one weight
  tensor in a repeat loop measures cache, not DRAM -- it overstated the EXL3 GEMV rate
  by 26%, and a first pass concluded "the kernels are healthy" from exactly that error.
  Cycle a working set several times cache size. Kernel figures here are DRAM-resident.
- **Command-processor dispatch gap ~2.1-2.7 us between dependent kernels**, not removed
  by graph replay (see "dispatch-gap census"). Launch count is GPU time.
- **Noise floor ~4.7% spread** (1.6% stdev); the first run reads high. No claim under
  ~5% survives one measurement; below that use alternating A/B rounds.

Reference failure shape for the next regression: as first shipped, Gemma-4-31B decode
was **not memory-bound** — 97.6% of GPU time in `exl3_gemm`/`exl3_mgemm` running a
16-row tile for one useful row. The tile GEMM measures a flat ~53 GB/s at bsz 1, 4
*and* 16: the signature of padding rows (see "Decode lost the GEMV path").

## Toolchain

- **`-fgpu-rdc` hides codegen failures**: it defers device codegen to link time, so a
  TU with inline PTX that merely *parses* on amdgcn (only `"r"` constraints) fails only
  at link. `hipcc_probe.sh` compiles without rdc, retrying with it only when
  "undefined symbol" is the sole error. Its exclusion regex must match `ROCM_EXCLUDE`
  in `setup.py`.
- **Inline PTX lives in three files**: `ptx.cuh` (19 blocks), `quant/codebook.cuh`
  (10), `quant/exl3_gemv_int8_kernel.cuh` (1, written `asm ("dp4a...` with a space).
  Search with `grep -rn "asm[[:space:]]*("`.
- **HIP lacks `__dp4a`, `__ldcs`, `__ldcg`**, bridged in `hip_compat.hip.h`. `__ldcg`
  bypasses L1 for cross-block visibility, so it is an agent-scope atomic load.
- Under the shim `__CUDA_ARCH__` is 1: arch-gated upstream code takes `arch < 1000`.
- **LLVM address-math trap:** in a loop, LICM hoists a 32-bit lane offset's
  zero-extension to the preheader; ISel then emits 2 VALU of 64-bit address math per
  load instead of the saddr form. An empty `asm volatile("" : "+v"(off))` per
  iteration keeps the zext in the loop (zero cost).
- `__builtin_bit_cast` on an ext-vector *element* lvalue reads element 0; copy it out.
- setup.py uses one flag set per TU, so kernels wanting CU mode carry
  `__attribute__((target("cumode")))` (device pass only): 3-9% faster at large WMMA
  GEMM shapes.
- torch >= 2.14 headers need `-std=c++20`; setup.py picks it from a real version parse
  (`"2.9" >= "2.14"` is lexicographically true).

## Why the siblings differ

### `hip_compat.hip.h` — warp-sync primitives

`__syncwarp` maps to a wavefront-scope release fence, `wave_barrier()`, and an acquire
fence. `__builtin_amdgcn_wave_barrier()` alone is a *scheduling* barrier and emits no
`s_waitcnt`, silently dropping the shared-memory ordering half of `__syncwarp`. That
breaks any LDS cross-lane exchange where write and read use different addresses —
`routing.cu`'s radix sorts and `hadamard_inner.cuh`'s `had_hf_r_128_d_inner`.
`__ballot_sync` and `__activemask` cast to `unsigned`: HIP's `__ballot` is 64-bit, and
the uncast form selects the wrong `__ffs`/`__popc` overload.

### `exl3_gemm_inner_rdna.hip.h` — LDS layout and split-K

- **B dequant staging stride is 18 halves, not 17.** 17 puts adjacent active-lane
  groups on the same LDS banks (~24% stall time by PMC); 18 halves = 9 dwords, coprime
  with 32. It lives in `EXL3_GEMM_SH_B_DQ_STRIDE` for all consumers — it was once
  duplicated, and the host launch accounting drifted to 17 while the kernel indexed at
  18, under-allocating dynamic LDS by 256 bytes on the *shipped* path while the
  standalone harness passed.
- **`TILESIZE_N = 192` is invalid** (`static_assert`): it fails `N % 128 == 0`, and
  `FRAGS_N_PER_WARP = TILEBLOCKS_N / NUM_WARPS` = 12/8 = 1 silently drops a third of the
  tile.
- **`sh_b_dq` is sized `NUM_WARPS * TILEBLOCKS_K` and indexed by the block-wide warp
  id.** At `TILEBLOCKS_K == 2` the block holds 16 warps while `warp_id = t / 32` only
  spanned 8, so `sub_k` 0 and 1 warps staged different B fragments into one buffer.
- **`threadblock_reduce()` indexes `sh_c` by `t` on both sides.** An earlier version
  read at `t + src * EXL3_GEMM_BASE_THREADS`, summing unwritten LDS past the block end.

The last two broke GLM-4.6V's fused MoE and were invisible for a long time because
every RDNA shape uses `TILESIZE_K = 16`, which compiles split-K away; the MoE kernel is
its only live caller. **Dead-code-that-wasn't** recurs in this port: a path that
"cannot run" usually has one caller nobody listed. v1.5.3 adds the half-integer width
(`TILE_U16 = 16 * bits + 8 * half_k`, `dq_dispatch<bits, cb, half_k>`).

### `exl3_moe_shape_rdna.hip.h` — MoE tile-K

Upstream's `MOE_TILESIZE_K` is a bare `#define`, so `-D` cannot override it. The value
is upstream's 32; `EXL3_RDNA_MOE_TILESIZE_K=16` forces the single-K path at 1.42–1.56x
MoE cost. Include it after `exl3_moe_common.cuh` in both kernel and host siblings: the
host derives `blockDim` from it, so a mismatch is a silently wrong launch.

### `exl3_kernel_map_rdna.hip` — shapes and LDS budget

RDNA shapes use `TILESIZE_K = 16` and 256-thread blocks to fit 64 KB LDS.
`EXL3_RDNA_SMEM` requests each shape's actual LDS need; `SMEM_MAX` on a 64 KB part pins
residency at one block per WGP and would make `exl3_mgemm`'s multi-z grids illegal. The
RDNA runtime accounting is the single source of LDS sizes (upstream's constexpr smem
functions over the CUDA table are not carried). Cooperative launch works on gfx1151:
across 13 cases results agree exactly with non-cooperative work, and an oversubscribed
grid is refused, not hung. Forced shapes that do not divide the problem are rejected
(the inner kernel floors `size_n / TILESIZE_N` with no remainder pass; a forced
non-dividing shape once produced a benchmark "win" computing 75% of the output).

### `rope_rdna.hip` — fused RMS norm (history)

Through v1.4.4 upstream's `apply_norm` reduced a warp total with `__shfl_down` (result
in lane 0 only), then had all 32 lanes store it to one `sums[]` slot. With distinct
lane values 0..31 (true sum 496):

```
__shfl_down : lane0=496  lane1=512  lane16=752  lane31=992
__shfl_xor  : every lane 496
unguarded store lands 992      <- lane 31 wins on RDNA, lane 0 on NVIDIA
```

On RDNA QK-norm was silently scaled wrong per head (`test_rope` 30 failed -> passing
with the guard); the sibling guarded the store to lane 0 (`EXL3_RDNA_NORM_LANE0`).
**Upstream v1.5.0 adopted the guard**; the sibling is back to one `= {}` line. Lesson:
a first probe called the race benign because it used uniform lane values (clamp and
wrap indistinguishable). **Reduction probes need distinct per-lane values.**

### Other siblings with ROCm logic

- `exl3_gemm_rdna.hip`: GEMV / mgemv / multi-row routing, width lists, half-K,
  autotune. The mgemv fast path declines `num_tokens > 1 && min_index >= 0` (since
  v1.4.4 upstream masks in place; mgemv's grouped reduce still compacts and divides
  `packed / num_tokens`). Only TP-sharded / CPU-split expert maps produce it.
- `cuda_drv_rdna.cpp` (runtime resolution, "ROCm wheel stack" trap 3), `graph_rdna.hip`
  ("HIP graphs"), `hgemm_rdna.hip` (narrow-N, WMMA backend), `routing_rdna.hip` and
  `hc_mix_rdna.hip` ("Decode leftovers", "Qwen3.8 decode").
- Stubs: `exl3_moe_coop` (raises), int8 GEMV (returns false), `hgemm_f16acc`
  (`hgemm_recon` = `hgemm`; RDNA has no fp32-accumulate rate penalty),
  `quantize_tiles_use_optimized()` hard false.

## Why mgemm is capped at ~1/3 roofline at m == 1

The cooperative `exl3_mgemm` is **occupancy-starved by the cooperative launch** — not
registers, dequant or bandwidth. rocprofv3 on `bench_mgemm.py` (Laguna experts): ~12%
occupancy at VGPR 144, 248 *and* 256. The grid is 20 workgroups (one per WGP) x 8
waves over 80 SIMDs = 2 waves per SIMD = 12.5% of 16, matching the measurement with
nothing left over. Two waves per SIMD cannot keep loads in flight: `MemUnitBusy`
26-32%, 55-66 GB/s. The grid cannot widen: `grid.sync()` needs every block co-resident
and heavy shapes fit one workgroup per WGP (`EXL3_RDNA_SMS_MULT=2` is refused). The
default sizing beats every forced `force_num_sms`, and the selector's shape pick is
already the fastest. **The cooperative GEMM cannot exceed ~1/3 of roofline at m == 1 on
this part regardless of how much parallel work exists.** Hence the plain-launch
multi-matrix GEMV (`quant/exl3_mgemv_rdna.hip`, design in its header).

## Decode lost the GEMV path

**Resolved.** Kept because it documents *how* the path was lost: both halves were
comments true when written and invalidated by changes elsewhere, a failure shape this
port has hit three times. The legacy 0.0.29 fork ran essentially all decode on the RDNA
GEMV; on v1.4.1 ~75-78% of decode ran 16-row tile kernels for one useful row.

- **Half 1 — the `!graph` guard stopped being cheap.** `EXL3_RDNA_GEMV_GRAPH` declined
  GEMV during capture. In 0.0.29 almost nothing was captured; upstream then added
  `bc_attn.py` and the MoE bszN graph path, decode became ~100% captured, and the
  guard ("costs coverage during capture and nothing else") declined everything.
- **Half 2 — retiring the mgemm guard closed the other door.** The fork had disabled
  mgemm, so fused q/k/v and gate/up ran as separate `exl3_gemm` calls and took GEMV.
  Once retired, `exl3_mgemm` had **no GEMV path at all** (42-48% of Laguna decode).

**Packing rows is not the alternative:** routed experts each need a different B, so
they cannot share the 16 tile rows. Packing pays only where rows share weights
(concurrent sequences, speculative decode).

The fix: the fp32-output GEMV made faster than the tile GEMM; the graph-parameter
contract extended to multi-kernel GEMV (`Graph::record()` walks nodes and
`graph_sites` in lockstep and `break`s on the first function mismatch, so sites are
pushed in launch order); and an m == 1 GEMV call site in `exl3_mgemm`. Width-list
calls (`size_n_list`/`c_ptrs`, DS4's `bc_dsa` fan sites) take mgemv too; both lists are
device arrays read per launch, so capture needed no new patch sites. Trap: a
`hipMalloc` during stream capture invalidates the graph — prewarm parameter blocks.

## The barrier-free dot-tile core

`exl3_gemv_dot_tile_direct` (`exl3_gemv_kernel_rdna.hip.h`) keeps accumulation in dq's
native fragment layout — lane L holds rows `(L%4)*2+{0,1,8,9}` of columns
`(L/8)*2+((L>>2)&1)` and `+8` — so each k-tile is 4 `v_dot2_f32_f16` per lane with B
read from global: no LDS, no barriers, no idle lanes. A 2-hop `__shfl_xor` quad reduce
and a broadcast remap at the end of the k-range restore "lane l returns column l", so
every wrapper takes any core. The old LDS core (lanes 16-31 idle in the dot) ran
129-148 GB/s on Gemma mid shapes; the direct core gained 1.12-1.49x. `EXL3_GEMV_LDS=1`
pins the old core (smem is passed identically, graph patch sites untouched). The tiles
core has since replaced the direct core with identical arithmetic.

**Split-K**: capped at `EXL3_GEMV_SPLITK_MAX_TILES = 2048`. The wave count is
`exl3_gemv_splitk_warps(k_tiles, n_tiles, bszm)` (sweep data at the function): short k
(<= 128 k-tiles) and saturated grids (>= 1024 blocks = n_tiles x bszm) take 4, starved
grids (< 128 blocks) and long k (>= 1024 k-tiles) 16, else 8. This rule is also the
per-row reduction order that keeps multi-row and batched MoE bit-identical to m == 1.

Closed without implementation: **VOPD** (the compiler already emits `v_dual_*` where
legal, incl. `v_dual_dot2acc_f32_f16` in the hot loop; the rest is VOP3/VOP3P, not
pairable) and **register-rotation software pipelining** (built, slower everywhere, -6
to -8% on short-k split-K: plain launches at high occupancy already hide latency with
other waves; intra-wave latency hiding is for kernels that cannot oversubscribe).
Batching U k-tiles' loads with no rotation *did* win once the decode was cheap ("GEMV
tiles core").

**Validation discipline:** `gemv_check.hip` runs every case on every core against an
independent reconstruct reference; fp32 truth for real weights is
`A @ LinearEXL3.get_weight_tensor()` (folds suh/svh and the Hadamards). Max-relative
error with a small clamp false-flags near-zero outputs; use gemv_check's gates
(`d > 0.01*denom + 0.05`, RMS ratio primary). Profile before implementing.

## MoE decode route restored

v1.5.0 replaced upstream's per-token three-`exl3_mgemm` MoE decode with
`exl3_moe_coop` (not ported); the sync steered bsz <= 8 to the fused `exl3_moe`, a
16-row WMMA tile GEMM with 15/16 padding at bsz 1 running at the tile path's flat ~53
GB/s plus per-layer argsort/bincount host syncs. Laguna decode fell 20.8 -> 10.4 t/s.
Fix (`rocm_py`, no upstream or kernel edit): during `BlockSparseMLP.forward`, `self.bc`
is a proxy whose `run_bszN` is the v1.4.4 loop writing `experts_cfg.out_bszn[i]`;
everything else forwards to the real `BC_BlockSparseMLP`. `bc_sh_exp` is forced False
after `load_local` so the Python shared-expert tail runs; expert-range shards pass
`cfg.min_expert / max_expert`. Each call lands on mgemv. fp32-reference error at bsz
1/3/8 equals the fused kernel's; Laguna **10.4 -> 21.1 t/s**. `exl3_moe_coop` would have
to beat mgemv to matter. The route is now batched and fused ("Decode leftovers").

## The narrow-output hgemm (Narrow-N hgemm)

`BC_Attention`'s headwise gate, `hgemm_gr(x2, g_weight, s.g2)` in
`libtorch/attention.cpp` — a (1 x 3072) @ (3072 x 48) fp16 product — cost 15% of Laguna
decode. Issued from C++, so Python patches never fire, and an LD_PRELOAD interposer on
`hipblasGemmEx` sees nothing (versioned symbol).

Library behaviour at narrow N (`hgemm_narrow_probe.py`, rotating working sets, not
Infinity-Cache numbers):
- **rocBLAS** (hipBLAS default): ONE 128x128-tile workgroup walks K in 96 serial LDS
  round trips, 63-75 us flat for m = 1..32, ~50x the bandwidth bound. Wide N is fine.
- **hipBLASLt** (`ROCBLAS_USE_HIPBLASLT=1`): 10-22 us, but loses at m >= 128 and wide N,
  costs prefill 13-17% process-wide, no per-call switch, and linking it directly would
  duplicate torch's copy. **Rejected.**
- An ATen fp32 GEMV recipe wins only to N ~64-192 and cannot run under capture.
- torch.matmul uses rocBLAS on the 7.2.4 wheel (74 us) and hipBLASLt on 10.0 (10 us):
  why ROCm 10 looked "fixed" in torch but not in the ext, and why DS4's torch fp16
  matmuls are faster there.

Fix: `rocm/hgemm_rdna.hip` runs m <= 8, N <= 256 (`EXL3_RDNA_HGEMM_NARROW_N`) as split-K
partials (one block per 64-row K-slice) into the DevCtx workspace plus a reduce into C
in its dtype and stride. No allocations: capture-safe. 8-24 us at m = 1; Laguna decode
**21.1 -> 23.3 t/s**. Not covered: skinny fp16 GEMMs with N > 256 (e.g. an unquantized
drafter) still hit the rocBLAS pathology.

## Launch-count fusion: landed

mgemv used three kernels per call (`had_in` -> split-K dot -> `had_out`), each paying
the CP dispatch gap. Now:
- **Multi-matrix (`exl3_mgemv_rdna.hip`)**: each dot block rotates its expert's input
  into LDS; the last-arriving warp of each 128-wide output segment (atomic counter per
  (slot, segment), **self-resetting**, in the parameter block) rotates the segment in
  place, and with routing weights the last rotated slot runs the grouped reduce. One
  launch per call; the dot kernel hosts the graph patch sites.
- **Single-matrix graph path (`exl3_gemv_rdna.hip`)**: same folds, six patch sites. The
  plain path (lm_head, calls without su/sv) is untouched.
- `EXL3_GEMV_FUSE_OUT=0` restores the output kernels. Helpers under "Launch-count
  fusion" in `exl3_gemv_kernel_rdna.hip.h`.

Every fused kernel is **bit-identical** to what it replaced (`mgemv_bitwise.py`,
`decode_bitwise.py`); graph replay needs no per-replay memset. Laguna launches/token
1961 -> 1152 (-41%); decode +1-2.5%.

**The motivating diagnosis was wrong, and that is the lasting result.** Decode was
called host-bound from Kineto (GPU busy 60%). Measured without the profiler: 3.3 us of
host per `exl3_mgemm`, the forward enqueued in ~4 ms then ~31 ms waiting —
**unprofiled decode is GPU-bound**; Kineto inflates per-launch host cost ~10x. **Never
diagnose host-vs-GPU from a Kineto profile alone**; time the host segment directly
(forward entry/exit with and without a sync). The LDS rotation prologue later proved to
cost more occupancy than its launch saved ("Row tile 1"); the fused output epilogue
and counters are the lasting parts.

## Multi-row GEMV, m = 2..8

### Where the m = 3 verify step goes

MTP / draft verification runs m = ndt + 1 rows. Before this, `profile_decode.py -ndt 2`
(GPU ms, plain token / m = 3 step):

| class | Qwen3.8 | DS4 |
|---|---|---|
| dense linears (GEMV at m=1 -> cooperative GEMM at m=3) | 12 / 47 (**4x**) | 16 / 72 (**4.4x**) |
| MoE experts (per-token mgemv loop) | 9.7 / 29 | 28 / 45 |
| attention / GDN | 9.6 / 16 | 11.4 / 20.6 |
| whole step | 40 / 110 | 61 / 156 |

Memory-bound, an m = 3 step would cost ~1.1-1.3x an m = 1 step; it cost 2.6-3.5x and
MTP was a net loss. The dense linears were the lever. The fused LDS rotation prologue
does not extend to m rows (8 x 12288 halves = 196 KB), so at m > 1 the rotation is its
own kernel (launch cost is irrelevant at a GPU-bound 100+ ms step).

### Design

`rocm/quant/exl3_gemv_multirow_rdna.hip` (+ `.hip.h`), routed from `exl3_gemm_gr` and
`exl3_mgemm_gr` for m <= `EXL3_GEMV_MAX_M` before the cooperative kernels, in and out of
capture: a rotation kernel writes the m rows to A_had (hosting the graph patch sites,
republished through a per-device block); a split-K dot kernel with row tile M in {1, 2,
4, 8} dequantizes each B tile once for M fdot2 accumulator pairs; the fused
last-arriving-warp epilogue finishes. **Row r of an m-row call is bit-identical to an
m == 1 call on that row** (same chain, same split-K order — the wave rule is the m == 1
rule). Handles routing weights (`EXL3_ROCM_MR_WEIGHTED`); declines sliced mode. Qwen3.8
MTP ndt=2 20.0 -> 32.1 t/s, DS4 12.1 -> 19.5 (m = 3 dense call 284 -> 71 us).

### Row tile 1

At m = 3 the multi-row kernel (71 us) beat the fused m == 1 kernel (78 us) with three
times the rows: A from L2 and 8 KB of smem vs 13-31 KB means more blocks per CU.
`EXL3_GEMV_MR_M1` (default on) routes m == 1 through it with row tile 1: bit-identical
logits (lm_head-scale widths stay on the single-warp m == 1 form so K splits
identically), DS4 16.2 -> 17.0, Qwen 23.8 -> 24.4.

## GEMV tiles core

`rocm/quant/exl3_gemv_tiles_rdna.hip.h` is the dot core for every EXL3 GEMV
(`EXL3_ROCM_GEMV_TILES=0` restores the direct core). Same lane layout, same fdot2 chains
in the same order, same split-K reduction: **every output is bit-identical** to the
direct core. DS4 tg128 **22.0 -> 27.7 t/s**, MTP ndt=2 24.7 -> 32.0, Qwen3.8 +4%,
Gemma +2.4% (K6, near its roofline).

**Why.** "VALU issue per wave-cycle 0.04" read as low VALU use; per SIMD it is 0.04 x
~14 resident waves ~= 0.55. With the VALU rate table (see "Hardware"), the old K=2 loop
was 56 instructions = **97 VALU cycles per 16x16 tile per lane** (8 quarter-rate hash
multiplies, 8 half-rate dp4a byte sums, 64-bit shifts, ~17 ops of 64-bit address math,
a divergent exec-mask loop): the decode GEMVs were VALU-bound.

**What the core does (exact integer arithmetic):**
- Hash multiply `x * C mod 2^32` for 16-bit x: `v_pk_mul_lo_u16 src, [0|C_hi]` then
  `v_mad_u32_u16 src, C_lo, r` — two full-rate ops for one quarter-rate op; x in either
  register half, so windows at bit 0/16 need no extraction. The cb 0 additive constant
  rides in the same ops (`v_pk_mad_u16`).
- Byte sum: `v_sad_u8(P, 0, 0x64006400)` then `v_sad_hi_u8(P1, 0, s)` yields the pair
  packed as `[0x6400+s0 | 0x6400+s1]` (mul1, cb 2; mcg/3inst keep lop3+hadd).
- fshift at K = 1/2/4 is one `v_alignbit`. K = 5/6/8 keep the generic 64-bit fshift
  (those shapes already run 200-227 GB/s, memory-bound; not worth changing).
- `readfirstlane`'d bases, saddr loads with a loop-invariant 32-bit lane offset (LLVM
  trap under "Toolchain").
- U k-tiles' loads issued together, decoded in k order (U = 2-4 at K <= 3), and T = 2
  adjacent N-tiles per wave sharing the A loads. Per-arch tables `exl3_tiles_u_splitk` /
  `exl3_gemv_tiles_tpb`; T = 2 only while the halved grid keeps >= 1024 waves (scaled by
  multiProcessorCount).

K=2 split-K: 97 -> 39 VALU cycles per tile; DS4 routed gate/up 121 -> 74 us per call,
wo_b 210 GB/s, lm_head 227 GB/s. Max dot-kernel VGPR 105 (gfx1151) / 112 (gfx1201), no
scratch. Verify: `gemv_tiles_bench.hip` (every (U, T) bit-for-bit), `gemv_check`,
`bench_gemv_kernels.py --ab`, `mgemv_bitwise`, `decode_bitwise`, `multirow_check`.

## Decode leftovers

Decode-only changes, each switchable (default on), each **bit-identical** (decode_bitwise
DS4 / Qwen3.8 / Gemma, all on vs all off). DS4 tg128 **27.5 -> 30.1 t/s**, MTP ndt=2
31.4 -> 41.7; Qwen3.8 +4%. Starting point: GEMVs at roofline, host ~7 ms ahead
(GPU-bound); left were the router (37.8 us, 55 GB/s), the weighted down projection, and
launches (7 per routed-MoE layer, 4 per mHC site).

| switch | change | where |
|---|---|---|
| `EXL3_ROCM_ROUTER_GEMV` | router GEMV m = 1..8; m == 1 with 16-byte loads + per-wave LDS transpose keeping upstream's lane -> column chain | `rocm/routing_rdna.hip` |
| `EXL3_ROCM_ROUTER_FUSE` | router + top-k in one launch (last-arriving block runs the top-k body verbatim) | same |
| `EXL3_ROCM_MR_WEIGHTED` | weighted MoE down on the multi-row path | `exl3_gemv_multirow_rdna.hip` |
| `EXL3_ROCM_MOE_BATCH` | all bsz <= 8 tokens in one set of launches; act over valid rows; out_bszn aliases out_d | `rocm_py/__init__.py` |
| `EXL3_ROCM_MOE_FUSED` | `moe_decode`: gate+up as one 2S-slot GEMV, silu*up folded into down's input rotation; 4 launches per layer (was 7 at bsz 1, 21 at bsz 3) | multirow sibling + rocm_py |
| `EXL3_ROCM_HC_DPP` | hc_mix partials / sinkhorn cross-lane ops on DPP | `rocm/hc_mix_rdna.hip` |
| `EXL3_ROCM_HC_FUSE` | mHC apply deferred into the next site's partials kernel | same + rocm_py |
| `EXL3_ROCM_HC_NORM` | RMSNorm after each mix replayed in the finalize (one block per row, row in LDS) | same + rocm_py |

- **Router.** Loads in flight alone stopped at 22.3 us: 512 K dword loads for 2 MB is an
  address/TA rate limit, not latency. 16-byte loads + an LDS transpose reading back
  columns l, l+32, l+64, l+96 keep upstream's chain: 13.2 us at `EXL3_ROCM_ROUTER_U=8`.
- **MTP greedy output is token-identical to plain greedy** (DS4): the verify step's
  router used to run on hgemm, so near-tie expert picks could flip. Every verify row now
  takes m == 1 arithmetic end to end (dense multi-row, batched MoE with the per-token
  wave rule, router rows, mHC). Qwen3.8 lost this at v1.5.3 (see there).
- **`moe_decode`** is registered from the sibling with `TORCH_LIBRARY_FRAGMENT` (no
  upstream binding edits). Its split-K wave count is sized from one token's slots per
  matrix (top_k), the separate calls' rule, so every slot keeps its reduction order. It
  raises "shape exceeds the arrival counters" for bsz * top_k > 64; rocm_py sends those
  calls to the mgemm route.
- **Weighted down**: skipped slots (negative indices) arrive as zero rows so the grouped
  reduce completes (+0 is exactly skipping); honours `EXL3_MGEMV=0`.
- **mHC**: every hc_mix cross-lane op moves a value within a 16-lane row, so DPP
  row_shl / quad_perm / row_xmask / v_permlanex16 carry the same values in the same adds.
  The apply fold works because a partials block owns the same columns of all 4 streams.
  At a 256-thread launch bound the compiler capped VGPRs at 64 and serialized 24 loads;
  issuing them before the barrier plus `__launch_bounds__(256, 1)` fixed it. The deferred
  apply is flushed before anything else can see the streams (other HC calls, HyperHead,
  device change); nothing is deferred while exporting states.
- `profile_region.py` enqueues 8 x steps + 8 tokens so MTP runs do not finish mid-region.

Open: ~0.47 ms/token step-boundary idle (host); DS4 attention small ops in upstream
`dsv4_attn.cpp` (~5 launches per layer) are the next launch-count lever.

## DSA decode MQA kernel

`_dsa_attn_split_kernel` (DeepSeek-V4 decode sparse attention) cost 160-225 us per call,
43 calls/token, 16% of DS4 decode, even after `EXL3_ROCM_DSA_TUNE`. **Root cause:** a
loop-invariant q tile held as the WMMA A operand (256 VGPRs for 16 x 512, replicated
across half-waves) plus a BLOCK_H x 576 fp32 accumulator: 256 VGPRs, ~700 spills, 2.2 KB
scratch per lane in every tuning, and each split sees only ~8-40 keys, so the call is
all fixed cost. WMMA was emitted; "no WMMA" was not the problem.

**Fix.** `rocm_py/dsa_decode_rdna.py:_dsa_decode_mqa_kernel`, a drop-in (same arguments,
workspace and combine; unchanged C++ launch). One program = HP heads x one BD-wide
output column block x one key split. Scores are the full 512-wide q.K reduced in a
**runtime** loop of KC chunks with q re-read per chunk (L1/L2 hits) — a static unroll
or an invariant q brings the spills back; scores are recomputed per column block. The
virtual key row is [c | r] (ring/chunk rows contiguous, pool rows pool_c ++ pool_r);
packed pools (QC) use a column-range plane loader in upstream's H32 domain. The kernel
reads pid % (H/BLOCK_H) as (head group, column block); m / l come from column block 0.

**Production tiling:** bc_dsa.BLOCK_H 16 (the combine's head block), HP 32, BD 256
(accumulator 32 x 256 fp32 = 64 VGPRs over 128 lanes), BLOCK_N = BLOCK_W 32, KC 64, 4
warps, stages 1, 8 splits (for ctx 512 and 16K; BCDsaBatch hardcodes 8). Sweep facts
(`bench_dsa_decode.py --sweep-old/--sweep-new`, graph replay): the best retune of the
old kernel was ~70 us and still spilled; BLOCK_N 32 beats 16; KC 128 spills at 4 warps
and KC 32 is slower; 8 warps win only for QC; 8 splits beat 2/4/16 at both contexts
(16 is slower even for window-only layers); KSTAGES 2 is slower except QC (rejected).

Result: csa ctx512 169 -> 15.5 us, top-k 16K 218 -> 33, 0 spills (fp16 pools); DS4
tg128 **18.07 -> 21.52 t/s**, MTP +15%. **Numerics:** different reduction order;
`decode_agree.py` (greedy agreement with an old-vs-old bit-exact control) shows
divergence only at near-ties, the same class as upstream with only N_SPLITS 16 -> 8.
`test_dsa_kernels.py` ALL PASS (H64/D512 cases run the new kernel). Prefill uses the
one-shot kernel, so PPL is unaffected. Declined (upstream kernel): Q_SPLIT / OUT_LATENT
(GLM-5.2 DSA-on-MLA), non-power-of-two D, H not tileable. Open: QC pools spill 169-243
VGPRs (H32 rotation in the KC loop) and re-enter the stream as scratch users with `-cq`
— watch for the "DS4 per-layer stall".

## WMMA GEMM backend for hgemm

`hgemm` / `hgemm_recon` (hence `exl3.py`'s reconstruct-then-GEMM prefill, `fp16.py`'s
mixed-dtype linear, `blocksparse_mlp.cpp`) run m > 8 on the MIT-licensed rocm_wmma_gemm
kernels (Adel Johar, ea3aa74) wherever that measured faster than hipBLAS. **Why:**
hipBLAS's fp32-output GEMM on gfx1151 is a VALU kernel with no matrix cores
(`Cijk_..._HSS_..._MT64x32x8`), 3-6x slower than its fp16 path, and DS4 needs fp32
output for its fp32 residual stream. DS4 pp2048 174 -> 216 t/s.

**Files:** `vendor/rocm_wmma_gemm/` (LICENSE + six headers; two `EXL3 MOD` edits:
separate output type with fp32 accumulate, `__half` bit-casts under
`__HIP_NO_HALF_CONVERSIONS__`), `wmma_gemm_rdna.hip(.h)`, generated
`wmma_gemm_table_rdna.hip.h` + `wmma_gemm_inst{0..3}_rdna.hip`
(`gen_wmma_gemm_table.py` from the library JSON + `wmma_gemm_tuned_gfx1151.json`).

**Selection.** `gcnArchName` picks a table: gfx1151 (library + local tuning),
gfx1100/1101 (library's gfx1100 table, fp32 only at m >= 512, untested on hardware);
other archs stay on hipBLAS, and kernel bodies compile only in device passes whose
table uses them. Within a table: exact (M, N, K), else closest K then smallest dM^2 +
dN^2, ties to the larger M. The library's tables start at M, N >= 1024 and their big
tiles idle the GPU on small-M prefill GEMMs, so `tune_wmma_gemm.py` measured every
config against hipBLAS on 520 shapes; fp32 output won everywhere, fp16 at 433. Entries
carry fp32/fp16 route flags. **Routing conditions** (else hipBLAS, never an error): m >
8, packed C rows, 16-byte aligned A/B/C, N % 8 == 0, K % block_k == 0 (a partial K tile
would leak an inf/NaN of A into a neighbouring row), each matrix < 2 GB. No allocation,
no host sync: graph replay is bitwise equal to eager.

**Numerics.** fp32 accumulate for both output dtypes. Not bit-identical to hipBLAS at
fp32 output: WMMA's f32 accumulator rounds each 16-deep step toward zero (bias ~-1e-5
at K = 4096, linear in K), still <= 2% of the fp32 gamma_K bound. fp16 output is the
round-to-nearest of the WMMA fp32 result. PPL DS4 -0.007%, Gemma -0.009%.

**Retune** after a toolchain bump or for a new arch: `gen_wmma_gemm_table.py --tune`,
rebuild, `tune_wmma_gemm.py` (~45 min, extends the JSON), `gen_wmma_gemm_table.py`,
rebuild. `--check` reports stale generated files. Verify with `wmma_gemm_check.py`.

## Pipelined MoE mainloop

The fused MoE prefill kernel `exl3_moe_kernel` has its own mainloop,
`rocm/quant/exl3_moe_inner_rdna.hip.h` (`EXL3_ROCM_MOE_PIPE=0` restores the shared
`exl3_gemm_kernel_inner`, which is not edited). DS4 pp512 **123 -> ~280**, pp2048
**217 -> ~370** t/s; Qwen3.8 pp512 272 -> 596; decode unchanged.

**The loop** (details in the header):
- Each wave loads only the B dwords its lanes decode (`exl3_lane_plan`) into a register
  ring DB = 4 k-tiles deep with compile-time slots; no block barrier per k-tile.
- The tiles core's decoder (`exl3_dq_tile_decode`) writes *transposed* into wave-private
  LDS, so the WMMA B fragment is 2 x `ds_load_b128` (was 16 stores + 16 loads + 8
  shuffles). `rdna_wmma.hip.h` fragment logic is untouched.
- A (gathered input, shared by 16 waves) is staged through LDS in 8-k-tile chunks,
  XOR-swizzled by row (`(r>>1)^(r>>3)`) instead of padded.
- LDS-only fences (`__builtin_amdgcn_fence(..., "workgroup", "local")`): no vmcnt(0)
  drain, no `buffer_gl0_inv`.
- Row tiles 16/32/48/64 per expert inside the kernel, sharing one decoded B fragment.
- g, u, d through one call site (three made clang outline the inner). The column-end
  reduce is inlined per unrolled step behind `[[unlikely]]` (an out-of-line reduce
  behind a `switch` made clang tail-merge the steps with a vmcnt(0) per tile).
- Budget <= 192 VGPR (`amdgpu_waves_per_eu(8)`), <= 32 KB LDS: **two blocks per WGP**,
  launched only when `hipOccupancyMaxActiveBlocksPerMultiprocessor` says 2 (5 groups x
  8; Python sizes 5 expert buffers). A 64 KB request never relies on two blocks.
- rocm_py raises the fused-row cap to 512 (`EXL3_ROCM_MOE_FUSED_ROWS`) so hot experts
  stay in the kernel. Group width stays upstream's 8: every block owns whole output
  columns for DS4, so no fp16 partial sums.

**Where the time goes — read before optimizing further.** The old loop was
latency-bound (1.4 us per k-tile). Once pipelined, ring depth stopped mattering (DB
2/4/8 within 2%); the loop is **issue/VALU-bound**. By removal: no decode -30%, no WMMA
-20%, no A staging -7%, no LDS transpose -5%, no B loads -2%. Memory-only streams at
~180 GB/s, so the gap is compute. At T = 1792 the column-end reduce is ~13% of GEMM
time (`-DEXL3_MOE_PIPE_PROF`).

**Numerics.** On the same grid (`EXL3_ROCM_MOE_BPS=1`) **bit-identical** to the old
mainloop (`bench_moe_kernel.py --check`, `moe_inner_bench`). The default grid changes
the stream-K split (removing the fp16 partial-sum round trip for DS4): `moe_ref32` error
equal or slightly closer; PPL DS4 +0.009%, Qwen3.8 +0.023% (mostly the row cap).
Instances: gfx1151 168-192 VGPR, a few spills outside the loops. Mixed-K models get the
16-row tile only. Open: fewer VALU per weight in K2 decode, a cheaper column-end reduce.

## DSA prefill MQA kernel

`_dsa_attn_kernel` (one-shot sparse attention, every DS4 prefill chunk with R > 8 rows)
was 32.8% of pp2048 GPU time. Same root cause as decode: BLOCK_H 32 heads x full output
width plus a resident WMMA q tile: 256 VGPR, 2878 spills, 5164 B scratch. The best of 36
upstream retunes was 1.55x and still spilled 1005.

**Fix.** `rocm_py/dsa_prefill_rdna.py:_dsa_prefill_mqa_kernel`, a drop-in with the same
arguments, layout and features (window ring + chunk, sinks, eq. 26 de-rotation,
group-major store, dense / gathered pool, online packed pools, NC_CHUNK, NC_BLOCK
DSpark draft). One program = HP heads x one BD column block x one query row; the D =
512 score reduction is a runtime KC loop with q re-read. **Production tiling:** HP 64
(all heads share the latent KV head), BD 512 (no score recompute), KC 32, BLOCK_N =
BLOCK_W 64, 16 warps: 5 spills / 20 B (25 / 104 gathered). Wiring: `dsa_attn` looks
`_dsa_attn_kernel` up as a module global at call time, so a launch proxy routes eligible
calls with grid `(R * H/HP * D/BD,)`. Declined: Q_SPLIT / OUT_LATENT, non-power-of-two D
(V3.2's 576), H < 16 or not a multiple of HP, D_r == 0.

Sweep facts (`bench_dsa_prefill.py`, 162 compiled variants): KC 128 and BN 64 at 4 warps
spill hundreds; the winner beat every zero-spill variant (9.8 vs 14.2 ms at R2048); the
residual 20 B scratch is allocator noise. Rejected: KSTAGES 2, BLOCK_W < BLOCK_N, 32
warps (1024 threads x 256 VGPR, HSA INVALID_DISPATCH_PARAMETERS), key splits for short
tails (R 64 per-row cost within ~20% of R 2048).

Result: 5-7x per call; DS4 **pp2048 368 -> 518**, pp512 +25%, regen tails -15 to -21%.
Error vs fp64 unchanged (different order); PPL +0.025%. Open: online packed pools (-cq,
tail < 64 rows) spill 307 VGPRs.

## Qwen3.8 decode, prefill and n-gram table lock

About a quarter of Qwen3.8-Flash-Next's ~5.0 GB/token decode stream is fp16
GatedResidual matrices on their own kernels (gr_dots / gr_finalize), not the EXL3 GEMV,
and its full-attention layers ran a graphed Triton kernel compiled without AMD's
buffer-op specialization. tg128 26.6 -> 29.0, MTP ndt=2 39.3 -> 43.9, pp2048 750 -> 848.
DS4 runs none of these paths (mHC, routing_ds3, DSA/MLA) and was A/B'd flat.

| switch | change | numerics |
|---|---|---|
| `EXL3_ROCM_GR_DOTS` (`_GR_RB`) | `gr_dots_rows_kernel` (`hc_mix_rdna.hip`): 4 fn rows per block, stream stack in registers, a row's loads in flight; 60.8 -> 36.3 us per site | bit-identical |
| `EXL3_ROCM_GR_PREFILL` | `torch.ops.exl3_rocm.gr_gate_mean`: prefill gate-mean tail in one pass (3.6 -> 0.4 ms per site at R 2048) | bit-identical to torch (under `#pragma clang fp contract(off)`: torch rounds the product before the sum) |
| `EXL3_ROCM_BC_BUFOPS` | bc_attn's AOT GQA decode split kernels (paged + QSA sparse) compiled with the `tt.pointer_range = 32` / `tt.divisibility = 16` attributes the Triton JIT adds on AMD, at 8 warps / 1 stage; gated per layer on K/V storage < 2 GB. In-model ctx 512 91.5 -> 31.8 us, QSA 8K 117 -> 31.9, MTP verify (q_len 3) 163 -> 62 | bit-identical (decode_bitwise at 21- and 4001-token prompts) |
| `EXL3_ROCM_ROUTER_STD_MR` | `routing_std` at 2..8 rows on the multi-row router GEMV (105 -> 40 us per verify) | verify rows use the m == 1 chain |
| `EXL3_ROCM_PREFILL_HD256` | paged prefill at head_dim 256: 1 stage, 128-row tile (64 below 128 query rows); upstream's spilled (256 VGPR + 772 B); 2-3x per kernel | PPL +0.074%, error vs fp32 unchanged |

- The JIT path was never the problem: the JIT launch of the same kernel took 13.5 us;
  the graphed AOT compile lost the specialization.
- **BC attention warps/stages, in-model sweep** (us, ctx 512 / 8K QSA): 4/2 38.9 / 51.9
  (scratch 276 / 524 B), 8/2 33.8 / 48.6, **8/1 31.8 / 31.9 (0 scratch)**, 16/1 33.4 /
  77.1, 4/1 38.0 / 44.3, 2/2 63.9 / 88.9.
- Prefill at hd 256 (q 1792 fresh): upstream 64x32 w8 s2 5770 us, 128x32 w8 s1 2917; at
  q 64 after 4096 the 64-row tile wins (612 vs 857), hence the 128-row threshold.
- **gr_finalize stays upstream's form** (~180 GB/s): hoisting the up-gate loads above the
  prologue with DPP butterflies was slower (36.5 -> 42.4 us; with the prologue loads
  issued first, 36.5 -> 37.9 us). Hoisted loads make the prologue wait: vmcnt retires in
  order.
- GR prefill matmuls on the WMMA backend would need transposed B copies (+1.3 GB); the
  n-gram memory budget does not have it. The PLE / n-gram host path needs nothing.

**N-gram table lock (`-ngl`).** `exllamav3/rocm_py/ngram_lock.py` (used by `server.py
-ngl`, `run_bench.py --ngram_lock`; no upstream edits). Modes: disk streaming (default),
`-ngr` RAM, `-ngl` locked RAM. `-ngl` loads exactly as `-ngr` (one contiguous CPU slab)
then `mlock(2)`s that range: no copy, no read-path change, no speed cost. Preflight
before loading: `RLIMIT_MEMLOCK` must cover the table (the soft limit is raised to the
hard limit; `CAP_IPC_LOCK` bypasses; otherwise it exits with `ulimit -l` / limits.conf /
`LimitMEMLOCK` / `prlimit` / `setcap` recipes), and weights + table + KV + headroom
(`EXL3_NGRAM_LOCK_HEADROOM_GB`) must fit MemAvailable — a locked table is never
reclaimed, so it refuses rather than meet the OOM killer. `/props` reports
`ngram_table`. This box's hard limit (15.6 GiB) is below the 36.4 GiB table; a full lock
needs root to raise it.

Open: hc_apply fold into the next GatedResidual site; GDN launch fusion; vendored fla
`recompute_w_u_fwd_kernel` spills 1608 B.

## HIP graphs

`graph.cu` is excluded on ROCm for `rocm/graph_rdna.hip`, which gates capture on the
loaded runtime: `hipRuntimeGetVersion() >= 71400000` (7.14+) turns graphs ON; older
runtimes run each BC step's `run_gr()` eagerly on the live stream (the path of each
slot's first warmup). `EXL3_ROCM_HIP_GRAPHS=1/0` overrides (bisect handle); rocm_py
reports the state in `describe()`. A 7.2.4 user never hits the 7.2.x capture hang; a
ROCm 10 user gets graphs without a flag. `graph_rdna.hip` checks every node-update rc
(rocm-systems PR #10714, kernarg-exhaustion stale packets): keep it that way.

**Proven vs suspected**, so nobody re-litigates the wrong part:
- 7.2.x: intermittent capture stalls (same class as vLLM's ROCm 7.2.x capture hang;
  llama.cpp ships HIP graphs off), an old MoE BC graph route that died with "Graph
  update failed" + segfault, and no capture-time validation (pytorch#155684).
  `stream_wedge_check.hip` hangs 7.2.4 reproducibly with no graph involved.
- The patch/replay primitives are **not** the defect: the graphpatch and graph_order
  checks pass on 7.2.4.
- 7.14 and ROCm 10 (HIP 7.15): graphs **work** — all checks pass, multi-job generation
  coherent, `decode_bitwise` graphs-off vs -on bit-identical (incl. the fused and
  multi-row patch sites), `chat_probe` token-identical to eager.
- Graphs buy **nothing** in decode speed (flat on every model and runtime). An old
  "-15% decode without graphs" figure does not reproduce; treat it as stale.

**Dispatch-gap census** (why graphs are flat). `rocm_tools/gap_profile.py` (Kineto,
1-token prompt): graphs collapse the host side as designed (53k `hipLaunchKernel` ->
7.6k plus ~96 `hipGraphLaunch` per token), yet the device timeline is identical: same
~2.1 us median inter-kernel gap, same busy time. That gap is the command processor's
per-dispatch latency between dependent kernels, which graph replay on ROCm does not
remove; the host was never the bottleneck. Recoverable only by launching **fewer**
kernels ("Launch-count fusion", "Decode leftovers"). At the time: ~1950 / 1620 / 2170
launches per token (Laguna / Gemma / DS4), gaps 10% / 2.9% / 7.8% of span — fusion pays
most on fast MoE tokens, least on big dense.

### DS4 per-layer stall

~41 big gaps per token (once per layer, 100-500 us, ~8% of decode) sat between
`dsv4_compress_store` and `_dsa_attn_split` on the **same** stream with the next kernel
enqueued: a device-side stall, unaffected by graphs. `_dsa_attn_split_kernel` at
upstream tuning (BLOCK_H 16, 4 warps) had ~2050 spills and 5.3 KB/item scratch — the
only scratch user in the decode stream, so every layer's dispatch after scratch-free
kernels paid the queue's **scratch reconfiguration**. BLOCK_H 8 + 8 warps (438 spills)
cut the big gaps 1369 -> 45: DS4 decode 15.9 -> 17.9 t/s. Deeper-spill variants were
worse: past the stall threshold, tile shape matters more than residual spills. Shipped
as `EXL3_ROCM_DSA_TUNE`: `bc_dsa.BLOCK_H` via module attr, warps via a `_compile_kernel`
wrapper keyed on kernel name, rebound in bc_attn, bc_dsa and bc_mla (each holds its own
from-import; bc_mla's DSA-on-MLA hardcodes BLOCK_H 16 and gets only the warps half). The
DSA decode MQA kernel has since replaced the split kernel. **Rule: a lone scratch user
in an otherwise scratch-free stream stalls the queue at every switch** — check this
first when a spilling Triton kernel appears in decode (GLM-5.3's MLA decode / absorb
kernels are the current instance). Read spills from `.vgpr_spill_count` /
`.private_segment_fixed_size` in `ck.asm["amdgcn"]`. Watch item:
`test_mla_dsa.py::test_dsa_selection[300]` failed once, unreproduced — bisect with
`EXL3_ROCM_DSA_TUNE=0` first if it recurs.

### The long-context "coherency collapse" was not graphs

It was sampler arithmetic: OAI frequency/presence penalties with TabbyAPI's default
`penalty_range = max_seq_len`, over `past_ids` including the prompt. At 8K context a
common token carries ~-35 logits; function words die, then generation flees to
never-used tokens. Fix: bounded penalty_range (512-2048) or freq_p ~= 0. Still-valid
exonerations: attention parity to 16K incl. the 8192 split, rope to pos 32K, YaRN,
cache rotate, per-position NLL flat to 16K. "Replay repeats itself on long context" was
a client stop-token gap: Laguna ends turns with `</assistant>` (in
`config.eos_token_id_list`); exl3_server unions both EOS lists.

**Greedy loop-collapse is not a coherence metric**: greedy loop-collapse probes are
decoding chaos (1/8 collapse in every config, at different knife-edge points).
Coherence is human-judged on `chat_probe.py` / `examples/chat.py` output, or measured by
PPL and bitwise gates. (Bare `tokenizer.encode` drops BOS and fakes corruption.)

## ROCm wheel stack (TheRock)

Production stays on system ROCm 7.2.4. The pip stacks — 7.13 (torch 2.11), 7.14 (torch
2.15 nightly) and ROCm 10.0 (`https://stable.repo.amd.com/rocm/whl-next/`, torch 2.13 +
triton 3.8, HIP 7.15; branch `rocm-10`) — all pass validation, but **decode is flat**:
this port's own kernels already run near roofline; community gains come from stacks
bottlenecked on hipBLASLt / attention libraries or host overhead. ROCm 10 is faster
only where torch's own fp16 matmuls take hipBLASLt (DS4).

Build (ROCm 10): scrub the login shell's `/opt/rocm` leakage (`env -i`), then
`PATH=$SDK/bin:... ROCM_PATH=$SDK ROCM_HOME=$SDK pip install -e . --no-build-isolation`
with `$SDK = site-packages/_rocm_sdk_devel` after `pip install "rocm[devel]"` and
`rocm-sdk init`. **ROCM_HOME matters**: torch's extension builder takes the rpath from
it; without it the .so carries RUNPATH `/opt/rocm-7.2.4/lib` (the duplicate-runtime
trap waiting to happen). Harness binaries carry no rpath: run with
`LD_LIBRARY_PATH=_rocm_sdk_core/lib`. Check `/proc/self/maps` for the mapped runtime.

Traps:
1. Wheels are runtime-only: building needs `rocm[devel]` + `rocm-sdk init` (core hipcc
   cannot find device bitcode, and there are no thrust headers).
2. 7.13's `amd_hip_cooperative_groups.h` defines `this_cluster()` without `inline`:
   duplicate symbols at `-fgpu-rdc` link (fixed in 7.14).
3. **Dual HIP runtime via CudaDrv's dlopen**: a second runtime beside torch's works by
   luck at the same version and fails at the first Triton launch otherwise
   (hipErrorContextIsDestroyed / "CUDA driver error"). `cuda_drv_rdna.cpp` tries
   `dlopen("libamdhip64.so.7", RTLD_NOLOAD)` first (torch >= 2.15 loads it RTLD_LOCAL, so
   `dlopen(nullptr)` misses it), then the loaded image, then named dlopens.
4. triton >= 3.6 unloads modules in `CompiledKernel.__del__`, which fires mid-capture
   if BC kernels compile lazily. Not firing today (BC kernels compile eagerly; triton
   3.8 unloads only launched modules, and `bc_attn._compile_kernel` never launches the
   Triton object). Workaround if it returns: no-op the destructor.
5. torch >= 2.14 needs `-std=c++20` (setup.py handles it).
6. `requirements_rocm.txt`'s `triton-rocm` **clobbers** pytorch-triton-rocm on a
   pytorch-index stack; install minus triton, then the exact pinned
   `pytorch-triton-rocm`.
7. The devel SDK's lib symlinks resolve into torch's venv `_rocm_sdk_libraries/`;
   symlink it into the SDK venv or linking fails ("unable to find -lrocblas...").

All stacks: DeepSeek-V4-Flash prints Triton "no matching matrix core intrinsic for wmma
version 1" from `dsa_triton.py`; Triton falls back and output is coherent.

## Profiling on this machine

- **At most THREE rocprofv3 counters per pass** ("Request exceeds the capabilities of
  the hardware to collect"). `OccupancyPercent MemUnitBusy FETCH_SIZE` is the useful
  triple.
- **PyTorch processes need torch's bundled rocprofiler libs moved aside.** The wheel's
  `librocprofiler-sdk.so` / `librocprofiler-register.so` (`RPATH=$ORIGIN`) differ from
  the system copies, two instances load and registration fails ("Configuration request
  occurred outside of valid rocprofiler configuration period"). `LD_PRELOAD`,
  `LD_LIBRARY_PATH`, `LIBKINETO_NOROCTRACER` all fail; rename the two files in
  `torch/lib` and restore afterwards. rocprofv2 does not support Strix Halo.
- **Counter collection (PMC) deadlocks co-resident kernels**: it serialises
  dispatches, breaking the co-residency `grid.sync()` needs (`gemm_coop_check` hangs at
  2% GPU); graph-captured kernels fail with "Timeout while waiting for queue sync".
  Exclude `exl3_moe` and cooperative kernels from counter passes (`bench/run_counters.sh
  EXCLUDE=`); `bench_mgemm.py` reaches mgemm outside any graph for this reason.
- A tool that calls `os._exit()` produces **no CSV** (rocprofv3 writes from exit hooks).
  Return normally; the teardown segfault comes after the flush.
- **Kineto device capture can wedge machine-wide.** After a HIP process SIGSEGVs,
  torch.profiler can return zero device events in every fresh process; a reboot clears
  it. If a profile shows 0.000 s GPU time with a populated host table, test capture with
  a trivial matmul before trusting any "HOST-BOUND" verdict. A Kineto session started
  mid-generation captures nothing on ROCm.
- **Kineto inflates per-launch host cost ~10x.** Never call decode host-bound from it.
- **Kernels-in-window is not decode-kernels**: a profiled job runs its prompt's prefill
  in the same `iterate()` loop. Three wrong localizations came from that (incl. "30% of
  decode in exl3_moe" — decode never calls it at bsz 1). `profile_decode.py` uses a
  1-token prompt; `-ndt N` profiles MTP.
- **Never run `stream_wedge_check` without asking** (hangs 7.2.4; machine-wedge risk).

## Test status on RDNA

`tests/` hardcode a device index (`cuda:2`, `cuda:1` in `test_reconstruct_had.py`), so a
single-GPU machine rewrites them first. `bench/run_gates.sh` does that in scratch copies
(upstream files are not edited) and runs `mgemv_check`, `test_reconstruct_had`,
`test_dsa_kernels`, the WMMA gate and pytest (985 passed / 30 skipped at v1.5.3).

Several `tests/test_*.py` are `main()` scripts, not pytest modules: pytest says "no
tests collected", which reads as a pass. **Run those directly** — they hold the
reference checks for the newest kernels (`test_reconstruct_had.py`,
`test_dsa_kernels.py`).

Upstream test defects (fail the same way on CUDA): `test_ext_norm_` (calls
`ext.rms_norm` with 4 of 8 parameters; with the real signature the kernel matches fp32
to 4.2e-4), `test_kv_quant` and `test_dsv4_compress_kernel.py` (arity), `test_dsv4_cached`
/ `test_dsv4_state` (import an uncommitted `compare_deepseek_v4_hf_`), `test_qgemm`,
`test_quant_fn`, `test_ple_prefetch_gen_`, `test_ngram_prefetch_` (models at
`/mnt/str/...`), `test_mla` / `test_mla_dsa` (the scratch FakeSTC needs the loader's
`arena=` kwarg). `test_routing_gemm_det.py`, `test_gr_mix_tiled.py` are CUDA-only.
**A `TypeError: incompatible function arguments` from `tests/` is an upstream staleness
signal**; check the declaration before suspecting the port. Known flakes:
`test_dflash2.py::test_topk_cuda_matches_torch` (never fails alone),
`test_mla_dsa.py::test_dsa_selection[300]`.

### End-to-end generation

Coherent on Gemma-4-31B, GLM-4.6V, Laguna-S-2.1, DeepSeek-V4-Flash, Qwen3.8-Flash-Next,
GLM-5.3-Flash, MiMo-V2.6-Flash. The dense model is the cheapest control for an MoE
fault; run it first. A human judges `chat_probe.py` / `gen_smoke.py` output.

### Teardown segfault after model load

Any process that has loaded a model exits with SIGSEGV *after* all output and work are
complete. It needs a loaded model, is not model-specific, and shows no Python traceback
under `PYTHONFAULTHANDLER=1`: native teardown (static destructor ordering against an
already-torn-down HIP runtime). `os._exit()` after the work avoids it — why the bench
tools and the server exit that way, except under rocprofv3 (needs the normal exit to
write its CSV; the segfault comes after the flush). Harmless to generation; not known
whether it predates v1.4.1.

## Verification tools

Under `rocm_tools/` (build scripts write to `/tmp` or `$OUT_DIR`). One row per tool.

| tool | checks |
|---|---|
| `wmma_check.hip` | WMMA operand order and fragment layout vs a CPU reference |
| `wmma_gate.hip`, `build_wmma_gate.sh`, `wmma_gate.golden` | bit-exact WMMA layout / regression gate of `rdna_wmma.hip.h` against a golden file |
| `gemm_check.hip` | GEMM kernel inner vs a CPU reference |
| `gemv_check.hip` | GEMV, all three dot cores vs reconstruct; `GEMV_SWEEP=1` DRAM-resident bandwidth sweep |
| `gemm_coop_check.hip`, `build_coop_check.sh` | cooperative launch vs the same work without it |
| `gemv_tiles_bench.hip`, `build_gemv_tiles_bench.sh` | standalone direct vs tiles core at every (U, T), bit-for-bit and us / GB/s; the build script also builds gemv_check and moe_inner_bench (`SRC=`) |
| `moe_inner_bench.hip` | old vs pipelined MoE mainloop on one expert shape, bit-for-bit and ms / GB/s |
| `exl3_stack_check.py` | ext.exl3_gemm through host dispatch and autotune vs reconstruct + hgemm, no model |
| `mgemv_check.py` | mgemv vs the cooperative kernel on real weights, every routing config, masked-2tok bitwise |
| `mgemv_bitwise.py` | multi-matrix path bit-for-bit against references saved on another build |
| `decode_bitwise.py` | 48-step greedy decode, every logit vs a saved reference |
| `multirow_check.py` | multi-row GEMV: row r == the m == 1 call |
| `half_gemv_check.py` | half-integer rates: GEMM / GEMV cores / multi-row / mgemm / moe_decode |
| `frac_check.py` | 1.5 / 2.5 / 3.5 bpw reconstruct and GEMM vs upstream unpack + decode |
| `isa_diff.py` | two .so builds compared instruction by instruction (isolation proof) |
| `moe_ref32.py` | fused MoE and per-expert path vs an fp32 reference |
| `moe_check.py` | fused MoE vs the per-expert path |
| `bench_moe_kernel.py` | `ext.exl3_moe` alone on synthetic expert tables; `--check` pipe 0 vs 1 bitwise |
| `bench_moe.py` | real-model MoE layer timing |
| `bench_mgemm.py` | mgemm from Python outside any graph (profilable), coverage-checked |
| `bench_gemv_kernels.py` | real GEMV dispatches on DS4 shapes; `--ab` tiles 0 vs 1 |
| `bench_gemv_vs_gemm.py` | GEMV vs GEMM routing at decode shapes |
| `bench_dsa_decode.py` | DSA decode split + combine: fp64 check, graph-replay us, spills; `--sweep-old/--sweep-new` |
| `bench_dsa_prefill.py` | DSA one-shot prefill kernel: timing and sweeps |
| `decode_agree.py` | greedy-agreement A/B of two builds / switches (prefix match, KL, top-10) |
| `wmma_gemm_check.py` | WMMA GEMM backend correctness (pinned configs, partial tiles, strides, replay) |
| `bench_wmma_gemm.py` | WMMA GEMM backend vs hipBLAS |
| `tune_wmma_gemm.py`, `gen_wmma_gemm_table.py` | WMMA GEMM tuning and table generation (`--tune`, `--check`) |
| `gen_det_siblings.py` | generates the declining deterministic-kernel siblings (anchored edits) |
| `hgemm_narrow_probe.py` | narrow-N hgemm across rocBLAS / hipBLASLt / ATen / ext |
| `gr_mix_bench.py` | GatedResidual mix and prefill gate-mean bit-identity and timing |
| `shfl_up_scan_check.hip` | `__shfl_up_sync` scans with distinct per-lane values |
| `attn_check.py` | upstream Triton paged attention vs a reference (no ext needed) |
| `attn_8k_check.py` | paged attention vs fp32 at KV 4K-16K across the 8192 prefill split |
| `bench_prefill_tiles.py` | prefill attention tile choice |
| `bench_decode_splits.py` | decode attention split count vs the shipped heuristic |
| `bench_membw.py` | achievable memory bandwidth |
| `bench_compute.py` | peak fp16 GEMM vs bandwidth: is decode bandwidth- or compute-bound |
| `bench_model.py` | model prefill / decode timing, median of repeats |
| `bench_mtp.py` | plain vs MTP / drafter decode, acceptance, text |
| `dflash_census.py` | DFlash round census: verify, draft forward, per-call drafter linears |
| `nan_locate.py` | first module whose output goes non-finite |
| `profile_decode.py` | rocprofv3 / Kineto decode profile, 1-token prompt; `-ndt` for MTP |
| `gap_profile.py` | dispatch-gap census from a Kineto trace |
| `chat_probe.py` | multi-turn + concurrent-job coherence probe with repetition scores (human-judged) |
| `gen_smoke.py` | sampled generation smoke test (human-judged, not a metric) |
| `hipcc_probe.sh` | per-file compile probe without rdc; `GPU_ARCH=` for other archs (`--all` 134/134 on gfx1151/1100/1101/1200/1201) |
| `graphpatch_check.hip`, `graphpatch_module_check.hip`, `graphpatch_multinode_check.hip`, `probe_module.hip` | hipGraphExecKernelNodeSetParams semantics: runtime-, module- (Triton-style, via `probe_module`) and multi-node-patched graphs |
| `graph_order_check.hip` | back-to-back hipGraphLaunch ordering under deep queues |
| `stream_wedge_check.hip` | spin kernel + pageable hipMemcpyAsync; **hangs 7.2.4** — ask before running |
| `exl3_server/` | OpenAI-compatible server (`-ngl`, warmup, EOS union) |

Outside `rocm_tools/`: `bench/run_bench.py` (suite timing, `--repo` for another
worktree), `bench/run_ppl.sh` (wikitext2 PPL, 100 x 2048), `bench/run_gates.sh`,
`bench/run_counters.sh`, `bench/regress_bitwise.py`. The bitwise gates are the default
acceptance for any "same arithmetic" change: save references on the old build, compare
on the new.

## Syncing to a new upstream release

Every sibling derives from exactly one upstream file. Before syncing one, measure our
drift from the upstream file it was generated from and upstream's churn since:

```sh
diff <(git show vOLD:exllamav3/exllamav3_ext/quant/reconstruct.cu) \
     exllamav3/exllamav3_ext/rocm/quant/reconstruct_rdna.hip | grep -c '^[<>]'
git diff --numstat vOLD vNEW -- exllamav3/exllamav3_ext/quant/reconstruct.cu
```

Low drift + high churn is the *cheap* case (regenerate, inherit upstream's code); high
drift is expensive (deviations re-applied by hand). Methods, in order of preference:

1. **Regenerate by `sed` on include lines** — siblings whose whole delta is includes
   (`reconstruct_rdna.hip`, `moe_handoff_rdna.hip`, `exl3_dq_rdna.hip.h`).
2. **Scripted re-application** of anchored edits, asserting each anchor matches exactly
   the expected number of times so a moved anchor fails loudly (`rope_rdna.hip`,
   `exl3_gemm_kernel_rdna.hip.h`, `gen_det_siblings.py`).
3. **Three-way merge** (`git merge-file`, base = old upstream, ours = sibling, theirs =
   new upstream) when a deviation restructures a block upstream owns
   (`exl3_gemm_rdna.hip`, `routing_rdna.hip`, `hc_mix_rdna.hip`).

Never hand-copy: the earlier port replaced `__funnelshift_r` with a non-wrapping
`fshift` that way (it landed only in dead code). After syncing, the diff against the
new upstream file must be *exactly* the documented deviations. Then: `hipcc_probe.sh
--all`, bitwise gates against references saved on the pre-merge build,
`bench/run_gates.sh`, PPL, and the DS4 / Qwen / Gemma bench table.

What bit before:
- **A stale sibling can be silent memory corruption**, not a compile error (v1.4.4's
  `moe_handoff` flag-region layout change; the MoE tile-K constant).
- **`EXL3_MGEMM_ARGS` / `EXL3_GEMM_T_ARGS` must match upstream's `exl3_kernel_map.cuh`
  exactly**: the comp_units instantiate against them.
- **The autotune hash**: a new key re-tunes, a re-tune may pick another grid and so
  another split-K order, which alone breaks bit-identity.
- **Upstream numerics changes look like porting bugs** until switched off.
- **New device code with warp ops** needs a distinct-per-lane probe (v1.4.4's
  `dsa_topk.cu` scans: `shfl_up_scan_check.hip`, PASS).
- **CC-major checks** pick Blackwell / tensor-core paths on RDNA (HIP reports 11).
- `requirements_rocm.txt` duplicates the dependency list; mirror upstream swaps.
- No upstream `.py` file carries a ROCm edit: Python changes are `rocm_py` hooks (only
  the two-line hook at the end of `exllamav3/__init__.py` touches an upstream file),
  logged in `rocm_patches/UPSTREAM_PY_CHANGES.md`.

History: v1.3.0 -> v1.4.1 touched 5 siblings (the fragile GEMM inner / MoE / comp_units
files were byte-identical). v1.4.1 -> v1.4.4 had zero conflicts and added the mgemv
`num_tokens > 1 && min_index >= 0` decline. v1.5.0 below, v1.5.3 in "v1.5.3 sync".

### v1.5.0: what the RDNA layer does

| upstream | RDNA layer |
|---|---|
| `exl3_mgemm` sliced mode (one-launch Q/K/V) | mirrored into the GEMM siblings; mgemv bypassed in sliced mode; **Python keeps it off** (`EXL3_ROCM_QKV_SLICE=1`) |
| fused MoE tiers, deterministic slots + gather, 32/64-row tiles | applied verbatim; 32/64-row tiles fall back to the 16-row instance; rocm_py `MTILE=False` |
| MoE decode via `exl3_moe_coop` | stub; rocm_py reinstates the mgemm route ("MoE decode route restored") |
| batched expert reconstruct, `hgemm_batched` | applied (shim maps the strided-batched call); **Python keeps it off** (`EXL3_ROCM_BATCH_RECON=1`) |
| `hgemm_f16acc`, quantizer sm_120 LUT | stub / `quantize_tiles_use_optimized()` false |
| `rope.cu` lane-0 guard | adopted upstream; sibling back to one line |

## v1.5.3 sync

Upstream v1.5.0 -> v1.5.3 (133 commits) under the perf stack. Conflicts only in
`setup.py` (ROCm backend block kept; upstream's `util/cuda_flags.py` loader added for
CUDA) and `attention_fn/triton_paged.py`, taken **verbatim from upstream**: the fork's
`_is_rocm` narrow-kv prefill tile moved to rocm_py (`EXL3_ROCM_PREFILL_HD128`) and its
decode-split comment is the WGP note under "Hardware". `rocm_patches/BASE` = the merge.

**Siblings.** sed-regenerated: `reconstruct_rdna.hip` (+ `bits_k.cuh`),
`moe_handoff_rdna.hip`, `exl3_dq_rdna.hip.h`, new `quantize_tiles_frac_kernel_rdna.hip.h`.
Half-integer hand ports: kernel map (`half_k` in `EXL3_GEMM_T_ARGS` at upstream's
position, `_H` instance macros, half-aware `exl3_gemm_smem_bytes`; half rates
shape-select as K + 1 like upstream), GEMM kernel / inner, `exl3_gemm_rdna.hip` (`float
K` at the mgemm boundary, half-aware autotune / selection), `exl3_moe_rdna.hip` (K in
half-bit units), comp units `exl3_comp_unit_h{1,2,3}.hip`. Three-way merges:
`routing_rdna.hip` (upstream's `gate_i8` / `gate_sb` args + deterministic router math)
and `hc_mix_rdna.hip` (upstream's GatedResidual decode pair merged verbatim; default
stays on the RDNA rows kernel; `EXL3_ROCM_GR_DOTS=0` selects upstream's dispatch).
Generated declining siblings `routing_gemm_rdna.hip`, `hc_mix_tiled_rdna.hip`: upstream's
deterministic int8 tensor-core kernels (det_gemm.cuh PTX) compile out under the shim,
but `routing_gemm_det_fits()` would still select them (HIP reports CC major 11). Shim:
empty `cuda_shim/mma.h`, `cudaDevAttrComputeCapability{Major,Minor}`, a real
`__cvta_generic_to_shared`. `hipcc_probe.sh` and `ROCM_EXCLUDE` gained `/routing_gemm.cu`
and `/hc_mix_tiled.cu`.

**Autotune hash rule.** Upstream mixes `half_k` into the key always; here only when
set, so integer-K keys equal v1.5.0's and the on-disk tune cache stays valid (the first
merge build lost bit-identity from a re-tune alone).

**Features on ROCm:** fractional 1.5 / 2.5 / 3.5 bpw supported (and on the fast paths,
"Half-integer bitrates"). Unsupported (stubs / declining siblings): int8 GEMV,
`exl3_moe_coop` incl. `sh_coop` (rocm_py sets `block_sparse_mlp._moe_shared_coop =
False`), `hgemm_f16acc`, deterministic int8 router / tiled GatedResidual (rocm_py stops
`_gate_t` building the int8 tables). Per-device smem ladders are active with
`smem.smem_limit()` seeded from `shared_memory_per_block`; the DSA proxies answer the
ladder's compile-only probe for the MQA kernel they will launch, and the buffer-op
compile wrapper mirrors `bc_attn`'s `BCKernelTooLarge` gate. `model.warmup()` runs
(server `-nwu` = `-nw`). Upstream lowered GatedResidual `FUSED_MAX_R` to 8 for a tiled
int8 path ROCm lacks, so rocm_py keeps 32. SAM corpus, DFlash2, MiMo-V2, Kimi-Linear
build (portable code).

**Numerics: what moved.** Every difference from the pre-merge stack is an upstream
numerics change; with those switched back the merge is **bit-identical**
(decode_bitwise), except (3):
1. **Router math** (DS4, Qwen3.8 top-k weights): FMA-only `exp_det` / `softplus_det`,
   correctly rounded div / sqrt, for cross-architecture TP agreement.
   `EXL3_ROCM_ROUTER_DET=0` restores v1.5.0 fast math. The only v1.5.3 numerics change
   on DS4's decode path.
2. **GDN prefill** (Qwen3.8): fp16 projection output, token-major conv read; upstream's
   `EXL3_GDN_PROJ_FP32=1` / `EXL3_GDN_CONV_TOKEN_MAJOR=0` restore.
3. **Paged decode attention** (Qwen3.8 full attention, Gemma): split count
   `cdiv(kv, 4 * block_n)` -> `cdiv(kv, block_n)` plus a sub-tiled combine; ~1e-5 max abs.
   **No switch** (restoring needs an upstream-file edit or a hook on `BCAttn`'s split
   helper; maintainer call). Consequence: Qwen3.8 plain and MTP greedy are no longer
   token-identical (verify and plain get different split counts); DS4 (DSA) still is.

Tokens identical on every tested prompt; PPL DS4 -0.025%, Qwen3.8 -0.26%, Gemma exact;
performance within noise (`model.warmup()` costs ~1% prefill; `-nw` recovers it).

## Half-integer bitrates on the fast paths

At the v1.5.3 sync every RDNA fast path declined half K, so MiMo-V2.6-Flash 2.27 bpw
(17664 expert tensors at 2.5 bpw) ran the cooperative `exl3_mgemm_kernel` at m = 1 in
decode (65% of decode, ~32 GB/s) and the non-pipelined `exl3_moe` in prefill. Now: MiMo
tg **13.2 -> 30.0 t/s**, pp512 158 -> 341, MTP ndt 2 12.1 -> 35.3.

| switch | change |
|---|---|
| `EXL3_ROCM_HALF_GEMV` | half rates on the multi-row GEMV (single + multi, m = 1..8, weighted) and `moe_decode`; C++ reads it per call, rocm_py at load |
| `EXL3_ROCM_HALF_MOE_PIPE` | uniform half-rate MoE layers on the pipelined mainloop (`comp_units_rdna/exl3_moe_inst_h{1,2,3}_cb2.hip`) |

**Mechanism: a pseudo width, not a new template parameter.** A K + 0.5 bpw tensor rides
the integer `bits` slot as `EXL3_HALF_BITS(K) = 16 + K` (17 / 18 / 19). `Exl3Width<bits>`
(`exl3_gemv_tiles_rdna.hip.h`) gives `half`, `ka`, `tile_bytes` (32 * bits, or 16 * (2K
+ 1)); every half branch is an `if constexpr` on it, so integer instantiations keep
their names and code (`isa_diff.py`: all 3354 existing gfx1151 device functions
**instruction-identical**), and bodies templated on bits take half rates with only the
tile size changing. Callers pass the pseudo width only for half-rate weights and only
while the switch is on.

**Decoder.** Upstream's `dq8_half`: positions alternate K and K + 1 bits; a lane's 8
windows are two groups of four, each two dwords and one funnel shift < 32. Lane plan 4
dwords + 2 shifts; decode 2 `v_alignbit` + 6 shifts + 4 fast pair decodes (tiles-core
hash + sad_u8). Same VALU as K2 within a few ops, 4 VMEM per tile instead of 2. U table
`exl3_tiles_u_half_splitk` (not swept: 194-198 GB/s, at the roof). The half pipe
matches the K2 pipe on the same prefill shapes.

**Numerics.** Tiles core == direct core and row r == the m = 1 call, bit-identical
(`half_gemv_check.py`); half pipe == old half mainloop on the same grid. MiMo PPL
+0.064% with both switches on, exact with both off — the prefill MoE grid (stream-K
split), not the decoder. MTP acceptance 64% -> 73% (verify rows take plain arithmetic).
Not covered: half-rate lm_head-scale dense outputs and `EXL3_GEMV_MR_M1=0` take the
cooperative GEMM at m = 1; mixed half/integer MoE layers keep the old K = 0 mainloop and
mgemm decode route (no model needs either).

## Model bring-up notes

- **GLM-5.3-Flash** (KDA + MLA/DSA, QK / V head 256): failed on 64 KB LDS —
  `_mla_unfold_kernel` needs 69-82 KB and `MLAAttention.autosplit_prepare` does not
  catch `BCKernelTooLarge` (the forward path does); `mla_attn_triton_prefill_mha`'s
  q-tile loop cannot help when the 64 x 256 K and V tiles alone overflow at 2 stages.
  `EXL3_ROCM_MLA_LDS_FIT`: BC unfold retries at BLOCK_K 64 (memoised), eager
  `mla_unfold` picks BLOCK_K via `smem.pick_config`, MHA prefill walks (block_n 32,
  stages 2) -> (32, 1) -> (16, 1). Upstream bug on any 64 KB part. Lead:
  `_mla_decode_split_kernel` (2420 B scratch) and `_mla_absorb_kernel` (816 B) are the
  "DS4 per-layer stall" pattern.
- **MiMo-V2.6-Flash** (GQA 64/4, QK 192 / V 128, SWA 128): runs out of the box; its
  half-integer experts are covered above.
- **Laguna + DFlash drafter**: the unquantized BF16 drafter's m == 1 linears hit the
  rocBLAS skinny-GEMM pathology beyond N = 256 ("Narrow-N hgemm"), so drafting is a net
  loss; a plain fp16 GEMV for any N at m <= 8 would fix it.
