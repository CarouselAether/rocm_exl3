# RDNA notes

Hardware and toolchain facts this port depends on, and why each RDNA sibling
differs from the upstream file it replaces. Everything here was measured on
gfx1151 (Strix Halo, RDNA 3.5, wave32) under ROCm 7.2.4 — not inferred from
documentation. Each item has a plausible-looking wrong answer, which is why it is
written down.

## WMMA

All four WMMA variants exist on gfx1151 and all take operand order **`(B, A, C)`**:
`f32_16x16x16_f16`, `f16_16x16x16_f16`, `i32_16x16x16_iu8`, `f32_16x16x16_bf16`.
Wrong operand order produces a transposed result, not a crash.

They share one fragment layout:

| fragment | mapping |
|---|---|
| A | lane holds row `L % 16`, all 16 columns |
| B | lane holds column `L % 16`, all 16 rows |
| C | `row = L % 16`, `col_base = (L >= 16) ? 1 : 0`, element `i` at column `i*2 + col_base` |

- bf16 accumulates to **fp32 with the same C layout**, so it reuses `WmmaFragC`
  and every existing store/accumulate helper.
- fp16-accumulate packs into every other half-slot selected by `opsel` (a
  template/immediate, not runtime) and preserves the other half, so two
  independent accumulators fit in one fragment.
- Single-accumulator fp16 saves no registers — gfx11's fp16 C fragment is 8
  VGPRs, the same as fp32 — so fp32 accumulate is strictly better unless the
  `opsel` packing is used.
- **int8 sign-flag trap:** the builtin is `(s0, v0, s1, v1, C, clamp)` and each
  flag pairs with the vector that *follows* it, so under `(B, A, C)` ordering the
  first flag describes **B**, not A. `mma_sync_i8<signed_a, signed_b>` in
  `rdna_wmma.hip.h` hides this.

HIP's documentation covers MFMA (CDNA), not WMMA (RDNA); the RDNA ISA PDF is the
only authority. `rocm_tools/wmma_check.hip` compiles the shipped header and
checks it against a CPU reference with non-symmetric inputs.

## Hardware

- **int8 dot product is `__builtin_amdgcn_sudot4`, never `sdot4`.** gfx1151 lacks
  `dot1-insts`, so `sdot4` fails to compile and reads like "no int8 support". It
  has `dot8-insts`. `udot4` and `fdot2` are available; `udot2` is not.
- **`hipDeviceProp_t` reports `major = 11, minor = 5`**, so upstream's
  `prop.major >= 10` Blackwell test (`exl3_devctx.cu:39`) classifies RDNA as
  Blackwell. `exl3_kernel_map_rdna.hip` ignores `cc` entirely for that reason.
- **LDS is 64 KB per workgroup** (`sharedMemPerBlock`), against the ~90–100 KB
  that CUDA shape tables assume. The dynamic LDS base is 32-byte aligned, and
  misaligned vector loads split rather than fault.
- **`multi_processor_count` reports WGPs, not CUs** — 20 on a 40-CU part.
- **`__funnelshift_r` is native**, exactly matches PTX `shf.r.wrap.b32` (shift
  masked `& 31`), and lowers to one `v_alignbit_b32`. A hand-rolled uint64
  version with `& 63` is wrong for every shift >= 32.
- **Prefill is slightly compute-bound and lands near the achievable roofline.**
  ~9.8 TFLOP/s at the model level (Gemma-4-31B, 158 t/s at 512 tokens) against a
  peak fp16 GEMM of 24-32 TFLOP/s and a practical envelope of 30-37.
- **Achievable memory bandwidth is ~206 GB/s** (`rocm_tools/bench_membw.py`),
  ~80% of the 256 GB/s theoretical, and *flat* from 64 MiB to 16 GiB. There is no
  VRAM-vs-GTT cliff: on this unified-memory part the 512 MiB "VRAM" aperture
  rocm-smi reports is a legacy carveout, not a constraint, and torch's
  `total_memory` is the GTT pool. bench_membw's figure is not the ceiling for
  pure streamed reads, though: the DRAM-resident GEMV sweep (`GEMV_SWEEP=1
  gemv_check`, four rotating B buffers) sustains **226 GB/s**, which is the
  right roofline for weight-streaming kernels — it puts Gemma-4-31B's decode
  ceiling at ~9.7 t/s, not the ~8.8 the 206 figure implies.
- **There is a 32 MiB Infinity Cache, and it will flatter any kernel benchmark
  whose working set fits.** Measured 653 GB/s at 16 MiB against 213 GB/s at
  64 MiB. Timing one weight tensor in a repeat loop measures cache, not DRAM --
  it overstated the EXL3 GEMV rate by 26% here, and a first pass at this
  concluded "the kernels are healthy" from exactly that error. Cycle a working
  set several times cache size.
- **Token generation was NOT memory-bound as first shipped — it was bound by
  running a 16-row tile GEMM for one useful row.** (Historical: this is the
  measurement that motivated the GEMV-path work; the exclusions below are
  closed as of 2026-08-08 — see "Decode lost the GEMV path" and "The
  barrier-free dot-tile core". Kept because the profile method and the failure
  shape are the reference for the next regression.) Measured on Gemma-4-31B,
  decode at bsz=1:
  `rocm_tools/profile_decode.py` reports the GPU 97.4% busy, with **97.6% of all
  GPU time in `exl3_gemm`/`exl3_mgemm`** -- attention is 0.8%, norms 0.1%. The
  effective rate is 27.0 GB of weights in 477 ms/token = **~57 GB/s, 27% of the
  206 GB/s the hardware delivers**. The GEMV path is 2.4x faster per call
  (0.392 ms vs 0.90-0.97 ms) and reaches 122 GB/s DRAM-resident, but it is
  excluded from ~78% of decode by two separate things:
  - `exl3_mgemm` has **no GEMV path at all** (`exl3_gemv_try_launch` is called
    only from `exl3_gemm`), so every fused q/k/v and gate/up runs the tile GEMM.
    That alone is ~50% of decode GPU time.
  - this port's own `EXL3_RDNA_GEMV_GRAPH` guard declines GEMV whenever a graph
    is capturing, and decode is ~100% captured (120 `hipGraphLaunch` per token,
    2 per layer). Disabling BC-attn graphs moves `exl3_gemv_dot_kernel` from 31
    calls to 4061 and gains 8% decode throughput -- so the guard's comment,
    "costs coverage during capture and nothing else", is wrong in the one case
    that matters.

  The GEMM path measures ~53 GB/s at bsz 1, 4 *and* 16, i.e. flat, which is the
  signature of the 16-row tile: the work is the same whether the rows are useful.
  Closing both gaps is worth roughly 2x decode on paper (122 vs 53 GB/s).
- **Benchmark noise floor is ~4.7% spread** (1.6% stdev), and the first run reads
  high. No perf claim under ~5% survives a single measurement.

## Toolchain

- **`-fgpu-rdc` hides codegen failures.** It defers device codegen to link time,
  so a translation unit containing inline PTX that merely *parses* on amdgcn —
  anything using only `"r"` constraints — produces an object file and fails only
  at link. `ptx.cuh` fails at parse time instead, on the `f`/`l` constraint
  letters, which is why those failures were always visible and these were not.
  `rocm_tools/hipcc_probe.sh` compiles without rdc and retries with it only when
  "undefined symbol" is the sole error class.
- **Inline PTX lives in three files**, not one: `ptx.cuh` (19 blocks),
  `quant/codebook.cuh` (10), `quant/exl3_gemv_int8_kernel.cuh` (1). Searching for
  it needs `grep -rn "asm[[:space:]]*("` — the dp4a in `exl3_gemv_int8_kernel.cuh`
  is written `asm ("dp4a...` with a space, and a tighter pattern misses it.
- **Several ROCm 7.1-era workarounds are obsolete on 7.2.4.** `__shfl_*_sync`
  exists and is default-on, `__funnelshift_r` is native, hipBLAS hgemm works.
  The hardware findings above are unaffected.
- **HIP lacks `__dp4a`, `__ldcs`, `__ldcg`**, all bridged in `hip_compat.hip.h`.
  `__ldcg` is not a performance hint: it bypasses L1 for cross-block visibility,
  so it maps to an agent-scope atomic load rather than a plain load.

## Why the siblings differ

### `hip_compat.hip.h` — warp-sync primitives

`__syncwarp` maps to a wavefront-scope release fence, `wave_barrier()`, and an
acquire fence. `__builtin_amdgcn_wave_barrier()` alone is a *scheduling* barrier
and emits no `s_waitcnt`, which silently drops the shared-memory ordering half of
CUDA's `__syncwarp` contract. That breaks any cross-lane exchange through LDS
where the write and read use different addresses, because the compiler has no
dependency to wait on — `routing.cu`'s radix sorts and
`hadamard_inner.cuh`'s `had_hf_r_128_d_inner` both do exactly that.

`__ballot_sync` and `__activemask` cast to `unsigned`: HIP's `__ballot` returns
64-bit, and the uncast form selects the wrong `__ffs`/`__popc` overload at any
call site that does not launder it through an `unsigned int` first.

### `exl3_gemm_inner_rdna.hip.h` — LDS layout and split-K

- **B dequant staging stride is 18 halves, not 17.** 17 puts adjacent
  active-lane groups (0–3 against 16–19, 8–11 against 24–27) on the same LDS
  banks, measured at ~24% stall time by PMC. 18 halves is 9 dwords, coprime with
  32. The value lives in `EXL3_GEMM_SH_B_DQ_STRIDE` and feeds all consumers from
  one place — it was previously duplicated, and the host-side launch accounting
  drifted to 17 while the kernel indexed at 18, under-allocating dynamic LDS by
  256 bytes on the *shipped* path while the standalone harness passed.
- **`TILESIZE_N = 192` is invalid** and is rejected by a `static_assert`. It
  fails `N % 128 == 0`, and `FRAGS_N_PER_WARP = TILEBLOCKS_N / NUM_WARPS` is
  integer division, so 12/8 = 1 silently drops a third of the output tile.
- **`sh_b_dq` is sized `NUM_WARPS * TILEBLOCKS_K` and indexed by the block-wide
  warp id.** `NUM_WARPS` counts warps per `sub_k` group, but `blockDim` is
  `EXL3_GEMM_BASE_THREADS * TILEBLOCKS_K`, so at `TILEBLOCKS_K == 2` the block
  holds 16 warps while `warp_id = t / 32` only spans 8. Sizing and indexing for 8
  made the `sub_k` 0 and `sub_k` 1 warps sharing a `warp_id` stage different B
  fragments into the same buffer with only a `__syncwarp` between them.
- **`threadblock_reduce()` indexes `sh_c` by `t` on both sides** of the exchange.
  The exchange is serialised by its `__syncthreads()` pair, so one slot per
  thread suffices. An earlier version wrote at `t` and read at
  `t + src * EXL3_GEMM_BASE_THREADS`, summing a region nothing had written and
  running past the end of the LDS block, since `sh_c` is its last allocation.

The last two were invisible for a long time because every shape in the RDNA shape
table uses `TILESIZE_K = 16`, which makes `TILEBLOCKS_K == 1` and compiles the
whole split-K path away. The MoE kernel is its only live caller.

### `exl3_moe_shape_rdna.hip.h` — MoE tile-K

Exists because upstream's `MOE_TILESIZE_K` is a bare `#define`, so `-D` cannot
override it. The value is upstream's 32; `EXL3_RDNA_MOE_TILESIZE_K=16` forces the
single-K path — the geometry the RDNA shape table is validated on — at a cost of
1.42–1.56× MoE throughput (1836 vs 2864 ms at 256 tokens, 3699 vs 5238 at 1024).
The header must be included after `exl3_moe_common.cuh` by both the kernel
sibling and the host sibling: the host derives `blockDim` from the constant, so a
mismatch is a silently wrong launch rather than a compile error.

### `exl3_kernel_map_rdna.hip` — shapes and LDS budget

RDNA shapes all use `TILESIZE_K = 16` and 256-thread blocks to fit the 64 KB LDS
budget. `EXL3_RDNA_SMEM` requests each shape's actual LDS requirement rather than
upstream's `SMEM_MAX`; requesting `SMEM_MAX` on a 64 KB part reserves the whole
workgroup allocation and pins residency at one block per WGP regardless of tile
size, and would make `exl3_mgemm`'s multi-z grids illegal.

Cooperative launch itself works on gfx1151 (`cooperativeLaunch = 1`). Measured
across 13 cases — shapes 1/2/3, bits 2/4/8, cb 0/1/2, fp16 and fp32 out,
m = 16/32/48, grids 1/10/20 — cooperative results agree exactly with the same
work done without cooperative machinery. An oversubscribed grid is refused by the
runtime, not hung.

### `rope_rdna.hip` — fused RMS norm

Upstream's `apply_norm` / `apply_norm_uw` compute a warp total with
`warp_reduce_sum_f`, a `__shfl_down` reduction that leaves the result in lane 0
only, and then have all 32 lanes store it to the same `sums[]` slot. Which lane
wins is vendor-dependent. Measured with distinct lane values 0..31 (true sum 496):

```
__shfl_down : lane0=496  lane1=512  lane16=752  lane31=992
__shfl_xor  : every lane 496
unguarded store lands 992      <- lane 31 wins on RDNA, lane 0 on NVIDIA
```

Upstream is therefore correct on NVIDIA by accident. On RDNA the slot receives a
partial sum, making the RMS scale wrong by a constant factor per head and
silently corrupting QK-norm. `tests/test_rope.py` went from 30 failed / 30 passed
to 60 passed. The sibling guards the store to lane 0 (`EXL3_RDNA_NORM_LANE0`) at
both call sites.

`norm.cu` already handles this correctly with `__shfl_xor` plus an
`if (lane_id == 0)`, and `rope.cu` carries the `int lane_id` line commented out.
This is a latent upstream bug rather than a ROCm-specific one.

A first probe of this reported the race as benign and was wrong: it used uniform
lane values (all 1.0), where clamp-to-self and wrap are indistinguishable because
`v += v` doubles to 32 either way. Reduction probes need distinct per-lane values.

## Why mgemm is capped at ~1/3 roofline at m == 1

Measured with rocprofv3 on `rocm_tools/bench_mgemm.py` (Laguna, 10 experts,
3072->1024, K=4). The kernel is **occupancy-starved by the cooperative launch**,
and it is not register pressure, not dequant, and not bandwidth:

| shape | VGPR | duration | OccupancyPercent | MemUnitBusy |
|---|---|---|---|---|
| 1 (N=128) | 144 | 481.5 us | 12.16% | 46.40% |
| 4 (N=512, selected) | 248 | 251.2 us | 11.88% | 25.89% |
| 3 (N=384) | 256 | ~~211.8 us~~ INVALID | 11.66% | 32.05% |

**Shape 3's timings in these tables are truncation artifacts, discovered after
they were first written up.** The inner kernel floors `size_n / TILESIZE_N`
with no remainder pass, and 1024 % 384 != 0, so a forced shape 3 computed 768
of 1024 columns per expert -- 75% of the work in 84% of the time, i.e. *slower*
per useful byte than shape 4. The occupancy/MemUnitBusy counters are still
valid (the kernel that ran, ran at ~12%); the duration and any GB/s derived
from it are not comparable. Forced shapes that do not divide the problem are
now rejected outright (`select_exl3_*gemm_kernel`), the selector's
compatibility check is per-matrix (it was computed on the bszm-scaled width,
which admits tiles that truncate every matrix, e.g. N=1024 x 3 experts for the
384 tile), and `bench_mgemm.py` NaN-fills C and verifies coverage before
timing anything.

Occupancy is ~12% at VGPR 144, 248 **and** 256, so registers are not the cap.
`Grid_Size` is 5120 *threads* = 20 workgroups of 256, which is exactly
`get_num_sms()` -- `multiProcessorCount`, reporting **WGPs (20), not CUs (40)**.
The grid computes to `(2, 1, 10)` for 10 experts at `exl3_gemm_rdna.hip`:

    num_sms = tiles;
    if (num_sms * bszm > total_sms) num_sms = MAX(total_sms / bszm, 1);
    concurrency = MIN(total_sms / num_sms, bszm);

The occupancy figure is fully accounted for by that grid. gfx1151 is 20 WGPs ->
40 CUs (2 per WGP) -> **80 SIMD32** (2 per CU), and RDNA3 allows 16 wave32 per
SIMD. So 20 workgroups x 8 waves = 160 waves over 80 SIMDs = **2 waves per SIMD**,
and 2/16 = **12.5%** against a measured 11.66-12.16%. Nothing is left over for
another explanation, which is what rules out register pressure and LDS: they
would have to show up as a *shortfall* against this number, and there is none.

Two waves per SIMD is far too few to keep loads in flight, which is why
`MemUnitBusy` sits at 26-32% and the achieved rate is 55-66 GB/s against the
206 GB/s roofline.

**`force_num_sms` in the mgemm path** used to be inert -- `num_sms = tiles`
overwrote it unconditionally, so sweeping it 20/40/80/160/320 changed nothing
and looked like evidence the grid size did not matter. It is honoured now
(exactly as given; an oversubscribed value gets the runtime's cooperative-launch
refusal, which is the informative outcome a sweep wants). Measured after the
fix: the default sizing (grid 4x5 for 10 experts) beats every forced value
tried (0: 282.6 us; 5: 341.8; 10: 315.7; 20: 407.2), consistent with the
co-residency ceiling being the binding constraint.

**The grid cannot simply be widened.** `EXL3_RDNA_SMS_MULT` (added for this
experiment, default 1 = no change) multiplies `total_sms`. At 2 the runtime
refuses shapes 3 and 4 outright -- "too many blocks in cooperative launch" --
because `grid.sync()` requires every block co-resident and these shapes' LDS and
VGPR use allows only one workgroup per WGP. The lighter shapes do accept it:

| shape | MULT=1 | MULT=2 | note |
|---|---|---|---|
| 1 (N=128) | 29.1 GB/s | 45.2 GB/s | 1.55x |
| 2 (N=256) | 44.7 GB/s | 61.7 GB/s | 1.38x |
| 3 (N=384) | ~~66.2 GB/s~~ INVALID | refused | truncation artifact, see above |
| 4 (N=512) | 55.5 GB/s | refused | what the selector picks -- correctly |

So 20 blocks is the genuine co-residency limit, not a WGP-vs-CU miscount.

**One conclusion, not two.** The write-up originally drew a second, cheap
conclusion here -- "the selector picks N=512 where N=384 is 19% faster, worth
~8% of decode" -- which is dead: the shape-3 numbers were truncation artifacts
(see above), and among the shapes that actually compute the full output the
selector's pick was the fastest all along (55.7 vs 44.1 vs 30.0 GB/s for
shapes 4/2/1, re-measured under the coverage check). What survives is the
structural conclusion, now with nothing left to soften it: **the cooperative
GEMM cannot exceed ~1/3 of roofline at m == 1 on this part, because it cannot
oversubscribe, and there is no tuning inside it worth having.** The RDNA GEMV
is a plain launch and reaches 203 GB/s (the roofline) on lm_head, so the fix is
the non-cooperative multi-matrix GEMV for m == 1 -- implemented as
`quant/exl3_mgemv_rdna.hip`, see its header comment for the design (plain-
launch pipeline of four kernels, expert axis on grid.y, graph patching through
a prologue-published device parameter block, cooperative-identical packing and
reduction semantics).

## The barrier-free dot-tile core (2026-08-08, second session)

Two findings from profiling dense Gemma-4-31B decode after the split-K session:

- **Decode runs ZERO cooperative kernels on a dense model.** Every coop call in
  a 24-token profile belonged to the prompt's prefill. The previous handoff's
  theory — that Gemma sat flat at 4.7 t/s because fused q/k/v with per-matrix
  width lists (`size_n_list`/`c_ptrs`) declines to the cooperative kernel — was
  wrong: Gemma's fused qg/kv/gate-up mgemm calls pass **no** lists (the lists
  form is used only by DS4's `bc_dsa.py` fan/fan2 sites). Gemma was already
  fully on the GEMV paths.
- What actually capped it: **the LDS dot-tile core ran 129–148 GB/s** on
  Gemma's mid shapes while the identical core hit 207 GB/s on the lm_head
  shape. The per-k-tile round trip (stage quantized → dq → `__shfl_down`
  unswizzle → LDS scatter → `__syncwarp` → LDS gather → dot, lanes 16-31 idle
  in the dot) was the cost.

The fix is `exl3_gemv_dot_tile_direct` (exl3_gemv_kernel_rdna.hip.h): keep the
accumulation in dq's native fragment layout — lane L holds rows
`(L%4)*2+{0,1,8,9}` of columns `(L/8)*2+((L>>2)&1)` and `+8` — so each k-tile
is 4 `v_dot2_f32_f16` per lane with B read straight from global, no LDS, no
barriers, no idle lanes. One 2-hop `__shfl_xor` quad reduction and a broadcast
remap at the END of the k-range restore the "lane l returns column l" contract,
so all six kernel wrappers (single/split-K x plain/graph/mgemv) take either
core. Runtime selection via the kernels' trailing `lds_core` argument
(`EXL3_GEMV_LDS=1` pins the old core; default is the direct core); smem is
passed identically in both modes — LDS was never the occupancy limiter, and
keeping the carve fixed leaves the graph patch sites untouched.

Validated: `gemv_check.hip` (now runs every case on both cores) — all bits,
codebooks, wave counts, dtypes pass, direct core slightly tighter;
`mgemv_check.py` on Laguna — all routing configs pass; fp32 ground-truth
parity on real Gemma weights — both cores rms 6e-4 from reference. Measured on
real Gemma weights (m=1, whole dispatch): 5376→8192 **1.49x** (238 GB/s
effective), 8192→5376 1.45x (228), 21504→5376 1.41x (177), 5376→21504 1.12x
(158), lm_head 1.18x (225). Remaining headroom in the core: VOPD dual-issue
interleaving (ISA doc in `exlproject/rocm_docs`); the wide gate/up shape
(158 GB/s) suggests re-sweeping warps/block and the split-K threshold with the
new core. Software-pipelining the B loads was tried and measured a strict
loss — see "Software-pipelining the direct core: tried, rejected" below.

## Decode state after the GEMV-path work (2026-08-08)

End-to-end decode, `rocm_tools/bench_model.py`, median of 3, this machine:

| model | bpw | tg t/s | pre-GEMV-work | note |
|---|---|---|---|---|
| Gemma-4-31B (dense) | 6.00 | **7.9** | 4.7 | ~81% of the 9.7 t/s roofline; 7.8 → 7.9 from the split-K wave selector (2026-08-13) |
| Laguna-S-2.1 (MoE 256e top-10) | 4.03 | **20.8** | 15.7 | llama.cpp does 22–25 on this box |
| DeepSeek-V4-Flash | 2.07 | **15.6** | 8.8 | 13.7 → 15.6 when the width-list sites moved to mgemv (2026-08-13) |

Split-K is capped at `EXL3_GEMV_SPLITK_MAX_TILES = 2048`
(`exl3_gemv_kernel_rdna.hip.h`), raised from 512 after a sweep with the direct
core: split-K still wins +29% at 1344 tiles and reaches parity at 16384. The
old 512 was tuned for the LDS core.

Coherence after the direct core: user-verified via `examples/chat.py` on all
three models above, 2026-08-08. (chat.py only — bare `tokenizer.encode` drops
BOS and fakes corruption.)

Kill switches, each re-read per call: `EXL3_GEMV_LDS=1` (pin the old LDS dot
core), `EXL3_MGEMV=0`, `EXL3_GEMV_GRAPH=0`, `EXL3_GEMV_SPLITK=0`, `EXL3_GEMV=0`,
`EXL3_GEMV_SPLITK_WARPS=4|8|16` (force one split-K wave count everywhere,
overriding the shape-aware selector).
Graph-captured kernels bake the switches at capture time. Related trap: a
`hipMalloc` during stream capture invalidates the graph — allocate (prewarm)
parameter blocks before capture begins.

Open items, split by scope. General items are code-path and algorithmic work
that would carry to any GPU running this port; Strix Halo items are tuning or
ISA use whose value is established only for this part.

General:

1. **mgemv split-K underperforms its single-matrix form**: the fused gate/up
   shape captured only ~10% of the 22% the single-matrix split-K gained.
   Unexplained, and likely structural (per-matrix z-slices) rather than a
   gfx1151 quirk.

**RETRACTED (2026-08-13, same day it was filed): "`exl3_moe_kernel` is ~30%
of DS4 decode in ~one call per token."** Python-level instrumentation of
`ext.exl3_moe` (wrap the binding, record module key/phase/shapes per call)
shows decode NEVER calls it: at bsz == 1 every MoE layer is `bszn_eligible`
and takes `run_bszN` → mgemv. The 42 calls in the profile were the 64-token
prompt's PREFILL — one fused call per MoE layer — inside the profiler window,
because `profile_decode.py` wrapped the whole Job and a Job runs its prompt's
prefill in the same iterate() loop as decode. This is the THIRD wrong
localization produced by kernels-in-window ≠ decode-kernels (the cooperative
"decode" calls that were prefill; the width-list theory built on them; now
this). profile_decode.py now uses a 1-token prompt for the profiled job, so
the window contains decode-shaped work only. (Starting the Kineto session
mid-generation instead captures zero device events on ROCm — that approach
does not work.) The ~23 ms/layer fused-MoE prefill call itself is a
plausible PREFILL lever (at 64 rows it streams essentially all 256 experts'
weights), but DS4 prefill is 103–162 t/s and healthy; low priority.

DS4 decode after the width-list work is dominated by the mgemv split-K dots
and DSA attention — there is no hidden MoE cost.

Width-list support in mgemv — the former item here — landed 2026-08-13:
`size_n_list`/`c_ptrs` calls (DS4's `bc_dsa` fan/fan2 sites, the only users)
now take the plain-launch mgemv instead of falling to the cooperative kernel.
Per-matrix width gates the tile grid, `c_list[mat_index]` replaces the
`j * size_n` output stride, and both lists are device arrays read per launch
(the `B_list` indirection), so graph capture needed no new patch sites. The
cooperative kernel disappeared from the DS4 decode profile (was 254 ms /
7.8% / 1333 calls; the same work now adds ~49 ms on the mgemv split-K rows —
~5x per call), decode 13.7 → **15.6 t/s** (+14%, spread 0.2%). The extra gain
over the 7.8% share is the cooperative launch overhead going with it.
Validated: greedy A/B vs `EXL3_MGEMV=0` produces equivalent coherent text
(fp16-noise wording drift only), `mgemv_check.py` all-PASS,
`test_dsa_kernels.py` ALL PASS, Laguna 20.8 / Gemma 7.7 unchanged.

Strix Halo (gfx1151) specific: none open.

**Launch-geometry re-sweep: done (2026-08-13).** The fixed split-K wave count
(8, tuned with the LDS core) is now `exl3_gemv_splitk_warps(k_tiles, n_tiles,
bszm)` — 4/8/16 chosen per shape, shared by all three split-K sites, with the
fit and the sweep data recorded at the function (exl3_gemv_rdna.hip). The
rule's drivers: short k (≤128 k-tiles) and saturated grids (≥1024 blocks,
where blocks = n_tiles × bszm — the mgemv grid multiplies by expert count)
prefer 4; starved grids (<128 blocks) and long k (≥1024 k-tiles) prefer 16.
`EXL3_GEMV_SPLITK_WARPS` forces one count everywhere (model-level A/B);
gemv_check now covers split-K correctness at all three counts, both cores.

Honest end-to-end outcome: per-shape kernel gains up to +8% (starved grids)
and +5-6% (short k) in the DRAM-resident sweep, but Gemma is the only model
that moves — 7.8 → **7.9** t/s (its 21504→5376 down at W=16, gate/up at W=4).
DS4 (15.6) and Laguna (20.7 vs 20.6 pinned-8) are flat: their dominant mgemv
expert shapes sit with bszm-multiplied grids in regions where the old 8 was
already right or the delta is diluted below the noise floor. The selector
ships because it is never worse, fixes the single-matrix starved-grid cases,
and the env override is the sweep tool the next core change will want.

### VOPD: checked, closed (2026-08-13)

The former open item — hand-interleave the direct core for VOPD dual-issue —
is closed on ISA-level evidence, no implementation needed:

- **The compiler already emits VOPD where it is legal.** The probe TU shows
  56 `v_dual_*` instructions, including `v_dual_dot2acc_f32_f16` inside the
  bits=2 hot loop. (`-Rpass-missed=gcn-vopd` reports nothing.)
- **The op mix caps what is left.** The bits=2/cb=2 (DS4) inner-loop
  histogram: 8 `v_mul_lo_u32`, 8 `v_dot4_u32_u8`, 7 `v_bfe_u32`,
  4 `v_pk_fma_f16` dominate — all VOP3/VOP3P-class, ineligible for VOPD
  pairing by ISA restriction (§7.6: VOPD pairs a restricted op list, wave32
  only, VGPR-bank port limits). The pairable remainder (a few `v_and_b32`,
  shifts, moves, the dot2accs) is single-digit percent of the loop's VALU,
  and the compiler is already pairing within it.

Hand-rolled VOPD asm would fight the compiler's `s_delay_alu` scheduling to
chase <10% of VALU on shapes that are only partially VALU-bound. If low-bpw
decode ALU ever needs to shrink, the lever is reducing the op count of the
3INST decode itself, not dual-issuing the current ops. (RDNA2 note for the
record: VOPD does not exist pre-RDNA3, and this port does not target RDNA2 —
its prefill would be blocked on WMMA absence anyway.)

### Software-pipelining the direct core: tried, rejected (2026-08-13)

The former open item — overlap tile t+1's B loads with tile t's dq/dot —
was implemented and measured, and the code was reverted. Record of both
halves, because each kills a different future re-attempt:

**The premise was half right.** The compiler does NOT pipeline the plain
loop: the ISA (probe TU over `exl3_gemv_dot_kernel`, bits 4 and 6) issues all
of an iteration's loads at the top, staggers `s_waitcnt vmcnt(2/1/0)` through
the dequant, and issues the next iteration's loads only after the last
`v_dot2acc`. Full global-load latency is exposed every k-tile, per wave.

**The conclusion drawn from that was still wrong.** A depth-2 register
pipeline (dq split into `dq_load_dispatch`/`dq_decode_dispatch` halves,
prologue load, rotate `cur = nxt`, epilogue) compiled to the intended
schedule — next tile's loads interleaved between the current tile's dots,
waits rotated to the loop top — passed all 40 gemv_check cases on both cores,
cost only +3-4 VGPRs, and was **slower everywhere**. Same-session A/B against
a HEAD-built binary, DRAM-resident sweep: no shape improved; large shapes
-1% typical; short-k split-K shapes (Laguna expert 3072→1024, 1024→3072)
**-6 to -8%** consistent across wave counts; untouched LDS-core control rows
flat ±0.5%, so the rig was sound.

Why: these kernels are plain launches at high occupancy — when one wave sits
in `vmcnt`, the SIMD issues another wave. Wave-level parallelism was already
covering the load latency (lm_head runs 225 GB/s against the 226 measured
roofline — there was nothing left to unlock), so intra-wave pipelining
contributed only its overhead: the register rotate and the duplicated
address math, proportionally worst where split-K makes per-warp k-ranges
short. Intra-wave latency hiding is for kernels that CANNOT oversubscribe —
the cooperative GEMM was such a kernel; the GEMV path is not.

The corollary for the remaining gaps: mid shapes at 180–205 GB/s are not
latency-limited (pipelining would have moved them), which points the
remaining headroom at launch geometry (open item 4) and decode ALU
throughput (open item 3), not at the memory pipeline.

Validation discipline for any change here: `gemv_check.hip` runs every case on
both cores against an independent reconstruct reference; fp32 ground truth for
real weights is `A @ LinearEXL3.get_weight_tensor()` (it folds suh/svh and the
Hadamards). Max-relative-error with a small denominator clamp false-flags
near-zero outputs — accumulation-order noise reads as mismatch; use
gemv_check's gates (`d > 0.01*denom + 0.05`, RMS ratio as primary). And
profile before implementing: `rocm_tools/profile_decode.py`.

## Profiling on this machine

rocprofv3 works, with three constraints found the hard way:

- **At most THREE counters per pass.** A fourth returns "Request exceeds the
  capabilities of the hardware to collect". `OccupancyPercent MemUnitBusy
  FETCH_SIZE` fits and is the useful triple.
- **PyTorch processes need torch's bundled rocprofiler libs moved aside.** The
  wheel ships `librocprofiler-sdk.so` and `librocprofiler-register.so` with
  `RPATH=$ORIGIN` at different versions from the system copies, so two instances
  load and registration fails with "Configuration request occurred outside of
  valid rocprofiler configuration period". `LIBKINETO_NOROCTRACER`,
  `LIBKINETO_NOCUPTI`, `LD_PRELOAD` (direct and appended via wrapper) and
  `LD_LIBRARY_PATH` all fail; RPATH beats them. Renaming the two files in
  `torch/lib` works and torch falls through to the system copies. Restore them
  afterwards. rocprofv2 is not an option -- it does not support Strix Halo.
- **Counter collection deadlocks cooperative kernels.** PMC serialises dispatches,
  which breaks the co-residency `grid.sync()` depends on. `gemm_coop_check` hangs
  with the GPU at 2% and no output. Graph-captured kernels fail differently, with
  "Timeout while waiting for queue sync: N kernels still active". This is why
  `bench_mgemm.py` exists: it reaches mgemm from Python, outside any graph, with a
  plain launch.
- A tool that calls `os._exit()` produces **no CSV** -- rocprofv3 writes from exit
  hooks. Return normally; the teardown segfault happens after the flush.
- **Kineto device-activity capture can wedge machine-wide.** Observed
  2026-08-13: torch.profiler CUDA activity returned zero device events in
  every fresh process (even a bare matmul) after a HIP process died with
  SIGSEGV mid-run earlier in the session, where identical profiles worked
  hours before. No stray processes or /dev/shm state to clean; driver/
  tracer-side, and a reboot clears it (verified 2026-08-13). If a profile
  shows 0.000 s GPU time with a populated host-op table, test capture with a
  trivial matmul before trusting any "HOST-BOUND" verdict.

## Decode lost the GEMV path — how, and what it takes to get it back

**Status: resolved as of 2026-08-08.** All three items under "What the fix
requires" landed — the plain-launch mgemv and the graph-GEMV parameter
contract (`Plain-launch GEMV paths for m == 1`, ae1855d), in-block split-K
(e639593), and the barrier-free core plus the 2048-tile split-K cap (f69507b,
5db0ab7). End-to-end results are in "Decode state after the GEMV-path work"
below. The section is kept as written because it documents *how* the path was
lost — both halves were comments that were true when written and invalidated
by changes elsewhere, a failure shape this port has now hit three times.

The legacy 0.0.29 fork routed essentially all of decode through the RDNA GEMV,
leaving GEMM for prefill and weight loading. That is the correct division and it
is no longer what happens: measured on v1.4.1, **~75-78% of decode GPU time runs
`exl3_gemm`/`exl3_mgemm` 16-row tile kernels for one useful row**, at ~20-27% of
achievable bandwidth. Neither half was lost to a deliberate change.

**Half 1 — the `!graph` guard stopped being cheap.** `exl3_gemm_rdna.hip` declines
GEMV while a graph is capturing (`EXL3_RDNA_GEMV_GRAPH`), and the legacy fork had
the identical guard at the same site (`exl3_gemm.cu:110`, `size_m == 1 && !graph`).
In 0.0.29 that cost almost nothing, because almost nothing was captured — there
was no `bc_attn.py` and no MoE `bszN` graph path. Upstream has since added both,
so decode is now ~100% captured (120 `hipGraphLaunch` per token, 2 per layer) and
the guard declines GEMV for *everything*. Toggling `EXL3_BC_ATTN=0` moves
`exl3_gemv_dot_kernel` from 31 calls to 4061 and gains 8% on dense Gemma. The
guard's comment claimed it "costs coverage during capture and nothing else" —
true when written, false now. Same pattern as the split-K defects: a comment
asserting a path is harmless, invalidated by a change elsewhere.

**Half 2 — retiring the mgemm guard closed the other door.** The fork disabled
MultiLinear/mgemm outright, so fused q/k/v and gate/up ran as separate
`exl3_gemm` calls at m == 1 and took GEMV. That guard was retired 2026-08-07 for
good reasons (it was producing degenerate output), but `exl3_mgemm` has **no GEMV
path at all** — `exl3_gemv_try_launch` is called only from `exl3_gemm`. Nothing
declines; nothing asks. On MoE at bsz=1 this is the dominant cost: `bszn_eligible`
routes bsz <= `MAX_BSZN` (8) through mgemm, so Laguna decode spends 42-48% there
across ~139 calls/token, and `exl3_moe_kernel` is called about once per token.

**Packing rows is not the alternative.** The tile's M dimension is rows sharing
one weight matrix; routed experts each need a different B, so they cannot be
packed into the 16 rows. mgemm already gives each expert its own z-slice with one
useful row of sixteen. Packing only pays where rows share weights — concurrent
sequences, or speculative decode. For single-user decode GEMV is the answer. (The
user tried row packing early in the project; it did not work, for this reason.)

### What the fix requires

In dependency order — 1 must land first or 2 and 3 measure as no gains:

1. **The fp32-output GEMV is slower than the tile GEMM it would replace.**
   Measured on Laguna: `exl3_gemv_dot_kernel<4,true>` 911 ms over 2256 calls
   against `exl3_gemm_kernel<4,true>` 710 ms over 2304. The fp16 form is 2.2x
   *faster* (327 vs 709 ms). This asymmetry is why unblocking GEMV nets zero on
   Laguna while gaining 8% on Gemma, and it is a real defect, not tuning.

2. **Extend the graph-parameter contract to a multi-kernel path, then drop the
   guard.** Upstream's GEMV is one kernel carrying exl3_gemm's full 10-argument
   signature, so capture just patches offsets 7/8/9. The RDNA GEMV is three
   kernels and those offsets exist on none of them, which is why
   `exl3_gemv_try_launch` deliberately reports `nullptr` — failing loudly beats
   corrupting a node. `Graph::record_param(kernel, param_id, offset)` already
   keys on the kernel function pointer, so the path needs six sites instead of
   three:

   | param | site |
   |---|---|
   | `GP_gemm_A` | `had_in` 0 |
   | `GP_gemm_A_had` | `had_in` 1, `dot` 0 |
   | `GP_gemm_B_suh` | `had_in` 2 |
   | `GP_gemm_B_trellis` | `dot` 1 |
   | `GP_gemm_C` | `dot` 2, `had_out` 0 and 1 |
   | `GP_gemm_B_svh` | `had_out` 2 |

   `Graph::record()` walks nodes and `graph_sites` in lockstep and `break`s on
   the first function mismatch, so sites must be pushed in launch order:
   `had_in`, then `dot`, then `had_out`. Verify that before trusting it.

3. **Give `exl3_mgemm` a GEMV call site** for m == 1 — the 42-48% item on MoE.

## ROCm wheel stack (TheRock): tested 7.13, no benefit, four traps

Tested 2026-08-13 against community claims of large Strix Halo gains on the
new pip-distributed ROCm ("7.14"): isolated venv (`pip install
--index-url https://repo.amd.com/rocm/whl/gfx1151/ torch` → torch
2.11+rocm7.13, the newest STABLE gfx1151 pairing; 7.14+ exists only as
nightlies, renumbered to 10.x from Aug 2026) plus a git worktree so the
working build stays untouched. Full validation ladder passed (mgemv_check,
moe_ref32, coherence). Verdict: **decode flat on all three models (15.6 /
20.1 / 7.7), prefill 3-8% SLOWER; stay on system 7.2.4.** Expected in
hindsight — decode runs this port's own kernels at near-roofline, so a
runtime upgrade has nothing to give here; the community wins come from
stacks bottlenecked on hipBLASLt/attention libraries or host overhead.

The traps, for the next attempt:

1. **TheRock wheels are runtime-only by default.** Building the extension
   needs `pip install "rocm[devel]"` (1.7 GB) and then `rocm-sdk init` to
   materialize the SDK; the resulting
   `site-packages/_rocm_sdk_devel` is a drop-in `ROCM_PATH` (has
   `.info/version`, hipcc, device libs). The core package's hipcc alone
   cannot find its device bitcode (needs `HIP_DEVICE_LIB_PATH`) and ships no
   thrust headers, which torch's headers require.
2. **7.13 header bug:** `amd_hip_cooperative_groups.h` defines
   `this_cluster()` without `inline`, so every TU including it emits the
   symbol and the `-fgpu-rdc` device link fails with duplicate symbols.
   One-word patch (`inline`) in the venv header.
3. **Dual HIP runtime via CudaDrv's dlopen** — fixed on main
   (`cuda_drv_rdna.cpp`): wheel stacks bundle `libamdhip64` without the
   `.so` dev symlink, so the old name-first dlopen loaded the SYSTEM
   runtime as a second instance beside torch's. Same-version instances
   interoperate by luck (the pre-fix state on 7.2.4 wheels + system
   7.2.4); mismatched versions fail at first triton-kernel launch with
   hipErrorContextIsDestroyed (709). The fix prefers the already-loaded
   image (`dlopen(nullptr)` + probe) with named dlopens as fallback.
4. **triton >= 3.6 unloads modules in `CompiledKernel.__del__`.**
   `bc_attn.py:_compile_kernel` copies the cubin into its own module and
   drops the triton object; on triton 3.6 the destructor then fires — during
   lazy compilation this happens mid-graph-capture ("operation not permitted
   when stream is capturing" spam). Workaround: no-op the destructor
   (bounded leak — kernels are cached per shape). Needed only if the stack
   ever moves to triton >= 3.6; the 7.2-era triton does not unload.

### 7.14 retest (2026-08-27): stack works, graphs work, still no benefit

The 2026-08-17 NO-GO (self-consistent 7.14.0a SDK segfaulting in rocr
GpuAgent::InitDma at hsa_init on this kernel) is obsolete. PyTorch nightly
now ships `torch 2.15.0.dev+rocm7.14` manylinux wheels that DEPEND on
AMD's TheRock wheels directly — `rocm-sdk-core 7.14.0` **final** plus
per-arch device packages including gfx1151 — and that release runtime
initializes fine on the same kernel the June alpha crashed. Stack under
test: `.venv714` (torch nightly, runtime) + `venv714sdk`
(`rocm-sdk-devel 7.14.0a20260624`, hipcc only; no final devel wheel is
published) + the `rocm_exl3_714` worktree.

Findings, in test order: hsa_init gate PASS (matmul, gfx1151 in the
wheel's arch list); all four graphpatch checks PASS (graph_order 0/300
violations, kernel-node / module-node / multi-node patching effective);
EXL3_ROCM_HIP_GRAPHS=1 coherent on Laguna and GLM-4.6V through single-
and multi-job generation — the 7.2.x capture hang, "Graph update failed"
and cross-job BC corruption are all absent; mgemv_check full PASS
(masked-2tok bitwise); decode A/B **flat everywhere** (Laguna 20.9/20.8,
Gemma 7.8/7.8, DS4 15.6/15.7 — and statistically identical to system
7.2.4). Same verdict as 7.13: **stay on system 7.2.4 for production.**
Graphs stay passthrough-off: correct now, but the mgemv-fast-path decode
no longer contains the route they accelerated.

Trap status on 7.14, numbered as above: (2) FIXED — the cooperative-groups
header now carries `__forceinline__`, no patch needed. (3) mutated and is
FIXED IN CODE: torch 2.15 loads its bundled runtime RTLD_LOCAL, so
CudaDrv's `dlopen(nullptr)` probe misses it and the named fallback loaded
the SYSTEM 7.2.4 runtime as a mismatched second instance ("CUDA driver
error" at every TritonKernel module load). `cuda_drv_rdna.cpp` now tries
`dlopen("libamdhip64.so.7", RTLD_NOLOAD)` — the already-mapped instance by
soname — before any filesystem search. (4) did not fire: BC kernels
compile eagerly before any capture, so the triton-3.6 destructor runs
outside capture; keep the workaround note if lazy compilation ever moves
inside. New traps: (5) torch >= 2.14 headers need `-std=c++20`
(`requires` clauses in TensorBase.h); setup.py now picks the standard from
the torch version — with a real version parse, since `"2.9" >= "2.14"` is
lexicographically true. (6) `requirements_rocm.txt`'s `triton-rocm` pin
CLOBBERS pytorch-triton-rocm on a pytorch-index stack — both own the
`triton/` import path and the mix imports as a chimera; on such stacks
install requirements minus triton, then reinstall the exact
`pytorch-triton-rocm` the nightly pins. (7) the devel SDK's lib/
dev-symlinks resolve into `../../_rocm_sdk_libraries/`, which torch's venv
owns, not the SDK venv — symlink `_rocm_sdk_libraries` into the SDK venv's
site-packages (also means the ext links exactly the libs that run) or the
link fails with "unable to find -lrocblas/-lhipblas/-lhiprand".

## Syncing to a new upstream release

Every sibling is derived from exactly one upstream file. Before deciding how to
sync one, measure both numbers — our drift from the upstream file it was
generated against, and what upstream did to that file since:

```sh
diff <(git show vOLD:exllamav3/exllamav3_ext/quant/reconstruct.cu) \
     exllamav3/exllamav3_ext/rocm/quant/reconstruct_rdna.hip | grep -c '^[<>]'
git diff --numstat vOLD vNEW -- exllamav3/exllamav3_ext/quant/reconstruct.cu
```

Low drift and high churn is the *cheap* case, not the expensive one: it means the
sibling is a mechanical rewrite and can simply be regenerated, inheriting
upstream's new code for free. The expensive case is high drift, because the
deviations have to be re-applied by hand.

Three methods, in order of preference:

1. **Regenerate by `sed` on include lines.** For siblings whose entire delta is
   include rewrites. `reconstruct_rdna.hip` is the pure case — its diff against
   upstream is six `#include` lines and nothing else, so a v1.3.0 → v1.4.1 sync
   absorbed 230 lines of new kernel with no review.
2. **Regenerate with a scripted re-application** of a small, anchored set of
   edits, asserting each anchor is found exactly the expected number of times so
   a moved or renamed anchor fails loudly instead of silently skipping. This is
   how `rope_rdna.hip` and `exl3_gemm_kernel_rdna.hip.h` are synced.
3. **Three-way merge** (`git merge-file` with base = old upstream, ours = the
   sibling, theirs = new upstream) when a deviation re-indents or restructures a
   block upstream still owns, which a textual rewrite cannot express.
   `exl3_gemm_rdna.hip`'s autotune-graph guard is the case that needs this.

Never hand-copy. The earlier port replaced upstream's `__funnelshift_r` with a
non-wrapping `fshift` that way, and it happened to land only in dead code.

After syncing, the diff against the new upstream file should be *exactly* the
documented deviations — check that, not just that it compiles. Then run
`rocm_tools/hipcc_probe.sh --all`, whose exclusion regex must stay in step with
`ROCM_EXCLUDE` in `setup.py`.

### v1.3.0 → v1.4.1

Recorded because it is the worked example, and because the cost was concentrated
in a way that is not obvious in advance. Upstream moved 115 files and +12.5k
lines; the port needed five siblings touched.

| sibling | our drift | upstream churn | method |
|---|---|---|---|
| `reconstruct_rdna.hip` | 10 | +230 | regenerate (sed) |
| `exl3_kernel_map_rdna.hip.h` | 180 | 3+/1- | two args added to `EXL3_MGEMM_ARGS` |
| `exl3_gemm_kernel_rdna.hip.h` | 58 | 15+/11- | regenerate (scripted) |
| `exl3_gemm_rdna.hip` | 146 | 35+/6- | three-way merge |
| `rope_rdna.hip` | 65 | 155+/36- | regenerate (scripted) |

The other thirteen siblings had **zero** upstream churn, including every
high-drift one — `exl3_gemv_kernel_rdna.hip.h` (666), `exl3_gemm_inner_rdna.hip.h`
(885), `exl3_gemv_rdna.hip` (418), `codebook_rdna.hip.h` (163). In particular
`exl3_gemm_inner.cuh`, `exl3_moe_kernel.cuh`, `exl3_moe.cu` and `comp_units/` are
byte-identical between the two tags, so the split-K fixes above carried forward
untouched and `MOE_TILESIZE_K` did not need re-validating.

Upstream's own change to the GEMM siblings is additive: two kernel arguments
(`size_n_list`, `C_list`) that let one `exl3_mgemm` call write outputs of
differing widths to separate pointers. They are consumed only by
`libtorch/dsv4_attn.cpp`; every other caller passes neither, leaving
`size_n_list_ptr` null and the kernel on its existing path. The four-line change
to `EXL3_MGEMM_ARGS` must match upstream's `exl3_kernel_map.cuh` exactly, since
the comp_units instantiate against it.

The upstream surface stayed at four files and the conflict was four lines
(`setup.py` swapping `kbnf`+`formatron` for `llguidance`, `__init__.py` exporting
`LLGuidanceFilter`, two README model-list lines). `triton_paged.py` was untouched
upstream. `requirements_rocm.txt` is ours and additive, but it duplicates the
dependency list, so it needs the same `llguidance` swap or a cold install breaks.

### v1.4.1 → v1.4.4

71 upstream commits (GLM 5.2, DSA-on-MLA, quantized-vision defaults, CPU-MoE
expert maps, new optimization pipeline). Cheaper than the previous sync: zero
merge conflicts (`setup.py`, `__init__.py`, `triton_paged.py`, even README
merged clean), and the fragile paths (`exl3_gemm_inner.cuh`, all of
`exl3_moe*`/`exl3_gemv*`/`comp_units`, reconstruct, quantize, kernel map) are
byte-identical between the tags — the split-K fixes and the GEMV decode work
carried forward untouched. `EXL3_MGEMM_ARGS` did not change, so the comp_units
needed nothing.

| sibling | our drift | upstream churn | method |
|---|---|---|---|
| `rope_rdna.hip` | 71 | 51+/~90- | regenerate (scripted); upstream deleted `post_rope_norm`/`apply_norm_uw` and Nanochat, taking the second lane-0-bug site with it — one guarded site remains (`apply_norm`), bug still live upstream |
| `moe_handoff_rdna.hip` | 3 (include lines) | +59, layout change | regenerate (sed) — `MOE_FLAGS_SIZE` grew 2×→3× slot regions (`consumed[]`), new `MOE_JOB_KIND_COMPUTE_GATED`; all inherited for free, but a stale sibling here is silent shared-memory corruption, not a compile error |
| `exl3_gemm_rdna.hip` | 272 | 11+/4- | apply the two hunks (doc comment + drop the `num_tokens == 1 \|\| min_index < 0` TORCH_CHECK); drift is 272 now, not the 146 recorded at v1.4.1 — the mgemv routing and width-list work grew it post-sync |
| `exl3_gemm_kernel_rdna.hip.h` | 58 (unchanged) | 34+/12- | apply the three hunks: position-preserving `-1` masking for `num_tokens > 1` range filtering, plus the two stale-scratch reduction guards |

**Port-specific consequence of the masking change:** upstream made
`num_tokens > 1` legal *with* expert-range filtering by switching the
cooperative kernel from index compaction to in-place masking. Our mgemv fast
path (`exl3_mgemv_rdna.hip`) still compacts — its grouped reduce divides
`packed / num_tokens`, which the new combination breaks (per-token slot runs
collapse, and packed need not divide). Fixed by declining
`num_tokens > 1 && min_index >= 0` in `exl3_mgemv_try_launch` so those calls
fall through to the cooperative kernel. Only TP-sharded / CPU-split expert maps
produce the combination; if CPU-split MoE decode ever matters for throughput,
that fall-through is the place to look.

New device code is all in `dsa_topk.cu` (+481, DSA-on-MLA top-k split/merge):
six guarded Hillis–Steele `__shfl_up_sync` scans, no asm/PTX/mma. Validated
with distinct per-lane values by `rocm_tools/shfl_up_scan_check.hip` (PASS on
gfx1151 — the `lane >= o` guard makes clamp-vs-wrap moot, in-range delivery is
correct). `routing.cu`'s +241 and `mla_attention.cpp`'s +416 lines contain no
new warp ops. `graph_rdna.hip` needed nothing — it includes `graph.cuh`, so the
new `GP_dsa_indices` enum flows through.

## Test status on RDNA

Measured at v1.4.1. `tests/` hardcode a device index — `cuda:2` in most files,
`cuda:1` in `test_reconstruct_had.py` — so a single-GPU machine has to rewrite
both before anything collects.

Note that several `tests/test_*.py` files are `main()` scripts rather than pytest
modules. pytest reports "no tests collected" and moves on, which reads as a pass
at a glance. **Run those directly** — they carry the reference checks for the
newest kernels, and two of the three most valuable results below come from them.

| test | result |
|---|---|
| `test_rope` | 64 passed (was 30 failed before the lane-0 fix; 60 at v1.3.0) |
| `test_rope_yarn` | 8 passed |
| `test_cache_rotate` | 32 passed |
| `test_mla` | 53 passed |
| `test_gated_delta_rule` | 21 passed |
| `test_sampler` | 140 passed |
| `test_triton_paged_overflow` | 3 passed |
| `test_reconstruct_had.py` (script) | ALL PASS, rel err ~1e-3 — covers the `reconstruct_had` kernel new in v1.4.1 |
| `test_dsa_kernels.py` (script) | ALL PASS — DeepSeek V4 sparse attention, indexer and top-k, rel err ~3e-4 |
| `test_ext_norm_` | 336 failed — **upstream test defect, not a kernel bug.** New in v1.4.1; calls `ext.rms_norm(x, w, y, eps)` against a binding upstream itself declares with 8 parameters. `norm.cu`, `norm.cuh` and `bindings.cpp` are byte-identical to upstream here, so it fails the same way on CUDA. Called with the real signature the kernel matches an fp32 reference to 4.2e-4 across 32 shapes. |
| `test_kv_quant` | 60 failed — same class, and long-standing: `quant_cache_paged()` arity mismatch |
| `test_dsv4_compress_kernel.py` (script) | same class again — `dsv4_compress()` arity mismatch. The kernel itself runs correctly under a real DeepSeek V4 generation. |
| `test_dsv4_cached`, `test_dsv4_state` | collection error — both `import compare_deepseek_v4_hf_`, which upstream never committed (`git ls-tree v1.4.1` does not contain it) |
| `test_qgemm`, `test_quant_fn` | collection error — require models at hardcoded `/mnt/str/...` paths |

Three separate upstream tests now call an ext binding with the wrong arity. Treat
a `TypeError: incompatible function arguments` from `tests/` as an upstream
staleness signal and check the declaration before suspecting the port.

### End-to-end generation

| model | result |
|---|---|
| Gemma-4-31B-it (dense) | coherent |
| GLM-4.6V 3.55bpw (MoE) | coherent |
| DeepSeek-V4-Flash 2.04bpw | coherent — new in v1.4.1, working with no ROCm-specific code |

The dense model remains the cheapest control for an MoE fault; run it first.

### Open: segfault at interpreter teardown

Any process that has loaded a model exits with SIGSEGV *after* all output is
produced and all work has completed. Characterised so far:

- needs a loaded model — plain torch HIP allocation and a bare `rms_norm` call
  both exit cleanly;
- not model-specific — dense Gemma and MoE GLM both do it, so it is not the CPU
  MoE handoff's worker threads;
- no Python traceback under `PYTHONFAULTHANDLER=1`, so it is native teardown
  (static destructor ordering against an already-torn-down HIP runtime);
- `os._exit(0)` after the work avoids it completely, which is the workaround if
  it matters for a script.

**Not established whether this predates v1.4.1** — it produces no output before
the process dies, so it could have been present and unnoticed. Bisecting it means
rebuilding the pre-rebase branch (`pre-v141-rebase`) and re-testing. Harmless to
generation quality either way, but it will show up in a server shutdown.

## Verification tools

Under `rocm_tools/`:

| tool | checks |
|---|---|
| `wmma_check.hip` | WMMA operand order and fragment layout against a CPU reference |
| `gemm_check.hip`, `gemv_check.hip` | GEMM / GEMV kernels against a CPU reference; gemv_check runs both dot cores, and `GEMV_SWEEP=1` adds the DRAM-resident bandwidth sweep (build with the FLAGS block from `build_coop_check.sh` minus the torch libs, single TU) |
| `mgemv_check.py` | mgemv against the cooperative kernel on real weights: packing, grouped reduce, every routing config |
| `gemm_coop_check.hip` | cooperative launch against the same work without it |
| `moe_ref32.py` | fused MoE **and** the per-expert path against an fp32 reference built from dequantized weights |
| `moe_check.py` | fused MoE against the per-expert path (two fp16 implementations — see its own caveats) |
| `nan_locate.py` | names the first module in a forward pass whose output goes non-finite |
| `bench_model.py`, `bench_moe.py`, `bench_mgemm.py`, `bench_gemv_vs_gemm.py`, `bench_prefill_tiles.py`, `bench_decode_splits.py` | timing, median of repeats, flagging spreads above the noise floor |
| `profile_decode.py` | rocprofv3 wrapper for a decode run; the profile-before-implementing tool |
| `hipcc_probe.sh` | per-file compile probe, without rdc |
| `attn_8k_check.py` | Triton paged attention vs fp32 oracle at KV 4K-16K, straddling the 8192 prefill-split activation; SWA windows, batched decode (attn_check.py's old max ctx was 1000) |
| `graphpatch_check.hip`, `graphpatch_module_check.hip`, `graphpatch_multinode_check.hip` | hipGraphExecKernelNodeSetParams semantics: runtime-launched, module-launched (Triton-style), and multi-node/ping-pong patched graphs. All PASS on 7.2.4 — the graph corruption is not the patch primitive |
| `graph_order_check.hip` | back-to-back hipGraphLaunch ordering through a shared buffer under deep queues. PASSES on 7.2.4 |
| `stream_wedge_check.hip` | spin kernel + pageable hipMemcpyAsync on one stream. **HANGS the process on 7.2.4 (reproducible)** — the objective repro for this stack's async/stream defects; rerun on every new ROCm before trusting HIP graphs |

## HIP graphs: disabled on ROCm (graph_rdna.hip), and how we got there (2026-08-15)

`graph.cu` is excluded on ROCm in favor of `rocm/graph_rdna.hip`, which by
default never begins a capture: every BC step executes its `run_gr()` sequence
eagerly on the live stream (the same code path as each slot's first warmup run).
`EXL3_ROCM_HIP_GRAPHS=1` restores upstream capture/replay. Measured cost on
Laguna-S-2.1: decode 20.8 -> 17.7 t/s (-15%), prefill unchanged. What it buys:
graph-capture hangs are impossible by construction (observed on this stack as
intermittent stalls; same class as vLLM's open capture-hang issue on ROCm 7.2.x,
and llama.cpp ships HIP graphs off by default behind a CMake flag), and it
removes exposure to HIP's missing capture-time validation (pytorch#155684:
operations CUDA rejects during capture are silently captured on ROCm).

What is PROVEN vs SUSPECTED, so nobody re-litigates the wrong part:

- The patch/replay primitives are NOT the defect: see the four graph checks
  above, all passing on 7.2.4.
- A months-old in-tree datapoint: the MoE BC graph route died with "Graph
  update failed" + segfault on GLM decode (recorded at the rocm_py mgemm
  patch), closed rather than diagnosed at the time.
- `stream_wedge_check.hip` hangs this stack reproducibly without any graph
  involvement — the async machinery under HIP graphs is demonstrably unsound
  here.
- The multi-turn "coherency collapse" that triggered this investigation was
  NOT graphs and NOT ROCm at all — it was resolved the same day as sampler
  arithmetic: OAI-style frequency/presence penalties (0.10/0.15) with
  TabbyAPI's default penalty_range = max_seq_len, applied by SS_PresFreqP over
  past_ids = the full sequence INCLUDING the prompt. At 8K context a common
  token carries freq_penalty × ~350 occurrences ≈ −35 logits: function words
  die first, then generation flees to the only unpenalized vocab region
  (never-used tokens — emoji/hashtag spam). Same numbers are harmless on
  backends that bound the window (llama.cpp repeat_last_n=64) or count only
  the completion (OpenAI), which is why they looked innocent. Fix: bounded
  penalty_range (512-2048) or freq_p ≈ 0. Confirmed by the user in real chat.
  Along the way, kernel-level exonerations that remain valid: attention parity
  to 16K incl. the 8192 split path, rope to pos 32K, YaRN config, cache
  rotate, per-position NLL flat to 16K through the nc path. Also a
  methodological note: greedy loop-collapse probes are pure decoding chaos
  (1/8 collapse in EVERY config at different knife-edge prompt points) — never
  use them as a coherence metric.
- ROCm 7.14 reworked graph replay ("allocation nodes no longer block during
  replay; physical memory reused across nodes instead of mapped/unmapped per
  launch") — the right neighborhood for the suspected allocator interaction.
  **TESTED 2026-08-27 — graphs WORK on 7.14, but buy nothing.** See "ROCm
  wheel stack" above for the stack and findings. On torch nightly
  2.15+rocm7.14 (final 7.14.0 runtime, hsa_init fine on this kernel): all
  four graphpatch checks pass, EXL3_ROCM_HIP_GRAPHS=1 runs Laguna and
  GLM-4.6V through single and multi-job generation coherently — no capture
  hang, no "Graph update failed", no cross-job corruption. Decode A/B is
  flat on all three models (20.9/20.8, 7.8/7.8, 15.6/15.7): the Aug-13
  decode work (mgemv fast path, width lists) removed the graph-dependent
  cooperative route from decode, so the −15% passthrough penalty this flag
  once recovered no longer exists. Graphs stay off by default; the flag is
  now known-safe on 7.14-class runtimes should a path that benefits return.
  WHY flat, PROVEN by Kineto trace (2026-08-27, Laguna, 31 decode tokens
  on the 7.14 stack; scratch tool `gap_profile.py`, built on
  profile_decode's 1-token-prompt method): decode launches ~1953 kernels
  per token (the gemv/mgemv had_in + dot + had_out triplets are ~800 of
  them). Graphs-ON collapses the HOST side exactly as designed — 53,351
  hipLaunchKernel calls fall to 7,595 plus 2,976 hipGraphLaunch replays
  (~96/token, param patching visible as hipGraphExecKernelNodeSetParams)
  — yet the DEVICE timeline is identical in both traces: same 2.1us
  median inter-kernel gap, same ~10%-of-span idle under the profiler,
  same busy time. Conclusion: that 2.1us gap is the command processor's
  per-dispatch latency between dependent kernels, which graph replay on
  ROCm does not remove; the host was never the bottleneck (async
  submission, GPU 100% busy via rocm-smi). The recoverable overhead is
  reachable only by launching FEWER kernels (fusing the had/dot/had
  triplet, batching expert slots), not by replaying the same kernels
  from a graph — that dispatch-gap budget (~4ms/token profiled, less
  unprofiled; A/B bounds it under 0.5% real) belongs to the mgemv
  split-K/fusion arc. Cross-model (same tool, 31 tokens, profiled):
  launches/token 1953 Laguna / 1622 Gemma / 2172 DS4; dispatch-gap
  3.5-5.2 ms/token everywhere, but as % of span it is 10.0 / 2.9 / 7.8 —
  fusion pays most on fast MoE tokens, least on big dense.
  RESOLVED (2026-08-28), DS4 per-layer stall: 1271 big gaps (~41/token,
  once per layer, ~100-500us each, ~5 ms/token ≈ 8%) sat between
  dsv4_compress_store and _dsa_attn_split — SAME stream (tid 1), next
  kernel already enqueued (host parked in one ~60 ms hipDeviceSynchronize
  per token), so a DEVICE-side dispatch stall, unaffected by
  EXL3_ROCM_HIP_GRAPHS (1369 vs 1374 gaps A/B). Root cause confirmed by
  compile inspection: _dsa_attn_split_kernel at upstream's CUDA tuning
  (BLOCK_H=16, num_warps=4) sits at the 256-VGPR ceiling with 2050 VGPR
  spills and 5332 B/item scratch — the only scratch user in the decode
  stream, so every layer's dispatch after scratch-free kernels pays the
  queue's scratch reconfiguration. 16-variant compile sweep
  (scratch tracks spills; none reach zero — residual ~130 at w16/H4/N16):
  BLOCK_H=8 + num_warps=8 (BLOCK_N untouched) is the bench winner —
  spills 438, scratch 1756 B, big gaps 1369 -> 45, decode 15.6 -> 17.8
  t/s on the 7.14 stack and 15.9 -> 17.9 on system 7.2.4 (+13-14%).
  Deeper-spill variants bench WORSE (H4/w16 14.9, H4/w8 17.0): past the
  stall threshold, tile shape matters more than residual spills. Shipped
  as the EXL3_ROCM_DSA_TUNE rocm_py patch (default on; =0 restores
  upstream tuning): bc_dsa.BLOCK_H=8 via module attr + num_warps=8 via a
  _compile_kernel wrapper keyed on the kernel name, rebound in bc_attn,
  bc_dsa AND bc_mla (each holds its own from-import reference). bc_mla's
  DSA-on-MLA (GLM 5.2) hardcodes a local BLOCK_H=16, so it gets only the
  warps half (~1428 spills) — no model small enough to test here anyway.
  Watch item: tests/test_mla_dsa.py::test_dsa_selection[300] failed ONCE
  ("row 2..." assertion) on the first tuned full-suite run, then passed 5
  consecutive full-suite runs and in isolation; baseline also clean.
  Unreproduced — if it recurs, bisect with EXL3_ROCM_DSA_TUNE=0 first.
  Tools: spill sweep = scratchpad dsa_sweep.py pattern (triton.compile
  interception, parse .vgpr_spill_count / .private_segment_fixed_size
  from ck.asm["amdgcn"]); gap evidence = rocm_tools/gap_profile.py. The "-15% decode without graphs" recorded
  2026-08-15 does NOT reproduce today on either stack (system 7.2.4
  graphs-off matches historical graphs-on numbers exactly); treat it as
  stale — most plausibly it amortized the cooperative kernel's
  per-launch setup in a configuration the mgemv fast path had already
  made obsolete, or it was first-run-high measurement noise. stream_wedge_check itself was deliberately NOT run
  (machine-wedge risk; user's call) — the graphpatch ladder was accepted
  as the gating evidence.

## v1.5.0 sync (2026-09-20) -- what changed under the siblings, and what is unverified

Brought forward by `git merge v1.5.0` plus, for each sibling whose upstream twin
changed, applying upstream's own v1.4.4 -> v1.5.0 diff to the sibling (it applied
cleanly to nine of ten; `quantize_rdna.hip` and `rope_rdna.hip` were regenerated
from the v1.5.0 twins instead). *At the time of the merge*, none of this had been
compiled or executed on RDNA -- the machine that did the sync had no ROCm
toolchain. **Superseded:** it has since been built and validated on gfx1151; see
"v1.5.0 validated on gfx1151" below for what ran and what remains opt-in.

| Area | What upstream did | What the RDNA layer does now |
|---|---|---|
| `exl3_mgemm` sliced mode | New `n_stride_list` / `had_src_list` / `num_had_src` args; per-source input Hadamard, strided B/C rows (`SlicedMultiLinear`, used for one-launch Q/K/V decode) | Mirrored into `exl3_gemm_kernel_rdna.hip.h` and `exl3_gemm_inner_rdna.hip.h` (`size_n_stride`, `blocks_n_full`). The m == 1 mgemv shortcut is bypassed in sliced mode. **Python keeps it off** (`EXL3_ROCM_QKV_SLICE=1` to test): the v1.4.4 pairwise bundles are the validated path. |
| Fused MoE (`exl3_moe`) | `count_lo`/`count_hi` expert tiers, `output_scratch`/`fused_base` deterministic slots + `exl3_moe_gather` (on by default: `EXL3_MOE_FUSED_DET`), 32/64-row tile instances (mul1) | Tiers and scratch path applied verbatim (they are addressing, not tensor-core work). 32/64-row tiles **fall back to the 16-row instance** over the same expert range -- identical numerics. `rocm_py` sets `MTILE=False` so Python issues one launch, not three. |
| MoE decode (bsz <= 8) | `BC_BlockSparseMLP::run_bszN` now calls `exl3_moe_coop` (PTX GEMV based); the three-mgemm graph route is gone | Stub; `rocm_py` reinstates the v1.4.4 per-token `exl3_mgemm` route (2026-09-23, see "MoE decode route restored" below). The fused-`exl3_moe` steer remains behind `EXL3_ROCM_MOE_MGEMM_ROUTE=0`. |
| Batched expert reconstruct | `reconstruct_had_batch` / `reconstruct_batch` (blockIdx.z over a pointer table), `hgemm_batched` (`cublasGemmStridedBatchedEx`), `had_r_128_batch` | Batch kernels applied to `reconstruct_rdna.hip` (same tile body); shim maps the strided-batched hipBLAS call. **Python keeps the tier off** (`EXL3_ROCM_BATCH_RECON=1` to test). |
| `hgemm_f16acc` | fp16-accumulate MMA GEMM for GeForce | Stub; `hgemm_recon` = `hgemm`. RDNA has no fp32-accumulate rate penalty, so nothing is lost. |
| Quantizer | Tile length 160 (n-gram rows), block geometry per K, `quantize_tiles_scratch`, sm_120 "optimized" specialisations with a codebook LUT | `quantize_tiles_kernel_rdna.hip.h` regenerated (include swap only); instance files gain the `bool optimized` parameter and `_l160` getters; `quantize_tiles_use_optimized()` is hard `false` and the LUT code is dropped. Note `__CUDA_ARCH__` is 1 under the shim, so `qt_num_threads` takes the `arch < 1000` branch in both passes. |
| `rope.cu` | Adopted the lane-0 guard on the norm warp-sum store, padding guard, `record_param` index 26 -> 25 | Sibling regenerated; back to the single `= {}` line. |
| `graph.cu` | Error checks on node update / launch | Applied; only reachable with `EXL3_ROCM_HIP_GRAPHS=1`. |
| Shim | new `cublasGemmStridedBatchedEx`, `cudaFuncGetAttributes`, `__threadfence` (native) | Added the first two to `cuda_shim/cublas_v2.h` and `hip_compat.hip.h`. |

Files whose twins did not change: `exl3_gemv_rdna.hip`, `exl3_gemv_kernel_rdna.hip.h`,
`exl3_gemv_int8_rdna.hip`, `exl3_mgemv_rdna.hip`, `exl3_dq_rdna.hip.h`,
`codebook_rdna.hip.h`, `rdna_wmma.hip.h`, `cuda_drv_rdna.cpp`, `cpu/moe_handoff_rdna.hip`.

First things to run on hardware, in order: `rocm_tools/hipcc_probe.sh --all`; a
dense model (Gemma-4) for the sliced-free attention path; an MoE model (GLM-4.6V)
at bsz 1 to exercise the rerouted decode and the deterministic gather; then flip
`EXL3_ROCM_QKV_SLICE=1` and `EXL3_ROCM_BATCH_RECON=1` one at a time and compare.

### v1.5.0 validated on gfx1151 (2026-09-20), and the RDNA4 fallback

The "first things to run" above all ran, plus more; everything passed:
hipcc_probe --all clean; sibling drift audit exact against the documented
deviation sets (rope's lane-0 fix is upstream now, so the sibling is down to
the header block and the `= {}` line); mgemv_check full PASS incl. the
masked-2tok bitwise case; test_reconstruct_had 12/12 against the +227-line
batched rework; test_dsa_kernels ALL PASS; pytest suites 161/161; coherent
generation on Gemma-4-31B, Laguna, GLM-4.6V, DeepSeek-V4-Flash and
Qwen 3.8-Flash-Next (ngram table load, QSA attention, 512-expert fused-MoE
steer, single and multi-job). test_ple_prefetch_gen_ / test_ngram_prefetch_
need upstream's /mnt/str stub models and could not run; Qwen 3.8 loading its
PLE table end to end is the working coverage. Benchmarks deliberately not
taken yet: the MoE decode reroute is a known regression to quantify
separately (see the table above; exl3_moe_coop is the item).

**RDNA4 (gfx1200/gfx1201) compile fix.** A gfx1201 build died in all 20
fused-MoE comp units: "Cannot select: intrinsic llvm.amdgcn.wmma.f32.16x16x16.f16"
— the gfx11 WMMA intrinsics have no gfx12 encoding, and the fused-MoE
instantiations of exl3_gemm_inner_rdna.hip.h are the ONLY live WMMA users
left (the GEMM/GEMV instantiations take the dot-core paths and compile
clean; classic dead-code-that-wasn't, in reverse). Fix shipped:
`rdna_wmma::mma_sync` compiles to `__builtin_trap()` under
`__gfx1200__/__gfx1201__` (loud, never silently wrong; the other wrappers
keep the bare gfx11 builtins so any future gfx12 instantiation fails at
compile, which is correct), and rocm_py clears `fused_mode_buffers` on
gfx120x devices so MoE runs the per-expert path and the trap is unreachable
(`EXL3_ROCM_RDNA4_FUSED_MOE=1` bypasses the steer for a future gfx12 port).
Verified by `GPU_ARCH=gfx1201 hipcc_probe.sh --all` — compile-level only;
**no RDNA4 hardware has ever run this port.** A real gfx12 WMMA port means
half-size fragments (no operand duplication across wave halves) and new
lane mappings in rdna_wmma.hip.h — hardware required to validate.

## ROCm 10.0 (2026-09-22): stack up, graphs on by default via a runtime gate

Branch `rocm-10` (worktree `rocm_exl3_10/`), venv `.venv10`. Stack: AMD's
stable index `https://stable.repo.amd.com/rocm/whl-next/` -- `torch
2.13.0+rocm10.0.0` (same torch as the 7.2.4 production venv, so an A/B
isolates the ROCm change), `triton 3.8.0+git4cff872c.rocm10.0.0` (NOT the
3.7.1 the pre-install research predicted), `rocm[libraries,devel,device-gfx1151]
==10.0.0`. `rocm-sdk init` expands `site-packages/_rocm_sdk_devel` (hipcc,
device libs, `.info/version` = 10.0.0), a drop-in `ROCM_PATH`. The ROCm 10.0
runtime is HIP 7.15: `torch.version.hip` = 7.15.26333, `hipRuntimeGetVersion`
= 71526333 (7.2.4: 70253211, 7.14 wheel: 71460850).

Build: `env -i ... PATH=$SDK/bin:... ROCM_PATH=$SDK ROCM_HOME=$SDK pip install
-e . --no-build-isolation`. Scrub the login shell's /opt/rocm leakage first.
**ROCM_HOME matters**: torch's extension builder takes the rpath from it, not
from ROCM_PATH; without it the .so carries RUNPATH /opt/rocm-7.2.4/lib. That
was inert here (torch loads its bundled libamdhip64.so.7 first and the loader
reuses it by soname -- verified via /proc/self/maps: only the wheel's runtime
is mapped) but it is the duplicate-runtime trap waiting to happen.

Results, all on gfx1151 / this kernel: hsa_init gate PASS; `hipcc_probe.sh
--all` 117/117 under LLVM 24 with no new warnings; full `-fgpu-rdc` build
links (the new-offload-driver RDC ABI concern did not materialise);
exl3_stack_check PASS; all seven rocm_py patches apply (DSA retune included);
graphpatch_check / graphpatch_module_check / graphpatch_multinode_check all
EFFECTIVE and graph_order_check 0/300 violations, run against the wheel
runtime (`LD_LIBRARY_PATH=_rocm_sdk_core/lib` -- the harness binaries carry no
rpath and would otherwise pick up /opt/rocm-7.2.4); `EXL3_ROCM_HIP_GRAPHS=1`
Laguna generation coherent with no capture errors.

**Triton 3.8 does not need the `__del__` no-op** (trap 4 above): its
`CompiledKernel.__del__` unloads only when `self.module` was initialised by a
launch, and `bc_attn._compile_kernel` never launches the Triton object (it
copies the hsaco into ext.TritonKernel). No "operation not permitted when
stream is capturing" spam observed with graphs on.

**Default policy (the point of this branch):** `graph_rdna.hip` now gates
capture on the loaded runtime -- `hipRuntimeGetVersion() >= 71400000` (7.14+)
turns graphs ON, anything older stays eager passthrough. So a 7.2.4 user can
never hit the 7.2.x capture hang, and a ROCm 10 user gets graphs without an
environment flag. `EXL3_ROCM_HIP_GRAPHS=1/0` overrides either way (bisect
handle). rocm_py reports the effective state in `describe()`.

Not done yet: decode A/B 7.2.4 vs 10.0 (graphs were flat on 7.14 -- see
"7.14 retest" -- so expect no headline change; the ROCm 10 gains, if any,
will come from the profiler (rocprofv3 1.3.5: 228 gfx1151 counters vs 31 on
7.2.4) pointing at fusion targets), `stream_wedge_check` (never run without
asking -- machine-wedge risk), rocm-systems PR #10714 (kernarg-exhaustion
stale-packet bug: graph_rdna.hip checks every update rc, keep it that way).
Pre-existing, unrelated to ROCm 10: loading DeepSeek-V4-Flash prints Triton
"no matching matrix core intrinsic for wmma version 1" diagnostics from
dsa_triton.py's tl.dot shapes on both stacks; Triton falls back to a non-WMMA
dot and generation is coherent.

**Multi-turn acceptance (2026-09-22, `rocm_tools/chat_probe.py`, Laguna
4bpw, six turns to ~1.5K ctx + two concurrent jobs):** graphs-on ROCm 10,
graphs-off ROCm 10 and 7.2.4 produce the same per-turn token counts and the
same repetition scores for the same seed -- replay is token-identical to eager.
With proper stop ids the worst rep4 is 0.03. The "replay repeats itself over
and over on longer context" symptom remembered from earlier sessions
reproduced here on ALL THREE stacks with no stop ids: Laguna ends turns with
`</assistant>` (id 24, in `config.eos_token_id_list`), and a client that stops
only on the tokenizer's EOS runs past the finished answer and repeats it
verbatim (rep4 0.51-0.55 by turn 6). It was a client stop-token gap, not
graphs. exl3_server already unions `eos_token_id_list` with `eos_token_id`.

## MoE decode route restored (2026-09-23): 10.4 -> 21.1 t/s on Laguna

`bench_model.py` on Laguna-S-2.1 4.03bpw (MoE, 256 experts top-10, shared
expert), `-p 512 2048 -n 128 -r 4`, system 7.2.4: decode **10.4 t/s** with the
v1.5.0-sync steer (bsz <= 8 through the fused `exl3_moe` kernel) against the
**20.8** recorded at v1.4.4 on the same model and tool. "Benchmarks
deliberately not taken" at the sync hid a 2x regression on every MoE model's
decode; dense models were unaffected. ROCm 10.0 measured the same 10.6 with
graphs on or off, so the runtime was never the variable.

Why: `exl3_moe` is a 16-row WMMA tile GEMM. At bsz 1 each expert has one
useful row, so fifteen sixteenths of every tile is padding and the kernel
runs at the flat ~53 GB/s the tile path always shows, plus the argsort/bincount
host syncs the fused path pays per layer. Upstream's v1.4.4 route ran three
`exl3_mgemm` calls per token (gate, up, down-with-weights), each num_tokens
== 1, and on RDNA every one of those lands on the mgemv fast path
(`exl3_mgemv_try_launch`: barrier-free dot core, weights streamed once,
120-240 GB/s). v1.5.0 deleted that route for `exl3_moe_coop`, which is not
ported.

Fix (`rocm_py`, no upstream edit, no kernel change): for the duration of
`BlockSparseMLP.forward`, `self.bc` is a proxy whose `run_bszN` is the v1.4.4
loop written into `experts_cfg.out_bszn[i]` (what the bszN branch reads
back); every other attribute forwards to the real `BC_BlockSparseMLP`.
`bc_sh_exp` is forced False after `load_local` so the forward tail runs the
Python shared-expert path (the fused kernel merged it in-kernel). Expert-range
shards pass `cfg.min_expert / max_expert` as `run_bszN` does.

Validated: route vs fp32 reference at bsz 1 / 3 / 8 over two layers
(`route_ref32` on top of `moe_ref32.py`): relmean 0.02-0.10%, identical to
the fused kernel's error at every point. Six-turn `chat_probe.py` coherent.
Decode **21.1 t/s**, prefill unchanged (192 / 314 t/s). Switches:
`EXL3_ROCM_MOE_MGEMM_ROUTE=0` restores the fused steer; `EXL3_ROCM_MOE_BSZN=1`
leaves upstream dispatch alone (raises in the stub). Porting `exl3_moe_coop`
is no longer the decode item; it would have to beat the mgemv path to matter.

`profile_decode.py` (24 tokens, Kineto) confirms the dispatch: with the route
the top kernel is `exl3_mgemv_dot_kernel_splitk<4,false,2,8>` (3408 calls =
3 per MoE layer per token, 231 ms) and no `exl3_moe_kernel` runs at all; with
`EXL3_ROCM_MOE_MGEMM_ROUTE=0` `exl3_moe_kernel<4,256,2,16>` is 1382 ms =
66.6% of GPU time (1.22 ms per layer vs ~0.34 ms for the route's three calls)
plus a per-layer `hipMemcpyWithStream` readback sync (274 ms host). Two leads
the same profile exposes, both independent of the MoE route:
- Host launch cost is now the limiter: ~1720 launches/token, GPU idle 36% of
  wall under the profiler (~20% unprofiled, from 21.1 t/s vs 38 ms/token of
  kernel time). The route issues ~9 launches per MoE layer; on a runtime with
  working graphs (ROCm 10, branch `rocm-10`) capturing the per-layer loop into
  a torch CUDA graph (static buffers already exist: yh / interm_* / out_d /
  bcast_sel_bsz1) would collapse them to one replay. First concrete ROCm 10
  item.
- `Cijk_Ailk_Bljk_HHS_BH_MT128x128x32...` (rocBLAS/hipBLASLt HGEMM, 128x128
  tile) costs 134 ms = 15% of decode GPU time, once per layer per token, in
  BOTH profiles: an unquantized linear is going through hgemm at m == 1 on a
  tile kernel. Identify the caller; a GEMV-shaped path would recover most of
  it.

## The narrow-output hgemm (2026-09-23): 21.1 -> 23.3 t/s on Laguna

The 15%-of-decode `Cijk_..._MT128x128x32_MI16x16x16` kernel above is
`BC_Attention`'s headwise attention gate: `hgemm_gr(x2, g_weight, s.g2)` in
`libtorch/attention.cpp`, a (1 x 3072) @ (3072 x 48) fp16 product once per
layer per token (Laguna stores `g_proj` unquantized). It is not the Python
`LinearFP16` path (a patch there never fired: the BC step issues the GEMM
from C++), and an LD_PRELOAD interposer on `hipblasGemmEx` sees nothing
because hipBLAS's headers map that entry point to a versioned symbol; the
caller was found by reading `attention.cpp`.

What the libraries do with it (`rocm_tools/hgemm_narrow_probe.py`, K = 3072,
rotating working sets, 7.2.4 unless noted):
- rocBLAS (hipBLAS's default backend, what `cublasGemmEx` reaches): ONE
  128x128 tile workgroup walking K in 96 serial LDS round trips on one WGP,
  63-75 us flat from m = 1 to 32 for a 295 KB matrix, ~50x its bandwidth
  bound. Wide N is fine on the same path (N = 12288 at m = 1: 371 us =
  ~200 GB/s), so the pathology is narrow N without a K split.
- `ROCBLAS_USE_HIPBLASLT=1`: the same entry point takes 10-22 us at narrow N
  and ~40% less at N = 2048, on 7.2.4 and 10.0 alike -- but hipBLASLt loses
  to rocBLAS at m >= 128 and at wide N on this part, and as a process-wide
  switch it cost prefill 13-17% (192 -> 167, 314 -> 259 t/s). There is no
  per-call backend switch in rocBLAS or hipBLAS, and linking hipBLASLt
  directly would add a second copy of a library torch already bundles.
- An fp32 GEMV recipe through ATen (torch.mv / broadcast multiply + sum,
  shipped briefly as the first sibling) wins only up to N = 192 / 128 / 96 /
  64 at m = 1 / 2 / 4 / 8, is 2-5x SLOWER above that, and cannot run inside
  graph capture (caching-allocator state in the graph, see graph_rdna.hip).
- torch.matmul on the 7.2.4 wheel also goes through rocBLAS (74 us); on the
  10.0 wheel it goes through hipBLASLt (10 us). That is why ROCm 10 looked
  "fixed" in torch and not in the ext.

Fix: `rocm/hgemm_rdna.hip` (sibling of `hgemm.cu`, which setup.py excludes)
runs m <= 8, N <= 256 (`EXL3_RDNA_HGEMM_NARROW_N`, 0 disables) on two plain
kernels: split-K partials (one block per 64-row K-slice, thread = column x
row-lane, LDS reduce over lanes) into the DevCtx workspace, then a reduce
into C in its dtype and row stride. No allocations, so it is capture-safe and
the captured BC gate node takes it too. Everything else stays on rocBLAS.
Measured (ext.hgemm column): 8 / 12 / 13 / 16 / 24 us at m = 1 for N = 16 /
48 / 64 / 128 / 256, 14-45 us at m = 8 -- below both libraries and the
recipe at every shape in the regime. `--check` compares against fp32 at every
shape, fp16 and fp32 outputs.

Result on 7.2.4 (bench_model -p 512 2048 -n 128 -r 4): decode 21.1 ->
**23.3 t/s** (+10%), prefill unchanged (192 / 308), exl3_stack_check PASS,
chat_probe coherent (worst rep4 0.02), profile: the gate is now
`hgemm_narrow_partial_kernel` + `_reduce_kernel` at 19 ms per 1152 calls
(16 us each) against 134 ms before. Under the profiler the wall time does not
move (host-bound there); a kernel-time win must be confirmed with the
unprofiled bench.
