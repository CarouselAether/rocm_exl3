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

> **2026-09-27 update:** the conclusion below held for the core as it was
> (~97 VALU cycles per tile). Once the decode itself was cut to ~40 cycles
> per tile, batching the loads of U = 2-4 k-tiles (no rotation, no
> prologue/epilogue, in-order decode) became a large win at K = 1-4 --
> see "GEMV tiles core" at the end of this file. The rotation form measured
> here is still not what to redo.

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

### v1.5.0 -> v1.5.3

133 upstream commits; conflicts in `setup.py` and `triton_paged.py` only. Seventeen siblings
touched (table in "v1.5.3 sync" at the end of this file): three regenerated by sed, two new
generated siblings (`rocm_tools/gen_det_siblings.py`, method 2), two three-way merges
(`routing_rdna.hip`, `hc_mix_rdna.hip`), and the half-integer bitrate threaded by hand through
the GEMM kernel map / inner / host and the MoE kernel. The expensive surprises were not in the
kernels: the autotune hash (a new key re-tunes, and a re-tune changes bit-identity), and three
upstream numerics changes that each look like a porting bug until switched off.

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
| `gemm_check.hip`, `gemv_check.hip` | GEMM / GEMV kernels against a CPU reference; gemv_check runs all three dot cores (direct, LDS, tiles; build: `SRC=gemv_check rocm_tools/build_gemv_tiles_bench.sh`), and `GEMV_SWEEP=1` adds the DRAM-resident bandwidth sweep (build with the FLAGS block from `build_coop_check.sh` minus the torch libs, single TU) |
| `mgemv_check.py` | mgemv against the cooperative kernel on real weights: packing, grouped reduce, every routing config |
| `gemv_tiles_bench.hip` (+ `build_gemv_tiles_bench.sh`) | standalone (no torch) direct core vs tiles core at every (U, T): bit-for-bit comparison and DRAM-resident us / GB/s on DS4/Qwen/Gemma shapes |
| `bench_gemv_kernels.py` | the real GEMV dispatches (ext.exl3_gemm / exl3_mgemm) on DS4 decode shapes, `--ab` = EXL3_ROCM_GEMV_TILES 0 vs 1 in one process |
| `gemm_coop_check.hip` | cooperative launch against the same work without it |
| `moe_ref32.py` | fused MoE **and** the per-expert path against an fp32 reference built from dequantized weights |
| `moe_check.py` | fused MoE against the per-expert path (two fp16 implementations — see its own caveats) |
| `moe_inner_bench.hip` (build: `SRC=moe_inner_bench rocm_tools/build_gemv_tiles_bench.sh`) | standalone (no torch) old vs pipelined MoE mainloop on one expert-GEMM shape over 64 DRAM-resident matrices: bit-for-bit comparison and ms / GB/s per row count; `MIB_G`/`MIB_CONC` set blocks per group and groups (40 blocks = two per WGP) |
| `bench_moe_kernel.py` | `ext.exl3_moe` alone on synthetic DS4 / Qwen expert tables (any bits, mul1 or mcg), ms and GB/s per token count; `--check` compares `EXL3_ROCM_MOE_PIPE=0` vs `1` bitwise in one process |
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

## Next: launch-count fusion (brief for the next session, 2026-09-23)

Decode on Laguna is host-bound: ~1720 launches/token, GPU idle ~20% of wall
unprofiled (from 23.3 t/s = 43 ms/token against ~33 ms of kernel time), 36-40%
under Kineto. The restored MoE route issues ~9 launches per MoE layer: three
`exl3_mgemm` calls (gate, up, down) x three kernels each
(`exl3_mgemv_had_in_kernel` -> `exl3_mgemv_dot_kernel_splitk` ->
`exl3_mgemv_had_out_kernel`, all in `rocm/quant/exl3_mgemv_rdna.hip`; the
single-warp and graph-captured variants have the same shape). The 2026-08
dispatch-gap census measured the CP's ~2.1 us gap between dependent kernels
and found graph replay does not remove it, so the lever is fewer kernels, on
every runtime, and it matters more on a discrete card where kernels are
shorter and the gap and host cost are the same.

Plan, in order of return per risk:
1. Fold `had_in` into the dot kernel's prologue. A is m x K halfs (tiny); each
   warp can rotate the 128-blocks of its own K-segment on the fly (128-point
   Hadamard + suh scale), redundantly across N-tiles. Cost is trivial; saves
   one launch per call. Keep bit-identity with the current path (the same
   arithmetic order) so mgemv_check's bitwise cases still pass.
2. Fold `had_out` into the dot kernel's epilogue with a last-arriving-block
   pattern per N-tile: split-K partials as today, an atomic counter per
   N-tile, and the block that observes count == splits applies svh + the
   output Hadamard + routing weights and writes C. The counter array can live
   in the DevCtx workspace or the existing `Exl3MgemvParams` block; it must be
   zeroed by the last block itself (self-resetting) so graph replay needs no
   patch. Saves the third launch.
3. Gate + up in one call: same A, same indices, two B pointer tables; one
   launch with a z-dimension of 2 (or a matrix-list of 2*top_k entries)
   writing interm_g and interm_u. With 1 and 2 done this takes a MoE layer
   from 9 launches to 2 (gate+up, down).
4. Then the same for `exl3_mgemv` on the dense/attention side (q/k/v,
   gate/up bundles) -- it is the same kernel family.

Acceptance: `rocm_tools/gemv_check` and `mgemv_check.py` full PASS incl. the
masked-2tok bitwise case; `exl3_stack_check` PASS; `moe_ref32`-style route vs
fp32 at bsz 1/3/8; `chat_probe.py` clean; `bench_model.py` decode up with
prefill flat; `profile_decode.py` showing the launch count per token drop.
Noise floor ~5%: repeat runs. Do it on `main` (7.2.4), merge to `rocm-10`,
rebuild `.venv10`.

## Launch-count fusion: landed, and what it showed (2026-09-23)

Steps 1, 2 and 4 of the brief above are done on `main`; step 3 (gate+up in one
call) is deliberately not, see below. Every fused kernel is bit-identical to
the form it replaced, by two new gates: `rocm_tools/mgemv_bitwise.py` (the
multi-matrix path, every routing configuration of mgemv_check, saved on the
old build and compared on the new) and `rocm_tools/decode_bitwise.py` (a
48-step greedy decode on Laguna, every step's logits, which is the only way to
reach the single-matrix graph path inside the BC modules). Both PASS, the
decode one also with `EXL3_GEMV_FUSE_OUT=0`; mgemv_check and exl3_stack_check
PASS.

What changed (`rocm/quant/exl3_gemv_kernel_rdna.hip.h` holds the shared
helpers under "Launch-count fusion"):

- **Multi-matrix (`exl3_mgemv_rdna.hip`)**: the had_in kernel is gone -- each
  dot block rotates its expert's input into LDS (block-cooperative, one
  `__syncthreads`) and the dot core reads A from LDS; the had_out and reduce
  kernels are gone -- the last-arriving warp of each 128-wide output segment
  (an atomic counter per (slot, segment), self-resetting, in the parameter
  block) rotates the segment in place, and with routing weights the last
  rotated slot of a segment runs the grouped reduce for its 128 columns. One
  launch per mgemm call instead of three or four; the dot kernel hosts the four
  graph patch sites, recorded against the launched instantiation.
- **Single-matrix graph path (`exl3_gemv_rdna.hip`)**: same two folds; one
  launch instead of three; six patch sites on the dot kernel. The plain
  (non-graph) path is untouched: it also serves calls without su/sv and runs
  once per token (lm_head).
- Kill switch `EXL3_GEMV_FUSE_OUT=0` restores the separate output kernels on
  both paths (the input fold has no switch; its arithmetic is the same
  function). A `(void) A_had` marks the scratch the fused paths no longer touch.

Measured (Laguna-S-2.1 4bpw, gfx1151, 7.2.4; bench_model -p 512 2048 -n 128
-r 4; profile_decode -n 32; every figure repeated):

| | decode t/s | launches/token | profiler wall (32 tok) | GPU busy (Kineto) | kernel time |
|---|---|---|---|---|---|
| baseline 47df37d | 23.3 | 1961 | 1.768 s | 60.0% | 1.061 s |
| + step 1 (had_in) | 23.2 | -- | 1.684 s | 64.2% | 1.080 s |
| + step 2 (had_out+reduce) | 23.2 | -- | 1.639 s | 66.0% | 1.082 s |
| + step 4 (dense graph path) | **23.5** (x2) | **1152** | 1.581 s | 68.6% | 1.085 s |

Prefill flat throughout (191-192 / 312-316 t/s). Launches per token fell 41%;
the profiler's wall fell 10.6%; unprofiled decode moved +0.9%, inside the noise
floor but the same direction on every run. `gap_profile.py` on the fused
build: per token 33.58 ms busy, 3.41 ms of launch-shaped gaps (1239 gaps,
median 2.7 us -- the CP gap the census measured, now on 41% fewer kernels),
0.32 ms of big gaps.

**The diagnosis in the brief was wrong, and this is the useful result.**
Decode was called host-bound from Kineto profiles (GPU busy 60% of wall, 880
ms of `hipLaunchKernel` CPU time for 58k launches = 15 us each). Measured
without the profiler:

- `exl3_mgemm` costs the host **3.3 us** per call; a whole MoE layer (gate,
  up, act, down, copy: 5 calls) costs **15 us** of host time against **305
  us** of GPU time. Fusion took the layer from 21 to 15 us host.
- Instrumenting `Model.forward` in the generator loop (1-token prompt, 64
  tokens): the host enqueues a full forward in **4.15 ms**, then waits **31.4
  ms** for the GPU; with a sync at forward exit the forward is **35.5 ms** and
  the pure host segment between tokens (sampler, generator, next forward's
  Python up to its first kernel) is **0.13 ms**.

So unprofiled decode is **GPU-bound**, with ~31 ms of host slack per token.
Kineto inflates per-launch host cost roughly tenfold, which is what made the
GPU look idle. What fusion actually buys is device-side: fewer CP gaps (~2
ms/token less, by the profiler's count) against +0.7 ms/token of kernel time
(the rotation prologue is redundant across the N-tile blocks of one expert,
+9% on the biggest kernel; the fused reduce tail is +8% on the down kernel
after its row loop was reordered for load overlap). Net ~+1 ms/token, which
is the +0.9% seen. On rocm-10 with graphs on, the launch count is the thing
graph replay does not fix (census), so the 41% cut carries over there intact.

**Next, in order of return:**

1. Remove the rotation redundancy: TILES_PER_BLOCK=2 (or 4) N-tiles per
   split-K block sharing one rotation, keeping split depth so the wave count
   is unchanged (block = tiles x splits warps). Claws back most of the +0.7
   ms/token; then fusion is a clear GPU-side win. Contained in the two dot
   kernels + the smem carve; measure with bench_mgemm / gemv_check sweep.
2. Step 3 (gate+up in one call) is deprioritized: it saves 47 launches/token
   (~0.1 ms of CP gap) and needs a doubled index list no existing argument
   carries -- a new ROCm-only binding (bindings.cpp deviation) or a mode flag
   smuggled through an int argument. Not worth it at this return. Same for
   folding the activation into the down prologue (47 tiny kernels/token).
3. The GPU-side levers are now the kernels themselves: the expert GEMVs run
   at ~141 GB/s effective (2.2 GB of expert weight per token in 15.6 ms)
   against the ~226 GB/s the single-matrix sweep reaches -- occupancy/tail
   effects of 64-tile grids at top-10, not launch overhead.

Measurement discipline learned: **never diagnose host-vs-GPU from a Kineto
profile alone**. Time the host segment directly (forward entry/exit with and
without a sync, as above) before attributing idle GPU to the host.

### Fused build on DS4 and Qwen 3.8, plain and with MTP (2026-09-23)

`rocm_tools/bench_mtp.py` (new: one load, plain + `-ndt 3 2 1` with the
model's MTP head, tok/s + acceptance + text) and bench_model, fused build vs
`EXL3_GEMV_FUSE_OUT=0`. Coherent text on every path of both models.

| model | decode128 fused | FUSE_OUT=0 | prefill512 | MTP ndt=1 | ndt=2 | ndt=3 |
|---|---|---|---|---|---|---|
| DS4-Flash 2.04bpw (draft route: dflash) | **16.2** | 15.8 | 104 | 9.9 (acc 81%) | 12.1 (80%) | 10.7 (75%) |
| Qwen3.8-Flash-Next 4bpw (route: mtp_draft) | **23.9** | 23.3 | 226 | 15.7 (83%) | 20.0 (78%) | 18.4 (64%) |

(MTP columns: greedy 256 tokens on natural prompts, median of 3, tok/s.)
The output fusion is worth +2.5% on both; prefill flat. DS4's 16.2 is below
the 17.9 recorded 2026-08-28 (pre-1.5.0 sync) -- not this change (fused >
unfused); the 1.5.0 sync / route changes are the suspect, untracked.

**MTP is a net loss on both models, and the reason is the m > 1 verify
path.** Tokens per verify step = 1 + ndt x acceptance: Qwen ndt=2 gets 2.56
tokens per step at 20.0 t/s = 128 ms/step against 42 ms for an m=1 step, so
an m=3 forward costs **3.05x** an m=1 forward; DS4 ndt=2: 215 vs 62 ms,
**3.5x**. Memory-bound it would be ~1.1-1.3x (the weights are read once
either way), which would put ndt=2 near 2x plain. At m > 1 the dense linears
leave the GEMV for the cooperative GEMM and the MoE takes the bszN route,
neither tuned for 2-4 rows. This is the case for the multi-row GEMV
(m = 2..8, row tile {1,2,4,8}) plan: it is what makes the MTP head pay.
ndt=2 is the best draft length on both models today; 3 loses acceptance, 1
loses too much per step.

### Where the m = 3 verify step goes (profile_decode -ndt 2, 2026-09-23)

`profile_decode.py -ndt N` now profiles MTP decode (the model's own MTP head,
verify rows m = N + 1). GPU ms per plain token vs per m=3 verify step, by
kernel class (48 tokens, 1-token prompt, Kineto kernel times):

| | Qwen3.8 plain | Qwen3.8 m=3 step | ratio | DS4 plain | DS4 m=3 step | ratio |
|---|---|---|---|---|---|---|
| dense linears (graph GEMV at m=1 -> cooperative GEMM/mgemm at m=3) | 12 | 47 | **4x** | 16 | 72 | **4.4x** |
| MoE experts (per-token mgemv loop) | 9.7 | 29 | 3x | 28 | 45 | 1.6x |
| attention / GDN | 9.6 | 16 | 1.7x | 11.4 | 20.6 | 1.8x |
| whole step | 40 | 110 | 2.75x | 61 | 156 | 2.6x |

The dense linears are the lever: at m=3 they leave the GEMV for the
cooperative tile GEMM (284 us per call on Qwen's 5-bit shapes vs 78 us for
the m=1 GEMV; DS4's lm_head 9.4 ms per call vs 1.9) and become the largest
class on both models. A multi-row GEMV that reads the weights once for
m <= 8 rows would take them to ~1.2x the m=1 cost: the m=3 step falls from
110 to ~75 ms on Qwen and 156 to ~105 on DS4, which at ndt=2's 2.5-2.6
tokens per step is ~27 t/s (vs 23.9 plain) and ~25 t/s (vs 16.2). The MoE
class is 3 tokens' worth of distinct experts (top-10 of 256-512: little
overlap) -- inherent, though the three per-token launches could be one
batched mgemm call (num_tokens > 1 without packing) for occupancy. The DSA
attention and GDN scale sub-linearly already.

Design constraint for the multi-row form: the fused LDS rotation prologue
does not extend to m rows (8 x 12288 halves is 196 KB, and the redundant
rotation would be m x the +9%). At m > 1 the input rotation should go back
to its own kernel writing A_had (launch cost is irrelevant there: the step
is GPU-bound at 100+ ms), with the dot kernels reading A rows from global
and the fused output epilogue looping over rows. Row tile {2, 4, 8} as a
template parameter, m == 1 untouched (bit-identity gate stays).

## Multi-row GEMV, m = 2..8: landed (2026-09-23)

`rocm/quant/exl3_gemv_multirow_rdna.hip` (+ `.hip.h`), routed from
`exl3_gemm_gr` and `exl3_mgemm_gr` for 2 <= m <= `EXL3_GEMV_MAX_M` (default
8; 1 switches it off) before the cooperative kernels, in and out of capture.
Design as decided from the verify-step profile above: the pre-fusion
three-kernel structure -- a rotation kernel writing the m rows of every slab
to A_had (and hosting the graph patch sites, republished through a per-device
block), then a split-K dot kernel with a row tile M in {2, 4, 8} (smallest
>= m) that dequantizes each B tile once for M pairs of fdot2 accumulators,
with the fused last-arriving-warp output epilogue rotating the m row segments.
Row r of an m-row call is bit-identical to a separate m == 1 call on that row
(same chain, same split-K order since the wave rule is the m == 1 rule):
`rocm_tools/multirow_check.py`, synthetic trellises, 123 cases over both
entry points (broadcast / per-slot inputs, indices, expert-range packing,
per-matrix width/output lists, lm_head-scale widths, both C dtypes), all
PASS; each case's error against the cooperative kernel equals the m == 1
path's own. The multi-matrix form declines routing weights (the weighted
reduce is the per-token MoE route's, always m == 1), sliced mode and
multi-token calls. exl3_stack_check PASS (m 2..16), decode_bitwise PASS
(m == 1 untouched), Laguna m == 1 decode unchanged at 23.5.

Measured (bench_mtp.py, greedy 256 tokens, natural prompts, median of 3):

| model | plain | MTP ndt=1 | ndt=2 | ndt=3 | ndt=2 before |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next 4bpw | 23.8 | 30.0 (acc 82%) | **32.1** (73%) | 32.1 (66%) | 20.0 |
| DS4-Flash 2.04bpw | 16.2 | 17.6 (84%) | **19.5** (79%) | 19.4 (76%) | 12.1 |

MTP is now a win on both: +35% on Qwen, +20% on DS4, coherent text on every
setting; ndt=2 remains the best draft length. Qwen's m=3 step went from
110 to 73 ms of GPU (profile_decode -ndt 2): the dense linears from 47 to
~13 ms/step -- the new kernel runs 71 us per call on the 5-bit shapes
against the cooperative GEMM's 284 us. What remains of the step is the
per-token MoE loop (24 ms), the GDN kernels (16 ms) and a rocBLAS hgemm
at m=3 (4.6 ms/step, 100 us per call -- the fp16 projections that take
gdn_ba_gemv at m == 1; a narrow-N split-K case like the attention gate's).

**Finding for the m == 1 path:** at m=3 the multi-row kernel's 71 us per call
is *below* the fused m == 1 kernel's 78 us on the same shapes, with three
times the rows. The multi-row form reads A from L2 and carries no rotated
row in LDS (8 KB of smem vs 13-31 KB), so it runs more blocks per CU; the
fused prologue's LDS footprint and redundant rotation cost more than the
two launches it saves. Next: instantiate a row tile of 1 and route m == 1
through the multi-row structure under a switch, bench Laguna/Qwen/DS4; if
it wins, the LDS rotation prologue becomes the K <= 3072-only form or goes.

DS4's m=3 step went from 156 to 115 ms of GPU: dense linears 72 -> ~20 ms
(the multi-matrix form covers its bundle/fan sites too), and what is left is
the per-token MoE loop at 49 ms (43% of the step: three tokens' 2-bit expert
GEMVs launched one token at a time, 64-tile grids each) and DSA attention at
23 ms. The MoE loop is the next lever on DS4: one batched mgemm call per
layer (num_tokens = m, no packing -- the mgemv path handles that form) so
the three tokens' slots share one grid; the bytes stay 3x but the
occupancy of narrow 2-bit expert GEMVs does not.

Longer drafts (same tool, ndt 7 / 5 / 3 -> verify rows m = 8 / 6 / 4):

| model | ndt=3 (m=4) | ndt=5 (m=6) | ndt=7 (m=8) |
|---|---|---|---|
| Qwen3.8 (mtp_draft) | 32.1 (acc 66%) | 28.5 (50%) | 17.9 (40%) |
| DS4 (dflash) | 19.4 (76%) | 19.9 (76%) | 19.9 (76%) |

Qwen's one-layer MTP head loses acceptance fast past three drafts, so the
m = 8 verify does more work per accepted token and throughput falls; 2-3
drafts is its range. DS4's drafter holds 76% at every length, so tokens per
step grow linearly (6.3 at ndt=7) -- and so does the step cost (317 ms at
m=8 vs 170 at m=4), because the per-token MoE loop is linear in rows: the
verify step is now MoE-loop-bound on DS4, and the flat 19.9 says the
multi-row dense path is no longer what limits it. Coherent at every setting.
The m = 8 row tile is exercised and correct (multirow_check covers m=8
directly; the ndt=7 text is clean).

### Row tile 1: m == 1 through the multi-row structure (2026-09-23)

The A/B the 71-vs-78 us observation asked for. `EXL3_GEMV_MR_M1` routes
m == 1 through exl3_gemv_multirow_rdna.hip with a row tile of 1 (rotation
kernel to A_had, dot kernel reading its row from L2, 8 KB of smem) instead
of the fused m == 1 kernels (rotation in LDS: 13-31 KB of smem). Bit-identical
logits either way (decode_bitwise PASS with the switch on; lm_head-scale
widths stay on the single-warp m == 1 form so the two routes split K
identically). Measured, bench_model -p 512 -n 128 -r 4:

| | fused m == 1 (LDS prologue) | row tile 1 (A from L2) |
|---|---|---|
| Laguna 4bpw | 23.5 | 23.6 |
| Qwen3.8 4bpw | 23.8 | **24.4** |
| DS4 2.04bpw | 16.2 | **17.0** |

Laguna profile: gate/up kernel 343 -> 319 ms per 32 tokens, the dense
kernels -8%, total kernel time 1.085 -> 1.063 s -- back to the pre-fusion
figure, with one launch more per call than the fused form. So the LDS
rotation prologue cost more occupancy than its launch saved; **row tile 1
is now the default** (`EXL3_GEMV_MR_M1=0` restores the fused form). What
stays on the fused kernels: the weighted-reduce MoE down projection (the
multi-row path declines weights) and lm_head. Two follow-ups fall out:
give the multi-row epilogue the weighted reduce so the down projection can
move too, and drop the LDS prologue from the m == 1 kernels once nothing
routes there but those two cases. The launch-count fusion's lasting parts
are the fused output epilogue (used by both structures) and the counters.

### Laguna with its DFlash drafter (2026-09-23)

`bench_mtp.py -dm ~/models/Laguna-S-2.1-DFlash` (route dflash, drafter's
default_draft_size 15), multi-row build: net loss. Plain 24.7 t/s greedy;
ndt 1/2/3/5/7 = 19.4/19.3/17.7/14.4/11.5 at acceptance 72/52/45/34/23% on
the natural prompts (76% on profile_decode's 1-token prompt, so acceptance
is prompt-sensitive rather than broken). profile_decode -dm -ndt 2: 99 ms
of GPU per verify step, of which the per-token MoE loop is 40 ms (three
tokens' gate/up/down, one token at a time), the multi-row dense verify 13
ms -- and the **drafter itself 28 ms**: it is an unquantized BF16 model
whose m == 1 linears run through torch.mm / rocBLAS, and rocBLAS handles
them as skinny GEMMs (`Cijk_..._HSS_BH_MT128x32x16` at 557 us per call,
`aten::mm` at 216 us) -- the same one-workgroup-walks-K pathology the
attention gate had (RDNA_NOTES "Narrow-N hgemm"), which the fp16 split-K
kernels only cover for N <= 256. Extending those to any N at m <= 8 (a plain
fp16 GEMV, trivially bandwidth-bound) would take the drafter to a few ms per
step; with the MoE loop batched as well the step would be ~50 ms at 2.5
tokens, i.e. the DFlash route would pay. Both items are on the list; neither
is a kernel-numerics problem.

### ROCm 10 (HIP 7.15) with the fusion + multi-row work, graphs on (2026-09-23)

`rocm-10` = main dd7a670 merged (c77c47d), `.venv10` rebuilt with the SDK
recipe above. Everything below ran env-scrubbed on the wheel runtime with
the runtime gate leaving graphs ON (rocm_py reports "HIP graphs ON: HIP
runtime 7.15.26333"); `EXL3_ROCM_HIP_GRAPHS=0` for the off runs.

- Correctness: exl3_stack_check PASS, multirow_check 123/123 PASS,
  mgemv_check PASS.
- **Graph capture of the new kernels**: decode_bitwise saved with graphs
  OFF and compared with graphs ON -- all 48 steps bit-identical. The dot
  kernels' patch sites (recorded per template instantiation) and the
  multi-row rotation kernels' sites are patched correctly on replay; the
  self-resetting epilogue counters need no per-replay memset, as designed.
- chat_probe (six turns + concurrent jobs, graphs on): worst rep4 0.03, clean.
- Speed, bench_model / bench_mtp (7.2.4 figures from the same day in
  parentheses): Laguna decode 23.7 graphs on / 23.9 off, prefill 195
  (23.6 / 192); Qwen3.8 plain 24.2, MTP ndt=2 32.3 (24.4 / 32.1); DS4
  decode **18.1**, prefill 112 (17.0 / 103) -- DS4 gains on ROCm 10 because
  its fp16 torch matmuls take hipBLASLt there (the narrow-N finding above).
  Graphs on vs off remains a wash on the device timeline, as before.

## WMMA GEMM backend for hgemm (2026-09-27): DS4 pp2048 174 -> 216 t/s

**What.** `hgemm` / `hgemm_recon` (and so `exl3.py`'s reconstruct-then-GEMM prefill
path, `fp16.py`'s mixed-dtype linear, and `blocksparse_mlp.cpp`) now run m > 8 on the
MIT-licensed rocm_wmma_gemm kernels (Adel Johar, commit ea3aa74) instead of hipBLAS
wherever that measured faster. hipBLAS's fp32-output GEMM on gfx1151 is a VALU kernel
with no matrix cores (`Cijk_..._HSS_..._MT64x32x8`, PROFILE.md section 6): 3-6x slower
than its own fp16-output path, and DS4 needs fp32 output for its fp32 residual stream.
The fp32 output stays; only the kernel changes.

**Files** (all ROCm-only; `hgemm_rdna.hip` gains one call after the narrow-N GEMV):
- `vendor/rocm_wmma_gemm/` -- LICENSE plus the six kernel headers. Two marked
  (`EXL3 MOD`) edits: `kernel.hpp` takes a separate output type (fp32 accumulate,
  fp16 or fp32 store), and `fragment.hpp` bit-casts `__half` where this build's
  `__HIP_NO_HALF_CONVERSIONS__` removes the implicit conversion. **Trap** found on the
  way: `__builtin_bit_cast` applied directly to an ext-vector *element* lvalue reads
  element 0 (every row of a 16x16 tile came back as rows 0/1); copy the element out
  first.
- `wmma_gemm_rdna.hip.h` / `wmma_gemm_rdna.hip` -- kernel wrapper, routing, selection.
- `wmma_gemm_table_rdna.hip.h`, `wmma_gemm_inst{0..3}_rdna.hip` -- generated by
  `rocm_tools/gen_wmma_gemm_table.py` from the library's f32-accumulator JSON
  (vendored, layout row/row/row only) plus `wmma_gemm_tuned_gfx1151.json`.
- `rocm_tools/tune_wmma_gemm.py`, `wmma_gemm_check.py`, `bench_wmma_gemm.py`.

**Accumulation.** fp32 for both output dtypes, like hipBLAS's `CUBLAS_COMPUTE_32F`. The
library's fp16-output kernels accumulate in fp16 and are not used.

**Selection.** The runtime `gcnArchName` picks a table: gfx1151 (library + local
measurements), gfx1100 and gfx1101 (gfx1100's library table). gfx1200/1201, gfx1150 and
the rest have none and stay on hipBLAS. The kernel bodies are compiled only in the
device passes of archs whose table uses them. Other passes get empty stubs that are never
launched, which keeps the gfx11 WMMA builtins out of gfx12 code objects. Within a table
the rule is the library's: an exact (M, N, K) hit; otherwise the entries with the
closest K, then the smallest dM^2 + dN^2. Ties go to the larger M (the library takes the
smaller, which measured 0.6x at m = 160). The chosen entry then carries route flags
(fp32 / fp16) saying whether WMMA beat hipBLAS there.

The library's tables hold only M, N >= 1024. Extrapolated to prefill's small-M expert
and chunk-tail GEMMs, their big tiles leave the GPU nearly idle. Example: 160x1024x4096
got a 96x512 tile, 4 workgroups each walking all of K, and ran at 0.76x hipBLAS. So
`tune_wmma_gemm.py` measured every config against hipBLAS on a grid of 520 shapes:
M 16..2048 (13 values), N 256..32768, K 512..8192. The best config was one of 19. fp32
output won at all 520 shapes (1.18x-15x); fp16 won at 433 of them. The
gfx1100 table has no local measurements, so it routes fp32 only, and only at m >= 512
(`min_m`), close to where its tiles were tuned. That is untested on hardware.

**Routing conditions** (anything else falls back to hipBLAS, never errors): arch has
a table; m > 8 (`EXL3_ROCM_WMMA_GEMM_MIN_M`, default 9, the narrow/GEMV regime stays
as it was); C rows packed (`exl3.py`'s column-slice writes into `y_[:, n0:n1]` fall
back); A/B/C 16-byte aligned; N % 8 == 0; K % block_k == 0 for the chosen config; each
matrix < 2 GB. Bounds: rows past M read zeros (buffer loads clamped to the allocation),
the unaligned-epilogue variant bounds-checks every store. A partial K tile would read
the next row of A times zero rows of B -- right for finite A, but an inf/NaN would
leak into a neighbouring row -- hence the K condition. No allocation, no host sync:
graph capture + replay is bitwise equal to eager (checked).

**Env switches.** `EXL3_ROCM_WMMA_GEMM=0` = off, `1` = on (the default: the table's
route flags decide), `2` = every eligible call (tests, tuning).
`EXL3_ROCM_WMMA_GEMM_F16=0` = fp16 output stays on hipBLAS. `EXL3_ROCM_WMMA_GEMM_CFG=<i>`
pins a config (tuning). `EXL3_ROCM_WMMA_GEMM_TRACE=1` logs one stderr line per m > 8
call (routed config or hipblas). All are read per call, so an in-process A/B works.

**-mcumode.** The library builds with `-mcumode` and tunes under it. setup.py uses one
flag set for every TU, so the wrapper kernels carry `__attribute__((target("cumode")))`
instead (device pass only). That measured 3-9% faster at large shapes (1792x4096x2048
fp32: 1.03 -> 0.94 ms).

**Numerics.** Not bit-identical to hipBLAS in fp32 output. The RDNA3 WMMA f32
accumulator rounds each 16-deep step toward zero. Measured with all-positive inputs:
relative bias -1e-5 at K = 4096, growing about linearly in K. hipBLAS's VALU FMA
kernel's error grows about as sqrt(K). The result is fp32 max|err| 3-17x hipBLAS's
(e.g. 7e-5 vs 1e-5 at 1792x4096x8192, outputs of O(1)), which is still <= 2% of the
fp32 gamma_K bound, K * 2^-24 * (|A||B|), at every tested shape. hipBLAS's own
fp16-output kernels use the same instruction. The WMMA fp16 output was bitwise
identical to hipBLAS's fp16 output on 5 of 7 shapes tested, and 96-97% of elements on
the other two. It is always exactly the round-to-nearest of the WMMA fp32 output.
Perplexity (wikitext2, 100 x 2048, `bench/run_ppl.sh`): DS4 6.461140 vs baseline 6.461586
(-0.007%), Gemma 4 31B 18.633857 vs 18.635452 (-0.009%).

**Measured** (gfx1151, torch 2.15.0.dev20260926+rocm10.0, HIP 7.15):
- `rocm_tools/wmma_gemm_check.py`: 127/127 PASS. That covers the DS4 shapes x M in
  {1..1792}, every compiled config pinned on partial and exact tiles, M 1-17 forced
  through the kernel, odd N/K fallbacks, 3-D A, a misaligned A, strided C with its
  neighbours untouched, and graph replay.
- `rocm_tools/bench_wmma_gemm.py`, 98 shape x dtype points, M 9..1792: fp32 total
  110.9 -> 23.3 ms (4.8x, never slower). Examples: 1792x4096x8192 19.8 -> 3.6 ms,
  1792x4096x2048 4.43 -> 0.96, 255x4096x8192 3.00 -> 0.52, 160x4096x2048 0.41 -> 0.10.
  fp16 total 27.5 -> 21.2 ms (1.29x). Worst point 1792x512x4096 at 0.97x (fp16 route
  flag set by a 3% margin at the neighbouring grid point); the rest are >= 0.98x.
- DS4 `bench/run_bench.py --pp 512 2048 --tg 128`, same build, on vs
  `EXL3_ROCM_WMMA_GEMM=0`: pp512 **123.1** vs 112.6 (+9%), pp2048 **216.5** vs 174.2
  (+24%), tg128 17.98 vs 17.96 (decode is m = 1, untouched). PROFILE.md baseline:
  112.8 / 178.5 / 18.02.
- `bench/run_gates.sh`: PASS (879 passed, 9 skipped).
- `hipcc_probe.sh --all`: 123/123 for gfx1100 and gfx1201.

**Retune** after a toolchain bump or on a new table arch:
`gen_wmma_gemm_table.py --tune`, rebuild, `tune_wmma_gemm.py` (~45 min; it extends an
existing JSON, so a re-run with new `-m/-n/-k` measures only those), then
`gen_wmma_gemm_table.py` again and rebuild. `gen_wmma_gemm_table.py --check` tells you
whether the generated files are stale.

## DSA decode MQA kernel (2026-09-27): DS4 tg128 18.07 -> 21.52 t/s

`_dsa_attn_split_kernel` (DeepSeek-V4 decode sparse attention, split phase) cost
160-225 us per call at every context: 43 calls/token, 16% of decode (PROFILE.md §4).
The 2026-08-28 retune (`EXL3_ROCM_DSA_TUNE`, BLOCK_H 8 / 8 warps) cut the spills enough
to kill the per-layer scratch stall, but the kernel itself stayed slow.

**Root cause.** Two things drive the register pressure:
- A loop-invariant q tile is hoisted and held as the WMMA A operand. RDNA3 WMMA
  replicates A/B across the two half-waves, so a 16 x 512 A panel is 256 VGPRs per warp
  on its own.
- Every program also carries a BLOCK_H x 576 fp32 accumulator (D_c padded to 512, plus
  64 rope columns).

The result is 256 VGPRs, ~700 spilled, and 2.2 KB scratch per lane in every tuning
(sweep below). Each split sees only ~8-40 keys, so the call is all fixed cost.
Window-only layers, which have the FEWEST keys, were the slowest at 217 us. WMMA was
emitted (205 v_wmma), so "no WMMA" was not the problem.

**Fix.** `rocm_py/dsa_decode_rdna.py:_dsa_decode_mqa_kernel` is a drop-in split kernel
(same arguments, same workspace layout, same combine, unchanged C++ launch):
- One program = HP heads (32) x one BD-wide output column block (256) x one key split.
- Scores are the full 512-wide q.K, reduced in a RUNTIME loop of KC = 64 chunks with q
  re-read per chunk (L1/L2 hits). A static unroll or an invariant q brings the spills
  back. Scores are recomputed per column block, which is cheap at decode key counts.
- Accumulator is 32 x 256 fp32 = 64 VGPRs over 128 lanes.
- The virtual key row is [c | r] = 512 wide: ring/chunk rows are contiguous; pool rows
  are pool_c ++ pool_r.
- Packed pools (QC) use a column-range variant of the plane loader, in the same H32
  domain as upstream.
- Grid: the C++ launches (rows x H/BLOCK_H, n_splits); the kernel reads pid % (H/BLOCK_H)
  as (head group, column block). m / l are written by column block 0; the other column
  blocks compute identical values.

**Tuning (production).** bc_dsa.BLOCK_H 16 (the combine's head block), HP 32 -> BD 256,
BLOCK_N = BLOCK_W 32, KC 64, 4 warps, stages 1, N_SPLITS 8 (single value for ctx 512 and
16K; BCDsaBatch hardcodes 8). Tool: `rocm_tools/bench_dsa_decode.py` (`--sweep-old`,
`--sweep-new` with `EXL3_DSA_SWEEP=tile|splits`). Measured with CUDA-graph replay, us per
split + combine, DS4 shapes.

Old kernel sweep (csa ctx 512 seq 1; baseline H8 w8 s2 ns16 = 169 us):

| knob | values -> us |
|---|---|
| n_splits (H8 w8 s2) | 1: 160, 2: 137, 4: 105, 8: 123, 16: 170; at 16K top-k 255 / 194 / 174 / 160 / 221 |
| BLOCK_H x warps, s1 | H8: w4 115, w8 181, w16 374; H16: w4 80, w8 96; H32: w4 70, w8 70; H64: w8 74 |
| stages 2 | same or worse everywhere (H32 w4 s2 192) |
| BLOCK_N / BLOCK_W | N16W16 153, N32W16 166, N32W32 295; N64 exceeds 64 KB LDS |

Every old variant spills (555-3219 spills). The best is ~70 us, so tuning alone cannot
reach 40 us.

New kernel sweep (4 splits unless noted, columns = csa512 | csa16K-topk | csa512 seq3 |
csa512 qc4):
- BLOCK_N 32 beats 16 everywhere (16: 26-43 us at 512).
- KC 128 spills at 4 warps, and KC 32 is slower.
- 8 warps beat 4 only for the QC column.
- Best tiles:
  - H16/HP32/BD256/w4: 18.1 | 39.1 | 28.8 | 42.2
  - H8/HP32/BD128/w4: 20.1 | 38.6 | 34.9 | 34.0
  - H16/HP64/BD128/w4: 20.1 | 40.3 | 32.1 | 40.3
- Splits (H16/HP32/BD256/w4): 2: 28.7, 4: 18.1, 8: **15.5**, 16: 23.4 at 512; 16K top-k
  77 / 40 / **34** / 37. 16 splits is slower even for window-only layers (each program is
  latency-bound, and the combine grows).
- KSTAGES 2 (software pipelining of the KC loop) is slower everywhere except QC: rejected.

**Before / after** (us per split + combine call; old = retuned upstream kernel, H8 w8 ns16):

| case | old | new | new spills / scratch B |
|---|---|---|---|
| csa ctx512 seq1 | 169 | **15.5** | 0 / 0 |
| csa ctx512 seq1 qc4 | 159 | 30.6 | 169 / 616 |
| csa ctx512 seq3 (MTP verify) | 451 | 28.4 | 0 / 0 |
| csa ctx512 seq3 qc4 | 383 | 49.2 | 169 / 616 |
| hca ctx512 seq1 / seq3 | 212 / 575 | 20.2 / 33.5 | 0 / 0 |
| win ctx512 seq1 / seq3 | 217 / 592 | 15.4 / 28.3 | 0 / 0 |
| csa-topk ctx16K seq1 | 218 | **33.0** | 10 / 32 |
| csa-topk ctx16K seq1 qc4 | 218 | 64.8 | 177 / 624 |
| csa-topk ctx16K seq3 / seq3 qc4 | 649 / 609 | 57.9 / 95.3 | |
| hca ctx16K seq1 / seq3 | 170 / 455 | 15.3 / 28.5 | 0 / 0 |
| MULTIROW csa512 B2 / topk16K seq3 qc4 B2 / hca16K seq2 B3 | 244 / 836 / 614 | 33.6 / 172 / 59.4 | |

Error vs the fp64 reference is unchanged (4.7e-4 vs 5.0e-4 at csa512; QC 7e-4..1.4e-3 on
both). Old: 256 VGPR, 709 spills, 2256 B scratch. New (fp16 pool): 256 VGPR allocated,
0 spills, 0 scratch.

**End to end** (DS4 2.04bpw, `bench/run_bench.py`, same build, `EXL3_ROCM_DSA_DECODE=0`
vs default):

| | off | on |
|---|---|---|
| tg128 | 18.07 | **21.52** (+19%) |
| tg64@16384 | 17.39 | **20.46** (+18%) |
| pp512 | 122.9 | 121.2 (untouched path; noise) |
| MTP ndt 2 | 20.73 (acc 77%) | **23.82** (acc 81%) |

MTP spread is 15-18% in both runs (per-prompt variance). Saved ~8.8 ms/token against
43 x (207 - 19) us = 8.1 ms predicted.

**Numerics / validation.**
- `test_dsa_kernels.py` ALL PASS. The eager `dsa_attn` split path goes through a launch
  proxy, so the test's `run(8)` exercises the new kernel on every H64/D512 case; H8,
  H16/D288 and H128/D576 fall back to upstream.
- PPL is unaffected by construction: prefill uses the one-shot `_dsa_attn_kernel`.
- Decode A/B (`rocm_tools/decode_agree.py`, 3 prompts x 256 greedy tokens, the third a
  3K-token prompt reaching the top-k regime; old run twice = bit-exact control):

  | prompt | prefix match | KL max | top-10 max |
  |---|---|---|---|
  | 0 | 37/256 (diverges at a near-tie, top-2 gap 0.109) | 5.0e-2 | 1.9 |
  | 1 | 169/256 (near-tie, top-2 gap 0.031) | 6.4e-3 | 0.5 |
  | 2 | 256/256 | 1.8e-2 | 1.5 |

  Noise floor: the UPSTREAM kernel with only N_SPLITS 16 -> 8 (a pure reduction-order
  change) gives the same class: top-10 max 2.1 / 0.77 / 1.66, KL max 3.4e-2 / 1.9e-2 /
  4.9e-3, and the same divergence at prompt 1 step 169. `EXL3_ROCM_DSA_TUNE=0` vs 1 is
  bit-identical, so the head-block split does not reorder anything.

  The same A/B on the other two routes:

  | route | prefix match (prompts 0 / 1 / 2) | top-2 gap at divergence | KL max | top-10 max |
  |---|---|---|---|---|
  | `--batch` (3 concurrent jobs: BCDsaBatch MULTIROW, 8 splits) | 256 / 172 / 256 | 0.000 | 3.4e-2 / 8.4e-3 / 1.6e-2 | 3.1 / 0.91 / 1.6 |
  | `--cq 8` (packed pools, QC 8) | 51 / 54 / 256 | 0.125 / 0.000 | 8.1e-2 / 8.9e-3 / 2.5e-2 | 1.5 / 0.44 / 1.7 |
- DS4 PPL (`MODELS=ds4 bench/run_ppl.sh`): 6.461140, identical to the stack.
- `bench/run_gates.sh`: PASS (879 passed, 9 skipped; test_mla_dsa's Q_SPLIT paths take the
  fallback).

**Switches.**
- `EXL3_ROCM_DSA_DECODE=0` restores the retuned upstream kernel.
- `EXL3_ROCM_DSA_DECODE_{SPLITS,BLOCK_H,HP,BLOCK_N,BLOCK_W,KC,KSTAGES,WARPS}` override
  the tuning.
- Declined shapes fall back to upstream: Q_SPLIT / OUT_LATENT (GLM-5.2 DSA-on-MLA,
  bc_mla), non-power-of-two D, and H not tileable.
- Kernels changed for the coherence check: decode split attention in every DS4 layer
  (graphed BCDsa, batched BCDsaBatch, eager dsa_attn split path). Prefill and the
  combine are unchanged.

**Open.**
- QC pools still spill 169-243 VGPRs (616-916 B scratch). That is the H32 rotation of
  q / window chunks in the KC loop; pre-rotating q once would need a workspace.
- CSA top-k at 16K: 10 spills / 32 B.
- seq > 1 rows are separate programs: the MTP verify re-reads the shared keys per row.
- Scratch users re-enter the decode stream only with `-cq`. Watch for the per-layer
  scratch stall (see "DS4 per-layer stall" above) if -cq decode looks slow.

## GEMV tiles core (2026-09-27): DS4 tg128 22.0 -> 27.7 t/s, bit-identical

Branch `opt/gemv-tiles` (PROFILE.md §8 rank 2, CODE_SCAN #3/#4/#18). A new dot core
for every EXL3 GEMV, `rocm/quant/exl3_gemv_tiles_rdna.hip.h`, selected at runtime
(`EXL3_ROCM_GEMV_TILES=0` restores the direct core; `EXL3_GEMV_LDS=1` still pins the
LDS core). Same lane layout, same fdot2 chains in the same order, same split-K
reduction: **every output is bit-identical** to the direct core.

**The diagnosis was half wrong, in a useful way.** PROFILE §5 read "VALU issue per
wave-cycle 0.04" as low VALU use. Per SIMD that is 0.04 x ~14 resident waves ~= 0.55,
and a per-op throughput microbench (instr/SIMD/clk, wave32, gfx1151) showed where it
went:

| op | rate | op | rate |
|---|---|---|---|
| v_add, v_bfe, v_perm, v_alignbit, v_lshl_or | 0.96 | **v_mul_lo_u32, v_mul_hi_u32** | **0.24** (quarter) |
| **v_sad_u8, v_sad_hi_u8, v_msad_u8** | 0.96 | **v_dot4_u32_u8, v_dot8_u32_u4** | **0.49** (half) |
| v_mad_u32_u16, v_pk_mul_lo_u16, v_mul_lo_u16 | 0.93-0.96 | v_mul_u32_u24, v_mad_u32_u24 | 0.96 |
| v_pk_fma_f16, v_dot2_f32_f16 | 0.92-0.95 | | |

The old K=2 loop was 56 VALU instructions = **97 VALU cycles per 16x16 tile per
lane** (8 quarter-rate hash multiplies, 8 half-rate dp4a byte sums, two 64-bit
shifts, ~17 ops of 64-bit per-lane address math, a divergent exec-mask loop); K=4 95
cycles. At the measured 120 us/call that is ~65% of the SIMD's issue -- VALU-bound,
with the rest exposed latency (one 64-byte tile in flight per wave at K=2).

**What the core does (all exact integer arithmetic):**
- Hash multiply `x * C mod 2^32` for a 16-bit x: `r = v_pk_mul_lo_u16 src, [0|C_hi]`
  (op_sel picks x's half; the low lane is x*0) then `v_mad_u32_u16 src, C_lo, r`.
  Two full-rate ops for one quarter-rate op; x may sit in either half of a register,
  so windows at bit 0 / 16 need no extraction and the rest one shift (no mask). The
  cb 0 additive constant rides in the same two ops (`v_pk_mad_u16`).
- Byte sum: `v_sad_u8(P, 0, 0x64006400)` then `v_sad_hi_u8(P1, 0, s)` gives the
  decoded pair already packed as `[0x6400+s0 | 0x6400+s1]` -- replaces two half-rate
  dp4a and the and/lshl_or packing (mul1, cb 2; mcg/3inst keep their lop3+hadd tail).
- fshift at K=1/2/4 is one `v_alignbit` (shift < 32).
- Addressing: tile base, k range, pointers are `readfirstlane`'d, loads are
  `global_load v, vOff, s[base] offset:imm` with a loop-invariant 32-bit lane offset.
  **LLVM trap:** in a loop, LICM hoists the offset's zero-extension to the preheader,
  ISel then sees a bare 64-bit VGPR add and emits 2 VALU of address math per load
  (never the saddr form). An empty `asm volatile("" : "+v"(off))` per iteration keeps
  the zext in the loop; the register is carried unchanged, zero cost.
- U k-tiles' loads issued together, then decoded in k order (U = 2-4 at K <= 3; the
  fdot2 chains are the same sequence as U = 1), and T = 2 adjacent N-tiles per wave
  sharing the A loads (one 2 x 32*K byte contiguous read per k-tile). Per-arch table
  `exl3_tiles_u_splitk` / `exl3_gemv_tiles_tpb`; T = 2 only while the halved grid
  keeps >= 1024 waves (scaled by multiProcessorCount), else T = 1.

**Instruction counts** (hot loop, per tile per lane, from the ISA of
`rocm_tools/gemv_tiles_bench.hip`; VALU cycles weight quarter-rate x4, half-rate x2):

| kernel | before VALU / cycles / VMEM | after VALU / cycles / VMEM |
|---|---|---|
| K=2 mul1 split-K (gate/up) | 56 / 97 / 4 | 39 / 39 / 3 (U2T2: 3 VMEM incl. shared A) |
| K=2 mul1 split-K, A in LDS (down) | 53 / 94 / 2+1 lds | 39.5 / 39.5 / 2+0.5 lds |
| K=4 mul1 split-K | 57 / 95 / 4 | 37 / 37 / 3 |
| K=6 mul1 single-warp (head) | 72 / 134 / 6 | 48 / 72 / 6 |
| K=6 mcg split-K (Gemma) | 85 / 139 / 6 | 52 / 76 / 5 |

(K = 5/6/8 keep the generic dq4 64-bit fshift, 8 quarter-rate shifts per tile.)

**Per-kernel, real dispatch** (`rocm_tools/bench_gemv_kernels.py --ab`: whole call =
rotation + dot + fused epilogue, DRAM-resident, DS4 shapes; before = 3cf11c5 build):

| shape | before us | off us | on us | on GB/s | on G w/s | x vs before |
|---|---|---|---|---|---|---|
| routed gate/up K2 4096->2048 x6 | 121.0 | 115.8 | **74.2** | **169.7** | 679 | 1.63 |
| routed gate/up K2, m=3 | 133.2 | 133.3 | 83.8 | 150.2 | 601 | 1.59 |
| routed down K2 2048->4096 x6, weighted | 132.2 | 126.3 | **82.3** | **152.8** | 611 | 1.61 |
| wo_a K4 4096->1024 x8 | 106.0 | 105.1 | 84.3 | 199.0 | 398 | 1.26 |
| wo_a K4, m=3 | 112.3 | 110.7 | 86.9 | 193.0 | 386 | 1.29 |
| shared gate/up K4 4096->2048 x2 | 59.8 | 59.0 | 46.3 | 181.2 | 362 | 1.29 |
| wo_b K4 8192->4096 | 102.0 | 98.6 | 79.7 | 210.4 | 421 | 1.28 |
| wo_b K4, m=3 | 109.9 | 105.0 | 80.3 | 208.9 | 418 | 1.37 |
| wq_b K4 1024->32768 | 99.7 | 96.7 | 84.0 | 199.8 | 400 | 1.19 |
| wq_b K4, m=3 | 108.8 | 102.5 | 85.8 | 195.6 | 391 | 1.27 |
| shared down K4 2048->4096 (T=1 by the wave floor) | 30.2 | 29.2 | 26.2 | 160.2 | 320 | 1.15 |
| head K6 4096->129280 | 1906 | 1874 | 1746 | 227.5 | 303 | 1.09 |

("off" is the new build with `EXL3_ROCM_GEMV_TILES=0`: the direct core with the
same launch geometry as before; it reads a few % faster than the old build on some
rows, unexplained, within run-to-run noise of the kernel bench.)

The U x T sweep behind the table (`rocm_tools/gemv_tiles_bench.hip`, core only, no
epilogue): T = 2 wins at every K on split-K shapes (K2 gate/up: U1T1 94.8, U2T1
77.6, U4T1 75.1, **U2T2 70.2**, U4T2 73.5 us); K4 prefers U1T2 (wo_b 74.7 vs
U2T1 77.3); K7-8 gain only from the decode (U1). The single-warp lm_head prefers
U4T1. Short-k, small grids (256 tiles x 4 warps) lose with T = 2 -> the wave floor.

**End to end** (bench/run_bench.py, same build, switch on/off; stack before 21.5 /
20.5 / 23.8):

| | tiles on | tiles off | |
|---|---|---|---|
| DS4 tg128 | **27.68** | 22.02 | +25.7% |
| DS4 tg64 @ 16K | **26.07** | 20.92 | +24.6% |
| DS4 MTP ndt=2 (greedy, acc 81%) | **32.01** | 24.72 | +29% (spread 15-17% both: prompt mix) |
| DS4 pp512 | 123.3 | 122.4 | untouched path |
| Qwen3.8 tg128 | 25.41 | 24.44 | +4.0% |
| Gemma-4-31B tg128 | 7.70 | 7.52 | +2.4% (K6 mcg, near its ~9.7 roofline) |

**Correctness.** `rocm_tools/gemv_tiles_bench` 19 shapes x 7 (U,T) modes bit-identical
to the direct core (bits 1-8, cb 0/1/2, M 1/4, A in global and LDS); `gemv_check`
now runs all three cores against the independent reconstruct reference, all pass;
`mgemv_bitwise` DS4 + Qwen PASS against references saved on the 3cf11c5 build;
`decode_bitwise` DS4 / Qwen / Gemma PASS (and DS4 with `EXL3_GEMV_TILES_T=1` and
with `=0`); `multirow_check` PASS; `bench/run_gates.sh` PASS (879 passed).
PPL (wikitext2 100 x 2048): DS4 6.461140 (= stack); Qwen3.8 4.746908 with the switch on
**and** off -- the -0.017% against the Phase 0 4.747702 is the stack's WMMA prefill
GEMM, not this change (PPL runs the prefill GEMMs; the GEMV path is bit-identical).
Portability: `hipcc_probe --all` 123/123 on gfx1151/1100/1101/1200/1201; max dot-kernel VGPR
105 (gfx1151/1100) / 112 (gfx1201) for the M=4 row tile at K <= 3 (12 waves/SIMD;
M = 1 kernels 41-70), no scratch anywhere.

**Tried and not taken / open:**
- mul24 decomposition of the hash multiply (`v_mul_u32_u24` + `v_mad_u32_u24` + a
  shift): 3-4 full-rate ops vs 2 for the pk_mul/mad16 pair. Not built.
- `v_dot4` -> `v_sad_u8` alone would be a 1-cycle-per-weight win; `v_sad_hi_u8`
  also folds the packing, so both were taken together.
- K = 5/6/8 still run the generic dq4 with the 64-bit funnel shift (8 quarter-rate
  ops per tile). A per-lane precomputed `shift >= 32` operand select would make it
  alignbit + 2 cndmask; worth ~2-5% on Gemma (K6) and lm_head, which already sit
  at 200-227 GB/s. Not done.
- Release-only fences for non-last split-K arrivals (CODE_SCAN #10): not measured.
  With T = 2 a warp arrives twice per block; the fence cost is in the epilogue only.
- The weighted down projection is still the LDS-prologue mgemv kernel: 82 us real
  vs 69 us for the same core with A pre-staged -- the per-block rotation and the
  reduce tail. Moving it to the multi-row path with a weighted epilogue (CODE_SCAN
  #6) is the next ~0.4 ms/token.
- The M = 4 row tile at K <= 3 is 99-105 VGPR (12 waves); U = 1 for M = 4 was slower
  in the core sweep (83.5 vs 78.6 us), so the table keeps U = 2.

## Pipelined MoE mainloop (2026-09-27): DS4 pp512 123 -> 276, pp2048 217 -> 371 t/s

Branch `opt/moe-mainloop` (PROFILE.md §8 rank 1, CODE_SCAN #2/#5/#19/#20). The fused
MoE prefill kernel `exl3_moe_kernel` gets its own GEMM mainloop,
`rocm/quant/exl3_moe_inner_rdna.hip.h`, selected at run time (`EXL3_ROCM_MOE_PIPE=0`
restores the shared `exl3_gemm_kernel_inner`, read per call). `exl3_gemm` /
`exl3_mgemm` are untouched: the shared inner is not edited.

**Kernel, synthetic DS4 expert tables** (`rocm_tools/bench_moe_kernel.py`: 256 experts,
4096<->2048, K2 mul1, uniform top-6, fused_rows 128; GB/s = trellis bytes of the
touched experts / time):

| tokens / forward | rows / expert | old ms (GB/s) | new ms (GB/s) | x |
|---|---|---|---|---|
| 256 | ~6 (max 15) | 56.27 (28.5) | **15.70 (102.2)** | 3.58 |
| 512 | ~12 (max 23) | 61.25 (26.3) | **16.89 (95.4)** | 3.63 |
| 1792 | ~42 (max 61) | 154.40 (10.4) | **29.37 (54.8)** | 5.26 |

**Real model, MoE layer** (`bench_moe.py`, DS4, BlockSparseMLP.forward incl. routing,
shared expert, reconstruct tier, gather; 42 layers):

| prompt | old ms/layer | new, cap 128 | new, cap 512 (default) |
|---|---|---|---|
| 256 | 39.96 | **12.07** | |
| 512 | 51.59 | 17.18 | **16.78** |
| 1792 | 110.67 | 52.03 | **44.47** |

**End to end** (`bench/run_bench.py`, same build, switch off / on; results in
`bench/results/moe_mainloop/`):

| | PIPE=0 | PIPE=1, cap 128 | PIPE=1, cap 512 (default) |
|---|---|---|---|
| DS4 pp512 | 122.73 | 275.67 | **281.97** (final rerun 278.43) |
| DS4 pp2048 | 217.18 | 343.83 | **371.15** (final rerun 362.00) |
| DS4 tg128 | 27.71 | 27.73 | 27.78 (27.67) |
| Qwen3.8 pp512 | 271.68 (spread 5.9%) | | **596.14** |
| Qwen3.8 pp2048 | 474.99 | | **740.90** |
| Qwen3.8 tg128 | 25.44 | | 25.46 |

pp2048 profile after (rocprofv3 kernel trace): `exl3_moe_kernel` 40% -> 18.5% of GPU
time; `_dsa_attn_kernel` (32.8%) is now the top prefill kernel.

**What the loop does** (details in the header of `exl3_moe_inner_rdna.hip.h`):
- Each wave loads its own B (its FN 16x16 blocks per k-tile, only the dwords its lanes
  decode: `exl3_lane_plan`) into a register ring DB = 4 k-tiles deep with
  compile-time slots (unrolled loop), SGPR-base `global_load`s. No block barrier per
  k-tile.
- The GEMV tiles core's exact-integer decoder (`exl3_dq_tile_decode`), written
  *transposed* ([n][k]) into a wave-private LDS block, so the WMMA B fragment is 2 x
  `ds_load_b128` (was 16 `ds_store_b16` + 16 `ds_load_u16` + 8 shuffles). Same halves
  in the same fragment slots -- `rdna_wmma.hip.h` fragment logic untouched, WMMA gate
  PASS.
- A (the gathered input, shared by all 16 waves) staged through LDS in chunks of
  8 / row-blocks k-tiles: one uint4 per thread and one LDS-only barrier per chunk;
  16-byte chunks XOR-swizzled by row (`(r>>1)^(r>>3)`) instead of padding.
- LDS-only fences (`__builtin_amdgcn_fence(..., "workgroup", "local")`): no vmcnt(0)
  drain, no `buffer_gl0_inv`.
- Row tiles 16/32/48/64 chosen per expert inside the kernel (CODE_SCAN #20), sharing
  one decoded B fragment across row blocks.
- g, u, d through one call site (the old kernel had three, so clang outlined the inner:
  FLAT loads, callee-saved spills). The column-end reduce is inlined per unrolled step
  behind `[[unlikely]]`.
- Budget: <= 192 VGPR (`amdgpu_waves_per_eu(EXL3_MOE_PIPE_WPE = 8)`), <= 32 KB LDS
  (`smem_launch_bytes`), so **two blocks per WGP**. The host launches 40 blocks only
  when `hipOccupancyMaxActiveBlocksPerMultiprocessor` says 2 (`EXL3_ROCM_MOE_BPS=1`
  forces one). `exl3_moe_max_concurrency` returns 40 / 8 = 5 groups, so Python sizes 5
  expert buffers; a mainloop switch after load only lowers the group count.
- Python side (`rocm_py`, Py-hook): fused-row cap 128 -> 512 while the pipe is on
  (`EXL3_ROCM_MOE_FUSED_ROWS`; upstream `EXL3_MOE_FUSED_ROWS` wins): hot experts stay
  in the kernel instead of reconstruct + hgemm. DS4 pp2048 343.8 -> 371.2 (+8%).

**The diagnosis moved twice -- read before optimizing this further:**
1. CODE_SCAN #2's latency-bound reading was right for the OLD loop (1.4 us per
   k-tile = the DRAM round trip). Once loads were pipelined, B-ring depth stopped
   mattering: DB = 2 / 4 / 8 within 2% everywhere, and removing the B loads entirely
   gains 2%.
2. The new loop is **issue/VALU-bound**, measured by removal on the inner bench (1 block
   per WGP, 12 rows, 1.296 ms base): no decode -30%, no WMMA -20%, no A staging -7%,
   no LDS transpose -5%, no B loads -2%. The hot step is ~75 VALU (K2 decode of two
   16x16 blocks) + 2 WMMA (32 cycles each) + ~25 SALU per wave; four waves per SIMD
   reach ~60% of VALU peak. A second block per WGP (8 waves/SIMD) adds 11-13%.
   Memory-only (loads, no compute) streams B at ~180 GB/s, so the 120 GB/s target is a
   compute problem at these shapes, not a memory one.

In-kernel phase split (debug build `-DEXL3_MOE_PIPE_PROF`, T=512): GEMMs 81%, gather +
input Hadamard 5%, g/u Hadamard + activation 6%, d-out Hadamard 3%, group barriers 3%,
scheduler 2%. At T=1792 the column-end reduce (fp16 2-byte stores, lock protocol,
vmcnt drain) is ~13% of GEMM time.

**Tried and not taken (inner bench, DS4 4096x2048 K2, ms for 12 / 42 rows):**
- A per wave from global (L2) instead of LDS chunks: loads-only rate 105 vs 180 GB/s;
  with compute present, ~equal at one block per WGP -- kept LDS for the LDS/VGPR budget.
- Decode one tile ahead into a second staging set (`EXL3_MOE_PIPE_AHEAD=1`): 1.392 ->
  1.295 at one block per WGP, but +16 KB LDS breaks the 32 KB two-block budget, and at
  two blocks it is within noise. Off.
- A chunk depth 4 vs 8 k-tiles: 1.178 / 2.156 vs 1.133 / 1.999 (two blocks). 8 kept.
- A rows padded to 40 halves (like the shared inner) vs unpadded: same speed as the
  swizzle but +25% LDS; unpadded without swizzle 1.232 / 2.542.
- B staging stride 24 vs 16 halves: 16 costs ~4% unswizzled; the half-swap swizzle
  recovers most of it and the stride fits the budget.
- 64 KB LDS request at two blocks per WGP: the bench co-scheduled 40 blocks, but the
  runtime's occupancy query says one block per 64 KB multiprocessor, so the launch
  never relies on it.
- Single out-of-line reduce re-entered through a `switch` over the ring slot: clang
  tail-merged the unrolled steps and copied the A ring through phis with an
  `s_waitcnt vmcnt(0)` per tile. Replaced by an inlined `[[unlikely]]` reduce per step.
- Chunk barrier at a dynamic position (`a_slot == 0` counter): vmcnt(0) before the
  chunk store every 8 tiles. The unroll is max(DB, AG) so chunk and slot positions are
  compile-time.
- Group width (`EXL3_ROCM_MOE_GROUP`, blocks per expert) at two blocks per WGP, T =
  256/512/1792 ms: 4 -> 16.09/17.13/29.39, 5 -> 16.04/17.42/30.55, **8 -> 15.79/17.11/
  29.98**, 10 -> 16.50/18.02/32.03, 20 -> 18.08/19.90/36.69. Upstream's 8 kept
  (MOE_SMS_PER_EXPERT unchanged); at 8 every block owns whole output columns for DS4
  (1024 k-tiles / 8 = one column of gate/up, two of down), so no fp16 partial sums.
- DB = 2 for the 48-row tile (to drop 6 spills): 7% slower at 42 rows; the spills sit
  in the kernel prologue / expert tail, outside every loop, so DB = 4 stays.

**Numerics.** On the same grid (`EXL3_ROCM_MOE_BPS=1`: 2 groups x 10) the pipelined
kernel is **bit-identical** to the old mainloop: `bench_moe_kernel --check` DS4 K2
mul1, K1/K3/K5 mcg, K8, Qwen K4 mul1 and K6 mcg at 64/256/512/1792 tokens, and
`moe_inner_bench` at 6-61 rows. The default grid (5 groups x 8) changes the stream-K
split, which removes the fp16 partial-sum round trip for DS4 shapes: max |diff| vs the
old grid 2.1e-3 (7.9e-4 of max |out|), and `moe_ref32` (fp32 reference, DS4 layers 0-1)
relmean 0.0109/0.0112/0.0084/0.0108% vs 0.0111/0.0113/0.0085/0.0111% for the old
kernel (bsz 64 / 256) -- equal or slightly closer. The fused-row cap moves hot experts
from reconstruct + WMMA hgemm into the kernel (different accumulation order).
PPL (wikitext2 100 x 2048, `bench/results/ppl_moe_mainloop/`): DS4 **6.461696** (stack
6.461140, +0.0086%), Qwen3.8 **4.748012** (4.746908, +0.023%; with the cap left at 128
it is 4.746794, -0.002%, so the cap -- hot experts through the fused kernel instead of
reconstruct + WMMA hgemm -- is most of that), Gemma **18.633857** (dense, identical).
`moe_check` DS4 bsz 16/64/256 worst 0.028%; gemm_check all pass; `bench/run_gates.sh`
PASS (mgemv_check, reconstruct_had, dsa_kernels, WMMA gate, 879 pytest).

**Portability.** `hipcc_probe --all` 123/123 on gfx1151/1100/1101/1200/1201. Pipe
kernel K2 mul1: gfx1151 168 (N128) / 192 (N256, 6 spills, 28 B scratch outside the
loops), gfx1100 172 / 192 (12 spills, 52 B, outside the loops), gfx1201 94 / 95 (the
gfx12 WMMA traps as before; rocm_py steers MoE off gfx12). The old instances are
unchanged (248 / 256 VGPR, 48-80 B scratch).

**Open:** decode (VALU) and WMMA now bound the loop at pp512 shapes (95-102 GB/s
synthetic, short of 120); the next levers are fewer VALU per weight in the K2 decode
and trimming the column-end reduce (half2 stores via a lane exchange, release-only
fences). Mixed-K models (K instances = 0) get the pipelined loop with the 16-row tile
only; no mixed-K model was run.

## DSA prefill MQA kernel (2026-09-28): DS4 pp2048 368 -> 518 t/s

Branch `opt/dsa-prefill` (PROFILE.md §6, CODE_SCAN #1). `_dsa_attn_kernel`, the one-shot
sparse attention kernel every DeepSeek-V4 prefill chunk runs (R > 8 rows), was the top
prefill kernel after the MoE mainloop fix: 32.8% of pp2048 GPU time, 43 calls x 22 ms.

**Root cause.** Same as the decode split kernel (section "DSA decode MQA kernel"):
- Each program holds BLOCK_H 32 heads x the whole output width, so a 32 x 576 fp32
  accumulator over 128 lanes is 144 VGPRs.
- The loop-invariant 32 x 512 q tile is resident as the WMMA A operand, which RDNA3
  replicates across half-waves.
- Result: 256 VGPR, 2878 spills, 5164 B scratch per lane (upstream default tuning).

**Fix.** `rocm_py/dsa_prefill_rdna.py:_dsa_prefill_mqa_kernel` is a drop-in for the
one-shot kernel: same arguments, same output layout, same features (window ring + chunk,
sinks, eq. 26 de-rotation, group-major store, dense / gathered pool, online packed pools in
the H32 domain, NC_CHUNK image chunks, NC_BLOCK DSpark draft).
- One program = HP heads x one BD-wide output column block x one query row.
- The score reduction over D = 512 is a runtime loop of KC-wide chunks with q re-read per
  chunk, so q is never a resident WMMA operand. The virtual key row is [c | r]: ring/chunk
  rows are contiguous, pool rows are pool_c ++ pool_r.
- Production tiling: HP 64 (all heads share the one latent KV head), BD 512 (the whole row,
  so no score recompute), KC 32, BLOCK_N = BLOCK_W 64, 16 warps. The 64 x 512 accumulator
  is 64 VGPRs over 512 lanes.
- Wiring: `dsa_attn` looks `_dsa_attn_kernel` up as a module global at call time, so a
  launch proxy (as for decode) routes eligible calls with their own grid,
  `(R * H/HP * D/BD,)`. No upstream edits.
- Declined (upstream kernel): Q_SPLIT / OUT_LATENT (GLM-5.2 DSA-on-MLA), non-power-of-two
  D (V3.2's 576), H < 16 or not a multiple of HP, D_r == 0. The split path (R <= 8) is the
  decode kernel's and is unchanged.

**Microbench** (`rocm_tools/bench_dsa_prefill.py`; the whole `dsa_attn` call as dsv4's
cached prefill makes it: single block-table row, sinks, derot, 8 output groups, packed-pool
staging when qc; CUDA events, best of 6; err = max abs vs fp64 reference over 40 rows /
max |ref|, identical for old and new in every row):

| case (R new rows at ctx) | old ms | new ms | speedup | new spill / scratch B |
|---|---|---|---|---|
| csa dense R512 ctx0 | 9.36 | **1.40** | 6.7x | 5 / 20 |
| csa dense R2048 ctx0 (pp2048) | 52.6 | **9.81** | 5.4x | 5 / 20 |
| hca R2048 ctx0 | 31.5 | **4.64** | 6.8x | 5 / 20 |
| csa top-k R64 ctx3K | 3.03 | **0.62** | 4.9x | 25 / 104 |
| csa top-k R255 ctx3K | 10.56 | **2.03** | 5.2x | 25 / 104 |
| csa top-k R1792 ctx3K | 71.7 | **14.1** | 5.1x | 25 / 104 |
| hca R255 ctx3K | 4.49 | **0.67** | 6.7x | 5 / 20 |
| csa top-k R64 ctx16K | 3.33 | **0.63** | 5.3x | 25 / 104 |
| csa top-k R255 ctx16K | 11.36 | **2.12** | 5.4x | 25 / 104 |
| csa top-k R2048 ctx16K | 86.9 | **16.0** | 5.4x | 25 / 104 |
| hca R64 ctx16K | 1.79 | **0.26** | 6.8x | 5 / 20 |
| hca R2048 ctx16K | 44.2 | **7.90** | 5.6x | 5 / 20 |
| csa top-k R255 ctx3K qc4 (staged) | 10.51 | **2.11** | 5.0x | 25 / 104 |
| csa top-k R2048 ctx16K qc4 (staged) | 88.1 | **16.1** | 5.5x | 25 / 104 |
| hca R255 ctx16K qc4 (staged) | 6.02 | **0.95** | 6.3x | 5 / 20 |
| csa top-k R40 ctx3K qc4 (online, R < 64) | 2.08 | **0.79** | 2.7x | 307 / 880 |

Old: 256 VGPR, 2851-2878 spills, 5164-5368 B scratch in every case.

**Sweeps** (ms, columns = csa R2048 ctx0 | csa top-k R255 16K | csa top-k R64 16K | hca R2048 16K):
- Upstream kernel params (BLOCK_H x BLOCK_N x warps x stages, 36 variants, first two
  columns): the best is H32 N16 w4 s1 at 34.6 | 7.0 (1.55x over the default 53.7 | 10.9),
  still 1005 spills / 3.1 KB. Every variant spills 661-2878. Retuning alone cannot reach 2x.
- New kernel, compile-only over HP {16,32,64} x BD {128,256,512} x KC {32,64,128} x BN
  {16,32,64} x warps {4,8} (162 variants): KC 128 and BN 64 at 4 warps spill hundreds; BD
  512 at 4 warps spills unless HP is 16.
- Timed (selection):

  | HP / BD / KC / BN / warps | ms | spills |
  |---|---|---|
  | 32 / 256 / 64 / 32 / 4 (decode kernel's tile) | 14.7 / 3.20 / 0.88 / 10.7 | 13 |
  | 32 / 256 / 32 / 32 / 8 | 18.5 / 3.91 / 1.12 / 13.4 | 0 |
  | 64 / 256 / 32 / 32 / 8 | 14.2 / 2.96 / 0.89 / 10.8 | 0 |
  | 64 / 256 / 32 / 64 / 8 | 12.4 / 2.60 / 0.79 / 9.78 | 30 |
  | 64 / 256 / 16 / 64 / 8 | 11.9 / 2.49 / 0.74 / 9.49 | 19 |
  | 64 / 128 / 64 / 64 / 8 | 20.3 / 4.23 / 1.10 / 18.1 | 0 |
  | 64 / 256 / 32 / 64 / 16 | 17.6 / 3.40 / 0.87 / 14.4 | 0 |
  | **64 / 512 / 32 / 64 / 16** | **9.8 / 2.08 / 0.63 / 7.9** | 5 |
  | 64 / 512 / 16 / 64 / 16 | 13.2 / 2.25 / 0.71 / 9.95 | 46 |
  | 64 / 512 / 64 / 64 / 16 | 11.3 / 2.38 / 0.72 / 9.13 | 12 |
  | 64 / 512 / 32 / 32 / 16 | 16.0 / 3.31 / 1.01 / 12.2 | 0 |
  | 32 / 512 / 32 / 64 / 16 | 24.3 / 4.90 / 1.29 / 19.6 | 18 |
  | 64 / 512 / 32 / 128 / 16 | 20.2 / 3.53 / 0.99 / 17.3 | 315 |

- Rejected, with numbers:
  - KSTAGES 2 (software-pipelined KC loop): 14.2 vs 12.4 ms (64/256/32/64/8).
  - BLOCK_W < BLOCK_N (smaller window tile at the winner): BW 32 13.0 ms, BW 16 20.0 ms.
  - 32 warps: exceeds the dispatch limits (1024 threads x 256 VGPR; HSA
    INVALID_DISPATCH_PARAMETERS). BN 128 + KC 16 at 16 warps needs 128 KB LDS.
  - Zero-spill variants are all slower (best: 64/256/32/32/8 at 14.2 ms). The remaining 20
    B scratch per lane (104 B gathered) is register-allocator noise in the main loops, not
    the epilogue: with DEROTATE off it goes UP to 50 spills.
  - Key splits for short tails: not needed. Per-row cost at R 64 is within ~20% of R 2048
    (0.63 ms / 64 rows vs 16 ms / 2048 at 16K).

**End to end** (DS4 2.04bpw, `bench/run_bench.py --pp 512 2048 --tg 128 --regen 3000 8000
--long 16384`, same build, `EXL3_ROCM_DSA_PREFILL=0` vs default):

| | off | on |
|---|---|---|
| pp512 | 281.4 | **352.0** (+25%) |
| pp2048 | 367.7 | **518.0** (+41%) |
| regen 3000 (184-token tail) | 867.9 ms | **681.4 ms** (-21%) |
| regen 8000 (64-token tail) | 669.4 ms | **572.3 ms** (-15%) |
| tg128 | 27.77 | 27.76 (untouched) |
| tg64@16384 | 26.26 | 26.28 (decode untouched) |

pp2048 kernel trace after (rocprofv3, `logs/prof/dsa_prefill_pp2048`,
`bench/results/dsa_prefill_pp2048_trace_summary.txt`): wall 2962 -> 2003 ms per step;
the attention kernel 950 -> 157 ms per step (22.1 -> 3.6 ms per call), 32.8% -> 8.0%.
Top 5 now: exl3_moe_kernel 34.5%, _dsa_prefill_mqa_kernel 8.0%, exl3_wmma_gemm (fp16,
1x2x4) 7.9%, exl3_wmma_gemm (fp32 out) 5.2%, exl3_wmma_gemm_ks1 3.8%.

**Numerics / validation.**
- Different reduction order (D chunked by 32; 64-key tiles instead of 32; window tiles 64
  instead of 16). Error vs the fp64 reference is unchanged in every microbench case.
- `test_dsa_kernels.py` ALL PASS; its H64/D512 one-shot cases (R 7/9/40/200/390/130/4096,
  incl. NC_CHUNK, QC online and staged) run the new kernel (launch counter checked).
  H8, H16/D288, H128/D576 fall back.
- NC_BLOCK (DSpark draft; no test covers it) and token-major / no-sinks / no-derot: new vs
  old kernel 4.9e-4 / 5.8e-4 of max |out| (fp16 output rounding), no NaN.
- DS4 PPL (`MODELS=ds4 bench/run_ppl.sh`): 6.463333 vs 6.461696 (+0.025%).
- `bench/run_gates.sh`: PASS (879 passed, 9 skipped).

**Switches.**
- `EXL3_ROCM_DSA_PREFILL=0` restores the upstream one-shot kernel.
- `EXL3_ROCM_DSA_PREFILL_{HP,BD,KC,BLOCK_N,BLOCK_W,WARPS,KSTAGES}` override the tiling.
- Kernels changed for the coherence check: prefill sparse attention in every DS4 layer
  (every chunk with more than 8 rows, image chunks, the regeneration tail, DSpark draft).
  Decode, the split/combine kernels, the indexer and GLM-5.2's DSA-on-MLA are unchanged.

**Open.**
- Online packed pools (-cq with a tail under 64 rows) still spill 307 VGPRs / 880 B: the H32
  rotation of q and window chunks inside the KC loop (the same open item as decode).
- The kernel is now below MoE and on par with the WMMA GEMMs. Next prefill levers are
  outside attention (MoE 34.5%; PROFILE §8).

## Decode leftovers (2026-09-28): DS4 tg128 27.5 -> 30.1 t/s, MTP ndt=2 31.4 -> 41.7, bit-identical

Branch `opt/decode-leftovers` (PLAN target: 30+ plain decode on DS4). Seven changes, each
behind its own switch in one build; every one keeps the output bit-identical (decode_bitwise
DS4 / Qwen3.8 / Gemma all-on vs all-off PASS).

**The profile first** (`bench/results/decode_profile_c6c19ca.txt`, rocprofv3, ctx 512, 32
steps, c6c19ca): 37.20 ms/step wall under the profiler, 32.77 busy, 1735 kernels, 4.42 ms of
gaps (mean 2.55 us). The GEMVs were already at the ~206 GB/s roofline (dense K4 199-210,
routed K2 170-180, head 227). The host runs **~7 ms ahead** of the GPU (p50 launch-to-start
slack over every dispatch), so decode is GPU-bound and every dependent launch costs its ~2.5
us command-processor gap in GPU time. What was left was the router (37.8 us/call, 55 GB/s),
the weighted down projection on the LDS-prologue kernel (82.5 us vs 69.8 for the same bytes
on gate/up), and launches: 7 per routed-MoE layer, 4 per mHC site (86 sites).

MTP (-ndt 2): the per-token MoE loop (21 launches per layer at m = 3) and a router that
went to hgemm at m = 3 (narrow split-K, two launches, ~40 us). **profile_region.py fix:** it
enqueued a steps + 8 = 40-token job, which an MTP run finishes mid-region at ~2.5 tokens per
step (the tail steps were idle generator iterations, diluting every per-step figure); the job
is now 8 x steps + 8. With the fix, same build: all switches off 77.82 ms/step (busy 60.90,
2854 kernels), all on 68.25 (busy 52.93, 2017 kernels), 82 tokens per 32 steps both. MTP
also carries ~15 ms/step of host-sync gaps under the profiler (7-8 pageable
hipMemcpyWithStream + 3 hipDeviceSynchronize per step, generator side, both builds).

### What changed

| switch (default on) | change | where |
|---|---|---|
| `EXL3_ROCM_ROUTER_GEMV` | router GEMV, m = 1..8, loads in flight; m == 1 with 16-byte loads and a per-wave LDS transpose that keeps upstream's lane -> column chain (bit-identical) | `rocm/routing_rdna.hip` (sibling of routing.cu) |
| `EXL3_ROCM_ROUTER_FUSE` | router + top-k in one launch (last-arriving block runs the top-k body verbatim) | same |
| `EXL3_ROCM_MR_WEIGHTED` | weighted MoE down on the multi-row path with the mgemv fused epilogue (grouped reduce) | `rocm/quant/exl3_gemv_multirow_rdna.hip` |
| `EXL3_ROCM_MOE_BATCH` | MoE decode route runs all bsz <= 8 tokens in one set of launches; act over the S valid rows only; out_bszn aliases out_d (no copy_) | `rocm_py/__init__.py` |
| `EXL3_ROCM_MOE_FUSED` | `torch.ops.exl3_rocm.moe_decode`: gate+up as one 2S-slot GEMV, silu*up folded into the down projection's input rotation; 4 launches per layer (was 7 at bsz 1, 21 at bsz 3) | multirow sibling + rocm_py |
| `EXL3_ROCM_HC_DPP` | hc_mix partials / sinkhorn cross-lane ops on DPP instead of ds_bpermute | `rocm/hc_mix_rdna.hip` (sibling of hc_mix.cu) |
| `EXL3_ROCM_HC_FUSE` | mHC apply_ deferred into the next site's mix, fused into its partials kernel | same + rocm_py |
| `EXL3_ROCM_HC_NORM` | the RMSNorm after each mix replayed inside the mix's finalize (one block per row, row in LDS) | same + rocm_py |

(`EXL3_ROCM_ROUTER_U` = 2/4/8/16 picks the router's blocks in flight, default 8.)

### Step by step (rocprofv3, same harness, ms/step profiled wall; each row adds to the one above)

| step | wall | kernels/step | what moved |
|---|---|---|---|
| c6c19ca | 37.20 | 1735 | |
| router m <= 8 (dword kernel, U16) + weighted down on multi-row + batched route | 36.13 (-1.07) | 1735 | router 37.8 -> 22.3 us; down 82.5 -> 69.7 us; per-token copy_ gone, down's rotation now its own launch |
| router 16-byte loads + LDS transpose (U8) + fused top-k | 35.85 (-0.28) | 1695 | router + top-k 37.8 + 4.2 us -> 17.7 us, 40 launches |
| fused MoE decode op | 34.97 (-0.88) | 1566 | gate + up one launch: 2 x 70.8 -> 131.3 us; act folded; 7 -> 4 launches per layer |
| hc_mix on DPP | 34.66 (-0.31) | 1566 | partials 8.4 -> 6.6 us, finalize 5.3 -> 4.0 us |
| hc apply folded into the next mix | 34.47 (-0.19) | 1481 | apply 2.2 + partials 6.6 -> 7.8 us |
| RMSNorm folded into the finalize | 34.27 (-0.20) | 1395 | finalize 4.0 + rms_norm 2.3 -> 6.7 us |
| **total** | **-2.93** | **-340** | busy 32.77 -> 30.67, gaps 4.42 -> 3.59 ms |

### End to end (bench/run_bench.py, one build, median of 3; `=0` rows turn one switch off)

DS4 `--pp 512 --tg 128 --long 16384`:

| config | pp512 | tg128 | tg64 @ 16K |
|---|---|---|---|
| **all on** | 353.6 | **29.99** | **27.93** |
| all off (= c6c19ca paths) | 350.0 | 27.52 | 25.78 |
| ROUTER_GEMV=0 (also drops the fuse) | 348.0 | 29.08 | 27.21 |
| ROUTER_FUSE=0 | 349.5 | 30.02 | 27.93 |
| MR_WEIGHTED=0 | 349.8 | 30.07 | 28.04 |
| MOE_BATCH=0 (also drops the fused op) | 349.3 | 29.18 | 27.24 |
| MOE_FUSED=0 | 351.4 | 29.40 | 27.47 |
| HC_DPP=0 | 347.7 | 29.95 | 27.89 |
| HC_FUSE=0 | 351.4 | 29.81 | 27.79 |
| HC_NORM=0 | 348.7 | 30.01 | 27.93 |

(pp512 moves within noise: every change is decode-only -- R <= 32 / m <= 8 gates.)
MR_WEIGHTED is flat because the fused op carries the down projection. ROUTER_FUSE,
HC_DPP and HC_NORM were each below one run's noise in that matrix, so they got an
alternating A/B, tg128 only, 5 runs per point, three rounds (on / fuse=0 / dpp=0 / norm=0
in turn):

| round | all on | ROUTER_FUSE=0 | HC_DPP=0 | HC_NORM=0 |
|---|---|---|---|---|
| 1 | 30.10 | 30.06 | 29.96 | 30.02 |
| 2 | 30.13 | 30.09 | 29.98 | 30.00 |
| 3 | 30.12 | 30.08 | 29.97 | 29.99 |
| ms/token vs on | 33.20 | +0.04 | +0.17 | +0.12 |

Small, but every round orders the same way; all three stay on.

DS4 MTP `--mtp -ndt 2` (greedy, three natural prompts; per-prompt t/s and acceptance):

| config | median | per prompt | acceptance |
|---|---|---|---|
| **all on** | **41.67** | 35.03 / 41.67 / 42.47 | 0.770 / 0.781 / 0.914 |
| all off | 31.36 | 30.16 / 35.06 / 31.36 | 0.781 / 0.781 / 0.785 |
| ROUTER_GEMV=0 | 35.78 | 34.33 / 40.96 / 35.78 | 0.781 / 0.781 / 0.785 |
| MOE_BATCH=0 | 36.36 | 31.16 / 36.36 / 37.60 | 0.770 / 0.781 / 0.914 |
| MOE_FUSED=0 | 40.72 | 34.23 / 40.72 / 41.59 | same as all on |
| HC_FUSE=0 | 41.54 | 34.78 / 41.54 / 42.38 | same |
| HC_NORM=0 | 41.53 | 34.87 / 41.53 / 42.47 | same |

The router switch changes prompt 3's acceptance (0.785 -> 0.914), so its median is not a
like-for-like speed number: per prompt it is worth ~2% (40.96 -> 41.67, 34.33 -> 35.03).
The reason is a correctness improvement: **MTP greedy output is now token-identical to
plain greedy decode** on all three prompts (128 tokens each; all on). With every switch off,
MTP diverged from plain greedy at token 37 and 88 on two of them -- the verify step's
router ran on hgemm, a different reduction than decode's routing_gemv (the dense multi-row
GEMVs and the per-token MoE loop were already per-row identical), so near-tie expert picks
could flip. Now every verify row takes the m == 1 arithmetic end to end: dense multi-row,
batched MoE with the per-token wave rule, router rows, mHC.

Qwen3.8-Flash-Next 4bpw `--pp 512 --tg 128`: all on 597.3 / **26.64**, all off 594.8 / 25.57
(+4.2%); MOE_FUSED=0 25.81, ROUTER_GEMV=0 26.36. Gemma-4-31B (dense, no router / mHC):
7.69 / 7.69, unchanged as expected (pp512 248.3 / 247.7).

### Validation

- decode_bitwise all-on vs all-off (reference saved in the same build): DS4, Qwen3.8,
  Gemma PASS (48 steps, every logit bit-identical).
- mgemv_bitwise DS4 + Qwen vs the 3cf11c5 references PASS (the weighted down now runs
  on the multi-row path); multirow_check PASS; mgemv_check PASS (after the EXL3_MGEMV /
  skipped-slot fixes above); negative-index weighted check vs the cooperative kernel PASS.
- `bench/run_gates.sh` PASS (879 passed, 9 skipped).
- PPL (wikitext2 100 x 2048): DS4 6.463333, Qwen3.8 4.748012, Gemma 18.633857 -- identical
  to the c6c19ca references (PPL runs the prefill paths, which these changes do not touch).
- `hipcc_probe --all` 123/123 on gfx1151, gfx1100, gfx1101, gfx1200, gfx1201.

### Details worth keeping

**Router.** Upstream's kernel is one warp per expert row (E / 8 = 32 blocks), 4-byte
loads, lane l accumulating columns l, l + 32, ... in order. Loads-in-flight alone (RG_U = 16
dword loads per lane before the chain consumes them, 2-wave blocks) took it from 37.8 to
22.3 us and no further: 512 K dword load instructions for 2 MB is the limit (address/TA
rate), not latency. The m == 1 kernel now loads 16 bytes per lane (4 adjacent half2
columns of a 128-column block), writes the block to the wave's LDS slice and reads back
its own columns l, l + 32, l + 64, l + 96 -- the chain is upstream's, column for column,
bit-identical. U (blocks in flight per wave): U2 20.5, U4 19.7, **U8 13.2 us** (150 GB/s).
The fused top-k (last-arriving block, release/acquire + self-resetting counter) is 17.7 us
for both; in the e2e A/B it is within noise (the top-k body is serial either way).
m = 2..8 (MTP verify): each row the m == 1 chain -- verify-row router scores are now
bit-identical to plain decode's (they were hgemm's, a different reduction).

**Weighted down on the multi-row path.** Same dot core, wave count and N-tiles per block
as the LDS-prologue mgemv kernel; the epilogue is exl3_gemv_fused_epilogue verbatim, so the
routed sum is bit-identical (mgemv_bitwise DS4 + Qwen vs the 3cf11c5 references PASS).
Skipped slots (negative indices, the documented "skip" API) arrive as all-zero rows so the
grouped reduce completes (adding +0 is exactly skipping); the path honors EXL3_MGEMV=0 like
the mgemv path it stands in for (mgemv_check's coop-only masked case relies on that). A
direct negative-index test against the cooperative kernel: fp32 max rel err 3e-5, no NaN.
With the fused op on, nothing weighted reaches it at decode (its e2e A/B is flat); it stays
for callers of exl3_mgemm with weights.

**Fused MoE op** (`torch.ops.exl3_rocm.moe_decode`, registered from the sibling with
TORCH_LIBRARY_FRAGMENT -- no upstream binding edits). The split-K wave count is sized from
one token's slots per matrix (top_k), the rule the separate calls apply, so every slot keeps
its reduction order; the same rule is what lets the batched MTP verify stay bit-identical
per row (exl3_mgemm's num_tokens is passed to gate/up for this). The 2S-slot gate+up grid is
also simply faster than two S-slot grids (131 vs 142 us).

**mHC.** Every cross-lane op in hc_mix moves a value within a row of 16 lanes (or row 1 ->
row 0 for shfl_down 16), so DPP row_shl / quad_perm / row_xmask / v_permlanex16 carry the
same values and every add keeps its operands. The apply fold works because partials chunks
never straddle a stream: a block of 4 x 64 threads owns chunk c of each of the 4 streams,
i.e. the same columns of all 4, so it has every x value the apply needs (via LDS, before any
store), updates x in place without racing other blocks, and gives each chunk the unfused
kernel's exact per-thread sequence and 2-warp reduce. First version ran 16.0 us: at the
256-thread launch bound the compiler capped VGPRs at 64 and serialized the 24 fn loads
(load / wait x 24); issuing them into registers before the barrier and
`__launch_bounds__(256, 1)` gave 7.8 us. The norm fold replays rms_norm_kernel's single-pass
form (1024 virtual threads, sum_sq4, reduce_dyn's two xor butterflies, (x * w) * rmf, same
half rounding); a last-arriving-block version cost 7.3 us (two passes over the row from
global), one block per row with the row in LDS 6.7 us. The deferred apply is flushed before
anything else can see the streams (any other HC call, HyperHead, a device change in
prepare_for_device); nothing is deferred while exporting states or converting.

### Tried / rejected / not done (with numbers)

- **K = 5/6/8 funnel shift** (GEMV tiles open item): not done. DS4's only K6 GEMV is lm_head at
  1757 us = 227 GB/s, above bench_membw's 206 GB/s -- memory-bound, so fewer VALU ops per tile
  cannot move it; Gemma's K6 mcg shapes sit at 200-227 GB/s too (GEMV-tiles notes: 7.70 t/s
  against a ~9.7 roofline set by bytes). Expected < 0.05 ms/token on DS4.
- **Router, loads-in-flight only** (dword loads, U = 16, 2-wave blocks): 22.3 us -- address/TA
  bound at 512 K load instructions; superseded by the 16-byte LDS-transpose kernel (13.2 us).
- **Router + top-k fusion**: 17.7 us for both vs 13.2 + 4.2 us + a gap; e2e within noise
  (tg128 30.02 off vs 29.99 on, one 3-run median each). Kept (bit-identical, 40 launches).
- **HC apply fold, first version**: 16.0 us per site (vs 8.8 unfused) -- serialized fn loads
  at a 64-VGPR budget; fixed as above.
- **Norm fold via a last-arriving block** (counters, row re-read from global twice): 7.3 us
  finalize; one block per row with the row in LDS: 6.7 us.
- **DSA decode: fewer splits / no combine at short context**: not re-tried -- the DSA decode
  sweep has 8 splits (split 15.5 us incl. combine) beating 2 (28.7) and 4 (18.1).
- **Gate + up as one 12-slot exl3_mgemm from Python** (CODE_SCAN #7): needs a (1, 2S) index
  tensor (one more launch) and, with 12 slots, the split-K rule picks 4 waves instead of 8
  (numerics change); done inside moe_decode instead, where both problems disappear.

### Open

- Step boundary: ~0.47 ms/token of idle between the last kernel of a token and the first of
  the next (sampling sync, generator turnaround; copyBuffer -> copyBuffer gaps). Host side.
- hc_apply_partials runs 16 blocks (one per 4-chunk group); splitting each chunk's two warps
  across blocks (finalize summing the pair, which is the unfused reduce's exact order) would
  double the grid.
- Attention small ops (copy2d, compress_store x1-2, ring_append, q_a rms_norm: ~5 launches
  per layer) are in upstream dsv4_attn.cpp -- a sibling of that file is the next launch-count
  lever (~0.5 ms/token).
- Shared expert (5 launches + the routed + shared add): moe_decode could carry it, but its
  gate/up intermediates are fp32 (act_mul_kernel_f), a second code path.

## Qwen3.8 (2026-09-28): tg128 26.6 -> 29.0 t/s, MTP ndt=2 39.3 -> 43.9, pp2048 750 -> 848; n-gram table lock

Branch `opt/qwen` (base b22b243). Profile: `bench/results/qwen_profile_b22b243.txt`. Four changes, each behind
its own switch (default on), plus the `-ngl` n-gram lock mode in the server.

**Why Qwen was "a weirdo".** About a quarter of its ~5.0 GB/token decode stream is the fp16 GatedResidual
("low-rank hyper-connection") matrices: 97 sites x (324 x 10240 + 10240 x 320) halves. They run on their own two
kernels (gr_dots / gr_finalize), not on the EXL3 GEMV, so GEMV-tiles never touched them. The EXL3 GEMVs (K4 experts,
K5/K6 dense) were already at 168-228 GB/s. And the 12 full-attention layers ran a graphed Triton kernel compiled
without AMD's buffer-op specialization, at 91.5 us per call instead of ~14.

### What changed

| switch (default on) | change | where | kernels changed | numerics |
|---|---|---|---|---|
| `EXL3_ROCM_GR_DOTS` (`_GR_RB` 2/4/8, default 4) | gr_dots_rows_kernel: 4 fn rows per block, stream stack loaded once into registers, a whole row's loads in flight. 60.8 -> 36.3 us per site | `rocm/hc_mix_rdna.hip` | GatedResidual decode mix (R <= 32), Qwen only | bit-identical (same per-thread fmaf chain, same shuffle/DPP tree, same 4-warp sum) |
| `EXL3_ROCM_GR_PREFILL` | `torch.ops.exl3_rocm.gr_gate_mean`: the prefill gate-mean tail (2 fp32 upcasts, sigmoid, mul, mean) in one pass. 3.6 ms -> 0.4 ms per site at R = 2048 | same + `rocm_py` | GatedResidual prefill mix (R > 32) | bit-identical to torch's expression (under `#pragma clang fp contract(off)`; torch rounds the product before the sum) |
| `EXL3_ROCM_BC_BUFOPS` (`_BC_ATTN_WARPS` / `_STAGES`, default 8 / 1) | bc_attn's AOT GQA decode split kernels (paged split + QSA sparse split) compiled with the `tt.pointer_range = 32` / `tt.divisibility = 16` attributes that the Triton JIT adds on AMD, at 8 warps / 1 stage (scratch-free). In-model: ctx 512 91.5 -> 31.8 us, QSA sparse at 8K 117 -> 31.9 us, MTP verify 163 -> 62 us | `rocm_py` (wraps `bc_attn._compile_kernel`, gated per layer on K/V storage < 2 GB) | graphed GQA decode attention (Qwen; any BCAttn model with an fp16 cache) | bit-identical (decode_bitwise PASS, 21- and 4001-token prompts) |
| `EXL3_ROCM_ROUTER_STD_MR` | `routing_std` at 2..8 rows passes the transposed gate, so rows go to the existing multi-row router GEMV (DS4's) instead of hgemm: 105 -> 40 us per MTP verify call | `rocm_py` | std softmax routing at bsz 2..8 (Qwen MTP verify) | each verify row now uses the m == 1 chain: MTP greedy output is token-identical to plain greedy on 3/3 prompts (1/3 diverged at token 34 before) |
| `EXL3_ROCM_PREFILL_HD256` | paged prefill attention at head_dim 256: 1 stage, 128-row tile (64 below 128 query rows), still 16 rows per warp. Upstream's tile spilled (256 VGPR + 772 B). Kernel: q 1792 fresh 5770 -> 2917 us, q 2048 after 16K 121.7 -> 39.8 ms | `rocm_py` (wraps `paged_attn_triton_prefill`) | Qwen full-attention prefill, Gemma's 256-dim layers | different tiling: Qwen PPL +0.074% (4.748012 -> 4.751540; that is the whole PPL delta of the branch); error vs an fp32 reference unchanged (1.15e-3 / 4.2e-5) |

### End to end (bench/run_bench.py, one build, median of 3, `-ngr`; `=0` rows turn one switch off)

| config | pp512 | pp2048 | tg128 | regen3000 ms | regen8000 ms | tg64 @ 8K | MTP ndt2 |
|---|---|---|---|---|---|---|---|
| b22b243 (before) | 596.0 | 749.6 | 26.61 | 377.8 | 305.4 | | 39.32 |
| **all on** | 622.0 | **848.5** | **29.05** | 363.6 | 300.7 | 28.54 | **43.92** |
| GR_DOTS=0 + GR_PREFILL=0 (d365a4f A/B) | 593.4 | 747.8 | 26.64 | 376.2 | 305.5 | | |
| BC_BUFOPS=0 | 615.2 | 829.9 | 28.34 | 362.6 | 300.4 | 27.60 | 43.31 |
| ROUTER_STD_MR=0 | | | | | | | 41.42 |
| PREFILL_HD256=0 | 618.3 | 833.1 | 29.02 | 363.4 | 299.5 | 28.58 | |

Long prompts, PREFILL_HD256 on/off: pp8192 775.8 / 767.1, pp16384 761.8 / 755.1. Past the 2048-token indexer
budget the QSA layers switch to sparse prefill, so the dense kernel matters less there. Gemma-4-31B pp2048
332.9 / 319.3 (+4%); tg unchanged at 8.34.

DS4 (the hard rule: no regression). Full set on vs off for each commit, same build:

| DS4 | pp512 | pp2048 | tg128 | regen3000 ms | regen8000 ms | MTP ndt2 |
|---|---|---|---|---|---|---|
| b22b243 reference | ~354 | ~518 | ~30.1 | ~681 | ~572 | ~41.7 |
| GR on / off (d365a4f) | 351.7 / 347.2 | 521.9 / 512.3 | 30.12 / 30.12 | 679.4 / 685.9 | 572.6 / 577.6 | 41.45 / 41.53 |
| BUFOPS + ROUTER_STD_MR on / off (6684d1d) | 353.0 / 351.2 | 524.0 / 519.3 | 30.15 / 30.12 | 681.1 / 675.4 | 572.7 / 572.6 | 41.57 / 41.81 |
| PREFILL_HD256 on / off | 349.0 / 350.3 | 522.0 / 516.9 | 30.16 / 30.12 | 679.1 / 671.6 | 571.5 / 573.3 | 41.56 / 41.56 |

DS4 does not execute any of the new paths: its hc sites are mHC (hc_mix), its router is routing_ds3, and its
attention is DSA/MLA. DS4 decode_bitwise PASS against a reference saved with every switch off.

### Validation

- decode_bitwise Qwen, all on vs every switch off: PASS for 21-token, 801-token (prefill R > 32, the gate-mean path)
  and 4001-token prompts (with PREFILL_HD256 off, since that one changes prefill tiling). DS4 PASS.
- `rocm_tools/gr_mix_bench.py`: gr_mix bit-identical to the one-row kernel (RB 2/4/8, R 1 and 3, site and
  final-mixer forms); `--prefill` gr_gate_mean bit-identical to the torch expression at R 33 / 256 / 1792 / 2048.
- MTP greedy vs plain greedy (3 prompts x 128 tokens): all identical (switch off: one diverges).
- `bench/run_gates.sh` PASS (879 passed, 9 skipped; mgemv_check, reconstruct_had, dsa_kernels, WMMA gate).
- PPL (wikitext2 100 x 2048): Qwen 4.751540 (+0.074% vs 4.748012; 4.748012 exactly with PREFILL_HD256=0),
  DS4 6.463333 (=), Gemma 18.633857 (=).
- `hipcc_probe --all`: 123/123 on gfx1151, gfx1100, gfx1101, gfx1200, gfx1201 (no hardcoded CU counts; the new
  kernels size grids from the shapes, RB is a runtime switch).

### N-gram table lock (`-ngl`)

The maintainer asked for three modes: disk streaming (the default, unchanged), `-ngr` RAM (unchanged), and a new
locked-RAM mode. `exllamav3/rocm_py/ngram_lock.py` provides it; `server.py -ngl` and `run_bench.py --ngram_lock` use it.
There are no upstream edits: model_init / ngram_embedding are untouched.
- `-ngl` sets `ngram_ram`, so the table loads exactly as with `-ngr` (one contiguous CPU slab).
- After the load, `mlock(2)` is applied to that tensor's address range. There is no second copy and no change to
  the read path.
- Preflight, before anything loads:
  - `RLIMIT_MEMLOCK` must cover the table. An unprivileged process raises its soft limit to the hard limit itself;
    `CAP_IPC_LOCK` bypasses the check. If neither works, it exits with the `ulimit -l` / limits.conf /
    `LimitMEMLOCK` / `prlimit` / `setcap` recipes.
  - Weights + table + KV estimate + headroom (8 GiB, `EXL3_NGRAM_LOCK_HEADROOM_GB`) must be <= MemAvailable.
    A locked table can never be reclaimed, so the server refuses rather than end in the OOM killer.
- Postflight re-checks the headroom before locking. `/props` reports `ngram_table` (disk / ram / ram_locked).
- Measured:
  - This box's hard limit is 16375860 KiB (15.6 GiB), so a full 36.4 GiB lock is refused with the instructions
    (verified: `server.py -ngl` exits in 4 s, before loading).
  - `run_bench.py --ngram_lock --ngram_lock_max_gb 15` locked 15.0 GiB of the real table. `/proc/<pid>/status` of
    the bench process showed `VmLck: 15728644 kB` (external check).
  - Decode with the lock vs plain `-ngr`, two alternating rounds: tg128 28.96 / 29.03 vs 29.04 / 29.01, pp512
    622.5 / 626.2 vs 623.5 / 623.8. The lock costs nothing, as expected: the read path is identical.
  - A synthetic 12 GiB lock: VmLck 0 -> 12288 MiB; a second lock past the limit fails with ENOMEM and leaves the
    first intact.
- A full-table lock needs the maintainer to raise the limit (root). Budget at `-cs 65536`: 65.1 + 36.4 + 1.5 + 8 =
  111 GiB of ~115 GiB available.

### Tried / rejected / not done (with numbers)

- **gr_finalize with the up-gate loads hoisted above the prologue + DPP butterflies**: 36.5 -> 42.4 us (the hoisted
  loads made the prologue wait on them: vmcnt retires in order); with the prologue loads issued first, 37.9 us.
  Still slower than the upstream form (already ~180 GB/s). Removed.
- **Decode attention via the python JIT path**: that path was never the problem. The JIT launch of the same
  kernel was already 13.5 us. The graphed AOT compile lost the specialization.
- **BC attention warps/stages** (in-model, ctx 512 / 8K QSA, us per call): 4/2 38.9 / 51.9 (scratch 276 / 524), 8/2
  33.8 / 48.6, **8/1 31.8 / 31.9 (0 scratch)**, 16/1 33.4 / 77.1, 4/1 38.0 / 44.3, 2/2 63.9 / 88.9.
- **Prefill tiles at hd 256** (q 1792 fresh, us): 64x32 w8 s2 (upstream) 5770, 128x32 w8 s1 2917, 64x64 w4 s1 3271,
  64x32 w8 s1 3520, 64x16 w8 s1 3570. At q 33 all within noise; at q 64 after 4096 the 64-row tile wins (612 vs 857).
- **GR prefill matmuls on the WMMA backend** (hipBLAS runs them at 17-25 TF, ~54 ms per 1792-row forward): the
  backend needs a row-major B, which means transposed copies of proj/up, +1.3 GB. The `-ngr` / `-ngl` memory budget
  does not have that. Not done.
- **PLE / n-gram host path**: no action needed. Disk vs RAM decode are equal (39.08 vs 38.81 ms/step profiled); per
  token there is one event sync on staging-set reuse and 3 small pinned uploads.

### Open

- hc_apply folded into the next GatedResidual site's gr_dots (96 launches per token, ~0.45 ms, the DS4 HC_FUSE pattern).
- Decode gaps 4.1 ms/token over 1498 launches. GDN in_proj / ba / conv / recurrent fusion is the launch-count lever.
- Vendored fla GDN chunked prefill: `recompute_w_u_fwd_kernel` spills 1608 B (autotuned; vendored upstream code).
- The GatedResidual mix is still ~180 GB/s against ~210-227 achievable: ~1 ms/token left.

## v1.5.3 sync (2026-09-29): upstream v1.5.0 -> v1.5.3 under the perf stack

Branch `merge/v1.5.3` = `perf/stack` (7eefda8) + `git merge v1.5.3` (133 upstream commits). Textual
conflicts: `setup.py` (kept the ROCm backend block, added upstream's `util/cuda_flags.py` loader for
the CUDA path) and `attention_fn/triton_paged.py` (taken **verbatim from upstream**: the fork's
in-file `_is_rocm` narrow-kv prefill tile moved to `rocm_py` as `EXL3_ROCM_PREFILL_HD128`, and its
decode-split comment is recorded here: `multi_processor_count` reports WGPs on RDNA -- gfx1151
returns 20 for 40 CUs -- so upstream's `2 * sms` split target is half what it assumes on NVIDIA;
doubling it was tried and reverted at v1.5.0, split counts 5..12 land within ~2% at ctx 1-4K,
`rocm_tools/bench_decode_splits.py`). README merged clean. No upstream `.py` file carries a ROCm
edit after the merge (`rocm_patches/BASE` = the merge commit).

### Siblings: drift before / upstream churn / drift after

| sibling | drift at v1.5.0 | upstream churn | method | drift at v1.5.3 |
|---|---|---|---|---|
| `quant/reconstruct_rdna.hip` | 12 | 55+/34- (half-integer K) | regenerate (sed, + `bits_k.cuh` include) | 14 |
| `cpu/moe_handoff_rdna.hip` | 6 | 12+/2- (aarch64 pause, pool priming) | regenerate (sed) | 6 |
| `quant/exl3_dq_rdna.hip.h` | 3 | 39+/2- (`dq8_half`) | regenerate (sed) | 3 |
| `quant/quantize_rdna.hip` | 92 | 64+ (`quantize_tiles_frac`) | function copied verbatim | 93 |
| `quant/quantize_tiles_frac_kernel_rdna.hip.h` | new | 233+ | regenerate (sed, codebook include) + `comp_units_rdna/quantize_tiles_frac_inst.hip` | 19 (header) |
| `quant/exl3_kernel_map_rdna.hip(.h)` | 335 / 180 | 71+/12-, 98+/13- | hand port: `half_k` template arg in `EXL3_GEMM_T_ARGS` (upstream position), `_H` instance/extern macros, `exl3_gemm_shape_smem` / `exl3_gemm_check_smem`, half-aware `exl3_gemm_smem_bytes(bits, shape, half_k)`; half rates shape-select as K + 1 like upstream. Upstream's constexpr smem functions over the CUDA shape table are **not** carried (the RDNA runtime accounting is the single source) | 376 / 265 |
| `quant/exl3_gemm_kernel_rdna.hip.h`, `exl3_gemm_inner_rdna.hip.h` | 64 / 1142 | 2+/2-, 20+/7- | hand port: `TILE_U16 = 16 * bits + 8 * half_k` in the B staging / strides / load map, `dq_dispatch<bits, cb, half_k>` | 64 / 1157 |
| `quant/comp_units_rdna/exl3_comp_unit_h{1,2,3}.hip` | new | new upstream units | the fp32/fp16 gemm + mgemm instances at 1.5 / 2.5 / 3.5 bpw | -- |
| `quant/exl3_gemm_rdna.hip` | 317 | 42+/27- | hand port: `half_k` from the tile width, `float K` at the mgemm boundary (`bits_from_K`), half-aware autotune / shape_compat / selection. The GEMV, multi-row and mgemv fast paths **decline half_k** (integer-K dot cores). Autotune hash: upstream mixes `half_k` first, always; here only when set, so integer-K keys equal v1.5.0's and the on-disk tune cache stays valid (a re-tune may pick another grid and with it another split-K order -- this alone broke bit-identity in the first merge build) | 346 |
| `quant/exl3_gemv_rdna.hip` | 989 | 25+/7- | `half_k` parameter, declines; direct `exl3_gemv` raises on a half tile | 1017 |
| `quant/exl3_moe_rdna.hip`, `exl3_moe_kernel_rdna.hip.h` | 199 / 275 | 19+/9-, 14+/9- | `float K`, runtime K passed in half-bit units (`k2_from_K`) as upstream; the non-pipelined inner switch gains the 3 / 5 / 7 (half, cb 2) cases; the pipelined mainloop is integer-K only, so any half rate forces `EXL3_ROCM_MOE_PIPE`'s old mainloop for that call | 212 / 282 |
| `quant/exl3_moe_coop_rdna.hip` (stub) | -- | float K signatures | signatures only | -- |
| `routing_rdna.hip` | 440 | 33+/13- | three-way merge (base v1.5.0): `gate_i8` / `gate_sb` args + upstream's deterministic router math (`exp_det`, `softplus_det`, `__fdiv_rn`, `__fsqrt_rn`). **`EXL3_ROCM_ROUTER_DET=0`** restores the v1.5.0 fast math (device flag, written once per device only when 0) | 500 |
| `hc_mix_rdna.hip` | 958 | 279+ | three-way merge: upstream's new GatedResidual decode pair (`gr_dots_kernel2` / `gr_finalize_kernel2`) merged verbatim; default stays on the RDNA rows dots (bit-identical to v1.5.0); **`EXL3_ROCM_GR_DOTS=0` now selects upstream v1.5.3's dispatch** (the new pair where eligible, else generic) | 1028 |
| `routing_gemm_rdna.hip`, `hc_mix_tiled_rdna.hip` | new | 397+, 590+ | generated (`rocm_tools/gen_det_siblings.py`, anchored edits): upstream's deterministic int8 tensor-core kernels (det_gemm.cuh PTX) compile out under the shim, but `routing_gemm_det_fits()` would still select them (HIP reports CC major 11). Siblings decline / raise; `det_quant_weight` / `det_math_test` stay upstream | 24 / 20 |

Shim: `cuda_shim/mma.h` (empty; det_gemm.cuh includes it but uses PTX only), `cudaDevAttrComputeCapability{Major,Minor}`, a real `__cvta_generic_to_shared` (generic -> LDS address space). `rocm_tools/hipcc_probe.sh`'s exclusion regex and `ROCM_EXCLUDE` gained `/routing_gemm.cu` and `/hc_mix_tiled.cu`. `rocm_tools/{gemm_check,gemm_coop_check,moe_inner_bench}.hip` updated for the new template argument.

Twins that did not change: every GEMV core (`exl3_gemv_kernel_rdna.hip.h`, tiles, multi-row, mgemv), `codebook_rdna.hip.h`, `exl3_moe_inner_rdna.hip.h`, `hgemm_rdna.hip`, `graph_rdna.hip`, `rope_rdna.hip`, `cuda_drv_rdna.cpp`.

### New upstream features on ROCm

| Feature | ROCm status |
|---|---|
| Fractional bitrates 1.5 / 2.5 / 3.5 bpw (mul1) | **Supported**: reconstruct, cooperative GEMM and mgemm (all row counts), fused MoE on the non-pipelined inner, quantizer (`quantize_tiles_frac`, `pack/unpack_trellis_frac`). No GEMV fast path (the RDNA cores decline; upstream has half GEMV instances). `rocm_py` keeps half-rate MoE layers off `moe_decode` (integer-K) onto the mgemm route. Verified by `rocm_tools/frac_check.py` (below); no fractional model exists to run end to end |
| int8 GEMV (`EXL3_INT8_GEMV`, half variants) | Unsupported, as before (stub; returns false, callers take the GEMM) |
| `exl3_moe_coop` (incl. new h1..h3 instances, shared-expert `sh_coop`) | Unsupported, as before (stub raises). `rocm_py` sets `block_sparse_mlp._moe_shared_coop = False` so the BC constructor does not build the unused shared-expert parameter block |
| `hgemm_f16acc` (+Ada tile) | Unsupported, as before (stub; hgemm_recon = hgemm) |
| Deterministic int8 router GEMM / tiled GatedResidual prefill (TP rank agreement) | Unsupported: declining siblings. Python already gates the tiled mix off under `torch.version.hip`; `rocm_py` stops `_gate_t` building the (unused) int8 router tables (`EXL3_ROCM_ROUTER_I8=1` restores) |
| Turing sm_75 paths, per-device smem budget ladders (#325) | Ladders active. torch on ROCm has no `shared_memory_per_block_optin`, so `smem.smem_limit()` would read 96 KB; `rocm_py` seeds it from `shared_memory_per_block` (64 KB on gfx1151; `EXL3_ROCM_SMEM_LIMIT=0`). The DSA decode / prefill proxies answer the ladder's compile-only probe (`kernel.run(warmup=True)`) for the MQA kernel they will actually launch, so the stock tile is picked and the launch is unchanged; `bc_attn`'s `BCKernelTooLarge` gate is mirrored in the buffer-op compile wrapper |
| `model.warmup()` in `model_init.init` (`-nw` skips) | Runs; the server's `-nwu` is now an alias of `-nw` (the old `--no_warmup` clashed with model_init's) and skips both warmups. Warmup's "rows 8" pass exposed a pre-existing perf/stack bug: `moe_decode` raised "shape exceeds the arrival counters" for bsz * top_k > 64 (Qwen3.8 top_k 10 at 7..8 rows, also reachable by a 7..8-token prefill chunk); `rocm_py` now sends such calls to the mgemm route |
| n-gram SAM corpus drafting (`-ngram_corpus`) | CPU (sam.cpp); builds. The server passes `ngram_corpus` through |
| DFlash2 (`dflash2.cu`), DFlash rework | Builds (plain CUDA, wave32-safe); see the Laguna DFlash run below |
| MiMo-V2 (asymmetric V head dim, SWA ring, vision, MTP) | Builds: attention.cpp V_DIM, Triton combine `V_DIM`, loader `fdequant` are all portable code; our BC buffer-op wrapper keys on argument names, which are unchanged. **Untested** (no model) |
| Kimi-Linear (KDA via the fla Triton kernels, MLA NoPE BC path) | Builds; the KDA kernels are the same vendored fla Triton family GLM-5.3 uses (no inline asm on AMD: `safe_dot`'s PTX wrapper is Blackwell-only). **Untested** (no model) |
| GatedResidual `FUSED_MAX_R` 32 -> 8 | Upstream lowered it because its tiled int8 path wins above 8 rows. That path does not exist on ROCm, so `rocm_py` keeps 32 (`EXL3_ROCM_GR_FUSED_R`); rows 9..32 stay on the fused pair |
| GDN/KDA prefill fp16 projections, token-major conv (`EXL3_GDN_PROJ_FP32`, `EXL3_GDN_CONV_TOKEN_MAJOR`) | Upstream defaults kept (numerics change, see below) |
| fla autotune configs for 64 KB-smem parts | `check_shared_mem('ada')` is false on gfx1151, so the new sm_75 configs join the autotune lists here too (autotuned by timing) |

### Numerics: what moved, and why

Every difference from `perf/stack` traced to an upstream v1.5.3 numerics change; with those reverted
by their switches the merge is **bit-identical** (`rocm_tools/decode_bitwise.py`, 48 greedy steps,
references saved on `perf/stack` 7eefda8 with its own build):

| model / prompt | default (upstream v1.5.3 semantics) | `EXL3_ROCM_ROUTER_DET=0 EXL3_GDN_PROJ_FP32=1 EXL3_GDN_CONV_TOKEN_MAJOR=0` |
|---|---|---|
| DS4, 19-token prompt | logits differ (max 0.0625), tokens identical | **bit-identical** |
| DS4, ~800-token prompt | logits differ (max 3.8), tokens identical | **bit-identical** |
| Qwen3.8, 21-token prompt | logits differ, tokens identical | **bit-identical** |
| Qwen3.8, ~800-token prompt | logits differ, tokens identical | differs (decode attention, below), tokens identical |
| Gemma-4-31B, 21-token prompt | **bit-identical** | -- |
| Gemma-4-31B, ~800-token prompt | differs from the first decode step, tokens identical | -- |

The three sources:
1. **Router math** (DS4, Qwen3.8 top-k weights): upstream's FMA-only `exp_det` / `softplus_det` and
   correctly rounded division / sqrt, for cross-architecture TP agreement. `EXL3_ROCM_ROUTER_DET=0`.
2. **GDN prefill** (Qwen3.8): fp16 projection output and the token-major conv read (421f590).
   Upstream's own `EXL3_GDN_PROJ_FP32=1` / `EXL3_GDN_CONV_TOKEN_MAJOR=0`.
3. **Paged decode attention** (Qwen3.8 full-attention layers, Gemma): the split count went from
   `cdiv(kv, 4 * block_n)` to `cdiv(kv, block_n)` (triton_paged.py and bc_attn.py) and the combine
   kernel now reduces `S_BLK` splits per step with `tl.sum` in `ROWS_SUB x D_SUB` sub-tiles.
   Kernel-level A/B of the old vs new `paged_attn_triton_decode` on identical inputs: identical
   at 1 split, ~1e-5 max abs difference at 4-10 splits (ctx 760 / 4000). No switch. Prefill
   attention is bit-identical old vs new at every tested shape (hd 256 / 512, past 0 / 512).

Consequence of 3 for MTP: plain and MTP greedy are no longer token-identical on Qwen3.8 (2/3
prompts diverge at tokens 34 / 102; `perf/stack` was 3/3 identical): the verify step (q_len 3)
and plain decode (q_len 1) now get different split counts. With the old `4 * block_n` split
formula in bc_attn.py (scratch patch) and the switches above, 3/3 identical again with the same
acceptance as `perf/stack`. DS4 (DSA attention) stays 3/3 token-identical with the defaults.

PPL (`bench/run_ppl.sh`, 100 x 2048): DS4 6.461734 (was 6.463333, -0.025%), Qwen3.8 4.739359
(was 4.751540, -0.26%), Gemma 18.633857 (unchanged, exact).

### Performance (same machine, same day, `bench/run_bench.py`; perf/stack via `--repo` on a 7eefda8 worktree with its own .so)

| | perf/stack | merge/v1.5.3 |
|---|---|---|
| DS4 pp512 | 350.8 | 346.1 (6 runs; 352.9 with `-nw`) |
| DS4 pp2048 | 518.1 | 513.2 (6 runs; 520.6 with `-nw`) |
| DS4 tg128 | 29.97 | 29.93 |
| DS4 regen 3000 / 8000 (ms) | 673.8 / 572.2 | 674.5 / 571.5 |
| DS4 MTP ndt 2 (acceptance) | 41.44, 41.50 (78%) | 40.50, 41.30 (78%) |
| Qwen3.8 `-ngr` pp512 / pp2048 | 617.8 / 842.9 | 630.3 / 862.2 |
| Qwen3.8 tg128 | 28.91 | 29.04 |
| Qwen3.8 regen 3000 / 8000 (ms) | 365.8 / 296.9 | 364.1 / 296.9 |
| Qwen3.8 MTP ndt 2 (acceptance) | 43.76, 43.93 (75%) | 44.12 (75%) |
| Gemma pp512 / tg128 | 248.0 / 8.27 | 247.6 / 8.34 |

The first merge suite run had DS4 pp512 318 / pp2048 491 with two slow runs of three (spread
10-16%); a 6-run rerun and a `-nw` run were steady, so it was a transient, not the code. The
residual ~1% prefill gap with `model.warmup()` on vs off is within noise but consistent across
both prefill sizes; `-nw` recovers it. A Qwen3.8 MTP `-ngr` run was OOM-killed once at the 112 GB
scope cap during load (system memory peaks ~122 GB vs ~121 GB on perf/stack; warmup transients);
the rerun passed.

### Validation

- rocm_py report: every patch applied, no `!! FAILED` (new lines: smem budget from
  sharedMemPerBlock 64 KB, router int8 tables not built, PREFILL_HD128, GR_FUSED_R 32).
- `bench/run_gates.sh` on f40f315 (gates5): **PASS** -- mgemv_check PASS, reconstruct_had ALL PASS,
  dsa_kernels ALL PASS, WMMA gate PASS, pytest 985 passed / 30 skipped (was 879 / 11 at 97063b3).
  One earlier run (gates4) had `test_dflash2.py::test_topk_cuda_matches_torch` fail once; it passed
  3/3 alone, in an isolated reproduction and in the final full run (see Open). The gate
  script now also symlinks `util/` (test_build_sam.py), accepts the loader's new `arena=` kwarg in
  the scratch copy of test_mla.py's FakeSTC (upstream test defect: all of test_mla / test_mla_dsa
  fail on CUDA too), and ignores test_routing_gemm_det.py / test_gr_mix_tiled.py (CUDA-only by
  design). `zstandard` installed into .venv10 (upstream `requirements_sam.txt`).
- `rocm_tools/frac_check.py`: 1.5 / 2.5 / 3.5 bpw reconstruct equals upstream's
  `unpack_trellis_frac` + `decode` per tile; exl3_gemm rows 1..32 and exl3_mgemm m 1 match
  reconstruct + matmul at 2.6e-6..1.1e-5 rel (integer K controls 7.7e-7..1.1e-5). The fused MoE
  kernel's half-rate path compiles but is not exercised (no fractional MoE tensors to build one from).
- Laguna-S-2.1 4bpw + DFlash drafter (`rocm_tools/bench_mtp.py -dm`): coherent; plain 34.67 /
  35.48 t/s, ndt 5 25.17 t/s acc 28.8%, ndt 3 29.52 t/s acc 44.2% (perf/stack: 34.67 / 35.42,
  24.32 @ 27.9%, 28.69 @ 44.2%).
- Server (`serve_ds4f.sh -port 8099`): model_init warmup + server warmup, one chat request, log
  line `150 generated (39.24 t/s) ... draft: 83/107 accepted`.

### Open

- The decode split / combine change (3 above) is the one upstream numerics change without a
  switch, and the one that breaks Qwen3.8's MTP = plain greedy identity. Restoring `4 * block_n`
  would need an upstream-file edit or a hook on `BCAttn`'s split helper; not done (maintainer call).
- `test_dflash2.py::test_topk_cuda_matches_torch` failed once in the full pytest run, never alone.
- MiMo-V2 and Kimi-Linear build but are untested on ROCm (no models).
