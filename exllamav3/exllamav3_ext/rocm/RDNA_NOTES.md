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
  `total_memory` is the GTT pool.
- **There is a 32 MiB Infinity Cache, and it will flatter any kernel benchmark
  whose working set fits.** Measured 653 GB/s at 16 MiB against 213 GB/s at
  64 MiB. Timing one weight tensor in a repeat loop measures cache, not DRAM --
  it overstated the EXL3 GEMV rate by 26% here, and a first pass at this
  concluded "the kernels are healthy" from exactly that error. Cycle a working
  set several times cache size.
- **Token generation is NOT memory-bound. It is bound by running a 16-row tile
  GEMM for one useful row.** Measured on Gemma-4-31B, decode at bsz=1:
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
| `gemm_check.hip`, `gemv_check.hip` | GEMM / GEMV kernels against a CPU reference |
| `gemm_coop_check.hip` | cooperative launch against the same work without it |
| `moe_ref32.py` | fused MoE **and** the per-expert path against an fp32 reference built from dequantized weights |
| `moe_check.py` | fused MoE against the per-expert path (two fp16 implementations — see its own caveats) |
| `nan_locate.py` | names the first module in a forward pass whose output goes non-finite |
| `bench_moe.py`, `bench_prefill_tiles.py`, `bench_decode_splits.py` | timing, median of repeats, flagging spreads above the noise floor |
| `hipcc_probe.sh` | per-file compile probe, without rdc |
