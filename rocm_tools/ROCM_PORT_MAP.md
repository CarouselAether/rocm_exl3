# ROCm port map: `rocm_exl3` (v0.0.29 base) → upstream v1.3.0

What the existing ROCm fork changed, what upstream has done since, and what each
item costs to bring forward. Written to answer "how much of my work survives?"

Measured against:
- fork: `rocm_exl3_legacy` @ `d2e6a19`, exllamav3 **v0.0.29**, ROCm 7.1-era toolchain
- upstream: `exllamav3` @ `0b9745c`, **v1.3.0**
- target: ROCm **7.2.4** / HIP 7.2.53211 / torch 2.13.0+rocm7.2, gfx1151

## Working tree layout

| Directory | Role |
|---|---|
| `rocm_exl3/` | **The deliverable.** Upstream v1.3.0 + `exllamav3_ext/rocm/`, branch `main`. `origin` → CarouselAether/rocm_exl3, `upstream` → turboderp-org/exllamav3 |
| `rocm_exl3_legacy/` | The original v0.0.29 fork. Kernel reference for Phase 4; not built |
| `exllamav3/` | Pristine upstream, `master`. Diff reference only |
| `tabbyAPI/` | Editable install for end-to-end testing |
| `rocm_docs/` | ISA + programming guide PDFs, plus 53 scraped markdown pages |
| `rocm_tools/` | `hipcc_probe.sh`, `phase2_attn_check.py`, `scrape_rocm_docs.py` |

The Python package stays named `exllamav3` so the build is a drop-in replacement
for TabbyAPI and anything else importing it. Only the repo is `rocm_exl3`.

**Verified state of the fork** (per its author): loads a model and produces
coherent tokens. Quantization was never confirmed working. So the inference path
(GEMV single-token, GEMM multi-token) has real evidence behind it; the quantize
path does not, and should be treated as unproven rather than regressed.

---

## Headline

Three separate things happened, and they have very different costs:

| Category | Fork effort | Survives to v1.3.0? |
|---|---|---|
| **A.** Include swaps / CUDA→HIP symbol bridging (~30 files) | large, tedious | **Obsoleted** — the shim does it, zero upstream edits |
| **B.** RDNA kernel rewrites (WMMA, GEMV, dequant) | very large | **Partly** — see per-file table |
| **C.** Python-level ROCm guards | small | Mostly, but several should be retested |

Category A is the good news: that work no longer needs to exist. Category B is
where the remaining effort is, and it is concentrated in one file.

---

## Part A — What the shim now does for free

The fork edited ~30 upstream `.cu`/`.cuh` files to swap `#include <cuda_fp16.h>`
etc. for `hip_compat.cuh`, plus `#ifdef USE_ROCM` branches. **None of that is
needed now.** `exllamav3_ext/rocm/` handles it from the build line:

- `-I rocm/cuda_shim` ahead of torch's include dir → upstream's `#include
  <cuda_fp16.h>`, `<cublas_v2.h>`, `<cooperative_groups.h>`, `<curand_kernel.h>`,
  `<ATen/cuda/CUDAContext.h>`, `<c10/cuda/CUDAGuard.h>` resolve to HIP forwarders
- `-include rocm/hip_compat.hip.h` → runtime/graph/driver type and symbol aliases,
  `__hmin2`/`__hmax2` for `__half2`, `__dp4a`, bf16 rounding-mode conversions

Result: upstream sources stay **byte-identical** to the CUDA branch, so future
rebases are fast-forwards instead of ~30 per-file merge conflicts.

Specific fork changes this retires:

| Fork change | Now handled by |
|---|---|
| `hip_compat.cuh` include swap in ~30 files | `-include` force-injection |
| `util.cuh` hipBLAS error-string overload | `cuda_shim/cublas_v2.h` (incl. the `CUBLAS_STATUS_LICENSE_ERROR` gap) |
| `graph.cuh` gated CUDA includes | `cuda_shim/` + graph aliases in `hip_compat` |
| `bindings.cpp` `#ifndef USE_ROCM` around `parallel/*` | `setup.py` `ROCM_EXCLUDE` |
| `arch_list.py` early-return on ROCm | still needed (Python side) |

**Current state:** 31 of 50 ROCm-built sources compile unmodified. The remaining
19 are shim gaps being closed, not kernel problems — see "Open shim gaps".

---

## Part B — RDNA kernel rewrites, and how far upstream moved

Upstream drift measured as changed lines between the fork's copy of each upstream
file and v1.3.0's.

| Fork file (lines) | Replaces upstream | Upstream drift | Port cost |
|---|---|---|---|
| `exl3_dq_rdna.hip.h` (317) | `exl3_dq.cuh` | **0 lines** | **Free.** Carries over as-is |
| `codebook_rdna.hip.h` (207) | `codebook.cuh` | 73 lines | Low — review the delta |
| `exl3_gemm_kernel_rdna.hip.h` (238) | `exl3_gemm_kernel.cuh` | 21 lines | Low |
| `exl3_gemm_inner_rdna.hip.h` (392) | `exl3_gemm_inner.cuh` | **501 of 733** | **High — effectively a re-derivation** |
| `exl3_gemv_kernel_rdna.hip.h` (523) | `exl3_gemv_kernel.cuh` | 46 lines | Moderate |
| `exl3_gemv_rdna.hip` (453) | `exl3_gemv.cu` (RDNA path) | — | Moderate, self-contained |
| `exl3_kernel_map_rdna.hip` (497) | `exl3_kernel_map.cu` | −165 (restructured) | Moderate |
| `exl3_moe_kernel_rdna.hip.h` (323) | `exl3_moe_kernel.cuh` | −3 lines | Low (but disabled at Python level) |
| `reconstruct_rdna.hip` (128) | `reconstruct.cu` | 25 lines | Low |
| `rdna_wmma.hip` (368) | *(new — no upstream equivalent)* | n/a | **Free.** Re-verified on 7.2.4 |

### `rdna_wmma.hip` is re-proven
`tests/wmma_smoke2.cpp` rebuilt and run on ROCm 7.2.4 / gfx1151: **256/256 cells
correct**, 0 transposed, 0 swapped. The `(B, A, C)` operand order and the
`row = lane % 16, col_base = (lane >= 16) ? 1 : 0` store layout both still hold.
This is the highest-risk piece of the port and it needs no rework.

### The real work is `exl3_gemm_inner`
501 of 733 lines changed upstream — the CUDA inner loop was substantially
rewritten between v0.0.29 and v1.3.0. The fork's WMMA version was derived from
the old one, so it cannot be copied forward; it has to be re-derived against the
new structure. This is the single largest Phase 4 item.

---

## Part C — Structural divergence: `comp_units`

This is the biggest surprise and is not visible from file counts alone.

**Fork `comp_units_rdna/`** — 16 files = 8 bitwidths × (`.hip` + `.hip.h`).
Instantiations are keyed on bitwidth only.

**Upstream v1.3.0 `comp_units/`** — 77 files (66 `.cu`), keyed on several axes:

| Family | Instantiations |
|---|---|
| EXL3 GEMM | 8 bitwidths × 3 codebooks (`_cb0/_cb1/_cb2`) = 24 |
| int8 GEMV | `coop_k1..k8` (8) + `sq_k1..k6` (6) = 14 |
| MoE | `k0_n128/n256 × cb1/cb2`, `k1..k8 × cb1/cb2` = 20 |
| quantize tiles | `k1..k8` = 8 |

So the fork covers **8 of ~66 instantiation slots, in one family only**. The
codebook axis (`cb0/cb1/cb2`), the int8 GEMV family, the MoE instantiations, and
`quantize_tiles` did not exist in the fork's base.

Consequence: `comp_units_rdna` is not "port 16 files" — it is a per-family
decision about which instantiations RDNA needs. `quantize_tiles_*` in particular
is on the quantization path, which the fork never validated.

---

## Part D — Python-level guards

| Fork guard | Recommendation |
|---|---|
| `attn.py` — skip `MultiLinear` K/V fusion on ROCm | Retest. Upstream attention was rewritten (`attention_fn/`), so the original justification may not apply |
| `mlp.py` — skip gate/up fusion, `BC_GatedMLP` | Retest on 7.2.4 |
| `block_sparse_mlp.py` — force `is_quantized = False` | Keep until `exl3_moe_kernel_rdna` is validated |
| `exl3.py` — `EXLLAMAV3_FORCE_TORCH_MODE` | Keep; useful diagnostic |
| `arch_list.py` — early-return on ROCm | Keep; still correct |

Also new in upstream and **not** in the fork: `triton_paged.py:1779` keys a
Blackwell tile config off `get_device_capability()[0] >= 10`. gfx1151 reports
`(11, 5)` — measured — so it misfires. Attention is still numerically correct
(18/18 in `rocm_tools/phase2_attn_check.py`), so this costs no accuracy. Measured impact on prefill throughput: see the tile-config section above -- the misdetection is in fact beneficial on gfx1151.

---

## Part E — Reversible / questionable fork decisions

| Item | Finding |
|---|---|
| `hgemm.cu` → `at::mm` swap | Fork's own probes showed `hipblasHgemm` works on 7.2.1. Likely cosmetic; revert once the port is stable |
| `rope.cu` warp-store guard | Already reverted by the fork — made output strictly worse. Do not reintroduce |
| `__shfl_*_sync` mask stripping | Still wanted, but the *reason* changed. HIP 7.x provides `_sync` variants (default-on since ROCm 7.0); the override is now a deliberate wave32 simplification, not a missing-function workaround |
| Tensor parallel disabled | Still correct — `parallel/` is CUDA IPC + inline PTX with no HIP port |

---

## Prefill tile config: the "Blackwell misdetection" is benign — do not naively fix

`triton_paged.py:1779` misfires on gfx1151 (capability reports `(11, 5)`, so
`>= 10` is true and prefill takes the Blackwell tile). Measured on gfx1151,
head_dim 128, 32/8 heads — `rocm_tools/bench_prefill_tiles.py`:

Advantage of `block_n=32` over `block_n=64`, three independent runs:

| q_len | ctx | run 1 | run 2 | run 3 | verdict |
|---|---|---|---|---|---|
| 512 | 0 | −7.5% | **+3.0%** | −7.3% | noise |
| 1024 | 0 | **+0.7%** | −5.2% | −4.7% | noise |
| 2048 | 0 | −9.3% | −8.8% | −10.1% | real |
| 4096 | 0 | −12.2% | −12.1% | −11.6% | **real** |
| 2048 | 2048 | −12.4% | −11.9% | −12.0% | **real** |

(negative = the narrow tile is faster)

The misdetected config is **~12% faster at long prefill**, reproducing to under
1% across runs. Correcting the capability check without re-tuning would make
prefill slower. Short prefill (≤1024) is noise — too short to be compute-bound.

Measured noise floor on this hardware: **4.7% spread, 1.6% stdev** over 8 repeats
of one config, with the first run reading high (clock ramp). Any claimed gain
below ~5% needs repeat measurement before it means anything — see the decode
split-K note below for one that did not survive.

Why: upstream sizes these tiles for "~100 KB of smem", a Hopper/Blackwell
assumption. RDNA 3.5 has 64 KB LDS per workgroup, so the narrow kv tile fits
where the wide one costs occupancy. Right answer, unrelated reason.

### The load-bearing parameter is `block_m / num_warps`, not `block_n`

Sweeping block_m × block_n × num_warps shows warp count dominates:

| block_m | num_warps | ratio | TF/s (q=2048, ctx=2048) |
|---|---|---|---|
| 128 | 8 | **16** | 15.9–16.3 |
| 64 | 4 | **16** | 9.9–10.4 |
| 64 | 8 | 8 | 6.3–8.9 |
| 128 | 4 | 32 | 4.2–10.4 |

Every good config has `block_m / num_warps == 16` — one WMMA 16×16 fragment row
per warp. Off-ratio configs lose up to 4x. Anyone re-tuning tiles for RDNA must
hold this ratio; changing `block_m` alone is a trap.

Best measured: `(128, 16, 8)` at 16.29 TF/s with context, `(64, 64, 4)` at
13.27 TF/s without. Upstream's accidental default `(128, 32, 8)` lands at 15.94
and 13.12 — within ~2% of optimal in both cases.

**Applied:** `triton_paged.py` now selects the narrow tile explicitly via
`_is_rocm`, rather than inheriting it from a misfiring capability test, so an
upstream change to the Blackwell heuristic cannot silently regress RDNA. Behaviour
is unchanged on gfx1151; the point is that it is now deliberate.

Further tuning upside is ~2% — not worth chasing. The real compute-bound headroom
is in the GEMM kernels, not attention tiles.

### Decode split-K: investigated, NOT changed

`multi_processor_count` reports **WGPs** on ROCm, not CUs — gfx1151 returns 20 for
a 40-CU part — so the decode split target (`2 * count`) is half what the same code
assumes on NVIDIA, and decode under-splits when `programs` is small (bsz=1).

Doubling it looked like a 6–7% win at ctx 1024–4096. It did not survive:

- repeat measurement put the decode step's run-to-run spread at 4.7%
- a second full sweep showed the "fixed" version *slower* at ctx=1024
- isolating split counts 4..16 showed everything in 5..12 landing within ~2%

So the observation is real but the lever is not. Reverted, with a note in the
source. `rocm_tools/bench_decode_splits.py` re-checks it on other parts, where the
CU/WGP ratio or CU count may make it matter.

## The shim boundary: inline PTX

Final probe: **43 of 50 sources compile unmodified.** The 7 failures are not a
long tail — they are two categories, and one of them is principled:

**6 files: inline PTX.** `ptx.cuh` carries 22 asm blocks of NVIDIA ISA. Not
shimmable in principle. Every failing file pulls it in:

| File | Fork equivalent |
|---|---|
| `quant/exl3_gemm.cu` | `exl3_gemm_inner_rdna` / `_kernel_rdna` |
| `quant/exl3_gemv.cu` | `exl3_gemv_rdna` |
| `quant/exl3_kernel_map.cu` | `exl3_kernel_map_rdna` |
| `quant/reconstruct.cu` | `reconstruct_rdna` |
| `quant/exl3_gemv_int8.cu` | **none — new in v1.3.0** |
| `cpu/moe_handoff.cu` | **none — new in v1.3.0** |

The PTX boundary lands exactly on the files the fork rewrote. Nothing rewritten
was work the shim could have done; nothing shimmable still needs a rewrite.

**1 file: `rope.cu`** — `((half2*)p)[t] = {}` is ambiguous on HIP with or without
`__HIP_NO_HALF_OPERATORS__`. Genuinely unreachable from a shim.

### What the PTX actually needs, by tier

| Tier | Ops | Difficulty |
|---|---|---|
| Tensor core | `mma.sync.aligned.m8n8k4`, `m16n8k16` (f32 and f16 accum), `ldmatrix.sync.aligned.m8n8.x4` | **Solved** — `rdna_wmma.hip`, re-verified on 7.2.4 |
| Memory ordering | `ld.global.acquire`, `st.global.release`, `red.relaxed.global.add`, `st.global.wt` | Mechanical — `__hip_atomic_*`, same pattern as the `cuda::atomic_ref` shim |
| Bit/arith | `bfe.u64`, `shf.r.wrap.b32`, `mul.hi.u32`, `mul.lo.u32` | Mechanical |
| Async copy | `cp.async.*`, `cp.async.wait_group` | **The open question.** RDNA 3.5 has no direct equivalent |

`cp.async` is likely a large part of why `exl3_gemm_inner` is the expensive file.

### gfx1151 has hardware int8 dot product — `exl3_gemv_int8` is portable

`exl3_gemv_int8_kernel.cuh:60` wraps exactly one PTX instruction:
`dp4a.u32.s32` (4×uint8 · 4×int8 → int32 accumulate). Probed availability on
gfx1151:

| Builtin | Instruction | gfx1151 |
|---|---|---|
| `__builtin_amdgcn_sudot4` | `v_dot4_i32_iu8` (dot8) | **available** |
| `__builtin_amdgcn_udot4` | `v_dot4_u32_u8` (dot7) | **available** |
| `__builtin_amdgcn_fdot2` | `v_dot2_f32_f16` (dot10) | **available** |
| `__builtin_amdgcn_sdot4` | `v_dot4_i32_i8` (dot1) | **NOT available** |
| `__builtin_amdgcn_udot2` | `v_dot2_i32_i16` (dot2) | **NOT available** |

`sudot4(false, a, true, b, c, false)` was verified numerically against a scalar
reference across sign boundaries (0x80 = −128, 0xFF = 255u/−1s): **exact match**.
Note the trap: the obvious choice `sdot4` is *absent* on gfx1151; the mixed-sign
`sudot4` is the correct mapping.

This matters because int8 GEMV is the single-token decode path — the one that
most affects interactive throughput on a bandwidth-limited APU.

## Open shim gaps (Phase 3, in progress)

19 of 50 sources still failing, in clusters:

| Gap | Files | Fix |
|---|---|---|
| `cuda/atomic` (libcu++) | 6 | Needs a minimal `cuda::atomic_ref` shim over HIP atomics |
| `curandStatePhilox4_32_10_t` etc. | 4 | Name aliases onto hipRAND |
| `cudaStreamNonBlocking`, `cudaErrorHostMemoryAlreadyRegistered` | 3 | Enum aliases |
| `hipFuncSetAttribute` signature | 2 | HIP takes a typed function pointer where CUDA takes `const void*` |
| `cudaLaunchCooperativeKernel` | 1 | Alias |
| `rsqrtf` in `attention.cu` | 1 | Under investigation |
| `lm_clamp_` in `q_cache.cu` | 1 | Likely half2 operator resolution |
| `half2 x = {}` in `rope.cu:141` | 1 | **Not shimmable** — ambiguous on HIP either way. Needs a one-token upstream edit or a build-time patch |

---

## Suggested order for Phase 4

1. `exl3_dq_rdna` — free, 0 drift
2. `rdna_wmma` — free, re-verified
3. `codebook_rdna`, `exl3_gemm_kernel_rdna`, `reconstruct_rdna` — low drift
4. `exl3_gemv_rdna` + `exl3_gemv_kernel_rdna` — moderate; this is the path with
   actual evidence behind it (single-token inference worked)
5. `exl3_kernel_map_rdna` — restructured upstream
6. `exl3_gemm_inner_rdna` — **the big one**, re-derive against v1.3.0
7. `comp_units_rdna` — decide per family; codebook axis is new
8. MoE + quantize_tiles — unproven territory, lowest confidence

Validate in the same order the fork's evidence supports: model load → coherent
tokens → perplexity → quantization. The last of those has never worked and
should not be assumed.
