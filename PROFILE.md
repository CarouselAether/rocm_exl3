# PROFILE.md: Phase 0 findings, DeepSeek-V4-Flash on gfx1151

Phase 0 of `exlproject/PLAN.md`. **No kernel or runtime changes were made.** Everything below is measured on
branch `perf/phase0`:
- base commit 97063b3 (rocm-10), with the Phase 0 tooling in 67d25b5 and later;
- torch 2.15.0.dev20260926+rocm10.0, HIP 7.15.26333, ROCm SDK 10.0.0 (pip);
- CPU boost off, swap off, GPU perf level `auto` except during counter runs.

Raw data is in `exlproject/logs/` and `bench/results/`. A running log is in `exlproject/NOTE.md`.

**STOP: waiting for maintainer review before Phase 1.**

---

## 1. Summary

| | Baseline | PLAN target |
|---|---|---|
| Decode tg128 (ctx 512) | **18.02 t/s** (55.5 ms/token) | 30+ |
| Decode at 16K ctx | 17.33 t/s | |
| MTP ndt=2 | 19.78 t/s (80% acceptance) | |
| pp512 | **112.8 t/s** | 600+ |
| pp2048 | 178.5 t/s | |

PLAN §1 assumed ~400 t/s pp512. No measurement shows that, and the harness reproduces the server's own log lines (§2). Every target below is scaled from the measured baseline.

**Decode is GPU-bound** (91.9% busy). The quantized GEMVs are *instruction-count bound per weight*: every EXL3 GEMV kernel moves about the same ~120 G weights/s whatever its bitrate. They are not bandwidth or latency bound. The second-largest decode item is a spilling DSA attention kernel with a fixed cost of 8.5 ms/token at any context length.

**Prefill is dominated by the fused MoE kernel at ~44 GB/s**, which is 68% of pp512. After that come the spilling DSA prefill kernel and hipBLAS fp32-output GEMMs that run without matrix cores. On top of all that, the generator's recurrent-checkpoint policy adds a whole extra expert-weight stream to every prompt (Q-12).

---

## 2. Harness, gates, baselines

- **Bench** (`bench/run_bench.py`):
  - loads through `model_init.init` with the server's arguments (`-cs 65536`) and uses the server's own speed formulas;
  - 1 warmup + 3 timed runs, median reported; spreads were 0.1–1.2%;
  - records the commit, stack, boost state, and clock/power samples;
  - live-server cross-check: server 17.85 t/s decode and 151 t/s prefill on 921 tokens, both on the harness curves.
- **Gates** (`bench/run_gates.sh`):
  - `mgemv_check` PASS (including bit-exact 2-token);
  - `test_reconstruct_had` and `test_dsa_kernels` ALL PASS;
  - pytest **879 passed**; 8 files excluded with reasons (CUDA-only by design, or needs 2 GPUs / upstream stub models);
  - WMMA gate **27/27** (see §9).
- **Perplexity** (`bench/run_ppl.sh`, wikitext2, 100 × 2048):

| Model | PPL |
|---|---|
| DS4 | **6.461586** |
| Qwen3.8 | **4.747702** (identical with `-ngr`) |
| Gemma-4-31B | **18.635452** (`-gp`) |

  Gemma needs BOS: without `-gp`, `ppl.py` gives 1545.
- **Portability**: `hipcc_probe --all` compiles 118/118 for gfx1151, gfx1100, gfx1101, gfx1200 and gfx1201. The probe needed a C++20 fix for torch ≥ 2.14.
- **Stable reference**: torch 2.13 stable gave pp512 113.2, pp2048 181.4, tg128 17.86. The nightly is performance-neutral.

---

## 3. Environment (§0.2)

- **Clocks and power:**
  - sclk holds 2.87–2.90 GHz through every benchmark;
  - the GPU sustains about 120 W package power (139 W short bursts);
  - no clock sag was seen across runs.
- **Thermal:**
  - One THERMTRIP power-off (reset reason `0x00200a00`) happened with CPU boost on.
  - With boost off, DS4 runs peak at 76–86 °C.
  - The firmware regulates Tctl at about 98 °C. All GPU jobs run under `exlproject/thermal_guard.py` (kill at 99.5 °C). See Q-11.
- **Memory:**
  - Carve-out VRAM is **512 MB**; all weights live in **GTT** (110 GiB limit via `ttm.pages_limit`).
  - Allocations are ordinary coarse-grained hipMalloc; no fine-grained or host-coherent buffers were found on the hot paths.
- **Chunking, a config finding.** `-chunk_size` (in both the server and model_init) only sizes load-time buffers. The Generator chunks at its own `max_chunk_size` (default 2048, `generator.py:34`), which the server never sets.

---

## 4. Decode profile (§0.5)

rocprofv3 1.3.5, `--selected-regions --kernel-trace --hip-trace --marker-trace --memory-copy-trace`, 32 steps at ctx 512 (`logs/prof/ds4_decode`).

| | per token |
|---|---|
| Wall | 57.8 ms (profiler overhead included; 55.5 unprofiled) |
| **GPU busy** | **53.2 ms (91.9%)** |
| Kernels | 1735 |
| Idle gaps | 1735, 4.6 ms total, mean 2.7 µs (graphs on) |
| Host syncs | 1 `hipDeviceSynchronize` per token (sampling); **no per-layer syncs, no device→host copies** |

**Weight floor:** routed 1.62 + attention/shared 3.26 + head 0.40 + router 0.09 ≈ **5.4 GB/token**, about **25.6 ms** at 210 GB/s.

**Top kernels** (µs per token; share of busy time):

| # | µs/token | share | calls | avg µs | kernel | achieved |
|---|---|---|---|---|---|---|
| 1 | 10,733 | 20.2% | 86 | 125 | routed gate/up `exl3_gemv_mr_dot_multi<2,…,1>` (K=2) | ~101 GB/s |
| 2 | 8,515 | 16.0% | 43 | 198 | **`_dsa_attn_split_kernel`**: 256 VGPR, **1840 B scratch** | fixed cost |
| 3 | 6,413 | 12.1% | 86 | 75 | `exl3_gemv_mr_dot_multi<4,…,1>` (K=4) | ~163 GB/s class |
| 4 | 5,825 | 11.0% | 43 | 135 | routed down, weighted `exl3_mgemv_dot_kernel_splitk<2,true,2,4>` | ~93 GB/s |
| 5 | 5,793 | 10.9% | 43 | 135 | `exl3_gemv_mr_dot_single<4,true,…>` | |
| 6 | 4,282 | 8.1% | 43 | 100 | `exl3_gemv_mr_dot_single<4,false,2,4,1>` | |
| 7 | 2,339 | 4.4% | 43 | 54 | `exl3_gemv_mr_dot_multi<4,true,…>` | |
| 8 | 1,950 | 3.7% | 43 | 45 | `routing_gemv_kernel` | ~46 GB/s (roofline ~10 µs) |
| 9 | 1,897 | 3.6% | 1 | 1897 | lm_head `exl3_gemv_dot_kernel<6,…>` | ~210 GB/s (roofline) |
| 10 | 1,199 | 2.3% | 43 | 28 | `exl3_gemv_mr_dot_single<4,true,2,4,1>` | |
| 11 | 837 | 1.6% | 86 | 10 | `hc_mix_partials_kernel` | |
| 12 | 649 | 1.2% | 215 | 3 | `exl3_gemv_mr_had_in_multi` | |
| 13 | 446 | 0.8% | 86 | 5 | `hc_mix_finalize_kernel` | |
| 14 | 376 | 0.7% | 43 | 9 | `_dsa_attn_combine_kernel` | |
| 15 | 297 | 0.6% | 129 | 2 | `rms_norm_kernel` | |

By class:

| Class | ms/token |
|---|---|
| Quantized GEMVs | ~38.4 (routed K=2 16.5, dense K=4 ~20.0, head 1.9) |
| DSA decode attention | 8.9 |
| Router | 1.95 |
| Hyper-connections and norms | ~1.6 |
| Gaps | 4.6 |

**Context and MTP:**
- At **16K context**: 54.6 ms busy; `_dsa_attn_split_kernel` costs 204 µs per call, **the same as at 512**. The DSA cost is fixed overhead, not context-proportional.
- **MTP** (ndt=2): the verify step is 57.2 ms busy for ~1.3 tokens. Routed K=2 is 36% and DSA split 17% (363 µs per call).

---

## 5. Kernel-level counters (§0.6)

`bench/run_counters.sh`: 6 `--pmc` passes at `profile_standard` clocks. Absolute GB/s reads low under those clocks, so compare rows against each other. Data: `logs/prof/ds4_decode_pmc`, `ds4_prefill_pmc`.

| Kernel | GB/s (stable clk) | Weights/s | MemUnitBusy | Occupancy | VALU/wave-cycle | LDS conflict | Limiter |
|---|---|---|---|---|---|---|---|
| **Decode:** routed gate/up K=2 | 32 | **~128 G** | 92% | 89% | 0.038 | 0 | per-tile instruction count |
| **Decode:** dense K=4 (4 variants) | 56–60 | **~115–120 G** | 84–93% | 70–88% | ~0.04 | 0 | same |
| **Decode:** lm_head K=6 | 88 | ~117 G | 95% | 91% | 0.035 | 0 | same |
| **Decode:** `_dsa_attn_split_kernel` | 108 (mostly scratch) | n/a | 84% | **22.6%** | 0.018 | 0.38 | **VGPRs** (13.6 M VGPR-full stalls per dispatch), spills |
| **Prefill:** `_dsa_attn_kernel` | 153 (mostly scratch; 78 B/VALU) | n/a | 91% | **17.3%** | 0.014 | 0.43 | VGPRs, spills |
| **Prefill:** fp32-out hipBLAS `Cijk_…HSS…MT64x32x8` | 11 | n/a | 91% | 86% | 0.040 | 0.10 | **VALU FMA: no matrix cores** |
| **Prefill:** `reconstruct_kernel<4,2>` | 27 | n/a | 78% | 77% | 0.025 | 0.51 | minor |
| `routing_gemv_kernel` | 31 | n/a | 70% | **14.7%** | 0.015 | 0 | grid (32 blocks) |

The fused MoE kernel (`exl3_moe_kernel`) was **excluded** from PMC collection. Its co-resident design deadlocks under counter serialization (RDNA_NOTES). Its numbers come from the trace: **36.9 ms per layer-forward on ~1.61 GB ≈ 44 GB/s**, VGPR 248, 116 B scratch, one 512-thread block per WGP.

**Roofline placement:**
- The EXL3 GEMVs sit under both roofs.
- They have high occupancy and a busy memory unit, yet low DRAM bandwidth and low VALU issue. So they are limited by *instructions per weight* (address, load and decode per 16×16 tile), not by bytes or latency. K=2 gets half the GB/s of K=4 only because each byte carries twice the weights.
- The DSA kernels are register/spill-bound.
- The fp32-out GEMM is compute-bound on VALU because it misses WMMA.

### Resource table (hot kernels; waves/SIMD approximate, wave32, 1536 VGPR slots per SIMD)

| Kernel | VGPR | SGPR | LDS B | Scratch B/lane | Waves/SIMD | Note |
|---|---|---|---|---|---|---|
| `exl3_gemv_mr_dot_multi<2|4,…>` | 24 | 128 | 0 | 0 | 16 | |
| `exl3_gemv_mr_dot_single<4,…>` | 24 | 128 | 0 | 0 | 16 | |
| `exl3_mgemv_dot_kernel_splitk<2,true,2,4>` | 40 | 128 | 0 | 0 | 16 | |
| `exl3_gemv_dot_kernel<6,…>` (head) | 40 | 128 | 0 | 0 | 16 | |
| **`_dsa_attn_split_kernel`** (decode) | **256** | 128 | 0 | **1840** | 6 | **SPILL (top priority)** |
| **`_dsa_attn_kernel`** (prefill) | **256** | 128 | 0 | **5176** | 6 | **SPILL (top priority)** |
| `exl3_moe_kernel<2,256,2,16>` | 248 | 128 | dyn (64 KB requested) | 116 (prologue) | 4 (1 block/WGP) | co-resident |
| hipBLAS `…HSS…MT64x32x8` | 48 | 128 | 1536 | 0 | 16 | no WMMA |
| hipBLAS `…HHS…MT128x128x32_MI16x16x16` | 256 | 128 | 50176 | 0 | ≤6 | WMMA |
| `routing_gemv_kernel` | 16 | 128 | 0 | 0 | 16 | grid-limited |
| `hc_mix_partials_kernel` | 96 | 128 | 512 | 0 | 16 | |

*(SGPR reads 128 for every dispatch record; it is the allocation granule, not the use.)*

---

## 6. Prefill profile (§0.7)

| | pp512 | pp2048 |
|---|---|---|
| Wall (region) | 4.71 s | 11.65 s |
| GPU busy | 97.3% | 98.7% |
| Forwards | **2** (256 + 255) | **2** (1792 + 255) |
| Fused MoE `exl3_moe_kernel` | **3.10 s (68%)** | 4.66 s (40%) |
| DSA attention `_dsa_attn_kernel` (spilling) | 0.46 s (10%) | **2.07 s (18%)** |
| fp32-out hipBLAS GEMMs (no WMMA) | 0.42 s (9%) | **2.02 s (17.6%)** |
| reconstruct + fp16 WMMA GEMMs + Hadamard | ~0.25 s | ~0.8 s |
| Routing and other | small | small |
| Host syncs | ~5 per layer per forward (`hipMemcpyWithStream`, `hipStreamSynchronize`); <3% at 97–99% busy | same |

**Two forwards per prompt.** On recurrent-state models, `generator/job.py:1305-1313` prefills the last partial page in a separate forward. On DS4 each forward re-streams nearly all expert weights (~1.55 s of MoE at today's speed). This is logged as **Q-12**.

**fp32-output GEMM micro-benchmark** (`ext.hgemm`, same shapes):

| M × K × N | fp16 out | fp32 out | Slowdown |
|---|---|---|---|
| 1792 × 8192 × 4096 | 5.56 ms (21.7 TF) | 21.14 ms (5.7 TF) | **3.8×** |
| 1792 × 2048 × 4096 | 0.85 ms (35.3 TF) | 4.65 ms (6.5 TF) | **5.5×** |
| 255 × 8192 × 4096 | 0.59 ms | 2.93 ms | 4.9× |
| 160 × 2048 × 4096 | 0.14 ms | 0.38 ms | 2.7× |

These are the down projections into the fp32 residual: 8192→4096, 2048→4096, and hot experts with >128 rows.

**Chunk sweep, pp2048** (Generator `max_chunk_size`; `bench/run_bench.py --gen_chunk`):

| Chunk | pp2048 |
|---|---|
| 512 | 136.9 t/s |
| 1024 | 163.4 t/s |
| 2048 | **181.1 t/s** |

Gains are steep: +32% from 512 to 2048. Per PLAN's reading, that means prefill is **weight-streaming bound**: each forward streams the whole expert set at ~44 GB/s, so fewer and larger forwards win. This agrees with the MoE finding and with Q-12. 2048 is already the default. Going above 2048 needs `max_chunk_size` raised in both the Generator and the load-time buffers, and was not tested.

---

## 7. Classification (§0.8)

| Observation | Diagnosis | First move |
|---|---|---|
| Decode: busy 92% of wall; GEMVs at high occupancy and MemUnitBusy, low GB/s, low VALU issue, **constant weights/s across K** | **Not in PLAN's table.** Instruction-issue bound per weight in the load/address/decode path. Not latency, not occupancy, not bandwidth. | Fewer instructions per weight: multi-tile per wave with wide loads, plus decode and address VALU cleanup |
| Decode: DSA split kernel 256 VGPR, spills, 22% occupancy, context-independent cost | Register/spill bound, fixed overhead | Re-tile or rewrite the decode DSA kernel to fit registers (MQA-aware: read K/V once for all heads) |
| Prefill: fused MoE at ~44 GB/s, steep chunk curve | Weight-streaming latency bound (serialized per-k-tile drain, 1 block per WGP) | Pipeline the MoE mainloop (CODE_SCAN #2) |
| Gaps: 4.6 ms/token, 2.7 µs mean, graphs on | Launch/dispatch gaps, about 8% of decode | Fusion (later; small) |

---

## 8. Ranked optimization plan (expected gains are estimates from the data above)

Decode arithmetic is relative to 55.5 ms/token; prefill arithmetic is relative to measured GPU seconds.

| Rank | Branch | Change | Evidence | Expected gain | Scope |
|---|---|---|---|---|---|
| **1** | `opt/moe-mainloop` | Fused MoE prefill kernel. Stop draining all memory every k-tile: register-staged prefetch, spread loads over all threads, `global_load` instead of FLAT (force-inline or address-space casts). | 68% of pp512 at ~44 GB/s; CODE_SCAN #2 ISA | MoE ~3× faster. **pp512 113 → ~180–200; pp2048 178 → ~250** | Sibling (`rocm/quant/exl3_gemm_inner_rdna.hip.h`, `exl3_moe_kernel_rdna.hip.h`); shared with `exl3_gemm` m≤144, so revalidate gemm/mgemv |
| **2** | `opt/gemv-tiles` | EXL3 GEMV: several adjacent N-tiles per wave with wider (128 B) loads and shared A loads; hoisted SGPR address math, `global_load` instead of FLAT, uniform split-K loop (`readfirstlane`). | All GEMVs at ~120 G weights/s; K=2 at half the GB/s of K=4 | Weights/s ~1.5× across ~38 ms of GEMVs, about −12 ms/token. **Decode 18 → ~22–23 t/s** (routed and dense both benefit) | Sibling (`rocm/quant/exl3_gemv_*`, `exl3_dq_rdna.hip.h`); bit-identical intended (`mgemv_bitwise`, `decode_bitwise`) |
| **3** | `opt/dsa-decode` | Decode DSA split kernel: remove the spill (fewer VGPRs: head block / accumulator split), fewer splits at short context, or an MQA-specialized kernel. | 8.5 ms/token, 256 VGPR + 1840 B scratch, 22% occupancy, context-independent | 8.5 → ~1.5–2 ms, about −6.5 ms/token, **+12–13% decode**; MTP gains more (363 µs per call) | Py-hook for tuning (Triton); a new kernel would be a Triton sibling |
| **4** | `opt/fp32-gemm` | Route fp32-output reconstructed GEMMs to a WMMA path: hipBLASLt with fp32 D, fp16 output plus a widening add, or an fp32-epilogue WMMA kernel. Pick per the numerics check. | 9–18% of prefill; 3.8–5.5× slower than fp16 out | **pp2048 +15%, pp512 +7%** | Python hook (`modules/quant/exl3.py:198`) + sibling if a kernel is needed; numerics note (fp16 intermediate rounding) → PPL gate |
| **5** | `opt/dsa-prefill` | Prefill DSA attention: tune BLOCK_H/BLOCK_N/warps/stages to drop the 5176 B spill (compile-only sweep first). | 10% pp512, 18% pp2048, 17% occupancy | **pp2048 +8–12%** | Py-hook (call arguments; CODE_SCAN #1) |
| 6 | `opt/router-grid` | Router GEMV: a bigger grid, wider loads. | 45 µs per call vs ~10 µs roofline, 14.7% occupancy | −1.4 ms/token, **+2.5% decode** | Sibling+exclude (`routing.cu` is upstream) |
| 7 | (policy) | Q-12: last-page prefill forward. | Extra full expert stream per prompt | pp512 +20–35% today; shrinks after #1 | Py-hook, **maintainer decision** |
| 8 | `opt/decode-fusion` | Fuse hyper-connection and norm kernels, attention small ops. | 4.6 ms gaps, ~1735 launches/token | −1–2 ms/token, +2–3% | Sibling+exclude |
| 9 | (server) | Warmup request at server start. | First requests 10.9 and 6.8 t/s (JIT/graph capture) | UX only | `rocm_tools/exl3_server` |

**Combined estimate:**
- **Decode:** 55.5 − 12 (#2) − 6.5 (#3) − 1.4 (#6) − 1.5 (#8) ≈ **34 ms → ~29–30 t/s**. That reaches the PLAN's 30 t/s target; MTP adds on top.
- **Prefill:** pp512 ≈ 4.6 s − 2.0 (#1) − 0.3 (#4) − 0.2 (#5) ≈ 2.0 s → **~250 t/s**, and ~350 with Q-12 option B. pp2048 ≈ 11.5 − 3.1 − 1.5 − 1.0 ≈ 5.9 s → **~350 t/s**.
- The PLAN's 600 t/s pp512 is not reachable from these items. It would need the MoE near the bandwidth roof *and* a single forward per prompt, and it is still bounded by the ~1.6 GB/layer expert stream per forward.

**Suggested Phase 1 order:** #1 and #2 (largest), then #3 (cheap, big for decode and MTP), then #4 and #5 (cheap prefill wins), then the rest. Items #2 and #3 are independent and could run on parallel branches.

### CODE_SCAN cross-check

| Scan item | Verdict |
|---|---|
| #0 MTP MoE loop | Consistent (MTP verify step dominated by routed K=2); not separately measured |
| #1 DSA prefill spills | **Confirmed** (5176 B scratch, 17% occupancy, 10–18%) |
| #2 MoE mainloop drain | **Confirmed** as the top prefill item (~44 GB/s) |
| #3 K=2 bytes in flight | **Reframed**: the limit is per-weight instructions (constant weights/s across K), so the fix is the same shape (multi-tile per wave) and applies to K=4 too |
| #4 K=2 VALU overhead | Part of #2 above; VALU issue per wave is low, so address/load instruction count matters as much as decode math |
| #5 MOE_SMS_PER_EXPERT | Not measured (MoE excluded from PMC); revisit with #1 |
| #6 Weighted down on the LDS-prologue kernel | Consistent (down K=2 at ~93 GB/s, slightly below gate/up) |
| #8 DSA N_SPLITS | **Much bigger than estimated**: 16% of decode, not 1–2% |
| #12 o_proj 8× wo_a | Confirmed present (10 hgemm_recon per layer per forward); small |
| #15 router grid | **Confirmed** (14.7% occupancy) |
| #16 prefill host syncs | Present but <3% |
| **New:** fp32-out GEMM without WMMA | Not in the scan; 9–18% of prefill |
| **New:** last-page extra forward | Not in the scan; Q-12 |

### RDNA 3.5 hunt list (§5)

| Item | Applies? |
|---|---|
| CUDA idioms (shuffles via `ds_bpermute`) | Present, not hot |
| VGPR pressure | **Yes**: DSA kernels (256 + spill), MoE (248) |
| Missed packed math / dual issue in decode | Part of rank #2 (instruction count per weight) |
| DPP/permlane instead of LDS | Minor |
| LDS bank conflicts | 0.4–0.5 ratio in DSA and reconstruct kernels; 0 in the GEMVs; MoE unmeasured |
| Global load width / coalescing | **Yes**: rank #2 |
| Grids too small for 40 CUs | **Yes**: router GEMV (14.7% occupancy), hc kernels |

---

## 9. WMMA (§0.3)

- `docs/RDNA_WMMA.md` documents every layout rule the code uses, with file:line citations.
- **Gate:** `rocm_tools/wmma_gate.hip` runs 27 cases through the production `rdna_wmma.hip.h` helpers:
  - int8 is exact against the math;
  - f16/bf16 are checked within a measured bound;
  - **every case is bit-exact against `rocm_tools/wmma_gate.golden`**.
  - Result: PASS on gfx1151. It compiles for gfx1100 and gfx1201 (the gfx12 route is the documented trap). It is wired into `bench/run_gates.sh`.
- **New hardware fact:** gfx11 WMMA fp32 accumulation from f16/bf16 inputs is *not* IEEE-exact, even when every product and the exact sum are representable. The errors are 1–5 ulp of the largest intermediate: small-term alignment truncation plus a mixed-sign bias of about −0.5 LSB (§8 of RDNA_WMMA.md). Consequences:
  - bit-exactness against math or NVIDIA is not a valid oracle for f16 WMMA paths;
  - golden-output regression is the right gate.
- Measured once on gfx1151: lanes 16–31 feed the odd output columns, so the lane-replication rule is required.

---

## 10. Open questions for the maintainer

- **Q-12**: keep or relax the recurrent last-page prefill forward for DS4 (§6)?
- Approve the Phase 1 order in §8. Items #4 and #6 touch hipBLAS routing and the upstream `routing.cu`, via a Python hook and a sibling+exclude respectively.
