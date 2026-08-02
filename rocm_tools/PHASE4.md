# Phase 4 handoff — RDNA kernel port

Start-here doc for continuing the ROCm port in a fresh session. Phases 1–3 are
done and committed; this covers what is true, what is next, and what must not be
re-derived from scratch.

Read alongside [ROCM_PORT_MAP.md](ROCM_PORT_MAP.md) (fork-vs-upstream analysis)
and [../exllamav3/exllamav3_ext/rocm/README.md](../exllamav3/exllamav3_ext/rocm/README.md)
(shim design).

---

## Environment (already set up)

| | |
|---|---|
| ROCm | **7.2.4** (`/opt/rocm-7.2.4`), HIP 7.2.53211, clang 22 roc-7.2.4 |
| GPU | gfx1151 (Strix Halo, RDNA 3.5, wave32), 40 CUs = **20 WGPs** |
| venv | `../../.venv` — torch 2.13.0+rocm7.2, triton-rocm 3.7.1, TabbyAPI editable |
| Repo | `rocm_exl3/` on `main`; `origin` = CarouselAether/rocm_exl3, `upstream` = turboderp-org/exllamav3 |

Sibling directories: `rocm_exl3_legacy/` (the v0.0.29 fork — source of the RDNA
kernels), `exllamav3/` (pristine upstream v1.3.0, diff reference), `rocm_docs/`
(ISA + programming guide PDFs, 53 scraped markdown pages).

Package name stays `exllamav3` — drop-in for TabbyAPI. Only the repo is renamed.

## State

- **44 of 50** ROCm-built sources compile unmodified (`hipcc_probe.sh --all`)
- Attention verified correct: **18/18** (`phase2_attn_check.py`)
- The 6 remaining failures are *exactly* the inline-PTX files

```
cpu/moe_handoff.cu      quant/exl3_gemm.cu       quant/exl3_gemv.cu
quant/exl3_gemv_int8.cu quant/exl3_kernel_map.cu quant/reconstruct.cu
```

All six fail on `ptx.cuh`. Nothing else is outstanding.

---

## Findings that must survive — do not re-derive

**WMMA operand order is `(B, A, C)`, not `(A, B, C)`.** Store layout is
`row = lane % 16, col_base = (lane >= 16) ? 1 : 0`. Established empirically in the
original port, **re-proven on ROCm 7.2.4**: 256/256 cells, 0 transposed, 0 swapped.

```bash
hipcc -o /tmp/wmma ../../rocm_exl3_legacy/tests/wmma_smoke2.cpp -std=c++17 --offload-arch=gfx1151 && /tmp/wmma
```

Getting this wrong produces plausible-looking but wrong tensors. Re-run it before
trusting any GEMM output.

**int8 dot product: `sudot4`, never `sdot4`.** gfx1151 lacks `dot1-insts`, so the
obvious `__builtin_amdgcn_sdot4` fails to compile and looks like "no int8 support".
It has `dot8-insts`: `__builtin_amdgcn_sudot4(false, a, true, b, c, false)` is an
exact replacement for PTX `dp4a.u32.s32`, verified numerically across sign
boundaries. `udot4` and `fdot2` are also available; `udot2` is not.

**`block_m / num_warps == 16`** — one WMMA 16×16 fragment row per warp. Every good
prefill config holds it; off-ratio configs measured up to **4× slower**. Any tile
retuning must preserve it.

**Noise floor is 4.7%** (1.6% stdev, first run reads high). Any perf claim under
~5% needs repeat measurement. One decode "win" already evaporated under this test.

**LDS is 64 KB, not ~100 KB.** Upstream sizes tiles for a Hopper/Blackwell smem
budget. This is why narrow kv tiles win on RDNA, and likely why cooperative GEMM
kernels stall on WGP pairing.

---

## Order of work

### 1. `ptx.cuh` → `rocm/ptx_rdna.hip.h` — gates everything else

22 asm blocks. Add a `cuda_shim/ptx.cuh` that redirects on ROCm, implementing the
same inline-function interface. Three tiers:

| Tier | Ops | Approach |
|---|---|---|
| Tensor core | `mma.sync.aligned.m8n8k4`, `m16n8k16` (f32 + f16 accum), `ldmatrix.sync.aligned.m8n8.x4` | `rdna_wmma.hip` from the legacy fork — already correct |
| Memory ordering | `ld.global.acquire.{gpu,sys}`, `st.global.release.sys`, `red.relaxed.gpu.global.add.s32`, `st.global.wt` | `__hip_atomic_*`, same pattern as the existing `cuda_shim/cuda/atomic` |
| Bit/arith | `bfe.u64`, `shf.r.wrap.b32`, `mul.hi.u32`, `mul.lo.u32` | plain C++ / `__builtin_amdgcn_*` |
| **Async copy** | `cp.async.*`, `cp.async.wait_group` | **open question — no RDNA 3.5 equivalent** |

`cp.async` is the one with no clean answer. RDNA has no async global→LDS copy;
the fallback is ordinary loads through registers, which costs the pipelining the
CUDA kernel assumes. Likely a large part of why `exl3_gemm_inner` is expensive.

### 2. `exl3_gemv_int8` — best value per unit effort

1223 lines whose only int8 primitive is one `dp4a.u32.s32` wrapper
(`exl3_gemv_int8_kernel.cuh:60`), and the replacement is proven. This is the
**single-token decode path** — the one that governs interactive throughput on a
bandwidth-limited APU. New in v1.3.0, so no legacy version to port.

### 3. `exl3_gemv`, `reconstruct`, `exl3_kernel_map`

Port from `rocm_exl3_legacy`. Upstream drift is low-to-moderate:
`exl3_dq.cuh` **0 lines** (free), `codebook.cuh` 73, `exl3_gemm_kernel.cuh` 21,
`exl3_gemv_kernel.cuh` 46, `exl3_kernel_map.cu` −165 (restructured).

Put them in `exllamav3_ext/rocm/quant/`. Convention: `*_rdna` for ports of
upstream files, `rdna_*` for new RDNA primitives (`rdna_wmma.hip`).

### 4. `exl3_gemm_inner` — the expensive one

**501 of 733 lines changed upstream** between v0.0.29 and v1.3.0. The legacy
`exl3_gemm_inner_rdna.hip.h` (392 lines) was a WMMA rewrite of the *old* version,
so it cannot be copied forward — it needs re-deriving against the new structure.

### 5. `comp_units_rdna` — decide per family

Legacy covers **8 of ~66** upstream instantiation slots, in one family:

| Family | Upstream slots | Legacy |
|---|---|---|
| EXL3 GEMM | 8 bitwidths × 3 codebooks = 24 | 8 (bitwidth only — **no codebook axis**) |
| int8 GEMV | `coop_k1..8` + `sq_k1..6` = 14 | none |
| MoE | 20 | none |
| `quantize_tiles` | 8 | none |

The `_cb0/_cb1/_cb2` codebook axis did not exist in the fork's base. This is a
per-family scoping decision, not a file copy.

---

## Validation order — follow the evidence

The original fork **loaded a model and produced coherent tokens**; quantization
was never confirmed working. So:

1. build + import
2. model load
3. coherent tokens (known-good territory)
4. `eval/ppl.py` against a reference perplexity
5. TabbyAPI end-to-end
6. **quantization — never worked, do not assume regression vs never-worked**

## Commands

```bash
cd /home/carousel/Desktop/exlproject
source .venv/bin/activate

rocm_exl3/rocm_tools/hipcc_probe.sh --all        # expect 44/50
python rocm_exl3/rocm_tools/phase2_attn_check.py # expect 18/18
python rocm_exl3/rocm_tools/bench_prefill_tiles.py
python rocm_exl3/rocm_tools/bench_decode_splits.py

EXL3_BACKEND=rocm pip install -e rocm_exl3       # once kernels land
```

Docs are greppable markdown in `rocm_docs/` — faster than the PDFs for API
questions. Note: HIP docs cover **MFMA (CDNA) but not WMMA (RDNA)**, so the ISA
PDF is the only authority for WMMA layout.

## Open decisions

- `cp.async` replacement strategy for `exl3_gemm_inner`
- Whether `comp_units_rdna` needs the full codebook axis or a subset
- `hgemm.cu`'s `at::mm` swap is probably revertible (hipBLAS works on 7.2.4) —
  retest once stable
- Python guards in the legacy fork (`MultiLinear` fusion, `BC_GatedMLP`,
  block-sparse MoE) are marked **retest**, not keep — ROCm changed a lot between
  7.1 and 7.2. `arch_list.py`'s guard is already redundant; upstream handles it.
