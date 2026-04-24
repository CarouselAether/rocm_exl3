# ROCm Port Changes

Summary of every upstream file modified during the ROCm/RDNA 3.5 port, plus every new file added.

Target platform: **Radeon 8060S (gfx1151, Strix Halo, RDNA 3.5)**, ROCm 7.2.1, PyTorch 2.11.0+rocm7.2, Python 3.12.

The port philosophy: keep upstream CUDA code intact. All RDNA-specific kernels live in `_rdna`-suffixed sibling files. Upstream `.cu` files are modified only to swap CUDA-specific includes for a central `hip_compat.cuh` shim, plus small `#ifdef USE_ROCM` branches where host-side behavior must diverge.

---

## Categories of change

- **[A]** Pure include swap — CUDA headers (`<cuda_fp16.h>`, `<c10/cuda/CUDAGuard.h>`, `<ATen/cuda/CUDAContext.h>`, `<cuda_runtime.h>`) replaced with `#include "hip_compat.cuh"` at top of file. Logic untouched. ~30 files fall in this bucket.
- **[B]** Substantive change — `#ifdef USE_ROCM` branch with new code path, or added error/utility code guarded on ROCm. Called out per-file below.
- **[G]** ROCm guard — Python-level `if hasattr(torch.version, 'hip') and torch.version.hip is not None` used to skip CUDA-only code paths (fused kernels, cooperative launches) that are unstable on gfx1151.

---

## Modified upstream files

### Build configuration
- **`setup.py`** — **[B]** Added ROCm branch with a custom `HIPBuildExtension` that invokes `hipcc` directly (bypassing hipify-python, which double-substitutes our macros). Adds ROCm-specific defines (`USE_ROCM=1`, `__HIP_PLATFORM_AMD__=1`, `HIPBLAS_V2`, `__HIP_NO_HALF_OPERATORS__=1`, `__HIP_NO_HALF_CONVERSIONS__=1`, `HIPBLAS_USE_HIP_HALF`, `HIP_DISABLE_WARP_SYNC_BUILTINS=1`, `TORCH_USE_HIP_DSA`), plus `-fgpu-rdc` for cooperative kernels and `-Wno-register` for upstream C++17-deprecated `register` decls. On ROCm, excludes `parallel/*`, `comp_units/*`, `reconstruct.cu`, `exl3_kernel_map.cu` (replaced by `_rdna` siblings). On CUDA, excludes everything `_rdna.*` and `comp_units_rdna/*`.

### `exllamav3_ext/` — top level

| File | Cat | Change |
|---|---|---|
| `activation.cu` | A | Include swap |
| `add.cu` | A | Include swap |
| `attention.cu` | A | Include swap (upstream attention kernel — not RDNA-specific) |
| `bindings.cpp` | B | Gated `parallel/*` includes under `#ifndef USE_ROCM`; conditionally includes `quant/exl3_kernel_map_rdna.cuh` vs `quant/exl3_kernel_map.cuh` based on `USE_ROCM`; skipped `pg_*` pybind registrations on ROCm |
| `cache/q_cache.cu` | A | Include swap |
| `causal_conv1d.cu` | A | Include swap |
| `gnd.cu` | A | Include swap |
| `graph.cu` | A | Include swap |
| `graph.cuh` | B | Added `hip_compat.cuh`; gated `<cuda_runtime.h>` and `<cuda_fp16.h>` under `#ifndef USE_ROCM` |
| **`hgemm.cu`** | B | On ROCm: swapped `cublasHgemm` / `cublasGemmEx` path for `at::mm` (PyTorch native matmul). Preserves the cuBLAS path for CUDA under `#else`. Originally believed to be a workaround for broken hipBLAS Tensile on gfx1151; standalone probes (`tests/hipblas_hgemm_probe.cpp`, `tests/hipblas_pytorch_handle_probe.py`) later showed hipblasHgemm actually works correctly on ROCm 7.2.1. **Probably cosmetic; reversible.** |
| `histogram.cu` | A | Include swap |
| `norm.cu` | A | Include swap |
| `rope.cu` | A | Include swap. A warp-store guard was added mid-session in `apply_norm` / `apply_norm_uw` to gate `sums[...] = warp_reduce_sum_f(sum)` with `if (lane_id == 0)`; this made output STRICTLY WORSE and was reverted. File is now at upstream state + hip_compat include only. |
| `routing.cu` | A | Include swap |
| `softcap.cu` | A | Include swap |
| `stloader.cpp` | A | Include swap |
| `stloader_cu.cu` | A | Include swap |
| `util.cuh` | B | Added `hip_compat.cuh` include. Added `cublasGetErrorString` overload for `hipblasStatus_t` and corresponding error-check macro under `#ifdef USE_ROCM` (hipBLAS enum names differ from cuBLAS). |

### `exllamav3_ext/generator/`

All include-swap only (Category A):
- `cache.cu`
- `gumbel.cu`
- `rep_pen.cu`
- `sampling_basic.cu`
- `sampling_extra.cu`

### `exllamav3_ext/libtorch/`

All include-swap only (Category A):
- `blocksparse_mlp.cpp`
- `gated_delta_net.cpp`
- `gated_rmsnorm.cpp`
- `linear.cpp`
- `mlp.cpp`

### `exllamav3_ext/quant/`

| File | Cat | Change |
|---|---|---|
| `exl3_devctx.cu` | A | Include swap |
| **`exl3_gemm.cu`** | B | Include swap; added `#ifdef USE_ROCM` dispatch: `if (size_m == 1 && !graph) { exl3_gemv_rdna(...); return 0; }`. Single-token inference on RDNA now routes to the non-cooperative GEMV kernel; the cooperative `exl3_gemm` kernel (which uses ~64 KB LDS and stalls `grid.sync` on RDNA WGP pairing) is used only for `size_m > 1` on ROCm. |
| `exl3_gemv.cu` | A | Include swap; conditionally includes `exl3_gemv_kernel_rdna.cuh` + `exl3_kernel_map_rdna.cuh` on ROCm. (The function body itself is a hardcoded-K=4 stub on both platforms and is not actually called on ROCm — `exl3_gemv_rdna` handles that path.) |
| `exl3_moe.cu` | A | Include swap; conditionally includes `exl3_moe_kernel_rdna.cuh` on ROCm |
| `hadamard.cu` | A | Include swap |
| `pack.cu` | A | Include swap |
| `quantize.cu` | A | Include swap |
| `util.cu` | A | Include swap |

### Python side

| File | Cat | Change |
|---|---|---|
| `exllamav3/modules/attn.py` | G | Added ROCm detection; skip `MultiLinear` K/V fusion on ROCm (mgemm cooperative kernel is unstable on gfx1151 wave32). |
| `exllamav3/modules/block_sparse_mlp.py` | G | Force `is_quantized = False` on ROCm (block-sparse MoE cooperative kernel not validated on RDNA 3.5). |
| `exllamav3/modules/mlp.py` | G | Skip `MultiLinear` gate/up fusion on ROCm; skip `BC_GatedMLP` cooperative path. |
| `exllamav3/modules/quant/exl3.py` | B | Added `import os` + `EXLLAMAV3_FORCE_TORCH_MODE` env var support in `forward` (forces `reconstruct + hgemm + hadamard` path regardless of `bsz`). Diagnostic/debug aid. |
| `exllamav3/util/arch_list.py` | G | Early-return from `maybe_set_arch_list_env` on ROCm (hipcc uses `PYTORCH_ROCM_ARCH`, not `TORCH_CUDA_ARCH_LIST`). |

---

## New files added

### Compat layer
- **`exllamav3_ext/hip_compat.cuh`** — Central ROCm/CUDA compatibility shim. Aliases CUDA runtime/types/APIs to HIP (`cudaStream_t → hipStream_t`, `cudaGetLastError → hipGetLastError`, etc.), strips 64-bit masks from `__shfl_*_sync` / `__syncwarp` / `__ballot_sync` variants (RDNA is wave32, all lanes active in our kernels), provides `__hmin2`/`__hmax2` half2 fallback (ROCm 7.2.1 dropped these), wraps `rsqrtf` for host+device use, and forwards `at::cuda::getCurrentCUDAStream` to `at::hip::getCurrentHIPStream`.

### WMMA primitives (RDNA 3.5 wave32)
- **`exllamav3_ext/rdna_wmma.hip`** — WMMA 16×16×16 fp16→fp32 wrapper. Vector types (`Vec`, `FragB`, `WmmaFragA/B/C`), `rdna_wmma` namespace with `load_matrix_a`, `load_matrix_b`, `mma_sync`, `store_matrix_c*`, `load_accumulate_c*`. Also `FSHF_IMM` / `BFE16_IMM` / `bfe64` macros (used by dequant) and `mem_fence`. Validated correct via `tests/wmma_smoke2.cpp` (non-symmetric inputs distinguishing arg-order and store-layout bugs). Arg order is `(b, a, c)`; store layout is `row = lane % 16, col_base = (lane >= 16) ? 1 : 0`.

### RDNA-specific quant kernels
All under `exllamav3_ext/quant/`:

- `codebook_rdna.cuh` — lop3 emulation + vabsdiff4 byte-sum for EXL3 codebook (cb 0/1/2). Verified bit-identical to upstream `codebook.cuh`.
- `exl3_dq_rdna.cuh` — Bit-extract dequant dispatch. Verified bit-identical to upstream `exl3_dq.cuh` modulo a redundant `& 0xffff` after `BFE16_IMM`.
- `exl3_gemm_inner_rdna.cuh` — Main cooperative GEMM inner loop. Byte-identical to working-fork version. Uses WMMA, shuffle-based B-fragment unswizzle, threadblock reduction, cross-block barrier via `atomicAdd` + `__threadfence()` (device scope).
- `exl3_gemm_kernel_rdna.cuh` — Cooperative GEMM outer wrapper; does SUH/SVH Hadamard and calls `exl3_gemm_kernel_inner` in a `grid.sync()`-coordinated loop.
- **`exl3_gemv_rdna.{cu,cuh}`** — RDNA GEMV (single-token matmul). Full-featured port of working fork: bits 1–8, cb 0/1/2, c_fp32, SUH pre-Hadamard, SVH post-Hadamard, dynamic wave selection (4/8/16 warps). Invoked from `exl3_gemm.cu` for `size_m == 1` on ROCm.
- `exl3_gemv_kernel_rdna.cuh` — Inner GEMV kernel (from preserved overlay, different from `exl3_gemv_rdna.cu`'s self-contained implementation).
- `exl3_kernel_map_rdna.{cu,cuh}` — RDNA tile-shape table. Shapes 1–4 all use `TILESIZE_K=16` (i.e., `TILEBLOCKS_K=1`), fitting within the 64 KB LDS budget.
- `exl3_moe_kernel_rdna.cuh` — RDNA MoE cooperative kernel (currently disabled at Python level — see `block_sparse_mlp.py` ROCm guard).
- `reconstruct_rdna.cu` — Full-weight reconstruction kernel (dequantize → Hadamard → matrix) for `ext.reconstruct` Python binding. Used by `torch_mode` path.
- `comp_units_rdna/` (16 files) — Per-bitwidth comp-unit instantiations: `exl3_comp_unit_{1..8}_rdna.{cu,cuh}`. Each pairs a `.cu` (instances) with a `.cuh` (extern decls).

### Tests
- `tests/wmma_smoke.cpp`, `tests/wmma_smoke2.cpp` — WMMA correctness probes. `smoke2` uses asymmetric inputs to distinguish correct output from swapped-args or transposed-store bugs.
- `tests/hipblas_hgemm_probe.cpp` — Standalone hipblasHgemm correctness check (four shapes). PASS on gfx1151 / ROCm 7.2.1.
- `tests/hipblas_pytorch_handle_probe.py` — Same probe via PyTorch-shared hipBLAS handle with stream / pointer-mode / workspace overrides. PASS — confirms our `hgemm.cu` invocation was NOT the bug we thought it was.
- `tests/hip_compat_probe.{cpp,sh}`, `tests/torch_rocm_api_probe.{cpp,sh}` — Early-session probes validating ROCm 7.2.1 + PyTorch 2.11 C++ API patterns.

### Docs
- `rocm_docs/ISA_notes.md` — Running notes from RDNA 3.5 ISA PDF with page refs and ranked improvement list (10 items, perf-oriented).
- `rocm_docs/ROCm.md` — Running notes from ROCm 7.2.1 programming guide PDF with code cross-refs.
- `rocm_docs/rdna35_instruction_set_architecture.pdf` — AMD authoritative ISA reference.
- `rocm_docs/rocm-handbook-amd-com-amd-rocm-programming-guide-en-latest.pdf` — AMD authoritative programming guide.

---

## Pending cleanup

1. **File rename** (scheduled): all `_rdna.cu` → `_rdna.hip`, all `_rdna.cuh` → `_rdna.hip.h`, `hip_compat.cuh` → `hip_compat.hip.h`. Ripples through `setup.py` source discovery and every `#include` site.
2. **`hgemm.cu` at::mm swap**: may be cosmetic per probe results. Consider reverting to restore the direct cuBLAS path for a marginal perf gain, once the rest of the port is validated.
3. **`rope.cu` cleanup**: file still carries the `hip_compat.cuh` include (correct), but was the site of a reverted guard experiment. No further change needed.
4. **Python ROCm guards**: the `MultiLinear` / `BC_GatedMLP` / block-sparse MoE paths are disabled on ROCm. Once the RDNA MoE kernel is validated, revisit these.

## Root-cause notes

The "inference produces garbage tokens" bug that dominated the debug session was **not** in any of our modified files. The actual cause was **flash-attn falling back to its Triton implementation** on ROCm (the pip-installed `flash_attn==2.8.4` has no compiled `flash_attn_2_cuda.so` for HIP). The Triton fallback produces silently-wrong attention output on ROCm 7.2.1 / gfx1151. Fix: build native ROCm kernels from the in-tree `flash-attention/` source:

```
cd flash-attention && pip install --no-build-isolation .
```

Once `flash_attn_2_cuda` loads successfully (no "falling back to Triton" warning at startup), inference produces sensible output.
