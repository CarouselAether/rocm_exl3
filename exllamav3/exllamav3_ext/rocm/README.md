# ROCm backend

Everything ROCm-specific lives in this directory. **No upstream `.cu` / `.cuh` /
`.cpp` file is modified for ROCm** — that is the core constraint, and it is what
keeps rebasing onto new upstream releases cheap.

Built and validated on **gfx1151** (Strix Halo, RDNA 3.5, wave32) with
ROCm 7.2.4 / HIP 7.2.53211 / torch 2.13.0+rocm7.2.

## Selecting a backend

```bash
pip install .                  # auto-detects from the installed torch
EXL3_BACKEND=rocm pip install . # force ROCm
EXL3_BACKEND=cuda pip install . # force CUDA
```

Auto-detection reads `torch.version.hip` / `torch.version.cuda`. An explicit
`EXL3_BACKEND` always wins.

Target GPUs are detected via `rocminfo` and filtered against `SUPPORTED_GPU_ARCHS`
in `setup.py`. Override with `PYTORCH_ROCM_ARCH=gfx1151` (or `GPU_ARCHS`).

## How upstream sources compile unchanged

Two mechanisms, both applied from `setup.py` — never by editing upstream code:

1. **`-include rocm/hip_compat.hip.h`** force-injects the compat layer ahead of
   every translation unit. It aliases CUDA runtime/graph types and functions onto
   HIP, and supplies intrinsics HIP lacks.

2. **`-I rocm/cuda_shim`** placed *before* torch's include directory. Upstream's
   `#include <cuda_fp16.h>`, `<cublas_v2.h>`, `<ATen/cuda/CUDAContext.h>` etc.
   then resolve to the shims here, which forward to the HIP equivalents.

The second point matters for the ATen/c10 headers specifically. A ROCm torch wheel
ships **both** `ATen/cuda/` (the pre-hipify CUDA sources) and `ATen/hip/` (hipify
output). The `cuda` ones pull `<cuda_runtime_api.h>` and do not compile under
hipcc — verified by probe. The shims redirect to the `hip` ones, which do. No
namespace aliasing is needed because hipify deliberately preserves the CUDA
spellings: `c10/hip/HIPGuard.h` declares `namespace c10::cuda` with
`struct CUDAGuard`.

## Why not torch's hipify

`torch.utils.cpp_extension` auto-hipifies `.cu` sources on ROCm, rewriting them
into `*_hip.cpp` / `*_hip.cuh`. That pass is incomplete for this codebase — it
leaves `cudaKernelNodeParams`, `CUDA_KERNEL_NODE_PARAMS` and
`CUBLAS_STATUS_LICENSE_ERROR` unmapped, and emits `.cuh` headers that the *host*
compiler then parses, where `__align__` is undefined. `HIPBuildExtension` in
`setup.py` invokes `hipcc` on the pristine sources instead.

## What `hip_compat.hip.h` actually bridges

Each entry was verified against the installed 7.2.4 headers, not assumed from an
older port. ROCm changed substantially between 7.1 and 7.2.

| Bridged | Why |
|---|---|
| Runtime/graph types and calls | Straight CUDA→HIP renames |
| `__hmin2` / `__hmax2` for `__half2` | ROCm 7.2.4 ships these for `__hip_bfloat162` only |
| `__float2bfloat16_rn` / `_rz` | HIP has only `__float2bfloat16`; `_rz` is implemented as true truncation, not aliased to round-to-nearest |
| `__shfl_*_sync`, `__ballot_sync`, `__syncwarp` | See below |
| host `rsqrtf` | HIP declares the device form only |

### On the warp-sync macros

HIP 7.x **does** provide `__shfl_*_sync` (default-enabled since ROCm 7.0,
`amd_detail/amd_warp_sync_functions.h`), so the "they don't exist" rationale from
7.1-era ports no longer applies. They are still overridden here because HIP's
signatures take a 64-bit mask while upstream passes 32-bit literals, and because
RDNA is wave32 with all lanes converged at every call site in this codebase.
Dropping the mask is correct *for these kernels* and avoids the width mismatch.
The build sets `-DHIP_DISABLE_WARP_SYNC_BUILTINS=1` so these macros are the sole
definition rather than racing HIP's.

**This is a wave32, fully-converged assumption.** A future kernel with divergent
lanes at a shuffle would need the real masked forms.

## Excluded upstream sources

`ROCM_EXCLUDE` in `setup.py`, with reasons:

- `parallel/` (8 files) — CUDA IPC + inline PTX. Tensor-parallel is unavailable
  on ROCm.
- `quant/comp_units/` (66 files) — cooperative-launch EXL3 GEMM instantiations
  that stall `grid.sync()` on RDNA WGP pairing. Replaced by
  `rocm/quant/comp_units_rdna/`.

## Status

- [x] Upstream sources compile unmodified under hipcc via the shim
- [x] Dual-backend `setup.py` with `EXL3_BACKEND` selection
- [ ] RDNA kernel port (`comp_units_rdna/`, WMMA, GEMV) — **in progress**
- [ ] End-to-end validation (perplexity, TabbyAPI)

Until the kernel port lands, a ROCm build links but has no EXL3 quant kernels.

## Re-verifying against a new ROCm

The claims above are checkable, not folklore:

```bash
tools/hipcc_probe.sh --all          # compile every source with the shim
tools/scrape_rocm_docs.py all       # refresh rocm_docs/ (edit the pinned URLs)
```

Anything in `hip_compat.hip.h` marked as "absent from HIP" should be re-grepped
against `/opt/rocm/include` after a toolchain bump and deleted if HIP grew it.
