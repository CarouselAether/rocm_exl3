# ROCm backend

Everything ROCm-specific lives in this directory. **No upstream `.cu` / `.cuh` /
`.cpp` file is modified for ROCm** — that is the core constraint, and it is what
keeps rebasing onto new upstream releases cheap.

Built and validated on **gfx1151** (Strix Halo, RDNA 3.5, wave32) with
ROCm 7.2.4 / HIP 7.2.53211 / torch 2.13.0+rocm7.2.

## Selecting a backend

```bash
pip install --no-build-isolation .                  # auto-detects from installed torch
EXL3_BACKEND=rocm pip install --no-build-isolation . # force ROCm
EXL3_BACKEND=cuda pip install --no-build-isolation . # force CUDA
```

`--no-build-isolation` is required: pip otherwise builds in an isolated
environment containing no torch, and a torch C++ extension must be compiled
against the same torch it will run against. Without it the build stops with an
explanation rather than installing a package with no extension in it.

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
- `quant/comp_units/` (66 files) — EXL3 GEMM instantiations that reach the inline
  PTX in `ptx.cuh`. Replaced by `rocm/quant/comp_units_rdna/` (24 GEMM slots,
  8 bitwidths x 3 codebooks; MoE slots pending).

  An earlier version of this note said these "stall `grid.sync()` on RDNA WGP
  pairing". That is wrong, and it has now been tested rather than argued about:
  `rocm_tools/gemm_coop_check.hip` runs the cooperative kernel across 13 shape /
  bitwidth / codebook / grid-size combinations and every one matches the
  non-cooperative decomposition exactly.

  `SMEM_MAX` on every launch is still the wrong thing to pass on a 64 KB part,
  but not because of deadlock — an oversubscribed cooperative grid is *refused*
  by the runtime ("too many blocks in cooperative launch"), not hung. The real
  costs are halved occupancy (2 blocks/CU becomes 1 at the shapes that fit two)
  and no headroom for `exl3_mgemm`, whose `dim3(num_sms, 1, concurrency)` grid
  would be rejected outright once concurrency > 1.
  `rocm/quant/exl3_gemm_rdna.hip` passes the shape's actual requirement — see
  EXL3_RDNA_SMEM there.

- `quant/exl3_gemv.cu` — PTX `mma.sync` (locally defined as `mma_ab_h`, not one
  of `ptx.cuh`'s named wrappers) plus `cp_async` and a cooperative grid.
  Replaced by `rocm/quant/exl3_gemv_rdna.hip`, which is the fork's fdot2
  dot-product kernel behind upstream's API. m == 1 only; larger m falls through
  to the GEMM.

- `quant/exl3_gemv_int8.cu` — inline `dp4a.u32.s32` and `cp_async`.
  **Not ported.** `rocm/quant/exl3_gemv_int8_rdna.hip` defines the interface
  with the path disabled so the build links and every caller uses the fp16
  kernels. The int8 WMMA wrappers it would need (`mma_sync_i8`) are implemented
  and validated in `rocm/rdna_wmma.hip.h`.

## LDS budget on non-Strix RDNA parts

Strix Halo (gfx1151) has **64 KB** of LDS per workgroup, measured. Other RDNA
parts tolerate upstream's 90 KB figure, so this is a build-time knob rather than
a constant:

```bash
hipcc -DEXL3_RDNA_SMEM_MAX=92160 ...    # 90 KB parts
# default is 64 * 1024
```

It is deliberately **not** selected with an arch macro. `__gfx1151__` exists only
during the device pass, while the value is also needed by host code (shape
admission, launch sizing) — an `#if` would give the two passes different numbers
in one build. A multi-arch fat binary must use the smallest target's value,
since the shape table and kernel instantiations are shared across archs.

Whatever the build-time value, `exl3_rdna_smem_budget()` clamps it to the
device's real `sharedMemPerBlock` at runtime and every shape-admission test uses
the clamped figure. So an over-large build-time value costs availability of some
shapes, never a failed launch. Verified with
`rocm_tools/gemm_coop_check --smem`.

## Status

- [x] Upstream sources compile unmodified under hipcc via the shim
- [x] Dual-backend `setup.py` with `EXL3_BACKEND` selection
- [x] RDNA GEMM kernel + 24 instantiations — numerically validated, links
- [x] RDNA GEMV — numerically validated (`rocm_tools/gemv_check.hip`)
- [x] Cooperative launch — **verified on gfx1151** (`rocm_tools/gemm_coop_check.hip`),
      13 differential cases exact, and an oversubscribed grid is refused by the
      runtime rather than deadlocking
- [x] MoE kernel + 20 instantiations — builds and links, **not yet executed**
- [x] `quantize` + 8 tile instantiations — builds and links, **not yet executed**
- [x] `ROCM_EXCLUDE` covers every replaced source (verified: setup.py keeps 102
      sources, the probe compiles 102, no upstream twins remain)
- [ ] int8 GEMV — currently a disabled stub
- [ ] End-to-end validation (perplexity, TabbyAPI)

`ROCM_EXCLUDE` and `rocm_tools/hipcc_probe.sh`'s exclusion regex are the same
statement written twice — keep them in step. Every `_rdna` sibling is only
correct if its upstream twin is excluded; building both is a duplicate-symbol
link failure.

## Re-verifying against a new ROCm

The claims above are checkable, not folklore:

```bash
rocm_tools/hipcc_probe.sh --all     # compile every source with the shim
```

Anything in `hip_compat.hip.h` marked as "absent from HIP" should be re-grepped
against `/opt/rocm/include` after a toolchain bump and deleted if HIP grew it.
