#!/usr/bin/env bash
# Compile exllamav3_ext sources with the ROCm shim, exactly as setup.py's
# HIPBuildExtension will. Drives the "fix what the compiler reports" loop without
# waiting on a full build, and doubles as a portability check on other RDNA parts.
#
#   rocm_tools/hipcc_probe.sh norm.cu       # one file
#   rocm_tools/hipcc_probe.sh --all         # every ROCm-built source, pass/fail
#   GPU_ARCH=gfx1100 rocm_tools/hipcc_probe.sh --all    # target another card
#
# Uses the active virtualenv's torch if one is active, else whatever `python3`
# resolves to. Override with PYTHON=/path/to/python.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
E=$REPO/exllamav3/exllamav3_ext
R=$E/rocm

PY=${PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}
PY=${PY:-python3}
command -v "$PY" >/dev/null 2>&1 || { echo "python not found: $PY" >&2; exit 1; }

TORCH_INC=$("$PY" -c "import torch,os;p=os.path.dirname(torch.__file__);print(os.path.join(p,'include'))" 2>/dev/null) || {
    echo "torch not importable from $PY -- activate the venv or set PYTHON=" >&2; exit 1; }
PY_INC=$("$PY" -c "import sysconfig;print(sysconfig.get_path('include'))")
ROCM=${ROCM_PATH:-/opt/rocm}

# Default to the installed GPU when rocminfo is available.
if [[ -z "${GPU_ARCH:-}" ]]; then
    GPU_ARCH=$(rocminfo 2>/dev/null | awk '/^  Name:.*gfx/{print $2; exit}')
    GPU_ARCH=${GPU_ARCH:-gfx1151}
fi

# NOTE: deliberately NOT -fgpu-rdc, even though setup.py builds with it.
#
# -fgpu-rdc defers device code generation to the link step, so a translation
# unit containing inline PTX that *parses* on amdgcn -- anything using only "r"
# constraints, e.g. the mma.sync in quant/exl3_gemv_kernel.cuh:35 or the dp4a in
# quant/exl3_gemv_int8_kernel.cuh:63 -- produces an object file happily and only
# fails at link. This probe reported a false pass for exl3_gemv because of it.
#
# Note that ptx.cuh fails differently, at parse time, on the 'f'/'l' constraint
# letters -- which is why those failures were always visible and these were not.
#
# Compiling per-file without rdc forces codegen now, so "ok" means the ISA was
# actually emitted.
# Mirror setup.py's GPU_ARCH_SMEM table. The APUs have 64 KB of LDS per
# workgroup (measured on gfx1151); the discrete RDNA parts take 90 KB. This
# probe claims to compile exactly as HIPBuildExtension does, so it has to make
# the same choice -- otherwise it would validate shapes the real build rejects,
# or vice versa, on any non-Strix card.
case "$GPU_ARCH" in
  gfx1150|gfx1151) SMEM_MAX_BYTES=65536 ;;
  gfx1100|gfx1101|gfx1102|gfx1200|gfx1201) SMEM_MAX_BYTES=92160 ;;
  *)               SMEM_MAX_BYTES=65536 ;;   # unknown: conservative
esac

FLAGS=(
  --offload-arch="$GPU_ARCH" -std=c++17 -fPIC -O3
  -DEXL3_RDNA_SMEM_MAX=$SMEM_MAX_BYTES
  -Wno-register
  -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2
  -DHIP_DISABLE_WARP_SYNC_BUILTINS=1
  -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1
  -DHIPBLAS_USE_HIP_HALF
  -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=exllamav3_ext
  -include "$R/hip_compat.hip.h"
  -I"$R/cuda_shim" -I"$E"
  -I"$TORCH_INC" -I"$TORCH_INC/torch/csrc/api/include" -I"$ROCM/include" -I"$PY_INC"
  -Wno-unused-command-line-argument -Wno-deprecated-declarations
  -Wno-c++20-extensions -Wno-unused-variable -Wno-unused-function
  -Wno-missing-field-initializers -Wno-#pragma-messages -Wno-pass-failed
)

# Compile without -fgpu-rdc first, so device code generation actually happens and
# inline PTX that merely *parses* on amdgcn cannot slip through (see FLAGS note).
#
# Some sources legitimately need -fgpu-rdc: they reference cross-TU device
# symbols, e.g. `extern __device__ int64_t v_indices[128]` in the RDNA GEMM
# kernel, which the comp_units instantiations depend on. Those fail the no-rdc
# pass with "undefined ... symbol" and nothing else. Retry those with rdc and
# report them as ok(rdc) rather than as failures -- but only when that is the
# *only* class of error, so a real codegen fault is never masked.
compile_one() {
  local src="$1" out log
  out=$(mktemp /tmp/hipprobe.XXXXXX.o)
  log=$(mktemp /tmp/hipprobe.XXXXXX.log)
  if hipcc -c "$src" -o "$out" "${FLAGS[@]}" >"$log" 2>&1; then
    rm -f "$out" "$log"; return 0
  fi
  # Count error *classes*, not error lines. A trailing "clang++: error: <tool>
  # command failed" is a consequence of the errors above it, not a class of its
  # own -- counting it made this equality test fail for every rdc-needing source
  # (3 undefined-symbol errors + 1 wrapper != 3), so the retry never ran and all
  # 24 comp_units_rdna were reported as hard failures when they build fine.
  local n_err n_undef
  n_err=$(grep -E "error:" "$log" | grep -vcE "error: .* command failed")
  n_undef=$(grep -cE "error:.*undefined.*symbol" "$log")
  if [ "$n_err" -eq 0 ] || [ "$n_err" -eq "$n_undef" ]; then
    # Exit 2 = "ok, but needed rdc". compile_one runs inside a command
    # substitution, so a variable set here would be discarded with the subshell;
    # the status is the only channel back to the caller.
    if hipcc -c "$src" -o "$out" "${FLAGS[@]}" -fgpu-rdc >"$log" 2>&1; then
      rm -f "$out" "$log"; return 2
    fi
  fi
  echo "$log"; rm -f "$out"; return 1
}

if [[ "${1:-}" == "--all" ]]; then
  echo "arch: $GPU_ARCH   torch: $(dirname "$TORCH_INC")"
  # Mirrors setup.py ROCM_EXCLUDE: parallel/ is CUDA IPC + inline PTX,
  # quant/comp_units/ is replaced by the RDNA instantiations, rope.cu by
  # rocm/rope_rdna.hip. ROCm-only .hip sources under rocm/ are included; the
  # headers there (cuda_shim/, hip_compat) are not compiled directly.
  #
  # Sources with an _rdna sibling under rocm/ are excluded here and built from
  # the sibling instead. They cannot be shimmed in place: each one reaches the
  # inline PTX in ptx.cuh (and, through exl3_dq.cuh, codebook.cuh) by *relative
  # quoted* include, which -I cannot intercept.
  mapfile -t SRCS < <( { find "$E" -name '*.cu' -o -name '*.cpp' \
        | grep -vE "/(parallel|comp_units)/|/rocm/|/rope\.cu$|/reconstruct\.cu$|/moe_handoff\.cu$|/exl3_gemm\.cu$|/exl3_gemv\.cu$|/exl3_gemv_int8\.cu$|/exl3_kernel_map\.cu$|/quantize\.cu$|/cuda_drv\.cpp$|/exl3_moe\.cu$"
      find "$E/rocm" -name '*.hip' 2>/dev/null; } | sort)
  pass=0; fail=0; declare -a FAILED=()
  for s in "${SRCS[@]}"; do
    log=$(compile_one "$s"); rc=$?
    if (( rc == 0 || rc == 2 )); then
      pass=$((pass+1))
      if (( rc == 2 )); then printf "  ok*   %s   (needs -fgpu-rdc)\n" "${s#$E/}"
      else printf "  ok    %s\n" "${s#$E/}"; fi
    else
      fail=$((fail+1)); FAILED+=("${s#$E/}|$log"); printf "  FAIL  %s\n" "${s#$E/}"
    fi
  done
  echo
  echo "=== $pass passed, $fail failed of ${#SRCS[@]} ==="
  if (( fail )); then
    echo
    echo "=== distinct first errors ==="
    for f in "${FAILED[@]}"; do
      src=${f%%|*}; log=${f##*|}
      printf "%-44s %s\n" "$src" "$(grep -m1 -E "error:" "$log" | sed 's/.*error: //' | cut -c1-90)"
      rm -f "$log"
    done
  fi
  exit $(( fail > 0 ))
fi

SRC="${1:?usage: hipcc_probe.sh <source.cu|--all>}"
[[ -f "$SRC" ]] || SRC="$E/$SRC"
log=$(compile_one "$SRC"); rc=$?
if (( rc == 0 || rc == 2 )); then
  if (( rc == 2 )); then echo "ok (needs -fgpu-rdc): $SRC"
  else echo "ok: $SRC"; fi
else
  grep -E "error:|fatal error:" "$log" | head -25
  echo "--- full log: $log"
  exit 1
fi
