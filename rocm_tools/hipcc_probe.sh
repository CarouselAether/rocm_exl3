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

FLAGS=(
  --offload-arch="$GPU_ARCH" -std=c++17 -fPIC -O3 -fgpu-rdc
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

compile_one() {
  local src="$1" out log
  out=$(mktemp /tmp/hipprobe.XXXXXX.o)
  log=$(mktemp /tmp/hipprobe.XXXXXX.log)
  if hipcc -c "$src" -o "$out" "${FLAGS[@]}" >"$log" 2>&1; then
    rm -f "$out" "$log"; return 0
  fi
  echo "$log"; rm -f "$out"; return 1
}

if [[ "${1:-}" == "--all" ]]; then
  echo "arch: $GPU_ARCH   torch: $(dirname "$TORCH_INC")"
  # Mirrors setup.py ROCM_EXCLUDE: parallel/ is CUDA IPC + inline PTX,
  # quant/comp_units/ is replaced by the RDNA instantiations, rope.cu by
  # rocm/rope_rdna.hip. ROCm-only .hip sources under rocm/ are included; the
  # headers there (cuda_shim/, hip_compat) are not compiled directly.
  mapfile -t SRCS < <( { find "$E" -name '*.cu' -o -name '*.cpp' \
        | grep -vE "/(parallel|comp_units)/|/rocm/|/rope\.cu$"
      find "$E/rocm" -name '*.hip' 2>/dev/null; } | sort)
  pass=0; fail=0; declare -a FAILED=()
  for s in "${SRCS[@]}"; do
    if log=$(compile_one "$s"); then
      pass=$((pass+1)); printf "  ok    %s\n" "${s#$E/}"
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
if log=$(compile_one "$SRC"); then
  echo "ok: $SRC"
else
  grep -E "error:|fatal error:" "$log" | head -25
  echo "--- full log: $log"
  exit 1
fi
