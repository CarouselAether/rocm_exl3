#!/usr/bin/env bash
# Build rocm_tools/gemv_tiles_bench.hip (standalone, no torch) -> ${OUT:-/tmp/gemv_tiles_bench}
#   SRC=gemv_check builds rocm_tools/gemv_check.hip the same way (-> /tmp/gemv_check)
#   GPU_ARCH=gfx1151 (default) ; SAVE_TEMPS=1 keeps the ISA (.s) next to OUT
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
E=$(dirname "$HERE")/exllamav3/exllamav3_ext
SRC=${SRC:-gemv_tiles_bench}
OUT=${OUT:-/tmp/$SRC}
ARCH=${GPU_ARCH:-gfx1151}
EXTRA=()
[[ -n "${SAVE_TEMPS:-}" ]] && EXTRA+=(-save-temps="obj")
cd "$(dirname "$OUT")"
hipcc --offload-arch="$ARCH" -O3 -std=c++20 \
  -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1 -DHIP_DISABLE_WARP_SYNC_BUILTINS=1 \
  -include "$E/rocm/hip_compat.hip.h" -I"$E/rocm/cuda_shim" -I"$E" \
  -Wno-unused-result -Wno-unused-variable -Wno-unused-function -Wno-deprecated-declarations \
  "${EXTRA[@]}" "$HERE/$SRC.hip" -o "$OUT"
echo "built $OUT ($ARCH)"
