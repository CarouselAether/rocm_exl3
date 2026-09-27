#!/usr/bin/env bash
# Build rocm_tools/wmma_gate.hip (bit-exact WMMA layout gate for rdna_wmma.hip.h).
#
#   rocm_tools/build_wmma_gate.sh                   # gfx1151 -> $OUT_DIR/wmma_gate
#   GPU_ARCH=gfx1100 rocm_tools/build_wmma_gate.sh  # -> $OUT_DIR/wmma_gate_gfx1100
#   <out>/wmma_gate [--golden PATH]                 # math check + bitwise vs golden
#   <out>/wmma_gate --record                        # (re)write rocm_tools/wmma_gate.golden
#   rocm_tools/build_wmma_gate.sh --design-check    # host-only g++ build + run:
#                                                   # validates the test's exactness
#                                                   # guarantees, touches no GPU
#
# Standalone like wmma_check.hip: the header needs only hip_runtime/hip_fp16/
# hip_bf16, so no torch, no shim, no -fgpu-rdc. hipcc is taken from
# $ROCM_PATH/bin when ROCM_PATH is set, else from PATH (override with HIPCC=).
# For the pip SDK, scrub the login-shell ROCm first, e.g.:
#
#   env -u ROCM_HOME -u HIP_PATH -u LD_LIBRARY_PATH \
#       PATH=$(echo "$PATH" | tr : '\n' | grep -v /opt/rocm | paste -sd:) \
#       ROCM_PATH=$SDK rocm_tools/build_wmma_gate.sh
#
# The gfx12 targets compile (f32 kernels build the header's __builtin_trap()
# path, the other variants are #if-excluded) but the binary refuses to run on
# a non-gfx11 device and reports FAIL.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$HERE/wmma_gate.hip
OUT_DIR=${OUT_DIR:-/tmp/claude-1000/-home-carousel-Desktop-exlproject/22e0f5a9-bfaf-4643-af9d-f1b51fa95cd1/scratchpad/wmma_gate}
mkdir -p "$OUT_DIR"

if [[ "${1:-}" == "--design-check" ]]; then
  CXX=${CXX:-g++}
  "$CXX" -std=c++17 -O1 -Wall -Wextra -x c++ -DWMMA_GATE_DESIGN_CHECK "$SRC" \
         -o "$OUT_DIR/wmma_gate_design"
  exec "$OUT_DIR/wmma_gate_design"
fi

GPU_ARCH=${GPU_ARCH:-gfx1151}
HIPCC=${HIPCC:-${ROCM_PATH:+$ROCM_PATH/bin/}hipcc}
OUT=$OUT_DIR/wmma_gate
[[ "$GPU_ARCH" == gfx1151 ]] || OUT=${OUT}_$GPU_ARCH

"$HIPCC" --offload-arch="$GPU_ARCH" -std=c++17 -O3 -Wall \
  -Wno-unused-command-line-argument \
  "$SRC" -o "$OUT"
echo "built $OUT ($GPU_ARCH, $("$HIPCC" --version 2>/dev/null | grep -m1 -i 'clang version' || echo hipcc))"
