#!/usr/bin/env bash
# Probe which PyTorch C++ API patterns compile under ROCm 7.2.1 / PyTorch 2.11.
# Usage:
#   source ~/rocm_llm/bin/activate
#   bash tests/torch_rocm_api_probe.sh

set -u

SRC="tests/torch_rocm_api_probe.cpp"
if [ ! -f "$SRC" ]; then
    echo "Run from the exllamav3 repo root"
    exit 1
fi

TORCH_PATH=$(python -c 'import os, torch; print(os.path.dirname(torch.__file__))')
TORCH_INC="$TORCH_PATH/include"
PY_INC=$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

COMMON_FLAGS=(
    -c
    -std=c++17
    -fPIC
    -O0
    -DUSE_ROCM=1
    -D__HIP_PLATFORM_AMD__=1
    -DTORCH_API_INCLUDE_EXTENSION_H
    -DTORCH_EXTENSION_NAME=probe
    -I"$TORCH_INC"
    -I"$TORCH_INC/torch/csrc/api/include"
    -I"$ROCM_PATH/include"
    -I"$PY_INC"
    --offload-arch=gfx1151
    "$SRC"
    -o /tmp/probe.o
)

OUT="/tmp/probe.log"

run_variant() {
    local name="$1"
    shift
    : > "$OUT"
    if hipcc "${COMMON_FLAGS[@]}" "$@" 2>"$OUT"; then
        echo "  OK     $name"
    else
        local first=$(grep -m1 -E "error:|fatal error:" "$OUT" | head -c 200)
        echo "  FAIL   $name  -- ${first:-(no error line captured, see $OUT)}"
    fi
}

echo "== Device guard variants =="
run_variant "at::cuda::OptionalCUDAGuard"                           -DPROBE_INCLUDE_CUDA_GUARD -DVARIANT_AT_CUDA_GUARD
run_variant "c10::cuda::OptionalCUDAGuard"                          -DPROBE_INCLUDE_CUDA_GUARD -DVARIANT_C10_CUDA_GUARD
run_variant "c10::hip::OptionalCUDAGuard"                           -DPROBE_INCLUDE_HIP_GUARD  -DVARIANT_C10_HIP_GUARD
run_variant "at::hip::OptionalHIPGuardMasqueradingAsCUDA"           -DPROBE_INCLUDE_HIP_GUARD  -DVARIANT_AT_HIP_GUARD_MASQ
run_variant "c10::cuda::OptionalCUDAGuard via HIPGuard.h (masq)"    -DPROBE_INCLUDE_HIP_GUARD  -DVARIANT_C10_CUDA_GUARD_VIA_HIP_HEADER
run_variant "at::cuda::OptionalCUDAGuard via HIPGuard.h (masq)"     -DPROBE_INCLUDE_HIP_GUARD  -DVARIANT_AT_CUDA_GUARD_VIA_HIP_HEADER

echo ""
echo "== Stream getter variants =="
run_variant "at::cuda::getCurrentCUDAStream().stream()"             -DPROBE_INCLUDE_CUDA_CONTEXT -DVARIANT_STREAM_AT_CUDA
run_variant "at::hip::getCurrentHIPStream().stream()"               -DPROBE_INCLUDE_HIP_CONTEXT  -DVARIANT_STREAM_AT_HIP
run_variant "c10::cuda::getCurrentCUDAStream().stream()"            -DPROBE_INCLUDE_CUDA_CONTEXT -DVARIANT_STREAM_C10_CUDA
run_variant "c10::hip::getCurrentHIPStream().stream()"              -DPROBE_INCLUDE_HIP_CONTEXT  -DVARIANT_STREAM_C10_HIP

echo ""
echo "== Stream type typedefs =="
run_variant "cudaStream_t"                                          -DVARIANT_CUDA_STREAM_T
run_variant "hipStream_t"                                           -DVARIANT_HIP_STREAM_T

echo ""
echo "Full error logs (for failing variants) are in /tmp/probe.log from the last run."
