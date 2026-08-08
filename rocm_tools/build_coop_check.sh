#!/usr/bin/env bash
# Build rocm_tools/gemm_coop_check.hip -> /tmp/gemm_coop_check
#
# Needs -fgpu-rdc and a second TU (exl3_kernel_map_rdna.hip) because the harness
# calls exl3_gemm_smem_bytes() -- the real function the launch path uses, not a
# copy of its arithmetic. That TU uses TORCH_CHECK, so libtorch/libc10 are linked
# even though no tensor is ever created.
#
#   rocm_tools/build_coop_check.sh && timeout 300 /tmp/gemm_coop_check
#   timeout 120 /tmp/gemm_coop_check --oversubscribe    # may hang, see the file
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
E=$REPO/exllamav3/exllamav3_ext
R=$E/rocm

PY=${PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}
PY=${PY:-python3}
TORCH_DIR=$("$PY" -c "import torch,os;print(os.path.dirname(torch.__file__))") || exit 1
TORCH_INC=$TORCH_DIR/include
TORCH_LIB=$TORCH_DIR/lib
PY_INC=$("$PY" -c "import sysconfig;print(sysconfig.get_path('include'))")
ROCM=${ROCM_PATH:-/opt/rocm}
GPU_ARCH=${GPU_ARCH:-gfx1151}

FLAGS=(
  --offload-arch="$GPU_ARCH" -std=c++17 -fPIC -O3 -fgpu-rdc
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
  ${EXTRA_FLAGS:-}
)

OBJ=$(mktemp -d /tmp/coopcheck.XXXXXX)
set -e

# The kernel map's instance tables (tfp_exl3_gemm_kernel_*) are defined by the
# 24 comp_units_rdna TUs, so they have to be in the link. That also means the
# harness can compare get_gemm_kernel_ptr()'s answer against the template it
# instantiated itself, which checks the map wiring rather than assuming it.
# exl3_devctx.cu supplies DevCtx, which the map's shape selector calls.
SRCS=("$HERE/gemm_coop_check.hip" "$R/quant/exl3_kernel_map_rdna.hip"
      "$E/quant/exl3_devctx.cu" "$R"/quant/comp_units_rdna/*.hip)

i=0
for s in "${SRCS[@]}"; do
  hipcc -c "$s" -o "$OBJ/$i.o" "${FLAGS[@]}" &
  i=$((i+1))
  if (( i % 4 == 0 )); then wait; fi
done
wait

hipcc -o /tmp/gemm_coop_check "$OBJ"/*.o \
  --offload-arch="$GPU_ARCH" -fgpu-rdc \
  -L"$TORCH_LIB" -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_LIB"
rm -rf "$OBJ"
echo "built /tmp/gemm_coop_check ($i TUs)"
