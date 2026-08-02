// Shim: upstream `#include <cuda_fp16.h>` -> HIP.
// __half / __half2 keep their names on HIP. __hmin2/__hmax2 for __half2 are
// supplied by rocm/hip_compat.hip.h (force-included) -- ROCm 7.2.4 ships those
// only for __hip_bfloat162, verified by grep over /opt/rocm/include.
#pragma once
#include <hip/hip_fp16.h>
