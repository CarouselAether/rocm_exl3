// Shim: `#include <curand_kernel.h>` -> hipRAND device API.
// The generator/*.cu sampling kernels use curandState / curand_init / curand_uniform,
// all of which hipRAND provides under the same spellings via its CUDA-compat layer.
#pragma once
#include <hiprand/hiprand_kernel.h>
