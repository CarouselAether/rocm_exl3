// Shim: `#include <curand_kernel.h>` -> hipRAND device API.
//
// The generator/*.cu sampling kernels use curandStatePhilox4_32_10_t, curand_init
// and curand_uniform. hipRAND provides all of them under hiprand* names with
// identical semantics, so this is a straight prefix rename.
#pragma once
#include <hiprand/hiprand_kernel.h>

using curandStatePhilox4_32_10_t = hiprandStatePhilox4_32_10_t;
using curandState_t              = hiprandState_t;
using curandState                = hiprandState_t;
using curandStateXORWOW_t        = hiprandStateXORWOW_t;

#define curand_init      hiprand_init
#define curand_uniform   hiprand_uniform
#define curand_uniform4  hiprand_uniform4
#define curand_normal    hiprand_normal
#define curand           hiprand
