// Shim: see ATen/cuda/CUDAContext.h. c10/hip/HIPGuard.h declares `namespace
// c10::cuda` with struct CUDAGuard / OptionalCUDAGuard, so CUDA spellings work.
#pragma once
#include <c10/hip/HIPGuard.h>
