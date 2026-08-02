// Shim: torch's ROCm wheel ships BOTH ATen/cuda/ (pre-hipify CUDA sources, which
// pull <cuda_runtime_api.h> and fail under hipcc -- verified by probe) and
// ATen/hip/ (hipify output, which compiles). Redirect to the HIP one.
// No namespace aliasing needed: hipify preserves CUDA spellings, so
// at::cuda::getCurrentCUDAStream() still resolves through this header.
#pragma once
#include <ATen/hip/HIPContext.h>
