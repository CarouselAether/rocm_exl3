// Shim: `#include <cuda_bf16.h>`. HIP spells the type __hip_bfloat16 while
// upstream uses __nv_bfloat16, so alias both spellings.
#pragma once
#include <hip/hip_bf16.h>
using __nv_bfloat16  = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
