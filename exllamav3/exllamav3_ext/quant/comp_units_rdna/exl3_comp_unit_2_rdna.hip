// Target: gfx1151 (RDNA 3.5), wave32 only
// Intrinsic: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32

#include <hip/hip_fp16.h>
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../../util.h"
#include "../../util.cuh"
#include "../../rdna_wmma.hip"
#include "../exl3_gemm_kernel_rdna.hip.h"
#include "exl3_comp_unit_2_rdna.hip.h"

ALL_EXL3_KERNEL_INSTANCES(2)
