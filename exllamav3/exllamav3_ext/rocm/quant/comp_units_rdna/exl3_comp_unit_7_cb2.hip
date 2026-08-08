// RDNA instantiation unit: bitwidth 7, codebook 2.
//
// Generated -- see rocm_tools/gen_comp_units_rdna.py. Defines exactly the
// symbols that quant/comp_units/exl3_comp_unit_7.cuh declares, so upstream's
// extern header is reused unmodified and the kernel map needs no changes.
//
// One (bitwidth, codebook) pair per translation unit, matching upstream v1.3.0.
// The original RDNA port packed all three codebooks into one array per
// bitwidth; keeping upstream's split means this file set tracks upstream's.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "../../../util.h"
#include "../../../util.cuh"
#include "../exl3_gemm_kernel_rdna.hip.h"

EXL3_KERNEL_INSTANCES_CB(7, 2)
