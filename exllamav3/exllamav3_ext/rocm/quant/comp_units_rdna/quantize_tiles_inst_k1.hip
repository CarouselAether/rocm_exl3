// Generated quantize-tiles instantiation for RDNA. Mirrors
// quant/comp_units/quantize_tiles_inst_k1.cu with the kernel header swapped
// for the RDNA sibling (which swaps codebook.cuh for codebook_rdna.hip.h).
//
// These exist because ROCM_EXCLUDE drops all of quant/comp_units/, which is
// where upstream defines these getters. quantize.cu consumes them by
// declaration only, so without these TUs the ROCm build compiles and then fails
// at link with undefined quantize_tiles_kernel_k*_cb* -- a failure the per-file
// probe structurally cannot see.

#include "../../../quant/comp_units/quantize_tiles_instances.cuh"
#include "../quantize_tiles_kernel_rdna.hip.h"

fp_quantize_tiles_kernel quantize_tiles_kernel_k1_cb0() { return quantize_tiles_kernel<1, 0>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k1_cb1() { return quantize_tiles_kernel<1, 1>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k1_cb2() { return quantize_tiles_kernel<1, 2>; }
