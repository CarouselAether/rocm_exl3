// Generated quantize-tiles instantiation for RDNA (v1.5.0). Mirrors
// quant/comp_units/quantize_tiles_inst_k1.cu with the kernel header swapped for
// the RDNA sibling (which swaps codebook.cuh for codebook_rdna.hip.h).
//
// `optimized` is accepted for signature compatibility with the upstream
// declarations and ignored: quantize_tiles_optimized.cuh has no RDNA sibling and
// quantize_rdna.hip never asks for it (quantize_tiles_use_optimized() is false).
//
// These exist because ROCM_EXCLUDE drops all of quant/comp_units/, which is
// where upstream defines these getters; without them the ROCm build fails at
// link with undefined quantize_tiles_kernel_k*_cb*.

#include "../../../quant/comp_units/quantize_tiles_instances.cuh"
#include "../quantize_tiles_kernel_rdna.hip.h"

fp_quantize_tiles_kernel quantize_tiles_kernel_k1_cb0(bool optimized) { (void) optimized; return quantize_tiles_kernel<1, 0>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k1_cb1(bool optimized) { (void) optimized; return quantize_tiles_kernel<1, 1>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k1_cb2(bool optimized) { (void) optimized; return quantize_tiles_kernel<1, 2>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k1_cb2_l160(bool optimized) { (void) optimized; return quantize_tiles_kernel<1, 2, 160>; }
