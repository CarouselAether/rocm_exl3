// Generated fractional-rate quantize-tiles instantiation for RDNA (v1.5.3). Mirrors
// quant/comp_units/quantize_tiles_frac_inst.cu with the kernel header swapped for
// the RDNA sibling (codebook.cuh -> codebook_rdna.hip.h). ROCM_EXCLUDE drops all of
// quant/comp_units/, so without these getters quantize_rdna.hip's
// quantize_tiles_frac fails to link.

#include "../../../quant/comp_units/quantize_tiles_instances.cuh"
#include "../quantize_tiles_frac_kernel_rdna.hip.h"

fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a1_maaaa() { return quantize_tiles_frac_kernel<1, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a2_maaaa() { return quantize_tiles_frac_kernel<2, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a3_maaaa() { return quantize_tiles_frac_kernel<3, 0xaaaau>; }
