// Device-code companion for graphpatch_module_check.hip and
// graphpatch_multinode_check.hip. Compiled to a raw code object and loaded via
// hipModuleLoadData to mimic how graph.cu captures Triton AOT kernels.
// Build: hipcc --genco -o probe_module.hsaco probe_module.hip
#include <hip/hip_runtime.h>

extern "C" __global__ void probe_kernel(const float* in, float* out, int scalar)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        out[0] = in[0];
        out[1] = (float) scalar;
    }
}
