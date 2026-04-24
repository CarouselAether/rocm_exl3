// Validate that hip_compat.cuh lets upstream-style code compile under hipcc.
// Build with the same command as tests/torch_rocm_api_probe.sh.

#include "../exllamav3/exllamav3_ext/hip_compat.cuh"

#include <torch/extension.h>

void probe(const at::Tensor& t, float* data_ptr)
{
    // Device guard (masquerade name from HIPGuard.h)
    const at::cuda::OptionalCUDAGuard device_guard(t.device());

    // Stream getter via our forwarder
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    (void) stream;

    // Runtime API macro aliases
    int device;
    cudaGetDevice(&device);
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    cudaError_t err = cudaPeekAtLastError();
    (void) err;
}

__global__ void probe_kernel(half2* h, float* f)
{
    // half2 min/max fallbacks
    h[0] = __hmin2(h[0], h[1]);
    h[0] = __hmax2(h[0], h[1]);

    // Shuffle-sync aliases (3-arg and 4-arg forms)
    float v = f[threadIdx.x];
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_sync(0xffffffff, v, 0);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_up_sync(0xffffffff, v, 4);
    v += __shfl_sync(0xffffffff, v, 0, 4);  // 4-arg width form
    f[threadIdx.x] = v;
}

int main() { return 0; }
