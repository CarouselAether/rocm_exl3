// Probe which PyTorch C++ API patterns compile under ROCm 7.2.1 / PyTorch 2.11.
//
// Build with:
//   source ~/rocm_llm/bin/activate
//   bash tests/torch_rocm_api_probe.sh
//
// Each VARIANT_N define tests one calling convention. Exactly one should be
// defined per compile. The build script compiles with each variant in turn
// and reports compile success/failure.

#include <torch/extension.h>
#include <ATen/Tensor.h>

#ifdef PROBE_INCLUDE_CUDA_GUARD
#include <c10/cuda/CUDAGuard.h>
#endif
#ifdef PROBE_INCLUDE_HIP_GUARD
#include <c10/hip/HIPGuard.h>
#endif
#ifdef PROBE_INCLUDE_CUDA_CONTEXT
#include <ATen/cuda/CUDAContext.h>
#endif
#ifdef PROBE_INCLUDE_HIP_CONTEXT
#include <ATen/hip/HIPContext.h>
#endif

#include <hip/hip_runtime.h>

void probe(const at::Tensor& t)
{
#ifdef VARIANT_AT_CUDA_GUARD
    const at::cuda::OptionalCUDAGuard device_guard(t.device());
#endif
#ifdef VARIANT_C10_CUDA_GUARD
    const c10::cuda::OptionalCUDAGuard device_guard(t.device());
#endif
#ifdef VARIANT_C10_HIP_GUARD
    const c10::hip::OptionalCUDAGuard device_guard(t.device());
#endif
#ifdef VARIANT_AT_HIP_GUARD_MASQ
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(t.device());
#endif
#ifdef VARIANT_C10_CUDA_GUARD_VIA_HIP_HEADER
    // HIPGuard.h defines c10::cuda::OptionalCUDAGuard as masquerade alias
    const c10::cuda::OptionalCUDAGuard device_guard(t.device());
#endif
#ifdef VARIANT_AT_CUDA_GUARD_VIA_HIP_HEADER
    const at::cuda::OptionalCUDAGuard device_guard(t.device());
#endif

#ifdef VARIANT_STREAM_AT_CUDA
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    (void) stream;
#endif
#ifdef VARIANT_STREAM_AT_HIP
    auto stream = at::hip::getCurrentHIPStream().stream();
    (void) stream;
#endif
#ifdef VARIANT_STREAM_C10_CUDA
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    (void) stream;
#endif
#ifdef VARIANT_STREAM_C10_HIP
    auto stream = c10::hip::getCurrentHIPStream().stream();
    (void) stream;
#endif

#ifdef VARIANT_CUDA_STREAM_T
    cudaStream_t s = nullptr;
    (void) s;
#endif
#ifdef VARIANT_HIP_STREAM_T
    hipStream_t s = nullptr;
    (void) s;
#endif
}

int main() { return 0; }
