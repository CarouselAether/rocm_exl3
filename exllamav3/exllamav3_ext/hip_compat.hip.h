// Central ROCm/CUDA compatibility shim.
//
// Include this INSTEAD of <cuda_fp16.h> / <c10/cuda/CUDAGuard.h> /
// <ATen/cuda/CUDAContext.h> / <cooperative_groups.h> at the top of any .cu
// or .cpp file that needs to build under both CUDA (nvcc) and ROCm (hipcc
// with -DUSE_ROCM=1 -D__HIP_PLATFORM_AMD__=1).
//
// ROCm 7.2.1 notes:
//   - <c10/cuda/CUDAGuard.h> is broken in the PyTorch 2.11 ROCm wheel
//     (missing cuda_cmake_macros.h) so we must use <c10/hip/HIPGuard.h>
//     and rely on the cuda:: masquerade aliases it defines.
//   - <ATen/cuda/CUDAContext.h> transitively pulls cuda_runtime_api.h
//     which is absent; use <ATen/hip/HIPContext.h> instead.
//   - cudaStream_t is NOT typedef'd to hipStream_t in this build; we
//     alias it ourselves.
//   - __hmin2 / __hmax2 have no half2 overload (bf16 only); we provide
//     inline float-roundtrip fallbacks.

#pragma once

#ifdef USE_ROCM

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_cooperative_groups.h>
#include <cmath>  // rsqrtf / sqrtf / etc. for host-side code in .cu files

#include <c10/hip/HIPGuard.h>
#include <ATen/hip/HIPContext.h>
#include <hipblas/hipblas.h>
#include <hiprand/hiprand_kernel.h>

// -----------------------------------------------------------------------------
// Type aliases (CUDA name -> HIP type)
// -----------------------------------------------------------------------------
#define cudaStream_t             hipStream_t
#define cudaError_t              hipError_t
#define cudaSuccess              hipSuccess
#define cudaDeviceProp           hipDeviceProp_t
#define cudaGraph_t              hipGraph_t
#define cudaGraphExec_t          hipGraphExec_t
#define cudaGraphNode_t          hipGraphNode_t
#define cudaGraphNodeType        hipGraphNodeType
#define cudaKernelNodeParams     hipKernelNodeParams

// -----------------------------------------------------------------------------
// Runtime API (CUDA name -> HIP function)
// -----------------------------------------------------------------------------
#define cudaPeekAtLastError      hipPeekAtLastError
#define cudaGetLastError         hipGetLastError
#define cudaGetErrorString       hipGetErrorString
#define cudaGetDevice            hipGetDevice
#define cudaSetDevice            hipSetDevice
#define cudaGetDeviceProperties  hipGetDeviceProperties
#define cudaDeviceSynchronize    hipDeviceSynchronize
#define cudaDeviceGetAttribute   hipDeviceGetAttribute
#define cudaMalloc               hipMalloc
#define cudaMallocHost           hipHostMalloc
#define cudaFreeHost             hipHostFree
#define cudaMemset               hipMemset
#define cudaMemcpyAsync          hipMemcpyAsync
// hipFuncSetAttribute takes const void* (not void* like CUDA) and clang
// won't implicitly convert function-pointer -> const void*. Cast at the
// call site via the macro.
#define cudaFuncSetAttribute(func, attr, val) \
    hipFuncSetAttribute((const void*)(func), (attr), (val))
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel

// Stream + graph API
#define cudaStreamCreateWithFlags          hipStreamCreateWithFlags
#define cudaStreamDestroy                  hipStreamDestroy
#define cudaStreamBeginCapture             hipStreamBeginCapture
#define cudaStreamEndCapture               hipStreamEndCapture
#define cudaGraphDestroy                   hipGraphDestroy
#define cudaGraphExecDestroy               hipGraphExecDestroy
#define cudaGraphInstantiate               hipGraphInstantiate
#define cudaGraphGetNodes                  hipGraphGetNodes
#define cudaGraphNodeGetType               hipGraphNodeGetType
#define cudaGraphKernelNodeGetParams       hipGraphKernelNodeGetParams
#define cudaGraphExecKernelNodeSetParams   hipGraphExecKernelNodeSetParams
#define cudaGraphLaunch                    hipGraphLaunch

// Enum constants
#define cudaStreamNonBlocking              hipStreamNonBlocking
#define cudaStreamCaptureModeThreadLocal   hipStreamCaptureModeThreadLocal
#define cudaGraphNodeTypeKernel            hipGraphNodeTypeKernel
#define cudaMemcpyDefault                  hipMemcpyDefault
#define cudaMemcpyHostToDevice             hipMemcpyHostToDevice
#define cudaDevAttrMultiProcessorCount     hipDeviceAttributeMultiprocessorCount
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// cuBLAS aliases (handle + ops + enums). Error-code names for CUBLAS_STATUS_*
// are handled separately in util.cuh because the enum sets differ.
#define cublasHandle_t             hipblasHandle_t
#define cublasStatus_t             hipblasStatus_t
#define cublasSetStream            hipblasSetStream
#define cublasSetWorkspace         hipblasSetWorkspace
#define cublasSetPointerMode       hipblasSetPointerMode
#define cublasHgemm                hipblasHgemm
#define cublasGemmEx               hipblasGemmEx
#define CUBLAS_OP_N                HIPBLAS_OP_N
#define CUBLAS_POINTER_MODE_HOST   HIPBLAS_POINTER_MODE_HOST
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_COMPUTE_32F         HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_16F         HIPBLAS_COMPUTE_16F
#define CUDA_R_16F                 HIP_R_16F
#define CUDA_R_32F                 HIP_R_32F
#define CUDA_R_16BF                HIP_R_16BF
#define CUDA_R_8F_E4M3             HIP_R_8F_E4M3
#define CUDA_R_8F_E5M2             HIP_R_8F_E5M2

// cuRAND aliases
#define curandStatePhilox4_32_10_t hiprandStatePhilox4_32_10_t
#define curand_init                hiprand_init
#define curand_uniform             hiprand_uniform

// bfloat16 conversion helpers. HIP only provides __float2bfloat16 (round to
// nearest). Map _rn directly; _rz has no HIP equivalent so we fall back to
// round-to-nearest — acceptable for the single call site in gnd.cu where
// final-layer quantization loses the sub-ulp difference anyway.
#define __float2bfloat16_rn  __float2bfloat16
#define __float2bfloat16_rz  __float2bfloat16

// HIP declares rsqrtf as __device__ only, so host-side calls like
// `scale = rsqrtf((float)dim)` in kernel launch wrappers fail to resolve.
// Provide a __host__ __device__ wrapper and redirect via macro.
__host__ __device__ inline float __hip_compat_rsqrtf(float x)
{
#ifdef __HIP_DEVICE_COMPILE__
    return ::rsqrtf(x);
#else
    return 1.0f / sqrtf(x);
#endif
}
#define rsqrtf __hip_compat_rsqrtf

// -----------------------------------------------------------------------------
// bfloat16 types
// -----------------------------------------------------------------------------
#define __nv_bfloat16   __hip_bfloat16
#define __nv_bfloat162  __hip_bfloat162

// -----------------------------------------------------------------------------
// Warp-sync intrinsics: strip the mask (RDNA 3.5 is wave32, implicit).
// Variadic tail preserves an optional width argument.
// -----------------------------------------------------------------------------
#define __shfl_xor_sync(mask, var, laneMask, ...)  __shfl_xor(var, laneMask, ##__VA_ARGS__)
#define __shfl_sync(mask, var, src, ...)           __shfl(var, src, ##__VA_ARGS__)
#define __shfl_down_sync(mask, var, delta, ...)    __shfl_down(var, delta, ##__VA_ARGS__)
#define __shfl_up_sync(mask, var, delta, ...)      __shfl_up(var, delta, ##__VA_ARGS__)

// HIP's templated __syncwarp(mask) and __ballot_sync(mask, pred) require a
// 64-bit mask (static_assert in amd_warp_sync_functions.h). Override them.
// On RDNA 3.5 wave32 all lanes are active within a warp for our kernels, so
// dropping the mask is equivalent to passing 0xFFFFFFFF.
__device__ __forceinline__ void __hip_syncwarp_nomask()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
#define __syncwarp(...)                   __hip_syncwarp_nomask()
#define __ballot_sync(mask, predicate)    ((unsigned) __ballot(predicate))
#define __any_sync(mask, predicate)       __any(predicate)
#define __all_sync(mask, predicate)       __all(predicate)

// -----------------------------------------------------------------------------
// half2 min/max: ROCm 7.2.1 dropped the half2 overloads (kept only bf162).
// Implement via native V_PK_MIN_F16 / V_PK_MAX_F16 inline asm (VOP3P, 1 cycle).
// Avoids the half2 → float2 → fminf/fmaxf → half2 round-trip that a C-level
// emulation would require, saving 2 type conversions + 2 scalar ops per call.
// ISA reference: RDNA 3.5 ISA §7.5, packed math, table p. 67.
// -----------------------------------------------------------------------------
__device__ __forceinline__ half2 __hmin2(half2 a, half2 b)
{
    half2 r;
    asm("v_pk_min_f16 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}

__device__ __forceinline__ half2 __hmax2(half2 a, half2 b)
{
    half2 r;
    asm("v_pk_max_f16 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}

// -----------------------------------------------------------------------------
// Stream getter forwarder. PyTorch 2.11 ROCm exposes only
// at::hip::getCurrentHIPStream; we add at::cuda::getCurrentCUDAStream as a
// thin forwarder so upstream call sites compile unchanged.
// -----------------------------------------------------------------------------
namespace at { namespace cuda {
    inline c10::cuda::CUDAStream getCurrentCUDAStream()
    {
        return at::hip::getCurrentHIPStream();
    }
    inline c10::cuda::CUDAStream getCurrentCUDAStream(c10::DeviceIndex device_index)
    {
        return at::hip::getCurrentHIPStream(device_index);
    }
}}

#else  // !USE_ROCM  --- CUDA build

#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>

#endif  // USE_ROCM
