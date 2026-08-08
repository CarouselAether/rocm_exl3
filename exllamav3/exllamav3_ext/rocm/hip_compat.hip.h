// ROCm/HIP compatibility layer for exllamav3.
//
// Force-injected into every translation unit on the ROCm backend via
// `hipcc -include rocm/hip_compat.hip.h`, so upstream .cu/.cuh sources compile
// unmodified. Header redirection for `#include <cuda_*.h>` and the ATen/c10 CUDA
// headers is handled separately by rocm/cuda_shim/ on the include path.
//
// Design rule: upstream sources are never edited. Anything CUDA-shaped that does
// not exist on HIP is bridged here or in cuda_shim/, never at the call site.
//
// Written against ROCm 7.2.4 / HIP 7.2.53211 on gfx1151 (RDNA 3.5, wave32).
// Every claim below was probed against the installed headers rather than carried
// over from an older port -- ROCm changed substantially across 7.1 -> 7.2.

#pragma once

#ifndef USE_ROCM
#error "hip_compat.hip.h is ROCm-only; it must not be force-included on the CUDA backend"
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// ---------------------------------------------------------------------------
// __CUDA_ARCH__ during the device pass
// ---------------------------------------------------------------------------
// Upstream uses __CUDA_ARCH__ two ways: as a plain "am I compiling device code?"
// test (cache/lmq.cuh:6 picks device min/max vs a host helper), and as an SM
// version comparison (== 860, > 890, >= 800, < 750). HIP never defines it, so
// the device pass takes the *host* branch and then calls a __host__ function
// from device code.
//
// Defining it to 1 fixes the presence tests while leaving every SM-gated fast
// path disabled -- 1 fails ==860, >890 and >=800, which is exactly right for
// RDNA. compat.cuh's `< 750` branch is already selected by USE_ROCM.
//
// Scoped to the device pass so host code is unaffected.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
#define __CUDA_ARCH__ 1
#endif

// ---------------------------------------------------------------------------
// Runtime API: CUDA spellings -> HIP
// ---------------------------------------------------------------------------
// HIP is API-compatible here, so plain aliases suffice. Types are `using` rather
// than #define so they interact correctly with templates and overload resolution.

using cudaStream_t       = hipStream_t;
using cudaEvent_t        = hipEvent_t;
using cudaError_t        = hipError_t;
using cudaDeviceProp     = hipDeviceProp_t;
using cudaMemcpyKind     = hipMemcpyKind;
using cudaFuncAttributes = hipFuncAttributes;

#define cudaSuccess                         hipSuccess
#define cudaErrorInvalidValue               hipErrorInvalidValue
#define cudaErrorHostMemoryAlreadyRegistered \
        hipErrorHostMemoryAlreadyRegistered
#define cudaErrorNotSupported               hipErrorNotSupported
#define cudaErrorPeerAccessAlreadyEnabled   hipErrorPeerAccessAlreadyEnabled

#define cudaStreamNonBlocking               hipStreamNonBlocking
#define cudaStreamDefault                   hipStreamDefault
#define cudaStreamCreate                    hipStreamCreate
#define cudaStreamCreateWithFlags           hipStreamCreateWithFlags
#define cudaStreamDestroy                   hipStreamDestroy
#define cudaStreamWaitEvent                 hipStreamWaitEvent
#define cudaStreamQuery                     hipStreamQuery

#define cudaGetLastError                    hipGetLastError
#define cudaPeekAtLastError                 hipPeekAtLastError
#define cudaGetErrorString                  hipGetErrorString
#define cudaDeviceSynchronize               hipDeviceSynchronize
#define cudaStreamSynchronize               hipStreamSynchronize
#define cudaGetDevice                       hipGetDevice
#define cudaSetDevice                       hipSetDevice
#define cudaGetDeviceCount                  hipGetDeviceCount
#define cudaGetDeviceProperties             hipGetDeviceProperties
#define cudaDeviceGetAttribute              hipDeviceGetAttribute

#define cudaMalloc                          hipMalloc
#define cudaFree                            hipFree
#define cudaMemcpy                          hipMemcpy
#define cudaMemcpyAsync                     hipMemcpyAsync
#define cudaMemset                          hipMemset
#define cudaMemsetAsync                     hipMemsetAsync
#define cudaMemcpyHostToDevice              hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost              hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice            hipMemcpyDeviceToDevice

#define cudaFuncAttributeMaxDynamicSharedMemorySize \
        hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributePreferredSharedMemoryCarveout \
        hipFuncAttributePreferredSharedMemoryCarveout

// cudaFuncSetAttribute is templated in CUDA and accepts a typed kernel pointer.
// hipFuncSetAttribute takes a strict `const void*`, so call sites passing a
// kernel symbol fail overload resolution. Wrap rather than alias.
template <typename FuncT>
static inline hipError_t exl3_hip_func_set_attribute(FuncT* func, hipFuncAttribute attr, int value)
{
    return hipFuncSetAttribute(reinterpret_cast<const void*>(func), attr, value);
}
#define cudaFuncSetAttribute                exl3_hip_func_set_attribute

// Cooperative launch. Same const void* issue as above.
template <typename FuncT>
static inline hipError_t exl3_hip_launch_coop(FuncT* func, dim3 grid, dim3 block,
                                              void** args, size_t shmem, hipStream_t stream)
{
    return hipLaunchCooperativeKernel(reinterpret_cast<const void*>(func),
                                      grid, block, args, shmem, stream);
}
#define cudaLaunchCooperativeKernel         exl3_hip_launch_coop
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor \
        hipOccupancyMaxActiveBlocksPerMultiprocessor

// Events
using cudaEvent = hipEvent_t;
#define cudaEventCreate                     hipEventCreate
#define cudaEventCreateWithFlags            hipEventCreateWithFlags
#define cudaEventDestroy                    hipEventDestroy
#define cudaEventRecord                     hipEventRecord
#define cudaEventSynchronize                hipEventSynchronize
#define cudaEventElapsedTime                hipEventElapsedTime
#define cudaEventDisableTiming              hipEventDisableTiming

// Pinned / mapped host memory
#define cudaHostRegister                    hipHostRegister
#define cudaHostUnregister                  hipHostUnregister
#define cudaHostGetDevicePointer            hipHostGetDevicePointer
#define cudaHostRegisterMapped              hipHostRegisterMapped
#define cudaHostRegisterPortable            hipHostRegisterPortable
#define cudaHostAlloc                       hipHostMalloc
#define cudaFreeHost                        hipHostFree
#define cudaMallocHost                      hipHostMalloc
#define cudaErrorHostMemoryNotRegistered    hipErrorHostMemoryNotRegistered
#define cudaErrorCudartUnloading            hipErrorDeinitialized

// Device attributes
using cudaDeviceAttr = hipDeviceAttribute_t;
#define cudaDevAttrL2CacheSize              hipDeviceAttributeL2CacheSize
#define cudaDevAttrMultiProcessorCount      hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxSharedMemoryPerBlockOptin \
        hipDeviceAttributeSharedMemPerBlockOptin
#define cudaDevAttrMaxThreadsPerBlock       hipDeviceAttributeMaxThreadsPerBlock
#define cudaDevAttrWarpSize                 hipDeviceAttributeWarpSize

// ---------------------------------------------------------------------------
// Driver API
// ---------------------------------------------------------------------------
// exllamav3 reaches the driver API to load and launch Triton-generated modules
// (graph.cu, cuda_drv.cpp, triton_kernel.cpp). HIP mirrors it under hipModule*.
//
// NOTE: hipStreamWaitValue32 / hipStreamWriteValue32 are the closest analogues of
// cuStreamWaitValue32 / cuStreamWriteValue32, but their availability is
// device/driver dependent on ROCm. The call sites belong to the multi-GPU
// synchronisation path, which is excluded from the ROCm build (setup.py
// ROCM_EXCLUDE "parallel/"), so these mappings are for completeness only and are
// not exercised. Verify before enabling tensor-parallel on ROCm.

using CUmodule    = hipModule_t;
using CUfunction  = hipFunction_t;
using CUstream    = hipStream_t;
using CUresult    = hipError_t;
using CUdeviceptr = hipDeviceptr_t;
using CUgraphExec = hipGraphExec_t;
using CUgraphNode = hipGraphNode_t;

#define CUDA_SUCCESS                        hipSuccess
#define cuModuleLoadData                    hipModuleLoadData
#define cuModuleGetFunction                 hipModuleGetFunction
#define cuModuleUnload                      hipModuleUnload
#define cuLaunchKernel                      hipModuleLaunchKernel
#define cuFuncSetAttribute                  hipFuncSetAttribute
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES \
        hipFuncAttributeMaxDynamicSharedMemorySize
#define cuStreamWaitValue32                 hipStreamWaitValue32
#define cuStreamWriteValue32                hipStreamWriteValue32
#define CU_STREAM_WAIT_VALUE_GEQ            hipStreamWaitValueGte

#define cudaLaunchKernel                    hipLaunchKernel

// __nanosleep is a CUDA (sm_70+) intrinsic used in the ptx.cuh spin barrier.
// The AMD analogue is s_sleep, whose operand counts in units of 64 clocks and
// saturates well below CUDA's nanosecond argument -- so this backs off by a
// comparable order of magnitude rather than an equal duration. Only used for
// spin-wait pacing, where exact timing does not affect correctness.
//
// The declaration must be visible in the *host* pass too, not just the device
// pass: clang parses __device__ function bodies during host compilation, so
// guarding the whole function on __HIP_DEVICE_COMPILE__ makes every host-pass
// use an undeclared-identifier error. Only the builtin call is device-only.
// (This went unnoticed until cpu/moe_handoff.cu became the first compiling TU
// to reach a __nanosleep call site.)
__device__ __forceinline__ void __nanosleep(unsigned int)
{
#ifdef __HIP_DEVICE_COMPILE__
    __builtin_amdgcn_s_sleep(1);
#endif
}

// ---------------------------------------------------------------------------
// CUDA graph API
// ---------------------------------------------------------------------------
// graph.cuh captures kernel launches into a replayable graph. HIP mirrors the
// runtime graph API one-to-one.
//
// CUDA_KERNEL_NODE_PARAMS is the *driver*-API struct; upstream uses it for nodes
// captured from Triton cubins, which the runtime API cannot read back. HIP has no
// separate driver-side struct -- hipKernelNodeParams serves both -- so both CUDA
// spellings map onto it. The driver-API accessors likewise fold onto the runtime
// ones. Triton-on-ROCm emits code objects rather than cubins, so this path is not
// expected to be exercised; if graph capture is enabled on ROCm it needs testing.

using cudaGraph_t             = hipGraph_t;
using cudaGraphExec_t         = hipGraphExec_t;
using cudaGraphNode_t         = hipGraphNode_t;
using cudaKernelNodeParams    = hipKernelNodeParams;
using CUDA_KERNEL_NODE_PARAMS = hipKernelNodeParams;
using cudaGraphNodeType       = hipGraphNodeType;

#define cudaGraphNodeTypeKernel             hipGraphNodeTypeKernel
#define cudaStreamBeginCapture              hipStreamBeginCapture
#define cudaStreamEndCapture                hipStreamEndCapture
#define cudaStreamCaptureModeGlobal         hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeThreadLocal    hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeRelaxed        hipStreamCaptureModeRelaxed
#define cudaGraphInstantiate                hipGraphInstantiate
#define cudaGraphLaunch                     hipGraphLaunch
#define cudaGraphDestroy                    hipGraphDestroy
#define cudaGraphExecDestroy                hipGraphExecDestroy
#define cudaGraphGetNodes                   hipGraphGetNodes
#define cudaGraphNodeGetType                hipGraphNodeGetType
#define cudaGraphKernelNodeGetParams        hipGraphKernelNodeGetParams
#define cudaGraphExecKernelNodeSetParams    hipGraphExecKernelNodeSetParams
#define cuGraphKernelNodeGetParams          hipGraphKernelNodeGetParams
#define cuGraphExecKernelNodeSetParams      hipGraphExecKernelNodeSetParams

// ---------------------------------------------------------------------------
// half2 min/max
// ---------------------------------------------------------------------------
// ROCm 7.2.4 ships __hmin2 / __hmax2 for __hip_bfloat162 only (amd_hip_bf16.h),
// with no __half2 overloads -- verified by grep over /opt/rocm/include/hip.
// Upstream uses the __half2 forms, so provide them. Still required as of 7.2.4;
// re-check on future ROCm bumps and delete if HIP grows native versions.

__device__ __forceinline__ __half2 __hmin2(const __half2 a, const __half2 b)
{
    return __half2{__hlt(a.x, b.x) ? a.x : b.x,
                   __hlt(a.y, b.y) ? a.y : b.y};
}

__device__ __forceinline__ __half2 __hmax2(const __half2 a, const __half2 b)
{
    return __half2{__hgt(a.x, b.x) ? a.x : b.x,
                   __hgt(a.y, b.y) ? a.y : b.y};
}

// ---------------------------------------------------------------------------
// bfloat16 conversions with explicit rounding mode
// ---------------------------------------------------------------------------
// ROCm 7.2.4 ships only __float2bfloat16 (round-to-nearest-even, via the
// __hip_bfloat16 float ctor). CUDA's _rn / _rz spellings are absent -- verified
// by grep over amd_hip_bf16.h. Every other conversion intrinsic exllamav3 uses
// (__float2half_rn, __float2half2_rn, __int2half_rn, __halves2bfloat162,
// __bfloat1622float2, __float22bfloat162_rn, __half2half2) is already present.

__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __float2bfloat16_rn(const float f)
{
    return __float2bfloat16(f);   // HIP's default rounding is round-to-nearest-even
}

__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __float2bfloat16_rz(const float f)
{
    // Round toward zero. For bfloat16 that is exactly truncation of the fp32
    // significand, i.e. keep the high 16 bits and drop the rest. Implemented
    // directly rather than aliased to __float2bfloat16, which rounds to nearest
    // and would change results in the last bit.
    unsigned int u;
    __builtin_memcpy(&u, &f, sizeof(u));
    return __hip_bfloat16(__hip_bfloat16_raw{static_cast<unsigned short>(u >> 16)});
}

// ---------------------------------------------------------------------------
// Warp-sync primitives (wave32)
// ---------------------------------------------------------------------------
// HIP 7.2.4 *does* declare __shfl_*_sync / __ballot_sync / __syncwarp in
// amd_detail/amd_warp_sync_functions.h, templated on the mask type -- so the
// "they don't exist" rationale from the 7.1-era port no longer holds.
//
// They are still bridged here for a different reason: upstream passes 32-bit
// literal masks (0xffffffff) while HIP's declarations take a 64-bit mask, and
// RDNA is wave32 with all lanes active in every kernel we build. Dropping the
// mask entirely is both correct for these kernels and avoids the literal-width
// mismatch. The build defines HIP_DISABLE_WARP_SYNC_BUILTINS=1 so these macros
// are the single source of truth rather than racing HIP's own declarations.
//
// NOTE: correct *because* every exllamav3 kernel has all lanes converged at
// these call sites. A future kernel with divergent lanes would need the real
// masked forms.
//
// __syncwarp has a SECOND half to its contract that lane convergence does not
// cover: CUDA's __syncwarp() also orders shared-memory accesses within the warp.
// __builtin_amdgcn_wave_barrier() alone is a *scheduling* barrier -- it
// constrains instruction motion and emits no s_waitcnt -- so a bare mapping
// silently drops the memory ordering. That breaks cross-lane communication
// through LDS, where the write and the read use different addresses and the
// compiler therefore has no dependency to wait on:
//
//     src_lane_map[dest] = lane_id;     // routing.cu warp_radixsort_*
//     __syncwarp(active);
//     int src = src_lane_map[myrank];   // dest != myrank -- no wait inserted
//
// The wavefront-scope release/acquire pair restores it. This is the legacy
// fork's __hip_syncwarp_nomask() verbatim; the bare-barrier form was a
// regression introduced when hip_compat was rewritten for this port.
__device__ __forceinline__ void __exl3_rdna_syncwarp()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

#define __shfl_sync(mask, var, srcLane, ...)      __shfl(var, srcLane, ##__VA_ARGS__)
#define __shfl_up_sync(mask, var, delta, ...)     __shfl_up(var, delta, ##__VA_ARGS__)
#define __shfl_down_sync(mask, var, delta, ...)   __shfl_down(var, delta, ##__VA_ARGS__)
#define __shfl_xor_sync(mask, var, laneMask, ...) __shfl_xor(var, laneMask, ##__VA_ARGS__)
#define __ballot_sync(mask, pred)                 ((unsigned) __ballot(pred))
#define __activemask()                            ((unsigned) __ballot(1))
#define __syncwarp(...)                           __exl3_rdna_syncwarp()

// ---------------------------------------------------------------------------
// Integer dot-product intrinsic
// ---------------------------------------------------------------------------
// HIP has no __dp4a (verified: no hits under /opt/rocm/include). exllamav3 uses
// it only in the unsigned form -- codebook.cuh and exl3_gemv_kernel.cuh pass
// uint32_t lanes against 0x01010101u to byte-sum -- so it maps onto
// __builtin_amdgcn_udot4.
//
// Deliberately NOT __builtin_amdgcn_sdot4: that requires target feature
// dot1-insts, which gfx1151 does not have (probed -- it is a hard compile
// error). The unsigned builtin is available, because gfx1151 does have
// dot8-insts. The failure mode to watch for is that sdot4 not compiling reads
// like "gfx1151 has no int8 dot product" when in fact it has one.
//
// Signed and mixed-sign forms are reachable when needed, via sudot4 rather
// than a manual expansion: __builtin_amdgcn_sudot4(sign_a, a, sign_b, b, c,
// clamp). quant/exl3_gemv_int8_kernel.cuh's `dp4a.u32.s32` is exactly
// sudot4(false, a, true, b, c, false). No such overload is declared here
// because nothing upstream calls __dp4a in a signed form today.

__device__ __forceinline__ unsigned int __dp4a(unsigned int a, unsigned int b, unsigned int c)
{
    return __builtin_amdgcn_udot4(a, b, c, false);
}

// ---------------------------------------------------------------------------
// Host-side rsqrtf
// ---------------------------------------------------------------------------
// HIP declares rsqrtf as __device__ only (__clang_hip_math.h:671). attention.cu
// calls it from host code to compute the softmax scale, which fails overload
// resolution with "call to __device__ function from __host__ function".
//
// Declared __host__ and NOT guarded on __HIP_DEVICE_COMPILE__: clang parses host
// function bodies during the device pass too, so a host-pass-only definition
// still leaves the device pass unable to resolve the call. HIP allows __host__
// and __device__ overloads of one name, so this coexists with HIP's device form
// and each pass picks the right one.
#include <cmath>
__host__ inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }

// ---------------------------------------------------------------------------
// LM_CLAMP_IDX
// ---------------------------------------------------------------------------
// cache/lmq.cuh defines this macro behind `#ifndef LM_CLAMP_IDX`, choosing
// device min/max when __CUDA_ARCH__ is set and a `static inline` host helper
// otherwise. Even with __CUDA_ARCH__ defined above, the *host* pass still takes
// the host branch and then expands it inside __device__ functions -- nvcc
// tolerates that, clang does not.
//
// Pre-defining the macro here (force-include runs first) makes lmq.cuh's own
// guard skip the whole block. A __host__ __device__ helper keeps single
// evaluation of the arguments.
__host__ __device__ __forceinline__ int exl3_lm_clamp(int x, int lo, int hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}
#define LM_CLAMP_IDX(idx, lo, hi) exl3_lm_clamp((idx), (lo), (hi))

// ---------------------------------------------------------------------------
// Cache-hint loads: __ldcs / __ldcg
// ---------------------------------------------------------------------------
// Both are CUDA intrinsics with no HIP equivalent. They are NOT the same kind
// of thing, and mapping them alike would introduce a subtle bug.
//
// __ldcs ("cache streaming", evict-first) is a pure performance hint. Dropping
// it is always semantically safe. quant/exl3_gemv_kernel.cuh uses it for the
// weight prefetch ring on the decode path, which is memory-bound, so it is
// worth preserving: __builtin_nontemporal_load is the RDNA equivalent.
//
// __ldcg ("cache global") bypasses L1 and reads at L2. On NVIDIA, L1 is not
// coherent across SMs, so this is load-bearing wherever a kernel reads values
// another block wrote -- which is exactly how quant/exl3_gemv_int8_kernel.cuh
// uses it ("Reads bypass L1: the contributions arrived from other blocks").
// RDNA has the same hazard: L0 is per-CU, L2 is the device-coherent point. A
// plain load would be free to hit a stale L0 line, so this maps to a relaxed
// atomic load at agent scope, which forces the read to the coherent level.
// Same reasoning as ldg_cv_u32 in rocm/rdna_wmma.hip.h.

template <typename T>
__device__ __forceinline__ T __ldcs(const T* p)
{
    return __builtin_nontemporal_load(p);
}

template <typename T>
__device__ __forceinline__ T __ldcg(const T* p)
{
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
