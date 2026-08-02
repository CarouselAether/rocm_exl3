// Shim: `#include <cublas_v2.h>` -> hipBLAS.
//
// exllamav3 uses a narrow slice of cuBLAS: a handle, SetStream/SetPointerMode/
// SetWorkspace, and one cublasGemmEx call in hgemm.cu. hipBLAS is API-compatible
// across all of it, so plain aliases suffice.
#pragma once

#include <hipblas/hipblas.h>

using cublasHandle_t   = hipblasHandle_t;
using cublasStatus_t   = hipblasStatus_t;
using cublasOperation_t = hipblasOperation_t;
using cudaDataType_t   = hipDataType;
using cublasComputeType_t = hipblasComputeType_t;

#define cublasSetStream       hipblasSetStream
#define cublasSetPointerMode  hipblasSetPointerMode
#define cublasSetWorkspace    hipblasSetWorkspace
#define cublasGemmEx          hipblasGemmEx
#define cublasCreate          hipblasCreate
#define cublasDestroy         hipblasDestroy

#define CUBLAS_OP_N               HIPBLAS_OP_N
#define CUBLAS_OP_T               HIPBLAS_OP_T
#define CUBLAS_POINTER_MODE_HOST  HIPBLAS_POINTER_MODE_HOST

#define CUDA_R_16F  HIP_R_16F
#define CUDA_R_32F  HIP_R_32F

#define CUBLAS_COMPUTE_32F            HIPBLAS_COMPUTE_32F
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT

#define CUBLAS_STATUS_SUCCESS           HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED   HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED      HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE     HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_MAPPING_ERROR     HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED  HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR    HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED     HIPBLAS_STATUS_NOT_SUPPORTED
#define CUBLAS_STATUS_ARCH_MISMATCH     HIPBLAS_STATUS_ARCH_MISMATCH

// hipBLAS has no license-error status. upstream's cublasGetErrorString switches
// over it, so it needs a value that is a valid constant expression, is distinct
// from every real status, and can never be returned.
//
// It must stay inside the enum's representable range: hipblasStatus_t has no
// fixed underlying type and its largest enumerator is 11, so the range is 0..15.
// A value outside that (1000) is not a valid constant expression and the case
// label fails to compile -- which is exactly what hipify's output does today.
#define CUBLAS_STATUS_LICENSE_ERROR     ((cublasStatus_t)15)
