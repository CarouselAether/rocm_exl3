// =============================================================================
// hgemm_f16acc.cu for RDNA -- disabled stub
// =============================================================================
//
// Upstream v1.5.0 adds an fp16-accumulator tensor-core GEMM (hgemm_f16acc.cu:
// cp.async + mma.sync PTX, tuned for GeForce where fp32-accumulate HMMA runs at
// half rate). RDNA has no such rate penalty and the kernel is inline PTX, so it
// is not ported. This file keeps the four entry points the rest of the build
// references (hgemm.cu's hgemm_batched, libtorch/blocksparse_mlp.cpp's
// hgemm_recon, bindings.cpp) and routes everything to hipBLAS:
//
//   hgemm_f16acc_try     -> false   (caller falls back to cublasGemm*Ex)
//   hgemm_recon          -> hgemm   (the pre-v1.5.0 behaviour)
//   hgemm_f16acc_status  -> 0       ("cuBLAS")
//   hgemm_f16acc         -> error   (tests / benchmarks that force the kernel)
//
// setup.py's ROCM_EXCLUDE drops hgemm_f16acc.cu; building both is a duplicate
// symbol link failure.
// =============================================================================

#include <cuda_fp16.h>
#include "../hgemm.cuh"
#include "../util.h"

bool hgemm_f16acc_try(const at::Tensor& a, const at::Tensor& b, at::Tensor& c)
{
    (void) a; (void) b; (void) c;
    return false;
}

void hgemm_f16acc(at::Tensor a, at::Tensor b, at::Tensor c)
{
    (void) a; (void) b; (void) c;
    TORCH_CHECK(false, "hgemm_f16acc: the fp16-accumulator GEMM is CUDA-only (inline PTX); "
                       "not available on ROCm. hgemm / hgemm_recon use hipBLAS instead.");
}

int hgemm_f16acc_status(int device)
{
    (void) device;
    return 0;
}

void hgemm_recon(at::Tensor a, at::Tensor b, at::Tensor c)
{
    hgemm(a, b, c);
}
