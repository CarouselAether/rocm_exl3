"""Probe: does hipblasHgemm work correctly when called via PyTorch's shared
hipBLAS handle (with stream, pointer-mode, and workspace overrides), mirroring
what our exllamav3_ext/hgemm.cu did before the at::mm swap?

This complements the standalone test at tests/hipblas_hgemm_probe.cpp which
showed hipblasHgemm is correct in isolation. If this test ALSO passes, the
bug was somewhere else. If this test FAILS, we've localized the bug to one
of the handle/workspace/pointer-mode setup calls.

Build/run:
  source /home/carousel/rocm_llm/bin/activate
  python tests/hipblas_pytorch_handle_probe.py
"""
import torch
from torch.utils.cpp_extension import load_inline
import os

# Mimic our hgemm.cu path: grab PyTorch's shared hipBLAS handle, override
# stream + pointer mode + workspace, then call hipblasHgemm directly.
HGEMM_SRC = r"""
#include <hipblas/hipblas.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

// Match our hgemm.cu workspace size (4 MB) -- the actual DevCtx workspace is
// device-resident; we allocate a fresh one here to isolate the effect of
// SetWorkspace from the effect of the specific pointer value.
#define WORKSPACE_SIZE (4 * 1024 * 1024)

static void* g_ws = nullptr;

void probe_pytorch_hgemm(
    at::Tensor a, at::Tensor b, at::Tensor c,
    bool do_set_stream,
    bool do_set_pointer_mode,
    bool do_set_workspace
) {
    const at::cuda::OptionalCUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(a.scalar_type() == at::kHalf, "a must be fp16");
    TORCH_CHECK(b.scalar_type() == at::kHalf, "b must be fp16");
    TORCH_CHECK(c.scalar_type() == at::kHalf, "c must be fp16");

    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);
    TORCH_CHECK(b.size(0) == K, "b shape mismatch");
    TORCH_CHECK(c.size(0) == M && c.size(1) == N, "c shape mismatch");

    hipblasHandle_t handle = (hipblasHandle_t) at::cuda::getCurrentCUDABlasHandle();

    if (do_set_stream) {
        hipblasSetStream(handle, stream);
    }
    if (do_set_pointer_mode) {
        hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    }
    if (do_set_workspace) {
        if (!g_ws) {
            hipMalloc(&g_ws, WORKSPACE_SIZE);
        }
        hipblasSetWorkspace(handle, g_ws, WORKSPACE_SIZE);
    }

    _Float16 alpha = (_Float16) 1.0f;
    _Float16 beta  = (_Float16) 0.0f;

    // Row-major A(M,K), B(K,N), C(M,N) via col-major transposition trick.
    auto status = hipblasHgemm(
        handle,
        HIPBLAS_OP_N, HIPBLAS_OP_N,
        N, M, K,
        (const hipblasHalf*) &alpha,
        (const hipblasHalf*) b.data_ptr(), N,
        (const hipblasHalf*) a.data_ptr(), K,
        (const hipblasHalf*) &beta,
        (hipblasHalf*) c.data_ptr(), N
    );
    TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS,
                "hipblasHgemm failed, status=", (int) status);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe_pytorch_hgemm", &probe_pytorch_hgemm,
          "hipblasHgemm via PyTorch shared handle, with toggleable setup");
}
"""

print("Building JIT extension... (may take 30-60s)")
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
ext = load_inline(
    name="hipblas_pytorch_handle_probe",
    cpp_sources=[HGEMM_SRC],
    extra_cflags=[
        "-O2",
        f"-I{ROCM_PATH}/include",
        f"-I{ROCM_PATH}/include/hip",
    ],
    extra_ldflags=[
        f"-L{ROCM_PATH}/lib",
        "-lhipblas",
        "-lamdhip64",
    ],
    verbose=False,
)
print("Built.\n")


def run_probe(M, K, N, label, do_set_stream, do_set_pm, do_set_ws):
    torch.manual_seed(42)
    a = (torch.randn(M, K, dtype=torch.half, device="cuda") * 0.5).contiguous()
    b = (torch.randn(K, N, dtype=torch.half, device="cuda") * 0.5).contiguous()
    c = torch.zeros(M, N, dtype=torch.half, device="cuda")

    ext.probe_pytorch_hgemm(a, b, c, do_set_stream, do_set_pm, do_set_ws)
    torch.cuda.synchronize()

    ref = torch.mm(a.float(), b.float()).half()

    diff = (c.float() - ref.float()).abs()
    max_err = diff.max().item()
    tol = 0.05 * (K ** 0.5)
    bad = (diff > tol).sum().item()
    zeros = (c == 0).sum().item()
    total = M * N

    print(f"=== {label} (stream={do_set_stream}, pm={do_set_pm}, ws={do_set_ws}) "
          f"M={M} K={K} N={N} ===")
    print(f"  tolerance abs:   {tol:.4f}")
    print(f"  max abs error:   {max_err:.4f}")
    print(f"  out-of-tol:      {bad} / {total} ({100.0*bad/total:.1f}%)")
    print(f"  zero cells:      {zeros} / {total}")
    print(f"  sum_got={c.sum().item():.4f}  sum_ref={ref.sum().item():.4f}")
    print(f"  sample C[0,0]=  got={c[0,0].item():.4f}  ref={ref[0,0].item():.4f}")
    print(f"  sample C[-1,-1] got={c[-1,-1].item():.4f}  ref={ref[-1,-1].item():.4f}")
    ok = zeros < total // 4 and bad < total // 100
    print(f"  {'PASS' if ok else '** FAIL **'}\n")
    return ok


# Size matrix matching the C++ probe
shapes = [
    (32, 128, 64, "small"),
    (128, 512, 1024, "medium"),
    (256, 2048, 2048, "large"),
    (17, 257, 129, "non-aligned"),
]

# Full setup — exactly what our hgemm.cu did
print(">>>>>>>> With all overrides (stream + pm + workspace) <<<<<<<<\n")
all_ok = True
for M, K, N, label in shapes:
    all_ok &= run_probe(M, K, N, label, True, True, True)

# Probe each override in isolation, if the combined run failed
if not all_ok:
    print("\n>>>>>>>> Combined run failed. Isolating which override breaks it <<<<<<<<\n")
    for only in [("stream only", True, False, False),
                 ("pointer-mode only", False, True, False),
                 ("workspace only", False, False, True),
                 ("no overrides (PyTorch defaults)", False, False, False)]:
        label, s, pm, ws = only
        run_probe(128, 512, 1024, f"medium / {label}", s, pm, ws)

print("\nDone.")
