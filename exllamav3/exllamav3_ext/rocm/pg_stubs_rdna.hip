// =============================================================================
// Tensor-parallel (pg_*) stubs for RDNA -- path unavailable, not ported
// =============================================================================
//
// ROCM_EXCLUDE drops parallel/ (8 files: CUDA IPC + inline PTX, no HIP port).
// But bindings.cpp registers the eight pg_* entry points unconditionally, so
// dropping the definitions leaves the extension importable-looking and then
// failing at dlopen with:
//
//     ImportError: ... undefined symbol: _Z15pg_init_contextm
//
// A shared library links happily with undefined symbols; the failure only
// surfaces when Python loads it. That is why the earlier link checks missed it:
// they grepped undefined symbols for the exl3_/quantize_ prefixes rather than
// checking everything the module actually needs.
//
// The fork solved this by putting `#ifndef USE_ROCM` around the includes and
// the pg_* registrations in bindings.cpp. That is an upstream edit, which this
// port does not do -- so the symbols are defined here instead, and each one
// raises a clear error rather than being silently absent.
//
// Note the path: rocm/pg_stubs_rdna.hip, NOT rocm/parallel/. ROCM_EXCLUDE
// matches "parallel/" as a substring, so a file under rocm/parallel/ would
// exclude itself and reintroduce the exact bug this file fixes.
//
// Signatures are copied verbatim from parallel/{context,broadcast,barrier,
// gather,all_reduce}.cuh. They must stay byte-identical or the definitions will
// not satisfy bindings.cpp's references -- a mismatch reappears as the same
// undefined-symbol ImportError, since C++ mangles the whole signature.
//
// The set of symbols to stub is NOT "everything starting with pg_". An earlier
// version of this file left out run_cpu_reduce_jobs / end_cpu_reduce_jobs on the
// reasoning that they were unreferenced -- which came from grepping bindings.cpp
// for "pg_", a pattern that cannot match them. They are bound at bindings.cpp
// lines 100-101 and the module failed to load on exactly those two symbols.
//
// The reliable check is not a prefix grep of the sources but a sweep of the
// built module:
//
//   nm -DC exllamav3_ext*.so | grep " U " \
//     | grep -vE "std::|torch|c10::|at::|pybind11|^ *U Py|_Py|hip|GLIBC|CXXABI"
//
// Anything left is a definition the ROCm build owes the linker.
// =============================================================================

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <vector>
#include <cstdint>

#define PG_UNAVAILABLE(name) \
    TORCH_CHECK(false, \
        "exllamav3: " name " is unavailable on the ROCm backend. Tensor-parallel " \
        "is built on CUDA IPC and inline PTX (exllamav3_ext/parallel/), which has " \
        "no HIP port -- see ROCM_EXCLUDE in setup.py. Load the model without " \
        "tensor-parallel (drop -tp / tensor_parallel).")

void pg_init_context(uintptr_t ctx)
{
    PG_UNAVAILABLE("pg_init_context");
}

void pg_barrier
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& abort_flag
)
{
    PG_UNAVAILABLE("pg_barrier");
}

void pg_broadcast
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int src_device,
    at::Tensor& tensor,
    uintptr_t shbuf_dev,
    size_t shbuf_size,
    at::Tensor& abort_flag
)
{
    PG_UNAVAILABLE("pg_broadcast");
}

void pg_broadcast_ll
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int src_device,
    at::Tensor& tensor,
    uintptr_t shbuf_dev,
    size_t shbuf_size,
    at::Tensor& abort_flag
)
{
    PG_UNAVAILABLE("pg_broadcast_ll");
}

void pg_gather
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int out_device,
    at::Tensor& tensor,
    c10::optional<at::Tensor>& out_tensor,
    std::vector<size_t> ldims,
    uintptr_t shbuf_dev,
    size_t shbuf_size,
    at::Tensor& abort_flag
)
{
    PG_UNAVAILABLE("pg_gather");
}

void pg_gather_small
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int out_device,
    at::Tensor& tensor,
    c10::optional<at::Tensor>& out_tensor,
    std::vector<size_t> ldims,
    uintptr_t shbuf_dev,
    size_t shbuf_size,
    at::Tensor& abort_flag
)
{
    PG_UNAVAILABLE("pg_gather_small");
}

void pg_all_reduce
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    uintptr_t shbuf_dev,
    size_t shbuf_size,
    at::Tensor& abort_flag
)
{
    PG_UNAVAILABLE("pg_all_reduce");
}

void pg_all_reduce_cpu
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    bool contributor,
    uintptr_t shbuf_dev,
    size_t shbuf_size,
    bool is_master,
    at::Tensor& abort_flag
)
{
    PG_UNAVAILABLE("pg_all_reduce_cpu");
}

// Bound at bindings.cpp:100-101. These run on the CPU helper thread for the
// CPU-assisted all-reduce, so they are not GPU code at all -- but they live in
// parallel/all_reduce_cpu.cu, which the exclusion drops wholesale.
void run_cpu_reduce_jobs
(
    uintptr_t ctx_ptr,
    uintptr_t shbuf,
    size_t shbuf_size
)
{
    PG_UNAVAILABLE("run_cpu_reduce_jobs");
}

void end_cpu_reduce_jobs
(
    uintptr_t ctx_ptr
)
{
    PG_UNAVAILABLE("end_cpu_reduce_jobs");
}
