from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import concurrent.futures
import importlib.util
import os
import subprocess
import sys
import sysconfig
import threading

if torch := importlib.util.find_spec("torch") is not None:
    from torch.utils import cpp_extension
    from torch import version as torch_version
    import torch as _torch_mod

extension_name = "exllamav3_ext"
precompile = "EXLLAMA_NOCOMPILE" not in os.environ
verbose = "EXLLAMA_VERBOSE" in os.environ
ext_debug = "EXLLAMA_EXT_DEBUG" in os.environ

if precompile and not torch:
    # Upstream prints and continues, which produces a SUCCESSFUL install
    # containing no extension at all (setup_kwargs is empty without torch) that
    # then fails at the first import. On ROCm it is worse: the backend probe
    # reads torch to decide, so a torch-less build also silently announces and
    # configures itself as CUDA.
    #
    # That is escalated to a hard failure for ROCm ONLY. A CUDA build keeps
    # upstream's exact behaviour -- this port's changes to shared code paths
    # should be invisible to anyone not building for ROCm.
    #
    # Detected without torch (which is the thing that is missing): an explicit
    # EXL3_BACKEND, or ROCm being installed on the machine at all.
    #
    # The usual cause is pip's build isolation: pyproject.toml's build-system
    # requirements cannot name a ROCm torch, so the isolated build env has none. Hence --no-build-isolation, which needs torch already installed in
    # the target env (true by construction on ROCm -- see requirements_rocm.txt).
    _rocm_intended = (
        os.environ.get("EXL3_BACKEND", "").strip().lower() == "rocm"
        or os.path.isdir(os.environ.get("ROCM_PATH", "/opt/rocm"))
    )
    if not _rocm_intended:
        print("Cannot precompile unless torch is installed.")
        print("To explicitly JIT install run EXLLAMA_NOCOMPILE= pip install <xyz>")
    else:
        raise SystemExit(
            "exllamav3: torch is not importable, so no extension would be built\n"
            "and the install would succeed with nothing in it.\n"
            "\n"
            "ROCm was detected on this machine. Install torch from the ROCm index\n"
            "first, then build against it:\n"
            "\n"
            "    pip install -r requirements_rocm.txt\n"
            "    pip install --no-build-isolation .\n"
            "\n"
            "--no-build-isolation is required: pip otherwise builds in an isolated\n"
            "environment that has no torch, and a torch extension must be compiled\n"
            "against the same torch it will run against.\n"
            "\n"
            "To deliberately install the Python package without the extension, set\n"
            "EXLLAMA_NOCOMPILE=1."
        )

windows = os.name == "nt"

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
# EXL3_BACKEND=cuda|rocm forces a backend; otherwise it follows the installed
# torch. Explicit beats inferred so a ROCm build can be requested on a machine
# whose torch reports both, and so CI can pin it.

def _resolve_backend():
    explicit = os.environ.get("EXL3_BACKEND", "").strip().lower()
    if explicit:
        if explicit not in ("cuda", "rocm"):
            raise SystemExit(f"EXL3_BACKEND must be 'cuda' or 'rocm', got {explicit!r}")
        return explicit
    if not torch:
        return "cuda"
    if getattr(torch_version, "hip", None):
        return "rocm"
    if getattr(torch_version, "cuda", None):
        return "cuda"
    raise SystemExit(
        "Could not determine a GPU backend: the installed torch reports neither "
        "CUDA nor HIP. Install a CUDA or ROCm torch build, or set EXL3_BACKEND."
    )

# Resolved unconditionally, NOT gated on `precompile`. It used to read
# `_resolve_backend() if precompile else "cuda"`, which made EXLLAMA_NOCOMPILE
# silently override both EXL3_BACKEND and the installed torch -- a ROCm machine
# would configure and announce itself as a CUDA build. The gate was also
# redundant: _resolve_backend() already returns "cuda" when torch is missing,
# which is the only case it could have been guarding.
BACKEND = _resolve_backend()
IS_ROCM = BACKEND == "rocm"
print(f"exllamav3: building for backend = {BACKEND}")

library_dir = "exllamav3"
sources_dir = os.path.join(library_dir, extension_name)
rocm_dir = os.path.join(sources_dir, "rocm")

# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------
# ROCm keeps every upstream .cu/.cpp unmodified -- the CUDA-isms are bridged by
# rocm/hip_compat.hip.h (force-included) and rocm/cuda_shim/ (include-path
# redirection), never by editing upstream sources. See rocm/README.md.
#
# ROCM_EXCLUDE lists upstream paths the ROCm build skips because they cannot
# build or run correctly on RDNA as-is (inline PTX, CUDA IPC, a runtime bug, a
# library name, an LDS assumption). Each entry needs a reason; an unexplained exclusion is a silently missing
# feature.
ROCM_EXCLUDE = (
    # 8 files. Multi-GPU peer kernels built on CUDA IPC + inline PTX. No HIP port
    # yet, so tensor-parallel is unavailable on ROCm.
    "parallel/",
    # 92 files. Per-bitwidth EXL3 GEMM / MoE / quantize instantiations that
    # reach the inline PTX in ptx.cuh. ROCm replaces them with
    # rocm/quant/comp_units_rdna/ (24 GEMM, 20 MoE, 8 quantize-tile units).
    "quant/comp_units/",
    # Replaced by rocm/rope_rdna.hip, which differs by one line: `half2 x = {}`
    # is ambiguous against HIP's assignment overloads. Same exported symbols.
    "rope.cu",

    # --- sources with an _rdna sibling under rocm/ ---
    #
    # Each of these reaches inline PTX by *relative quoted* include (ptx.cuh
    # directly, or codebook.cuh through exl3_dq.cuh), which -I cannot intercept,
    # so they cannot be shimmed in place. The sibling is built instead and
    # exports the same symbols. Excluding them is not optional: building both
    # the upstream source and its sibling is a duplicate-symbol link failure.
    #
    # Keep in step with the exclusion regex in rocm_tools/hipcc_probe.sh --
    # the two are the same statement written twice.
    "quant/exl3_gemm.cu",        # -> rocm/quant/exl3_gemm_rdna.hip
    "quant/exl3_gemv.cu",        # -> rocm/quant/exl3_gemv_rdna.hip
    "quant/exl3_gemv_int8.cu",   # -> rocm/quant/exl3_gemv_int8_rdna.hip (disabled stub)
    "quant/exl3_kernel_map.cu",  # -> rocm/quant/exl3_kernel_map_rdna.hip
    "quant/reconstruct.cu",      # -> rocm/quant/reconstruct_rdna.hip
    "quant/quantize.cu",         # -> rocm/quant/quantize_rdna.hip
    "cpu/moe_handoff.cu",        # -> rocm/cpu/moe_handoff_rdna.hip
    # Compiles and links unmodified, but READS SMEM_MAX (via exl3_moe_common.cuh's
    # 90 KB default) to size the MoE launch, which fails on a 64 KB part.
    "quant/exl3_moe.cu",         # -> rocm/quant/exl3_moe_rdna.hip
    # dlopen's libcuda.so.1 by name to resolve the driver API used by the Triton
    # attention kernels. Only the library name differs on ROCm; see the sibling.
    "cuda_drv.cpp",              # -> rocm/cuda_drv_rdna.cpp
    # HIP graph capture/replay corrupts BC decode across generator jobs and
    # intermittently hangs at capture on ROCm 7.2.x (runtime-level, not this
    # codebase -- see the sibling's header for the exclusion experiments). The
    # sibling gates capture on the loaded HIP runtime: eager passthrough below
    # 7.14, capture/replay on from 7.14 / ROCm 10 (validated there);
    # EXL3_ROCM_HIP_GRAPHS=1/0 overrides.
    "graph.cu",                  # -> rocm/graph_rdna.hip
    # v1.5.0: fp16-accumulator tensor-core GEMM (cp.async + mma.sync PTX). The
    # stub sibling keeps hgemm_recon / hgemm_batched on hipBLAS.
    "hgemm_f16acc.cu",           # -> rocm/hgemm_f16acc_rdna.hip (disabled stub)
    # rocBLAS answers narrow fp16 products (BC_Attention's headwise gate: 1 x 3072 @ 3072 x 48,
    # once per layer per token) with one 128x128 tile workgroup at ~50x their bandwidth bound,
    # and hipBLASLt costs prefill. The sibling runs m <= 8, N <= 256 on its own split-K GEMV
    # kernels (capture-safe, no allocations); everything else stays on rocBLAS. See its header.
    "hgemm.cu",                  # -> rocm/hgemm_rdna.hip
    # v1.5.0: fused decode-shaped MoE kernel built on exl3_gemv_kernel.cuh (PTX
    # mma + cp.async). Not ported: the sibling stubs the entry points and
    # rocm_py routes bsz <= MAX_BSZN MoE decode to the fused exl3_moe kernel
    # instead. Its comp_units instances are already covered by "quant/comp_units/".
    "quant/exl3_moe_coop.cu",    # -> rocm/quant/exl3_moe_coop_rdna.hip (disabled stub)
)

def _collect_sources():
    src = []
    for root, _, files in os.walk(sources_dir):
        rel_root = os.path.relpath(root, start=os.path.dirname(__file__) or ".")
        for file in files:
            path = os.path.join(rel_root, file)
            norm = path.replace(os.sep, "/")
            is_hip_only = file.endswith(".hip")
            is_cuda_src = file.endswith((".c", ".cpp", ".cu"))
            if not (is_hip_only or is_cuda_src):
                continue
            in_rocm_tree = "/rocm/" in norm or norm.endswith("/rocm")
            if IS_ROCM:
                if any(x in norm for x in ROCM_EXCLUDE):
                    continue
            else:
                # CUDA build never sees the ROCm tree or .hip files
                if in_rocm_tree or is_hip_only:
                    continue
            src.append(path)
    return sorted(src)

sources = _collect_sources()
if verbose:
    print(f"exllamav3: {len(sources)} sources for {BACKEND}")

# ---------------------------------------------------------------------------
# ROCm build
# ---------------------------------------------------------------------------

def _build_jobs():
    """Parallel hipcc jobs. MAX_JOBS overrides, matching torch/flash-attn convention.

    Bounded by RAM as well as cores: hipcc on these template-heavy TUs peaks around
    2 GB, and -fgpu-rdc keeps device IR live to link time, so a 32-thread machine
    with 16 GB would OOM long before it ran out of cores. Budgeting ~2.5 GB per job
    makes the default safe on small machines without throttling large ones.
    """
    env = os.environ.get("MAX_JOBS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            raise SystemExit(f"MAX_JOBS must be an integer, got {env!r}")
    cpu = os.cpu_count() or 1
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        by_mem = max(1, int(total / (2.5 * 1024 ** 3)))
    except (ValueError, OSError, AttributeError):
        by_mem = cpu
    return max(1, min(cpu, by_mem, 32))


# ---------------------------------------------------------------------------
# Minimum ROCm
# ---------------------------------------------------------------------------
# 7.2.4 is a hard floor, not a recommendation. ROCm changed materially between
# 7.1 and 7.2 in ways this port now depends on -- notably __shfl_*_sync exists
# and is default-on (rocm/hip_compat.hip.h overrides them rather than defining
# them, and sets HIP_DISABLE_WARP_SYNC_BUILTINS to win the race), and hipBLAS
# hgemm is usable. On an older ROCm those overrides land differently and the
# result compiles but computes the wrong thing, which is the worst failure mode
# available. Fail the build instead.
#
# Parsed from ROCM_PATH/.info/version, which is a bare "7.2.4". hipcc --version
# reports the HIP runtime version ("7.2.53211") on a different numbering, and
# torch.version.hip reports that same runtime number, so neither is comparable
# against a ROCm release version without a mapping table.
MIN_ROCM = (7, 2, 4)

def _rocm_version(rocm_path):
    info = os.path.join(rocm_path, ".info", "version")
    try:
        with open(info, encoding="utf8") as fp:
            raw = fp.read().strip()
    except OSError:
        return None, None
    # "7.2.4" or "7.2.4-12345"; take the numeric head
    head = raw.split("-")[0].strip()
    parts = head.split(".")
    try:
        return tuple(int(p) for p in parts[:3]), raw
    except ValueError:
        return None, raw

def _check_rocm_version():
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    ver, raw = _rocm_version(rocm_path)
    if ver is None:
        # Unreadable rather than old: warn and continue. Refusing to build on a
        # layout we simply do not recognise would be worse than letting a
        # knowledgeable user proceed.
        print(
            f"exllamav3: WARNING -- could not read a ROCm version from "
            f"{rocm_path}/.info/version"
            + (f" (found {raw!r})" if raw else "")
            + f"; this port requires ROCm >= {'.'.join(map(str, MIN_ROCM))}. "
            "Set EXL3_SKIP_ROCM_VERSION_CHECK=1 to silence.",
            file=sys.stderr,
        )
        return
    if ver < MIN_ROCM and not os.environ.get("EXL3_SKIP_ROCM_VERSION_CHECK"):
        raise SystemExit(
            f"exllamav3: ROCm {'.'.join(map(str, ver))} found at {rocm_path}, but this "
            f"port requires >= {'.'.join(map(str, MIN_ROCM))}.\n"
            "Older ROCm builds this extension successfully and then computes wrong "
            "results (the warp-sync intrinsic overrides in rocm/hip_compat.hip.h "
            "assume 7.2 semantics), so the build stops here rather than shipping "
            "silently-wrong kernels.\n"
            "Set EXL3_SKIP_ROCM_VERSION_CHECK=1 to override at your own risk."
        )
    print(f"exllamav3: ROCm {'.'.join(map(str, ver))}")


SUPPORTED_GPU_ARCHS = {
    # RDNA3 / RDNA3.5 / RDNA4 consumer + APU parts this port targets.
    "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
}

# LDS (shared memory) per workgroup the EXL3 kernels may assume, per arch.
#
# The APUs are the odd ones out: Strix / Strix Halo report sharedMemPerBlock =
# 65536 (measured on gfx1151), while the discrete parts take upstream's 90 KB.
# Getting this wrong is not a correctness bug -- exl3_rdna_smem_budget() clamps
# to the device's real sharedMemPerBlock at runtime and shape admission uses the
# clamped value -- but too high a figure costs a failed shape selection and too
# low costs the wider tiles.
GPU_ARCH_SMEM = {
    "gfx1100": 92160, "gfx1101": 92160, "gfx1102": 92160,
    "gfx1150": 65536, "gfx1151": 65536,   # Strix / Strix Halo APUs
    "gfx1200": 92160, "gfx1201": 92160,
}
DEFAULT_ARCH_SMEM = 65536   # unknown arch: assume the smaller budget


def _resolve_smem_max(archs):
    """LDS budget for a build targeting `archs`.

    A fat binary shares one shape table and one set of kernel instantiations
    across every arch in it, so the budget has to satisfy the *smallest* target.
    Empty list = hipcc auto-detect, where the targets are unknown at configure
    time, so take the conservative value.
    """
    if not archs:
        return DEFAULT_ARCH_SMEM
    return min(GPU_ARCH_SMEM.get(a, DEFAULT_ARCH_SMEM) for a in archs)

def _resolve_offload_archs():
    """Pin --offload-arch so hipcc does not try to build for every installed GPU."""
    if env := os.environ.get("PYTORCH_ROCM_ARCH") or os.environ.get("GPU_ARCHS"):
        return [a.strip() for a in env.replace(",", " ").split() if a.strip()]
    try:
        out = subprocess.check_output(["rocminfo"], text=True, stderr=subprocess.DEVNULL)
        found = {
            ln.split()[1] for ln in out.splitlines()
            if ln.strip().startswith("Name:") and "gfx" in ln
        }
    except Exception:
        # No rocminfo (container/CI sysroot). Let hipcc auto-detect.
        return []
    supported = sorted(a for a in found if a in SUPPORTED_GPU_ARCHS)
    if not supported:
        raise SystemExit(
            f"No supported AMD GPU found. Detected {sorted(found) or 'none'}; "
            f"this port supports {sorted(SUPPORTED_GPU_ARCHS)}. "
            f"Set PYTORCH_ROCM_ARCH explicitly to override."
        )
    return supported


class HIPExtension(Extension):
    pass


class HIPBuildExtension(build_ext):
    """Compile with hipcc directly, bypassing torch's hipify-python.

    torch's CUDAExtension hipifies .cu sources under ROCm, rewriting them into
    *_hip.cpp / *_hip.cuh. That pass does not cover everything exllamav3 uses --
    it leaves cudaKernelNodeParams, CUDA_KERNEL_NODE_PARAMS and
    CUBLAS_STATUS_LICENSE_ERROR unmapped, and emits .cuh headers that get
    compiled by the host compiler where __align__ is undefined.

    Compiling the pristine sources with hipcc plus the shim avoids the rewrite
    entirely, so upstream files stay byte-identical to the CUDA branch.
    """

    def build_extensions(self):
        torch_path = os.path.dirname(_torch_mod.__file__)
        self._torch_path = torch_path
        self._torch_include = [
            os.path.join(torch_path, "include"),
            os.path.join(torch_path, "include", "torch", "csrc", "api", "include"),
            os.path.join(torch_path, "include", "TH"),
            os.path.join(torch_path, "include", "THC"),
        ]
        self._rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        for ext in self.extensions:
            if isinstance(ext, HIPExtension):
                self._build_hip(ext)
            else:
                super().build_extension(ext)

    def _build_hip(self, ext):
        here = os.path.abspath(os.path.dirname(__file__) or ".")
        rocm_abs = os.path.join(here, rocm_dir)
        ext_abs = os.path.join(here, sources_dir)

        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)

        # Before anything is compiled: an old ROCm builds this fine and then
        # computes wrong results, so the failure has to happen here.
        _check_rocm_version()

        archs = _resolve_offload_archs()
        smem_max = _resolve_smem_max(archs)
        print(f"exllamav3: offload archs = {archs or '(hipcc auto-detect)'}")
        print(f"exllamav3: LDS budget = {smem_max} bytes ({smem_max // 1024} KB)")

        defines = [
            "-DUSE_ROCM=1",
            # Applied to BOTH compiler passes deliberately. This cannot be an
            # arch macro: __gfx1151__ and friends exist only in the device pass,
            # and the value is also needed by host code (shape admission, launch
            # sizing), so an #if would give the two passes different budgets in
            # one build. See EXL3_RDNA_SMEM_MAX in rocm/quant/exl3_kernel_map_rdna.hip.h.
            f"-DEXL3_RDNA_SMEM_MAX={smem_max}",
            "-D__HIP_PLATFORM_AMD__=1",
            "-DHIPBLAS_V2",
            "-DHIPBLAS_USE_HIP_HALF",
            "-DCUDA_HAS_FP16=1",
            "-D__HIP_NO_HALF_OPERATORS__=1",
            "-D__HIP_NO_HALF_CONVERSIONS__=1",
            # HIP 7.x enables __shfl_*_sync by default with a 64-bit mask.
            # rocm/hip_compat.hip.h replaces them with wave32 mask-free macros;
            # disabling the builtins keeps those macros the single definition
            # rather than racing HIP's own declarations.
            "-DHIP_DISABLE_WARP_SYNC_BUILTINS=1",
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            f"-DTORCH_EXTENSION_NAME={ext.name}",
        ]

        # MoE tile-K. Default (32, upstream's value) lives in
        # rocm/quant/exl3_moe_shape_rdna.hip.h. Like EXL3_RDNA_SMEM_MAX it must
        # reach BOTH passes: the device pass compiles the kernel with it, and the
        # host pass derives blockDim from it, so a define that reached only one
        # would produce a silently mismatched launch rather than a compile error.
        #
        # 16 forces the single-K path -- the tile geometry every RDNA GEMM shape is
        # validated on. Split-K held two defects (both fixed 2026-08-07) that were
        # invisible at TILEBLOCKS_K == 1, so this is the first bisect step if fused
        # MoE output ever looks wrong again. It costs ~1.4-1.6x MoE throughput.
        moe_tk = os.environ.get("EXL3_RDNA_MOE_TILESIZE_K", "").strip()
        if moe_tk:
            if moe_tk not in ("16", "32"):
                raise SystemExit(
                    f"EXL3_RDNA_MOE_TILESIZE_K must be 16 or 32, got {moe_tk!r}"
                )
            if moe_tk == "16":
                print("exllamav3: MoE tile-K forced to 16 (single-K path, "
                      "~1.4-1.6x slower MoE -- bisect setting)")
            defines.append(f"-DEXL3_RDNA_MOE_TILESIZE_K={moe_tk}")

        # Warnings from code we do not own, which otherwise fire by the thousand.
        quiet = [] if os.environ.get("EXLLAMA_VERBOSE_BUILD") == "1" else [
            "-Wno-unused-command-line-argument",
            "-Wno-deprecated-declarations",
            "-Wno-unused-variable",
            "-Wno-unused-function",
            "-Wno-unused-value",
            "-Wno-missing-field-initializers",
            "-Wno-#pragma-messages",
            "-Wno-pass-failed",
            "-Wno-c++20-extensions",
        ]

        includes = [f"-I{d}" for d in (
            os.path.join(rocm_abs, "cuda_shim"),   # must precede torch's include dir
            ext_abs,
            *self._torch_include,
            os.path.join(self._rocm_path, "include"),
            sysconfig.get_path("include"),
        )]

        # torch >= 2.14 headers use C++20 `requires` clauses and torch's own
        # cpp_extension passes -std=c++20; older torch headers compile fine
        # under c++20 too (ROCm clang is 17+), so track torch's requirement.
        torch_mm = tuple(int(x) for x in _torch_mod.__version__.split("+")[0].split(".")[:2])
        cxx_std = "-std=c++20" if torch_mm >= (2, 14) else "-std=c++17"
        common = [
            "-fPIC", cxx_std,
            "-O0" if ext_debug else "-O3",
            # attention.cu uses C++17-deprecated `register`; hipcc errors by default
            "-Wno-register",
            # Force-inject the compat layer ahead of every TU so upstream sources
            # need no #include edits.
            "-include", os.path.join(rocm_abs, "hip_compat.hip.h"),
        ] + quiet

        arch_flags = [f"--offload-arch={a}" for a in archs]
        # -fgpu-rdc: relocatable device code, required for cooperative launches
        hip_flags = common + ["-fgpu-rdc"] + arch_flags

        cpp_sources = [s for s in ext.sources if s.endswith((".c", ".cpp"))]
        gpu_sources = [s for s in ext.sources if s.endswith((".cu", ".hip"))]

        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        objs = []

        # Host sources also go through hipcc: hip_compat.hip.h transitively pulls
        # <hip/hip_bf16.h>, which uses clang __builtin_elementwise_* intrinsics
        # that g++ does not implement.
        #
        # Compiled in parallel. The CUDA path gets this free from ninja via
        # torch's BuildExtension; this builder drives hipcc directly and so has to
        # do it itself. Serially this is ~25 min for ~117 sources on a 32-thread
        # box with 31 threads idle -- the single largest cost of a source install.
        #
        # Threads, not processes: each task is a subprocess.run that blocks on an
        # external compiler, so the GIL is released for essentially the whole task.
        jobs = _build_jobs()
        work = []
        for group, flags, label in ((cpp_sources, common, "cpp"), (gpu_sources, hip_flags, "hip")):
            for src in group:
                obj = os.path.join(build_temp, src.replace(os.sep, "_") + ".o")
                os.makedirs(os.path.dirname(obj), exist_ok=True)
                work.append((label, src, obj, ["hipcc", "-c", src, "-o", obj] + flags + includes + defines))

        # Object order is fixed here rather than by completion order, so the link
        # line is identical run to run regardless of how the pool interleaves.
        objs = [w[2] for w in work]
        print(f"exllamav3: compiling {len(work)} sources with {jobs} parallel job(s)", flush=True)

        done = [0]
        lock = threading.Lock()

        def _compile(item):
            label, src, obj, cmd = item
            proc = subprocess.run(cmd, capture_output = True, text = True)
            with lock:
                done[0] += 1
                if verbose:
                    print(f"[{label}] ({done[0]}/{len(work)}) {src}")
                    print("  " + " ".join(cmd))
                else:
                    print(f"[{label}] ({done[0]}/{len(work)}) {src}", flush=True)
                # Warnings are captured too, so surface them next to their source
                # rather than interleaved with whatever else was in flight.
                if proc.stderr and (verbose or proc.returncode != 0):
                    sys.stderr.write(proc.stderr)
            if proc.returncode != 0:
                raise RuntimeError(f"hipcc failed on {src}")

        if jobs > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers = jobs) as pool:
                futures = [pool.submit(_compile, w) for w in work]
                errors = [f.exception() for f in concurrent.futures.as_completed(futures)]
            errors = [e for e in errors if e]
            if errors:
                # One line, not a stack of tracebacks from every worker that died.
                raise SystemExit(f"exllamav3: build failed -- {errors[0]}")
        else:
            for w in work:
                _compile(w)

        lib_args = [
            f"-L{os.path.join(self._torch_path, 'lib')}",
            f"-L{os.path.join(self._rocm_path, 'lib')}",
        ]
        if python_lib := sysconfig.get_config_var("LIBDIR"):
            lib_args.append(f"-L{python_lib}")

        link_libs = [
            "-lc10", "-ltorch", "-ltorch_cpu", "-ltorch_hip", "-ltorch_python",
            "-lc10_hip", "-lamdhip64", "-lhipblas", "-lrocblas", "-lhiprand",
        ]

        cmd = (["hipcc", "-shared", "-fgpu-rdc", "--hip-link", "-o", ext_path]
               + objs + lib_args + link_libs + ["-fPIC"])
        print(f"[link] {ext_path}", flush=True)
        if verbose:
            print("  " + " ".join(cmd))
        subprocess.check_call(cmd)


# ---------------------------------------------------------------------------
# CUDA build (unchanged from upstream)
# ---------------------------------------------------------------------------

extra_cflags = []
extra_cuda_cflags = [
    "-lineinfo", "-O3", "--use_fast_math",
    "-Xcudafe", "--diag_suppress=177",
    "-Xcudafe", "--diag_suppress=20012",
]

if windows:
    # NOMINMAX: windows.h otherwise defines min/max function-like macros that break every
    # std::min/std::max call site parsed after it (WIN32_LEAN_AND_MEAN does not suppress them).
    # Defined globally so it holds regardless of include order in any TU.
    # No -std flags here: torch's cpp_extension appends its own (unconditionally on the Windows
    # nvcc path), and a second -std argument is a fatal nvcc error, not an override.
    extra_cflags += ["/Ox", "/Zc:preprocessor", "/DWIN32_LEAN_AND_MEAN", "/DNOMINMAX"]
    extra_cuda_cflags += ["-DWIN32_LEAN_AND_MEAN", "-DNOMINMAX", "-Xcompiler=/Zc:preprocessor"]
    if ext_debug:
        extra_cflags += ["/Zi"]
else:
    extra_cflags += ["-Ofast"]
    if ext_debug:
        extra_cflags += ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]

if cuda_host_cxx := os.environ.get("CUDAHOSTCXX"):
    extra_cuda_cflags += ["-ccbin", cuda_host_cxx]

extra_compile_args = {
    "cxx": extra_cflags,
    "nvcc": extra_cuda_cflags,
}

if not (precompile and torch):
    setup_kwargs = {}
elif IS_ROCM:
    setup_kwargs = {
        "ext_modules": [HIPExtension(extension_name, sources=sources)],
        "cmdclass": {"build_ext": HIPBuildExtension},
    }
else:
    setup_kwargs = {
        "ext_modules": [
            cpp_extension.CUDAExtension(
                extension_name,
                sources,
                extra_compile_args=extra_compile_args,
                libraries=["cublas"] if windows else [],
            )
        ],
        "cmdclass": {"build_ext": cpp_extension.BuildExtension},
    }

# ---------------------------------------------------------------------------
# Metadata and dependencies live in pyproject.toml (upstream since v1.5.0),
# including torch>=2.6.0. On ROCm that requirement is satisfied by the
# pre-installed ROCm wheel (see requirements_rocm.txt), and a torch-less ROCm
# build stops above rather than letting pip resolve torch from PyPI (the CUDA
# build). `exllamav3.rocm_py` is picked up by pyproject's package discovery on
# both backends; it is pure Python whose first act is `if not is_rocm(): return`.
# ---------------------------------------------------------------------------


setup(
    verbose=verbose,
    **setup_kwargs,
)
