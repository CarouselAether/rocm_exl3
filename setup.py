from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import importlib.util
import os
import subprocess
import sys
import sysconfig

if torch := importlib.util.find_spec("torch") is not None:
    from torch.utils import cpp_extension
    from torch import version as torch_version
    import torch as _torch_mod

extension_name = "exllamav3_ext"
precompile = "EXLLAMA_NOCOMPILE" not in os.environ
verbose = "EXLLAMA_VERBOSE" in os.environ
ext_debug = "EXLLAMA_EXT_DEBUG" in os.environ

if precompile and not torch:
    print("Cannot precompile unless torch is installed.")
    print("To explicitly JIT install run EXLLAMA_NOCOMPILE= pip install <xyz>")

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

BACKEND = _resolve_backend() if precompile else "cuda"
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
# ROCM_EXCLUDE lists upstream paths the ROCm build skips because they are
# CUDA-cooperative-kernel or PTX-inline-asm paths with no HIP equivalent.
# Each entry needs a reason; an unexplained exclusion is a silently missing
# feature.
ROCM_EXCLUDE = (
    # 8 files. Multi-GPU peer kernels built on CUDA IPC + inline PTX. No HIP port
    # yet, so tensor-parallel is unavailable on ROCm.
    "parallel/",
    # 66 files. Per-bitwidth EXL3 GEMM instantiations using cooperative launches
    # and ~64 KB LDS, which stall grid.sync() on RDNA WGP pairing. ROCm replaces
    # them with rocm/quant/comp_units_rdna/.
    # NOTE: until that port lands, a ROCm build links but has no EXL3 quant
    # kernels -- exllamav3 will import and run unquantized paths only.
    "quant/comp_units/",
    # Replaced by rocm/rope_rdna.hip, which differs by one line: `half2 x = {}`
    # is ambiguous against HIP's assignment overloads. Same exported symbols.
    "rope.cu",
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

SUPPORTED_GPU_ARCHS = {
    # RDNA3 / RDNA3.5 / RDNA4 consumer + APU parts this port targets.
    "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
}

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

        archs = _resolve_offload_archs()
        print(f"exllamav3: offload archs = {archs or '(hipcc auto-detect)'}")

        defines = [
            "-DUSE_ROCM=1",
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

        common = [
            "-fPIC", "-std=c++17",
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
        for group, flags, label in ((cpp_sources, common, "cpp"), (gpu_sources, hip_flags, "hip")):
            for src in group:
                obj = os.path.join(build_temp, src.replace(os.sep, "_") + ".o")
                os.makedirs(os.path.dirname(obj), exist_ok=True)
                cmd = ["hipcc", "-c", src, "-o", obj] + flags + includes + defines
                if verbose:
                    print(f"[{label}] {src}")
                    print("  " + " ".join(cmd))
                else:
                    print(f"[{label}] {src}", flush=True)
                subprocess.check_call(cmd)
                objs.append(obj)

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

version_py = {}
with open("exllamav3/version.py", encoding="utf8") as fp:
    exec(fp.read(), version_py)
version = version_py["__version__"]
print("Version:", version)

setup(
    name="exllamav3",
    version=version,
    packages=[
        "exllamav3",
        "exllamav3.generator",
        "exllamav3.generator.sampler",
        "exllamav3.generator.filter",
        "exllamav3.conversion",
        "exllamav3.conversion.standard_cal_data",
        "exllamav3.integration",
        "exllamav3.architecture",
        "exllamav3.architecture.mm_processing",
        "exllamav3.model",
        "exllamav3.modules",
        "exllamav3.modules.attention_fn",
        "exllamav3.modules.arch_specific",
        "exllamav3.modules.gated_delta_net_fn",
        "exllamav3.modules.quant",
        "exllamav3.modules.quant.exl3_lib",
        "exllamav3.tokenizer",
        "exllamav3.cache",
        "exllamav3.loader",
        "exllamav3.util",
    ],
    url="https://github.com/turboderp-org/exllamav3",
    license="MIT",
    author="turboderp",
    install_requires=[
        "torch>=2.6.0",
        "tokenizers>=0.21.1",
        "numpy>=2.1.0",
        "rich",
        "typing_extensions",
        "safetensors>=0.3.2",
        "ninja",
        "pillow",
        "pyyaml",
        "marisa_trie",
        "pydantic",
        "llguidance>=1.7.0",
        "flash-linear-attention>=0.5.0",
    ],
    include_package_data=True,
    package_data={
        "": ["py.typed"],
    },
    verbose=verbose,
    **setup_kwargs,
)
