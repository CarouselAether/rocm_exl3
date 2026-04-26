from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import importlib.util
import os
import sys
import subprocess
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
is_rocm = bool(torch and getattr(torch_version, "hip", None))

library_dir = "exllamav3"
sources_dir = os.path.join(library_dir, extension_name)

# ---------------------------------------------------------------------------
# Source file collection
# ---------------------------------------------------------------------------
# On ROCm we exclude upstream CUDA-only files that have HIP siblings or
# that pull in NVIDIA PTX (parallel/ is multi-GPU only, not needed on a
# single-GPU ROCm setup).
ROCM_EXCLUDE_PATTERNS = (
    os.sep + "parallel" + os.sep,
    os.sep + "comp_units" + os.sep,
    os.sep + "reconstruct.cu",          # replaced by reconstruct_rdna.hip
    os.sep + "exl3_kernel_map.cu",      # replaced by exl3_kernel_map_rdna.hip
)

# On CUDA we exclude everything HIP-specific. HIP sources live in .hip
# and .hip.h files (the extension is the RDNA marker, per the project-wide
# rename from _rdna.cu/_rdna.cuh). comp_units_rdna/ stays excluded by dir.
CUDA_EXCLUDE_PATTERNS = (
    os.sep + "comp_units_rdna" + os.sep,
    ".hip",    # matches .hip sources and .hip.h headers alike via endswith
)

def _want_source(path):
    if is_rocm:
        if any(pat in path for pat in ROCM_EXCLUDE_PATTERNS):
            return False
    else:
        # For CUDA exclusions: check substring match on the dir pattern,
        # suffix match on extensions. .hip* covers .hip and .hip.h.
        if os.sep + "comp_units_rdna" + os.sep in path:
            return False
        if path.endswith(".hip") or path.endswith(".hip.h"):
            return False
    return True

# On ROCm we also want .hip files (rdna_wmma.hip, *_rdna.hip). hipcc treats
# them like .cu. Headers (.hip.h) are not compilation units so not listed.
_SOURCE_SUFFIXES = (".c", ".cpp", ".cu", ".hip") if is_rocm else (".c", ".cpp", ".cu")

sources = [
    os.path.relpath(os.path.join(root, f), start=os.path.dirname(__file__))
    for root, _, files in os.walk(sources_dir)
    for f in files
    if f.endswith(_SOURCE_SUFFIXES)
]
sources = [s for s in sources if _want_source(s)]

if verbose:
    print("Sources:")
    for s in sorted(sources):
        print(" ", s)

# ---------------------------------------------------------------------------
# Extension configuration
# ---------------------------------------------------------------------------
setup_kwargs = {}

if precompile and torch:
    if is_rocm:
        # -----------------------------------------------------------------
        # ROCm path: bypass hipify and invoke hipcc directly. PyTorch's
        # CUDAExtension runs hipify-python over .cu sources under ROCm,
        # which breaks our compat layer (macros would be double-
        # substituted). The custom HIPBuildExtension below compiles every
        # .cu/.hip with hipcc unchanged.
        # -----------------------------------------------------------------

        # Supported GPU architectures. Aligns with flash-attention's
        # allowed_archs: RDNA3+ (gfx11+, gfx12+) and CDNA2+ (gfx90a, gfx94x,
        # gfx95x). Pre-RDNA3 (gfx1030 etc.) and pre-CDNA2 (gfx906 etc.) lack
        # the WMMA / matrix-core ISA the kernels depend on and will either
        # fail to compile or produce kernels that crash at runtime.
        SUPPORTED_GPU_ARCHS = {
            "gfx90a", "gfx942", "gfx950",
            "gfx1100", "gfx1101", "gfx1102",
            "gfx1150", "gfx1151",
            "gfx1200", "gfx1201",
        }

        def _resolve_offload_archs():
            """Return list of --offload-arch values to compile for.

            GPU_ARCHS env var: explicit semicolon-separated list. Validated
            against SUPPORTED_GPU_ARCHS; unsupported entries hard-fail.

            Otherwise auto-detect via rocminfo and filter to supported. On
            mixed-generation systems this excludes pre-RDNA3 GPUs with a
            warning rather than letting hipcc try to build for them.
            """
            env = os.environ.get("GPU_ARCHS", "").strip()
            if env:
                requested = [a.strip().lower() for a in env.split(";") if a.strip()]
                bad = [a for a in requested if a not in SUPPORTED_GPU_ARCHS]
                if bad:
                    raise RuntimeError(
                        f"GPU_ARCHS contains unsupported arch(s): {bad}. "
                        f"Supported: {sorted(SUPPORTED_GPU_ARCHS)}. "
                        f"Pre-RDNA3 / pre-CDNA2 GPUs lack the matrix-core ISA "
                        f"this kernel suite requires."
                    )
                return requested

            # Auto-detect: parse rocminfo for all installed gfx archs.
            try:
                ri = subprocess.check_output(
                    ["/opt/rocm/bin/rocminfo"], text=True, stderr=subprocess.DEVNULL
                )
            except Exception:
                # No rocminfo (e.g. CI sysroot build). Let hipcc auto-detect
                # at runtime; the per-call --offload-arch will just be omitted.
                return []

            detected = set()
            for line in ri.splitlines():
                line = line.strip()
                # rocminfo prints lines like "  Name:                    gfx1151"
                if line.startswith("Name:") and "gfx" in line:
                    arch = line.split()[1].strip().lower()
                    detected.add(arch)

            supported = sorted(detected & SUPPORTED_GPU_ARCHS)
            unsupported = sorted(detected - SUPPORTED_GPU_ARCHS)
            if unsupported:
                print(
                    f"[setup.py] WARNING: ignoring unsupported GPU arch(s): "
                    f"{unsupported}. Building only for: {supported or '(none)'}.",
                    file=sys.stderr,
                )
            if not supported:
                raise RuntimeError(
                    f"No supported GPU arch detected by rocminfo. Detected: "
                    f"{sorted(detected) or '(empty)'}. Supported: "
                    f"{sorted(SUPPORTED_GPU_ARCHS)}. Set GPU_ARCHS explicitly "
                    f"to override."
                )
            return supported

        OFFLOAD_ARCHS = _resolve_offload_archs()

        class HIPExtension(Extension):
            pass

        class HIPBuildExtension(build_ext):
            def build_extensions(self):
                torch_path = os.path.dirname(_torch_mod.__file__)
                torch_include = [
                    os.path.join(torch_path, "include"),
                    os.path.join(torch_path, "include", "torch", "csrc", "api", "include"),
                    os.path.join(torch_path, "include", "TH"),
                    os.path.join(torch_path, "include", "THC"),
                ]
                rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
                hip_include = os.path.join(rocm_path, "include")

                for ext in self.extensions:
                    if isinstance(ext, HIPExtension):
                        self._build_hip_extension(ext, torch_include, hip_include, rocm_path, torch_path)
                    else:
                        super().build_extension(ext)

            def _build_hip_extension(self, ext, torch_include, hip_include, rocm_path, torch_path):
                ext_path = self.get_ext_fullpath(ext.name)
                os.makedirs(os.path.dirname(ext_path), exist_ok=True)

                cpp_sources = [s for s in ext.sources if s.endswith((".c", ".cpp"))]
                cu_sources  = [s for s in ext.sources if s.endswith((".cu", ".hip"))]

                include_dirs = [
                    *torch_include,
                    hip_include,
                    os.path.join(rocm_path, "include", "hip"),
                    sysconfig.get_path("include"),
                ]

                defines = [
                    "-DUSE_ROCM=1",
                    "-D__HIP_PLATFORM_AMD__=1",
                    "-DHIPBLAS_V2",
                    "-DCUDA_HAS_FP16=1",
                    "-D__HIP_NO_HALF_OPERATORS__=1",
                    "-D__HIP_NO_HALF_CONVERSIONS__=1",
                    # Intentionally NOT setting HIP_ENABLE_WARP_SYNC_BUILTINS:
                    # it makes HIP declare __shfl_*_sync with a uint64 mask
                    # which clashes with upstream calls using 0xffffffff
                    # literals. Our hip_compat.cuh redefines the names as
                    # macros that drop the mask and call __shfl_* directly.
                    # Belt-and-suspenders: also disable the template-based
                    # warp-sync builtins entirely so our compat macros are
                    # the single source of truth (guide §9.4.8, p. 76).
                    "-DHIP_DISABLE_WARP_SYNC_BUILTINS=1",
                    "-DHIPBLAS_USE_HIP_HALF",
                    "-DTORCH_API_INCLUDE_EXTENSION_H",
                    "-DTORCH_USE_HIP_DSA",
                    f"-DTORCH_EXTENSION_NAME={ext.name}",
                ]

                # Quiet noise from upstream code paths we don't own. These are
                # the warnings that fire by the thousand on a clean build:
                #   - unused-command-line-argument: PyTorch/HIP shim flags that
                #     clang doesn't understand for a given source.
                #   - deprecated-declarations: CUDA APIs we shim via hip_compat.
                #   - unused-variable / unused-function: heavy template code.
                #   - missing-field-initializers: PyTorch ATen structs.
                #   - pragma-messages: hipBLAS_V2 transition pragmas.
                # Set EXLLAMAV3_VERBOSE_BUILD=1 to keep all warnings visible.
                quiet_warnings = (
                    [] if os.environ.get("EXLLAMAV3_VERBOSE_BUILD") == "1" else
                    [
                        "-Wno-unused-command-line-argument",
                        "-Wno-deprecated-declarations",
                        "-Wno-unused-variable",
                        "-Wno-unused-function",
                        "-Wno-unused-value",  # nodiscard hipError_t in void ctx
                        "-Wno-missing-field-initializers",
                        "-Wno-#pragma-messages",
                        "-Wno-pass-failed",
                    ]
                )
                cxx_flags = ["-O3", "-fPIC", "-std=c++17", "-Wno-register"] + quiet_warnings
                # Pin --offload-arch explicitly to the supported, detected list.
                # Without this, hipcc tries to build for every installed GPU,
                # which on mixed-generation rigs (e.g. 7900 XTX + older card)
                # fails on the unsupported arch.
                offload_arch_flags = [f"--offload-arch={a}" for a in OFFLOAD_ARCHS]

                hip_flags = [
                    "-O3", "-fPIC", "-std=c++17",
                    # Relocatable device code — needed for cooperative kernel
                    # launches in exl3_moe.
                    "-fgpu-rdc",
                    # Upstream attention.cu uses C++17-deprecated `register`
                    # declarations; upstream treats the warning as harmless,
                    # hipcc treats it as an error by default.
                    "-Wno-register",
                ] + offload_arch_flags + quiet_warnings
                include_args = [f"-I{d}" for d in include_dirs]

                obj_files = []
                build_temp = self.build_temp
                os.makedirs(build_temp, exist_ok=True)

                # Compile host sources with hipcc too. We can't use g++ because
                # hip_compat.cuh transitively pulls <hip/hip_bf16.h>, which uses
                # clang __builtin_elementwise_* intrinsics that g++ doesn't know.
                # hipcc in host mode dispatches to clang which handles them.
                for src in cpp_sources:
                    obj = os.path.join(build_temp, src.replace(os.sep, "_") + ".o")
                    os.makedirs(os.path.dirname(obj), exist_ok=True)
                    cmd = ["hipcc", "-c", src, "-o", obj] + cxx_flags + include_args + defines
                    if verbose:
                        print(f"[cpp] {src}")
                        print(" ", " ".join(cmd))
                    subprocess.check_call(cmd)
                    obj_files.append(obj)

                # Compile HIP/CUDA sources with hipcc (no hipify)
                for src in cu_sources:
                    obj = os.path.join(build_temp, src.replace(os.sep, "_") + ".o")
                    os.makedirs(os.path.dirname(obj), exist_ok=True)
                    cmd = ["hipcc", "-c", src, "-o", obj] + hip_flags + include_args + defines
                    if verbose:
                        print(f"[hip] {src}")
                        print(" ", " ".join(cmd))
                    subprocess.check_call(cmd)
                    obj_files.append(obj)

                # Link
                torch_lib_path = os.path.join(torch_path, "lib")
                lib_args = [
                    f"-L{torch_lib_path}",
                    f"-L{os.path.join(rocm_path, 'lib')}",
                ]
                python_lib = sysconfig.get_config_var("LIBDIR")
                if python_lib:
                    lib_args.append(f"-L{python_lib}")

                link_libs = [
                    "-lc10", "-ltorch", "-ltorch_cpu", "-ltorch_hip", "-ltorch_python",
                    "-lc10_hip", "-lamdhip64", "-lhipblas", "-lrocblas", "-lhiprand",
                ]

                cmd = (
                    ["hipcc", "-shared", "-fgpu-rdc", "--hip-link", "-o", ext_path]
                    + obj_files + lib_args + link_libs + ["-fPIC"]
                )
                if verbose:
                    print(f"[link] {ext_path}")
                    print(" ", " ".join(cmd))
                subprocess.check_call(cmd)

        setup_kwargs["ext_modules"] = [HIPExtension(extension_name, sources=sources)]
        setup_kwargs["cmdclass"] = {"build_ext": HIPBuildExtension}

    else:
        # -----------------------------------------------------------------
        # CUDA path: stock PyTorch CUDAExtension
        # -----------------------------------------------------------------
        extra_cflags = []
        extra_cuda_cflags = ["-lineinfo", "-O3"]

        if windows:
            extra_cflags += ["/Ox"]
            if ext_debug:
                extra_cflags += ["/Zi"]
        else:
            extra_cflags += ["-Ofast"]
            if ext_debug:
                extra_cflags += ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]

        extra_compile_args = {
            "cxx": extra_cflags,
            "nvcc": extra_cuda_cflags,
        }

        setup_kwargs["ext_modules"] = [
            cpp_extension.CUDAExtension(
                extension_name,
                sources,
                extra_compile_args=extra_compile_args,
                libraries=["cublas"] if windows else [],
            )
        ]
        setup_kwargs["cmdclass"] = {"build_ext": cpp_extension.BuildExtension}

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------
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
        "exllamav3.modules.arch_specific",
        "exllamav3.modules.quant",
        "exllamav3.modules.quant.exl3_lib",
        "exllamav3.tokenizer",
        "exllamav3.cache",
        "exllamav3.loader",
        "exllamav3.util",
    ],
    url="https://github.com/turboderp/exllamav3",
    license="MIT",
    author="turboderp",
    install_requires=[
        "torch>=2.6.0",
        "flash_attn>=2.7.4.post1",
        "tokenizers>=0.21.1",
        "numpy>=2.1.0",
        "rich",
        "typing_extensions",
        "ninja",
        "safetensors>=0.3.2",
        "pyyaml",
        "marisa_trie",
        "kbnf>=0.4.2",
        "formatron>=0.5.0",
        "pydantic==2.11.0",
        "xformers"
    ],
    include_package_data=True,
    package_data = {
        "": ["py.typed"],
    },
    verbose=verbose,
    **setup_kwargs,
)
