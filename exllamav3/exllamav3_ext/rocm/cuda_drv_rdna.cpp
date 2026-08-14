// =============================================================================
// cuda_drv.cpp for RDNA -- driver-library name only
// =============================================================================
//
// Generated from cuda_drv.cpp. Exactly one thing changes: the library that gets
// dlopen'd. Everything else is already portable, which is worth understanding
// before "fixing" more of it:
//
//   - The struct members are declared `decltype(&cuModuleLoadData)` etc., and
//     hip_compat.hip.h #defines those names onto hipModuleLoadData and friends,
//     so the types are the HIP ones.
//   - DRV_STR stringifies AFTER macro expansion (that is what the two-level
//     DRV_STR2/DRV_STR dance is for), so DRV_STR(cuModuleLoadData) already
//     produces the string "hipModuleLoadData".
//
// So the symbol lookups were always correct on ROCm; only "libcuda.so.1" was
// wrong, and it failed as "Could not load the CUDA driver library" at the first
// Triton kernel compile in bc_attn.py.
//
// The HIP entry points are version-tagged in the .so (hipModuleLoadData@@hip_4.2).
// dlsym without a version resolves the default, so no versioned lookup is
// needed. Do not be misled by `nm -D | grep "T hipModuleLoadData$"` returning
// nothing -- the @@hip_4.2 suffix defeats an end-anchored pattern.
//
// libamdhip64 is already loaded into the process by torch by the time this
// runs; the dlopen just takes a handle to it.
// =============================================================================

#include <cstdio>
#include <c10/util/Exception.h>
#include "../cuda_drv.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define DRV_STR2(x) #x
#define DRV_STR(x) DRV_STR2(x)

static void* drv_sym(void* lib, const char* name)
{
    #ifdef _WIN32
        void* fp = (void*) GetProcAddress((HMODULE) lib, name);
    #else
        void* fp = dlsym(lib, name);
    #endif
    TORCH_CHECK(fp, "CUDA driver symbol not found: ", name);
    return fp;
}

const CudaDrv& CudaDrv::instance()
{
    static CudaDrv d = []
    {
        #ifdef _WIN32
            void* lib = (void*) LoadLibraryA("amdhip64.dll");
        #else
            // Prefer the runtime already linked into the process (torch's).
            // pip ROCm/torch wheels bundle libamdhip64 WITHOUT the .so dev
            // symlink, so a named dlopen resolves to the SYSTEM tree and
            // loads a second runtime instance beside torch's. Two instances
            // of the same version happen to interoperate -- which is how the
            // old order passed on a system-ROCm stack -- but mismatched
            // versions fail at first launch with hipErrorContextIsDestroyed
            // (709): found the hard way on torch 2.11+rocm7.13 wheels over
            // system ROCm 7.2.4. dlopen(nullptr) searches the global scope,
            // where torch's NEEDED libamdhip64 already lives; verify the
            // probe symbol resolves before trusting it, and keep the named
            // forms only as a fallback for processes where HIP is not yet
            // loaded.
            void* lib = dlopen(nullptr, RTLD_NOW | RTLD_GLOBAL);
            if (!lib || !dlsym(lib, DRV_STR(cuModuleLoadData)))
            {
                lib = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
                if (!lib) lib = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_GLOBAL);
            }
        #endif
        TORCH_CHECK(lib, "Could not load the HIP runtime library (libamdhip64.so)");

        CudaDrv d{};
        d.module_load_data                  = (decltype(&cuModuleLoadData))               drv_sym(lib, DRV_STR(cuModuleLoadData));
        d.module_unload                     = (decltype(&cuModuleUnload))                 drv_sym(lib, DRV_STR(cuModuleUnload));
        d.module_get_function               = (decltype(&cuModuleGetFunction))            drv_sym(lib, DRV_STR(cuModuleGetFunction));
        d.func_set_attribute                = (decltype(&cuFuncSetAttribute))             drv_sym(lib, DRV_STR(cuFuncSetAttribute));
        d.launch_kernel                     = (decltype(&cuLaunchKernel))                 drv_sym(lib, DRV_STR(cuLaunchKernel));
        d.graph_kernel_node_get_params      = (decltype(&cuGraphKernelNodeGetParams))     drv_sym(lib, DRV_STR(cuGraphKernelNodeGetParams));
        d.graph_exec_kernel_node_set_params = (decltype(&cuGraphExecKernelNodeSetParams)) drv_sym(lib, DRV_STR(cuGraphExecKernelNodeSetParams));
        return d;
    }
    ();
    return d;
}
