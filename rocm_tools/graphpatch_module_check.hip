// Micro-test 2: same patch flow, but the kernel is MODULE-launched (hipModuleLaunchKernel
// from a loaded code object) exactly like graph.cu captures Triton cubins, then patched
// through the RUNTIME GetParams/SetParams — the combination the BC graphs actually use
// (observed drv=0 for Triton nodes).
// Build:
//   hipcc --genco -o probe_module.hsaco probe_module.hip
//   hipcc -o graphpatch_test2 graphpatch_test2.hip
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <fstream>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("FAIL %s rc=%d (%s) line %d\n", #x, e, hipGetErrorString(e), __LINE__); return 1; } } while (0)

int main()
{
    std::ifstream f("probe_module.hsaco", std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    printf("hsaco bytes=%zu\n", code.size());
    if (code.empty()) { printf("FAIL: no hsaco\n"); return 1; }

    hipModule_t mod;
    CK(hipModuleLoadData(&mod, code.data()));
    hipFunction_t fn;
    CK(hipModuleGetFunction(&fn, mod, "probe_kernel"));

    float *in_a, *in_b, *out;
    CK(hipMalloc(&in_a, sizeof(float)));
    CK(hipMalloc(&in_b, sizeof(float)));
    CK(hipMalloc(&out, 2 * sizeof(float)));
    float va = 111.0f, vb = 222.0f;
    CK(hipMemcpy(in_a, &va, sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(in_b, &vb, sizeof(float), hipMemcpyHostToDevice));

    hipStream_t stream;
    CK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    const float* arg_in = in_a;
    float* arg_out = out;
    int arg_scalar = 10;
    void* kernel_args[] = { &arg_in, &arg_out, &arg_scalar };

    CK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, stream, kernel_args, nullptr));
    hipGraph_t graph;
    CK(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t exec;
    CK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    size_t num_nodes = 0;
    CK(hipGraphGetNodes(graph, nullptr, &num_nodes));
    hipGraphNode_t nodes[8];
    CK(hipGraphGetNodes(graph, nodes, &num_nodes));

    // Runtime-API GetParams on the module-launched node (graph.cu tries this first and
    // uses it whenever it succeeds)
    hipKernelNodeParams np{};
    hipError_t grc = hipGraphKernelNodeGetParams(nodes[0], &np);
    printf("runtime GetParams rc=%d (%s) func=%p kernelParams=%p\n",
           grc, hipGetErrorString(grc), np.func, (void*) np.kernelParams);
    if (grc != hipSuccess) { printf("runtime path unavailable -> graph.cu would use drv path\n"); return 3; }

    float host[2];
    CK(hipGraphLaunch(exec, stream));
    CK(hipStreamSynchronize(stream));
    CK(hipMemcpy(host, out, sizeof(host), hipMemcpyDeviceToHost));
    printf("replay0: in=%.0f scalar=%.0f (expect 111 10)\n", host[0], host[1]);

    void* new_in = (void*) in_b;
    memcpy(np.kernelParams[0], &new_in, sizeof(void*));
    int new_scalar = 77;
    memcpy(np.kernelParams[2], &new_scalar, sizeof(int));
    hipError_t rc = hipGraphExecKernelNodeSetParams(exec, nodes[0], &np);
    printf("SetParams rc=%d (%s)\n", rc, hipGetErrorString(rc));

    CK(hipGraphLaunch(exec, stream));
    CK(hipStreamSynchronize(stream));
    CK(hipMemcpy(host, out, sizeof(host), hipMemcpyDeviceToHost));
    printf("replay1: in=%.0f scalar=%.0f (expect 222 77)\n", host[0], host[1]);

    bool ok = host[0] == 222.0f && host[1] == 77.0f;
    printf(ok ? "MODULE-NODE PATCH EFFECTIVE\n" : "MODULE-NODE PATCH **INEFFECTIVE** -- silent no-op\n");
    return ok ? 0 : 2;
}
