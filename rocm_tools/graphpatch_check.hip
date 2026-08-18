// Micro-test: does hipGraphExecKernelNodeSetParams actually change a kernel argument,
// using EXACTLY the flow graph.cu uses?
//   1. capture a kernel launch (runtime API, explicit arg-pointer array) into a graph
//   2. instantiate exec
//   3. hipGraphKernelNodeGetParams ONCE, cache the struct
//   4. mutate the cached kernelParams storage (new input pointer / new scalar)
//   5. hipGraphExecKernelNodeSetParams with the cached struct
//   6. replay and check which values the kernel saw
// Build: hipcc -o graphpatch_test graphpatch_test.hip
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("FAIL %s rc=%d (%s) at line %d\n", #x, e, hipGetErrorString(e), __LINE__); return 1; } } while (0)

__global__ void probe_kernel(const float* in, float* out, int scalar)
{
    out[0] = in[0];
    out[1] = (float) scalar;
}

int main()
{
    float *in_a, *in_b, *out;
    CK(hipMalloc(&in_a, sizeof(float)));
    CK(hipMalloc(&in_b, sizeof(float)));
    CK(hipMalloc(&out, 2 * sizeof(float)));
    float va = 111.0f, vb = 222.0f;
    CK(hipMemcpy(in_a, &va, sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(in_b, &vb, sizeof(float), hipMemcpyHostToDevice));

    hipStream_t stream;
    CK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    // Capture: launch with in_a / scalar 10, same launch style as graph.cu's runtime path
    const float* arg_in = in_a;
    float* arg_out = out;
    int arg_scalar = 10;
    void* kernel_args[] = { &arg_in, &arg_out, &arg_scalar };

    CK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    CK(hipLaunchKernel((void*) probe_kernel, dim3(1), dim3(1), kernel_args, 0, stream));
    hipGraph_t graph;
    CK(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t exec;
    CK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    size_t num_nodes = 0;
    CK(hipGraphGetNodes(graph, nullptr, &num_nodes));
    hipGraphNode_t nodes[8];
    CK(hipGraphGetNodes(graph, nodes, &num_nodes));
    printf("nodes=%zu\n", num_nodes);

    hipKernelNodeParams np{};
    CK(hipGraphKernelNodeGetParams(nodes[0], &np));
    printf("func=%p kernelParams=%p\n", np.func, (void*) np.kernelParams);

    // Replay unmodified
    float host[2];
    CK(hipGraphLaunch(exec, stream));
    CK(hipStreamSynchronize(stream));
    CK(hipMemcpy(host, out, sizeof(host), hipMemcpyDeviceToHost));
    printf("replay0: in=%.0f scalar=%.0f (expect 111 10)\n", host[0], host[1]);

    // Patch exactly like Graph::launch: mutate cached storage, then SetParams
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
    printf(ok ? "PATCH EFFECTIVE\n" : "PATCH **INEFFECTIVE** -- silent no-op\n");
    return ok ? 0 : 2;
}
