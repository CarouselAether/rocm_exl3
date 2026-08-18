// Micro-test 3: multi-node graph (mixed runtime + module kernels in sequence), patch a
// subset of nodes, ping-pong the patches across many replays — mirrors the real BC graph
// shape (14+ nodes, 2-3 patched at a job boundary, repeatedly).
// Build: hipcc -o graphpatch_test3 graphpatch_test3.hip   (needs probe_module.hsaco)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <fstream>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("FAIL %s rc=%d (%s) line %d\n", #x, e, hipGetErrorString(e), __LINE__); return 1; } } while (0)

__global__ void stage1(const float* in, float* mid) { mid[0] = in[0] * 2.0f; }
__global__ void stage3(const float* mid2, float* out, int scalar) { out[0] = mid2[0] + 1000.0f * scalar; }

int main()
{
    std::ifstream f("probe_module.hsaco", std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    hipModule_t mod; CK(hipModuleLoadData(&mod, code.data()));
    hipFunction_t fn_mod; CK(hipModuleGetFunction(&fn_mod, mod, "probe_kernel"));
    // probe_kernel(in, out, scalar): out[0]=in[0]; out[1]=scalar

    float *in_a, *in_b, *mid, *mid2, *out;
    CK(hipMalloc(&in_a, sizeof(float)));
    CK(hipMalloc(&in_b, sizeof(float)));
    CK(hipMalloc(&mid, sizeof(float)));
    CK(hipMalloc(&mid2, 2 * sizeof(float)));
    CK(hipMalloc(&out, sizeof(float)));
    float va = 1.0f, vb = 5.0f;
    CK(hipMemcpy(in_a, &va, sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(in_b, &vb, sizeof(float), hipMemcpyHostToDevice));

    hipStream_t stream; CK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    // Chain: stage1(in->mid) [runtime] ; probe_kernel(mid->mid2, scalar) [module] ;
    //        stage3(mid2->out, scalar3) [runtime]
    const float* a_in = in_a; float* a_mid = mid;
    void* args1[] = { &a_in, &a_mid };
    const float* a_mid_c = mid; float* a_mid2 = mid2; int a_scalar = 3;
    void* args2[] = { &a_mid_c, &a_mid2, &a_scalar };
    const float* a_mid2_c = mid2; float* a_out = out; int a_scalar3 = 7;
    void* args3[] = { &a_mid2_c, &a_out, &a_scalar3 };

    CK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    CK(hipLaunchKernel((void*) stage1, dim3(1), dim3(1), args1, 0, stream));
    CK(hipModuleLaunchKernel(fn_mod, 1, 1, 1, 64, 1, 1, 0, stream, args2, nullptr));
    CK(hipLaunchKernel((void*) stage3, dim3(1), dim3(1), args3, 0, stream));
    hipGraph_t graph; CK(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t exec; CK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    size_t num_nodes = 0; CK(hipGraphGetNodes(graph, nullptr, &num_nodes));
    std::vector<hipGraphNode_t> nodes(num_nodes);
    CK(hipGraphGetNodes(graph, nodes.data(), &num_nodes));
    printf("nodes=%zu\n", num_nodes);

    // GetParams once for all nodes (like capture_end); identify by func
    std::vector<hipKernelNodeParams> np(num_nodes);
    int n_stage1 = -1, n_mod = -1, n_stage3 = -1;
    for (size_t i = 0; i < num_nodes; i++) {
        CK(hipGraphKernelNodeGetParams(nodes[i], &np[i]));
        if (np[i].func == (void*) stage1) n_stage1 = i;
        else if (np[i].func == (void*) stage3) n_stage3 = i;
        else n_mod = i;
    }
    printf("order: stage1=%d mod=%d stage3=%d\n", n_stage1, n_mod, n_stage3);
    if (n_stage1 < 0 || n_mod < 0 || n_stage3 < 0) { printf("FAIL: node id\n"); return 1; }

    float host;
    auto expect = [&](float e, const char* tag) {
        CK(hipGraphLaunch(exec, stream));
        CK(hipStreamSynchronize(stream));
        CK(hipMemcpy(&host, out, sizeof(float), hipMemcpyDeviceToHost));
        printf("%-28s out=%8.0f expect=%8.0f  %s\n", tag, host, e, host == e ? "ok" : "** MISMATCH **");
        return 0;
    };

    // replay0: in_a=1 -> mid=2 -> mid2[0]=2 -> out=2+7000
    expect(7002, "replay0 (baseline)");

    // Ping-pong the stage1 input and stage3 scalar 10 times like job transitions
    for (int i = 0; i < 10; i++) {
        bool flip = i & 1;
        void* new_in = flip ? (void*) in_a : (void*) in_b;
        memcpy(np[n_stage1].kernelParams[0], &new_in, sizeof(void*));
        hipError_t r1 = hipGraphExecKernelNodeSetParams(exec, nodes[n_stage1], &np[n_stage1]);
        int new_s3 = flip ? 7 : 9;
        memcpy(np[n_stage3].kernelParams[2], &new_s3, sizeof(int));
        hipError_t r2 = hipGraphExecKernelNodeSetParams(exec, nodes[n_stage3], &np[n_stage3]);
        if (r1 || r2) printf("SetParams rc: %d %d\n", r1, r2);
        float e = (flip ? 1.0f : 5.0f) * 2.0f + 1000.0f * (flip ? 7 : 9);
        char tag[64]; snprintf(tag, 64, "replay%d (flip=%d)", i + 1, (int) flip);
        expect(e, tag);
    }
    return 0;
}
