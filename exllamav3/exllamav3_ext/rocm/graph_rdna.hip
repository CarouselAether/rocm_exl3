// =============================================================================
// graph_rdna.hip -- Graph class sibling for ROCm: HIP graph capture/replay
// gated on the loaded HIP runtime (off below 7.14, on from 7.14 / ROCm 10),
// with eager passthrough when off
// =============================================================================
//
// Generated from graph.cu (which stays byte-identical to upstream; setup.py
// excludes it from the ROCm build in favor of this file). The diff against
// upstream is confined to the four hip_graphs_enabled() branches below plus
// this comment block; everything else is a verbatim copy so the file can be
// regenerated mechanically on rebase and audited by diff.
//
// WHY (measured 2026-08-15, gfx1151, ROCm 7.2.4, torch 2.13+rocm7.2):
//
// Replaying BC decode graphs across generator jobs corrupts output. Six-turn
// chat-style probes (Laguna-S-2.1) degenerate into babble by the final turn
// with graphs on, and are clean with EXL3_BC_ATTN=0 -- 5/5 reproducible,
// deterministic per probe. The corruption is NOT in this class's bookkeeping;
// each layer was ruled out by direct experiment on this machine:
//
//   - hipGraphExecKernelNodeSetParams is effective (rc=0 and semantically
//     applied) for runtime-launched and module-launched (Triton) nodes,
//     including multi-node graphs and repeated ping-pong patches
//     (rocm_tools-adjacent micro-tests graphpatch_test{,2,3}.hip);
//   - capture-time site->node matching resolves correctly (all BC nodes are
//     readable through the runtime API on HIP);
//   - forcing a full RECAPTURE of every slot on every new job still corrupts,
//     so it is not stale pointers or stale exec state;
//   - de-sharing the per-layer static buffers still corrupts, so it is not a
//     cross-graph race on the statics;
//   - plain graph->graph ordering through a shared buffer holds over hundreds
//     of deep-queued rounds (graphrace_test.hip).
//
// What remains is the HIP runtime / hipified-torch layer itself: captures that
// reference torch-caching-allocator state, and HIP's missing capture-time
// validation (CUDA raises "operation not permitted when stream is capturing"
// where HIP silently captures -- see pytorch#155684). Fresh captures taken in a
// process with allocator churn produce corrupt replays; first-job captures are
// clean. Capture also intermittently HANGS on this stack (observed here as
// lost prompt-processing speed; same class as vLLM's open capture-hang issue
// on ROCm 7.2.x). llama.cpp ships HIP graphs off by default behind a build
// flag for the same reasons.
//
// WHAT THIS SIBLING DOES: by default on ROCm, capture never begins. Every
// caller of this class follows the idiom
//
//     if (!graph->ready) { capture_begin(); run_gr(..., graph); capture_end(); }
//     graph->launch(params, stream);
//
// so with capture_begin() returning the current torch stream without starting
// a capture, ready never becoming true, record_param() a no-op and launch()
// returning immediately, run_gr() simply EXECUTES eagerly on the live stream
// every call -- the same code path as each slot's first (warmup) run, which is
// already exercised and correct. The fused single-C++-call decode step is
// preserved; only hipGraph replay is dropped, along with capture hangs, by
// construction.
//
// On runtimes >= 7.14 the gate below turns capture/replay back on by default;
// EXL3_ROCM_HIP_GRAPHS=1 / =0 overrides it either way (bisect handle).
// =============================================================================

#include <Python.h>
#include <cstring>
#include <cstdlib>
#include "graph.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
//#include <torch/extension.h>
#include "util.h"
#include "util.cuh"
#include "cuda_drv.h"
#include "quant/exl3_devctx.cuh"

//#define GRAPHDEBUG 1

// ROCm sibling: HIP graph capture is gated on the HIP runtime that is actually
// loaded into the process, not on a build-time constant, so one binary does the
// right thing on both stacks this port supports:
//
//   runtime <  7.14  (system ROCm 7.2.x)   eager passthrough. Capture hangs and
//                                          replays corrupt across jobs there; see
//                                          the file header.
//   runtime >= 7.14  (ROCm 7.14, 10.x)     capture/replay ON. Validated 2026-08-27
//                                          on 7.14 and 2026-09-22 on 10.0 (HIP
//                                          7.15): all graphpatch checks, ordering
//                                          check 0/300, coherent generation.
//
// hipRuntimeGetVersion packs MAJOR*10^7 + MINOR*10^5 + PATCH. Measured: 7.2.4
// reports 70253211, 7.14 reports 71460850, ROCm 10.0 (HIP 7.15) reports 71526333.
//
// EXL3_ROCM_HIP_GRAPHS=1 / =0 overrides the gate in either direction (bisect
// handle); unset or empty means "decide from the runtime".
#define EXL3_ROCM_GRAPHS_MIN_RUNTIME 71400000

static bool hip_graphs_enabled()
{
    static int v = -1;
    if (v < 0)
    {
        const char* e = getenv("EXL3_ROCM_HIP_GRAPHS");
        if (e && *e)
        {
            v = (*e == '1') ? 1 : 0;
        }
        else
        {
            int rt = 0;
            v = (hipRuntimeGetVersion(&rt) == hipSuccess && rt >= EXL3_ROCM_GRAPHS_MIN_RUNTIME) ? 1 : 0;
        }
    }
    return v == 1;
}

Graph::Graph()
{
    ready = false;
    ready_to_record = false;
    disabled = false;
    graph = NULL;
    graph_exec = NULL;
    need_cublas = false;
}

Graph::~Graph()
{
    if (graph) cudaGraphDestroy(graph);
    if (graph_exec) cudaGraphExecDestroy(graph_exec);
}

cudaStream_t Graph::capture_begin()
{
    // ROCm sibling: passthrough mode never begins a capture. run_gr() launches
    // on the live torch stream and the work executes immediately; capture_end()
    // leaves ready false so the caller re-enters this path every call.
    if (!hip_graphs_enabled())
    {
        capture_stream = at::cuda::getCurrentCUDAStream().stream();
        return capture_stream;
    }

    #ifdef GRAPHDEBUG
        printf("Begin graph capture\n");
    #endif

    // Make sure nothing is pending
    cudaDeviceSynchronize();

    // Create capture stream
    cuda_check(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking));

    // Begin capture
    cuda_check(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal));
    return capture_stream;
}

void Graph::capture_end()
{
    // ROCm sibling: nothing was captured; the step's work already ran eagerly
    if (!hip_graphs_enabled())
        return;

    // End capture
    cuda_check(cudaStreamEndCapture(capture_stream, &graph));
    cuda_check(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    //inspect_graph();

    // Get graph nodes
    size_t num_nodes;
    cudaGraphGetNodes(graph, nullptr, &num_nodes);
    nodes.resize(num_nodes);
    cudaGraphGetNodes(graph, nodes.data(), &num_nodes);

    // Store copies of all node param structures
    node_params.resize(num_nodes);
    node_params_drv.resize(num_nodes);
    node_is_driver.resize(num_nodes);
    node_needs_update.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
        node_needs_update[i] = false;

    int n = 0;
    int c = 0;
    while (true)
    {
        cudaGraphNodeType t{};
        cudaGraphNodeGetType(nodes[n], &t);

        // Node type: kernel
        if (t == cudaGraphNodeTypeKernel)
        {
            // Nodes captured from driver-API launches (Triton cubins) can't be read through the
            // runtime API; fall back to the driver API for those. The func handle recorded via
            // record_param is the CUfunction in that case, so matching works the same way
            void* node_func;
            node_is_driver[n] = 0;
            cudaError_t e = cudaGraphKernelNodeGetParams(nodes[n], &node_params[n]);
            if (e == cudaSuccess)
                node_func = (void*) node_params[n].func;
            else
            {
                (void) cudaGetLastError();
                cuda_check_drv(CudaDrv::instance().graph_kernel_node_get_params((CUgraphNode) nodes[n], &node_params_drv[n]));
                node_is_driver[n] = 1;
                node_func = (void*) node_params_drv[n].func;
            }

            for(; c < graph_sites.size(); c++)
            {
                void* func = std::get<0>(graph_sites[c]);
                if (func != node_func) break;

                int param_id     = std::get<1>(graph_sites[c]);
                int param_offset = std::get<2>(graph_sites[c]);
                int param_size   = std::get<3>(graph_sites[c]);

                graph_node_sites.push_back(std::tuple<int, int, int, int>(n, param_id, param_offset, param_size));
                if (param_id == GP_end) { c++; break; }
            }
        }

        n++;
        if (c == graph_sites.size()) break;
        if (n == num_nodes) TORCH_CHECK(false, "Graph recording failed");
    };

    // Destroy capture stream
    cuda_check(cudaStreamDestroy(capture_stream));

    // Graph is ready
    ready = true;

    #ifdef GRAPHDEBUG
        printf("End graph capture, num_nodes=%d, graph_sites.size()=%d\n", num_nodes, graph_sites.size());
    #endif
}

void Graph::record_param(void* kernel, int param_id, int param_offset, int size)
{
    // ROCm sibling: no capture, no patch sites; also keeps graph_sites from
    // growing without bound across the repeated passthrough "captures"
    if (!hip_graphs_enabled())
        return;

    graph_sites.push_back(std::tuple<void*, int, int, int>(kernel, param_id, param_offset, size));
}

void Graph::launch(std::vector<PPTR> params, cudaStream_t stream)
{
    // ROCm sibling: in passthrough mode the step's kernels were already
    // launched eagerly by run_gr() inside the caller's capture branch
    if (!ready)
        return;

    if (need_cublas)
    {
        cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublas_handle, stream);
        cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
        int device;
        cudaGetDevice(&device);
        void* ws = DevCtx::instance().get_ws(device);
        cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);
    }

    int p = 0;
    int n = 0;
    while (true)
    {
        if (std::get<1>(graph_node_sites[n]) == std::get<0>(params[p]))
        {
            if (std::get<0>(params[p]) != GP_end)
            {
                void* new_value  = std::get<1>(params[p]);
                int node_idx     = std::get<0>(graph_node_sites[n]);
                int param_offset = std::get<2>(graph_node_sites[n]);
                int param_size   = std::get<3>(graph_node_sites[n]);

                void** kernel_params = node_is_driver[node_idx] ?
                    node_params_drv[node_idx].kernelParams :
                    node_params[node_idx].kernelParams;
                void* p_old_value = kernel_params[param_offset];
                if (memcmp(p_old_value, &new_value, param_size))
                {
                    memcpy(p_old_value, &new_value, param_size);
                    node_needs_update[node_idx] = true;
                }
            }
            p++;
        }

        n++;
        if (p == params.size()) break;
        if (n == graph_node_sites.size()) TORCH_CHECK(false, "Graph update failed");
    }

    for (int n = 0; n < nodes.size(); ++n)
    {
        if (!node_needs_update[n]) continue;
        if (node_is_driver[n])
        {
            CUresult r = CudaDrv::instance().graph_exec_kernel_node_set_params((CUgraphExec) graph_exec, (CUgraphNode) nodes[n], &node_params_drv[n]);
            TORCH_CHECK(r == CUDA_SUCCESS, "Graph node parameter update failed (driver), CUresult ", (int) r);
        }
        else
            cuda_check(cudaGraphExecKernelNodeSetParams(graph_exec, nodes[n], &node_params[n]));
        node_needs_update[n] = false;
    }

    cuda_check(cudaGraphLaunch(graph_exec, stream));
}

void Graph::inspect_graph()
{
    // Get the number of nodes in the graph
    size_t numNodes;
    cudaGraphGetNodes(graph, nullptr, &numNodes);

    // Get the nodes in the graph
    std::vector<cudaGraphNode_t> nodes(numNodes);
    cudaGraphGetNodes(graph, nodes.data(), &numNodes);
    DBGI(nodes.size());

    // Inspect each node
    for (size_t i = 0; i < numNodes; ++i)
    {
        cudaGraphNodeType nodeType;
        cudaGraphNodeGetType(nodes[i], &nodeType);

        if (nodeType == cudaGraphNodeTypeKernel)
        {
            cudaKernelNodeParams nodeParams;
            cudaGraphKernelNodeGetParams(nodes[i], &nodeParams);
            std::cout << "Kernel node " << i << ":" << std::endl;
            std::cout << "  Function pointer: " << nodeParams.func << std::endl;
            std::cout << "  Grid dimensions: (" << nodeParams.gridDim.x << ", " << nodeParams.gridDim.y << ", " << nodeParams.gridDim.z << ")" << std::endl;
            std::cout << "  Block dimensions: (" << nodeParams.blockDim.x << ", " << nodeParams.blockDim.y << ", " << nodeParams.blockDim.z << ")" << std::endl;
            std::cout << "  Shared memory: " << nodeParams.sharedMemBytes << " bytes" << std::endl;

        } else {
            std::cout << "Node " << i << " is not a kernel node." << std::endl;
        }
    }
}
