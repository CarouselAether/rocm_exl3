// Do back-to-back hipGraphLaunch calls on ONE stream execute strictly in order?
// Graph A: y = f(x) (slow-ish kernel writing a shared intermediate)
// Graph B: z = g(y) (reads the shared intermediate)
// Alternate A,B,A,B... with distinct values per round; any stale read => ordering violation.
// Mirrors BC attention: per-layer graphs chained through shared static buffers.
// Build: hipcc -o graphrace_test graphrace_test.hip
#include <hip/hip_runtime.h>
#include <cstdio>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("FAIL %s rc=%d (%s) line %d\n", #x, e, hipGetErrorString(e), __LINE__); return 1; } } while (0)

#define N (1 << 20)

// Deliberately slow producer: repeated global-memory round trips
__global__ void producer(const float* in, float* mid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = in[0];
    for (int r = 0; r < 32; r++) v = v * 1.0000001f + 0.0000001f;
    mid[i] = v;
}

__global__ void consumer(const float* mid, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = mid[i];
}

int main()
{
    float *in, *mid, *out;
    CK(hipMalloc(&in, sizeof(float)));
    CK(hipMalloc(&mid, N * sizeof(float)));
    CK(hipMalloc(&out, N * sizeof(float)));

    hipStream_t stream;
    CK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    dim3 grid((N + 255) / 256), block(256);

    // Capture graph A (producer) and graph B (consumer) separately
    CK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    hipLaunchKernelGGL(producer, grid, block, 0, stream, in, mid);
    hipGraph_t ga; CK(hipStreamEndCapture(stream, &ga));
    hipGraphExec_t ea; CK(hipGraphInstantiate(&ea, ga, nullptr, nullptr, 0));

    CK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    hipLaunchKernelGGL(consumer, grid, block, 0, stream, mid, out);
    hipGraph_t gb; CK(hipStreamEndCapture(stream, &gb));
    hipGraphExec_t eb; CK(hipGraphInstantiate(&eb, gb, nullptr, nullptr, 0));

    int bad_rounds = 0;
    const int ROUNDS = 300;
    for (int r = 0; r < ROUNDS; r++)
    {
        float v = (float) (r + 1);
        CK(hipMemcpyAsync(in, &v, sizeof(float), hipMemcpyHostToDevice, stream));
        // Queue many alternations without host sync, like a decode loop over layers
        for (int k = 0; k < 8; k++)
        {
            CK(hipGraphLaunch(ea, stream));
            CK(hipGraphLaunch(eb, stream));
        }
        CK(hipStreamSynchronize(stream));

        float h0, h1;
        CK(hipMemcpy(&h0, out, sizeof(float), hipMemcpyDeviceToHost));
        CK(hipMemcpy(&h1, out + N - 1, sizeof(float), hipMemcpyDeviceToHost));
        float expect = v;
        for (int q = 0; q < 32; q++) expect = expect * 1.0000001f + 0.0000001f;
        if (h0 != expect || h1 != expect)
        {
            bad_rounds++;
            if (bad_rounds <= 5)
                printf("round %d: out[0]=%.7f out[N-1]=%.7f expect=%.7f  ** STALE/RACE **\n",
                       r, h0, h1, expect);
        }
    }
    printf("%d/%d rounds violated ordering\n", bad_rounds, ROUNDS);
    printf(bad_rounds ? "hipGraphLaunch inter-graph ordering BROKEN\n"
                      : "inter-graph ordering held on this stream\n");
    return bad_rounds ? 2 : 0;
}
