// Does hipMemcpyAsync from PAGEABLE host memory behave synchronously w.r.t. the host
// (CUDA contract), or can it read the buffer after the host has freed/reused it?
// Method: enqueue a long-running kernel to back up the stream, then hipMemcpyAsync from
// a pageable buffer, immediately overwrite the buffer, and check what the device saw.
// Build: hipcc -o pageable_async_test pageable_async_test.hip
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("FAIL %s rc=%d (%s) line %d\n", #x, e, hipGetErrorString(e), __LINE__); return 1; } } while (0)

__global__ void spin_kernel(long long cycles)
{
    long long start = clock64();
    while (clock64() - start < cycles) { }
}

int main()
{
    hipStream_t stream;
    CK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    int* dev;
    const int N = 1024;
    CK(hipMalloc(&dev, N * sizeof(int)));

    int corrupt = 0, iters = 200;
    for (int i = 0; i < iters; i++)
    {
        // Back up the stream (~5ms of GPU spin)
        hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(1), 0, stream, (long long) 10000000);

        int* host = (int*) malloc(N * sizeof(int));
        for (int j = 0; j < N; j++) host[j] = i;
        CK(hipMemcpyAsync(dev, host, N * sizeof(int), hipMemcpyHostToDevice, stream));
        // Host immediately overwrites and frees (mimics a temporary torch host tensor)
        for (int j = 0; j < N; j++) host[j] = -777;
        free(host);

        CK(hipStreamSynchronize(stream));
        int check[N];
        CK(hipMemcpy(check, dev, N * sizeof(int), hipMemcpyDeviceToHost));
        int bad = 0;
        for (int j = 0; j < N; j++) if (check[j] != i) bad++;
        if (bad) {
            corrupt++;
            if (corrupt <= 5)
                printf("iter %d: %d/%d values wrong (saw %d at [0])\n", i, bad, N, check[0]);
        }
    }
    printf("%d/%d iterations corrupted\n", corrupt, iters);
    printf(corrupt ? "hipMemcpyAsync from pageable memory is NOT host-synchronous -- HAZARD\n"
                   : "pageable async copies behaved host-synchronously (CUDA contract held)\n");
    return corrupt ? 2 : 0;
}
