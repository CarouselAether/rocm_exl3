// Standalone smoke test for RDNA 3.5 WMMA intrinsic
// Build: hipcc -o wmma_smoke wmma_smoke.cpp --offload-arch=gfx1151 -std=c++17

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <math.h>

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        printf("HIP error %s at line %d: %s\n", #cmd, __LINE__, hipGetErrorString(e)); \
        return 1; \
    } \
} while(0)

typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef float    float8_t  __attribute__((ext_vector_type(8)));

__global__ void wmma_test_kernel(float* out)
{
    int lane = threadIdx.x & 31;
    int row  = lane & 15;

    // A: identity matrix — row L has 1.0 at column L
    half16_t a;
    for (int i = 0; i < 16; i++)
        ((_Float16*)&a)[i] = (_Float16)(i == row ? 1.0f : 0.0f);

    // B: identity matrix — column L has 1.0 at row L
    half16_t b;
    for (int i = 0; i < 16; i++)
        ((_Float16*)&b)[i] = (_Float16)(i == row ? 1.0f : 0.0f);

    float8_t c = {0,0,0,0,0,0,0,0};

    // Intrinsic arg order: (B, A, C)
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b, a, c);

    // Store: lane 0-15 -> even cols, lane 16-31 -> odd cols
    int col_base = (lane >= 16) ? 1 : 0;
    for (int i = 0; i < 8; i++) {
        int col = i * 2 + col_base;
        out[row * 16 + col] = ((float*)&c)[i];
    }
}

int main()
{
    int ndev = 0;
    HIP_CHECK(hipGetDeviceCount(&ndev));
    printf("HIP devices: %d\n", ndev);
    if (ndev == 0) { printf("No GPU found\n"); return 1; }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s  gcnArchName: %s\n", prop.name, prop.gcnArchName);

    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, 256 * sizeof(float)));
    HIP_CHECK(hipMemset(d_out, 0, 256 * sizeof(float)));

    wmma_test_kernel<<<1, 32>>>(d_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    float h_out[256] = {};
    HIP_CHECK(hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_out));

    // Verify: identity × identity = identity
    int errors = 0;
    for (int r = 0; r < 16; r++) {
        for (int c = 0; c < 16; c++) {
            float expected = (r == c) ? 1.0f : 0.0f;
            float got = h_out[r * 16 + c];
            if (fabsf(got - expected) > 1e-3f) {
                printf("FAIL [%d][%d]: expected %.1f got %.4f\n", r, c, expected, got);
                if (++errors > 10) { printf("...(stopping)\n"); goto done; }
            }
        }
    }
done:
    if (errors == 0)
        printf("PASS: 16x16 WMMA identity multiply correct on %s\n", prop.gcnArchName);
    else
        printf("FAIL: %d errors\n", errors);

    return errors;
}
