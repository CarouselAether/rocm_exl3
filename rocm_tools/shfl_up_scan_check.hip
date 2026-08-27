// Probe: the guarded __shfl_up_sync inclusive-scan idiom that v1.4.4's
// dsa_topk.cu introduced (six sites, all identical in shape):
//
//     for (int o = 1; o < 32; o <<= 1) {
//         int n = __shfl_up_sync(0xffffffffu, incl, o);
//         if (lane >= o) incl += n;
//     }
//
// The shim maps __shfl_up_sync(mask, v, d) -> __shfl_up(v, d). The `lane >= o`
// guard makes the received value irrelevant for out-of-range lanes, so the only
// way this breaks on RDNA is wrong in-range delivery. Per the porting rule,
// the probe uses DISTINCT per-lane values: a uniform input cannot distinguish
// shuffle clamp from wrap (v += v doubles either way).
//
//   hipcc -o /tmp/shfl_up_scan_check rocm_tools/shfl_up_scan_check.hip \
//         -std=c++17 --offload-arch=gfx1151 -O3 -DUSE_ROCM \
//         -include exllamav3/exllamav3_ext/rocm/hip_compat.hip.h \
//         && /tmp/shfl_up_scan_check
//
// Expected: lane i holds sum of inputs 0..i. With inputs v[i] = 3*i + 7 the
// prefix sums are distinct and asymmetric, so any clamp/wrap/offset defect
// shows as a mismatch at a specific lane rather than a plausible-looking total.

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void scan_kernel(const int* __restrict__ in, int* __restrict__ out)
{
    int lane = threadIdx.x % 32;
    int incl = in[threadIdx.x];
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        int n = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += n;
    }
    out[threadIdx.x] = incl;
}

int main()
{
    const int N = 64;  // two warps: also catches cross-warp bleed at the 32 boundary
    int h_in[N], h_ref[N];
    for (int i = 0; i < N; ++i) h_in[i] = 3 * i + 7;
    for (int w = 0; w < N / 32; ++w)
    {
        int acc = 0;
        for (int l = 0; l < 32; ++l) { acc += h_in[w * 32 + l]; h_ref[w * 32 + l] = acc; }
    }

    int *d_in, *d_out;
    hipMalloc(&d_in, N * sizeof(int));
    hipMalloc(&d_out, N * sizeof(int));
    hipMemcpy(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice);
    scan_kernel<<<1, N>>>(d_in, d_out);
    int h_out[N];
    hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost);

    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (h_out[i] != h_ref[i])
        {
            printf("MISMATCH lane %d (warp %d): got %d want %d\n", i % 32, i / 32, h_out[i], h_ref[i]);
            bad++;
        }
    printf(bad ? "FAIL: %d mismatches\n" : "PASS: guarded __shfl_up_sync scan correct on both warps\n", bad);
    return bad ? 1 : 0;
}
