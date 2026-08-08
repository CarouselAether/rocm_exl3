// Cooperative-launch test for the RDNA GEMM.
//
//   rocm_tools/build_coop_check.sh && timeout 300 /tmp/gemm_coop_check
//
// gemm_check.hip validates exl3_gemm_kernel_inner with a PLAIN launch. Nothing
// has ever executed the cooperative wrapper on this hardware: cudaLaunchCoop-
// erativeKernel itself, the two grid.sync() barriers, or the EXL3_RDNA_SMEM
// launch sizing. That is the single largest untested assumption in the port,
// and the failure mode is a device hang rather than a wrong number -- hence the
// timeout in the command above, and the progress print before every launch.
//
// Design: differential, not absolute. The reference is the SAME work done
// without cooperative machinery --
//
//   ref  = input hadamard kernel (plain launch)
//        + exl3_gemm_kernel_inner per 16-row chunk (plain launch)
//   test = exl3_gemm_kernel (one cooperative launch, both grid.syncs)
//
// so any mismatch is attributable to the cooperative wrapper and not to
// arithmetic that gemm_check already covers. The reference hadamard mirrors the
// wrapper's own indexing including the `suh + (warp * 128) % size_k` wrap, which
// only matters for size_m > 1 and is easy to get wrong.
//
// Note the wrapper instantiates the inner with shmem_out_had = true, and
// output_had_sh_gl() dereferences post_scale unconditionally, so suh, A_had AND
// svh must all be non-null on this path. There is no "no hadamard" mode to test.

#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "../exllamav3/exllamav3_ext/util.h"
#include "../exllamav3/exllamav3_ext/util.cuh"
#include "../exllamav3/exllamav3_ext/rocm/quant/exl3_gemm_kernel_rdna.hip.h"
#include "../exllamav3/exllamav3_ext/rocm/quant/exl3_kernel_map_rdna.hip.h"
#include "../exllamav3/exllamav3_ext/quant/exl3_devctx.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <unistd.h>

static int g_fail = 0;

// Every hip call here used to be (void)-cast, so an allocation failure surfaced
// as a segfault on the next memcpy with no diagnostic at all. Check them.
#define HC(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) { \
        printf("\n    HIP FAIL %s:%d  %s -> %s\n", __FILE__, __LINE__, #call, hipGetErrorString(_e)); \
        fflush(stdout); g_fail++; } \
} while (0)

// ---------------------------------------------------------------------------
// Reference input hadamard. Mirrors the wrapper's block exactly, including the
// modulo wrap of the scale pointer across rows.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256)
void ref_had_in(const half* __restrict__ A, half* __restrict__ A_had,
                const half* __restrict__ suh, int size_m, int size_k)
{
    int total_warps = size_m * size_k / 128;
    int warps_grid = gridDim.x * blockDim.x / 32;
    int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

    for (; this_warp < total_warps; this_warp += warps_grid)
        had_hf_r_128_inner<true, false>
        (
            A + this_warp * 128,
            A_had + this_warp * 128,
            suh + (this_warp * 128) % size_k,
            0.088388347648f
        );
}

// Plain-launch wrapper around the inner, matching what the cooperative kernel
// calls per 16-row chunk (shmem_out_had = true, svh forwarded).
template <int bits, bool c_fp32, int cb, int TM, int TK, int TN, int SH, int FR>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TK / 16)
void ref_inner(const half* A, const uint16_t* B, void* C,
               int m, int k, int n, int* locks, const half* svh)
{
    exl3_gemm_kernel_inner<bits, c_fp32, cb, TM, TK, TN, SH, FR, true>
        (A, B, C, m, k, n, locks, svh);
}

static void fill_half(std::vector<half>& v, float lo, float hi)
{
    for (size_t i = 0; i < v.size(); i++)
        v[i] = __float2half(lo + (hi - lo) * (float)(rand() % 10001) / 10000.f);
}

// ---------------------------------------------------------------------------
template <int bits, bool c_fp32, int cb, int TM, int TK, int TN, int SH, int FR>
void run_case(int shape_idx, int size_m, int size_k, int size_n, int num_sms, const char* tag)
{
    const int k_blocks = size_k / 16;
    const int blocks_n = size_n / 16;
    const size_t bq_elems = (size_t) k_blocks * blocks_n * (256 * bits / 16);
    const int block_dim = EXL3_GEMM_BASE_THREADS * TK / 16;
    const size_t smem = exl3_gemm_smem_bytes(bits, shape_idx);
    const size_t c_elem = c_fp32 ? 4 : 2;

    std::vector<uint16_t> hBq(bq_elems);
    for (size_t i = 0; i < bq_elems; i++) hBq[i] = (uint16_t)(rand() & 0xffff);
    std::vector<half> hA((size_t) size_m * size_k); fill_half(hA, -0.25f, 0.25f);
    std::vector<half> hSuh((size_t) size_k);        fill_half(hSuh, -1.0f, 1.0f);
    std::vector<half> hSvh((size_t) size_n);        fill_half(hSvh, -1.0f, 1.0f);

    uint16_t* dBq; half *dA, *dAhad, *dAhadRef, *dSuh, *dSvh; void *dC, *dCref; int* dLocks;
    const size_t locks_ints = (size_t) MAX_TILES_C + MAX_BARRIERS * 2 + MOE_SCHED_INTS;
    HC(hipMalloc(&dBq, bq_elems * 2));
    HC(hipMalloc(&dA, hA.size() * 2));
    HC(hipMalloc(&dAhad, hA.size() * 2));
    HC(hipMalloc(&dAhadRef, hA.size() * 2));
    HC(hipMalloc(&dSuh, hSuh.size() * 2));
    HC(hipMalloc(&dSvh, hSvh.size() * 2));
    HC(hipMalloc(&dC, (size_t) size_m * size_n * c_elem));
    HC(hipMalloc(&dCref, (size_t) size_m * size_n * c_elem));
    HC(hipMalloc(&dLocks, locks_ints * 4));
    (void) hipMemcpy(dBq, hBq.data(), bq_elems * 2, hipMemcpyHostToDevice);
    (void) hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice);
    (void) hipMemcpy(dSuh, hSuh.data(), hSuh.size() * 2, hipMemcpyHostToDevice);
    (void) hipMemcpy(dSvh, hSvh.data(), hSvh.size() * 2, hipMemcpyHostToDevice);

    // ---- reference: plain launches ----
    (void) hipMemset(dCref, 0, (size_t) size_m * size_n * c_elem);
    hipLaunchKernelGGL(ref_had_in, dim3(num_sms), dim3(256), 0, 0,
                       dA, dAhadRef, dSuh, size_m, size_k);
    for (int row = 0; row < size_m; row += 16)
    {
        (void) hipMemset(dLocks, 0, locks_ints * 4);
        const half* A_ = dAhadRef + (size_t) row * size_k;
        void* C_ = c_fp32 ? (void*)((float*) dCref + (size_t) row * size_n)
                          : (void*)((half*)  dCref + (size_t) row * size_n);
        hipLaunchKernelGGL((ref_inner<bits, c_fp32, cb, TM, TK, TN, SH, FR>),
                           dim3(num_sms), dim3(block_dim), smem, 0,
                           A_, dBq, C_, MIN(size_m - row, 16), size_k, size_n, dLocks, dSvh);
    }
    hipError_t ref_err = hipDeviceSynchronize();

    // ---- test: one cooperative launch ----
    void* kernel = (void*) exl3_gemm_kernel<bits, c_fp32, cb, TM, TK, TN, SH, FR>;

    // The shipped dispatch does not instantiate the template, it looks the
    // pointer up in the map's per-(K, cb) instance tables. If those tables are
    // mis-wired -- wrong shape slot, wrong codebook, fp16/fp32 swapped -- every
    // number below would still be self-consistent and still be wrong, so check
    // the map agrees with the template before trusting the comparison.
    void* mapped = (void*) get_gemm_kernel_ptr(bits, shape_idx, c_fp32, cb);
    if (mapped != kernel)
    {
        printf("    MAP MISMATCH %-18s get_gemm_kernel_ptr(%d,%d,%d,%d)=%p, template=%p\n",
               tag, bits, shape_idx, (int) c_fp32, cb, mapped, kernel);
        g_fail++;
    }

    (void) hipFuncSetAttribute(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, smem);

    int blocks_per_cu = 0;
    (void) hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block_dim, smem);
    int device = 0; (void) hipGetDevice(&device);
    int mp = 0; (void) hipDeviceGetAttribute(&mp, hipDeviceAttributeMultiprocessorCount, device);
    int coresident = blocks_per_cu * mp;

    (void) hipMemset(dLocks, 0, locks_ints * 4);
    (void) hipMemset(dC, 0, (size_t) size_m * size_n * c_elem);
    (void) hipMemcpy(dAhad, hA.data(), hA.size() * 2, hipMemcpyHostToDevice);  // scratch, overwritten

    const half* A_arg = dA; const uint16_t* B_arg = dBq; void* C_arg = dC;
    int m_arg = size_m, k_arg = size_k, n_arg = size_n;
    int* locks_arg = dLocks; const half* suh_arg = dSuh; half* ahad_arg = dAhad;
    const half* svh_arg = dSvh;
    void* args[] = { &A_arg, &B_arg, &C_arg, &m_arg, &k_arg, &n_arg,
                     &locks_arg, &suh_arg, &ahad_arg, &svh_arg };

    // Printed BEFORE the launch: if the run dies to the timeout, this line is
    // the last thing on stdout and names the case that hung.
    size_t mfree = 0, mtot = 0;
    (void) hipMemGetInfo(&mfree, &mtot);
    printf("    launching %-22s grid=%-3d smem=%-6zu coresident=%-3d free=%zuMB ... ",
           tag, num_sms, smem, coresident, mfree >> 20);
    fflush(stdout);

    hipError_t lerr = hipLaunchCooperativeKernel(kernel, dim3(num_sms), dim3(block_dim),
                                                args, smem, 0);
    hipError_t serr = hipDeviceSynchronize();

    if (lerr != hipSuccess || serr != hipSuccess)
    {
        printf("FAIL  launch=%s sync=%s\n", hipGetErrorString(lerr), hipGetErrorString(serr));
        fflush(stdout);
        g_fail++;
    }
    else
    {
        std::vector<char> got((size_t) size_m * size_n * c_elem);
        std::vector<char> ref((size_t) size_m * size_n * c_elem);
        (void) hipMemcpy(got.data(), dC, got.size(), hipMemcpyDeviceToHost);
        (void) hipMemcpy(ref.data(), dCref, ref.size(), hipMemcpyDeviceToHost);

        double rms_ref = 0, rms_err = 0, worst = 0;
        size_t nelem = (size_t) size_m * size_n;
        for (size_t i = 0; i < nelem; i++)
        {
            double g = c_fp32 ? ((float*) got.data())[i] : __half2float(((half*) got.data())[i]);
            double r = c_fp32 ? ((float*) ref.data())[i] : __half2float(((half*) ref.data())[i]);
            double d = fabs(g - r);
            rms_ref += r * r; rms_err += d * d;
            if (d / (fabs(r) + 1e-3) > worst) worst = d / (fabs(r) + 1e-3);
        }
        rms_ref = sqrt(rms_ref / nelem); rms_err = sqrt(rms_err / nelem);
        bool bad = !(rms_ref > 0) || rms_err / rms_ref > 1e-4 || worst > 0.02;
        printf("%s  vs-plain err/RMS=%.6f worst_rel=%.4f%s\n",
               bad ? "FAIL" : "ok  ", rms_ref > 0 ? rms_err / rms_ref : 0.0, worst,
               ref_err == hipSuccess ? "" : "  (REFERENCE ERRORED)");
        fflush(stdout);
        if (bad || ref_err != hipSuccess) g_fail++;
    }

    HC(hipFree(dBq)); HC(hipFree(dA)); HC(hipFree(dAhad)); HC(hipFree(dAhadRef));
    HC(hipFree(dSuh)); HC(hipFree(dSvh)); HC(hipFree(dC)); HC(hipFree(dCref));
    HC(hipFree(dLocks));
}

int main(int argc, char** argv)
{
    srand(1234);
    bool oversub = (argc > 1 && std::string(argv[1]) == "--oversubscribe");
    bool one = (argc > 3 && std::string(argv[1]) == "--one");   // --one <k> <n> [grid]
    bool smem = (argc > 1 && std::string(argv[1]) == "--smem");

    int device = 0; (void) hipGetDevice(&device);
    int mp = 0; (void) hipDeviceGetAttribute(&mp, hipDeviceAttributeMultiprocessorCount, device);
    int coop = 0; (void) hipDeviceGetAttribute(&coop, hipDeviceAttributeCooperativeLaunch, device);
    printf("device %d: multiProcessorCount=%d (WGPs on RDNA), cooperativeLaunch=%d\n\n", device, mp, coop);

    if (!coop)
    {
        printf("cooperative launch NOT SUPPORTED -- exl3_gemm cannot run at all on this device\n");
        return 1;
    }

    if (smem)
    {
        // Does the runtime LDS budget actually gate shape selection? Build this
        // with -DEXL3_RDNA_SMEM_MAX=92160 on a 64 KB part: the compile-time
        // budget then over-promises, and every shape that fits 90 KB but not
        // 64 KB must be refused HERE rather than at launch.
        printf("compile-time SMEM_MAX = %d\n", (int) SMEM_MAX);
        printf("device sharedMemPerBlock = %zu\n", exl3_rdna_device_smem_max(-1));
        printf("effective budget = %zu\n\n", exl3_rdna_smem_budget(-1));
        for (int bits : {2, 4, 8})
            for (int sh = 1; sh <= EXL3_GEMM_NUM_SHAPES; sh++)
            {
                size_t need = exl3_gemm_smem_bytes(bits, sh);
                size_t budget = exl3_rdna_smem_budget(-1);
                static const int tn_tab[] = {EXL3_GEMM_TILESIZE_N};
                int tn = tn_tab[sh];
                // Report the two admission tests separately. Conflating them is
                // easy and misleading: an N that does not divide size_n is
                // refused for shape reasons, which looks identical to an LDS
                // refusal in a single pass/fail column.
                bool fits = need <= budget;
                bool divides = tn > 0 && (6144 % tn) == 0;
                printf("  bits=%d shape=%d N=%-4d needs %-6zu  lds:%-8s  div:%-8s\n",
                       bits, sh, tn, need,
                       fits ? "fits" : "TOO BIG", divides ? "ok" : "n/a");
            }
        fflush(stdout);
        _exit(0);
    }

    if (one)
    {
        int k = atoi(argv[2]), n = atoi(argv[3]);
        int g = argc > 4 ? atoi(argv[4]) : mp;
        printf("single case: s2 bits4 cb0 fp32 m16 k=%d n=%d grid=%d\n", k, n, g);
        run_case<4, true, 0, 16, 16, 256, 3, 3>(2, 16, k, n, g, "one");
        printf("\n%s\n", g_fail ? "FAILED" : "passed");
        fflush(stdout);
        _exit(g_fail ? 1 : 0);
    }

    if (!oversub)
    {
        printf("shape 1 (N=128), bits 4:\n");
        run_case<4, true,  0, 16, 16, 128, 4, 3>(1, 16, 512,  256,  mp, "s1 m16 fp32");
        run_case<4, true,  0, 16, 16, 128, 4, 3>(1, 32, 1024, 512,  mp, "s1 m32 fp32");
        run_case<4, false, 0, 16, 16, 128, 4, 3>(1, 16, 1024, 512,  mp, "s1 m16 fp16");

        printf("shape 2 (N=256), bits 4 / 2 / 8, cb 0 / 1 / 2:\n");
        run_case<4, true,  0, 16, 16, 256, 3, 3>(2, 16, 1024, 512,  mp, "s2 m16 b4 cb0");
        run_case<2, true,  0, 16, 16, 256, 3, 3>(2, 16, 1024, 512,  mp, "s2 m16 b2 cb0");
        run_case<8, true,  0, 16, 16, 256, 3, 3>(2, 16, 1024, 512,  mp, "s2 m16 b8 cb0");
        run_case<4, true,  1, 16, 16, 256, 3, 3>(2, 16, 1024, 512,  mp, "s2 m16 b4 cb1");
        run_case<4, true,  2, 16, 16, 256, 3, 3>(2, 16, 1024, 512,  mp, "s2 m16 b4 cb2");

        printf("shape 3 (N=384), bits 4:\n");
        run_case<4, true,  0, 16, 16, 384, 3, 3>(3, 16, 1024, 768,  mp, "s3 m16 fp32");
        run_case<4, true,  0, 16, 16, 384, 3, 3>(3, 48, 1024, 768,  mp, "s3 m48 fp32");

        printf("grid sizes below full width (partial co-residency):\n");
        run_case<4, true,  0, 16, 16, 256, 3, 3>(2, 16, 1024, 512, 1,      "s2 grid=1");
        run_case<4, true,  0, 16, 16, 256, 3, 3>(2, 16, 1024, 512, mp / 2, "s2 grid=half");
        run_case<4, true,  0, 16, 16, 256, 3, 3>(2, 16, 4096, 2048, mp,    "s2 big k/n");

        printf("\n%s\n", g_fail ? "FAILED" : "all cooperative cases passed");
    }
    else
    {
        // Deliberate over-subscription: a cooperative grid larger than what fits
        // co-resident. The question is whether the runtime REFUSES it (good) or
        // launches and deadlocks in grid.sync() (bad). Separate process and
        // separate flag because the bad outcome wedges the GPU.
        printf("OVERSUBSCRIPTION PROBE -- may hang; run under timeout\n");
        run_case<4, true, 0, 16, 16, 256, 3, 3>(2, 16, 1024, 512, mp * 4, "s2 grid=4x");
        printf("\nsurvived oversubscription\n");
    }

    // _exit, not return: the HSA runtime faults in its own __cxa_finalize path
    // on this stack (ROCm 7.2.4 + torch), which turned a fully passing run into
    // a SIGSEGV with the final results still sitting in the stdout buffer. The
    // fault is at library teardown, after every kernel has completed and been
    // verified -- it is not a GEMM result. Flush and leave.
    fflush(stdout);
    _exit(g_fail ? 1 : 0);
}
