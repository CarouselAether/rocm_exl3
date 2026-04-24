// WMMA smoke test 2 — non-symmetric inputs to distinguish:
//   - correct (D = A*B, correct store layout)
//   - swapped args (D = B*A, correct store layout)
//   - transposed store (D = A*B, row/col swapped on store)
//   - both errors (compensating or not)
//
// The first smoke test uses identity × identity = identity, which is
// symmetric in every way that masks these bugs. This test uses:
//   A[i][j] = i + 1          (each row constant, distinct row values)
//   B[i][j] = 2*j + 1        (each column constant, distinct col values)
//
// Correct D = A*B:
//   D[i][j] = sum_k A[i][k]*B[k][j] = sum_k (i+1)*(2j+1) = 16*(i+1)*(2j+1)
//   Example: D[0][0]=16, D[1][0]=32, D[0][1]=48, D[2][1]=144
//
// Swapped D' = B*A (args swapped in builtin):
//   D'[i][j] = sum_k B[i][k]*A[k][j] = sum_k (2k+1)*(k+1) = 2856
//   Constant across all cells — easy to spot.
//
// Transposed store of correct D:
//   D_T[i][j] = D[j][i] = 16*(j+1)*(2i+1)
//   Example: D_T[0][0]=16, D_T[1][0]=48 (vs correct 32), D_T[0][1]=32 (vs 48)
//
// Both errors: 2856 in all cells, same as swap alone.
//
// Build:
//   hipcc -o /tmp/wmma_smoke2 tests/wmma_smoke2.cpp -std=c++17

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

// This kernel mirrors the EXACT load/compute/store pattern used in
// rdna_wmma.hip so the test reproduces what exl3_gemm would do.
// If we find a mismatch here, the same mismatch exists in prod kernels.
__global__ void wmma_asym_kernel(float* out)
{
    int lane = threadIdx.x & 31;

    // === Load A: row-major, lane L holds row (L%16) ===
    // A[i][j] = i + 1 → lane L loads 16 copies of (L%16)+1 (every col same)
    half16_t a;
    {
        int row = lane % 16;
        _Float16 row_val = (_Float16)(float)(row + 1);
        #pragma unroll
        for (int i = 0; i < 16; i++)
            ((_Float16*)&a)[i] = row_val;
    }

    // === Load B: row-major, lane L holds column (L%16) ===
    // B[i][j] = 2*j + 1 → lane L loads 16 copies of 2*(L%16)+1 (every row same)
    half16_t b;
    {
        int col = lane % 16;
        _Float16 col_val = (_Float16)(float)(2 * col + 1);
        #pragma unroll
        for (int k = 0; k < 16; k++)
            ((_Float16*)&b)[k] = col_val;
    }

    float8_t c = {0,0,0,0,0,0,0,0};

    // === WMMA: using the SAME argument order as rdna_wmma.hip ===
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b, a, c);

    // === Store: using the SAME layout as rdna_wmma.hip wmma_store_c_row_major ===
    //   row = lane % 16
    //   col_base = (lane >= 16) ? 1 : 0
    //   for i in 0..7: out[row * 16 + (i*2 + col_base)] = frag[i]
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int col = i * 2 + col_base;
        out[row * 16 + col] = ((float*)&c)[i];
    }
}

int main()
{
    int ndev = 0;
    HIP_CHECK(hipGetDeviceCount(&ndev));
    if (ndev == 0) { printf("No GPU found\n"); return 1; }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s  gcnArchName: %s\n", prop.name, prop.gcnArchName);

    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, 256 * sizeof(float)));
    HIP_CHECK(hipMemset(d_out, 0, 256 * sizeof(float)));

    wmma_asym_kernel<<<1, 32>>>(d_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    float h_out[256] = {};
    HIP_CHECK(hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_out));

    // === Analyze output against four candidate patterns ===
    int match_correct  = 0;  // D[i][j] = 16*(i+1)*(2j+1)
    int match_trans    = 0;  // D[i][j] = 16*(j+1)*(2i+1) — store transposed
    int match_swap     = 0;  // D[i][j] = 2856                — args swapped
    int unclassified   = 0;
    const float tol = 1.0f;  // fp16 inputs, fp32 accumulate — exact for these ints

    for (int r = 0; r < 16; r++) {
        for (int c = 0; c < 16; c++) {
            float got       = h_out[r * 16 + c];
            float v_correct = 16.0f * (r + 1) * (2 * c + 1);
            float v_trans   = 16.0f * (c + 1) * (2 * r + 1);
            float v_swap    = 2856.0f;

            if      (fabsf(got - v_correct) <= tol) match_correct++;
            else if (fabsf(got - v_trans)   <= tol) match_trans++;
            else if (fabsf(got - v_swap)    <= tol) match_swap++;
            else                                    unclassified++;
        }
    }

    printf("\nResults over 256 cells:\n");
    printf("  matches correct   D[i][j] = 16*(i+1)*(2j+1):   %d\n", match_correct);
    printf("  matches transpose D[i][j] = 16*(j+1)*(2i+1):   %d\n", match_trans);
    printf("  matches swap      D[i][j] = 2856 (constant):   %d\n", match_swap);
    printf("  unclassified:                                  %d\n", unclassified);

    printf("\nSample corner values:\n");
    printf("  out[0][0]=%.1f  (correct=16,   trans=16,   swap=2856)\n",  h_out[0*16+0]);
    printf("  out[1][0]=%.1f  (correct=32,   trans=48,   swap=2856)\n",  h_out[1*16+0]);
    printf("  out[0][1]=%.1f  (correct=48,   trans=32,   swap=2856)\n",  h_out[0*16+1]);
    printf("  out[2][1]=%.1f  (correct=144,  trans=80,   swap=2856)\n",  h_out[2*16+1]);
    printf("  out[1][2]=%.1f  (correct=160,  trans=144,  swap=2856)\n",  h_out[1*16+2]);
    printf("  out[15][15]=%.1f (correct=7936, trans=7936, swap=2856)\n", h_out[15*16+15]);

    printf("\nDiagnosis:\n");
    if (match_correct == 256) {
        printf("  PASS — WMMA arg order and store layout both correct.\n");
        return 0;
    }
    if (match_swap >= 200) {
        printf("  ** SWAPPED WMMA ARGS ** — builtin expects (A, B, C),\n");
        printf("     our code calls with (b.data, a.data, c.data).\n");
        printf("     Fix: flip to (a.data, b.data, c.data) in rdna_wmma.hip.\n");
        return 2;
    }
    if (match_trans >= 200) {
        printf("  ** TRANSPOSED STORE ** — wmma_store_c_row_major treats lane\n");
        printf("     as row index, but ISA wave32 layout has lane as column\n");
        printf("     index. Fix: swap `row = lane%%16` and `col = 2*i + base`\n");
        printf("     in rdna_wmma.hip:194-204.\n");
        return 3;
    }
    if (match_correct >= 200) {
        printf("  MOSTLY CORRECT — may be edge-cell error; inspect samples.\n");
        return 4;
    }
    printf("  UNKNOWN PATTERN — output matches none of the candidates.\n");
    printf("  Something else is wrong. Dump full matrix:\n");
    for (int r = 0; r < 16; r++) {
        printf("  row %2d:", r);
        for (int c = 0; c < 16; c++)
            printf(" %7.0f", h_out[r * 16 + c]);
        printf("\n");
    }
    return 5;
}
