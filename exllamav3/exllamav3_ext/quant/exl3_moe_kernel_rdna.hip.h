#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "exl3_kernel_map_rdna.hip.h"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner_rdna.hip.h"
#include "exl3_devctx.cuh"

// Mirror upstream constants
#define MOE_ACT_SILU 0
#define MOE_ACT_GELU 1

#define MOE_SMS_PER_EXPERT 12
#define MOE_TILESIZE_K 32
#define MOE_TILESIZE_M 16
#define MOE_SH_STAGES 4
#define MOE_FRAG_STAGES 3

// Sense-flip inter-block barrier for a group of MOE_SMS_PER_EXPERT blocks.
// barrier_counters_sense layout: [group_id*2] = arrival counter, [group_id*2+1] = sense bit.
// All threads sync before and after so the caller doesn't need an additional __syncthreads.
//
// AMD note: atomicAdd/atomicExch on global memory are sequentially consistent on GFX11.
// __threadfence() before the sense flip ensures non-atomic writes from this block are
// globally visible before the waiting blocks exit their spin loop.
__device__ __forceinline__ void group_barrier_rdna
(
    int group_id,
    int group_size,
    int* barrier_counters_sense
)
{
    __syncthreads();

    if (threadIdx.x == 0)
    {
        int* counter   = &barrier_counters_sense[group_id * 2];
        int* sense_ptr = &barrier_counters_sense[group_id * 2 + 1];

        int old_sense = atomicAdd(sense_ptr, 0);        // relaxed load
        int old       = atomicAdd(counter, 1);          // acq_rel increment

        if (old == group_size - 1)
        {
            atomicExch(counter, 0);
            __threadfence();
            atomicExch(sense_ptr, 1 - old_sense);       // release: flip sense
        }
        else
        {
            while (atomicAdd(sense_ptr, 0) == old_sense) // acquire spin
            {
                // yield hint — reduces bus pressure on AMD
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }

    __syncthreads();
}

#define EXL3_MOE_KERNEL_ARGS                    \
    const half* __restrict__ hidden_state,      \
    half* __restrict__ temp_state_g,            \
    half* __restrict__ temp_state_u,            \
    half* __restrict__ temp_intermediate_g,     \
    half* __restrict__ temp_intermediate_u,     \
    float* __restrict__ output_state,           \
                                                \
    const uint16_t** __restrict__ gate_trellis, \
    const half** __restrict__ gate_suh,         \
    const half** __restrict__ gate_svh,         \
    const uint16_t** __restrict__ up_trellis,   \
    const half** __restrict__ up_suh,           \
    const half** __restrict__ up_svh,           \
    const uint16_t** __restrict__ down_trellis, \
    const half** __restrict__ down_suh,         \
    const half** __restrict__ down_svh,         \
                                                \
    const int64_t* __restrict__ expert_count,   \
    const int64_t* __restrict__ token_sorted,   \
    const half* __restrict__ weight_sorted,     \
                                                \
    const int hidden_dim,                       \
    const int intermediate_dim,                 \
    const int num_experts,                      \
    const int num_experts_per_tok,              \
    const int max_tokens_per_expert,            \
    const int concurrency,                      \
    const float act_limit,                      \
    const int act_function,                     \
    const int K_gate,                           \
    const int K_up,                             \
    const int K_down,                           \
                                                \
    int* __restrict__ locks

// Fused MoE MLP kernel for EXL3 quantized weights.
// Grid: dim3(MOE_SMS_PER_EXPERT, 1, concurrency)
// blockIdx.x = SM index within expert group
// blockIdx.z = expert group (concurrency slot)
// blockIdx.y = always 0 (hadamard_inner.cuh scale loads rely on this)
//
// Pipeline per expert:
//   1. Gather + input hadamard -> temp_state_g / temp_state_u
//   2. GEMM gate + GEMM up    -> temp_intermediate_g / temp_intermediate_u
//   3. Output hadamard + act*gate + input hadamard for down
//   4. GEMM down              -> temp_state_g
//   5. Output hadamard + scatter-add with routing weight -> output_state

template<int t_bits, int MOE_TILESIZE_N>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16)
void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)
{
    const int group_idx     = blockIdx.z;
    const int block_idx     = blockIdx.x;
    const int block_threads = blockDim.x;
    const int group_threads = MOE_SMS_PER_EXPERT * block_threads;
    const int warp_id       = threadIdx.x / 32;
    const int warps_per_group = group_threads / 32;
    const int warps_per_block = block_threads / 32;
    const int warp_idx0     = block_idx * warps_per_block + warp_id;

    // Per-group temporary buffers
    temp_state_g         += group_idx * max_tokens_per_expert * hidden_dim;
    temp_state_u         += group_idx * max_tokens_per_expert * hidden_dim;
    temp_intermediate_g  += group_idx * max_tokens_per_expert * intermediate_dim;
    temp_intermediate_u  += group_idx * max_tokens_per_expert * intermediate_dim;

    // Sense-flip counters for group barriers (stored past the per-tile GEMM locks)
    int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;

    // Per-tile GEMM locks for this group (each expert group gets its own slice)
    locks += group_idx * MAX(hidden_dim, intermediate_dim) / 128;

    int start = 0, end = 0;
    int expert_idx = 0, expert_idx_assign = 0;
    for (; expert_idx < num_experts; ++expert_idx)
    {
        start = end;
        end += expert_count[expert_idx];
        int token_count = end - start;

        if (token_count == 0) continue;
        if (token_count > max_tokens_per_expert) continue;
        if (expert_idx_assign++ % concurrency != group_idx) continue;

        const uint16_t* exp_gate_trellis = gate_trellis[expert_idx];
        const half*     exp_gate_suh     = gate_suh[expert_idx];
        const half*     exp_gate_svh     = gate_svh[expert_idx];
        const uint16_t* exp_up_trellis   = up_trellis[expert_idx];
        const half*     exp_up_suh       = up_suh[expert_idx];
        const half*     exp_up_svh       = up_svh[expert_idx];
        const uint16_t* exp_down_trellis = down_trellis[expert_idx];
        const half*     exp_down_suh     = down_suh[expert_idx];
        const half*     exp_down_svh     = down_svh[expert_idx];

        // --- Stage 1: gather + input hadamard for gate and up ---
        auto had_gather_gu_in = [&]()
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token];
                int token_off = warp_idx % warps_per_token;
                const half* in_ptr = hidden_state + token_idx * hidden_dim + token_off * 128;
                had_hf_r_128_inner(in_ptr, temp_state_g + 128 * warp_idx,
                    exp_gate_suh + 128 * token_off, nullptr, 0.088388347648f);
                had_hf_r_128_inner(in_ptr, temp_state_u + 128 * warp_idx,
                    exp_up_suh   + 128 * token_off, nullptr, 0.088388347648f);
            }
            group_barrier_rdna(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        };

        had_gather_gu_in();

        // --- Stage 2: GEMM gate and up ---
        auto gemm_up = [&](const half* in_addr, half* out_addr,
                           const uint16_t* trellis, const int K)
        {
            int size_m = token_count;
            while (size_m > 0)
            {
                #define ARGS            \
                    in_addr,            \
                    trellis,            \
                    out_addr,           \
                    size_m,             \
                    hidden_dim,         \
                    intermediate_dim,   \
                    locks
                #define SHAPE_ARGS      \
                    MOE_TILESIZE_M,     \
                    MOE_TILESIZE_K,     \
                    MOE_TILESIZE_N,     \
                    MOE_SH_STAGES,      \
                    MOE_FRAG_STAGES
                if constexpr (t_bits)
                    exl3_gemm_kernel_inner<t_bits, false, 1, SHAPE_ARGS>(ARGS);
                else switch(K)
                {
                    case 1: exl3_gemm_kernel_inner<1, false, 1, SHAPE_ARGS>(ARGS); break;
                    case 2: exl3_gemm_kernel_inner<2, false, 1, SHAPE_ARGS>(ARGS); break;
                    case 3: exl3_gemm_kernel_inner<3, false, 1, SHAPE_ARGS>(ARGS); break;
                    case 4: exl3_gemm_kernel_inner<4, false, 1, SHAPE_ARGS>(ARGS); break;
                    case 5: exl3_gemm_kernel_inner<5, false, 1, SHAPE_ARGS>(ARGS); break;
                    case 6: exl3_gemm_kernel_inner<6, false, 1, SHAPE_ARGS>(ARGS); break;
                    case 7: exl3_gemm_kernel_inner<7, false, 1, SHAPE_ARGS>(ARGS); break;
                    case 8: exl3_gemm_kernel_inner<8, false, 1, SHAPE_ARGS>(ARGS); break;
                }
                #undef ARGS
                #undef SHAPE_ARGS

                in_addr  += 16 * hidden_dim;
                out_addr += 16 * intermediate_dim;
                size_m   -= 16;
                group_barrier_rdna(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
            }
        };

        gemm_up(temp_state_g, temp_intermediate_g, exp_gate_trellis, K_gate);
        gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);

        // --- Stage 3: output had for g/u + activation*gate + input had for d ---
        auto had_guad = [&]()
        {
            const int warps_per_token = intermediate_dim / 128;
            const int total_warps = token_count * warps_per_token;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_off = warp_idx % warps_per_token;
                had_hf_r_128_guad_inner(
                    temp_intermediate_g + 128 * warp_idx,
                    temp_intermediate_u + 128 * warp_idx,
                    temp_intermediate_g + 128 * warp_idx,
                    exp_gate_svh  + 128 * token_off,
                    exp_up_svh    + 128 * token_off,
                    exp_down_suh  + 128 * token_off,
                    0.088388347648f,
                    act_limit,
                    act_function
                );
            }
            group_barrier_rdna(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        };

        had_guad();

        // --- Stage 4: GEMM down ---
        auto gemm_down = [&](const half* in_addr, half* out_addr,
                             const uint16_t* trellis, const int K)
        {
            int size_m = token_count;
            while (size_m > 0)
            {
                #define ARGS            \
                    in_addr,            \
                    trellis,            \
                    out_addr,           \
                    size_m,             \
                    intermediate_dim,   \
                    hidden_dim,         \
                    locks
                #define SHAPE_ARGS      \
                    MOE_TILESIZE_M,     \
                    MOE_TILESIZE_K,     \
                    MOE_TILESIZE_N,     \
                    MOE_SH_STAGES,      \
                    MOE_FRAG_STAGES
                if constexpr (t_bits)
                    exl3_gemm_kernel_inner<t_bits, false, 1, SHAPE_ARGS>(ARGS);
                else switch(K)
                {
                    case 1: exl3_gemm_kernel_inner<1, false, 1, SHAPE_ARGS>(ARGS); break;
                    case 2: exl3_gemm_kernel_inner<2, false, 1, SHAPE_ARGS>(ARGS); break;
                    case 3: exl3_gemm_kernel_inner<3, false, 1, SHAPE_ARGS>(ARGS); break;
                    case 4: exl3_gemm_kernel_inner<4, false, 1, SHAPE_ARGS>(ARGS); break;
                    case 5: exl3_gemm_kernel_inner<5, false, 1, SHAPE_ARGS>(ARGS); break;
                    case 6: exl3_gemm_kernel_inner<6, false, 1, SHAPE_ARGS>(ARGS); break;
                    case 7: exl3_gemm_kernel_inner<7, false, 1, SHAPE_ARGS>(ARGS); break;
                    case 8: exl3_gemm_kernel_inner<8, false, 1, SHAPE_ARGS>(ARGS); break;
                }
                #undef ARGS
                #undef SHAPE_ARGS

                in_addr  += 16 * intermediate_dim;
                out_addr += 16 * hidden_dim;
                size_m   -= 16;
                group_barrier_rdna(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
            }
        };

        gemm_down(temp_intermediate_g, temp_state_g, exp_down_trellis, K_down);

        // --- Stage 5: output had for d + scatter-add with routing weight ---
        auto had_d_out = [&]()
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int64_t* top_x  = token_sorted + start;
            const half*   weights = weight_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token];
                half weight   = weights[warp_idx / warps_per_token];
                int token_off = warp_idx % warps_per_token;
                float* out_ptr = output_state + token_idx * hidden_dim + token_off * 128;
                had_hf_r_128_d_inner(
                    temp_state_g + 128 * warp_idx,
                    out_ptr,
                    exp_down_svh + 128 * token_off,
                    0.088388347648f * __half2float(weight)
                );
            }
            group_barrier_rdna(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        };

        had_d_out();
    }
}
