#pragma once

// Getters for the pipelined-mainloop MoE kernels (exl3_moe_kernel<..., PIPE = true>).
// Defined next to the upstream-named getters in comp_units_rdna/exl3_moe_inst_*.hip;
// include after quant/comp_units/exl3_moe_instances.cuh (for fp_exl3_moe_kernel).

#define EXL3_MOE_DECLARE_PIPE_GETTERS(K) \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb1_pipe(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb1_pipe(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_pipe(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb2_pipe(); \

EXL3_MOE_DECLARE_PIPE_GETTERS(0);
EXL3_MOE_DECLARE_PIPE_GETTERS(1);
EXL3_MOE_DECLARE_PIPE_GETTERS(2);
EXL3_MOE_DECLARE_PIPE_GETTERS(3);
EXL3_MOE_DECLARE_PIPE_GETTERS(4);
EXL3_MOE_DECLARE_PIPE_GETTERS(5);
EXL3_MOE_DECLARE_PIPE_GETTERS(6);
EXL3_MOE_DECLARE_PIPE_GETTERS(7);
EXL3_MOE_DECLARE_PIPE_GETTERS(8);

#undef EXL3_MOE_DECLARE_PIPE_GETTERS
