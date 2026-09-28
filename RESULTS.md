# RESULTS.md: per-change log (PLAN.md §5)

Baseline: `perf/phase0` @ 4bf6bc2 (97063b3 + Phase 0 tooling), torch 2.15.0.dev20260926+rocm10.0, CPU boost off. The measured baseline is pp512 112.8, pp2048 178.5, tg128 18.02, MTP 19.78 t/s; PPL: DS4 6.461586, Qwen 4.747702, Gemma 18.635452.

**Branches:**
- One optimization per `opt/<name>` branch.
- `perf/stack` = `perf/phase0` + finished `opt/*` branches. It is an **unreviewed integration branch**, so the maintainer can run the stack.
- Nothing merges into `main` / `rocm-10` before the maintainer's coherence sign-off (gates 6–7).
- Every change keeps an env switch, so it can be measured on/off within a single build.

Status is one of: `reverted`, `ready for human check`, `approved`, `merged`.

| Branch | Change | tg128 before → after | pp512 before → after | pp2048 before → after | PPL Δ | Numerics notes | Status |
|---|---|---|---|---|---|---|---|
| `opt/wmma-gemm` (5386cd0) | hgemm / hgemm_recon routed to vendored rocm_wmma_gemm (MIT) WMMA kernels with per-arch tuned tables; fp32 output on the matrix cores (hipBLAS had only a VALU kernel on gfx1151). `EXL3_ROCM_WMMA_GEMM=0/1/2`. | 17.96 → 17.98 | 112.59 → **123.06** (+9%) | 174.18 → **216.45** (+24%) | DS4 −0.007%, Gemma −0.009% | Different accumulation order from hipBLAS. fp32 out: max err 3–17× hipBLAS's, but ≤2% of the fp32 error bound (RDNA3 WMMA round-toward-zero per 16-deep step). fp16 out: RNE of the fp32 WMMA result; bitwise equal to hipBLAS on 5/7 shapes. **Kernels changed:** hgemm path only (prefill reconstructed linears, all dense and expert-down GEMMs with m > 8). The decode GEMV path is untouched. | ready for human check |
| `opt/dsa-decode` | DeepSeek-V4 decode sparse-attention split kernel replaced by an MQA-specialized Triton kernel, `rocm_py/dsa_decode_rdna.py`. One program = 32 heads × a 256-column output block × a key split. The score reduction is streamed over D in a runtime loop, so q is never a resident WMMA operand. Same workspace, same combine and same C++ launch; wired via rocm_py (compile wrapper + eager launch proxy), no upstream edits. Tuning: BLOCK_H 16, 8 splits. Split + combine per call 169 → 15.5 µs at ctx 512, 218 → 33 µs at 16K top-k, MTP 3-row 451 → 28 µs; 0 spills (fp16 pool). `EXL3_ROCM_DSA_DECODE=0/1`. | 18.07 → **21.52** (+19%); tg64@16K 17.39 → **20.46**; MTP ndt2 20.73 → **23.82** | 122.9 → 121.2 (untouched path; noise) | not run (prefill untouched) | DS4 0.000% (6.461140) | Different reduction order in decode attention (8 splits instead of 16; D chunked by 64). Error vs fp64 unchanged (4.7e-4 vs 5.0e-4). Greedy 256-token A/B vs off: divergences only at near-ties (top-2 gap ≤ 0.125); top-k-regime prompt 256/256 identical in plain, batched and `-cq 8` runs. The same class of difference as upstream with only N_SPLITS 16 → 8 (`rocm_tools/decode_agree.py`). `test_dsa_kernels` ALL PASS. **Kernels changed:** DSA decode split attention in every DS4 layer (graphed BCDsa + batched BCDsaBatch + eager `dsa_attn` split path); combine, prefill and GLM-5.2 DSA-on-MLA unchanged. | ready for human check |
