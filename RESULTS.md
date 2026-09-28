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
