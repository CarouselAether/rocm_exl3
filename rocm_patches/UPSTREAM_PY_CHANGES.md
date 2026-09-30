# ROCm edits to upstream Python files

Kernel changes follow the sibling rule: new files go under `exllamav3/exllamav3_ext/rocm/`, and upstream `.cu`/`.cpp` files are never edited. This log covers edits to **upstream-owned Python** under `exllamav3/`, excluding `rocm_py/` and `exllamav3_ext/`. The maintainer allows these edits for ROCm, provided each one is recorded here so it can be re-applied after a rebase:

```
exlproject/rocm_py_patches.sh status   # what differs from rocm_patches/BASE
exlproject/rocm_py_patches.sh export   # -> rocm_patches/upstream_py.patch
exlproject/rocm_py_patches.sh apply    # after rebasing: git apply --3way
```

`BASE` is the commit the edits sit on top of: 1d637b5, the merge of upstream v1.5.3 (was 97063b3, rocm-10 at the start of the perf work). At that merge the one pre-existing in-file edit, `exllamav3/modules/attention_fn/triton_paged.py` (an `_is_rocm` narrow-kv prefill tile and a decode-split comment), was dropped in favour of upstream's file verbatim; the tile moved to `rocm_py` (`EXL3_ROCM_PREFILL_HD128`) and the comment to RDNA_NOTES. The only remaining upstream-file edit is the two-line `rocm_py` hook at the end of `exllamav3/__init__.py`, which predates this log.

For each file, record the reason, what changed, and how to verify it.

| File | Change | Why | Verify |
|---|---|---|---|
| `exllamav3/generator/generator.py` (`iterate_draftmodel_dflash_gen`), `exllamav3/architecture/dflash_laguna.py` + `dflash.py` (`sample_from_state`) | Without a draft calibrator, the generator sets `params["draft_rows"] = window + 1`; the two per-row-argmax DFlash drafters crop the state to those rows before the target's lm_head. DFlash2 (selector walk over all rows) and the MTP drafters ignore the key. | The DFlash block is always 16 rows, but only the anchor + ndt rows are consumed: the draft lm_head (Laguna: 100352 x 3072 at 6 bpw) ran at m = 16 for 4 useful rows. Laguna ndt 3: 3.75 -> 1.18 ms per round (opt/dflash-drafter, 2026-09-30). Platform-neutral; a candidate for upstream. | Not bit-identical to the m = 16 head: m = 4 takes the multi-row GEMV instead of the GEMM, and 1 of 104 rounds flipped a near-tie draft token. Output is unaffected, since the target verifies every draft. Re-check: the draft-id comparison in the opt/dflash-drafter census probe, plus DFlash acceptance within noise. |
| `exllamav3/architecture/dflash_laguna.py` (`prepare_inputs`, `_short_block_pays`), `exllamav3/modules/arch_specific/dflash.py` (`DFlashInputLayer.forward`) | A causal Laguna DFlash drafter with EXL3 weights sets `params["draft_block_rows"] = min(draft_rows, block_size)`; the input layer builds that many rows (anchor + masks) instead of `native_draft_len`. fp16 drafters and non-causal DFlash keep the full block. | Causal: row j attends only to rows <= j, so rows past anchor + ndt never change the draft. At m = 16 an EXL3 drafter falls off the m <= 8 GEMV paths onto the GEMM (down_proj 4 bpw: 420 us, ~45 GB/s); at m = 4 it stays on the multi-row GEMV. Laguna-S EXL3 4 bpw drafter, ndt 3: 13.3 -> 3.6 ms per draft step. An fp16 drafter gets slower (11.3 -> 13.7 ms, hipBLAS small-m), hence the gate. | Greedy draft-id comparison, truncated vs full block (opt/dflash-drafter census probe, 256 tokens): fp16 94/94 rounds identical (gate bypassed); EXL3 4 bpw 94/95 (one m = 4 GEMV vs m = 16 GEMM rounding flip). DFlash acceptance within noise. |
