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
| *(none yet)* | | | |
