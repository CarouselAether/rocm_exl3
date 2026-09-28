# ROCm edits to upstream Python files

Kernel changes follow the sibling rule: new files go under `exllamav3/exllamav3_ext/rocm/`, and upstream `.cu`/`.cpp` files are never edited. This log covers edits to **upstream-owned Python** under `exllamav3/`, excluding `rocm_py/` and `exllamav3_ext/`. The maintainer allows these edits for ROCm, provided each one is recorded here so it can be re-applied after a rebase:

```
exlproject/rocm_py_patches.sh status   # what differs from rocm_patches/BASE
exlproject/rocm_py_patches.sh export   # -> rocm_patches/upstream_py.patch
exlproject/rocm_py_patches.sh apply    # after rebasing: git apply --3way
```

`BASE` is the commit the edits sit on top of (97063b3: rocm-10 at the start of the perf work).

For each file, record the reason, what changed, and how to verify it.

| File | Change | Why | Verify |
|---|---|---|---|
| *(none yet)* | | | |
