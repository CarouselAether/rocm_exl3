# rocm_tools

Verification and measurement tools for the ROCm backend. Every claim in
[ROCM_PORT_MAP.md](ROCM_PORT_MAP.md) is reproducible with one of these — they
exist so the port's conclusions can be re-checked after a ROCm bump or on a
different RDNA card, rather than trusted.

All scripts resolve paths relative to the repo, use the active virtualenv's
torch, and detect the GPU via `rocminfo`. Nothing is hardcoded to one machine.

## `hipcc_probe.sh`

Compiles `exllamav3_ext` sources with the ROCm shim, using the same flags
`setup.py`'s `HIPBuildExtension` will. Much faster than a full build for
iterating on shim gaps.

```bash
rocm_tools/hipcc_probe.sh norm.cu        # one file
rocm_tools/hipcc_probe.sh --all          # all ROCm-built sources
GPU_ARCH=gfx1100 rocm_tools/hipcc_probe.sh --all   # another card
```

Current baseline on gfx1151 / ROCm 7.2.4: **44 of 50 pass**. The 6 failures are
exactly the inline-PTX files -- see ROCM_PORT_MAP.md. A number below 44 after a
toolchain change means the shim needs attention.

## `attn_check.py`

Checks upstream's in-tree Triton paged attention against an independent fp32
reference across decode/prefill/longq/paged, MHA and GQA, batched, head_dim
64/128/256.

```bash
python rocm_tools/attn_check.py
```

This is the gate that made the port viable: it establishes that FlashAttention
is not required, because upstream's own Triton kernels are numerically correct on
RDNA. Expected: **18/18**, max abs error ~1e-5 (fp16 rounding). Deliberately does
not import the compiled extension, so it runs before any kernel work.

## `bench_prefill_tiles.py`

Measures prefill throughput for the "Blackwell" tile config against the intended
non-Blackwell one.

```bash
python rocm_tools/bench_prefill_tiles.py
```

Exists because `triton_paged.py` selects tiles from
`get_device_capability()[0] >= 10`, which gfx1151 trips (it reports `(11, 5)`).
The measured result is counter-intuitive and worth re-checking on new hardware:
the misdetected narrow-kv config is *faster* on RDNA, because upstream sizes
tiles for ~100 KB of smem and RDNA 3.5 has 64 KB of LDS.

## Re-verifying after a ROCm upgrade

ROCm changed substantially between 7.1 and 7.2 — several workarounds in the
original fork became unnecessary, and at least one intrinsic changed
availability. Assume nothing carries over:

```bash
rocm_tools/hipcc_probe.sh --all          # shim still complete?
python rocm_tools/attn_check.py   # attention still correct?
python rocm_tools/bench_prefill_tiles.py # tile choice still right?
```

Anything in `exllamav3_ext/rocm/hip_compat.hip.h` commented as "absent from HIP"
should be re-grepped against `$ROCM_PATH/include` and deleted if HIP has since
grown it.

## `bench_decode_splits.py`

Sweeps decode split-K against the shipped heuristic.

```bash
python rocm_tools/bench_decode_splits.py
```

Exists because `multi_processor_count` reports WGPs on ROCm rather than CUs
(gfx1151: 20 for a 40-CU part), so the split target is half what the same code
assumes on NVIDIA. Doubling it was tried and reverted — the apparent 6-7% gain
did not survive repeat measurement. Kept so the question can be re-answered on
parts with a different CU count or CU/WGP ratio.

## On measurement noise

The decode step has a **4.7% run-to-run spread** (1.6% stdev over 8 repeats) on
gfx1151, with the first measurement reading high as clocks ramp. Anything under
~5% needs repeating before it means anything — one apparent decode win in this
port evaporated under that test, while the prefill tile result held at ~12%
across three runs with under 1% spread.

Both benchmarks are cheap. Run them several times before acting on a result.

## Hardware and kernel notes

The measured facts these tools exist to protect — WMMA operand order and fragment
layout, `sudot4` vs `sdot4`, the 64 KB LDS budget, the noise floor, and why each
RDNA sibling differs from the upstream file it replaces — are in
[`exllamav3/exllamav3_ext/rocm/RDNA_NOTES.md`](../exllamav3/exllamav3_ext/rocm/RDNA_NOTES.md).
