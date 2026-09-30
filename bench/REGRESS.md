# Regression suite

`bench/regress.sh` shows that an optimization branch hasn't regressed any model. It measures the branch and compares every number against a reference recorded on a known-good commit. Then it prints a PASS/FAIL table and writes a markdown report.

```
bench/regress.sh check quick          # ~14 min: DS4 + Qwen3.8, gates, isa diff
bench/regress.sh check full           # every model, every metric (~65-75 min)
bench/regress.sh check full --resume  # continue the latest interrupted check of this commit
bench/regress.sh show                 # print the reference table (also in bench/regress_ref/REF.md)
bench/regress.sh report <run dir>     # rebuild the report of a finished or interrupted run
```

The exit code is 0 only if every metric passes. Reports go to `bench/results/regress/<commit>_<timestamp>.md`. Per-step logs, the bench JSONs and `state.json` go to `exlproject/logs/regress_runs/<commit>_<mode>_<tier>_<timestamp>/`.

**Build first.** The suite measures whatever `exllamav3_ext*.so` sits in the repo. It does not build. Every report records the .so's sha256 and build time, and the isa step flags a .so identical to the reference's.

## What runs

Metric names are `<model>/<mode>/<workload>`, for example `ds4/plain/pp512`, `ds4/mtp2/tg128@d1024`, `laguna/dflash3/tg128@d2048`, `qwen/plain/regen8000`, `gemma/bitwise/long`, `glm/ppl/wiki2`, `suite/gates/gates`.

| Workload | Tool | What it measures |
|---|---|---|
| `pp512`, `pp2048` | `bench/run_bench.py` | Prefill tok/s on a random-token prompt. |
| `tg128@d1024`, `tg128@d2048` | `run_bench.py --tg_depth` | Decode tok/s for 128 new tokens after an N-token natural-text prompt (wikitext slice + summarize instruction). Plain mode uses DefaultSampler (seed 1234). |
| `mtp2/…`, `dflash3/…` | `run_bench.py --mtp -ndt 2` / `-dm <drafter> -ndt 3` | The same depth workload, greedy, with MTP (ndt 2) or the Laguna DFlash drafter (ndt 3). The report shows acceptance next to each value. |
| `regen3000`, `regen8000` | `run_bench.py --regen` | Latency in ms to re-send a primed prompt (cached-prefix + tail prefill). |
| `bitwise/short`, `bitwise/long` | `bench/regress_bitwise.py` | The method from `rocm_tools/decode_bitwise.py`: a greedy 48-token decode with every step's logits compared bit for bit against the reference. `short` is decode_bitwise's prompt (~20 tokens). `long` repeats it x40 (~800 tokens), so the prefill also takes the large-row paths. |
| `ppl/wiki2` | `eval/ppl.py` | wikitext2, 2048-token rows: 100 rows for DS4, Qwen and Gemma (Gemma with `-gp`, as in `run_ppl.sh`), 20 rows for MiMo and GLM. |
| `suite/gates` | `bench/run_gates.sh` | The mgemv / reconstruct / DSA ladder, the WMMA gate and the pytest suite. |
| `suite/isa` | `rocm_tools/isa_diff.py` | Device-code diff between the reference .so and the current one: the count of existing functions that changed. Informational only. |

Each model loads the way the server loads it (`model_init`):

- **DS4-Flash 2.04bpw** and **Qwen3.8 4bpw** use `-cs 65536`. Qwen also uses `-ngr`.
- **MiMo, GLM-5.3, Laguna and Gemma** use `-cs 32768`.

Each perf workload is 1 warmup + 3 timed runs, and the metric is the median.

### Tiers

| Tier | Models | Metrics |
|---|---|---|
| `quick` | DS4, Qwen3.8 (`-ngr`) | DS4: pp512, pp2048, regen3000, tg128@d1024, MTP ndt 2 @d1024, bitwise short. Qwen: pp512, pp2048, tg128@d1024, bitwise short. Plus gates (~4 min) and the isa diff (~1 min). |
| `full` | all six | Every model: pp512, pp2048, tg128@d1024/@d2048, and MTP (DS4, Qwen, MiMo, GLM) or DFlash (Laguna) @d1024/@d2048. DS4 and Qwen add regen 3000/8000. Every model runs bitwise short + long. PPL runs for every model except Laguna. Plus gates and the isa diff. |

To narrow a run, pass `--models ds4 gemma` (add `suite` to keep gates and isa) or `--only 'ds4/plain/*' 'qwen/bitwise/*'` (globs over metric names).

## Tolerances

| Kind | Fails when | Notes |
|---|---|---|
| Plain perf (pp, tg) | more than **2% below** the reference **and** the drop reproduces | A failing metric is re-measured once, automatically, in a new process. It fails only if both attempts fail, and the better attempt is reported. |
| Spec (MTP / DFlash) | more than **8% below** | Wider because acceptance depends on the text: each run uses a different wikitext slice. Reruns work as for plain perf. |
| Regen latency | more than **3% higher** | Reruns work as for plain perf. |
| PPL | more than **0.5% worse** | Any change is reported. |
| Decode logits | any bit differs | Pass `--allow-numerics <model> [...]` for an intentional numeric change: the diff then shows as CHANGED with the first differing step and the max abs diff, and does not fail. |
| Gates | a step fails, or a test fails that is not a known flake | Known flake: `test_dflash2.py::test_topk_cuda_matches_torch` (top-k tie order). Tests that already failed in the reference are also allowed. |
| isa diff | never | Informational. |

An improvement beyond the tolerance shows as IMPROVED and never fails. A metric missing from the output (crash, timeout, thermal kill) shows as ERROR, which fails.

## Safety and system state

- Every GPU step is its own process under `exlproject/thermal_guard.py` (kills at 99.5 C) and `systemd-run --user --scope -p MemoryMax=112G -p MemorySwapMax=0`, with a `timeout`.
- Only one model process runs at a time. Before each load, the suite waits until no other process holds `/dev/kfd`, MemAvailable is at least 96 GiB, and Tctl is below 55 C (40 C for GLM and Gemma; at most 10 min).
- **The suite refuses to run** when CPU boost is on, or when swap is on and Qwen (`-ngr`) is selected.
- `record` also refuses if `EXL3_*` overrides are set or the tree has uncommitted changes outside `bench/`.
- Every report records the system state: commit, torch / HIP / rocm version, kernel, boost, swap, GPU perf level, env overrides, the .so hash, and the peak Tctl per step.
- Results are saved after every step, so a crash loses at most the running step. `--resume` (or `--resume <run dir>`) skips the completed steps.

## Reference data

- **`bench/regress_ref/ref.json`** (in git) holds every metric's median, the per-run values, spread, acceptance, the commit and time it was recorded, the stack info, and the record history. `bench/regress_ref/REF.md` is the same data as a table.
- **Large binaries are not in git.** They live in `exlproject/logs/regress_refs/<commit>/`:
  - the bitwise logits (`<model>_<variant>.pt`, ~50 MB each);
  - a copy of the reference `exllamav3_ext*.so` for the isa diff.

  `ref.json` points to both.
- **`record` checks that bitwise references are deterministic.** It saves the logits, then decodes again in a second process and compares. A reference that doesn't reproduce is marked `nondeterministic` and is only reported, never used as a bar.
- **`record` re-measures noisy plain metrics.** Any plain or regen metric whose spread is above 5% is measured once more, and the attempt with the lower spread is kept.

## Re-recording after an intentional change

When a branch changes numbers on purpose (a faster kernel, different rounding) and has been reviewed:

1. Merge it, rebuild the .so, and make sure the tree is clean.
2. Run `bench/regress.sh check full --allow-numerics <models>` and read the report. Every difference in it should be one you intended.
3. Re-record what changed. `bench/regress.sh record full` re-records everything. To re-record part of it, use `bench/regress.sh record full --models ds4 suite` or `--only 'ds4/bitwise/*'`. A partial record merges into `ref.json`, and each metric keeps the commit it was recorded on.
4. Commit `bench/regress_ref/ref.json` and `REF.md`. The binaries stay in `exlproject/logs/regress_refs/<new commit>/`.

Re-record the isa reference (`--only suite/isa/isa`) whenever you accept new kernel code. Otherwise the "changed functions" count keeps growing.

To test the suite itself without touching the real reference, pass `--ref /some/scratch/ref.json`.

## Measured run times and known gaps (reference `dcc2f05`, 2026-09-29)

- **Timings.** `record full` took 61.5 min of steps: bench, bitwise save + verify, PPL, gates and isa for all six models. `check quick` took 13.4 min wall and passed with 11 PASS + 1 INFO. See `bench/results/regress/dcc2f05_20260929-223832.md`.
- **Eight metrics have no reference yet, because the thermal guard killed them.** They are `glm/plain/*` and `glm/mtp2/*` (GLM hits 100.1 C during its load warmup, which draws ~120 W), plus `gemma/plain/tg128@d2048` and `gemma/ppl/wiki2`. Each crossed 99.5 C Tctl on two attempts, again on a retry started from 39 C, and in a second retry run. The chart run earlier the same day peaked at 96.5 / 97.5 C on these models. Until they are recorded, `check full` shows them as NOREF, which is reported but does not fail. Once the box runs cooler, record them with `bench/regress.sh record full --only 'glm/plain/*' 'glm/mtp2/*' 'gemma/plain/tg128@d2048' 'gemma/ppl/wiki2'`. The guard limit is not to be raised.
