# Engineer questions

Shared question log for agents working in this repo (see PLAN.md §5a, kept at `exlproject/PLAN.md`).
Append new entries at the bottom; never delete or rewrite another entry. Only the maintainer answers.

Q-1 to Q-10 were answered by the maintainer in a kickoff interview (2026-09-26), before Phase 0.

### Q-1: What branch and stack are the baseline and opt/* branches built on?
- Status: answered
- Asked by: kickoff interview, 2026-09-26
- Context: `rocm_exl3/` is a fresh clone on `main`. The ROCm 10 work is on `origin/rocm-10`.
- Answer: Build `rocm_exl3/` in `.venv10`. Checking out rocm-10 is fine, and plain `main` should work too. Stay as current as possible on ROCm 10 and PyTorch. The venv may be uninstalled and reinstalled freely.

### Q-2: Upgrade .venv10 to the PyTorch nightly before the baseline?
- Status: answered
- Asked by: kickoff interview, 2026-09-26
- Answer: Yes. Upgrade to the nightly (download.pytorch.org/whl/nightly/rocm10.0) first, then take the baseline. Keep the stable 2.13 pip freeze for rollback, and record the stable numbers as a reference point.

### Q-3: PLAN says ~20 t/s decode / ~400 t/s pp512; RDNA_NOTES measured 18.1 / 112. Which is right?
- Status: answered
- Asked by: kickoff interview, 2026-09-26
- Answer: Re-measure. The bench harness must report the same numbers that exl3_server shows, so that measurements stay consistent.

### Q-4: Decode benchmark mode: plain, MTP, or both?
- Status: answered
- Asked by: kickoff interview, 2026-09-26
- Answer: Both. Plain decode is the primary metric that kernel work is judged on. MTP (dflash, ndt=2) is reported alongside it.

### Q-5: Where do commits and plan artifacts go?
- Status: answered
- Asked by: kickoff interview, 2026-09-26
- Answer: Commit locally in `rocm_exl3/` and never push; the maintainer pushes. Artifacts (bench/, PROFILE.md, RESULTS.md, this file) are committed in the repo.

### Q-6: How strict is gate #5 (upstream diff surface)?
- Status: answered
- Asked by: kickoff interview, 2026-09-26
- Answer: The rule is split:
  - **Python driver files** (`exllamav3/*.py`, modules, model code): ROCm hooks are allowed, and more than that is OK. Most rebase work is upstream adding new architectures, and that happens in the CUDA side.
  - **Hook script:** write a script that inserts the hooks, so rebasing stays easy. It lives in the `exlproject/` root, not in the repo.
  - **`exllamav3_ext`** (the engine): the sibling rule still holds. Kernel work goes in the `rocm/` siblings, never in upstream ext files.

### Q-7: Fixed PPL eval for gate #4?
- Status: answered
- Asked by: kickoff interview, 2026-09-26
- Answer: wikitext2 at 2048 ctx, with a fixed row count, on DeepSeek-V4-Flash 2.04bpw, Qwen3.8-Flash-Next 4bpw and Gemma-4-31B.

### Q-8: How do counter runs get stable clocks?
- Status: answered
- Asked by: kickoff interview, 2026-09-26
- Context: rocprofv3 and rocprof-compute do not set clocks themselves. On RDNA3/4 the perfmon clock is gated at `auto`.
- Answer: The maintainer runs `sudo chmod 666 /sys/class/drm/card0/device/power_dpm_force_performance_level` once per boot. After that the agent switches between `profile_standard` and `auto` itself, and must always restore `auto` when a run finishes. If the file is root-only again (after a reboot), ask the maintainer.

### Q-9: Can the GPU be shared with the maintainer during runs?
- Status: answered
- Asked by: kickoff interview, 2026-09-26
- Answer: No. The GPU is exclusive to the agent, which should run unattended.

### Q-10: What happens if a profiler wedges the GPU?
- Status: answered
- Asked by: kickoff interview, 2026-09-26
- Answer: Run a trivial canary capture first, and put a timeout on every profiler run. On a hang, kill it, log it here, and continue with non-GPU work until the maintainer reboots. Never run `stream_wedge_check`.

### Q-11: The APU thermal-tripped once during a DS4 bench. How should GPU work proceed?
- Status: blocking
- Asked by: Phase 0 session, 2026-09-26
- Context:
  - The box powered off during the first DS4 bench (about 19:58). The next boot logged `Previous system reset reason [0x00200a00]: internal CPU thermal limit was tripped`. Normal boots log 0x00200800.
  - Fans confirmed spinning by the maintainer. BIOS unchanged (EVO-X2 1.04, 05/14/2025).
  - Retest: a 90 s fp16 matmul under `exlproject/thermal_guard.py` (kill at Tctl 95 C, below the 100 C Tjmax spec). Tctl went 34 → 95 C in about 4 s at PPT 127 W, and the guard killed it.
  - GPU edge was only 58 C at the kill, a 36 C gap between hotspot and edge. Trace: `logs/thermal/20260926-200624.csv`.
- Options considered:
  - A: lower the BIOS power mode (e.g. balanced). This changes absolute bench numbers, but server and bench would match.
  - B: software power cap (ryzenadj / amd-smi power cap, needs root).
  - C: check the heatsink and paste, since a rise that fast points at die-to-cooler contact.
- Correction (same session): the 34 → 95 C ramp is normal behaviour for the Tctl hotspot sensor, not evidence of overheating. The chip never exceeded spec; the kill line (95 C) sat inside the normal full-load range. Options A–C were premature. The only evidence of a problem is the single trip.
- Maintainer (in chat): the fans are fine, and the box has run LLMs and builds without issue. The kill line may go to 98 C (never above the 100 C spec). The maintainer is running a CPU stress test.
- Next: rerun the 90 s ramp with the kill at 98 C. If Tctl plateaus below 98, treat the trip as a one-off and resume Phase 0 with the guard on every GPU job.
- What I did meanwhile: all GPU work paused. Non-GPU Phase 0 items (WMMA doc, resource tables from code objects, PPL/bench scripting) can continue.
- Answer:
