#!/usr/bin/env python3
"""Summarize a rocprofv3 CSV trace of one profile_region.py region (Phase 0.5 / 0.7).

    bench/analyze_trace.py OUTDIR [--region decode|prefill] [--top 15] [--json out.json]

Reads *kernel_trace.csv, *marker_api_trace.csv, *hip_api_trace.csv and
*memory_copy_trace.csv under OUTDIR (rocprofv3 -d OUTDIR --output-format csv).

Reports, for the chosen roctx region:
  - wall time (roctx range) and GPU busy time (union of kernel intervals), per step
  - top kernels by total time: calls/step, avg us, share of busy
  - idle gaps between consecutive kernels: count, histogram, total
  - D2H / H2D copies and host synchronizations per step
  - per-kernel resources: VGPR, AGPR, SGPR, LDS, scratch (from the dispatch records)
"""

import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics


def load(outdir, suffix):
    rows = []
    for p in glob.glob(os.path.join(outdir, "**", f"*{suffix}"), recursive=True):
        with open(p, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def col(row, *names):
    for n in names:
        if n in row and row[n] != "":
            return row[n]
    return None


def short(name, n=90):
    name = re.sub(r"\s+", " ", name)
    return name if len(name) <= n else name[: n - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--region", default=None, help="roctx range name (decode/prefill); default: whole trace")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    kern = load(a.outdir, "kernel_trace.csv")
    mark = load(a.outdir, "marker_api_trace.csv")
    hip = load(a.outdir, "hip_api_trace.csv")
    mcp = load(a.outdir, "memory_copy_trace.csv")

    def mname(r):
        return col(r, "Function", "Message", "Name") or ""

    # region window and step windows from roctx ranges
    t_lo, t_hi, steps = None, None, []
    for r in mark:
        s, e = int(col(r, "Start_Timestamp")), int(col(r, "End_Timestamp"))
        n = mname(r)
        if a.region and n == a.region:
            t_lo, t_hi = s, e
        if n == "step":
            steps.append((s, e))
    if a.region and t_lo is None:
        raise SystemExit(f"region '{a.region}' not found; markers seen: {sorted({mname(r) for r in mark})[:20]}")
    if t_lo is not None:
        steps = [x for x in steps if x[0] >= t_lo and x[1] <= t_hi]
    n_steps = max(len(steps), 1)

    ks = []
    for r in kern:
        s, e = int(col(r, "Start_Timestamp")), int(col(r, "End_Timestamp"))
        if t_lo is not None and (e < t_lo or s > t_hi + 2_000_000_000):
            continue
        ks.append((s, e, col(r, "Kernel_Name") or "?", r))
    ks.sort(key=lambda x: x[0])
    if not ks:
        raise SystemExit("no kernels in window")
    if t_lo is None:
        t_lo, t_hi = ks[0][0], ks[-1][1]
    # GPU work can trail the CPU range slightly; the window is the union of both
    g_lo, g_hi = min(t_lo, ks[0][0]), max(t_hi, ks[-1][1])
    wall = (t_hi - t_lo) / 1e3  # us

    busy, gaps, cur_s, cur_e = 0, [], ks[0][0], ks[0][1]
    for s, e, _, _ in ks[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            gaps.append(s - cur_e)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    busy /= 1e3

    per = collections.defaultdict(lambda: [0, 0.0])
    res = {}
    for s, e, n, r in ks:
        per[n][0] += 1
        per[n][1] += (e - s) / 1e3
        if n not in res:
            res[n] = {k: col(r, *v) for k, v in {
                "vgpr": ("Arch_VGPR_Count", "VGPR_Count"), "agpr": ("Accum_VGPR_Count",),
                "sgpr": ("SGPR_Count",), "lds": ("LDS_Block_Size", "Group_Segment_Size"),
                "scratch": ("Scratch_Size", "Private_Segment_Size"),
                "wg": ("Workgroup_Size", "Workgroup_Size_X"), "grid": ("Grid_Size", "Grid_Size_X")}.items()}
    top = sorted(per.items(), key=lambda x: -x[1][1])

    def in_win(r):
        s = col(r, "Start_Timestamp")
        return s is not None and g_lo <= int(s) <= g_hi

    hipc = collections.Counter(mname(r) for r in hip if in_win(r))
    sync_fns = {k: v for k, v in hipc.items() if re.search(r"Synchronize|StreamQuery|EventQuery|hipMemcpy(?!.*Async)", k)}
    copies = collections.Counter(col(r, "Direction") or "?" for r in mcp if in_win(r))

    gap_us = [g / 1e3 for g in gaps]
    hist = collections.Counter(
        "<2us" if g < 2 else "2-5us" if g < 5 else "5-20us" if g < 20 else "20-100us" if g < 100 else ">=100us"
        for g in gap_us)

    out = {
        "steps": n_steps, "wall_us_per_step": wall / n_steps, "busy_us_per_step": busy / n_steps,
        "busy_frac": busy / wall if wall else None, "kernels_per_step": len(ks) / n_steps,
        "gaps_per_step": len(gaps) / n_steps, "gap_us_per_step": sum(gap_us) / n_steps,
        "gap_mean_us": statistics.mean(gap_us) if gap_us else 0, "gap_hist": dict(hist),
        "sync_calls_per_step": {k: v / n_steps for k, v in sync_fns.items()},
        "copies_per_step": {k: v / n_steps for k, v in copies.items()},
        "top": [{"kernel": n, "calls_per_step": c / n_steps, "total_us_per_step": t / n_steps,
                 "avg_us": t / c, "share": t / busy, **res[n]} for n, (c, t) in top[: a.top]],
        "resources_all": res,
    }

    print(f"region {a.region or '(all)'}: {n_steps} steps")
    print(f"  wall  {out['wall_us_per_step'] / 1e3:8.2f} ms/step   busy {out['busy_us_per_step'] / 1e3:8.2f} ms/step "
          f"({out['busy_frac']:.1%})")
    print(f"  kernels/step {out['kernels_per_step']:.0f}   gaps/step {out['gaps_per_step']:.0f}   "
          f"gap total {out['gap_us_per_step'] / 1e3:.2f} ms/step   mean gap {out['gap_mean_us']:.2f} us")
    print(f"  gap histogram: {dict(sorted(hist.items()))}")
    print(f"  host syncs/step: {out['sync_calls_per_step']}")
    print(f"  copies/step: {out['copies_per_step']}")
    print(f"\n  {'share':>6} {'us/step':>9} {'calls':>6} {'avg us':>8} {'vgpr':>5} {'sgpr':>5} {'lds':>6} {'scr':>6}  kernel")
    for t in out["top"]:
        print(f"  {t['share']:6.1%} {t['total_us_per_step']:9.1f} {t['calls_per_step']:6.1f} {t['avg_us']:8.1f} "
              f"{t['vgpr'] or '':>5} {t['sgpr'] or '':>5} {t['lds'] or '':>6} {t['scratch'] or '':>6}  {short(t['kernel'])}")
    scr = [(n, r["scratch"]) for n, r in res.items() if r.get("scratch") not in (None, "0", "")]
    print(f"\n  kernels using scratch in window: {len(scr)}")
    for n, s in sorted(scr, key=lambda x: -per[x[0]][1])[:20]:
        print(f"    scratch {s:>6}  {per[n][1] / n_steps:8.1f} us/step  {short(n)}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
