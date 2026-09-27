#!/usr/bin/env python3
"""Per-kernel summary of a bench/run_counters.sh collection (Phase 0.6).

    bench/analyze_counters.py OUTDIR [--top 12] [--match REGEX]

Averages each counter per dispatch for every kernel, then derives:
  GB/s       FETCH_SIZE (KB) / dispatch duration (stable clocks: absolute GB/s reads low;
             compare kernels against each other, not against the 210 GB/s peak)
  MemBusy    MemUnitBusy (%)
  Occ        OccupancyPercent (%)
  VALU/wcyc  SQ_INSTS_VALU / SQ_WAVE_CYCLES: fraction of a wave's resident cycles that issue
             a VALU instruction (upper bound 1; high = ALU-bound, low = waiting on memory)
  B/VALU     bytes fetched per VALU instruction (low = ALU-heavy per byte)
  LDSconf    SQC_LDS_BANK_CONFLICT / SQC_LDS_IDX_ACTIVE
  VGPRfull / WAVEfull / TMPstall   SPI_RA_* resource-allocation stall counts per dispatch
"""

import argparse
import collections
import csv
import glob
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--match", default=None)
    a = ap.parse_args()

    val = collections.defaultdict(lambda: collections.defaultdict(list))   # kernel -> counter -> [v]
    dur = collections.defaultdict(list)                                     # from pass 1 only
    for p in sorted(glob.glob(os.path.join(a.outdir, "**", "*counter_collection.csv"), recursive=True)):
        seen = set()
        for r in csv.DictReader(open(p, newline="")):
            k = r["Kernel_Name"]
            val[k][r["Counter_Name"]].append(float(r["Counter_Value"]))
            d = r["Dispatch_Id"]
            if "pass_1" in p and d not in seen:
                seen.add(d)
                dur[k].append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3)

    def m(k, c):
        v = val[k].get(c)
        return sum(v) / len(v) if v else float("nan")

    rows = []
    for k in val:
        if a.match and not re.search(a.match, k):
            continue
        if not dur[k]:
            continue
        us = sum(dur[k]) / len(dur[k])
        tot = sum(dur[k])
        fetch = m(k, "FETCH_SIZE") * 1024
        rows.append(dict(
            k=k, calls=len(dur[k]), us=us, tot=tot,
            gbs=fetch / (us * 1e3) if us else float("nan"),
            mem=m(k, "MemUnitBusy"), occ=m(k, "OccupancyPercent"),
            valu_wc=m(k, "SQ_INSTS_VALU") / m(k, "SQ_WAVE_CYCLES"),
            b_valu=fetch / m(k, "SQ_INSTS_VALU"),
            lds=m(k, "SQC_LDS_BANK_CONFLICT") / max(m(k, "SQC_LDS_IDX_ACTIVE"), 1),
            vg=m(k, "SPI_RA_VGPR_SIMD_FULL_CSN"), wv=m(k, "SPI_RA_WAVE_SIMD_FULL_CSN"),
            tmp=m(k, "SPI_RA_TMP_STALL_CSN")))
    rows.sort(key=lambda r: -r["tot"])
    print(f"{'tot ms':>8} {'us':>8} {'GB/s':>6} {'Mem%':>5} {'Occ%':>5} {'VALU/wc':>7} {'B/VALU':>7} "
          f"{'LDScf':>6} {'VGPRf':>7} {'WAVEf':>7} {'TMPst':>7}  kernel")
    for r in rows[: a.top]:
        name = re.sub(r"\s+", " ", r["k"])[:70]
        print(f"{r['tot']/1e3:8.1f} {r['us']:8.1f} {r['gbs']:6.0f} {r['mem']:5.1f} {r['occ']:5.1f} "
              f"{r['valu_wc']:7.3f} {r['b_valu']:7.2f} {r['lds']:6.3f} {r['vg']:7.0f} {r['wv']:7.0f} "
              f"{r['tmp']:7.0f}  {name}")


if __name__ == "__main__":
    main()
