#!/usr/bin/env bash
# Phase 0.6: hardware counters on one profile_region.py region, at stable clocks.
#
#   bench/run_counters.sh NAME [profile_region.py args...]
#   e.g. bench/run_counters.sh ds4_decode_pmc --mode decode --ctx 512 --steps 8
#
# Switches the GPU perf level to profile_standard (the perfmon clock is gated at `auto` on
# RDNA3/4, so some counters read zero) and ALWAYS restores `auto` on exit, even on failure.
# Needs the sysfs power_dpm_force_performance_level file made writable by the maintainer.
# EXCLUDE=regex skips kernels from collection (exl3_moe: PMC deadlocks co-resident kernels, RDNA_NOTES).
# Every pass runs under thermal_guard with a timeout (Q-10). rocprofv3 re-runs the app per --pmc pass.
set -uo pipefail

ROOT=/home/carousel/Desktop/exlproject
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/benv.sh"
NAME=${1:?name}; shift
OUT=$ROOT/logs/prof/$NAME
LVL=/sys/class/drm/card0/device/power_dpm_force_performance_level
MODEL=${MODEL:-$HOME/models/DeepSeek-V4-Flash-0731-exl3-2.04bpw}

[ -w "$LVL" ] || { echo "perf-level file not writable; ask the maintainer to chmod it (Q-8)"; exit 1; }
restore() { echo auto > "$LVL"; echo " -- perf level restored: $(cat "$LVL")"; }
trap restore EXIT
echo profile_standard > "$LVL"
echo " -- perf level: $(cat "$LVL")"

python3 "$ROOT/thermal_guard.py" -- timeout "${TIMEOUT:-3000}" "$ROOT/.venv10/bin/rocprofv3" --selected-regions ${EXCLUDE:+--kernel-exclude-regex "$EXCLUDE"} \
  --pmc FETCH_SIZE \
  --pmc MemUnitBusy \
  --pmc OccupancyPercent \
  --pmc "SQ_WAVES SQ_INSTS_VALU SQ_WAVE_CYCLES SQ_BUSY_CYCLES GRBM_GUI_ACTIVE" \
  --pmc "SQC_LDS_BANK_CONFLICT SQC_LDS_IDX_ACTIVE SQ_INSTS_LDS SQ_WAIT_INST_ANY" \
  --pmc "SPI_RA_VGPR_SIMD_FULL_CSN SPI_RA_WAVE_SIMD_FULL_CSN SPI_RA_LDS_CU_FULL_CSN SPI_RA_TMP_STALL_CSN" \
  --output-format csv -d "$OUT" -- "$PY" "$REPO/bench/profile_region.py" -m "$MODEL" "$@" > "$OUT.log" 2>&1
rc=$?
echo " -- rocprofv3 rc=$rc, $(find "$OUT" -name '*counter_collection.csv' | wc -l) counter files"
exit $rc
