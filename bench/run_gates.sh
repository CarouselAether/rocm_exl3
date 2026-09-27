#!/usr/bin/env bash
# Agent-run gates 1-2 (PLAN.md §2): numeric ladder + pytest suite, each under thermal_guard.
#
#   bench/run_gates.sh [logdir]       # default: exlproject/logs/gates/<commit>
#
# Upstream tests hardcode a device index (cuda:1 / cuda:2) for turboderp's multi-GPU box
# (RDNA_NOTES "Test status on RDNA"). They are upstream files, so they are not edited in the
# repo: they are copied to a scratch dir and rewritten to cuda:0 there.
set -uo pipefail

ROOT=/home/carousel/Desktop/exlproject
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/benv.sh"
COMMIT=$(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet HEAD -- . ':!bench' || echo "-dirty")
L=${1:-$ROOT/logs/gates/$COMMIT}
mkdir -p "$L"
G="python3 $ROOT/thermal_guard.py --"
DS4=$HOME/models/DeepSeek-V4-Flash-0731-exl3-2.04bpw

# Some tests read sources at ../exllamav3/..., so the copy sits at $TT/tests next to a symlink
TT=$(mktemp -d "${TMPDIR:-/tmp}/exl3tests.XXXX")
T=$TT/tests
mkdir -p "$T"
cp -r "$REPO/tests/." "$T/"
ln -s "$REPO/exllamav3" "$TT/exllamav3"
sed -i -E 's/"cuda:[1-9]"/"cuda:0"/g' "$T"/*.py

fail=0
step() {   # name, command...
  local name=$1; shift
  (cd "$T" && $G timeout 3600 "$@") > "$L/$name.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && fail=1
  printf "  %-22s rc=%-3s %s\n" "$name" "$rc" "$(grep -E 'PASS|FAIL|passed|failed|error' "$L/$name.log" | tail -1)"
}

echo " -- gates for $COMMIT -> $L"
step mgemv_check        "$PY" "$REPO/rocm_tools/mgemv_check.py" -m "$DS4"
step reconstruct_had    "$PY" "$T/test_reconstruct_had.py"
step dsa_kernels        "$PY" "$T/test_dsa_kernels.py"
# Not runnable on this box (collection errors, not failures): two-GPU test; upstream /mnt/str
# stub models; the uncommitted compare_deepseek_v4_hf_ reference module. Plus two CUDA-only
# kernels the port rejects by design with a RuntimeError: hgemm_f16acc (inline PTX; hipBLAS is
# used instead) and exl3_moe_coop (PTX GEMV; disabled on ROCm). Baseline 97063b3 + torch nightly
# 2.15.0.dev20260926: 879 passed, 11 skipped.
IGNORE="test_hgemm_f16acc.py test_moe_coop.py test_device_copy_.py test_qgemm.py test_quant_fn.py test_ngram_prefetch_.py test_dsv4_cached.py test_dsv4_state.py"
step pytest             "$PY" -m pytest -q -p no:cacheprovider $(for f in $IGNORE; do echo --ignore="$T/$f"; done) "$T"
rm -rf "$TT"
echo " -- gates $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $fail
