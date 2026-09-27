#!/usr/bin/env bash
# Gate #4 perplexity eval (ENGINEER_QUESTIONS Q-7): wikitext2 test, 100 rows x 2048 tokens,
# on the three gate models. Each model runs in its own process under exlproject/thermal_guard.py.
#
#   bench/run_ppl.sh [outdir]          # default outdir: bench/results/ppl
#   MODELS="ds4" bench/run_ppl.sh      # subset: ds4 qwen gemma
#
# Output: <outdir>/<commit>_<model>.log, plus a one-line summary per model on stdout.
set -uo pipefail

ROOT=/home/carousel/Desktop/exlproject
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/benv.sh"
OUT=${1:-$REPO/bench/results/ppl}
mkdir -p "$OUT"
COMMIT=$(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet HEAD -- . ':!bench' || echo "-dirty")

declare -A PATHS=(
  [ds4]="$HOME/models/DeepSeek-V4-Flash-0731-exl3-2.04bpw"
  [qwen]="$HOME/models/Qwen3.8-Flash-Next-Uncensored-exl3-4bpw"
  [gemma]="$HOME/models/gemma-4-31b-it-exl3"
)
# Gemma needs BOS at position 0. ppl.py slices raw wikitext with no BOS (Gemma PPL 1163 on 10 rows);
# -gp prepends the chat template including BOS (16.88). Qwen/DS4 are unaffected and run without it.
declare -A FLAGS=([gemma]="-gp")

for m in ${MODELS:-ds4 qwen gemma}; do
  log="$OUT/${COMMIT}_${m}.log"
  echo " -- $m -> $log"
  python3 "$ROOT/thermal_guard.py" --kill 99.5 -- \
    timeout 5400 "$PY" "$REPO/eval/ppl.py" -m "${PATHS[$m]}" -r 100 -l 2048 ${FLAGS[$m]:-} > "$log" 2>&1
  echo "    rc=$? $(grep -E 'Perplexity|ppl' "$log" | tail -1)"
done
