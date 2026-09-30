#!/usr/bin/env bash
# Regression suite entry point: bench/regress.sh record|check|show|report [quick|full] [options]
# (see bench/REGRESS.md and bench/regress.py --help). Sources exlproject/benv.sh for the .venv10 stack.
set -uo pipefail
ROOT=/home/carousel/Desktop/exlproject
source "$ROOT/benv.sh"
exec "$PY" "$(cd "$(dirname "$0")" && pwd)/regress.py" "$@"
