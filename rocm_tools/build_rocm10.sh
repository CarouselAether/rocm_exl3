#!/usr/bin/env bash
# Build and install the extension against the ROCm 10 pip SDK in the active venv.
# Run from the repo root with the venv activated (see requirements_rocm10.txt).
#
# A login shell often exports a system ROCm (/opt/rocm) through PATH,
# LD_LIBRARY_PATH, ROCM_PATH or HIP_PATH. Left in place, the build can pick up
# the system hipcc or link the extension against the system runtime while torch
# loads the wheel's, and whichever libamdhip64 loads first wins for the whole
# process. So: scrub it, then point everything at the wheel SDK.
#
# ROCM_HOME matters, not just ROCM_PATH: torch's extension builder takes the
# runtime rpath from it.
#
# Extra args go to pip, e.g.  rocm_tools/build_rocm10.sh -e   (editable install).
# PYTORCH_ROCM_ARCH=gfx1201 and MAX_JOBS pass through.
set -euo pipefail

command -v rocm-sdk >/dev/null || { echo "rocm-sdk not found: activate the venv and pip install -r requirements_rocm10.txt" >&2; exit 1; }

unset ROCM_PATH ROCM_HOME HIP_PATH LD_LIBRARY_PATH LD_PRELOAD ROCP_TOOL_LIBRARIES
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/opt/rocm' | paste -sd ':')

SDK=$(rocm-sdk path --root)
# rocm-sdk init expands the devel wheel (hipcc, device libs) into $SDK; idempotent.
[ -x "$SDK/bin/hipcc" ] || rocm-sdk init

export PATH="$SDK/bin:$PATH" ROCM_PATH="$SDK" ROCM_HOME="$SDK"
echo "hipcc: $(command -v hipcc)  ($(hipcc --version | grep -m1 'HIP version'))"
pip install --no-build-isolation "$@" .
