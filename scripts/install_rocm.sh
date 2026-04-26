#!/usr/bin/env bash
# install_rocm.sh — bring-your-own-ROCm installer for exllamav3 + native CK flash-attn.
#
# What this does:
#   1. Sanity-checks ROCm + amdgpu kernel module, detects gfx arch.
#   2. Creates (or reuses) a Python venv with ROCm-built torch + minimum deps.
#   3. Builds flash-attention from the in-tree ./flash-attention/ source against
#      the detected arch, with parallelism tuned to actually use your cores.
#   4. Builds the exllamav3 C++/HIP extension in-place.
#   5. Runs a sub-minute smoke test that flash-attn matches torch SDPA.
#
# What it does NOT do:
#   - install ROCm itself (you handle that)
#   - install the amdgpu kernel module / DKMS (you handle that)
#   - prebuilt wheels: build is from source. By design.
#
# Usage:
#   bash scripts/install_rocm.sh
#
# Environment overrides (all optional):
#   VENV_DIR          path to venv (default: ~/rocm_llm)
#   GPU_ARCHS         override auto-detected arch (e.g. gfx1100;gfx942)
#   MAX_JOBS          override auto-computed parallel jobs
#   NVCC_THREADS      override per-job internal threads (default: 2)
#   TORCH_INDEX_URL   override the torch wheel index (default: rocm7.2 nightly)
#   SKIP_SMOKE_TEST   set to 1 to skip the post-install verify step

set -euo pipefail

# -----------------------------------------------------------------------------
# Pretty output
# -----------------------------------------------------------------------------
C_R=$'\e[31;1m'; C_G=$'\e[32;1m'; C_Y=$'\e[33;1m'; C_B=$'\e[34;1m'; C_0=$'\e[0m'
say()  { printf "%s==>%s %s\n" "$C_B" "$C_0" "$*"; }
ok()   { printf "%s ok %s %s\n" "$C_G" "$C_0" "$*"; }
warn() { printf "%swarn%s %s\n" "$C_Y" "$C_0" "$*"; }
die()  { printf "%sfail%s %s\n" "$C_R" "$C_0" "$*" >&2; exit 1; }

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# 1. Sanity checks
# -----------------------------------------------------------------------------
say "Sanity-checking host"
[[ "$(uname -s)" == "Linux" ]] || die "Linux only. This is ROCm."
[[ -d /opt/rocm ]] || die "ROCm not found at /opt/rocm. Install ROCm first."
ROCM_VER="$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
ok "ROCm $ROCM_VER"

# amdgpu kernel module loaded?
if ! lsmod | grep -q '^amdgpu'; then
    warn "amdgpu kernel module not loaded. GPU work will fail."
fi

# ROCm dev-package check. Building FA + exllamav3 needs the dev-side headers
# and libraries, not just the runtime meta-package. Without rocm-dev (or
# equivalent), the build fails with "no such file" on hip/hip_runtime.h or
# undefined reference at link time. Probe a representative subset.
REQUIRED_DEV_PROBES=(
    "/opt/rocm/bin/hipcc"                              # hipcc / hip-runtime-amd
    "/opt/rocm/bin/rocminfo"                           # rocminfo
    "/opt/rocm/include/hip/hip_runtime.h"              # hip-dev
    "/opt/rocm/include/hipblas/hipblas.h"              # hipblas-dev
    "/opt/rocm/include/hiprand/hiprand.hpp"            # hiprand-dev
    "/opt/rocm/lib/libhipblas.so"                      # hipblas
    "/opt/rocm/lib/libamdhip64.so"                     # hip-runtime-amd
)
MISSING_DEV=()
for p in "${REQUIRED_DEV_PROBES[@]}"; do
    [[ -e "$p" ]] || MISSING_DEV+=("$p")
done
if (( ${#MISSING_DEV[@]} > 0 )); then
    # Distro-aware install hint. Reads /etc/os-release ID and ID_LIKE to pick
    # the right package manager. AMD packages are named consistently across
    # distros for the rocm-dev meta-package.
    DISTRO_ID=""
    DISTRO_LIKE=""
    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        DISTRO_ID="$(. /etc/os-release && echo "${ID:-}")"
        DISTRO_LIKE="$(. /etc/os-release && echo "${ID_LIKE:-}")"
    fi

    case "$DISTRO_ID $DISTRO_LIKE" in
        *ubuntu*|*debian*)
            INSTALL_HINT="sudo apt install rocm-dev" ;;
        *fedora*|*rhel*|*centos*|*rocky*|*almalinux*)
            INSTALL_HINT="sudo dnf install rocm-dev" ;;
        *suse*|*sles*|*opensuse*)
            INSTALL_HINT="sudo zypper install rocm-dev" ;;
        *arch*)
            INSTALL_HINT="sudo pacman -S rocm-hip-sdk rocm-device-libs" ;;
        *)
            INSTALL_HINT="(your distro: install the rocm-dev / ROCm SDK meta-package)" ;;
    esac

    printf '%s\n' "${MISSING_DEV[@]}" | sed 's/^/  missing: /' >&2
    die "ROCm dev packages incomplete. Install with:
       $INSTALL_HINT
   AMD's universal installer also works on all supported distros:
       sudo amdgpu-install --usecase=rocm,rocmdev
   The runtime-only ROCm meta-package does not include the headers + libs
   needed to build flash-attn / exllamav3."
fi
ok "ROCm dev-tools present"

# Detect ALL installed gfx archs (multi-GPU systems) and filter to the supported
# set. RDNA3+ (gfx11+/gfx12+) and CDNA2+ (gfx90a/gfx94x/gfx95x) only — earlier
# GPUs lack the matrix-core ISA the kernels need.
SUPPORTED_ARCHS=(gfx90a gfx942 gfx950 gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1200 gfx1201)
mapfile -t DETECTED_ARCHS < <(
    /opt/rocm/bin/rocminfo 2>/dev/null \
        | awk '/Name:[[:space:]]+gfx/ {print $2}' \
        | sort -u
)
[[ ${#DETECTED_ARCHS[@]} -gt 0 ]] || die "No gfx-arch detected by rocminfo. GPU not visible?"

if [[ -z "${GPU_ARCHS:-}" ]]; then
    SUPPORTED=()
    UNSUPPORTED=()
    for a in "${DETECTED_ARCHS[@]}"; do
        if [[ " ${SUPPORTED_ARCHS[*]} " == *" $a "* ]]; then
            SUPPORTED+=("$a")
        else
            UNSUPPORTED+=("$a")
        fi
    done
    if (( ${#UNSUPPORTED[@]} > 0 )); then
        warn "ignoring unsupported GPU arch(s): ${UNSUPPORTED[*]} (pre-RDNA3 / pre-CDNA2)"
    fi
    if (( ${#SUPPORTED[@]} == 0 )); then
        die "No supported GPU detected. Found: ${DETECTED_ARCHS[*]}. Need one of: ${SUPPORTED_ARCHS[*]}"
    fi
    GPU_ARCHS="$(IFS=';'; echo "${SUPPORTED[*]}")"
fi
ok "gfx arch(s): $GPU_ARCHS"

# flash-attention source — clone if missing. Pinned to a commit that's been
# verified to build cleanly against ROCm 7.2.1 + torch 2.11+rocm7.2.
FLASH_ATTN_REPO="${FLASH_ATTN_REPO:-https://github.com/dao-ailab/flash-attention.git}"
FLASH_ATTN_REF="${FLASH_ATTN_REF:-b65ae6b175f2438de55601695b6a21971fc5e429}"

if [[ ! -d "$REPO_ROOT/flash-attention" ]]; then
    say "Cloning flash-attention ($FLASH_ATTN_REPO @ ${FLASH_ATTN_REF:0:8})"
    git clone --filter=blob:none "$FLASH_ATTN_REPO" "$REPO_ROOT/flash-attention"
    git -C "$REPO_ROOT/flash-attention" checkout --quiet "$FLASH_ATTN_REF"
elif [[ ! -f "$REPO_ROOT/flash-attention/setup.py" ]]; then
    die "flash-attention/ exists but setup.py is missing. Remove it and re-run."
else
    ok "flash-attention/ already present (using existing checkout)"
fi

# -----------------------------------------------------------------------------
# 2. Compute parallelism
# -----------------------------------------------------------------------------
# FA's auto-pick uses min(cores/2, available_GB / (5 * NVCC_THREADS)) which
# becomes 1 job on 32 GB systems. We override both knobs so total threads ≈ cores
# while staying under 5 GB / thread of memory pressure.
CORES="$(nproc)"
TOTAL_RAM_GB="$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
AVAIL_RAM_GB="$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)"

# Parallelism on ROCm has two layers:
#   (a) MAX_JOBS  — how many hipcc invocations ninja runs concurrently (-j N)
#   (b) -parallel-jobs=N  — per-hipcc-invocation parallelism for HIP, splits the
#       host + device-per-arch compilation pipeline across N processes. Fed via
#       the standard env var HIPCC_COMPILE_FLAGS_APPEND.
#
# For multi-arch builds (GPU_ARCHS="gfx1100;gfx942"), -parallel-jobs lets each
# arch compile in parallel within one hipcc call. For single-arch it splits
# host vs device. Either way it's a real wall-time win.
#
# NVCC_THREADS is NVCC-specific; clang/hipcc doesn't honor it. On ROCm we only
# set NVCC_THREADS=1 so FA's auto-pick gives the most generous RAM budget — we
# then override MAX_JOBS ourselves anyway.
NVCC_THREADS="${NVCC_THREADS:-1}"

# Default HIPCC_PARALLEL_JOBS: number of distinct archs in GPU_ARCHS (capped 4),
# minimum 2 to get host/device overlap on single-arch.
NUM_ARCHS=$(awk -F';' '{print NF}' <<<"$GPU_ARCHS")
HIPCC_PARALLEL_JOBS="${HIPCC_PARALLEL_JOBS:-$(( NUM_ARCHS > 1 ? (NUM_ARCHS > 4 ? 4 : NUM_ARCHS) : 2 ))}"

# Each in-flight hipcc invocation now spawns ~HIPCC_PARALLEL_JOBS internal
# compile processes, each ~5 GB peak.
PER_JOB_GB=$(( 5 * HIPCC_PARALLEL_JOBS ))

if [[ -z "${MAX_JOBS:-}" ]]; then
    BY_RAM=$(( AVAIL_RAM_GB / PER_JOB_GB ))
    MAX_JOBS=$BY_RAM
    [[ $MAX_JOBS -gt $CORES ]] && MAX_JOBS=$CORES
    [[ $MAX_JOBS -lt 2 ]] && MAX_JOBS=2     # never ship a 1-core build
fi

TOTAL_PROCS=$(( MAX_JOBS * HIPCC_PARALLEL_JOBS ))
EXPECTED_RAM_GB=$(( TOTAL_PROCS * 5 ))

say "Build parallelism plan"
ok "host: ${CORES} cores, ${TOTAL_RAM_GB} GB RAM (${AVAIL_RAM_GB} GB available)"
ok "MAX_JOBS=$MAX_JOBS  HIPCC_PARALLEL_JOBS=$HIPCC_PARALLEL_JOBS  →  ${TOTAL_PROCS} concurrent compile processes"
ok "expected peak RAM: ~${EXPECTED_RAM_GB} GB"

if [[ $EXPECTED_RAM_GB -gt $(( AVAIL_RAM_GB - 4 )) ]]; then
    warn "build may approach RAM ceiling. Lower MAX_JOBS or HIPCC_PARALLEL_JOBS if it OOMs."
fi

# -----------------------------------------------------------------------------
# 3. venv + torch
# -----------------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-$HOME/rocm_llm}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/rocm7.2}"

if [[ ! -d "$VENV_DIR" ]]; then
    say "Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "venv: $(python --version) at $VENV_DIR"

say "Installing/updating torch (ROCm)"
pip install --upgrade pip wheel setuptools
pip install --pre torch --index-url "$TORCH_INDEX_URL"

# Strict ROCm gate. If torch isn't the ROCm build, this script can't help —
# CUDA users have working PyPI flash-attn wheels and don't need any of this.
TORCH_FLAVOR="$(python - <<'EOF'
import torch
print("rocm" if torch.version.hip else ("cuda" if torch.version.cuda else "cpu"))
EOF
)"
ok "torch: $(python -c 'import torch; print(torch.__version__)') (${TORCH_FLAVOR})"

if [[ "$TORCH_FLAVOR" != "rocm" ]]; then
    die "torch is the ${TORCH_FLAVOR} build, not ROCm. This script is ROCm-only.
       For CUDA, install flash-attn from PyPI:
           pip install flash-attn --no-build-isolation
       and build the exllamav3 extension with:
           python setup.py build_ext --inplace"
fi

# -----------------------------------------------------------------------------
# 4. Build flash-attn from in-tree source (ROCm CK backend)
# -----------------------------------------------------------------------------
say "Building flash-attention from in-tree source (ROCm CK backend)"
echo "  GPU_ARCHS=$GPU_ARCHS"
echo "  MAX_JOBS=$MAX_JOBS  HIPCC_PARALLEL_JOBS=$HIPCC_PARALLEL_JOBS  (${TOTAL_PROCS} total compile processes)"
echo "  Plan for nothing-else-running about now. Grab coffee."

# Silence the warning torrent from the upstream FA codebase. Errors still go
# to stderr and pipefail will catch any real failure. Pip's progress bars and
# package-level status are unaffected — only compiler warnings are suppressed.
# Set EXLLAMAV3_VERBOSE_BUILD=1 to see them all.
if [[ "${EXLLAMAV3_VERBOSE_BUILD:-0}" == "1" ]]; then
    QUIET_FLAGS=""
else
    QUIET_FLAGS="-Wno-unused-command-line-argument -Wno-deprecated-declarations -Wno-unused-variable -Wno-unused-function -Wno-unused-value -Wno-missing-field-initializers -Wno-#pragma-messages -Wno-pass-failed"
fi

pushd "$REPO_ROOT/flash-attention" >/dev/null
GPU_ARCHS="$GPU_ARCHS" \
MAX_JOBS="$MAX_JOBS" \
NVCC_THREADS="$NVCC_THREADS" \
HIPCC_COMPILE_FLAGS_APPEND="-parallel-jobs=${HIPCC_PARALLEL_JOBS} ${QUIET_FLAGS}" \
    pip install --no-build-isolation .
popd >/dev/null

# Verify the native module exists (load order matters: import torch first)
python - <<'EOF'
import torch
import flash_attn_2_cuda  # noqa: F401
print(f"flash_attn_2_cuda native module OK: {flash_attn_2_cuda.__file__}")
EOF
ok "flash-attn native CK build complete"

# -----------------------------------------------------------------------------
# 5. Build exllamav3 extension
# -----------------------------------------------------------------------------
say "Building exllamav3 C++/HIP extension"
GPU_ARCHS="$GPU_ARCHS" \
MAX_JOBS="$MAX_JOBS" \
HIPCC_COMPILE_FLAGS_APPEND="-parallel-jobs=${HIPCC_PARALLEL_JOBS} ${QUIET_FLAGS}" \
    python setup.py build_ext --inplace
ok "exllamav3 extension built"

# -----------------------------------------------------------------------------
# 6. Smoke test
# -----------------------------------------------------------------------------
if [[ "${SKIP_SMOKE_TEST:-0}" == "1" ]]; then
    warn "SKIP_SMOKE_TEST=1, skipping verification"
else
    say "Running flash-attn vs torch SDPA correctness check"
    python - <<'EOF'
import torch, torch.nn.functional as F
from flash_attn import flash_attn_func
torch.manual_seed(0)
q = torch.randn(1, 512, 32, 128, device='cuda', dtype=torch.float16)
k = torch.randn(1, 512,  8, 128, device='cuda', dtype=torch.float16)
v = torch.randn(1, 512,  8, 128, device='cuda', dtype=torch.float16)
out_fa = flash_attn_func(q, k, v, causal=True)
qq = q.transpose(1, 2)
kk = k.repeat_interleave(4, dim=2).transpose(1, 2)
vv = v.repeat_interleave(4, dim=2).transpose(1, 2)
out_ref = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True).transpose(1, 2)
diff = (out_fa - out_ref).abs().max().item()
assert not torch.isnan(out_fa).any(), "FA produced NaN"
assert diff < 0.05, f"FA disagrees with SDPA: max_diff={diff}"
print(f"flash-attn vs SDPA max_diff={diff:.5f} (fp16 noise floor) ✓")
EOF
    ok "Smoke test passed"
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo
ok "All done."
echo "Activate the venv with:  source $VENV_DIR/bin/activate"
echo "Try a chat with:         python examples/chat.py -m <path-to-exl3-model> -mode <prompt-format>"
