#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build every pair in rdkit_build_matrix.yaml in parallel worktrees.
#
# Usage:
#   bash admin/deploy/build_full_matrix.sh [JOBS] [THREADS_PER_JOB]
#     JOBS              default 8
#     THREADS_PER_JOB   default 2
#
# Environment overrides:
#   CIBW_MANYLINUX_X86_64_IMAGE   default nvmolkit-manylinux-cuda12:local
#   WORKTREE_ROOT                 default $HOME/scratch/nvmolkit_wheels
#   WHEELHOUSE                    default <repo>/wheelhouse
#   NVMOLKIT_MATRIX_CACHE_ROOT    default $HOME/.cache/nvmolkit

set -euo pipefail

JOBS=${1:-8}
THREADS_PER_JOB=${2:-2}

if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: JOBS must be a positive integer, got: $JOBS" >&2
    exit 2
fi
if [[ ! "$THREADS_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: THREADS_PER_JOB must be a positive integer, got: $THREADS_PER_JOB" >&2
    exit 2
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

WHEELHOUSE=${WHEELHOUSE:-$REPO/wheelhouse}
WORKTREE_ROOT=${WORKTREE_ROOT:-$HOME/scratch/nvmolkit_wheels}
LOG_DIR=$WHEELHOUSE/logs
JOB_DIR=$WHEELHOUSE/jobs

mkdir -p "$WHEELHOUSE" "$LOG_DIR" "$JOB_DIR" "$WORKTREE_ROOT"

export CIBW_MANYLINUX_X86_64_IMAGE=${CIBW_MANYLINUX_X86_64_IMAGE:-nvmolkit-manylinux-cuda12:local}
export CMAKE_BUILD_PARALLEL_LEVEL=$THREADS_PER_JOB
export MAKEFLAGS=-j$THREADS_PER_JOB
export CONAN_CPU_COUNT=$THREADS_PER_JOB
export NVMOLKIT_MATRIX_CACHE_ROOT=${NVMOLKIT_MATRIX_CACHE_ROOT:-$HOME/.cache/nvmolkit}

if ! command -v cibuildwheel >/dev/null 2>&1; then
    echo "Error: cibuildwheel not found on PATH." >&2
    echo "       Activate an environment that supplies cibuildwheel first." >&2
    exit 1
fi
if ! IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$CIBW_MANYLINUX_X86_64_IMAGE" 2>/dev/null); then
    echo "Error: container image is not available locally: $CIBW_MANYLINUX_X86_64_IMAGE" >&2
    echo "       Build it as described in admin/distribute/README.md or pull an authenticated tag." >&2
    exit 1
fi

# Detached worktrees cannot include uncommitted changes.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: tracked repository changes are present." >&2
    echo "       Commit or stash them before building the release matrix." >&2
    exit 1
fi
BUILD_COMMIT=$(git rev-parse --verify HEAD)
export BUILD_COMMIT

# Parse the simple matrix shape without adding a PyYAML dependency.
PAIRS_FILE=$JOB_DIR/pairs.txt
awk '
    /^"[0-9]+\.[0-9]+\.[0-9]+":/ {
        gsub(/[":]/, "", $1)
        rdkit = $1
    }
    /^[[:space:]]+python_versions:/ {
        sub(/.*\[/, "")
        sub(/\].*/, "")
        gsub(/[",]/, "")
        for (i = 1; i <= NF; i++) print rdkit, $i
    }
' admin/distribute/rdkit_build_matrix.yaml > "$PAIRS_FILE"

NUM_PAIRS=$(wc -l < "$PAIRS_FILE")
RUN_STARTED_DISPLAY=$(date '+%Y-%m-%d %H:%M:%S %Z')
echo "Run started: $RUN_STARTED_DISPLAY"
echo "Matrix has $NUM_PAIRS (rdkit, python) pairs."
echo "Running $JOBS jobs in parallel, $THREADS_PER_JOB threads each."
echo "Build commit: $BUILD_COMMIT"
echo "Container:    $CIBW_MANYLINUX_X86_64_IMAGE"
echo "Image ID:     $IMAGE_ID"
echo "Wheelhouse: $WHEELHOUSE"
echo "Worktrees:  $WORKTREE_ROOT"
echo "Cache root: $NVMOLKIT_MATRIX_CACHE_ROOT"
echo

TIMINGS_TSV=$JOB_DIR/timings.tsv
if [ ! -f "$TIMINGS_TSV" ]; then
    printf 'rdkit\tpy\tstatus\tstarted_at\tended_at\telapsed_sec\n' > "$TIMINGS_TSV"
fi
export TIMINGS_TSV

WORKER=$REPO/admin/deploy/build_one_wheel.sh
if [ ! -x "$WORKER" ]; then
    echo "Error: worker script not executable at $WORKER" >&2
    exit 1
fi

export REPO WHEELHOUSE LOG_DIR WORKTREE_ROOT TIMINGS_TSV

START=$(date +%s)
set +e
xargs -a "$PAIRS_FILE" -P "$JOBS" -L 1 "$WORKER"
RC=$?
set -e
END=$(date +%s)
WALL=$((END - START))

echo
echo "All jobs finished. Wall time: ${WALL}s ($(printf '%dh%02dm%02ds' $((WALL/3600)) $(((WALL%3600)/60)) $((WALL%60))))"
echo

total=0
ok=0
missing=()
while read -r rdkit py; do
    total=$((total + 1))
    if compgen -G "$WHEELHOUSE/rdkit${rdkit}/py${py}/*.whl" > /dev/null; then
        ok=$((ok + 1))
    else
        missing+=("rdkit=$rdkit py=$py")
    fi
done < "$PAIRS_FILE"

echo "Wheels built: $ok / $total"
if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing:"
    for m in "${missing[@]}"; do
        echo "  $m"
    done
fi

echo
echo "Per-pair TSV: $TIMINGS_TSV"

git -C "$REPO" worktree prune
exit $RC
