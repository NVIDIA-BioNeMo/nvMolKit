#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build one matrix pair for build_full_matrix.sh.
#
# Usage: build_one_wheel.sh <rdkit_version> <python_version>

set -uo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <rdkit_version> <python_version>" >&2
    exit 2
fi

rdkit=$1
py=$2

: "${REPO:?REPO must be set}"
: "${WHEELHOUSE:?WHEELHOUSE must be set}"
: "${LOG_DIR:?LOG_DIR must be set}"
: "${WORKTREE_ROOT:?WORKTREE_ROOT must be set}"
: "${BUILD_COMMIT:?BUILD_COMMIT must be set}"
: "${NVMOLKIT_MATRIX_CACHE_ROOT:?NVMOLKIT_MATRIX_CACHE_ROOT must be set}"
: "${TIMINGS_TSV:?TIMINGS_TSV must be set}"

pyTag=${py//./}
outDir=$WHEELHOUSE/rdkit${rdkit}/py${py}
logFile=$LOG_DIR/rdkit${rdkit}_py${py}.log
wt=$WORKTREE_ROOT/wt_rdkit${rdkit}_py${py}

cleanup_worktree() {
    # Limit recursive cleanup to this worker's path shape.
    case "$wt" in
        "$WORKTREE_ROOT"/wt_rdkit*_py*) ;;
        *)
            echo "Error: refusing to clean unexpected worktree path: $wt" >&2
            return 1
            ;;
    esac
    git -C "$REPO" worktree remove --force "$wt" 2>/dev/null || true
    if [ -e "$wt" ]; then
        rm -rf -- "$wt"
    fi
}

mkdir -p "$outDir"

startEpoch=$(date +%s)
startedAt=$(date '+%Y-%m-%d %H:%M:%S')

format_hms() {
    local secs=$1
    printf '%dh%02dm%02ds' $((secs / 3600)) $(((secs % 3600) / 60)) $((secs % 60))
}

if compgen -G "$outDir/*.whl" > /dev/null; then
    echo "[skip] rdkit=$rdkit py=$py (wheel already at $outDir)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$rdkit" "$py" "skip" "$startedAt" "$startedAt" 0 >> "$TIMINGS_TSV"
    exit 0
fi

echo "[start $startedAt] rdkit=$rdkit py=$py worktree=$wt"

cleanup_worktree
trap cleanup_worktree EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git -C "$REPO" worktree add --detach "$wt" "$BUILD_COMMIT" > "$logFile" 2>&1

{
    echo "=== nvmolkit wheel build ==="
    echo "    rdkit=$rdkit py=$py"
    echo "    commit=$BUILD_COMMIT"
    echo "    started_at=$startedAt"
    echo "    worktree=$wt"
    echo "============================"
} >> "$logFile"

(
    set -e
    cd "$wt"
    export CIBW_BUILD=cp${pyTag}-manylinux_x86_64
    # Conan caches are per-pair because concurrent exports race.
    export NVMOLKIT_CACHE_ROOT=$NVMOLKIT_MATRIX_CACHE_ROOT/py${py}
    export NVMOLKIT_CONAN_CACHE_ROOT=$NVMOLKIT_MATRIX_CACHE_ROOT/conan2/rdkit${rdkit}_py${py}
    bash admin/deploy/build_pip_wheels.sh "$rdkit" "$outDir"
) >> "$logFile" 2>&1
rc=$?

endEpoch=$(date +%s)
endedAt=$(date '+%Y-%m-%d %H:%M:%S')
elapsed=$((endEpoch - startEpoch))
elapsedHms=$(format_hms "$elapsed")

{
    echo
    echo "=== build finished ==="
    echo "    rdkit=$rdkit py=$py rc=$rc"
    echo "    ended_at=$endedAt"
    echo "    elapsed=${elapsed}s ($elapsedHms)"
    echo "======================"
} >> "$logFile"

cleanup_worktree
trap - EXIT INT TERM

if [ "$rc" -eq 0 ]; then
    status=ok
    printf '[ok %s] rdkit=%s py=%s elapsed=%s\n' \
        "$endedAt" "$rdkit" "$py" "$elapsedHms"
else
    status=fail
    printf '[FAIL %s] rdkit=%s py=%s rc=%d elapsed=%s (see %s)\n' \
        "$endedAt" "$rdkit" "$py" "$rc" "$elapsedHms" "$logFile"
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$rdkit" "$py" "$status" "$startedAt" "$endedAt" "$elapsed" >> "$TIMINGS_TSV"

exit "$rc"
