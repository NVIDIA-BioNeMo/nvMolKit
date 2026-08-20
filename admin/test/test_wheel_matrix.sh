#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test requested wheel-matrix pairs serially, creating CPython Conda envs as needed.
#
# Usage:
#   bash admin/test/test_wheel_matrix.sh <wheelhouse_dir> <smoke|full|both> [pairs_file]
# smoke tests every discovered wheel; full uses selected pairs from
# full_test_subset.txt unless a pairs file is supplied; both does both.

set -uo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <wheelhouse_dir> <smoke|full|both> [pairs_file]" >&2
    exit 2
fi

WHEELHOUSE=$(cd "$1" 2>/dev/null && pwd) || {
    echo "Error: wheelhouse_dir '$1' is not a readable directory" >&2
    exit 2
}
MODE=$2
PAIRS_FILE_ARG=${3:-}

case "$MODE" in
    smoke|full|both) ;;
    *) echo "Error: mode must be 'smoke', 'full', or 'both', got '$MODE'" >&2; exit 2 ;;
esac

if [ "$MODE" = "both" ] && [ -n "$PAIRS_FILE_ARG" ]; then
    echo "Error: explicit pairs_file is not supported with mode=both" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
WORKER=$SCRIPT_DIR/test_one_wheel.sh
SUBSET_FILE=$SCRIPT_DIR/full_test_subset.txt

if [ ! -x "$WORKER" ]; then
    echo "Error: per-pair worker not executable at $WORKER" >&2
    exit 2
fi
if [ ! -d "$REPO/nvmolkit/tests" ]; then
    echo "Error: nvmolkit/tests not found under $REPO" >&2
    exit 2
fi

if [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then
    CONDA_BIN=$CONDA_EXE
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN=$(command -v conda)
else
    echo "Error: 'conda' not found on PATH and CONDA_EXE not set." >&2
    echo "       This script needs conda to provide cpython interpreters" >&2
    echo "       for testing wheels (3.11, 3.12, 3.13, 3.14)." >&2
    exit 2
fi

CONDA_BASE=$("$CONDA_BIN" info --base 2>/dev/null) || {
    echo "Error: '$CONDA_BIN info --base' failed" >&2
    exit 2
}
ENVS_ROOT=$CONDA_BASE/envs

IFACE_ENV_PREFIX=${IFACE_ENV_PREFIX:-nvmolkit_iface_}
VENV_ROOT=${VENV_ROOT:-${TMPDIR:-/tmp}/nvmolkit_test_venvs}
TEST_LOG_DIR=$WHEELHOUSE/test_logs
TIMINGS_TSV=${TIMINGS_TSV:-$TEST_LOG_DIR/timings.tsv}

mkdir -p "$TEST_LOG_DIR" "$VENV_ROOT"
if [ ! -f "$TIMINGS_TSV" ]; then
    printf 'rdkit\tpy\tmode\tstatus\tstarted_at\tended_at\telapsed_sec\n' > "$TIMINGS_TSV"
fi

export REPO WHEELHOUSE TEST_LOG_DIR VENV_ROOT IFACE_ENV_PREFIX TIMINGS_TSV
export NVMOLKIT_CONDA_ENVS_ROOT=$ENVS_ROOT

ensure_iface_env() {
    local pyver=$1
    local envName=${IFACE_ENV_PREFIX}py${pyver}
    local envPython=$ENVS_ROOT/$envName/bin/python
    if [ -x "$envPython" ]; then
        return 0
    fi
    echo "Creating interpreter env $envName (python=$pyver)..."
    "$CONDA_BIN" create -y -n "$envName" -c conda-forge "python=$pyver" >&2 || {
        echo "Error: failed to create conda env $envName" >&2
        return 1
    }
    if [ ! -x "$envPython" ]; then
        echo "Error: conda env $envName created but python not at $envPython" >&2
        return 1
    fi
}

discover_smoke_pairs() {
    local out=$1
    : > "$out"
    shopt -s nullglob
    local wheelDir
    for wheelDir in "$WHEELHOUSE"/rdkit*/py*; do
        local rdkit py
        rdkit=$(basename "$(dirname "$wheelDir")")
        rdkit=${rdkit#rdkit}
        py=$(basename "$wheelDir")
        py=${py#py}
        local matches=("$wheelDir"/nvmolkit-*.whl)
        if [ ${#matches[@]} -gt 0 ]; then
            echo "$rdkit $py" >> "$out"
        fi
    done
    shopt -u nullglob
}

drive_one_mode() {
    local mode=$1
    local pairsFile=$2

    local effectivePairs
    effectivePairs=$(mktemp)
    grep -vE '^[[:space:]]*(#|$)' "$pairsFile" > "$effectivePairs" || true

    local numPairs runStartedDisplay
    numPairs=$(wc -l < "$effectivePairs")
    runStartedDisplay=$(date '+%Y-%m-%d %H:%M:%S %Z')

    echo
    echo "=========================================="
    echo "Mode: $mode"
    echo "Pairs file: $pairsFile ($numPairs pairs)"
    echo "Run started: $runStartedDisplay"
    echo "Wheelhouse: $WHEELHOUSE"
    echo "Test logs:  $TEST_LOG_DIR"
    echo "Timings:    $TIMINGS_TSV"
    echo "=========================================="

    local pyversNeeded
    pyversNeeded=$(awk '{print $2}' "$effectivePairs" | sort -u)
    local pyver
    for pyver in $pyversNeeded; do
        ensure_iface_env "$pyver" || {
            rm -f "$effectivePairs"
            return 1
        }
    done

    local start end wall
    start=$(date +%s)
    local fail_count=0 ok_count=0
    declare -a failed_pairs=()

    while IFS= read -r line; do
        [ -z "$line" ] && continue
        local rdkit py
        rdkit=$(echo "$line" | awk '{print $1}')
        py=$(echo "$line" | awk '{print $2}')
        if [ -z "$rdkit" ] || [ -z "$py" ]; then
            echo "Warning: malformed line: '$line', skipping" >&2
            continue
        fi

        "$WORKER" "$rdkit" "$py" "$mode"
        local rc=$?

        case $rc in
            0)
                ok_count=$((ok_count + 1))
                ;;
            *)
                fail_count=$((fail_count + 1))
                failed_pairs+=("rdkit=$rdkit py=$py")
                ;;
        esac
    done < "$effectivePairs"

    end=$(date +%s)
    wall=$((end - start))
    rm -f "$effectivePairs"

    echo
    echo "Mode '$mode' finished. Wall time: ${wall}s ($(printf '%dh%02dm%02ds' $((wall/3600)) $(((wall%3600)/60)) $((wall%60))))"
    echo "Results: ok=$ok_count fail=$fail_count total=$numPairs"
    if [ ${#failed_pairs[@]} -gt 0 ]; then
        echo "Failed:"
        local f
        for f in "${failed_pairs[@]}"; do
            echo "  $f"
        done
    fi

    return "$fail_count"
}

SMOKE_PAIRS_FILE=
FULL_PAIRS_FILE=
case "$MODE" in
    smoke)
        if [ -n "$PAIRS_FILE_ARG" ]; then
            [ -f "$PAIRS_FILE_ARG" ] || { echo "Error: pairs file not found: $PAIRS_FILE_ARG" >&2; exit 2; }
            SMOKE_PAIRS_FILE=$PAIRS_FILE_ARG
        fi
        ;;
    full)
        if [ -n "$PAIRS_FILE_ARG" ]; then
            [ -f "$PAIRS_FILE_ARG" ] || { echo "Error: pairs file not found: $PAIRS_FILE_ARG" >&2; exit 2; }
            FULL_PAIRS_FILE=$PAIRS_FILE_ARG
        else
            FULL_PAIRS_FILE=$SUBSET_FILE
        fi
        ;;
    both)
        FULL_PAIRS_FILE=$SUBSET_FILE
        ;;
esac

if { [ "$MODE" = "smoke" ] || [ "$MODE" = "both" ]; } && [ -z "$SMOKE_PAIRS_FILE" ]; then
    SMOKE_PAIRS_FILE=$(mktemp)
    # shellcheck disable=SC2064  # interpolate path now so the trap fires correctly
    trap "rm -f '$SMOKE_PAIRS_FILE'" EXIT
    discover_smoke_pairs "$SMOKE_PAIRS_FILE"
    if [ ! -s "$SMOKE_PAIRS_FILE" ]; then
        echo "Error: no wheels discovered under $WHEELHOUSE/rdkit*/py*/" >&2
        exit 2
    fi
fi

if { [ "$MODE" = "full" ] || [ "$MODE" = "both" ]; } && [ ! -f "$FULL_PAIRS_FILE" ]; then
    echo "Error: full pairs file not found: $FULL_PAIRS_FILE" >&2
    exit 2
fi

overall_fail=0
case "$MODE" in
    smoke)
        drive_one_mode smoke "$SMOKE_PAIRS_FILE"
        overall_fail=$?
        ;;
    full)
        drive_one_mode full "$FULL_PAIRS_FILE"
        overall_fail=$?
        ;;
    both)
        drive_one_mode smoke "$SMOKE_PAIRS_FILE"
        smoke_fail=$?
        drive_one_mode full "$FULL_PAIRS_FILE"
        full_fail=$?
        overall_fail=$((smoke_fail + full_fail))
        ;;
esac

echo
if [ "$overall_fail" -eq 0 ]; then
    echo "All requested tests passed."
    exit 0
else
    echo "$overall_fail pair(s) failed across all modes. See per-pair logs in $TEST_LOG_DIR."
    exit 1
fi
