#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test one wheel for test_wheel_matrix.sh.
# Usage: test_one_wheel.sh <rdkit_version> <python_version> <mode>
#   mode = smoke    -> import + tiny GPU op (admin/test/smoke_check.py)
#          full     -> smoke + full pytest from repo's nvmolkit/tests

set -uo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: $0 <rdkit_version> <python_version> <smoke|full>" >&2
    exit 2
fi

rdkit=$1
py=$2
mode=$3

case "$mode" in
    smoke|full) ;;
    *) echo "Error: mode must be 'smoke' or 'full', got '$mode'" >&2; exit 2 ;;
esac

: "${REPO:?REPO must be set}"
: "${WHEELHOUSE:?WHEELHOUSE must be set}"
: "${TEST_LOG_DIR:?TEST_LOG_DIR must be set}"
: "${VENV_ROOT:?VENV_ROOT must be set}"
: "${IFACE_ENV_PREFIX:?IFACE_ENV_PREFIX must be set}"
: "${NVMOLKIT_CONDA_ENVS_ROOT:?NVMOLKIT_CONDA_ENVS_ROOT must be set}"
: "${TIMINGS_TSV:?TIMINGS_TSV must be set}"

ifaceEnv=${IFACE_ENV_PREFIX}py${py}
ifacePython=$NVMOLKIT_CONDA_ENVS_ROOT/$ifaceEnv/bin/python
if [ ! -x "$ifacePython" ]; then
    echo "Error: interpreter env not found at $ifacePython" >&2
    echo "       Create it with: conda create -y -n ${ifaceEnv} -c conda-forge python=${py}" >&2
    exit 2
fi

wheelDir=$WHEELHOUSE/rdkit${rdkit}/py${py}
shopt -s nullglob
wheelMatches=("$wheelDir"/nvmolkit-*-cp${py//./}-cp${py//./}-*.whl)
shopt -u nullglob
if [ ${#wheelMatches[@]} -eq 0 ]; then
    echo "Error: no matching wheel for requested pair rdkit=$rdkit py=$py at $wheelDir" >&2
    exit 1
fi
if [ ${#wheelMatches[@]} -gt 1 ]; then
    echo "Error: multiple wheels in $wheelDir (${wheelMatches[*]})" >&2
    exit 2
fi
wheel=${wheelMatches[0]}

mkdir -p "$TEST_LOG_DIR" "$VENV_ROOT"
logFile=$TEST_LOG_DIR/rdkit${rdkit}_py${py}_${mode}.log
venv=$VENV_ROOT/rdkit${rdkit}_py${py}
smokeRunDir=$VENV_ROOT/rdkit${rdkit}_py${py}_smoke_run

startEpoch=$(date +%s)
startedAt=$(date '+%Y-%m-%d %H:%M:%S')

format_hms() {
    local secs=$1
    printf '%dh%02dm%02ds' $((secs / 3600)) $(((secs % 3600) / 60)) $((secs % 60))
}

echo "[start $startedAt] rdkit=$rdkit py=$py mode=$mode wheel=$(basename "$wheel")"

{
    echo "=== nvmolkit wheel test ==="
    echo "    rdkit=$rdkit py=$py mode=$mode"
    echo "    started_at=$startedAt"
    echo "    wheel=$wheel"
    echo "    interpreter=$ifacePython ($($ifacePython --version 2>&1))"
    echo "    venv=$venv"
    echo "==========================="
} > "$logFile"

run_step() {
    local label=$1
    shift
    {
        echo
        echo "--- $label ---"
        echo "+ $*"
    } >> "$logFile"
    "$@" >> "$logFile" 2>&1
    local rc=$?
    echo "--- $label exit=$rc ---" >> "$logFile"
    return $rc
}

run_step_in_dir() {
    local label=$1
    local workdir=$2
    shift 2
    {
        echo
        echo "--- $label ---"
        echo "cwd: $workdir"
        echo "+ $*"
    } >> "$logFile"
    (cd "$workdir" && "$@") >> "$logFile" 2>&1
    local rc=$?
    echo "--- $label exit=$rc ---" >> "$logFile"
    return $rc
}

rm -rf "$venv" "$smokeRunDir"
mkdir -p "$smokeRunDir"
if ! run_step "venv-create" "$ifacePython" -m venv "$venv"; then
    echo "[FAIL] rdkit=$rdkit py=$py mode=$mode step=venv-create (see $logFile)"
    rc=1
elif ! run_step "pip-upgrade" "$venv/bin/pip" install --upgrade pip; then
    echo "[FAIL] rdkit=$rdkit py=$py mode=$mode step=pip-upgrade (see $logFile)"
    rc=1
elif ! run_step "install-wheel" "$venv/bin/pip" install \
        "$wheel" "rdkit==${rdkit}"; then
    echo "[FAIL] rdkit=$rdkit py=$py mode=$mode step=install-wheel (see $logFile)"
    rc=1
elif ! run_step_in_dir "smoke-check" "$smokeRunDir" \
        "$venv/bin/python" "$REPO/admin/test/smoke_check.py"; then
    echo "[FAIL] rdkit=$rdkit py=$py mode=$mode step=smoke-check (see $logFile)"
    rc=1
elif [ "$mode" = "full" ]; then
    if ! run_step "install-test-deps" "$venv/bin/pip" install \
            pandas pytest psutil optuna; then
        echo "[FAIL] rdkit=$rdkit py=$py mode=$mode step=install-test-deps (see $logFile)"
        rc=1
    else
        testRoot=$VENV_ROOT/rdkit${rdkit}_py${py}_tests
        testRunDir=$testRoot/run
        rm -rf "$testRoot"
        mkdir -p "$testRoot/nvmolkit" "$testRoot/tests" "$testRunDir"
        cp -a "$REPO/nvmolkit/tests" "$testRoot/nvmolkit/"
        cp -a "$REPO/tests/test_data" "$testRoot/tests/"
        find "$testRoot" -type d -name __pycache__ -prune -exec rm -rf {} +
    fi
    if [ "${rc:-0}" -eq 0 ] && ! run_step_in_dir "pytest" "$testRunDir" "$venv/bin/pytest" \
            "$testRoot/nvmolkit/tests" -k "not long" -v; then
        echo "[FAIL] rdkit=$rdkit py=$py mode=$mode step=pytest (see $logFile)"
        rc=1
    elif [ "${rc:-0}" -eq 0 ]; then
        rc=0
    fi
else
    rc=0
fi

endEpoch=$(date +%s)
endedAt=$(date '+%Y-%m-%d %H:%M:%S')
elapsed=$((endEpoch - startEpoch))
elapsedHms=$(format_hms "$elapsed")

{
    echo
    echo "=== test finished ==="
    echo "    rdkit=$rdkit py=$py mode=$mode rc=$rc"
    echo "    ended_at=$endedAt"
    echo "    elapsed=${elapsed}s ($elapsedHms)"
    echo "====================="
} >> "$logFile"

rm -rf "$venv" "$smokeRunDir"
if [ -n "${testRoot:-}" ]; then
    rm -rf "$testRoot"
fi

if [ "$rc" -eq 0 ]; then
    status=ok
    printf '[ok %s] rdkit=%s py=%s mode=%s elapsed=%s\n' \
        "$endedAt" "$rdkit" "$py" "$mode" "$elapsedHms"
else
    status=fail
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$rdkit" "$py" "$mode" "$status" "$startedAt" "$endedAt" "$elapsed" \
    >> "$TIMINGS_TSV"

exit "$rc"
