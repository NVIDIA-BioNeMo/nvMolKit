#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -uo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <local_conda_endpoint> <pytest_directory> <log_dir>" >&2
  exit 1
fi

LOCAL_CONDA_ENDPOINT=$1
PYTEST_DIR=$2
LOG_DIR=$3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERIFY_SCRIPT="$SCRIPT_DIR/verify_conda_distributable.sh"

# Optional: use conda default locations unless you export CONDA_PKGS_DIRS / CONDA_ENVS_PATH (e.g. to scratch)
if [[ -n "${CONDA_PKGS_DIRS:-}" ]]; then
  mkdir -p "$CONDA_PKGS_DIRS"
fi
if [[ -n "${CONDA_ENVS_PATH:-}" ]]; then
  mkdir -p "$CONDA_ENVS_PATH"
fi

if [[ ! -x $VERIFY_SCRIPT ]]; then
  if [[ -f $VERIFY_SCRIPT ]]; then
    chmod +x "$VERIFY_SCRIPT"
  else
    echo "verify_conda_distributable.sh not found at '$VERIFY_SCRIPT'" >&2
    exit 1
  fi
fi

mkdir -p "$LOG_DIR"

SUMMARY_LOG="$LOG_DIR/summary.log"
printf 'Verification Summary - %s\n' "$(date)" >"$SUMMARY_LOG"

PYTHON_VERSIONS=("3.10" "3.11" "3.12" "3.13")
RDKIT_VERSIONS=("2025.03.1" "2025.03.2"  "2025.03.3"  "2025.03.4"  "2025.03.5"  "2025.03.6" "2025.09.1" "2025.09.2" "2025.09.3" "2025.09.4" "2025.09.5" "2025.09.6" "2026.03.1")

for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do
  for RDKIT_VERSION in "${RDKIT_VERSIONS[@]}"; do
    PY_LABEL=${PYTHON_VERSION//./}
    RD_LABEL=${RDKIT_VERSION//./}
    LOG_FILE="$LOG_DIR/verify_py${PY_LABEL}_rdkit${RD_LABEL}.log"

    printf 'Testing Python %s, RDKit %s ... ' "$PYTHON_VERSION" "$RDKIT_VERSION" >&2
    if "$VERIFY_SCRIPT" "$LOCAL_CONDA_ENDPOINT" "$PYTEST_DIR" "$RDKIT_VERSION" "$PYTHON_VERSION" >"$LOG_FILE" 2>&1; then
      echo 'PASS' >&2
      printf 'PASS: Python %s, RDKit %s\n' "$PYTHON_VERSION" "$RDKIT_VERSION" >> "$SUMMARY_LOG"
    else
      echo 'FAIL' >&2
      printf 'FAIL: Python %s, RDKit %s\n' "$PYTHON_VERSION" "$RDKIT_VERSION" >> "$SUMMARY_LOG"
    fi
  done
done

printf 'Verification completed. Summary available at %s\n' "$SUMMARY_LOG" >&2
