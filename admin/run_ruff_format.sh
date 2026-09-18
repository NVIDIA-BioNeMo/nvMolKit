#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# Runs Ruff lint fixes and formatting over the codebase.
# By default will modify files in-place. Use -d to do a dry-run.

set -ex

EXPECTED_RUFF_VERSION="0.15.8"
RUFF_CHECK_ARGS=(--fix)
RUFF_FORMAT_ARGS=()
while getopts ":d" opt; do
  case ${opt} in
    d )
      RUFF_CHECK_ARGS=()
      RUFF_FORMAT_ARGS+=(--check)
      ;;
    \? )
      echo "Usage: run_ruff_format.sh [-d]"
      exit 1
      ;;
  esac
done

ROOT_DIR=$(git rev-parse --show-toplevel)

if command -v ruff >/dev/null 2>&1; then
  ACTUAL_RUFF_VERSION=$(ruff --version | awk '{print $2}')
  if [ "$ACTUAL_RUFF_VERSION" != "$EXPECTED_RUFF_VERSION" ]; then
    echo "Warning: expected ruff version $EXPECTED_RUFF_VERSION, found $ACTUAL_RUFF_VERSION. Results may not match CI." >&2
  fi
else
  echo "Error: ruff is not installed; expected version $EXPECTED_RUFF_VERSION." >&2
  exit 1
fi

echo "Running ruff check:"
ruff check "${RUFF_CHECK_ARGS[@]}" "$ROOT_DIR"

echo "Running ruff format:"
ruff format "${RUFF_FORMAT_ARGS[@]}" "$ROOT_DIR"
