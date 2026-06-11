#!/bin/bash
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


# Enforce project-rooted #include "..." paths across the C++/CUDA tree.
# By default rewrites in place. Use -d for check-only mode (CI/PR gating).

set -e

CHECK=""
while getopts ":d" opt; do
  case ${opt} in
    d )
      CHECK="--check"
      ;;
    \? )
      echo "Usage: run_include_check.sh [-d]"
      exit 1
      ;;
  esac
done

ROOT=$(dirname "$(dirname "$(realpath "$0")")")

python3 "$ROOT/admin/include_check.py" --root "$ROOT" $CHECK
