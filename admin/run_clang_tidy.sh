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


# Runs clang-tidy on host C++ and CUDA translation units.

set -euo pipefail

ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
BUILD_DIR="${CLANG_TIDY_BUILD_DIR:-${ROOT}/clang_tidy_build}"
JOBS="${CLANG_TIDY_JOBS:-4}"
CUDA_ARCHITECTURE="${NVMOLKIT_CLANG_TIDY_CUDA_ARCHITECTURE:-80}"
FIX_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i)
      FIX_ARGS=(-fix -fix-errors)
      JOBS=1
      ;;
    *)
      echo "Usage: run_clang_tidy.sh [-i]" >&2
      exit 1
      ;;
  esac
  shift
done

find_tool() {
  local requested_name="$1"
  local fallback_name="$2"

  command -v "${requested_name}" 2>/dev/null || command -v "${fallback_name}" 2>/dev/null
}

CLANG_TIDY="${CLANG_TIDY_BINARY:-$(find_tool clang-tidy-22 clang-tidy)}"
RUN_CLANG_TIDY="${RUN_CLANG_TIDY_BINARY:-$(find_tool run-clang-tidy-22 run-clang-tidy)}"
CLANGXX="${CLANGXX_BINARY:-$(find_tool clang++-22 clang++)}"

if [[ -z "${CLANG_TIDY}" || -z "${RUN_CLANG_TIDY}" || -z "${CLANGXX}" ]]; then
  echo "clang-tidy, run-clang-tidy, and clang++ are required" >&2
  exit 1
fi

"${CLANG_TIDY}" --version
"${CLANG_TIDY}" --verify-config --config-file="${ROOT}/.clang-tidy"

cmake_args=(
  -S "${ROOT}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_COMPILER="${CLANGXX}"
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURE}"
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  -DNVMOLKIT_BUILD_TESTS=OFF
  -DNVMOLKIT_BUILD_BENCHMARKS=OFF
  -DNVMOLKIT_BUILD_PYTHON_BINDINGS=OFF
)

dependency_prefix="${RDKIT_PATH:-${CONDA_PREFIX:-}}"
if [[ -n "${dependency_prefix}" ]]; then
  cmake_args+=("-DCMAKE_PREFIX_PATH=${dependency_prefix}")
fi

cmake "${cmake_args[@]}"

"${RUN_CLANG_TIDY}" \
  -clang-tidy-binary "${CLANG_TIDY}" \
  -config-file "${ROOT}/.clang-tidy" \
  -p "${BUILD_DIR}" \
  -j "${JOBS}" \
  "${FIX_ARGS[@]}" \
  '.*\.cpp$'

PYTHON="${PYTHON_BINARY:-$(find_tool python3 python)}"
if [[ -z "${PYTHON}" ]]; then
  echo "python3 is required for CUDA analysis" >&2
  exit 1
fi

CUDA_TIDY_BUILD_DIR="${BUILD_DIR}/clang_tidy_cuda"
"${PYTHON}" "${ROOT}/admin/prepare_clang_tidy_cuda_compile_db.py" \
  --input "${BUILD_DIR}/compile_commands.json" \
  --output "${CUDA_TIDY_BUILD_DIR}/compile_commands.json" \
  --compiler "${CLANGXX}" \
  --cuda-path "${CUDA_PATH:-/usr/local/cuda}" \
  --cuda-architecture "${CUDA_ARCHITECTURE}" \
  --dependency-prefix "${dependency_prefix}"

# LLVM 22's device pass cannot parse the current RDKit headers. Its CUDA host
# pass still parses CUDA declarations and templates. Limit reported diagnostics
# to project sources rather than CUB and other dependencies.
"${RUN_CLANG_TIDY}" \
  -clang-tidy-binary "${CLANG_TIDY}" \
  -config-file "${ROOT}/.clang-tidy" \
  -p "${CUDA_TIDY_BUILD_DIR}" \
  -j "${JOBS}" \
  -extra-arg=--cuda-host-only \
  -header-filter "^${ROOT}/(src|rdkit_extensions)/" \
  -line-filter "[{\"name\":\"^${ROOT}/(src|rdkit_extensions)/\"}]" \
  "${FIX_ARGS[@]}" \
  '.*\.cu$'
