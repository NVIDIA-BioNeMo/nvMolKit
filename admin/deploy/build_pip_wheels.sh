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
#
# Drive cibuildwheel to produce nvMolKit manylinux_2_28 x86_64 wheels for the
# Python interpreters declared in pyproject.toml's [tool.cibuildwheel].build
# list. CIBW_BUILD on the command line can narrow the matrix.
#
# Usage:
#   ./build_pip_wheels.sh <rdkit_version> [output_dir]
#
# Arguments:
#   rdkit_version : entry in admin/distribute/rdkit_build_matrix.yaml
#                   (e.g., 2025.3.6, 2025.9.6, 2026.3.1)
#   output_dir    : where wheels land (default: ./wheelhouse)
#
# The pyproject.toml [tool.cibuildwheel.linux].before-build hook runs
# admin/distribute/cibuildwheel_before_build.sh inside the manylinux+CUDA
# container, which reproduces the rdkit-pypi build, pip-installs rdkit, and
# stages headers/libs at /tmp/nvmolkit_pip_inputs/. setup.py picks these up
# via NVMOLKIT_BUILD_AGAINST_PIP_* env vars set by [...environment].
#
# The container image (manylinux_2_28 + CUDA toolkit) must be available; build
# it from admin/container/manylinux_2_28_cuda12.Dockerfile and pass its tag
# via the CIBW_MANYLINUX_X86_64_IMAGE env var.

set -euxo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <rdkit_version> [output_dir]" >&2
    exit 1
fi

RDKIT_VERSION=$1
OUTPUT_DIR=${2:-wheelhouse}

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "${REPO_ROOT}"

if [ -z "${CIBW_MANYLINUX_X86_64_IMAGE:-}" ]; then
    echo "Error: CIBW_MANYLINUX_X86_64_IMAGE must point at the manylinux+CUDA image" >&2
    echo "Build it with:" >&2
    echo "  docker build -f admin/container/manylinux_2_28_cuda12.Dockerfile -t <image> ." >&2
    exit 1
fi

python -m pip install --upgrade pip
python -m pip install 'cibuildwheel>=2.16'

# Persistent caches across cibuildwheel invocations:
#   - rdkit_recipe : full reproduced rdkit + boost install tree (~30-50 min build)
#   - conan2       : conan package cache (saves boost rebuild on partial failure retry)
#   - pip          : pip download cache (numpy, pillow, conan source dist, etc.)
# All keyed on host $HOME so they survive reboots, unlike anything under /tmp.
NVMOLKIT_CACHE_ROOT="${NVMOLKIT_CACHE_ROOT:-${HOME}/.cache/nvmolkit}"
mkdir -p \
    "${NVMOLKIT_CACHE_ROOT}/rdkit_recipe" \
    "${NVMOLKIT_CACHE_ROOT}/conan2" \
    "${NVMOLKIT_CACHE_ROOT}/pip"

# Configure cibuildwheel's container engine at runtime: --network=host plus
# the bind-mounts for the caches above. cibuildwheel's TOML config cannot
# interpolate $HOME so we set this here.
CIBW_CONTAINER_ENGINE="docker; create_args: --network=host \
-v ${NVMOLKIT_CACHE_ROOT}/rdkit_recipe:/tmp/rdkit_recipe \
-v ${NVMOLKIT_CACHE_ROOT}/conan2:/root/.conan2 \
-v ${NVMOLKIT_CACHE_ROOT}/pip:/root/.cache/pip"

RDKIT_VERSION="${RDKIT_VERSION}" \
    CIBW_CONTAINER_ENGINE="${CIBW_CONTAINER_ENGINE}" \
    cibuildwheel --platform linux --output-dir "${OUTPUT_DIR}"

ls -la "${OUTPUT_DIR}"
