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
#                   (e.g., 2025.9.1, 2025.9.6, 2026.3.5)
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

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"

if [ -z "${CIBW_MANYLINUX_X86_64_IMAGE:-}" ]; then
    echo "Error: CIBW_MANYLINUX_X86_64_IMAGE must point at the manylinux+CUDA image" >&2
    echo "Build it with:" >&2
    echo "  docker build -f admin/container/manylinux_2_28_cuda12.Dockerfile -t <image> ." >&2
    exit 1
fi

# Require cibuildwheel to be available on PATH (typically via an activated
# conda env). Doing `pip install --upgrade pip / cibuildwheel` here races
# fatally when build_full_matrix.sh fans this script out across parallel
# workers sharing the same python env.
if ! command -v cibuildwheel >/dev/null 2>&1; then
    echo "Error: cibuildwheel not found on PATH." >&2
    echo "       Activate a conda env that provides it (e.g." >&2
    echo "       'conda activate nvmolkit_pip_build') before running." >&2
    exit 1
fi

# Persistent caches across cibuildwheel invocations:
#   - rdkit_recipe : full reproduced rdkit + boost install tree (~30-50 min build)
#   - conan2       : conan package cache (saves boost rebuild on partial failure retry)
#   - pip          : pip download cache (numpy, pillow, conan source dist, etc.)
# All keyed on host $HOME so they survive reboots, unlike anything under /tmp.
#
# The conan2 cache is not safe for concurrent `conan export` of the same
# recipe ref from multiple processes (races on .conan2/p/<hash>/s during
# population). NVMOLKIT_CONAN_CACHE_ROOT lets a parallel driver point each
# concurrent build at its own conan cache; default keeps single-build behavior.
NVMOLKIT_CACHE_ROOT="${NVMOLKIT_CACHE_ROOT:-${HOME}/.cache/nvmolkit}"
NVMOLKIT_CONAN_CACHE_ROOT="${NVMOLKIT_CONAN_CACHE_ROOT:-${NVMOLKIT_CACHE_ROOT}/conan2}"
mkdir -p \
    "${NVMOLKIT_CACHE_ROOT}/rdkit_recipe" \
    "${NVMOLKIT_CONAN_CACHE_ROOT}" \
    "${NVMOLKIT_CACHE_ROOT}/pip"

# Configure cibuildwheel's container engine at runtime: --network=host plus
# the bind-mounts for the caches above. cibuildwheel's TOML config cannot
# interpolate $HOME so we set this here.
CIBW_CONTAINER_ENGINE="docker; create_args: --network=host \
-v ${NVMOLKIT_CACHE_ROOT}/rdkit_recipe:/tmp/rdkit_recipe \
-v ${NVMOLKIT_CONAN_CACHE_ROOT}:/root/.conan2 \
-v ${NVMOLKIT_CACHE_ROOT}/pip:/root/.cache/pip"

# Pin the rdkit dependency in the wheel's Requires-Dist to the exact RDKit
# version this build links against. pyproject.toml keeps "rdkit" unpinned in
# version control so the dependency list stays readable for source/conda
# installs; we mutate the marker line below for the duration of the
# cibuildwheel run and restore it on exit (success, error, or signal).
if ! grep -q '# nvmolkit-rdkit-pin:' pyproject.toml; then
    echo "Error: pyproject.toml is missing the nvmolkit-rdkit-pin marker" >&2
    exit 1
fi
cp pyproject.toml pyproject.toml.bak
restore_pyproject() {
    mv pyproject.toml.bak pyproject.toml
}
trap restore_pyproject EXIT
sed -i "s|^.*# nvmolkit-rdkit-pin:.*\$|    \"rdkit==${RDKIT_VERSION}\",  # nvmolkit-rdkit-pin: pinned to RDKIT_VERSION by admin/deploy/build_pip_wheels.sh during cibuildwheel|" pyproject.toml

RDKIT_VERSION="${RDKIT_VERSION}" \
    CIBW_CONTAINER_ENGINE="${CIBW_CONTAINER_ENGINE}" \
    cibuildwheel --platform linux --output-dir "${OUTPUT_DIR}"

ls -la "${OUTPUT_DIR}"
