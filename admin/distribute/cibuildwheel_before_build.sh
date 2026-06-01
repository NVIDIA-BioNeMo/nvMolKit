#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# cibuildwheel "before-build" hook for the nvMolKit pip pipeline. Runs once
# per (python_version, wheel) inside the manylinux build container. Performs:
#
#   1. Reproduce the rdkit-pypi build for the matching rdkit-pypi tag and the
#      currently active CPython, producing rdkit headers + libs and boost
#      headers + libs.
#   2. pip install the matching rdkit wheel so its rdkit.libs/ directory is
#      available to link against (these are the auditwheel-rewritten SONAMEs
#      that the user will see at runtime).
#   3. Stage rdkit headers, boost headers, and rdkit pip libs at stable paths
#      under /tmp/nvmolkit_pip_inputs/ that pyproject.toml's
#      [tool.cibuildwheel.linux].environment block points the build at.
#
# Inputs:
#   $1            : project directory (passed by cibuildwheel as {project})
#   RDKIT_VERSION : an entry in admin/distribute/rdkit_build_matrix.yaml
#                   (passed via cibuildwheel's environment-pass)

set -euxo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <project_dir>" >&2
    exit 1
fi
PROJECT_DIR=$1
: "${RDKIT_VERSION:?RDKIT_VERSION must be set in the cibuildwheel environment}"

PY_VER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

RDKIT_PYPI_TAG=$(python "${PROJECT_DIR}/admin/distribute/lookup_rdkit_pypi_tag.py" \
                 "${RDKIT_VERSION}" "${PY_VER}")

RECIPE_OUT="/tmp/rdkit_recipe/${RDKIT_VERSION}/py${PY_VER}"
"${PROJECT_DIR}/admin/distribute/build_rdkit_recipe.sh" \
    "${RDKIT_PYPI_TAG}" "${PY_VER}" "${RECIPE_OUT}"

python -m pip install --upgrade pip
python -m pip install "rdkit==${RDKIT_VERSION}"

PIP_RDKIT_LIBDIR=$(python -c 'import rdkit, pathlib, sys
site = pathlib.Path(rdkit.__file__).resolve().parents[1]
libs = site / "rdkit.libs"
if not libs.exists():
    print(f"rdkit.libs not found at {libs}", file=sys.stderr)
    sys.exit(1)
print(libs)')

# Stable paths consumed via [tool.cibuildwheel.linux].environment in
# pyproject.toml. Recreate per-build to avoid stale links from prior wheels.
STABLE=/tmp/nvmolkit_pip_inputs
rm -rf "${STABLE}"
mkdir -p "${STABLE}"
ln -s "${PIP_RDKIT_LIBDIR}" "${STABLE}/rdkit_libs"
ln -s "${RECIPE_OUT}/install/rdkit_install/include/rdkit" "${STABLE}/rdkit_include"
ln -s "${RECIPE_OUT}/install/boost/include" "${STABLE}/boost_include"

ls -la "${STABLE}"
