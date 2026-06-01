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




# Reproduce the kuelumbus/rdkit-pypi build to produce an rdkit install tree
# (headers + libs) and a boost install tree (headers + libs) that exactly match
# what the corresponding rdkit PyPI wheel was built from.
#
# We need this because:
#   - rdkit's PyPI wheel does not ship headers, so we cannot build nvMolKit
#     against it directly.
#   - Headers from upstream rdkit at the matching tag are *almost* identical,
#     but generated headers (e.g., RDGeneral/export.h) come from the rdkit-pypi
#     CMake configure step. Reproducing that build is the only way to get
#     bit-exact headers that match the published shared libraries.
#
# Usage:
#   ./build_rdkit_recipe.sh <rdkit_pypi_tag> <python_version> <out_dir>
#
# Arguments:
#   rdkit_pypi_tag : kuelumbus/rdkit-pypi git tag (e.g., "2026.03.01"). Look
#                    up from admin/distribute/rdkit_build_matrix.yaml.
#   python_version : CPython minor version, e.g., "3.12". Must be installed
#                    and available as `python<version>`.
#   out_dir        : Output directory. On success, contains:
#                       <out_dir>/install/rdkit_install/include/rdkit/...
#                       <out_dir>/install/rdkit_install/lib/...
#                       <out_dir>/install/boost/include/boost/...
#                       <out_dir>/install/boost/lib/libboost_*.so*
#
# Requirements (assumed available on PATH):
#   git, python<version>, gcc, g++, cmake, ninja, conan>=2.0
#
# On Linux, rdkit-pypi's setup.py copies built shared libraries to
# /usr/local/lib and invokes ldconfig. This requires write access to those
# locations (root in the manylinux build container).

set -euxo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: $0 <rdkit_pypi_tag> <python_version> <out_dir>" >&2
    exit 1
fi

RDKIT_PYPI_TAG=$1
PYTHON_VERSION=$2
OUT_DIR=$(realpath -m "$3")

if [[ ! "${PYTHON_VERSION}" =~ ^3\.(10|11|12|13|14)$ ]]; then
    echo "Error: python_version must be one of 3.10..3.14, got: ${PYTHON_VERSION}" >&2
    exit 1
fi

PY="python${PYTHON_VERSION}"
if ! command -v "${PY}" >/dev/null 2>&1; then
    echo "Error: ${PY} not found on PATH" >&2
    exit 1
fi

# Activate gcc-toolset-14 to match kuelumbus/rdkit-pypi's CI build environment.
# cibuildwheel's manylinux entrypoint normally activates this for us, but when
# this script is invoked directly (e.g., a developer running `docker run`) the
# toolset must be sourced explicitly. Without it, the system gcc-8 is used,
# which lacks the gcc-toolset's libquadmath headers needed by boost::charconv,
# and (worse) produces shared libraries against a different toolchain than the
# rdkit PyPI wheel was built with.
GCC_TOOLSET_ENABLE=/opt/rh/gcc-toolset-14/enable
if [ -f "${GCC_TOOLSET_ENABLE}" ]; then
    # shellcheck disable=SC1090
    source "${GCC_TOOLSET_ENABLE}"
else
    echo "Error: gcc-toolset-14 enable script not found at ${GCC_TOOLSET_ENABLE}." >&2
    echo "       This script must run inside the manylinux_2_28 image used by" >&2
    echo "       admin/container/manylinux_2_28_cuda12.Dockerfile." >&2
    exit 1
fi

# Hard-fail if g++ on PATH isn't the toolset's gcc-14. Catches the case where
# the toolset enable script was sourced but a later PATH override hid it.
GXX_VERSION=$(g++ -dumpfullversion 2>/dev/null || true)
if [[ ! "${GXX_VERSION}" =~ ^14\. ]]; then
    echo "Error: expected g++ from gcc-toolset-14, got version: ${GXX_VERSION}" >&2
    echo "       which g++: $(command -v g++ || echo none)" >&2
    exit 1
fi

INSTALL_DIR="${OUT_DIR}/install"
RDKIT_INSTALL="${INSTALL_DIR}/rdkit_install"
BOOST_INSTALL="${INSTALL_DIR}/boost"

# Idempotency: a complete prior run leaves the canonical export.h in place.
if [ -f "${RDKIT_INSTALL}/include/rdkit/RDGeneral/export.h" ] \
   && [ -d "${BOOST_INSTALL}/include/boost" ]; then
    echo "rdkit recipe build already complete at ${INSTALL_DIR}, skipping"
    exit 0
fi

WORK_DIR="${OUT_DIR}/work"
mkdir -p "${WORK_DIR}"
RDKIT_PYPI_DIR="${WORK_DIR}/rdkit-pypi"

if [ ! -d "${RDKIT_PYPI_DIR}/.git" ]; then
    rm -rf "${RDKIT_PYPI_DIR}"
    git clone --depth 1 -b "${RDKIT_PYPI_TAG}" \
        https://github.com/kuelumbus/rdkit-pypi.git "${RDKIT_PYPI_DIR}"
fi

# Build dependencies for the rdkit-pypi recipe (setuptools, wheel, conan,
# ninja, numpy, pybind11-stubgen, Pillow) are pre-installed in the manylinux
# image (admin/container/manylinux_2_28_cuda12.Dockerfile). We pip-install
# without --upgrade only as a safety net for invocations from images that
# don't pre-bake them.
"${PY}" -m pip install \
    setuptools wheel \
    'conan>=2.0' ninja \
    numpy \
    pybind11-stubgen Pillow

# rdkit-pypi's setup.py keys behavior off CIBW_BUILD. Use a Linux value
# matching the requested Python version so the linux branches are taken.
CIBW_PY_TAG="cp${PYTHON_VERSION/./}"
export CIBW_BUILD="${CIBW_PY_TAG}-manylinux_x86_64"

# Run only the build_ext command. BuildRDKit.run() in their setup.py performs
# conan install, clone rdkit at Release_<tag>, patch CMakeLists.txt, configure,
# build, install, copy libs to /usr/local/lib, ldconfig, and build stubs.
BUILD_TEMP="${RDKIT_PYPI_DIR}/build/temp"
BUILD_LIB="${RDKIT_PYPI_DIR}/build/lib"
mkdir -p "${BUILD_TEMP}" "${BUILD_LIB}"

(
    cd "${RDKIT_PYPI_DIR}"
    "${PY}" setup.py build_ext \
        --build-temp="${BUILD_TEMP}" \
        --build-lib="${BUILD_LIB}"
)

RDKIT_INSTALL_SRC="${BUILD_TEMP}/rdkit_install"
BOOST_FROM_CONAN_SRC="${RDKIT_PYPI_DIR}/conan/direct_deploy/boost"

if [ ! -f "${RDKIT_INSTALL_SRC}/include/rdkit/RDGeneral/export.h" ]; then
    echo "Error: rdkit install tree missing RDGeneral/export.h after build" >&2
    exit 1
fi
if [ ! -d "${BOOST_FROM_CONAN_SRC}/include/boost" ]; then
    echo "Error: boost include tree missing from conan output" >&2
    exit 1
fi

mkdir -p "${INSTALL_DIR}"
rm -rf "${RDKIT_INSTALL}" "${BOOST_INSTALL}"
cp -a "${RDKIT_INSTALL_SRC}" "${RDKIT_INSTALL}"
cp -a "${BOOST_FROM_CONAN_SRC}" "${BOOST_INSTALL}"

cat <<EOF
rdkit recipe build complete:
  rdkit headers: ${RDKIT_INSTALL}/include/rdkit
  rdkit libs:    ${RDKIT_INSTALL}/lib
  boost headers: ${BOOST_INSTALL}/include
  boost libs:    ${BOOST_INSTALL}/lib
EOF
