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

set -eo pipefail

PYTHON_VERSION="${1:-3.12}"
RDKIT_VERSION="${2:-2025.09.1}"

MINIFORGE_VERSION="25.3.0-3"
MINIFORGE_PREFIX="/usr/local/anaconda"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential git wget ca-certificates

ARCH="$(uname -m)"
MINIFORGE_INSTALLER="Miniforge3-${MINIFORGE_VERSION}-Linux-${ARCH}.sh"
MINIFORGE_BASE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}"

wget -q -nc -P /var/tmp "${MINIFORGE_BASE_URL}/${MINIFORGE_INSTALLER}"
wget -q -nc -P /var/tmp "${MINIFORGE_BASE_URL}/${MINIFORGE_INSTALLER}.sha256"
(cd /var/tmp && sha256sum -c "${MINIFORGE_INSTALLER}.sha256")
bash "/var/tmp/${MINIFORGE_INSTALLER}" -b -p "${MINIFORGE_PREFIX}"
"${MINIFORGE_PREFIX}/bin/conda" init
ln -sf "${MINIFORGE_PREFIX}/etc/profile.d/conda.sh" /etc/profile.d/conda.sh

# shellcheck source=/dev/null
. "${MINIFORGE_PREFIX}/etc/profile.d/conda.sh"
conda activate base

conda config --add channels conda-forge --add channels nvidia
# Keep GCC 13 for CUDA 12.6 compatibility; prebuilt RDKit needs the GCC 14+ runtime.
conda install -q -y \
    mamba \
    "python=${PYTHON_VERSION}" \
    "rdkit=${RDKIT_VERSION}" \
    "gcc=13.*" \
    "gxx=13.*" \
    "libgcc-ng>=14" \
    "libstdcxx-ng>=14" \
    libboost-devel \
    libboost-headers \
    libboost-python-devel \
    librdkit-dev \
    pytest \
    "cmake=3.30.*" \
    eigen

# Prefer Conda's newer runtime when linking and running against prebuilt RDKit.
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/zz_nvmolkit_runtime.sh" <<'EOF'
export LDFLAGS="-L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib${LDFLAGS:+ ${LDFLAGS}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
EOF
