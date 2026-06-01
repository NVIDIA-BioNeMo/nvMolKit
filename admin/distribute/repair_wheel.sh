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
# cibuildwheel repair_wheel hook for the nvMolKit pip pipeline.
#
# Two stages:
#   1. auditwheel repair, with --exclude lists targeting:
#        - rdkit's libRDKit* and libboost_* libs (provided at runtime by the
#          user's pip-installed rdkit wheel under <site-packages>/rdkit.libs)
#        - the system libs auditwheel rolled into rdkit.libs during rdkit-pypi's
#          own repair (cairo, fontconfig, freetype, libpng, libpixman, libxcb,
#          libXau, libuuid, libbz2, libquadmath); these have hash-suffixed
#          SONAMEs so the globs allow the suffix.
#        - cuda runtime (libcudart*, libcuda*) provided at runtime by the
#          nvidia-cuda-runtime-cu12 wheel under <site-packages>/nvidia/.
#   2. patchelf RPATH on every nvmolkit/_*.so so the dynamic linker can find
#      those externally-shipped libs at runtime. From <site-packages>/nvmolkit/
#      we add:
#        - $ORIGIN/../rdkit.libs                (rdkit + boost + system libs)
#        - $ORIGIN/../nvidia/cuda_runtime/lib   (libcudart.so.12)
#      and preserve auditwheel's existing $ORIGIN/../nvmolkit.libs entry that
#      points at any libs it actually bundled (e.g. libgomp).
#
# Usage (invoked by cibuildwheel via [tool.cibuildwheel.linux].repair-wheel-command):
#   ./repair_wheel.sh <dest_dir> <wheel>

set -euxo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <dest_dir> <wheel>" >&2
    exit 1
fi

DEST_DIR=$1
WHEEL=$2

# Clear stale wheels from a prior failed invocation that left files in
# DEST_DIR. Without this, the post-auditwheel `ls` could pick up an old
# wheel and silently re-process it instead of the one we just produced.
mkdir -p "${DEST_DIR}"
rm -f "${DEST_DIR}"/*.whl

auditwheel repair \
    --exclude 'libRDKit*' \
    --exclude 'libboost_*' \
    --exclude 'libcudart*' \
    --exclude 'libcuda*' \
    --exclude 'libXau*' \
    --exclude 'libxcb*' \
    --exclude 'libcairo-*' \
    --exclude 'libfontconfig*' \
    --exclude 'libfreetype*' \
    --exclude 'libpixman*' \
    --exclude 'libpng*' \
    --exclude 'libquadmath*' \
    --exclude 'libuuid*' \
    --exclude 'libbz2*' \
    -w "${DEST_DIR}" "${WHEEL}"

# auditwheel produced exactly one repaired wheel for this input.
shopt -s nullglob
REPAIRED_WHEELS=("${DEST_DIR}"/*.whl)
shopt -u nullglob
if [ "${#REPAIRED_WHEELS[@]}" -ne 1 ]; then
    echo "Error: expected exactly one repaired wheel in ${DEST_DIR}, got ${#REPAIRED_WHEELS[@]}: ${REPAIRED_WHEELS[*]}" >&2
    exit 1
fi
REPAIRED_WHEEL=${REPAIRED_WHEELS[0]}
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT

unzip -q "${REPAIRED_WHEEL}" -d "${WORK}"

# Each nvmolkit/_*.so resolves siblings in nvmolkit.libs/ (auditwheel's vendor
# dir), and externally-shipped libs in rdkit.libs/ and nvidia/cuda_runtime/lib
# under the same site-packages root. Use DT_RPATH (--force-rpath) rather than
# DT_RUNPATH so the search applies recursively to second-level deps. The libs
# inside rdkit.libs/ have no rpath of their own and rely on RPATH inheritance
# from the entry-point module to find their rdkit.libs/ siblings - rdkit's
# own python bindings work the same way.
NEW_RPATH='$ORIGIN/../nvmolkit.libs:$ORIGIN/../rdkit.libs:$ORIGIN/../nvidia/cuda_runtime/lib'
find "${WORK}/nvmolkit" -maxdepth 1 -name '_*.so' -type f | while read -r so; do
    patchelf --force-rpath --set-rpath "${NEW_RPATH}" "${so}"
done

# Repack the wheel (preserves original filename).
WHEEL_BASENAME=$(basename "${REPAIRED_WHEEL}")
rm -f "${REPAIRED_WHEEL}"
(cd "${WORK}" && zip -qr "${DEST_DIR}/${WHEEL_BASENAME}" .)

echo "repair_wheel.sh: ${DEST_DIR}/${WHEEL_BASENAME}"
