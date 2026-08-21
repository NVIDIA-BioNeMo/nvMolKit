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

# Generate one PEP 503 index per rdkit<X.Y.Z> variant directory.
# Usage: generate_simple_index.sh <variants_root> <out_dir> <release_url_base>

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <variants_root> <out_dir> <release_url_base>" >&2
    exit 1
fi

VARIANTS_ROOT="$1"
OUT_DIR="$2"
RELEASE_URL_BASE="$3"

count=0
for variant_dir in "${VARIANTS_ROOT}"/rdkit*/; do
    [ -d "${variant_dir}" ] || continue
    variant_name=$(basename "${variant_dir}")
    rdkit_ver="${variant_name#rdkit}"

    out="${OUT_DIR}/rdkit${rdkit_ver}/simple/nvmolkit"
    mkdir -p "${out}"

    {
        echo '<!DOCTYPE html>'
        echo '<html><head><meta name="pypi:repository-version" content="1.0"></head><body>'
        for whl in "${variant_dir}"*.whl; do
            [ -f "${whl}" ] || continue
            fname=$(basename "${whl}")
            hash=$(sha256sum "${whl}" | cut -d' ' -f1)
            # PEP 440 local versions contain '+', which must be percent-encoded in URLs.
            url_fname="${fname//+/%2B}"
            echo "<a href=\"${RELEASE_URL_BASE}/${url_fname}#sha256=${hash}\">${fname}</a><br>"
        done
        echo '</body></html>'
    } > "${out}/index.html"

    count=$((count + 1))
done

echo "Generated ${count} simple-index page(s) under ${OUT_DIR}"
