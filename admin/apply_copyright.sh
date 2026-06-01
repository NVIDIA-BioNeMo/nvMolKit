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

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
CHECK_ONLY=false
FIX_YEARS=false
ERRORS=0

usage() {
    echo "Usage: apply_copyright.sh [-d] [-f]"
}

while getopts ":df" opt; do
    case "${opt}" in
        d)
            CHECK_ONLY=true
            ;;
        f)
            FIX_YEARS=true
            ;;
        \?)
            usage
            exit 1
            ;;
    esac
done

if [ "${CHECK_ONLY}" = true ] && [ "${FIX_YEARS}" = true ]; then
    usage
    exit 1
fi

report_issue() {
    local message="$1"
    echo "${message}"
    ERRORS=$((ERRORS + 1))
}

find_included_files() {
    find "${REPO_ROOT}" -maxdepth 1 -type f "$@" -print0
    find \
        "${REPO_ROOT}/src" \
        "${REPO_ROOT}/nvmolkit" \
        "${REPO_ROOT}/tests" \
        "${REPO_ROOT}/benchmarks" \
        "${REPO_ROOT}/admin" \
        "${REPO_ROOT}/.github" \
        -type f "$@" -not -path "*/rdkit_extensions/*" -print0
}

get_file_year() {
    local file="$1"
    local relative_path="${file#${REPO_ROOT}/}"
    local commit_year

    commit_year=$(git log --diff-filter=A --follow --format=%ad --date=format:%Y -- "${relative_path}" | tail -n 1 || true)
    if [ -z "${commit_year}" ]; then
        commit_year=$(date +%Y)
    fi

    if [ "${commit_year}" -le 2025 ]; then
        echo "2025"
    else
        echo "2026"
    fi
}

get_existing_header_year() {
    local file="$1"
    local year_match

    year_match=$(sed -nE 's/^.*SPDX-FileCopyrightText: Copyright \(c\) ([0-9]{4}).*$/\1/p' "${file}" | head -n 1)
    if [ -n "${year_match}" ]; then
        echo "${year_match}"
    fi
}

make_header() {
    local comment_prefix="$1"
    local year="$2"

    cat <<EOF
${comment_prefix} SPDX-FileCopyrightText: Copyright (c) ${year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
${comment_prefix} SPDX-License-Identifier: Apache-2.0
${comment_prefix}
${comment_prefix} Licensed under the Apache License, Version 2.0 (the "License");
${comment_prefix} you may not use this file except in compliance with the License.
${comment_prefix} You may obtain a copy of the License at
${comment_prefix}
${comment_prefix} http://www.apache.org/licenses/LICENSE-2.0
${comment_prefix}
${comment_prefix} Unless required by applicable law or agreed to in writing, software
${comment_prefix} distributed under the License is distributed on an "AS IS" BASIS,
${comment_prefix} WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
${comment_prefix} See the License for the specific language governing permissions and
${comment_prefix} limitations under the License.

EOF
}

replace_year() {
    local file="$1"
    local year="$2"

    sed -i -E \
        "0,/SPDX-FileCopyrightText: Copyright \\(c\\) [0-9]{4}/s//SPDX-FileCopyrightText: Copyright (c) ${year}/" \
        "${file}"
}

write_header() {
    local file="$1"
    local comment_prefix="$2"
    local year="$3"
    local preserve_shebang="$4"
    local temp_file
    local first_line

    temp_file=$(mktemp)

    if [ "${preserve_shebang}" = true ] && IFS= read -r first_line < "${file}" && [[ "${first_line}" == '#!'* ]]; then
        printf '%s\n' "${first_line}" > "${temp_file}"
        make_header "${comment_prefix}" "${year}" >> "${temp_file}"
        tail -n +2 "${file}" >> "${temp_file}"
    else
        make_header "${comment_prefix}" "${year}" > "${temp_file}"
        cat "${file}" >> "${temp_file}"
    fi

    mv "${temp_file}" "${file}"
}

process_file() {
    local file="$1"
    local comment_prefix="$2"
    local file_label="$3"
    local preserve_shebang="$4"
    local expected_year
    local existing_year

    expected_year=$(get_file_year "${file}")
    existing_year=$(get_existing_header_year "${file}")

    if [ "${existing_year:-}" = "${expected_year}" ]; then
        return
    fi

    if [ -n "${existing_year:-}" ]; then
        if [ "${CHECK_ONLY}" = true ]; then
            report_issue "Wrong copyright year in ${file_label} file: ${file} (expected ${expected_year})"
            return
        fi

        if [ "${FIX_YEARS}" = true ]; then
            echo "Fixing copyright year in ${file_label} file: ${file}"
            replace_year "${file}" "${expected_year}"
        else
            echo "${file_label} file already has copyright header with a different year: ${file}"
        fi
        return
    fi

    if [ "${CHECK_ONLY}" = true ]; then
        report_issue "Missing copyright header in ${file_label} file: ${file}"
        return
    fi

    echo "Adding copyright header to ${file_label} file: ${file}"
    write_header "${file}" "${comment_prefix}" "${expected_year}" "${preserve_shebang}"
}

while IFS= read -r -d '' file; do
    process_file "${file}" "#" "Python" false
done < <(find_included_files -name "*.py")

while IFS= read -r -d '' file; do
    process_file "${file}" "//" "C++" false
done < <(find_included_files \( -name "*.h" -o -name "*.cpp" -o -name "*.cuh" -o -name "*.cu" \))

while IFS= read -r -d '' file; do
    process_file "${file}" "#" "shell script" true
done < <(find_included_files -name "*.sh")

while IFS= read -r -d '' file; do
    process_file "${file}" "#" "CMake" false
done < <(find_included_files \( -name "CMakeLists.txt" -o -name "*.cmake" \))

while IFS= read -r -d '' file; do
    process_file "${file}" "#" "Dockerfile" false
done < <(find_included_files \( -name "Dockerfile" -o -name "Dockerfile.*" -o -name "*.Dockerfile" \))

while IFS= read -r -d '' file; do
    process_file "${file}" "#" "YAML" false
done < <(find_included_files \( -name "*.yaml" -o -name "*.yml" \))

if [ "${CHECK_ONLY}" = true ] && [ "${ERRORS}" -ne 0 ]; then
    exit 1
fi
