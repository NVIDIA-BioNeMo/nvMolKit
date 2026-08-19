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

"""Add a PEP 440 local version to a wheel and update its metadata."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import zipfile


def sha256_b64(data: bytes) -> str:
    """Return PEP 376 RECORD hash entry for the given bytes."""
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def patch_metadata(metadata_bytes: bytes, new_version: str) -> bytes:
    """Replace the Version: line in a METADATA file."""
    out_lines = []
    found = False
    for line in metadata_bytes.decode("utf-8").splitlines(keepends=True):
        if not found and line.startswith("Version:"):
            out_lines.append(f"Version: {new_version}\n")
            found = True
        else:
            out_lines.append(line)
    if not found:
        raise RuntimeError("Version: line not found in METADATA")
    return "".join(out_lines).encode("utf-8")


def patch_sbom(sbom_bytes: bytes, old_version: str, new_version: str, old_filename: str, new_filename: str) -> bytes:
    """Rewrite version/filename references in the auditwheel CycloneDX SBOM."""
    sbom = json.loads(sbom_bytes.decode("utf-8"))

    old_purl = f"pkg:pypi/nvmolkit@{old_version}?file_name={old_filename}"
    new_purl = f"pkg:pypi/nvmolkit@{new_version}?file_name={new_filename}"

    component = sbom.get("metadata", {}).get("component", {})
    if component.get("bom-ref") == old_purl:
        component["bom-ref"] = new_purl
    if component.get("purl") == old_purl:
        component["purl"] = new_purl
    if component.get("version") == old_version:
        component["version"] = new_version

    for dep in sbom.get("dependencies", []):
        if dep.get("ref") == old_purl:
            dep["ref"] = new_purl

    return json.dumps(sbom, indent=2).encode("utf-8")


def retag(input_path: str, output_dir: str, local_version: str) -> str:
    """Produce a retagged wheel under output_dir; return its path."""
    with zipfile.ZipFile(input_path, "r") as zin:
        names = zin.namelist()
        infos = {info.filename: info for info in zin.infolist()}
        entries = {name: zin.read(name) for name in names}

    dist_info_dirs = {n.split("/", 1)[0] for n in names if n.endswith(".dist-info/RECORD")}
    if len(dist_info_dirs) != 1:
        raise RuntimeError(f"expected 1 dist-info dir, found {dist_info_dirs!r}")
    old_dist_info = dist_info_dirs.pop()
    assert old_dist_info.endswith(".dist-info")
    distribution, _, old_version = old_dist_info[: -len(".dist-info")].rpartition("-")
    if not distribution or not old_version:
        raise RuntimeError(f"cannot parse distribution-version from {old_dist_info!r}")
    if "+" in old_version:
        raise RuntimeError(f"wheel {input_path} already has a local version segment")

    new_version = f"{old_version}+{local_version}"
    new_dist_info = f"{distribution}-{new_version}.dist-info"

    input_basename = os.path.basename(input_path)
    parts = input_basename.split("-")
    if parts[0] != distribution or parts[1] != old_version:
        raise RuntimeError(f"filename {input_basename!r} does not match dist-info {old_dist_info!r}")
    parts[1] = new_version
    output_basename = "-".join(parts)

    metadata_path_old = f"{old_dist_info}/METADATA"
    entries[metadata_path_old] = patch_metadata(entries[metadata_path_old], new_version)

    sbom_path_old = f"{old_dist_info}/sboms/auditwheel.cdx.json"
    if sbom_path_old in entries:
        entries[sbom_path_old] = patch_sbom(
            entries[sbom_path_old],
            old_version=old_version,
            new_version=new_version,
            old_filename=input_basename,
            new_filename=output_basename,
        )

    renamed = {}
    for name, data in entries.items():
        if name == old_dist_info or name.startswith(old_dist_info + "/"):
            new_name = new_dist_info + name[len(old_dist_info) :]
        else:
            new_name = name
        renamed[new_name] = data
    entries = renamed
    rename_map = {
        old_name: (new_dist_info + old_name[len(old_dist_info) :])
        if (old_name == old_dist_info or old_name.startswith(old_dist_info + "/"))
        else old_name
        for old_name in names
    }

    record_path = f"{new_dist_info}/RECORD"
    record_lines = []
    for old_name in names:
        new_name = rename_map[old_name]
        if new_name.endswith("/"):
            continue
        if new_name == record_path:
            record_lines.append(f"{new_name},,\n")
        else:
            data = entries[new_name]
            record_lines.append(f"{new_name},{sha256_b64(data)},{len(data)}\n")
    entries[record_path] = "".join(record_lines).encode("utf-8")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_basename)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for old_name in names:
            new_name = rename_map[old_name]
            old_info = infos[old_name]
            new_info = zipfile.ZipInfo(filename=new_name, date_time=old_info.date_time)
            new_info.external_attr = old_info.external_attr
            new_info.compress_type = zipfile.ZIP_DEFLATED if not new_name.endswith("/") else zipfile.ZIP_STORED
            zout.writestr(new_info, entries[new_name])

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_wheel")
    parser.add_argument("output_dir")
    parser.add_argument("local_version", help="PEP 440 local version segment, e.g. rdkit2025.9.6")
    args = parser.parse_args()

    out = retag(args.input_wheel, args.output_dir, args.local_version)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
