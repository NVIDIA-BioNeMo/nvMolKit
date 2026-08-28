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

#!/usr/bin/env python3
"""Pin a wheel's RDKit runtime dependency and refresh RECORD."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path


RDKit_REQUIREMENT_RE = re.compile(
    r"^(Requires-Dist:\s*rdkit)(?:\s*(?:[<>=!~]=?|===).*)?$",
    re.IGNORECASE,
)


def sha256_b64(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def pin_metadata(metadata: bytes, rdkit_version: str) -> tuple[bytes, bool]:
    out_lines: list[str] = []
    changed = False
    found = False
    pinned_line = f"Requires-Dist: rdkit=={rdkit_version}\n"

    for line in metadata.decode("utf-8").splitlines(keepends=True):
        if RDKit_REQUIREMENT_RE.match(line.rstrip("\r\n")):
            found = True
            if line != pinned_line:
                changed = True
            out_lines.append(pinned_line)
        else:
            out_lines.append(line)

    if not found:
        raise RuntimeError("Requires-Dist: rdkit line not found in METADATA")

    return "".join(out_lines).encode("utf-8"), changed


def rewrite_wheel(path: Path, rdkit_version: str, *, check: bool) -> bool:
    with zipfile.ZipFile(path, "r") as zin:
        infos = zin.infolist()
        names = [info.filename for info in infos]
        record_paths = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(record_paths) != 1:
            raise RuntimeError(f"{path}: expected 1 RECORD file, found {record_paths!r}")
        record_path = record_paths[0]
        dist_info = record_path.rsplit("/", 1)[0]
        metadata_path = f"{dist_info}/METADATA"
        if metadata_path not in names:
            raise RuntimeError(f"{path}: missing {metadata_path}")

        metadata, changed = pin_metadata(zin.read(metadata_path), rdkit_version)
        if check:
            if changed:
                raise RuntimeError(f"{path}: rdkit dependency is not pinned to {rdkit_version}")
            return False
        if not changed:
            return False

        entries = {info.filename: zin.read(info.filename) for info in infos}

    entries[metadata_path] = metadata

    record_lines = []
    for info in infos:
        name = info.filename
        if name.endswith("/"):
            continue
        if name == record_path:
            record_lines.append(f"{name},,\n")
        else:
            data = entries[name]
            record_lines.append(f"{name},{sha256_b64(data)},{len(data)}\n")
    entries[record_path] = "".join(record_lines).encode("utf-8")

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for info in infos:
                new_info = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
                new_info.external_attr = info.external_attr
                new_info.compress_type = zipfile.ZIP_STORED if info.filename.endswith("/") else zipfile.ZIP_DEFLATED
                zout.writestr(new_info, entries[info.filename])
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify wheels are already pinned without rewriting them",
    )
    parser.add_argument("rdkit_version", help="Exact RDKit runtime version, e.g. 2026.3.1")
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()

    changed = 0
    failed = 0
    for wheel in args.wheels:
        try:
            rewritten = rewrite_wheel(wheel, args.rdkit_version, check=args.check)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            failed += 1
            continue
        if args.check:
            print(f"ok {wheel}")
        elif rewritten:
            changed += 1
            print(f"pinned {wheel}")
        else:
            print(f"already pinned {wheel}")
    if args.check:
        print(f"Checked {len(args.wheels)} wheel(s); failed {failed}.")
    else:
        print(f"Processed {len(args.wheels)} wheel(s); changed {changed}; failed {failed}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
