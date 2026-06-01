#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Look up the kuelumbus/rdkit-pypi git tag for a given pip rdkit version, and
# verify that the requested CPython minor is supported. Reads
# admin/distribute/rdkit_build_matrix.yaml using stdlib regex (no PyYAML
# dependency required).
#
# Usage:
#   python3 lookup_rdkit_pypi_tag.py <rdkit_version> <python_version>
#
# Prints the rdkit-pypi tag to stdout; exits non-zero with an error on stderr
# if the lookup fails.

import argparse
import re
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rdkit_version")
    parser.add_argument("python_version")
    parser.add_argument(
        "--matrix",
        default=str(Path(__file__).resolve().parent / "rdkit_build_matrix.yaml"),
        help="Path to the build matrix YAML.",
    )
    args = parser.parse_args()

    text = Path(args.matrix).read_text()

    block_pat = re.compile(
        r'^"' + re.escape(args.rdkit_version) + r'":\s*\n'
        r'(?:(?:[ \t]+\S.*\n)+)',
        re.MULTILINE,
    )
    block_match = block_pat.search(text)
    if not block_match:
        print(
            f"Error: rdkit version not in matrix: {args.rdkit_version}",
            file=sys.stderr,
        )
        return 1
    block = block_match.group(0)

    tag_match = re.search(r'rdkit_pypi_tag:\s*"([^"]+)"', block)
    pys_match = re.search(r'python_versions:\s*\[([^\]]*)\]', block)
    if not tag_match or not pys_match:
        print(
            f"Error: malformed entry for {args.rdkit_version} in {args.matrix}",
            file=sys.stderr,
        )
        return 1

    pys = [p.strip().strip('"') for p in pys_match.group(1).split(",") if p.strip()]
    if args.python_version not in pys:
        print(
            f"Error: python {args.python_version} not listed for rdkit "
            f"{args.rdkit_version} (allowed: {pys})",
            file=sys.stderr,
        )
        return 1

    print(tag_match.group(1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
