#!/usr/bin/env python3
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

"""Enforce project-rooted #include "..." paths for nvMolKit C++/CUDA sources.

Two modes:
  rewrite (default): rewrites non-conforming includes in place.
  --check:           reports violations and exits non-zero, no edits.

Project policy:
  Every project-local #include "..." must use the path relative to the
  repository root, e.g. #include "src/forcefields/mmff.h". Bare neighbor
  includes (e.g. #include "mmff.h" referring to a same-dir header) and
  upward-relative includes (e.g. #include "../foo/bar.h") are violations.
  System / third-party includes using <...> are ignored.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

# Directories scanned for source files. Anything outside this list is ignored.
SCAN_DIRS = ("src", "rdkit_extensions", "tests", "benchmarks", "nvmolkit")

# Top-level directories considered "project-rooted" for the purposes of
# rewriting bare or relative includes. Order matters when a header name is
# unique only by its containing directory.
ROOT_DIRS = ("src", "rdkit_extensions", "tests", "benchmarks", "nvmolkit")

# File extensions that we touch.
SOURCE_EXTS = (".h", ".hpp", ".cpp", ".cu", ".cuh", ".cxx", ".cc")

# Directory names skipped during walks.
SKIP_DIRS = {".git", "build", "_deps", "__pycache__", ".venv", "venv"}


class Violation:
    __slots__ = ("path", "lineno", "original", "rewritten", "reason")

    def __init__(
        self,
        path: Path,
        lineno: int,
        original: str,
        rewritten: str | None,
        reason: str,
    ) -> None:
        self.path = path
        self.lineno = lineno
        self.original = original
        self.rewritten = rewritten
        self.reason = reason


def iter_source_files(project_root: Path) -> Iterable[Path]:
    for top in SCAN_DIRS:
        base = project_root / top
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                if name.endswith(SOURCE_EXTS):
                    yield Path(dirpath) / name


def build_header_index(project_root: Path) -> dict[str, list[str]]:
    """Map basename -> list of project-rooted paths owning that basename."""
    index: dict[str, list[str]] = {}
    for src in iter_source_files(project_root):
        rel = src.relative_to(project_root).as_posix()
        index.setdefault(src.name, []).append(rel)
    return index


def normalize(rel_posix: str) -> str:
    return os.path.normpath(rel_posix).replace(os.sep, "/")


def resolve_include(
    project_root: Path,
    file_path: Path,
    include_str: str,
    header_index: dict[str, list[str]],
) -> tuple[str | None, str | None]:
    """Return (rewritten_path, reason) where rewritten_path is the
    project-rooted form, or (None, None) if the include is acceptable as-is
    or is not a project header at all.
    """
    if include_str.startswith("/"):
        return None, None

    # Already project-rooted form.
    candidate_root = project_root / include_str
    if candidate_root.is_file():
        # Acceptable iff it sits under one of ROOT_DIRS.
        first = include_str.split("/", 1)[0]
        if first in ROOT_DIRS:
            return None, None

    file_dir = file_path.parent

    # Upward-relative: always a violation.
    if include_str.startswith("../") or "/../" in include_str:
        target = (file_dir / include_str).resolve()
        try:
            target_rel = target.relative_to(project_root.resolve()).as_posix()
        except ValueError:
            return None, "include points outside project root"
        if (project_root / target_rel).is_file():
            return target_rel, "upward-relative include"
        return None, f"upward-relative include with no resolvable target: {include_str}"

    # Bare (no slash) or partial-path include not anchored at project root.
    neighbor = (file_dir / include_str).resolve()
    try:
        neighbor_rel = neighbor.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        neighbor_rel = None

    if neighbor_rel and (project_root / neighbor_rel).is_file():
        return neighbor_rel, "include resolves to project header but is not project-rooted"

    # The include did not resolve to a file via our local search. It may be
    # a partial path that resolves only via another library's include dir;
    # try the basename in our header index.
    base = include_str.rsplit("/", 1)[-1]
    matches = header_index.get(base, [])
    matches = [m for m in matches if m.endswith(include_str)]
    if len(matches) == 1:
        return matches[0], "include resolves via library include dir; rewrite to project-rooted form"
    if len(matches) > 1:
        return None, (
            f"include {include_str!r} is ambiguous; could be any of: "
            + ", ".join(sorted(matches))
        )

    # Not a project header (system or third-party).
    return None, None


_INCLUDE_QUOTE = '#include "'
_INCLUDE_ANGLE = "#include <"


def process_line(
    line: str,
    project_root: Path,
    file_path: Path,
    header_index: dict[str, list[str]],
) -> tuple[str, Violation | None]:
    stripped = line.lstrip()
    if stripped.startswith(_INCLUDE_QUOTE):
        prefix = _INCLUDE_QUOTE
        close_char = '"'
    elif stripped.startswith(_INCLUDE_ANGLE):
        prefix = _INCLUDE_ANGLE
        close_char = ">"
    else:
        return line, None
    indent = line[: len(line) - len(stripped)]
    after = stripped[len(prefix):]
    end = after.find(close_char)
    if end < 0:
        return line, None
    include_str = after[:end]
    trailing = after[end + 1:]

    rewritten, reason = resolve_include(
        project_root, file_path, include_str, header_index
    )
    if rewritten is None:
        # Already-correct quoted form, system include, or third-party.
        if close_char == ">" and reason is None:
            # Detect angle-bracket project headers that should be quoted.
            base = include_str.rsplit("/", 1)[-1]
            matches = header_index.get(base, [])
            matches = [m for m in matches if m.endswith(include_str)]
            if len(matches) == 1:
                target = normalize(matches[0])
                new_line = f'{indent}#include "{target}"{trailing}'
                if not new_line.endswith("\n") and line.endswith("\n"):
                    new_line += "\n"
                violation = Violation(
                    file_path,
                    0,
                    f"<{include_str}>",
                    target,
                    "angle-bracket project header; should be quoted and project-rooted",
                )
                return new_line, violation
        return line, None

    rewritten = normalize(rewritten)
    new_line = f'{indent}#include "{rewritten}"{trailing}'
    if not new_line.endswith("\n") and line.endswith("\n"):
        new_line += "\n"
    violation = Violation(file_path, 0, include_str, rewritten, reason or "")
    return new_line, violation


def process_file(
    file_path: Path,
    project_root: Path,
    header_index: dict[str, list[str]],
    check_only: bool,
) -> list[Violation]:
    text = file_path.read_text()
    lines = text.splitlines(keepends=True)
    violations: list[Violation] = []
    changed = False
    for i, line in enumerate(lines, start=1):
        new_line, violation = process_line(
            line, project_root, file_path, header_index
        )
        if violation is not None:
            violation.lineno = i
            violations.append(violation)
            if new_line != line:
                lines[i - 1] = new_line
                changed = True
    if changed and not check_only:
        file_path.write_text("".join(lines))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--check",
        action="store_true",
        help="report violations and exit non-zero; do not modify files",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="project root (defaults to git toplevel of the script)",
    )
    args = parser.parse_args()

    if args.root is not None:
        project_root = args.root.resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent

    header_index = build_header_index(project_root)

    all_violations: list[Violation] = []
    for src in iter_source_files(project_root):
        all_violations.extend(
            process_file(src, project_root, header_index, args.check)
        )

    if not all_violations:
        if args.check:
            print("include check: 0 violations")
        else:
            print("include rewrite: nothing to change")
        return 0

    fixable = [v for v in all_violations if v.rewritten is not None]
    unfixable = [v for v in all_violations if v.rewritten is None]

    if args.check:
        for v in all_violations:
            rel = v.path.relative_to(project_root)
            if v.rewritten:
                print(
                    f'{rel}:{v.lineno}: #include "{v.original}" -> '
                    f'"{v.rewritten}" ({v.reason})'
                )
            else:
                print(f"{rel}:{v.lineno}: {v.reason}")
        print(
            f"include check: {len(fixable)} fixable, "
            f"{len(unfixable)} unfixable, {len(all_violations)} total"
        )
        return 1

    for v in fixable:
        rel = v.path.relative_to(project_root)
        print(
            f'{rel}:{v.lineno}: rewrote "{v.original}" -> "{v.rewritten}"'
        )
    for v in unfixable:
        rel = v.path.relative_to(project_root)
        print(f"{rel}:{v.lineno}: WARNING {v.reason}")

    if unfixable:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
