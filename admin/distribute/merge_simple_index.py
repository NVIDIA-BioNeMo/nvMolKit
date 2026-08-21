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
"""Merge generated PEP 503 simple-index pages into an existing Pages tree."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from html.parser import HTMLParser
from pathlib import Path


@dataclass(frozen=True)
class Link:
    href: str
    text: str


class AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[Link] = []
        self._href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href is None:
            return
        self._href = href
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return
        text = "".join(self._text_parts).strip()
        if text:
            self.links.append(Link(href=self._href, text=text))
        self._href = None
        self._text_parts = []


def read_links(path: Path) -> list[Link]:
    parser = AnchorParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.links


def write_index(path: Path, links: list[Link]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "<!DOCTYPE html>",
        '<html><head><meta name="pypi:repository-version" content="1.0"></head><body>',
    ]
    for link in links:
        lines.append(f'<a href="{escape(link.href, quote=True)}">{escape(link.text)}</a><br>')
    lines.append("</body></html>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_index(generated_index: Path, destination_index: Path) -> bool:
    merged_by_filename: dict[str, Link] = {}

    if destination_index.exists():
        for link in read_links(destination_index):
            merged_by_filename[link.text] = link

    for link in read_links(generated_index):
        merged_by_filename[link.text] = link

    merged_links = [merged_by_filename[name] for name in sorted(merged_by_filename)]
    old_text = destination_index.read_text(encoding="utf-8") if destination_index.exists() else None
    write_index(destination_index, merged_links)
    return old_text != destination_index.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge generated rdkit*/simple/nvmolkit/index.html pages into "
            "an existing docs/wheels tree without dropping older wheel links."
        )
    )
    parser.add_argument(
        "generated_root",
        type=Path,
        help="Root produced by generate_simple_index.sh",
    )
    parser.add_argument(
        "destination_root",
        type=Path,
        help="Destination docs/wheels root in the GitHub Pages worktree",
    )
    args = parser.parse_args()

    generated_root = args.generated_root.resolve()
    destination_root = args.destination_root.resolve()
    index_paths = sorted(generated_root.glob("rdkit*/simple/nvmolkit/index.html"))
    if not index_paths:
        raise SystemExit(f"No generated index pages found under {generated_root}")

    changed = 0
    for generated_index in index_paths:
        relpath = generated_index.relative_to(generated_root)
        if merge_index(generated_index, destination_root / relpath):
            changed += 1

    print(
        f"Merged {len(index_paths)} generated simple-index page(s) into {destination_root}; {changed} page(s) changed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
