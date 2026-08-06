# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured result output shared by nvMolKit benchmark drivers."""

import csv
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO


def result_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return keys in first-seen order across heterogeneous result rows."""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def write_csv_rows(
    rows: Sequence[Mapping[str, Any]],
    output: str | Path | None = None,
    *,
    stream: TextIO | None = None,
) -> None:
    """Write benchmark rows as valid CSV to a path and/or text stream.

    Parent directories are created for file output. Heterogeneous rows are
    supported: missing fields are emitted empty and columns retain first-seen
    order. Passing neither ``output`` nor ``stream`` is a no-op.
    """
    if not rows or (output is None and stream is None):
        return

    fieldnames = result_fieldnames(rows)

    def write(destination: TextIO) -> None:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    if stream is not None:
        write(stream)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as destination:
            write(destination)


def print_csv_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Print benchmark rows as CSV to stdout."""
    write_csv_rows(rows, stream=sys.stdout)
