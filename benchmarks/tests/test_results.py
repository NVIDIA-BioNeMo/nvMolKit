# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import io

from bench_utils.results import result_fieldnames, write_csv_rows


def test_result_fieldnames_preserve_first_seen_order():
    rows = [{"method": "a", "time_ms": 1.0}, {"method": "b", "pairs": 10}]
    assert result_fieldnames(rows) == ["method", "time_ms", "pairs"]


def test_write_csv_rows_quotes_values_and_fills_missing_fields():
    rows = [{"method": "a", "input": "path,with,commas"}, {"method": "b", "pairs": 10}]
    stream = io.StringIO()

    write_csv_rows(rows, stream=stream)

    assert list(csv.DictReader(io.StringIO(stream.getvalue()))) == [
        {"method": "a", "input": "path,with,commas", "pairs": ""},
        {"method": "b", "input": "", "pairs": "10"},
    ]


def test_write_csv_rows_creates_parent_directories(tmp_path):
    output = tmp_path / "nested" / "results.csv"
    write_csv_rows([{"method": "a"}], output)
    assert output.read_text().splitlines() == ["method", "a"]
