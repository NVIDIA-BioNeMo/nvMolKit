# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for benchmark molecule-loader candidate reserves."""

import os

from bench_utils import loaders, molprep


def test_load_smiles_can_return_candidate_buffer(tmp_path, monkeypatch):
    smiles_path = tmp_path / "mols.smi"
    smiles_path.write_text("\n".join(["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"]))
    monkeypatch.setattr(loaders, "process_map", lambda fn, values, **_kwargs: [fn(value) for value in values])

    trimmed = loaders.load_smiles(str(smiles_path), max_count=3, seed=42)
    buffered = loaders.load_smiles(str(smiles_path), max_count=3, seed=42, keep_buffer=True)

    assert len(trimmed) == 3
    assert len(buffered) == 4


def test_buffered_count_reserves_at_least_one_candidate():
    assert loaders._buffered_count(0) == 0
    assert loaders._buffered_count(1) == 2
    assert loaders._buffered_count(1000) == 1100


def test_available_cpu_count_supports_python_312(monkeypatch):
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {2, 4, 6}, raising=False)

    assert molprep.available_cpu_count() == 3
