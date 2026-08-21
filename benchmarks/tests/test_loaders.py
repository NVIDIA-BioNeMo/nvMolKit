# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for benchmark molecule-loader candidate reserves."""

import os

from bench_utils import loaders, molprep
from rdkit import Chem


def test_load_smiles_can_return_candidate_buffer(tmp_path, monkeypatch):
    smiles_path = tmp_path / "mols.smi"
    smiles_path.write_text("\n".join(["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"]))
    monkeypatch.setattr(loaders, "_process_map_batches", lambda fn, values, **_kwargs: [fn(value) for value in values])

    trimmed = loaders.load_smiles(str(smiles_path), max_count=3, seed=42)
    buffered = loaders.load_smiles(str(smiles_path), max_count=3, seed=42, keep_buffer=True)

    assert len(trimmed) == 3
    assert len(buffered) == 4


def test_process_map_batches_reports_completed_batches_and_preserves_order(monkeypatch):
    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def submit(self, function, *args):
            return FakeFuture(function(*args))

    class FakeProgress:
        def __init__(self, *, total, desc):
            self.total = total
            self.desc = desc
            self.updates = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def update(self, count):
            self.updates.append(count)

    progress_bars = []

    def make_progress(**kwargs):
        progress = FakeProgress(**kwargs)
        progress_bars.append(progress)
        return progress

    monkeypatch.setattr(loaders, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(loaders, "as_completed", lambda futures: reversed(list(futures)))
    monkeypatch.setattr(loaders, "tqdm", make_progress)

    results = loaders._process_map_batches(lambda value: value * 2, [0, 1, 2, 3, 4], desc="Working", batch_size=2)

    assert results == [0, 2, 4, 6, 8]
    assert progress_bars[0].total == 5
    assert progress_bars[0].desc == "Working"
    assert progress_bars[0].updates == [1, 2, 2]


def test_buffered_count_reserves_at_least_one_candidate():
    assert loaders._buffered_count(0) == 0
    assert loaders._buffered_count(1) == 2
    assert loaders._buffered_count(1000) == 1100


def test_available_cpu_count_supports_python_312(monkeypatch):
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {2, 4, 6}, raising=False)

    assert molprep.available_cpu_count() == 3


def test_slice_conformers_copies_and_limits_each_molecule():
    mol = Chem.MolFromSmiles("CC")
    for _ in range(3):
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()), assignId=True)

    sliced = molprep.slice_conformers([mol], 2)

    assert sliced[0] is not mol
    assert sliced[0].GetNumConformers() == 2
    assert mol.GetNumConformers() == 3
