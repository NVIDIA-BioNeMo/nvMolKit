# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for benchmark molecule-loader candidate reserves."""

import io
import os

import pytest
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


def test_load_csv_preserves_selected_properties(tmp_path, monkeypatch):
    csv_path = tmp_path / "mols.csv"
    csv_path.write_text("smiles,score,ignored\nCC,1.5,x\nCCC,2.5,y\n")
    monkeypatch.setattr(loaders, "_process_map_batches", lambda fn, values, **_kwargs: [fn(value) for value in values])

    molecules = loaders.load_csv(str(csv_path), property_columns=["score"], seed=42)

    assert sorted(molecule.GetProp("score") for molecule in molecules) == ["1.5", "2.5"]
    assert all(not molecule.HasProp("ignored") for molecule in molecules)


def test_load_csv_uses_shared_reservoir_sampling(tmp_path, monkeypatch):
    csv_path = tmp_path / "mols.csv"
    csv_path.write_text("smiles,score\n" + "\n".join(f"{'C' * size},{size}" for size in range(1, 11)))
    monkeypatch.setattr(loaders, "_process_map_batches", lambda fn, values, **_kwargs: [fn(value) for value in values])

    first = loaders.load_csv(str(csv_path), property_columns=["score"], max_count=3, seed=7)
    second = loaders.load_csv(str(csv_path), property_columns=["score"], max_count=3, seed=7)

    assert len(first) == 3
    assert sorted(molecule.GetProp("score") for molecule in first) == sorted(
        molecule.GetProp("score") for molecule in second
    )


@pytest.mark.parametrize(
    "header, row, missing_column",
    [
        ("smiles,score\n", "CC\n", "score"),
        ("score,smiles\n", "1.5\n", "smiles"),
    ],
)
def test_load_csv_rejects_ragged_required_values(tmp_path, header, row, missing_column):
    csv_path = tmp_path / "mols.csv"
    csv_path.write_text(header + row)

    with pytest.raises(ValueError, match=rf"CSV row 2 is missing required values for: {missing_column}"):
        loaders.load_csv(str(csv_path), property_columns=["score"])


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


def test_load_smiles_uniformly_samples_the_whole_file(tmp_path, monkeypatch):
    smiles_path = tmp_path / "ordered_mols.smi"
    smiles_path.write_text("\n".join("C" * length for length in range(1, 101)))
    monkeypatch.setattr(loaders, "_process_map_batches", lambda fn, values, **_kwargs: [fn(value) for value in values])

    sample = loaders.load_smiles(str(smiles_path), max_count=10, seed=42)
    repeated = loaders.load_smiles(str(smiles_path), max_count=10, seed=42)
    different_seed = loaders.load_smiles(str(smiles_path), max_count=10, seed=43)

    atom_counts = [mol.GetNumAtoms() for mol in sample]
    assert atom_counts == [mol.GetNumAtoms() for mol in repeated]
    assert atom_counts != [mol.GetNumAtoms() for mol in different_seed]
    assert set(atom_counts) != set(range(1, 11))
    assert max(atom_counts) > 50


def test_buffered_count_reserves_at_least_one_candidate():
    assert loaders._buffered_count(0) == 0
    assert loaders._buffered_count(1) == 2
    assert loaders._buffered_count(1000) == 1100


def test_available_cpu_count_supports_python_312(monkeypatch):
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {2, 4, 6}, raising=False)

    assert molprep.available_cpu_count() == 3


def test_physical_cpu_count_deduplicates_smt_siblings_within_affinity(monkeypatch):
    cpuinfo = """
processor : 0
physical id : 0
core id : 0

processor : 1
physical id : 0
core id : 0

processor : 2
physical id : 0
core id : 1

processor : 3
physical id : 0
core id : 1
"""
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    monkeypatch.setattr("builtins.open", lambda _path: io.StringIO(cpuinfo))

    assert molprep.available_physical_cpu_count() == 2


def test_physical_cpu_count_distinguishes_sockets_and_respects_affinity():
    cpuinfo = "\n\n".join(
        f"processor : {processor}\nphysical id : {socket}\ncore id : 0"
        for processor, socket in [(0, 0), (1, 0), (2, 1), (3, 1)]
    )
    assert molprep._physical_cpu_count_from_cpuinfo(cpuinfo, {0, 1, 2, 3}) == 2
    assert molprep._physical_cpu_count_from_cpuinfo(cpuinfo, {1, 3}) == 2
    assert molprep._physical_cpu_count_from_cpuinfo(cpuinfo, {0, 1}) == 1
    assert molprep._physical_cpu_count_from_cpuinfo(cpuinfo, set()) is None


@pytest.mark.parametrize("cpuinfo", ["", "processor : x\nphysical id : 0\ncore id : 0", "processor : 0"])
def test_physical_cpu_count_falls_back_without_topology(monkeypatch, cpuinfo):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {2, 4, 6})
    monkeypatch.setattr("builtins.open", lambda _path: io.StringIO(cpuinfo))
    assert molprep.available_physical_cpu_count() == 3


def test_physical_cpu_count_falls_back_without_proc(monkeypatch):
    def fail_open(_):
        raise OSError("no proc filesystem")

    monkeypatch.delattr(os, "sched_getaffinity")
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    monkeypatch.setattr("builtins.open", fail_open)
    assert molprep.available_physical_cpu_count() == 4


def test_slice_conformers_copies_and_limits_each_molecule():
    mol = Chem.MolFromSmiles("CC")
    for _ in range(3):
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()), assignId=True)

    sliced = molprep.slice_conformers([mol], 2)

    assert sliced[0] is not mol
    assert sliced[0].GetNumConformers() == 2
    assert mol.GetNumConformers() == 3


def test_substructure_thread_resolution_uses_one_explicit_budget():
    assert molprep.resolve_substructure_threads(worker_threads=-1, num_gpus=8, physical_cores=30) == (3, 6)
    assert molprep.resolve_substructure_threads(worker_threads=8, num_gpus=8, physical_cores=128) == (8, 64)
    assert molprep.resolve_substructure_threads(worker_threads=-1, num_gpus=8, physical_cores=128) == (4, 96)
    assert molprep.resolve_substructure_threads(worker_threads=8, num_gpus=8, physical_cores=32) == (8, 1)
    assert molprep.resolve_substructure_threads(worker_threads=-1, num_gpus=1, physical_cores=1) == (1, 1)
    assert molprep.resolve_substructure_threads(
        worker_threads=-1, num_gpus=8, preprocessing_threads=2, physical_cores=30
    ) == (3, 2)


@pytest.mark.parametrize(
    "workers, gpus, cores, expected",
    [(-1, 8, 32, (4, 1)), (-1, 1, 4, (4, 1)), (-1, 8, 4, (1, 1)), (8, 8, 32, (8, 1))],
)
def test_substructure_threads_keep_required_workers_when_budget_is_exhausted(workers, gpus, cores, expected):
    assert molprep.resolve_substructure_threads(workers, gpus, physical_cores=cores) == expected


def test_substructure_threads_honor_explicit_preprocessing_over_budget():
    assert molprep.resolve_substructure_threads(4, 8, preprocessing_threads=3, physical_cores=32) == (4, 3)
