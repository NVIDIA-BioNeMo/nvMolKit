# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for capped descriptor benchmark reporting."""

import csv
import importlib
from contextlib import nullcontext

import descriptors3d_bench as bench
import pytest
from bench_utils import TimingResult
from bench_utils.concurrent import BoundedProcessMapResult
from rdkit import Chem


@pytest.mark.parametrize(
    ("version", "correct"),
    [
        ("2025.09.6", False),
        ("2026.09.1", False),
        ("2026.09.1pre", False),
        ("2026.09.2", True),
        ("2026.09.3pre", True),
        ("2027.03.1", True),
    ],
)
def test_eem_reference_version_accepts_development_suffix(monkeypatch, version, correct):
    """Development RDKit builds import successfully and select the appropriate reference mask."""
    try:
        with monkeypatch.context() as patch:
            patch.setattr(bench.rdkit, "__version__", version)
            importlib.reload(bench)
            assert bench.RDKIT_EEM_CHARGES_CORRECT is correct
            charged = Chem.MolFromSmiles("C[NH3+]")
            charged.AddConformer(Chem.Conformer(charged.GetNumAtoms()), assignId=True)
            neutral = Chem.MolFromSmiles("CCO")
            neutral.AddConformer(Chem.Conformer(neutral.GetNumAtoms()), assignId=True)
            assert bench._eem_comparable_rows([charged, neutral]).tolist() == [correct] * 2 + [True] * 3
    finally:
        importlib.reload(bench)


@pytest.mark.parametrize("completed", [0, 1, 2])
@pytest.mark.parametrize("device_input", [False, True])
def test_descriptor_rows_and_csv_handle_zero_partial_and_complete_cpu_work(
    monkeypatch, tmp_path, completed, device_input
):
    """Unavailable CPU throughput omits ratios while preserving GPU results and CSV output."""
    mol = Chem.MolFromSmiles("CCO")
    for _ in range(2):
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()), assignId=True)
    monkeypatch.setattr(bench, "load_smiles", lambda *args, **kwargs: [mol])
    monkeypatch.setattr(bench, "embed_and_jitter", lambda *args, **kwargs: [mol])
    monkeypatch.setattr(bench, "_pack_device_coordinates", lambda mols: object())
    monkeypatch.setattr(bench, "_time_rdkit", lambda *args: (TimingResult(times_ms=[20]), completed))
    monkeypatch.setattr(bench, "time_it", lambda *args, **kwargs: TimingResult(times_ms=[10]))
    output = tmp_path / "results.csv"

    rows = bench.run(
        smiles_path="unused.smi",
        num_mols_list=[1],
        conformers_per_mol_list=[2],
        property_set_names=["single"],
        runs=1,
        warmups=0,
        seed=42,
        prep_workers=1,
        prep_threads=1,
        rdkit_threads=1,
        rdkit_max_seconds=0.001,
        validate_count=0,
        validate_tolerance=None,
        device_input=device_input,
        precision=bench.PrecisionMode.SINGLE,
        no_rdkit=False,
        no_nvmolkit=False,
        output=str(output),
    )

    row = rows[0]
    assert row["rdkit_conformers_processed"] == completed
    assert row["rdkit_conformers_per_s"] == pytest.approx(completed * 50)
    assert row["nvmolkit_conformers_per_s"] == pytest.approx(200)
    if completed:
        assert row["speedup"] == pytest.approx(4 / completed)
    else:
        assert "speedup" not in row
    if device_input:
        assert row["nvmolkit_device_input_conformers_per_s"] == pytest.approx(200)
        if completed:
            assert row["device_input_speedup"] == pytest.approx(4 / completed)
        else:
            assert "device_input_speedup" not in row
    else:
        assert "device_input_speedup" not in row
    with output.open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == 1
    assert int(saved[0]["rdkit_conformers_processed"]) == completed
    assert ("speedup" in saved[0]) == bool(completed)
    assert ("device_input_speedup" in saved[0]) == (bool(completed) and device_input)


def test_rdkit_timeout_before_first_result_reports_zero_progress(monkeypatch):
    """A first-chunk timeout records a usable timing and zero completed conformers."""
    mol = Chem.MolFromSmiles("CCO")
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()), assignId=True)
    monkeypatch.setattr(bench, "process_pool", lambda *args: nullcontext(object()))
    monkeypatch.setattr(
        bench, "process_map_bounded", lambda *args, **kwargs: BoundedProcessMapResult([], timed_out=True)
    )

    timing, completed = bench._time_rdkit(
        [mol], bench.PROPERTY_SETS["single"], threads=1, runs=1, warmups=2, max_seconds=0.001
    )

    assert completed == timing.progress == 0
    assert timing.truncated
    assert timing.median_ms >= 0
