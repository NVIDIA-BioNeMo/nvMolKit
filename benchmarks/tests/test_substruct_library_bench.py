# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from bench_utils import Deadline
from rdkit import Chem
from rdkit.Chem import rdSubstructLibrary
from substruct_library_bench import (
    Measurement,
    benchmark_nvmolkit,
    benchmark_rdkit,
    first_mismatch,
    load_queries,
    result_row,
    run_rdkit_queries,
)

TARGETS = ["CCO", "c1ccccc1O", "CC(=O)O", "CCN", "c1ccncc1", "OCCO"]
QUERIES = ["[OX2H]", "c1ccccc1", "[#7]", "C=O", "[Si]"]


@pytest.fixture
def mols():
    return [Chem.MolFromSmiles(smiles) for smiles in TARGETS]


@pytest.fixture
def queries():
    return [Chem.MolFromSmarts(smarts) for smarts in QUERIES]


def expected_matches(mols, queries):
    return [[index for index, mol in enumerate(mols) if mol.HasSubstructMatch(query)] for query in queries]


@pytest.mark.parametrize("holder", ["mol", "cached-pattern"])
def test_rdkit_backend_matches_direct_substructure_search(mols, queries, holder):
    measurements = benchmark_rdkit(
        mols,
        queries,
        operations=["has", "count", "get"],
        holder=holder,
        num_threads=1,
        max_results=-1,
        runs=1,
        max_seconds=0,
    )
    expected = expected_matches(mols, queries)
    assert measurements["get"].results == expected
    assert measurements["count"].results == [len(matches) for matches in expected]
    assert measurements["has"].results == [bool(matches) for matches in expected]
    assert all(measurement.completed_queries == len(queries) for measurement in measurements.values())


def test_rdkit_queries_stop_at_the_deadline(mols, queries):
    library = rdSubstructLibrary.SubstructLibrary(rdSubstructLibrary.MolHolder())
    for mol in mols:
        library.AddMol(mol)

    class ExpiresAfter(Deadline):
        def __init__(self, checks):
            super().__init__(0)
            self.checks = checks

        def expired(self):
            self.checks -= 1
            return self.checks < 0

    assert run_rdkit_queries(library, queries, "get", -1, 1, ExpiresAfter(2)) == expected_matches(mols, queries)[:2]


def test_validation_compares_the_completed_prefix():
    assert first_mismatch([[0, 2], [1], [3]], [[0, 2], [1]], "get") is None
    assert "query 1" in first_mismatch([[0, 2], [1]], [[0, 2], [1, 4]], "get")


def test_result_row_reports_throughput_over_completed_queries():
    measurement = Measurement(
        staging_ms=10, finalize_ms=20, steady_ms=50, steady_std_ms=1, results=[[1], [], [2, 3]], completed_queries=3
    )
    row = result_row(measurement, operation="get", num_mols=100, num_queries=4, backend="nvmolkit")

    assert row["num_queries"] == 4
    assert row["positive_queries"] == 2
    assert row["amortized_ms"] == 80
    assert row["steady_queries_per_s"] == pytest.approx(60)
    assert row["steady_pairs_per_s"] == pytest.approx(6000)
    assert row["amortized_queries_per_s"] == pytest.approx(37.5)


def test_smiles_queries_ignore_stereochemistry(tmp_path):
    path = tmp_path / "queries.smi"
    path.write_text("C[C@H](O)CC\n")
    args = SimpleNamespace(smarts=None, query_smiles=str(path), num_queries=0, sanitize=True, seed=0)

    (query,) = load_queries(args)
    assert Chem.MolToSmiles(query) == "CCC(C)O"


def test_nvmolkit_backend_matches_rdkit_in_every_query_mode(mols, queries):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    from nvmolkit.substructure import SubstructSearchConfig

    measurements = benchmark_nvmolkit(
        mols,
        queries,
        operations=["has", "count", "get"],
        query_modes=["serial", "concurrent"],
        config=SubstructSearchConfig(),
        max_results=-1,
        runs=1,
        warmups=0,
    )
    expected = expected_matches(mols, queries)
    for mode in ("serial", "concurrent"):
        assert measurements[("get", mode)].results == expected
        assert measurements[("count", mode)].results == [len(matches) for matches in expected]
        assert measurements[("has", mode)].results == [bool(matches) for matches in expected]
