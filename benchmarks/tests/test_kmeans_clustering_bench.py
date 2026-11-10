# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import kmeans_clustering_bench as bench
import numpy as np
import pytest


@pytest.mark.parametrize("flag", ["--no-cuml", "--no_cuml"])
def test_backend_selection_accepts_cuml_disable_spellings(flag):
    args = bench._build_parser().parse_args(["--smiles", "input.smi", flag])

    assert args.no_cuml
    assert not args.no_nvmolkit


@pytest.mark.parametrize("flag", ["--no-nvmolkit", "--no_nvmolkit"])
def test_backend_selection_accepts_nvmolkit_disable_spellings(flag):
    args = bench._build_parser().parse_args(["--smiles", "input.smi", flag])

    assert args.no_nvmolkit
    assert not args.no_cuml


def test_timing_fields_use_standard_mean_schema():
    timing = bench.time_it(lambda: None, runs=2, warmups=0)

    fields = bench._timing_fields(timing)

    assert set(fields) == {"time_ms", "std_ms", "runs_completed"}
    assert fields["time_ms"] == timing.mean_ms
    assert fields["std_ms"] == timing.std_ms
    assert fields["runs_completed"] == 2


def test_parser_uses_standard_input_and_workload_options():
    args = bench._build_parser().parse_args(
        [
            "--smiles",
            "input.smi",
            "--num_mols",
            "10",
            "20",
            "--n_clusters",
            "4",
            "8",
            "--metrics",
            "tanimoto",
            "cosine",
            "--max_iter",
            "50",
            "--fp_size",
            "512",
            "-r",
            "2",
        ]
    )

    assert args.smiles == "input.smi"
    assert args.num_mols == [10, 20]
    assert args.n_clusters == [4, 8]
    assert args.metrics == ["tanimoto", "cosine"]
    assert args.init == "k-means++"
    assert args.max_iter == 50
    assert args.fp_size == 512
    assert args.runs == 2
    assert args.output is None


def test_cuml_is_not_imported_when_disabled(monkeypatch):
    def unexpected_import(_name):
        raise AssertionError("cuml import should not be attempted")

    monkeypatch.setattr(bench.importlib, "import_module", unexpected_import)

    module, status, detail = bench._load_cuml(False)

    assert module is None
    assert status == "disabled"
    assert detail == ""


def test_missing_cuml_is_reported_as_optional(monkeypatch):
    def missing_import(_name):
        raise ImportError("no cuml")

    monkeypatch.setattr(bench.importlib, "import_module", missing_import)

    module, status, detail = bench._load_cuml(True)

    assert module is None
    assert status == "not_installed"
    assert detail == "no cuml"


def test_cluster_distribution_ignores_empty_clusters():
    distribution = bench._cluster_distribution(np.array([3, 0, 1, 1]), 5)

    assert distribution["num_clusters"] == 3
    assert distribution["mean_cluster_size"] == pytest.approx(5 / 3)
    assert distribution["cluster_size_p50"] == 1
    assert distribution["max_cluster_size"] == 3
    assert distribution["singleton_cluster_fraction"] == pytest.approx(2 / 3)
    assert distribution["items_in_singletons_fraction"] == pytest.approx(2 / 5)
    assert distribution["largest_cluster_fraction"] == pytest.approx(3 / 5)
