# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import bitbirch_clustering_bench as bench
import pytest


@pytest.mark.parametrize("flag", ["--no-bblean", "--no_bblean"])
def test_backend_selection_accepts_bblean_disable_spellings(flag):
    args = bench._build_parser().parse_args(["--smiles", "input.smi", flag])

    assert args.no_bblean
    assert not args.no_nvmolkit


@pytest.mark.parametrize("flag", ["--no-nvmolkit", "--no_nvmolkit"])
def test_backend_selection_accepts_nvmolkit_disable_spellings(flag):
    args = bench._build_parser().parse_args(["--smiles", "input.smi", flag])

    assert args.no_nvmolkit
    assert not args.no_bblean


def test_timing_fields_use_standard_mean_schema():
    timing = bench.time_it(lambda: None, runs=2, warmups=0)

    fields = bench._timing_fields(timing)

    assert set(fields) == {"time_ms", "std_ms", "runs_completed"}
    assert fields["time_ms"] == timing.mean_ms
    assert fields["std_ms"] == timing.std_ms
    assert fields["runs_completed"] == 2


def test_bblean_deadline_accepts_standard_spellings():
    parser = bench._build_parser()

    hyphenated = parser.parse_args(["--smiles", "input.smi", "--bblean-max-seconds", "12.5"])
    underscored = parser.parse_args(["--smiles", "input.smi", "--bblean_max_seconds", "7.5"])

    assert hyphenated.bblean_max_seconds == 12.5
    assert underscored.bblean_max_seconds == 7.5


def test_parser_uses_standard_input_and_workload_options():
    args = bench._build_parser().parse_args(
        [
            "--smiles",
            "input.smi",
            "--num_mols",
            "10",
            "20",
            "--branching_factor",
            "64",
            "--batch_size",
            "512",
            "--fp_size",
            "512",
            "-r",
            "2",
        ]
    )

    assert args.smiles == "input.smi"
    assert args.num_mols == [10, 20]
    assert args.branching_factor == 64
    assert args.batch_size == 512
    assert args.fp_size == 512
    assert args.runs == 2
    assert args.output is None


def test_bblean_is_not_imported_when_disabled(monkeypatch):
    def unexpected_import(_name):
        raise AssertionError("bblean import should not be attempted")

    monkeypatch.setattr(bench.importlib, "import_module", unexpected_import)

    module, status, detail = bench._load_bblean(False)

    assert module is None
    assert status == "disabled"
    assert detail == ""


def test_missing_bblean_is_reported_as_optional(monkeypatch):
    def missing_import(_name):
        raise ImportError("no bblean")

    monkeypatch.setattr(bench.importlib, "import_module", missing_import)

    module, status, detail = bench._load_bblean(True)

    assert module is None
    assert status == "not_installed"
    assert detail == "no bblean"


def test_cluster_distribution_reports_sizes_and_singletons():
    distribution = bench._cluster_distribution([[0, 1, 2], [3], [4]], 5)

    assert distribution["num_clusters"] == 3
    assert distribution["mean_cluster_size"] == pytest.approx(5 / 3)
    assert distribution["cluster_size_p50"] == 1
    assert distribution["cluster_size_p90"] == pytest.approx(2.6)
    assert distribution["cluster_size_p99"] == pytest.approx(2.96)
    assert distribution["max_cluster_size"] == 3
    assert distribution["singleton_cluster_fraction"] == pytest.approx(2 / 3)
    assert distribution["items_in_singletons_fraction"] == pytest.approx(2 / 5)
    assert distribution["largest_cluster_fraction"] == pytest.approx(3 / 5)
