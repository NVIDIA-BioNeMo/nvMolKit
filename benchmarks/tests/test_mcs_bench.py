# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mcs_bench import (
    _build_parser,
    _load_config_dataframe,
    _normalize_config_row,
    _validate_results,
    sample_pairs,
)


def _args(**overrides):
    values = {
        "num_pairs": 17,
        "batch_size": 0,
        "block_size": 128,
        "workers": -1,
        "prep_threads": -1,
        "executors_per_runner": -1,
        "num_gpus": 1,
        "atom_compare": "elements",
        "bond_compare": "order",
        "match_valences": False,
        "match_formal_charge": False,
        "ring_matches_ring_only": False,
        "timeout_seconds": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_partial_config_row_inherits_cli_defaults_and_parses_booleans():
    config = _normalize_config_row(
        {"block_size": 256, "match_formal_charge": "yes"},
        _args(workers=3),
    )

    assert config["num_pairs"] == 17
    assert config["workers"] == 3
    assert config["block_size"] == 256
    assert config["match_formal_charge"]


def test_config_dataframe_supports_multiple_rows(tmp_path):
    path = tmp_path / "mcs.csv"
    pd.DataFrame([{"num_pairs": 10, "block_size": 64}, {"num_pairs": 20, "block_size": 512}]).to_csv(
        path,
        index=False,
    )

    rows = _load_config_dataframe(str(path))

    assert [(row["num_pairs"], row["block_size"]) for row in rows] == [(10, 64), (20, 512)]


def test_config_validation_rejects_invalid_gpu_settings():
    with pytest.raises(ValueError, match="block_size"):
        _normalize_config_row({"block_size": 32}, _args())
    with pytest.raises(ValueError, match="num_gpus"):
        _normalize_config_row({"num_gpus": 0}, _args())


def test_pair_sampling_is_unique_deterministic_and_in_bounds():
    pairs = sample_pairs(num_mols=10, num_pairs=25, seed=123)

    assert pairs == sample_pairs(num_mols=10, num_pairs=25, seed=123)
    assert len(pairs) == len(set(pairs)) == 25
    assert all(0 <= idx_a < idx_b < 10 for idx_a, idx_b in pairs)


def test_pair_sampling_caps_at_all_possible_pairs():
    assert sample_pairs(num_mols=3, num_pairs=100, seed=123) == [(0, 1), (0, 2), (1, 2)]


def test_validation_compares_all_atom_and_bond_counts():
    class FakeResult:
        num_atoms = np.asarray([3, 1], dtype=np.uint32)
        num_bonds = np.asarray([2, 0], dtype=np.uint32)
        pairs = ((0, 1), (1, 2))

        def __len__(self):
            return len(self.pairs)

    result = FakeResult()

    _validate_results(result, [(3, 2), (1, 0)])
    with pytest.raises(AssertionError, match="pair 1"):
        _validate_results(result, [(3, 2), (2, 1)])


def test_parser_matches_benchmark_input_and_warmup_conventions():
    parser = _build_parser()

    args = parser.parse_args(["--smiles", "input.smi", "--config", "sweep.parquet", "--no_warmup"])

    assert args.smiles == "input.smi"
    assert args.config == "sweep.parquet"
    assert not args.warmup


def test_parser_matches_substructure_autotune_conventions():
    parser = _build_parser()

    args = parser.parse_args(
        [
            "--smiles",
            "input.smi",
            "--autotune",
            "--autotune_save",
            "mcs.json",
            "--autotune_trials",
            "7",
            "--autotune_time_budget",
            "2.5",
            "--autotune_calibration_size",
            "20",
            "--autotune_seed",
            "9",
        ]
    )

    assert args.autotune
    assert args.autotune_save == "mcs.json"
    assert args.autotune_trials == 7
    assert args.autotune_time_budget == 2.5
    assert args.autotune_calibration_size == 20
    assert args.autotune_seed == 9
