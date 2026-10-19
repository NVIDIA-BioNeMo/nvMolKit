# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for precision selection shared by autotuned benchmarks."""

import argparse
import sys
import types

import pytest
from bench_utils import autotune_support

from nvmolkit.types import PrecisionMode


@pytest.fixture
def fake_autotune_module(monkeypatch):
    calls = []
    module = types.ModuleType("fake_autotune_module")

    def optimize(mols, precision=PrecisionMode.FULL):
        calls.append(precision)

    module.optimize = optimize
    monkeypatch.setitem(sys.modules, "fake_autotune_module", module)
    return module, calls


def test_add_precision_arg_defaults_to_full():
    parser = argparse.ArgumentParser()
    autotune_support.add_precision_arg(parser)

    assert parser.parse_args([]).precision == "full"
    assert parser.parse_args(["--precision", "single"]).precision == "single"
    with pytest.raises(SystemExit):
        parser.parse_args(["--precision", "half"])


def test_precision_mode_maps_names():
    assert autotune_support.precision_mode("full") == PrecisionMode.FULL
    assert autotune_support.precision_mode("single") == PrecisionMode.SINGLE


def test_autotune_at_precision_binds_precision_and_restores(fake_autotune_module):
    module, calls = fake_autotune_module
    original = module.optimize

    with autotune_support.autotune_at_precision("fake_autotune_module", "optimize", "single"):
        module.optimize([])
    module.optimize([])

    assert calls == [PrecisionMode.SINGLE, PrecisionMode.FULL]
    assert module.optimize is original


def test_autotune_at_precision_restores_after_error(fake_autotune_module):
    module, _ = fake_autotune_module
    original = module.optimize

    with (
        pytest.raises(RuntimeError),
        autotune_support.autotune_at_precision("fake_autotune_module", "optimize", "single"),
    ):
        raise RuntimeError("tuning failed")

    assert module.optimize is original


def test_autotune_at_precision_rejects_a_missing_function(fake_autotune_module):
    with (
        pytest.raises(AttributeError),
        autotune_support.autotune_at_precision("fake_autotune_module", "renamed_function", "single"),
    ):
        pass


@pytest.mark.parametrize(
    "module_name, function_name",
    [
        ("nvmolkit.autotune.tune_mmff_optimize", "MMFFOptimizeMoleculesConfs"),
        ("nvmolkit.autotune.tune_uff_optimize", "UFFOptimizeMoleculesConfs"),
        ("nvmolkit.autotune.tune_embed_molecules", "EmbedMolecules"),
    ],
)
def test_installed_autotune_modules_expose_the_patched_functions(module_name, function_name):
    """Guards the shim against an nvMolKit release renaming what the tuners call."""
    with autotune_support.autotune_at_precision(module_name, function_name, "single"):
        pass
