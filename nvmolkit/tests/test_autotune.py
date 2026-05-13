# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for :mod:`nvmolkit.autotune`."""

import importlib
import importlib.util
import json
import os
import sys

import pytest

from rdkit import Chem
from rdkit.Chem.AllChem import ETKDGv3

import nvmolkit.autotune as autotune
from nvmolkit.autotune import _calibration, _core, _ff_common
from nvmolkit.autotune.tune_embed_molecules import _default_embed_search_space
from nvmolkit.autotune.tune_substructure import (
    _default_substruct_search_space,
    _suggest_preprocessing_threads,
)
from nvmolkit.substructure import (
    SubstructSearchConfig,
    countSubstructMatches,
    getSubstructMatches,
    hasSubstructMatch,
)

from nvmolkit.types import HardwareOptions


SDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "tests",
    "test_data",
    "MMFF94_dative.sdf",
)


def _load_test_mols(num_mols: int) -> list:
    """Load a small batch of MMFF molecules from the project's test data."""
    if not os.path.exists(SDF_PATH):
        pytest.skip(f"Test data file not found: {SDF_PATH}")
    supplier = Chem.SDMolSupplier(SDF_PATH, removeHs=False, sanitize=True)
    molecules = []
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        if i >= num_mols:
            break
        molecules.append(mol)
    if len(molecules) < num_mols:
        pytest.skip(f"Expected {num_mols} molecules, found {len(molecules)}")
    return molecules


def _embed_mols_for_optimize(num_mols: int, num_confs: int = 2) -> list:
    """Return molecules with conformers attached, suitable for FF tests."""
    base = _load_test_mols(num_mols)
    embedded = []
    params = ETKDGv3()
    params.useRandomCoords = True
    from nvmolkit.embedMolecules import EmbedMolecules

    fresh = []
    for mol in base:
        copy = Chem.Mol(mol)
        copy.RemoveAllConformers()
        fresh.append(copy)
    EmbedMolecules(fresh, params, confsPerMolecule=num_confs)
    for mol in fresh:
        if mol.GetNumConformers() == 0:
            pytest.skip("Failed to embed conformers for autotune test")
        embedded.append(mol)
    return embedded


# =============================================================================
# Always-on tests: must pass with or without optuna installed.
# =============================================================================


def test_import_without_optuna_succeeds():
    """Importing the autotune package must never depend on optuna."""
    importlib.reload(autotune)
    assert hasattr(autotune, "is_available")
    assert hasattr(autotune, "tune_embed_molecules")


def test_tune_raises_clear_error_when_optuna_missing(monkeypatch):
    """Calling a tune wrapper without optuna raises a helpful ImportError."""
    monkeypatch.setitem(sys.modules, "optuna", None)
    monkeypatch.setattr(_core, "is_optuna_available", lambda: False)
    mols = [Chem.MolFromSmiles("CCO")]
    params = ETKDGv3()
    params.useRandomCoords = True
    with pytest.raises(ImportError) as exc_info:
        autotune.tune_embed_molecules(mols, params, n_trials=1)
    message = str(exc_info.value)
    assert "optuna" in message
    assert "conda" in message


def test_is_available_matches_find_spec():
    """``is_available`` reflects optuna's importability without importing it."""
    expected = importlib.util.find_spec("optuna") is not None
    assert autotune.is_available() is expected
    assert autotune.is_optuna_available() is expected


def test_install_hint_mentions_optuna_and_conda_forge():
    """The install hint must guide both pip and conda-forge users."""
    hint = autotune.OPTUNA_INSTALL_HINT
    assert "optuna" in hint
    assert "pip install" in hint
    assert "conda" in hint


def test_hardware_options_to_from_dict_roundtrip():
    """``HardwareOptions`` serializes losslessly through ``to_dict``/``from_dict``."""
    options = HardwareOptions(preprocessingThreads=4, batchSize=256, batchesPerGpu=2, gpuIds=[0, 1])
    encoded = options.to_dict()
    assert encoded == {
        "preprocessingThreads": 4,
        "batchSize": 256,
        "batchesPerGpu": 2,
        "gpuIds": [0, 1],
    }
    restored = HardwareOptions.from_dict(encoded)
    assert restored.preprocessingThreads == 4
    assert restored.batchSize == 256
    assert restored.batchesPerGpu == 2
    assert restored.gpuIds == [0, 1]


def test_hardware_options_from_dict_rejects_unknown_keys():
    with pytest.raises(KeyError, match="Unknown HardwareOptions keys"):
        HardwareOptions.from_dict({"batchSize": 100, "bogus": 1})


def test_substruct_config_to_from_dict_roundtrip():
    """``SubstructSearchConfig`` serializes losslessly through ``to_dict``/``from_dict``."""
    config = SubstructSearchConfig(
        batchSize=512,
        workerThreads=2,
        preprocessingThreads=4,
        maxMatches=8,
        uniquify=True,
        gpuIds=[0],
    )
    encoded = config.to_dict()
    assert encoded == {
        "batchSize": 512,
        "workerThreads": 2,
        "preprocessingThreads": 4,
        "maxMatches": 8,
        "uniquify": True,
        "gpuIds": [0],
    }
    restored = SubstructSearchConfig.from_dict(encoded)
    assert restored.batchSize == 512
    assert restored.workerThreads == 2
    assert restored.preprocessingThreads == 4
    assert restored.maxMatches == 8
    assert restored.uniquify is True
    assert restored.gpuIds == [0]


def test_save_load_hardware_options_roundtrip(tmp_path):
    """End-to-end JSON persistence works without optuna."""
    options = HardwareOptions(batchSize=128, batchesPerGpu=2, gpuIds=[0])
    path = tmp_path / "opts.json"
    autotune.save(options, path)
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["_nvmolkit_config_type"] == "HardwareOptions"

    loaded = autotune.load(path)
    assert isinstance(loaded, HardwareOptions)
    assert loaded.batchSize == 128
    assert loaded.batchesPerGpu == 2
    assert loaded.gpuIds == [0]


def test_save_load_substruct_config_roundtrip(tmp_path):
    config = SubstructSearchConfig(
        batchSize=2048, workerThreads=2, preprocessingThreads=8, maxMatches=4, uniquify=True
    )
    path = tmp_path / "ss.json"
    autotune.save(config, path)
    loaded = autotune.load(path)
    assert isinstance(loaded, SubstructSearchConfig)
    assert loaded.batchSize == 2048
    assert loaded.workerThreads == 2
    assert loaded.preprocessingThreads == 8
    assert loaded.maxMatches == 4
    assert loaded.uniquify is True


def test_save_rejects_unsupported_type(tmp_path):
    with pytest.raises(TypeError):
        autotune.save({"not": "a config"}, tmp_path / "x.json")


def test_auto_subsample_caps_and_seeds():
    """``auto_subsample`` respects the cap and is deterministic with a seed."""
    indices = _calibration.auto_subsample(10000, fraction=0.1, max_size=200, seed=42)
    assert len(indices) == 200
    repeat = _calibration.auto_subsample(10000, fraction=0.1, max_size=200, seed=42)
    assert indices == repeat
    different_seed = _calibration.auto_subsample(10000, fraction=0.1, max_size=200, seed=43)
    assert indices != different_seed


def test_auto_subsample_handles_small_workload():
    indices = _calibration.auto_subsample(5, fraction=0.5, max_size=200)
    assert 1 <= len(indices) <= 5
    assert all(0 <= idx < 5 for idx in indices)
    assert len(set(indices)) == len(indices)


def test_normalize_calibration_set_explicit_indices():
    indices = _calibration.normalize_calibration_set([2, 0, 4], 5)
    assert indices == [2, 0, 4]


def test_normalize_calibration_set_rejects_out_of_range():
    with pytest.raises(IndexError):
        _calibration.normalize_calibration_set([0, 5], 5)


def test_shrink_halves_within_floor():
    assert _calibration.shrink([1, 2, 3, 4, 5, 6, 7, 8], factor=0.5) == [1, 2, 3, 4]
    assert _calibration.shrink([1, 2], factor=0.5, min_size=1) == [1]


# =============================================================================
# Search-space scaling: per-GPU and joint-CPU-budget constraints.
# =============================================================================


def test_default_ff_search_space_caps_batches_per_gpu_by_cpu_count():
    """``batchesPerGpu`` upper bound divides the CPU budget by the GPU count."""
    space_1gpu = _ff_common.default_ff_search_space(num_gpus=1, cpus=32)
    space_4gpu = _ff_common.default_ff_search_space(num_gpus=4, cpus=32)
    space_64gpu = _ff_common.default_ff_search_space(num_gpus=64, cpus=32)

    assert space_1gpu["batchesPerGpu"] == (1, 32)
    assert space_4gpu["batchesPerGpu"] == (1, 8)
    assert space_64gpu["batchesPerGpu"] == (1, 1)


def test_default_substruct_search_space_caps_per_pool():
    """Substruct: workerThreads capped per-GPU, preprocessingThreads capped at cpus."""
    space_1gpu = _default_substruct_search_space(num_gpus=1, cpus=16)
    space_4gpu = _default_substruct_search_space(num_gpus=4, cpus=16)
    space_64gpu = _default_substruct_search_space(num_gpus=64, cpus=16)

    assert space_1gpu["workerThreads"] == (1, 16)
    assert space_4gpu["workerThreads"] == (1, 4)
    assert space_64gpu["workerThreads"] == (1, 1)

    assert space_1gpu["preprocessingThreads"] == (1, 16)
    assert space_4gpu["preprocessingThreads"] == (1, 16)
    assert space_64gpu["preprocessingThreads"] == (1, 16)

    assert space_1gpu["batchSize"] == (128, 8192, "log")


class _RecordingTrial:
    """Stub Optuna trial that records the bounds passed to ``suggest_int``."""

    def __init__(self, picker=lambda low, high: low):
        self.calls: list[dict] = []
        self.picker = picker

    def suggest_int(self, name: str, low: int, high: int, log: bool = False) -> int:
        self.calls.append({"name": name, "low": low, "high": high, "log": log})
        return int(self.picker(low, high))


def test_suggest_preprocessing_threads_clamps_to_remaining_cpu_budget():
    """``preprocessingThreads`` is clamped by ``cpus - num_gpus * workerThreads``."""
    trial = _RecordingTrial()
    spec = (1, 32)
    value = _suggest_preprocessing_threads(trial, spec, worker_threads=6, num_gpus=2, cpus=16)
    assert trial.calls == [{"name": "preprocessingThreads", "low": 1, "high": 4, "log": False}]
    assert value == 1


def test_suggest_preprocessing_threads_respects_low_floor_when_budget_exhausted():
    """When the joint CPU budget is exhausted, the low bound is preserved."""
    trial = _RecordingTrial(picker=lambda low, high: high)
    spec = (4, 32)
    value = _suggest_preprocessing_threads(trial, spec, worker_threads=8, num_gpus=2, cpus=16)
    assert trial.calls == [{"name": "preprocessingThreads", "low": 4, "high": 4, "log": False}]
    assert value == 4


def test_suggest_preprocessing_threads_does_not_clamp_when_budget_available():
    """``high`` remains the user-specified upper bound when CPU budget allows."""
    trial = _RecordingTrial(picker=lambda low, high: high)
    spec = (1, 8)
    value = _suggest_preprocessing_threads(trial, spec, worker_threads=2, num_gpus=2, cpus=32)
    assert trial.calls == [{"name": "preprocessingThreads", "low": 1, "high": 8, "log": False}]
    assert value == 8


def test_suggest_preprocessing_threads_propagates_log_flag():
    """A ``"log"`` spec is forwarded through clamping as a log-uniform suggestion."""
    trial = _RecordingTrial()
    spec = (1, 32, "log")
    _suggest_preprocessing_threads(trial, spec, worker_threads=2, num_gpus=1, cpus=8)
    assert trial.calls == [{"name": "preprocessingThreads", "low": 1, "high": 6, "log": True}]


def test_default_embed_search_space_caps_per_pool():
    """Embed: per-GPU pool capped at cpus/numGpus, total pool capped at cpus."""
    space = _default_embed_search_space(num_gpus=4, cpus=16)
    assert space["batchesPerGpu"] == (1, 4)
    assert space["preprocessingThreads"] == (1, 16)


def test_resolve_num_gpus_prefers_explicit_list():
    """Explicit gpuIds override CUDA-reported count."""
    assert _ff_common.resolve_num_gpus([0, 1, 2]) == 3
    assert _ff_common.resolve_num_gpus([5]) == 1


def test_resolve_cpu_budget_falls_back_to_cpu_count(monkeypatch):
    """``cpu_budget=None`` defers to ``cpu_count()``."""
    monkeypatch.setattr(_ff_common, "cpu_count", lambda: 24)
    assert _ff_common.resolve_cpu_budget(None) == 24


def test_resolve_cpu_budget_uses_explicit_value():
    """An explicit ``cpu_budget`` overrides whatever the OS reports."""
    assert _ff_common.resolve_cpu_budget(14) == 14


def test_resolve_cpu_budget_rejects_non_positive():
    with pytest.raises(ValueError):
        _ff_common.resolve_cpu_budget(0)
    with pytest.raises(ValueError):
        _ff_common.resolve_cpu_budget(-1)


# =============================================================================
# Optuna-required tests: each guards itself with ``pytest.importorskip`` so
# the rest of the module still runs on conda-forge installs without optuna.
# =============================================================================


def test_warmup_shrinks_when_default_is_too_slow(monkeypatch):
    """Warm-up shrinks the calibration when the default exceeds the time budget."""
    state = _core.CalibrationState(indices=list(range(64)))

    call_log: list[int] = []

    def fake_runner(s: _core.CalibrationState) -> int:
        call_log.append(len(s.indices))
        return len(s.indices)

    elapsed_iter = iter([5.0, 4.5, 0.5])

    def fake_timed_run(runner, current_state):
        runner(current_state)
        return _core.TrialOutcome(elapsed_seconds=next(elapsed_iter), items=len(current_state.indices))

    monkeypatch.setattr(_core, "_timed_run", fake_timed_run)

    final_state = _core._run_warmup(
        runner=fake_runner,
        state=state,
        target_seconds_per_trial=1.0,
        max_shrinks=3,
        shrink_factor=0.5,
        min_calibration_size=1,
        verbose=False,
    )

    assert call_log == [64, 32, 16]
    assert len(final_state.indices) == 16


def test_warmup_stops_after_max_shrinks(monkeypatch):
    """Warm-up returns the smallest tried slice once retries are exhausted."""
    state = _core.CalibrationState(indices=list(range(32)))

    def fake_timed_run(runner, current_state):
        runner(current_state)
        return _core.TrialOutcome(elapsed_seconds=10.0, items=len(current_state.indices))

    monkeypatch.setattr(_core, "_timed_run", fake_timed_run)

    final_state = _core._run_warmup(
        runner=lambda s: len(s.indices),
        state=state,
        target_seconds_per_trial=1.0,
        max_shrinks=2,
        shrink_factor=0.5,
        min_calibration_size=1,
        verbose=False,
    )
    assert len(final_state.indices) == 8


def test_run_study_returns_completed_result(monkeypatch):
    """A simple run_study with a synthetic objective returns a sane TuneResult."""
    optuna = pytest.importorskip("optuna")
    state = _core.CalibrationState(indices=list(range(10)))

    def fake_timed_run(runner, current_state):
        items = runner(current_state)
        return _core.TrialOutcome(elapsed_seconds=1.0, items=items)

    monkeypatch.setattr(_core, "_timed_run", fake_timed_run)

    def trial_runner(trial, current_state):
        knob = trial.suggest_int("knob", 1, 10)
        return knob * len(current_state.indices)

    result = _core.run_study(
        default_runner=lambda s: len(s.indices),
        trial_runner=trial_runner,
        build_config=lambda params: params,
        initial_state=state,
        n_trials=4,
        target_seconds_per_trial=10.0,
        seed=0,
    )

    assert result.n_trials_run == 4
    assert result.calibration_size == 10
    assert isinstance(result.study, optuna.Study)
    assert result.best_throughput > 0


# =============================================================================
# End-to-end smoke tests for each tune_* wrapper.
# =============================================================================


@pytest.fixture
def small_mols():
    return _load_test_mols(num_mols=4)


def test_tune_substructure_rejects_unknown_api():
    """``api`` must be one of the three supported substructure entry points."""
    pytest.importorskip("optuna")
    targets = [Chem.MolFromSmiles("CCO")]
    queries = [Chem.MolFromSmarts("C")]
    with pytest.raises(ValueError, match="hasSubstructMatch"):
        autotune.tune_substructure(targets, queries, api=lambda *args, **kwargs: None, n_trials=1)


def test_tune_substructure_rejects_empty_targets():
    pytest.importorskip("optuna")
    queries = [Chem.MolFromSmarts("C")]
    with pytest.raises(ValueError, match="targets"):
        autotune.tune_substructure([], queries, n_trials=1)


def test_tune_substructure_rejects_empty_queries():
    pytest.importorskip("optuna")
    targets = [Chem.MolFromSmiles("CCO")]
    with pytest.raises(ValueError, match="queries"):
        autotune.tune_substructure(targets, [], n_trials=1)


@pytest.mark.parametrize("api", [hasSubstructMatch, countSubstructMatches, getSubstructMatches])
def test_tune_substructure_smoke(small_mols, api):
    pytest.importorskip("optuna")
    queries = [Chem.MolFromSmarts("C"), Chem.MolFromSmarts("CO")]

    result = autotune.tune_substructure(
        small_mols,
        queries,
        api=api,
        n_trials=2,
        target_seconds_per_trial=30.0,
        calibration_fraction=1.0,
        calibration_max_size=len(small_mols),
        seed=0,
    )
    assert isinstance(result.best_config, SubstructSearchConfig)
    assert result.best_throughput > 0
    assert result.n_trials_run == 2
    assert result.best_config.batchSize >= 1
    assert result.best_config.workerThreads >= 1
    assert result.best_config.preprocessingThreads >= 1

    api(small_mols, queries, result.best_config)


@pytest.fixture
def small_optimized_mols():
    return _embed_mols_for_optimize(num_mols=3, num_confs=2)


def test_tune_embed_molecules_smoke(small_mols):
    pytest.importorskip("optuna")
    params = ETKDGv3()
    params.useRandomCoords = True

    result = autotune.tune_embed_molecules(
        small_mols,
        params,
        confsPerMolecule=1,
        n_trials=2,
        target_seconds_per_trial=30.0,
        calibration_fraction=1.0,
        calibration_max_size=len(small_mols),
        seed=0,
    )

    assert isinstance(result.best_config, HardwareOptions)
    assert result.best_throughput > 0
    assert result.n_trials_run == 2
    assert result.best_config.batchSize >= 1
    assert result.best_config.batchesPerGpu >= 1
    assert result.best_config.preprocessingThreads >= 1

    fresh = [Chem.Mol(mol) for mol in small_mols]
    for mol in fresh:
        mol.RemoveAllConformers()
    from nvmolkit.embedMolecules import EmbedMolecules

    EmbedMolecules(fresh, params, confsPerMolecule=1, hardwareOptions=result.best_config)
    assert all(mol.GetNumConformers() == 1 for mol in fresh)


def test_tune_mmff_optimize_smoke(small_optimized_mols):
    pytest.importorskip("optuna")
    from rdkit.Chem import AllChem

    if not all(AllChem.MMFFHasAllMoleculeParams(mol) for mol in small_optimized_mols):
        pytest.skip("MMFF parameters unavailable for one or more molecules")

    result = autotune.tune_mmff_optimize(
        small_optimized_mols,
        maxIters=20,
        n_trials=2,
        target_seconds_per_trial=30.0,
        calibration_fraction=1.0,
        calibration_max_size=len(small_optimized_mols),
        seed=0,
    )
    assert isinstance(result.best_config, HardwareOptions)
    assert result.best_throughput > 0
    assert result.n_trials_run == 2

    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

    fresh = [Chem.Mol(mol) for mol in small_optimized_mols]
    energies = MMFFOptimizeMoleculesConfs(fresh, maxIters=10, hardwareOptions=result.best_config)
    assert len(energies) == len(fresh)


def test_tune_uff_optimize_smoke(small_optimized_mols):
    pytest.importorskip("optuna")
    from rdkit.Chem import rdForceFieldHelpers

    if not all(rdForceFieldHelpers.UFFHasAllMoleculeParams(mol) for mol in small_optimized_mols):
        pytest.skip("UFF parameters unavailable for one or more molecules")

    result = autotune.tune_uff_optimize(
        small_optimized_mols,
        maxIters=20,
        n_trials=2,
        target_seconds_per_trial=30.0,
        calibration_fraction=1.0,
        calibration_max_size=len(small_optimized_mols),
        seed=0,
    )
    assert isinstance(result.best_config, HardwareOptions)
    assert result.best_throughput > 0
    assert result.n_trials_run == 2

    from nvmolkit.uffOptimization import UFFOptimizeMoleculesConfs

    fresh = [Chem.Mol(mol) for mol in small_optimized_mols]
    energies = UFFOptimizeMoleculesConfs(fresh, maxIters=10, hardwareOptions=result.best_config)
    assert len(energies) == len(fresh)


def test_tune_batched_forcefield_smoke(small_optimized_mols):
    pytest.importorskip("optuna")
    from rdkit.Chem import AllChem

    if not all(AllChem.MMFFHasAllMoleculeParams(mol) for mol in small_optimized_mols):
        pytest.skip("MMFF parameters unavailable for one or more molecules")

    from nvmolkit.batchedForcefield import MMFFBatchedForcefield

    def factory(mols, hw_options):
        return MMFFBatchedForcefield(mols, hardwareOptions=hw_options)

    result = autotune.tune_batched_forcefield(
        small_optimized_mols,
        factory,
        maxIters=20,
        n_trials=2,
        target_seconds_per_trial=30.0,
        calibration_fraction=1.0,
        calibration_max_size=len(small_optimized_mols),
        seed=0,
    )
    assert isinstance(result.best_config, HardwareOptions)
    assert result.best_throughput > 0
    assert result.n_trials_run == 2

    fresh = [Chem.Mol(mol) for mol in small_optimized_mols]
    ff = MMFFBatchedForcefield(fresh, hardwareOptions=result.best_config)
    energies, _ = ff.minimize(maxIters=10)
    assert len(energies) == len(fresh)
