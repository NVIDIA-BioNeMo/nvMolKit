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

"""Autotune wrapper for :func:`nvmolkit.mcs.findMCS`."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from rdkit.Chem import Mol

from nvmolkit.autotune._calibration import normalize_calibration_set
from nvmolkit.autotune._core import (
    CalibrationState,
    TuneResult,
    _require_optuna,
    collect_int_from_space,
    resolve_search_space,
    run_study,
    suggest_from_space,
)
from nvmolkit.autotune._ff_common import resolve_cpu_budget
from nvmolkit.mcs import MCSConfig, findMCS


def _default_mcs_search_space(num_gpus: int, cpus: int) -> dict:
    """Build an MCS execution search space scaled to available CPUs and GPUs."""
    per_gpu_worker_max = max(1, min(8, cpus // max(1, num_gpus)))
    return {
        "batchSize": [128, 256, 512],
        "blockSize": [64, 128, 256, 512, 1024, 2048, 4096],
        "workerThreads": (1, per_gpu_worker_max),
        "preprocessingThreads": (1, cpus),
        "executorsPerRunner": (1, 4),
    }


def _suggest_preprocessing_threads(trial: Any, spec: Any, worker_threads: int, num_gpus: int, cpus: int) -> int:
    """Sample preprocessing threads without exceeding the joint CPU budget."""
    if not (isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[0], int) and isinstance(spec[1], int)):
        return int(suggest_from_space(trial, "preprocessingThreads", spec))
    low, high = int(spec[0]), int(spec[1])
    remaining = max(1, cpus - num_gpus * max(1, worker_threads))
    effective_high = max(low, min(high, remaining))
    log = len(spec) == 3 and spec[2] == "log"
    return int(trial.suggest_int("preprocessingThreads", low, effective_high, log=log))


def _coerce_pairs(pairs: Sequence[Sequence[int]], num_mols: int) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for pair_idx, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError(f"pairs[{pair_idx}] must contain exactly two molecule indices")
        idx_a, idx_b = int(pair[0]), int(pair[1])
        if not (0 <= idx_a < num_mols and 0 <= idx_b < num_mols):
            raise IndexError(f"pairs[{pair_idx}] contains a molecule index outside [0, {num_mols})")
        out.append((idx_a, idx_b))
    return tuple(out)


def tune_mcs(
    mols: Sequence[Mol],
    pairs: Sequence[Sequence[int]],
    *,
    atom_compare: str = "elements",
    bond_compare: str = "order",
    match_valences: bool = False,
    match_formal_charge: bool = False,
    ring_matches_ring_only: bool | None = None,
    atom_ring_matches_ring_only: bool | None = None,
    bond_ring_matches_ring_only: bool | None = None,
    timeout_seconds: int = 0,
    gpuIds: Optional[Iterable[int]] = None,
    calibration_set: Optional[Iterable[int]] = None,
    calibration_fraction: float = 0.1,
    calibration_max_size: int = 2000,
    target_seconds_per_trial: float = 10.0,
    n_trials: int = 30,
    search_space_overrides: Optional[dict[str, Any]] = None,
    cpu_budget: Optional[int] = None,
    sampler: Any = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> TuneResult:
    """Tune :class:`MCSConfig` for an explicit-pair, GPU-required MCS workload.

    Search semantics are fixed across trials; only execution settings in
    :class:`MCSConfig` are tuned. ``calibration_set`` contains indices into
    ``pairs``.
    """
    _require_optuna()

    mol_list = list(mols)
    if not mol_list:
        raise ValueError("mols must be non-empty for autotuning")
    pair_list = _coerce_pairs(pairs, len(mol_list))
    if not pair_list:
        raise ValueError("pairs must be non-empty for autotuning")

    indices = normalize_calibration_set(
        calibration_set,
        len(pair_list),
        fraction=calibration_fraction,
        max_size=calibration_max_size,
    )
    fixed_gpu_ids = list(gpuIds) if gpuIds is not None else []
    num_gpus = len(fixed_gpu_ids) if fixed_gpu_ids else 1
    cpus = resolve_cpu_budget(cpu_budget)
    space = resolve_search_space(_default_mcs_search_space(num_gpus, cpus), search_space_overrides)

    def make_config(values: dict[str, Any]) -> MCSConfig:
        return MCSConfig(
            batchSize=int(values.get("batchSize", 1024)),
            blockSize=int(values.get("blockSize", 128)),
            workerThreads=int(values.get("workerThreads", -1)),
            preprocessingThreads=int(values.get("preprocessingThreads", -1)),
            executorsPerRunner=int(values.get("executorsPerRunner", -1)),
            gpuIds=fixed_gpu_ids if fixed_gpu_ids else None,
        )

    def run_once(config: MCSConfig, state: CalibrationState) -> int:
        selected_pairs = [pair_list[index] for index in state.indices]
        findMCS(
            mol_list,
            mode="pairs",
            pairs=selected_pairs,
            atom_compare=atom_compare,
            bond_compare=bond_compare,
            match_valences=match_valences,
            match_formal_charge=match_formal_charge,
            ring_matches_ring_only=ring_matches_ring_only,
            atom_ring_matches_ring_only=atom_ring_matches_ring_only,
            bond_ring_matches_ring_only=bond_ring_matches_ring_only,
            require_gpu=True,
            timeout_seconds=timeout_seconds,
            config=config,
        )
        return len(selected_pairs)

    def default_runner(state: CalibrationState) -> int:
        return run_once(MCSConfig(gpuIds=fixed_gpu_ids if fixed_gpu_ids else None), state)

    def trial_runner(trial, state: CalibrationState) -> int:
        values: dict[str, Any] = {}
        for name, spec in space.items():
            if name == "preprocessingThreads":
                continue
            values[name] = suggest_from_space(trial, name, spec)
        worker_threads = int(values.get("workerThreads", 1))
        prep_spec = space.get("preprocessingThreads")
        if prep_spec is not None:
            values["preprocessingThreads"] = _suggest_preprocessing_threads(
                trial,
                prep_spec,
                worker_threads,
                num_gpus,
                cpus,
            )
        return run_once(make_config(values), state)

    def build_config(params_dict: dict[str, Any]) -> MCSConfig:
        merged = {name: params_dict.get(name, collect_int_from_space(spec)) for name, spec in space.items()}
        return make_config(merged)

    return run_study(
        default_runner=default_runner,
        trial_runner=trial_runner,
        build_config=build_config,
        initial_state=CalibrationState(indices=list(indices)),
        n_trials=n_trials,
        target_seconds_per_trial=target_seconds_per_trial,
        sampler=sampler,
        seed=seed,
        verbose=verbose,
    )
