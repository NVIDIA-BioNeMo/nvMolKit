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

"""Autotune wrapper for :func:`nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Iterable, Optional

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
from nvmolkit.autotune._ff_common import (
    clone_with_confs,
    coerce_gpu_ids,
    default_ff_search_space,
    resolve_cpu_budget,
    resolve_num_gpus,
    total_conformers,
)
from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
from nvmolkit.types import HardwareOptions

if TYPE_CHECKING:
    from rdkit.ForceField.rdForceField import MMFFMolProperties


def tune_mmff_optimize(
    molecules: list[Mol],
    *,
    maxIters: int = 200,
    properties: "MMFFMolProperties | Sequence[MMFFMolProperties | None] | None" = None,
    nonBondedThreshold: float | Sequence[float] = 100.0,
    ignoreInterfragInteractions: bool | Sequence[bool] = True,
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
    """Tune :class:`HardwareOptions` for :func:`MMFFOptimizeMoleculesConfs`.

    Each trial clones the calibration molecules (preserving their conformers)
    and runs MMFF optimization with the trial-suggested hardware options.

    Args:
        molecules: Workload of pre-embedded RDKit molecules. Each molecule
            must already contain the conformers to optimize.
        maxIters: ``maxIters`` argument forwarded to each trial.
        properties: ``properties`` argument forwarded to each trial. If a
            per-molecule sequence is provided, it must match
            ``len(molecules)`` (the wrapper subsets it for each trial).
        nonBondedThreshold: ``nonBondedThreshold`` forwarded to each trial.
            Accepts a scalar or per-molecule sequence.
        ignoreInterfragInteractions: ``ignoreInterfragInteractions``
            forwarded to each trial.
        gpuIds: GPU device IDs to use. Fixed across the study.
        calibration_set: Optional explicit indices into ``molecules``.
        calibration_fraction: Fraction of the workload to auto-sample.
        calibration_max_size: Cap on the auto-sampled calibration size.
        target_seconds_per_trial: Target wall-clock budget for one trial.
        n_trials: Number of Optuna trials to run after warm-up.
        search_space_overrides: Optional overrides for ``batchSize`` /
            ``batchesPerGpu`` ranges.
        cpu_budget: Optional explicit cap on total CPU threads. The default
            (``None``) uses ``os.cpu_count()``. Set this when normalizing
            tuning runs across machines with different core counts so the
            search space stays comparable.
        sampler: Optional Optuna sampler.
        seed: Seed for the default sampler.
        verbose: Print warm-up and trial diagnostics.

    Returns:
        :class:`TuneResult` with ``best_config`` set to a fully-populated
        :class:`HardwareOptions` instance.
    """
    optuna = _require_optuna()  # noqa: F841

    if not molecules:
        raise ValueError("molecules must be non-empty for autotuning")

    indices = normalize_calibration_set(
        calibration_set,
        len(molecules),
        fraction=calibration_fraction,
        max_size=calibration_max_size,
    )
    fixed_gpu_ids = coerce_gpu_ids(gpuIds)
    num_gpus = resolve_num_gpus(fixed_gpu_ids)
    cpus = resolve_cpu_budget(cpu_budget)
    space = resolve_search_space(default_ff_search_space(num_gpus, cpus), search_space_overrides)

    is_seq_props = isinstance(properties, Sequence) and not hasattr(properties, "SetMMFFVariant")
    is_seq_nbt = isinstance(nonBondedThreshold, Sequence) and not isinstance(nonBondedThreshold, (str, bytes))
    is_seq_ifg = isinstance(ignoreInterfragInteractions, Sequence) and not isinstance(
        ignoreInterfragInteractions, (str, bytes)
    )

    def _slice_param(param, is_sequence: bool, idxs: list[int]):
        if is_sequence:
            return [param[i] for i in idxs]
        return param

    def _make_options(values: dict[str, Any]) -> HardwareOptions:
        return HardwareOptions(
            batchSize=int(values.get("batchSize", -1)),
            batchesPerGpu=int(values.get("batchesPerGpu", -1)),
            gpuIds=fixed_gpu_ids if fixed_gpu_ids else None,
        )

    def _run_once(options: HardwareOptions, state: CalibrationState) -> int:
        slice_mols = [molecules[i] for i in state.indices]
        cloned = clone_with_confs(slice_mols)
        MMFFOptimizeMoleculesConfs(
            cloned,
            maxIters=maxIters,
            properties=_slice_param(properties, is_seq_props, state.indices),
            nonBondedThreshold=_slice_param(nonBondedThreshold, is_seq_nbt, state.indices),
            ignoreInterfragInteractions=_slice_param(ignoreInterfragInteractions, is_seq_ifg, state.indices),
            hardwareOptions=options,
        )
        return total_conformers(cloned)

    def default_runner(state: CalibrationState) -> int:
        return _run_once(
            HardwareOptions(gpuIds=fixed_gpu_ids if fixed_gpu_ids else None),
            state,
        )

    def trial_runner(trial, state: CalibrationState) -> int:
        values = {name: suggest_from_space(trial, name, spec) for name, spec in space.items()}
        return _run_once(_make_options(values), state)

    def build_config(params_dict: dict[str, Any]) -> HardwareOptions:
        merged = {name: params_dict.get(name, collect_int_from_space(spec)) for name, spec in space.items()}
        return _make_options(merged)

    initial_state = CalibrationState(indices=list(indices))
    return run_study(
        default_runner=default_runner,
        trial_runner=trial_runner,
        build_config=build_config,
        initial_state=initial_state,
        n_trials=n_trials,
        target_seconds_per_trial=target_seconds_per_trial,
        sampler=sampler,
        seed=seed,
        verbose=verbose,
    )
