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

"""Autotune wrappers for :class:`MMFFBatchedForcefield` and :class:`UFFBatchedForcefield`.

Because the batched-forcefield API exposes per-element constraint setup through
Python views, the wrapper takes a user-provided factory callable that rebuilds
a batched forcefield (with whatever properties / constraints the user wants)
given a slice of molecules and a :class:`HardwareOptions`. This keeps the
autotuner agnostic to constraint configuration while still rebuilding the
native forcefield with each trial's hardware settings.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Union

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
from nvmolkit.batchedForcefield import MMFFBatchedForcefield, UFFBatchedForcefield
from nvmolkit.types import HardwareOptions

BatchedForcefield = Union[MMFFBatchedForcefield, UFFBatchedForcefield]
ForcefieldFactory = Callable[[list[Mol], HardwareOptions], BatchedForcefield]


def tune_batched_forcefield(
    molecules: list[Mol],
    factory: ForcefieldFactory,
    *,
    maxIters: Optional[int] = None,
    forceTol: Optional[float] = None,
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
    """Tune :class:`HardwareOptions` for a batched-forcefield workflow.

    Args:
        molecules: Workload of pre-embedded RDKit molecules.
        factory: Callable invoked once per trial with the cloned calibration
            molecule slice and the trial-suggested :class:`HardwareOptions`.
            Should return a fully-configured :class:`MMFFBatchedForcefield` or
            :class:`UFFBatchedForcefield` (with any per-element properties or
            constraints already applied).
        maxIters: Optional override for the ``minimize`` ``maxIters`` argument.
            When ``None``, the default of the returned forcefield class is used.
        forceTol: Optional override for the ``minimize`` ``forceTol`` argument.
            When ``None``, the default of the returned forcefield class is used.
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

    minimize_kwargs: dict[str, Any] = {}
    if maxIters is not None:
        minimize_kwargs["maxIters"] = int(maxIters)
    if forceTol is not None:
        minimize_kwargs["forceTol"] = float(forceTol)

    def _make_options(values: dict[str, Any]) -> HardwareOptions:
        return HardwareOptions(
            batchSize=int(values.get("batchSize", -1)),
            batchesPerGpu=int(values.get("batchesPerGpu", -1)),
            gpuIds=fixed_gpu_ids if fixed_gpu_ids else None,
        )

    def _run_once(options: HardwareOptions, state: CalibrationState) -> int:
        slice_mols = [molecules[i] for i in state.indices]
        cloned = clone_with_confs(slice_mols)
        ff = factory(cloned, options)
        ff.minimize(**minimize_kwargs)
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
