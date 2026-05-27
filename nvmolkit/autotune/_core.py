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

"""Generic Optuna-backed tuning core for nvMolKit hardware options.

The functions here are API-agnostic: callers provide a *trial function* that
builds a configuration object from an :class:`optuna.trial.Trial`, runs the
target API once on the active calibration slice, and returns a throughput
value (items / second). The core orchestrates a warm-up phase that adapts the
calibration size to fit a user-provided per-trial time budget, and then runs
an Optuna study with that fixed calibration slice.
"""

from __future__ import annotations

import importlib.util
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

OPTUNA_INSTALL_HINT = (
    "nvMolKit autotune requires the 'optuna' package, which is an optional dependency. "
    "Install it with 'pip install optuna' (or 'pip install nvMolKit[autotune]' if you "
    "installed nvMolKit via pip). conda-forge users: 'conda install -c conda-forge optuna'."
)


def is_optuna_available() -> bool:
    """Return ``True`` if optuna can be imported, ``False`` otherwise.

    This check uses :func:`importlib.util.find_spec` and does not actually
    import optuna.
    """
    return importlib.util.find_spec("optuna") is not None


def _require_optuna():
    """Import optuna or raise an :class:`ImportError` with install instructions."""
    if not is_optuna_available():
        raise ImportError(OPTUNA_INSTALL_HINT)
    import optuna  # noqa: PLC0415

    return optuna


@dataclass
class CalibrationState:
    """Mutable state describing the calibration slice used for a tuning study.

    Attributes:
        indices: Indices into the user-provided workload that participate in
            each trial. Each adaptive-shrink iteration replaces this list with
            a smaller subsample.
    """

    indices: list[int]


@dataclass
class TrialOutcome:
    """Result of one timed trial execution.

    Attributes:
        elapsed_seconds: Wall-clock time of the trial including a CUDA
            synchronize before measurement ends.
        items: Number of items the trial processed; throughput is computed as
            ``items / elapsed_seconds``.
    """

    elapsed_seconds: float
    items: int


@dataclass
class TuneResult:
    """Result returned by every ``tune_*`` wrapper.

    Attributes:
        best_config: The configuration object (e.g. ``HardwareOptions`` or
            ``SubstructSearchConfig``) corresponding to the best trial.
        best_throughput: Throughput in items per second of the best trial.
        best_params: Raw parameter dictionary of the best optuna trial.
        calibration_size: Number of indices used in each trial after any
            adaptive shrink.
        n_trials_run: Number of optuna trials that completed successfully.
        study: The underlying ``optuna.Study`` object, or ``None`` if optuna
            is unavailable.
    """

    best_config: Any
    best_throughput: float
    best_params: dict[str, Any] = field(default_factory=dict)
    calibration_size: int = 0
    n_trials_run: int = 0
    study: Any = None


def _timed_run(runner: Callable[[CalibrationState], int], state: CalibrationState) -> TrialOutcome:
    """Run ``runner`` once and return its wall-clock duration and item count.

    The runner is expected to return the number of items it processed (used to
    compute throughput).
    """
    torch.cuda.synchronize()
    start = time.perf_counter()
    items = runner(state)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return TrialOutcome(elapsed_seconds=elapsed, items=int(items))


def _run_warmup(
    runner: Callable[[CalibrationState], int],
    state: CalibrationState,
    target_seconds_per_trial: float,
    max_shrinks: int,
    shrink_factor: float,
    min_calibration_size: int,
    verbose: bool,
) -> CalibrationState:
    """Time a default-config run and shrink the calibration slice if needed.

    Repeatedly halves the calibration slice (multiplied by ``shrink_factor``)
    when a single run takes longer than ``2 * target_seconds_per_trial``. Stops
    after ``max_shrinks`` retries or when the slice would fall below
    ``min_calibration_size``. The returned state always has at least one item.
    """
    if not state.indices:
        raise ValueError("Calibration set is empty; cannot run autotune warm-up")

    threshold = 2.0 * target_seconds_per_trial
    for attempt in range(max_shrinks + 1):
        outcome = _timed_run(runner, state)
        if verbose:
            print(
                f"[autotune] warm-up attempt {attempt}: "
                f"size={len(state.indices)} elapsed={outcome.elapsed_seconds:.3f}s"
            )
        if outcome.elapsed_seconds <= threshold:
            return state
        if attempt == max_shrinks:
            break
        new_size = max(min_calibration_size, int(len(state.indices) * shrink_factor))
        if new_size >= len(state.indices):
            break
        state = CalibrationState(indices=state.indices[:new_size])

    return state


def run_study(
    *,
    default_runner: Callable[[CalibrationState], int],
    trial_runner: Callable[["object", CalibrationState], int],
    build_config: Callable[[dict[str, Any]], Any],
    initial_state: CalibrationState,
    n_trials: int = 30,
    target_seconds_per_trial: float = 10.0,
    max_calibration_shrinks: int = 3,
    calibration_shrink_factor: float = 0.5,
    min_calibration_size: int = 1,
    sampler: Optional[Any] = None,
    direction: str = "maximize",
    verbose: bool = False,
    seed: Optional[int] = None,
) -> TuneResult:
    """Run a tuning study against a user-provided trial runner.

    The flow is:

    1. Run ``default_runner`` once on ``initial_state`` to time the default
       configuration. If it overruns ``2 * target_seconds_per_trial``, shrink
       the calibration slice and retry up to ``max_calibration_shrinks`` times.
    2. Build an Optuna study and run ``n_trials`` trials. Each trial calls
       ``trial_runner(trial, state)`` which is expected to invoke
       ``trial.suggest_*`` to materialize hyperparameters, run the API once on
       ``state.indices``, and return the number of items processed.
    3. The objective value is items / elapsed-seconds (higher is better when
       ``direction="maximize"``).

    Args:
        default_runner: Callable invoked once during warm-up with the initial
            calibration state. Should run the API with default options and
            return the number of items processed.
        trial_runner: Callable invoked for each Optuna trial. Receives the
            optuna trial and the (possibly shrunk) calibration state, and
            should run the API with the trial-suggested options. Returns the
            number of items processed.
        build_config: Callable that converts a trial parameter dictionary into
            the user-facing configuration object reported in ``best_config``.
        initial_state: Starting calibration state. Mutated only via copy.
        n_trials: Number of Optuna trials to run after warm-up.
        target_seconds_per_trial: Target wall-clock time per trial; warm-up
            shrinks the calibration when the default exceeds twice this value.
        max_calibration_shrinks: Maximum number of warm-up shrink attempts.
        calibration_shrink_factor: Fraction of the previous calibration size
            to keep on each shrink.
        min_calibration_size: Floor on the calibration size during shrink.
        sampler: Optional Optuna sampler. Defaults to ``TPESampler(seed=seed)``.
        direction: Optuna study direction. Default ``"maximize"``.
        verbose: If ``True``, print warm-up and trial diagnostics.
        seed: Seed for the default sampler. Ignored if ``sampler`` is provided.

    Returns:
        :class:`TuneResult` populated with the best configuration found.
    """
    optuna = _require_optuna()

    state = _run_warmup(
        runner=default_runner,
        state=initial_state,
        target_seconds_per_trial=target_seconds_per_trial,
        max_shrinks=max_calibration_shrinks,
        shrink_factor=calibration_shrink_factor,
        min_calibration_size=min_calibration_size,
        verbose=verbose,
    )

    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=seed) if seed is not None else optuna.samplers.TPESampler()
    study = optuna.create_study(direction=direction, sampler=sampler)

    def _objective(trial):
        outcome = _timed_run(lambda current_state: trial_runner(trial, current_state), state)
        if outcome.elapsed_seconds <= 0.0:
            raise optuna.TrialPruned("Non-positive elapsed time")
        throughput = outcome.items / outcome.elapsed_seconds
        if verbose:
            print(
                f"[autotune] trial {trial.number}: params={trial.params} "
                f"elapsed={outcome.elapsed_seconds:.3f}s throughput={throughput:.2f} items/s"
            )
        return throughput

    study.optimize(_objective, n_trials=n_trials)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("Autotune produced no completed trials")

    best_trial = study.best_trial
    return TuneResult(
        best_config=build_config(dict(best_trial.params)),
        best_throughput=float(best_trial.value),
        best_params=dict(best_trial.params),
        calibration_size=len(state.indices),
        n_trials_run=len(completed),
        study=study,
    )


def resolve_search_space(defaults: dict[str, Any], overrides: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Merge user search-space overrides into the wrapper defaults.

    Each value is either a tuple ``(low, high)`` (passed to ``suggest_int``) or
    a list of categorical choices (passed to ``suggest_categorical``). The
    merged dictionary keeps default keys not present in ``overrides``.
    """
    result = dict(defaults)
    if overrides:
        for key, value in overrides.items():
            if key not in result:
                raise KeyError(f"Unknown search-space override '{key}'. Known keys: {sorted(result.keys())}")
            result[key] = value
    return result


def suggest_from_space(trial, name: str, spec: Any) -> Any:
    """Suggest a value from a search-space ``spec`` describing one knob.

    ``spec`` may be:

    - ``(low, high)`` tuple — uniform integer range.
    - ``(low, high, "log")`` tuple — log-uniform integer range.
    - ``(low, high, step)`` tuple — uniform integer range restricted to
      multiples of ``step`` (preserves ordering for TPE, unlike a categorical
      list of the same values).
    - ``list`` of choices — categorical search.

    A literal scalar is returned unchanged (acts as a fixed value rather than
    a search dimension). A 2- or 3-element ``list`` shaped like a range raises
    :class:`TypeError` rather than being treated as categorical.
    """
    if isinstance(spec, tuple) and len(spec) == 2 and all(isinstance(v, int) for v in spec):
        low, high = spec
        return trial.suggest_int(name, int(low), int(high))
    if isinstance(spec, tuple) and len(spec) == 3 and spec[2] == "log":
        low, high, _ = spec
        return trial.suggest_int(name, int(low), int(high), log=True)
    if isinstance(spec, tuple) and len(spec) == 3 and all(isinstance(v, int) for v in spec):
        low, high, step = spec
        if step <= 0:
            raise ValueError(f"Search-space override for {name!r}: step must be a positive integer, got {step!r}.")
        return trial.suggest_int(name, int(low), int(high), step=int(step))
    if isinstance(spec, list) and len(spec) == 2 and all(isinstance(v, int) for v in spec):
        raise TypeError(
            f"Search-space override for {name!r} is a 2-element int list {spec!r}; "
            "use a tuple (low, high) for an integer range, or wrap in a list of "
            "more than two values to request a categorical search."
        )
    if isinstance(spec, list) and len(spec) == 3 and spec[2] == "log":
        raise TypeError(
            f"Search-space override for {name!r} is a 3-element list {spec!r} ending in 'log'; "
            "use a tuple (low, high, 'log') for a log-uniform integer range."
        )
    if isinstance(spec, list) and len(spec) == 3 and all(isinstance(v, int) for v in spec):
        raise TypeError(
            f"Search-space override for {name!r} is a 3-element int list {spec!r}; "
            "use a tuple (low, high, step) for a stepped integer range, or wrap in "
            "a list of more than three values to request a categorical search."
        )
    if isinstance(spec, (list, tuple)):
        choices = list(spec)
        return trial.suggest_categorical(name, choices)
    return spec


def collect_int_from_space(spec: Any) -> int:
    """Return a representative integer from ``spec`` (used for default trials).

    For a uniform integer range ``(low, high)`` the arithmetic midpoint is
    returned. For a log-uniform range ``(low, high, "log")`` the geometric
    midpoint is returned (rounded to the nearest int, clamped to ``[low,
    high]``). For a stepped range ``(low, high, step)`` the arithmetic
    midpoint snapped to the nearest multiple of ``step`` from ``low`` is
    returned, clamped to ``[low, high]``. For a categorical list the first
    listed choice is returned, on the convention that callers list their
    preferred default first. A bare scalar is returned unchanged.
    """
    if isinstance(spec, tuple) and len(spec) == 3 and spec[2] == "log":
        low, high, _ = spec
        low_int = int(low)
        high_int = int(high)
        if low_int <= 0 or high_int <= 0:
            raise ValueError(f"Log-uniform range {spec!r} requires strictly positive bounds.")
        midpoint = int(round(math.sqrt(low_int * high_int)))
        return max(low_int, min(high_int, midpoint))
    if isinstance(spec, tuple) and len(spec) == 3 and all(isinstance(v, int) for v in spec):
        low, high, step = (int(v) for v in spec)
        midpoint = (low + high) // 2
        snapped = low + round((midpoint - low) / step) * step
        return max(low, min(high, snapped))
    if isinstance(spec, tuple) and len(spec) == 2:
        low, high = spec
        return (int(low) + int(high)) // 2
    if isinstance(spec, (list, tuple)) and spec:
        return int(spec[0])
    return int(spec)
