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

"""Shared timing utilities for nvMolKit benchmarks."""

import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TimingResult:
    """Holds timing results from a benchmark run."""

    times_ms: list[float] = field(default_factory=list)
    progress: int | None = None
    progress_target: int | None = None

    @property
    def median_ms(self) -> float:
        """Median time in milliseconds."""
        return statistics.median(self.times_ms)

    @property
    def mean_ms(self) -> float:
        """Mean time in milliseconds."""
        return statistics.mean(self.times_ms)

    @property
    def std_ms(self) -> float:
        """Sample standard deviation in milliseconds."""
        if len(self.times_ms) < 2:
            return 0.0
        return statistics.stdev(self.times_ms)

    @property
    def median_s(self) -> float:
        """Median time in seconds."""
        return self.median_ms / 1000.0

    @property
    def truncated(self) -> bool:
        """Whether a bounded measurement stopped before its progress target."""
        return self.progress is not None and self.progress_target is not None and self.progress < self.progress_target


def time_it(
    func: Callable,
    runs: int = 3,
    warmups: int = 1,
    gpu_sync: bool = False,
    *,
    setup: Callable[[], None] | None = None,
    max_seconds: float | None = None,
    progress_getter: Callable[[], int] | None = None,
    progress_target: int | None = None,
) -> TimingResult:
    """Time a callable, optionally enforcing a deadline and tracking progress.

    Args:
        func: Zero-argument callable to benchmark. When ``max_seconds`` is
              provided, the callable instead receives the shared
              :class:`Deadline` as its sole argument.
        runs: Number of timed iterations.
        warmups: Number of untimed warmup iterations. Skipped when a positive
            ``max_seconds`` caps the measurement.
        gpu_sync: If True, call torch.cuda.synchronize() before and after each
                  timed iteration to ensure GPU work is included in the measurement.
        setup: Optional preparation called before every warmup and timed
               iteration, outside the measured interval.
        max_seconds: Total wall-clock budget shared by all timed iterations.
                     ``0`` disables expiry while retaining progress tracking.
        progress_getter: Returns the work completed by the latest iteration.
        progress_target: Work required for an iteration to be complete.

    Returns:
        A :class:`TimingResult` with retained per-iteration timings and, for a
        bounded measurement, progress from the same retained sample set.

        Complete samples are retained together and a later partial sample is
        discarded. If the first sample is partial, only its timing and actual
        progress are returned. This prevents full-run timing statistics from
        being paired with output or throughput from a different partial run.
    """
    if runs <= 0:
        raise ValueError(f"runs must be positive, got {runs}")
    if warmups < 0:
        raise ValueError(f"warmups must be non-negative, got {warmups}")

    bounded = max_seconds is not None
    if bounded != (progress_getter is not None) or bounded != (progress_target is not None):
        raise ValueError("max_seconds, progress_getter, and progress_target must be provided together")
    if progress_target is not None and progress_target < 0:
        raise ValueError(f"progress_target must be non-negative, got {progress_target}")

    if gpu_sync:
        import torch

        sync = torch.cuda.synchronize
    else:

        def sync() -> None:
            pass

    warmup_deadline = Deadline(0.0)
    for _ in range(0 if bounded and max_seconds > 0 else warmups):
        if setup is not None:
            setup()
        if bounded:
            func(warmup_deadline)
        else:
            func()
        sync()

    deadline = Deadline(max_seconds) if bounded else None
    complete_times_ms: list[float] = []
    partial_time_ms: float | None = None
    last_progress: int | None = None
    for run_idx in range(runs):
        if run_idx > 0 and deadline is not None and deadline.expired():
            break
        if setup is not None:
            setup()
        sync()
        t0 = time.perf_counter()
        if deadline is not None:
            func(deadline)
        else:
            func()
        sync()
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        if progress_getter is not None and progress_target is not None:
            last_progress = progress_getter()
            if not 0 <= last_progress <= progress_target:
                raise ValueError(f"progress must be between 0 and {progress_target}, got {last_progress}")
            if last_progress < progress_target:
                partial_time_ms = elapsed_ms
                break
        complete_times_ms.append(elapsed_ms)

    if not bounded:
        return TimingResult(times_ms=complete_times_ms)
    if complete_times_ms:
        return TimingResult(
            times_ms=complete_times_ms,
            progress=progress_target,
            progress_target=progress_target,
        )
    if partial_time_ms is None or last_progress is None:
        raise RuntimeError("bounded timing completed without recording a sample")
    return TimingResult(
        times_ms=[partial_time_ms],
        progress=last_progress,
        progress_target=progress_target,
    )


class Deadline:
    """Wall-clock budget that benchmark loops can poll for early termination.

    A ``max_seconds`` of ``0`` (or negative) disables the budget, in which
    case :meth:`expired` always returns ``False``. Construction starts the
    clock; pass the same instance to nested loops to share one deadline.
    """

    def __init__(self, max_seconds: float) -> None:
        """Start a deadline lasting ``max_seconds``; non-positive disables it."""
        self._end: float | None = time.perf_counter() + max_seconds if max_seconds > 0 else None

    def expired(self) -> bool:
        """Return whether the active deadline has elapsed."""
        return self._end is not None and time.perf_counter() >= self._end

    @property
    def active(self) -> bool:
        """``True`` when a real budget is being enforced."""
        return self._end is not None

    def remaining_seconds(self) -> float | None:
        """Seconds remaining, or ``None`` when the deadline is disabled."""
        if self._end is None:
            return None
        return max(0.0, self._end - time.perf_counter())


def throughput_per_s(items: float, elapsed_ms: float) -> float:
    """Items per second from a millisecond count; ``NaN`` if ``elapsed_ms <= 0``."""
    if elapsed_ms <= 0:
        return float("nan")
    return items / (elapsed_ms / 1000.0)


def add_rdkit_max_seconds_arg(parser: argparse.ArgumentParser, *, extra_help: str = "") -> None:
    """Register the shared ``--rdkit_max_seconds`` CLI flag.

    ``extra_help`` is appended to the standard help string so individual
    benchmarks can describe how partial-run semantics apply to their RDKit
    code path (e.g. per-molecule vs. per-query truncation).
    """
    base_help = (
        "Stop the RDKit comparison after this many wall-clock seconds and "
        "report throughput on the work actually completed. 0 disables the "
        "cap and runs the full workload (default: 0)."
    )
    parser.add_argument(
        "--rdkit_max_seconds",
        type=float,
        default=0.0,
        help=f"{base_help} {extra_help}".rstrip(),
    )
