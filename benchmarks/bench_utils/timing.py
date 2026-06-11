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


def time_it(func: Callable, runs: int = 3, warmups: int = 1, gpu_sync: bool = False) -> TimingResult:
    """Time a callable with warmup iterations and optional CUDA synchronization.

    Args:
        func: Zero-argument callable to benchmark.
        runs: Number of timed iterations.
        warmups: Number of untimed warmup iterations.
        gpu_sync: If True, call torch.cuda.synchronize() before and after each
                  timed iteration to ensure GPU work is included in the measurement.

    Returns:
        A TimingResult with per-iteration times in milliseconds.
    """
    if runs <= 0:
        raise ValueError(f"runs must be positive, got {runs}")

    if gpu_sync:
        import torch

        sync = torch.cuda.synchronize
    else:

        def sync() -> None:
            pass

    for _ in range(warmups):
        func()
        sync()

    times_ms = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter()
        func()
        sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return TimingResult(times_ms=times_ms)


class Deadline:
    """Wall-clock budget that benchmark loops can poll for early termination.

    A ``max_seconds`` of ``0`` (or negative) disables the budget, in which
    case :meth:`expired` always returns ``False``. Construction starts the
    clock; pass the same instance to nested loops to share one deadline.
    """

    def __init__(self, max_seconds: float) -> None:
        self._end: float | None = time.perf_counter() + max_seconds if max_seconds > 0 else None

    def expired(self) -> bool:
        return self._end is not None and time.perf_counter() >= self._end

    @property
    def active(self) -> bool:
        """``True`` when a real budget is being enforced."""
        return self._end is not None


def throughput_per_s(items: float, elapsed_ms: float) -> float:
    """Items per second from a millisecond count; ``NaN`` if ``elapsed_ms <= 0``."""
    if elapsed_ms <= 0:
        return float("nan")
    return items / (elapsed_ms / 1000.0)


def time_it_bounded(
    run: Callable[[Deadline], None],
    runs: int,
    max_seconds: float,
    progress_getter: Callable[[], int],
    progress_target: int,
) -> tuple[float, float, int]:
    """Repeat ``run`` up to ``runs`` times, stopping early on budget exhaustion.

    A single :class:`Deadline` covering the whole call is constructed from
    ``max_seconds`` and passed to ``run`` on every invocation; the closure
    must poll it inside its inner work loop to honour the budget mid-run.
    After each invocation, ``progress_getter()`` reports how much of the
    workload was actually completed; a value below ``progress_target`` is
    treated as a partial run and further iterations are skipped.

    Returns ``(avg_ms, std_ms, last_progress)``. ``avg`` and ``std`` are
    computed only over runs that completed end-to-end; if no full run
    finished, the single partial timing is returned with ``std=0``.
    """
    deadline = Deadline(max_seconds)
    completed_times_ms: list[float] = []
    partial_time_ms: float | None = None
    last_progress = 0
    for _ in range(runs):
        if deadline.expired():
            break
        start = time.perf_counter()
        run(deadline)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        last_progress = progress_getter()
        if last_progress < progress_target:
            partial_time_ms = elapsed_ms
            break
        completed_times_ms.append(elapsed_ms)
    if completed_times_ms:
        avg_ms = statistics.mean(completed_times_ms)
        std_ms = statistics.pstdev(completed_times_ms) if len(completed_times_ms) > 1 else 0.0
        return avg_ms, std_ms, last_progress
    if partial_time_ms is not None:
        return partial_time_ms, 0.0, last_progress
    return 0.0, 0.0, last_progress


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
