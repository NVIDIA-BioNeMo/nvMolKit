# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared bench_utils timing helpers."""

import math
import time

import pytest
from bench_utils.timing import Deadline, throughput_per_s, time_it_bounded


@pytest.mark.parametrize("max_seconds", [0.0, -1.0])
def test_deadline_non_positive_is_disabled(max_seconds):
    deadline = Deadline(max_seconds)
    assert not deadline.active
    assert not deadline.expired()


def test_deadline_expires_after_budget():
    deadline = Deadline(0.01)
    assert deadline.active
    assert not deadline.expired()
    time.sleep(0.05)
    assert deadline.expired()


def test_deadline_independent_instances():
    long = Deadline(10.0)
    short = Deadline(0.01)
    time.sleep(0.05)
    assert short.expired()
    assert not long.expired()


def test_throughput_per_s_simple_conversion():
    # 100 items in 500ms -> 200 items/s
    assert throughput_per_s(100, 500.0) == pytest.approx(200.0)


@pytest.mark.parametrize("elapsed_ms", [0.0, -5.0])
def test_throughput_per_s_non_positive_elapsed_returns_nan(elapsed_ms):
    assert math.isnan(throughput_per_s(100, elapsed_ms))


def test_time_it_bounded_runs_to_completion_when_progress_full():
    call_count = [0]

    def run(_deadline):
        call_count[0] += 1
        time.sleep(0.001)

    avg_ms, std_ms, progress = time_it_bounded(
        run, runs=3, max_seconds=0.0, progress_getter=lambda: 10, progress_target=10
    )

    assert call_count[0] == 3
    assert avg_ms > 0
    assert std_ms >= 0
    assert progress == 10


def test_time_it_bounded_stops_after_partial_run():
    call_count = [0]
    progress = [10]

    def run(_deadline):
        call_count[0] += 1
        progress[0] = 3

    avg_ms, std_ms, last_progress = time_it_bounded(
        run, runs=5, max_seconds=0.0, progress_getter=lambda: progress[0], progress_target=10
    )

    assert call_count[0] == 1
    assert last_progress == 3
    assert std_ms == 0.0
    assert avg_ms > 0


def test_time_it_bounded_stops_when_budget_exhausted_between_runs():
    call_count = [0]

    def run(_deadline):
        call_count[0] += 1
        time.sleep(0.05)

    # 5 runs * 50ms = 250ms total, but budget is only 60ms.
    # Run 1 completes at ~50ms (deadline check before run 2 still passes), run 2
    # completes at ~100ms, and the deadline check before run 3 stops the loop.
    avg_ms, _std_ms, _progress = time_it_bounded(
        run, runs=5, max_seconds=0.06, progress_getter=lambda: 1, progress_target=1
    )

    assert 1 <= call_count[0] < 5
    assert avg_ms > 0


def test_time_it_bounded_zero_runs_returns_zero():
    avg_ms, std_ms, progress = time_it_bounded(
        lambda _deadline: None, runs=0, max_seconds=0.0, progress_getter=lambda: 0, progress_target=1
    )
    assert avg_ms == 0.0
    assert std_ms == 0.0
    assert progress == 0


def test_time_it_bounded_stddev_positive_for_multiple_completed_runs():
    delays = iter([0.001, 0.02, 0.001])

    def run(_deadline):
        time.sleep(next(delays))

    _avg_ms, std_ms, _progress = time_it_bounded(
        run, runs=3, max_seconds=0.0, progress_getter=lambda: 1, progress_target=1
    )
    assert std_ms > 0.0


def test_time_it_bounded_shared_deadline_caps_inner_loop():
    """Verify ``time_it_bounded`` exposes a single shared :class:`Deadline`.

    The ``run`` callback honours the budget mid-iteration, and a second call
    receives the same (already-elapsed) deadline rather than a fresh one,
    which is the property that makes ``max_seconds`` a true total cap.
    """
    iterations_per_run: list[int] = []
    progress = [0]

    def run(deadline):
        n_done = 0
        for _ in range(1000):
            if deadline.expired():
                break
            time.sleep(0.005)
            n_done += 1
        iterations_per_run.append(n_done)
        progress[0] = n_done

    _avg_ms, _std_ms, _progress = time_it_bounded(
        run, runs=5, max_seconds=0.05, progress_getter=lambda: progress[0], progress_target=1000
    )

    assert iterations_per_run, "run should have been invoked at least once"
    # The first run consumes essentially the whole budget; any subsequent call
    # must see an already-expired deadline and exit immediately.
    for later_count in iterations_per_run[1:]:
        assert later_count == 0
