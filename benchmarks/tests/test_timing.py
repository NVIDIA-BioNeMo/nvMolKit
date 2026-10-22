# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared bench_utils timing helpers."""

import math
import time

import pytest
from bench_utils.timing import Deadline, throughput_per_s, time_it


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


def test_deadline_reports_remaining_seconds():
    disabled = Deadline(0.0)
    active = Deadline(1.0)

    assert disabled.remaining_seconds() is None
    assert 0.0 < active.remaining_seconds() <= 1.0


def test_throughput_per_s_simple_conversion():
    # 100 items in 500ms -> 200 items/s
    assert throughput_per_s(100, 500.0) == pytest.approx(200.0)


@pytest.mark.parametrize("elapsed_ms", [0.0, -5.0])
def test_throughput_per_s_non_positive_elapsed_returns_nan(elapsed_ms):
    assert math.isnan(throughput_per_s(100, elapsed_ms))


def test_time_it_deadline_mode_reports_complete_progress():
    seen_deadlines = []
    progress = [0]

    def run(deadline):
        seen_deadlines.append(deadline)
        progress[0] = 4

    timing = time_it(
        run,
        runs=3,
        warmups=0,
        max_seconds=0.0,
        progress_getter=lambda: progress[0],
        progress_target=4,
    )

    assert len(timing.times_ms) == 3
    assert timing.progress == timing.progress_target == 4
    assert not timing.truncated
    assert len({id(deadline) for deadline in seen_deadlines}) == 1


def test_time_it_runs_setup_before_each_iteration_outside_timing(monkeypatch):
    events = []
    clock = iter([10.0, 10.25])

    monkeypatch.setattr("bench_utils.timing.time.perf_counter", lambda: next(clock))

    timing = time_it(
        lambda: events.append("run"),
        runs=1,
        warmups=1,
        setup=lambda: events.append("setup"),
    )

    assert events == ["setup", "run", "setup", "run"]
    assert timing.times_ms == [250.0]


@pytest.mark.parametrize("max_seconds, expected_calls", [(0.0, 4), (-1.0, 4), (10.0, 1)])
def test_time_it_skips_full_warmups_only_for_positive_caps(max_seconds, expected_calls):
    deadlines = []
    setups = []
    timing = time_it(
        deadlines.append,
        runs=1,
        warmups=3,
        setup=lambda: setups.append(True),
        max_seconds=max_seconds,
        progress_getter=lambda: 1,
        progress_target=1,
    )
    assert len(deadlines) == len(setups) == expected_calls
    assert all(deadline.active == (max_seconds > 0) for deadline in deadlines)
    assert len(timing.times_ms) == 1
    assert timing.progress == 1


def test_time_it_deadline_mode_retains_first_partial_sample():
    progress = [0]
    calls = [0]

    def run(_deadline):
        calls[0] += 1
        progress[0] = 2

    timing = time_it(
        run,
        runs=3,
        warmups=0,
        max_seconds=0.0,
        progress_getter=lambda: progress[0],
        progress_target=4,
    )

    assert calls[0] == 1
    assert len(timing.times_ms) == 1
    assert timing.progress == 2
    assert timing.progress_target == 4
    assert timing.truncated


def test_time_it_deadline_mode_discards_later_partial_sample():
    progresses = iter([4, 4, 2])
    progress = [0]

    def run(_deadline):
        progress[0] = next(progresses)

    timing = time_it(
        run,
        runs=3,
        warmups=0,
        max_seconds=0.0,
        progress_getter=lambda: progress[0],
        progress_target=4,
    )

    assert len(timing.times_ms) == 2
    assert timing.progress == 4
    assert not timing.truncated


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_seconds": 1.0},
        {"progress_getter": lambda: 1},
        {"progress_target": 1},
        {"max_seconds": 1.0, "progress_getter": lambda: 1},
    ],
)
def test_time_it_rejects_incomplete_deadline_configuration(kwargs):
    with pytest.raises(ValueError, match="must be provided together"):
        time_it(lambda: None, runs=1, warmups=0, **kwargs)


def test_time_it_rejects_negative_warmups():
    with pytest.raises(ValueError, match="warmups must be non-negative"):
        time_it(lambda: None, runs=1, warmups=-1)


def test_time_it_rejects_negative_progress_target():
    with pytest.raises(ValueError, match="progress_target must be non-negative"):
        time_it(
            lambda _deadline: None,
            runs=1,
            warmups=0,
            max_seconds=0.0,
            progress_getter=lambda: 0,
            progress_target=-1,
        )


@pytest.mark.parametrize("progress", [-1, 3])
def test_time_it_rejects_progress_outside_target(progress):
    with pytest.raises(ValueError, match="progress must be between"):
        time_it(
            lambda _deadline: None,
            runs=1,
            warmups=0,
            max_seconds=0.0,
            progress_getter=lambda: progress,
            progress_target=2,
        )


def test_time_it_stops_when_budget_exhausted_between_runs():
    call_count = [0]

    def run(_deadline):
        call_count[0] += 1
        time.sleep(0.05)

    # 5 runs * 50ms = 250ms total, but budget is only 60ms.
    # Run 1 completes at ~50ms (deadline check before run 2 still passes), run 2
    # completes at ~100ms, and the deadline check before run 3 stops the loop.
    timing = time_it(
        run,
        runs=5,
        warmups=0,
        max_seconds=0.06,
        progress_getter=lambda: 1,
        progress_target=1,
    )

    assert 1 <= call_count[0] < 5
    assert timing.mean_ms > 0


def test_time_it_runs_first_sample_when_deadline_already_expired(monkeypatch):
    monkeypatch.setattr(Deadline, "expired", lambda _self: True)
    call_count = [0]

    def run(_deadline):
        call_count[0] += 1

    timing = time_it(
        run,
        runs=3,
        warmups=0,
        max_seconds=1.0,
        progress_getter=lambda: call_count[0],
        progress_target=1,
    )

    assert call_count[0] == 1
    assert len(timing.times_ms) == 1
    assert timing.progress == 1


def test_time_it_stddev_positive_for_multiple_completed_runs():
    delays = iter([0.001, 0.02, 0.001])

    def run(_deadline):
        time.sleep(next(delays))

    timing = time_it(
        run,
        runs=3,
        warmups=0,
        max_seconds=0.0,
        progress_getter=lambda: 1,
        progress_target=1,
    )
    assert timing.std_ms > 0.0


def test_time_it_shared_deadline_caps_inner_loop():
    """Verify ``time_it`` exposes a single shared :class:`Deadline`.

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

    timing = time_it(
        run,
        runs=5,
        warmups=0,
        max_seconds=0.05,
        progress_getter=lambda: progress[0],
        progress_target=1000,
    )

    assert iterations_per_run, "run should have been invoked at least once"
    assert timing.truncated
    # The first run consumes essentially the whole budget; any subsequent call
    # must see an already-expired deadline and exit immediately.
    for later_count in iterations_per_run[1:]:
        assert later_count == 0
