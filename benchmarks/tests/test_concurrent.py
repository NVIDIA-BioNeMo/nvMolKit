# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from bench_utils.concurrent import process_map_bounded, ready_spawn_pool
from bench_utils.timing import Deadline


def _sleep_and_return(item):
    delay, value = item
    time.sleep(delay)
    return value


def test_process_map_bounded_returns_all_results_without_deadline():
    result = process_map_bounded(
        _sleep_and_return,
        [(0.001, 1), (0.001, 2)],
        max_workers=2,
        deadline=Deadline(0.0),
    )
    assert sorted(result.results) == [1, 2]
    assert not result.timed_out


def test_process_map_bounded_retains_completed_results_on_timeout():
    # Start workers before the deadline, as benchmarks do, so spawn start-up is not part of the budget.
    pool = ready_spawn_pool(1)
    result = process_map_bounded(
        _sleep_and_return,
        [(0.001, 1), (2.0, 2)],
        max_workers=1,
        deadline=Deadline(0.25),
        pool=pool,
    )
    assert result.results == [1]
    assert result.timed_out


def test_process_map_bounded_reuses_a_ready_pool_across_calls():
    pool = ready_spawn_pool(2)
    try:
        first = process_map_bounded(_sleep_and_return, [(0.001, 1)], max_workers=2, deadline=Deadline(0.0), pool=pool)
        second = process_map_bounded(_sleep_and_return, [(0.001, 2)], max_workers=2, deadline=Deadline(0.0), pool=pool)
    finally:
        pool.terminate()
        pool.join()
    assert first.results == [1]
    assert second.results == [2]


def _fail_initializer():
    raise ValueError("bad initialization")


def _fail_task(_):
    raise ValueError("bad task")


def _slow_initializer():
    time.sleep(10)


def test_ready_spawn_pool_reports_initializer_failure():
    with pytest.raises(RuntimeError, match="bad initialization"):
        ready_spawn_pool(2, _fail_initializer)


def test_ready_spawn_pool_bounds_startup_wait():
    with pytest.raises(TimeoutError, match="initialize"):
        ready_spawn_pool(1, _slow_initializer, startup_timeout=0.1)


@pytest.mark.parametrize("workers, timeout", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_ready_spawn_pool_rejects_invalid_settings(workers, timeout):
    with pytest.raises(ValueError):
        ready_spawn_pool(workers, startup_timeout=timeout)


def test_bounded_map_empty_input_does_not_start_workers(monkeypatch):
    monkeypatch.setattr("bench_utils.concurrent.ready_spawn_pool", lambda *args: pytest.fail("unexpected pool"))
    result = process_map_bounded(_fail_task, [], max_workers=1, deadline=Deadline(0))
    assert result.results == []
    assert not result.timed_out


def test_bounded_map_task_failure_terminates_borrowed_pool():
    pool = ready_spawn_pool(2)
    workers = list(pool._pool)
    with pytest.raises(ValueError, match="bad task"):
        process_map_bounded(_fail_task, [1, 2], max_workers=2, deadline=Deadline(0), pool=pool)
    assert all(not worker.is_alive() for worker in workers)


def test_bounded_map_timeout_terminates_workers():
    pool = ready_spawn_pool(1)
    workers = list(pool._pool)
    result = process_map_bounded(_sleep_and_return, [(10, 1)], max_workers=1, deadline=Deadline(0.05), pool=pool)
    assert result.results == []
    assert result.timed_out
    assert all(not worker.is_alive() for worker in workers)
