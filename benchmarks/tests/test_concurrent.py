# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from bench_utils.concurrent import process_map_bounded, process_pool
from bench_utils.timing import Deadline


def _sleep_and_return(item):
    delay, value = item
    time.sleep(delay)
    return value


def _fail_initializer():
    raise ValueError("bad initialization")


def _fail_task(_):
    raise ValueError("bad task")


def _initialize_multiplier(multiplier):
    global _multiplier  # noqa: PLW0603 - process-local initializer state
    _multiplier = multiplier


def _multiply(value):
    return value * _multiplier


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
    with process_pool(1) as pool:
        workers = list(pool._pool)
        result = process_map_bounded(
            _sleep_and_return,
            [(0.001, 1), (2.0, 2)],
            max_workers=1,
            deadline=Deadline(0.25),
            pool=pool,
        )
    assert result.results == [1]
    assert result.timed_out
    assert all(not worker.is_alive() for worker in workers)


def test_process_map_bounded_counts_fast_results_behind_a_slow_first_task():
    with process_pool(3) as pool:
        result = process_map_bounded(
            _sleep_and_return,
            [(2.0, 1), (0.001, 2), (0.001, 3)],
            max_workers=3,
            deadline=Deadline(0.25),
            pool=pool,
        )
    assert sorted(result.results) == [2, 3]
    assert result.timed_out


@pytest.mark.parametrize("max_workers", [1, 2, 3])
def test_process_pool_initializes_reuses_and_closes_workers(max_workers):
    with process_pool(max_workers, _initialize_multiplier, (7,)) as pool:
        workers = list(pool._pool)
        for values in ([1, 2, 3], [], [4, 5]):
            result = process_map_bounded(_multiply, values, max_workers=max_workers, deadline=Deadline(0), pool=pool)
            assert sorted(result.results) == [value * 7 for value in values]
            assert not result.timed_out
    assert all(not worker.is_alive() for worker in workers)


def test_process_pool_reports_initializer_failure():
    with pytest.raises(RuntimeError, match="bad initialization"), process_pool(2, _fail_initializer):
        pytest.fail("failed workers must not enter the context")


def test_process_pool_cleans_up_after_task_failure():
    with pytest.raises(ValueError, match="bad task"), process_pool(2) as pool:
        workers = list(pool._pool)
        process_map_bounded(_fail_task, [1, 2], max_workers=2, deadline=Deadline(0), pool=pool)
    assert all(not worker.is_alive() for worker in workers)


def test_process_pool_cleans_up_after_caller_failure():
    with pytest.raises(ValueError, match="caller failed"), process_pool(2) as pool:
        workers = list(pool._pool)
        raise ValueError("caller failed")
    assert all(not worker.is_alive() for worker in workers)
