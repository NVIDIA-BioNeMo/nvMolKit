# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded multiprocessing helpers for benchmark preparation/reference work."""

import multiprocessing
import multiprocessing.pool
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import TimeoutError as MultiprocessingTimeoutError
from queue import Empty
from typing import Any, Generic, TypeVar

from bench_utils.timing import Deadline

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class BoundedProcessMapResult(Generic[OutputT]):
    """Completed results from a bounded process map."""

    results: list[OutputT]
    timed_out: bool


def _initialize_worker(ready_queue, initializer, initargs) -> None:
    try:
        if initializer is not None:
            initializer(*initargs)
    except BaseException as error:
        ready_queue.put((False, repr(error)))
        return
    ready_queue.put((True, os.getpid()))


@contextmanager
def process_pool(
    max_workers: int,
    initializer: Callable[..., Any] | None = None,
    initargs: tuple[Any, ...] = (),
    *,
    startup_timeout: float = 120.0,
) -> Iterator[multiprocessing.pool.Pool]:
    """Manage spawned workers, including initialization before the context body.

    Initialization failures or a stalled startup terminate the pool and raise. The timeout bounds
    total startup time; it is separate from the benchmark's measured work deadline.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if startup_timeout <= 0:
        raise ValueError("startup_timeout must be positive")
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    pool = context.Pool(max_workers, initializer=_initialize_worker, initargs=(ready_queue, initializer, initargs))
    deadline = Deadline(startup_timeout)
    try:
        for _ in range(max_workers):
            try:
                succeeded, detail = ready_queue.get(timeout=deadline.remaining_seconds())
            except Empty as error:
                raise TimeoutError("Timed out waiting for benchmark workers to initialize") from error
            if not succeeded:
                raise RuntimeError(f"Benchmark worker initialization failed: {detail}")
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    finally:
        ready_queue.close()
        ready_queue.join_thread()
    try:
        yield pool
    finally:
        pool.terminate()
        pool.join()


def process_map_bounded(
    function: Callable[[InputT], OutputT],
    items: Sequence[InputT],
    *,
    max_workers: int,
    deadline: Deadline,
    initializer: Callable[..., Any] | None = None,
    initargs: tuple[Any, ...] = (),
    pool: multiprocessing.pool.Pool | None = None,
) -> BoundedProcessMapResult[OutputT]:
    """Map process-safe work and retain completed items when a deadline expires.

    Unlike :func:`tqdm.contrib.concurrent.process_map`, this helper uses the
    deadline as a total wall-clock budget, returns completed results, and
    terminates workers still processing items when the budget expires.
    Callers should submit coarse batches when individual tasks are tiny.
    Result order is completion order; include an index in each result when
    input order must be reconstructed.

    Pass ``pool`` (from :func:`process_pool`) to reuse workers started outside the timed region; it
    is left open unless the deadline expires, in which case it is terminated and must not be reused.
    """
    if not items:
        return BoundedProcessMapResult([], False)

    if pool is None:
        with process_pool(max_workers, initializer, initargs) as workers:
            return process_map_bounded(function, items, max_workers=max_workers, deadline=deadline, pool=workers)
    completed: list[OutputT] = []
    timed_out = False
    try:
        iterator = pool.imap_unordered(function, items, chunksize=1)
        for _ in items:
            remaining = deadline.remaining_seconds()
            if remaining is not None:
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    completed.append(iterator.next(timeout=remaining))
                except MultiprocessingTimeoutError:
                    timed_out = True
                    break
            else:
                completed.append(next(iterator))
    except BaseException:
        timed_out = True
        raise
    finally:
        if timed_out:
            pool.terminate()
            pool.join()
    return BoundedProcessMapResult(completed, timed_out)
