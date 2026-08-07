# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded multiprocessing helpers for benchmark preparation/reference work."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing import Pool
from multiprocessing import TimeoutError as MultiprocessingTimeoutError
from typing import Any, Generic, TypeVar

from bench_utils.timing import Deadline

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class BoundedProcessMapResult(Generic[OutputT]):
    """Completed results from a bounded process map."""

    results: list[OutputT]
    timed_out: bool


def process_map_bounded(
    function: Callable[[InputT], OutputT],
    items: Sequence[InputT],
    *,
    max_workers: int,
    deadline: Deadline,
    initializer: Callable[..., Any] | None = None,
    initargs: tuple[Any, ...] = (),
) -> BoundedProcessMapResult[OutputT]:
    """Map process-safe work and retain completed items when a deadline expires.

    Unlike :func:`tqdm.contrib.concurrent.process_map`, this helper uses the
    deadline as a total wall-clock budget, returns completed results, and
    terminates workers still processing items when the budget expires.
    Callers should submit coarse batches when individual tasks are tiny.
    Result order is completion order; include an index in each result when
    input order must be reconstructed.
    """
    if not items:
        return BoundedProcessMapResult([], False)

    pool = Pool(max_workers, initializer=initializer, initargs=initargs)
    iterator = pool.imap_unordered(function, items, chunksize=1)
    completed: list[OutputT] = []
    timed_out = False
    try:
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
    finally:
        if timed_out:
            pool.terminate()
        else:
            pool.close()
        pool.join()
    return BoundedProcessMapResult(completed, timed_out)
