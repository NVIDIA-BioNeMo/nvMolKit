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

"""Helpers for selecting and resizing calibration slices used during autotune."""

from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence


def auto_subsample(
    workload_size: int,
    *,
    fraction: float = 0.1,
    max_size: int = 2000,
    min_size: int = 1,
    seed: int = 0,
) -> list[int]:
    """Choose a representative slice of indices from a workload.

    The slice size is ``min(max_size, max(min_size, round(fraction * workload_size)))``
    and is shuffled deterministically with ``seed`` so trials sample roughly
    uniformly across the workload.
    """
    if workload_size <= 0:
        raise ValueError("workload_size must be positive")
    target = min(max_size, max(min_size, int(round(fraction * workload_size))))
    target = min(target, workload_size)
    rng = random.Random(seed)
    indices = list(range(workload_size))
    rng.shuffle(indices)
    return indices[:target]


def normalize_calibration_set(
    calibration_set: Optional[Iterable[int]],
    workload_size: int,
    *,
    fraction: float = 0.1,
    max_size: int = 2000,
    seed: int = 0,
) -> list[int]:
    """Return calibration indices from an explicit set or auto-sample.

    Explicitly-provided indices are returned as given; order and duplicates
    are preserved. Out-of-range entries raise :class:`IndexError`.

    Args:
        calibration_set: User-provided iterable of indices, or ``None`` to
            auto-subsample.
        workload_size: Total size of the user workload.
        fraction: Auto-subsample fraction when ``calibration_set`` is ``None``.
        max_size: Auto-subsample cap when ``calibration_set`` is ``None``.
        seed: Seed for auto-subsampling.
    """
    if calibration_set is None:
        return auto_subsample(workload_size, fraction=fraction, max_size=max_size, seed=seed)

    indices = [int(i) for i in calibration_set]
    if not indices:
        raise ValueError("calibration_set must be non-empty")
    for idx in indices:
        if idx < 0 or idx >= workload_size:
            raise IndexError(f"Calibration index {idx} out of range for workload size {workload_size}")
    return indices


def shrink(indices: Sequence[int], factor: float = 0.5, *, min_size: int = 1) -> list[int]:
    """Return a prefix of ``indices`` sized ``round(len(indices) * factor)``.

    A floor of ``min_size`` is enforced; the result is never empty when the
    input is non-empty.
    """
    if factor <= 0.0 or factor >= 1.0:
        raise ValueError("factor must be in (0, 1)")
    new_size = max(min_size, int(round(len(indices) * factor)))
    new_size = min(new_size, len(indices))
    return list(indices[:new_size])
