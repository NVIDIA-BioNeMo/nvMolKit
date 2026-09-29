# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-accelerated diversity selection.

Pickers select an ordered subset of items from a precomputed distance matrix.
"""

import operator
from typing import Sequence

import torch

from nvmolkit import _clustering
from nvmolkit._distance_inputs import _prepare_distance_matrix
from nvmolkit.clustering import OutputMode, _validate_output
from nvmolkit.types import ArrayInput, AsyncGpuResult


def _index_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    try:
        return tuple(operator.index(value) for value in values)
    except TypeError:
        raise TypeError(f"{name} must be a sequence of integers") from None


def _resolve_selection_output(result, output: OutputMode):
    indices = AsyncGpuResult(result)
    if output is OutputMode.DEVICE:
        return indices
    return tuple(int(index) for index in indices.numpy())


def leader(
    distance_matrix: ArrayInput,
    cutoff: float,
    *,
    pick_size: int = 0,
    first_picks: Sequence[int] = (),
    stream: torch.cuda.Stream | None = None,
    output: OutputMode = OutputMode.DEVICE,
) -> AsyncGpuResult | tuple[int, ...]:
    """Select leaders from a distance matrix by sphere exclusion.

    Candidates are visited in input order. Each candidate that has not been
    excluded becomes a leader and excludes every remaining candidate within
    ``cutoff`` of it, as in RDKit's ``LeaderPicker``.

    Args:
        distance_matrix: Square float32 or float64 matrix of shape ``(N, N)``.
            Element ``[i, j]`` is the distance from item ``i`` to item ``j``.
            Values are converted to float32 for comparisons. Values that
            overflow float32 during conversion become infinity.
        cutoff: Inclusive exclusion distance, rounded to float32. Must be
            between zero and the largest finite float32 value.
        pick_size: Maximum number of leaders, or ``0`` for no limit. All
            ``first_picks`` are retained even if they exceed this limit.
        first_picks: Unique indices selected as leaders, in order, before the
            input-order pass.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Result representation.

    Returns:
        An :class:`~nvmolkit.types.AsyncGpuResult` of int32 indices for
        ``OutputMode.DEVICE``, or a tuple of indices for ``OutputMode.RDKIT``.
    """
    _validate_output(output)
    matrix, active_stream = _prepare_distance_matrix(distance_matrix, stream)
    with torch.cuda.stream(active_stream):
        result = _clustering.leader(
            matrix.__cuda_array_interface__,
            cutoff,
            operator.index(pick_size),
            _index_tuple("first_picks", first_picks),
            active_stream.cuda_stream,
        )
        return _resolve_selection_output(result, output)
