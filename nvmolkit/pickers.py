# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-accelerated diversity selection.

Each picker has a matrix form that takes a precomputed distance matrix and a
``fused_`` form that computes distances from fingerprints or molecules as needed.
"""

import operator
from typing import Sequence

import torch

from nvmolkit import _clustering
from nvmolkit._distance_inputs import _packed_metric_name, _prepare_distance_matrix, _prepare_fused_input
from nvmolkit.clustering import OutputMode, _validate_output
from nvmolkit.similarity import Metric
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

    Distances are compared with ``cutoff`` in float32, so a distance within
    float32 rounding of the cutoff can be classified differently than by
    RDKit's ``LeaderPicker``, which compares in double precision. For Tanimoto
    distances this can happen at cutoffs that float32 cannot represent exactly,
    such as 0.3.

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


def fused_leader(
    x,
    cutoff: float,
    *,
    metric: Metric = "tanimoto",
    pick_size: int = 0,
    first_picks: Sequence[int] = (),
    stream: torch.cuda.Stream | None = None,
    output: OutputMode = OutputMode.DEVICE,
) -> AsyncGpuResult | tuple[int, ...]:
    """Select leaders by sphere exclusion, computing distances as needed.

    Equivalent to :func:`leader` on the matrix of ``1 - similarity`` values,
    with memory that scales as ``O(N)``. Distances are computed in float32, so
    results can differ slightly from RDKit's double-precision results.

    Args:
        x: Packed int32 or uint32 fingerprints of shape ``(N, num_words)``.
        cutoff: Inclusive exclusion distance in ``[0, 1]``, rounded to float32.
        metric: Similarity metric. :class:`~nvmolkit.similarity.AAPMetric` is not
            yet supported.
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
    resolved, inputs, active_stream = _prepare_fused_input(x, metric, stream, "fused_leader")
    pick_size = operator.index(pick_size)
    first_picks = _index_tuple("first_picks", first_picks)
    with torch.cuda.stream(active_stream):
        result = _clustering.fused_leader(
            inputs.__cuda_array_interface__,
            cutoff,
            _packed_metric_name(resolved),
            pick_size,
            first_picks,
            active_stream.cuda_stream,
        )
        return _resolve_selection_output(result, output)
