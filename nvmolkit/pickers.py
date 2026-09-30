# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-accelerated diversity selection.

Each picker has a matrix form that takes a precomputed distance matrix and a
``fused_`` form that computes distances from fingerprints or molecules as needed.
"""

import operator
from typing import Sequence

import numpy as np
import torch

from nvmolkit import _clustering
from nvmolkit._distance_inputs import _aap_args, _packed_metric_name, _prepare_distance_matrix, _prepare_fused_input
from nvmolkit.clustering import OutputMode, _validate_output
from nvmolkit.similarity import AAPMetric, Metric
from nvmolkit.types import ArrayInput, AsyncGpuResult


def _validate_maxmin_threshold(threshold: float | None) -> float:
    if threshold is None:
        return -1.0
    if not np.isfinite(threshold) or threshold < 0 or threshold > float(np.finfo(np.float32).max):
        raise ValueError(f"threshold must be between 0 and the largest finite float32 value, got {threshold}")
    return float(threshold)


def _index_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    try:
        return tuple(operator.index(value) for value in values)
    except TypeError:
        raise TypeError(f"{name} must be a sequence of integers") from None


def _resolve_selection_output(result, output: OutputMode, *, maxmin: bool):
    indices_obj, last_distance = result
    indices = AsyncGpuResult(indices_obj)
    if output is OutputMode.RDKIT:
        indices = tuple(int(index) for index in indices.numpy())
    return (indices, float(last_distance)) if maxmin else indices


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
        return _resolve_selection_output(result, output, maxmin=False)


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
        x: Packed int32 or uint32 fingerprints of shape ``(N, num_words)`` for
            Tanimoto and cosine, or a sequence of RDKit molecules for AAP.
        cutoff: Inclusive exclusion distance in ``[0, 1]``, rounded to float32.
        metric: Similarity metric.
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
    resolved, inputs, active_stream = _prepare_fused_input(x, metric, stream)
    pick_size = operator.index(pick_size)
    first_picks = _index_tuple("first_picks", first_picks)
    with torch.cuda.stream(active_stream):
        if isinstance(resolved, AAPMetric):
            result = _clustering.aap_leader(
                inputs, cutoff, pick_size, first_picks, *_aap_args(resolved), active_stream.cuda_stream
            )
        else:
            result = _clustering.fused_leader(
                inputs.__cuda_array_interface__,
                cutoff,
                _packed_metric_name(resolved),
                pick_size,
                first_picks,
                active_stream.cuda_stream,
            )
        return _resolve_selection_output(result, output, maxmin=False)


def maxmin(
    distance_matrix: ArrayInput,
    pick_size: int,
    *,
    first_picks: Sequence[int] = (),
    seed: int = -1,
    threshold: float | None = None,
    stream: torch.cuda.Stream | None = None,
    output: OutputMode = OutputMode.DEVICE,
) -> tuple[AsyncGpuResult | tuple[int, ...], float]:
    """Select a diverse subset from a distance matrix by greedy MaxMin.

    Each step adds the candidate whose distance to its nearest pick is largest,
    breaking ties by lowest index, as in RDKit's ``MaxMinPicker``. A given
    ``seed`` selects the same random first pick as RDKit.

    Distances are compared in float32, so distances within float32 rounding of
    each other tie, and a distance within float32 rounding of ``threshold`` can
    be classified differently than by RDKit's ``MaxMinPicker``, which compares
    in double precision.

    Args:
        distance_matrix: Square float32 or float64 matrix of shape ``(N, N)``.
            Element ``[i, j]`` is the distance from item ``i`` to item ``j``.
            Values are converted to float32 for comparisons. Values that
            overflow float32 during conversion become infinity.
        pick_size: Number of items to select, from 1 through ``N``. All
            ``first_picks`` are retained even if they exceed this limit.
        first_picks: Unique indices that start the selection, in order. If
            empty, the first pick is drawn at random.
        seed: Seed for the random first pick. Negative values use system
            entropy.
        threshold: Stop before adding a candidate whose nearest-pick distance
            is at most this value, rounded to float32. Must be between zero
            and the largest finite float32 value, or None for no threshold.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Result representation.

    Returns:
        ``(indices, last_distance)``. ``indices`` is an
        :class:`~nvmolkit.types.AsyncGpuResult` of int32 indices for
        ``OutputMode.DEVICE``, or a tuple of indices for ``OutputMode.RDKIT``.
        ``last_distance`` is the nearest-pick distance of the last item added,
        or ``-1`` if none was added after ``first_picks``.
    """
    _validate_output(output)
    matrix, active_stream = _prepare_distance_matrix(distance_matrix, stream)
    with torch.cuda.stream(active_stream):
        result = _clustering.maxmin(
            matrix.__cuda_array_interface__,
            operator.index(pick_size),
            _index_tuple("first_picks", first_picks),
            operator.index(seed),
            _validate_maxmin_threshold(threshold),
            active_stream.cuda_stream,
        )
        return _resolve_selection_output(result, output, maxmin=True)


def fused_maxmin(
    x,
    pick_size: int,
    *,
    metric: Metric = "tanimoto",
    first_picks: Sequence[int] = (),
    seed: int = -1,
    threshold: float | None = None,
    stream: torch.cuda.Stream | None = None,
    output: OutputMode = OutputMode.DEVICE,
) -> tuple[AsyncGpuResult | tuple[int, ...], float]:
    """Select a diverse subset by greedy MaxMin, computing distances as needed.

    Equivalent to :func:`maxmin` on the matrix of ``1 - similarity`` values,
    with memory that scales as ``O(N)``. Distances are computed in float32, so
    results can differ slightly from RDKit's double-precision results.

    Args:
        x: Packed int32 or uint32 fingerprints of shape ``(N, num_words)`` for
            Tanimoto and cosine, or a sequence of RDKit molecules for AAP.
        pick_size: Number of items to select, from 1 through ``N``. All
            ``first_picks`` are retained even if they exceed this limit.
        metric: Similarity metric.
        first_picks: Unique indices that start the selection, in order. If
            empty, the first pick is drawn at random.
        seed: Seed for the random first pick. Negative values use system
            entropy.
        threshold: Stop before adding a candidate whose nearest-pick distance
            is at most this value. Distances and the threshold are compared
            in float32. Must be in ``[0, 1]``, or None for no threshold.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Result representation.

    Returns:
        ``(indices, last_distance)``. ``indices`` is an
        :class:`~nvmolkit.types.AsyncGpuResult` of int32 indices for
        ``OutputMode.DEVICE``, or a tuple of indices for ``OutputMode.RDKIT``.
        ``last_distance`` is the nearest-pick distance of the last item added,
        or ``-1`` if none was added after ``first_picks``.
    """
    _validate_output(output)
    resolved, inputs, active_stream = _prepare_fused_input(x, metric, stream)
    pick_size = operator.index(pick_size)
    first_picks = _index_tuple("first_picks", first_picks)
    seed = operator.index(seed)
    threshold = _validate_maxmin_threshold(threshold)
    with torch.cuda.stream(active_stream):
        if isinstance(resolved, AAPMetric):
            result = _clustering.aap_maxmin(
                inputs, pick_size, first_picks, seed, threshold, *_aap_args(resolved), active_stream.cuda_stream
            )
        else:
            result = _clustering.fused_maxmin(
                inputs.__cuda_array_interface__,
                pick_size,
                _packed_metric_name(resolved),
                first_picks,
                seed,
                threshold,
                active_stream.cuda_stream,
            )
        return _resolve_selection_output(result, output, maxmin=True)
