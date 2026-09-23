# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GPU-accelerated clustering from distance matrices, fingerprints, or ordered RDKit molecules."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, overload

import numpy as np
import torch

from nvmolkit import _clustering
from nvmolkit._fingerprint_inputs import _prepare_packed_fingerprints
from nvmolkit.similarity import (
    _DEFAULT_AAP_METRIC,
    AAPMetric,
    CosineMetric,
    Metric,
    TanimotoMetric,
    _resolve_metric,
)
from nvmolkit.types import ArrayInput, AsyncGpuResult, _as_cuda_tensor, _resolve_cuda_stream

_VALID_NEIGHBORLIST_SIZES = (8, 16, 24, 32, 64, 128)

_RDKitClusters = tuple[tuple[int, ...], ...]


class OutputMode(Enum):
    """Result representation for clustering and selection functions.

    ``DEVICE`` returns :class:`~nvmolkit.types.AsyncGpuResult` buffers on the
    GPU. ``RDKIT`` returns host tuples of input indices in RDKit's format.
    """

    RDKIT = "rdkit"
    DEVICE = "device"


@dataclass(frozen=True)
class ClusterDeviceResult:
    """Device-resident clustering result.

    Attributes:
        cluster_ids: int32 cluster ID of each input, shape ``(N,)``. IDs are
            contiguous from zero.
        centroids: int32 input index of each cluster's centroid, shape
            ``(num_clusters,)``.
        cluster_sizes: int64 member count of each cluster, shape
            ``(num_clusters,)``.
    """

    cluster_ids: AsyncGpuResult
    centroids: AsyncGpuResult
    cluster_sizes: AsyncGpuResult


def _validate_output(output: OutputMode) -> None:
    if not isinstance(output, OutputMode):
        raise TypeError(f"output must be an OutputMode, got {type(output).__name__}")


def _validate_assignment(assignment: str) -> None:
    if assignment not in ("first", "nearest"):
        raise ValueError(f"assignment must be one of ['first', 'nearest'], got {assignment!r}")


def _aap_args(metric: AAPMetric) -> tuple:
    return (metric.max_path_length, metric.histogram_bins, metric.sinkhorn_iterations, metric.sinkhorn_temperature)


def _packed_metric_name(metric: TanimotoMetric | CosineMetric) -> str:
    return "tanimoto" if isinstance(metric, TanimotoMetric) else "cosine"


def _cluster_arrays_to_rdkit(cluster_ids_array, centroids_array) -> _RDKitClusters:
    cluster_ids_array = np.asarray(cluster_ids_array)
    centroids_array = np.asarray(centroids_array)
    member_order = np.argsort(cluster_ids_array, kind="stable")
    sorted_cluster_ids = cluster_ids_array[member_order]
    cluster_offsets = np.searchsorted(sorted_cluster_ids, np.arange(centroids_array.size + 1))

    clusters = []
    for cluster_id, centroid_value in enumerate(centroids_array):
        centroid = int(centroid_value)
        members = member_order[cluster_offsets[cluster_id] : cluster_offsets[cluster_id + 1]]
        clusters.append(tuple([centroid] + [int(member) for member in members if member != centroid]))
    return tuple(clusters)


def _resolve_cluster_output(result, output: OutputMode) -> _RDKitClusters | ClusterDeviceResult:
    cluster_ids_obj, centroids_obj, cluster_sizes_obj = result
    if output is OutputMode.DEVICE:
        return ClusterDeviceResult(
            AsyncGpuResult(cluster_ids_obj),
            AsyncGpuResult(centroids_obj),
            AsyncGpuResult(cluster_sizes_obj),
        )
    return _cluster_arrays_to_rdkit(cluster_ids_obj, centroids_obj)


def _wrap_cluster_arrays(result) -> tuple[AsyncGpuResult, AsyncGpuResult]:
    cluster_ids_obj, centroids_obj = result
    return AsyncGpuResult(cluster_ids_obj), AsyncGpuResult(centroids_obj)


def _wrap_butina_device_result(result) -> ClusterDeviceResult:
    cluster_ids, centroids = _wrap_cluster_arrays(result)
    cluster_ids_int64 = cluster_ids.torch().to(torch.int64)
    cluster_sizes = torch.zeros_like(centroids.torch(), dtype=torch.int64)
    cluster_sizes.index_add_(0, cluster_ids_int64, torch.ones_like(cluster_ids_int64))
    return ClusterDeviceResult(cluster_ids, centroids, AsyncGpuResult(cluster_sizes))


def _resolve_butina_output(result, output: OutputMode) -> _RDKitClusters | ClusterDeviceResult:
    if output is OutputMode.DEVICE:
        return _wrap_butina_device_result(result)
    cluster_ids, centroids = _wrap_cluster_arrays(result)
    return _cluster_arrays_to_rdkit(cluster_ids.numpy(), centroids.numpy())


def _check_distance_matrix(name: str, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape={tuple(x.shape)}")
    if x.dtype != torch.float64:
        raise ValueError(f"{name} must have dtype float64")
    return x.contiguous()


def aap_dise(
    molecules,
    similarity_threshold: float = 0.217,
    *,
    assignment: Literal["first", "nearest"] = "nearest",
    metric: AAPMetric = _DEFAULT_AAP_METRIC,
    stream: torch.cuda.Stream | None = None,
    output: OutputMode = OutputMode.DEVICE,
) -> ClusterDeviceResult | _RDKitClusters:
    """Cluster ordered RDKit molecules by directed sphere exclusion (DISE) with AAP similarity.

    Molecules are visited in input order. Each molecule not yet assigned becomes
    a centroid and claims every remaining molecule whose similarity from it is at
    least ``similarity_threshold``. With ``assignment="first"`` each molecule
    keeps that centroid; with ``"nearest"`` each non-centroid joins its most
    similar centroid. Clusters are ordered by descending size, then by centroid
    order.

    Args:
        molecules: RDKit molecules in priority order.
        similarity_threshold: Inclusive similarity threshold in ``[0, 1]``.
        assignment: ``"first"`` or ``"nearest"``.
        metric: AAP parameters.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Result representation.

    Returns:
        A :class:`ClusterDeviceResult` for ``OutputMode.DEVICE``, or
        centroid-first tuples of input indices for ``OutputMode.RDKIT``.

    Note:
        For the method, see `Gobbi et al. (2015)
        <https://doi.org/10.1186/s13321-015-0056-8>`_.
    """
    _validate_output(output)
    _validate_assignment(assignment)
    if not 0 <= similarity_threshold <= 1:
        raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")
    if not isinstance(metric, AAPMetric):
        raise TypeError(f"metric must be an AAPMetric, got {type(metric).__name__}")

    active_stream = _resolve_cuda_stream(stream)
    function = _clustering.aap_similarity_clustering if assignment == "first" else _clustering.aap_dise_clustering
    result = function(
        list(molecules),
        similarity_threshold,
        *_aap_args(metric),
        output is OutputMode.DEVICE,
        active_stream.cuda_stream,
    )
    return _resolve_cluster_output(result, output)


@overload
def butina(
    distance_matrix: ArrayInput,
    cutoff: float,
    neighborlist_max_size: int = 64,
    reordering: bool = True,
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[OutputMode.DEVICE] = OutputMode.DEVICE,
) -> ClusterDeviceResult: ...


@overload
def butina(
    distance_matrix: ArrayInput,
    cutoff: float,
    neighborlist_max_size: int = 64,
    reordering: bool = True,
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[OutputMode.RDKIT],
) -> _RDKitClusters: ...


def butina(
    distance_matrix: ArrayInput,
    cutoff: float,
    neighborlist_max_size: int = 64,
    reordering: bool = True,
    stream: torch.cuda.Stream | None = None,
    *,
    output: OutputMode = OutputMode.DEVICE,
) -> _RDKitClusters | ClusterDeviceResult:
    """Perform Butina clustering on a distance matrix.

    The Butina algorithm is a deterministic clustering method that groups items based
    on distance thresholds. It iteratively:
    1. Finds the item with the most neighbors within the cutoff distance
    2. Forms a cluster with that item and all its neighbors
    3. Removes clustered items from consideration
    4. Repeats until all items are clustered

    Args:
        distance_matrix: Square distance matrix of shape (N, N) where N is the number
                        of items. Can be an AsyncGpuResult, torch.Tensor, or numpy.ndarray.
                        CPU tensors and NumPy arrays are copied to CUDA. Inputs
                        must have dtype float64.
        cutoff: Distance threshold for clustering. Items are neighbors if their
                distance is less than or equal to this cutoff.
        neighborlist_max_size: Maximum size of the neighborlist used for small cluster
                              optimization. Must be 8, 16, 24, 32, 64, or 128. Larger values
                              allow parallel processing of larger clusters but use more
                              shared memory. Ignored when reordering is False.
        reordering: Whether to update neighbor counts among unassigned items
                    after each cluster is formed. Defaults to True, while
                    RDKit's ``Butina.ClusterData`` defaults to False.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Result representation.

    Returns:
        A :class:`ClusterDeviceResult` for ``OutputMode.DEVICE``, or the
        centroid-first cluster tuples returned by RDKit's
        ``Butina.ClusterData`` for ``OutputMode.RDKIT``.

    Note:
        The distance matrix should be symmetric and have zeros on the diagonal.
    """
    _validate_output(output)
    if neighborlist_max_size not in _VALID_NEIGHBORLIST_SIZES:
        raise ValueError(
            f"neighborlist_max_size must be one of {_VALID_NEIGHBORLIST_SIZES}, got {neighborlist_max_size}"
        )
    active_stream = _resolve_cuda_stream(stream, distance_matrix)
    with torch.cuda.stream(active_stream):
        distance_matrix_tensor = _as_cuda_tensor("distance_matrix", distance_matrix, stream=active_stream)
        distance_matrix_tensor = _check_distance_matrix("distance_matrix", distance_matrix_tensor)
        result = _clustering.butina(
            distance_matrix_tensor.__cuda_array_interface__,
            cutoff,
            neighborlist_max_size,
            True,
            reordering,
            active_stream.cuda_stream,
        )
        return _resolve_butina_output(result, output)


@overload
def fused_butina(
    x: ArrayInput,
    cutoff: float,
    metric: Metric = "tanimoto",
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[OutputMode.DEVICE] = OutputMode.DEVICE,
) -> ClusterDeviceResult: ...


@overload
def fused_butina(
    x: ArrayInput,
    cutoff: float,
    metric: Metric = "tanimoto",
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[OutputMode.RDKIT],
) -> _RDKitClusters: ...


def fused_butina(
    x: ArrayInput,
    cutoff: float,
    metric: Metric = "tanimoto",
    stream: torch.cuda.Stream | None = None,
    *,
    output: OutputMode = OutputMode.DEVICE,
) -> _RDKitClusters | ClusterDeviceResult:
    """Perform Butina clustering on fingerprints, computing distances as needed.

    Equivalent to :func:`butina` on the matrix of ``1 - similarity`` values,
    without forming the ``N x N`` matrix.

    Args:
        x: Packed int32 or uint32 fingerprints of shape ``(N, num_words)``. Can
           be an AsyncGpuResult, torch.Tensor, or numpy.ndarray. CPU tensors
           and NumPy arrays are copied to CUDA.
        cutoff: Inclusive neighbor distance in ``[0, 1]``.
        metric: Similarity metric. :class:`~nvmolkit.similarity.AAPMetric` is
            not yet supported.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Result representation.

    Returns:
        A :class:`ClusterDeviceResult` for ``OutputMode.DEVICE``, or the
        centroid-first cluster tuples returned by RDKit's
        ``Butina.ClusterData`` for ``OutputMode.RDKIT``.
    """
    _validate_output(output)
    resolved = _resolve_metric(metric)
    if isinstance(resolved, AAPMetric):
        raise NotImplementedError("fused_butina does not yet support AAPMetric")

    if not 0 <= cutoff <= 1:
        raise ValueError(f"cutoff must be in [0, 1], got {cutoff}")

    (x,), active_stream = _prepare_packed_fingerprints(("x", x), stream=stream)
    with torch.cuda.stream(active_stream):
        result = _clustering.fused_butina(
            x.__cuda_array_interface__, cutoff, True, _packed_metric_name(resolved), active_stream.cuda_stream
        )
        return _resolve_butina_output(result, output)
