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
from nvmolkit.similarity import aap_similarity  # noqa: F401 - compatibility re-export
from nvmolkit.types import ArrayInput, AsyncGpuResult, _as_cuda_tensor, _resolve_cuda_stream

_VALID_NEIGHBORLIST_SIZES = (8, 16, 24, 32, 64, 128)

_RDKitClusters = tuple[tuple[int, ...], ...]


class DISEOutputMode(Enum):
    """Output format for directed sphere exclusion (DISE) clustering."""

    RDKIT = "rdkit"
    DEVICE = "device"


@dataclass(frozen=True)
class DISEDeviceResult:
    """GPU-resident directed sphere exclusion (DISE) clustering result.

    Attributes:
        cluster_ids: One zero-based int32 cluster ID per input molecule.
        centroids: Centroid indices by cluster ID, int32 and shape ``(num_clusters,)``.
        cluster_sizes: Member counts by cluster ID, int64 and shape ``(num_clusters,)``.
    """

    cluster_ids: AsyncGpuResult
    centroids: AsyncGpuResult
    cluster_sizes: AsyncGpuResult


def _validate_dise_output(output: DISEOutputMode) -> None:
    if not isinstance(output, DISEOutputMode):
        raise TypeError(f"output must be a DISEOutputMode, got {type(output).__name__}")


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


def _resolve_dise_output(result, output: DISEOutputMode) -> _RDKitClusters | DISEDeviceResult:
    cluster_ids_obj, centroids_obj, cluster_sizes_obj = result
    if output is DISEOutputMode.DEVICE:
        return DISEDeviceResult(
            AsyncGpuResult(cluster_ids_obj),
            AsyncGpuResult(centroids_obj),
            AsyncGpuResult(cluster_sizes_obj),
        )
    return _cluster_arrays_to_rdkit(cluster_ids_obj, centroids_obj)


@overload
def aap_dise(
    molecules,
    similarity_threshold: float = 0.217,
    *,
    assignment: Literal["first", "nearest"] = "nearest",
    max_path_length: int = 7,
    histogram_bins: int = 2048,
    sinkhorn_iterations: int = 8,
    sinkhorn_temperature: float = 0.104,
    stream: torch.cuda.Stream | None = None,
    output: Literal[DISEOutputMode.DEVICE] = DISEOutputMode.DEVICE,
) -> DISEDeviceResult: ...


@overload
def aap_dise(
    molecules,
    similarity_threshold: float = 0.217,
    *,
    assignment: Literal["first", "nearest"] = "nearest",
    max_path_length: int = 7,
    histogram_bins: int = 2048,
    sinkhorn_iterations: int = 8,
    sinkhorn_temperature: float = 0.104,
    stream: torch.cuda.Stream | None = None,
    output: Literal[DISEOutputMode.RDKIT],
) -> _RDKitClusters: ...


# TODO: Explore a GPU-resident directed sphere exclusion control loop and asynchronous result production.
def aap_dise(
    molecules,
    similarity_threshold: float = 0.217,
    *,
    assignment: Literal["first", "nearest"] = "nearest",
    max_path_length: int = 7,
    histogram_bins: int = 2048,
    sinkhorn_iterations: int = 8,
    sinkhorn_temperature: float = 0.104,
    stream: torch.cuda.Stream | None = None,
    output: DISEOutputMode = DISEOutputMode.DEVICE,
) -> _RDKitClusters | DISEDeviceResult:
    """Cluster ordered RDKit molecules with directed sphere exclusion.

    Atom-Atom Path (AAP) similarity drives directed sphere exclusion (DISE).

    Input order supplies the direction: the first unassigned molecule becomes
    the next centroid. ``assignment="first"`` retains the first qualifying
    centroid assignment made during sphere exclusion; ``"nearest"`` performs
    the complete second pass and assigns every non-centroid to its most similar
    selected centroid.

    Args:
        molecules: RDKit molecules in priority order, each with at most 64 atoms.
        similarity_threshold: Inclusive AAP threshold used to select centroids.
        assignment: Assignment rule for non-centroid molecules.
        max_path_length: Maximum rooted path length in bonds.
        histogram_bins: Number of hashed path bins, at most 32767.
        sinkhorn_iterations: Number of Sinkhorn normalization iterations.
        sinkhorn_temperature: Sinkhorn temperature.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Output representation. Defaults to ``DISEOutputMode.DEVICE``.

    Returns:
        ``DISEOutputMode.DEVICE`` returns zero-based cluster IDs, centroids,
        and cluster sizes on the GPU. ``DISEOutputMode.RDKIT`` returns a tuple
        of centroid-first cluster tuples on the host. Both representations
        order clusters by descending size, with centroid order breaking ties.

    Note:
        The current DISE control loop makes host-side decisions and therefore
        completes its CUDA stream work before returning either output mode.
        For method details, see `Gobbi et al. (2015)
        <https://doi.org/10.1186/s13321-015-0056-8>`_.
    """
    _validate_dise_output(output)
    if not 0 <= similarity_threshold <= 1:
        raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")
    if assignment not in ("first", "nearest"):
        raise ValueError(f"assignment must be one of ['first', 'nearest'], got {assignment!r}")

    active_stream = _resolve_cuda_stream(stream)
    function = _clustering.aap_similarity_clustering if assignment == "first" else _clustering.aap_dise_clustering
    native_result = function(
        list(molecules),
        similarity_threshold,
        max_path_length,
        histogram_bins,
        sinkhorn_iterations,
        sinkhorn_temperature,
        output is DISEOutputMode.DEVICE,
        active_stream.cuda_stream,
    )
    return _resolve_dise_output(native_result, output)


class ButinaOutputMode(Enum):
    """Output format for :func:`butina` and :func:`fused_butina`.

    ``RDKIT`` returns the same tuple of clusters as RDKit's
    ``Butina.ClusterData``. ``DEVICE`` returns a :class:`ButinaDeviceResult`.
    """

    RDKIT = "rdkit"
    DEVICE = "device"


@dataclass(frozen=True)
class ButinaDeviceResult:
    """GPU-resident Butina clustering result.

    Attributes:
        cluster_ids: One int32 cluster ID per input item, shape ``(N,)``.
        centroids: Centroid indices by cluster ID, int32 and shape ``(num_clusters,)``.
        cluster_sizes: Member counts by cluster ID, int64 and shape ``(num_clusters,)``.
    """

    cluster_ids: AsyncGpuResult
    centroids: AsyncGpuResult
    cluster_sizes: AsyncGpuResult


def _wrap_cluster_arrays(result) -> tuple[AsyncGpuResult, AsyncGpuResult]:
    cluster_ids_obj, centroids_obj = result
    return AsyncGpuResult(cluster_ids_obj), AsyncGpuResult(centroids_obj)


def _wrap_device_result(result) -> ButinaDeviceResult:
    cluster_ids, centroids = _wrap_cluster_arrays(result)
    cluster_ids_int64 = cluster_ids.torch().to(torch.int64)
    cluster_sizes = torch.zeros_like(centroids.torch(), dtype=torch.int64)
    cluster_sizes.index_add_(0, cluster_ids_int64, torch.ones_like(cluster_ids_int64))
    return ButinaDeviceResult(cluster_ids, centroids, AsyncGpuResult(cluster_sizes))


def _to_rdkit_clusters(cluster_ids: AsyncGpuResult, centroids: AsyncGpuResult) -> _RDKitClusters:
    return _cluster_arrays_to_rdkit(cluster_ids.numpy(), centroids.numpy())


def _resolve_output(result, output: ButinaOutputMode) -> _RDKitClusters | ButinaDeviceResult:
    if output is ButinaOutputMode.DEVICE:
        return _wrap_device_result(result)
    return _to_rdkit_clusters(*_wrap_cluster_arrays(result))


def _validate_output(output: ButinaOutputMode) -> None:
    if not isinstance(output, ButinaOutputMode):
        raise TypeError(f"output must be a ButinaOutputMode, got {type(output).__name__}")


def _check_distance_matrix(name: str, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape={tuple(x.shape)}")
    if x.dtype != torch.float64:
        raise ValueError(f"{name} must have dtype float64")
    return x.contiguous()


@overload
def butina(
    distance_matrix: ArrayInput,
    cutoff: float,
    neighborlist_max_size: int = 64,
    reordering: bool = True,
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[ButinaOutputMode.DEVICE] = ButinaOutputMode.DEVICE,
) -> ButinaDeviceResult: ...


@overload
def butina(
    distance_matrix: ArrayInput,
    cutoff: float,
    neighborlist_max_size: int = 64,
    reordering: bool = True,
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[ButinaOutputMode.RDKIT],
) -> _RDKitClusters: ...


def butina(
    distance_matrix: ArrayInput,
    cutoff: float,
    neighborlist_max_size: int = 64,
    reordering: bool = True,
    stream: torch.cuda.Stream | None = None,
    *,
    output: ButinaOutputMode = ButinaOutputMode.DEVICE,
) -> _RDKitClusters | ButinaDeviceResult:
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
        output: Output representation. Defaults to ``ButinaOutputMode.DEVICE``.

    Returns:
        The representation selected by ``output``.

        ``ButinaOutputMode.RDKIT`` returns a tuple containing one tuple per
        cluster. Each cluster tuple contains input indices, with the centroid
        first. Constructing this representation synchronizes the CUDA work and
        copies the clustering result to the host.

        ``ButinaOutputMode.DEVICE`` returns a :class:`ButinaDeviceResult`
        containing three :class:`AsyncGpuResult` objects on the active CUDA
        device. ``cluster_ids`` is int32 with shape ``(N,)`` and maps each input
        index to a cluster ID. Cluster IDs are contiguous from zero through
        ``num_clusters - 1``. ``centroids`` is int32 with shape
        ``(num_clusters,)``; element ``k`` is an input index whose cluster ID is
        ``k``. ``cluster_sizes`` is int64 with shape ``(num_clusters,)``;
        element ``k`` equals the number of entries in ``cluster_ids`` that are
        equal to ``k``, and the sizes sum to ``N``. The return is
        asynchronous: each field's ``.torch()`` method exposes its CUDA tensor
        without a host copy, while ``.numpy()`` synchronizes and copies that
        field to the host.

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
        return _resolve_output(result, output)


@overload
def fused_butina(
    x: ArrayInput,
    cutoff: float,
    metric: str = "tanimoto",
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[ButinaOutputMode.DEVICE] = ButinaOutputMode.DEVICE,
) -> ButinaDeviceResult: ...


@overload
def fused_butina(
    x: ArrayInput,
    cutoff: float,
    metric: str = "tanimoto",
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[ButinaOutputMode.RDKIT],
) -> _RDKitClusters: ...


def fused_butina(
    x: ArrayInput,
    cutoff: float,
    metric: str = "tanimoto",
    stream: torch.cuda.Stream | None = None,
    *,
    output: ButinaOutputMode = ButinaOutputMode.DEVICE,
) -> _RDKitClusters | ButinaDeviceResult:
    """Perform fused Butina clustering on a set of fingerprints.

    This function uses a fused implementation of Butina clustering that computes
    similarities and neighbors on-the-fly, avoiding the need to compute and store
    the full distance matrix. This makes it suitable for large datasets.

    Args:
        x: Tensor-like object of shape (N, D) containing packed int32 or uint32 fingerprints
           to cluster. Can be an AsyncGpuResult, torch.Tensor, or numpy.ndarray.
           CPU tensors and NumPy arrays are copied to CUDA.
        cutoff: Distance threshold for clustering. Items are neighbors if their
                distance is at most this cutoff (i.e. similarity >= 1 - cutoff).
        metric: Metric to use for similarity computation. Currently only "tanimoto"
                and "cosine" are supported.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Output representation. Defaults to ``ButinaOutputMode.DEVICE``.

    Returns:
        The representation selected by ``output``.

        ``ButinaOutputMode.RDKIT`` returns a tuple containing one tuple per
        cluster. Each cluster tuple contains input indices, with the centroid
        first. Constructing this representation synchronizes the CUDA work and
        copies the clustering result to the host.

        ``ButinaOutputMode.DEVICE`` returns a :class:`ButinaDeviceResult`
        containing three :class:`AsyncGpuResult` objects on the active CUDA
        device. ``cluster_ids`` is int32 with shape ``(N,)`` and maps each input
        index to a cluster ID. Cluster IDs are contiguous from zero through
        ``num_clusters - 1``. ``centroids`` is int32 with shape
        ``(num_clusters,)``; element ``k`` is an input index whose cluster ID is
        ``k``. ``cluster_sizes`` is int64 with shape ``(num_clusters,)``;
        element ``k`` equals the number of entries in ``cluster_ids`` that are
        equal to ``k``, and the sizes sum to ``N``. The return is
        asynchronous: each field's ``.torch()`` method exposes its CUDA tensor
        without a host copy, while ``.numpy()`` synchronizes and copies that
        field to the host.

    """
    _validate_output(output)
    if metric not in ("tanimoto", "cosine"):
        raise ValueError(f"metric must be one of ['tanimoto', 'cosine'], got {metric}")

    if not 0 <= cutoff <= 1:
        raise ValueError(f"cutoff must be in [0, 1], got {cutoff}")

    (x,), active_stream = _prepare_packed_fingerprints(("x", x), stream=stream)
    with torch.cuda.stream(active_stream):
        result = _clustering.fused_butina(x.__cuda_array_interface__, cutoff, True, metric, active_stream.cuda_stream)
        return _resolve_output(result, output)
