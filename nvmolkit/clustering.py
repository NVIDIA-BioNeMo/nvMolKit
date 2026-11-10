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

"""GPU-accelerated Butina, directed sphere exclusion (DISE), BitBIRCH, and k-means clustering.

Butina and DISE accept precomputed distance matrices or compute distances as
needed through their ``fused_`` functions. BitBIRCH and k-means cluster packed
binary fingerprints directly.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal, overload

import numpy as np
import torch

from nvmolkit import _clustering
from nvmolkit._distance_inputs import _aap_args, _packed_metric_name, _prepare_distance_matrix, _prepare_fused_input
from nvmolkit._fingerprint_inputs import _prepare_packed_fingerprints
from nvmolkit.similarity import AAPMetric, Metric, _resolve_metric
from nvmolkit.types import ArrayInput, AsyncGpuResult, _as_cuda_tensor, _resolve_cuda_stream

_VALID_NEIGHBORLIST_SIZES = (8, 16, 24, 32, 64, 128)
_BITBIRCH_MAX_BRANCHING_FACTOR = 65534  # Matches kMaxBranchingFactor in src/bitbirch.cu.

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


def dise(
    distance_matrix: ArrayInput,
    cutoff: float,
    *,
    assignment: Literal["first", "nearest"] = "nearest",
    stream: torch.cuda.Stream | None = None,
    output: OutputMode = OutputMode.DEVICE,
) -> ClusterDeviceResult | _RDKitClusters:
    """Cluster a distance matrix by directed sphere exclusion (DISE).

    `Gobbi et al., J. Cheminform. 7, 11 (2015)
    <https://doi.org/10.1186/s13321-015-0056-8>`_.

    Centroids are the leaders selected by :func:`~nvmolkit.pickers.leader`. With
    ``assignment="first"``, each item joins the first centroid that excluded
    it; with ``"nearest"``, each non-centroid joins its nearest centroid, with
    ties going to the earlier centroid. Clusters are ordered by descending
    size, then by centroid selection order.

    Distances are compared in float32, so a distance within float32 rounding
    of the cutoff can be classified differently than in double precision, and
    distances within float32 rounding of each other tie in ``"nearest"``
    assignment.

    Args:
        distance_matrix: Square float32 or float64 matrix of shape ``(N, N)``.
            Element ``[i, j]`` is the distance from item ``i`` to item ``j``.
            Values are converted to float32 for comparisons. Values that
            overflow float32 during conversion become infinity.
        cutoff: Inclusive exclusion distance, rounded to float32. Must be
            between zero and the largest finite float32 value.
        assignment: ``"first"`` or ``"nearest"``.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Result representation.

    Returns:
        A :class:`ClusterDeviceResult` for ``OutputMode.DEVICE``, or
        centroid-first tuples of input indices for ``OutputMode.RDKIT``.
    """
    _validate_output(output)
    _validate_assignment(assignment)
    matrix, active_stream = _prepare_distance_matrix(distance_matrix, stream)
    with torch.cuda.stream(active_stream):
        result = _clustering.dise(
            matrix.__cuda_array_interface__,
            cutoff,
            assignment == "nearest",
            output is OutputMode.DEVICE,
            active_stream.cuda_stream,
        )
        return _resolve_cluster_output(result, output)


def fused_dise(
    x,
    cutoff: float,
    *,
    metric: Metric = "tanimoto",
    assignment: Literal["first", "nearest"] = "nearest",
    stream: torch.cuda.Stream | None = None,
    output: OutputMode = OutputMode.DEVICE,
) -> ClusterDeviceResult | _RDKitClusters:
    """Cluster by directed sphere exclusion (DISE), computing distances as needed.

    `Gobbi et al., J. Cheminform. 7, 11 (2015)
    <https://doi.org/10.1186/s13321-015-0056-8>`_.

    Equivalent to :func:`dise` on the matrix of ``1 - similarity`` values,
    with memory that scales as ``O(N)``. Distances are computed in float32, so
    results can differ slightly from RDKit's double-precision results.

    Args:
        x: Packed int32 or uint32 fingerprints of shape ``(N, num_words)`` for
            Tanimoto and cosine, or a sequence of RDKit molecules for AAP.
        cutoff: Inclusive exclusion distance in ``[0, 1]``, rounded to float32.
        metric: Similarity metric.
        assignment: ``"first"`` or ``"nearest"``.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Result representation.

    Returns:
        A :class:`ClusterDeviceResult` for ``OutputMode.DEVICE``, or
        centroid-first tuples of input indices for ``OutputMode.RDKIT``.
    """
    _validate_output(output)
    _validate_assignment(assignment)
    resolved, inputs, active_stream = _prepare_fused_input(x, metric, stream)
    with torch.cuda.stream(active_stream):
        if isinstance(resolved, AAPMetric):
            result = _clustering.aap_dise(
                inputs,
                cutoff,
                assignment == "nearest",
                *_aap_args(resolved),
                output is OutputMode.DEVICE,
                active_stream.cuda_stream,
            )
        else:
            result = _clustering.fused_dise(
                inputs.__cuda_array_interface__,
                cutoff,
                _packed_metric_name(resolved),
                assignment == "nearest",
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

    `Butina, JCICS 39, 747-750 (1999) <https://doi.org/10.1021/ci9803381>`_.

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

    `Butina, JCICS 39, 747-750 (1999) <https://doi.org/10.1021/ci9803381>`_.

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


def bitbirch(
    x: ArrayInput,
    threshold: float,
    *,
    branching_factor: int = 1024,
    batch_size: int = 1024,
    return_centroids: bool = False,
    stream: torch.cuda.Stream | None = None,
) -> AsyncGpuResult | tuple[AsyncGpuResult, AsyncGpuResult]:
    """Cluster packed binary fingerprints on the GPU using BitBIRCH.

    `Pickering et al., BitBIRCH-Lean, bioRxiv (2025)
    <https://doi.org/10.1101/2025.10.22.684015>`_.

    Args:
        x: Shape ``(N, W)`` packed int32 or uint32 fingerprints, ``W <= 128``
           (at most 4096 bits). Host arrays are copied to the GPU.
        threshold: Minimum combined-cluster iSIM Jaccard--Tanimoto similarity.
        branching_factor: Maximum entries per tree node. Must be in [3, 65534].
        batch_size: Number of fingerprints processed together. Must be positive.
        return_centroids: Also return packed majority centroids with shape
                          ``(num_clusters, W)``.
        stream: CUDA stream to use. If None, uses the current stream.

    Returns:
        One cluster ID per input fingerprint, numbered by each cluster's earliest
        member, and the centroids in cluster-ID order if requested.

    Notes:
        A row represents exactly ``32 * W`` bits. Results are deterministic for
        fixed input and options; changing ``batch_size`` can change the clustering.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if not 3 <= branching_factor <= _BITBIRCH_MAX_BRANCHING_FACTOR:
        raise ValueError(f"branching_factor must be in [3, {_BITBIRCH_MAX_BRANCHING_FACTOR}], got {branching_factor}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    (x,), active_stream = _prepare_packed_fingerprints(("x", x), stream=stream)
    if x.shape[1] > 128:
        raise ValueError(f"bitbirch supports at most 128 fingerprint words (4096 bits), got {x.shape[1]}")
    with torch.cuda.stream(active_stream):
        labels, centroids = _clustering.bitbirch(
            x.__cuda_array_interface__,
            threshold,
            branching_factor,
            batch_size,
            return_centroids,
            active_stream.cuda_stream,
        )
        return (AsyncGpuResult(labels), AsyncGpuResult(centroids)) if return_centroids else AsyncGpuResult(labels)


@dataclass(frozen=True)
class KMeansResult:
    """Result of :func:`kmeans`, on the GPU.

    Attributes:
        labels: int32 cluster ID of each fingerprint, shape ``(N,)``.
        centers: float32 cluster centers, shape ``(n_clusters, num_words * 32)``,
            in the bit order of :func:`~nvmolkit.fingerprints.unpack_fingerprint`.
            Each center is the mean of its members' bits (of L2-normalized bits
            for ``metric="cosine"``).
        cluster_sizes: int64 number of members of each cluster.
        inertia: Sum of the fitted fingerprints' distances to their centers.
        n_iter: Number of iterations run.
    """

    labels: AsyncGpuResult
    centers: AsyncGpuResult
    cluster_sizes: AsyncGpuResult
    inertia: float
    n_iter: int


_KMEANS_METRICS = ("tanimoto", "cosine")
_KMEANS_INITS = ("k-means++", "random", "maxmin")


def kmeans(
    x: ArrayInput,
    n_clusters: int,
    *,
    metric: Literal["tanimoto", "cosine"] = "tanimoto",
    init: Literal["k-means++", "random", "maxmin"] | ArrayInput = "k-means++",
    n_init: int = 1,
    max_iter: int = 300,
    tol: float = 1e-4,
    fit_size: int | None = None,
    random_state: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> KMeansResult:
    """Cluster packed fingerprints with k-means.

    Requires a GPU of compute capability 8.0 or newer.

    Args:
        x: Packed int32 or uint32 fingerprints of shape ``(N, num_words)``. Can
            be an AsyncGpuResult, torch.Tensor, or numpy.ndarray. CPU tensors
            and NumPy arrays are copied to CUDA.
        n_clusters: Number of clusters, between 1 and the number of fitted
            fingerprints.
        metric: ``"tanimoto"`` assigns each fingerprint to the center of highest
            Tanimoto similarity. ``"cosine"`` assigns it to the nearest center by
            Euclidean distance between L2-normalized bits.
        init: ``"k-means++"``, ``"random"`` (distinct fingerprints chosen
            uniformly), ``"maxmin"`` (MaxMin picks), or initial centers of
            shape ``(n_clusters, num_words * 32)`` in the space of
            :attr:`KMeansResult.centers` (for ``metric="cosine"``, the space
            of L2-normalized bits).
        n_init: Number of initializations; the run with the lowest inertia is
            kept. Ignored when ``init`` is an array of centers.
        max_iter: Maximum number of iterations per initialization.
        tol: Stop when no label changes or when the relative decrease of the
            inertia is at most ``tol``. ``tol=0`` runs exactly ``max_iter``
            iterations.
        fit_size: Fit the centers on this many randomly sampled fingerprints,
            then label all fingerprints against them. ``None`` fits on all.
        random_state: Seed for the fit sample and the initialization.
        stream: CUDA stream to use. If None, uses the current stream.

    Returns:
        A :class:`KMeansResult`.

    Notes:
        Results are deterministic for a fixed ``random_state``.
    """
    if metric not in _KMEANS_METRICS:
        raise ValueError(f"metric must be one of {list(_KMEANS_METRICS)}, got {metric!r}")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}")
    if n_init < 1:
        raise ValueError(f"n_init must be positive, got {n_init}")
    if max_iter < 0 or tol < 0:
        raise ValueError("max_iter and tol must be non-negative")
    if fit_size is not None and fit_size < n_clusters:
        raise ValueError(f"fit_size must be at least n_clusters ({n_clusters}), got {fit_size}")
    (x,), active_stream = _prepare_packed_fingerprints(("x", x), stream=stream)
    num_rows = x.shape[0]
    num_bits = x.shape[1] * 32
    if n_clusters > num_rows:
        raise ValueError(f"n_clusters ({n_clusters}) must not exceed the number of fingerprints ({num_rows})")
    base_seed = int(np.random.SeedSequence(random_state).generate_state(1, np.uint64)[0])

    with torch.cuda.stream(active_stream):
        fit = x
        if fit_size is not None and fit_size < num_rows:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(base_seed & 0x7FFFFFFFFFFFFFFF)
            rows = torch.randperm(num_rows, generator=generator, device=x.device)[:fit_size]
            fit = x.view(torch.int32)[rows].view(torch.uint32).contiguous()

        given_centers = None
        if not isinstance(init, str):
            given_centers = _as_cuda_tensor("init", init, stream=active_stream).to(torch.float32).contiguous()
            if given_centers.shape != (n_clusters, num_bits):
                raise ValueError(f"init must have shape {(n_clusters, num_bits)}, got {tuple(given_centers.shape)}")
            n_init = 1
        elif init not in _KMEANS_INITS:
            raise ValueError(f"init must be one of {list(_KMEANS_INITS)} or an array, got {init!r}")

        best = None
        for run in range(n_init):
            seed = (base_seed + 0x9E3779B97F4A7C15 * run) & 0xFFFFFFFFFFFFFFFF
            labels, centers, sizes, inertia, n_iter, _ = _clustering.kmeans_fit(
                fit.__cuda_array_interface__,
                None if given_centers is None else given_centers.__cuda_array_interface__,
                init if given_centers is None else "random",
                n_clusters,
                metric,
                max_iter,
                tol,
                seed,
                active_stream.cuda_stream,
            )
            if best is None or inertia < best[3]:
                best = (labels, centers, sizes, inertia, n_iter)
        labels, centers, sizes, inertia, n_iter = best
        labels = torch.as_tensor(labels, device=x.device)
        centers = torch.as_tensor(centers, device=x.device)
        sizes = torch.as_tensor(sizes, device=x.device)
        if fit is not x:
            labels, _ = _clustering.kmeans_assign(
                x.__cuda_array_interface__,
                centers.__cuda_array_interface__,
                n_clusters,
                metric,
                active_stream.cuda_stream,
            )
            labels = torch.as_tensor(labels, device=x.device)
            sizes = torch.bincount(labels.to(torch.int64), minlength=n_clusters)
        return KMeansResult(
            AsyncGpuResult(labels), AsyncGpuResult(centers), AsyncGpuResult(sizes), float(inertia), int(n_iter)
        )
