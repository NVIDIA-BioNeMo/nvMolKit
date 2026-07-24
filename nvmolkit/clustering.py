# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Contains GPU-accelerated Butina clustering implementations.

The standard ``butina()`` path accepts a full N x N distance matrix and
materializes its neighbor relationships. This is the right choice when you
already have a distance matrix or plan to reuse it (e.g. at multiple cutoffs),
and when N is small enough that the O(N^2) data fits comfortably in GPU memory.

``fused_butina()`` avoids materializing the distance matrix entirely.  Each
clustering round recomputes only the similarities it needs on the fly with
CUDA kernels that fuse popcount-based fingerprint similarity with the
neighbor-count and cluster-extraction steps. This trades extra compute for
drastically lower memory: usage is O(N) rather than O(N^2), making it the
better choice for large N where the full matrix would be prohibitively large.
"""

import torch

from nvmolkit import _clustering
from nvmolkit._fingerprint_inputs import _prepare_packed_fingerprints
from nvmolkit.types import ArrayInput, AsyncGpuResult, _as_cuda_tensor, _resolve_cuda_stream

_VALID_NEIGHBORLIST_SIZES = (8, 16, 24, 32, 64, 128)


def _wrap_result(result, return_centroids: bool):
    if return_centroids:
        cluster_ids, centroids = result
        return AsyncGpuResult(cluster_ids), AsyncGpuResult(centroids)
    return AsyncGpuResult(result)


def _check_distance_matrix(name: str, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape={tuple(x.shape)}")
    if x.dtype != torch.float64:
        raise ValueError(f"{name} must have dtype float64")
    return x.contiguous()


def butina(
    distance_matrix: ArrayInput,
    cutoff: float,
    neighborlist_max_size: int = 64,
    return_centroids: bool = False,
    reordering: bool = True,
    stream: torch.cuda.Stream | None = None,
) -> AsyncGpuResult | tuple[AsyncGpuResult, AsyncGpuResult]:
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
        return_centroids: Whether to return centroid indices for each cluster.
        reordering: Whether to update neighbor counts among unassigned items
                    after each cluster is formed. Defaults to True, while
                    RDKit's ``Butina.ClusterData`` defaults to False.
        stream: CUDA stream to use. If None, uses the current stream.

    Returns:
        AsyncGpuResult of shape ``(N,)`` with cluster IDs (cluster 0 is the
        largest) when ``return_centroids`` is False. When ``return_centroids``
        is True, returns a tuple ``(cluster_ids, centroids)`` where *centroids* is
        an AsyncGpuResult of shape ``(num_clusters,)`` containing the centroid
        index for each cluster ID.

    Note:
        The distance matrix should be symmetric and have zeros on the diagonal.
    """
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
            return_centroids,
            reordering,
            active_stream.cuda_stream,
        )
        return _wrap_result(result, return_centroids)


def fused_butina(
    x: ArrayInput,
    cutoff: float,
    return_centroids: bool = False,
    metric: str = "tanimoto",
    stream: torch.cuda.Stream | None = None,
) -> AsyncGpuResult | tuple[AsyncGpuResult, AsyncGpuResult]:
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
        return_centroids: Whether to return centroid indices for each cluster.
        metric: Metric to use for similarity computation. Currently only "tanimoto"
                and "cosine" are supported.
        stream: CUDA stream to use. If None, uses the current stream.

    Returns:
        AsyncGpuResult of shape ``(N,)`` containing one cluster ID per input item
        when ``return_centroids`` is False. When ``return_centroids`` is True,
        returns ``(cluster_ids, centroids)``, where *centroids* contains the input
        index selected as the centroid for each cluster ID.
    """
    if metric not in ("tanimoto", "cosine"):
        raise ValueError(f"metric must be one of ['tanimoto', 'cosine'], got {metric}")

    if not 0 <= cutoff <= 1:
        raise ValueError(f"cutoff must be in [0, 1], got {cutoff}")

    (x,), active_stream = _prepare_packed_fingerprints(("x", x), stream=stream)
    with torch.cuda.stream(active_stream):
        result = _clustering.fused_butina(
            x.__cuda_array_interface__, cutoff, return_centroids, metric, active_stream.cuda_stream
        )
        return _wrap_result(result, return_centroids)
