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

The standard ``butina()`` path precomputes a full N x N distance matrix and then
clusters from it.  This is the right choice when you already have a distance
matrix or plan to reuse it (e.g. at multiple cutoffs), and when N is small
enough that the O(N^2) matrix fits comfortably in GPU memory.

``fused_butina()`` avoids materializing the distance matrix entirely.  Each
clustering round recomputes only the similarities it needs on the fly using
Triton kernels that fuse popcount-based fingerprint similarity with the
neighbor-count and cluster-extraction steps.  This trades extra compute for
drastically lower memory: usage is O(N) rather than O(N^2), making it the
better choice for large N where the full matrix would be prohibitively large.
"""

import torch

from nvmolkit import _clustering
from nvmolkit._arrayHelpers import *  # noqa: F403
from nvmolkit._fusedButina import _check_fingerprint_matrix, extract_cluster_and_singletons, update_neighbor_counts
from nvmolkit.types import ArrayInput, AsyncGpuResult, _as_cuda_tensor, _resolve_cuda_stream, _validate_cuda_stream

_VALID_NEIGHBORLIST_SIZES = frozenset({8, 16, 24, 32, 64, 128})


def _check_distance_matrix(name: str, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape={tuple(x.shape)}")
    if x.dtype != torch.float64:
        raise ValueError(f"{name} must have dtype float64")
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def butina(
    distance_matrix: ArrayInput,
    cutoff: float,
    neighborlist_max_size: int = 64,
    return_centroids: bool = False,
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
                distance is less than this cutoff.
        neighborlist_max_size: Maximum size of the neighborlist used for small cluster
                              optimization. Must be 8, 16, 24, 32, 64, or 128. Larger values
                              allow parallel processing of larger clusters but use more
                              shared memory.
        return_centroids: Whether to return centroid indices for each cluster.
        stream: CUDA stream to use. If None, uses the current stream.

    Returns:
        AsyncGpuResult of shape ``(N,)`` with cluster IDs (cluster 0 is the
        largest) when ``return_centroids`` is False.  When ``return_centroids``
        is True, returns a tuple ``(clusters, centroids)`` where *centroids* is
        an AsyncGpuResult of shape ``(num_clusters,)`` containing the centroid
        index for each cluster ID.

    Note:
        The distance matrix should be symmetric and have zeros on the diagonal.
    """
    if neighborlist_max_size not in _VALID_NEIGHBORLIST_SIZES:
        raise ValueError(
            f"neighborlist_max_size must be one of {sorted(_VALID_NEIGHBORLIST_SIZES)}, got {neighborlist_max_size}"
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
            active_stream.cuda_stream,
        )
    if return_centroids:
        clusters, centroids = result
        return AsyncGpuResult(clusters), AsyncGpuResult(centroids)
    return AsyncGpuResult(result)


def fused_butina(
    x: ArrayInput,
    cutoff: float,
    return_centroids: bool = False,
    stream: torch.cuda.Stream | None = None,
    metric: str = "tanimoto",
):
    """Perform fused Butina clustering on a set of fingerprints.

    This function uses a fused implementation of Butina clustering that computes
    similarities and neighbors on-the-fly, avoiding the need to compute and store
    the full distance matrix. This makes it suitable for large datasets.

    Args:
        x: Tensor-like object of shape (N, D) containing packed int32 fingerprints
           to cluster. Can be an AsyncGpuResult, torch.Tensor, or numpy.ndarray.
           CPU tensors and NumPy arrays are copied to CUDA.
        cutoff: Distance threshold for clustering. Items are neighbors if their
                distance is less than this cutoff (i.e. similarity > 1 - cutoff).
        return_centroids: Whether to return centroid indices for each cluster.
        stream: CUDA stream to use. If None, uses the current stream.
        metric: Metric to use for similarity computation. Currently only "tanimoto"
                and "cosine" are supported.

    Returns:
        A tuple ``(clusters, cluster_sizes)`` where *clusters* is a list of tuples
        representing each cluster (with the first element being the centroid), and
        *cluster_sizes* is a list of cumulative cluster sizes.
        If ``return_centroids`` is True, returns a tuple ``(clusters, cluster_sizes, centroids)``
        where *centroids* is a list of centroid indices.
    """
    if metric not in ["tanimoto", "cosine"]:
        raise ValueError(f"metric must be one of ['tanimoto', 'cosine'], got {metric}")

    _validate_cuda_stream(stream)

    if cutoff < 0 or cutoff > 1:
        raise ValueError(f"cutoff must be in [0, 1], got {cutoff}")

    active_stream = _resolve_cuda_stream(stream, x)
    with torch.cuda.stream(active_stream):
        x = _as_cuda_tensor("x", x, stream=active_stream)
        _check_fingerprint_matrix("x", x)
        n_start = x.shape[0]
        device = x.device
        indices = torch.arange(n_start, dtype=torch.int32, device=device)
        # CPU mirror of indices avoids a D2H sync to record each centroid.
        indices_host = list(range(n_start))
        cluster_count = torch.zeros(2, dtype=torch.int32, device=device)
        cluster_count[1] = n_start - 1
        cluster_indices = torch.zeros(n_start, dtype=torch.int32, device=device)
        cluster_sizes = [0]
        centroids = []
        is_free = torch.ones(n_start, dtype=torch.bool, device=device)
        neigh = torch.zeros(n_start, dtype=torch.int32, device=device)
        threshold = float(1 - cutoff)
        y = x
        first_run = True
        # cc[0] = next cluster start index, cc[1] = last valid index.
        # Initialized to [0, n_start-1] matching cluster_count, so the initial
        # while condition is satisfied for n_start > 0 without a D2H sync.
        cc = [0, n_start - 1]
        while cc[0] <= cc[1] and x.shape[0] > 0:
            update_neighbor_counts(x, y, neigh, threshold, subtract=not first_run, metric=metric)
            first_run = False

            # Batch max and last-argmax into one D2H transfer (sync 1 of 2).
            neigh_flipped = neigh.flip(0)
            batch_ma = torch.stack([neigh.max().to(torch.int64), neigh_flipped.argmax().to(torch.int64)]).tolist()
            max_val = int(batch_ma[0])
            if max_val == 0:
                break
            id_max = neigh.shape[0] - 1 - int(batch_ma[1])
            centroids.append(indices_host[id_max])  # CPU mirror, no sync

            extract_cluster_and_singletons(
                x, id_max, is_free, neigh, cluster_count, cluster_indices, threshold, indices, metric=metric
            )

            # Batch cluster_count and is_free into one D2H transfer (sync 2 of 2).
            # combined[:2] = cluster_count, combined[2:] = is_free as int32.
            combined = torch.cat([cluster_count, is_free.to(torch.int32)]).tolist()
            cc = [int(combined[0]), int(combined[1])]
            is_free_host = [bool(v) for v in combined[2:]]

            cluster_sizes.append(cc[0])

            # is_free is already updated in-place on GPU by extract_cluster_and_singletons;
            # use it directly to avoid a H2D→CPU→H2D roundtrip.
            # is_free_host (from the combined download above) is still needed for indices_host.
            y = x[~is_free, :].contiguous()
            x = x[is_free, :].contiguous()
            indices = indices[is_free].contiguous()
            neigh = neigh[is_free].contiguous()
            is_free = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            indices_host = [idx for idx, keep in zip(indices_host, is_free_host) if keep]

        cluster_indices_cpu = cluster_indices.cpu()
        indices_cpu = cluster_indices_cpu.numpy()
        for i in range(n_start - cluster_sizes[-1]):
            item = cluster_sizes[-1]
            cluster_sizes.append(cluster_sizes[-1] + 1)
            centroids.append(int(indices_cpu[item]))

        clusters = []
        for i in range(len(cluster_sizes) - 1):
            start_idx = cluster_sizes[i]
            end_idx = cluster_sizes[i + 1]
            cluster_members = indices_cpu[start_idx:end_idx].tolist()

            centroid = centroids[i]
            members = [centroid] + [m for m in cluster_members if m != centroid]
            clusters.append(tuple(members))
        if return_centroids:
            return clusters, cluster_sizes, centroids
        return clusters, cluster_sizes
