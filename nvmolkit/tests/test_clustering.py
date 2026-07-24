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

import numpy as np
import pytest
import torch
from rdkit.ML.Cluster.Butina import ClusterData

from nvmolkit.clustering import butina, fused_butina
from nvmolkit.types import AsyncGpuResult


def check_butina_correctness(hit_mat, clusts):
    hit_mat = hit_mat.clone()
    seen = set()

    for i, clust in enumerate(clusts):
        assert len(clust) > 0, "Empty cluster found"
        clust_size = len(clust)

        if clust_size == 1:
            remaining_items = []
            for remaining_clust in clusts[i:]:
                assert len(remaining_clust) == 1, "Expected all remaining clusters to be singletons"
                remaining_items.append(remaining_clust[0])

            remaining_set = set(remaining_items)
            assert len(remaining_set) == len(remaining_items), "Duplicate items in singleton clusters"
            assert remaining_set.isdisjoint(seen), "Singleton item was already seen"
            seen.update(remaining_set)
            break
        counts = hit_mat.sum(-1)
        assert clust_size == counts.max(), (
            f"Cluster size {clust_size} doesn't match max available count {counts.max()}"
        )
        for item in clust:
            assert item not in seen, f"Point {item} assigned to multiple clusters"
            seen.add(item)
            hit_mat[item, :] = False
            hit_mat[:, item] = False
    assert len(seen) == hit_mat.shape[0]


def _nvmolkit_butina_clusters(distance_matrix, cutoff, *, reordering):
    labels, centroids = butina(
        torch.tensor(distance_matrix, device="cuda"),
        cutoff,
        reordering=reordering,
        return_centroids=True,
    )
    labels = labels.torch().cpu().numpy()
    centroids = centroids.torch().cpu().numpy()

    clusters = []
    for cluster_id, centroid in enumerate(centroids):
        members = np.flatnonzero(labels == cluster_id)
        clusters.append(tuple([int(centroid)] + [int(member) for member in members if member != centroid]))
    return tuple(clusters)


def _rdkit_butina_clusters(distance_matrix, cutoff, *, reordering):
    return ClusterData(
        np.asarray(distance_matrix, dtype=np.float64),
        int(distance_matrix.shape[0]),
        cutoff,
        isDistData=True,
        reordering=reordering,
    )


@pytest.mark.parametrize("reordering", [False, True])
def test_butina_reordering_matches_rdkit(reordering):
    distance_matrix = np.array(
        [
            [0.0, 0.2, 1.0, 1.0],
            [0.2, 0.0, 0.2, 1.0],
            [1.0, 0.2, 0.0, 0.2],
            [1.0, 1.0, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    cutoff = 0.2

    expected = _rdkit_butina_clusters(distance_matrix, cutoff, reordering=reordering)
    got = _nvmolkit_butina_clusters(distance_matrix, cutoff, reordering=reordering)

    assert got == expected


@pytest.mark.parametrize(
    "size,neighborlist_max_size", [(s, n) for s in (1, 10, 100, 1000) for n in (8, 16, 24, 32, 64, 128)]
)
def test_butina_clustering(size, neighborlist_max_size):
    n = size
    cutoff = 0.1
    dists = np.random.default_rng(42).random((n, n))
    dists = np.abs(dists - dists.T)
    torch_dists = torch.tensor(dists).to("cuda")
    nvmol_res = butina(torch_dists, cutoff, neighborlist_max_size=neighborlist_max_size).torch()
    nvmol_clusts = [tuple(torch.argwhere(nvmol_res == i).flatten().tolist()) for i in range(nvmol_res.max() + 1)]

    check_butina_correctness(torch_dists <= cutoff, nvmol_clusts)


@pytest.mark.parametrize("neighborlist_max_size", [8, 16, 24, 32, 64, 128])
def test_butina_edge_one_cluster(neighborlist_max_size):
    n = 10
    cutoff = 100.0
    torch_dists = torch.zeros((n, n), dtype=torch.float64, device="cuda")
    nvmol_res = butina(torch_dists, cutoff, neighborlist_max_size=neighborlist_max_size).torch()
    assert torch.all(nvmol_res == 0)


@pytest.mark.parametrize("neighborlist_max_size", [8, 16, 24, 32, 64, 128])
def test_butina_edge_n_clusters(neighborlist_max_size):
    n = 10
    cutoff = 1e-8
    torch_dists = torch.ones((n, n), dtype=torch.float64, device="cuda")
    torch_dists.fill_diagonal_(0)
    nvmol_res = butina(torch_dists, cutoff, neighborlist_max_size=neighborlist_max_size).torch()
    assert torch.all(nvmol_res.sort()[0] == torch.arange(10).to("cuda"))


def test_butina_returns_centroids():
    n = 25
    cutoff = 0.2
    dists = np.random.default_rng(123).random((n, n))
    dists = np.abs(dists - dists.T)
    torch_dists = torch.tensor(dists).to("cuda")
    cluster_ids, centroids = butina(torch_dists, cutoff, return_centroids=True)
    cluster_ids_tensor = cluster_ids.torch()
    centroids_tensor = centroids.torch()

    num_clusters = int(cluster_ids_tensor.max().item()) + 1
    assert centroids_tensor.numel() == num_clusters

    adjacency = torch_dists <= cutoff
    for cluster_id in range(num_clusters):
        centroid = int(centroids_tensor[cluster_id].item())
        assert cluster_ids_tensor[centroid].item() == cluster_id
        members = torch.nonzero(cluster_ids_tensor == cluster_id, as_tuple=False).flatten()
        for member in members:
            assert adjacency[centroid, member].item()


@pytest.mark.parametrize("input_kind", ["async", "cpu_tensor", "numpy"])
def test_butina_accepts_array_input_types(input_kind):
    n = 20
    cutoff = 0.2
    dists = np.random.default_rng(456).random((n, n))
    dists = np.abs(dists - dists.T)
    torch_dists = torch.tensor(dists, device="cuda", dtype=torch.float64)
    expected = butina(torch_dists, cutoff).torch().cpu()

    if input_kind == "async":
        inp = AsyncGpuResult(torch_dists)
    elif input_kind == "cpu_tensor":
        inp = torch.tensor(dists, dtype=torch.float64)
    else:
        inp = dists

    got = butina(inp, cutoff).torch().cpu()
    torch.testing.assert_close(got, expected)


def test_butina_on_explicit_stream():
    n = 100
    cutoff = 0.1
    dists = np.random.default_rng(42).random((n, n))
    dists = np.abs(dists - dists.T)
    torch_dists = torch.tensor(dists).to("cuda")

    s = torch.cuda.Stream()
    result = butina(torch_dists, cutoff, stream=s).torch()
    s.synchronize()

    nvmol_clusts = [tuple(torch.argwhere(result == i).flatten().tolist()) for i in range(result.max() + 1)]
    check_butina_correctness(torch_dists <= cutoff, nvmol_clusts)


def test_butina_invalid_stream_type():
    n = 10
    dists = torch.zeros(n, n, device="cuda", dtype=torch.float64)
    with pytest.raises(TypeError):
        butina(dists, 0.1, stream=42)


def test_butina_rejects_non_float64_distance_matrix():
    dists = torch.zeros(10, 10, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="distance_matrix must have dtype float64"):
        butina(dists, 0.1)


@pytest.mark.parametrize("invalid_size", [0, 1, 7, 9, 15, 33, 48, 100, 256])
def test_butina_invalid_neighborlist_max_size(invalid_size):
    """Test that invalid neighborlist_max_size values are rejected before reaching the GPU."""
    n = 10
    dists = torch.zeros(n, n, dtype=torch.float64)
    with pytest.raises(ValueError, match="neighborlist_max_size must be one of"):
        butina(dists, 0.1, neighborlist_max_size=invalid_size)


# ---------------------------------------------------------------------------
# Helpers for fused_butina tests
# ---------------------------------------------------------------------------


def generate_clustered_fingerprints(n, num_words=32, num_clusters=10, noise_range=2, seed=42):
    """Create bit-packed int32 fingerprints with controllable cluster structure."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    base_vectors = torch.randint(
        -(2**31 - 1),
        2**31 - 1,
        size=(num_clusters, num_words),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    x = torch.zeros((n, num_words), dtype=torch.int32, device="cuda")
    for i in range(n):
        x[i] = base_vectors[i % num_clusters]
        noise = torch.randint(
            0,
            noise_range,
            size=(num_words,),
            dtype=torch.int32,
            device="cuda",
            generator=generator,
        )
        x[i] = x[i] ^ noise
    return x


def compute_pairwise_similarity_cpu(x_np, metric="tanimoto"):
    """Compute NxN similarity from (N, D) int32 bit-packed fingerprints on CPU."""
    n, d = x_np.shape
    bits = np.unpackbits(x_np.view(np.uint8).reshape(n, d * 4), axis=1, bitorder="little").astype(np.float64)
    popcnt = bits.sum(axis=1)
    dots = bits @ bits.T
    if metric == "tanimoto":
        denom = popcnt[:, None] + popcnt[None, :] - dots
        sim = np.zeros_like(dots)
        np.divide(dots, denom, out=sim, where=denom > 0)
        sim[denom == 0] = 1.0
    elif metric == "cosine":
        denom = np.sqrt(popcnt[:, None] * popcnt[None, :])
        sim = np.zeros_like(dots)
        np.divide(dots, denom, out=sim, where=denom > 0)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return sim


def check_fused_butina_basic(clusters, n):
    """Structural sanity checks on clusters reconstructed from fused_butina output."""
    all_items = []
    for c in clusters:
        all_items.extend(c)
    assert sorted(all_items) == list(range(n)), "Not all items assigned exactly once"

    sizes = [len(c) for c in clusters]
    for i in range(len(sizes) - 1):
        assert sizes[i] >= sizes[i + 1], "Clusters not in non-increasing size order"


def fused_butina_clusters(x, cutoff, metric="tanimoto", stream=None):
    cluster_ids, centroids = fused_butina(
        x,
        cutoff,
        return_centroids=True,
        metric=metric,
        stream=stream,
    )
    cluster_ids = cluster_ids.numpy()
    centroids = centroids.numpy()

    clusters = []
    for cluster_id, centroid in enumerate(centroids):
        members = np.flatnonzero(cluster_ids == cluster_id)
        clusters.append(tuple([int(centroid)] + [int(member) for member in members if member != centroid]))
    return tuple(clusters)


# ---------------------------------------------------------------------------
# fused_butina tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,metric,num_words",
    [
        (50, "tanimoto", 32),
        (100, "tanimoto", 64),
        (200, "tanimoto", 32),
        (50, "tanimoto", 256),
        (50, "cosine", 32),
        (100, "cosine", 64),
        (200, "cosine", 32),
    ],
)
def test_fused_butina_basic_correctness(n, metric, num_words):
    x = generate_clustered_fingerprints(n, num_words=num_words, num_clusters=10)
    cutoff = 0.4
    clusters = fused_butina_clusters(x, cutoff=cutoff, metric=metric)

    check_fused_butina_basic(clusters, n)

    sim = compute_pairwise_similarity_cpu(x.cpu().numpy(), metric=metric)
    hit_mat = torch.tensor(sim >= (1.0 - cutoff), dtype=torch.bool).cuda()
    check_butina_correctness(hit_mat, clusters)


def test_fused_butina_single_item():
    x = torch.randint(-(2**31 - 1), 2**31 - 1, (1, 32), dtype=torch.int32).cuda()
    clusters = fused_butina_clusters(x, cutoff=0.5)
    assert len(clusters) == 1
    assert clusters[0] == (0,)


@pytest.mark.parametrize("metric", ["tanimoto", "cosine"])
def test_fused_butina_all_identical(metric):
    n = 50
    base = torch.randint(-(2**31 - 1), 2**31 - 1, (1, 32), dtype=torch.int32).cuda()
    x = base.expand(n, -1).contiguous()
    clusters = fused_butina_clusters(x, cutoff=0.5, metric=metric)
    assert len(clusters) == 1
    assert len(clusters[0]) == n
    assert set(clusters[0]) == set(range(n))


@pytest.mark.parametrize("metric", ["tanimoto", "cosine"])
def test_fused_butina_all_singletons(metric):
    n = 50
    generator = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randint(
        -(2**31 - 1),
        2**31 - 1,
        (n, 32),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    clusters = fused_butina_clusters(x, cutoff=0.001, metric=metric)
    assert len(clusters) == n
    for c in clusters:
        assert len(c) == 1


@pytest.mark.parametrize("n,metric", [(50, "tanimoto"), (50, "cosine"), (200, "tanimoto"), (200, "cosine")])
def test_fused_butina_returns_centroids(n, metric):
    cutoff = 0.4
    x = generate_clustered_fingerprints(n, num_words=32, num_clusters=10)
    cluster_ids, centroids = fused_butina(x, cutoff=cutoff, return_centroids=True, metric=metric)
    cluster_ids = cluster_ids.numpy()
    centroids = centroids.numpy()

    sim = compute_pairwise_similarity_cpu(x.cpu().numpy(), metric=metric)
    threshold = 1.0 - cutoff

    for cluster_id, centroid in enumerate(centroids):
        assert 0 <= centroid < n
        members = np.flatnonzero(cluster_ids == cluster_id)
        assert centroid in members
        for member in members:
            if member != centroid:
                assert sim[centroid, member] >= threshold - 1e-6


@pytest.mark.parametrize("input_kind", ["async", "cpu_tensor", "numpy"])
def test_fused_butina_accepts_array_input_types(input_kind):
    x = generate_clustered_fingerprints(50, num_words=32, num_clusters=10)
    cutoff = 0.4
    expected_cluster_ids = fused_butina(x, cutoff=cutoff).torch().cpu()

    if input_kind == "async":
        inp = AsyncGpuResult(x)
    elif input_kind == "cpu_tensor":
        inp = x.cpu()
    else:
        inp = x.cpu().numpy()

    cluster_ids = fused_butina(inp, cutoff=cutoff).torch().cpu()
    torch.testing.assert_close(cluster_ids, expected_cluster_ids)


def test_fused_butina_accepts_int32_and_uint32():
    fingerprints_int32 = generate_clustered_fingerprints(50, num_words=32, num_clusters=10)
    fingerprints_uint32 = fingerprints_int32.view(torch.uint32)

    expected_cluster_ids = fused_butina(fingerprints_int32, cutoff=0.4).torch()
    cluster_ids = fused_butina(fingerprints_uint32, cutoff=0.4).torch()

    torch.testing.assert_close(cluster_ids, expected_cluster_ids)


def test_fused_butina_on_explicit_stream():
    x = generate_clustered_fingerprints(100, num_words=32, num_clusters=10)
    expected = fused_butina(x, cutoff=0.4).torch()

    s = torch.cuda.Stream()
    actual = fused_butina(x, cutoff=0.4, stream=s).torch()
    s.synchronize()

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("metric", "expected_cluster_count"),
    [("tanimoto", 1), ("cosine", 10)],
)
def test_fused_butina_empty_fingerprints(metric, expected_cluster_count):
    x = torch.zeros((10, 32), dtype=torch.uint32, device="cuda")

    clusters = fused_butina_clusters(x, cutoff=0.5, metric=metric)

    assert len(clusters) == expected_cluster_count


def test_fused_butina_invalid_metric():
    x = torch.randint(-(2**31 - 1), 2**31 - 1, (10, 32), dtype=torch.int32).cuda()
    with pytest.raises(ValueError, match="metric must be one of"):
        fused_butina(x, cutoff=0.5, metric="euclidean")


def test_fused_butina_invalid_stream_type():
    x = torch.randint(-(2**31 - 1), 2**31 - 1, (10, 32), dtype=torch.int32).cuda()
    with pytest.raises(TypeError):
        fused_butina(x, cutoff=0.5, stream=42)
