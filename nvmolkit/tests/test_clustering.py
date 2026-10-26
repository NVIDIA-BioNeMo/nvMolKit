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

import numpy as np
import pytest
import torch
from rdkit.ML.Cluster.Butina import ClusterData

import nvmolkit.clustering as clustering
from nvmolkit.clustering import ClusterDeviceResult, OutputMode, bitbirch, butina, fused_butina
from nvmolkit.similarity import AAPMetric, CosineMetric, TanimotoMetric
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
    result = butina(
        torch.tensor(distance_matrix, device="cuda"),
        cutoff,
        reordering=reordering,
        output=OutputMode.DEVICE,
    )
    labels = result.cluster_ids.torch().cpu().numpy()
    centroids = result.centroids.torch().cpu().numpy()

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
    nvmol_res = butina(torch_dists, cutoff, neighborlist_max_size=neighborlist_max_size).cluster_ids.torch()
    nvmol_clusts = [tuple(torch.argwhere(nvmol_res == i).flatten().tolist()) for i in range(nvmol_res.max() + 1)]

    check_butina_correctness(torch_dists <= cutoff, nvmol_clusts)


@pytest.mark.parametrize("neighborlist_max_size", [8, 16, 24, 32, 64, 128])
def test_butina_edge_one_cluster(neighborlist_max_size):
    n = 10
    cutoff = 100.0
    torch_dists = torch.zeros((n, n), dtype=torch.float64, device="cuda")
    nvmol_res = butina(torch_dists, cutoff, neighborlist_max_size=neighborlist_max_size).cluster_ids.torch()
    assert torch.all(nvmol_res == 0)


@pytest.mark.parametrize("neighborlist_max_size", [8, 16, 24, 32, 64, 128])
def test_butina_edge_n_clusters(neighborlist_max_size):
    n = 10
    cutoff = 1e-8
    torch_dists = torch.ones((n, n), dtype=torch.float64, device="cuda")
    torch_dists.fill_diagonal_(0)
    nvmol_res = butina(torch_dists, cutoff, neighborlist_max_size=neighborlist_max_size).cluster_ids.torch()
    assert torch.all(nvmol_res.sort()[0] == torch.arange(10).to("cuda"))


def test_butina_returns_centroids():
    n = 25
    cutoff = 0.2
    dists = np.random.default_rng(123).random((n, n))
    dists = np.abs(dists - dists.T)
    torch_dists = torch.tensor(dists).to("cuda")
    result = butina(torch_dists, cutoff)
    cluster_ids_tensor = result.cluster_ids.torch()
    centroids_tensor = result.centroids.torch()

    num_clusters = int(cluster_ids_tensor.max().item()) + 1
    assert centroids_tensor.numel() == num_clusters

    adjacency = torch_dists <= cutoff
    for cluster_id in range(num_clusters):
        centroid = int(centroids_tensor[cluster_id].item())
        assert cluster_ids_tensor[centroid].item() == cluster_id
        members = torch.nonzero(cluster_ids_tensor == cluster_id, as_tuple=False).flatten()
        for member in members:
            assert adjacency[centroid, member].item()


@pytest.mark.parametrize("reordering", [False, True])
def test_butina_rdkit_output_matches_rdkit(reordering):
    distance_matrix = np.array(
        [
            [0.0, 0.2, 1.0, 1.0],
            [0.2, 0.0, 0.2, 1.0],
            [1.0, 0.2, 0.0, 0.2],
            [1.0, 1.0, 0.2, 0.0],
        ],
        dtype=np.float64,
    )

    got = butina(
        distance_matrix,
        0.2,
        reordering=reordering,
        output=OutputMode.RDKIT,
    )
    device_result = butina(
        distance_matrix,
        0.2,
        reordering=reordering,
        output=OutputMode.DEVICE,
    )

    assert got == _rdkit_butina_clusters(distance_matrix, 0.2, reordering=reordering)
    assert [cluster[0] for cluster in got] == device_result.centroids.numpy().tolist()


@pytest.mark.parametrize("explicit_output", [False, True])
def test_butina_device_output_has_fixed_result_type(explicit_output):
    dists = torch.tensor(
        [
            [0.0, 0.1, 1.0, 1.0],
            [0.1, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.1],
            [1.0, 1.0, 0.1, 0.0],
        ],
        dtype=torch.float64,
        device="cuda",
    )

    output_args = {"output": OutputMode.DEVICE} if explicit_output else {}
    result = butina(dists, 0.2, **output_args)

    assert isinstance(result, ClusterDeviceResult)
    torch.testing.assert_close(result.cluster_sizes.torch(), torch.tensor([2, 2], device="cuda"))
    assert result.cluster_ids.torch().shape == (4,)
    assert result.centroids.torch().shape == (2,)


def test_to_rdkit_clusters_preserves_cluster_and_member_order():
    cluster_ids = np.asarray([1, 0, 1, 2, 0, 2], dtype=np.int32)
    centroids = np.asarray([4, 2, 5], dtype=np.int32)

    assert clustering._cluster_arrays_to_rdkit(cluster_ids, centroids) == ((4, 1), (2, 0), (5, 3))


@pytest.mark.parametrize("input_kind", ["async", "cpu_tensor", "numpy"])
def test_butina_accepts_array_input_types(input_kind):
    n = 20
    cutoff = 0.2
    dists = np.random.default_rng(456).random((n, n))
    dists = np.abs(dists - dists.T)
    torch_dists = torch.tensor(dists, device="cuda", dtype=torch.float64)
    expected = butina(torch_dists, cutoff).cluster_ids.torch().cpu()

    if input_kind == "async":
        inp = AsyncGpuResult(torch_dists)
    elif input_kind == "cpu_tensor":
        inp = torch.tensor(dists, dtype=torch.float64)
    else:
        inp = dists

    got = butina(inp, cutoff).cluster_ids.torch().cpu()
    torch.testing.assert_close(got, expected)


def test_butina_on_explicit_stream():
    n = 100
    cutoff = 0.1
    dists = np.random.default_rng(42).random((n, n))
    dists = np.abs(dists - dists.T)
    torch_dists = torch.tensor(dists).to("cuda")

    s = torch.cuda.Stream()
    result = butina(torch_dists, cutoff, stream=s).cluster_ids.torch()
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


def test_butina_rejects_invalid_output():
    dists = torch.zeros(2, 2, dtype=torch.float64)
    with pytest.raises(TypeError, match="output must be an OutputMode"):
        butina(dists, 0.1, output="device")


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
    result = fused_butina(
        x,
        cutoff,
        metric=metric,
        stream=stream,
        output=OutputMode.DEVICE,
    )
    cluster_ids = result.cluster_ids.numpy()
    centroids = result.centroids.numpy()

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
    result = fused_butina(x, cutoff=cutoff, metric=metric, output=OutputMode.DEVICE)
    cluster_ids = result.cluster_ids.numpy()
    centroids = result.centroids.numpy()

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
    expected_cluster_ids = fused_butina(x, cutoff=cutoff, output=OutputMode.DEVICE).cluster_ids.torch().cpu()

    if input_kind == "async":
        inp = AsyncGpuResult(x)
    elif input_kind == "cpu_tensor":
        inp = x.cpu()
    else:
        inp = x.cpu().numpy()

    cluster_ids = fused_butina(inp, cutoff=cutoff, output=OutputMode.DEVICE).cluster_ids.torch().cpu()
    torch.testing.assert_close(cluster_ids, expected_cluster_ids)


def test_fused_butina_accepts_int32_and_uint32():
    fingerprints_int32 = generate_clustered_fingerprints(50, num_words=32, num_clusters=10)
    fingerprints_uint32 = fingerprints_int32.view(torch.uint32)

    expected_cluster_ids = fused_butina(fingerprints_int32, cutoff=0.4, output=OutputMode.DEVICE).cluster_ids.torch()
    cluster_ids = fused_butina(fingerprints_uint32, cutoff=0.4, output=OutputMode.DEVICE).cluster_ids.torch()

    torch.testing.assert_close(cluster_ids, expected_cluster_ids)


def test_fused_butina_on_explicit_stream():
    x = generate_clustered_fingerprints(100, num_words=32, num_clusters=10)
    expected = fused_butina(x, cutoff=0.4, output=OutputMode.DEVICE).cluster_ids.torch()

    s = torch.cuda.Stream()
    actual = fused_butina(x, cutoff=0.4, stream=s, output=OutputMode.DEVICE).cluster_ids.torch()
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
    with pytest.raises(ValueError, match="metric must be"):
        fused_butina(x, cutoff=0.5, metric="euclidean")


@pytest.mark.parametrize("name, instance", [("tanimoto", TanimotoMetric()), ("cosine", CosineMetric())])
def test_fused_butina_metric_objects_match_names(name, instance):
    x = torch.randint(0, 2**31 - 1, (200, 8), dtype=torch.int32).cuda()

    by_name = fused_butina(x, cutoff=0.6, metric=name, output=OutputMode.RDKIT)

    assert fused_butina(x, cutoff=0.6, metric=instance, output=OutputMode.RDKIT) == by_name


@pytest.mark.parametrize("metric", ["aap", AAPMetric()])
def test_fused_butina_does_not_yet_support_aap(metric):
    x = torch.zeros((2, 1), dtype=torch.int32).cuda()
    with pytest.raises(NotImplementedError, match="AAPMetric"):
        fused_butina(x, cutoff=0.5, metric=metric)


def test_fused_butina_invalid_stream_type():
    x = torch.randint(-(2**31 - 1), 2**31 - 1, (10, 32), dtype=torch.int32).cuda()
    with pytest.raises(TypeError):
        fused_butina(x, cutoff=0.5, stream=42)


def test_fused_butina_defaults_to_device_output():
    x = generate_clustered_fingerprints(50, num_words=32, num_clusters=10)

    default_result = fused_butina(x, cutoff=0.4)
    explicit_result = fused_butina(x, cutoff=0.4, output=OutputMode.DEVICE)

    assert isinstance(default_result, ClusterDeviceResult)
    torch.testing.assert_close(default_result.cluster_ids.torch(), explicit_result.cluster_ids.torch())
    torch.testing.assert_close(default_result.centroids.torch(), explicit_result.centroids.torch())
    torch.testing.assert_close(default_result.cluster_sizes.torch(), explicit_result.cluster_sizes.torch())


def test_fused_butina_rdkit_and_device_outputs_agree():
    x = generate_clustered_fingerprints(50, num_words=32, num_clusters=10)

    rdkit_result = fused_butina(x, cutoff=0.4, output=OutputMode.RDKIT)
    device_result = fused_butina(x, cutoff=0.4, output=OutputMode.DEVICE)

    assert isinstance(rdkit_result, tuple)
    assert isinstance(device_result, ClusterDeviceResult)
    assert rdkit_result == fused_butina_clusters(x, cutoff=0.4)
    assert [cluster[0] for cluster in rdkit_result] == device_result.centroids.numpy().tolist()
    assert device_result.cluster_sizes.torch().tolist() == [len(cluster) for cluster in rdkit_result]


# ---------------------------------------------------------------------------
# BitBIRCH tests
# ---------------------------------------------------------------------------


def test_bitbirch_merges_at_exact_diameter_threshold_and_returns_centroids():
    x = torch.tensor([[0b0011], [0b0001], [0b1100], [0b0100]], dtype=torch.int32, device="cuda")
    labels, centroids = bitbirch(x, threshold=0.5, branching_factor=3, return_centroids=True)
    torch.testing.assert_close(labels.torch().cpu(), torch.tensor([0, 0, 1, 1], dtype=torch.int32))
    torch.testing.assert_close(
        centroids.torch().cpu().view(torch.int32),
        torch.tensor([[0b0011], [0b1100]], dtype=torch.int32),
    )


@pytest.mark.parametrize("branching_factor", [3, 4, 7])
def test_bitbirch_cascading_splits_are_deterministic(branching_factor):
    x = (
        torch.ones(24, dtype=torch.int32, device="cuda") << torch.arange(24, dtype=torch.int32, device="cuda")
    ).reshape(-1, 1)
    labels = bitbirch(x, threshold=0.9, branching_factor=branching_factor).torch()
    torch.testing.assert_close(labels, torch.arange(24, dtype=torch.int32, device="cuda"))


@pytest.mark.parametrize("input_kind", ["async", "cpu_tensor", "numpy"])
def test_bitbirch_accepts_array_input_types(input_kind):
    x = torch.tensor([[3], [1], [12], [4]], dtype=torch.int32, device="cuda")
    if input_kind == "async":
        inp = AsyncGpuResult(x)
    elif input_kind == "cpu_tensor":
        inp = x.cpu()
    else:
        inp = x.cpu().numpy()
    labels = bitbirch(inp, threshold=0.5, branching_factor=3).torch().cpu()
    torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1], dtype=torch.int32))


def test_bitbirch_explicit_stream():
    x = torch.tensor([[3], [1], [12], [4]], dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    labels = bitbirch(x, threshold=0.5, branching_factor=3, stream=stream).torch()
    stream.synchronize()
    torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="cuda"))


def test_bitbirch_empty_and_all_zero_fingerprints():
    empty = torch.empty((0, 3), dtype=torch.uint32, device="cuda")
    empty_labels, empty_centroids = bitbirch(empty, threshold=0.5, return_centroids=True)
    assert empty_labels.torch().shape == (0,)
    assert empty_centroids.torch().shape == (0, 3)

    zeros = torch.zeros((12, 3), dtype=torch.uint32, device="cuda")
    labels, centroids = bitbirch(zeros, threshold=1.0, branching_factor=3, return_centroids=True)
    torch.testing.assert_close(labels.torch(), torch.zeros(12, dtype=torch.int32, device="cuda"))
    torch.testing.assert_close(centroids.torch(), torch.zeros((1, 3), dtype=torch.uint32, device="cuda"))


def test_bitbirch_threshold_extremes_and_all_one_majority_centroid():
    x = torch.tensor([[0], [1], [3], [0xFFFFFFFF]], dtype=torch.uint32, device="cuda")
    labels, centroids = bitbirch(x, threshold=0.0, branching_factor=3, return_centroids=True)
    torch.testing.assert_close(labels.torch(), torch.zeros(4, dtype=torch.int32, device="cuda"))
    torch.testing.assert_close(centroids.torch(), torch.tensor([[3]], dtype=torch.uint32, device="cuda"))

    duplicates = torch.full((17, 2), 0xFFFFFFFF, dtype=torch.uint32, device="cuda")
    labels, centroids = bitbirch(duplicates, threshold=1.0, return_centroids=True)
    torch.testing.assert_close(labels.torch(), torch.zeros(17, dtype=torch.int32, device="cuda"))
    torch.testing.assert_close(centroids.torch(), torch.full((1, 2), 0xFFFFFFFF, dtype=torch.uint32, device="cuda"))


def test_bitbirch_accepts_noncontiguous_packed_input():
    storage = torch.tensor(
        [[3, 99, 1, 99], [1, 99, 1, 99], [12, 99, 4, 99], [4, 99, 4, 99]],
        dtype=torch.uint32,
        device="cuda",
    )
    x = storage[:, ::2]
    assert not x.is_contiguous()
    labels = bitbirch(x, threshold=0.5, branching_factor=3).torch()
    torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="cuda"))


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"threshold": -0.1}, r"threshold must be finite and in \[0, 1\]"),
        ({"threshold": np.nan}, r"threshold must be finite and in \[0, 1\]"),
        ({"threshold": np.inf}, r"threshold must be finite and in \[0, 1\]"),
        ({"threshold": 0.5, "branching_factor": 2}, r"branching_factor must be in \[3, 65534\]"),
        ({"threshold": 0.5, "branching_factor": 65535}, r"branching_factor must be in \[3, 65534\]"),
        ({"threshold": 0.5, "batch_size": 0}, "batch_size must be positive"),
    ],
)
def test_bitbirch_validation(kwargs, error):
    x = torch.zeros((2, 1), dtype=torch.int32)
    with pytest.raises(ValueError, match=error):
        bitbirch(x, **kwargs)


def test_bitbirch_rejects_zero_word_fingerprints():
    x = torch.empty((4, 0), dtype=torch.uint32, device="cuda")
    with pytest.raises(ValueError, match="at least one fingerprint word"):
        bitbirch(x, threshold=0.5)
