# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for batched BitBIRCH against an independent CPU model."""

import numpy as np
import pytest
import torch
from _bitbirch_reference import ORDERED_WARMUP_SIZE, BatchedBitBirch

from nvmolkit.clustering import bitbirch


def reference(packed, threshold, branching_factor, batch_size):
    bytes_ = packed.view(np.uint8).reshape(len(packed), packed.shape[1] * 4)
    tree = BatchedBitBirch(threshold, branching_factor, batch_size).fit(bytes_)
    tree.audit()
    centroids = np.stack([entry.packed for entry in tree.clusters()]).view(np.uint32).reshape(-1, packed.shape[1])
    return tree.labels(), centroids


def assert_matches_reference(packed, threshold, branching_factor, batch_size):
    expected, expected_centroids = reference(packed, threshold, branching_factor, batch_size)
    labels, centroids = bitbirch(
        packed, threshold, branching_factor=branching_factor, batch_size=batch_size, return_centroids=True
    )
    np.testing.assert_array_equal(labels.numpy(), expected)
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids)


@pytest.mark.parametrize("branching_factor,batch_size,words", [(3, 7, 1), (7, 32, 3), (254, 32, 32)])
@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.7, 1.0])
def test_matches_independent_reference(branching_factor, batch_size, words, threshold):
    packed = np.random.default_rng(740 + words).integers(0, 2**32, size=(97, words), dtype=np.uint32)
    packed[30:40] = packed[:10]
    assert_matches_reference(packed, threshold, branching_factor, batch_size)


@pytest.mark.parametrize("words", [127, 128])
def test_widest_fingerprints_match_reference(words):
    packed = np.random.default_rng(202 + words).integers(0, 2**32, size=(43, words), dtype=np.uint32)
    packed[30:40] = packed[:10]
    assert_matches_reference(packed, 0.4, 3, 16)


@pytest.mark.parametrize("branching_factor,words", [(63, 2), (64, 2), (257, 1), (256, 64), (300, 32), (600, 64)])
def test_leaves_spanning_split_tiles_match_reference(branching_factor, words):
    # Overflowing leaves of branching_factor + 1 entries cover one or more 64-entry split tiles.
    rows = 4 * branching_factor + 77
    packed = np.random.default_rng(1234 + branching_factor).integers(0, 2**32, size=(rows, words), dtype=np.uint32)
    packed[rows // 2 : rows // 2 + 60] = packed[:60]
    assert_matches_reference(packed, 0.55, branching_factor, 512)


def test_default_branching_factor_matches_reference():
    packed = np.random.default_rng(5).integers(0, 2**32, size=(4 * 1025 + 77, 32), dtype=np.uint32)
    packed[2000:2100] = packed[:100]
    assert_matches_reference(packed, 0.55, 1024, 512)


def test_wide_nodes_with_tied_similarities_match_reference():
    # One-word fingerprints give few distinct similarities, so ties decide most choices.
    packed = np.random.default_rng(17).integers(0, 2**32, size=(5000, 1), dtype=np.uint32)
    assert_matches_reference(packed, 1.0, 2048, 1024)


@pytest.mark.parametrize("threshold", [0.6, 1.0])
def test_cascading_splits_of_multi_tile_inner_nodes_match_reference(threshold):
    # Over 70 leaves force inner nodes of 71 entries (two split tiles) to split as well.
    packed = np.random.default_rng(41).integers(0, 2**32, size=(6000, 4), dtype=np.uint32)
    assert_matches_reference(packed, threshold, 70, 512)


def test_grouped_merges_after_warmup_match_reference():
    # Clustered rows past the ordered warm-up, in batches that do not divide it.
    rng = np.random.default_rng(5)
    base = rng.integers(0, 2**32, size=(400, 2), dtype=np.uint32)
    noise = np.bitwise_and.reduce([rng.integers(0, 2**32, size=(20000, 2), dtype=np.uint32) for _ in range(3)])
    packed = base[rng.integers(0, len(base), len(noise))] ^ noise
    assert len(packed) > ORDERED_WARMUP_SIZE and ORDERED_WARMUP_SIZE % 1000 != 0
    assert_matches_reference(packed, 0.5, 50, 1000)


def test_split_wider_than_one_block_matches_reference():
    packed = np.random.default_rng(82).integers(0, 2**32, size=(529, 1), dtype=np.uint32)
    assert_matches_reference(packed, 1.0, 257, 512)


@pytest.mark.parametrize("branching_factor", [254, 255])
def test_full_1024bit_leaves_match_reference(branching_factor):
    packed = np.random.default_rng(991).integers(0, 2**32, size=(529, 32), dtype=np.uint32)
    packed[510:520] = packed[:10]
    assert_matches_reference(packed, 0.55, branching_factor, 512)


def test_multiple_owners_and_cascading_splits():
    packed = np.random.default_rng(331).integers(0, 2**32, size=(43, 2), dtype=np.uint32)
    packed[29:36] = packed[:7]
    assert_matches_reference(packed, 0.4, 3, 16)


def test_concurrent_leaf_allocation_is_schedule_independent():
    packed = np.random.default_rng(73).integers(0, 2**32, size=(503, 3), dtype=np.uint32)
    expected, _ = reference(packed, 0.55, 3, 128)
    for _ in range(5):
        labels = bitbirch(packed, 0.55, branching_factor=3, batch_size=128).numpy()
        np.testing.assert_array_equal(labels, expected)


def test_empty_zeros_and_duplicates():
    labels, centroids = bitbirch(np.zeros((0, 2), dtype=np.uint32), 0.5, return_centroids=True)
    assert labels.numpy().shape == (0,)
    assert centroids.numpy().shape == (0, 2)
    packed = np.zeros((65537, 1), dtype=np.uint32)
    labels, centroids = bitbirch(packed, 1.0, return_centroids=True)
    np.testing.assert_array_equal(labels.numpy(), np.zeros(len(packed), dtype=np.int32))
    np.testing.assert_array_equal(centroids.numpy(), np.zeros((1, 1), dtype=np.uint32))


def test_explicit_stream_orders_input_clustering_and_consumption():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        packed = torch.tensor([[1], [2], [4], [8], [1], [2], [4], [8]], dtype=torch.int32, device="cuda")
        labels = bitbirch(packed, 1.0, branching_factor=3, batch_size=4, stream=stream)
        copied = labels.torch().clone()
    stream.synchronize()
    expected, _ = reference(packed.cpu().numpy().view(np.uint32), 1.0, 3, 4)
    np.testing.assert_array_equal(copied.cpu().numpy(), expected)


def test_memory_mapped_input_matches_device_input(tmp_path):
    packed = np.random.default_rng(871).integers(0, 2**32, (529, 3), dtype=np.uint32)
    packed[250:300] = packed[:50]
    path = tmp_path / "fingerprints.npy"
    np.save(path, packed)
    mapped = np.load(path, mmap_mode="r")
    options = {"branching_factor": 3, "batch_size": 32, "return_centroids": True}
    expected, expected_centroids = bitbirch(torch.from_numpy(packed.view(np.int32)).cuda(), 0.4, **options)
    labels, centroids = bitbirch(mapped, 0.4, **options)
    np.testing.assert_array_equal(labels.numpy(), expected.numpy())
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids.numpy())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": float("nan")},
        {"threshold": -1},
        {"threshold": 0.5, "branching_factor": 2},
        {"threshold": 0.5, "batch_size": 0},
    ],
)
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        bitbirch(np.zeros((1, 1), dtype=np.uint32), **kwargs)


def test_fingerprints_wider_than_4096_bits_rejected():
    with pytest.raises(ValueError, match="at most 128 fingerprint words"):
        bitbirch(np.zeros((4, 129), dtype=np.uint32), 0.5)
