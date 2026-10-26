# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from _bitbirch_reference import BatchedBitBirch
from rdkit.Chem import rdFingerprintGenerator

from nvmolkit.clustering import bitbirch
from nvmolkit.fingerprints import MorganFingerprintGenerator


def _reference(packed, threshold, branching_factor, batch_size):
    byte_rows = packed.view(np.uint8).reshape(len(packed), packed.shape[1] * 4)
    tree = BatchedBitBirch(threshold, branching_factor, batch_size).fit(byte_rows)
    tree.audit()
    centroids = np.stack([entry.packed for entry in tree.clusters()]).view(np.uint32)
    return tree.labels(), centroids


def test_rdkit_fingerprints_to_bitbirch_matches_independent_schedule(size_limited_mols):
    mols = size_limited_mols[:80]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    bits = np.asarray([generator.GetFingerprint(mol).ToList() for mol in mols], dtype=np.uint8)
    packed = np.packbits(bits, axis=1, bitorder="little").view(np.uint32)
    expected_labels, expected_centroids = _reference(packed, 0.25, 7, 32)

    labels, centroids = bitbirch(
        packed,
        0.25,
        branching_factor=7,
        batch_size=32,
        return_centroids=True,
    )

    np.testing.assert_array_equal(labels.numpy(), expected_labels)
    np.testing.assert_array_equal(centroids.numpy(), expected_centroids)


def test_gpu_fingerprint_generation_and_clustering_chain_on_explicit_stream(size_limited_mols):
    mols = size_limited_mols[:32] * 2
    stream = torch.cuda.Stream()
    fingerprints = MorganFingerprintGenerator(radius=3, fpSize=512).GetFingerprints(
        mols,
        num_threads=1,
        stream=stream,
    )
    labels = bitbirch(
        fingerprints,
        threshold=1.0,
        branching_factor=4,
        batch_size=32,
        stream=stream,
    ).torch()
    stream.synchronize()

    packed = fingerprints.torch().cpu().numpy().view(np.uint32)
    expected, _ = _reference(packed, 1.0, 4, 32)
    np.testing.assert_array_equal(labels.cpu().numpy(), expected)


def test_host_rdkit_and_device_nvmolkit_fingerprints_cluster_identically(size_limited_mols):
    mols = size_limited_mols[:40]
    rdkit_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
    rdkit_bits = np.asarray([rdkit_generator.GetFingerprint(mol).ToList() for mol in mols], dtype=np.uint8)
    host_fingerprints = np.packbits(rdkit_bits, axis=1, bitorder="little").view(np.uint32)
    device_fingerprints = MorganFingerprintGenerator(radius=2, fpSize=256).GetFingerprints(mols, num_threads=1)

    host_labels = bitbirch(host_fingerprints, threshold=0.5, branching_factor=7, batch_size=16).numpy()
    device_labels = bitbirch(device_fingerprints, threshold=0.5, branching_factor=7, batch_size=16).numpy()

    np.testing.assert_array_equal(device_labels, host_labels)
