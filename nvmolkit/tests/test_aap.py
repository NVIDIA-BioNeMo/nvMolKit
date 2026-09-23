# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings

import numpy as np
import pytest
import torch
from rdkit import Chem

from nvmolkit.clustering import (
    ClusterDeviceResult,
    OutputMode,
    aap_dise,
)
from nvmolkit.similarity import AAPMetric, TanimotoMetric, aap_similarity


def _mol(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return molecule


def _cluster_ids(molecules, similarity_threshold=0.217, assignment="first", stream=None, **options):
    result = aap_dise(
        molecules,
        similarity_threshold=similarity_threshold,
        assignment=assignment,
        metric=AAPMetric(**options),
        stream=stream,
    )
    return result.cluster_ids.numpy().tolist()


def test_aap_similarity_is_one_for_identical_and_renumbered_molecules():
    molecule = _mol("CC(=O)Oc1ccccc1C(=O)O")
    renumbered = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))

    assert aap_similarity(molecule, molecule) == 1.0
    assert aap_similarity(molecule, renumbered) == 1.0
    assert aap_similarity(renumbered, molecule) == 1.0
    assert _cluster_ids([molecule, renumbered], similarity_threshold=1.0) == [0, 0]


def test_aap_similarity_has_expected_directed_numerical_result():
    ethane = _mol("CC")
    propane = _mol("CCC")

    assert aap_similarity(ethane, propane) == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert aap_similarity(propane, ethane) == pytest.approx(1.0 / 5.0, abs=1e-6)


def test_aap_similarity_distinguishes_atom_bond_aromaticity_and_topology():
    assert aap_similarity(_mol("C"), _mol("N")) == 0.0
    assert aap_similarity(_mol("c1ccccc1"), _mol("C1CCCCC1")) == 0.0

    single_to_double = aap_similarity(_mol("CC"), _mol("C=C"))
    single_to_triple = aap_similarity(_mol("CC"), _mol("C#N"))
    chain_to_branch = aap_similarity(_mol("CCCC"), _mol("CC(C)C"))
    assert single_to_double == pytest.approx(1.0 / 5.0, abs=1e-6)
    assert single_to_triple == pytest.approx(1.0 / 11.0, abs=1e-6)
    assert single_to_double > single_to_triple
    assert 0.0 < chain_to_branch < 1.0


def test_aap_similarity_is_finite_and_bounded_over_varied_chemistry():
    molecules = [
        _mol(smiles)
        for smiles in (
            "C",
            "CCO",
            "CC(=O)O",
            "CC(C)C",
            "c1ccccc1",
            "c1ccncc1",
            "C1CCCCC1",
            "[NH4+]",
            "ClCCCl",
        )
    ]

    scores = [aap_similarity(left, right) for left in molecules for right in molecules]
    assert all(math.isfinite(score) for score in scores)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert all(scores[index * len(molecules) + index] == 1.0 for index in range(len(molecules)))


@pytest.mark.parametrize(
    "option, value",
    [
        ("max_path_length", 1),
        ("histogram_bins", 1),
        ("sinkhorn_iterations", 1),
        ("sinkhorn_temperature", 0.5),
    ],
)
def test_aap_similarity_options_change_the_computation(option, value):
    left = _mol("CCO")
    right = _mol("CCN")
    baseline = aap_similarity(left, right)
    configured = aap_similarity(left, right, metric=AAPMetric(**{option: value}))

    assert 0.0 <= configured <= 1.0
    assert configured != pytest.approx(baseline, abs=1e-3)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_path_length": 0}, "maxPathLength must be positive"),
        ({"histogram_bins": 0}, "histogramBins must be between 1 and 32767"),
        ({"histogram_bins": 32768}, "histogramBins must be between 1 and 32767"),
        ({"sinkhorn_iterations": 0}, "sinkhornIterations must be positive"),
        ({"sinkhorn_temperature": 0.0}, "sinkhornTemperature must be finite and positive"),
        ({"sinkhorn_temperature": float("nan")}, "sinkhornTemperature must be finite and positive"),
        ({"sinkhorn_temperature": float("inf")}, "sinkhornTemperature must be finite and positive"),
        (
            {"sinkhorn_temperature": 1e-40},
            "sinkhornTemperature must be at least the smallest positive normal float",
        ),
    ],
)
def test_aap_options_are_validated_for_pair_and_empty_clustering(kwargs, message):
    molecule = _mol("CCO")

    with pytest.raises(ValueError, match=message):
        aap_similarity(molecule, molecule, metric=AAPMetric(**kwargs))
    with pytest.raises(ValueError, match=message):
        aap_dise([], metric=AAPMetric(**kwargs))


def test_aap_accepts_smallest_positive_normal_sinkhorn_temperature():
    temperature = float(np.finfo(np.float32).tiny)
    molecules = [_mol("CCO"), _mol("CCN")]

    score = aap_similarity(*molecules, metric=AAPMetric(sinkhorn_temperature=temperature))

    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0
    assert _cluster_ids(molecules, sinkhorn_temperature=temperature) == [0, 1]


def test_aap_supports_paths_beyond_the_reference_default():
    molecules = [_mol("CCCCCCCCC"), _mol("CCCCCCCCO")]

    score = aap_similarity(*molecules, metric=AAPMetric(max_path_length=8))

    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_aap_clustering_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="threshold must be in"):
        aap_dise([], similarity_threshold=threshold)


def test_aap_rejects_empty_oversized_null_and_unsupported_molecules():
    molecule = _mol("CCO")
    empty = Chem.RWMol().GetMol()
    oversized = _mol("C" * 65)
    unsupported = Chem.RWMol()
    unsupported.AddAtom(Chem.Atom(6))
    unsupported.AddAtom(Chem.Atom(6))
    unsupported.AddBond(0, 1, Chem.BondType.UNSPECIFIED)

    with pytest.raises(ValueError, match="does not support empty molecules"):
        aap_similarity(empty, molecule)
    with pytest.raises(ValueError, match="at most 64 atoms"):
        aap_similarity(oversized, oversized)
    with pytest.raises(ValueError, match="Invalid molecule at index 0"):
        aap_dise([None])
    with pytest.raises(ValueError, match="supports only single, double, triple, and aromatic bonds"):
        aap_similarity(unsupported.GetMol(), molecule)


def test_aap_supports_the_64_atom_boundary():
    molecule = _mol("C" * 64)

    assert _cluster_ids([molecule]) == [0]


def test_aap_clustering_handles_empty_singleton_and_generator_inputs():
    ethanol = _mol("CCO")
    benzene = _mol("c1ccccc1")

    assert _cluster_ids([]) == []
    assert _cluster_ids([ethanol]) == [0]
    molecules = (molecule for molecule in (ethanol, ethanol, benzene))
    assert _cluster_ids(molecules, similarity_threshold=1.0) == [0, 0, 1]


def test_aap_clustering_threshold_is_inclusive():
    ethane = _mol("CC")
    propane = _mol("CCC")
    score = aap_similarity(ethane, propane)

    assert _cluster_ids([ethane, propane], similarity_threshold=score) == [0, 0]
    assert _cluster_ids([ethane, propane], similarity_threshold=score + 1e-4) == [0, 1]


def test_aap_clustering_is_directed_and_uses_input_order_centroids():
    ethane = _mol("CC")
    propane = _mol("CCC")

    assert _cluster_ids([ethane, propane], similarity_threshold=0.25) == [0, 0]
    assert _cluster_ids([propane, ethane], similarity_threshold=0.25) == [0, 1]


def test_aap_clustering_is_centroid_based_not_transitive():
    ethane = _mol("CC")
    propane = _mol("CCC")
    butane = _mol("CCCC")

    assert aap_similarity(ethane, propane) >= 0.3
    assert aap_similarity(propane, butane) >= 0.3
    assert aap_similarity(ethane, butane) < 0.3
    assert _cluster_ids([ethane, propane, butane], similarity_threshold=0.3) == [0, 0, 1]


def test_aap_clustering_renumbers_clusters_by_size_then_centroid_order():
    benzene = _mol("c1ccccc1")
    ethanol = _mol("CCO")
    propane = _mol("CCC")
    molecules = [benzene, ethanol, ethanol, ethanol, propane, propane]

    assert _cluster_ids(molecules, similarity_threshold=1.0) == [2, 0, 0, 0, 1, 1]


def test_aap_clustering_threshold_zero_assigns_every_molecule_to_first_centroid():
    molecules = [_mol(smiles) for smiles in ("CCO", "c1ccccc1", "[NH4+]", "ClCCCl")]

    assert _cluster_ids(molecules, similarity_threshold=0.0) == [0, 0, 0, 0]


def test_aap_dise_nearest_assignment_handles_empty_singleton_duplicates_and_generator():
    ethanol = _mol("CCO")
    benzene = _mol("c1ccccc1")

    assert _cluster_ids([], assignment="nearest") == []
    assert _cluster_ids([ethanol], assignment="nearest") == [0]
    molecules = (molecule for molecule in (ethanol, ethanol, benzene))
    assert _cluster_ids(molecules, similarity_threshold=1.0, assignment="nearest") == [0, 0, 1]


def test_aap_dise_reassigns_noncentroids_to_the_nearest_centroid():
    molecules = [_mol(smiles) for smiles in ("CCCC", "CCCO", "CCOC")]

    # The selection pass absorbs CCCO into the first sphere, while CCOC becomes
    # a second centroid. The complete directed sphere exclusion workflow then
    # assigns CCCO to CCOC because that selected centroid is more similar.
    assert _cluster_ids(molecules, similarity_threshold=0.2) == [0, 0, 1]
    assert aap_similarity(molecules[2], molecules[1]) > aap_similarity(molecules[0], molecules[1])
    assert _cluster_ids(molecules, similarity_threshold=0.2, assignment="nearest") == [1, 0, 0]


@pytest.mark.parametrize(
    "assignment, expected_ids, expected_centroids, expected_sizes, expected_clusters",
    [
        ("first", [0, 0, 1], [0, 2], [2, 1], ((0, 1), (2,))),
        ("nearest", [1, 0, 0], [2, 0], [2, 1], ((2, 1), (0,))),
    ],
)
def test_aap_dise_output_modes_share_one_cluster_contract(
    assignment, expected_ids, expected_centroids, expected_sizes, expected_clusters
):
    molecules = [_mol(smiles) for smiles in ("CCCC", "CCCO", "CCOC")]

    device = aap_dise(molecules, similarity_threshold=0.2, assignment=assignment)
    rdkit = aap_dise(
        molecules,
        similarity_threshold=0.2,
        assignment=assignment,
        output=OutputMode.RDKIT,
    )

    assert isinstance(device, ClusterDeviceResult)
    assert device.cluster_ids.torch().dtype == torch.int32
    assert device.centroids.torch().dtype == torch.int32
    assert device.cluster_sizes.torch().dtype == torch.int64
    assert device.cluster_ids.numpy().tolist() == expected_ids
    assert device.centroids.numpy().tolist() == expected_centroids
    assert device.cluster_sizes.numpy().tolist() == expected_sizes
    assert rdkit == expected_clusters


def test_aap_dise_output_modes_handle_empty_input():
    device = aap_dise([])

    assert device.cluster_ids.torch().shape == (0,)
    assert device.centroids.torch().shape == (0,)
    assert device.cluster_sizes.torch().shape == (0,)
    assert aap_dise([], output=OutputMode.RDKIT) == ()


@pytest.mark.parametrize("output", ["device", None, ClusterDeviceResult])
def test_aap_dise_rejects_invalid_output(output):
    with pytest.raises(TypeError, match="output must be an OutputMode"):
        aap_dise([], output=output)


@pytest.mark.parametrize("assignment", ["", "closest", None])
def test_aap_dise_rejects_invalid_assignment(assignment):
    with pytest.raises(ValueError, match="assignment must be one of"):
        aap_dise([], assignment=assignment)


def test_aap_similarity_and_clustering_are_deterministic_across_streams():
    left = _mol("CCO")
    right = _mol("CCN")
    molecules = [left, left, right, _mol("c1ccccc1")]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    scores = [aap_similarity(left, right, stream=stream) for stream in streams]
    labels = [_cluster_ids(molecules, similarity_threshold=0.2, stream=stream) for stream in streams]
    assert scores[0] == scores[1]
    assert labels[0] == labels[1]


def test_aap_dise_device_output_on_explicit_stream_matches_default():
    molecules = [_mol(smiles) for smiles in ("CCCC", "CCCO", "CCOC", "c1ccccc1")]
    expected = aap_dise(molecules, similarity_threshold=0.2)
    stream = torch.cuda.Stream()

    actual = aap_dise(molecules, similarity_threshold=0.2, stream=stream)

    assert actual.cluster_ids.device == stream.device
    assert actual.centroids.device == stream.device
    assert actual.cluster_sizes.device == stream.device
    torch.testing.assert_close(actual.cluster_ids.torch(), expected.cluster_ids.torch())
    torch.testing.assert_close(actual.centroids.torch(), expected.centroids.torch())
    torch.testing.assert_close(actual.cluster_sizes.torch(), expected.cluster_sizes.torch())


def test_aap_similarity_tracks_optional_rdkit_reference_over_corpus():
    reference = pytest.importorskip("rdkit.Contrib.AtomAtomSimilarity.AtomAtomPathSimilarity")
    molecules = [
        _mol(smiles)
        for smiles in (
            "CCO",
            "CCN",
            "CCC",
            "CCCO",
            "CC(=O)O",
            "COC",
            "c1ccccc1",
            "c1ccncc1",
            "c1ccccc1O",
            "C1CCCCC1",
            "CC(C)C",
            "ClCCCl",
        )
    ]
    pairs = [(left, right) for index, left in enumerate(molecules) for right in molecules[index + 1 :]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.asarray([reference.AtomAtomPathSimilarity(left, right) for left, right in pairs])
    actual = np.asarray([aap_similarity(left, right) for left, right in pairs])

    assert np.corrcoef(expected, actual)[0, 1] >= 0.95
    assert np.mean(np.abs(expected - actual)) <= 0.03


def test_aap_matches_optional_ligand_clustering_source():
    reference = pytest.importorskip("gpu_ligand_clustering.aap")
    smiles = ["C", "CC", "CCC", "CCCC", "CCO", "CCN", "c1ccccc1", "c1ccncc1"]
    molecules = [_mol(value) for value in smiles]
    pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)]

    expected = [
        reference.aap_similarity(smiles[left], smiles[right], sinkhorn_execution="eager") for left, right in pairs
    ]
    actual = [aap_similarity(molecules[left], molecules[right]) for left, right in pairs]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)

    for threshold in (0.2, 0.217, 0.3, 0.5):
        expected_labels = reference.aap_similarity_clustering(
            smiles, dist_thresh=threshold, sinkhorn_execution="eager"
        )
        actual_labels = [cluster_id + 1 for cluster_id in _cluster_ids(molecules, similarity_threshold=threshold)]
        assert actual_labels == expected_labels


def test_aap_metric_name_is_equivalent_to_default_metric_object():
    molecules = [_mol(smiles) for smiles in ("CCCC", "CCCO", "CCOC", "c1ccccc1")]

    assert aap_similarity(molecules[0], molecules[1], metric="aap") == aap_similarity(
        molecules[0], molecules[1], metric=AAPMetric()
    )
    assert aap_dise(molecules, 0.2, metric="aap", output=OutputMode.RDKIT) == aap_dise(
        molecules, 0.2, metric=AAPMetric(), output=OutputMode.RDKIT
    )


@pytest.mark.parametrize(
    ("metric", "error"), [("tanimoto", TypeError), (TanimotoMetric(), TypeError), ("euclidean", ValueError)]
)
def test_aap_apis_reject_other_metrics(metric, error):
    molecule = _mol("CCO")

    with pytest.raises(error, match="metric must be"):
        aap_similarity(molecule, molecule, metric=metric)
    with pytest.raises(error, match="metric must be"):
        aap_dise([molecule], metric=metric)
