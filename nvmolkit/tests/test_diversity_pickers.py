# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import weakref
from functools import cache

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.SimDivFilters import rdSimDivPickers

from nvmolkit import _clustering, _distance_inputs, clustering, pickers
from nvmolkit.clustering import (
    ClusterDeviceResult,
    OutputMode,
    dise,
    fused_butina,
    fused_dise,
)
from nvmolkit.pickers import fused_leader, fused_maxmin, leader, maxmin
from nvmolkit.similarity import AAPMetric, CosineMetric, TanimotoMetric, aap_similarity, crossTanimotoSimilarity
from nvmolkit.types import AsyncGpuResult

RDKIT = OutputMode.RDKIT


def _reference_leader(distances, cutoff, pick_size=0, first_picks=()):
    distances = distances.astype(np.float32)
    cutoff = np.float32(cutoff)
    selected = list(first_picks)
    for candidate in range(len(distances)):
        if pick_size and len(selected) >= pick_size:
            break
        if candidate in selected:
            continue
        if all(distances[picked, candidate] > cutoff for picked in selected):
            selected.append(candidate)
    return tuple(selected)


@pytest.fixture(scope="module")
def aap_molecules(chembl_molecules):
    """110 AAP-compatible molecules plus atom-renumbered copies of two of them."""
    molecules = [molecule for molecule in chembl_molecules if molecule.GetNumAtoms() <= 64][:110]
    duplicates = [
        Chem.RenumberAtoms(molecules[index], list(reversed(range(molecules[index].GetNumAtoms()))))
        for index in (5, 17)
    ]
    return molecules + duplicates


def _reference_dise(distances, cutoff, assignment):
    """Direct NumPy DISE in RDKit cluster format."""
    cutoff = distances.dtype.type(cutoff)
    num_items = len(distances)
    active = np.ones(num_items, dtype=bool)
    labels = np.full(num_items, -1)
    centroids = []
    for item in range(num_items):
        if not active[item]:
            continue
        excluded = active & (distances[item] <= cutoff)
        excluded[item] = True
        labels[excluded] = len(centroids)
        active &= ~excluded
        centroids.append(item)
    if assignment == "nearest":
        non_centroids = np.ones(num_items, dtype=bool)
        non_centroids[centroids] = False
        labels[non_centroids] = np.argmin(distances[centroids], axis=0)[non_centroids]
    sizes = np.bincount(labels, minlength=len(centroids))
    order = sorted(range(len(centroids)), key=lambda cluster: -sizes[cluster])
    return tuple(
        (centroids[cluster], *(int(item) for item in np.flatnonzero(labels == cluster) if item != centroids[cluster]))
        for cluster in order
    )


def _fingerprint_distance_matrix(fingerprints, metric):
    counts = np.asarray([sum(int(word).bit_count() for word in row) for row in fingerprints])
    result = np.zeros((len(fingerprints), len(fingerprints)), dtype=np.float64)
    for left in range(len(fingerprints)):
        for right in range(len(fingerprints)):
            intersection = sum(
                (int(left_word) & int(right_word)).bit_count()
                for left_word, right_word in zip(fingerprints[left], fingerprints[right], strict=True)
            )
            if metric == "tanimoto":
                denominator = counts[left] + counts[right] - intersection
                similarity = intersection / denominator if denominator else 1.0
            else:
                denominator = np.sqrt(counts[left] * counts[right])
                similarity = intersection / denominator if denominator else 0.0
            result[left, right] = 1.0 - similarity
    return result


# ---------------------------------------------------------------------------
# Integration: ChEMBL fingerprints against RDKit and the matrix APIs
# ---------------------------------------------------------------------------


# Distinct Tanimoto values keep their order in float32, so results equal RDKit's double-precision pickers except
# where a distance equals a cutoff that float32 cannot represent exactly. RDKit parity tests use dyadic cutoffs.
@pytest.mark.parametrize("cutoff", [0.0, 0.25, 0.5, 0.625, 1.0])
def test_leader_matches_rdkit_on_chembl(chembl_fingerprints, chembl_distances, cutoff):
    bit_vectors, _ = chembl_fingerprints
    expected = tuple(rdSimDivPickers.LeaderPicker().LazyBitVectorPick(bit_vectors, len(bit_vectors), cutoff))

    assert leader(chembl_distances["tanimoto"], cutoff, output=RDKIT) == expected


@pytest.mark.parametrize("cutoff", [0.0, 0.25, 0.5, 0.625, 1.0])
def test_fused_leader_matches_rdkit_on_chembl(chembl_fingerprints, cutoff):
    bit_vectors, packed = chembl_fingerprints
    expected = tuple(rdSimDivPickers.LeaderPicker().LazyBitVectorPick(bit_vectors, len(bit_vectors), cutoff))

    assert fused_leader(packed, cutoff, output=RDKIT) == expected


@pytest.mark.parametrize("num_first_picks", [3, 40])
def test_leader_first_picks_and_pick_size_match_rdkit_on_chembl(
    chembl_fingerprints, chembl_distances, num_first_picks
):
    bit_vectors, _ = chembl_fingerprints
    first_picks = [int(index) for index in np.random.default_rng(5).permutation(len(bit_vectors))[:num_first_picks]]
    expected = tuple(
        rdSimDivPickers.LeaderPicker().LazyBitVectorPick(
            bit_vectors, len(bit_vectors), 0.5, pickSize=50, firstPicks=first_picks
        )
    )

    assert leader(chembl_distances["tanimoto"], 0.5, pick_size=50, first_picks=first_picks, output=RDKIT) == expected


@pytest.mark.parametrize("num_first_picks", [3, 40])
def test_fused_leader_first_picks_and_pick_size_match_rdkit_on_chembl(chembl_fingerprints, num_first_picks):
    bit_vectors, packed = chembl_fingerprints
    first_picks = [int(index) for index in np.random.default_rng(5).permutation(len(bit_vectors))[:num_first_picks]]
    expected = tuple(
        rdSimDivPickers.LeaderPicker().LazyBitVectorPick(
            bit_vectors, len(bit_vectors), 0.5, pickSize=50, firstPicks=first_picks
        )
    )

    assert fused_leader(packed, 0.5, pick_size=50, first_picks=first_picks, output=RDKIT) == expected


@pytest.mark.parametrize(
    "pick_size, first_picks, seed",
    [(200, (), 42), (200, (), 0), (100, (7, 900, 3), -1), (999, (), 5)],
)
def test_fused_maxmin_matches_rdkit_on_chembl(chembl_fingerprints, pick_size, first_picks, seed):
    bit_vectors, packed = chembl_fingerprints
    expected = tuple(
        rdSimDivPickers.MaxMinPicker().LazyBitVectorPick(
            bit_vectors, len(bit_vectors), pick_size, firstPicks=list(first_picks), seed=seed
        )
    )

    actual, _ = fused_maxmin(packed, pick_size, first_picks=first_picks, seed=seed, output=RDKIT)

    assert actual == expected


@pytest.mark.parametrize("threshold", [0.5, 0.75, 0.875])
def test_fused_maxmin_threshold_matches_rdkit_on_chembl(chembl_fingerprints, threshold):
    bit_vectors, packed = chembl_fingerprints
    expected, expected_last = rdSimDivPickers.MaxMinPicker().LazyBitVectorPickWithThreshold(
        bit_vectors, len(bit_vectors), len(bit_vectors), threshold, seed=11
    )

    actual, actual_last = fused_maxmin(packed, len(bit_vectors), seed=11, threshold=threshold, output=RDKIT)

    assert actual == tuple(expected)
    assert actual_last == pytest.approx(expected_last, rel=1e-6)


@pytest.mark.parametrize("metric", ["tanimoto", "cosine"])
@pytest.mark.parametrize("cutoff", [0.3, 0.55])
def test_fused_algorithms_match_matrix_forms_on_chembl(chembl_fingerprints, chembl_distances, metric, cutoff):
    _, packed = chembl_fingerprints
    distances = chembl_distances[metric]

    assert fused_leader(packed, cutoff, metric=metric, output=RDKIT) == leader(distances, cutoff, output=RDKIT)
    assert fused_maxmin(packed, 150, metric=metric, seed=3, output=RDKIT) == maxmin(
        distances, 150, seed=3, output=RDKIT
    )
    for assignment in ("first", "nearest"):
        assert fused_dise(packed, cutoff, metric=metric, assignment=assignment, output=RDKIT) == dise(
            distances, cutoff, assignment=assignment, output=RDKIT
        )


def test_float32_and_float64_matrices_agree_on_chembl(chembl_distances):
    distances = chembl_distances["tanimoto"]
    widened = distances.astype(np.float64)

    assert leader(widened, 0.3, output=RDKIT) == leader(distances, 0.3, output=RDKIT)
    assert maxmin(widened, 100, seed=2, output=RDKIT) == maxmin(distances, 100, seed=2, output=RDKIT)
    assert dise(widened, 0.3, output=RDKIT) == dise(distances, 0.3, output=RDKIT)


# A matrix from crossTanimotoSimilarity must hold the same float32 distances as the fused forms compute, so MaxMin
# resolves ties between equal similarity fractions as RDKit does.
@pytest.mark.parametrize("pick_size, seed", [(200, 42), (999, 5)])
def test_matrix_maxmin_on_cross_similarity_matches_rdkit_on_chembl(chembl_fingerprints, pick_size, seed):
    bit_vectors, packed = chembl_fingerprints
    distances = 1.0 - crossTanimotoSimilarity(packed).torch()
    expected = tuple(
        rdSimDivPickers.MaxMinPicker().LazyBitVectorPick(bit_vectors, len(bit_vectors), pick_size, seed=seed)
    )

    assert maxmin(distances, pick_size, seed=seed, output=RDKIT)[0] == expected


@pytest.mark.parametrize("assignment", ["first", "nearest"])
@pytest.mark.parametrize("cutoff", [0.3, 0.6])
def test_dise_matches_reference_on_chembl(chembl_distances, assignment, cutoff):
    distances = chembl_distances["tanimoto"]

    assert dise(distances, cutoff, assignment=assignment, output=RDKIT) == _reference_dise(
        distances, cutoff, assignment
    )


def test_dise_device_result_is_consistent_on_chembl(chembl_fingerprints):
    _, packed = chembl_fingerprints

    result = fused_dise(packed, 0.4)
    cluster_ids = result.cluster_ids.numpy()
    centroids = result.centroids.numpy()
    sizes = result.cluster_sizes.numpy()

    assert cluster_ids.shape == (len(packed),)
    assert np.array_equal(np.bincount(cluster_ids, minlength=len(centroids)), sizes)
    assert np.array_equal(cluster_ids[centroids], np.arange(len(centroids)))
    assert np.all(np.diff(sizes) <= 0)


@pytest.mark.parametrize("num_items", [31, 32, 33, 255, 256, 257])
@pytest.mark.parametrize("cutoff", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("selection", ["unlimited", "one", "forced", "forced_over_limit", "forced_window"])
def test_leader_directed_window_and_block_boundaries(num_items, cutoff, selection):
    generator = np.random.default_rng(num_items)
    distances = generator.integers(1, 5, size=(num_items, num_items)).astype(np.float32) / 4
    np.fill_diagonal(distances, 1.0)
    first_picks = (num_items - 1, 1, num_items // 2) if selection.startswith("forced") else ()
    if selection == "forced_window":
        first_picks = tuple(int(index) for index in generator.permutation(num_items)[:40])
    pick_size = {"unlimited": 0, "one": 1, "forced": 17, "forced_over_limit": 2, "forced_window": 0}[selection]
    expected = _reference_leader(distances, cutoff, pick_size, first_picks)

    for _ in range(2):
        assert leader(distances, cutoff, pick_size=pick_size, first_picks=first_picks, output=RDKIT) == expected


@pytest.mark.parametrize(
    "distance, cutoff",
    [(0.50000001, 0.5), (0.5, np.nextafter(0.5, 0.0)), (0.50000006, 0.5)],
)
def test_matrix_distances_and_cutoffs_use_float32_comparisons(distance, cutoff):
    distances = np.asarray([[0.0, distance], [distance, 0.0]], dtype=np.float64)

    expected = _reference_leader(distances, cutoff)
    assert leader(distances, cutoff, output=RDKIT) == expected
    for assignment in ("first", "nearest"):
        assert dise(distances, cutoff, assignment=assignment, output=RDKIT) == _reference_dise(
            distances.astype(np.float32), cutoff, assignment
        )
    expected_maxmin = ((0,), -1.0) if len(expected) == 1 else ((0, 1), float(np.float32(distance)))
    assert maxmin(distances, 2, first_picks=(0,), threshold=cutoff, output=RDKIT) == expected_maxmin


def test_matrix_values_that_overflow_float32_convert_to_infinity():
    distances = np.asarray([[0.0, 2e40], [2e40, 0.0]], dtype=np.float64)
    largest_float32 = float(np.finfo(np.float32).max)

    assert leader(distances, largest_float32, output=RDKIT) == (0, 1)
    assert dise(distances, largest_float32, output=RDKIT) == ((0,), (1,))
    assert maxmin(distances, 2, first_picks=(0,), threshold=largest_float32, output=RDKIT) == ((0, 1), float("inf"))


def test_maxmin_signed_zero_distances_tie_to_the_lower_index():
    distances = np.zeros((3, 3))
    distances[0, 1] = -0.0

    assert maxmin(distances, 2, first_picks=(0,), output=RDKIT)[0] == (0, 1)


def test_matrix_cutoff_and_threshold_accept_largest_finite_float32():
    largest_float32 = float(np.finfo(np.float32).max)
    distances = np.asarray([[0.0, largest_float32], [largest_float32, 0.0]])

    assert leader(distances, largest_float32, output=RDKIT) == (0,)
    assert dise(distances, largest_float32, output=RDKIT) == ((0, 1),)
    assert maxmin(distances, 2, first_picks=(0,), threshold=largest_float32, output=RDKIT) == ((0,), -1.0)


_STREAM_CASES = [
    (leader, "matrix"),
    (maxmin, "matrix"),
    (dise, "matrix"),
    (fused_leader, "tanimoto"),
    (fused_leader, "cosine"),
    (fused_maxmin, "tanimoto"),
    (fused_maxmin, "cosine"),
    (fused_dise, "tanimoto"),
    (fused_dise, "cosine"),
    (fused_leader, "aap"),
    (fused_maxmin, "aap"),
    (fused_dise, "aap"),
]


def _stream_case_input(metric, device):
    fingerprints = np.asarray([[3], [2], [12], [7]], dtype=np.int32)
    if metric == "matrix":
        inputs = torch.from_numpy(_fingerprint_distance_matrix(fingerprints, "tanimoto")).to(device)
        options = {}
    elif metric == "aap":
        inputs = [Chem.MolFromSmiles(smiles) for smiles in ("CC", "CCC", "CO", "CCO")]
        options = {"metric": metric}
    else:
        inputs = torch.from_numpy(fingerprints).to(device)
        options = {"metric": metric}
    return inputs, options


@pytest.mark.parametrize("function, metric", _STREAM_CASES)
@pytest.mark.parametrize("output", [RDKIT, OutputMode.DEVICE])
def test_native_call_and_output_wrapping_use_selected_stream(monkeypatch, function, metric, output):
    stream = torch.cuda.Stream()
    inputs, options = _stream_case_input(metric, stream.device)
    argument = 3 if function in (maxmin, fused_maxmin) else 0.5
    if function in (maxmin, fused_maxmin):
        options["first_picks"] = (0,)
    expected = function(inputs, argument, output=RDKIT, **options)
    torch.cuda.current_stream().synchronize()
    native_name = function.__name__.replace("fused_", "aap_") if metric == "aap" else function.__name__
    native = getattr(_clustering, native_name)
    module = clustering if function in (dise, fused_dise) else pickers
    resolve_name = "_resolve_cluster_output" if module is clustering else "_resolve_selection_output"
    resolve = getattr(module, resolve_name)
    calls = []

    def check_native(*args):
        assert torch.cuda.current_stream() == stream
        calls.append("native")
        return native(*args)

    def check_output(*args, **kwargs):
        assert torch.cuda.current_stream() == stream
        calls.append("output")
        return resolve(*args, **kwargs)

    monkeypatch.setattr(_clustering, native_name, check_native)
    monkeypatch.setattr(module, resolve_name, check_output)
    original_stream = torch.cuda.current_stream()
    result = function(inputs, argument, stream=stream, output=output, **options)
    stream.synchronize()

    assert calls == ["native", "output"]
    assert torch.cuda.current_stream() == original_stream
    if output is OutputMode.DEVICE:
        if module is clustering:
            arrays = (result.cluster_ids, result.centroids, result.cluster_sizes)
            assert all(array.device == stream.device for array in arrays)
            result = clustering._cluster_arrays_to_rdkit(result.cluster_ids.numpy(), result.centroids.numpy())
        elif function in (maxmin, fused_maxmin):
            assert result[0].device == stream.device
            result = (tuple(result[0].numpy()), result[1])
        else:
            assert result.device == stream.device
            result = tuple(result.numpy())
    assert result == expected


@pytest.mark.parametrize("function, metric", _STREAM_CASES)
@pytest.mark.parametrize("stream_kind", ["explicit", "default"])
def test_selected_device_can_differ_from_current_device(function, metric, stream_kind):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    with torch.cuda.device(1):
        inputs, options = _stream_case_input(metric, "cuda:1")
        argument = 3 if function in (maxmin, fused_maxmin) else 0.5
        if function in (maxmin, fused_maxmin):
            options["first_picks"] = (0,)
        expected = function(inputs, argument, output=RDKIT, **options)
        stream = torch.cuda.Stream() if stream_kind == "explicit" else torch.cuda.default_stream()
        torch.cuda.synchronize()

    with torch.cuda.device(0):
        stream_argument = stream if stream_kind == "explicit" or metric == "aap" else None
        result = function(inputs, argument, stream=stream_argument, **options)
        assert torch.cuda.current_device() == 0
        stream.synchronize()
        if function in (dise, fused_dise):
            assert result.cluster_ids.device.index == 1
            assert result.centroids.device.index == 1
            assert result.cluster_sizes.device.index == 1
            actual = clustering._cluster_arrays_to_rdkit(result.cluster_ids.numpy(), result.centroids.numpy())
        elif function in (maxmin, fused_maxmin):
            assert result[0].device.index == 1
            actual = (tuple(result[0].numpy()), result[1])
        else:
            assert result.device.index == 1
            actual = tuple(result.numpy())
        assert actual == expected


@pytest.mark.parametrize("num_words", [3, 260])
@pytest.mark.parametrize("metric", ["tanimoto", "cosine"])
def test_fused_forms_match_matrix_forms_for_unusual_widths(float32_distances, num_words, metric):
    # 3 words cannot use vector loads; 260 words exceed the shared-memory source tile.
    rng = np.random.default_rng(num_words)
    bits = rng.random((300, num_words * 32)) < 0.03
    bits[:40] = bits[40:80] | (rng.random((40, num_words * 32)) < 0.01)
    packed = np.packbits(bits, axis=1, bitorder="little").view(np.uint32)
    distances = float32_distances(packed, metric)

    assert fused_leader(packed, 0.9, metric=metric, output=RDKIT) == leader(distances, 0.9, output=RDKIT)
    assert fused_maxmin(packed, 40, metric=metric, seed=4, output=RDKIT) == maxmin(distances, 40, seed=4, output=RDKIT)
    assert fused_dise(packed, 0.9, metric=metric, output=RDKIT) == dise(distances, 0.9, output=RDKIT)


def test_fused_forms_handle_buffers_not_aligned_for_vector_loads(chembl_fingerprints):
    _, packed = chembl_fingerprints
    storage = torch.zeros(packed.size + 1, dtype=torch.int32, device="cuda")
    shifted = storage[1:].view(packed.shape)
    shifted.copy_(torch.from_numpy(packed.view(np.int32)))
    assert shifted.data_ptr() % 16 != 0

    assert fused_leader(shifted, 0.4, output=RDKIT) == fused_leader(packed, 0.4, output=RDKIT)
    assert fused_dise(shifted, 0.4, output=RDKIT) == fused_dise(packed, 0.4, output=RDKIT)


@pytest.mark.parametrize("form", ["int32", "torch_cpu", "torch_cuda", "async", "non_contiguous"])
def test_fused_inputs_accept_array_forms(chembl_fingerprints, form):
    _, packed = chembl_fingerprints
    expected = fused_leader(packed, 0.4, output=RDKIT)
    if form == "int32":
        value = packed.view(np.int32)
    elif form == "torch_cpu":
        value = torch.from_numpy(packed.view(np.int32))
    elif form == "torch_cuda":
        value = torch.from_numpy(packed.view(np.int32)).cuda()
    elif form == "async":
        value = AsyncGpuResult(torch.from_numpy(packed.view(np.int32)).cuda())
    else:
        value = torch.from_numpy(np.asfortranarray(packed.view(np.int32))).cuda()
        value = value.t().contiguous().t()
        assert not value.is_contiguous()

    assert fused_leader(value, 0.4, output=RDKIT) == expected


@pytest.mark.parametrize("function", [fused_leader, fused_maxmin, fused_dise])
@pytest.mark.parametrize("form", ["numpy", "torch_cpu", "non_contiguous_cuda"])
@pytest.mark.parametrize("metric", ["tanimoto", "cosine"])
def test_fused_prepared_input_owns_storage_through_native_call(monkeypatch, function, form, metric):
    fingerprints = np.asarray([[3, 0], [2, 0], [12, 0], [7, 0], [0, 16], [3, 0]], dtype=np.int32)
    distances = _fingerprint_distance_matrix(fingerprints, metric).astype(np.float32)
    argument = 4 if function is fused_maxmin else 0.5
    options = {"first_picks": (0,)} if function is fused_maxmin else {}
    if function is fused_maxmin:
        expected = maxmin(distances, argument, output=RDKIT, **options)
    elif function is fused_leader:
        expected = _reference_leader(distances, 0.5)
    else:
        expected = _reference_dise(distances, 0.5, "nearest")
    prepared_reference = None
    prepare = _distance_inputs._prepare_packed_fingerprints

    def capture_prepared(*args, **kwargs):
        nonlocal prepared_reference
        tensors, active_stream = prepare(*args, **kwargs)
        prepared_reference = weakref.ref(tensors[0])
        return tensors, active_stream

    native = getattr(_clustering, function.__name__)

    def allocate_before_native(array_interface, *args):
        # A freed preparation buffer can be recycled by this same-size allocation.
        overwrite = torch.empty(fingerprints.shape, dtype=torch.int32, device="cuda").zero_()
        assert prepared_reference() is not None
        assert overwrite.data_ptr() != array_interface["data"][0]
        return native(array_interface, *args)

    monkeypatch.setattr(_distance_inputs, "_prepare_packed_fingerprints", capture_prepared)
    monkeypatch.setattr(_clustering, function.__name__, allocate_before_native)
    if form == "numpy":
        inputs = fingerprints.copy()
    elif form == "torch_cpu":
        inputs = torch.from_numpy(fingerprints.copy())
    else:
        storage = torch.from_numpy(fingerprints.T.copy()).cuda()
        inputs = storage.T
        assert not inputs.is_contiguous()

    actual = function(inputs, argument, metric=metric, output=RDKIT, **options)
    if function is fused_maxmin:
        assert actual[0] == expected[0]
        assert actual[1] == pytest.approx(expected[1], abs=1e-6)
    else:
        assert actual == expected


@pytest.mark.parametrize("metric", ["matrix", "tanimoto", "cosine", "aap"])
@pytest.mark.parametrize("assignment", ["first", "nearest"])
@pytest.mark.parametrize("output", [RDKIT, OutputMode.DEVICE])
def test_dise_singleton_has_initialized_cluster_arrays(metric, assignment, output):
    inputs, options = _stream_case_input(metric, "cuda")
    inputs = inputs[:1, :1] if metric == "matrix" else inputs[:1]
    function = dise if metric == "matrix" else fused_dise

    result = function(inputs, 0.5, assignment=assignment, output=output, **options)

    if output is OutputMode.DEVICE:
        assert result.cluster_ids.numpy().tolist() == [0]
        assert result.centroids.numpy().tolist() == [0]
        assert result.cluster_sizes.numpy().tolist() == [1]
    else:
        assert result == ((0,),)


def test_explicit_stream_matches_default_stream_on_chembl(chembl_fingerprints):
    _, packed = chembl_fingerprints
    stream = torch.cuda.Stream()

    selection, _ = fused_maxmin(packed, 64, seed=1, stream=stream)
    clusters = fused_dise(packed, 0.4, stream=stream)
    stream.synchronize()

    assert selection.numpy().tolist() == list(fused_maxmin(packed, 64, seed=1, output=RDKIT)[0])
    assert clusters.cluster_ids.numpy().tolist() == fused_dise(packed, 0.4).cluster_ids.numpy().tolist()


# ---------------------------------------------------------------------------
# Integration: AAP as a fused metric, against RDKit's lazy pickers
# ---------------------------------------------------------------------------


def _aap_distance(molecules):
    @cache
    def distance(selected, candidate):
        return 1.0 - aap_similarity(molecules[selected], molecules[candidate])

    return distance


def test_aap_leader_and_first_assignment_match_rdkit_lazy_picker(aap_molecules):
    # AAP similarities across diverse ChEMBL compounds are low; this cutoff still yields ~30 multi-member clusters.
    cutoff = 0.95
    distance = _aap_distance(aap_molecules)
    # RDKit's LeaderPicker evaluates func(leader, candidate).
    expected_leaders = tuple(rdSimDivPickers.LeaderPicker().LazyPick(distance, len(aap_molecules), cutoff))

    assert fused_leader(aap_molecules, cutoff, metric="aap", output=RDKIT) == expected_leaders

    labels = [
        next(k for k, leader_index in enumerate(expected_leaders) if distance(leader_index, item) <= cutoff)
        if item not in expected_leaders
        else expected_leaders.index(item)
        for item in range(len(aap_molecules))
    ]
    result = fused_dise(aap_molecules, cutoff, metric="aap", assignment="first")
    centroids = result.centroids.numpy()
    assert [int(centroids[label]) for label in result.cluster_ids.numpy()] == [expected_leaders[k] for k in labels]


def test_aap_maxmin_matches_rdkit_lazy_picker(aap_molecules):
    distance = _aap_distance(aap_molecules)
    # RDKit's MaxMinPicker evaluates func(candidate, pick); nvMolKit measures from the pick.
    expected = tuple(
        rdSimDivPickers.MaxMinPicker().LazyPick(
            lambda candidate, pick: distance(pick, candidate), len(aap_molecules), 8, firstPicks=[3]
        )
    )

    actual, last_distance = fused_maxmin(aap_molecules, 8, metric=AAPMetric(), first_picks=(3,), output=RDKIT)

    assert actual == expected
    assert last_distance == pytest.approx(min(distance(pick, actual[-1]) for pick in actual[:-1]))


def test_aap_nearest_assignment_picks_the_nearest_centroid(aap_molecules):
    molecules = aap_molecules[:40]
    cutoff = 0.7
    distance = _aap_distance(molecules)

    result = fused_dise(molecules, cutoff, metric="aap", assignment="nearest")
    centroids = result.centroids.numpy().tolist()
    for item, label in enumerate(result.cluster_ids.numpy().tolist()):
        if item in centroids:
            assert centroids[label] == item
            continue
        nearest = min(distance(centroid, item) for centroid in centroids)
        assert distance(centroids[label], item) == nearest


def test_aap_is_not_yet_supported_by_fused_butina():
    with pytest.raises(NotImplementedError, match="AAPMetric"):
        fused_butina(np.zeros((1, 1), dtype=np.uint32), 0.5, metric="aap")


# ---------------------------------------------------------------------------
# Semantics and pathological inputs
# ---------------------------------------------------------------------------


def test_matrix_rows_are_distances_from_the_selected_item():
    distances = np.asarray(
        [
            [0.0, 0.1, 0.8],
            [0.9, 0.0, 0.1],
            [0.2, 0.9, 0.0],
        ],
        dtype=np.float64,
    )

    assert leader(distances, 0.2, output=RDKIT) == (0, 2)
    # From pick 0, item 2 is farthest by row 0 even though column 0 says otherwise.
    picks, last_distance = maxmin(distances, 2, first_picks=(0,), output=RDKIT)
    assert picks == (0, 2)
    assert last_distance == pytest.approx(0.8)
    assert dise(distances, 0.2, assignment="first", output=RDKIT) == ((0, 1), (2,))


def test_forced_picks_are_kept_even_when_they_exclude_each_other():
    points = np.asarray([0.0, 0.05, 0.25, 0.6, 0.65, 1.0])
    distances = np.abs(points[:, None] - points[None, :])
    distance = lambda left, right: distances[left, right]  # noqa: E731

    expected = tuple(rdSimDivPickers.LeaderPicker().LazyPick(distance, 6, 0.2, pickSize=1, firstPicks=[4, 3, 1]))
    assert leader(distances, 0.2, pick_size=1, first_picks=(4, 3, 1), output=RDKIT) == expected == (4, 3, 1)
    assert maxmin(distances, 2, first_picks=(4, 3, 1), output=RDKIT) == ((4, 3, 1), -1.0)


def test_identical_inputs_collapse_to_one_leader_and_lowest_index_maxmin():
    fingerprints = np.tile(np.asarray([[0b1011, 0b0110]], dtype=np.uint32), (300, 1))

    assert leader(np.zeros((300, 300)), 0.0, output=RDKIT) == (0,)
    assert fused_leader(fingerprints, 0.0, output=RDKIT) == (0,)
    assert fused_dise(fingerprints, 0.0, output=RDKIT) == (tuple(range(300)),)
    assert fused_maxmin(fingerprints, 3, first_picks=(299,), output=RDKIT) == ((299, 0, 1), 0.0)


@pytest.mark.parametrize("metric", [TanimotoMetric(), CosineMetric()])
@pytest.mark.parametrize("cutoff", [0.0, 0.5, 1.0])
def test_empty_fingerprints_follow_each_metric_convention(metric, cutoff):
    # Tanimoto defines two empty fingerprints as identical; cosine defines them as unrelated.
    metric_name = "tanimoto" if isinstance(metric, TanimotoMetric) else "cosine"
    fingerprints = np.asarray([[0], [0b0011], [0], [0b0010], [0b1100]], dtype=np.uint32)
    distances = _fingerprint_distance_matrix(fingerprints, metric_name)

    assert fused_leader(fingerprints, cutoff, metric=metric, output=RDKIT) == leader(distances, cutoff, output=RDKIT)
    assert fused_maxmin(fingerprints, 5, metric=metric, first_picks=(0,), output=RDKIT) == maxmin(
        distances, 5, first_picks=(0,), output=RDKIT
    )
    for assignment in ("first", "nearest"):
        assert fused_dise(fingerprints, cutoff, metric=metric, assignment=assignment, output=RDKIT) == dise(
            distances, cutoff, assignment=assignment, output=RDKIT
        )


def test_fused_tanimoto_distance_matches_double_precision_matrix_at_cutoff_boundary():
    # Intersection 1 and union 3: 2/3 rounds up in float32, but 1 - float32(1/3) rounds down.
    fingerprints = np.asarray([[0b001], [0b111]], dtype=np.uint32)
    distances = np.asarray([[0.0, 2 / 3], [2 / 3, 0.0]])
    cutoff = float(np.nextafter(np.float32(2 / 3), np.float32(0)))

    assert fused_leader(fingerprints, cutoff, output=RDKIT) == leader(distances, cutoff, output=RDKIT) == (0, 1)


def test_leader_removes_itself_without_a_zero_diagonal():
    distances = np.full((3, 3), 0.8)

    assert leader(distances, 0.1, output=RDKIT) == (0, 1, 2)


def test_maxmin_threshold_can_stop_before_any_addition():
    distances = np.ones((3, 3))
    np.fill_diagonal(distances, 0.0)

    assert maxmin(distances, 3, first_picks=(0,), threshold=1.0, output=RDKIT) == ((0,), -1.0)
    assert maxmin(distances, 2, first_picks=(0,), output=RDKIT) == ((0, 1), 1.0)


def test_empty_and_singleton_inputs():
    empty_matrix = np.empty((0, 0))
    empty_fingerprints = np.empty((0, 4), dtype=np.uint32)
    singleton = np.zeros((1, 1))

    assert leader(empty_matrix, 0.0, output=RDKIT) == ()
    assert dise(empty_matrix, 0.0, output=RDKIT) == ()
    assert fused_leader(empty_fingerprints, 0.5, output=RDKIT) == ()
    assert fused_dise(empty_fingerprints, 0.5, output=RDKIT) == ()
    assert fused_leader([], 0.5, metric="aap", output=RDKIT) == ()
    assert fused_dise([], 0.5, metric="aap", output=RDKIT) == ()
    assert leader(singleton, 0.0, output=RDKIT) == (0,)
    assert maxmin(singleton, 1, seed=7, output=RDKIT) == ((0,), -1.0)
    assert dise(singleton, 0.0, output=RDKIT) == ((0,),)
    with pytest.raises(ValueError, match="pick_size"):
        fused_maxmin(empty_fingerprints, 1)


def test_numpy_and_torch_integer_arguments_are_accepted():
    points = np.asarray([0.0, 0.05, 0.25, 0.6, 0.65, 1.0])
    distances = np.abs(points[:, None] - points[None, :])

    assert leader(distances, 0.2, pick_size=np.int64(2), first_picks=np.asarray([3]), output=RDKIT) == (3, 0)
    assert maxmin(distances, np.int32(3), first_picks=torch.tensor([5]), seed=np.int64(1), output=RDKIT)[0] == (
        5,
        0,
        3,
    )


@pytest.mark.parametrize(
    "function, args",
    [
        (fused_leader, (0.5,)),
        (fused_maxmin, (3,)),
        (fused_dise, (0.5,)),
        (fused_butina, (0.5,)),
    ],
)
def test_metric_names_and_instances_are_equivalent(function, args):
    fingerprints = np.asarray([[0b0011], [0b0010], [0b1100], [0b0111]], dtype=np.uint32)
    extra = {"first_picks": (0,)} if function is fused_maxmin else {}

    for name, instance in (("tanimoto", TanimotoMetric()), ("cosine", CosineMetric())):
        by_name = function(fingerprints, *args, metric=name, output=RDKIT, **extra)
        assert function(fingerprints, *args, metric=instance, output=RDKIT, **extra) == by_name


def test_device_outputs_have_documented_types():
    points = np.asarray([0.0, 0.05, 0.25, 0.6, 0.65, 1.0])
    distances = np.abs(points[:, None] - points[None, :])

    selection, last_distance = maxmin(distances, 3, first_picks=(0,))
    leaders = leader(distances, 0.2)
    clusters = dise(distances, 0.2)

    assert isinstance(selection, AsyncGpuResult)
    assert selection.torch().dtype == torch.int32
    assert selection.torch().shape == (3,)
    assert last_distance == pytest.approx(0.4)
    assert isinstance(leaders, AsyncGpuResult)
    assert isinstance(clusters, ClusterDeviceResult)
    assert clusters.cluster_ids.torch().dtype == torch.int32
    assert clusters.centroids.torch().dtype == torch.int32
    assert clusters.cluster_sizes.torch().dtype == torch.int64


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _small_matrix():
    return np.zeros((6, 6))


@pytest.mark.parametrize("function", [leader, dise])
@pytest.mark.parametrize("cutoff", [-0.1, np.nan, np.inf, np.nextafter(float(np.finfo(np.float32).max), np.inf)])
def test_matrix_sphere_exclusion_rejects_invalid_cutoffs(function, cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        function(_small_matrix(), cutoff)


@pytest.mark.parametrize("function", [fused_leader, fused_dise])
@pytest.mark.parametrize("metric", ["tanimoto", "aap"])
@pytest.mark.parametrize("cutoff", [-0.1, 1.1, np.nan])
def test_fused_sphere_exclusion_rejects_invalid_cutoffs(function, metric, cutoff):
    x = [Chem.MolFromSmiles("CC")] if metric == "aap" else np.asarray([[1]], dtype=np.uint32)
    with pytest.raises(ValueError, match="cutoff"):
        function(x, cutoff, metric=metric)


@pytest.mark.parametrize(
    "matrix, message",
    [
        (np.zeros(3), "square 2D"),
        (np.zeros((2, 3)), "square 2D"),
        (np.zeros((2, 2), dtype=np.float16), "float32 or float64"),
        (np.zeros((2, 2), dtype=np.int32), "float32 or float64"),
    ],
)
def test_matrix_shape_and_dtype_are_validated(matrix, message):
    with pytest.raises(ValueError, match=message):
        leader(matrix, 0.2)


@pytest.mark.parametrize("first_picks", [(0, 0), (-1,), (6,)])
def test_first_picks_are_validated(first_picks):
    with pytest.raises(ValueError, match="first_picks"):
        leader(_small_matrix(), 0.2, first_picks=first_picks)
    with pytest.raises(ValueError, match="first_picks"):
        maxmin(_small_matrix(), 2, first_picks=first_picks)


def test_first_picks_must_be_integers():
    with pytest.raises(TypeError, match="first_picks"):
        leader(_small_matrix(), 0.2, first_picks=(1.5,))


@pytest.mark.parametrize("function, pick_sizes", [(leader, (-1, 7)), (maxmin, (0, -1, 7))])
def test_pick_size_is_validated(function, pick_sizes):
    for pick_size in pick_sizes:
        with pytest.raises(ValueError, match="pick_size"):
            if function is leader:
                leader(_small_matrix(), 0.2, pick_size=pick_size)
            else:
                maxmin(_small_matrix(), pick_size)


@pytest.mark.parametrize(
    "threshold", [-0.1, -1.0, np.nan, np.inf, np.nextafter(float(np.finfo(np.float32).max), np.inf)]
)
def test_maxmin_threshold_is_validated(threshold):
    with pytest.raises(ValueError, match="threshold"):
        maxmin(_small_matrix(), 2, threshold=threshold)
    with pytest.raises(ValueError, match="threshold"):
        fused_maxmin(np.asarray([[1], [2]], dtype=np.uint32), 2, threshold=threshold)


def test_fused_maxmin_threshold_above_one_is_rejected():
    with pytest.raises(ValueError, match="threshold"):
        fused_maxmin(np.asarray([[1], [2]], dtype=np.uint32), 2, threshold=1.1)


def test_fingerprint_input_is_validated():
    with pytest.raises(ValueError, match="2D"):
        fused_leader(np.asarray([1, 2], dtype=np.uint32), 0.2)
    with pytest.raises(ValueError, match="dtype"):
        fused_leader(np.asarray([[1.0]]), 0.2)
    with pytest.raises(ValueError, match="at least one fingerprint word"):
        fused_leader(np.empty((2, 0), dtype=np.uint32), 0.2)


def test_metric_output_and_assignment_are_validated():
    with pytest.raises(ValueError, match="metric must be"):
        fused_leader(np.asarray([[1]], dtype=np.uint32), 0.2, metric="euclidean")
    with pytest.raises(TypeError, match="OutputMode"):
        leader(_small_matrix(), 0.2, output="rdkit")
    with pytest.raises(ValueError, match="assignment"):
        dise(_small_matrix(), 0.2, assignment="unknown")
