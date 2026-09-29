# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
import torch
from rdkit.SimDivFilters import rdSimDivPickers

from nvmolkit import _clustering, pickers
from nvmolkit.clustering import (
    OutputMode,
    fused_butina,
)
from nvmolkit.pickers import leader
from nvmolkit.similarity import CosineMetric, TanimotoMetric
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


# ---------------------------------------------------------------------------
# Integration: ChEMBL fingerprints against RDKit
# ---------------------------------------------------------------------------


# Distinct Tanimoto values keep their order in float32, so results equal RDKit's double-precision pickers except
# where a distance equals a cutoff that float32 cannot represent exactly. RDKit parity tests use dyadic cutoffs.
@pytest.mark.parametrize("cutoff", [0.0, 0.25, 0.5, 0.625, 1.0])
def test_leader_matches_rdkit_on_chembl(chembl_fingerprints, chembl_distances, cutoff):
    bit_vectors, _ = chembl_fingerprints
    expected = tuple(rdSimDivPickers.LeaderPicker().LazyBitVectorPick(bit_vectors, len(bit_vectors), cutoff))

    assert leader(chembl_distances["tanimoto"], cutoff, output=RDKIT) == expected


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


def test_float32_and_float64_matrices_agree_on_chembl(chembl_distances):
    distances = chembl_distances["tanimoto"]
    widened = distances.astype(np.float64)

    assert leader(widened, 0.3, output=RDKIT) == leader(distances, 0.3, output=RDKIT)


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

    assert leader(distances, cutoff, output=RDKIT) == _reference_leader(distances, cutoff)


def test_matrix_values_that_overflow_float32_convert_to_infinity():
    distances = np.asarray([[0.0, 2e40], [2e40, 0.0]], dtype=np.float64)
    largest_float32 = float(np.finfo(np.float32).max)

    assert leader(distances, largest_float32, output=RDKIT) == (0, 1)


def test_matrix_cutoff_accepts_largest_finite_float32():
    largest_float32 = float(np.finfo(np.float32).max)
    distances = np.asarray([[0.0, largest_float32], [largest_float32, 0.0]])

    assert leader(distances, largest_float32, output=RDKIT) == (0,)


@pytest.mark.parametrize("output", [RDKIT, OutputMode.DEVICE])
def test_native_call_and_output_wrapping_use_selected_stream(monkeypatch, output):
    stream = torch.cuda.Stream()
    points = np.asarray([0.0, 0.05, 0.25, 0.6, 0.65, 1.0])
    distances = np.abs(points[:, None] - points[None, :])
    inputs = torch.from_numpy(distances).to(stream.device)
    expected = _reference_leader(distances, 0.2)
    torch.cuda.current_stream().synchronize()
    native = _clustering.leader
    resolve = pickers._resolve_selection_output
    calls = []

    def check_native(*args):
        assert torch.cuda.current_stream() == stream
        calls.append("native")
        return native(*args)

    def check_output(*args, **kwargs):
        assert torch.cuda.current_stream() == stream
        calls.append("output")
        return resolve(*args, **kwargs)

    monkeypatch.setattr(_clustering, "leader", check_native)
    monkeypatch.setattr(pickers, "_resolve_selection_output", check_output)
    original_stream = torch.cuda.current_stream()
    result = leader(inputs, 0.2, stream=stream, output=output)
    stream.synchronize()

    assert calls == ["native", "output"]
    assert torch.cuda.current_stream() == original_stream
    if output is OutputMode.DEVICE:
        assert result.device == stream.device
        result = tuple(result.numpy())
    assert result == expected


@pytest.mark.parametrize("stream_kind", ["explicit", "default"])
def test_selected_device_can_differ_from_current_device(stream_kind):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    points = np.asarray([0.0, 0.05, 0.25, 0.6, 0.65, 1.0])
    distances = np.abs(points[:, None] - points[None, :])
    expected = _reference_leader(distances, 0.2)
    with torch.cuda.device(1):
        inputs = torch.from_numpy(distances).to("cuda:1")
        stream = torch.cuda.Stream() if stream_kind == "explicit" else torch.cuda.default_stream()
        torch.cuda.synchronize()

    with torch.cuda.device(0):
        stream_argument = stream if stream_kind == "explicit" else None
        result = leader(inputs, 0.2, stream=stream_argument)
        assert torch.cuda.current_device() == 0
        stream.synchronize()
        assert result.device.index == 1
        assert tuple(result.numpy()) == expected


def test_explicit_stream_matches_default_stream_on_chembl(chembl_distances):
    distances = chembl_distances["tanimoto"]
    stream = torch.cuda.Stream()

    selection = leader(distances, 0.4, stream=stream)
    stream.synchronize()

    assert selection.numpy().tolist() == list(leader(distances, 0.4, output=RDKIT))


# ---------------------------------------------------------------------------
# AAP metric support
# ---------------------------------------------------------------------------


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


def test_forced_picks_are_kept_even_when_they_exclude_each_other():
    points = np.asarray([0.0, 0.05, 0.25, 0.6, 0.65, 1.0])
    distances = np.abs(points[:, None] - points[None, :])
    distance = lambda left, right: distances[left, right]  # noqa: E731

    expected = tuple(rdSimDivPickers.LeaderPicker().LazyPick(distance, 6, 0.2, pickSize=1, firstPicks=[4, 3, 1]))
    assert leader(distances, 0.2, pick_size=1, first_picks=(4, 3, 1), output=RDKIT) == expected == (4, 3, 1)


def test_identical_inputs_collapse_to_one_leader():
    assert leader(np.zeros((300, 300)), 0.0, output=RDKIT) == (0,)


def test_leader_removes_itself_without_a_zero_diagonal():
    distances = np.full((3, 3), 0.8)

    assert leader(distances, 0.1, output=RDKIT) == (0, 1, 2)


def test_empty_and_singleton_inputs():
    empty_matrix = np.empty((0, 0))
    singleton = np.zeros((1, 1))

    assert leader(empty_matrix, 0.0, output=RDKIT) == ()
    assert leader(singleton, 0.0, output=RDKIT) == (0,)


def test_numpy_and_torch_integer_arguments_are_accepted():
    points = np.asarray([0.0, 0.05, 0.25, 0.6, 0.65, 1.0])
    distances = np.abs(points[:, None] - points[None, :])

    assert leader(distances, 0.2, pick_size=np.int64(2), first_picks=np.asarray([3]), output=RDKIT) == (3, 0)


@pytest.mark.parametrize(
    "function, args",
    [
        (fused_butina, (0.5,)),
    ],
)
def test_metric_names_and_instances_are_equivalent(function, args):
    fingerprints = np.asarray([[0b0011], [0b0010], [0b1100], [0b0111]], dtype=np.uint32)

    for name, instance in (("tanimoto", TanimotoMetric()), ("cosine", CosineMetric())):
        by_name = function(fingerprints, *args, metric=name, output=RDKIT)
        assert function(fingerprints, *args, metric=instance, output=RDKIT) == by_name


def test_device_outputs_have_documented_types():
    points = np.asarray([0.0, 0.05, 0.25, 0.6, 0.65, 1.0])
    distances = np.abs(points[:, None] - points[None, :])

    selection = leader(distances, 0.2)

    assert isinstance(selection, AsyncGpuResult)
    assert selection.torch().dtype == torch.int32
    assert selection.torch().shape == (4,)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _small_matrix():
    return np.zeros((6, 6))


@pytest.mark.parametrize("function", [leader])
@pytest.mark.parametrize("cutoff", [-0.1, np.nan, np.inf, np.nextafter(float(np.finfo(np.float32).max), np.inf)])
def test_matrix_sphere_exclusion_rejects_invalid_cutoffs(function, cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        function(_small_matrix(), cutoff)


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


def test_first_picks_must_be_integers():
    with pytest.raises(TypeError, match="first_picks"):
        leader(_small_matrix(), 0.2, first_picks=(1.5,))


@pytest.mark.parametrize("pick_size", [-1, 7])
def test_pick_size_is_validated(pick_size):
    with pytest.raises(ValueError, match="pick_size"):
        leader(_small_matrix(), 0.2, pick_size=pick_size)


def test_metric_and_output_are_validated():
    with pytest.raises(ValueError, match="metric must be"):
        fused_butina(np.asarray([[1]], dtype=np.uint32), 0.2, metric="euclidean")
    with pytest.raises(TypeError, match="OutputMode"):
        leader(_small_matrix(), 0.2, output="rdkit")
