# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkCosineSimilarity, BulkTanimotoSimilarity

from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.similarity import FingerprintSearchIndex

RDKIT_BULK = {"tanimoto": BulkTanimotoSimilarity, "cosine": BulkCosineSimilarity}


def _fingerprints(mols, radius, fp_size):
    rdkit_fps = [
        rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size).GetFingerprint(m) for m in mols
    ]
    packed = MorganFingerprintGenerator(radius=radius, fpSize=fp_size).GetFingerprints(mols, num_threads=1).torch()
    return rdkit_fps, packed


def _rdkit_top_k(query_fps, library_fps, k, threshold, metric, exclude_self=False):
    """Brute-force top k with RDKit similarities: descending score, then ascending library index."""
    scores = np.full((len(query_fps), k), np.nan)
    indices = np.full((len(query_fps), k), -1, dtype=np.int64)
    for row, query in enumerate(query_fps):
        sims = RDKIT_BULK[metric](query, library_fps)
        hits = [(-s, i) for i, s in enumerate(sims) if s >= threshold and not (exclude_self and i == row)]
        hits.sort()
        for slot, (neg_score, index) in enumerate(hits[:k]):
            scores[row, slot] = -neg_score
            indices[row, slot] = index
    return scores, indices


def _assert_hits_equal(got, want):
    got_scores, got_indices = (r.numpy() for r in got)
    want_scores, want_indices = want
    np.testing.assert_array_equal(got_indices, want_indices)
    np.testing.assert_array_equal(np.isnan(got_scores), np.isnan(want_scores))
    np.testing.assert_array_equal(got_scores[~np.isnan(want_scores)], want_scores[~np.isnan(want_scores)])


@pytest.fixture
def library(size_limited_mols):
    # Radius 0 and 1 fingerprints are sparse, so many library entries tie and the tie order is exercised.
    return list(size_limited_mols) * 8


@pytest.mark.parametrize("metric", ["tanimoto", "cosine"])
@pytest.mark.parametrize("radius,fp_size", [(0, 512), (1, 1024), (3, 2048)])
@pytest.mark.parametrize("k,threshold", [(1, 0.0), (7, 0.0), (64, 0.3), (2000, 0.0)])
def test_search_matches_rdkit(library, metric, radius, fp_size, k, threshold):
    rdkit_fps, packed = _fingerprints(library, radius, fp_size)
    queries = slice(0, len(library), 9)
    index = FingerprintSearchIndex(packed, metric=metric)
    want = _rdkit_top_k(rdkit_fps[queries], rdkit_fps, k, threshold, metric)
    _assert_hits_equal(index.search(packed[queries], k=k, threshold=threshold), want)
    _assert_hits_equal(index.search(packed[queries], k=k, threshold=threshold, prune=False), want)


@pytest.mark.parametrize("metric", ["tanimoto", "cosine"])
@pytest.mark.parametrize("k,threshold", [(1, 0.0), (10, 0.0), (64, 0.3)])
def test_held_out_queries_match_rdkit(size_limited_mols, metric, k, threshold):
    # Queries are molecules that are not in the library, so no query finds itself.
    half = len(size_limited_mols) // 2
    library_fps, library_packed = _fingerprints(list(size_limited_mols[:half]) * 8, 2, 1024)
    query_fps, query_packed = _fingerprints(size_limited_mols[half:], 2, 1024)
    index = FingerprintSearchIndex(library_packed, metric=metric)
    want = _rdkit_top_k(query_fps, library_fps, k, threshold, metric)
    _assert_hits_equal(index.search(query_packed, k=k, threshold=threshold), want)
    _assert_hits_equal(index.search(query_packed, k=k, threshold=threshold, prune=False), want)


@pytest.mark.parametrize("metric", ["tanimoto", "cosine"])
def test_self_search_excludes_only_self(library, metric):
    rdkit_fps, packed = _fingerprints(library, 2, 1024)
    index = FingerprintSearchIndex(packed, metric=metric)
    scores, indices = index.search(None, k=12)
    _assert_hits_equal((scores, indices), _rdkit_top_k(rdkit_fps, rdkit_fps, 12, 0.0, metric, exclude_self=True))
    # The library repeats every molecule, so each row's best hits are its copies at similarity 1.
    assert (scores.numpy()[:, 0] == 1.0).all()


def test_threshold_one_returns_only_identical(library):
    rdkit_fps, packed = _fingerprints(library, 2, 1024)
    index = FingerprintSearchIndex(packed)
    want = _rdkit_top_k(rdkit_fps[:20], rdkit_fps, 20, 1.0, "tanimoto")
    _assert_hits_equal(index.search(packed[:20], k=20, threshold=1.0), want)


def test_accepts_numpy_and_cpu_inputs_and_streams(library):
    rdkit_fps, packed = _fingerprints(library, 2, 1024)
    index = FingerprintSearchIndex(packed.cpu().numpy())
    stream = torch.cuda.Stream()
    got = index.search(packed[:10].cpu(), k=5, stream=stream)
    stream.synchronize()
    _assert_hits_equal(got, _rdkit_top_k(rdkit_fps[:10], rdkit_fps, 5, 0.0, "tanimoto"))
    assert len(index) == len(library)
    assert index.metric == "tanimoto"


def test_cpu_queries_search_index_on_another_device(library):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires >= 2 GPUs")
    rdkit_fps, packed = _fingerprints(library, 2, 1024)
    index = FingerprintSearchIndex(packed.to("cuda:1"))
    with torch.cuda.device(0):
        got = index.search(packed[:10].cpu(), k=5)
    torch.cuda.synchronize(1)
    _assert_hits_equal(got, _rdkit_top_k(rdkit_fps[:10], rdkit_fps, 5, 0.0, "tanimoto"))


def test_result_types_and_padding():
    packed = torch.zeros((3, 2), dtype=torch.int32, device="cuda")
    packed[1, 0] = 1
    index = FingerprintSearchIndex(packed)
    scores, indices = index.search(packed, k=5)
    assert scores.torch().dtype == torch.float64
    assert indices.torch().dtype == torch.int64
    assert tuple(scores.torch().shape) == (3, 5)
    got_indices = indices.numpy()
    # Two all-zero fingerprints have Tanimoto 1 with each other; the remaining slots are padding.
    assert got_indices[:, 3:].tolist() == [[-1, -1]] * 3
    assert np.isnan(scores.numpy()[:, 3:]).all()


def test_empty_queries_and_library():
    empty = torch.zeros((0, 4), dtype=torch.int32, device="cuda")
    one = torch.ones((1, 4), dtype=torch.int32, device="cuda")
    scores, indices = FingerprintSearchIndex(one).search(empty, k=3)
    assert tuple(scores.torch().shape) == (0, 3)
    scores, indices = FingerprintSearchIndex(empty).search(one, k=3)
    assert indices.numpy().tolist() == [[-1, -1, -1]]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"k": 0}, "k must be a positive integer"),
        ({"k": 2.0}, "k must be a positive integer"),
        ({"k": 1, "threshold": 1.5}, "threshold must be in"),
        ({"k": 1, "threshold": -0.1}, "threshold must be in"),
    ],
)
def test_search_rejects_invalid_arguments(kwargs, error):
    index = FingerprintSearchIndex(torch.ones((2, 4), dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match=error):
        index.search(None, **kwargs)


def test_rejects_mismatched_width_and_metric():
    index = FingerprintSearchIndex(torch.ones((2, 4), dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="words per fingerprint"):
        index.search(torch.ones((2, 8), dtype=torch.int32, device="cuda"), k=1)
    with pytest.raises(ValueError, match="metric must be"):
        FingerprintSearchIndex(torch.ones((2, 4), dtype=torch.int32, device="cuda"), metric="dice")
