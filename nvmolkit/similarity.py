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

"""GPU-accelerated molecular and fingerprint similarity calculations.

This module provides GPU-accelerated implementations of common RDKit operations
found in the DataStructs module, along with molecular similarity methods.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import torch

from nvmolkit import _clustering, _DataStructs
from nvmolkit._fingerprint_inputs import _prepare_packed_fingerprints
from nvmolkit.types import ArrayInput, AsyncGpuResult, _resolve_cuda_stream, _validate_cuda_stream


@dataclass(frozen=True)
class TanimotoMetric:
    """Tanimoto similarity on packed fingerprints for fused clustering and selection.

    ``metric="tanimoto"`` is equivalent.
    """


@dataclass(frozen=True)
class CosineMetric:
    """Cosine similarity on packed fingerprints for fused clustering and selection.

    ``metric="cosine"`` is equivalent.
    """


@dataclass(frozen=True)
class AAPMetric:
    """Approximate Atom-Atom Path (AAP) similarity on RDKit molecules.

    ``metric="aap"`` is equivalent to ``AAPMetric()``. Fused clustering and
    selection score each selected molecule against the candidates.

    Molecules must be nonempty, contain at most 64 atoms including explicit
    hydrogens, and use only single, double, triple, and aromatic bonds. Other
    inputs raise :class:`ValueError` whose second argument maps ``"none"``,
    ``"empty"``, ``"too_many_atoms"``, and ``"unsupported_bond"`` to lists of
    input indices.

    Attributes:
        max_path_length: Maximum rooted path length in bonds.
        histogram_bins: Number of hashed path bins, from 1 through 32767.
        sinkhorn_iterations: Number of Sinkhorn normalization iterations.
        sinkhorn_temperature: Positive Sinkhorn temperature, at least the
            smallest positive normal float32 value.
        num_threads: Maximum CPU descriptor workers, including the calling
            thread. -1 uses hardware concurrency; positive values set a limit.
            Small inputs may use fewer workers.
    """

    max_path_length: int = 7
    histogram_bins: int = 2048
    sinkhorn_iterations: int = 8
    sinkhorn_temperature: float = 0.104
    num_threads: int = -1


Metric: TypeAlias = Literal["tanimoto", "cosine", "aap"] | TanimotoMetric | CosineMetric | AAPMetric

_DEFAULT_AAP_METRIC = AAPMetric()
_NAMED_METRICS = {"tanimoto": TanimotoMetric(), "cosine": CosineMetric(), "aap": _DEFAULT_AAP_METRIC}


def _resolve_metric(metric: Metric) -> TanimotoMetric | CosineMetric | AAPMetric:
    if isinstance(metric, (TanimotoMetric, CosineMetric, AAPMetric)):
        return metric
    if isinstance(metric, str) and metric in _NAMED_METRICS:
        return _NAMED_METRICS[metric]
    raise ValueError(
        "metric must be 'tanimoto', 'cosine', 'aap', or a TanimotoMetric, CosineMetric, or AAPMetric instance, "
        f"got {metric!r}"
    )


def _resolve_aap_metric(metric: Literal["aap"] | AAPMetric) -> AAPMetric:
    resolved = _resolve_metric(metric)
    if not isinstance(resolved, AAPMetric):
        raise TypeError(f"metric must be 'aap' or an AAPMetric, got {metric!r}")
    return resolved


def aap_similarity(
    left,
    right,
    *,
    metric: Literal["aap"] | AAPMetric = _DEFAULT_AAP_METRIC,
    stream: torch.cuda.Stream | None = None,
) -> float:
    """Compute approximate Atom-Atom Path (AAP) similarity between two molecules.

    The score is directed: ``aap_similarity(a, b)`` and ``aap_similarity(b, a)``
    can differ. Input requirements are described in :class:`AAPMetric`.

    Args:
        left: Reference RDKit molecule.
        right: Candidate RDKit molecule.
        metric: AAP parameters, or ``"aap"`` for the defaults. Use
            ``AAPMetric(num_threads=4)`` to limit CPU descriptor workers.
        stream: CUDA stream to use. If None, uses the current stream.

    Returns:
        Similarity in the interval ``[0, 1]``.

    Note:
        For method details, see `Gobbi et al. (2015)
        <https://doi.org/10.1186/s13321-015-0056-8>`_.
    """
    metric = _resolve_aap_metric(metric)
    active_stream = _resolve_cuda_stream(stream)
    return _clustering.aap_similarity(
        left,
        right,
        metric.max_path_length,
        metric.histogram_bins,
        metric.sinkhorn_iterations,
        metric.sinkhorn_temperature,
        active_stream.cuda_stream,
        metric.num_threads,
    )


# --------------------------------
# Tanimoto similarity
# --------------------------------


def _fingerprint_inputs(
    fingerprint_group_one: ArrayInput,
    fingerprint_group_two: ArrayInput | None,
    stream: torch.cuda.Stream | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Stream]:
    if fingerprint_group_two is None:
        (bits_one,), active_stream = _prepare_packed_fingerprints(
            ("fingerprint_group_one", fingerprint_group_one), stream=stream
        )
        bits_two = bits_one
    else:
        (bits_one, bits_two), active_stream = _prepare_packed_fingerprints(
            ("fingerprint_group_one", fingerprint_group_one),
            ("fingerprint_group_two", fingerprint_group_two),
            stream=stream,
        )
    if bits_one.shape[1] != bits_two.shape[1]:
        raise ValueError("fingerprint_group_one and fingerprint_group_two must have the same feature dimension")
    return bits_one, bits_two, active_stream


def crossTanimotoSimilarity(
    fingerprint_group_one: ArrayInput,
    fingerprint_group_two: ArrayInput | None = None,
    stream: torch.cuda.Stream | None = None,
) -> AsyncGpuResult:
    """Returns the Tanimoto similarity within a set of fingerprints or between two sets of fingerprints.

    Expects fingerprints generated by nvMolKit, a torch tensor, or a numpy array, with the leading dimension corresponding to
    the number of fingerprints, and the second dimension representing the packed fingerprint
    bitfield. CPU tensors and NumPy arrays are copied to CUDA.

    The special case of fingerprint_group_1 as a 1 x n_bits tensor is equivalent to RDKit's BulkTanimotoSimilarity.

    Args:
        fingerprint_group_one: A torch Tensor, numpy.ndarray, or AsyncGpuResult computed from nvMolKit fingerprints
        fingerprint_group_two: A torch Tensor, numpy.ndarray, or AsyncGpuResult computed from nvMolKit fingerprints,
            or None for all-to-all similarity within fingerprint_group_one.
        stream: CUDA stream to use. If None, uses the current stream.

    Returns:
        An n x m matrix of Tanimoto similarities, with index [i, j] corresponding to the
        similarity between fingerprint i in fingerprint_group_one and fingerprint j in
        fingerprint_group_two. If fingerprint_group_two is None, computes all-to-all
        similarity within fingerprint_group_one.
    """
    bits_one, bits_two, active_stream = _fingerprint_inputs(fingerprint_group_one, fingerprint_group_two, stream)
    with torch.cuda.stream(active_stream):
        result = AsyncGpuResult(
            _DataStructs.CrossTanimotoSimilarityRawBuffers(
                bits_one.__cuda_array_interface__, bits_two.__cuda_array_interface__, active_stream.cuda_stream
            )
        )
    result._input_refs = (bits_one, bits_two)
    return result


def crossTanimotoSimilarityMemoryConstrained(
    fingerprint_group_one: ArrayInput, fingerprint_group_two: ArrayInput | None = None
) -> np.ndarray:
    """Returns the Tanimoto similarity within a set of fingerprints or between two sets of fingerprints.

    Computes results on the GPU, but returns a numpy array on the CPU. Will perform computation in chunks if necessary to avoid running out of memory on the GPU. Will still
    fail if the resulting matrix is too large to fit on the CPU.

    Expects fingerprints generated by nvMolKit, a torch tensor, or a numpy array, with the leading dimension corresponding to
    the number of fingerprints, and the second dimension representing the packed fingerprint
    bitfield. CPU tensors and NumPy arrays are copied to CUDA.

    The special case of fingerprint_group_1 as a 1 x n_bits tensor is equivalent to RDKit's BulkTanimotoSimilarity.

    Args:
        fingerprint_group_one: A torch Tensor, numpy.ndarray, or AsyncGpuResult computed from nvMolKit fingerprints
        fingerprint_group_two: A torch Tensor, numpy.ndarray, or AsyncGpuResult computed from nvMolKit fingerprints,
            or None for all-to-all similarity within fingerprint_group_one.

    Returns:
        A NumPy array where element [i, j] is the similarity between fingerprints i and j.
    """
    bits_one, bits_two, active_stream = _fingerprint_inputs(fingerprint_group_one, fingerprint_group_two, None)
    with torch.cuda.stream(active_stream):
        return _DataStructs.CrossTanimotoSimilarityCPURawBuffers(
            bits_one.__cuda_array_interface__, bits_two.__cuda_array_interface__, active_stream.cuda_stream
        )


# --------------------------------
# Cosine similarity
# --------------------------------


def crossCosineSimilarity(
    fingerprint_group_one: ArrayInput,
    fingerprint_group_two: ArrayInput | None = None,
    stream: torch.cuda.Stream | None = None,
) -> AsyncGpuResult:
    """Returns the Cosine similarity between two sets of fingerprints.

    Expects fingerprints generated by nvMolKit, a torch tensor, or a numpy array, with the leading dimension corresponding to
    the number of fingerprints, and the second dimension representing the packed fingerprint
    bitfield. CPU tensors and NumPy arrays are copied to CUDA.

    The special case of fingerprint_group_1 as a 1 x n_bits tensor is equivalent to RDKit's BulkCosineSimilarity.

    Args:
        fingerprint_group_one: A torch Tensor, numpy.ndarray, or AsyncGpuResult computed from nvMolKit fingerprints
        fingerprint_group_two: A torch Tensor, numpy.ndarray, or AsyncGpuResult computed from nvMolKit fingerprints,
            or None for all-to-all similarity within fingerprint_group_one.
        stream: CUDA stream to use. If None, uses the current stream.

    Returns:
        An AsyncGpuResult object containing the Cosine similarities, with index [i, j] corresponding to the similarity between
        fingerprint i in fingerprint_group_one and fingerprint j in fingerprint_group_two.
    """
    bits_one, bits_two, active_stream = _fingerprint_inputs(fingerprint_group_one, fingerprint_group_two, stream)
    with torch.cuda.stream(active_stream):
        result = AsyncGpuResult(
            _DataStructs.CrossCosineSimilarityRawBuffers(
                bits_one.__cuda_array_interface__, bits_two.__cuda_array_interface__, active_stream.cuda_stream
            )
        )
    result._input_refs = (bits_one, bits_two)
    return result


def crossCosineSimilarityMemoryConstrained(
    fingerprint_group_one: ArrayInput, fingerprint_group_two: ArrayInput | None = None
) -> np.ndarray:
    """Returns the Cosine similarity between two sets of fingerprints.

    Computes results on the GPU, but returns a numpy array on the CPU. Will perform computation in chunks if necessary to avoid running out of memory on the GPU. Will still
    fail if the resulting matrix is too large to fit on the CPU.

    Expects fingerprints generated by nvMolKit, a torch tensor, or a numpy array, with the leading dimension corresponding to
    the number of fingerprints, and the second dimension representing the packed fingerprint
    bitfield. CPU tensors and NumPy arrays are copied to CUDA.

    The special case of fingerprint_group_1 as a 1 x n_bits tensor is equivalent to RDKit's BulkCosineSimilarity.

    Args:
        fingerprint_group_one: A torch Tensor, numpy.ndarray, or AsyncGpuResult computed from nvMolKit fingerprints
        fingerprint_group_two: A torch Tensor, numpy.ndarray, or AsyncGpuResult computed from nvMolKit fingerprints,
            or None for all-to-all similarity within fingerprint_group_one.

    Returns:
        A NumPy array where element [i, j] is the similarity between fingerprints i and j.
    """
    bits_one, bits_two, active_stream = _fingerprint_inputs(fingerprint_group_one, fingerprint_group_two, None)
    with torch.cuda.stream(active_stream):
        return _DataStructs.CrossCosineSimilarityCPURawBuffers(
            bits_one.__cuda_array_interface__, bits_two.__cuda_array_interface__, active_stream.cuda_stream
        )


# --------------------------------
# Nearest-neighbor search
# --------------------------------


class FingerprintSearchIndex:
    """Exact top-k similarity search over a fixed library of fingerprints on the GPU.

    The library is copied to the GPU once, sorted by popcount, and can then be searched repeatedly. A search returns
    the k most similar library fingerprints for each query without materializing the query x library similarity
    matrix. Library rows that cannot enter a query's current top k are skipped using the popcount bound on
    similarity, so the cost falls as k shrinks or the threshold rises.

    Results are exact: hits are ordered by descending similarity, then ascending library index, and equal a
    brute-force search with the same ordering. Similarities are computed in double precision exactly as
    :func:`crossTanimotoSimilarity` and :func:`crossCosineSimilarity` compute them.

    Example:
        >>> index = FingerprintSearchIndex(library_fps)
        >>> scores, indices = index.search(query_fps, k=10)
        >>> neighbor_scores, neighbors = index.search(None, k=10)  # library against itself

    Args:
        fingerprints: Packed fingerprints with shape (n, n_words), as produced by nvMolKit fingerprint generators.
            A torch Tensor, numpy.ndarray, or AsyncGpuResult. CPU inputs are copied to CUDA.
        metric: ``"tanimoto"`` or ``"cosine"``.
        stream: CUDA stream to build the index on. If None, uses the current stream. Construction synchronizes this
            stream, after which the input fingerprints are no longer referenced.
    """

    def __init__(
        self,
        fingerprints: ArrayInput,
        metric: str = "tanimoto",
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Build the index from library fingerprints."""
        (bits,), active_stream = _prepare_packed_fingerprints(("fingerprints", fingerprints), stream=stream)
        self._device = bits.device
        with torch.cuda.device(self._device), torch.cuda.stream(active_stream):
            self._index = _DataStructs.FingerprintSearchIndex(
                bits.__cuda_array_interface__, metric, active_stream.cuda_stream
            )
        self._metric = metric

    def __len__(self) -> int:
        """Return the number of library fingerprints."""
        return self._index.size()

    @property
    def metric(self) -> str:
        """Return the similarity metric of this index."""
        return self._metric

    def search(
        self,
        queries: ArrayInput | None,
        k: int,
        threshold: float = 0.0,
        prune: bool = True,
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[AsyncGpuResult, AsyncGpuResult]:
        """Return the k most similar library fingerprints for each query.

        Args:
            queries: Packed query fingerprints with shape (m, n_words), matching the library width. If None, the
                library is searched against itself and each fingerprint's match with itself is excluded; other
                library entries with identical fingerprints are still returned.
            k: Maximum number of hits per query.
            threshold: Only hits with similarity >= threshold are returned. Must be in [0, 1].
            prune: Skip library rows that cannot enter the top k. Disabling it scans every library row and returns
                identical results; it exists for measuring the effect of pruning.
            stream: CUDA stream on the index's device. If None, uses that device's current stream.

        Returns:
            A ``(scores, indices)`` pair of AsyncGpuResults, each with shape (m, k). ``scores`` is float64 and
            ``indices`` is int64 library row indices. Row i lists hits for query i (library fingerprint i when
            ``queries`` is None), best first. Queries with fewer than k hits are padded with NaN scores and -1
            indices.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")

        _validate_cuda_stream(stream)
        active_stream = stream if stream is not None else torch.cuda.current_stream(self._device)
        if active_stream.device != self._device:
            raise ValueError(f"stream is on {active_stream.device}, but the index is on {self._device}")

        with torch.cuda.device(self._device), torch.cuda.stream(active_stream):
            if queries is None:
                scores, indices = self._index.searchSelf(k, float(threshold), bool(prune), active_stream.cuda_stream)
            else:
                (bits,), _ = _prepare_packed_fingerprints(("queries", queries), stream=active_stream)
                if bits.device != self._device:
                    raise ValueError(f"queries are on {bits.device}, but the index is on {self._device}")
                if bits.shape[1] != self._index.numWords():
                    raise ValueError(
                        f"queries have {bits.shape[1]} words per fingerprint, "
                        f"but the index has {self._index.numWords()}"
                    )
                # Keep the caller's query memory from being reused by another stream before the search finishes.
                bits.record_stream(active_stream)
                scores, indices = self._index.search(
                    bits.__cuda_array_interface__, k, float(threshold), bool(prune), active_stream.cuda_stream
                )

        gpu_id = self._device.index
        return AsyncGpuResult(scores, gpu_id=gpu_id), AsyncGpuResult(indices, gpu_id=gpu_id)
