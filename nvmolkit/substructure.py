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

"""GPU-accelerated substructure search.

Provides batch substructure matching of RDKit molecules on GPU.  All three
entry points accept a list of target molecules and a list of query molecules
(typically built from SMARTS via ``Chem.MolFromSmarts``), and return results
for every (target, query) pair.

* :func:`hasSubstructMatch` -- boolean existence check (fastest)
* :func:`countSubstructMatches` -- match counts per pair
* :func:`getSubstructMatches` -- full atom-index mappings

Execution can be tuned with :class:`SubstructSearchConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from rdkit.Chem import Mol

from nvmolkit._substructure import SubstructSearchConfig as _NativeSubstructSearchConfig
from nvmolkit._substructure import countSubstructMatches as _countSubstructMatches
from nvmolkit._substructure import getSubstructMatches as _getSubstructMatches
from nvmolkit._substructure import hasSubstructMatch as _hasSubstructMatch

__all__ = [
    "SubstructSearchConfig",
    "SubstructMatchResults",
    "getSubstructMatches",
    "countSubstructMatches",
    "hasSubstructMatch",
]


class SubstructSearchConfig:
    """Configuration for GPU substructure search execution.

    Args:
        batchSize: Number of (target, query) pairs per GPU batch. Default 1024.
        workerThreads: GPU runner threads per GPU. -1 for autoselect.
        preprocessingThreads: CPU threads for preprocessing. -1 for autoselect.
        maxMatches: Maximum matches per pair. 0 for unlimited (default).
        uniquify: If True, remove duplicate matches that differ only in
            atom enumeration order. Default False.
        gpuIds: GPU device IDs to use. ``None`` or empty list uses current device only.
    """

    def __init__(
        self,
        batchSize: int = 1024,
        workerThreads: int = -1,
        preprocessingThreads: int = -1,
        maxMatches: int = 0,
        uniquify: bool = False,
        gpuIds: list[int] | None = None,
    ) -> None:
        native = _NativeSubstructSearchConfig()
        native.batchSize = int(batchSize)
        native.workerThreads = int(workerThreads)
        native.preprocessingThreads = int(preprocessingThreads)
        native.maxMatches = int(maxMatches)
        native.uniquify = bool(uniquify)
        native.gpuIds = list(gpuIds) if gpuIds is not None else []
        self._native = native

    @property
    def batchSize(self) -> int:
        """Number of (target, query) pairs per GPU batch."""
        return self._native.batchSize

    @batchSize.setter
    def batchSize(self, value: int) -> None:
        self._native.batchSize = int(value)

    @property
    def workerThreads(self) -> int:
        """GPU runner threads per GPU. -1 for autoselect."""
        return self._native.workerThreads

    @workerThreads.setter
    def workerThreads(self, value: int) -> None:
        self._native.workerThreads = int(value)

    @property
    def preprocessingThreads(self) -> int:
        """CPU threads for preprocessing. -1 for autoselect."""
        return self._native.preprocessingThreads

    @preprocessingThreads.setter
    def preprocessingThreads(self, value: int) -> None:
        self._native.preprocessingThreads = int(value)

    @property
    def maxMatches(self) -> int:
        """Maximum matches per pair. 0 for unlimited."""
        return self._native.maxMatches

    @maxMatches.setter
    def maxMatches(self, value: int) -> None:
        self._native.maxMatches = int(value)

    @property
    def uniquify(self) -> bool:
        """Remove duplicate matches differing only in atom enumeration order."""
        return self._native.uniquify

    @uniquify.setter
    def uniquify(self, value: bool) -> None:
        self._native.uniquify = bool(value)

    @property
    def gpuIds(self) -> list[int]:
        """GPU device IDs to use. Empty list uses current device only."""
        return list(self._native.gpuIds)

    @gpuIds.setter
    def gpuIds(self, value: list[int]) -> None:
        self._native.gpuIds = list(value)

    def _as_native(self):
        """Internal: return the underlying native config object."""
        return self._native

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of this object's fields."""
        return {
            "batchSize": self.batchSize,
            "workerThreads": self.workerThreads,
            "preprocessingThreads": self.preprocessingThreads,
            "maxMatches": self.maxMatches,
            "uniquify": self.uniquify,
            "gpuIds": list(self.gpuIds),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubstructSearchConfig":
        """Create a :class:`SubstructSearchConfig` from a dictionary produced by :meth:`to_dict`."""
        known = {"batchSize", "workerThreads", "preprocessingThreads", "maxMatches", "uniquify", "gpuIds"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown SubstructSearchConfig keys: {sorted(unknown)}")
        return cls(**{key: data[key] for key in known if key in data})


@dataclass(frozen=True)
class SubstructMatchResults:
    """Results of a batch substructure search.

    Indexable as ``results[target_idx][query_idx]`` to obtain a list of numpy
    arrays, each containing the target atom indices for one match.  Also
    supports ``results.get_pair(target_idx, query_idx)`` for direct access.
    """

    atom_indices: np.ndarray
    match_indptr: np.ndarray
    pair_indptr: np.ndarray
    shape: tuple[int, int]

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, target_idx: int) -> _SubstructTargetView:
        return _SubstructTargetView(self, target_idx)

    def get_pair(self, target_idx: int, query_idx: int) -> list[np.ndarray]:
        """Return matches for a single (target, query) pair.

        Args:
            target_idx: Index into the targets list.
            query_idx: Index into the queries list.

        Returns:
            List of numpy arrays, each containing target atom indices for one match.
        """
        num_targets, num_queries = self.shape
        if target_idx < 0:
            target_idx += num_targets
        if query_idx < 0:
            query_idx += num_queries
        if not (0 <= target_idx < num_targets and 0 <= query_idx < num_queries):
            raise IndexError("pair index out of range")

        pair_idx = target_idx * num_queries + query_idx
        m0 = int(self.pair_indptr[pair_idx])
        m1 = int(self.pair_indptr[pair_idx + 1])
        out: list[np.ndarray] = []
        for m in range(m0, m1):
            a0 = int(self.match_indptr[m])
            a1 = int(self.match_indptr[m + 1])
            out.append(self.atom_indices[a0:a1])
        return out


@dataclass(frozen=True)
class _SubstructTargetView:
    parent: SubstructMatchResults
    target_idx: int

    def __len__(self) -> int:
        return self.parent.shape[1]

    def __getitem__(self, query_idx: int) -> list[np.ndarray]:
        return self.parent.get_pair(self.target_idx, query_idx)


def hasSubstructMatch(
    targets: Sequence[Mol],
    queries: Sequence[Mol],
    config: SubstructSearchConfig | None = None,
) -> np.ndarray:
    """Check if targets contain query substructures (boolean results).

    More efficient than :func:`getSubstructMatches` when only existence is needed.

    Supports recursive SMARTS queries. Does not currently support
    chirality-aware matching (``useChirality``), enhanced stereochemistry
    (``useEnhancedStereo``), or other advanced RDKit
    ``SubstructMatchParameters`` options.

    Args:
        targets: List of target RDKit molecules.
        queries: List of query RDKit molecules (typically from SMARTS).
        config: :class:`SubstructSearchConfig` with execution settings.
            If ``None``, uses default configuration.

    Returns:
        2-D numpy array of ``uint8`` with shape ``(len(targets), len(queries))``.
        ``results[target_idx, query_idx]`` is 1 if the query is a substructure
        of the target, 0 otherwise.
    """
    if config is None:
        config = SubstructSearchConfig()
    return _hasSubstructMatch(targets, queries, config._as_native())


def countSubstructMatches(
    targets: Sequence[Mol],
    queries: Sequence[Mol],
    config: SubstructSearchConfig | None = None,
) -> np.ndarray:
    """Count substructure matches per target/query pair.

    Supports match deduplication (``uniquify``) and recursive SMARTS queries.
    Does not currently support chirality-aware matching (``useChirality``),
    enhanced stereochemistry (``useEnhancedStereo``), or other advanced RDKit
    ``SubstructMatchParameters`` options.

    Args:
        targets: List of target RDKit molecules.
        queries: List of query RDKit molecules (typically from SMARTS).
        config: :class:`SubstructSearchConfig` with execution settings.
            If ``None``, uses default configuration.

    Returns:
        2-D numpy array of ``int`` with shape ``(len(targets), len(queries))``.
        ``results[target_idx, query_idx]`` is the number of substructure matches.
    """
    if config is None:
        config = SubstructSearchConfig()
    return _countSubstructMatches(targets, queries, config._as_native())


def getSubstructMatches(
    targets: Sequence[Mol],
    queries: Sequence[Mol],
    config: SubstructSearchConfig | None = None,
) -> SubstructMatchResults:
    """Perform batch substructure matching on GPU, returning full atom-index mappings.

    Supports match deduplication (``uniquify``) and recursive SMARTS queries.
    Does not currently support chirality-aware matching (``useChirality``),
    enhanced stereochemistry (``useEnhancedStereo``), or other advanced RDKit
    ``SubstructMatchParameters`` options.

    Args:
        targets: List of target RDKit molecules.
        queries: List of query RDKit molecules (typically from SMARTS).
        config: :class:`SubstructSearchConfig` with execution settings.
            If ``None``, uses default configuration.

    Returns:
        A :class:`SubstructMatchResults` providing list-like access to match results.
        Index as ``results[target_idx][query_idx]`` to get a list of numpy
        arrays, each containing the target atom indices for one match.
    """
    if config is None:
        config = SubstructSearchConfig()
    atom_indices, match_indptr, pair_indptr, shape = _getSubstructMatches(targets, queries, config._as_native())
    return SubstructMatchResults(atom_indices, match_indptr, pair_indptr, shape)
