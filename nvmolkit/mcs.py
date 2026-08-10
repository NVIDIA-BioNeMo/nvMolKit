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

"""GPU-accelerated maximum common substructure search.

The public :func:`findMCS` entry point searches many independent molecule
pairs in one call.  It can generate pairs from one molecule table, accept an
explicit list of pairs, or zip two equally sized molecule lists.  See the
function docstring for the pair ordering, index spaces, fallback behavior, and
currently unsupported RDKit fMCS features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from rdkit.Chem import Mol

from nvmolkit._mcs import _findMCSBatch

__all__ = ["MCSBatchResult", "MCSConfig", "MCSResult", "findMCS"]


Pair = tuple[int, int]

_ATOM_COMPARE_ALIASES = {
    "any": "any",
    "elements": "elements",
    "element": "elements",
    "isotopes": "isotopes",
    "isotope": "isotopes",
    "any_heavy_atom": "any_heavy_atom",
    "anyheavyatom": "any_heavy_atom",
}

_BOND_COMPARE_ALIASES = {
    "any": "any",
    "order": "order",
    "orders": "order",
    "order_exact": "order_exact",
    "orderexact": "order_exact",
    "exact": "order_exact",
}


@dataclass(frozen=True)
class MCSResult:
    """Result for one molecule pair."""

    pair: Pair
    num_atoms: int
    num_bonds: int
    canceled: bool
    atom_mapping: np.ndarray
    bond_mapping: np.ndarray


@dataclass(frozen=True)
class MCSBatchResult:
    """Flat batch of MCS results in generated-pair order."""

    pairs: tuple[Pair, ...]
    mode: str
    num_atoms: np.ndarray
    num_bonds: np.ndarray
    canceled: np.ndarray
    atom_mapping: np.ndarray
    atom_mapping_indptr: np.ndarray
    bond_mapping: np.ndarray
    bond_mapping_indptr: np.ndarray

    def __len__(self) -> int:
        """Return the number of generated pair results."""
        return len(self.pairs)

    def __getitem__(self, pair_idx: int) -> MCSResult:
        """Return a materialized result object for one generated pair."""
        return self.get_result(pair_idx)

    def get_result(self, pair_idx: int) -> MCSResult:
        """Return a materialized result object for one generated pair."""
        if pair_idx < 0:
            pair_idx += len(self)
        if not 0 <= pair_idx < len(self):
            raise IndexError("MCS result index out of range")

        atom_start = int(self.atom_mapping_indptr[pair_idx])
        atom_end = int(self.atom_mapping_indptr[pair_idx + 1])
        bond_start = int(self.bond_mapping_indptr[pair_idx])
        bond_end = int(self.bond_mapping_indptr[pair_idx + 1])

        return MCSResult(
            pair=self.pairs[pair_idx],
            num_atoms=int(self.num_atoms[pair_idx]),
            num_bonds=int(self.num_bonds[pair_idx]),
            canceled=bool(self.canceled[pair_idx]),
            atom_mapping=self.atom_mapping[atom_start:atom_end],
            bond_mapping=self.bond_mapping[bond_start:bond_end],
        )


class MCSConfig:
    """Configuration for GPU MCS execution.

    Args:
        batchSize: Optional GPU batch chunk size. ``0`` processes each
            nonempty size tier as one chunk.
        blockSize: CUDA threads per fMCS pair block. Supported values are
            ``64``, ``128``, ``256``, and ``512``.
        workerThreads: GPU runner threads per GPU. ``-1`` autoselects.
        preprocessingThreads: CPU threads for pair preprocessing. ``-1``
            autoselects.
        executorsPerRunner: Number of asynchronous GPU executor streams used
            for chunked fMCS tier dispatch. ``-1`` autoselects.
        gpuIds: GPU device IDs to use. ``None`` or empty uses the current
            device only, not all available devices.
    """

    def __init__(
        self,
        batchSize: int = 0,
        blockSize: int = 128,
        workerThreads: int = -1,
        preprocessingThreads: int = -1,
        executorsPerRunner: int = -1,
        gpuIds: Sequence[int] | None = None,
    ) -> None:
        """Initialize GPU execution settings."""
        self.batchSize = int(batchSize)
        self.blockSize = int(blockSize)
        self.workerThreads = int(workerThreads)
        self.preprocessingThreads = int(preprocessingThreads)
        self.executorsPerRunner = int(executorsPerRunner)
        self.gpuIds = list(gpuIds) if gpuIds is not None else []

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of this object's fields."""
        return {
            "batchSize": self.batchSize,
            "blockSize": self.blockSize,
            "workerThreads": self.workerThreads,
            "preprocessingThreads": self.preprocessingThreads,
            "executorsPerRunner": self.executorsPerRunner,
            "gpuIds": list(self.gpuIds),
        }

    def to_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by :func:`findMCS`."""
        return {
            "batch_size": self.batchSize,
            "block_size": self.blockSize,
            "worker_threads": self.workerThreads,
            "preprocessing_threads": self.preprocessingThreads,
            "executors_per_runner": self.executorsPerRunner,
            "gpu_ids": list(self.gpuIds),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCSConfig":
        """Create an :class:`MCSConfig` from a dictionary produced by :meth:`to_dict`."""
        known = {
            "batchSize",
            "blockSize",
            "workerThreads",
            "preprocessingThreads",
            "executorsPerRunner",
            "gpuIds",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown MCSConfig keys: {sorted(unknown)}")
        return cls(**{key: data[key] for key in known if key in data})


def _normalize_mode(mode: str) -> str:
    normalized = mode.lower().replace("-", "_")
    if normalized not in {"all_pairs", "pairs", "paired_lists"}:
        raise ValueError("mode must be one of 'all_pairs', 'pairs', or 'paired_lists'")
    return normalized


def _normalize_atom_compare(value: str) -> str:
    normalized = value.lower().replace("-", "_")
    if normalized not in _ATOM_COMPARE_ALIASES:
        raise ValueError(f"Unsupported atom_compare value: {value!r}")
    return _ATOM_COMPARE_ALIASES[normalized]


def _normalize_bond_compare(value: str) -> str:
    normalized = value.lower().replace("-", "_")
    if normalized not in _BOND_COMPARE_ALIASES:
        raise ValueError(f"Unsupported bond_compare value: {value!r}")
    return _BOND_COMPARE_ALIASES[normalized]


def _coerce_pairs(pairs: Sequence[Sequence[int]]) -> tuple[Pair, ...]:
    out: list[Pair] = []
    for i, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError(f"pairs[{i}] must contain exactly two molecule indices")
        out.append((int(pair[0]), int(pair[1])))
    return tuple(out)


def _all_pairs(num_mols: int, upper_triangle: bool, include_diagonal: bool) -> tuple[Pair, ...]:
    pairs: list[Pair] = []
    if upper_triangle:
        for i in range(num_mols):
            begin = i if include_diagonal else i + 1
            for j in range(begin, num_mols):
                pairs.append((i, j))
    else:
        for i in range(num_mols):
            for j in range(num_mols):
                if include_diagonal or i != j:
                    pairs.append((i, j))
    return tuple(pairs)


def findMCS(
    mols: Sequence[Mol],
    *,
    mode: str = "all_pairs",
    pairs: Sequence[Sequence[int]] | None = None,
    mols_b: Sequence[Mol] | None = None,
    upper_triangle: bool = True,
    include_diagonal: bool = True,
    atom_compare: str = "elements",
    bond_compare: str = "order",
    match_valences: bool = False,
    match_formal_charge: bool = False,
    ring_matches_ring_only: bool | None = None,
    atom_ring_matches_ring_only: bool | None = None,
    bond_ring_matches_ring_only: bool | None = None,
    match_isotope: bool = False,
    maximize_bonds: bool = True,
    connected_only: bool = True,
    require_gpu: bool = False,
    timeout_seconds: int = 0,
    config: MCSConfig | None = None,
    batch_size: int = 0,
    block_size: int = 128,
    worker_threads: int = -1,
    preprocessing_threads: int = -1,
    executors_per_runner: int = -1,
    gpu_ids: Sequence[int] | None = None,
) -> MCSBatchResult:
    """Find maximum common substructures for a batch of molecule pairs.

    Input modes:
        ``mode="all_pairs"`` uses ``mols`` as a single indexed table.  The
        default generates ``(i, j)`` in upper-triangle row-major order for
        ``i <= j``, including self-pairs.  Set ``include_diagonal=False`` to
        omit self-pairs.  Set ``upper_triangle=False`` to generate the full
        row-major Cartesian square instead; ``include_diagonal`` still
        controls self-pairs.

        ``mode="pairs"`` also uses one ``mols`` table.  ``pairs`` is a
        sequence of index pairs such as ``[(i, j), (i, j), ...]``; the search
        processes exactly those entries in their given order.  Repeated,
        reversed, and self-pairs are retained, and both indices in every pair
        address ``mols``.

        ``mode="paired_lists"`` zips two equally sized lists, searching
        ``mols[i]`` against ``mols_b[i]``.  Internally the lists are combined,
        so ``result.pairs`` contains ``(i, len(mols) + i)``.  The first and
        second columns of each atom or bond mapping still refer to the
        corresponding molecules from ``mols`` and ``mols_b``, respectively.

    Results are flat and follow the generated pair order.  ``result[k]``
    materializes the MCS for ``result.pairs[k]``; it does not index results by
    molecule ID.  Mapping arrays contain ``(first_molecule_index,
    second_molecule_index)`` atom or bond index pairs.

    Fallback behavior:
        Molecule pairs outside the native GPU limits (currently more than 128
        atoms, more than 128 bonds, or atom degree greater than 8), and pairs
        whose GPU search queue overflows, transparently use RDKit on the CPU.
        Set ``require_gpu=True`` to raise instead.  This fallback is
        pair-specific; unsupported search options listed below raise for the
        whole call and do not trigger fallback.

    Timeouts:
        ``timeout_seconds`` is an independent budget for each generated pair,
        not one deadline shared by the batch.  A timed-out pair has
        ``canceled=True`` and may contain the best valid partial MCS found so
        far; other pairs continue independently.  GPU timing starts when that
        pair's kernel block begins and therefore excludes host preprocessing
        and dispatch time.  A pair routed to RDKit receives the same timeout.
        Zero disables the timeout.

    Unsupported features:
        The interface currently supports exact, connected, bond-maximizing
        two-molecule MCS searches only.  ``connected_only=False`` and
        ``maximize_bonds=False`` raise :class:`ValueError`.

        ``atom_compare="any_heavy_atom"`` and ``match_isotope=True`` are not
        implemented and also raise.  Use ``atom_compare="isotopes"`` for
        isotope-based atom comparison.

        Other RDKit fMCS features are not exposed: disconnected results,
        ``StoreAll``, thresholds other than 1.0, initial SMARTS seeds, custom
        atom/bond typers or callbacks, chirality and bond-stereo matching,
        complete-ring and fused-ring constraints, atom maximum-distance
        constraints, and verbose progress output.

    Args:
        mols: Primary molecule table. In ``pairs`` and ``all_pairs`` modes,
            generated pair indices refer to this table.
        mode: Dispatch shape. ``"all_pairs"`` builds square pair specs from
            ``mols``. ``"pairs"`` uses explicit ``pairs`` over ``mols``.
            ``"paired_lists"`` pairs ``mols[i]`` with ``mols_b[i]``.
        pairs: For ``mode="pairs"``, a sequence of molecule-index pairs such
            as ``[(0, 1), (2, 3)]``. Each ``(i, j)`` searches ``mols[i]``
            against ``mols[j]``, and results preserve the sequence order.
        mols_b: Second molecule list for ``mode="paired_lists"``.
        upper_triangle: For ``mode="all_pairs"``, generate only upper-triangle
            pairs when true, otherwise generate full row-major square pairs.
        include_diagonal: Include ``(i, i)`` all-pairs entries.
        atom_compare: ``"any"``, ``"elements"``, or ``"isotopes"``.
            ``"any_heavy_atom"`` is recognized but not implemented and
            raises.
        bond_compare: ``"any"``, ``"order"``, or ``"order_exact"``.
        match_valences: Match atom total valence.
        match_formal_charge: Match atom formal charge.
        ring_matches_ring_only: Convenience value applied to both atom and bond
            ring matching unless the axis-specific arguments are supplied.
        atom_ring_matches_ring_only: Match ring atoms only to ring atoms.
        bond_ring_matches_ring_only: Match ring bonds only to ring bonds.
        match_isotope: Not currently implemented; ``True`` raises. Use
            ``atom_compare="isotopes"`` for isotope-based comparison.
        maximize_bonds: Maximize bonds, matching RDKit's default fMCS objective.
            ``False`` is not currently supported and raises.
        connected_only: Require connected MCS. ``False`` is not currently
            supported and raises.
        require_gpu: Raise instead of using per-pair RDKit fallback for native
            GPU size, degree, or queue-capacity limits.
        timeout_seconds: Per-pair timeout in seconds. GPU results canceled by
            timeout return the best partial MCS found so far; RDKit fallback
            receives the same timeout.
        config: Optional :class:`MCSConfig` with GPU execution settings. Cannot
            be combined with explicit GPU execution keyword options.
        batch_size: Optional GPU batch chunk size. ``0`` processes each
            nonempty size tier as one chunk.
        block_size: CUDA threads per fMCS pair block. Supported values are
            ``64``, ``128``, ``256``, and ``512``.
        worker_threads: GPU runner threads per GPU. ``-1`` autoselects.
        preprocessing_threads: CPU threads for pair preprocessing. ``-1``
            autoselects.
        executors_per_runner: Number of asynchronous GPU executor streams used
            for chunked fMCS tier dispatch. ``-1`` autoselects.
        gpu_ids: GPU device IDs to use. ``None`` or empty uses the current
            device only, not all available devices.

    Returns:
        :class:`MCSBatchResult` in generated-pair order.
    """
    mode = _normalize_mode(mode)
    mol_list = list(mols)

    if config is not None:
        explicit_options = []
        if batch_size != 0:
            explicit_options.append("batch_size")
        if block_size != 128:
            explicit_options.append("block_size")
        if worker_threads != -1:
            explicit_options.append("worker_threads")
        if preprocessing_threads != -1:
            explicit_options.append("preprocessing_threads")
        if executors_per_runner != -1:
            explicit_options.append("executors_per_runner")
        if gpu_ids is not None:
            explicit_options.append("gpu_ids")
        if explicit_options:
            joined = ", ".join(explicit_options)
            raise ValueError(f"config cannot be combined with explicit GPU execution options: {joined}")
        batch_size = config.batchSize
        block_size = config.blockSize
        worker_threads = config.workerThreads
        preprocessing_threads = config.preprocessingThreads
        executors_per_runner = config.executorsPerRunner
        gpu_ids = config.gpuIds

    if mode == "all_pairs":
        if pairs is not None:
            raise ValueError("pairs is only valid with mode='pairs'")
        if mols_b is not None:
            raise ValueError("mols_b is only valid with mode='paired_lists'")
        native_mols = mol_list
        pair_list = _all_pairs(len(native_mols), bool(upper_triangle), bool(include_diagonal))
    elif mode == "pairs":
        if pairs is None:
            raise ValueError("pairs is required with mode='pairs'")
        if mols_b is not None:
            raise ValueError("mols_b is only valid with mode='paired_lists'")
        native_mols = mol_list
        pair_list = _coerce_pairs(pairs)
    else:
        if pairs is not None:
            raise ValueError("pairs is only valid with mode='pairs'")
        if mols_b is None:
            raise ValueError("mols_b is required with mode='paired_lists'")
        mols_b_list = list(mols_b)
        if len(mol_list) != len(mols_b_list):
            raise ValueError("mols and mols_b must have the same length with mode='paired_lists'")
        native_mols = mol_list + mols_b_list
        pair_list = tuple((i, len(mol_list) + i) for i in range(len(mol_list)))

    if ring_matches_ring_only is not None:
        if atom_ring_matches_ring_only is None:
            atom_ring_matches_ring_only = ring_matches_ring_only
        if bond_ring_matches_ring_only is None:
            bond_ring_matches_ring_only = ring_matches_ring_only

    (
        num_atoms,
        num_bonds,
        canceled,
        atom_mapping,
        atom_mapping_indptr,
        bond_mapping,
        bond_mapping_indptr,
    ) = _findMCSBatch(
        native_mols,
        list(pair_list),
        {
            "atom_compare": _normalize_atom_compare(atom_compare),
            "bond_compare": _normalize_bond_compare(bond_compare),
            "maximize_bonds": bool(maximize_bonds),
            "connected_only": bool(connected_only),
            "require_gpu": bool(require_gpu),
            "timeout_seconds": int(timeout_seconds),
            "batch_size": int(batch_size),
            "block_size": int(block_size),
            "worker_threads": int(worker_threads),
            "preprocessing_threads": int(preprocessing_threads),
            "executors_per_runner": int(executors_per_runner),
            "gpu_ids": list(gpu_ids) if gpu_ids is not None else [],
            "match_valences": bool(match_valences),
            "match_formal_charge": bool(match_formal_charge),
            "atom_ring_matches_ring_only": bool(atom_ring_matches_ring_only)
            if atom_ring_matches_ring_only is not None
            else False,
            "match_isotope": bool(match_isotope),
            "bond_ring_matches_ring_only": bool(bond_ring_matches_ring_only)
            if bond_ring_matches_ring_only is not None
            else False,
        },
    )

    return MCSBatchResult(
        pairs=pair_list,
        mode=mode,
        num_atoms=num_atoms,
        num_bonds=num_bonds,
        canceled=canceled,
        atom_mapping=atom_mapping,
        atom_mapping_indptr=atom_mapping_indptr,
        bond_mapping=bond_mapping,
        bond_mapping_indptr=bond_mapping_indptr,
    )
