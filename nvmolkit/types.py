# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Types facilitating GPU-accelerated operations."""

from enum import Enum
from typing import Any, Iterable, List, NamedTuple, Optional

import torch

from nvmolkit import _embedMolecules  # type: ignore


class HardwareOptions:
    """Configures GPU hardware settings for batch operations.

    Use this class to control threading and batching behavior for GPU-accelerated
    workflows such as ETKDG embedding and MMFF optimization.

    Parameters:
        preprocessingThreads: Number of CPU threads for preprocessing. Use -1 to auto-detect and use all CPU threads.
        batchSize: Number of conformers processed per batch. -1 for default.
        batchesPerGpu: Number of batches processed concurrently on each GPU. Use -1 for default.
        gpuIds: GPU device IDs to target. Provide an empty list to use all available GPUs.
    """

    def __init__(
        self,
        preprocessingThreads: int = -1,
        batchSize: int = -1,
        batchesPerGpu: int = -1,
        gpuIds: Iterable[int] | None = None,
    ) -> None:
        if _embedMolecules is None:  # propagate real import failure early
            raise ImportError("nvmolkit._embedMolecules is not available; build native extensions")
        native = _embedMolecules.BatchHardwareOptions()
        native.preprocessingThreads = int(preprocessingThreads)
        native.batchSize = int(batchSize)
        native.gpuIds = list(gpuIds) if gpuIds is not None else []
        self._native = native
        self.batchesPerGpu = batchesPerGpu  # reuses setter validation

    @property
    def preprocessingThreads(self) -> int:
        """Number of CPU threads for preprocessing. -1 auto-detects."""
        return self._native.preprocessingThreads

    @preprocessingThreads.setter
    def preprocessingThreads(self, value: int) -> None:
        self._native.preprocessingThreads = int(value)

    @property
    def batchSize(self) -> int:
        """Number of conformers per batch. -1 selects an auto-tuned value."""
        return self._native.batchSize

    @batchSize.setter
    def batchSize(self, value: int) -> None:
        self._native.batchSize = int(value)

    @property
    def batchesPerGpu(self) -> int:
        """Batches processed concurrently on each GPU. -1 auto."""
        return self._native.batchesPerGpu

    @batchesPerGpu.setter
    def batchesPerGpu(self, value: int) -> None:
        value = int(value)
        if value != -1 and value <= 0:
            raise ValueError("batchesPerGpu must be greater than 0 or -1 for automatic")
        self._native.batchesPerGpu = value

    @property
    def gpuIds(self) -> List[int]:
        """GPU device IDs to target. Empty list uses all available GPUs."""
        return list(self._native.gpuIds)

    @gpuIds.setter
    def gpuIds(self, value: Iterable[int]) -> None:
        self._native.gpuIds = list(value)

    def _as_native(self):
        """Internal: return the underlying BatchHardwareOptions object."""
        return self._native

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of this object's fields.

        The returned dictionary can be persisted with :func:`json.dump` and
        round-tripped through :meth:`from_dict`.
        """
        return {
            "preprocessingThreads": self.preprocessingThreads,
            "batchSize": self.batchSize,
            "batchesPerGpu": self.batchesPerGpu,
            "gpuIds": list(self.gpuIds),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HardwareOptions":
        """Create a :class:`HardwareOptions` from a dictionary produced by :meth:`to_dict`.

        Unknown keys are rejected so callers catch typos early. Missing keys
        fall back to the constructor defaults.
        """
        known = {"preprocessingThreads", "batchSize", "batchesPerGpu", "gpuIds"}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"Unknown HardwareOptions keys: {sorted(unknown)}")
        return cls(**{key: data[key] for key in known if key in data})


class AsyncGpuResult:
    """Handle to a GPU result.

    Populates the ``__cuda_array_interface__`` attribute which can be consumed by other libraries. Note that
    this result is async, and the data cannot be accessed without a sync, such as ``torch.cuda.synchronize()``.
    """

    def __init__(self, obj, gpu_id: Optional[int] = None):
        """Internal construction of the AsyncGpuResult object.

        Args:
            obj: An object exposing ``__cuda_array_interface__``.
            gpu_id: Optional GPU device id where the underlying buffer lives. If omitted, torch
                infers the device from the CUDA pointer attributes (typically the current device).
        """
        if not hasattr(obj, "__cuda_array_interface__"):
            raise TypeError(f"Object {obj} does not have a __cuda_array_interface__ attribute")
        device = "cuda" if gpu_id is None else f"cuda:{int(gpu_id)}"
        self.arr = torch.as_tensor(obj, device=device)

    @property
    def __cuda_array_interface__(self):
        """Return the CUDA array interface for the underlying data."""
        return self.arr.__cuda_array_interface__

    @property
    def device(self):
        """Return the device of the underlying data."""
        return self.arr.device

    def torch(self):
        """Return the underlying data as a torch tensor. This is an asynchronous operation."""
        return self.arr

    def numpy(self):
        """Return the underlying data as a numpy array. This is a blocking operation."""
        torch.cuda.synchronize()
        return self.arr.cpu().numpy()


class CoordinateOutput(Enum):
    """Selects how conformer-producing APIs return optimized coordinates.

    - ``RDKIT_CONFORMERS``: Optimized coordinates are written back into each input molecule's
      RDKit conformer list and energies (where applicable) are returned as Python lists.
    - ``DEVICE``: coordinates and (where applicable) energies stay on the GPU and are returned
      as a :class:`Device3DResult`. Use this to chain GPU-accelerated workflows without host
      round-trips.
    """

    RDKIT_CONFORMERS = "rdkit"
    DEVICE = "device"


class Dense3DResult(NamedTuple):
    """Dense padded view of a :class:`Device3DResult`.

    All three tensors share the same ``(n_mols, max_confs, max_atoms[, 3])`` batch shape.

    - ``values``: float64, shape ``(n_mols, max_confs, max_atoms, 3)``. Padded entries hold
      the ``pad_value`` passed to :meth:`Device3DResult.dense` (default NaN).
    - ``conf_mask``: bool, shape ``(n_mols, max_confs)``. ``True`` where a real conformer
      exists; ``False`` for pad slots (molecules with fewer conformers than ``max_confs``).
    - ``atom_mask``: bool, shape ``(n_mols, max_confs, max_atoms)``. ``True`` where a real
      atom-vector exists; ``False`` where the atom or conformer slot is padded.
    """

    values: "torch.Tensor"
    conf_mask: "torch.Tensor"
    atom_mask: "torch.Tensor"


class Device3DResult:
    """On-device, flat CSR-style result of a per-atom 3-vector pipeline.

    Used for conformer coordinates produced by ETKDG and MMFF/UFF minimization. All buffers
    live on a single GPU identified by :attr:`gpu_id`. Sizes are linked as follows:

    - :attr:`values` has shape ``(total_atoms, 3)`` (float64). Holds conformer positions.
    - :attr:`atom_starts` has shape ``(n_conformers + 1,)`` (int32). The slice
      ``values[atom_starts[i]:atom_starts[i+1]]`` holds conformer ``i``'s atoms.
    - :attr:`mol_indices` has shape ``(n_conformers,)`` (int32) and maps each conformer to its
      input molecule index.
    - :attr:`conf_indices` has shape ``(n_conformers,)`` (int32) and gives each conformer's
      per-molecule conformer index.
    - :attr:`energies` (MMFF/UFF minimization only) has shape ``(n_conformers,)`` (float64).
    - :attr:`converged` (MMFF/UFF minimization only) has shape ``(n_conformers,)`` (int8; 1 = converged).
    - :attr:`n_mols` is the number of molecules in the original input batch, including those that
      produced zero conformers. This is the authoritative outer-list length for per-molecule views.

    Buffers are exposed as :class:`AsyncGpuResult` to enable zero-copy interoperability with
    PyTorch / CuPy. Synchronize before consuming on the host (e.g. ``torch.cuda.synchronize()``).
    """

    def __init__(
        self,
        values: AsyncGpuResult,
        atom_starts: AsyncGpuResult,
        mol_indices: AsyncGpuResult,
        conf_indices: AsyncGpuResult,
        gpu_id: int,
        n_mols: int,
        energies: Optional[AsyncGpuResult] = None,
        converged: Optional[AsyncGpuResult] = None,
    ) -> None:
        self.values = values
        self.atom_starts = atom_starts
        self.mol_indices = mol_indices
        self.conf_indices = conf_indices
        self.energies = energies
        self.converged = converged
        self.gpu_id = int(gpu_id)
        self.n_mols = int(n_mols)

    @property
    def num_conformers(self) -> int:
        """Total number of conformers represented in this result."""
        return int(self.atom_starts.torch().numel()) - 1

    def per_molecule(self) -> List[List["torch.Tensor"]]:
        """Return a nested list of per-molecule, per-conformer atom-vector views.

        The outer list has length :attr:`n_mols` and is indexed by input molecule index.
        Molecules that produced zero conformers have an empty inner list. The inner list
        contains one ``(n_atoms, 3)`` torch view per conformer. Views share storage with
        :attr:`values` (no copy). Reading the index tensors via ``.tolist()`` implicitly
        synchronizes; reading values still requires the caller to synchronize.
        """
        values = self.values.torch()
        atom_starts = self.atom_starts.torch().tolist()
        mol_indices = self.mol_indices.torch().tolist()
        result: List[List[torch.Tensor]] = [[] for _ in range(self.n_mols)]
        for conf_idx, mol_idx in enumerate(mol_indices):
            start = atom_starts[conf_idx]
            stop = atom_starts[conf_idx + 1]
            result[mol_idx].append(values[start:stop])
        return result

    def dense(self, pad_value: float = float("nan")) -> "Dense3DResult":
        """Materialize a padded dense ``(n_mols, max_confs, max_atoms, 3)`` tensor.

        Axes are padded as follows:

        - **conformer axis**: molecules with fewer than ``max_confs`` conformers (including
          complete failures) receive ``pad_value``-filled conformer slices.
        - **atom axis**: conformers with fewer than ``max_atoms`` atoms are padded with
          ``pad_value``.

        Returns a :class:`Dense3DResult` with ``values``, ``conf_mask``
        ``(n_mols, max_confs)`` and ``atom_mask`` ``(n_mols, max_confs, max_atoms)``.
        Both masks are ``True`` where the data is real, ``False`` where padded.
        Reading the index tensors synchronizes implicitly.
        """
        values = self.values.torch()
        atom_starts = self.atom_starts.torch().to(torch.int64)
        mol_indices = self.mol_indices.torch().to(torch.int64)
        conf_indices = self.conf_indices.torch().to(torch.int64)

        device = values.device
        dtype = values.dtype
        n_conformers = mol_indices.numel()

        if n_conformers == 0:
            dense_vals = torch.full((self.n_mols, 0, 0, 3), pad_value, dtype=dtype, device=device)
            conf_mask = torch.zeros((self.n_mols, 0), dtype=torch.bool, device=device)
            atom_mask = torch.zeros((self.n_mols, 0, 0), dtype=torch.bool, device=device)
            return Dense3DResult(values=dense_vals, conf_mask=conf_mask, atom_mask=atom_mask)

        sizes = atom_starts[1:] - atom_starts[:-1]
        confs_per_mol = torch.bincount(mol_indices, minlength=self.n_mols)
        max_confs = int(confs_per_mol.max().item())
        max_atoms = int(sizes.max().item())

        dense_vals = torch.full(
            (self.n_mols, max_confs, max_atoms, 3),
            pad_value,
            dtype=dtype,
            device=device,
        )
        conf_mask = torch.zeros((self.n_mols, max_confs), dtype=torch.bool, device=device)
        atom_mask = torch.zeros((self.n_mols, max_confs, max_atoms), dtype=torch.bool, device=device)

        conf_mask[mol_indices, conf_indices] = True

        mol_idx_per_atom = mol_indices.repeat_interleave(sizes)
        conf_idx_per_atom = conf_indices.repeat_interleave(sizes)
        total_atoms = values.shape[0]
        atom_within_conf = torch.arange(total_atoms, device=device, dtype=torch.int64) - atom_starts[
            :-1
        ].repeat_interleave(sizes)

        dense_vals[mol_idx_per_atom, conf_idx_per_atom, atom_within_conf, :] = values
        atom_mask[mol_idx_per_atom, conf_idx_per_atom, atom_within_conf] = True

        return Dense3DResult(values=dense_vals, conf_mask=conf_mask, atom_mask=atom_mask)
