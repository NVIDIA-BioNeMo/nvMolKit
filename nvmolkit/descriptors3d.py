# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched GPU calculation of molecular 3D properties.

Supported properties are the three principal moments of inertia and the radius
of gyration. Coordinates may come from RDKit conformers or directly from a
:class:`~nvmolkit.types.Device3DResult` produced by another nvMolKit stage.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch

from nvmolkit import _descriptors3d
from nvmolkit.types import AsyncGpuResult, Device3DResult, PrecisionMode, _resolve_cuda_stream


class Property3D(Enum):
    """3D properties implemented by :func:`Calc3DProperties`.

    ``PMI1``, ``PMI2`` and ``PMI3`` are the principal moments of inertia in ascending order, as
    computed by RDKit's ``CalcPMI1``/``CalcPMI2``/``CalcPMI3``. ``RADIUS_OF_GYRATION`` is the radius
    of gyration, as computed by RDKit's ``CalcRadiusOfGyration``. Wherever a property is accepted,
    its string value (``"PMI1"``, ``"PMI2"``, ``"PMI3"``, ``"RadiusOfGyration"``) may be used instead.
    """

    PMI1 = "PMI1"
    PMI2 = "PMI2"
    PMI3 = "PMI3"
    RADIUS_OF_GYRATION = "RadiusOfGyration"


@dataclass(frozen=True)
class Dense3DPropertyResult:
    """Dense padded view of a :class:`Device3DPropertyResult`.

    Attributes:
        values: One tensor of shape ``(n_mols, max_confs)`` per property, in request order, with the
            dtype of the source result. Padded slots hold the ``pad_value`` passed to
            :meth:`Device3DPropertyResult.dense`.
        conf_mask: bool ``(n_mols, max_confs)``; ``True`` where a real conformer exists.
    """

    values: dict[str, torch.Tensor]
    conf_mask: torch.Tensor


class Device3DPropertyResult(Mapping[str, AsyncGpuResult]):
    """Per-conformer 3D properties on the GPU, labeled by molecule and conformer.

    Behaves as a read-only mapping from property name to an :class:`~nvmolkit.types.AsyncGpuResult`
    of shape ``(n_conformers,)``, in request order. Values are float32 for
    :attr:`~nvmolkit.types.PrecisionMode.SINGLE` and float64 for
    :attr:`~nvmolkit.types.PrecisionMode.FULL`. Keys may be given as names or
    :class:`Property3D` members.

    Row ``i`` of every property belongs to conformer ``conf_indices[i]`` of input molecule
    ``mol_indices[i]``. ``conf_indices`` is the conformer's position within its molecule, not
    its RDKit conformer ID.
    When the properties were calculated from a ``Device3DResult``, rows follow that result and the
    label buffers are shared with it, so rows align with its ``energies`` and ``converged``.

    Attributes:
        mol_indices: int32 ``(n_conformers,)`` input-molecule index of each row.
        conf_indices: int32 ``(n_conformers,)`` per-molecule conformer position of each row.
        gpu_id: GPU holding every buffer.
        n_mols: Number of input molecules, including those without conformers.
    """

    def __init__(
        self,
        properties: dict[str, AsyncGpuResult],
        mol_indices: AsyncGpuResult,
        conf_indices: AsyncGpuResult,
        gpu_id: int,
        n_mols: int,
    ) -> None:
        """Create a result from per-property device buffers and their row labels."""
        self._properties = properties
        self.mol_indices = mol_indices
        self.conf_indices = conf_indices
        self.gpu_id = gpu_id
        self.n_mols = n_mols

    def __getitem__(self, key: Property3D | str) -> AsyncGpuResult:
        """Return the device buffer of one requested property."""
        try:
            return self._properties[_normalize_property(key).value]
        except ValueError as exc:
            raise KeyError(key) from exc

    def __iter__(self) -> Iterator[str]:
        """Iterate over property names in request order."""
        return iter(self._properties)

    def __len__(self) -> int:
        """Return the number of requested properties."""
        return len(self._properties)

    @property
    def n_conformers(self) -> int:
        """Number of rows (conformers) in every property buffer."""
        return self.mol_indices.torch().numel()

    def dense(self, pad_value: float = float("nan")) -> Dense3DPropertyResult:
        """Materialize padded ``(n_mols, max_confs)`` tensors for every property.

        Molecules with fewer than ``max_confs`` conformers (including none) receive ``pad_value``.
        Reading the index tensors synchronizes implicitly.
        """
        mol_indices = self.mol_indices.torch().to(torch.int64)
        conf_indices = self.conf_indices.torch().to(torch.int64)
        device = mol_indices.device
        max_confs = int(torch.bincount(mol_indices, minlength=self.n_mols).max().item()) if mol_indices.numel() else 0

        conf_mask = torch.zeros((self.n_mols, max_confs), dtype=torch.bool, device=device)
        conf_mask[mol_indices, conf_indices] = True
        values = {}
        for name, result in self._properties.items():
            source = result.torch()
            dense_values = torch.full((self.n_mols, max_confs), pad_value, dtype=source.dtype, device=device)
            dense_values[mol_indices, conf_indices] = source
            values[name] = dense_values
        return Dense3DPropertyResult(values=values, conf_mask=conf_mask)


def _normalize_property(property_name: Property3D | str) -> Property3D:
    if isinstance(property_name, Property3D):
        return property_name
    try:
        return Property3D(property_name)
    except (TypeError, ValueError) as exc:
        supported = ", ".join(prop.value for prop in Property3D)
        raise ValueError(f"Unknown 3D property {property_name!r}; supported properties are: {supported}") from exc


def _normalize_properties(
    properties: Property3D | str | Sequence[Property3D | str],
) -> tuple[Property3D, ...]:
    if isinstance(properties, (Property3D, str)):
        normalized = (_normalize_property(properties),)
    else:
        normalized = tuple(_normalize_property(prop) for prop in properties)
    if not normalized:
        raise ValueError("properties must contain at least one 3D property")
    if len(set(normalized)) != len(normalized):
        raise ValueError("properties must not contain duplicates")
    return normalized


def _normalize_molecules(mols) -> list:
    result = [mols] if hasattr(mols, "GetNumAtoms") else list(mols)
    for mol_idx, mol in enumerate(result):
        if mol is None:
            raise ValueError(f"mol at index {mol_idx} must not be None")
    return result


def _require_device_tensor(name: str, tensor: torch.Tensor, dtype: torch.dtype, device: torch.device) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"coordinates.{name} must be CUDA-resident")
    if tensor.device != device:
        raise ValueError(f"coordinates.{name} is on {tensor.device}, expected {device}")
    if tensor.dtype != dtype:
        raise TypeError(f"coordinates.{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"coordinates.{name} must be contiguous")


def _device_coordinate_interfaces(coordinates: Device3DResult, n_mols: int, device: torch.device) -> tuple:
    """Validate a Device3DResult and return the tuple consumed by the native binding."""
    if coordinates.n_mols != n_mols:
        raise ValueError(f"coordinates.n_mols is {coordinates.n_mols}, but {n_mols} molecules were provided")
    if device.index != coordinates.gpu_id:
        raise ValueError(f"coordinates are on cuda:{coordinates.gpu_id}, but stream is on {device}")

    values = coordinates.values.torch()
    atom_starts = coordinates.atom_starts.torch()
    mol_indices = coordinates.mol_indices.torch()
    conf_indices = coordinates.conf_indices.torch()
    _require_device_tensor("values", values, torch.float64, device)
    _require_device_tensor("atom_starts", atom_starts, torch.int32, device)
    _require_device_tensor("mol_indices", mol_indices, torch.int32, device)
    _require_device_tensor("conf_indices", conf_indices, torch.int32, device)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"coordinates.values must have shape (total_atoms, 3), got {tuple(values.shape)}")
    if atom_starts.ndim != 1 or mol_indices.ndim != 1 or atom_starts.numel() != mol_indices.numel() + 1:
        raise ValueError("coordinates.atom_starts must be one entry longer than coordinates.mol_indices")
    if conf_indices.shape != mol_indices.shape:
        raise ValueError("coordinates.conf_indices must have the same shape as coordinates.mol_indices")
    return (
        values.__cuda_array_interface__,
        atom_starts.__cuda_array_interface__,
        mol_indices.__cuda_array_interface__,
        n_mols,
    )


def Calc3DProperties(
    mols,
    properties: Property3D | str | Sequence[Property3D | str],
    *,
    coordinates: Device3DResult | None = None,
    useAtomicMasses: bool = True,
    precision: PrecisionMode = PrecisionMode.SINGLE,
    stream: torch.cuda.Stream | None = None,
) -> Device3DPropertyResult:
    """Calculate selected 3D properties for every conformer in a molecule batch.

    Args:
        mols: One RDKit molecule or an iterable of molecules. Molecules provide
            atom identity even when coordinates are supplied separately.
        properties: Ordered property selection. Supported values are ``PMI1``,
            ``PMI2``, ``PMI3``, and ``RadiusOfGyration``.
        coordinates: Optional device-resident coordinates from an nvMolKit 3D
            operation, read in place. When omitted, coordinates are taken from
            each molecule's RDKit conformers. Rows whose ``mol_indices`` entry
            is out of range, whose ``atom_starts`` range falls outside
            ``values``, or whose atom count differs from their molecule's
            produce NaN rather than an error, so no host synchronization is
            needed.
        useAtomicMasses: Match RDKit's mass-weighted default. ``False`` gives
            every atom unit weight.
        precision: ``PrecisionMode.SINGLE`` (default) computes and returns
            float32 values; ``PrecisionMode.FULL`` uses float64 throughout and
            matches RDKit to near double-precision rounding.
        stream: CUDA stream used for transfers and calculation. Defaults to the
            coordinate device's current stream, or the current CUDA stream.

    Returns:
        A :class:`Device3DPropertyResult` mapping each requested property name,
        in request order, to a device vector with one row per conformer, plus
        the molecule and conformer labels of every row. Without ``coordinates``,
        rows follow input-molecule order, then RDKit conformer order.
    """
    normalized_mols = _normalize_molecules(mols)
    normalized_properties = _normalize_properties(properties)
    if coordinates is not None and not isinstance(coordinates, Device3DResult):
        raise TypeError(f"coordinates must be a Device3DResult or None, got {type(coordinates).__name__}")
    input_values = () if coordinates is None else (coordinates.values,)
    active_stream = _resolve_cuda_stream(stream, *input_values)
    coordinate_interfaces = (
        None
        if coordinates is None
        else _device_coordinate_interfaces(coordinates, len(normalized_mols), active_stream.device)
    )

    raw_results, raw_mol_indices, raw_conf_indices = _descriptors3d.Calc3DProperties(
        normalized_mols,
        [prop.value for prop in normalized_properties],
        useAtomicMasses,
        coordinate_interfaces,
        precision,
        active_stream.cuda_stream,
    )
    gpu_id = active_stream.device.index
    properties_by_name = {}
    for name, raw in raw_results.items():
        result = AsyncGpuResult(raw, gpu_id=gpu_id)
        if coordinates is not None:
            # The kernel reads caller-owned device coordinates asynchronously on active_stream.
            result._input_owners = (coordinates,)
        properties_by_name[name] = result

    if coordinates is None:
        mol_indices = AsyncGpuResult(raw_mol_indices, gpu_id=gpu_id)
        conf_indices = AsyncGpuResult(raw_conf_indices, gpu_id=gpu_id)
    else:
        mol_indices = coordinates.mol_indices
        conf_indices = coordinates.conf_indices
    return Device3DPropertyResult(
        properties_by_name,
        mol_indices=mol_indices,
        conf_indices=conf_indices,
        gpu_id=gpu_id,
        n_mols=len(normalized_mols),
    )
