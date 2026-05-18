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

"""Batched forcefield wrappers with energy, gradient, and minimization support.

This module provides :class:`MMFFBatchedForcefield` and
:class:`UFFBatchedForcefield` for GPU-accelerated evaluation and BFGS
minimization of multiple molecules with multiple conformers each.

All conformers of each molecule are evaluated as a batch.  Properties and
constraints are specified per-molecule and apply to every conformer of that
molecule.  Results are returned nested as ``list[list[...]]`` where the outer
list is per-molecule and the inner list is per-conformer.

Typical usage:

.. code-block:: python

   from rdkit import Chem
   from rdkit.Chem import rdDistGeom
   from nvmolkit.batchedForcefield import MMFFBatchedForcefield

   mol_a = Chem.AddHs(Chem.MolFromSmiles("CCO"))
   mol_b = Chem.AddHs(Chem.MolFromSmiles("CCCC"))

   rdDistGeom.EmbedMultipleConfs(mol_a, numConfs=5)
   rdDistGeom.EmbedMultipleConfs(mol_b, numConfs=3)

   ff = MMFFBatchedForcefield([mol_a, mol_b])
   energies = ff.compute_energy()      # [[5 floats], [3 floats]]
   gradients = ff.compute_gradients()  # [[5 grad-lists], [3 grad-lists]]

   ff[0].add_position_constraint(0, 0.1, 50.0)
   opt_energies, converged = ff.minimize()  # ([[5 floats], [3 floats]], [[5 bools], [3 bools]])

The wrapper also supports per-entry settings such as custom property objects,
non-bonded thresholds, and interfragment interaction handling.  These options
may be passed either as scalars, which are broadcast to the whole batch, or
as per-molecule sequences:

.. code-block:: python

   ff = MMFFBatchedForcefield(
       [mol_a, mol_b],
       properties=[props_a, props_b],
       nonBondedThreshold=[100.0, 25.0],
       ignoreInterfragInteractions=[True, False],
   )

Constraint terms are added through per-molecule element views:

.. code-block:: python

   ff = MMFFBatchedForcefield([mol_a, mol_b])

   ff[0].add_distance_constraint(0, 2, True, 0.2, 0.5, 20.0)
   ff[1].add_torsion_constraint(0, 1, 2, 3, True, 10.0, 20.0, 8.0)

   energies, converged = ff.minimize()

.. note::
   Calling :meth:`~MMFFBatchedForcefield.compute_energy` or
   :meth:`~MMFFBatchedForcefield.compute_gradients` individually from Python
   incurs per-call overhead that can dominate the actual GPU work. This wrapper
   is most useful for correctness checks and Python workflows that need batched
   forcefield energies or gradients. For high-throughput optimization, prefer
   the dedicated minimizer APIs.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

from nvmolkit import _batchedForcefield  # type: ignore
from nvmolkit._arrayHelpers import *  # noqa: F403  # registers PyArray for DEVICE-mode returns
from nvmolkit._mmff_bridge import default_rdkit_mmff_properties, make_internal_mmff_properties
from nvmolkit.types import CoordinateOutput, Device3DResult, HardwareOptions

if TYPE_CHECKING:
    from rdkit.Chem import Mol
    from rdkit.ForceField.rdForceField import MMFFMolProperties as RDKitMMFFMolProperties


@dataclass
class _DistanceConstraint:
    idx1: int
    idx2: int
    relative: bool
    min_len: float
    max_len: float
    force_constant: float


@dataclass
class _PositionConstraint:
    idx: int
    max_displ: float
    force_constant: float


@dataclass
class _AngleConstraint:
    idx1: int
    idx2: int
    idx3: int
    relative: bool
    min_angle_deg: float
    max_angle_deg: float
    force_constant: float


@dataclass
class _TorsionConstraint:
    idx1: int
    idx2: int
    idx3: int
    idx4: int
    relative: bool
    min_dihedral_deg: float
    max_dihedral_deg: float
    force_constant: float


def _serialize_distance_constraints(constraints: list[_DistanceConstraint]):
    return [(c.idx1, c.idx2, c.relative, c.min_len, c.max_len, c.force_constant) for c in constraints]


def _serialize_position_constraints(constraints: list[_PositionConstraint]):
    return [(c.idx, c.max_displ, c.force_constant) for c in constraints]


def _serialize_angle_constraints(constraints: list[_AngleConstraint]):
    return [
        (c.idx1, c.idx2, c.idx3, c.relative, c.min_angle_deg, c.max_angle_deg, c.force_constant) for c in constraints
    ]


def _serialize_torsion_constraints(constraints: list[_TorsionConstraint]):
    return [
        (c.idx1, c.idx2, c.idx3, c.idx4, c.relative, c.min_dihedral_deg, c.max_dihedral_deg, c.force_constant)
        for c in constraints
    ]


def _serialize_all_constraints(
    distance: list[list[_DistanceConstraint]],
    position: list[list[_PositionConstraint]],
    angle: list[list[_AngleConstraint]],
    torsion: list[list[_TorsionConstraint]],
):
    return (
        [_serialize_distance_constraints(c) for c in distance],
        [_serialize_position_constraints(c) for c in position],
        [_serialize_angle_constraints(c) for c in angle],
        [_serialize_torsion_constraints(c) for c in torsion],
    )


class _BatchElementBase:
    """Per-molecule view for adding constraints to a batched forcefield."""

    def __init__(self, parent, idx: int):
        self._parent = parent
        self._idx = idx

    @property
    def num_atoms(self) -> int:
        """Return the number of atoms in this molecule."""
        return self._parent._molecules[self._idx].GetNumAtoms()

    def add_distance_constraint(
        self,
        idx1: int,
        idx2: int,
        relative: bool,
        min_len: float,
        max_len: float,
        force_constant: float,
    ) -> None:
        """Add a distance constraint to this molecule.

        Args:
            idx1: Index of the first constrained atom.
            idx2: Index of the second constrained atom.
            relative: If ``True``, interpret ``min_len`` and ``max_len`` as
                offsets from the current distance.  If ``False``, interpret
                them as absolute distances in Angstroms.
            min_len: Lower bound of the allowed distance range.
            max_len: Upper bound of the allowed distance range.
            force_constant: Constraint force constant.
        """
        self._parent._validate_atom_indices(self._idx, idx1, idx2)
        self._parent._distance_constraints[self._idx].append(
            _DistanceConstraint(idx1, idx2, relative, min_len, max_len, force_constant)
        )
        self._parent._dirty = True

    def add_position_constraint(
        self,
        idx: int,
        max_displ: float,
        force_constant: float,
    ) -> None:
        """Add a position constraint to one atom in this molecule.

        Args:
            idx: Index of the constrained atom.
            max_displ: Maximum displacement allowed from the current atom
                position before the restraint contributes energy.
            force_constant: Constraint force constant.
        """
        self._parent._validate_atom_indices(self._idx, idx)
        self._parent._position_constraints[self._idx].append(_PositionConstraint(idx, max_displ, force_constant))
        self._parent._dirty = True

    def add_angle_constraint(
        self,
        idx1: int,
        idx2: int,
        idx3: int,
        relative: bool,
        min_angle_deg: float,
        max_angle_deg: float,
        force_constant: float,
    ) -> None:
        """Add an angle constraint to this molecule.

        Args:
            idx1: Index of the first atom in the angle.
            idx2: Index of the central atom.
            idx3: Index of the third atom in the angle.
            relative: If ``True``, interpret angle bounds relative to the
                current angle.  If ``False``, interpret them as absolute
                angles in degrees.
            min_angle_deg: Lower bound of the allowed angle range in degrees.
            max_angle_deg: Upper bound of the allowed angle range in degrees.
            force_constant: Constraint force constant.
        """
        self._parent._validate_atom_indices(self._idx, idx1, idx2, idx3)
        self._parent._angle_constraints[self._idx].append(
            _AngleConstraint(idx1, idx2, idx3, relative, min_angle_deg, max_angle_deg, force_constant)
        )
        self._parent._dirty = True

    def add_torsion_constraint(
        self,
        idx1: int,
        idx2: int,
        idx3: int,
        idx4: int,
        relative: bool,
        min_dihedral_deg: float,
        max_dihedral_deg: float,
        force_constant: float,
    ) -> None:
        """Add a torsion constraint to this molecule.

        Args:
            idx1: Index of the first atom in the torsion.
            idx2: Index of the second atom in the torsion.
            idx3: Index of the third atom in the torsion.
            idx4: Index of the fourth atom in the torsion.
            relative: If ``True``, interpret dihedral bounds relative to the
                current dihedral angle.  If ``False``, interpret them as
                absolute dihedral bounds in degrees.
            min_dihedral_deg: Lower bound of the allowed dihedral range in
                degrees.
            max_dihedral_deg: Upper bound of the allowed dihedral range in
                degrees.
            force_constant: Constraint force constant.
        """
        self._parent._validate_atom_indices(self._idx, idx1, idx2, idx3, idx4)
        self._parent._torsion_constraints[self._idx].append(
            _TorsionConstraint(idx1, idx2, idx3, idx4, relative, min_dihedral_deg, max_dihedral_deg, force_constant)
        )
        self._parent._dirty = True


class MMFFBatchElement(_BatchElementBase):
    """Per-molecule view for configuring one molecule in an MMFF batch.

    Retrieve an element with ``ff[i]`` and use it to inspect the number of
    atoms or to add constraints for that molecule.  Constraints apply to all
    conformers of the molecule.

    Example:

    .. code-block:: python

       ff = MMFFBatchedForcefield([mol_a, mol_b])
       ff[0].add_position_constraint(0, 0.1, 50.0)
       ff[1].add_distance_constraint(1, 4, False, 2.0, 2.2, 25.0)
       energies, converged = ff.minimize()
    """


class UFFBatchElement(_BatchElementBase):
    """Per-molecule view for configuring one molecule in a UFF batch.

    Retrieve an element with ``ff[i]`` and use it to add constraints for
    that molecule.  Constraints apply to all conformers of the molecule.

    Example:

    .. code-block:: python

       ff = UFFBatchedForcefield([mol_a, mol_b])
       ff[0].add_position_constraint(0, 0.1, 50.0)
       energies, converged = ff.minimize()
    """


def _normalize_scalar_or_list(value, n: int, name: str):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != n:
            raise ValueError(f"Expected {n} values for {name}, got {len(value)}")
        return list(value)
    return [value for _ in range(n)]


class _BatchedForcefieldBase:
    """Shared implementation for MMFF and UFF batched forcefield wrappers."""

    _element_class: type[_BatchElementBase] = _BatchElementBase

    def _init_common(
        self,
        molecules: list["Mol"],
        ignoreInterfragInteractions,
        hardwareOptions: HardwareOptions | None,
    ) -> None:
        self._molecules = molecules
        self._ignore_interfrag_interactions = _normalize_scalar_or_list(
            ignoreInterfragInteractions, len(molecules), "ignoreInterfragInteractions"
        )
        self._hardware_options = hardwareOptions if hardwareOptions is not None else HardwareOptions()
        self._distance_constraints: list[list[_DistanceConstraint]] = [[] for _ in molecules]
        self._position_constraints: list[list[_PositionConstraint]] = [[] for _ in molecules]
        self._angle_constraints: list[list[_AngleConstraint]] = [[] for _ in molecules]
        self._torsion_constraints: list[list[_TorsionConstraint]] = [[] for _ in molecules]
        self._native_ff = None
        self._dirty = True
        self.num_molecules = len(molecules)
        self.data_dim = 3

    def __len__(self) -> int:
        """Return the number of molecules in the batch."""
        return len(self._molecules)

    def __getitem__(self, idx: int):
        """Return a per-molecule view for adding constraints."""
        if idx < 0 or idx >= len(self._molecules):
            raise IndexError(f"Batch element index {idx} out of range")
        return self._element_class(self, idx)

    def _validate_atom_indices(self, batch_idx: int, *indices: int) -> None:
        num_atoms = self._molecules[batch_idx].GetNumAtoms()
        for idx in indices:
            if idx < 0 or idx >= num_atoms:
                raise IndexError(f"Atom index {idx} out of range for molecule {batch_idx} with {num_atoms} atoms")

    def _serialized_constraints(self):
        return _serialize_all_constraints(
            self._distance_constraints,
            self._position_constraints,
            self._angle_constraints,
            self._torsion_constraints,
        )

    def _build_native(self):
        raise NotImplementedError

    def _build(self) -> None:
        """Build the native forcefield for the current batch settings."""
        if not self._molecules:
            self._native_ff = None
            self._dirty = False
            return
        self._native_ff = self._build_native()
        self._dirty = False

    def _ensure_built(self) -> None:
        if self._dirty or self._native_ff is None:
            self._build()

    def rebuild(self) -> None:
        """Rebuild the forcefield after changing constraints or settings."""
        self._build()

    def compute_energy(self) -> list[list[float]]:
        """Return forcefield energies for all conformers of all molecules.

        Returns:
            ``result[mol_idx][conf_idx]`` -- one energy per conformer.
        """
        if not self._molecules:
            return []
        self._ensure_built()
        return self._native_ff.computeEnergy()

    def compute_gradients(self) -> list[list[list[float]]]:
        """Return forcefield gradients for all conformers of all molecules.

        Returns:
            ``result[mol_idx][conf_idx]`` -- one flattened ``[x0, y0, z0, ...]``
            gradient vector per conformer.
        """
        if not self._molecules:
            return []
        self._ensure_built()
        return self._native_ff.computeGradients()

    def _minimize(
        self,
        maxIters: int,
        forceTol: float,
        output: CoordinateOutput = CoordinateOutput.RDKIT_CONFORMERS,
        target_gpu: int | None = None,
    ):
        if not self._molecules:
            if output == CoordinateOutput.DEVICE:
                raise ValueError("minimize(output=DEVICE) requires at least one molecule")
            return [], []
        self._ensure_built()
        if output == CoordinateOutput.DEVICE:
            return self._native_ff.minimizeDevice(maxIters, forceTol, -1 if target_gpu is None else int(target_gpu))
        energies, converged = self._native_ff.minimize(maxIters, forceTol)
        return energies, converged


class MMFFBatchedForcefield(_BatchedForcefieldBase):
    """Evaluate MMFF energies and gradients, or run BFGS minimization, for a
    batch of molecules with all their conformers.

    Properties and constraints are per-molecule and are shared across all
    conformers of that molecule.  Results are nested as
    ``list[list[...]]`` — outer per-molecule, inner per-conformer.

    Examples:

    .. code-block:: python

       ff = MMFFBatchedForcefield([mol_a, mol_b])
       energies = ff.compute_energy()  # [[...], [...]]

    .. code-block:: python

       ff = MMFFBatchedForcefield(
           [mol_a, mol_b],
           properties=[props_a, None],
           nonBondedThreshold=[100.0, 20.0],
       )
       ff[0].add_distance_constraint(0, 4, False, 1.8, 2.2, 50.0)
       opt_energies, converged = ff.minimize()
    """

    _element_class = MMFFBatchElement

    def __init__(
        self,
        molecules: list["Mol"],
        properties: "RDKitMMFFMolProperties | Sequence[RDKitMMFFMolProperties | None] | None" = None,
        nonBondedThreshold: float | Sequence[float] = 100.0,
        ignoreInterfragInteractions: bool | Sequence[bool] = True,
        hardwareOptions: HardwareOptions | None = None,
    ):
        """Create a batched MMFF forcefield wrapper.

        All conformers of each molecule are included in the batch
        automatically.

        Args:
            molecules: RDKit molecules to evaluate.
            properties: RDKit ``MMFFMolProperties`` object, a per-molecule
                list of those objects, or ``None`` to use default MMFF94
                settings.  A single object is broadcast to all molecules.
            nonBondedThreshold: Non-bonded cutoff distance or per-molecule
                values.
            ignoreInterfragInteractions: Whether to omit interfragment
                non-bonded interactions, as a scalar or per-molecule list.
            hardwareOptions: GPU device and batching configuration.  Uses
                reasonable defaults when ``None``.
        """
        self._init_common(molecules, ignoreInterfragInteractions, hardwareOptions)
        self._properties = self._normalize_properties(properties)
        self._non_bonded_thresholds = _normalize_scalar_or_list(
            nonBondedThreshold, len(molecules), "nonBondedThreshold"
        )

    def __getitem__(self, idx: int) -> MMFFBatchElement:
        return super().__getitem__(idx)

    def _normalize_properties(
        self, properties: "RDKitMMFFMolProperties | Sequence[RDKitMMFFMolProperties | None] | None"
    ) -> list["RDKitMMFFMolProperties | None"]:
        if properties is None:
            return [None for _ in self._molecules]
        if isinstance(properties, Sequence) and not hasattr(properties, "SetMMFFVariant"):
            if len(properties) != len(self._molecules):
                raise ValueError(f"Expected {len(self._molecules)} MMFFMolProperties objects, got {len(properties)}")
            return list(properties)
        return [properties for _ in self._molecules]

    def _copy_mmff_properties(
        self, mol: "Mol", properties, non_bonded_threshold: float, ignore_interfrag_interactions: bool
    ):
        """Convert RDKit MMFF properties into the native representation."""
        source = default_rdkit_mmff_properties(mol) if properties is None else properties
        return make_internal_mmff_properties(
            source,
            non_bonded_threshold=float(non_bonded_threshold),
            ignore_interfrag_interactions=bool(ignore_interfrag_interactions),
        )

    def _build_native(self):
        native_properties = [
            self._copy_mmff_properties(mol, props, threshold, ignore)
            for mol, props, threshold, ignore in zip(
                self._molecules,
                self._properties,
                self._non_bonded_thresholds,
                self._ignore_interfrag_interactions,
            )
        ]
        dist, pos, ang, tor = self._serialized_constraints()
        return _batchedForcefield.NativeMMFFBatchedForcefield(
            self._molecules,
            native_properties,
            dist,
            pos,
            ang,
            tor,
            self._hardware_options._as_native(),
        )

    @overload
    def minimize(
        self,
        maxIters: int = 200,
        forceTol: float = 1e-4,
        output: Literal[CoordinateOutput.RDKIT_CONFORMERS] = CoordinateOutput.RDKIT_CONFORMERS,
        target_gpu: int | None = None,
    ) -> tuple[list[list[float]], list[list[bool]]]: ...
    @overload
    def minimize(
        self,
        maxIters: int = 200,
        forceTol: float = 1e-4,
        *,
        output: Literal[CoordinateOutput.DEVICE],
        target_gpu: int | None = None,
    ) -> Device3DResult: ...
    def minimize(
        self,
        maxIters: int = 200,
        forceTol: float = 1e-4,
        output: CoordinateOutput = CoordinateOutput.RDKIT_CONFORMERS,
        target_gpu: int | None = None,
    ):
        """Run BFGS minimization on all conformers of all molecules.

        In ``RDKIT_CONFORMERS`` mode, optimized coordinates are written back
        into the RDKit conformers in-place. In ``DEVICE`` mode, optimized
        coordinates and energies stay on the GPU, the wrapper's persistent
        on-device positions buffer is updated in-place (no host roundtrip),
        and a :class:`Device3DResult` is returned.

        Args:
            maxIters: Maximum number of BFGS iterations.
            forceTol: Gradient convergence tolerance.
            output: ``RDKIT_CONFORMERS`` (default) or ``DEVICE``.
            target_gpu: In DEVICE mode, the GPU to consolidate the result on.
                ``None`` (the default) selects the wrapper's own GPU. The
                wrapper is single-GPU - the only supported value is the
                wrapper's GPU id; passing a different GPU raises
                ``invalid_argument``. For cross-GPU consolidation use the
                standalone ``MMFFOptimizeMoleculesConfs(output=DEVICE,
                targetGpu=...)`` API.

        Returns:
            For RDKit mode: ``(energies, converged)`` nested host lists.
            For DEVICE mode: a :class:`Device3DResult` whose ``values`` field
            holds the optimized coordinates, ``energies`` holds final energies,
            and ``converged`` holds per-conformer convergence flags.
        """
        return self._minimize(maxIters, forceTol, output=output, target_gpu=target_gpu)


class UFFBatchedForcefield(_BatchedForcefieldBase):
    """Evaluate UFF energies and gradients, or run BFGS minimization, for a
    batch of molecules with all their conformers.

    Constraints are per-molecule and are shared across all conformers of
    that molecule.  Results are nested as ``list[list[...]]`` — outer
    per-molecule, inner per-conformer.

    Examples:

    .. code-block:: python

       ff = UFFBatchedForcefield([mol_a, mol_b])
       energies = ff.compute_energy()  # [[...], [...]]

    .. code-block:: python

       ff = UFFBatchedForcefield([mol_a, mol_b])
       ff[0].add_position_constraint(0, 0.1, 50.0)
       opt_energies, converged = ff.minimize()
    """

    _element_class = UFFBatchElement

    def __init__(
        self,
        molecules: list["Mol"],
        vdwThreshold: float | Sequence[float] = 10.0,
        ignoreInterfragInteractions: bool | Sequence[bool] = True,
        hardwareOptions: HardwareOptions | None = None,
    ):
        """Create a batched UFF forcefield wrapper.

        All conformers of each molecule are included in the batch
        automatically.

        Args:
            molecules: RDKit molecules to evaluate.
            vdwThreshold: Van der Waals threshold, scalar or per-molecule.
            ignoreInterfragInteractions: Whether to omit interfragment
                non-bonded interactions, as a scalar or per-molecule list.
            hardwareOptions: GPU device and batching configuration.  Uses
                reasonable defaults when ``None``.
        """
        self._init_common(molecules, ignoreInterfragInteractions, hardwareOptions)
        self._vdw_thresholds = _normalize_scalar_or_list(vdwThreshold, len(molecules), "vdwThreshold")

    def __getitem__(self, idx: int) -> UFFBatchElement:
        return super().__getitem__(idx)

    def _build_native(self):
        dist, pos, ang, tor = self._serialized_constraints()
        return _batchedForcefield.NativeUFFBatchedForcefield(
            self._molecules,
            [float(v) for v in self._vdw_thresholds],
            [bool(v) for v in self._ignore_interfrag_interactions],
            dist,
            pos,
            ang,
            tor,
            self._hardware_options._as_native(),
        )

    @overload
    def minimize(
        self,
        maxIters: int = 1000,
        forceTol: float = 1e-4,
        output: Literal[CoordinateOutput.RDKIT_CONFORMERS] = CoordinateOutput.RDKIT_CONFORMERS,
        target_gpu: int | None = None,
    ) -> tuple[list[list[float]], list[list[bool]]]: ...
    @overload
    def minimize(
        self,
        maxIters: int = 1000,
        forceTol: float = 1e-4,
        *,
        output: Literal[CoordinateOutput.DEVICE],
        target_gpu: int | None = None,
    ) -> Device3DResult: ...
    def minimize(
        self,
        maxIters: int = 1000,
        forceTol: float = 1e-4,
        output: CoordinateOutput = CoordinateOutput.RDKIT_CONFORMERS,
        target_gpu: int | None = None,
    ):
        """Run BFGS minimization on all conformers of all molecules.

        In ``RDKIT_CONFORMERS`` mode, optimized coordinates are written back
        into the RDKit conformers in-place. In ``DEVICE`` mode, optimized
        coordinates and energies stay on the GPU, the wrapper's persistent
        on-device positions buffer is updated in-place (no host roundtrip),
        and a :class:`Device3DResult` is returned.

        Args:
            maxIters: Maximum number of BFGS iterations.
            forceTol: Gradient convergence tolerance.
            output: ``RDKIT_CONFORMERS`` (default) or ``DEVICE``.
            target_gpu: In DEVICE mode, the GPU to consolidate the result on.
                ``None`` (the default) selects the wrapper's own GPU. The
                wrapper is single-GPU - the only supported value is the
                wrapper's GPU id; passing a different GPU raises
                ``invalid_argument``. For cross-GPU consolidation use the
                standalone ``UFFOptimizeMoleculesConfs(output=DEVICE,
                targetGpu=...)`` API.

        Returns:
            For RDKit mode: ``(energies, converged)`` nested host lists.
            For DEVICE mode: a :class:`Device3DResult` whose ``values`` field
            holds the optimized coordinates, ``energies`` holds final energies,
            and ``converged`` holds per-conformer convergence flags.
        """
        return self._minimize(maxIters, forceTol, output=output, target_gpu=target_gpu)
