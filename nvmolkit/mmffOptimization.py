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

"""GPU-accelerated MMFF optimization for molecular conformers.

This module provides GPU-accelerated implementations of MMFF (Molecular Mechanics Force Field)
optimization for multiple molecules and conformers using CUDA and OpenMP.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, overload

from rdkit.Chem import AllChem

if TYPE_CHECKING:
    from rdkit.Chem import Mol
    from rdkit.ForceField.rdForceField import MMFFMolProperties

from nvmolkit import _mmffOptimization
from nvmolkit._arrayHelpers import *  # noqa: F403  # registers PyArray for DEVICE-mode returns
from nvmolkit._mmff_bridge import default_rdkit_mmff_properties, make_internal_mmff_properties
from nvmolkit.types import CoordinateOutput, Device3DResult, HardwareOptions


@overload
def MMFFOptimizeMoleculesConfs(
    molecules: list["Mol"],
    maxIters: int = 200,
    properties: "MMFFMolProperties | Sequence[MMFFMolProperties | None] | None" = None,
    nonBondedThreshold: float | Sequence[float] = 100.0,
    ignoreInterfragInteractions: bool | Sequence[bool] = True,
    hardwareOptions: HardwareOptions | None = None,
    output: Literal[CoordinateOutput.RDKIT_CONFORMERS] = CoordinateOutput.RDKIT_CONFORMERS,
    targetGpu: int = -1,
) -> list[list[float]]: ...
@overload
def MMFFOptimizeMoleculesConfs(
    molecules: list["Mol"],
    maxIters: int = 200,
    properties: "MMFFMolProperties | Sequence[MMFFMolProperties | None] | None" = None,
    nonBondedThreshold: float | Sequence[float] = 100.0,
    ignoreInterfragInteractions: bool | Sequence[bool] = True,
    hardwareOptions: HardwareOptions | None = None,
    *,
    output: Literal[CoordinateOutput.DEVICE],
    targetGpu: int = -1,
) -> Device3DResult: ...
def MMFFOptimizeMoleculesConfs(
    molecules: list["Mol"],
    maxIters: int = 200,
    properties: "MMFFMolProperties | Sequence[MMFFMolProperties | None] | None" = None,
    nonBondedThreshold: float | Sequence[float] = 100.0,
    ignoreInterfragInteractions: bool | Sequence[bool] = True,
    hardwareOptions: HardwareOptions | None = None,
    output: CoordinateOutput = CoordinateOutput.RDKIT_CONFORMERS,
    targetGpu: int = -1,
):
    """Optimize conformers for multiple molecules using MMFF force field with BFGS minimization.

    This function performs GPU-accelerated MMFF optimization on multiple molecules with
    multiple conformers each. It uses CUDA for GPU acceleration and OpenMP for CPU
    parallelization to achieve high performance.

    Args:
        molecules: List of RDKit molecules to optimize. Each molecule should have
                  conformers already generated.
        maxIters: Maximum number of BFGS optimization iterations (default: 200)
        properties: RDKit ``MMFFMolProperties`` object, a per-molecule sequence
            of those objects, or ``None`` to use default MMFF94 settings.
        nonBondedThreshold: Radius threshold used to exclude long-range
            non-bonded interactions, either as a scalar or per-molecule sequence.
        ignoreInterfragInteractions: If ``True``, omit non-bonded terms between
            fragments. May also be provided as a per-molecule sequence.
        hardwareOptions: Configures CPU and GPU batching, threading, and device selection. Will attempt to use reasonable defaults if not set.
        output: ``RDKIT_CONFORMERS`` (default) writes optimized coordinates back into each input
            molecule's RDKit conformers and returns nested host-side energy lists. ``DEVICE``
            keeps optimized coordinates and energies on GPU and returns a :class:`Device3DResult`.
        targetGpu: In DEVICE mode, the GPU to consolidate the result onto. ``-1`` selects the
            first configured execution GPU.

    Returns:
        For ``RDKIT_CONFORMERS``: list of lists of energies, where each inner list contains the
        optimized energies for all conformers of the corresponding molecule. The order matches
        the input molecule order and conformer iteration order.
        For ``DEVICE``: a :class:`Device3DResult` whose ``values`` field carries optimized
        coordinates ``(total_atoms, 3)``, plus ``energies``, ``converged``, and CSR indices.

    Raises:
        ValueError: If any molecules in the input list are None or lack MMFF atom types.
            ``e.args[0]`` is a summary message, ``e.args[1]`` is a dict
            with keys ``"none"`` (indices of None molecules) and ``"no_params"``
            (indices of molecules lacking MMFF atom types). Example::

                try:
                    MMFFOptimizeMoleculesConfs(mols, ...)
                except ValueError as e:
                    failed = e.args[1]
                    none_idx = failed["none"]
                    no_params_idx = failed["no_params"]
        RuntimeError: If CUDA operations fail or optimization encounters errors

    Example:
        >>> from rdkit import Chem
        >>> from rdkit.Chem import rdDistGeom
        >>> from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
        >>> from nvmolkit.types import HardwareOptions
        >>>
        >>> # Load molecules and generate conformers
        >>> mol1 = Chem.AddHs(Chem.MolFromSmiles('CCO'))
        >>> mol2 = Chem.AddHs(Chem.MolFromSmiles('CCC'))
        >>> rdDistGeom.EmbedMultipleConfs(mol1, numConfs=5)
        >>> rdDistGeom.EmbedMultipleConfs(mol2, numConfs=3)
        >>>
        >>> # Set custom runtime performance options (optional)
        >>> hardware_options = HardwareOptions(batchSize=200, batchesPerGpu=4)
        >>> energies = MMFFOptimizeMoleculesConfs(
        ...     [mol1, mol2],
        ...     maxIters=500,
        ...     hardwareOptions=hardware_options
        ... )
        >>>
        >>> # energies[0] contains 5 energies for mol1's conformers
        >>> # energies[1] contains 3 energies for mol2's conformers

    Note:
        - Input molecules are modified in-place with optimized conformer coordinates
    """
    if not molecules:
        if output == CoordinateOutput.DEVICE:
            raise ValueError("MMFFOptimizeMoleculesConfs(output=DEVICE) requires at least one molecule")
        return []

    none_indices = []
    no_params_indices = []
    for i, mol in enumerate(molecules):
        if mol is None:
            none_indices.append(i)
        elif not AllChem.MMFFHasAllMoleculeParams(mol):
            no_params_indices.append(i)

    if none_indices or no_params_indices:
        parts = []
        if none_indices:
            parts.append(f"None at indices {none_indices}")
        if no_params_indices:
            parts.append(f"lacking MMFF atom types at indices {no_params_indices}")
        raise ValueError(
            "; ".join(parts),
            {"none": none_indices, "no_params": no_params_indices},
        )

    def _normalize_scalar_or_list(value, name: str):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != len(molecules):
                raise ValueError(f"Expected {len(molecules)} values for {name}, got {len(value)}")
            return list(value)
        return [value for _ in molecules]

    def _normalize_properties(value):
        if value is None:
            return [default_rdkit_mmff_properties(mol) for mol in molecules]
        if isinstance(value, Sequence) and not hasattr(value, "SetMMFFVariant"):
            if len(value) != len(molecules):
                raise ValueError(f"Expected {len(molecules)} MMFFMolProperties objects, got {len(value)}")
            return [
                default_rdkit_mmff_properties(mol) if props is None else props for mol, props in zip(molecules, value)
            ]
        return [value for _ in molecules]

    # Call the C++ implementation
    if hardwareOptions is None:
        hardwareOptions = HardwareOptions()
    native_options = hardwareOptions._as_native()
    properties_list = _normalize_properties(properties)
    thresholds = _normalize_scalar_or_list(nonBondedThreshold, "nonBondedThreshold")
    interfrag_flags = _normalize_scalar_or_list(ignoreInterfragInteractions, "ignoreInterfragInteractions")
    native_properties = [
        make_internal_mmff_properties(
            props,
            non_bonded_threshold=float(threshold),
            ignore_interfrag_interactions=bool(ignore_interfrag),
        )
        for props, threshold, ignore_interfrag in zip(properties_list, thresholds, interfrag_flags)
    ]
    if output == CoordinateOutput.DEVICE:
        return _mmffOptimization.MMFFOptimizeMoleculesConfsDevice(
            molecules, maxIters, native_properties, native_options, targetGpu
        )
    return _mmffOptimization.MMFFOptimizeMoleculesConfs(molecules, maxIters, native_properties, native_options)
