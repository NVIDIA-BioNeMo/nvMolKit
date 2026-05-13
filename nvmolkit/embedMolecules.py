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

"""GPU-accelerated ETKDG conformer generation for multiple molecules.

This module provides GPU-accelerated implementations of ETKDG (Experimental-Torsion-Knowledge Distance-Geometry) conformer generation for multiple molecules using CUDA and OpenMP.
"""

from typing import TYPE_CHECKING, Literal, Optional, overload

if TYPE_CHECKING:
    from rdkit.Chem import Mol
    from rdkit.Chem.rdDistGeom import EmbedParameters

__all__ = ["EmbedMolecules"]

from nvmolkit import _embedMolecules  # type: ignore
from nvmolkit._arrayHelpers import *  # noqa: F403  # registers PyArray for DEVICE-mode returns
from nvmolkit.types import CoordinateOutput, Device3DResult, HardwareOptions


@overload
def EmbedMolecules(
    molecules: list["Mol"],
    params: "EmbedParameters",
    confsPerMolecule: int = 1,
    maxIterations: int = -1,
    hardwareOptions: Optional[HardwareOptions] = None,
    output: Literal[CoordinateOutput.RDKIT_CONFORMERS] = CoordinateOutput.RDKIT_CONFORMERS,
    targetGpu: int = -1,
) -> None: ...
@overload
def EmbedMolecules(
    molecules: list["Mol"],
    params: "EmbedParameters",
    confsPerMolecule: int = 1,
    maxIterations: int = -1,
    hardwareOptions: Optional[HardwareOptions] = None,
    *,
    output: Literal[CoordinateOutput.DEVICE],
    targetGpu: int = -1,
) -> Device3DResult: ...
def EmbedMolecules(
    molecules: list["Mol"],
    params: "EmbedParameters",
    confsPerMolecule: int = 1,
    maxIterations: int = -1,
    hardwareOptions: Optional[HardwareOptions] = None,
    output: CoordinateOutput = CoordinateOutput.RDKIT_CONFORMERS,
    targetGpu: int = -1,
):
    """Embed multiple molecules with multiple conformers on GPUs.

    This function performs GPU-accelerated ETKDG conformer generation on multiple molecules.
    It uses CUDA for GPU acceleration and OpenMP for CPU parallelization to achieve high
    performance embedding of large molecule sets.

    nvMolKit implements a subset of features specified in the EmbedParameters class. The following features are restricted:

        - useRandomCoords must be True
        - Bounds matrices are not supported (setBoundsMat)
        - Custom Coulomb potentials are not supported (SetCPCI)
        - Coordinate constraints are not supported (SetCoordMap)
        - embedFragmentsSeparately is not supported. All fragments will be embedded together.

    Args:
        molecules: List of RDKit molecules to embed. Molecules should be prepared
                  (sanitized, explicit hydrogens added if needed).
        params: RDKit EmbedParameters object with embedding settings. Must have
               useRandomCoords=True for ETKDG.
        confsPerMolecule: Number of conformers to generate per molecule (default: 1)
        maxIterations: Maximum ETKDG iterations, -1 for automatic calculation (default: -1)
        hardwareOptions: HardwareOptions with hardware settings. If None, uses defaults.
        output: ``RDKIT_CONFORMERS`` (default) writes generated conformers back into each input
            molecule in-place and returns ``None``. ``DEVICE`` retains conformer coordinates on
            GPU and returns a :class:`Device3DResult`; RDKit conformers are NOT modified.
            DEVICE mode is incompatible with ``params.pruneRmsThresh > 0``.
        targetGpu: In DEVICE mode, the GPU to consolidate the result onto. ``-1`` selects the
            first configured execution GPU.

    Returns:
        For ``RDKIT_CONFORMERS``: ``None``; input molecules are modified in-place with
        generated conformers.
        For ``DEVICE``: a :class:`Device3DResult` whose ``values`` field carries
        ``(total_atoms, 3)`` coordinates plus CSR indices.

    Raises:
        ValueError: If any molecule in the input list is invalid, or if hardware
                   configuration parameters are invalid
        RuntimeError: If CUDA operations fail or embedding encounters errors

    Example:
        >>> from rdkit import Chem
        >>> from rdkit.Chem.rdDistGeom import ETKDGv3
        >>> from nvmolkit.types import HardwareOptions
        >>> import nvmolkit.embedMolecules as embed
        >>>
        >>> # Load molecules
        >>> mol1 = Chem.AddHs(Chem.MolFromSmiles('CCO'))
        >>> mol2 = Chem.AddHs(Chem.MolFromSmiles('CCC'))
        >>>
        >>> # Set up embedding parameters
        >>> params = ETKDGv3()
        >>> params.useRandomCoords = True  # Required for nvMolKit ETKDG
        >>>
        >>> # Configure hardware options
        >>> hardware_opts = HardwareOptions(
        ...     preprocessingThreads=8,
        ...     batchSize=500,
        ...     batchesPerGpu=4,
        ...     gpuIds=[0, 1],
        ... )
        >>> embed.EmbedMolecules([mol1, mol2], params, confsPerMolecule=5, hardwareOptions=hardware_opts)
        >>>
        >>> # Check conformers were generated
        >>> mol1.GetNumConformers()  # Should be 5
        >>> mol2.GetNumConformers()  # Should be 5

    Note:
        - In ``RDKIT_CONFORMERS`` mode (default), input molecules are modified in-place with
          generated conformers. In ``DEVICE`` mode, RDKit conformers are not touched.
        - params.useRandomCoords must be True for ETKDG algorithm
        - If gpuIds is empty, all available GPUs (0 to N-1) will be used automatically
    """
    if not molecules:
        if output == CoordinateOutput.DEVICE:
            raise ValueError("EmbedMolecules(output=DEVICE) requires at least one molecule")
        return None

    for i, mol in enumerate(molecules):
        if mol is None:
            raise ValueError(f"Molecule at index {i} is None")

    if not params.useRandomCoords:
        raise ValueError("ETKDG requires useRandomCoords=True in EmbedParameters")

    if hardwareOptions is None:
        hardwareOptions = HardwareOptions()
    native_options = hardwareOptions._as_native()

    if output == CoordinateOutput.DEVICE:
        return _embedMolecules.EmbedMoleculesDevice(
            molecules, params, confsPerMolecule, maxIterations, native_options, int(targetGpu)
        )
    _embedMolecules.EmbedMolecules(molecules, params, confsPerMolecule, maxIterations, native_options)
    return None
