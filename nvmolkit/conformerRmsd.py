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

"""GPU-accelerated pairwise RMSD matrix for molecular conformers."""

from typing import TYPE_CHECKING

import torch

from nvmolkit import _conformerRmsd
from nvmolkit.types import AsyncGpuResult

if TYPE_CHECKING:
    from rdkit.Chem import Mol

__all__ = ["GetConformerRMSMatrix", "GetConformerRMSMatrixBatch"]


_VALID_OUTPUT_FORMATS = frozenset({"condensed", "square"})


def _check_output_format(output_format: str) -> None:
    if output_format not in _VALID_OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {sorted(_VALID_OUTPUT_FORMATS)}, got {output_format!r}")


def _condensed_to_square(rmsd: AsyncGpuResult, num_confs: int) -> torch.Tensor:
    values = rmsd.torch()
    square = torch.zeros((num_confs, num_confs), device=values.device, dtype=values.dtype)
    if num_confs < 2:
        return square

    idx = torch.tril_indices(num_confs, num_confs, offset=-1, device=values.device)
    square[idx[0], idx[1]] = values
    return square + square.T


def GetConformerRMSMatrix(
    mol: "Mol",
    prealigned: bool = False,
    stream: torch.cuda.Stream | None = None,
    *,
    output_format: str = "condensed",
) -> AsyncGpuResult | torch.Tensor:
    """Compute the pairwise RMSD matrix between all conformers of a molecule on GPU.

    GPU-accelerated equivalent of ``AllChem.GetConformerRMSMatrix(mol,
    prealigned=prealigned)``.  For N conformers with M atoms, computes N*(N-1)/2
    pairwise RMSD values using one GPU thread-block per pair.  When ``prealigned``
    is False (default), each pair is optimally superimposed via the Kabsch algorithm
    before computing RMSD.

    **Differences from RDKit:**

    * Zero-atom molecules always raise ``ValueError`` regardless of conformer count.
      RDKit returns ``[nan]`` for exactly 2 zero-atom conformers and raises
      ``ZeroDivisionError`` for 3 or more.
    * Results are returned as an :class:`AsyncGpuResult` (device tensor) rather
      than a Python list, to keep the conformer-selection pipeline on the GPU.

    By default, the result matches RDKit's condensed lower-triangle shape.  Use
    ``output_format="square"`` when a downstream API expects an ``N x N`` distance
    matrix, such as :func:`nvmolkit.clustering.butina`.

    Args:
        mol: RDKit molecule with two or more conformers.  Strip hydrogens first
             (``Chem.RemoveHs``) if you want heavy-atom RMSD, as this function
             operates on all atoms present in the molecule.
        prealigned: If True, skip Kabsch alignment and compute RMSD on raw
                    coordinates.  If False (default), optimally align each pair.
        stream: CUDA stream to use.  If None, uses the current stream.
        output_format: ``"condensed"`` returns an :class:`AsyncGpuResult` wrapping
                       a 1-D RDKit-style lower-triangle vector.  ``"square"``
                       returns a symmetric ``torch.Tensor`` of shape ``(N, N)``
                       on the GPU with zeros on the diagonal.

    Returns:
        With ``output_format="condensed"``, an :class:`AsyncGpuResult` wrapping a
        1-D tensor of shape ``(N*(N-1)/2,)`` containing RMSD values in
        lower-triangle condensed order.  The RMSD for conformer pair (i, j) with
        i > j is at index ``i*(i-1)//2 + j``.

        With ``output_format="square"``, a symmetric CUDA ``torch.Tensor`` of
        shape ``(N, N)``.

    Raises:
        ValueError: If ``mol`` is None or has conformers but no atoms.
        TypeError: If ``stream`` is not a ``torch.cuda.Stream`` or None.

    Example:
        >>> from rdkit import Chem
        >>> from rdkit.Chem import AllChem, rdDistGeom
        >>> from nvmolkit.conformerRmsd import GetConformerRMSMatrix
        >>> from nvmolkit.clustering import butina
        >>>
        >>> mol = Chem.AddHs(Chem.MolFromSmiles('CCCCCC'))
        >>> rdDistGeom.EmbedMultipleConfs(mol, numConfs=50)
        >>> no_h = Chem.RemoveHs(mol)
        >>>
        >>> # GPU equivalent of: AllChem.GetConformerRMSMatrix(no_h)
        >>> rmsd_matrix = GetConformerRMSMatrix(no_h)
        >>>
        >>> # Request square output for GPU Butina clustering
        >>> square = GetConformerRMSMatrix(no_h, output_format="square")
        >>> clusters = butina(square, cutoff=0.5)
    """
    if mol is None:
        raise ValueError("mol must not be None")
    if stream is not None and not isinstance(stream, torch.cuda.Stream):
        raise TypeError(f"stream must be a torch.cuda.Stream or None, got {type(stream).__name__}")
    _check_output_format(output_format)

    execution_stream = stream if stream is not None else torch.cuda.current_stream()
    stream_ptr = execution_stream.cuda_stream
    result = AsyncGpuResult(_conformerRmsd.GetConformerRMSMatrix(mol, prealigned, stream_ptr))
    if output_format == "square":
        with torch.cuda.stream(execution_stream):
            return _condensed_to_square(result, mol.GetNumConformers())
    return result


def GetConformerRMSMatrixBatch(
    mols: list["Mol"],
    prealigned: bool = False,
    stream: torch.cuda.Stream | None = None,
    *,
    output_format: str = "condensed",
) -> list[AsyncGpuResult] | list[torch.Tensor]:
    """Compute pairwise RMSD matrices for a batch of molecules on GPU.

    All molecules are processed in a single kernel launch, so their conformer
    pairs execute concurrently.  This improves GPU saturation over repeated
    single-molecule calls, particularly for molecules with few conformers.

    Args:
        mols: List of RDKit molecules, each with zero or more conformers.
              Strip hydrogens first (``Chem.RemoveHs``) for heavy-atom RMSD.
              Molecules with fewer than 2 conformers return an empty result.
        prealigned: If True, skip Kabsch alignment and compute RMSD on raw
                    coordinates.  If False (default), optimally align each pair.
        stream: CUDA stream to use.  If None, uses the current stream.
        output_format: ``"condensed"`` returns one :class:`AsyncGpuResult` per
                       molecule, each wrapping a 1-D RDKit-style lower-triangle
                       vector.  ``"square"`` returns one symmetric CUDA
                       ``torch.Tensor`` of shape ``(N, N)`` per molecule.

    Returns:
        With ``output_format="condensed"``, a list of :class:`AsyncGpuResult`,
        one per input molecule, in the same order as ``mols``.  Each element
        wraps a 1-D tensor of shape ``(N*(N-1)/2,)`` for a molecule with N
        conformers, or shape ``(0,)`` for molecules with fewer than 2
        conformers.  The RMSD for conformer pair (i, j) with i > j is at index
        ``i*(i-1)//2 + j``.

        With ``output_format="square"``, a list of symmetric CUDA
        ``torch.Tensor`` objects with shapes ``(N, N)``.

    Raises:
        ValueError: If any element of ``mols`` is None.
        TypeError: If ``stream`` is not a ``torch.cuda.Stream`` or None.

    Example:
        >>> from rdkit import Chem
        >>> from rdkit.Chem import rdDistGeom
        >>> from nvmolkit.conformerRmsd import GetConformerRMSMatrixBatch
        >>>
        >>> mols = [Chem.RemoveHs(Chem.AddHs(Chem.MolFromSmiles(s)))
        ...         for s in ["CCCCCC", "c1ccccc1"]]
        >>> for mol in mols:
        ...     rdDistGeom.EmbedMultipleConfs(mol, numConfs=20)
        >>>
        >>> results = GetConformerRMSMatrixBatch(mols)
        >>> # results[0] is AsyncGpuResult for mols[0], etc.
    """
    if stream is not None and not isinstance(stream, torch.cuda.Stream):
        raise TypeError(f"stream must be a torch.cuda.Stream or None, got {type(stream).__name__}")
    _check_output_format(output_format)

    for i, mol in enumerate(mols):
        if mol is None:
            raise ValueError(f"mol at index {i} must not be None")

    execution_stream = stream if stream is not None else torch.cuda.current_stream()
    stream_ptr = execution_stream.cuda_stream
    raw = _conformerRmsd.GetConformerRMSMatrixBatch(mols, prealigned, stream_ptr)
    results = [AsyncGpuResult(r) for r in raw]
    if output_format == "square":
        with torch.cuda.stream(execution_stream):
            return [_condensed_to_square(result, mol.GetNumConformers()) for result, mol in zip(results, mols)]
    return results
