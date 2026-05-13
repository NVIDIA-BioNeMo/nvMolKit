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

"""DEVICE-mode tests for the batched forcefield wrappers (issue #140).

Exercises ``minimize(output=DEVICE)`` on both :class:`MMFFBatchedForcefield` and
:class:`UFFBatchedForcefield` and checks the device-mode result against the host-mode
nested-list result on identical input.
"""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from nvmolkit.batchedForcefield import MMFFBatchedForcefield, UFFBatchedForcefield
from nvmolkit.types import CoordinateOutput, Device3DResult


def make_embedded_mol(smiles: str, num_confs: int = 1, seed: int = 0xC0FFEE):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    return mol


def _flatten(nested):
    return [v for inner in nested for v in inner]


@pytest.fixture(scope="module")
def two_mols():
    return [
        make_embedded_mol("CCO", num_confs=2, seed=1),
        make_embedded_mol("CCCC", num_confs=3, seed=2),
    ]


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_minimize_device_returns_device3d_with_optimized_positions(ff_cls, two_mols):
    """minimize(output=DEVICE) returns Device3DResult; values reflect updated coords."""
    mols_initial = [Chem.Mol(m) for m in two_mols]
    pos_initial_per_mol = []
    for mol in mols_initial:
        for conf_idx in range(mol.GetNumConformers()):
            conf = mol.GetConformer(conf_idx)
            for atom_idx in range(mol.GetNumAtoms()):
                p = conf.GetAtomPosition(atom_idx)
                pos_initial_per_mol.extend([p.x, p.y, p.z])

    ff = ff_cls([Chem.Mol(m) for m in two_mols])
    result = ff.minimize(maxIters=15, output=CoordinateOutput.DEVICE)
    assert isinstance(result, Device3DResult)
    torch.cuda.synchronize()
    pos_after = result.values.torch().reshape(-1).tolist()
    assert len(pos_after) == len(pos_initial_per_mol)
    assert any(abs(a - b) > 1e-6 for a, b in zip(pos_after, pos_initial_per_mol))


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_minimize_device_energies_match_host(ff_cls, two_mols):
    mols_host = [Chem.Mol(m) for m in two_mols]
    mols_device = [Chem.Mol(m) for m in two_mols]

    ff_host = ff_cls(mols_host)
    energies_host_nested, _ = ff_host.minimize(maxIters=20)
    energies_host = _flatten(energies_host_nested)

    ff_device = ff_cls(mols_device)
    result = ff_device.minimize(maxIters=20, output=CoordinateOutput.DEVICE)
    assert result.energies is not None
    assert result.converged is not None
    torch.cuda.synchronize()
    energies_device = result.energies.torch().tolist()

    assert len(energies_device) == len(energies_host)
    for h, d in zip(energies_host, energies_device):
        assert abs(h - d) < 1e-3, f"host {h} vs device {d}"


@pytest.mark.parametrize("ff_cls", [MMFFBatchedForcefield, UFFBatchedForcefield])
def test_minimize_device_rejects_cross_gpu_target(ff_cls, two_mols):
    """The wrapper is single-GPU; minimize(output=DEVICE, target_gpu=other) must raise."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires >= 2 GPUs to exercise cross-GPU rejection")
    with torch.cuda.device(0):
        ff = ff_cls(two_mols)
        with pytest.raises(Exception, match="does not support target_gpu"):
            ff.minimize(maxIters=5, output=CoordinateOutput.DEVICE, target_gpu=1)
