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
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.ForceField import rdForceField as _rdForceField  # noqa: F401
from rdkit.Geometry import Point3D

import nvmolkit.uffOptimization as nvmolkit_uff
from nvmolkit.types import CoordinateOutput, Device3DResult, HardwareOptions


@pytest.fixture
def uff_test_mols(num_mols=5):
    """Load a handful of UFF-valid molecules from the shared validation set."""
    sdf_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "tests",
        "test_data",
        "MMFF94_dative.sdf",
    )

    if not os.path.exists(sdf_path):
        pytest.skip(f"Test data file not found: {sdf_path}")

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    molecules = []
    for mol in supplier:
        if mol is None:
            continue
        if not rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
            continue
        molecules.append(mol)
        if len(molecules) >= num_mols:
            break

    if len(molecules) < num_mols:
        pytest.skip(f"Expected {num_mols} UFF-valid molecules, found {len(molecules)}")

    return molecules


def create_hard_copy_mols(molecules):
    return [Chem.Mol(mol) for mol in molecules]


def make_fragmented_mol():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.CC"))
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = 42
    conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=1, params=params)
    if not conf_ids:
        raise RuntimeError("EmbedMultipleConfs failed to generate a conformer for fragmented mol")
    conf = mol.GetConformer()
    fragments = Chem.GetMolFrags(mol)
    if len(fragments) != 2:
        raise AssertionError("Expected two fragments for interfragment interaction test")
    anchor = conf.GetAtomPosition(fragments[0][0])
    moved = conf.GetAtomPosition(fragments[1][0])
    shift = Point3D(anchor.x - moved.x + 2.0, anchor.y - moved.y, anchor.z - moved.z)
    for atom_idx in fragments[1]:
        pos = conf.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(atom_idx, Point3D(pos.x + shift.x, pos.y + shift.y, pos.z + shift.z))
    return mol


def calculate_rdkit_uff_energies(
    molecules,
    maxIters=1000,
    vdwThreshold: float = 10.0,
    ignoreInterfragInteractions: bool = True,
):
    all_energies = []
    for mol in molecules:
        mol_energies = []
        for conf_id in range(mol.GetNumConformers()):
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(
                mol,
                vdwThresh=vdwThreshold,
                confId=conf_id,
                ignoreInterfragInteractions=ignoreInterfragInteractions,
            )
            ff.Initialize()
            ff.Minimize(maxIts=maxIters)
            mol_energies.append(ff.CalcEnergy())
        all_energies.append(mol_energies)
    return all_energies


def test_uff_optimization_serial_vs_rdkit(uff_test_mols):
    rdkit_mols = create_hard_copy_mols(uff_test_mols)
    nvmolkit_mols = create_hard_copy_mols(uff_test_mols)

    rdkit_energies = calculate_rdkit_uff_energies(rdkit_mols, maxIters=200)

    nvmolkit_energies = []
    for mol in nvmolkit_mols:
        mol_energies = nvmolkit_uff.UFFOptimizeMoleculesConfs([mol], maxIters=200)
        nvmolkit_energies.extend(mol_energies)

    assert len(rdkit_energies) == len(nvmolkit_energies)
    for mol_idx, (rdkit_mol_energies, nvmolkit_mol_energies) in enumerate(zip(rdkit_energies, nvmolkit_energies)):
        assert len(rdkit_mol_energies) == len(nvmolkit_mol_energies)
        for conf_idx, (rdkit_energy, nvmolkit_energy) in enumerate(zip(rdkit_mol_energies, nvmolkit_mol_energies)):
            diff = abs(rdkit_energy - nvmolkit_energy)
            rel = diff / abs(rdkit_energy) if abs(rdkit_energy) > 1e-10 else diff
            assert rel < 1e-3, (
                f"Molecule {mol_idx}, conformer {conf_idx}: "
                f"RDKit={rdkit_energy:.6f} nvMolKit={nvmolkit_energy:.6f} rel={rel:.6f}"
            )


def test_uff_optimization_batch_vs_rdkit(uff_test_mols):
    rdkit_mols = create_hard_copy_mols(uff_test_mols)
    nvmolkit_mols = create_hard_copy_mols(uff_test_mols)

    rdkit_energies = calculate_rdkit_uff_energies(rdkit_mols, maxIters=200)
    hardware_options = HardwareOptions(batchSize=2, batchesPerGpu=1)
    nvmolkit_energies = nvmolkit_uff.UFFOptimizeMoleculesConfs(
        nvmolkit_mols,
        maxIters=200,
        hardwareOptions=hardware_options,
    )

    assert len(rdkit_energies) == len(nvmolkit_energies)
    for mol_idx, (rdkit_mol_energies, nvmolkit_mol_energies) in enumerate(zip(rdkit_energies, nvmolkit_energies)):
        assert len(rdkit_mol_energies) == len(nvmolkit_mol_energies)
        for conf_idx, (rdkit_energy, nvmolkit_energy) in enumerate(zip(rdkit_mol_energies, nvmolkit_mol_energies)):
            diff = abs(rdkit_energy - nvmolkit_energy)
            rel = diff / abs(rdkit_energy) if abs(rdkit_energy) > 1e-10 else diff
            assert rel < 1e-3, (
                f"Molecule {mol_idx}, conformer {conf_idx}: "
                f"RDKit={rdkit_energy:.6f} nvMolKit={nvmolkit_energy:.6f} rel={rel:.6f}"
            )


def test_uff_optimization_empty_input():
    assert nvmolkit_uff.UFFOptimizeMoleculesConfs([]) == []


def test_uff_optimization_invalid_input():
    unsupported = Chem.MolFromSmiles("*")
    with pytest.raises(ValueError) as exc_info:
        nvmolkit_uff.UFFOptimizeMoleculesConfs([None, unsupported])
    assert exc_info.value.args[1]["none"] == [0]
    assert exc_info.value.args[1]["no_params"] == [1]


def test_uff_optimization_threshold_and_interfrag_vs_rdkit():
    mols = [make_fragmented_mol(), make_fragmented_mol()]
    rdkit_mols = create_hard_copy_mols(mols)
    nvmolkit_mols = create_hard_copy_mols(mols)

    thresholds = [25.0, 100.0]
    ignore_interfrag = [False, True]

    rdkit_energies = [
        calculate_rdkit_uff_energies(
            [mol],
            maxIters=1000,
            vdwThreshold=threshold,
            ignoreInterfragInteractions=ignore,
        )[0]
        for mol, threshold, ignore in zip(rdkit_mols, thresholds, ignore_interfrag)
    ]
    nvmolkit_energies = nvmolkit_uff.UFFOptimizeMoleculesConfs(
        nvmolkit_mols,
        maxIters=1000,
        vdwThreshold=thresholds,
        ignoreInterfragInteractions=ignore_interfrag,
    )

    assert len(rdkit_energies) == len(nvmolkit_energies)
    for mol_idx, (rdkit_mol_energies, nvmolkit_mol_energies) in enumerate(zip(rdkit_energies, nvmolkit_energies)):
        assert len(rdkit_mol_energies) == len(nvmolkit_mol_energies)
        for conf_idx, (rdkit_energy, nvmolkit_energy) in enumerate(zip(rdkit_mol_energies, nvmolkit_mol_energies)):
            diff = abs(rdkit_energy - nvmolkit_energy)
            rel = diff / abs(rdkit_energy) if abs(rdkit_energy) > 1e-10 else diff
            assert rel < 1e-3, (
                f"Molecule {mol_idx}, conformer {conf_idx}: "
                f"RDKit={rdkit_energy:.6f} nvMolKit={nvmolkit_energy:.6f} rel={rel:.6f}"
            )


def test_uff_optimization_device_output_matches_host(uff_test_mols):
    """UFFOptimizeMoleculesConfs(output=DEVICE) returns Device3DResult; energies match host path."""
    host_mols = create_hard_copy_mols(uff_test_mols)
    device_mols = create_hard_copy_mols(uff_test_mols)

    host_energies = nvmolkit_uff.UFFOptimizeMoleculesConfs(host_mols, maxIters=200)

    result = nvmolkit_uff.UFFOptimizeMoleculesConfs(
        device_mols,
        maxIters=200,
        output=CoordinateOutput.DEVICE,
    )
    assert isinstance(result, Device3DResult)
    torch.cuda.synchronize()
    device_energies_flat = result.energies.torch().tolist()

    expected_n = sum(m.GetNumConformers() for m in device_mols)
    assert len(device_energies_flat) == expected_n

    host_flat = [e for inner in host_energies for e in inner]
    assert len(host_flat) == len(device_energies_flat)
    for h, d in zip(host_flat, device_energies_flat):
        rel = abs(h - d) / max(abs(h), 1e-10)
        assert rel < 1e-2, f"host {h} vs device {d}"
