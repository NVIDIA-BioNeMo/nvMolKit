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

import os
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.Chem.AllChem import ETKDGv3
from rdkit.ForceField import rdForceField as _rdForceField  # noqa: F401
from rdkit.Geometry import Point3D

from nvmolkit.embedMolecules import EmbedMolecules
import nvmolkit.mmffOptimization as nvmolkit_mmff
from nvmolkit.types import CoordinateOutput, Device3DResult, HardwareOptions


@pytest.fixture
def mmff_test_mols(num_mols=5):
    """Load molecules from MMFF94_dative.sdf for testing.

    Args:
        num_mols: Number of molecules to load (default: 5)

    Returns:
        list: A list of RDKit molecules with conformers from the SDF file.
    """
    # Path from nvmolkit/tests/ to tests/test_data/
    sdf_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",  # Go up to project root
        "tests",
        "test_data",
        "MMFF94_dative.sdf",
    )

    if not os.path.exists(sdf_path):
        pytest.skip(f"Test data file not found: {sdf_path}")

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    molecules = []

    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        if i >= num_mols:  # Load only requested number of molecules
            break
        molecules.append(mol)

    if len(molecules) < num_mols:
        pytest.skip(f"Expected {num_mols} molecules, but found only {len(molecules)} in {sdf_path}")

    return molecules


def create_hard_copy_mols(molecules):
    """Create true hard copies of molecules with their conformers.

    Args:
        molecules: List of RDKit molecules to copy

    Returns:
        list: List of copied molecules with identical conformers
    """
    copied_mols = []
    for mol in molecules:
        # Create a new molecule from the original's structure
        copied_mol = Chem.Mol(mol)
        copied_mols.append(copied_mol)

    return copied_mols


def make_fragmented_mol():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.CC"))
    params = ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = 42
    rdDistGeom.EmbedMultipleConfs(mol, numConfs=1, params=params)
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


def make_rdkit_mmff_properties(mol, settings: dict | None = None):
    settings = {} if settings is None else settings
    variant = settings.get("variant", "MMFF94")
    mmff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    if mmff_props is None:
        raise ValueError("RDKit could not create MMFF properties for molecule")
    mmff_props.SetMMFFVariant(variant)
    mmff_props.SetMMFFDielectricConstant(settings.get("dielectric_constant", 1.0))
    mmff_props.SetMMFFDielectricModel(settings.get("dielectric_model", 1))
    mmff_props.SetMMFFBondTerm(settings.get("bond_term", True))
    mmff_props.SetMMFFAngleTerm(settings.get("angle_term", True))
    mmff_props.SetMMFFStretchBendTerm(settings.get("stretch_bend_term", True))
    mmff_props.SetMMFFOopTerm(settings.get("oop_term", True))
    mmff_props.SetMMFFTorsionTerm(settings.get("torsion_term", True))
    mmff_props.SetMMFFVdWTerm(settings.get("vdw_term", True))
    mmff_props.SetMMFFEleTerm(settings.get("ele_term", True))
    return mmff_props


def calculate_rdkit_mmff_energies(
    molecules,
    maxIters=200,
    property_settings: dict | None = None,
    nonBondedThreshold: float = 100.0,
    ignoreInterfragInteractions: bool = True,
):
    """Calculate MMFF energies using RDKit for all conformers of all molecules.

    Args:
        molecules: List of RDKit molecules with conformers

    Returns:
        list: List of lists containing energies for each molecule's conformers
    """
    all_energies = []

    for mol in molecules:
        mol_energies = []
        num_conformers = mol.GetNumConformers()

        if num_conformers == 0:
            all_energies.append([])
            continue

        # Optimize all conformers for this molecule using RDKit
        # The signature shows it's a method on the molecule object
        mmff_props = make_rdkit_mmff_properties(mol, property_settings)
        for conf_id in range(num_conformers):
            ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
                mol,
                mmff_props,
                nonBondedThresh=nonBondedThreshold,
                confId=conf_id,
                ignoreInterfragInteractions=ignoreInterfragInteractions,
            )
            ff.Initialize()
            ff.Minimize(maxIts=maxIters)
            mol_energies.append(ff.CalcEnergy())

        all_energies.append(mol_energies)

    return all_energies


def test_mmff_optimization_serial_vs_rdkit(mmff_test_mols):
    """Test nvMolKit MMFF optimization one molecule at a time against RDKit reference.

    This test compares the energy results when optimizing molecules individually
    using nvMolKit vs RDKit's MMFFOptimizeMoleculeConfs function.
    """
    # Create hard copies for fair comparison
    rdkit_mols = create_hard_copy_mols(mmff_test_mols)
    nvmolkit_mols = create_hard_copy_mols(mmff_test_mols)

    # Get RDKit reference energies
    rdkit_energies = calculate_rdkit_mmff_energies(rdkit_mols)

    # Get nvMolKit energies one molecule at a time (serial mode)
    nvmolkit_energies = []
    for mol in nvmolkit_mols:
        if mol.GetNumConformers() == 0:
            nvmolkit_energies.append([])
            continue

        # Call nvMolKit with single molecule
        mol_energies = nvmolkit_mmff.MMFFOptimizeMoleculesConfs(
            [mol],
            maxIters=200,
        )
        nvmolkit_energies.extend(mol_energies)

    # Verify we have the same number of molecules
    assert len(rdkit_energies) == len(nvmolkit_energies), (
        f"Mismatch in number of molecules: RDKit={len(rdkit_energies)}, nvMolKit={len(nvmolkit_energies)}"
    )

    # Compare energies for each molecule
    for mol_idx, (rdkit_mol_energies, nvmolkit_mol_energies) in enumerate(zip(rdkit_energies, nvmolkit_energies)):
        assert len(rdkit_mol_energies) == len(nvmolkit_mol_energies), (
            f"Molecule {mol_idx}: conformer count mismatch: RDKit={len(rdkit_mol_energies)}, nvMolKit={len(nvmolkit_mol_energies)}"
        )

        # Compare each conformer's energy with tolerance
        for conf_idx, (rdkit_energy, nvmolkit_energy) in enumerate(zip(rdkit_mol_energies, nvmolkit_mol_energies)):
            energy_diff = abs(rdkit_energy - nvmolkit_energy)
            rel_error = energy_diff / abs(rdkit_energy) if abs(rdkit_energy) > 1e-10 else energy_diff

            assert rel_error < 1e-3, (
                f"Molecule {mol_idx}, Conformer {conf_idx}: energy mismatch: "
                f"RDKit={rdkit_energy:.6f}, nvMolKit={nvmolkit_energy:.6f}, "
                f"abs_diff={energy_diff:.6f}, rel_error={rel_error:.6f}"
            )


@pytest.mark.parametrize("gpu_ids", [[0, 1], [0], [1]])
@pytest.mark.parametrize("batchesize", [0, 2, 5])
@pytest.mark.parametrize("batches_per_gpu", [1, 3])
def test_mmff_optimization_batch_vs_rdkit(mmff_test_mols, gpu_ids, batchesize, batches_per_gpu):
    """Test nvMolKit MMFF batch optimization against RDKit reference.

    This test compares the energy results when optimizing all molecules together
    in batch mode using nvMolKit vs individual RDKit optimization.
    """
    available_devices = torch.cuda.device_count()
    if available_devices == 1 and 1 in gpu_ids:
        pytest.skip("Test requires at least 2 GPUs for batch mode comparison")
    # Create hard copies for fair comparison
    rdkit_mols = create_hard_copy_mols(mmff_test_mols)
    nvmolkit_mols = create_hard_copy_mols(mmff_test_mols)

    # Get RDKit reference energies
    rdkit_energies = calculate_rdkit_mmff_energies(rdkit_mols)

    hardware_options = HardwareOptions(
        gpuIds=gpu_ids,
        batchSize=batchesize,
        batchesPerGpu=batches_per_gpu,
    )

    # Get nvMolKit energies in batch mode (all molecules at once)
    nvmolkit_energies = nvmolkit_mmff.MMFFOptimizeMoleculesConfs(
        nvmolkit_mols, maxIters=200, hardwareOptions=hardware_options
    )

    # Verify we have the same number of molecules
    assert len(rdkit_energies) == len(nvmolkit_energies), (
        f"Mismatch in number of molecules: RDKit={len(rdkit_energies)}, nvMolKit={len(nvmolkit_energies)}"
    )

    # Compare energies for each molecule
    for mol_idx, (rdkit_mol_energies, nvmolkit_mol_energies) in enumerate(zip(rdkit_energies, nvmolkit_energies)):
        assert len(rdkit_mol_energies) == len(nvmolkit_mol_energies), (
            f"Molecule {mol_idx}: conformer count mismatch: RDKit={len(rdkit_mol_energies)}, nvMolKit={len(nvmolkit_mol_energies)}"
        )

        # Compare each conformer's energy with tolerance
        for conf_idx, (rdkit_energy, nvmolkit_energy) in enumerate(zip(rdkit_mol_energies, nvmolkit_mol_energies)):
            energy_diff = abs(rdkit_energy - nvmolkit_energy)
            rel_error = energy_diff / abs(rdkit_energy) if abs(rdkit_energy) > 1e-10 else energy_diff

            assert rel_error < 1e-3, (
                f"Molecule {mol_idx}, Conformer {conf_idx}: energy mismatch: "
                f"RDKit={rdkit_energy:.6f}, nvMolKit={nvmolkit_energy:.6f}, "
                f"abs_diff={energy_diff:.6f}, rel_error={rel_error:.6f}"
            )


def test_mmff_optimization_empty_input():
    """Test nvMolKit MMFF optimization with empty input."""
    result = nvmolkit_mmff.MMFFOptimizeMoleculesConfs([])
    assert result == []


def test_mmff_optimization_invalid_input():
    """Test nvMolKit MMFF optimization with invalid input."""
    with pytest.raises(ValueError, match="None at indices") as exc_info:
        nvmolkit_mmff.MMFFOptimizeMoleculesConfs([None])
    assert exc_info.value.args[1] == {"none": [0], "no_params": []}


def test_mmff_optimization_allows_large_molecule_interleaved():
    """Ensure a large (>256 atoms) molecule in batch is accepted and optimized."""
    small1 = Chem.AddHs(Chem.MolFromSmiles("CCCCCC"), explicitOnly=False)
    small2 = Chem.AddHs(Chem.MolFromSmiles("CCC"), explicitOnly=False)
    big = Chem.AddHs(Chem.MolFromSmiles("C" * 100), explicitOnly=False)
    assert big.GetNumAtoms() > 256

    rdDistGeom.EmbedMultipleConfs(small1, numConfs=1)
    rdDistGeom.EmbedMultipleConfs(small2, numConfs=1)
    rdDistGeom.EmbedMultipleConfs(big, numConfs=1)

    mols = [small1, big, small2]
    rdkit_mols = create_hard_copy_mols(mols)
    rdkit_energies = calculate_rdkit_mmff_energies(rdkit_mols, maxIters=10)

    energies = nvmolkit_mmff.MMFFOptimizeMoleculesConfs(mols, maxIters=10)
    assert len(energies) == 3

    for mol_idx, (rdkit_mol_energies, nvmolkit_mol_energies) in enumerate(zip(rdkit_energies, energies)):
        assert len(rdkit_mol_energies) == len(nvmolkit_mol_energies), (
            f"Molecule {mol_idx}: conformer count mismatch: RDKit={len(rdkit_mol_energies)}, nvMolKit={len(nvmolkit_mol_energies)}"
        )

        # Compare each conformer's energy with tolerance
        for conf_idx, (rdkit_energy, nvmolkit_energy) in enumerate(zip(rdkit_mol_energies, nvmolkit_mol_energies)):
            energy_diff = abs(rdkit_energy - nvmolkit_energy)
            rel_error = energy_diff / abs(rdkit_energy) if abs(rdkit_energy) > 1e-10 else energy_diff

            assert rel_error < 1e-3, (
                f"Molecule {mol_idx}, Conformer {conf_idx}: energy mismatch: "
                f"RDKit={rdkit_energy:.6f}, nvMolKit={nvmolkit_energy:.6f}, "
                f"abs_diff={energy_diff:.6f}, rel_error={rel_error:.6f}"
            )


def test_mmff_optimization_custom_properties_vs_rdkit(mmff_test_mols):
    custom_property_settings = {
        "dielectric_constant": 2.0,
        "dielectric_model": 2,
    }
    custom_props = make_rdkit_mmff_properties(mmff_test_mols[0], custom_property_settings)

    # Step 0: compare initial energies (no minimization) to verify properties are applied
    for label, props, settings in [("default", None, None), ("custom", custom_props, custom_property_settings)]:
        rdkit_mols_0 = create_hard_copy_mols(mmff_test_mols[:2])
        nvmolkit_mols_0 = create_hard_copy_mols(mmff_test_mols[:2])
        rdkit_e0 = calculate_rdkit_mmff_energies(rdkit_mols_0, maxIters=0, property_settings=settings)
        nvmolkit_e0 = nvmolkit_mmff.MMFFOptimizeMoleculesConfs(
            nvmolkit_mols_0,
            maxIters=0,
            properties=props,
        )
        for mol_idx, (r, n) in enumerate(zip(rdkit_e0, nvmolkit_e0)):
            for conf_idx, (re, ne) in enumerate(zip(r, n)):
                diff = abs(re - ne)
                rel = diff / abs(re) if abs(re) > 1e-10 else diff
                assert rel < 1e-3, (
                    f"[{label}] Step-0 mol {mol_idx} conf {conf_idx}: RDKit={re:.6f} nvMolKit={ne:.6f} rel={rel:.6f}"
                )

    # Verify custom properties actually change the energy
    default_mols = create_hard_copy_mols(mmff_test_mols[:2])
    custom_mols = create_hard_copy_mols(mmff_test_mols[:2])
    default_e0 = calculate_rdkit_mmff_energies(default_mols, maxIters=0)
    custom_e0 = calculate_rdkit_mmff_energies(custom_mols, maxIters=0, property_settings=custom_property_settings)
    for mol_idx, (de, ce) in enumerate(zip(default_e0, custom_e0)):
        for conf_idx, (d, c) in enumerate(zip(de, ce)):
            assert abs(d - c) > 1e-3, (
                f"Mol {mol_idx} conf {conf_idx}: default and custom energies "
                f"should differ: default={d:.6f} custom={c:.6f}"
            )

    # Now test with minimization
    rdkit_mols = create_hard_copy_mols(mmff_test_mols[:2])
    nvmolkit_mols = create_hard_copy_mols(mmff_test_mols[:2])
    rdkit_energies = calculate_rdkit_mmff_energies(
        rdkit_mols,
        maxIters=100,
        property_settings=custom_property_settings,
    )
    nvmolkit_energies = nvmolkit_mmff.MMFFOptimizeMoleculesConfs(
        nvmolkit_mols,
        maxIters=100,
        properties=custom_props,
    )

    assert len(rdkit_energies) == len(nvmolkit_energies)
    for mol_idx, (rdkit_mol_energies, nvmolkit_mol_energies) in enumerate(zip(rdkit_energies, nvmolkit_energies)):
        assert len(rdkit_mol_energies) == len(nvmolkit_mol_energies)
        for conf_idx, (rdkit_energy, nvmolkit_energy) in enumerate(zip(rdkit_mol_energies, nvmolkit_mol_energies)):
            energy_diff = abs(rdkit_energy - nvmolkit_energy)
            rel_error = energy_diff / abs(rdkit_energy) if abs(rdkit_energy) > 1e-10 else energy_diff
            assert rel_error < 1e-2, (
                f"Molecule {mol_idx}, Conformer {conf_idx}: energy mismatch: "
                f"RDKit={rdkit_energy:.6f}, nvMolKit={nvmolkit_energy:.6f}, "
                f"abs_diff={energy_diff:.6f}, rel_error={rel_error:.6f}"
            )


def test_mmff_optimization_per_molecule_properties_vs_rdkit(mmff_test_mols):
    mols = create_hard_copy_mols(mmff_test_mols[:2])
    rdkit_mols = create_hard_copy_mols(mols)
    nvmolkit_mols = create_hard_copy_mols(mols)

    property_settings = [
        {"variant": "MMFF94s", "dielectric_constant": 2.0, "dielectric_model": 2},
        {
            "bond_term": False,
            "angle_term": False,
            "stretch_bend_term": False,
            "oop_term": False,
            "torsion_term": False,
        },
    ]
    properties = [make_rdkit_mmff_properties(mol, settings) for mol, settings in zip(nvmolkit_mols, property_settings)]

    rdkit_energies = [
        calculate_rdkit_mmff_energies([mol], maxIters=100, property_settings=settings)[0]
        for mol, settings in zip(rdkit_mols, property_settings)
    ]
    nvmolkit_energies = nvmolkit_mmff.MMFFOptimizeMoleculesConfs(
        nvmolkit_mols,
        maxIters=100,
        properties=properties,
    )

    assert len(rdkit_energies) == len(nvmolkit_energies)
    for mol_idx, (rdkit_mol_energies, nvmolkit_mol_energies) in enumerate(zip(rdkit_energies, nvmolkit_energies)):
        assert len(rdkit_mol_energies) == len(nvmolkit_mol_energies)
        for conf_idx, (rdkit_energy, nvmolkit_energy) in enumerate(zip(rdkit_mol_energies, nvmolkit_mol_energies)):
            energy_diff = abs(rdkit_energy - nvmolkit_energy)
            rel_error = energy_diff / abs(rdkit_energy) if abs(rdkit_energy) > 1e-10 else energy_diff
            assert rel_error < 1e-2, (
                f"Molecule {mol_idx}, Conformer {conf_idx}: energy mismatch: "
                f"RDKit={rdkit_energy:.6f}, nvMolKit={nvmolkit_energy:.6f}, "
                f"abs_diff={energy_diff:.6f}, rel_error={rel_error:.6f}"
            )


def test_mmff_optimization_per_molecule_thresholds_and_interfrag_vs_rdkit():
    mols = [make_fragmented_mol(), make_fragmented_mol()]
    rdkit_mols = create_hard_copy_mols(mols)
    nvmolkit_mols = create_hard_copy_mols(mols)
    properties = [make_rdkit_mmff_properties(mol) for mol in nvmolkit_mols]
    non_bonded_thresholds = [25.0, 100.0]
    ignore_interfrag_interactions = [False, True]

    rdkit_energies = [
        calculate_rdkit_mmff_energies(
            [mol],
            maxIters=100,
            nonBondedThreshold=threshold,
            ignoreInterfragInteractions=ignore_interfrag,
        )[0]
        for mol, threshold, ignore_interfrag in zip(rdkit_mols, non_bonded_thresholds, ignore_interfrag_interactions)
    ]
    nvmolkit_energies = nvmolkit_mmff.MMFFOptimizeMoleculesConfs(
        nvmolkit_mols,
        maxIters=100,
        properties=properties,
        nonBondedThreshold=non_bonded_thresholds,
        ignoreInterfragInteractions=ignore_interfrag_interactions,
    )

    assert len(rdkit_energies) == len(nvmolkit_energies)
    for mol_idx, (rdkit_mol_energies, nvmolkit_mol_energies) in enumerate(zip(rdkit_energies, nvmolkit_energies)):
        assert len(rdkit_mol_energies) == len(nvmolkit_mol_energies)
        for conf_idx, (rdkit_energy, nvmolkit_energy) in enumerate(zip(rdkit_mol_energies, nvmolkit_mol_energies)):
            energy_diff = abs(rdkit_energy - nvmolkit_energy)
            rel_error = energy_diff / abs(rdkit_energy) if abs(rdkit_energy) > 1e-10 else energy_diff
            assert rel_error < 1e-2, (
                f"Molecule {mol_idx}, Conformer {conf_idx}: energy mismatch: "
                f"RDKit={rdkit_energy:.6f}, nvMolKit={nvmolkit_energy:.6f}, "
                f"abs_diff={energy_diff:.6f}, rel_error={rel_error:.6f}"
            )


# Testing github issue 9 - openmp error handling
def test_error_case_throws_properly():
    smiles = "CC1(C)OB(CC2=CC=CC=C2)OC1(C)C"
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    params = ETKDGv3()
    params.useRandomCoords = True
    EmbedMolecules([mol], params, confsPerMolecule=1)

    with pytest.raises(ValueError, match="lacking MMFF atom types") as exc_info:
        nvmolkit_mmff.MMFFOptimizeMoleculesConfs([mol], maxIters=200)
    assert exc_info.value.args[1] == {"none": [], "no_params": [0]}


def test_mmff_optimization_device_output_matches_host(mmff_test_mols):
    """MMFFOptimizeMoleculesConfs(output=DEVICE) returns Device3DResult; energies match host path."""
    host_mols = create_hard_copy_mols(mmff_test_mols)
    device_mols = create_hard_copy_mols(mmff_test_mols)

    host_energies = nvmolkit_mmff.MMFFOptimizeMoleculesConfs(host_mols, maxIters=50)

    result = nvmolkit_mmff.MMFFOptimizeMoleculesConfs(
        device_mols,
        maxIters=50,
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
