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
from rdkit.Chem import rdDistGeom, AllChem
from rdkit.Chem.rdDistGeom import EmbedParameters

import nvmolkit.embedMolecules as embed
from nvmolkit.types import CoordinateOutput, Device3DResult, HardwareOptions


@pytest.fixture
def embed_test_mols(num_mols=5):
    """Load molecules from MMFF94_dative.sdf for embedding tests.

    Args:
        num_mols: Number of molecules to load (default: 5)

    Returns:
        list: A list of RDKit molecules without conformers.
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
        # Clear any existing conformers for clean embedding tests
        mol.RemoveAllConformers()
        molecules.append(mol)

    if len(molecules) < num_mols:
        pytest.skip(f"Expected {num_mols} molecules, but found only {len(molecules)} in {sdf_path}")

    return molecules


def create_hard_copy_mols(molecules):
    """Create true hard copies of molecules.

    Args:
        molecules: List of RDKit molecules to copy

    Returns:
        list: List of copied molecules with no conformers
    """
    copied_mols = []
    for mol in molecules:
        # Create a new molecule from the original's structure
        copied_mol = Chem.Mol(mol)
        # Ensure no conformers
        copied_mol.RemoveAllConformers()
        copied_mols.append(copied_mol)

    return copied_mols


def embed_with_rdkit(molecules, confs_per_mol=5, params=None):
    """Embed molecules using RDKit's EmbedMultipleConfs for comparison.

    Args:
        molecules: List of RDKit molecules to embed
        confs_per_mol: Number of conformers to generate per molecule
        params: RDKit EmbedParameters object. If None, uses default ETKDGv3 with random seed 42 and useRandomCoords=True

    Returns:
        list: List of lists containing conformer IDs for each molecule
    """
    # Use default parameters if none provided
    if params is None:
        params = rdDistGeom.ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = 42

    all_conf_ids = []

    for mol in molecules:
        # Embed multiple conformers using RDKit with provided parameters
        conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=confs_per_mol, params=params)
        all_conf_ids.append(list(conf_ids))

    return all_conf_ids


def compare_conformers_rmsd(rdkit_mols, nvmolkit_mols, rmsd_threshold=0.2, min_match_fraction=0.5):
    """Compare conformers between RDKit and nvMolKit using RMSD metrics.

    For each molecule, calculate RMSD between every nvMolKit conformer and every RDKit conformer.
    A nvMolKit conformer is considered "similar" if it has RMSD < threshold to at least one RDKit conformer.

    Args:
        rdkit_mols: List of RDKit molecules with embedded conformers
        nvmolkit_mols: List of nvMolKit molecules with embedded conformers
        rmsd_threshold: Maximum RMSD for conformers to be considered similar (default: 0.2 Å)
        min_match_fraction: Minimum fraction of nvMolKit conformers that must match (default: 0.5)
    """
    for mol_idx, (rdkit_mol, nvmolkit_mol) in enumerate(zip(rdkit_mols, nvmolkit_mols)):
        # Check for None molecules to prevent crashes
        if rdkit_mol is None or nvmolkit_mol is None:
            continue

        rdkit_conf_count = rdkit_mol.GetNumConformers()
        nvmolkit_conf_count = nvmolkit_mol.GetNumConformers()

        if rdkit_conf_count == 0 or nvmolkit_conf_count == 0:
            continue

        # Count how many nvMolKit conformers have a similar RDKit conformer
        similar_conformers = 0

        try:
            # Create a temporary molecule with all conformers for RMSD calculation
            temp_mol = Chem.Mol(rdkit_mol)  # Copy the molecule structure
            temp_mol.RemoveAllConformers()  # Clear conformers

            # Add all RDKit conformers first (indices 0 to rdkit_conf_count-1)
            for conf_id in range(rdkit_conf_count):
                conf = rdkit_mol.GetConformer(conf_id)
                temp_mol.AddConformer(conf, assignId=False)  # Let RDKit assign IDs

            # Add all nvMolKit conformers next (indices rdkit_conf_count to rdkit_conf_count+nvmolkit_conf_count-1)
            for conf_id in range(nvmolkit_conf_count):
                conf = nvmolkit_mol.GetConformer(conf_id)
                temp_mol.AddConformer(conf, assignId=False)  # Let RDKit assign IDs

            # Calculate RMSD matrix for all conformers
            rmsd_matrix = AllChem.GetConformerRMSMatrix(temp_mol, prealigned=False)

            # Check each nvMolKit conformer against all RDKit conformers
            for nvmolkit_idx in range(nvmolkit_conf_count):
                # nvMolKit conformer index in the combined molecule
                nvmolkit_conf_idx = rdkit_conf_count + nvmolkit_idx

                # Find minimum RMSD between this nvMolKit conformer and all RDKit conformers
                min_rmsd = float("inf")
                for rdkit_conf_idx in range(rdkit_conf_count):
                    rmsd_index = nvmolkit_conf_idx * (nvmolkit_conf_idx + 1) // 2 + rdkit_conf_idx - nvmolkit_conf_idx
                    rmsd = rmsd_matrix[rmsd_index]
                    min_rmsd = min(min_rmsd, rmsd)

                # If this nvMolKit conformer is similar to at least one RDKit conformer
                if min_rmsd <= rmsd_threshold:
                    similar_conformers += 1

        except Exception as e:
            # Skip this molecule if RMSD calculation fails
            print(f"Error calculating RMSD for molecule {mol_idx}: {e}")
            continue

        # Calculate the fraction of similar conformers
        similar_fraction = similar_conformers / nvmolkit_conf_count if nvmolkit_conf_count > 0 else 0.0

        assert similar_fraction >= min_match_fraction, (
            f"Molecule {mol_idx}: Only {similar_conformers}/{nvmolkit_conf_count} "
            f"({similar_fraction:.2f}) nvMolKit conformers are similar to RDKit conformers "
            f"(RMSD < {rmsd_threshold} Å). Expected at least {min_match_fraction:.2f} fraction."
        )


@pytest.mark.parametrize("etkdg_variant", ["ETKDG", "ETKDGv2", "ETKDGv3", "srETKDGv3", "KDG", "ETDG", "DG"])
def test_embed_molecules_serial_vs_rdkit(embed_test_mols, etkdg_variant):
    """Test nvMolKit EmbedMolecules one molecule at a time against RDKit reference.

    This test compares the conformer generation when embedding molecules individually
    using nvMolKit vs RDKit's EmbedMultipleConfs function for different ETKDG variants.
    """
    confs_per_mol = 5

    # Create hard copies for fair comparison
    rdkit_mols = create_hard_copy_mols(embed_test_mols)
    nvmolkit_mols = create_hard_copy_mols(embed_test_mols)

    # Set up embedding parameters for the specified variant
    if etkdg_variant == "ETKDG":
        params = rdDistGeom.ETKDG()
    elif etkdg_variant == "ETKDGv2":
        params = rdDistGeom.ETKDGv2()
    elif etkdg_variant == "ETKDGv3":
        params = rdDistGeom.ETKDGv3()
    elif etkdg_variant == "srETKDGv3":
        params = rdDistGeom.srETKDGv3()
    elif etkdg_variant == "KDG":
        params = rdDistGeom.KDG()
    elif etkdg_variant == "ETDG":
        params = rdDistGeom.ETDG()
    elif etkdg_variant == "DG":
        params = rdDistGeom.KDG()
        params.useBasicKnowledge = False
    else:
        raise ValueError(f"Unknown ETKDG variant: {etkdg_variant}")

    params.useRandomCoords = True  # Required for nvmolkit ETKDG
    params.randomSeed = 42  # For reproducibility

    # Get RDKit reference conformer counts using the same parameters
    rdkit_conf_ids = embed_with_rdkit(rdkit_mols, confs_per_mol, params)

    # Get nvMolKit conformers one molecule at a time (serial mode)
    nvmolkit_conf_counts = []
    for mol in nvmolkit_mols:
        # Call nvMolKit with single molecule using hardware options
        hardware_opts = HardwareOptions(
            preprocessingThreads=1,
            batchSize=1,
            batchesPerGpu=1,
        )

        embed.EmbedMolecules(
            [mol], params, confsPerMolecule=confs_per_mol, maxIterations=-1, hardwareOptions=hardware_opts
        )
        nvmolkit_conf_counts.append(mol.GetNumConformers())

    # Verify we have the same number of molecules
    assert len(rdkit_conf_ids) == len(nvmolkit_conf_counts), (
        f"Mismatch in number of molecules: RDKit={len(rdkit_conf_ids)}, nvMolKit={len(nvmolkit_conf_counts)}"
    )

    # Compare conformer counts for each molecule
    for mol_idx, (rdkit_confs, nvmolkit_count) in enumerate(zip(rdkit_conf_ids, nvmolkit_conf_counts)):
        rdkit_count = len(rdkit_confs)
        assert rdkit_count == nvmolkit_count, (
            f"Molecule {mol_idx}: conformer count mismatch: RDKit={rdkit_count}, nvMolKit={nvmolkit_count}"
        )

        # Verify we got the expected number of conformers
        assert nvmolkit_count == confs_per_mol, (
            f"Molecule {mol_idx}: expected {confs_per_mol} conformers, got {nvmolkit_count}"
        )

    # Compare conformer similarity using RMSD
    compare_conformers_rmsd(rdkit_mols, nvmolkit_mols, rmsd_threshold=0.2, min_match_fraction=0.5)


@pytest.mark.parametrize("etkdg_variant", ["ETKDG", "ETKDGv2", "ETKDGv3", "srETKDGv3", "KDG", "ETDG", "DG"])
@pytest.mark.parametrize("gpu_ids", [[], [0], [1], [0, 1]])
def test_embed_molecules_batch_vs_rdkit(embed_test_mols, etkdg_variant, gpu_ids):
    """Test nvMolKit EmbedMolecules batch mode against RDKit reference.

    This test compares the conformer generation when embedding all molecules together
    in batch mode using nvMolKit vs individual RDKit embedding for different ETKDG variants.
    """
    available_devices = torch.cuda.device_count()
    if available_devices == 1 and 1 in gpu_ids:
        pytest.skip("Test requires at least 2 GPUs for batch mode comparison")
    confs_per_mol = 5

    # Create hard copies for fair comparison
    rdkit_mols = create_hard_copy_mols(embed_test_mols)
    nvmolkit_mols = create_hard_copy_mols(embed_test_mols)

    # Set up embedding parameters for the specified variant
    if etkdg_variant == "ETKDG":
        params = rdDistGeom.ETKDG()
    elif etkdg_variant == "ETKDGv2":
        params = rdDistGeom.ETKDGv2()
    elif etkdg_variant == "ETKDGv3":
        params = rdDistGeom.ETKDGv3()
    elif etkdg_variant == "srETKDGv3":
        params = rdDistGeom.srETKDGv3()
    elif etkdg_variant == "KDG":
        params = rdDistGeom.KDG()
    elif etkdg_variant == "ETDG":
        params = rdDistGeom.ETDG()
    elif etkdg_variant == "DG":
        params = rdDistGeom.KDG()
        params.useBasicKnowledge = False
    else:
        raise ValueError(f"Unknown ETKDG variant: {etkdg_variant}")

    params.useRandomCoords = True  # Required for nvmolkit ETKDG
    params.randomSeed = 42  # For reproducibility

    # Get RDKit reference conformer counts using the same parameters
    rdkit_conf_ids = embed_with_rdkit(rdkit_mols, confs_per_mol, params)

    # Get nvMolKit conformers in batch mode (all molecules at once)
    hardware_opts = HardwareOptions(
        preprocessingThreads=4,
        batchSize=10,
        batchesPerGpu=2,
        gpuIds=gpu_ids,
    )

    embed.EmbedMolecules(
        nvmolkit_mols, params, confsPerMolecule=confs_per_mol, maxIterations=-1, hardwareOptions=hardware_opts
    )

    # Get nvMolKit conformer counts
    nvmolkit_conf_counts = [mol.GetNumConformers() for mol in nvmolkit_mols]

    # Verify we have the same number of molecules
    assert len(rdkit_conf_ids) == len(nvmolkit_conf_counts), (
        f"Mismatch in number of molecules: RDKit={len(rdkit_conf_ids)}, nvMolKit={len(nvmolkit_conf_counts)}"
    )

    # Compare conformer counts for each molecule
    for mol_idx, (rdkit_confs, nvmolkit_count) in enumerate(zip(rdkit_conf_ids, nvmolkit_conf_counts)):
        rdkit_count = len(rdkit_confs)
        assert rdkit_count == nvmolkit_count, (
            f"Molecule {mol_idx}: conformer count mismatch: RDKit={rdkit_count}, nvMolKit={nvmolkit_count}"
        )

        # Verify we got the expected number of conformers
        assert nvmolkit_count == confs_per_mol, (
            f"Molecule {mol_idx}: expected {confs_per_mol} conformers, got {nvmolkit_count}"
        )

    # Compare conformer similarity using RMSD
    compare_conformers_rmsd(rdkit_mols, nvmolkit_mols, rmsd_threshold=0.2, min_match_fraction=0.5)


def test_embed_molecules_empty_input():
    """Test nvMolKit EmbedMolecules with empty input."""
    params = EmbedParameters()
    params.useRandomCoords = True

    # Should not raise any errors
    embed.EmbedMolecules([], params)


def test_embed_molecules_invalid_input():
    """Test nvMolKit EmbedMolecules with invalid input."""
    params = EmbedParameters()
    params.useRandomCoords = True

    # Test with None molecule
    with pytest.raises(ValueError, match="Molecule at index 0 is None"):
        embed.EmbedMolecules([None], params)


def test_embed_molecules_invalid_params():
    """Test nvMolKit EmbedMolecules with invalid parameters."""
    mol = Chem.MolFromSmiles("CCO")

    # Test with useRandomCoords=False
    params = EmbedParameters()
    params.useRandomCoords = False

    with pytest.raises(ValueError, match="ETKDG requires useRandomCoords=True"):
        embed.EmbedMolecules([mol], params)


def test_embed_molecules_with_hardware_options(embed_test_mols):
    """Test nvMolKit EmbedMolecules using hardware options wrapper."""
    confs_per_mol = 3

    # Create hard copies for fair comparison
    nvmolkit_mols = create_hard_copy_mols(embed_test_mols)

    # Set up nvMolKit embedding parameters
    params = EmbedParameters()
    params.useRandomCoords = True
    params.randomSeed = 42  # For reproducibility

    # Create hardware options
    hardware_opts = HardwareOptions(
        preprocessingThreads=2,
        batchSize=5,
        batchesPerGpu=1,
    )

    # Embed molecules using the struct interface
    embed.EmbedMolecules(
        nvmolkit_mols, params, confsPerMolecule=confs_per_mol, maxIterations=-1, hardwareOptions=hardware_opts
    )

    # Verify conformer counts
    for mol_idx, mol in enumerate(nvmolkit_mols):
        conf_count = mol.GetNumConformers()
        assert conf_count == confs_per_mol, (
            f"Molecule {mol_idx}: expected {confs_per_mol} conformers, got {conf_count}"
        )


def test_embed_molecules_allows_large_molecule_interleaved():
    """Ensure a large (>256 atoms) molecule in batch is accepted and embedded."""
    small1 = Chem.AddHs(Chem.MolFromSmiles("CCCCCC"))  # 6 atoms
    small2 = Chem.AddHs(Chem.MolFromSmiles("CCC"))  # 3 atoms
    big = Chem.AddHs(Chem.MolFromSmiles("C" * 100))
    assert big.GetNumAtoms() > 256

    params = EmbedParameters()
    params.useRandomCoords = True
    params.maxIterations = 5

    embed.EmbedMolecules([small1, big, small2], params, confsPerMolecule=1)
    assert small1.GetNumConformers() == 1
    assert small2.GetNumConformers() == 1
    assert big.GetNumConformers() == 1


def test_embed_molecules_prune_rmsthresh():
    mols = [Chem.MolFromSmiles("c1ccccc1"), Chem.MolFromSmiles("C" * 30)]
    params = EmbedParameters()
    params.useRandomCoords = True
    params.pruneRmsThresh = 0.5
    embed.EmbedMolecules(mols, params, confsPerMolecule=5)
    assert mols[0].GetNumConformers() == 1
    assert mols[1].GetNumConformers() == 5


def test_embed_molecules_device_output_returns_device3d_no_writeback():
    """EmbedMolecules(output=DEVICE) returns Device3DResult and does NOT modify RDKit conformers."""
    mols = [Chem.AddHs(Chem.MolFromSmiles("CCO")), Chem.AddHs(Chem.MolFromSmiles("CCCC"))]
    params = EmbedParameters()
    params.useRandomCoords = True
    params.randomSeed = 42

    confs_per_mol = 3
    result = embed.EmbedMolecules(
        mols,
        params,
        confsPerMolecule=confs_per_mol,
        output=CoordinateOutput.DEVICE,
    )
    assert isinstance(result, Device3DResult)
    assert result.n_mols == 2

    for mol in mols:
        assert mol.GetNumConformers() == 0

    torch.cuda.synchronize()
    mol_indices = result.mol_indices.torch().tolist()
    atom_starts = result.atom_starts.torch().tolist()
    assert atom_starts[0] == 0
    assert len(mol_indices) == len(atom_starts) - 1

    cursor = 0
    for mol_idx, mol in enumerate(mols):
        natoms = mol.GetNumAtoms()
        n_conf_for_mol = sum(1 for m in mol_indices if m == mol_idx)
        assert n_conf_for_mol > 0
        for _ in range(n_conf_for_mol):
            cursor += natoms
        # Total atoms accounted for so far must match the cumulative atom_starts entry
        # for this molecule's last conformer.
    assert atom_starts[-1] == cursor

    # ETKDG does not produce energies / convergence flags - those Optional fields must be None,
    # not AsyncGpuResults wrapping empty tensors.
    assert result.energies is None
    assert result.converged is None


def test_embed_molecules_device_output_empty_input_raises():
    params = EmbedParameters()
    params.useRandomCoords = True
    with pytest.raises(ValueError, match="requires at least one molecule"):
        embed.EmbedMolecules([], params, output=CoordinateOutput.DEVICE)
