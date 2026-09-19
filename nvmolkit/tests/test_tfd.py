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

"""Tests for GPU-accelerated TFD calculation."""

import os

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints

import nvmolkit.tfd as tfd

# Tolerance for comparing GPU vs RDKit results
TOLERANCE = 0.01


def generate_conformers(mol, num_confs, seed=42):
    """Generate conformers for a molecule using ETKDG."""
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    return mol


@pytest.fixture
def simple_mol_with_conformers():
    """Create a simple molecule with conformers."""
    mol = Chem.MolFromSmiles("CCCCC")  # n-pentane
    return generate_conformers(mol, 5)


@pytest.fixture
def multiple_mols_with_conformers():
    """Create multiple molecules with conformers."""
    smiles_list = ["CCCC", "CCCCC", "CCCCCC", "CCO", "CCCO"]
    mols = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = generate_conformers(mol, 3 + i, seed=42 + i)
            if mol.GetNumConformers() >= 2:
                mols.append(mol)
    return mols


class TestGetTFDMatrix:
    """Tests for single-molecule TFD calculation."""

    def test_basic_tfd_matrix(self, simple_mol_with_conformers):
        """Test basic TFD matrix computation."""
        mol = simple_mol_with_conformers
        num_conf = mol.GetNumConformers()

        result = tfd.GetTFDMatrix(mol)

        # Check size: C*(C-1)/2 for C conformers
        expected_size = num_conf * (num_conf - 1) // 2
        assert len(result) == expected_size

        # All TFD values should be non-negative
        for val in result:
            assert val >= 0.0

    def test_single_conformer_returns_empty(self):
        """Test that single conformer returns empty matrix."""
        mol = Chem.MolFromSmiles("CCCC")
        generate_conformers(mol, 1)
        assert mol.GetNumConformers() == 1

        result = tfd.GetTFDMatrix(mol)
        assert len(result) == 0

    def test_no_weights(self, simple_mol_with_conformers):
        """Test computation without weights."""
        mol = simple_mol_with_conformers

        with_weights = tfd.GetTFDMatrix(mol, useWeights=True)
        no_weights = tfd.GetTFDMatrix(mol, useWeights=False)

        # Both should have same size
        assert len(with_weights) == len(no_weights)

        # Values may differ when weights are disabled
        # Just verify computation completes

    def test_maxdev_spec(self, simple_mol_with_conformers):
        """Test with specific max deviation mode."""
        mol = simple_mol_with_conformers

        equal_result = tfd.GetTFDMatrix(mol, maxDev="equal")
        spec_result = tfd.GetTFDMatrix(mol, maxDev="spec")

        assert len(equal_result) == len(spec_result)

    def test_invalid_maxdev_raises(self, simple_mol_with_conformers):
        """Test that invalid maxDev raises error."""
        mol = simple_mol_with_conformers

        with pytest.raises(ValueError):
            tfd.GetTFDMatrix(mol, maxDev="invalid")


class TestGetTFDMatrices:
    """Tests for batch TFD calculation."""

    def test_batch_processing(self, multiple_mols_with_conformers):
        """Test batch processing of multiple molecules."""
        mols = multiple_mols_with_conformers

        results = tfd.GetTFDMatrices(mols)

        assert len(results) == len(mols)

        for i, (mol, result) in enumerate(zip(mols, results)):
            num_conf = mol.GetNumConformers()
            expected_size = num_conf * (num_conf - 1) // 2
            assert len(result) == expected_size, f"Molecule {i} has wrong result size"

    def test_empty_input(self):
        """Test with empty molecule list."""
        results = tfd.GetTFDMatrices([])
        assert len(results) == 0

    def test_batch_vs_individual(self, multiple_mols_with_conformers):
        """Test that batch results match individual processing."""
        mols = multiple_mols_with_conformers

        batch_results = tfd.GetTFDMatrices(mols)
        individual_results = [tfd.GetTFDMatrix(mol) for mol in mols]

        assert len(batch_results) == len(individual_results)

        for batch, individual in zip(batch_results, individual_results):
            assert len(batch) == len(individual)
            for b, ind in zip(batch, individual):
                assert abs(b - ind) < 1e-4


class TestGpuResidentOutput:
    """Tests for GPU-resident TFD output via return_type parameter."""

    def test_tensor_return_single(self, simple_mol_with_conformers):
        """Test return_type='tensor' for a single molecule."""
        mol = simple_mol_with_conformers

        tensor = tfd.GetTFDMatrix(mol, return_type="tensor")

        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cuda"
        num_conf = mol.GetNumConformers()
        expected_size = num_conf * (num_conf - 1) // 2
        assert tensor.shape[0] == expected_size

    def test_tensor_return_batch(self, multiple_mols_with_conformers):
        """Test return_type='tensor' for batch."""
        mols = multiple_mols_with_conformers

        tensors = tfd.GetTFDMatrices(mols, return_type="tensor")
        assert len(tensors) == len(mols)

        for i, (mol, tensor) in enumerate(zip(mols, tensors)):
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device.type == "cuda"
            num_conf = mol.GetNumConformers()
            expected_size = num_conf * (num_conf - 1) // 2
            assert tensor.shape[0] == expected_size, f"Molecule {i} has wrong tensor size"

    def test_list_matches_tensor(self, multiple_mols_with_conformers):
        """Test that list and tensor return types give consistent results."""
        mols = multiple_mols_with_conformers

        lists = tfd.GetTFDMatrices(mols, return_type="list")
        tensors = tfd.GetTFDMatrices(mols, return_type="tensor")

        assert len(lists) == len(tensors)
        for gpu_list, tensor in zip(lists, tensors):
            tensor_list = tensor.cpu().tolist()
            assert len(gpu_list) == len(tensor_list)
            for l, t in zip(gpu_list, tensor_list):
                assert abs(l - t) < 1e-4

    def test_return_type_numpy(self, multiple_mols_with_conformers):
        """Test return_type='numpy' returns correct numpy arrays."""
        import numpy as np

        mols = multiple_mols_with_conformers
        lists = tfd.GetTFDMatrices(mols, return_type="list")
        arrays = tfd.GetTFDMatrices(mols, return_type="numpy")

        assert len(arrays) == len(lists)
        for arr, lst in zip(arrays, lists):
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.float32
            assert len(arr) == len(lst)
            for a, l in zip(arr.tolist(), lst):
                assert abs(a - l) < 1e-4

    def test_return_type_tensor(self, multiple_mols_with_conformers):
        """Test return_type='tensor' returns correct GPU tensors."""
        import torch

        mols = multiple_mols_with_conformers
        lists = tfd.GetTFDMatrices(mols, return_type="list")
        tensors = tfd.GetTFDMatrices(mols, return_type="tensor")

        assert len(tensors) == len(lists)
        for tensor, lst in zip(tensors, lists):
            assert isinstance(tensor, torch.Tensor)
            assert tensor.is_cuda
            assert tensor.dtype == torch.float32
            assert len(tensor) == len(lst)
            for t, l in zip(tensor.cpu().tolist(), lst):
                assert abs(t - l) < 1e-4


class TestCompareWithRDKit:
    """Tests comparing nvMolKit TFD with RDKit TFD."""

    def test_simple_chain_molecule(self):
        """Test simple chain molecule against RDKit."""
        mol = Chem.MolFromSmiles("CCCCC")
        generate_conformers(mol, 4)

        nvmolkit_result = tfd.GetTFDMatrix(mol, useWeights=True, maxDev="equal")
        rdkit_result = TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev="equal")

        assert len(nvmolkit_result) == len(rdkit_result)

        for nv, rd in zip(nvmolkit_result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE, f"nvMolKit={nv}, RDKit={rd}"

    def test_no_weights_comparison(self):
        """Test without weights against RDKit."""
        mol = Chem.MolFromSmiles("CCCCCC")
        generate_conformers(mol, 4)

        nvmolkit_result = tfd.GetTFDMatrix(mol, useWeights=False, maxDev="equal")
        rdkit_result = TorsionFingerprints.GetTFDMatrix(mol, useWeights=False, maxDev="equal")

        assert len(nvmolkit_result) == len(rdkit_result)

        for nv, rd in zip(nvmolkit_result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE

    def test_ring_molecule(self):
        """Test ring molecule (multi-quartet) against RDKit."""
        mol = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
        generate_conformers(mol, 4)

        nvmolkit_result = tfd.GetTFDMatrix(mol, useWeights=True, maxDev="equal")
        rdkit_result = TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev="equal")

        assert len(nvmolkit_result) == len(rdkit_result)

        for nv, rd in zip(nvmolkit_result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE, f"nvMolKit={nv}, RDKit={rd}"

    def test_ring_with_substituent(self):
        """Test molecule with both ring and non-ring torsions against RDKit."""
        mol = Chem.MolFromSmiles("c1ccccc1CC")  # ethylbenzene
        generate_conformers(mol, 4)

        nvmolkit_result = tfd.GetTFDMatrix(mol, useWeights=True, maxDev="equal")
        rdkit_result = TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev="equal")

        assert len(nvmolkit_result) == len(rdkit_result)

        for nv, rd in zip(nvmolkit_result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE, f"nvMolKit={nv}, RDKit={rd}"

    @pytest.mark.parametrize(
        "smiles",
        ["CCCCC", "CC(C)CC"],
    )
    def test_add_hs(self, smiles):
        """Test molecules with explicit hydrogens against RDKit."""
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        generate_conformers(mol, 4)

        nvmolkit_result = tfd.GetTFDMatrix(mol, useWeights=True, maxDev="equal")
        rdkit_result = TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev="equal")

        assert len(nvmolkit_result) == len(rdkit_result)

        for nv, rd in zip(nvmolkit_result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE, f"AddHs({smiles}): nvMolKit={nv}, RDKit={rd}"

    def test_symmetric_molecule(self):
        """Test branched molecule against RDKit."""
        mol = Chem.MolFromSmiles("CC(C)CC")  # isopentane
        generate_conformers(mol, 4)

        result = tfd.GetTFDMatrix(mol)
        rdkit_result = TorsionFingerprints.GetTFDMatrix(mol)

        assert len(result) == len(rdkit_result)

        for nv, rd in zip(result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE, f"nvMolKit={nv}, RDKit={rd}"

    @pytest.mark.parametrize("symm_radius", [0, 1, 3])
    def test_symm_radius(self, symm_radius):
        """Test different symmRadius values against RDKit.

        Uses 4-propylheptane: the two propyl chains are symmetric at radius 1
        but distinguishable at radius 3 due to the longer heptane backbone.
        """
        mol = Chem.MolFromSmiles("CCCC(CCC)CCC")
        generate_conformers(mol, 4)

        nvmolkit_result = tfd.GetTFDMatrix(
            mol,
            symmRadius=symm_radius,
        )
        rdkit_result = TorsionFingerprints.GetTFDMatrix(
            mol,
            symmRadius=symm_radius,
        )

        assert len(nvmolkit_result) == len(rdkit_result)

        for nv, rd in zip(nvmolkit_result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE, f"symmRadius={symm_radius}: nvMolKit={nv}, RDKit={rd}"

    def test_ignore_colinear_bonds_false(self):
        """Test ignoreColinearBonds=False against RDKit."""
        mol = Chem.MolFromSmiles("CCCC#CCC")
        generate_conformers(mol, 4)

        nvmolkit_result = tfd.GetTFDMatrix(
            mol,
            ignoreColinearBonds=False,
        )
        rdkit_result = TorsionFingerprints.GetTFDMatrix(
            mol,
            ignoreColinearBonds=False,
        )

        assert len(nvmolkit_result) == len(rdkit_result)

        for nv, rd in zip(nvmolkit_result, rdkit_result):
            assert abs(nv - rd) < TOLERANCE, f"ignoreColinearBonds=False: nvMolKit={nv}, RDKit={rd}"

    def test_batch_against_rdkit(self):
        """Test GetTFDMatrices batch against per-molecule RDKit on nontrivial molecules.

        Loads 10 diverse molecules from the MMFF94 validation SDF, generates
        conformers, and checks that the batch result matches RDKit per-molecule.
        """
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

        # Select 10 diverse molecules with enough atoms for torsions
        mols = []
        for mol in supplier:
            if mol is None or mol.GetNumAtoms() < 8:
                continue
            generate_conformers(mol, 5)
            if mol.GetNumConformers() >= 2:
                mols.append(mol)
            if len(mols) >= 10:
                break

        assert len(mols) == 10, f"Expected 10 molecules, got {len(mols)}"

        # Batch call
        nvmolkit_results = tfd.GetTFDMatrices(mols)

        assert len(nvmolkit_results) == len(mols)

        # Compare each molecule against RDKit
        for mol_idx, (nv_result, mol) in enumerate(zip(nvmolkit_results, mols)):
            rdkit_result = TorsionFingerprints.GetTFDMatrix(mol)

            assert len(nv_result) == len(rdkit_result), (
                f"Mol {mol_idx}: size mismatch nvMolKit={len(nv_result)}, RDKit={len(rdkit_result)}"
            )
            for i, (nv, rd) in enumerate(zip(nv_result, rdkit_result)):
                assert abs(nv - rd) < TOLERANCE, f"Mol {mol_idx}, pair {i}: nvMolKit={nv}, RDKit={rd}"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_ethane_no_torsions(self):
        """Test ethane (terminal bonds only, no rotatable bonds)."""
        mol = Chem.MolFromSmiles("CC")
        generate_conformers(mol, 3)

        result = tfd.GetTFDMatrix(mol)

        # Should return zeros
        for val in result:
            assert val == 0.0

    def test_triple_bond_chain_no_torsions(self):
        """Test that triple bond chains produce no torsions.

        1,3-butadiyne (HC#C-C#CH) has bonds that look rotatable but the
        colinear triple bonds should exclude them from torsion assignment.
        """
        mol = Chem.MolFromSmiles("C#CC#C")
        generate_conformers(mol, 3)

        result = tfd.GetTFDMatrix(mol)

        for val in result:
            assert val == 0.0

    def test_large_molecule(self):
        """Test larger molecule."""
        # Irinotecan (topoisomerase inhibitor, 66 atoms, multiple rings + flexible chains)
        mol = Chem.MolFromSmiles("CCC1(CC)C2=C(COC1=O)C(=O)N1CC3=CC4=C(N=C3C=C1C2=O)C=CC1=C4CN(C1)C(=O)OCC")
        generate_conformers(mol, 5)

        result = tfd.GetTFDMatrix(mol)

        num_conf = mol.GetNumConformers()
        expected_size = num_conf * (num_conf - 1) // 2
        assert len(result) == expected_size

        for val in result:
            assert 0.0 <= val <= 1.0  # TFD should be normalized

    def test_invalid_molecule_raises(self):
        """Test that None molecule raises error."""
        with pytest.raises(ValueError):
            tfd.GetTFDMatrix(None)

    def test_invalid_molecule_in_batch_raises(self, simple_mol_with_conformers):
        """Test that None in batch raises error."""
        mols = [simple_mol_with_conformers, None]

        with pytest.raises(ValueError):
            tfd.GetTFDMatrices(mols)
