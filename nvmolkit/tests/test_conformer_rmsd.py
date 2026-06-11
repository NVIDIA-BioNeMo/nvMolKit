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

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from nvmolkit.conformerRmsd import GetConformerRMSMatrix, GetConformerRMSMatrixBatch
from nvmolkit.types import AsyncGpuResult


def _numpy_kabsch_rmsd(p, q):
    """Independent Kabsch RMSD using numpy SVD (gold reference)."""
    p_c = p - p.mean(axis=0)
    q_c = q - q.mean(axis=0)
    H = p_c.T @ q_c
    _u, S, _vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(H))
    S[-1] *= d if d != 0.0 else 1.0
    Sp = np.sum(p_c**2)
    Sq = np.sum(q_c**2)
    return np.sqrt(max((Sp + Sq - 2.0 * np.sum(S)) / len(p), 0.0))


def _embed_mol(smiles, num_confs=10, seed=42):
    """Helper: create a molecule with conformers."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    return mol


def _rdkit_rmsd_matrix(mol, prealigned=False):
    """Helper: compute reference RMSD matrix using RDKit."""
    return list(AllChem.GetConformerRMSMatrix(mol, prealigned=prealigned))


def _numpy_rmsd_matrix(mol, prealigned=False):
    """Compute full RMSD matrix using numpy Kabsch (independent reference)."""
    confs = mol.GetConformers()
    n = len(confs)
    coords = [np.array(c.GetPositions()) for c in confs]
    result = []
    for i in range(n):
        for j in range(i):
            if prealigned:
                diff = coords[i] - coords[j]
                rmsd = np.sqrt(np.sum(diff**2) / len(coords[i]))
            else:
                rmsd = _numpy_kabsch_rmsd(coords[i], coords[j])
            result.append(rmsd)
    return result


@pytest.mark.parametrize("smiles", ["CCCCCC", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
def test_rmsd_matches_numpy_kabsch(smiles):
    """GPU RMSD matrix matches numpy Kabsch SVD reference within tolerance."""
    mol = _embed_mol(smiles, num_confs=20)
    no_h = Chem.RemoveHs(mol)

    ref_rms = _numpy_rmsd_matrix(no_h, prealigned=False)
    gpu_result = GetConformerRMSMatrix(no_h, prealigned=False)
    torch.cuda.synchronize()
    gpu_rms = gpu_result.numpy().tolist()

    assert len(gpu_rms) == len(ref_rms), f"Length mismatch: GPU={len(gpu_rms)}, ref={len(ref_rms)}"

    for i, (g, r) in enumerate(zip(gpu_rms, ref_rms)):
        assert abs(g - r) < 0.01, f"RMSD mismatch at index {i}: GPU={g:.6f}, numpy={r:.6f}, diff={abs(g - r):.6f}"


@pytest.mark.parametrize("smiles", ["CCCCCC", "c1ccccc1"])
def test_rmsd_prealigned_matches_rdkit(smiles):
    """GPU RMSD matrix with prealigned=True matches RDKit reference."""
    mol = _embed_mol(smiles, num_confs=10)
    no_h = Chem.RemoveHs(mol)

    rdkit_rms = _rdkit_rmsd_matrix(no_h, prealigned=True)
    gpu_result = GetConformerRMSMatrix(no_h, prealigned=True)
    torch.cuda.synchronize()
    gpu_rms = gpu_result.numpy().tolist()

    assert len(gpu_rms) == len(rdkit_rms)
    for i, (g, r) in enumerate(zip(gpu_rms, rdkit_rms)):
        assert abs(g - r) < 0.01, f"RMSD mismatch at index {i}: GPU={g:.6f}, RDKit={r:.6f}"


def test_rmsd_large_conformer_set():
    """Test with a larger conformer set (typical production size)."""
    mol = _embed_mol("CCCCCCCCCC", num_confs=100)
    no_h = Chem.RemoveHs(mol)

    n = no_h.GetNumConformers()
    expected_pairs = n * (n - 1) // 2

    gpu_result = GetConformerRMSMatrix(no_h, prealigned=False)
    torch.cuda.synchronize()
    gpu_rms = gpu_result.numpy()

    assert gpu_rms.shape[0] == expected_pairs
    assert np.all(np.isfinite(gpu_rms))
    assert np.all(gpu_rms >= 0.0)


def test_rmsd_two_conformers():
    """Minimal case: exactly two conformers."""
    mol = _embed_mol("CCCC", num_confs=2)
    no_h = Chem.RemoveHs(mol)

    ref_rms = _numpy_rmsd_matrix(no_h)
    gpu_result = GetConformerRMSMatrix(no_h)
    torch.cuda.synchronize()
    gpu_rms = gpu_result.numpy()

    assert gpu_rms.shape[0] == 1
    assert abs(gpu_rms[0] - ref_rms[0]) < 0.01


def test_rmsd_explicit_condensed_output_matches_default():
    """Explicit condensed output preserves the default API behavior."""
    mol = _embed_mol("CCCCCC", num_confs=10)
    no_h = Chem.RemoveHs(mol)

    default_result = GetConformerRMSMatrix(no_h)
    condensed_result = GetConformerRMSMatrix(no_h, output_format="condensed")
    torch.cuda.synchronize()

    assert isinstance(condensed_result, AsyncGpuResult)
    assert torch.allclose(condensed_result.torch(), default_result.torch(), atol=1e-10)


def test_rmsd_square_output_matches_condensed():
    """Square output expands the default RDKit-style condensed vector."""
    mol = _embed_mol("CCCCCC", num_confs=10)
    no_h = Chem.RemoveHs(mol)
    n = no_h.GetNumConformers()

    condensed = GetConformerRMSMatrix(no_h).torch()
    square = GetConformerRMSMatrix(no_h, output_format="square")
    torch.cuda.synchronize()

    expected = torch.zeros((n, n), device=square.device, dtype=square.dtype)
    idx = torch.tril_indices(n, n, offset=-1, device=square.device)
    expected[idx[0], idx[1]] = condensed
    expected = expected + expected.T

    assert isinstance(square, torch.Tensor)
    assert square.shape == (n, n)
    assert torch.allclose(square, expected, atol=1e-10)
    assert torch.allclose(torch.diag(square), torch.zeros(n, device=square.device, dtype=square.dtype))


def test_rmsd_square_output_for_fewer_than_two_conformers():
    """Square output handles the same <2 conformer inputs as condensed output."""
    mol_zero = Chem.MolFromSmiles("CCO")
    mol_one = Chem.RemoveHs(_embed_mol("CCO", num_confs=1))

    square_zero = GetConformerRMSMatrix(mol_zero, output_format="square")
    square_one = GetConformerRMSMatrix(mol_one, output_format="square")
    torch.cuda.synchronize()

    assert isinstance(square_zero, torch.Tensor)
    assert square_zero.shape == (0, 0)
    assert isinstance(square_one, torch.Tensor)
    assert square_one.shape == (1, 1)
    assert square_one.item() == 0.0


def test_rmsd_rigid_molecule():
    """Rigid molecule (benzene) — all conformers should have near-zero RMSD."""
    mol = _embed_mol("c1ccccc1", num_confs=5)
    no_h = Chem.RemoveHs(mol)

    gpu_result = GetConformerRMSMatrix(no_h, prealigned=False)
    torch.cuda.synchronize()
    gpu_rms = gpu_result.numpy()

    assert np.all(gpu_rms < 0.5), f"Rigid molecule should have small RMSD, got max={gpu_rms.max():.4f}"


def test_rmsd_explicit_stream():
    """Test execution on an explicit CUDA stream."""
    mol = _embed_mol("CCCCCC", num_confs=10)
    no_h = Chem.RemoveHs(mol)

    s = torch.cuda.Stream()
    gpu_result = GetConformerRMSMatrix(no_h, stream=s)
    s.synchronize()

    ref_rms = _numpy_rmsd_matrix(no_h)
    gpu_rms = gpu_result.numpy().tolist()

    for g, r in zip(gpu_rms, ref_rms):
        assert abs(g - r) < 0.01


def test_rmsd_square_output_explicit_stream():
    """Square conversion is enqueued on the caller-provided stream."""
    mol = _embed_mol("CCCCCC", num_confs=10)
    no_h = Chem.RemoveHs(mol)

    s = torch.cuda.Stream()
    square = GetConformerRMSMatrix(no_h, stream=s, output_format="square")
    s.synchronize()

    ref_rms = _numpy_rmsd_matrix(no_h)
    n = no_h.GetNumConformers()
    expected = torch.zeros((n, n), device=square.device, dtype=square.dtype)
    idx = torch.tril_indices(n, n, offset=-1, device=square.device)
    expected[idx[0], idx[1]] = torch.tensor(ref_rms, device=square.device, dtype=square.dtype)
    expected = expected + expected.T

    assert torch.allclose(square, expected, atol=0.01)


def test_rmsd_invalid_input_none():
    """None molecule should raise ValueError."""
    with pytest.raises(ValueError, match="mol must not be None"):
        GetConformerRMSMatrix(None)


def test_rmsd_fewer_than_two_conformers():
    """Molecule with fewer than 2 conformers returns empty result."""
    mol = Chem.MolFromSmiles("CCO")
    result = GetConformerRMSMatrix(mol)
    assert result.numpy().shape[0] == 0


def test_rmsd_invalid_stream_type():
    """Non-stream argument should raise TypeError."""
    mol = _embed_mol("CCCC", num_confs=2)
    no_h = Chem.RemoveHs(mol)
    with pytest.raises(TypeError):
        GetConformerRMSMatrix(no_h, stream=42)


def test_rmsd_invalid_output_format():
    """Unknown output_format should fail before dispatch."""
    mol = _embed_mol("CCCC", num_confs=2)
    no_h = Chem.RemoveHs(mol)
    with pytest.raises(ValueError, match="output_format must be one of"):
        GetConformerRMSMatrix(no_h, output_format="matrix")


def test_rmsd_output_format_is_keyword_only():
    """output_format is keyword-only to avoid ambiguous positional calls."""
    mol = _embed_mol("CCCC", num_confs=2)
    no_h = Chem.RemoveHs(mol)
    with pytest.raises(TypeError):
        GetConformerRMSMatrix(no_h, False, None, "square")


# ---------------------------------------------------------------------------
# Batch API tests
# ---------------------------------------------------------------------------


def test_batch_matches_single():
    """Batch results match the single-molecule API for each molecule."""
    smiles_list = ["CCCCCC", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    mols = [Chem.RemoveHs(_embed_mol(s, num_confs=10)) for s in smiles_list]

    batch_results = GetConformerRMSMatrixBatch(mols, prealigned=False)
    torch.cuda.synchronize()

    for mol, batch_result in zip(mols, batch_results):
        single_result = GetConformerRMSMatrix(mol, prealigned=False)
        torch.cuda.synchronize()

        batch_rms = batch_result.numpy()
        single_rms = single_result.numpy()
        np.testing.assert_allclose(batch_rms, single_rms, atol=1e-10, err_msg="Batch and single-mol results differ")


def test_batch_mixed_conformer_counts():
    """Batch handles molecules with different conformer counts."""
    mol_many = Chem.RemoveHs(_embed_mol("CCCCCC", num_confs=20))
    mol_few = Chem.RemoveHs(_embed_mol("CC", num_confs=3))
    mol_one = Chem.RemoveHs(_embed_mol("CCO", num_confs=1))  # below threshold

    results = GetConformerRMSMatrixBatch([mol_many, mol_few, mol_one])
    torch.cuda.synchronize()

    n_many = mol_many.GetNumConformers()
    n_few = mol_few.GetNumConformers()

    assert results[0].numpy().shape[0] == n_many * (n_many - 1) // 2
    assert results[1].numpy().shape[0] == n_few * (n_few - 1) // 2
    assert results[2].numpy().shape[0] == 0


def test_batch_explicit_condensed_output_matches_default():
    """Explicit batch condensed output preserves the default API behavior."""
    mols = [Chem.RemoveHs(_embed_mol(s, num_confs=5)) for s in ["CCCC", "CCCCC"]]

    default_results = GetConformerRMSMatrixBatch(mols)
    condensed_results = GetConformerRMSMatrixBatch(mols, output_format="condensed")
    torch.cuda.synchronize()

    for default_result, condensed_result in zip(default_results, condensed_results):
        assert isinstance(condensed_result, AsyncGpuResult)
        assert torch.allclose(condensed_result.torch(), default_result.torch(), atol=1e-10)


def test_batch_square_output():
    """Batch square output returns one NxN tensor per molecule."""
    mols = [
        Chem.RemoveHs(_embed_mol("CCCCCC", num_confs=8)),
        Chem.RemoveHs(_embed_mol("CC", num_confs=3)),
        Chem.MolFromSmiles("CCO"),
        Chem.RemoveHs(_embed_mol("CCO", num_confs=1)),
    ]

    condensed_results = GetConformerRMSMatrixBatch(mols)
    square_results = GetConformerRMSMatrixBatch(mols, output_format="square")
    torch.cuda.synchronize()

    for mol, condensed_result, square in zip(mols, condensed_results, square_results):
        n = mol.GetNumConformers()
        expected = torch.zeros((n, n), device=square.device, dtype=square.dtype)
        if n >= 2:
            idx = torch.tril_indices(n, n, offset=-1, device=square.device)
            expected[idx[0], idx[1]] = condensed_result.torch()
            expected = expected + expected.T

        assert isinstance(square, torch.Tensor)
        assert square.shape == (n, n)
        assert torch.allclose(square, expected, atol=1e-10)


def test_batch_empty_list():
    """Empty input returns an empty list."""
    results = GetConformerRMSMatrixBatch([])
    assert results == []


def test_batch_prealigned_matches_single():
    """Batch prealigned=True path matches the single-molecule API."""
    mols = [Chem.RemoveHs(_embed_mol(s, num_confs=8)) for s in ["CCCCCC", "c1ccccc1"]]

    batch_results = GetConformerRMSMatrixBatch(mols, prealigned=True)
    torch.cuda.synchronize()

    for mol, batch_result in zip(mols, batch_results):
        single_result = GetConformerRMSMatrix(mol, prealigned=True)
        torch.cuda.synchronize()
        np.testing.assert_allclose(
            batch_result.numpy(),
            single_result.numpy(),
            atol=1e-10,
            err_msg="Batch prealigned and single-mol results differ",
        )


def test_batch_invalid_none():
    """None molecule in list raises ValueError."""
    mol = Chem.RemoveHs(_embed_mol("CCCC", num_confs=2))
    with pytest.raises(ValueError):
        GetConformerRMSMatrixBatch([mol, None])


def test_batch_invalid_output_format():
    """Unknown batch output_format should fail before dispatch."""
    mol = Chem.RemoveHs(_embed_mol("CCCC", num_confs=2))
    with pytest.raises(ValueError, match="output_format must be one of"):
        GetConformerRMSMatrixBatch([mol], output_format="matrix")


def test_batch_output_format_is_keyword_only():
    """Batch output_format is keyword-only to avoid ambiguous positional calls."""
    mol = Chem.RemoveHs(_embed_mol("CCCC", num_confs=2))
    with pytest.raises(TypeError):
        GetConformerRMSMatrixBatch([mol], False, None, "square")


def test_batch_explicit_stream():
    """Batch results are correct on an explicit CUDA stream."""
    mols = [Chem.RemoveHs(_embed_mol(s, num_confs=5)) for s in ["CCCC", "CCCCC"]]

    s = torch.cuda.Stream()
    results = GetConformerRMSMatrixBatch(mols, stream=s)
    s.synchronize()

    for mol, result in zip(mols, results):
        ref = _numpy_rmsd_matrix(mol)
        rms = result.numpy().tolist()
        for g, r in zip(rms, ref):
            assert abs(g - r) < 0.01


def test_batch_square_output_explicit_stream():
    """Batch square conversion is enqueued on the caller-provided stream."""
    mols = [Chem.RemoveHs(_embed_mol(s, num_confs=5)) for s in ["CCCC", "CCCCC"]]

    s = torch.cuda.Stream()
    squares = GetConformerRMSMatrixBatch(mols, stream=s, output_format="square")
    s.synchronize()

    for mol, square in zip(mols, squares):
        ref_rms = _numpy_rmsd_matrix(mol)
        n = mol.GetNumConformers()
        expected = torch.zeros((n, n), device=square.device, dtype=square.dtype)
        idx = torch.tril_indices(n, n, offset=-1, device=square.device)
        expected[idx[0], idx[1]] = torch.tensor(ref_rms, device=square.device, dtype=square.dtype)
        expected = expected + expected.T

        assert torch.allclose(square, expected, atol=0.01)


def test_rmsd_zero_atoms():
    """0-atom molecule with multiple conformers raises ValueError.

    nvMolKit intentionally diverges from RDKit here: RDKit returns [nan] for
    exactly 2 zero-atom conformers and raises ZeroDivisionError for 3+.
    nvMolKit raises ValueError consistently for all degenerate zero-atom inputs.
    Such molecules cannot be produced by standard RDKit embedding workflows.
    """
    mol = Chem.RWMol()
    mol.AddConformer(Chem.Conformer(0), assignId=True)
    mol.AddConformer(Chem.Conformer(0), assignId=True)
    with pytest.raises(ValueError):
        GetConformerRMSMatrix(mol.GetMol())
