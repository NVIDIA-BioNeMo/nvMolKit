# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


@pytest.fixture
def one_hundred_smiles():
    """Load one hundred smiles from a CSV file.

    Returns:
        list: A list of one hundred SMILES strings.
    """
    path = os.path.join(os.path.dirname(__file__), "testdata", "smiles.csv")
    df = pd.read_csv(path)
    return df["smiles"].tolist()


@pytest.fixture
def one_hundred_mols(one_hundred_smiles):
    """Generate RDKit molecules from one hundred smiles.

    Args:
        one_hundred_smiles (list): A list of one hundred SMILES strings.

    Returns:
        list: A list of one hundred RDKit molecules.
    """
    return [Chem.MolFromSmiles(smi) for smi in one_hundred_smiles]


@pytest.fixture
def size_limited_mols(one_hundred_mols):
    """Generate RDKit molecules from one hundred smiles and discard any that have more than 128 atoms or bonds.

    Args:
        one_hundred_mols (list): A list of one hundred RDKit molecules.

    Returns:
        list: A list of RDKit molecules with at most 128 atoms and bonds. Up to 100 molecules are returned.
    """
    return [mol for mol in one_hundred_mols if mol.GetNumAtoms() <= 128 and mol.GetNumBonds() <= 128]


def _float32_distances(packed, metric):
    """Distances computed the way the fused kernels compute them, in correctly rounded float32."""
    bits = np.unpackbits(packed.view(np.uint8), axis=1).astype(np.float32)
    intersections = (bits @ bits.T).astype(np.float32)
    counts = bits.sum(axis=1)
    if metric == "tanimoto":
        unions = counts[:, None] + counts[None, :] - intersections
        with np.errstate(invalid="ignore", divide="ignore"):
            similarity = np.where(unions > 0, intersections / unions, np.float32(1.0))
    else:
        norms = np.sqrt(counts[:, None] * counts[None, :])
        with np.errstate(invalid="ignore", divide="ignore"):
            similarity = np.where(norms > 0, intersections / norms, np.float32(0.0))
    return (np.float32(1.0) - similarity.astype(np.float32)).astype(np.float32)


@pytest.fixture(scope="session")
def float32_distances():
    """Return a function computing float32 fingerprint distances as the fused clustering kernels do."""
    return _float32_distances


@pytest.fixture(scope="session")
def chembl_molecules():
    """Parse the ~1000 ChEMBL molecules in tests/test_data/chembl_1k.smi."""
    path = Path(__file__).resolve().parents[2] / "tests" / "test_data" / "chembl_1k.smi"
    lines = [line.split()[0] for line in path.read_text().splitlines() if line and not line.startswith("#")]
    molecules = [Chem.MolFromSmiles(smiles) for smiles in lines]
    return [molecule for molecule in molecules if molecule is not None]


@pytest.fixture(scope="session")
def chembl_fingerprints(chembl_molecules):
    """RDKit Morgan bit vectors and the same bits packed as ``(N, 32)`` uint32 words."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    bit_vectors = [generator.GetFingerprint(molecule) for molecule in chembl_molecules]
    bits = np.zeros((len(bit_vectors), 1024), dtype=np.uint8)
    for row, bit_vector in enumerate(bit_vectors):
        DataStructs.ConvertToNumpyArray(bit_vector, bits[row])
    return bit_vectors, np.packbits(bits, axis=1, bitorder="little").view(np.uint32)


@pytest.fixture(scope="session")
def chembl_distances(chembl_fingerprints):
    """Float32 Tanimoto and cosine distance matrices for the ChEMBL fingerprints."""
    _, packed = chembl_fingerprints
    return {metric: _float32_distances(packed, metric) for metric in ("tanimoto", "cosine")}
