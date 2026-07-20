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

import pytest
from rdkit import Chem
from rdkit.Chem import rdFMCS

from nvmolkit.mcs import MCSConfig, findMCS


def _mols(smiles: list[str]):
    return [Chem.MolFromSmiles(smi) for smi in smiles]


def _rdkit_params(
    *,
    atom_compare: str = "elements",
    bond_compare: str = "order",
    ring_matches_ring_only: bool = False,
):
    params = rdFMCS.MCSParameters()
    params.MaximizeBonds = True
    params.AtomCompareParameters.RingMatchesRingOnly = ring_matches_ring_only
    params.BondCompareParameters.RingMatchesRingOnly = ring_matches_ring_only

    atom_types = {
        "any": rdFMCS.AtomCompare.CompareAny,
        "elements": rdFMCS.AtomCompare.CompareElements,
        "isotopes": rdFMCS.AtomCompare.CompareIsotopes,
    }
    bond_types = {
        "any": rdFMCS.BondCompare.CompareAny,
        "order": rdFMCS.BondCompare.CompareOrder,
        "order_exact": rdFMCS.BondCompare.CompareOrderExact,
    }
    params.AtomTyper = atom_types[atom_compare]
    params.BondTyper = bond_types[bond_compare]
    return params


def _assert_matches_rdkit(result, mol_table, *, atom_compare="elements", bond_compare="order", **kwargs):
    params = _rdkit_params(atom_compare=atom_compare, bond_compare=bond_compare, **kwargs)
    assert len(result.num_atoms) == len(result.pairs)
    assert len(result.num_bonds) == len(result.pairs)
    for pair_idx, (idx_a, idx_b) in enumerate(result.pairs):
        rd_result = rdFMCS.FindMCS([mol_table[idx_a], mol_table[idx_b]], params)
        item = result[pair_idx]
        assert item.num_atoms == rd_result.numAtoms
        assert item.num_bonds == rd_result.numBonds
        assert item.atom_mapping.shape[1] == 2
        assert item.bond_mapping.shape[1] == 2


def test_pairs_mode_matches_rdkit_and_preserves_order():
    mols = _mols(["CCO", "CCN", "c1ccccc1", "c1ccc(O)cc1"])
    pairs = [(0, 1), (2, 3), (0, 2)]

    result = findMCS(mols, mode="pairs", pairs=pairs)

    assert result.mode == "pairs"
    assert result.pairs == tuple(pairs)
    assert len(result) == len(pairs)
    assert result.used_gpu.any()
    _assert_matches_rdkit(result, mols)


def test_pairs_mode_chunked_multi_executor_matches_rdkit():
    mols = _mols(["CCO", "CCN", "c1ccccc1", "c1ccc(O)cc1", "CC(C)O"])
    pairs = [(0, 1), (2, 3), (4, 0), (1, 4), (3, 2)]

    result = findMCS(
        mols,
        mode="pairs",
        pairs=pairs,
        batch_size=1,
        block_size=64,
        executors_per_runner=2,
    )

    assert result.pairs == tuple(pairs)
    assert result.used_gpu.any()
    _assert_matches_rdkit(result, mols)


def test_pairs_mode_threaded_gpu_options_match_rdkit():
    mols = _mols(["CCO", "CCN", "c1ccccc1", "c1ccc(O)cc1", "CC(C)O"])
    pairs = [(0, 1), (2, 3), (4, 0), (1, 4), (3, 2)]

    result = findMCS(
        mols,
        mode="pairs",
        pairs=pairs,
        batch_size=1,
        block_size=256,
        worker_threads=2,
        preprocessing_threads=2,
        executors_per_runner=1,
        gpu_ids=[],
    )

    assert result.pairs == tuple(pairs)
    assert result.used_gpu.any()
    _assert_matches_rdkit(result, mols)


def test_config_path_matches_rdkit_and_rejects_duplicate_execution_options():
    mols = _mols(["CCO", "CCN", "c1ccccc1", "c1ccc(O)cc1"])
    pairs = [(0, 1), (2, 3)]
    config = MCSConfig(
        batchSize=1,
        blockSize=64,
        workerThreads=1,
        preprocessingThreads=1,
        executorsPerRunner=1,
    )

    result = findMCS(mols, mode="pairs", pairs=pairs, config=config)

    assert result.pairs == tuple(pairs)
    assert result.used_gpu.any()
    _assert_matches_rdkit(result, mols)

    with pytest.raises(ValueError, match="config cannot be combined"):
        findMCS(mols, mode="pairs", pairs=pairs, config=config, block_size=256)


def test_all_pairs_default_is_upper_triangle_with_diagonal():
    mols = _mols(["CCO", "CCN", "c1ccccc1"])

    result = findMCS(mols)

    assert result.mode == "all_pairs"
    assert result.pairs == ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    _assert_matches_rdkit(result, mols)


def test_all_pairs_full_matrix_without_diagonal():
    mols = _mols(["CCO", "CCN", "c1ccccc1"])

    result = findMCS(mols, upper_triangle=False, include_diagonal=False)

    assert result.pairs == ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
    _assert_matches_rdkit(result, mols)


def test_paired_lists_mode_matches_rdkit():
    mols_a = _mols(["CCO", "c1ccccc1"])
    mols_b = _mols(["CCN", "c1ccc(O)cc1"])

    result = findMCS(mols_a, mode="paired_lists", mols_b=mols_b)

    assert result.mode == "paired_lists"
    assert result.pairs == ((0, 2), (1, 3))
    _assert_matches_rdkit(result, mols_a + mols_b)


def test_compare_and_ring_options_match_rdkit():
    mols = _mols(["C1CCCCC1", "CCCCCC", "c1ccccc1"])
    pairs = [(0, 1), (0, 2)]

    result = findMCS(
        mols,
        mode="pairs",
        pairs=pairs,
        atom_compare="any",
        bond_compare="any",
        ring_matches_ring_only=True,
    )

    _assert_matches_rdkit(result, mols, atom_compare="any", bond_compare="any", ring_matches_ring_only=True)


def test_invalid_mode_and_optional_arguments():
    mols = _mols(["CCO", "CCN"])

    with pytest.raises(ValueError, match="mode must be"):
        findMCS(mols, mode="wat")
    with pytest.raises(ValueError, match="pairs is required"):
        findMCS(mols, mode="pairs")
    with pytest.raises(ValueError, match="mols_b is only valid"):
        findMCS(mols, mode="pairs", pairs=[(0, 1)], mols_b=mols)
    with pytest.raises(ValueError, match="pairs is only valid"):
        findMCS(mols, pairs=[(0, 1)])
    with pytest.raises(ValueError, match="mols_b is required"):
        findMCS(mols, mode="paired_lists")
    with pytest.raises(ValueError, match="pairs is only valid"):
        findMCS(mols, mode="paired_lists", pairs=[(0, 1)], mols_b=mols)
    with pytest.raises(ValueError, match="same length"):
        findMCS(mols, mode="paired_lists", mols_b=mols[:1])
    with pytest.raises(ValueError, match="exactly two"):
        findMCS(mols, mode="pairs", pairs=[(0, 1, 2)])
    with pytest.raises(ValueError, match="Unsupported atom_compare"):
        findMCS(mols, mode="pairs", pairs=[(0, 1)], atom_compare="mass")
    with pytest.raises(ValueError, match="Unsupported bond_compare"):
        findMCS(mols, mode="pairs", pairs=[(0, 1)], bond_compare="shape")
    with pytest.raises(ValueError, match="blockSize"):
        findMCS(mols, mode="pairs", pairs=[(0, 1)], block_size=32)


def test_out_of_range_pair_raises_from_native_layer():
    mols = _mols(["CCO", "CCN"])

    with pytest.raises(RuntimeError, match="pair index out of range"):
        findMCS(mols, mode="pairs", pairs=[(0, 3)])
