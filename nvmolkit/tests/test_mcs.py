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

import gc

import numpy as np
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
    match_valences: bool = False,
    match_formal_charge: bool = False,
    atom_ring_matches_ring_only: bool = False,
    bond_ring_matches_ring_only: bool = False,
):
    params = rdFMCS.MCSParameters()
    params.MaximizeBonds = True
    params.AtomCompareParameters.MatchValences = match_valences
    params.AtomCompareParameters.MatchFormalCharge = match_formal_charge
    params.AtomCompareParameters.RingMatchesRingOnly = atom_ring_matches_ring_only
    params.BondCompareParameters.RingMatchesRingOnly = bond_ring_matches_ring_only

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


def _assert_result_storage(result, mol_table):
    num_results = len(result.pairs)
    assert result.num_atoms.shape == (num_results,)
    assert result.num_atoms.dtype == np.uint32
    assert result.num_bonds.shape == (num_results,)
    assert result.num_bonds.dtype == np.uint32
    assert result.canceled.shape == (num_results,)
    assert result.canceled.dtype == np.uint8
    assert result.atom_mapping.ndim == 2
    assert result.atom_mapping.shape[1] == 2
    assert result.atom_mapping.dtype == np.int32
    assert result.bond_mapping.ndim == 2
    assert result.bond_mapping.shape[1] == 2
    assert result.bond_mapping.dtype == np.int32
    assert result.atom_mapping_indptr.shape == (num_results + 1,)
    assert result.atom_mapping_indptr.dtype == np.int32
    assert result.bond_mapping_indptr.shape == (num_results + 1,)
    assert result.bond_mapping_indptr.dtype == np.int32
    assert result.atom_mapping_indptr[0] == 0
    assert result.bond_mapping_indptr[0] == 0
    assert np.all(np.diff(result.atom_mapping_indptr) >= 0)
    assert np.all(np.diff(result.bond_mapping_indptr) >= 0)
    assert result.atom_mapping_indptr[-1] == len(result.atom_mapping)
    assert result.bond_mapping_indptr[-1] == len(result.bond_mapping)

    for pair_idx, (idx_a, idx_b) in enumerate(result.pairs):
        item = result[pair_idx]
        assert item.pair == (idx_a, idx_b)
        assert item.atom_mapping.shape == (item.num_atoms, 2)
        assert item.bond_mapping.shape == (item.num_bonds, 2)

        if item.num_atoms:
            assert np.all((0 <= item.atom_mapping[:, 0]) & (item.atom_mapping[:, 0] < mol_table[idx_a].GetNumAtoms()))
            assert np.all((0 <= item.atom_mapping[:, 1]) & (item.atom_mapping[:, 1] < mol_table[idx_b].GetNumAtoms()))
            assert len(np.unique(item.atom_mapping[:, 0])) == item.num_atoms
            assert len(np.unique(item.atom_mapping[:, 1])) == item.num_atoms
        if item.num_bonds:
            assert np.all((0 <= item.bond_mapping[:, 0]) & (item.bond_mapping[:, 0] < mol_table[idx_a].GetNumBonds()))
            assert np.all((0 <= item.bond_mapping[:, 1]) & (item.bond_mapping[:, 1] < mol_table[idx_b].GetNumBonds()))
            assert len(np.unique(item.bond_mapping[:, 0])) == item.num_bonds
            assert len(np.unique(item.bond_mapping[:, 1])) == item.num_bonds

        atom_a_to_b = dict(item.atom_mapping.tolist())
        for bond_a_idx, bond_b_idx in item.bond_mapping:
            bond_a = mol_table[idx_a].GetBondWithIdx(int(bond_a_idx))
            bond_b = mol_table[idx_b].GetBondWithIdx(int(bond_b_idx))
            mapped_endpoints = {
                atom_a_to_b[bond_a.GetBeginAtomIdx()],
                atom_a_to_b[bond_a.GetEndAtomIdx()],
            }
            assert mapped_endpoints == {bond_b.GetBeginAtomIdx(), bond_b.GetEndAtomIdx()}


def _assert_matches_rdkit(result, mol_table, *, atom_compare="elements", bond_compare="order", **kwargs):
    params = _rdkit_params(atom_compare=atom_compare, bond_compare=bond_compare, **kwargs)
    _assert_result_storage(result, mol_table)
    for pair_idx, (idx_a, idx_b) in enumerate(result.pairs):
        rd_result = rdFMCS.FindMCS([mol_table[idx_a], mol_table[idx_b]], params)
        item = result[pair_idx]
        assert item.num_atoms == rd_result.numAtoms
        assert item.num_bonds == rd_result.numBonds
        assert not item.canceled


def test_pairs_mode_matches_rdkit_and_preserves_order():
    mols = _mols(["CCO", "CCN", "c1ccccc1", "c1ccc(O)cc1"])
    pairs = [(0, 1), (2, 3), (0, 2)]

    result = findMCS(mols, mode="pairs", pairs=pairs)

    assert result.mode == "pairs"
    assert result.pairs == tuple(pairs)
    assert len(result) == len(pairs)
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

    _assert_matches_rdkit(
        result,
        mols,
        atom_compare="any",
        bond_compare="any",
        atom_ring_matches_ring_only=True,
        bond_ring_matches_ring_only=True,
    )


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "pairs", "pairs": []},
        {"mode": "paired_lists", "mols_b": []},
        {"upper_triangle": False, "include_diagonal": False},
    ],
)
def test_empty_batches_have_well_formed_zero_length_storage(kwargs):
    result = findMCS([], **kwargs)

    assert len(result) == 0
    assert result.pairs == ()
    _assert_result_storage(result, [])
    assert result.atom_mapping.shape == (0, 2)
    assert result.bond_mapping.shape == (0, 2)
    np.testing.assert_array_equal(result.atom_mapping_indptr, [0])
    np.testing.assert_array_equal(result.bond_mapping_indptr, [0])


@pytest.mark.parametrize(
    "smiles_a,smiles_b,expected_atoms,expected_bonds",
    [
        ("", "CC", 0, 0),
        ("[Na+]", "[Cl-]", 0, 0),
        ("C", "C", 1, 0),
        ("CC", "C=C", 1, 0),
    ],
)
def test_zero_and_singleton_mappings_round_trip(smiles_a, smiles_b, expected_atoms, expected_bonds):
    mols = _mols([smiles_a, smiles_b])
    result = findMCS(mols, mode="pairs", pairs=[(0, 1)], require_gpu=True)

    _assert_result_storage(result, mols)
    item = result[0]
    assert (item.num_atoms, item.num_bonds) == (expected_atoms, expected_bonds)
    assert item.atom_mapping.shape == (expected_atoms, 2)
    assert item.bond_mapping.shape == (expected_bonds, 2)


def test_ragged_mapping_offsets_slicing_and_negative_indexing():
    mols = _mols(["CCO", "CCN", "c1ccccc1", "[Na+]", "[Cl-]"])
    pairs = [(0, 1), (2, 2), (3, 4), (0, 0)]
    result = findMCS(mols, mode="pairs", pairs=pairs, require_gpu=True)

    _assert_matches_rdkit(result, mols)
    np.testing.assert_array_equal(np.diff(result.atom_mapping_indptr), result.num_atoms)
    np.testing.assert_array_equal(np.diff(result.bond_mapping_indptr), result.num_bonds)
    assert result[-1].pair == pairs[-1]
    np.testing.assert_array_equal(result[-1].atom_mapping, result.atom_mapping[-3:])
    with pytest.raises(IndexError, match="MCS result index out of range"):
        _ = result[len(result)]
    with pytest.raises(IndexError, match="MCS result index out of range"):
        _ = result[-len(result) - 1]


def test_result_arrays_and_materialized_slices_keep_native_buffers_alive():
    mols = _mols(["CCO", "CCN", "c1ccccc1"])
    result = findMCS(mols, mode="pairs", pairs=[(0, 1), (1, 2)], require_gpu=True)
    item = result[0]
    num_atoms = result.num_atoms
    atom_mapping = result.atom_mapping
    atom_mapping_indptr = result.atom_mapping_indptr
    expected_num_atoms = num_atoms.copy()
    expected_atom_mapping = atom_mapping.copy()
    expected_item_mapping = item.atom_mapping.copy()

    del result
    gc.collect()

    np.testing.assert_array_equal(num_atoms, expected_num_atoms)
    np.testing.assert_array_equal(atom_mapping, expected_atom_mapping)
    np.testing.assert_array_equal(item.atom_mapping, expected_item_mapping)
    assert atom_mapping_indptr[-1] == len(atom_mapping)


@pytest.mark.parametrize(
    "atom_compare,bond_compare",
    [(atom, bond) for atom in ("any", "elements", "isotopes") for bond in ("any", "order", "order_exact")],
)
def test_atom_and_bond_comparison_matrix_matches_rdkit(atom_compare, bond_compare):
    mols = _mols(["CCO", "CNC", "C=C(O)C", "c1ccncc1", "C1CCNCC1", "[13CH3]CO"])
    pairs = [(0, 1), (0, 2), (3, 4), (0, 5), (2, 3)]
    result = findMCS(
        mols,
        mode="pairs",
        pairs=pairs,
        atom_compare=atom_compare,
        bond_compare=bond_compare,
        require_gpu=True,
    )

    _assert_matches_rdkit(result, mols, atom_compare=atom_compare, bond_compare=bond_compare)


@pytest.mark.parametrize(
    "atom_ring,bond_ring",
    [(True, False), (False, True), (True, True)],
)
def test_axis_specific_ring_matching_matches_rdkit(atom_ring, bond_ring):
    mols = _mols(["C1CCCCC1", "CCCCCC", "c1ccccc1", "C1=CCCCC1"])
    pairs = [(0, 1), (0, 2), (2, 3)]
    result = findMCS(
        mols,
        mode="pairs",
        pairs=pairs,
        atom_ring_matches_ring_only=atom_ring,
        bond_ring_matches_ring_only=bond_ring,
        require_gpu=True,
    )

    _assert_matches_rdkit(
        result,
        mols,
        atom_ring_matches_ring_only=atom_ring,
        bond_ring_matches_ring_only=bond_ring,
    )


@pytest.mark.parametrize(
    "smiles_a,smiles_b,kwargs,expected_atoms,expected_bonds",
    [
        ("[13CH3]CO", "CCO", {"atom_compare": "isotopes"}, 2, 1),
        ("C[NH2+]C", "CNC", {"match_formal_charge": True}, 1, 0),
        ("P(C)(C)C", "P(C)(C)(C)(C)C", {"match_valences": True}, 1, 0),
        ("CC.[Na+]", "CCC.[K+]", {}, 2, 1),
        ("c1ccccc1", "C1CCCCC1", {}, 6, 6),
        ("c1ccccc1", "C1CCCCC1", {"bond_compare": "order_exact"}, 1, 0),
        ("C1CCCCC1", "CCCCCC", {"atom_ring_matches_ring_only": True}, 0, 0),
        ("C1CCCCC1", "CCCCCC", {"bond_ring_matches_ring_only": True}, 1, 0),
        ("F[C@H](Cl)Br", "F[C@@H](Cl)Br", {}, 4, 3),
    ],
)
def test_special_molecule_semantics_match_rdkit(smiles_a, smiles_b, kwargs, expected_atoms, expected_bonds):
    mols = _mols([smiles_a, smiles_b])
    result = findMCS(mols, mode="pairs", pairs=[(0, 1)], require_gpu=True, **kwargs)

    _assert_matches_rdkit(result, mols, **kwargs)
    assert (result[0].num_atoms, result[0].num_bonds) == (expected_atoms, expected_bonds)


@pytest.mark.parametrize("block_size", [64, 128, 256, 512])
def test_supported_block_sizes_match_rdkit(block_size):
    mols = _mols(["CCOC(=O)N", "CCNC(=O)O"])
    result = findMCS(
        mols,
        mode="pairs",
        pairs=[(0, 1)],
        block_size=block_size,
        worker_threads=1,
        preprocessing_threads=1,
        executors_per_runner=1,
        require_gpu=True,
    )

    _assert_matches_rdkit(result, mols)


def test_higher_dispatch_tiers_and_pair_orientation_match_rdkit():
    cases = [
        (
            "CCCCCCCCCCCCCC(=O)NCc1ccc(C(=O)N[C@H](C(=O)O)[C@@H](C)CC)cc1",
            "CCO",
            True,
        ),
        (
            "C#C/C=C\\CCCCCCCCCCCCCC/C=C\\CCCCC(O)/C=C/CCCC#C[C@H](O)C#CCCCCCC/C=C/[C@@H](O)C#C",
            "CCO",
            False,
        ),
        (
            "NCCCC[C@@H](C=O)NC(=O)CC(CCCN)NC(=O)CC(CCCN)NC(=O)CC(CCCN)NC(=O)CC(CCCN)NC(=O)CC(CCCN)"
            "NC(=O)CC(CCCN)NC(=O)CC(N)CCCN",
            "CCN",
            True,
        ),
        (
            "CCCCCCCCCCCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO",
            "CCO",
            False,
        ),
    ]
    mols = []
    pairs = []
    for large_smiles, partner_smiles, large_first in cases:
        large, partner = _mols([large_smiles, partner_smiles])
        large_idx = len(mols)
        partner_idx = large_idx + 1
        mols.extend([large, partner])
        pairs.append((large_idx, partner_idx) if large_first else (partner_idx, large_idx))

    result = findMCS(mols, mode="pairs", pairs=pairs, require_gpu=True)

    _assert_matches_rdkit(result, mols)


def test_timeout_is_isolated_per_pair_and_partial_result_converts_cleanly():
    easy_a = "CCCCCCCCCCCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO"
    hard_a = (
        "C[C@H](NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](N)[C@@H](C)O)"
        "C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCCN)"
        "C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](Cc1ccccc1)C(=O)O"
    )
    hard_b = (
        "CSCC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)CNC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](Cc1ccccc1)NC(=O)"
        "[C@@H](CCC(N)=O)NC(=O)[C@@H](CCC(N)=O)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](CCCCN)NC(=O)[C@H]1CCCN1C(=O)"
        "[C@H](N)CCCN=C(N)N)C(N)=O"
    )
    mols = _mols([easy_a, "CCO", hard_a, hard_b])
    result = findMCS(
        mols,
        mode="pairs",
        pairs=[(0, 1), (2, 3)],
        timeout_seconds=1,
        batch_size=2,
        worker_threads=1,
        preprocessing_threads=1,
        executors_per_runner=1,
        require_gpu=True,
    )

    _assert_result_storage(result, mols)
    assert not result[0].canceled
    easy_rdkit = rdFMCS.FindMCS([mols[0], mols[1]], _rdkit_params())
    assert (result[0].num_atoms, result[0].num_bonds) == (easy_rdkit.numAtoms, easy_rdkit.numBonds)
    assert result[1].canceled
    assert result[1].num_atoms > 0
    assert result[1].num_bonds > 0


def test_rdkit_fallback_is_transparent_and_require_gpu_rejects_it():
    hypervalent = Chem.MolFromSmiles("[Fe](C)(C)(C)(C)(C)(C)(C)(C)C", sanitize=False)
    assert hypervalent.GetAtomWithIdx(0).GetDegree() == 9

    result = findMCS([hypervalent], mode="pairs", pairs=[(0, 0)])
    _assert_matches_rdkit(result, [hypervalent])
    assert (result[0].num_atoms, result[0].num_bonds) == (10, 9)

    with pytest.raises(RuntimeError, match="GPU MCS path unavailable"):
        findMCS([hypervalent], mode="pairs", pairs=[(0, 0)], require_gpu=True)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"connected_only": False}, "connected MCS only"),
        ({"maximize_bonds": False}, "MaximizeBonds only"),
        ({"atom_compare": "any_heavy_atom"}, "AnyHeavyAtom"),
        ({"match_isotope": True}, "matchIsotope"),
        ({"executors_per_runner": 0}, "executorsPerRunner"),
        ({"executors_per_runner": 9}, "executorsPerRunner"),
    ],
)
def test_unsupported_native_options_raise_clear_errors(kwargs, error):
    mols = _mols(["CN", "CO"])
    with pytest.raises(ValueError, match=error):
        findMCS(mols, mode="pairs", pairs=[(0, 1)], **kwargs)


def test_pair_conversion_rejects_negative_and_non_molecule_inputs():
    mols = _mols(["CC", "CO"])
    with pytest.raises(OverflowError):
        findMCS(mols, mode="pairs", pairs=[(-1, 0)])
    with pytest.raises(ValueError, match="Invalid molecule at index 1"):
        findMCS([mols[0], None], mode="pairs", pairs=[(0, 1)])
