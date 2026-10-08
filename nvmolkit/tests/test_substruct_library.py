# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import Future

import pytest
from rdkit import Chem
from test_substructure import PLAIN_QUERIES, PLAIN_QUERY_TARGETS, TEST_DATA_DIR, load_smarts_file

from nvmolkit.substruct_library import SubstructLibrary
from nvmolkit.substructure import SubstructSearchConfig


def test_finalize_makes_molecules_searchable():
    library = SubstructLibrary()
    assert library.addMols(Chem.MolFromSmiles(smiles) for smiles in ["CCO", "c1ccccc1", "CC(=O)O"]) == [0, 1, 2]
    assert len(library) == 0
    assert library.pendingSize == 3

    query = Chem.MolFromSmarts("[#6]")
    with pytest.raises(RuntimeError, match="finalized"):
        library.hasMatch(query).result()

    library.finalize()
    assert len(library) == 3
    assert library.pendingSize == 0
    assert library.maxConcurrentQueries >= 1
    assert library.getMatches(query).result() == [0, 1, 2]
    assert library.getMatches(query, maxResults=2).result() == [0, 1]
    assert library.countMatches(query).result() == 3
    assert library.hasMatch(Chem.MolFromSmarts("c")).result()

    futures = [library.hasMatch(Chem.MolFromSmarts(pattern)) for pattern in ["C", "N", "O"]]
    assert all(isinstance(future, Future) for future in futures)
    assert [future.result() for future in futures] == [True, False, True]

    assert library.addMol(Chem.MolFromSmiles("N")) == 3
    assert library.getMatches(query).result() == [0, 1, 2]
    library.finalize()
    assert len(library) == 4


class _FailingFinalizeNative:
    """Delegate to a native library but fail finalize, as running out of GPU memory would.

    Native rollback is covered by the C++ tests; this checks the wrapper's own executor handling.
    """

    def __init__(self, native):
        self._native = native

    def __getattr__(self, name):
        return getattr(self._native, name)

    def finalize(self):
        raise RuntimeError("simulated finalize failure")


def test_wrapper_keeps_serving_queries_after_a_failed_finalize():
    library = SubstructLibrary()
    library.addMols([Chem.MolFromSmiles("c1ccccc1")])
    library.finalize()
    library.addMol(Chem.MolFromSmiles("Oc1ccccc1"))

    native = library._native
    library._native = _FailingFinalizeNative(native)
    with pytest.raises(RuntimeError, match="simulated"):
        library.finalize()
    library._native = native

    query = Chem.MolFromSmarts("c")
    assert library.getMatches(query).result() == [0]
    library.finalize()
    assert library.getMatches(query).result() == [0, 1]


def test_future_callbacks_can_finalize_and_query():
    library = SubstructLibrary()
    library.addMol(Chem.MolFromSmiles("CCO"))
    library.finalize()
    query = Chem.MolFromSmarts("[#8]")
    finished = threading.Event()
    seen = []

    def grow_and_requery(_future):
        library.addMol(Chem.MolFromSmiles("OCCO"))
        library.finalize()
        seen.append(library.getMatchesSync(query))
        finished.set()

    library.getMatches(query).add_done_callback(grow_and_requery)
    assert finished.wait(timeout=60)
    assert seen == [[0, 1]]


def test_invalid_library_configuration():
    config = SubstructSearchConfig(gpuIds=[0, 0])
    with pytest.raises(ValueError, match="unique"):
        SubstructLibrary(config=config)


@pytest.mark.parametrize("query_smiles", PLAIN_QUERIES)
def test_plain_molecule_query_atoms_match_like_rdkit(query_smiles):
    library = SubstructLibrary()
    targets = [Chem.MolFromSmiles(smiles) for smiles in PLAIN_QUERY_TARGETS]
    library.addMols(targets)
    library.finalize()

    query = Chem.MolFromSmiles(query_smiles)
    expected = [index for index, target in enumerate(targets) if target.HasSubstructMatch(query)]
    assert library.getMatches(query).result() == expected


def test_chirality_is_ignored_except_in_smarts_queries():
    library = SubstructLibrary()
    library.addMols([Chem.MolFromSmiles(smiles) for smiles in ["C[C@H](O)CC", "C[C@@H](O)CC", "CCC"]])
    library.finalize()

    assert library.getMatchesSync(Chem.MolFromSmiles("C[C@H](O)CC")) == [0, 1]
    with pytest.raises(RuntimeError, match="chirality"):
        library.getMatchesSync(Chem.MolFromSmarts("[C@H](C)(O)CC"))


def test_sync_methods_return_the_future_results():
    library = SubstructLibrary()
    library.addMols(iter([Chem.MolFromSmiles(smiles) for smiles in ["CCO", "CCN", "OCCO"]]))
    library.finalize()

    query = Chem.MolFromSmarts("[OX2H]")
    assert library.getMatchesSync(query) == library.getMatches(query).result() == [0, 2]
    assert library.getMatchesSync(query, maxResults=1) == [0]
    assert library.countMatchesSync(query) == 2
    assert library.hasMatchSync(query)
    assert not library.hasMatchSync(Chem.MolFromSmarts("[Si]"))


def test_matches_rdkit_on_real_molecules_and_query_sets(one_hundred_mols):
    library = SubstructLibrary()
    assert library.addMols(one_hundred_mols) == list(range(len(one_hundred_mols)))
    library.finalize()

    queries, smarts = load_smarts_file(TEST_DATA_DIR / "SMARTS" / "rdkit_fragment_descriptors_supported.txt")
    futures = [(library.getMatches(query), library.getMatches(query, maxResults=3)) for query in queries]
    for query, pattern, (all_matches, limited) in zip(queries, smarts, futures):
        expected = [index for index, mol in enumerate(one_hundred_mols) if mol.HasSubstructMatch(query)]
        assert all_matches.result() == expected, pattern
        assert limited.result() == expected[:3], pattern
