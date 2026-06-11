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

"""Tests for GPU-accelerated substructure search."""

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from rdkit import Chem

from nvmolkit.substructure import (
    SubstructSearchConfig,
    countSubstructMatches,
    getSubstructMatches,
    hasSubstructMatch,
)


TEST_DATA_DIR = Path(__file__).parent.parent.parent / "tests" / "test_data"
MAX_ATOMS = 128
NUM_SMILES = 300


def get_rdkit_matches(target: Chem.Mol, query: Chem.Mol, uniquify: bool = False) -> list[tuple[int, ...]]:
    """Get RDKit substructure matches for comparison."""
    return list(target.GetSubstructMatches(query, uniquify=uniquify))


def matches_equal(gpu_matches: list, rdkit_matches: list[tuple[int, ...]]) -> bool:
    """Compare GPU matches to RDKit matches (order-independent).

    GPU matches can be numpy arrays or lists.
    """
    gpu_set = {tuple(m) for m in gpu_matches}
    rdkit_set = set(rdkit_matches)
    return gpu_set == rdkit_set


# =============================================================================
# Basic Tests
# =============================================================================


class TestBasicSubstructureSearch:
    """Basic substructure search functionality tests."""

    def test_single_target_single_query(self):
        """Test single target with single query."""
        targets = [Chem.MolFromSmiles("CCO")]
        queries = [Chem.MolFromSmarts("C")]

        results = getSubstructMatches(targets, queries)

        assert len(results) == 1
        assert len(results[0]) == 1

        rdkit_matches = get_rdkit_matches(targets[0], queries[0])
        assert len(results[0][0]) == len(rdkit_matches)

    def test_multiple_targets_single_query(self):
        """Test multiple targets with single query."""
        targets = [
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("CCCC"),
            Chem.MolFromSmiles("c1ccccc1"),
        ]
        queries = [Chem.MolFromSmarts("C")]

        results = getSubstructMatches(targets, queries)

        assert len(results) == 3
        for t_idx, target in enumerate(targets):
            rdkit_matches = get_rdkit_matches(target, queries[0])
            assert len(results[t_idx][0]) == len(rdkit_matches)

    def test_single_target_multiple_queries(self):
        """Test single target with multiple queries."""
        targets = [Chem.MolFromSmiles("CCO")]
        queries = [
            Chem.MolFromSmarts("C"),
            Chem.MolFromSmarts("O"),
            Chem.MolFromSmarts("CC"),
        ]

        results = getSubstructMatches(targets, queries)

        assert len(results) == 1
        assert len(results[0]) == 3

        for q_idx, query in enumerate(queries):
            rdkit_matches = get_rdkit_matches(targets[0], query)
            assert len(results[0][q_idx]) == len(rdkit_matches)

    def test_batch_all_to_all(self):
        """Test batch matching of multiple targets against multiple queries."""
        targets = [
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("c1ccccc1"),
            Chem.MolFromSmiles("c1ccc(O)cc1"),
            Chem.MolFromSmiles("CCN"),
        ]
        queries = [
            Chem.MolFromSmarts("C"),
            Chem.MolFromSmarts("O"),
            Chem.MolFromSmarts("c"),
            Chem.MolFromSmarts("N"),
        ]

        results = getSubstructMatches(targets, queries)

        assert len(results) == 4
        for t_idx, target in enumerate(targets):
            assert len(results[t_idx]) == 4
            for q_idx, query in enumerate(queries):
                rdkit_matches = get_rdkit_matches(target, query)
                assert len(results[t_idx][q_idx]) == len(rdkit_matches), f"Mismatch at target {t_idx}, query {q_idx}"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for substructure search."""

    def test_oversized_target_rdkit_fallback(self):
        """Test that oversized targets (>128 atoms) are handled via RDKit fallback."""
        # Create a large molecule with >128 atoms using a linear chain
        large_smiles = "C" * 200  # 200 carbons
        large_mol = Chem.MolFromSmiles(large_smiles)
        assert large_mol is not None
        assert large_mol.GetNumAtoms() > 128, "Test molecule should have >128 atoms"

        # Also include a regular-sized molecule
        small_mol = Chem.MolFromSmiles("CCO")

        targets = [large_mol, small_mol]
        queries = [Chem.MolFromSmarts("C"), Chem.MolFromSmarts("CC")]

        results = getSubstructMatches(targets, queries)

        assert len(results) == 2
        # Both targets should have results
        assert len(results[0]) == 2  # Large mol results for both queries
        assert len(results[1]) == 2  # Small mol results for both queries

        # Verify against RDKit
        for t_idx, target in enumerate(targets):
            for q_idx, query in enumerate(queries):
                rdkit_matches = get_rdkit_matches(target, query)
                assert len(results[t_idx][q_idx]) == len(rdkit_matches), (
                    f"Mismatch at target {t_idx}, query {q_idx}: GPU/fallback={len(results[t_idx][q_idx])}, RDKit={len(rdkit_matches)}"
                )

    def test_mixed_size_targets(self):
        """Test batch with both normal and oversized targets."""
        # Mix of normal and oversized targets
        targets = [
            Chem.MolFromSmiles("CCO"),  # Normal
            Chem.MolFromSmiles("C" * 150),  # Oversized
            Chem.MolFromSmiles("c1ccccc1"),  # Normal
        ]
        queries = [Chem.MolFromSmarts("C")]

        results = getSubstructMatches(targets, queries)

        assert len(results) == 3
        for t_idx, target in enumerate(targets):
            rdkit_matches = get_rdkit_matches(target, queries[0])
            assert len(results[t_idx][0]) == len(rdkit_matches)

    def test_buffer_overflow_rdkit_fallback(self):
        """Test that buffer overflow cases are handled via RDKit fallback.

        CCCCCC with CC query produces 10 non-unique matches but the buffer
        only holds ~6 (target atom count). RDKit fallback should provide all matches.
        """
        target = Chem.MolFromSmiles("CCCCCC")  # 6 atoms
        query = Chem.MolFromSmarts("CC")  # 2 atom query

        # Hexane with CC has 10 non-unique matches:
        # (0,1), (1,0), (1,2), (2,1), (2,3), (3,2), (3,4), (4,3), (4,5), (5,4)
        results = getSubstructMatches([target], [query])
        rdkit_matches = get_rdkit_matches(target, query)

        assert len(rdkit_matches) == 10, "RDKit should find 10 matches"
        assert len(results[0][0]) == 10, f"Should get all 10 matches via RDKit fallback, got {len(results[0][0])}"
        assert matches_equal(results[0][0], rdkit_matches), "Matches should be identical to RDKit results"

    @pytest.mark.parametrize("preprocessing_threads", [1, 2, -1])
    def test_disconnected_smarts_raises(self, preprocessing_threads: int):
        """Disconnected (fragment) SMARTS must raise rather than crash the process.

        Regression test for GH issue 203: with more than one preprocessing
        thread the validation error escaped the OpenMP region and terminated the
        process. The unsupported-query error must surface as a RuntimeError for
        every preprocessing-thread count.
        """
        targets = [Chem.MolFromSmiles(smi) for smi in ("CC", "CCC", "CCCC")]
        queries = [Chem.MolFromSmarts("C.C")]

        config = SubstructSearchConfig(preprocessingThreads=preprocessing_threads)

        with pytest.raises(RuntimeError, match="disconnected"):
            hasSubstructMatch(targets, queries, config)

    def test_no_match_possible(self):
        """Test when no match is possible."""
        targets = [Chem.MolFromSmiles("CCCC")]
        queries = [Chem.MolFromSmarts("N")]

        results = getSubstructMatches(targets, queries)

        assert len(results[0][0]) == 0

    def test_aromatic_vs_aliphatic(self):
        """Test aromatic vs aliphatic carbon matching."""
        targets = [Chem.MolFromSmiles("c1ccccc1")]  # Benzene (aromatic)
        queries = [Chem.MolFromSmarts("C")]  # Aliphatic carbon

        results = getSubstructMatches(targets, queries)

        rdkit_matches = get_rdkit_matches(targets[0], queries[0])
        assert len(results[0][0]) == len(rdkit_matches)
        assert len(results[0][0]) == 0  # Aromatic c doesn't match aliphatic C

    def test_empty_targets(self):
        """Test with empty target list."""
        targets = []
        queries = [Chem.MolFromSmarts("C")]

        results = getSubstructMatches(targets, queries)

        assert len(results) == 0

    def test_empty_queries(self):
        """Test with empty query list."""
        targets = [Chem.MolFromSmiles("CCO")]
        queries = []

        results = getSubstructMatches(targets, queries)

        assert len(results) == 1
        assert len(results[0]) == 0

    def test_multi_atom_query(self):
        """Test multi-atom query patterns."""
        targets = [Chem.MolFromSmiles("CCO")]
        queries = [Chem.MolFromSmarts("CC")]

        results = getSubstructMatches(targets, queries)

        rdkit_matches = get_rdkit_matches(targets[0], queries[0])
        assert len(results[0][0]) == len(rdkit_matches)

    def test_three_atom_query(self):
        """Test 3-atom query patterns."""
        targets = [Chem.MolFromSmiles("CCOCC")]  # Diethyl ether
        queries = [Chem.MolFromSmarts("COC")]

        results = getSubstructMatches(targets, queries)

        rdkit_matches = get_rdkit_matches(targets[0], queries[0])
        assert len(results[0][0]) == len(rdkit_matches)


# =============================================================================
# Compound Query Tests (OR/NOT)
# =============================================================================


class TestCompoundQueries:
    """Tests for compound SMARTS queries with OR/NOT operators."""

    def test_or_query_matches_both_types(self):
        """Test OR query [C,N] matches both carbons and nitrogens."""
        targets = [Chem.MolFromSmiles("CCN")]
        queries = [Chem.MolFromSmarts("[C,N]")]

        results = getSubstructMatches(targets, queries)

        rdkit_matches = get_rdkit_matches(targets[0], queries[0])
        assert len(results[0][0]) == len(rdkit_matches)
        assert len(results[0][0]) == 3  # 2 carbons + 1 nitrogen

    def test_not_query_excludes_atom(self):
        """Test NOT query [!C] excludes carbon."""
        targets = [Chem.MolFromSmiles("CCO")]
        queries = [Chem.MolFromSmarts("[!C]")]

        results = getSubstructMatches(targets, queries)

        rdkit_matches = get_rdkit_matches(targets[0], queries[0])
        assert len(results[0][0]) == len(rdkit_matches)
        assert len(results[0][0]) == 1  # Only oxygen

    def test_three_way_or_query(self):
        """Test 3-way OR: [C,N,O]."""
        targets = [Chem.MolFromSmiles("CCNO")]
        queries = [Chem.MolFromSmarts("[C,N,O]")]

        results = getSubstructMatches(targets, queries)

        rdkit_matches = get_rdkit_matches(targets[0], queries[0])
        assert len(results[0][0]) == len(rdkit_matches)
        assert len(results[0][0]) == 4  # All atoms match


# =============================================================================
# Ring Queries
# =============================================================================


class TestRingQueries:
    """Tests for ring-related SMARTS queries."""

    def test_any_ring_membership(self):
        """Test [R] any ring membership query."""
        targets = [
            Chem.MolFromSmiles("C1CCC1C"),  # Cyclobutane with methyl
            Chem.MolFromSmiles("CCCCC"),  # Pentane (no rings)
        ]
        queries = [Chem.MolFromSmarts("[R]")]

        results = getSubstructMatches(targets, queries)

        # Cyclobutane: 4 ring atoms
        rdkit_matches_0 = get_rdkit_matches(targets[0], queries[0])
        assert len(results[0][0]) == len(rdkit_matches_0)

        # Pentane: no ring atoms
        rdkit_matches_1 = get_rdkit_matches(targets[1], queries[0])
        assert len(results[1][0]) == len(rdkit_matches_1)
        assert len(results[1][0]) == 0

    def test_aromatic_ring_pattern(self):
        """Test aromatic ring patterns."""
        targets = [Chem.MolFromSmiles("c1ccccc1")]  # Benzene
        queries = [Chem.MolFromSmarts("[c,n]")]

        results = getSubstructMatches(targets, queries)

        rdkit_matches = get_rdkit_matches(targets[0], queries[0])
        assert len(results[0][0]) == len(rdkit_matches)


# =============================================================================
# Degree and Connectivity Queries
# =============================================================================


class TestDegreeQueries:
    """Tests for degree and connectivity SMARTS queries."""

    def test_degree_query_d3(self):
        """Test [D3] degree query."""
        targets = [
            Chem.MolFromSmiles("CC(C)C"),  # Isobutane - central C has degree 3
            Chem.MolFromSmiles("CCC"),  # Propane - no degree 3 atoms
        ]
        queries = [Chem.MolFromSmarts("[D3]")]

        results = getSubstructMatches(targets, queries)

        for t_idx, target in enumerate(targets):
            rdkit_matches = get_rdkit_matches(target, queries[0])
            assert len(results[t_idx][0]) == len(rdkit_matches)

    def test_total_connectivity_x4(self):
        """Test [X4] total connectivity query."""
        targets = [
            Chem.MolFromSmiles("CC"),  # Ethane - carbons have X4
            Chem.MolFromSmiles("C=C"),  # Ethene - carbons have X3
        ]
        queries = [Chem.MolFromSmarts("[X4]")]

        results = getSubstructMatches(targets, queries)

        for t_idx, target in enumerate(targets):
            rdkit_matches = get_rdkit_matches(target, queries[0])
            assert len(results[t_idx][0]) == len(rdkit_matches)


# =============================================================================
# maxMatches Configuration Tests
# =============================================================================


class TestMaxMatchesConfig:
    """Tests for maxMatches configuration parameter."""

    def test_max_matches_zero_unlimited(self):
        """Test maxMatches=0 means unlimited (like RDKit)."""
        targets = [Chem.MolFromSmiles("CCO")]
        queries = [Chem.MolFromSmarts("C")]

        config = SubstructSearchConfig()
        config.maxMatches = 0  # Unlimited

        results = getSubstructMatches(targets, queries, config)

        # With maxMatches=0 (unlimited), all matches are stored
        assert len(results[0][0]) == 2

    def test_max_matches_limited(self):
        """Test maxMatches limits stored matches."""
        targets = [Chem.MolFromSmiles("CCCC")]
        queries = [Chem.MolFromSmarts("C")]

        config = SubstructSearchConfig()
        config.maxMatches = 2

        results = getSubstructMatches(targets, queries, config)

        # Only 2 matches stored
        assert len(results[0][0]) == 2

    def test_max_matches_greater_than_actual(self):
        """Test maxMatches greater than actual doesn't affect results."""
        targets = [Chem.MolFromSmiles("CC")]
        queries = [Chem.MolFromSmarts("C")]

        config = SubstructSearchConfig()
        config.maxMatches = 10

        results = getSubstructMatches(targets, queries, config)

        # All 2 matches stored
        assert len(results[0][0]) == 2


# =============================================================================
# Uniquify Tests
# =============================================================================


class TestUniquify:
    """Tests for the uniquify option that removes duplicate matches."""

    def test_uniquify_cyclohexane_ccc(self):
        """Test uniquify on cyclohexane with CCC query (classic example)."""
        target = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
        query = Chem.MolFromSmarts("CCC")

        # Without uniquify: 12 matches
        config_no_uniquify = SubstructSearchConfig()
        config_no_uniquify.uniquify = False
        results_no_uniquify = getSubstructMatches([target], [query], config_no_uniquify)

        rdkit_non_unique = list(target.GetSubstructMatches(query, uniquify=False))
        assert len(results_no_uniquify[0][0]) == len(rdkit_non_unique)
        assert len(results_no_uniquify[0][0]) == 12

        # With uniquify: 6 matches
        config_uniquify = SubstructSearchConfig()
        config_uniquify.uniquify = True
        results_uniquify = getSubstructMatches([target], [query], config_uniquify)

        rdkit_unique = list(target.GetSubstructMatches(query, uniquify=True))
        assert len(results_uniquify[0][0]) == len(rdkit_unique)
        assert len(results_uniquify[0][0]) == 6

    def test_uniquify_hexane_cc(self):
        """Test uniquify on hexane with CC query."""
        target = Chem.MolFromSmiles("CCCCCC")  # hexane
        query = Chem.MolFromSmarts("CC")

        # Verify against RDKit
        rdkit_non_unique = list(target.GetSubstructMatches(query, uniquify=False))
        rdkit_unique = list(target.GetSubstructMatches(query, uniquify=True))

        # Without uniquify
        config_no_uniquify = SubstructSearchConfig()
        config_no_uniquify.uniquify = False
        results_no_uniquify = getSubstructMatches([target], [query], config_no_uniquify)
        assert len(results_no_uniquify[0][0]) == len(rdkit_non_unique)

        # With uniquify
        config_uniquify = SubstructSearchConfig()
        config_uniquify.uniquify = True
        results_uniquify = getSubstructMatches([target], [query], config_uniquify)
        assert len(results_uniquify[0][0]) == len(rdkit_unique)

    def test_uniquify_symmetric_query(self):
        """Test uniquify with symmetric query COC on diethyl ether."""
        target = Chem.MolFromSmiles("CCOCC")  # diethyl ether
        query = Chem.MolFromSmarts("COC")

        # Without uniquify
        config_no_uniquify = SubstructSearchConfig()
        config_no_uniquify.uniquify = False
        results_no_uniquify = getSubstructMatches([target], [query], config_no_uniquify)

        rdkit_non_unique = list(target.GetSubstructMatches(query, uniquify=False))
        assert len(results_no_uniquify[0][0]) == len(rdkit_non_unique)

        # With uniquify
        config_uniquify = SubstructSearchConfig()
        config_uniquify.uniquify = True
        results_uniquify = getSubstructMatches([target], [query], config_uniquify)

        rdkit_unique = list(target.GetSubstructMatches(query, uniquify=True))
        assert len(results_uniquify[0][0]) == len(rdkit_unique)

    def test_uniquify_no_effect_asymmetric(self):
        """Test that uniquify has no effect on asymmetric query."""
        target = Chem.MolFromSmiles("CCOCC")  # diethyl ether
        query = Chem.MolFromSmarts("CCO")  # asymmetric

        config_no_uniquify = SubstructSearchConfig()
        config_no_uniquify.uniquify = False
        results_no_uniquify = getSubstructMatches([target], [query], config_no_uniquify)

        config_uniquify = SubstructSearchConfig()
        config_uniquify.uniquify = True
        results_uniquify = getSubstructMatches([target], [query], config_uniquify)

        # Asymmetric queries should have same count
        assert len(results_uniquify[0][0]) == len(results_no_uniquify[0][0])

    def test_uniquify_single_atom_query(self):
        """Test that uniquify has no effect on single atom queries."""
        target = Chem.MolFromSmiles("CCCC")
        query = Chem.MolFromSmarts("C")

        config_no_uniquify = SubstructSearchConfig()
        config_no_uniquify.uniquify = False
        results_no_uniquify = getSubstructMatches([target], [query], config_no_uniquify)

        config_uniquify = SubstructSearchConfig()
        config_uniquify.uniquify = True
        results_uniquify = getSubstructMatches([target], [query], config_uniquify)

        # Single atom queries can't have duplicates
        assert len(results_uniquify[0][0]) == len(results_no_uniquify[0][0])
        assert len(results_uniquify[0][0]) == 4

    def test_uniquify_batch(self):
        """Test uniquify with batch of molecules and queries."""
        targets = [
            Chem.MolFromSmiles("C1CCCCC1"),  # cyclohexane
            Chem.MolFromSmiles("CCOCC"),  # diethyl ether
            Chem.MolFromSmiles("c1ccccc1"),  # benzene
        ]
        queries = [
            Chem.MolFromSmarts("CC"),
            Chem.MolFromSmarts("CCC"),
        ]

        config = SubstructSearchConfig()
        config.uniquify = True

        results = getSubstructMatches(targets, queries, config)

        # Verify each pair matches RDKit with uniquify=True
        for t_idx, target in enumerate(targets):
            for q_idx, query in enumerate(queries):
                rdkit_matches = list(target.GetSubstructMatches(query, uniquify=True))
                assert len(results[t_idx][q_idx]) == len(rdkit_matches), f"Mismatch at target {t_idx}, query {q_idx}"

    def test_uniquify_config_property(self):
        """Test that uniquify config property can be set."""
        config = SubstructSearchConfig()
        assert config.uniquify is False

        config.uniquify = True
        assert config.uniquify is True

        config.uniquify = False
        assert config.uniquify is False


# =============================================================================
# hasSubstructMatch Tests
# =============================================================================


class TestHasSubstructMatch:
    """Tests for hasSubstructMatch boolean API."""

    def test_basic_has_match(self):
        """Test basic hasSubstructMatch functionality."""
        targets = [
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("CCCC"),
            Chem.MolFromSmiles("c1ccccc1"),
        ]
        queries = [
            Chem.MolFromSmarts("C"),
            Chem.MolFromSmarts("O"),
            Chem.MolFromSmarts("N"),
        ]

        results = hasSubstructMatch(targets, queries)

        assert results.shape == (3, 3)

        # CCO contains C and O, but not N
        assert results[0, 0] == 1  # CCO has C
        assert results[0, 1] == 1  # CCO has O
        assert results[0, 2] == 0  # CCO has no N

        # CCCC contains C, but not O or N
        assert results[1, 0] == 1  # CCCC has C
        assert results[1, 1] == 0  # CCCC has no O
        assert results[1, 2] == 0  # CCCC has no N

        # Benzene has aromatic c, not aliphatic C, O, or N
        assert results[2, 0] == 0  # benzene has no aliphatic C
        assert results[2, 1] == 0  # benzene has no O
        assert results[2, 2] == 0  # benzene has no N

    def test_has_match_multi_atom_query(self):
        """Test hasSubstructMatch with multi-atom queries."""
        targets = [
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("CCC"),
        ]
        queries = [
            Chem.MolFromSmarts("CO"),
            Chem.MolFromSmarts("CC"),
        ]

        results = hasSubstructMatch(targets, queries)

        # CCO contains CO and CC
        assert results[0, 0] == 1
        assert results[0, 1] == 1

        # CCC contains CC but not CO
        assert results[1, 0] == 0
        assert results[1, 1] == 1

    def test_has_match_empty_inputs(self):
        """Test hasSubstructMatch with empty inputs."""
        targets = []
        queries = [Chem.MolFromSmarts("C")]

        results = hasSubstructMatch(targets, queries)

        assert results.shape == (0, 1)


# =============================================================================
# countSubstructMatches Tests
# =============================================================================


class TestCountSubstructMatch:
    """Tests for countSubstructMatches count API."""

    def test_basic_count(self):
        """Test basic countSubstructMatches functionality."""
        targets = [
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("CCCC"),
            Chem.MolFromSmiles("c1ccccc1"),
        ]
        queries = [
            Chem.MolFromSmarts("C"),
            Chem.MolFromSmarts("O"),
            Chem.MolFromSmarts("N"),
        ]

        results = countSubstructMatches(targets, queries)

        assert results.shape == (3, 3)

        for t_idx, target in enumerate(targets):
            for q_idx, query in enumerate(queries):
                rdkit_count = len(target.GetSubstructMatches(query, uniquify=False))
                assert results[t_idx, q_idx] == rdkit_count, (
                    f"Mismatch at target {t_idx}, query {q_idx}: GPU={results[t_idx, q_idx]}, RDKit={rdkit_count}"
                )

        assert (results == 0).any(), "Test data should include zero-count pairs"
        assert (results > 0).any(), "Test data should include nonzero-count pairs"
        assert (results > 1).any(), "Test data should include multi-match pairs"

    def test_count_matches_get_length(self):
        """Test that count results match len(getSubstructMatches) results."""
        targets = [
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("CCOCC"),
            Chem.MolFromSmiles("c1ccc(O)cc1"),
        ]
        queries = [
            Chem.MolFromSmarts("C"),
            Chem.MolFromSmarts("CC"),
            Chem.MolFromSmarts("O"),
        ]

        counts = countSubstructMatches(targets, queries)
        matches = getSubstructMatches(targets, queries)

        for t_idx in range(len(targets)):
            for q_idx in range(len(queries)):
                assert counts[t_idx, q_idx] == len(matches[t_idx][q_idx]), (
                    f"Count/get mismatch at target {t_idx}, query {q_idx}"
                )

    def test_count_multi_atom_query(self):
        """Test countSubstructMatches with multi-atom queries."""
        targets = [
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("CCC"),
        ]
        queries = [
            Chem.MolFromSmarts("CO"),
            Chem.MolFromSmarts("CC"),
        ]

        results = countSubstructMatches(targets, queries)

        for t_idx, target in enumerate(targets):
            for q_idx, query in enumerate(queries):
                rdkit_count = len(target.GetSubstructMatches(query, uniquify=False))
                assert results[t_idx, q_idx] == rdkit_count

    def test_count_no_match(self):
        """Test countSubstructMatches when no match exists."""
        targets = [Chem.MolFromSmiles("CCCC")]
        queries = [Chem.MolFromSmarts("N")]

        results = countSubstructMatches(targets, queries)

        assert results[0, 0] == 0

    def test_count_empty_targets(self):
        """Test countSubstructMatches with empty targets."""
        targets = []
        queries = [Chem.MolFromSmarts("C")]

        results = countSubstructMatches(targets, queries)

        assert results.shape == (0, 1)

    def test_count_empty_queries(self):
        """Test countSubstructMatches with empty queries."""
        targets = [Chem.MolFromSmiles("CCO")]
        queries = []

        results = countSubstructMatches(targets, queries)

        assert results.shape == (1, 0)

    def test_count_symmetric_query(self):
        """Test count with symmetric query that produces non-unique matches."""
        target = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
        query = Chem.MolFromSmarts("CCC")

        results = countSubstructMatches([target], [query])
        rdkit_count = len(target.GetSubstructMatches(query, uniquify=False))

        assert rdkit_count == 12
        assert results[0, 0] == rdkit_count

    def test_count_batch(self):
        """Test countSubstructMatches with batch of molecules."""
        targets = [
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("c1ccccc1"),
            Chem.MolFromSmiles("CCN"),
            Chem.MolFromSmiles("CCOCC"),
        ]
        queries = [
            Chem.MolFromSmarts("C"),
            Chem.MolFromSmarts("O"),
            Chem.MolFromSmarts("c"),
        ]

        results = countSubstructMatches(targets, queries)

        assert results.shape == (4, 3)
        for t_idx, target in enumerate(targets):
            for q_idx, query in enumerate(queries):
                rdkit_count = len(target.GetSubstructMatches(query, uniquify=False))
                assert results[t_idx, q_idx] == rdkit_count, f"Mismatch at target {t_idx}, query {q_idx}"


# =============================================================================
# SubstructSearchConfig Tests
# =============================================================================


class TestSubstructSearchConfig:
    """Tests for SubstructSearchConfig parameters."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SubstructSearchConfig()

        assert config.batchSize == 1024
        assert config.workerThreads == -1  # -1 = autoselect
        assert config.preprocessingThreads == -1  # -1 = autoselect
        assert config.maxMatches == 0  # 0 = unlimited (like RDKit)
        assert config.uniquify is False

    def test_config_modification(self):
        """Test configuration parameter modification."""
        config = SubstructSearchConfig()
        config.batchSize = 512
        config.workerThreads = 4
        config.preprocessingThreads = 8
        config.maxMatches = 100

        assert config.batchSize == 512
        assert config.workerThreads == 4
        assert config.preprocessingThreads == 8
        assert config.maxMatches == 100

    def test_gpu_ids_property(self):
        """Test gpuIds property get/set."""
        config = SubstructSearchConfig()

        # Default is empty
        assert config.gpuIds == []

        # Set via list
        config.gpuIds = [0, 1]
        assert config.gpuIds == [0, 1]

        # Set via tuple
        config.gpuIds = (2, 3)
        assert config.gpuIds == [2, 3]


# =============================================================================
# Integration Tests - Validate Against RDKit
# =============================================================================


class TestRDKitValidation:
    """Integration tests validating GPU results against RDKit."""

    @pytest.mark.parametrize(
        "smiles,smarts",
        [
            ("CCO", "C"),
            ("CCO", "O"),
            ("CCO", "CC"),
            ("c1ccccc1", "c"),
            ("c1ccccc1", "[c,n]"),
            ("CCN", "[C,N]"),
            ("CCNO", "[!C]"),
            ("CC(C)C", "[D3]"),
            ("C1CCCCC1", "[R]"),
        ],
    )
    def test_matches_rdkit(self, smiles: str, smarts: str):
        """Validate GPU matches against RDKit for various patterns."""
        target = Chem.MolFromSmiles(smiles)
        query = Chem.MolFromSmarts(smarts)

        results = getSubstructMatches([target], [query])
        rdkit_matches = get_rdkit_matches(target, query)

        assert len(results[0][0]) == len(rdkit_matches), (
            f"Count mismatch for {smiles} with {smarts}: GPU={len(results[0][0])}, RDKit={len(rdkit_matches)}"
        )

        if len(results[0][0]) > 0:
            assert matches_equal(results[0][0], rdkit_matches), f"Mapping mismatch for {smiles} with {smarts}"


class TestLargerMolecules:
    """Tests with larger, more complex molecules."""

    def test_caffeine(self):
        """Test substructure search on caffeine."""
        caffeine = Chem.MolFromSmiles("Cn1cnc2c1c(=O)n(c(=O)n2C)C")
        queries = [
            Chem.MolFromSmarts("c"),
            Chem.MolFromSmarts("N"),
            Chem.MolFromSmarts("C"),
        ]

        results = getSubstructMatches([caffeine], queries)

        for q_idx, query in enumerate(queries):
            rdkit_matches = get_rdkit_matches(caffeine, query)
            assert len(results[0][q_idx]) == len(rdkit_matches)

    def test_batch_with_size_limited_mols(self, size_limited_mols):
        """Test batch processing with fixture molecules."""
        if len(size_limited_mols) == 0:
            pytest.skip("No molecules in fixture")

        targets = size_limited_mols[:10]  # Use first 10
        queries = [
            Chem.MolFromSmarts("C"),
            Chem.MolFromSmarts("N"),
            Chem.MolFromSmarts("O"),
        ]

        results = getSubstructMatches(targets, queries)

        assert len(results) == len(targets)
        for t_idx, target in enumerate(targets):
            assert len(results[t_idx]) == len(queries)
            for q_idx, query in enumerate(queries):
                rdkit_matches = get_rdkit_matches(target, query)
                assert len(results[t_idx][q_idx]) == len(rdkit_matches), f"Mismatch at target {t_idx}, query {q_idx}"


# =============================================================================
# Integration Tests - Large Scale Dataset Validation
# =============================================================================


def load_smiles_file(filepath: Path, max_count: int = NUM_SMILES, max_atoms: int = MAX_ATOMS) -> list[Chem.Mol]:
    """Load molecules from a SMILES file, filtering by atom count."""
    mols = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smiles = line.split()[0] if " " in line or "\t" in line else line
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and mol.GetNumAtoms() <= max_atoms:
                mols.append(mol)
                if len(mols) >= max_count:
                    break
    return mols


def load_smarts_file(filepath: Path) -> tuple[list[Chem.Mol], list[str]]:
    """Load query molecules from a SMARTS file.

    Returns:
        Tuple of (query_mols, smarts_strings)
    """
    queries = []
    smarts_strings = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            mol = Chem.MolFromSmarts(line)
            if mol is not None:
                queries.append(mol)
                smarts_strings.append(line)
    return queries, smarts_strings


@dataclass
class MismatchInfo:
    """Details about a single mismatch between GPU and RDKit results."""

    target_idx: int
    query_idx: int
    target_smiles: str
    query_smarts: str
    gpu_count: int
    rdkit_count: int
    is_count_mismatch: bool  # True = count mismatch, False = mapping mismatch


@dataclass
class ValidationResult:
    """Result of validating GPU results against RDKit."""

    total_pairs: int
    count_mismatches: list[MismatchInfo]
    mapping_mismatches: list[MismatchInfo]


def validate_against_rdkit(
    targets: list[Chem.Mol],
    queries: list[Chem.Mol],
    results: list[list[list[list[int]]]],
    query_smarts: list[str] | None = None,
) -> ValidationResult:
    """Validate GPU results against RDKit.

    Args:
        targets: List of target molecules
        queries: List of query molecules
        results: GPU match results
        query_smarts: Optional list of SMARTS strings for better error reporting

    Returns:
        ValidationResult with detailed mismatch information
    """
    total_pairs = 0
    count_mismatches = []
    mapping_mismatches = []

    for t_idx, target in enumerate(targets):
        target_smiles = Chem.MolToSmiles(target)
        for q_idx, query in enumerate(queries):
            total_pairs += 1
            rdkit_matches = get_rdkit_matches(target, query)
            gpu_matches = results[t_idx][q_idx]

            smarts_str = query_smarts[q_idx] if query_smarts else Chem.MolToSmarts(query)

            if len(gpu_matches) != len(rdkit_matches):
                count_mismatches.append(
                    MismatchInfo(
                        target_idx=t_idx,
                        query_idx=q_idx,
                        target_smiles=target_smiles,
                        query_smarts=smarts_str,
                        gpu_count=len(gpu_matches),
                        rdkit_count=len(rdkit_matches),
                        is_count_mismatch=True,
                    )
                )
            elif len(gpu_matches) > 0 and not matches_equal(gpu_matches, rdkit_matches):
                mapping_mismatches.append(
                    MismatchInfo(
                        target_idx=t_idx,
                        query_idx=q_idx,
                        target_smiles=target_smiles,
                        query_smarts=smarts_str,
                        gpu_count=len(gpu_matches),
                        rdkit_count=len(rdkit_matches),
                        is_count_mismatch=False,
                    )
                )

    return ValidationResult(total_pairs, count_mismatches, mapping_mismatches)


@pytest.mark.long
class TestIntegrationChemblSmarts:
    """Integration tests using ChEMBL molecules and SMARTS datasets."""

    @pytest.fixture
    def chembl_mols(self) -> list[Chem.Mol]:
        """Load ChEMBL molecules from test data."""
        smiles_path = TEST_DATA_DIR / "chembl_1k.smi"
        if not smiles_path.exists():
            pytest.skip(f"Test data not found: {smiles_path}")
        return load_smiles_file(smiles_path)

    @pytest.mark.parametrize(
        "smarts_file",
        [
            "rdkit_fragment_descriptors_supported.txt",
            "pwalters_alert_collection_supported.txt",
            "openbabel_functional_groups_supported.txt",
        ],
    )
    def test_chembl_vs_smarts_dataset(self, chembl_mols: list[Chem.Mol], smarts_file: str):
        """Test ChEMBL molecules against SMARTS dataset files."""
        smarts_path = TEST_DATA_DIR / "SMARTS" / smarts_file
        if not smarts_path.exists():
            pytest.skip(f"SMARTS file not found: {smarts_path}")

        queries, smarts_strings = load_smarts_file(smarts_path)
        if not queries:
            pytest.skip(f"No valid queries in {smarts_file}")

        config = SubstructSearchConfig()
        config.batchSize = 128

        results = getSubstructMatches(chembl_mols, queries, config)

        assert len(results) == len(chembl_mols)
        for t_idx in range(len(chembl_mols)):
            assert len(results[t_idx]) == len(queries)

        validation = validate_against_rdkit(chembl_mols, queries, results, smarts_strings)

        if validation.count_mismatches:
            details = "\n".join(
                f"  [{m.target_idx},{m.query_idx}] {m.target_smiles} vs {m.query_smarts}: "
                f"GPU={m.gpu_count}, RDKit={m.rdkit_count}"
                for m in validation.count_mismatches[:20]  # Show first 20
            )
            more = (
                f"\n  ... and {len(validation.count_mismatches) - 20} more"
                if len(validation.count_mismatches) > 20
                else ""
            )
            pytest.fail(
                f"Count mismatches: {len(validation.count_mismatches)}/{validation.total_pairs}\n{details}{more}"
            )

        if validation.mapping_mismatches:
            details = "\n".join(
                f"  [{m.target_idx},{m.query_idx}] {m.target_smiles} vs {m.query_smarts}"
                for m in validation.mapping_mismatches[:20]
            )
            more = (
                f"\n  ... and {len(validation.mapping_mismatches) - 20} more"
                if len(validation.mapping_mismatches) > 20
                else ""
            )
            pytest.fail(
                f"Mapping mismatches: {len(validation.mapping_mismatches)}/{validation.total_pairs}\n{details}{more}"
            )

    def test_chembl_basic_queries(self, chembl_mols: list[Chem.Mol]):
        """Test ChEMBL molecules against basic SMARTS queries."""
        smarts_strings = [
            "[CX3]=[OX1]",  # Carbonyl
            "[NX3;H2]",  # Primary amine
            "[OX2H]",  # Hydroxyl
            "c1ccccc1",  # Benzene ring
            "[#7]",  # Any nitrogen
        ]
        queries = [Chem.MolFromSmarts(s) for s in smarts_strings]

        config = SubstructSearchConfig()
        config.batchSize = 128

        results = getSubstructMatches(chembl_mols, queries, config)

        validation = validate_against_rdkit(chembl_mols, queries, results, smarts_strings)

        if validation.count_mismatches:
            details = "\n".join(
                f"  [{m.target_idx},{m.query_idx}] {m.target_smiles} vs {m.query_smarts}: "
                f"GPU={m.gpu_count}, RDKit={m.rdkit_count}"
                for m in validation.count_mismatches[:20]
            )
            pytest.fail(f"Count mismatches: {len(validation.count_mismatches)}/{validation.total_pairs}\n{details}")

    def test_chembl_has_substruct_match(self, chembl_mols: list[Chem.Mol]):
        """Test hasSubstructMatch on ChEMBL molecules."""
        queries = [
            Chem.MolFromSmarts("[CX3]=[OX1]"),
            Chem.MolFromSmarts("[NX3]"),
            Chem.MolFromSmarts("[OX2H]"),
        ]

        config = SubstructSearchConfig()
        config.batchSize = 128

        results = hasSubstructMatch(chembl_mols, queries, config)

        assert results.shape == (len(chembl_mols), len(queries))
        for t_idx, target in enumerate(chembl_mols):
            for q_idx, query in enumerate(queries):
                rdkit_has_match = target.HasSubstructMatch(query)
                assert bool(results[t_idx, q_idx]) == rdkit_has_match, f"Mismatch at target {t_idx}, query {q_idx}"

    def test_chembl_count_substruct_matches(self, chembl_mols: list[Chem.Mol]):
        """Test countSubstructMatches on ChEMBL molecules."""
        queries = [
            Chem.MolFromSmarts("[CX3]=[OX1]"),
            Chem.MolFromSmarts("[NX3]"),
            Chem.MolFromSmarts("[OX2H]"),
        ]

        config = SubstructSearchConfig()
        config.batchSize = 128

        results = countSubstructMatches(chembl_mols, queries, config)

        assert results.shape == (len(chembl_mols), len(queries))
        for t_idx, target in enumerate(chembl_mols):
            for q_idx, query in enumerate(queries):
                rdkit_count = len(target.GetSubstructMatches(query, uniquify=False))
                assert results[t_idx, q_idx] == rdkit_count, (
                    f"Mismatch at target {t_idx}, query {q_idx}: GPU={results[t_idx, q_idx]}, RDKit={rdkit_count}"
                )


@pytest.mark.long
class TestIntegrationConfig:
    """Integration tests for different configuration settings."""

    @pytest.fixture
    def test_mols(self) -> tuple[list[Chem.Mol], list[Chem.Mol]]:
        """Load a subset of molecules for config testing."""
        smiles_path = TEST_DATA_DIR / "chembl_1k.smi"
        if not smiles_path.exists():
            pytest.skip(f"Test data not found: {smiles_path}")

        targets = load_smiles_file(smiles_path, max_count=50)
        queries = [
            Chem.MolFromSmarts("C"),
            Chem.MolFromSmarts("[CX3]=[OX1]"),
            Chem.MolFromSmarts("[NX3]"),
        ]
        return targets, queries

    def test_multithreaded_config(self, test_mols):
        """Test with multiple worker threads."""
        targets, queries = test_mols

        config = SubstructSearchConfig()
        config.workerThreads = 2

        results = getSubstructMatches(targets, queries, config)

        validation = validate_against_rdkit(targets, queries, results)
        assert len(validation.count_mismatches) == 0

    def test_preprocessing_threads(self, test_mols):
        """Test with preprocessing threads."""
        targets, queries = test_mols

        config = SubstructSearchConfig()
        config.preprocessingThreads = 4

        results = getSubstructMatches(targets, queries, config)

        validation = validate_against_rdkit(targets, queries, results)
        assert len(validation.count_mismatches) == 0

    def test_small_batch_size(self, test_mols):
        """Test with smaller batch size."""
        targets, queries = test_mols

        config = SubstructSearchConfig()
        config.batchSize = 64

        results = getSubstructMatches(targets, queries, config)

        validation = validate_against_rdkit(targets, queries, results)
        assert len(validation.count_mismatches) == 0

    def test_results_exist(self, test_mols):
        """Test basic match results exist."""
        targets, queries = test_mols

        results = getSubstructMatches(targets, queries)

        validation = validate_against_rdkit(targets, queries, results)
        assert len(validation.count_mismatches) == 0
