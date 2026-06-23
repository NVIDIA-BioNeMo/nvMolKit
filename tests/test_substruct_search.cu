// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <GraphMol/MolOps.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <vector>

#include "src/substruct/substruct_search.h"
#include "src/testutils/substruct_validation.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"

using nvMolKit::algorithmName;
using nvMolKit::checkReturnCode;
using nvMolKit::countSubstructMatches;
using nvMolKit::getRDKitSubstructMatches;
using nvMolKit::getSubstructMatches;
using nvMolKit::hasSubstructMatch;
using nvMolKit::HasSubstructMatchResults;
using nvMolKit::ScopedStream;
using nvMolKit::SubstructAlgorithm;
using nvMolKit::SubstructSearchConfig;
using nvMolKit::SubstructSearchResults;

namespace {

std::unique_ptr<RDKit::ROMol> makeMolFromSmiles(const std::string& smiles) {
  return std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
}

std::unique_ptr<RDKit::ROMol> makeMolFromSmarts(const std::string& smarts) {
  return std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
}

}  // namespace

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class SubstructureSearchTest : public ::testing::TestWithParam<SubstructAlgorithm> {
 protected:
  ScopedStream stream_;

  SubstructAlgorithm algorithm() const { return GetParam(); }

  void SetUp() override {}

  /**
   * @brief Parse target and query molecules from SMILES/SMARTS strings.
   */
  void parseMolecules(const std::vector<std::string>&             targetSmiles,
                      const std::vector<std::string>&             querySmarts,
                      std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                      std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols) {
    targetMols.clear();
    queryMols.clear();

    for (const auto& smiles : targetSmiles) {
      auto mol = makeMolFromSmiles(smiles);
      ASSERT_NE(mol, nullptr) << "Failed to parse target SMILES: " << smiles;
      targetMols.push_back(std::move(mol));
    }

    for (const auto& smarts : querySmarts) {
      auto mol = makeMolFromSmarts(smarts);
      ASSERT_NE(mol, nullptr) << "Failed to parse query SMARTS: " << smarts;
      queryMols.push_back(std::move(mol));
    }
  }

  /**
   * @brief Get raw pointers from unique_ptr vectors for API call.
   */
  static std::vector<const RDKit::ROMol*> getRawPtrs(const std::vector<std::unique_ptr<RDKit::ROMol>>& mols) {
    std::vector<const RDKit::ROMol*> ptrs;
    ptrs.reserve(mols.size());
    for (const auto& mol : mols) {
      ptrs.push_back(mol.get());
    }
    return ptrs;
  }

  /**
   * @brief Extract GPU matches for a (target, query) pair from results.
   */
  static std::vector<std::vector<int>> extractGpuMatches(const SubstructSearchResults& results,
                                                         int                           targetIdx,
                                                         int                           queryIdx,
                                                         int /* numQueryAtoms */) {
    return results.getMatches(targetIdx, queryIdx);
  }

  /**
   * @brief Compare two sets of matches (order-independent).
   */
  static bool matchSetsEqual(const std::vector<std::vector<int>>& gpuMatches,
                             const std::vector<std::vector<int>>& rdkitMatches) {
    if (gpuMatches.size() != rdkitMatches.size()) {
      return false;
    }

    std::set<std::vector<int>> gpuSet(gpuMatches.begin(), gpuMatches.end());
    std::set<std::vector<int>> rdkitSet(rdkitMatches.begin(), rdkitMatches.end());

    return gpuSet == rdkitSet;
  }

  /**
   * @brief Verify GPU matches RDKit for a single (target, query) pair.
   *
   * Checks both match count AND actual atom mappings.
   */
  void expectMatchesRDKit(const SubstructSearchResults& results,
                          const RDKit::ROMol&           target,
                          const RDKit::ROMol&           query,
                          int                           targetIdx,
                          int                           queryIdx,
                          const std::string&            description = "") {
    const auto rdkitMatches    = getRDKitSubstructMatches(target, query, false);
    const int  gpuMatchCount   = results.matchCount(targetIdx, queryIdx);
    const int  rdkitMatchCount = static_cast<int>(rdkitMatches.size());

    std::string context = description.empty() ? "" : " (" + description + ")";

    EXPECT_EQ(gpuMatchCount, rdkitMatchCount)
      << "Match count mismatch" << context << " using " << algorithmName(algorithm()) << ": GPU=" << gpuMatchCount
      << ", RDKit=" << rdkitMatchCount;

    if (gpuMatchCount == rdkitMatchCount && gpuMatchCount > 0) {
      const int  numQueryAtoms = static_cast<int>(query.getNumAtoms());
      const auto gpuMatches    = extractGpuMatches(results, targetIdx, queryIdx, numQueryAtoms);

      EXPECT_TRUE(matchSetsEqual(gpuMatches, rdkitMatches))
        << "Match indices mismatch" << context << " using " << algorithmName(algorithm()) << ": counts match ("
        << gpuMatchCount << ") but atom mappings differ";
    }
  }

  /**
   * @brief Compare GPU results against RDKit ground truth (with uniquify=false).
   *
   * @param results GPU results
   * @param targetMols Target molecules for RDKit comparison
   * @param queryMols Query molecules for RDKit comparison
   * @param expectMatch If true, expect tests to pass; if false, expect current failures
   */
  void compareWithRDKit(const SubstructSearchResults&                     results,
                        const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                        const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                        bool                                              expectMatch = false) {
    for (int t = 0; t < results.numTargets; ++t) {
      for (int q = 0; q < results.numQueries; ++q) {
        // Use uniquify=false to match our non-uniquifying GPU algorithm
        const auto rdkitMatches    = getRDKitSubstructMatches(*targetMols[t], *queryMols[q], false);
        const int  gpuMatchCount   = results.matchCount(t, q);
        const int  rdkitMatchCount = static_cast<int>(rdkitMatches.size());

        if (expectMatch) {
          EXPECT_EQ(gpuMatchCount, rdkitMatchCount)
            << "Match count mismatch for target " << t << ", query " << q << " using algorithm "
            << algorithmName(algorithm()) << ": GPU=" << gpuMatchCount << ", RDKit=" << rdkitMatchCount;

          // Also verify actual match indices if counts match
          if (gpuMatchCount == rdkitMatchCount && gpuMatchCount > 0) {
            const int  numQueryAtoms = static_cast<int>(queryMols[q]->getNumAtoms());
            const auto gpuMatches    = extractGpuMatches(results, t, q, numQueryAtoms);

            EXPECT_TRUE(matchSetsEqual(gpuMatches, rdkitMatches))
              << "Match indices mismatch for target " << t << ", query " << q << " using algorithm "
              << algorithmName(algorithm()) << ": counts match (" << gpuMatchCount << ") but atom mappings differ";
          }
        }
      }
    }
  }
};

// Instantiate parameterized tests for all algorithms
INSTANTIATE_TEST_SUITE_P(AllAlgorithms,
                         SubstructureSearchTest,
                         ::testing::Values(SubstructAlgorithm::GSI, SubstructAlgorithm::VF2),
                         [](const ::testing::TestParamInfo<SubstructAlgorithm>& info) {
                           return algorithmName(info.param);
                         });

// Fixture for recursive SMARTS tests - VF2 doesn't support recursion
class RecursiveSubstructureSearchTest : public SubstructureSearchTest {};

INSTANTIATE_TEST_SUITE_P(GSIOnly,
                         RecursiveSubstructureSearchTest,
                         ::testing::Values(SubstructAlgorithm::GSI),
                         [](const ::testing::TestParamInfo<SubstructAlgorithm>& info) {
                           return algorithmName(info.param);
                         });

// =============================================================================
// Basic Tests - Run with all algorithms
// =============================================================================

TEST_P(SubstructureSearchTest, SingleTargetSingleQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({"CCO"}, {"C"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  EXPECT_EQ(results.numTargets, 1);
  EXPECT_EQ(results.numQueries, 1);

  // Compare with RDKit - expect match once algorithms are working
  compareWithRDKit(results, targetMols, queryMols, true);
}

TEST_P(SubstructureSearchTest, MultipleTargetsSingleQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({"CCO", "CCCC", "c1ccccc1"}, {"C"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  EXPECT_EQ(results.numTargets, 3);
  EXPECT_EQ(results.numQueries, 1);

  compareWithRDKit(results, targetMols, queryMols, true);
}

TEST_P(SubstructureSearchTest, SingleTargetMultipleQueries) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({"CCO"}, {"C", "O", "CC"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  EXPECT_EQ(results.numTargets, 1);
  EXPECT_EQ(results.numQueries, 3);

  compareWithRDKit(results, targetMols, queryMols, true);
}

TEST_P(SubstructureSearchTest, BatchAllToAll) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Use molecules and queries that produce reasonable match counts.
  // Single-atom queries and aromatic targets are used for simpler test cases.
  parseMolecules({"CCO", "c1ccccc1", "c1ccc(O)cc1", "CCN"},  // 4 targets: 3, 6, 7, 3 atoms
                 {"C", "O", "c", "N"},                       // 4 single-atom queries
                 targetMols,
                 queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  EXPECT_EQ(results.numTargets, 4);
  EXPECT_EQ(results.numQueries, 4);

  compareWithRDKit(results, targetMols, queryMols, true);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_P(SubstructureSearchTest, NoMatchPossible) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Carbon chain vs nitrogen query - no match possible
  parseMolecules({"CCCC"}, {"N"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "CCCC with N query");
}

TEST_P(SubstructureSearchTest, AromaticVsAliphatic) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Benzene (aromatic) vs aliphatic carbon query
  parseMolecules({"c1ccccc1"}, {"C"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "benzene with aliphatic C query");
}

TEST_P(SubstructureSearchTest, LargerMolecule) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Caffeine as a larger test case
  parseMolecules({"Cn1cnc2c1c(=O)n(c(=O)n2C)C"}, {"c", "N", "C"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  EXPECT_EQ(results.numTargets, 1);
  EXPECT_EQ(results.numQueries, 3);

  compareWithRDKit(results, targetMols, queryMols, true);
}

TEST_P(SubstructureSearchTest, DifferentMoleculeSizes) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Different sized molecules to test buffer allocation
  parseMolecules({"C", "CCC", "CCCCC"},  // 1, 3, 5 atoms
                 {"C", "N"},             // 1, 1 query atoms
                 targetMols,
                 queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  // Verify match counts are correct
  // C (query) matches: 1 in C, 3 in CCC, 5 in CCCCC
  EXPECT_EQ(results.matchCount(0, 0), 1);
  EXPECT_EQ(results.matchCount(1, 0), 3);
  EXPECT_EQ(results.matchCount(2, 0), 5);

  // N (query) matches: 0 in all targets
  EXPECT_EQ(results.matchCount(0, 1), 0);
  EXPECT_EQ(results.matchCount(1, 1), 0);
  EXPECT_EQ(results.matchCount(2, 1), 0);

  compareWithRDKit(results, targetMols, queryMols, true);
}

TEST_P(SubstructureSearchTest, MultiAtomQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test multi-atom queries
  // CCO with CC: 2 non-unique matches (0,1) and (1,0)
  parseMolecules({"CCO"},  // 3 atoms
                 {"CC"},   // 2 atom query - should get 2 matches with uniquify=false
                 targetMols,
                 queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "CCO with CC query");
}

TEST_P(SubstructureSearchTest, ThreeAtomQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test 3-atom query: CCOCC with COC
  // Should get 2 matches: (1,2,3) and (3,2,1) - both directions through the ether
  parseMolecules({"CCOCC"},  // 5 atoms - diethyl ether
                 {"COC"},    // 3 atom query
                 targetMols,
                 queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "CCOCC with COC query");
}

TEST_P(SubstructureSearchTest, OverflowHandledByRDKitFallback) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // CCCCCC (6 atoms) with CC query has 10 non-unique matches
  // Buffer sized to target atoms (6), but RDKit fallback should provide all matches
  parseMolecules({"CCCCCC"}, {"CC"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  // RDKit returns 10 non-unique matches
  auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(rdkitMatches.size(), 10u);

  // With RDKit fallback, we should get all 10 matches
  EXPECT_EQ(results.matchCount(0, 0), 10) << "Should have all 10 matches via fallback";
}

// =============================================================================
// Non-Parameterized RDKit Reference Tests
// =============================================================================

class RDKitReferenceTest : public ::testing::Test {};

TEST_F(RDKitReferenceTest, HexaneCCMatches) {
  // Example from user: CCCCCC target, CC query
  // With uniquify=true: 5 matches
  // With uniquify=false: 10 matches
  auto target = makeMolFromSmiles("CCCCCC");
  auto query  = makeMolFromSmarts("CC");

  auto matchesUnique    = getRDKitSubstructMatches(*target, *query, true);
  auto matchesNonUnique = getRDKitSubstructMatches(*target, *query, false);

  EXPECT_EQ(matchesUnique.size(), 5u);
  EXPECT_EQ(matchesNonUnique.size(), 10u);
}

// =============================================================================
// Compound Query Tests (OR/NOT support)
// =============================================================================

TEST_P(SubstructureSearchTest, OrQueryMatchesBothTypes) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test OR query: [C,N] should match both carbons and nitrogens
  // CCN has 2 carbons and 1 nitrogen, so [C,N] should match all 3 atoms
  parseMolecules({"CCN"}, {"[C,N]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[C,N] in CCN");
}

TEST_P(SubstructureSearchTest, OrQuerySelectiveMatch) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test OR query: [N,O] should match nitrogens and oxygens but not carbons
  // CCO has 2 carbons and 1 oxygen, so [N,O] should match only the oxygen (1 atom)
  parseMolecules({"CCO"}, {"[N,O]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[N,O] in CCO");
}

TEST_P(SubstructureSearchTest, NotQueryExcludesAtom) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test NOT query: [!C] should match everything except carbon
  // CCO has 2 carbons and 1 oxygen, so [!C] should match only the oxygen
  parseMolecules({"CCO"}, {"[!C]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[!C] in CCO");
}

TEST_P(SubstructureSearchTest, NotQueryMatchesMultiple) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test NOT query: [!C] in molecule with multiple non-carbons
  // CCNO has 2 carbons, 1 nitrogen, and 1 oxygen, so [!C] should match 2 atoms
  parseMolecules({"CCNO"}, {"[!C]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[!C] in CCNO");
}

TEST_P(SubstructureSearchTest, MultiAtomOrQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test OR query with multi-atom pattern: [C,N][C,N]
  // CCN should match CC, CN, NC, and would match NN if present
  parseMolecules({"CCN"}, {"[C,N][C,N]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[C,N][C,N] in CCN");
}

TEST_P(SubstructureSearchTest, ThreeWayOrQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test 3-way OR: [C,N,O] should match all of C, N, and O
  // CCNO has all three types, should match 4 atoms
  parseMolecules({"CCNO"}, {"[C,N,O]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[C,N,O] in CCNO");
}

TEST_P(SubstructureSearchTest, NestedAndOrQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test nested AND/OR: [C,N;!R1] = (C OR N) AND NOT(in 1 ring)
  // In "CCN" (no rings), all 3 atoms should match
  // In "C1CC1N" (cyclopropane + N), only N should match (ring carbons fail !R1)
  // In "C1CCC1" (cyclobutane), 0 atoms match (all ring carbons fail !R1)
  parseMolecules({"CCN", "C1CC1N", "C1CCC1"}, {"[C,N;!R1]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[C,N;!R1] in CCN");
  expectMatchesRDKit(results, *targetMols[1], *queryMols[0], 1, 0, "[C,N;!R1] in C1CC1N");
  expectMatchesRDKit(results, *targetMols[2], *queryMols[0], 2, 0, "[C,N;!R1] in C1CCC1");
}

TEST_P(SubstructureSearchTest, DeepNestedOrAndOrQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test deep nesting: [C,N;R1,O] (complex SMARTS with multiple operators)
  // In cyclopentane C1CCCC1: ring carbons match
  // In "CCCCO": behavior depends on SMARTS precedence rules
  parseMolecules({"C1CCCC1", "CCCCO"}, {"[C,N;R1,O]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[C,N;R1,O] in C1CCCC1");
  expectMatchesRDKit(results, *targetMols[1], *queryMols[0], 1, 0, "[C,N;R1,O] in CCCCO");
}

TEST_P(SubstructureSearchTest, MultipleNotWithAndQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [!C;!N] = NOT(C) AND NOT(N) - matches anything except C or N
  // In "CCNO": only O matches
  parseMolecules({"CCNO"}, {"[!C;!N]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[!C;!N] in CCNO");
}

TEST_P(SubstructureSearchTest, NotWithOrQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [!C,!N] = NOT(C) OR NOT(N) - matches anything except C AND N
  // C atoms: NOT(C)=false, NOT(N)=true => true
  // N atom: NOT(C)=true, NOT(N)=false => true
  // O atom: NOT(C)=true, NOT(N)=true => true
  // All 4 atoms match
  parseMolecules({"CCNO"}, {"[!C,!N]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[!C,!N] in CCNO");
}

TEST_P(SubstructureSearchTest, SimpleAndNotQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [C;!R1] = C AND NOT(in 1 ring)
  // In C1CC1CCN (cyclopropane with chain): ring carbons (0,1,2) excluded, chain carbons (3,4) match
  parseMolecules({"C1CC1CCN"}, {"[C;!R1]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[C;!R1] in C1CC1CCN");
}

TEST_P(SubstructureSearchTest, BondedOrAtomQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test bonded pattern with OR atoms: [C,N]-[O,S]
  // In "CCNO": C-N-O, so N-O bond matches (N matches [C,N], O matches [O,S])
  // In "CCSO": C-S-O, so S-O bond matches if S in query... wait, S doesn't match [C,N]
  // Actually "CCS": C-C-S, no match for [C,N]-[O,S] since S doesn't connect to O
  // Use "CCO" (ethanol): C-C-O, C matches [C,N], O matches [O,S], so C-O matches
  parseMolecules({"CCO", "CCS"}, {"[C,N]-[O,S]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[C,N]-[O,S] in CCO");
  expectMatchesRDKit(results, *targetMols[1], *queryMols[0], 1, 0, "[C,N]-[O,S] in CCS");
}

TEST_P(SubstructureSearchTest, MultiAtomMixedBooleanQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test multi-atom query with different boolean logic per atom: [C,N]-[!O]
  // First atom is OR, second is NOT
  // In "CCO": C-C bond matches (C for [C,N], C for [!O]), C-O doesn't match (!O fails)
  // In "CCN": C-C and C-N bonds all match
  parseMolecules({"CCO", "CCN"}, {"[C,N]-[!O]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[C,N]-[!O] in CCO");
  expectMatchesRDKit(results, *targetMols[1], *queryMols[0], 1, 0, "[C,N]-[!O] in CCN");
}

TEST_P(SubstructureSearchTest, ThreeAtomNestedBooleanQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test 3-atom pattern with nested boolean: [C,N]-[!O]-[C,O]
  // In "CCCCO": C-C-C-C-O, should find C-C-C and C-C-O patterns
  parseMolecules({"CCCCO"}, {"[C,N]-[!O]-[C,O]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[C,N]-[!O]-[C,O] in CCCCO");
}

TEST_P(SubstructureSearchTest, AromaticOrQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test aromatic OR: [c,n] should match aromatic carbons and nitrogens
  // In pyridine "c1ccncc1": 5 aromatic carbons + 1 aromatic nitrogen = 6 matches
  parseMolecules({"c1ccncc1"}, {"[c,n]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[c,n] in pyridine");
}

TEST_P(SubstructureSearchTest, AromaticNotQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test aromatic NOT: [!n] should match anything except aromatic nitrogen
  // In pyridine "c1ccncc1": 5 aromatic carbons match, nitrogen doesn't
  parseMolecules({"c1ccncc1"}, {"[!n]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[!n] in pyridine");
}

TEST_P(SubstructureSearchTest, AromaticRingPatternWithOr) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test aromatic ring pattern with OR: c1[c,n]cccc1 (benzene or pyridine-like ring)
  // In benzene "c1ccccc1": all carbons form 6-ring, position 1 matches [c,n]
  // In pyridine "c1ccncc1": position 1 is nitrogen which matches [c,n]
  parseMolecules({"c1ccccc1", "c1ccncc1"}, {"c1[c,n]cccc1"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  // Benzene should match (symmetric, many automorphisms)
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for c1[c,n]cccc1 in benzene using " << algorithmName(algorithm());

  // Pyridine should match
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for c1[c,n]cccc1 in pyridine using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, AnyRingMembershipQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [R] any ring membership query
  // C1CCC1C: 4 ring atoms, 1 non-ring atom
  // CCCCC: no ring atoms
  parseMolecules({"C1CCC1C", "CCCCC"}, {"[R]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  // Cyclobutane with methyl: 4 ring atoms match [R]
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [R] in C1CCC1C using " << algorithmName(algorithm());

  // Pentane: no ring atoms, should get 0 matches
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [R] in CCCCC using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, AnyRingSizeQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [r] any ring size query (same semantics as [R])
  // c1ccccc1: 6 ring atoms
  // CCCCC: no ring atoms
  parseMolecules({"c1ccccc1", "CCCCC"}, {"[r]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  // Benzene: 6 ring atoms match [r]
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [r] in benzene using " << algorithmName(algorithm());

  // Pentane: no ring atoms, should get 0 matches
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [r] in CCCCC using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, AnyRingCombinedWithAtomType) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [C;R] carbon in any ring
  // C1CCC1C: 4 ring carbons, 1 non-ring carbon
  // c1ccccc1: aromatic carbons (not aliphatic C)
  parseMolecules({"C1CCC1C", "c1ccccc1"}, {"[C;R]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  // Cyclobutane with methyl: 4 aliphatic ring carbons match [C;R]
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [C;R] in C1CCC1C using " << algorithmName(algorithm());

  // Benzene: aromatic carbons don't match aliphatic C
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [C;R] in benzene using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, IsotopeCarbon13Query) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [13C] isotope query
  // [13C]CC: one carbon-13 atom
  // CC: natural abundance carbons (isotope = 0)
  parseMolecules({"[13C]CC", "CC"}, {"[13C]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  // First target has one 13C
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [13C] in [13C]CC using " << algorithmName(algorithm());

  // Second target has no isotope labels
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [13C] in CC using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, IsotopeDeuteriumQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [2H] deuterium query
  // [2H]C([2H])([2H])[2H]: deuterated methane with 4 deuterium atoms
  // C: regular methane (no explicit H with isotope)
  parseMolecules({"[2H]C([2H])([2H])[2H]", "C"}, {"[2H]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  // First target has 4 deuterium atoms
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [2H] in CD4 using " << algorithmName(algorithm());

  // Second target has no deuterium
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [2H] in CH4 using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, IsotopeNitrogen15Query) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [15N] nitrogen-15 query
  // [15N]CC: nitrogen-15 labeled
  // NCC: natural abundance nitrogen
  parseMolecules({"[15N]CC", "NCC"}, {"[15N]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  // First target has one 15N
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [15N] in [15N]CC using " << algorithmName(algorithm());

  // Second target has no isotope labels
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [15N] in NCC using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, DegreeQueryD0) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [D0] degree query - atom with no explicit bonds
  // C: methane - single atom (degree 0)
  // CC: ethane - both atoms have degree 1
  parseMolecules({"C", "CC"}, {"[D0]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [D0] in methane using " << algorithmName(algorithm());

  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [D0] in ethane using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, DegreeQueryD1) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [D1] degree query - terminal atoms
  // CC: ethane - both atoms have degree 1
  // CCC: propane - 2 terminal atoms with degree 1
  parseMolecules({"CC", "CCC"}, {"[D1]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [D1] in ethane using " << algorithmName(algorithm());

  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [D1] in propane using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, DegreeQueryD3) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [D3] degree query
  // CC(C)C: isobutane - central carbon has degree 3
  // CCC: propane - no degree 3 atoms
  parseMolecules({"CC(C)C", "CCC"}, {"[D3]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  // Isobutane: one atom with degree 3
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [D3] in isobutane using " << algorithmName(algorithm());

  // Propane: no degree 3 atoms
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [D3] in propane using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, TotalConnectivityQueryX1) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [X1] total connectivity query
  // [H][H]: H2 molecule - each H has X1 (1 bond + 0 H = 1)
  // CC: ethane - carbons have X4
  parseMolecules({"[H][H]", "CC"}, {"[X1]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [X1] in H2 using " << algorithmName(algorithm());

  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [X1] in ethane using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, TotalConnectivityQueryX2) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [X2] total connectivity query
  // C#C: acetylene - carbons have X2 (1 bond + 1 H = 2)
  // CC: ethane - carbons have X4
  parseMolecules({"C#C", "CC"}, {"[X2]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [X2] in acetylene using " << algorithmName(algorithm());

  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [X2] in ethane using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, TotalConnectivityQueryX3) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [X3] total connectivity query
  // C=C: ethene - carbons have X3 (1 bond + 2 H = 3)
  // CC: ethane - carbons have X4
  parseMolecules({"C=C", "CC"}, {"[X3]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [X3] in ethene using " << algorithmName(algorithm());

  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [X3] in ethane using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, TotalConnectivityQueryX4) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [X4] total connectivity query
  // CC: ethane - all carbons have X4 (1 bond + 3 H = 4)
  // C=C: ethene - carbons have X3 (1 bond + 2 H = 3)
  parseMolecules({"CC", "C=C"}, {"[X4]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  // Ethane: 2 atoms with X4
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [X4] in ethane using " << algorithmName(algorithm());

  // Ethene: no X4 atoms
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [X4] in ethene using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, DegreeWithAtomTypeQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Test [CD3] carbon with degree 3
  // CC(C)C: isobutane
  // CN(C)C: trimethylamine - nitrogen has degree 3, not carbon
  parseMolecules({"CC(C)C", "CN(C)C"}, {"[CD3]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  // Isobutane: one carbon with degree 3
  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for [CD3] in isobutane using " << algorithmName(algorithm());

  // Trimethylamine: no carbon with degree 3 (N has degree 3)
  auto rdkitMatches1 = getRDKitSubstructMatches(*targetMols[1], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(1, 0), static_cast<int>(rdkitMatches1.size()))
    << "GPU should match RDKit for [CD3] in trimethylamine using " << algorithmName(algorithm());
}

TEST_P(SubstructureSearchTest, ImplicitHCountMatch) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({"C=N"}, {"[NH]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[NH] in C=N");
}

TEST_P(SubstructureSearchTest, ImplicitHCountNoMatch) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({"CC"}, {"[CH2]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[CH2] in CC");
}

TEST_P(SubstructureSearchTest, DoubleOrAromaticBond) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Quinone pattern uses =,: which is "double or aromatic" bond
  // Target: theophylline derivative with quinone moiety
  const std::string target = "Cn1c(=O)c2c3c(cnc2n(C)c1=O)C(=O)C=CC3=O";
  const std::string query  = "[!#6&!#1]=[#6]-1-[#6]=,:[#6]-[#6](=[!#6&!#1])-[#6]=,:[#6]-1";
  parseMolecules({target}, {query}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "quinone_A pattern");
}

TEST_P(SubstructureSearchTest, NotRingBondSimple) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Simple pattern with single AND not-ring bond: 2 atoms connected by non-ring single bond
  // Target: propylamine CCCN - all bonds are single and non-ring
  const std::string target = "CCCN";
  const std::string query  = "[C,N]-&!@[C,N]";
  parseMolecules({target}, {query}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "simple non-ring bond");
}

TEST_P(SubstructureSearchTest, NotRingBondChain) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Pattern with single AND not-ring bonds: chain of 7 atoms connected by non-ring bonds
  // Target: peptide-like chain
  const std::string target =
    "C[C@@H](O)[C@H](N)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(N)=O";
  const std::string query = "[N,C,S,O]-&!@[N,C,S,O]-&!@[N,C,S,O]-&!@[N,C,S,O]-&!@[N,C,S,O]-&!@[N,C,S,O]-&!@[N,C,S,O]";
  parseMolecules({target}, {query}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "non-ring bond chain pattern");
}

TEST_P(SubstructureSearchTest, ImpossibleBondConstraint) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Pattern with -: (single AND aromatic) which is impossible
  // This should never match anything
  const std::string target = "c1ccccc1";  // Benzene - has aromatic bonds
  const std::string query  = "[!#1]-:a";  // Not-hydrogen with single-AND-aromatic bond to aromatic atom
  parseMolecules({target}, {query}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches.size()))
    << "Impossible bond constraint should match 0 (RDKit says " << rdkitMatches.size() << ")";
  EXPECT_EQ(results.matchCount(0, 0), 0) << "Impossible bond constraint (single AND aromatic) should never match";
}

TEST_P(SubstructureSearchTest, ImpossibleAtomConstraint) {
  // [C;a] combines uppercase C (aliphatic carbon) with ;a (aromatic requirement)
  // This is contradictory and should never match anything
  const std::string query = "[C;a]";

  for (const auto& target : {"CCCCC", "c1ccccc1"}) {
    std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
    std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

    parseMolecules({target}, {query}, targetMols, queryMols);

    SubstructSearchResults results;
    getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

    auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
    EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches.size()))
      << "Impossible atom constraint [C;a] on " << target << " should match 0 (RDKit says " << rdkitMatches.size()
      << ")";
    EXPECT_EQ(results.matchCount(0, 0), 0)
      << "Impossible atom constraint [C;a] (aliphatic AND aromatic) should never match " << target;
  }
}

TEST_P(SubstructureSearchTest, ImpossibleChargeConstraint) {
  // [OX1;+0;-1] has charge +0 AND charge -1 in an AND, which is contradictory
  const std::string query = "[OX1;+0;-1]";

  for (const auto& target : {"O=S(=O)(CCO)c1ccccc1", "[O-]C"}) {
    std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
    std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

    parseMolecules({target}, {query}, targetMols, queryMols);

    SubstructSearchResults results;
    getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

    auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
    EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches.size()))
      << "Impossible charge constraint [OX1;+0;-1] on " << target << " should match 0 (RDKit says "
      << rdkitMatches.size() << ")";
    EXPECT_EQ(results.matchCount(0, 0), 0)
      << "Impossible charge constraint [OX1;+0;-1] (charge 0 AND -1) should never match " << target;
  }
}

TEST_P(SubstructureSearchTest, WildcardAtoms) {
  // Tests that wildcard atoms (*) with empty boolean trees correctly match any atom
  const std::string target = "CCCCCC";   // hexane
  const std::string query  = "C~*~*~C";  // C-any-any-C

  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({target}, {query}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches.size()))
    << "Wildcard pattern C~*~*~C should match " << rdkitMatches.size() << " times (RDKit), got "
    << results.matchCount(0, 0);
  EXPECT_GT(results.matchCount(0, 0), 0) << "Wildcard pattern should find matches in hexane";
}

TEST_P(SubstructureSearchTest, WildcardAtomsInRing) {
  // Ring pattern with wildcard atoms: C1~*~*~C~*~*~1
  const std::string target = "C1CCCCC1";        // cyclohexane
  const std::string query  = "C1~*~*~C~*~*~1";  // 6-membered ring with wildcards

  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({target}, {query}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches.size()))
    << "Wildcard ring pattern should match " << rdkitMatches.size() << " times (RDKit), got "
    << results.matchCount(0, 0);
  EXPECT_GT(results.matchCount(0, 0), 0) << "Wildcard ring pattern should find matches in cyclohexane";
}

TEST_P(SubstructureSearchTest, WildcardAtomsFusedRings) {
  // Two fused 6-membered rings with wildcards (decalin pattern)
  const std::string target = "C1CCC2CCCCC2C1";             // decalin
  const std::string query  = "C12~*~*~*~*~C~1~*~*~*~*~2";  // two 6-rings sharing edge

  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({target}, {query}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches.size()))
    << "Fused ring wildcard pattern should match " << rdkitMatches.size() << " times (RDKit), got "
    << results.matchCount(0, 0);
  EXPECT_GT(results.matchCount(0, 0), 0) << "Fused ring wildcard pattern should find matches in decalin";
}

TEST_P(SubstructureSearchTest, NegatedBondType) {
  // !- means NOT single bond (should only match double, triple, or aromatic)
  // This ring pattern requires non-single bonds between atoms
  const std::string target = "C=CCn1cc(C[C@@H]2NC(=O)[C@@H]3CCCN3C2=O)c2ccc(OC)cc21";
  const std::string query  = "[c,C]1(~[O;D1])~*!-*~[c,C](~[O;D1])~*!-*~1";

  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({target}, {query}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches.size()))
    << "Negated bond type !- query should match " << rdkitMatches.size() << " times (RDKit), got "
    << results.matchCount(0, 0);
}

// =============================================================================
// New Query Type Tests - Ring Bond Count, Implicit H, Heteroatom Neighbors, Ranges
// =============================================================================

TEST_P(SubstructureSearchTest, RingBondCountQuery) {
  // [x2] matches atoms with exactly 2 ring bonds (e.g., atoms in a single ring)
  // [x4] matches atoms with 4 ring bonds (e.g., bridgehead atoms in fused rings)
  struct TestCase {
    std::string target;
    std::string query;
    std::string description;
  };

  const std::vector<TestCase> cases = {
    {      "C1CCCCC1",  "[x2]",        "Cyclohexane atoms have 2 ring bonds"},
    {"C1CCC2CCCCC2C1",  "[x4]", "Decalin bridgehead atoms have 4 ring bonds"},
    {      "c1ccccc1", "[cx2]", "Benzene aromatic carbons have 2 ring bonds"},
    {"c1ccc2ccccc2c1", "[cx3]", "Naphthalene fusion atoms have 3 ring bonds"},
  };

  for (const auto& tc : cases) {
    std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
    std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

    parseMolecules({tc.target}, {tc.query}, targetMols, queryMols);

    SubstructSearchResults results;
    getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

    auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
    EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches.size()))
      << tc.description << " - " << tc.query << " on " << tc.target;
  }
}

TEST_P(SubstructureSearchTest, ImplicitHCountQuery) {
  // [h1] matches atoms with exactly 1 implicit hydrogen
  // [h] matches atoms with any implicit hydrogens
  struct TestCase {
    std::string target;
    std::string query;
    std::string description;
  };

  const std::vector<TestCase> cases = {
    {         "CC",  "[h3]",                          "Methyl carbons have 3 implicit H"},
    {         "CC",   "[h]",                              "Both carbons have implicit H"},
    {     "CC(C)C",  "[h1]",              "Central carbon in isobutane has 1 implicit H"},
    {"C(C)(C)(C)C",  "[h0]",                        "Quaternary carbon has 0 implicit H"},
    {          "N", "[Nh2]", "NH3 nitrogen has 2 implicit H (one is explicit in SMILES)"},
  };

  for (const auto& tc : cases) {
    std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
    std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

    parseMolecules({tc.target}, {tc.query}, targetMols, queryMols);

    SubstructSearchResults results;
    getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

    auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
    EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches.size()))
      << tc.description << " - " << tc.query << " on " << tc.target;
  }
}

TEST_P(SubstructureSearchTest, HeteroatomNeighborsQuery) {
  // [z1] matches atoms with exactly 1 heteroatom neighbor
  // [z2] matches atoms with 2 heteroatom neighbors
  struct TestCase {
    std::string target;
    std::string query;
    std::string description;
  };

  const std::vector<TestCase> cases = {
    { "CCO", "[Cz1]",                "Carbon next to oxygen has 1 heteroatom neighbor"},
    {"OCCO", "[Cz2]", "Central carbons in ethylene glycol have 2 heteroatom neighbors"},
    { "CCN", "[Cz1]",              "Carbon next to nitrogen has 1 heteroatom neighbor"},
    {"NCCN", "[Cz1]",         "Central carbons in EDA each have 1 heteroatom neighbor"},
    {"CCCC", "[Cz0]",                     "Alkane carbons have 0 heteroatom neighbors"},
  };

  for (const auto& tc : cases) {
    std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
    std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

    parseMolecules({tc.target}, {tc.query}, targetMols, queryMols);

    SubstructSearchResults results;
    getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

    auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
    EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches.size()))
      << tc.description << " - " << tc.query << " on " << tc.target;
  }
}

TEST_P(SubstructureSearchTest, RangeRingSizeQuery) {
  // [r{5-7}] matches atoms in rings of size 5, 6, or 7
  // [r{-6}] matches atoms in rings of size <= 6
  // [r{5-}] matches atoms in rings of size >= 5
  struct TestCase {
    std::string target;
    std::string query;
    std::string description;
  };

  const std::vector<TestCase> cases = {
    {   "C1CCCC1", "[r{5-6}]",                 "Cyclopentane atoms match [r{5-6}]"},
    {  "C1CCCCC1", "[r{5-6}]",                  "Cyclohexane atoms match [r{5-6}]"},
    {  "C1CCCCC1",  "[r{-6}]",  "Cyclohexane atoms match [r{-6}] (ring size <= 6)"},
    {"C1CCCCCCC1",  "[r{5-}]",  "Cyclooctane atoms match [r{5-}] (ring size >= 5)"},
    {     "C1CC1",  "[r{-4}]", "Cyclopropane atoms match [r{-4}] (ring size <= 4)"},
  };

  for (const auto& tc : cases) {
    std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
    std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

    parseMolecules({tc.target}, {tc.query}, targetMols, queryMols);

    SubstructSearchResults results;
    getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

    auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
    EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches.size()))
      << tc.description << " - " << tc.query << " on " << tc.target;
  }
}

TEST_P(SubstructureSearchTest, RangeNumRingsQuery) {
  // [R{1-2}] matches atoms in 1 or 2 rings
  struct TestCase {
    std::string target;
    std::string query;
    std::string description;
  };

  const std::vector<TestCase> cases = {
    {      "C1CCCCC1", "[R{1-2}]", "Cyclohexane atoms are in exactly 1 ring"},
    {"C1CCC2CCCCC2C1",  "[R{2-}]", "Decalin bridgehead atoms are in 2 rings"},
    {"c1ccc2ccccc2c1", "[R{1-2}]",      "Naphthalene atoms are in 1-2 rings"},
  };

  for (const auto& tc : cases) {
    std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
    std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

    parseMolecules({tc.target}, {tc.query}, targetMols, queryMols);

    SubstructSearchResults results;
    getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

    auto rdkitMatches = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
    EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches.size()))
      << tc.description << " - " << tc.query << " on " << tc.target;
  }
}

// KEEP this as the last test
TEST_P(SubstructureSearchTest, SingleMolSingleQueryForDebugging) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  const std::string target = "CN1CCc2cccc3c2[C@H]1Cc1ccc(CO)c(O)c1-3";
  const std::string query  = "[$(c1(-[OX2H])ccccc1);!$(cc-!:[CH2]-[OX2H]);!$(cc-!:C(=O)[O;H1,-]);!$(cc-!:C(=O)-[NH2])]";
  parseMolecules({target}, {query}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  auto rdkitMatches0 = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches0.size()))
    << "GPU should match RDKit for query " << algorithmName(algorithm());
}

// =============================================================================
// Nested Recursive SMARTS Integration Tests
// =============================================================================

TEST_P(RecursiveSubstructureSearchTest, NestedRecursiveSimple) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({"CN", "CCN", "CCC"}, {"[$([C;$(*-N)])]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0, "[$([C;$(*-N)])] in CN");
  expectMatchesRDKit(results, *targetMols[1], *queryMols[0], 1, 0, "[$([C;$(*-N)])] in CCN");
  expectMatchesRDKit(results, *targetMols[2], *queryMols[0], 2, 0, "[$([C;$(*-N)])] in CCC");
}

TEST_P(RecursiveSubstructureSearchTest, NestedRecursiveWithNegation) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({"CCN", "CCC"}, {"[C;!$([C;$(*-N)])]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  expectMatchesRDKit(results, *targetMols[0], *queryMols[0], 0, 0);
  expectMatchesRDKit(results, *targetMols[1], *queryMols[0], 1, 0);
}

TEST_P(RecursiveSubstructureSearchTest, NestedRecursiveBatchProcessing) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  std::vector<std::string> targets = {"CN", "CC", "CCN", "CCCN", "CNO", "CNOF", "c1ccccc1N", "c1ccccc1", "NC(=O)C"};

  parseMolecules(targets, {"[$([C;$(*-N)])]"}, targetMols, queryMols);

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  for (size_t t = 0; t < targets.size(); ++t) {
    expectMatchesRDKit(results, *targetMols[t], *queryMols[0], static_cast<int>(t), 0, targets[t]);
  }
}

TEST_P(RecursiveSubstructureSearchTest, AmideRecursiveQueryMatchesSmallTargets) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({"CC(=O)N", "NC=O", "CC(=O)O"}, {"[$(NC=O)]"}, targetMols, queryMols);

  HasSubstructMatchResults boolResults;
  hasSubstructMatch(getRawPtrs(targetMols), getRawPtrs(queryMols), boolResults, algorithm(), stream_.stream());

  EXPECT_TRUE(boolResults.matches(0, 0));
  EXPECT_TRUE(boolResults.matches(1, 0));
  EXPECT_FALSE(boolResults.matches(2, 0));

  std::vector<int> counts;
  countSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), counts, algorithm(), stream_.stream());

  ASSERT_EQ(counts.size(), targetMols.size() * queryMols.size());
  for (size_t t = 0; t < targetMols.size(); ++t) {
    const auto rdkitMatches = getRDKitSubstructMatches(*targetMols[t], *queryMols[0], false);
    EXPECT_EQ(counts[t], static_cast<int>(rdkitMatches.size())) << "target " << t;
  }
}

TEST_P(SubstructureSearchTest, InvalidSlotsPerRunnerThrows) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({"CCO"}, {"C"}, targetMols, queryMols);

  SubstructSearchResults results;
  SubstructSearchConfig  config;

  config.executorsPerRunner = 0;
  EXPECT_THROW(
    getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream(), config),
    std::invalid_argument);

  config.executorsPerRunner = 9;
  EXPECT_THROW(
    getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream(), config),
    std::invalid_argument);
}

TEST_P(SubstructureSearchTest, ValidSlotsPerRunnerWorks) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({"CCO", "CCCO"}, {"C", "CC"}, targetMols, queryMols);

  SubstructSearchConfig config;
  config.batchSize = 256;

  for (int executors = 1; executors <= 8; ++executors) {
    config.executorsPerRunner = executors;
    SubstructSearchResults results;
    EXPECT_NO_THROW(getSubstructMatches(getRawPtrs(targetMols),
                                        getRawPtrs(queryMols),
                                        results,
                                        algorithm(),
                                        stream_.stream(),
                                        config))
      << "executorsPerRunner=" << executors << " should be valid";
    EXPECT_GT(results.matchCount(0, 0), 0) << "Should find matches with executors=" << executors;
  }
}

// =============================================================================
// maxMatches Parameter Tests
// =============================================================================

TEST_P(SubstructureSearchTest, MaxMatchesZeroUnlimited) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // CCO with C query has 2 carbon matches
  parseMolecules({"CCO"}, {"C"}, targetMols, queryMols);

  SubstructSearchConfig config;
  config.maxMatches = 0;  // Unlimited (like RDKit)

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream(), config);

  // Should store all matches
  EXPECT_EQ(results.matchCount(0, 0), 2) << "maxMatches=0 should store all 2 carbon atoms";
}

TEST_P(SubstructureSearchTest, MaxMatchesLimitedToN) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // CCCC with C query has 4 carbon matches
  parseMolecules({"CCCC"}, {"C"}, targetMols, queryMols);

  SubstructSearchConfig config;
  config.maxMatches = 2;  // Limit to 2 matches

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream(), config);

  // Should only store up to maxMatches
  EXPECT_EQ(results.matchCount(0, 0), 2) << "Should store only 2 matches when maxMatches=2";
}

TEST_P(SubstructureSearchTest, MaxMatchesGreaterThanActual) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // CC with C query has 2 carbon matches
  parseMolecules({"CC"}, {"C"}, targetMols, queryMols);

  SubstructSearchConfig config;
  config.maxMatches = 10;  // More than actual matches

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream(), config);

  // Should store all matches when maxMatches > actual
  EXPECT_EQ(results.matchCount(0, 0), 2) << "Should store all 2 matches when maxMatches > actual";
}

TEST_P(SubstructureSearchTest, MaxMatchesOneEarlyExit) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Large molecule with many matches
  parseMolecules({"CCCCCCCCCC"}, {"C"}, targetMols, queryMols);  // 10 carbons

  SubstructSearchConfig config;
  config.maxMatches = 1;  // Stop after first match

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream(), config);

  // Should only store 1 match
  EXPECT_EQ(results.matchCount(0, 0), 1) << "Should store only 1 match when maxMatches=1";
}

TEST_P(SubstructureSearchTest, MaxMatchesWithMultiAtomQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // CCCCCC with CC query has 10 non-unique matches
  parseMolecules({"CCCCCC"}, {"CC"}, targetMols, queryMols);

  SubstructSearchConfig config;
  config.maxMatches = 3;  // Limit to 3 matches

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream(), config);

  // Should store exactly 3 matches
  EXPECT_EQ(results.matchCount(0, 0), 3) << "Should store exactly 3 matches when maxMatches=3";

  // Each stored match should have 2 atoms (CC query)
  const auto& matches = results.getMatches(0, 0);
  EXPECT_EQ(matches.size(), 3u);
  for (const auto& match : matches) {
    EXPECT_EQ(match.size(), 2u) << "Each match should have 2 atoms for CC query";
  }
}

// =============================================================================
// hasSubstructMatch Tests
// =============================================================================

TEST_P(SubstructureSearchTest, HasSubstructMatchBasic) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({"CCO", "CCCC", "c1ccccc1"}, {"C", "O", "N"}, targetMols, queryMols);

  HasSubstructMatchResults results;
  hasSubstructMatch(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  EXPECT_EQ(results.numTargets, 3);
  EXPECT_EQ(results.numQueries, 3);

  // CCO contains C and O, but not N
  EXPECT_TRUE(results.matches(0, 0)) << "CCO should contain C";
  EXPECT_TRUE(results.matches(0, 1)) << "CCO should contain O";
  EXPECT_FALSE(results.matches(0, 2)) << "CCO should not contain N";

  // CCCC contains C, but not O or N
  EXPECT_TRUE(results.matches(1, 0)) << "CCCC should contain C";
  EXPECT_FALSE(results.matches(1, 1)) << "CCCC should not contain O";
  EXPECT_FALSE(results.matches(1, 2)) << "CCCC should not contain N";

  // benzene contains aromatic c, but not aliphatic C, O, or N
  EXPECT_FALSE(results.matches(2, 0)) << "benzene should not contain aliphatic C";
  EXPECT_FALSE(results.matches(2, 1)) << "benzene should not contain O";
  EXPECT_FALSE(results.matches(2, 2)) << "benzene should not contain N";
}

TEST_P(SubstructureSearchTest, HasSubstructMatchMultiAtomQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({"CCO", "CCC"}, {"CO", "CC"}, targetMols, queryMols);

  HasSubstructMatchResults results;
  hasSubstructMatch(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  // CCO contains CO and CC
  EXPECT_TRUE(results.matches(0, 0)) << "CCO should contain CO";
  EXPECT_TRUE(results.matches(0, 1)) << "CCO should contain CC";

  // CCC contains CC but not CO
  EXPECT_FALSE(results.matches(1, 0)) << "CCC should not contain CO";
  EXPECT_TRUE(results.matches(1, 1)) << "CCC should contain CC";
}

TEST_P(SubstructureSearchTest, HasSubstructMatchEmptyInputs) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Empty targets
  parseMolecules({}, {"C"}, targetMols, queryMols);

  HasSubstructMatchResults results;
  hasSubstructMatch(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream());

  EXPECT_EQ(results.numTargets, 0);
  EXPECT_EQ(results.numQueries, 1);
}

// =============================================================================
// countSubstructMatches Tests
// =============================================================================

TEST_P(SubstructureSearchTest, CountSubstructMatchesBasic) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  parseMolecules({"CCO", "CCCC"}, {"N", "O", "C"}, targetMols, queryMols);

  std::vector<int> counts;
  countSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), counts, algorithm(), stream_.stream());

  const int numTargets = static_cast<int>(targetMols.size());
  const int numQueries = static_cast<int>(queryMols.size());
  EXPECT_EQ(counts.size(), static_cast<size_t>(numTargets * numQueries));

  auto idx = [numQueries](int t, int q) { return t * numQueries + q; };

  EXPECT_EQ(counts[idx(0, 0)], 0);
  EXPECT_EQ(counts[idx(0, 1)], 1);
  EXPECT_EQ(counts[idx(0, 2)], 2);
  EXPECT_EQ(counts[idx(1, 0)], 0);
  EXPECT_EQ(counts[idx(1, 1)], 0);
  EXPECT_EQ(counts[idx(1, 2)], 4);
}

// =============================================================================
// Uniquify Tests
// =============================================================================

TEST_P(SubstructureSearchTest, UniquifyCyclohexaneCCC) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // CCC query on cyclohexane: without uniquify gives 12, with uniquify gives 6
  parseMolecules({"C1CCCCC1"}, {"CCC"}, targetMols, queryMols);

  // Without uniquify
  SubstructSearchResults resultsNoUniquify;
  SubstructSearchConfig  configNoUniquify;
  configNoUniquify.uniquify = false;
  getSubstructMatches(getRawPtrs(targetMols),
                      getRawPtrs(queryMols),
                      resultsNoUniquify,
                      algorithm(),
                      stream_.stream(),
                      configNoUniquify);

  auto rdkitMatchesNonUnique = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsNoUniquify.matchCount(0, 0), static_cast<int>(rdkitMatchesNonUnique.size()))
    << "Without uniquify should match RDKit non-unique count";
  EXPECT_EQ(resultsNoUniquify.matchCount(0, 0), 12) << "CCC on cyclohexane without uniquify should have 12 matches";

  // With uniquify
  SubstructSearchResults resultsUniquify;
  SubstructSearchConfig  configUniquify;
  configUniquify.uniquify = true;
  getSubstructMatches(getRawPtrs(targetMols),
                      getRawPtrs(queryMols),
                      resultsUniquify,
                      algorithm(),
                      stream_.stream(),
                      configUniquify);

  auto rdkitMatchesUnique = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], true);
  EXPECT_EQ(resultsUniquify.matchCount(0, 0), static_cast<int>(rdkitMatchesUnique.size()))
    << "With uniquify should match RDKit unique count";
  EXPECT_EQ(resultsUniquify.matchCount(0, 0), 6) << "CCC on cyclohexane with uniquify should have 6 matches";
}

TEST_P(SubstructureSearchTest, UniquifyCC) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // CC query on hexane
  parseMolecules({"CCCCCC"}, {"CC"}, targetMols, queryMols);

  auto rdkitMatchesNonUnique = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  auto rdkitMatchesUnique    = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], true);

  // Without uniquify
  SubstructSearchResults resultsNoUniquify;
  SubstructSearchConfig  configNoUniquify;
  configNoUniquify.uniquify = false;
  getSubstructMatches(getRawPtrs(targetMols),
                      getRawPtrs(queryMols),
                      resultsNoUniquify,
                      algorithm(),
                      stream_.stream(),
                      configNoUniquify);

  EXPECT_EQ(resultsNoUniquify.matchCount(0, 0), static_cast<int>(rdkitMatchesNonUnique.size()))
    << "CC on hexane without uniquify should match RDKit";

  // With uniquify
  SubstructSearchResults resultsUniquify;
  SubstructSearchConfig  configUniquify;
  configUniquify.uniquify = true;
  getSubstructMatches(getRawPtrs(targetMols),
                      getRawPtrs(queryMols),
                      resultsUniquify,
                      algorithm(),
                      stream_.stream(),
                      configUniquify);

  EXPECT_EQ(resultsUniquify.matchCount(0, 0), static_cast<int>(rdkitMatchesUnique.size()))
    << "CC on hexane with uniquify should match RDKit";

  // Verify uniquify reduces count
  EXPECT_LT(resultsUniquify.matchCount(0, 0), resultsNoUniquify.matchCount(0, 0))
    << "Uniquify should reduce match count for symmetric query";
}

TEST_P(SubstructureSearchTest, UniquifySymmetricQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Symmetric query COC on diethyl ether: 2 matches without uniquify, 1 with
  parseMolecules({"CCOCC"}, {"COC"}, targetMols, queryMols);

  // Without uniquify
  SubstructSearchResults resultsNoUniquify;
  SubstructSearchConfig  configNoUniquify;
  configNoUniquify.uniquify = false;
  getSubstructMatches(getRawPtrs(targetMols),
                      getRawPtrs(queryMols),
                      resultsNoUniquify,
                      algorithm(),
                      stream_.stream(),
                      configNoUniquify);

  auto rdkitMatchesNonUnique = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], false);
  EXPECT_EQ(resultsNoUniquify.matchCount(0, 0), static_cast<int>(rdkitMatchesNonUnique.size()));

  // With uniquify
  SubstructSearchResults resultsUniquify;
  SubstructSearchConfig  configUniquify;
  configUniquify.uniquify = true;
  getSubstructMatches(getRawPtrs(targetMols),
                      getRawPtrs(queryMols),
                      resultsUniquify,
                      algorithm(),
                      stream_.stream(),
                      configUniquify);

  auto rdkitMatchesUnique = getRDKitSubstructMatches(*targetMols[0], *queryMols[0], true);
  EXPECT_EQ(resultsUniquify.matchCount(0, 0), static_cast<int>(rdkitMatchesUnique.size()));
}

TEST_P(SubstructureSearchTest, UniquifyNoEffect) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Asymmetric query CCO: matches are already unique
  parseMolecules({"CCOCC"}, {"CCO"}, targetMols, queryMols);

  SubstructSearchResults resultsNoUniquify;
  SubstructSearchConfig  configNoUniquify;
  configNoUniquify.uniquify = false;
  getSubstructMatches(getRawPtrs(targetMols),
                      getRawPtrs(queryMols),
                      resultsNoUniquify,
                      algorithm(),
                      stream_.stream(),
                      configNoUniquify);

  SubstructSearchResults resultsUniquify;
  SubstructSearchConfig  configUniquify;
  configUniquify.uniquify = true;
  getSubstructMatches(getRawPtrs(targetMols),
                      getRawPtrs(queryMols),
                      resultsUniquify,
                      algorithm(),
                      stream_.stream(),
                      configUniquify);

  // For asymmetric queries, uniquify shouldn't change the count
  EXPECT_EQ(resultsUniquify.matchCount(0, 0), resultsNoUniquify.matchCount(0, 0))
    << "Asymmetric query should have same count with or without uniquify";
}

TEST_P(SubstructureSearchTest, UniquifySingleAtomQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Single atom query: uniquify has no effect (can't have duplicates)
  parseMolecules({"CCCC"}, {"C"}, targetMols, queryMols);

  SubstructSearchResults resultsNoUniquify;
  SubstructSearchConfig  configNoUniquify;
  configNoUniquify.uniquify = false;
  getSubstructMatches(getRawPtrs(targetMols),
                      getRawPtrs(queryMols),
                      resultsNoUniquify,
                      algorithm(),
                      stream_.stream(),
                      configNoUniquify);

  SubstructSearchResults resultsUniquify;
  SubstructSearchConfig  configUniquify;
  configUniquify.uniquify = true;
  getSubstructMatches(getRawPtrs(targetMols),
                      getRawPtrs(queryMols),
                      resultsUniquify,
                      algorithm(),
                      stream_.stream(),
                      configUniquify);

  EXPECT_EQ(resultsUniquify.matchCount(0, 0), resultsNoUniquify.matchCount(0, 0))
    << "Single atom query should have same count with or without uniquify";
  EXPECT_EQ(resultsUniquify.matchCount(0, 0), 4);
}

TEST_P(SubstructureSearchTest, UniquifyBatch) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Multiple targets and queries with uniquify
  parseMolecules({"C1CCCCC1", "CCOCC", "c1ccccc1"}, {"CC", "CCC"}, targetMols, queryMols);

  SubstructSearchConfig config;
  config.uniquify = true;

  SubstructSearchResults results;
  getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream(), config);

  // Verify each pair matches RDKit with uniquify=true
  for (int t = 0; t < results.numTargets; ++t) {
    for (int q = 0; q < results.numQueries; ++q) {
      auto rdkitMatches = getRDKitSubstructMatches(*targetMols[t], *queryMols[q], true);
      EXPECT_EQ(results.matchCount(t, q), static_cast<int>(rdkitMatches.size()))
        << "Mismatch at target " << t << ", query " << q;
    }
  }
}

// =============================================================================
// RDKit Fallback Tests - High Ring Count Molecules
// =============================================================================

TEST_P(SubstructureSearchTest, HighRingCountFallback) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // C60 buckyball has atoms with ring count > 15, requiring RDKit fallback
  // Mix with normal molecule to test batch handling
  const std::string buckyball =
    "c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c7c2c2c7c8c3c3c8c9c4c4c9c%10c5c5c1c1c6c6c%11c2c2c7c3c3c8c4c4c9c5c1c1c6c2c3c41";

  parseMolecules({"c1ccccc1", buckyball}, {"c"}, targetMols, queryMols);

  SubstructSearchResults results;
  EXPECT_NO_THROW(
    getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream()))
    << "Buckyball should be handled via RDKit fallback without throwing";

  for (int t = 0; t < results.numTargets; ++t) {
    auto rdkitMatches = getRDKitSubstructMatches(*targetMols[t], *queryMols[0], false);
    EXPECT_EQ(results.matchCount(t, 0), static_cast<int>(rdkitMatches.size()));
  }
}

TEST_P(SubstructureSearchTest, HypervalentAtomFallback) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Create a hypervalent metal center with 9 bonds (exceeds kMaxBondsPerAtom=8)
  // Use sanitize=false to allow chemically unusual structures
  RDKit::SmilesParserParams params;
  params.sanitize  = false;
  auto hypervalent = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol("[Fe](C)(C)(C)(C)(C)(C)(C)(C)C", params));
  ASSERT_NE(hypervalent, nullptr) << "Failed to parse hypervalent SMILES";
  RDKit::MolOps::symmetrizeSSSR(*hypervalent);

  // Verify the Fe atom has 9 bonds
  ASSERT_GT(hypervalent->getAtomWithIdx(0)->getDegree(), 8u) << "Test molecule should have atom with >8 bonds";

  targetMols.push_back(makeMolFromSmiles("c1ccccc1"));  // Normal molecule
  targetMols.push_back(std::move(hypervalent));
  queryMols.push_back(makeMolFromSmiles("C"));

  SubstructSearchResults results;
  EXPECT_NO_THROW(
    getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream()))
    << "Hypervalent molecule should be handled via RDKit fallback without throwing";

  for (int t = 0; t < results.numTargets; ++t) {
    auto rdkitMatches = getRDKitSubstructMatches(*targetMols[t], *queryMols[0], false);
    EXPECT_EQ(results.matchCount(t, 0), static_cast<int>(rdkitMatches.size()));
  }
}

TEST_P(RecursiveSubstructureSearchTest, DeepRecursionDepthFallback) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Deeply nested recursive SMARTS that exceeds kMaxSmartsNestingDepth (4)
  // Pattern has depth 4: [$([*;$([*;$([*;$([*;$(*-N)])])])])]
  const std::string deepQuery = "[$([*;$([*;$([*;$([*;$(*-N)])])])])]";

  parseMolecules({"CCN", "CCCCN", "CCC"}, {deepQuery}, targetMols, queryMols);

  SubstructSearchResults results;
  EXPECT_NO_THROW(
    getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream()))
    << "Deep recursion query should be handled via RDKit fallback without throwing";

  for (int t = 0; t < results.numTargets; ++t) {
    auto rdkitMatches = getRDKitSubstructMatches(*targetMols[t], *queryMols[0], false);
    EXPECT_EQ(results.matchCount(t, 0), static_cast<int>(rdkitMatches.size()))
      << "Target " << t << " should match RDKit results via fallback";
  }
}
TEST_P(RecursiveSubstructureSearchTest, DeepRecursionMixedWithNormalQueries) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targetMols;
  std::vector<std::unique_ptr<RDKit::ROMol>> queryMols;

  // Mix of normal queries and deep recursive query that needs fallback
  const std::string deepQuery = "[$([*;$([*;$([*;$([*;$(*-N)])])])])]";

  parseMolecules({"CCN", "CCCCN"}, {"C", deepQuery, "N"}, targetMols, queryMols);

  SubstructSearchResults results;
  EXPECT_NO_THROW(
    getSubstructMatches(getRawPtrs(targetMols), getRawPtrs(queryMols), results, algorithm(), stream_.stream()))
    << "Mixed queries should handle deep recursion via RDKit fallback";

  // Verify all results match RDKit
  for (int t = 0; t < results.numTargets; ++t) {
    for (int q = 0; q < results.numQueries; ++q) {
      auto rdkitMatches = getRDKitSubstructMatches(*targetMols[t], *queryMols[q], false);
      EXPECT_EQ(results.matchCount(t, q), static_cast<int>(rdkitMatches.size()))
        << "Target " << t << ", Query " << q << " should match RDKit results";
    }
  }
}
