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

#include <gmock/gmock.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmartsWrite.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include "src/substruct/molecules_device.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"

using nvMolKit::AsyncDeviceVector;
using nvMolKit::checkReturnCode;
using nvMolKit::getMolecule;
using nvMolKit::MoleculesDevice;
using nvMolKit::MoleculesHost;
using nvMolKit::MoleculeType;
using nvMolKit::ScopedStream;

namespace {

std::unique_ptr<RDKit::ROMol> makeQuery(const std::string& smarts) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
  EXPECT_NE(mol, nullptr) << "Failed to parse SMARTS: " << smarts;
  return mol;
}

}  // namespace

// =============================================================================
// Compound Query Tests (OR/NOT)
// =============================================================================

struct CompoundQueryTestCase {
  std::string smarts;
  int         numAtoms;        ///< Expected number of atoms in query
  int         atom0NumLeaves;  ///< Expected numLeaves for first atom's tree
  int         atom0MinInstrs;  ///< Minimum expected instructions for first atom
};

class CompoundQueryParsingTest : public ::testing::TestWithParam<CompoundQueryTestCase> {};

TEST_P(CompoundQueryParsingTest, TreeStructureMatchesExpected) {
  const auto& testCase = GetParam();
  auto        mol      = makeQuery(testCase.smarts);
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  nvMolKit::addQueryToBatch(mol.get(), batch);

  ASSERT_EQ(batch.numMolecules(), 1);
  ASSERT_EQ(static_cast<int>(batch.atomQueryTrees.size()), testCase.numAtoms)
    << "Atom count mismatch for SMARTS: " << testCase.smarts;

  EXPECT_EQ(batch.atomQueryTrees[0].numLeaves, testCase.atom0NumLeaves)
    << "Leaf count mismatch at atom 0 for SMARTS: " << testCase.smarts;

  EXPECT_GE(batch.atomQueryTrees[0].numInstructions, testCase.atom0MinInstrs)
    << "Instruction count too low at atom 0 for SMARTS: " << testCase.smarts;
}

TEST_P(CompoundQueryParsingTest, TreeStructureMatchesOnDevice) {
  const auto& testCase = GetParam();
  auto        mol      = makeQuery(testCase.smarts);
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  nvMolKit::addQueryToBatch(mol.get(), batch);

  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch);

  // Verify device view has query trees populated
  auto view = device.view<MoleculeType::Query>();
  EXPECT_NE(view.atomQueryTrees, nullptr) << "Device should have query trees for SMARTS: " << testCase.smarts;
  EXPECT_NE(view.queryInstructions, nullptr) << "Device should have query instructions for SMARTS: " << testCase.smarts;
  EXPECT_NE(view.queryLeafMasks, nullptr) << "Device should have query leaf masks for SMARTS: " << testCase.smarts;
}

INSTANTIATE_TEST_SUITE_P(OrQueries,
                         CompoundQueryParsingTest,
                         ::testing::Values(
                           // Simple OR queries
                           CompoundQueryTestCase{"[C,N]", 1, 2, 3},      // 2 leaves + 1 OR
                           CompoundQueryTestCase{"[N,O]", 1, 2, 3},      // 2 leaves + 1 OR
                           CompoundQueryTestCase{"[C,N,O]", 1, 3, 5},    // 3 leaves + 2 ORs
                           CompoundQueryTestCase{"[C,N,O,S]", 1, 4, 7},  // 4 leaves + 3 ORs

                           // OR with aromatic/aliphatic variants
                           CompoundQueryTestCase{"[c,n]", 1, 2, 3},  // aromatic c OR n
                           CompoundQueryTestCase{"[C,c]", 1, 2, 3},  // aliphatic C OR aromatic c

                           // Multi-atom OR queries
                           CompoundQueryTestCase{"[C,N][C,N]", 2, 2, 3},  // two atoms, each with OR

                           // OR combined with other properties
                           CompoundQueryTestCase{"[C,N;R1]", 1, 3, 5}),  // (C or N) AND R1: 3 leaves + OR + AND
                         [](const ::testing::TestParamInfo<CompoundQueryTestCase>& info) {
                           std::string name;
                           for (char c : info.param.smarts) {
                             if (std::isalnum(c)) {
                               name += c;
                             } else if (c == ',') {
                               name += "Or";
                             } else if (c == '[') {
                               name += "L";
                             } else if (c == ']') {
                               name += "R";
                             } else if (c == ';') {
                               name += "Semi";
                             } else {
                               name += '_';
                             }
                           }
                           return name;
                         });

INSTANTIATE_TEST_SUITE_P(NotQueries,
                         CompoundQueryParsingTest,
                         ::testing::Values(
                           // Simple NOT queries
                           CompoundQueryTestCase{"[!C]", 1, 1, 2},  // 1 leaf + 1 NOT
                           CompoundQueryTestCase{"[!N]", 1, 1, 2},  // 1 leaf + 1 NOT
                           CompoundQueryTestCase{"[!c]", 1, 1, 2},  // NOT aromatic carbon

                           // NOT with specific properties
                           CompoundQueryTestCase{"[!R1]", 1, 1, 2},  // NOT in exactly 1 ring
                           CompoundQueryTestCase{"[!r6]", 1, 1, 2},  // NOT in 6-membered ring

                           // AND with NOT
                           CompoundQueryTestCase{"[C;!R1]", 1, 2, 4},  // C AND NOT(R1): 2 leaves + NOT + AND
                           CompoundQueryTestCase{"[N;!R1]", 1, 2, 4},  // N AND NOT(R1)

                           // Multi-atom NOT queries
                           CompoundQueryTestCase{"[!C][!N]", 2, 1, 2}),  // two atoms with NOT
                         [](const ::testing::TestParamInfo<CompoundQueryTestCase>& info) {
                           std::string name;
                           for (char c : info.param.smarts) {
                             if (std::isalnum(c)) {
                               name += c;
                             } else if (c == '!') {
                               name += "Not";
                             } else if (c == '[') {
                               name += "L";
                             } else if (c == ']') {
                               name += "R";
                             } else if (c == ';') {
                               name += "Semi";
                             } else {
                               name += '_';
                             }
                           }
                           return name;
                         });

INSTANTIATE_TEST_SUITE_P(
  CombinedQueries,
  CompoundQueryParsingTest,
  ::testing::Values(
    // OR combined with NOT
    CompoundQueryTestCase{"[!C,!N]", 1, 2, 5},  // NOT(C) OR NOT(N): 2 leaves + 2 NOTs + OR

    // Complex combinations with multiple levels
    CompoundQueryTestCase{"[C,N;!R1]", 1, 3, 6},  // (C OR N) AND NOT(R1): 3 leaves + OR + NOT + AND

    // Nested alternating AND/OR: (C AND R1) OR N  (semicolon binds tighter due to left-to-right)
    CompoundQueryTestCase{"[C;R1,N]", 1, 3, 5},  // 3 leaves + AND + OR

    // Multiple ORs with AND: (C OR N OR O) AND R1
    CompoundQueryTestCase{"[C,N,O;R1]", 1, 4, 7},  // 4 leaves + 2 ORs + AND

    // Multiple ANDs with OR: C AND (R1 OR R2) - expressed as [C&R1,C&R2] workaround
    // Actually [C;R1,R2] = (C AND R1) OR R2
    CompoundQueryTestCase{"[C;R1,R2]", 1, 3, 5},  // 3 leaves + AND + OR

    // Deep nesting: ((C OR N) AND R1) OR O
    CompoundQueryTestCase{"[C,N;R1,O]", 1, 4, 7},  // 4 leaves + OR + AND + OR

    // Multiple NOTs with AND
    CompoundQueryTestCase{"[!C;!N]", 1, 2, 5},  // NOT(C) AND NOT(N): 2 leaves + 2 NOTs + AND

    // Triple nesting: (C OR N) AND (R1 OR R2) - need explicit grouping via semicolons
    // [C,N;R1,R2] actually parses as ((C OR N) AND R1) OR R2 due to left-to-right
    CompoundQueryTestCase{"[C,N;R1;R2]", 1, 4, 7}),  // (C OR N) AND R1 AND R2: 4 leaves + OR + 2 ANDs
  [](const ::testing::TestParamInfo<CompoundQueryTestCase>& info) {
    std::string name;
    for (char c : info.param.smarts) {
      if (std::isalnum(c)) {
        name += c;
      } else if (c == ',') {
        name += "Or";
      } else if (c == '!') {
        name += "Not";
      } else if (c == '[') {
        name += "L";
      } else if (c == ']') {
        name += "R";
      } else if (c == ';') {
        name += "Semi";
      } else {
        name += '_';
      }
    }
    return name;
  });

// =============================================================================
// Unsupported Composite Query Tests (should throw)
// =============================================================================

TEST(QueryCompositeTest, RingConnectivityQuerySucceeds) {
  // [x2] ring connectivity query - atoms with 2 ring bonds
  auto mol = makeQuery("[x2]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
}

TEST(QueryCompositeTest, RingConnectivityWithAtomTypeSucceeds) {
  // [Cx2] carbon with 2 ring bonds
  auto mol = makeQuery("[Cx2]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
}

TEST(QueryCompositeTest, ImplicitHCountQuerySucceeds) {
  // [h1] implicit H count query
  auto mol = makeQuery("[h1]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
}

TEST(QueryCompositeTest, HasImplicitHQuerySucceeds) {
  // [h] has any implicit hydrogens
  auto mol = makeQuery("[h]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
}

TEST(QueryCompositeTest, RangRingSizeQuerySucceeds) {
  // [r{5-7}] ring size in range 5-7
  auto mol = makeQuery("[r{5-7}]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
}

TEST(QueryCompositeTest, LessRingSizeQuerySucceeds) {
  // [r{-6}] ring size <= 6
  auto mol = makeQuery("[r{-6}]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
}

TEST(QueryCompositeTest, GreaterRingSizeQuerySucceeds) {
  // [r{5-}] ring size >= 5
  auto mol = makeQuery("[r{5-}]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
}

TEST(QueryCompositeTest, RangeNumRingsQuerySucceeds) {
  // [R{1-3}] in 1-3 rings
  auto mol = makeQuery("[R{1-3}]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
}

TEST(QueryCompositeTest, HeteroatomNeighborsQuerySucceeds) {
  // [z1] atom with 1 heteroatom neighbor (RDKit extension)
  auto mol = makeQuery("[z1]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
}

TEST(QueryCompositeTest, RangeHeteroatomNeighborsQuerySucceeds) {
  // [z{1-2}] atom with 1-2 heteroatom neighbors
  auto mol = makeQuery("[z{1-2}]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
}

TEST(QueryCompositeTest, ChiralityQueryThrows) {
  // [@] chirality query
  auto mol = makeQuery("[C@H](F)(Cl)Br");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, ExcessiveOrBranchesThrows) {
  // Create a SMARTS with many OR alternatives that exceeds kMaxBoolScratchSize.
  // Each alternative needs 1 leaf + 1 OR (except the first), so N alternatives
  // need N leaves + (N-1) ORs = 2N-1 scratch slots.
  // With kMaxBoolScratchSize=128, we need at least 65 alternatives to overflow.
  std::string smarts = "[#1";  // Start with hydrogen
  for (int i = 2; i <= 200; ++i) {
    smarts += ",#" + std::to_string(i);  // Add element 2-100 as OR alternatives
  }
  smarts += "]";  // 100 alternatives = 199 scratch slots needed

  auto mol = makeQuery(smarts);
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, FragmentQueryThrows) {
  auto mol = makeQuery("C.C");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, MultiFragmentQueryThrows) {
  auto mol = makeQuery("C[O;D1].C[O;D1].C[O;D1]");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(QueryCompositeTest, WildcardAtomSucceeds) {
  auto mol = makeQuery("[*]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, AnyAromaticAtomSucceeds) {
  auto mol = makeQuery("[a]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, AnyAliphaticAtomSucceeds) {
  auto mol = makeQuery("[A]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, AnyRingCountSucceeds) {
  auto mol = makeQuery("[R]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, AnyRingSizeSucceeds) {
  auto mol = makeQuery("[r]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, AnyRingWithAtomTypeSucceeds) {
  auto mol = makeQuery("[C;R]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, IsotopeQuerySucceeds) {
  auto mol = makeQuery("[13C]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, DeuteriumQuerySucceeds) {
  auto mol = makeQuery("[2H]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, TritiumQuerySucceeds) {
  auto mol = makeQuery("[3H]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, DegreeQueryD0Succeeds) {
  auto mol = makeQuery("[D0]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, DegreeQueryD1Succeeds) {
  auto mol = makeQuery("[D1]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, DegreeQueryD3Succeeds) {
  auto mol = makeQuery("[D3]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, DegreeWithAtomTypeSucceeds) {
  auto mol = makeQuery("[CD3]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, TotalConnectivityQueryX1Succeeds) {
  auto mol = makeQuery("[X1]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, TotalConnectivityQueryX2Succeeds) {
  auto mol = makeQuery("[X2]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, TotalConnectivityQueryX4Succeeds) {
  auto mol = makeQuery("[X4]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, TotalConnectivityWithAtomTypeSucceeds) {
  auto mol = makeQuery("[CX4]");
  ASSERT_NE(mol, nullptr);
  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
}

TEST(QueryCompositeTest, AnyBondSucceeds) {
  // C~C uses "any bond" (~) which should be supported
  auto mol = makeQuery("C~C");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
  // Each carbon should have 1 "any" bond
  EXPECT_EQ(batch.bondTypeCounts[0].any, 1);
  EXPECT_EQ(batch.bondTypeCounts[1].any, 1);
}

TEST(QueryCompositeTest, MixedBondTypesSucceeds) {
  // C~C-C has both "any" bond and single bond
  auto mol = makeQuery("C~C-C");
  ASSERT_NE(mol, nullptr);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
  // First C: 1 any bond
  EXPECT_EQ(batch.bondTypeCounts[0].any, 1);
  EXPECT_EQ(batch.bondTypeCounts[0].single, 0);
  // Middle C: 1 any bond + 1 single bond
  EXPECT_EQ(batch.bondTypeCounts[1].any, 1);
  EXPECT_EQ(batch.bondTypeCounts[1].single, 1);
  // Last C: 1 single bond
  EXPECT_EQ(batch.bondTypeCounts[2].any, 0);
  EXPECT_EQ(batch.bondTypeCounts[2].single, 1);
}

// =============================================================================
// Batch with Multiple Query Molecules
// =============================================================================

TEST(QueryBatchTest, MultipleQueriesInBatch) {
  auto q1 = makeQuery("[#6]");
  auto q2 = makeQuery("[#7]");
  auto q3 = makeQuery("[#8]");

  MoleculesHost batch;
  nvMolKit::addQueryToBatch(q1.get(), batch);
  nvMolKit::addQueryToBatch(q2.get(), batch);
  nvMolKit::addQueryToBatch(q3.get(), batch);

  EXPECT_EQ(batch.numMolecules(), 3);
  EXPECT_EQ(batch.totalAtoms(), 3);
}

TEST(QueryBatchTest, MultipleQueriesOnDevice) {
  auto q1 = makeQuery("[#6][#6]");  // 2 atoms
  auto q2 = makeQuery("[#7]");      // 1 atom
  auto q3 = makeQuery("[#8][#8]");  // 2 atoms

  MoleculesHost batch;
  nvMolKit::addQueryToBatch(q1.get(), batch);
  nvMolKit::addQueryToBatch(q2.get(), batch);
  nvMolKit::addQueryToBatch(q3.get(), batch);

  EXPECT_EQ(batch.numMolecules(), 3);
  EXPECT_EQ(batch.totalAtoms(), 5);

  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  EXPECT_NO_THROW(device.copyFromHost(batch));
}

TEST(QueryBatchTest, MixedAliphaticAromaticQueries) {
  auto q = makeQuery("c1ccccc1C");  // benzene with aliphatic carbon

  MoleculesHost batch;
  nvMolKit::addQueryToBatch(q.get(), batch);

  EXPECT_EQ(batch.numMolecules(), 1);
  EXPECT_EQ(batch.totalAtoms(), 7);
}

// =============================================================================
// Recursive SMARTS Pattern Extraction Tests
// =============================================================================

TEST(RecursivePatternExtraction, NoRecursivePatterns) {
  auto q = makeQuery("[CH3]");

  EXPECT_FALSE(nvMolKit::hasRecursiveSmarts(q.get()));

  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_TRUE(info.empty());
  EXPECT_EQ(info.size(), 0);
  EXPECT_FALSE(info.hasRecursivePatterns);
}

TEST(RecursivePatternExtraction, SimpleRecursivePattern) {
  auto q = makeQuery("[$([OH])]");

  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(q.get()));

  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_FALSE(info.empty());
  EXPECT_EQ(info.size(), 1);
  EXPECT_TRUE(info.hasRecursivePatterns);

  EXPECT_EQ(info.patterns[0].queryAtomIdx, 0);
  EXPECT_EQ(info.patterns[0].patternId, 0);
  EXPECT_NE(info.patterns[0].queryMol, nullptr);
}

TEST(RecursivePatternExtraction, MultipleRecursivePatterns) {
  auto q = makeQuery("[$([OH]),$([NH2])]");

  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(q.get()));

  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_EQ(info.size(), 2);

  EXPECT_EQ(info.patterns[0].patternId, 0);
  EXPECT_EQ(info.patterns[1].patternId, 1);
  EXPECT_EQ(info.patterns[0].queryAtomIdx, 0);
  EXPECT_EQ(info.patterns[1].queryAtomIdx, 0);
}

TEST(RecursivePatternExtraction, RecursivePatternsOnDifferentAtoms) {
  auto q = makeQuery("[$([OH])]-[$([C]=O)]");

  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(q.get()));

  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_EQ(info.size(), 2);

  std::set<int> atomIndices;
  for (const auto& pattern : info.patterns) {
    atomIndices.insert(pattern.queryAtomIdx);
  }
  EXPECT_EQ(atomIndices.size(), 2);
}

TEST(RecursivePatternExtraction, MixedRecursiveAndNonRecursive) {
  auto q = makeQuery("C[$([OH])]");

  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(q.get()));

  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_EQ(info.size(), 1);
  EXPECT_EQ(info.patterns[0].queryAtomIdx, 1);
}

TEST(RecursivePatternExtraction, ComplexRecursivePattern) {
  auto q = makeQuery("[$([CX3]=[OX1]),$([CX3+]-[OX1-])]");

  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(q.get()));

  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_EQ(info.size(), 2);
}

TEST(RecursivePatternExtraction, MaxPatternsAllowed) {
  std::string smarts = "[C";
  for (int i = 0; i < nvMolKit::RecursivePatternInfo::kMaxPatterns; ++i) {
    smarts += ";$(*-N)";
  }
  smarts += "]";

  auto q = makeQuery(smarts);

  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_EQ(info.size(), nvMolKit::RecursivePatternInfo::kMaxPatterns);

  for (int i = 0; i < nvMolKit::RecursivePatternInfo::kMaxPatterns; ++i) {
    EXPECT_EQ(info.patterns[i].patternId, i);
  }
}

TEST(RecursivePatternExtraction, TooManyPatternsThrows) {
  std::string smarts = "[C";
  for (int i = 0; i < nvMolKit::RecursivePatternInfo::kMaxPatterns + 1; ++i) {
    smarts += ";$(*-N)";
  }
  smarts += "]";

  auto q = makeQuery(smarts);
  EXPECT_THROW(nvMolKit::extractRecursivePatterns(q.get()), std::runtime_error);
}

TEST(RecursivePatternExtraction, NullMolecule) {
  EXPECT_FALSE(nvMolKit::hasRecursiveSmarts(nullptr));

  auto info = nvMolKit::extractRecursivePatterns(nullptr);
  EXPECT_TRUE(info.empty());
}

// =============================================================================
// Nested Recursive Pattern Extraction Tests
// =============================================================================

TEST(NestedRecursivePatternExtraction, SimpleNestedRecursive) {
  auto q = makeQuery("[$([C;$(*-N)])]");
  ASSERT_NE(q, nullptr);
  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(q.get()));

  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_FALSE(info.empty());
  EXPECT_GE(info.size(), 2u);
  EXPECT_GE(info.maxDepth, 1);
  EXPECT_TRUE(info.hasNestedPatterns());
}

TEST(NestedRecursivePatternExtraction, DepthOrdering) {
  auto q = makeQuery("[$([C;$(*-N)])]");
  ASSERT_NE(q, nullptr);

  auto info = nvMolKit::extractRecursivePatterns(q.get());
  ASSERT_GE(info.size(), 2u);

  int maxSeenDepth = 0;
  for (const auto& pattern : info.patterns) {
    EXPECT_GE(pattern.depth, maxSeenDepth) << "Patterns should be sorted by depth";
    maxSeenDepth = std::max(maxSeenDepth, pattern.depth);
  }
}

TEST(NestedRecursivePatternExtraction, DoublyNestedRecursive) {
  auto q = makeQuery("[$([C;$([N;$(*-O)])])]");
  ASSERT_NE(q, nullptr);
  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(q.get()));

  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_GE(info.size(), 3u);
  EXPECT_GE(info.maxDepth, 2);
}

TEST(NestedRecursivePatternExtraction, MultipleNestedOnSameAtom) {
  auto q = makeQuery("[C;$([N;$(*-O)]);$([S;$(*-F)])]");
  ASSERT_NE(q, nullptr);
  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(q.get()));

  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_GE(info.size(), 4u);
}

TEST(NestedRecursivePatternExtraction, NestedPatternIdAssignment) {
  auto q = makeQuery("[$([C;$(*-N)])]");
  ASSERT_NE(q, nullptr);

  auto info = nvMolKit::extractRecursivePatterns(q.get());

  std::set<int> ids;
  for (const auto& p : info.patterns) {
    EXPECT_EQ(ids.count(p.patternId), 0u) << "Duplicate pattern ID: " << p.patternId;
    ids.insert(p.patternId);
  }
}

TEST(NestedRecursivePatternExtraction, LocalIdInParentAssignment) {
  auto q = makeQuery("[C;$(*-N);$(*-O)]");
  ASSERT_NE(q, nullptr);

  auto info = nvMolKit::extractRecursivePatterns(q.get());
  ASSERT_EQ(info.size(), 2u);

  EXPECT_EQ(info.patterns[0].localIdInParent, 0);
  EXPECT_EQ(info.patterns[1].localIdInParent, 1);
}

TEST(NestedRecursivePatternExtraction, ParentPatternIndexTracking) {
  auto q = makeQuery("[$([C;$(*-N)])]");
  ASSERT_NE(q, nullptr);

  auto info = nvMolKit::extractRecursivePatterns(q.get());
  ASSERT_GE(info.size(), 2u);

  int leafCount         = 0;
  int nonLeafCount      = 0;
  int childPatternCount = 0;
  for (const auto& p : info.patterns) {
    if (p.depth == 0) {
      leafCount++;
    } else {
      nonLeafCount++;
    }
    if (p.parentPatternIdx != -1) {
      childPatternCount++;
    }
  }
  EXPECT_GE(leafCount, 1) << "Should have at least one leaf pattern";
  EXPECT_GE(nonLeafCount, 1) << "Should have at least one non-leaf pattern";
  EXPECT_GE(childPatternCount, 1) << "Should have at least one pattern with a parent";
}

TEST(NestedRecursivePatternExtraction, NestedWithOrBranch) {
  auto q = makeQuery("[$([C,$([N;$(*-O)])])]");
  ASSERT_NE(q, nullptr);
  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(q.get()));

  auto info = nvMolKit::extractRecursivePatterns(q.get());
  EXPECT_GE(info.size(), 2u);
}
