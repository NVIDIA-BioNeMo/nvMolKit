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

#include <GraphMol/QueryAtom.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "src/substruct/atom_data_packed.h"
#include "src/substruct/graph_labeler.cuh"
#include "src/substruct/molecules.h"
#include "src/substruct/molecules_device.cuh"
#include "src/testutils/substruct_validation.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"

using nvMolKit::addQueryToBatch;
using nvMolKit::AtomQuery;
using nvMolKit::AtomQueryAtomicNum;
using nvMolKit::AtomQueryIsAromatic;
using nvMolKit::compareLabelMatrices;
using nvMolKit::computeGpuLabelMatrix;
using nvMolKit::MoleculesHost;
using nvMolKit::ScopedStream;

namespace {

std::unique_ptr<RDKit::ROMol> makeMolFromSmiles(const std::string& smiles) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
  return mol;
}

std::unique_ptr<RDKit::ROMol> makeMolFromSmarts(const std::string& smarts) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
  return mol;
}

}  // namespace

// =============================================================================
// Full Graph Labeling Tests with Real Molecules
// =============================================================================

class GraphLabelerTest : public ::testing::Test {
 protected:
  ScopedStream stream_;

  void SetUp() override {}

  void runLabelingTest(const std::string&                 targetSmiles,
                       const std::string&                 querySmarts,
                       std::vector<std::vector<uint8_t>>& expectedMatrix) {
    auto targetMol = makeMolFromSmiles(targetSmiles);
    auto queryMol  = makeMolFromSmarts(querySmarts);
    ASSERT_NE(targetMol, nullptr) << "Failed to parse target: " << targetSmiles;
    ASSERT_NE(queryMol, nullptr) << "Failed to parse query: " << querySmarts;

    auto gpuMatrix = computeGpuLabelMatrix(*targetMol, *queryMol, stream_.stream());

    const int numTargetAtoms = static_cast<int>(gpuMatrix.size());
    const int numQueryAtoms  = numTargetAtoms > 0 ? static_cast<int>(gpuMatrix[0].size()) : 0;

    ASSERT_EQ(expectedMatrix.size(), numTargetAtoms);
    for (int i = 0; i < numTargetAtoms; ++i) {
      ASSERT_EQ(expectedMatrix[i].size(), numQueryAtoms);
      for (int j = 0; j < numQueryAtoms; ++j) {
        EXPECT_EQ(gpuMatrix[i][j], expectedMatrix[i][j]) << "Mismatch at target atom " << i << ", query atom " << j
                                                         << " for target=" << targetSmiles << ", query=" << querySmarts;
      }
    }
  }
};

TEST_F(GraphLabelerTest, EthaneQueryCarbon) {
  // Target: CC (ethane) - 2 carbons, each with 1 bond
  // Query: C (single aliphatic carbon)
  // Both target atoms should match the query
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // Target atom 0 matches query atom 0
    {true}   // Target atom 1 matches query atom 0
  };
  runLabelingTest("CC", "C", expected);
}

TEST_F(GraphLabelerTest, EthaneQueryNitrogen) {
  // Target: CC (ethane)
  // Query: N (nitrogen)
  // No target atoms should match
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}};
  runLabelingTest("CC", "N", expected);
}

TEST_F(GraphLabelerTest, EthanolQueryOxygen) {
  // Target: CCO (ethanol) - C, C, O
  // Query: O (oxygen)
  // Only the oxygen should match
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // First C
    {false},  // Second C
    {true}    // O
  };
  runLabelingTest("CCO", "O", expected);
}

TEST_F(GraphLabelerTest, BenzeneQueryAromaticCarbon) {
  // Target: c1ccccc1 (benzene) - 6 aromatic carbons
  // Query: c (aromatic carbon)
  // All 6 should match
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}, {true}};
  runLabelingTest("c1ccccc1", "c", expected);
}

TEST_F(GraphLabelerTest, BenzeneQueryAliphaticCarbon) {
  // Target: c1ccccc1 (benzene) - 6 aromatic carbons
  // Query: C (aliphatic carbon)
  // None should match (aromatic vs aliphatic)
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}, {false}, {false}, {false}};
  runLabelingTest("c1ccccc1", "C", expected);
}

TEST_F(GraphLabelerTest, PropaneQueryCC) {
  // Target: CCC (propane) - C0-C1-C2
  // Query: CC (two carbons bonded)
  // C0 can match Q0 or Q1 (1 bond each side of query)
  // C1 can match Q0 or Q1 (2 bonds, enough for either end)
  // C2 can match Q0 or Q1 (1 bond each side of query)
  std::vector<std::vector<uint8_t>> expected = {
    {true, true}, // C0: 1 bond, matches both query atoms (each has 1 bond)
    {true, true}, // C1: 2 bonds, matches both query atoms
    {true, true}  // C2: 1 bond, matches both query atoms
  };
  runLabelingTest("CCC", "CC", expected);
}

TEST_F(GraphLabelerTest, MethaneQueryCC) {
  // Target: C (methane) - 1 carbon with 0 heavy-atom bonds
  // Query: CC (two bonded carbons) - each has 1 bond
  std::vector<std::vector<uint8_t>> expected = {
    {true, true}
  };
  runLabelingTest("C", "CC", expected);
}

TEST_F(GraphLabelerTest, TolueneQueryAromaticCarbon) {
  // Target: Cc1ccccc1 (toluene) - 1 aliphatic C + 6 aromatic c
  // Query: c (aromatic carbon)
  // Only the 6 aromatic carbons should match
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // Methyl carbon (aliphatic)
    {true},   // Aromatic
    {true},   // Aromatic
    {true},   // Aromatic
    {true},   // Aromatic
    {true},   // Aromatic
    {true}    // Aromatic
  };
  runLabelingTest("Cc1ccccc1", "c", expected);
}

TEST_F(GraphLabelerTest, PyridineQueryAromaticNitrogen) {
  // Target: c1ccncc1 (pyridine) - 5 aromatic C, 1 aromatic N
  // Query: n (aromatic nitrogen)
  // Only the nitrogen should match
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c
    {false},  // c
    {false},  // c
    {true},   // n
    {false},  // c
    {false}   // c
  };
  runLabelingTest("c1ccncc1", "n", expected);
}

TEST_F(GraphLabelerTest, AtomicNumberOnlyQuery) {
  // Target: CCO
  // Query: [#6] (any carbon regardless of aromaticity)
  // Both carbons should match
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // C
    {true},  // C
    {false}  // O
  };
  runLabelingTest("CCO", "[#6]", expected);
}

// =============================================================================
// Bond Count Matching Tests
// =============================================================================

TEST_F(GraphLabelerTest, BondCountsPreventMatch) {
  // Target: C (methane - 0 bonds to heavy atoms)
  // Query: CC (each carbon has 1 bond)
  std::vector<std::vector<uint8_t>> expected = {
    {true, true}
  };
  runLabelingTest("C", "CC", expected);
}

TEST_F(GraphLabelerTest, CentralCarbonHasMoreBonds) {
  // Target: CC(C)C (isobutane) - central carbon has 3 bonds
  // Query: CC (each carbon has 1 bond)
  // All carbons should match query since they all have >= 1 bond
  std::vector<std::vector<uint8_t>> expected = {
    {true, true}, // Terminal C (1 bond)
    {true, true}, // Central C (3 bonds)
    {true, true}, // Terminal C (1 bond)
    {true, true}  // Terminal C (1 bond)
  };
  runLabelingTest("CC(C)C", "CC", expected);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(GraphLabelerTest, SingleAtomTargetAndQuery) {
  std::vector<std::vector<uint8_t>> expected = {{true}};
  runLabelingTest("C", "C", expected);
}

TEST_F(GraphLabelerTest, SingleAtomNoMatch) {
  std::vector<std::vector<uint8_t>> expected = {{false}};
  runLabelingTest("C", "N", expected);
}

// =============================================================================
// Hydrogen Count Query Tests
// =============================================================================

// Note: SMARTS H count queries like [CH3] check total H count (explicit + implicit).
// For molecules with explicit Hs in SMILES, these work correctly.

TEST_F(GraphLabelerTest, HCountExplicitHydrogens) {
  std::vector<std::vector<uint8_t>> expected = {{true}};
  runLabelingTest("[CH4]", "[CH4]", expected);
}

TEST_F(GraphLabelerTest, HCountExplicitMismatch) {
  std::vector<std::vector<uint8_t>> expected = {{false}};
  runLabelingTest("[CH4]", "[CH3]", expected);
}

TEST_F(GraphLabelerTest, ImplicitHCountMatch) {
  std::vector<std::vector<uint8_t>> expected = {{false}, {true}};
  runLabelingTest("C=N", "[NH]", expected);
}

TEST_F(GraphLabelerTest, ImplicitHCountNoMatch) {
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}};
  runLabelingTest("CC", "[CH2]", expected);
}

// =============================================================================
// Formal Charge Query Tests
// =============================================================================

TEST_F(GraphLabelerTest, FormalChargePositive) {
  // Target: [NH4+] (ammonium)
  // Query: [+1] (any atom with +1 charge)
  std::vector<std::vector<uint8_t>> expected = {
    {true}  // N has +1 charge
  };
  runLabelingTest("[NH4+]", "[+1]", expected);
}

TEST_F(GraphLabelerTest, FormalChargeNegative) {
  // Target: [O-]C=O (formate)
  // Query: [-1] (any atom with -1 charge)
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // O- has -1 charge
    {false},  // C has 0 charge
    {false}   // O has 0 charge
  };
  runLabelingTest("[O-]C=O", "[-1]", expected);
}

TEST_F(GraphLabelerTest, FormalChargeNeutralMolecule) {
  // Target: CCO (ethanol - all neutral)
  // Query: [+1] (any atom with +1 charge)
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // C
    {false},  // C
    {false}   // O
  };
  runLabelingTest("CCO", "[+1]", expected);
}

// =============================================================================
// Hybridization Query Tests
// =============================================================================

TEST_F(GraphLabelerTest, HybridizationSP3) {
  // Target: CCO (ethanol - all sp3)
  // Query: [^3] (sp3 hybridized)
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // C sp3
    {true},  // C sp3
    {true}   // O sp3
  };
  runLabelingTest("CCO", "[^3]", expected);
}

TEST_F(GraphLabelerTest, HybridizationSP2) {
  // Target: C=CC (propene)
  // Query: [^2] (sp2 hybridized)
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // C= sp2
    {true},  // =C sp2
    {false}  // C sp3
  };
  runLabelingTest("C=CC", "[^2]", expected);
}

TEST_F(GraphLabelerTest, HybridizationSP) {
  // Target: C#CC (propyne)
  // Query: [^1] (sp hybridized)
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // C# sp
    {true},  // #C sp
    {false}  // C sp3
  };
  runLabelingTest("C#CC", "[^1]", expected);
}

// =============================================================================
// Ring Membership Query Tests
// =============================================================================

TEST_F(GraphLabelerTest, AnyRingMembershipQuery) {
  // Target: C1CCC1C (cyclobutane with methyl)
  // Query: [R] (atom in any ring)
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // Ring C
    {true},  // Ring C
    {true},  // Ring C
    {true},  // Ring C
    {false}  // Methyl C (not in ring)
  };
  runLabelingTest("C1CCC1C", "[R]", expected);
}

TEST_F(GraphLabelerTest, AnyRingSizeQuery) {
  // Target: C1CCC1C (cyclobutane with methyl)
  // Query: [r] (atom in any ring)
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // Ring C
    {true},  // Ring C
    {true},  // Ring C
    {true},  // Ring C
    {false}  // Methyl C (not in ring)
  };
  runLabelingTest("C1CCC1C", "[r]", expected);
}

TEST_F(GraphLabelerTest, AnyRingWithAtomType) {
  // Target: c1ccncc1C (methylpyridine)
  // Query: [C;R] (aliphatic carbon in any ring)
  // Only aliphatic C in ring matches - the methyl C is not in ring
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c (aromatic)
    {false},  // c (aromatic)
    {false},  // c (aromatic)
    {false},  // n (nitrogen)
    {false},  // c (aromatic)
    {false},  // c (aromatic)
    {false}   // C (not in ring)
  };
  runLabelingTest("c1ccncc1C", "[C;R]", expected);
}

TEST_F(GraphLabelerTest, AnyRingNoRingAtoms) {
  // Target: CCCCC (pentane - no rings)
  // Query: [R] (atom in any ring)
  // No atoms match since there are no rings
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}, {false}, {false}};
  runLabelingTest("CCCCC", "[R]", expected);
}

TEST_F(GraphLabelerTest, RingMembershipExactCount) {
  // Target: C1CCC1C (cyclobutane with methyl)
  // Query: [R1] (atom in exactly 1 ring)
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // Ring C
    {true},  // Ring C
    {true},  // Ring C
    {true},  // Ring C
    {false}  // Methyl C (not in ring)
  };
  runLabelingTest("C1CCC1C", "[R1]", expected);
}

TEST_F(GraphLabelerTest, RingMembershipNotInRing) {
  // Target: C1CCC1C (cyclobutane with methyl)
  // Query: [R0] (atom not in any ring)
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // Ring C
    {false},  // Ring C
    {false},  // Ring C
    {false},  // Ring C
    {true}    // Methyl C (not in ring)
  };
  runLabelingTest("C1CCC1C", "[R0]", expected);
}

TEST_F(GraphLabelerTest, RingMembershipExactlyOneRing) {
  // Target: c1ccccc1 (benzene - each atom in 1 ring)
  // Query: [R1] (atom in exactly 1 ring)
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}, {true}};
  runLabelingTest("c1ccccc1", "[R1]", expected);
}

TEST_F(GraphLabelerTest, RingMembershipTwoRings) {
  // Target: c1ccc2ccccc2c1 (naphthalene - fused atoms in 2 rings)
  // Query: [R2] (atom in exactly 2 rings)
  // Atom order: c0,c1,c2,c3(fusion),c4,c5,c6,c7,c8(fusion),c9
  // Fusion atoms 3 and 8 are in 2 rings
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c0
    {false},  // c1
    {false},  // c2
    {true},   // c3 (fusion)
    {false},  // c4
    {false},  // c5
    {false},  // c6
    {false},  // c7
    {true},   // c8 (fusion)
    {false}   // c9
  };
  runLabelingTest("c1ccc2ccccc2c1", "[R2]", expected);
}

TEST_F(GraphLabelerTest, RingMembershipIndole) {
  // Target: c1ccc2[nH]ccc2c1 (indole - 6-membered benzene fused with 5-membered pyrrole)
  // Query: [R2] (atom in exactly 2 rings)
  // Atom order: c0,c1,c2,c3(fusion),[nH]4,c5,c6,c7(fusion),c8
  // Fusion atoms 3 and 7 are in 2 rings
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c0 (benzene only)
    {false},  // c1 (benzene only)
    {false},  // c2 (benzene only)
    {true},   // c3 (fusion - benzene/pyrrole)
    {false},  // nH (pyrrole only)
    {false},  // c5 (pyrrole only)
    {false},  // c6 (pyrrole only)
    {true},   // c7 (fusion - benzene/pyrrole)
    {false}   // c8 (benzene only)
  };
  runLabelingTest("c1ccc2[nH]ccc2c1", "[R2]", expected);
}

TEST_F(GraphLabelerTest, RingMembershipIndoleSingleRing) {
  // Target: c1ccc2[nH]ccc2c1 (indole)
  // Query: [R1] (atom in exactly 1 ring)
  // Non-fusion atoms are in exactly 1 ring
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // c0 (benzene only)
    {true},   // c1 (benzene only)
    {true},   // c2 (benzene only)
    {false},  // c3 (fusion - in 2 rings)
    {true},   // nH (pyrrole only)
    {true},   // c5 (pyrrole only)
    {true},   // c6 (pyrrole only)
    {false},  // c7 (fusion - in 2 rings)
    {true}    // c8 (benzene only)
  };
  runLabelingTest("c1ccc2[nH]ccc2c1", "[R1]", expected);
}

// =============================================================================
// Ring Size Query Tests
// =============================================================================

TEST_F(GraphLabelerTest, RingSizeFive) {
  // Target: C1CCCC1 (cyclopentane)
  // Query: [r5] (atom in 5-membered ring)
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}};
  runLabelingTest("C1CCCC1", "[r5]", expected);
}

TEST_F(GraphLabelerTest, RingSizeSix) {
  // Target: c1ccccc1 (benzene)
  // Query: [r6] (atom in 6-membered ring)
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}, {true}};
  runLabelingTest("c1ccccc1", "[r6]", expected);
}

TEST_F(GraphLabelerTest, RingSizeMismatch) {
  // Target: C1CCCC1 (cyclopentane - 5-membered)
  // Query: [r6] (atom in 6-membered ring)
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}, {false}, {false}};
  runLabelingTest("C1CCCC1", "[r6]", expected);
}

TEST_F(GraphLabelerTest, RingSizeIndoleFiveMembered) {
  // Target: c1ccc2[nH]ccc2c1 (indole)
  // Query: [r5] (atom in 5-membered ring)
  // Pyrrole atoms (including fusion) have min ring size 5
  // Benzene-only atoms have min ring size 6
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c0 (benzene, min size 6)
    {false},  // c1 (benzene, min size 6)
    {false},  // c2 (benzene, min size 6)
    {true},   // c3 (fusion, min size 5 from pyrrole)
    {true},   // nH (pyrrole, min size 5)
    {true},   // c5 (pyrrole, min size 5)
    {true},   // c6 (pyrrole, min size 5)
    {true},   // c7 (fusion, min size 5 from pyrrole)
    {false}   // c8 (benzene, min size 6)
  };
  runLabelingTest("c1ccc2[nH]ccc2c1", "[r5]", expected);
}

TEST_F(GraphLabelerTest, RingSizeIndoleSixMembered) {
  // Target: c1ccc2[nH]ccc2c1 (indole)
  // Query: [r6] (atom in 6-membered ring)
  // Only benzene-only atoms have min ring size 6
  // Fusion and pyrrole atoms have min ring size 5
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // c0 (benzene, min size 6)
    {true},   // c1 (benzene, min size 6)
    {true},   // c2 (benzene, min size 6)
    {false},  // c3 (fusion, min size 5)
    {false},  // nH (pyrrole, min size 5)
    {false},  // c5 (pyrrole, min size 5)
    {false},  // c6 (pyrrole, min size 5)
    {false},  // c7 (fusion, min size 5)
    {true}    // c8 (benzene, min size 6)
  };
  runLabelingTest("c1ccc2[nH]ccc2c1", "[r6]", expected);
}

// =============================================================================
// Isotope Query Tests
// =============================================================================

TEST_F(GraphLabelerTest, IsotopeCarbon13) {
  // Target: [13C]CC (ethane with carbon-13)
  // Query: [13C] (carbon-13)
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // 13C
    {false},  // C (natural abundance = 0)
    {false}   // C (natural abundance)
  };
  runLabelingTest("[13C]CC", "[13C]", expected);
}

TEST_F(GraphLabelerTest, IsotopeDeuterium) {
  // Target: [2H]C([2H])([2H])[2H] (deuterated methane CD4)
  // Query: [2H] (deuterium)
  // Atom order: D, C, D, D, D
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // D
    {false},  // C
    {true},   // D
    {true},   // D
    {true}    // D
  };
  runLabelingTest("[2H]C([2H])([2H])[2H]", "[2H]", expected);
}

TEST_F(GraphLabelerTest, IsotopeNoMatch) {
  // Target: CC (ethane, natural abundance)
  // Query: [13C] (carbon-13)
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}};
  runLabelingTest("CC", "[13C]", expected);
}

TEST_F(GraphLabelerTest, IsotopeWithAtomType) {
  // Target: [13C]C[13N] - carbon-13 and nitrogen-15 (using nitrogen for variety)
  // Query: [13#6] (isotope 13 + carbon)
  // Note: [13C] queries for carbon-13 specifically
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // 13C
    {false},  // C (natural abundance)
    {false}   // 13N (nitrogen, not carbon)
  };
  runLabelingTest("[13C]C[15N]", "[13C]", expected);
}

// =============================================================================
// Degree Query Tests (D)
// =============================================================================

TEST_F(GraphLabelerTest, DegreeQueryD0) {
  // Target: C (methane - single atom with only implicit H)
  // Query: [D0] (atom with 0 explicit bonds)
  std::vector<std::vector<uint8_t>> expected = {
    {true}  // C (degree 0 - no explicit bonds)
  };
  runLabelingTest("C", "[D0]", expected);
}

TEST_F(GraphLabelerTest, DegreeQueryD1) {
  // Target: CC (ethane)
  // Query: [D1] (atom with 1 explicit bond)
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // CH3 (degree 1)
    {true}   // CH3 (degree 1)
  };
  runLabelingTest("CC", "[D1]", expected);
}

TEST_F(GraphLabelerTest, DegreeQueryD2) {
  // Target: CCC (propane)
  // Query: [D2] (atom with 2 explicit bonds)
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // CH3 (degree 1)
    {true},   // CH2 (degree 2)
    {false}   // CH3 (degree 1)
  };
  runLabelingTest("CCC", "[D2]", expected);
}

TEST_F(GraphLabelerTest, DegreeQueryD3) {
  // Target: CC(C)C (isobutane - central carbon has degree 3)
  // Query: [D3] (atom with 3 explicit bonds)
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // CH3 (degree 1)
    {true},   // central C (degree 3)
    {false},  // CH3 (degree 1)
    {false}   // CH3 (degree 1)
  };
  runLabelingTest("CC(C)C", "[D3]", expected);
}

TEST_F(GraphLabelerTest, DegreeQueryD4) {
  // Target: CC(C)(C)C (neopentane - central carbon has degree 4)
  // Query: [D4] (atom with 4 explicit bonds)
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // CH3 (degree 1)
    {true},   // central C (degree 4)
    {false},  // CH3 (degree 1)
    {false},  // CH3 (degree 1)
    {false}   // CH3 (degree 1)
  };
  runLabelingTest("CC(C)(C)C", "[D4]", expected);
}

TEST_F(GraphLabelerTest, DegreeWithAtomType) {
  // Target: CC(C)C (isobutane)
  // Query: [CD3] (carbon with degree 3)
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // CH3 (degree 1)
    {true},   // central C (degree 3)
    {false},  // CH3 (degree 1)
    {false}   // CH3 (degree 1)
  };
  runLabelingTest("CC(C)C", "[CD3]", expected);
}

// =============================================================================
// Total Connectivity Query Tests (X)
// =============================================================================

TEST_F(GraphLabelerTest, TotalConnectivityX1) {
  // Target: [H][H] (H2 molecule with explicit hydrogens)
  // Query: [X1] (atom with 1 total connection)
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // H (1 bond + 0H = 1)
    {true}   // H (1 bond + 0H = 1)
  };
  runLabelingTest("[H][H]", "[X1]", expected);
}

TEST_F(GraphLabelerTest, TotalConnectivityX2) {
  // Target: C#C (acetylene)
  // Query: [X2] (atom with 2 total connections)
  // Each carbon has degree 1 (triple bond) + 1 H = 2
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // C (1 bond + 1H = 2)
    {true}   // C (1 bond + 1H = 2)
  };
  runLabelingTest("C#C", "[X2]", expected);
}

TEST_F(GraphLabelerTest, TotalConnectivityX3) {
  // Target: C=C (ethene)
  // Query: [X3] (atom with 3 total connections)
  // Each carbon has degree 1 (double bond) + 2 H = 3
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // C (1 bond + 2H = 3)
    {true}   // C (1 bond + 2H = 3)
  };
  runLabelingTest("C=C", "[X3]", expected);
}

TEST_F(GraphLabelerTest, TotalConnectivityX4) {
  // Target: CC (ethane)
  // Query: [X4] (atom with 4 total connections: degree + H count)
  // Each carbon has degree 1 + 3 implicit H = 4
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // C (1 bond + 3H = 4)
    {true}   // C (1 bond + 3H = 4)
  };
  runLabelingTest("CC", "[X4]", expected);
}

TEST_F(GraphLabelerTest, TotalConnectivityWithAtomType) {
  // Target: CC(C)C (isobutane)
  // Query: [CX4] (carbon with 4 total connections)
  // All carbons have 4 total connections (sp3)
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // CH3
    {true},  // central C
    {true},  // CH3
    {true}   // CH3
  };
  runLabelingTest("CC(C)C", "[CX4]", expected);
}

TEST_F(GraphLabelerTest, TotalConnectivityMixed) {
  // Target: CC=C (propene)
  // Query: [X4] (4 connections)
  // CH3 has X4, but C=C carbons have X3
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // CH3 (X4)
    {false},  // =CH (X3)
    {false}   // =CH2 (X3)
  };
  runLabelingTest("CC=C", "[X4]", expected);
}

TEST_F(GraphLabelerTest, TotalConnectivityNoMatch) {
  // Target: CC (ethane - all X4)
  // Query: [X3] (3 connections)
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}};
  runLabelingTest("CC", "[X3]", expected);
}

// =============================================================================
// Wildcard and Any Aromaticity Query Tests
// =============================================================================

TEST_F(GraphLabelerTest, WildcardMatchesAll) {
  // Target: CCO (ethanol)
  // Query: [*] (any atom)
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // C
    {true},  // C
    {true}   // O
  };
  runLabelingTest("CCO", "[*]", expected);
}

TEST_F(GraphLabelerTest, WildcardMatchesMixed) {
  // Target: c1ccccc1C (toluene - mixed aromaticity)
  // Query: [*] (any atom)
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}, {true}, {true}};
  runLabelingTest("c1ccccc1C", "[*]", expected);
}

TEST_F(GraphLabelerTest, AnyAromaticAtom) {
  // Target: c1ccccc1C (toluene)
  // Query: [a] (any aromatic atom)
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // c
    {true},  // c
    {true},  // c
    {true},  // c
    {true},  // c
    {true},  // c
    {false}  // C (aliphatic)
  };
  runLabelingTest("c1ccccc1C", "[a]", expected);
}

TEST_F(GraphLabelerTest, AnyAliphaticAtom) {
  // Target: c1ccccc1C (toluene)
  // Query: [A] (any aliphatic atom)
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c
    {false},  // c
    {false},  // c
    {false},  // c
    {false},  // c
    {false},  // c
    {true}    // C (aliphatic)
  };
  runLabelingTest("c1ccccc1C", "[A]", expected);
}

// =============================================================================
// Explicit AND Query Tests
// =============================================================================

TEST_F(GraphLabelerTest, ExplicitAndAmpersandCarbonInRing) {
  // Target: C1CCC1C (cyclobutane with methyl)
  // Query: [C&R1] (aliphatic carbon AND in exactly 1 ring)
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // Ring C
    {true},  // Ring C
    {true},  // Ring C
    {true},  // Ring C
    {false}  // Methyl C (not in ring)
  };
  runLabelingTest("C1CCC1C", "[C&R1]", expected);
}

TEST_F(GraphLabelerTest, ExplicitAndSemicolonAromaticInRing) {
  // Target: c1ccccc1C (toluene)
  // Query: [c;R1] (aromatic carbon AND in exactly 1 ring)
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // c in ring
    {true},  // c in ring
    {true},  // c in ring
    {true},  // c in ring
    {true},  // c in ring
    {true},  // c in ring
    {false}  // C (aliphatic, not matching aromatic query)
  };
  runLabelingTest("c1ccccc1C", "[c;R1]", expected);
}

TEST_F(GraphLabelerTest, ExplicitAndAtomicNumRing) {
  // Target: c1ccncc1 (pyridine)
  // Query: [#6;R1] (carbon AND in exactly 1 ring)
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // c
    {true},   // c
    {true},   // c
    {false},  // n (nitrogen, not carbon)
    {true},   // c
    {true}    // c
  };
  runLabelingTest("c1ccncc1", "[#6;R1]", expected);
}

TEST_F(GraphLabelerTest, ExplicitAndHCountAromaticity) {
  // Target: c1ccccc1[CH3] (toluene with explicit methyl Hs)
  // Query: [CH3] - aliphatic carbon with 3 Hs
  // Note: H count matching requires explicit Hs in target SMILES
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c (aromatic)
    {false},  // c
    {false},  // c
    {false},  // c
    {false},  // c
    {false},  // c
    {true}    // [CH3] (methyl with explicit Hs)
  };
  runLabelingTest("c1ccccc1[CH3]", "[CH3]", expected);
}

TEST_F(GraphLabelerTest, CombinedRingMembershipAndSize) {
  // Target: c1ccccc1 (benzene)
  // Query: [R1;r6] (in exactly 1 ring AND ring size 6)
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}, {true}};
  runLabelingTest("c1ccccc1", "[R1;r6]", expected);
}

// =============================================================================
// Multiple Chained AND Query Tests (3+ conditions)
// =============================================================================

TEST_F(GraphLabelerTest, TripleAndAromaticRingRingSize) {
  // Target: c1ccccc1 (benzene) - aromatic, in 1 ring, ring size 6
  // Query: [c&R1&r6] (aromatic carbon AND in 1 ring AND ring size 6)
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}, {true}};
  runLabelingTest("c1ccccc1", "[c&R1&r6]", expected);
}

TEST_F(GraphLabelerTest, TripleAndNoMatch) {
  // Target: c1ccccc1 (benzene) - 6-membered ring
  // Query: [c&R1&r5] (aromatic carbon AND in 1 ring AND ring size 5)
  // No match because benzene is 6-membered, not 5
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}, {false}, {false}, {false}};
  runLabelingTest("c1ccccc1", "[c&R1&r5]", expected);
}

TEST_F(GraphLabelerTest, TripleAndCyclopentane) {
  // Target: C1CCCC1 (cyclopentane) - aliphatic, in 1 ring, ring size 5, sp3
  // Query: [C&R1&r5] (aliphatic carbon AND in 1 ring AND ring size 5)
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}};
  runLabelingTest("C1CCCC1", "[C&R1&r5]", expected);
}

TEST_F(GraphLabelerTest, TripleAndWithHybridization) {
  // Target: C1CCCC1 (cyclopentane) - all sp3
  // Query: [C&R1&^3] (aliphatic carbon AND in 1 ring AND sp3)
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}};
  runLabelingTest("C1CCCC1", "[C&R1&^3]", expected);
}

TEST_F(GraphLabelerTest, TripleAndSemicolonSyntax) {
  // Target: C1CCCC1 (cyclopentane)
  // Query: [#6;R1;r5] (carbon AND in 1 ring AND ring size 5) using semicolon
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}};
  runLabelingTest("C1CCCC1", "[#6;R1;r5]", expected);
}

TEST_F(GraphLabelerTest, TripleAndIndolePyrrole) {
  // Target: c1ccc2[nH]ccc2c1 (indole)
  // Query: [c&R1&r5] (aromatic carbon AND in exactly 1 ring AND ring size 5)
  // Only pyrrole carbons (not at fusion) match: atoms 5, 6
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c0 (benzene, r6)
    {false},  // c1 (benzene, r6)
    {false},  // c2 (benzene, r6)
    {false},  // c3 (fusion, R2)
    {false},  // nH (nitrogen, not carbon)
    {true},   // c5 (pyrrole, R1, r5)
    {true},   // c6 (pyrrole, R1, r5)
    {false},  // c7 (fusion, R2)
    {false}   // c8 (benzene, r6)
  };
  runLabelingTest("c1ccc2[nH]ccc2c1", "[c&R1&r5]", expected);
}

// =============================================================================
// OR Query Tests (Boolean Tree)
// =============================================================================

TEST_F(GraphLabelerTest, SimpleOrQuery) {
  // Target: CCN (two carbons, one nitrogen)
  // Query: [C,N] (carbon OR nitrogen)
  // All atoms should match
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}};
  runLabelingTest("CCN", "[C,N]", expected);
}

TEST_F(GraphLabelerTest, OrQueryPartialMatch) {
  // Target: CCO (two carbons, one oxygen)
  // Query: [N,O] (nitrogen OR oxygen)
  // Only oxygen matches
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {true}};
  runLabelingTest("CCO", "[N,O]", expected);
}

TEST_F(GraphLabelerTest, OrQueryNoMatch) {
  // Target: CCC (all carbons)
  // Query: [N,O] (nitrogen OR oxygen)
  // Nothing matches
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}};
  runLabelingTest("CCC", "[N,O]", expected);
}

TEST_F(GraphLabelerTest, ThreeWayOrQuery) {
  // Target: CCNO (carbon, carbon, nitrogen, oxygen)
  // Query: [C,N,O] (carbon OR nitrogen OR oxygen)
  // All atoms match
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}};
  runLabelingTest("CCNO", "[C,N,O]", expected);
}

TEST_F(GraphLabelerTest, OrQueryAromaticAliphatic) {
  // Target: c1ccccc1C (benzene with methyl)
  // Query: [c,C] (aromatic c OR aliphatic C)
  // All 7 carbons match
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}, {true}, {true}};
  runLabelingTest("c1ccccc1C", "[c,C]", expected);
}

// =============================================================================
// NOT Query Tests (Boolean Tree)
// =============================================================================

TEST_F(GraphLabelerTest, SimpleNotQuery) {
  // Target: CCO (two carbons, one oxygen)
  // Query: [!C] (NOT carbon)
  // Only oxygen matches
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {true}};
  runLabelingTest("CCO", "[!C]", expected);
}

TEST_F(GraphLabelerTest, NotQueryMatchesMultiple) {
  // Target: CCNO (carbon, carbon, nitrogen, oxygen)
  // Query: [!C] (NOT carbon)
  // Nitrogen and oxygen match
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {true}, {true}};
  runLabelingTest("CCNO", "[!C]", expected);
}

TEST_F(GraphLabelerTest, NotQueryRingMembership) {
  // Target: C1CC1C (cyclopropane with methyl)
  // Query: [!R1] (NOT in exactly 1 ring)
  // Only methyl carbon (index 3) matches
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}, {true}};
  runLabelingTest("C1CC1C", "[!R1]", expected);
}

TEST_F(GraphLabelerTest, NotQueryNoMatch) {
  // Target: CCC (all carbons)
  // Query: [!C] (NOT carbon)
  // Nothing matches
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}};
  runLabelingTest("CCC", "[!C]", expected);
}

// =============================================================================
// Combined OR/NOT/AND Query Tests (Nested Boolean Trees)
// =============================================================================

TEST_F(GraphLabelerTest, OrWithAndQuery) {
  // Target: C1CCCCC1N (cyclohexane with nitrogen)
  // Query: [C,N;R1] = (C OR N) AND in-1-ring
  // Ring carbons (6) match, external N does not
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}, {true}, {false}};
  runLabelingTest("C1CCCCC1N", "[C,N;R1]", expected);
}

TEST_F(GraphLabelerTest, AndWithNotQuery) {
  // Target: C1CC1CCN (cyclopropane chain with nitrogen)
  // Query: [C;!R1] = carbon AND NOT in-1-ring
  // Ring carbons (0,1,2) don't match; chain carbons (3,4) match; N doesn't match (not carbon)
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}, {true}, {true}, {false}};
  runLabelingTest("C1CC1CCN", "[C;!R1]", expected);
}

TEST_F(GraphLabelerTest, OrThenAndQuery) {
  // Target: C1CCCCC1O (cyclohexane with oxygen)
  // Query: [C;R1,O] - SMARTS precedence: C AND (R1 OR O)
  // Since O here acts as a ring-size constraint (not atom type), only ring carbons match
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}, {true}, {false}};
  runLabelingTest("C1CCCCC1O", "[C;R1,O]", expected);
}

TEST_F(GraphLabelerTest, DeepNestedQuery) {
  // Target: C1CCCC1CCO (cyclopentane chain with oxygen)
  // Query: [C,N;R1,O] - SMARTS precedence: (C OR N) AND (R1 OR O-constraint)
  // Only ring carbons (0-4) match; chain carbons and oxygen don't satisfy ring constraint
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}, {true}, {false}, {false}, {false}};
  runLabelingTest("C1CCCC1CCO", "[C,N;R1,O]", expected);
}

TEST_F(GraphLabelerTest, DoubleNotWithAndQuery) {
  // Target: CCNO (carbon, carbon, nitrogen, oxygen)
  // Query: [!C;!N] = NOT(C) AND NOT(N) - only O matches
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}, {true}};
  runLabelingTest("CCNO", "[!C;!N]", expected);
}

TEST_F(GraphLabelerTest, NotWithOrQuery) {
  // Target: CCNO (carbon, carbon, nitrogen, oxygen)
  // Query: [!C,!N] = NOT(C) OR NOT(N)
  // C: NOT(C)=false, NOT(N)=true => true
  // N: NOT(C)=true, NOT(N)=false => true
  // O: NOT(C)=true, NOT(N)=true => true
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {true}};
  runLabelingTest("CCNO", "[!C,!N]", expected);
}

TEST_F(GraphLabelerTest, OrQueryMultiAtom) {
  // Target: CCN
  // Query: [C,N][C,N] (two-atom query, both can be C or N)
  // C0 can match q0 or q1 (both [C,N])
  // C1 can match q0 or q1
  // N2 can match q0 or q1
  std::vector<std::vector<uint8_t>> expected = {
    {true, true},
    {true, true},
    {true, true}
  };
  runLabelingTest("CCN", "[C,N][C,N]", expected);
}

TEST_F(GraphLabelerTest, NestedAndFailsOnSecondCondition) {
  // Target: CCCN (chain with nitrogen at end)
  // Query: [C,N;R1] = (C OR N) AND in-1-ring
  // All atoms match (C OR N), but NONE are in a ring - all should fail
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}, {false}};
  runLabelingTest("CCCN", "[C,N;R1]", expected);
}

TEST_F(GraphLabelerTest, NestedAndPartialMatch) {
  // Target: C1CC1CCN (cyclopropane + chain + nitrogen)
  // Query: [C,N;R1] = (C OR N) AND in-1-ring
  // Ring carbons (0,1,2): match C AND R1 -> true
  // Chain carbons (3,4): match C but NOT R1 -> false (AND fails)
  // Nitrogen (5): matches N but NOT R1 -> false (AND fails)
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}, {true}, {false}, {false}, {false}};
  runLabelingTest("C1CC1CCN", "[C,N;R1]", expected);
}

TEST_F(GraphLabelerTest, NestedAndWithOrBothFail) {
  // Target: CCSO (carbon, carbon, sulfur, oxygen)
  // Query: [C,N;R1] = (C OR N) AND in-1-ring
  // C atoms: match C but not R1 -> false
  // S atom: doesn't match (C OR N) -> false
  // O atom: doesn't match (C OR N) -> false
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}, {false}};
  runLabelingTest("CCSO", "[C,N;R1]", expected);
}

TEST_F(GraphLabelerTest, OrWithNestedNotAndFails) {
  // Target: C1CC1CN (cyclopropane + methyl + nitrogen)
  // Query: [C;!R1,N] - SMARTS: C AND (!R1 OR something)
  // This tests that ring carbons fail the !R1 check within nested AND
  // Ring C (0,1,2): C matches, but we need to check !R1 behavior
  // Chain C (3): C matches, !R1 matches -> should match
  // N (4): Not C, so fails outer C constraint
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}, {true}, {false}};
  runLabelingTest("C1CC1CN", "[C;!R1,N]", expected);
}

// =============================================================================
// Total Valence Tests
// =============================================================================

TEST_F(GraphLabelerTest, TotalValenceMethane) {
  // Methane: C with 4 hydrogens, total valence = 4
  std::vector<std::vector<uint8_t>> expected = {{true}};
  runLabelingTest("C", "[v4]", expected);
}

TEST_F(GraphLabelerTest, TotalValenceAmmonia) {
  // Ammonia: N with 3 hydrogens, total valence = 3
  std::vector<std::vector<uint8_t>> expected = {{true}};
  runLabelingTest("N", "[v3]", expected);
}

TEST_F(GraphLabelerTest, TotalValenceWater) {
  // Water: O with 2 hydrogens, total valence = 2
  std::vector<std::vector<uint8_t>> expected = {{true}};
  runLabelingTest("O", "[v2]", expected);
}

TEST_F(GraphLabelerTest, TotalValenceMismatch) {
  // Methane has valence 4, not 3
  std::vector<std::vector<uint8_t>> expected = {{false}};
  runLabelingTest("C", "[v3]", expected);
}

TEST_F(GraphLabelerTest, TotalValenceWithAtomType) {
  // Carbon with valence 4
  std::vector<std::vector<uint8_t>> expected = {{true}};
  runLabelingTest("C", "[C&v4]", expected);
}

TEST_F(GraphLabelerTest, TotalValenceEthane) {
  // Ethane: two carbons, each with valence 4
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}};
  runLabelingTest("CC", "[v4]", expected);
}

TEST_F(GraphLabelerTest, TotalValenceEthene) {
  // Ethene: two sp2 carbons, each with valence 4 (double bond counts as 2)
  std::vector<std::vector<uint8_t>> expected = {{true}, {true}};
  runLabelingTest("C=C", "[v4]", expected);
}

TEST_F(GraphLabelerTest, TotalValenceFormaldehyde) {
  // Formaldehyde CH2O: C has valence 4, O has valence 2
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // C (valence 4)
    {false}  // O (valence 2)
  };
  runLabelingTest("C=O", "[v4]", expected);
}

// =============================================================================
// Any Bond (~) Tests
// =============================================================================

TEST_F(GraphLabelerTest, AnyBondMatchesSingle) {
  // Query C~C should match target C-C (single bond)
  // Both target carbons are equivalent, so each can match either query carbon
  std::vector<std::vector<uint8_t>> expected = {
    {true, true},
    {true, true}
  };
  runLabelingTest("CC", "C~C", expected);
}

TEST_F(GraphLabelerTest, AnyBondMatchesDouble) {
  // Query C~C should match target C=C (double bond)
  // Both target carbons are equivalent, so each can match either query carbon
  std::vector<std::vector<uint8_t>> expected = {
    {true, true},
    {true, true}
  };
  runLabelingTest("C=C", "C~C", expected);
}

TEST_F(GraphLabelerTest, AnyBondMatchesAromatic) {
  // Query c~c should match aromatic bonds in benzene
  // All 6 carbons can match either query atom
  std::vector<std::vector<uint8_t>> expected = {
    {true, true},
    {true, true},
    {true, true},
    {true, true},
    {true, true},
    {true, true}
  };
  runLabelingTest("c1ccccc1", "c~c", expected);
}

TEST_F(GraphLabelerTest, AnyBondInRing) {
  // Query with any bond in a ring pattern
  // C1~C~C~C1 should match cyclobutane
  std::vector<std::vector<uint8_t>> expected = {
    {true, true, true, true},
    {true, true, true, true},
    {true, true, true, true},
    {true, true, true, true}
  };
  runLabelingTest("C1CCC1", "C1~C~C~C1", expected);
}

// =============================================================================
// GPU-Optimized Warp-Parallel Labeling Tests
// =============================================================================

TEST_F(GraphLabelerTest, OptimizedWarpParallelLabeling) {
  auto gpuMatrix = computeGpuLabelMatrix(*makeMolFromSmiles("c1ccccc1"), *makeMolFromSmarts("c"), stream_.stream());

  // All 6 aromatic carbons should match the aromatic carbon query
  for (int i = 0; i < 6; ++i) {
    EXPECT_TRUE(gpuMatrix[i][0]) << "Atom " << i << " should match aromatic carbon query";
  }
}

TEST_F(GraphLabelerTest, OptimizedLabelingMultiAtomQuery) {
  // Target: CCC (propane) - 3 carbons
  // Query: CC (two bonded carbons)
  auto gpuMatrix = computeGpuLabelMatrix(*makeMolFromSmiles("CCC"), *makeMolFromSmarts("CC"), stream_.stream());

  // All 3 carbons should match both query atoms (each has >= 1 bond)
  for (int t = 0; t < 3; ++t) {
    for (int q = 0; q < 2; ++q) {
      EXPECT_TRUE(gpuMatrix[t][q]) << "Target atom " << t << " should match query atom " << q;
    }
  }
}

TEST_F(GraphLabelerTest, OptimizedLabelingNoMatches) {
  // Target: C (methane) - single carbon with 0 bonds to heavy atoms
  // Query: CC (two bonded carbons, each with 1 bond)
  auto gpuMatrix = computeGpuLabelMatrix(*makeMolFromSmiles("C"), *makeMolFromSmarts("CC"), stream_.stream());

  EXPECT_TRUE(gpuMatrix[0][0]);
  EXPECT_TRUE(gpuMatrix[0][1]);
}

TEST_F(GraphLabelerTest, OptimizedLabelingMixedMatch) {
  // Target: CCO (ethanol) - 2 carbons, 1 oxygen
  // Query: O (oxygen)
  auto gpuMatrix = computeGpuLabelMatrix(*makeMolFromSmiles("CCO"), *makeMolFromSmarts("O"), stream_.stream());

  // Only oxygen should match
  EXPECT_FALSE(gpuMatrix[0][0]);  // C doesn't match O
  EXPECT_FALSE(gpuMatrix[1][0]);  // C doesn't match O
  EXPECT_TRUE(gpuMatrix[2][0]);   // O matches O
}

// =============================================================================
// Packed Atom Data Tests
// =============================================================================

TEST(AtomDataPackedTest, PackedDataRoundTrip) {
  nvMolKit::AtomDataPacked packed;
  packed.setAtomicNum(6);
  packed.setNumExplicitHs(2);
  packed.setExplicitValence(4);
  packed.setImplicitValence(0);
  packed.setFormalCharge(-1);
  packed.setChiralTag(1);
  packed.setNumRadicalElectrons(0);
  packed.setHybridization(3);
  packed.setMinRingSize(6);
  packed.setNumRings(1);
  packed.setIsAromatic(true);
  packed.setIsotope(13);
  packed.setDegree(3);
  packed.setTotalConnectivity(4);

  EXPECT_EQ(packed.atomicNum(), 6);
  EXPECT_EQ(packed.numExplicitHs(), 2);
  EXPECT_EQ(packed.explicitValence(), 4);
  EXPECT_EQ(packed.implicitValence(), 0);
  EXPECT_EQ(packed.formalCharge(), -1);
  EXPECT_EQ(packed.chiralTag(), 1);
  EXPECT_EQ(packed.numRadicalElectrons(), 0);
  EXPECT_EQ(packed.hybridization(), 3);
  EXPECT_EQ(packed.minRingSize(), 6);
  EXPECT_EQ(packed.numRings(), 1);
  EXPECT_TRUE(packed.isAromatic());
  EXPECT_EQ(packed.isotope(), 13);
  EXPECT_EQ(packed.degree(), 3);
  EXPECT_EQ(packed.totalConnectivity(), 4);
}

TEST(AtomQueryMaskTest, BuildQueryMaskAtomicNum) {
  nvMolKit::AtomDataPacked queryAtom;
  queryAtom.setAtomicNum(6);

  nvMolKit::AtomQueryMask mask = nvMolKit::buildQueryMask(queryAtom, nvMolKit::AtomQueryAtomicNum);

  // Mask should have 0xFF in byte 0 (atomicNum position)
  EXPECT_EQ(mask.maskLo & 0xFF, 0xFF);
  EXPECT_EQ(mask.expectedLo & 0xFF, 6);

  // Rest should be zero
  EXPECT_EQ(mask.maskHi, 0);
  EXPECT_EQ(mask.expectedHi, 0);
}

TEST(AtomQueryMaskTest, BuildQueryMaskAromatic) {
  nvMolKit::AtomDataPacked queryAtom;
  queryAtom.setIsAromatic(true);

  nvMolKit::AtomQueryMask mask = nvMolKit::buildQueryMask(queryAtom, nvMolKit::AtomQueryIsAromatic);

  // isAromatic is now bit 54 (kDegreeByte * 8 + kIsAromaticBit = 6 * 8 + 6)
  constexpr uint64_t aromaticBit =
    1ULL << (nvMolKit::AtomDataPacked::kDegreeByte * 8 + nvMolKit::AtomDataPacked::kIsAromaticBit);
  EXPECT_EQ(mask.maskHi & aromaticBit, aromaticBit);
  EXPECT_EQ(mask.expectedHi & aromaticBit, aromaticBit);  // expect true

  // Rest should be zero
  EXPECT_EQ(mask.maskLo, 0);
  EXPECT_EQ(mask.expectedLo, 0);
}

TEST(BondTypeCountsTest, SufficientBonds) {
  nvMolKit::BondTypeCounts target;
  target.single  = 2;  // 2 single bonds
  target.double_ = 1;  // 1 double bond

  nvMolKit::BondTypeCounts query;
  query.single  = 1;  // 1 single bond
  query.double_ = 1;  // 1 double bond

  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, InsufficientBonds) {
  nvMolKit::BondTypeCounts target;
  target.single = 1;  // 1 single bond

  nvMolKit::BondTypeCounts query;
  query.single = 2;  // 2 single bonds needed

  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, AromaticBonds) {
  nvMolKit::BondTypeCounts target;
  target.aromatic = 2;  // 2 aromatic bonds

  nvMolKit::BondTypeCounts query;
  query.aromatic = 2;  // need 2 aromatic bonds

  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, AromaticInsufficientBonds) {
  nvMolKit::BondTypeCounts target;
  target.aromatic = 1;  // 1 aromatic bond

  nvMolKit::BondTypeCounts query;
  query.aromatic = 2;  // need 2 aromatic bonds

  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, AnyBondMatchesRemaining) {
  // Target has 2 single, 1 double = 3 total
  nvMolKit::BondTypeCounts target;
  target.single  = 2;
  target.double_ = 1;

  // Query needs 1 single, 1 "any" = 2 total
  nvMolKit::BondTypeCounts query;
  query.single = 1;
  query.any    = 1;  // "any" bond from SMARTS (~)

  // Should match: 1 single satisfied, "any" can use remaining (1 single or 1 double)
  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, AnyBondInsufficientTotal) {
  // Target has only 1 single bond
  nvMolKit::BondTypeCounts target;
  target.single = 1;

  // Query needs 1 single + 1 "any" = 2 total
  nvMolKit::BondTypeCounts query;
  query.single = 1;
  query.any    = 1;

  // Should fail: after satisfying single requirement, no bonds left for "any"
  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, AnyBondMatchesAromatic) {
  // Target has 2 aromatic bonds
  nvMolKit::BondTypeCounts target;
  target.aromatic = 2;

  // Query needs 1 "any" = 1 total
  nvMolKit::BondTypeCounts query;
  query.any = 1;

  // Should match: "any" can match the aromatic bond
  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedSingleDoubleTargetHasMore) {
  // Target has 2 single, 2 double
  nvMolKit::BondTypeCounts target;
  target.single  = 2;
  target.double_ = 2;

  // Query needs 1 single, 1 double
  nvMolKit::BondTypeCounts query;
  query.single  = 1;
  query.double_ = 1;

  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedSingleDoubleQueryNeedsMore) {
  // Target has 1 single, 1 double
  nvMolKit::BondTypeCounts target;
  target.single  = 1;
  target.double_ = 1;

  // Query needs 2 single, 1 double - not enough singles
  nvMolKit::BondTypeCounts query;
  query.single  = 2;
  query.double_ = 1;

  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedSingleTripleTargetHasMore) {
  // Target has 3 single, 1 triple
  nvMolKit::BondTypeCounts target;
  target.single = 3;
  target.triple = 1;

  // Query needs 1 single, 1 triple
  nvMolKit::BondTypeCounts query;
  query.single = 1;
  query.triple = 1;

  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedSingleTripleQueryNeedsMore) {
  // Target has 1 single, 1 triple
  nvMolKit::BondTypeCounts target;
  target.single = 1;
  target.triple = 1;

  // Query needs 1 single, 2 triple - not enough triples
  nvMolKit::BondTypeCounts query;
  query.single = 1;
  query.triple = 2;

  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedDoubleAromaticTargetHasMore) {
  // Target has 1 double, 2 aromatic
  nvMolKit::BondTypeCounts target;
  target.double_  = 1;
  target.aromatic = 2;

  // Query needs 1 double, 1 aromatic
  nvMolKit::BondTypeCounts query;
  query.double_  = 1;
  query.aromatic = 1;

  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedDoubleAromaticQueryNeedsMore) {
  // Target has 1 double, 1 aromatic
  nvMolKit::BondTypeCounts target;
  target.double_  = 1;
  target.aromatic = 1;

  // Query needs 2 double, 1 aromatic - not enough doubles
  nvMolKit::BondTypeCounts query;
  query.double_  = 2;
  query.aromatic = 1;

  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedAllTypesPass) {
  // Target has 2 of each type
  nvMolKit::BondTypeCounts target;
  target.single   = 2;
  target.double_  = 2;
  target.triple   = 2;
  target.aromatic = 2;

  // Query needs 1 of each type
  nvMolKit::BondTypeCounts query;
  query.single   = 1;
  query.double_  = 1;
  query.triple   = 1;
  query.aromatic = 1;

  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedAllTypesFailOnOneType) {
  // Target has 2 single, 2 double, 2 triple, but only 0 aromatic
  nvMolKit::BondTypeCounts target;
  target.single   = 2;
  target.double_  = 2;
  target.triple   = 2;
  target.aromatic = 0;

  // Query needs 1 of each - will fail on aromatic
  nvMolKit::BondTypeCounts query;
  query.single   = 1;
  query.double_  = 1;
  query.triple   = 1;
  query.aromatic = 1;

  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedWithAnyPass) {
  // Target has 1 single, 1 double, 1 aromatic = 3 total
  nvMolKit::BondTypeCounts target;
  target.single   = 1;
  target.double_  = 1;
  target.aromatic = 1;

  // Query needs 1 single, 1 any = 2 total
  nvMolKit::BondTypeCounts query;
  query.single = 1;
  query.any    = 1;

  // Should pass: single satisfied, any can match double or aromatic
  EXPECT_TRUE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedWithAnyFailTotal) {
  // Target has 1 single, 1 double = 2 total
  nvMolKit::BondTypeCounts target;
  target.single  = 1;
  target.double_ = 1;

  // Query needs 1 single, 1 double, 1 any = 3 total
  nvMolKit::BondTypeCounts query;
  query.single  = 1;
  query.double_ = 1;
  query.any     = 1;

  // Should fail: specific types satisfied but not enough total for any
  EXPECT_FALSE(target.canMatchQuery(query));
}

TEST(BondTypeCountsTest, MixedWithAnyFailSpecific) {
  // Target has 2 single, 1 aromatic = 3 total
  nvMolKit::BondTypeCounts target;
  target.single   = 2;
  target.aromatic = 1;

  // Query needs 1 double, 1 any = 2 total
  nvMolKit::BondTypeCounts query;
  query.double_ = 1;
  query.any     = 1;

  // Should fail: no double bonds in target, even though total is enough
  EXPECT_FALSE(target.canMatchQuery(query));
}

// =============================================================================
// AtomDataPacked Bit Packing Tests
// =============================================================================

TEST(AtomDataPackedBitPacking, IsAromaticSetAndGet) {
  nvMolKit::AtomDataPacked packed;

  EXPECT_FALSE(packed.isAromatic());

  packed.setIsAromatic(true);
  EXPECT_TRUE(packed.isAromatic());

  packed.setIsAromatic(false);
  EXPECT_FALSE(packed.isAromatic());
}

TEST(AtomDataPackedBitPacking, IsInRingSetAndGet) {
  nvMolKit::AtomDataPacked packed;

  EXPECT_FALSE(packed.isInRing());

  packed.setIsInRing(true);
  EXPECT_TRUE(packed.isInRing());

  packed.setIsInRing(false);
  EXPECT_FALSE(packed.isInRing());
}

TEST(AtomDataPackedBitPacking, IsAromaticAndIsInRingIndependent) {
  nvMolKit::AtomDataPacked packed;

  packed.setIsAromatic(true);
  packed.setIsInRing(false);
  EXPECT_TRUE(packed.isAromatic());
  EXPECT_FALSE(packed.isInRing());

  packed.setIsInRing(true);
  EXPECT_TRUE(packed.isAromatic());
  EXPECT_TRUE(packed.isInRing());

  packed.setIsAromatic(false);
  EXPECT_FALSE(packed.isAromatic());
  EXPECT_TRUE(packed.isInRing());
}

TEST(AtomDataPackedBitPacking, DegreeUseSixBits) {
  nvMolKit::AtomDataPacked packed;

  packed.setDegree(0);
  EXPECT_EQ(packed.degree(), 0);

  packed.setDegree(1);
  EXPECT_EQ(packed.degree(), 1);

  packed.setDegree(63);
  EXPECT_EQ(packed.degree(), 63);

  packed.setDegree(64);
  EXPECT_EQ(packed.degree(), 0);

  packed.setDegree(127);
  EXPECT_EQ(packed.degree(), 63);
}

TEST(AtomDataPackedBitPacking, DegreeDoesNotAffectAromaticOrInRing) {
  nvMolKit::AtomDataPacked packed;

  packed.setIsAromatic(true);
  packed.setIsInRing(true);
  packed.setDegree(42);

  EXPECT_EQ(packed.degree(), 42);
  EXPECT_TRUE(packed.isAromatic());
  EXPECT_TRUE(packed.isInRing());

  packed.setDegree(0);
  EXPECT_EQ(packed.degree(), 0);
  EXPECT_TRUE(packed.isAromatic());
  EXPECT_TRUE(packed.isInRing());
}

TEST(AtomDataPackedBitPacking, AromaticAndInRingDoNotAffectDegree) {
  nvMolKit::AtomDataPacked packed;

  packed.setDegree(35);
  packed.setIsAromatic(true);
  EXPECT_EQ(packed.degree(), 35);

  packed.setIsInRing(true);
  EXPECT_EQ(packed.degree(), 35);

  packed.setIsAromatic(false);
  packed.setIsInRing(false);
  EXPECT_EQ(packed.degree(), 35);
}

// Note: Recursive match bits are no longer stored in AtomDataPacked.
// They are now stored in a separate buffer and passed to evaluateBoolTree.
// Tests for recursive match bit handling are in test_boolean_tree.cu.

// =============================================================================
// New AtomDataPacked Field Tests - Ring Bond Count, Implicit H, Heteroatom Neighbors
// =============================================================================

TEST(AtomDataPackedBitPacking, RingBondCountSetAndGet) {
  nvMolKit::AtomDataPacked packed;

  EXPECT_EQ(packed.ringBondCount(), 0);

  packed.setRingBondCount(0);
  EXPECT_EQ(packed.ringBondCount(), 0);

  packed.setRingBondCount(2);
  EXPECT_EQ(packed.ringBondCount(), 2);

  packed.setRingBondCount(4);
  EXPECT_EQ(packed.ringBondCount(), 4);

  packed.setRingBondCount(7);
  EXPECT_EQ(packed.ringBondCount(), 7);
}

TEST(AtomDataPackedBitPacking, RingBondCountMaxValue) {
  nvMolKit::AtomDataPacked packed;

  packed.setRingBondCount(7);
  EXPECT_EQ(packed.ringBondCount(), 7);
}

TEST(AtomDataPackedBitPacking, NumImplicitHsSetAndGet) {
  nvMolKit::AtomDataPacked packed;

  EXPECT_EQ(packed.numImplicitHs(), 0);

  packed.setNumImplicitHs(0);
  EXPECT_EQ(packed.numImplicitHs(), 0);

  packed.setNumImplicitHs(1);
  EXPECT_EQ(packed.numImplicitHs(), 1);

  packed.setNumImplicitHs(3);
  EXPECT_EQ(packed.numImplicitHs(), 3);

  packed.setNumImplicitHs(7);
  EXPECT_EQ(packed.numImplicitHs(), 7);
}

TEST(AtomDataPackedBitPacking, NumHeteroatomNeighborsSetAndGet) {
  nvMolKit::AtomDataPacked packed;

  EXPECT_EQ(packed.numHeteroatomNeighbors(), 0);

  packed.setNumHeteroatomNeighbors(0);
  EXPECT_EQ(packed.numHeteroatomNeighbors(), 0);

  packed.setNumHeteroatomNeighbors(1);
  EXPECT_EQ(packed.numHeteroatomNeighbors(), 1);

  packed.setNumHeteroatomNeighbors(2);
  EXPECT_EQ(packed.numHeteroatomNeighbors(), 2);

  packed.setNumHeteroatomNeighbors(7);
  EXPECT_EQ(packed.numHeteroatomNeighbors(), 7);
}

TEST(AtomDataPackedBitPacking, NewFieldsIndependent) {
  nvMolKit::AtomDataPacked packed;

  packed.setRingBondCount(2);
  packed.setNumImplicitHs(3);
  packed.setNumHeteroatomNeighbors(1);

  EXPECT_EQ(packed.ringBondCount(), 2);
  EXPECT_EQ(packed.numImplicitHs(), 3);
  EXPECT_EQ(packed.numHeteroatomNeighbors(), 1);

  packed.setRingBondCount(5);
  EXPECT_EQ(packed.ringBondCount(), 5);
  EXPECT_EQ(packed.numImplicitHs(), 3);
  EXPECT_EQ(packed.numHeteroatomNeighbors(), 1);

  packed.setNumImplicitHs(0);
  EXPECT_EQ(packed.ringBondCount(), 5);
  EXPECT_EQ(packed.numImplicitHs(), 0);
  EXPECT_EQ(packed.numHeteroatomNeighbors(), 1);
}

TEST(AtomDataPackedBitPacking, NewFieldsDoNotAffectExistingFields) {
  nvMolKit::AtomDataPacked packed;

  packed.setAtomicNum(6);
  packed.setFormalCharge(0);
  packed.setMinRingSize(6);
  packed.setNumRings(1);
  packed.setDegree(3);
  packed.setIsAromatic(true);
  packed.setIsInRing(true);

  packed.setRingBondCount(2);
  packed.setNumImplicitHs(1);
  packed.setNumHeteroatomNeighbors(2);

  EXPECT_EQ(packed.atomicNum(), 6);
  EXPECT_EQ(packed.formalCharge(), 0);
  EXPECT_EQ(packed.minRingSize(), 6);
  EXPECT_EQ(packed.numRings(), 1);
  EXPECT_EQ(packed.degree(), 3);
  EXPECT_TRUE(packed.isAromatic());
  EXPECT_TRUE(packed.isInRing());
  EXPECT_EQ(packed.ringBondCount(), 2);
  EXPECT_EQ(packed.numImplicitHs(), 1);
  EXPECT_EQ(packed.numHeteroatomNeighbors(), 2);
}

TEST(AtomDataPackedBitPacking, ExistingFieldsDoNotAffectNewFields) {
  nvMolKit::AtomDataPacked packed;

  packed.setRingBondCount(4);
  packed.setNumImplicitHs(2);
  packed.setNumHeteroatomNeighbors(3);

  packed.setAtomicNum(7);
  packed.setMinRingSize(5);
  packed.setDegree(2);
  packed.setIsAromatic(false);
  packed.setIsInRing(true);

  EXPECT_EQ(packed.ringBondCount(), 4);
  EXPECT_EQ(packed.numImplicitHs(), 2);
  EXPECT_EQ(packed.numHeteroatomNeighbors(), 3);
}

// =============================================================================
// Bond Query Match Mask Tests
// =============================================================================

// Helper to get bond match mask from a SMARTS pattern (from first atom's first bond)
uint32_t getBondMatchMask(const std::string& smarts) {
  auto mol = makeMolFromSmarts(smarts);
  EXPECT_NE(mol, nullptr) << "Failed to parse SMARTS: " << smarts;

  MoleculesHost batch;
  addQueryToBatch(mol.get(), batch);

  EXPECT_GT(batch.queryAtomBonds.size(), 0u) << "No query bonds for SMARTS: " << smarts;
  EXPECT_GT(batch.queryAtomBonds[0].degree, 0u) << "First atom has no bonds for SMARTS: " << smarts;

  return batch.queryAtomBonds[0].matchMask[0];
}

// A NeverMatches bond has matchMask == 0 (no target bond type can match)
TEST(BondQueryFlags, SingleBondOnly) {
  // "-" = single bond, should match
  auto mask = getBondMatchMask("C-C");
  EXPECT_NE(mask, 0u) << "Single bond should match something";
}

TEST(BondQueryFlags, AromaticBondOnly) {
  // ":" = aromatic bond
  auto mask = getBondMatchMask("c:c");
  EXPECT_NE(mask, 0u) << "Aromatic bond should match something";
}

TEST(BondQueryFlags, SingleAndAromatic_Impossible) {
  // "-:" = single AND aromatic - impossible combination
  auto mask = getBondMatchMask("C-:C");
  EXPECT_EQ(mask, 0u) << "Single AND aromatic should be NeverMatches (mask=0)";
}

TEST(BondQueryFlags, SingleAndNotAromatic_Valid) {
  // "-!:" = single AND NOT aromatic - valid combination (aliphatic single bond)
  auto mask = getBondMatchMask("C-!:C");
  EXPECT_NE(mask, 0u) << "Single AND NOT aromatic should NOT be NeverMatches";
}

TEST(BondQueryFlags, NotSingleAndAromatic_Valid) {
  // "!-:" = NOT single AND aromatic - valid (aromatic bonds aren't single)
  auto mask = getBondMatchMask("c!-:c");
  EXPECT_NE(mask, 0u) << "NOT single AND aromatic should NOT be NeverMatches";
}

TEST(BondQueryFlags, DoubleAndAromatic_Impossible) {
  // "=:" = double AND aromatic - impossible combination
  auto mask = getBondMatchMask("C=:C");
  EXPECT_EQ(mask, 0u) << "Double AND aromatic should be NeverMatches (mask=0)";
}

TEST(BondQueryFlags, DoubleAndNotAromatic_Valid) {
  // "=!:" = double AND NOT aromatic - valid (aliphatic double bond)
  auto mask = getBondMatchMask("C=!:C");
  EXPECT_NE(mask, 0u) << "Double AND NOT aromatic should NOT be NeverMatches";
}

TEST(BondQueryFlags, TripleAndAromatic_Impossible) {
  // "#:" = triple AND aromatic - impossible combination
  auto mask = getBondMatchMask("C#:C");
  EXPECT_EQ(mask, 0u) << "Triple AND aromatic should be NeverMatches (mask=0)";
}

TEST(BondQueryFlags, TripleAndNotAromatic_Valid) {
  // "#!:" = triple AND NOT aromatic - valid (aliphatic triple bond)
  auto mask = getBondMatchMask("C#!:C");
  EXPECT_NE(mask, 0u) << "Triple AND NOT aromatic should NOT be NeverMatches";
}

TEST(BondQueryFlags, NotSingleNotAromatic_Valid) {
  // "!-!:" = NOT single AND NOT aromatic - valid (double or triple aliphatic)
  auto mask = getBondMatchMask("C!-!:C");
  EXPECT_NE(mask, 0u) << "NOT single AND NOT aromatic should NOT be NeverMatches";
}

TEST(BondQueryFlags, RingBondConstraint) {
  // "@" = ring bond - should only match in-ring bonds (upper 16 bits)
  auto mask = getBondMatchMask("C@C");
  EXPECT_NE(mask, 0u) << "Ring bond should not be NeverMatches";
  // Upper 16 bits are for in-ring bonds, lower 16 bits for not-in-ring
  const uint32_t inRingMask    = mask >> 16;
  const uint32_t notInRingMask = mask & 0xFFFF;
  EXPECT_NE(inRingMask, 0u) << "Ring bond should match in-ring bond types";
  EXPECT_EQ(notInRingMask, 0u) << "Ring bond should not match not-in-ring bonds";
}

TEST(BondQueryFlags, NotRingBondConstraint) {
  // "!@" = not ring bond - should only match not-in-ring bonds (lower 16 bits)
  auto mask = getBondMatchMask("C!@C");
  EXPECT_NE(mask, 0u) << "Not-ring bond should not be NeverMatches";
  // Upper 16 bits are for in-ring bonds, lower 16 bits for not-in-ring
  const uint32_t inRingMask    = mask >> 16;
  const uint32_t notInRingMask = mask & 0xFFFF;
  EXPECT_EQ(inRingMask, 0u) << "Not-ring bond should not match in-ring bonds";
  EXPECT_NE(notInRingMask, 0u) << "Not-ring bond should match not-in-ring bond types";
}