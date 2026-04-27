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

#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "src/tfd/tfd_common.h"
#include "src/tfd/tfd_cpu.h"

namespace {

constexpr double kTolerance = 1e-4;

//! Generate conformers for a molecule using RDKit
void generateConformers(RDKit::ROMol& mol, int numConformers, int seed = 42) {
  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.randomSeed                           = seed;
  params.numThreads                           = 1;
  RDKit::DGeomHelpers::EmbedMultipleConfs(mol, numConformers, params);
}

}  // namespace

class TFDCpuTest : public ::testing::Test {
 protected:
  nvMolKit::TFDCpuGenerator generator_;
};

// =============================================================================
// extractTorsionList
// =============================================================================

TEST_F(TFDCpuTest, ExtractTorsionListRDKitReference) {
  // Compare extractTorsionList output against RDKit CalculateTorsionLists.
  // Parameters: maxDev='equal', symmRadius=2, ignoreColinearBonds=True

  // --- CCCC: 1 non-ring torsion, 0 ring ---
  {
    SCOPED_TRACE("CCCC");
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
    ASSERT_NE(mol, nullptr);
    auto tl = nvMolKit::extractTorsionList(*mol);

    ASSERT_EQ(tl.nonRingTorsions.size(), 1u);
    EXPECT_EQ(tl.ringTorsions.size(), 0u);

    ASSERT_EQ(tl.nonRingTorsions[0].atomQuartets.size(), 1u);
    EXPECT_EQ(tl.nonRingTorsions[0].atomQuartets[0], (std::array<int, 4>{0, 1, 2, 3}));
    EXPECT_NEAR(tl.nonRingTorsions[0].maxDev, 180.0f, 0.01f);
  }

  // --- CCCCC: 2 non-ring torsions, 0 ring ---
  {
    SCOPED_TRACE("CCCCC");
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCC"));
    ASSERT_NE(mol, nullptr);
    auto tl = nvMolKit::extractTorsionList(*mol);

    ASSERT_EQ(tl.nonRingTorsions.size(), 2u);
    EXPECT_EQ(tl.ringTorsions.size(), 0u);

    ASSERT_EQ(tl.nonRingTorsions[0].atomQuartets.size(), 1u);
    EXPECT_EQ(tl.nonRingTorsions[0].atomQuartets[0], (std::array<int, 4>{0, 1, 2, 3}));

    ASSERT_EQ(tl.nonRingTorsions[1].atomQuartets.size(), 1u);
    EXPECT_EQ(tl.nonRingTorsions[1].atomQuartets[0], (std::array<int, 4>{1, 2, 3, 4}));
  }

  // --- CCCCCC: 3 non-ring torsions, 0 ring ---
  {
    SCOPED_TRACE("CCCCCC");
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCCC"));
    ASSERT_NE(mol, nullptr);
    auto tl = nvMolKit::extractTorsionList(*mol);

    ASSERT_EQ(tl.nonRingTorsions.size(), 3u);
    EXPECT_EQ(tl.ringTorsions.size(), 0u);

    EXPECT_EQ(tl.nonRingTorsions[0].atomQuartets[0], (std::array<int, 4>{0, 1, 2, 3}));
    EXPECT_EQ(tl.nonRingTorsions[1].atomQuartets[0], (std::array<int, 4>{1, 2, 3, 4}));
    EXPECT_EQ(tl.nonRingTorsions[2].atomQuartets[0], (std::array<int, 4>{2, 3, 4, 5}));
  }

  // --- c1ccccc1 (benzene): 0 non-ring, 1 ring torsion with 6 quartets ---
  {
    SCOPED_TRACE("c1ccccc1");
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("c1ccccc1"));
    ASSERT_NE(mol, nullptr);
    auto tl = nvMolKit::extractTorsionList(*mol);

    EXPECT_EQ(tl.nonRingTorsions.size(), 0u);
    ASSERT_EQ(tl.ringTorsions.size(), 1u);

    const auto& ringTorsion = tl.ringTorsions[0];
    ASSERT_EQ(ringTorsion.atomQuartets.size(), 6u);

    // Ring torsion maxDev = 180 * exp(-0.025 * (6-14)^2) ≈ 36.34
    EXPECT_NEAR(ringTorsion.maxDev, 36.34f, 0.1f);

    // Quartets from RDKit (ring order: 0, 5, 4, 3, 2, 1)
    EXPECT_EQ(ringTorsion.atomQuartets[0], (std::array<int, 4>{0, 5, 4, 3}));
    EXPECT_EQ(ringTorsion.atomQuartets[1], (std::array<int, 4>{5, 4, 3, 2}));
    EXPECT_EQ(ringTorsion.atomQuartets[2], (std::array<int, 4>{4, 3, 2, 1}));
    EXPECT_EQ(ringTorsion.atomQuartets[3], (std::array<int, 4>{3, 2, 1, 0}));
    EXPECT_EQ(ringTorsion.atomQuartets[4], (std::array<int, 4>{2, 1, 0, 5}));
    EXPECT_EQ(ringTorsion.atomQuartets[5], (std::array<int, 4>{1, 0, 5, 4}));
  }

  // --- CC(=O)[C@H]1CCCC[C@@H]1CN: 2 non-ring + 1 ring (cyclohexane with substituents) ---
  {
    SCOPED_TRACE("CC(=O)[C@H]1CCCC[C@@H]1CN");
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CC(=O)[C@H]1CCCC[C@@H]1CN"));
    ASSERT_NE(mol, nullptr);
    auto tl = nvMolKit::extractTorsionList(*mol);

    ASSERT_EQ(tl.nonRingTorsions.size(), 2u);
    ASSERT_EQ(tl.ringTorsions.size(), 1u);

    EXPECT_EQ(tl.nonRingTorsions[0].atomQuartets[0], (std::array<int, 4>{0, 1, 3, 4}));
    EXPECT_EQ(tl.nonRingTorsions[1].atomQuartets[0], (std::array<int, 4>{3, 8, 9, 10}));

    ASSERT_EQ(tl.ringTorsions[0].atomQuartets.size(), 6u);
    EXPECT_NEAR(tl.ringTorsions[0].maxDev, 36.34f, 0.1f);
    EXPECT_EQ(tl.ringTorsions[0].atomQuartets[0], (std::array<int, 4>{3, 8, 7, 6}));
    EXPECT_EQ(tl.ringTorsions[0].atomQuartets[5], (std::array<int, 4>{4, 3, 8, 7}));
  }

  // --- C[C@@H](CN(C)C(=O)C#CCN)C1CC1: 4 non-ring + 1 ring (cyclopropane + triple bond) ---
  {
    SCOPED_TRACE("C[C@@H](CN(C)C(=O)C#CCN)C1CC1");
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("C[C@@H](CN(C)C(=O)C#CCN)C1CC1"));
    ASSERT_NE(mol, nullptr);
    auto tl = nvMolKit::extractTorsionList(*mol);

    ASSERT_EQ(tl.nonRingTorsions.size(), 4u);
    ASSERT_EQ(tl.ringTorsions.size(), 1u);

    EXPECT_EQ(tl.nonRingTorsions[0].atomQuartets[0], (std::array<int, 4>{0, 1, 2, 3}));
    EXPECT_EQ(tl.nonRingTorsions[1].atomQuartets[0], (std::array<int, 4>{1, 2, 3, 4}));
    EXPECT_EQ(tl.nonRingTorsions[2].atomQuartets[0], (std::array<int, 4>{4, 3, 5, 7}));
    ASSERT_EQ(tl.nonRingTorsions[3].atomQuartets.size(), 2u);
    EXPECT_EQ(tl.nonRingTorsions[3].atomQuartets[0], (std::array<int, 4>{0, 1, 11, 12}));
    EXPECT_EQ(tl.nonRingTorsions[3].atomQuartets[1], (std::array<int, 4>{0, 1, 11, 13}));

    ASSERT_EQ(tl.ringTorsions[0].atomQuartets.size(), 3u);
    EXPECT_NEAR(tl.ringTorsions[0].maxDev, 8.74f, 0.1f);
  }
}

// =============================================================================
// computeTorsionWeights
// =============================================================================

TEST_F(TFDCpuTest, ComputeTorsionWeightsRDKitReference) {
  // Compare computeTorsionWeights output against RDKit CalculateTorsionWeights.
  // Reference generated with: TorsionFingerprints.CalculateTorsionWeights(mol, ignoreColinearBonds=True)

  struct TestCase {
    const char*        smiles;
    std::vector<float> expectedWeights;
  };

  // clang-format off
  std::vector<TestCase> cases = {
    {"CCCC",   {1.0f}},                                             // 1 torsion: central bond
    {"CCCCC",  {1.0f, 0.1f}},                                      // 2 torsions: central + terminal
    {"CCCCCC", {0.1f, 1.0f, 0.1f}},                                // 3 torsions: symmetric around center
    {"CC(=O)[C@H]1CCCC[C@@H]1CN", {0.3593813664f, 0.3593813664f, 0.1748047999f}},  // cyclohexane + substituents
    {"C[C@@H](CN(C)C(=O)C#CCN)C1CC1", {0.5623413252f, 1.0f, 0.5623413252f, 0.1f, 0.0025021508f}},  // cyclopropane + triple bond
  };
  // clang-format on

  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.smiles);
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(tc.smiles));
    ASSERT_NE(mol, nullptr);

    auto tl      = nvMolKit::extractTorsionList(*mol);
    auto weights = nvMolKit::computeTorsionWeights(*mol, tl);

    ASSERT_EQ(weights.size(), tc.expectedWeights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
      EXPECT_NEAR(weights[i], tc.expectedWeights[i], 5e-4f) << "Weight[" << i << "] mismatch";
    }
  }
}

// =============================================================================
// buildTFDSystem
// =============================================================================

TEST_F(TFDCpuTest, BuildTFDSystem) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 5);
  ASSERT_EQ(mol->getNumConformers(), 5);

  nvMolKit::TFDComputeOptions options;
  auto                        system = nvMolKit::buildTFDSystem(*mol, options);

  EXPECT_EQ(system.numMolecules(), 1);
  EXPECT_EQ(system.molDescriptors[0].numConformers, 5);
  EXPECT_GT(system.totalTorsions(), 0);

  // TFD output size: 5 conformers = 5*4/2 = 10 pairs
  EXPECT_EQ(system.totalTFDOutputs(), 10);

  // Single-quartet molecule: totalQuartets == totalTorsions
  EXPECT_EQ(system.totalQuartets(), system.totalTorsions());
}

TEST_F(TFDCpuTest, BuildTFDSystemMultiQuartet) {
  // Verify multi-quartet fields for CC(C)CC (isopentane)
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CC(C)CC"));
  ASSERT_NE(mol, nullptr);
  generateConformers(*mol, 3);

  nvMolKit::TFDComputeOptions options;
  auto                        system = nvMolKit::buildTFDSystem(*mol, options);

  EXPECT_EQ(system.numMolecules(), 1);
  EXPECT_EQ(system.totalTorsions(), 1);  // 1 torsion
  EXPECT_EQ(system.totalQuartets(), 2);  // 2 quartets

  // Verify torsion type
  ASSERT_EQ(system.torsionTypes.size(), 1u);
  EXPECT_EQ(system.torsionTypes[0], nvMolKit::TorsionType::Symmetric);

  // Verify quartetStarts CSR
  ASSERT_EQ(system.quartetStarts.size(), 2u);  // 1 torsion + 1
  EXPECT_EQ(system.quartetStarts[0], 0);
  EXPECT_EQ(system.quartetStarts[1], 2);

  // Verify 2 quartets in torsionAtoms
  ASSERT_EQ(system.torsionAtoms.size(), 2u);
}

TEST_F(TFDCpuTest, BuildTFDSystemRingTorsion) {
  // Verify multi-quartet fields for C1CCCCC1 (cyclohexane)
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("C1CCCCC1"));
  ASSERT_NE(mol, nullptr);
  generateConformers(*mol, 3);

  nvMolKit::TFDComputeOptions options;
  auto                        system = nvMolKit::buildTFDSystem(*mol, options);

  EXPECT_EQ(system.totalTorsions(), 1);  // 1 ring torsion
  EXPECT_EQ(system.totalQuartets(), 6);  // 6 quartets in the ring

  ASSERT_EQ(system.torsionTypes.size(), 1u);
  EXPECT_EQ(system.torsionTypes[0], nvMolKit::TorsionType::Ring);
}

// =============================================================================
// computeDihedralAngles
// =============================================================================

TEST_F(TFDCpuTest, KnownDihedralAngle) {
  // Test dihedral angle computation with hand-crafted geometry (RDKit-independent).
  // Create n-butane and manually set conformer coordinates to known dihedral angles.
  //
  // Dihedral angle definition: Looking down the C1-C2 bond (central bond),
  // the angle between C0 and C3 measured clockwise from C0.
  // - Trans (180°): C0 and C3 on opposite sides
  // - Gauche (60°): C0 and C3 at 60° apart

  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
  ASSERT_NE(mol, nullptr);
  ASSERT_EQ(mol->getNumAtoms(), 4);

  // Trans conformer (180°): C0-C1 and C2-C3 vectors are antiparallel
  {
    auto* conf = new RDKit::Conformer(4);
    conf->setId(0);
    conf->setAtomPos(0, RDGeom::Point3D(1.0, 0.0, 0.0));
    conf->setAtomPos(1, RDGeom::Point3D(0.0, 0.0, 0.0));
    conf->setAtomPos(2, RDGeom::Point3D(0.0, 1.5, 0.0));
    conf->setAtomPos(3, RDGeom::Point3D(1.0, 1.5, 0.0));
    mol->addConformer(conf, true);
  }

  // Gauche conformer (60°): C3 rotated to give 60° dihedral
  {
    auto* conf = new RDKit::Conformer(4);
    conf->setId(1);
    conf->setAtomPos(0, RDGeom::Point3D(1.0, 0.0, 0.0));
    conf->setAtomPos(1, RDGeom::Point3D(0.0, 0.0, 0.0));
    conf->setAtomPos(2, RDGeom::Point3D(0.0, 1.5, 0.0));
    conf->setAtomPos(3, RDGeom::Point3D(-0.5, 1.5, 0.866));
    mol->addConformer(conf, true);
  }

  ASSERT_EQ(mol->getNumConformers(), 2);

  nvMolKit::TFDComputeOptions options;
  options.useWeights = false;
  options.maxDevMode = nvMolKit::TFDMaxDevMode::Equal;

  auto tl     = nvMolKit::extractTorsionList(*mol, options.maxDevMode, options.symmRadius, options.ignoreColinearBonds);
  auto angles = generator_.computeDihedralAngles(*mol, tl);

  int totalQuartets = static_cast<int>(tl.totalCount());
  ASSERT_GE(totalQuartets, 1);

  EXPECT_NEAR(angles[0], 180.0f, 5.0f) << "Trans conformer should have ~180° dihedral";
  EXPECT_NEAR(angles[totalQuartets], 60.0f, 5.0f) << "Gauche conformer should have ~60° dihedral";

  // TFD: difference = |180 - 60| = 120°, normalized by 180° ≈ 0.667
  auto tfdMatrix = generator_.GetTFDMatrix(*mol, options);
  ASSERT_EQ(tfdMatrix.size(), 1u);
  EXPECT_NEAR(tfdMatrix[0], 0.667, 0.05);
}

TEST_F(TFDCpuTest, ComputeDihedralAnglesRDKitReference) {
  // Compare computeDihedralAngles output against RDKit rdMolTransforms.GetDihedralDeg.
  // 4 conformers per molecule, seed=42, ETKDGv3.
  //
  // NOTE: Our dihedral convention is offset by 180° from RDKit's GetDihedralDeg.
  // This does NOT affect TFD (circularDifference is invariant to shared offset).
  // Reference values below are RDKit values + 180° (mod 360).
  //
  // Original RDKit values generated with:
  //   angle = rdMolTransforms.GetDihedralDeg(conf, a, b, c, d)
  //   our_angle = (angle + 180) % 360

  struct TestCase {
    const char*        smiles;
    int                numTorsions;
    std::vector<float> expectedAngles;  // [numConf * numTors], our convention
  };

  // clang-format off
  std::vector<TestCase> cases = {
    {"CCCC", 1, {
      120.0130f,  // conf[0] tors[0]  (RDKit: 300.013)
        0.0000f,  // conf[1] tors[0]  (RDKit: 180.000)
      120.0000f,  // conf[2] tors[0]  (RDKit: 300.000)
        0.0000f   // conf[3] tors[0]  (RDKit: 180.000)
    }},
    {"CCCCC", 2, {
      239.9983f, 240.0002f,   // conf[0]  (RDKit: 60.0, 60.0)
      119.9983f, 240.0001f,   // conf[1]  (RDKit: 300.0, 60.0)
      359.9974f, 240.0025f,   // conf[2]  (RDKit: 180.0, 60.0)
        0.0037f, 120.0134f    // conf[3]  (RDKit: 180.0, 300.0)
    }},
    {"CCCCCC", 3, {
      240.0055f, 359.9997f, 119.9928f,  // conf[0]  (RDKit: 60, 180, 300)
      359.9961f, 239.9959f, 119.9996f,  // conf[1]  (RDKit: 180, 60, 300)
      239.9980f, 359.9978f, 239.9987f,  // conf[2]  (RDKit: 60, 180, 60)
      239.9980f, 119.9980f, 119.9976f   // conf[3]  (RDKit: 60, 300, 300)
    }},
  };
  // clang-format on

  // Use circular distance to handle 0°/360° wrap-around
  constexpr float kAngleTolerance = 0.05f;  // degrees; generous for float vs double

  nvMolKit::TFDComputeOptions options;
  options.useWeights          = true;
  options.maxDevMode          = nvMolKit::TFDMaxDevMode::Equal;
  options.symmRadius          = 2;
  options.ignoreColinearBonds = true;

  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.smiles);
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(tc.smiles));
    ASSERT_NE(mol, nullptr);

    generateConformers(*mol, 4, 42);
    ASSERT_EQ(mol->getNumConformers(), 4);

    auto tl = nvMolKit::extractTorsionList(*mol, options.maxDevMode, options.symmRadius, options.ignoreColinearBonds);
    auto angles = generator_.computeDihedralAngles(*mol, tl);

    int numConf = 4;
    int numTors = tc.numTorsions;
    ASSERT_EQ(static_cast<int>(angles.size()), numConf * numTors);
    ASSERT_EQ(static_cast<int>(tc.expectedAngles.size()), numConf * numTors);

    for (size_t i = 0; i < angles.size(); ++i) {
      // Circular distance handles 0°/360° boundary
      float diff = std::abs(angles[i] - tc.expectedAngles[i]);
      if (diff > 180.0f) {
        diff = 360.0f - diff;
      }
      EXPECT_LE(diff, kAngleTolerance) << "Angle[" << i << "] (conf=" << i / numTors << " tors=" << i % numTors
                                       << "): got " << angles[i] << ", expected " << tc.expectedAngles[i];
    }
  }
}

// =============================================================================
// GetTFDMatrix (pipeline)
// =============================================================================

TEST_F(TFDCpuTest, ComputeTFDMatrixSelfComparison) {
  // Identical conformers should produce TFD = 0
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 1);
  ASSERT_EQ(mol->getNumConformers(), 1);

  // Duplicate the conformer
  RDKit::Conformer conf = mol->getConformer(0);
  conf.setId(1);
  mol->addConformer(new RDKit::Conformer(conf), true);
  ASSERT_EQ(mol->getNumConformers(), 2);

  nvMolKit::TFDComputeOptions options;
  auto                        tfdMatrix = generator_.GetTFDMatrix(*mol, options);

  ASSERT_EQ(tfdMatrix.size(), 1u);
  EXPECT_NEAR(tfdMatrix[0], 0.0, kTolerance);
}

TEST_F(TFDCpuTest, NoTorsionsMolecule) {
  // Methane has no rotatable bonds — TFD should be zero
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("C"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 3);

  nvMolKit::TFDComputeOptions options;
  auto                        tfdMatrix = generator_.GetTFDMatrix(*mol, options);

  ASSERT_EQ(tfdMatrix.size(), 3u);  // 3 conformers = 3 pairs
  for (double tfd : tfdMatrix) {
    EXPECT_NEAR(tfd, 0.0, kTolerance);
  }
}

TEST_F(TFDCpuTest, SingleConformer) {
  // Single conformer should return empty matrix (no pairs)
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 1);
  ASSERT_EQ(mol->getNumConformers(), 1);

  nvMolKit::TFDComputeOptions options;
  auto                        tfdMatrix = generator_.GetTFDMatrix(*mol, options);

  EXPECT_TRUE(tfdMatrix.empty());
}

TEST_F(TFDCpuTest, UseWeightsOption) {
  // Weighted and unweighted TFD should differ for multi-torsion molecules
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCCCCC"));  // n-octane
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 4);
  ASSERT_GE(mol->getNumConformers(), 2);

  nvMolKit::TFDComputeOptions optionsWithWeights;
  optionsWithWeights.useWeights = true;

  nvMolKit::TFDComputeOptions optionsNoWeights;
  optionsNoWeights.useWeights = false;

  auto tfdWithWeights = generator_.GetTFDMatrix(*mol, optionsWithWeights);
  auto tfdNoWeights   = generator_.GetTFDMatrix(*mol, optionsNoWeights);

  ASSERT_EQ(tfdWithWeights.size(), tfdNoWeights.size());
  ASSERT_GT(tfdWithWeights.size(), 0u);

  for (size_t i = 0; i < tfdWithWeights.size(); ++i) {
    EXPECT_GE(tfdWithWeights[i], 0.0);
    EXPECT_GE(tfdNoWeights[i], 0.0);
  }

  bool anyDifferent = false;
  for (size_t i = 0; i < tfdWithWeights.size(); ++i) {
    if (std::abs(tfdWithWeights[i] - tfdNoWeights[i]) > 1e-6) {
      anyDifferent = true;
      break;
    }
  }
  EXPECT_TRUE(anyDifferent) << "Weighted and unweighted TFD should differ for n-octane";
}

TEST_F(TFDCpuTest, MaxDevModes) {
  // Both Equal and Spec modes should produce valid finite results
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 3);

  nvMolKit::TFDComputeOptions optionsEqual;
  optionsEqual.maxDevMode = nvMolKit::TFDMaxDevMode::Equal;

  nvMolKit::TFDComputeOptions optionsSpec;
  optionsSpec.maxDevMode = nvMolKit::TFDMaxDevMode::Spec;

  auto tfdEqual = generator_.GetTFDMatrix(*mol, optionsEqual);
  auto tfdSpec  = generator_.GetTFDMatrix(*mol, optionsSpec);

  ASSERT_EQ(tfdEqual.size(), tfdSpec.size());

  for (size_t i = 0; i < tfdEqual.size(); ++i) {
    EXPECT_TRUE(std::isfinite(tfdEqual[i]));
    EXPECT_TRUE(std::isfinite(tfdSpec[i]));
  }
}

TEST_F(TFDCpuTest, MaxDevModesMultiQuartet) {
  // Spec maxDev mode uses torsion-specific normalization for non-ring torsions:
  //   - CC(C)CC symmetric torsion: maxDev=90 (Spec) vs 180 (Equal)
  // Ring torsions always use the ring-specific formula (matching RDKit),
  // so Equal vs Spec produces identical results for ring-only molecules.

  // --- CC(C)CC: Spec and Equal should differ ---
  {
    SCOPED_TRACE("CC(C)CC");

    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CC(C)CC"));
    ASSERT_NE(mol, nullptr);

    generateConformers(*mol, 4);
    ASSERT_GE(mol->getNumConformers(), 2);

    nvMolKit::TFDComputeOptions optionsEqual;
    optionsEqual.maxDevMode = nvMolKit::TFDMaxDevMode::Equal;

    nvMolKit::TFDComputeOptions optionsSpec;
    optionsSpec.maxDevMode = nvMolKit::TFDMaxDevMode::Spec;

    auto tfdEqual = generator_.GetTFDMatrix(*mol, optionsEqual);
    auto tfdSpec  = generator_.GetTFDMatrix(*mol, optionsSpec);

    ASSERT_EQ(tfdEqual.size(), tfdSpec.size());
    ASSERT_GT(tfdEqual.size(), 0u);

    for (size_t i = 0; i < tfdEqual.size(); ++i) {
      EXPECT_TRUE(std::isfinite(tfdEqual[i]));
      EXPECT_TRUE(std::isfinite(tfdSpec[i]));
      EXPECT_GE(tfdEqual[i], 0.0);
      EXPECT_GE(tfdSpec[i], 0.0);
    }

    // Symmetric non-ring torsion: maxDev=90 (Spec) vs 180 (Equal)
    bool anyDifferent = false;
    for (size_t i = 0; i < tfdEqual.size(); ++i) {
      if (std::abs(tfdEqual[i] - tfdSpec[i]) > 1e-6) {
        anyDifferent = true;
        break;
      }
    }
    EXPECT_TRUE(anyDifferent) << "Spec and Equal should differ for symmetric non-ring torsion";
  }

  // --- C1CCCCC1: Spec and Equal should be identical (ring maxDev is always formula-based) ---
  {
    SCOPED_TRACE("C1CCCCC1");

    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("C1CCCCC1"));
    ASSERT_NE(mol, nullptr);

    generateConformers(*mol, 4);
    ASSERT_GE(mol->getNumConformers(), 2);

    nvMolKit::TFDComputeOptions optionsEqual;
    optionsEqual.maxDevMode = nvMolKit::TFDMaxDevMode::Equal;

    nvMolKit::TFDComputeOptions optionsSpec;
    optionsSpec.maxDevMode = nvMolKit::TFDMaxDevMode::Spec;

    auto tfdEqual = generator_.GetTFDMatrix(*mol, optionsEqual);
    auto tfdSpec  = generator_.GetTFDMatrix(*mol, optionsSpec);

    ASSERT_EQ(tfdEqual.size(), tfdSpec.size());
    ASSERT_GT(tfdEqual.size(), 0u);

    for (size_t i = 0; i < tfdEqual.size(); ++i) {
      EXPECT_TRUE(std::isfinite(tfdEqual[i]));
      EXPECT_GE(tfdEqual[i], 0.0);
      // Ring-only molecule: maxDev mode has no effect
      EXPECT_NEAR(tfdEqual[i], tfdSpec[i], 1e-10) << "Ring torsion maxDev should be identical for Equal and Spec modes";
    }
  }
}

TEST_F(TFDCpuTest, IgnoreColinearBondsFalse) {
  // Test ignoreColinearBonds=false with a molecule containing a triple bond.
  // With ignoreColinearBonds=true (default), bonds adjacent to triple bonds are skipped.
  // With false, alternative atoms are found and torsions are included.
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC#CCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 4);
  ASSERT_GE(mol->getNumConformers(), 2);

  nvMolKit::TFDComputeOptions optionsIgnore;
  optionsIgnore.ignoreColinearBonds = true;

  nvMolKit::TFDComputeOptions optionsKeep;
  optionsKeep.ignoreColinearBonds = false;

  auto torsionsIgnore = nvMolKit::extractTorsionList(*mol, nvMolKit::TFDMaxDevMode::Equal, 2, true);
  auto torsionsKeep   = nvMolKit::extractTorsionList(*mol, nvMolKit::TFDMaxDevMode::Equal, 2, false);

  // With ignoreColinearBonds=false, we should get more (or at least as many) torsions
  EXPECT_GE(torsionsKeep.totalCount(), torsionsIgnore.totalCount());

  auto tfdIgnore = generator_.GetTFDMatrix(*mol, optionsIgnore);
  auto tfdKeep   = generator_.GetTFDMatrix(*mol, optionsKeep);

  // Both should produce valid results
  for (double val : tfdIgnore) {
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_GE(val, 0.0);
  }
  for (double val : tfdKeep) {
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_GE(val, 0.0);
  }
}

TEST_F(TFDCpuTest, CompareWithRDKitReference) {
  // Compare full TFD pipeline against pre-computed RDKit reference values.
  // Includes single-quartet, multi-quartet (ring, symmetric), ring+substituent,
  // and triple-bond molecules.
  //
  // Reference values generated with RDKit Python:
  //   mol = Chem.MolFromSmiles(smiles)
  //   params = AllChem.ETKDGv3()
  //   params.randomSeed = <seed>
  //   AllChem.EmbedMultipleConfs(mol, <numConfs>, params)
  //   tfd = TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev='equal', symmRadius=2)

  struct TestCase {
    const char*         smiles;
    int                 numConfs;
    int                 seed;
    std::vector<double> reference;
  };

  // clang-format off
  std::vector<TestCase> cases = {
    {"CCCC", 4, 42, {              // n-butane, 1 torsion
      0.6667389132, 0.0000726610, 0.6666662521,
      0.6667387931, 0.0000001200, 0.6666661321
    }},
    {"CCCCC", 4, 42, {             // n-pentane, 2 torsions
      0.6060606631, 0.6060573299, 0.6060662252,
      0.6666872992, 0.6666326206, 0.0606323184
    }},
    {"CCCCCC", 4, 42, {            // n-hexane, 3 torsions
      0.6111276139, 0.0555704226, 0.6666744357,
      0.5555532106, 0.6111014381, 0.6111123144
    }},
    {"CC(C)CC", 4, 42, {           // isopentane, 1 symmetric torsion (2 quartets)
      0.0000045777, 0.0433253929, 0.0433299706,
      0.0000027031, 0.0000072808, 0.0433226898
    }},
    {"C1CCCCC1", 4, 42, {          // cyclohexane, 1 ring torsion (6 quartets)
      0.1343662730, 0.3653621455, 0.2309958724,
      0.1017138867, 0.2360801597, 0.4670760321
    }},
    {"c1ccccc1CC", 4, 42, {        // ethylbenzene, 1 symmetric non-ring + 1 ring torsion
      0.0000114982, 0.0000104863, 0.0000020789,
      0.0000100311, 0.0000018047, 0.0000013199
    }},
    {"CC(=O)[C@H]1CCCC[C@@H]1CN", 5, 44, {  // cyclohexane + ketone/amine substituents
      0.3538817018, 0.5908750389, 0.2992543709, 0.5450659700, 0.6307899400,
      0.3315705683, 0.5590067080, 0.3311160076, 0.2999948940, 0.5678678322
    }},
    {"C[C@@H](CN(C)C(=O)C#CCN)C1CC1", 5, 45, {  // cyclopropane + triple bond
      0.6173216954, 0.2578548043, 0.8622476874, 0.4208182281, 0.7014846040,
      0.1738311915, 0.1683341272, 0.4489912639, 0.4239728997, 0.2524937077
    }},
  };
  // clang-format on

  nvMolKit::TFDComputeOptions options;
  options.useWeights          = true;
  options.maxDevMode          = nvMolKit::TFDMaxDevMode::Equal;
  options.symmRadius          = 2;
  options.ignoreColinearBonds = true;

  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.smiles);

    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(tc.smiles));
    ASSERT_NE(mol, nullptr);

    generateConformers(*mol, tc.numConfs, tc.seed);
    auto tfdMatrix = generator_.GetTFDMatrix(*mol, options);

    ASSERT_EQ(tfdMatrix.size(), tc.reference.size());
    constexpr double kRDKitTolerance = 5e-4;
    for (size_t i = 0; i < tfdMatrix.size(); ++i) {
      EXPECT_NEAR(tfdMatrix[i], tc.reference[i], kRDKitTolerance) << "TFD[" << i << "] mismatch with RDKit reference";
    }
  }
}

TEST_F(TFDCpuTest, CompareWithRDKitReferenceAddHs) {
  // Compare TFD with explicit hydrogens (AddHs) against RDKit reference values.
  // AddHs produces more quartets per torsion, exercising multi-quartet handling
  // on molecules that would otherwise be single-quartet.
  //
  // Reference values generated with RDKit Python:
  //   mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
  //   params = AllChem.ETKDGv3()
  //   params.randomSeed = 42
  //   AllChem.EmbedMultipleConfs(mol, 4, params)
  //   tfd = TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev='equal', symmRadius=2)

  struct TestCase {
    const char*         smiles;
    std::vector<double> reference;
  };

  // clang-format off
  std::vector<TestCase> cases = {
    {"CCCCC", {                   // n-pentane with AddHs (symmetric torsions from H)
      0.6666346588,
      0.0606342183,
      0.6666024736,
      0.6666717502,
      0.6666935910,
      0.6061117436
    }},
    {"CC(C)CC", {                 // isopentane with AddHs
      0.0155798214,
      0.0118375755,
      0.0180868258,
      0.0000140721,
      0.0091266798,
      0.0336807192
    }},
  };
  // clang-format on

  nvMolKit::TFDComputeOptions options;
  options.useWeights          = true;
  options.maxDevMode          = nvMolKit::TFDMaxDevMode::Equal;
  options.symmRadius          = 2;
  options.ignoreColinearBonds = true;

  for (const auto& tc : cases) {
    SCOPED_TRACE(std::string(tc.smiles) + " (AddHs)");

    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(tc.smiles));
    ASSERT_NE(mol, nullptr);

    RDKit::MolOps::addHs(*mol);

    generateConformers(*mol, 4, 42);
    ASSERT_EQ(mol->getNumConformers(), 4);

    auto tfdMatrix = generator_.GetTFDMatrix(*mol, options);

    ASSERT_EQ(tfdMatrix.size(), tc.reference.size());
    constexpr double kRDKitTolerance = 5e-4;
    for (size_t i = 0; i < tfdMatrix.size(); ++i) {
      EXPECT_NEAR(tfdMatrix[i], tc.reference[i], kRDKitTolerance) << "TFD[" << i << "] mismatch with RDKit reference";
    }
  }
}

// =============================================================================
// GetTFDMatrices (batch)
// =============================================================================

TEST_F(TFDCpuTest, BatchProcessing) {
  // Test batch processing of multiple molecules via GetTFDMatrices
  // clang-format off
  const std::vector<std::string> testSmiles = {
    "CCCC", "CC(C)C", "c1ccccc1", "CCO", "CCCCC",
    "CC(=O)O", "c1ccc(cc1)O", "CCCCCC", "CC(C)(C)C", "c1ccc2ccccc2c1",
  };
  // clang-format on

  std::vector<std::unique_ptr<RDKit::RWMol>> mols;
  std::vector<const RDKit::ROMol*>           molPtrs;

  for (const auto& smiles : testSmiles) {
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(smiles));
    if (mol) {
      generateConformers(*mol, 3);
      if (mol->getNumConformers() >= 2) {
        molPtrs.push_back(mol.get());
        mols.push_back(std::move(mol));
      }
    }
  }

  ASSERT_GE(mols.size(), 3u);

  nvMolKit::TFDComputeOptions options;
  auto                        results = generator_.GetTFDMatrices(molPtrs, options);

  ASSERT_EQ(results.size(), molPtrs.size());

  for (size_t i = 0; i < results.size(); ++i) {
    int numConf       = molPtrs[i]->getNumConformers();
    int expectedPairs = numConf * (numConf - 1) / 2;

    EXPECT_EQ(results[i].size(), static_cast<size_t>(expectedPairs)) << "Mismatch for molecule " << i;

    for (double tfd : results[i]) {
      EXPECT_TRUE(std::isfinite(tfd));
      EXPECT_GE(tfd, 0.0);
    }
  }
}
