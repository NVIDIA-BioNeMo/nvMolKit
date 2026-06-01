// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <ForceField/ForceField.h>
#include <gmock/gmock.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <gtest/gtest.h>

#include <filesystem>

#include "rdkit_extensions/dist_geom_flattened_builder.h"
#include "rdkit_extensions/mmff_flattened_builder.h"
#include "src/embedder_utils.h"
#include "tests/test_utils.h"

TEST(FlattenedBuilderTest, NullMolecule) {
  RDKit::ROMol mol;
  mol.addConformer(new RDKit::Conformer(), true);
  auto ffParams = nvMolKit::MMFF::constructForcefieldContribs(mol);
  EXPECT_EQ(ffParams.bondTerms.idx1.size(), 0);
  EXPECT_EQ(ffParams.angleTerms.idx1.size(), 0);
  EXPECT_EQ(ffParams.bendTerms.idx1.size(), 0);
  EXPECT_EQ(ffParams.oopTerms.idx1.size(), 0);
  EXPECT_EQ(ffParams.torsionTerms.idx1.size(), 0);
  EXPECT_EQ(ffParams.vdwTerms.idx1.size(), 0);
  EXPECT_EQ(ffParams.eleTerms.idx1.size(), 0);
}

class FlattenedBuilderTestFixture : public ::testing::Test {
 public:
  FlattenedBuilderTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }
  void SetUp() override {
    // Check for test_data/rdkit_smallmol_1.mol2
    const std::string mol2FilePath = testDataFolderPath_ + "/rdkit_smallmol_1.mol2";
    ASSERT_TRUE(std::filesystem::exists(mol2FilePath)) << "Could not find " << mol2FilePath;
    mol_ = std::unique_ptr<RDKit::RWMol>(RDKit::MolFileToMol(mol2FilePath, false));
    ASSERT_NE(mol_, nullptr);
    RDKit::MolOps::sanitizeMol(*mol_);
  }

 protected:
  std::string                   testDataFolderPath_;
  std::unique_ptr<RDKit::RWMol> mol_;
};

TEST_F(FlattenedBuilderTestFixture, TestMMFFlattenedBuilderMol1) {
  auto ffParams = nvMolKit::MMFF::constructForcefieldContribs(*mol_);

  // Small molecule 1 - known good constant values, taken from RDKit
  constexpr std::array<double, 8> bondForceConstants = {9.505, 5.170, 5.170, 4.539, 5.170, 4.766, 4.766, 4.766};
  constexpr std::array<double, 8> bondIdealLengths   = {1.333, 1.083, 1.083, 1.482, 1.083, 1.093, 1.093, 1.093};

  // Do exhaustive checks for bond terms.
  EXPECT_THAT(ffParams.bondTerms.kb, testing::ElementsAreArray(bondForceConstants));
  EXPECT_THAT(ffParams.bondTerms.r0, testing::ElementsAreArray(bondIdealLengths));

  // Do size checks and spot checks for each other term:

  // Angles
  const int wantNumAngleTerms = 12;
  EXPECT_EQ(ffParams.angleTerms.ka.size(), wantNumAngleTerms);
  EXPECT_EQ(ffParams.angleTerms.theta0.size(), wantNumAngleTerms);
  EXPECT_EQ(ffParams.angleTerms.idx1.size(), wantNumAngleTerms);
  EXPECT_EQ(ffParams.angleTerms.idx2.size(), wantNumAngleTerms);
  EXPECT_EQ(ffParams.angleTerms.idx3.size(), wantNumAngleTerms);
  EXPECT_EQ(ffParams.angleTerms.isLinear.size(), wantNumAngleTerms);
  EXPECT_EQ(bool(ffParams.angleTerms.isLinear[10]), false);
  EXPECT_EQ(ffParams.angleTerms.idx1[10], 6);
  EXPECT_EQ(ffParams.angleTerms.idx2[10], 2);
  EXPECT_EQ(ffParams.angleTerms.idx3[10], 8);
  EXPECT_EQ(ffParams.angleTerms.ka[10], 0.516);
  EXPECT_EQ(ffParams.angleTerms.theta0[10], 108.836);

  // Bend-stretch
  const int wantNumBendStretchTerms = 12;
  EXPECT_EQ(ffParams.bendTerms.theta0.size(), wantNumBendStretchTerms);
  EXPECT_EQ(ffParams.bendTerms.idx1.size(), wantNumBendStretchTerms);
  EXPECT_EQ(ffParams.bendTerms.idx2.size(), wantNumBendStretchTerms);
  EXPECT_EQ(ffParams.bendTerms.idx3.size(), wantNumBendStretchTerms);
  EXPECT_EQ(ffParams.bendTerms.restLen1.size(), wantNumBendStretchTerms);
  EXPECT_EQ(ffParams.bendTerms.restLen2.size(), wantNumBendStretchTerms);
  EXPECT_EQ(ffParams.bendTerms.forceConst1.size(), wantNumBendStretchTerms);
  EXPECT_EQ(ffParams.bendTerms.forceConst2.size(), wantNumBendStretchTerms);

  EXPECT_EQ(ffParams.bendTerms.theta0[1], 121.004);
  EXPECT_EQ(ffParams.bendTerms.idx1[1], 1);
  EXPECT_EQ(ffParams.bendTerms.idx2[1], 0);
  EXPECT_EQ(ffParams.bendTerms.idx3[1], 4);
  EXPECT_EQ(ffParams.bendTerms.restLen1[1], 1.333);
  EXPECT_EQ(ffParams.bendTerms.restLen2[1], 1.083);
  EXPECT_EQ(ffParams.bendTerms.forceConst1[1], 0.207);
  EXPECT_EQ(ffParams.bendTerms.forceConst2[1], 0.157);

  // OOP
  const int wantNumOopTerms = 6;
  EXPECT_EQ(ffParams.oopTerms.koop.size(), wantNumOopTerms);
  EXPECT_EQ(ffParams.oopTerms.idx1.size(), wantNumOopTerms);
  EXPECT_EQ(ffParams.oopTerms.idx2.size(), wantNumOopTerms);
  EXPECT_EQ(ffParams.oopTerms.idx3.size(), wantNumOopTerms);
  EXPECT_EQ(ffParams.oopTerms.idx4.size(), wantNumOopTerms);

  EXPECT_EQ(ffParams.oopTerms.idx1[3], 0);
  EXPECT_EQ(ffParams.oopTerms.idx2[3], 1);
  EXPECT_EQ(ffParams.oopTerms.idx3[3], 2);
  EXPECT_EQ(ffParams.oopTerms.idx4[3], 5);
  EXPECT_EQ(ffParams.oopTerms.koop[3], 0.013);

  // Torsions
  const int wantNumTorsionTerms = 10;
  EXPECT_EQ(ffParams.torsionTerms.idx1.size(), wantNumTorsionTerms);
  EXPECT_EQ(ffParams.torsionTerms.idx2.size(), wantNumTorsionTerms);
  EXPECT_EQ(ffParams.torsionTerms.idx3.size(), wantNumTorsionTerms);
  EXPECT_EQ(ffParams.torsionTerms.idx4.size(), wantNumTorsionTerms);
  EXPECT_EQ(ffParams.torsionTerms.V1.size(), wantNumTorsionTerms);
  EXPECT_EQ(ffParams.torsionTerms.V2.size(), wantNumTorsionTerms);
  EXPECT_EQ(ffParams.torsionTerms.V3.size(), wantNumTorsionTerms);

  EXPECT_EQ(ffParams.torsionTerms.idx1[0], 3);
  EXPECT_EQ(ffParams.torsionTerms.idx2[0], 0);
  EXPECT_EQ(ffParams.torsionTerms.idx3[0], 1);
  EXPECT_EQ(ffParams.torsionTerms.idx4[0], 2);
  EXPECT_EQ(ffParams.torsionTerms.V1[0], 0.0);
  EXPECT_EQ(ffParams.torsionTerms.V2[0], 12.0);
  EXPECT_EQ(ffParams.torsionTerms.V3[0], 0.0);

  // VDW
  constexpr int wantNumVdwTerms = 16;
  EXPECT_EQ(ffParams.vdwTerms.idx1.size(), wantNumVdwTerms);
  EXPECT_EQ(ffParams.vdwTerms.idx2.size(), wantNumVdwTerms);
  EXPECT_EQ(ffParams.vdwTerms.wellDepth.size(), wantNumVdwTerms);
  EXPECT_EQ(ffParams.vdwTerms.R_ij_star.size(), wantNumVdwTerms);

  EXPECT_EQ(ffParams.vdwTerms.idx1[3], 2);
  EXPECT_EQ(ffParams.vdwTerms.idx2[3], 3);
  EXPECT_THAT(ffParams.vdwTerms.wellDepth[3], testing::DoubleNear(0.0280777, 1e-5));
  EXPECT_THAT(ffParams.vdwTerms.R_ij_star[3], testing::DoubleNear(3.59879, 1e-5));

  // electrostatics
  constexpr int wantNumEleTerms = 4;
  EXPECT_EQ(ffParams.eleTerms.idx1.size(), wantNumEleTerms);
  EXPECT_EQ(ffParams.eleTerms.idx2.size(), wantNumEleTerms);
  EXPECT_EQ(ffParams.eleTerms.chargeTerm.size(), wantNumEleTerms);
  EXPECT_EQ(ffParams.eleTerms.dielModel.size(), wantNumEleTerms);
  EXPECT_EQ(ffParams.eleTerms.is1_4.size(), wantNumEleTerms);

  EXPECT_EQ(ffParams.eleTerms.idx1[2], 3);
  EXPECT_EQ(ffParams.eleTerms.idx2[2], 5);
}

TEST_F(FlattenedBuilderTestFixture, TestETKDGFlattenedBuilderMol1) {
  unsigned int nAtoms = mol_->getNumAtoms();

  auto                                        options = RDKit::DGeomHelpers::ETKDGv3;
  nvMolKit::detail::EmbedArgs                 eargs;
  std::vector<std::unique_ptr<RDGeom::Point>> positions;
  std::unique_ptr<ForceFields::ForceField>    field =
    nvMolKit::DGeomHelpers::generateRDKitFF(*mol_,
                                            options,
                                            eargs,
                                            positions,
                                            nvMolKit::DGeomHelpers::Dimensionality::DIM_4D);

  auto ffParams = nvMolKit::DistGeom::constructForceFieldContribs(eargs.dim, *eargs.mmat, eargs.chiralCenters);

  // DistViolation
  const int wantNumDistViolationTerms = 36;
  EXPECT_EQ(ffParams.distTerms.idx1.size(), wantNumDistViolationTerms);
  EXPECT_EQ(ffParams.distTerms.idx2.size(), wantNumDistViolationTerms);
  EXPECT_EQ(ffParams.distTerms.lb2.size(), wantNumDistViolationTerms);
  EXPECT_EQ(ffParams.distTerms.ub2.size(), wantNumDistViolationTerms);
  EXPECT_EQ(ffParams.distTerms.weight.size(), wantNumDistViolationTerms);

  EXPECT_EQ(ffParams.distTerms.idx1[0], 1);
  EXPECT_EQ(ffParams.distTerms.idx2[0], 0);
  EXPECT_NEAR(ffParams.distTerms.lb2[0], 1.73932, 1e-5);
  EXPECT_NEAR(ffParams.distTerms.ub2[0], 1.79247, 1e-5);
  EXPECT_EQ(ffParams.distTerms.weight[0], 1.0);

  // ChiralViolation
  const int wantNumChiralViolationTerms = 1;
  EXPECT_EQ(ffParams.chiralTerms.idx1.size(), wantNumChiralViolationTerms);
  EXPECT_EQ(ffParams.chiralTerms.idx2.size(), wantNumChiralViolationTerms);
  EXPECT_EQ(ffParams.chiralTerms.idx3.size(), wantNumChiralViolationTerms);
  EXPECT_EQ(ffParams.chiralTerms.idx4.size(), wantNumChiralViolationTerms);
  EXPECT_EQ(ffParams.chiralTerms.volLower.size(), wantNumChiralViolationTerms);
  EXPECT_EQ(ffParams.chiralTerms.volUpper.size(), wantNumChiralViolationTerms);

  EXPECT_EQ(ffParams.chiralTerms.idx1[0], 1);
  EXPECT_EQ(ffParams.chiralTerms.idx2[0], 6);
  EXPECT_EQ(ffParams.chiralTerms.idx3[0], 7);
  EXPECT_EQ(ffParams.chiralTerms.idx4[0], 8);
  EXPECT_EQ(ffParams.chiralTerms.volLower[0], 5);
  EXPECT_EQ(ffParams.chiralTerms.volUpper[0], 100);

  // FourthDim
  const int wantNumFourthDimTerms = 9;
  EXPECT_EQ(ffParams.fourthTerms.idx.size(), wantNumFourthDimTerms);

  EXPECT_EQ(ffParams.fourthTerms.idx[0], 0);
}
