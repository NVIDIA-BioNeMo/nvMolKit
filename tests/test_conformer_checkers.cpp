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

#include <GraphMol/Conformer.h>
#include <GraphMol/ROMol.h>

#include <memory>

#include "gtest/gtest.h"
#include "src/testutils/conformer_checkers.h"

class ConformerCheckersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test molecules with different numbers of conformers
    mol1 = std::make_unique<RDKit::ROMol>();
    mol2 = std::make_unique<RDKit::ROMol>();
    mol3 = std::make_unique<RDKit::ROMol>();
    mol4 = std::make_unique<RDKit::ROMol>();

    // Add conformers to molecules
    // mol1: 3 conformers
    for (int i = 0; i < 3; ++i) {
      auto conf = std::make_unique<RDKit::Conformer>();
      mol1->addConformer(conf.release());
    }

    // mol2: 2 conformers
    for (int i = 0; i < 2; ++i) {
      auto conf = std::make_unique<RDKit::Conformer>();
      mol2->addConformer(conf.release());
    }

    // mol3: 1 conformer
    auto conf = std::make_unique<RDKit::Conformer>();
    mol3->addConformer(conf.release());

    // mol4: 0 conformers (empty)

    // Create vectors for different test scenarios
    allMols        = {mol1.get(), mol2.get(), mol3.get(), mol4.get()};
    completeMols   = {mol1.get(), mol1.get(), mol1.get()};  // All have 3 conformers
    incompleteMols = {mol2.get(), mol3.get(), mol4.get()};  // 2, 1, 0 conformers
  }

  std::unique_ptr<RDKit::ROMol>    mol1, mol2, mol3, mol4;
  std::vector<const RDKit::ROMol*> allMols, completeMols, incompleteMols;
};

TEST_F(ConformerCheckersTest, ThrowsExceptionWhenNoToleranceProvided) {
  EXPECT_THROW(nvMolKit::checkForCompletedConformers(allMols, 3), std::invalid_argument);
}

TEST_F(ConformerCheckersTest, PassesWhenAllMoleculesHaveExpectedConformers) {
  // All molecules have 3 conformers, expecting 3
  EXPECT_TRUE(nvMolKit::checkForCompletedConformers(completeMols, 3, std::nullopt, 0));

  // Using total tolerance
  EXPECT_TRUE(nvMolKit::checkForCompletedConformers(completeMols, 3, 0, std::nullopt));

  // Using both tolerances
  EXPECT_TRUE(nvMolKit::checkForCompletedConformers(completeMols, 3, 0, 0));
}

TEST_F(ConformerCheckersTest, FailsWhenMoleculesHaveFewerConformers) {
  // mol2 has 2, mol3 has 1, mol4 has 0, expecting 3
  // Total failures: 1 + 2 + 3 = 6

  // Fails with strict total tolerance
  EXPECT_FALSE(nvMolKit::checkForCompletedConformers(incompleteMols, 3, 0, std::nullopt));

  // Fails with strict per-molecule tolerance
  EXPECT_FALSE(nvMolKit::checkForCompletedConformers(incompleteMols, 3, std::nullopt, 2));

  // Fails with both strict tolerances
  EXPECT_FALSE(nvMolKit::checkForCompletedConformers(incompleteMols, 3, 0, 2));
}

TEST_F(ConformerCheckersTest, PassesWithSufficientTotalTolerance) {
  // Total failures: 1 + 2 + 3 = 6
  EXPECT_TRUE(nvMolKit::checkForCompletedConformers(incompleteMols, 3, 6, std::nullopt));

  EXPECT_TRUE(nvMolKit::checkForCompletedConformers(incompleteMols, 3, 10, std::nullopt));

  EXPECT_FALSE(nvMolKit::checkForCompletedConformers(incompleteMols, 3, 5, std::nullopt));
}

TEST_F(ConformerCheckersTest, PassesWithSufficientPerMoleculeTolerance) {
  // mol2 has 2 conformers (1 failure), mol3 has 1 (2 failures), mol4 has 0 (3 failures)

  // Should pass if per-molecule tolerance allows for the worst case (3 failures)
  EXPECT_TRUE(nvMolKit::checkForCompletedConformers(incompleteMols, 3, std::nullopt, 3));

  // Should fail if per-molecule tolerance is too strict
  EXPECT_FALSE(nvMolKit::checkForCompletedConformers(incompleteMols, 3, std::nullopt, 2));
}

TEST_F(ConformerCheckersTest, AcceptEitherMetricAsPassWorksCorrectly) {
  // Scenario: total fails but per-molecule passes
  std::vector<const RDKit::ROMol*> mols = {mol2.get(), mol2.get()};  // Both have 2 conformers
  // Total failures: 2, per-molecule failures: 1 each

  // Should fail when acceptEitherMetricAsPass is false
  EXPECT_FALSE(nvMolKit::checkForCompletedConformers(mols,
                                                     3,
                                                     1,
                                                     2,
                                                     false));  // total tolerance = 1 (fail), per-molecule = 2 (pass)

  // Should pass when acceptEitherMetricAsPass is true
  EXPECT_TRUE(
    nvMolKit::checkForCompletedConformers(mols, 3, 1, 2, true));  // total tolerance = 1 (fail), per-molecule = 2 (pass)
}

TEST_F(ConformerCheckersTest, HandlesEmptyMoleculeVector) {
  std::vector<const RDKit::ROMol*> emptyMols;

  EXPECT_TRUE(nvMolKit::checkForCompletedConformers(emptyMols, 3, 0, std::nullopt));

  EXPECT_TRUE(nvMolKit::checkForCompletedConformers(emptyMols, 3, std::nullopt, 0));
}

TEST_F(ConformerCheckersTest, HandlesZeroExpectedConformers) {
  // When expecting 0 conformers, all should pass
  EXPECT_TRUE(nvMolKit::checkForCompletedConformers(allMols, 0, 0, std::nullopt));

  EXPECT_TRUE(nvMolKit::checkForCompletedConformers(allMols, 0, std::nullopt, 0));
}

TEST_F(ConformerCheckersTest, HandlesExcessConformers) {
  // Molecules with more conformers than expected should pass
  EXPECT_TRUE(nvMolKit::checkForCompletedConformers(completeMols, 2, 0, std::nullopt));  // Expecting 2, have 3

  EXPECT_TRUE(nvMolKit::checkForCompletedConformers(completeMols, 1, std::nullopt, 0));  // Expecting 1, have 3
}

TEST_F(ConformerCheckersTest, MixedScenario) {
  // Mixed scenario: mol1 (3 confs), mol2 (2 confs), mol3 (1 conf), expecting 2
  std::vector<const RDKit::ROMol*> mixedMols = {mol1.get(), mol2.get(), mol3.get()};

  // mol1: 3 >= 2 (pass), mol2: 2 >= 2 (pass), mol3: 1 < 2 (1 failure)
  // Total failures: 1

  EXPECT_TRUE(
    nvMolKit::checkForCompletedConformers(mixedMols, 2, 1, std::nullopt));  // Total tolerance allows 1 failure

  EXPECT_FALSE(
    nvMolKit::checkForCompletedConformers(mixedMols, 2, 0, std::nullopt));  // Total tolerance doesn't allow failures

  EXPECT_TRUE(
    nvMolKit::checkForCompletedConformers(mixedMols, 2, std::nullopt, 1));  // Per-molecule tolerance allows 1 failure
}

TEST_F(ConformerCheckersTest, PrintResultsDoesNotAffectOutcome) {
  // Test that printResults parameter doesn't affect the return value
  bool result1 = nvMolKit::checkForCompletedConformers(incompleteMols, 3, 5, std::nullopt, false, false);

  bool result2 = nvMolKit::checkForCompletedConformers(incompleteMols, 3, 5, std::nullopt, false, true);

  EXPECT_EQ(result1, result2);
}
