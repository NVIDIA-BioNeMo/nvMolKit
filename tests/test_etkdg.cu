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

#include <gmock/gmock.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/FileParsers/MolWriters.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <unordered_map>

#include "rdkit_extensions/conformer_pruning.h"
#include "src/embedder_utils.h"
#include "src/etkdg.h"
#include "src/etkdg_impl.h"
#include "src/etkdg_stage_coordgen.h"
#include "src/etkdg_stage_update_conformers.h"
#include "src/forcefields/dist_geom.h"
#include "src/minimizer/bfgs_minimize.h"
#include "src/utils/device.h"
#include "src/utils/host_vector.h"
#include "tests/test_utils.h"

using ::nvMolKit::detail::ETKDGContext;
using ::nvMolKit::detail::ETKDGDriver;
using ::nvMolKit::detail::ETKDGStage;

class ProgrammableStep : public ETKDGStage {
 public:
  void execute(ETKDGContext& ctx) override {
    ASSERT_LT(iteration, failedPerIteration.size());
    ctx.failedThisStage.copyFromHost(failedPerIteration[iteration]);
    iteration++;
  }
  std::string                       name() const override { return "Programmable Test Stage"; }
  int                               iteration = 0;
  std::vector<std::vector<uint8_t>> failedPerIteration;
};

TEST(ETKDGDriverFailTest, NoConformers) {
  auto                                     context = std::make_unique<ETKDGContext>();
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  stages.push_back(std::make_unique<ProgrammableStep>());
  EXPECT_THROW(ETKDGDriver(std::move(context), std::move(stages)), std::runtime_error);
}

TEST(ETKDGDriverFailTest, NoStages) {
  auto context           = std::make_unique<ETKDGContext>();
  context->nTotalSystems = 4;
  context->failedThisStage.resize(context->nTotalSystems);
  context->activeThisStage.resize(context->nTotalSystems);
  context->finishedOnIteration.resize(context->nTotalSystems);
  context->countFinishedThisIteration.set(0);
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  EXPECT_THROW(ETKDGDriver(std::move(context), std::move(stages)), std::runtime_error);
}

TEST(ETKDGDriverEdgeCaseTest, SingleConformer) {
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  auto                                     stage = std::make_unique<ProgrammableStep>();
  stage->failedPerIteration                      = {{1}, {0}, {0}};
  auto stage2                                    = std::make_unique<ProgrammableStep>();
  stage2->failedPerIteration                     = {{0}, {1}, {0}};
  stages.push_back(std::move(stage));
  stages.push_back(std::move(stage2));

  auto context           = std::make_unique<ETKDGContext>();
  context->nTotalSystems = 1;
  context->failedThisStage.resize(context->nTotalSystems);
  context->activeThisStage.resize(context->nTotalSystems);
  std::vector<int16_t> copyFrom(context->nTotalSystems, -1);
  context->finishedOnIteration.resize(context->nTotalSystems);
  context->finishedOnIteration.copyFromHost(copyFrom);
  context->countFinishedThisIteration.set(0);
  for (size_t i = 0; i < stages.size(); i++) {
    context->totalFailures.emplace_back(context->nTotalSystems);
    context->totalFailures.back().zero();
  }

  ETKDGDriver driver(std::move(context), std::move(stages));
  driver.run(5);
  EXPECT_EQ(driver.numConfsFinished(), 1);
  EXPECT_EQ(driver.iterationsComplete(), 3);

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 2);
  EXPECT_THAT(failureCounts[0], testing::ElementsAre(1));
  EXPECT_THAT(failureCounts[1], testing::ElementsAre(1));
}

class ETKDGDriverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_                = std::make_unique<ETKDGContext>();
    context_->nTotalSystems = 4;
    context_->failedThisStage.resize(context_->nTotalSystems);
    context_->activeThisStage.resize(context_->nTotalSystems);
    std::vector<int16_t> copyFrom(context_->nTotalSystems, -1);
    context_->finishedOnIteration.resize(context_->nTotalSystems);
    context_->finishedOnIteration.copyFromHost(copyFrom);
    context_->countFinishedThisIteration.set(0);
  }

  std::unique_ptr<ETKDGContext> context_;
};

TEST_F(ETKDGDriverTest, SingleStageAllPassFirstIteration) {
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  auto                                     stage = std::make_unique<ProgrammableStep>();
  stage->failedPerIteration                      = {
    {
     0, 0,
     0, 0,
     }
  };

  stages.push_back(std::move(stage));
  for (size_t i = 0; i < stages.size(); i++) {
    context_->totalFailures.emplace_back(context_->nTotalSystems);
    context_->totalFailures.back().zero();
  }

  ETKDGDriver driver(std::move(context_), std::move(stages));
  driver.run(5);

  EXPECT_EQ(driver.numConfsFinished(), 4);
  EXPECT_EQ(driver.iterationsComplete(), 1);

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 1);
  EXPECT_THAT(failureCounts[0], testing::ElementsAre(0, 0, 0, 0));
  auto completed = driver.completedConformers();
  EXPECT_THAT(completed, testing::ElementsAre(1, 1, 1, 1));
}

TEST_F(ETKDGDriverTest, SingleStageAllPassSecondIteration) {
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  auto                                     stage = std::make_unique<ProgrammableStep>();
  stage->failedPerIteration                      = {
    {1, 1, 1, 1},
    {0, 0, 0, 0}
  };

  stages.push_back(std::move(stage));
  for (size_t i = 0; i < stages.size(); i++) {
    context_->totalFailures.emplace_back(context_->nTotalSystems);
    context_->totalFailures.back().zero();
  }

  ETKDGDriver driver(std::move(context_), std::move(stages));
  driver.run(5);

  EXPECT_EQ(driver.numConfsFinished(), 4);
  EXPECT_EQ(driver.iterationsComplete(), 2);

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 1);
  EXPECT_THAT(failureCounts[0], testing::ElementsAre(1, 1, 1, 1));
  auto completed = driver.completedConformers();
  EXPECT_THAT(completed, testing::ElementsAre(1, 1, 1, 1));
}

TEST_F(ETKDGDriverTest, SingleStageVariablePass) {
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  auto                                     stage = std::make_unique<ProgrammableStep>();
  stage->failedPerIteration                      = {
    {0, 1, 1, 1},
    {0, 0, 1, 1},
    {0, 0, 0, 1},
    {0, 0, 0, 0},
  };

  stages.push_back(std::move(stage));
  for (size_t i = 0; i < stages.size(); i++) {
    context_->totalFailures.emplace_back(context_->nTotalSystems);
    context_->totalFailures.back().zero();
  }

  ETKDGDriver driver(std::move(context_), std::move(stages));
  driver.run(5);

  EXPECT_EQ(driver.numConfsFinished(), 4);
  EXPECT_EQ(driver.iterationsComplete(), 4);

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 1);
  EXPECT_THAT(failureCounts[0], testing::ElementsAre(0, 1, 2, 3));
  auto completed = driver.completedConformers();
  EXPECT_THAT(completed, testing::ElementsAre(1, 1, 1, 1));
}

TEST_F(ETKDGDriverTest, SingleStageSomeNotPassed) {
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  auto                                     stage = std::make_unique<ProgrammableStep>();
  stage->failedPerIteration                      = {
    {0, 1, 1, 1},
    {0, 0, 1, 1},
    {0, 0, 1, 1},
    {0, 0, 1, 1},
    {0, 0, 1, 1}
  };

  stages.push_back(std::move(stage));
  for (size_t i = 0; i < stages.size(); i++) {
    context_->totalFailures.emplace_back(context_->nTotalSystems);
    context_->totalFailures.back().zero();
  }
  ETKDGDriver driver(std::move(context_), std::move(stages));
  driver.run(5);

  EXPECT_EQ(driver.numConfsFinished(), 2);
  EXPECT_EQ(driver.iterationsComplete(), 5);

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 1);
  EXPECT_THAT(failureCounts[0], testing::ElementsAre(0, 1, 5, 5));
}

TEST_F(ETKDGDriverTest, MultiStageAllFail) {
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  auto                                     stage = std::make_unique<ProgrammableStep>();
  stage->failedPerIteration                      = {
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0}
  };
  auto stage2                = std::make_unique<ProgrammableStep>();
  stage2->failedPerIteration = {
    {1, 1, 1, 1},
    {1, 1, 1, 1},
    {1, 1, 1, 1},
    {1, 1, 1, 1},
    {1, 1, 1, 1}
  };
  auto stage3                = std::make_unique<ProgrammableStep>();
  stage3->failedPerIteration = {
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0}
  };

  stages.push_back(std::move(stage));
  stages.push_back(std::move(stage2));
  stages.push_back(std::move(stage3));
  for (size_t i = 0; i < stages.size(); i++) {
    context_->totalFailures.emplace_back(context_->nTotalSystems);
    context_->totalFailures.back().zero();
  }
  ETKDGDriver driver(std::move(context_), std::move(stages));
  driver.run(5);

  EXPECT_EQ(driver.numConfsFinished(), 0);
  EXPECT_EQ(driver.iterationsComplete(), 5);

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 3);
  EXPECT_THAT(failureCounts[0], testing::ElementsAre(0, 0, 0, 0));
  EXPECT_THAT(failureCounts[1], testing::ElementsAre(5, 5, 5, 5));
  EXPECT_THAT(failureCounts[2], testing::ElementsAre(0, 0, 0, 0));

  auto completed = driver.completedConformers();
  EXPECT_THAT(completed, testing::ElementsAre(0, 0, 0, 0));
}

TEST_F(ETKDGDriverTest, MultiStageMixed) {
  // System
  // 0 - Fails on stage 1
  // 1 - passes first iteration
  // 2 - passes after a few fails
  // 3 - fails on different stages.
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  auto                                     stage = std::make_unique<ProgrammableStep>();
  stage->failedPerIteration                      = {
    {0, 0, 0, 1},
    {0, 0, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 0}
  };
  auto stage2                = std::make_unique<ProgrammableStep>();
  stage2->failedPerIteration = {
    {1, 0, 1, 0},
    {1, 0, 1, 1},
    {1, 0, 0, 0},
    {1, 0, 0, 1},
    {1, 0, 0, 0}
  };
  auto stage3                = std::make_unique<ProgrammableStep>();
  stage3->failedPerIteration = {
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 1},
    {0, 0, 0, 0},
    {0, 0, 0, 1}
  };

  stages.push_back(std::move(stage));
  stages.push_back(std::move(stage2));
  stages.push_back(std::move(stage3));
  for (size_t i = 0; i < stages.size(); i++) {
    context_->totalFailures.emplace_back(context_->nTotalSystems);
    context_->totalFailures.back().zero();
  }
  ETKDGDriver driver(std::move(context_), std::move(stages));
  driver.run(5);

  EXPECT_EQ(driver.numConfsFinished(), 2);
  EXPECT_EQ(driver.iterationsComplete(), 5);

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 3);
  EXPECT_THAT(failureCounts[0], testing::ElementsAre(0, 0, 2, 1));
  EXPECT_THAT(failureCounts[1], testing::ElementsAre(5, 0, 2, 2));
  EXPECT_THAT(failureCounts[2], testing::ElementsAre(0, 0, 0, 2));

  auto completed = driver.completedConformers();
  EXPECT_THAT(completed, testing::ElementsAre(0, 1, 1, 0));
}

class ETKDGPipelineUpdateConformersTestFixture : public ::testing::Test {
 public:
  ETKDGPipelineUpdateConformersTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

  void SetUp() override {
    // Load three molecules
    getMols(testDataFolderPath_ + "/MMFF94_dative.sdf", molsPtrs_, /*count=*/3);
    ASSERT_EQ(molsPtrs_.size(), 3) << "Expected to load 3 molecules";

    // Clear conformers and prepare for testing
    for (auto& molPtr : molsPtrs_) {
      molPtr->clearConformers();
      mols_.push_back(molPtr.get());
    }
    ASSERT_EQ(mols_.size(), 3) << "Expected 3 molecules";
  }

 protected:
  std::string                                testDataFolderPath_;
  std::vector<std::unique_ptr<RDKit::ROMol>> molsPtrs_;
  std::vector<RDKit::ROMol*>                 mols_;
};

TEST_F(ETKDGPipelineUpdateConformersTestFixture, UpdateConformersStage) {
  // Create eargs with dim=3 for all molecules
  auto                                     params = DGeomHelpers::ETKDGv3;
  std::vector<nvMolKit::detail::EmbedArgs> eargs;
  for (size_t i = 0; i < mols_.size(); ++i) {
    auto& earg = eargs.emplace_back();
    earg.dim   = 3;
  }

  // Create context and set activeThisStage to all 1s
  ETKDGContext context;
  context.nTotalSystems = mols_.size();
  context.activeThisStage.resize(context.nTotalSystems);
  std::vector<uint8_t> activeRef(context.nTotalSystems, 1);
  context.activeThisStage.copyFromHost(activeRef);

  // Initialize atomStarts in context
  context.systemHost.atomStarts = {0};
  int totalAtoms                = 0;
  for (const auto& mol : mols_) {
    totalAtoms += mol->getNumAtoms();
    context.systemHost.atomStarts.push_back(totalAtoms);
  }

  // Create reference positions with random values
  std::vector<double> refPositions(totalAtoms * 3);
  for (size_t i = 0; i < refPositions.size(); ++i) {
    refPositions[i] = static_cast<double>(rand()) / RAND_MAX;  // Random value between 0 and 1
  }
  context.systemDevice.positions.resize(refPositions.size());
  context.systemDevice.positions.copyFromHost(refPositions);

  // Create and execute the stage
  std::unordered_map<const RDKit::ROMol*, std::vector<std::unique_ptr<Conformer>>> conformers;

  // Create scratch buffers for the stage
  nvMolKit::PinnedHostVector<double>  positionsScratch(totalAtoms * 3);
  nvMolKit::PinnedHostVector<uint8_t> activeScratch(mols_.size());

  nvMolKit::detail::ETKDGUpdateConformersStage
    stage(mols_, eargs, conformers, positionsScratch, activeScratch, nullptr, nullptr, -1);
  stage.execute(context);
  for (size_t i = 0; i < mols_.size(); ++i) {
    auto it = conformers.find(mols_[i]);
    ASSERT_NE(it, conformers.end());
    nvmolkit::addConformersToMoleculeWithPruning(*mols_[i], it->second, params);
  }

  // Verify positions in conformers match reference positions
  for (size_t i = 0; i < mols_.size(); ++i) {
    const auto& mol = mols_[i];
    ASSERT_EQ(mol->getNumConformers(), 1) << "Molecule " << i << " should have one conformer";

    const auto& conf     = mol->getConformer();
    const int   startIdx = context.systemHost.atomStarts[i] * 3;

    for (unsigned int j = 0; j < mol->getNumAtoms(); ++j) {
      const int   posIdx = startIdx + j * 3;
      const auto& pos    = conf.getAtomPos(j);

      EXPECT_DOUBLE_EQ(pos.x, refPositions[posIdx]) << "Molecule " << i << " atom " << j << " x coordinate";
      EXPECT_DOUBLE_EQ(pos.y, refPositions[posIdx + 1]) << "Molecule " << i << " atom " << j << " y coordinate";
      EXPECT_DOUBLE_EQ(pos.z, refPositions[posIdx + 2]) << "Molecule " << i << " atom " << j << " z coordinate";
    }
  }
}

TEST_F(ETKDGPipelineUpdateConformersTestFixture, UpdateConformersStageWithInactiveMolecule) {
  // Create eargs with dim=3 for all molecules
  std::vector<nvMolKit::detail::EmbedArgs> eargs;
  for (size_t i = 0; i < mols_.size(); ++i) {
    auto& earg = eargs.emplace_back();
    earg.dim   = 3;
  }

  // Create context and set activeThisStage to all 1s except for the second molecule
  ETKDGContext context;
  context.nTotalSystems = mols_.size();
  context.activeThisStage.resize(context.nTotalSystems);
  std::vector<uint8_t> activeRef(context.nTotalSystems, 1);
  activeRef[1] = 0;  // Mark second molecule as inactive
  context.activeThisStage.copyFromHost(activeRef);

  // Initialize atomStarts in context
  context.systemHost.atomStarts = {0};
  int totalAtoms                = 0;
  for (const auto& mol : mols_) {
    totalAtoms += mol->getNumAtoms();
    context.systemHost.atomStarts.push_back(totalAtoms);
  }

  // Create reference positions with random values
  std::vector<double> refPositions(totalAtoms * 3);
  for (size_t i = 0; i < refPositions.size(); ++i) {
    refPositions[i] = static_cast<double>(rand()) / RAND_MAX;  // Random value between 0 and 1
  }
  context.systemDevice.positions.resize(refPositions.size());
  context.systemDevice.positions.copyFromHost(refPositions);

  // Create and execute the stage
  auto                                                                             params = DGeomHelpers::ETKDGv3;
  std::unordered_map<const RDKit::ROMol*, std::vector<std::unique_ptr<Conformer>>> conformers;

  // Create scratch buffers for the stage
  nvMolKit::PinnedHostVector<double>  positionsScratch(totalAtoms * 3);
  nvMolKit::PinnedHostVector<uint8_t> activeScratch(mols_.size());

  nvMolKit::detail::ETKDGUpdateConformersStage
    stage(mols_, eargs, conformers, positionsScratch, activeScratch, nullptr, nullptr, -1);
  stage.execute(context);
  for (size_t i = 0; i < mols_.size(); ++i) {
    if (i == 1) {
      continue;
    }
    auto it = conformers.find(mols_[i]);
    ASSERT_NE(it, conformers.end());
    nvmolkit::addConformersToMoleculeWithPruning(*mols_[i], it->second, params);
  }
  // Verify positions in conformers match reference positions
  for (size_t i = 0; i < mols_.size(); ++i) {
    const auto& mol = mols_[i];
    if (i == 1) {
      // Second molecule should not have a conformer
      EXPECT_EQ(mol->getNumConformers(), 0) << "Molecule " << i << " should not have a conformer (marked as inactive)";
      continue;
    }

    ASSERT_EQ(mol->getNumConformers(), 1) << "Molecule " << i << " should have one conformer";

    const auto& conf     = mol->getConformer();
    const int   startIdx = context.systemHost.atomStarts[i] * 3;

    for (unsigned int j = 0; j < mol->getNumAtoms(); ++j) {
      const int   posIdx = startIdx + j * 3;
      const auto& pos    = conf.getAtomPos(j);

      EXPECT_DOUBLE_EQ(pos.x, refPositions[posIdx]) << "Molecule " << i << " atom " << j << " x coordinate";
      EXPECT_DOUBLE_EQ(pos.y, refPositions[posIdx + 1]) << "Molecule " << i << " atom " << j << " y coordinate";
      EXPECT_DOUBLE_EQ(pos.z, refPositions[posIdx + 2]) << "Molecule " << i << " atom " << j << " z coordinate";
    }
  }
}

namespace {

// Helper function to test energy improvement for molecules
void testEnergyImprovement(const std::vector<RDKit::ROMol*>&    mols,
                           int                                  confsPerMolecule = 1,
                           RDKit::DGeomHelpers::EmbedParameters params           = RDKit::DGeomHelpers::ETKDGv3,
                           nvMolKit::BfgsBackend                backend = nvMolKit::BfgsBackend::PER_MOLECULE) {
  // Store initial energies for each molecule
  std::vector<double> initialEnergies;
  initialEnergies.reserve(mols.size());

  // Calculate initial energies for each molecule
  for (const auto& mol : mols) {
    ASSERT_NE(mol, nullptr) << "Molecule should not be null";

    // Setup initial force field and get reference energy
    nvMolKit::detail::EmbedArgs                 eargs;
    std::vector<std::unique_ptr<RDGeom::Point>> positions;
    std::unique_ptr<ForceFields::ForceField>    field;

    // Use RDKit's default first minimization force field for energy comparison
    auto option = RDKit::DGeomHelpers::ETKDGv3;
    nvMolKit::DGeomHelpers::setupRDKitFFWithPos(mol, option, field, eargs, positions);

    // Calculate and store initial energy
    const double initialEnergy = field->calcEnergy();
    ASSERT_GE(initialEnergy, 0.0) << "Initial energy should be positive";
    initialEnergies.push_back(initialEnergy);
  }

  // Run embedding for all molecules
  params.useRandomCoords = true;
  params.basinThresh     = 1e8;

  nvMolKit::BatchHardwareOptions hardwareOptions;
  hardwareOptions.preprocessingThreads = 10;
  hardwareOptions.batchSize            = 100;
  hardwareOptions.batchesPerGpu        = 10;

  nvMolKit::embedMolecules(mols, params, confsPerMolecule, -1, true, nullptr, hardwareOptions, backend);

  // Calculate and verify final energies for each molecule and conformer
  for (size_t i = 0; i < mols.size(); ++i) {
    auto* mol = mols[i];

    // Verify molecule has at least one conformer
    ASSERT_GT(mol->getNumConformers(), 0) << "Molecule " << i << " should have at least one conformer after embedding";

    // Check energy improvement for each conformer
    const int numConformers = mol->getNumConformers();
    for (int confId = 0; confId < numConformers; ++confId) {
      // Get the conformer
      const auto& conf = mol->getConformer(confId);

      // Setup force field with the current conformer
      nvMolKit::detail::EmbedArgs                 eargs;
      std::vector<std::unique_ptr<RDGeom::Point>> positions;
      std::unique_ptr<ForceFields::ForceField>    field;

      // Use RDKit's default first minimization force field for energy comparison
      auto option = RDKit::DGeomHelpers::ETKDGv3;
      nvMolKit::DGeomHelpers::setupRDKitFFWithPos(mol, option, field, eargs, positions, confId);

      // Calculate final energy
      const double finalEnergy = field->calcEnergy();
      ASSERT_GE(finalEnergy, 0.0) << "Molecule " << i << " conformer " << confId << ": Final energy should be positive";

      // Verify energy has improved
      EXPECT_LT(finalEnergy, initialEnergies[i])
        << "Molecule " << i << " conformer " << confId << ": Final energy (" << finalEnergy
        << ") should be less than initial energy (" << initialEnergies[i] << ")";
    }
  }
}

// Helper function to compare conformer energies between RDKit and nvMolKit
void testConformerEnergyComparison(const std::vector<RDKit::ROMol*>&    mols,
                                   int                                  confsPerMolecule = 1,
                                   RDKit::DGeomHelpers::EmbedParameters params           = RDKit::DGeomHelpers::ETKDGv3,
                                   nvMolKit::BatchHardwareOptions       hardwareOptions  = {10, 7, 10},
                                   nvMolKit::BfgsBackend                backend = nvMolKit::BfgsBackend::PER_MOLECULE) {
  // Create hard copies of input molecules
  std::vector<std::unique_ptr<RDKit::RWMol>> molCopies;
  std::vector<RDKit::ROMol*>                 molCopyPtrs;
  for (const auto& mol : mols) {
    molCopies.push_back(std::make_unique<RDKit::RWMol>(*mol));
    molCopyPtrs.push_back(molCopies.back().get());
  }
  params.useRandomCoords = true;
  params.basinThresh     = 1e8;
  // Generate conformers using RDKit for copied molecules
  for (size_t i = 0; i < molCopyPtrs.size(); ++i) {
    auto*                       molCopy = molCopyPtrs[i];
    nvMolKit::detail::EmbedArgs eargs;

    std::vector<int> res;
    RDKit::DGeomHelpers::EmbedMultipleConfs(*molCopy, res, confsPerMolecule, params);
  }

  nvMolKit::embedMolecules(mols, params, confsPerMolecule, -1, true, nullptr, hardwareOptions, backend);

  // Compare energies for each pair of molecules
  for (size_t i = 0; i < mols.size(); ++i) {
    auto* mol     = mols[i];
    auto* molCopy = molCopyPtrs[i];

    // Verify both molecules have the expected number of conformers
    ASSERT_EQ(mol->getNumConformers(), confsPerMolecule)
      << "Original molecule " << i << " should have " << confsPerMolecule << " conformers";
    ASSERT_EQ(molCopy->getNumConformers(), confsPerMolecule)
      << "Copied molecule " << i << " should have " << confsPerMolecule << " conformers";

    // Calculate average energies across all conformers
    double totalEnergy1 = 0.0;
    double totalEnergy2 = 0.0;

    for (int confId = 0; confId < confsPerMolecule; ++confId) {
      // Get energies for both conformers
      nvMolKit::detail::EmbedArgs                 eargs1, eargs2;
      std::vector<std::unique_ptr<RDGeom::Point>> positions1, positions2;
      std::unique_ptr<ForceFields::ForceField>    field1, field2;

      // Use RDKit's default first minimization force field for energy comparison
      auto option = RDKit::DGeomHelpers::ETKDGv3;
      nvMolKit::DGeomHelpers::setupRDKitFFWithPos(mol, option, field1, eargs1, positions1, confId);
      nvMolKit::DGeomHelpers::setupRDKitFFWithPos(molCopy, option, field2, eargs2, positions2, confId);

      totalEnergy1 += field1->calcEnergy();
      totalEnergy2 += field2->calcEnergy();
    }

    double avgEnergy1 = totalEnergy1 / confsPerMolecule;
    double avgEnergy2 = totalEnergy2 / confsPerMolecule;

    EXPECT_NEAR(avgEnergy1, avgEnergy2, 2e-1)
      << "Molecule " << i << ": Average energy difference between embedMultipleConfs (" << avgEnergy2
      << ") and embedMolecules (" << avgEnergy1 << ") exceeds tolerance";
  }
}

}  // anonymous namespace

class ETKDGPipelineEnergyTestFixture : public ::testing::TestWithParam<std::tuple<ETKDGOption, nvMolKit::BfgsBackend>> {
 public:
  ETKDGPipelineEnergyTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

  void SetUp() override {
    // Load multiple different molecules from MMFF94_dative.sdf
    std::vector<std::unique_ptr<RDKit::ROMol>> tempMols;
    constexpr int                              count = 5;
    getMols(testDataFolderPath_ + "/MMFF94_dative.sdf", tempMols, /*count=*/count);
    ASSERT_EQ(tempMols.size(), count) << "Expected to load 5 molecules";

    // Convert to RWMol and prepare molecules
    for (auto& tempMol : tempMols) {
      molsPtrs_.push_back(std::make_unique<RDKit::RWMol>(*tempMol));
    }

    // Clear conformers and sanitize all molecules and prepare mols_ vector with pointers
    for (auto& molPtr : molsPtrs_) {
      molPtr->clearConformers();
      RDKit::MolOps::sanitizeMol(*molPtr);
      mols_.push_back(molPtr.get());
    }
    ASSERT_EQ(mols_.size(), count) << "Expected 5 molecules";
  }

 protected:
  std::string                                testDataFolderPath_;
  std::vector<std::unique_ptr<RDKit::RWMol>> molsPtrs_;
  std::vector<RDKit::ROMol*>                 mols_;
};

TEST_P(ETKDGPipelineEnergyTestFixture, SingleMoleculeEnergyImprovement) {
  auto                       params    = getETKDGOption(std::get<0>(GetParam()));
  auto                       backend   = std::get<1>(GetParam());
  std::vector<RDKit::ROMol*> singleMol = {mols_[0]};
  testEnergyImprovement(singleMol, 1, params, backend);
}

TEST_P(ETKDGPipelineEnergyTestFixture, MultipleMoleculesEnergyImprovement) {
  auto params  = getETKDGOption(std::get<0>(GetParam()));
  auto backend = std::get<1>(GetParam());
  testEnergyImprovement(mols_, 1, params, backend);
}

TEST_P(ETKDGPipelineEnergyTestFixture, SingleMoleculeMultipleConformers) {
  auto                       params    = getETKDGOption(std::get<0>(GetParam()));
  auto                       backend   = std::get<1>(GetParam());
  std::vector<RDKit::ROMol*> singleMol = {mols_[0]};
  testEnergyImprovement(singleMol, 5, params, backend);
}

TEST_P(ETKDGPipelineEnergyTestFixture, MultipleMoleculesMultipleConformers) {
  auto params  = getETKDGOption(std::get<0>(GetParam()));
  auto backend = std::get<1>(GetParam());
  testEnergyImprovement(mols_, 3, params, backend);
}

TEST_P(ETKDGPipelineEnergyTestFixture, SingleMoleculeConformerEnergyComparison) {
  auto                       params    = getETKDGOption(std::get<0>(GetParam()));
  auto                       backend   = std::get<1>(GetParam());
  std::vector<RDKit::ROMol*> singleMol = {mols_[0]};
  testConformerEnergyComparison(singleMol, 1, params, {10, 7, 10}, backend);
}

TEST_P(ETKDGPipelineEnergyTestFixture, MultipleMoleculesConformerEnergyComparison) {
  auto params  = getETKDGOption(std::get<0>(GetParam()));
  auto backend = std::get<1>(GetParam());
  testConformerEnergyComparison(mols_, 1, params, {10, 7, 10}, backend);
}

TEST_P(ETKDGPipelineEnergyTestFixture, SingleMoleculeMultipleConformersEnergyComparison) {
  auto                       params    = getETKDGOption(std::get<0>(GetParam()));
  auto                       backend   = std::get<1>(GetParam());
  std::vector<RDKit::ROMol*> singleMol = {mols_[0]};
  testConformerEnergyComparison(singleMol, 10, params, {10, 7, 10}, backend);
}

TEST_P(ETKDGPipelineEnergyTestFixture, MultipleMoleculesMultipleConformersEnergyComparison) {
  auto params  = getETKDGOption(std::get<0>(GetParam()));
  auto backend = std::get<1>(GetParam());
  testConformerEnergyComparison(mols_, 10, params, {10, 7, 10}, backend);
}

TEST_P(ETKDGPipelineEnergyTestFixture, DefaultHardwareOptionsOpenMPMaxThreads) {
  // Test using default BatchHardwareOptions which should use omp_get_max_threads()
  auto                                 params  = getETKDGOption(std::get<0>(GetParam()));
  auto                                 backend = std::get<1>(GetParam());
  const nvMolKit::BatchHardwareOptions defaultOptions;                   // Uses -1 values for automatic detection
  const std::vector<RDKit::ROMol*>     testMols = {mols_[0], mols_[1]};  // Use subset for efficiency
  testConformerEnergyComparison(testMols, 2, params, defaultOptions, backend);
}

TEST_P(ETKDGPipelineEnergyTestFixture, SpecificGpuIds) {
  // Test using GPU ID 0 (should always be available if CUDA is working)
  auto                           params  = getETKDGOption(std::get<0>(GetParam()));
  auto                           backend = std::get<1>(GetParam());
  nvMolKit::BatchHardwareOptions customOptions;
  customOptions.preprocessingThreads = 1;
  customOptions.batchSize            = 5;
  customOptions.batchesPerGpu        = 1;
  customOptions.gpuIds.push_back(0);  // Use GPU 0

  const std::vector<RDKit::ROMol*> testMols = {mols_[0]};
  testConformerEnergyComparison(testMols, 2, params, customOptions, backend);
}

TEST_P(ETKDGPipelineEnergyTestFixture, NonZeroGPUID) {
  // Requires multiple GPUs
  const int numDevices = nvMolKit::countCudaDevices();
  if (numDevices < 2) {
    GTEST_SKIP() << "Test requires multiple GPUs, only " << numDevices << " available";
  }

  auto                           params  = getETKDGOption(std::get<0>(GetParam()));
  auto                           backend = std::get<1>(GetParam());
  nvMolKit::BatchHardwareOptions customOptions;
  customOptions.preprocessingThreads = 1;
  customOptions.batchSize            = 5;
  customOptions.batchesPerGpu        = 1;
  customOptions.gpuIds.push_back(1);  // Use GPU 1 (second GPU)

  const std::vector<RDKit::ROMol*> testMols = {mols_[0]};
  testConformerEnergyComparison(testMols, 2, params, customOptions, backend);
}

TEST_P(ETKDGPipelineEnergyTestFixture, MultiGPUSpecificIds) {
  // Requires multiple GPUs
  const int numDevices = nvMolKit::countCudaDevices();
  if (numDevices < 2) {
    GTEST_SKIP() << "Test requires multiple GPUs, only " << numDevices << " available";
  }

  auto                           params  = getETKDGOption(std::get<0>(GetParam()));
  auto                           backend = std::get<1>(GetParam());
  nvMolKit::BatchHardwareOptions customOptions;
  customOptions.preprocessingThreads = 1;
  customOptions.batchSize            = 5;
  customOptions.batchesPerGpu        = 1;
  customOptions.gpuIds.push_back(0);
  customOptions.gpuIds.push_back(1);

  testConformerEnergyComparison(mols_, 2, params, customOptions, backend);
}

// Instantiate parameterized tests for different ETKDG variants and backends
INSTANTIATE_TEST_SUITE_P(ETKDGVariants,
                         ETKDGPipelineEnergyTestFixture,
                         ::testing::Combine(::testing::Values(ETKDGOption::ETKDG,
                                                              ETKDGOption::ETKDGv2,
                                                              ETKDGOption::srETKDGv3,
                                                              ETKDGOption::ETKDGv3,
                                                              ETKDGOption::KDG,
                                                              ETKDGOption::ETDG,
                                                              ETKDGOption::DG),
                                            ::testing::Values(nvMolKit::BfgsBackend::BATCHED,
                                                              nvMolKit::BfgsBackend::PER_MOLECULE)),
                         [](const ::testing::TestParamInfo<std::tuple<ETKDGOption, nvMolKit::BfgsBackend>>& info) {
                           std::string backend_name =
                             (std::get<1>(info.param) == nvMolKit::BfgsBackend::BATCHED) ? "Batched" : "PerMolecule";
                           return getETKDGOptionName(std::get<0>(info.param)) + "_" + backend_name;
                         });

class ETKDGBackendTestFixture : public ::testing::TestWithParam<nvMolKit::BfgsBackend> {};

TEST_P(ETKDGBackendTestFixture, LargeMoleculeSoloEmbeds) {
  auto backend = GetParam();

  // Oversized linear hydrocarbon (>256 atoms)
  const std::string bigSmiles(100, 'C');
  auto              big = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(bigSmiles));
  RDKit::MolOps::addHs(*big);
  ASSERT_NE(big, nullptr);
  ASSERT_GT(big->getNumAtoms(), 256u);

  // Params
  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.useRandomCoords                      = true;
  params.maxIterations                        = 10;

  std::vector<RDKit::ROMol*>        mols = {big.get()};
  std::vector<std::vector<int16_t>> failures;
  nvMolKit::embedMolecules(mols, params, 1, -1, false, &failures, {}, backend);
  EXPECT_EQ(big->getNumConformers(), 1);
}

TEST_P(ETKDGBackendTestFixture, LargeMoleculeInterleavedEmbeds) {
  auto backend = GetParam();

  // Small molecules
  auto small1 = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCCC"));
  auto small2 = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCC"));
  ASSERT_NE(small1, nullptr);
  ASSERT_NE(small2, nullptr);

  // Oversized linear hydrocarbon (>256 atoms)
  const std::string bigSmiles(100, 'C');
  auto              big = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(bigSmiles));
  RDKit::MolOps::addHs(*big);
  ASSERT_NE(big, nullptr);
  ASSERT_GT(big->getNumAtoms(), 256u);

  // Params
  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.useRandomCoords                      = true;
  params.maxIterations                        = 5;

  std::vector<RDKit::ROMol*>        mols = {small1.get(), big.get(), small2.get()};
  std::vector<std::vector<int16_t>> failures;
  nvMolKit::embedMolecules(mols, params, 1, -1, false, &failures, {}, backend);
  EXPECT_EQ(small1->getNumConformers(), 1);
  EXPECT_EQ(small2->getNumConformers(), 1);
  EXPECT_EQ(big->getNumConformers(), 1);
}

TEST_P(ETKDGBackendTestFixture, MultipleMoleculesSomeDuplicationLikely) {
  auto backend = GetParam();

  const auto benzene = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("c1ccccc1"));
  ASSERT_NE(benzene, nullptr);

  const std::string longChainSmiles(30, 'C');
  const auto        longChain = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(longChainSmiles));
  ASSERT_NE(longChain, nullptr);

  const std::vector<RDKit::ROMol*> mols   = {benzene.get(), longChain.get()};
  auto                             params = RDKit::DGeomHelpers::ETKDGv3;
  params.useRandomCoords                  = true;
  params.pruneRmsThresh                   = 0.5;
  nvMolKit::embedMolecules(mols, params, /*confsPerMolecule=*/5, -1, false, nullptr, {}, backend);
  // Benzene only ever has one conformer
  EXPECT_EQ(benzene->getNumConformers(), 1);
  // Long chain should all be sufficiently sepa
  EXPECT_EQ(longChain->getNumConformers(), 5);
}

INSTANTIATE_TEST_SUITE_P(BfgsBackends,
                         ETKDGBackendTestFixture,
                         ::testing::Values(nvMolKit::BfgsBackend::BATCHED, nvMolKit::BfgsBackend::PER_MOLECULE),
                         [](const ::testing::TestParamInfo<nvMolKit::BfgsBackend>& info) {
                           return (info.param == nvMolKit::BfgsBackend::BATCHED) ? "Batched" : "PerMolecule";
                         });