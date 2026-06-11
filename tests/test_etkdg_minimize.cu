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

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/ChiralSet.h>
#include <ForceField/ForceField.h>
#include <gmock/gmock.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <gtest/gtest.h>

#include <filesystem>

#include "src/embedder_utils.h"
#include "src/etkdg_impl.h"
#include "src/etkdg_stage_coordgen.h"
#include "src/etkdg_stage_distgeom_minimize.h"
#include "src/forcefields/dist_geom.h"
#include "src/utils/host_vector.h"
#include "tests/test_utils.h"

using namespace ::nvMolKit::detail;

using ETKDGStageTestParams = std::tuple<ETKDGOption, int>;

namespace {

// Helper function for common initialization logic
void initTestComponentsCommon(const std::vector<const RDKit::ROMol*>&           mols,
                              const std::vector<std::unique_ptr<RDKit::RWMol>>& molsPtrs,
                              ETKDGContext&                                     context,
                              std::vector<nvMolKit::detail::EmbedArgs>&         eargsVec,
                              RDKit::DGeomHelpers::EmbedParameters&             embedParam) {
  // Initialize context
  context.nTotalSystems         = mols.size();
  context.systemHost.atomStarts = {0};
  context.systemHost.positions.clear();

  // Process each molecule
  for (size_t i = 0; i < mols.size(); ++i) {
    nvMolKit::detail::EmbedArgs eargs;
    eargs.dim = 4;
    std::vector<std::unique_ptr<RDGeom::Point>> positions;
    std::unique_ptr<ForceFields::ForceField>    field;

    // Setup force field and get parameters
    nvMolKit::DGeomHelpers::setupRDKitFFWithPos(molsPtrs[i].get(),
                                                embedParam,
                                                field,
                                                eargs,
                                                positions,
                                                -1,  // Use default conformer
                                                nvMolKit::DGeomHelpers::Dimensionality::DIM_4D);

    // Add molecule to context with positions
    nvMolKit::DistGeom::addMoleculeToContextWithPositions(eargs.posVec,
                                                          eargs.dim,
                                                          context.systemHost.atomStarts,
                                                          context.systemHost.positions);

    // Store embed args for later use
    eargsVec.push_back(std::move(eargs));
  }

  // Send context to device
  nvMolKit::DistGeom::sendContextToDevice(context.systemHost.positions,
                                          context.systemDevice.positions,
                                          context.systemHost.atomStarts,
                                          context.systemDevice.atomStarts);
}

// Helper function to calculate initial energies for one or more molecules
std::vector<double> calculateInitialEnergies(const std::vector<std::unique_ptr<RDKit::RWMol>>& mols) {
  std::vector<double> initialEnergies;
  initialEnergies.reserve(mols.size());
  auto params = RDKit::DGeomHelpers::ETKDGv3;
  for (size_t i = 0; i < mols.size(); ++i) {
    nvMolKit::detail::EmbedArgs                 eargs;
    std::vector<std::unique_ptr<RDGeom::Point>> positions;
    std::unique_ptr<ForceFields::ForceField>    field;
    nvMolKit::DGeomHelpers::setupRDKitFFWithPos(mols[i].get(), params, field, eargs, positions);
    const double initialEnergy = field->calcEnergy();
    EXPECT_GE(initialEnergy, 0.0) << "Initial energy should be non-negative for molecule " << i;
    initialEnergies.push_back(initialEnergy);
  }

  return initialEnergies;
}

// Helper function to check final energies against initial energies and failure counts
void checkFinalEnergies(const std::vector<double>&                        finalEnergies,
                        const std::vector<double>&                        initialEnergies,
                        const std::vector<std::unique_ptr<RDKit::RWMol>>& mols,
                        const std::vector<std::vector<int16_t>>&          failureCounts,
                        const std::string&                                context = "") {
  for (size_t i = 0; i < mols.size(); ++i) {
    // Skip if initial energy is 0 or molecule failed in any stage
    bool shouldSkip = (initialEnergies[i] == 0.0);
    for (const auto& stageFailures : failureCounts) {
      if (stageFailures[i] > 0) {
        shouldSkip = true;
        break;
      }
    }
    if (shouldSkip) {
      continue;
    }

    EXPECT_LT(finalEnergies[i], initialEnergies[i])
      << context << "Molecule " << i << ": Final energy (" << finalEnergies[i]
      << ") should be less than initial energy (" << initialEnergies[i] << ")";

    const int    numAtoms      = mols[i]->getNumAtoms();
    const double energyPerAtom = finalEnergies[i] / numAtoms;
    EXPECT_LT(energyPerAtom, nvMolKit::detail::MAX_MINIMIZED_E_PER_ATOM)
      << context << "Molecule " << i << ": Energy per atom (" << energyPerAtom << ") should be below threshold ("
      << nvMolKit::detail::MAX_MINIMIZED_E_PER_ATOM << ")";
  }
}

}  // anonymous namespace

// Test fixture for single molecule tests
class ETKDGMinimizeSingleMolTestFixture : public ::testing::TestWithParam<ETKDGOption> {
 public:
  ETKDGMinimizeSingleMolTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

  void SetUp() override {
    // Load molecule
    const std::string mol2FilePath = testDataFolderPath_ + "/rdkit_smallmol_1.mol2";
    ASSERT_TRUE(std::filesystem::exists(mol2FilePath)) << "Could not find " << mol2FilePath;
    molPtr_ = std::unique_ptr<RDKit::RWMol>(RDKit::MolFileToMol(mol2FilePath, false));
    ASSERT_NE(molPtr_, nullptr);
    molPtr_->clearConformers();
    RDKit::MolOps::sanitizeMol(*molPtr_);

    // Initialize mols_ vector with the single molecule
    mols_.push_back(molPtr_.get());

    // Initialize molsPtrs_ vector for the common function
    molsPtrs_.push_back(std::move(molPtr_));

    // Initialize common test components
    embedParam_                 = getETKDGOption(GetParam());
    embedParam_.useRandomCoords = true;
    initTestComponents();

    // Create minimizer after context is initialized
    minimizer_ = std::make_unique<nvMolKit::BfgsBatchMinimizer>(4, nvMolKit::DebugLevel::NONE, true, nullptr);

    // Pre-allocate scratch buffers for stages
    const size_t totalAtoms = context_.systemHost.atomStarts.back();
    positionsScratch_.resize(totalAtoms * 4);  // 4D for ETKDG
    activeScratch_.resize(mols_.size());
  }

  void initTestComponents() { initTestComponentsCommon(mols_, molsPtrs_, context_, eargs_, embedParam_); }

 protected:
  std::string                                   testDataFolderPath_;
  std::unique_ptr<RDKit::RWMol>                 molPtr_;
  std::vector<std::unique_ptr<RDKit::RWMol>>    molsPtrs_;
  std::vector<const RDKit::ROMol*>              mols_;
  ETKDGContext                                  context_;
  std::vector<nvMolKit::detail::EmbedArgs>      eargs_;
  RDKit::DGeomHelpers::EmbedParameters          embedParam_;
  std::unique_ptr<nvMolKit::BfgsBatchMinimizer> minimizer_;
  nvMolKit::PinnedHostVector<double>            positionsScratch_;
  nvMolKit::PinnedHostVector<uint8_t>           activeScratch_;
};

// BFGS Stage Tests
TEST_P(ETKDGMinimizeSingleMolTestFixture, FirstMinimizeStageBFGSTest) {
  // Calculate initial energy
  const std::vector<double> initialEnergies = calculateInitialEnergies(molsPtrs_);

  // Create FirstMinimizeStage
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  auto                                     stage    = std::make_unique<nvMolKit::detail::DistGeomMinimizeStage>(mols_,
                                                                         eargs_,
                                                                         embedParam_,
                                                                         context_,
                                                                         *minimizer_,
                                                                         1.0,
                                                                         0.1,
                                                                         400,
                                                                         true,
                                                                         "First Minimization");
  auto*                                    stagePtr = stage.get();  // Store pointer before moving
  stages.push_back(std::move(stage));

  // Create and run driver
  ETKDGDriver driver(std::make_unique<ETKDGContext>(std::move(context_)), std::move(stages));
  driver.run(5);

  // Get final energy from the stage
  std::vector<double> finalEnergies(stagePtr->molSystemDevice.energyOuts.size());
  stagePtr->molSystemDevice.energyOuts.copyToHost(finalEnergies);

  // Check other results first
  EXPECT_EQ(driver.numConfsFinished(), 1);
  EXPECT_EQ(driver.iterationsComplete(), 1);

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 1);                      // One stage
  EXPECT_THAT(failureCounts[0], testing::ElementsAre(0));  // FirstMinimizeStage

  auto completed = driver.completedConformers();
  EXPECT_THAT(completed, testing::ElementsAre(1));

  // Check energy reduction and threshold
  checkFinalEnergies(finalEnergies, initialEnergies, molsPtrs_, failureCounts);
}

TEST_P(ETKDGMinimizeSingleMolTestFixture, FourthDimMinimizeStageBFGSTest) {
  // Create FourthDimMinimizeStage
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  stages.push_back(std::make_unique<nvMolKit::detail::DistGeomMinimizeStage>(mols_,
                                                                             eargs_,
                                                                             embedParam_,
                                                                             context_,
                                                                             *minimizer_,
                                                                             0.2,
                                                                             1.0,
                                                                             200,
                                                                             false,
                                                                             "Fourth Dimension Minimization"));

  // Create and run driver
  ETKDGDriver driver(std::make_unique<ETKDGContext>(std::move(context_)), std::move(stages));
  driver.run(5);

  // Check results
  EXPECT_EQ(driver.numConfsFinished(), 1);
  EXPECT_EQ(driver.iterationsComplete(), 1);

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 1);                      // One stage
  EXPECT_THAT(failureCounts[0], testing::ElementsAre(0));  // FourthDimMinimizeStage

  auto completed = driver.completedConformers();
  EXPECT_THAT(completed, testing::ElementsAre(1));
}

TEST_P(ETKDGMinimizeSingleMolTestFixture, FullMinimizationPipelineBFGSTest) {
  // Calculate initial energy
  const std::vector<double> initialEnergies = calculateInitialEnergies(molsPtrs_);

  // Create stages - first is base DistGeomMinimizeStage, second is wrapper with different weights
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  auto                                     firstStage = std::make_unique<nvMolKit::detail::DistGeomMinimizeStage>(mols_,
                                                                              eargs_,
                                                                              embedParam_,
                                                                              context_,
                                                                              *minimizer_,
                                                                              1.0,
                                                                              0.1,
                                                                              400,
                                                                              true,
                                                                              "First Minimization");
  auto*                                    firstStagePtr = firstStage.get();  // Store pointer before moving
  stages.push_back(std::move(firstStage));
  stages.push_back(std::make_unique<nvMolKit::detail::DistGeomMinimizeWrapperStage>(*firstStagePtr,
                                                                                    0.2,
                                                                                    1.0,
                                                                                    200,
                                                                                    false,
                                                                                    "Fourth Dimension Minimization"));

  // Create and run driver
  ETKDGDriver driver(std::make_unique<ETKDGContext>(std::move(context_)), std::move(stages));
  driver.run(5);

  // Get final energy from the first stage
  std::vector<double> finalEnergies(firstStagePtr->molSystemDevice.energyOuts.size());
  firstStagePtr->molSystemDevice.energyOuts.copyToHost(finalEnergies);

  // Check other results first
  EXPECT_EQ(driver.numConfsFinished(), 1);
  EXPECT_EQ(driver.iterationsComplete(), 1);

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 2);                      // Two stages
  EXPECT_THAT(failureCounts[0], testing::ElementsAre(0));  // FirstMinimizeStage
  EXPECT_THAT(failureCounts[1], testing::ElementsAre(0));  // FourthDimMinimizeStage

  auto completed = driver.completedConformers();
  EXPECT_THAT(completed, testing::ElementsAre(1));

  // Check energy reduction and threshold
  checkFinalEnergies(finalEnergies, initialEnergies, molsPtrs_, failureCounts);
}

TEST_P(ETKDGMinimizeSingleMolTestFixture, FirstPartETKDGPipelineBFGSTest) {
  constexpr int16_t         maxFailedIterations = 2;
  // Calculate initial energy
  const std::vector<double> initialEnergies     = calculateInitialEnergies(molsPtrs_);

  // Zero out positions on device since we are using coordgen stage for generating initial coordinates
  context_.systemDevice.positions.zero();

  // Create stages in order: coordgen -> first minimize BFGS -> fourthdim BFGS
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  stages.push_back(std::make_unique<nvMolKit::detail::ETKDGCoordGenRDKitStage>(embedParam_,
                                                                               mols_,
                                                                               eargs_,
                                                                               positionsScratch_,
                                                                               activeScratch_));
  auto  firstStage    = std::make_unique<nvMolKit::detail::DistGeomMinimizeStage>(mols_,
                                                                              eargs_,
                                                                              embedParam_,
                                                                              context_,
                                                                              *minimizer_,
                                                                              1.0,
                                                                              0.1,
                                                                              400,
                                                                              true,
                                                                              "First Minimization");
  auto* firstStagePtr = firstStage.get();  // Store pointer before moving
  stages.push_back(std::move(firstStage));
  stages.push_back(std::make_unique<nvMolKit::detail::DistGeomMinimizeWrapperStage>(*firstStagePtr,
                                                                                    0.2,
                                                                                    1.0,
                                                                                    200,
                                                                                    false,
                                                                                    "Fourth Dimension Minimization"));

  // Create and run driver
  ETKDGDriver driver(std::make_unique<ETKDGContext>(std::move(context_)), std::move(stages));
  driver.run(3);

  // Get final energy from the first stage
  std::vector<double> finalEnergies(firstStagePtr->molSystemDevice.energyOuts.size());
  firstStagePtr->molSystemDevice.energyOuts.copyToHost(finalEnergies);

  // Check other results first
  EXPECT_EQ(driver.numConfsFinished(), 1);
  EXPECT_LE(driver.iterationsComplete(), 2);  // Allow for 1 failure.

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 3);                                              // Three stages
  EXPECT_THAT(failureCounts[0], testing::Each(0));                                 // CoordGenStage
  EXPECT_THAT(failureCounts[1], testing::Each(testing::Le(maxFailedIterations)));  // FirstMinimizeStage
  EXPECT_THAT(failureCounts[2], testing::Each(testing::Le(maxFailedIterations)));  // FourthDimMinimizeStage

  auto completed = driver.completedConformers();
  EXPECT_THAT(completed, testing::ElementsAre(1));

  // Check energy reduction and threshold
  checkFinalEnergies(finalEnergies, initialEnergies, molsPtrs_, failureCounts);
}

// Test fixture for multiple diverse molecules tests
class ETKDGMinimizeMultiMolDiverseTestFixture : public ::testing::TestWithParam<ETKDGOption> {
 public:
  ETKDGMinimizeMultiMolDiverseTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

  void SetUp() override {
    // Load multiple different molecules from MMFF94_dative.sdf
    std::vector<std::unique_ptr<RDKit::ROMol>> tempMols;
    getMols(testDataFolderPath_ + "/MMFF94_dative.sdf", tempMols, /*count=*/5);
    ASSERT_EQ(tempMols.size(), 5) << "Expected to load 5 molecules";

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
    ASSERT_EQ(mols_.size(), 5) << "Expected 5 molecules";

    // Initialize common test components
    embedParam_                 = getETKDGOption(GetParam());
    embedParam_.useRandomCoords = true;
    initTestComponents();

    // Create minimizer after context is initialized
    minimizer_ = std::make_unique<nvMolKit::BfgsBatchMinimizer>(4, nvMolKit::DebugLevel::NONE, true, nullptr);

    // Pre-allocate scratch buffers for stages
    const size_t totalAtoms = context_.systemHost.atomStarts.back();
    positionsScratch_.resize(totalAtoms * 4);  // 4D for ETKDG
    activeScratch_.resize(mols_.size());
  }

  void initTestComponents() { initTestComponentsCommon(mols_, molsPtrs_, context_, eargs_, embedParam_); }

 protected:
  std::string                                   testDataFolderPath_;
  std::vector<std::unique_ptr<RDKit::RWMol>>    molsPtrs_;
  std::vector<const RDKit::ROMol*>              mols_;
  ETKDGContext                                  context_;
  std::vector<nvMolKit::detail::EmbedArgs>      eargs_;
  RDKit::DGeomHelpers::EmbedParameters          embedParam_;
  std::unique_ptr<nvMolKit::BfgsBatchMinimizer> minimizer_;
  nvMolKit::PinnedHostVector<double>            positionsScratch_;
  nvMolKit::PinnedHostVector<uint8_t>           activeScratch_;
};

// BFGS Stage Tests for diverse molecules
TEST_P(ETKDGMinimizeMultiMolDiverseTestFixture, FirstMinimizeStageBFGSTest) {
  constexpr int16_t maxFailedIterations = 2;

  // Calculate initial energies for all molecules
  const std::vector<double> initialEnergies = calculateInitialEnergies(molsPtrs_);

  // Create FirstMinimizeStage
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  auto                                     stage    = std::make_unique<nvMolKit::detail::DistGeomMinimizeStage>(mols_,
                                                                         eargs_,
                                                                         embedParam_,
                                                                         context_,
                                                                         *minimizer_,
                                                                         1.0,
                                                                         0.1,
                                                                         400,
                                                                         true,
                                                                         "First Minimization");
  auto*                                    stagePtr = stage.get();  // Store pointer before moving
  stages.push_back(std::move(stage));

  // Create and run driver
  ETKDGDriver driver(std::make_unique<ETKDGContext>(std::move(context_)), std::move(stages));
  driver.run(2);

  // Get final energies from the stage
  std::vector<double> finalEnergies(stagePtr->molSystemDevice.energyOuts.size());
  stagePtr->molSystemDevice.energyOuts.copyToHost(finalEnergies);

  // Get failure counts
  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 1);                                              // One stage
  EXPECT_THAT(failureCounts[0], testing::Each(testing::Le(maxFailedIterations)));  // FirstMinimizeStage

  // Check other results
  EXPECT_GE(driver.numConfsFinished(), 3);
  EXPECT_LE(driver.iterationsComplete(), 2);

  auto completed = driver.completedConformers();
  EXPECT_THAT(completed, testing::Each(testing::AnyOf(0, 1)));

  // Check energy reduction and threshold for each molecule
  checkFinalEnergies(finalEnergies, initialEnergies, molsPtrs_, failureCounts);
}

TEST_P(ETKDGMinimizeMultiMolDiverseTestFixture, FourthDimMinimizeStageBFGSTest) {
  // Create FourthDimMinimizeStage
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  stages.push_back(std::make_unique<nvMolKit::detail::DistGeomMinimizeStage>(mols_,
                                                                             eargs_,
                                                                             embedParam_,
                                                                             context_,
                                                                             *minimizer_,
                                                                             0.2,
                                                                             1.0,
                                                                             200,
                                                                             false,
                                                                             "Fourth Dimension Minimization"));

  // Create and run driver
  ETKDGDriver driver(std::make_unique<ETKDGContext>(std::move(context_)), std::move(stages));
  driver.run(2);

  // Check results
  EXPECT_EQ(driver.numConfsFinished(), 5);
  EXPECT_EQ(driver.iterationsComplete(), 1);

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 1);               // One stage
  EXPECT_THAT(failureCounts[0], testing::Each(0));  // FourthDimMinimizeStage

  auto completed = driver.completedConformers();
  EXPECT_THAT(completed, testing::Each(1));
}

TEST_P(ETKDGMinimizeMultiMolDiverseTestFixture, FullMinimizationPipelineBFGSTest) {
  constexpr int16_t maxFailedIterations = 2;

  // Calculate initial energies for all molecules
  const std::vector<double> initialEnergies = calculateInitialEnergies(molsPtrs_);

  // Create stages - first is base DistGeomMinimizeStage, second is wrapper with different weights
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  auto                                     firstStage = std::make_unique<nvMolKit::detail::DistGeomMinimizeStage>(mols_,
                                                                              eargs_,
                                                                              embedParam_,
                                                                              context_,
                                                                              *minimizer_,
                                                                              1.0,
                                                                              0.1,
                                                                              400,
                                                                              true,
                                                                              "First Minimization");
  auto*                                    firstStagePtr = firstStage.get();  // Store pointer before moving
  stages.push_back(std::move(firstStage));
  stages.push_back(std::make_unique<nvMolKit::detail::DistGeomMinimizeWrapperStage>(*firstStagePtr,
                                                                                    0.2,
                                                                                    1.0,
                                                                                    200,
                                                                                    false,
                                                                                    "Fourth Dimension Minimization"));

  // Create and run driver
  ETKDGDriver driver(std::make_unique<ETKDGContext>(std::move(context_)), std::move(stages));
  driver.run(2);

  // Get final energies from the first stage
  std::vector<double> finalEnergies(firstStagePtr->molSystemDevice.energyOuts.size());
  firstStagePtr->molSystemDevice.energyOuts.copyToHost(finalEnergies);

  // Get failure counts
  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 2);                                              // Two stages
  EXPECT_THAT(failureCounts[0], testing::Each(testing::Le(maxFailedIterations)));  // FirstMinimizeStage
  EXPECT_THAT(failureCounts[1], testing::Each(testing::Le(maxFailedIterations)));  // FourthDimMinimizeStage

  // Check other results
  EXPECT_GE(driver.numConfsFinished(), 3);
  EXPECT_LE(driver.iterationsComplete(), 2);

  auto completed = driver.completedConformers();
  EXPECT_THAT(completed, testing::Each(testing::AnyOf(0, 1)));

  // Check energy reduction and threshold for each molecule
  checkFinalEnergies(finalEnergies, initialEnergies, molsPtrs_, failureCounts);
}

TEST_P(ETKDGMinimizeMultiMolDiverseTestFixture, FirstPartETKDGPipelineBFGSTest) {
  constexpr int16_t         maxFailedIterations = 2;
  // Calculate initial energies for all molecules
  const std::vector<double> initialEnergies     = calculateInitialEnergies(molsPtrs_);

  // Zero out positions on device since we are using coordgen stage for generating initial coordinates
  context_.systemDevice.positions.zero();

  // Create stages in order: coordgen -> first minimize BFGS -> fourthdim BFGS
  std::vector<std::unique_ptr<ETKDGStage>> stages;

  stages.push_back(std::make_unique<nvMolKit::detail::ETKDGCoordGenRDKitStage>(embedParam_,
                                                                               mols_,
                                                                               eargs_,
                                                                               positionsScratch_,
                                                                               activeScratch_));
  auto  firstStage    = std::make_unique<nvMolKit::detail::DistGeomMinimizeStage>(mols_,
                                                                              eargs_,
                                                                              embedParam_,
                                                                              context_,
                                                                              *minimizer_,
                                                                              1.0,
                                                                              0.1,
                                                                              400,
                                                                              true,
                                                                              "First Minimization");
  auto* firstStagePtr = firstStage.get();  // Store pointer before moving
  stages.push_back(std::move(firstStage));
  stages.push_back(std::make_unique<nvMolKit::detail::DistGeomMinimizeWrapperStage>(*firstStagePtr,
                                                                                    0.2,
                                                                                    1.0,
                                                                                    200,
                                                                                    false,
                                                                                    "Fourth Dimension Minimization"));
  // Create and run driver
  ETKDGDriver driver(std::make_unique<ETKDGContext>(std::move(context_)), std::move(stages));
  driver.run(3);
  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);

  // Get final energies from the first stage
  std::vector<double> finalEnergies(firstStagePtr->molSystemDevice.energyOuts.size());
  firstStagePtr->molSystemDevice.energyOuts.copyToHost(finalEnergies);
  cudaDeviceSynchronize();

  // Get failure counts
  EXPECT_EQ(failureCounts.size(), 3);                                              // Three stages
  EXPECT_THAT(failureCounts[0], testing::Each(0));                                 // CoordGenStage
  EXPECT_THAT(failureCounts[1], testing::Each(testing::Le(maxFailedIterations)));  // FirstMinimizeStage
  EXPECT_THAT(failureCounts[2], testing::Each(testing::Le(maxFailedIterations)));  // FourthDimMinimizeStage

  // Check other results
  EXPECT_GE(driver.numConfsFinished(), 3);
  EXPECT_LE(driver.iterationsComplete(), 3);

  // Check energy reduction and threshold for each molecule
  checkFinalEnergies(finalEnergies, initialEnergies, molsPtrs_, failureCounts);
}

TEST_P(ETKDGMinimizeMultiMolDiverseTestFixture, FirstMinimizeStageBFGSWithInactiveMolecules) {
  // Create FirstMinimizeStage
  auto stage = std::make_unique<nvMolKit::detail::DistGeomMinimizeStage>(mols_,
                                                                         eargs_,
                                                                         embedParam_,
                                                                         context_,
                                                                         *minimizer_,
                                                                         1.0,
                                                                         0.1,
                                                                         400,
                                                                         true,
                                                                         "First Minimization");

  // Set some molecules as inactive (let's say molecules 1 and 3)
  std::vector<uint8_t> activeRef(context_.nTotalSystems, 1);
  activeRef[1] = 0;  // Mark second molecule as inactive
  activeRef[3] = 0;  // Mark fourth molecule as inactive
  context_.activeThisStage.resize(context_.nTotalSystems);
  context_.activeThisStage.copyFromHost(activeRef);
  context_.failedThisStage.resize(context_.nTotalSystems);

  // Execute the stage
  stage->execute(context_);

  // Copy energy outputs from device to host
  std::vector<double> energyOuts(stage->molSystemDevice.energyOuts.size());
  stage->molSystemDevice.energyOuts.copyToHost(energyOuts);

  // Check that inactive molecules have zero energy
  for (int i = 0; i < context_.nTotalSystems; ++i) {
    if (activeRef[i] == 0) {
      // Empirical observation: inactive molecules typically have energy > 50
      // since they haven't been minimized by BFGS
      EXPECT_GT(energyOuts[i], 50.0) << "Inactive molecule " << i
                                     << " should have high energy (>50) since it wasn't minimized";
    } else {
      // Empirical observation: active molecules typically have energy < 0.1
      // after successful BFGS minimization
      EXPECT_LT(energyOuts[i], 0.1) << "Active molecule " << i
                                    << " should have low energy (<0.1) after successful minimization";
    }
  }
}

// Instantiate test suites for both fixtures
INSTANTIATE_TEST_SUITE_P(
  ETKDGOptions,
  ETKDGMinimizeSingleMolTestFixture,
  ::testing::Values(ETKDGOption::ETKDGv3, ETKDGOption::ETKDGv2, ETKDGOption::ETKDG, ETKDGOption::KDG),
  [](const ::testing::TestParamInfo<ETKDGOption>& info) { return getETKDGOptionName(info.param); });

// TODO: Currently only testing ETKDGv3 due to non-deterministic failures when testing multiple options.
// When multiple ETKDGOptions are tested together (even though each may pass individually),
// some tests randomly fail. This is likely due to stochastic processes in initTestComponentsCommon,
// specifically in the setupRDKitFFWithPos call which ports RDKit's original ETKDG pipeline.
// Previous attempts to resolve similar issues by cleaning up the RDKit porting were partially
// successful but not definitive. Further investigation is needed
INSTANTIATE_TEST_SUITE_P(ETKDGOptions,
                         ETKDGMinimizeMultiMolDiverseTestFixture,
                         ::testing::Values(ETKDGOption::ETKDGv3),
                         [](const ::testing::TestParamInfo<ETKDGOption>& info) {
                           return getETKDGOptionName(info.param);
                         });
