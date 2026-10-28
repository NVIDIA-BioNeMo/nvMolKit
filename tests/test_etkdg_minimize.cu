// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <DistGeom/DistGeomUtils.h>
#include <ForceField/ForceField.h>
#include <gmock/gmock.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionAngleContribs.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <gtest/gtest.h>
#include <RDGeneral/utils.h>

#include <filesystem>
#include <fstream>
#include <optional>

#include "rdkit_extensions/dist_geom_flattened_builder.h"
#include "rdkit_extensions/torsion_compat.h"
#include "src/embedder_utils.h"
#include "src/etkdg_impl.h"
#include "src/etkdg_stage_aio_minimization.h"
#include "src/etkdg_stage_coordgen.h"
#include "src/etkdg_stage_distgeom_minimize.h"
#include "src/etkdg_stage_etk_minimization.h"
#include "src/etkdg_stage_stereochem_checks.h"
#include "src/forcefields/dist_geom.h"
#include "src/utils/host_vector.h"
#include "tests/test_utils.h"
#include "versions.h"

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

#if RDKIT_AIO_ETKDG_API
namespace {

//! RDKit's MAX_MINIMIZED_E_PER_ATOM_AIO.
constexpr double kMaxAllInOneEnergyPerAtom = 0.1;

//! RDKit 2026.09's TorsionAngleContribs::getGrad mis-evaluates the six-fold term (fixed in rdkit/rdkit#9699). This
//! wrapper keeps RDKit's torsion energy and replaces the gradient with a central difference of it, so the reference
//! minimization follows the same energy surface as nvMolKit's analytic gradient.
class FiniteDifferenceTorsionContribs : public ForceFields::ForceFieldContrib {
 public:
  FiniteDifferenceTorsionContribs(ForceFields::ContribPtr torsions, const unsigned int numCoords)
      : torsions_(std::move(torsions)),
        numCoords_(numCoords) {}

  double getEnergy(double* pos) const override { return torsions_->getEnergy(pos); }

  void getGrad(double* pos, double* grad) const override {
    constexpr double step = 1e-6;
    for (unsigned int coord = 0; coord < numCoords_; ++coord) {
      const double original = pos[coord];
      pos[coord]            = original + step;
      const double plus     = torsions_->getEnergy(pos);
      pos[coord]            = original - step;
      const double minus    = torsions_->getEnergy(pos);
      pos[coord]            = original;
      grad[coord] += (plus - minus) / (2.0 * step);
    }
  }

  ForceFields::ForceFieldContrib* copy() const override { return new FiniteDifferenceTorsionContribs(*this); }

 private:
  ForceFields::ContribPtr torsions_;
  unsigned int            numCoords_;
};

void addTorsionTermsWithFiniteDifferenceGradient(ForceFields::ForceField&                    field,
                                                 const EmbedArgs&                            earg,
                                                 const RDKit::DGeomHelpers::EmbedParameters& params) {
  ::DistGeom::addTorsionTerms(&field, earg.etkdgDetails, params.useBasicKnowledge, params.useExpTorsionAnglePrefs);
  const unsigned int numCoords = field.positions().size() * field.dimension();
  for (auto& contrib : field.contribs()) {
    if (dynamic_cast<const ForceFields::CrystalFF::TorsionAngleContribs*>(contrib.get()) != nullptr) {
      contrib.reset(new FiniteDifferenceTorsionContribs(contrib, numCoords));
    }
  }
}

//! All-in-one force field of RDKit's minimizeAllInOne in its second phase, over the given 4D coordinates.
std::unique_ptr<ForceFields::ForceField> allInOneReferenceForcefield(EmbedArgs&                                  earg,
                                                                     std::vector<RDGeom::PointND>&               points,
                                                                     const RDKit::DGeomHelpers::EmbedParameters& params,
                                                                     const bool torsionTermsEnabled) {
  RDGeom::PointPtrVect pointPtrs;
  for (auto& point : points) {
    pointPtrs.push_back(&point);
  }
  std::unique_ptr<ForceFields::ForceField> field(
    ::DistGeom::constructAllInOneForceField(*earg.mmat, pointPtrs, earg.etkdgDetails, &earg.chiralCenters));
  if (torsionTermsEnabled) {
    addTorsionTermsWithFiniteDifferenceGradient(*field, earg, params);
  }
  field->initialize();
  return field;
}

std::vector<RDGeom::PointND> toPoints(const std::vector<double>& positions, const int atomStart, const int atomEnd) {
  std::vector<RDGeom::PointND> points;
  for (int atomIdx = atomStart; atomIdx < atomEnd; ++atomIdx) {
    auto& point = points.emplace_back(4);
    for (int dim = 0; dim < 4; ++dim) {
      point[dim] = positions[atomIdx * 4 + dim];
    }
  }
  return points;
}

//! RDKit's minimizeAllInOne from the given coordinates; returns the final all-in-one energy.
double minimizeAllInOneWithRdkit(EmbedArgs&                                  earg,
                                 std::vector<RDGeom::PointND>                points,
                                 const RDKit::DGeomHelpers::EmbedParameters& params,
                                 const unsigned int                          numHeavyAtoms) {
  // As RDKit's minimizeAllInOne, each phase runs only when the energy is above ERROR_TOL.
  constexpr double errorTolerance = 1e-5;
  auto             field          = allInOneReferenceForcefield(earg, points, params, false);
  if (field->calcEnergy() > errorTolerance) {
    field->minimize(numHeavyAtoms, params.optimizerForceTol);
  }
  addTorsionTermsWithFiniteDifferenceGradient(*field, earg, params);
  if (field->calcEnergy() > errorTolerance) {
    for (int cycle = 0; cycle < 8 && field->minimize(100, params.optimizerForceTol); ++cycle) {
    }
  }
  return field->calcEnergy();
}

using AllInOneStageParams = std::tuple<ETKDGOption, nvMolKit::PrecisionMode, nvMolKit::BfgsBackend>;

class ETKDGAllInOneStageTest : public ::testing::TestWithParam<AllInOneStageParams> {};

TEST_P(ETKDGAllInOneStageTest, MinimizedEnergiesMatchRdkitFromIdenticalCoordinates) {
  const auto [option, precision, backend] = GetParam();
  std::vector<std::unique_ptr<RDKit::ROMol>> loaded;
  getMols(getTestDataFolderPath() + "/MMFF94_dative.sdf", loaded, 40);
  std::vector<std::unique_ptr<RDKit::RWMol>> molOwners;
  std::vector<const RDKit::ROMol*>           mols;
  for (const auto& mol : loaded) {
    auto& owner = molOwners.emplace_back(std::make_unique<RDKit::RWMol>(*mol));
    owner->clearConformers();
    mols.push_back(owner.get());
  }

  auto params                    = getETKDGOption(option);
  params.useRandomCoords         = true;
  params.useLegacyImplementation = false;
  ETKDGContext           context;
  std::vector<EmbedArgs> eargs;
  // The reference setup helper draws from RDKit's global RNG.
  const auto             randomState = RDKit::getRandomGenerator();
  RDKit::getRandomGenerator(42);
  initTestComponentsCommon(mols, molOwners, context, eargs, params);
  RDKit::getRandomGenerator() = randomState;
  const auto startPositions   = context.systemHost.positions;
  const auto atomStarts       = context.systemHost.atomStarts;
  context.activeThisStage.setFromVector(std::vector<uint8_t>(mols.size(), 1));
  context.failedThisStage.setFromVector(std::vector<uint8_t>(mols.size(), 0));

  std::vector<double> nvMolKitPositions(startPositions.size());
  if (precision == nvMolKit::PrecisionMode::FULL) {
    nvMolKit::BfgsBatchMinimizer            minimizer(4, nvMolKit::DebugLevel::NONE, true, nullptr, backend);
    ETKDGAllInOneMinimizationStageT<double> stage(mols, eargs, params, context, minimizer);
    stage.execute(context);
  } else {
    nvMolKit::BfgsBatchMinimizerSingle     minimizer(4, nvMolKit::DebugLevel::NONE, true, nullptr, backend);
    ETKDGAllInOneMinimizationStageT<float> stage(mols, eargs, params, context, minimizer);
    stage.execute(context);
  }
  std::vector<uint8_t> failed(mols.size());
  context.systemDevice.positions.copyToHost(nvMolKitPositions);
  context.failedThisStage.copyToHost(failed);
  CHECK_CUDA_RETURN(cudaDeviceSynchronize());

  // BFGS trajectories on strained molecules are chaotic, so compare acceptance counts and, where both accept, the
  // minimized energies rather than coordinates. RDKit's BFGS sometimes reports convergence on a negligible step while
  // the energy is still high, so it may reject more than nvMolKit but should not reject fewer.
  int                 nvMolKitFailures = 0;
  int                 rdkitFailures    = 0;
  std::vector<double> gotEnergies, wantEnergies, startEnergies;
  for (std::size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    const int    atomStart  = atomStarts[molIdx];
    const int    atomEnd    = atomStarts[molIdx + 1];
    auto         start      = toPoints(startPositions, atomStart, atomEnd);
    auto         got        = toPoints(nvMolKitPositions, atomStart, atomEnd);
    const double want       = minimizeAllInOneWithRdkit(eargs[molIdx], start, params, mols[molIdx]->getNumHeavyAtoms());
    const bool   rdkitFails = !(want / (atomEnd - atomStart) < kMaxAllInOneEnergyPerAtom);
    nvMolKitFailures += failed[molIdx];
    rdkitFailures += rdkitFails;
    if (!failed[molIdx] && !rdkitFails) {
      startEnergies.push_back(allInOneReferenceForcefield(eargs[molIdx], start, params, true)->calcEnergy());
      gotEnergies.push_back(allInOneReferenceForcefield(eargs[molIdx], got, params, true)->calcEnergy());
      wantEnergies.push_back(want);
    }
  }
  EXPECT_LE(nvMolKitFailures, rdkitFailures + 2)
    << "nvMolKit rejected " << nvMolKitFailures << " and RDKit " << rdkitFailures << " of " << mols.size();
  const int maxOutliers = std::max(1, static_cast<int>(gotEnergies.size()) / 4);
  expectEnergyMatch(gotEnergies, wantEnergies, startEnergies, {0.05, maxOutliers, 0.01, true});
}

INSTANTIATE_TEST_SUITE_P(
  ETKDGOptions,
  ETKDGAllInOneStageTest,
  ::testing::Combine(::testing::Values(ETKDGOption::ETKDGv3, ETKDGOption::ETKDGv2, ETKDGOption::ETDG, ETKDGOption::KDG),
                     ::testing::Values(nvMolKit::PrecisionMode::FULL, nvMolKit::PrecisionMode::SINGLE),
                     ::testing::Values(nvMolKit::BfgsBackend::BATCHED, nvMolKit::BfgsBackend::PER_MOLECULE)),
  [](const ::testing::TestParamInfo<AllInOneStageParams>& info) {
    return getETKDGOptionName(std::get<0>(info.param)) +
           (std::get<1>(info.param) == nvMolKit::PrecisionMode::FULL ? "_Double" : "_Float") +
           (std::get<2>(info.param) == nvMolKit::BfgsBackend::BATCHED ? "_Batched" : "_PerMolecule");
  });

using AllInOneIndependenceParams = std::tuple<nvMolKit::PrecisionMode, nvMolKit::BfgsBackend>;

class ETKDGAllInOneIndependenceTest : public ::testing::TestWithParam<AllInOneIndependenceParams> {};

//! Runs the all-in-one stage on mols from coordinates drawn with a fixed RDKit seed. When lastFourthDim is set, the
//! last atom's fourth coordinate is replaced by it. Returns the starting and final coordinates.
std::pair<std::vector<double>, std::vector<double>> runAllInOneStage(const std::vector<std::string>& smiles,
                                                                     const nvMolKit::PrecisionMode   precision,
                                                                     const nvMolKit::BfgsBackend     backend,
                                                                     const std::optional<double> lastFourthDim = {}) {
  std::vector<std::unique_ptr<RDKit::RWMol>> molOwners;
  std::vector<const RDKit::ROMol*>           mols;
  for (const auto& entry : smiles) {
    std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(entry));
    RDKit::MolOps::addHs(*mol);
    mols.push_back(molOwners.emplace_back(std::move(mol)).get());
  }
  auto params                    = RDKit::DGeomHelpers::ETKDGv3;
  params.useRandomCoords         = true;
  params.useLegacyImplementation = false;
  ETKDGContext           context;
  std::vector<EmbedArgs> eargs;
  const auto             randomState = RDKit::getRandomGenerator();
  RDKit::getRandomGenerator(7);
  initTestComponentsCommon(mols, molOwners, context, eargs, params);
  RDKit::getRandomGenerator() = randomState;
  if (lastFourthDim.has_value()) {
    context.systemHost.positions.back() = *lastFourthDim;
    context.systemDevice.positions.setFromVector(context.systemHost.positions);
  }
  const auto start = context.systemHost.positions;
  context.activeThisStage.setFromVector(std::vector<uint8_t>(mols.size(), 1));
  context.failedThisStage.setFromVector(std::vector<uint8_t>(mols.size(), 0));
  if (precision == nvMolKit::PrecisionMode::FULL) {
    nvMolKit::BfgsBatchMinimizer            minimizer(4, nvMolKit::DebugLevel::NONE, true, nullptr, backend);
    ETKDGAllInOneMinimizationStageT<double> stage(mols, eargs, params, context, minimizer);
    stage.execute(context);
  } else {
    nvMolKit::BfgsBatchMinimizerSingle     minimizer(4, nvMolKit::DebugLevel::NONE, true, nullptr, backend);
    ETKDGAllInOneMinimizationStageT<float> stage(mols, eargs, params, context, minimizer);
    stage.execute(context);
  }
  std::vector<double> positions(context.systemDevice.positions.size());
  context.systemDevice.positions.copyToHost(positions);
  CHECK_CUDA_RETURN(cudaDeviceSynchronize());
  return {start, positions};
}

//! As in RDKit's per-molecule minimizeAllInOne, a molecule's result does not depend on its batchmates: a molecule that
//! converges early is not minimized again while a slower one finishes. A system whose energy is already below
//! ERROR_TOL is not minimized at all: a single atom whose only term, the fourth dimension, holds about 2e-8.
TEST_P(ETKDGAllInOneIndependenceTest, MoleculeResultDoesNotDependOnBatchmates) {
  const auto [precision, backend] = GetParam();
  const auto [aloneStart, alone]  = runAllInOneStage({"CCO"}, precision, backend);
  const auto [start, withOthers] = runAllInOneStage({"CCO", "CCCCCCCCCCCCCCCCCCCC", "[Na+]"}, precision, backend, 1e-4);
  ASSERT_TRUE(std::equal(aloneStart.begin(), aloneStart.end(), start.begin())) << "the runs must start alike";
  for (std::size_t coord = 0; coord < alone.size(); ++coord) {
    EXPECT_NEAR(withOthers[coord], alone[coord], 1e-6) << "coordinate " << coord;
  }
  // Single precision rounds the coordinates through float even when nothing is minimized.
  for (auto coord = start.size() - 4; coord < start.size(); ++coord) {
    const double unmoved = precision == nvMolKit::PrecisionMode::FULL ? start[coord] : static_cast<float>(start[coord]);
    EXPECT_EQ(withOthers[coord], unmoved) << "the single atom moved";
  }
}

INSTANTIATE_TEST_SUITE_P(
  Backends,
  ETKDGAllInOneIndependenceTest,
  ::testing::Combine(::testing::Values(nvMolKit::PrecisionMode::FULL, nvMolKit::PrecisionMode::SINGLE),
                     ::testing::Values(nvMolKit::BfgsBackend::BATCHED, nvMolKit::BfgsBackend::PER_MOLECULE)),
  [](const ::testing::TestParamInfo<AllInOneIndependenceParams>& info) {
    return std::string(std::get<0>(info.param) == nvMolKit::PrecisionMode::FULL ? "Double" : "Float") +
           (std::get<1>(info.param) == nvMolKit::BfgsBackend::BATCHED ? "_Batched" : "_PerMolecule");
  });

}  // namespace
#endif

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
