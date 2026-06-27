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

#include <DistGeom/DistGeomUtils.h>
#include <GraphMol/FileParsers/FileParsers.h>

#include <cmath>
#include <filesystem>
#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/etkdg_impl.h"
#include "src/etkdg_stage_etk_minimization.h"
#include "src/minimizer/bfgs_minimize.h"
#include "tests/test_utils.h"
constexpr int DIM = 4;
void          initTestComponentsCommon(const std::vector<RDKit::ROMol*>&         mols,
                                       ETKDGContext&                             context,
                                       std::vector<nvMolKit::detail::EmbedArgs>& eargsVec,
                                       RDKit::DGeomHelpers::EmbedParameters&     embedParam) {
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
    nvMolKit::DGeomHelpers::setupRDKitFFWithPos(mols[i],
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

void perturbConformer(RDKit::Conformer& conf, const float delta = 0.1, const int seed = 0) {
  std::mt19937                          gen(seed);  // Mersenne Twister engine
  std::uniform_real_distribution<float> dist(-delta, delta);
  for (unsigned int i = 0; i < conf.getNumAtoms(); ++i) {
    RDGeom::Point3D pos = conf.getAtomPos(i);
    pos.x += delta * dist(gen);
    pos.y += delta * dist(gen);
    pos.z += delta * dist(gen);
    conf.setAtomPos(i, pos);
  }
}

std::vector<double> getReferenceEnergy(const std::vector<const RDKit::ROMol*>& mols,
                                       const std::vector<EmbedArgs>&           eargs,
                                       bool                                    minimize,
                                       bool                                    minimizerTol      = 1e-3,
                                       double*                                 posOverride       = nullptr,
                                       std::vector<double>*                    posOut            = nullptr,
                                       bool                                    useBasicKnowledge = true) {
  std::vector<double> results;
  results.reserve(mols.size());
  int startIdx = 0;
  for (size_t i = 0; i < mols.size(); ++i) {
    const auto& mol = *mols[i];
    if (eargs[i].dim != DIM) {
      throw std::runtime_error("getReferenceEnergy only supports 4D coordinates");
    }
    RDGeom::POINT3D_VECT   nonConst_data;
    RDGeom::Point3DPtrVect vec;
    for (auto& pos : mol.getConformer().getPositions()) {
      nonConst_data.push_back(pos);
    }
    for (auto& pos : nonConst_data) {
      vec.push_back(&pos);
    }
    // Create a force field for the molecule
    std::unique_ptr<ForceFields::ForceField> ff;
    if (useBasicKnowledge) {
      // ETKDG or KDG - use full 3D force field
      ff.reset(DistGeom::construct3DForceField(*eargs[i].mmat, vec, eargs[i].etkdgDetails));
    } else {
      // ETDG - use plain 3D force field (no improper torsions)
      ff.reset(DistGeom::constructPlain3DForceField(*eargs[i].mmat, vec, eargs[i].etkdgDetails));
    }
    ff->initialize();
    if (minimize) {
      ff->minimize(300, minimizerTol);
    }
    double energy = posOverride == nullptr ? ff->calcEnergy() : ff->calcEnergy(posOverride + startIdx);
    startIdx += mol.getNumAtoms() * 3;
    results.push_back(energy);
    if (posOut != nullptr) {
      for (const auto& pos : ff->positions()) {
        posOut->push_back((*pos)[0]);
        posOut->push_back((*pos)[1]);
        posOut->push_back((*pos)[2]);
      }
    }
  }
  return results;
}

std::vector<int16_t> getPassFailHost(const std::vector<const RDKit::ROMol*>& mols,
                                     const std::vector<EmbedArgs>&           eargs,
                                     double*                                 posOverride       = nullptr,
                                     bool                                    useBasicKnowledge = true) {
  std::vector<int16_t> results;
  results.reserve(mols.size());
  int startIdx = 0;
  for (size_t i = 0; i < mols.size(); ++i) {
    const auto& mol = *mols[i];
    if (eargs[i].dim != DIM) {
      throw std::runtime_error("getReferenceEnergy only supports 4D coordinates");
    }
    RDGeom::POINT3D_VECT   nonConst_data;
    RDGeom::Point3DPtrVect vec;
    for (auto& pos : mol.getConformer().getPositions()) {
      nonConst_data.push_back(pos);
    }
    for (auto& pos : nonConst_data) {
      vec.push_back(&pos);
    }
    // Only do planarity check if useBasicKnowledge is true (ETKDG/KDG variants)
    if (useBasicKnowledge) {
      // Create a force field for the molecule
      const std::unique_ptr<ForceFields::ForceField> ff(
        DistGeom::construct3DImproperForceField(*eargs[i].mmat, vec, eargs[i].etkdgDetails));
      ff->initialize();
      const double energy    = posOverride == nullptr ? ff->calcEnergy() : ff->calcEnergy(posOverride + startIdx);
      const double tolerance = 0.7 * eargs[i].etkdgDetails.improperAtoms.size();
      results.push_back(energy > tolerance);
    } else {
      // ETDG variant - no planarity check, always pass
      results.push_back(0);
    }
    startIdx += mol.getNumAtoms() * 3;
  }
  return results;
}

std::vector<int16_t> getGPuPassFailHost(const std::vector<const RDKit::ROMol*>&    mols,
                                        const nvMolKit::AsyncDeviceVector<double>& positions,
                                        const std::vector<EmbedArgs>&              eargs,
                                        bool                                       useBasicKnowledge = true) {
  std::vector<double> hostPos(positions.size());
  positions.copyToHost(hostPos);
  cudaDeviceSynchronize();

  std::vector<double> hostPos3;
  // convert 4D to 3D
  hostPos3.reserve(hostPos.size() / DIM * 3);
  for (size_t i = 0; i < hostPos.size(); i += DIM) {
    hostPos3.push_back(hostPos[i]);      // x
    hostPos3.push_back(hostPos[i + 1]);  // y
    hostPos3.push_back(hostPos[i + 2]);  // z
  }
  // Use original mols as the reference positions for energy/force.
  return getPassFailHost(mols, eargs, hostPos3.data(), useBasicKnowledge);
}

std::vector<double> getGPUEnergy(const std::vector<const RDKit::ROMol*>&    mols,
                                 const nvMolKit::AsyncDeviceVector<double>& positions,
                                 const std::vector<EmbedArgs>&              eargs,
                                 bool                                       useBasicKnowledge = true) {
  std::vector<double> hostPos(positions.size());
  positions.copyToHost(hostPos);
  cudaDeviceSynchronize();

  std::vector<double> hostPos3;
  // convert 4D to 3D
  hostPos3.reserve(hostPos.size() / DIM * 3);
  for (size_t i = 0; i < hostPos.size(); i += DIM) {
    hostPos3.push_back(hostPos[i]);      // x
    hostPos3.push_back(hostPos[i + 1]);  // y
    hostPos3.push_back(hostPos[i + 2]);  // z
  }
  // Use original mols as the reference positions for energy/force.
  return getReferenceEnergy(mols, eargs, false, 0.001, hostPos3.data(), nullptr, useBasicKnowledge);
}

using ETKStageTestParam = std::tuple<ETKDGOption, nvMolKit::BfgsBackend>;

class ETKStageSingleMolTestFixture : public ::testing::TestWithParam<ETKStageTestParam> {
 public:
  ETKStageSingleMolTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

  void SetUp() override {
    // Load molecule
    const std::string mol2FilePath = testDataFolderPath_ + "/rdkit_smallmol_1.mol2";
    ASSERT_TRUE(std::filesystem::exists(mol2FilePath)) << "Could not find " << mol2FilePath;
    molPtr_ = std::unique_ptr<RDKit::RWMol>(RDKit::MolFileToMol(mol2FilePath, false));
    ASSERT_NE(molPtr_, nullptr);
    RDKit::MolOps::sanitizeMol(*molPtr_);
    perturbConformer(molPtr_->getConformer(), 0.5);

    // Initialize mols_ vector with the single molecule
    mols_.push_back(molPtr_.get());
  }

  void initTestComponents() { initTestComponentsCommon(mols_, context_, eargs_, embedParam_); }

 protected:
  std::string                              testDataFolderPath_;
  std::unique_ptr<RDKit::RWMol>            molPtr_;
  std::vector<RDKit::ROMol*>               mols_;
  ETKDGContext                             context_;
  std::vector<nvMolKit::detail::EmbedArgs> eargs_;
  RDKit::DGeomHelpers::EmbedParameters     embedParam_;
};

TEST_P(ETKStageSingleMolTestFixture, MinimizeCompare) {
  // Set up embed parameters from test parameter
  const auto [etkdgOption, backend] = GetParam();
  embedParam_                       = getETKDGOption(etkdgOption);
  embedParam_.useRandomCoords       = true;

  // Initialize test components after setting embedParam_
  initTestComponents();

  // Determine useBasicKnowledge from embedParam_
  const bool useBasicKnowledge = embedParam_.useBasicKnowledge;

  // Create minimizer for the test
  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(4, nvMolKit::DebugLevel::NONE, true, nullptr, backend);

  // Create FirstMinimizeStage
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  std::vector<const RDKit::ROMol*>         molsPtrs;
  molsPtrs.push_back(molPtr_.get());
  auto stage = std::make_unique<nvMolKit::detail::ETKMinimizationStage>(molsPtrs,
                                                                        eargs_,
                                                                        embedParam_,
                                                                        context_,
                                                                        bfgsMinimizer,
                                                                        nullptr);
  stages.push_back(std::move(stage));

  // Create and run driver
  nvMolKit::detail::ETKDGDriver driver(std::make_unique<ETKDGContext>(std::move(context_)), std::move(stages));
  std::vector<double>           origGpuEnergy =
    getGPUEnergy(molsPtrs, driver.context().systemDevice.positions, eargs_, useBasicKnowledge);

  driver.run(1);

  // Check other results first
  EXPECT_EQ(driver.numConfsFinished(), 1);
  EXPECT_EQ(driver.iterationsComplete(), 1);

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  const auto                          failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 1);                      // One stage
  EXPECT_THAT(failureCounts[0], testing::ElementsAre(0));  // FirstMinimizeStage

  auto completed = driver.completedConformers();
  EXPECT_THAT(completed, testing::ElementsAre(1));

  std::vector<double> refEnergies =
    getReferenceEnergy(molsPtrs, eargs_, true, embedParam_.optimizerForceTol, nullptr, nullptr, useBasicKnowledge);
  std::vector<double> gpuEnergies =
    getGPUEnergy(molsPtrs, driver.context().systemDevice.positions, eargs_, useBasicKnowledge);

  // Check that GPU minimization actually reduced the energy
  EXPECT_THAT(gpuEnergies, ::testing::Pointwise(testing::Lt(), origGpuEnergy));

  EXPECT_THAT(refEnergies, ::testing::Pointwise(testing::Ge(), gpuEnergies));
}

namespace {
std::vector<ETKStageTestParam> makeETKStageParams(const std::vector<ETKDGOption>& options) {
  std::vector<ETKStageTestParam> params;
  for (const auto option : options) {
    params.emplace_back(option, nvMolKit::BfgsBackend::BATCHED);
    params.emplace_back(option, nvMolKit::BfgsBackend::PER_MOLECULE);
  }
  return params;
}

std::string makeETKStageName(const ETKStageTestParam& param) {
  std::string name = getETKDGOptionName(std::get<0>(param));
  name += std::get<1>(param) == nvMolKit::BfgsBackend::BATCHED ? "_Batched" : "_PerMolecule";
  return name;
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(ETKDGVariants,
                         ETKStageSingleMolTestFixture,
                         ::testing::ValuesIn(makeETKStageParams({ETKDGOption::ETKDG,
                                                                 ETKDGOption::ETKDGv2,
                                                                 ETKDGOption::srETKDGv3,
                                                                 ETKDGOption::ETKDGv3,
                                                                 ETKDGOption::KDG,
                                                                 ETKDGOption::ETDG,
                                                                 ETKDGOption::DG})),
                         [](const ::testing::TestParamInfo<ETKStageTestParam>& info) {
                           return makeETKStageName(info.param);
                         });

class ETKStageMultiMolTestFixture : public ::testing::TestWithParam<ETKStageTestParam> {
 public:
  ETKStageMultiMolTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

  void SetUp() override {
    // Load molecule
    getMols(testDataFolderPath_ + "/MMFF94_dative.sdf", molsHolder, 20);

    for (auto& mol : molsHolder) {
      auto* molPtr = dynamic_cast<RWMol*>(mol.get());
      ASSERT_NE(molPtr, nullptr) << "Failed to cast to RWMol";
      RDKit::MolOps::sanitizeMol(*molPtr);
      mols_.push_back(mol.get());
      perturbConformer(mol->getConformer(), 0.5);
    }
  }

  void initTestComponents() { initTestComponentsCommon(mols_, context_, eargs_, embedParam_); }

 protected:
  std::string                                testDataFolderPath_;
  std::vector<std::unique_ptr<RDKit::ROMol>> molsHolder;
  std::vector<RDKit::ROMol*>                 mols_;
  ETKDGContext                               context_;
  std::vector<nvMolKit::detail::EmbedArgs>   eargs_;
  RDKit::DGeomHelpers::EmbedParameters       embedParam_;
};

TEST_P(ETKStageMultiMolTestFixture, MinimizeCompare) {
  // Set up embed parameters from test parameter
  const auto [etkdgOption, backend] = GetParam();
  embedParam_                       = getETKDGOption(etkdgOption);
  embedParam_.useRandomCoords       = true;

  // Initialize test components after setting embedParam_
  initTestComponents();

  // Determine useBasicKnowledge from embedParam_
  const bool useBasicKnowledge = embedParam_.useBasicKnowledge;

  // Create minimizer for the test
  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(4, nvMolKit::DebugLevel::NONE, true, nullptr, backend);

  // Create FirstMinimizeStage
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  std::vector<const RDKit::ROMol*>         molsPtrs;
  for (auto& molPtr : mols_) {
    molsPtrs.push_back(molPtr);
  }
  const int count = molsPtrs.size();

  auto stage = std::make_unique<nvMolKit::detail::ETKMinimizationStage>(molsPtrs,
                                                                        eargs_,
                                                                        embedParam_,
                                                                        context_,
                                                                        bfgsMinimizer,
                                                                        nullptr);
  stages.push_back(std::move(stage));

  // Create and run driver
  nvMolKit::detail::ETKDGDriver driver(std::make_unique<ETKDGContext>(std::move(context_)), std::move(stages));
  std::vector<double>           origGpuEnergy =
    getGPUEnergy(molsPtrs, driver.context().systemDevice.positions, eargs_, useBasicKnowledge);

  driver.run(1);

  // Check other results first
  ASSERT_EQ(driver.iterationsComplete(), 1);

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  const auto                          failureCounts = driver.getFailures(failuresScratch);
  ASSERT_EQ(failureCounts.size(), 1);  // One stage

  // Minimize the molecules on the CPU to compare results.
  std::vector<double> refPosMinimized;
  std::vector<double> refEnergies = getReferenceEnergy(molsPtrs,
                                                       eargs_,
                                                       true,
                                                       embedParam_.optimizerForceTol,
                                                       nullptr,
                                                       &refPosMinimized,
                                                       useBasicKnowledge);
  // Energies of GPU-minimized molecules.
  std::vector<double> gpuEnergies =
    getGPUEnergy(molsPtrs, driver.context().systemDevice.positions, eargs_, useBasicKnowledge);
  // Pass-fail based on planarity check. Gpu Pass fail is GPU-coordinates computed by RDKit
  std::vector<int16_t> gpuPassFail =
    getGPuPassFailHost(molsPtrs, driver.context().systemDevice.positions, eargs_, useBasicKnowledge);
  std::vector<int16_t> refPassFail = getPassFailHost(molsPtrs, eargs_, refPosMinimized.data(), useBasicKnowledge);

  // Check that the GPU pass/fail matches RDKit-calculated version on the same coordinates.
  EXPECT_THAT(failureCounts[0], ::testing::Pointwise(::testing::Eq(), gpuPassFail));
  // Check that we're within 5% of the reference minimization failure counts.
  const int totalFailsGpu = std::count(gpuPassFail.begin(), gpuPassFail.end(), 1);
  const int totalFailsRef = std::count(refPassFail.begin(), refPassFail.end(), 1);
  EXPECT_LE(totalFailsGpu, totalFailsRef * 1.05)
    << "GPU minimization failed more than 5% of reference failures: " << totalFailsGpu << " vs " << totalFailsRef;

  std::vector<int> higherThanIndices;
  for (int i = 0; i < count; ++i) {
    if (refEnergies[i] < gpuEnergies[i]) {
      higherThanIndices.push_back(i);
    }
  }
  // Make sure ~80% of cases did better than reference energy.
  EXPECT_LE(higherThanIndices.size(), count / 5) << "Too many exceptions";
  for (int idx : higherThanIndices) {
    // Check that we've shrunk the energy sufficiently, or that we're pretty close to the reference value.
    // Pass if we're within 10%, or if we've shrunk by at least 90%.
    if ((gpuEnergies[idx] - refEnergies[idx]) / refEnergies[idx] < .1) {
      continue;
    }
    EXPECT_LE(gpuEnergies[idx] / origGpuEnergy[idx], 0.1)
      << "Molecule " << idx << " energy did not shrink sufficiently: " << origGpuEnergy[idx] << " vs "
      << gpuEnergies[idx] << ", reference minimized: " << refEnergies[idx];
  }
}

// Instantiate parameterized tests for different ETKDG variants and backends
INSTANTIATE_TEST_SUITE_P(ETKDGVariants,
                         ETKStageMultiMolTestFixture,
                         ::testing::ValuesIn(makeETKStageParams({ETKDGOption::ETDG,
                                                                 ETKDGOption::ETKDG,
                                                                 ETKDGOption::ETKDGv2,
                                                                 ETKDGOption::srETKDGv3,
                                                                 ETKDGOption::ETKDGv3,
                                                                 ETKDGOption::KDG,
                                                                 ETKDGOption::DG})),
                         [](const ::testing::TestParamInfo<ETKStageTestParam>& info) {
                           return makeETKStageName(info.param);
                         });
