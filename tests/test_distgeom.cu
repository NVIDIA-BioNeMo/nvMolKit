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
#include <DistGeom/DistGeomUtils.h>
#include <ForceField/ForceField.h>
#include <gmock/gmock.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <random>

#include "rdkit_extensions/dist_geom_flattened_builder.h"
#include "src/embedder_utils.h"
#include "src/forcefields/dg_batched_forcefield.h"
#include "src/forcefields/dist_geom.h"
#include "src/forcefields/dist_geom_kernels.h"
#include "src/forcefields/ff_utils.h"
#include "src/forcefields/kernel_utils.cuh"
#include "src/utils/device_vector.h"
#include "tests/test_utils.h"

using namespace nvMolKit::DistGeom;
using nvMolKit::AsyncDeviceVector;

namespace {

double computeEnergyViaForcefield(nvMolKit::DistGeom::BatchedMolecularSystemHost& systemHost,
                                  const std::vector<int>&                         atomStartsHost,
                                  AsyncDeviceVector<double>&                      positionsDevice,
                                  const uint8_t*                                  activeSystemMask = nullptr) {
  nvMolKit::DGBatchedForcefield forcefield(systemHost, atomStartsHost, 1.0, 0.1);
  AsyncDeviceVector<double>     energyOutsDevice;
  energyOutsDevice.resize(atomStartsHost.size() - 1);
  energyOutsDevice.zero();
  CHECK_CUDA_RETURN(forcefield.computeEnergy(energyOutsDevice.data(), positionsDevice.data(), activeSystemMask));
  double energy = 0.0;
  CHECK_CUDA_RETURN(cudaMemcpy(&energy, energyOutsDevice.data(), sizeof(double), cudaMemcpyDeviceToHost));
  return energy;
}

std::vector<double> computeGradientViaForcefield(nvMolKit::DistGeom::BatchedMolecularSystemHost& systemHost,
                                                 const std::vector<int>&                         atomStartsHost,
                                                 AsyncDeviceVector<double>&                      positionsDevice,
                                                 const uint8_t* activeSystemMask = nullptr) {
  nvMolKit::DGBatchedForcefield forcefield(systemHost, atomStartsHost, 1.0, 0.1);
  AsyncDeviceVector<double>     gradDevice;
  gradDevice.resize(positionsDevice.size());
  gradDevice.zero();
  CHECK_CUDA_RETURN(forcefield.computeGradients(gradDevice.data(), positionsDevice.data(), activeSystemMask));
  std::vector<double> grad(positionsDevice.size(), 0.0);
  gradDevice.copyToHost(grad);
  cudaDeviceSynchronize();
  return grad;
}

}  // namespace

using ETKDGTestParams = std::tuple<ETKDGOption, int>;

class ETKDGFFGpuTestFixture : public ::testing::TestWithParam<ETKDGTestParams> {
 public:
  ETKDGFFGpuTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

  void SetUp() override {
    const std::string mol2FilePath = testDataFolderPath_ + "/rdkit_smallmol_1.mol2";
    ASSERT_TRUE(std::filesystem::exists(mol2FilePath)) << "Could not find " << mol2FilePath;
    mol_ = std::unique_ptr<RDKit::RWMol>(RDKit::MolFileToMol(mol2FilePath, false));
    ASSERT_NE(mol_, nullptr);
    RDKit::MolOps::sanitizeMol(*mol_);

    auto                        options = getETKDGOption(std::get<0>(GetParam()));
    nvMolKit::detail::EmbedArgs eargs;
    field_ = nvMolKit::DGeomHelpers::generateRDKitFF(*mol_, options, eargs, positions_);

    std::vector<double> posVec   = convertPositionsToVector(positions_, eargs.dim);
    auto                ffParams = constructForceFieldContribs(eargs.dim, *eargs.mmat, eargs.chiralCenters);

    addMoleculeToBatch(ffParams, posVec, systemHost_, eargs.dim, atomStartsHost_, positionsHost_);
    sendContribsAndIndicesToDevice(systemHost_, systemDevice_);
    sendContextToDevice(positionsHost_, positionsDevice_, atomStartsHost_, atomStartsDevice_);
    setupDeviceBuffers(systemHost_, systemDevice_, positionsHost_, atomStartsHost_.size() - 1);
  }

 protected:
  std::string                                 testDataFolderPath_;
  std::unique_ptr<RDKit::RWMol>               mol_;
  std::unique_ptr<ForceFields::ForceField>    field_;
  std::vector<std::unique_ptr<RDGeom::Point>> positions_;
  BatchedMolecularSystemHost                  systemHost_;
  BatchedMolecularDeviceBuffers               systemDevice_;
  std::vector<int>                            atomStartsHost_ = {0};
  AsyncDeviceVector<int>                      atomStartsDevice_;
  std::vector<double>                         positionsHost_;
  AsyncDeviceVector<double>                   positionsDevice_;
};

INSTANTIATE_TEST_SUITE_P(ETKDGOptionsAndActiveStages,
                         ETKDGFFGpuTestFixture,
                         ::testing::Combine(::testing::Values(ETKDGOption::ETKDGv3,
                                                              ETKDGOption::ETKDGv2,
                                                              ETKDGOption::ETKDG,
                                                              ETKDGOption::KDG),
                                            ::testing::Values(-1, 0, 1)  // active stages
                                            ),
                         [](const ::testing::TestParamInfo<ETKDGTestParams>& info) {
                           std::string name = getETKDGOptionName(std::get<0>(info.param));
                           switch (std::get<1>(info.param)) {
                             case -1:
                               name += "_NullActiveStage";
                               break;
                             case 0:
                               name += "_Inactive";
                               break;
                             case 1:
                               name += "_Active";
                               break;
                           }
                           return name;
                         });

// Test energy calculation with both ETKDG options and active stage status
TEST_P(ETKDGFFGpuTestFixture, CombinedEnergies) {
  double gotEnergy;

  if (std::get<1>(GetParam()) == -1) {
    double wantEnergy = field_->calcEnergy();
    gotEnergy         = computeEnergyViaForcefield(systemHost_, atomStartsHost_, positionsDevice_);
    EXPECT_NEAR(gotEnergy, wantEnergy, 1e-6);
  } else {
    // Test with active stage parameters
    std::vector<uint8_t>                 h_activeThisStage(1, std::get<1>(GetParam()));
    nvMolKit::AsyncDeviceVector<uint8_t> d_activeThisStage;
    d_activeThisStage.setFromVector(h_activeThisStage);

    gotEnergy = computeEnergyViaForcefield(systemHost_, atomStartsHost_, positionsDevice_, d_activeThisStage.data());

    if (std::get<1>(GetParam()) == 1) {
      double wantEnergy = field_->calcEnergy();
      EXPECT_NEAR(gotEnergy, wantEnergy, 1e-6);
    } else {
      EXPECT_NEAR(gotEnergy, 0.0, 1e-6);
    }
  }
}

// Test gradient calculation with both ETKDG options and active stage status
TEST_P(ETKDGFFGpuTestFixture, CombinedGradients) {
  if (std::get<1>(GetParam()) == -1) {
    std::vector<double> wantGradients(field_->dimension() * field_->positions().size(), 0.0);
    field_->calcGrad(wantGradients.data());
    std::vector<double> gotGrad = computeGradientViaForcefield(systemHost_, atomStartsHost_, positionsDevice_);
    EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(1e-4), wantGradients));
  } else {
    // Test with active stage parameters
    std::vector<uint8_t>                 h_activeThisStage(1, std::get<1>(GetParam()));
    nvMolKit::AsyncDeviceVector<uint8_t> d_activeThisStage;
    d_activeThisStage.setFromVector(h_activeThisStage);

    std::vector<double> gotGrad =
      computeGradientViaForcefield(systemHost_, atomStartsHost_, positionsDevice_, d_activeThisStage.data());

    if (std::get<1>(GetParam()) == 1) {
      std::vector<double> wantGradients(field_->dimension() * field_->positions().size(), 0.0);
      field_->calcGrad(wantGradients.data());
      EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(1e-4), wantGradients));
    } else {
      // For inactive molecules, expect zero gradients
      std::vector<double> zeroGradients(field_->dimension() * field_->positions().size(), 0.0);
      EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(1e-4), zeroGradients));
    }
  }
}

class ETKDGFFGpuEdgeCases : public ::testing::TestWithParam<std::string> {
 public:
  void SetUp() override {
    // Get the SMILES string from the test parameter
    std::string smiles = GetParam();
    mol_               = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(smiles));

    ASSERT_NE(mol_, nullptr);
    RDKit::MolOps::sanitizeMol(*mol_);

    auto                        options = RDKit::DGeomHelpers::ETKDGv3;
    nvMolKit::detail::EmbedArgs eargs;
    field_ = nvMolKit::DGeomHelpers::generateRDKitFF(*mol_, options, eargs, positions_);

    std::vector<double> posVec   = convertPositionsToVector(positions_, eargs.dim);
    auto                ffParams = constructForceFieldContribs(eargs.dim, *eargs.mmat, eargs.chiralCenters);
    addMoleculeToBatch(ffParams, posVec, systemHost_, eargs.dim, atomStartsHost_, positionsHost_);
    sendContribsAndIndicesToDevice(systemHost_, systemDevice_);
    sendContextToDevice(positionsHost_, positionsDevice_, atomStartsHost_, atomStartsDevice_);
    setupDeviceBuffers(systemHost_, systemDevice_, positionsHost_, atomStartsHost_.size() - 1);
  }

 protected:
  std::unique_ptr<ForceFields::ForceField>    field_;
  std::unique_ptr<RDKit::RWMol>               mol_;
  std::vector<std::unique_ptr<RDGeom::Point>> positions_;
  BatchedMolecularSystemHost                  systemHost_;
  BatchedMolecularDeviceBuffers               systemDevice_;
  std::vector<int>                            atomStartsHost_ = {0};
  AsyncDeviceVector<int>                      atomStartsDevice_;
  std::vector<double>                         positionsHost_;
  AsyncDeviceVector<double>                   positionsDevice_;
};

INSTANTIATE_TEST_SUITE_P(ETKDGFFOneTwoAtoms,
                         ETKDGFFGpuEdgeCases,
                         ::testing::Values("C",       // 1 atom: methane
                                           "O",       // 1 atom: oxygen
                                           "N",       // 1 atom: nitrogen
                                           "[NH4+]",  // 1 atom: ammonium cation
                                           "CC",      // 2 atoms: ethane
                                           "CO",      // 2 atoms: methanol
                                           "CN",      // 2 atoms: methylamine
                                           "C=O",     // 2 atoms: formaldehyde
                                           "C#N",     // 2 atoms: hydrogen cyanide
                                           "C[O-]"    // 2 atoms: methoxide anion
                                           ));

// Test energy calculation
TEST_P(ETKDGFFGpuEdgeCases, CombinedEnergies) {
  double wantEnergy = field_->calcEnergy();
  double gotEnergy  = computeEnergyViaForcefield(systemHost_, atomStartsHost_, positionsDevice_);
  EXPECT_NEAR(gotEnergy, wantEnergy, 1e-6);
}

// Test gradient calculation
TEST_P(ETKDGFFGpuEdgeCases, CombinedGradients) {
  std::vector<double> wantGradients(field_->dimension() * field_->positions().size(), 0.0);
  field_->calcGrad(wantGradients.data());

  std::vector<double> gotGrad = computeGradientViaForcefield(systemHost_, atomStartsHost_, positionsDevice_);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(1e-4), wantGradients));
}

using BatchedTestParams = std::tuple<ETKDGOption, std::string>;

class ETKDGFFGPUBatchTestFixture : public ::testing::TestWithParam<BatchedTestParams> {
 public:
  void SetUp() override {
    std::string filePath = getTestDataFolderPath() + "/" + std::get<1>(GetParam());
    numMols_             = 100;  // or whatever number you want to use
    getMols(filePath, mols_, numMols_);
    etkdgOption_ = std::get<0>(GetParam());

    activeStages_.resize(numMols_);

    // Randomly initialize activeThisStage with 0s and 1s
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    for (int i = 0; i < numMols_; ++i) {
      activeStages_[i] = dis(gen);
    }
  }

 protected:
  std::vector<std::unique_ptr<RDKit::ROMol>> mols_;
  ETKDGOption                                etkdgOption_;
  int                                        numMols_;
  std::vector<uint8_t>                       activeStages_;
};

// TODO: Consolidate the validation failure struct and batch unit test helper functions below with MMFF
// once term-wise unit-tests for ETKDG force fields are enabled.
// Currently, this is not supported as RDKit lacks term-wise force field contribution addition
// function, and the relevant scripts and header files differ between the 2023* and 2024* versions.
struct ValidationFailures {
  std::string name;
  double      delta = -1.0;
  std::string exception;
};

void checkFailures(const std::vector<ValidationFailures>  failures,
                   const std::vector<ValidationFailures>& gradFailures,
                   const int                              numMols) {
  if (!failures.empty()) {
    std::cerr << "Energy Failed on " << failures.size() << " out of " << numMols << " molecules\n";
    std::cerr << std::setw(20) << std::left << "Molecule"
              << "DeltaEnergy ";
    std::cerr << "\n";
    for (const auto& failure : failures) {
      if (failure.exception.size()) {
        std::cerr << failure.name << " " << failure.exception << "\n";
      } else {
        std::cerr << std::setw(20) << std::left << failure.name << " " << std::fixed << std::setprecision(4)
                  << std::setw(10) << failure.delta << " ";
        std::cerr << "\n";
      }
    }
    FAIL();
  }
  if (!gradFailures.empty()) {
    std::cerr << "Gradient Failed on " << gradFailures.size() << " out of " << numMols << " molecules\n";
    std::cerr << std::setw(20) << std::left << "Molecule"
              << "Max Delta grad";
    std::cerr << "\n";
    for (const auto& failure : gradFailures) {
      if (failure.exception.size()) {
        std::cerr << failure.name << " " << failure.exception << "\n";
      } else {
        std::cerr << std::setw(20) << std::left << failure.name << " " << std::fixed << std::setprecision(4)
                  << std::setw(10) << failure.delta << " ";
        std::cerr << "\n";
      }
    }
    FAIL();
  }
}

void runTestInBatch(const std::vector<std::unique_ptr<RDKit::ROMol>>& mols,
                    ETKDGOption                                       etkdgOption,
                    double                                            energyTolerance   = 1e-4,
                    double                                            gradTolerance     = 1e-4,
                    const uint8_t*                                    h_activeThisStage = nullptr,
                    const uint8_t*                                    d_activeThisStage = nullptr) {
  const int                        numMols = mols.size();
  std::vector<ValidationFailures>  failures;
  std::vector<ValidationFailures>  gradFailures;
  std::vector<std::vector<double>> wantGrads;
  std::vector<double>              wantEnergies;

  BatchedMolecularSystemHost    systemHost;
  BatchedMolecularDeviceBuffers systemDevice;
  std::vector<int>              atomStartsHost = {0};
  AsyncDeviceVector<int>        atomStartsDevice;
  std::vector<double>           positionsHost;
  AsyncDeviceVector<double>     positionsDevice;

  for (const auto& mol : mols) {
    nvMolKit::detail::EmbedArgs                 eargs;
    std::vector<std::unique_ptr<RDGeom::Point>> positions;
    std::unique_ptr<ForceFields::ForceField>    field;
    auto                                        params = getETKDGOption(etkdgOption);

    nvMolKit::DGeomHelpers::setupRDKitFFWithPos(mol.get(), params, field, eargs, positions);

    // Check if this molecule is active
    bool isActive = true;
    if (h_activeThisStage) {
      isActive = h_activeThisStage[wantEnergies.size()] == 1;
    }
    if (isActive) {
      wantEnergies.push_back(field->calcEnergy());
      std::vector<double> wantGrad(field->dimension() * mol->getNumAtoms(), 0.0);
      field->calcGrad(wantGrad.data());
      wantGrads.push_back(wantGrad);
    } else {
      // For inactive molecules, add zero energy and zero gradient
      wantEnergies.push_back(0.0);
      wantGrads.push_back(std::vector<double>(field->dimension() * mol->getNumAtoms(), 0.0));
    }

    auto ffParams = constructForceFieldContribs(eargs.dim, *eargs.mmat, eargs.chiralCenters);
    addMoleculeToBatch(ffParams, eargs.posVec, systemHost, eargs.dim, atomStartsHost, positionsHost);
  }

  sendContextToDevice(positionsHost, positionsDevice, atomStartsHost, atomStartsDevice);
  nvMolKit::DGBatchedForcefield forcefield(systemHost, atomStartsHost, 1.0, 0.1);
  AsyncDeviceVector<double>     energyOutsDevice;
  AsyncDeviceVector<double>     gradDevice;
  energyOutsDevice.resize(atomStartsHost.size() - 1);
  energyOutsDevice.zero();
  gradDevice.resize(positionsHost.size());
  gradDevice.zero();

  CHECK_CUDA_RETURN(forcefield.computeEnergy(energyOutsDevice.data(), positionsDevice.data(), d_activeThisStage));
  CHECK_CUDA_RETURN(forcefield.computeGradients(gradDevice.data(), positionsDevice.data(), d_activeThisStage));
  std::vector<double> gotEnergies(energyOutsDevice.size(), 0.0);
  energyOutsDevice.copyToHost(gotEnergies);
  std::vector<double> gotGradFlat(positionsHost.size(), 0.0);
  gradDevice.copyToHost(gotGradFlat);
  std::vector<std::vector<double>> gotGradFormatted;
  for (int i = 0; i < numMols; i++) {
    const int posStart = atomStartsHost[i] * systemHost.dimension;
    const int posEnd   = atomStartsHost[i + 1] * systemHost.dimension;

    gotGradFormatted.push_back(std::vector<double>(gotGradFlat.begin() + posStart, gotGradFlat.begin() + posEnd));
  }

  // Check energies first.
  for (int i = 0; i < numMols; i++) {
    const double gotEnergy  = gotEnergies[i];
    const double wantEnergy = wantEnergies[i];

    if (std::abs(gotEnergy - wantEnergy) > energyTolerance) {
      auto& failure = failures.emplace_back();
      failure.name  = mols[i]->getProp<std::string>("_Name");
      failure.delta = gotEnergy - wantEnergy;
    }
  }

  // Check gradients
  for (int i = 0; i < numMols; i++) {
    const std::vector<double>& gotGrad  = gotGradFormatted[i];
    const std::vector<double>& wantGrad = wantGrads[i];

    for (size_t j = 0; j < wantGrad.size(); j++) {
      if (std::abs(wantGrad[j] - gotGrad[j]) > gradTolerance) {
        auto& failure = gradFailures.emplace_back();
        failure.name  = mols[i]->getProp<std::string>("_Name");
        failure.delta = std::abs(wantGrad[j] - gotGrad[j]);
        break;
      }
    }
  }
  checkFailures(failures, gradFailures, numMols);
}

void runTestInSerial(const std::vector<std::unique_ptr<RDKit::ROMol>>& mols,
                     ETKDGOption                                       etkdgOption,
                     double                                            energyTolerance = 1e-4,
                     double                                            gradTolerance   = 1e-4) {
  const int                        numMols = mols.size();
  std::vector<ValidationFailures>  failures;
  std::vector<ValidationFailures>  gradFailures;
  std::vector<std::vector<double>> wantGrads;
  std::vector<double>              wantEnergies;

  for (const auto& mol : mols) {
    nvMolKit::detail::EmbedArgs                 eargs;
    std::vector<std::unique_ptr<RDGeom::Point>> positions;
    std::unique_ptr<ForceFields::ForceField>    field;
    auto                                        params = getETKDGOption(etkdgOption);

    nvMolKit::DGeomHelpers::setupRDKitFFWithPos(mol.get(), params, field, eargs, positions);
    auto ffParams = constructForceFieldContribs(eargs.dim, *eargs.mmat, eargs.chiralCenters);

    BatchedMolecularSystemHost systemHost;
    std::vector<int>           atomStartsHost = {0};
    AsyncDeviceVector<int>     atomStartsDevice;
    std::vector<double>        positionsHost;
    AsyncDeviceVector<double>  positionsDevice;
    addMoleculeToBatch(ffParams, eargs.posVec, systemHost, eargs.dim, atomStartsHost, positionsHost);
    sendContextToDevice(positionsHost, positionsDevice, atomStartsHost, atomStartsDevice);
    nvMolKit::DGBatchedForcefield forcefield(systemHost, atomStartsHost, 1.0, 0.1);

    // Check energies first.
    double      wantEnergy = field->calcEnergy();
    double      gotEnergy  = 0.0;
    std::string exceptionStr;
    try {
      AsyncDeviceVector<double> energyOutsDevice;
      energyOutsDevice.resize(1);
      energyOutsDevice.zero();
      CHECK_CUDA_RETURN(forcefield.computeEnergy(energyOutsDevice.data(), positionsDevice.data()));
      CHECK_CUDA_RETURN(cudaMemcpy(&gotEnergy, energyOutsDevice.data(), sizeof(double), cudaMemcpyDeviceToHost));
    } catch (const std::runtime_error& e) {
      exceptionStr = e.what();
    };
    if (exceptionStr.size() || std::abs(gotEnergy - wantEnergy) > energyTolerance) {
      auto& failure = failures.emplace_back();
      failure.name  = mol->getProp<std::string>("_Name");
      if (exceptionStr.size()) {
        failure.exception = exceptionStr;
      }
      failure.delta = gotEnergy - wantEnergy;
    }

    // Check gradients
    std::vector<double> wantGrad(field->dimension() * mol->getNumAtoms(), 0.0);
    field->calcGrad(wantGrad.data());
    AsyncDeviceVector<double> gradDevice;
    gradDevice.resize(positionsHost.size());
    gradDevice.zero();
    CHECK_CUDA_RETURN(forcefield.computeGradients(gradDevice.data(), positionsDevice.data()));
    std::vector<double> gotGrad(positionsHost.size(), 0.0);
    gradDevice.copyToHost(gotGrad);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < wantGrad.size(); i++) {
      if (std::abs(wantGrad[i] - gotGrad[i]) > gradTolerance) {
        auto& failure = gradFailures.emplace_back();
        failure.name  = mol->getProp<std::string>("_Name");
        failure.delta = std::abs(wantGrad[i] - gotGrad[i]);
        break;
      }
    }
  }
  checkFailures(failures, gradFailures, numMols);
}

INSTANTIATE_TEST_SUITE_P(ETKDGFFGPUBatchTests,
                         ETKDGFFGPUBatchTestFixture,
                         ::testing::Combine(::testing::Values(ETKDGOption::ETKDGv3,
                                                              ETKDGOption::ETKDGv2,
                                                              ETKDGOption::ETKDG,
                                                              ETKDGOption::ETDG,
                                                              ETKDGOption::KDG),
                                            ::testing::Values("MMFF94_dative.sdf", "MMFF94_hypervalent.sdf")),
                         [](const ::testing::TestParamInfo<BatchedTestParams>& info) {
                           return getETKDGOptionName(std::get<0>(info.param)) + "_" +
                                  std::get<1>(info.param).substr(0, std::get<1>(info.param).find('.'));
                         });

TEST_P(ETKDGFFGPUBatchTestFixture, SerialTest) {
  runTestInSerial(mols_, etkdgOption_);
}

TEST_P(ETKDGFFGPUBatchTestFixture, BatchTest) {
  runTestInBatch(mols_, etkdgOption_);
}

TEST_P(ETKDGFFGPUBatchTestFixture, BatchTestWithActiveStage) {
  nvMolKit::AsyncDeviceVector<uint8_t> d_activeThisStage;
  d_activeThisStage.setFromVector(activeStages_);
  runTestInBatch(mols_, etkdgOption_, 1e-4, 1e-4, activeStages_.data(), d_activeThisStage.data());
}

struct EdgeTestCase {
  std::vector<std::string> molecules;
  std::vector<uint8_t>     activeStages;
};

class ETKDGFFGpuBatchEdgeCases : public ::testing::TestWithParam<EdgeTestCase> {
 public:
  void SetUp() override {
    // Get the molecules from the test parameter
    for (const auto& smiles : GetParam().molecules) {
      mols_.push_back(std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(smiles)));
    }
    // Store the active stages configuration
    activeStages_ = GetParam().activeStages;
  }

 protected:
  std::vector<std::unique_ptr<RDKit::ROMol>> mols_;
  std::vector<uint8_t>                       activeStages_;
};

INSTANTIATE_TEST_SUITE_P(ETKDGFFBatchOneTwoAtoms,
                         ETKDGFFGpuBatchEdgeCases,
                         ::testing::Values(
                           // One-atom molecules
                           EdgeTestCase{
                             {"C", "O"},
                             {  1,   1}
},                         // Both active
                           EdgeTestCase{{"C", "N"}, {0, 1}},  // First inactive
                           EdgeTestCase{{"O", "C"}, {1, 0}},  // Second inactive
                           EdgeTestCase{{"N", "O"}, {0, 0}},  // Both inactive

                           // One atom + Two atoms
                           EdgeTestCase{{"C", "CC"}, {1, 1}},  // Both active
                           EdgeTestCase{{"O", "CO"}, {0, 1}},  // First inactive
                           EdgeTestCase{{"N", "CN"}, {1, 0}},  // Second inactive
                           EdgeTestCase{{"C", "CN"}, {0, 0}},  // Both inactive

                           // Two atoms + One atom
                           EdgeTestCase{{"CC", "O"}, {1, 1}},  // Both active
                           EdgeTestCase{{"CO", "N"}, {0, 1}},  // First inactive
                           EdgeTestCase{{"CN", "C"}, {1, 0}},  // Second inactive
                           EdgeTestCase{{"CC", "N"}, {0, 0}},  // Both inactive

                           // Two-atom molecules
                           EdgeTestCase{{"CC", "CO"}, {1, 1}},  // Both active
                           EdgeTestCase{{"CN", "CO"}, {0, 1}},  // First inactive
                           EdgeTestCase{{"CO", "CN"}, {1, 0}},  // Second inactive
                           EdgeTestCase{{"CC", "CN"}, {0, 0}}   // Both inactive
                           ));

TEST_P(ETKDGFFGpuBatchEdgeCases, SerialTest) {
  runTestInSerial(mols_, ETKDGOption::ETKDGv3, 1e-6, 1e-4);
}

TEST_P(ETKDGFFGpuBatchEdgeCases, BatchTest) {
  runTestInBatch(mols_, ETKDGOption::ETKDGv3, 1e-6, 1e-4);
}

TEST_P(ETKDGFFGpuBatchEdgeCases, BatchTestWithActiveStage) {
  nvMolKit::AsyncDeviceVector<uint8_t> d_activeThisStage;
  d_activeThisStage.setFromVector(activeStages_);

  runTestInBatch(mols_, ETKDGOption::ETKDGv3, 1e-6, 1e-4, activeStages_.data(), d_activeThisStage.data());
}
