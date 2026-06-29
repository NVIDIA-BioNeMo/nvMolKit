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

#include <ForceField/AngleConstraints.h>
#include <ForceField/DistanceConstraints.h>
#include <ForceField/ForceField.h>
#include <ForceField/MMFF/PositionConstraint.h>
#include <ForceField/MMFF/TorsionConstraint.h>
#include <gmock/gmock.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/ForceFieldHelpers/UFF/Builder.h>
#include <GraphMol/ForceFieldHelpers/UFF/UFF.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "rdkit_extensions/uff_flattened_builder.h"
#include "src/forcefields/ff_utils.h"
#include "src/forcefields/forcefield_constraints.h"
#include "src/forcefields/uff.h"
#include "src/forcefields/uff_batched_forcefield.h"
#include "src/minimizer/bfgs_minimize.h"
#include "src/minimizer/uff_minimize.h"
#include "src/utils/device_vector.h"
#include "tests/test_utils.h"

using namespace nvMolKit::UFF;
using nvMolKit::AsyncDeviceVector;

namespace {

constexpr double kEnergyTol         = 1.0e-6;
constexpr double kGradTol           = 2.0e-4;
constexpr double kMinimizeEnergyTol = 5.0e-3;

std::vector<double> positionsFromMol(const RDKit::ROMol& mol, const int confId = -1) {
  std::vector<double> positions;
  nvMolKit::confPosToVect(mol, positions, confId);
  return positions;
}

double computeEnergyViaForcefield(BatchedMolecularSystemHost& systemHost, AsyncDeviceVector<double>& positionsDevice) {
  nvMolKit::UFFBatchedForcefield forcefield(systemHost);
  AsyncDeviceVector<double>      energyOuts;
  energyOuts.resize(1);
  energyOuts.zero();
  CHECK_CUDA_RETURN(forcefield.computeEnergy(energyOuts.data(), positionsDevice.data()));
  double energy = 0.0;
  CHECK_CUDA_RETURN(cudaMemcpy(&energy, energyOuts.data(), sizeof(double), cudaMemcpyDeviceToHost));
  return energy;
}

std::vector<double> computeGradientViaForcefield(BatchedMolecularSystemHost& systemHost,
                                                 AsyncDeviceVector<double>&  positionsDevice) {
  nvMolKit::UFFBatchedForcefield forcefield(systemHost);
  AsyncDeviceVector<double>      gradDevice;
  gradDevice.resize(positionsDevice.size());
  gradDevice.zero();
  CHECK_CUDA_RETURN(forcefield.computeGradients(gradDevice.data(), positionsDevice.data()));
  std::vector<double> grad(positionsDevice.size(), 0.0);
  gradDevice.copyToHost(grad);
  cudaDeviceSynchronize();
  return grad;
}

std::unique_ptr<ForceFields::ForceField> buildReferenceForceField(RDKit::ROMol& mol, const int confId = -1) {
  auto ff = std::unique_ptr<ForceFields::ForceField>(RDKit::UFF::constructForceField(mol, 100.0, confId, true));
  ff->initialize();
  return ff;
}

std::unique_ptr<RDKit::ROMol> buildEmbeddedEdgeCaseMol(const std::string& smiles) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
  EXPECT_NE(mol, nullptr);
  auto withHs = std::unique_ptr<RDKit::ROMol>(RDKit::MolOps::addHs(*mol));
  RDKit::DGeomHelpers::EmbedMolecule(*withHs);
  return withHs;
}

}  // namespace

class UFFGpuTestFixture : public ::testing::Test {
 public:
  UFFGpuTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

  void SetUp() override {
    const std::string mol2FilePath = testDataFolderPath_ + "/rdkit_smallmol_1.mol2";
    ASSERT_TRUE(std::filesystem::exists(mol2FilePath)) << "Could not find " << mol2FilePath;
    mol_ = std::unique_ptr<RDKit::RWMol>(RDKit::MolFileToMol(mol2FilePath, false));
    ASSERT_NE(mol_, nullptr);
    RDKit::MolOps::sanitizeMol(*mol_);

    positions_ = positionsFromMol(*mol_);
    contribs_  = constructForcefieldContribs(*mol_);
    addMoleculeToBatch(contribs_, positions_, systemHost_);
    sendContribsAndIndicesToDevice(systemHost_, systemDevice_);
    systemDevice_.positions.setFromVector(systemHost_.positions);
    allocateIntermediateBuffers(systemHost_, systemDevice_);
    systemDevice_.grad.resize(systemHost_.positions.size());
    systemDevice_.grad.zero();
  }

 protected:
  std::string                   testDataFolderPath_;
  std::unique_ptr<RDKit::RWMol> mol_;
  std::vector<double>           positions_;
  EnergyForceContribsHost       contribs_;
  BatchedMolecularSystemHost    systemHost_;
  BatchedMolecularDeviceBuffers systemDevice_;
};

TEST_F(UFFGpuTestFixture, FlattenedBuilderPopulatesAllTerms) {
  EXPECT_GT(contribs_.bondTerms.idx1.size(), 0);
  EXPECT_GT(contribs_.angleTerms.idx1.size(), 0);
  EXPECT_GT(contribs_.torsionTerms.idx1.size(), 0);
  EXPECT_GT(contribs_.inversionTerms.idx1.size(), 0);
  EXPECT_GT(contribs_.vdwTerms.idx1.size(), 0);

  EXPECT_EQ(contribs_.bondTerms.idx1.size(), contribs_.bondTerms.forceConstant.size());
  EXPECT_EQ(contribs_.angleTerms.idx1.size(), contribs_.angleTerms.C2.size());
  EXPECT_EQ(contribs_.torsionTerms.idx1.size(), contribs_.torsionTerms.cosTerm.size());
  EXPECT_EQ(contribs_.inversionTerms.idx1.size(), contribs_.inversionTerms.C2.size());
  EXPECT_EQ(contribs_.vdwTerms.idx1.size(), contribs_.vdwTerms.threshold.size());
}

TEST_F(UFFGpuTestFixture, CombinedEnergyMatchesRDKit) {
  auto         referenceFF = buildReferenceForceField(*mol_);
  const double wantEnergy  = referenceFF->calcEnergy(positions_.data());
  const double gotEnergy   = computeEnergyViaForcefield(systemHost_, systemDevice_.positions);
  EXPECT_NEAR(gotEnergy, wantEnergy, kEnergyTol);
}

TEST_F(UFFGpuTestFixture, CombinedGradientMatchesRDKit) {
  auto                referenceFF = buildReferenceForceField(*mol_);
  std::vector<double> wantGrad(referenceFF->dimension() * referenceFF->positions().size(), 0.0);
  referenceFF->calcGrad(positions_.data(), wantGrad.data());
  const std::vector<double> gotGrad = computeGradientViaForcefield(systemHost_, systemDevice_.positions);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::DoubleNear(kGradTol), wantGrad));
}

TEST_F(UFFGpuTestFixture, BlockPerMolMatchesRDKit) {
  auto referenceFF = buildReferenceForceField(*mol_);
  systemDevice_.energyOuts.zero();
  CHECK_CUDA_RETURN(computeEnergyBlockPerMol(systemDevice_));
  double gotEnergy = 0.0;
  CHECK_CUDA_RETURN(cudaMemcpy(&gotEnergy, systemDevice_.energyOuts.data(), sizeof(double), cudaMemcpyDeviceToHost));

  std::vector<double> wantGrad(referenceFF->dimension() * referenceFF->positions().size(), 0.0);
  referenceFF->calcGrad(positions_.data(), wantGrad.data());
  systemDevice_.grad.zero();
  CHECK_CUDA_RETURN(computeGradBlockPerMol(systemDevice_));
  std::vector<double> gotGrad(systemDevice_.positions.size(), 0.0);
  systemDevice_.grad.copyToHost(gotGrad);
  cudaDeviceSynchronize();

  EXPECT_NEAR(gotEnergy, referenceFF->calcEnergy(positions_.data()), kEnergyTol);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::DoubleNear(kGradTol), wantGrad));
}

class UFFGpuEdgeCases : public ::testing::TestWithParam<std::string> {};

TEST_P(UFFGpuEdgeCases, CombinedEnergyAndGradient) {
  auto mol = buildEmbeddedEdgeCaseMol(GetParam());
  ASSERT_NE(mol, nullptr);

  auto                       positions = positionsFromMol(*mol);
  const auto                 contribs  = constructForcefieldContribs(*mol);
  BatchedMolecularSystemHost host;
  addMoleculeToBatch(contribs, positions, host);

  AsyncDeviceVector<double> positionsDevice;
  positionsDevice.setFromVector(host.positions);
  const double gotEnergy = computeEnergyViaForcefield(host, positionsDevice);
  const auto   gotGrad   = computeGradientViaForcefield(host, positionsDevice);

  auto                referenceFF = buildReferenceForceField(*mol);
  std::vector<double> wantGrad(referenceFF->dimension() * referenceFF->positions().size(), 0.0);
  referenceFF->calcGrad(positions.data(), wantGrad.data());

  EXPECT_NEAR(gotEnergy, referenceFF->calcEnergy(positions.data()), 1.0e-5);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::DoubleNear(2.0e-4), wantGrad));
}

INSTANTIATE_TEST_SUITE_P(UFFOneTwoAtoms, UFFGpuEdgeCases, ::testing::Values("C", "O", "CC", "CO", "CCC"));

TEST(UFFValidationSuite, BatchMatchesRDKitValidationSet) {
  const std::string                          sdfPath = getTestDataFolderPath() + "/MMFF94_dative.sdf";
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(sdfPath, mols, 25);
  ASSERT_FALSE(mols.empty());

  BatchedMolecularSystemHost       host;
  std::vector<double>              wantEnergies;
  std::vector<std::vector<double>> wantGrads;
  for (size_t i = 0; i < mols.size(); ++i) {
    auto& mol       = *mols[i];
    auto  positions = positionsFromMol(mol);
    auto  contribs  = constructForcefieldContribs(mol);
    addMoleculeToBatch(contribs, positions, host);

    auto ff = buildReferenceForceField(mol);
    wantEnergies.push_back(ff->calcEnergy(positions.data()));
    std::vector<double> wantGrad(ff->dimension() * ff->positions().size(), 0.0);
    ff->calcGrad(positions.data(), wantGrad.data());
    wantGrads.push_back(std::move(wantGrad));
  }

  nvMolKit::UFFBatchedForcefield forcefield(host);
  AsyncDeviceVector<double>      positionsDevice;
  AsyncDeviceVector<double>      energiesDevice;
  AsyncDeviceVector<double>      gradDevice;
  positionsDevice.setFromVector(host.positions);
  energiesDevice.resize(mols.size());
  energiesDevice.zero();
  gradDevice.resize(host.positions.size());
  gradDevice.zero();

  CHECK_CUDA_RETURN(forcefield.computeEnergy(energiesDevice.data(), positionsDevice.data()));
  CHECK_CUDA_RETURN(forcefield.computeGradients(gradDevice.data(), positionsDevice.data()));

  std::vector<double> gotEnergies(mols.size(), 0.0);
  std::vector<double> gotGrad(host.positions.size(), 0.0);
  energiesDevice.copyToHost(gotEnergies);
  gradDevice.copyToHost(gotGrad);
  cudaDeviceSynchronize();

  for (size_t i = 0; i < wantEnergies.size(); ++i) {
    EXPECT_NEAR(gotEnergies[i], wantEnergies[i], 1.0e-5) << "molecule " << i;
    const int           atomStart = host.indices.atomStarts[i];
    const int           atomEnd   = host.indices.atomStarts[i + 1];
    std::vector<double> gotGradMol(gotGrad.begin() + atomStart * 3, gotGrad.begin() + atomEnd * 3);
    EXPECT_THAT(gotGradMol, ::testing::Pointwise(::testing::DoubleNear(2.0e-4), wantGrads[i])) << "molecule " << i;
  }
}

TEST(UFFMinimizer, BatchMinimizerMatchesRDKitFinalEnergies) {
  const std::string                          sdfPath = getTestDataFolderPath() + "/MMFF94_dative.sdf";
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(sdfPath, mols, 8);
  ASSERT_FALSE(mols.empty());

  BatchedMolecularSystemHost host;
  std::vector<double>        referenceFinalEnergies;
  for (auto& molPtr : mols) {
    auto& mol       = *molPtr;
    auto  positions = positionsFromMol(mol);
    auto  contribs  = constructForcefieldContribs(mol);
    addMoleculeToBatch(contribs, positions, host);

    const auto optimizeResult = RDKit::UFF::UFFOptimizeMolecule(mol, 1000, 100.0, -1, true);
    referenceFinalEnergies.push_back(optimizeResult.second);
  }

  AsyncDeviceVector<double> positionsDevice;
  AsyncDeviceVector<double> gradDevice;
  AsyncDeviceVector<double> energiesDevice;
  positionsDevice.setFromVector(host.positions);
  gradDevice.resize(host.positions.size());
  gradDevice.zero();
  energiesDevice.resize(mols.size());
  energiesDevice.zero();

  nvMolKit::UFFBatchedForcefield forcefield(host);
  nvMolKit::BfgsBatchMinimizer   minimizer;
  const bool needsMore = minimizer.minimize(1000, 1.0e-6, forcefield, positionsDevice, gradDevice, energiesDevice);
  EXPECT_FALSE(needsMore);

  std::vector<double> gotFinalEnergies(mols.size(), 0.0);
  energiesDevice.copyToHost(gotFinalEnergies);
  cudaDeviceSynchronize();
  for (size_t i = 0; i < gotFinalEnergies.size(); ++i) {
    EXPECT_NEAR(gotFinalEnergies[i], referenceFinalEnergies[i], kMinimizeEnergyTol) << "molecule " << i;
  }
}

TEST(UFFMinimizer, FireWrapperImprovesAndApproachesRDKitFinalEnergies) {
  const std::string                          sdfPath = getTestDataFolderPath() + "/MMFF94_dative.sdf";
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(sdfPath, mols, 2);
  ASSERT_FALSE(mols.empty());

  std::vector<RDKit::ROMol*> molPtrs;
  std::vector<double>        initialEnergies;
  std::vector<double>        referenceFinalEnergies;
  molPtrs.reserve(mols.size());
  initialEnergies.reserve(mols.size());
  referenceFinalEnergies.reserve(mols.size());
  for (auto& molPtr : mols) {
    auto referenceMol = std::make_unique<RDKit::ROMol>(*molPtr);
    auto initialFF    = buildReferenceForceField(*referenceMol);
    initialEnergies.push_back(initialFF->calcEnergy());

    const auto optimizeResult = RDKit::UFF::UFFOptimizeMolecule(*referenceMol, 1000, 100.0, -1, true);
    referenceFinalEnergies.push_back(optimizeResult.second);
    molPtrs.push_back(molPtr.get());
  }

  nvMolKit::FireOptions options{};
  options.useMass               = false;
  options.stuckDetectionEnabled = false;
  options.gradTol               = 1e-3;
  options.dtInit                = 0.05;
  options.dMax                  = 0.2;

  const std::vector<double> vdwThresholds(molPtrs.size(), 100.0);
  const std::vector<bool>   ignoreInterfragInteractions(molPtrs.size(), true);
  const auto                result = UFFMinimizeMoleculesConfsFire(molPtrs,
                                                    /*maxIters=*/10000,
                                                    options,
                                                    vdwThresholds,
                                                    ignoreInterfragInteractions);

  ASSERT_FALSE(result.device.has_value());
  ASSERT_EQ(result.energies.size(), referenceFinalEnergies.size());
  for (size_t i = 0; i < result.energies.size(); ++i) {
    ASSERT_EQ(result.energies[i].size(), 1u);
    EXPECT_LT(result.energies[i][0], initialEnergies[i]) << "molecule " << i;
    const double tolerance = 2.0 + 0.05 * std::abs(referenceFinalEnergies[i]);
    EXPECT_NEAR(result.energies[i][0], referenceFinalEnergies[i], tolerance) << "molecule " << i;
  }
}

namespace {

double getConstraintEnergyViaForcefield(const EnergyForceContribsHost& contribs, const std::vector<double>& positions) {
  BatchedMolecularSystemHost systemHost;
  addMoleculeToBatch(contribs, positions, systemHost);
  AsyncDeviceVector<double> positionsDevice;
  positionsDevice.setFromVector(systemHost.positions);
  return computeEnergyViaForcefield(systemHost, positionsDevice);
}

std::vector<double> getConstraintGradientViaForcefield(const EnergyForceContribsHost& contribs,
                                                       const std::vector<double>&     positions) {
  BatchedMolecularSystemHost systemHost;
  addMoleculeToBatch(contribs, positions, systemHost);
  AsyncDeviceVector<double> positionsDevice;
  positionsDevice.setFromVector(systemHost.positions);
  return computeGradientViaForcefield(systemHost, positionsDevice);
}

double getReferenceConstraintEnergy(RDKit::ROMol&                  mol,
                                    const EnergyForceContribsHost& contribs,
                                    const std::vector<double>&     positions) {
  auto referenceFF = std::make_unique<ForceFields::ForceField>();
  nvMolKit::setFFPosFromConf(mol, referenceFF.get());
  referenceFF->initialize();

  for (size_t i = 0; i < contribs.distanceConstraintTerms.idx1.size(); ++i) {
    auto* dc = new ForceFields::DistanceConstraintContribs(referenceFF.get());
    dc->addContrib(contribs.distanceConstraintTerms.idx1[i],
                   contribs.distanceConstraintTerms.idx2[i],
                   contribs.distanceConstraintTerms.minLen[i],
                   contribs.distanceConstraintTerms.maxLen[i],
                   contribs.distanceConstraintTerms.forceConstant[i]);
    referenceFF->contribs().push_back(ForceFields::ContribPtr(dc));
  }
  for (size_t i = 0; i < contribs.positionConstraintTerms.idx.size(); ++i) {
    auto* pc = new ForceFields::MMFF::PositionConstraintContrib(referenceFF.get(),
                                                                contribs.positionConstraintTerms.idx[i],
                                                                contribs.positionConstraintTerms.maxDispl[i],
                                                                contribs.positionConstraintTerms.forceConstant[i]);
    referenceFF->contribs().push_back(ForceFields::ContribPtr(pc));
  }
  for (size_t i = 0; i < contribs.angleConstraintTerms.idx1.size(); ++i) {
    auto* ac = new ForceFields::AngleConstraintContribs(referenceFF.get());
    ac->addContrib(contribs.angleConstraintTerms.idx1[i],
                   contribs.angleConstraintTerms.idx2[i],
                   contribs.angleConstraintTerms.idx3[i],
                   contribs.angleConstraintTerms.minAngleDeg[i],
                   contribs.angleConstraintTerms.maxAngleDeg[i],
                   contribs.angleConstraintTerms.forceConstant[i]);
    referenceFF->contribs().push_back(ForceFields::ContribPtr(ac));
  }
  for (size_t i = 0; i < contribs.torsionConstraintTerms.idx1.size(); ++i) {
    auto* tc = new ForceFields::MMFF::TorsionConstraintContrib(referenceFF.get(),
                                                               contribs.torsionConstraintTerms.idx1[i],
                                                               contribs.torsionConstraintTerms.idx2[i],
                                                               contribs.torsionConstraintTerms.idx3[i],
                                                               contribs.torsionConstraintTerms.idx4[i],
                                                               contribs.torsionConstraintTerms.minDihedralDeg[i],
                                                               contribs.torsionConstraintTerms.maxDihedralDeg[i],
                                                               contribs.torsionConstraintTerms.forceConstant[i]);
    referenceFF->contribs().push_back(ForceFields::ContribPtr(tc));
  }

  std::vector<double> evalPositions = positions;
  return referenceFF->calcEnergy(evalPositions.data());
}

std::vector<double> getReferenceConstraintGradient(RDKit::ROMol&                  mol,
                                                   const EnergyForceContribsHost& contribs,
                                                   const std::vector<double>&     positions) {
  auto referenceFF = std::make_unique<ForceFields::ForceField>();
  nvMolKit::setFFPosFromConf(mol, referenceFF.get());
  referenceFF->initialize();

  for (size_t i = 0; i < contribs.distanceConstraintTerms.idx1.size(); ++i) {
    auto* dc = new ForceFields::DistanceConstraintContribs(referenceFF.get());
    dc->addContrib(contribs.distanceConstraintTerms.idx1[i],
                   contribs.distanceConstraintTerms.idx2[i],
                   contribs.distanceConstraintTerms.minLen[i],
                   contribs.distanceConstraintTerms.maxLen[i],
                   contribs.distanceConstraintTerms.forceConstant[i]);
    referenceFF->contribs().push_back(ForceFields::ContribPtr(dc));
  }
  for (size_t i = 0; i < contribs.positionConstraintTerms.idx.size(); ++i) {
    auto* pc = new ForceFields::MMFF::PositionConstraintContrib(referenceFF.get(),
                                                                contribs.positionConstraintTerms.idx[i],
                                                                contribs.positionConstraintTerms.maxDispl[i],
                                                                contribs.positionConstraintTerms.forceConstant[i]);
    referenceFF->contribs().push_back(ForceFields::ContribPtr(pc));
  }
  for (size_t i = 0; i < contribs.angleConstraintTerms.idx1.size(); ++i) {
    auto* ac = new ForceFields::AngleConstraintContribs(referenceFF.get());
    ac->addContrib(contribs.angleConstraintTerms.idx1[i],
                   contribs.angleConstraintTerms.idx2[i],
                   contribs.angleConstraintTerms.idx3[i],
                   contribs.angleConstraintTerms.minAngleDeg[i],
                   contribs.angleConstraintTerms.maxAngleDeg[i],
                   contribs.angleConstraintTerms.forceConstant[i]);
    referenceFF->contribs().push_back(ForceFields::ContribPtr(ac));
  }
  for (size_t i = 0; i < contribs.torsionConstraintTerms.idx1.size(); ++i) {
    auto* tc = new ForceFields::MMFF::TorsionConstraintContrib(referenceFF.get(),
                                                               contribs.torsionConstraintTerms.idx1[i],
                                                               contribs.torsionConstraintTerms.idx2[i],
                                                               contribs.torsionConstraintTerms.idx3[i],
                                                               contribs.torsionConstraintTerms.idx4[i],
                                                               contribs.torsionConstraintTerms.minDihedralDeg[i],
                                                               contribs.torsionConstraintTerms.maxDihedralDeg[i],
                                                               contribs.torsionConstraintTerms.forceConstant[i]);
    referenceFF->contribs().push_back(ForceFields::ContribPtr(tc));
  }

  std::vector<double> evalPositions = positions;
  std::vector<double> gradients(positions.size(), 0.0);
  referenceFF->calcGrad(evalPositions.data(), gradients.data());
  return gradients;
}

}  // namespace

TEST_F(UFFGpuTestFixture, DistanceConstraintEnergy) {
  EnergyForceContribsHost                                       contribs;
  const nvMolKit::ForceFieldConstraints::DistanceConstraintSpec spec{0, 2, true, 0.3, 0.6, 15.0};
  nvMolKit::ForceFieldConstraints::appendDistanceConstraint(contribs, positions_, spec);

  const double wantEnergy = getReferenceConstraintEnergy(*mol_, contribs, positions_);
  EXPECT_NEAR(getConstraintEnergyViaForcefield(contribs, positions_), wantEnergy, kEnergyTol);
}

TEST_F(UFFGpuTestFixture, DistanceConstraintGradient) {
  EnergyForceContribsHost                                       contribs;
  const nvMolKit::ForceFieldConstraints::DistanceConstraintSpec spec{0, 2, true, 0.3, 0.6, 15.0};
  nvMolKit::ForceFieldConstraints::appendDistanceConstraint(contribs, positions_, spec);

  const auto wantGrad = getReferenceConstraintGradient(*mol_, contribs, positions_);
  const auto gotGrad  = getConstraintGradientViaForcefield(contribs, positions_);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::DoubleNear(kGradTol), wantGrad));
}

TEST_F(UFFGpuTestFixture, PositionConstraintEnergy) {
  EnergyForceContribsHost                                       contribs;
  const nvMolKit::ForceFieldConstraints::PositionConstraintSpec spec{0, 0.1, 50.0};
  nvMolKit::ForceFieldConstraints::appendPositionConstraint(contribs, positions_, spec);

  std::vector<double> evalPositions = positions_;
  evalPositions[0] += 0.25;
  const double wantEnergy = getReferenceConstraintEnergy(*mol_, contribs, evalPositions);
  EXPECT_NEAR(getConstraintEnergyViaForcefield(contribs, evalPositions), wantEnergy, kEnergyTol);
}

TEST_F(UFFGpuTestFixture, PositionConstraintGradient) {
  EnergyForceContribsHost                                       contribs;
  const nvMolKit::ForceFieldConstraints::PositionConstraintSpec spec{0, 0.1, 50.0};
  nvMolKit::ForceFieldConstraints::appendPositionConstraint(contribs, positions_, spec);

  std::vector<double> evalPositions = positions_;
  evalPositions[0] += 0.25;
  const auto wantGrad = getReferenceConstraintGradient(*mol_, contribs, evalPositions);
  const auto gotGrad  = getConstraintGradientViaForcefield(contribs, evalPositions);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::DoubleNear(kGradTol), wantGrad));
}

TEST_F(UFFGpuTestFixture, AngleConstraintEnergy) {
  EnergyForceContribsHost                                    contribs;
  const nvMolKit::ForceFieldConstraints::AngleConstraintSpec spec{0, 1, 2, true, 5.0, 10.0, 20.0};
  nvMolKit::ForceFieldConstraints::appendAngleConstraint(contribs, positions_, spec);

  const double wantEnergy = getReferenceConstraintEnergy(*mol_, contribs, positions_);
  EXPECT_NEAR(getConstraintEnergyViaForcefield(contribs, positions_), wantEnergy, kEnergyTol);
}

TEST_F(UFFGpuTestFixture, AngleConstraintGradient) {
  EnergyForceContribsHost                                    contribs;
  const nvMolKit::ForceFieldConstraints::AngleConstraintSpec spec{0, 1, 2, true, 5.0, 10.0, 20.0};
  nvMolKit::ForceFieldConstraints::appendAngleConstraint(contribs, positions_, spec);

  const auto wantGrad = getReferenceConstraintGradient(*mol_, contribs, positions_);
  const auto gotGrad  = getConstraintGradientViaForcefield(contribs, positions_);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::DoubleNear(1.0e-3), wantGrad));
}

TEST_F(UFFGpuTestFixture, TorsionConstraintEnergy) {
  EnergyForceContribsHost                                      contribs;
  const nvMolKit::ForceFieldConstraints::TorsionConstraintSpec spec{0, 1, 2, 3, true, 15.0, 30.0, 12.0};
  nvMolKit::ForceFieldConstraints::appendTorsionConstraint(contribs, positions_, spec);

  const double wantEnergy = getReferenceConstraintEnergy(*mol_, contribs, positions_);
  EXPECT_NEAR(getConstraintEnergyViaForcefield(contribs, positions_), wantEnergy, kEnergyTol);
}

TEST_F(UFFGpuTestFixture, TorsionConstraintGradient) {
  EnergyForceContribsHost                                      contribs;
  const nvMolKit::ForceFieldConstraints::TorsionConstraintSpec spec{0, 1, 2, 3, true, 15.0, 30.0, 12.0};
  nvMolKit::ForceFieldConstraints::appendTorsionConstraint(contribs, positions_, spec);

  const auto wantGrad = getReferenceConstraintGradient(*mol_, contribs, positions_);
  const auto gotGrad  = getConstraintGradientViaForcefield(contribs, positions_);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::DoubleNear(1.0e-3), wantGrad));
}
