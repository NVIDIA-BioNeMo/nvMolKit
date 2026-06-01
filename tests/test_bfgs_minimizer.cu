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

#include <ForceField/AngleConstraints.h>
#include <ForceField/DistanceConstraints.h>
#include <ForceField/ForceField.h>
#include <ForceField/MMFF/PositionConstraint.h>
#include <ForceField/MMFF/TorsionConstraint.h>
#include <gmock/gmock.h>
// clang-format off
// Bug in RDKit, includes need to be ordered.
#include <GraphMol/ROMol.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <Numerics/Optimizer/BFGSOpt.h>
// clang-format on
#include <gtest/gtest.h>

#include <random>

#include "rdkit_extensions/mmff_flattened_builder.h"
#include "src/forcefields/batched_forcefield.h"
#include "src/forcefields/forcefield_constraints.h"
#include "src/forcefields/mmff.h"
#include "src/forcefields/mmff_batched_forcefield.h"
#include "src/minimizer/bfgs_minimize.h"
#include "src/utils/device.h"
#include "tests/test_utils.h"

using ::nvMolKit::MMFF::BatchedMolecularDeviceBuffers;
using ::nvMolKit::MMFF::BatchedMolecularSystemHost;
using ::nvMolKit::MMFF::EnergyForceContribsHost;

TEST(BFGSMinimizerTest, AllocationAndIdentity) {
  // Create a BFGS minimizer
  nvMolKit::BfgsBatchMinimizer bfgsMinimizer;

  nvMolKit::MMFF::BatchedMolecularSystemHost    host;
  nvMolKit::MMFF::BatchedMolecularDeviceBuffers dev;

  // 3 systems in batch.
  std::vector<int> numAtoms = {2, 3, 5, 2};
  host.indices.atomStarts   = {0, 2, 5, 10, 12};
  dev.indices.atomStarts.resize(host.indices.atomStarts.size());
  dev.indices.atomStarts.setFromVector(host.indices.atomStarts);

  bfgsMinimizer.initialize(host.indices.atomStarts,
                           dev.indices.atomStarts.data(),
                           nullptr,
                           nullptr,
                           nullptr,
                           nvMolKit::BfgsBackend::BATCHED,
                           nullptr);
  // expect (2 * 3)^2 + (3 * 3)^2 + (5*3)^2 (2*3)^2 = 378
  int           hessianStorageSize = bfgsMinimizer.inverseHessian_.size();
  constexpr int wantStorageSize    = 378;
  ASSERT_EQ(hessianStorageSize, wantStorageSize);

  bfgsMinimizer.setHessianToIdentity();

  const double*       hessian = bfgsMinimizer.inverseHessian_.data();
  std::vector<double> hessianHost(hessianStorageSize);
  ASSERT_EQ(cudaMemcpy(hessianHost.data(), hessian, hessianStorageSize * sizeof(double), cudaMemcpyDeviceToHost), 0);

  // For each system, expect (0,0), (1,1)... set to 1.
  std::vector<int> want;
  for (int numAtomsInMol : numAtoms) {
    for (int j = 0; j < (3 * numAtomsInMol); j++) {
      for (int k = 0; k < (3 * numAtomsInMol); k++) {
        want.push_back(j == k ? 1 : 0);
      }
    }
  }
  ASSERT_THAT(want, ::testing::SizeIs(wantStorageSize)) << "Invalid expected setup";
  EXPECT_THAT(hessianHost, ::testing::ElementsAreArray(want));
}

TEST(BFGSMinimizerTest, CountFinishedLineSearch) {
  // Create a BFGS minimizer
  nvMolKit::BfgsBatchMinimizer bfgsMinimizer;

  nvMolKit::MMFF::BatchedMolecularSystemHost    host;
  nvMolKit::MMFF::BatchedMolecularDeviceBuffers dev;

  host.indices.atomStarts = {0, 2, 5, 10, 12, 15, 18};
  dev.indices.atomStarts.resize(host.indices.atomStarts.size());
  dev.indices.atomStarts.setFromVector(host.indices.atomStarts);

  bfgsMinimizer.initialize(host.indices.atomStarts,
                           dev.indices.atomStarts.data(),
                           nullptr,
                           nullptr,
                           nullptr,
                           nvMolKit::BfgsBackend::BATCHED,
                           nullptr);
  std::vector<int16_t> finished        = {-2, -1, 0, 1, -2, -1, 0};
  constexpr int        wantNumFinished = 5;  // non -2 means finished, regardless of error status.

  bfgsMinimizer.lineSearchStatus_.zero();
  bfgsMinimizer.lineSearchStatus_.setFromVector(finished);
  const double gotNumFinished = bfgsMinimizer.lineSearchCountFinished();
  EXPECT_EQ(gotNumFinished, wantNumFinished);
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

struct ConstraintSpecs {
  nvMolKit::ForceFieldConstraints::DistanceConstraintSpec distance{0, 2, true, 0.3, 0.6, 15.0};
  nvMolKit::ForceFieldConstraints::PositionConstraintSpec position{0, 0.1, 50.0};
  nvMolKit::ForceFieldConstraints::AngleConstraintSpec    angle{0, 1, 2, true, 5.0, 10.0, 20.0};
  nvMolKit::ForceFieldConstraints::TorsionConstraintSpec  torsion{0, 1, 2, 3, true, 15.0, 30.0, 12.0};
};

void addConstraintSpecsToContribs(EnergyForceContribsHost&   contribs,
                                  const std::vector<double>& positions,
                                  const ConstraintSpecs&     specs = {}) {
  nvMolKit::ForceFieldConstraints::appendDistanceConstraint(contribs, positions, specs.distance);
  nvMolKit::ForceFieldConstraints::appendPositionConstraint(contribs, positions, specs.position);
  nvMolKit::ForceFieldConstraints::appendAngleConstraint(contribs, positions, specs.angle);
  nvMolKit::ForceFieldConstraints::appendTorsionConstraint(contribs, positions, specs.torsion);
}

void addConstraintSpecsToForcefield(ForceFields::ForceField& forcefield, const ConstraintSpecs& specs = {}) {
  auto* distanceContribs = new ForceFields::DistanceConstraintContribs(&forcefield);
  distanceContribs->addContrib(specs.distance.idx1,
                               specs.distance.idx2,
                               specs.distance.relative,
                               specs.distance.minLen,
                               specs.distance.maxLen,
                               specs.distance.forceConstant);
  forcefield.contribs().push_back(ForceFields::ContribPtr(distanceContribs));

  auto* positionContrib = new ForceFields::MMFF::PositionConstraintContrib(&forcefield,
                                                                           specs.position.idx,
                                                                           specs.position.maxDispl,
                                                                           specs.position.forceConstant);
  forcefield.contribs().push_back(ForceFields::ContribPtr(positionContrib));

  auto* angleContribs = new ForceFields::AngleConstraintContribs(&forcefield);
  angleContribs->addContrib(specs.angle.idx1,
                            specs.angle.idx2,
                            specs.angle.idx3,
                            specs.angle.relative,
                            specs.angle.minAngleDeg,
                            specs.angle.maxAngleDeg,
                            specs.angle.forceConstant);
  forcefield.contribs().push_back(ForceFields::ContribPtr(angleContribs));

  auto* torsionContrib = new ForceFields::MMFF::TorsionConstraintContrib(&forcefield,
                                                                         specs.torsion.idx1,
                                                                         specs.torsion.idx2,
                                                                         specs.torsion.idx3,
                                                                         specs.torsion.idx4,
                                                                         specs.torsion.relative,
                                                                         specs.torsion.minDihedralDeg,
                                                                         specs.torsion.maxDihedralDeg,
                                                                         specs.torsion.forceConstant);
  forcefield.contribs().push_back(ForceFields::ContribPtr(torsionContrib));
}

std::unique_ptr<ForceFields::ForceField> constructConstrainedRDKitForcefield(RDKit::ROMol&          mol,
                                                                             const ConstraintSpecs& specs = {}) {
  auto molProps   = std::make_unique<RDKit::MMFF::MMFFMolProperties>(mol);
  auto forcefield = std::unique_ptr<ForceFields::ForceField>(RDKit::MMFF::constructForceField(mol, molProps.get()));
  addConstraintSpecsToForcefield(*forcefield, specs);
  return forcefield;
}

class BFGSMinimizerTestFixture : public ::testing::Test {
 protected:
  void setUpMMFFSystems(int numMols, bool duplicateFirstMol = false, bool addConstraints = false) {
    getMols(getTestDataFolderPath() + "/MMFF94_dative.sdf", mols, duplicateFirstMol ? 1 : numMols);
    if (duplicateFirstMol) {
      mols.resize(1);
      for (int i = 1; i < numMols; ++i) {
        mols.push_back(std::make_unique<RDKit::ROMol>(*mols[0]));
      }
    }

    setUpCommon(addConstraints);
  }

  void setUpConstrainedMMFFSystems(int numMols, bool duplicateFirstMol = false) {
    setUpMMFFSystems(numMols, duplicateFirstMol, true);
  }

  void setUpCommon(bool addConstraints = false) {
    int runningIdx = 0;
    for (const auto& mol : mols) {
      perturbConformer(mol->getConformer(), 0.3, runningIdx++);
      std::vector<double> positions(3 * mol->getNumAtoms());
      for (unsigned int i = 0; i < mol->getNumAtoms(); ++i) {
        RDGeom::Point3D pos  = mol->getConformer().getAtomPos(i);
        positions[3 * i]     = pos.x;
        positions[3 * i + 1] = pos.y;
        positions[3 * i + 2] = pos.z;
      }
      auto ffParams = nvMolKit::MMFF::constructForcefieldContribs(*mol);
      if (addConstraints) {
        addConstraintSpecsToContribs(ffParams, positions);
      }
      nvMolKit::MMFF::addMoleculeToBatch(ffParams, positions, systemHost);
    }
    nvMolKit::MMFF::sendContribsAndIndicesToDevice(systemHost, systemDevice);
    nvMolKit::MMFF::allocateIntermediateBuffers(systemHost, systemDevice);
    systemDevice.energyOuts.zero();
    systemDevice.positions.setFromVector(systemHost.positions);
    systemDevice.grad.resize(systemDevice.positions.size());
    systemDevice.grad.zero();
  }

  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  BatchedMolecularSystemHost                 systemHost;
  BatchedMolecularDeviceBuffers              systemDevice;
};

// Parameterized test for both BFGS backends
class BFGSMinimizerBackendTest : public BFGSMinimizerTestFixture,
                                 public ::testing::WithParamInterface<nvMolKit::BfgsBackend> {};

namespace {

void minimizeMMFF(nvMolKit::BfgsBatchMinimizer&     minimizer,
                  const int                         numIters,
                  const double                      gradTol,
                  const BatchedMolecularSystemHost& systemHost,
                  BatchedMolecularDeviceBuffers&    systemDevice,
                  cudaStream_t                      stream          = nullptr,
                  const uint8_t*                    activeThisStage = nullptr) {
  if (minimizer.resolveBackend(systemHost.indices.atomStarts) == nvMolKit::BfgsBackend::BATCHED) {
    nvMolKit::MMFFBatchedForcefield forcefield(systemHost, {}, stream);
    minimizer.minimize(numIters,
                       gradTol,
                       forcefield,
                       systemDevice.positions,
                       systemDevice.grad,
                       systemDevice.energyOuts,
                       activeThisStage);
    return;
  }
  minimizer.minimizeWithMMFF(numIters, gradTol, systemHost.indices.atomStarts, systemDevice, activeThisStage);
}

void refLineSearchSetup(unsigned int  dim,
                        const double* oldPt,
                        const double* grad,
                        double*       dir,
                        double        maxStep,
                        double&       slopeOut,
                        double&       lambdaMinOut) {
  double sum = 0.0, test = 0.0;
  for (unsigned int i = 0; i < dim; i++) {
    sum += dir[i] * dir[i];
  }
  sum = std::sqrt(sum);

  // rescale if we're trying to move too far:
  if (sum > maxStep) {
    for (unsigned int i = 0; i < dim; i++) {
      dir[i] *= maxStep / sum;
    }
  }
  // make sure our direction has at least some component along
  // -grad
  slopeOut = 0.0;
  for (unsigned int i = 0; i < dim; i++) {
    slopeOut += dir[i] * grad[i];
  }
  if (slopeOut >= 0.0) {
    return;
  }

  test = 0.0;
  for (unsigned int i = 0; i < dim; i++) {
    double temp = fabs(dir[i]) / std::max(fabs(oldPt[i]), 1.0);
    if (temp > test) {
      test = temp;
    }
  }
  lambdaMinOut = 1e-7 / test;
}
}  // namespace

TEST_F(BFGSMinimizerTestFixture, LineSearchSetup) {
  const int           numMols = 3;
  std::vector<double> wantDirs;
  std::vector<double> wantSlopes;
  std::vector<double> wantLambdaMins;

  std::vector<double> accumGrads;

  setUpMMFFSystems(numMols);

  for (const auto& mol : mols) {
    std::vector<double> positions(3 * mol->getNumAtoms());
    for (unsigned int i = 0; i < mol->getNumAtoms(); ++i) {
      RDGeom::Point3D pos  = mol->getConformer().getAtomPos(i);
      positions[3 * i]     = pos.x;
      positions[3 * i + 1] = pos.y;
      positions[3 * i + 2] = pos.z;
    }
    // Now compute reference quantities.
    auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
    std::unique_ptr<ForceFields::ForceField> molFF(RDKit::MMFF::constructForceField(*mol, molProps.get()));

    std::vector<double> grads(3 * mol->getNumAtoms());
    molFF->calcGrad(&grads[0]);
    accumGrads.insert(accumGrads.end(), grads.begin(), grads.end());
    std::vector<double> dir(3 * mol->getNumAtoms());
    std::transform(grads.begin(), grads.end(), dir.begin(), std::negate<double>());

    double& wantSlope     = wantSlopes.emplace_back(0.0);
    double& wantLambdaMin = wantLambdaMins.emplace_back(0.0);

    refLineSearchSetup(3 * mol->getNumAtoms(), &positions[0], &grads[0], &dir[0], 0.1, wantSlope, wantLambdaMin);
    wantDirs.insert(wantDirs.end(), dir.begin(), dir.end());
  }

  systemDevice.grad.resize(accumGrads.size());
  systemDevice.grad.setFromVector(accumGrads);

  nvMolKit::BfgsBatchMinimizer bfgsMinimizer;
  bfgsMinimizer.initialize(systemHost.indices.atomStarts,
                           systemDevice.indices.atomStarts.data(),
                           systemDevice.positions.data(),
                           systemDevice.grad.data(),
                           systemDevice.energyOuts.data(),
                           nvMolKit::BfgsBackend::BATCHED);
  bfgsMinimizer.numUnfinishedSystems_ = systemHost.indices.atomStarts.size() - 1;

  std::vector<double> accumDirs(accumGrads.size());
  std::transform(accumGrads.begin(), accumGrads.end(), accumDirs.begin(), std::negate<double>());
  bfgsMinimizer.lineSearchDir_.setFromVector(accumDirs);

  std::vector<double> maxStepsHost(numMols, 0.1);
  bfgsMinimizer.lineSearchMaxSteps_.setFromVector(maxStepsHost);
  bfgsMinimizer.doLineSearchSetup(systemDevice.energyOuts.data());

  ASSERT_EQ(bfgsMinimizer.lineSearchDir_.size(), wantDirs.size());
  std::vector<double> gotDirs(wantDirs.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotDirs.data(),
                       bfgsMinimizer.lineSearchDir_.data(),
                       wantDirs.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));
  std::vector<double> gotSlopes(numMols);
  ASSERT_EQ(0,
            cudaMemcpy(gotSlopes.data(),
                       bfgsMinimizer.lineSearchSlope_.data(),
                       wantSlopes.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));
  std::vector<double> gotLambdaMins(numMols);
  ASSERT_EQ(0,
            cudaMemcpy(gotLambdaMins.data(),
                       bfgsMinimizer.lineSearchLambdaMins_.data(),
                       wantLambdaMins.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));
  EXPECT_THAT(gotDirs, ::testing::Pointwise(::testing::DoubleNear(1e-4), wantDirs));
  EXPECT_THAT(gotSlopes, ::testing::Pointwise(::testing::DoubleNear(1e-4), wantSlopes));
  EXPECT_THAT(gotLambdaMins, ::testing::Pointwise(::testing::DoubleNear(1e-4), wantLambdaMins));
}

TEST_F(BFGSMinimizerTestFixture, ComputeMaxSteps) {
  const int           numMols = 3;
  std::vector<double> wantMaxSteps;
  setUpMMFFSystems(numMols);

  for (const auto& mol : mols) {
    const uint32_t      numAtoms = mol->getNumAtoms();
    std::vector<double> positions(3 * numAtoms);
    for (unsigned int i = 0; i < numAtoms; ++i) {
      RDGeom::Point3D pos  = mol->getConformer().getAtomPos(i);
      positions[3 * i]     = pos.x;
      positions[3 * i + 1] = pos.y;
      positions[3 * i + 2] = pos.z;
    }
    // Now compute reference quantities.
    double sum = 0.0;
    for (const double& pos : positions) {
      sum += pos * pos;
    }
    constexpr double MAXSTEP = 100.0;
    // pick a max step size:
    wantMaxSteps.push_back(MAXSTEP * std::max(sqrt(sum), static_cast<double>(3 * numAtoms)));
  }

  nvMolKit::BfgsBatchMinimizer bfgsMinimizer;
  bfgsMinimizer.initialize(systemHost.indices.atomStarts,
                           systemDevice.indices.atomStarts.data(),
                           systemDevice.positions.data(),
                           systemDevice.grad.data(),
                           systemDevice.energyOuts.data(),
                           nvMolKit::BfgsBackend::BATCHED);
  bfgsMinimizer.numUnfinishedSystems_ = systemHost.indices.atomStarts.size() - 1;

  bfgsMinimizer.setMaxStep();

  std::vector<double> gotMaxSteps(numMols);
  ASSERT_EQ(0,
            cudaMemcpy(gotMaxSteps.data(),
                       bfgsMinimizer.lineSearchMaxSteps_.data(),
                       gotMaxSteps.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));
  EXPECT_THAT(gotMaxSteps, ::testing::Pointwise(::testing::DoubleNear(1e-4), wantMaxSteps));
}

template <typename T> std::string debugDump(const nvMolKit::AsyncDeviceVector<T>& data, const std::string& name) {
  std::vector<T> dataHost(data.size());
  cudaMemcpy(dataHost.data(), data.data(), data.size() * sizeof(T), cudaMemcpyDeviceToHost);
  std::string result  = name + ": {";
  T*          dataPtr = dataHost.data();
  for (size_t i = 0; i < data.size(); ++i) {
    result += std::to_string(dataPtr[i]) + ", ";
  }
  result += "}";
  return result;
}

TEST_F(BFGSMinimizerTestFixture, FullLineSearch) {
  const int           numMols = 3;
  std::vector<double> wantMaxSteps;
  setUpMMFFSystems(numMols);

  std::vector<double> accumPositions;
  std::vector<double> accumEnergies;
  std::vector<int>    accumStatuses;

  for (const auto& mol : mols) {
    const uint32_t      numAtoms = mol->getNumAtoms();
    std::vector<double> positions(3 * numAtoms);
    for (uint32_t i = 0; i < numAtoms; ++i) {
      RDGeom::Point3D pos  = mol->getConformer().getAtomPos(i);
      positions[3 * i]     = pos.x;
      positions[3 * i + 1] = pos.y;
      positions[3 * i + 2] = pos.z;
    }

    // Pre-line search setup.
    constexpr double MAXSTEP = 100.0;
    double           sum     = 0.0;
    for (const double& pos : positions) {
      sum += pos * pos;
    }
    const double                             maxStep = MAXSTEP * std::max(sqrt(sum), static_cast<double>(3 * numAtoms));
    auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
    std::unique_ptr<ForceFields::ForceField> molFF(RDKit::MMFF::constructForceField(*mol, molProps.get()));
    const double                             energy = molFF->calcEnergy();
    std::vector<double>                      grads(3 * numAtoms);
    molFF->calcGrad(&grads[0]);
    std::vector<double> dir(3 * mol->getNumAtoms());
    std::transform(grads.begin(), grads.end(), dir.begin(), std::negate<double>());
    std::vector<double> gotPositions(3 * numAtoms);

    auto energyLambda = [&molFF](double* positions) -> double { return molFF->calcEnergy(positions); };

    double gotEnergy  = 0.0;
    int    gotResCode = -2;

    // pick a max step size:
    BFGSOpt::linearSearch(3 * numAtoms,
                          positions.data(),
                          energy,
                          grads.data(),
                          dir.data(),
                          gotPositions.data(),
                          gotEnergy,
                          energyLambda,
                          maxStep,
                          gotResCode);
    accumPositions.insert(accumPositions.end(), gotPositions.begin(), gotPositions.end());
    accumEnergies.push_back(gotEnergy);
    accumStatuses.push_back(gotResCode);
  }
  systemDevice.grad.resize(accumPositions.size());
  systemDevice.grad.zero();
  nvMolKit::BfgsBatchMinimizer bfgsMinimizer;
  bfgsMinimizer.initialize(systemHost.indices.atomStarts,
                           systemDevice.indices.atomStarts.data(),
                           systemDevice.positions.data(),
                           systemDevice.grad.data(),
                           systemDevice.energyOuts.data(),
                           nvMolKit::BfgsBackend::BATCHED);
  bfgsMinimizer.numUnfinishedSystems_ = systemHost.indices.atomStarts.size() - 1;

  bfgsMinimizer.setHessianToIdentity();
  bfgsMinimizer.setMaxStep();

  auto eFunc = [&](const double* positions) { nvMolKit::MMFF::computeEnergy(systemDevice, positions); };

  eFunc(nullptr);
  nvMolKit::MMFF::computeGradients(systemDevice);
  nvMolKit::copyAndInvert(systemDevice.grad, bfgsMinimizer.lineSearchDir_);
  bfgsMinimizer.doLineSearchSetup(systemDevice.energyOuts.data());
  auto             res                    = debugDump(bfgsMinimizer.lineSearchStatus_, "statuses");
  int              lineSearchIter         = 0;
  constexpr double MAX_ITER_LINEAR_SEARCH = 1000;
  while (lineSearchIter < MAX_ITER_LINEAR_SEARCH && bfgsMinimizer.lineSearchCountFinished() < numMols) {
    bfgsMinimizer.doLineSearchPerturb();
    auto res4 = debugDump(bfgsMinimizer.lineSearchStatus_, "statuses2");

    systemDevice.energyOuts.zero();
    systemDevice.energyBuffer.zero();
    auto res2 = debugDump(bfgsMinimizer.scratchPositions_, "scratchpos");
    eFunc(bfgsMinimizer.scratchPositions_.data());
    auto res3 = debugDump(systemDevice.energyOuts, "energyOuts");
    bfgsMinimizer.doLineSearchPostEnergy(lineSearchIter);
    lineSearchIter++;
  }
  bfgsMinimizer.doLineSearchPostLoop();

  std::vector<double> gotPositions(accumPositions.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotPositions.data(),
                       bfgsMinimizer.scratchPositions_.data(),
                       gotPositions.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));
  std::vector<double> gotEnergies(accumEnergies.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotEnergies.data(),
                       systemDevice.energyOuts.data(),
                       gotEnergies.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));
  std::vector<int16_t> gotStatuses(accumStatuses.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotStatuses.data(),
                       bfgsMinimizer.lineSearchStatus_.data(),
                       gotStatuses.size() * sizeof(int16_t),
                       cudaMemcpyDeviceToHost));

  EXPECT_THAT(gotPositions, ::testing::Pointwise(::testing::DoubleNear(1e-4), accumPositions));
  EXPECT_THAT(gotEnergies, ::testing::Pointwise(::testing::DoubleNear(1e-4), accumEnergies));
  EXPECT_THAT(gotStatuses, ::testing::Pointwise(::testing::Eq(), accumStatuses));
}

TEST_P(BFGSMinimizerBackendTest, E2EMinimizationSingleSystemUnconvergedMatches) {
  const nvMolKit::BfgsBackend backend = GetParam();
  nvMolKit::ScopedStream      stream;
  const int                   numMols  = 1;
  const int                   maxIters = 10;
  setUpMMFFSystems(numMols);

  cudaStreamSynchronize(nullptr);
  nvMolKit::MMFF::setStreams(systemDevice, stream.stream());
  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(/*dataDim=*/3,
                                             nvMolKit::DebugLevel::STEPWISE,
                                             true,
                                             stream.stream(),
                                             backend);

  minimizeMMFF(bfgsMinimizer, maxIters, 1e-4, systemHost, systemDevice, stream.stream());

  std::vector<double> refEnergies;
  for (auto& mol : mols) {
    RDKit::MMFF::MMFFOptimizeMolecule(*mol, maxIters, "MMFF94", 100.0);
    auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
    std::unique_ptr<ForceFields::ForceField> molFF(RDKit::MMFF::constructForceField(*mol, molProps.get()));
    refEnergies.push_back(molFF->calcEnergy());
  }

  std::vector<double> gotEnergies(systemDevice.energyOuts.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotEnergies.data(),
                       systemDevice.energyOuts.data(),
                       gotEnergies.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));

  EXPECT_THAT(gotEnergies, ::testing::Pointwise(::testing::DoubleNear(5e-3), refEnergies));

  std::vector<int16_t> gotStatuses(systemDevice.energyOuts.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotStatuses.data(),
                       bfgsMinimizer.statuses_.data(),
                       gotStatuses.size() * sizeof(int16_t),
                       cudaMemcpyDeviceToHost));
  EXPECT_THAT(gotStatuses, ::testing::Pointwise(::testing::Eq(), std::vector<int16_t>(numMols, 1)));
  nvMolKit::MMFF::setStreams(systemDevice, nullptr);
}

TEST_P(BFGSMinimizerBackendTest, E2EMinimizationSingleSystemConvergedMatches) {
  const nvMolKit::BfgsBackend backend  = GetParam();
  const int                   numMols  = 1;
  const int                   maxIters = 50;
  setUpMMFFSystems(numMols);

  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(/*dataDim=*/3, nvMolKit::DebugLevel::STEPWISE, true, nullptr, backend);

  minimizeMMFF(bfgsMinimizer, maxIters, 1e-4, systemHost, systemDevice);

  std::vector<double> refEnergies;
  for (auto& mol : mols) {
    RDKit::MMFF::MMFFOptimizeMolecule(*mol, maxIters, "MMFF94", 100.0);
    auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
    std::unique_ptr<ForceFields::ForceField> molFF(RDKit::MMFF::constructForceField(*mol, molProps.get()));
    refEnergies.push_back(molFF->calcEnergy());
  }

  std::vector<double> gotEnergies(systemDevice.energyOuts.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotEnergies.data(),
                       systemDevice.energyOuts.data(),
                       gotEnergies.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));

  EXPECT_THAT(gotEnergies, ::testing::Pointwise(::testing::DoubleNear(1e-4), refEnergies));

  // Status checking only works with BATCHED backend (PER_MOLECULE doesn't track detailed convergence yet)
  std::vector<int16_t> gotStatuses(systemDevice.energyOuts.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotStatuses.data(),
                       bfgsMinimizer.statuses_.data(),
                       gotStatuses.size() * sizeof(int16_t),
                       cudaMemcpyDeviceToHost));
  EXPECT_THAT(gotStatuses, ::testing::Pointwise(::testing::Eq(), std::vector<int16_t>(numMols, 0)));
}

TEST_P(BFGSMinimizerBackendTest, E2EMinimizationSingleSystemConstrainedMatches) {
  const nvMolKit::BfgsBackend backend  = GetParam();
  const int                   numMols  = 1;
  const int                   maxIters = 200;
  setUpConstrainedMMFFSystems(numMols);

  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(/*dataDim=*/3, nvMolKit::DebugLevel::STEPWISE, true, nullptr, backend);

  minimizeMMFF(bfgsMinimizer, maxIters, 1e-4, systemHost, systemDevice);

  std::vector<double> refEnergies;
  for (auto& mol : mols) {
    auto molFF = constructConstrainedRDKitForcefield(*mol);
    EXPECT_EQ(molFF->minimize(maxIters), 0);
    refEnergies.push_back(molFF->calcEnergy());
  }

  std::vector<double> gotEnergies(systemDevice.energyOuts.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotEnergies.data(),
                       systemDevice.energyOuts.data(),
                       gotEnergies.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));

  EXPECT_THAT(gotEnergies, ::testing::Pointwise(::testing::DoubleNear(1e-4), refEnergies));

  std::vector<int16_t> gotStatuses(systemDevice.energyOuts.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotStatuses.data(),
                       bfgsMinimizer.statuses_.data(),
                       gotStatuses.size() * sizeof(int16_t),
                       cudaMemcpyDeviceToHost));
  EXPECT_THAT(gotStatuses, ::testing::Pointwise(::testing::Eq(), std::vector<int16_t>(numMols, 0)));
}

TEST_P(BFGSMinimizerBackendTest, E2EMinimizationMultiSystemSameMolMatchesUnconverged) {
  const nvMolKit::BfgsBackend backend  = GetParam();
  const int                   numMols  = 10;
  const int                   maxIters = 10;
  setUpMMFFSystems(numMols, true);

  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(/*dim=*/3, nvMolKit::DebugLevel::STEPWISE, true, nullptr, backend);

  minimizeMMFF(bfgsMinimizer, maxIters, 1e-4, systemHost, systemDevice);

  std::vector<double> refEnergies;
  for (auto& mol : mols) {
    RDKit::MMFF::MMFFOptimizeMolecule(*mol, maxIters, "MMFF94", 100.0);
    auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
    std::unique_ptr<ForceFields::ForceField> molFF(RDKit::MMFF::constructForceField(*mol, molProps.get()));
    refEnergies.push_back(molFF->calcEnergy());
  }

  std::vector<double> gotEnergies(systemDevice.energyOuts.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotEnergies.data(),
                       systemDevice.energyOuts.data(),
                       gotEnergies.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));

  EXPECT_THAT(gotEnergies, ::testing::Pointwise(::testing::DoubleNear(5e-3), refEnergies));

  // Status checking only works with BATCHED backend (PER_MOLECULE doesn't track detailed convergence yet)
  std::vector<int16_t> gotStatuses(systemDevice.energyOuts.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotStatuses.data(),
                       bfgsMinimizer.statuses_.data(),
                       gotStatuses.size() * sizeof(int16_t),
                       cudaMemcpyDeviceToHost));
  EXPECT_THAT(gotStatuses, ::testing::Pointwise(::testing::Eq(), std::vector<int16_t>(numMols, 1)));
}

TEST_P(BFGSMinimizerBackendTest, E2EMinimizationMultiSystemSameMolMatchesConverged) {
  const nvMolKit::BfgsBackend backend  = GetParam();
  const int                   numMols  = 10;
  const int                   maxIters = 100;
  setUpMMFFSystems(numMols, true);

  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(/*dim=*/3, nvMolKit::DebugLevel::STEPWISE, true, nullptr, backend);

  minimizeMMFF(bfgsMinimizer, maxIters, 1e-4, systemHost, systemDevice);

  std::vector<double> refEnergies;
  for (auto& mol : mols) {
    RDKit::MMFF::MMFFOptimizeMolecule(*mol, maxIters, "MMFF94", 100.0);
    auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
    std::unique_ptr<ForceFields::ForceField> molFF(RDKit::MMFF::constructForceField(*mol, molProps.get()));
    refEnergies.push_back(molFF->calcEnergy());
  }

  std::vector<double> gotEnergies(systemDevice.energyOuts.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotEnergies.data(),
                       systemDevice.energyOuts.data(),
                       gotEnergies.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));

  EXPECT_THAT(gotEnergies, ::testing::Pointwise(::testing::DoubleNear(1e-4), refEnergies));

  std::vector<int16_t> gotStatuses(systemDevice.energyOuts.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotStatuses.data(),
                       bfgsMinimizer.statuses_.data(),
                       gotStatuses.size() * sizeof(int16_t),
                       cudaMemcpyDeviceToHost));
  EXPECT_THAT(gotStatuses, ::testing::Pointwise(::testing::Eq(), std::vector<int16_t>(numMols, 0)));
}

TEST_P(BFGSMinimizerBackendTest, E2EMinimizationMultiSystemMultiMolsMatchesConverged) {
  const nvMolKit::BfgsBackend backend  = GetParam();
  const int                   numMols  = 250;
  const int                   maxIters = 1000;
  setUpMMFFSystems(numMols, false);

  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(/*dim=*/3, nvMolKit::DebugLevel::STEPWISE, true, nullptr, backend);

  minimizeMMFF(bfgsMinimizer, maxIters, 1e-4, systemHost, systemDevice);

  std::vector<double> refEnergies;
  for (auto& mol : mols) {
    RDKit::MMFF::MMFFOptimizeMolecule(*mol, maxIters, "MMFF94", 100.0);
    auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
    std::unique_ptr<ForceFields::ForceField> molFF(RDKit::MMFF::constructForceField(*mol, molProps.get()));
    refEnergies.push_back(molFF->calcEnergy());
  }

  std::vector<double> gotEnergies(systemDevice.energyOuts.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotEnergies.data(),
                       systemDevice.energyOuts.data(),
                       gotEnergies.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));

  EXPECT_THAT(gotEnergies, ::testing::Pointwise(::testing::DoubleNear(1e-2), refEnergies));

  double avergedEnergyDiff = 0.0;
  for (size_t i = 0; i < gotEnergies.size(); ++i) {
    avergedEnergyDiff += fabs(gotEnergies[i] - refEnergies[i]);
  }
  avergedEnergyDiff /= static_cast<double>(gotEnergies.size());
  EXPECT_NEAR(avergedEnergyDiff, 0.0, 1e-4)
    << "Average energy difference between RDKit and nvMolKit minimizations is too large, despite max delta being acceptable";
  std::vector<int16_t> gotStatuses(systemDevice.energyOuts.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotStatuses.data(),
                       bfgsMinimizer.statuses_.data(),
                       gotStatuses.size() * sizeof(int16_t),
                       cudaMemcpyDeviceToHost));
  EXPECT_THAT(gotStatuses, ::testing::Pointwise(::testing::Eq(), std::vector<int16_t>(numMols, 0)));
}

TEST_P(BFGSMinimizerBackendTest, E2EMinimizationLargePathMatches) {
  const nvMolKit::BfgsBackend backend = GetParam();
  getMols(getTestDataFolderPath() + "/60plus_atom_mols.sdf", mols, 1);
  setUpCommon();
  const int                    maxIters = 400;
  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(/*dim=*/3, nvMolKit::DebugLevel::STEPWISE, true, nullptr, backend);

  minimizeMMFF(bfgsMinimizer, maxIters, 1e-4, systemHost, systemDevice);

  std::vector<double> refEnergies;
  for (auto& mol : mols) {
    RDKit::MMFF::MMFFOptimizeMolecule(*mol, maxIters, "MMFF94", 100.0);
    auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
    std::unique_ptr<ForceFields::ForceField> molFF(RDKit::MMFF::constructForceField(*mol, molProps.get()));
    refEnergies.push_back(molFF->calcEnergy());
  }

  std::vector<double> gotEnergies(systemDevice.energyOuts.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotEnergies.data(),
                       systemDevice.energyOuts.data(),
                       gotEnergies.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));

  EXPECT_THAT(gotEnergies, ::testing::Pointwise(::testing::DoubleNear(1e-4), refEnergies));

  std::vector<int16_t> gotStatuses(systemDevice.energyOuts.size());
  ASSERT_EQ(0,
            cudaMemcpy(gotStatuses.data(),
                       bfgsMinimizer.statuses_.data(),
                       gotStatuses.size() * sizeof(int16_t),
                       cudaMemcpyDeviceToHost));
  EXPECT_THAT(gotStatuses, ::testing::Pointwise(::testing::Eq(), std::vector<int16_t>(1, 0)));
}

template <bool computeLastDim>
__global__ void quarticEFunc(const int numTerms, const int* outIdx, const double* positions, double* energies) {
  constexpr int dim = 4;
  const int     idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numTerms)
    return;

  constexpr int iterMax    = computeLastDim ? 4 : 3;
  double        energyTerm = 0.0;
  for (int i = 0; i < iterMax; ++i) {
    const int    posIdx  = dim * idx + i;
    const double wantVal = posIdx;
    const double gotVal  = positions[posIdx];
    double       diff    = gotVal - wantVal;
    // Quartic potential: (x - x0)^4
    energyTerm += diff * diff * diff * diff;
  }
  atomicAdd(&energies[outIdx[idx]], energyTerm);
}

template <bool computeLastDim> __global__ void quarticGFunc(const int numTerms, const double* positions, double* grad) {
  constexpr int dim = 4;
  const int     idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numTerms)
    return;

  constexpr int iterMax = computeLastDim ? 4 : 3;
  for (int i = 0; i < iterMax; ++i) {
    const int    posIdx  = dim * idx + i;
    const double wantVal = posIdx;
    const double gotVal  = positions[posIdx];
    double       diff    = gotVal - wantVal;
    // Gradient of quartic: 4 * (x - x0)^3
    grad[posIdx]         = 4.0 * diff * diff * diff;
  }
}

class QuarticBatchedForcefield final : public nvMolKit::BatchedForcefield {
 public:
  QuarticBatchedForcefield(const std::vector<int>& atomStartsHost,
                           const int*              atomStartsDevice,
                           const int               numTerms,
                           const int*              outIdx,
                           const bool              computeLastDim,
                           const int               numBlocks,
                           const int               blockSize)
      : BatchedForcefield(nvMolKit::ForceFieldType::DG, 4, atomStartsHost, atomStartsDevice),
        numTerms_(numTerms),
        outIdx_(outIdx),
        computeLastDim_(computeLastDim),
        numBlocks_(numBlocks),
        blockSize_(blockSize) {}

  cudaError_t computeEnergy(double*        energyOuts,
                            const double*  positions,
                            const uint8_t* activeSystemMask = nullptr,
                            cudaStream_t   stream           = nullptr) override {
    if (activeSystemMask != nullptr) {
      return cudaErrorNotSupported;
    }
    if (computeLastDim_) {
      quarticEFunc<true><<<numBlocks_, blockSize_, 0, stream>>>(numTerms_, outIdx_, positions, energyOuts);
    } else {
      quarticEFunc<false><<<numBlocks_, blockSize_, 0, stream>>>(numTerms_, outIdx_, positions, energyOuts);
    }
    return cudaGetLastError();
  }

  cudaError_t computeGradients(double*        grad,
                               const double*  positions,
                               const uint8_t* activeSystemMask = nullptr,
                               cudaStream_t   stream           = nullptr) override {
    if (activeSystemMask != nullptr) {
      return cudaErrorNotSupported;
    }
    if (computeLastDim_) {
      quarticGFunc<true><<<numBlocks_, blockSize_, 0, stream>>>(numTerms_, positions, grad);
    } else {
      quarticGFunc<false><<<numBlocks_, blockSize_, 0, stream>>>(numTerms_, positions, grad);
    }
    return cudaGetLastError();
  }

 private:
  int        numTerms_       = 0;
  const int* outIdx_         = nullptr;
  bool       computeLastDim_ = false;
  int        numBlocks_      = 0;
  int        blockSize_      = 0;
};

class BFGSMinimizerHarmonicTestFixture : public ::testing::Test {
 protected:
  void setUpSystems(bool computeLastDim, int seed = 42) {
    computeLastDim_                     = computeLastDim;
    constexpr int          dim          = 4;
    const std::vector<int> atomStarts   = {0, 3, 10, 12};
    const std::vector<int> writeIndices = {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2};

    atomStarts_    = atomStarts;
    writeIndices_  = writeIndices;
    dim_           = dim;
    numSystems_    = atomStarts.size() - 1;
    totalNumAtoms_ = atomStarts.back();
    numGradTerms_  = dim * totalNumAtoms_;

    std::mt19937 rng(std::random_device{}());
    rng.seed(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    positions_.resize(totalNumAtoms_ * dim);
    for (int i = 0; i < numGradTerms_; ++i) {
      positions_[i] = static_cast<double>(i) + dist(rng);
    }

    // Set up device vectors
    atomStartsDevice_.resize(atomStarts.size());
    energyOutsDevice_.resize(numSystems_);
    gradDevice_.resize(numGradTerms_);
    positionsDevice_.resize(numGradTerms_);
    writeIndicesDevice_.resize(writeIndices.size());

    atomStartsDevice_.setFromVector(atomStarts);
    energyOutsDevice_.zero();
    gradDevice_.zero();
    positionsDevice_.setFromVector(positions_);
    writeIndicesDevice_.setFromVector(writeIndices);

    blockSize_ = 128;
    numBlocks_ = totalNumAtoms_ / blockSize_ + 1;
  }

  QuarticBatchedForcefield makeForcefield() const {
    return QuarticBatchedForcefield(atomStarts_,
                                    atomStartsDevice_.data(),
                                    static_cast<int>(writeIndices_.size()),
                                    writeIndicesDevice_.data(),
                                    computeLastDim_,
                                    numBlocks_,
                                    blockSize_);
  }

  void verifyPositions(const std::vector<double>& gotPositions, double tolerance = 1e-3) {
    for (int i = 0; i < numGradTerms_; ++i) {
      if ((i + 1) % 4 == 0 && !computeLastDim_) {
        continue;  // skip last dimension if not computing it
      }
      const double wantVal = i;
      const double gotVal  = gotPositions[i];
      EXPECT_NEAR(gotVal, wantVal, tolerance) << "Mismatch at index " << i;
    }
  }

  std::vector<double> getPositionsFromDevice() {
    std::vector<double> gotPositions(totalNumAtoms_ * dim_);
    positionsDevice_.copyToHost(gotPositions);
    EXPECT_EQ(cudaDeviceSynchronize(), 0);
    return gotPositions;
  }

  std::vector<int16_t> getStatusesFromDevice(const nvMolKit::BfgsBatchMinimizer& minimizer) {
    std::vector<int16_t> gotStatuses(numSystems_);
    EXPECT_EQ(0,
              cudaMemcpy(gotStatuses.data(),
                         minimizer.statuses_.data(),
                         gotStatuses.size() * sizeof(int16_t),
                         cudaMemcpyDeviceToHost));
    return gotStatuses;
  }

  // Member variables
  bool computeLastDim_;
  int  dim_;
  int  numSystems_;
  int  totalNumAtoms_;
  int  numGradTerms_;
  int  blockSize_;
  int  numBlocks_;

  std::vector<int>    atomStarts_;
  std::vector<int>    writeIndices_;
  std::vector<double> positions_;

  nvMolKit::AsyncDeviceVector<int>    atomStartsDevice_;
  nvMolKit::AsyncDeviceVector<double> energyOutsDevice_;
  nvMolKit::AsyncDeviceVector<double> gradDevice_;
  nvMolKit::AsyncDeviceVector<double> positionsDevice_;
  nvMolKit::AsyncDeviceVector<int>    writeIndicesDevice_;
};

class BFGSMinimizerTest4DTest : public BFGSMinimizerHarmonicTestFixture, public ::testing::WithParamInterface<bool> {};

TEST_P(BFGSMinimizerTest4DTest, BFGSMinimizer4DQuartic) {
  const bool computeLastDim = GetParam();

  // Set up harmonic system using the shared fixture
  setUpSystems(computeLastDim);

  std::vector<double> refPositions = getPositionsFromDevice();

  // Run minimization
  nvMolKit::BfgsBatchMinimizer minimizer(dim_, nvMolKit::DebugLevel::STEPWISE, false);
  auto                         forcefield = makeForcefield();

  minimizer.minimize(400, 1e-5, forcefield, positionsDevice_, gradDevice_, energyOutsDevice_, nullptr);

  std::vector<double> gotPositions = getPositionsFromDevice();
  verifyPositions(gotPositions, 0.1);

  // RDKit version.
  std::vector<double>              rdkEnergies;
  std::vector<std::vector<double>> rdkFinalPositions;
  std::vector<int>                 rdkStatuses;

  // Create CPU functors for the quartic potential
  auto createQuarticEnergyFunc = [this](int systemIdx) {
    return [this, systemIdx](const double* pos) -> double {
      const int start   = atomStarts_[systemIdx];
      const int end     = atomStarts_[systemIdx + 1];
      const int iterMax = computeLastDim_ ? 4 : 3;

      double totalEnergy = 0.0;
      for (int atomIdx = start; atomIdx < end; ++atomIdx) {
        for (int dimIdx = 0; dimIdx < iterMax; ++dimIdx) {
          // Global position index in the full system
          const int globalPosIdx = dim_ * atomIdx + dimIdx;
          // Local position index in this system's position array
          const int localPosIdx  = (atomIdx - start) * dim_ + dimIdx;

          const double wantVal = static_cast<double>(globalPosIdx);  // Target is the global position index
          const double gotVal  = pos[localPosIdx];
          const double diff    = gotVal - wantVal;
          // Quartic potential: (x - x0)^4
          totalEnergy += diff * diff * diff * diff;
        }
      }
      return totalEnergy;
    };
  };

  auto createQuarticGradientFunc = [this](int systemIdx) {
    return [this, systemIdx](const double* pos, double* grad) -> double {
      const int start   = atomStarts_[systemIdx];
      const int end     = atomStarts_[systemIdx + 1];
      const int iterMax = computeLastDim_ ? 4 : 3;

      // Zero out gradient (calculate size inline)
      const int systemSize = (end - start) * dim_;
      std::fill(grad, grad + systemSize, 0.0);

      double gradMagnitudeSquared = 0.0;
      for (int atomIdx = start; atomIdx < end; ++atomIdx) {
        for (int dimIdx = 0; dimIdx < iterMax; ++dimIdx) {
          // Global position index in the full system
          const int globalPosIdx = dim_ * atomIdx + dimIdx;
          // Local position index in this system's position array
          const int localPosIdx  = (atomIdx - start) * dim_ + dimIdx;

          const double wantVal       = static_cast<double>(globalPosIdx);  // Target is the global position index
          const double gotVal        = pos[localPosIdx];
          const double diff          = gotVal - wantVal;
          // Gradient of quartic: 4 * (x - x0)^3
          const double gradComponent = 4.0 * diff * diff * diff;
          grad[localPosIdx]          = gradComponent;
          gradMagnitudeSquared += gradComponent * gradComponent;
        }
      }
      // Return gradient magnitude (RDKit expects this as a scale factor)
      return std::sqrt(gradMagnitudeSquared);
    };
  };

  // Run RDKit BFGS for each system
  for (int i = 0; i < numSystems_; ++i) {
    const int start      = atomStarts_[i];
    const int end        = atomStarts_[i + 1];
    const int systemSize = (end - start) * dim_;

    // Extract initial positions for this system
    std::vector<double> rdkPositions(systemSize);
    std::copy(positions_.begin() + start * dim_, positions_.begin() + end * dim_, rdkPositions.begin());

    // Create functors for this system
    auto energyFunc = createQuarticEnergyFunc(i);
    auto gradFunc   = createQuarticGradientFunc(i);

    // Print initial energies

    // Run RDKit BFGS minimization
    unsigned int numIters = 0;
    double       funcVal  = 0.0;
    int          status   = BFGSOpt::minimize(systemSize,           // dimension
                                   rdkPositions.data(),  // positions
                                   1e-5,                 // gradient tolerance
                                   numIters,             // number of iterations (output)
                                   funcVal,              // final function value (output)
                                   energyFunc,           // energy function
                                   gradFunc,             // gradient function
                                   0,                    // snapshot frequency
                                   nullptr,              // snapshot vector
                                   1e-6,                 // function tolerance
                                   400                   // max iterations
    );

    rdkEnergies.push_back(funcVal);
    rdkFinalPositions.push_back(rdkPositions);
    rdkStatuses.push_back(status);
  }

  // Compare results
  std::vector<double> gotEnergies(numSystems_);
  energyOutsDevice_.copyToHost(gotEnergies);

  // gotPositions is already declared above
  std::vector<int16_t> gotStatuses = getStatusesFromDevice(minimizer);

  // Verify energies match
  EXPECT_THAT(gotEnergies, ::testing::Pointwise(::testing::DoubleNear(1e-3), rdkEnergies));
  // Verify final positions match (system by system)
  for (int i = 0; i < numSystems_; ++i) {
    const int start = atomStarts_[i];
    const int end   = atomStarts_[i + 1];

    std::vector<double> systemGotPositions(gotPositions.begin() + start * dim_, gotPositions.begin() + end * dim_);

    EXPECT_THAT(systemGotPositions, ::testing::Pointwise(::testing::DoubleNear(1e-5), rdkFinalPositions[i]))
      << "Position mismatch for system " << i;
  }

  // Verify convergence status (RDKit: 0=success, 1=too many iterations; our system: 0=converged, 1=unconverged)
  for (int i = 0; i < numSystems_; ++i) {
    if (rdkStatuses[i] == 0) {  // RDKit converged
      EXPECT_EQ(gotStatuses[i], 0) << "System " << i << " should have converged";
    }
  }
}

TEST_F(BFGSMinimizerHarmonicTestFixture, MultipleMinimizeCallsConvergedSystemUnchanged) {
  // Set up harmonic system
  setUpSystems(false);

  // First, fully converge the system
  nvMolKit::BfgsBatchMinimizer minimizer(dim_, nvMolKit::DebugLevel::STEPWISE, false);
  auto                         forcefield = makeForcefield();

  minimizer.minimize(50,  // Assuming full convergence happens after 50 steps
                     1e-3,
                     forcefield,
                     positionsDevice_,
                     gradDevice_,
                     energyOutsDevice_,
                     nullptr);

  // Verify convergence
  std::vector<int16_t> statuses = getStatusesFromDevice(minimizer);
  EXPECT_THAT(statuses, ::testing::Each(0));  // expect all converged

  std::vector<double> energiesBefore(energyOutsDevice_.size());
  energyOutsDevice_.copyToHost(energiesBefore);
  cudaStreamSynchronize(energyOutsDevice_.stream());
  ASSERT_THAT(energiesBefore, ::testing::Each(::testing::DoubleNear(0.0, 1e-5)));
  // Call minimize again on the converged system
  minimizer.minimize(50, 1e-4, forcefield, positionsDevice_, gradDevice_, energyOutsDevice_, nullptr);

  std::vector<double> energiesAfter(energyOutsDevice_.size());
  energyOutsDevice_.copyToHost(energiesAfter);
  cudaStreamSynchronize(energyOutsDevice_.stream());
  ASSERT_THAT(energiesAfter, ::testing::Each(::testing::DoubleNear(0.0, 1e-5)));
}

TEST_F(BFGSMinimizerHarmonicTestFixture, MultipleMinimizeCallsEquivalentToSingleCall) {
  // Set up harmonic system
  setUpSystems(false);

  // Test Case 1: Single minimize call with N iterations
  nvMolKit::BfgsBatchMinimizer minimizer1(dim_, nvMolKit::DebugLevel::STEPWISE, false);
  auto                         forcefield1 = makeForcefield();

  // A few of the systems will converge in 12 steps.
  minimizer1.minimize(50,  // N iterations
                      1e-3,
                      forcefield1,
                      positionsDevice_,
                      gradDevice_,
                      energyOutsDevice_,
                      nullptr);

  std::vector<double>  singleCallPositions = getPositionsFromDevice();
  std::vector<int16_t> singleCallStatuses  = getStatusesFromDevice(minimizer1);

  energyOutsDevice_.zero();
  CHECK_CUDA_RETURN(forcefield1.computeEnergy(energyOutsDevice_.data(), positionsDevice_.data()));
  std::vector<double> singleCallEnergies(energyOutsDevice_.size());
  energyOutsDevice_.copyToHost(singleCallEnergies);

  // Test Case 2: Two minimize calls with N/2 iterations each
  // Reset the system to initial state
  setUpSystems(false);  // This will reset positions to initial state

  nvMolKit::BfgsBatchMinimizer minimizer2(dim_, nvMolKit::DebugLevel::STEPWISE, false);
  auto                         forcefield2 = makeForcefield();

  // First part of iterations
  minimizer2.minimize(5, 1e-4, forcefield2, positionsDevice_, gradDevice_, energyOutsDevice_, nullptr);

  // Second half of iterations
  minimizer2.minimize(45, 1e-4, forcefield2, positionsDevice_, gradDevice_, energyOutsDevice_, nullptr);

  std::vector<double>  doubleCallPositions = getPositionsFromDevice();
  std::vector<int16_t> doubleCallStatuses  = getStatusesFromDevice(minimizer2);

  energyOutsDevice_.zero();
  CHECK_CUDA_RETURN(forcefield2.computeEnergy(energyOutsDevice_.data(), positionsDevice_.data()));
  std::vector<double> doubleCallEnergies(energyOutsDevice_.size());
  energyOutsDevice_.copyToHost(doubleCallEnergies);

  // Verify that both approaches produce the same results
  EXPECT_THAT(doubleCallStatuses, ::testing::Pointwise(::testing::Eq(), singleCallStatuses));
  EXPECT_THAT(doubleCallEnergies, ::testing::Pointwise(::testing::DoubleNear(1e-3), singleCallEnergies));
}

TEST_P(BFGSMinimizerBackendTest, ReuseMinimizer_VariedSizes) {
  const nvMolKit::BfgsBackend backend = GetParam();

  // System 1: Small (1 molecule)
  setUpMMFFSystems(1, false);
  BatchedMolecularSystemHost    system1Host   = systemHost;
  BatchedMolecularDeviceBuffers system1Device = std::move(systemDevice);

  // System 2: Medium (5 molecules) - clear fixture state first
  mols.clear();
  systemHost   = BatchedMolecularSystemHost();
  systemDevice = BatchedMolecularDeviceBuffers();
  setUpMMFFSystems(5, true);
  BatchedMolecularSystemHost    system2Host   = systemHost;
  BatchedMolecularDeviceBuffers system2Device = std::move(systemDevice);

  // System 3: Large (20 molecules) - clear fixture state first
  mols.clear();
  systemHost   = BatchedMolecularSystemHost();
  systemDevice = BatchedMolecularDeviceBuffers();
  setUpMMFFSystems(20, true);
  BatchedMolecularSystemHost    system3Host   = systemHost;
  BatchedMolecularDeviceBuffers system3Device = std::move(systemDevice);

  constexpr int maxIters = 50;

  // Get reference results with fresh minimizers
  std::vector<std::vector<double>> referenceEnergies(3);
  std::vector<std::vector<double>> referencePositions(3);

  // Reference for system 1
  {
    nvMolKit::BfgsBatchMinimizer minimizer(3, nvMolKit::DebugLevel::NONE, true, nullptr, backend);
    minimizeMMFF(minimizer, maxIters, 1e-4, system1Host, system1Device);
    referenceEnergies[0].resize(system1Device.energyOuts.size());
    system1Device.energyOuts.copyToHost(referenceEnergies[0]);
    referencePositions[0].resize(system1Device.positions.size());
    system1Device.positions.copyToHost(referencePositions[0]);
  }

  // Reference for system 2
  {
    nvMolKit::BfgsBatchMinimizer minimizer(3, nvMolKit::DebugLevel::NONE, true, nullptr, backend);
    minimizeMMFF(minimizer, maxIters, 1e-4, system2Host, system2Device);
    referenceEnergies[1].resize(system2Device.energyOuts.size());
    system2Device.energyOuts.copyToHost(referenceEnergies[1]);
    referencePositions[1].resize(system2Device.positions.size());
    system2Device.positions.copyToHost(referencePositions[1]);
  }

  // Reference for system 3
  {
    nvMolKit::BfgsBatchMinimizer minimizer(3, nvMolKit::DebugLevel::NONE, true, nullptr, backend);
    minimizeMMFF(minimizer, maxIters, 1e-4, system3Host, system3Device);
    referenceEnergies[2].resize(system3Device.energyOuts.size());
    system3Device.energyOuts.copyToHost(referenceEnergies[2]);
    referencePositions[2].resize(system3Device.positions.size());
    system3Device.positions.copyToHost(referencePositions[2]);
  }

  // Reset all systems to initial state (clear fixture state first)
  mols.clear();
  systemHost   = BatchedMolecularSystemHost();
  systemDevice = BatchedMolecularDeviceBuffers();

  setUpMMFFSystems(1, false);
  system1Host   = systemHost;
  system1Device = std::move(systemDevice);

  mols.clear();
  systemHost   = BatchedMolecularSystemHost();
  systemDevice = BatchedMolecularDeviceBuffers();

  setUpMMFFSystems(5, true);
  system2Host   = systemHost;
  system2Device = std::move(systemDevice);

  mols.clear();
  systemHost   = BatchedMolecularSystemHost();
  systemDevice = BatchedMolecularDeviceBuffers();

  setUpMMFFSystems(20, true);
  system3Host   = systemHost;
  system3Device = std::move(systemDevice);

  // Reused minimizer: small -> medium -> large -> medium -> small
  nvMolKit::BfgsBatchMinimizer reusedMinimizer(3, nvMolKit::DebugLevel::NONE, true, nullptr, backend);

  // Minimize system 1 (small)
  minimizeMMFF(reusedMinimizer, maxIters, 1e-4, system1Host, system1Device);
  std::vector<double> energy1(system1Device.energyOuts.size());
  system1Device.energyOuts.copyToHost(energy1);
  EXPECT_THAT(energy1, ::testing::Pointwise(::testing::DoubleNear(1e-5), referenceEnergies[0]))
    << "System 1 (first run) energies should match reference";

  // Minimize system 2 (medium)
  minimizeMMFF(reusedMinimizer, maxIters, 1e-4, system2Host, system2Device);
  std::vector<double> energy2(system2Device.energyOuts.size());
  system2Device.energyOuts.copyToHost(energy2);
  EXPECT_THAT(energy2, ::testing::Pointwise(::testing::DoubleNear(1e-5), referenceEnergies[1]))
    << "System 2 (first run) energies should match reference";

  // Minimize system 3 (large)
  minimizeMMFF(reusedMinimizer, maxIters, 1e-4, system3Host, system3Device);
  std::vector<double> energy3(system3Device.energyOuts.size());
  system3Device.energyOuts.copyToHost(energy3);
  EXPECT_THAT(energy3, ::testing::Pointwise(::testing::DoubleNear(1e-5), referenceEnergies[2]))
    << "System 3 (first run) energies should match reference";

  // Reset systems and minimize again in reverse order to test going from large back to small
  mols.clear();
  systemHost   = BatchedMolecularSystemHost();
  systemDevice = BatchedMolecularDeviceBuffers();

  setUpMMFFSystems(5, true);
  system2Host   = systemHost;
  system2Device = std::move(systemDevice);

  mols.clear();
  systemHost   = BatchedMolecularSystemHost();
  systemDevice = BatchedMolecularDeviceBuffers();

  setUpMMFFSystems(1, false);
  system1Host   = systemHost;
  system1Device = std::move(systemDevice);

  // Minimize system 2 again (after system 3)
  minimizeMMFF(reusedMinimizer, maxIters, 1e-4, system2Host, system2Device);
  std::vector<double> energy2Second(system2Device.energyOuts.size());
  system2Device.energyOuts.copyToHost(energy2Second);
  EXPECT_THAT(energy2Second, ::testing::Pointwise(::testing::DoubleNear(1e-5), referenceEnergies[1]))
    << "System 2 (second run) energies should match reference";

  // Minimize system 1 again (after system 2)
  minimizeMMFF(reusedMinimizer, maxIters, 1e-4, system1Host, system1Device);
  std::vector<double> energy1Second(system1Device.energyOuts.size());
  system1Device.energyOuts.copyToHost(energy1Second);
  EXPECT_THAT(energy1Second, ::testing::Pointwise(::testing::DoubleNear(1e-5), referenceEnergies[0]))
    << "System 1 (second run) energies should match reference";
}

INSTANTIATE_TEST_SUITE_P(BFGSMinimizer4DTest, BFGSMinimizerTest4DTest, ::testing::Values(false, true));

INSTANTIATE_TEST_SUITE_P(BFGSBackends,
                         BFGSMinimizerBackendTest,
                         ::testing::Values(nvMolKit::BfgsBackend::BATCHED,
                                           nvMolKit::BfgsBackend::PER_MOLECULE,
                                           nvMolKit::BfgsBackend::HYBRID),
                         [](const ::testing::TestParamInfo<nvMolKit::BfgsBackend>& info) {
                           switch (info.param) {
                             case nvMolKit::BfgsBackend::BATCHED:
                               return "Batched";
                             case nvMolKit::BfgsBackend::PER_MOLECULE:
                               return "PerMolecule";
                             case nvMolKit::BfgsBackend::HYBRID:
                               return "Hybrid";
                             default:
                               return "Unknown";
                           }
                         });
