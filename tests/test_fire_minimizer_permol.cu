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

#include <gmock/gmock.h>
// clang-format off
// Bug in RDKit, includes need to be ordered.
#include <GraphMol/ROMol.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
// clang-format on
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "rdkit_extensions/mmff_flattened_builder.h"
#include "src/forcefields/mmff.h"
#include "src/forcefields/mmff_batched_forcefield.h"
#include "src/minimizer/fire_minimize_permol_kernels.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"
#include "tests/test_utils.h"

using nvMolKit::checkReturnCode;
using ::nvMolKit::MMFF::BatchedMolecularDeviceBuffers;
using ::nvMolKit::MMFF::BatchedMolecularSystemHost;

namespace {

void perturbConformer(RDKit::Conformer& conf, const float delta = 0.1, const int seed = 0) {
  std::mt19937                          gen(seed);
  std::uniform_real_distribution<float> dist(-delta, delta);
  for (unsigned int i = 0; i < conf.getNumAtoms(); ++i) {
    RDGeom::Point3D pos = conf.getAtomPos(i);
    pos.x += delta * dist(gen);
    pos.y += delta * dist(gen);
    pos.z += delta * dist(gen);
    conf.setAtomPos(i, pos);
  }
}

struct PerMolFireFixture {
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  BatchedMolecularSystemHost                 systemHost;
  BatchedMolecularDeviceBuffers              systemDevice;

  void setup(int numMols) {
    getMols(getTestDataFolderPath() + "/MMFF94_dative.sdf", mols, numMols);
    int runningSeed = 0;
    for (const auto& mol : mols) {
      perturbConformer(mol->getConformer(), 0.3, runningSeed++);
      std::vector<double> positions(3 * mol->getNumAtoms());
      for (unsigned int i = 0; i < mol->getNumAtoms(); ++i) {
        const RDGeom::Point3D pos = mol->getConformer().getAtomPos(i);
        positions[3 * i]          = pos.x;
        positions[3 * i + 1]      = pos.y;
        positions[3 * i + 2]      = pos.z;
      }
      const auto ffParams = nvMolKit::MMFF::constructForcefieldContribs(*mol);
      nvMolKit::MMFF::addMoleculeToBatch(ffParams, positions, systemHost);
    }
    nvMolKit::MMFF::sendContribsAndIndicesToDevice(systemHost, systemDevice);
    nvMolKit::MMFF::allocateIntermediateBuffers(systemHost, systemDevice);
    systemDevice.energyOuts.zero();
    systemDevice.positions.setFromVector(systemHost.positions);
    systemDevice.grad.resize(systemDevice.positions.size());
    systemDevice.grad.zero();
  }
};

std::vector<double> computeReferenceEnergies(const std::vector<std::unique_ptr<RDKit::ROMol>>& mols) {
  std::vector<double> energies;
  energies.reserve(mols.size());
  for (const auto& mol : mols) {
    const auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
    const std::unique_ptr<ForceFields::ForceField> molFF(RDKit::MMFF::constructForceField(*mol, molProps.get()));
    molFF->initialize();
    molFF->minimize(500, 1e-4);
    energies.push_back(molFF->calcEnergy());
  }
  return energies;
}

int maxAtomsInBatch(const std::vector<int>& atomStarts) {
  int maxAtoms = 0;
  for (size_t i = 0; i + 1 < atomStarts.size(); ++i) {
    maxAtoms = std::max(maxAtoms, atomStarts[i + 1] - atomStarts[i]);
  }
  return maxAtoms;
}

bool allConverged(const nvMolKit::AsyncDeviceVector<uint8_t>& statuses) {
  std::vector<uint8_t> statusHost(statuses.size());
  statuses.copyToHost(statusHost);
  cudaCheckError(cudaDeviceSynchronize());
  return std::all_of(statusHost.begin(), statusHost.end(), [](const uint8_t status) { return status == 0; });
}

}  // namespace

TEST(FireMinimizerPerMolMMFF, DirectLauncherConvergesNearReference) {
  PerMolFireFixture fixture;
  fixture.setup(/*numMols=*/4);
  const std::vector<double> refEnergies = computeReferenceEnergies(fixture.mols);
  const int                 numMols     = static_cast<int>(fixture.systemHost.indices.atomStarts.size()) - 1;

  nvMolKit::FireOptions options{};
  options.useMass               = false;
  options.stuckDetectionEnabled = false;
  options.gradTol               = 1e-3;
  options.dtInit                = 0.05;
  options.dMax                  = 0.2;

  std::vector<int> molIdsHost(numMols);
  std::iota(molIdsHost.begin(), molIdsHost.end(), 0);

  nvMolKit::AsyncDeviceVector<int> molIds;
  molIds.setFromVector(molIdsHost);

  nvMolKit::AsyncDeviceVector<double> velocities;
  velocities.resize(fixture.systemDevice.positions.size());
  velocities.zero();

  nvMolKit::AsyncDeviceVector<double> alphas;
  alphas.setFromVector(std::vector<double>(numMols, options.alphaInit));

  nvMolKit::AsyncDeviceVector<double> dts;
  dts.setFromVector(std::vector<double>(numMols, options.dtInit));

  nvMolKit::AsyncDeviceVector<int> nStepsPositive;
  nStepsPositive.resize(numMols);
  nStepsPositive.zero();

  nvMolKit::AsyncDeviceVector<uint8_t> statuses;
  statuses.setFromVector(std::vector<uint8_t>(numMols, 1));

  auto       terms          = nvMolKit::MMFF::toEnergyForceContribsDevicePtr(fixture.systemDevice);
  auto       systemIndices  = nvMolKit::MMFF::toBatchedIndicesDevicePtr(fixture.systemDevice);
  const bool hasConstraints = nvMolKit::MMFF::batchHasConstraints(fixture.systemDevice.contribs);

  bool converged = false;
  for (int extension = 0; extension < 5 && !converged; ++extension) {
    const cudaError_t err = nvMolKit::launchFirePerMolKernel(numMols,
                                                             molIds.data(),
                                                             maxAtomsInBatch(fixture.systemHost.indices.atomStarts),
                                                             fixture.systemDevice.indices.atomStarts.data(),
                                                             options,
                                                             /*numIters=*/2000,
                                                             options.gradTol,
                                                             terms,
                                                             systemIndices,
                                                             hasConstraints,
                                                             fixture.systemDevice.positions.data(),
                                                             fixture.systemDevice.grad.data(),
                                                             velocities.data(),
                                                             alphas.data(),
                                                             dts.data(),
                                                             nStepsPositive.data(),
                                                             /*masses=*/nullptr,
                                                             fixture.systemDevice.energyOuts.data(),
                                                             statuses.data());
    cudaCheckError(err);
    converged = allConverged(statuses);
  }

  std::vector<double> energiesHost(fixture.systemDevice.energyOuts.size());
  fixture.systemDevice.energyOuts.copyToHost(energiesHost);
  cudaCheckError(cudaDeviceSynchronize());

  for (size_t i = 0; i < energiesHost.size(); ++i) {
    const double tolerance = 1.0 + 0.05 * std::abs(refEnergies[i]);
    EXPECT_NEAR(energiesHost[i], refEnergies[i], tolerance) << "system " << i;
  }
}

TEST(FireMinimizerPerMolMMFF, DirectLauncherNoopsForEmptyBatch) {
  nvMolKit::FireOptions options{};
  const cudaError_t     err = nvMolKit::launchFirePerMolKernel(/*numMols=*/0,
                                                           /*molIds=*/nullptr,
                                                           /*maxAtoms=*/0,
                                                           /*atomStarts=*/nullptr,
                                                           options,
                                                           /*numIters=*/10,
                                                           /*gradTol=*/options.gradTol,
                                                           {},
                                                           {},
                                                           /*hasConstraints=*/false,
                                                           /*positions=*/nullptr,
                                                           /*grad=*/nullptr,
                                                           /*velocities=*/nullptr,
                                                           /*alphas=*/nullptr,
                                                           /*dts=*/nullptr,
                                                           /*nStepsPositive=*/nullptr,
                                                           /*masses=*/nullptr,
                                                           /*energyOuts=*/nullptr,
                                                           /*statuses=*/nullptr);
  EXPECT_EQ(err, cudaSuccess);
}
