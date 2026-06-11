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
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ROMol.h>
#include <nanobench.h>

#include <filesystem>
#include <memory>

#include "rdkit_extensions/dist_geom_flattened_builder.h"
#include "src/embedder_utils.h"
#include "src/utils/cuda_error_check.h"
#include "tests/test_utils.h"

using nvMolKit::AsyncDeviceVector;
using namespace nvMolKit::DistGeom;

void benchRDKit(const std::vector<RDKit::ROMol*>& mols, const int size) {
  std::vector<std::unique_ptr<ForceFields::ForceField>> forceFields;
  std::vector<std::vector<double>>                      positions;
  std::vector<std::vector<double>>                      gradients;

  for (int i = 0; i < size; i++) {
    auto* mol     = mols[i];
    auto  options = RDKit::DGeomHelpers::ETKDGv3;

    // Generate force field using the reference function
    nvMolKit::detail::EmbedArgs                 eargs;
    std::vector<std::unique_ptr<RDGeom::Point>> pointPositions;
    forceFields.push_back(nvMolKit::DGeomHelpers::generateRDKitFF(*mol, options, eargs, pointPositions));

    // Store positions and gradients
    std::vector<double>& position = positions.emplace_back();
    for (unsigned int j = 0; j < mol->getNumAtoms(); ++j) {
      position.push_back(mol->getConformer().getAtomPos(j).x);
      position.push_back(mol->getConformer().getAtomPos(j).y);
      position.push_back(mol->getConformer().getAtomPos(j).z);
      // Add fourth dimension if force field dimension is 4
      if (forceFields[i]->dimension() == 4) {
        position.push_back(0.0);  // Fourth dimension value
      }
    }

    std::vector<double>& gradient = gradients.emplace_back();
    gradient.resize(forceFields[i]->dimension() * mol->getNumAtoms(), 0.0);
  }

  // Energy calculation benchmark
  ankerl::nanobench::Bench().run("RDKit ETKDG calc energy, nmols = " + std::to_string(size), [&] {
    for (int i = 0; i < size; i++) {
      ankerl::nanobench::doNotOptimizeAway(forceFields[i]->calcEnergy(positions[i].data()));
    }
  });

  // Gradient calculation benchmark
  ankerl::nanobench::Bench().run("RDKit ETKDG calc gradient, nmols = " + std::to_string(size), [&] {
    for (int i = 0; i < size; i++) {
      forceFields[i]->calcGrad(positions[i].data(), gradients[i].data());
    }
  });
}

void benchNvMolKitBatch(const std::vector<RDKit::ROMol*>& mols, const int size) {
  nvMolKit::DistGeom::BatchedMolecularSystemHost    systemHost;
  nvMolKit::DistGeom::BatchedMolecularDeviceBuffers systemDevice;
  std::vector<int>                                  atomStartsHost = {0};
  AsyncDeviceVector<int>                            atomStartsDevice;
  std::vector<double>                               positionsHost;
  AsyncDeviceVector<double>                         positionsDevice;

  for (int i = 0; i < size; i++) {
    auto*                                       mol = mols[i];
    nvMolKit::detail::EmbedArgs                 eargs;
    std::vector<std::unique_ptr<RDGeom::Point>> positions;
    std::unique_ptr<ForceFields::ForceField>    field;

    // Setup force field using ETKDGv3
    auto params = DGeomHelpers::ETKDGv3;
    nvMolKit::DGeomHelpers::setupRDKitFFWithPos(mol, params, field, eargs, positions);

    // Construct force field contributions and add to batch
    auto ffParams = constructForceFieldContribs(eargs.dim, *eargs.mmat, eargs.chiralCenters);
    addMoleculeToBatch(ffParams, eargs.posVec, systemHost, eargs.dim, atomStartsHost, positionsHost);
  }

  // Send data to device
  sendContribsAndIndicesToDevice(systemHost, systemDevice);
  sendContextToDevice(positionsHost, positionsDevice, atomStartsHost, atomStartsDevice);
  setupDeviceBuffers(systemHost, systemDevice, positionsHost, atomStartsHost.size() - 1);

  // Energy benchmark
  ankerl::nanobench::Bench().run("nvMolKit ETKDG calc energy, nmols = " + std::to_string(size), [&] {
    systemDevice.energyBuffer.zero();
    systemDevice.energyOuts.zero();
    ankerl::nanobench::doNotOptimizeAway(computeEnergy(systemDevice, atomStartsDevice, positionsDevice, 1.0, 0.1));
    ankerl::nanobench::doNotOptimizeAway(cudaDeviceSynchronize());
  });

  // Gradient benchmark
  ankerl::nanobench::Bench().run("nvMolKit ETKDG calc gradient, nmols = " + std::to_string(size), [&] {
    systemDevice.grad.zero();
    ankerl::nanobench::doNotOptimizeAway(computeGradients(systemDevice, atomStartsDevice, positionsDevice, 1.0, 0.1));
    ankerl::nanobench::doNotOptimizeAway(cudaDeviceSynchronize());
  });
}

int main() {
  const std::string                          fileName = getTestDataFolderPath() + "/MMFF94_hypervalent.sdf";
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(fileName, mols);
  std::vector<RDKit::ROMol*> molsPtrs;

  constexpr int maxMols = 10000;
  while (molsPtrs.size() < maxMols) {
    for (const auto& mol : mols) {
      molsPtrs.push_back(mol.get());
      if (molsPtrs.size() >= maxMols) {
        break;
      }
    }
  }
  constexpr std::array<int, 10> sizes = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
  for (const auto size : sizes) {
    benchRDKit(molsPtrs, size);
  }
  for (const auto size : sizes) {
    benchNvMolKitBatch(molsPtrs, size);
  }
  return 0;
}
