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

#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "src/conformer/device_conformer_pruning.h"
#include "src/conformer/device_coord_collector.h"
#include "src/conformer/device_coord_result.h"
#include "src/etkdg.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"

using namespace nvMolKit;

namespace {

nvMolKit::BatchHardwareOptions singleThreadOptions() {
  nvMolKit::BatchHardwareOptions options;
  options.preprocessingThreads = 1;
  options.batchSize            = 64;
  options.batchesPerGpu        = 1;
  options.gpuIds               = {0};
  return options;
}

template <typename T> std::vector<T> downloadDeviceVector(const AsyncDeviceVector<T>& vec) {
  std::vector<T> host(vec.size());
  if (!host.empty()) {
    vec.copyToHost(host);
    cudaCheckError(cudaStreamSynchronize(vec.stream()));
  }
  return host;
}

DeviceCoordResult makeDeviceResult(const std::vector<double>&  positions,
                                   const std::vector<int32_t>& molIndices,
                                   const int                   nMols) {
  // Tests pass flat, fixed-size conformers; build the matching device-side index arrays.
  const WithDevice     withDevice(0);
  const size_t         atomsPerConformer = positions.size() / (molIndices.size() * 3);
  std::vector<int32_t> atomStarts(molIndices.size() + 1);
  std::vector<int32_t> confIndices(molIndices.size(), 0);
  for (size_t i = 0; i < atomStarts.size(); ++i) {
    atomStarts[i] = static_cast<int32_t>(i * atomsPerConformer);
  }

  DeviceCoordResult result;
  result.gpuId       = 0;
  result.nMols       = nMols;
  result.positions   = AsyncDeviceVector<double>(positions.size());
  result.atomStarts  = AsyncDeviceVector<int32_t>(atomStarts.size());
  result.molIndices  = AsyncDeviceVector<int32_t>(molIndices.size());
  result.confIndices = AsyncDeviceVector<int32_t>(molIndices.size());

  result.positions.copyFromHost(positions);
  result.atomStarts.copyFromHost(atomStarts);
  result.molIndices.copyFromHost(molIndices);
  result.confIndices.copyFromHost(confIndices);
  cudaCheckError(cudaStreamSynchronize(nullptr));
  return result;
}

RDKit::DGeomHelpers::EmbedParameters pruningParams(const double threshold) {
  auto params                  = RDKit::DGeomHelpers::ETKDGv3;
  params.pruneRmsThresh        = threshold;
  params.useSymmetryForPruning = false;
  params.onlyHeavyAtomsForRMS  = false;
  return params;
}

}  // namespace

TEST(EmbedMoleculesDeviceOutput, EthanolDeviceModeShape) {
  auto ethanol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCO"));
  ASSERT_NE(ethanol, nullptr);
  const unsigned int nAtoms = ethanol->getNumAtoms();

  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.useRandomCoords                      = true;
  params.randomSeed                           = 12345;
  params.pruneRmsThresh                       = -1.0;

  std::vector<RDKit::ROMol*> mols   = {ethanol.get()};
  const auto                 result = nvMolKit::embedMolecules(mols,
                                               params,
                                               /*confsPerMolecule=*/1,
                                               -1,
                                               false,
                                               nullptr,
                                               singleThreadOptions(),
                                               nvMolKit::BfgsBackend::PER_MOLECULE,
                                               nvMolKit::CoordinateOutput::DEVICE,
                                               /*targetGpu=*/0);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->gpuId, 0);
  EXPECT_EQ(ethanol->getNumConformers(), 0u) << "DEVICE mode must not modify the host RDKit conformer list.";

  const auto positions  = downloadDeviceVector(result->positions);
  const auto atomStarts = downloadDeviceVector(result->atomStarts);
  const auto molIndices = downloadDeviceVector(result->molIndices);
  const auto confIdx    = downloadDeviceVector(result->confIndices);

  ASSERT_EQ(molIndices.size(), 1u);
  ASSERT_EQ(confIdx.size(), 1u);
  ASSERT_EQ(atomStarts.size(), 2u);
  EXPECT_EQ(molIndices[0], 0);
  EXPECT_EQ(confIdx[0], 0);
  EXPECT_EQ(atomStarts[0], 0);
  EXPECT_EQ(static_cast<size_t>(atomStarts[1]), nAtoms);
  ASSERT_EQ(positions.size(), static_cast<size_t>(nAtoms) * 3);

  // Sanity check: positions are finite and not all zero (ETKDG produced something usable).
  bool anyNonZero = false;
  for (const double pos : positions) {
    EXPECT_TRUE(std::isfinite(pos));
    if (std::abs(pos) > 1e-9) {
      anyNonZero = true;
    }
  }
  EXPECT_TRUE(anyNonZero);
}

TEST(EmbedMoleculesDeviceOutput, EmptyDeviceResultInitializesAtomStarts) {
  const WithDevice withDevice(0);
  ScopedStream     stream;

  std::vector<detail::DeviceCoordCollector> collectors(1);
  collectors[0].gpuId  = 0;
  collectors[0].stream = stream.stream();
  collectors[0].positions.setStream(stream.stream());

  const auto result     = detail::finalizeOnTarget(collectors, /*targetGpu=*/0, /*nMols=*/2);
  const auto atomStarts = downloadDeviceVector(result.atomStarts);

  EXPECT_EQ(result.gpuId, 0);
  EXPECT_EQ(result.nMols, 2);
  EXPECT_EQ(result.positions.size(), 0u);
  EXPECT_EQ(result.molIndices.size(), 0u);
  EXPECT_EQ(result.confIndices.size(), 0u);
  ASSERT_EQ(atomStarts.size(), 1u);
  EXPECT_EQ(atomStarts[0], 0);
}

TEST(EmbedMoleculesDeviceOutput, MultipleMoleculesProduceCorrectIndexing) {
  // Two distinct molecules in one batch. The CSR output must group conformers by global
  // mol index and report the right atom counts; the actual positions are produced by
  // ETKDG and we only check shape and that values are finite.
  auto methanol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CO"));
  auto propanol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCO"));
  ASSERT_NE(methanol, nullptr);
  ASSERT_NE(propanol, nullptr);

  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.useRandomCoords                      = true;
  params.randomSeed                           = 1;
  params.pruneRmsThresh                       = -1.0;

  std::vector<RDKit::ROMol*> mols   = {methanol.get(), propanol.get()};
  const auto                 result = nvMolKit::embedMolecules(mols,
                                               params,
                                               /*confsPerMolecule=*/2,
                                               -1,
                                               false,
                                               nullptr,
                                               singleThreadOptions(),
                                               nvMolKit::BfgsBackend::PER_MOLECULE,
                                               nvMolKit::CoordinateOutput::DEVICE,
                                               /*targetGpu=*/0);
  ASSERT_TRUE(result.has_value());

  const auto molIndices = downloadDeviceVector(result->molIndices);
  const auto confIdx    = downloadDeviceVector(result->confIndices);
  const auto atomStarts = downloadDeviceVector(result->atomStarts);
  const auto positions  = downloadDeviceVector(result->positions);

  ASSERT_EQ(molIndices.size(), 4u) << "Expected 2 mols x 2 confs = 4 conformers";
  std::array<int, 2> seenPerMol = {0, 0};
  for (size_t conformerIdx = 0; conformerIdx < molIndices.size(); ++conformerIdx) {
    const int molId = molIndices[conformerIdx];
    ASSERT_GE(molId, 0);
    ASSERT_LT(molId, 2);
    EXPECT_EQ(confIdx[conformerIdx], seenPerMol[static_cast<size_t>(molId)]);
    ++seenPerMol[static_cast<size_t>(molId)];
    const int natomsThisConf = atomStarts[conformerIdx + 1] - atomStarts[conformerIdx];
    if (molId == 0) {
      EXPECT_EQ(static_cast<unsigned int>(natomsThisConf), methanol->getNumAtoms());
    } else {
      EXPECT_EQ(static_cast<unsigned int>(natomsThisConf), propanol->getNumAtoms());
    }
  }
  EXPECT_EQ(seenPerMol[0], 2);
  EXPECT_EQ(seenPerMol[1], 2);
  ASSERT_EQ(positions.size(), static_cast<size_t>(2u * methanol->getNumAtoms() + 2u * propanol->getNumAtoms()) * 3u);
  for (const double pos : positions) {
    EXPECT_TRUE(std::isfinite(pos));
  }
}

TEST(EmbedMoleculesDeviceOutput, PrunesRigidMoleculeInDeviceMode) {
  auto benzene = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("c1ccccc1"));
  ASSERT_NE(benzene, nullptr);

  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.useRandomCoords                      = true;
  params.randomSeed                           = 12345;
  params.pruneRmsThresh                       = 0.5;

  std::vector<RDKit::ROMol*> mols   = {benzene.get()};
  const auto                 result = nvMolKit::embedMolecules(mols,
                                               params,
                                               5,
                                               -1,
                                               false,
                                               nullptr,
                                               singleThreadOptions(),
                                               nvMolKit::BfgsBackend::PER_MOLECULE,
                                               nvMolKit::CoordinateOutput::DEVICE);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(benzene->getNumConformers(), 0u);
  EXPECT_EQ(result->molIndices.size(), 1u);
  EXPECT_EQ(result->positions.size(), static_cast<size_t>(benzene->getNumAtoms()) * 3);
}

TEST(DeviceConformerPruning, PreservesGreedyOrderAcrossInterleavedMolecules) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CC"));
  ASSERT_NE(mol, nullptr);

  // RMSD is half the bond-length difference here. At threshold 0.75,
  // molecule 0 must keep lengths 1 and 3 even though length 2 conflicts
  // with both. Molecule 1 keeps only length 1.
  const std::vector<double> positions = {
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  // mol 0, conf 0
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  // mol 1, conf 0
    0.0, 0.0, 0.0, 2.0, 0.0, 0.0,  // mol 0, conf 1
    0.0, 0.0, 0.0, 1.5, 0.0, 0.0,  // mol 1, conf 1
    0.0, 0.0, 0.0, 3.0, 0.0, 0.0,  // mol 0, conf 2
  };
  const std::vector<int32_t> molIndices = {0, 1, 0, 1, 0};

  auto                       input  = makeDeviceResult(positions, molIndices, 2);
  auto                       params = pruningParams(0.75);
  std::vector<RDKit::ROMol*> mols   = {mol.get(), mol.get()};

  const auto result = detail::pruneDeviceConformers(std::move(input), mols, params);
  EXPECT_EQ(
    downloadDeviceVector(result.positions),
    (std::vector<double>{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0}));
  EXPECT_EQ(downloadDeviceVector(result.atomStarts), (std::vector<int32_t>{0, 2, 4, 6}));
  EXPECT_EQ(downloadDeviceVector(result.molIndices), (std::vector<int32_t>{0, 1, 0}));
  EXPECT_EQ(downloadDeviceVector(result.confIndices), (std::vector<int32_t>{0, 0, 1}));
}

TEST(DeviceConformerPruning, UsesSymmetryAtomMappings) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CC(C)C"));
  ASSERT_NE(mol, nullptr);

  auto params                  = pruningParams(1e-4);
  params.useSymmetryForPruning = true;

  // Atoms 0 and 2 are equivalent terminal carbons. Swapping their positions
  // should therefore produce the same conformer when symmetry is enabled.
  const std::vector<double> positions = {
    0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 0.1, 2.3, 0.2, 0.2, 0.4, 3.4,
    0.1, 2.3, 0.2, 1.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 3.4,
  };
  auto                       input = makeDeviceResult(positions, {0, 0}, 1);
  std::vector<RDKit::ROMol*> mols  = {mol.get()};

  const auto result = detail::pruneDeviceConformers(std::move(input), mols, params);
  EXPECT_EQ(result.molIndices.size(), 1u);
}

TEST(DeviceConformerPruning, HonorsHeavyAtomOnlyOption) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("C"));
  ASSERT_NE(mol, nullptr);
  RDKit::MolOps::addHs(*mol);
  ASSERT_EQ(mol->getNumAtoms(), 5u);

  const std::vector<double> positions = {
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0,
    0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0, -6.0, -7.0, -8.0,
  };
  const std::vector<int32_t> molIndices = {0, 0};
  std::vector<RDKit::ROMol*> mols       = {mol.get()};

  auto params                 = pruningParams(0.1);
  params.onlyHeavyAtomsForRMS = true;
  auto       heavyOnlyInput   = makeDeviceResult(positions, molIndices, 1);
  const auto heavyOnlyResult  = detail::pruneDeviceConformers(std::move(heavyOnlyInput), mols, params);
  EXPECT_EQ(heavyOnlyResult.molIndices.size(), 1u);

  params.onlyHeavyAtomsForRMS = false;
  auto       allAtomInput     = makeDeviceResult(positions, molIndices, 1);
  const auto allAtomResult    = detail::pruneDeviceConformers(std::move(allAtomInput), mols, params);
  EXPECT_EQ(allAtomResult.molIndices.size(), 2u);
}

TEST(DeviceConformerPruning, AllSurviveFastPathRenumbersInterleavedConformers) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CC"));
  ASSERT_NE(mol, nullptr);

  // Both molecules keep both conformers. Their interleaved input order checks
  // that the no-copy path still assigns per-molecule conformer indices.
  const std::vector<double> positions = {
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0,
  };
  const std::vector<int32_t> molIndices = {0, 1, 0, 1};
  auto                       input      = makeDeviceResult(positions, molIndices, 2);
  auto                       params     = pruningParams(0.75);
  std::vector<RDKit::ROMol*> mols       = {mol.get(), mol.get()};

  const auto result = detail::pruneDeviceConformers(std::move(input), mols, params);
  EXPECT_EQ(downloadDeviceVector(result.positions), positions);
  EXPECT_EQ(downloadDeviceVector(result.molIndices), molIndices);
  EXPECT_EQ(downloadDeviceVector(result.confIndices), (std::vector<int32_t>{0, 0, 1, 1}));
}

TEST(EmbedMoleculesDeviceOutput, MultipleConformersMatchPerMolIndices) {
  auto propane = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCC"));
  ASSERT_NE(propane, nullptr);

  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.useRandomCoords                      = true;
  params.randomSeed                           = 42;
  params.pruneRmsThresh                       = -1.0;

  std::vector<RDKit::ROMol*> mols   = {propane.get()};
  const auto                 result = nvMolKit::embedMolecules(mols,
                                               params,
                                               /*confsPerMolecule=*/3,
                                               -1,
                                               false,
                                               nullptr,
                                               singleThreadOptions(),
                                               nvMolKit::BfgsBackend::PER_MOLECULE,
                                               nvMolKit::CoordinateOutput::DEVICE,
                                               /*targetGpu=*/0);
  ASSERT_TRUE(result.has_value());
  const auto molIndices = downloadDeviceVector(result->molIndices);
  const auto confIdx    = downloadDeviceVector(result->confIndices);
  ASSERT_EQ(molIndices.size(), 3u);
  ASSERT_EQ(confIdx.size(), 3u);
  for (const auto idx : molIndices) {
    EXPECT_EQ(idx, 0);
  }
  std::vector<int32_t> sorted = {confIdx[0], confIdx[1], confIdx[2]};
  std::sort(sorted.begin(), sorted.end());
  EXPECT_EQ(sorted[0], 0);
  EXPECT_EQ(sorted[1], 1);
  EXPECT_EQ(sorted[2], 2);
}
