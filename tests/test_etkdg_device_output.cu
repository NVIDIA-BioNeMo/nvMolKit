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
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <vector>

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

TEST(EmbedMoleculesDeviceOutput, RejectsPruningInDeviceMode) {
  auto ethanol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCO"));
  ASSERT_NE(ethanol, nullptr);

  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.useRandomCoords                      = true;
  params.pruneRmsThresh                       = 0.5;

  std::vector<RDKit::ROMol*> mols = {ethanol.get()};
  EXPECT_THROW(nvMolKit::embedMolecules(mols,
                                        params,
                                        1,
                                        -1,
                                        false,
                                        nullptr,
                                        singleThreadOptions(),
                                        nvMolKit::BfgsBackend::PER_MOLECULE,
                                        nvMolKit::CoordinateOutput::DEVICE),
               std::invalid_argument);
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
