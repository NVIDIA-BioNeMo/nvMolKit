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

#include <DataStructs/BitOps.h>
#include <gmock/gmock-matchers.h>
#include <GraphMol/ROMol.h>
#include <gtest/gtest.h>

#include <fstream>
#include <vector>

#include "src/morgan_fingerprint.h"
#include "src/similarity.h"
#include "src/testutils/mol_data.h"

std::vector<const RDKit::ROMol*> makeMolsView(const std::vector<std::unique_ptr<RDKit::ROMol>>& mols) {
  std::vector<const RDKit::ROMol*> molsView;
  for (const auto& mol : mols) {
    molsView.push_back(mol.get());
  }
  return molsView;
}

class FingerprintSimIntegrationTest : public ::testing::Test {};

TEST(FingerprintSimIntegrationTest, Basics) {
  constexpr int radius                      = 3;
  constexpr int fpSize                      = 1024;
  auto [mols, smiles]                       = nvMolKit::testing::loadNChemblMolecules(100, 128);
  std::vector<const RDKit::ROMol*> molsView = makeMolsView(mols);

  // Create generator
  auto generator    = nvMolKit::MorganFingerprintGenerator(radius, fpSize);
  auto refGenerator = std::unique_ptr<RDKit::FingerprintGenerator<std::uint32_t>>(
    RDKit::MorganFingerprint::getMorganGenerator<
      std::uint32_t>(radius, false, false, true, false, nullptr, nullptr, fpSize, {1, 2, 4, 8}, false, false));

  // Compute fingerprints
  nvMolKit::FingerprintComputeOptions options;
  options.backend      = nvMolKit::FingerprintComputeBackend::GPU;
  auto gpuFingerprints = generator.GetFingerprintsGpuBuffer<fpSize>(molsView, nullptr, options);
  std::vector<std::unique_ptr<ExplicitBitVect>> refFingerprints;
  for (size_t i = 0; i < mols.size(); i++) {
    const auto& mol            = mols[i];
    auto        refFingerprint = std::unique_ptr<ExplicitBitVect>(refGenerator->getFingerprint(*mol));
    ASSERT_NE(refFingerprint, nullptr);
    refFingerprints.push_back(std::move(refFingerprint));
  }

  // Compute simliarities from FP-0.
  std::vector<double> refSimilarities;
  refSimilarities.reserve(mols.size());
  for (size_t i = 0; i < mols.size(); i++) {
    refSimilarities.push_back(TanimotoSimilarity(*refFingerprints[0], *refFingerprints[i]));
  }
  const std::uint32_t*                       bitsOnePtr  = reinterpret_cast<std::uint32_t*>(gpuFingerprints.data());
  const cuda::std::span<const std::uint32_t> spanBitsOne = {bitsOnePtr, fpSize / (8 * sizeof(std::uint32_t))};
  const cuda::std::span<const std::uint32_t> spanBitsTwo = {bitsOnePtr,
                                                            mols.size() * fpSize / (8 * sizeof(std::uint32_t))};

  auto                gpuSimilarities = nvMolKit::crossTanimotoSimilarityGpuResult(spanBitsOne, spanBitsTwo, fpSize);
  std::vector<double> cpuSimilarities(gpuSimilarities.size());
  gpuSimilarities.copyToHost(cpuSimilarities);
  cudaDeviceSynchronize();
  EXPECT_THAT(cpuSimilarities, testing::Pointwise(testing::DoubleNear(1e-5), refSimilarities));
}
