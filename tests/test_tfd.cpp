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
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "src/tfd/tfd_cpu.h"
#include "src/tfd/tfd_gpu.h"

namespace {

constexpr double kTolerance = 1e-3;

void generateConformers(RDKit::ROMol& mol, int numConformers, int seed = 42) {
  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.randomSeed                           = seed;
  params.numThreads                           = 1;
  RDKit::DGeomHelpers::EmbedMultipleConfs(mol, numConformers, params);
}

}  // namespace

// ========== TFDGpuGenerator API Tests (validated against CPU reference) ==========

class TFDGeneratorTest : public ::testing::Test {};

TEST_F(TFDGeneratorTest, SingleMoleculeMatchesCPU) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 4);

  nvMolKit::TFDGpuGenerator generator;
  nvMolKit::TFDCpuGenerator cpuGenerator;

  auto gpuResult = generator.GetTFDMatrix(*mol);
  auto cpuResult = cpuGenerator.GetTFDMatrix(*mol);

  ASSERT_EQ(gpuResult.size(), cpuResult.size());
  for (size_t i = 0; i < gpuResult.size(); ++i) {
    EXPECT_NEAR(gpuResult[i], cpuResult[i], kTolerance) << "Mismatch at index " << i;
  }
}

TEST_F(TFDGeneratorTest, BatchProcessingMatchesCPU) {
  std::vector<std::unique_ptr<RDKit::RWMol>> mols;
  std::vector<const RDKit::ROMol*>           molPtrs;

  for (int i = 0; i < 3; ++i) {
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCC"));
    generateConformers(*mol, 3 + i);
    if (mol->getNumConformers() >= 2) {
      molPtrs.push_back(mol.get());
      mols.push_back(std::move(mol));
    }
  }

  ASSERT_GE(mols.size(), 2u);

  nvMolKit::TFDGpuGenerator generator;
  nvMolKit::TFDCpuGenerator cpuGenerator;

  auto gpuResults = generator.GetTFDMatrices(molPtrs);
  auto cpuResults = cpuGenerator.GetTFDMatrices(molPtrs);

  ASSERT_EQ(gpuResults.size(), cpuResults.size());
  for (size_t m = 0; m < gpuResults.size(); ++m) {
    ASSERT_EQ(gpuResults[m].size(), cpuResults[m].size());
    for (size_t i = 0; i < gpuResults[m].size(); ++i) {
      EXPECT_NEAR(gpuResults[m][i], cpuResults[m][i], kTolerance) << "Molecule " << m << " index " << i;
    }
  }
}

TEST_F(TFDGeneratorTest, GpuBufferMethodWorks) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCC"));
  ASSERT_NE(mol, nullptr);
  generateConformers(*mol, 4);

  std::vector<const RDKit::ROMol*> molPtrs = {mol.get()};

  nvMolKit::TFDGpuGenerator   gpuGenerator;
  nvMolKit::TFDComputeOptions options;

  auto gpuResult = gpuGenerator.GetTFDMatricesGpuBuffer(molPtrs, options);

  auto extracted = gpuResult.extractAll();
  ASSERT_EQ(extracted.size(), 1u);
  EXPECT_EQ(extracted[0].size(), 6u);  // 4*(4-1)/2 = 6
}

TEST_F(TFDGeneratorTest, EmptyInput) {
  nvMolKit::TFDGpuGenerator        generator;
  std::vector<const RDKit::ROMol*> emptyMols;

  auto results = generator.GetTFDMatrices(emptyMols);
  EXPECT_TRUE(results.empty());
}
