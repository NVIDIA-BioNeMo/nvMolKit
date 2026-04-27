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
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "src/tfd/tfd_cpu.h"
#include "src/tfd/tfd_gpu.h"

namespace {

constexpr double kTolerance = 1e-3;  // GPU has lower precision due to float

//! Generate conformers for a molecule using RDKit
void generateConformers(RDKit::ROMol& mol, int numConformers, int seed = 42) {
  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.randomSeed                           = seed;
  params.numThreads                           = 1;
  RDKit::DGeomHelpers::EmbedMultipleConfs(mol, numConformers, params);
}

}  // namespace

// ========== GPU Generator Tests ==========

class TFDGpuTest : public ::testing::Test {
 protected:
  void SetUp() override { gpuGenerator_ = std::make_unique<nvMolKit::TFDGpuGenerator>(); }

  nvMolKit::TFDCpuGenerator                  cpuGenerator_;
  std::unique_ptr<nvMolKit::TFDGpuGenerator> gpuGenerator_;
};

TEST_F(TFDGpuTest, MatchesCPUReference) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 5);
  ASSERT_EQ(mol->getNumConformers(), 5);

  nvMolKit::TFDComputeOptions options;

  auto cpuTFD = cpuGenerator_.GetTFDMatrix(*mol, options);
  ASSERT_FALSE(cpuTFD.empty());

  auto gpuTFD = gpuGenerator_->GetTFDMatrix(*mol, options);

  ASSERT_EQ(gpuTFD.size(), cpuTFD.size());
  for (size_t i = 0; i < cpuTFD.size(); ++i) {
    EXPECT_NEAR(gpuTFD[i], cpuTFD[i], kTolerance) << "Mismatch at TFD index " << i;
  }
}

TEST_F(TFDGpuTest, MatchesCPUReferenceUnweighted) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 5);
  ASSERT_EQ(mol->getNumConformers(), 5);

  nvMolKit::TFDComputeOptions options;
  options.useWeights = false;

  auto cpuTFD = cpuGenerator_.GetTFDMatrix(*mol, options);
  ASSERT_FALSE(cpuTFD.empty());

  auto gpuTFD = gpuGenerator_->GetTFDMatrix(*mol, options);

  ASSERT_EQ(gpuTFD.size(), cpuTFD.size());
  for (size_t i = 0; i < cpuTFD.size(); ++i) {
    EXPECT_NEAR(gpuTFD[i], cpuTFD[i], kTolerance) << "Mismatch at TFD index " << i;
  }
}

TEST_F(TFDGpuTest, BatchMultipleMolecules) {
  // Mix of single-quartet and multi-quartet molecules in a single batch
  std::vector<std::string> smilesList = {
    "CCCC",                           // n-butane (single-quartet)
    "CCCCC",                          // n-pentane (single-quartet)
    "CC(C)CC",                        // isopentane (symmetric, 2 quartets)
    "C1CCCCC1",                       // cyclohexane (ring, 6 quartets)
    "c1ccccc1CC",                     // ethylbenzene (ring + symmetric)
    "CCO",                            // ethanol
    "CC(=O)O",                        // acetic acid
    "CC(=O)[C@H]1CCCC[C@@H]1CN",      // cyclohexane + ketone/amine substituents
    "C[C@@H](CN(C)C(=O)C#CCN)C1CC1",  // cyclopropane + triple bond
  };

  std::vector<std::unique_ptr<RDKit::RWMol>> mols;
  std::vector<const RDKit::ROMol*>           molPtrs;

  for (const auto& smiles : smilesList) {
    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(smiles));
    if (mol) {
      generateConformers(*mol, 4);
      if (mol->getNumConformers() >= 2) {
        molPtrs.push_back(mol.get());
        mols.push_back(std::move(mol));
      }
    }
  }

  ASSERT_GE(mols.size(), 3u);

  nvMolKit::TFDComputeOptions options;

  auto cpuResults = cpuGenerator_.GetTFDMatrices(molPtrs, options);
  auto gpuResults = gpuGenerator_->GetTFDMatrices(molPtrs, options);

  ASSERT_EQ(gpuResults.size(), cpuResults.size());
  for (size_t m = 0; m < cpuResults.size(); ++m) {
    ASSERT_EQ(gpuResults[m].size(), cpuResults[m].size()) << "Size mismatch for molecule " << m;

    for (size_t i = 0; i < cpuResults[m].size(); ++i) {
      EXPECT_NEAR(gpuResults[m][i], cpuResults[m][i], kTolerance) << "Mismatch at molecule " << m << " TFD index " << i;
    }
  }
}

TEST_F(TFDGpuTest, SingleConformer) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 1);
  ASSERT_EQ(mol->getNumConformers(), 1);

  nvMolKit::TFDComputeOptions options;
  auto                        tfdMatrix = gpuGenerator_->GetTFDMatrix(*mol, options);

  EXPECT_TRUE(tfdMatrix.empty());
}

TEST_F(TFDGpuTest, NoTorsionsMolecule) {
  // Methane has no rotatable bonds
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("C"));
  ASSERT_NE(mol, nullptr);

  generateConformers(*mol, 3);

  nvMolKit::TFDComputeOptions options;

  auto cpuTFD = cpuGenerator_.GetTFDMatrix(*mol, options);
  auto gpuTFD = gpuGenerator_->GetTFDMatrix(*mol, options);

  ASSERT_EQ(gpuTFD.size(), cpuTFD.size());
  for (size_t i = 0; i < gpuTFD.size(); ++i) {
    EXPECT_NEAR(gpuTFD[i], 0.0, kTolerance);
    EXPECT_NEAR(cpuTFD[i], 0.0, kTolerance);
  }
}

TEST_F(TFDGpuTest, GeneratorReuse) {
  // Verify that calling GetTFDMatrix twice on the same generator works.
  // The first call moves device_.tfdOutput into the result; the second call
  // must work correctly with the reallocated output buffer.
  auto mol1 = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
  ASSERT_NE(mol1, nullptr);
  generateConformers(*mol1, 3);

  auto mol2 = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCC"));
  ASSERT_NE(mol2, nullptr);
  generateConformers(*mol2, 4);

  nvMolKit::TFDComputeOptions options;

  auto gpuTFD1 = gpuGenerator_->GetTFDMatrix(*mol1, options);
  auto gpuTFD2 = gpuGenerator_->GetTFDMatrix(*mol2, options);

  auto cpuTFD1 = cpuGenerator_.GetTFDMatrix(*mol1, options);
  auto cpuTFD2 = cpuGenerator_.GetTFDMatrix(*mol2, options);

  ASSERT_EQ(gpuTFD1.size(), cpuTFD1.size());
  for (size_t i = 0; i < cpuTFD1.size(); ++i) {
    EXPECT_NEAR(gpuTFD1[i], cpuTFD1[i], kTolerance) << "First call mismatch at index " << i;
  }

  ASSERT_EQ(gpuTFD2.size(), cpuTFD2.size());
  for (size_t i = 0; i < cpuTFD2.size(); ++i) {
    EXPECT_NEAR(gpuTFD2[i], cpuTFD2[i], kTolerance) << "Second call mismatch at index " << i;
  }
}

// ========== GPU Result Extraction Tests ==========

TEST_F(TFDGpuTest, GpuResultExtraction) {
  std::vector<std::unique_ptr<RDKit::RWMol>> mols;
  std::vector<const RDKit::ROMol*>           molPtrs;

  auto mol1 = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
  generateConformers(*mol1, 3);
  molPtrs.push_back(mol1.get());
  mols.push_back(std::move(mol1));

  auto mol2 = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCCC"));
  generateConformers(*mol2, 4);
  molPtrs.push_back(mol2.get());
  mols.push_back(std::move(mol2));

  nvMolKit::TFDComputeOptions options;

  auto gpuResult = gpuGenerator_->GetTFDMatricesGpuBuffer(molPtrs, options);

  // Test extractMolecule
  auto mol0Extracted = gpuResult.extractMolecule(0);
  auto mol1Extracted = gpuResult.extractMolecule(1);

  // Verify sizes: 3 conformers = 3 pairs, 4 conformers = 6 pairs
  EXPECT_EQ(mol0Extracted.size(), 3u);  // 3*(3-1)/2 = 3
  EXPECT_EQ(mol1Extracted.size(), 6u);  // 4*(4-1)/2 = 6

  // Test extractAll
  auto allExtracted = gpuResult.extractAll();
  ASSERT_EQ(allExtracted.size(), 2u);
  EXPECT_EQ(allExtracted[0].size(), mol0Extracted.size());
  EXPECT_EQ(allExtracted[1].size(), mol1Extracted.size());

  // Values should match
  for (size_t i = 0; i < mol0Extracted.size(); ++i) {
    EXPECT_DOUBLE_EQ(allExtracted[0][i], mol0Extracted[i]);
  }
  for (size_t i = 0; i < mol1Extracted.size(); ++i) {
    EXPECT_DOUBLE_EQ(allExtracted[1][i], mol1Extracted[i]);
  }

  // Compare with CPU reference
  auto cpuResults = cpuGenerator_.GetTFDMatrices(molPtrs, options);
  for (size_t m = 0; m < allExtracted.size(); ++m) {
    ASSERT_EQ(allExtracted[m].size(), cpuResults[m].size());
    for (size_t i = 0; i < allExtracted[m].size(); ++i) {
      EXPECT_NEAR(allExtracted[m][i], cpuResults[m][i], kTolerance) << "Mismatch at molecule " << m << " index " << i;
    }
  }
}

TEST_F(TFDGpuTest, GpuResultExtractionOutOfRange) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCCC"));
  ASSERT_NE(mol, nullptr);
  generateConformers(*mol, 3);

  std::vector<const RDKit::ROMol*> molPtrs = {mol.get()};

  nvMolKit::TFDComputeOptions options;
  auto                        gpuResult = gpuGenerator_->GetTFDMatricesGpuBuffer(molPtrs, options);

  EXPECT_NO_THROW(gpuResult.extractMolecule(0));
  EXPECT_THROW(gpuResult.extractMolecule(-1), std::out_of_range);
  EXPECT_THROW(gpuResult.extractMolecule(1), std::out_of_range);
}

TEST_F(TFDGpuTest, AddHsMatchesCPU) {
  // Test with explicit hydrogens (AddHs) — produces more multi-quartet torsions.
  // Verifies GPU generator matches CPU for molecules with hydrogen atoms.
  const std::vector<std::string> testSmiles = {"CCCCC", "CC(C)CC"};

  for (const auto& smiles : testSmiles) {
    SCOPED_TRACE(std::string(smiles) + " (AddHs)");

    auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol(smiles));
    ASSERT_NE(mol, nullptr);

    RDKit::MolOps::addHs(*mol);
    generateConformers(*mol, 4, 42);
    ASSERT_EQ(mol->getNumConformers(), 4);

    nvMolKit::TFDComputeOptions options;

    auto cpuTFD = cpuGenerator_.GetTFDMatrix(*mol, options);
    ASSERT_FALSE(cpuTFD.empty());

    auto gpuTFD = gpuGenerator_->GetTFDMatrix(*mol, options);

    ASSERT_EQ(gpuTFD.size(), cpuTFD.size());
    for (size_t i = 0; i < cpuTFD.size(); ++i) {
      EXPECT_NEAR(gpuTFD[i], cpuTFD[i], kTolerance) << "Mismatch at TFD index " << i;
    }
  }
}
