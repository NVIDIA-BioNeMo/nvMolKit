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
#include <DistGeom/TriangleSmooth.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include "src/triangle_smooth.h"
#include "tests/test_utils.h"

namespace {
void initETKDG(const RDKit::ROMol*                         mol,
               const RDKit::DGeomHelpers::EmbedParameters& params,
               ForceFields::CrystalFF::CrystalFFDetails&   etkdgDetails) {
  PRECONDITION(mol, "bad molecule");
  const unsigned int nAtoms = mol->getNumAtoms();
  if (params.useExpTorsionAnglePrefs || params.useBasicKnowledge) {
    ForceFields::CrystalFF::getExperimentalTorsions(*mol,
                                                    etkdgDetails,
                                                    params.useExpTorsionAnglePrefs,
                                                    params.useSmallRingTorsions,
                                                    params.useMacrocycleTorsions,
                                                    params.useBasicKnowledge,
                                                    params.ETversion,
                                                    params.verbose);
    etkdgDetails.atomNums.resize(nAtoms);
    for (unsigned int i = 0; i < nAtoms; ++i) {
      etkdgDetails.atomNums[i] = mol->getAtomWithIdx(i)->getAtomicNum();
    }
  }
  etkdgDetails.boundsMatForceScaling = params.boundsMatForceScaling;
}
}  // namespace

class TriangleSmoothTest : public ::testing::Test {
 protected:
  void SetUp() override { testDataFolderPath_ = getTestDataFolderPath(); }

  void TearDown() override {
    molPtr_.reset();
    multipleMols_.clear();
  }

  // Helper function to load a single molecule
  void loadSingleMol() {
    const std::string mol2FilePath = testDataFolderPath_ / "rdkit_smallmol_1.mol2";
    ASSERT_TRUE(std::filesystem::exists(mol2FilePath)) << "Could not find " << mol2FilePath;
    molPtr_ = std::unique_ptr<RDKit::RWMol>(RDKit::MolFileToMol(mol2FilePath, false));
    ASSERT_NE(molPtr_, nullptr);
    molPtr_->clearConformers();
    RDKit::MolOps::sanitizeMol(*molPtr_);
  }

  // Helper function to load multiple molecules from SDF
  void loadMultipleMols(int count = 5) {
    const std::string sdfFilePath = testDataFolderPath_ / "MMFF94_dative.sdf";

    // Use getMols utility function from test_utils.h
    std::vector<std::unique_ptr<RDKit::ROMol>> tempMols;
    getMols(sdfFilePath, tempMols, count);
    ASSERT_EQ(tempMols.size(), count) << "Expected to load " << count << " molecules";

    // Convert to RWMol and prepare molecules
    multipleMols_.clear();
    for (auto& tempMol : tempMols) {
      auto rwMol = std::make_unique<RDKit::RWMol>(*tempMol);
      rwMol->clearConformers();
      RDKit::MolOps::sanitizeMol(*rwMol);
      multipleMols_.push_back(std::move(rwMol));
    }

    ASSERT_EQ(multipleMols_.size(), count) << "Expected to load " << count << " molecules";
  }

  // Helper function to create bounds matrix for a molecule
  DistGeom::BoundsMatPtr createBoundsMatrix(const RDKit::ROMol& mol) {
    const unsigned int nAtoms = mol.getNumAtoms();

    // Create ETKDGv3 parameters for proper ETKDG initialization
    RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;

    // Initialize ETKDG details
    ForceFields::CrystalFF::CrystalFFDetails etkdgDetails;
    initETKDG(&mol, params, etkdgDetails);

    // Create bounds matrix using RDKit's expected shared_ptr type for API calls
    DistGeom::BoundsMatPtr boundsMatrix = boost::make_shared<DistGeom::BoundsMatrix>(nAtoms);

    // Initialize bounds matrix
    RDKit::DGeomHelpers::initBoundsMat(boundsMatrix.get());

    // Use the first signature of setTopolBounds with bonds and angles from ETKDG details
    if (params.useExpTorsionAnglePrefs || params.useBasicKnowledge) {
      RDKit::DGeomHelpers::setTopolBounds(mol,
                                          boundsMatrix,
                                          etkdgDetails.bonds,
                                          etkdgDetails.angles,
                                          true,
                                          false,
                                          params.useMacrocycle14config,
                                          params.forceTransAmides);
    } else {
      // Fallback to second signature if ETKDG features are not enabled
      RDKit::DGeomHelpers::setTopolBounds(mol,
                                          boundsMatrix,
                                          true,
                                          false,
                                          params.useMacrocycle14config,
                                          params.forceTransAmides);
    }

    return boundsMatrix;
  }

  // Helper function to create a copy of bounds matrix
  DistGeom::BoundsMatPtr copyBoundsMatrix(const DistGeom::BoundsMatrix& original) {
    const unsigned int nAtoms = original.numRows();
    auto               copy   = boost::make_shared<DistGeom::BoundsMatrix>(nAtoms);

    for (unsigned int i = 0; i < nAtoms; ++i) {
      for (unsigned int j = 0; j < nAtoms; ++j) {
        copy->setVal(i, j, original.getVal(i, j));
      }
    }

    return copy;
  }

  // Helper function to compare two bounds matrices
  bool compareBoundsMatrices(const DistGeom::BoundsMatrix& mat1,
                             const DistGeom::BoundsMatrix& mat2,
                             double                        tolerance = 1e-10) {
    if (mat1.numRows() != mat2.numRows() || mat1.numCols() != mat2.numCols()) {
      return false;
    }

    const unsigned int nAtoms = mat1.numRows();
    for (unsigned int i = 0; i < nAtoms; ++i) {
      for (unsigned int j = 0; j < nAtoms; ++j) {
        const double val1 = mat1.getVal(i, j);
        const double val2 = mat2.getVal(i, j);
        if (std::abs(val1 - val2) > tolerance) {
          std::cout << "Mismatch at (" << i << "," << j << "): " << val1 << " vs " << val2
                    << " (diff: " << std::abs(val1 - val2) << ")" << std::endl;
          return false;
        }
      }
    }
    return true;
  }

  std::filesystem::path                      testDataFolderPath_;
  std::unique_ptr<RDKit::RWMol>              molPtr_;
  std::vector<std::unique_ptr<RDKit::RWMol>> multipleMols_;
};

// Test single molecule triangle smoothing with needSmoothing tracking
TEST_F(TriangleSmoothTest, SingleMolTriangleSmoothTest) {
  loadSingleMol();

  // Create bounds matrices
  auto originalMatrix = createBoundsMatrix(*molPtr_);
  auto rdkitMatrix    = copyBoundsMatrix(*originalMatrix);
  auto batchMatrix    = copyBoundsMatrix(*originalMatrix);

  // Test individual RDKit triangle smoothing
  bool rdkitSuccess = DistGeom::triangleSmoothBounds(rdkitMatrix);

  // Test batch GPU triangle smoothing with needSmoothing tracking
  std::vector<DistGeom::BoundsMatPtr> singleMolBatch;
  singleMolBatch.push_back(std::move(batchMatrix));

  std::vector<uint8_t> needSmoothing(1, 1);  // Single molecule, initially needs smoothing
  nvMolKit::triangleSmoothBoundsBatch(singleMolBatch, &needSmoothing);

  // Check needSmoothing result
  if (rdkitSuccess) {
    // If RDKit succeeded, our batch should have converged too (needSmoothing[0] should be 0)
    EXPECT_EQ(needSmoothing[0], 0) << "Single molecule should have converged (needSmoothing should be 0)";

    // Compare matrix results
    EXPECT_TRUE(compareBoundsMatrices(*rdkitMatrix, *singleMolBatch[0]))
      << "Single molecule batch result differs from RDKit result";
  } else {
    // If RDKit failed, we can't make strong assumptions, but the function should still complete
    std::cout << "RDKit triangle smoothing failed for single molecule, needSmoothing = "
              << static_cast<int>(needSmoothing[0]) << std::endl;
  }
}

// Test batch triangle smoothing - basic functionality
TEST_F(TriangleSmoothTest, BatchTriangleSmoothBasicTest) {
  loadMultipleMols(5);

  // Create bounds matrices for all molecules
  std::vector<DistGeom::BoundsMatPtr> originalMatrices;
  std::vector<DistGeom::BoundsMatPtr> rdkitMatrices;
  std::vector<DistGeom::BoundsMatPtr> batchMatrices;

  for (const auto& mol : multipleMols_) {
    auto originalMatrix = createBoundsMatrix(*mol);
    auto rdkitMatrix    = copyBoundsMatrix(*originalMatrix);
    auto batchMatrix    = copyBoundsMatrix(*originalMatrix);

    originalMatrices.push_back(std::move(originalMatrix));
    rdkitMatrices.push_back(std::move(rdkitMatrix));
    batchMatrices.push_back(std::move(batchMatrix));
  }

  // Test individual RDKit triangle smoothing
  std::vector<bool> rdkitResults;
  for (auto& matrix : rdkitMatrices) {
    bool success = DistGeom::triangleSmoothBounds(matrix);
    rdkitResults.push_back(success);
  }

  // Test batch GPU triangle smoothing
  std::vector<uint8_t> needSmoothing(multipleMols_.size(), 1);  // Initialize all molecules as needing smoothing
  nvMolKit::triangleSmoothBoundsBatch(batchMatrices, &needSmoothing);

  // Check needSmoothing results - molecules that converged should have needSmoothing[i] = 0
  for (size_t i = 0; i < multipleMols_.size(); ++i) {
    if (rdkitResults[i]) {
      // If RDKit succeeded, our batch should have converged too (needSmoothing[i] should be 0)
      EXPECT_EQ(needSmoothing[i], 0) << "Molecule " << i << " should have converged (needSmoothing should be 0)";
    }
    // Note: If RDKit failed, we can't make assumptions about needSmoothing value
  }

  // Compare individual results
  for (size_t i = 0; i < multipleMols_.size(); ++i) {
    if (rdkitResults[i]) {
      EXPECT_TRUE(compareBoundsMatrices(*rdkitMatrices[i], *batchMatrices[i]))
        << "Batch result differs from RDKit result for molecule " << i;
    }
  }
}

// Test batch triangle smoothing with DeviceBoundsMatrixBatch interface
TEST_F(TriangleSmoothTest, BatchDeviceMatrixInterfaceTest) {
  loadMultipleMols(3);

  // Create bounds matrices for all molecules
  std::vector<DistGeom::BoundsMatPtr> originalMatrices;
  std::vector<DistGeom::BoundsMatPtr> rdkitMatrices;
  std::vector<unsigned int>           matrixSizes;

  for (const auto& mol : multipleMols_) {
    auto originalMatrix = createBoundsMatrix(*mol);
    auto rdkitMatrix    = copyBoundsMatrix(*originalMatrix);
    matrixSizes.push_back(originalMatrix->numRows());

    originalMatrices.push_back(std::move(originalMatrix));
    rdkitMatrices.push_back(std::move(rdkitMatrix));
  }

  // Test individual RDKit triangle smoothing
  std::vector<bool> rdkitResults;
  for (auto& matrix : rdkitMatrices) {
    bool success = DistGeom::triangleSmoothBounds(matrix);
    rdkitResults.push_back(success);
  }

  // Test batch GPU triangle smoothing with DeviceBoundsMatrixBatch
  nvMolKit::DeviceBoundsMatrixBatch deviceBatch(matrixSizes);
  deviceBatch.copyFromHost(originalMatrices);

  nvMolKit::triangleSmoothBoundsBatch(deviceBatch);

  // Copy results back to host
  std::vector<DistGeom::BoundsMatPtr> batchResults;
  for (size_t i = 0; i < multipleMols_.size(); ++i) {
    batchResults.push_back(boost::make_shared<DistGeom::BoundsMatrix>(matrixSizes[i]));
  }
  deviceBatch.copyToHost(batchResults);

  // Compare individual results
  for (size_t i = 0; i < multipleMols_.size(); ++i) {
    if (rdkitResults[i]) {
      EXPECT_TRUE(compareBoundsMatrices(*rdkitMatrices[i], *batchResults[i]))
        << "DeviceBoundsMatrixBatch result differs from RDKit result for molecule " << i;
    }
  }
}

// Test batch triangle smoothing edge cases
TEST_F(TriangleSmoothTest, BatchEdgeCasesTest) {
  // Test with empty batch
  std::vector<DistGeom::BoundsMatPtr> emptyBatch;
  nvMolKit::triangleSmoothBoundsBatch(emptyBatch);  // Should succeed without throwing

  // Test with single molecule in batch
  std::unique_ptr<RDKit::ROMol> singleROMol(RDKit::SmilesToMol("CCO"));
  auto                          singleMol = std::make_unique<RDKit::RWMol>(*singleROMol);
  singleMol->clearConformers();
  RDKit::MolOps::sanitizeMol(*singleMol);

  auto singleMatrix = createBoundsMatrix(*singleMol);
  auto rdkitCopy    = copyBoundsMatrix(*singleMatrix);
  auto batchCopy    = copyBoundsMatrix(*singleMatrix);

  // Individual RDKit test
  bool rdkitSuccess = DistGeom::triangleSmoothBounds(rdkitCopy);

  // Single molecule batch test
  std::vector<DistGeom::BoundsMatPtr> singleBatch;
  singleBatch.push_back(std::move(batchCopy));
  nvMolKit::triangleSmoothBoundsBatch(singleBatch);

  if (rdkitSuccess) {
    EXPECT_TRUE(compareBoundsMatrices(*rdkitCopy, *singleBatch[0])) << "Single molecule batch: results differ";
  }

  // Test with very small molecules batch
  std::vector<std::string>            smallSmiles = {"C", "C"};
  std::vector<DistGeom::BoundsMatPtr> smallOriginals;
  std::vector<DistGeom::BoundsMatPtr> smallRdkit;
  std::vector<DistGeom::BoundsMatPtr> smallBatch;

  for (const auto& smile : smallSmiles) {
    std::unique_ptr<RDKit::ROMol> romol(RDKit::SmilesToMol(smile));
    auto                          rwmol = std::make_unique<RDKit::RWMol>(*romol);
    rwmol->clearConformers();
    RDKit::MolOps::sanitizeMol(*rwmol);

    auto originalMatrix = createBoundsMatrix(*rwmol);
    auto rdkitMatrix    = copyBoundsMatrix(*originalMatrix);
    auto batchMatrix    = copyBoundsMatrix(*originalMatrix);

    smallOriginals.push_back(std::move(originalMatrix));
    smallRdkit.push_back(std::move(rdkitMatrix));
    smallBatch.push_back(std::move(batchMatrix));
  }

  // Test individual RDKit triangle smoothing
  std::vector<bool> smallRdkitResults;
  for (auto& matrix : smallRdkit) {
    bool success = DistGeom::triangleSmoothBounds(matrix);
    smallRdkitResults.push_back(success);
  }

  // Test batch GPU triangle smoothing
  nvMolKit::triangleSmoothBoundsBatch(smallBatch);

  // Compare results
  for (size_t i = 0; i < smallSmiles.size(); ++i) {
    if (smallRdkitResults[i]) {
      EXPECT_TRUE(compareBoundsMatrices(*smallRdkit[i], *smallBatch[i]))
        << "Small molecules batch result differs for " << smallSmiles[i];
    }
  }
}
