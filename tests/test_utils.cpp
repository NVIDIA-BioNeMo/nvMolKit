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

#include "tests/test_utils.h"

#include <Geometry/point.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

RDKit::DGeomHelpers::EmbedParameters getETKDGOption(ETKDGOption opt) {
  switch (opt) {
    case ETKDGOption::srETKDGv3:
      return RDKit::DGeomHelpers::srETKDGv3;
    case ETKDGOption::ETKDGv3:
      return RDKit::DGeomHelpers::ETKDGv3;
    case ETKDGOption::ETKDGv2:
      return RDKit::DGeomHelpers::ETKDGv2;
    case ETKDGOption::ETKDG:
      return RDKit::DGeomHelpers::ETKDG;
    case ETKDGOption::ETDG:
      return RDKit::DGeomHelpers::ETDG;
    case ETKDGOption::KDG:
      return RDKit::DGeomHelpers::KDG;
    case ETKDGOption::DG: {
      auto params              = RDKit::DGeomHelpers::KDG;
      params.useBasicKnowledge = false;
      return params;
    }
    default:
      throw std::runtime_error("Unknown ETKDG option");
  }
}

std::string getETKDGOptionName(ETKDGOption opt) {
  switch (opt) {
    case ETKDGOption::srETKDGv3:
      return "srETKDGv3";
    case ETKDGOption::ETKDGv3:
      return "ETKDGv3";
    case ETKDGOption::ETKDGv2:
      return "ETKDGv2";
    case ETKDGOption::ETKDG:
      return "ETKDG";
    case ETKDGOption::ETDG:
      return "ETDG";
    case ETKDGOption::KDG:
      return "KDG";
    case ETKDGOption::DG:
      return "DG";
    default:
      return "Unknown";
  }
}

std::string getTestDataFolderPath() {
  // Check for the NVMOLKIT_TESTDATA env variable
  const char* testDataEnv = std::getenv("NVMOLKIT_TESTDATA");
  if (testDataEnv != nullptr) {
    return testDataEnv;
  }

  // Get the path to the tests directory (where test_utils.cpp is located)
  const std::string           currentFilePath = __FILE__;
  const std::filesystem::path testsDirPath    = std::filesystem::path(currentFilePath).parent_path();

  // Look for test_data as a subdirectory of tests
  const std::filesystem::path testDataPath = testsDirPath / "test_data";
  if (std::filesystem::exists(testDataPath)) {
    return testDataPath.string();
  }
  throw std::runtime_error("Could not find test data folder. Please set the NVMOLKIT_TESTDATA environment variable.");
}

void getMols(const std::string& fileName, std::vector<std::unique_ptr<RDKit::ROMol>>& mols, std::optional<int> count) {
  const std::string mol2FilePath = fileName;
  ASSERT_TRUE(std::filesystem::exists(mol2FilePath)) << "Could not find " << mol2FilePath;
  int                  stop_count = count.value_or(std::numeric_limits<int>::max());
  RDKit::SDMolSupplier suppl(mol2FilePath, true, false);
  ASSERT_TRUE(suppl.length() > 0);
  while (!suppl.atEnd() && stop_count > 0) {
    mols.push_back(std::unique_ptr<RDKit::ROMol>(suppl.next()));
    ASSERT_NE(mols.back(), nullptr);
    ASSERT_GT(mols.back()->getNumAtoms(), 1) << mols.back()->getProp<std::string>("_Name");
    ASSERT_GT(mols.back()->getNumBonds(), 0);
    ASSERT_GT(mols.back()->getNumConformers(), 0);
    stop_count--;
  }
}

std::vector<double> convertPositionsToVector(const std::vector<std::unique_ptr<RDGeom::Point>>& positions,
                                             unsigned int                                       dim) {
  std::vector<double> posVec(positions.size() * dim);
  for (unsigned int i = 0; i < positions.size(); i++) {
    for (unsigned int j = 0; j < dim; j++) {
      posVec[i * dim + j] = (*positions[i])[j];
    }
  }
  return posVec;
}
