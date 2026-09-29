// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_TEST_UTILS_H
#define NVMOLKIT_TEST_UTILS_H

#include <Geometry/point.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ROMol.h>

#include <optional>
#include <string>
#include <vector>

/**
 * @brief Macro to check CUDA operation success
 * @param x The CUDA operation result to check
 */
#define CHECK_CUDA_RETURN(x) EXPECT_EQ(x, cudaSuccess)

/**
 * @brief Enum representing different ETKDG options
 */
enum class ETKDGOption {
  srETKDGv3,
  ETKDGv3,
  ETKDGv2,
  ETKDG,
  ETDG,
  KDG,
  DG
};

/**
 * @brief Get the RDKit::DGeomHelpers::EmbedParameters for a given ETKDG option
 *
 * @param opt The ETKDG option to get the parameters for
 * @return RDKit::DGeomHelpers::EmbedParameters The parameters for the given option
 */
RDKit::DGeomHelpers::EmbedParameters getETKDGOption(ETKDGOption opt);

/**
 * @brief Get the name of an ETKDG option
 *
 * @param opt The ETKDG option to get the name for
 * @return std::string The name of the given option
 */
std::string getETKDGOptionName(ETKDGOption opt);

/**
 * @brief Gets the path to the test data folder
 *
 * This function attempts to find the test data folder in the following order:
 * 1. Checks for NVMOLKIT_TESTDATA environment variable
 * 2. Looks for a "test_data" directory under the tests directory
 *
 *
 * @return std::string The path to the test data folder
 * @throw std::runtime_error If the test data folder cannot be found
 */
std::string getTestDataFolderPath();

/**
 * @brief Loads molecules from a file into a vector
 *
 * Reads molecules from a file (typically SD or MOL2 format) and stores them in a vector.
 * Performs several assertions to ensure the molecules are valid:
 * - File exists
 * - File contains at least one molecule
 * - Each molecule has more than 1 atom
 * - Each molecule has at least 1 bond
 * - Each molecule has at least 1 conformer
 *
 * @param fileName Path to the file containing molecules
 * @param mols Vector to store the loaded molecules
 */
void getMols(const std::string&                          fileName,
             std::vector<std::unique_ptr<RDKit::ROMol>>& mols,
             std::optional<int>                          count = std::nullopt);

// Helper function to convert positions from vector of unique pointers to a flattened vector
std::vector<double> convertPositionsToVector(const std::vector<std::unique_ptr<RDGeom::Point>>& positions,
                                             unsigned int                                       dim);

//! Tolerance policy for comparing per-molecule minimized energies against an RDKit reference.
//! With `maxOutliers == 0` this reduces exactly to a strict Pointwise(DoubleNear(tightTolerance)) check.
struct EnergyMatchTolerance {
  double tightTolerance;
  int    maxOutliers      = 0;
  double averageTolerance = 0.0;  // 0 disables the average-difference check
  bool   requireProgress  = false;
};

//! Compares `got` against `ref` per-molecule, allowing up to `tol.maxOutliers` molecules (e.g. ones that
//! converged to an alternate local minimum) to exceed `tol.tightTolerance`.
void expectEnergyMatch(const std::vector<double>&  got,
                       const std::vector<double>&  ref,
                       const std::vector<double>&  start,
                       const EnergyMatchTolerance& tol);

#endif  // NVMOLKIT_TEST_UTILS_H
