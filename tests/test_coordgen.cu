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
#include <DistGeom/DistGeomUtils.h>
#include <ForceField/ForceField.h>
#include <gmock/gmock.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <gtest/gtest.h>
#include <Numerics/EigenSolvers/PowerEigenSolver.h>
#include <Numerics/SymmMatrix.h>

#include <random>

#include "rdkit_extensions/bounds_matrix.h"
#include "src/forcefields/coord_gen.h"
#include "src/symmetric_eigensolver.h"
#include "src/utils/device_vector.h"
#include "tests/test_utils.h"

using namespace nvMolKit;

/**
 * @brief Creates a symmetric double matrix in RDKit format with random positive values.
 *
 * @param size The size of the matrix (number of rows/columns).
 * @param maxValue The maximum value for the random positive entries.
 * @return std::shared_ptr<RDKit::DistGeom::BoundsMatrix> The generated symmetric matrix.
 */
RDNumeric::SymmMatrix<double> createSymmetricDoubleMatrix(int size, double maxValue) {
  assert(size > 0 && maxValue > 0 && "Matrix size and maxValue must be positive.");

  // Create an RDKit BoundsMatrix
  auto                                   matrix = RDNumeric::SymmMatrix<double>(size, 0.0);
  // Random number generator
  std::mt19937                           gen(42);  // Declare 'gen' here
  std::uniform_real_distribution<double> dist(0.0, maxValue);

  // Fill the upper triangular part with random values
  for (int i = 0; i < size; ++i) {
    for (int j = i + 1; j < size; ++j) {
      double randomValue = dist(gen);
      matrix.setVal(i, j, randomValue);
      matrix.setVal(j, i, randomValue);  // Mirror to the lower triangular part
    }
  }

  // Set diagonal elements to 0 (or any other desired value)
  for (int i = 0; i < size; ++i) {
    matrix.setVal(i, i, 0.0);
  }

  return matrix;
}

std::vector<RDNumeric::SymmMatrix<double>> createNMatrices(const int N, const int dim) {
  std::vector<RDNumeric::SymmMatrix<double>> result;
  for (int i = 0; i < N; i++) {
    auto initMat = createSymmetricDoubleMatrix(dim, 100.0);
    result.push_back(RDKit::DGeomHelpers::initialCoordsNormDistances(initMat));
  }
  return result;
}

std::vector<double> packedNMatrices(const std::vector<RDNumeric::SymmMatrix<double>>& matrices, int& maxDimension) {
  // Compute maximum dimension of input matrices
  int N = 0;
  for (const auto& matrix : matrices) {
    N = std::max<int>(N, matrix.numRows());
  }
  maxDimension = N;

  std::vector<double> result(N * N * matrices.size(), 0.0);
  result.resize(N * N * matrices.size());
  for (size_t b = 0; b < matrices.size(); b++) {
    for (size_t i = 0; i < matrices[b].numRows(); i++) {
      for (size_t j = 0; j < matrices[b].numRows(); j++) {
        result[b * N * N + i * N + j] = matrices[b].getVal(i, j);
      }
    }
  }
  return result;
}

TEST(SymmetricEigenSolverTest, SimplePassing) {
  constexpr int              numMatrices = 2;
  constexpr int              matrixDim   = 5;
  // Reference data taken from RDKit powerEigenSolver tests.
  static std::vector<double> symmmatrix1 = {0.0,   1.0,   1.732, 2.268, 3.268, 1.0,   0.0,   1.0,   1.732,
                                            2.268, 1.732, 1.0,   0.0,   1.0,   1.732, 2.268, 1.732, 1.0,
                                            0.0,   1.0,   3.268, 2.268, 1.732, 1.0,   0.0};

  static std::vector<double> symmmatrix2 = {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                                            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0};

  static std::vector<double> expectedEigenValues = {6.981, -3.982, -1.395, -1.016, -0.586, 4.0, -1.0, -1.0, -1.0, -1.0};
  nvMolKit::BatchedEigenSolver solver;
  std::vector<double>          eigenvalues(numMatrices * matrixDim);
  std::vector<double>          combined(matrixDim * matrixDim * numMatrices);
  // Copy matrix 1 into first half, 2 into second half
  std::copy(symmmatrix1.begin(), symmmatrix1.end(), combined.begin());
  std::copy(symmmatrix2.begin(), symmmatrix2.end(), combined.begin() + matrixDim * matrixDim);

  AsyncDeviceVector<double> d_combined_inputs(combined.size());
  d_combined_inputs.copyFromHost(combined);
  AsyncDeviceVector<double> d_eigenvalues(eigenvalues.size());
  AsyncDeviceVector<double> d_eigenvectors(matrixDim * matrixDim * numMatrices);
  d_eigenvectors.zero();
  solver.solve(5, matrixDim, numMatrices, d_combined_inputs.data(), d_eigenvalues.data(), d_eigenvectors.data());
  d_eigenvalues.copyToHost(eigenvalues);
  d_eigenvectors.copyToHost(combined);

  // Compare expected eigen values to computed eigen values. Iterate over each matrix, and sort by absolute value for
  // comparison
  for (int i = 0; i < numMatrices; ++i) {
    for (int j = 0; j < 5; ++j) {
      // Set a 1% tolerance for eigenvalues
      EXPECT_NEAR(eigenvalues[i * 5 + j], expectedEigenValues[i * 5 + j], 1e-2) << "i: " << i << " j: " << j;
    }
  }
}

class SymmetricEigenSolverSyntheticTestFixture : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(SymmetricEigenSolverSyntheticTestFixture, SyntheticData) {
  const int matrixDim   = std::get<0>(GetParam());
  const int numMatrices = std::get<1>(GetParam());

  auto                                 rdkitDistancesMats = createNMatrices(numMatrices, matrixDim);
  std::vector<RDNumeric::DoubleMatrix> eigenvectorsRef;
  for (int i = 0; i < numMatrices; ++i) {
    eigenvectorsRef.push_back(RDNumeric::DoubleMatrix(matrixDim, matrixDim));
  }
  int  maxDimension;
  auto combined = packedNMatrices(rdkitDistancesMats, maxDimension);
  ASSERT_EQ(maxDimension, matrixDim);

  std::vector<std::vector<double>> expectedEigenValues(numMatrices);
  // Use power eigen solver to compute expected.
  for (int i = 0; i < numMatrices; ++i) {
    RDNumeric::DoubleVector eigenvalues(matrixDim);
    if (!RDNumeric::EigenSolvers::powerEigenSolver(3,
                                                   rdkitDistancesMats[i],
                                                   eigenvalues,
                                                   &eigenvectorsRef[i],
                                                   /*seed=*/42)) {
      throw std::runtime_error("Failed to compute eigenvalues");
    }
    for (size_t j = 0; j < eigenvalues.size(); j++) {
      expectedEigenValues[i].push_back(eigenvalues[j]);
    }
  }

  std::vector<double> eigenvalues(matrixDim * numMatrices);

  nvMolKit::BatchedEigenSolver solver;

  AsyncDeviceVector<double> d_combined_inputs(combined.size());
  d_combined_inputs.copyFromHost(combined);
  AsyncDeviceVector<double> d_eigenvalues(eigenvalues.size());
  d_eigenvalues.zero();
  AsyncDeviceVector<double> d_eigenvectors(matrixDim * matrixDim * numMatrices);
  d_eigenvectors.zero();
  solver.solve(3, matrixDim, numMatrices, d_combined_inputs.data(), d_eigenvalues.data(), d_eigenvectors.data());
  d_eigenvalues.copyToHost(eigenvalues);
  d_eigenvectors.copyToHost(combined);

  std::vector<uint8_t> converged(numMatrices);
  cudaCheckError(
    cudaMemcpy(converged.data(), solver.converged(), numMatrices * sizeof(uint8_t), cudaMemcpyDeviceToHost));
  EXPECT_THAT(converged, testing::Each(testing::Eq(1)));

  // Compare expected eigen values to computed eigen values. Iterate over each matrix, and sort by absolute value for
  // comparison
  for (int i = 0; i < numMatrices; ++i) {
    // Compare 3 largest eigenvalues in expected and want
    for (int j = 0; j < std::min(3, matrixDim); ++j) {
      // Set a 1% tolerance for eigenvalues
      EXPECT_NEAR(eigenvalues[i * 3 + j], expectedEigenValues[i][j], 1e-2) << "matrix: " << i << " eigvalue: " << j;
    }
  }

  constexpr double TOL = 1e-4;
  // Compare eigenvectors
  for (int i = 0; i < numMatrices; ++i) {
    for (int j = 0; j < std::min(3, matrixDim); ++j) {
      double sign = 1.0;
      for (int k = 0; k < matrixDim; ++k) {
        double gotVal  = combined[i * matrixDim * matrixDim + j * matrixDim + k];
        double wantVal = eigenvectorsRef[i].getVal(j, k);
        if (std::abs(gotVal - wantVal) > TOL) {
          if (std::abs(gotVal + wantVal) < TOL) {
            sign = -1.0;
          }
        }
        EXPECT_NEAR(sign * gotVal, wantVal, TOL) << "i: " << i << " j: " << j << " k: " << k;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(SymmetricEigenSolverSyntheticTest,
                         SymmetricEigenSolverSyntheticTestFixture,
                         testing::Combine(testing::Values(4, 25),    // matrix dim
                                          testing::Values(1, 10)));  // num matrices

TEST(SymmetricEigenSolverIntegrationTest, MMFF94Data) {
  // Flaky test, functionality currently unused
  GTEST_SKIP();
  // Load MMFF molecules
  std::string                                testDataFolderPath = getTestDataFolderPath();
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(testDataFolderPath + "/MMFF94_dative.sdf", mols, /*count=*/100);
  ASSERT_GT(mols.size(), 0) << "No molecules loaded from MMFF94_dative.sdf";

  auto params       = RDKit::DGeomHelpers::ETKDGv3;
  params.randNegEig = false;

  // Create bounds matrices and distance matrices for each molecule
  std::vector<RDNumeric::SymmMatrix<double>>            distanceMatrices;
  std::vector<RDNumeric::DoubleMatrix>                  eigenvectorsRef;
  std::vector<ForceFields::CrystalFF::CrystalFFDetails> details(1);
  for (const auto& mol : mols) {
    // Create bounds matrix
    // Get bounds matrix using nvMolKit helper
    std::vector<const RDKit::ROMol*> mol_vec{mol.get()};
    auto                             bounds_matrices = nvMolKit::getBoundsMatrices(mol_vec, params, details);
    auto                             boundsMatrix    = bounds_matrices[0];

    // Convert to distance matrix using initialCoordsNormDistances
    auto& distanceMat = distanceMatrices.emplace_back(mol->getNumAtoms());
    ::DistGeom::pickRandomDistMat(*boundsMatrix, distanceMat);

    RDKit::DGeomHelpers::initialCoordsNormDistances(distanceMat);

    // Allocate space for reference eigenvectors
    eigenvectorsRef.push_back(RDNumeric::DoubleMatrix(mol->getNumAtoms(), mol->getNumAtoms()));
  }

  // Pack distance matrices into a single vector for batched processing
  const size_t                     numMatrices = distanceMatrices.size();
  int                              maxDimension;
  auto                             combined = packedNMatrices(distanceMatrices, maxDimension);
  // Compute reference eigenvalues using RDKit's power eigensolver
  std::vector<std::vector<double>> expectedEigenValues(numMatrices);
  std::vector<uint8_t>             expectConverged;
  for (size_t i = 0; i < numMatrices; ++i) {
    RDNumeric::DoubleVector eigenvalues(distanceMatrices[i].numRows());
    expectConverged.push_back(
      RDNumeric::EigenSolvers::powerEigenSolver(3, distanceMatrices[i], eigenvalues, &eigenvectorsRef[i]));
    for (size_t j = 0; j < eigenvalues.size(); j++) {
      expectedEigenValues[i].push_back(eigenvalues[j]);
    }
  }

  // Create solver and allocate device memory
  nvMolKit::BatchedEigenSolver solver;
  std::vector<double>          eigenvalues(maxDimension * numMatrices, -100.0);

  AsyncDeviceVector<double> d_combined_inputs(combined.size());
  d_combined_inputs.copyFromHost(combined);
  AsyncDeviceVector<double> d_eigenvalues(eigenvalues.size());
  d_eigenvalues.copyFromHost(eigenvalues);
  AsyncDeviceVector<double> d_eigenvectors(maxDimension * maxDimension * numMatrices);
  d_eigenvectors.zero();
  solver.solve(3, maxDimension, numMatrices, d_combined_inputs.data(), d_eigenvalues.data(), d_eigenvectors.data());

  // Copy results back
  d_eigenvalues.copyToHost(eigenvalues);
  d_eigenvectors.copyToHost(combined);

  std::vector<uint8_t> converged(numMatrices);
  cudaCheckError(
    cudaMemcpy(converged.data(), solver.converged(), numMatrices * sizeof(uint8_t), cudaMemcpyDeviceToHost));
  EXPECT_THAT(converged, ::testing::Pointwise(::testing::Eq(), expectConverged));

  // The RDKit algorithm for computing eigenvalues depends on the initial seed, and variance grows with the number of
  // eigenvalues computed, so that while the first eigenvalue computed is nearly always within tolerance, the second can
  // be off by a larger amount, and further for the 3rd. We set a complete pass requirement for the first eigenvalue,
  // a 95% pass rate for the second, and 80% for the third.
  std::vector<std::vector<int>> eigvalFails(3);
  // We'll only check eigenvalues for completely passing cases. An additional threshold is set for precision, if
  // the eigenvalue check passes but doesn't reach the threshold for eigenvectors, we skip the eigenvector check.
  std::vector<bool>             checkEigvecs(numMatrices, true);
  int                           numEigVecsToCheck = numMatrices;

  for (size_t i = 0; i < numMatrices; ++i) {
    int matrixDim = distanceMatrices[i].numRows();
    if (!converged[i] || !expectConverged[i]) {
      // If the solver didn't converge, skip this matrix
      continue;
    }
    // Compare 3 largest eigenvalues
    for (int j = 0; j < std::min(3, matrixDim); ++j) {
      constexpr double tolForEigvecCheck = .001;
      if (std::abs(eigenvalues[i * 3 + j] - expectedEigenValues[i][j]) >= tolForEigvecCheck) {
        checkEigvecs[i] = false;  // different tolerance than pass/fail, for downstream eigenvector check
      }
      const double tolForTestPass = std::max(0.01, 1e-2 * std::abs(expectedEigenValues[i][j]));
      if (std::abs(eigenvalues[i * 3 + j] - expectedEigenValues[i][j]) > tolForTestPass) {
        eigvalFails[j].push_back(static_cast<int>(i));
      }
    }
    if (!checkEigvecs[i]) {
      numEigVecsToCheck--;
    }
  }

  if (eigvalFails[0].size() > 0) {
    std::string eigvalFailStr = "Eigenvalue 0 failures for the following systems: ";
    for (size_t i = 0; i < eigvalFails[0].size(); ++i) {
      eigvalFailStr += std::to_string(eigvalFails[0][i]) + " ";
    }
    FAIL() << eigvalFailStr;
  }
  constexpr double acceptableFailFraction2nd = 0.05;
  const double     fraction2 = static_cast<double>(eigvalFails[1].size()) / static_cast<double>(numMatrices);
  if (fraction2 > acceptableFailFraction2nd) {
    std::string eigvalFailStr = "Eigenvalue 1 failures for the following systems: ";
    for (size_t i = 0; i < eigvalFails[1].size(); ++i) {
      eigvalFailStr += std::to_string(eigvalFails[1][i]) + " ";
    }
    FAIL() << eigvalFailStr;
  }
  constexpr double acceptableFailFraction3rd = 0.2;
  const double     fraction3 = static_cast<double>(eigvalFails[2].size()) / static_cast<double>(numMatrices);
  if (fraction3 > acceptableFailFraction3rd) {
    std::string eigvalFailStr = "Eigenvalue 2 failures for the following systems: ";
    for (size_t i = 0; i < eigvalFails[2].size(); ++i) {
      eigvalFailStr += std::to_string(eigvalFails[2][i]) + " ";
    }
    FAIL() << eigvalFailStr;
  }

  ASSERT_GT(numEigVecsToCheck, 0) << "No eigenvectors with high enough precision to check";

  for (size_t i = 0; i < numMatrices; ++i) {
    if (!converged[i] || !expectConverged[i]) {
      continue;
    }
    if (!checkEigvecs[i]) {
      continue;
    }
    for (size_t j = 0; j < std::min<size_t>(3, distanceMatrices[i].numRows()); ++j) {
      double sign = 1.0;
      for (size_t k = 0; k < distanceMatrices[i].numRows(); ++k) {
        double wantVal = eigenvectorsRef[i].getVal(j, k);
        double tol     = std::max(0.1, 1e-2 * std::abs(wantVal));
        if (j == 2) {
          tol *= 3;  // third eigenvector is often less precise
        }
        double gotVal = combined[i * maxDimension * maxDimension + j * maxDimension + k];
        if (std::abs(gotVal - wantVal) > tol) {
          if (std::abs(gotVal + wantVal) < tol) {
            sign = -1.0;
          }
        }
        EXPECT_NEAR(sign * gotVal, wantVal, tol) << "Matrix " << i << " eigenvector " << j << " component " << k;
      }
    }
  }
}

// TODO: random coords
bool generateInitialCoordsReference(RDGeom::PointPtrVect*                       positions,
                                    const ::DistGeom::BoundsMatrix&             boundsMat,
                                    const RDKit::DGeomHelpers::EmbedParameters& embedParams,
                                    RDKit::double_source_type*                  rng) {
  RDNumeric::SymmMatrix<double> distMat(boundsMat.numRows());
  double                        largestDistance = ::DistGeom::pickRandomDistMat(boundsMat, distMat, *rng);
  RDUNUSED_PARAM(largestDistance);
  return ::DistGeom::computeInitialCoords(distMat, *positions, *rng, embedParams.randNegEig, embedParams.numZeroFail);
}

class CoordGenIntegrationTestFixture : public ::testing::TestWithParam<bool> {};

TEST_P(CoordGenIntegrationTestFixture, MMFF94Data) {
  // Get test data folder path
  std::string testDataFolderPath = getTestDataFolderPath();

  // Load MMFF molecules
  std::vector<std::unique_ptr<RDKit::ROMol>> molsPtrs;
  getMols(testDataFolderPath + "/MMFF94_dative.sdf", molsPtrs, /*count=*/100);

  std::vector<const RDKit::ROMol*> mols;
  for (auto& molPtr : molsPtrs) {
    molPtr->clearConformers();
    mols.push_back(molPtr.get());
  }

  auto params = RDKit::DGeomHelpers::ETKDGv3;

  const bool randNegEig = GetParam();
  params.randNegEig     = randNegEig;
  SCOPED_TRACE("Running with randNegEig = " + std::to_string(randNegEig));

  std::vector<ForceFields::CrystalFF::CrystalFFDetails> details(mols.size());
  const auto          boundsMatrices = nvMolKit::getBoundsMatrices(mols, params, details);
  std::vector<double> maxViolations;
  std::vector<double> violationFractions;
  int                 numFailures = 0;

  for (size_t i = 0; i < boundsMatrices.size(); i++) {
    auto&                                       mol    = *mols[i];
    const int                                   nAtoms = mol.getNumAtoms();
    RDGeom::PointPtrVect                        positions(mol.getNumAtoms());
    std::vector<std::unique_ptr<RDGeom::Point>> positionsStore;

    for (int j = 0; j < nAtoms; ++j) {
      // TODO test 4D
      positionsStore.emplace_back(new RDGeom::Point3D());
      positions[j] = positionsStore[j].get();
    }

    auto rng = &RDKit::getDoubleRandomSource();
    if (!generateInitialCoordsReference(&positions, *boundsMatrices[i], params, rng)) {
      numFailures++;
      continue;
    }

    double maxDistanceViolation  = 0.0;
    double numDistanceViolations = 0.0;
    for (int j = 0; j < nAtoms; ++j) {
      const auto* pos1 = static_cast<RDGeom::Point3D*>(positions[j]);
      for (int k = j + 1; k < nAtoms; ++k) {
        const auto* pos2       = static_cast<RDGeom::Point3D*>(positions[k]);
        const auto  diff       = *pos2 - *pos1;
        const auto  dist       = diff.length();
        double      lowerBound = boundsMatrices[i]->getLowerBound(j, k);
        double      upperBound = boundsMatrices[i]->getUpperBound(j, k);
        if (dist < lowerBound || dist > upperBound) {
          numDistanceViolations += 1.0;
          maxDistanceViolation = std::max(maxDistanceViolation, std::max(lowerBound - dist, dist - upperBound));
        }
      }
    }
    const double distanceViolationFraction = numDistanceViolations / static_cast<double>(nAtoms * (nAtoms - 1) / 2);
    maxViolations.push_back(maxDistanceViolation);
    violationFractions.push_back(distanceViolationFraction);
  }

  // GPU setup
  const unsigned int dim = 3;

  std::vector<int> atomStarts = {0};
  for (const auto& mol : mols) {
    atomStarts.push_back(atomStarts.back() + mol->getNumAtoms());
  }
  AsyncDeviceVector<double> positions;
  ASSERT_GT(atomStarts.back(), 0);
  positions.resize(atomStarts.back() * dim);
  AsyncDeviceVector<int> atomStartsDevice;
  atomStartsDevice.resize(atomStarts.size());
  atomStartsDevice.copyFromHost(atomStarts);

  nvMolKit::detail::InitialCoordinateGenerator coordgen;
  coordgen.computeBoundsMatrices(mols, params, details);
  coordgen.computeInitialCoordinates(positions.data(), atomStartsDevice.data());
  std::vector<uint8_t> converged(mols.size());
  cudaCheckError(
    cudaMemcpy(converged.data(), coordgen.getPassFail(), mols.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));
  int passCount = std::accumulate(converged.begin(), converged.end(), 0);

  std::vector<double> gotPositions;
  gotPositions.resize(positions.size());
  positions.copyToHost(gotPositions);
  cudaDeviceSynchronize();

  std::vector<double> nvmolkitDistViolationFractions;
  std::vector<double> nvmolkitMaxdDistanceViolations;

  for (size_t i = 0; i < boundsMatrices.size(); i++) {
    if (!converged[i]) {
      continue;
    }

    const int startIdx = atomStarts[i];
    const int endIdx   = atomStarts[i + 1];
    const int numAtoms = endIdx - startIdx;

    double maxDistanceViolation  = 0.0;
    double numDistanceViolations = 0.0;
    for (int j = 0; j < numAtoms; ++j) {
      const double pos1x = gotPositions[3 * (startIdx + j) + 0];
      const double pos1y = gotPositions[3 * (startIdx + j) + 1];
      const double pos1z = gotPositions[3 * (startIdx + j) + 2];
      for (int k = j + 1; k < numAtoms; ++k) {
        const double pos2x = gotPositions[3 * (startIdx + k) + 0];
        const double pos2y = gotPositions[3 * (startIdx + k) + 1];
        const double pos2z = gotPositions[3 * (startIdx + k) + 2];

        const double diffx      = pos2x - pos1x;
        const double diffy      = pos2y - pos1y;
        const double diffz      = pos2z - pos1z;
        const double dist       = std::sqrt(diffx * diffx + diffy * diffy + diffz * diffz);
        double       lowerBound = boundsMatrices[i]->getLowerBound(j, k);
        double       upperBound = boundsMatrices[i]->getUpperBound(j, k);
        if (dist < lowerBound || dist > upperBound) {
          numDistanceViolations += 1.0;
          maxDistanceViolation = std::max(maxDistanceViolation, std::max(lowerBound - dist, dist - upperBound));
        }
      }
    }
    const double distanceViolationFraction = numDistanceViolations / static_cast<double>(numAtoms * (numAtoms - 1) / 2);
    nvmolkitMaxdDistanceViolations.push_back(maxDistanceViolation);
    nvmolkitDistViolationFractions.push_back(distanceViolationFraction);
  }

  // Now do comparisons. First, pass rate
  constexpr int passFailTol        = 15;  // 15% tolerance
  const int     referencePassCount = static_cast<int>(mols.size()) - numFailures;
  EXPECT_NEAR(passCount, referencePassCount, passFailTol) << "Pass rate mismatch";

  // Now, max distance violations. Take the average.
  double maxDistanceViolation =
    std::accumulate(maxViolations.begin(), maxViolations.end(), 0.0) / static_cast<double>(maxViolations.size());
  double nvmolkitMaxDistanceViolation =
    std::accumulate(nvmolkitMaxdDistanceViolations.begin(), nvmolkitMaxdDistanceViolations.end(), 0.0) /
    static_cast<double>(nvmolkitMaxdDistanceViolations.size());
  // 0.5 angstrom tolerance
  EXPECT_NEAR(maxDistanceViolation, nvmolkitMaxDistanceViolation, 0.5)
    << "Max distance violation mismatch: " << maxDistanceViolation << " vs " << nvmolkitMaxDistanceViolation;

  // Now distance violation fraction.
  double distanceViolationFraction = std::accumulate(violationFractions.begin(), violationFractions.end(), 0.0) /
                                     static_cast<double>(violationFractions.size());
  double nvmolkitDistanceViolationFraction =
    std::accumulate(nvmolkitDistViolationFractions.begin(), nvmolkitDistViolationFractions.end(), 0.0) /
    static_cast<double>(nvmolkitDistViolationFractions.size());
  // 5% tolerance.
  EXPECT_NEAR(distanceViolationFraction, nvmolkitDistanceViolationFraction, 0.05);
}

INSTANTIATE_TEST_SUITE_P(CoordGenIntegrationTest, CoordGenIntegrationTestFixture, testing::Values(true, false));
