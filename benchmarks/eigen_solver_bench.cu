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

#include <nanobench.h>
#include <Numerics/EigenSolvers/PowerEigenSolver.h>
#include <Numerics/Matrix.h>

#include <iostream>
#include <random>
#include <vector>

#include "src/symmetric_eigensolver.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

using namespace nvMolKit;

#undef NDEBUG

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

RDNumeric::SymmMatrix<double> initialCoordsNormDistances(const RDNumeric::SymmMatrix<double>& initialDistMat) {
  constexpr double EIGVAL_TOL = 0.001;
  const int        N          = initialDistMat.numRows();

  RDNumeric::SymmMatrix<double> sqMat(N), T(N, 0.0);

  double* sqDat = sqMat.getData();

  int           dSize   = initialDistMat.getDataSize();
  const double* data    = initialDistMat.getData();
  double        sumSqD2 = 0.0;
  for (int i = 0; i < dSize; i++) {
    sqDat[i] = data[i] * data[i];
    sumSqD2 += sqDat[i];
  }
  sumSqD2 /= (N * N);

  RDNumeric::DoubleVector sqD0i(N, 0.0);
  double*                 sqD0iData = sqD0i.getData();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sqD0iData[i] += sqMat.getVal(i, j);
    }
    sqD0iData[i] /= N;
    sqD0iData[i] -= sumSqD2;

    if ((sqD0iData[i] < EIGVAL_TOL) && (N > 3)) {
      sqD0iData[i] = 10 * EIGVAL_TOL;
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      double val = 0.5 * (sqD0iData[i] + sqD0iData[j] - sqMat.getVal(i, j));
      T.setVal(i, j, val);
    }
  }
  return T;
}

std::vector<RDNumeric::SymmMatrix<double>> createNMatrices(const int N, const int dim) {
  std::vector<RDNumeric::SymmMatrix<double>> result;
  for (int i = 0; i < N; i++) {
    auto initMat = createSymmetricDoubleMatrix(dim, 100.0);
    result.push_back(initialCoordsNormDistances(initMat));
  }
  return result;
}

std::vector<double> packedNMatrices(const std::vector<RDNumeric::SymmMatrix<double>>& matrices) {
  std::vector<double> result;
  const int           N = matrices[0].numRows();
  result.resize(N * N * matrices.size());
  for (size_t b = 0; b < matrices.size(); b++) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        result[b * N * N + i * N + j] = matrices[b].getVal(i, j);
      }
    }
  }
  return result;
}

int main() {
  // Matrix dimensions to test
  std::vector<int> matrix_dims = {5, 10, 20, 50, 100};
  // Batch sizes to test
  std::vector<int> batch_sizes = {10, 100, 1000};

  // Create nanobench runner
  printf("Running benchmarks...\n");
  // Run benchmarks
  for (int n : matrix_dims) {
    for (int batch_size : batch_sizes) {
      auto                                                    rdkitDistancesMats = createNMatrices(batch_size, n);
      std::vector<std::vector<RDNumeric::SymmMatrix<double>>> rdkitDistancesMatsReplicates;
      for (int i = 0; i < 6; ++i) {
        rdkitDistancesMatsReplicates.push_back(rdkitDistancesMats);
      }

      auto combined = packedNMatrices(rdkitDistancesMats);

      std::vector<RDNumeric::DoubleMatrix> eigenvectorsRef;
      for (int i = 0; i < batch_size; ++i) {
        eigenvectorsRef.push_back(RDNumeric::DoubleMatrix(n, n));
      }
      int                     cpuIdx = 0;
      // Use power eigen solver to compute expected.
      RDNumeric::DoubleVector eigenvalues(n);
      ankerl::nanobench::Bench().warmup(1).epochIterations(1).epochs(5).run(
        std::string("CPU Eigen n=") + std::to_string(n) + " batch=" + std::to_string(batch_size),
        [&] {
          for (int i = 0; i < batch_size; ++i) {
            RDNumeric::DoubleVector eigenvalues(n);
            if (!RDNumeric::EigenSolvers::powerEigenSolver(3,
                                                           rdkitDistancesMatsReplicates[cpuIdx][i],
                                                           eigenvalues,
                                                           &eigenvectorsRef[i],
                                                           42)) {
              throw std::runtime_error("Failed to compute eigenvalues");
            }
          }
          cpuIdx++;
        });
      assert(cpuIdx == 6);

      BatchedEigenSolver solver;

      std::vector<AsyncDeviceVector<double>> d_matrices;
      std::vector<AsyncDeviceVector<double>> d_eigenvectors;
      for (int i = 0; i < 6; ++i) {
        auto& d_matrix = d_matrices.emplace_back(n * n * batch_size);
        d_matrix.copyFromHost(combined);
        d_eigenvectors.emplace_back(n * n * batch_size);
      }
      AsyncDeviceVector<double> d_eigenvalues(batch_size * n);
      cudaDeviceSynchronize();
      int gpuIdx = 0;
      ankerl::nanobench::Bench().warmup(1).epochIterations(1).epochs(5).run(
        std::string("GPU Eigen n=") + std::to_string(n) + " batch=" + std::to_string(batch_size),
        [&] {
          solver
            .solve(3, n, batch_size, d_matrices[gpuIdx].data(), d_eigenvalues.data(), d_eigenvectors[gpuIdx].data());
          cudaDeviceSynchronize();
          gpuIdx++;
        });
      assert(gpuIdx == 6);
    }
  }

  return 0;
}
