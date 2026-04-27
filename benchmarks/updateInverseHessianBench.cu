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

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/minimizer/bfgs_hessian.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

using nvMolKit::checkReturnCode;

namespace {

constexpr int kDataDim        = 4;
constexpr int kMaxAtomsShared = 256;
constexpr int kNumRuns        = 10;
constexpr int kMaxSupported   = 1000;

bool parseBoolArg(const std::string& arg) {
  std::string lowered;
  lowered.resize(arg.size());
  std::transform(arg.begin(), arg.end(), lowered.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
}

[[noreturn]] void printUsageAndExit(const char* prog) {
  std::cerr << "Usage: " << prog << " <atoms_per_system> <batch_size> <use_large_kernel (true|false)>" << std::endl;
  std::exit(EXIT_FAILURE);
}

double computeMean(const std::vector<float>& timings) {
  if (timings.size() <= 1) {
    return 0.0;
  }
  const double sum = std::accumulate(timings.begin() + 1, timings.end(), 0.0);
  return sum / static_cast<double>(timings.size() - 1);
}

double computeStdDev(const std::vector<float>& timings, double mean) {
  if (timings.size() <= 1) {
    return 0.0;
  }
  const size_t count = timings.size() - 1;
  double       accum = 0.0;
  for (size_t i = 1; i < timings.size(); ++i) {
    const double diff = static_cast<double>(timings[i]) - mean;
    accum += diff * diff;
  }
  return std::sqrt(accum / static_cast<double>(count));
}

// Seed deterministic vectors so the benchmark is reproducible across runs and hosts.
std::vector<double> makeVector(size_t size, double scale) {
  std::vector<double> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = scale + 0.001 * static_cast<double>((i % 97) + 1);
  }
  return data;
}

// Populate a symmetric positive definite Hessian deterministically. Pure RNG might activate the error path.
std::vector<double> makeHessian(size_t systems, int dim) {
  const size_t        totalSize = systems * static_cast<size_t>(dim) * static_cast<size_t>(dim);
  std::vector<double> data(totalSize, 0.0);
  for (size_t sys = 0; sys < systems; ++sys) {
    const double base = 1.0 + 0.01 * static_cast<double>(sys + 1);
    for (int row = 0; row < dim; ++row) {
      for (int col = 0; col < dim; ++col) {
        const size_t idx = sys * static_cast<size_t>(dim) * static_cast<size_t>(dim) +
                           static_cast<size_t>(row) * static_cast<size_t>(dim) + static_cast<size_t>(col);
        if (row == col) {
          data[idx] = base;
        } else {
          data[idx] = 0.005 * static_cast<double>((row + col + 1) % 17 + 1);
        }
      }
    }
  }
  return data;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      printUsageAndExit(argv[0]);
    }

    const int  atomsPerSystem = std::stoi(argv[1]);
    const int  batchSize      = std::stoi(argv[2]);
    const bool useLarge       = parseBoolArg(argv[3]);

    if (atomsPerSystem <= 0 || batchSize <= 0) {
      throw std::runtime_error("atoms_per_system and batch_size must be positive integers");
    }
    if (atomsPerSystem > kMaxSupported || batchSize > kMaxSupported) {
      throw std::runtime_error("atoms_per_system and batch_size must be <= 1000 for this benchmark");
    }
    if (!useLarge && atomsPerSystem > kMaxAtomsShared) {
      throw std::runtime_error("Shared kernel path only supports up to 256 atoms per system");
    }

    const int    numSystems = batchSize;
    const int    dim        = atomsPerSystem * kDataDim;
    const size_t totalDim   = static_cast<size_t>(numSystems) * static_cast<size_t>(dim);
    const size_t totalHess  = static_cast<size_t>(numSystems) * static_cast<size_t>(dim) * static_cast<size_t>(dim);

    std::vector<int> atomStarts(numSystems + 1, 0);
    std::vector<int> hessianStarts(numSystems + 1, 0);
    std::vector<int> activeSystemIndices(numSystems, 0);

    for (int i = 0; i < numSystems; ++i) {
      atomStarts[i + 1]      = atomStarts[i] + atomsPerSystem;
      hessianStarts[i + 1]   = hessianStarts[i] + dim * dim;
      activeSystemIndices[i] = i;
    }

    std::vector<double> hostInvHessian = makeHessian(static_cast<size_t>(numSystems), dim);
    std::vector<double> hostDGrad      = makeVector(totalDim, 0.1);
    std::vector<double> hostXi         = makeVector(totalDim, 0.2);
    std::vector<double> hostGrad       = makeVector(totalDim, 0.3);
    std::vector<double> hostHessDGrad  = makeVector(totalDim, 0.4);

    nvMolKit::AsyncDeviceVector<int>    dAtomStarts(atomStarts.size());
    nvMolKit::AsyncDeviceVector<int>    dHessianStarts(hessianStarts.size());
    nvMolKit::AsyncDeviceVector<double> dInvHessian(totalHess);
    nvMolKit::AsyncDeviceVector<double> dDGrad(totalDim);
    nvMolKit::AsyncDeviceVector<double> dXi(totalDim);
    nvMolKit::AsyncDeviceVector<double> dGrad(totalDim);
    nvMolKit::AsyncDeviceVector<double> dHessDGrad(totalDim);

    dAtomStarts.copyFromHost(atomStarts);
    dHessianStarts.copyFromHost(hessianStarts);

    nvMolKit::AsyncDeviceVector<int> dActiveSystemIndices;
    dActiveSystemIndices.resize(activeSystemIndices.size());
    dActiveSystemIndices.copyFromHost(activeSystemIndices);

    nvMolKit::AsyncDeviceVector<int>    dBlockIdxToSys;
    nvMolKit::AsyncDeviceVector<int>    dBlockWithinSys;
    nvMolKit::AsyncDeviceVector<int>    dRowToSystemMap;
    nvMolKit::AsyncDeviceVector<int>    dRowToLocalRowMap;
    nvMolKit::AsyncDeviceVector<double> dIntermediateSums;

    std::vector<float> timings;
    timings.reserve(kNumRuns);

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    cudaCheckError(cudaEventCreate(&startEvent));
    cudaCheckError(cudaEventCreate(&stopEvent));

    for (int iter = 0; iter < kNumRuns + 1; ++iter) {
      dInvHessian.copyFromHost(hostInvHessian);
      dDGrad.copyFromHost(hostDGrad);
      dXi.copyFromHost(hostXi);
      dGrad.copyFromHost(hostGrad);
      dHessDGrad.copyFromHost(hostHessDGrad);
      if (useLarge) {
        dIntermediateSums.zero();
      }

      cudaCheckError(cudaEventRecord(startEvent));

      nvMolKit::updateInverseHessianBFGSBatch(numSystems,
                                              nullptr,
                                              dHessianStarts.data(),
                                              dAtomStarts.data(),
                                              dInvHessian.data(),
                                              dDGrad.data(),
                                              dXi.data(),
                                              dHessDGrad.data(),
                                              dGrad.data(),
                                              kDataDim,
                                              useLarge,
                                              dActiveSystemIndices.data());
      cudaCheckError(cudaEventRecord(stopEvent));
      cudaCheckError(cudaEventSynchronize(stopEvent));

      float elapsedMs = 0.0f;
      cudaCheckError(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
      if (iter > 0) {
        timings.push_back(elapsedMs);
      }
    }

    cudaCheckError(cudaEventDestroy(startEvent));
    cudaCheckError(cudaEventDestroy(stopEvent));

    const double mean   = computeMean(timings);
    const double stdDev = computeStdDev(timings, mean);

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout << std::setprecision(6);
    std::cout << "Atoms: " << std::setw(4) << atomsPerSystem << ", batchsize: " << std::setw(4) << batchSize
              << ", kernel: " << std::setw(6) << std::left << (useLarge ? "large" : "shared") << std::right
              << ", time: " << std::setw(11) << mean << " milliseconds, std: " << std::setw(11) << stdDev << std::endl;

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return EXIT_FAILURE;
  }
}
