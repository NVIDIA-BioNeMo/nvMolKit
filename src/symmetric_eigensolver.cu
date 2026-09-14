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

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cub/cub.cuh>
#include <stdexcept>

#include "src/symmetric_eigensolver.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

namespace {

// Function to compute the L2 norm of a vector. Done within a single thread.
__device__ __forceinline__ double L2Norm(const double* v, const int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += v[i] * v[i];
  }
  return sqrt(sum);
}

// Function to perform matrix-vector multiplication. Each thread computes one output element.
__device__ __forceinline__ void matrixVectorMultiply(const int     relIdx,
                                                     const double* matrix,
                                                     const double* vector,
                                                     double*       result,
                                                     const int     matrixDim,
                                                     const int     leadingDim) {
  double& out          = result[relIdx];
  out                  = 0.0;
  const int& matRowIdx = relIdx;
  for (int i = 0; i < matrixDim; i++) {
    out += matrix[matRowIdx * leadingDim + i] * vector[i];
  }
}

// Returns the element that has the largest absolute value.
struct MaxFunctor {
  __device__ __forceinline__ double operator()(const double& a, const double& b) {
    return cuda::std::abs(a) > cuda::std::abs(b) ? a : b;
  }
};

// Port of RDKit power eigensolver. Original code:
// https://github.com/rdkit/rdkit/blob/master/Code/Numerics/EigenSolvers/PowerEigenSolver.cpp
template <int BLOCK_SIZE>
__global__ void batchEigensolverKernel(const int      numEigs,
                                       const int      matrixDim,
                                       double*        mutableBoundsMatrices,
                                       double*        eigenvaluesOut,
                                       double*        eigenvectorsOut,
                                       uint8_t*       converged,
                                       curandState*   state,
                                       const uint8_t* active,
                                       const int      seed,
                                       const int*     matrixDimensions,
                                       const int*     randomSeeds,
                                       const int*     eigenDimensions) {
  constexpr unsigned int MAX_ITERATIONS = 1000;
  constexpr double       TOLERANCE      = 0.001;
  constexpr double       TINY_EIGVAL    = 1.0e-10;

  using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
  // Used for reducing to largest element in z
  __shared__ typename BlockReduce::TempStorage temp_storage;
  // Shared memory for v and z vectors.
  __shared__ double                            v[BLOCK_SIZE];
  __shared__ double                            z[BLOCK_SIZE];

  __shared__ double localEig;
  __shared__ bool   localConverged;
  __shared__ double prevEigval;

  const int relIdxWithinSystem = threadIdx.x;
  const int systemIdx          = blockIdx.x;
  const int systemDim          = matrixDimensions == nullptr ? matrixDim : matrixDimensions[systemIdx];
  const int systemNumEigs =
    min(numEigs, min(eigenDimensions == nullptr ? numEigs : eigenDimensions[systemIdx], systemDim));

  // Should be identical per block so no concerns about divergence.
  if (active != nullptr && active[systemIdx] == 0) {
    return;
  }

  // Responsible for setting sync shared variables and writing out eigenvalues.
  const bool isFirstThread  = relIdxWithinSystem == 0;
  // Participates in matrix-vector multiplication and normalization. Reads input and writes eigenvectors.
  const bool isLoaderThread = relIdxWithinSystem < systemDim;

  // System-local starting points for global arrays.
  double*      localBoundsMatrix   = mutableBoundsMatrices + systemIdx * matrixDim * matrixDim;
  double*      localEigenvaluesOut = eigenvaluesOut + systemIdx * numEigs;
  double*      localEigenvectors   = eigenvectorsOut + systemIdx * matrixDim * matrixDim;
  curandState* localState          = state + systemIdx * matrixDim;
  const int    localSeed           = randomSeeds == nullptr ? seed : randomSeeds[systemIdx];

  z[relIdxWithinSystem] = 0.0;
  for (int ei = 0; ei < systemNumEigs; ei++) {
    __syncthreads();  // for initial variable write.
    if (isFirstThread) {
      localEig = -1000.0;
    }
    double norm = 0.0;
    // Initial random matrix for V with normalization.
    if (isLoaderThread) {
      const int eigenSeed = localSeed + (ei * (ei + 1)) / 2;
      curand_init(eigenSeed, relIdxWithinSystem, 0, &localState[relIdxWithinSystem]);
      v[relIdxWithinSystem] = curand_uniform_double(&localState[relIdxWithinSystem]);
    }
    __syncthreads();  // v must be fully written before norm
    if (isLoaderThread) {
      norm = L2Norm(v, systemDim);
    }
    __syncthreads();  // v must be normed before update
    if (isLoaderThread) {
      v[relIdxWithinSystem] /= norm;
    } else {
      v[relIdxWithinSystem] = 0.0;
    }

    if (isFirstThread) {
      localConverged = false;
    }
    // Iteration loop
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
      __syncthreads();  // handle previous writes
      if (isFirstThread) {
        prevEigval = localEig;
      }
      __syncthreads();
      // Initial Matrix X v = z
      if (isLoaderThread) {
        matrixVectorMultiply(relIdxWithinSystem, localBoundsMatrix, v, z, systemDim, matrixDim);
      }
      __syncthreads();  // Finish multiply before reduce.

      // Find largest element in z
      const double largestZ = BlockReduce(temp_storage).Reduce(z, MaxFunctor());
      if (isFirstThread) {
        localEig = largestZ;
      }
      __syncthreads();                 // wait for localEig to write.
      const double eigVal = localEig;  // broadcast to all threads.
      if (cuda::std::abs(eigVal) < TINY_EIGVAL) {
        break;
      }

      v[relIdxWithinSystem] = z[relIdxWithinSystem] / eigVal;
      if (fabs(eigVal - prevEigval) < TOLERANCE) {
        if (isFirstThread) {
          localConverged = true;
        }
        break;
      }
    }

    // Check if we converged for that eigenvalue.
    __syncthreads();  // Catch up to latest localConvergedWrite.
    if (!localConverged) {
      break;
    }
    // Normalize v
    if (isLoaderThread) {
      norm = L2Norm(v, systemDim);
    }
    __syncthreads();  // Wait for norm to be written.
    if (isLoaderThread) {
      v[relIdxWithinSystem] /= norm;
      localEigenvectors[ei * matrixDim + relIdxWithinSystem] = v[relIdxWithinSystem];
    }
    if (isFirstThread) {
      localEigenvaluesOut[ei] = localEig;
    }
    __syncthreads();  // wait for v normalization to finish.
    // Remove eigenvalue space from matrix
    if (isLoaderThread) {
      for (int j = 0; j < systemDim; j++) {
        localBoundsMatrix[relIdxWithinSystem * matrixDim + j] -= (localEig * v[relIdxWithinSystem] * v[j]);
      }
    }
  }
  if (isFirstThread) {
    for (int ei = systemNumEigs; ei < numEigs; ++ei) {
      localEigenvaluesOut[ei] = 0.0;
    }
    converged[systemIdx] = localConverged;
  }
}

void launchBatchEigensolverKernel(const int      numEigs,
                                  const int      numSystems,
                                  const int      matrixDim,
                                  double*        mutableBoundsMatrices,
                                  double*        eigenvaluesOut,
                                  double*        eigenvectorsOut,
                                  uint8_t*       converged,
                                  curandState*   states,
                                  const uint8_t* active,
                                  const int      seed,
                                  const int*     matrixDimensions,
                                  const int*     randomSeeds,
                                  const int*     eigenDimensions,
                                  cudaStream_t   stream) {
  const int numBlocks = numSystems;
  if (matrixDim > 256) {
    throw std::runtime_error("Matrix dimension is too large for the kernel");
  }
  if (matrixDim <= 32) {
    batchEigensolverKernel<32><<<numBlocks, 32, 0, stream>>>(numEigs,
                                                             matrixDim,
                                                             mutableBoundsMatrices,
                                                             eigenvaluesOut,
                                                             eigenvectorsOut,
                                                             converged,
                                                             states,
                                                             active,
                                                             seed,
                                                             matrixDimensions,
                                                             randomSeeds,
                                                             eigenDimensions);
  } else if (matrixDim <= 64) {
    batchEigensolverKernel<64><<<numBlocks, 64, 0, stream>>>(numEigs,
                                                             matrixDim,
                                                             mutableBoundsMatrices,
                                                             eigenvaluesOut,
                                                             eigenvectorsOut,
                                                             converged,
                                                             states,
                                                             active,
                                                             seed,
                                                             matrixDimensions,
                                                             randomSeeds,
                                                             eigenDimensions);
  } else if (matrixDim <= 128) {
    batchEigensolverKernel<128><<<numBlocks, 128, 0, stream>>>(numEigs,
                                                               matrixDim,
                                                               mutableBoundsMatrices,
                                                               eigenvaluesOut,
                                                               eigenvectorsOut,
                                                               converged,
                                                               states,
                                                               active,
                                                               seed,
                                                               matrixDimensions,
                                                               randomSeeds,
                                                               eigenDimensions);
  } else {
    batchEigensolverKernel<256><<<numBlocks, 256, 0, stream>>>(numEigs,
                                                               matrixDim,
                                                               mutableBoundsMatrices,
                                                               eigenvaluesOut,
                                                               eigenvectorsOut,
                                                               converged,
                                                               states,
                                                               active,
                                                               seed,
                                                               matrixDimensions,
                                                               randomSeeds,
                                                               eigenDimensions);
  }
  cudaCheckError(cudaGetLastError());
}

}  // namespace

class BatchedEigenSolver::Impl {
 public:
  void solve(int                              numEigs,
             int                              matrixDim,
             int                              batch_size,
             double*                          matrices,
             double*                          eigenvalues,
             double*                          eigenvectors,
             const BatchedEigenSolverOptions& options) {
    converged_.setStream(options.stream);
    states_.setStream(options.stream);
    converged_.resize(batch_size);
    cudaCheckError(cudaMemsetAsync(converged_.data(), 0, batch_size * sizeof(uint8_t), options.stream));

    const size_t statesNeeded = static_cast<size_t>(batch_size) * static_cast<size_t>(matrixDim);
    if (statesNeeded > states_.size()) {
      states_.resize(statesNeeded);
    }

    launchBatchEigensolverKernel(numEigs,
                                 batch_size,
                                 matrixDim,
                                 matrices,
                                 eigenvalues,
                                 eigenvectors,
                                 converged_.data(),
                                 states_.data(),
                                 options.active,
                                 options.randomSeed,
                                 options.matrixDimensions,
                                 options.randomSeeds,
                                 options.eigenDimensions,
                                 options.stream);
  }

  const uint8_t* converged() const { return converged_.data(); }

 private:
  AsyncDeviceVector<uint8_t>     converged_;
  AsyncDeviceVector<curandState> states_;
};

BatchedEigenSolver::BatchedEigenSolver() : pimpl_(std::make_unique<Impl>()) {}
BatchedEigenSolver::~BatchedEigenSolver() = default;

void BatchedEigenSolver::solve(int                              numEigs,
                               int                              matrixDim,
                               int                              batch_size,
                               double*                          matrices,
                               double*                          eigenvalues,
                               double*                          eigenvectors,
                               const BatchedEigenSolverOptions& options) {
  pimpl_->solve(numEigs, matrixDim, batch_size, matrices, eigenvalues, eigenvectors, options);
}

const uint8_t* BatchedEigenSolver::converged() const {
  return pimpl_->converged();
}
}  // namespace nvMolKit
