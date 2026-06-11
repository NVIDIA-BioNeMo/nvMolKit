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
                                                     const int     matrixDim) {
  double& out          = result[relIdx];
  out                  = 0.0;
  const int& matRowIdx = relIdx;
  for (int i = 0; i < matrixDim; i++) {
    out += matrix[matRowIdx * matrixDim + i] * vector[i];
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
__global__ void batchEigensolverKernel(const int      numEigs,
                                       const int      matrixDim,
                                       double*        mutableBoundsMatrices,
                                       double*        eigenvaluesOut,
                                       double*        eigenvectorsOut,
                                       uint8_t*       converged,
                                       curandState*   state,
                                       const uint8_t* active,
                                       const int      seed = 42) {
  constexpr unsigned int MAX_ITERATIONS = 1000;
  constexpr double       TOLERANCE      = 0.001;
  constexpr double       TINY_EIGVAL    = 1.0e-10;

  using BlockReduce = cub::BlockReduce<double, 256>;
  // Used for reducing to largest element in z
  __shared__ BlockReduce::TempStorage temp_storage;
  // Shared memory for v and z vectors.
  __shared__ double                   v[256];
  __shared__ double                   z[256];

  __shared__ double localEigs[4];  // max number
  __shared__ bool   localConverged;
  __shared__ double prevEigval;

  const int relIdxWithinSystem = threadIdx.x;
  const int systemIdx          = blockIdx.x;

  // Should be identical per block so no concerns about divergence.
  if (active != nullptr && active[systemIdx] == 0) {
    return;
  }

  // Responsible for setting sync shared variables and writing out eigenvalues.
  const bool isFirstThread  = relIdxWithinSystem == 0;
  // Participates in matrix-vector multiplication and normalization. Reads input and writes eigenvectors.
  const bool isLoaderThread = relIdxWithinSystem < matrixDim;

  // System-local starting points for global arrays.
  double*      localBoundsMatrix   = mutableBoundsMatrices + systemIdx * matrixDim * matrixDim;
  double*      localEigenvaluesOut = eigenvaluesOut + systemIdx * numEigs;
  double*      localEigenvectors   = eigenvectorsOut + systemIdx * matrixDim * matrixDim;
  curandState* localState          = state + systemIdx * matrixDim;

  if (isLoaderThread) {
    curand_init(seed, relIdxWithinSystem, 0, &localState[relIdxWithinSystem]);
  }

  z[relIdxWithinSystem] = 0.0;
  for (int ei = 0; ei < numEigs; ei++) {
    __syncthreads();  // for initial variable write.
    if (isFirstThread) {
      localEigs[ei] = -1000.0;
    }
    double norm = 0.0;
    // Initial random matrix for V with normalization.
    if (isLoaderThread) {
      v[relIdxWithinSystem] = curand_uniform_double(&localState[relIdxWithinSystem]);
    }
    __syncthreads();  // v must be fully written before norm
    if (isLoaderThread) {
      norm = L2Norm(v, matrixDim);
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
      prevEigval = localEigs[ei];
      // Initial Matrix X v = z
      if (relIdxWithinSystem < matrixDim) {
        matrixVectorMultiply(relIdxWithinSystem, localBoundsMatrix, v, z, matrixDim);
      }
      __syncthreads();  // Finish multiply before reduce.

      // Find largest element in z
      const double largestZ = BlockReduce(temp_storage).Reduce(z, MaxFunctor());
      if (isFirstThread) {
        localEigs[ei] = largestZ;
      }
      __syncthreads();                      // wait for localEigs to write.
      const double eigVal = localEigs[ei];  // broadcast to all threads.
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
      norm = L2Norm(v, matrixDim);
    }
    __syncthreads();  // Wait for norm to be written.
    if (isLoaderThread) {
      v[relIdxWithinSystem] /= norm;
      localEigenvectors[ei * matrixDim + relIdxWithinSystem] = v[relIdxWithinSystem];
    }
    if (isFirstThread) {
      localEigenvaluesOut[ei] = localEigs[ei];
    }
    __syncthreads();  // wait for v normalization to finish.
    // Remove eigenvalue space from matrix
    if (isLoaderThread) {
      for (int j = 0; j < matrixDim; j++) {
        localBoundsMatrix[relIdxWithinSystem * matrixDim + j] -= (localEigs[ei] * v[relIdxWithinSystem] * v[j]);
      }
    }
  }
  if (isFirstThread) {
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
                                  const int      seed = 42) {
  const int numBlocks          = numSystems;
  const int numThreadsPerBlock = 256;
  if (matrixDim > 256) {
    throw std::runtime_error("Matrix dimension is too large for the kernel");
  }
  batchEigensolverKernel<<<numBlocks, numThreadsPerBlock>>>(numEigs,
                                                            matrixDim,
                                                            mutableBoundsMatrices,
                                                            eigenvaluesOut,
                                                            eigenvectorsOut,
                                                            converged,
                                                            states,
                                                            active,
                                                            seed);
  cudaCheckError(cudaGetLastError());
}

}  // namespace

class BatchedEigenSolver::Impl {
 public:
  void solve(int            numEigs,
             int            matrixDim,
             int            batch_size,
             double*        matrices,
             double*        eigenvalues,
             double*        eigenvectors,
             const uint8_t* active,
             int            randomSeed) {
    converged_.resize(batch_size);
    cudaCheckError(cudaMemsetAsync(converged_.data(), 0, batch_size * sizeof(uint8_t)));

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
                                 active,
                                 randomSeed);
  }

  const uint8_t* converged() const { return converged_.data(); }

 private:
  AsyncDeviceVector<uint8_t>     converged_;
  AsyncDeviceVector<curandState> states_;
};

BatchedEigenSolver::BatchedEigenSolver() : pimpl_(std::make_unique<Impl>()) {}
BatchedEigenSolver::~BatchedEigenSolver() = default;

void BatchedEigenSolver::solve(int            numEigs,
                               int            matrixDim,
                               int            batch_size,
                               double*        matrices,
                               double*        eigenvalues,
                               double*        eigenvectors,
                               const uint8_t* active,
                               int            randomSeed) {
  pimpl_->solve(numEigs, matrixDim, batch_size, matrices, eigenvalues, eigenvectors, active, randomSeed);
}

const uint8_t* BatchedEigenSolver::converged() const {
  return pimpl_->converged();
}
}  // namespace nvMolKit
