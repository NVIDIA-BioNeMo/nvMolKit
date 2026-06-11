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

#include <cmath>

#include "src/triangle_smooth.h"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit {

//! CUDA kernel for batch triangle smoothing bounds matrices
//! Each thread processes one (i,j) pair for a given k iteration across multiple matrices
__global__ void triangleSmoothBatchKernel(double*      boundsMatrices,
                                          const int*   matrixStarts,
                                          const int*   molIndices,
                                          unsigned int totalElements,
                                          unsigned int k,
                                          uint8_t*     needSmoothing,
                                          double       tol) {
  // Calculate linear thread index
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Early exit if thread index is out of bounds
  if (idx >= totalElements) {
    return;
  }

  // Get molecule index for this thread
  const int molIdx = molIndices[idx];

  // Early exit if this molecule doesn't need smoothing or has already failed
  if (needSmoothing[molIdx] == 0 || needSmoothing[molIdx] == 2) {
    return;
  }

  // Get matrix boundaries for this molecule
  const int matrixStart = matrixStarts[molIdx];
  const int matrixEnd   = matrixStarts[molIdx + 1];
  const int matrixSize  = matrixEnd - matrixStart;

  // Calculate matrix dimension (npt = sqrt(matrixSize))
  const unsigned int npt = static_cast<unsigned int>(sqrt(static_cast<double>(matrixSize)));

  // Early exit if k is larger than this matrix's size
  if (k >= npt) {
    return;
  }

  // Calculate local index within this matrix
  const int localIdx = idx - matrixStart;
  if (localIdx < 0 || localIdx >= matrixSize) {
    return;
  }

  // Convert linear index to 2D coordinates within this matrix
  const unsigned int i = static_cast<unsigned int>(localIdx) / npt;
  const unsigned int j = static_cast<unsigned int>(localIdx) % npt;

  // Early exit conditions - match original algorithm: i < npt-1, j > i, j < npt
  if (i >= npt - 1 || j <= i || j >= npt || i == k || j == k) {
    return;
  }

  // Calculate matrix indices for symmetric access (offset by matrixStart)
  unsigned int ii = i;
  unsigned int ik = k;
  if (ii > ik) {
    unsigned int temp = ii;
    ii                = ik;
    ik                = temp;
  }

  unsigned int jj = j;
  unsigned int jk = k;
  if (jj > jk) {
    unsigned int temp = jj;
    jj                = jk;
    jk                = temp;
  }

  // Get bounds values (offset indices by matrixStart)
  const double Uik = boundsMatrices[matrixStart + ii * npt + ik];  // upper bound
  const double Lik = boundsMatrices[matrixStart + ik * npt + ii];  // lower bound
  const double Ukj = boundsMatrices[matrixStart + jj * npt + jk];  // upper bound
  const double Ljk = boundsMatrices[matrixStart + jk * npt + jj];  // lower bound

  // Calculate potential new bounds
  const double sumUikUkj  = Uik + Ukj;
  const double diffLikUjk = Lik - Ukj;
  const double diffLjkUik = Ljk - Uik;

  // Adjust upper bound
  if (boundsMatrices[matrixStart + i * npt + j] > sumUikUkj) {
    boundsMatrices[matrixStart + i * npt + j] = sumUikUkj;
  }

  // Adjust lower bound
  if (boundsMatrices[matrixStart + j * npt + i] < diffLikUjk) {
    boundsMatrices[matrixStart + j * npt + i] = diffLikUjk;
  } else if (boundsMatrices[matrixStart + j * npt + i] < diffLjkUik) {
    boundsMatrices[matrixStart + j * npt + i] = diffLjkUik;
  }

  // Final consistency check (matches original algorithm)
  const double lBound = boundsMatrices[matrixStart + j * npt + i];
  const double uBound = boundsMatrices[matrixStart + i * npt + j];

  if (tol > 0.0 && (lBound - uBound) > 0.0 && (lBound - uBound) / lBound < tol) {
    // Adjust the upper bound to match lower bound
    boundsMatrices[matrixStart + i * npt + j] = lBound;
  } else if (lBound - uBound > 0.0) {
    // Inconsistent bounds - signal failure
    needSmoothing[molIdx] = 2;
  }
}

//! Host function to launch batch triangle smoothing on GPU
void triangleSmoothBoundsBatch(DeviceBoundsMatrixBatch& deviceMatrixBatch,
                               std::vector<uint8_t>*    needSmoothing,
                               double                   tol,
                               cudaStream_t             stream) {
  const unsigned int numMatrices = deviceMatrixBatch.numMatrices();
  if (numMatrices == 0) {
    return;  // Trivial case
  }

  if (needSmoothing && needSmoothing->size() != numMatrices) {
    throw std::runtime_error("needSmoothing vector size doesn't match number of matrices");
  }

  // Find the maximum matrix size to determine the maximum k value
  unsigned int maxNpt = 0;
  for (unsigned int i = 0; i < numMatrices; ++i) {
    maxNpt = std::max(maxNpt, deviceMatrixBatch.matrixSize(i));
  }

  if (maxNpt <= 1) {
    return;  // All matrices are trivial
  }

  // Device memory for needSmoothing flags
  AsyncDeviceVector<uint8_t> d_needSmoothing(numMatrices, stream);
  std::vector<uint8_t>       allOnes;  // Keep in scope for async copy
  if (needSmoothing) {
    d_needSmoothing.copyFromHost(*needSmoothing);
  } else {
    // Set all elements to 1 (all molecules need smoothing)
    allOnes.resize(numMatrices, 1);
    d_needSmoothing.copyFromHost(allOnes);
  }

  // Configure 1D grid and block dimensions
  const unsigned int blockSize    = 256;  // Optimal block size for most GPUs
  const unsigned int totalThreads = static_cast<unsigned int>(deviceMatrixBatch.totalElements());
  const unsigned int numBlocks    = (totalThreads + blockSize - 1) / blockSize;

  // Main triangle smoothing loop over k (up to maximum matrix size)
  for (unsigned int k = 0; k < maxNpt; k++) {
    // Launch triangle smoothing kernel for this k
    triangleSmoothBatchKernel<<<numBlocks, blockSize, 0, stream>>>(deviceMatrixBatch.data(),
                                                                   deviceMatrixBatch.matrixStarts(),
                                                                   deviceMatrixBatch.molIndices(),
                                                                   totalThreads,  // Pass totalElements
                                                                   k,
                                                                   d_needSmoothing.data(),
                                                                   tol);
    cudaCheckError(cudaGetLastError());
  }

  // Synchronize after all k iterations
  cudaCheckError(cudaStreamSynchronize(stream));

  // Copy results back to host
  if (needSmoothing) {
    d_needSmoothing.copyToHost(*needSmoothing);
    cudaStreamSynchronize(stream);

    // Convert the values: 1 → 0 (success), 2 → 1 (needs further smoothing)
    for (uint8_t& needs : *needSmoothing) {
      if (needs == 1) {
        needs = 0;  // Molecules that still had 1 never failed, so they succeeded
      } else if (needs == 2) {
        needs = 1;  // Molecules marked as 2 failed, so they still need smoothing
      }
    }
  }
}

//! Convenience function that works with vector of RDKit BoundsMatrix
void triangleSmoothBoundsBatch(std::vector<::DistGeom::BoundsMatPtr>& boundsMatrices,
                               std::vector<uint8_t>*                  needSmoothing,
                               double                                 tol,
                               cudaStream_t                           stream) {
  if (boundsMatrices.empty()) {
    return;
  }

  if (needSmoothing && needSmoothing->size() != boundsMatrices.size()) {
    throw std::runtime_error("needSmoothing vector size doesn't match number of matrices");
  }

  // Extract matrix sizes
  std::vector<unsigned int> matrixSizes;
  matrixSizes.reserve(boundsMatrices.size());
  for (const auto& matrix : boundsMatrices) {
    if (!matrix) {
      throw std::runtime_error("Null matrix encountered in bounds matrices vector");
    }
    matrixSizes.push_back(static_cast<unsigned int>(matrix->numRows()));
  }

  // Create device batch matrix and copy data (only for molecules that need smoothing)
  DeviceBoundsMatrixBatch deviceMatrixBatch(matrixSizes);
  deviceMatrixBatch.copyFromHost(boundsMatrices, needSmoothing);

  // Keep a copy of original needSmoothing to know what was processed (only if we have selective smoothing)
  std::vector<uint8_t> beforeSmoothing;
  if (needSmoothing) {
    beforeSmoothing = *needSmoothing;
  }

  // Perform GPU batch triangle smoothing
  triangleSmoothBoundsBatch(deviceMatrixBatch, needSmoothing, tol, stream);

  // Copy results back to host
  if (needSmoothing) {
    // Selective copying - only copy back molecules that converged
    deviceMatrixBatch.copyToHost(boundsMatrices, &beforeSmoothing, needSmoothing);
  } else {
    // Copy everything back when no selective smoothing
    deviceMatrixBatch.copyToHost(boundsMatrices, nullptr, nullptr);
  }
}

//! DeviceBoundsMatrixBatch implementation
DeviceBoundsMatrixBatch::DeviceBoundsMatrixBatch(const std::vector<unsigned int>& matrixSizes)
    : matrixSizes_(matrixSizes),
      totalElements_(0) {
  if (matrixSizes_.empty()) {
    throw std::runtime_error("Cannot create empty batch of matrices");
  }

  // Calculate total elements and set up matrix starts
  std::vector<int> hostMatrixStarts;
  std::vector<int> hostMolIndices;
  hostMatrixStarts.reserve(matrixSizes_.size() + 1);
  hostMatrixStarts.push_back(0);

  for (size_t molIdx = 0; molIdx < matrixSizes_.size(); ++molIdx) {
    const unsigned int matrixSize  = matrixSizes_[molIdx];
    const size_t       numElements = static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize);

    // Add molecule indices for all elements in this matrix
    for (size_t elemIdx = 0; elemIdx < numElements; ++elemIdx) {
      hostMolIndices.push_back(static_cast<int>(molIdx));
    }

    totalElements_ += numElements;
    hostMatrixStarts.push_back(static_cast<int>(totalElements_));
  }

  // Initialize device vectors
  data_.resize(totalElements_);
  data_.zero();

  matrixStarts_.resize(hostMatrixStarts.size());
  matrixStarts_.setFromVector(hostMatrixStarts);

  molIndices_.resize(hostMolIndices.size());
  molIndices_.setFromVector(hostMolIndices);
  cudaStreamSynchronize(matrixStarts_.stream());  // Sync before local vectors go out of scope
}

void DeviceBoundsMatrixBatch::copyFromHost(const std::vector<::DistGeom::BoundsMatPtr>& hostMatrices,
                                           const std::vector<uint8_t>*                  needSmoothing) {
  if (hostMatrices.size() != matrixSizes_.size()) {
    throw std::runtime_error("Number of host matrices doesn't match batch size");
  }

  if (needSmoothing && needSmoothing->size() != matrixSizes_.size()) {
    throw std::runtime_error("needSmoothing vector size doesn't match batch size");
  }

  // Prepare flattened host data
  std::vector<double> hostData(totalElements_);
  size_t              globalOffset = 0;

  for (size_t molIdx = 0; molIdx < hostMatrices.size(); ++molIdx) {
    const auto&        hostMatrix = hostMatrices[molIdx];
    const unsigned int matrixSize = matrixSizes_[molIdx];

    if (hostMatrix->numRows() != matrixSize || hostMatrix->numCols() != matrixSize) {
      throw std::runtime_error("Matrix size mismatch in copyFromHost for molecule " + std::to_string(molIdx));
    }

    // Copy matrix data if needSmoothing is null (copy all) or if this molecule needs smoothing
    if (!needSmoothing || (*needSmoothing)[molIdx]) {
      for (unsigned int i = 0; i < matrixSize; ++i) {
        for (unsigned int j = 0; j < matrixSize; ++j) {
          hostData[globalOffset + i * matrixSize + j] = hostMatrix->getVal(i, j);
        }
      }
    }

    globalOffset += static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize);
  }

  // Copy to device
  data_.setFromVector(hostData);
  cudaStreamSynchronize(data_.stream());  // Sync before local hostData goes out of scope
}

void DeviceBoundsMatrixBatch::copyToHost(std::vector<::DistGeom::BoundsMatPtr>& hostMatrices,
                                         const std::vector<uint8_t>*            beforeSmoothing,
                                         const std::vector<uint8_t>*            afterSmoothing) const {
  // Copy device data to host vector first
  std::vector<double> hostData(totalElements_);
  data_.copyToHost(hostData);
  cudaCheckError(cudaStreamSynchronize(data_.stream()));

  // Distribute data to individual matrices
  size_t globalOffset = 0;

  for (size_t molIdx = 0; molIdx < hostMatrices.size(); ++molIdx) {
    auto&              hostMatrix = hostMatrices[molIdx];
    const unsigned int matrixSize = matrixSizes_[molIdx];

    if (hostMatrix->numRows() != matrixSize || hostMatrix->numCols() != matrixSize) {
      throw std::runtime_error("Matrix size mismatch in copyToHost for molecule " + std::to_string(molIdx));
    }

    bool shouldCopy = false;
    if (!beforeSmoothing && !afterSmoothing) {
      // Both null - copy everything
      shouldCopy = true;
    } else if (beforeSmoothing && afterSmoothing) {
      // Both provided - copy only molecules that converged (needed smoothing before but not after)
      shouldCopy = (*beforeSmoothing)[molIdx] && !(*afterSmoothing)[molIdx];
    } else {
      // Only one provided - this shouldn't happen in normal usage, but default to copying
      shouldCopy = true;
    }

    if (shouldCopy) {
      for (unsigned int i = 0; i < matrixSize; ++i) {
        for (unsigned int j = 0; j < matrixSize; ++j) {
          hostMatrix->setVal(i, j, hostData[globalOffset + i * matrixSize + j]);
        }
      }
    }

    globalOffset += static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize);
  }
}

}  // namespace nvMolKit
