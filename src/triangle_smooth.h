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

#ifndef NVMOLKIT_TRIANGLE_SMOOTH_H
#define NVMOLKIT_TRIANGLE_SMOOTH_H

#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>

#include <memory>
#include <vector>

#include "src/utils/device_vector.h"
namespace nvMolKit {

//! GPU-compatible batch bounds matrix buffer for triangle smoothing operations on multiple molecules
//! Stores multiple bounds matrices in a flattened format for efficient batch processing
class DeviceBoundsMatrixBatch {
 public:
  //! Constructor for batch of matrices with given sizes
  explicit DeviceBoundsMatrixBatch(const std::vector<unsigned int>& matrixSizes);

  //! Get the number of matrices in the batch
  unsigned int numMatrices() const noexcept { return static_cast<unsigned int>(matrixSizes_.size()); }

  //! Get the size of a specific matrix by index
  unsigned int matrixSize(unsigned int matrixIdx) const {
    if (matrixIdx >= matrixSizes_.size()) {
      throw std::runtime_error("Matrix index out of bounds");
    }
    return matrixSizes_[matrixIdx];
  }

  //! Get the total number of elements across all matrices
  size_t totalElements() const noexcept { return totalElements_; }

  //! Get raw device pointer to the flattened data
  double* data() noexcept { return data_.data(); }

  //! Get raw device pointer to the matrix starts array
  int* matrixStarts() noexcept { return matrixStarts_.data(); }

  //! Get raw device pointer to the molecule index mapping array
  int* molIndices() noexcept { return molIndices_.data(); }

  //! Get the underlying AsyncDeviceVectors
  AsyncDeviceVector<double>&       deviceData() noexcept { return data_; }
  const AsyncDeviceVector<double>& deviceData() const noexcept { return data_; }
  AsyncDeviceVector<int>&          deviceMatrixStarts() noexcept { return matrixStarts_; }
  const AsyncDeviceVector<int>&    deviceMatrixStarts() const noexcept { return matrixStarts_; }
  AsyncDeviceVector<int>&          deviceMolIndices() noexcept { return molIndices_; }
  const AsyncDeviceVector<int>&    deviceMolIndices() const noexcept { return molIndices_; }

  //! Zero out all matrices
  void zero() { data_.zero(); }

  //! Copy data from host RDKit BoundsMatrix vector
  void copyFromHost(const std::vector<::DistGeom::BoundsMatPtr>& hostMatrices,
                    const std::vector<uint8_t>*                  needSmoothing = nullptr);

  //! Copy data to host RDKit BoundsMatrix vector
  void copyToHost(std::vector<::DistGeom::BoundsMatPtr>& hostMatrices,
                  const std::vector<uint8_t>*            beforeSmoothing = nullptr,
                  const std::vector<uint8_t>*            afterSmoothing  = nullptr) const;

 private:
  std::vector<unsigned int> matrixSizes_;    //! Size of each matrix in the batch
  size_t                    totalElements_;  //! Total number of elements across all matrices
  AsyncDeviceVector<double> data_;           //! Flattened device storage for all matrix data
  AsyncDeviceVector<int>    matrixStarts_;   //! Start indices for each matrix (size: numMatrices + 1)
  AsyncDeviceVector<int>    molIndices_;     //! Molecule index for each element in flattened data
};

//! GPU-accelerated batch triangle smoothing function
//! @param deviceMatrixBatch Device batch bounds matrices to be smoothed
//! @param needSmoothing Host vector indicating which molecules need smoothing (1=need smoothing, 0=skip), or nullptr
//! for all molecules
//! @param tol Tolerance for bound consistency (default: 0.0)
//! @param stream CUDA stream for asynchronous execution (default: nullptr for default stream)
void triangleSmoothBoundsBatch(DeviceBoundsMatrixBatch& deviceMatrixBatch,
                               std::vector<uint8_t>*    needSmoothing = nullptr,
                               double                   tol           = 0.0,
                               cudaStream_t             stream        = nullptr);

//! Convenience function that works with vector of RDKit BoundsMatrix
//! @param boundsMatrices Vector of host RDKit bounds matrices to be smoothed
//! @param needSmoothing Host vector indicating which molecules need smoothing (1=need smoothing, 0=skip), or nullptr
//! for all molecules
//! @param tol Tolerance for bound consistency (default: 0.0)
//! @param stream CUDA stream for asynchronous execution (default: nullptr for default stream)
void triangleSmoothBoundsBatch(std::vector<::DistGeom::BoundsMatPtr>& boundsMatrices,
                               std::vector<uint8_t>*                  needSmoothing = nullptr,
                               double                                 tol           = 0.0,
                               cudaStream_t                           stream        = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_TRIANGLE_SMOOTH_H
