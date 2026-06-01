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

#ifndef NVMOLKIT_SYMMETRIC_EIGENSOLVER_H_
#define NVMOLKIT_SYMMETRIC_EIGENSOLVER_H_

#include <memory>
#include <vector>

namespace nvMolKit {

//! Batched symmetric eigensolver implemented as a custom power-iteration CUDA kernel.
class BatchedEigenSolver {
 public:
  BatchedEigenSolver();
  ~BatchedEigenSolver();

  BatchedEigenSolver(const BatchedEigenSolver&)            = delete;
  BatchedEigenSolver& operator=(const BatchedEigenSolver&) = delete;
  BatchedEigenSolver(BatchedEigenSolver&&)                 = default;

  //! Compute eigenvalues and eigenvectors of a batch of symmetric matrices
  //!
  //! \param numEigs: Number of eigenvalues to compute. Must be less than or equal to n.
  //! \param matrixDim: Matrix dimension
  //! \param batch_size: Number of matrices in the batch
  //! \param matrices: Pointer to the first element of the array of matrices on device. Dimensions n * n * batchsize.
  //! Overwritten by the algorithm.
  //! \param eigenvalues: Pointer to the first element of the array of eigenvalues on
  //! device. Dimensions n * batchsize
  //! \param eigenvectors: Pointer to the first element of the array of eigenvectors on
  //! device. Dimensions n * n * batchsize
  //! \param active: Optional pointer to an array of uint8_t on device. Dimensions batchSize. If nullptr, all matrices
  //! are processed. If not nullptr, only the matrices with active[i] == 1 are processed. \param seed: Random seed for
  //! the algorithm.
  void solve(int            numEigs,
             int            matrixDim,
             int            batch_size,
             double*        matrices,
             double*        eigenvalues,
             double*        eigenvectors,
             const uint8_t* active     = nullptr,
             int            randomSeed = 42);

  const uint8_t* converged() const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_SYMMETRIC_EIGENSOLVER_H_
