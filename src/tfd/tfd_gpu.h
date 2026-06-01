// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_TFD_GPU_H
#define NVMOLKIT_TFD_GPU_H

#include <vector>

#include "src/tfd/tfd_common.h"
#include "src/utils/device.h"

namespace nvMolKit {

//! Result structure for GPU TFD computation with GPU-resident data
struct TFDGpuResult {
  //! GPU-resident TFD values (flattened across all molecules)
  AsyncDeviceVector<float> tfdValues;

  //! CSR index: TFD output boundaries per molecule [nMols + 1]
  std::vector<int> tfdOutputStarts;

  //! Number of conformers per molecule [nMols]
  std::vector<int> conformerCounts;

  //! Extract TFD matrix for a single molecule (copies to host)
  //! @param molIdx Index of the molecule in the batch
  //! @return Lower triangular TFD matrix as flat vector
  std::vector<double> extractMolecule(int molIdx) const;

  //! Extract all TFD matrices (copies to host)
  //! @return Vector of TFD matrices, one per molecule
  std::vector<std::vector<double>> extractAll() const;
};

//! GPU implementation of TFD computation
class TFDGpuGenerator {
 public:
  TFDGpuGenerator();

  //! Compute TFD matrix for a single molecule
  //! @param mol Molecule with conformers
  //! @param options Computation options
  //! @return Lower triangular TFD matrix as flat vector [C*(C-1)/2 values]
  std::vector<double> GetTFDMatrix(const RDKit::ROMol& mol, const TFDComputeOptions& options = TFDComputeOptions{});

  //! Compute TFD matrices for multiple molecules
  //! @param mols Vector of molecules
  //! @param options Computation options
  //! @return Vector of TFD matrices, one per molecule
  std::vector<std::vector<double>> GetTFDMatrices(const std::vector<const RDKit::ROMol*>& mols,
                                                  const TFDComputeOptions& options = TFDComputeOptions{});

  //! Compute TFD matrices and keep results on GPU
  //! @param mols Vector of molecules
  //! @param options Computation options
  //! @return TFDGpuResult with GPU-resident data
  TFDGpuResult GetTFDMatricesGpuBuffer(const std::vector<const RDKit::ROMol*>& mols,
                                       const TFDComputeOptions&                options = TFDComputeOptions{});

 private:
  // NOTE: Declaration order matters! Stream must be declared BEFORE device_
  // because members are destroyed in reverse order, and device_ buffers need
  // a valid stream for cudaFreeAsync during destruction.
  ScopedStream    stream_;
  TFDSystemDevice device_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_TFD_GPU_H
