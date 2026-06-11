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

#ifndef NVMOLKIT_ETK_BATCHED_FORCEFIELD_H
#define NVMOLKIT_ETK_BATCHED_FORCEFIELD_H

#include "src/forcefields/batched_forcefield.h"
#include "src/forcefields/dist_geom.h"

namespace nvMolKit {

//! \brief Batched-forcefield adapter for ETK 3D systems.
//!
//! This wrapper exposes ETK host-side data through the generic
//! `BatchedForcefield` interface so batched BFGS can evaluate ETK energies and
//! gradients without ETK-specific dispatch in the minimizer.
class ETKBatchedForcefield final : public BatchedForcefield {
 public:
  //! \brief Builds a generic batched-forcefield view over ETK host data.
  //! \param molSystemHost Flattened ETK host-side system description.
  //! \param atomStartsHost Host-side atom offsets for the minimized systems.
  //! \param useBasicKnowledge Selects ETK `ALL` terms or the `PLAIN` subset.
  //! \param metadata Optional mapping from concrete systems back to logical molecules/conformers.
  //! \param stream CUDA stream used for internal device allocations and uploads.
  ETKBatchedForcefield(const DistGeom::BatchedMolecularSystem3DHost& molSystemHost,
                       const std::vector<int>&                       atomStartsHost,
                       bool                                          useBasicKnowledge,
                       BatchedForcefieldMetadata                     metadata = {},
                       cudaStream_t                                  stream   = nullptr);

  //! \brief Computes ETK energies through the generic batched-forcefield API.
  cudaError_t computeEnergy(double*        energyOuts,
                            const double*  positions,
                            const uint8_t* activeSystemMask = nullptr,
                            cudaStream_t   stream           = nullptr) override;

  //! \brief Computes ETK gradients through the generic batched-forcefield API.
  cudaError_t computeGradients(double*        grad,
                               const double*  positions,
                               const uint8_t* activeSystemMask = nullptr,
                               cudaStream_t   stream           = nullptr) override;

  //! \brief Computes the planar ETK subset used by the post-minimization check.
  cudaError_t computePlanarEnergy(double*        energyOuts,
                                  const double*  positions,
                                  const uint8_t* activeSystemMask = nullptr,
                                  cudaStream_t   stream           = nullptr);

  //! \brief Returns the uploaded ETK contribution buffers for auxiliary kernels.
  const DistGeom::Energy3DForceContribsDevice& contribs() const { return systemDevice_.contribs; }

 private:
  DistGeom::BatchedMolecular3DDeviceBuffers systemDevice_;
  AsyncDeviceVector<int>                    atomStartsDevice_;
  DistGeom::ETKTerm                         term_ = DistGeom::ETKTerm::ALL;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_ETK_BATCHED_FORCEFIELD_H
