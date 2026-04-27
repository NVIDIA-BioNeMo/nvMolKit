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

#ifndef NVMOLKIT_DG_BATCHED_FORCEFIELD_H
#define NVMOLKIT_DG_BATCHED_FORCEFIELD_H

#include "src/forcefields/batched_forcefield.h"
#include "src/forcefields/dist_geom.h"

namespace nvMolKit {

//! \brief Batched-forcefield adapter for 4D distance-geometry systems.
//!
//! This wrapper exposes DG host-side data through the generic
//! `BatchedForcefield` interface so batched BFGS can call into DG energy and
//! gradient evaluation without keeping DG-specific dispatch in the minimizer.
class DGBatchedForcefield final : public BatchedForcefield {
 public:
  //! \brief Builds a generic batched-forcefield view over DG host data.
  //! \param molSystemHost Flattened DG host-side system description.
  //! \param atomStartsHost Host-side atom offsets for the minimized systems.
  //! \param chiralWeight Weight applied to the DG chirality term.
  //! \param fourthDimWeight Weight applied to the DG fourth-dimension term.
  //! \param metadata Optional mapping from concrete systems back to logical molecules/conformers.
  //! \param stream CUDA stream used for internal device allocations and uploads.
  DGBatchedForcefield(const DistGeom::BatchedMolecularSystemHost& molSystemHost,
                      const std::vector<int>&                     atomStartsHost,
                      double                                      chiralWeight,
                      double                                      fourthDimWeight,
                      BatchedForcefieldMetadata                   metadata = {},
                      cudaStream_t                                stream   = nullptr);

  //! \brief Computes DG energies through the generic batched-forcefield API.
  cudaError_t computeEnergy(double*        energyOuts,
                            const double*  positions,
                            const uint8_t* activeSystemMask = nullptr,
                            cudaStream_t   stream           = nullptr) override;

  //! \brief Computes DG gradients through the generic batched-forcefield API.
  cudaError_t computeGradients(double*        grad,
                               const double*  positions,
                               const uint8_t* activeSystemMask = nullptr,
                               cudaStream_t   stream           = nullptr) override;

 private:
  DistGeom::BatchedMolecularDeviceBuffers systemDevice_;
  AsyncDeviceVector<int>                  atomStartsDevice_;
  double                                  chiralWeight_    = 1.0;
  double                                  fourthDimWeight_ = 0.1;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_DG_BATCHED_FORCEFIELD_H
