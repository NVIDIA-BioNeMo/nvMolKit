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

#ifndef NVMOLKIT_MMFF_BATCHED_FORCEFIELD_H
#define NVMOLKIT_MMFF_BATCHED_FORCEFIELD_H

#include <variant>

#include "src/forcefields/batched_forcefield.h"
#include "src/forcefields/mmff.h"
#include "src/forcefields/precision_workspace.h"
#include "src/precision/precision_mode.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

//! \brief Batched-forcefield adapter for MMFF systems.
//!
//! This wrapper exposes an `MMFF::BatchedMolecularSystemHost` through the
//! generic `BatchedForcefield` interface so host-driven batched BFGS can
//! evaluate MMFF energies and gradients without MMFF-specific dispatch code.
class MMFFBatchedForcefield final : public BatchedForcefield, public SinglePrecisionBatchedForcefield {
 public:
  //! \brief Builds a generic batched-forcefield view over MMFF host data.
  //! \param molSystemHost Flattened MMFF host-side system description.
  //! \param metadata Optional mapping from concrete systems back to logical molecules/conformers.
  //! \param stream CUDA stream used for internal device allocations and uploads.
  explicit MMFFBatchedForcefield(const MMFF::BatchedMolecularSystemHost& molSystemHost,
                                 BatchedForcefieldMetadata               metadata  = {},
                                 cudaStream_t                            stream    = nullptr,
                                 PrecisionMode                           precision = PrecisionMode::FULL);

  //! \brief Computes MMFF energies through the generic batched-forcefield API.
  cudaError_t computeEnergy(double*        energyOuts,
                            const double*  positions,
                            const uint8_t* activeSystemMask = nullptr,
                            cudaStream_t   stream           = nullptr) override;

  //! \brief Computes MMFF gradients through the generic batched-forcefield API.
  cudaError_t computeGradients(double*        grad,
                               const double*  positions,
                               const uint8_t* activeSystemMask = nullptr,
                               cudaStream_t   stream           = nullptr) override;
  cudaError_t computeEnergy(double*        energyOuts,
                            const float*   positions,
                            const uint8_t* activeSystemMask = nullptr,
                            cudaStream_t   stream           = nullptr) override;
  cudaError_t computeGradients(float*         grad,
                               const float*   positions,
                               const uint8_t* activeSystemMask = nullptr,
                               cudaStream_t   stream           = nullptr) override;

 private:
  std::variant<MMFF::BatchedMolecularDeviceBuffers, MMFF::BatchedMolecularDeviceBuffersF32> systemDevice_;
  FullForcefieldConversionWorkspace                                                         fullConversion_;
  SingleForcefieldConversionWorkspace                                                       singleConversion_;
  bool                                                                                      singlePrecision_ = false;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_MMFF_BATCHED_FORCEFIELD_H
