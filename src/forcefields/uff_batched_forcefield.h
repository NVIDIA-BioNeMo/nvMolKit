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

#ifndef NVMOLKIT_UFF_BATCHED_FORCEFIELD_H
#define NVMOLKIT_UFF_BATCHED_FORCEFIELD_H

#include <variant>

#include "src/forcefields/batched_forcefield.h"
#include "src/forcefields/precision_workspace.h"
#include "src/forcefields/uff.h"
#include "src/precision/precision_mode.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

class UFFBatchedForcefield final : public BatchedForcefield, public SinglePrecisionBatchedForcefield {
 public:
  explicit UFFBatchedForcefield(const UFF::BatchedMolecularSystemHost& molSystemHost,
                                BatchedForcefieldMetadata              metadata  = {},
                                cudaStream_t                           stream    = nullptr,
                                PrecisionMode                          precision = PrecisionMode::FULL);

  cudaError_t computeEnergy(double*        energyOuts,
                            const double*  positions,
                            const uint8_t* activeSystemMask = nullptr,
                            cudaStream_t   stream           = nullptr) override;

  cudaError_t   computeGradients(double*        grad,
                                 const double*  positions,
                                 const uint8_t* activeSystemMask = nullptr,
                                 cudaStream_t   stream           = nullptr) override;
  cudaError_t   computeEnergy(float*         energyOuts,
                              const float*   positions,
                              const uint8_t* activeSystemMask = nullptr,
                              cudaStream_t   stream           = nullptr) override;
  cudaError_t   computeGradients(float*         grad,
                                 const float*   positions,
                                 const uint8_t* activeSystemMask = nullptr,
                                 cudaStream_t   stream           = nullptr) override;
  PrecisionMode precision() const override;

 private:
  std::variant<UFF::BatchedMolecularDeviceBuffers, UFF::BatchedMolecularDeviceBuffersSingle> systemDevice_;
  SingleForcefieldConversionWorkspace                                                        singleConversion_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_UFF_BATCHED_FORCEFIELD_H
