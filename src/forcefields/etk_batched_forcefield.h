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

#include <variant>

#include "src/forcefields/batched_forcefield.h"
#include "src/forcefields/dist_geom.h"
#include "src/forcefields/precision_workspace.h"
#include "src/precision/precision_mode.h"

namespace nvMolKit {

//! \brief Batched-forcefield adapter for ETK 3D systems.
//!
//! This wrapper exposes ETK host-side data through the generic
//! `BatchedForcefield` interface so batched BFGS can evaluate ETK energies and
//! gradients without ETK-specific dispatch in the minimizer.
class ETKBatchedForcefield final : public BatchedForcefield, public SinglePrecisionBatchedForcefield {
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
                       BatchedForcefieldMetadata                     metadata  = {},
                       cudaStream_t                                  stream    = nullptr,
                       PrecisionMode                                 precision = PrecisionMode::FULL);

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
  cudaError_t computeEnergy(float*, const float*, const uint8_t* = nullptr, cudaStream_t = nullptr) override;
  cudaError_t computeGradients(float*, const float*, const uint8_t* = nullptr, cudaStream_t = nullptr) override;

  //! \brief Computes the planar ETK subset used by the post-minimization check.
  cudaError_t computePlanarEnergy(double*        energyOuts,
                                  const double*  positions,
                                  const uint8_t* activeSystemMask = nullptr,
                                  cudaStream_t   stream           = nullptr);

  //! \brief Returns the full-precision ETK contribution buffers for auxiliary kernels.
  //! \pre This force field was constructed with PrecisionMode::FULL.
  const DistGeom::Energy3DForceContribsDevice& contribs() const {
    return std::get<DistGeom::BatchedMolecular3DDeviceBuffers>(systemDevice_).contribs;
  }

  //! \brief Visits uploaded ETK contribution buffers in their native precision.
  template <typename Visitor> decltype(auto) visitContribs(Visitor&& visitor) const {
    return std::visit([&](const auto& buffers) -> decltype(auto) { return visitor(buffers.contribs); }, systemDevice_);
  }

 private:
  std::variant<DistGeom::BatchedMolecular3DDeviceBuffers, DistGeom::BatchedMolecular3DDeviceBuffersSingle>
                                      systemDevice_;
  AsyncDeviceVector<int>              atomStartsDevice_;
  SingleForcefieldConversionWorkspace singleConversion_;
  DistGeom::ETKTerm                   term_ = DistGeom::ETKTerm::ALL;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_ETK_BATCHED_FORCEFIELD_H
