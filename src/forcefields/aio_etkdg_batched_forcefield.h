// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_AIO_ETKDG_BATCHED_FORCEFIELD_H
#define NVMOLKIT_AIO_ETKDG_BATCHED_FORCEFIELD_H

#include <variant>

#include "src/forcefields/batched_forcefield.h"
#include "src/forcefields/dist_geom.h"
#include "src/forcefields/precision_workspace.h"
#include "src/precision/precision_mode.h"

namespace nvMolKit {

//! \brief Batched-forcefield adapter for RDKit's all-in-one ETKDG coordinate refinement.
//!
//! Evaluates the 4D distance-geometry terms together with the all-in-one harmonic distance, linear angle,
//! experimental torsion and planarity terms. The torsion and planarity terms start disabled, matching RDKit's first
//! minimization phase.
class AllInOneETKDGBatchedForcefield final : public BatchedForcefield, public SinglePrecisionBatchedForcefield {
 public:
  //! \param systemHost Flattened all-in-one host-side system description.
  //! \param atomStartsHost Host-side atom offsets for the minimized systems.
  //! \param weights Force constants shared by every system in the batch.
  //! \param metadata Optional mapping from concrete systems back to logical molecules/conformers.
  //! \param stream CUDA stream used for internal device allocations and uploads.
  //! \param precision Arithmetic precision of the device buffers and kernels.
  AllInOneETKDGBatchedForcefield(const DistGeom::BatchedAllInOneSystemHost& systemHost,
                                 const std::vector<int>&                    atomStartsHost,
                                 const DistGeom::AllInOneForceWeights&      weights,
                                 BatchedForcefieldMetadata                  metadata  = {},
                                 cudaStream_t                               stream    = nullptr,
                                 PrecisionMode                              precision = PrecisionMode::FULL);

  cudaError_t   computeEnergy(double*        energyOuts,
                              const double*  positions,
                              const uint8_t* activeSystemMask = nullptr,
                              cudaStream_t   stream           = nullptr) override;
  cudaError_t   computeGradients(double*        grad,
                                 const double*  positions,
                                 const uint8_t* activeSystemMask = nullptr,
                                 cudaStream_t   stream           = nullptr) override;
  cudaError_t   computeEnergy(float*, const float*, const uint8_t* = nullptr, cudaStream_t = nullptr) override;
  cudaError_t   computeGradients(float*, const float*, const uint8_t* = nullptr, cudaStream_t = nullptr) override;
  PrecisionMode precision() const override {
    return std::holds_alternative<DistGeom::BatchedAllInOneDeviceBuffersSingle>(systemDevice_) ? PrecisionMode::SINGLE :
                                                                                                 PrecisionMode::FULL;
  }

  //! Enables the experimental torsion and planarity terms for RDKit's second minimization phase.
  void setTorsionTermsEnabled(bool enabled) { torsionTermsEnabled_ = enabled; }

 private:
  std::variant<DistGeom::BatchedAllInOneDeviceBuffers, DistGeom::BatchedAllInOneDeviceBuffersSingle> systemDevice_;
  AsyncDeviceVector<int>                                                                             atomStartsDevice_;
  SingleForcefieldConversionWorkspace                                                                singleConversion_;
  DistGeom::AllInOneForceWeights                                                                     weights_;
  bool torsionTermsEnabled_ = false;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_AIO_ETKDG_BATCHED_FORCEFIELD_H
