// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/forcefields/aio_etkdg_batched_forcefield.h"
#include "src/forcefields/dist_geom_kernels.h"
#include "src/utils/device_convert.cuh"

namespace nvMolKit {

AllInOneETKDGBatchedForcefield::AllInOneETKDGBatchedForcefield(const DistGeom::BatchedAllInOneSystemHost& systemHost,
                                                               const std::vector<int>&               atomStartsHost,
                                                               const DistGeom::AllInOneForceWeights& weights,
                                                               BatchedForcefieldMetadata             metadata,
                                                               const cudaStream_t                    stream,
                                                               const PrecisionMode                   precision)
    : BatchedForcefield(ForceFieldType::AIO_ETKDG, 4, atomStartsHost, nullptr, std::move(metadata)),
      weights_(weights) {
  atomStartsDevice_.setStream(stream);
  if (usesSinglePrecision(precision)) {
    DistGeom::sendAllInOneSystemToDevice(systemHost,
                                         systemDevice_.emplace<DistGeom::BatchedAllInOneDeviceBuffersSingle>(),
                                         stream);
  } else {
    DistGeom::sendAllInOneSystemToDevice(systemHost,
                                         systemDevice_.emplace<DistGeom::BatchedAllInOneDeviceBuffers>(),
                                         stream);
  }
  atomStartsDevice_.setFromVector(atomStartsHost);
  setAtomStartsDevice(atomStartsDevice_.data());
}

cudaError_t AllInOneETKDGBatchedForcefield::computeEnergy(double*        energyOuts,
                                                          const double*  positions,
                                                          const uint8_t* activeSystemMask,
                                                          cudaStream_t   stream) {
  if (std::holds_alternative<DistGeom::BatchedAllInOneDeviceBuffersSingle>(systemDevice_)) {
    singleConversion_.positions.setStream(stream);
    singleConversion_.energies.setStream(stream);
    singleConversion_.positions.resize(totalPositions());
    singleConversion_.energies.resize(numMolecules());
    auto err = detail::convertDeviceArray(singleConversion_.positions.data(), positions, totalPositions(), stream);
    if (err == cudaSuccess) {
      err =
        computeEnergy(singleConversion_.energies.data(), singleConversion_.positions.data(), activeSystemMask, stream);
    }
    return err == cudaSuccess ?
             detail::convertDeviceArray(energyOuts, singleConversion_.energies.data(), numMolecules(), stream) :
             err;
  }
  const auto& buffers = std::get<DistGeom::BatchedAllInOneDeviceBuffers>(systemDevice_);
  return DistGeom::launchBlockPerMolEnergyKernelAIO(
    numMolecules(),
    DistGeom::toAllInOneContribsDevicePtr(buffers, atomStartsDevice_.data()),
    positions,
    energyOuts,
    weights_,
    torsionTermsEnabled_,
    activeSystemMask,
    stream);
}

cudaError_t AllInOneETKDGBatchedForcefield::computeGradients(double*        grad,
                                                             const double*  positions,
                                                             const uint8_t* activeSystemMask,
                                                             cudaStream_t   stream) {
  if (std::holds_alternative<DistGeom::BatchedAllInOneDeviceBuffersSingle>(systemDevice_)) {
    singleConversion_.positions.setStream(stream);
    singleConversion_.gradients.setStream(stream);
    singleConversion_.positions.resize(totalPositions());
    singleConversion_.gradients.resize(totalPositions());
    auto err = detail::convertDeviceArray(singleConversion_.positions.data(), positions, totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    singleConversion_.gradients.zero();
    err = computeGradients(singleConversion_.gradients.data(),
                           singleConversion_.positions.data(),
                           activeSystemMask,
                           stream);
    return err == cudaSuccess ?
             detail::convertDeviceArray(grad, singleConversion_.gradients.data(), totalPositions(), stream) :
             err;
  }
  const auto& buffers = std::get<DistGeom::BatchedAllInOneDeviceBuffers>(systemDevice_);
  return DistGeom::launchBlockPerMolGradKernelAIO(
    numMolecules(),
    DistGeom::toAllInOneContribsDevicePtr(buffers, atomStartsDevice_.data()),
    positions,
    grad,
    weights_,
    torsionTermsEnabled_,
    activeSystemMask,
    stream);
}

cudaError_t AllInOneETKDGBatchedForcefield::computeEnergy(float*         energyOuts,
                                                          const float*   positions,
                                                          const uint8_t* activeSystemMask,
                                                          cudaStream_t   stream) {
  if (!std::holds_alternative<DistGeom::BatchedAllInOneDeviceBuffersSingle>(systemDevice_)) {
    return cudaErrorInvalidValue;
  }
  const auto& buffers = std::get<DistGeom::BatchedAllInOneDeviceBuffersSingle>(systemDevice_);
  return DistGeom::launchBlockPerMolEnergyKernelAIO(
    numMolecules(),
    DistGeom::toAllInOneContribsDevicePtr(buffers, atomStartsDevice_.data()),
    positions,
    energyOuts,
    weights_,
    torsionTermsEnabled_,
    activeSystemMask,
    stream);
}

cudaError_t AllInOneETKDGBatchedForcefield::computeGradients(float*         grad,
                                                             const float*   positions,
                                                             const uint8_t* activeSystemMask,
                                                             cudaStream_t   stream) {
  if (!std::holds_alternative<DistGeom::BatchedAllInOneDeviceBuffersSingle>(systemDevice_)) {
    return cudaErrorInvalidValue;
  }
  const auto& buffers = std::get<DistGeom::BatchedAllInOneDeviceBuffersSingle>(systemDevice_);
  return DistGeom::launchBlockPerMolGradKernelAIO(
    numMolecules(),
    DistGeom::toAllInOneContribsDevicePtr(buffers, atomStartsDevice_.data()),
    positions,
    grad,
    weights_,
    torsionTermsEnabled_,
    activeSystemMask,
    stream);
}

}  // namespace nvMolKit
