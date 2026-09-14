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

#include "src/forcefields/dg_batched_forcefield.h"
#include "src/forcefields/dist_geom_kernels.h"
#include "src/utils/device_convert.cuh"

namespace nvMolKit {

namespace {
void allocateEnergyScratch(const DistGeom::BatchedMolecularSystemHost& molSystemHost,
                           DistGeom::BatchedMolecularDeviceBuffers&    systemDevice) {
  systemDevice.energyBuffer.resize(molSystemHost.indices.energyBufferStarts.back());
  systemDevice.energyBuffer.zero();
  systemDevice.dimension = molSystemHost.dimension;
}
}  // namespace

DGBatchedForcefield::DGBatchedForcefield(const DistGeom::BatchedMolecularSystemHost& molSystemHost,
                                         const std::vector<int>&                     atomStartsHost,
                                         const double                                chiralWeight,
                                         const double                                fourthDimWeight,
                                         BatchedForcefieldMetadata                   metadata,
                                         const cudaStream_t                          stream,
                                         const PrecisionMode                         precision)
    : BatchedForcefield(ForceFieldType::DG, molSystemHost.dimension, atomStartsHost, nullptr, std::move(metadata)),
      chiralWeight_(chiralWeight),
      fourthDimWeight_(fourthDimWeight) {
  singlePrecision_ = usesSinglePrecision(precision);
  atomStartsDevice_.setStream(stream);
  fullConversion_.setStream(stream);
  singleConversion_.setStream(stream);
  if (singlePrecision_) {
    auto& buffers = systemDevice_.emplace<DistGeom::BatchedMolecularDeviceBuffersSingle>();
    DistGeom::setStreams(buffers, stream);
    DistGeom::sendContribsAndIndicesToDevice(molSystemHost, buffers);
    buffers.dimension = molSystemHost.dimension;
  } else {
    auto& buffers = systemDevice_.emplace<DistGeom::BatchedMolecularDeviceBuffers>();
    DistGeom::setStreams(buffers, stream);
    DistGeom::sendContribsAndIndicesToDevice(molSystemHost, buffers);
    allocateEnergyScratch(molSystemHost, buffers);
  }
  atomStartsDevice_.setFromVector(atomStartsHost);
  setAtomStartsDevice(atomStartsDevice_.data());
}

cudaError_t DGBatchedForcefield::computeEnergy(double*        energyOuts,
                                               const double*  positions,
                                               const uint8_t* activeSystemMask,
                                               cudaStream_t   stream) {
  if (singlePrecision_) {
    singleConversion_.positions.resize(totalPositions());
    const auto err =
      detail::convertDeviceArray(singleConversion_.positions.data(), positions, totalPositions(), stream);
    return err == cudaSuccess ?
             computeEnergy(energyOuts, singleConversion_.positions.data(), activeSystemMask, stream) :
             err;
  }
  auto& buffers = std::get<DistGeom::BatchedMolecularDeviceBuffers>(systemDevice_);
  return DistGeom::computeEnergy(buffers,
                                 energyOuts,
                                 atomStartsDevice_.data(),
                                 positions,
                                 chiralWeight_,
                                 fourthDimWeight_,
                                 activeSystemMask,
                                 positions,
                                 stream);
}

cudaError_t DGBatchedForcefield::computeGradients(double*        grad,
                                                  const double*  positions,
                                                  const uint8_t* activeSystemMask,
                                                  cudaStream_t   stream) {
  if (singlePrecision_) {
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
  auto& buffers = std::get<DistGeom::BatchedMolecularDeviceBuffers>(systemDevice_);
  return DistGeom::computeGradients(buffers,
                                    grad,
                                    atomStartsDevice_.data(),
                                    positions,
                                    chiralWeight_,
                                    fourthDimWeight_,
                                    activeSystemMask,
                                    stream);
}

cudaError_t DGBatchedForcefield::computeEnergy(double*        energyOuts,
                                               const float*   positions,
                                               const uint8_t* activeSystemMask,
                                               cudaStream_t   stream) {
  if (!singlePrecision_) {
    fullConversion_.positions.resize(totalPositions());
    const auto err = detail::convertDeviceArray(fullConversion_.positions.data(), positions, totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    return computeEnergy(energyOuts, fullConversion_.positions.data(), activeSystemMask, stream);
  }
  const auto& buffers = std::get<DistGeom::BatchedMolecularDeviceBuffersSingle>(systemDevice_);
  return DistGeom::launchBlockPerMolEnergyKernel(numMolecules(),
                                                 DistGeom::toEnergyForceContribsDevicePtr(buffers),
                                                 DistGeom::toBatchedIndicesDevicePtr(buffers, atomStartsDevice_.data()),
                                                 positions,
                                                 energyOuts,
                                                 dataDim(),
                                                 static_cast<float>(chiralWeight_),
                                                 static_cast<float>(fourthDimWeight_),
                                                 activeSystemMask,
                                                 stream);
}

cudaError_t DGBatchedForcefield::computeGradients(float*         grad,
                                                  const float*   positions,
                                                  const uint8_t* activeSystemMask,
                                                  cudaStream_t   stream) {
  if (!singlePrecision_) {
    fullConversion_.positions.resize(totalPositions());
    fullConversion_.gradients.resize(totalPositions());
    auto err = detail::convertDeviceArray(fullConversion_.positions.data(), positions, totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    fullConversion_.gradients.zero();
    err =
      computeGradients(fullConversion_.gradients.data(), fullConversion_.positions.data(), activeSystemMask, stream);
    return err == cudaSuccess ?
             detail::convertDeviceArray(grad, fullConversion_.gradients.data(), totalPositions(), stream) :
             err;
  }
  const auto& buffers = std::get<DistGeom::BatchedMolecularDeviceBuffersSingle>(systemDevice_);
  return DistGeom::launchBlockPerMolGradKernel(numMolecules(),
                                               DistGeom::toEnergyForceContribsDevicePtr(buffers),
                                               DistGeom::toBatchedIndicesDevicePtr(buffers, atomStartsDevice_.data()),
                                               positions,
                                               grad,
                                               dataDim(),
                                               static_cast<float>(chiralWeight_),
                                               static_cast<float>(fourthDimWeight_),
                                               activeSystemMask,
                                               stream);
}

}  // namespace nvMolKit
