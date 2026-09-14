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

#include "src/forcefields/dist_geom_kernels.h"
#include "src/forcefields/etk_batched_forcefield.h"
#include "src/utils/device_convert.cuh"

namespace nvMolKit {

namespace {
void allocateEnergyScratch(const DistGeom::BatchedMolecularSystem3DHost& molSystemHost,
                           DistGeom::BatchedMolecular3DDeviceBuffers&    systemDevice) {
  systemDevice.energyBuffer.resize(molSystemHost.indices.energyBufferStarts.back());
  systemDevice.energyBuffer.zero();
}
}  // namespace

ETKBatchedForcefield::ETKBatchedForcefield(const DistGeom::BatchedMolecularSystem3DHost& molSystemHost,
                                           const std::vector<int>&                       atomStartsHost,
                                           const bool                                    useBasicKnowledge,
                                           BatchedForcefieldMetadata                     metadata,
                                           const cudaStream_t                            stream,
                                           const PrecisionMode                           precision)
    // ETK evaluates xyz terms, but the ETKDG minimization stage retains the
    // four-strided coordinate buffer until minimization is complete.
    : BatchedForcefield(ForceFieldType::ETK, 4, atomStartsHost, nullptr, std::move(metadata)),
      term_(useBasicKnowledge ? DistGeom::ETKTerm::ALL : DistGeom::ETKTerm::PLAIN) {
  singlePrecision_ = usesSinglePrecision(precision);
  atomStartsDevice_.setStream(stream);
  fullConversion_.setStream(stream);
  singleConversion_.setStream(stream);
  if (singlePrecision_) {
    auto& buffers = systemDevice_.emplace<DistGeom::BatchedMolecular3DDeviceBuffersSingle>();
    DistGeom::setStreams(buffers, stream);
    DistGeom::sendContribsAndIndicesToDevice3D(molSystemHost, buffers);
  } else {
    auto& buffers = systemDevice_.emplace<DistGeom::BatchedMolecular3DDeviceBuffers>();
    DistGeom::setStreams(buffers, stream);
    DistGeom::sendContribsAndIndicesToDevice3D(molSystemHost, buffers);
    allocateEnergyScratch(molSystemHost, buffers);
  }
  atomStartsDevice_.setFromVector(atomStartsHost);
  setAtomStartsDevice(atomStartsDevice_.data());
}

cudaError_t ETKBatchedForcefield::computeEnergy(double*        energyOuts,
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
  auto& buffers = std::get<DistGeom::BatchedMolecular3DDeviceBuffers>(systemDevice_);
  return DistGeom::computeEnergyETK(buffers,
                                    energyOuts,
                                    atomStartsDevice_.data(),
                                    positions,
                                    activeSystemMask,
                                    positions,
                                    term_,
                                    stream);
}

cudaError_t ETKBatchedForcefield::computeGradients(double*        grad,
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
  auto& buffers = std::get<DistGeom::BatchedMolecular3DDeviceBuffers>(systemDevice_);
  return DistGeom::computeGradientsETK(buffers,
                                       grad,
                                       atomStartsDevice_.data(),
                                       positions,
                                       activeSystemMask,
                                       term_,
                                       stream);
}

cudaError_t ETKBatchedForcefield::computePlanarEnergy(double*        energyOuts,
                                                      const double*  positions,
                                                      const uint8_t* activeSystemMask,
                                                      cudaStream_t   stream) {
  if (singlePrecision_) {
    singleConversion_.positions.resize(totalPositions());
    const auto err =
      detail::convertDeviceArray(singleConversion_.positions.data(), positions, totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    const auto& buffers = std::get<DistGeom::BatchedMolecular3DDeviceBuffersSingle>(systemDevice_);
    return DistGeom::launchPlanarEnergyKernelETK(
      numMolecules(),
      DistGeom::toEnergy3DForceContribsDevicePtr(buffers),
      DistGeom::toBatchedIndices3DDevicePtr(buffers, atomStartsDevice_.data()),
      singleConversion_.positions.data(),
      energyOuts,
      activeSystemMask,
      stream);
  }
  auto& buffers = std::get<DistGeom::BatchedMolecular3DDeviceBuffers>(systemDevice_);
  return DistGeom::computePlanarEnergy(buffers,
                                       energyOuts,
                                       atomStartsDevice_.data(),
                                       positions,
                                       activeSystemMask,
                                       positions,
                                       stream);
}

cudaError_t ETKBatchedForcefield::computeEnergy(double*        energyOuts,
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
  const auto& buffers = std::get<DistGeom::BatchedMolecular3DDeviceBuffersSingle>(systemDevice_);
  return DistGeom::launchBlockPerMolEnergyKernelETK(
    numMolecules(),
    DistGeom::toEnergy3DForceContribsDevicePtr(buffers),
    DistGeom::toBatchedIndices3DDevicePtr(buffers, atomStartsDevice_.data()),
    positions,
    energyOuts,
    activeSystemMask,
    stream);
}

cudaError_t ETKBatchedForcefield::computeGradients(float*         grad,
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
  const auto& buffers = std::get<DistGeom::BatchedMolecular3DDeviceBuffersSingle>(systemDevice_);
  return DistGeom::launchBlockPerMolGradKernelETK(
    numMolecules(),
    DistGeom::toEnergy3DForceContribsDevicePtr(buffers),
    DistGeom::toBatchedIndices3DDevicePtr(buffers, atomStartsDevice_.data()),
    positions,
    grad,
    activeSystemMask,
    stream);
}

}  // namespace nvMolKit
