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

#include "src/forcefields/mmff_batched_forcefield.h"
#include "src/forcefields/mmff_kernels.h"
#include "src/utils/device_convert.cuh"

namespace nvMolKit {

namespace {
void allocateEnergyScratch(const MMFF::BatchedMolecularSystemHost& molSystemHost,
                           MMFF::BatchedMolecularDeviceBuffers&    systemDevice) {
  systemDevice.energyBuffer.resize(molSystemHost.indices.energyBufferStarts.back());
  systemDevice.energyBuffer.zero();
}

template <typename Buffers, typename storageT>
cudaError_t launchEnergy(Buffers&        buffers,
                         int             numMols,
                         const storageT* positions,
                         double*         energies,
                         const uint8_t*  activeSystemMask,
                         cudaStream_t    stream) {
  return MMFF::launchBlockPerMolEnergyKernel(numMols,
                                             MMFF::toEnergyForceContribsDevicePtr(buffers),
                                             MMFF::toBatchedIndicesDevicePtr(buffers),
                                             positions,
                                             energies,
                                             MMFF::batchHasConstraints(buffers.contribs),
                                             stream,
                                             activeSystemMask);
}

template <typename Buffers, typename storageT>
cudaError_t launchGrad(Buffers&        buffers,
                       int             numMols,
                       const storageT* positions,
                       storageT*       gradients,
                       const uint8_t*  activeSystemMask,
                       cudaStream_t    stream) {
  return MMFF::launchBlockPerMolGradKernel(numMols,
                                           MMFF::toEnergyForceContribsDevicePtr(buffers),
                                           MMFF::toBatchedIndicesDevicePtr(buffers),
                                           positions,
                                           gradients,
                                           MMFF::batchHasConstraints(buffers.contribs),
                                           stream,
                                           activeSystemMask);
}
}  // namespace

MMFFBatchedForcefield::MMFFBatchedForcefield(const MMFF::BatchedMolecularSystemHost& molSystemHost,
                                             BatchedForcefieldMetadata               metadata,
                                             const cudaStream_t                      stream,
                                             const PrecisionMode                     precision)
    : BatchedForcefield(ForceFieldType::MMFF, 3, molSystemHost.indices.atomStarts, nullptr, std::move(metadata)) {
  singlePrecision_ = usesSinglePrecision(precision);
  fullConversion_.setStream(stream);
  singleConversion_.setStream(stream);
  if (singlePrecision_) {
    auto& buffers = systemDevice_.emplace<MMFF::BatchedMolecularDeviceBuffersF32>();
    MMFF::setStreams(buffers, stream);
    MMFF::sendContribsAndIndicesToDevice(molSystemHost, buffers);
    setAtomStartsDevice(buffers.indices.atomStarts.data());
  } else {
    auto& buffers = systemDevice_.emplace<MMFF::BatchedMolecularDeviceBuffers>();
    MMFF::setStreams(buffers, stream);
    MMFF::sendContribsAndIndicesToDevice(molSystemHost, buffers);
    allocateEnergyScratch(molSystemHost, buffers);
    setAtomStartsDevice(buffers.indices.atomStarts.data());
  }
}

cudaError_t MMFFBatchedForcefield::computeEnergy(double*        energyOuts,
                                                 const float*   positions,
                                                 const uint8_t* activeSystemMask,
                                                 cudaStream_t   stream) {
  if (!singlePrecision_) {
    fullConversion_.positions.resize(totalPositions());
    const auto err = detail::convertDeviceArray(fullConversion_.positions.data(), positions, totalPositions(), stream);
    return err == cudaSuccess ? computeEnergy(energyOuts, fullConversion_.positions.data(), activeSystemMask, stream) :
                                err;
  }
  auto& buffers = std::get<MMFF::BatchedMolecularDeviceBuffersF32>(systemDevice_);
  return launchEnergy(buffers, numMolecules(), positions, energyOuts, activeSystemMask, stream);
}

cudaError_t MMFFBatchedForcefield::computeGradients(float*         grad,
                                                    const float*   positions,
                                                    const uint8_t* activeSystemMask,
                                                    cudaStream_t   stream) {
  if (!singlePrecision_) {
    fullConversion_.positions.resize(totalPositions());
    fullConversion_.gradients.resize(totalPositions());
    auto err = detail::convertDeviceArray(fullConversion_.positions.data(), positions, totalPositions(), stream);
    if (err == cudaSuccess)
      err =
        computeGradients(fullConversion_.gradients.data(), fullConversion_.positions.data(), activeSystemMask, stream);
    return err == cudaSuccess ?
             detail::convertDeviceArray(grad, fullConversion_.gradients.data(), totalPositions(), stream) :
             err;
  }
  auto& buffers = std::get<MMFF::BatchedMolecularDeviceBuffersF32>(systemDevice_);
  return launchGrad(buffers, numMolecules(), positions, grad, activeSystemMask, stream);
}

cudaError_t MMFFBatchedForcefield::computeEnergy(double*        energyOuts,
                                                 const double*  positions,
                                                 const uint8_t* activeSystemMask,
                                                 cudaStream_t   stream) {
  if (singlePrecision_) {
    singleConversion_.positions.resize(totalPositions());
    auto err = detail::convertDeviceArray(singleConversion_.positions.data(), positions, totalPositions(), stream);
    return err == cudaSuccess ?
             computeEnergy(energyOuts, singleConversion_.positions.data(), activeSystemMask, stream) :
             err;
  }
  auto& buffers = std::get<MMFF::BatchedMolecularDeviceBuffers>(systemDevice_);
  return launchEnergy(buffers, numMolecules(), positions, energyOuts, activeSystemMask, stream);
}

cudaError_t MMFFBatchedForcefield::computeGradients(double*        grad,
                                                    const double*  positions,
                                                    const uint8_t* activeSystemMask,
                                                    cudaStream_t   stream) {
  if (singlePrecision_) {
    singleConversion_.positions.resize(totalPositions());
    singleConversion_.gradients.resize(totalPositions());
    auto err = detail::convertDeviceArray(singleConversion_.positions.data(), positions, totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    auto& buffers = std::get<MMFF::BatchedMolecularDeviceBuffersF32>(systemDevice_);
    err           = launchGrad(buffers,
                     numMolecules(),
                     singleConversion_.positions.data(),
                     singleConversion_.gradients.data(),
                     activeSystemMask,
                     stream);
    if (err == cudaSuccess)
      err = detail::convertDeviceArray(grad, singleConversion_.gradients.data(), totalPositions(), stream);
    return err;
  }
  auto& buffers = std::get<MMFF::BatchedMolecularDeviceBuffers>(systemDevice_);
  return launchGrad(buffers, numMolecules(), positions, grad, activeSystemMask, stream);
}

}  // namespace nvMolKit
