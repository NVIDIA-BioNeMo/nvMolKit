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

#include "src/forcefields/uff_batched_forcefield.h"
#include "src/forcefields/uff_kernels.h"
#include "src/utils/device_convert.cuh"

namespace nvMolKit {
namespace {
void allocateEnergyScratch(const UFF::BatchedMolecularSystemHost& molSystemHost,
                           UFF::BatchedMolecularDeviceBuffers&    systemDevice) {
  systemDevice.energyBuffer.resize(molSystemHost.indices.energyBufferStarts.back());
  systemDevice.energyBuffer.zero();
}

template <typename Buffers, typename storageT>
cudaError_t launchEnergy(Buffers&        buffers,
                         int             numMols,
                         const storageT* positions,
                         storageT*       energies,
                         const uint8_t*  activeSystemMask,
                         cudaStream_t    stream) {
  return UFF::launchBlockPerMolEnergyKernel(numMols,
                                            UFF::toEnergyForceContribsDevicePtr(buffers),
                                            UFF::toBatchedIndicesDevicePtr(buffers),
                                            positions,
                                            energies,
                                            UFF::batchHasConstraints(buffers.contribs),
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
  return UFF::launchBlockPerMolGradKernel(numMols,
                                          UFF::toEnergyForceContribsDevicePtr(buffers),
                                          UFF::toBatchedIndicesDevicePtr(buffers),
                                          positions,
                                          gradients,
                                          UFF::batchHasConstraints(buffers.contribs),
                                          stream,
                                          activeSystemMask);
}
}  // namespace

UFFBatchedForcefield::UFFBatchedForcefield(const UFF::BatchedMolecularSystemHost& molSystemHost,
                                           BatchedForcefieldMetadata              metadata,
                                           const cudaStream_t                     stream,
                                           const PrecisionMode                    precision)
    : BatchedForcefield(ForceFieldType::UFF, 3, molSystemHost.indices.atomStarts, nullptr, std::move(metadata)) {
  if (usesSinglePrecision(precision)) {
    auto& buffers = systemDevice_.emplace<UFF::BatchedMolecularDeviceBuffersSingle>();
    UFF::setStreams(buffers, stream);
    UFF::sendContribsAndIndicesToDevice(molSystemHost, buffers);
    setAtomStartsDevice(buffers.indices.atomStarts.data());
  } else {
    auto& buffers = systemDevice_.emplace<UFF::BatchedMolecularDeviceBuffers>();
    UFF::setStreams(buffers, stream);
    UFF::sendContribsAndIndicesToDevice(molSystemHost, buffers);
    allocateEnergyScratch(molSystemHost, buffers);
    setAtomStartsDevice(buffers.indices.atomStarts.data());
  }
}

cudaError_t UFFBatchedForcefield::computeEnergy(float*         energyOuts,
                                                const float*   positions,
                                                const uint8_t* activeSystemMask,
                                                cudaStream_t   stream) {
  if (!std::holds_alternative<UFF::BatchedMolecularDeviceBuffersSingle>(systemDevice_)) {
    return cudaErrorInvalidValue;
  }
  auto& buffers = std::get<UFF::BatchedMolecularDeviceBuffersSingle>(systemDevice_);
  return launchEnergy(buffers, numMolecules(), positions, energyOuts, activeSystemMask, stream);
}

cudaError_t UFFBatchedForcefield::computeGradients(float*         grad,
                                                   const float*   positions,
                                                   const uint8_t* activeSystemMask,
                                                   cudaStream_t   stream) {
  if (!std::holds_alternative<UFF::BatchedMolecularDeviceBuffersSingle>(systemDevice_)) {
    return cudaErrorInvalidValue;
  }
  auto& buffers = std::get<UFF::BatchedMolecularDeviceBuffersSingle>(systemDevice_);
  return launchGrad(buffers, numMolecules(), positions, grad, activeSystemMask, stream);
}

cudaError_t UFFBatchedForcefield::computeEnergy(double*        energyOuts,
                                                const double*  positions,
                                                const uint8_t* activeSystemMask,
                                                cudaStream_t   stream) {
  if (std::holds_alternative<UFF::BatchedMolecularDeviceBuffersSingle>(systemDevice_)) {
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
  auto& buffers = std::get<UFF::BatchedMolecularDeviceBuffers>(systemDevice_);
  return launchEnergy(buffers, numMolecules(), positions, energyOuts, activeSystemMask, stream);
}

cudaError_t UFFBatchedForcefield::computeGradients(double*        grad,
                                                   const double*  positions,
                                                   const uint8_t* activeSystemMask,
                                                   cudaStream_t   stream) {
  if (std::holds_alternative<UFF::BatchedMolecularDeviceBuffersSingle>(systemDevice_)) {
    singleConversion_.positions.setStream(stream);
    singleConversion_.gradients.setStream(stream);
    singleConversion_.positions.resize(totalPositions());
    singleConversion_.gradients.resize(totalPositions());
    auto err = detail::convertDeviceArray(singleConversion_.positions.data(), positions, totalPositions(), stream);
    if (err != cudaSuccess)
      return err;
    auto& buffers = std::get<UFF::BatchedMolecularDeviceBuffersSingle>(systemDevice_);
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
  auto& buffers = std::get<UFF::BatchedMolecularDeviceBuffers>(systemDevice_);
  return launchGrad(buffers, numMolecules(), positions, grad, activeSystemMask, stream);
}

}  // namespace nvMolKit
