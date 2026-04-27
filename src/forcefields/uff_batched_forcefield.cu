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

namespace nvMolKit {
namespace {
void allocateEnergyScratch(const UFF::BatchedMolecularSystemHost& molSystemHost,
                           UFF::BatchedMolecularDeviceBuffers&    systemDevice) {
  systemDevice.energyBuffer.resize(molSystemHost.indices.energyBufferStarts.back());
  systemDevice.energyBuffer.zero();
}
}  // namespace

UFFBatchedForcefield::UFFBatchedForcefield(const UFF::BatchedMolecularSystemHost& molSystemHost,
                                           BatchedForcefieldMetadata              metadata,
                                           const cudaStream_t                     stream)
    : BatchedForcefield(ForceFieldType::UFF, 3, molSystemHost.indices.atomStarts, nullptr, std::move(metadata)) {
  UFF::setStreams(systemDevice_, stream);
  UFF::sendContribsAndIndicesToDevice(molSystemHost, systemDevice_);
  allocateEnergyScratch(molSystemHost, systemDevice_);
  setAtomStartsDevice(systemDevice_.indices.atomStarts.data());
}

cudaError_t UFFBatchedForcefield::computeEnergy(double*        energyOuts,
                                                const double*  positions,
                                                const uint8_t* activeSystemMask,
                                                cudaStream_t   stream) {
  return UFF::computeEnergy(systemDevice_, energyOuts, positions, activeSystemMask, stream);
}

cudaError_t UFFBatchedForcefield::computeGradients(double*        grad,
                                                   const double*  positions,
                                                   const uint8_t* activeSystemMask,
                                                   cudaStream_t   stream) {
  return UFF::computeGradients(systemDevice_, positions, grad, activeSystemMask, stream);
}

}  // namespace nvMolKit
