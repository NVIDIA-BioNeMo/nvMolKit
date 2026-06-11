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

namespace nvMolKit {

namespace {
void allocateEnergyScratch(const MMFF::BatchedMolecularSystemHost& molSystemHost,
                           MMFF::BatchedMolecularDeviceBuffers&    systemDevice) {
  systemDevice.energyBuffer.resize(molSystemHost.indices.energyBufferStarts.back());
  systemDevice.energyBuffer.zero();
}
}  // namespace

MMFFBatchedForcefield::MMFFBatchedForcefield(const MMFF::BatchedMolecularSystemHost& molSystemHost,
                                             BatchedForcefieldMetadata               metadata,
                                             const cudaStream_t                      stream)
    : BatchedForcefield(ForceFieldType::MMFF, 3, molSystemHost.indices.atomStarts, nullptr, std::move(metadata)) {
  MMFF::setStreams(systemDevice_, stream);
  MMFF::sendContribsAndIndicesToDevice(molSystemHost, systemDevice_);
  allocateEnergyScratch(molSystemHost, systemDevice_);
  setAtomStartsDevice(systemDevice_.indices.atomStarts.data());
}

cudaError_t MMFFBatchedForcefield::computeEnergy(double*        energyOuts,
                                                 const double*  positions,
                                                 const uint8_t* activeSystemMask,
                                                 cudaStream_t   stream) {
  return MMFF::computeEnergy(systemDevice_, energyOuts, positions, activeSystemMask, stream);
}

cudaError_t MMFFBatchedForcefield::computeGradients(double*        grad,
                                                    const double*  positions,
                                                    const uint8_t* activeSystemMask,
                                                    cudaStream_t   stream) {
  return MMFF::computeGradients(systemDevice_, positions, grad, activeSystemMask, stream);
}

}  // namespace nvMolKit
