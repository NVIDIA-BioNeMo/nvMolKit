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

#include "src/tfd/tfd_common.h"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit {

void TFDSystemDevice::setStream(cudaStream_t stream) {
  positions.setStream(stream);
  confPositionStarts.setStream(stream);
  torsionAtoms.setStream(stream);
  torsionWeights.setStream(stream);
  torsionMaxDevs.setStream(stream);
  quartetStarts.setStream(stream);
  torsionTypes.setStream(stream);
  molDescriptors.setStream(stream);
  dihedralWorkStarts.setStream(stream);
  tfdWorkStarts.setStream(stream);
  dihedralAngles.setStream(stream);
  tfdOutput.setStream(stream);
}

void transferToDevice(const TFDSystemHost& host, TFDSystemDevice& device, cudaStream_t stream) {
  device.setStream(stream);

  // Positions and conformer offsets
  device.positions.setFromVector(host.positions);
  device.confPositionStarts.setFromVector(host.confPositionStarts);

  // std::array<int,4> is layout-compatible with int[4]; copy directly
  device.torsionAtoms.setFromArray(reinterpret_cast<const int*>(host.torsionAtoms.data()),
                                   host.torsionAtoms.size() * 4);

  // Torsion metadata
  device.torsionWeights.setFromVector(host.torsionWeights);
  device.torsionMaxDevs.setFromVector(host.torsionMaxDevs);
  device.quartetStarts.setFromVector(host.quartetStarts);

  // TorsionType is enum class : uint8_t; copy directly
  device.torsionTypes.setFromArray(reinterpret_cast<const uint8_t*>(host.torsionTypes.data()),
                                   host.torsionTypes.size());

  // Per-molecule descriptors (compact: ~32 bytes per molecule)
  device.molDescriptors.setFromArray(host.molDescriptors.data(), host.molDescriptors.size());
  device.dihedralWorkStarts.setFromVector(host.dihedralWorkStarts);
  device.tfdWorkStarts.setFromVector(host.tfdWorkStarts);

  // Allocate and zero the active prefix of output buffers.
  int totalDihedrals = host.totalDihedrals();
  if (static_cast<int>(device.dihedralAngles.size()) < totalDihedrals) {
    device.dihedralAngles.resize(totalDihedrals);
  }
  cudaCheckError(cudaMemsetAsync(device.dihedralAngles.data(), 0, totalDihedrals * sizeof(float), stream));

  int totalTFDOutputs = host.totalTFDOutputs();
  if (static_cast<int>(device.tfdOutput.size()) < totalTFDOutputs) {
    device.tfdOutput.resize(totalTFDOutputs);
  }
  cudaCheckError(cudaMemsetAsync(device.tfdOutput.data(), 0, totalTFDOutputs * sizeof(float), stream));
}

}  // namespace nvMolKit
