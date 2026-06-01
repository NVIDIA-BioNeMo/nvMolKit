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

#include "src/conformer/device_coord_collector.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <unordered_map>

#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/p2p.h"

namespace nvMolKit {
namespace detail {

DeviceCoordResult finalizeOnTarget(std::vector<DeviceCoordCollector>& collectors,
                                   const int                          targetGpu,
                                   const int                          nMols) {
  // Pre-enable peer access from target to every contributing GPU once.
  for (const auto& collector : collectors) {
    if (collector.gpuId != targetGpu && !collector.atomCounts.empty()) {
      enablePeerAccess(targetGpu, collector.gpuId);
    }
  }

  int  totalConformers = 0;
  int  totalAtoms      = 0;
  bool hasEnergies     = false;
  bool hasConverged    = false;
  for (const auto& collector : collectors) {
    totalConformers += static_cast<int>(collector.atomCounts.size());
    for (const int natoms : collector.atomCounts) {
      totalAtoms += natoms;
    }
    if (collector.energies.size() > 0) {
      hasEnergies = true;
    }
    if (collector.converged.size() > 0) {
      hasConverged = true;
    }
  }

  const WithDevice  withTarget(targetGpu);
  ScopedStream      targetStream("DeviceCoord Finalize");
  DeviceCoordResult result;
  result.gpuId       = targetGpu;
  result.nMols       = nMols;
  result.positions   = AsyncDeviceVector<double>(static_cast<size_t>(totalAtoms) * 3, targetStream.stream());
  result.atomStarts  = AsyncDeviceVector<int32_t>(static_cast<size_t>(totalConformers + 1), targetStream.stream());
  result.molIndices  = AsyncDeviceVector<int32_t>(static_cast<size_t>(totalConformers), targetStream.stream());
  result.confIndices = AsyncDeviceVector<int32_t>(static_cast<size_t>(totalConformers), targetStream.stream());
  if (hasEnergies) {
    result.energies = AsyncDeviceVector<double>(static_cast<size_t>(totalConformers), targetStream.stream());
  }
  if (hasConverged) {
    result.converged = AsyncDeviceVector<int8_t>(static_cast<size_t>(totalConformers), targetStream.stream());
  }

  std::vector<int32_t> atomStartsHost(static_cast<size_t>(totalConformers + 1), 0);
  std::vector<int32_t> molIndicesHost(static_cast<size_t>(totalConformers), 0);
  std::vector<int32_t> confIndicesHost(static_cast<size_t>(totalConformers), 0);

  std::unordered_map<int, int> perMolCounter;
  int                          confCursor = 0;
  int                          atomCursor = 0;
  for (auto& collector : collectors) {
    const int numConfs = static_cast<int>(collector.atomCounts.size());
    if (numConfs == 0) {
      continue;
    }

    copyDeviceToDeviceAsync(result.positions.data() + static_cast<size_t>(atomCursor) * 3,
                            collector.positions.data(),
                            collector.positions.size() * sizeof(double),
                            collector.gpuId,
                            collector.stream,
                            targetGpu,
                            targetStream.stream());
    if (hasEnergies && collector.energies.size() > 0) {
      copyDeviceToDeviceAsync(result.energies.data() + confCursor,
                              collector.energies.data(),
                              collector.energies.size() * sizeof(double),
                              collector.gpuId,
                              collector.stream,
                              targetGpu,
                              targetStream.stream());
    }
    if (hasConverged && collector.converged.size() > 0) {
      copyDeviceToDeviceAsync(result.converged.data() + confCursor,
                              collector.converged.data(),
                              collector.converged.size() * sizeof(int8_t),
                              collector.gpuId,
                              collector.stream,
                              targetGpu,
                              targetStream.stream());
    }

    const bool useExplicitConfIds = !collector.confIds.empty();
    for (int conformerIdx = 0; conformerIdx < numConfs; ++conformerIdx) {
      atomStartsHost[static_cast<size_t>(confCursor)] = atomCursor;
      const int molId                                 = collector.molIds[conformerIdx];
      molIndicesHost[static_cast<size_t>(confCursor)] = molId;
      confIndicesHost[static_cast<size_t>(confCursor)] =
        useExplicitConfIds ? collector.confIds[conformerIdx] : perMolCounter[molId]++;
      atomCursor += collector.atomCounts[conformerIdx];
      ++confCursor;
    }
  }
  atomStartsHost[static_cast<size_t>(totalConformers)] = atomCursor;

  result.atomStarts.copyFromHost(atomStartsHost);
  if (totalConformers > 0) {
    result.molIndices.copyFromHost(molIndicesHost);
    result.confIndices.copyFromHost(confIndicesHost);
  }
  cudaCheckError(cudaStreamSynchronize(targetStream.stream()));

  // The local ScopedStream is about to be destroyed; rebind every result buffer to the default
  // stream so subsequent operations on the result do not dereference a freed cudaStream_t.
  result.positions.setStream(nullptr);
  result.atomStarts.setStream(nullptr);
  result.molIndices.setStream(nullptr);
  result.confIndices.setStream(nullptr);
  if (hasEnergies) {
    result.energies.setStream(nullptr);
  }
  if (hasConverged) {
    result.converged.setStream(nullptr);
  }
  return result;
}

}  // namespace detail
}  // namespace nvMolKit
