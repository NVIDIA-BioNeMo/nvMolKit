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

#include "ff_device_collect.h"

#include <cuda_runtime.h>
#include <GraphMol/ROMol.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_error_check.h"
#include "device.h"
#include "p2p.h"

namespace nvMolKit {
namespace detail {

void appendBatch(const std::vector<ConformerInfo>& batchConformers,
                 const AsyncDeviceVector<double>&  positionsDevice,
                 const AsyncDeviceVector<double>&  energiesDevice,
                 const AsyncDeviceVector<int16_t>& statusesDevice,
                 DeviceCoordCollector&             collector) {
  const int numConformers = static_cast<int>(batchConformers.size());
  if (numConformers == 0) {
    return;
  }
  if (energiesDevice.size() != static_cast<size_t>(numConformers)) {
    throw std::invalid_argument("energiesDevice size does not match batch size");
  }
  if (statusesDevice.size() != static_cast<size_t>(numConformers)) {
    throw std::invalid_argument("statusesDevice size does not match batch size");
  }

  const WithDevice withDevice(collector.gpuId);

  // Make collector.stream wait on whatever stream produced positionsDevice / energiesDevice so the
  // device-to-device copies below see the minimizer's writes. When the producer stream and
  // collector.stream are the same the cross-stream sync is unnecessary; skip it. The `statusesDevice`
  // stream is synced fully on the host immediately below because we read its bytes on host.
  const cudaStream_t collectorStream = collector.stream;
  const cudaStream_t positionsStream = positionsDevice.stream();
  const cudaStream_t energiesStream  = energiesDevice.stream();
  const auto         waitOnStream    = [collectorStream](cudaStream_t producerStream) {
    if (producerStream == collectorStream) {
      return;
    }
    ScopedCudaEvent ready;
    cudaCheckError(cudaEventRecord(ready.event(), producerStream));
    cudaCheckError(cudaStreamWaitEvent(collectorStream, ready.event(), 0));
  };
  waitOnStream(positionsStream);
  if (energiesStream != positionsStream) {
    waitOnStream(energiesStream);
  }

  std::vector<int16_t> statusesHost(numConformers);
  statusesDevice.copyToHost(statusesHost.data(), static_cast<size_t>(numConformers));
  cudaCheckError(cudaStreamSynchronize(statusesDevice.stream()));

  int totalNewAtoms = 0;
  for (const auto& confInfo : batchConformers) {
    totalNewAtoms += static_cast<int>(confInfo.mol->getNumAtoms());
  }
  if (positionsDevice.size() != static_cast<size_t>(totalNewAtoms) * 3) {
    throw std::invalid_argument("positionsDevice size does not match batch atom count");
  }

  const size_t prevPositionsSize = collector.positions.size();
  const size_t prevConfCount     = collector.atomCounts.size();
  collector.positions.resize(prevPositionsSize + static_cast<size_t>(totalNewAtoms) * 3);
  collector.energies.resize(prevConfCount + static_cast<size_t>(numConformers));
  collector.converged.resize(prevConfCount + static_cast<size_t>(numConformers));

  cudaCheckError(cudaMemcpyAsync(collector.positions.data() + prevPositionsSize,
                                 positionsDevice.data(),
                                 static_cast<size_t>(totalNewAtoms) * 3 * sizeof(double),
                                 cudaMemcpyDeviceToDevice,
                                 collector.stream));
  cudaCheckError(cudaMemcpyAsync(collector.energies.data() + prevConfCount,
                                 energiesDevice.data(),
                                 static_cast<size_t>(numConformers) * sizeof(double),
                                 cudaMemcpyDeviceToDevice,
                                 collector.stream));

  std::vector<int8_t> convergedHost(numConformers);
  for (int i = 0; i < numConformers; ++i) {
    convergedHost[i] = static_cast<int8_t>(statusesHost[i] == 0);
  }
  cudaCheckError(cudaMemcpyAsync(collector.converged.data() + prevConfCount,
                                 convergedHost.data(),
                                 static_cast<size_t>(numConformers) * sizeof(int8_t),
                                 cudaMemcpyHostToDevice,
                                 collector.stream));

  collector.atomCounts.reserve(prevConfCount + static_cast<size_t>(numConformers));
  collector.molIds.reserve(prevConfCount + static_cast<size_t>(numConformers));
  collector.confIds.reserve(prevConfCount + static_cast<size_t>(numConformers));
  for (const auto& confInfo : batchConformers) {
    collector.atomCounts.push_back(static_cast<int>(confInfo.mol->getNumAtoms()));
    collector.molIds.push_back(static_cast<int>(confInfo.molIdx));
    collector.confIds.push_back(static_cast<int>(confInfo.confIdx));
  }
  // Sync so the pageable host-side `convergedHost` we just used in cudaMemcpyAsync is no longer
  // needed once we return.
  cudaCheckError(cudaStreamSynchronize(collector.stream));
}

namespace {

template <typename T> std::vector<T> downloadToHost(const AsyncDeviceVector<T>& vec) {
  std::vector<T> host(vec.size());
  if (!host.empty()) {
    vec.copyToHost(host);
    cudaCheckError(cudaStreamSynchronize(vec.stream()));
  }
  return host;
}

}  // namespace

DeviceInputIndex buildDeviceInputIndex(const DeviceCoordResult&          deviceInput,
                                       const std::vector<ConformerInfo>& allConformers) {
  DeviceInputIndex index;
  index.sourceGpu = deviceInput.gpuId;

  const WithDevice withSrc(deviceInput.gpuId);
  index.atomStartsHost                   = downloadToHost(deviceInput.atomStarts);
  const std::vector<int32_t> molIdxHost  = downloadToHost(deviceInput.molIndices);
  const std::vector<int32_t> confIdxHost = downloadToHost(deviceInput.confIndices);

  if (molIdxHost.size() != allConformers.size() || confIdxHost.size() != allConformers.size()) {
    throw std::invalid_argument("device_input conformer count (" + std::to_string(molIdxHost.size()) +
                                ") does not match host-flattened conformer count (" +
                                std::to_string(allConformers.size()) + ")");
  }
  if (index.atomStartsHost.size() != allConformers.size() + 1) {
    throw std::invalid_argument("device_input atom_starts has unexpected length");
  }

  // Build a quick lookup from (molIdx, confIdx) -> source conformer index, then validate that
  // the host-flattened list maps one-to-one onto the source list. A duplicate (molIdx, confIdx)
  // pair on the source side would silently let multiple destination slots resolve to the same
  // source position; reject it loudly here.
  std::unordered_map<long long, int> srcLookup;
  srcLookup.reserve(molIdxHost.size());
  for (size_t i = 0; i < molIdxHost.size(); ++i) {
    const long long key      = (static_cast<long long>(molIdxHost[i]) << 32) | static_cast<uint32_t>(confIdxHost[i]);
    const auto      inserted = srcLookup.insert({key, static_cast<int>(i)});
    if (!inserted.second) {
      throw std::invalid_argument("device_input contains duplicate (molIdx=" + std::to_string(molIdxHost[i]) +
                                  ", confIdx=" + std::to_string(confIdxHost[i]) + ") at source indices " +
                                  std::to_string(inserted.first->second) + " and " + std::to_string(i));
    }
  }
  index.conformerIndexBy.resize(allConformers.size());
  for (size_t i = 0; i < allConformers.size(); ++i) {
    const auto&     confInfo = allConformers[i];
    const long long key = (static_cast<long long>(confInfo.molIdx) << 32) | static_cast<uint32_t>(confInfo.confIdx);
    auto            it  = srcLookup.find(key);
    if (it == srcLookup.end()) {
      throw std::invalid_argument("device_input is missing an entry for (molIdx=" + std::to_string(confInfo.molIdx) +
                                  ", confIdx=" + std::to_string(confInfo.confIdx) + ")");
    }
    index.conformerIndexBy[i] = it->second;
  }

  for (size_t i = 0; i < allConformers.size(); ++i) {
    const int srcIdx        = index.conformerIndexBy[i];
    const int srcAtomCount  = index.atomStartsHost[srcIdx + 1] - index.atomStartsHost[srcIdx];
    const int hostAtomCount = static_cast<int>(allConformers[i].mol->getNumAtoms());
    if (srcAtomCount != hostAtomCount) {
      throw std::invalid_argument("device_input atom count mismatch for conformer " + std::to_string(i) + ": got " +
                                  std::to_string(srcAtomCount) + ", expected " + std::to_string(hostAtomCount));
    }
  }

  return index;
}

void broadcastDeviceInputBatch(const DeviceCoordResult&   deviceInput,
                               const DeviceInputIndex&    index,
                               const std::vector<int>&    batchSrcIndices,
                               const std::vector<int>&    batchAtomCounts,
                               const int                  executingGpu,
                               cudaStream_t               executingStream,
                               AsyncDeviceVector<double>& positionsDevice) {
  if (batchSrcIndices.size() != batchAtomCounts.size()) {
    throw std::invalid_argument("batchSrcIndices and batchAtomCounts must have the same size");
  }
  if (executingGpu != index.sourceGpu) {
    enablePeerAccess(executingGpu, index.sourceGpu);
  }

  size_t dstAtomOffset = 0;
  for (size_t i = 0; i < batchSrcIndices.size(); ++i) {
    const int    srcIdx       = batchSrcIndices[i];
    const int    natoms       = batchAtomCounts[i];
    const size_t srcAtomStart = static_cast<size_t>(index.atomStartsHost[srcIdx]);
    copyDeviceToDeviceAsync(positionsDevice.data() + dstAtomOffset * 3,
                            deviceInput.positions.data() + srcAtomStart * 3,
                            static_cast<size_t>(natoms) * 3 * sizeof(double),
                            index.sourceGpu,
                            deviceInput.positions.stream(),
                            executingGpu,
                            executingStream);
    dstAtomOffset += static_cast<size_t>(natoms);
  }
}

}  // namespace detail
}  // namespace nvMolKit
