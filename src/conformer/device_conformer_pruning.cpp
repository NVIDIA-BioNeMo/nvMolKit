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

#include "src/conformer/device_conformer_pruning.h"

#include <cuda_runtime.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rdkit_extensions/conformer_pruning.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"

namespace nvMolKit {
namespace detail {

constexpr uint32_t kKeptState = 2;

DeviceCoordResult pruneDeviceConformers(DeviceCoordResult                           result,
                                        const std::vector<RDKit::ROMol*>&           mols,
                                        const RDKit::DGeomHelpers::EmbedParameters& params) {
  const size_t numConformers = result.molIndices.size();
  if (params.pruneRmsThresh <= 0.0 || numConformers < 2) {
    return result;
  }

  const WithDevice     withDevice(result.gpuId);
  std::vector<int32_t> atomStarts(result.atomStarts.size());
  std::vector<int32_t> molIndices(numConformers);
  result.atomStarts.copyToHost(atomStarts);
  result.molIndices.copyToHost(molIndices);
  cudaCheckError(cudaStreamSynchronize(nullptr));

  // Group conformer ids by molecule because collectors may return them interleaved.
  std::vector<ConformerPruningMolInfo> molInfo(mols.size());
  for (const int32_t molIdx : molIndices) {
    ++molInfo[static_cast<size_t>(molIdx)].confCount;
  }

  int64_t totalPairs = 0;
  for (const auto& info : molInfo) {
    totalPairs += static_cast<int64_t>(info.confCount) * (info.confCount - 1) / 2;
  }
  if (totalPairs == 0) {
    return result;
  }
  if (totalPairs > std::numeric_limits<int>::max()) {
    throw std::overflow_error("Too many conformer pairs to prune in one CUDA launch");
  }
  const int numPairs = static_cast<int>(totalPairs);

  int confBegin = 0;
  int pairBegin = 0;
  for (auto& info : molInfo) {
    info.confBegin = confBegin;
    info.pairBegin = pairBegin;
    confBegin += info.confCount;
    pairBegin += static_cast<int>(static_cast<int64_t>(info.confCount) * (info.confCount - 1) / 2);
  }

  std::vector<int32_t> groupedConfIds(numConformers);
  std::vector<int>     nextConf;
  nextConf.reserve(molInfo.size());
  for (const auto& info : molInfo) {
    nextConf.push_back(info.confBegin);
  }
  for (size_t confIdx = 0; confIdx < numConformers; ++confIdx) {
    const int molIdx                   = molIndices[confIdx];
    groupedConfIds[nextConf[molIdx]++] = static_cast<int32_t>(confIdx);
  }

  // RDKit supplies the atom orders needed for heavy-atom and symmetry-aware RMSD.
  std::vector<int32_t> atomMaps;
  for (size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    auto& info = molInfo[molIdx];
    if (info.confCount < 2) {
      continue;
    }
    const auto maps      = RDKit::DGeomHelpers::getMolSelfMatches(*mols[molIdx], params);
    info.atomMapBegin    = static_cast<int>(atomMaps.size());
    info.atomMapCount    = static_cast<int>(maps.size());
    info.mappedAtomCount = static_cast<int>(maps.front().size());
    for (const auto& map : maps) {
      atomMaps.insert(atomMaps.end(), map.begin(), map.end());
    }
  }

  AsyncDeviceVector<ConformerPruningMolInfo> molInfoDevice(molInfo.size());
  AsyncDeviceVector<int32_t>                 groupedConfIdsDevice(groupedConfIds.size());
  AsyncDeviceVector<int32_t>                 atomMapsDevice(atomMaps.size());
  AsyncDeviceVector<uint8_t>                 conflicts(static_cast<size_t>(numPairs));
  AsyncDeviceVector<uint32_t>                states(numConformers);
  molInfoDevice.copyFromHost(molInfo);
  groupedConfIdsDevice.copyFromHost(groupedConfIds);
  atomMapsDevice.copyFromHost(atomMaps);
  conflicts.zero();
  states.zero();

  conformerPruneMaskGpu(cuda::std::span<const double>(result.positions.data(), result.positions.size()),
                        cuda::std::span<const int32_t>(result.atomStarts.data(), result.atomStarts.size()),
                        cuda::std::span<const int32_t>(groupedConfIdsDevice.data(), groupedConfIdsDevice.size()),
                        cuda::std::span<const ConformerPruningMolInfo>(molInfoDevice.data(), molInfoDevice.size()),
                        cuda::std::span<const int32_t>(atomMapsDevice.data(), atomMapsDevice.size()),
                        cuda::std::span<uint8_t>(conflicts.data(), conflicts.size()),
                        cuda::std::span<uint32_t>(states.data(), states.size()),
                        numPairs,
                        params.pruneRmsThresh,
                        nullptr);

  std::vector<uint32_t> selected(states.size());
  states.copyToHost(selected);
  cudaCheckError(cudaStreamSynchronize(nullptr));

  std::vector<uint8_t> keep(numConformers, 0);
  for (size_t groupedIdx = 0; groupedIdx < groupedConfIds.size(); ++groupedIdx) {
    keep[static_cast<size_t>(groupedConfIds[groupedIdx])] = selected[groupedIdx] == kKeptState;
  }

  size_t keptConformers = 0;
  size_t keptAtoms      = 0;
  for (size_t confIdx = 0; confIdx < numConformers; ++confIdx) {
    if (keep[confIdx] != 0) {
      ++keptConformers;
      keptAtoms += static_cast<size_t>(atomStarts[confIdx + 1] - atomStarts[confIdx]);
    }
  }

  DeviceCoordResult compacted;
  compacted.gpuId       = result.gpuId;
  compacted.nMols       = result.nMols;
  compacted.positions   = AsyncDeviceVector<double>(keptAtoms * 3);
  compacted.atomStarts  = AsyncDeviceVector<int32_t>(keptConformers + 1);
  compacted.molIndices  = AsyncDeviceVector<int32_t>(keptConformers);
  compacted.confIndices = AsyncDeviceVector<int32_t>(keptConformers);

  std::vector<int32_t> compactedAtomStarts(keptConformers + 1, 0);
  std::vector<int32_t> compactedMolIndices;
  std::vector<int32_t> compactedConfIndices;
  std::vector<int32_t> nextConfIndex(mols.size(), 0);
  compactedMolIndices.reserve(keptConformers);
  compactedConfIndices.reserve(keptConformers);

  size_t dstAtom = 0;
  for (size_t srcConf = 0; srcConf < numConformers; ++srcConf) {
    if (keep[srcConf] == 0) {
      continue;
    }
    const size_t atomCount = static_cast<size_t>(atomStarts[srcConf + 1] - atomStarts[srcConf]);
    cudaCheckError(cudaMemcpyAsync(compacted.positions.data() + dstAtom * 3,
                                   result.positions.data() + static_cast<size_t>(atomStarts[srcConf]) * 3,
                                   atomCount * 3 * sizeof(double),
                                   cudaMemcpyDeviceToDevice,
                                   nullptr));

    dstAtom += atomCount;
    compactedAtomStarts[compactedMolIndices.size() + 1] = static_cast<int32_t>(dstAtom);
    const int32_t molIdx                                = molIndices[srcConf];
    compactedMolIndices.push_back(molIdx);
    compactedConfIndices.push_back(nextConfIndex[static_cast<size_t>(molIdx)]++);
  }

  compacted.atomStarts.copyFromHost(compactedAtomStarts);
  compacted.molIndices.copyFromHost(compactedMolIndices);
  compacted.confIndices.copyFromHost(compactedConfIndices);
  cudaCheckError(cudaStreamSynchronize(nullptr));
  return compacted;
}

}  // namespace detail
}  // namespace nvMolKit
