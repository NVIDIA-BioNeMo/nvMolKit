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
#include <utility>
#include <vector>

#include "rdkit_extensions/conformer_pruning.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"

namespace nvMolKit {
namespace detail {

DeviceCoordResult pruneDeviceConformers(DeviceCoordResult                           result,
                                        const std::vector<RDKit::ROMol*>&           mols,
                                        const RDKit::DGeomHelpers::EmbedParameters& params) {
  const size_t numConformers = result.molIndices.size();
  if (params.pruneRmsThresh <= 0.0 || numConformers < 2) {
    return result;
  }

  // Only the small index arrays come back to the CPU. Coordinates stay on the GPU.
  const WithDevice     withDevice(result.gpuId);
  std::vector<int32_t> atomStarts(result.atomStarts.size());
  std::vector<int32_t> molIndices(numConformers);
  result.atomStarts.copyToHost(atomStarts);
  result.molIndices.copyToHost(molIndices);
  cudaCheckError(cudaStreamSynchronize(nullptr));

  std::vector<ConformerPruningMolInfo> molInfos(mols.size());
  for (const int32_t molIdx : molIndices) {
    ++molInfos[static_cast<size_t>(molIdx)].confCount;
  }

  int confBegin = 0;

  // Record where each molecule's group begins.
  for (auto& info : molInfos) {
    info.confBegin = confBegin;
    confBegin += info.confCount;
  }

  // Conformers from different molecules may be interleaved, so group their IDs
  // while preserving each molecule's input order.
  std::vector<int32_t> groupedConfIds(numConformers);
  std::vector<int>     nextConf(molInfos.size());
  for (size_t molIdx = 0; molIdx < molInfos.size(); ++molIdx) {
    nextConf[molIdx] = molInfos[molIdx].confBegin;
  }
  for (size_t confIdx = 0; confIdx < numConformers; ++confIdx) {
    const int molIdx                   = molIndices[confIdx];
    groupedConfIds[nextConf[molIdx]++] = static_cast<int32_t>(confIdx);
  }

  // Store RDKit's heavy-atom and symmetry mappings back-to-back so each
  // molecule can find its mappings from molInfos.
  std::vector<int32_t> atomMaps;
  for (size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    auto& info = molInfos[molIdx];
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

  AsyncDeviceVector<ConformerPruningMolInfo> molInfosDevice(molInfos.size());
  AsyncDeviceVector<int32_t>                 groupedConfIdsDevice(groupedConfIds.size());
  AsyncDeviceVector<int32_t>                 atomMapsDevice(atomMaps.size());
  AsyncDeviceVector<uint8_t>                 selected(numConformers);
  molInfosDevice.copyFromHost(molInfos);
  groupedConfIdsDevice.copyFromHost(groupedConfIds);
  atomMapsDevice.copyFromHost(atomMaps);

  conformerPruneMaskGpu(cuda::std::span<const double>(result.positions.data(), result.positions.size()),
                        cuda::std::span<const int32_t>(result.atomStarts.data(), result.atomStarts.size()),
                        cuda::std::span<int32_t>(groupedConfIdsDevice.data(), groupedConfIdsDevice.size()),
                        cuda::std::span<const ConformerPruningMolInfo>(molInfosDevice.data(), molInfosDevice.size()),
                        cuda::std::span<const int32_t>(atomMapsDevice.data(), atomMapsDevice.size()),
                        cuda::std::span<uint8_t>(selected.data(), selected.size()),
                        params.pruneRmsThresh,
                        nullptr);

  // Bring back the keep decisions so we can size and build the compacted result.
  // The GPU groups them by molecule, so restore their input order first.
  std::vector<uint8_t> selectedHost(selected.size());
  selected.copyToHost(selectedHost);
  cudaCheckError(cudaStreamSynchronize(nullptr));

  std::vector<uint8_t> keep(numConformers, 0);
  for (size_t groupedIdx = 0; groupedIdx < groupedConfIds.size(); ++groupedIdx) {
    keep[static_cast<size_t>(groupedConfIds[groupedIdx])] = selectedHost[groupedIdx];
  }

  // Count how much space is needed for the conformers that survived pruning.
  size_t keptConformers = 0;
  size_t keptAtoms      = 0;
  for (size_t confIdx = 0; confIdx < numConformers; ++confIdx) {
    if (keep[confIdx] != 0) {
      ++keptConformers;
      keptAtoms += static_cast<size_t>(atomStarts[confIdx + 1] - atomStarts[confIdx]);
    }
  }
  if (keptConformers == numConformers) {
    // If every conformer survives, keep the coordinates and only renumber the
    // conformers within each molecule because the input may be interleaved.
    std::vector<int32_t> renumberedConfIndices(numConformers);
    std::vector<int32_t> nextConfIndex(mols.size(), 0);
    for (size_t confIdx = 0; confIdx < numConformers; ++confIdx) {
      renumberedConfIndices[confIdx] = nextConfIndex[static_cast<size_t>(molIndices[confIdx])]++;
    }
    result.confIndices.copyFromHost(renumberedConfIndices);
    cudaCheckError(cudaStreamSynchronize(nullptr));
    return result;
  }

  DeviceCoordResult compacted;
  compacted.gpuId       = result.gpuId;
  compacted.nMols       = result.nMols;
  compacted.positions   = AsyncDeviceVector<double>(keptAtoms * kNumCoordinateDimensions);
  compacted.atomStarts  = AsyncDeviceVector<int32_t>(keptConformers + 1);
  compacted.molIndices  = AsyncDeviceVector<int32_t>(keptConformers);
  compacted.confIndices = AsyncDeviceVector<int32_t>(keptConformers);

  std::vector<int32_t> compactedAtomStarts(keptConformers + 1, 0);
  std::vector<int32_t> compactedMolIndices(keptConformers);
  std::vector<int32_t> compactedConfIndices(keptConformers);
  std::vector<int32_t> sourceConformerIds(keptConformers);
  std::vector<int32_t> nextConfIndex(mols.size(), 0);

  // Build the compact result metadata on the CPU.
  size_t dstAtom = 0;
  size_t dstConf = 0;
  for (size_t srcConf = 0; srcConf < numConformers; ++srcConf) {
    if (keep[srcConf] == 0) {
      continue;
    }
    const size_t atomCount = static_cast<size_t>(atomStarts[srcConf + 1] - atomStarts[srcConf]);
    dstAtom += atomCount;
    compactedAtomStarts[dstConf + 1] = static_cast<int32_t>(dstAtom);
    const int32_t molIdx             = molIndices[srcConf];
    compactedMolIndices[dstConf]     = molIdx;
    compactedConfIndices[dstConf]    = nextConfIndex[static_cast<size_t>(molIdx)]++;
    sourceConformerIds[dstConf]      = static_cast<int32_t>(srcConf);
    ++dstConf;
  }

  AsyncDeviceVector<int32_t> sourceConformerIdsDevice(sourceConformerIds.size());
  sourceConformerIdsDevice.copyFromHost(sourceConformerIds);
  compacted.atomStarts.copyFromHost(compactedAtomStarts);
  compacted.molIndices.copyFromHost(compactedMolIndices);
  compacted.confIndices.copyFromHost(compactedConfIndices);

  // Copy all retained coordinates in one GPU launch instead of issuing a copy
  // for every conformer.
  compactConformerPositionsGpu(
    cuda::std::span<const double>(result.positions.data(), result.positions.size()),
    cuda::std::span<const int32_t>(result.atomStarts.data(), result.atomStarts.size()),
    cuda::std::span<const int32_t>(sourceConformerIdsDevice.data(), sourceConformerIdsDevice.size()),
    cuda::std::span<const int32_t>(compacted.atomStarts.data(), compacted.atomStarts.size()),
    cuda::std::span<double>(compacted.positions.data(), compacted.positions.size()),
    nullptr);
  cudaCheckError(cudaStreamSynchronize(nullptr));
  return compacted;
}

}  // namespace detail
}  // namespace nvMolKit
