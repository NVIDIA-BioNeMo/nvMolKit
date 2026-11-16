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

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "src/conformer/device_coord_gather.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/p2p.h"

namespace nvMolKit {
namespace detail {

namespace {

//! A conformer kept in the result: its result key (molecule, conformer index) and where it lives in its collector.
struct KeptConformer {
  int molId;
  int confIdx;
  int collectorIdx;
  int srcConfIdx;
  int srcAtomStart;
  int numAtoms;
};

}  // namespace

DeviceCoordResult finalizeOnTarget(std::vector<DeviceCoordCollector>& collectors,
                                   const int                          targetGpu,
                                   const int                          nMols,
                                   const int                          maxConformersPerMol) {
  // Pre-enable peer access from target to every contributing GPU once.
  for (const auto& collector : collectors) {
    if (collector.gpuId != targetGpu && !collector.atomCounts.empty()) {
      enablePeerAccess(targetGpu, collector.gpuId);
    }
  }

  // Collectors carrying attempt IDs may hold more conformers than a molecule needs (parallel attempts that all
  // succeeded, or entries displaced by a lower ID). Keep the lowest IDs per molecule and rank them, so the result
  // does not depend on which collector got which attempt. rank < 0 marks a dropped conformer.
  struct AttemptRef {
    int molId;
    int attemptId;
    int collectorIdx;
    int confIdx;
  };
  const bool filterByAttempt = std::any_of(collectors.begin(), collectors.end(), [](const auto& collector) {
    return !collector.attemptIds.empty();
  });
  std::vector<std::vector<int>> ranks(collectors.size());
  if (filterByAttempt) {
    std::vector<AttemptRef> refs;
    for (size_t c = 0; c < collectors.size(); ++c) {
      const auto& collector = collectors[c];
      if (collector.attemptIds.size() != collector.atomCounts.size()) {
        throw std::invalid_argument("attemptIds must be populated for every conformer or none");
      }
      ranks[c].assign(collector.atomCounts.size(), -1);
      for (size_t j = 0; j < collector.atomCounts.size(); ++j) {
        refs.push_back({collector.molIds[j], collector.attemptIds[j], static_cast<int>(c), static_cast<int>(j)});
      }
    }
    // Bucket by molecule in O(n), then order each (small) bucket by attempt ID.
    std::vector<size_t> bucketStart(static_cast<size_t>(nMols) + 1, 0);
    for (const auto& ref : refs) {
      ++bucketStart[static_cast<size_t>(ref.molId) + 1];
    }
    std::partial_sum(bucketStart.begin(), bucketStart.end(), bucketStart.begin());
    std::vector<AttemptRef> bucketed(refs.size());
    std::vector<size_t>     fill(bucketStart.begin(), bucketStart.end() - 1);
    for (const auto& ref : refs) {
      bucketed[fill[static_cast<size_t>(ref.molId)]++] = ref;
    }
    for (size_t m = 0; m < static_cast<size_t>(nMols); ++m) {
      std::sort(bucketed.begin() + bucketStart[m],
                bucketed.begin() + bucketStart[m + 1],
                [](const AttemptRef& a, const AttemptRef& b) { return a.attemptId < b.attemptId; });
      for (size_t r = bucketStart[m]; r < bucketStart[m + 1]; ++r) {
        const int rank = static_cast<int>(r - bucketStart[m]);
        if (maxConformersPerMol <= 0 || rank < maxConformersPerMol) {
          ranks[bucketed[r].collectorIdx][bucketed[r].confIdx] = rank;
        }
      }
    }
  }
  const auto isKept = [&](const size_t collectorIdx, const int confIdx) {
    return !filterByAttempt || ranks[collectorIdx][confIdx] >= 0;
  };

  int  totalConformers = 0;
  int  totalAtoms      = 0;
  bool hasEnergies     = false;
  bool hasConverged    = false;
  for (size_t c = 0; c < collectors.size(); ++c) {
    const auto& collector = collectors[c];
    for (size_t j = 0; j < collector.atomCounts.size(); ++j) {
      if (isKept(c, static_cast<int>(j))) {
        ++totalConformers;
        totalAtoms += collector.atomCounts[j];
      }
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

  // Batches land in collectors in completion order. Order the kept conformers by (molecule, conformer index) so the
  // result is in input order.
  std::vector<KeptConformer> kept;
  kept.reserve(static_cast<size_t>(totalConformers));
  std::unordered_map<int, int> perMolCounter;
  size_t                       nonEmptyCollectors = 0;
  for (size_t c = 0; c < collectors.size(); ++c) {
    const auto& collector = collectors[c];
    if (!collector.atomCounts.empty()) {
      ++nonEmptyCollectors;
    }
    const bool useExplicitConfIds = !collector.confIds.empty();
    int        srcAtomStart       = 0;
    for (int srcConfIdx = 0; srcConfIdx < static_cast<int>(collector.atomCounts.size()); ++srcConfIdx) {
      const int numAtoms = collector.atomCounts[srcConfIdx];
      if (isKept(c, srcConfIdx)) {
        const int molId   = collector.molIds[srcConfIdx];
        const int confIdx = useExplicitConfIds ? collector.confIds[srcConfIdx] :
                            filterByAttempt    ? ranks[c][srcConfIdx] :
                                                 perMolCounter[molId]++;
        kept.push_back({molId, confIdx, static_cast<int>(c), srcConfIdx, srcAtomStart, numAtoms});
      }
      srcAtomStart += numAtoms;
    }
  }
  std::stable_sort(kept.begin(), kept.end(), [](const KeptConformer& a, const KeptConformer& b) {
    return a.molId != b.molId ? a.molId < b.molId : a.confIdx < b.confIdx;
  });
  const auto continuesRun = [&](const size_t i) {
    return i > 0 && kept[i].collectorIdx == kept[i - 1].collectorIdx &&
           kept[i].srcConfIdx == kept[i - 1].srcConfIdx + 1;
  };
  size_t numRuns = 0;
  for (size_t i = 0; i < kept.size(); ++i) {
    if (!continuesRun(i)) {
      ++numRuns;
    }
  }

  // Copy calls are per run of conformers that are contiguous in both source and result. When there are more runs than
  // collectors (dropped entries or interleaved batches), gather the positions with one kernel instead.
  const bool                 gatherPositions = numRuns > nonEmptyCollectors;
  std::vector<const double*> gatherSrc;
  std::vector<int>           gatherDstAtomStarts;
  std::vector<int>           gatherAtomCounts;

  // The gather kernel only reads memory on the target GPU: stream-ordered allocations on other GPUs are not covered by
  // cudaDeviceEnablePeerAccess, so stage those collectors' buffers over with one peer copy each first.
  std::vector<AsyncDeviceVector<double>> stagedPositions(collectors.size());
  std::vector<const double*>             gatherBase(collectors.size(), nullptr);
  if (gatherPositions) {
    for (size_t c = 0; c < collectors.size(); ++c) {
      auto& collector = collectors[c];
      if (collector.atomCounts.empty()) {
        continue;
      }
      if (collector.gpuId == targetGpu) {
        gatherBase[c] = collector.positions.data();
        continue;
      }
      stagedPositions[c] = AsyncDeviceVector<double>(collector.positions.size(), targetStream.stream());
      copyDeviceToDeviceAsync(stagedPositions[c].data(),
                              collector.positions.data(),
                              collector.positions.size() * sizeof(double),
                              collector.gpuId,
                              collector.stream,
                              targetGpu,
                              targetStream.stream());
      gatherBase[c] = stagedPositions[c].data();
    }
  }

  int dstAtomStart = 0;
  for (size_t runStart = 0; runStart < kept.size();) {
    size_t runEnd = runStart + 1;
    while (runEnd < kept.size() && continuesRun(runEnd)) {
      ++runEnd;
    }
    const int runDstAtomStart = dstAtomStart;
    for (size_t i = runStart; i < runEnd; ++i) {
      const auto& conf   = kept[i];
      atomStartsHost[i]  = dstAtomStart;
      molIndicesHost[i]  = conf.molId;
      confIndicesHost[i] = conf.confIdx;
      if (gatherPositions) {
        gatherSrc.push_back(gatherBase[conf.collectorIdx] + static_cast<size_t>(conf.srcAtomStart) * 3);
        gatherDstAtomStarts.push_back(dstAtomStart);
        gatherAtomCounts.push_back(conf.numAtoms);
      }
      dstAtomStart += conf.numAtoms;
    }

    // The run [runStart, runEnd) is contiguous in both its collector and the result: copy it in one call per buffer.
    const auto&  first     = kept[runStart];
    const auto&  collector = collectors[first.collectorIdx];
    const size_t runConfs  = runEnd - runStart;
    if (!gatherPositions) {
      copyDeviceToDeviceAsync(result.positions.data() + static_cast<size_t>(runDstAtomStart) * 3,
                              collector.positions.data() + static_cast<size_t>(first.srcAtomStart) * 3,
                              static_cast<size_t>(dstAtomStart - runDstAtomStart) * 3 * sizeof(double),
                              collector.gpuId,
                              collector.stream,
                              targetGpu,
                              targetStream.stream());
    }
    if (hasEnergies && collector.energies.size() > 0) {
      copyDeviceToDeviceAsync(result.energies.data() + runStart,
                              collector.energies.data() + first.srcConfIdx,
                              runConfs * sizeof(double),
                              collector.gpuId,
                              collector.stream,
                              targetGpu,
                              targetStream.stream());
    }
    if (hasConverged && collector.converged.size() > 0) {
      copyDeviceToDeviceAsync(result.converged.data() + runStart,
                              collector.converged.data() + first.srcConfIdx,
                              runConfs * sizeof(int8_t),
                              collector.gpuId,
                              collector.stream,
                              targetGpu,
                              targetStream.stream());
    }
    runStart = runEnd;
  }
  atomStartsHost[static_cast<size_t>(totalConformers)] = dstAtomStart;

  if (gatherPositions) {
    // Collector streams are idle once appendActive returns, but make sure before reading same-GPU buffers directly.
    for (const auto& collector : collectors) {
      if (collector.gpuId != targetGpu) {
        continue;
      }
      const WithDevice withCollector(collector.gpuId);
      cudaCheckError(cudaStreamSynchronize(collector.stream));
    }
    gatherConformerPositions(gatherSrc,
                             gatherDstAtomStarts,
                             gatherAtomCounts,
                             result.positions.data(),
                             targetStream.stream());
  }

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
