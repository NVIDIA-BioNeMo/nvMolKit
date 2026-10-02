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

#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/p2p.h"

namespace nvMolKit {
namespace detail {

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

  std::unordered_map<int, int> perMolCounter;
  int                          confCursor = 0;
  int                          atomCursor = 0;
  for (size_t c = 0; c < collectors.size(); ++c) {
    auto&     collector = collectors[c];
    const int numConfs  = static_cast<int>(collector.atomCounts.size());
    if (numConfs == 0) {
      continue;
    }

    // Copy each contiguous run of kept conformers in one go. Without filtering this is a single run.
    int        srcAtomCursor = 0;
    int        runSrcAtom    = 0;
    int        runSrcConf    = 0;
    int        runAtoms      = 0;
    int        runConfs      = 0;
    const auto flushRun      = [&]() {
      if (runConfs == 0) {
        return;
      }
      copyDeviceToDeviceAsync(result.positions.data() + static_cast<size_t>(atomCursor - runAtoms) * 3,
                              collector.positions.data() + static_cast<size_t>(runSrcAtom) * 3,
                              static_cast<size_t>(runAtoms) * 3 * sizeof(double),
                              collector.gpuId,
                              collector.stream,
                              targetGpu,
                              targetStream.stream());
      if (hasEnergies && collector.energies.size() > 0) {
        copyDeviceToDeviceAsync(result.energies.data() + (confCursor - runConfs),
                                collector.energies.data() + runSrcConf,
                                static_cast<size_t>(runConfs) * sizeof(double),
                                collector.gpuId,
                                collector.stream,
                                targetGpu,
                                targetStream.stream());
      }
      if (hasConverged && collector.converged.size() > 0) {
        copyDeviceToDeviceAsync(result.converged.data() + (confCursor - runConfs),
                                collector.converged.data() + runSrcConf,
                                static_cast<size_t>(runConfs) * sizeof(int8_t),
                                collector.gpuId,
                                collector.stream,
                                targetGpu,
                                targetStream.stream());
      }
      runAtoms = 0;
      runConfs = 0;
    };

    const bool useExplicitConfIds = !collector.confIds.empty();
    for (int conformerIdx = 0; conformerIdx < numConfs; ++conformerIdx) {
      const int natoms = collector.atomCounts[conformerIdx];
      if (!isKept(c, conformerIdx)) {
        flushRun();
        srcAtomCursor += natoms;
        continue;
      }
      if (runConfs == 0) {
        runSrcAtom = srcAtomCursor;
        runSrcConf = conformerIdx;
      }
      atomStartsHost[static_cast<size_t>(confCursor)] = atomCursor;
      const int molId                                 = collector.molIds[conformerIdx];
      molIndicesHost[static_cast<size_t>(confCursor)] = molId;
      if (useExplicitConfIds) {
        confIndicesHost[static_cast<size_t>(confCursor)] = collector.confIds[conformerIdx];
      } else if (filterByAttempt) {
        confIndicesHost[static_cast<size_t>(confCursor)] = ranks[c][conformerIdx];
      } else {
        confIndicesHost[static_cast<size_t>(confCursor)] = perMolCounter[molId]++;
      }
      atomCursor += natoms;
      srcAtomCursor += natoms;
      runAtoms += natoms;
      ++runConfs;
      ++confCursor;
    }
    flushRun();
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
