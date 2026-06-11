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

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>

#include "src/substruct/molecules_device.cuh"
#include "src/substruct/recursive_preprocessor.h"
#include "src/substruct/substruct_debug.h"
#include "src/substruct/substruct_kernels.h"
#include "src/substruct/substruct_launch_config.h"
#include "src/substruct/substruct_search_internal.h"

namespace nvMolKit {

void LeafSubpatterns::buildAllPatterns(const MoleculesHost& queriesHost) {
  ScopedNvtxRange buildRange("LeafSubpatterns::buildAllPatterns");

  const int numQueries = static_cast<int>(queriesHost.numMolecules());

  // First pass: build pattern molecules and register in patternIndexMap
  for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
    if (queryIdx >= static_cast<int>(queriesHost.recursivePatterns.size())) {
      continue;
    }

    const auto& recursiveInfo = queriesHost.recursivePatterns[queryIdx];
    if (recursiveInfo.empty()) {
      continue;
    }

    for (const auto& entry : recursiveInfo.patterns) {
      if (entry.queryMol == nullptr) {
        continue;
      }

      LeafSubpatternKey key{queryIdx, entry.patternId};
      if (patternIndexMap.find(key) != patternIndexMap.end()) {
        continue;
      }

      int molIdx = static_cast<int>(patternsHost.numMolecules());

      std::vector<std::pair<int, int>> childrenByLocalId;
      for (const auto& p : recursiveInfo.patterns) {
        if (p.parentPatternId == entry.patternId) {
          childrenByLocalId.emplace_back(p.localIdInParent, p.patternId);
        }
      }
      std::sort(childrenByLocalId.begin(), childrenByLocalId.end());

      std::vector<int> childPatternIds;
      for (const auto& [localId, childId] : childrenByLocalId) {
        childPatternIds.push_back(childId);
      }

      if constexpr (kDebugPaintRecursive) {
        printf("[LeafSubpatterns] buildAllPatterns: queryIdx=%d, patternId=%d, found %zu children: [",
               queryIdx,
               entry.patternId,
               childPatternIds.size());
        for (size_t i = 0; i < childPatternIds.size(); ++i) {
          printf("%d%s", childPatternIds[i], i + 1 < childPatternIds.size() ? "," : "");
        }
        printf("]\n");
      }

      if (childPatternIds.empty()) {
        addQueryToBatch(entry.queryMol, patternsHost);
      } else {
        addQueryToBatch(entry.queryMol, patternsHost, childPatternIds);
      }

      patternIndexMap[key] = molIdx;
    }
  }

  if (patternsHost.numMolecules() > 0) {
    for (size_t i = 0; i < patternsHost.numMolecules(); ++i) {
      const int atoms  = patternsHost.batchAtomStarts[i + 1] - patternsHost.batchAtomStarts[i];
      maxPatternAtoms_ = std::max(maxPatternAtoms_, atoms);
    }
  }

  // Second pass: build precomputed BatchedPatternEntry structures
  perQueryPatterns.resize(numQueries);
  perQueryMaxDepth.resize(numQueries, 0);

  for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
    if (queryIdx >= static_cast<int>(queriesHost.recursivePatterns.size())) {
      continue;
    }

    const auto& recursiveInfo = queriesHost.recursivePatterns[queryIdx];
    if (recursiveInfo.empty()) {
      continue;
    }

    perQueryMaxDepth[queryIdx] = recursiveInfo.maxDepth;

    for (const auto& entry : recursiveInfo.patterns) {
      if (entry.queryMol == nullptr) {
        continue;
      }

      const int patternMolIdx = getPatternIndex(queryIdx, entry.patternId);
      if (patternMolIdx < 0) {
        continue;
      }

      if (entry.depth > kMaxSmartsNestingDepth) {
        continue;
      }

      BatchedPatternEntry batchEntry;
      batchEntry.mainQueryIdx    = queryIdx;
      batchEntry.patternId       = entry.patternId;
      batchEntry.patternMolIdx   = patternMolIdx;
      batchEntry.depth           = entry.depth;
      batchEntry.localIdInParent = entry.localIdInParent;

      perQueryPatterns[queryIdx][entry.depth].push_back(batchEntry);
    }
  }

  // Build combined all-queries pattern entries for mini-batches that contain all queries
  for (auto& vec : allQueriesPatternsAtDepth) {
    vec.clear();
  }
  allQueriesMaxDepth = 0;

  for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
    if (queryIdx >= static_cast<int>(perQueryMaxDepth.size())) {
      continue;
    }
    const int queryMaxDepth = std::min(perQueryMaxDepth[queryIdx], kMaxSmartsNestingDepth);
    if (queryMaxDepth > allQueriesMaxDepth) {
      allQueriesMaxDepth = queryMaxDepth;
    }

    for (int d = 0; d <= queryMaxDepth; ++d) {
      const auto& srcEntries  = perQueryPatterns[queryIdx][d];
      auto&       destEntries = allQueriesPatternsAtDepth[d];
      destEntries.insert(destEntries.end(), srcEntries.begin(), srcEntries.end());
    }
  }
}

void LeafSubpatterns::syncToDevice(cudaStream_t stream) {
  ScopedNvtxRange syncRange("LeafSubpatterns::syncToDevice");

  if (!patternsHost.numMolecules()) {
    return;
  }
  patternsDevice.copyFromHost(patternsHost, stream);
}

void RecursivePatternPreprocessor::buildPatterns(const MoleculesHost& queriesHost) {
  leafSubpatterns_.buildAllPatterns(queriesHost);
}

void RecursivePatternPreprocessor::syncToDevice(cudaStream_t stream) {
  leafSubpatterns_.syncToDevice(stream);
}

void RecursivePatternPreprocessor::preprocessMiniBatch(
  const SubstructTemplateConfig                                                   templateConfig,
  const MoleculesDevice&                                                          targetsDevice,
  MiniBatchResultsDevice&                                                         miniBatchResults,
  const int                                                                       numQueries,
  const int                                                                       miniBatchPairOffset,
  const int                                                                       miniBatchSize,
  const SubstructAlgorithm                                                        algorithm,
  cudaStream_t                                                                    stream,
  RecursiveScratchBuffers&                                                        scratch,
  const std::array<std::vector<BatchedPatternEntry>, kMaxSmartsNestingDepth + 1>& patternsAtDepth,
  const int                                                                       maxDepth,
  const int                                                                       firstTargetInMiniBatch,
  const int                                                                       numTargetsInMiniBatch,
  cudaEvent_t*                                                                    depthEvents,
  const int                                                                       numDepthEvents) const {
  ScopedNvtxRange processRecursiveRange("launchRecursivePaintKernels");

  scratch.setStream(stream);

  const auto baseProps        = getTemplateConfigProperties(templateConfig);
  const int  paintQueryAtoms  = std::max(baseProps.maxQueryAtoms, leafSubpatterns_.maxPatternAtoms());
  const int  paintTargetAtoms = std::max(baseProps.maxTargetAtoms, paintQueryAtoms);
  const auto paintConfig      = selectTemplateConfig(paintTargetAtoms, paintQueryAtoms, baseProps.maxBondsPerAtom);

  constexpr int gsiBuffersPerBlock = 2;

  const int maxPaintPairsPerSubBatch = std::max(miniBatchSize, 1024);

  bool isFirstLabelKernel = true;

  for (int currentDepth = 0; currentDepth <= maxDepth; ++currentDepth) {
    ScopedNvtxRange depthRange("Process recursive depth level " + std::to_string(currentDepth));

    const auto& patternsForDepth = patternsAtDepth[currentDepth];

    if (patternsForDepth.empty()) {
      if (currentDepth < numDepthEvents && depthEvents != nullptr) {
        cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
      }
      continue;
    }

    const size_t numPatterns         = patternsForDepth.size();
    const int    patternsPerSubBatch = std::max(1, maxPaintPairsPerSubBatch / numTargetsInMiniBatch);

    for (size_t patternStart = 0; patternStart < numPatterns; patternStart += patternsPerSubBatch) {
      ScopedNvtxRange subBatchRange("Process sub-batch " + std::to_string(patternStart));

      const size_t patternEnd            = std::min(patternStart + patternsPerSubBatch, numPatterns);
      const size_t numPatternsInSubBatch = patternEnd - patternStart;
      const size_t numBlocksInSubBatch   = numTargetsInMiniBatch * numPatternsInSubBatch;

      ScopedNvtxRange prepareRange("GPU: Upload pattern entries");
      const int       bufferIdx = scratch.acquireBufferIndex();
      scratch.waitForBuffer(bufferIdx);
      scratch.ensureCapacity(bufferIdx, static_cast<int>(numPatternsInSubBatch));
      for (size_t i = 0; i < numPatternsInSubBatch; ++i) {
        scratch.patternsAtDepthHost[bufferIdx][i] = patternsForDepth[patternStart + i];
      }
      prepareRange.pop();

      const int    buffersPerBlock = gsiBuffersPerBlock;
      const size_t overflowNeeded  = numBlocksInSubBatch * buffersPerBlock * kOverflowEntriesPerBuffer;

      if (scratch.overflow.size() < overflowNeeded) {
        scratch.overflow.resize(static_cast<size_t>(overflowNeeded * 1.5));
      }

      const size_t labelMatrixNeeded = numBlocksInSubBatch * kLabelMatrixWords;
      if (scratch.labelMatrixBuffer.size() < labelMatrixNeeded) {
        scratch.labelMatrixBuffer.resize(static_cast<size_t>(labelMatrixNeeded * 1.5));
      }

      if (scratch.patternEntries.size() < numPatternsInSubBatch) {
        scratch.patternEntries.resize(static_cast<size_t>(numPatternsInSubBatch * 1.5));
      }

      scratch.patternEntries.copyFromHost(scratch.patternsAtDepthHost[bufferIdx].data(), numPatternsInSubBatch);
      scratch.recordCopy(bufferIdx, scratch.patternEntries.stream());

      const uint32_t* recursiveBitsForLabel = (currentDepth > 0) ? miniBatchResults.recursiveMatchBits() : nullptr;

      std::optional<ZeroBuffersSpec> zeroBuffers;
      if (isFirstLabelKernel) {
        zeroBuffers = ZeroBuffersSpec{miniBatchResults.recursiveMatchBits(),
                                      miniBatchSize * miniBatchResults.maxTargetAtoms(),
                                      miniBatchResults.overflowFlags(),
                                      miniBatchSize};
      }
      isFirstLabelKernel = false;

      launchLabelMatrixPaintKernel(paintConfig,
                                   targetsDevice.view<MoleculeType::Target>(),
                                   leafSubpatterns_.view(),
                                   scratch.patternEntries.data(),
                                   static_cast<int>(numPatternsInSubBatch),
                                   numBlocksInSubBatch,
                                   numQueries,
                                   miniBatchPairOffset,
                                   miniBatchSize,
                                   scratch.labelMatrixBuffer.data(),
                                   firstTargetInMiniBatch,
                                   recursiveBitsForLabel,
                                   miniBatchResults.maxTargetAtoms(),
                                   zeroBuffers,
                                   stream);

      launchSubstructPaintKernel(paintConfig,
                                 algorithm,
                                 targetsDevice.view<MoleculeType::Target>(),
                                 leafSubpatterns_.view(),
                                 scratch.patternEntries.data(),
                                 static_cast<int>(numPatternsInSubBatch),
                                 numBlocksInSubBatch,
                                 miniBatchResults.recursiveMatchBits(),
                                 miniBatchResults.maxTargetAtoms(),
                                 numQueries,
                                 0,
                                 0,
                                 miniBatchPairOffset,
                                 miniBatchSize,
                                 scratch.overflow.data(),
                                 scratch.overflow.data(),
                                 kOverflowEntriesPerBuffer,
                                 scratch.labelMatrixBuffer.data(),
                                 firstTargetInMiniBatch,
                                 stream);
    }

    if (currentDepth < numDepthEvents && depthEvents != nullptr) {
      cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
    }
  }

  cudaCheckError(cudaGetLastError());
}

void preprocessRecursiveSmarts(SubstructTemplateConfig           templateConfig,
                               const MoleculesDevice&            targetsDevice,
                               const MoleculesHost&              queriesHost,
                               const LeafSubpatterns&            leafSubpatterns,
                               MiniBatchResultsDevice&           miniBatchResults,
                               const int                         numQueries,
                               const int                         miniBatchPairOffset,
                               const int                         miniBatchSize,
                               const SubstructAlgorithm          algorithm,
                               cudaStream_t                      stream,
                               RecursiveScratchBuffers&          scratch,
                               std::vector<BatchedPatternEntry>& scratchPatternEntries,
                               cudaEvent_t*                      depthEvents,
                               int                               numDepthEvents) {
  ScopedNvtxRange processRecursiveRange("Process recursive mini-batch with events");

  // Configure kernels for max shared memory carveout (once per process)
  configureSubstructKernelsSharedMem();

  ScopedNvtxRange processRecursiveRangeSetup("Process recursive mini-batch setup");

  scratch.setStream(stream);

  std::vector<BatchedPatternEntry>& patternEntriesHost = scratchPatternEntries;
  patternEntriesHost.clear();

  const int firstQueryInMiniBatch = miniBatchPairOffset % numQueries;
  const int numUniqueQueries      = std::min(miniBatchSize, numQueries);
  const int recursivePatternsSize = static_cast<int>(queriesHost.recursivePatterns.size());

  int maxDepth = 0;
  for (int i = 0; i < numUniqueQueries; ++i) {
    const int queryIdx = (firstQueryInMiniBatch + i) % numQueries;

    if (queryIdx >= recursivePatternsSize) {
      continue;
    }

    const auto& recursiveInfo = queriesHost.recursivePatterns[queryIdx];
    if (recursiveInfo.empty()) {
      continue;
    }

    maxDepth = std::max(maxDepth, recursiveInfo.maxDepth);

    for (const auto& entry : recursiveInfo.patterns) {
      if (entry.queryMol == nullptr) {
        continue;
      }

      const int patternMolIdx = leafSubpatterns.getPatternIndex(queryIdx, entry.patternId);
      if (patternMolIdx < 0) {
        throw std::runtime_error("Pattern not found in pre-built LeafSubpatterns: queryIdx=" +
                                 std::to_string(queryIdx) + ", patternId=" + std::to_string(entry.patternId));
      }

      BatchedPatternEntry& batchEntry = patternEntriesHost.emplace_back();
      batchEntry.mainQueryIdx         = queryIdx;
      batchEntry.patternId            = entry.patternId;
      batchEntry.patternMolIdx        = patternMolIdx;
      batchEntry.depth                = entry.depth;
      batchEntry.localIdInParent      = entry.localIdInParent;
    }
  }

  if (patternEntriesHost.empty()) {
    return;
  }

  const int firstTargetInMiniBatch = miniBatchPairOffset / numQueries;
  const int lastTargetInMiniBatch  = (miniBatchPairOffset + miniBatchSize - 1) / numQueries;
  const int numTargetsInMiniBatch  = lastTargetInMiniBatch - firstTargetInMiniBatch + 1;

  const auto baseProps        = getTemplateConfigProperties(templateConfig);
  const int  paintQueryAtoms  = std::max(baseProps.maxQueryAtoms, leafSubpatterns.maxPatternAtoms());
  const int  paintTargetAtoms = std::max(baseProps.maxTargetAtoms, paintQueryAtoms);
  const auto paintConfig      = selectTemplateConfig(paintTargetAtoms, paintQueryAtoms, baseProps.maxBondsPerAtom);

  constexpr int gsiBuffersPerBlock = 2;

  const int maxPaintPairsPerSubBatch = std::max(miniBatchSize, 1024);
  processRecursiveRangeSetup.pop();

  bool isFirstLabelKernel = true;

  for (int currentDepth = 0; currentDepth <= maxDepth; ++currentDepth) {
    ScopedNvtxRange depthRange("Process recursive depth level " + std::to_string(currentDepth));

    std::vector<BatchedPatternEntry> patternsAtDepth;
    for (const auto& entry : patternEntriesHost) {
      if (entry.depth == currentDepth) {
        patternsAtDepth.push_back(entry);
      }
    }

    if (patternsAtDepth.empty()) {
      if (currentDepth < numDepthEvents && depthEvents != nullptr) {
        cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
      }
      continue;
    }

    const size_t numPatterns         = patternsAtDepth.size();
    const int    patternsPerSubBatch = std::max(1, maxPaintPairsPerSubBatch / numTargetsInMiniBatch);

    for (size_t patternStart = 0; patternStart < numPatterns; patternStart += patternsPerSubBatch) {
      ScopedNvtxRange subBatchRange("Process sub-batch " + std::to_string(patternStart));

      const size_t patternEnd            = std::min(patternStart + patternsPerSubBatch, numPatterns);
      const size_t numPatternsInSubBatch = patternEnd - patternStart;
      const size_t numBlocksInSubBatch   = numTargetsInMiniBatch * numPatternsInSubBatch;

      ScopedNvtxRange prepareRange("CPU: Prepare pattern entries");
      const int       bufferIdx = scratch.acquireBufferIndex();
      scratch.waitForBuffer(bufferIdx);
      scratch.ensureCapacity(bufferIdx, static_cast<int>(numPatternsInSubBatch));
      for (size_t i = 0; i < numPatternsInSubBatch; ++i) {
        scratch.patternsAtDepthHost[bufferIdx][i] = patternsAtDepth[patternStart + i];
      }
      prepareRange.pop();

      const int    buffersPerBlock = gsiBuffersPerBlock;
      const size_t overflowNeeded  = numBlocksInSubBatch * buffersPerBlock * kOverflowEntriesPerBuffer;

      if (scratch.overflow.size() < overflowNeeded) {
        scratch.overflow.resize(static_cast<size_t>(overflowNeeded * 1.5));
      }

      const size_t labelMatrixNeeded = numBlocksInSubBatch * kLabelMatrixWords;
      if (scratch.labelMatrixBuffer.size() < labelMatrixNeeded) {
        scratch.labelMatrixBuffer.resize(static_cast<size_t>(labelMatrixNeeded * 1.5));
      }

      if (scratch.patternEntries.size() < numPatternsInSubBatch) {
        scratch.patternEntries.resize(static_cast<size_t>(numPatternsInSubBatch * 1.5));
      }

      scratch.patternEntries.copyFromHost(scratch.patternsAtDepthHost[bufferIdx].data(), numPatternsInSubBatch);
      scratch.recordCopy(bufferIdx, scratch.patternEntries.stream());

      const uint32_t* recursiveBitsForLabel = (currentDepth > 0) ? miniBatchResults.recursiveMatchBits() : nullptr;

      std::optional<ZeroBuffersSpec> zeroBuffers;
      if (isFirstLabelKernel) {
        zeroBuffers = ZeroBuffersSpec{miniBatchResults.recursiveMatchBits(),
                                      miniBatchSize * miniBatchResults.maxTargetAtoms(),
                                      miniBatchResults.overflowFlags(),
                                      miniBatchSize};
      }
      isFirstLabelKernel = false;

      launchLabelMatrixPaintKernel(paintConfig,
                                   targetsDevice.view<MoleculeType::Target>(),
                                   leafSubpatterns.view(),
                                   scratch.patternEntries.data(),
                                   static_cast<int>(numPatternsInSubBatch),
                                   numBlocksInSubBatch,
                                   numQueries,
                                   miniBatchPairOffset,
                                   miniBatchSize,
                                   scratch.labelMatrixBuffer.data(),
                                   firstTargetInMiniBatch,
                                   recursiveBitsForLabel,
                                   miniBatchResults.maxTargetAtoms(),
                                   zeroBuffers,
                                   stream);

      launchSubstructPaintKernel(paintConfig,
                                 algorithm,
                                 targetsDevice.view<MoleculeType::Target>(),
                                 leafSubpatterns.view(),
                                 scratch.patternEntries.data(),
                                 static_cast<int>(numPatternsInSubBatch),
                                 numBlocksInSubBatch,
                                 miniBatchResults.recursiveMatchBits(),
                                 miniBatchResults.maxTargetAtoms(),
                                 numQueries,
                                 0,
                                 0,
                                 miniBatchPairOffset,
                                 miniBatchSize,
                                 scratch.overflow.data(),
                                 scratch.overflow.data(),
                                 kOverflowEntriesPerBuffer,
                                 scratch.labelMatrixBuffer.data(),
                                 firstTargetInMiniBatch,
                                 stream);
    }

    if (currentDepth < numDepthEvents && depthEvents != nullptr) {
      cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
    }
  }

  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit
