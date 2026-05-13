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

#include "substruct_search_internal.h"

#include <GraphMol/ROMol.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include <mutex>
#include <vector>

#include "cuda_error_check.h"
#include "gpu_executor.h"
#include "nvtx.h"
#include "pinned_buffer_pool.h"
#include "substruct_launch_config.h"
#include "substruct_search.h"
#include "thread_worker_context.h"

namespace nvMolKit {

// =============================================================================
// MiniBatchResultsDevice Implementation
// =============================================================================

void MiniBatchResultsDevice::setStream(cudaStream_t stream) {
  stream_ = stream;
  matchCounts_.setStream(stream);
  reportedCounts_.setStream(stream);
  matchIndices_.setStream(stream);
  queryAtomCounts_.setStream(stream);
  overflowBuffer_.setStream(stream);
  recursiveMatchBits_.setStream(stream);
  labelMatrixBuffer_.setStream(stream);
}

void MiniBatchResultsDevice::allocateMiniBatch(int        miniBatchSize,
                                               const int* pairMatchStartsDevice,
                                               int        totalMiniBatchMatchIndices,
                                               int        numQueries,
                                               int        maxTargetAtoms,
                                               int        numBuffersPerBlock,
                                               int        maxMatchesToFind,
                                               bool       countOnly) {
  ScopedNvtxRange allocRange("MiniBatchResultsDevice::allocateMiniBatch");

  miniBatchSize_              = miniBatchSize;
  numQueries_                 = numQueries;
  maxTargetAtoms_             = maxTargetAtoms;
  totalMiniBatchMatchIndices_ = countOnly ? 0 : totalMiniBatchMatchIndices;
  overflowBuffersPerBlock_    = numBuffersPerBlock;
  maxMatchesToFind_           = maxMatchesToFind;
  countOnly_                  = countOnly;
  pairMatchStarts_            = pairMatchStartsDevice;

  if (matchCounts_.size() < static_cast<size_t>(miniBatchSize)) {
    matchCounts_.resize(static_cast<size_t>(miniBatchSize * 1.5));
  }

  if (!countOnly) {
    if (reportedCounts_.size() < static_cast<size_t>(miniBatchSize)) {
      reportedCounts_.resize(static_cast<size_t>(miniBatchSize * 1.5));
    }

    if (matchIndices_.size() < static_cast<size_t>(totalMiniBatchMatchIndices)) {
      matchIndices_.resize(static_cast<size_t>(totalMiniBatchMatchIndices) * 3 / 2);
    }
  }

  const int overflowEntries = miniBatchSize * numBuffersPerBlock * kOverflowEntriesPerBuffer;
  if (overflowBuffer_.size() < static_cast<size_t>(overflowEntries)) {
    overflowBuffer_.resize(static_cast<size_t>(overflowEntries * 1.5));
  }

  const size_t recursiveBitsSize = static_cast<size_t>(miniBatchSize) * maxTargetAtoms;
  if (recursiveMatchBits_.size() < recursiveBitsSize) {
    recursiveMatchBits_.resize(static_cast<size_t>(recursiveBitsSize * 1.5));
  }

  const size_t labelMatrixSize = static_cast<size_t>(miniBatchSize) * kLabelMatrixWords;
  if (labelMatrixBuffer_.size() < labelMatrixSize) {
    labelMatrixBuffer_.resize(static_cast<size_t>(labelMatrixSize * 1.5));
  }

  if (overflowFlags_.size() < static_cast<size_t>(miniBatchSize)) {
    overflowFlags_.resize(static_cast<size_t>(miniBatchSize * 1.5));
  }
}

void MiniBatchResultsDevice::setQueryAtomCounts(const int* queryAtomCounts, size_t count) {
  if (queryAtomCounts_.size() < count) {
    queryAtomCounts_.resize(static_cast<size_t>(count * 1.5));
  }
  queryAtomCounts_.copyFromHost(queryAtomCounts, count);
}

void MiniBatchResultsDevice::zeroRecursiveBits() {
  recursiveMatchBits_.zero();
}

void MiniBatchResultsDevice::copyMiniBatchToHost(int*     hostMatchCounts,
                                                 int*     hostReportedCounts,
                                                 int16_t* hostMatchIndices,
                                                 uint8_t* hostOverflowFlags) const {
  matchCounts_.copyToHost(hostMatchCounts, miniBatchSize_);
  reportedCounts_.copyToHost(hostReportedCounts, miniBatchSize_);
  matchIndices_.copyToHost(hostMatchIndices, totalMiniBatchMatchIndices_);
  overflowFlags_.copyToHost(hostOverflowFlags, miniBatchSize_);
}

void MiniBatchResultsDevice::copyCountsOnlyToHost(int* hostMatchCounts) const {
  matchCounts_.copyToHost(hostMatchCounts, miniBatchSize_);
}

// =============================================================================
// RDKit Fallback Implementation
// =============================================================================

void processWithRDKitFallback(const RDKit::ROMol*       target,
                              const RDKit::ROMol*       query,
                              int                       targetIdx,
                              int                       queryIdx,
                              SubstructSearchResults&   results,
                              std::mutex&               resultsMutex,
                              int                       maxMatches,
                              HasSubstructMatchResults* boolResults,
                              std::vector<int>*         countResults) {
  RDKit::SubstructMatchParameters params;
  params.uniquify             = false;
  params.maxMatches           = (maxMatches > 0) ? static_cast<unsigned int>(maxMatches) : 0;
  params.useChirality         = false;
  params.useQueryQueryMatches = false;

  std::vector<RDKit::MatchVectType> rdkitMatches = RDKit::SubstructMatch(*target, *query, params);

  const int matchCount = static_cast<int>(rdkitMatches.size());
  if (matchCount == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(resultsMutex);

  if (boolResults) {
    boolResults->setMatch(targetIdx, queryIdx, true);
  } else if (countResults) {
    const int64_t pairIdx                         = static_cast<int64_t>(targetIdx) * results.numQueries + queryIdx;
    (*countResults)[static_cast<size_t>(pairIdx)] = matchCount;
  } else {
    std::vector<std::vector<int>> convertedMatches;
    convertedMatches.reserve(rdkitMatches.size());
    for (const auto& match : rdkitMatches) {
      std::vector<int> mapping(match.size());
      for (size_t i = 0; i < match.size(); ++i) {
        mapping[i] = match[i].second;
      }
      convertedMatches.push_back(std::move(mapping));
    }

    auto& targetMatches = results.getMatchesMut(targetIdx, queryIdx);
    targetMatches.insert(targetMatches.end(),
                         std::make_move_iterator(convertedMatches.begin()),
                         std::make_move_iterator(convertedMatches.end()));
  }
}

RDKitFallbackQueue::RDKitFallbackQueue(const std::vector<const RDKit::ROMol*>* targets,
                                       const std::vector<const RDKit::ROMol*>* queries,
                                       SubstructSearchResults*                 results,
                                       std::mutex*                             resultsMutex,
                                       int                                     maxMatches,
                                       HasSubstructMatchResults*               boolResults,
                                       std::vector<int>*                       countResults)
    : targets_(targets),
      queries_(queries),
      results_(results),
      boolResults_(boolResults),
      countResults_(countResults),
      resultsMutex_(resultsMutex),
      maxMatches_(maxMatches) {}

void RDKitFallbackQueue::enqueue(const std::vector<RDKitFallbackEntry>& entries) {
  if (entries.empty())
    return;
  queue_.pushBatch(entries);
}

void RDKitFallbackQueue::enqueue(const RDKitFallbackEntry& entry) {
  queue_.push(entry);
}

void RDKitFallbackQueue::registerProducer() {
  std::lock_guard<std::mutex> lock(producerMutex_);
  ++activeProducers_;
}

void RDKitFallbackQueue::unregisterProducer() {
  std::lock_guard<std::mutex> lock(producerMutex_);
  --activeProducers_;
  closeQueueIfDone();
}

void RDKitFallbackQueue::closeQueueIfDone() {
  if (activeProducers_ == 0) {
    queue_.close();
  }
}

size_t RDKitFallbackQueue::processedCount() const {
  return processedCount_.load(std::memory_order_relaxed);
}

std::mutex& RDKitFallbackQueue::getResultsMutex() {
  return *resultsMutex_;
}

bool RDKitFallbackQueue::tryProcessOne() {
  auto optEntry = queue_.tryPop();
  if (!optEntry) {
    return false;
  }
  processEntry(*optEntry);
  return true;
}

bool RDKitFallbackQueue::hasWork() const {
  return !queue_.empty();
}

void RDKitFallbackQueue::processEntry(const RDKitFallbackEntry& entry) {
  ScopedNvtxRange pairRange("RDKit fallback T" + std::to_string(entry.originalTargetIdx) + "/Q" +
                            std::to_string(entry.originalQueryIdx));

  const RDKit::ROMol* target = (*targets_)[entry.originalTargetIdx];
  const RDKit::ROMol* query  = (*queries_)[entry.originalQueryIdx];

  const int effectiveMaxMatches = boolResults_ ? 1 : maxMatches_;
  processWithRDKitFallback(target,
                           query,
                           entry.originalTargetIdx,
                           entry.originalQueryIdx,
                           *results_,
                           *resultsMutex_,
                           effectiveMaxMatches,
                           boolResults_,
                           countResults_);

  processedCount_.fetch_add(1, std::memory_order_relaxed);
}

// =============================================================================
// Batch Results Accumulation
// =============================================================================

void initiateResultsCopyToHost(GpuExecutor& executor, const PinnedHostBuffer& hostBuffer) {
  ScopedNvtxRange copyRange("initiateResultsCopyToHost");
  executor.deviceResults.copyMiniBatchToHost(hostBuffer.matchCounts.data(),
                                             hostBuffer.reportedCounts.data(),
                                             hostBuffer.matchIndices.data(),
                                             hostBuffer.overflowFlags.data());
  cudaCheckError(cudaEventRecord(executor.copyDoneEvent.event(), executor.stream()));
}

void initiateCountsOnlyCopyToHost(GpuExecutor& executor, const PinnedHostBuffer& hostBuffer) {
  ScopedNvtxRange copyRange("initiateCountsOnlyCopyToHost");
  executor.deviceResults.copyCountsOnlyToHost(hostBuffer.matchCounts.data());
  cudaCheckError(cudaEventRecord(executor.copyDoneEvent.event(), executor.stream()));
}

namespace {

/**
 * @brief Resolved pair indices from mini-batch to original indices.
 */
struct ResolvedPairIndices {
  int originalTargetIdx;
  int queryIdx;
};

/**
 * @brief Resolve pair indices from mini-batch-local to original indices.
 */
inline ResolvedPairIndices resolvePairIndices(int                        miniBatchIdx,
                                              const ThreadWorkerContext& ctx,
                                              const PinnedHostBuffer&    hostBuffer) {
  const int batchLocalPairIdx = hostBuffer.pairIndices[miniBatchIdx];
  const int localTargetIdx    = batchLocalPairIdx / ctx.numQueries;
  const int queryIdx          = batchLocalPairIdx % ctx.numQueries;
  const int originalTargetIdx = (*ctx.targetOriginalIndices)[localTargetIdx];
  return {originalTargetIdx, queryIdx};
}

struct PairUpdate {
  int targetIdx;
  int queryIdx;
  int miniBatchLocalOffset;
  int reportedMatches;
  int queryAtoms;
};

}  // namespace

void accumulateMiniBatchResults(GpuExecutor&               executor,
                                const ThreadWorkerContext& ctx,
                                SubstructSearchResults&    results,
                                std::mutex&                resultsMutex,
                                const PinnedHostBuffer&    hostBuffer,
                                RDKitFallbackQueue*        fallbackQueue) {
  ScopedNvtxRange accumRange("accumulateMiniBatchResults");

  std::vector<PairUpdate> updates;
  updates.reserve(executor.plan.numPairsInMiniBatch);

  for (int i = 0; i < executor.plan.numPairsInMiniBatch; ++i) {
    const auto [targetIdx, queryIdx] = resolvePairIndices(i, ctx, hostBuffer);

    const int queryAtoms      = ctx.queryAtomCounts[queryIdx];
    const int actualMatches   = hostBuffer.matchCounts[i];
    const int reportedMatches = hostBuffer.reportedCounts[i];

    const bool isOutputOverflow  = (actualMatches > reportedMatches) && (ctx.maxMatches == 0);
    const bool isPartialOverflow = hostBuffer.overflowFlags[i] != 0;
    if ((isOutputOverflow || isPartialOverflow) && fallbackQueue != nullptr) {
      fallbackQueue->enqueue({targetIdx, queryIdx});
      continue;
    }

    if (reportedMatches > 0) {
      updates.push_back({targetIdx, queryIdx, hostBuffer.miniBatchPairMatchStarts[i], reportedMatches, queryAtoms});
    }
  }

  if (updates.empty()) {
    return;
  }

  std::vector<std::vector<std::vector<int>>*> matchRefs;
  matchRefs.reserve(updates.size());

  {
    std::lock_guard<std::mutex> lock(resultsMutex);
    for (const auto& u : updates) {
      matchRefs.push_back(&results.getMatchesMut(u.targetIdx, u.queryIdx));
    }
  }

  for (size_t i = 0; i < updates.size(); ++i) {
    const auto& u             = updates[i];
    auto&       targetMatches = *matchRefs[i];

    targetMatches.reserve(targetMatches.size() + u.reportedMatches);
    const int16_t* src = hostBuffer.matchIndices.data() + u.miniBatchLocalOffset;

    for (int m = 0; m < u.reportedMatches; ++m) {
      auto& match = targetMatches.emplace_back(u.queryAtoms);
      for (int a = 0; a < u.queryAtoms; ++a) {
        match[a] = src[m * u.queryAtoms + a];
      }
    }
  }
}

void accumulateMiniBatchResultsBoolean(GpuExecutor&               executor,
                                       const ThreadWorkerContext& ctx,
                                       HasSubstructMatchResults&  results,
                                       std::mutex&                resultsMutex,
                                       const PinnedHostBuffer&    hostBuffer) {
  ScopedNvtxRange accumRange("accumulateMiniBatchResultsBoolean");

  std::lock_guard<std::mutex> lock(resultsMutex);

  for (int i = 0; i < executor.plan.numPairsInMiniBatch; ++i) {
    if (hostBuffer.matchCounts[i] == 0) {
      continue;
    }
    const auto [targetIdx, queryIdx] = resolvePairIndices(i, ctx, hostBuffer);
    results.setMatch(targetIdx, queryIdx, true);
  }
}

void accumulateMiniBatchResultsCounts(GpuExecutor&               executor,
                                      const ThreadWorkerContext& ctx,
                                      std::vector<int>&          counts,
                                      std::mutex&                resultsMutex,
                                      const PinnedHostBuffer&    hostBuffer) {
  ScopedNvtxRange accumRange("accumulateMiniBatchResultsCounts");

  std::lock_guard<std::mutex> lock(resultsMutex);

  for (int i = 0; i < executor.plan.numPairsInMiniBatch; ++i) {
    const auto [targetIdx, queryIdx]           = resolvePairIndices(i, ctx, hostBuffer);
    const int64_t globalPairIdx                = static_cast<int64_t>(targetIdx) * ctx.numQueries + queryIdx;
    counts[static_cast<size_t>(globalPairIdx)] = hostBuffer.matchCounts[i];
  }
}

}  // namespace nvMolKit
