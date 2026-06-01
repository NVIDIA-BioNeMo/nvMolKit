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

#ifndef NVMOLKIT_SUBSTRUCTURE_SEARCH_INTERNAL_H
#define NVMOLKIT_SUBSTRUCTURE_SEARCH_INTERNAL_H

/**
 * @file substruct_search_internal.h
 * @brief Internal implementation details for substructure search.
 *
 * This header exposes internal types and functions needed for testing.
 * Not part of the public API - may change without notice.
 */

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "src/utils/cuda_error_check.h"
#include "src/utils/host_vector.h"
#include "src/utils/nvtx.h"
#include "src/utils/pinned_host_allocator.h"
#include "src/utils/thread_safe_queue.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit
#include "src/substruct/molecules.h"
#include "src/substruct/recursive_preprocessor.h"
#include "src/substruct/substruct_search.h"
#include "src/substruct/thread_worker_context.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

// Forward declarations for friend function
struct DeviceTimingsData;

/**
 * @brief Mini-batch-local device-side storage for substructure match results.
 *
 * Owns device memory for a single mini-batch and provides views for kernel access.
 * Results are copied back to host after each mini-batch and accumulated.
 */
class MiniBatchResultsDevice {
 public:
  MiniBatchResultsDevice() = default;
  explicit MiniBatchResultsDevice(cudaStream_t stream) : stream_(stream) { setStream(stream); }

  /**
   * @brief Allocate mini-batch-local buffers for a specific mini-batch.
   *
   * @param miniBatchSize Number of pairs in this mini-batch
   * @param pairMatchStartsDevice Device pointer to mini-batch-local offsets into matchIndices [miniBatchSize + 1].
   *                              Must remain valid until the mini-batch is complete.
   * @param totalMiniBatchMatchIndices Total match indices capacity for this mini-batch
   * @param numQueries Total number of queries (for kernel view)
   * @param maxTargetAtoms Max atoms per target (stride for recursiveMatchBits)
   * @param numBuffersPerBlock Overflow buffers per block (2 for GSI)
   * @param maxMatchesToFind Stop searching after this many matches (-1 = no limit)
   * @param countOnly If true, count matches but don't store them
   */
  void allocateMiniBatch(int        miniBatchSize,
                         const int* pairMatchStartsDevice,
                         int        totalMiniBatchMatchIndices,
                         int        numQueries,
                         int        maxTargetAtoms,
                         int        numBuffersPerBlock,
                         int        maxMatchesToFind = -1,
                         bool       countOnly        = false);

  void setStream(cudaStream_t stream);

  /**
   * @brief Zero the recursive match bits buffer for a new mini-batch.
   */
  void zeroRecursiveBits();

  /**
   * @brief Copy mini-batch results to raw pinned memory pointers.
   *
   * @param hostMatchCounts Output: match counts for this mini-batch [miniBatchSize]
   * @param hostReportedCounts Output: reported counts for this mini-batch [miniBatchSize]
   * @param hostMatchIndices Output: match indices for this mini-batch
   * @param hostOverflowFlags Output: overflow flags for this mini-batch [miniBatchSize]
   */
  void copyMiniBatchToHost(int*     hostMatchCounts,
                           int*     hostReportedCounts,
                           int16_t* hostMatchIndices,
                           uint8_t* hostOverflowFlags) const;

  /**
   * @brief Copy only match counts to host (for boolean output mode).
   *
   * Skips copying reportedCounts and matchIndices for efficiency when
   * only existence of matches is needed.
   *
   * @param hostMatchCounts Output: match counts for this mini-batch [miniBatchSize]
   */
  void copyCountsOnlyToHost(int* hostMatchCounts) const;

  void setQueryAtomCounts(const int* queryAtomCounts, size_t count);

  [[nodiscard]] int  miniBatchSize() const { return miniBatchSize_; }
  [[nodiscard]] int  numQueries() const { return numQueries_; }
  [[nodiscard]] int  maxTargetAtoms() const { return maxTargetAtoms_; }
  [[nodiscard]] int  overflowBuffersPerBlock() const { return overflowBuffersPerBlock_; }
  [[nodiscard]] int  maxMatchesToFind() const { return maxMatchesToFind_; }
  [[nodiscard]] bool countOnly() const { return countOnly_; }

  [[nodiscard]] int*          matchCounts() const { return matchCounts_.data(); }
  [[nodiscard]] int*          reportedCounts() const { return reportedCounts_.data(); }
  [[nodiscard]] const int*    pairMatchStarts() const { return pairMatchStarts_; }
  [[nodiscard]] int16_t*      matchIndices() const { return matchIndices_.data(); }
  [[nodiscard]] const int*    queryAtomCounts() const { return queryAtomCounts_.data(); }
  [[nodiscard]] PartialMatch* overflowBuffer() const { return overflowBuffer_.data(); }
  [[nodiscard]] uint32_t*     recursiveMatchBits() const { return recursiveMatchBits_.data(); }
  [[nodiscard]] uint32_t*     labelMatrixBuffer() const { return labelMatrixBuffer_.data(); }
  [[nodiscard]] uint8_t*      overflowFlags() const { return overflowFlags_.data(); }

 private:
  cudaStream_t stream_ = nullptr;

  int miniBatchSize_  = 0;
  int numQueries_     = 0;
  int maxTargetAtoms_ = 0;

  AsyncDeviceVector<int>     matchCounts_;
  AsyncDeviceVector<int>     reportedCounts_;
  const int*                 pairMatchStarts_ = nullptr;  ///< External device pointer, not owned
  AsyncDeviceVector<int16_t> matchIndices_;
  AsyncDeviceVector<int>     queryAtomCounts_;

  AsyncDeviceVector<PartialMatch> overflowBuffer_;
  int                             overflowBuffersPerBlock_ = 0;

  AsyncDeviceVector<uint32_t> recursiveMatchBits_;

  AsyncDeviceVector<uint32_t> labelMatrixBuffer_;

  AsyncDeviceVector<uint8_t> overflowFlags_;  ///< Per-pair overflow detection

  int totalMiniBatchMatchIndices_ = 0;

  // Early exit control
  int  maxMatchesToFind_ = -1;
  bool countOnly_        = false;
};

// =============================================================================
// Query Preprocessing Context
// =============================================================================

/**
 * @brief Per-query preprocessing data used during batch search setup.
 *
 * Contains precomputed information about all queries in a batch, including
 * atom counts and recursive pattern metadata for pipeline scheduling.
 */
struct QueryPreprocessContext {
  PinnedHostVector<int> queryAtomCounts;
  std::vector<int>      queryPipelineDepths;  ///< Pipeline stage depth (maxRecursiveDepth + 1, 0 if none)
  std::vector<int>      queryMaxDepths;
  std::vector<int8_t>   queryHasPatterns;
  std::vector<int8_t>   queryNeedsFallback;  ///< 1 if query exceeds limits and needs RDKit fallback
  int                   numQueries    = 0;
  int                   maxQueryAtoms = 0;
};

// =============================================================================
// RDKit Fallback Processing
// =============================================================================

/**
 * @brief Process a single (target, query) pair using RDKit's CPU implementation.
 *
 * Used as fallback for oversized targets or overflow cases.
 *
 * @param boolResults Optional boolean results to populate instead of full matches.
 *                    When non-null, only sets match flag without storing mappings.
 * @param countResults Optional count results to populate instead of full matches.
 */
void processWithRDKitFallback(const RDKit::ROMol*       target,
                              const RDKit::ROMol*       query,
                              int                       targetIdx,
                              int                       queryIdx,
                              SubstructSearchResults&   results,
                              std::mutex&               resultsMutex,
                              int                       maxMatches,
                              HasSubstructMatchResults* boolResults  = nullptr,
                              std::vector<int>*         countResults = nullptr);

/**
 * @brief Thread-safe queue for RDKit fallback processing.
 *
 * Worker threads wait on a condition variable and consume entries as they arrive.
 * Supports concurrent producers (GPU batch accumulators) and consumers (RDKit workers).
 */
class RDKitFallbackQueue {
 public:
  RDKitFallbackQueue(const std::vector<const RDKit::ROMol*>* targets,
                     const std::vector<const RDKit::ROMol*>* queries,
                     SubstructSearchResults*                 results,
                     std::mutex*                             resultsMutex,
                     int                                     maxMatches,
                     HasSubstructMatchResults*               boolResults  = nullptr,
                     std::vector<int>*                       countResults = nullptr);

  void enqueue(const std::vector<RDKitFallbackEntry>& entries);
  void enqueue(const RDKitFallbackEntry& entry);

  void registerProducer();
  void unregisterProducer();

  [[nodiscard]] size_t processedCount() const;
  std::mutex&          getResultsMutex();
  bool                 tryProcessOne();

  [[nodiscard]] bool hasWork() const;

 private:
  void processEntry(const RDKitFallbackEntry& entry);
  void closeQueueIfDone();

  const std::vector<const RDKit::ROMol*>* targets_;
  const std::vector<const RDKit::ROMol*>* queries_;
  SubstructSearchResults*                 results_;
  HasSubstructMatchResults*               boolResults_;
  std::vector<int>*                       countResults_;
  std::mutex*                             resultsMutex_;
  int                                     maxMatches_;

  ThreadSafeQueue<RDKitFallbackEntry> queue_;
  mutable std::mutex                  producerMutex_;
  int                                 activeProducers_ = 0;
  std::atomic<size_t>                 processedCount_{0};
};

/**
 * @brief RAII helper to register/unregister as a producer on the fallback queue.
 */
class FallbackQueueProducerGuard {
 public:
  explicit FallbackQueueProducerGuard(RDKitFallbackQueue* queue) : queue_(queue) {
    if (queue_)
      queue_->registerProducer();
  }
  ~FallbackQueueProducerGuard() {
    if (queue_)
      queue_->unregisterProducer();
  }
  FallbackQueueProducerGuard(const FallbackQueueProducerGuard&)            = delete;
  FallbackQueueProducerGuard& operator=(const FallbackQueueProducerGuard&) = delete;

 private:
  RDKitFallbackQueue* queue_;
};

// =============================================================================
// Batch Results Accumulation
// =============================================================================

struct GpuExecutor;
struct PinnedHostBuffer;

/**
 * @brief Initiate async D2H copy of full match results.
 */
void initiateResultsCopyToHost(GpuExecutor& executor, const PinnedHostBuffer& hostBuffer);

/**
 * @brief Initiate async D2H copy of match counts only.
 */
void initiateCountsOnlyCopyToHost(GpuExecutor& executor, const PinnedHostBuffer& hostBuffer);

/**
 * @brief Accumulate full match results from a completed mini-batch.
 */
void accumulateMiniBatchResults(GpuExecutor&               executor,
                                const ThreadWorkerContext& ctx,
                                SubstructSearchResults&    results,
                                std::mutex&                resultsMutex,
                                const PinnedHostBuffer&    hostBuffer,
                                RDKitFallbackQueue*        fallbackQueue = nullptr);

/**
 * @brief Accumulate boolean match results from a completed mini-batch.
 */
void accumulateMiniBatchResultsBoolean(GpuExecutor&               executor,
                                       const ThreadWorkerContext& ctx,
                                       HasSubstructMatchResults&  results,
                                       std::mutex&                resultsMutex,
                                       const PinnedHostBuffer&    hostBuffer);

/**
 * @brief Accumulate match count results from a completed mini-batch.
 */
void accumulateMiniBatchResultsCounts(GpuExecutor&               executor,
                                      const ThreadWorkerContext& ctx,
                                      std::vector<int>&          counts,
                                      std::mutex&                resultsMutex,
                                      const PinnedHostBuffer&    hostBuffer);

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCTURE_SEARCH_INTERNAL_H
