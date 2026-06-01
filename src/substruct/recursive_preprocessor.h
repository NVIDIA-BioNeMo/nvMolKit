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

#ifndef NVMOLKIT_RECURSIVE_PREPROCESSOR_H
#define NVMOLKIT_RECURSIVE_PREPROCESSOR_H

#include <cuda_runtime.h>

#include <array>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/substruct/molecules.h"
#include "src/substruct/molecules_device.cuh"
#include "src/substruct/substruct_types.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"
#include "src/utils/nvtx.h"
#include "src/utils/pinned_host_allocator.h"

namespace nvMolKit {

class MiniBatchResultsDevice;

/**
 * @brief Key for mapping (queryIdx, patternId) to leaf subpattern molecule index.
 */
struct LeafSubpatternKey {
  int queryIdx;
  int patternId;

  bool operator==(const LeafSubpatternKey& other) const {
    return queryIdx == other.queryIdx && patternId == other.patternId;
  }
};

/**
 * @brief Hash function for LeafSubpatternKey.
 */
struct LeafSubpatternKeyHash {
  std::size_t operator()(const LeafSubpatternKey& key) const {
    return std::hash<int>()(key.queryIdx) ^ (std::hash<int>()(key.patternId) << 16);
  }
};

/**
 * @brief Pre-built collection of all recursive SMARTS leaf subpatterns.
 *
 * Contains all recursive patterns from all queries, built once before batch
 * processing begins. Kernels access patterns by molecule index via the
 * patternIndexMap lookup.
 *
 * The device-side data is shared (read-only) across all worker threads.
 */
struct LeafSubpatterns {
  std::unordered_map<LeafSubpatternKey, int, LeafSubpatternKeyHash> patternIndexMap;
  MoleculesHost                                                     patternsHost;
  MoleculesDevice                                                   patternsDevice;

  /// Precomputed pattern entries per query, organized by depth.
  /// perQueryPatterns[queryIdx][depth] = vector of BatchedPatternEntry
  std::vector<std::array<std::vector<BatchedPatternEntry>, kMaxSmartsNestingDepth + 1>> perQueryPatterns;

  /// Max recursion depth per query (0 if no recursive patterns)
  std::vector<int> perQueryMaxDepth;

  /// Precomputed pattern entries for ALL queries combined, organized by depth.
  /// Used when a mini-batch contains all queries to avoid redundant per-query iteration.
  std::array<std::vector<BatchedPatternEntry>, kMaxSmartsNestingDepth + 1> allQueriesPatternsAtDepth;

  /// Max recursion depth across all queries
  int allQueriesMaxDepth = 0;

  int maxPatternAtoms_ = 0;

  LeafSubpatterns() = default;

  /**
   * @brief Build all leaf subpatterns from all queries.
   *
   * Iterates through all queries and extracts all recursive patterns,
   * building them into a single MoleculesHost batch. Must be called
   * before batch processing begins.
   *
   * @param queriesHost Host-side query data containing recursivePatterns
   */
  void buildAllPatterns(const MoleculesHost& queriesHost);

  /**
   * @brief Upload patterns to device.
   *
   * @param stream CUDA stream for async operations
   */
  void syncToDevice(cudaStream_t stream);

  /**
   * @brief Look up a pattern's molecule index.
   *
   * @param queryIdx Index of the query containing the pattern
   * @param patternId Pattern ID within the query
   * @return The molecule index in patternsHost/patternsDevice, or -1 if not found
   */
  [[nodiscard]] int getPatternIndex(int queryIdx, int patternId) const {
    LeafSubpatternKey key{queryIdx, patternId};
    auto              it = patternIndexMap.find(key);
    return (it != patternIndexMap.end()) ? it->second : -1;
  }

  /**
   * @brief Check if any patterns were built.
   */
  [[nodiscard]] bool empty() const { return patternIndexMap.empty(); }

  /**
   * @brief Get the number of patterns.
   */
  [[nodiscard]] size_t size() const { return patternIndexMap.size(); }

  /**
   * @brief Max atom count across all leaf subpatterns.
   */
  [[nodiscard]] int maxPatternAtoms() const { return maxPatternAtoms_; }

  /**
   * @brief Get view for kernel access.
   */
  [[nodiscard]] QueryMoleculesDeviceView view() const { return patternsDevice.view<MoleculeType::Query>(); }
};

/**
 * @brief Scratch buffers for recursive SMARTS preprocessing.
 *
 * Reusable device memory to avoid repeated alloc/free between kernels.
 * For nested patterns, intermediateBits holds results from child levels
 * that become input for parent patterns.
 *
 * Uses double-buffered pinned memory for pattern entries to avoid CPU stalls
 * waiting for H2D copies to complete. While one buffer is being copied, the
 * other can be filled with the next sub-batch's data.
 */
struct RecursiveScratchBuffers {
  AsyncDeviceVector<BatchedPatternEntry> patternEntries;
  AsyncDeviceVector<PartialMatch>        overflow;
  AsyncDeviceVector<uint32_t>            labelMatrixBuffer;
  AsyncDeviceVector<uint32_t>            intermediateBits;  ///< Child pattern results for nested recursion

  /// Double-buffered pinned pattern entries for overlap
  std::array<PinnedHostView<BatchedPatternEntry>, 2> patternsAtDepthHost         = {};
  std::array<int, 2>                                 patternsAtDepthHostCapacity = {0, 0};
  std::array<ScopedCudaEvent, 2>                     patternsAtDepthHostCopyDone;
  std::array<bool, 2>                                patternsAtDepthHostCopyPending = {false, false};
  int                                                currentPatternBuffer = 0;  ///< Index of buffer to fill next
  PinnedHostAllocator                                pinnedAllocator_;

  explicit RecursiveScratchBuffers(cudaStream_t stream)
      : patternEntries(),
        overflow(),
        labelMatrixBuffer(),
        intermediateBits() {
    patternEntries.setStream(stream);
    overflow.setStream(stream);
    labelMatrixBuffer.setStream(stream);
    intermediateBits.setStream(stream);
  }

  ~RecursiveScratchBuffers() = default;

  RecursiveScratchBuffers(const RecursiveScratchBuffers&)            = delete;
  RecursiveScratchBuffers& operator=(const RecursiveScratchBuffers&) = delete;
  RecursiveScratchBuffers(RecursiveScratchBuffers&&)                 = delete;
  RecursiveScratchBuffers& operator=(RecursiveScratchBuffers&&)      = delete;

  void setStream(cudaStream_t stream) {
    patternEntries.setStream(stream);
    overflow.setStream(stream);
    labelMatrixBuffer.setStream(stream);
    intermediateBits.setStream(stream);
  }

  void setPinnedBuffer(const std::array<PinnedHostView<BatchedPatternEntry>, 2>& views, int capacity) {
    for (int i = 0; i < 2; ++i) {
      patternsAtDepthHost[i]         = views[i];
      patternsAtDepthHostCapacity[i] = capacity;
    }
  }

  /**
   * @brief Allocate owned pinned buffers with given capacity.
   * For tests and standalone usage.
   */
  void allocateBuffers(int capacity) {
    const size_t bufferBytes       = static_cast<size_t>(capacity) * sizeof(BatchedPatternEntry);
    pinnedAllocator_               = PinnedHostAllocator(bufferBytes * 2 + 256);
    patternsAtDepthHost[0]         = pinnedAllocator_.allocate<BatchedPatternEntry>(capacity);
    patternsAtDepthHost[1]         = pinnedAllocator_.allocate<BatchedPatternEntry>(capacity);
    patternsAtDepthHostCapacity[0] = capacity;
    patternsAtDepthHostCapacity[1] = capacity;
  }

  /**
   * @brief Get the current buffer index and advance to next for double-buffering.
   */
  int acquireBufferIndex() {
    int idx              = currentPatternBuffer;
    currentPatternBuffer = 1 - currentPatternBuffer;
    return idx;
  }

  /**
   * @brief Wait for a specific buffer's copy to complete if pending.
   */
  void waitForBuffer(int bufferIdx) {
    if (patternsAtDepthHostCopyPending[bufferIdx]) {
      cudaCheckError(cudaEventSynchronize(patternsAtDepthHostCopyDone[bufferIdx].event()));
      patternsAtDepthHostCopyPending[bufferIdx] = false;
    }
  }

  /**
   * @brief Record that a copy has been initiated on a buffer.
   */
  void recordCopy(int bufferIdx, cudaStream_t stream) {
    cudaCheckError(cudaEventRecord(patternsAtDepthHostCopyDone[bufferIdx].event(), stream));
    patternsAtDepthHostCopyPending[bufferIdx] = true;
  }

  /**
   * @brief Check that pinned buffer has sufficient capacity.
   * @throws std::runtime_error if capacity is exceeded or buffer not initialized
   */
  void ensureCapacity(int bufferIdx, int requiredCapacity) {
    if (patternsAtDepthHostCapacity[bufferIdx] >= requiredCapacity) {
      return;
    }
    throw std::runtime_error(
      "Recursive SMARTS pattern count (" + std::to_string(requiredCapacity) + ") exceeds pre-allocated capacity (" +
      std::to_string(patternsAtDepthHostCapacity[bufferIdx]) + "). Ensure buffers are properly initialized.");
  }
};

/**
 * @brief Recursive SMARTS preprocessing component.
 */
class RecursivePatternPreprocessor {
 public:
  RecursivePatternPreprocessor() = default;

  void buildPatterns(const MoleculesHost& queriesHost);
  void syncToDevice(cudaStream_t stream);

  [[nodiscard]] const LeafSubpatterns& leafSubpatterns() const { return leafSubpatterns_; }

  void preprocessMiniBatch(
    SubstructTemplateConfig                                                         templateConfig,
    const MoleculesDevice&                                                          targetsDevice,
    MiniBatchResultsDevice&                                                         miniBatchResults,
    int                                                                             numQueries,
    int                                                                             miniBatchPairOffset,
    int                                                                             miniBatchSize,
    SubstructAlgorithm                                                              algorithm,
    cudaStream_t                                                                    stream,
    RecursiveScratchBuffers&                                                        scratch,
    const std::array<std::vector<BatchedPatternEntry>, kMaxSmartsNestingDepth + 1>& patternsAtDepth,
    int                                                                             maxDepth,
    int                                                                             firstTargetInMiniBatch,
    int                                                                             numTargetsInMiniBatch,
    cudaEvent_t*                                                                    depthEvents,
    int                                                                             numDepthEvents) const;

 private:
  LeafSubpatterns leafSubpatterns_;
};

/**
 * @brief Preprocess ALL recursive SMARTS patterns for a mini-batch.
 *
 * Uses pre-built leaf subpatterns to run paint kernels for all recursive patterns
 * that affect pairs in the current mini-batch. Optionally records events after each
 * depth level for pipeline synchronization.
 *
 * @param targetsDevice Device-resident target molecules
 * @param queriesHost Host-side query data (contains recursivePatterns per query)
 * @param leafSubpatterns Pre-built leaf subpattern molecules (device-resident)
 * @param miniBatchResults The mini-batch results buffer where recursiveMatchBits will be written
 * @param numQueries Total number of queries (for computing pair indices)
 * @param miniBatchPairOffset Global pair index where current mini-batch starts
 * @param miniBatchSize Number of pairs in this mini-batch
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param scratch Reusable scratch buffers (avoids alloc/free between kernels)
 * @param scratchPatternEntries Vector to store pattern entries for the mini-batch
 * @param depthEvents Array of events to record after each depth level, or nullptr
 * @param numDepthEvents Number of events in the array (typically kMaxSmartsNestingDepth)
 */
void preprocessRecursiveSmarts(SubstructTemplateConfig           templateConfig,
                               const MoleculesDevice&            targetsDevice,
                               const MoleculesHost&              queriesHost,
                               const LeafSubpatterns&            leafSubpatterns,
                               MiniBatchResultsDevice&           miniBatchResults,
                               int                               numQueries,
                               int                               miniBatchPairOffset,
                               int                               miniBatchSize,
                               SubstructAlgorithm                algorithm,
                               cudaStream_t                      stream,
                               RecursiveScratchBuffers&          scratch,
                               std::vector<BatchedPatternEntry>& scratchPatternEntries,
                               cudaEvent_t*                      depthEvents,
                               int                               numDepthEvents);

}  // namespace nvMolKit

#endif  // NVMOLKIT_RECURSIVE_PREPROCESSOR_H
