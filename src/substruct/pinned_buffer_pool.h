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

#ifndef NVMOLKIT_PINNED_BUFFER_POOL_H
#define NVMOLKIT_PINNED_BUFFER_POOL_H

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "src/substruct/substruct_types.h"
#include "src/utils/pinned_host_allocator.h"
#include "src/utils/thread_safe_queue.h"

namespace nvMolKit {

/**
 * @brief Compute bytes required for the consolidated (fixed-width) portion of a pinned host buffer.
 *
 * This is the size needed for a single H2D memcpy of all maxBatchSize-based buffers.
 */
size_t computeConsolidatedBufferBytes(int maxBatchSize);

/**
 * @brief Compute total bytes required for a pinned host buffer (consolidated + variable).
 */
size_t computePinnedHostBufferBytes(int maxBatchSize, int maxMatchIndicesEstimate, int maxPatternsPerDepth);

/**
 * @brief Metadata for the consolidated fixed-width region of a PinnedHostBuffer.
 *
 * All fixed-width buffers (based on maxBatchSize) are allocated contiguously
 * and can be copied to the device in a single memcpy operation.
 */
struct ConsolidatedBufferInfo {
  std::byte* basePtr      = nullptr;  ///< Start of consolidated region
  size_t     totalBytes   = 0;        ///< Total bytes in consolidated region
  int        maxBatchSize = 0;        ///< Max batch size used to compute offsets
};

/**
 * @brief Host-side pinned buffer for a mini-batch.
 */
struct PinnedHostBuffer {
  PinnedHostView<int>     pairIndices;
  PinnedHostView<int>     miniBatchPairMatchStarts;
  PinnedHostView<int>     matchCounts;
  PinnedHostView<int>     reportedCounts;
  PinnedHostView<int16_t> matchIndices;
  PinnedHostView<uint8_t> overflowFlags;

  std::array<PinnedHostView<int>, kMaxSmartsNestingDepth + 1> matchGlobalPairIndicesHost = {};
  std::array<PinnedHostView<int>, kMaxSmartsNestingDepth + 1> matchBatchLocalIndicesHost = {};
  std::array<PinnedHostView<BatchedPatternEntry>, 2>          patternsAtDepthHost        = {};

  ConsolidatedBufferInfo consolidated;  ///< Info for single-copy H2D transfer of fixed-width buffers
};

/**
 * @brief Pool for pinned host buffers used by pipeline workers.
 */
class PinnedHostBufferPool {
 public:
  void initialize(int poolSize, int maxBatchSize, int maxMatchIndicesEstimate, int maxPatternsPerDepth);

  PinnedHostBuffer* acquire();
  void              release(PinnedHostBuffer* buffer);
  void              shutdown();

 private:
  static std::unique_ptr<PinnedHostBuffer> createBuffer(int maxBatchSize,
                                                        int maxMatchIndicesEstimate,
                                                        int maxPatternsPerDepth);

  std::vector<std::unique_ptr<PinnedHostBuffer>>      buffers_;
  std::unique_ptr<ThreadSafeQueue<PinnedHostBuffer*>> available_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_PINNED_BUFFER_POOL_H
