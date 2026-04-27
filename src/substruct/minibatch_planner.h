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

#ifndef NVMOLKIT_MINIBATCH_PLANNER_H
#define NVMOLKIT_MINIBATCH_PLANNER_H

#include <array>
#include <vector>

#include "src/substruct/pinned_buffer_pool.h"
#include "src/substruct/substruct_types.h"

namespace nvMolKit {

struct ThreadWorkerContext;
struct LeafSubpatterns;

/**
 * @brief Planning metadata for a single mini-batch.
 */
struct MiniBatchPlan {
  int miniBatchPairOffset         = 0;
  int numPairsInMiniBatch         = 0;
  int totalMatchIndices           = 0;
  int recursiveMaxDepth           = 0;
  int firstTargetInMiniBatch      = 0;
  int numTargetsInMiniBatch       = 0;
  int maxPipelineDepthInMiniBatch = 0;  ///< Max pipeline stage depth in batch

  /// Pointer to precomputed all-queries pattern entries (shared across all mini-batches)
  const std::array<std::vector<BatchedPatternEntry>, kMaxSmartsNestingDepth + 1>* patternsAtDepth = nullptr;

  std::array<int, kMaxSmartsNestingDepth + 1> matchPairsCounts = {};
};

/**
 * @brief CPU-side planner for mini-batch work and recursive scheduling.
 */
class MiniBatchPlanner {
 public:
  void prepareMiniBatch(MiniBatchPlan&             plan,
                        PinnedHostBuffer&          buffer,
                        const ThreadWorkerContext& ctx,
                        const LeafSubpatterns&     leafSubpatterns,
                        int                        miniBatchPairOffset,
                        int                        maxPairsInMiniBatch) const;

 private:
  void precomputePipelineSchedule(MiniBatchPlan& plan, const ThreadWorkerContext& ctx, PinnedHostBuffer& buffer) const;

  void prepareRecursiveMiniBatch(MiniBatchPlan&             plan,
                                 const ThreadWorkerContext& ctx,
                                 const LeafSubpatterns&     leafSubpatterns,
                                 PinnedHostBuffer&          buffer) const;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_MINIBATCH_PLANNER_H
