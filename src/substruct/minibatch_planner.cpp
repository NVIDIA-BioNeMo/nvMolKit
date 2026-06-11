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

#include "src/substruct/minibatch_planner.h"

#include <algorithm>

#include "src/substruct/recursive_preprocessor.h"
#include "src/substruct/thread_worker_context.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {

void MiniBatchPlanner::precomputePipelineSchedule(MiniBatchPlan&             plan,
                                                  const ThreadWorkerContext& ctx,
                                                  PinnedHostBuffer&          buffer) const {
  ScopedNvtxRange scheduleRange("CPU: precomputePipelineSchedule");
  int             maxDepth = 0;

  plan.matchPairsCounts.fill(0);

  int queryIdx = plan.miniBatchPairOffset % ctx.numQueries;
  for (int i = 0; i < plan.numPairsInMiniBatch; ++i) {
    const int depth  = ctx.queryPipelineDepths[queryIdx];
    const int offset = plan.matchPairsCounts[depth]++;

    buffer.matchGlobalPairIndicesHost[depth][offset] = buffer.pairIndices[i];
    buffer.matchBatchLocalIndicesHost[depth][offset] = i;

    if (depth > maxDepth) {
      maxDepth = depth;
    }

    if (++queryIdx >= ctx.numQueries) {
      queryIdx = 0;
    }
  }
  plan.maxPipelineDepthInMiniBatch = maxDepth;
}

void MiniBatchPlanner::prepareRecursiveMiniBatch(MiniBatchPlan&             plan,
                                                 const ThreadWorkerContext& ctx,
                                                 const LeafSubpatterns&     leafSubpatterns,
                                                 PinnedHostBuffer&          buffer) const {
  ScopedNvtxRange prepRecRange("prepareRecursiveMiniBatchOnCPU");

  precomputePipelineSchedule(plan, ctx, buffer);

  plan.firstTargetInMiniBatch     = plan.miniBatchPairOffset / ctx.numQueries;
  const int lastTargetInMiniBatch = (plan.miniBatchPairOffset + plan.numPairsInMiniBatch - 1) / ctx.numQueries;
  plan.numTargetsInMiniBatch      = lastTargetInMiniBatch - plan.firstTargetInMiniBatch + 1;

  // Use precomputed all-queries pattern entries (shared across all mini-batches)
  plan.patternsAtDepth   = &leafSubpatterns.allQueriesPatternsAtDepth;
  plan.recursiveMaxDepth = leafSubpatterns.allQueriesMaxDepth;
}

void MiniBatchPlanner::prepareMiniBatch(MiniBatchPlan&             plan,
                                        PinnedHostBuffer&          buffer,
                                        const ThreadWorkerContext& ctx,
                                        const LeafSubpatterns&     leafSubpatterns,
                                        const int                  miniBatchPairOffset,
                                        const int                  maxPairsInMiniBatch) const {
  ScopedNvtxRange prepRange("prepareMiniBatchOnCPU");

  const int numPairs            = ctx.numTargets * ctx.numQueries;
  const int miniBatchEnd        = std::min(miniBatchPairOffset + maxPairsInMiniBatch, numPairs);
  const int numPairsInMiniBatch = miniBatchEnd - miniBatchPairOffset;

  plan.miniBatchPairOffset = miniBatchPairOffset;
  plan.numPairsInMiniBatch = numPairsInMiniBatch;

  const bool useMaxMatchesLimit = ctx.maxMatches > 0;
  int        targetIdx          = miniBatchPairOffset / ctx.numQueries;
  int        queryIdx           = miniBatchPairOffset % ctx.numQueries;

  buffer.miniBatchPairMatchStarts[0] = 0;
  for (int i = 0; i < numPairsInMiniBatch; ++i) {
    const int targetAtoms  = (*ctx.targetAtomCounts)[targetIdx];
    const int queryAtoms   = ctx.queryAtomCounts[queryIdx];
    const int pairCapacity = useMaxMatchesLimit ? (ctx.maxMatches * queryAtoms) : (targetAtoms * queryAtoms);
    buffer.miniBatchPairMatchStarts[i + 1] = buffer.miniBatchPairMatchStarts[i] + pairCapacity;
    buffer.pairIndices[i]                  = miniBatchPairOffset + i;

    if (++queryIdx >= ctx.numQueries) {
      queryIdx = 0;
      ++targetIdx;
    }
  }
  plan.totalMatchIndices = buffer.miniBatchPairMatchStarts[numPairsInMiniBatch];

  prepareRecursiveMiniBatch(plan, ctx, leafSubpatterns, buffer);
}

}  // namespace nvMolKit
