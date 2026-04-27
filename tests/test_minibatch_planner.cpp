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

#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/substruct/minibatch_planner.h"
#include "src/substruct/molecules.h"
#include "src/substruct/pinned_buffer_pool.h"
#include "src/substruct/recursive_preprocessor.h"
#include "src/substruct/thread_worker_context.h"

using nvMolKit::kMaxSmartsNestingDepth;
using nvMolKit::LeafSubpatterns;
using nvMolKit::MiniBatchPlan;
using nvMolKit::MiniBatchPlanner;
using nvMolKit::MoleculesHost;
using nvMolKit::PinnedHostBuffer;
using nvMolKit::PinnedHostBufferPool;
using nvMolKit::RecursivePatternPreprocessor;
using nvMolKit::ThreadWorkerContext;

namespace {

std::unique_ptr<RDKit::ROMol> makeMolFromSmiles(const std::string& smiles) {
  return std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
}

std::unique_ptr<RDKit::ROMol> makeMolFromSmarts(const std::string& smarts) {
  return std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
}

std::vector<int> moleculeAtomCounts(const MoleculesHost& host) {
  const int        numMolecules = static_cast<int>(host.numMolecules());
  std::vector<int> counts(static_cast<size_t>(numMolecules));
  for (int i = 0; i < numMolecules; ++i) {
    const int atomStart = host.batchAtomStarts[i];
    const int atomEnd   = host.batchAtomStarts[i + 1];
    counts[i]           = atomEnd - atomStart;
  }
  return counts;
}

}  // namespace

TEST(MiniBatchPlannerTest, BuildsScheduleAndRecursiveEntries) {
  auto target0 = makeMolFromSmiles("CN");
  auto target1 = makeMolFromSmiles("CC");
  auto query0  = makeMolFromSmarts("CC");
  auto query1  = makeMolFromSmarts("[$(*-N)]");
  auto query2  = makeMolFromSmarts("[$([C;$([C;$(*-N)])])]");

  ASSERT_NE(target0, nullptr);
  ASSERT_NE(target1, nullptr);
  ASSERT_NE(query0, nullptr);
  ASSERT_NE(query1, nullptr);
  ASSERT_NE(query2, nullptr);

  std::vector<const RDKit::ROMol*> targets = {target0.get(), target1.get()};
  std::vector<const RDKit::ROMol*> queries = {query0.get(), query1.get(), query2.get()};
  std::vector<int>                 emptySortOrder;

  MoleculesHost targetsHost;
  nvMolKit::buildTargetBatchParallelInto(targetsHost, 1, targets, emptySortOrder);
  MoleculesHost queriesHost = nvMolKit::buildQueryBatchParallel(queries, emptySortOrder, 1);

  RecursivePatternPreprocessor preprocessor;
  preprocessor.buildPatterns(queriesHost);
  const LeafSubpatterns& leafSubpatterns = preprocessor.leafSubpatterns();

  const std::vector<int> queryAtomCounts  = moleculeAtomCounts(queriesHost);
  const std::vector<int> targetAtomCounts = moleculeAtomCounts(targetsHost);

  std::vector<int>    queryPipelineDepths(static_cast<size_t>(queries.size()));
  std::vector<int>    queryMaxDepths(static_cast<size_t>(queries.size()), 0);
  std::vector<int8_t> queryHasPatterns(static_cast<size_t>(queries.size()), 0);

  const int precomputedSize      = static_cast<int>(leafSubpatterns.perQueryPatterns.size());
  const int perQueryMaxDepthSize = static_cast<int>(leafSubpatterns.perQueryMaxDepth.size());
  for (int q = 0; q < static_cast<int>(queries.size()); ++q) {
    queryPipelineDepths[q] = nvMolKit::getQueryPipelineDepth(queriesHost, q);
    const int maxDepth     = (q < perQueryMaxDepthSize) ? leafSubpatterns.perQueryMaxDepth[q] : 0;
    queryMaxDepths[q]      = maxDepth;
    const bool hasPatterns = (q < precomputedSize) && (maxDepth > 0 || !leafSubpatterns.perQueryPatterns[q][0].empty());
    queryHasPatterns[q]    = hasPatterns ? 1 : 0;
  }
  ASSERT_EQ(queryPipelineDepths.size(), 3u);
  EXPECT_EQ(queryMaxDepths[0], 0);
  EXPECT_EQ(queryMaxDepths[1], 1);
  EXPECT_EQ(queryMaxDepths[2], 3);
  // Pipeline depth is maxRecursiveDepth + 1 so recursive queries wait for paint passes.
  EXPECT_EQ(queryPipelineDepths[0], 0);
  EXPECT_EQ(queryPipelineDepths[1], 2);
  EXPECT_EQ(queryPipelineDepths[2], 4);

  ThreadWorkerContext ctx;
  ctx.queryAtomCounts     = queryAtomCounts.data();
  ctx.queryPipelineDepths = queryPipelineDepths.data();
  ctx.queryMaxDepths      = queryMaxDepths.data();
  ctx.queryHasPatterns    = queryHasPatterns.data();
  ctx.targetAtomCounts    = &targetAtomCounts;
  ctx.numTargets          = static_cast<int>(targets.size());
  ctx.numQueries          = static_cast<int>(queries.size());
  ctx.maxMatches          = 0;

  const int            maxPairsInBatch = ctx.numTargets * ctx.numQueries;
  PinnedHostBufferPool pool;
  pool.initialize(1, maxPairsInBatch, 1, 8);

  PinnedHostBuffer* buffer = pool.acquire();
  ASSERT_NE(buffer, nullptr);

  MiniBatchPlanner planner;
  MiniBatchPlan    plan;
  planner.prepareMiniBatch(plan, *buffer, ctx, leafSubpatterns, 0, maxPairsInBatch);

  EXPECT_EQ(plan.miniBatchPairOffset, 0);
  EXPECT_EQ(plan.numPairsInMiniBatch, maxPairsInBatch);
  for (int i = 0; i < plan.numPairsInMiniBatch; ++i) {
    EXPECT_EQ(buffer->pairIndices[i], i);
  }

  std::vector<int> expectedMatchStarts(static_cast<size_t>(plan.numPairsInMiniBatch + 1), 0);
  int              targetIdx = 0;
  int              queryIdx  = 0;
  for (int i = 0; i < plan.numPairsInMiniBatch; ++i) {
    const int targetAtoms      = (*ctx.targetAtomCounts)[targetIdx];
    const int queryAtoms       = ctx.queryAtomCounts[queryIdx];
    const int pairCapacity     = targetAtoms * queryAtoms;
    expectedMatchStarts[i + 1] = expectedMatchStarts[i] + pairCapacity;
    if (++queryIdx >= ctx.numQueries) {
      queryIdx = 0;
      ++targetIdx;
    }
  }
  for (int i = 0; i <= plan.numPairsInMiniBatch; ++i) {
    EXPECT_EQ(buffer->miniBatchPairMatchStarts[i], expectedMatchStarts[i]);
  }
  EXPECT_EQ(plan.totalMatchIndices, expectedMatchStarts[plan.numPairsInMiniBatch]);

  // With 2 targets and 3 queries, the pair ordering is:
  // (T0,Q0)=0,(T0,Q1)=1,(T0,Q2)=2,(T1,Q0)=3,(T1,Q1)=4,(T1,Q2)=5.
  // Pipeline depths are Q0=0 (no recursion), Q1=2 (maxDepth=1), Q2=4 (maxDepth=3).
  EXPECT_EQ(plan.maxPipelineDepthInMiniBatch, 4);
  EXPECT_EQ(plan.matchPairsCounts[0], 2);
  EXPECT_EQ(plan.matchPairsCounts[2], 2);
  EXPECT_EQ(plan.matchPairsCounts[4], 2);
  EXPECT_EQ(buffer->matchGlobalPairIndicesHost[0][0], 0);
  EXPECT_EQ(buffer->matchGlobalPairIndicesHost[0][1], 3);
  EXPECT_EQ(buffer->matchGlobalPairIndicesHost[2][0], 1);
  EXPECT_EQ(buffer->matchGlobalPairIndicesHost[2][1], 4);
  EXPECT_EQ(buffer->matchGlobalPairIndicesHost[4][0], 2);
  EXPECT_EQ(buffer->matchGlobalPairIndicesHost[4][1], 5);

  EXPECT_EQ(buffer->matchBatchLocalIndicesHost[0][0], 0);
  EXPECT_EQ(buffer->matchBatchLocalIndicesHost[0][1], 3);
  EXPECT_EQ(buffer->matchBatchLocalIndicesHost[2][0], 1);
  EXPECT_EQ(buffer->matchBatchLocalIndicesHost[2][1], 4);
  EXPECT_EQ(buffer->matchBatchLocalIndicesHost[4][0], 2);
  EXPECT_EQ(buffer->matchBatchLocalIndicesHost[4][1], 5);

  int expectedRecursiveMaxDepth = 0;
  for (size_t q = 0; q < queries.size(); ++q) {
    if (queryHasPatterns[q]) {
      expectedRecursiveMaxDepth = std::max(expectedRecursiveMaxDepth, queryMaxDepths[q]);
    }
  }
  EXPECT_EQ(plan.recursiveMaxDepth, expectedRecursiveMaxDepth);

  for (int depth = 0; depth <= nvMolKit::kMaxSmartsNestingDepth; ++depth) {
    size_t expectedCount = 0;
    for (size_t q = 0; q < queries.size(); ++q) {
      if (!queryHasPatterns[q]) {
        continue;
      }
      expectedCount += leafSubpatterns.perQueryPatterns[q][depth].size();
    }
    EXPECT_EQ((*plan.patternsAtDepth)[depth].size(), expectedCount);
  }

  pool.release(buffer);
}
