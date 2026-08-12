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

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/mcs/fmcs_cuda/fmcs.cuh"
#include "src/mcs/mcs_common/mcs_types.cuh"

namespace {

using mcs::Graph;
using mcs::MCSResult;
using mcs::fmcs::Parameters;

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

Graph g(std::size_t n, std::vector<std::pair<std::size_t, std::size_t>> edges) {
  return mcs::buildGraphFromEdges(n, std::move(edges));
}

MCSResult findSingleMCES(const Graph& a, const Graph& b, Parameters params = {}, cudaStream_t stream = nullptr) {
  const auto results = mcs::fmcs::findMCESfMCSBatch({a}, {b}, params, stream);
  EXPECT_EQ(results.size(), 1);
  return results.empty() ? MCSResult{} : results.front();
}

// Path graph: numAtoms vertices in a chain, (i, i+1) edges.
Graph path(int numAtoms) {
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  for (int i = 0; i + 1 < numAtoms; ++i) {
    edges.emplace_back(i, i + 1);
  }
  return mcs::buildGraphFromEdges(static_cast<std::size_t>(numAtoms), std::move(edges));
}

// Cycle graph: numAtoms vertices in a ring.
Graph cycle(int numAtoms) {
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  for (int i = 0; i + 1 < numAtoms; ++i)
    edges.emplace_back(i, i + 1);
  if (numAtoms >= 3)
    edges.emplace_back(0, numAtoms - 1);
  return mcs::buildGraphFromEdges(static_cast<std::size_t>(numAtoms), std::move(edges));
}

// Star K1,n: vertex 0 is the hub, vertices 1..n are leaves.
Graph star(int numLeaves) {
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  for (int i = 1; i <= numLeaves; ++i)
    edges.emplace_back(0, i);
  return mcs::buildGraphFromEdges(static_cast<std::size_t>(numLeaves + 1), std::move(edges));
}

// Naphthalene: two fused 6-rings sharing edge (4,5).
//
//   0 - 1
//   |   |
//   5 - 4 - 6
//   |       |
//   ...    ...
//
// Atoms: 0,1,2,3,4,5 (ring A), 5,4,6,7,8,9 (ring B sharing edge 4-5).
// 10 atoms total.
Graph naphthalene() {
  return g(10,
           {
             {0, 1},
             {1, 2},
             {2, 3},
             {3, 4},
             {4, 5},
             {0, 5}, // ring A
             {4, 6},
             {6, 7},
             {7, 8},
             {8, 9},
             {5, 9}, // ring B (shares 4-5)
  });
}

// Phenanthrene: three angularly fused 6-rings, 14 atoms / 16 bonds.
// Ring A: 0-1-2-3-4-5-0
// Ring B: 4-5-6-7-8-9 (shares edge 4-5)
// Ring C: 8-9-10-11-12-13 (shares edge 8-9)
Graph phenanthrene() {
  return g(14,
           {
             { 0,  1},
             { 1,  2},
             { 2,  3},
             { 3,  4},
             { 4,  5},
             { 0,  5}, // ring A
             { 4,  6},
             { 6,  7},
             { 7,  8},
             { 8,  9},
             { 5,  9}, // ring B
             { 8, 10},
             {10, 11},
             {11, 12},
             {12, 13},
             { 9, 13}, // ring C
  });
}

void expectFullSelfPair(const MCSResult& r, const Graph& gph) {
  EXPECT_FALSE(r.timedOut);
  EXPECT_FALSE(r.overflowed);
  EXPECT_EQ(r.numCommonVertices, gph.numVertices);
  EXPECT_EQ(r.numCommonEdges, gph.numEdges);
}

// Verify the returned mappings form a valid subgraph isomorphism:
//   - mappingA / mappingB cover numCommonVertices distinct query / target
//     atoms; the i-th matched vertex pairs (mappingA[i], mappingB[i]).
//   - For each (qBond, tBond) in (edgeMappingA[i], edgeMappingB[i]), the
//     target endpoints are the targets that the query bond's endpoints
//     mapped to.
void expectMappingsConsistent(const MCSResult& r, const Graph& a, const Graph& b) {
  ASSERT_EQ(static_cast<int>(r.mappingA.size()), r.numCommonVertices);
  ASSERT_EQ(static_cast<int>(r.mappingB.size()), r.numCommonVertices);
  ASSERT_EQ(static_cast<int>(r.edgeMappingA.size()), r.numCommonEdges);
  ASSERT_EQ(static_cast<int>(r.edgeMappingB.size()), r.numCommonEdges);

  // mappingA / mappingB are bijections within the matched subset.
  std::set<std::size_t> qAtoms(r.mappingA.begin(), r.mappingA.end());
  std::set<std::size_t> tAtoms(r.mappingB.begin(), r.mappingB.end());
  EXPECT_EQ(qAtoms.size(), r.mappingA.size()) << "mappingA has duplicate query atoms";
  EXPECT_EQ(tAtoms.size(), r.mappingB.size()) << "mappingB has duplicate target atoms";

  // Every query atom is in graph a; every target atom is in graph b.
  for (std::size_t qa : qAtoms) {
    EXPECT_LT(qa, static_cast<std::size_t>(a.numVertices));
  }
  for (std::size_t ta : tAtoms) {
    EXPECT_LT(ta, static_cast<std::size_t>(b.numVertices));
  }

  // Build atom-mapping lookup: query atom -> target atom.
  std::vector<std::size_t> qToT(a.numVertices, static_cast<std::size_t>(-1));
  for (int i = 0; i < r.numCommonVertices; ++i) {
    qToT[r.mappingA[i]] = r.mappingB[i];
  }

  // Each matched edge: target endpoints == qToT of query endpoints (in
  // either bond orientation).
  for (int i = 0; i < r.numCommonEdges; ++i) {
    const auto qE = r.edgeMappingA[i];
    const auto tE = r.edgeMappingB[i];
    const auto qU = qE.first, qV = qE.second;
    const auto tU = tE.first, tV = tE.second;
    EXPECT_LT(qU, static_cast<std::size_t>(a.numVertices));
    EXPECT_LT(qV, static_cast<std::size_t>(a.numVertices));
    EXPECT_LT(tU, static_cast<std::size_t>(b.numVertices));
    EXPECT_LT(tV, static_cast<std::size_t>(b.numVertices));
    const auto mappedU = qToT[qU];
    const auto mappedV = qToT[qV];
    const bool fwd     = (mappedU == tU && mappedV == tV);
    const bool rev     = (mappedU == tV && mappedV == tU);
    EXPECT_TRUE(fwd || rev) << "Edge mapping (" << qU << "," << qV << ") -> (" << tU << "," << tV
                            << ") inconsistent with atom mapping (" << qU << "->" << mappedU << ", " << qV << "->"
                            << mappedV << ")";
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// FMCSRegression: larger topology cases that have previously regressed.
// ---------------------------------------------------------------------------

TEST(FMCSRegression, FixturePair00x03UnlabeledTopology) {
  // First non-overflow mismatch from tests/test_fmcs_parity.py:
  // sampled_smiles pair00x03, unlabeled.  RDKit's connected MCES is
  // the 17-bond subgraph of the first molecule that omits edge (2,3).
  const auto a = g(17,
                   {
                     { 0,  1},
                     { 1,  2},
                     { 2,  3},
                     { 3,  4},
                     { 4,  5},
                     { 5,  6},
                     { 5,  7},
                     { 7,  8},
                     { 8,  9},
                     { 9, 10},
                     {10, 11},
                     {11, 12},
                     {12, 13},
                     {12, 14},
                     {14, 15},
                     { 4, 16},
                     { 1, 16},
                     { 9, 15}
  });
  const auto b = g(18,
                   {
                     { 0,  1},
                     { 1,  2},
                     { 2,  3},
                     { 3,  4},
                     { 4,  5},
                     { 5,  6},
                     { 6,  7},
                     { 7,  8},
                     { 8,  9},
                     { 8, 10},
                     {10, 11},
                     {11, 12},
                     {12, 13},
                     {13, 14},
                     {13, 15},
                     {15, 16},
                     { 3, 17},
                     { 1, 17},
                     { 3,  6},
                     {10, 16}
  });
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 17);
  EXPECT_EQ(r.numCommonEdges, 17);
  EXPECT_FALSE(r.overflowed);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSRegression, FixturePair01x03UnlabeledTopology) {
  // Current first non-overflow mismatch after restoring Stage 1 coverage:
  // sampled_smiles pair01x03, unlabeled.  RDKit's connected MCES is 17/17.
  const auto a = g(19,
                   {
                     { 0,  1},
                     { 1,  2},
                     { 2,  3},
                     { 2,  4},
                     { 4,  5},
                     { 5,  6},
                     { 6,  7},
                     { 7,  8},
                     { 8,  9},
                     { 9, 10},
                     {10, 11},
                     {11, 12},
                     {10, 13},
                     {13, 14},
                     { 1, 15},
                     {15, 16},
                     {16, 17},
                     {16, 18},
                     { 1, 18},
                     { 8, 14}
  });
  const auto b = g(18,
                   {
                     { 0,  1},
                     { 1,  2},
                     { 2,  3},
                     { 3,  4},
                     { 4,  5},
                     { 5,  6},
                     { 6,  7},
                     { 7,  8},
                     { 8,  9},
                     { 8, 10},
                     {10, 11},
                     {11, 12},
                     {12, 13},
                     {13, 14},
                     {13, 15},
                     {15, 16},
                     { 3, 17},
                     { 1, 17},
                     { 3,  6},
                     {10, 16}
  });
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 17);
  EXPECT_EQ(r.numCommonEdges, 17);
  EXPECT_FALSE(r.overflowed);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSRegression, FiveNodePathInsideTriangleWithLeaves) {
  // Query contains a 4-edge path 4-0-2-1-3 plus the extra chord (0,1).
  // Target is exactly a 4-edge path 0-3-2-1-4.
  const auto a = g(5,
                   {
                     {0, 1},
                     {0, 2},
                     {0, 4},
                     {1, 2},
                     {1, 3}
  });
  const auto b = g(5,
                   {
                     {0, 3},
                     {1, 2},
                     {1, 4},
                     {2, 3}
  });
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 5);
  EXPECT_EQ(r.numCommonEdges, 4);
  EXPECT_FALSE(r.overflowed);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSTiers, MaxSize16) {
  const auto p = path(16);
  auto       r = findSingleMCES(p, p);
  expectFullSelfPair(r, p);
}

TEST(FMCSTiers, MaxSize32) {
  const auto p = path(32);
  auto       r = findSingleMCES(p, p);
  expectFullSelfPair(r, p);
}

TEST(FMCSTiers, MaxSize64) {
  const auto p = path(64);
  auto       r = findSingleMCES(p, p);
  expectFullSelfPair(r, p);
}

TEST(FMCSTiers, MaxSize127) {
  const auto p = path(127);
  auto       r = findSingleMCES(p, p);
  expectFullSelfPair(r, p);
}

TEST(FMCSOverflow, RejectsSize128) {
  const auto atoms128   = path(128);
  auto       atomResult = findSingleMCES(atoms128, atoms128);
  EXPECT_TRUE(atomResult.overflowed);
  EXPECT_EQ(atomResult.numCommonEdges, 0);
  EXPECT_EQ(atomResult.numCommonVertices, 0);

  std::vector<std::pair<std::size_t, std::size_t>> edges;
  for (std::size_t atom = 0; atom + 1 < 127; ++atom)
    edges.emplace_back(atom, atom + 1);
  edges.emplace_back(0, 126);
  edges.emplace_back(0, 2);
  const auto bonds128   = mcs::buildGraphFromEdges(127, std::move(edges));
  auto       bondResult = findSingleMCES(bonds128, bonds128);
  EXPECT_TRUE(bondResult.overflowed);
  EXPECT_EQ(bondResult.numCommonEdges, 0);
  EXPECT_EQ(bondResult.numCommonVertices, 0);
}

// A nonzero timeout routes tiers <= 64 to the production kernel instead of
// the fast block-per-pair kernel; keep that path covered with a timeout far
// too generous to actually fire.
TEST(FMCSTiers, GenerousTimeoutMatchesFastKernelResult) {
  const auto p = path(32);
  Parameters params;
  params.timeoutMs = 60000.0f;
  auto r           = findSingleMCES(p, p, params);
  expectFullSelfPair(r, p);
}

TEST(FMCSInputValidation, RejectsDegreeAboveEight) {
  const auto degreeNine = star(9);
  EXPECT_THROW((void)mcs::fmcs::findMCESfMCSBatch({degreeNine}, {degreeNine}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// FMCSObjective: MaximizeBonds tie-break.
// ---------------------------------------------------------------------------

TEST(FMCSObjective, MaximizeBondsPreferredOverVerticesWhenTie) {
  // A query that contains both a 3-atom path (3 atoms, 2 bonds) and a
  // 3-atom triangle (3 atoms, 3 bonds) as connected subgraphs.  Target
  // is just the triangle.  Both candidate MCSes have 3 atoms; the
  // triangle has more bonds, so MaximizeBonds picks it.
  const auto a = g(4,
                   {
                     {0, 1},
                     {1, 2},
                     {0, 2},
                     {2, 3}
  });
  // Triangle on a, plus a tail 2-3.
  const auto b = cycle(3);
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 3);
  EXPECT_EQ(r.numCommonEdges, 3);
  expectMappingsConsistent(r, a, b);
}

// ---------------------------------------------------------------------------
// FMCSBatch: batch-mode dispatch.
// ---------------------------------------------------------------------------

TEST(FMCSBatch, MixedSizes) {
  // One pair per tier.
  std::vector<Graph> a{path(8), path(24), path(48), path(96)};
  std::vector<Graph> b  = a;
  auto               rs = mcs::fmcs::findMCESfMCSBatch(a, b);
  ASSERT_EQ(rs.size(), 4u);
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_FALSE(rs[i].overflowed) << "pair " << i;
    EXPECT_EQ(rs[i].numCommonEdges, a[i].numEdges) << "pair " << i;
  }
}

TEST(FMCSBatch, EmptyInputReturnsEmpty) {
  std::vector<Graph> a;
  std::vector<Graph> b;
  auto               rs = mcs::fmcs::findMCESfMCSBatch(a, b);
  EXPECT_TRUE(rs.empty());
}

TEST(FMCSBatch, TwoPairsDifferentAnswersSameTier) {
  // Same launch, same tier, different known answers.  A per-block slab
  // indexing bug tends to show up as pair 0 receiving pair 1's answer,
  // or vice versa.
  std::vector<Graph> a{path(3), star(3)};
  std::vector<Graph> b{path(6), star(4)};
  auto               rs = mcs::fmcs::findMCESfMCSBatch(a, b);
  ASSERT_EQ(rs.size(), 2u);

  EXPECT_EQ(rs[0].numCommonVertices, 3);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  expectMappingsConsistent(rs[0], a[0], b[0]);

  EXPECT_EQ(rs[1].numCommonVertices, 4);
  EXPECT_EQ(rs[1].numCommonEdges, 3);
  expectMappingsConsistent(rs[1], a[1], b[1]);
}

TEST(FMCSBatch, NoCommonPairAdjacentToFullMatch) {
  // The zero-bond pairs should not leave stale mappings/results that
  // contaminate the full-match pair in the middle.
  std::vector<Graph> a{g(3, {}), path(5), path(2)};
  std::vector<Graph> b{path(4), path(5), g(2, {})};
  auto               rs = mcs::fmcs::findMCESfMCSBatch(a, b);
  ASSERT_EQ(rs.size(), 3u);

  EXPECT_EQ(rs[0].numCommonVertices, 0);
  EXPECT_EQ(rs[0].numCommonEdges, 0);
  EXPECT_FALSE(rs[0].overflowed);

  expectFullSelfPair(rs[1], a[1]);
  expectMappingsConsistent(rs[1], a[1], b[1]);

  EXPECT_EQ(rs[2].numCommonVertices, 0);
  EXPECT_EQ(rs[2].numCommonEdges, 0);
  EXPECT_FALSE(rs[2].overflowed);
}

TEST(FMCSBatch, EmptyPairInBatch) {
  // Middle pair empty; flanking pairs should still produce results.
  std::vector<Graph> a{path(3), g(0, {}), path(5)};
  std::vector<Graph> b  = a;
  auto               rs = mcs::fmcs::findMCESfMCSBatch(a, b);
  ASSERT_EQ(rs.size(), 3u);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  EXPECT_EQ(rs[1].numCommonEdges, 0);
  EXPECT_EQ(rs[2].numCommonEdges, 4);
}

TEST(FMCSBatch, HonorsBatchSizeChunksResults) {
  std::vector<Graph> a{path(3), star(3), cycle(4), path(5), g(4, {})};
  std::vector<Graph> b{path(6), star(4), cycle(4), path(7), path(3)};
  Parameters         params;
  params.batchSize = 2;

  auto rs = mcs::fmcs::findMCESfMCSBatch(a, b, params);
  ASSERT_EQ(rs.size(), a.size());
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  EXPECT_EQ(rs[1].numCommonEdges, 3);
  EXPECT_EQ(rs[2].numCommonEdges, 4);
  EXPECT_EQ(rs[3].numCommonEdges, 4);
  EXPECT_EQ(rs[4].numCommonEdges, 0);
  for (size_t i = 0; i < rs.size(); ++i) {
    EXPECT_FALSE(rs[i].overflowed) << "pair " << i;
    expectMappingsConsistent(rs[i], a[i], b[i]);
  }
}

TEST(FMCSBatch, MultiExecutorChunksResults) {
  std::vector<Graph> a{path(3), star(3), cycle(4), path(5), g(4, {})};
  std::vector<Graph> b{path(6), star(4), cycle(4), path(7), path(3)};
  Parameters         params;
  params.batchSize          = 1;
  params.executorsPerRunner = 2;

  auto rs = mcs::fmcs::findMCESfMCSBatch(a, b, params);
  ASSERT_EQ(rs.size(), a.size());
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  EXPECT_EQ(rs[1].numCommonEdges, 3);
  EXPECT_EQ(rs[2].numCommonEdges, 4);
  EXPECT_EQ(rs[3].numCommonEdges, 4);
  EXPECT_EQ(rs[4].numCommonEdges, 0);
  for (size_t i = 0; i < rs.size(); ++i) {
    EXPECT_FALSE(rs[i].overflowed) << "pair " << i;
    expectMappingsConsistent(rs[i], a[i], b[i]);
  }
}

TEST(FMCSBatch, ManyTinyPairsReusePerBlockSlabs) {
  // This is intentionally moderate while kFmcsQueueCapacity is still
  // over-provisioned per pair.  It still creates far more blocks than
  // the small correctness batches and exercises repeated queue/cache
  // slab slices within one tier launch.
  constexpr int      kNumPairs = 512;
  std::vector<Graph> a;
  std::vector<Graph> b;
  a.reserve(kNumPairs);
  b.reserve(kNumPairs);
  for (int i = 0; i < kNumPairs; ++i) {
    const int n = (i % 2 == 0) ? 3 : 4;
    a.push_back(path(n));
    b.push_back(path(n));
  }

  auto rs = mcs::fmcs::findMCESfMCSBatch(a, b);
  ASSERT_EQ(rs.size(), static_cast<size_t>(kNumPairs));
  for (int i = 0; i < kNumPairs; ++i) {
    EXPECT_FALSE(rs[i].overflowed) << "pair " << i;
    EXPECT_EQ(rs[i].numCommonVertices, a[i].numVertices) << "pair " << i;
    EXPECT_EQ(rs[i].numCommonEdges, a[i].numEdges) << "pair " << i;
  }
}

TEST(FMCSBatch, TargetLargerThanQueryUsesTargetTier) {
  // MatchResult target-side bitsets are tier-sized too.  This pair
  // should dispatch to the 64 tier even though the query itself fits
  // in tier 16.
  const auto small = path(4);
  const auto large = path(64);

  auto forward = findSingleMCES(small, large);
  EXPECT_EQ(forward.numCommonVertices, small.numVertices);
  EXPECT_EQ(forward.numCommonEdges, small.numEdges);
  EXPECT_FALSE(forward.overflowed);
  expectMappingsConsistent(forward, small, large);

  auto reversed = findSingleMCES(large, small);
  EXPECT_EQ(reversed.numCommonVertices, small.numVertices);
  EXPECT_EQ(reversed.numCommonEdges, small.numEdges);
  EXPECT_FALSE(reversed.overflowed);
  expectMappingsConsistent(reversed, large, small);
}

TEST(FMCSBatch, OverflowPairDoesNotBlockNeighbors) {
  std::vector<Graph> a{path(4), path(200), path(3)};
  std::vector<Graph> b  = a;
  auto               rs = mcs::fmcs::findMCESfMCSBatch(a, b);
  ASSERT_EQ(rs.size(), 3u);

  EXPECT_EQ(rs[0].numCommonEdges, 3);
  EXPECT_FALSE(rs[0].overflowed);

  EXPECT_TRUE(rs[1].overflowed);
  EXPECT_EQ(rs[1].numCommonVertices, 0);
  EXPECT_EQ(rs[1].numCommonEdges, 0);

  EXPECT_EQ(rs[2].numCommonEdges, 2);
  EXPECT_FALSE(rs[2].overflowed);
}

// ---------------------------------------------------------------------------
// FMCSOverflow: graph too large for the largest tier.
// ---------------------------------------------------------------------------

TEST(FMCSOverflow, GraphTooLargeFlagSet) {
  // Tier caps are fewer than 128 atoms and bonds. Build a path with 200 atoms
  // (199 bonds) -- exceeds tier 128 in atoms AND bonds.
  const auto p = path(200);
  auto       r = findSingleMCES(p, p);
  EXPECT_TRUE(r.overflowed);
  EXPECT_EQ(r.numCommonEdges, 0);
  EXPECT_EQ(r.numCommonVertices, 0);
}

TEST(FMCSOverflow, TargetTooLargeFlagSetEvenWhenQueryFits) {
  const auto small = path(4);
  const auto large = path(200);
  auto       r     = findSingleMCES(small, large);
  EXPECT_TRUE(r.overflowed);
  EXPECT_EQ(r.numCommonEdges, 0);
  EXPECT_EQ(r.numCommonVertices, 0);
}

TEST(FMCSTimeout, PartialResultReturned) {
  const auto p = path(127);
  Parameters params;
  params.timeoutMs = 0.0001f;

  auto r = findSingleMCES(p, p, params);
  EXPECT_TRUE(r.timedOut);
  EXPECT_FALSE(r.overflowed);
  EXPECT_LT(r.numCommonEdges, p.numEdges);
  EXPECT_LT(r.numCommonVertices, p.numVertices);
}

// ---------------------------------------------------------------------------
// FMCSMappingConsistency: dedicated mapping-correctness tests.
//
// expectMappingsConsistent already checks the bond-mapping structure;
// these tests pin specific known properties on top of that.
// ---------------------------------------------------------------------------

TEST(FMCSMappingConsistency, SelfPairMappingsAreBijection) {
  const auto c = cycle(6);
  auto       r = findSingleMCES(c, c);
  expectFullSelfPair(r, c);

  // Bijection: every query atom mapped exactly once, every target atom
  // mapped exactly once.
  std::set<std::size_t> qSeen(r.mappingA.begin(), r.mappingA.end());
  std::set<std::size_t> tSeen(r.mappingB.begin(), r.mappingB.end());
  ASSERT_EQ(qSeen.size(), 6u);
  ASSERT_EQ(tSeen.size(), 6u);
  for (int i = 0; i < 6; ++i) {
    EXPECT_TRUE(qSeen.count(i)) << "missing query atom " << i;
    EXPECT_TRUE(tSeen.count(i)) << "missing target atom " << i;
  }
}

TEST(FMCSMappingConsistency, BondMappingsConnectMatchedAtoms) {
  const auto a = path(5);
  const auto b = path(5);
  auto       r = findSingleMCES(a, b);
  expectFullSelfPair(r, a);
  // expectMappingsConsistent does the structural check; we additionally
  // assert that every query bond appears in edgeMappingA exactly once.
  std::set<std::pair<std::size_t, std::size_t>> qBonds;
  for (const auto& e : r.edgeMappingA) {
    auto canon = std::make_pair(std::min(e.first, e.second), std::max(e.first, e.second));
    EXPECT_TRUE(qBonds.insert(canon).second) << "duplicate query bond (" << e.first << "," << e.second << ")";
  }
  EXPECT_EQ(static_cast<int>(qBonds.size()), r.numCommonEdges);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSMappingConsistency, EdgeMappingArrayLengths) {
  // numCommonEdges must equal both edgeMappingA.size() and
  // edgeMappingB.size().  numCommonVertices must equal both
  // mappingA.size() and mappingB.size().
  const auto a = naphthalene();
  const auto b = phenanthrene();
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(static_cast<int>(r.mappingA.size()), r.numCommonVertices);
  EXPECT_EQ(static_cast<int>(r.mappingB.size()), r.numCommonVertices);
  EXPECT_EQ(static_cast<int>(r.edgeMappingA.size()), r.numCommonEdges);
  EXPECT_EQ(static_cast<int>(r.edgeMappingB.size()), r.numCommonEdges);
  expectMappingsConsistent(r, a, b);
}
