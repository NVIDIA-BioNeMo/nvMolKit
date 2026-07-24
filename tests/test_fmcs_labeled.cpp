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
#include <cstdint>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "src/mcs/fmcs_cuda/fmcs.cuh"
#include "src/mcs/labeled_graph.h"
#include "src/mcs/mcs_common/mcs_types.cuh"

namespace {

using mcs::Graph;
using mcs::LabeledGraph;
using mcs::MCSResult;
using mcs::fmcs::Parameters;

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

LabeledGraph buildLabeled(int                                         numAtoms,
                          std::vector<std::pair<int, int>>            edges,
                          std::vector<uint16_t>                       vertexLabels,
                          std::vector<std::tuple<int, int, uint16_t>> edgeLabelTriples) {
  LabeledGraph                                     out;
  std::vector<std::pair<std::size_t, std::size_t>> edgesPair;
  edgesPair.reserve(edges.size());
  for (const auto& e : edges)
    edgesPair.emplace_back(e.first, e.second);
  out.graph        = mcs::buildGraphFromEdges(static_cast<std::size_t>(numAtoms), std::move(edgesPair));
  out.vertexLabels = std::move(vertexLabels);
  out.edgeLabels.assign(static_cast<std::size_t>(numAtoms) * numAtoms, 0);
  for (const auto& t : edgeLabelTriples) {
    const int      u                                           = std::get<0>(t);
    const int      v                                           = std::get<1>(t);
    const uint16_t lbl                                         = std::get<2>(t);
    out.edgeLabels[static_cast<std::size_t>(u) * numAtoms + v] = lbl;
    out.edgeLabels[static_cast<std::size_t>(v) * numAtoms + u] = lbl;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Result-checking helpers
// ---------------------------------------------------------------------------

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
TEST(FMCSLabels, NullLabelTopology) {
  // All-zero vertex labels; edge labels uniform.  Should match
  // identically to the topology-only case (path-3 self-pair).
  auto                      m = buildLabeled(3,
                                             {
                          {0, 1},
                          {1, 2}
  },
                        /*vertexLabels=*/{0, 0, 0},
                        /*edges=*/{{0, 1, 1}, {1, 2, 1}});
  std::vector<LabeledGraph> a{m};
  std::vector<LabeledGraph> b{m};
  auto                      rs = mcs::fmcs::findMCESfMCSBatchLabeled(a, b);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonVertices, 3);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
}

TEST(FMCSLabels, VertexLabelMatch) {
  // Two paths with identical vertex labels at corresponding positions.
  auto                      m = buildLabeled(3,
                                             {
                          {0, 1},
                          {1, 2}
  },
                                             {7, 8, 9},
                                             {{0, 1, 1}, {1, 2, 1}});
  std::vector<LabeledGraph> a{m};
  std::vector<LabeledGraph> b{m};
  auto                      rs = mcs::fmcs::findMCESfMCSBatchLabeled(a, b);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonVertices, 3);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
}

TEST(FMCSLabels, VertexLabelMismatch) {
  // Labels {7, 8, 9} vs {7, 8, 10} -- atom 2 doesn't match across.
  // The connected MCES drops the edge (1,2) and ends at the (0,1) edge.
  auto                      a = buildLabeled(3,
                                             {
                          {0, 1},
                          {1, 2}
  },
                                             {7, 8, 9},
                                             {{0, 1, 1}, {1, 2, 1}});
  auto                      b = buildLabeled(3,
                                             {
                          {0, 1},
                          {1, 2}
  },
                                             {7, 8, 10},
                                             {{0, 1, 1}, {1, 2, 1}});
  std::vector<LabeledGraph> as{a};
  std::vector<LabeledGraph> bs{b};
  auto                      rs = mcs::fmcs::findMCESfMCSBatchLabeled(as, bs);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonEdges, 1);
  EXPECT_EQ(rs[0].numCommonVertices, 2);
}

TEST(FMCSLabels, EdgeLabelMatch) {
  // Two cycles with identical edge labels.
  auto                      m = buildLabeled(3,
                                             {
                          {0, 1},
                          {1, 2},
                          {0, 2}
  },
                                             {0, 0, 0},
                                             {{0, 1, 5}, {1, 2, 5}, {0, 2, 5}});
  std::vector<LabeledGraph> a{m};
  std::vector<LabeledGraph> b{m};
  auto                      rs = mcs::fmcs::findMCESfMCSBatchLabeled(a, b);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonVertices, 3);
  EXPECT_EQ(rs[0].numCommonEdges, 3);
}

TEST(FMCSLabels, EdgeLabelMismatch) {
  // Path-3, but one bond's label differs.  That edge can't appear in
  // the MCES; only the matching one survives.
  auto                      a = buildLabeled(3,
                                             {
                          {0, 1},
                          {1, 2}
  },
                                             {0, 0, 0},
                                             {{0, 1, 5}, {1, 2, 7}});
  auto                      b = buildLabeled(3,
                                             {
                          {0, 1},
                          {1, 2}
  },
                                             {0, 0, 0},
                                             {{0, 1, 5}, {1, 2, 9}});
  std::vector<LabeledGraph> as{a};
  std::vector<LabeledGraph> bs{b};
  auto                      rs = mcs::fmcs::findMCESfMCSBatchLabeled(as, bs);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonEdges, 1);
  EXPECT_EQ(rs[0].numCommonVertices, 2);
}

TEST(FMCSLabels, BothLabelsPartialOverlap) {
  // Path-4 with labels diverging at vertex 3 AND edge (2,3) on side B.
  // MCES is the agreeing prefix: 3 atoms, 2 bonds.
  auto                      a = buildLabeled(4,
                                             {
                          {0, 1},
                          {1, 2},
                          {2, 3}
  },
                                             {1, 2, 3, 4},
                                             {{0, 1, 5}, {1, 2, 5}, {2, 3, 5}});
  auto                      b = buildLabeled(4,
                                             {
                          {0, 1},
                          {1, 2},
                          {2, 3}
  },
                                             {1, 2, 3, 9},                        // vertex 3 label diverges
                                             {{0, 1, 5}, {1, 2, 5}, {2, 3, 7}});  // and edge (2,3) too
  std::vector<LabeledGraph> as{a};
  std::vector<LabeledGraph> bs{b};
  auto                      rs = mcs::fmcs::findMCESfMCSBatchLabeled(as, bs);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  EXPECT_EQ(rs[0].numCommonVertices, 3);
}

TEST(FMCSLabels, AtomCompareAnyIgnoresVertexLabels) {
  auto a = buildLabeled(3,
                        {
                          {0, 1},
                          {1, 2}
  },
                        {1, 2, 3},
                        {{0, 1, 5}, {1, 2, 5}});
  auto b = buildLabeled(3,
                        {
                          {0, 1},
                          {1, 2}
  },
                        {7, 8, 9},
                        {{0, 1, 5}, {1, 2, 5}});

  Parameters params;
  params.matchVertexLabels = false;
  params.matchEdgeLabels   = true;
  auto rs                  = mcs::fmcs::findMCESfMCSBatchLabeled({a}, {b}, params);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  EXPECT_EQ(rs[0].numCommonVertices, 3);
  expectMappingsConsistent(rs[0], a.graph, b.graph);
}

TEST(FMCSLabels, BondCompareAnyIgnoresEdgeLabels) {
  auto a = buildLabeled(3,
                        {
                          {0, 1},
                          {1, 2}
  },
                        {1, 2, 3},
                        {{0, 1, 5}, {1, 2, 7}});
  auto b = buildLabeled(3,
                        {
                          {0, 1},
                          {1, 2}
  },
                        {1, 2, 3},
                        {{0, 1, 11}, {1, 2, 13}});

  Parameters params;
  params.matchVertexLabels = true;
  params.matchEdgeLabels   = false;
  auto rs                  = mcs::fmcs::findMCESfMCSBatchLabeled({a}, {b}, params);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  EXPECT_EQ(rs[0].numCommonVertices, 3);
  expectMappingsConsistent(rs[0], a.graph, b.graph);
}

TEST(FMCSLabels, RingMembershipEncodedLabelsRestrictMatches) {
  constexpr uint16_t kRingAtom = static_cast<uint16_t>(6 | (1u << 9));
  constexpr uint16_t kRingBond = static_cast<uint16_t>(1 | (1u << 9));
  auto               ring      = buildLabeled(3,
                                              {
                             {0, 1},
                             {1, 2},
                             {0, 2}
  },
                                              {kRingAtom, kRingAtom, kRingAtom},
                                              {{0, 1, kRingBond}, {1, 2, kRingBond}, {0, 2, kRingBond}});
  auto               chain     = buildLabeled(3,
                                              {
                              {0, 1},
                              {1, 2}
  },
                                              {6, 6, 6},
                                              {{0, 1, 1}, {1, 2, 1}});

  auto strict = mcs::fmcs::findMCESfMCSBatchLabeled({ring}, {chain});
  ASSERT_EQ(strict.size(), 1u);
  EXPECT_EQ(strict[0].numCommonEdges, 0);
  EXPECT_EQ(strict[0].numCommonVertices, 0);

  Parameters compareAny;
  compareAny.matchVertexLabels = false;
  compareAny.matchEdgeLabels   = false;
  auto topologyOnly            = mcs::fmcs::findMCESfMCSBatchLabeled({ring}, {chain}, compareAny);
  ASSERT_EQ(topologyOnly.size(), 1u);
  EXPECT_EQ(topologyOnly[0].numCommonEdges, 2);
  EXPECT_EQ(topologyOnly[0].numCommonVertices, 3);
  expectMappingsConsistent(topologyOnly[0], ring.graph, chain.graph);
}

TEST(FMCSBatch, LabeledMixed) {
  auto                      m1 = buildLabeled(3,
                                              {
                           {0, 1},
                           {1, 2}
  },
                                              {1, 2, 3},
                                              {{0, 1, 7}, {1, 2, 7}});
  auto                      m2 = buildLabeled(4,
                                              {
                           {0, 1},
                           {1, 2},
                           {2, 3}
  },
                                              {1, 2, 3, 4},
                                              {{0, 1, 7}, {1, 2, 7}, {2, 3, 7}});
  std::vector<LabeledGraph> as{m1, m2};
  std::vector<LabeledGraph> bs{m1, m2};
  auto                      rs = mcs::fmcs::findMCESfMCSBatchLabeled(as, bs);
  ASSERT_EQ(rs.size(), 2u);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  EXPECT_EQ(rs[1].numCommonEdges, 3);
}

TEST(FMCSBatch, LabeledTwoPairsDifferentAnswersSameTier) {
  auto full     = buildLabeled(4,
                               {
                             {0, 1},
                             {1, 2},
                             {2, 3}
  },
                               {1, 2, 3, 4},
                               {{0, 1, 7}, {1, 2, 7}, {2, 3, 7}});
  auto partialA = buildLabeled(3,
                               {
                                 {0, 1},
                                 {1, 2}
  },
                               {1, 2, 3},
                               {{0, 1, 5}, {1, 2, 7}});
  auto partialB = buildLabeled(3,
                               {
                                 {0, 1},
                                 {1, 2}
  },
                               {1, 2, 9},
                               {{0, 1, 5}, {1, 2, 7}});

  std::vector<LabeledGraph> as{full, partialA};
  std::vector<LabeledGraph> bs{full, partialB};
  auto                      rs = mcs::fmcs::findMCESfMCSBatchLabeled(as, bs);
  ASSERT_EQ(rs.size(), 2u);
  EXPECT_EQ(rs[0].numCommonVertices, 4);
  EXPECT_EQ(rs[0].numCommonEdges, 3);
  EXPECT_EQ(rs[1].numCommonVertices, 2);
  EXPECT_EQ(rs[1].numCommonEdges, 1);
}
