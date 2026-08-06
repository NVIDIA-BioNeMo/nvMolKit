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

// Standard six-membered ring (used as benzene topology).
Graph benzene() {
  return cycle(6);
}

// Toluene: benzene + a methyl substituent on atom 0 -> atom 6.
Graph toluene() {
  return g(7,
           {
             {0, 1},
             {1, 2},
             {2, 3},
             {3, 4},
             {4, 5},
             {0, 5},
             {0, 6}
  });
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

// Biphenyl: two C6 rings linked by a single bond between atom 0 and atom 6.
Graph biphenyl() {
  return g(12,
           {
             { 0,  1},
             { 1,  2},
             { 2,  3},
             { 3,  4},
             { 4,  5},
             { 0,  5}, // ring A
             { 6,  7},
             { 7,  8},
             { 8,  9},
             { 9, 10},
             {10, 11},
             { 6, 11}, // ring B
             { 0,  6}, // linker
  });
}

// Build a labelled LabeledGraph with the given topology, vertex
// labels, and explicit (u, v, label) edge-label triples (symmetrized
// into the dense edgeLabels matrix).
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
// FMCSDispatch: smoke tests on the host-level entry points.
// ---------------------------------------------------------------------------

TEST(FMCSDispatch, BatchSingleEntrySelfPathReturnsFullPath) {
  const auto a = path(4);
  auto       r = findSingleMCES(a, a);
  expectFullSelfPair(r, a);
  expectMappingsConsistent(r, a, a);
}

TEST(FMCSDispatch, BatchOfThreeReturnsExpectedSizes) {
  std::vector<Graph> graphs{path(2), path(3), path(4)};
  auto               rs = mcs::fmcs::findMCESfMCSBatch(graphs, graphs);
  ASSERT_EQ(rs.size(), graphs.size());
  for (size_t i = 0; i < graphs.size(); ++i) {
    EXPECT_FALSE(rs[i].overflowed);
    EXPECT_FALSE(rs[i].timedOut);
    EXPECT_EQ(rs[i].numCommonEdges, graphs[i].numEdges);
    EXPECT_EQ(rs[i].numCommonVertices, graphs[i].numVertices);
  }
}

TEST(FMCSDispatch, MismatchedBatchSizesThrows) {
  std::vector<Graph> a{path(2)};
  std::vector<Graph> b{};
  EXPECT_THROW(mcs::fmcs::findMCESfMCSBatch(a, b), std::runtime_error);
}

// ---------------------------------------------------------------------------
// FMCSDegenerate: empty / single-vertex / no-shared-bond inputs.
// ---------------------------------------------------------------------------

TEST(FMCSDegenerate, EmptyGraphs) {
  const auto a = g(0, {});
  const auto b = g(0, {});
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 0);
  EXPECT_EQ(r.numCommonEdges, 0);
}

TEST(FMCSDegenerate, SingleVertex) {
  // No bonds anywhere -> seed-grow lattice is empty (Phase 1 enumerates
  // bond pairs, of which there are zero).  Connected MCES is reported
  // as size zero.
  const auto a = g(1, {});
  const auto b = g(1, {});
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 0);
  EXPECT_EQ(r.numCommonEdges, 0);
}

TEST(FMCSDegenerate, SingleEdge) {
  const auto a = g(2,
                   {
                     {0, 1}
  });
  const auto b = g(2,
                   {
                     {0, 1}
  });
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 2);
  EXPECT_EQ(r.numCommonEdges, 1);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSDegenerate, OneSideEmpty) {
  const auto a = g(0, {});
  const auto b = path(3);
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 0);
  EXPECT_EQ(r.numCommonEdges, 0);
}

TEST(FMCSDegenerate, DisjointInputs) {
  // Triangle vs three isolated atoms: target has zero bonds, so no
  // (q_bond, t_bond) pair compatible -> empty MCES.
  const auto a = cycle(3);
  const auto b = g(3, {});
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 0);
  EXPECT_EQ(r.numCommonEdges, 0);
}

// ---------------------------------------------------------------------------
// FMCSBasics: hand-derived expected sizes on small graph topologies.
// ---------------------------------------------------------------------------

TEST(FMCSBasics, PathEqualLength) {
  const auto p = path(4);
  auto       r = findSingleMCES(p, p);
  expectFullSelfPair(r, p);
  expectMappingsConsistent(r, p, p);
}

TEST(FMCSBasics, PathShorterVsLonger) {
  const auto a = path(3);
  const auto b = path(6);
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 3);
  EXPECT_EQ(r.numCommonEdges, 2);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSBasics, TreeVsTree) {
  // K1,3 vs K1,4: the smaller star is a connected subgraph of the larger.
  const auto a = star(3);
  const auto b = star(4);
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 4);
  EXPECT_EQ(r.numCommonEdges, 3);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSBasics, HighFanoutStarSelfPair) {
  // Regression for the grow-step boundary scratch space: after seeding
  // one spoke, the hub exposes seven more spokes at once.  A too-small
  // new-bond buffer silently truncated this and missed the full star.
  const auto s = star(8);
  auto       r = findSingleMCES(s, s);
  expectFullSelfPair(r, s);
  expectMappingsConsistent(r, s, s);
}

TEST(FMCSBasics, StarKeepsHubFrontierForSiblingSpokes) {
  // Query side after host swapping is K1,4; target is a diamond
  // (K4 missing one edge).  The exact connected MCES is K1,3: center
  // at either degree-3 diamond vertex.  If a singleton grow from one
  // spoke drops the hub from lastAddedAtoms, sibling spokes are never
  // considered and the search gets stuck at 2 bonds.
  const auto diamond  = g(4,
                          {
                           {0, 1},
                           {0, 2},
                           {1, 2},
                           {1, 3},
                           {2, 3}
  });
  const auto fourStar = star(4);
  auto       r        = findSingleMCES(diamond, fourStar);
  EXPECT_EQ(r.numCommonVertices, 4);
  EXPECT_EQ(r.numCommonEdges, 3);
  EXPECT_FALSE(r.overflowed);
  expectMappingsConsistent(r, diamond, fourStar);
}

TEST(FMCSBasics, CycleVsCycleSame) {
  const auto c = cycle(6);
  auto       r = findSingleMCES(c, c);
  expectFullSelfPair(r, c);
  expectMappingsConsistent(r, c, c);
}

TEST(FMCSBasics, CycleVsCycleLarger) {
  // C3 (triangle) vs C4 (square).  C4 has no triangle subgraph; the
  // best connected common edge subgraph is a 2-bond path.
  const auto a = cycle(3);
  const auto b = cycle(4);
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonEdges, 2);
  EXPECT_EQ(r.numCommonVertices, 3);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSBasics, TreeVsCycle) {
  const auto a = path(3);
  const auto b = cycle(3);
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonEdges, 2);  // path-3 in C3
  EXPECT_EQ(r.numCommonVertices, 3);
  expectMappingsConsistent(r, a, b);
}

// ---------------------------------------------------------------------------
// FMCSConnected: connected-MCES behaviour on graphs whose unconstrained
// common subgraph would be disconnected.
// ---------------------------------------------------------------------------

TEST(FMCSConnected, DisconnectedCommonGraphReturnsOnlyLargestConnected) {
  // Two disjoint edges on each side: 4 atoms / 2 bonds, no path between
  // the two components.  Connected MCES is therefore at most a single
  // edge (the largest connected subgraph of either component).
  const auto a = g(4,
                   {
                     {0, 1},
                     {2, 3}
  });
  const auto b = g(4,
                   {
                     {0, 1},
                     {2, 3}
  });
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonEdges, 1);
  EXPECT_EQ(r.numCommonVertices, 2);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSConnected, TwoComponentsEachReturnsOneComponent) {
  // Each side has a path-3 component and an isolated edge; connected
  // MCES is the larger component (path-3 -> 2 bonds).
  const auto a = g(5,
                   {
                     {0, 1},
                     {1, 2},
                     {3, 4}
  });
  const auto b = g(5,
                   {
                     {0, 1},
                     {1, 2},
                     {3, 4}
  });
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonEdges, 2);
  EXPECT_EQ(r.numCommonVertices, 3);
  expectMappingsConsistent(r, a, b);
}

// ---------------------------------------------------------------------------
// FMCSMolecule: hand-encoded molecule pairs with reference values
// computed from rdFMCS.FindMCS(MaximizeBonds=True, Threshold=1.0,
// AtomCompareElements, BondCompareOrder).  Topology-only here -- the
// labelled equivalents live in FMCSLabels below.
// ---------------------------------------------------------------------------

TEST(FMCSMolecule, BenzeneVsBenzene) {
  const auto a = benzene();
  auto       r = findSingleMCES(a, a);
  expectFullSelfPair(r, a);
  expectMappingsConsistent(r, a, a);
}

TEST(FMCSMolecule, BenzeneVsToluene) {
  // Benzene ring is a connected subgraph of toluene (NullPolicy =
  // topology only, so aromatic-vs-single bond labelling is ignored).
  const auto a = benzene();
  const auto b = toluene();
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 6);
  EXPECT_EQ(r.numCommonEdges, 6);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSMolecule, BenzeneVsCyclohexaneTopologyOnly) {
  // Same topology (6-cycle), different chemistry; NullPolicy ignores
  // bond labels so both look like C6.
  const auto a = benzene();
  const auto b = cycle(6);  // cyclohexane topology = 6-cycle
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 6);
  EXPECT_EQ(r.numCommonEdges, 6);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSMolecule, NaphthaleneVsPhenanthrene) {
  // Naphthalene is a connected subgraph of phenanthrene (the two
  // terminal rings share an edge, matching naphthalene's topology).
  const auto a = naphthalene();
  const auto b = phenanthrene();
  auto       r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 10);
  EXPECT_EQ(r.numCommonEdges, 11);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSMolecule, BiphenylVsBiphenyl) {
  const auto a = biphenyl();
  auto       r = findSingleMCES(a, a);
  expectFullSelfPair(r, a);
  expectMappingsConsistent(r, a, a);
}
