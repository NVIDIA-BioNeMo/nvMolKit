// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "fmcs_cuda/fmcs_policy.cuh"

namespace {

using mcs::LabeledGraph;
using mcs::fmcs::LabeledFMCSPolicy;
using mcs::fmcs::MatchTableHost;
using mcs::fmcs::NullFMCSPolicy;

LabeledGraph labeledPath(const std::vector<std::uint16_t>& vertices,
                         std::uint16_t                      firstBond,
                         std::uint16_t                      secondBond) {
  LabeledGraph result;
  result.graph        = mcs::buildGraphFromEdges(3, {{0, 1}, {1, 2}});
  result.vertexLabels = vertices;
  result.edgeLabels.assign(9, 0);
  result.edgeLabels[1] = result.edgeLabels[3] = firstBond;
  result.edgeLabels[5] = result.edgeLabels[7] = secondBond;
  return result;
}

TEST(FMCSPolicy, EnumeratesUndirectedBondsDeterministically) {
  const auto graph = mcs::buildGraphFromEdges(4, {{2, 3}, {0, 2}, {1, 2}});
  EXPECT_EQ(mcs::fmcs::enumerateBonds(graph),
            (std::vector<std::pair<int, int>>{{0, 2}, {1, 2}, {2, 3}}));
}

TEST(FMCSPolicy, NullPolicyMakesEveryPairCompatible) {
  const auto query  = mcs::buildGraphFromEdges(2, {{0, 1}});
  const auto target = mcs::buildGraphFromEdges(3, {{0, 1}, {1, 2}});

  MatchTableHost atoms;
  MatchTableHost bonds;
  NullFMCSPolicy::buildAtomMatchTable(query, target, atoms, true);
  NullFMCSPolicy::buildBondMatchTable(query, target, bonds, true);

  for (int q = 0; q < atoms.nRows; ++q)
    for (int t = 0; t < atoms.nCols; ++t)
      EXPECT_TRUE(atoms.testBit(q, t));
  for (int q = 0; q < bonds.nRows; ++q)
    for (int t = 0; t < bonds.nCols; ++t)
      EXPECT_TRUE(bonds.testBit(q, t));
}

TEST(FMCSPolicy, VertexLabelsRestrictCompatibility) {
  const auto query  = labeledPath({6, 6, 8}, 1, 2);
  const auto target = labeledPath({8, 6, 7}, 1, 2);

  MatchTableHost table;
  LabeledFMCSPolicy::buildAtomMatchTable(query, target, table, true);

  EXPECT_FALSE(table.testBit(0, 0));
  EXPECT_TRUE(table.testBit(0, 1));
  EXPECT_TRUE(table.testBit(2, 0));
  EXPECT_FALSE(table.testBit(2, 2));
}

TEST(FMCSPolicy, MissingVertexLabelsBehaveAsZero) {
  auto query         = labeledPath({}, 1, 2);
  auto target        = labeledPath({0, 7, 0}, 1, 2);
  MatchTableHost table;

  LabeledFMCSPolicy::buildAtomMatchTable(query, target, table, true);
  EXPECT_TRUE(table.testBit(0, 0));
  EXPECT_FALSE(table.testBit(0, 1));
  EXPECT_TRUE(table.testBit(2, 2));
}

TEST(FMCSPolicy, BondLabelsRestrictCompatibility) {
  const auto query  = labeledPath({6, 6, 8}, 1, 2);
  const auto target = labeledPath({6, 6, 8}, 2, 1);

  MatchTableHost table;
  LabeledFMCSPolicy::buildBondMatchTable(query, target, table, true);

  EXPECT_FALSE(table.testBit(0, 0));
  EXPECT_TRUE(table.testBit(0, 1));
  EXPECT_TRUE(table.testBit(1, 0));
  EXPECT_FALSE(table.testBit(1, 1));
}

TEST(FMCSPolicy, LabelMatchingCanBeDisabled) {
  const auto query  = labeledPath({1, 2, 3}, 4, 5);
  const auto target = labeledPath({6, 7, 8}, 9, 10);

  MatchTableHost atoms;
  MatchTableHost bonds;
  LabeledFMCSPolicy::buildAtomMatchTable(query, target, atoms, false);
  LabeledFMCSPolicy::buildBondMatchTable(query, target, bonds, false);

  for (int q = 0; q < atoms.nRows; ++q)
    for (int t = 0; t < atoms.nCols; ++t)
      EXPECT_TRUE(atoms.testBit(q, t));
  for (int q = 0; q < bonds.nRows; ++q)
    for (int t = 0; t < bonds.nCols; ++t)
      EXPECT_TRUE(bonds.testBit(q, t));
}

}  // namespace
