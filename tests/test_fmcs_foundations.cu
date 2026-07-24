// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "fmcs_cuda/fmcs_match_tables.cuh"
#include "mcs_common/mcs_types.cuh"

namespace {

using mcs::fmcs::MatchTableHost;
using mcs::fmcs::PairMatchTablesHost;

TEST(FMCSGraph, BuildsSortedSymmetricCsr) {
  const auto graph = mcs::buildGraphFromEdges(4,
                                              {
                                                {2, 0},
                                                {1, 2},
                                                {3, 2}
  });

  EXPECT_EQ(graph.numVertices, 4);
  EXPECT_EQ(graph.numEdges, 3);
  EXPECT_EQ(graph.rowOffsets, (std::vector<std::size_t>{0, 1, 2, 5, 6}));
  EXPECT_EQ(graph.colIndices, (std::vector<std::size_t>{2, 2, 0, 1, 3, 2}));
}

TEST(FMCSGraph, BuildsEmptyAndDisconnectedGraphs) {
  const auto empty = mcs::buildGraphFromEdges(0, {});
  EXPECT_EQ(empty.rowOffsets, (std::vector<std::size_t>{0}));
  EXPECT_TRUE(empty.colIndices.empty());

  const auto disconnected = mcs::buildGraphFromEdges(3, {});
  EXPECT_EQ(disconnected.rowOffsets, (std::vector<std::size_t>{0, 0, 0, 0}));
  EXPECT_TRUE(disconnected.colIndices.empty());
}

TEST(FMCSGraph, RejectsInvalidEdges) {
  EXPECT_THROW((void)mcs::buildGraphFromEdges(2,
                                              {
                                                {0, 2}
  }),
               std::runtime_error);
  EXPECT_THROW((void)mcs::buildGraphFromEdges(2,
                                              {
                                                {1, 1}
  }),
               std::runtime_error);
}

TEST(FMCSMatchTable, PacksAcrossWordBoundaries) {
  MatchTableHost table;
  table.resize(2, 65);

  EXPECT_EQ(table.wordsPerRow, 3);
  EXPECT_EQ(table.data.size(), 6);

  table.setBit(0, 0);
  table.setBit(0, 32);
  table.setBit(1, 64);
  EXPECT_TRUE(table.testBit(0, 0));
  EXPECT_TRUE(table.testBit(0, 32));
  EXPECT_TRUE(table.testBit(1, 64));
  EXPECT_FALSE(table.testBit(1, 63));
}

TEST(FMCSMatchTable, UploadsPairsIntoOneContiguousBuffer) {
  std::vector<PairMatchTablesHost> host(2);
  host[0].atoms.resize(1, 2);
  host[0].atoms.setBit(0, 1);
  host[0].bonds.resize(1, 1);
  host[0].bonds.setBit(0, 0);
  host[1].atoms.resize(1, 33);
  host[1].atoms.setBit(0, 32);

  void*       buffer = nullptr;
  std::size_t bytes  = 0;
  const auto  device = mcs::fmcs::uploadPairMatchTables(host, nullptr, &buffer, &bytes);

  ASSERT_NE(buffer, nullptr);
  ASSERT_EQ(device.size(), 2);
  EXPECT_EQ(bytes, 4 * sizeof(std::uint32_t));
  EXPECT_EQ(device[0].atoms.data, static_cast<std::uint32_t*>(buffer));
  EXPECT_EQ(device[0].bonds.data, static_cast<std::uint32_t*>(buffer) + 1);
  EXPECT_EQ(device[1].atoms.data, static_cast<std::uint32_t*>(buffer) + 2);
  EXPECT_EQ(device[1].atoms.wordsPerRow, 2);

  std::vector<std::uint32_t> copied(4);
  ASSERT_EQ(cudaMemcpy(copied.data(), buffer, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(copied, (std::vector<std::uint32_t>{2u, 1u, 0u, 1u}));

  mcs::fmcs::freePairMatchTablesBuffer(buffer, nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(FMCSMatchTable, EmptyUploadReturnsNoAllocation) {
  void*       buffer = reinterpret_cast<void*>(1);
  std::size_t bytes  = 1;
  const auto  device = mcs::fmcs::uploadPairMatchTables({}, nullptr, &buffer, &bytes);

  EXPECT_TRUE(device.empty());
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(bytes, 0);
  mcs::fmcs::freePairMatchTablesBuffer(buffer, nullptr);
}

}  // namespace
