// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "src/mcs/fmcs_cuda/fmcs_match_tables.cuh"
#include "src/mcs/mcs_common/mcs_types.cuh"
#include "src/utils/cuda_error_check.h"

namespace {

using mcs::fmcs::MatchTableHost;
using mcs::fmcs::PairMatchTablesHost;
using nvMolKit::checkReturnCode;

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

  const auto  upload = mcs::fmcs::uploadPairMatchTables(host, nullptr);
  const auto& device = upload.tables;

  ASSERT_NE(upload.storage.data(), nullptr);
  ASSERT_EQ(device.size(), 2);
  EXPECT_EQ(upload.storage.size(), 4);
  EXPECT_EQ(device[0].atoms.data, upload.storage.data());
  EXPECT_EQ(device[0].bonds.data, upload.storage.data() + 1);
  EXPECT_EQ(device[1].atoms.data, upload.storage.data() + 2);
  EXPECT_EQ(device[1].atoms.wordsPerRow, 2);

  std::vector<std::uint32_t> copied(4);
  upload.storage.copyToHost(copied);
  cudaCheckError(cudaStreamSynchronize(upload.storage.stream()));
  EXPECT_EQ(copied, (std::vector<std::uint32_t>{2u, 1u, 0u, 1u}));
}

TEST(FMCSMatchTable, EmptyUploadReturnsNoAllocation) {
  const auto upload = mcs::fmcs::uploadPairMatchTables({}, nullptr);

  EXPECT_TRUE(upload.tables.empty());
  EXPECT_EQ(upload.storage.data(), nullptr);
  EXPECT_EQ(upload.storage.size(), 0);
}

}  // namespace
