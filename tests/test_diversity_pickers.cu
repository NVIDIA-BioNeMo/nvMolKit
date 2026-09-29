// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "src/diversity_pickers.h"
#include "src/utils/device.h"

namespace {

using nvMolKit::AsyncDeviceVector;

template <typename T> AsyncDeviceVector<T> upload(const std::vector<T>& host, cudaStream_t stream) {
  AsyncDeviceVector<T> device(host.size(), stream);
  device.copyFromHost(host);
  return device;
}

std::vector<int> downloadPicks(const nvMolKit::PickerResult& result, cudaStream_t stream) {
  std::vector<int> host(result.indices.size());
  result.indices.copyToHost(host);
  cudaStreamSynchronize(stream);
  return host;
}

TEST(DiversityPickerLeader, HonorsInclusiveCutoffDirectedRowsAndFirstPicks) {
  nvMolKit::ScopedStream const streamOwner;
  const auto                   stream    = streamOwner.stream();
  const std::vector<double>    distances = {
    0.0,
    0.1,
    0.8,
    0.9,
    0.0,
    0.1,
    0.2,
    0.9,
    0.0,
  };
  auto device = upload(distances, stream);

  auto result = nvMolKit::leaderFromDistanceMatrix(toSpan(device), 3, 0.1, 0, {}, stream);
  EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(0, 2));

  result = nvMolKit::leaderFromDistanceMatrix(toSpan(device), 3, 0.1, 2, {2}, stream);
  EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(2, 0));

  const std::vector<double> nonzeroDiagonal = {
    1.0,
    0.8,
    0.8,
    0.8,
    1.0,
    0.8,
    0.8,
    0.8,
    1.0,
  };
  auto nonzeroDiagonalDevice = upload(nonzeroDiagonal, stream);
  result                     = nvMolKit::leaderFromDistanceMatrix(toSpan(nonzeroDiagonalDevice), 3, 0.1, 0, {}, stream);
  EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(0, 1, 2));
}

TEST(DiversityPickerEdges, HandlesEmptyAndSingletonInputs) {
  nvMolKit::ScopedStream const streamOwner;
  const auto                   stream = streamOwner.stream();
  AsyncDeviceVector<double>    emptyMatrix(0, stream);

  auto picks = nvMolKit::leaderFromDistanceMatrix(toSpan(emptyMatrix), 0, 0.1, 0, {}, stream);
  EXPECT_TRUE(downloadPicks(picks, stream).empty());

  auto singleton = upload(std::vector<double>{0.0}, stream);
  picks          = nvMolKit::leaderFromDistanceMatrix(toSpan(singleton), 1, 0.0, 0, {}, stream);
  EXPECT_THAT(downloadPicks(picks, stream), ::testing::ElementsAre(0));
}

TEST(DiversityPickerValidation, RejectsMalformedArguments) {
  nvMolKit::ScopedStream const streamOwner;
  const auto                   stream = streamOwner.stream();
  auto                         matrix = upload(std::vector<double>{0.0, 1.0, 1.0, 0.0}, stream);

  EXPECT_THROW(nvMolKit::leaderFromDistanceMatrix(toSpan(matrix), 3, 0.2, 0, {}, stream), std::invalid_argument);
  EXPECT_THROW(
    nvMolKit::leaderFromDistanceMatrix(toSpan(matrix), 2, std::numeric_limits<double>::quiet_NaN(), 0, {}, stream),
    std::invalid_argument);
  EXPECT_THROW(nvMolKit::leaderFromDistanceMatrix(toSpan(matrix), 2, 0.2, 0, {0, 0}, stream), std::invalid_argument);
  EXPECT_THROW(nvMolKit::leaderFromDistanceMatrix(toSpan(matrix), 2, 0.2, 0, {2}, stream), std::invalid_argument);
}

}  // namespace
