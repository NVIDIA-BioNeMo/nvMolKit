// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "src/diversity_picker_algorithms.cuh"
#include "src/diversity_pickers.h"
#include "src/utils/device.h"

namespace {

using nvMolKit::AsyncDeviceVector;
using nvMolKit::FingerprintSimilarityMetric;

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

struct UnevaluatedDistanceProvider {
  int numItems;

  int size() const { return numItems; }
  int leaderWindow() const { return 32; }

  template <typename Operation> void accumulateDistances(const int*, int, const Operation&, cudaStream_t) const {
    throw std::logic_error("This selection must not evaluate distances");
  }
};

TEST(DiversityPickerLeader, ReturnsCompleteSelectionsWithoutEvaluatingDistances) {
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream = streamOwner.stream();
  UnevaluatedDistanceProvider  provider{65};

  auto result = nvMolKit::detail::leaderPick(provider, 0.5F, 1, {}, nullptr, stream);
  EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(0));

  for (const int limit : {1, 2, 3}) {
    result = nvMolKit::detail::leaderPick(provider, 0.5F, limit, {64, 0, 32}, nullptr, stream);
    EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(64, 0, 32));
    EXPECT_FLOAT_EQ(result.lastDistance, -1.0F);
  }
  EXPECT_THROW(nvMolKit::detail::leaderPick(provider, 0.5F, 1, {0, 0}, nullptr, stream), std::invalid_argument);
  EXPECT_THROW(nvMolKit::detail::leaderPick(provider, 0.5F, 1, {65}, nullptr, stream), std::invalid_argument);

  provider.numItems = 1;
  result            = nvMolKit::detail::leaderPick(provider, 0.0F, 0, {}, nullptr, stream);
  EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(0));
}

TEST(DiversityPickerLeader, CompleteSelectionsStillEvaluateDistancesWhenLabelsAreRequested) {
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream = streamOwner.stream();
  UnevaluatedDistanceProvider  provider{3};
  AsyncDeviceVector<int>       labels(3, stream);

  EXPECT_THROW(nvMolKit::detail::leaderPick(provider, 0.5F, 1, {2, 0}, labels.data(), stream), std::logic_error);
  EXPECT_THROW(nvMolKit::detail::leaderPick(provider, 0.5F, 2, {2, 0}, labels.data(), stream), std::logic_error);
}

TEST(DiversityPickerDise, InitializesSingletonLabelsForBothAssignments) {
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream       = streamOwner.stream();
  auto                         distances    = upload(std::vector<double>{1.0}, stream);
  auto                         fingerprints = upload(std::vector<std::uint32_t>{0U}, stream);

  for (const bool nearest : {false, true}) {
    const auto matrixResult = nvMolKit::diseFromDistanceMatrix(toSpan(distances), 1, 0.0, nearest, stream);
    EXPECT_THAT(matrixResult.clusterIds, ::testing::ElementsAre(0));
    EXPECT_THAT(matrixResult.centroids, ::testing::ElementsAre(0));
    EXPECT_THAT(matrixResult.clusterSizes, ::testing::ElementsAre(1));
    for (const auto metric :
         {nvMolKit::FingerprintSimilarityMetric::Tanimoto, nvMolKit::FingerprintSimilarityMetric::Cosine}) {
      const auto fusedResult = nvMolKit::fusedDiseGpu(toSpan(fingerprints), 1, 1, 0.0, metric, nearest, stream);
      EXPECT_THAT(fusedResult.clusterIds, ::testing::ElementsAre(0));
      EXPECT_THAT(fusedResult.centroids, ::testing::ElementsAre(0));
      EXPECT_THAT(fusedResult.clusterSizes, ::testing::ElementsAre(1));
    }
  }
}

TEST(DiversityPickerLeader, LimitedSelectionKeepsForcedOrderAndContinuesAcrossWindows) {
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream    = streamOwner.stream();
  constexpr int                numItems  = 65;
  auto                         distances = upload(std::vector<float>(numItems * numItems, 1.0F), stream);
  for (const int limit : {1, 2, 31, 32, 33, 64, 65}) {
    std::vector<int> expected{64};
    for (int index = 0; static_cast<int>(expected.size()) < limit; ++index) {
      expected.push_back(index);
    }
    const auto result = nvMolKit::leaderFromDistanceMatrix(toSpan(distances), numItems, 0.5, limit, {64}, stream);
    EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAreArray(expected));
  }
}

TEST(DiversityPickerPrecision, ComparesRoundedDistancesAndCutoffs) {
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream    = streamOwner.stream();
  auto                         distances = upload(std::vector<double>{0.0, 0.50000001, 0.50000001, 0.0}, stream);
  auto                         result    = nvMolKit::leaderFromDistanceMatrix(toSpan(distances), 2, 0.5, 0, {}, stream);
  EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(0));

  auto floatDistances = upload(std::vector<float>{0.0F, 0.5F, 0.5F, 0.0F}, stream);
  result = nvMolKit::leaderFromDistanceMatrix(toSpan(floatDistances), 2, std::nextafter(0.5, 0.0), 0, {}, stream);
  EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(0));
  const float lowerCutoff = std::nextafter(0.5F, 0.0F);
  result                  = nvMolKit::leaderFromDistanceMatrix(toSpan(floatDistances), 2, lowerCutoff, 0, {}, stream);
  EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(0, 1));
}

TEST(DiversityPickerValidation, RejectsCutoffsBeyondFloatRange) {
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream    = streamOwner.stream();
  const double                 maximum   = std::numeric_limits<float>::max();
  auto                         distances = upload(std::vector<double>{0.0, maximum, maximum, 0.0}, stream);
  const auto result = nvMolKit::leaderFromDistanceMatrix(toSpan(distances), 2, maximum, 0, {}, stream);
  EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(0));
  for (const double cutoff : {std::nextafter(maximum, std::numeric_limits<double>::infinity()), 1e40}) {
    EXPECT_THROW(nvMolKit::leaderFromDistanceMatrix(toSpan(distances), 2, cutoff, 0, {}, stream),
                 std::invalid_argument);
    EXPECT_THROW(nvMolKit::diseFromDistanceMatrix(toSpan(distances), 2, cutoff, false, stream), std::invalid_argument);
    EXPECT_THROW(nvMolKit::maxMinFromDistanceMatrix(toSpan(distances), 2, 2, {0}, 7, cutoff, stream),
                 std::invalid_argument);
  }
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

TEST(DiversityPickerMaxMin, HonorsOrderTieBreakAndInclusiveThreshold) {
  nvMolKit::ScopedStream const streamOwner;
  const auto                   stream    = streamOwner.stream();
  const std::vector<double>    distances = {
    0.0,
    0.1,
    0.4,
    0.9,
    0.1,
    0.0,
    0.3,
    0.8,
    0.4,
    0.3,
    0.0,
    0.5,
    0.9,
    0.8,
    0.5,
    0.0,
  };
  auto device = upload(distances, stream);

  auto result = nvMolKit::maxMinFromDistanceMatrix(toSpan(device), 4, 4, {0}, 42, -1.0, stream);
  EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(0, 3, 2, 1));
  EXPECT_FLOAT_EQ(result.lastDistance, 0.1F);

  result = nvMolKit::maxMinFromDistanceMatrix(toSpan(device), 4, 4, {0}, 42, 0.4, stream);
  EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(0, 3));
  EXPECT_FLOAT_EQ(result.lastDistance, 0.9F);

  const std::vector<double> tiedDistances = {
    0.0,
    1.0,
    1.0,
    1.0,
    0.0,
    1.0,
    1.0,
    1.0,
    0.0,
  };
  auto tiedDevice = upload(tiedDistances, stream);
  result          = nvMolKit::maxMinFromDistanceMatrix(toSpan(tiedDevice), 3, 2, {0}, 42, -1.0, stream);
  EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(0, 1));
}

TEST(DiversityPickerDISE, DistinguishesFirstAndNearestAssignmentAndOrdersBySize) {
  nvMolKit::ScopedStream const streamOwner;
  const auto                   stream    = streamOwner.stream();
  const std::vector<double>    distances = {
    0.0,
    0.2,
    0.8,
    0.2,
    0.0,
    0.1,
    0.8,
    0.1,
    0.0,
  };
  auto device = upload(distances, stream);

  const auto first = nvMolKit::diseFromDistanceMatrix(toSpan(device), 3, 0.3, false, stream);
  EXPECT_THAT(first.clusterIds, ::testing::ElementsAre(0, 0, 1));
  EXPECT_THAT(first.centroids, ::testing::ElementsAre(0, 2));
  EXPECT_THAT(first.clusterSizes, ::testing::ElementsAre(2, 1));

  const auto nearest = nvMolKit::diseFromDistanceMatrix(toSpan(device), 3, 0.3, true, stream);
  EXPECT_THAT(nearest.clusterIds, ::testing::ElementsAre(1, 0, 0));
  EXPECT_THAT(nearest.centroids, ::testing::ElementsAre(2, 0));
  EXPECT_THAT(nearest.clusterSizes, ::testing::ElementsAre(2, 1));
}

TEST(DiversityPickerFused, CosineZeroFingerprintIsSelectedOnlyOnce) {
  nvMolKit::ScopedStream const     streamOwner;
  const auto                       stream       = streamOwner.stream();
  const std::vector<std::uint32_t> fingerprints = {0U, 0b0011U, 0b0010U, 0b1100U};
  auto                             device       = upload(fingerprints, stream);

  const auto result =
    nvMolKit::fusedLeaderGpu(toSpan(device), 4, 1, 0.5, FingerprintSimilarityMetric::Cosine, 0, {}, stream);
  EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(0, 1, 3));

  const auto clusters =
    nvMolKit::fusedDiseGpu(toSpan(device), 4, 1, 0.5, FingerprintSimilarityMetric::Cosine, true, stream);
  EXPECT_THAT(clusters.clusterIds, ::testing::ElementsAre(1, 0, 0, 2));
  EXPECT_THAT(clusters.centroids, ::testing::ElementsAre(1, 0, 3));
  EXPECT_THAT(clusters.clusterSizes, ::testing::ElementsAre(2, 1, 1));
}

TEST(DiversityPickerFused, TanimotoMaxMinMatchesKnownSequence) {
  nvMolKit::ScopedStream const     streamOwner;
  const auto                       stream       = streamOwner.stream();
  const std::vector<std::uint32_t> fingerprints = {0b0011U, 0b0010U, 0b1100U, 0b1111U};
  auto                             device       = upload(fingerprints, stream);

  const auto result =
    nvMolKit::fusedMaxMinGpu(toSpan(device), 4, 1, 4, FingerprintSimilarityMetric::Tanimoto, {0}, 42, -1.0, stream);
  EXPECT_THAT(downloadPicks(result, stream), ::testing::ElementsAre(0, 2, 1, 3));
  EXPECT_FLOAT_EQ(result.lastDistance, 0.5F);
}

TEST(DiversityPickerEdges, HandlesEmptyAndSingletonInputs) {
  const nvMolKit::ScopedStream               streamOwner;
  const auto                                 stream = streamOwner.stream();
  const cuda::std::span<const double>        emptyMatrix;
  const cuda::std::span<const std::uint32_t> emptyFingerprints;

  auto picks = nvMolKit::leaderFromDistanceMatrix(emptyMatrix, 0, 0.1, 0, {}, stream);
  EXPECT_TRUE(downloadPicks(picks, stream).empty());
  EXPECT_TRUE(nvMolKit::diseFromDistanceMatrix(emptyMatrix, 0, 0.1, true, stream).clusterIds.empty());
  EXPECT_TRUE(nvMolKit::fusedDiseGpu(emptyFingerprints, 0, 1, 0.1, FingerprintSimilarityMetric::Tanimoto, true, stream)
                .clusterIds.empty());

  auto singleton = upload(std::vector<double>{0.0}, stream);
  picks          = nvMolKit::leaderFromDistanceMatrix(toSpan(singleton), 1, 0.0, 0, {}, stream);
  EXPECT_THAT(downloadPicks(picks, stream), ::testing::ElementsAre(0));
  picks = nvMolKit::maxMinFromDistanceMatrix(toSpan(singleton), 1, 1, {}, 7, -1.0, stream);
  EXPECT_THAT(downloadPicks(picks, stream), ::testing::ElementsAre(0));
  EXPECT_FLOAT_EQ(picks.lastDistance, -1.0F);
}

TEST(DiversityPickerValidation, RejectsMalformedArguments) {
  nvMolKit::ScopedStream const streamOwner;
  const auto                   stream       = streamOwner.stream();
  auto                         matrix       = upload(std::vector<double>{0.0, 1.0, 1.0, 0.0}, stream);
  auto                         fingerprints = upload(std::vector<std::uint32_t>{1U, 2U}, stream);

  EXPECT_THROW(nvMolKit::leaderFromDistanceMatrix(toSpan(matrix), 3, 0.2, 0, {}, stream), std::invalid_argument);
  EXPECT_THROW(
    nvMolKit::leaderFromDistanceMatrix(toSpan(matrix), 2, std::numeric_limits<double>::quiet_NaN(), 0, {}, stream),
    std::invalid_argument);
  EXPECT_THROW(nvMolKit::leaderFromDistanceMatrix(toSpan(matrix), 2, 0.2, 0, {0, 0}, stream), std::invalid_argument);
  EXPECT_THROW(nvMolKit::leaderFromDistanceMatrix(toSpan(matrix), 2, 0.2, 0, {2}, stream), std::invalid_argument);
  EXPECT_THROW(nvMolKit::maxMinFromDistanceMatrix(toSpan(matrix), 2, 0, {}, 7, -1.0, stream), std::invalid_argument);
  EXPECT_THROW(nvMolKit::maxMinFromDistanceMatrix(toSpan(matrix), 2, 1, {}, 7, -0.5, stream), std::invalid_argument);
  EXPECT_THROW(
    nvMolKit::maxMinFromDistanceMatrix(toSpan(matrix), 2, 1, {}, 7, std::numeric_limits<double>::infinity(), stream),
    std::invalid_argument);
  EXPECT_THROW(
    nvMolKit::fusedLeaderGpu(toSpan(fingerprints), 2, 0, 0.2, FingerprintSimilarityMetric::Tanimoto, 0, {}, stream),
    std::invalid_argument);
  EXPECT_THROW(
    nvMolKit::fusedMaxMinGpu(toSpan(fingerprints), 2, 1, 1, FingerprintSimilarityMetric::Cosine, {}, 7, 1.1, stream),
    std::invalid_argument);
}

}  // namespace
