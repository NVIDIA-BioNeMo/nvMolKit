// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "src/butina.h"
#include "src/utils/device.h"
#include "src/utils/host_vector.h"

using nvMolKit::AsyncDeviceVector;

namespace {

std::vector<double> makeSymmetricDifferenceMatrix(const int nPts, std::mt19937& rng) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::vector<double>                    distances(nPts * nPts, 0.0);
  for (int row = 0; row < nPts; ++row) {
    // row + 1 to ignore diagonal.
    for (int col = row + 1; col < nPts; ++col) {
      const double value            = dist(rng);
      distances[(row * nPts) + col] = value;
      distances[(col * nPts) + row] = value;
    }
  }
  return distances;
}

std::vector<uint8_t> makeAdjacency(const std::vector<double>& distances, double cutoff) {
  std::vector<uint8_t> adjacency(distances.size(), 0);
  for (size_t idx = 0; idx < distances.size(); ++idx) {
    adjacency[idx] = distances[idx] <= cutoff ? 1U : 0U;
  }
  return adjacency;
}

std::pair<std::vector<int>, int> runButina(const std::vector<double>& distances,
                                           const int                  nPts,
                                           const double               cutoff,
                                           const int                  neighborlistMaxSize,
                                           cudaStream_t               stream) {
  AsyncDeviceVector<double> distancesDev(distances.size(), stream);
  AsyncDeviceVector<int>    resultDev(nPts, stream);
  distancesDev.copyFromHost(distances);
  const int numClusters =
    nvMolKit::butinaGpu(toSpan(distancesDev), toSpan(resultDev), cutoff, neighborlistMaxSize, {}, true, stream);
  std::vector<int> got(nPts);
  resultDev.copyToHost(got);
  cudaStreamSynchronize(stream);
  return {got, numClusters};
}

std::tuple<std::vector<int>, std::vector<int>, int> runButinaWithCentroids(const std::vector<double>& distances,
                                                                           const int                  nPts,
                                                                           const double               cutoff,
                                                                           const int    neighborlistMaxSize,
                                                                           cudaStream_t stream) {
  AsyncDeviceVector<double> distancesDev(distances.size(), stream);
  AsyncDeviceVector<int>    resultDev(nPts, stream);
  AsyncDeviceVector<int>    centroidsDev(nPts, stream);
  distancesDev.copyFromHost(distances);
  const int        numClusters = nvMolKit::butinaGpu(toSpan(distancesDev),
                                              toSpan(resultDev),
                                              cutoff,
                                              neighborlistMaxSize,
                                              toSpan(centroidsDev),
                                              true,
                                              stream);
  std::vector<int> got(nPts);
  std::vector<int> centroids(nPts);
  resultDev.copyToHost(got);
  centroidsDev.copyToHost(centroids);
  cudaStreamSynchronize(stream);
  return {got, centroids, numClusters};
}

void checkButinaCorrectness(const std::vector<uint8_t>& adjacency, const std::vector<int>& labels) {
  const int nPts = static_cast<int>(labels.size());
  ASSERT_EQ(adjacency.size(), static_cast<size_t>(nPts) * static_cast<size_t>(nPts));

  std::vector<bool> seen(nPts, false);
  int               seenCount = 0;

  const int maxLabelId = *std::ranges::max_element(labels.begin(), labels.end());

  // Build clusters
  std::vector<std::vector<int>> clusters(maxLabelId + 1);
  for (int idx = 0; idx < nPts; ++idx) {
    clusters[labels[idx]].push_back(idx);
  }

  // Verify clusters are ordered by size (descending) - cluster 0 should be largest
  for (size_t i = 1; i < clusters.size(); ++i) {
    ASSERT_GE(clusters[i - 1].size(), clusters[i].size())
      << "Clusters not in descending size order: cluster " << (i - 1) << " has size " << clusters[i - 1].size()
      << " but cluster " << i << " has size " << clusters[i].size();
  }

  for (size_t clustIdx = 0; clustIdx < clusters.size(); ++clustIdx) {
    const auto& cluster     = clusters[clustIdx];
    const auto  clusterSize = static_cast<int>(cluster.size());
    ASSERT_GT(clusterSize, 0) << "Empty cluster found";

    // Verify no point is assigned to multiple clusters
    for (const int member : cluster) {
      ASSERT_FALSE(seen[member]) << "Point " << member << " assigned to multiple clusters";
      seen[member] = true;
    }

    // Verify valid Butina cluster: there exists a centroid that is neighbor of all other members
    bool validCluster = false;
    for (const int centroid : cluster) {
      bool allNeighbors = true;
      for (const int member : cluster) {
        if (member != centroid) {
          const size_t idx = static_cast<size_t>(centroid) * nPts + member;
          if (adjacency[idx] != 1U) {
            allNeighbors = false;
            break;
          }
        }
      }
      if (allNeighbors) {
        validCluster = true;
        break;
      }
    }
    ASSERT_TRUE(validCluster) << "Cluster " << clustIdx << " has no valid centroid";

    seenCount += clusterSize;
  }

  ASSERT_EQ(seenCount, nPts);
}

}  // namespace

class ButinaSinglePointFixture : public ::testing::TestWithParam<int> {};
TEST_P(ButinaSinglePointFixture, HandlesSinglePoint) {
  constexpr int                nPts                = 1;
  constexpr double             cutoff              = 0.2;
  const int                    neighborlistMaxSize = GetParam();
  nvMolKit::ScopedStream const scopedStream;
  cudaStream_t                 stream = scopedStream.stream();

  AsyncDeviceVector<double> distancesDev(nPts * nPts, stream);
  AsyncDeviceVector<int>    resultDev(nPts, stream);
  distancesDev.copyFromHost(std::vector<double>{0.0});

  const int numClusters =
    nvMolKit::butinaGpu(toSpan(distancesDev), toSpan(resultDev), cutoff, neighborlistMaxSize, {}, true, stream);
  std::vector<int> got(nPts);
  resultDev.copyToHost(got);
  cudaStreamSynchronize(stream);
  EXPECT_THAT(got, ::testing::ElementsAre(0));
  EXPECT_EQ(numClusters, 1);
}
INSTANTIATE_TEST_SUITE_P(ButinaClusterTest, ButinaSinglePointFixture, ::testing::Values(8, 16, 24, 32, 64, 128));

class ButinaClusterTestFixture : public ::testing::TestWithParam<std::tuple<int, int>> {};
TEST_P(ButinaClusterTestFixture, ClusteringMatchesReference) {
  nvMolKit::ScopedStream const scopedStream;
  cudaStream_t                 stream = scopedStream.stream();
  std::mt19937                 rng(42);

  const auto [nPts, neighborlistMaxSize] = GetParam();
  constexpr double cutoff                = 0.1;
  const auto       distances             = makeSymmetricDifferenceMatrix(nPts, rng);
  const auto       adjacency             = makeAdjacency(distances, cutoff);
  const auto [labels, numClusters]       = runButina(distances, nPts, cutoff, neighborlistMaxSize, stream);
  SCOPED_TRACE(::testing::Message() << "nPts=" << nPts << " neighborlistMaxSize=" << neighborlistMaxSize);
  checkButinaCorrectness(adjacency, labels);
  EXPECT_EQ(numClusters, *std::ranges::max_element(labels.begin(), labels.end()) + 1);
}

INSTANTIATE_TEST_SUITE_P(ButinaClusterTest,
                         ButinaClusterTestFixture,
                         ::testing::Combine(::testing::Values(1, 10, 100, 1000),
                                            ::testing::Values(8, 16, 24, 32, 64, 128)));

class ButinaEdgeTestFixture : public ::testing::TestWithParam<int> {};
TEST_P(ButinaEdgeTestFixture, EdgeOneCluster) {
  constexpr int                nPts                = 10;
  constexpr double             cutoff              = 100.0;
  const int                    neighborlistMaxSize = GetParam();
  nvMolKit::ScopedStream const scopedStream;
  cudaStream_t                 stream = scopedStream.stream();

  std::vector<double> distances(static_cast<size_t>(nPts) * nPts, 0.5);
  for (int i = 0; i < nPts; ++i) {
    distances[static_cast<size_t>(i) * nPts + i] = 0.0;
  }

  const auto [labels, numClusters] = runButina(distances, nPts, cutoff, neighborlistMaxSize, stream);
  EXPECT_THAT(labels, ::testing::Each(0));
  EXPECT_EQ(numClusters, 1);
}

TEST_P(ButinaEdgeTestFixture, EdgeNClusters) {
  constexpr int                nPts                = 10;
  constexpr double             cutoff              = 1e-8;
  const int                    neighborlistMaxSize = GetParam();
  nvMolKit::ScopedStream const scopedStream;
  cudaStream_t                 stream = scopedStream.stream();

  std::vector<double> distances(static_cast<size_t>(nPts) * nPts, 1.0);
  for (int i = 0; i < nPts; ++i) {
    distances[static_cast<size_t>(i) * nPts + i] = 0.0;
  }

  const auto [labels, numClusters] = runButina(distances, nPts, cutoff, neighborlistMaxSize, stream);
  std::vector<int> sorted          = labels;
  std::ranges::sort(sorted);
  std::vector<int> want(nPts);
  std::iota(want.begin(), want.end(), 0);
  EXPECT_THAT(sorted, ::testing::ElementsAreArray(want));
  EXPECT_EQ(numClusters, nPts);
}

INSTANTIATE_TEST_SUITE_P(ButinaClusterEdgeTest, ButinaEdgeTestFixture, ::testing::Values(8, 16, 24, 32, 64, 128));

TEST(ButinaCentroids, ReturnCentroids) {
  constexpr int                nPts                = 10;
  constexpr double             cutoff              = 0.1;
  constexpr int                neighborlistMaxSize = 64;
  nvMolKit::ScopedStream const scopedStream;
  cudaStream_t                 stream = scopedStream.stream();

  const std::vector<double> distances = {
    0.0,  0.05, 0.05, 0.05, 1.0,  1.0,  1.0,  1.0, 1.0, 1.0, 0.05, 0.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0, 1.0,
    0.05, 1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0, 1.0, 0.05, 1.0, 1.0, 0.0, 1.0,  1.0, 1.0, 1.0, 1.0, 1.0,
    1.0,  1.0,  1.0,  1.0,  0.0,  0.05, 0.05, 1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 0.05, 0.0, 1.0, 1.0, 1.0, 1.0,
    1.0,  1.0,  1.0,  1.0,  0.05, 1.0,  0.0,  1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 0.0, 1.0, 1.0,
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 0.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0, 0.0,
  };

  const auto [labels, centroids, numClusters] =
    runButinaWithCentroids(distances, nPts, cutoff, neighborlistMaxSize, stream);
  ASSERT_EQ(numClusters, 5);

  std::vector<std::vector<int>> clusters(numClusters);
  for (int idx = 0; idx < nPts; ++idx) {
    clusters[labels[idx]].push_back(idx);
  }

  EXPECT_THAT(clusters[0], ::testing::UnorderedElementsAre(0, 1, 2, 3));
  EXPECT_EQ(centroids[0], 0);
  EXPECT_THAT(clusters[1], ::testing::UnorderedElementsAre(4, 5, 6));
  EXPECT_EQ(centroids[1], 4);

  for (int clusterId = 2; clusterId < numClusters; ++clusterId) {
    ASSERT_EQ(clusters[clusterId].size(), 1U);
    EXPECT_EQ(centroids[clusterId], clusters[clusterId][0]);
  }
}
