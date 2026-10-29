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
#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include "src/fingerprint_search.h"
#include "src/utils/device.h"

using nvMolKit::AsyncDeviceVector;
using nvMolKit::FingerprintSearchIndex;
using nvMolKit::FingerprintSimilarityMetric;

namespace {

struct Hits {
  std::vector<double>       scores;
  std::vector<std::int64_t> indices;
};

// Random fingerprints whose densities vary per row. Low densities and repeated rows produce many tied scores.
std::vector<std::uint32_t> makeFingerprints(const int numRows, const int numWords, std::mt19937& rng) {
  std::vector<std::uint32_t>             bits(static_cast<std::size_t>(numRows) * numWords, 0);
  std::uniform_real_distribution<double> densityDist(0.0, 0.15);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int row = 0; row < numRows; ++row) {
    if (row > 0 && unit(rng) < 0.05) {
      std::copy_n(bits.begin() + static_cast<std::ptrdiff_t>(row - 1) * numWords,
                  numWords,
                  bits.begin() + static_cast<std::ptrdiff_t>(row) * numWords);
      continue;
    }
    const double density = row % 37 == 0 ? 0.0 : densityDist(rng);
    for (int bit = 0; bit < numWords * 32; ++bit) {
      if (unit(rng) < density) {
        bits[static_cast<std::size_t>(row) * numWords + bit / 32] |= 1U << (bit % 32);
      }
    }
  }
  return bits;
}

double similarity(const std::uint32_t*        lhs,
                  const std::uint32_t*        rhs,
                  const int                   numWords,
                  FingerprintSimilarityMetric metric) {
  int lhsCount     = 0;
  int rhsCount     = 0;
  int intersection = 0;
  for (int word = 0; word < numWords; ++word) {
    lhsCount += std::popcount(lhs[word]);
    rhsCount += std::popcount(rhs[word]);
    intersection += std::popcount(lhs[word] & rhs[word]);
  }
  if (metric == FingerprintSimilarityMetric::Tanimoto) {
    const double denominator = static_cast<double>(lhsCount + rhsCount - intersection);
    return denominator > 0.0 ? intersection / denominator : 1.0;
  }
  const double denominator = std::sqrt(static_cast<double>(lhsCount) * static_cast<double>(rhsCount));
  return denominator > 0.0 ? intersection / denominator : 0.0;
}

Hits bruteForce(const std::vector<std::uint32_t>& queries,
                const std::vector<std::uint32_t>& library,
                const int                         numWords,
                const int                         k,
                const double                      threshold,
                const bool                        selfSearch,
                FingerprintSimilarityMetric       metric) {
  const int numQueries = static_cast<int>(queries.size()) / numWords;
  const int numLibrary = static_cast<int>(library.size()) / numWords;
  Hits      hits{std::vector<double>(static_cast<std::size_t>(numQueries) * k, std::nan("")),
            std::vector<std::int64_t>(static_cast<std::size_t>(numQueries) * k, -1)};
  for (int query = 0; query < numQueries; ++query) {
    std::vector<std::pair<double, int>> row;
    for (int item = 0; item < numLibrary; ++item) {
      if (selfSearch && item == query) {
        continue;
      }
      const double score = similarity(queries.data() + static_cast<std::size_t>(query) * numWords,
                                      library.data() + static_cast<std::size_t>(item) * numWords,
                                      numWords,
                                      metric);
      if (score >= threshold) {
        row.emplace_back(score, item);
      }
    }
    std::sort(row.begin(), row.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
    });
    for (int slot = 0; slot < std::min<int>(k, static_cast<int>(row.size())); ++slot) {
      hits.scores[static_cast<std::size_t>(query) * k + slot]  = row[slot].first;
      hits.indices[static_cast<std::size_t>(query) * k + slot] = row[slot].second;
    }
  }
  return hits;
}

// Release builds use -ffast-math, under which std::isnan may be folded to false; check the bit pattern instead.
bool isNan(const double value) {
  const auto bits = std::bit_cast<std::uint64_t>(value);
  return (bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL && (bits & 0x000fffffffffffffULL) != 0;
}

Hits copyHits(nvMolKit::FingerprintSearchResult& result, cudaStream_t stream) {
  Hits hits{std::vector<double>(result.scores.size()), std::vector<std::int64_t>(result.indices.size())};
  result.scores.copyToHost(hits.scores);
  result.indices.copyToHost(hits.indices);
  cudaStreamSynchronize(stream);
  return hits;
}

void expectSameHits(const Hits& got, const Hits& want) {
  ASSERT_EQ(got.indices.size(), want.indices.size());
  for (std::size_t i = 0; i < want.indices.size(); ++i) {
    ASSERT_EQ(got.indices[i], want.indices[i]) << "slot " << i;
    if (want.indices[i] < 0) {
      ASSERT_TRUE(isNan(got.scores[i])) << "slot " << i;
    } else {
      ASSERT_EQ(got.scores[i], want.scores[i]) << "slot " << i;
    }
  }
}

using SearchCase = std::tuple<FingerprintSimilarityMetric, int, int, int, double, bool>;

class FingerprintSearchTest : public ::testing::TestWithParam<SearchCase> {};

TEST_P(FingerprintSearchTest, MatchesBruteForce) {
  const auto [metric, numWords, numLibrary, k, threshold, prune] = GetParam();
  std::mt19937                     rng(1234 + numWords + numLibrary + k);
  const auto                       library = makeFingerprints(numLibrary, numWords, rng);
  const auto                       queries = makeFingerprints(45, numWords, rng);
  const nvMolKit::ScopedStream     stream;
  AsyncDeviceVector<std::uint32_t> libraryDev(library.size(), stream.stream());
  AsyncDeviceVector<std::uint32_t> queriesDev(queries.size(), stream.stream());
  libraryDev.copyFromHost(library);
  queriesDev.copyFromHost(queries);

  const FingerprintSearchIndex index(toSpan(libraryDev), numLibrary, numWords, metric, stream.stream());
  auto                         result = index.search(toSpan(queriesDev), 45, k, threshold, prune, stream.stream());
  expectSameHits(copyHits(result, stream.stream()),
                 bruteForce(queries, library, numWords, k, threshold, false, metric));

  auto selfResult = index.searchSelf(k, threshold, prune, stream.stream());
  expectSameHits(copyHits(selfResult, stream.stream()),
                 bruteForce(library, library, numWords, k, threshold, true, metric));
}

INSTANTIATE_TEST_SUITE_P(
  Cases,
  FingerprintSearchTest,
  ::testing::Values(
    // Word counts below, at and above the 32-word chunk; libraries smaller and larger than a tile.
    SearchCase{FingerprintSimilarityMetric::Tanimoto, 2, 50, 5, 0.0, true},
    SearchCase{FingerprintSimilarityMetric::Tanimoto, 32, 700, 10, 0.0, true},
    SearchCase{FingerprintSimilarityMetric::Tanimoto, 64, 1500, 1, 0.0, true},
    SearchCase{FingerprintSimilarityMetric::Tanimoto, 64, 1500, 100, 0.0, true},
    SearchCase{FingerprintSimilarityMetric::Tanimoto, 64, 1500, 100, 0.0, false},
    SearchCase{FingerprintSimilarityMetric::Tanimoto, 64, 1500, 300, 0.2, true},
    SearchCase{FingerprintSimilarityMetric::Tanimoto, 33, 900, 2000, 0.0, true},
    SearchCase{FingerprintSimilarityMetric::Tanimoto, 64, 1500, 20, 0.9, true},
    SearchCase{FingerprintSimilarityMetric::Cosine, 64, 1500, 50, 0.0, true},
    SearchCase{FingerprintSimilarityMetric::Cosine, 16, 800, 30, 0.4, true},
    SearchCase{FingerprintSimilarityMetric::Cosine, 16, 800, 30, 0.4, false}));

TEST(FingerprintSearch, EmptyLibraryPadsEveryRow) {
  const nvMolKit::ScopedStream     stream;
  AsyncDeviceVector<std::uint32_t> libraryDev(0, stream.stream());
  AsyncDeviceVector<std::uint32_t> queriesDev(4 * 2, stream.stream());
  queriesDev.zero();
  const FingerprintSearchIndex index(toSpan(libraryDev), 0, 2, FingerprintSimilarityMetric::Tanimoto, stream.stream());
  auto                         result = index.search(toSpan(queriesDev), 4, 3, 0.0, true, stream.stream());
  const Hits                   hits   = copyHits(result, stream.stream());
  EXPECT_TRUE(std::all_of(hits.indices.begin(), hits.indices.end(), [](std::int64_t index) { return index == -1; }));
  EXPECT_TRUE(std::all_of(hits.scores.begin(), hits.scores.end(), [](double score) { return isNan(score); }));
}

TEST(FingerprintSearch, RejectsInvalidArguments) {
  const nvMolKit::ScopedStream     stream;
  AsyncDeviceVector<std::uint32_t> libraryDev(4, stream.stream());
  libraryDev.zero();
  const FingerprintSearchIndex index(toSpan(libraryDev), 2, 2, FingerprintSimilarityMetric::Tanimoto, stream.stream());
  EXPECT_THROW((void)index.searchSelf(0, 0.0, true, stream.stream()), std::invalid_argument);
  EXPECT_THROW((void)index.searchSelf(1, 1.5, true, stream.stream()), std::invalid_argument);
  EXPECT_THROW((void)index.search(toSpan(libraryDev), 3, 1, 0.0, true, stream.stream()), std::invalid_argument);
  EXPECT_THROW(FingerprintSearchIndex(toSpan(libraryDev), 3, 2, FingerprintSimilarityMetric::Tanimoto, stream.stream()),
               std::invalid_argument);
}

}  // namespace
