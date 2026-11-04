// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "src/kmeans.h"
#include "src/kmeans_kernels.cuh"
#include "src/utils/device.h"

namespace {

using nvMolKit::AsyncDeviceVector;
using nvMolKit::KMeansMetric;

template <typename T> AsyncDeviceVector<T> upload(const std::vector<T>& host, cudaStream_t stream) {
  AsyncDeviceVector<T> device(host.size(), stream);
  device.copyFromHost(host);
  return device;
}

template <typename T> std::vector<T> download(const AsyncDeviceVector<T>& device, cudaStream_t stream) {
  std::vector<T> host(device.size());
  device.copyToHost(host);
  cudaStreamSynchronize(stream);
  return host;
}

std::vector<std::uint32_t> randomFingerprints(const int rows, const int words, const double density, const int seed) {
  std::mt19937                generator(seed);
  std::bernoulli_distribution bit(density);
  std::vector<std::uint32_t>  packed(static_cast<std::size_t>(rows) * words, 0U);
  for (int row = 2; row < rows; ++row) {  // rows 0 and 1 stay empty
    for (int index = 0; index < words * 32; ++index) {
      if (bit(generator)) {
        packed[static_cast<std::size_t>(row) * words + index / 32] |= 1U << (index % 32);
      }
    }
  }
  return packed;
}

bool testBit(const std::vector<std::uint32_t>& packed, const int words, const int row, const int bit) {
  return (packed[static_cast<std::size_t>(row) * words + bit / 32] >> (bit % 32) & 1U) != 0U;
}

//! Reference score (larger is better) in double precision with fp16-rounded centers.
double referenceScore(const std::vector<std::uint32_t>& packed,
                      const std::vector<float>&         centers,
                      const int                         words,
                      const int                         row,
                      const int                         center,
                      const KMeansMetric                metric) {
  const int bits  = words * 32;
  double    dot   = 0.0;
  double    norm  = 0.0;
  int       count = 0;
  for (int bit = 0; bit < bits; ++bit) {
    const double value = __half2float(__float2half_rn(centers[static_cast<std::size_t>(center) * bits + bit]));
    norm += value * value;
    if (testBit(packed, words, row, bit)) {
      dot += value;
      ++count;
    }
  }
  switch (metric) {
    case KMeansMetric::Cosine:
      return (count > 0 ? 2.0 * dot / std::sqrt(static_cast<double>(count)) : 0.0) - norm;
    case KMeansMetric::Tanimoto: {
      const double denominator = count + norm - dot;
      return denominator > 0.0 ? dot / denominator : 1.0;
    }
  }
  return 0.0;
}

//! Reference distance for a reference score: squared Euclidean distance between normalized vectors for cosine (an
//! empty row is the zero vector), 1 - Tanimoto otherwise.
double referenceDistance(const double score, const int count, const KMeansMetric metric) {
  if (metric == KMeansMetric::Cosine) {
    return std::max((count > 0 ? 1.0 : 0.0) - score, 0.0);
  }
  return 1.0 - score;
}

int rowBitCount(const std::vector<std::uint32_t>& packed, const int words, const int row) {
  int count = 0;
  for (int word = 0; word < words; ++word) {
    count += __builtin_popcount(packed[static_cast<std::size_t>(row) * words + word]);
  }
  return count;
}

/**
 * Runs one assignment pass and checks every row against the double-precision reference: the chosen center's score is
 * within float rounding of the best score, and the returned distance matches the chosen center's reference distance.
 * Returns the labels.
 */
std::vector<int> assignAndCheck(const std::vector<std::uint32_t>& packed,
                                const std::vector<float>&         centers,
                                const int                         rows,
                                const int                         words,
                                const int                         clusters,
                                const KMeansMetric                metric,
                                cudaStream_t                      stream) {
  auto                     devicePacked  = upload(packed, stream);
  auto                     deviceCenters = upload(centers, stream);
  AsyncDeviceVector<int>   labels(rows, stream);
  AsyncDeviceVector<float> distances(rows, stream);
  nvMolKit::kmeansAssignGpu({devicePacked.data(), devicePacked.size()},
                            rows,
                            words,
                            deviceCenters.data(),
                            clusters,
                            metric,
                            labels.data(),
                            distances.data(),
                            stream);
  const auto hostLabels    = download(labels, stream);
  const auto hostDistances = download(distances, stream);
  for (int row = 0; row < rows; ++row) {
    double best = -INFINITY;
    for (int center = 0; center < clusters; ++center) {
      best = std::max(best, referenceScore(packed, centers, words, row, center, metric));
    }
    EXPECT_GE(hostLabels[row], 0) << "row " << row;
    EXPECT_LT(hostLabels[row], clusters) << "row " << row;
    if (hostLabels[row] < 0 || hostLabels[row] >= clusters) {
      continue;
    }
    const double chosen = referenceScore(packed, centers, words, row, hostLabels[row], metric);
    EXPECT_LE(best - chosen, 1e-5 * std::max(1.0, std::abs(best))) << "row " << row;
    const double distance = referenceDistance(chosen, rowBitCount(packed, words, row), metric);
    EXPECT_NEAR(hostDistances[row], distance, 1e-5 * std::max(1.0, std::abs(distance)) + 1e-4) << "row " << row;
  }
  return hostLabels;
}

//! Sets center to row's bits, L2-normalized for cosine: the row's unique best center under either metric.
void plantRow(std::vector<float>&               centers,
              const std::vector<std::uint32_t>& packed,
              const int                         words,
              const int                         center,
              const int                         row,
              const KMeansMetric                metric) {
  const int   bits = words * 32;
  const float value =
    metric == KMeansMetric::Cosine ? 1.0F / std::sqrt(static_cast<float>(rowBitCount(packed, words, row))) : 1.0F;
  for (int bit = 0; bit < bits; ++bit) {
    centers[static_cast<std::size_t>(center) * bits + bit] = testBit(packed, words, row, bit) ? value : 0.0F;
  }
}

class KMeansAssignTest : public ::testing::TestWithParam<std::tuple<KMeansMetric, int, int, int>> {};

TEST_P(KMeansAssignTest, MatchesReference) {
  const auto [metric, rows, words, clusters] = GetParam();
  if (!nvMolKit::detail::mmaSupported(64)) {
    GTEST_SKIP() << "k-means needs an sm_80 or newer GPU";
  }
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream = streamOwner.stream();
  const auto                   packed = randomFingerprints(rows, words, 0.06, rows + words);

  std::mt19937                          generator(7);
  std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
  std::vector<float>                    centers(static_cast<std::size_t>(clusters) * words * 32);
  for (auto& value : centers) {
    value = std::pow(uniform(generator), 3.0F);
  }

  // Beyond one 256-center tile, plant row copies in later tiles so those rows have a known winner there. Planted
  // rows are non-empty and pairwise distinct, so each planted center is the unique best for its row.
  std::vector<std::pair<int, int>> planted;  // (row, center)
  if (clusters > 256) {
    // Distinct centers in later tiles: the first center of the second tile, the last center, and one in between.
    std::vector<int> targets = {256};
    if (clusters - 1 > 256) {
      targets.push_back(clusters - 1);
    }
    if (clusters - 1 > 300) {
      targets.push_back(300);
    }
    std::vector<int> rowsUsed;
    for (int row = 0; row < rows && rowsUsed.size() < targets.size(); ++row) {
      const auto begin    = packed.begin() + static_cast<std::ptrdiff_t>(row) * words;
      const bool distinct = std::none_of(rowsUsed.begin(), rowsUsed.end(), [&](const int other) {
        return std::equal(begin, begin + words, packed.begin() + static_cast<std::ptrdiff_t>(other) * words);
      });
      if (rowBitCount(packed, words, row) > 0 && distinct) {
        rowsUsed.push_back(row);
      }
    }
    for (std::size_t index = 0; index < rowsUsed.size(); ++index) {
      planted.emplace_back(rowsUsed[index], targets[index]);
    }
  }
  for (const auto& [row, center] : planted) {
    plantRow(centers, packed, words, center, row, metric);
  }
  const auto labels = assignAndCheck(packed, centers, rows, words, clusters, metric, stream);
  for (const auto& [row, center] : planted) {
    EXPECT_EQ(labels[row], center) << "row " << row;
  }
}

INSTANTIATE_TEST_SUITE_P(KMeans,
                         KMeansAssignTest,
                         ::testing::Combine(::testing::Values(KMeansMetric::Cosine, KMeansMetric::Tanimoto),
                                            ::testing::Values(130),
                                            ::testing::Values(1, 9, 32, 67),
                                            ::testing::Values(1, 5, 130, 257, 513)));

class KMeansAssignTieTest : public ::testing::TestWithParam<KMeansMetric> {};

// Center i is a copy of distinct center (i % 64) / 4, and distinct center 0 is all zeros. Each distinct center so
// appears at offsets that are resolved by every stage of the kernel's argmax: +1 (same thread), +2 (another lane of the
// quad), +64 (another warp) and +256 (the next center tile, same thread). Identical centers score identically, so every
// row must take the lowest copy, an index below 64 that is a multiple of 4; the two empty rows score best against the
// zero center (Tanimoto 1 by convention, cosine 0 against negative scores) and must take center 0.
TEST_P(KMeansAssignTieTest, LowestIndexWinsTies) {
  const KMeansMetric metric = GetParam();
  if (!nvMolKit::detail::mmaSupported(64)) {
    GTEST_SKIP() << "k-means needs an sm_80 or newer GPU";
  }
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream   = streamOwner.stream();
  constexpr int                rows     = 130;
  constexpr int                words    = 9;
  constexpr int                distinct = 16;
  constexpr int                clusters = 512;
  constexpr int                bits     = words * 32;
  const auto                   packed   = randomFingerprints(rows, words, 0.06, 11);

  std::mt19937                          generator(3);
  std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
  std::vector<float>                    unique(static_cast<std::size_t>(distinct) * bits, 0.0F);
  for (std::size_t index = bits; index < unique.size(); ++index) {
    unique[index] = std::pow(uniform(generator), 3.0F);
  }
  std::vector<float> centers(static_cast<std::size_t>(clusters) * bits);
  for (int center = 0; center < clusters; ++center) {
    const int source = (center % 64) / 4;
    std::copy(unique.begin() + static_cast<std::ptrdiff_t>(source) * bits,
              unique.begin() + static_cast<std::ptrdiff_t>(source + 1) * bits,
              centers.begin() + static_cast<std::ptrdiff_t>(center) * bits);
  }

  const auto labels = assignAndCheck(packed, centers, rows, words, clusters, metric, stream);
  EXPECT_EQ(labels[0], 0);
  EXPECT_EQ(labels[1], 0);
  for (int row = 0; row < rows; ++row) {
    EXPECT_TRUE(labels[row] < 64 && labels[row] % 4 == 0)
      << "row " << row << " took center " << labels[row] << " instead of the lowest copy";
  }
}

INSTANTIATE_TEST_SUITE_P(KMeans, KMeansAssignTieTest, ::testing::Values(KMeansMetric::Cosine, KMeansMetric::Tanimoto));

// A build without sm_80+ code for this device (for example CMAKE_CUDA_ARCHITECTURES=75 on an sm_80+ GPU) has only the
// kernel stub; assignment must report that instead of launching it.
TEST(KMeansAssign, ThrowsWithoutUsableKernel) {
  if (nvMolKit::detail::mmaSupported(64)) {
    GTEST_SKIP() << "this build has a usable k-means kernel for this GPU";
  }
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream  = streamOwner.stream();
  auto                         device  = upload(randomFingerprints(10, 2, 0.2, 1), stream);
  auto                         centers = upload(std::vector<float>(2 * 64, 0.5F), stream);
  AsyncDeviceVector<int>       labels(10, stream);
  AsyncDeviceVector<float>     distances(10, stream);
  EXPECT_THROW(nvMolKit::kmeansAssignGpu({device.data(), device.size()},
                                         10,
                                         2,
                                         centers.data(),
                                         2,
                                         KMeansMetric::Tanimoto,
                                         labels.data(),
                                         distances.data(),
                                         stream),
               std::invalid_argument);
}

TEST(KMeansAssign, RejectsInvalidArguments) {
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream  = streamOwner.stream();
  auto                         device  = upload(randomFingerprints(10, 2, 0.2, 1), stream);
  auto                         centers = upload(std::vector<float>(2 * 64, 0.5F), stream);
  AsyncDeviceVector<int>       labels(10, stream);
  AsyncDeviceVector<float>     distances(10, stream);
  // Buffer size does not match 10 x 3 words.
  EXPECT_THROW(nvMolKit::kmeansAssignGpu({device.data(), device.size()},
                                         10,
                                         3,
                                         centers.data(),
                                         2,
                                         KMeansMetric::Tanimoto,
                                         labels.data(),
                                         distances.data(),
                                         stream),
               std::invalid_argument);
  EXPECT_THROW(nvMolKit::kmeansAssignGpu({device.data(), device.size()},
                                         10,
                                         2,
                                         centers.data(),
                                         0,
                                         KMeansMetric::Tanimoto,
                                         labels.data(),
                                         distances.data(),
                                         stream),
               std::invalid_argument);
}

TEST(KMeansFit, LabelsSizesAndInertiaAreConsistent) {
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream   = streamOwner.stream();
  constexpr int                rows     = 400;
  constexpr int                words    = 4;
  constexpr int                clusters = 12;
  const auto                   packed   = randomFingerprints(rows, words, 0.15, 3);
  auto                         device   = upload(packed, stream);

  auto       initRows = nvMolKit::kmeansInitGpu({device.data(), device.size()},
                                          rows,
                                          words,
                                          clusters,
                                          KMeansMetric::Tanimoto,
                                          nvMolKit::KMeansInit::MaxMin,
                                          5,
                                          stream);
  const auto picks    = download(initRows, stream);
  EXPECT_EQ(std::set<int>(picks.begin(), picks.end()).size(), static_cast<std::size_t>(clusters));
  auto centers =
    nvMolKit::kmeansCentersFromRows({device.data(), device.size()}, words, initRows, KMeansMetric::Tanimoto, stream);

  nvMolKit::KMeansOptions options;
  options.numClusters = clusters;
  options.metric      = KMeansMetric::Tanimoto;
  auto result = nvMolKit::kmeansFitGpu({device.data(), device.size()}, rows, words, centers.data(), options, stream);
  const auto labels = download(result.labels, stream);
  const auto fitted = download(result.centers, stream);
  const auto sizes  = download(result.clusterSizes, stream);
  EXPECT_GE(result.numIterations, 1);

  std::vector<std::int64_t> counted(clusters, 0);
  for (int row = 0; row < rows; ++row) {
    ++counted[labels[row]];
  }
  EXPECT_EQ(counted, sizes);
  double inertia = 0.0;
  for (int row = 0; row < rows; ++row) {
    double best = -INFINITY;
    for (int center = 0; center < clusters; ++center) {
      best = std::max(best, referenceScore(packed, fitted, words, row, center, KMeansMetric::Tanimoto));
    }
    const double chosen = referenceScore(packed, fitted, words, row, labels[row], KMeansMetric::Tanimoto);
    EXPECT_LE(best - chosen, 1e-5);
    inertia += 1.0 - chosen;
  }
  EXPECT_NEAR(result.inertia, inertia, 1e-3);
}

class KMeansReseedTest : public ::testing::TestWithParam<KMeansMetric> {};

// Four planted groups and initial centers [p0, p0, p0, p1]: the two duplicate centers lose every member, and the two
// reseeds must land in different far groups (groups 2 and 3) for the fit to recover the planted partition.
TEST_P(KMeansReseedTest, SpreadsReseedsAcrossFarGroups) {
  const KMeansMetric            metric = GetParam();
  const nvMolKit::ScopedStream  streamOwner;
  const auto                    stream  = streamOwner.stream();
  constexpr int                 words   = 64;
  constexpr int                 groups  = 4;
  constexpr int                 members = 25;
  constexpr int                 rows    = groups * members;
  constexpr int                 setBits = 60;
  constexpr int                 flipped = 4;
  std::mt19937                  generator(0);
  std::vector<std::vector<int>> prototypes(groups);
  for (auto& prototype : prototypes) {
    std::vector<int> bits(words * 32);
    std::iota(bits.begin(), bits.end(), 0);
    std::shuffle(bits.begin(), bits.end(), generator);
    prototype.assign(bits.begin(), bits.begin() + setBits);
  }
  std::vector<std::uint32_t>         packed(static_cast<std::size_t>(rows) * words, 0U);
  std::vector<int>                   truth(rows);
  std::uniform_int_distribution<int> anyBit(0, words * 32 - 1);
  for (int row = 0; row < rows; ++row) {
    truth[row] = row % groups;
    for (const int bit : prototypes[truth[row]]) {
      packed[static_cast<std::size_t>(row) * words + bit / 32] |= 1U << (bit % 32);
    }
    for (int flip = 0; flip < flipped; ++flip) {
      const int bit = anyBit(generator);
      packed[static_cast<std::size_t>(row) * words + bit / 32] ^= 1U << (bit % 32);
    }
  }
  std::vector<float> initial(static_cast<std::size_t>(groups) * words * 32, 0.0F);
  const int          initialGroups[groups] = {0, 0, 0, 1};
  for (int center = 0; center < groups; ++center) {
    const auto& prototype = prototypes[initialGroups[center]];
    for (const int bit : prototype) {
      // Cosine centers are L2-normalized.
      initial[static_cast<std::size_t>(center) * words * 32 + bit] =
        metric == KMeansMetric::Cosine ? 1.0F / std::sqrt(static_cast<float>(setBits)) : 1.0F;
    }
  }
  auto                    device  = upload(packed, stream);
  auto                    centers = upload(initial, stream);
  nvMolKit::KMeansOptions options;
  options.numClusters = groups;
  options.metric      = metric;
  auto result = nvMolKit::kmeansFitGpu({device.data(), device.size()}, rows, words, centers.data(), options, stream);
  const auto labels = download(result.labels, stream);
  EXPECT_GE(result.reseeded, 2);
  std::set<std::pair<int, int>> pairs;
  for (int row = 0; row < rows; ++row) {
    pairs.emplace(labels[row], truth[row]);
  }
  EXPECT_EQ(pairs.size(), static_cast<std::size_t>(groups)) << "a planted group was split or merged";
}

INSTANTIATE_TEST_SUITE_P(KMeans, KMeansReseedTest, ::testing::Values(KMeansMetric::Cosine, KMeansMetric::Tanimoto));

//! Packs rows given as lists of set bits.
std::vector<std::uint32_t> packRows(const std::vector<std::vector<int>>& rows, const int words) {
  std::vector<std::uint32_t> packed(rows.size() * words, 0U);
  for (std::size_t row = 0; row < rows.size(); ++row) {
    for (const int bit : rows[row]) {
      packed[row * words + bit / 32] |= 1U << (bit % 32);
    }
  }
  return packed;
}

std::vector<int> bitRange(const int first, const int last) {
  std::vector<int> bits(last - first + 1);
  std::iota(bits.begin(), bits.end(), first);
  return bits;
}

struct HostFit {
  std::vector<int>          labels;
  std::vector<float>        centers;
  std::vector<std::int64_t> sizes;
  double                    inertia    = 0.0;
  int                       iterations = 0;
  std::int64_t              reseeded   = 0;
};

HostFit fitFromCenters(const std::vector<std::uint32_t>& packed,
                       const int                         words,
                       const std::vector<float>&         initialCenters,
                       nvMolKit::KMeansOptions           options,
                       cudaStream_t                      stream) {
  const int rows      = static_cast<int>(packed.size() / words);
  options.numClusters = static_cast<int>(initialCenters.size() / (words * 32));
  auto device         = upload(packed, stream);
  auto centers        = upload(initialCenters, stream);
  auto result = nvMolKit::kmeansFitGpu({device.data(), device.size()}, rows, words, centers.data(), options, stream);
  return {download(result.labels, stream),
          download(result.centers, stream),
          download(result.clusterSizes, stream),
          result.inertia,
          result.numIterations,
          result.reseeded};
}

//! Initial centers taken from the given rows (normalized for cosine).
std::vector<float> centersFromRows(const std::vector<std::uint32_t>& packed,
                                   const int                         words,
                                   const std::vector<int>&           rows,
                                   const KMeansMetric                metric,
                                   cudaStream_t                      stream) {
  auto device     = upload(packed, stream);
  auto deviceRows = upload(rows, stream);
  return download(nvMolKit::kmeansCentersFromRows({device.data(), device.size()}, words, deviceRows, metric, stream),
                  stream);
}

std::vector<int> maxMinRows(const std::vector<std::uint32_t>& packed,
                            const int                         words,
                            const int                         clusters,
                            const KMeansMetric                metric,
                            cudaStream_t                      stream) {
  auto device = upload(packed, stream);
  return download(nvMolKit::kmeansInitGpu({device.data(), device.size()},
                                          static_cast<int>(packed.size() / words),
                                          words,
                                          clusters,
                                          metric,
                                          nvMolKit::KMeansInit::MaxMin,
                                          0,
                                          stream),
                  stream);
}

//! True when labels and truth define the same partition (equal up to renaming the clusters).
::testing::AssertionResult samePartition(const std::vector<int>& labels, const std::vector<int>& truth) {
  std::set<std::pair<int, int>> pairs;
  std::set<int>                 labelSet;
  std::set<int>                 truthSet;
  for (std::size_t row = 0; row < labels.size(); ++row) {
    pairs.emplace(labels[row], truth[row]);
    labelSet.insert(labels[row]);
    truthSet.insert(truth[row]);
  }
  if (pairs.size() == labelSet.size() && pairs.size() == truthSet.size()) {
    return ::testing::AssertionSuccess();
  }
  auto failure = ::testing::AssertionFailure() << "labels";
  for (const int label : labels) {
    failure << ' ' << label;
  }
  failure << " do not match the partition";
  for (const int group : truth) {
    failure << ' ' << group;
  }
  return failure;
}

enum class SmallInit {
  OneRowPerGroup,
  MaxMin
};

using SmallParams = std::tuple<KMeansMetric, bool, SmallInit>;
class KMeansSmallPartitionTest : public ::testing::TestWithParam<SmallParams> {};

/**
 * Eight rows in three groups of sizes 3, 3 and 2, interleaved (A B C A B C A B):
 *   A: bits 0-5, 0-4, 1-5;  B: bits 10-15, 10-14, 11-15;  C: bits 20-25, 20-24.
 * With shared bits, every row also sets bits 30 and 31, so rows of different groups are similar but still much less
 * than rows of the same group. k = 3 must give exactly this partition, and group C's center is the hand-computed mean
 * of its two rows.
 */
TEST_P(KMeansSmallPartitionTest, RecoversThreeThreeTwo) {
  const auto [metric, sharedBits, init] = GetParam();
  const nvMolKit::ScopedStream  streamOwner;
  const auto                    stream = streamOwner.stream();
  constexpr int                 words  = 1;
  std::vector<std::vector<int>> rows   = {bitRange(0, 5),
                                          bitRange(10, 15),
                                          bitRange(20, 25),
                                          bitRange(0, 4),
                                          bitRange(10, 14),
                                          bitRange(20, 24),
                                          bitRange(1, 5),
                                          bitRange(11, 15)};
  const std::vector<int>        truth  = {0, 1, 2, 0, 1, 2, 0, 1};
  const std::vector<int>        shared = sharedBits ? std::vector<int>{30, 31} : std::vector<int>{};
  for (auto& row : rows) {
    row.insert(row.end(), shared.begin(), shared.end());
  }
  const auto packed = packRows(rows, words);
  const auto initRows =
    init == SmallInit::MaxMin ? maxMinRows(packed, words, 3, metric, stream) : std::vector<int>{0, 1, 2};

  nvMolKit::KMeansOptions options;
  options.metric = metric;
  const auto fit =
    fitFromCenters(packed, words, centersFromRows(packed, words, initRows, metric, stream), options, stream);
  EXPECT_TRUE(samePartition(fit.labels, truth));
  auto sizes = fit.sizes;
  std::sort(sizes.begin(), sizes.end());
  EXPECT_EQ(sizes, (std::vector<std::int64_t>{2, 3, 3}));
  EXPECT_EQ(fit.reseeded, 0);

  // Group C: rows with 6 + s and 5 + s bits (s shared bits). Tanimoto centers average the bits; cosine centers
  // average the L2-normalized rows.
  const double longCount  = 6.0 + static_cast<double>(shared.size());
  const double shortCount = 5.0 + static_cast<double>(shared.size());
  const double both =
    metric == KMeansMetric::Tanimoto ? 1.0 : (1.0 / std::sqrt(longCount) + 1.0 / std::sqrt(shortCount)) / 2.0;
  const double longOnly = metric == KMeansMetric::Tanimoto ? 0.5 : 1.0 / (2.0 * std::sqrt(longCount));
  const float* center   = fit.centers.data() + static_cast<std::size_t>(fit.labels[2]) * words * 32;
  for (int bit = 0; bit < words * 32; ++bit) {
    double expected = 0.0;
    if ((bit >= 20 && bit <= 24) || std::find(shared.begin(), shared.end(), bit) != shared.end()) {
      expected = both;
    } else if (bit == 25) {
      expected = longOnly;
    }
    EXPECT_NEAR(center[bit], expected, 1e-6) << "bit " << bit;
  }
}

INSTANTIATE_TEST_SUITE_P(KMeans,
                         KMeansSmallPartitionTest,
                         ::testing::Combine(::testing::Values(KMeansMetric::Cosine, KMeansMetric::Tanimoto),
                                            ::testing::Bool(),
                                            ::testing::Values(SmallInit::OneRowPerGroup, SmallInit::MaxMin)));

class KMeansFitMetricTest : public ::testing::TestWithParam<KMeansMetric> {};

// Groups of 60, 25 and 5 rows around random 40-bit prototypes in 512 bits, three bits flipped per row. MaxMin
// initialization and Lloyd must keep the 5-row group instead of absorbing it into a large one.
TEST_P(KMeansFitMetricTest, RecoversUnevenPlantedGroups) {
  const KMeansMetric                 metric = GetParam();
  const nvMolKit::ScopedStream       streamOwner;
  const auto                         stream     = streamOwner.stream();
  constexpr int                      words      = 16;
  const std::vector<int>             groupSizes = {60, 25, 5};
  std::mt19937                       generator(7);
  std::uniform_int_distribution<int> anyBit(0, words * 32 - 1);
  std::vector<std::vector<int>>      rows;
  std::vector<int>                   truth;
  for (int group = 0; group < static_cast<int>(groupSizes.size()); ++group) {
    std::vector<int> bits(words * 32);
    std::iota(bits.begin(), bits.end(), 0);
    std::shuffle(bits.begin(), bits.end(), generator);
    const std::vector<int> prototype(bits.begin(), bits.begin() + 40);
    for (int member = 0; member < groupSizes[group]; ++member) {
      std::vector<bool> row(words * 32, false);
      for (const int bit : prototype) {
        row[bit] = true;
      }
      for (int flip = 0; flip < 3; ++flip) {
        const int bit = anyBit(generator);
        row[bit]      = !row[bit];
      }
      std::vector<int> set;
      for (int bit = 0; bit < words * 32; ++bit) {
        if (row[bit]) {
          set.push_back(bit);
        }
      }
      rows.push_back(set);
      truth.push_back(group);
    }
  }
  // Shuffle the row order so groups are not contiguous.
  std::vector<int> order(rows.size());
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), generator);
  std::vector<std::vector<int>> shuffledRows;
  std::vector<int>              shuffledTruth;
  for (const int index : order) {
    shuffledRows.push_back(rows[index]);
    shuffledTruth.push_back(truth[index]);
  }
  const auto packed = packRows(shuffledRows, words);

  nvMolKit::KMeansOptions options;
  options.metric     = metric;
  const auto initial = centersFromRows(packed, words, maxMinRows(packed, words, 3, metric, stream), metric, stream);
  const auto fit     = fitFromCenters(packed, words, initial, options, stream);
  EXPECT_TRUE(samePartition(fit.labels, shuffledTruth));
}

// k = 1: every row in cluster 0 and the center is the mean of all rows, reached after one update.
TEST_P(KMeansFitMetricTest, SingleClusterIsTheMean) {
  const KMeansMetric                  metric = GetParam();
  const nvMolKit::ScopedStream        streamOwner;
  const auto                          stream = streamOwner.stream();
  constexpr int                       words  = 1;
  // Bit 0 in all four rows, bit 1 in two, bit 2 in one; rows have 1, 2, 2 and 3 bits.
  const std::vector<std::vector<int>> rows   = {
    {0},
    {0, 1},
    {0, 2},
    {0, 1, 3}
  };
  const auto              packed = packRows(rows, words);
  nvMolKit::KMeansOptions options;
  options.metric = metric;
  const auto fit = fitFromCenters(packed, words, centersFromRows(packed, words, {2}, metric, stream), options, stream);
  EXPECT_EQ(fit.labels, (std::vector<int>{0, 0, 0, 0}));
  EXPECT_EQ(fit.sizes, (std::vector<std::int64_t>{4}));
  EXPECT_EQ(fit.iterations, 1);
  std::vector<double> expected(32, 0.0);
  for (const auto& row : rows) {
    const double weight = metric == KMeansMetric::Tanimoto ? 1.0 : 1.0 / std::sqrt(static_cast<double>(row.size()));
    for (const int bit : row) {
      expected[bit] += weight / static_cast<double>(rows.size());
    }
  }
  for (int bit = 0; bit < 32; ++bit) {
    EXPECT_NEAR(fit.centers[bit], expected[bit], 1e-6) << "bit " << bit;
  }
}

// k = number of rows, each row its own initial center: every row stays alone and the objective is zero.
TEST_P(KMeansFitMetricTest, OneClusterPerRow) {
  const KMeansMetric           metric = GetParam();
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream = streamOwner.stream();
  constexpr int                rows   = 37;
  constexpr int                words  = 3;
  auto                         packed = randomFingerprints(rows, words, 0.3, 11);
  packed[0]                           = 1U;  // randomFingerprints leaves rows 0 and 1 empty; keep rows distinct
  std::vector<int> all(rows);
  std::iota(all.begin(), all.end(), 0);
  nvMolKit::KMeansOptions options;
  options.metric = metric;
  const auto fit = fitFromCenters(packed, words, centersFromRows(packed, words, all, metric, stream), options, stream);
  EXPECT_EQ(fit.labels, all);
  EXPECT_EQ(fit.sizes, std::vector<std::int64_t>(rows, 1));
  EXPECT_NEAR(fit.inertia, 0.0, 1e-5);
  EXPECT_EQ(fit.reseeded, 0);
}

// Empty fingerprints form their own cluster around the zero center; nothing becomes NaN.
TEST_P(KMeansFitMetricTest, EmptyRowsClusterTogether) {
  const KMeansMetric                  metric = GetParam();
  const nvMolKit::ScopedStream        streamOwner;
  const auto                          stream = streamOwner.stream();
  constexpr int                       words  = 1;
  const std::vector<std::vector<int>> rows =
    {{}, bitRange(0, 7), {}, bitRange(16, 23), bitRange(0, 6), {}, bitRange(17, 23)};
  const std::vector<int>  truth  = {0, 1, 0, 2, 1, 0, 2};
  const auto              packed = packRows(rows, words);
  nvMolKit::KMeansOptions options;
  options.metric = metric;
  const auto fit =
    fitFromCenters(packed, words, centersFromRows(packed, words, {0, 1, 3}, metric, stream), options, stream);
  EXPECT_TRUE(samePartition(fit.labels, truth));
  for (const float value : fit.centers) {
    EXPECT_TRUE(std::isfinite(value));
  }
  const float* emptyCenter = fit.centers.data() + static_cast<std::size_t>(fit.labels[0]) * 32;
  EXPECT_EQ(std::count(emptyCenter, emptyCenter + 32, 0.0F), 32);
  EXPECT_TRUE(std::isfinite(fit.inertia));
}

// Every row empty, two clusters: the second cluster never gets a member and is reseeded, without NaN.
TEST_P(KMeansFitMetricTest, AllRowsEmpty) {
  const KMeansMetric               metric = GetParam();
  const nvMolKit::ScopedStream     streamOwner;
  const auto                       stream = streamOwner.stream();
  constexpr int                    words  = 2;
  const std::vector<std::uint32_t> packed(6 * words, 0U);
  nvMolKit::KMeansOptions          options;
  options.metric = metric;
  const auto fit =
    fitFromCenters(packed, words, centersFromRows(packed, words, {0, 1}, metric, stream), options, stream);
  EXPECT_EQ(fit.labels, std::vector<int>(6, 0));
  EXPECT_EQ(fit.inertia, 0.0);
  for (const float value : fit.centers) {
    EXPECT_EQ(value, 0.0F);
  }
}

// Two distinct rows, each repeated four times, and k = 3: duplicates always share a label, the objective is zero, and
// the third cluster (a duplicate center) is reseeded rather than breaking the fit.
TEST_P(KMeansFitMetricTest, DuplicateRows) {
  const KMeansMetric            metric = GetParam();
  const nvMolKit::ScopedStream  streamOwner;
  const auto                    stream = streamOwner.stream();
  constexpr int                 words  = 1;
  std::vector<std::vector<int>> rows;
  std::vector<int>              truth;
  for (int copy = 0; copy < 4; ++copy) {
    rows.push_back(bitRange(0, 9));
    truth.push_back(0);
    rows.push_back(bitRange(12, 20));
    truth.push_back(1);
  }
  const auto              packed = packRows(rows, words);
  nvMolKit::KMeansOptions options;
  options.metric = metric;
  const auto fit =
    fitFromCenters(packed, words, centersFromRows(packed, words, {0, 2, 1}, metric, stream), options, stream);
  EXPECT_TRUE(samePartition(fit.labels, truth));
  EXPECT_NEAR(fit.inertia, 0.0, 1e-5);
  EXPECT_GE(fit.reseeded, 1);
}

// Fitting again from converged centers changes nothing: one iteration, same labels, same centers, no reseeds.
TEST_P(KMeansFitMetricTest, ConvergedCentersAreAFixedPoint) {
  const KMeansMetric           metric = GetParam();
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream = streamOwner.stream();
  constexpr int                words  = 4;
  const auto                   packed = randomFingerprints(300, words, 0.15, 5);
  nvMolKit::KMeansOptions      options;
  options.metric = metric;
  options.tol    = 1e-12;
  const auto first =
    fitFromCenters(packed,
                   words,
                   centersFromRows(packed, words, maxMinRows(packed, words, 10, metric, stream), metric, stream),
                   options,
                   stream);
  ASSERT_LT(first.iterations, options.maxIter);
  const auto second = fitFromCenters(packed, words, first.centers, options, stream);
  EXPECT_EQ(second.iterations, 1);
  EXPECT_EQ(second.labels, first.labels);
  EXPECT_EQ(second.centers, first.centers);
  EXPECT_EQ(second.reseeded, 0);
}

// The same inputs give bit-identical results.
TEST_P(KMeansFitMetricTest, IsDeterministic) {
  const KMeansMetric           metric = GetParam();
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream = streamOwner.stream();
  constexpr int                words  = 8;
  const auto                   packed = randomFingerprints(2000, words, 0.1, 9);
  nvMolKit::KMeansOptions      options;
  options.metric     = metric;
  const auto initial = centersFromRows(packed, words, maxMinRows(packed, words, 40, metric, stream), metric, stream);
  const auto first   = fitFromCenters(packed, words, initial, options, stream);
  const auto second  = fitFromCenters(packed, words, initial, options, stream);
  EXPECT_EQ(first.labels, second.labels);
  EXPECT_EQ(first.centers, second.centers);
  EXPECT_EQ(first.inertia, second.inertia);
  EXPECT_EQ(first.iterations, second.iterations);
}

INSTANTIATE_TEST_SUITE_P(KMeans, KMeansFitMetricTest, ::testing::Values(KMeansMetric::Cosine, KMeansMetric::Tanimoto));

TEST(KMeansFit, RejectsInvalidArguments) {
  const nvMolKit::ScopedStream streamOwner;
  const auto                   stream = streamOwner.stream();
  auto                         device = upload(randomFingerprints(10, 2, 0.2, 1), stream);
  EXPECT_THROW(nvMolKit::kmeansInitGpu({device.data(), device.size()},
                                       10,
                                       2,
                                       11,
                                       KMeansMetric::Tanimoto,
                                       nvMolKit::KMeansInit::Random,
                                       0,
                                       stream),
               std::invalid_argument);
  EXPECT_THROW(nvMolKit::kmeansInitGpu({device.data(), device.size()},
                                       10,
                                       3,
                                       2,
                                       KMeansMetric::Tanimoto,
                                       nvMolKit::KMeansInit::Random,
                                       0,
                                       stream),
               std::invalid_argument);
}

}  // namespace
