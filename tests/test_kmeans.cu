// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <tuple>
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
  const auto hostLabels = download(labels, stream);
  for (int row = 0; row < rows; ++row) {
    double best = -INFINITY;
    for (int center = 0; center < clusters; ++center) {
      best = std::max(best, referenceScore(packed, centers, words, row, center, metric));
    }
    ASSERT_GE(hostLabels[row], 0);
    ASSERT_LT(hostLabels[row], clusters);
    const double chosen = referenceScore(packed, centers, words, row, hostLabels[row], metric);
    EXPECT_LE(best - chosen, 1e-5 * std::max(1.0, std::abs(best))) << "row " << row;
  }
}

INSTANTIATE_TEST_SUITE_P(KMeans,
                         KMeansAssignTest,
                         ::testing::Combine(::testing::Values(KMeansMetric::Cosine, KMeansMetric::Tanimoto),
                                            ::testing::Values(130),
                                            ::testing::Values(1, 9, 32, 67),
                                            ::testing::Values(1, 5, 130)));

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

}  // namespace
