// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/bitbirch.h"
#include "src/bitbirch_common.cuh"
#include "src/bitbirch_internal.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"

namespace {

struct PrimitiveResults {
  bool   centroidBits[5];
  double isim;
  bool   exactThreshold;
  bool   rejectsAboveThreshold;
};

__global__ void evaluatePrimitivesKernel(PrimitiveResults* output) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  PrimitiveResults result{};
  for (std::uint64_t linearSum = 0; linearSum < 5; ++linearSum) {
    result.centroidBits[linearSum] = nvMolKit::bitbirch::majorityCentroidBit(linearSum, 4);
  }
  nvMolKit::bitbirch::ISimTanimotoTerms terms{};
  nvMolKit::bitbirch::accumulateISimTanimotoTerm(terms, 2, 3);
  nvMolKit::bitbirch::accumulateISimTanimotoTerm(terms, 1, 3);
  result.isim                  = nvMolKit::bitbirch::isimTanimoto(terms, 3);
  result.exactThreshold        = nvMolKit::bitbirch::isimTanimotoAtLeast(terms, 3, 0.2);
  result.rejectsAboveThreshold = !nvMolKit::bitbirch::isimTanimotoAtLeast(terms, 3, 0.2000001);
  *output                      = result;
}

TEST(BitBirchPrimitives, DeviceEvaluationMatchesPaperEquations) {
  nvMolKit::AsyncDevicePtr<PrimitiveResults> deviceResults;
  evaluatePrimitivesKernel<<<1, 1>>>(deviceResults.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  PrimitiveResults result{};
  deviceResults.get(result);
  EXPECT_FALSE(result.centroidBits[0]);
  EXPECT_FALSE(result.centroidBits[1]);
  EXPECT_TRUE(result.centroidBits[2]);
  EXPECT_TRUE(result.centroidBits[3]);
  EXPECT_TRUE(result.centroidBits[4]);
  EXPECT_DOUBLE_EQ(result.isim, 0.2);
  EXPECT_TRUE(result.exactThreshold);
  EXPECT_TRUE(result.rejectsAboveThreshold);
}

TEST(BitBirchPrimitives, HostEdgeCaseConventionsAreExplicit) {
  nvMolKit::bitbirch::ISimTanimotoTerms empty{};
  EXPECT_DOUBLE_EQ(nvMolKit::bitbirch::isimTanimoto(empty, 0), 1.0);
  EXPECT_DOUBLE_EQ(nvMolKit::bitbirch::isimTanimoto(empty, 1), 1.0);
  EXPECT_DOUBLE_EQ(nvMolKit::bitbirch::isimTanimoto(empty, 8), 1.0);

  nvMolKit::bitbirch::ISimTanimotoTerms large{};
  const std::uint64_t                   count = std::uint64_t{1} << 40;
  nvMolKit::bitbirch::accumulateISimTanimotoTerm(large, count, count);
  nvMolKit::bitbirch::accumulateISimTanimotoTerm(large, count / 2, count);
  EXPECT_TRUE(std::isfinite(large.commonPairs));
  EXPECT_TRUE(std::isfinite(large.mismatches));
  EXPECT_GT(nvMolKit::bitbirch::isimTanimoto(large, count), 0.0);
  EXPECT_LT(nvMolKit::bitbirch::isimTanimoto(large, count), 1.0);
}

nvMolKit::BitBirchOptions treeOptions(const int branchingFactor, const int batchSize, const bool centroids = false) {
  nvMolKit::BitBirchOptions options;
  options.branchingFactor = branchingFactor;
  options.batchSize       = batchSize;
  options.returnCentroids = centroids;
  return options;
}

TEST(BitBirchValidation, RejectsInvalidOptionsBeforeLaunching) {
  const cuda::std::span<const std::uint32_t> empty;
  EXPECT_THROW(nvMolKit::bitBirchGpu(empty, 0, 1, std::numeric_limits<double>::quiet_NaN(), treeOptions(3, 1)),
               std::invalid_argument);
  EXPECT_THROW(nvMolKit::bitBirchGpu(empty, 0, 1, 0.5, treeOptions(2, 1)), std::invalid_argument);
  EXPECT_THROW(nvMolKit::bitBirchGpu(empty, 0, 1, 0.5, treeOptions(3, 0)), std::invalid_argument);
}

TEST(BitBirch, MergesAtThresholdAndReturnsMajorityCentroids) {
  const std::vector<std::uint32_t>           fingerprints{0b0011U, 0b0001U, 0b1100U, 0b0100U};
  nvMolKit::AsyncDeviceVector<std::uint32_t> deviceFingerprints(fingerprints.size());
  deviceFingerprints.copyFromHost(fingerprints);

  auto                       result = nvMolKit::bitBirchGpu({deviceFingerprints.data(), deviceFingerprints.size()},
                                      4,
                                      1,
                                      0.5,
                                      treeOptions(3, 4, /*centroids=*/true));
  std::vector<int>           labels(4);
  std::vector<std::uint32_t> centroids(result.centroids.size());
  result.clusterIds.copyToHost(labels);
  result.centroids.copyToHost(centroids);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(result.numClusters, 2);
  EXPECT_EQ(labels, (std::vector<int>{0, 0, 1, 1}));
  EXPECT_EQ(centroids, (std::vector<std::uint32_t>{0b0011U, 0b1100U}));
}

TEST(BitBirch, CascadingSplitsPreserveDenseDeterministicLabels) {
  std::vector<std::uint32_t> fingerprints(24);
  for (int index = 0; index < 24; ++index) {
    fingerprints[index] = std::uint32_t{1} << index;
  }
  nvMolKit::AsyncDeviceVector<std::uint32_t> deviceFingerprints(fingerprints.size());
  deviceFingerprints.copyFromHost(fingerprints);

  auto result =
    nvMolKit::bitBirchGpu({deviceFingerprints.data(), deviceFingerprints.size()}, 24, 1, 0.9, treeOptions(3, 7));
  std::vector<int> labels(24);
  result.clusterIds.copyToHost(labels);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(result.numClusters, 24);
  for (int index = 0; index < 24; ++index) {
    EXPECT_EQ(labels[index], index);
  }
}

struct Clustering {
  int                        numClusters = 0;
  std::vector<int>           labels;
  std::vector<std::uint32_t> centroids;
};

Clustering clusterPackedRows(const std::vector<std::uint32_t>& fingerprints,
                             const int                         numWords,
                             const double                      threshold,
                             const nvMolKit::BitBirchOptions&  options,
                             const int                         offset             = 0,
                             const int                         sharedMemoryBudget = 0,
                             const cudaStream_t                stream             = nullptr) {
  std::vector<std::uint32_t> padded(offset, 0);
  padded.insert(padded.end(), fingerprints.begin(), fingerprints.end());
  nvMolKit::AsyncDeviceVector<std::uint32_t> deviceFingerprints(padded.size(), stream);
  deviceFingerprints.copyFromHost(padded);
  const int                                  rows = static_cast<int>(fingerprints.size() / numWords);
  const cuda::std::span<const std::uint32_t> packed{deviceFingerprints.data() + offset, fingerprints.size()};
  auto result = sharedMemoryBudget == 0 ? nvMolKit::bitBirchGpu(packed, rows, numWords, threshold, options, stream) :
                                          nvMolKit::bitbirch::clusterWithSharedMemoryBudget(packed,
                                                                                            rows,
                                                                                            numWords,
                                                                                            threshold,
                                                                                            options,
                                                                                            sharedMemoryBudget,
                                                                                            stream);
  Clustering clustering{result.numClusters,
                        std::vector<int>(rows),
                        std::vector<std::uint32_t>(result.centroids.size())};
  result.clusterIds.copyToHost(clustering.labels);
  result.centroids.copyToHost(clustering.centroids);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  return clustering;
}

//! Clusters one-word fingerprints and copies the result to the host.
Clustering clusterRows(const std::vector<std::uint32_t>& fingerprints,
                       const double                      threshold,
                       const int                         branchingFactor,
                       const int                         batchSize) {
  return clusterPackedRows(fingerprints, 1, threshold, treeOptions(branchingFactor, batchSize, true));
}

// Expected values in the tests below come from nvmolkit/tests/_bitbirch_reference.py.

TEST(BitBirch, SplitsWithMergesMatchReference) {
  // Batches of 16 with branching factor 3: merges, ties and cascading splits across several rounds.
  const std::vector<std::uint32_t> fingerprints{
    0xF1E54A8BU, 0xA00641A9U, 0xAF266A95U, 0xE5AFCDBCU, 0x940AF962U, 0xC693565FU, 0xD56A2742U, 0x39A72DABU,
    0x0E375145U, 0x4CD7B299U, 0x48FA2734U, 0xDFA132D7U, 0xE9A1AC70U, 0x01591126U, 0x7FF206DDU, 0xD23C068FU,
    0x21A6195EU, 0xCC0CBDF9U, 0x1E7C3921U, 0x77CA95C7U, 0xD103DC48U, 0x4D93887AU, 0x57735285U, 0x4746E6A2U,
    0xB82F3EF0U, 0x413F221FU, 0xFD8EC592U, 0x71F08551U, 0x7A67F71EU, 0x812A131FU, 0xA00641A9U, 0xAF266A95U,
    0x826B236BU, 0xFED91B48U, 0xCEC30D63U, 0x4CD7B299U, 0xB3452FB7U, 0x9F472352U, 0x574D796EU, 0xA00641A9U,
  };
  const auto clustering = clusterRows(fingerprints, 0.4, 3, 16);
  EXPECT_EQ(clustering.numClusters, 13);
  EXPECT_EQ(clustering.labels, (std::vector<int>{0,  1, 0, 0,  2,  0, 2, 0, 3,  4, 5, 0, 0, 6,  0, 0, 7, 4, 8, 9,
                                                 10, 4, 9, 11, 12, 0, 9, 9, 12, 7, 1, 0, 2, 12, 2, 4, 0, 2, 2, 1}));
  EXPECT_EQ(clustering.centroids,
            (std::vector<std::uint32_t>{0xFFA72E9FU,
                                        0xA00641A9U,
                                        0xD74B2B62U,
                                        0x0E375145U,
                                        0x4CD7BAF9U,
                                        0x48FA2734U,
                                        0x01591126U,
                                        0xA1AE1B5FU,
                                        0x1E7C3921U,
                                        0x77FAD5D7U,
                                        0xD103DC48U,
                                        0x4746E6A2U,
                                        0xFA6F3F58U}));
}

TEST(BitBirch, LeafSplitsAcrossTwoPairSearchTilesMatchReference) {
  // Branching factor 64: overflowing leaves hold 65 entries, two 64-entry split tiles. The last 60 rows are
  // near-duplicates of earlier rows, so their merges depend on how the splits routed those rows.
  const std::vector<std::uint32_t> fingerprints{
    0x2240CC4EU, 0x20E9FA10U, 0xCC0D7A67U, 0x7FD0AC8AU, 0x970C647CU, 0x99FBCBDEU, 0xB650EBD3U, 0x075829B0U, 0x7C49F18DU,
    0x25DE7BDFU, 0x66C83754U, 0xED9F3CD3U, 0x8C398A35U, 0x12071537U, 0x8AF14376U, 0x2138DD94U, 0xC122D059U, 0xF2C5A74CU,
    0xFABCF3B1U, 0x9F33C35CU, 0xDE4B2490U, 0x5E765556U, 0x254F393BU, 0x82EA74DBU, 0x71988AD5U, 0xA9B01363U, 0xFE9A2F13U,
    0x467AA377U, 0xDB1059FFU, 0x2351E028U, 0x5905EC7EU, 0xC9BCF67EU, 0x3F6464D7U, 0xAB9CC051U, 0x7580A7C2U, 0x832B7CBFU,
    0xF0F7B702U, 0xD115A399U, 0xD6D3DE9EU, 0x8C903262U, 0xFB75E337U, 0xFB1D2800U, 0x229735DFU, 0x345ABB68U, 0x4EF11A2AU,
    0x8DC145E7U, 0xD2E8A31BU, 0x7BCED400U, 0xFBD805DBU, 0x5A70388CU, 0xEDA67F42U, 0x9772CA32U, 0xB884EF0AU, 0x3C3CB396U,
    0x978E63F5U, 0xCD5D27B3U, 0xE2FE3F88U, 0xDE099257U, 0xF7A31851U, 0x20F664D0U, 0xC75A4A51U, 0x77921C13U, 0xAF89B73DU,
    0x46F2F7B9U, 0x03BB0743U, 0x154727D3U, 0xF90106D9U, 0xE55C9B2EU, 0x4DAD7A8EU, 0x6E111E13U, 0x3DACF591U, 0x25CF18D7U,
    0xDB856C6DU, 0xAC6179B6U, 0x139F6A96U, 0x33C46DFAU, 0x90502131U, 0xE6C42FEAU, 0xFEF7C141U, 0x37970737U, 0x9C89FC5FU,
    0x0877952CU, 0x2CAFFC05U, 0x3365981CU, 0x71277A71U, 0x5882EEC2U, 0xBA1C5D99U, 0x780A5D8AU, 0x56C999EEU, 0xE7F86B87U,
    0xA0172773U, 0xB286419DU, 0xBEF433E2U, 0x56DDB804U, 0xE42CF02CU, 0x045210AEU, 0x4E8A8CB1U, 0x28EA34A3U, 0x0106E31AU,
    0xFF166BEAU, 0x16FE88C2U, 0x75AFF24BU, 0xD1D47083U, 0xB0E7FDF0U, 0xDB4B1271U, 0x0DFEB9DFU, 0x7F48C077U, 0x08B784E2U,
    0x3104E055U, 0xD88C4107U, 0x12BBD7B1U, 0x967F6E4DU, 0x0528B45EU, 0x4F079A0BU, 0xEEC9C643U, 0x513F9869U, 0x4F1EBDE5U,
    0x16D840B1U, 0xC6528A29U, 0x2C341331U, 0x7FB90559U, 0x064B466FU, 0x05A57A40U, 0xD6D0E2D5U, 0x02015B0EU, 0x775FA576U,
    0x6D5E6738U, 0x20905ECCU, 0xA51ABC04U, 0xBD3F487BU, 0xFC2928ACU, 0x32164DC9U, 0xEC590FA1U, 0x0FDA0128U, 0x242A041AU,
    0x993039A3U, 0xDE06D510U, 0xE5506148U, 0xBD6EDED8U, 0x06E5C36EU, 0x71B88AC5U, 0xE47C9B2EU, 0x826874DBU, 0x25CE73DFU,
    0xD0502121U, 0x91303923U, 0x0DEEB9DDU, 0x7BC6D440U, 0x22414C4EU, 0xC9B8FE7EU, 0x2250CC4CU, 0xC6508B29U, 0x254FBD3BU,
    0x9F33C758U, 0xE9BCF47EU, 0x313CDD94U, 0xCD9F3CD7U, 0x245200AEU, 0xE6DE3F88U, 0x50D47083U, 0x4CAD7A9EU, 0x0528BC5CU,
    0x58C6EEC2U, 0x4FBEBDE5U, 0x67C83F54U, 0x2240CC5CU, 0xFB19A800U, 0x13BB6A96U, 0x950C64FCU, 0xDB1D2000U, 0xB884ED1AU,
    0x568989EEU, 0x128F6A96U, 0xDE0990D7U, 0xA9B81367U, 0x12851537U, 0x4DA97A9EU, 0x0428B65EU, 0xA8572773U, 0x12071F37U,
    0x064BC26FU, 0x467AAB37U, 0x468A8CA1U, 0x282A34A3U, 0xC022D019U, 0x0FD80328U, 0xF8F5B702U, 0x2361E028U, 0x6E11BE13U,
    0xAC6079BEU, 0x06D848B1U, 0xDB431270U, 0x3C3C9296U, 0x503F9879U, 0xA9B00263U, 0xFABC73B9U, 0x2C349131U, 0xC2E8A33BU,
    0x30164DCDU, 0xB2C5874CU,
  };
  const std::vector<int> expected{
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,
    23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,
    46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,
    69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
    92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 9,   105, 106, 107, 108, 109, 110, 111, 112, 113,
    114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
    137, 138, 139, 140, 23,  9,   76,  134, 9,   47,  0,   31,  0,   117, 141, 19,  31,  142, 11,  95,  56,  102, 143,
    144, 85,  145, 146, 0,   147, 148, 4,   147, 52,  88,  149, 57,  150, 13,  143, 144, 151, 13,  120, 27,  96,  97,
    16,  132, 36,  29,  69,  73,  116, 152, 53,  114, 150, 18,  118, 27,  130, 17};
  std::vector<std::uint32_t> expectedCentroids(153, 0);
  for (int cluster = 0; cluster < 153; ++cluster) {
    int              members = 0;
    std::vector<int> sums(32, 0);
    for (std::size_t row = 0; row < fingerprints.size(); ++row) {
      if (expected[row] != cluster) {
        continue;
      }
      ++members;
      for (int bit = 0; bit < 32; ++bit) {
        sums[bit] += (fingerprints[row] >> bit) & 1U;
      }
    }
    for (int bit = 0; bit < 32; ++bit) {
      if (2 * sums[bit] >= members) {
        expectedCentroids[cluster] |= std::uint32_t{1} << bit;
      }
    }
  }
  for (const int numWords : {1, 128}) {
    std::vector<std::uint32_t> packed;
    for (const auto row : fingerprints) {
      packed.insert(packed.end(), numWords, row);
    }
    for (const int budget : {0, 48 * 1024}) {
      SCOPED_TRACE(::testing::Message() << "words=" << numWords << ", budget=" << budget);
      const auto clustering = clusterPackedRows(packed, numWords, 0.7, treeOptions(64, 256, true), 0, budget);
      ASSERT_EQ(clustering.numClusters, 153);
      EXPECT_EQ(clustering.labels, expected);
      ASSERT_EQ(clustering.centroids.size(), 153u * numWords);
      for (int cluster = 0; cluster < 153; ++cluster) {
        for (int word = 0; word < numWords; ++word) {
          EXPECT_EQ(clustering.centroids[cluster * numWords + word], expectedCentroids[cluster]);
        }
      }
    }
  }
}

TEST(BitBirch, OffsetSpansMatchAlignedInputsAcrossWidthsAndSplits) {
  for (const int numWords : {1, 4, 32, 128}) {
    std::vector<std::uint32_t> fingerprints;
    for (const std::uint32_t value : {0x3U, 0x1U, 0xCU, 0x4U, 0x30U, 0x10U, 0xC0U, 0x40U}) {
      fingerprints.insert(fingerprints.end(), numWords, value);
    }
    const auto options = treeOptions(3, 3, true);
    for (const int offset : {0, 1, 2, 3}) {
      SCOPED_TRACE(::testing::Message() << "words=" << numWords << ", offset=" << offset);
      nvMolKit::ScopedStream stream;
      const auto result = clusterPackedRows(fingerprints, numWords, 0.5, options, offset, 0, stream.stream());
      ASSERT_EQ(result.numClusters, 5);
      EXPECT_EQ(result.labels, (std::vector<int>{0, 0, 1, 1, 2, 2, 3, 4}));
      ASSERT_EQ(result.centroids.size(), 5u * numWords);
      const std::vector<std::uint32_t> expectedCentroids{0x3U, 0xCU, 0x30U, 0xC0U, 0x40U};
      for (int cluster = 0; cluster < 5; ++cluster) {
        for (int word = 0; word < numWords; ++word) {
          EXPECT_EQ(result.centroids[cluster * numWords + word], expectedCentroids[cluster]);
        }
      }
    }
  }
}

TEST(BitBirch, SmallSharedMemoryBudgetPreservesCascadingSplitsAtMaximumWidth) {
  constexpr int numWords = nvMolKit::kBitBirchMaxWords;
  constexpr int rows     = 132;
  EXPECT_EQ(nvMolKit::bitbirch::selectSplitTileSize(numWords, 48 * 1024), 32);
  EXPECT_EQ(nvMolKit::bitbirch::selectSplitTileSize(numWords, 64 * 1024), 32);
  EXPECT_EQ(nvMolKit::bitbirch::selectSplitTileSize(4, 48 * 1024), 64);
  std::vector<std::uint32_t> fingerprints(rows * numWords, 0);
  for (int row = 0; row < rows; ++row) {
    fingerprints[row * numWords + row / 32] = std::uint32_t{1} << (row % 32);
  }
  for (const int branchingFactor : {3, 64, 128}) {
    for (const int budget : {0, 48 * 1024}) {
      SCOPED_TRACE(::testing::Message() << "branching=" << branchingFactor << ", budget=" << budget);
      const auto result =
        clusterPackedRows(fingerprints, numWords, 1.0, treeOptions(branchingFactor, 37, true), 0, budget);
      ASSERT_EQ(result.numClusters, rows);
      ASSERT_EQ(result.labels.size(), rows);
      EXPECT_EQ(result.centroids, fingerprints);
      for (int row = 0; row < rows; ++row) {
        EXPECT_EQ(result.labels[row], row);
      }
    }
  }
  EXPECT_THROW(clusterPackedRows(fingerprints, numWords, 1.0, treeOptions(3, 37), 0, 1), std::runtime_error);
}

TEST(BitBirch, EmptySingletonAndThresholdBoundaries) {
  for (const int numWords : {1, 4, 128}) {
    const auto empty = nvMolKit::bitBirchGpu({}, 0, numWords, 0.5);
    EXPECT_EQ(empty.numClusters, 0);
    EXPECT_EQ(empty.numWords, numWords);
    EXPECT_EQ(empty.clusterIds.size(), 0u);
    EXPECT_EQ(empty.centroids.size(), 0u);

    const std::vector<std::uint32_t> singleton(numWords, 0xA5A5A5A5U);
    const auto single = clusterPackedRows(singleton, numWords, 1.0, treeOptions(65534, INT_MAX, true));
    EXPECT_EQ(single.numClusters, 1);
    EXPECT_EQ(single.labels, (std::vector<int>{0}));
    EXPECT_EQ(single.centroids, singleton);

    std::vector<std::uint32_t> fingerprints(4 * numWords, 0);
    fingerprints[2 * numWords] = 1;
    fingerprints[3 * numWords] = 1;
    for (const bool centroids : {false, true}) {
      for (const double threshold : {0.0, 1.0}) {
        SCOPED_TRACE(::testing::Message() << "words=" << numWords << ", threshold=" << threshold);
        const auto result = clusterPackedRows(fingerprints, numWords, threshold, treeOptions(3, 3, centroids));
        EXPECT_EQ(result.numClusters, threshold == 0.0 ? 1 : 2);
        EXPECT_EQ(result.labels, threshold == 0.0 ? std::vector<int>({0, 0, 0, 0}) : std::vector<int>({0, 0, 1, 1}));
        if (centroids) {
          std::vector<std::uint32_t> expected(result.numClusters * numWords, 0);
          expected[threshold == 0.0 ? 0 : numWords] = 1;
          EXPECT_EQ(result.centroids, expected);
        } else {
          EXPECT_TRUE(result.centroids.empty());
        }
      }
    }
  }
}

TEST(BitBirchValidation, RejectsMatrixBoundsAndNonfiniteThresholds) {
  const cuda::std::span<const std::uint32_t> empty;
  for (const int rows : {-1, 1, INT_MAX}) {
    EXPECT_THROW(nvMolKit::bitBirchGpu(empty, rows, 1, 0.5), std::invalid_argument);
  }
  for (const int words : {0, -1, 129}) {
    EXPECT_THROW(nvMolKit::bitBirchGpu(empty, 0, words, 0.5), std::invalid_argument);
  }
  for (const double threshold : {-0.1,
                                 1.1,
                                 std::numeric_limits<double>::infinity(),
                                 -std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(nvMolKit::bitBirchGpu(empty, 0, 1, threshold), std::invalid_argument);
  }
  EXPECT_THROW(nvMolKit::bitBirchGpu(empty, 0, 1, 0.5, treeOptions(65535, 1)), std::invalid_argument);
  EXPECT_THROW(nvMolKit::bitBirchGpu(empty, 0, 1, 0.5, treeOptions(3, -1)), std::invalid_argument);
}

TEST(BitBirch, GroupedMergeAcceptsWholeRunsAndFallsBackInInputOrder) {
  constexpr int warmup = 16384;
  for (const int batchSize : {256, 1024}) {
    for (const bool rejectGroup : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "batch=" << batchSize << ", reject=" << rejectGroup);
      std::vector<std::uint32_t> fingerprints(warmup, 0xCU);
      fingerprints[0] = 0x3U;
      fingerprints.push_back(0x1U);
      fingerprints.push_back(rejectGroup ? 0x2U : 0x1U);
      std::vector<int> expected(warmup + 2, 1);
      expected[0] = expected[warmup] = 0;
      expected[warmup + 1]           = rejectGroup ? 2 : 0;
      for (int repeat = 0; repeat < 2; ++repeat) {
        const auto result = clusterRows(fingerprints, 0.5, 3, batchSize);
        EXPECT_EQ(result.labels, expected);
        EXPECT_EQ(result.numClusters, rejectGroup ? 3 : 2);
        EXPECT_EQ(
          result.centroids,
          rejectGroup ? std::vector<std::uint32_t>({0x3U, 0xCU, 0x2U}) : std::vector<std::uint32_t>({0x1U, 0xCU}));
      }
    }
  }
}

TEST(BitBirch, WarmupBoundariesAndIndependentGroupsPreserveLabels) {
  constexpr int warmup = 16384;
  for (const int rows : {warmup - 1, warmup, warmup + 1}) {
    std::vector<std::uint32_t> fingerprints(rows, 0xCU);
    fingerprints[0] = 0x3U;
    if (rows > warmup) {
      fingerprints[warmup] = 0x1U;
    }
    std::vector<int> expected(rows, 1);
    expected[0] = 0;
    if (rows > warmup) {
      expected[warmup] = 0;
    }
    const auto result = clusterRows(fingerprints, 0.5, 3, 256);
    EXPECT_EQ(result.numClusters, 2);
    EXPECT_EQ(result.labels, expected);
    EXPECT_EQ(result.centroids, (std::vector<std::uint32_t>{0x3U, 0xCU}));
  }
  std::vector<std::uint32_t> fingerprints(warmup, 0xCU);
  fingerprints[0] = 0x3U;
  fingerprints.insert(fingerprints.end(), {0x1U, 0xCU, 0x2U, 0xCU});
  std::vector<int> expected(warmup + 4, 1);
  expected[0] = expected[warmup] = 0;
  expected[warmup + 2]           = 2;
  const auto result              = clusterRows(fingerprints, 0.5, 3, 1024);
  EXPECT_EQ(result.numClusters, 3);
  EXPECT_EQ(result.labels, expected);
  EXPECT_EQ(result.centroids, (std::vector<std::uint32_t>{0x3U, 0xCU, 0x2U}));
}

TEST(BitBirch, GroupedRoundsKeepClusterInvariantsOnGeneratedData) {
  // Noisy copies of random prototypes, past the ordered warm-up so grouped merges run. Raw mt19937 output is
  // specified by the standard, so the data is the same on every platform.
  constexpr int              numWords  = 8;
  constexpr int              rows      = 20000;
  constexpr double           threshold = 0.6;
  std::mt19937               rng(7);
  std::vector<std::uint32_t> prototypes(200 * numWords);
  for (auto& word : prototypes) {
    word = rng() & rng() & rng();
  }
  std::vector<std::uint32_t> fingerprints(static_cast<std::size_t>(rows) * numWords);
  for (int row = 0; row < rows; ++row) {
    const auto prototype = rng() % 200;
    for (int word = 0; word < numWords; ++word) {
      fingerprints[row * numWords + word] =
        prototypes[prototype * numWords + word] ^ (rng() & rng() & rng() & rng() & rng());
    }
  }
  const auto clustering = clusterPackedRows(fingerprints, numWords, threshold, treeOptions(50, 1024, true));
  const auto repeat     = clusterPackedRows(fingerprints, numWords, threshold, treeOptions(50, 1024, true));
  EXPECT_EQ(repeat.labels, clustering.labels);
  EXPECT_EQ(repeat.centroids, clustering.centroids);

  std::vector<std::vector<int>> members(clustering.numClusters);
  for (int row = 0; row < rows; ++row) {
    ASSERT_GE(clustering.labels[row], 0);
    ASSERT_LT(clustering.labels[row], clustering.numClusters);
    members[clustering.labels[row]].push_back(row);
  }
  int lateMerges = 0;
  for (int cluster = 0; cluster < clustering.numClusters; ++cluster) {
    const auto& rowsOf = members[cluster];
    ASSERT_FALSE(rowsOf.empty());
    if (cluster > 0) {
      EXPECT_LT(members[cluster - 1].front(), rowsOf.front());
    }
    lateMerges += rowsOf.front() < 16384 && rowsOf.back() >= 16384;
    nvMolKit::bitbirch::ISimTanimotoTerms terms{};
    for (int word = 0; word < numWords; ++word) {
      std::uint32_t centroid = 0;
      for (int bit = 0; bit < 32; ++bit) {
        std::uint64_t sum = 0;
        for (const int row : rowsOf) {
          sum += (fingerprints[row * numWords + word] >> bit) & 1U;
        }
        nvMolKit::bitbirch::accumulateISimTanimotoTerm(terms, sum, rowsOf.size());
        centroid |= static_cast<std::uint32_t>(nvMolKit::bitbirch::majorityCentroidBit(sum, rowsOf.size())) << bit;
      }
      EXPECT_EQ(clustering.centroids[cluster * numWords + word], centroid) << "cluster " << cluster;
    }
    EXPECT_TRUE(nvMolKit::bitbirch::isimTanimotoAtLeast(terms, rowsOf.size(), threshold)) << "cluster " << cluster;
  }
  EXPECT_GT(lateMerges, 0);
}

TEST(BitBirch, ConcurrentCallsUseIndependentStreamsAndWidths) {
  const auto cluster = [](const int numWords) {
    nvMolKit::ScopedStream     stream;
    std::vector<std::uint32_t> fingerprints;
    for (const std::uint32_t value : {0x3U, 0x1U, 0xCU, 0x4U}) {
      fingerprints.insert(fingerprints.end(), numWords, value);
    }
    for (int repeat = 0; repeat < 4; ++repeat) {
      const auto result =
        clusterPackedRows(fingerprints, numWords, 0.5, treeOptions(3, 4, true), 0, 0, stream.stream());
      EXPECT_EQ(result.numClusters, 2);
      EXPECT_EQ(result.labels, (std::vector<int>{0, 0, 1, 1}));
      ASSERT_EQ(result.centroids.size(), 2u * numWords);
      for (int word = 0; word < numWords; ++word) {
        EXPECT_EQ(result.centroids[word], 0x3U);
        EXPECT_EQ(result.centroids[numWords + word], 0xCU);
      }
    }
  };
  auto narrow = std::async(std::launch::async, cluster, 4);
  auto wide   = std::async(std::launch::async, cluster, 128);
  narrow.get();
  wide.get();
}

}  // namespace
