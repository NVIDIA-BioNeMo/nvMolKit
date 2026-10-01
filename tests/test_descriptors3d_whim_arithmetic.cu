// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "src/descriptors3d_projection.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

using nvMolKit::AsyncDeviceVector;
using nvMolKit::checkReturnCode;
using nvMolKit::descriptors3d_detail::roundWhimThousandths;
using nvMolKit::descriptors3d_detail::thousandthsToUnits;
using nvMolKit::descriptors3d_detail::WhimThreshold;

namespace {

// RDKit's expressions, evaluated with IEEE float64 division as RDKit does.
__device__ double referenceUnits(const int32_t a) {
  return __ddiv_rn(static_cast<double>(a), 1000.0);
}

__global__ void countDivisionMismatches(unsigned long long* mismatches) {
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t bits = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x; bits < (uint64_t{1} << 32);
       bits += stride) {
    const auto a = static_cast<int32_t>(static_cast<uint32_t>(bits));
    if (__double_as_longlong(thousandthsToUnits(a)) != __double_as_longlong(referenceUnits(a))) {
      atomicAdd(mismatches, 1ull);
    }
  }
}

__global__ void countRoundingMismatches(const double* values, const int count, unsigned long long* mismatches) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < count) {
    const double expected = __ddiv_rn(round(values[idx] * 1000.0), 1000.0);
    if (__double_as_longlong(roundWhimThousandths(values[idx])) != __double_as_longlong(expected)) {
      atomicAdd(mismatches, 1ull);
    }
  }
}

//! Mismatch counters: [0] mirrored(), [1] centered(), [2] mayMirror() rejecting a mirrored pair.
__global__ void countThresholdMismatches(const WhimThreshold threshold,
                                         const int32_t*      a,
                                         const int32_t*      b,
                                         const int           count,
                                         unsigned long long* mismatches) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= count) {
    return;
  }
  const bool expectedMirrored = fabs(referenceUnits(a[idx]) + referenceUnits(b[idx])) <= threshold.value;
  const bool expectedCentered = fabs(referenceUnits(a[idx])) < threshold.value;
  if (threshold.mirrored(a[idx], b[idx]) != expectedMirrored) {
    atomicAdd(&mismatches[0], 1ull);
  }
  if (threshold.centered(a[idx]) != expectedCentered) {
    atomicAdd(&mismatches[1], 1ull);
  }
  if (expectedMirrored && !threshold.mayMirror(a[idx], b[idx])) {
    atomicAdd(&mismatches[2], 1ull);
  }
}

std::vector<unsigned long long> downloadCounts(AsyncDeviceVector<unsigned long long>& counts) {
  std::vector<unsigned long long> host(counts.size());
  counts.copyToHost(host);
  cudaCheckError(cudaStreamSynchronize(counts.stream()));
  return host;
}

}  // namespace

TEST(WhimArithmetic, ThousandthsToUnitsMatchesDivisionForEveryInt32) {
  AsyncDeviceVector<unsigned long long> mismatches(1);
  mismatches.zero();
  countDivisionMismatches<<<1024, 256, 0, mismatches.stream()>>>(mismatches.data());
  cudaCheckError(cudaGetLastError());
  EXPECT_EQ(downloadCounts(mismatches)[0], 0u);
}

TEST(WhimArithmetic, RoundWhimThousandthsMatchesDivision) {
  std::vector<double>                    values = {0.0,
                                                   -0.0,
                                                   0.0004,
                                                   -0.0004,
                                                   0.0005,
                                                   -0.0005,
                                                   2147483.647,
                                                   2147483.6475,
                                                   -2147483.6485,
                                                   1e12,
                                                   -1e12,
                                                   std::numeric_limits<double>::quiet_NaN(),
                                                   std::numeric_limits<double>::infinity(),
                                                   -std::numeric_limits<double>::infinity()};
  std::mt19937_64                        rng(42);
  std::uniform_real_distribution<double> small(-10.0, 10.0);
  std::uniform_real_distribution<double> large(-5e6, 5e6);
  for (int i = 0; i < 1'000'000; ++i) {
    values.push_back(i % 2 == 0 ? small(rng) : large(rng));
  }
  AsyncDeviceVector<double> deviceValues(values.size());
  deviceValues.copyFromHost(values);
  AsyncDeviceVector<unsigned long long> mismatches(1);
  mismatches.zero();
  const int count = static_cast<int>(values.size());
  countRoundingMismatches<<<(count + 255) / 256, 256, 0, mismatches.stream()>>>(deviceValues.data(),
                                                                                count,
                                                                                mismatches.data());
  cudaCheckError(cudaGetLastError());
  EXPECT_EQ(downloadCounts(mismatches)[0], 0u);
}

TEST(WhimArithmetic, ThresholdComparisonsMatchRdkitExpressions) {
  // Pairs whose sums straddle every threshold below, plus extremes and random int32 pairs.
  std::vector<int32_t> a;
  std::vector<int32_t> b;
  for (int32_t value = -20'000; value <= 20'000; ++value) {
    for (int32_t offset = -1'005; offset <= 1'005; offset += (offset >= -15 && offset < 15) ? 1 : 99) {
      a.push_back(value);
      b.push_back(-value + offset);
    }
  }
  constexpr int32_t kMax = std::numeric_limits<int32_t>::max();
  constexpr int32_t kMin = std::numeric_limits<int32_t>::min();
  for (const int32_t extreme : {kMax, kMin, kMax - 10, kMin + 10, 1 << 30, -(1 << 30)}) {
    for (const int32_t other : {kMax, kMin, 0, -extreme / 2, 10, -10}) {
      a.push_back(extreme);
      b.push_back(other);
    }
  }
  std::mt19937                           rng(7);
  std::uniform_int_distribution<int32_t> anyInt(kMin, kMax);
  for (int i = 0; i < 1'000'000; ++i) {
    a.push_back(anyInt(rng));
    b.push_back(anyInt(rng));
  }
  const int                  count = static_cast<int>(a.size());
  AsyncDeviceVector<int32_t> deviceA(a.size());
  AsyncDeviceVector<int32_t> deviceB(b.size());
  deviceA.copyFromHost(a);
  deviceB.copyFromHost(b);

  // On the 0.001 grid (RDKit's default 0.01 among them), just off it, between grid points, and degenerate.
  for (const double value :
       {0.0, 1e-300, 1e-9, 0.001, 0.01, 0.01 + 1e-12, 0.01 - 1e-12, 0.0105, 0.5, 1.0, 1e7, 3e9, 1e300}) {
    SCOPED_TRACE("threshold " + std::to_string(value));
    AsyncDeviceVector<unsigned long long> mismatches(3);
    mismatches.zero();
    countThresholdMismatches<<<(count + 255) / 256, 256, 0, mismatches.stream()>>>(WhimThreshold::make(value),
                                                                                   deviceA.data(),
                                                                                   deviceB.data(),
                                                                                   count,
                                                                                   mismatches.data());
    cudaCheckError(cudaGetLastError());
    const auto counts = downloadCounts(mismatches);
    EXPECT_EQ(counts[0], 0u) << "mirrored()";
    EXPECT_EQ(counts[1], 0u) << "centered()";
    EXPECT_EQ(counts[2], 0u) << "mayMirror() rejected a mirrored pair";
  }
}
