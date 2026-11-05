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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "src/utils/cccl_compat.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

using nvMolKit::checkReturnCode;
using nvMolKit::compat::ceil_div;
using nvMolKit::compat::detail::ceilDivFallback;

namespace {

static_assert(std::is_same_v<decltype(ceil_div(int{}, int{})), int>);
static_assert(std::is_same_v<decltype(ceil_div(std::size_t{}, int{})), std::size_t>);
static_assert(std::is_same_v<decltype(ceil_div(int{}, std::uint32_t{})), std::uint32_t>);
static_assert(std::is_same_v<decltype(ceilDivFallback(int{}, int{})), int>);
static_assert(std::is_same_v<decltype(ceilDivFallback(std::size_t{}, int{})), std::size_t>);
static_assert(std::is_same_v<decltype(ceilDivFallback(int{}, std::uint32_t{})), std::uint32_t>);
static_assert(ceil_div(0, 256) == 0);
static_assert(ceil_div(256, 256) == 1);
static_assert(ceil_div(257, 256) == 2);
static_assert(ceilDivFallback(0, 256) == 0);
static_assert(ceilDivFallback(257, 256) == 2);

__global__ void ceilDivKernel(const int* dividends, const int* divisors, const int count, int* compat, int* fallback) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < count) {
    compat[idx]   = ceil_div(dividends[idx], divisors[idx]);
    fallback[idx] = ceilDivFallback(dividends[idx], divisors[idx]);
  }
}

}  // namespace

TEST(CcclCompat, CeilDivMatchesReferenceOnHost) {
  for (int b = 1; b <= 70; ++b) {
    for (int a = 0; a <= 1000; ++a) {
      const int expected = a / b + (a % b != 0 ? 1 : 0);
      ASSERT_EQ(ceil_div(a, b), expected) << a << " / " << b;
      ASSERT_EQ(ceilDivFallback(a, b), expected) << a << " / " << b;
    }
  }
}

TEST(CcclCompat, CeilDivDoesNotOverflowNearTypeMax) {
  constexpr std::uint32_t kMax = std::numeric_limits<std::uint32_t>::max();
  EXPECT_EQ(ceil_div(kMax, 2U), kMax / 2U + 1U);
  EXPECT_EQ(ceilDivFallback(kMax, 2U), kMax / 2U + 1U);
  constexpr int kIntMax = std::numeric_limits<int>::max();
  EXPECT_EQ(ceil_div(kIntMax, 256), kIntMax / 256 + 1);
  EXPECT_EQ(ceilDivFallback(kIntMax, 256), kIntMax / 256 + 1);
}

TEST(CcclCompat, CeilDivMatchesReferenceOnDevice) {
  std::vector<int> dividends;
  std::vector<int> divisors;
  for (int b = 1; b <= 70; ++b) {
    for (int a = 0; a <= 300; ++a) {
      dividends.push_back(a);
      divisors.push_back(b);
    }
  }
  dividends.push_back(std::numeric_limits<int>::max());
  divisors.push_back(256);
  const int count = static_cast<int>(dividends.size());

  nvMolKit::AsyncDeviceVector<int> deviceDividends(count);
  nvMolKit::AsyncDeviceVector<int> deviceDivisors(count);
  nvMolKit::AsyncDeviceVector<int> deviceCompat(count);
  nvMolKit::AsyncDeviceVector<int> deviceFallback(count);
  deviceDividends.copyFromHost(dividends);
  deviceDivisors.copyFromHost(divisors);
  constexpr int kBlockSize = 256;
  ceilDivKernel<<<ceil_div(count, kBlockSize), kBlockSize>>>(deviceDividends.data(),
                                                             deviceDivisors.data(),
                                                             count,
                                                             deviceCompat.data(),
                                                             deviceFallback.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> compat(count);
  std::vector<int> fallback(count);
  deviceCompat.copyToHost(compat);
  deviceFallback.copyToHost(fallback);
  cudaCheckError(cudaDeviceSynchronize());
  for (int i = 0; i < count; ++i) {
    const int expected = dividends[i] / divisors[i] + (dividends[i] % divisors[i] != 0 ? 1 : 0);
    ASSERT_EQ(compat[i], expected) << dividends[i] << " / " << divisors[i];
    ASSERT_EQ(fallback[i], expected) << dividends[i] << " / " << divisors[i];
  }
}
