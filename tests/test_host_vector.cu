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
#include <numeric>
#include <vector>

#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"

using namespace nvMolKit;

// Basic constructor and destructor tests
TEST(PinnedHostVector, DefaultConstructor) {
  PinnedHostVector<int> vec;
  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.data(), nullptr);
  EXPECT_TRUE(vec.empty());
}

TEST(PinnedHostVector, SizeConstructor) {
  PinnedHostVector<int> vec(10);
  EXPECT_EQ(vec.size(), 10);
  EXPECT_NE(vec.data(), nullptr);
  EXPECT_FALSE(vec.empty());
}

TEST(PinnedHostVector, SizeValueConstructor) {
  PinnedHostVector<int> vec(5, 42);
  EXPECT_EQ(vec.size(), 5);
  EXPECT_NE(vec.data(), nullptr);
  for (size_t i = 0; i < vec.size(); ++i) {
    EXPECT_EQ(vec[i], 42);
  }
}

TEST(PinnedHostVector, ZeroSizeConstructor) {
  PinnedHostVector<int> vec(0);
  EXPECT_EQ(vec.size(), 0);
  EXPECT_TRUE(vec.empty());
}

// Move semantics tests
TEST(PinnedHostVector, MoveConstructor) {
  PinnedHostVector<int> vec(10);
  int*                  originalData = vec.data();

  PinnedHostVector<int> vec2(std::move(vec));
  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.data(), nullptr);
  EXPECT_EQ(vec2.size(), 10);
  EXPECT_EQ(vec2.data(), originalData);
}

TEST(PinnedHostVector, MoveAssignment) {
  PinnedHostVector<int> vec(10);
  PinnedHostVector<int> vec2(20);
  int*                  originalData = vec.data();

  vec2 = std::move(vec);
  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.data(), nullptr);
  EXPECT_EQ(vec2.size(), 10);
  EXPECT_EQ(vec2.data(), originalData);
}

// Element access tests
TEST(PinnedHostVector, ElementAccess) {
  PinnedHostVector<int> vec(5);
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = static_cast<int>(i * 2);
  }

  for (size_t i = 0; i < vec.size(); ++i) {
    EXPECT_EQ(vec[i], static_cast<int>(i * 2));
  }
}

TEST(PinnedHostVector, Iterators) {
  PinnedHostVector<int> vec(5);
  std::iota(vec.begin(), vec.end(), 1);

  const std::vector<int> expected = {1, 2, 3, 4, 5};
  EXPECT_THAT(std::vector<int>(vec.begin(), vec.end()), ::testing::ElementsAreArray(expected));
}

// Capacity operations tests
TEST(PinnedHostVector, Resize) {
  PinnedHostVector<int> vec(5);
  std::iota(vec.begin(), vec.end(), 1);

  // Resize larger
  vec.resize(10);
  EXPECT_EQ(vec.size(), 10);
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(vec[i], static_cast<int>(i + 1));
  }

  // Resize smaller
  vec.resize(3);
  EXPECT_EQ(vec.size(), 3);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(vec[i], static_cast<int>(i + 1));
  }
}

TEST(PinnedHostVector, ResizeWithValue) {
  PinnedHostVector<int> vec(3);
  std::iota(vec.begin(), vec.end(), 1);

  vec.resize(6, 99);
  EXPECT_EQ(vec.size(), 6);
  EXPECT_THAT(std::vector<int>(vec.begin(), vec.end()), ::testing::ElementsAre(1, 2, 3, 99, 99, 99));
}

TEST(PinnedHostVector, ResizeToZero) {
  PinnedHostVector<int> vec(10);
  vec.resize(0);
  EXPECT_EQ(vec.size(), 0);
  EXPECT_TRUE(vec.empty());
}

TEST(PinnedHostVector, Clear) {
  PinnedHostVector<int> vec(10);
  vec.clear();
  EXPECT_EQ(vec.size(), 0);
  EXPECT_TRUE(vec.empty());
}

// Utility methods tests
TEST(PinnedHostVector, Fill) {
  PinnedHostVector<int> vec(5);
  vec.fill(42);
  for (size_t i = 0; i < vec.size(); ++i) {
    EXPECT_EQ(vec[i], 42);
  }
}

TEST(PinnedHostVector, Zero) {
  PinnedHostVector<int> vec(5);
  vec.fill(123);
  vec.zero();
  for (size_t i = 0; i < vec.size(); ++i) {
    EXPECT_EQ(vec[i], 0);
  }
}

// Compatibility with AsyncDeviceVector tests
TEST(PinnedHostVectorCompatibility, CopyToDevice) {
  PinnedHostVector<int> hostVec(5);
  std::iota(hostVec.begin(), hostVec.end(), 1);

  AsyncDeviceVector<int> deviceVec(5);
  hostVec.copyToDevice(deviceVec);

  std::vector<int> result(5);
  deviceVec.copyToHost(result);

  // Synchronize to ensure async operations complete
  cudaDeviceSynchronize();

  EXPECT_THAT(result, ::testing::ElementsAre(1, 2, 3, 4, 5));
}

TEST(PinnedHostVectorCompatibility, CopyFromDevice) {
  AsyncDeviceVector<int> deviceVec(5);
  std::vector<int>       data = {10, 20, 30, 40, 50};
  deviceVec.copyFromHost(data);

  PinnedHostVector<int> hostVec(5);
  hostVec.copyFromDevice(deviceVec);

  // Synchronize to ensure async operations complete
  cudaDeviceSynchronize();

  EXPECT_THAT(std::vector<int>(hostVec.begin(), hostVec.end()), ::testing::ElementsAre(10, 20, 30, 40, 50));
}

TEST(PinnedHostVectorCompatibility, CopyToDeviceSizeMismatch) {
  PinnedHostVector<int>  hostVec(5);
  AsyncDeviceVector<int> deviceVec(10);

  EXPECT_THROW(hostVec.copyToDevice(deviceVec), std::out_of_range);
}

TEST(PinnedHostVectorCompatibility, CopyFromDeviceSizeMismatch) {
  AsyncDeviceVector<int> deviceVec(5);
  PinnedHostVector<int>  hostVec(10);

  EXPECT_THROW(hostVec.copyFromDevice(deviceVec), std::out_of_range);
}

TEST(PinnedHostVectorCompatibility, StreamSupport) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  {
    PinnedHostVector<int> hostVec(3);
    std::iota(hostVec.begin(), hostVec.end(), 1);

    AsyncDeviceVector<int> deviceVec(3);
    deviceVec.setStream(stream);

    // Test copy operations with custom stream
    hostVec.copyToDevice(deviceVec, stream);

    PinnedHostVector<int> resultVec(3);
    resultVec.copyFromDevice(deviceVec, stream);

    cudaStreamSynchronize(stream);

    EXPECT_THAT(std::vector<int>(resultVec.begin(), resultVec.end()), ::testing::ElementsAre(1, 2, 3));

    // Reset to default stream before destruction to avoid use-after-free
    deviceVec.setStream(nullptr);

    // Explicitly destroy device vector before stream by going out of scope
  }

  cudaStreamDestroy(stream);
}

// Edge cases and error handling
TEST(PinnedHostVector, EmptyVectorOperations) {
  PinnedHostVector<int> vec;

  // These should work on empty vectors
  vec.clear();
  vec.resize(0);
  vec.zero();

  EXPECT_EQ(vec.size(), 0);
  EXPECT_TRUE(vec.empty());
}

TEST(PinnedHostVectorCompatibility, EmptyVectorCompatibility) {
  PinnedHostVector<int>  hostVec;
  AsyncDeviceVector<int> deviceVec(0);

  // These should work with empty vectors
  hostVec.copyToDevice(deviceVec);
  hostVec.copyFromDevice(deviceVec);

  EXPECT_EQ(hostVec.size(), 0);
  EXPECT_EQ(deviceVec.size(), 0);
}

// Test fixture for more complex scenarios
class PinnedHostVectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    hostVec.resize(5);
    std::iota(hostVec.begin(), hostVec.end(), 1);

    deviceVec.resize(5);
    hostVec.copyToDevice(deviceVec);

    // Synchronize to ensure setup operations complete
    cudaDeviceSynchronize();
  }

  PinnedHostVector<int>  hostVec;
  AsyncDeviceVector<int> deviceVec;
};

TEST_F(PinnedHostVectorTest, RoundTripCopy) {
  PinnedHostVector<int> resultVec(5);

  // Copy host -> device -> host
  hostVec.copyToDevice(deviceVec);
  resultVec.copyFromDevice(deviceVec);

  // Synchronize to ensure async operations complete
  cudaDeviceSynchronize();

  EXPECT_THAT(std::vector<int>(resultVec.begin(), resultVec.end()), ::testing::ElementsAre(1, 2, 3, 4, 5));
}

TEST_F(PinnedHostVectorTest, ModifyAndCopy) {
  // Modify host vector
  hostVec[2] = 99;

  // Copy to device and back
  hostVec.copyToDevice(deviceVec);

  PinnedHostVector<int> resultVec(5);
  resultVec.copyFromDevice(deviceVec);

  // Synchronize to ensure async operations complete
  cudaDeviceSynchronize();

  EXPECT_THAT(std::vector<int>(resultVec.begin(), resultVec.end()), ::testing::ElementsAre(1, 2, 99, 4, 5));
}

// Performance/memory tests
TEST(PinnedHostVector, LargeVectorAllocation) {
  constexpr size_t      largeSize = 1000000;
  PinnedHostVector<int> vec(largeSize);

  EXPECT_EQ(vec.size(), largeSize);
  EXPECT_NE(vec.data(), nullptr);

  // Test that we can actually use the memory
  vec.fill(42);
  EXPECT_EQ(vec[0], 42);
  EXPECT_EQ(vec[largeSize - 1], 42);
}

TEST(PinnedHostVector, MultipleResizes) {
  PinnedHostVector<int> vec(10);
  std::iota(vec.begin(), vec.end(), 1);

  // Multiple resize operations
  vec.resize(50, 0);
  EXPECT_EQ(vec[1], 2);
  EXPECT_EQ(vec[30], 0);
  EXPECT_EQ(vec.size(), 50);
}
