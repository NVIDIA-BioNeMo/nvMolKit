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

#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

using namespace nvMolKit;

// Add device_vector tests here
TEST(AsyncDeviceVector, Constructor) {
  nvMolKit::AsyncDeviceVector<int> vec(10);
  EXPECT_EQ(vec.size(), 10);
}

TEST(AsyncDeviceVector, MoveConstructor) {
  nvMolKit::AsyncDeviceVector<int> vec(10);
  nvMolKit::AsyncDeviceVector<int> vec2(std::move(vec));
  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.data(), nullptr);
  EXPECT_EQ(vec2.size(), 10);
  EXPECT_NE(vec2.data(), nullptr);
}

TEST(AsyncDeviceVector, MoveAssignment) {
  nvMolKit::AsyncDeviceVector<int> vec(10);
  nvMolKit::AsyncDeviceVector<int> vec2(20);
  vec2 = std::move(vec);
  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.data(), nullptr);
  EXPECT_EQ(vec2.size(), 10);
  EXPECT_NE(vec2.data(), nullptr);
}

TEST(AsyncDeviceVector, EmptyAssign) {
  nvMolKit::AsyncDeviceVector<int> vec(0);
  EXPECT_EQ(vec.size(), 0);
}

TEST(AsyncDeviceVector, GoodData) {
  nvMolKit::AsyncDeviceVector<int> vec(10);
  cudaCheckError(cudaMemset(vec.data(), 0, 10 * sizeof(int)));
  int* data = vec.data();
  EXPECT_NE(data, nullptr);
}

TEST(AsyncDeviceVector, Zero) {
  nvMolKit::AsyncDeviceVector<int> vec(10);
  cudaCheckError(cudaMemset(vec.data(), 1, 10 * sizeof(int)));
  vec.zero();
  int* data             = vec.data();
  int  returnedData[10] = {0};
  cudaCheckError(cudaMemcpy(returnedData, data, 10 * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(returnedData[i], 0);
  }
}

TEST(AsyncDeviceVector, CopyFromHostZero) {
  nvMolKit::AsyncDeviceVector<int> vec(10);
  vec.zero();
  int data[10] = {1};
  vec.copyFromHost(data, 0);
  int* deviceData       = vec.data();
  int  returnedData[10] = {0};
  cudaMemcpy(returnedData, deviceData, 10 * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(returnedData[i], 0);
  }
}

TEST(AsyncDeviceVector, CopyFromHostThrows) {
  nvMolKit::AsyncDeviceVector<int> vec(10);
  int                              data[12] = {1};
  EXPECT_THROW(vec.copyFromHost(data, 12), std::out_of_range);
  EXPECT_THROW(vec.copyFromHost(data, 9, 0, 2), std::out_of_range);
}

TEST(AsyncDeviceVector, CopyFromHost) {
  nvMolKit::AsyncDeviceVector<int> vec(10);
  cudaMemset(vec.data(), 0, 10 * sizeof(int));
  std::vector<int> data = {1, 2, 3, 4, 5};
  // Vec should now have {0, 0 2, 3, 0}
  vec.copyFromHost(data.data(), 2, /*firstElementHost=*/1, /*firstElementDevice=*/2);
  int*             deviceData   = vec.data();
  std::vector<int> returnedData = {-1, -1, -1, -1, -1};
  cudaMemcpy(returnedData.data(), deviceData, 5 * sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_THAT(returnedData, ::testing::ElementsAre(0, 0, 2, 3, 0));
}

TEST(AsyncDeviceVector, CopyFromHostVectorThrows) {
  nvMolKit::AsyncDeviceVector<int> vec(10);
  std::vector<int>                 data = {1, 2, 3, 4, 5};
  EXPECT_THROW(vec.copyFromHost(data, 6), std::out_of_range);
  EXPECT_THROW(vec.copyFromHost(data, 5, 1), std::out_of_range);
}

TEST(AsyncDeviceVector, CopyFromHostVector) {
  nvMolKit::AsyncDeviceVector<int> vec(5);
  vec.zero();
  std::vector<int> data = {1, 2, 3, 4, 5};
  vec.copyFromHost(data, 3, /*firstElementHost=*/1, /*firstElementDevice=*/2);
  int*             deviceData   = vec.data();
  std::vector<int> returnedData = {-1, -1, -1, -1, -1};
  cudaMemcpy(returnedData.data(), deviceData, 5 * sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_THAT(returnedData, ::testing::ElementsAre(0, 0, 2, 3, 4));
}

TEST(AsyncDeviceVector, CopyFromHostFull) {
  nvMolKit::AsyncDeviceVector<int> vec(5);
  std::vector<int>                 data = {1, 2, 3, 4, 5};
  vec.copyFromHost(data);
  int*             deviceData   = vec.data();
  std::vector<int> returnedData = {-1, -1, -1, -1, -1};
  cudaMemcpy(returnedData.data(), deviceData, 5 * sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_THAT(returnedData, ::testing::ElementsAre(1, 2, 3, 4, 5));
}

TEST(AsyncDeviceVector, CopyToHostZero) {
  nvMolKit::AsyncDeviceVector<int> vec(5);
  std::vector<int>                 data       = {1, 1, 1, 1, 1};
  int*                             deviceData = vec.data();
  cudaMemset(deviceData, 0, 5 * sizeof(int));
  vec.copyToHost(data.data(), 0);
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(data[i], 1);
  }
}

TEST(AsyncDeviceVector, CopyToHostThrows) {
  nvMolKit::AsyncDeviceVector<int> vec(10);
  int                              data[12] = {1};
  EXPECT_THROW(vec.copyToHost(data, 12), std::out_of_range);
  EXPECT_THROW(vec.copyToHost(data, 9, 0, 2), std::out_of_range);
}

TEST(AsyncDeviceVector, CopyToHost) {
  nvMolKit::AsyncDeviceVector<int> vec(10);
  std::vector<int>                 data = {1, 2, 3, 4, 5};
  cudaMemcpy(vec.data(), data.data(), 5 * sizeof(int), cudaMemcpyHostToDevice);

  std::vector<int> returnedData = {-1, -1, -1, -1, -1};
  vec.copyToHost(returnedData.data(), 3, /*firstElementHost=*/1, /*firstElementDevice=*/2);
  EXPECT_THAT(returnedData, ::testing::ElementsAre(-1, 3, 4, 5, -1));
}

TEST(AsyncDeviceVector, CopyToHostVectorThrows) {
  nvMolKit::AsyncDeviceVector<int> vec(10);
  std::vector<int>                 data = {1, 2, 3, 4, 5};
  EXPECT_THROW(vec.copyToHost(data, 6), std::out_of_range);
  EXPECT_THROW(vec.copyToHost(data, 5, 1), std::out_of_range);
}

TEST(AsyncDeviceVector, CopyToHostVector) {
  nvMolKit::AsyncDeviceVector<int> vec(10);
  std::vector<int>                 data = {1, 2, 3, 4, 5};
  cudaMemcpy(vec.data(), data.data(), 5 * sizeof(int), cudaMemcpyHostToDevice);

  std::vector<int> returnedData = {-1, -1, -1, -1, -1};
  vec.copyToHost(returnedData, 3, /*firstElementHost=*/1, /*firstElementDevice=*/2);
  EXPECT_THAT(returnedData, ::testing::ElementsAre(-1, 3, 4, 5, -1));
}

TEST(AsyncDeviceVector, CopyToHostFull) {
  nvMolKit::AsyncDeviceVector<int> vec(5);
  std::vector<int>                 data = {1, 2, 3, 4, 5};
  cudaMemcpy(vec.data(), data.data(), 5 * sizeof(int), cudaMemcpyHostToDevice);

  std::vector<int> returnedData = {-1, -1, -1, -1, -1};
  vec.copyToHost(returnedData);
  EXPECT_THAT(returnedData, ::testing::ElementsAre(1, 2, 3, 4, 5));
}

TEST(AsyncDevicePtr, UninitializedConstructor) {
  nvMolKit::AsyncDevicePtr<int> ptr;
  cudaDeviceSynchronize();
  EXPECT_NE(ptr.data(), nullptr);
}

TEST(AsyncDevicePtr, ConstructorAndSet) {
  nvMolKit::AsyncDevicePtr<int> ptr(10);
  int                           result = -1;
  cudaMemcpy(&result, ptr.data(), sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_EQ(result, 10);
  ptr.set(20);
  cudaMemcpy(&result, ptr.data(), sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_EQ(result, 20);
}

TEST(AsyncDevicePtr, MoveConstructor) {
  nvMolKit::AsyncDevicePtr<int> ptr(10);
  nvMolKit::AsyncDevicePtr<int> ptr2(std::move(ptr));
  EXPECT_EQ(ptr.data(), nullptr);
  int result = -1;
  cudaMemcpy(&result, ptr2.data(), sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_EQ(result, 10);
}

TEST(AsyncDeviceVector, Release) {
  nvMolKit::AsyncDeviceVector<int> ptr(2, 0);
  ptr.zero();

  int* data = ptr.release();
  EXPECT_EQ(ptr.data(), nullptr);

  std::vector<int> result(2, -1);
  cudaMemcpy(result.data(), data, 2 * sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_THAT(result, ::testing::ElementsAre(0, 0));
  cudaFree(data);
}

TEST(AsyncDeviceVector, resizeEmpty) {
  nvMolKit::AsyncDeviceVector<int> vec;
  vec.resize(0);
  EXPECT_EQ(vec.size(), 0);
}

TEST(AsyncDeviceVector, StartEmptyWithExplicitConstructor) {
  AsyncDeviceVector<int> vec(0);
  EXPECT_EQ(vec.size(), 0);
  vec.resize(5);
  EXPECT_EQ(vec.size(), 5);
}

class AsyncDeviceVectorResizeTests : public ::testing::Test {
 protected:
  void SetUp() override {
    vec  = nvMolKit::AsyncDeviceVector<int>(5);
    data = {1, 2, 3, 4, 5};
    vec.copyFromHost(data);
  }

  std::vector<int> getFirstElements() {
    std::vector<int> result(5, -1);
    EXPECT_EQ(cudaMemcpy(result.data(), vec.data(), 5 * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    return result;
  }

  std::vector<int>                 data;
  nvMolKit::AsyncDeviceVector<int> vec;
};

TEST_F(AsyncDeviceVectorResizeTests, resizeSelf) {
  vec.resize(10);
  EXPECT_EQ(vec.size(), 10);
  std::vector<int> result = getFirstElements();
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(result[i], i + 1);
  }
}

TEST_F(AsyncDeviceVectorResizeTests, resizeLarger) {
  vec.resize(20);
  EXPECT_EQ(vec.size(), 20);
  std::vector<int> result = getFirstElements();
  EXPECT_THAT(result, ::testing::ElementsAre(1, 2, 3, 4, 5));
}

TEST_F(AsyncDeviceVectorResizeTests, resizeSmaller) {
  vec.resize(3);
  EXPECT_EQ(vec.size(), 3);
  std::vector<int> res(3, -1);
  vec.copyToHost(res);
  EXPECT_THAT(res, ::testing::ElementsAre(1, 2, 3));
}

TEST_F(AsyncDeviceVectorResizeTests, resizeZero) {
  vec.resize(0);
  EXPECT_EQ(vec.size(), 0);
}

TEST(AsyncDeviceVector, setFromVectorEmpty) {
  std::vector<int>                 empty;
  nvMolKit::AsyncDeviceVector<int> vec;
  vec.setFromVector(empty);
  EXPECT_EQ(vec.size(), 0);
}

TEST(AsyncDeviceVector, setFromVector) {
  std::vector<int>                 val = {1, 2, 3};
  nvMolKit::AsyncDeviceVector<int> vec;
  vec.setFromVector(val);
  EXPECT_EQ(vec.size(), 3);
  std::vector<int> result(3, -1);
  vec.copyToHost(result);
  EXPECT_THAT(result, ::testing::ElementsAre(1, 2, 3));
}

TEST(AsyncDeviceVector, setFromArrayEmpty) {
  int                              empty[] = {};
  nvMolKit::AsyncDeviceVector<int> vec;
  vec.setFromArray(empty, 0);
  EXPECT_EQ(vec.size(), 0);
}

TEST(AsyncDeviceVector, setFromArray) {
  int                              val[] = {1, 2, 3};
  nvMolKit::AsyncDeviceVector<int> vec;
  vec.setFromArray(val, 3);
  EXPECT_EQ(vec.size(), 3);
  std::vector<int> result(3, -1);
  vec.copyToHost(result);
  EXPECT_THAT(result, ::testing::ElementsAre(1, 2, 3));
}