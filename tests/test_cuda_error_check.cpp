// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "src/utils/cuda_error_check.h"

using namespace nvMolKit;

TEST(CudaErrorCheck, NoThrowOnGoodCode) {
  cudaCheckError(cudaSuccess);  // NOLINT
}

TEST(CudaErrorCheck, ThrowOnBadCode) {  // NOLINT(readability-function-cognitive-complexity)
  EXPECT_THROW(
    {
      try {
        cudaCheckError(cudaErrorInvalidValue);
      } catch (const CudaBadReturnCode& e) {
        EXPECT_EQ(e.rc(), cudaErrorInvalidValue);
        throw;
      }
    },
    CudaBadReturnCode);
}

TEST(CudaErrorCheck, NoThrowOnGoodCodeNoThrow) {
  cudaCheckErrorNoThrow(cudaSuccess);
}

TEST(CudaErrorCheck, NoThrowOnBadCodeNoThrow) {
  cudaCheckErrorNoThrow(cudaErrorInvalidValue);
}

TEST(CudaErrorCheck, ThrowClearsLastError) {
  void* ptr = nullptr;
  EXPECT_THROW(cudaCheckError(cudaMalloc(&ptr, size_t{1} << 62)), CudaBadReturnCode);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(CudaErrorCheck, NoThrowClearsLastError) {
  void* ptr = nullptr;
  cudaCheckErrorNoThrow(cudaMalloc(&ptr, size_t{1} << 62));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
