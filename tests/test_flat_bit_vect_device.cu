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

#include "src/data_structures/flat_bit_vect.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"

using nvMolKit::AsyncDeviceVector;
using nvMolKit::BitMatrix2DView;
using nvMolKit::checkReturnCode;
using nvMolKit::FlatBitVect;
using nvMolKit::ScopedStream;

// =============================================================================
// FlatBitVect Device Tests
// =============================================================================

__global__ void setBitKernel(FlatBitVect<64>* fbv, int bitIdx, bool value) {
  fbv->setBit(bitIdx, value);
}

__global__ void readBitKernel(const FlatBitVect<64>* fbv, int bitIdx, uint8_t* result) {
  *result = (*fbv)[bitIdx];
}

__global__ void clearKernel(FlatBitVect<64>* fbv) {
  fbv->clear();
}

template <std::size_t NBits> __global__ void setBitAtomicKernel(FlatBitVect<NBits>* fbv) {
  const std::size_t bitIdx = static_cast<std::size_t>(threadIdx.x);
  if (bitIdx < NBits) {
    fbv->setBitAtomic(bitIdx);
  }
}

template <std::size_t NBits> __global__ void clearParallelKernel(FlatBitVect<NBits>* fbv) {
  fbv->clearParallel(threadIdx.x, blockDim.x);
  __syncthreads();
}

TEST(FlatBitVectDevice, SetAndReadBit) {
  ScopedStream stream;

  FlatBitVect<64>                    hostFbv(false);
  AsyncDeviceVector<FlatBitVect<64>> deviceFbv(1, stream.stream());
  deviceFbv.setFromVector(std::vector<FlatBitVect<64>>{hostFbv});

  AsyncDeviceVector<uint8_t> resultDev(1, stream.stream());

  // Set bit 37
  setBitKernel<<<1, 1, 0, stream.stream()>>>(deviceFbv.data(), 37, true);
  cudaCheckError(cudaGetLastError());

  // Read bit 37
  readBitKernel<<<1, 1, 0, stream.stream()>>>(deviceFbv.data(), 37, resultDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<uint8_t> result(1);
  resultDev.copyToHost(result);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_TRUE(result[0]);

  // Read bit 36 (should be false)
  readBitKernel<<<1, 1, 0, stream.stream()>>>(deviceFbv.data(), 36, resultDev.data());
  cudaCheckError(cudaGetLastError());
  resultDev.copyToHost(result);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_FALSE(result[0]);
}

TEST(FlatBitVectDevice, ClearOnDevice) {
  ScopedStream stream;

  FlatBitVect<64>                    hostFbv(true);  // All bits set
  AsyncDeviceVector<FlatBitVect<64>> deviceFbv(1, stream.stream());
  deviceFbv.setFromVector(std::vector<FlatBitVect<64>>{hostFbv});

  // Clear on device
  clearKernel<<<1, 1, 0, stream.stream()>>>(deviceFbv.data());
  cudaCheckError(cudaGetLastError());

  // Copy back and verify
  std::vector<FlatBitVect<64>> hostResult(1);
  deviceFbv.copyToHost(hostResult);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < 64; ++i) {
    EXPECT_FALSE(hostResult[0][i]) << "Bit " << i << " should be cleared";
  }
}

TEST(FlatBitVectDevice, AtomicSetBit) {
  constexpr std::size_t kBits = 64;
  ScopedStream          stream;

  FlatBitVect<kBits>                    hostFbv(false);
  AsyncDeviceVector<FlatBitVect<kBits>> deviceFbv(1, stream.stream());
  deviceFbv.setFromVector(std::vector<FlatBitVect<kBits>>{hostFbv});

  constexpr int kThreads = static_cast<int>(kBits);
  setBitAtomicKernel<kBits><<<1, kThreads, 0, stream.stream()>>>(deviceFbv.data());
  cudaCheckError(cudaGetLastError());

  std::vector<FlatBitVect<kBits>> hostResult(1);
  deviceFbv.copyToHost(hostResult);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < static_cast<int>(kBits); ++i) {
    EXPECT_TRUE(hostResult[0][i]) << "Bit " << i << " should be set";
  }
}

TEST(FlatBitVectDevice, ParallelClear) {
  constexpr std::size_t kBits = 128;
  ScopedStream          stream;

  FlatBitVect<kBits>                    hostFbv(true);
  AsyncDeviceVector<FlatBitVect<kBits>> deviceFbv(1, stream.stream());
  deviceFbv.setFromVector(std::vector<FlatBitVect<kBits>>{hostFbv});

  constexpr int kThreads = 64;
  clearParallelKernel<kBits><<<1, kThreads, 0, stream.stream()>>>(deviceFbv.data());
  cudaCheckError(cudaGetLastError());

  std::vector<FlatBitVect<kBits>> hostResult(1);
  deviceFbv.copyToHost(hostResult);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < static_cast<int>(kBits); ++i) {
    EXPECT_FALSE(hostResult[0][i]) << "Bit " << i << " should be cleared";
  }
}

// =============================================================================
// BitMatrix2DView Device Tests
// =============================================================================

template <std::size_t Rows, std::size_t Cols>
__global__ void setMatrix2DKernel(FlatBitVect<Rows * Cols>* storage, int row, int col, bool value) {
  BitMatrix2DView<Rows, Cols> view(storage);
  view.set(row, col, value);
}

template <std::size_t Rows, std::size_t Cols>
__global__ void getMatrix2DKernel(FlatBitVect<Rows * Cols>* storage, int row, int col, uint8_t* result) {
  BitMatrix2DView<Rows, Cols> view(storage);
  *result = view.get(row, col);
}

template <std::size_t Rows, std::size_t Cols> __global__ void clearMatrix2DKernel(FlatBitVect<Rows * Cols>* storage) {
  BitMatrix2DView<Rows, Cols> view(storage);
  view.clear();
}

template <std::size_t Rows, std::size_t Cols>
__global__ void setMatrix2DAtomicKernel(FlatBitVect<Rows * Cols>* storage) {
  BitMatrix2DView<Rows, Cols> view(storage);
  const std::size_t           linearIdx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linearIdx < Rows * Cols) {
    const std::size_t row = linearIdx / Cols;
    const std::size_t col = linearIdx % Cols;
    view.setAtomic(row, col);
  }
}

template <std::size_t Rows, std::size_t Cols>
__global__ void clearMatrix2DParallelKernel(FlatBitVect<Rows * Cols>* storage) {
  BitMatrix2DView<Rows, Cols> view(storage);
  view.clearParallel(threadIdx.x, blockDim.x);
  __syncthreads();
}

TEST(BitMatrix2DViewDevice, SetAndGet) {
  constexpr std::size_t kRows = 8;
  constexpr std::size_t kCols = 4;

  ScopedStream stream;

  FlatBitVect<kRows * kCols>                    hostStorage(false);
  AsyncDeviceVector<FlatBitVect<kRows * kCols>> deviceStorage(1, stream.stream());
  deviceStorage.setFromVector(std::vector<FlatBitVect<kRows * kCols>>{hostStorage});

  AsyncDeviceVector<uint8_t> resultDev(1, stream.stream());

  // Set bit at (3, 2)
  setMatrix2DKernel<kRows, kCols><<<1, 1, 0, stream.stream()>>>(deviceStorage.data(), 3, 2, true);
  cudaCheckError(cudaGetLastError());

  // Read bit at (3, 2)
  getMatrix2DKernel<kRows, kCols><<<1, 1, 0, stream.stream()>>>(deviceStorage.data(), 3, 2, resultDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<uint8_t> result(1);
  resultDev.copyToHost(result);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_TRUE(result[0]);

  // Read bit at (3, 1) - should be false
  getMatrix2DKernel<kRows, kCols><<<1, 1, 0, stream.stream()>>>(deviceStorage.data(), 3, 1, resultDev.data());
  cudaCheckError(cudaGetLastError());
  resultDev.copyToHost(result);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_FALSE(result[0]);

  // Read bit at (2, 2) - should be false
  getMatrix2DKernel<kRows, kCols><<<1, 1, 0, stream.stream()>>>(deviceStorage.data(), 2, 2, resultDev.data());
  cudaCheckError(cudaGetLastError());
  resultDev.copyToHost(result);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_FALSE(result[0]);
}

TEST(BitMatrix2DViewDevice, LinearIndexing) {
  constexpr std::size_t kRows = 8;
  constexpr std::size_t kCols = 4;

  // Verify linear indexing is row-major
  EXPECT_EQ((BitMatrix2DView<kRows, kCols>::linearIndex(0, 0)), 0);
  EXPECT_EQ((BitMatrix2DView<kRows, kCols>::linearIndex(0, 1)), 1);
  EXPECT_EQ((BitMatrix2DView<kRows, kCols>::linearIndex(0, 3)), 3);
  EXPECT_EQ((BitMatrix2DView<kRows, kCols>::linearIndex(1, 0)), 4);
  EXPECT_EQ((BitMatrix2DView<kRows, kCols>::linearIndex(2, 0)), 8);
  EXPECT_EQ((BitMatrix2DView<kRows, kCols>::linearIndex(7, 3)), 31);
}

TEST(BitMatrix2DViewDevice, Clear) {
  constexpr std::size_t kRows = 4;
  constexpr std::size_t kCols = 4;

  ScopedStream stream;

  FlatBitVect<kRows * kCols>                    hostStorage(true);  // All bits set
  AsyncDeviceVector<FlatBitVect<kRows * kCols>> deviceStorage(1, stream.stream());
  deviceStorage.setFromVector(std::vector<FlatBitVect<kRows * kCols>>{hostStorage});

  // Clear on device
  clearMatrix2DKernel<kRows, kCols><<<1, 1, 0, stream.stream()>>>(deviceStorage.data());
  cudaCheckError(cudaGetLastError());

  // Copy back and verify
  std::vector<FlatBitVect<kRows * kCols>> hostResult(1);
  deviceStorage.copyToHost(hostResult);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (size_t i = 0; i < kRows * kCols; ++i) {
    EXPECT_FALSE(hostResult[0][i]) << "Bit " << i << " should be cleared";
  }
}

TEST(BitMatrix2DViewDevice, AtomicSet) {
  constexpr std::size_t kRows = 8;
  constexpr std::size_t kCols = 8;

  ScopedStream stream;

  FlatBitVect<kRows * kCols>                    hostStorage(false);
  AsyncDeviceVector<FlatBitVect<kRows * kCols>> deviceStorage(1, stream.stream());
  deviceStorage.setFromVector(std::vector<FlatBitVect<kRows * kCols>>{hostStorage});

  constexpr int kThreads = 64;
  setMatrix2DAtomicKernel<kRows, kCols><<<1, kThreads, 0, stream.stream()>>>(deviceStorage.data());
  cudaCheckError(cudaGetLastError());

  std::vector<FlatBitVect<kRows * kCols>> hostResult(1);
  deviceStorage.copyToHost(hostResult);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  BitMatrix2DView<kRows, kCols> view(hostResult[0]);
  for (int row = 0; row < static_cast<int>(kRows); ++row) {
    for (int col = 0; col < static_cast<int>(kCols); ++col) {
      EXPECT_TRUE(view.get(row, col)) << "Bit (" << row << ", " << col << ") should be set";
    }
  }
}

TEST(BitMatrix2DViewDevice, ParallelClear) {
  constexpr std::size_t kRows = 8;
  constexpr std::size_t kCols = 8;

  ScopedStream stream;

  FlatBitVect<kRows * kCols>                    hostStorage(true);
  AsyncDeviceVector<FlatBitVect<kRows * kCols>> deviceStorage(1, stream.stream());
  deviceStorage.setFromVector(std::vector<FlatBitVect<kRows * kCols>>{hostStorage});

  constexpr int kThreads = 64;
  clearMatrix2DParallelKernel<kRows, kCols><<<1, kThreads, 0, stream.stream()>>>(deviceStorage.data());
  cudaCheckError(cudaGetLastError());

  std::vector<FlatBitVect<kRows * kCols>> hostResult(1);
  deviceStorage.copyToHost(hostResult);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < static_cast<int>(kRows * kCols); ++i) {
    EXPECT_FALSE(hostResult[0][i]) << "Bit " << i << " should be cleared";
  }
}

// Test large matrix
TEST(BitMatrix2DViewDevice, LargeMatrix) {
  constexpr std::size_t kRows = 64;
  constexpr std::size_t kCols = 32;

  ScopedStream stream;

  FlatBitVect<kRows * kCols>                    hostStorage(false);
  AsyncDeviceVector<FlatBitVect<kRows * kCols>> deviceStorage(1, stream.stream());
  deviceStorage.setFromVector(std::vector<FlatBitVect<kRows * kCols>>{hostStorage});

  // Set a pattern: every (row, col) where row == col * 2
  for (size_t col = 0; col < kCols; ++col) {
    size_t row = col * 2;
    if (row < kRows) {
      setMatrix2DKernel<kRows, kCols><<<1, 1, 0, stream.stream()>>>(deviceStorage.data(), row, col, true);
    }
  }
  cudaCheckError(cudaGetLastError());

  // Verify on host
  std::vector<FlatBitVect<kRows * kCols>> hostResult(1);
  deviceStorage.copyToHost(hostResult);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  BitMatrix2DView<kRows, kCols> view(hostResult[0]);
  for (size_t row = 0; row < kRows; ++row) {
    for (size_t col = 0; col < kCols; ++col) {
      bool expected = (row == col * 2);
      EXPECT_EQ(view.get(row, col), expected) << "Mismatch at (" << row << ", " << col << ")";
    }
  }
}