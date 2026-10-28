// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <stdexcept>

#include "src/substruct/pattern_screen.h"
#include "src/utils/cub_helpers.cuh"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit {

namespace {

constexpr int kScreenBlockSize = 256;
constexpr int kSlicesPerStep   = 16;

// One thread per 32-target word: AND the query's bit slices, rarest first,
// stopping as soon as no target in the word survives.
__global__ void intersectPatternSlicesKernel(const std::uint32_t* __restrict__ slices,
                                             std::size_t sliceWords,
                                             int         numTargets,
                                             const std::uint16_t* __restrict__ queryBits,
                                             int numQueryBits,
                                             std::uint32_t* __restrict__ survivors) {
  const std::size_t word = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (word >= sliceWords) {
    return;
  }
  const int     remaining = numTargets - static_cast<int>(word * 32);
  std::uint32_t alive     = remaining >= 32 ? ~std::uint32_t{0} : ((std::uint32_t{1} << remaining) - 1);
  // Issue several independent slice loads per step: the loop is bound by load
  // latency, not bandwidth, and the early exit only needs checking per step.
  for (int index = 0; index < numQueryBits && alive != 0; index += kSlicesPerStep) {
    std::uint32_t step = ~std::uint32_t{0};
#pragma unroll
    for (int offset = 0; offset < kSlicesPerStep; ++offset) {
      if (index + offset < numQueryBits) {
        step &= __ldg(&slices[static_cast<std::size_t>(queryBits[index + offset]) * sliceWords + word]);
      }
    }
    alive &= step;
  }
  survivors[word] = alive;
}

struct SurvivorPredicate {
  const std::uint32_t* survivors;
  const int*           batchAtomStarts;
  int                  numQueryAtoms;

  __device__ __forceinline__ bool operator()(int target) const {
    if (((survivors[target >> 5] >> (target & 31)) & 1U) == 0) {
      return false;
    }
    return batchAtomStarts[target + 1] - batchAtomStarts[target] >= numQueryAtoms;
  }
};

std::size_t selectTempBytes(int numTargets, cudaStream_t stream) {
  std::size_t tempBytes = 0;
  cudaCheckError(cub::DeviceSelect::If(nullptr,
                                       tempBytes,
                                       thrust::counting_iterator<int>(0),
                                       static_cast<int*>(nullptr),
                                       static_cast<int*>(nullptr),
                                       numTargets,
                                       SurvivorPredicate{},
                                       stream));
  return tempBytes;
}

// The workspace's stream and buffers belong to the current device, so it must be the requested one.
int requireCurrentDevice(int deviceId) {
  int currentDevice = 0;
  cudaCheckError(cudaGetDevice(&currentDevice));
  if (currentDevice != deviceId) {
    throw std::invalid_argument("PatternScreenWorkspace must be constructed with its device current");
  }
  return deviceId;
}

}  // namespace

PatternScreenWorkspace::PatternScreenWorkspace(int deviceId)
    : deviceId_(requireCurrentDevice(deviceId)),
      stream_("pattern screen"),
      hostCount_(1) {
  indices_.setStream(stream_.stream());
  count_.setStream(stream_.stream());
  survivors_.setStream(stream_.stream());
  queryBits_.setStream(stream_.stream());
  tempStorage_.setStream(stream_.stream());
  count_.resize(1);
}

std::size_t PatternScreenWorkspace::estimateDeviceBytes(std::size_t numTargets) {
  if (numTargets == 0) {
    return 0;
  }
  return numTargets * sizeof(int) + patternSliceWords(numTargets) * sizeof(std::uint32_t) +
         kPatternFingerprintBits * sizeof(std::uint16_t) + sizeof(int) +
         selectTempBytes(static_cast<int>(numTargets), nullptr);
}

void PatternScreenWorkspace::reserve(std::size_t numTargets, std::size_t numQueryBits) {
  if (indices_.size() < numTargets) {
    indices_.resize(numTargets);
  }
  if (survivors_.size() < patternSliceWords(numTargets)) {
    survivors_.resize(patternSliceWords(numTargets));
  }
  if (tempTargets_ < numTargets) {
    const std::size_t tempBytes = selectTempBytes(static_cast<int>(numTargets), stream_.stream());
    if (tempStorage_.size() < tempBytes) {
      tempStorage_.resize(tempBytes);
    }
    tempTargets_ = numTargets;
  }
  const std::size_t bitCapacity = std::max<std::size_t>(1, numQueryBits);
  if (queryBits_.size() < bitCapacity) {
    queryBits_.resize(kPatternFingerprintBits);
  }
  if (hostQueryBits_.size() < bitCapacity) {
    hostQueryBits_ = PinnedHostVector<std::uint16_t>(kPatternFingerprintBits);
  }
  if (hostIndices_.size() < numTargets) {
    hostIndices_ = PinnedHostVector<int>(numTargets);
  }
}

void PatternScreenWorkspace::screen(const std::uint32_t*      bitSlices,
                                    const int*                batchAtomStarts,
                                    int                       numTargets,
                                    const PatternScreenQuery& query) {
  cudaStream_t      stream       = stream_.stream();
  const std::size_t numQueryBits = bitSlices == nullptr ? 0 : query.bits.size();
  reserve(static_cast<std::size_t>(numTargets), numQueryBits);

  if (numQueryBits != 0) {
    std::copy(query.bits.begin(), query.bits.end(), hostQueryBits_.data());
    cudaCheckError(cudaMemcpyAsync(queryBits_.data(),
                                   hostQueryBits_.data(),
                                   numQueryBits * sizeof(std::uint16_t),
                                   cudaMemcpyHostToDevice,
                                   stream));
  }
  const std::size_t sliceWords = patternSliceWords(static_cast<std::size_t>(numTargets));
  const auto        blocks     = static_cast<unsigned int>((sliceWords + kScreenBlockSize - 1) / kScreenBlockSize);
  intersectPatternSlicesKernel<<<blocks, kScreenBlockSize, 0, stream>>>(bitSlices,
                                                                        sliceWords,
                                                                        numTargets,
                                                                        queryBits_.data(),
                                                                        static_cast<int>(numQueryBits),
                                                                        survivors_.data());
  cudaCheckError(cudaGetLastError());

  std::size_t tempBytes = tempStorage_.size();
  cudaCheckError(cub::DeviceSelect::If(tempStorage_.data(),
                                       tempBytes,
                                       thrust::counting_iterator<int>(0),
                                       indices_.data(),
                                       count_.data(),
                                       numTargets,
                                       SurvivorPredicate{survivors_.data(), batchAtomStarts, query.numAtoms},
                                       stream));
  cudaCheckError(cudaMemcpyAsync(hostCount_.data(), count_.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaCheckError(cudaStreamSynchronize(stream));
  if (hostCount_[0] > 0) {
    cudaCheckError(cudaMemcpyAsync(hostIndices_.data(),
                                   indices_.data(),
                                   static_cast<std::size_t>(hostCount_[0]) * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   stream));
    cudaCheckError(cudaStreamSynchronize(stream));
  }
}

}  // namespace nvMolKit
