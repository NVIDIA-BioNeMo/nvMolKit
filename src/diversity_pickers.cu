// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cub/block/block_scan.cuh>
#include <stdexcept>
#include <vector>

#include "src/diversity_picker_algorithms.cuh"
#include "src/diversity_pickers.h"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit {
namespace detail {
namespace {

__global__ void gatherLeaderWindowKernel(const LeaderState state, const int numItems, const int windowSize) {
  using BlockScan = cub::BlockScan<int, kPickerBlockSize>;
  __shared__ typename BlockScan::TempStorage scanStorage;
  __shared__ int                             found;

  if (threadIdx.x < kMaxLeaderWindow) {
    state.window[threadIdx.x] = -1;
  }
  if (threadIdx.x == 0) {
    found = 0;
  }
  __syncthreads();
  for (int base = *state.scanStart; base < numItems; base += kPickerBlockSize) {
    const int alreadyFound = found;
    if (alreadyFound >= windowSize) {
      break;
    }
    const int index  = base + static_cast<int>(threadIdx.x);
    const int flag   = index < numItems && state.active[index] != 0 ? 1 : 0;
    int       offset = 0;
    int       total  = 0;
    BlockScan(scanStorage).ExclusiveSum(flag, offset, total);
    if (flag != 0 && alreadyFound + offset < windowSize) {
      state.window[alreadyFound + offset] = index;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      found = min(windowSize, alreadyFound + total);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *state.scanStart = found == windowSize ? state.window[windowSize - 1] + 1 : numItems;
  }
}

//! One warp: lane j loads window member j and its hits, then the warp resolves members in order from registers.
__global__ void resolveLeaderWindowKernel(const LeaderState                state,
                                          const int                        windowSize,
                                          const bool                       forced,
                                          const int                        limit,
                                          const cudaGraphConditionalHandle loop) {
  const int           lane   = static_cast<int>(threadIdx.x);
  const int           member = lane < windowSize ? state.window[lane] : -1;
  const std::uint32_t hits   = member >= 0 ? state.hits[member] : 0U;
  const int           start  = *state.count;

  int           count    = start;
  std::uint32_t accepted = 0U;
  for (int position = 0; position < windowSize; ++position) {
    const int           candidate     = __shfl_sync(0xffffffffU, member, position);
    const std::uint32_t candidateHits = __shfl_sync(0xffffffffU, hits, position);
    if (candidate < 0 || (!forced && count >= limit)) {
      break;
    }
    if (forced || (candidateHits & accepted) == 0U) {
      accepted |= 1U << position;
      ++count;
    }
  }

  if ((accepted >> lane & 1U) != 0U) {
    const int ordinal    = start + __popc(accepted & ((1U << lane) - 1U));
    state.picks[ordinal] = member;
    state.active[member] = 0;
  }
  if (lane == 0) {
    *state.count    = count;
    *state.accepted = accepted;
    if (!forced) {
      cudaGraphSetConditional(loop, member >= 0 && count < limit ? 1 : 0);
    }
  }
}

__global__ void applyLeaderWindowKernel(const LeaderState state, const int numItems) {
  const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (candidate >= numItems || state.active[candidate] == 0) {
    return;
  }
  if ((state.hits[candidate] & *state.accepted) != 0U) {
    state.active[candidate] = 0;
  }
}

__device__ __forceinline__ bool validSource(const int source, const int numItems) {
  return source >= 0 && source < numItems;
}

template <typename Scalar, typename Op>
__global__ void matrixDistancesKernel(const Scalar* distances,
                                      const int     numItems,
                                      const int*    sources,
                                      const int     numSources,
                                      const Op      op) {
  const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (candidate >= numItems || op.skip(candidate)) {
    return;
  }
  auto state = op.start(candidate);
  for (int ordinal = 0; ordinal < numSources; ++ordinal) {
    const int source = sources[ordinal];
    if (validSource(source, numItems)) {
      const auto distance = static_cast<float>(distances[static_cast<std::size_t>(source) * numItems + candidate]);
      op.visit(state, ordinal, source == candidate, distance);
    }
  }
  op.finish(candidate, state);
}

template <typename Scalar> class MatrixDistanceProvider {
 public:
  MatrixDistanceProvider(const cuda::std::span<const Scalar> distances, const int numItems)
      : distances_(distances),
        numItems_(numItems) {}

  int size() const { return numItems_; }
  int leaderWindow() const { return kMaxLeaderWindow; }

  template <typename Op>
  void forEachDistance(const int* sources, const int numSources, const Op& op, cudaStream_t stream) const {
    if (numItems_ == 0 || numSources == 0) {
      return;
    }
    matrixDistancesKernel<<<pickerGridSize(numItems_), kPickerBlockSize, 0, stream>>>(distances_.data(),
                                                                                      numItems_,
                                                                                      sources,
                                                                                      numSources,
                                                                                      op);
    cudaCheckError(cudaGetLastError());
  }

 private:
  cuda::std::span<const Scalar> distances_;
  int                           numItems_;
};

template <typename Scalar>
void validateDistanceMatrix(const cuda::std::span<const Scalar> distanceMatrix, const int numItems) {
  if (numItems < 0 || distanceMatrix.size() != static_cast<std::size_t>(numItems) * numItems) {
    throw std::invalid_argument("Distance matrix buffer size does not match its square shape");
  }
}

void validateMatrixCutoff(const double cutoff) {
  if (!std::isfinite(cutoff) || cutoff < 0.0) {
    throw std::invalid_argument("cutoff must be finite and non-negative");
  }
}

template <typename Scalar>
PickerResult leaderFromMatrix(const cuda::std::span<const Scalar> distanceMatrix,
                              const int                           numItems,
                              const double                        cutoff,
                              const int                           pickSize,
                              const std::vector<int>&             firstPicks,
                              cudaStream_t                        stream) {
  validateDistanceMatrix(distanceMatrix, numItems);
  validateMatrixCutoff(cutoff);
  MatrixDistanceProvider<Scalar> provider(distanceMatrix, numItems);
  return leaderPick(provider, static_cast<float>(cutoff), pickSize, firstPicks, stream);
}

}  // namespace

void launchGatherLeaderWindow(const LeaderState& state, const int numItems, const int windowSize, cudaStream_t stream) {
  gatherLeaderWindowKernel<<<1, kPickerBlockSize, 0, stream>>>(state, numItems, windowSize);
  cudaCheckError(cudaGetLastError());
}

void launchResolveLeaderWindow(const LeaderState&               state,
                               const int                        windowSize,
                               const bool                       forced,
                               const int                        limit,
                               const cudaGraphConditionalHandle loop,
                               cudaStream_t                     stream) {
  resolveLeaderWindowKernel<<<1, 32, 0, stream>>>(state, windowSize, forced, limit, loop);
  cudaCheckError(cudaGetLastError());
}

void launchApplyLeaderWindow(const LeaderState& state, const int numItems, cudaStream_t stream) {
  applyLeaderWindowKernel<<<pickerGridSize(numItems), kPickerBlockSize, 0, stream>>>(state, numItems);
  cudaCheckError(cudaGetLastError());
}

void validateFirstPicks(const std::vector<int>& firstPicks, const int numItems) {
  std::vector<std::uint8_t> seen(numItems, 0);
  for (const int pick : firstPicks) {
    if (pick < 0 || pick >= numItems) {
      throw std::invalid_argument("first_picks contains an index outside the input pool");
    }
    if (seen[pick] != 0) {
      throw std::invalid_argument("first_picks must not contain duplicate indices");
    }
    seen[pick] = 1;
  }
}

}  // namespace detail

PickerResult leaderFromDistanceMatrix(const cuda::std::span<const float> distanceMatrix,
                                      const int                          numItems,
                                      const double                       cutoff,
                                      const int                          pickSize,
                                      const std::vector<int>&            firstPicks,
                                      cudaStream_t                       stream) {
  return detail::leaderFromMatrix(distanceMatrix, numItems, cutoff, pickSize, firstPicks, stream);
}

PickerResult leaderFromDistanceMatrix(const cuda::std::span<const double> distanceMatrix,
                                      const int                           numItems,
                                      const double                        cutoff,
                                      const int                           pickSize,
                                      const std::vector<int>&             firstPicks,
                                      cudaStream_t                        stream) {
  return detail::leaderFromMatrix(distanceMatrix, numItems, cutoff, pickSize, firstPicks, stream);
}

}  // namespace nvMolKit
