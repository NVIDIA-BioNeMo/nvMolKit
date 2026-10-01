// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cub/block/block_scan.cuh>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "src/diversity_picker_algorithms.cuh"
#include "src/diversity_pickers.h"
#include "src/fingerprint_similarity_device.cuh"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit {
namespace detail {
namespace {

//! Sources per shared-memory tile in the multi-source fingerprint kernel.
constexpr int kSourceTile    = 32;
//! Largest fingerprint, in 32-bit words, that the tiled kernel stages in shared memory.
constexpr int kMaxTiledWords = 256;

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
    const int ordinal          = start + __popc(accepted & ((1U << lane) - 1U));
    state.windowOrdinals[lane] = ordinal;
    state.picks[ordinal]       = member;
    state.active[member]       = 0;
    if (state.labels != nullptr) {
      state.labels[member] = ordinal;
    }
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
  const std::uint32_t excludedBy = state.hits[candidate] & *state.accepted;
  if (excludedBy == 0U) {
    return;
  }
  state.active[candidate] = 0;
  if (state.labels != nullptr) {
    state.labels[candidate] = state.windowOrdinals[__ffs(static_cast<int>(excludedBy)) - 1];
  }
}

__global__ void markIndicesKernel(const int* indices, const int count, std::uint8_t* flags) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) {
    flags[indices[index]] = 1;
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

__global__ void fingerprintBitCountsKernel(const std::uint32_t* fingerprints,
                                           int*                 bitCounts,
                                           const int            numItems,
                                           const int            numWords) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= numItems) {
    return;
  }
  int count = 0;
  for (int word = 0; word < numWords; ++word) {
    count += __popc(fingerprints[static_cast<std::size_t>(index) * numWords + word]);
  }
  bitCounts[index] = count;
}

template <bool Vectorized>
__device__ __forceinline__ int intersectionCount(const std::uint32_t* left,
                                                 const std::uint32_t* right,
                                                 const int            numWords) {
  int count = 0;
  if constexpr (Vectorized) {
    const auto* left4  = reinterpret_cast<const uint4*>(left);
    const auto* right4 = reinterpret_cast<const uint4*>(right);
    for (int word = 0; word < numWords / 4; ++word) {
      const uint4 a = left4[word];
      const uint4 b = right4[word];
      count += __popc(a.x & b.x) + __popc(a.y & b.y) + __popc(a.z & b.z) + __popc(a.w & b.w);
    }
  } else {
    for (int word = 0; word < numWords; ++word) {
      count += __popc(left[word] & right[word]);
    }
  }
  return count;
}

//! One thread per candidate; source rows are read through the cache. Used for single sources and wide fingerprints.
template <FingerprintSimilarityMetric Metric, bool Vectorized, typename Op>
__global__ void fingerprintDistancesKernel(const std::uint32_t* fingerprints,
                                           const int*           bitCounts,
                                           const int            numItems,
                                           const int            numWords,
                                           const int*           sources,
                                           const int            numSources,
                                           const Op             op) {
  const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (candidate >= numItems || op.skip(candidate)) {
    return;
  }
  const std::uint32_t* row      = fingerprints + static_cast<std::size_t>(candidate) * numWords;
  const int            rowCount = bitCounts[candidate];
  auto                 state    = op.start(candidate);
  for (int ordinal = 0; ordinal < numSources; ++ordinal) {
    const int source = sources[ordinal];
    if (validSource(source, numItems)) {
      const int intersection =
        intersectionCount<Vectorized>(fingerprints + static_cast<std::size_t>(source) * numWords, row, numWords);
      op.visit(state,
               ordinal,
               source == candidate,
               fingerprintDistance<Metric>(intersection, bitCounts[source], rowCount));
    }
  }
  op.finish(candidate, state);
}

//! One thread per candidate against tiles of kSourceTile sources staged in shared memory, reading each candidate
//! word once per tile.
template <FingerprintSimilarityMetric Metric, bool Vectorized, typename Op>
__global__ void fingerprintTiledDistancesKernel(const std::uint32_t* fingerprints,
                                                const int*           bitCounts,
                                                const int            numItems,
                                                const int            numWords,
                                                const int*           sources,
                                                const int            numSources,
                                                const Op             op) {
  extern __shared__ uint4 tileStorage[];
  auto*                   tile = reinterpret_cast<std::uint32_t*>(tileStorage);
  __shared__ int          tileSources[kSourceTile];
  __shared__ int          tileCounts[kSourceTile];

  const int  candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const bool live      = candidate < numItems && !op.skip(candidate);
  if (__syncthreads_or(live) == 0) {
    return;
  }
  const std::uint32_t* row      = fingerprints + static_cast<std::size_t>(live ? candidate : 0) * numWords;
  const int            rowCount = live ? bitCounts[candidate] : 0;
  typename Op::State   state{};
  if (live) {
    state = op.start(candidate);
  }

  for (int base = 0; base < numSources; base += kSourceTile) {
    const int tileLength = min(kSourceTile, numSources - base);
    __syncthreads();
    if (threadIdx.x < kSourceTile) {
      const int  source        = static_cast<int>(threadIdx.x) < tileLength ? sources[base + threadIdx.x] : -1;
      const bool valid         = validSource(source, numItems);
      tileSources[threadIdx.x] = valid ? source : -1;
      tileCounts[threadIdx.x]  = valid ? bitCounts[source] : 0;
    }
    __syncthreads();
    for (int index = static_cast<int>(threadIdx.x); index < tileLength * numWords; index += kPickerBlockSize) {
      const int source = tileSources[index / numWords];
      tile[index] = source >= 0 ? fingerprints[static_cast<std::size_t>(source) * numWords + index % numWords] : 0U;
    }
    __syncthreads();
    if (!live) {
      continue;
    }

    int intersections[kSourceTile] = {};
    if constexpr (Vectorized) {
      const auto* row4   = reinterpret_cast<const uint4*>(row);
      const auto* tile4  = reinterpret_cast<const uint4*>(tile);
      const int   words4 = numWords / 4;
      for (int word = 0; word < words4; ++word) {
        const uint4 a = row4[word];
#pragma unroll
        for (int t = 0; t < kSourceTile; ++t) {
          const uint4 b = tile4[t * words4 + word];
          intersections[t] += __popc(a.x & b.x) + __popc(a.y & b.y) + __popc(a.z & b.z) + __popc(a.w & b.w);
        }
      }
    } else {
      for (int word = 0; word < numWords; ++word) {
        const std::uint32_t a = row[word];
#pragma unroll
        for (int t = 0; t < kSourceTile; ++t) {
          intersections[t] += __popc(a & tile[t * numWords + word]);
        }
      }
    }
#pragma unroll
    for (int t = 0; t < kSourceTile; ++t) {
      if (t < tileLength && tileSources[t] >= 0) {
        op.visit(state,
                 base + t,
                 tileSources[t] == candidate,
                 fingerprintDistance<Metric>(intersections[t], tileCounts[t], rowCount));
      }
    }
  }
  if (live) {
    op.finish(candidate, state);
  }
}

template <typename Scalar> class MatrixDistanceProvider {
 public:
  MatrixDistanceProvider(const cuda::std::span<const Scalar> distances, const int numItems)
      : distances_(distances),
        numItems_(numItems) {}

  int size() const { return numItems_; }
  int leaderWindow() const { return kMaxLeaderWindow; }

  template <typename Op>
  void accumulateDistances(const int* sources, const int numSources, const Op& op, cudaStream_t stream) const {
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

template <FingerprintSimilarityMetric Metric> class FingerprintDistanceProvider {
 public:
  FingerprintDistanceProvider(const cuda::std::span<const std::uint32_t> fingerprints,
                              const int                                  numItems,
                              const int                                  numWords,
                              cudaStream_t                               stream)
      : fingerprints_(fingerprints),
        numItems_(numItems),
        numWords_(numWords),
        vectorized_(numWords % 4 == 0 && reinterpret_cast<std::uintptr_t>(fingerprints.data()) % alignof(uint4) == 0),
        bitCounts_(numItems, stream) {
    if (numItems > 0) {
      fingerprintBitCountsKernel<<<pickerGridSize(numItems), kPickerBlockSize, 0, stream>>>(fingerprints_.data(),
                                                                                            bitCounts_.data(),
                                                                                            numItems_,
                                                                                            numWords_);
      cudaCheckError(cudaGetLastError());
    }
  }

  int size() const { return numItems_; }
  int leaderWindow() const { return kMaxLeaderWindow; }

  template <typename Op>
  void accumulateDistances(const int* sources, const int numSources, const Op& op, cudaStream_t stream) const {
    if (numItems_ == 0 || numSources == 0) {
      return;
    }
    if (vectorized_) {
      launch<true>(sources, numSources, op, stream);
    } else {
      launch<false>(sources, numSources, op, stream);
    }
  }

 private:
  template <bool Vectorized, typename Op>
  void launch(const int* sources, const int numSources, const Op& op, cudaStream_t stream) const {
    const int grid = pickerGridSize(numItems_);
    if (numSources > 1 && numWords_ <= kMaxTiledWords) {
      const auto sharedBytes = static_cast<std::size_t>(kSourceTile) * numWords_ * sizeof(std::uint32_t);
      fingerprintTiledDistancesKernel<Metric, Vectorized>
        <<<grid, kPickerBlockSize, sharedBytes, stream>>>(fingerprints_.data(),
                                                          bitCounts_.data(),
                                                          numItems_,
                                                          numWords_,
                                                          sources,
                                                          numSources,
                                                          op);
    } else {
      fingerprintDistancesKernel<Metric, Vectorized><<<grid, kPickerBlockSize, 0, stream>>>(fingerprints_.data(),
                                                                                            bitCounts_.data(),
                                                                                            numItems_,
                                                                                            numWords_,
                                                                                            sources,
                                                                                            numSources,
                                                                                            op);
    }
    cudaCheckError(cudaGetLastError());
  }

  cuda::std::span<const std::uint32_t> fingerprints_;
  int                                  numItems_;
  int                                  numWords_;
  bool                                 vectorized_;
  AsyncDeviceVector<int>               bitCounts_;
};

template <typename Scalar>
void validateDistanceMatrix(const cuda::std::span<const Scalar> distanceMatrix, const int numItems) {
  if (numItems < 0 || distanceMatrix.size() != static_cast<std::size_t>(numItems) * numItems) {
    throw std::invalid_argument("Distance matrix buffer size does not match its square shape");
  }
}

void validateMatrixCutoff(const double cutoff) {
  if (!std::isfinite(cutoff) || cutoff < 0.0 || cutoff > std::numeric_limits<float>::max()) {
    throw std::invalid_argument("cutoff must be finite, non-negative, and no larger than the maximum float32 value");
  }
}

void validateFingerprintInput(const cuda::std::span<const std::uint32_t> fingerprints,
                              const int                                  numItems,
                              const int                                  numWords) {
  if (numItems < 0 || numWords <= 0 ||
      fingerprints.size() != static_cast<std::size_t>(numItems) * static_cast<std::size_t>(numWords)) {
    throw std::invalid_argument("Fingerprint buffer size does not match its shape");
  }
}

template <typename Function>
auto withFingerprintProvider(const cuda::std::span<const std::uint32_t> fingerprints,
                             const int                                  numItems,
                             const int                                  numWords,
                             const FingerprintSimilarityMetric          metric,
                             cudaStream_t                               stream,
                             Function&&                                 function) {
  validateFingerprintInput(fingerprints, numItems, numWords);
  if (metric == FingerprintSimilarityMetric::Tanimoto) {
    FingerprintDistanceProvider<FingerprintSimilarityMetric::Tanimoto> provider(fingerprints,
                                                                                numItems,
                                                                                numWords,
                                                                                stream);
    return function(provider);
  }
  if (metric == FingerprintSimilarityMetric::Cosine) {
    FingerprintDistanceProvider<FingerprintSimilarityMetric::Cosine> provider(fingerprints, numItems, numWords, stream);
    return function(provider);
  }
  throw std::invalid_argument("Unsupported fingerprint similarity metric");
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
  return leaderPick(provider, static_cast<float>(cutoff), pickSize, firstPicks, nullptr, stream);
}

template <typename Scalar>
ClusteringResult diseFromMatrix(const cuda::std::span<const Scalar> distanceMatrix,
                                const int                           numItems,
                                const double                        cutoff,
                                const bool                          nearestAssignment,
                                cudaStream_t                        stream) {
  validateDistanceMatrix(distanceMatrix, numItems);
  validateMatrixCutoff(cutoff);
  MatrixDistanceProvider<Scalar> provider(distanceMatrix, numItems);
  return diseCluster(provider, static_cast<float>(cutoff), nearestAssignment, stream);
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

void launchMarkIndices(const int* indices, const int count, std::uint8_t* flags, cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  markIndicesKernel<<<pickerGridSize(count), kPickerBlockSize, 0, stream>>>(indices, count, flags);
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

void validateUnitCutoff(const double cutoff) {
  if (!(cutoff >= 0.0 && cutoff <= 1.0)) {
    throw std::invalid_argument("cutoff must be in [0, 1]");
  }
}

ClusteringResult buildClusteringResult(const std::vector<int>& labels, const std::vector<int>& centroids) {
  const int                 numClusters = static_cast<int>(centroids.size());
  std::vector<std::int64_t> sizes(numClusters, 0);
  for (const int label : labels) {
    ++sizes[label];
  }

  std::vector<int> order(numClusters);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&sizes](const int left, const int right) {
    return sizes[left] > sizes[right];
  });
  std::vector<int> remap(numClusters);
  for (int newId = 0; newId < numClusters; ++newId) {
    remap[order[newId]] = newId;
  }

  ClusteringResult result;
  result.clusterIds.resize(labels.size());
  result.centroids.resize(numClusters);
  result.clusterSizes.resize(numClusters);
  std::transform(labels.begin(), labels.end(), result.clusterIds.begin(), [&remap](const int label) {
    return remap[label];
  });
  for (int newId = 0; newId < numClusters; ++newId) {
    const int oldId            = order[newId];
    result.centroids[newId]    = centroids[oldId];
    result.clusterSizes[newId] = sizes[oldId];
  }
  return result;
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

ClusteringResult diseFromDistanceMatrix(const cuda::std::span<const float> distanceMatrix,
                                        const int                          numItems,
                                        const double                       cutoff,
                                        const bool                         nearestAssignment,
                                        cudaStream_t                       stream) {
  return detail::diseFromMatrix(distanceMatrix, numItems, cutoff, nearestAssignment, stream);
}

ClusteringResult diseFromDistanceMatrix(const cuda::std::span<const double> distanceMatrix,
                                        const int                           numItems,
                                        const double                        cutoff,
                                        const bool                          nearestAssignment,
                                        cudaStream_t                        stream) {
  return detail::diseFromMatrix(distanceMatrix, numItems, cutoff, nearestAssignment, stream);
}

PickerResult fusedLeaderGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                            const int                                  numFingerprints,
                            const int                                  numWords,
                            const double                               cutoff,
                            const FingerprintSimilarityMetric          metric,
                            const int                                  pickSize,
                            const std::vector<int>&                    firstPicks,
                            cudaStream_t                               stream) {
  detail::validateUnitCutoff(cutoff);
  return detail::withFingerprintProvider(fingerprints, numFingerprints, numWords, metric, stream, [&](auto& provider) {
    return detail::leaderPick(provider, static_cast<float>(cutoff), pickSize, firstPicks, nullptr, stream);
  });
}

ClusteringResult fusedDiseGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                              const int                                  numFingerprints,
                              const int                                  numWords,
                              const double                               cutoff,
                              const FingerprintSimilarityMetric          metric,
                              const bool                                 nearestAssignment,
                              cudaStream_t                               stream) {
  detail::validateUnitCutoff(cutoff);
  return detail::withFingerprintProvider(fingerprints, numFingerprints, numWords, metric, stream, [&](auto& provider) {
    return detail::diseCluster(provider, static_cast<float>(cutoff), nearestAssignment, stream);
  });
}

}  // namespace nvMolKit
