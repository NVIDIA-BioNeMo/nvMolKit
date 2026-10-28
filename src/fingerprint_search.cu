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

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <limits>
#include <stdexcept>

#include "src/fingerprint_search.h"
#include "src/fingerprint_similarity_device.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {
namespace {

constexpr int   kSetupBlockSize   = 256;
constexpr int   kWarpSize         = 32;
// One warp per query; the block shares each library tile across its queries.
constexpr int   kQueriesPerBlock  = 8;
constexpr int   kSearchBlockSize  = kQueriesPerBlock * kWarpSize;
// Each lane scores two rows of a 64-row tile.
constexpr int   kTileRows         = 2 * kWarpSize;
constexpr int   kWordChunk        = 32;
constexpr int   kBitsPerWord      = 32;
constexpr int   kDefaultSmemLimit = 48 * 1024;
// The float prefilter compares against a slightly lowered bound so rounding never rejects a true hit. Exact double
// scores decide what enters the list.
constexpr float kBoundSlack       = 1.0F - 1.0e-5F;

__global__ void fingerprintBitCountKernel(const std::uint32_t* fingerprints,
                                          int                  numFingerprints,
                                          int                  numWords,
                                          int*                 bitCounts,
                                          int*                 rowIndices) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= numFingerprints) {
    return;
  }
  int bitCount = 0;
  for (int word = 0; word < numWords; ++word) {
    bitCount += __popc(fingerprints[static_cast<std::size_t>(row) * numWords + word]);
  }
  bitCounts[row]  = bitCount;
  rowIndices[row] = row;
}

__global__ void gatherRowsKernel(const std::uint32_t* source,
                                 const int*           rowOrder,
                                 int                  numRows,
                                 int                  numWords,
                                 std::uint32_t*       destination) {
  const std::size_t total = static_cast<std::size_t>(numRows) * numWords;
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t row  = i / numWords;
    const std::size_t word = i % numWords;
    destination[i]         = source[static_cast<std::size_t>(rowOrder[row]) * numWords + word];
  }
}

__global__ void iotaKernel(int* values, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    values[i] = i;
  }
}

struct SearchParams {
  const std::uint32_t* libraryBits;
  const int*           libraryCounts;
  const int*           libraryIndices;
  int                  numLibrary;
  int                  numWords;
  //! Queries are visited in popcount order. Entry i of these arrays describes the i-th query in that order.
  const std::uint32_t* queryBits;
  const int*           queryRows;
  const int*           queryCounts;
  const int*           outputRows;
  int                  numQueries;
  //! Self search: query i in popcount order is library row i, which is skipped.
  bool                 selfSearch;
  int                  k;
  double               threshold;
  bool                 prune;
  double*              scores;
  std::int64_t*        indices;
};

// Total order of hits: higher similarity first, then lower library index.
__device__ __forceinline__ bool precedes(const double lhsScore,
                                         const int    lhsIndex,
                                         const double rhsScore,
                                         const int    rhsIndex) {
  return lhsScore > rhsScore || (lhsScore == rhsScore && lhsIndex < rhsIndex);
}

__device__ __forceinline__ int lowerBound(const int* values, int count, int target) {
  int lo = 0;
  int hi = count;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (values[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

struct SearchShared {
  std::uint32_t tile[kTileRows][kWordChunk + 1];
  double        stagedScores[kQueriesPerBlock][kTileRows];
  int           stagedIndices[kQueriesPerBlock][kTileRows];
  double        sortedScores[kQueriesPerBlock][kTileRows];
  int           sortedIndices[kQueriesPerBlock][kTileRows];
  int           sortedRanks[kQueriesPerBlock][kTileRows];
};

//! Per-warp running top-k list. The list lives in the query's output row; the warp keeps its length and last entry.
struct TopKState {
  int    count    = 0;
  double kthScore = 0.0;
  int    kthIndex = 0;
  float  boundF   = 0.0F;
};

__device__ __forceinline__ float loweredBound(const double bound) {
  return bound > 0.0 ? static_cast<float>(bound) * kBoundSlack : -1.0F;
}

// Merge the warp's staged candidates into its top-k list in the output row. All lanes of the warp participate.
__device__ void mergeCandidates(SearchShared&       shared,
                                const int           warp,
                                const int           lane,
                                const int           numStaged,
                                const SearchParams& p,
                                double*             rowScores,
                                std::int64_t*       rowIndices,
                                TopKState&          state) {
  double* const stagedScores  = shared.stagedScores[warp];
  int* const    stagedIndices = shared.stagedIndices[warp];
  double* const sortedScores  = shared.sortedScores[warp];
  int* const    sortedIndices = shared.sortedIndices[warp];
  int* const    sortedRanks   = shared.sortedRanks[warp];

  // Rank-sort the staged candidates. Library indices are unique within a query, so ranks are distinct.
  for (int j = lane; j < numStaged; j += kWarpSize) {
    const double score = stagedScores[j];
    const int    index = stagedIndices[j];
    int          rank  = 0;
    for (int i = 0; i < numStaged; ++i) {
      rank += precedes(stagedScores[i], stagedIndices[i], score, index) ? 1 : 0;
    }
    sortedScores[rank]  = score;
    sortedIndices[rank] = index;
  }
  __syncwarp();

  // Final position of each candidate: its rank among candidates plus the list entries that precede it.
  const int listCount = state.count;
  for (int j = lane; j < numStaged; j += kWarpSize) {
    int lo = 0;
    int hi = listCount;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (precedes(rowScores[mid], static_cast<int>(rowIndices[mid]), sortedScores[j], sortedIndices[j])) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    sortedRanks[j] = j + lo;
  }
  __syncwarp();

  // Shift list entries toward the tail, last chunk first. Every entry moves to a position at or after its own, so
  // reading a chunk before writing it never overwrites an entry that has not moved yet.
  for (int chunkStart = ((listCount - 1) / kWarpSize) * kWarpSize; listCount > 0 && chunkStart >= 0;
       chunkStart -= kWarpSize) {
    const int    i        = chunkStart + lane;
    const bool   hasEntry = i < listCount;
    double       score    = 0.0;
    std::int64_t index    = 0;
    int          target   = i;
    if (hasEntry) {
      score  = rowScores[i];
      index  = rowIndices[i];
      int lo = 0;
      int hi = numStaged;
      while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (precedes(sortedScores[mid], sortedIndices[mid], score, static_cast<int>(index))) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      target = i + lo;
    }
    __syncwarp();
    if (hasEntry && target != i && target < p.k) {
      rowScores[target]  = score;
      rowIndices[target] = index;
    }
    __syncwarp();
  }

  for (int j = lane; j < numStaged; j += kWarpSize) {
    const int target = sortedRanks[j];
    if (target < p.k) {
      rowScores[target]  = sortedScores[j];
      rowIndices[target] = sortedIndices[j];
    }
  }
  __syncwarp();

  state.count = min(p.k, listCount + numStaged);
  if (state.count == p.k) {
    state.kthScore = rowScores[p.k - 1];
    state.kthIndex = static_cast<int>(rowIndices[p.k - 1]);
    state.boundF   = loweredBound(state.kthScore);
  }
}

// Score one library tile [start, end) against every query in the block. Returns whether any query could still gain a
// hit from the tile; when pruning, an infeasible tile is skipped without reading its fingerprints.
template <FingerprintSimilarityMetric Metric>
__device__ bool processTile(SearchShared&        shared,
                            const std::uint32_t* queryWords,
                            const int            start,
                            const int            end,
                            const bool           active,
                            const int            queryCount,
                            const int            excludedRow,
                            const SearchParams&  p,
                            double*              rowScores,
                            std::int64_t*        rowIndices,
                            TopKState&           state) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int rows = end - start;

  bool feasible = active;
  if (feasible && p.prune) {
    // Library counts are sorted, so the tile's best possible similarity comes from the count closest to the query's.
    const int closest = min(max(queryCount, p.libraryCounts[start]), p.libraryCounts[end - 1]);
    feasible = state.boundF <= 0.0F || detail::fingerprintSimilarityCanReach<Metric>(closest, queryCount, state.boundF);
  }
  if (__syncthreads_or(feasible) == 0) {
    return false;
  }

  int intersection0 = 0;
  int intersection1 = 0;
  for (int chunkStart = 0; chunkStart < p.numWords; chunkStart += kWordChunk) {
    const int chunkWords = min(kWordChunk, p.numWords - chunkStart);
    for (int e = threadIdx.x; e < rows * chunkWords; e += kSearchBlockSize) {
      const int row          = e / chunkWords;
      const int word         = e % chunkWords;
      shared.tile[row][word] = p.libraryBits[static_cast<std::size_t>(start + row) * p.numWords + chunkStart + word];
    }
    __syncthreads();
    if (feasible) {
      for (int word = 0; word < chunkWords; ++word) {
        const std::uint32_t queryWord = queryWords[chunkStart + word];
        intersection0 += __popc(queryWord & shared.tile[lane][word]);
        intersection1 += __popc(queryWord & shared.tile[lane + kWarpSize][word]);
      }
    }
    __syncthreads();
  }

  if (!feasible) {
    return true;
  }

  // Stage hits that can enter the list, compacted with a warp ballot.
  const bool  full      = state.count == p.k;
  const float bound     = full ? state.boundF : loweredBound(p.threshold);
  int         numStaged = 0;
  for (int half = 0; half < 2; ++half) {
    const int row       = lane + half * kWarpSize;
    const int intersect = half == 0 ? intersection0 : intersection1;
    bool      take      = false;
    double    score     = 0.0;
    int       index     = 0;
    if (row < rows && start + row != excludedRow) {
      const int   libraryCount = p.libraryCounts[start + row];
      const float denominator =
        detail::fingerprintSimilarityDenominator<Metric, float>(intersect, libraryCount, queryCount);
      if (bound <= 0.0F || static_cast<float>(intersect) >= bound * denominator) {
        score = detail::fingerprintSimilarity<Metric, double>(intersect, libraryCount, queryCount);
        index = p.libraryIndices[start + row];
        take  = score >= p.threshold && (!full || precedes(score, index, state.kthScore, state.kthIndex));
      }
    }
    const unsigned ballot = __ballot_sync(0xffffffffU, take);
    if (take) {
      const int slot                   = numStaged + __popc(ballot & ((1U << lane) - 1U));
      shared.stagedScores[warp][slot]  = score;
      shared.stagedIndices[warp][slot] = index;
    }
    numStaged += __popc(ballot);
  }
  __syncwarp();

  if (numStaged > 0) {
    mergeCandidates(shared, warp, lane, numStaged, p, rowScores, rowIndices, state);
  }
  return true;
}

template <FingerprintSimilarityMetric Metric>
__global__ void __launch_bounds__(kSearchBlockSize) fingerprintTopKKernel(const SearchParams p) {
  extern __shared__ std::uint32_t queryWordsShared[];
  __shared__ SearchShared         shared;

  const int  warp        = threadIdx.x / kWarpSize;
  const int  lane        = threadIdx.x % kWarpSize;
  const int  firstQuery  = blockIdx.x * kQueriesPerBlock;
  const int  lastQuery   = min(p.numQueries, firstQuery + kQueriesPerBlock) - 1;
  const int  query       = firstQuery + warp;
  const bool active      = query < p.numQueries;
  const int  queryCount  = active ? p.queryCounts[query] : 0;
  const int  excludedRow = active && p.selfSearch ? query : -1;

  std::uint32_t* const queryWords = queryWordsShared + static_cast<std::size_t>(warp) * p.numWords;
  double*              rowScores  = nullptr;
  std::int64_t*        rowIndices = nullptr;
  if (active) {
    const std::uint32_t* source = p.queryBits + static_cast<std::size_t>(p.queryRows[query]) * p.numWords;
    for (int word = lane; word < p.numWords; word += kWarpSize) {
      queryWords[word] = source[word];
    }
    rowScores  = p.scores + static_cast<std::size_t>(p.outputRows[query]) * p.k;
    rowIndices = p.indices + static_cast<std::size_t>(p.outputRows[query]) * p.k;
  }
  __syncthreads();

  TopKState state;
  state.boundF = loweredBound(p.threshold);

  // Rows whose popcount lies within the block's query popcounts are scanned first. Rows above (below) every query's
  // popcount only get less similar further out, so each outward direction stops at its first infeasible tile.
  const int coreStart = lowerBound(p.libraryCounts, p.numLibrary, p.queryCounts[firstQuery]);
  const int coreEnd   = lowerBound(p.libraryCounts, p.numLibrary, p.queryCounts[lastQuery] + 1);
  for (int start = coreStart; start < coreEnd; start += kTileRows) {
    processTile<Metric>(shared,
                        queryWords,
                        start,
                        min(start + kTileRows, coreEnd),
                        active,
                        queryCount,
                        excludedRow,
                        p,
                        rowScores,
                        rowIndices,
                        state);
  }

  int  up       = coreEnd;
  int  down     = coreStart;
  bool upOpen   = up < p.numLibrary;
  bool downOpen = down > 0;
  while (upOpen || downOpen) {
    if (upOpen) {
      const int  end = min(up + kTileRows, p.numLibrary);
      const bool any = processTile<
        Metric>(shared, queryWords, up, end, active, queryCount, excludedRow, p, rowScores, rowIndices, state);
      up     = end;
      upOpen = up < p.numLibrary && (any || !p.prune);
    }
    if (downOpen) {
      const int  start = max(down - kTileRows, 0);
      const bool any   = processTile<
        Metric>(shared, queryWords, start, down, active, queryCount, excludedRow, p, rowScores, rowIndices, state);
      down     = start;
      downOpen = down > 0 && (any || !p.prune);
    }
  }
}

template <FingerprintSimilarityMetric Metric> void launchSearch(const SearchParams& params, cudaStream_t stream) {
  const std::size_t dynamicSmem = static_cast<std::size_t>(kQueriesPerBlock) * params.numWords * sizeof(std::uint32_t);
  if (dynamicSmem + sizeof(SearchShared) > kDefaultSmemLimit) {
    cudaCheckError(cudaFuncSetAttribute(fingerprintTopKKernel<Metric>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(dynamicSmem)));
  }
  const int numBlocks = (params.numQueries + kQueriesPerBlock - 1) / kQueriesPerBlock;
  fingerprintTopKKernel<Metric><<<numBlocks, kSearchBlockSize, dynamicSmem, stream>>>(params);
  cudaCheckError(cudaGetLastError());
}

// Popcount every row and return the row order sorted by popcount, with the sorted popcounts.
void sortRowsByBitCount(const std::uint32_t*    fingerprints,
                        int                     numRows,
                        int                     numWords,
                        AsyncDeviceVector<int>& sortedCounts,
                        AsyncDeviceVector<int>& sortedRows,
                        cudaStream_t            stream) {
  AsyncDeviceVector<int> counts(numRows, stream);
  AsyncDeviceVector<int> rows(numRows, stream);
  sortedCounts = AsyncDeviceVector<int>(numRows, stream);
  sortedRows   = AsyncDeviceVector<int>(numRows, stream);
  if (numRows == 0) {
    return;
  }
  fingerprintBitCountKernel<<<(numRows + kSetupBlockSize - 1) / kSetupBlockSize, kSetupBlockSize, 0, stream>>>(
    fingerprints,
    numRows,
    numWords,
    counts.data(),
    rows.data());
  cudaCheckError(cudaGetLastError());

  // Popcounts never exceed the fingerprint width, so only the low bits need sorting. The sort is stable, so equal
  // popcounts keep their original row order.
  int endBit = 1;
  while ((1LL << endBit) <= static_cast<long long>(numWords) * kBitsPerWord) {
    ++endBit;
  }
  std::size_t storageBytes = 0;
  cudaCheckError(cub::DeviceRadixSort::SortPairs(nullptr,
                                                 storageBytes,
                                                 counts.data(),
                                                 sortedCounts.data(),
                                                 rows.data(),
                                                 sortedRows.data(),
                                                 numRows,
                                                 0,
                                                 endBit,
                                                 stream));
  AsyncDeviceVector<std::uint8_t> storage(storageBytes, stream);
  cudaCheckError(cub::DeviceRadixSort::SortPairs(storage.data(),
                                                 storageBytes,
                                                 counts.data(),
                                                 sortedCounts.data(),
                                                 rows.data(),
                                                 sortedRows.data(),
                                                 numRows,
                                                 0,
                                                 endBit,
                                                 stream));
}

void checkDevice(const int deviceId) {
  int current = 0;
  cudaCheckError(cudaGetDevice(&current));
  if (current != deviceId) {
    throw std::runtime_error("FingerprintSearchIndex must be searched on the device it was built on (device " +
                             std::to_string(deviceId) + "), but the current device is " + std::to_string(current));
  }
}

void checkSearchArguments(const int k, const double threshold) {
  if (k <= 0) {
    throw std::invalid_argument("k must be positive");
  }
  if (!(threshold >= 0.0 && threshold <= 1.0)) {
    throw std::invalid_argument("threshold must be in [0, 1]");
  }
}

FingerprintSearchResult allocateResult(const int numQueries, const int k, cudaStream_t stream) {
  const std::size_t       size = static_cast<std::size_t>(numQueries) * k;
  FingerprintSearchResult result{AsyncDeviceVector<double>(size, stream),
                                 AsyncDeviceVector<std::int64_t>(size, stream)};
  // All-ones bytes are NaN as a double and -1 as an int64, which is the padding for missing hits.
  if (size > 0) {
    cudaCheckError(cudaMemsetAsync(result.scores.data(), 0xff, size * sizeof(double), stream));
    cudaCheckError(cudaMemsetAsync(result.indices.data(), 0xff, size * sizeof(std::int64_t), stream));
  }
  return result;
}

void runSearch(SearchParams& params, const FingerprintSimilarityMetric metric, cudaStream_t stream) {
  if (params.numQueries == 0 || params.numLibrary == 0) {
    return;
  }
  if (metric == FingerprintSimilarityMetric::Tanimoto) {
    launchSearch<FingerprintSimilarityMetric::Tanimoto>(params, stream);
  } else {
    launchSearch<FingerprintSimilarityMetric::Cosine>(params, stream);
  }
}

}  // namespace

FingerprintSearchIndex::FingerprintSearchIndex(const cuda::std::span<const std::uint32_t> fingerprints,
                                               const int                                  numFingerprints,
                                               const int                                  numWords,
                                               const FingerprintSimilarityMetric          metric,
                                               cudaStream_t                               stream)
    : numFingerprints_(numFingerprints),
      numWords_(numWords),
      metric_(metric) {
  const ScopedNvtxRange range("FingerprintSearchIndex build");
  if (numFingerprints < 0 || numWords <= 0) {
    throw std::invalid_argument("Fingerprint matrix dimensions must be non-negative with at least one word");
  }
  if (fingerprints.size() != static_cast<std::size_t>(numFingerprints) * static_cast<std::size_t>(numWords)) {
    throw std::invalid_argument("Fingerprint buffer size does not match its shape");
  }
  if (metric != FingerprintSimilarityMetric::Tanimoto && metric != FingerprintSimilarityMetric::Cosine) {
    throw std::invalid_argument("Unsupported fingerprint similarity metric");
  }
  cudaCheckError(cudaGetDevice(&deviceId_));

  sortRowsByBitCount(fingerprints.data(), numFingerprints, numWords, sortedCounts_, originalIndices_, stream);
  sortedBits_ = AsyncDeviceVector<std::uint32_t>(fingerprints.size(), stream);
  if (numFingerprints > 0) {
    gatherRowsKernel<<<1024, kSetupBlockSize, 0, stream>>>(fingerprints.data(),
                                                           originalIndices_.data(),
                                                           numFingerprints,
                                                           numWords,
                                                           sortedBits_.data());
    cudaCheckError(cudaGetLastError());
  }
  // Searches may run on other streams; the index must be complete before any of them reads it.
  cudaCheckError(cudaStreamSynchronize(stream));
}

FingerprintSearchResult FingerprintSearchIndex::search(const cuda::std::span<const std::uint32_t> queries,
                                                       const int                                  numQueries,
                                                       const int                                  k,
                                                       const double                               threshold,
                                                       const bool                                 prune,
                                                       cudaStream_t                               stream) const {
  const ScopedNvtxRange range("FingerprintSearchIndex search");
  checkSearchArguments(k, threshold);
  checkDevice(deviceId_);
  if (numQueries < 0 || queries.size() != static_cast<std::size_t>(numQueries) * numWords_) {
    throw std::invalid_argument("Query buffer size does not match the index fingerprint width");
  }
  FingerprintSearchResult result = allocateResult(numQueries, k, stream);
  if (numQueries == 0) {
    return result;
  }

  AsyncDeviceVector<int> queryCounts;
  AsyncDeviceVector<int> queryRows;
  sortRowsByBitCount(queries.data(), numQueries, numWords_, queryCounts, queryRows, stream);

  SearchParams params{sortedBits_.data(),
                      sortedCounts_.data(),
                      originalIndices_.data(),
                      numFingerprints_,
                      numWords_,
                      queries.data(),
                      queryRows.data(),
                      queryCounts.data(),
                      queryRows.data(),
                      numQueries,
                      false,
                      k,
                      threshold,
                      prune,
                      result.scores.data(),
                      result.indices.data()};
  runSearch(params, metric_, stream);
  return result;
}

FingerprintSearchResult FingerprintSearchIndex::searchSelf(const int    k,
                                                           const double threshold,
                                                           const bool   prune,
                                                           cudaStream_t stream) const {
  const ScopedNvtxRange range("FingerprintSearchIndex self search");
  checkSearchArguments(k, threshold);
  checkDevice(deviceId_);
  FingerprintSearchResult result = allocateResult(numFingerprints_, k, stream);
  if (numFingerprints_ == 0) {
    return result;
  }

  // The sorted library is already in popcount order: query i reads sorted row i and writes its original row.
  AsyncDeviceVector<int> identity(numFingerprints_, stream);
  iotaKernel<<<(numFingerprints_ + kSetupBlockSize - 1) / kSetupBlockSize, kSetupBlockSize, 0, stream>>>(
    identity.data(),
    numFingerprints_);
  cudaCheckError(cudaGetLastError());

  SearchParams params{sortedBits_.data(),
                      sortedCounts_.data(),
                      originalIndices_.data(),
                      numFingerprints_,
                      numWords_,
                      sortedBits_.data(),
                      identity.data(),
                      sortedCounts_.data(),
                      originalIndices_.data(),
                      numFingerprints_,
                      true,
                      k,
                      threshold,
                      prune,
                      result.scores.data(),
                      result.indices.data()};
  runSearch(params, metric_, stream);
  return result;
}

}  // namespace nvMolKit
