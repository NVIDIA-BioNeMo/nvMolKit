// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <stdexcept>

#include "src/butina.h"
#include "src/butina_common.cuh"
#include "src/fingerprint_similarity_device.cuh"
#include "src/utils/cub_helpers.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {
namespace {

namespace cg = cooperative_groups;

constexpr int kInitialArgMaxBlockSize = 256;
constexpr int kInitialTileSize        = 64;
constexpr int kInitialBlockSize       = 256;
constexpr int kInitialColumnGroups    = kInitialTileSize / 32;
constexpr int kInitialWordChunkSize   = 32;
constexpr int kInitialRowsPerWarp     = kInitialTileSize / (kInitialBlockSize / 32);
constexpr int kMaxGridDimensionY      = 65535;
constexpr int kIterationBlockSize     = 128;

template <FingerprintSimilarityMetric Metric>
__device__ __forceinline__ bool fingerprintsWithinThreshold(const std::uint32_t* lhs,
                                                            const std::uint32_t* rhs,
                                                            int                  lhsBitCount,
                                                            int                  rhsBitCount,
                                                            int                  numWords,
                                                            float                threshold) {
  if (!detail::fingerprintSimilarityCanReach<Metric>(lhsBitCount, rhsBitCount, threshold)) {
    return false;
  }
  int intersection = 0;
  for (int word = 0; word < numWords; ++word) {
    intersection += __popc(lhs[word] & rhs[word]);
  }

  return detail::fingerprintSimilarityAtLeast<Metric>(intersection, lhsBitCount, rhsBitCount, threshold);
}

// Compute and save each fingerprint's set-bit count. Both similarity metrics reuse these counts over and over again.
__global__ void fingerprintBitCountKernel(const cuda::std::span<const std::uint32_t> fingerprints,
                                          const cuda::std::span<int>                 bitCounts,
                                          const cuda::std::span<int>                 originalIndices,
                                          int                                        numWords) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= static_cast<int>(bitCounts.size())) {
    return;
  }
  int bitCount = 0;
  for (int word = 0; word < numWords; ++word) {
    bitCount += __popc(fingerprints[static_cast<std::size_t>(row) * numWords + word]);
  }
  bitCounts[row]       = bitCount;
  originalIndices[row] = row;
}

// Build the initial neighbor counts without storing an N x N matrix. Each 256-thread block handles one upper-triangular
// 64x64 tile. Sorted bit counts help to cheaply reject impossible tiles and pairs because max Tanimoto = min(a, b) /
// max(a, b) and max cosine = sqrt(min(a, b) / max(a, b)). Eight warps cover the 64 rows, with 8 rows handled per warp
// (8 * 32 = 256 threads). Within each warp, its 32 lanes handle 32 columns at a time. All eight warps process their
// rows in parallel, then reuse their lanes for the other 32 columns. Fingerprint words are loaded into shared memory in
// 32-word chunks. Each lane keeps eight intersections and ballots its neighbor decisions into hit masks. Counting those
// masks produces the initial neighbor counts, then argmax chooses the first centroid.
template <FingerprintSimilarityMetric Metric>
__global__ void initialNeighborCountKernel(const cuda::std::span<const std::uint32_t> fingerprints,
                                           const cuda::std::span<const int>           bitCounts,
                                           const cuda::std::span<const int>           sortedBitCounts,
                                           const cuda::std::span<const int>           sortedIndices,
                                           const cuda::std::span<int>                 neighborCounts,
                                           int                                        numWords,
                                           float                                      threshold,
                                           int                                        rowTileOffset) {
  const int n                 = static_cast<int>(neighborCounts.size());
  const int numTiles          = (n + kInitialTileSize - 1) / kInitialTileSize;
  const int firstRowTile      = rowTileOffset + static_cast<int>(blockIdx.y);
  const int secondRowTile     = numTiles - 1 - firstRowTile;
  const int firstRowTileCount = numTiles - firstRowTile;
  const int gridColumn        = blockIdx.x;

  // Due to the symmetric nature of the similarity matrix, pair outer rows of its upper triangle in a compact grid,
  // avoiding almost half the blocks. For an odd tile count, the unused half of the center row returns without repeats.
  int rowTile;
  int columnTile;
  if (gridColumn < firstRowTileCount) {
    rowTile    = firstRowTile;
    columnTile = firstRowTile + gridColumn;
  } else {
    if (firstRowTile == secondRowTile) {
      return;
    }
    rowTile    = secondRowTile;
    columnTile = secondRowTile + gridColumn - firstRowTileCount;
  }

  const int rowStart    = rowTile * kInitialTileSize;
  const int columnStart = columnTile * kInitialTileSize;
  const int rowEnd      = min(rowStart + kInitialTileSize, n) - 1;

  // In an off-diagonal tile, rowEnd and columnStart are the closest bit counts, so they bound its best pair.
  // In a diagonal tile they are the widest-separated counts, so the tile must not be pruned.
  const bool allPairsFeasible =
    detail::fingerprintSimilarityCanReach<Metric>(sortedBitCounts[0], sortedBitCounts[n - 1], threshold);
  if (columnTile != rowTile && !allPairsFeasible &&
      !detail::fingerprintSimilarityCanReach<Metric>(sortedBitCounts[rowEnd],
                                                     sortedBitCounts[columnStart],
                                                     threshold)) {
    return;
  }

  // Pad each row by one word. The 33-word stride spreads lanes across shared-memory banks instead of making them
  // contend for the same bank.
  __shared__ std::uint32_t rowWords[kInitialTileSize][kInitialWordChunkSize + 1];
  __shared__ std::uint32_t columnWords[32][kInitialWordChunkSize + 1];
  __shared__ std::uint32_t hitMasks[kInitialTileSize][kInitialColumnGroups];

  // Each warp handles 8 of the 64 rows. Each lane handles one column.
  const auto warpGroup = cg::tiled_partition<32>(cg::this_thread_block());
  const int  lane      = static_cast<int>(warpGroup.thread_rank());
  const int  warp      = threadIdx.x / warpSize;

  // Handle 32 columns at a time, so two passes cover all 64 columns.
  for (int columnGroup = 0; columnGroup < kInitialColumnGroups; ++columnGroup) {
    // Each lane compares its column against the warp's 8 rows and keeps one running intersection for each row.
    int       intersections[kInitialRowsPerWarp]{};
    bool      canReachThreshold[kInitialRowsPerWarp];
    const int column         = columnStart + columnGroup * warpSize + lane;
    const int columnBitCount = column < n ? (allPairsFeasible ? bitCounts[column] : sortedBitCounts[column]) : 0;
    for (int rowInWarp = 0; rowInWarp < kInitialRowsPerWarp; ++rowInWarp) {
      const int tileRow     = warp * kInitialRowsPerWarp + rowInWarp;
      const int row         = rowStart + tileRow;
      const int rowBitCount = row < n ? (allPairsFeasible ? bitCounts[row] : sortedBitCounts[row]) : 0;
      canReachThreshold[rowInWarp] =
        row < n && column < n &&
        (allPairsFeasible || detail::fingerprintSimilarityCanReach<Metric>(rowBitCount, columnBitCount, threshold));
    }

    // Read 32 words at a time. Each chunk is added to the same running intersections.
    for (int wordStart = 0; wordStart < numWords; wordStart += kInitialWordChunkSize) {
      for (int item = threadIdx.x; item < kInitialTileSize * kInitialWordChunkSize; item += blockDim.x) {
        const int tileRow              = item / kInitialWordChunkSize;
        const int wordInChunk          = item % kInitialWordChunkSize;
        const int row                  = rowStart + tileRow;
        const int sourceWord           = wordStart + wordInChunk;
        const int originalRow          = row < n ? (allPairsFeasible ? row : sortedIndices[row]) : 0;
        rowWords[tileRow][wordInChunk] = row < n && sourceWord < numWords ?
                                           fingerprints[static_cast<std::size_t>(originalRow) * numWords + sourceWord] :
                                           0;
      }
      // Cooperatively load the current 32-column word chunk into shared memory.
      for (int item = threadIdx.x; item < 32 * kInitialWordChunkSize; item += blockDim.x) {
        const int tileColumn     = item / kInitialWordChunkSize;
        const int wordInChunk    = item % kInitialWordChunkSize;
        const int column         = columnStart + columnGroup * warpSize + tileColumn;
        const int sourceWord     = wordStart + wordInChunk;
        const int originalColumn = column < n ? (allPairsFeasible ? column : sortedIndices[column]) : 0;
        columnWords[tileColumn][wordInChunk] =
          column < n && sourceWord < numWords ?
            fingerprints[static_cast<std::size_t>(originalColumn) * numWords + sourceWord] :
            0;
      }
      // Wait until every shared row and column word is available.
      __syncthreads();

      // Add this chunk to the running intersection for each row-column pair.
      for (int rowInWarp = 0; rowInWarp < kInitialRowsPerWarp; ++rowInWarp) {
        if (canReachThreshold[rowInWarp]) {
          const int tileRow = warp * kInitialRowsPerWarp + rowInWarp;
          for (int word = 0; word < kInitialWordChunkSize; ++word) {
            intersections[rowInWarp] += __popc(rowWords[tileRow][word] & columnWords[lane][word]);
          }
        }
      }
      // Wait until all threads finish reading shared memory before the next chunk overwrites it.
      __syncthreads();
    }

    // Convert the accumulated intersections into one ballot mask per tile row.
    for (int rowInWarp = 0; rowInWarp < kInitialRowsPerWarp; ++rowInWarp) {
      const int tileRow     = warp * kInitialRowsPerWarp + rowInWarp;
      const int row         = rowStart + tileRow;
      const int rowBitCount = row < n ? (allPairsFeasible ? bitCounts[row] : sortedBitCounts[row]) : 0;
      const int isNeighbor =
        canReachThreshold[rowInWarp] &&
        detail::fingerprintSimilarityAtLeast<Metric>(intersections[rowInWarp], rowBitCount, columnBitCount, threshold);

      // Pack the warp's 32 yes-or-no neighbor results into one mask.
      const std::uint32_t hitMask = warpGroup.ballot(isNeighbor);
      if (lane == 0) {
        hitMasks[tileRow][columnGroup] = hitMask;
      }
    }
    // Wait until every warp has stored its masks before the row and column reductions read them.
    __syncthreads();
  }

  if (threadIdx.x < kInitialTileSize) {
    const int row = rowStart + threadIdx.x;
    if (row < n) {
      int rowCount = 0;
      // Count the set bits in both masks to get this row's neighbors.
      for (int columnGroup = 0; columnGroup < kInitialColumnGroups; ++columnGroup) {
        rowCount += __popc(hitMasks[threadIdx.x][columnGroup]);
      }
      const int originalRow = allPairsFeasible ? row : sortedIndices[row];
      atomicAdd(&neighborCounts[originalRow], rowCount);
    }

    // Reuse the same masks for the transposed column counts, so off-diagonal pairs are not computed twice.
    if (columnTile != rowTile) {
      const int column = columnStart + threadIdx.x;
      if (column < n) {
        const int columnGroup = threadIdx.x / warpSize;
        const int columnLane  = threadIdx.x % warpSize;
        int       columnCount = 0;
        // Read this column's bit from every row mask to get its neighbors.
        for (int tileRow = 0; tileRow < kInitialTileSize; ++tileRow) {
          columnCount += (hitMasks[tileRow][columnGroup] >> columnLane) & 1U;
        }
        const int originalColumn = allPairsFeasible ? column : sortedIndices[column];
        atomicAdd(&neighborCounts[originalColumn], columnCount);
      }
    }
  }
}

// Select the initial centroid from the neighbor counts.
__global__ void initialArgMaxKernel(const cuda::std::span<const int> neighborCounts, int* maxValue, int* maxIndex) {
  std::uint64_t candidate = 0;
  // Scan neighbor counts in thread-strided order.
  for (int row = threadIdx.x; row < static_cast<int>(neighborCounts.size()); row += blockDim.x) {
    candidate = cubMax{}(candidate, makeButinaCandidate(neighborCounts[row], row));
  }

  // Reduce to the maximum candidate.
  __shared__ typename cub::BlockReduce<std::uint64_t, kInitialArgMaxBlockSize>::TempStorage storage;
  candidate = cub::BlockReduce<std::uint64_t, kInitialArgMaxBlockSize>(storage).Reduce(candidate, cubMax());
  if (threadIdx.x == 0) {
    storeButinaCandidate(candidate, maxValue, maxIndex);
  }
}

// Compare the selected centroid against every active molecule and extract all of its neighbors into the current
// cluster.
template <FingerprintSimilarityMetric Metric>
__global__ void extractClusterKernel(const cuda::std::span<const std::uint32_t> fingerprints,
                                     const cuda::std::span<const int>           bitCounts,
                                     const cuda::std::span<int>                 clusterMembers,
                                     const cuda::std::span<int>                 clusterIds,
                                     int*                                       outputCount,
                                     const int*                                 clusterCount,
                                     const int*                                 maxValue,
                                     const int*                                 maxIndex,
                                     int                                        numWords,
                                     float                                      threshold) {
  // Ignore threads outside the molecule array, molecules already assigned to a cluster, and centroids with only 1
  // neighbor (themselves)
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= static_cast<int>(clusterIds.size()) || clusterIds[row] >= 0 || *maxValue <= 1) {
    return;
  }

  // Compare this active molecule with the selected centroid and test whether their fingerprints meet the similarity
  // threshold.
  const int center = *maxIndex;
  if (!fingerprintsWithinThreshold<Metric>(fingerprints.data() + static_cast<std::size_t>(center) * numWords,
                                           fingerprints.data() + static_cast<std::size_t>(row) * numWords,
                                           bitCounts[center],
                                           bitCounts[row],
                                           numWords,
                                           threshold)) {
    return;
  }

  // Reserve one contiguous output range per warp instead of performing one atomic add per matching thread, reducing
  // the number of atomic operations by up to the warp size.
  const auto matchingThreads = cg::coalesced_threads();
  int        warpOutputStart = 0;

  if (matchingThreads.thread_rank() == 0) {
    warpOutputStart = atomicAdd(outputCount, matchingThreads.num_threads());
  }
  warpOutputStart      = matchingThreads.shfl(warpOutputStart, 0);
  const int outputSlot = warpOutputStart + matchingThreads.thread_rank();

  clusterMembers[outputSlot] = row;
  clusterIds[row]            = *clusterCount;
}

// Record the centroid for the cluster that was just extracted.
// This kernel must be launched with exactly one thread.
__global__ void recordClusterKernel(const int*                 maxValue,
                                    const int*                 maxIndex,
                                    int*                       clusterCount,
                                    const cuda::std::span<int> centroids) {
  // A maximum of one means that every remaining molecule only neighbors itself, so no non-singleton cluster was
  // extracted.
  if (*maxValue <= 1) {
    return;
  }
  const int cluster = (*clusterCount)++;
  if (!centroids.empty()) {
    centroids[cluster] = *maxIndex;
  }
}

// Update every active molecule's neighbor count after removing the current cluster, then produce one maximum candidate
// per block.
template <FingerprintSimilarityMetric Metric>
__global__ void updateNeighborCountsAndBlockMaxKernel(const cuda::std::span<const std::uint32_t> fingerprints,
                                                      const cuda::std::span<const int>           bitCounts,
                                                      const cuda::std::span<const int>           clusterIds,
                                                      const cuda::std::span<int>                 neighborCounts,
                                                      const cuda::std::span<const int>           clusterMembers,
                                                      const int*                                 clusterStart,
                                                      const int*                                 outputCount,
                                                      const int*                                 maxIndex,
                                                      const cuda::std::span<std::uint64_t>       blockMaxima,
                                                      int                                        numWords,
                                                      float                                      threshold) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;

  int updatedCount = -1;
  if (row < static_cast<int>(clusterIds.size()) && clusterIds[row] < 0) {
    int decrement = 0;
    // Only compare against members added to the current cluster in the current iteration step.
    for (int memberSlot = *clusterStart; memberSlot < *outputCount; ++memberSlot) {
      const int removed = clusterMembers[memberSlot];
      // If we are here (active molecule) and we are comparing to the centroid, extractClusterKernel ran previously, and
      // it must not have found this molecule to be within threshold
      if (removed == *maxIndex) {
        continue;
      }
      decrement +=
        fingerprintsWithinThreshold<Metric>(fingerprints.data() + static_cast<std::size_t>(row) * numWords,
                                            fingerprints.data() + static_cast<std::size_t>(removed) * numWords,
                                            bitCounts[row],
                                            bitCounts[removed],
                                            numWords,
                                            threshold);
    }
    // Remove the matching rows from this molecule's stored active-neighbor count.
    updatedCount        = neighborCounts[row] - decrement;
    neighborCounts[row] = updatedCount;
  }

  // Pack the updated neighbor count and original row index, then reduce all candidates in this block to one maximum.
  // Doing this here avoids reading the entire neighbor count array again in a separate kernel.
  __shared__ typename cub::BlockReduce<std::uint64_t, kIterationBlockSize>::TempStorage storage;
  const std::uint64_t                                                                   candidate =
    cub::BlockReduce<std::uint64_t, kIterationBlockSize>(storage).Reduce(makeButinaCandidate(updatedCount, row),
                                                                         cubMax());
  if (threadIdx.x == 0) {
    blockMaxima[blockIdx.x] = candidate;
  }
}

// Reduce the block maxima to select the next centroid.
__global__ void selectNextCentroidKernel(const cuda::std::span<const std::uint64_t> blockMaxima,
                                         int*                                       maxValue,
                                         int*                                       maxIndex,
                                         cudaGraphConditionalHandle                 handle) {
  std::uint64_t candidate = 0;
  // Scan block maxima in thread-strided order.
  for (int block = threadIdx.x; block < static_cast<int>(blockMaxima.size()); block += blockDim.x) {
    candidate = cubMax{}(candidate, blockMaxima[block]);
  }

  __shared__ typename cub::BlockReduce<std::uint64_t, kIterationBlockSize>::TempStorage storage;
  candidate = cub::BlockReduce<std::uint64_t, kIterationBlockSize>(storage).Reduce(candidate, cubMax());
  if (threadIdx.x == 0) {
    // Continue while the selected centroid has an active neighbor besides itself.
    const int maxNeighborCount = storeButinaCandidate(candidate, maxValue, maxIndex);
    cudaGraphSetConditional(handle, maxNeighborCount > 1 ? 1 : 0);
  }
}

// Append all molecules left active after the graph loop as singleton clusters.
__global__ void appendSingletonsKernel(const cuda::std::span<int> clusterIds,
                                       int*                       clusterCount,
                                       const cuda::std::span<int> centroids) {
  // Use one thread so singleton clusters are emitted deterministically from highest to lowest original index.
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int cluster = *clusterCount;
  // Every remaining active molecule has no active neighbor other than itself.
  for (int row = static_cast<int>(clusterIds.size()) - 1; row >= 0; --row) {
    if (clusterIds[row] >= 0) {
      continue;
    }
    clusterIds[row] = cluster;
    if (!centroids.empty()) {
      centroids[cluster] = row;
    }
    ++cluster;
  }
  // Store the final cluster count for the host copy.
  *clusterCount = cluster;
}

// Reordering fused Butina pipeline:
//
// 1) Compute each fingerprint popcount and its initial neighbor count. Similarities are consumed immediately; no
//    N x N similarity, distance, or hit matrix is materialized.
// 2) Find the last active row with the largest neighbor count. This is the first centroid used by the graph.
// 3) Run a conditional CUDA graph WHILE loop entirely on the GPU:
//      a) compare active rows with the selected centroid and append matching rows to the cluster output;
//      b) subtract only those newly removed rows from the surviving neighbor counts;
//      c) reduce a maximum per update block, then reduce those block maxima to select the next centroid;
//      d) update the graph condition from the new maximum count.
//    Fixed original row indices and negative cluster IDs replace the Python boolean-index compactions.
// 4) Once no centroid has another active neighbor, append all remaining active rows as singleton clusters. Cluster IDs
//    and centroids remain in device memory; only the final cluster count is copied to the host.
template <FingerprintSimilarityMetric Metric>
ButinaResult fusedButinaGpuImpl(cuda::std::span<const std::uint32_t> fingerprints,
                                int                                  n,
                                int                                  numWords,
                                float                                threshold,
                                bool                                 returnCentroids,
                                cudaStream_t                         stream) {
  ScopedNvtxRange setupRange("Fused Butina Setup");

  // All (non scalar) buffers are O(N). Fingerprints remain in caller-owned storage and no pairwise matrix is allocated.
  const int              numIterationBlocks = (n + kIterationBlockSize - 1) / kIterationBlockSize;
  AsyncDeviceVector<int> bitCounts(n, stream);
  AsyncDeviceVector<int> sortedBitCounts(n, stream);
  AsyncDeviceVector<int> originalIndices(n, stream);
  AsyncDeviceVector<int> sortedIndices(n, stream);
  AsyncDeviceVector<int> neighborCounts(n, stream);
  AsyncDeviceVector<int> clusterMembers(n, stream);
  ButinaResult result{AsyncDeviceVector<int>(n, stream), AsyncDeviceVector<int>(returnCentroids ? n : 0, stream), 0};
  AsyncDeviceVector<std::uint64_t> blockMaxima(numIterationBlocks, stream);
  AsyncDevicePtr<int>              clusterStart(0, stream);
  AsyncDevicePtr<int>              outputCount(0, stream);
  AsyncDevicePtr<int>              clusterCount(0, stream);
  AsyncDevicePtr<int>              maxValue(-1, stream);
  AsyncDevicePtr<int>              maxIndex(-1, stream);

  const auto bitCountsSpan       = toSpan(bitCounts);
  const auto sortedBitCountsSpan = toSpan(sortedBitCounts);
  const auto originalIndicesSpan = toSpan(originalIndices);
  const auto sortedIndicesSpan   = toSpan(sortedIndices);
  const auto neighborCountsSpan  = toSpan(neighborCounts);
  const auto clusterMembersSpan  = toSpan(clusterMembers);
  const auto clusterIdsSpan      = toSpan(result.clusterIds);
  const auto centroidsSpan       = returnCentroids ? toSpan(result.centroids) : cuda::std::span<int>{};
  const auto blockMaximaSpan     = toSpan(blockMaxima);

  cudaCheckError(cudaMemsetAsync(clusterIdsSpan.data(), 0xff, clusterIdsSpan.size_bytes(), stream));

  // Setup runs once before the graph: cache bit counts, compute initial degrees, and choose the first centroid.
  fingerprintBitCountKernel<<<(n + kIterationBlockSize - 1) / kIterationBlockSize, kIterationBlockSize, 0, stream>>>(
    fingerprints,
    bitCountsSpan,
    originalIndicesSpan,
    numWords);
  cudaCheckError(cudaGetLastError());

  // Sort bit counts and indices to skip impossible tiles/unnecessary work in the downstream kernels
  std::size_t sortStorageBytes = 0;
  cudaCheckError(cub::DeviceRadixSort::SortPairs(nullptr,
                                                 sortStorageBytes,
                                                 bitCountsSpan.data(),
                                                 sortedBitCountsSpan.data(),
                                                 originalIndicesSpan.data(),
                                                 sortedIndicesSpan.data(),
                                                 n,
                                                 0,
                                                 sizeof(int) * 8,
                                                 stream));
  AsyncDeviceVector<std::uint8_t> sortStorage(sortStorageBytes, stream);
  cudaCheckError(cub::DeviceRadixSort::SortPairs(sortStorage.data(),
                                                 sortStorageBytes,
                                                 bitCountsSpan.data(),
                                                 sortedBitCountsSpan.data(),
                                                 originalIndicesSpan.data(),
                                                 sortedIndicesSpan.data(),
                                                 n,
                                                 0,
                                                 sizeof(int) * 8,
                                                 stream));
  cudaCheckError(cudaMemsetAsync(neighborCountsSpan.data(), 0, neighborCountsSpan.size_bytes(), stream));
  const int numInitialTiles    = (n + kInitialTileSize - 1) / kInitialTileSize;
  const int numInitialRowTiles = (numInitialTiles + 1) / 2;
  for (int rowTileOffset = 0; rowTileOffset < numInitialRowTiles; rowTileOffset += kMaxGridDimensionY) {
    const int numRowsInBatch = std::min(kMaxGridDimensionY, numInitialRowTiles - rowTileOffset);
    initialNeighborCountKernel<Metric>
      <<<dim3(numInitialTiles + 1, numRowsInBatch), kInitialBlockSize, 0, stream>>>(fingerprints,
                                                                                    bitCountsSpan,
                                                                                    sortedBitCountsSpan,
                                                                                    sortedIndicesSpan,
                                                                                    neighborCountsSpan,
                                                                                    numWords,
                                                                                    threshold,
                                                                                    rowTileOffset);
    cudaCheckError(cudaGetLastError());
  }
  initialArgMaxKernel<<<1, kInitialArgMaxBlockSize, 0, stream>>>(neighborCountsSpan, maxValue.data(), maxIndex.data());
  cudaCheckError(cudaGetLastError());
  setupRange.pop();

  ScopedNvtxRange            buildRange("Build fused Butina graph");
  const ConditionalLoopGraph graph([&](cudaStream_t captureStream, cudaGraphConditionalHandle handle) {
    cudaCheckError(
      cudaMemcpyAsync(clusterStart.data(), outputCount.data(), sizeof(int), cudaMemcpyDeviceToDevice, captureStream));
    extractClusterKernel<Metric><<<numIterationBlocks, kIterationBlockSize, 0, captureStream>>>(fingerprints,
                                                                                                bitCountsSpan,
                                                                                                clusterMembersSpan,
                                                                                                clusterIdsSpan,
                                                                                                outputCount.data(),
                                                                                                clusterCount.data(),
                                                                                                maxValue.data(),
                                                                                                maxIndex.data(),
                                                                                                numWords,
                                                                                                threshold);
    cudaCheckError(cudaGetLastError());
    recordClusterKernel<<<1, 1, 0, captureStream>>>(maxValue.data(),
                                                    maxIndex.data(),
                                                    clusterCount.data(),
                                                    centroidsSpan);
    cudaCheckError(cudaGetLastError());
    updateNeighborCountsAndBlockMaxKernel<Metric>
      <<<numIterationBlocks, kIterationBlockSize, 0, captureStream>>>(fingerprints,
                                                                      bitCountsSpan,
                                                                      clusterIdsSpan,
                                                                      neighborCountsSpan,
                                                                      clusterMembersSpan,
                                                                      clusterStart.data(),
                                                                      outputCount.data(),
                                                                      maxIndex.data(),
                                                                      blockMaximaSpan,
                                                                      numWords,
                                                                      threshold);
    cudaCheckError(cudaGetLastError());
    selectNextCentroidKernel<<<1, kIterationBlockSize, 0, captureStream>>>(blockMaximaSpan,
                                                                           maxValue.data(),
                                                                           maxIndex.data(),
                                                                           handle);
    cudaCheckError(cudaGetLastError());
  });
  buildRange.pop();

  const ScopedNvtxRange loopRange("Fused Butina graph loop");
  graph.launch(stream);

  appendSingletonsKernel<<<1, 1, 0, stream>>>(clusterIdsSpan, clusterCount.data(), centroidsSpan);
  cudaCheckError(cudaGetLastError());

  // Copy only the result size to the host. The O(N) result buffers remain on the device.
  int clusterCountHost = 0;
  clusterCount.get(clusterCountHost);
  cudaCheckError(cudaStreamSynchronize(stream));

  if (clusterCountHost < 0 || clusterCountHost > n) {
    throw std::runtime_error("Fused Butina produced inconsistent output sizes");
  }

  result.numClusters = clusterCountHost;
  return result;
}

}  // namespace

ButinaResult fusedButinaGpu(cuda::std::span<const std::uint32_t> fingerprints,
                            int                                  numFingerprints,
                            int                                  numWords,
                            double                               cutoff,
                            FingerprintSimilarityMetric          metric,
                            bool                                 returnCentroids,
                            cudaStream_t                         stream) {
  // Validate dimensions before allocating device buffers. The Python wrapper performs dtype and contiguity checks.
  if (numFingerprints < 0 || numWords <= 0) {
    throw std::invalid_argument("Fingerprint matrix dimensions must be non-negative with at least one word");
  }
  if (fingerprints.size() != static_cast<std::size_t>(numFingerprints) * static_cast<std::size_t>(numWords)) {
    throw std::invalid_argument("Fingerprint buffer size does not match its shape");
  }
  if (!(cutoff >= 0.0 && cutoff <= 1.0)) {
    throw std::invalid_argument("cutoff must be in [0, 1]");
  }
  if (metric != FingerprintSimilarityMetric::Tanimoto && metric != FingerprintSimilarityMetric::Cosine) {
    throw std::invalid_argument("Unsupported fingerprint similarity metric");
  }
  if (numFingerprints == 0) {
    return {AsyncDeviceVector<int>(0, stream), AsyncDeviceVector<int>(0, stream), 0};
  }

  // Compile separate metric-specialized graphs so the hot pair predicate contains no runtime metric branch.
  const float threshold = static_cast<float>(1.0 - cutoff);
  if (metric == FingerprintSimilarityMetric::Tanimoto) {
    return fusedButinaGpuImpl<FingerprintSimilarityMetric::Tanimoto>(fingerprints,
                                                                     numFingerprints,
                                                                     numWords,
                                                                     threshold,
                                                                     returnCentroids,
                                                                     stream);
  }
  return fusedButinaGpuImpl<FingerprintSimilarityMetric::Cosine>(fingerprints,
                                                                 numFingerprints,
                                                                 numWords,
                                                                 threshold,
                                                                 returnCentroids,
                                                                 stream);
}

}  // namespace nvMolKit
