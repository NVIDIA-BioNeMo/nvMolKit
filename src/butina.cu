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
#include <cooperative_groups.h>

#include <climits>
#include <cub/cub.cuh>

#include "src/butina.h"
#include "src/utils/cub_helpers.cuh"
#include "src/utils/host_vector.h"
#include "src/utils/nvtx.h"

/**
 * TODO: Future optimizations
 * - Keep a live list of active indices and only dispatch counts for those.
 */
namespace nvMolKit {

namespace {
constexpr int blockSizeCount             = 256;
constexpr int fixedOrderPrepareBlockSize = 512;
constexpr int kSubTileSize               = 8;
constexpr int kMinLoopSizeForAssignment  = 2;

__device__ __forceinline__ void sumCountsAndStoreClusterSize(const int                  tid,
                                                             const int                  pointIdx,
                                                             const cuda::std::span<int> clusterSizes,
                                                             const int                  localCount) {
  __shared__ cub::BlockReduce<int, blockSizeCount>::TempStorage tempStorage;
  const int totalCount = cub::BlockReduce<int, blockSizeCount>(tempStorage).Sum(localCount);
  if (tid == 0) {
    clusterSizes[pointIdx] = totalCount;
  }
}

//! Kernel to count the size of each cluster around each point
//! Assigns singleton clusters to a sentinel value for later processing.
//! Looks up and skips finished clusters.
__global__ void butinaKernelCountClusterSize(const cuda::std::span<const uint8_t> hitMatrix,
                                             const cuda::std::span<int>           clusters,
                                             const cuda::std::span<int>           clusterSizes) {
  const auto tid       = static_cast<int>(threadIdx.x);
  const auto pointIdx  = static_cast<int>(blockIdx.x);
  const auto numPoints = static_cast<int>(clusters.size());

  if (clusters[pointIdx] >= 0) {
    clusterSizes[pointIdx] = 0;
    return;
  }

  const cuda::std::span<const uint8_t> hits = hitMatrix.subspan(static_cast<size_t>(pointIdx) * numPoints, numPoints);
  int                                  localCount = 0;
  for (int i = tid; i < numPoints; i += blockSizeCount) {
    if (hits[i]) {
      const int cluster = clusters[i];
      if (cluster < 0) {
        localCount++;
      }
    }
  }

  sumCountsAndStoreClusterSize(tid, pointIdx, clusterSizes, localCount);
}

//! Kernel to count the size of each cluster around each point, assigning a neighborlist for later use.
//! IMPORTANT: This assumes that the maximum cluster size is small enough to fit in the neighborlist, so should only
//! be called when that is known to be true.
template <int NeighborlistMaxSize>
__global__ void butinaKernelCountClusterSizeWithNeighborlist(const cuda::std::span<const uint8_t> hitMatrix,
                                                             const cuda::std::span<int>           clusters,
                                                             const cuda::std::span<int>           clusterSizes,
                                                             const cuda::std::span<int>           neighborList) {
  static_assert(NeighborlistMaxSize % kSubTileSize == 0, "NeighborlistMaxSize must be multiple of kSubTileSize");
  const auto tid       = static_cast<int>(threadIdx.x);
  const auto pointIdx  = static_cast<int>(blockIdx.x);
  const auto numPoints = static_cast<int>(clusters.size());

  __shared__ int neighborlistIndex;
  __shared__ int sharedNeighborlist[NeighborlistMaxSize];

  if (tid == 0) {
    neighborlistIndex = 0;
  }
  if (clusters[pointIdx] >= 0) {
    clusterSizes[pointIdx] = 0;
    return;
  }

  const cuda::std::span<const uint8_t> hits = hitMatrix.subspan(static_cast<size_t>(pointIdx) * numPoints, numPoints);
  int                                  localCount = 0;
  __syncthreads();  // for neighborlistIndex init
  for (int i = tid; i < numPoints; i += blockSizeCount) {
    if (hits[i]) {
      const int cluster = clusters[i];
      if (cluster < 0) {
        localCount++;
        const int index           = atomicAdd(&neighborlistIndex, 1);
        sharedNeighborlist[index] = i;
      }
    }
  }

  // Coalesced write of neighborlist using loop for variable sizes
  __syncthreads();  // for sharedNeighborlist final value
  for (int i = tid; i < NeighborlistMaxSize; i += blockSizeCount) {
    neighborList[pointIdx * NeighborlistMaxSize + i] = (i < neighborlistIndex) ? sharedNeighborlist[i] : -1;
  }

  sumCountsAndStoreClusterSize(tid, pointIdx, clusterSizes, localCount);
}

namespace cg = cooperative_groups;

constexpr int blockSizeAssign      = 128;
constexpr int kTilesPerBlockAssign = blockSizeAssign / kSubTileSize;

template <int NeighborlistMaxSize>
__global__ void attemptAssignClustersFromNeighborlist(const cuda::std::span<int>       clusters,
                                                      const cuda::std::span<const int> clusterSizes,
                                                      const cuda::std::span<const int> neighborList,
                                                      const cuda::std::span<int>       centroids,
                                                      const int*                       designatedMaxIdx,
                                                      int*                             nextClusterIdx) {
  static_assert(NeighborlistMaxSize % kSubTileSize == 0, "NeighborlistMaxSize must be multiple of kSubTileSize");

  const auto     tile8       = cg::tiled_partition<kSubTileSize>(cg::this_thread_block());
  const int      rankInBlock = tile8.meta_group_rank();
  const int      tid         = tile8.thread_rank();
  __shared__ int candidateNeighborsBlock[kTilesPerBlockAssign][NeighborlistMaxSize];
  __shared__ int foundIssueBlock[kTilesPerBlockAssign];

  int* sharedFoundIssue         = &foundIssueBlock[rankInBlock];
  int* sharedCandidateNeighbors = &candidateNeighborsBlock[rankInBlock][0];

  if (tid == 0) {
    foundIssueBlock[rankInBlock] = 0;
  }

  // For global tile index across the grid:
  constexpr int tilesPerBlock = blockSizeAssign / kSubTileSize;
  const int     pointIdx      = blockIdx.x * tilesPerBlock + rankInBlock;
  if (pointIdx >= clusters.size()) {
    return;
  }

  const int clustId = clusters[pointIdx];
  if (clustId >= 0) {
    return;
  }
  const int clusterSize     = clusterSizes[pointIdx];
  const int isDesignatedMax = (pointIdx == *designatedMaxIdx);

  // Load neighborlist into shared memory using loop for variable sizes
  for (int i = tid; i < NeighborlistMaxSize; i += kSubTileSize) {
    sharedCandidateNeighbors[i] = neighborList[pointIdx * NeighborlistMaxSize + i];
  }
  tile8.sync();

  for (int i = 0; i < clusterSize; i++) {
    const int candidateNeighbor            = sharedCandidateNeighbors[i];
    const int candidateNeighborClusterSize = clusterSizes[candidateNeighbor];

    // If neighbor has larger cluster, they should be processed instead
    if (candidateNeighborClusterSize > clusterSize) {
      return;
    }

    // If neighbor has SAME cluster size and lower index, defer to them for consistency
    // Also defer if neighbor is the designated max (guarantees only designated max assigns among ties)
    // Designated max itself skips this check to guarantee forward progress
    if (!isDesignatedMax && candidateNeighborClusterSize == clusterSize &&
        (candidateNeighbor < pointIdx || candidateNeighbor == *designatedMaxIdx)) {
      return;
    }

    // If neighbor has smaller cluster size, we're the better centroid - continue

    // Now we verify that all of these neighbors have the same or fewer neighbors we do. Each thread checks 1 candidate
    // at a time. This will rule out our neighbors being connected to a larger cluster.
    for (int oidx = tid; oidx < candidateNeighborClusterSize; oidx += kSubTileSize) {
      const int otherNeighbor = neighborList[candidateNeighbor * NeighborlistMaxSize + oidx];
      bool      foundMatch    = false;
      // One of the neighbors will be ourselves, by definition.
      if (otherNeighbor == pointIdx) {
        foundMatch = true;
      } else {
        for (int j = 0; j < clusterSize; j++) {
          if (otherNeighbor == sharedCandidateNeighbors[j]) {
            foundMatch = true;
            break;
          }
        }
      }
      if (!foundMatch) {
        // We might still be ok if that neighbor is a smaller cluster.
        // Designated max only bails on strictly larger (which can't happen for the true max).
        if (clusterSizes[otherNeighbor] > clusterSize ||
            (clusterSizes[otherNeighbor] == clusterSize && !isDesignatedMax)) {
          atomicExch(sharedFoundIssue, 1);
        }
      }
    }
    tile8.sync();
    if (*sharedFoundIssue) {
      return;
    }
  }

  // At this point, we have a valid cluster. Assign it.
  int clusterVal;
  if (tid == 0) {
    clusterVal         = atomicAdd(nextClusterIdx, 1);
    clusters[pointIdx] = clusterVal;
    if (!centroids.empty()) {
      centroids[clusterVal] = pointIdx;
    }
  }
  tile8.sync();
  clusterVal = tile8.shfl(clusterVal, 0);
  // Assign neighbors using loop for variable sizes
  for (int i = tid; i < clusterSize; i += kSubTileSize) {
    const int assignIdx = sharedCandidateNeighbors[i];
    if (clusters[assignIdx] < 0) {
      clusters[assignIdx] = clusterVal;
    }
  }
}

//! Kernel to write the cluster assignment for the largest cluster found
__global__ void butinaWriteClusterValue(const cuda::std::span<const uint8_t> hitMatrix,
                                        const cuda::std::span<int>           clusters,
                                        const cuda::std::span<int>           centroids,
                                        const int*                           centralIdx,
                                        const int*                           clusterIdx,
                                        const int*                           maxClusterSize) {
  const size_t numPoints = clusters.size();
  const size_t tid       = threadIdx.x + blockIdx.x * blockDim.x;
  const int    clusterSz = *maxClusterSize;
  if (clusterSz < kMinLoopSizeForAssignment) {
    return;
  }
  const int pointIdx = *centralIdx;
  if (pointIdx < 0) {
    return;
  }
  const int                            clusterVal = *clusterIdx;
  const cuda::std::span<const uint8_t> hits = hitMatrix.subspan(static_cast<size_t>(pointIdx) * numPoints, numPoints);
  if (tid < numPoints) {
    if (hits[tid]) {
      if (clusters[tid] < 0) {
        clusters[tid] = clusterVal;
      }
    }
  }
  if (tid == 0 && !centroids.empty()) {
    centroids[clusterVal] = pointIdx;
  }
}

//! Kernel to increment cluster index after assignment. Must be launched with <<<1, 1>>>.
__global__ void bumpClusterIdxKernel(int* clusterIdx, const int* lastClusterSize) {
  if (*lastClusterSize >= kMinLoopSizeForAssignment) {
    *clusterIdx += 1;
  }
}

constexpr int kSingletonBlockSize = 512;

//! Assign all remaining unassigned points their own singleton cluster IDs.
__global__ void assignSingletonIdsKernel(const cuda::std::span<int> clusters,
                                         const cuda::std::span<int> centroids,
                                         int*                       nextClusterIdx) {
  __shared__ int sharedClusterIdx;
  const int      tid       = threadIdx.x;
  const int      numPoints = static_cast<int>(clusters.size());

  if (tid == 0) {
    sharedClusterIdx = *nextClusterIdx;
  }
  __syncthreads();

  for (int i = tid; i < numPoints; i += kSingletonBlockSize) {
    if (clusters[i] < 0) {
      const int myClusterIdx = atomicAdd(&sharedClusterIdx, 1);
      clusters[i]            = myClusterIdx;
      if (!centroids.empty()) {
        centroids[myClusterIdx] = i;
      }
    }
  }

  __syncthreads();
  if (tid == 0) {
    *nextClusterIdx = sharedClusterIdx;
  }
}

//! Count the size of each cluster and store the result in clusterSizes.
__global__ void countClusterSizesKernel(const cuda::std::span<const int> clusters,
                                        const cuda::std::span<int>       clusterSizes) {
  const int numPoints = static_cast<int>(clusters.size());
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numPoints; i += blockDim.x * gridDim.x) {
    const int clusterId = clusters[i];
    atomicAdd(&clusterSizes[clusterId], 1);
  }
}

//! Build the remapping array from sorted cluster IDs. After sorting by (-size, originalId),
//! the position in the sorted array is the new cluster ID.
__global__ void createNewIndexMapping(const cuda::std::span<const int> sortedOriginalIds,
                                      const cuda::std::span<int>       remap) {
  const int numClusters = static_cast<int>(sortedOriginalIds.size());
  for (int newId = blockIdx.x * blockDim.x + threadIdx.x; newId < numClusters; newId += blockDim.x * gridDim.x) {
    const int originalId = sortedOriginalIds[newId];
    remap[originalId]    = newId;
  }
}

//! Apply the remapping to all cluster assignments.
__global__ void applyNewIndices(const cuda::std::span<int> clusters, const cuda::std::span<const int> remap) {
  const int numPoints = static_cast<int>(clusters.size());
  const int tid       = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid < numPoints) {
    clusters[tid] = remap[clusters[tid]];
  }
}

__global__ void remapCentroidsKernel(const cuda::std::span<const int> sortedOriginalIds,
                                     const cuda::std::span<const int> centroids,
                                     const cuda::std::span<int>       remappedCentroids) {
  const int numClusters = static_cast<int>(sortedOriginalIds.size());
  const int idx         = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < numClusters) {
    const int originalId   = sortedOriginalIds[idx];
    remappedCentroids[idx] = centroids[originalId];
  }
}

//! Setup sort keys for cluster renumbering: keys[i] = -sizes[i] (for descending), ids[i] = i
__global__ void setupSortKeysKernel(const cuda::std::span<const int> sizes,
                                    const cuda::std::span<int>       keys,
                                    const cuda::std::span<int>       ids) {
  const int numClusters = static_cast<int>(sizes.size());
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numClusters; idx += blockDim.x * gridDim.x) {
    keys[idx] = -sizes[idx];
    ids[idx]  = idx;
  }
}

/**
 * @brief Renumber cluster IDs so larger clusters have smaller IDs
 *
 * 1. Maps clusters by size to cluster ID.
 * 2. Then sorts by size (descending)
 * 3. Creates mapping of old ID -> new ID based on sorted order.
 * 4. Applies new IDs to all points.
 */
void renumberClustersBySize(const cuda::std::span<int> clusters,
                            const cuda::std::span<int> centroids,
                            const int                  numClusters,
                            cudaStream_t               stream) {
  if (numClusters <= 1) {
    return;
  }

  const int numPoints = static_cast<int>(clusters.size());

  AsyncDeviceVector<int> clusterSizes(numClusters, stream);
  AsyncDeviceVector<int> sortKeys(numClusters, stream);
  AsyncDeviceVector<int> originalIds(numClusters, stream);
  AsyncDeviceVector<int> sortedOriginalIds(numClusters, stream);

  clusterSizes.zero();

  constexpr int blockSize         = 256;
  const int     numBlocksRenumber = (numClusters + blockSize - 1) / blockSize;

  // Count cluster sizes
  countClusterSizesKernel<<<numBlocksRenumber, blockSize, 0, stream>>>(clusters, toSpan(clusterSizes));
  cudaCheckError(cudaGetLastError());

  // Prepare sort keys: negative size for descending order
  setupSortKeysKernel<<<numBlocksRenumber, blockSize, 0, stream>>>(toSpan(clusterSizes),
                                                                   toSpan(sortKeys),
                                                                   toSpan(originalIds));
  cudaCheckError(cudaGetLastError());

  // Sort by (negative size, original id) to get descending size order with stable tiebreak
  // Reuse clusterSizes as sortedKeys output (we never read the sorted keys)
  std::size_t sortTempBytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr,
                                  sortTempBytes,
                                  sortKeys.data(),
                                  clusterSizes.data(),
                                  originalIds.data(),
                                  sortedOriginalIds.data(),
                                  numClusters,
                                  0,
                                  sizeof(int) * 8,
                                  stream);
  const AsyncDeviceVector<uint8_t> sortTemp(sortTempBytes, stream);
  cub::DeviceRadixSort::SortPairs(sortTemp.data(),
                                  sortTempBytes,
                                  sortKeys.data(),
                                  clusterSizes.data(),
                                  originalIds.data(),
                                  sortedOriginalIds.data(),
                                  numClusters,
                                  0,
                                  sizeof(int) * 8,
                                  stream);
  cudaCheckError(cudaGetLastError());

  // Build remap: remap[originalId] = newId
  // Reuse sortKeys as remap (sortKeys is unused after the sort)
  const auto remap = toSpan(sortKeys);
  createNewIndexMapping<<<numBlocksRenumber, blockSize, 0, stream>>>(toSpan(sortedOriginalIds), remap);
  cudaCheckError(cudaGetLastError());

  // Apply new indices to all points
  const int numBlocks = (numPoints + blockSize - 1) / blockSize;
  applyNewIndices<<<numBlocks, blockSize, 0, stream>>>(clusters, remap);
  cudaCheckError(cudaGetLastError());

  if (!centroids.empty()) {
    AsyncDeviceVector<int> remappedCentroids(numClusters, stream);
    remapCentroidsKernel<<<numBlocksRenumber, blockSize, 0, stream>>>(toSpan(sortedOriginalIds),
                                                                      centroids,
                                                                      toSpan(remappedCentroids));
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaMemcpyAsync(centroids.data(),
                                   remappedCentroids.data(),
                                   numClusters * sizeof(int),
                                   cudaMemcpyDeviceToDevice,
                                   stream));
  }
}

// Build packed sort keys: higher hit count (primary key), then higher point index (secondary key).
// Bits 32-63 hold the hit count; bits 0-31 hold the point index and can be recovered with a bit mask.
__global__ void setupFixedOrderSortKeysKernel(const cuda::std::span<const int> hitCounts,
                                              const cuda::std::span<uint64_t>  sortKeys) {
  const int numPoints = hitCounts.size();
  const int idx       = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numPoints) {
    const uint64_t count = hitCounts[idx];
    const uint64_t point = idx;
    sortKeys[idx]        = (count << 32) | point;
  }
}

// Sort candidates once. Only one sort needed since this kernel will be used for the reordering = False version of clustering
void sortFixedOrderCandidates(const cuda::std::span<const int> hitCounts,
                              const cuda::std::span<uint64_t>  sortKeys,
                              const cuda::std::span<uint64_t>  sortedKeys,
                              cudaStream_t                     stream) {
  const int numPoints = hitCounts.size();
  if (numPoints == 0) {
    return;
  }

  constexpr int blockSize = 256;

  // make enough blocks so that we have enough threads for radix sort
  const int numBlocks = (numPoints + blockSize - 1) / blockSize;
  setupFixedOrderSortKeysKernel<<<numBlocks, blockSize, 0, stream>>>(hitCounts, sortKeys);
  cudaCheckError(cudaGetLastError());

  std::size_t sortTempBytes = 0;

  // 1) find how much temp memory is needed for the radix sort (nullptr passed in as parameter)
  cub::DeviceRadixSort::SortKeysDescending(nullptr,
                                           sortTempBytes,
                                           sortKeys.data(),
                                           sortedKeys.data(),
                                           numPoints,
                                           0,
                                           sizeof(uint64_t) * 8,
                                           stream);

  // 2) allocate the temp memory (note that even though the allocation is async, we are using the same stream
  // so FIFO order is maintained and mem will be allocated before the next sorting step)
  const AsyncDeviceVector<uint8_t> sortTemp(sortTempBytes, stream);

  // 3) run the radix sort
  cub::DeviceRadixSort::SortKeysDescending(sortTemp.data(),
                                           sortTempBytes,
                                           sortKeys.data(),
                                           sortedKeys.data(),
                                           numPoints,
                                           0,
                                           sizeof(uint64_t) * 8,
                                           stream);
  cudaCheckError(cudaGetLastError());
}

// Select the next fixed-order centroid. We use fixedOrderPrepareBlockSize threads and process
// the vector chunk by chunk (processing of each chunk is parallelized).
__global__ void prepareFixedOrderCandidateKernel(const cuda::std::span<const uint64_t> sortedKeys,
                                                 const cuda::std::span<const int>      hitCounts,
                                                 const cuda::std::span<int>            clusters,
                                                 const cuda::std::span<int>            centroids,
                                                 int*                                  cursor,
                                                 int*                                  activePoint,
                                                 int*                                  activeCluster,
                                                 int*                                  nextClusterIdx,
                                                 int*                                  keepGoing) {
  const int numPoints = clusters.size();
  const int tid       = threadIdx.x;

  // we should be launching this kernel with only one block regardless
  if (blockIdx.x != 0) {
    return;
  }

  __shared__ cub::BlockReduce<int, fixedOrderPrepareBlockSize>::TempStorage tempStorage;
  __shared__ int                                                            firstPos;

  if (tid == 0) {
    *activePoint   = -1;
    *activeCluster = -1;
  }

  int base = *cursor;
  while (base < numPoints) {
    const int sortedPos = base + tid;

    // INT_MAX will be used as identity later when we do the block reduce with cubMin()
    int candidate = INT_MAX;

    if (sortedPos < numPoints) {
      // Use the bit mask to get the point index from the packed sort key.
      const int pointIdx = sortedKeys[sortedPos] & 0xffffffffULL;
      // Points already assigned to a cluster cannot become centroids.
      if (clusters[pointIdx] < 0) {
        candidate = sortedPos;
      }
    }

    // now we reduce all those indexes and get the smallest one (corresponds to first valid element in the sorted hit
    // count vector)
    const int reducedFirstPos =
      cub::BlockReduce<int, fixedOrderPrepareBlockSize>(tempStorage).Reduce(candidate, cubMin());
    if (tid == 0) {
      firstPos = reducedFirstPos;
    }
    // sync is necessary for 1) allows us to reuse temp storage the next while loop iteration with no issues, 2) tid==0
    // shares the reduced value with the other threads
    __syncthreads();

    if (firstPos == INT_MAX) {
      // we have not yet found a valid candidate
      base += fixedOrderPrepareBlockSize;
      continue;
    }

    const int pointIdx     = sortedKeys[firstPos] & 0xffffffffULL;
    bool      hasNeighbors = hitCounts[pointIdx] > 1;

    if (tid == 0) {
      // update the cursor all the way to the next possible valid idx
      *cursor = firstPos + 1;

      const int clusterVal = *nextClusterIdx;
      *nextClusterIdx += 1;

      if (hasNeighbors) {
        // Prepare the parameters for the parallel row scan.
        *activePoint   = pointIdx;
        *activeCluster = clusterVal;
        *keepGoing     = 1;
      } else {
        // this is a singleton, no need for the parallel row scan, we can instantly assign the cluster
        clusters[pointIdx] = clusterVal;
        if (!centroids.empty()) {
          centroids[clusterVal] = pointIdx;
        }
      }
    }

    // threads should exit to run the parallel row scan
    if (hasNeighbors) {
      return;
    }

    base = firstPos + 1;
  }

  if (tid == 0) {
    *cursor    = numPoints;
    *keepGoing = 0;
  }
}

// Parallel Scan. Assign still-unassigned neighbors to the cluster belonging to the centroid we just selected
__global__ void assignFixedOrderActiveRowKernel(const cuda::std::span<const uint8_t> hitMatrix,
                                                const cuda::std::span<const int>     hitCounts,
                                                const cuda::std::span<int>           clusters,
                                                const cuda::std::span<int>           centroids,
                                                const int*                           activePoint,
                                                const int*                           activeCluster) {
  const int pointIdx   = *activePoint;
  const int clusterVal = *activeCluster;
  if (pointIdx < 0 || clusterVal < 0) {
    return;
  }

  const int numPoints = clusters.size();
  const int tid       = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    // We make the selected point the centroid and assign it to a cluster.
    clusters[pointIdx] = clusterVal;
    if (!centroids.empty()) {
      // This is only filled when the caller requests centroids.
      centroids[clusterVal] = pointIdx;
    }
  }

  // If hit_count == 1, no neighbors to scan this point only hit itself
  if (hitCounts[pointIdx] <= 1) {
    return;
  }

  const cuda::std::span<const uint8_t> hits = hitMatrix.subspan(static_cast<size_t>(pointIdx) * numPoints, numPoints);

  // if this point is a neighbor of the selected centroid and we have not yet selected a cluster for this point, assign
  // this point to the same cluster as the centroid we just selected
  if (tid < numPoints && hits[tid] && clusters[tid] < 0) {
    clusters[tid] = clusterVal;
  }
}

// Update the graph WHILE condition.
__global__ void checkFixedOrderGraphConditionKernel(cudaGraphConditionalHandle handle, const int* keepGoing) {
  cudaGraphSetConditional(handle, (*keepGoing != 0) ? 1 : 0);
}

void validateButinaInputs(const cuda::std::span<const uint8_t> hitMatrix,
                          const cuda::std::span<int>           clusters,
                          const cuda::std::span<int>           centroids) {
  const size_t numPoints = clusters.size();
  if (!centroids.empty() && centroids.size() != numPoints) {
    throw std::invalid_argument("Centroids size mismatch: " + std::to_string(centroids.size()) +
                                " != " + std::to_string(numPoints));
  }
  if (const size_t matSize = hitMatrix.size(); numPoints * numPoints != matSize) {
    throw std::runtime_error("Butina size mismatch" + std::to_string(numPoints) +
                             " points^2 != " + std::to_string(matSize) + " neighbor matrix size");
  }
}

}  // namespace

#if CUB_VERSION < 200800
constexpr int argMaxBlockSize = 512;

//! Custom ArgMax kernel that returns the largest value and index.
//! Used when CUB's new ArgMax API is not available (CCCL < 2.8.0)
__global__ void lastArgMaxKernel(const int* values, int numItems, int* outVal, int* outIdx) {
  int            maxVal = cuda::std::numeric_limits<int>::min();
  int            maxID  = -1;
  __shared__ int foundMaxVal[argMaxBlockSize];
  __shared__ int foundMaxIds[argMaxBlockSize];
  const auto     tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < numItems; i += argMaxBlockSize) {
    if (const int val = values[i]; val >= maxVal) {
      maxID  = i;
      maxVal = val;
    }
  }
  foundMaxVal[tid] = maxVal;
  foundMaxIds[tid] = maxID;

  __shared__ cub::BlockReduce<int, argMaxBlockSize>::TempStorage storage;
  const int actualMaxVal = cub::BlockReduce<int, argMaxBlockSize>(storage).Reduce(maxVal, cubMax());
  __syncthreads();  // For shared memory write of maxVal and maxID
  if (tid == 0) {
    *outVal = actualMaxVal;
    for (int i = argMaxBlockSize - 1; i >= 0; i--) {
      if (foundMaxVal[i] == actualMaxVal) {
        *outIdx = foundMaxIds[i];
        break;
      }
    }
  }
}
#endif  // CUB_VERSION < 200800

//! Helper class to run ArgMax on device data.
//! Uses CUB's DeviceReduce::ArgMax when available (CCCL >= 2.8.0), otherwise falls back to custom kernel.
class ArgMaxRunner {
 public:
  ArgMaxRunner([[maybe_unused]] size_t num_items, cudaStream_t stream)
      : stream_(stream)
#if CUB_VERSION >= 200800
        ,
        temp_storage_(getTempStorageSize(num_items, stream), stream)
#endif
  {
  }

  void operator()(int* d_in, int* d_max_value_out, int* d_max_index_out, int num_items) {
#if CUB_VERSION >= 200800
    size_t temp_storage_bytes = temp_storage_.size();
    cudaCheckError(cub::DeviceReduce::ArgMax(temp_storage_.data(),
                                             temp_storage_bytes,
                                             d_in,
                                             d_max_value_out,
                                             d_max_index_out,
                                             static_cast<int64_t>(num_items),
                                             stream_));
#else
    lastArgMaxKernel<<<1, argMaxBlockSize, 0, stream_>>>(d_in, num_items, d_max_value_out, d_max_index_out);
    cudaCheckError(cudaGetLastError());
#endif
  }

  //! Run ArgMax on a specific stream (used during graph capture)
  void captureOn(cudaStream_t captureStream, int* d_in, int* d_max_value_out, int* d_max_index_out, int num_items) {
#if CUB_VERSION >= 200800
    size_t temp_storage_bytes = temp_storage_.size();
    cudaCheckError(cub::DeviceReduce::ArgMax(temp_storage_.data(),
                                             temp_storage_bytes,
                                             d_in,
                                             d_max_value_out,
                                             d_max_index_out,
                                             static_cast<int64_t>(num_items),
                                             captureStream));
#else
    lastArgMaxKernel<<<1, argMaxBlockSize, 0, captureStream>>>(d_in, num_items, d_max_value_out, d_max_index_out);
    cudaCheckError(cudaGetLastError());
#endif
  }

 private:
#if CUB_VERSION >= 200800
  static size_t getTempStorageSize(size_t num_items, cudaStream_t stream) {
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::ArgMax(nullptr,
                              temp_storage_bytes,
                              static_cast<int*>(nullptr),
                              static_cast<int*>(nullptr),
                              static_cast<int*>(nullptr),
                              static_cast<int64_t>(num_items),
                              stream);
    return temp_storage_bytes;
  }
#endif

  cudaStream_t stream_;
#if CUB_VERSION >= 200800
  AsyncDeviceVector<uint8_t> temp_storage_;
#endif
};

/**
 * @brief Prune neighborlists by removing assigned neighbors and reordering.
 */
template <int NeighborlistMaxSize>
__global__ void pruneNeighborlistKernel(const cuda::std::span<int> clusters,
                                        const cuda::std::span<int> clusterSizes,
                                        const cuda::std::span<int> neighborList) {
  constexpr int kWarpSize       = 32;
  constexpr int kItemsPerThread = (NeighborlistMaxSize + kWarpSize - 1) / kWarpSize;
  static_assert(NeighborlistMaxSize <= 128, "NeighborlistMaxSize must be <= 128");
  static_assert(NeighborlistMaxSize % 8 == 0, "NeighborlistMaxSize must be multiple of 8");

  using WarpMergeSort = cub::WarpMergeSort<int, kItemsPerThread, kWarpSize, int>;
  using WarpReduce    = cub::WarpReduce<int>;

  constexpr int                                  kWarpsPerBlock = 4;
  __shared__ typename WarpMergeSort::TempStorage sortStorage[kWarpsPerBlock];
  __shared__ WarpReduce::TempStorage reduceStorage[kWarpsPerBlock];

  const auto tile     = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  const int  tid      = tile.thread_rank();
  const int  warpId   = tile.meta_group_rank();
  const int  pointIdx = blockIdx.x * kWarpsPerBlock + warpId;

  if (pointIdx >= static_cast<int>(clusters.size())) {
    return;
  }

  if (clusters[pointIdx] >= 0) {
    clusterSizes[pointIdx] = 0;
    return;
  }

  const int currentSize = clusterSizes[pointIdx];
  const int baseOffset  = pointIdx * NeighborlistMaxSize;

  // Each thread loads kItemsPerThread neighbors in blocked arrangement
  int keys[kItemsPerThread];
  int values[kItemsPerThread];

  for (int item = 0; item < kItemsPerThread; ++item) {
    const int globalIdx = tid * kItemsPerThread + item;
    if (globalIdx < NeighborlistMaxSize) {
      values[item]     = neighborList[baseOffset + globalIdx];
      const bool valid = (globalIdx < currentSize) && (values[item] >= 0) && (clusters[values[item]] < 0);
      keys[item]       = valid ? 0 : 1;  // 0 = valid (sort first), 1 = invalid (sort last)
    } else {
      values[item] = -1;
      keys[item]   = 1;
    }
  }

  // Sort by key ascending: valid neighbors (key=0) come first
  WarpMergeSort(sortStorage[warpId]).Sort(keys, values, cubLess{});

  // Count valid entries across all items in this thread
  int localValidCount = 0;
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int globalIdx = tid * kItemsPerThread + item;
    if (globalIdx < NeighborlistMaxSize && keys[item] == 0) {
      ++localValidCount;
    }
  }

  int newCount = WarpReduce(reduceStorage[warpId]).Sum(localValidCount);
  newCount     = tile.shfl(newCount, 0);

  if (tid == 0) {
    clusterSizes[pointIdx] = newCount;
  }

  for (int item = 0; item < kItemsPerThread; ++item) {
    const int globalIdx = tid * kItemsPerThread + item;
    if (globalIdx < NeighborlistMaxSize) {
      neighborList[baseOffset + globalIdx] = (globalIdx < newCount) ? values[item] : -1;
    }
  }
}

// TODO - consolidate this to device vector code.
template <typename T> __global__ void setAllKernel(const size_t numElements, T value, T* dst) {
  const size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = value;
  }
}
template <typename T> void setAll(const cuda::std::span<T>& vec, const T& value, cudaStream_t stream) {
  const size_t numElements = vec.size();
  if (numElements == 0) {
    return;
  }
  constexpr int blockSize = 128;
  const size_t  numBlocks = (numElements + blockSize - 1) / blockSize;
  setAllKernel<<<numBlocks, blockSize, 0, stream>>>(numElements, value, vec.data());
  cudaCheckError(cudaGetLastError());
}

//! Kernel to check loop condition and set the conditional handle for CUDA Graph WHILE node.
//! This runs at the END of each loop iteration to determine if the next iteration should execute.
__global__ void checkLoopConditionKernel(cudaGraphConditionalHandle handle, const int* maxValue, int threshold) {
  // Continue looping if maxValue >= threshold
  cudaGraphSetConditional(handle, (*maxValue >= threshold) ? 1 : 0);
}

//! Inner loop iteration for Butina clustering.
void innerButinaLoop(const int                            numPoints,
                     const cuda::std::span<const uint8_t> hitMatrix,
                     const cuda::std::span<int>           clusters,
                     const cuda::std::span<int>           clusterSizesSpan,
                     const cuda::std::span<int>           centroids,
                     int*                                 maxIndexPtr,
                     int*                                 maxValuePtr,
                     int*                                 clusterIdxPtr,
                     ArgMaxRunner&                        argMaxRunner,
                     cudaStream_t                         stream) {
  const int numBlocksFlat = ((static_cast<int>(clusterSizesSpan.size()) - 1) / blockSizeCount) + 1;

  butinaKernelCountClusterSize<<<numPoints, blockSizeCount, 0, stream>>>(hitMatrix, clusters, clusterSizesSpan);
  cudaCheckError(cudaGetLastError());

  argMaxRunner.captureOn(stream,
                         clusterSizesSpan.data(),
                         maxValuePtr,
                         maxIndexPtr,
                         static_cast<int>(clusterSizesSpan.size()));

  butinaWriteClusterValue<<<numBlocksFlat, blockSizeCount, 0, stream>>>(hitMatrix,
                                                                        clusters,
                                                                        centroids,
                                                                        maxIndexPtr,
                                                                        clusterIdxPtr,
                                                                        maxValuePtr);
  cudaCheckError(cudaGetLastError());
  bumpClusterIdxKernel<<<1, 1, 0, stream>>>(clusterIdxPtr, maxValuePtr);
  cudaCheckError(cudaGetLastError());
}

//! Inner loop iteration that attempts assignment then prunes neighborlists.
template <int NeighborlistMaxSize>
void innerButinaLoopWithPruning(const int                  numPoints,
                                const cuda::std::span<int> clusters,
                                const cuda::std::span<int> clusterSizesSpan,
                                const cuda::std::span<int> centroids,
                                int*                       maxIndexPtr,
                                int*                       maxValuePtr,
                                int*                       clusterIdxPtr,
                                const cuda::std::span<int> neighborList,
                                ArgMaxRunner&              argMaxRunner,
                                cudaStream_t               stream) {
  const int numBlocksAssign = (numPoints + kTilesPerBlockAssign - 1) / kTilesPerBlockAssign;
  attemptAssignClustersFromNeighborlist<NeighborlistMaxSize>
    <<<numBlocksAssign, blockSizeAssign, 0, stream>>>(clusters,
                                                      clusterSizesSpan,
                                                      neighborList,
                                                      centroids,
                                                      maxIndexPtr,
                                                      clusterIdxPtr);
  cudaCheckError(cudaGetLastError());

  // Prune assigned neighbors from all neighborlists and update counts
  constexpr int kWarpsPerBlock  = 4;
  constexpr int kPruneBlockSize = kWarpsPerBlock * 32;
  const int     numBlocksPrune  = (numPoints + kWarpsPerBlock - 1) / kWarpsPerBlock;
  pruneNeighborlistKernel<NeighborlistMaxSize>
    <<<numBlocksPrune, kPruneBlockSize, 0, stream>>>(clusters, clusterSizesSpan, neighborList);
  cudaCheckError(cudaGetLastError());

  // Compute argmax for next iteration
  argMaxRunner.captureOn(stream,
                         clusterSizesSpan.data(),
                         maxValuePtr,
                         maxIndexPtr,
                         static_cast<int>(clusterSizesSpan.size()));
}

//! CUDA Graph wrapper for the inner Butina loop using conditional WHILE node.
//! The GPU decides when to exit the loop - no CPU synchronization needed per iteration.
class ButinaInnerLoopGraph {
 public:
  ButinaInnerLoopGraph(int                                  numPoints,
                       const cuda::std::span<const uint8_t> hitMatrix,
                       const cuda::std::span<int>           clusters,
                       const cuda::std::span<int>           clusterSizesSpan,
                       const cuda::std::span<int>           centroids,
                       int*                                 maxIndexPtr,
                       int*                                 maxValuePtr,
                       int*                                 clusterIdxPtr,
                       int                                  threshold,
                       ArgMaxRunner&                        argMaxRunner) {
    // Create the parent graph
    cudaCheckError(cudaGraphCreate(&graph_, 0));

    // Create conditional handle with default value = 1 (enter loop at least once, do-while semantics)
    cudaCheckError(cudaGraphConditionalHandleCreate(&handle_, graph_, 1, cudaGraphCondAssignDefault));

    // Create the conditional WHILE node
    cudaGraphNodeParams cParams = {};
    cParams.type                = cudaGraphNodeTypeConditional;
    cParams.conditional.handle  = handle_;
    cParams.conditional.type    = cudaGraphCondTypeWhile;
    cParams.conditional.size    = 1;
    cudaGraphNode_t conditionalNode;
#if CUDART_VERSION >= 13000
    cudaCheckError(cudaGraphAddNode(&conditionalNode, graph_, nullptr, nullptr, 0, &cParams));
#else
    cudaCheckError(cudaGraphAddNode(&conditionalNode, graph_, nullptr, 0, &cParams));
#endif

    // Get the body graph to populate
    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    // Use stream capture to populate the body graph (easier than explicit API)
    cudaStream_t captureStream;
    cudaCheckError(cudaStreamCreate(&captureStream));

    cudaCheckError(
      cudaStreamBeginCaptureToGraph(captureStream, bodyGraph, nullptr, nullptr, 0, cudaStreamCaptureModeRelaxed));

    // Capture the inner loop kernel sequence
    innerButinaLoop(numPoints,
                    hitMatrix,
                    clusters,
                    clusterSizesSpan,
                    centroids,
                    maxIndexPtr,
                    maxValuePtr,
                    clusterIdxPtr,
                    argMaxRunner,
                    captureStream);

    // Check condition for next iteration (sets handle)
    checkLoopConditionKernel<<<1, 1, 0, captureStream>>>(handle_, maxValuePtr, threshold);

    cudaCheckError(cudaStreamEndCapture(captureStream, nullptr));
    cudaCheckError(cudaStreamDestroy(captureStream));

    // Instantiate the graph
    cudaCheckError(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
  }

  ~ButinaInnerLoopGraph() {
    if (graphExec_) {
      cudaGraphExecDestroy(graphExec_);
    }
    if (graph_) {
      cudaGraphDestroy(graph_);
    }
  }

  //! Launch the graph - GPU executes all iterations until condition fails
  void launch(cudaStream_t stream) { cudaCheckError(cudaGraphLaunch(graphExec_, stream)); }

 private:
  cudaGraph_t                graph_     = nullptr;
  cudaGraphExec_t            graphExec_ = nullptr;
  cudaGraphConditionalHandle handle_    = {};
};

//! CUDA Graph wrapper for the pruning loop using conditional WHILE node.
//! This handles small clusters with neighborlist-based assignment and pruning.
template <int NeighborlistMaxSize> class ButinaPruningLoopGraph {
 public:
  ButinaPruningLoopGraph(int                        numPoints,
                         const cuda::std::span<int> clusters,
                         const cuda::std::span<int> clusterSizesSpan,
                         const cuda::std::span<int> neighborListSpan,
                         const cuda::std::span<int> centroids,
                         int*                       maxIndexPtr,
                         int*                       maxValuePtr,
                         int*                       clusterIdxPtr,
                         ArgMaxRunner&              argMaxRunner) {
    // Create the parent graph
    cudaCheckError(cudaGraphCreate(&graph_, 0));

    // Create conditional handle with default value = 1 (enter loop at least once, do-while semantics)
    cudaCheckError(cudaGraphConditionalHandleCreate(&handle_, graph_, 1, cudaGraphCondAssignDefault));

    // Create the conditional WHILE node
    cudaGraphNodeParams cParams = {};
    cParams.type                = cudaGraphNodeTypeConditional;
    cParams.conditional.handle  = handle_;
    cParams.conditional.type    = cudaGraphCondTypeWhile;
    cParams.conditional.size    = 1;
    cudaGraphNode_t conditionalNode;
#if CUDART_VERSION >= 13000
    cudaCheckError(cudaGraphAddNode(&conditionalNode, graph_, nullptr, nullptr, 0, &cParams));
#else
    cudaCheckError(cudaGraphAddNode(&conditionalNode, graph_, nullptr, 0, &cParams));
#endif

    // Get the body graph to populate
    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    // Use stream capture to populate the body graph
    cudaStream_t captureStream;
    cudaCheckError(cudaStreamCreate(&captureStream));

    cudaCheckError(
      cudaStreamBeginCaptureToGraph(captureStream, bodyGraph, nullptr, nullptr, 0, cudaStreamCaptureModeRelaxed));

    // Capture the pruning loop kernel sequence
    innerButinaLoopWithPruning<NeighborlistMaxSize>(numPoints,
                                                    clusters,
                                                    clusterSizesSpan,
                                                    centroids,
                                                    maxIndexPtr,
                                                    maxValuePtr,
                                                    clusterIdxPtr,
                                                    neighborListSpan,
                                                    argMaxRunner,
                                                    captureStream);

    // Check condition for next iteration (sets handle)
    checkLoopConditionKernel<<<1, 1, 0, captureStream>>>(handle_, maxValuePtr, kMinLoopSizeForAssignment);

    cudaCheckError(cudaStreamEndCapture(captureStream, nullptr));
    cudaCheckError(cudaStreamDestroy(captureStream));

    // Instantiate the graph
    cudaCheckError(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
  }

  ~ButinaPruningLoopGraph() {
    if (graphExec_) {
      cudaGraphExecDestroy(graphExec_);
    }
    if (graph_) {
      cudaGraphDestroy(graph_);
    }
  }

  //! Launch the graph - GPU executes all iterations until condition fails
  void launch(cudaStream_t stream) { cudaCheckError(cudaGraphLaunch(graphExec_, stream)); }

 private:
  cudaGraph_t                graph_     = nullptr;
  cudaGraphExec_t            graphExec_ = nullptr;
  cudaGraphConditionalHandle handle_    = {};
};

/**
 * @brief Build the initial neighborlist and cluster sizes from the hit matrix.
 *
 * This is called once before entering the pruning loop.
 */
template <int NeighborlistMaxSize>
void buildInitialNeighborlist(const int                            numPoints,
                              const cuda::std::span<const uint8_t> hitMatrix,
                              const cuda::std::span<int>           clusters,
                              const cuda::std::span<int>           clusterSizesSpan,
                              const cuda::std::span<int>           neighborList,
                              cudaStream_t                         stream) {
  const ScopedNvtxRange range("Build initial neighborlist");
  butinaKernelCountClusterSizeWithNeighborlist<NeighborlistMaxSize>
    <<<numPoints, blockSizeCount, 0, stream>>>(hitMatrix, clusters, clusterSizesSpan, neighborList);
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaStreamSynchronize(stream));
}

//! CUDA Graph wrapper for the reordering=false loop.
class FixedOrderButinaGraph {
 public:
  FixedOrderButinaGraph(const cuda::std::span<const uint8_t>  hitMatrix,
                        const cuda::std::span<const uint64_t> sortedKeys,
                        const cuda::std::span<const int>      hitCounts,
                        const cuda::std::span<int>            clusters,
                        const cuda::std::span<int>            centroids,
                        int*                                  cursorPtr,
                        int*                                  activePointPtr,
                        int*                                  activeClusterPtr,
                        int*                                  clusterIdxPtr,
                        int*                                  keepGoingPtr) {
    const int numPoints     = clusters.size();
    const int rowScanBlocks = (numPoints + blockSizeCount - 1) / blockSizeCount;

    cudaCheckError(cudaGraphCreate(&graph_, 0));
    cudaCheckError(cudaGraphConditionalHandleCreate(&handle_, graph_, 1, cudaGraphCondAssignDefault));

    cudaGraphNodeParams cParams = {};
    cParams.type                = cudaGraphNodeTypeConditional;
    cParams.conditional.handle  = handle_;
    cParams.conditional.type    = cudaGraphCondTypeWhile;
    cParams.conditional.size    = 1;
    cudaGraphNode_t conditionalNode;
#if CUDART_VERSION >= 13000
    cudaCheckError(cudaGraphAddNode(&conditionalNode, graph_, nullptr, nullptr, 0, &cParams));
#else
    cudaCheckError(cudaGraphAddNode(&conditionalNode, graph_, nullptr, 0, &cParams));
#endif

    cudaGraph_t  bodyGraph = cParams.conditional.phGraph_out[0];
    cudaStream_t captureStream;
    cudaCheckError(cudaStreamCreate(&captureStream));
    cudaCheckError(
      cudaStreamBeginCaptureToGraph(captureStream, bodyGraph, nullptr, nullptr, 0, cudaStreamCaptureModeRelaxed));

    prepareFixedOrderCandidateKernel<<<1, fixedOrderPrepareBlockSize, 0, captureStream>>>(sortedKeys,
                                                                                          hitCounts,
                                                                                          clusters,
                                                                                          centroids,
                                                                                          cursorPtr,
                                                                                          activePointPtr,
                                                                                          activeClusterPtr,
                                                                                          clusterIdxPtr,
                                                                                          keepGoingPtr);
    cudaCheckError(cudaGetLastError());

    assignFixedOrderActiveRowKernel<<<rowScanBlocks, blockSizeCount, 0, captureStream>>>(hitMatrix,
                                                                                         hitCounts,
                                                                                         clusters,
                                                                                         centroids,
                                                                                         activePointPtr,
                                                                                         activeClusterPtr);
    cudaCheckError(cudaGetLastError());

    checkFixedOrderGraphConditionKernel<<<1, 1, 0, captureStream>>>(handle_, keepGoingPtr);
    cudaCheckError(cudaGetLastError());

    cudaCheckError(cudaStreamEndCapture(captureStream, nullptr));
    cudaCheckError(cudaStreamDestroy(captureStream));
    cudaCheckError(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
  }

  ~FixedOrderButinaGraph() {
    if (graphExec_) {
      cudaGraphExecDestroy(graphExec_);
    }
    if (graph_) {
      cudaGraphDestroy(graph_);
    }
  }

  void launch(cudaStream_t stream) { cudaCheckError(cudaGraphLaunch(graphExec_, stream)); }

 private:
  cudaGraph_t                graph_     = nullptr;
  cudaGraphExec_t            graphExec_ = nullptr;
  cudaGraphConditionalHandle handle_    = {};
};

//! Run fixed-order Butina clustering for reordering=false.
int butinaGpuNoReorderingImpl(const cuda::std::span<const uint8_t> hitMatrix,
                              const cuda::std::span<int>           clusters,
                              const cuda::std::span<int>           centroids,
                              const cuda::std::span<int>           initialHitCounts,
                              cudaStream_t                         stream) {
  ScopedNvtxRange setupRange("Butina No-Reordering Setup");

  // validate input
  const int numPoints = clusters.size();
  validateButinaInputs(hitMatrix, clusters, centroids);
  if (numPoints == 0) {
    return 0;
  }

  // sort the points
  AsyncDeviceVector<uint64_t> sortKeys(numPoints, stream);
  AsyncDeviceVector<uint64_t> sortedKeys(numPoints, stream);
  sortFixedOrderCandidates(initialHitCounts, toSpan(sortKeys), toSpan(sortedKeys), stream);

  const AsyncDevicePtr<int> clusterIdx(0, stream);
  setupRange.pop();

  const AsyncDevicePtr<int> cursor(0, stream);
  const AsyncDevicePtr<int> activePoint(-1, stream);
  const AsyncDevicePtr<int> activeCluster(-1, stream);
  const AsyncDevicePtr<int> keepGoing(1, stream);

  ScopedNvtxRange       buildRange("Build no-reordering Butina graph");
  FixedOrderButinaGraph graph(hitMatrix,
                              toSpan(sortedKeys),
                              initialHitCounts,
                              clusters,
                              centroids,
                              cursor.data(),
                              activePoint.data(),
                              activeCluster.data(),
                              clusterIdx.data(),
                              keepGoing.data());
  buildRange.pop();

  const ScopedNvtxRange loopRange("No-reordering Butina graph loop");
  graph.launch(stream);

  // we pin this host memory since it allows us to more efficiently copy data from the device back to host again
  PinnedHostVector<int> numClusters(1);
  cudaCheckError(cudaMemcpyAsync(numClusters.data(), clusterIdx.data(), sizeof(int), cudaMemcpyDefault, stream));

  // wait for the memcpy to finish
  cudaCheckError(cudaStreamSynchronize(stream));

  return numClusters[0];
}

template <int NeighborlistMaxSize>
[[maybe_unused]] int butinaGpuImpl(const cuda::std::span<const uint8_t> hitMatrix,
                                   const cuda::std::span<int>           clusters,
                                   const cuda::std::span<int>           centroids,
                                   cudaStream_t                         stream) {
  ScopedNvtxRange setupRange("Butina Setup");
  const size_t    numPoints = clusters.size();
  validateButinaInputs(hitMatrix, clusters, centroids);
  setAll(clusters, -1, stream);
  AsyncDeviceVector<int> clusterSizes(clusters.size(), stream);
  clusterSizes.zero();
  AsyncDeviceVector<int> neighborList(NeighborlistMaxSize * numPoints, stream);
  const auto             neighborListSpan = toSpan(neighborList);

  const AsyncDevicePtr<int> maxIndex(-1, stream);
  const AsyncDevicePtr<int> maxValue(std::numeric_limits<int>::max(), stream);
  const AsyncDevicePtr<int> clusterIdx(0, stream);
  PinnedHostVector<int>     maxCluster(1);
  maxCluster[0] = std::numeric_limits<int>::max();

  ArgMaxRunner argMaxRunner(clusters.size(), stream);

  setupRange.pop();
  const auto clusterSizesSpan = toSpan(clusterSizes);

  // If a neighborlist is up to N, then the cluster is up to N+1 (including the central point).
  constexpr int clusterSizeWithMaxNeighborlist = NeighborlistMaxSize + 1;

  // Use CUDA Graph with conditional WHILE node for fully GPU-side loop control.
  // The GPU decides when to exit - no CPU synchronization needed per iteration.
  {
    ScopedNvtxRange      buildRange("Build inner loop graph with WHILE node");
    ButinaInnerLoopGraph innerLoopGraph(static_cast<int>(numPoints),
                                        hitMatrix,
                                        clusters,
                                        clusterSizesSpan,
                                        centroids,
                                        maxIndex.data(),
                                        maxValue.data(),
                                        clusterIdx.data(),
                                        clusterSizeWithMaxNeighborlist,
                                        argMaxRunner);
    buildRange.pop();

    // Launch once - GPU executes all iterations until maxValue < threshold
    const ScopedNvtxRange loopRange("Large cluster Butina Loop (conditional WHILE graph)");
    innerLoopGraph.launch(stream);
    cudaCheckError(cudaStreamSynchronize(stream));

    // Copy final maxValue to host for subsequent pruning loop
    cudaCheckError(cudaMemcpyAsync(maxCluster.data(), maxValue.data(), sizeof(int), cudaMemcpyDefault, stream));
    cudaCheckError(cudaStreamSynchronize(stream));
  }

  // Build neighborlist once, then prune dynamically using CUDA Graph with conditional WHILE node
  if (maxCluster[0] >= kMinLoopSizeForAssignment) {
    buildInitialNeighborlist<NeighborlistMaxSize>(numPoints,
                                                  hitMatrix,
                                                  clusters,
                                                  clusterSizesSpan,
                                                  neighborListSpan,
                                                  stream);

    // Initial argmax to prime the loop (buildInitialNeighborlist already synced)
    argMaxRunner(clusterSizesSpan.data(), maxValue.data(), maxIndex.data(), static_cast<int>(clusterSizesSpan.size()));
    cudaCheckError(cudaStreamSynchronize(stream));

    // Use CUDA Graph with conditional WHILE node for fully GPU-side pruning loop control
    ScopedNvtxRange                             buildRange("Build pruning loop graph with WHILE node");
    ButinaPruningLoopGraph<NeighborlistMaxSize> pruningLoopGraph(numPoints,
                                                                 clusters,
                                                                 clusterSizesSpan,
                                                                 neighborListSpan,
                                                                 centroids,
                                                                 maxIndex.data(),
                                                                 maxValue.data(),
                                                                 clusterIdx.data(),
                                                                 argMaxRunner);
    buildRange.pop();

    // Launch once - GPU executes all iterations until maxValue < kMinLoopSizeForAssignment
    const ScopedNvtxRange loopRange("Small cluster Butina Loop with pruning (conditional WHILE graph)");
    pruningLoopGraph.launch(stream);
    cudaCheckError(cudaStreamSynchronize(stream));
  }

  assignSingletonIdsKernel<<<1, kSingletonBlockSize, 0, stream>>>(clusters, centroids, clusterIdx.data());
  cudaCheckError(cudaGetLastError());

  // Renumber clusters to be in descending order.
  cudaCheckError(cudaMemcpyAsync(maxCluster.data(), clusterIdx.data(), sizeof(int), cudaMemcpyDefault, stream));
  cudaCheckError(cudaStreamSynchronize(stream));
  renumberClustersBySize(clusters, centroids, maxCluster[0], stream);
  cudaCheckError(cudaStreamSynchronize(stream));
  return maxCluster[0];
}

[[maybe_unused]] int butinaGpu(const cuda::std::span<const uint8_t> hitMatrix,
                               const cuda::std::span<int>           clusters,
                               const int                            neighborlistMaxSize,
                               const cuda::std::span<int>           centroids,
                               const bool                           reordering,
                               cudaStream_t                         stream) {
  if (!reordering) {
    AsyncDeviceVector<int> initialHitCounts(clusters.size(), stream);
    if (!clusters.empty()) {
      setAll(clusters, -1, stream);
      butinaKernelCountClusterSize<<<clusters.size(), blockSizeCount, 0, stream>>>(hitMatrix,
                                                                                   clusters,
                                                                                   toSpan(initialHitCounts));
      cudaCheckError(cudaGetLastError());
    }
    return butinaGpuNoReorderingImpl(hitMatrix, clusters, centroids, toSpan(initialHitCounts), stream);
  }

  switch (neighborlistMaxSize) {
    case 8:
      return butinaGpuImpl<8>(hitMatrix, clusters, centroids, stream);
    case 16:
      return butinaGpuImpl<16>(hitMatrix, clusters, centroids, stream);
    case 24:
      return butinaGpuImpl<24>(hitMatrix, clusters, centroids, stream);
    case 32:
      return butinaGpuImpl<32>(hitMatrix, clusters, centroids, stream);
    case 64:
      return butinaGpuImpl<64>(hitMatrix, clusters, centroids, stream);
    case 128:
      return butinaGpuImpl<128>(hitMatrix, clusters, centroids, stream);
    default:
      throw std::invalid_argument("neighborlistMaxSize must be 8, 16, 24, 32, 64, or 128. Got: " +
                                  std::to_string(neighborlistMaxSize));
  }
}

namespace {

__global__ void thresholdDistanceMatrixKernel(const double* __restrict__ matrix,
                                              uint8_t* __restrict__ hits,
                                              const double cutoff,
                                              const size_t numElements) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    hits[idx] = (matrix[idx] <= cutoff);
  }
}

// Build the uint8_t hit matrix and fixed-order counts in one pass.
__global__ void thresholdDistanceMatrixAndCountKernel(const cuda::std::span<const double> matrix,
                                                      const cuda::std::span<uint8_t>      hits,
                                                      const cuda::std::span<int>          hitCounts,
                                                      const double                        cutoff) {
  const int tid        = threadIdx.x;
  const int pointIdx   = blockIdx.x;
  const int numPoints  = hitCounts.size();
  int       localCount = 0;
  for (int i = tid; i < numPoints; i += blockSizeCount) {
    const size_t idx = static_cast<size_t>(pointIdx) * numPoints + i;
    const bool   hit = matrix[idx] <= cutoff;
    hits[idx]        = hit;
    if (hit) {
      localCount++;
    }
  }
  sumCountsAndStoreClusterSize(tid, pointIdx, hitCounts, localCount);
}

}  // namespace

[[maybe_unused]] int butinaGpu(const cuda::std::span<const double> distanceMatrix,
                               const cuda::std::span<int>          clusters,
                               const double                        cutoff,
                               const int                           neighborlistMaxSize,
                               const cuda::std::span<int>          centroids,
                               const bool                          reordering,
                               cudaStream_t                        stream) {
  AsyncDeviceVector<uint8_t> hitMatrix(distanceMatrix.size(), stream);

  if (reordering) {
    constexpr int blockSize = 256;
    const size_t  numBlocks = (distanceMatrix.size() + blockSize - 1) / blockSize;
    thresholdDistanceMatrixKernel<<<numBlocks, blockSize, 0, stream>>>(distanceMatrix.data(),
                                                                       hitMatrix.data(),
                                                                       cutoff,
                                                                       distanceMatrix.size());
    cudaCheckError(cudaGetLastError());
    return butinaGpu(toSpan(hitMatrix), clusters, neighborlistMaxSize, centroids, reordering, stream);
  }

  AsyncDeviceVector<int> initialHitCounts(clusters.size(), stream);
  if (!clusters.empty()) {
    setAll(clusters, -1, stream);
    thresholdDistanceMatrixAndCountKernel<<<clusters.size(), blockSizeCount, 0, stream>>>(distanceMatrix,
                                                                                          toSpan(hitMatrix),
                                                                                          toSpan(initialHitCounts),
                                                                                          cutoff);
    cudaCheckError(cudaGetLastError());
  }
  return butinaGpuNoReorderingImpl(toSpan(hitMatrix), clusters, centroids, toSpan(initialHitCounts), stream);
}

}  // namespace nvMolKit
