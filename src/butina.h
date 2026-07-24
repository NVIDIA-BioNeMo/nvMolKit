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

#ifndef NVMOLKIT_BUTINA_H
#define NVMOLKIT_BUTINA_H

#include <cuda_runtime.h>

#include <cstdint>
#include <cuda/std/span>

#include "src/fingerprint_similarity.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

/**
 * @brief Device-side output shared by the Butina implementations.
 *
 * - @p clusterIds contains one cluster ID per input item.
 * - @p centroids is allocated to the input size when requested; its first @p numClusters entries contain centroid
 * indices.
 * - @p numClusters contains the number of clusters represented by the output.
 */
struct ButinaResult {
  AsyncDeviceVector<int> clusterIds;
  AsyncDeviceVector<int> centroids;
  int                    numClusters = 0;
};

/**
 * @brief Perform fused Butina clustering directly on packed fingerprints.
 *
 * This implementation computes fingerprint similarities and neighbor relationships
 * on demand instead of materializing an NxN distance or hit matrix. Use it when the
 * input consists of packed fingerprints and quadratic matrix storage is undesirable.
 * Use @ref butinaFromDistanceMatrix when a distance matrix is already available,
 * or @ref butinaFromHitMatrix when neighbor relationships have already been
 * thresholded. This implementation updates active neighbor counts after every
 * cluster selection, corresponding to `reordering=true` in the matrix APIs.
 *
 * Only Tanimoto and cosine similarity are supported. The distance cutoff is converted
 * to a similarity threshold of `1 - cutoff`. The result buffers remain in device
 * memory.
 *
 * @param fingerprints Packed device array with @p numFingerprints entries of @p numWords words each.
 * @param numFingerprints Number of fingerprints in @p fingerprints.
 * @param numWords Number of 32-bit words in each fingerprint. Must be greater than zero.
 * @param cutoff Distance cutoff in [0, 1]; neighbors have similarity of at least `1 - cutoff`.
 * @param metric Fingerprint similarity metric used to determine neighbors.
 * @param returnCentroids Whether to allocate and populate centroid output.
 * @param stream CUDA stream on which to execute. Defaults to stream 0.
 * @return Device-owned cluster IDs, optional centroids, and the number of clusters.
 */
ButinaResult fusedButinaGpu(cuda::std::span<const std::uint32_t> fingerprints,
                            int                                  numFingerprints,
                            int                                  numWords,
                            double                               cutoff,
                            FingerprintSimilarityMetric          metric,
                            bool                                 returnCentroids = false,
                            cudaStream_t                         stream          = nullptr);

/**
 * @brief Perform Butina clustering on a distance matrix with automatic thresholding.
 *
 * This function converts the distance matrix into a binary hit matrix by
 * thresholding at the specified cutoff, then performs Butina clustering. The
 * algorithm iteratively selects the item with the most unclustered neighbors
 * and forms clusters. Output buffers are allocated on the device and returned
 * to the caller. The distance matrix is expected to be symmetric.
 *
 *
 *
 * @param distanceMatrix Square NxN matrix where distanceMatrix[i*N+j] contains the distance between items i and j.
 * @param numPoints Number of items represented by the distance matrix.
 * @param cutoff Distance threshold for clustering. Items with distance <= cutoff are considered neighbors.
 * @param neighborlistMaxSize Small-cluster neighbor-list capacity. Must be 8, 16, 24, 32, 64, or 128. Ignored when
 * reordering is disabled.
 * @param returnCentroids Whether to return one centroid index per cluster.
 * @param reordering Whether to dynamically reorder candidates after each cluster assignment.
 * @param stream CUDA stream on which to execute. Defaults to stream 0.
 * @return Device-owned cluster IDs, optional centroids, and the number of clusters.
 */
ButinaResult butinaFromDistanceMatrix(cuda::std::span<const double> distanceMatrix,
                                      int                           numPoints,
                                      double                        cutoff,
                                      int                           neighborlistMaxSize = 64,
                                      bool                          returnCentroids     = false,
                                      bool                          reordering          = true,
                                      cudaStream_t                  stream              = nullptr);

/**
 * @brief Perform Butina clustering on a precomputed hit matrix.
 *
 * This function accepts a binary hit matrix where element (i,j) indicates
 * whether items i and j are neighbors. It avoids allocating and thresholding a
 * floating-point distance matrix when neighbor relationships are already known.
 * Output buffers are allocated on the device and returned to the caller. The
 * matrix is expected to be symmetric with a nonzero diagonal.
 *
 * @param hitMatrix Binary NxN neighbor matrix.
 * @param numPoints Number of molecules.
 * @param neighborlistMaxSize Small-cluster neighbor-list capacity; ignored when reordering is disabled.
 * @param returnCentroids Whether to return cluster centroid indices.
 * @param reordering Whether to update neighbor counts after each cluster assignment.
 * @param stream CUDA stream; defaults to stream 0.
 * @return Device-owned cluster IDs, optional centroids, and the number of clusters.
 */
ButinaResult butinaFromHitMatrix(cuda::std::span<const uint8_t> hitMatrix,
                                 int                            numPoints,
                                 int                            neighborlistMaxSize = 64,
                                 bool                           returnCentroids     = false,
                                 bool                           reordering          = true,
                                 cudaStream_t                   stream              = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_BUTINA_H
