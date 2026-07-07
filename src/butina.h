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

#include "src/utils/device_vector.h"

namespace nvMolKit {

/**
 * @brief Perform Butina clustering on a distance matrix with automatic thresholding.
 *
 * This is a convenience wrapper that converts the distance matrix into a binary hit
 * matrix by thresholding at the specified cutoff, then performs Butina clustering.
 * The Butina algorithm is a deterministic clustering method that iteratively selects
 * the item with the most unclustered neighbors and forms clusters.
 *
 * @param distanceMatrix Square distance matrix of size NxN where distanceMatrix[i*N+j]
 *                       contains the distance between items i and j.
 * @param clusters Output array of size N. Each element will contain the cluster ID for
 *                 that item. Modified in-place.
 * @param cutoff Distance threshold for clustering. Items with distance <= cutoff are
 *               considered neighbors.
 * @param neighborlistMaxSize Maximum size of the neighborlist used for small cluster optimization.
 *                            Must be 8, 16, 24, 32, 64, or 128. Larger values allow parallel
 *                            processing of larger clusters but use more memory.
 * @param centroids Optional output array of size N. If provided, centroids[i] stores the
 *                  centroid index for cluster i. Must be empty or size N.
 * @param reordering Whether to dynamically reorder candidates after each cluster assignment.
 * @param stream CUDA stream to execute operations on. Defaults to stream 0.
 * @return Number of clusters assigned.
 */
[[maybe_unused]] int butinaGpu(cuda::std::span<const double> distanceMatrix,
                               cuda::std::span<int>          clusters,
                               double                        cutoff,
                               int                           neighborlistMaxSize = 64,
                               cuda::std::span<int>          centroids           = {},
                               bool                          reordering          = true,
                               cudaStream_t                  stream              = nullptr);

/**
 * @brief Perform Butina clustering on a precomputed hit matrix.
 *
 * This is the core GPU implementation of the Butina clustering algorithm. It takes
 * a binary hit matrix where element (i,j) indicates whether items i and j are neighbors.
 *
 * @param hitMatrix Binary matrix of size NxN where hitMatrix[i*N+j] = 1 if items i and j
 *                  are neighbors (distance <= cutoff), 0 otherwise.
 * @param clusters Output array of size N. Each element will contain the cluster ID for
 *                 that item. Modified in-place.
 * @param neighborlistMaxSize Maximum size of the neighborlist used for small cluster optimization.
 *                            Must be 8, 16, 24, 32, 64, or 128. Larger values allow parallel
 *                            processing of larger clusters but use more memory.
 * @param centroids Optional output array of size N. If provided, centroids[i] stores the
 *                  centroid index for cluster i. Must be empty or size N.
 * @param reordering Whether to dynamically reorder candidates after each cluster assignment.
 * @param stream CUDA stream to execute operations on. Defaults to stream 0.
 * @return Number of clusters assigned.
 */
[[maybe_unused]] int butinaGpu(cuda::std::span<const uint8_t> hitMatrix,
                               cuda::std::span<int>           clusters,
                               int                            neighborlistMaxSize = 64,
                               cuda::std::span<int>           centroids           = {},
                               bool                           reordering          = true,
                               cudaStream_t                   stream              = nullptr);
}  // namespace nvMolKit

#endif  // NVMOLKIT_BUTINA_H
