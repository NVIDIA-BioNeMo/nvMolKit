// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_KMEANS_H
#define NVMOLKIT_KMEANS_H

#include <cuda_runtime.h>

#include <cstdint>
#include <cuda/std/span>

#include "src/utils/device_vector.h"

namespace nvMolKit {

/**
 * Metric used by k-means on packed bit fingerprints. Each row x is a {0,1} vector of numWords * 32 bits.
 *
 * - Cosine: squared Euclidean distance between L2-normalized rows and centers that are means of normalized rows (the
 *   same problem cuML solves after normalizing rows). Rows without set bits are the zero vector.
 * - Tanimoto: continuous Tanimoto similarity x.c / (|x| + |c|^2 - x.c), maximized; distance is 1 - similarity and
 *   centers are bit means.
 */
enum class KMeansMetric {
  Cosine,
  Tanimoto
};

//! k-means++ seeding is batched on top of kmeansAssignGpu in the Python layer.
enum class KMeansInit {
  Random,  //!< numClusters distinct rows chosen uniformly.
  MaxMin,  //!< Fused MaxMin picks (Tanimoto distance, or cosine distance for the cosine metric).
};

struct KMeansOptions {
  int           numClusters = 8;
  KMeansMetric  metric      = KMeansMetric::Tanimoto;
  int           maxIter     = 300;
  //! Stop when the relative decrease of the objective, (previous - current) / current, is at most tol, or when no
  //! label changes. tol == 0 runs exactly maxIter iterations.
  double        tol         = 1e-4;
  //! Seed for empty-cluster reseeding ties and the initialization.
  std::uint64_t seed        = 0;
};

struct KMeansFitResult {
  AsyncDeviceVector<int>          labels;
  //! Row-major float32 centers, numClusters x (numWords * 32), in the metric's space.
  AsyncDeviceVector<float>        centers;
  AsyncDeviceVector<std::int64_t> clusterSizes;
  //! Objective of the final labels: summed squared Euclidean distance on normalized bits (cosine), or summed Tanimoto
  //! distance.
  double                          inertia       = 0.0;
  //! Number of center updates.
  int                             numIterations = 0;
  //! Clusters reseeded because they lost every member, summed over all iterations.
  std::int64_t                    reseeded      = 0;
};

/**
 * @brief Choose numClusters initial centers from the rows of a packed fingerprint matrix.
 *
 * Returns row indices; centers are the (normalized, for cosine) rows. Deterministic for a given seed.
 */
AsyncDeviceVector<int> kmeansInitGpu(cuda::std::span<const std::uint32_t> fingerprints,
                                     int                                  numFingerprints,
                                     int                                  numWords,
                                     int                                  numClusters,
                                     KMeansMetric                         metric,
                                     KMeansInit                           init,
                                     std::uint64_t                        seed,
                                     cudaStream_t                         stream = nullptr);

//! Builds float32 centers from fingerprint rows, normalized for the cosine metric.
AsyncDeviceVector<float> kmeansCentersFromRows(cuda::std::span<const std::uint32_t> fingerprints,
                                               int                                  numWords,
                                               const AsyncDeviceVector<int>&        rows,
                                               KMeansMetric                         metric,
                                               cudaStream_t                         stream = nullptr);

/**
 * @brief Lloyd iterations from the given initial centers (numClusters x numWords * 32 float32, device).
 *
 * The packed matrix is never unpacked. Memory beyond the input is O(numClusters * numWords * 32) for the centers and
 * their sums plus O(numFingerprints) for labels and distances.
 */
KMeansFitResult kmeansFitGpu(cuda::std::span<const std::uint32_t> fingerprints,
                             int                                  numFingerprints,
                             int                                  numWords,
                             const float*                         initialCenters,
                             const KMeansOptions&                 options,
                             cudaStream_t                         stream = nullptr);

//! As above, taking ownership of the initial centers buffer (it becomes the fitted centers; no copy).
KMeansFitResult kmeansFitGpu(cuda::std::span<const std::uint32_t> fingerprints,
                             int                                  numFingerprints,
                             int                                  numWords,
                             AsyncDeviceVector<float>             initialCenters,
                             const KMeansOptions&                 options,
                             cudaStream_t                         stream = nullptr);

/**
 * @brief Assigns each fingerprint to its best center.
 *
 * Writes each row's best center and its distance (squared Euclidean on normalized bits, or 1 - Tanimoto). Ties go to
 * the lowest center.
 */
void kmeansAssignGpu(cuda::std::span<const std::uint32_t> fingerprints,
                     int                                  numFingerprints,
                     int                                  numWords,
                     const float*                         centers,
                     int                                  numClusters,
                     KMeansMetric                         metric,
                     int*                                 labels,
                     float*                               distances,
                     cudaStream_t                         stream = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_KMEANS_H
