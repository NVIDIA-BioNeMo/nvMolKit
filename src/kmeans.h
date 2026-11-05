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
