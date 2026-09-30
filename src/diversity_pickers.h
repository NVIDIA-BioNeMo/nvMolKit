// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DIVERSITY_PICKERS_H
#define NVMOLKIT_DIVERSITY_PICKERS_H

#include <cuda_runtime.h>

#include <cstdint>
#include <cuda/std/span>
#include <vector>

#include "src/clustering_result.h"
#include "src/fingerprint_similarity.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

/**
 * Ordered picks. @c lastDistance is the nearest-pick distance of the last MaxMin addition, or -1 when none was added
 * after the initial picks.
 */
struct PickerResult {
  AsyncDeviceVector<int> indices;
  float                  lastDistance = -1.0F;
};

/**
 * @brief Common input conventions for distance-based pickers.
 *
 * Distance matrices are square and row-major; element [i, j] is the distance from selected item i to candidate j.
 * Fingerprints are packed row-major with shape (num items, num words); distance is 1 - similarity.
 * Matrix values, cutoffs, and thresholds are converted to single precision for comparisons. Matrix cutoffs and
 * enabled thresholds must be finite, non-negative, and no larger than the maximum finite float32 value.
 * pickSize == 0 means no limit for Leader; all firstPicks are retained in order even when they exceed pickSize.
 * A negative MaxMin seed seeds from system entropy, and threshold == -1 disables the MaxMin early stop.
 */

PickerResult leaderFromDistanceMatrix(cuda::std::span<const float> distanceMatrix,
                                      int                          numItems,
                                      double                       cutoff,
                                      int                          pickSize,
                                      const std::vector<int>&      firstPicks = {},
                                      cudaStream_t                 stream     = nullptr);

PickerResult leaderFromDistanceMatrix(cuda::std::span<const double> distanceMatrix,
                                      int                           numItems,
                                      double                        cutoff,
                                      int                           pickSize,
                                      const std::vector<int>&       firstPicks = {},
                                      cudaStream_t                  stream     = nullptr);

PickerResult fusedLeaderGpu(cuda::std::span<const std::uint32_t> fingerprints,
                            int                                  numFingerprints,
                            int                                  numWords,
                            double                               cutoff,
                            FingerprintSimilarityMetric          metric,
                            int                                  pickSize,
                            const std::vector<int>&              firstPicks = {},
                            cudaStream_t                         stream     = nullptr);

PickerResult maxMinFromDistanceMatrix(cuda::std::span<const float> distanceMatrix,
                                      int                          numItems,
                                      int                          pickSize,
                                      const std::vector<int>&      firstPicks = {},
                                      int                          seed       = -1,
                                      double                       threshold  = -1.0,
                                      cudaStream_t                 stream     = nullptr);

PickerResult maxMinFromDistanceMatrix(cuda::std::span<const double> distanceMatrix,
                                      int                           numItems,
                                      int                           pickSize,
                                      const std::vector<int>&       firstPicks = {},
                                      int                           seed       = -1,
                                      double                        threshold  = -1.0,
                                      cudaStream_t                  stream     = nullptr);

PickerResult fusedMaxMinGpu(cuda::std::span<const std::uint32_t> fingerprints,
                            int                                  numFingerprints,
                            int                                  numWords,
                            int                                  pickSize,
                            FingerprintSimilarityMetric          metric,
                            const std::vector<int>&              firstPicks = {},
                            int                                  seed       = -1,
                            double                               threshold  = -1.0,
                            cudaStream_t                         stream     = nullptr);

ClusteringResult diseFromDistanceMatrix(cuda::std::span<const float> distanceMatrix,
                                        int                          numItems,
                                        double                       cutoff,
                                        bool                         nearestAssignment,
                                        cudaStream_t                 stream = nullptr);

ClusteringResult diseFromDistanceMatrix(cuda::std::span<const double> distanceMatrix,
                                        int                           numItems,
                                        double                        cutoff,
                                        bool                          nearestAssignment,
                                        cudaStream_t                  stream = nullptr);

ClusteringResult fusedDiseGpu(cuda::std::span<const std::uint32_t> fingerprints,
                              int                                  numFingerprints,
                              int                                  numWords,
                              double                               cutoff,
                              FingerprintSimilarityMetric          metric,
                              bool                                 nearestAssignment,
                              cudaStream_t                         stream = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_DIVERSITY_PICKERS_H
