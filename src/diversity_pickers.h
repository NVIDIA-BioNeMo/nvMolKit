// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DIVERSITY_PICKERS_H
#define NVMOLKIT_DIVERSITY_PICKERS_H

#include <cuda_runtime.h>

#include <cstdint>
#include <cuda/std/span>
#include <vector>

#include "src/utils/device_vector.h"

namespace nvMolKit {

/** Ordered picks. */
struct PickerResult {
  AsyncDeviceVector<int> indices;
};

// Distance matrices are square and row-major; element [i, j] is the distance from selected item i to candidate j.
// Distances are compared in single precision. pickSize == 0 means no limit for Leader.

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

}  // namespace nvMolKit

#endif  // NVMOLKIT_DIVERSITY_PICKERS_H
