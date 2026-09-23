// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_CLUSTERING_RESULT_H
#define NVMOLKIT_CLUSTERING_RESULT_H

#include <cstdint>
#include <vector>

namespace nvMolKit {

/** Host representation shared by clustering implementations. */
struct ClusteringResult {
  std::vector<int>          clusterIds;
  std::vector<int>          centroids;
  std::vector<std::int64_t> clusterSizes;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_CLUSTERING_RESULT_H
