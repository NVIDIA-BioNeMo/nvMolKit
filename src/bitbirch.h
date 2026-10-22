// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This implementation is independently derived from the published algorithm
// and equations in BitBIRCH (https://doi.org/10.1039/D5DD00030K), BitBIRCH
// refinement strategies (https://doi.org/10.1021/acs.jcim.5c00627), and
// BitBIRCH-Lean (https://doi.org/10.1101/2025.10.22.684015). No source from the
// GPL-licensed bblean or bitbirch packages is used.

#ifndef NVMOLKIT_BITBIRCH_H
#define NVMOLKIT_BITBIRCH_H

#include <cuda_runtime.h>

#include <cstdint>
#include <cuda/std/span>

#include "src/utils/device_vector.h"

namespace nvMolKit {

struct BitBirchResult {
  AsyncDeviceVector<int>           clusterIds;
  AsyncDeviceVector<std::uint32_t> centroids;
  int                              numClusters = 0;
  int                              numWords    = 0;
};

struct BitBirchOptions {
  int  branchingFactor = 1024;   //!< Maximum entries per tree node; at least 3.
  int  batchSize       = 1024;   //!< Fingerprints routed per insertion epoch.
  bool returnCentroids = false;  //!< Also return packed majority centroids in cluster-ID order.
};

//! Most fingerprint words bitBirchGpu accepts (4096 bits).
constexpr int kBitBirchMaxWords = 128;

/**
 * Cluster packed device fingerprints with one BitBIRCH tree.
 *
 * Each insertion epoch routes a batch against a stable snapshot of the tree,
 * then one ordered owner per leaf merges or appends its queries, summaries are
 * refreshed bottom-up, and overflowing nodes split at a barrier. Cluster IDs
 * are ordered by each cluster's earliest input row.
 */
BitBirchResult bitBirchGpu(cuda::std::span<const std::uint32_t> fingerprints,
                           int                                  numFingerprints,
                           int                                  numWords,
                           double                               threshold,
                           const BitBirchOptions&               options = BitBirchOptions(),
                           cudaStream_t                         stream  = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_BITBIRCH_H
