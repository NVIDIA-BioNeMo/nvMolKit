// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_AAP_H
#define NVMOLKIT_AAP_H

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace RDKit {
class ROMol;
}

namespace nvMolKit {

/** Configuration for approximate GPU Atom-Atom Path (AAP) similarity. */
struct AapOptions {
  int   maxPathLength       = 7;
  int   histogramBins       = 2048;
  int   sinkhornIterations  = 8;
  float sinkhornTemperature = 0.104F;
};

/** Clustering assignments and metadata, ordered by descending cluster size. */
struct AapClusteringResult {
  std::vector<int>          clusterIds;
  std::vector<int>          centroids;
  std::vector<std::int64_t> clusterSizes;
};

/**
 * Compute directed approximate Atom-Atom Path (AAP) similarity on the GPU.
 *
 * Rooted paths are hashed into per-atom histograms and compatible atoms are
 * assigned with fixed-iteration Sinkhorn normalization. The current fused
 * implementation supports molecules containing at most 64 atoms.
 */
float aapSimilarityGpu(const RDKit::ROMol& left,
                       const RDKit::ROMol& right,
                       const AapOptions&   options = {},
                       cudaStream_t        stream  = nullptr);

/**
 * Cluster molecules with input-order directed sphere exclusion (DISE).
 *
 * The first unassigned molecule is selected as the next centroid and claims
 * all remaining molecules whose directed Atom-Atom Path (AAP) similarity
 * meets @p threshold.
 * Cluster IDs are zero-based and renumbered by descending cluster size, with
 * centroid order breaking ties. Centroids and sizes use the same cluster-ID
 * order.
 */
AapClusteringResult aapSimilarityClustering(const std::vector<const RDKit::ROMol*>& molecules,
                                            float                                   threshold = 0.217F,
                                            const AapOptions&                       options   = {},
                                            cudaStream_t                            stream    = nullptr);

/** Run full two-stage directed sphere exclusion (DISE) with nearest-centroid assignment. */
AapClusteringResult aapDiseClustering(const std::vector<const RDKit::ROMol*>& molecules,
                                      float                                   threshold = 0.217F,
                                      const AapOptions&                       options   = {},
                                      cudaStream_t                            stream    = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_AAP_H
