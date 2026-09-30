// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_AAP_H
#define NVMOLKIT_AAP_H

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/clustering_result.h"
#include "src/diversity_pickers.h"

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

/** Molecules that AAP cannot process, grouped by reason as input indices. */
class AapInvalidMoleculesError : public std::invalid_argument {
 public:
  AapInvalidMoleculesError(const std::string& message,
                           std::vector<int>   none,
                           std::vector<int>   empty,
                           std::vector<int>   tooManyAtoms,
                           std::vector<int>   unsupportedBonds)
      : std::invalid_argument(message),
        none(std::move(none)),
        empty(std::move(empty)),
        tooManyAtoms(std::move(tooManyAtoms)),
        unsupportedBonds(std::move(unsupportedBonds)) {}

  std::vector<int> none;
  std::vector<int> empty;
  std::vector<int> tooManyAtoms;
  std::vector<int> unsupportedBonds;
};

/**
 * Compute approximate Atom-Atom Path (AAP) similarity from @p left to @p right on the GPU.
 *
 * Rooted paths are hashed into per-atom histograms and compatible atoms are
 * assigned with fixed-iteration Sinkhorn normalization. Molecules may contain
 * at most 64 atoms. Invalid molecules raise AapInvalidMoleculesError.
 */
float aapSimilarityGpu(const RDKit::ROMol& left,
                       const RDKit::ROMol& right,
                       const AapOptions&   options = {},
                       cudaStream_t        stream  = nullptr);

// AAP distance is 1 - aapSimilarityGpu(selected, candidate). Leader, MaxMin, and DISE follow the conventions of the
// matching functions in src/diversity_pickers.h.

PickerResult aapLeader(const std::vector<const RDKit::ROMol*>& molecules,
                       double                                  cutoff,
                       const AapOptions&                       options    = {},
                       int                                     pickSize   = 0,
                       const std::vector<int>&                 firstPicks = {},
                       cudaStream_t                            stream     = nullptr);

PickerResult aapMaxMin(const std::vector<const RDKit::ROMol*>& molecules,
                       int                                     pickSize,
                       const AapOptions&                       options    = {},
                       const std::vector<int>&                 firstPicks = {},
                       int                                     seed       = -1,
                       double                                  threshold  = -1.0,
                       cudaStream_t                            stream     = nullptr);

ClusteringResult aapDise(const std::vector<const RDKit::ROMol*>& molecules,
                         double                                  cutoff,
                         const AapOptions&                       options,
                         bool                                    nearestAssignment,
                         cudaStream_t                            stream = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_AAP_H
