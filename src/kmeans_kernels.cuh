// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_KMEANS_KERNELS_CUH
#define NVMOLKIT_KMEANS_KERNELS_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/kmeans.h"

namespace nvMolKit::detail {

//! Centers per MMA block tile and fingerprint bits per pipeline stage.
constexpr int kMmaBlockCenters = 256;
constexpr int kMmaStageBits    = 64;

struct AssignArgs {
  const std::uint32_t* fingerprints;
  const int*           rowCounts;
  int                  numRows;
  int                  numWords;
  const __half*        centers;
  const float*         norms;
  int                  numClusters;
  int*                 labels;
  float*               distances;
  int*                 changed;
};

/**
 * Per-row scoring for an assignment: larger scores are better. The cosine metric's squared Euclidean distance between
 * normalized vectors is 1 - score (0 - score for an empty row); the Tanimoto distance is 1 - score.
 */
template <KMeansMetric Metric> struct RowScorer {
  float count;
  float weight;

  __device__ __forceinline__ explicit RowScorer(const int bitCount)
      : count(static_cast<float>(bitCount)),
        weight(Metric == KMeansMetric::Cosine && bitCount > 0 ? (1.0F / sqrtf(static_cast<float>(bitCount))) : 1.0F) {}

  __device__ __forceinline__ float score(const float dot, const float norm) const {
    if constexpr (Metric == KMeansMetric::Cosine) {
      return 2.0F * weight * dot - norm;
    } else {
      const float denominator = count + norm - dot;
      return denominator > 0.0F ? __fdiv_rn(dot, denominator) : 1.0F;
    }
  }

  __device__ __forceinline__ float distance(const float score) const {
    if constexpr (Metric == KMeansMetric::Cosine) {
      return fmaxf((count > 0.0F ? 1.0F : 0.0F) - score, 0.0F);
    } else {
      return 1.0F - score;
    }
  }
};

//! Stores a row's final assignment and counts label changes against the previous labels.
template <KMeansMetric Metric>
__device__ __forceinline__ bool storeAssignment(const AssignArgs&        args,
                                                const int                row,
                                                const RowScorer<Metric>& scorer,
                                                const float              score,
                                                const int                index) {
  const int previous  = args.labels[row];
  args.labels[row]    = index;
  args.distances[row] = scorer.distance(score);
  return previous != index;
}

//! True when the current device runs the mma.sync kernel (sm_80 or newer) with fingerprints of paddedBits bits.
bool mmaSupported(int paddedBits);

template <KMeansMetric Metric> void launchMmaAssign(const AssignArgs& args, int paddedBits, cudaStream_t stream);

}  // namespace nvMolKit::detail

#endif  // NVMOLKIT_KMEANS_KERNELS_CUH
