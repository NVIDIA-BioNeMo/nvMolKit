// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_BITBIRCH_COMMON_CUH
#define NVMOLKIT_BITBIRCH_COMMON_CUH

#include <cuda_runtime.h>

#include <cstdint>

namespace nvMolKit::bitbirch {

struct ISimTanimotoTerms {
  double commonPairs = 0.0;
  double mismatches  = 0.0;
};

//! Return floor(LS / N + 1/2), with exact ties set to one. N must be positive.
__host__ __device__ __forceinline__ bool majorityCentroidBit(const std::uint64_t linearSum, const std::uint64_t count) {
  return count != 0 && linearSum >= count / 2 + count % 2;
}

//! Add one Bit Feature component to the iSIM Jaccard--Tanimoto terms.
__host__ __device__ __forceinline__ void accumulateISimTanimotoTerm(ISimTanimotoTerms&  terms,
                                                                    const std::uint64_t linearSum,
                                                                    const std::uint64_t count) {
  // Convert before multiplying so integer component products cannot overflow.
  const double componentCount = static_cast<double>(linearSum);
  const double clusterCount   = static_cast<double>(count);
  terms.commonPairs += 0.5 * componentCount * (componentCount - 1.0);
  terms.mismatches += componentCount * (clusterCount - componentCount);
}

/**
 * Evaluate iSIM Jaccard--Tanimoto from accumulated terms.
 *
 * Empty and singleton Bit Features have no distinct pairs and are assigned a
 * value of one. An all-zero Bit Feature is also assigned one, matching
 * nvMolKit's pairwise Tanimoto convention for all-zero fingerprints.
 */
__host__ __device__ __forceinline__ double isimTanimoto(const ISimTanimotoTerms terms, const std::uint64_t count) {
  if (count <= 1) {
    return 1.0;
  }
  const double denominator = terms.commonPairs + terms.mismatches;
  return denominator > 0.0 ? terms.commonPairs / denominator : 1.0;
}

//! Compare iSIM against a threshold without performing the final division.
__host__ __device__ __forceinline__ bool isimTanimotoAtLeast(const ISimTanimotoTerms terms,
                                                             const std::uint64_t     count,
                                                             const double            threshold) {
  if (count <= 1) {
    return threshold <= 1.0;
  }
  const double denominator = terms.commonPairs + terms.mismatches;
  return denominator > 0.0 ? terms.commonPairs >= threshold * denominator : threshold <= 1.0;
}

}  // namespace nvMolKit::bitbirch

#endif  // NVMOLKIT_BITBIRCH_COMMON_CUH
