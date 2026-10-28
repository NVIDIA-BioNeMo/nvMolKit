// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_BUTINA_COMMON_CUH
#define NVMOLKIT_BUTINA_COMMON_CUH

#include <cuda_runtime.h>

#include <cstdint>

namespace nvMolKit {

// Pack a neighbor count and point index into a 64 bit unsigned int.Stores the neighbor hit count in bits 32-63 and the
// point index in bits 0-31. Ultimate goal is so that when we sort, we break neighbor count ties by point index
__device__ __forceinline__ std::uint64_t makeButinaCandidate(const int value, const int index) {
  if (value < 0) {
    return 0;
  }
  const auto encodedValue = static_cast<std::uint32_t>(value) + 1U;
  return (static_cast<std::uint64_t>(encodedValue) << 32) | static_cast<std::uint32_t>(index);
}

// Return the point index from a candidate produced by makeButinaCandidate(), or -1 for an inactive candidate.
__device__ __forceinline__ int butinaCandidateIndex(const std::uint64_t candidate) {
  return static_cast<std::uint32_t>(candidate >> 32) == 0 ? -1 :
                                                            static_cast<int>(static_cast<std::uint32_t>(candidate));
}

// Decode a candidate produced by makeButinaCandidate() and store its neighbor count and point index.
__device__ __forceinline__ int decodeAndStoreButinaCandidate(const std::uint64_t candidate, int* value, int* index) {
  const auto encodedValue = static_cast<std::uint32_t>(candidate >> 32);
  *value                  = encodedValue == 0 ? -1 : static_cast<int>(encodedValue) - 1;
  *index                  = butinaCandidateIndex(candidate);
  return *value;
}

// Update a conditional WHILE graph from a device-side scalar threshold.
static __global__ void setConditionalLoopGraphCondition(cudaGraphConditionalHandle handle,
                                                        const int*                 value,
                                                        const int                  threshold) {
  cudaGraphSetConditional(handle, *value >= threshold ? 1 : 0);
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_BUTINA_COMMON_CUH
