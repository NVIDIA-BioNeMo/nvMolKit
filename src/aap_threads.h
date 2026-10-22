// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_AAP_THREADS_H
#define NVMOLKIT_AAP_THREADS_H

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace nvMolKit::detail {

inline void validateAapThreadCount(const int requestedThreads) {
  if (requestedThreads != -1 && requestedThreads <= 0) {
    throw std::invalid_argument("numThreads must be -1 or positive");
  }
}

/** Select host descriptor workers from the molecule count and CPU budget. */
inline std::size_t aapDescriptorThreadCount(const std::size_t  numMolecules,
                                            const int          requestedThreads,
                                            const unsigned int availableThreads) {
  validateAapThreadCount(requestedThreads);
  constexpr std::size_t kMoleculesPerThread = 64;
  const std::size_t     threadLimit = requestedThreads == -1 ? std::max(1U, availableThreads) : requestedThreads;
  return std::clamp<std::size_t>(numMolecules / kMoleculesPerThread, 1, threadLimit);
}

}  // namespace nvMolKit::detail

#endif  // NVMOLKIT_AAP_THREADS_H
