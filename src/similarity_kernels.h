// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SIMILARITY_KERNELS_H
#define SIMILARITY_KERNELS_H

#include <boost/dynamic_bitset.hpp>
#include <cstdint>
#include <vector>

#include "src/utils/device_vector.h"

namespace nvMolKit {
namespace internal {

//! Threads per pair of fps
constexpr int kNumThreadsPerFingerPrintTemporaryFixed = 32;
using kBlockType                                      = boost::dynamic_bitset<>::block_type;

}  // namespace internal

// --------------------------------
// Tanimoto similarity launch functions
// --------------------------------

//! Launches a kernel to compute the all-to-all Tanimoto similarity between a list of fingerprints.
//! \param bits The list of fingerprints
//! \param results The output similarity, not safe to use until a sync.
template <typename kThreadReductionType>
void launchCrossTanimotoSimilarity(const AsyncDeviceVector<internal::kBlockType>& bitsOne,
                                   const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
                                   const size_t                                   numBitsPerMolecule,
                                   AsyncDeviceVector<double>&                     results,
                                   const size_t                                   offset);

//! Launches a kernel to compute the all-to-all Tanimoto similarity between a list of fingerprints.
//! \param bits The list of fingerprints
//! \param results The output similarity, not safe to use until a sync.
void launchCrossTanimotoSimilarity(const cuda::std::span<const std::uint32_t> bitsOne,
                                   const cuda::std::span<const std::uint32_t> bitsTwo,
                                   const size_t                               numBitsPerMolecule,
                                   const cuda::std::span<double>              results,
                                   const size_t                               offset,
                                   cudaStream_t                               stream = nullptr);

// --------------------------------
// Tanimoto similarity explicit template instantiations
// --------------------------------

extern template void launchCrossTanimotoSimilarity<typename std::uint32_t>(
  const AsyncDeviceVector<internal::kBlockType>& bitsOne,
  const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
  const size_t                                   numBitsPerMolecule,
  AsyncDeviceVector<double>&                     results,
  const size_t                                   offset

);

// --------------------------------
// Cosine similarity launch functions
// --------------------------------

//! Launches a kernel to compute the all-to-all Cosine similarity between a list of fingerprints.
//! \param bits The list of fingerprints
//! \param results The output similarity, not safe to use until a sync.
template <typename kThreadReductionType>
void launchCrossCosineSimilarity(const AsyncDeviceVector<internal::kBlockType>& bitsOne,
                                 const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
                                 const size_t                                   numBitsPerMolecule,
                                 AsyncDeviceVector<double>&                     results,
                                 const size_t                                   offset);

//! Launches a kernel to compute the all-to-all Cosine similarity between a list of fingerprints.
//! \param bits The list of fingerprints
//! \param results The output similarity, not safe to use until a sync.
void launchCrossCosineSimilarity(const cuda::std::span<const std::uint32_t> bitsOne,
                                 const cuda::std::span<const std::uint32_t> bitsTwo,
                                 const size_t                               numBitsPerMolecule,
                                 const cuda::std::span<double>              results,
                                 const size_t                               offset,
                                 cudaStream_t                               stream = nullptr);

// --------------------------------
// Cosine similarity explicit template instantiations
// --------------------------------

extern template void launchCrossCosineSimilarity<typename std::uint32_t>(
  const AsyncDeviceVector<internal::kBlockType>& bitsOne,
  const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
  const size_t                                   numBitsPerMolecule,
  AsyncDeviceVector<double>&                     results,
  const size_t                                   offset

);

}  // namespace nvMolKit

#endif  // SIMILARITY_KERNELS_H
