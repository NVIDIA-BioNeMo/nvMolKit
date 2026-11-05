// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_FINGERPRINT_SEARCH_H
#define NVMOLKIT_FINGERPRINT_SEARCH_H

#include <cuda_runtime.h>

#include <cstdint>
#include <cuda/std/span>

#include "src/fingerprint_similarity.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

//! Top-k neighbors per query, row-major with shape (numQueries, k).
//! Rows with fewer than k hits are padded with NaN scores and -1 indices.
struct FingerprintSearchResult {
  AsyncDeviceVector<double>       scores;
  AsyncDeviceVector<std::int64_t> indices;
};

//! Exact top-k similarity search over a fixed library of packed fingerprints.
//!
//! The library is copied to the device sorted by popcount. Each query scans outward from library rows with a similar
//! popcount and stops once the popcount bound shows that no remaining row can enter its current top k. Results are
//! ordered by descending similarity, then ascending library index, and match a brute-force search exactly.
class FingerprintSearchIndex {
 public:
  FingerprintSearchIndex(cuda::std::span<const std::uint32_t> fingerprints,
                         int                                  numFingerprints,
                         int                                  numWords,
                         FingerprintSimilarityMetric          metric,
                         cudaStream_t                         stream = nullptr);

  //! Search queries (numQueries x numWords, row-major) against the library.
  //! Only hits with similarity >= threshold are returned. prune=false scans every library row and gives identical
  //! results; it exists to measure the effect of pruning.
  [[nodiscard]] FingerprintSearchResult search(cuda::std::span<const std::uint32_t> queries,
                                               int                                  numQueries,
                                               int                                  k,
                                               double                               threshold,
                                               bool                                 prune,
                                               cudaStream_t                         stream = nullptr) const;

  //! Search the library against itself. Each fingerprint's match with itself is excluded; output row i belongs to
  //! library fingerprint i.
  [[nodiscard]] FingerprintSearchResult searchSelf(int          k,
                                                   double       threshold,
                                                   bool         prune,
                                                   cudaStream_t stream = nullptr) const;

  [[nodiscard]] int                         size() const { return numFingerprints_; }
  [[nodiscard]] int                         numWords() const { return numWords_; }
  [[nodiscard]] FingerprintSimilarityMetric metric() const { return metric_; }

 private:
  int                              numFingerprints_;
  int                              numWords_;
  FingerprintSimilarityMetric      metric_;
  int                              deviceId_ = 0;
  AsyncDeviceVector<std::uint32_t> sortedBits_;
  AsyncDeviceVector<int>           sortedCounts_;
  AsyncDeviceVector<int>           originalIndices_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_FINGERPRINT_SEARCH_H
