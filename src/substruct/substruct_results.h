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

#ifndef NVMOLKIT_SUBSTRUCT_RESULTS_H
#define NVMOLKIT_SUBSTRUCT_RESULTS_H

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nvMolKit {

/**
 * @brief Algorithm choice for substructure matching.
 */
enum class SubstructAlgorithm {
  VF2,  ///< VF2 iterative stack-based DFS
  GSI   ///< GSI-inspired BFS level-by-level join
};

/**
 * @brief Configuration for substructure search execution.
 *
 * Controls threading, batching, and multi-GPU behavior. Default configuration uses
 * autoselect (-1) which determines optimal thread counts based on hardware.
 *
 * Multi-GPU mode: When gpuIds is non-empty, work is distributed across the specified
 * GPUs using round-robin assignment. Each GPU gets workerThreads workers, so total
 * worker threads = workerThreads * gpuIds.size().
 *
 * Threading autoselect (-1 for any thread count):
 * - preprocessingThreads: uses hardware_concurrency
 * - workerThreads: min(4, hardware_concurrency / numGpus)
 *
 * RDKit fallback (for oversized molecules or overflow) is processed opportunistically
 * by preprocessing threads while waiting for GPU work to complete.
 */
struct SubstructSearchConfig {
  int batchSize            = 1024;  ///< Number of (target, query) pairs per GPU batch
  int workerThreads        = -1;    ///< GPU runner threads per GPU (-1 = autoselect)
  int preprocessingThreads = -1;  ///< CPU threads for preprocessing and opportunistic RDKit fallback (-1 = autoselect)
  int executorsPerRunner   = -1;  ///< GPU executors per runner thread (-1 = auto: 3 for single runner, 2 otherwise)
  std::vector<int> gpuIds;        ///< GPU device IDs to use (empty = current device only)
  int              maxMatches = 0;      ///< Max matches per pair (0 = unlimited, like RDKit)
  bool             uniquify   = false;  ///< Remove duplicate matches differing only in atom enumeration order
};

/**
 * @brief Accumulated results from substructure matching.
 *
 * Uses sparse storage: only pairs with matches are stored.
 * Memory is allocated proportional to actual matches found.
 */
struct SubstructSearchResults {
  /// Sparse storage: pairIndex -> vector of matches
  /// Each match is a vector<int> of target atom indices (one per query atom)
  std::unordered_map<int64_t, std::vector<std::vector<int>>> matches;

  int numTargets = 0;
  int numQueries = 0;

  void resize(int nTargets, int nQueries) {
    numTargets = nTargets;
    numQueries = nQueries;
    matches.clear();
    // Reserve based on expected sparsity (assume ~10% of pairs have matches)
    const size_t expectedPairs = static_cast<size_t>(nTargets) * nQueries / 10 + 1;
    matches.reserve(expectedPairs);
  }

  /// Compute flat pair index
  [[nodiscard]] int64_t pairIndex(int targetIdx, int queryIdx) const {
    return static_cast<int64_t>(targetIdx) * numQueries + queryIdx;
  }

  /// Number of matches for this pair
  [[nodiscard]] int matchCount(int targetIdx, int queryIdx) const {
    auto it = matches.find(pairIndex(targetIdx, queryIdx));
    return (it != matches.end()) ? static_cast<int>(it->second.size()) : 0;
  }

  /// Get the matches for a (target, query) pair (returns empty if none)
  [[nodiscard]] const std::vector<std::vector<int>>& getMatches(int targetIdx, int queryIdx) const {
    static const std::vector<std::vector<int>> empty;
    auto                                       it = matches.find(pairIndex(targetIdx, queryIdx));
    return (it != matches.end()) ? it->second : empty;
  }

  /// Mutable access to matches (creates entry if needed)
  std::vector<std::vector<int>>& getMatchesMut(int targetIdx, int queryIdx) {
    return matches[pairIndex(targetIdx, queryIdx)];
  }
};

/**
 * @brief Results from hasSubstructMatch - boolean per (target, query) pair.
 *
 * More efficient than full match enumeration when only existence is needed.
 * Uses uint8_t instead of bool to avoid std::vector<bool> specialization issues.
 */
struct HasSubstructMatchResults {
  std::vector<uint8_t> hasMatch;  ///< Flattened [target * numQueries + query], 0=false, non-zero=true
  int                  numTargets = 0;
  int                  numQueries = 0;

  void resize(int nTargets, int nQueries) {
    numTargets = nTargets;
    numQueries = nQueries;
    hasMatch.assign(static_cast<size_t>(nTargets) * nQueries, 0);
  }

  /// Compute flat pair index
  [[nodiscard]] int64_t pairIndex(int targetIdx, int queryIdx) const {
    return static_cast<int64_t>(targetIdx) * numQueries + queryIdx;
  }

  /// Check if target contains query as substructure
  [[nodiscard]] bool matches(int targetIdx, int queryIdx) const {
    return hasMatch[static_cast<size_t>(pairIndex(targetIdx, queryIdx))] != 0;
  }

  /// Set match result for a pair
  void setMatch(int targetIdx, int queryIdx, bool value) {
    hasMatch[static_cast<size_t>(pairIndex(targetIdx, queryIdx))] = value ? 1 : 0;
  }
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_RESULTS_H
