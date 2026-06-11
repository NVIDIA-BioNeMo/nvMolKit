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

#ifndef NVMOLKIT_SUBSTRUCT_TYPES_H
#define NVMOLKIT_SUBSTRUCT_TYPES_H

#include <cstdint>

#include "src/substruct/substruct_constants.h"
#include "src/substruct/substruct_results.h"
#include "src/substruct/substruct_template_config.h"

namespace nvMolKit {

/**
 * @brief Buffers to zero on the first label matrix kernel launch in a mini-batch.
 *
 * Passed as an optional parameter to launchLabelMatrixPaintKernel; when absent
 * (std::nullopt), no zeroing is performed.
 */
struct ZeroBuffersSpec {
  uint32_t* recursiveBits     = nullptr;
  int       recursiveBitsSize = 0;
  uint8_t*  overflowFlags     = nullptr;
  int       overflowFlagsSize = 0;
};

/**
 * @brief Per-pattern metadata for batched recursive preprocessing kernel.
 *
 * Each entry describes one recursive pattern in the combined batch:
 * which main query it belongs to, what bit to paint, and where the
 * pattern data starts in the combined pattern batch.
 */
struct BatchedPatternEntry {
  int mainQueryIdx;     ///< Index of the main query this pattern belongs to
  int patternId;        ///< Bit position (0-31) to paint for this pattern
  int patternMolIdx;    ///< Index into the combined patterns MoleculesDevice
  int depth;            ///< Nesting depth (0=leaf, higher=parent of children)
  int localIdInParent;  ///< Bit position in parent's input (for nested patterns)
};

// =============================================================================
// Partial Match Structure (for GSI algorithm queue)
// =============================================================================

/**
 * @brief Partial match for BFS-style algorithms.
 *
 * @tparam MaxQueryAtoms Maximum query atoms for array sizing
 *
 * Represents a partial mapping from query atoms to target atoms.
 * Stored compactly for queue-based BFS exploration.
 *
 * Complete matches are never stored in the queue, so we only need
 * MaxQueryAtoms - 1 slots (matching atoms 0 through numQueryAtoms-2).
 */
template <std::size_t MaxQueryAtoms = kMaxQueryAtoms> struct PartialMatchT {
  static constexpr std::size_t kMaxQueryAtomsValue = MaxQueryAtoms;
  int8_t mapping[MaxQueryAtoms - 1];  ///< mapping[q] = target atom (only [0..nextQueryAtom-1] valid)
  int8_t nextQueryAtom;               ///< Next query atom to extend (also serves as depth)
};

/// Type alias for max-sized partial match (backward compatibility)
using PartialMatch = PartialMatchT<kMaxQueryAtoms>;
static_assert(sizeof(PartialMatch) == kMaxQueryAtoms, "PartialMatch must be kMaxQueryAtoms bytes");

/**
 * @brief Entry representing a (target, query) pair that needs RDKit fallback processing.
 *
 * Used when GPU processing cannot handle a pair, either due to:
 * - Target molecule exceeding kMaxTargetAtoms
 * - Output buffer overflow during GPU matching
 */
struct RDKitFallbackEntry {
  int originalTargetIdx;  ///< Index in the original input targets vector
  int originalQueryIdx;   ///< Index in the original input queries vector

  bool operator<(const RDKitFallbackEntry& other) const {
    if (originalTargetIdx != other.originalTargetIdx) {
      return originalTargetIdx < other.originalTargetIdx;
    }
    return originalQueryIdx < other.originalQueryIdx;
  }
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_TYPES_H
