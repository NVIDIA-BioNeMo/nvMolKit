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

#ifndef NVMOLKIT_MCS_TYPES_H
#define NVMOLKIT_MCS_TYPES_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace nvMolKit {

/// Atom compatibility modes corresponding to RDKit's built-in FMCS comparators.
enum class MCSAtomCompare : std::uint8_t {
  Any,          ///< Match all atoms.
  Elements,     ///< Match atomic numbers (the default).
  Isotopes,     ///< Match isotope values; this is the supported isotope-matching mode.
  AnyHeavyAtom  ///< Unsupported; findMCSBatch throws std::invalid_argument.
};

/// Bond compatibility modes corresponding to RDKit's built-in FMCS comparators.
enum class MCSBondCompare : std::uint8_t {
  Any,        ///< Match all bonds.
  Order,      ///< Match bond order with RDKit's aromatic/single compatibility (the default).
  OrderExact  ///< Match exact bond order, including aromaticity.
};

struct MCSAtomCompareParameters {
  bool matchValences       = false;  ///< Require equal total valence.
  bool matchChiralTag      = false;  ///< Unsupported; setting true throws std::invalid_argument.
  bool matchFormalCharge   = false;  ///< Require equal formal charge.
  bool ringMatchesRingOnly = false;  ///< Prevent ring atoms from matching non-ring atoms.
  bool completeRingsOnly   = false;  ///< Unsupported; setting true throws std::invalid_argument.

  /// Unsupported; setting true throws std::invalid_argument.
  /// Use atomCompare=MCSAtomCompare::Isotopes instead.
  bool matchIsotope = false;

  /// Unsupported; values other than -1.0 throw std::invalid_argument.
  double maxDistance = -1.0;
};

struct MCSBondCompareParameters {
  bool ringMatchesRingOnly   = false;  ///< Prevent ring bonds from matching non-ring bonds.
  bool completeRingsOnly     = false;  ///< Unsupported; setting true throws std::invalid_argument.
  bool matchFusedRings       = false;  ///< Unsupported; setting true throws std::invalid_argument.
  bool matchFusedRingsStrict = false;  ///< Unsupported; setting true throws std::invalid_argument.
  bool matchStereo           = false;  ///< Unsupported; setting true throws std::invalid_argument.
};

/// Search, execution, and fallback controls for the RDKit-molecule batch API.
struct MCSParameters {
  /// Unsupported; setting true throws std::invalid_argument.
  bool storeAll = false;

  /// Must be true. Setting false (maximize atoms) throws std::invalid_argument.
  bool maximizeBonds = true;

  /// Must be 1.0 for this pairwise API; other values throw std::invalid_argument.
  double threshold = 1.0;

  /// Unsupported; setting true throws std::invalid_argument.
  bool verbose = false;

  /// Must be true. Disconnected MCS is unsupported and throws std::invalid_argument.
  bool connectedOnly = true;

  /// Unsupported; non-empty seed SMARTS throws std::invalid_argument.
  std::string initialSeed;

  /// Controls pair-specific RDKit fallback only.
  ///
  /// false: pairs beyond GPU molecule limits or pairs whose GPU search
  /// overflows are recomputed with RDKit. true: those cases throw
  /// std::runtime_error. Unsupported batch-wide parameters, invalid inputs or
  /// configuration, CUDA errors, and timeouts never fall back for either value.
  bool requireGpu = false;

  /// Per-pair budget in whole seconds; zero disables it. A timeout returns a
  /// possibly partial result with canceled=true and never falls back.
  unsigned int timeoutSeconds = 0;

  /// Pairs per GPU chunk; values <= 0 use all pairs assigned to the runner.
  int batchSize = 0;

  /// CUDA block size; valid values are 64, 128, 256, and 512.
  int blockSize = 128;

  /// Runner threads; -1 selects automatically, otherwise clamped to at least 1.
  int workerThreads = -1;

  /// Pair-preparation threads; -1 uses hardware concurrency, otherwise clamped to at least 1.
  int preprocessingThreads = -1;

  /// Executors per runner; -1 selects automatically, otherwise valid from 1 through 8.
  int executorsPerRunner = -1;

  /// CUDA device IDs. Empty uses the current CUDA device only.
  std::vector<int> gpuIds;

  /// Any, Elements, and Isotopes are supported; AnyHeavyAtom throws.
  MCSAtomCompare atomCompare = MCSAtomCompare::Elements;

  /// Any, Order, and OrderExact are supported.
  MCSBondCompare           bondCompare = MCSBondCompare::Order;
  MCSAtomCompareParameters atomCompareParameters;
  MCSBondCompareParameters bondCompareParameters;
};

struct MCSResult {
  unsigned int numAtoms     = 0;      ///< Atoms in the best common subgraph found.
  unsigned int numBonds     = 0;      ///< Bonds in the best common subgraph found.
  bool         canceled     = false;  ///< Timeout occurred; mappings may contain a valid partial result.
  bool         overflowed   = false;  ///< Normally replaced by fallback or an exception in the public API.
  bool         usedGpu      = false;  ///< Result came from the GPU solver.
  bool         usedFallback = false;  ///< Pair was recomputed by RDKit on the CPU.

  /// Populated for RDKit fallback results; native GPU results currently leave it empty.
  std::string smartsString;

  /// Atom-index pairs in (first molecule, second molecule) order.
  std::vector<std::pair<int, int>> atomMapping;

  /// Bond-index pairs in (first molecule, second molecule) order.
  std::vector<std::pair<int, int>> bondMapping;

  [[nodiscard]] bool isCompleted() const { return !canceled; }
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_MCS_TYPES_H
