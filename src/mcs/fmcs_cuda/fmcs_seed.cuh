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

#ifndef FMCS_CUDA_FMCS_SEED_CUH
#define FMCS_CUDA_FMCS_SEED_CUH

#include <cstdint>
#include <type_traits>

namespace mcs {

/// Bitset word type: uint32_t for maxSize <= 32, uint64_t otherwise.
template <int maxSize> using BitWord = std::conditional_t<(maxSize <= 32), uint32_t, uint64_t>;

namespace fmcs {

/// Query bond that dangles off the current seed boundary and is eligible to
/// be added in the next grow step.  @c endAtomSeedIdx == kNotInSeed indicates
/// the other endpoint is not yet present in the seed (an atom-adding bond);
/// anything else is a ring-closing bond onto an existing seed atom.
/// @c alive is cleared by the individual-bond pruning pass.
struct NewBond {
  static constexpr uint16_t kNotInSeed = 0xFFFFu;

  uint16_t bondIdx        = 0;
  uint16_t newAtomIdx     = 0;
  uint16_t endAtomSeedIdx = kNotInSeed;
  bool     alive          = true;
};

/// Connected subgraph of the query molecule, represented as bitsets over the
/// query's atom and bond index spaces.  Per-target match state lives
/// separately in @ref MatchResult.
template <int maxAtoms, int maxBonds> struct Seed {
  using atom_word_type                  = BitWord<maxAtoms>;
  using bond_word_type                  = BitWord<maxBonds>;
  static constexpr int kAtomBitsPerWord = sizeof(atom_word_type) * 8;
  static constexpr int kBondBitsPerWord = sizeof(bond_word_type) * 8;
  static constexpr int kAtomWords       = (maxAtoms + kAtomBitsPerWord - 1) / kAtomBitsPerWord;
  static constexpr int kBondWords       = (maxBonds + kBondBitsPerWord - 1) / kBondBitsPerWord;

  atom_word_type atoms[kAtomWords]          = {};
  bond_word_type bonds[kBondWords]          = {};
  /// Query bonds forbidden from further grows of this seed, to avoid
  /// reaching the same seed via two different grow paths.
  bond_word_type excludedBonds[kBondWords]  = {};
  /// Subset of @c atoms added in the most recent grow step.  This mirrors
  /// RDKit's LastAddedAtomsBeginIdx scan: fillNewBondsCooperative only
  /// enumerates query bonds incident to atoms that were newly appended to the
  /// seed, while sibling/subset coverage comes from RDKit Stage 1/Stage 2.
  atom_word_type lastAddedAtoms[kAtomWords] = {};

  uint16_t numAtoms = 0;
  uint16_t numBonds = 0;

  /// Upper bound on additional atoms/bonds reachable from this seed,
  /// cached by @ref computeRemainingSize for incumbent-size pruning.
  uint16_t remainingAtoms = 0;
  uint16_t remainingBonds = 0;
  /// RDKit Seed::GrowingStage analogue.  0 means a fresh seed that should run
  /// the outer grow stage, 1 means resume the same parent at the inner
  /// singleton/subset stage.
  uint16_t growingStage   = 0;
};

constexpr uint16_t kSeedGrowStageOuter = 0;
constexpr uint16_t kSeedGrowStageInner = 1;

/// Sentinel stored in @c MatchResult::targetAtomIdx[q] /
/// @c targetBondIdx[q] when the query atom / bond @c q has not yet been
/// mapped to any target counterpart.  uint8 max value, distinguishable
/// from any valid target index since tiers cap maxAtoms/maxBonds at 128.
constexpr std::uint8_t kUnmappedTargetIdx = 0xFFu;

/// Per-target incremental match cache attached to a Seed.  When non-empty,
/// @ref matchIncrementalFast extends the recorded mapping by just the bonds
/// added since @c matchedBondSize; the visited-atom/bond bitsets keep
/// successive extensions from reusing a target atom or bond.
template <int maxAtoms, int maxBonds, int maxTargetAtoms, int maxTargetBonds> struct MatchResult {
  using target_atom_word                      = BitWord<maxTargetAtoms>;
  using target_bond_word                      = BitWord<maxTargetBonds>;
  static constexpr int kTargetAtomBitsPerWord = sizeof(target_atom_word) * 8;
  static constexpr int kTargetBondBitsPerWord = sizeof(target_bond_word) * 8;
  static constexpr int kTargetAtomWords       = (maxTargetAtoms + kTargetAtomBitsPerWord - 1) / kTargetAtomBitsPerWord;
  static constexpr int kTargetBondWords       = (maxTargetBonds + kTargetBondBitsPerWord - 1) / kTargetBondBitsPerWord;

  /// queryAtomIdx -> targetAtomIdx, 0xFF for unmapped.  Query-indexed so the
  /// map does not have to be rebuilt when the seed's atom list grows.
  uint8_t targetAtomIdx[maxAtoms];
  /// queryBondIdx -> targetBondIdx, 0xFF for unmapped.
  uint8_t targetBondIdx[maxBonds];

  target_atom_word visitedTargetAtoms[kTargetAtomWords] = {};
  target_bond_word visitedTargetBonds[kTargetBondWords] = {};

  uint16_t matchedAtomSize = 0;
  uint16_t matchedBondSize = 0;
  bool     empty           = true;
};

/// Storage wrapper bundling a @ref Seed with its per-target @ref MatchResult.
/// The seed queue and per-block incumbent both hold these so the recorded
/// embedding travels with the subgraph it describes.
///
/// @c alignas(16) ensures every @c QueuedSeed in an array (whether in
/// shared or global memory) starts on a 16-byte boundary, which is
/// what the int4-granularity @c warpCopy path requires.
/// Without this, a sequence of two @c __shared__ @c QueuedSeed
/// declarations or @c QueuedSeed[N] arrays at non-tier-128 sizes
/// would put the second element at a 4- or 8-byte boundary and
/// trigger @c cudaErrorMisalignedAddress on the int4 load/store.
template <int maxAtoms, int maxBonds, int maxTargetAtoms, int maxTargetBonds> struct alignas(16) QueuedSeed {
  Seed<maxAtoms, maxBonds>                                        seed;
  MatchResult<maxAtoms, maxBonds, maxTargetAtoms, maxTargetBonds> match;
};

// ---------------------------------------------------------------------------
// Helper signatures.
//
// Naming convention: functions that operate within a single thread carry
// the @c WithinThread suffix; cooperative-group helpers carry the
// @c Cooperative suffix and take a @c GroupT.  Only the version that is
// actually called by the kernel is provided -- a pair of within-thread
// + cooperative variants is added only when both call sites exist.
//
// Concurrency contract for @c *WithinThread helpers: the caller is
// responsible for ensuring exclusive write access to the target Seed or
// MatchResult.  Internally these do non-atomic read-modify-writes
// (e.g. @c numAtoms += 1, bitword OR) and would silently lose updates
// under concurrent calls on the same target.  Today the kernel
// guarantees exclusivity by giving each lane (Phase 1) or each group's
// designated leader (Phase 2 Stage 0 patch) its own per-lane / per-
// group destination slot.
// ---------------------------------------------------------------------------

/// Within-thread: sets bit @p atomIdx in both @c seed.atoms and
/// @c seed.lastAddedAtoms, and increments @c numAtoms.  Caller is a
/// single lane that already owns @p seed.  Callers wrap a batch of
/// @ref seedAddAtomWithinThread calls in a
/// @ref seedBeginGrowStepWithinThread to clear @c lastAddedAtoms
/// first, so only THIS step's atoms appear in the boundary set used by
/// @ref fillNewBondsCooperative.
///
/// Idempotent: adding an atom already in @c seed.atoms is a no-op, so the
/// invariant @c numAtoms == popcount(atoms) always holds.  A single grow
/// step can present the same new atom through two boundary bonds (a ring
/// closed within one step reaches the new atom from two seed atoms); the
/// guard keeps that from double-counting @c numAtoms.
template <int maxAtoms, int maxBonds>
__device__ __forceinline__ void seedAddAtomWithinThread(Seed<maxAtoms, maxBonds>& seed, const int atomIdx) {
  using AtomWord              = typename Seed<maxAtoms, maxBonds>::atom_word_type;
  constexpr int  kBitsPerWord = Seed<maxAtoms, maxBonds>::kAtomBitsPerWord;
  const int      wordIdx      = atomIdx / kBitsPerWord;
  const AtomWord mask         = static_cast<AtomWord>(1) << (atomIdx % kBitsPerWord);
  if ((seed.atoms[wordIdx] & mask) != 0)
    return;
  seed.atoms[wordIdx] |= mask;
  seed.lastAddedAtoms[wordIdx] |= mask;
  seed.numAtoms += 1;
}

/// Within-thread: sets bit @p bondIdx in @c seed.bonds AND
/// @c seed.excludedBonds (excludedBonds prevents the same bond being
/// re-added through a different grow path), and increments @c numBonds.
template <int maxAtoms, int maxBonds>
__device__ __forceinline__ void seedAddBondWithinThread(Seed<maxAtoms, maxBonds>& seed, const int bondIdx) {
  using BondWord              = typename Seed<maxAtoms, maxBonds>::bond_word_type;
  constexpr int  kBitsPerWord = Seed<maxAtoms, maxBonds>::kBondBitsPerWord;
  const int      wordIdx      = bondIdx / kBitsPerWord;
  const BondWord mask         = static_cast<BondWord>(1) << (bondIdx % kBitsPerWord);
  seed.bonds[wordIdx] |= mask;
  seed.excludedBonds[wordIdx] |= mask;
  seed.numBonds += 1;
}

/// Within-thread: marks a query bond as unavailable for future grows without
/// adding it to the seed.  RDKit uses this while constructing initial seeds:
/// later initial seeds exclude earlier query bonds, and a mismatched initial
/// bond is also excluded from seeds already admitted.
template <int maxAtoms, int maxBonds>
__device__ __forceinline__ void seedExcludeBondWithinThread(Seed<maxAtoms, maxBonds>& seed, const int bondIdx) {
  using BondWord              = typename Seed<maxAtoms, maxBonds>::bond_word_type;
  constexpr int  kBitsPerWord = Seed<maxAtoms, maxBonds>::kBondBitsPerWord;
  const int      wordIdx      = bondIdx / kBitsPerWord;
  const BondWord mask         = static_cast<BondWord>(1) << (bondIdx % kBitsPerWord);
  seed.excludedBonds[wordIdx] |= mask;
}

/// Within-thread: clear @c seed.lastAddedAtoms.  Subsequent
/// @ref seedAddAtomWithinThread calls in this step then
/// populate @c lastAddedAtoms with the atoms touched by this grow step;
/// @ref fillNewBondsCooperative reads that bitset to decide which query
/// bonds should be enumerated as new boundary candidates.
template <int maxAtoms, int maxBonds>
__device__ __forceinline__ void seedBeginGrowStepWithinThread(Seed<maxAtoms, maxBonds>& seed) {
  for (int i = 0; i < Seed<maxAtoms, maxBonds>::kAtomWords; ++i) {
    seed.lastAddedAtoms[i] = 0;
  }
}

/// Within-thread: pure scalar predicate; safe for every lane to evaluate
/// independently against a uniform @p seed reference.  Returns true iff
/// the seed could still grow strictly larger than the
/// (@p bestBonds, @p bestAtoms) incumbent under MaximizeBonds tie-break,
/// using the cached @c remainingAtoms / @c remainingBonds upper bound
/// populated by @ref seedComputeRemainingSizeRdkitCooperative.
template <int maxAtoms, int maxBonds>
__device__ __forceinline__ bool seedCanGrowBiggerThanWithinThread(const Seed<maxAtoms, maxBonds>& seed,
                                                                  int                             bestBonds,
                                                                  int                             bestAtoms) {
  const int possibleBonds = seed.numBonds + seed.remainingBonds;
  if (possibleBonds > bestBonds)
    return true;
  if (possibleBonds < bestBonds)
    return false;
  const int possibleAtoms = seed.numAtoms + seed.remainingAtoms;
  return possibleAtoms > bestAtoms;
}

/// Within-thread: zeroes a Seed in-place.  Tier-128 cost is ~58 B; well
/// under the per-lane budget of a serial clear, so cooperative variant
/// not provided.
template <int maxAtoms, int maxBonds>
__device__ __forceinline__ void seedClearWithinThread(Seed<maxAtoms, maxBonds>& seed) {
  for (int i = 0; i < Seed<maxAtoms, maxBonds>::kAtomWords; ++i) {
    seed.atoms[i]          = 0;
    seed.lastAddedAtoms[i] = 0;
  }
  for (int i = 0; i < Seed<maxAtoms, maxBonds>::kBondWords; ++i) {
    seed.bonds[i]         = 0;
    seed.excludedBonds[i] = 0;
  }
  seed.numAtoms       = 0;
  seed.numBonds       = 0;
  seed.remainingAtoms = 0;
  seed.remainingBonds = 0;
  seed.growingStage   = kSeedGrowStageOuter;
}

/// Within-thread: zeroes a MatchResult in-place.  Tier-128 cost is
/// ~280 B (mostly the 0xFF fill of targetAtomIdx[maxAtoms] and
/// targetBondIdx[maxBonds]).  Currently called by the lane that won
/// a per-thread queue slot in Phase 1, so the per-lane serial cost is
/// unavoidable without restructuring; revisit in Step 6 if profiling
/// shows it dominant.
template <int maxAtoms, int maxBonds, int maxTA, int maxTB>
__device__ __forceinline__ void matchResultClearWithinThread(MatchResult<maxAtoms, maxBonds, maxTA, maxTB>& match) {
  for (int i = 0; i < maxAtoms; ++i)
    match.targetAtomIdx[i] = kUnmappedTargetIdx;
  for (int i = 0; i < maxBonds; ++i)
    match.targetBondIdx[i] = kUnmappedTargetIdx;
  for (int i = 0; i < MatchResult<maxAtoms, maxBonds, maxTA, maxTB>::kTargetAtomWords; ++i) {
    match.visitedTargetAtoms[i] = 0;
  }
  for (int i = 0; i < MatchResult<maxAtoms, maxBonds, maxTA, maxTB>::kTargetBondWords; ++i) {
    match.visitedTargetBonds[i] = 0;
  }
  match.matchedAtomSize = 0;
  match.matchedBondSize = 0;
  match.empty           = true;
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_SEED_CUH
