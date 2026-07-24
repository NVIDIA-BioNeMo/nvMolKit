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

#ifndef FMCS_CUDA_FMCS_MATCH_CUH
#define FMCS_CUDA_FMCS_MATCH_CUH

#include <cstdint>

#include "src/mcs/fmcs_cuda/fmcs_match_tables.cuh"
#include "src/mcs/fmcs_cuda/fmcs_seed.cuh"
#include "src/mcs/fmcs_cuda/fmcs_topology.cuh"

namespace mcs {
namespace fmcs {

/// Resolved target endpoints for a successful single-(query bond, target
/// bond, orientation) compatibility check.  Populated by
/// @ref matchSingleBondWithinThread on success only; contents are
/// unspecified on failure.  @c targetAtomU is the target atom that the
/// query bond's u endpoint was mapped to (likewise V).
struct SingleBondMatch {
  uint8_t targetAtomU;
  uint8_t targetAtomV;
};

template <int maxAtoms, int maxTargetAtoms> struct FmcsSubstructureScratch {
  // Scratch for the RDKit checkIfMatchAndAppend fallback.
  std::uint8_t seedAtomList[maxAtoms];
  std::uint8_t seedAtoms[maxAtoms];
  std::uint8_t seedDegree[maxAtoms];
  std::uint8_t targetDegree[maxTargetAtoms];
  std::uint8_t orderedQueryAtom[maxAtoms];
  std::uint8_t queryOrderPos[maxAtoms];
  std::uint8_t targetAtomForQuery[maxAtoms];
  int          currentCount;
  int          nextCount;
  int          found;
  int          overflowed;
};

template <int maxAtoms, int maxBonds>
__device__ __forceinline__ bool seedContainsBondWithinThread(const Seed<maxAtoms, maxBonds>& seed,
                                                             const int                       queryBondIdx) {
  using SeedT                    = Seed<maxAtoms, maxBonds>;
  using BondWord                 = typename SeedT::bond_word_type;
  constexpr int kBondBitsPerWord = SeedT::kBondBitsPerWord;
  if (queryBondIdx < 0 || queryBondIdx >= maxBonds)
    return false;
  const BondWord word = seed.bonds[queryBondIdx / kBondBitsPerWord];
  return ((word >> (queryBondIdx % kBondBitsPerWord)) & 1) != 0;
}

/// Within-thread: per-lane single-(query bond, target bond, orientation)
/// compatibility check used by Phase 1 initial-seed enumeration.  Writes
/// resolved target atom indices for the two endpoints of @p queryBondIdx
/// into @p outMatch and returns true on success; on false the caller
/// should not read @p outMatch.
///
/// @p reversed selects the orientation: when false, the query bond's u
/// endpoint maps to the target bond's u endpoint; when true, to the
/// target bond's v endpoint.
__device__ __forceinline__ bool matchSingleBondWithinThread(const int                    queryBondIdx,
                                                            const int                    targetBondIdx,
                                                            const bool                   reversed,
                                                            const DeviceCsrView&         queryTopology,
                                                            const DeviceCsrView&         targetTopology,
                                                            const PairMatchTablesDevice& tables,
                                                            SingleBondMatch&             outMatch) {
  // Cheap bond-table check first; if the bond labels are incompatible
  // we never need to touch the atom table or decode endpoints.
  if (!tables.bonds.testBit(queryBondIdx, targetBondIdx))
    return false;

  // Decode the packed (u<<16 | v) endpoint encoding for both bonds.
  const std::uint32_t queryEndpoints = queryTopology.bondEndpoints[queryBondIdx];
  const int           queryEndpointU = static_cast<int>(queryEndpoints >> kBondEndpointShift);
  const int           queryEndpointV = static_cast<int>(queryEndpoints & kBondEndpointMask);

  const std::uint32_t targetEndpoints = targetTopology.bondEndpoints[targetBondIdx];
  const int           targetEndpointU = static_cast<int>(targetEndpoints >> kBondEndpointShift);
  const int           targetEndpointV = static_cast<int>(targetEndpoints & kBondEndpointMask);

  // Pick which target endpoint the query's u maps to; v gets the other.
  // reversed=false: queryU -> targetU, queryV -> targetV.
  // reversed=true:  queryU -> targetV, queryV -> targetU.
  const int targetForQueryU = reversed ? targetEndpointV : targetEndpointU;
  const int targetForQueryV = reversed ? targetEndpointU : targetEndpointV;

  // Both atom-pairings must be label-compatible; either failure rejects
  // this orientation.
  if (!tables.atoms.testBit(queryEndpointU, targetForQueryU))
    return false;
  if (!tables.atoms.testBit(queryEndpointV, targetForQueryV))
    return false;

  outMatch.targetAtomU = static_cast<uint8_t>(targetForQueryU);
  outMatch.targetAtomV = static_cast<uint8_t>(targetForQueryV);
  return true;
}

/// Cooperative greedy fast path: try to extend @p match by every query
/// bond in @p seed.bonds whose @c match.targetBondIdx[q] is still
/// @ref kUnmappedTargetIdx (i.e., unmapped by the parent's recorded
/// embedding).  For each such bond:
///   - Both endpoints already mapped -> ring-closing case.  The lanes
///     of @p group scan target bonds in parallel for one whose endpoint
///     pair exactly matches the mapped (queryU, queryV) target atoms
///     and is unvisited, with the bond-match-table bit set.  First
///     compatible target bond commits.
///   - Exactly one endpoint mapped -> atom-adding case.  Lane-parallel
///     scan for a target bond incident to the mapped target atom whose
///     other end is unvisited, atom-table-compatible with the unmapped
///     query atom, and bond-table-compatible.  The first compatible
///     candidate commits both the new bond mapping and the new atom
///     mapping, and marks both visited.  This path does not backtrack
///     if that greedy choice prevents a later bond from matching.
///   - Both endpoints unmapped -> defensive fail (shouldn't occur on
///     well-formed seeds, where Phase 1 maps both initial atoms before
///     pushing).
/// A true return proves that @p match was extended successfully.  A false
/// return is inconclusive: the greedy choices may have missed another valid
/// embedding.  The caller must run the exact substructure fallback before
/// rejecting the seed or marking a NewBond dead.  On false, @p match is left
/// in an unspecified state and must not be reused as a valid embedding.
template <int maxAtoms, int maxBonds, int maxTA, int maxTB, class GroupT>
__device__ __forceinline__ bool tryMatchIncrementalGreedyCooperative(
  const GroupT&                                  group,
  const Seed<maxAtoms, maxBonds>&                seed,
  const DeviceCsrView&                           queryTopology,
  const DeviceCsrView&                           targetTopology,
  const PairMatchTablesDevice&                   tables,
  MatchResult<maxAtoms, maxBonds, maxTA, maxTB>& match) {
  using SeedT    = Seed<maxAtoms, maxBonds>;
  using MatchT   = MatchResult<maxAtoms, maxBonds, maxTA, maxTB>;
  using BondWord = typename SeedT::bond_word_type;

  constexpr int kBondBitsPerWord       = SeedT::kBondBitsPerWord;
  constexpr int kBondWords             = SeedT::kBondWords;
  constexpr int kTargetAtomBitsPerWord = MatchT::kTargetAtomBitsPerWord;
  constexpr int kTargetBondBitsPerWord = MatchT::kTargetBondBitsPerWord;
  using TargetAtomWord                 = typename MatchT::target_atom_word;
  using TargetBondWord                 = typename MatchT::target_bond_word;

  const int laneRank  = static_cast<int>(group.thread_rank());
  const int laneCount = static_cast<int>(group.num_threads());

  // Outer loop walks set bits of seed.bonds via __ffs/__ffsll.  Each
  // iteration handles one query bond q; if q is already mapped (parent
  // saw it) we skip, otherwise we extend the match by one bond +
  // possibly one atom.
  for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
    BondWord remainingBondBits = seed.bonds[wordIdx];
    while (remainingBondBits != 0) {
      int bitPosInWord;
      if constexpr (sizeof(BondWord) == 4) {
        bitPosInWord = __ffs(static_cast<unsigned int>(remainingBondBits)) - 1;
      } else {
        bitPosInWord = __ffsll(static_cast<unsigned long long>(remainingBondBits)) - 1;
      }
      const int queryBondIdx = wordIdx * kBondBitsPerWord + bitPosInWord;
      remainingBondBits &= remainingBondBits - 1;  // clear lowest set bit

      // Keep this control decision uniform across the group; diverging before
      // the later ballot/shuffle would make the winning lane undefined.
      int mappedTargetBond = kUnmappedTargetIdx;
      if (laneRank == 0)
        mappedTargetBond = match.targetBondIdx[queryBondIdx];
      mappedTargetBond = group.shfl(mappedTargetBond, 0);
      if (mappedTargetBond != kUnmappedTargetIdx)
        continue;

      // Decode this bond's query endpoints from the packed (u<<16 | v).
      const std::uint32_t queryEndpoints = queryTopology.bondEndpoints[queryBondIdx];
      const int           queryEndpointU = static_cast<int>(queryEndpoints >> kBondEndpointShift);
      const int           queryEndpointV = static_cast<int>(queryEndpoints & kBondEndpointMask);

      // Look up the parent's atom mapping for both endpoints; either
      // may already be mapped (from an earlier bond) or still unmapped
      // (this bond is the one bringing it in).
      int targetForQueryU = kUnmappedTargetIdx;
      int targetForQueryV = kUnmappedTargetIdx;
      if (laneRank == 0) {
        targetForQueryU = match.targetAtomIdx[queryEndpointU];
        targetForQueryV = match.targetAtomIdx[queryEndpointV];
      }
      targetForQueryU           = group.shfl(targetForQueryU, 0);
      targetForQueryV           = group.shfl(targetForQueryV, 0);
      const bool queryUIsMapped = targetForQueryU != kUnmappedTargetIdx;
      const bool queryVIsMapped = targetForQueryV != kUnmappedTargetIdx;

      // Both endpoints unmapped means the seed is missing an earlier
      // bond that should have brought one of them in.  Phase 1 always
      // anchors both initial atoms before pushing, so this never hits
      // for well-formed seeds; defensive return false.
      if (!queryUIsMapped && !queryVIsMapped)
        return false;

      // Each lane records its first compatible target bond, and (for
      // the atom-adding case) the resulting new target atom.  After
      // the per-lane scan, the warp ballots and the lowest-rank winner
      // is broadcast as the committed extension.
      int chosenTargetBond            = -1;
      int chosenTargetAtomForUnmapped = -1;  // -1 means ring-closing.

      if (queryUIsMapped && queryVIsMapped) {
        // Ring-closing: the new bond connects two atoms that are both
        // already in the seed's mapping.  Find a target bond whose
        // endpoints are exactly the pair { targetForQueryU, targetForQueryV }.
        const int srcTargetAtom = targetForQueryU;
        const int dstTargetAtom = targetForQueryV;
        if (srcTargetAtom >= 0 && srcTargetAtom < targetTopology.numAtoms) {
          const int begin = static_cast<int>(targetTopology.rowOffsets[srcTargetAtom]);
          const int end   = static_cast<int>(targetTopology.rowOffsets[srcTargetAtom + 1]);
          for (int adjIdx = begin + laneRank; adjIdx < end && chosenTargetBond < 0; adjIdx += laneCount) {
            const int otherTargetAtom = static_cast<int>(targetTopology.colIndices[adjIdx]);
            if (otherTargetAtom != dstTargetAtom)
              continue;
            const int targetBondIdx = static_cast<int>(targetTopology.bondIndices[adjIdx]);
            if (targetBondIdx < 0 || targetBondIdx >= targetTopology.numBonds || targetBondIdx >= maxTB) {
              continue;
            }
            const TargetBondWord visitedBondsWord = match.visitedTargetBonds[targetBondIdx / kTargetBondBitsPerWord];
            if ((visitedBondsWord >> (targetBondIdx % kTargetBondBitsPerWord)) & 1) {
              continue;
            }
            if (!tables.bonds.testBit(queryBondIdx, targetBondIdx))
              continue;
            chosenTargetBond = targetBondIdx;
          }
        }
      } else {
        // Atom-adding: one query endpoint (`src`) is already mapped to
        // a target atom; the other (`dst`) is what we're trying to
        // place.  Scan target bonds incident to srcTargetAtom for one
        // whose other endpoint is unvisited, atom-table-compatible
        // with the unmapped query atom, and bond-table-compatible.
        const int unmappedQueryAtom = queryUIsMapped ? queryEndpointV : queryEndpointU;
        const int srcTargetAtom     = queryUIsMapped ? targetForQueryU : targetForQueryV;
        if (srcTargetAtom >= 0 && srcTargetAtom < targetTopology.numAtoms) {
          const int begin = static_cast<int>(targetTopology.rowOffsets[srcTargetAtom]);
          const int end   = static_cast<int>(targetTopology.rowOffsets[srcTargetAtom + 1]);
          for (int adjIdx = begin + laneRank; adjIdx < end && chosenTargetBond < 0; adjIdx += laneCount) {
            const int targetBondIdx = static_cast<int>(targetTopology.bondIndices[adjIdx]);
            if (targetBondIdx < 0 || targetBondIdx >= targetTopology.numBonds || targetBondIdx >= maxTB) {
              continue;
            }
            const TargetBondWord visitedBondsWord = match.visitedTargetBonds[targetBondIdx / kTargetBondBitsPerWord];
            if ((visitedBondsWord >> (targetBondIdx % kTargetBondBitsPerWord)) & 1) {
              continue;
            }
            const int candidateTargetAtom = static_cast<int>(targetTopology.colIndices[adjIdx]);
            if (candidateTargetAtom < 0 || candidateTargetAtom >= targetTopology.numAtoms ||
                candidateTargetAtom >= maxTA) {
              continue;
            }
            const TargetAtomWord visitedAtomsWord =
              match.visitedTargetAtoms[candidateTargetAtom / kTargetAtomBitsPerWord];
            if ((visitedAtomsWord >> (candidateTargetAtom % kTargetAtomBitsPerWord)) & 1) {
              continue;
            }
            if (!tables.bonds.testBit(queryBondIdx, targetBondIdx))
              continue;
            if (!tables.atoms.testBit(unmappedQueryAtom, candidateTargetAtom))
              continue;
            chosenTargetBond            = targetBondIdx;
            chosenTargetAtomForUnmapped = candidateTargetAtom;
          }
        }
      }

      // Warp-wide pick: ballot for any lane with a candidate, broadcast
      // the lowest-rank winner's choice to all lanes.  No candidate
      // anywhere -> this bond can't be extended -> abandon the seed.
      const unsigned ballot = group.ballot(chosenTargetBond >= 0 ? 1u : 0u);
      if (ballot == 0u)
        return false;
      const int firstWinningLane    = __ffs(ballot) - 1;
      const int committedTargetBond = group.shfl(chosenTargetBond, firstWinningLane);
      const int committedTargetAtom = group.shfl(chosenTargetAtomForUnmapped, firstWinningLane);
      if (committedTargetBond < 0 || committedTargetBond >= targetTopology.numBonds || committedTargetBond >= maxTB)
        return false;
      if (committedTargetAtom < -1 || committedTargetAtom >= targetTopology.numAtoms || committedTargetAtom >= maxTA)
        return false;

      // Lane 0 commits the new mapping into the shared MatchResult.
      // Subsequent bonds in this same call will read the updated
      // visited bitsets after group.sync below.
      if (laneRank == 0) {
        match.targetBondIdx[queryBondIdx] = static_cast<std::uint8_t>(committedTargetBond);
        match.visitedTargetBonds[committedTargetBond / kTargetBondBitsPerWord] |=
          static_cast<TargetBondWord>(1) << (committedTargetBond % kTargetBondBitsPerWord);
        match.matchedBondSize += 1;
        match.empty = false;
        if (committedTargetAtom >= 0) {
          // Atom-adding case: also commit the new atom mapping.
          const int unmappedQueryAtom            = queryUIsMapped ? queryEndpointV : queryEndpointU;
          match.targetAtomIdx[unmappedQueryAtom] = static_cast<std::uint8_t>(committedTargetAtom);
          match.visitedTargetAtoms[committedTargetAtom / kTargetAtomBitsPerWord] |=
            static_cast<TargetAtomWord>(1) << (committedTargetAtom % kTargetAtomBitsPerWord);
          match.matchedAtomSize += 1;
        }
      }
      group.sync();
    }
  }
  return true;
}

template <class TargetTopology>
__device__ __forceinline__ bool findTargetBondBetweenAtomsWithinThread(const int                    targetAtomA,
                                                                       const int                    targetAtomB,
                                                                       const int                    queryBondIdx,
                                                                       const TargetTopology&        targetTopology,
                                                                       const PairMatchTablesDevice& tables,
                                                                       int&                         outTargetBondIdx) {
  if constexpr (topologyHasAdjacencyBondIndices<TargetTopology>()) {
    if (targetAtomA < 0 || targetAtomA >= targetTopology.numAtoms) {
      return false;
    }
    const int begin = static_cast<int>(targetTopology.rowOffsets[targetAtomA]);
    const int end   = static_cast<int>(targetTopology.rowOffsets[targetAtomA + 1]);
    for (int adjIdx = begin; adjIdx < end; ++adjIdx) {
      const int otherTargetAtom = static_cast<int>(targetTopology.colIndices[adjIdx]);
      if (otherTargetAtom != targetAtomB)
        continue;
      const int targetBondIdx = static_cast<int>(targetTopology.bondIndices[adjIdx]);
      if (targetBondIdx < 0 || targetBondIdx >= targetTopology.numBonds) {
        continue;
      }
      if (!tables.bonds.testBit(queryBondIdx, targetBondIdx))
        continue;
      outTargetBondIdx = targetBondIdx;
      return true;
    }
    return false;
  } else {
    const bool scanAdjacency = targetTopology.rowOffsets != nullptr && targetTopology.colIndices != nullptr &&
                               targetTopology.bondIndices != nullptr && targetAtomA >= 0 &&
                               targetAtomA < targetTopology.numAtoms;
    if (scanAdjacency) {
      const int begin = static_cast<int>(targetTopology.rowOffsets[targetAtomA]);
      const int end   = static_cast<int>(targetTopology.rowOffsets[targetAtomA + 1]);
      for (int adjIdx = begin; adjIdx < end; ++adjIdx) {
        const int otherTargetAtom = static_cast<int>(targetTopology.colIndices[adjIdx]);
        if (otherTargetAtom != targetAtomB)
          continue;
        const int targetBondIdx = static_cast<int>(targetTopology.bondIndices[adjIdx]);
        if (targetBondIdx < 0 || targetBondIdx >= targetTopology.numBonds) {
          continue;
        }
        if (!tables.bonds.testBit(queryBondIdx, targetBondIdx))
          continue;
        outTargetBondIdx = targetBondIdx;
        return true;
      }
      return false;
    }

    for (int targetBondIdx = 0; targetBondIdx < targetTopology.numBonds; ++targetBondIdx) {
      const std::uint32_t targetEndpoints = targetTopology.bondEndpoints[targetBondIdx];
      const int           targetEndpointU = static_cast<int>(targetEndpoints >> kBondEndpointShift);
      const int           targetEndpointV = static_cast<int>(targetEndpoints & kBondEndpointMask);
      const bool          endpointsMatch  = (targetEndpointU == targetAtomA && targetEndpointV == targetAtomB) ||
                                  (targetEndpointU == targetAtomB && targetEndpointV == targetAtomA);
      if (!endpointsMatch)
        continue;
      if (!tables.bonds.testBit(queryBondIdx, targetBondIdx))
        continue;
      outTargetBondIdx = targetBondIdx;
      return true;
    }
    return false;
  }
}

template <int maxAtoms, int maxBonds, int maxTA, int maxTB, class QueryTopology, class TargetTopology>
__device__ __forceinline__ bool rebuildMatchFromSubstructureMappingWithinThread(
  const Seed<maxAtoms, maxBonds>&                seed,
  const QueryTopology&                           queryTopology,
  const TargetTopology&                          targetTopology,
  const PairMatchTablesDevice&                   tables,
  MatchResult<maxAtoms, maxBonds, maxTA, maxTB>& match,
  FmcsSubstructureScratch<maxAtoms, maxTA>&      scratch) {
  using SeedT          = Seed<maxAtoms, maxBonds>;
  using MatchT         = MatchResult<maxAtoms, maxBonds, maxTA, maxTB>;
  using BondWord       = typename SeedT::bond_word_type;
  using TargetBondWord = typename MatchT::target_bond_word;
  using TargetAtomWord = typename MatchT::target_atom_word;

  constexpr int kBondBitsPerWord       = SeedT::kBondBitsPerWord;
  constexpr int kBondWords             = SeedT::kBondWords;
  constexpr int kTargetAtomBitsPerWord = MatchT::kTargetAtomBitsPerWord;
  constexpr int kTargetBondBitsPerWord = MatchT::kTargetBondBitsPerWord;

  matchResultClearWithinThread(match);
  for (int i = 0; i < seed.numAtoms; ++i) {
    const int queryAtomIdx  = scratch.seedAtomList[i];
    const int targetAtomIdx = scratch.targetAtomForQuery[queryAtomIdx];
    if (targetAtomIdx == kUnmappedTargetIdx)
      return false;
    match.targetAtomIdx[queryAtomIdx] = static_cast<std::uint8_t>(targetAtomIdx);
    match.visitedTargetAtoms[targetAtomIdx / kTargetAtomBitsPerWord] |= static_cast<TargetAtomWord>(1)
                                                                     << (targetAtomIdx % kTargetAtomBitsPerWord);
  }
  match.matchedAtomSize = seed.numAtoms;

  int matchedBondCount = 0;
  for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
    BondWord remaining = seed.bonds[wordIdx];
    while (remaining != 0) {
      int bitPosInWord;
      if constexpr (sizeof(BondWord) == 4) {
        bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
      } else {
        bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
      }
      const int queryBondIdx = wordIdx * kBondBitsPerWord + bitPosInWord;
      remaining &= remaining - 1;

      const std::uint32_t queryEndpoints  = queryTopology.bondEndpoints[queryBondIdx];
      const int           queryEndpointU  = static_cast<int>(queryEndpoints >> kBondEndpointShift);
      const int           queryEndpointV  = static_cast<int>(queryEndpoints & kBondEndpointMask);
      const int           targetEndpointU = match.targetAtomIdx[queryEndpointU];
      const int           targetEndpointV = match.targetAtomIdx[queryEndpointV];
      if (targetEndpointU == kUnmappedTargetIdx || targetEndpointV == kUnmappedTargetIdx) {
        matchResultClearWithinThread(match);
        return false;
      }

      int targetBondIdx = -1;
      if (!findTargetBondBetweenAtomsWithinThread(targetEndpointU,
                                                  targetEndpointV,
                                                  queryBondIdx,
                                                  targetTopology,
                                                  tables,
                                                  targetBondIdx)) {
        matchResultClearWithinThread(match);
        return false;
      }
      const TargetBondWord visitedWord = match.visitedTargetBonds[targetBondIdx / kTargetBondBitsPerWord];
      if ((visitedWord >> (targetBondIdx % kTargetBondBitsPerWord)) & 1) {
        matchResultClearWithinThread(match);
        return false;
      }
      match.targetBondIdx[queryBondIdx] = static_cast<std::uint8_t>(targetBondIdx);
      match.visitedTargetBonds[targetBondIdx / kTargetBondBitsPerWord] |= static_cast<TargetBondWord>(1)
                                                                       << (targetBondIdx % kTargetBondBitsPerWord);
      ++matchedBondCount;
    }
  }
  match.matchedBondSize = static_cast<std::uint16_t>(matchedBondCount);
  if (matchedBondCount != seed.numBonds) {
    matchResultClearWithinThread(match);
    return false;
  }
  match.empty = false;
  return true;
}

template <int maxAtoms, int maxTA, class TargetTopology, class GroupT>
__device__ __forceinline__ void initializeSeedSubstructureScratchCooperative(
  const GroupT&                             group,
  const TargetTopology&                     targetTopology,
  FmcsSubstructureScratch<maxAtoms, maxTA>& scratch) {
  const int laneRank  = static_cast<int>(group.thread_rank());
  const int laneCount = static_cast<int>(group.num_threads());

  for (int i = laneRank; i < maxAtoms; i += laneCount) {
    scratch.seedDegree[i]         = 0;
    scratch.orderedQueryAtom[i]   = 0;
    scratch.queryOrderPos[i]      = kUnmappedTargetIdx;
    scratch.targetAtomForQuery[i] = kUnmappedTargetIdx;
  }

  if constexpr (topologyHasAdjacencyBondIndices<TargetTopology>()) {
    for (int targetAtomIdx = laneRank; targetAtomIdx < targetTopology.numAtoms; targetAtomIdx += laneCount) {
      scratch.targetDegree[targetAtomIdx] = static_cast<std::uint8_t>(targetTopology.rowOffsets[targetAtomIdx + 1] -
                                                                      targetTopology.rowOffsets[targetAtomIdx]);
    }
  } else {
    if (targetTopology.rowOffsets != nullptr) {
      for (int targetAtomIdx = laneRank; targetAtomIdx < targetTopology.numAtoms; targetAtomIdx += laneCount) {
        scratch.targetDegree[targetAtomIdx] = static_cast<std::uint8_t>(targetTopology.rowOffsets[targetAtomIdx + 1] -
                                                                        targetTopology.rowOffsets[targetAtomIdx]);
      }
    } else {
      for (int i = laneRank; i < maxTA; i += laneCount) {
        scratch.targetDegree[i] = 0;
      }
      group.sync();
      if (laneRank == 0) {
        for (int targetBondIdx = 0; targetBondIdx < targetTopology.numBonds; ++targetBondIdx) {
          const std::uint32_t targetEndpoints = targetTopology.bondEndpoints[targetBondIdx];
          const int           targetEndpointU = static_cast<int>(targetEndpoints >> kBondEndpointShift);
          const int           targetEndpointV = static_cast<int>(targetEndpoints & kBondEndpointMask);
          ++scratch.targetDegree[targetEndpointU];
          ++scratch.targetDegree[targetEndpointV];
        }
      }
    }
  }

  if (laneRank == 0) {
    scratch.currentCount = 0;
    scratch.nextCount    = 0;
    scratch.found        = 0;
    scratch.overflowed   = 0;
  }
  group.sync();
}

template <int maxAtoms, int maxBonds, int maxTA, class QueryTopology, class TargetTopology>
__device__ __forceinline__ bool prepareSeedSubstructureSearchWithinThread(
  const Seed<maxAtoms, maxBonds>&           seed,
  const QueryTopology&                      queryTopology,
  const TargetTopology&                     targetTopology,
  const PairMatchTablesDevice&              tables,
  FmcsSubstructureScratch<maxAtoms, maxTA>& scratch,
  int&                                      numSeedAtoms) {
  using SeedT    = Seed<maxAtoms, maxBonds>;
  using AtomWord = typename SeedT::atom_word_type;
  using BondWord = typename SeedT::bond_word_type;

  constexpr int kAtomBitsPerWord = SeedT::kAtomBitsPerWord;
  constexpr int kAtomWords       = SeedT::kAtomWords;
  constexpr int kBondBitsPerWord = SeedT::kBondBitsPerWord;
  constexpr int kBondWords       = SeedT::kBondWords;

  if (seed.numAtoms > targetTopology.numAtoms || seed.numBonds > targetTopology.numBonds) {
    return false;
  }

  numSeedAtoms = 0;
  for (int wordIdx = 0; wordIdx < kAtomWords; ++wordIdx) {
    AtomWord remaining = seed.atoms[wordIdx];
    while (remaining != 0) {
      int bitPosInWord;
      if constexpr (sizeof(AtomWord) == 4) {
        bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
      } else {
        bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
      }
      const int queryAtomIdx = wordIdx * kAtomBitsPerWord + bitPosInWord;
      remaining &= remaining - 1;
      if (queryAtomIdx < queryTopology.numAtoms) {
        scratch.seedAtomList[numSeedAtoms++] = static_cast<std::uint8_t>(queryAtomIdx);
      }
    }
  }
  if (numSeedAtoms != seed.numAtoms)
    return false;

  for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
    BondWord remaining = seed.bonds[wordIdx];
    while (remaining != 0) {
      int bitPosInWord;
      if constexpr (sizeof(BondWord) == 4) {
        bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
      } else {
        bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
      }
      const int queryBondIdx = wordIdx * kBondBitsPerWord + bitPosInWord;
      remaining &= remaining - 1;

      const std::uint32_t queryEndpoints = queryTopology.bondEndpoints[queryBondIdx];
      const int           queryEndpointU = static_cast<int>(queryEndpoints >> kBondEndpointShift);
      const int           queryEndpointV = static_cast<int>(queryEndpoints & kBondEndpointMask);
      ++scratch.seedDegree[queryEndpointU];
      ++scratch.seedDegree[queryEndpointV];
    }
  }

  for (int orderPos = 0; orderPos < numSeedAtoms; ++orderPos) {
    int bestAtom                = -1;
    int bestMappedNeighborCount = -1;
    int bestDegree              = -1;
    int bestCandidateCount      = maxTA + 1;

    for (int atomListIdx = 0; atomListIdx < numSeedAtoms; ++atomListIdx) {
      const int queryAtomIdx = scratch.seedAtomList[atomListIdx];
      if (scratch.orderedQueryAtom[queryAtomIdx])
        continue;

      int mappedNeighborCount = 0;
      if (orderPos > 0) {
        if constexpr (topologyHasAdjacencyBondIndices<QueryTopology>()) {
          const int begin = static_cast<int>(queryTopology.rowOffsets[queryAtomIdx]);
          const int end   = static_cast<int>(queryTopology.rowOffsets[queryAtomIdx + 1]);
          for (int adjIdx = begin; adjIdx < end; ++adjIdx) {
            const int queryBondIdx = static_cast<int>(queryTopology.bondIndices[adjIdx]);
            if (queryBondIdx >= queryTopology.numBonds ||
                !seedContainsBondWithinThread<maxAtoms, maxBonds>(seed, queryBondIdx)) {
              continue;
            }
            const int otherQueryAtom = static_cast<int>(queryTopology.colIndices[adjIdx]);
            if (otherQueryAtom >= 0 && otherQueryAtom < queryTopology.numAtoms &&
                scratch.orderedQueryAtom[otherQueryAtom]) {
              ++mappedNeighborCount;
            }
          }
        } else {
          const bool scanQueryAdjacency = queryTopology.rowOffsets != nullptr && queryTopology.colIndices != nullptr &&
                                          queryTopology.bondIndices != nullptr && queryAtomIdx >= 0 &&
                                          queryAtomIdx < queryTopology.numAtoms;
          if (scanQueryAdjacency) {
            const int begin = static_cast<int>(queryTopology.rowOffsets[queryAtomIdx]);
            const int end   = static_cast<int>(queryTopology.rowOffsets[queryAtomIdx + 1]);
            for (int adjIdx = begin; adjIdx < end; ++adjIdx) {
              const int queryBondIdx = static_cast<int>(queryTopology.bondIndices[adjIdx]);
              if (queryBondIdx >= queryTopology.numBonds ||
                  !seedContainsBondWithinThread<maxAtoms, maxBonds>(seed, queryBondIdx)) {
                continue;
              }
              const int otherQueryAtom = static_cast<int>(queryTopology.colIndices[adjIdx]);
              if (otherQueryAtom >= 0 && otherQueryAtom < queryTopology.numAtoms &&
                  scratch.orderedQueryAtom[otherQueryAtom]) {
                ++mappedNeighborCount;
              }
            }
          } else {
            for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
              BondWord remaining = seed.bonds[wordIdx];
              while (remaining != 0) {
                int bitPosInWord;
                if constexpr (sizeof(BondWord) == 4) {
                  bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
                } else {
                  bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
                }
                const int queryBondIdx = wordIdx * kBondBitsPerWord + bitPosInWord;
                remaining &= remaining - 1;

                const std::uint32_t queryEndpoints = queryTopology.bondEndpoints[queryBondIdx];
                const int           queryEndpointU = static_cast<int>(queryEndpoints >> kBondEndpointShift);
                const int           queryEndpointV = static_cast<int>(queryEndpoints & kBondEndpointMask);
                if (queryEndpointU == queryAtomIdx && scratch.orderedQueryAtom[queryEndpointV]) {
                  ++mappedNeighborCount;
                } else if (queryEndpointV == queryAtomIdx && scratch.orderedQueryAtom[queryEndpointU]) {
                  ++mappedNeighborCount;
                }
              }
            }
          }
        }
      }
      if (orderPos > 0 && mappedNeighborCount == 0)
        continue;

      int candidateCount = 0;
      for (int targetAtomIdx = 0; targetAtomIdx < targetTopology.numAtoms; ++targetAtomIdx) {
        if (scratch.targetDegree[targetAtomIdx] < scratch.seedDegree[queryAtomIdx]) {
          continue;
        }
        if (!tables.atoms.testBit(queryAtomIdx, targetAtomIdx))
          continue;
        ++candidateCount;
      }
      if (candidateCount == 0)
        return false;

      const int  degree = scratch.seedDegree[queryAtomIdx];
      const bool better = bestAtom < 0 || mappedNeighborCount > bestMappedNeighborCount ||
                          (mappedNeighborCount == bestMappedNeighborCount && degree > bestDegree) ||
                          (mappedNeighborCount == bestMappedNeighborCount && degree == bestDegree &&
                           candidateCount < bestCandidateCount) ||
                          (mappedNeighborCount == bestMappedNeighborCount && degree == bestDegree &&
                           candidateCount == bestCandidateCount && queryAtomIdx < bestAtom);
      if (better) {
        bestAtom                = queryAtomIdx;
        bestMappedNeighborCount = mappedNeighborCount;
        bestDegree              = degree;
        bestCandidateCount      = candidateCount;
      }
    }

    if (bestAtom < 0) {
      for (int atomListIdx = 0; atomListIdx < numSeedAtoms; ++atomListIdx) {
        const int queryAtomIdx = scratch.seedAtomList[atomListIdx];
        if (!scratch.orderedQueryAtom[queryAtomIdx]) {
          bestAtom = queryAtomIdx;
          break;
        }
      }
    }
    if (bestAtom < 0)
      return false;
    scratch.seedAtoms[orderPos]        = static_cast<std::uint8_t>(bestAtom);
    scratch.orderedQueryAtom[bestAtom] = 1;
    scratch.queryOrderPos[bestAtom]    = static_cast<std::uint8_t>(orderPos);
  }
  return true;
}

__device__ __forceinline__ bool partialUsesTargetAtomWithinThread(const std::uint8_t* partial,
                                                                  const int           depth,
                                                                  const int           targetAtomIdx) {
  for (int i = 0; i < depth; ++i) {
    if (partial[i] == targetAtomIdx)
      return true;
  }
  return false;
}

template <int maxAtoms, int maxBonds, int maxTA, class QueryTopology>
__device__ __forceinline__ bool findMappedQueryNeighborWithinThread(
  const Seed<maxAtoms, maxBonds>&                 seed,
  const QueryTopology&                            queryTopology,
  const FmcsSubstructureScratch<maxAtoms, maxTA>& scratch,
  const int                                       depth,
  const int                                       queryAtomIdx,
  int&                                            outNeighborOrderPos) {
  using SeedT                    = Seed<maxAtoms, maxBonds>;
  using BondWord                 = typename SeedT::bond_word_type;
  constexpr int kBondBitsPerWord = SeedT::kBondBitsPerWord;
  constexpr int kBondWords       = SeedT::kBondWords;

  outNeighborOrderPos = -1;
  if constexpr (topologyHasAdjacencyBondIndices<QueryTopology>()) {
    const int begin = static_cast<int>(queryTopology.rowOffsets[queryAtomIdx]);
    const int end   = static_cast<int>(queryTopology.rowOffsets[queryAtomIdx + 1]);
    for (int adjIdx = begin; adjIdx < end; ++adjIdx) {
      const int queryBondIdx = static_cast<int>(queryTopology.bondIndices[adjIdx]);
      if (queryBondIdx >= queryTopology.numBonds ||
          !seedContainsBondWithinThread<maxAtoms, maxBonds>(seed, queryBondIdx)) {
        continue;
      }
      const int otherQueryAtom = static_cast<int>(queryTopology.colIndices[adjIdx]);
      if (otherQueryAtom < 0 || otherQueryAtom >= queryTopology.numAtoms) {
        continue;
      }
      const int otherOrderPos = scratch.queryOrderPos[otherQueryAtom];
      if (otherOrderPos != kUnmappedTargetIdx && otherOrderPos < depth) {
        outNeighborOrderPos = otherOrderPos;
        return true;
      }
    }
    return false;
  } else {
    const bool scanQueryAdjacency = queryTopology.rowOffsets != nullptr && queryTopology.colIndices != nullptr &&
                                    queryTopology.bondIndices != nullptr && queryAtomIdx >= 0 &&
                                    queryAtomIdx < queryTopology.numAtoms;
    if (scanQueryAdjacency) {
      const int begin = static_cast<int>(queryTopology.rowOffsets[queryAtomIdx]);
      const int end   = static_cast<int>(queryTopology.rowOffsets[queryAtomIdx + 1]);
      for (int adjIdx = begin; adjIdx < end; ++adjIdx) {
        const int queryBondIdx = static_cast<int>(queryTopology.bondIndices[adjIdx]);
        if (queryBondIdx >= queryTopology.numBonds ||
            !seedContainsBondWithinThread<maxAtoms, maxBonds>(seed, queryBondIdx)) {
          continue;
        }
        const int otherQueryAtom = static_cast<int>(queryTopology.colIndices[adjIdx]);
        if (otherQueryAtom < 0 || otherQueryAtom >= queryTopology.numAtoms) {
          continue;
        }
        const int otherOrderPos = scratch.queryOrderPos[otherQueryAtom];
        if (otherOrderPos != kUnmappedTargetIdx && otherOrderPos < depth) {
          outNeighborOrderPos = otherOrderPos;
          return true;
        }
      }
      return false;
    }

    for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
      BondWord remaining = seed.bonds[wordIdx];
      while (remaining != 0) {
        int bitPosInWord;
        if constexpr (sizeof(BondWord) == 4) {
          bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
        } else {
          bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
        }
        const int queryBondIdx = wordIdx * kBondBitsPerWord + bitPosInWord;
        remaining &= remaining - 1;

        const std::uint32_t queryEndpoints = queryTopology.bondEndpoints[queryBondIdx];
        const int           queryEndpointU = static_cast<int>(queryEndpoints >> kBondEndpointShift);
        const int           queryEndpointV = static_cast<int>(queryEndpoints & kBondEndpointMask);
        int                 otherQueryAtom = -1;
        if (queryEndpointU == queryAtomIdx) {
          otherQueryAtom = queryEndpointV;
        } else if (queryEndpointV == queryAtomIdx) {
          otherQueryAtom = queryEndpointU;
        } else {
          continue;
        }

        const int otherOrderPos = scratch.queryOrderPos[otherQueryAtom];
        if (otherOrderPos != kUnmappedTargetIdx && otherOrderPos < depth) {
          outNeighborOrderPos = otherOrderPos;
          return true;
        }
      }
    }
    return false;
  }
}

template <int maxAtoms, int maxBonds, int maxTA, class QueryTopology, class TargetTopology>
__device__ __forceinline__ bool substructurePartialEdgeConsistentWithinThread(
  const Seed<maxAtoms, maxBonds>&                 seed,
  const QueryTopology&                            queryTopology,
  const TargetTopology&                           targetTopology,
  const PairMatchTablesDevice&                    tables,
  const FmcsSubstructureScratch<maxAtoms, maxTA>& scratch,
  const std::uint8_t*                             partial,
  const int                                       depth,
  const int                                       queryAtomIdx,
  const int                                       targetAtomIdx) {
  using SeedT                    = Seed<maxAtoms, maxBonds>;
  using BondWord                 = typename SeedT::bond_word_type;
  constexpr int kBondBitsPerWord = SeedT::kBondBitsPerWord;
  constexpr int kBondWords       = SeedT::kBondWords;

  if constexpr (topologyHasAdjacencyBondIndices<QueryTopology>()) {
    const int begin = static_cast<int>(queryTopology.rowOffsets[queryAtomIdx]);
    const int end   = static_cast<int>(queryTopology.rowOffsets[queryAtomIdx + 1]);
    for (int adjIdx = begin; adjIdx < end; ++adjIdx) {
      const int queryBondIdx = static_cast<int>(queryTopology.bondIndices[adjIdx]);
      if (queryBondIdx >= queryTopology.numBonds ||
          !seedContainsBondWithinThread<maxAtoms, maxBonds>(seed, queryBondIdx)) {
        continue;
      }
      const int otherQueryAtom = static_cast<int>(queryTopology.colIndices[adjIdx]);
      if (otherQueryAtom < 0 || otherQueryAtom >= queryTopology.numAtoms) {
        continue;
      }

      const int otherOrderPos = scratch.queryOrderPos[otherQueryAtom];
      if (otherOrderPos == kUnmappedTargetIdx || otherOrderPos >= depth) {
        continue;
      }
      const int otherTargetAtom = partial[otherOrderPos];

      int targetBondIdx = -1;
      if (!findTargetBondBetweenAtomsWithinThread(targetAtomIdx,
                                                  otherTargetAtom,
                                                  queryBondIdx,
                                                  targetTopology,
                                                  tables,
                                                  targetBondIdx)) {
        return false;
      }
    }
    return true;
  } else {
    const bool scanQueryAdjacency = queryTopology.rowOffsets != nullptr && queryTopology.colIndices != nullptr &&
                                    queryTopology.bondIndices != nullptr && queryAtomIdx >= 0 &&
                                    queryAtomIdx < queryTopology.numAtoms;
    if (scanQueryAdjacency) {
      const int begin = static_cast<int>(queryTopology.rowOffsets[queryAtomIdx]);
      const int end   = static_cast<int>(queryTopology.rowOffsets[queryAtomIdx + 1]);
      for (int adjIdx = begin; adjIdx < end; ++adjIdx) {
        const int queryBondIdx = static_cast<int>(queryTopology.bondIndices[adjIdx]);
        if (queryBondIdx >= queryTopology.numBonds ||
            !seedContainsBondWithinThread<maxAtoms, maxBonds>(seed, queryBondIdx)) {
          continue;
        }
        const int otherQueryAtom = static_cast<int>(queryTopology.colIndices[adjIdx]);
        if (otherQueryAtom < 0 || otherQueryAtom >= queryTopology.numAtoms) {
          continue;
        }

        const int otherOrderPos = scratch.queryOrderPos[otherQueryAtom];
        if (otherOrderPos == kUnmappedTargetIdx || otherOrderPos >= depth) {
          continue;
        }
        const int otherTargetAtom = partial[otherOrderPos];

        int targetBondIdx = -1;
        if (!findTargetBondBetweenAtomsWithinThread(targetAtomIdx,
                                                    otherTargetAtom,
                                                    queryBondIdx,
                                                    targetTopology,
                                                    tables,
                                                    targetBondIdx)) {
          return false;
        }
      }
      return true;
    }

    for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
      BondWord remaining = seed.bonds[wordIdx];
      while (remaining != 0) {
        int bitPosInWord;
        if constexpr (sizeof(BondWord) == 4) {
          bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
        } else {
          bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
        }
        const int queryBondIdx = wordIdx * kBondBitsPerWord + bitPosInWord;
        remaining &= remaining - 1;

        const std::uint32_t queryEndpoints = queryTopology.bondEndpoints[queryBondIdx];
        const int           queryEndpointU = static_cast<int>(queryEndpoints >> kBondEndpointShift);
        const int           queryEndpointV = static_cast<int>(queryEndpoints & kBondEndpointMask);
        int                 otherQueryAtom = -1;
        if (queryEndpointU == queryAtomIdx) {
          otherQueryAtom = queryEndpointV;
        } else if (queryEndpointV == queryAtomIdx) {
          otherQueryAtom = queryEndpointU;
        } else {
          continue;
        }

        const int otherOrderPos = scratch.queryOrderPos[otherQueryAtom];
        if (otherOrderPos == kUnmappedTargetIdx || otherOrderPos >= depth) {
          continue;
        }
        const int otherTargetAtom = partial[otherOrderPos];

        int targetBondIdx = -1;
        if (!findTargetBondBetweenAtomsWithinThread(targetAtomIdx,
                                                    otherTargetAtom,
                                                    queryBondIdx,
                                                    targetTopology,
                                                    tables,
                                                    targetBondIdx)) {
          return false;
        }
      }
    }
    return true;
  }
}

template <int maxAtoms, int maxBonds, int maxTA, int maxTB, class QueryTopology, class TargetTopology, class GroupT>
__device__ __forceinline__ bool matchSeedSubstructureCooperative(const GroupT&                   group,
                                                                 const Seed<maxAtoms, maxBonds>& seed,
                                                                 const QueryTopology&            queryTopology,
                                                                 const TargetTopology&           targetTopology,
                                                                 const PairMatchTablesDevice&    tables,
                                                                 MatchResult<maxAtoms, maxBonds, maxTA, maxTB>& match,
                                                                 FmcsSubstructureScratch<maxAtoms, maxTA>&      scratch,
                                                                 std::uint8_t* partialStorage,
                                                                 int           partialCapacity,
                                                                 bool*         overflowedFlag) {
  const int laneRank  = static_cast<int>(group.thread_rank());
  const int laneCount = static_cast<int>(group.num_threads());

  int numSeedAtoms = 0;
  int prepared     = 0;
  if (laneRank == 0) {
    matchResultClearWithinThread(match);
  }
  if (seed.numAtoms != 0) {
    initializeSeedSubstructureScratchCooperative(group, targetTopology, scratch);
  }
  if (laneRank == 0) {
    if (seed.numAtoms == 0) {
      prepared = (seed.numBonds == 0) ? 2 : -1;
    } else {
      prepared =
        prepareSeedSubstructureSearchWithinThread(seed, queryTopology, targetTopology, tables, scratch, numSeedAtoms) ?
          1 :
          -1;
    }
    if ((partialStorage == nullptr || partialCapacity <= 0) && prepared == 1) {
      scratch.overflowed = 1;
      prepared           = -1;
    }
  }
  group.sync();
  prepared     = group.shfl(prepared, 0);
  numSeedAtoms = group.shfl(numSeedAtoms, 0);
  if (prepared == 2)
    return true;
  if (prepared < 0) {
    if (laneRank == 0 && scratch.overflowed && overflowedFlag != nullptr) {
      *overflowedFlag = true;
    }
    return false;
  }

  const int     stride            = numSeedAtoms;
  const int     halfBytes         = partialCapacity * maxAtoms;
  const int     effectiveCapacity = halfBytes / stride;
  std::uint8_t* currentPartials   = partialStorage;
  std::uint8_t* nextPartials      = partialStorage + halfBytes;

  const int firstQueryAtom = scratch.seedAtoms[0];
  for (int targetAtomIdx = laneRank; targetAtomIdx < targetTopology.numAtoms; targetAtomIdx += laneCount) {
    if (scratch.targetDegree[targetAtomIdx] < scratch.seedDegree[firstQueryAtom]) {
      continue;
    }
    if (!tables.atoms.testBit(firstQueryAtom, targetAtomIdx))
      continue;

    const int slot = atomicAdd(&scratch.currentCount, 1);
    if (slot < effectiveCapacity) {
      currentPartials[slot * stride] = static_cast<std::uint8_t>(targetAtomIdx);
    } else {
      atomicExch(&scratch.overflowed, 1);
    }
  }
  group.sync();

  if (numSeedAtoms == 1) {
    if (scratch.currentCount > 0) {
      if (laneRank == 0) {
        scratch.targetAtomForQuery[firstQueryAtom] = currentPartials[0];
        prepared =
          rebuildMatchFromSubstructureMappingWithinThread(seed, queryTopology, targetTopology, tables, match, scratch) ?
            1 :
            -1;
      }
      group.sync();
      prepared = group.shfl(prepared, 0);
      return prepared == 1;
    }
    return false;
  }

  for (int depth = 1; depth < numSeedAtoms; ++depth) {
    if (scratch.currentCount == 0 || scratch.found != 0)
      break;
    if (laneRank == 0)
      scratch.nextCount = 0;
    group.sync();

    const int queryAtomIdx = scratch.seedAtoms[depth];
    const int numPartials  = scratch.currentCount < effectiveCapacity ? scratch.currentCount : effectiveCapacity;

    for (int partialIdx = 0; partialIdx < numPartials && scratch.found == 0; ++partialIdx) {
      const std::uint8_t* partial          = currentPartials + partialIdx * stride;
      int                 neighborOrderPos = -1;
      const bool          hasMappedNeighbor =
        findMappedQueryNeighborWithinThread(seed, queryTopology, scratch, depth, queryAtomIdx, neighborOrderPos);
      const bool scanAdjacency =
        hasMappedNeighbor && targetTopology.rowOffsets != nullptr && targetTopology.colIndices != nullptr;
      const int targetScanBegin =
        scanAdjacency ? static_cast<int>(targetTopology.rowOffsets[partial[neighborOrderPos]]) : 0;
      const int targetScanEnd = scanAdjacency ?
                                  static_cast<int>(targetTopology.rowOffsets[partial[neighborOrderPos] + 1]) :
                                  targetTopology.numAtoms;

      for (int targetScanIdx = targetScanBegin + laneRank; targetScanIdx < targetScanEnd && scratch.found == 0;
           targetScanIdx += laneCount) {
        const int targetAtomIdx =
          scanAdjacency ? static_cast<int>(targetTopology.colIndices[targetScanIdx]) : targetScanIdx;
        if (scratch.targetDegree[targetAtomIdx] < scratch.seedDegree[queryAtomIdx]) {
          continue;
        }
        if (!tables.atoms.testBit(queryAtomIdx, targetAtomIdx))
          continue;
        if (partialUsesTargetAtomWithinThread(partial, depth, targetAtomIdx)) {
          continue;
        }
        if (!substructurePartialEdgeConsistentWithinThread(seed,
                                                           queryTopology,
                                                           targetTopology,
                                                           tables,
                                                           scratch,
                                                           partial,
                                                           depth,
                                                           queryAtomIdx,
                                                           targetAtomIdx)) {
          continue;
        }

        if (depth == numSeedAtoms - 1) {
          if (atomicCAS(&scratch.found, 0, 1) == 0) {
            for (int orderPos = 0; orderPos < depth; ++orderPos) {
              const int mappedQueryAtom                   = scratch.seedAtoms[orderPos];
              scratch.targetAtomForQuery[mappedQueryAtom] = partial[orderPos];
            }
            scratch.targetAtomForQuery[queryAtomIdx] = static_cast<std::uint8_t>(targetAtomIdx);
          }
        } else {
          const int slot = atomicAdd(&scratch.nextCount, 1);
          if (slot < effectiveCapacity) {
            std::uint8_t* next = nextPartials + slot * stride;
            for (int orderPos = 0; orderPos < depth; ++orderPos) {
              next[orderPos] = partial[orderPos];
            }
            next[depth] = static_cast<std::uint8_t>(targetAtomIdx);
          } else {
            atomicExch(&scratch.overflowed, 1);
          }
        }
      }
    }
    group.sync();

    if (scratch.found != 0)
      break;
    if (laneRank == 0)
      scratch.currentCount = scratch.nextCount;
    std::uint8_t* tmp = currentPartials;
    currentPartials   = nextPartials;
    nextPartials      = tmp;
    group.sync();
  }

  int found = scratch.found;
  if (found != 0) {
    if (laneRank == 0) {
      found =
        rebuildMatchFromSubstructureMappingWithinThread(seed, queryTopology, targetTopology, tables, match, scratch) ?
          1 :
          0;
    }
    group.sync();
    found = group.shfl(found, 0);
    return found != 0;
  }

  if (laneRank == 0) {
    matchResultClearWithinThread(match);
    if (scratch.overflowed && overflowedFlag != nullptr) {
      *overflowedFlag = true;
    }
  }
  group.sync();
  return false;
}

template <int maxAtoms, int maxBonds, int maxTA, int maxTB, class QueryTopology, class TargetTopology, class GroupT>
__device__ __forceinline__ bool matchSeedWithSubstructureFallbackCooperative(
  const GroupT&                                  group,
  const Seed<maxAtoms, maxBonds>&                seed,
  const QueryTopology&                           queryTopology,
  const TargetTopology&                          targetTopology,
  const PairMatchTablesDevice&                   tables,
  MatchResult<maxAtoms, maxBonds, maxTA, maxTB>& match,
  FmcsSubstructureScratch<maxAtoms, maxTA>&      scratch,
  int*                                           scratchLock,
  std::uint8_t*                                  partialStorage,
  int                                            partialCapacity,
  bool*                                          overflowedFlag) {
  if (matchIncrementalFastCooperative(group, seed, queryTopology, targetTopology, tables, match)) {
    return true;
  }
  if (group.thread_rank() == 0) {
    while (atomicCAS(scratchLock, 0, 1) != 0) {
    }
  }
  group.sync();
  const bool ok = matchSeedSubstructureCooperative(group,
                                                   seed,
                                                   queryTopology,
                                                   targetTopology,
                                                   tables,
                                                   match,
                                                   scratch,
                                                   partialStorage,
                                                   partialCapacity,
                                                   overflowedFlag);
  group.sync();
  if (group.thread_rank() == 0) {
    atomicExch(scratchLock, 0);
  }
  group.sync();
  return ok;
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_MATCH_CUH
