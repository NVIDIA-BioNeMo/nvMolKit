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

namespace mcs {
namespace fmcs {

/// Bond endpoints are stored across the kernel as a single uint32 with
/// the u-endpoint atom index in the high 16 bits and the v-endpoint
/// atom index in the low 16 bits.  Tiers cap maxAtoms at 128 so 16 bits
/// per index is plenty.
constexpr int           kBondEndpointShift = 16;
constexpr std::uint32_t kBondEndpointMask  = 0xFFFFu;

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
  // Scratch for the RDKit checkIfMatchAndAppend fallback.  This is deliberately
  // caller-owned shared memory, not function-local state: tier-128 scratch is
  // too large to risk compiler-created stack/local memory in the matcher.
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

template <class Topology> __host__ __device__ constexpr bool topologyHasAdjacencyBondIndices() {
  if constexpr (requires { Topology::kHasAdjacencyBondIndices; }) {
    return Topology::kHasAdjacencyBondIndices;
  } else {
    return false;
  }
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
///
/// @p queryTopology and @p targetTopology must expose a
/// @c bondEndpoints array of packed (u<<16 | v) entries.
template <class QueryTopology, class TargetTopology>
__device__ __forceinline__ bool matchSingleBondWithinThread(const int                    queryBondIdx,
                                                            const int                    targetBondIdx,
                                                            const bool                   reversed,
                                                            const QueryTopology&         queryTopology,
                                                            const TargetTopology&        targetTopology,
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

/// Cooperative: extend @p match by every query bond in @p seed.bonds
/// whose @c match.targetBondIdx[q] is still @ref kUnmappedTargetIdx
/// (i.e., unmapped by the parent's recorded embedding).  For each such
/// bond:
///   - Both endpoints already mapped -> ring-closing case.  The lanes
///     of @p group scan target bonds in parallel for one whose endpoint
///     pair exactly matches the mapped (queryU, queryV) target atoms
///     and is unvisited, with the bond-match-table bit set.  First
///     compatible target bond commits.
///   - Exactly one endpoint mapped -> atom-adding case.  Lane-parallel
///     scan for a target bond incident to the mapped target atom whose
///     other end is unvisited, atom-table-compatible with the unmapped
///     query atom, and bond-table-compatible.  First compatible
///     candidate commits both the new bond mapping and the new atom
///     mapping, and marks both visited.
///   - Both endpoints unmapped -> defensive fail (shouldn't occur on
///     well-formed seeds, where Phase 1 maps both initial atoms before
///     pushing).
/// Any bond that fails to extend causes the function to return false;
/// @p match is left in an unspecified state and the caller should
/// discard the seed.
template <int maxAtoms, int maxBonds, int maxTA, int maxTB, class QueryTopology, class TargetTopology, class GroupT>
__device__ __forceinline__ bool matchIncrementalFastCooperative(const GroupT&                   group,
                                                                const Seed<maxAtoms, maxBonds>& seed,
                                                                const QueryTopology&            queryTopology,
                                                                const TargetTopology&           targetTopology,
                                                                const PairMatchTablesDevice&    tables,
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
        if constexpr (topologyHasAdjacencyBondIndices<TargetTopology>()) {
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
          const bool scanAdjacency = targetTopology.rowOffsets != nullptr && targetTopology.colIndices != nullptr &&
                                     targetTopology.bondIndices != nullptr && srcTargetAtom >= 0 &&
                                     srcTargetAtom < targetTopology.numAtoms;
          if (scanAdjacency) {
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
          } else {
            for (int targetBondIdx = laneRank; targetBondIdx < targetTopology.numBonds && chosenTargetBond < 0;
                 targetBondIdx += laneCount) {
              // Skip target bonds already used by the parent's match.
              const TargetBondWord visitedBondsWord = match.visitedTargetBonds[targetBondIdx / kTargetBondBitsPerWord];
              if ((visitedBondsWord >> (targetBondIdx % kTargetBondBitsPerWord)) & 1) {
                continue;
              }
              const std::uint32_t targetEndpoints = targetTopology.bondEndpoints[targetBondIdx];
              const int           targetEndpointU = static_cast<int>(targetEndpoints >> kBondEndpointShift);
              const int           targetEndpointV = static_cast<int>(targetEndpoints & kBondEndpointMask);
              // Match either orientation -- target bonds are undirected.
              const bool endpointsMatch = (targetEndpointU == srcTargetAtom && targetEndpointV == dstTargetAtom) ||
                                          (targetEndpointU == dstTargetAtom && targetEndpointV == srcTargetAtom);
              if (!endpointsMatch)
                continue;
              if (!tables.bonds.testBit(queryBondIdx, targetBondIdx))
                continue;
              chosenTargetBond = targetBondIdx;
            }
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
        if constexpr (topologyHasAdjacencyBondIndices<TargetTopology>()) {
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
        } else {
          const bool scanAdjacency = targetTopology.rowOffsets != nullptr && targetTopology.colIndices != nullptr &&
                                     targetTopology.bondIndices != nullptr && srcTargetAtom >= 0 &&
                                     srcTargetAtom < targetTopology.numAtoms;
          if (scanAdjacency) {
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
          } else {
            for (int targetBondIdx = laneRank; targetBondIdx < targetTopology.numBonds && chosenTargetBond < 0;
                 targetBondIdx += laneCount) {
              const TargetBondWord visitedBondsWord = match.visitedTargetBonds[targetBondIdx / kTargetBondBitsPerWord];
              if ((visitedBondsWord >> (targetBondIdx % kTargetBondBitsPerWord)) & 1) {
                continue;
              }
              const std::uint32_t targetEndpoints = targetTopology.bondEndpoints[targetBondIdx];
              const int           targetEndpointU = static_cast<int>(targetEndpoints >> kBondEndpointShift);
              const int           targetEndpointV = static_cast<int>(targetEndpoints & kBondEndpointMask);
              // Identify the candidate target atom on the far side of the
              // bond from srcTargetAtom; skip bonds not incident to it.
              int                 candidateTargetAtom;
              if (targetEndpointU == srcTargetAtom) {
                candidateTargetAtom = targetEndpointV;
              } else if (targetEndpointV == srcTargetAtom) {
                candidateTargetAtom = targetEndpointU;
              } else {
                continue;
              }
              // Candidate target atom must not already be in the embedding.
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

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_MATCH_CUH
