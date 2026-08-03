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

// Warp-per-pair depth-first substructure search.
//
// Given a pair's label matrix -- bit (t, q) set iff target atom t satisfies
// query atom q's atom-level predicate, computed by populateLabelMatrix -- find
// the injective maps from query atoms to target atoms that also satisfy the
// query's bond constraints, which the label matrix does not cover.
//
// The search itself is nvMolKit::dfsFromRoots (src/subgraph/warp_dfs.cuh);
// this header is the substructure frontend: it builds the per-pair lookup
// tables the candidates oracle reads, supplies that oracle, and implements the
// output modes as terminal handlers. Query atoms are matched in index order,
// so depth d means "choose the target atom for query atom d".
//
// One warp per pair, kWarpsPerBlock pairs per CTA. Lane L searches the subtrees
// rooted at target atoms L, L+32, L+64, L+96; those subtrees are disjoint, so
// all search state is lane-local registers and lanes only meet at the final
// reduction. Shared memory holds the per-warp lookup tables and result counters.
//
// Three tables are built per pair before the search, each depending on the last:
// tryBuildQueryAtomCandidates (the label-compatibility term of the oracle's
// intersection), analyzeQueryBonds (the back edges), buildTargetAdjacency (the
// per-edge neighbour sets).
//
// Future optimization candidates:
// - Store target adjacency as the two packed degree-8 words at construction,
//   deleting packAdjacencyRow.
// - Produce the label matrix query-major for this backend, deleting the ballot
//   transpose in tryBuildQueryAtomCandidates.
// - Have the label kernel flag pairs with an empty column and compact the DFS
//   pair list on device.
// - Precompute analyzeQueryBonds per query at host pack time (see its note).
// - Reorder query atoms most-constrained-first at construction, with an inverse
//   permutation on mapping download.
// - Route size-1 queries to a popcount-over-label-column kernel.
// - Drop query>target pairs in the minibatch planner.

#ifndef NVMOLKIT_SUBSTRUCT_DFS_CUH
#define NVMOLKIT_SUBSTRUCT_DFS_CUH

#include <cstddef>
#include <cstdint>

#include "src/subgraph/occupancy.cuh"
#include "src/subgraph/target_mask.cuh"
#include "src/subgraph/warp_dfs.cuh"
#include "src/subgraph/warp_reduce.cuh"
#include "src/substruct/molecules_device.cuh"
#include "src/substruct/packed_bonds_device.cuh"
#include "src/substruct/substruct_algos.cuh"

namespace nvMolKit {
namespace dfs {

/// Warps, and therefore independent pairs, per CTA.
constexpr int kWarpsPerBlock = 8;
constexpr int kBlockSize     = kWarpsPerBlock * 32;

/// How a warp records the embeddings it finds.
enum class DfsOutputMode {
  Count,  ///< Per-pair embedding count only; no mapping storage, no per-match atomics.
  Store,  ///< Full mappings written to the match index buffer.
  Paint   ///< Recursive SMARTS bit painted for mapping[0]; stops each root at its first embedding.
};

// =============================================================================
// Query bond classification
// =============================================================================

// Entry layout in WarpSharedState::backEdges. A back edge of query atom d is an
// edge to a query atom with a smaller index, i.e. one already mapped at depth d.
constexpr unsigned char kBackEdgeAtomMask    = 63u;  ///< Low 6 bits: the earlier query atom's index.
constexpr unsigned char kBackEdgeAltMaskFlag = 64u;  ///< Bit 6 (Dual only): use the Alt adjacency table.

/**
 * @brief How many distinct bond masks (QueryAtomBondsT::matchMask) the query's
 *        back edges carry.
 *
 * With at most two, the target neighbour sets those masks induce can be
 * precomputed once per pair, turning the per-edge work in the inner loop into a
 * table lookup.
 */
enum class BondMaskCase {
  Uniform,  ///< One mask. One adjacency table suffices.
  Dual,     ///< Two masks. Two adjacency tables; each back edge flags which it uses.
  General   ///< Three or more. Neighbour sets recomputed per edge during the search.
};

/// Classification of the query's bond constraints; identical in every lane.
struct QueryBondPlan {
  BondMaskCase maskCase       = BondMaskCase::General;
  uint32_t     minBondMask    = 0;  ///< Smallest bond mask on any back edge.
  uint32_t     maxBondMask    = 0;  ///< Largest bond mask on any back edge; equals minBondMask iff Uniform.
  /// Bitset over query atoms: bit d set iff depth d's only back edge goes to
  /// depth d-1 and resolves through targetAdjacency. Such a depth needs no
  /// back-edge lookup, since the caller just chose depth d-1's target atom.
  uint64_t     chainDepths    = 0;
  /// As chainDepths, for depths resolving through targetAdjacencyAlt.
  uint64_t     chainDepthsAlt = 0;
};

// =============================================================================
// Per-warp shared state
// =============================================================================

/**
 * @brief Per-warp lookup tables and counters, sized for a whole CTA.
 *
 * Warp w touches only [w][...], so no synchronisation beyond __syncwarp() is
 * needed. This is the kernel's only __shared__ allocation, so its size alone
 * determines how many CTAs fit per SM (minBlocksPerSM).
 *
 * The adjacency tables hold different things per BondMaskCase; buildTargetAdjacency
 * is the only place that fills them:
 *   Uniform: targetAdjacency[a] = atoms bonded to a over a bond the query's
 *            single mask accepts. Alt unused.
 *   Dual:    targetAdjacency[a] for minBondMask, targetAdjacencyAlt[a] for
 *            maxBondMask; each back edge picks one via kBackEdgeAltMaskFlag.
 *   General: nothing is precomputable, so the tables cache the packed adjacency
 *            row instead -- targetAdjacency[a].lo the neighbour-index bytes,
 *            targetAdjacencyAlt[a].lo the bond-info bytes.
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms> struct WarpSharedState {
  static_assert(MaxQueryAtoms <= 64, "Back-edge indices pack into 6 bits and chain depths into a 64-bit mask");

  /// Roots each lane owns: lane, lane + 32, ... below MaxTargetAtoms.
  static constexpr int kRoots = static_cast<int>(MaxTargetAtoms / 32);
  using Mask                  = TargetMask<MaxTargetAtoms>;

  /**
   * @brief One adjacency table slot, read as whichever member the case stores.
   *
   * A union because the General case caches packed adjacency rows, which are
   * always 64 bits wide, whereas a mask is only as wide as MaxTargetAtoms. This
   * keeps the table exactly as large as the wider of the two, so narrowing Mask
   * costs no shared memory here while still narrowing it in registers.
   */
  union AdjacencyEntry {
    Mask     mask;
    uint64_t packedRow;
  };

  AdjacencyEntry targetAdjacency[kWarpsPerBlock][MaxTargetAtoms];
  AdjacencyEntry targetAdjacencyAlt[kWarpsPerBlock][MaxTargetAtoms];
  /// Transposed label matrix: the target atoms compatible with each query atom.
  Mask           queryAtomCandidates[kWarpsPerBlock][MaxQueryAtoms];
  /// Per query atom, its back edges, kBackEdge*-encoded.
  unsigned char  backEdges[kWarpsPerBlock][MaxQueryAtoms][kMaxBondsPerAtom];
  unsigned char  backEdgeCounts[kWarpsPerBlock][MaxQueryAtoms];  ///< Valid entries in backEdges[w][q].
  int            matchCount[kWarpsPerBlock];                     ///< Embeddings found for the warp's pair.
  int            reportedCount[kWarpsPerBlock];                  ///< Of those, how many fit in the output buffer.
};

/// Second __launch_bounds__ argument for the DFS kernels: eight resident CTAs
/// is the occupancy the search is tuned at, clamped to what WarpSharedState --
/// the kernels' only shared allocation -- and the SM thread limit allow (see
/// src/subgraph/occupancy.cuh). The 128-atom target specialisations cannot
/// reach eight at any register count -- their adjacency tables alone are too
/// large -- so they ask for what fits.
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms> constexpr int minBlocksPerSM() {
  return nvMolKit::minBlocksPerSM<sizeof(WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>), kBlockSize, 8>();
}

// =============================================================================
// Packed adjacency helpers
// =============================================================================

/**
 * @brief Pack one target atom's adjacency into two words, one byte per bond slot.
 *
 * Byte j of @p neighbors is the j-th bonded atom's index, byte j of @p bondInfo
 * its packed (bondType, isInRing) descriptor. Consumers then walk the bonds by
 * shifting two registers instead of re-reading the 20-byte TargetAtomBonds.
 * Slots past the atom's degree hold kNoNeighbor, which ends the walk.
 */
__device__ __forceinline__ void packAdjacencyRow(const TargetAtomBonds& bonds,
                                                 uint64_t&              neighbors,
                                                 uint64_t&              bondInfo) {
  neighbors     = ~0ULL;
  bondInfo      = 0;
  const int deg = bonds.degree;
#pragma unroll
  for (int j = 0; j < kMaxBondsPerAtom; ++j) {
    if (j >= deg) {
      break;
    }
    const int shift = 8 * j;
    neighbors &= ~(0xFFULL << shift);
    neighbors |= static_cast<uint64_t>(bonds.neighborIdx[j]) << shift;
    bondInfo |= static_cast<uint64_t>(bonds.bondInfo[j]) << shift;
  }
}

/**
 * @brief Neighbours of one target atom reachable over a bond @p queryBondMask accepts.
 *
 * @p neighbors and @p bondInfo are that atom's packed row from packAdjacencyRow.
 * A query bond mask holds one bit per (isInRing, bondType) at position
 * `isInRing * 16 + bondType`; see QueryAtomBondsT.
 */
template <std::size_t MaxAtoms>
__device__ __forceinline__ TargetMask<MaxAtoms> neighborsMatchingBondMask(uint64_t neighbors,
                                                                          uint64_t bondInfo,
                                                                          uint32_t queryBondMask) {
  TargetMask<MaxAtoms> mask;
  mask.clear();
#pragma unroll 1
  for (int k = 0; k < kMaxBondsPerAtom; ++k) {
    const uint32_t neighbor = static_cast<uint32_t>(neighbors & 0xFFu);
    if (neighbor == kNoNeighbor) {
      break;
    }
    const uint32_t info = static_cast<uint32_t>(bondInfo & 0xFFu);
    const uint32_t code = ((info >> 4) & 1u) * 16u + (info & 15u);
    mask.setIf(static_cast<int>(neighbor), static_cast<uint64_t>((queryBondMask >> code) & 1u));
    neighbors >>= 8;
    bondInfo >>= 8;
  }
  return mask;
}

// =============================================================================
// Candidate construction
// =============================================================================

/**
 * @brief The target atoms query atom @p depth may still map to: the candidates
 *        oracle handed to nvMolKit::dfsFromRoots.
 *
 * Evaluates the intersection described in src/subgraph/warp_dfs.cuh, in four
 * shapes, cheapest first:
 *   - chain depth: the only back edge goes to depth-1, whose target atom the
 *     caller passes as @p prevTargetAtom, so the back-edge table is not read;
 *   - Uniform: every back edge intersects targetAdjacency;
 *   - Dual: each back edge's kBackEdgeAltMaskFlag picks its table;
 *   - General: recompute the neighbour set per edge from the cached packed row.
 *
 * Only the General case needs @p seen: it walks the query atom's raw bond slots,
 * which can name the same neighbour twice, whereas analyzeQueryBonds already
 * deduplicated the table.
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ __forceinline__ TargetMask<MaxTargetAtoms> buildCandidates(
  const WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>& shared,
  int                                                   warp,
  int                                                   depth,
  const unsigned char*                                  mapping,
  const TargetMask<MaxTargetAtoms>&                     used,
  const QueryMoleculeView&                              query,
  const QueryBondPlan&                                  plan,
  int                                                   prevTargetAtom) {
  using Mask = TargetMask<MaxTargetAtoms>;

  Mask candidates = shared.queryAtomCandidates[warp][depth];
  candidates.andNotEq(used);

  const uint64_t depthBit = 1ULL << depth;
  if (plan.chainDepths & depthBit) {
    candidates.andEq(shared.targetAdjacency[warp][prevTargetAtom].mask);
    return candidates;
  }
  if (plan.chainDepthsAlt & depthBit) {
    candidates.andEq(shared.targetAdjacencyAlt[warp][prevTargetAtom].mask);
    return candidates;
  }

  const int numBackEdges = shared.backEdgeCounts[warp][depth];

  if (plan.maskCase == BondMaskCase::Uniform) {
    for (int k = 0; k < numBackEdges && !candidates.empty(); ++k) {
      candidates.andEq(
        shared.targetAdjacency[warp][mapping[shared.backEdges[warp][depth][k] & kBackEdgeAtomMask]].mask);
    }
    return candidates;
  }

  if (plan.maskCase == BondMaskCase::Dual) {
    for (int k = 0; k < numBackEdges && !candidates.empty(); ++k) {
      const uint32_t edge   = shared.backEdges[warp][depth][k];
      const int      mapped = mapping[edge & kBackEdgeAtomMask];
      candidates.andEq((edge & kBackEdgeAltMaskFlag) ? shared.targetAdjacencyAlt[warp][mapped].mask :
                                                       shared.targetAdjacency[warp][mapped].mask);
    }
    return candidates;
  }

  const QueryAtomBonds& queryBonds = query.getQueryBonds(depth);
  const int             degree     = queryBonds.degree;
  uint64_t              seen       = 0;
  for (int slot = 0; slot < kMaxBondsPerAtom && !candidates.empty(); ++slot) {
    if (slot >= degree) {
      break;
    }
    const uint32_t neighborQueryAtom = queryBonds.neighborIdx[slot];
    if (neighborQueryAtom < static_cast<uint32_t>(depth) && !((seen >> neighborQueryAtom) & 1ULL)) {
      seen |= 1ULL << neighborQueryAtom;
      const int mapped = mapping[neighborQueryAtom];
      candidates.andEq(neighborsMatchingBondMask<MaxTargetAtoms>(shared.targetAdjacency[warp][mapped].packedRow,
                                                                 shared.targetAdjacencyAlt[warp][mapped].packedRow,
                                                                 queryBonds.matchMask[slot]));
    }
  }
  return candidates;
}

// =============================================================================
// Per-pair setup
// =============================================================================

/**
 * @brief Target atom @p atom 's label matrix row, as a bitset over query atoms.
 *
 * The label matrix is a BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>:
 * row-major with the target atom as the row, packed least significant bit first,
 * so a row is a contiguous MaxQueryAtoms-bit field.
 */
template <std::size_t MaxQueryAtoms>
__device__ __forceinline__ uint64_t readLabelRow(const uint32_t* labelWords, int atom) {
  if constexpr (MaxQueryAtoms == 16) {
    return static_cast<uint64_t>((labelWords[atom >> 1] >> ((atom & 1) * 16)) & 0xFFFFu);
  } else if constexpr (MaxQueryAtoms == 32) {
    return static_cast<uint64_t>(labelWords[atom]);
  } else {
    return reinterpret_cast<const uint64_t*>(labelWords)[atom];
  }
}

/**
 * @brief Transpose the pair's label matrix into shared.queryAtomCandidates.
 *
 * The label matrix is target-major (row t, bit q) but the search needs it
 * query-major: at depth d, the target atoms compatible with query atom d. Each
 * lane holds the rows of the target atoms it owns, so the transpose is one
 * __ballot_sync per (query atom, root group) -- balloting bit d yields 32 bits
 * of column d in target atom order.
 *
 * @return false if some query atom has no compatible target atom, in which case
 *         the pair cannot match and the caller must stop. The table is fully
 *         written either way.
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ __forceinline__ bool tryBuildQueryAtomCandidates(WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>& shared,
                                                            int                                             warp,
                                                            int                                             lane,
                                                            const uint32_t*                                 labelWords,
                                                            int numTargetAtoms,
                                                            int numQueryAtoms) {
  constexpr int kRoots = WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>::kRoots;

  uint64_t rows[kRoots];
#pragma unroll
  for (int k = 0; k < kRoots; ++k) {
    const int atom = lane + 32 * k;
    rows[k]        = atom < numTargetAtoms ? readLabelRow<MaxQueryAtoms>(labelWords, atom) : 0;
  }

  uint32_t anyEmptyColumn = 0;
  for (int depth = 0; depth < numQueryAtoms; ++depth) {
    uint32_t ballots[kRoots];
    uint32_t occupied = 0;
#pragma unroll
    for (int k = 0; k < kRoots; ++k) {
      ballots[k] = __ballot_sync(kFullWarpMask, static_cast<uint32_t>((rows[k] >> depth) & 1ULL));
      occupied |= ballots[k];
    }
    anyEmptyColumn |= static_cast<uint32_t>(occupied == 0u);

    // Ballots are uniform across the warp, so one lane assembles and stores.
    if (lane == 0) {
      TargetMask<MaxTargetAtoms> mask;
      mask.clear();
#pragma unroll
      for (int k = 0; k < kRoots; ++k) {
        mask.setWord32(k, ballots[k]);
      }
      shared.queryAtomCandidates[warp][depth] = mask;
    }
  }

  return anyEmptyColumn == 0u;
}

/**
 * @brief Build the back-edge table and classify the query's bond masks.
 *
 * Three passes over the query atoms, striped across the warp:
 *  1. Record each query atom's deduplicated back edges into
 *     shared.backEdges/backEdgeCounts, reducing the smallest and largest bond
 *     mask across the warp. Min and max are only a cheap fingerprint for the
 *     number of distinct masks: all masks equal implies min == max, and if
 *     exactly two values occur they are the min and the max.
 *  2. If min != max, flag the max-mask edges and watch for a third value, which
 *     demotes the pair to General.
 *  3. Uniform and Dual only: find the chain depths (see QueryBondPlan).
 *
 * This is a pure function of the query but runs once per pair; it could be
 * computed once per query at host pack time and loaded instead.
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ __forceinline__ QueryBondPlan analyzeQueryBonds(WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>& shared,
                                                           int                                             warp,
                                                           int                                             lane,
                                                           const QueryMoleculeView&                        query,
                                                           int numQueryAtoms) {
  QueryBondPlan plan;

  uint32_t laneMin = 0xFFFFFFFFu;
  uint32_t laneMax = 0u;

  for (int depth = lane; depth < numQueryAtoms; depth += 32) {
    const QueryAtomBonds& bonds  = query.getQueryBonds(depth);
    const int             degree = bonds.degree;
    int                   count  = 0;
    uint64_t              seen   = 0;
    for (int slot = 0; slot < kMaxBondsPerAtom; ++slot) {
      if (slot >= degree) {
        break;
      }
      const uint32_t neighbor = bonds.neighborIdx[slot];
      if (neighbor < static_cast<uint32_t>(depth) && !((seen >> neighbor) & 1ULL)) {
        seen |= 1ULL << neighbor;
        const uint32_t mask                  = bonds.matchMask[slot];
        shared.backEdges[warp][depth][count] = static_cast<unsigned char>(neighbor);
        laneMin                              = min(laneMin, mask);
        laneMax                              = max(laneMax, mask);
        ++count;
      }
    }
    shared.backEdgeCounts[warp][depth] = static_cast<unsigned char>(count);
  }

  plan.minBondMask   = warpReduceMin(laneMin);
  plan.maxBondMask   = warpReduceMax(laneMax);
  const bool uniform = (plan.minBondMask == plan.maxBondMask);
  __syncwarp();

  // The walk here must mirror the first pass exactly: it re-derives the same
  // slot ordering in order to flag the entries it wrote.
  uint32_t sawThirdMask = 0;
  if (!uniform) {
    for (int depth = lane; depth < numQueryAtoms; depth += 32) {
      const QueryAtomBonds& bonds  = query.getQueryBonds(depth);
      const int             degree = bonds.degree;
      int                   count  = 0;
      uint64_t              seen   = 0;
      for (int slot = 0; slot < kMaxBondsPerAtom; ++slot) {
        if (slot >= degree) {
          break;
        }
        const uint32_t neighbor = bonds.neighborIdx[slot];
        if (neighbor < static_cast<uint32_t>(depth) && !((seen >> neighbor) & 1ULL)) {
          seen |= 1ULL << neighbor;
          const uint32_t mask = bonds.matchMask[slot];
          if (mask != plan.minBondMask && mask != plan.maxBondMask) {
            sawThirdMask = 1;
          }
          if (mask == plan.maxBondMask) {
            shared.backEdges[warp][depth][count] |= kBackEdgeAltMaskFlag;
          }
          ++count;
        }
      }
    }
  }
  const bool dual = !uniform && (__ballot_sync(kFullWarpMask, sawThirdMask) == 0u);
  plan.maskCase   = uniform ? BondMaskCase::Uniform : (dual ? BondMaskCase::Dual : BondMaskCase::General);
  __syncwarp();

  if (plan.maskCase != BondMaskCase::General) {
    uint64_t lanePrimary = 0;
    uint64_t laneAlt     = 0;
    for (int depth = lane; depth < numQueryAtoms; depth += 32) {
      if (depth >= 1 && shared.backEdgeCounts[warp][depth] == 1) {
        const uint32_t edge = shared.backEdges[warp][depth][0];
        if ((edge & kBackEdgeAtomMask) == static_cast<uint32_t>(depth - 1)) {
          if (edge & kBackEdgeAltMaskFlag) {
            laneAlt |= 1ULL << depth;
          } else {
            lanePrimary |= 1ULL << depth;
          }
        }
      }
    }
    // Two 32-bit OR reductions per 64-bit lane mask; the reductions are 32-bit.
    plan.chainDepths = static_cast<uint64_t>(warpReduceOr(static_cast<uint32_t>(lanePrimary))) |
                       (static_cast<uint64_t>(warpReduceOr(static_cast<uint32_t>(lanePrimary >> 32))) << 32);
    plan.chainDepthsAlt = static_cast<uint64_t>(warpReduceOr(static_cast<uint32_t>(laneAlt))) |
                          (static_cast<uint64_t>(warpReduceOr(static_cast<uint32_t>(laneAlt >> 32))) << 32);
  }

  return plan;
}

/**
 * @brief Fill the per-warp adjacency tables for the classified plan, target
 *        atoms striped across the warp. See WarpSharedState for the contents.
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ __forceinline__ void buildTargetAdjacency(WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>& shared,
                                                     int                                             warp,
                                                     int                                             lane,
                                                     const TargetMoleculeView&                       target,
                                                     int                                             numTargetAtoms,
                                                     const QueryBondPlan&                            plan) {
  for (int atom = lane; atom < numTargetAtoms; atom += 32) {
    uint64_t neighbors;
    uint64_t bondInfo;
    packAdjacencyRow(target.targetAtomBonds[atom], neighbors, bondInfo);

    if (plan.maskCase == BondMaskCase::Uniform) {
      shared.targetAdjacency[warp][atom].mask =
        neighborsMatchingBondMask<MaxTargetAtoms>(neighbors, bondInfo, plan.maxBondMask);
    } else if (plan.maskCase == BondMaskCase::Dual) {
      shared.targetAdjacency[warp][atom].mask =
        neighborsMatchingBondMask<MaxTargetAtoms>(neighbors, bondInfo, plan.minBondMask);
      shared.targetAdjacencyAlt[warp][atom].mask =
        neighborsMatchingBondMask<MaxTargetAtoms>(neighbors, bondInfo, plan.maxBondMask);
    } else {
      shared.targetAdjacency[warp][atom].packedRow    = neighbors;
      shared.targetAdjacencyAlt[warp][atom].packedRow = bondInfo;
    }
  }
}

// =============================================================================
// Warp-per-pair DFS
// =============================================================================

/**
 * @brief Output plumbing for a single pair's DFS.
 */
struct DfsPairOutput {
  int*            matchCounts      = nullptr;  ///< Per-pair total embedding count
  int*            reportedCounts   = nullptr;  ///< Per-pair embeddings actually written
  int16_t*        matchIndices     = nullptr;  ///< Mapping storage
  int             matchOffset      = 0;        ///< Start of this pair's mapping storage
  int             storageCapacity  = 0;        ///< Mappings storable for this pair
  int             maxMatchesToFind = -1;       ///< Stop once this many are found (-1 = unlimited)
  bool            countOnly        = false;    ///< Suppress mapping writes
  PaintModeParams paint            = {};       ///< Paint destination (Paint mode only)
};

/**
 * @brief One warp searches one target/query pair.
 *
 * Builds the pair's three lookup tables, then runs nvMolKit::dfsFromRoots over
 * each lane's roots, with the output mode expressed as the terminal handler.
 *
 * The caller must have decoded the pair and pointed @p labelWords at that pair's
 * label matrix. All 32 lanes of the warp must call this.
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms, DfsOutputMode Mode>
__device__ void dfsSearchPair(const TargetMoleculeView&                       target,
                              const QueryMoleculeView&                        query,
                              const uint32_t*                                 labelWords,
                              WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>& shared,
                              int                                             warp,
                              int                                             lane,
                              int                                             resultIdx,
                              const DfsPairOutput&                            out) {
  constexpr int kRoots = WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>::kRoots;
  using Mask           = TargetMask<MaxTargetAtoms>;

  const int numTargetAtoms = target.numAtoms;
  const int numQueryAtoms  = query.numAtoms;

  // Count and Store leave the per-pair counters unwritten on every early exit
  // path unless we zero them here; the result buffers are not pre-zeroed.
  auto writeEmptyResult = [&]() {
    if constexpr (Mode != DfsOutputMode::Paint) {
      if (lane == 0) {
        out.matchCounts[resultIdx] = 0;
        if (!out.countOnly && out.reportedCounts != nullptr) {
          out.reportedCounts[resultIdx] = 0;
        }
      }
    }
  };

  if (numQueryAtoms > numTargetAtoms) {
    writeEmptyResult();
    return;
  }

  if (lane == 0) {
    shared.matchCount[warp]    = 0;
    shared.reportedCount[warp] = 0;
  }

  if (!tryBuildQueryAtomCandidates<MaxTargetAtoms,
                                   MaxQueryAtoms>(shared, warp, lane, labelWords, numTargetAtoms, numQueryAtoms)) {
    writeEmptyResult();
    return;
  }

  const int  limit     = out.maxMatchesToFind;
  const bool hasLimit  = limit >= 0;
  const int  lastDepth = numQueryAtoms - 1;

  // Writes one complete mapping. Returns true once the pair's limit is reached.
  auto emitMapping = [&](const unsigned char* mapping, int terminalAtom) -> bool {
    const int slot = atomicAdd(&shared.matchCount[warp], 1);
    if (!out.countOnly && slot < out.storageCapacity) {
      const int writeOffset = out.matchOffset + slot * numQueryAtoms;
      for (int q = 0; q < numQueryAtoms - 1; ++q) {
        out.matchIndices[writeOffset + q] = static_cast<int16_t>(mapping[q]);
      }
      out.matchIndices[writeOffset + numQueryAtoms - 1] = static_cast<int16_t>(terminalAtom);
      atomicAdd(&shared.reportedCount[warp], 1);
    }
    return hasLimit && (slot + 1) >= limit;
  };

  // Mode-dependent result write, shared by the single-atom and general paths.
  auto finishPair = [&](uint32_t laneTotal) {
    if constexpr (Mode == DfsOutputMode::Count) {
      uint32_t pairTotal = warpReduceAdd(laneTotal);
      if (hasLimit && pairTotal > static_cast<uint32_t>(limit)) {
        pairTotal = static_cast<uint32_t>(limit);
      }
      if (lane == 0) {
        out.matchCounts[resultIdx] = static_cast<int>(pairTotal);
        if (!out.countOnly && out.reportedCounts != nullptr) {
          out.reportedCounts[resultIdx] = 0;
        }
      }
    } else if constexpr (Mode == DfsOutputMode::Store) {
      // Store accumulated into the shared counters as it went, so this is just a
      // barrier before reading them.
      __syncwarp();
      if (lane == 0) {
        out.matchCounts[resultIdx] = shared.matchCount[warp];
        if (!out.countOnly && out.reportedCounts != nullptr) {
          out.reportedCounts[resultIdx] = shared.reportedCount[warp];
        }
      }
    }
  };

  // Builds the roots this lane owns -- target atoms lane, lane+32, lane+64,
  // lane+96 -- restricted to the ones label-compatible with query atom 0.
  auto buildLaneRoots = [&]() -> Mask {
    Mask rootMask;
    rootMask.clear();
#pragma unroll
    for (int k = 0; k < kRoots; ++k) {
      rootMask.set(lane + 32 * k);
    }
    Mask roots = shared.queryAtomCandidates[warp][0];
    roots.andEq(rootMask);
    return roots;
  };

  if (lastDepth == 0) {
    // Single-atom query: no bonds to satisfy, so every candidate root is already
    // a complete embedding. Exits before the bond analysis and adjacency build,
    // none of whose output it would read. (dfsFromRoots requires lastDepth >= 1,
    // so single-atom queries are the frontend's job by contract.)
    __syncwarp();  // The transpose's lane-0 shared writes.
    Mask     roots     = buildLaneRoots();
    uint32_t laneTotal = 0;
    if constexpr (Mode == DfsOutputMode::Count) {
      laneTotal = static_cast<uint32_t>(roots.popcount());
    } else if constexpr (Mode == DfsOutputMode::Paint) {
      while (!roots.empty()) {
        const int root = roots.lowest();
        roots.clearLowest();
        atomicOr(&out.paint.recursiveBits[out.paint.outputPairIdx * out.paint.maxTargetAtoms + root],
                 1u << out.paint.patternId);
      }
    } else {
      unsigned char mapping[1];
      while (!roots.empty()) {
        const int root = roots.lowest();
        roots.clearLowest();
        if (emitMapping(mapping, root)) {
          break;
        }
      }
    }
    finishPair(laneTotal);
    return;
  }

  const QueryBondPlan plan = analyzeQueryBonds<MaxTargetAtoms, MaxQueryAtoms>(shared, warp, lane, query, numQueryAtoms);

  buildTargetAdjacency<MaxTargetAtoms, MaxQueryAtoms>(shared, warp, lane, target, numTargetAtoms, plan);
  __syncwarp();

  uint32_t laneTotal = 0;

  auto candidatesAt = [&](int depth, const unsigned char* mapping, const Mask& used, int prevTargetAtom) -> Mask {
    return buildCandidates<MaxTargetAtoms,
                           MaxQueryAtoms>(shared, warp, depth, mapping, used, query, plan, prevTargetAtom);
  };

  // The output mode as a terminal handler; every bit of @p terminals completes
  // a distinct embedding with mapping[0..lastDepth-1].
  auto onTerminal = [&](Mask terminals, const unsigned char* mapping) -> DfsTerminalVerdict {
    DfsTerminalVerdict verdict{false, false};
    if constexpr (Mode == DfsOutputMode::Count) {
      laneTotal += static_cast<uint32_t>(terminals.popcount());
      // A lane that has already reached the pair limit on its own roots
      // cannot change the clamped answer, so it stops. Lanes cannot see
      // each other's partial totals mid-search without a warp collective
      // in divergent code, so the other lanes run to exhaustion.
      if (hasLimit && laneTotal >= static_cast<uint32_t>(limit)) {
        verdict.laneDone = true;
      }
    } else if constexpr (Mode == DfsOutputMode::Paint) {
      // Paint only asks whether the root can start an embedding, so the
      // first hit settles the root and skips the rest of its subtree.
      if (!terminals.empty()) {
        atomicOr(&out.paint.recursiveBits[out.paint.outputPairIdx * out.paint.maxTargetAtoms + mapping[0]],
                 1u << out.paint.patternId);
        verdict.rootDone = true;
      }
    } else {
      while (!terminals.empty()) {
        const int terminalAtom = terminals.lowest();
        terminals.clearLowest();
        if (emitMapping(mapping, terminalAtom)) {
          verdict.laneDone = true;
          break;
        }
      }
    }
    return verdict;
  };

  dfsFromRoots<static_cast<int>(MaxQueryAtoms)>(buildLaneRoots(), lastDepth, candidatesAt, onTerminal, [] {
    return false;
  });

  finishPair(laneTotal);
}

}  // namespace dfs
}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_DFS_CUH
