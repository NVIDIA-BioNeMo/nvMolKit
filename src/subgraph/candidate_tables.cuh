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

// Per-pair lookup tables and candidate sets for the warp DFS.
//
// This is the layer between nvMolKit::dfsFromRoots (which knows only about
// candidate bitsets) and a concrete matching problem. It owns the parts that
// every labelled-subgraph problem needs and none of the parts specific to one:
//
//   - the back-edge table: for each search depth, which earlier depths
//     constrain it, deduplicated, with each edge's predicate key;
//   - the bond-mask classification (Uniform / Dual / General): if the query's
//     back edges carry at most two distinct predicate keys, the target
//     neighbour sets those keys induce are precomputed once per pair, turning
//     per-edge work in the inner loop into a table lookup;
//   - buildCandidates, the candidates callback handed to dfsFromRoots.
//
// Everything problem-specific is behind an Adapter, which presents the query
// *in search order*: depth d is a search position, not necessarily query atom
// d. The substructure backend uses the identity order over all query atoms; a
// frontend that searches a permuted order supplies the permutation in its
// adapter. This layer never sees atom ids, only depths.
//
// Adapter contract (all __device__, all const):
//
//   using Mask = TargetMask<N>;                   // target-atom bitset
//   static constexpr bool kCachesTargetRows;      // see below
//
//   int      degreeAt(int depth) const;           // edge slots at this depth;
//                                                 // must not exceed
//                                                 // kMaxEdgeSlotsPerAtom, or
//                                                 // back edges are silently
//                                                 // dropped
//   int      neighborDepthAt(int depth, int slot) const;
//                                                 // search depth of that
//                                                 // neighbour
//   uint32_t edgeKeyAt(int depth, int slot) const;
//                                                 // opaque predicate key;
//                                                 // equal keys must mean
//                                                 // identical predicates
//   Mask     neighborsMatching(int targetAtom, uint32_t edgeKey) const;
//                                                 // target atoms reachable
//                                                 // from targetAtom over an
//                                                 // edge that edgeKey accepts
//
//   // Only when kCachesTargetRows: pack a target atom's adjacency into two
//   // words so the General case can recompute neighbour sets from shared
//   // memory instead of re-reading global memory on every descent.
//   void packTargetRow(int targetAtom, uint64_t& w0, uint64_t& w1) const;
//   Mask neighborsMatchingPacked(uint64_t w0, uint64_t w1, uint32_t edgeKey) const;
//
// kCachesTargetRows is a performance choice, not a semantic one: an adapter
// whose adjacency does not fit in 128 bits (uncapped degree, wide labels)
// sets it false and pays a global walk per General-case descent.
//
// The label-compatibility term (WarpSharedState::depthCandidates) is filled by
// the frontend, not the adapter: the substructure backend transposes its
// target-major label matrix with a ballot per depth. buildCandidates only
// reads that table.

#ifndef NVMOLKIT_SUBGRAPH_CANDIDATE_TABLES_CUH
#define NVMOLKIT_SUBGRAPH_CANDIDATE_TABLES_CUH

#include <cstddef>
#include <cstdint>

#include "src/subgraph/target_mask.cuh"
#include "src/subgraph/warp_reduce.cuh"

namespace nvMolKit {
namespace subgraph {

/// Max edge slots per atom the back-edge table reserves per depth.
constexpr int kMaxEdgeSlotsPerAtom = 8;

// Entry layout in WarpSharedState::backEdges. A back edge of depth d goes to a
// depth smaller than d, i.e. one already mapped when d is chosen.
constexpr unsigned char kBackEdgeDepthMask   = 63u;  ///< Low 6 bits: the earlier depth.
constexpr unsigned char kBackEdgeAltMaskFlag = 64u;  ///< Bit 6 (Dual only): use the Alt adjacency table.

/**
 * @brief How many distinct edge predicate keys the query's back edges carry.
 *
 * With at most two, the target neighbour sets those keys induce can be
 * precomputed once per pair.
 */
enum class BondMaskCase {
  Uniform,  ///< One key. One adjacency table suffices.
  Dual,     ///< Two keys. Two adjacency tables; each back edge flags which it uses.
  General   ///< Three or more. Neighbour sets recomputed per edge during the search.
};

/// Classification of the query's edge constraints; identical in every lane.
struct QueryBondPlan {
  BondMaskCase maskCase       = BondMaskCase::General;
  uint32_t     minBondMask    = 0;  ///< Smallest key on any back edge.
  uint32_t     maxBondMask    = 0;  ///< Largest key on any back edge; equals minBondMask iff Uniform.
  /// Bit d set iff depth d's only back edge goes to depth d-1 and resolves
  /// through targetAdjacency. Such a depth needs no back-edge lookup, since the
  /// caller just chose depth d-1's target atom.
  uint64_t     chainDepths    = 0;
  /// As chainDepths, for depths resolving through targetAdjacencyAlt.
  uint64_t     chainDepthsAlt = 0;
};

/**
 * @brief Per-warp lookup tables and counters, sized for a whole CTA.
 *
 * Warp w touches only [w][...], so no synchronisation beyond __syncwarp() is
 * needed. Where this is a kernel's only __shared__ allocation, its size alone
 * determines how many CTAs fit per SM (see occupancy.cuh).
 *
 * The adjacency tables hold different things per BondMaskCase; buildTargetAdjacency
 * is the only place that fills them:
 *   Uniform: targetAdjacency[a] = atoms reachable from a over an edge the
 *            query's single key accepts. Alt unused.
 *   Dual:    targetAdjacency[a] for minBondMask, targetAdjacencyAlt[a] for
 *            maxBondMask; each back edge picks one via kBackEdgeAltMaskFlag.
 *   General: with kCachesTargetRows, the tables cache the adapter's packed
 *            adjacency row instead; without it, they are unused.
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxSearchDepth, int WarpsPerBlock> struct WarpSharedState {
  static_assert(MaxSearchDepth <= 64, "Back-edge depths pack into 6 bits and chain depths into a 64-bit mask");

  /// Roots each lane owns: lane, lane + 32, ... below MaxTargetAtoms.
  static constexpr int kRoots = static_cast<int>(MaxTargetAtoms / 32);
  using Mask                  = TargetMask<MaxTargetAtoms>;

  /**
   * @brief One adjacency table slot, read as whichever member the case stores.
   *
   * A union because cached packed rows are always 64 bits wide, whereas a mask
   * is only as wide as MaxTargetAtoms. This keeps the table exactly as large as
   * the wider of the two, so narrowing Mask costs no shared memory here while
   * still narrowing it in registers.
   */
  union AdjacencyEntry {
    Mask     mask;
    uint64_t packedRow;
  };

  AdjacencyEntry targetAdjacency[WarpsPerBlock][MaxTargetAtoms];
  AdjacencyEntry targetAdjacencyAlt[WarpsPerBlock][MaxTargetAtoms];
  /// Per depth, the label-compatible target atoms.
  Mask           depthCandidates[WarpsPerBlock][MaxSearchDepth];
  /// Per depth, its back edges, kBackEdge*-encoded.
  unsigned char  backEdges[WarpsPerBlock][MaxSearchDepth][kMaxEdgeSlotsPerAtom];
  unsigned char  backEdgeCounts[WarpsPerBlock][MaxSearchDepth];  ///< Valid entries in backEdges[w][d].
  int            matchCount[WarpsPerBlock];                      ///< Embeddings found for the warp's pair.
  int            reportedCount[WarpsPerBlock];                   ///< Of those, how many fit in the output buffer.
};

/**
 * @brief Build the back-edge table and classify the query's edge keys.
 *
 * Three passes over the depths, striped across the warp:
 *  1. Record each depth's deduplicated back edges, reducing the smallest and
 *     largest key across the warp. Min and max are only a cheap fingerprint for
 *     the number of distinct keys: all keys equal implies min == max, and if
 *     exactly two values occur they are the min and the max.
 *  2. If min != max, flag the max-key edges and watch for a third value, which
 *     demotes the pair to General.
 *  3. Uniform and Dual only: find the chain depths (see QueryBondPlan).
 *
 * The result depends only on the query, yet is recomputed for every pair; an
 * adapter whose query side is fixed across a batch could compute it once at
 * host pack time and load it instead.
 */
template <class Adapter, std::size_t MaxTargetAtoms, std::size_t MaxSearchDepth, int WarpsPerBlock>
__device__ __forceinline__ QueryBondPlan
analyzeQueryEdges(WarpSharedState<MaxTargetAtoms, MaxSearchDepth, WarpsPerBlock>& shared,
                  int                                                             warp,
                  int                                                             lane,
                  const Adapter&                                                  adapter,
                  int                                                             numDepths) {
  QueryBondPlan plan;

  uint32_t laneMin = 0xFFFFFFFFu;
  uint32_t laneMax = 0u;

  for (int depth = lane; depth < numDepths; depth += 32) {
    const int degree = adapter.degreeAt(depth);
    int       count  = 0;
    uint64_t  seen   = 0;
    for (int slot = 0; slot < kMaxEdgeSlotsPerAtom && slot < degree; ++slot) {
      const int neighborDepth = adapter.neighborDepthAt(depth, slot);
      if (neighborDepth >= depth) {
        continue;
      }
      if ((seen >> neighborDepth) & 1ULL) {
        continue;
      }
      seen |= 1ULL << neighborDepth;
      const uint32_t key                   = adapter.edgeKeyAt(depth, slot);
      shared.backEdges[warp][depth][count] = static_cast<unsigned char>(neighborDepth);
      laneMin                              = min(laneMin, key);
      laneMax                              = max(laneMax, key);
      ++count;
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
    for (int depth = lane; depth < numDepths; depth += 32) {
      const int degree = adapter.degreeAt(depth);
      int       count  = 0;
      uint64_t  seen   = 0;
      for (int slot = 0; slot < kMaxEdgeSlotsPerAtom && slot < degree; ++slot) {
        const int neighborDepth = adapter.neighborDepthAt(depth, slot);
        if (neighborDepth >= depth) {
          continue;
        }
        if ((seen >> neighborDepth) & 1ULL) {
          continue;
        }
        seen |= 1ULL << neighborDepth;
        const uint32_t key = adapter.edgeKeyAt(depth, slot);
        if (key != plan.minBondMask && key != plan.maxBondMask) {
          sawThirdMask = 1;
        }
        if (key == plan.maxBondMask) {
          shared.backEdges[warp][depth][count] |= kBackEdgeAltMaskFlag;
        }
        ++count;
      }
    }
  }
  const bool dual = !uniform && (__ballot_sync(kFullWarpMask, sawThirdMask) == 0u);
  plan.maskCase   = uniform ? BondMaskCase::Uniform : (dual ? BondMaskCase::Dual : BondMaskCase::General);
  __syncwarp();

  if (plan.maskCase != BondMaskCase::General) {
    uint64_t lanePrimary = 0;
    uint64_t laneAlt     = 0;
    for (int depth = lane; depth < numDepths; depth += 32) {
      if (depth >= 1 && shared.backEdgeCounts[warp][depth] == 1) {
        const uint32_t edge = shared.backEdges[warp][depth][0];
        if ((edge & kBackEdgeDepthMask) == static_cast<uint32_t>(depth - 1)) {
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
template <class Adapter, std::size_t MaxTargetAtoms, std::size_t MaxSearchDepth, int WarpsPerBlock>
__device__ __forceinline__ void buildTargetAdjacency(
  WarpSharedState<MaxTargetAtoms, MaxSearchDepth, WarpsPerBlock>& shared,
  int                                                             warp,
  int                                                             lane,
  const Adapter&                                                  adapter,
  int                                                             numTargetAtoms,
  const QueryBondPlan&                                            plan) {
  for (int atom = lane; atom < numTargetAtoms; atom += 32) {
    if (plan.maskCase == BondMaskCase::Uniform) {
      shared.targetAdjacency[warp][atom].mask = adapter.neighborsMatching(atom, plan.maxBondMask);
    } else if (plan.maskCase == BondMaskCase::Dual) {
      shared.targetAdjacency[warp][atom].mask    = adapter.neighborsMatching(atom, plan.minBondMask);
      shared.targetAdjacencyAlt[warp][atom].mask = adapter.neighborsMatching(atom, plan.maxBondMask);
    } else if constexpr (Adapter::kCachesTargetRows) {
      uint64_t w0;
      uint64_t w1;
      adapter.packTargetRow(atom, w0, w1);
      shared.targetAdjacency[warp][atom].packedRow    = w0;
      shared.targetAdjacencyAlt[warp][atom].packedRow = w1;
    }
  }
}

/**
 * @brief The target atoms depth @p depth may still map to: the candidates
 *        callback handed to nvMolKit::dfsFromRoots.
 *
 * Evaluates the intersection described in warp_dfs.cuh, in four shapes,
 * cheapest first:
 *   - chain depth: the only back edge goes to depth-1, whose target atom the
 *     caller passes as @p prevTargetAtom, so the back-edge table is not read;
 *   - Uniform: every back edge intersects targetAdjacency;
 *   - Dual: each back edge's kBackEdgeAltMaskFlag picks its table;
 *   - General: recompute the neighbour set per edge, from the cached packed row
 *     if the adapter caches, otherwise from the adapter directly.
 *
 * The General case walks the adapter's raw edge slots, which can name the same
 * neighbour twice, so it dedups against @p seen; the other cases read the
 * already-deduplicated back-edge table.
 */
template <class Adapter, std::size_t MaxTargetAtoms, std::size_t MaxSearchDepth, int WarpsPerBlock>
__device__ __forceinline__ TargetMask<MaxTargetAtoms> buildCandidates(
  const WarpSharedState<MaxTargetAtoms, MaxSearchDepth, WarpsPerBlock>& shared,
  int                                                                   warp,
  int                                                                   depth,
  const unsigned char*                                                  mapping,
  const TargetMask<MaxTargetAtoms>&                                     used,
  const Adapter&                                                        adapter,
  const QueryBondPlan&                                                  plan,
  int                                                                   prevTargetAtom) {
  using Mask = TargetMask<MaxTargetAtoms>;

  Mask candidates = shared.depthCandidates[warp][depth];
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
        shared.targetAdjacency[warp][mapping[shared.backEdges[warp][depth][k] & kBackEdgeDepthMask]].mask);
    }
    return candidates;
  }

  if (plan.maskCase == BondMaskCase::Dual) {
    for (int k = 0; k < numBackEdges && !candidates.empty(); ++k) {
      const uint32_t edge   = shared.backEdges[warp][depth][k];
      const int      mapped = mapping[edge & kBackEdgeDepthMask];
      candidates.andEq((edge & kBackEdgeAltMaskFlag) ? shared.targetAdjacencyAlt[warp][mapped].mask :
                                                       shared.targetAdjacency[warp][mapped].mask);
    }
    return candidates;
  }

  const int degree = adapter.degreeAt(depth);
  uint64_t  seen   = 0;
  for (int slot = 0; slot < kMaxEdgeSlotsPerAtom && slot < degree && !candidates.empty(); ++slot) {
    const int neighborDepth = adapter.neighborDepthAt(depth, slot);
    if (neighborDepth >= depth) {
      continue;
    }
    if ((seen >> neighborDepth) & 1ULL) {
      continue;
    }
    seen |= 1ULL << neighborDepth;
    const int      mapped = mapping[neighborDepth];
    const uint32_t key    = adapter.edgeKeyAt(depth, slot);
    if constexpr (Adapter::kCachesTargetRows) {
      candidates.andEq(adapter.neighborsMatchingPacked(shared.targetAdjacency[warp][mapped].packedRow,
                                                       shared.targetAdjacencyAlt[warp][mapped].packedRow,
                                                       key));
    } else {
      candidates.andEq(adapter.neighborsMatching(mapped, key));
    }
  }
  return candidates;
}

}  // namespace subgraph
}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBGRAPH_CANDIDATE_TABLES_CUH
