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
// tables the candidates callback reads, supplies that callback, and implements the
// output modes as terminal handlers. Query atoms are matched in index order,
// so depth d means "choose the target atom for query atom d".
//
// One warp per pair, kWarpsPerBlock pairs per CTA. Lane L searches the subtrees
// rooted at target atoms L, L+32, L+64, L+96; those subtrees are disjoint, so
// all search state is lane-local registers and lanes only meet at the final
// reduction. Shared memory holds the per-warp lookup tables and result counters.
//
// Three tables are built per pair before the search, each depending on the last:
// tryBuildQueryAtomCandidates (the label-compatibility term of the candidate
// intersection), then analyzeQueryEdges (the back edges) and
// buildTargetAdjacency (the per-edge neighbour sets) from candidate_tables.cuh.
//
// Future optimization candidates:
// - Store target adjacency as the two packed degree-8 words at construction,
//   deleting packAdjacencyRow.
// - Produce the label matrix query-major for this backend, deleting the ballot
//   transpose in tryBuildQueryAtomCandidates.
// - Have the label kernel flag pairs with an empty column and compact the DFS
//   pair list on device.
// - Precompute analyzeQueryEdges per query at host pack time (see its note).
// - Reorder query atoms most-constrained-first at construction, with an inverse
//   permutation on mapping download.
// - Route size-1 queries to a popcount-over-label-column kernel.
// - Drop query>target pairs in the minibatch planner.

#ifndef NVMOLKIT_SUBSTRUCT_DFS_CUH
#define NVMOLKIT_SUBSTRUCT_DFS_CUH

#include <cstddef>
#include <cstdint>

#include "src/subgraph/candidate_tables.cuh"
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

// The per-pair table machinery, bond-mask classification, and candidates
// callback come from src/subgraph/candidate_tables.cuh; this header supplies the
// substructure adapter and output modes.
using nvMolKit::subgraph::BondMaskCase;
using nvMolKit::subgraph::QueryBondPlan;

/// Per-warp tables for a CTA of kWarpsPerBlock pairs.
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
using WarpSharedState = nvMolKit::subgraph::WarpSharedState<MaxTargetAtoms, MaxQueryAtoms, kWarpsPerBlock>;

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
// Substructure adapter
// =============================================================================

/**
 * @brief Presents a (query, target) molecule pair to the generic frontend.
 *
 * Search order is the identity over query atoms, so depth d is query atom d and
 * a neighbour's depth is just its atom index. Edge keys are QueryAtomBonds
 * match masks, and target rows pack into two words, so the General case caches
 * them in shared memory rather than re-reading global memory per descent.
 */
template <std::size_t MaxTargetAtoms> struct SubstructAdapter {
  using Mask                              = TargetMask<MaxTargetAtoms>;
  static constexpr bool kCachesTargetRows = true;

  const QueryMoleculeView*  query;
  const TargetMoleculeView* target;

  __device__ __forceinline__ int degreeAt(int depth) const { return query->getQueryBonds(depth).degree; }

  __device__ __forceinline__ int neighborDepthAt(int depth, int slot) const {
    return static_cast<int>(query->getQueryBonds(depth).neighborIdx[slot]);
  }

  __device__ __forceinline__ uint32_t edgeKeyAt(int depth, int slot) const {
    return query->getQueryBonds(depth).matchMask[slot];
  }

  __device__ __forceinline__ Mask neighborsMatching(int targetAtom, uint32_t edgeKey) const {
    uint64_t neighbors;
    uint64_t bondInfo;
    packAdjacencyRow(target->targetAtomBonds[targetAtom], neighbors, bondInfo);
    return neighborsMatchingBondMask<MaxTargetAtoms>(neighbors, bondInfo, edgeKey);
  }

  __device__ __forceinline__ void packTargetRow(int targetAtom, uint64_t& w0, uint64_t& w1) const {
    packAdjacencyRow(target->targetAtomBonds[targetAtom], w0, w1);
  }

  __device__ __forceinline__ Mask neighborsMatchingPacked(uint64_t w0, uint64_t w1, uint32_t edgeKey) const {
    return neighborsMatchingBondMask<MaxTargetAtoms>(w0, w1, edgeKey);
  }
};

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
 * @brief Transpose the pair's label matrix into shared.depthCandidates.
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
      shared.depthCandidates[warp][depth] = mask;
    }
  }

  return anyEmptyColumn == 0u;
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
    Mask roots = shared.depthCandidates[warp][0];
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

  const SubstructAdapter<MaxTargetAtoms> adapter{&query, &target};

  const QueryBondPlan plan = nvMolKit::subgraph::analyzeQueryEdges(shared, warp, lane, adapter, numQueryAtoms);

  nvMolKit::subgraph::buildTargetAdjacency(shared, warp, lane, adapter, numTargetAtoms, plan);
  __syncwarp();

  uint32_t laneTotal = 0;

  auto candidatesAt = [&](int depth, const unsigned char* mapping, const Mask& used, int prevTargetAtom) -> Mask {
    return nvMolKit::subgraph::buildCandidates(shared, warp, depth, mapping, used, adapter, plan, prevTargetAtom);
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
