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

// Warp-per-pair bitset DFS ported from the Kernel Factory count-matches winner
// (`gsi_count_bitset_dfs_v1`, candidate 81941d96..., 4.904x over the GSI
// baseline). The execution model of the reference is preserved verbatim: eight
// warps and eight independent pairs per CTA, one warp per target/query pair,
// each lane owning the roots at bit positions `lane + 32*k`, and lane-local DFS
// state. The extensions beyond the reference are 128-atom targets (two mask
// words instead of one), match-limit handling, stored mappings, and recursive
// paint, none of which change the reference's launch topology or search order.

#ifndef NVMOLKIT_SUBSTRUCT_DFS_CUH
#define NVMOLKIT_SUBSTRUCT_DFS_CUH

#include <cstddef>
#include <cstdint>

#include "src/substruct/molecules_device.cuh"
#include "src/substruct/packed_bonds_device.cuh"
#include "src/substruct/substruct_algos.cuh"

namespace nvMolKit {
namespace dfs {

/// Warps, and therefore independent pairs, per CTA. This is the reference's WPB.
constexpr int      kWarpsPerBlock = 8;
constexpr int      kBlockSize     = kWarpsPerBlock * 32;
constexpr uint32_t kFullWarpMask  = 0xFFFFFFFFu;

/// How a warp records the embeddings it finds.
enum class DfsOutputMode {
  Count,  ///< Per-pair embedding count only; no mapping storage, no per-match atomics.
  Store,  ///< Full mappings written to the match index buffer.
  Paint   ///< Recursive SMARTS bit painted for mapping[0]; stops each root at its first embedding.
};

// =============================================================================
// Warp reductions
// =============================================================================
//
// The reference uses __reduce_*_sync unconditionally. nvMolKit still builds for
// sm_70/sm_75 where those intrinsics do not exist, so pre-Ampere targets get a
// butterfly shuffle that produces identical values.

__device__ __forceinline__ uint32_t warpReduceAdd(uint32_t value) {
#if __CUDA_ARCH__ >= 800
  return __reduce_add_sync(kFullWarpMask, value);
#else
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFullWarpMask, value, offset);
  }
  return value;
#endif
}

__device__ __forceinline__ uint32_t warpReduceOr(uint32_t value) {
#if __CUDA_ARCH__ >= 800
  return __reduce_or_sync(kFullWarpMask, value);
#else
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value |= __shfl_xor_sync(kFullWarpMask, value, offset);
  }
  return value;
#endif
}

__device__ __forceinline__ uint32_t warpReduceMin(uint32_t value) {
#if __CUDA_ARCH__ >= 800
  return __reduce_min_sync(kFullWarpMask, value);
#else
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = min(value, __shfl_xor_sync(kFullWarpMask, value, offset));
  }
  return value;
#endif
}

__device__ __forceinline__ uint32_t warpReduceMax(uint32_t value) {
#if __CUDA_ARCH__ >= 800
  return __reduce_max_sync(kFullWarpMask, value);
#else
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = max(value, __shfl_xor_sync(kFullWarpMask, value, offset));
  }
  return value;
#endif
}

// =============================================================================
// Target atom bitsets
// =============================================================================

/**
 * @brief Bitset over target atoms, one bit per atom.
 *
 * Specialised rather than looped so that the 64-atom form is a bare uint64_t,
 * exactly as in the reference, and the 128-atom form never dynamically indexes
 * a register-resident array.
 */
template <int Words> struct TargetMask;

template <> struct TargetMask<1> {
  uint64_t lo;

  __device__ __forceinline__ void clear() { lo = 0; }
  __device__ __forceinline__ bool empty() const { return lo == 0; }
  __device__ __forceinline__ void set(int bit) { lo |= 1ULL << bit; }
  __device__ __forceinline__ void reset(int bit) { lo &= ~(1ULL << bit); }
  __device__ __forceinline__ void setIf(int bit, uint64_t value) { lo |= value << bit; }
  __device__ __forceinline__ int  popcount() const { return __popcll(lo); }
  __device__ __forceinline__ int  lowest() const { return __ffsll(static_cast<long long>(lo)) - 1; }
  __device__ __forceinline__ void clearLowest() { lo &= lo - 1; }
  __device__ __forceinline__ void andEq(const TargetMask& other) { lo &= other.lo; }
  __device__ __forceinline__ void andNotEq(const TargetMask& other) { lo &= ~other.lo; }
};

template <> struct TargetMask<2> {
  uint64_t lo;
  uint64_t hi;

  __device__ __forceinline__ void clear() { lo = hi = 0; }
  __device__ __forceinline__ bool empty() const { return (lo | hi) == 0; }
  __device__ __forceinline__ void set(int bit) {
    if (bit < 64) {
      lo |= 1ULL << bit;
    } else {
      hi |= 1ULL << (bit - 64);
    }
  }
  __device__ __forceinline__ void reset(int bit) {
    if (bit < 64) {
      lo &= ~(1ULL << bit);
    } else {
      hi &= ~(1ULL << (bit - 64));
    }
  }
  __device__ __forceinline__ void setIf(int bit, uint64_t value) {
    if (bit < 64) {
      lo |= value << bit;
    } else {
      hi |= value << (bit - 64);
    }
  }
  __device__ __forceinline__ int popcount() const { return __popcll(lo) + __popcll(hi); }
  __device__ __forceinline__ int lowest() const {
    return lo != 0 ? __ffsll(static_cast<long long>(lo)) - 1 : __ffsll(static_cast<long long>(hi)) + 63;
  }
  __device__ __forceinline__ void clearLowest() {
    if (lo != 0) {
      lo &= lo - 1;
    } else {
      hi &= hi - 1;
    }
  }
  __device__ __forceinline__ void andEq(const TargetMask& other) {
    lo &= other.lo;
    hi &= other.hi;
  }
  __device__ __forceinline__ void andNotEq(const TargetMask& other) {
    lo &= ~other.lo;
    hi &= ~other.hi;
  }
};

// =============================================================================
// Per-warp shared state
// =============================================================================

/**
 * @brief The reference's per-warp shared arrays, sized for a whole CTA.
 *
 * `tn8`/`tb8` serve double duty exactly as in the reference: bond-compatible
 * target neighbour masks in the uniform and dual cases, and the raw degree-8
 * packed adjacency rows in the general case, where only the low word is used.
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms> struct WarpSharedState {
  static constexpr int kWords = static_cast<int>((MaxTargetAtoms + 63) / 64);
  static constexpr int kRoots = static_cast<int>(MaxTargetAtoms / 32);
  using Mask                  = TargetMask<kWords>;

  Mask          tn8[kWarpsPerBlock][MaxTargetAtoms];
  Mask          tb8[kWarpsPerBlock][MaxTargetAtoms];
  Mask          lm[kWarpsPerBlock][MaxQueryAtoms];
  unsigned char bej[kWarpsPerBlock][MaxQueryAtoms][kMaxBondsPerAtom];
  unsigned char bcnt[kWarpsPerBlock][MaxQueryAtoms];
  int           matchCount[kWarpsPerBlock];
  int           reportedCount[kWarpsPerBlock];
};

/**
 * @brief Occupancy target for a DFS specialisation's __launch_bounds__.
 *
 * The reference pins `__launch_bounds__(256, 8)`. That second argument is what
 * caps its register allocation at 32 and is therefore part of the configuration
 * that produced its measured 4.904x, so it is matched here. The 128-atom target
 * specialisations cannot reach eight resident CTAs at any register count -- their
 * per-warp adjacency masks alone exceed the per-SM shared memory budget -- so
 * they ask only for what shared memory already allows.
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms> constexpr int minBlocksPerSM() {
  // Hopper's 164 KB per SM; the smallest per-SM budget among the architectures
  // that matter for this backend, so the target is never over-promised.
  constexpr std::size_t kSharedBudgetBytes = 164 * 1024;
  constexpr std::size_t kResident = kSharedBudgetBytes / sizeof(WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>);
  return kResident >= 8 ? 8 : (kResident < 1 ? 1 : static_cast<int>(kResident));
}

// =============================================================================
// Packed adjacency helpers
// =============================================================================

/**
 * @brief Pack a target atom's adjacency into the reference's two degree-8 words.
 *
 * The reference reads `tnbr`/`tbond` as separate 8-byte-stride arrays with a
 * single uint64_t load. nvMolKit interleaves degree, neighbours and bond info in
 * a 20-byte `TargetAtomBonds`, so the words are assembled here instead. Unused
 * slots take the reference's 255 sentinel so the consumers are unchanged.
 */
__device__ __forceinline__ void packTargetRow(const TargetAtomBonds& bonds, uint64_t& neighbors, uint64_t& bondInfo) {
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
 * @brief Reference `tnbrmask`: target neighbours reachable over a bond the query mask allows.
 */
template <int Words>
__device__ __forceinline__ TargetMask<Words> tnbrMask(uint64_t neighbors, uint64_t bondInfo, uint32_t queryMask) {
  TargetMask<Words> mask;
  mask.clear();
#pragma unroll 1
  for (int k = 0; k < kMaxBondsPerAtom; ++k) {
    const uint32_t neighbor = static_cast<uint32_t>(neighbors & 0xFFu);
    if (neighbor == 255u) {
      break;
    }
    const uint32_t info = static_cast<uint32_t>(bondInfo & 0xFFu);
    const uint32_t code = ((info >> 4) & 1u) * 16u + (info & 15u);
    mask.setIf(static_cast<int>(neighbor), static_cast<uint64_t>((queryMask >> code) & 1u));
    neighbors >>= 8;
    bondInfo >>= 8;
  }
  return mask;
}

// =============================================================================
// Candidate construction
// =============================================================================

/**
 * @brief Build the candidate set for one DFS depth.
 *
 * Mirrors the reference's CAND_BODY: the uniform and dual cases consume the
 * precomputed neighbour masks through the back-edge table, and the general case
 * recomputes per-edge masks from the raw packed rows.
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ __forceinline__ TargetMask<WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>::kWords> buildCandidates(
  const WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>&                     shared,
  int                                                                       warp,
  int                                                                       depth,
  const unsigned char*                                                      mapping,
  const TargetMask<WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>::kWords>& used,
  const QueryMoleculeView&                                                  query,
  bool                                                                      uniform,
  bool                                                                      dual,
  uint64_t                                                                  prevOnlyPlain,
  uint64_t                                                                  prevOnlyRing,
  int                                                                       prevTargetAtom) {
  constexpr int kWords = WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>::kWords;
  using Mask           = TargetMask<kWords>;

  Mask candidates = shared.lm[warp][depth];
  candidates.andNotEq(used);

  const uint64_t depthBit = 1ULL << depth;
  if (prevOnlyPlain & depthBit) {
    candidates.andEq(shared.tn8[warp][prevTargetAtom]);
    return candidates;
  }
  if (prevOnlyRing & depthBit) {
    candidates.andEq(shared.tb8[warp][prevTargetAtom]);
    return candidates;
  }

  const int backEdges = shared.bcnt[warp][depth];

  if (uniform) {
    for (int k = 0; k < backEdges && !candidates.empty(); ++k) {
      candidates.andEq(shared.tn8[warp][mapping[shared.bej[warp][depth][k] & 63]]);
    }
    return candidates;
  }

  if (dual) {
    for (int k = 0; k < backEdges && !candidates.empty(); ++k) {
      const uint32_t edge   = shared.bej[warp][depth][k];
      const int      mapped = mapping[edge & 63u];
      candidates.andEq((edge & 64u) ? shared.tb8[warp][mapped] : shared.tn8[warp][mapped]);
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
      candidates.andEq(
        tnbrMask<kWords>(shared.tn8[warp][mapped].lo, shared.tb8[warp][mapped].lo, queryBonds.matchMask[slot]));
    }
  }
  return candidates;
}

// =============================================================================
// Per-pair setup
// =============================================================================

/**
 * @brief Read target atom @p atom 's row out of the packed label matrix.
 *
 * nvMolKit's `BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>` is row-major with
 * the target as the row and LSB-first bit packing, which is bit-for-bit the
 * layout the reference's transpose expects.
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
 * @brief Transpose the pair's label matrix into per-query-atom target masks.
 *
 * @return true if every query atom has at least one candidate target atom.
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ __forceinline__ bool transposeLabelMatrix(WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>& shared,
                                                     int                                             warp,
                                                     int                                             lane,
                                                     const uint32_t*                                 labelWords,
                                                     int                                             numTargetAtoms,
                                                     int                                             numQueryAtoms) {
  constexpr int kRoots = WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>::kRoots;
  constexpr int kWords = WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>::kWords;

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

    if (lane == 0) {
      TargetMask<kWords> mask;
      mask.clear();
      mask.lo = static_cast<uint64_t>(ballots[0]);
      if constexpr (kRoots > 1) {
        mask.lo |= static_cast<uint64_t>(ballots[1]) << 32;
      }
      if constexpr (kRoots > 2) {
        mask.hi = static_cast<uint64_t>(ballots[2]) | (static_cast<uint64_t>(ballots[3]) << 32);
      }
      shared.lm[warp][depth] = mask;
    }
  }

  return anyEmptyColumn == 0u;
}

/**
 * @brief Build the back-edge tables and classify the query's bond masks.
 *
 * Fills `bej`/`bcnt` with each query atom's earlier neighbours, decides whether
 * every back edge carries the same bond mask (uniform), only two distinct masks
 * (dual), or more (general), and reports the depths whose only back edge goes to
 * the immediately preceding depth.
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ __forceinline__ void classifyQueryBonds(WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>& shared,
                                                   int                                             warp,
                                                   int                                             lane,
                                                   const QueryMoleculeView&                        query,
                                                   int                                             numQueryAtoms,
                                                   bool&                                           uniform,
                                                   bool&                                           dual,
                                                   uint32_t&                                       groupMin,
                                                   uint32_t&                                       groupMax,
                                                   uint64_t&                                       prevOnlyPlain,
                                                   uint64_t&                                       prevOnlyRing) {
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
        const uint32_t mask            = bonds.matchMask[slot];
        shared.bej[warp][depth][count] = static_cast<unsigned char>(neighbor);
        laneMin                        = min(laneMin, mask);
        laneMax                        = max(laneMax, mask);
        ++count;
      }
    }
    shared.bcnt[warp][depth] = static_cast<unsigned char>(count);
  }

  groupMin = warpReduceMin(laneMin);
  groupMax = warpReduceMax(laneMax);
  uniform  = (groupMin == groupMax);
  __syncwarp();

  uint32_t offDual = 0;
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
          if (mask != groupMin && mask != groupMax) {
            offDual = 1;
          }
          if (mask == groupMax) {
            shared.bej[warp][depth][count] |= 64u;
          }
          ++count;
        }
      }
    }
  }
  dual = !uniform && (__ballot_sync(kFullWarpMask, offDual) == 0u);
  __syncwarp();

  prevOnlyPlain = 0;
  prevOnlyRing  = 0;
  if (uniform || dual) {
    uint64_t lanePlain = 0;
    uint64_t laneRing  = 0;
    for (int depth = lane; depth < numQueryAtoms; depth += 32) {
      if (depth >= 1 && shared.bcnt[warp][depth] == 1) {
        const uint32_t edge = shared.bej[warp][depth][0];
        if ((edge & 63u) == static_cast<uint32_t>(depth - 1)) {
          if (edge & 64u) {
            laneRing |= 1ULL << depth;
          } else {
            lanePlain |= 1ULL << depth;
          }
        }
      }
    }
    prevOnlyPlain = static_cast<uint64_t>(warpReduceOr(static_cast<uint32_t>(lanePlain))) |
                    (static_cast<uint64_t>(warpReduceOr(static_cast<uint32_t>(lanePlain >> 32))) << 32);
    prevOnlyRing = static_cast<uint64_t>(warpReduceOr(static_cast<uint32_t>(laneRing))) |
                   (static_cast<uint64_t>(warpReduceOr(static_cast<uint32_t>(laneRing >> 32))) << 32);
  }
}

/**
 * @brief Fill the per-warp target adjacency masks for the classified bond case.
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ __forceinline__ void buildTargetAdjacency(WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>& shared,
                                                     int                                             warp,
                                                     int                                             lane,
                                                     const TargetMoleculeView&                       target,
                                                     int                                             numTargetAtoms,
                                                     bool                                            uniform,
                                                     bool                                            dual,
                                                     uint32_t                                        groupMin,
                                                     uint32_t                                        groupMax) {
  constexpr int kWords = WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>::kWords;

  for (int atom = lane; atom < numTargetAtoms; atom += 32) {
    uint64_t neighbors;
    uint64_t bondInfo;
    packTargetRow(target.targetAtomBonds[atom], neighbors, bondInfo);

    if (uniform) {
      shared.tn8[warp][atom] = tnbrMask<kWords>(neighbors, bondInfo, groupMax);
    } else if (dual) {
      shared.tn8[warp][atom] = tnbrMask<kWords>(neighbors, bondInfo, groupMin);
      shared.tb8[warp][atom] = tnbrMask<kWords>(neighbors, bondInfo, groupMax);
    } else {
      TargetMask<kWords> rawNeighbors;
      TargetMask<kWords> rawBondInfo;
      rawNeighbors.clear();
      rawBondInfo.clear();
      rawNeighbors.lo        = neighbors;
      rawBondInfo.lo         = bondInfo;
      shared.tn8[warp][atom] = rawNeighbors;
      shared.tb8[warp][atom] = rawBondInfo;
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
  constexpr int kWords = WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>::kWords;
  constexpr int kRoots = WarpSharedState<MaxTargetAtoms, MaxQueryAtoms>::kRoots;
  using Mask           = TargetMask<kWords>;

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

  if (!transposeLabelMatrix<MaxTargetAtoms,
                            MaxQueryAtoms>(shared, warp, lane, labelWords, numTargetAtoms, numQueryAtoms)) {
    writeEmptyResult();
    return;
  }

  bool     uniform       = false;
  bool     dual          = false;
  uint32_t groupMin      = 0;
  uint32_t groupMax      = 0;
  uint64_t prevOnlyPlain = 0;
  uint64_t prevOnlyRing  = 0;
  classifyQueryBonds<MaxTargetAtoms, MaxQueryAtoms>(shared,
                                                    warp,
                                                    lane,
                                                    query,
                                                    numQueryAtoms,
                                                    uniform,
                                                    dual,
                                                    groupMin,
                                                    groupMax,
                                                    prevOnlyPlain,
                                                    prevOnlyRing);

  buildTargetAdjacency<MaxTargetAtoms,
                       MaxQueryAtoms>(shared, warp, lane, target, numTargetAtoms, uniform, dual, groupMin, groupMax);
  __syncwarp();

  const int  limit     = out.maxMatchesToFind;
  const bool hasLimit  = limit >= 0;
  const int  lastDepth = numQueryAtoms - 1;

  // Roots owned by this lane: bit positions lane, lane+32, lane+64, lane+96.
  Mask rootMask;
  rootMask.clear();
#pragma unroll
  for (int k = 0; k < kRoots; ++k) {
    rootMask.set(lane + 32 * k);
  }
  Mask roots = shared.lm[warp][0];
  roots.andEq(rootMask);

  uint32_t laneTotal = 0;

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

  if (lastDepth == 0) {
    // Single-atom query: every candidate root is a complete embedding.
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
  } else {
    Mask          remaining[MaxQueryAtoms];
    unsigned char mapping[MaxQueryAtoms];

    while (!roots.empty()) {
      const int rootAtom = roots.lowest();
      roots.clearLowest();

      Mask used;
      used.clear();
      used.set(rootAtom);
      mapping[0] = static_cast<unsigned char>(rootAtom);

      int depth     = 1;
      remaining[1]  = buildCandidates<MaxTargetAtoms, MaxQueryAtoms>(shared,
                                                                    warp,
                                                                    1,
                                                                    mapping,
                                                                    used,
                                                                    query,
                                                                    uniform,
                                                                    dual,
                                                                    prevOnlyPlain,
                                                                    prevOnlyRing,
                                                                    rootAtom);
      bool rootDone = false;
      bool pairDone = false;

      while (depth >= 1) {
        if (depth == lastDepth) {
          if constexpr (Mode == DfsOutputMode::Count) {
            laneTotal += static_cast<uint32_t>(remaining[depth].popcount());
            // A lane that has already reached the pair limit on its own roots
            // cannot change the clamped answer, so it stops. Lanes cannot see
            // each other's partial totals mid-search without a warp collective
            // in divergent code, so the other lanes run to exhaustion.
            if (hasLimit && laneTotal >= static_cast<uint32_t>(limit)) {
              pairDone = true;
            }
          } else if constexpr (Mode == DfsOutputMode::Paint) {
            if (!remaining[depth].empty()) {
              atomicOr(&out.paint.recursiveBits[out.paint.outputPairIdx * out.paint.maxTargetAtoms + rootAtom],
                       1u << out.paint.patternId);
              rootDone = true;
            }
          } else {
            Mask terminal = remaining[depth];
            while (!terminal.empty()) {
              const int terminalAtom = terminal.lowest();
              terminal.clearLowest();
              if (emitMapping(mapping, terminalAtom)) {
                pairDone = true;
                break;
              }
            }
          }
          remaining[depth].clear();
          if (rootDone || pairDone) {
            break;
          }
        }

        if (remaining[depth].empty()) {
          --depth;
          if (depth >= 1) {
            used.reset(mapping[depth]);
          }
          continue;
        }

        const int candidate = remaining[depth].lowest();
        remaining[depth].clearLowest();
        mapping[depth] = static_cast<unsigned char>(candidate);
        used.set(candidate);
        ++depth;
        remaining[depth] = buildCandidates<MaxTargetAtoms, MaxQueryAtoms>(shared,
                                                                          warp,
                                                                          depth,
                                                                          mapping,
                                                                          used,
                                                                          query,
                                                                          uniform,
                                                                          dual,
                                                                          prevOnlyPlain,
                                                                          prevOnlyRing,
                                                                          candidate);
      }

      if (pairDone) {
        break;
      }
    }
  }

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
    __syncwarp();
    if (lane == 0) {
      out.matchCounts[resultIdx] = shared.matchCount[warp];
      if (!out.countOnly && out.reportedCounts != nullptr) {
        out.reportedCounts[resultIdx] = shared.reportedCount[warp];
      }
    }
  }
}

}  // namespace dfs
}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_DFS_CUH
