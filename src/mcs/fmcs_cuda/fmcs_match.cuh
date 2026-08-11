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

// Seed-in-target matching for the fMCS kernel.
//
// The kernel runs one block per pair (see fmcs_kernel.cuh); this header owns
// the block-shared pair state and the matcher that runs against it.
// Everything invariant for the pair -- both CSRs (byte-packed), the atom/bond
// compatibility rows as bitsets, per-atom incident-bond sets, degree buckets,
// and per-class target adjacency tables -- is loaded or derived once by
// @ref loadPairSharedCooperative and then shared by every grow group.  Per
// seed, only the most-constrained-first search order and its back-edge table
// are built (warp-parallel).
//
// The exact check is the shared lane-parallel DFS core
// (nvMolKit::dfsFromRoots, the same engine the substructure search product
// runs on), fed by candidate masks read from the per-class adjacency tables
// (src/subgraph/class_adjacency.cuh) -- the K-class generalisation of the
// substructure frontend's Uniform/Dual precompute, keyed here on
// deduplicated bond-compatibility rows.
//
// A greedy incremental extension of the parent's recorded witness runs
// first; a greedy miss is inconclusive and falls through to the exact
// search.

#ifndef FMCS_CUDA_FMCS_MATCH_CUH
#define FMCS_CUDA_FMCS_MATCH_CUH

#include <cstdint>

#include "src/mcs/fmcs_cuda/fmcs.cuh"
#include "src/mcs/fmcs_cuda/fmcs_match_tables.cuh"
#include "src/mcs/fmcs_cuda/fmcs_seed.cuh"
#include "src/mcs/fmcs_cuda/fmcs_seed_queue.cuh"
#include "src/mcs/fmcs_cuda/fmcs_topology.cuh"
#include "src/subgraph/class_adjacency.cuh"
#include "src/subgraph/target_mask.cuh"
#include "src/subgraph/warp_dfs.cuh"

namespace mcs {
namespace fmcs {

/// Largest seed-atom degree the degreeAtLeast buckets distinguish.  Degrees
/// above this clamp to the last bucket, which only weakens the filter.
constexpr int kFmcsMaxSeedDegree  = kMaxNeighborsPerAtom;
/// Back-edge slots per search depth; molecular degree is capped at 8.
constexpr int kFmcsMaxBackEdges   = kMaxNeighborsPerAtom;
/// Bond-compatibility classes precomputed per pair.  Query bonds beyond this
/// many distinct compatibility rows fall back to a shared-CSR recompute in
/// the candidate build.
constexpr int kFmcsMaxBondClasses = 8;
/// Cooperative-group (warp) size for the grow groups and the matcher.
constexpr int kFmcsGroupSize      = 32;

/// Bitset wide enough for a tier's index space (query and target sides are
/// square).  Tiers 16 and 32 share the 32-bit form.
template <int maxTargetAtoms>
constexpr int kFmcsMaskAtoms = (maxTargetAtoms <= 32 ? 32 : (maxTargetAtoms <= 64 ? 64 : 128));

template <int maxTargetAtoms> using FmcsTargetMask = nvMolKit::TargetMask<kFmcsMaskAtoms<maxTargetAtoms>>;

/// Per-tier block configuration.  Tiers up to 64 run sixteen grow groups and
/// cache per-depth candidate masks; tier 128 halves the group count and
/// recomputes depth candidates on the fly so the block state fits the 64 KB
/// opt-in shared-memory limit of the smallest supported architectures.
template <int maxAtoms> struct FmcsKernelConfig {
  static constexpr bool cacheDepthCandidates = maxAtoms <= 64;
  static constexpr int  numGroups            = maxAtoms <= 64 ? 16 : 8;
  static constexpr int  blockThreads         = numGroups * kFmcsGroupSize;
};

// ---------------------------------------------------------------------------
// TargetMask <-> Seed/MatchResult bit-word bridging.  Seed and MatchResult
// keep their word-array layouts (shared with the rest of fmcs_cuda and its
// tests); the matcher does its bitset math on TargetMask and converts at the
// boundaries.  All forms are layout-compatible 32/64/2x64-bit words.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t maskWord32(const nvMolKit::TargetMask<32>& m, int /*word*/) {
  return m.bits;
}
__device__ __forceinline__ uint32_t maskWord32(const nvMolKit::TargetMask<64>& m, int word) {
  return static_cast<uint32_t>(m.lo >> (32 * word));
}
__device__ __forceinline__ uint32_t maskWord32(const nvMolKit::TargetMask<128>& m, int word) {
  return static_cast<uint32_t>((word < 2 ? m.lo : m.hi) >> (32 * (word & 1)));
}

__device__ __forceinline__ bool maskEquals(const nvMolKit::TargetMask<32>& a, const nvMolKit::TargetMask<32>& b) {
  return a.bits == b.bits;
}
__device__ __forceinline__ bool maskEquals(const nvMolKit::TargetMask<64>& a, const nvMolKit::TargetMask<64>& b) {
  return a.lo == b.lo;
}
__device__ __forceinline__ bool maskEquals(const nvMolKit::TargetMask<128>& a, const nvMolKit::TargetMask<128>& b) {
  return a.lo == b.lo && a.hi == b.hi;
}

__device__ __forceinline__ void maskOrEq(nvMolKit::TargetMask<32>& a, const nvMolKit::TargetMask<32>& b) {
  a.bits |= b.bits;
}
__device__ __forceinline__ void maskOrEq(nvMolKit::TargetMask<64>& a, const nvMolKit::TargetMask<64>& b) {
  a.lo |= b.lo;
}
__device__ __forceinline__ void maskOrEq(nvMolKit::TargetMask<128>& a, const nvMolKit::TargetMask<128>& b) {
  a.lo |= b.lo;
  a.hi |= b.hi;
}

template <class GroupT>
__device__ __forceinline__ nvMolKit::TargetMask<32> warpOrMask(const GroupT& group, nvMolKit::TargetMask<32> m) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    m.bits |= group.shfl_xor(m.bits, off);
  return m;
}
template <class GroupT>
__device__ __forceinline__ nvMolKit::TargetMask<64> warpOrMask(const GroupT& group, nvMolKit::TargetMask<64> m) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    m.lo |= group.shfl_xor(m.lo, off);
  return m;
}
template <class GroupT>
__device__ __forceinline__ nvMolKit::TargetMask<128> warpOrMask(const GroupT& group, nvMolKit::TargetMask<128> m) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    m.lo |= group.shfl_xor(m.lo, off);
    m.hi |= group.shfl_xor(m.hi, off);
  }
  return m;
}

/// Load a Seed/MatchResult bit-word array into the tier's TargetMask form.
template <std::size_t MaskBits, class Word, int NumWords>
__device__ __forceinline__ nvMolKit::TargetMask<MaskBits> maskFromWords(const Word (&words)[NumWords]) {
  nvMolKit::TargetMask<MaskBits> m;
  m.clear();
  constexpr int kSub = static_cast<int>(sizeof(Word) / 4);
#pragma unroll
  for (int w = 0; w < NumWords; ++w)
#pragma unroll
    for (int k = 0; k < kSub; ++k)
      m.setWord32(w * kSub + k, static_cast<uint32_t>(words[w] >> (32 * k)));
  return m;
}

/// Store a TargetMask back into a Seed/MatchResult bit-word array.
template <class Word, int NumWords, std::size_t MaskBits>
__device__ __forceinline__ void maskToWords(const nvMolKit::TargetMask<MaskBits>& m, Word (&words)[NumWords]) {
  constexpr int kSub = static_cast<int>(sizeof(Word) / 4);
#pragma unroll
  for (int w = 0; w < NumWords; ++w) {
    Word v = 0;
#pragma unroll
    for (int k = 0; k < kSub; ++k)
      v |= static_cast<Word>(maskWord32(m, w * kSub + k)) << (32 * k);
    words[w] = v;
  }
}

template <class Word, int NumWords> __device__ __forceinline__ bool wordsTest(const Word (&words)[NumWords], int bit) {
  constexpr int kBits = static_cast<int>(sizeof(Word) * 8);
  return ((words[bit / kBits] >> (bit % kBits)) & 1) != 0;
}

template <class Word, int NumWords> __device__ __forceinline__ void wordsSet(Word (&words)[NumWords], int bit) {
  constexpr int kBits = static_cast<int>(sizeof(Word) * 8);
  words[bit / kBits] |= static_cast<Word>(1) << (bit % kBits);
}

/// Mask with bits [0, bit] set; clear mask for bit < 0.
template <std::size_t MaskBits> __device__ __forceinline__ nvMolKit::TargetMask<MaskBits> lowMaskThrough(int bit) {
  nvMolKit::TargetMask<MaskBits> m;
  m.clear();
#pragma unroll
  for (int k = 0; k * 32 < static_cast<int>(MaskBits); ++k) {
    const int lowBit = k * 32;
    uint32_t  v      = 0;
    if (bit >= lowBit + 31) {
      v = 0xFFFFFFFFu;
    } else if (bit >= lowBit) {
      v = (1u << (bit - lowBit + 1)) - 1u;
    }
    m.setWord32(k, v);
  }
  return m;
}

// ---------------------------------------------------------------------------
// Block-shared per-pair state.
// ---------------------------------------------------------------------------

template <int maxAtoms, int maxBonds> struct FmcsPairShared {
  using Config   = FmcsKernelConfig<maxAtoms>;
  using SeedT    = Seed<maxAtoms, maxBonds>;
  using MatchT   = MatchResult<maxAtoms, maxBonds, maxAtoms, maxBonds>;
  using QueuedT  = QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>;
  using AtomMask = FmcsTargetMask<maxAtoms>;
  using BondMask = FmcsTargetMask<maxBonds>;

  static constexpr int kNumGroups = Config::numGroups;
  static constexpr int kAtomRegs  = (maxAtoms + 31) / 32;

  // --- per-group hot state (16B-aligned members first) ---
  QueuedT current[kNumGroups];
  QueuedT biggest[kNumGroups];
  QueuedT best;

  // --- pair invariants ---
  AtomMask atomRow[maxAtoms];        ///< query atom -> compatible target atoms
  BondMask incidentBonds[maxAtoms];  ///< query atom -> incident query bonds
  BondMask bondRow[maxBonds];        ///< query bond -> compatible target bonds
  AtomMask degreeAtLeast[kFmcsMaxSeedDegree + 1];
  BondMask classRep[kFmcsMaxBondClasses];  ///< representative bondRow per class
  /// Per-class target adjacency, shared with src/subgraph.
  nvMolKit::subgraph::ClassAdjacency<kFmcsMaskAtoms<maxAtoms>, kFmcsMaxBondClasses> classAdj;
  BondMask                                                                          phase1Matched;

  /// Per-depth candidate masks; recomputed on the fly at tier 128 (see
  /// FmcsKernelConfig::cacheDepthCandidates), where caching them would push
  /// the block state past the smallest supported shared-memory limit.
  AtomMask depthCandidates[kNumGroups][Config::cacheDepthCandidates ? maxAtoms : 1];

  /// Back edge: low byte = earlier order position, high byte = query bond.
  uint16_t backEdges[kNumGroups][maxAtoms][kFmcsMaxBackEdges];
  NewBond  newBonds[kNumGroups][maxBonds];

  // Byte-packed CSRs for both graphs.
  uint8_t qRow[maxAtoms + 1], tRow[maxAtoms + 1];
  uint8_t qCol[2 * maxBonds], qBnd[2 * maxBonds], tCol[2 * maxBonds], tBnd[2 * maxBonds];
  uint8_t qEpU[maxBonds], qEpV[maxBonds], tEpU[maxBonds], tEpV[maxBonds];
  uint8_t bondClass[maxBonds];
  /// Phase-1 per-query-bond direct match results.
  uint8_t phase1TargetBond[maxBonds], phase1AtomU[maxBonds], phase1AtomV[maxBonds];

  // Per-group per-seed search tables.
  uint8_t order[kNumGroups][maxAtoms];     ///< order position -> query atom
  uint8_t orderPos[kNumGroups][maxAtoms];  ///< inverse of order
  uint8_t seedDegree[kNumGroups][maxAtoms];
  uint8_t targetForQuery[kNumGroups][maxAtoms];  ///< winning embedding
  uint8_t backEdgeCount[kNumGroups][maxAtoms];

  SeedQueue<QueuedT, ThreadBlockScope> queue;

  int  newBondCount[kNumGroups];
  bool popped[kNumGroups];
  int  found[kNumGroups];

  unsigned int       bestScore;
  int                bestCopyLock;
  bool               overflowed;
  bool               timedOut;
  bool               phase2Done;
  unsigned long long startClock;
  int                qNA, qNB, tNA, tNB, numClasses, phase1Count;
};

// ---------------------------------------------------------------------------
// Per-pair shared-state preload.  Block-cooperative: every thread of the
// block participates, striding by @p blockThreads; the caller must
// __syncthreads() after (the final barrier is included here).
// ---------------------------------------------------------------------------
template <int maxAtoms, int maxBonds>
__device__ void loadPairSharedCooperative(FmcsPairShared<maxAtoms, maxBonds>& S,
                                          const DeviceCsrView&                queryView,
                                          const DeviceCsrView&                targetView,
                                          const PairMatchTablesDevice&        tables,
                                          int                                 tid,
                                          int                                 blockThreads) {
  using AtomMask               = typename FmcsPairShared<maxAtoms, maxBonds>::AtomMask;
  using BondMask               = typename FmcsPairShared<maxAtoms, maxBonds>::BondMask;
  constexpr int kAtomMaskWords = static_cast<int>(sizeof(AtomMask) / 4);
  constexpr int kBondMaskWords = static_cast<int>(sizeof(BondMask) / 4);

  const int qNA = queryView.numAtoms;
  const int qNB = queryView.numBonds;
  const int tNA = targetView.numAtoms;
  const int tNB = targetView.numBonds;
  const int qE  = static_cast<int>(queryView.rowOffsets[qNA]);
  const int tE  = static_cast<int>(targetView.rowOffsets[tNA]);

  if (tid == 0) {
    S.qNA          = qNA;
    S.qNB          = qNB;
    S.tNA          = tNA;
    S.tNB          = tNB;
    S.bestScore    = 0;
    S.bestCopyLock = 0;
    S.overflowed   = false;
    S.timedOut     = false;
    S.phase2Done   = false;
  }

  for (int i = tid; i <= qNA; i += blockThreads)
    S.qRow[i] = static_cast<uint8_t>(queryView.rowOffsets[i]);
  for (int i = tid; i <= tNA; i += blockThreads)
    S.tRow[i] = static_cast<uint8_t>(targetView.rowOffsets[i]);
  for (int i = tid; i < qE; i += blockThreads) {
    S.qCol[i] = static_cast<uint8_t>(queryView.colIndices[i]);
    S.qBnd[i] = static_cast<uint8_t>(queryView.bondIndices[i]);
  }
  for (int i = tid; i < tE; i += blockThreads) {
    S.tCol[i] = static_cast<uint8_t>(targetView.colIndices[i]);
    S.tBnd[i] = static_cast<uint8_t>(targetView.bondIndices[i]);
  }
  for (int i = tid; i < qNB; i += blockThreads) {
    const std::uint32_t e = queryView.bondEndpoints[i];
    S.qEpU[i]             = static_cast<uint8_t>(e >> kBondEndpointShift);
    S.qEpV[i]             = static_cast<uint8_t>(e & kBondEndpointMask);
    BondMask row;
    row.clear();
#pragma unroll
    for (int w = 0; w < kBondMaskWords; ++w)
      if (w < tables.bonds.wordsPerRow)
        row.setWord32(w, tables.bonds.data[static_cast<size_t>(i) * tables.bonds.wordsPerRow + w]);
    S.bondRow[i] = row;
  }
  for (int i = tid; i < tNB; i += blockThreads) {
    const std::uint32_t e = targetView.bondEndpoints[i];
    S.tEpU[i]             = static_cast<uint8_t>(e >> kBondEndpointShift);
    S.tEpV[i]             = static_cast<uint8_t>(e & kBondEndpointMask);
  }
  for (int i = tid; i < qNA; i += blockThreads) {
    AtomMask row;
    row.clear();
#pragma unroll
    for (int w = 0; w < kAtomMaskWords; ++w)
      if (w < tables.atoms.wordsPerRow)
        row.setWord32(w, tables.atoms.data[static_cast<size_t>(i) * tables.atoms.wordsPerRow + w]);
    S.atomRow[i] = row;
  }
  __syncthreads();

  // Derived per-atom bitsets.
  for (int a = tid; a < qNA; a += blockThreads) {
    BondMask inc;
    inc.clear();
    const int b0 = S.qRow[a], b1 = S.qRow[a + 1];
    for (int e = b0; e < b1; ++e)
      inc.set(S.qBnd[e]);
    S.incidentBonds[a] = inc;
  }
  if (tid <= kFmcsMaxSeedDegree) {
    AtomMask m;
    m.clear();
    for (int a = 0; a < tNA; ++a) {
      const int d = S.tRow[a + 1] - S.tRow[a];
      if (d >= tid)
        m.set(a);
    }
    S.degreeAtLeast[tid] = m;
  }
  // Bond classes: query bonds with identical compatibility rows share a
  // class, which is the "equal keys mean identical predicates" contract of
  // the subgraph frontend.  One thread; qNB <= 128.
  if (tid == 0) {
    int nc = 0;
    for (int b = 0; b < qNB; ++b) {
      const BondMask row = S.bondRow[b];
      int            c   = kFmcsMaxBondClasses;
      for (int k = 0; k < nc; ++k)
        if (maskEquals(S.classRep[k], row)) {
          c = k;
          break;
        }
      if (c == kFmcsMaxBondClasses && nc < kFmcsMaxBondClasses) {
        S.classRep[nc] = row;
        c              = nc;
        ++nc;
      }
      S.bondClass[b] = static_cast<uint8_t>(c);
    }
    S.numClasses = nc;
  }
  __syncthreads();

  // Per-class target adjacency, filled through the shared subgraph component.
  nvMolKit::subgraph::fillClassAdjacency(S.classAdj, S.numClasses, tNA, tid, blockThreads, [&](int atom, int cls) {
    AtomMask m;
    m.clear();
    const BondMask row = S.classRep[cls];
    const int      b0 = S.tRow[atom], b1 = S.tRow[atom + 1];
    for (int e = b0; e < b1; ++e)
      if (row.test(S.tBnd[e]))
        m.set(S.tCol[e]);
    return m;
  });
  __syncthreads();
}

// ---------------------------------------------------------------------------
// Phase-1 direct single-bond match.
// ---------------------------------------------------------------------------

/// Resolved (target bond, endpoint) assignment for a successful single-bond
/// match.  @c targetAtomU is the target atom the query bond's u endpoint
/// mapped to (likewise V).  Contents are unspecified on failure.
struct SingleBondMatch {
  uint8_t targetAtomU;
  uint8_t targetAtomV;
  uint8_t targetBond;
};

/// Within-thread: find the first target bond that embeds query bond
/// @p queryBondIdx in either orientation.  Whether a one-bond seed embeds
/// depends only on its bond, so phase 1 tests every query bond with one call
/// each, in parallel.
template <int maxAtoms, int maxBonds>
__device__ __forceinline__ bool matchSingleBondWithinThread(const FmcsPairShared<maxAtoms, maxBonds>& S,
                                                            const int                                 queryBondIdx,
                                                            SingleBondMatch&                          outMatch) {
  using BondMask = typename FmcsPairShared<maxAtoms, maxBonds>::BondMask;
  const int u = S.qEpU[queryBondIdx], v = S.qEpV[queryBondIdx];
  BondMask  row = S.bondRow[queryBondIdx];
  while (!row.empty()) {
    const int j = row.lowest();
    row.clearLowest();
    const int tu = S.tEpU[j], tv = S.tEpV[j];
    if (S.atomRow[u].test(tu) && S.atomRow[v].test(tv)) {
      outMatch.targetAtomU = static_cast<uint8_t>(tu);
      outMatch.targetAtomV = static_cast<uint8_t>(tv);
      outMatch.targetBond  = static_cast<uint8_t>(j);
      return true;
    }
    if (S.atomRow[u].test(tv) && S.atomRow[v].test(tu)) {
      outMatch.targetAtomU = static_cast<uint8_t>(tv);
      outMatch.targetAtomV = static_cast<uint8_t>(tu);
      outMatch.targetBond  = static_cast<uint8_t>(j);
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Remaining-size bound (RDKit canGrowBiggerThan support), warp-parallel BFS
// over the shared query CSR.
// ---------------------------------------------------------------------------
template <int maxAtoms, int maxBonds, class GroupT>
__device__ __forceinline__ void seedComputeRemainingSizeCooperative(const GroupT&                       group,
                                                                    FmcsPairShared<maxAtoms, maxBonds>& S,
                                                                    Seed<maxAtoms, maxBonds>&           seed) {
  using AtomMask          = typename FmcsPairShared<maxAtoms, maxBonds>::AtomMask;
  using BondMask          = typename FmcsPairShared<maxAtoms, maxBonds>::BondMask;
  constexpr int kAtomRegs = FmcsPairShared<maxAtoms, maxBonds>::kAtomRegs;

  const int lane  = static_cast<int>(group.thread_rank());
  AtomMask  visA  = maskFromWords<kFmcsMaskAtoms<maxAtoms>>(seed.atoms);
  BondMask  visB  = maskFromWords<kFmcsMaskAtoms<maxBonds>>(seed.excludedBonds);
  AtomMask  front = maskFromWords<kFmcsMaskAtoms<maxAtoms>>(seed.lastAddedAtoms);
  const int baseA = visA.popcount();
  const int baseB = visB.popcount();

  while (!front.empty()) {
    BondMask bondAcc;
    AtomMask candAcc;
    bondAcc.clear();
    candAcc.clear();
#pragma unroll
    for (int r = 0; r < kAtomRegs; ++r) {
      const int a = lane + r * 32;
      if (a < maxAtoms && front.test(a)) {
        const int b0 = S.qRow[a], b1 = S.qRow[a + 1];
        for (int e = b0; e < b1; ++e) {
          const int b = S.qBnd[e];
          if (visB.test(b))
            continue;
          bondAcc.set(b);
          candAcc.set(S.qCol[e]);
        }
      }
    }
    bondAcc = warpOrMask(group, bondAcc);
    candAcc = warpOrMask(group, candAcc);
    maskOrEq(visB, bondAcc);
    AtomMask newFront = candAcc;
    newFront.andNotEq(visA);
    maskOrEq(visA, newFront);
    front = newFront;
  }
  if (lane == 0) {
    seed.remainingBonds = static_cast<uint16_t>(visB.popcount() - baseB);
    seed.remainingAtoms = static_cast<uint16_t>(visA.popcount() - baseA);
  }
  group.sync();
}

// ---------------------------------------------------------------------------
// Exact seed-in-target substructure check.
// ---------------------------------------------------------------------------

/// General-fallback candidate term for a query bond beyond the class-table
/// capacity: target atoms reachable from @p targetAtom over a target bond
/// the query bond's compatibility row accepts.  Reads only shared memory.
template <int maxAtoms, int maxBonds>
__device__ __forceinline__ typename FmcsPairShared<maxAtoms, maxBonds>::AtomMask
           neighborsMatchingSlow(const FmcsPairShared<maxAtoms, maxBonds>& S, int targetAtom, int queryBond) {
  using AtomMask = typename FmcsPairShared<maxAtoms, maxBonds>::AtomMask;
  AtomMask m;
  m.clear();
  const auto& row = S.bondRow[queryBond];
  const int   b0 = S.tRow[targetAtom], b1 = S.tRow[targetAtom + 1];
  for (int e = b0; e < b1; ++e)
    if (row.test(S.tBnd[e]))
      m.set(S.tCol[e]);
  return m;
}

/// The label+degree candidate term for search depth @p d of group @p g.
template <int maxAtoms, int maxBonds>
__device__ __forceinline__ typename FmcsPairShared<maxAtoms, maxBonds>::AtomMask
           depthCandidateTerm(const FmcsPairShared<maxAtoms, maxBonds>& S, int g, int d) {
  using Config = FmcsKernelConfig<maxAtoms>;
  if constexpr (Config::cacheDepthCandidates) {
    return S.depthCandidates[g][d];
  } else {
    const int a  = S.order[g][d];
    const int dg = S.seedDegree[g][a];
    auto      c  = S.atomRow[a];
    c.andEq(S.degreeAtLeast[dg > kFmcsMaxSeedDegree ? kFmcsMaxSeedDegree : dg]);
    return c;
  }
}

/// Build the most-constrained-first search order and back-edge table for
/// @p seed, warp-parallel.  Returns the number of seed atoms, or -1 if some
/// seed atom provably has no degree-compatible target.
template <int maxAtoms, int maxBonds, class GroupT>
__device__ int prepareSeedSearchCooperative(const GroupT&                       group,
                                            FmcsPairShared<maxAtoms, maxBonds>& S,
                                            const Seed<maxAtoms, maxBonds>&     seed,
                                            int                                 g) {
  using Config            = FmcsKernelConfig<maxAtoms>;
  using AtomMask          = typename FmcsPairShared<maxAtoms, maxBonds>::AtomMask;
  using BondMask          = typename FmcsPairShared<maxAtoms, maxBonds>::BondMask;
  constexpr int kAtomRegs = FmcsPairShared<maxAtoms, maxBonds>::kAtomRegs;

  const int      lane      = static_cast<int>(group.thread_rank());
  const int      qNA       = S.qNA;
  const BondMask seedBonds = maskFromWords<kFmcsMaskAtoms<maxBonds>>(seed.bonds);

  int      fail = 0;
  AtomMask neighborReg[kAtomRegs];
#pragma unroll
  for (int r = 0; r < kAtomRegs; ++r) {
    neighborReg[r].clear();
    const int a = lane + r * 32;
    if (a < qNA) {
      const bool inSeed = wordsTest(seed.atoms, a);
      S.orderPos[g][a]  = kUnmappedTargetIdx;
      int      deg      = 0;
      AtomMask ns;
      ns.clear();
      if (inSeed) {
        BondMask incident = S.incidentBonds[a];
        incident.andEq(seedBonds);
        deg          = incident.popcount();
        const int b0 = S.qRow[a], b1 = S.qRow[a + 1];
        for (int e = b0; e < b1; ++e)
          if (seedBonds.test(S.qBnd[e]))
            ns.set(S.qCol[e]);
        AtomMask cand = S.atomRow[a];
        cand.andEq(S.degreeAtLeast[deg > kFmcsMaxSeedDegree ? kFmcsMaxSeedDegree : deg]);
        if (cand.empty())
          fail = 1;
      }
      S.seedDegree[g][a] = static_cast<uint8_t>(deg);
      neighborReg[r]     = ns;
    }
  }
  group.sync();
  if (group.any(fail != 0))
    return -1;

  // Greedy order: at each position pick the unordered seed atom maximising
  // (mapped-neighbor count, seed degree, fewest candidates, lowest index),
  // packed into one warp-max key.  Same heuristic as before, evaluated
  // warp-parallel.
  const int n         = seed.numAtoms;
  AtomMask  unordered = maskFromWords<kFmcsMaskAtoms<maxAtoms>>(seed.atoms);
  AtomMask  ordered;
  ordered.clear();
  for (int d = 0; d < n; ++d) {
    unsigned bestKey = 0;
#pragma unroll
    for (int r = 0; r < kAtomRegs; ++r) {
      const int a = lane + r * 32;
      if (a < qNA && unordered.test(a)) {
        AtomMask mapped = neighborReg[r];
        mapped.andEq(ordered);
        const int mappedNeighbors = mapped.popcount();
        if (d == 0 || mappedNeighbors != 0) {
          const int deg  = S.seedDegree[g][a];
          AtomMask  cand = S.atomRow[a];
          cand.andEq(S.degreeAtLeast[deg > kFmcsMaxSeedDegree ? kFmcsMaxSeedDegree : deg]);
          const int      cc  = cand.popcount();
          const unsigned key = (static_cast<unsigned>(mappedNeighbors) << 24) | (static_cast<unsigned>(deg) << 20) |
                               (static_cast<unsigned>(128 - cc) << 8) | static_cast<unsigned>(127 - a) | 0x80u;
          bestKey = max(bestKey, key);
        }
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      bestKey = max(bestKey, group.shfl_xor(bestKey, off));
    int pick;
    if (bestKey == 0) {
      pick = unordered.lowest();  // disconnected fallback (unreached for connected seeds)
    } else {
      pick = 127 - static_cast<int>(bestKey & 0x7Fu);
    }
    if (lane == 0)
      S.order[g][d] = static_cast<uint8_t>(pick);
    S.orderPos[g][pick] = static_cast<uint8_t>(d);
    ordered.set(pick);
    unordered.reset(pick);
  }
  group.sync();

  // Per depth: candidate term (when cached) and deduplicated back edges.
  for (int d = lane; d < n; d += kFmcsGroupSize) {
    const int a = S.order[g][d];
    if constexpr (Config::cacheDepthCandidates) {
      const int dg   = S.seedDegree[g][a];
      AtomMask  cand = S.atomRow[a];
      cand.andEq(S.degreeAtLeast[dg > kFmcsMaxSeedDegree ? kFmcsMaxSeedDegree : dg]);
      S.depthCandidates[g][d] = cand;
    }
    int       cnt = 0;
    const int b0 = S.qRow[a], b1 = S.qRow[a + 1];
    for (int e = b0; e < b1 && cnt < kFmcsMaxBackEdges; ++e) {
      const int b = S.qBnd[e];
      if (!seedBonds.test(b))
        continue;
      const int o  = S.qCol[e];
      const int op = S.orderPos[g][o];
      if (op == kUnmappedTargetIdx || op >= d)
        continue;
      // Dedup by earlier order position, scanning the few recorded slots
      // (order positions exceed 63 at tier 128, so no bit-set shortcut).
      bool dup = false;
      for (int i = 0; i < cnt; ++i)
        if ((S.backEdges[g][d][i] & 0xFFu) == static_cast<unsigned>(op))
          dup = true;
      if (dup)
        continue;
      S.backEdges[g][d][cnt] = static_cast<uint16_t>(op | (b << 8));
      ++cnt;
    }
    S.backEdgeCount[g][d] = static_cast<uint8_t>(cnt);
  }
  group.sync();
  return n;
}

/// Candidate set for search depth @p d given the committed @p mapping: the
/// depth's label+degree term minus used targets, intersected per back edge
/// with the class-adjacency mask of the mapped earlier atom.
template <int maxAtoms, int maxBonds>
__device__ __forceinline__ typename FmcsPairShared<maxAtoms, maxBonds>::AtomMask seedCandidatesAt(
  const FmcsPairShared<maxAtoms, maxBonds>&                    S,
  int                                                          g,
  int                                                          d,
  const unsigned char*                                         mapping,
  const typename FmcsPairShared<maxAtoms, maxBonds>::AtomMask& used) {
  auto c = depthCandidateTerm(S, g, d);
  c.andNotEq(used);
  const int k = S.backEdgeCount[g][d];
  for (int i = 0; i < k && !c.empty(); ++i) {
    const uint16_t be  = S.backEdges[g][d][i];
    const int      m   = mapping[be & 0xFFu];
    const int      b   = be >> 8;
    const int      cls = S.bondClass[b];
    if (cls < kFmcsMaxBondClasses) {
      c.andEq(S.classAdj.neighbors[cls][m]);
    } else {
      c.andEq(neighborsMatchingSlow(S, m, b));
    }
  }
  return c;
}

template <int maxAtoms, int maxBonds, int maxTA, int maxTB>
__device__ __forceinline__ void matchResultClearCooperative(MatchResult<maxAtoms, maxBonds, maxTA, maxTB>& match,
                                                            int                                            lane) {
  for (int i = lane; i < maxAtoms; i += kFmcsGroupSize)
    match.targetAtomIdx[i] = kUnmappedTargetIdx;
  for (int i = lane; i < maxBonds; i += kFmcsGroupSize)
    match.targetBondIdx[i] = kUnmappedTargetIdx;
  if (lane == 0) {
    for (int i = 0; i < MatchResult<maxAtoms, maxBonds, maxTA, maxTB>::kTargetAtomWords; ++i)
      match.visitedTargetAtoms[i] = 0;
    for (int i = 0; i < MatchResult<maxAtoms, maxBonds, maxTA, maxTB>::kTargetBondWords; ++i)
      match.visitedTargetBonds[i] = 0;
    match.matchedAtomSize = 0;
    match.matchedBondSize = 0;
    match.empty           = true;
  }
  __syncwarp();
}

/// Lane-parallel DFS racing for one embedding via the shared core.  Lane L
/// owns roots L, L+32, ...; the first lane to complete an embedding wins
/// S.found[g] and records targetForQuery.  Losing lanes notice through the
/// per-root abort poll and a periodic poll inside the candidates callback.
template <int maxAtoms, int maxBonds, class GroupT>
__device__ void seedDfsSearchCooperative(const GroupT& group, FmcsPairShared<maxAtoms, maxBonds>& S, int g, int n) {
  using AtomMask = typename FmcsPairShared<maxAtoms, maxBonds>::AtomMask;
  const int lane = static_cast<int>(group.thread_rank());

  AtomMask laneStripe;
  laneStripe.clear();
  for (int a = lane; a < maxAtoms; a += kFmcsGroupSize)
    laneStripe.set(a);
  AtomMask roots = depthCandidateTerm(S, g, 0);
  roots.andEq(laneStripe);

  if (n == 1) {
    // No bonds to satisfy: the lowest compatible root is a complete
    // embedding.  Reduce lane-local lowest roots to the group minimum.
    int low = roots.empty() ? maxAtoms : roots.lowest();
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      low = min(low, group.shfl_xor(low, off));
    if (low >= maxAtoms)
      return;
    if (lane == 0) {
      S.targetForQuery[g][S.order[g][0]] = static_cast<uint8_t>(low);
      S.found[g]                         = 1;
    }
    group.sync();
    return;
  }

  int  poll         = 0;
  auto candidatesAt = [&](int depth, const unsigned char* mapping, const AtomMask& used, int /*prev*/) -> AtomMask {
    if (((++poll) & 127) == 0 && *(volatile int*)&S.found[g] != 0) {
      AtomMask empty;
      empty.clear();
      return empty;
    }
    return seedCandidatesAt(S, g, depth, mapping, used);
  };

  const int lastDepth  = n - 1;
  auto      onTerminal = [&](AtomMask terminals, const unsigned char* mapping) -> nvMolKit::DfsTerminalVerdict {
    nvMolKit::DfsTerminalVerdict verdict{false, false};
    if (!terminals.empty()) {
      if (atomicCAS(&S.found[g], 0, 1) == 0) {
        for (int p = 0; p < lastDepth; ++p)
          S.targetForQuery[g][S.order[g][p]] = mapping[p];
        S.targetForQuery[g][S.order[g][lastDepth]] = static_cast<uint8_t>(terminals.lowest());
      }
      verdict.laneDone = true;
    }
    return verdict;
  };

  auto abortRequested = [&]() -> bool { return *(volatile int*)&S.found[g] != 0; };

  nvMolKit::dfsFromRoots<maxAtoms>(roots, lastDepth, candidatesAt, onTerminal, abortRequested);
}

/// Rebuild the full (atom + bond) witness from the embedding recorded in
/// targetForQuery.
template <int maxAtoms, int maxBonds, class GroupT>
__device__ bool rebuildMatchCooperative(const GroupT&                                        group,
                                        FmcsPairShared<maxAtoms, maxBonds>&                  S,
                                        const Seed<maxAtoms, maxBonds>&                      seed,
                                        MatchResult<maxAtoms, maxBonds, maxAtoms, maxBonds>& match,
                                        int                                                  g) {
  using AtomMask = typename FmcsPairShared<maxAtoms, maxBonds>::AtomMask;
  using BondMask = typename FmcsPairShared<maxAtoms, maxBonds>::BondMask;
  const int lane = static_cast<int>(group.thread_rank());
  matchResultClearCooperative(match, lane);

  AtomMask visA;
  visA.clear();
  int okA = 1;
  for (int a = lane; a < maxAtoms; a += kFmcsGroupSize) {
    if (wordsTest(seed.atoms, a)) {
      const uint8_t t = S.targetForQuery[g][a];
      if (t == kUnmappedTargetIdx)
        okA = 0;
      else {
        match.targetAtomIdx[a] = t;
        visA.set(t);
      }
    }
  }
  visA = warpOrMask(group, visA);
  if (group.any(okA == 0) || visA.popcount() != seed.numAtoms) {
    matchResultClearCooperative(match, lane);
    return false;
  }

  BondMask visB;
  visB.clear();
  int okB = 1;
  for (int b = lane; b < maxBonds; b += kFmcsGroupSize) {
    if (wordsTest(seed.bonds, b)) {
      const int tu = S.targetForQuery[g][S.qEpU[b]];
      const int tv = S.targetForQuery[g][S.qEpV[b]];
      int       tb = -1;
      if (tu != kUnmappedTargetIdx && tv != kUnmappedTargetIdx) {
        const auto& row = S.bondRow[b];
        const int   e0 = S.tRow[tu], e1 = S.tRow[tu + 1];
        for (int e = e0; e < e1; ++e)
          if (S.tCol[e] == tv && row.test(S.tBnd[e])) {
            tb = S.tBnd[e];
            break;
          }
      }
      if (tb < 0)
        okB = 0;
      else {
        match.targetBondIdx[b] = static_cast<uint8_t>(tb);
        visB.set(tb);
      }
    }
  }
  visB = warpOrMask(group, visB);
  if (group.any(okB == 0) || visB.popcount() != seed.numBonds) {
    matchResultClearCooperative(match, lane);
    return false;
  }
  if (lane == 0) {
    maskToWords(visA, match.visitedTargetAtoms);
    maskToWords(visB, match.visitedTargetBonds);
    match.matchedAtomSize = seed.numAtoms;
    match.matchedBondSize = seed.numBonds;
    match.empty           = false;
  }
  group.sync();
  return true;
}

/// Exact seed-in-target substructure check: a lane-parallel DFS racing for
/// one embedding.  Exhaustive within bounded memory -- the stack is O(seed
/// atoms) of lane-local state -- so a false return proves the seed does not
/// embed in the target.  Group-uniform: all lanes of @p group call together.
template <int maxAtoms, int maxBonds, class GroupT>
__device__ __forceinline__ bool matchSeedSubstructureCooperative(
  const GroupT&                                        group,
  FmcsPairShared<maxAtoms, maxBonds>&                  S,
  const Seed<maxAtoms, maxBonds>&                      seed,
  MatchResult<maxAtoms, maxBonds, maxAtoms, maxBonds>& match,
  int                                                  g) {
  const int lane = static_cast<int>(group.thread_rank());
  if (seed.numAtoms == 0) {
    matchResultClearCooperative(match, lane);
    return seed.numBonds == 0;
  }
  if (seed.numAtoms > S.tNA || seed.numBonds > S.tNB) {
    matchResultClearCooperative(match, lane);
    return false;
  }
  if (lane == 0)
    S.found[g] = 0;
  group.sync();
  const int n = prepareSeedSearchCooperative(group, S, seed, g);
  if (n < 0) {
    matchResultClearCooperative(match, lane);
    return false;
  }
  seedDfsSearchCooperative(group, S, g, n);
  group.sync();
  if (*(volatile int*)&S.found[g] == 0) {
    matchResultClearCooperative(match, lane);
    return false;
  }
  return rebuildMatchCooperative(group, S, seed, match, g);
}

/// Greedy incremental extension of the parent's recorded witness by every
/// still-unmapped seed bond.  A true return proves the extension; false is
/// inconclusive and @p match must not be reused (the caller runs the exact
/// fallback).  Serial on lane 0: the whole walk is a handful of
/// shared-memory reads per bond, so warp coordination would cost more than
/// it saves.  Group-uniform.
template <int maxAtoms, int maxBonds, class GroupT>
__device__ bool tryMatchIncrementalGreedyCooperative(const GroupT&                                        group,
                                                     const FmcsPairShared<maxAtoms, maxBonds>&            S,
                                                     const Seed<maxAtoms, maxBonds>&                      seed,
                                                     MatchResult<maxAtoms, maxBonds, maxAtoms, maxBonds>& match) {
  using BondMask = typename FmcsPairShared<maxAtoms, maxBonds>::BondMask;
  const int lane = static_cast<int>(group.thread_rank());
  int       ok   = 1;
  if (lane == 0) {
    BondMask bits = maskFromWords<kFmcsMaskAtoms<maxBonds>>(seed.bonds);
    while (!bits.empty()) {
      const int b = bits.lowest();
      bits.clearLowest();
      if (match.targetBondIdx[b] != kUnmappedTargetIdx)
        continue;
      const int   u = S.qEpU[b], v = S.qEpV[b];
      const int   tu = match.targetAtomIdx[u], tv = match.targetAtomIdx[v];
      const auto& row     = S.bondRow[b];
      int         chosenB = -1, chosenA = -1;
      if (tu != kUnmappedTargetIdx && tv != kUnmappedTargetIdx) {
        // Ring-closing: both endpoints mapped; find the connecting bond.
        const int e0 = S.tRow[tu], e1 = S.tRow[tu + 1];
        for (int e = e0; e < e1; ++e) {
          if (S.tCol[e] != tv)
            continue;
          const int tb = S.tBnd[e];
          if (wordsTest(match.visitedTargetBonds, tb))
            continue;
          if (!row.test(tb))
            continue;
          chosenB = tb;
          break;
        }
      } else if (tu != kUnmappedTargetIdx || tv != kUnmappedTargetIdx) {
        // Atom-adding: extend from the mapped endpoint to an unvisited,
        // compatible neighbor.  No backtracking; a miss is inconclusive.
        const int   src   = (tu != kUnmappedTargetIdx) ? tu : tv;
        const int   qFree = (tu != kUnmappedTargetIdx) ? v : u;
        const auto& arow  = S.atomRow[qFree];
        const int   e0 = S.tRow[src], e1 = S.tRow[src + 1];
        for (int e = e0; e < e1; ++e) {
          const int tb = S.tBnd[e];
          if (wordsTest(match.visitedTargetBonds, tb))
            continue;
          const int ta = S.tCol[e];
          if (wordsTest(match.visitedTargetAtoms, ta))
            continue;
          if (!row.test(tb))
            continue;
          if (!arow.test(ta))
            continue;
          chosenB = tb;
          chosenA = ta;
          break;
        }
      } else {
        // Both endpoints unmapped: malformed for well-formed seeds.
        ok = 0;
        break;
      }
      if (chosenB < 0) {
        ok = 0;
        break;
      }
      match.targetBondIdx[b] = static_cast<uint8_t>(chosenB);
      wordsSet(match.visitedTargetBonds, chosenB);
      match.matchedBondSize += 1;
      match.empty = false;
      if (chosenA >= 0) {
        const int qFree            = (tu != kUnmappedTargetIdx) ? v : u;
        match.targetAtomIdx[qFree] = static_cast<uint8_t>(chosenA);
        wordsSet(match.visitedTargetAtoms, chosenA);
        match.matchedAtomSize += 1;
      }
    }
  }
  ok = group.shfl(ok, 0);
  return ok != 0;
}

/// RDKit checkIfMatchAndAppend analogue: greedy extension of a non-empty
/// parent witness first, exact substructure fallback otherwise.
template <int maxAtoms, int maxBonds, class GroupT>
__device__ __forceinline__ bool checkSeedMatchAndAppendCooperative(
  const GroupT&                                       group,
  FmcsPairShared<maxAtoms, maxBonds>&                 S,
  QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>& candidate,
  int                                                 g) {
  if (!candidate.match.empty) {
    if (tryMatchIncrementalGreedyCooperative(group, S, candidate.seed, candidate.match))
      return true;
  }
  return matchSeedSubstructureCooperative(group, S, candidate.seed, candidate.match, g);
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_MATCH_CUH
