// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FMCS_CUDA_FMCS_KERNEL_CUH
#define FMCS_CUDA_FMCS_KERNEL_CUH

#include "src/mcs/fmcs_cuda/fmcs_search_support.cuh"

namespace mcs {
namespace fmcs {

// One block per pair.  Same search semantics as RDKit's fMCS seed-grow with
// exact per-seed substructure verification and the stage 0/1/2 subset
// enumeration, restructured for the GPU:
//
//   Phase 1 tests every query bond's one-bond embedding directly in parallel
//   -- whether bond q embeds depends only on q -- and reduces RDKit's
//   sequential prefix/failed-bond exclusion bookkeeping to bitmask algebra.
//
//   Phase 2 runs FmcsKernelConfig::numGroups 32-lane grow groups against the
//   block's atomic LIFO worklist; all pair-invariant data is block-shared
//   (loadPairSharedCooperative) and the per-seed exact check runs on the
//   shared subgraph DFS core (fmcs_match.cuh).
template <int maxAtoms, int maxBonds>
__global__ __launch_bounds__(FmcsKernelConfig<maxAtoms>::blockThreads) void fmcsKernel(
  const DevicePerPairInput* __restrict__ pairs,
  DeviceMCSResult<maxAtoms, maxBonds>* __restrict__ results,
  QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>* __restrict__ queueStorageAll,
  int                queueCapacity,
  int                numPairs,
  unsigned long long timeoutClocks) {
  using SharedT              = FmcsPairShared<maxAtoms, maxBonds>;
  using QueuedT              = typename SharedT::QueuedT;
  using AtomMask             = typename SharedT::AtomMask;
  using BondMask             = typename SharedT::BondMask;
  constexpr int kNumGroups   = SharedT::kNumGroups;
  constexpr int blockThreads = FmcsKernelConfig<maxAtoms>::blockThreads;

  extern __shared__ __align__(16) char sharedRaw[];
  SharedT&                             S = *reinterpret_cast<SharedT*>(sharedRaw);

  const int pairIdx = static_cast<int>(blockIdx.x);
  if (pairIdx >= numPairs)
    return;

  auto      block   = cg::this_thread_block();
  auto      group   = cg::tiled_partition<kFmcsGroupSize>(block);
  const int tid     = static_cast<int>(block.thread_rank());
  const int groupId = tid / kFmcsGroupSize;
  const int lane    = static_cast<int>(group.thread_rank());

  const DevicePerPairInput& pair = pairs[pairIdx];
  DeviceCsrView             queryView;
  queryView.rowOffsets    = pair.queryRowOffsets;
  queryView.colIndices    = pair.queryColIndices;
  queryView.bondIndices   = pair.queryBondIndices;
  queryView.bondEndpoints = pair.queryBondEndpoints;
  queryView.numAtoms      = pair.queryNumAtoms;
  queryView.numBonds      = pair.queryNumBonds;
  DeviceCsrView targetView;
  targetView.rowOffsets    = pair.targetRowOffsets;
  targetView.colIndices    = pair.targetColIndices;
  targetView.bondIndices   = pair.targetBondIndices;
  targetView.bondEndpoints = pair.targetBondEndpoints;
  targetView.numAtoms      = pair.targetNumAtoms;
  targetView.numBonds      = pair.targetNumBonds;

  if (tid == 0) {
    S.startClock = clock64();
    S.queue.init(queueStorageAll + static_cast<size_t>(pairIdx) * queueCapacity, queueCapacity);
  }
  loadPairSharedCooperative(S, queryView, targetView, pair.tables, tid, blockThreads);

  // Clear the incumbent.
  if (tid == 0) {
    seedClearWithinThread(S.best.seed);
    matchResultClearWithinThread(S.best.match);
  }
  __syncthreads();

  const int qNB = S.qNB;

  // ---- Phase 1: RDKit makeInitialSeeds() analogue ----
  // One thread per query bond runs the direct single-bond match; the
  // matched/unmatched partition then yields each seed's exclusion set in
  // closed form: RDKit's prefix exclusion plus its retroactive exclusion of
  // later bonds that failed their own initial match.
  if (tid < qNB) {
    SingleBondMatch m;
    const bool      ok      = matchSingleBondWithinThread(S, tid, m);
    S.phase1TargetBond[tid] = ok ? m.targetBond : kUnmappedTargetIdx;
    S.phase1AtomU[tid]      = ok ? m.targetAtomU : kUnmappedTargetIdx;
    S.phase1AtomV[tid]      = ok ? m.targetAtomV : kUnmappedTargetIdx;
  }
  __syncthreads();

  if (tid == 0) {
    BondMask mm;
    mm.clear();
    for (int q = 0; q < qNB; ++q)
      if (S.phase1TargetBond[q] != kUnmappedTargetIdx)
        mm.set(q);
    S.phase1Matched = mm;
    S.phase1Count   = mm.popcount();
  }
  __syncthreads();
  const BondMask matchedMask = S.phase1Matched;
  const int      numMatched  = S.phase1Count;
  BondMask       notMatched  = lowMaskThrough<kFmcsMaskAtoms<maxBonds>>(qNB - 1);
  notMatched.andNotEq(matchedMask);

  for (int mi = groupId; mi < numMatched; mi += kNumGroups) {
    // The mi-th matched query bond.
    BondMask t = matchedMask;
    for (int k = 0; k < mi; ++k)
      t.clearLowest();
    const int q = t.lowest();

    QueuedT& c = S.current[groupId];
    if (lane == 0) {
      seedClearWithinThread(c.seed);
      seedAddBondWithinThread(c.seed, q);
      seedAddAtomWithinThread(c.seed, S.qEpU[q]);
      seedAddAtomWithinThread(c.seed, S.qEpV[q]);
      // RDKit prefix exclusion: earlier initial bonds are off limits.
      maskToWords(lowMaskThrough<kFmcsMaskAtoms<maxBonds>>(q), c.seed.excludedBonds);
    }
    group.sync();
    seedComputeRemainingSizeCooperative(group, S, c.seed);
    if (lane == 0) {
      // Full exclusion set: the prefix plus every later bond that failed its
      // own one-bond match (RDKit retro-excludes those from queued seeds).
      BondMask excl = lowMaskThrough<kFmcsMaskAtoms<maxBonds>>(q);
      BondMask nm   = notMatched;
      nm.andNotEq(excl);
      maskOrEq(excl, nm);
      maskToWords(excl, c.seed.excludedBonds);
    }
    matchResultClearCooperative(c.match, lane);
    if (lane == 0) {
      c.match.targetAtomIdx[S.qEpU[q]] = S.phase1AtomU[q];
      c.match.targetAtomIdx[S.qEpV[q]] = S.phase1AtomV[q];
      c.match.targetBondIdx[q]         = S.phase1TargetBond[q];
      wordsSet(c.match.visitedTargetAtoms, S.phase1AtomU[q]);
      wordsSet(c.match.visitedTargetAtoms, S.phase1AtomV[q]);
      wordsSet(c.match.visitedTargetBonds, S.phase1TargetBond[q]);
      c.match.matchedAtomSize = c.seed.numAtoms;
      c.match.matchedBondSize = 1;
      c.match.empty           = false;
    }
    group.sync();
    warpCopy(group, &S.queue.slot(mi), &c, sizeof(QueuedT));
    group.sync();
  }
  __syncthreads();
  if (tid == 0) {
    S.queue.setSizeWithinThread(numMatched);
    if (numMatched > 0)
      S.bestScore = (1u << 16) | 2u;
  }
  __syncthreads();
  if (numMatched > 0 && groupId == 0) {
    warpCopy(group, &S.best, &S.queue.slot(0), sizeof(QueuedT));
  }
  __syncthreads();

  QueuedT& myCurrent  = S.current[groupId];
  QueuedT& myBiggest  = S.biggest[groupId];
  NewBond* myNewBonds = S.newBonds[groupId];

  // ---- Phase 2: RDKit Seed::grow() analogue ----
  // An atomic LIFO worklist lets every warp group pop and grow one seed per
  // outer iteration.  This intentionally trades RDKit's sorted-front
  // scheduling order for multi-warp work while preserving the full
  // checkIfMatchAndAppend-style substructure validation.
  while (true) {
    if (tid == 0) {
      if (timeoutClocks > 0 && clock64() - S.startClock > timeoutClocks)
        S.timedOut = true;
      S.phase2Done = S.overflowed || S.timedOut || S.queue.empty();
    }
    block.sync();
    if (S.phase2Done)
      break;

    const bool poppedThisGroup = popBackCooperative(group, S.queue, myCurrent);
    if (lane == 0)
      S.popped[groupId] = poppedThisGroup;
    group.sync();
    // Complete every pop before any group can reserve and publish new work.
    block.sync();

    do {
      if (!S.popped[groupId])
        break;

      const unsigned int scoreSnapshot     = readBestScoreCooperative(group, &S.bestScore);
      const int          bestBondsSnapshot = static_cast<int>(scoreSnapshot >> 16);
      const int          bestAtomsSnapshot = static_cast<int>(scoreSnapshot & 0xFFFFu);
      if (!seedCanGrowBiggerThanWithinThread(myCurrent.seed, bestBondsSnapshot, bestAtomsSnapshot)) {
        break;
      }

      updateIncumbentCooperative(group, myCurrent, S.best, &S.bestScore, &S.bestCopyLock);

      const bool fillOk = fillNewBondsCooperative(group,
                                                  myCurrent.seed,
                                                  S.qEpU,
                                                  S.qEpV,
                                                  qNB,
                                                  myNewBonds,
                                                  &S.newBondCount[groupId],
                                                  maxBonds);
      group.sync();
      if (!fillOk) {
        if (lane == 0)
          S.overflowed = true;
        break;
      }
      if (S.newBondCount[groupId] == 0) {
        break;
      }

      bool runInnerStage = myCurrent.seed.growingStage != kSeedGrowStageOuter;

      // RDKit Seed::grow() stage 0: the all-bonds child.  If it matches with
      // more than one new bond, the parent is requeued at the inner stage.
      if (myCurrent.seed.growingStage == kSeedGrowStageOuter) {
        warpCopy(group, &myBiggest, &myCurrent, sizeof(QueuedT));
        group.sync();
        if (lane == 0) {
          seedBeginGrowStepWithinThread(myBiggest.seed);
          myBiggest.seed.growingStage = kSeedGrowStageOuter;
          for (int i = 0; i < S.newBondCount[groupId]; ++i) {
            seedAddNewBondWithinThread(myBiggest.seed, myNewBonds[i]);
          }
        }
        group.sync();
        seedComputeRemainingSizeCooperative(group, S, myBiggest.seed);

        const unsigned int childScoreSnapshot = readBestScoreCooperative(group, &S.bestScore);
        if (!seedCanGrowBiggerThanWithinThread(myBiggest.seed,
                                               static_cast<int>(childScoreSnapshot >> 16),
                                               static_cast<int>(childScoreSnapshot & 0xFFFFu))) {
          break;
        }

        const bool ok = checkSeedMatchAndAppendCooperative(group, S, myBiggest, groupId);
        if (ok) {
          updateIncumbentCooperative(group, myBiggest, S.best, &S.bestScore, &S.bestCopyLock);
          if (!pushBackCooperative(group, S.queue, myBiggest) && lane == 0) {
            S.overflowed = true;
          }
          group.sync();
          if (S.newBondCount[groupId] > 1) {
            if (lane == 0) {
              myCurrent.seed.growingStage = kSeedGrowStageInner;
            }
            group.sync();
            if (!pushBackCooperative(group, S.queue, myCurrent) && lane == 0) {
              S.overflowed = true;
            }
            group.sync();
          }
          break;
        }
        if (S.newBondCount[groupId] == 1)
          break;
        runInnerStage = true;
      }

      if (!runInnerStage)
        break;

      // RDKit Seed::grow() stage 1: try every individual outgoing bond.  A
      // failed individual match excludes that NewBond from later subset
      // enumeration.
      for (int i = 0; i < S.newBondCount[groupId]; ++i) {
        if (!myNewBonds[i].alive)
          continue;

        warpCopy(group, &myBiggest, &myCurrent, sizeof(QueuedT));
        group.sync();
        if (lane == 0) {
          seedBeginGrowStepWithinThread(myBiggest.seed);
          myBiggest.seed.growingStage = kSeedGrowStageOuter;
          seedAddNewBondWithinThread(myBiggest.seed, myNewBonds[i]);
        }
        group.sync();
        seedComputeRemainingSizeCooperative(group, S, myBiggest.seed);

        const unsigned int childScoreSnapshot = readBestScoreCooperative(group, &S.bestScore);
        if (!seedCanGrowBiggerThanWithinThread(myBiggest.seed,
                                               static_cast<int>(childScoreSnapshot >> 16),
                                               static_cast<int>(childScoreSnapshot & 0xFFFFu))) {
          continue;
        }

        const bool ok = checkSeedMatchAndAppendCooperative(group, S, myBiggest, groupId);
        if (ok) {
          updateIncumbentCooperative(group, myBiggest, S.best, &S.bestScore, &S.bestCopyLock);
          if (!pushBackCooperative(group, S.queue, myBiggest) && lane == 0) {
            S.overflowed = true;
          }
        } else if (lane == 0) {
          myNewBonds[i].alive = false;
        }
        group.sync();
      }

      // RDKit Seed::grow() stage 2: enumerate all non-singleton subsets of
      // the surviving NewBonds.  Singletons were handled by Stage 1; the
      // all-bonds subset was handled by Stage 0 unless Stage 1 erased one or
      // more individual bonds, in which case the all-surviving-bonds subset
      // is new work and must be checked.
      int aliveCount    = 0;
      int aliveOverflow = 0;
      if (lane == 0) {
        for (int i = 0; i < S.newBondCount[groupId]; ++i) {
          if (myNewBonds[i].alive)
            ++aliveCount;
        }
        aliveOverflow = aliveCount > 63 ? 1 : 0;
        if (aliveOverflow)
          S.overflowed = true;
      }
      aliveCount    = group.shfl(aliveCount, 0);
      aliveOverflow = group.shfl(aliveOverflow, 0);
      if (aliveOverflow)
        break;
      if (aliveCount > 1) {
        unsigned int erasedCount = 0;
        if (lane == 0) {
          erasedCount = static_cast<unsigned int>(S.newBondCount[groupId] - aliveCount);
        }
        erasedCount                             = group.shfl(erasedCount, 0);
        const unsigned long long maxComposition = (1ULL << aliveCount) - 1ULL;
        for (unsigned long long composition = maxComposition; composition != 0ULL; --composition) {
          if (isPowerOfTwo64(composition))
            continue;
          if (erasedCount == 0 && composition == maxComposition)
            continue;

          warpCopy(group, &myBiggest, &myCurrent, sizeof(QueuedT));
          group.sync();
          if (lane == 0) {
            seedBeginGrowStepWithinThread(myBiggest.seed);
            myBiggest.seed.growingStage = kSeedGrowStageOuter;
            int aliveBit                = 0;
            for (int i = 0; i < S.newBondCount[groupId]; ++i) {
              if (!myNewBonds[i].alive)
                continue;
              if ((composition & (1ULL << aliveBit)) != 0ULL) {
                seedAddNewBondWithinThread(myBiggest.seed, myNewBonds[i]);
              }
              ++aliveBit;
            }
          }
          group.sync();
          seedComputeRemainingSizeCooperative(group, S, myBiggest.seed);

          const unsigned int childScoreSnapshot = readBestScoreCooperative(group, &S.bestScore);
          if (!seedCanGrowBiggerThanWithinThread(myBiggest.seed,
                                                 static_cast<int>(childScoreSnapshot >> 16),
                                                 static_cast<int>(childScoreSnapshot & 0xFFFFu))) {
            continue;
          }

          const bool ok = checkSeedMatchAndAppendCooperative(group, S, myBiggest, groupId);
          if (ok) {
            updateIncumbentCooperative(group, myBiggest, S.best, &S.bestScore, &S.bestCopyLock);
            if (!pushBackCooperative(group, S.queue, myBiggest)) {
              if (lane == 0)
                S.overflowed = true;
            }
          }
          group.sync();
          if (readFlagCooperative(group, &S.overflowed))
            break;
        }
      }

    } while (false);

    // End-of-iteration rendezvous: every group must reach here before the
    // next pop/termination decision.
    block.sync();
  }

  // ---- Phase 3: writeback ----
  block.sync();
  if (tid == 0) {
    auto& dst             = results[pairIdx];
    dst.numCommonVertices = S.best.seed.numAtoms;
    dst.numCommonEdges    = S.best.seed.numBonds;
    dst.timedOut          = S.timedOut;
    dst.overflowed        = S.overflowed;

    // Walk set bits of best.seed.atoms (in increasing query atom idx order)
    // to fill mappingA/B, then best.seed.bonds for bondMapA/B.
    using BestSeedT                = decltype(S.best.seed);
    using AtomWord                 = typename BestSeedT::atom_word_type;
    constexpr int kAtomBitsPerWord = BestSeedT::kAtomBitsPerWord;
    constexpr int kAtomWords       = BestSeedT::kAtomWords;
    int           outIdx           = 0;
    for (int wordIdx = 0; wordIdx < kAtomWords; ++wordIdx) {
      AtomWord remaining = S.best.seed.atoms[wordIdx];
      while (remaining != 0) {
        int bitPosInWord;
        if constexpr (sizeof(AtomWord) == 4) {
          bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
        } else {
          bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
        }
        const int queryAtomIdx = wordIdx * kAtomBitsPerWord + bitPosInWord;
        remaining &= remaining - 1;
        dst.mappingA[outIdx] = static_cast<std::uint8_t>(queryAtomIdx);
        dst.mappingB[outIdx] = S.best.match.targetAtomIdx[queryAtomIdx];
        ++outIdx;
      }
    }

    using BondWord                 = typename BestSeedT::bond_word_type;
    constexpr int kBondBitsPerWord = BestSeedT::kBondBitsPerWord;
    constexpr int kBondWords       = BestSeedT::kBondWords;
    outIdx                         = 0;
    for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
      BondWord remaining = S.best.seed.bonds[wordIdx];
      while (remaining != 0) {
        int bitPosInWord;
        if constexpr (sizeof(BondWord) == 4) {
          bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
        } else {
          bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
        }
        const int queryBondIdx = wordIdx * kBondBitsPerWord + bitPosInWord;
        remaining &= remaining - 1;
        dst.bondMapA[outIdx] = static_cast<std::uint8_t>(queryBondIdx);
        dst.bondMapB[outIdx] = S.best.match.targetBondIdx[queryBondIdx];
        ++outIdx;
      }
    }
  }
}

enum class MaxSizeTier {
  k16,
  k32,
  k64,
  k128,
};

/// Smallest tier whose bitset width covers both counts. Tier 128 is reserved
/// for counts below 128 because its byte-packed CSR row offsets cannot
/// represent the 256 directed entries produced by 128 bonds. Returns -1 when
/// the largest tier does not fit; the caller must flag @ref MCSResult::overflowed.
inline int pickMaxSizeTier(int numAtoms, int numBonds) {
  if (numAtoms >= 128 || numBonds >= 128)
    return -1;
  const int need = numAtoms > numBonds ? numAtoms : numBonds;
  if (need <= 16)
    return 0;
  if (need <= 32)
    return 1;
  if (need <= 64)
    return 2;
  if (need < 128)
    return 3;
  return -1;
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_KERNEL_CUH
