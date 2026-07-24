// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FMCS_CUDA_FMCS_KERNEL_CUH
#define FMCS_CUDA_FMCS_KERNEL_CUH

#include "src/mcs/fmcs_cuda/fmcs_search_support.cuh"

namespace mcs {
namespace fmcs {

/// Tier-128 per-group substructure scratch does not fit in shared memory once
/// a block runs more than four warp groups, so those specializations take the
/// scratch from a global-memory slab instead.
template <int blockThreads, int maxAtoms>
inline constexpr bool kUseGlobalSubstructureScratch = maxAtoms == 128 && blockThreads >= 256;

template <int maxAtoms, int maxBonds, int blockThreads, class Policy, bool GlobalSubstructureScratch = false>
__global__ void fmcsKernel(const DevicePerPairInput* __restrict__ pairs,
                           DeviceMCSResult<maxAtoms, maxBonds>* __restrict__ results,
                           QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>* __restrict__ queueStorageAll,
                           FmcsSubstructureScratch<maxAtoms, maxAtoms>* __restrict__ scratchStorageAll,
                           int                queueCapacity,
                           int                numPairs,
                           unsigned long long timeoutClocks) {
  const int pairIdx = blockIdx.x;
  if (pairIdx >= numPairs)
    return;

  auto                      block = cg::this_thread_block();
  const DevicePerPairInput& pair  = pairs[pairIdx];

  using QueuedT                     = QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>;
  using SubstructureScratchT        = FmcsSubstructureScratch<maxAtoms, maxAtoms>;
  constexpr int kMaxNewBondsForTier = maxBonds;
  constexpr int kNumGroups          = FmcsBlockConfig<blockThreads>::numGroups;

  // Block-shared resources: the queue, the incumbent, and the early-exit
  // flags are visible to every group. Cross-group queue/incumbent updates use
  // atomics; loop-control reads are made uniform before any branch that can
  // skip a later block/group rendezvous.
  __shared__ SeedQueue<QueuedT, ThreadBlockScope> queue;
  __shared__ __align__(16) unsigned char bestStorage[sizeof(QueuedT)];
  QueuedT&                               best = *reinterpret_cast<QueuedT*>(bestStorage);
  // Atomic incumbent score: high 16 bits = numBonds, low 16 bits =
  // numAtoms.  Groups race through atomicCAS on this single int to
  // claim the right to write @c best.
  __shared__ unsigned int                bestScore;
  __shared__ int                         bestCopyLock;
  __shared__ DeviceCsrView               queryView;
  __shared__ DeviceCsrView               targetView;
  __shared__ bool                        overflowed;
  __shared__ bool                        timedOut;
  __shared__ bool                        phase2Done;
  __shared__ unsigned long long          startClock;
  SubstructureScratchT*                  substructureScratch;
  if constexpr (GlobalSubstructureScratch) {
    substructureScratch = scratchStorageAll + static_cast<size_t>(pairIdx) * kNumGroups;
  } else {
    (void)scratchStorageAll;
    __shared__ SubstructureScratchT sharedSubstructureScratch[kNumGroups];
    substructureScratch = sharedSubstructureScratch;
  }

  // Cooperative Phase 2 working state.  Approach 1 gives each warp group an
  // independent seed workspace and fallback scratch so groups can pop and
  // grow seeds concurrently.
  __shared__ __align__(16) unsigned char currentStorage[sizeof(QueuedT) * kNumGroups];
  __shared__ __align__(16) unsigned char biggestStorage[sizeof(QueuedT) * kNumGroups];
  QueuedT*                               current = reinterpret_cast<QueuedT*>(currentStorage);
  QueuedT*                               biggest = reinterpret_cast<QueuedT*>(biggestStorage);
  __shared__ NewBond                     newBondsArr[kNumGroups][kMaxNewBondsForTier];
  __shared__ int                         newBondCount[kNumGroups];
  __shared__ bool                        popped[kNumGroups];
  __shared__ bool                        stage0Ok[kNumGroups];
  __shared__ std::uint8_t remainingAtomStack[kNumGroups][maxAtoms];
  __shared__ typename Seed<maxAtoms, maxBonds>::atom_word_type
    remainingVisitedAtoms[kNumGroups][Seed<maxAtoms, maxBonds>::kAtomWords];
  __shared__ typename Seed<maxAtoms, maxBonds>::bond_word_type
                 remainingVisitedBonds[kNumGroups][Seed<maxAtoms, maxBonds>::kBondWords];
  __shared__ int remainingStackSize[kNumGroups];
  __shared__
    typename Seed<maxAtoms, maxBonds>::bond_word_type initialExcludedBonds[Seed<maxAtoms, maxBonds>::kBondWords];

  auto                  group                 = cg::tiled_partition<kFmcsGroupSize>(block);
  const int             groupId               = static_cast<int>(block.thread_rank()) / kFmcsGroupSize;
  const int             groupRank             = static_cast<int>(group.thread_rank());
  SubstructureScratchT& mySubstructureScratch = substructureScratch[groupId];

  QueuedT* myQueueStorage = queueStorageAll + static_cast<size_t>(pairIdx) * queueCapacity;

  if (block.thread_rank() == 0) {
    queue.init(myQueueStorage, queueCapacity);
    seedClearWithinThread(best.seed);
    matchResultClearWithinThread(best.match);
    bestScore    = 0;
    bestCopyLock = 0;
    overflowed   = false;
    timedOut     = false;
    phase2Done   = false;
    startClock   = clock64();

    queryView.rowOffsets    = pair.queryRowOffsets;
    queryView.colIndices    = pair.queryColIndices;
    queryView.bondIndices   = pair.queryBondIndices;
    queryView.bondEndpoints = pair.queryBondEndpoints;
    queryView.numAtoms      = pair.queryNumAtoms;
    queryView.numBonds      = pair.queryNumBonds;

    targetView.rowOffsets    = pair.targetRowOffsets;
    targetView.colIndices    = pair.targetColIndices;
    targetView.bondIndices   = pair.targetBondIndices;
    targetView.bondEndpoints = pair.targetBondEndpoints;
    targetView.numAtoms      = pair.targetNumAtoms;
    targetView.numBonds      = pair.targetNumBonds;
  }
  block.sync();

  QueuedT&      myCurrent               = current[groupId];
  QueuedT&      myBiggest               = biggest[groupId];
  NewBond*      myNewBonds              = newBondsArr[groupId];
  std::uint8_t* myRemainingAtomStack    = remainingAtomStack[groupId];
  auto*         myRemainingVisitedAtoms = remainingVisitedAtoms[groupId];
  auto*         myRemainingVisitedBonds = remainingVisitedBonds[groupId];

  // ---- Phase 1: RDKit makeInitialSeeds() analogue ----
  // RDKit creates one initial seed per query bond, not one per target
  // embedding.  Each candidate goes through checkIfMatchAndAppend(), which
  // runs substructure matching and stores one witness MatchResult on success.
  // Initial ExcludedBonds is prefix-like: later initial seeds exclude earlier
  // query bonds, and a mismatched initial bond is also excluded from seeds
  // already admitted.  Group 0 handles this serial state; the substructure
  // check remains cooperative across that group's lanes.
  if (block.thread_rank() < Seed<maxAtoms, maxBonds>::kBondWords) {
    initialExcludedBonds[block.thread_rank()] = 0;
  }
  block.sync();

  if (groupId == 0) {
    for (int qBond = 0; qBond < pair.queryNumBonds && !overflowed && !timedOut; ++qBond) {
      if (groupRank == 0) {
        seedClearWithinThread(myCurrent.seed);
        matchResultClearWithinThread(myCurrent.match);
        for (int wordIdx = 0; wordIdx < Seed<maxAtoms, maxBonds>::kBondWords; ++wordIdx) {
          myCurrent.seed.excludedBonds[wordIdx] = initialExcludedBonds[wordIdx];
        }

        const std::uint32_t queryEndpoints = queryView.bondEndpoints[qBond];
        const int           queryEndpointU = static_cast<int>(queryEndpoints >> kBondEndpointShift);
        const int           queryEndpointV = static_cast<int>(queryEndpoints & kBondEndpointMask);
        seedAddBondWithinThread(myCurrent.seed, qBond);
        seedAddAtomWithinThread(myCurrent.seed, queryEndpointU);
        seedAddAtomWithinThread(myCurrent.seed, queryEndpointV);
        myCurrent.seed.growingStage = kSeedGrowStageOuter;
      }
      group.sync();
      seedComputeRemainingSizeRdkitCooperative(group,
                                               myCurrent.seed,
                                               queryView,
                                               myRemainingAtomStack,
                                               myRemainingVisitedAtoms,
                                               myRemainingVisitedBonds,
                                               &remainingStackSize[groupId]);

      const bool matched =
        checkSeedMatchAndAppendCooperative(group, myCurrent, queryView, targetView, pair.tables, mySubstructureScratch);
      if (matched) {
        updateIncumbentCooperative(group, myCurrent, best, &bestScore, &bestCopyLock);
        if (!pushBackCooperative(group, queue, myCurrent)) {
          overflowed = true;
        }
      } else if (groupRank == 0) {
        const int queuedSeeds = queue.size();
        for (int i = 0; i < queuedSeeds; ++i) {
          seedExcludeBondWithinThread(queue.slot(i).seed, qBond);
        }
      }
      group.sync();

      if (groupRank == 0) {
        const int  wordIdx = qBond / Seed<maxAtoms, maxBonds>::kBondBitsPerWord;
        const auto mask    = static_cast<typename Seed<maxAtoms, maxBonds>::bond_word_type>(1)
                       << (qBond % Seed<maxAtoms, maxBonds>::kBondBitsPerWord);
        initialExcludedBonds[wordIdx] |= mask;
      }
      group.sync();
    }
  }

  block.sync();
  if (block.thread_rank() == 0 && timeoutClocks > 0 && clock64() - startClock > timeoutClocks) {
    timedOut = true;
  }
  block.sync();

  // ---- Phase 2: RDKit Seed::grow() analogue ----

  // Approach 1 uses an atomic LIFO worklist so every warp group can pop and
  // grow one seed per outer iteration.  This intentionally trades RDKit's
  // sorted-front scheduling order for useful multi-warp work while preserving
  // the full checkIfMatchAndAppend-style substructure validation.
  while (true) {
    if (block.thread_rank() == 0)
      phase2Done = overflowed || timedOut || queue.empty();
    block.sync();
    if (phase2Done)
      break;

    const bool poppedThisGroup = popBackCooperative(group, queue, myCurrent);
    if (groupRank == 0) {
      popped[groupId] = poppedThisGroup;
    }
    group.sync();
    // Complete every pop before any group can reserve and publish new work.
    block.sync();

    do {
      if (!popped[groupId])
        break;

      // RDKit growSeeds() increments TotalSteps before Seed::grow(), then
      // Seed::grow() first checks canGrowBiggerThan().
      const unsigned int scoreSnapshot     = readBestScoreCooperative(group, &bestScore);
      const int          bestBondsSnapshot = static_cast<int>(scoreSnapshot >> 16);
      const int          bestAtomsSnapshot = static_cast<int>(scoreSnapshot & 0xFFFFu);
      if (!seedCanGrowBiggerThanWithinThread(myCurrent.seed, bestBondsSnapshot, bestAtomsSnapshot)) {
        break;
      }

      updateIncumbentCooperative(group, myCurrent, best, &bestScore, &bestCopyLock);

      const bool fillOk = fillNewBondsCooperative(group,
                                                  myCurrent.seed,
                                                  queryView,
                                                  myNewBonds,
                                                  &newBondCount[groupId],
                                                  kMaxNewBondsForTier);
      group.sync();
      if (!fillOk) {
        if (groupRank == 0)
          overflowed = true;
        break;
      }

      if (newBondCount[groupId] == 0) {
        break;
      }

      bool runInnerStage = myCurrent.seed.growingStage != kSeedGrowStageOuter;

      // RDKit Seed::grow() stage 0: build the child containing all newly
      // discovered outgoing bonds and run checkIfMatchAndAppend().  If this
      // all-bonds child matches and there is more than one new bond, RDKit
      // returns immediately with the parent left at GrowingStage=1; the next
      // outer grow loop resumes the parent at the singleton/subset stage.
      if (myCurrent.seed.growingStage == kSeedGrowStageOuter) {
        warpCopy(group, &myBiggest, &myCurrent, sizeof(QueuedT));
        group.sync();
        if (groupRank == 0) {
          seedBeginGrowStepWithinThread(myBiggest.seed);
          myBiggest.seed.growingStage = kSeedGrowStageOuter;
          for (int i = 0; i < newBondCount[groupId]; ++i) {
            seedAddNewBondWithinThread(myBiggest.seed, myNewBonds[i]);
          }
        }
        group.sync();
        seedComputeRemainingSizeRdkitCooperative(group,
                                                 myBiggest.seed,
                                                 queryView,
                                                 myRemainingAtomStack,
                                                 myRemainingVisitedAtoms,
                                                 myRemainingVisitedBonds,
                                                 &remainingStackSize[groupId]);

        const unsigned int childScoreSnapshot = readBestScoreCooperative(group, &bestScore);
        const int          childBestBonds     = static_cast<int>(childScoreSnapshot >> 16);
        const int          childBestAtoms     = static_cast<int>(childScoreSnapshot & 0xFFFFu);
        if (!seedCanGrowBiggerThanWithinThread(myBiggest.seed, childBestBonds, childBestAtoms)) {
          break;
        }

        const bool ok = checkSeedMatchAndAppendCooperative(group,
                                                           myBiggest,
                                                           queryView,
                                                           targetView,
                                                           pair.tables,
                                                           mySubstructureScratch);
        if (groupRank == 0)
          stage0Ok[groupId] = ok;
        group.sync();

        if (stage0Ok[groupId]) {
          updateIncumbentCooperative(group, myBiggest, best, &bestScore, &bestCopyLock);
          if (!pushBackCooperative(group, queue, myBiggest)) {
            overflowed = true;
          }
          group.sync();
          if (newBondCount[groupId] > 1) {
            if (groupRank == 0) {
              myCurrent.seed.growingStage = kSeedGrowStageInner;
            }
            group.sync();
            if (!pushBackCooperative(group, queue, myCurrent)) {
              overflowed = true;
            }
            group.sync();
          }
          break;
        }
        if (newBondCount[groupId] == 1)
          break;
        runInnerStage = true;
      }

      if (!runInnerStage)
        break;

      // RDKit Seed::grow() stage 1: try every individual outgoing bond.
      // A failed individual match excludes that NewBond from later subset
      // enumeration and increments IndividualBondExcluded.
      for (int i = 0; i < newBondCount[groupId]; ++i) {
        if (!myNewBonds[i].alive)
          continue;

        warpCopy(group, &myBiggest, &myCurrent, sizeof(QueuedT));
        group.sync();

        if (groupRank == 0) {
          seedBeginGrowStepWithinThread(myBiggest.seed);
          myBiggest.seed.growingStage = kSeedGrowStageOuter;
          seedAddNewBondWithinThread(myBiggest.seed, myNewBonds[i]);
        }
        group.sync();
        seedComputeRemainingSizeRdkitCooperative(group,
                                                 myBiggest.seed,
                                                 queryView,
                                                 myRemainingAtomStack,
                                                 myRemainingVisitedAtoms,
                                                 myRemainingVisitedBonds,
                                                 &remainingStackSize[groupId]);

        const unsigned int childScoreSnapshot = readBestScoreCooperative(group, &bestScore);
        const int          childBestBonds     = static_cast<int>(childScoreSnapshot >> 16);
        const int          childBestAtoms     = static_cast<int>(childScoreSnapshot & 0xFFFFu);
        if (!seedCanGrowBiggerThanWithinThread(myBiggest.seed, childBestBonds, childBestAtoms)) {
          continue;
        }

        const bool ok = checkSeedMatchAndAppendCooperative(group,
                                                           myBiggest,
                                                           queryView,
                                                           targetView,
                                                           pair.tables,
                                                           mySubstructureScratch);
        if (ok) {
          updateIncumbentCooperative(group, myBiggest, best, &bestScore, &bestCopyLock);
          if (!pushBackCooperative(group, queue, myBiggest)) {
            overflowed = true;
          }
        } else if (groupRank == 0) {
          myNewBonds[i].alive = false;
        }
        group.sync();
      }

      // RDKit Seed::grow() stage 2: enumerate all non-singleton subsets of
      // the surviving NewBonds.  Singletons were handled by Stage 1; the
      // all-bonds subset was handled by Stage 0 unless Stage 1 erased one or
      // more individual bonds, in which case the all-surviving-bonds subset is
      // new work and must be checked.
      int aliveCount    = 0;
      int aliveOverflow = 0;
      if (groupRank == 0) {
        for (int i = 0; i < newBondCount[groupId]; ++i) {
          if (myNewBonds[i].alive)
            ++aliveCount;
        }
        aliveOverflow = aliveCount > 63 ? 1 : 0;
        if (aliveOverflow)
          overflowed = true;
      }
      aliveCount    = group.shfl(aliveCount, 0);
      aliveOverflow = group.shfl(aliveOverflow, 0);
      if (aliveOverflow)
        break;
      if (aliveCount > 1) {
        unsigned int erasedCount = 0;
        if (groupRank == 0) {
          erasedCount = static_cast<unsigned int>(newBondCount[groupId] - aliveCount);
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
          if (groupRank == 0) {
            seedBeginGrowStepWithinThread(myBiggest.seed);
            myBiggest.seed.growingStage = kSeedGrowStageOuter;
            int aliveBit                = 0;
            for (int i = 0; i < newBondCount[groupId]; ++i) {
              if (!myNewBonds[i].alive)
                continue;
              if ((composition & (1ULL << aliveBit)) != 0ULL) {
                seedAddNewBondWithinThread(myBiggest.seed, myNewBonds[i]);
              }
              ++aliveBit;
            }
          }
          group.sync();
          seedComputeRemainingSizeRdkitCooperative(group,
                                                   myBiggest.seed,
                                                   queryView,
                                                   myRemainingAtomStack,
                                                   myRemainingVisitedAtoms,
                                                   myRemainingVisitedBonds,
                                                   &remainingStackSize[groupId]);

          const unsigned int childScoreSnapshot = readBestScoreCooperative(group, &bestScore);
          const int          childBestBonds     = static_cast<int>(childScoreSnapshot >> 16);
          const int          childBestAtoms     = static_cast<int>(childScoreSnapshot & 0xFFFFu);
          if (!seedCanGrowBiggerThanWithinThread(myBiggest.seed, childBestBonds, childBestAtoms)) {
            continue;
          }

          const bool ok = checkSeedMatchAndAppendCooperative(group,
                                                             myBiggest,
                                                             queryView,
                                                             targetView,
                                                             pair.tables,
                                                             mySubstructureScratch);
          if (ok) {
            updateIncumbentCooperative(group, myBiggest, best, &bestScore, &bestCopyLock);
            if (!pushBackCooperative(group, queue, myBiggest)) {
              overflowed = true;
            }
          }
          group.sync();
          if (readFlagCooperative(group, &overflowed))
            break;
        }
      }

    } while (false);

    // End-of-iteration rendezvous: every group must reach here before
    // group 0 decides whether the sorted queue is empty.
    block.sync();
    if (block.thread_rank() == 0 && timeoutClocks > 0 && clock64() - startClock > timeoutClocks) {
      timedOut = true;
    }
    block.sync();

    if (block.thread_rank() == 0)
      phase2Done = overflowed || timedOut || queue.empty();
    block.sync();
    if (phase2Done)
      break;
  }

  // ---- Phase 3: writeback ----
  block.sync();
  if (block.thread_rank() == 0) {
    auto& dst             = results[pairIdx];
    dst.numCommonVertices = best.seed.numAtoms;
    dst.numCommonEdges    = best.seed.numBonds;
    dst.timedOut          = timedOut;
    dst.overflowed        = overflowed;

    // Walk set bits of best.seed.atoms (in increasing query atom idx
    // order, via __ffs/__ffsll) to fill mappingA/B.
    using BestSeedT                = decltype(best.seed);
    using AtomWord                 = typename BestSeedT::atom_word_type;
    constexpr int kAtomBitsPerWord = BestSeedT::kAtomBitsPerWord;
    constexpr int kAtomWords       = BestSeedT::kAtomWords;
    int           outIdx           = 0;
    for (int wordIdx = 0; wordIdx < kAtomWords; ++wordIdx) {
      AtomWord remaining = best.seed.atoms[wordIdx];
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
        dst.mappingB[outIdx] = best.match.targetAtomIdx[queryAtomIdx];
        ++outIdx;
      }
    }

    // Walk set bits of best.seed.bonds to fill bondMapA/B.
    using BondWord                 = typename BestSeedT::bond_word_type;
    constexpr int kBondBitsPerWord = BestSeedT::kBondBitsPerWord;
    constexpr int kBondWords       = BestSeedT::kBondWords;
    outIdx                         = 0;
    for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
      BondWord remaining = best.seed.bonds[wordIdx];
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
        dst.bondMapB[outIdx] = best.match.targetBondIdx[queryBondIdx];
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

/// Smallest tier whose bitset width covers both counts.  Returns -1 when
/// the largest tier (128) does not fit; the caller must flag
/// @ref MCSResult::overflowed.
inline int pickMaxSizeTier(int numAtoms, int numBonds) {
  const int need = numAtoms > numBonds ? numAtoms : numBonds;
  if (need <= 16)
    return 0;
  if (need <= 32)
    return 1;
  if (need <= 64)
    return 2;
  if (need <= 128)
    return 3;
  return -1;
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_KERNEL_CUH
