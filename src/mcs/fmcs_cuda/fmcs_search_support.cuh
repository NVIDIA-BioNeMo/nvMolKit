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

#ifndef FMCS_CUDA_FMCS_SEARCH_SUPPORT_CUH
#define FMCS_CUDA_FMCS_SEARCH_SUPPORT_CUH

#include <cooperative_groups.h>

#include <cstdint>

#include "src/mcs/fmcs_cuda/fmcs_grow.cuh"
#include "src/mcs/fmcs_cuda/fmcs_match.cuh"
#include "src/mcs/fmcs_cuda/fmcs_match_tables.cuh"
#include "src/mcs/fmcs_cuda/fmcs_seed.cuh"
#include "src/mcs/fmcs_cuda/fmcs_seed_queue.cuh"
#include "src/mcs/fmcs_cuda/fmcs_topology.cuh"
#include "src/mcs/mcs_common/mcs_cooperative_copy.cuh"

namespace mcs {
namespace fmcs {

namespace cg = cooperative_groups;

/// Per-pair descriptor passed to the kernel.  Non-owning: pointers refer
/// into host-uploaded device buffers.  The caller chooses the smaller
/// input as the query (fMCS only enumerates subgraphs of the query) and
/// records the choice in @c swapped so host-side expansion can un-swap
/// the result mappings.
struct DevicePerPairInput {
  int queryNumAtoms  = 0;
  int queryNumBonds  = 0;
  int targetNumAtoms = 0;
  int targetNumBonds = 0;

  const std::uint32_t* queryRowOffsets    = nullptr;
  const std::uint32_t* queryColIndices    = nullptr;
  /// Parallel to @c queryColIndices: undirected bond id for each CSR entry.
  const std::uint32_t* queryBondIndices   = nullptr;
  /// Packed (u << 16 | v), one entry per undirected bond, ordered to
  /// match the bond dimension of the match tables (@ref enumerateBonds).
  const std::uint32_t* queryBondEndpoints = nullptr;

  const std::uint32_t* targetRowOffsets    = nullptr;
  const std::uint32_t* targetColIndices    = nullptr;
  const std::uint32_t* targetBondIndices   = nullptr;
  const std::uint32_t* targetBondEndpoints = nullptr;

  PairMatchTablesDevice tables;

  bool swapped = false;
};

/// Fixed-size device-writable result.  mcs::MCSResult uses std::vector and
/// cannot be constructed on the device, so the kernel fills this POD and
/// the host expands it.
///
/// @c bondMapA[i] / @c bondMapB[i] hold the query / target bond index of
/// the i-th matched edge in the common subgraph.  The host recovers
/// (u, v) endpoints by indexing the @c bondEndpoints arrays it already
/// uploaded for each pair -- avoids duplicating endpoint data here at
/// 4 * maxBonds bytes per result.
template <int maxAtoms, int maxBonds> struct DeviceMCSResult {
  int  numCommonVertices = 0;
  int  numCommonEdges    = 0;
  bool timedOut          = false;
  bool overflowed        = false;

  /// mappingA[i] = query atom idx, mappingB[i] = target atom idx of
  /// the i-th matched vertex.
  uint8_t mappingA[maxAtoms];
  uint8_t mappingB[maxAtoms];
  /// bondMapA[i] = query bond idx, bondMapB[i] = target bond idx of
  /// the i-th matched edge.
  uint8_t bondMapA[maxBonds];
  uint8_t bondMapB[maxBonds];
};

template <class GroupT, class QueuedT>
__device__ __forceinline__ void updateIncumbentCooperative(const GroupT&  group,
                                                           const QueuedT& candidate,
                                                           QueuedT&       best,
                                                           unsigned int*  bestScore,
                                                           int*           bestCopyLock) {
  const int groupRank  = static_cast<int>(group.thread_rank());
  int       locked     = 0;
  int       shouldCopy = 0;
  if (groupRank == 0) {
    const unsigned int candidateScore =
      (static_cast<unsigned int>(candidate.seed.numBonds) << 16) | static_cast<unsigned int>(candidate.seed.numAtoms);
    unsigned int prev = *bestScore;
    bool         won  = false;
    while (candidateScore > prev) {
      const unsigned int seen = atomicCAS(bestScore, prev, candidateScore);
      if (seen == prev) {
        won = true;
        break;
      }
      prev = seen;
    }
    if (won) {
      while (atomicCAS(bestCopyLock, 0, 1) != 0) {
      }
      locked     = 1;
      shouldCopy = (candidateScore == *bestScore) ? 1 : 0;
    }
  }
  locked     = group.shfl(locked, 0);
  shouldCopy = group.shfl(shouldCopy, 0);
  if (shouldCopy) {
    warpCopy(group, &best, &candidate, sizeof(QueuedT));
  }
  group.sync();
  if (groupRank == 0 && locked) {
    __threadfence_block();
    atomicExch(bestCopyLock, 0);
  }
  group.sync();
}

__device__ __forceinline__ bool isPowerOfTwo64(unsigned long long value) {
  return value != 0ULL && (value & (value - 1ULL)) == 0ULL;
}

template <class GroupT, class QueuedT>
__device__ __forceinline__ bool insertSortedByBondsCooperative(const GroupT&                         group,
                                                               SeedQueue<QueuedT, ThreadBlockScope>& queue,
                                                               const QueuedT&                        element) {
  const int groupRank = static_cast<int>(group.thread_rank());
  int       oldSize   = 0;
  int       insertAt  = 0;
  int       ok        = 1;
  if (groupRank == 0) {
    oldSize  = queue.size();
    ok       = oldSize < queue.capacity() ? 1 : 0;
    insertAt = oldSize;
    if (ok) {
      for (int i = 0; i < oldSize; ++i) {
        if (queue.slot(i).seed.numBonds < element.seed.numBonds) {
          insertAt = i;
          break;
        }
      }
    }
  }
  oldSize  = group.shfl(oldSize, 0);
  insertAt = group.shfl(insertAt, 0);
  ok       = group.shfl(ok, 0);
  if (!ok)
    return false;

  for (int i = oldSize; i > insertAt; --i) {
    warpCopy(group, &queue.slot(i), &queue.slot(i - 1), sizeof(QueuedT));
    group.sync();
  }
  warpCopy(group, &queue.slot(insertAt), &element, sizeof(QueuedT));
  group.sync();
  if (groupRank == 0)
    queue.setSizeWithinThread(oldSize + 1);
  group.sync();
  return true;
}

template <class GroupT, class QueuedT>
__device__ __forceinline__ bool popFrontCooperative(const GroupT&                         group,
                                                    SeedQueue<QueuedT, ThreadBlockScope>& queue,
                                                    QueuedT&                              outElement) {
  const int groupRank = static_cast<int>(group.thread_rank());
  int       oldSize   = 0;
  if (groupRank == 0)
    oldSize = queue.size();
  oldSize = group.shfl(oldSize, 0);
  if (oldSize <= 0)
    return false;

  warpCopy(group, &outElement, &queue.slot(0), sizeof(QueuedT));
  group.sync();
  for (int i = 1; i < oldSize; ++i) {
    warpCopy(group, &queue.slot(i - 1), &queue.slot(i), sizeof(QueuedT));
    group.sync();
  }
  if (groupRank == 0)
    queue.setSizeWithinThread(oldSize - 1);
  group.sync();
  return true;
}

template <class GroupT, class QueuedT>
__device__ __forceinline__ bool pushBackCooperative(const GroupT&                         group,
                                                    SeedQueue<QueuedT, ThreadBlockScope>& queue,
                                                    const QueuedT&                        element) {
  const int slot = queue.batchReserveCooperative(group, 1);
  if (slot < 0)
    return false;
  warpCopy(group, &queue.slot(slot), &element, sizeof(QueuedT));
  group.sync();
  return true;
}

template <class GroupT, class QueuedT>
__device__ __forceinline__ bool popBackCooperative(const GroupT&                         group,
                                                   SeedQueue<QueuedT, ThreadBlockScope>& queue,
                                                   QueuedT&                              outElement) {
  const int oldTop = queue.popReserveCooperative(group);
  if (oldTop < 0)
    return false;
  warpCopy(group, &outElement, &queue.slot(oldTop - 1), sizeof(QueuedT));
  group.sync();
  return true;
}

template <class GroupT>
__device__ __forceinline__ unsigned int readBestScoreCooperative(const GroupT& group, const unsigned int* bestScore) {
  unsigned int score = 0;
  if (group.thread_rank() == 0)
    score = *bestScore;
  return group.shfl(score, 0);
}

template <class GroupT> __device__ __forceinline__ bool readFlagCooperative(const GroupT& group, const bool* flag) {
  int value = 0;
  if (group.thread_rank() == 0)
    value = *flag ? 1 : 0;
  return group.shfl(value, 0) != 0;
}

/// Per-block seed worklist capacity.  The backing slab lives in global
/// memory; only the cursor/header lives in shared memory.  Atomic LIFO
/// push/pop lets multiple warp groups own grow work concurrently.
constexpr int kFmcsQueueCapacity = 4096;

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_SEARCH_SUPPORT_CUH
