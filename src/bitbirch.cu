// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/bitbirch.h"
#include "src/bitbirch_common.cuh"
#include "src/bitbirch_internal.h"
#include "src/utils/cub_helpers.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/host_vector.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {
namespace {

constexpr int kBlockSize          = 256;
constexpr int kNumWarps           = kBlockSize / 32;
//! Bits each thread owns: thread t owns bits t, t + kBlockSize, ... up to kBitBirchMaxWords * 32.
constexpr int kBitParts           = kBitBirchMaxWords * 32 / kBlockSize;
//! Lanes that read one centroid together in the closest-entry search.
constexpr int kLanesPerEntry      = 4;
//! Rows inserted in batches that never merge groups, so the young tree grows in input order.
constexpr int kOrderedWarmup      = 16384;
//! Split pair orders are the low 32 bits of a 64-bit key, so a node holds fewer than 2^16 entries.
constexpr int kMaxBranchingFactor = 65534;

enum Status : int {
  kSuccess,
  kCapacity,
  kInvalidTree
};

//! Values the host reads back after each insertion round.
enum Control : int {
  kPending,
  kNodes,
  kEntries,
  kSlots,
  kStatus,
  kDepth,
  kNumControls
};

// One tree in flat device arrays. Every node keeps its entries contiguously in
// nodeEntries (stride branchingFactor + 1, enough for an overflowing node).
// Leaf entries are clusters: a single-member entry refers to its input row
// (entryRow >= 0) until a merge gives it a summary slot holding per-bit sums
// and the packed majority centroid. Entries of internal nodes summarize their
// child node and always have a slot.
struct Tree {
  const std::uint32_t* fingerprints;
  int*                 labels;  //!< Leaf entry of every row until finalization; -1 while pending.
  int                  numWords;
  int                  numBits;
  int                  branchingFactor;
  int                  stride;
  int                  batchBegin;  //!< Input row of the batch being inserted.
  int*                 nodeSize;
  int*                 nodeParent;
  int*                 nodeParentEntry;
  std::uint8_t*        nodeLeaf;
  int*                 nodeEntries;
  int*                 entryChild;
  std::uint32_t*       entryCount;
  int*                 entrySlot;
  int*                 entryRow;
  int*                 entryFirst;  //!< Earliest member row; the cluster ID after finalization.
  std::uint32_t*       sums;
  std::uint32_t*       centroids;
  int*                 root;
  int*                 nodeCursor;
  int*                 entryCursor;
  int*                 slotCursor;
  int*                 status;
  int                  maxNodes;
  int                  maxEntries;
  int                  maxSlots;
};

using IntegerReduce = cub::BlockReduce<unsigned long long, kBlockSize>;
using TermsReduce   = cub::BlockReduce<bitbirch::ISimTanimotoTerms, kBlockSize>;

//! Per-block scratch for reductions and broadcasts.
struct Scratch {
  const std::uint32_t* rows[kBlockSize];  //!< Row pointers resolved for one chunk of addRows.
  bool                 rowIsSums[kBlockSize];
  union {
    IntegerReduce::TempStorage integers;
    TermsReduce::TempStorage   terms;
  } reduce;
  unsigned long long reduced;  //!< Result of the last blockReduce.
  int                broadcast[3];
};

//! Per-bit sums owned by one thread; see kBitParts.
struct Bits {
  std::uint32_t parts[kBitParts] = {};
};

__device__ __forceinline__ int* entriesOf(const Tree& tree, const int node) {
  return tree.nodeEntries + static_cast<std::size_t>(node) * tree.stride;
}

__device__ __forceinline__ const std::uint32_t* rowOf(const Tree& tree, const int row) {
  return tree.fingerprints + static_cast<std::size_t>(row) * tree.numWords;
}

__device__ __forceinline__ const std::uint32_t* centroidOf(const Tree& tree, const int entry) {
  const int row = tree.entryRow[entry];
  return row >= 0 ? rowOf(tree, row) : tree.centroids + static_cast<std::size_t>(tree.entrySlot[entry]) * tree.numWords;
}

// Tanimoto similarity of packed rows, 1 when both are empty. Correctly rounded
// FP32 quotients order exactly like the true ratios here: distinct ratios a/b
// with b <= 4096 bits differ by more than 1/(4096 * 4095) > 2^-24, wider than
// any FP32 rounding cell in [0, 1], so comparisons and ties match exact math.
__device__ __forceinline__ float similarity(const int intersection, const int unionCount) {
  return unionCount == 0 ? 1.0f : __fdiv_rn(static_cast<float>(intersection), static_cast<float>(unionCount));
}

__device__ __forceinline__ bool isVectorAligned(const std::uint32_t* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(uint4) == 0;
}

//! Intersection and union popcounts of words first, first + step, ... of two rows.
__device__ __forceinline__ int2 popcounts(const std::uint32_t* lhs,
                                          const std::uint32_t* rhs,
                                          const int            numWords,
                                          const int            first = 0,
                                          const int            step  = 1) {
  int        intersection = 0;
  int        unionCount   = 0;
  const auto alignment    = reinterpret_cast<std::uintptr_t>(lhs) | reinterpret_cast<std::uintptr_t>(rhs);
  if (numWords % 4 == 0 && alignment % alignof(uint4) == 0) {
    const auto* lhsVectors = reinterpret_cast<const uint4*>(lhs);
    const auto* rhsVectors = reinterpret_cast<const uint4*>(rhs);
    for (int vector = first; vector < numWords / 4; vector += step) {
      const uint4 a = lhsVectors[vector];
      const uint4 b = rhsVectors[vector];
      intersection += __popc(a.x & b.x) + __popc(a.y & b.y) + __popc(a.z & b.z) + __popc(a.w & b.w);
      unionCount += __popc(a.x | b.x) + __popc(a.y | b.y) + __popc(a.z | b.z) + __popc(a.w | b.w);
    }
  } else {
    for (int word = first; word < numWords; word += step) {
      intersection += __popc(lhs[word] & rhs[word]);
      unionCount += __popc(lhs[word] | rhs[word]);
    }
  }
  return make_int2(intersection, unionCount);
}

__device__ __forceinline__ void addRow(Bits& bits, const Tree& tree, const std::uint32_t* row) {
#pragma unroll
  for (int part = 0; part < kBitParts; ++part) {
    const int bit = part * kBlockSize + threadIdx.x;
    if (bit < tree.numBits) {
      bits.parts[part] += (row[bit / 32] >> (bit % 32)) & 1U;
    }
  }
}

__device__ __forceinline__ void addSums(Bits& bits, const Tree& tree, const std::uint32_t* sums) {
#pragma unroll
  for (int part = 0; part < kBitParts; ++part) {
    const int bit = part * kBlockSize + threadIdx.x;
    if (bit < tree.numBits) {
      bits.parts[part] += sums[bit];
    }
  }
}

__device__ __forceinline__ void addEntry(Bits& bits, const Tree& tree, const int entry) {
  const int row = tree.entryRow[entry];
  if (row >= 0) {
    addRow(bits, tree, rowOf(tree, row));
  } else {
    addSums(bits, tree, tree.sums + static_cast<std::size_t>(tree.entrySlot[entry]) * tree.numBits);
  }
}

// Adds numItems rows to bits. resolve(index, isSums) returns item index's packed
// row, or its per-bit sums row with isSums set, or nullptr to skip it. A chunk of
// pointers is resolved in parallel first, so the loads that sum the rows do not
// wait on each other's lookups.
template <typename Resolve>
__device__ void addRows(Bits& bits, const Tree& tree, const int numItems, Scratch& scratch, Resolve resolve) {
  for (int chunk = 0; chunk < numItems; chunk += kBlockSize) {
    const int size = min(kBlockSize, numItems - chunk);
    __syncthreads();  // The previous chunk has been summed.
    if (static_cast<int>(threadIdx.x) < size) {
      bool isSums                    = false;
      scratch.rows[threadIdx.x]      = resolve(chunk + static_cast<int>(threadIdx.x), isSums);
      scratch.rowIsSums[threadIdx.x] = isSums;
    }
    __syncthreads();
    for (int index = 0; index < size; ++index) {
      if (scratch.rows[index] == nullptr) {
        continue;
      }
      if (scratch.rowIsSums[index]) {
        addSums(bits, tree, scratch.rows[index]);
      } else {
        addRow(bits, tree, scratch.rows[index]);
      }
    }
  }
}

//! Block-wide reduction; every thread gets the result.
template <typename Op>
__device__ unsigned long long blockReduce(const unsigned long long value, Op op, Scratch& scratch) {
  __syncthreads();  // Earlier users of the storage and the result are done.
  const unsigned long long result = IntegerReduce(scratch.reduce.integers).Reduce(value, op);
  if (threadIdx.x == 0) {
    scratch.reduced = result;
  }
  __syncthreads();
  return scratch.reduced;
}

__device__ unsigned long long blockSum(const unsigned long long value, Scratch& scratch) {
  return blockReduce(value, cubSum(), scratch);
}

__device__ unsigned long long blockMin(const unsigned long long value, Scratch& scratch) {
  return blockReduce(value, cubMin(), scratch);
}

// Stores bits as the entry's sums, majority centroid and member count, giving a
// single-member entry its summary slot. Every thread of the block calls it after
// it has finished reading the entry.
__device__ void storeEntry(const Tree&         tree,
                           const int           entry,
                           const Bits&         bits,
                           const std::uint32_t count,
                           Scratch&            scratch) {
  __syncthreads();
  if (threadIdx.x == 0) {
    int slot = tree.entrySlot[entry];
    if (slot < 0) {
      slot = atomicAdd(tree.slotCursor, 1);
      if (slot >= tree.maxSlots) {
        *tree.status = kCapacity;
        slot         = -1;
      }
    }
    scratch.broadcast[0] = slot;
  }
  __syncthreads();
  const int slot = scratch.broadcast[0];
  if (slot < 0) {
    return;
  }
  std::uint32_t* sums     = tree.sums + static_cast<std::size_t>(slot) * tree.numBits;
  std::uint32_t* centroid = tree.centroids + static_cast<std::size_t>(slot) * tree.numWords;
#pragma unroll
  for (int part = 0; part < kBitParts; ++part) {
    // numBits and kBlockSize are multiples of 32, so each warp covers one whole word.
    const int           bit  = part * kBlockSize + threadIdx.x;
    const bool          live = bit < tree.numBits;
    const std::uint32_t word =
      __ballot_sync(0xffffffffU, live && bitbirch::majorityCentroidBit(bits.parts[part], count));
    if (live) {
      sums[bit] = bits.parts[part];
      if (threadIdx.x % 32 == 0) {
        centroid[bit / 32] = word;
      }
    }
  }
  if (threadIdx.x == 0) {
    tree.entrySlot[entry]  = slot;
    tree.entryRow[entry]   = -1;
    tree.entryCount[entry] = count;
  }
  __syncthreads();
}

// Whether a cluster with these per-bit sums and count members passes the iSIM
// threshold. Every iSIM term is an integer, so FP64 sums are exact while they
// stay below 2^53, and the fixed reduction order keeps larger sums deterministic.
__device__ bool passesThreshold(const Tree&         tree,
                                const Bits&         bits,
                                const std::uint64_t count,
                                const double        threshold,
                                Scratch&            scratch) {
  bitbirch::ISimTanimotoTerms terms{};
#pragma unroll
  for (int part = 0; part < kBitParts; ++part) {
    if (part * kBlockSize + static_cast<int>(threadIdx.x) < tree.numBits) {
      bitbirch::accumulateISimTanimotoTerm(terms, bits.parts[part], count);
    }
  }
  __syncthreads();  // Earlier users of the reduction storage are done.
  const auto total =
    TermsReduce(scratch.reduce.terms)
      .Reduce(terms, [](const bitbirch::ISimTanimotoTerms& lhs, const bitbirch::ISimTanimotoTerms& rhs) {
        return bitbirch::ISimTanimotoTerms{lhs.commonPairs + rhs.commonPairs, lhs.mismatches + rhs.mismatches};
      });
  if (threadIdx.x == 0) {
    scratch.broadcast[1] = bitbirch::isimTanimotoAtLeast(total, count, threshold);
  }
  __syncthreads();
  return scratch.broadcast[1] != 0;
}

// Index with the greatest value over the block, the lowest index on ties, or -1
// when no thread holds one (index INT_MAX). Values are nonnegative, so their
// FP32 bits order like the values; the low word inverts the index so the lowest
// wins ties, and a key of zero means no index.
__device__ int blockArgMax(const float value, const int index, Scratch& scratch) {
  const unsigned long long key  = index == INT_MAX ? 0ULL :
                                                     static_cast<unsigned long long>(__float_as_uint(value)) << 32 |
                                                      (0xffffffffU - static_cast<std::uint32_t>(index));
  const unsigned long long best = blockReduce(key, cubMax(), scratch);
  return best == 0 ? -1 : static_cast<int>(0xffffffffU - static_cast<std::uint32_t>(best));
}

// Position of the entry most similar to query, the first one on ties, or -1 for
// an empty list. Groups of kLanesPerEntry lanes read one centroid together.
__device__ int closestEntry(const Tree&          tree,
                            const int*           entries,
                            const int            numEntries,
                            const std::uint32_t* query,
                            Scratch&             scratch) {
  const int      lane      = threadIdx.x % 32;
  const int      member    = threadIdx.x % kLanesPerEntry;
  const unsigned groupMask = ((1U << kLanesPerEntry) - 1U) << (lane - member);
  float          best      = -1.0f;
  int            bestIndex = INT_MAX;
  for (int position = threadIdx.x / kLanesPerEntry; position < numEntries; position += kBlockSize / kLanesPerEntry) {
    int2 counts = popcounts(centroidOf(tree, entries[position]), query, tree.numWords, member, kLanesPerEntry);
    for (int offset = kLanesPerEntry / 2; offset > 0; offset /= 2) {
      counts.x += __shfl_xor_sync(groupMask, counts.x, offset, kLanesPerEntry);
      counts.y += __shfl_xor_sync(groupMask, counts.y, offset, kLanesPerEntry);
    }
    // Each group visits ascending positions, so a strict comparison keeps the first tie.
    const float value = similarity(counts.x, counts.y);
    if (value > best) {
      best      = value;
      bestIndex = position;
    }
  }
  return blockArgMax(best, bestIndex, scratch);
}

//! First index of sorted keys[0, count) not below key.
__device__ int lowerBound(const int* keys, int count, const int key) {
  int begin = 0;
  while (begin < count) {
    const int middle = begin + (count - begin) / 2;
    if (keys[middle] < key) {
      begin = middle + 1;
    } else {
      count = middle;
    }
  }
  return begin;
}

//! Allocates an entry; any thread may call it.
__device__ int allocateEntry(const Tree& tree) {
  const int entry = atomicAdd(tree.entryCursor, 1);
  if (entry >= tree.maxEntries) {
    *tree.status = kCapacity;
    return -1;
  }
  tree.entryChild[entry] = -1;
  tree.entryCount[entry] = 0;
  tree.entrySlot[entry]  = -1;
  tree.entryRow[entry]   = -1;
  tree.entryFirst[entry] = INT_MAX;
  return entry;
}

//! Allocates a node; called by one thread at a time, so node IDs follow the split order.
__device__ int allocateNode(const Tree& tree, const bool leaf, const int parent) {
  const int node = (*tree.nodeCursor)++;
  if (node >= tree.maxNodes) {
    *tree.status = kCapacity;
    return -1;
  }
  tree.nodeSize[node]        = 0;
  tree.nodeParent[node]      = parent;
  tree.nodeParentEntry[node] = -1;
  tree.nodeLeaf[node]        = leaf;
  return node;
}

// Sums every entry of node into the summary entry that points to it.
__device__ void summarizeNode(const Tree& tree, const int node, const int entry, Scratch& scratch) {
  const int*         entries = entriesOf(tree, node);
  Bits               bits;
  unsigned long long count = 0;
  addRows(bits, tree, tree.nodeSize[node], scratch, [&](const int position, bool& isSums) {
    const int child = entries[position];
    const int row   = tree.entryRow[child];
    count += tree.entryCount[child];
    isSums = row < 0;
    return isSums ? tree.sums + static_cast<std::size_t>(tree.entrySlot[child]) * tree.numBits : rowOf(tree, row);
  });
  storeEntry(tree, entry, bits, static_cast<std::uint32_t>(blockSum(count, scratch)), scratch);
}

//! Row-major index of pair (lhs, rhs), lhs < rhs, in the upper triangle of size entries.
__host__ __device__ __forceinline__ std::uint32_t pairOrder(const int lhs, const int rhs, const int size) {
  return static_cast<std::uint32_t>(static_cast<std::int64_t>(lhs) * (2LL * size - lhs - 1) / 2 + rhs - lhs - 1);
}

//! Shared-memory bytes of the split pair-search centroid cache and its row pointers and popcounts.
template <int TileSize> constexpr std::size_t pairCacheBytes(const int numWords) {
  return (sizeof(void*) + sizeof(int)) * 2 * TileSize +
         sizeof(std::uint32_t) * static_cast<std::size_t>(2 * TileSize + 1) * numWords;
}

// This thread's least key (similarity bits << 32 | pair order) over the entry
// pairs of tiles lhsTile <= rhsTile. Nonnegative FP32 bit patterns order like
// their values, so the least key is the least-similar pair, first in pair order.
template <int TileSize>
__device__ __forceinline__ unsigned long long tilePairsMin(const Tree&    tree,
                                                           const int*     entries,
                                                           const int      size,
                                                           const int      lhsTile,
                                                           const int      rhsTile,
                                                           unsigned char* shared) {
  constexpr int cacheStride = 2 * TileSize + 1;
  auto**        rows        = reinterpret_cast<const std::uint32_t**>(shared);
  int*          bitCounts   = reinterpret_cast<int*>(rows + 2 * TileSize);
  auto*         cache       = reinterpret_cast<std::uint32_t*>(bitCounts + 2 * TileSize);
  const bool    diagonal    = lhsTile == rhsTile;
  const int     lhsBegin    = lhsTile * TileSize;
  const int     rhsBegin    = rhsTile * TileSize;
  const int     lhsCount    = min(TileSize, size - lhsBegin);
  const int     rhsCount    = diagonal ? lhsCount : min(TileSize, size - rhsBegin);
  const int     rhsColumn   = diagonal ? 0 : TileSize;
  __syncthreads();  // The previous tile pair has finished reading the cache.
  for (int column = threadIdx.x; column < 2 * TileSize; column += kBlockSize) {
    const int  position = column < TileSize ? lhsBegin + column : rhsBegin + column - TileSize;
    const bool used     = column < TileSize ? column < lhsCount : !diagonal && column - TileSize < rhsCount;
    rows[column]        = used ? centroidOf(tree, entries[position]) : nullptr;
  }
  __syncthreads();
  // Word-major with an odd stride, so pair reads of one word spread over banks.
  for (int item = threadIdx.x; item < 2 * TileSize * tree.numWords; item += kBlockSize) {
    const int column = item / tree.numWords;
    const int word   = item % tree.numWords;
    if (rows[column] != nullptr) {
      cache[word * cacheStride + column] = rows[column][word];
    }
  }
  __syncthreads();
  for (int column = threadIdx.x; column < 2 * TileSize; column += kBlockSize) {
    int count = 0;
    for (int word = 0; rows[column] != nullptr && word < tree.numWords; ++word) {
      count += __popc(cache[word * cacheStride + column]);
    }
    bitCounts[column] = count;
  }
  __syncthreads();
  unsigned long long best = ULLONG_MAX;
  for (int pair = threadIdx.x; pair < lhsCount * rhsCount; pair += kBlockSize) {
    const int lhs = pair / rhsCount;
    const int rhs = pair % rhsCount;
    if (diagonal && rhs <= lhs) {
      continue;
    }
    int intersection = 0;
    for (int word = 0; word < tree.numWords; ++word) {
      intersection += __popc(cache[word * cacheStride + lhs] & cache[word * cacheStride + rhsColumn + rhs]);
    }
    const int   unionCount = bitCounts[lhs] + bitCounts[rhsColumn + rhs] - intersection;
    const float value      = similarity(intersection, unionCount);
    const auto  key =
      static_cast<unsigned long long>(__float_as_uint(value)) << 32 | pairOrder(lhsBegin + lhs, rhsBegin + rhs, size);
    best = min(best, key);
  }
  return best;
}

//! Tile pair index work of numTiles tiles, row-major over lhs <= rhs.
__device__ int2 tilePair(int work, const int numTiles) {
  int lhs = 0;
  while (work >= numTiles - lhs) {
    work -= numTiles - lhs;
    ++lhs;
  }
  return make_int2(lhs, lhs + work);
}

//! Inverts pairOrder.
__device__ int2 pairFromOrder(const std::uint32_t order, const int size) {
  int       lhs       = 0;
  long long remaining = order;
  while (remaining >= size - lhs - 1) {
    remaining -= size - lhs - 1;
    ++lhs;
  }
  return make_int2(lhs, lhs + 1 + static_cast<int>(remaining));
}

// Splits size entries around the seeds at lhsSeed and rhsSeed: each other entry,
// in order, joins the seed it is more similar to (the smaller half on ties),
// while neither half exceeds (size + 1) / 2. Writes the left half, then the
// right half, to plan and returns the left size. sides holds size bytes.
__device__ int partitionEntries(const Tree&  tree,
                                const int*   entries,
                                const int    size,
                                const int    lhsSeed,
                                const int    rhsSeed,
                                int*         plan,
                                std::int8_t* sides,
                                Scratch&     scratch) {
  const std::uint32_t* lhsCentroid = centroidOf(tree, entries[lhsSeed]);
  const std::uint32_t* rhsCentroid = centroidOf(tree, entries[rhsSeed]);
  for (int position = threadIdx.x; position < size; position += kBlockSize) {
    const std::uint32_t* centroid = centroidOf(tree, entries[position]);
    const int2           lhs      = popcounts(centroid, lhsCentroid, tree.numWords);
    const int2           rhs      = popcounts(centroid, rhsCentroid, tree.numWords);
    const float          lhsValue = similarity(lhs.x, lhs.y);
    const float          rhsValue = similarity(rhs.x, rhs.y);
    sides[position]               = static_cast<std::int8_t>((lhsValue > rhsValue) - (lhsValue < rhsValue));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int capacity = (size + 1) / 2;
    int       lhsCount = 1;
    int       rhsCount = 1;
    for (int position = 0; position < size; ++position) {
      if (position == lhsSeed || position == rhsSeed) {
        continue;
      }
      bool left = sides[position] > 0 || (sides[position] == 0 && lhsCount <= rhsCount);
      if (lhsCount >= capacity) {
        left = false;
      } else if (rhsCount >= capacity) {
        left = true;
      }
      sides[position] = left ? 1 : -1;
      left ? ++lhsCount : ++rhsCount;
    }
    sides[lhsSeed] = 1;
    sides[rhsSeed] = -1;
    int lhs        = 1;
    int rhs        = lhsCount + 1;
    plan[0]        = entries[lhsSeed];
    plan[lhsCount] = entries[rhsSeed];
    for (int position = 0; position < size; ++position) {
      if (position != lhsSeed && position != rhsSeed) {
        plan[sides[position] > 0 ? lhs++ : rhs++] = entries[position];
      }
    }
    scratch.broadcast[0] = lhsCount;
  }
  __syncthreads();
  return scratch.broadcast[0];
}

// Gives node the first leftSize plan entries and a new sibling the rest, then
// links the sibling beside node: as a new entry of the shared parent, or under a
// new root. Returns the parent, or -1 on failure. Every thread calls it.
__device__ int splitNode(const Tree& tree, const int node, const int* plan, const int leftSize, Scratch& scratch) {
  const int  size = tree.nodeSize[node];
  const bool leaf = tree.nodeLeaf[node] != 0;
  __syncthreads();
  if (threadIdx.x == 0) {
    scratch.broadcast[0] = allocateNode(tree, leaf, tree.nodeParent[node]);
  }
  __syncthreads();
  const int sibling = scratch.broadcast[0];
  if (sibling < 0) {
    return -1;
  }
  int* nodeEntries    = entriesOf(tree, node);
  int* siblingEntries = entriesOf(tree, sibling);
  for (int position = threadIdx.x; position < size; position += kBlockSize) {
    const int entry  = plan[position];
    const int target = position < leftSize ? node : sibling;
    (position < leftSize ? nodeEntries[position] : siblingEntries[position - leftSize]) = entry;
    if (!leaf) {
      tree.nodeParent[tree.entryChild[entry]] = target;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    tree.nodeSize[node]    = leftSize;
    tree.nodeSize[sibling] = size - leftSize;
    int parent             = tree.nodeParent[node];
    if (parent < 0) {
      parent              = allocateNode(tree, false, -1);
      const int nodeEntry = parent >= 0 ? allocateEntry(tree) : -1;
      if (nodeEntry >= 0) {
        tree.nodeParent[node]      = parent;
        tree.nodeParent[sibling]   = parent;
        tree.nodeParentEntry[node] = nodeEntry;
        tree.entryChild[nodeEntry] = node;
        entriesOf(tree, parent)[0] = nodeEntry;
        tree.nodeSize[parent]      = 1;
        *tree.root                 = parent;
      } else {
        parent = -1;
      }
    }
    const int siblingEntry = parent >= 0 ? allocateEntry(tree) : -1;
    if (siblingEntry >= 0) {
      tree.nodeParentEntry[sibling]                    = siblingEntry;
      tree.entryChild[siblingEntry]                    = sibling;
      entriesOf(tree, parent)[tree.nodeSize[parent]++] = siblingEntry;
    }
    scratch.broadcast[1] = siblingEntry >= 0 ? parent : -1;
    scratch.broadcast[0] = sibling;
  }
  __syncthreads();
  return scratch.broadcast[1];
}

__global__ void initializeKernel(Tree tree) {
  *tree.nodeCursor  = 0;
  *tree.entryCursor = 0;
  *tree.slotCursor  = 0;
  *tree.status      = kSuccess;
  *tree.root        = allocateNode(tree, true, -1);
}

// One block per batch row: descend the tree snapshot to a leaf. Groups also
// test the leaf's closest entry, so rows that would each merge into the same
// entry can be merged together.
__global__ void routeKernel(const Tree   tree,
                            const bool   groups,
                            const double threshold,
                            int*         leafKeys,
                            int*         groupKeys,
                            int*         values) {
  __shared__ Scratch scratch;
  const int          offset = blockIdx.x;
  const int          row    = tree.batchBegin + offset;
  if (threadIdx.x == 0) {
    values[offset]    = offset;
    leafKeys[offset]  = INT_MAX;
    groupKeys[offset] = INT_MAX;
  }
  if (tree.labels[row] >= 0) {
    return;
  }
  const std::uint32_t* query = rowOf(tree, row);
  int                  node  = *tree.root;
  while (!tree.nodeLeaf[node]) {
    const int* entries = entriesOf(tree, node);
    node               = tree.entryChild[entries[closestEntry(tree, entries, tree.nodeSize[node], query, scratch)]];
  }
  if (threadIdx.x == 0) {
    leafKeys[offset] = node;
  }
  if (!groups || tree.nodeSize[node] == 0) {
    return;
  }
  const int* entries = entriesOf(tree, node);
  const int  entry   = entries[closestEntry(tree, entries, tree.nodeSize[node], query, scratch)];
  Bits       bits;
  addEntry(bits, tree, entry);
  addRow(bits, tree, query);
  if (passesThreshold(tree, bits, tree.entryCount[entry] + 1ULL, threshold, scratch) && threadIdx.x == 0) {
    groupKeys[offset] = entry;
  }
}

// One block per run of rows sorted by the entry they each passed: merge the
// whole run if the combined cluster passes, otherwise each row in order.
__global__ void groupsKernel(const Tree   tree,
                             const int    count,
                             const int*   keys,
                             const int*   values,
                             const double threshold) {
  __shared__ Scratch scratch;
  const int          first = blockIdx.x;
  const int          entry = keys[first];
  if (entry == INT_MAX || (first > 0 && keys[first - 1] == entry)) {
    return;
  }
  int end = first + 1;
  while (end < count && keys[end] == entry) {
    ++end;
  }
  Bits bits;
  addEntry(bits, tree, entry);
  addRows(bits, tree, end - first, scratch, [&](const int index, bool&) {
    return rowOf(tree, tree.batchBegin + values[first + index]);
  });
  const std::uint32_t combined = tree.entryCount[entry] + (end - first);
  if (passesThreshold(tree, bits, combined, threshold, scratch)) {
    storeEntry(tree, entry, bits, combined, scratch);
    for (int index = first + threadIdx.x; index < end; index += kBlockSize) {
      tree.labels[tree.batchBegin + values[index]] = entry;
    }
    if (threadIdx.x == 0) {
      tree.entryFirst[entry] = min(tree.entryFirst[entry], tree.batchBegin + values[first]);
    }
    return;
  }
  for (int index = first; index < end; ++index) {
    const int row = tree.batchBegin + values[index];
    Bits      single;
    addEntry(single, tree, entry);
    addRow(single, tree, rowOf(tree, row));
    const std::uint32_t merged = tree.entryCount[entry] + 1;
    if (passesThreshold(tree, single, merged, threshold, scratch)) {
      storeEntry(tree, entry, single, merged, scratch);
      if (threadIdx.x == 0) {
        tree.labels[row]       = entry;
        tree.entryFirst[entry] = min(tree.entryFirst[entry], row);
      }
    }
  }
}

//! Most pending rows a leaf owner compares with its leaf in one pass.
constexpr int         kOwnerBatch        = 16;
//! Device bytes for leaf-owner counts; wider batches give way to narrower ones past it.
constexpr std::size_t kOwnerScratchBytes = std::size_t{128} << 20;

//! Intersection and union popcounts packed in one word; both are at most 4096.
__device__ __forceinline__ std::uint32_t packCounts(const int intersection, const int unionCount) {
  return static_cast<std::uint32_t>(intersection) | static_cast<std::uint32_t>(unionCount) << 16;
}

__device__ __forceinline__ float packedSimilarity(const std::uint32_t counts) {
  return similarity(static_cast<int>(counts & 0xffffU), static_cast<int>(counts >> 16));
}

//! Rows of one leaf-owner batch, staged with their popcounts.
struct OwnerBatch {
  std::uint32_t queries[kOwnerBatch][kBitBirchMaxWords];
  int           queryBits[kOwnerBatch];
  std::uint32_t queryPairs[kOwnerBatch][kOwnerBatch];  //!< Packed counts of rows [later][earlier].
  int           pending[kBlockSize];
  int           warpOffsets[kNumWarps + 1];
};

// Inserts width pending rows into the leaf in order, as if each searched the
// leaf alone. One pass first counts every row against every leaf position into
// counts (row-major, stride per row); a merge then recounts the merged position
// for the later rows, and a new entry takes their counts against its row.
// Returns whether the leaf overflowed, leaving the remaining rows pending.
__device__ bool insertRows(const Tree&    tree,
                           const int      node,
                           int*           entries,
                           const int*     rows,
                           const int      width,
                           std::uint32_t* counts,
                           const double   threshold,
                           OwnerBatch&    batch,
                           Scratch&       scratch) {
  const int numWords = tree.numWords;
  for (int item = threadIdx.x; item < width * numWords; item += kBlockSize) {
    batch.queries[item / numWords][item % numWords] = rowOf(tree, rows[item / numWords])[item % numWords];
  }
  __syncthreads();
  for (int query = threadIdx.x; query < width; query += kBlockSize) {
    int bits = 0;
    for (int word = 0; word < numWords; ++word) {
      bits += __popc(batch.queries[query][word]);
    }
    batch.queryBits[query] = bits;
  }
  __syncthreads();
  for (int pair = threadIdx.x; pair < width * width; pair += kBlockSize) {
    const int later   = pair / width;
    const int earlier = pair % width;
    if (earlier < later) {
      int intersection = 0;
      for (int word = 0; word < numWords; ++word) {
        intersection += __popc(batch.queries[later][word] & batch.queries[earlier][word]);
      }
      batch.queryPairs[later][earlier] =
        packCounts(intersection, batch.queryBits[later] + batch.queryBits[earlier] - intersection);
    }
  }
  int size = tree.nodeSize[node];
  // One thread per position reads its centroid once for every row.
  for (int position = threadIdx.x; position < size; position += kBlockSize) {
    const std::uint32_t* centroid                   = centroidOf(tree, entries[position]);
    int                  intersections[kOwnerBatch] = {};
    int                  centroidBits               = 0;
    const auto           countWord                  = [&](const std::uint32_t value, const int word) {
      centroidBits += __popc(value);
#pragma unroll
      for (int query = 0; query < kOwnerBatch; ++query) {
        if (query < width) {
          intersections[query] += __popc(value & batch.queries[query][word]);
        }
      }
    };
    if (numWords % 4 == 0 && isVectorAligned(centroid)) {
      // 16-byte loads, several in flight: only this thread reads the row.
#pragma unroll 4
      for (int vector = 0; vector < numWords / 4; ++vector) {
        const uint4 value = reinterpret_cast<const uint4*>(centroid)[vector];
        countWord(value.x, 4 * vector);
        countWord(value.y, 4 * vector + 1);
        countWord(value.z, 4 * vector + 2);
        countWord(value.w, 4 * vector + 3);
      }
    } else {
      for (int word = 0; word < numWords; ++word) {
        countWord(centroid[word], word);
      }
    }
#pragma unroll
    for (int query = 0; query < kOwnerBatch; ++query) {
      if (query < width) {
        counts[query * tree.stride + position] =
          packCounts(intersections[query], centroidBits + batch.queryBits[query] - intersections[query]);
      }
    }
  }
  __syncthreads();
  for (int query = 0; query < width; ++query) {
    const int row    = rows[query];
    int       entry  = -1;
    bool      merged = false;
    if (size > 0) {
      float best      = -1.0f;
      int   bestIndex = INT_MAX;
      for (int position = threadIdx.x; position < size; position += kBlockSize) {
        const float value = packedSimilarity(counts[query * tree.stride + position]);
        if (value > best) {
          best      = value;
          bestIndex = position;
        }
      }
      const int position              = blockArgMax(best, bestIndex, scratch);
      entry                           = entries[position];
      // A single-member entry and the row sum to 2 on shared bits and 1 on the
      // others, so the merged iSIM terms are exactly |a & b| and |a | b| - |a & b|.
      const bool          singleton   = tree.entryRow[entry] >= 0;
      const std::uint32_t mergedCount = tree.entryCount[entry] + 1;
      if (singleton) {
        const std::uint32_t pair   = counts[query * tree.stride + position];
        const double        shared = pair & 0xffffU;
        merged                     = bitbirch::isimTanimotoAtLeast({shared, (pair >> 16) - shared}, 2, threshold);
      }
      Bits bits;
      if (!singleton || merged) {
        addEntry(bits, tree, entry);
        addRow(bits, tree, batch.queries[query]);
      }
      if (!singleton) {
        merged = passesThreshold(tree, bits, mergedCount, threshold, scratch);
      }
      if (merged) {
        storeEntry(tree, entry, bits, mergedCount, scratch);
        // One warp per later row recounts the merged centroid.
        const std::uint32_t* centroid = centroidOf(tree, entry);
        for (int later = query + 1 + static_cast<int>(threadIdx.x / 32); later < width; later += kNumWarps) {
          int intersection = 0;
          int centroidBits = 0;
          for (int word = threadIdx.x % 32; word < numWords; word += 32) {
            intersection += __popc(centroid[word] & batch.queries[later][word]);
            centroidBits += __popc(centroid[word]);
          }
          for (int offset = 16; offset > 0; offset /= 2) {
            intersection += __shfl_xor_sync(0xffffffffU, intersection, offset);
            centroidBits += __shfl_xor_sync(0xffffffffU, centroidBits, offset);
          }
          if (threadIdx.x % 32 == 0) {
            counts[later * tree.stride + position] =
              packCounts(intersection, centroidBits + batch.queryBits[later] - intersection);
          }
        }
      }
    }
    // Only thread zero allocates, so the new entry stays in its own register: a
    // shared broadcast here could overwrite the merge decision before every warp
    // has read it.
    if (!merged) {
      if (threadIdx.x == 0) {
        entry = allocateEntry(tree);
        if (entry >= 0) {
          tree.entryRow[entry]   = row;
          tree.entryCount[entry] = 1;
          entries[size]          = entry;
          tree.nodeSize[node]    = size + 1;
        }
      }
      // The new entry's centroid is this row.
      for (int later = query + 1 + static_cast<int>(threadIdx.x); later < width; later += kBlockSize) {
        counts[later * tree.stride + size] = batch.queryPairs[later][query];
      }
      ++size;
    }
    if (threadIdx.x == 0) {
      if (entry >= 0) {
        tree.labels[row]       = entry;
        tree.entryFirst[entry] = min(tree.entryFirst[entry], row);
      }
      // Other blocks may fail while this one runs, so one read decides for every thread.
      scratch.broadcast[2] = *tree.status;
    }
    __syncthreads();
    if (size > tree.branchingFactor || scratch.broadcast[2] != kSuccess) {
      return true;
    }
  }
  return false;
}

// One block per run of rows sorted by leaf: the leaf's only writer. Pending
// rows, in order, merge into their closest entry or start a new one, until the
// leaf overflows; later rows are routed again after the split. counts holds
// width * (branchingFactor + 1) words per run.
__global__ void leafOwnersKernel(const Tree     tree,
                                 const int      count,
                                 const int*     keys,
                                 const int*     values,
                                 const double   threshold,
                                 const int      depth,
                                 int*           dirty,
                                 int*           levelNodes,
                                 int*           levelCounts,
                                 const int      levelStride,
                                 int*           splitQueue,
                                 int*           splitCount,
                                 std::uint32_t* countScratch,
                                 const int      width) {
  __shared__ Scratch    scratch;
  __shared__ OwnerBatch batch;
  const int             first = blockIdx.x;
  const int             node  = keys[first];
  if (node == INT_MAX || (first > 0 && keys[first - 1] == node)) {
    return;
  }
  int*           entries    = entriesOf(tree, node);
  std::uint32_t* counts     = countScratch + static_cast<std::size_t>(first) * width * tree.stride;
  bool           overflowed = false;
  for (int chunk = first; !overflowed && chunk < count && keys[chunk] == node; chunk += kBlockSize) {
    // List this chunk's pending rows of the run in order.
    const int      index   = chunk + static_cast<int>(threadIdx.x);
    const int      row     = index < count && keys[index] == node ? tree.batchBegin + values[index] : -1;
    const bool     pending = row >= 0 && tree.labels[row] < 0;
    const unsigned ballot  = __ballot_sync(0xffffffffU, pending);
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
      batch.warpOffsets[threadIdx.x / 32 + 1] = __popc(ballot);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      batch.warpOffsets[0] = 0;
      for (int warp = 0; warp < kNumWarps; ++warp) {
        batch.warpOffsets[warp + 1] += batch.warpOffsets[warp];
      }
    }
    __syncthreads();
    if (pending) {
      batch.pending[batch.warpOffsets[threadIdx.x / 32] + __popc(ballot & ((1U << (threadIdx.x % 32)) - 1U))] = row;
    }
    __syncthreads();
    const int numPending = batch.warpOffsets[kNumWarps];
    for (int begin = 0; begin < numPending && !overflowed; begin += width) {
      overflowed = insertRows(tree,
                              node,
                              entries,
                              batch.pending + begin,
                              min(width, numPending - begin),
                              counts,
                              threshold,
                              batch,
                              scratch);
    }
  }
  if (threadIdx.x != 0) {
    return;
  }
  if (tree.nodeSize[node] > tree.branchingFactor) {
    splitQueue[atomicAdd(splitCount, 1)] = first;
  }
  // Every leaf has the same depth. The first owner to dirty an ancestor lists it at its level.
  int level = depth;
  for (int changed = node; tree.nodeParent[changed] >= 0; changed = tree.nodeParent[changed], --level) {
    if (atomicExch(dirty + changed, 1) == 0) {
      levelNodes[level * levelStride + atomicAdd(levelCounts + level, 1)] = changed;
    }
  }
}

// One block per dirty node of one level, deepest level first: add the round's
// accepted rows (leaves) or the children's deltas (internal nodes) to the
// summary entry that points to the node, and record the sum as this node's delta.
__global__ void refreshKernel(const Tree     tree,
                              const int*     nodes,
                              const int*     numNodes,
                              const int      count,
                              const int*     keys,
                              const int*     values,
                              int*           dirty,
                              int*           deltaSlots,
                              int*           deltaCursor,
                              const int      deltaCapacity,
                              std::uint32_t* deltaSums,
                              std::uint32_t* deltaCounts) {
  __shared__ Scratch scratch;
  if (static_cast<int>(blockIdx.x) >= *numNodes) {
    return;
  }
  const int node   = nodes[blockIdx.x];
  const int target = tree.nodeParentEntry[node];
  if (threadIdx.x == 0) {
    const int slot = atomicAdd(deltaCursor, 1);
    if (slot >= deltaCapacity || target < 0) {
      *tree.status = target < 0 ? kInvalidTree : kCapacity;
    }
    deltaSlots[node]     = slot;
    scratch.broadcast[0] = slot;
    // The node's run in the sorted batch, for leaves.
    scratch.broadcast[1] = lowerBound(keys, count, node);
    scratch.broadcast[2] = lowerBound(keys, count, node + 1);
  }
  __syncthreads();
  const int slot = scratch.broadcast[0];
  if (slot >= deltaCapacity || target < 0) {
    return;
  }
  Bits               delta;
  unsigned long long added = 0;
  if (tree.nodeLeaf[node]) {
    const int begin = scratch.broadcast[1];
    const int end   = scratch.broadcast[2];
    addRows(delta, tree, end - begin, scratch, [&](const int index, bool&) -> const std::uint32_t* {
      const int row = tree.batchBegin + values[begin + index];
      if (tree.labels[row] < 0) {
        return nullptr;
      }
      ++added;
      return rowOf(tree, row);
    });
  } else {
    const int* entries = entriesOf(tree, node);
    addRows(delta, tree, tree.nodeSize[node], scratch, [&](const int position, bool& isSums) -> const std::uint32_t* {
      const int childSlot = deltaSlots[tree.entryChild[entries[position]]];
      if (childSlot < 0) {
        return nullptr;
      }
      added += deltaCounts[childSlot];
      isSums = true;
      return deltaSums + static_cast<std::size_t>(childSlot) * tree.numBits;
    });
  }
  added                   = blockSum(added, scratch);
  std::uint32_t* deltaRow = deltaSums + static_cast<std::size_t>(slot) * tree.numBits;
#pragma unroll
  for (int part = 0; part < kBitParts; ++part) {
    const int bit = part * kBlockSize + threadIdx.x;
    if (bit < tree.numBits) {
      deltaRow[bit] = delta.parts[part];
    }
  }
  if (threadIdx.x == 0) {
    deltaCounts[slot] = static_cast<std::uint32_t>(added);
    dirty[node]       = 0;
  }
  addEntry(delta, tree, target);
  storeEntry(tree, target, delta, tree.entryCount[target] + static_cast<std::uint32_t>(added), scratch);
}

// Exact least-similar pair of every overflowing leaf, spread over blocks by tile pair.
template <int TileSize>
__global__ void splitPairsKernel(const Tree          tree,
                                 const int*          keys,
                                 const int*          splitQueue,
                                 const int*          splitCount,
                                 unsigned long long* pairKeys) {
  extern __shared__ uint4 dynamicShared[];
  __shared__ Scratch      scratch;
  const int               size     = tree.branchingFactor + 1;
  const int               numTiles = (size + TileSize - 1) / TileSize;
  const int               perLeaf  = numTiles * (numTiles + 1) / 2;
  const long long         total    = static_cast<long long>(*splitCount) * perLeaf;
  for (long long work = blockIdx.x; work < total; work += gridDim.x) {
    const int  index = splitQueue[work / perLeaf];
    const int2 tiles = tilePair(static_cast<int>(work % perLeaf), numTiles);
    const auto best  = blockMin(tilePairsMin<TileSize>(tree,
                                                      entriesOf(tree, keys[index]),
                                                      size,
                                                      tiles.x,
                                                      tiles.y,
                                                      reinterpret_cast<unsigned char*>(dynamicShared)),
                               scratch);
    if (threadIdx.x == 0) {
      atomicMin(pairKeys + index, best);
    }
  }
}

// One block per overflowing leaf: split its entries around the pair found by splitPairsKernel.
__global__ void splitPlansKernel(const Tree          tree,
                                 const int*          keys,
                                 const int*          splitQueue,
                                 const int*          splitCount,
                                 unsigned long long* pairKeys,
                                 int*                plans,
                                 int*                leftSizes,
                                 std::int8_t*        sides) {
  __shared__ Scratch scratch;
  if (static_cast<int>(blockIdx.x) >= *splitCount) {
    return;
  }
  const int  index = splitQueue[blockIdx.x];
  const int  size  = tree.branchingFactor + 1;
  const int2 seeds = pairFromOrder(static_cast<std::uint32_t>(pairKeys[index]), size);
  const int  left  = partitionEntries(tree,
                                    entriesOf(tree, keys[index]),
                                    size,
                                    seeds.x,
                                    seeds.y,
                                    plans + static_cast<std::size_t>(index) * tree.stride,
                                    sides + static_cast<std::size_t>(blockIdx.x) * tree.stride,
                                    scratch);
  if (threadIdx.x == 0) {
    leftSizes[index] = left;
    pairKeys[index]  = ULLONG_MAX;
  }
}

// Applies the split of the overflowing leaf at index of the sorted batch, and
// any parent splits it cascades into. The leaf halves' summaries are deferred.
template <int TileSize>
__device__ __forceinline__ void splitLeaf(const Tree&    tree,
                                          const int      index,
                                          const int*     keys,
                                          const int*     plans,
                                          const int*     leftSizes,
                                          int*           innerPlan,
                                          std::int8_t*   sides,
                                          int*           deferredNodes,
                                          int*           deferredEntries,
                                          int&           deferred,
                                          unsigned char* shared,
                                          Scratch&       scratch) {
  int        node     = keys[index];
  const int* plan     = plans + static_cast<std::size_t>(index) * tree.stride;
  int        leftSize = leftSizes[index];
  for (bool leaf = true;; leaf = false) {
    const int parent = splitNode(tree, node, plan, leftSize, scratch);
    if (parent < 0) {
      return;
    }
    const int sibling = scratch.broadcast[0];
    if (leaf) {
      if (threadIdx.x == 0) {
        deferredNodes[deferred]       = node;
        deferredEntries[deferred]     = tree.nodeParentEntry[node];
        deferredNodes[deferred + 1]   = sibling;
        deferredEntries[deferred + 1] = tree.nodeParentEntry[sibling];
      }
      deferred += 2;
    } else {
      // Inner halves are summarized now, while every child summary is current.
      // Deferred, they could read new entries that later leaf splits add below them.
      summarizeNode(tree, node, tree.nodeParentEntry[node], scratch);
      summarizeNode(tree, sibling, tree.nodeParentEntry[sibling], scratch);
    }
    if (tree.nodeSize[parent] <= tree.branchingFactor) {
      return;
    }
    __syncthreads();
    for (int pending = 0; pending < deferred; ++pending) {
      summarizeNode(tree, deferredNodes[pending], deferredEntries[pending], scratch);
    }
    deferred                    = 0;
    node                        = parent;
    // Seeds of the overflowing parent, searched by this block alone.
    const int          size     = tree.nodeSize[node];
    const int          numTiles = (size + TileSize - 1) / TileSize;
    const int*         entries  = entriesOf(tree, node);
    unsigned long long best     = ULLONG_MAX;
    for (int work = 0; work < numTiles * (numTiles + 1) / 2; ++work) {
      const int2 tiles = tilePair(work, numTiles);
      best             = min(best, tilePairsMin<TileSize>(tree, entries, size, tiles.x, tiles.y, shared));
    }
    const int2 seeds = pairFromOrder(static_cast<std::uint32_t>(blockMin(best, scratch)), size);
    leftSize         = partitionEntries(tree, entries, size, seeds.x, seeds.y, innerPlan, sides, scratch);
    plan             = innerPlan;
  }
}

// One block applies every leaf split in leaf order, as serial insertion would,
// so node IDs and parent entry order are deterministic. A parent that overflows
// splits at once, cascading upward, after the summaries it needs are current.
// Only the two summaries of each leaf split are left to summarizeKernel.
template <int TileSize>
__global__ void repairKernel(const Tree   tree,
                             const int    count,
                             const int*   keys,
                             const int*   plans,
                             const int*   leftSizes,
                             int*         innerPlan,
                             std::int8_t* sides,
                             int*         deferredNodes,
                             int*         deferredEntries,
                             int*         numDeferred) {
  extern __shared__ uint4 dynamicShared[];
  __shared__ Scratch      scratch;
  __shared__ int          queue[kBlockSize];
  __shared__ int          warpOffsets[kNumWarps + 1];
  int                     deferred = 0;
  for (int base = 0; base < count; base += kBlockSize) {
    // A leaf changes size only through its own split, so the chunk's overflowing
    // leaves are listed up front, in leaf order.
    const int  candidate = base + static_cast<int>(threadIdx.x);
    const bool overflow  = candidate < count && keys[candidate] != INT_MAX &&
                          (candidate == 0 || keys[candidate - 1] != keys[candidate]) &&
                          tree.nodeSize[keys[candidate]] > tree.branchingFactor;
    const unsigned ballot = __ballot_sync(0xffffffffU, overflow);
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
      warpOffsets[threadIdx.x / 32 + 1] = __popc(ballot);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      warpOffsets[0] = 0;
      for (int warp = 0; warp < kNumWarps; ++warp) {
        warpOffsets[warp + 1] += warpOffsets[warp];
      }
    }
    __syncthreads();
    if (overflow) {
      queue[warpOffsets[threadIdx.x / 32] + __popc(ballot & ((1U << (threadIdx.x % 32)) - 1U))] = candidate;
    }
    __syncthreads();
    for (int queued = 0; queued < warpOffsets[kNumWarps] && *tree.status == kSuccess; ++queued) {
      splitLeaf<TileSize>(tree,
                          queue[queued],
                          keys,
                          plans,
                          leftSizes,
                          innerPlan,
                          sides,
                          deferredNodes,
                          deferredEntries,
                          deferred,
                          reinterpret_cast<unsigned char*>(dynamicShared),
                          scratch);
    }
  }
  if (threadIdx.x == 0) {
    *numDeferred = deferred;
  }
}

__global__ void summarizeKernel(const Tree tree, const int* nodes, const int* entries, const int* numDeferred) {
  __shared__ Scratch scratch;
  if (static_cast<int>(blockIdx.x) < *numDeferred) {
    summarizeNode(tree, nodes[blockIdx.x], entries[blockIdx.x], scratch);
  }
}

// Publishes the round's controls and resets per-round scratch for the next one.
__global__ void controlKernel(const Tree tree,
                              const int  count,
                              int*       control,
                              int*       levelNodes,
                              int*       levelCounts,
                              const int  numLevels,
                              const int  levelStride,
                              int*       deltaSlots,
                              int*       deltaCursor,
                              int*       splitCount) {
  __shared__ int pending;
  if (threadIdx.x == 0) {
    pending = 0;
  }
  __syncthreads();
  int local = 0;
  for (int offset = threadIdx.x; offset < count; offset += blockDim.x) {
    local += tree.labels[tree.batchBegin + offset] < 0;
  }
  atomicAdd(&pending, local);
  for (int level = 0; level < numLevels; ++level) {
    for (int index = threadIdx.x; index < levelCounts[level]; index += blockDim.x) {
      deltaSlots[levelNodes[level * levelStride + index]] = -1;
    }
  }
  __syncthreads();
  for (int level = threadIdx.x; level < numLevels; level += blockDim.x) {
    levelCounts[level] = 0;
  }
  if (threadIdx.x == 0) {
    int depth = 0;
    // Node zero stays a leaf, and every leaf has the same depth.
    for (int node = 0; tree.nodeParent[node] >= 0; node = tree.nodeParent[node]) {
      ++depth;
    }
    control[kPending] = pending;
    control[kNodes]   = *tree.nodeCursor;
    control[kEntries] = *tree.entryCursor;
    control[kSlots]   = *tree.slotCursor;
    control[kStatus]  = *tree.status;
    control[kDepth]   = depth;
    *deltaCursor      = 0;
    *splitCount       = 0;
  }
}

__global__ void collectLeavesKernel(const Tree tree,
                                    const int  numEntries,
                                    int*       firstRows,
                                    int*       leafEntries,
                                    int*       cursor) {
  const int entry = blockIdx.x * blockDim.x + threadIdx.x;
  if (entry < numEntries && tree.entryChild[entry] < 0) {
    const int slot    = atomicAdd(cursor, 1);
    firstRows[slot]   = tree.entryFirst[entry];
    leafEntries[slot] = entry;
  }
}

// One thread per centroid word, or per cluster without centroids, so the copy is coalesced.
__global__ void numberClustersKernel(const Tree     tree,
                                     const int      numClusters,
                                     const int      width,
                                     const int*     leafEntries,
                                     std::uint32_t* centroids) {
  const std::size_t item = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item < static_cast<std::size_t>(numClusters) * width) {
    const int cluster = static_cast<int>(item / width);
    const int word    = static_cast<int>(item % width);
    const int entry   = leafEntries[cluster];
    if (centroids != nullptr) {
      centroids[item] = centroidOf(tree, entry)[word];
    }
    if (word == 0) {
      tree.entryFirst[entry] = cluster;
    }
  }
}

__global__ void labelRowsKernel(const Tree tree, const int numRows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < numRows) {
    tree.labels[row] = tree.entryFirst[tree.labels[row]];
  }
}

//! Bits of a radix sort over IDs below capacity and INT_MAX, which then sorts last.
int keyBits(const std::int64_t capacity) {
  int bits = 1;
  while (bits < 31 && (std::int64_t{1} << bits) <= capacity) {
    ++bits;
  }
  return bits;
}

template <typename T> void growTo(AsyncDeviceVector<T>& vector, const std::size_t size) {
  if (vector.size() < size) {
    vector.resize(std::max(size, vector.size() + vector.size() / 2));
  }
}

//! Resizes scratch whose contents are rewritten before use, without copying them.
template <typename T> void reserveScratch(AsyncDeviceVector<T>& vector, const std::size_t size) {
  if (vector.size() < size) {
    vector.resize(0);
    vector.resize(size);
  }
}

template <int TileSize> bool splitCacheFits(const int numWords, const int sharedMemoryBudget) {
  if (sharedMemoryBudget <= 0) {
    return false;
  }
  cudaFuncAttributes pairAttributes{};
  cudaFuncAttributes repairAttributes{};
  cudaCheckError(cudaFuncGetAttributes(&pairAttributes, splitPairsKernel<TileSize>));
  cudaCheckError(cudaFuncGetAttributes(&repairAttributes, repairKernel<TileSize>));
  const std::size_t staticBytes = std::max(pairAttributes.sharedSizeBytes, repairAttributes.sharedSizeBytes);
  return pairCacheBytes<TileSize>(numWords) + staticBytes <= static_cast<std::size_t>(sharedMemoryBudget);
}

template <int TileSize> std::size_t configureSplitCache(const int numWords, const int deviceSharedMemory) {
  const auto         bytes = pairCacheBytes<TileSize>(numWords);
  cudaFuncAttributes pairAttributes{};
  cudaFuncAttributes repairAttributes{};
  cudaCheckError(cudaFuncGetAttributes(&pairAttributes, splitPairsKernel<TileSize>));
  cudaCheckError(cudaFuncGetAttributes(&repairAttributes, repairKernel<TileSize>));
  // Kernel attributes are shared by concurrent calls, so their limits must not depend on the input width.
  const auto maxCache    = pairCacheBytes<TileSize>(kBitBirchMaxWords);
  const auto pairLimit   = std::min(maxCache, deviceSharedMemory - pairAttributes.sharedSizeBytes);
  const auto repairLimit = std::min(maxCache, deviceSharedMemory - repairAttributes.sharedSizeBytes);
  cudaCheckError(cudaFuncSetAttribute(splitPairsKernel<TileSize>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(pairLimit)));
  cudaCheckError(cudaFuncSetAttribute(repairKernel<TileSize>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(repairLimit)));
  return bytes;
}

class TreeBuilder {
 public:
  TreeBuilder(const std::uint32_t*   fingerprints,
              const int              numRows,
              const int              numWords,
              const double           threshold,
              const BitBirchOptions& options,
              const int              sharedMemoryBudget,
              const cudaStream_t     stream)
      : numRows_(numRows),
        numWords_(numWords),
        threshold_(threshold),
        options_(options),
        stream_(stream),
        batchCapacity_(std::min(options.batchSize, numRows)),
        stride_(options.branchingFactor + 1),
        // Balanced splits leave every non-root node at least half full.
        minNodeSize_((options.branchingFactor + 1) / 2) {
    for (auto* vector : {&nodeSize_,     &nodeParent_, &nodeParentEntry_, &nodeEntries_,
                         &entryChild_,   &entrySlot_,  &entryRow_,        &entryFirst_,
                         &labels_,       &keys_,       &sortedKeys_,      &values_,
                         &sortedValues_, &groupKeys_,  &sortedGroupKeys_, &sortedGroupValues_,
                         &dirty_,        &levelNodes_, &levelCounts_,     &deltaSlots_,
                         &deltaCursor_,  &splitQueue_, &splitCount_,      &plans_,
                         &leftSizes_,    &innerPlan_,  &deferredNodes_,   &deferredEntries_,
                         &numDeferred_,  &cursors_,    &control_}) {
      vector->setStream(stream);
    }
    for (auto* vector : {&entryCount_, &sums_, &centroids_, &deltaSums_, &deltaCounts_, &ownerCounts_}) {
      vector->setStream(stream);
    }
    nodeLeaf_.setStream(stream);
    sides_.setStream(stream);
    pairKeys_.setStream(stream);
    sortScratch_.setStream(stream);
    tree_.fingerprints    = fingerprints;
    tree_.numWords        = numWords;
    tree_.numBits         = numWords * 32;
    tree_.branchingFactor = options.branchingFactor;
    tree_.stride          = stride_;

    labels_.resize(numRows);
    cudaCheckError(cudaMemsetAsync(labels_.data(), 0xff, sizeof(int) * numRows, stream));
    for (auto* vector : {&keys_,
                         &sortedKeys_,
                         &values_,
                         &sortedValues_,
                         &groupKeys_,
                         &sortedGroupKeys_,
                         &sortedGroupValues_,
                         &splitQueue_,
                         &leftSizes_}) {
      vector->resize(batchCapacity_);
    }
    deferredNodes_.resize(2 * static_cast<std::size_t>(batchCapacity_));
    deferredEntries_.resize(2 * static_cast<std::size_t>(batchCapacity_));
    plans_.resize(static_cast<std::size_t>(batchCapacity_) * stride_);
    sides_.resize(static_cast<std::size_t>(batchCapacity_) * stride_);
    innerPlan_.resize(stride_);
    // Leaf owners batch as many rows as the count scratch budget allows, at least one.
    const std::size_t runBytes = sizeof(std::uint32_t) * static_cast<std::size_t>(batchCapacity_) * stride_;
    ownerWidth_ = static_cast<int>(std::clamp<std::size_t>(kOwnerScratchBytes / runBytes, 1, kOwnerBatch));
    ownerCounts_.resize(static_cast<std::size_t>(ownerWidth_) * batchCapacity_ * stride_);
    pairKeys_.resize(batchCapacity_);
    cudaCheckError(cudaMemsetAsync(pairKeys_.data(), 0xff, sizeof(unsigned long long) * batchCapacity_, stream));
    for (auto* vector : {&deltaCursor_, &splitCount_, &numDeferred_}) {
      vector->resize(1);
      cudaCheckError(cudaMemsetAsync(vector->data(), 0, sizeof(int), stream));
    }
    // root, nodeCursor, entryCursor, slotCursor, status.
    cursors_.resize(5);
    control_.resize(kNumControls);
    hostControl_ = PinnedHostVector<int>(kNumControls, 0);

    int device = 0;
    cudaCheckError(cudaGetDevice(&device));
    cudaCheckError(cudaDeviceGetAttribute(&numSMs_, cudaDevAttrMultiProcessorCount, device));
    int deviceSharedMemory = 0;
    cudaCheckError(cudaDeviceGetAttribute(&deviceSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    const int budget = std::min(sharedMemoryBudget, deviceSharedMemory);
    splitTileSize_   = bitbirch::selectSplitTileSize(numWords, budget);
    pairShared_      = splitTileSize_ == 64 ? configureSplitCache<64>(numWords, deviceSharedMemory) :
                                              configureSplitCache<32>(numWords, deviceSharedMemory);

    std::size_t sortBytes = 0;
    cudaCheckError(cub::DeviceRadixSort::SortPairs(nullptr,
                                                   sortBytes,
                                                   keys_.data(),
                                                   sortedKeys_.data(),
                                                   values_.data(),
                                                   sortedValues_.data(),
                                                   batchCapacity_,
                                                   0,
                                                   32,
                                                   stream));
    sortScratch_.resize(sortBytes);

    grow(batchCapacity_);
    bindTree();
    initializeKernel<<<1, 1, 0, stream>>>(tree_);
    cudaCheckError(cudaGetLastError());
    hostControl_[kNodes] = 1;
  }

  BitBirchResult run() {
    for (int begin = 0; begin < numRows_;) {
      // Warm-up batches end at kOrderedWarmup so the first batch with groups starts there.
      const int boundary = begin < kOrderedWarmup ? std::min(kOrderedWarmup, numRows_) : numRows_;
      const int count    = std::min(batchCapacity_, boundary - begin);
      insertBatch(begin, count);
      begin += count;
    }
    return finalize();
  }

 private:
  int clusters() const { return hostControl_[kEntries] - hostControl_[kNodes] + 1; }

  // Grows the tree for up to clusterBound clusters and one more round's summaries.
  void grow(const std::int64_t clusterBound) {
    const std::int64_t nodes   = std::min<std::int64_t>(clusterBound / std::max(1, minNodeSize_ - 1) + 8,
                                                      numRows_ / std::max(1, minNodeSize_ - 1) + 8);
    const std::int64_t entries = clusterBound + nodes;
    const std::int64_t slots =
      hostControl_[kSlots] + (clusterBound - clusters()) * (2LL * hostControl_[kDepth] + 5) + 4;
    if (entries > std::numeric_limits<int>::max() || slots > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("BitBIRCH tree exceeds its int32 index capacity");
    }
    const auto oldNodes = dirty_.size();
    for (auto* vector : {&nodeSize_, &nodeParent_, &nodeParentEntry_, &dirty_, &deltaSlots_}) {
      growTo(*vector, nodes);
    }
    growTo(nodeLeaf_, nodes);
    growTo(nodeEntries_, static_cast<std::size_t>(nodeSize_.size()) * stride_);
    if (dirty_.size() > oldNodes) {
      cudaCheckError(cudaMemsetAsync(dirty_.data() + oldNodes, 0, sizeof(int) * (dirty_.size() - oldNodes), stream_));
      cudaCheckError(
        cudaMemsetAsync(deltaSlots_.data() + oldNodes, 0xff, sizeof(int) * (deltaSlots_.size() - oldNodes), stream_));
    }
    for (auto* vector : {&entryChild_, &entrySlot_, &entryRow_, &entryFirst_}) {
      growTo(*vector, entries);
    }
    growTo(entryCount_, entries);
    growTo(sums_, static_cast<std::size_t>(slots) * numWords_ * 32);
    growTo(centroids_, static_cast<std::size_t>(slots) * numWords_);
    bindTree();
    tree_.maxNodes   = static_cast<int>(nodeSize_.size());
    tree_.maxEntries = static_cast<int>(entryChild_.size());
    tree_.maxSlots   = static_cast<int>(centroids_.size() / numWords_);
  }

  void bindTree() {
    tree_.labels          = labels_.data();
    tree_.nodeSize        = nodeSize_.data();
    tree_.nodeParent      = nodeParent_.data();
    tree_.nodeParentEntry = nodeParentEntry_.data();
    tree_.nodeLeaf        = nodeLeaf_.data();
    tree_.nodeEntries     = nodeEntries_.data();
    tree_.entryChild      = entryChild_.data();
    tree_.entryCount      = entryCount_.data();
    tree_.entrySlot       = entrySlot_.data();
    tree_.entryRow        = entryRow_.data();
    tree_.entryFirst      = entryFirst_.data();
    tree_.sums            = sums_.data();
    tree_.centroids       = centroids_.data();
    tree_.root            = cursors_.data();
    tree_.nodeCursor      = cursors_.data() + 1;
    tree_.entryCursor     = cursors_.data() + 2;
    tree_.slotCursor      = cursors_.data() + 3;
    tree_.status          = cursors_.data() + 4;
  }

  void sortPairs(const int* keys,
                 int*       sortedKeys,
                 const int* values,
                 int*       sortedValues,
                 const int  count,
                 const int  bits) {
    std::size_t bytes = sortScratch_.size();
    cudaCheckError(cub::DeviceRadixSort::SortPairs(sortScratch_.data(),
                                                   bytes,
                                                   keys,
                                                   sortedKeys,
                                                   values,
                                                   sortedValues,
                                                   count,
                                                   0,
                                                   bits,
                                                   stream_));
  }

  template <int TileSize> void splitOverflowingNodes(const int count) {
    splitPairsKernel<TileSize><<<2 * numSMs_, kBlockSize, pairShared_, stream_>>>(tree_,
                                                                                  sortedKeys_.data(),
                                                                                  splitQueue_.data(),
                                                                                  splitCount_.data(),
                                                                                  pairKeys_.data());
    cudaCheckError(cudaGetLastError());
    splitPlansKernel<<<count, kBlockSize, 0, stream_>>>(tree_,
                                                        sortedKeys_.data(),
                                                        splitQueue_.data(),
                                                        splitCount_.data(),
                                                        pairKeys_.data(),
                                                        plans_.data(),
                                                        leftSizes_.data(),
                                                        sides_.data());
    cudaCheckError(cudaGetLastError());
    repairKernel<TileSize><<<1, kBlockSize, pairShared_, stream_>>>(tree_,
                                                                    count,
                                                                    sortedKeys_.data(),
                                                                    plans_.data(),
                                                                    leftSizes_.data(),
                                                                    innerPlan_.data(),
                                                                    sides_.data(),
                                                                    deferredNodes_.data(),
                                                                    deferredEntries_.data(),
                                                                    numDeferred_.data());
    cudaCheckError(cudaGetLastError());
  }

  void insertBatch(const int begin, const int count) {
    const ScopedNvtxRange batchRange("BitBIRCH batch");
    const bool            groups = begin >= kOrderedWarmup;
    tree_.batchBegin             = begin;
    for (int pending = count; pending > 0;) {
      const int depth = hostControl_[kDepth];
      grow(static_cast<std::int64_t>(clusters()) + pending);
      // Each level lists at most one node per routed leaf; each node takes one delta per round.
      reserveScratch(levelNodes_, static_cast<std::size_t>(depth + 1) * batchCapacity_);
      if (levelCounts_.size() < static_cast<std::size_t>(depth) + 1) {
        levelCounts_.resize(0);
        levelCounts_.resize(depth + 1);
        cudaCheckError(cudaMemsetAsync(levelCounts_.data(), 0, sizeof(int) * (depth + 1), stream_));
      }
      const auto deltaCapacity = static_cast<std::size_t>(std::max(1, pending * depth));
      reserveScratch(deltaCounts_, deltaCapacity);
      reserveScratch(deltaSums_, deltaCapacity * numWords_ * 32);

      routeKernel<<<count, kBlockSize, 0, stream_>>>(tree_,
                                                     groups,
                                                     threshold_,
                                                     keys_.data(),
                                                     groupKeys_.data(),
                                                     values_.data());
      cudaCheckError(cudaGetLastError());
      if (groups) {
        sortPairs(groupKeys_.data(),
                  sortedGroupKeys_.data(),
                  values_.data(),
                  sortedGroupValues_.data(),
                  count,
                  keyBits(tree_.maxEntries));
        groupsKernel<<<count, kBlockSize, 0, stream_>>>(tree_,
                                                        count,
                                                        sortedGroupKeys_.data(),
                                                        sortedGroupValues_.data(),
                                                        threshold_);
        cudaCheckError(cudaGetLastError());
      }
      sortPairs(keys_.data(), sortedKeys_.data(), values_.data(), sortedValues_.data(), count, keyBits(tree_.maxNodes));
      leafOwnersKernel<<<count, kBlockSize, 0, stream_>>>(tree_,
                                                          count,
                                                          sortedKeys_.data(),
                                                          sortedValues_.data(),
                                                          threshold_,
                                                          depth,
                                                          dirty_.data(),
                                                          levelNodes_.data(),
                                                          levelCounts_.data(),
                                                          batchCapacity_,
                                                          splitQueue_.data(),
                                                          splitCount_.data(),
                                                          ownerCounts_.data(),
                                                          ownerWidth_);
      cudaCheckError(cudaGetLastError());
      for (int level = depth; level > 0; --level) {
        refreshKernel<<<count, kBlockSize, 0, stream_>>>(
          tree_,
          levelNodes_.data() + static_cast<std::size_t>(level) * batchCapacity_,
          levelCounts_.data() + level,
          count,
          sortedKeys_.data(),
          sortedValues_.data(),
          dirty_.data(),
          deltaSlots_.data(),
          deltaCursor_.data(),
          static_cast<int>(deltaCapacity),
          deltaSums_.data(),
          deltaCounts_.data());
        cudaCheckError(cudaGetLastError());
      }
      if (splitTileSize_ == 64) {
        splitOverflowingNodes<64>(count);
      } else {
        splitOverflowingNodes<32>(count);
      }
      summarizeKernel<<<2 * count, kBlockSize, 0, stream_>>>(tree_,
                                                             deferredNodes_.data(),
                                                             deferredEntries_.data(),
                                                             numDeferred_.data());
      cudaCheckError(cudaGetLastError());
      controlKernel<<<1, kBlockSize, 0, stream_>>>(tree_,
                                                   count,
                                                   control_.data(),
                                                   levelNodes_.data(),
                                                   levelCounts_.data(),
                                                   static_cast<int>(levelCounts_.size()),
                                                   batchCapacity_,
                                                   deltaSlots_.data(),
                                                   deltaCursor_.data(),
                                                   splitCount_.data());
      cudaCheckError(cudaGetLastError());
      cudaCheckError(cudaMemcpyAsync(hostControl_.data(),
                                     control_.data(),
                                     sizeof(int) * kNumControls,
                                     cudaMemcpyDeviceToHost,
                                     stream_));
      cudaCheckError(cudaStreamSynchronize(stream_));
      if (hostControl_[kStatus] != kSuccess) {
        throw std::runtime_error("BitBIRCH tree failure (status " + std::to_string(hostControl_[kStatus]) + ")");
      }
      if (hostControl_[kPending] >= pending) {
        throw std::runtime_error("BitBIRCH insertion failed to make progress");
      }
      pending = hostControl_[kPending];
    }
  }

  BitBirchResult finalize() {
    const ScopedNvtxRange  range("BitBIRCH finalize");
    const int              numClusters = clusters();
    const int              numEntries  = hostControl_[kEntries];
    AsyncDeviceVector<int> firstRows(numClusters, stream_);
    AsyncDeviceVector<int> leafEntries(numClusters, stream_);
    AsyncDeviceVector<int> sortedFirstRows(numClusters, stream_);
    AsyncDeviceVector<int> sortedLeafEntries(numClusters, stream_);
    AsyncDeviceVector<int> cursor(1, stream_);
    cudaCheckError(cudaMemsetAsync(cursor.data(), 0, sizeof(int), stream_));
    collectLeavesKernel<<<(numEntries + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream_>>>(tree_,
                                                                                                numEntries,
                                                                                                firstRows.data(),
                                                                                                leafEntries.data(),
                                                                                                cursor.data());
    cudaCheckError(cudaGetLastError());
    // First rows are distinct, so sorting by them numbers clusters by earliest member.
    std::size_t bytes = 0;
    cudaCheckError(cub::DeviceRadixSort::SortPairs(nullptr,
                                                   bytes,
                                                   firstRows.data(),
                                                   sortedFirstRows.data(),
                                                   leafEntries.data(),
                                                   sortedLeafEntries.data(),
                                                   numClusters,
                                                   0,
                                                   keyBits(numRows_ - 1),
                                                   stream_));
    AsyncDeviceVector<std::byte> scratch(bytes, stream_);
    cudaCheckError(cub::DeviceRadixSort::SortPairs(scratch.data(),
                                                   bytes,
                                                   firstRows.data(),
                                                   sortedFirstRows.data(),
                                                   leafEntries.data(),
                                                   sortedLeafEntries.data(),
                                                   numClusters,
                                                   0,
                                                   keyBits(numRows_ - 1),
                                                   stream_));
    BitBirchResult result{std::move(labels_), AsyncDeviceVector<std::uint32_t>(0, stream_), numClusters, numWords_};
    if (options_.returnCentroids) {
      result.centroids.resize(static_cast<std::size_t>(numClusters) * numWords_);
    }
    const int         width = options_.returnCentroids ? numWords_ : 1;
    const std::size_t items = static_cast<std::size_t>(numClusters) * width;
    numberClustersKernel<<<static_cast<unsigned>((items + kBlockSize - 1) / kBlockSize), kBlockSize, 0, stream_>>>(
      tree_,
      numClusters,
      width,
      sortedLeafEntries.data(),
      options_.returnCentroids ? result.centroids.data() : nullptr);
    cudaCheckError(cudaGetLastError());
    labelRowsKernel<<<(numRows_ + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream_>>>(tree_, numRows_);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaStreamSynchronize(stream_));
    return result;
  }

  const int                        numRows_;
  const int                        numWords_;
  const double                     threshold_;
  const BitBirchOptions            options_;
  const cudaStream_t               stream_;
  const int                        batchCapacity_;
  const int                        stride_;
  const int                        minNodeSize_;
  int                              numSMs_        = 0;
  int                              ownerWidth_    = 1;
  int                              splitTileSize_ = 64;
  std::size_t                      pairShared_    = 0;
  Tree                             tree_{};
  AsyncDeviceVector<int>           nodeSize_, nodeParent_, nodeParentEntry_, nodeEntries_;
  AsyncDeviceVector<std::uint8_t>  nodeLeaf_;
  AsyncDeviceVector<int>           entryChild_, entrySlot_, entryRow_, entryFirst_;
  AsyncDeviceVector<std::uint32_t> entryCount_, sums_, centroids_, ownerCounts_;
  AsyncDeviceVector<int>           labels_;
  AsyncDeviceVector<int> keys_, sortedKeys_, values_, sortedValues_, groupKeys_, sortedGroupKeys_, sortedGroupValues_;
  AsyncDeviceVector<int> dirty_, levelNodes_, levelCounts_, deltaSlots_, deltaCursor_;
  AsyncDeviceVector<std::uint32_t>      deltaSums_, deltaCounts_;
  AsyncDeviceVector<int>                splitQueue_, splitCount_, plans_, leftSizes_, innerPlan_;
  AsyncDeviceVector<std::int8_t>        sides_;
  AsyncDeviceVector<unsigned long long> pairKeys_;
  AsyncDeviceVector<int>                deferredNodes_, deferredEntries_, numDeferred_, cursors_, control_;
  AsyncDeviceVector<std::byte>          sortScratch_;
  PinnedHostVector<int>                 hostControl_;
};

}  // namespace

namespace bitbirch {

int selectSplitTileSize(const int numWords, const int sharedMemoryBudget) {
  if (splitCacheFits<64>(numWords, sharedMemoryBudget)) {
    return 64;
  }
  if (splitCacheFits<32>(numWords, sharedMemoryBudget)) {
    return 32;
  }
  throw std::runtime_error("BitBIRCH split cache exceeds the per-block shared memory budget");
}

BitBirchResult clusterWithSharedMemoryBudget(const cuda::std::span<const std::uint32_t> fingerprints,
                                             const int                                  numFingerprints,
                                             const int                                  numWords,
                                             const double                               threshold,
                                             const BitBirchOptions&                     options,
                                             const int                                  sharedMemoryBudget,
                                             const cudaStream_t                         stream) {
  return TreeBuilder(fingerprints.data(), numFingerprints, numWords, threshold, options, sharedMemoryBudget, stream)
    .run();
}

}  // namespace bitbirch

BitBirchResult bitBirchGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                           const int                                  numFingerprints,
                           const int                                  numWords,
                           const double                               threshold,
                           const BitBirchOptions&                     options,
                           const cudaStream_t                         stream) {
  const ScopedNvtxRange range("BitBIRCH");
  if (numFingerprints < 0 || numWords <= 0 || numWords > kBitBirchMaxWords ||
      fingerprints.size() != static_cast<std::size_t>(numFingerprints) * numWords) {
    throw std::invalid_argument("BitBIRCH fingerprints must be a matrix of 1 to 128 words per row");
  }
  if (!std::isfinite(threshold) || threshold < 0 || threshold > 1 || options.branchingFactor < 3 ||
      options.branchingFactor > kMaxBranchingFactor || options.batchSize < 1) {
    throw std::invalid_argument("BitBIRCH clustering options are invalid");
  }
  if (numFingerprints == 0) {
    return BitBirchResult{AsyncDeviceVector<int>(0, stream), AsyncDeviceVector<std::uint32_t>(0, stream), 0, numWords};
  }
  return bitbirch::clusterWithSharedMemoryBudget(fingerprints,
                                                 numFingerprints,
                                                 numWords,
                                                 threshold,
                                                 options,
                                                 std::numeric_limits<int>::max(),
                                                 stream);
}

}  // namespace nvMolKit
