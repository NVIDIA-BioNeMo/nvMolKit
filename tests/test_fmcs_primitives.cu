// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the fMCS seed / queue / cooperative-copy primitives.
// Each test launches a tiny __global__ driver that constructs inputs,
// calls one helper, and copies results back to host memory for assertion.

#include <cooperative_groups.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

#include "src/mcs/fmcs_cuda/fmcs_seed.cuh"
#include "src/mcs/fmcs_cuda/fmcs_seed_queue.cuh"
#include "src/mcs/mcs_common/mcs_cooperative_copy.cuh"
#include "src/utils/device_vector.h"

namespace {

namespace cg = cooperative_groups;

using nvMolKit::AsyncDevicePtr;
using nvMolKit::AsyncDeviceVector;

using mcs::fmcs::MatchResult;
using mcs::fmcs::QueuedSeed;
using mcs::fmcs::Seed;
using mcs::fmcs::SeedQueue;
using mcs::fmcs::ThreadBlockScope;

struct QueueResult {
  int  first;
  int  second;
  bool empty;
  bool overflowAccepted;
};

__global__ void seedMutationKernel(Seed<128, 128>* seed) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  *seed = {};
  mcs::fmcs::seedAddAtomWithinThread(*seed, 3);
  mcs::fmcs::seedAddAtomWithinThread(*seed, 67);
  mcs::fmcs::seedAddBondWithinThread(*seed, 7);
  mcs::fmcs::seedAddBondWithinThread(*seed, 71);
}

__global__ void queueKernel(int* storage, QueueResult* result) {
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  if (threadIdx.x == 0) {
    queue.init(storage, 2);
    queue.pushWithinThread(11);
    queue.pushWithinThread(22);
    result->overflowAccepted = queue.pushWithinThread(33);
    queue.popWithinThread(result->first);
    queue.popWithinThread(result->second);
    int ignored{};
    result->empty = !queue.popWithinThread(ignored);
  }
}

}  // namespace

TEST(FMCSPrimitives, SeedMutationCrossesBitsetWords) {
  AsyncDevicePtr<Seed<128, 128>> d_seed;

  seedMutationKernel<<<1, 1>>>(d_seed.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  Seed<128, 128> seed{};
  d_seed.get(seed);
  EXPECT_EQ(seed.numAtoms, 2);
  EXPECT_EQ(seed.atoms[0], 1ULL << 3);
  EXPECT_EQ(seed.atoms[1], 1ULL << 3);
  EXPECT_EQ(seed.numBonds, 2);
  EXPECT_EQ(seed.bonds[0], 1ULL << 7);
  EXPECT_EQ(seed.bonds[1], 1ULL << 7);
  EXPECT_EQ(seed.excludedBonds[0], seed.bonds[0]);
  EXPECT_EQ(seed.excludedBonds[1], seed.bonds[1]);
}

TEST(FMCSPrimitives, QueueIsBoundedLifo) {
  AsyncDeviceVector<int>      d_storage(2);
  AsyncDevicePtr<QueueResult> d_result;

  queueKernel<<<1, 1>>>(d_storage.data(), d_result.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  QueueResult result{};
  d_result.get(result);
  EXPECT_FALSE(result.overflowAccepted);
  EXPECT_EQ(result.first, 22);
  EXPECT_EQ(result.second, 11);
  EXPECT_TRUE(result.empty);
}

// ---------------------------------------------------------------------------
// matchResultClearWithinThread
// ---------------------------------------------------------------------------

namespace {

__global__ void matchResultClearDriverKernel(MatchResult<16, 16, 16, 16>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  MatchResult<16, 16, 16, 16> m;
  for (int i = 0; i < 16; ++i)
    m.targetAtomIdx[i] = 7;
  for (int i = 0; i < 16; ++i)
    m.targetBondIdx[i] = 7;
  // Pre-populate visited bitsets too so we can verify they're zeroed.
  m.visitedTargetAtoms[0] = 0xDEADBEEFu;
  m.visitedTargetBonds[0] = 0xCAFEBABEu;
  m.matchedAtomSize       = 13;
  m.matchedBondSize       = 11;
  m.empty                 = false;
  mcs::fmcs::matchResultClearWithinThread(m);
  *out = m;
}

}  // namespace

TEST(FMCSPrimitives, MatchResultClearZeroesEverything) {
  AsyncDevicePtr<MatchResult<16, 16, 16, 16>> d_out;

  matchResultClearDriverKernel<<<1, 1>>>(d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  MatchResult<16, 16, 16, 16> out{};
  d_out.get(out);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(out.targetAtomIdx[i], 0xFFu);
    EXPECT_EQ(out.targetBondIdx[i], 0xFFu);
  }
  EXPECT_EQ(out.visitedTargetAtoms[0], 0u);
  EXPECT_EQ(out.visitedTargetBonds[0], 0u);
  EXPECT_TRUE(out.empty);
  EXPECT_EQ(out.matchedAtomSize, 0);
  EXPECT_EQ(out.matchedBondSize, 0);
}

// ---------------------------------------------------------------------------
// seedAddAtomWithinThread
// ---------------------------------------------------------------------------

namespace {

__global__ void seedAddAtomDriverKernel(Seed<16, 16>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  Seed<16, 16> seed{};
  mcs::fmcs::seedAddAtomWithinThread(seed, 3);
  mcs::fmcs::seedAddAtomWithinThread(seed, 5);
  mcs::fmcs::seedAddAtomWithinThread(seed, 12);
  *out = seed;
}

// A single grow step can reach the same new atom through two boundary
// bonds (a ring closed within one step); the double-add must be a no-op
// so the invariant numAtoms == popcount(atoms) holds.
__global__ void seedDoubleAddAtomDriverKernel(Seed<16, 16>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  Seed<16, 16> seed{};
  mcs::fmcs::seedAddAtomWithinThread(seed, 4);
  mcs::fmcs::seedAddAtomWithinThread(seed, 4);
  mcs::fmcs::seedAddAtomWithinThread(seed, 9);
  mcs::fmcs::seedAddAtomWithinThread(seed, 9);
  *out = seed;
}

}  // namespace

TEST(FMCSPrimitives, SeedAddAtomSetsBitsAndCount) {
  AsyncDevicePtr<Seed<16, 16>> d_out;

  seedAddAtomDriverKernel<<<1, 1>>>(d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  Seed<16, 16> out{};
  d_out.get(out);
  // Tier 16 packs the atom bitset into one uint32 word.
  const std::uint32_t expected = (1u << 3) | (1u << 5) | (1u << 12);
  EXPECT_EQ(out.atoms[0], expected);
  EXPECT_EQ(out.numAtoms, 3);
  // No bond / excludedBonds touched.
  EXPECT_EQ(out.bonds[0], 0u);
  EXPECT_EQ(out.excludedBonds[0], 0u);
  EXPECT_EQ(out.numBonds, 0);
}

TEST(FMCSPrimitives, SeedAddAtomDoubleAddIsIdempotent) {
  AsyncDevicePtr<Seed<16, 16>> d_out;

  seedDoubleAddAtomDriverKernel<<<1, 1>>>(d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  Seed<16, 16> out{};
  d_out.get(out);
  const std::uint32_t expected = (1u << 4) | (1u << 9);
  EXPECT_EQ(out.atoms[0], expected);
  EXPECT_EQ(out.lastAddedAtoms[0], expected);
  EXPECT_EQ(out.numAtoms, 2);  // Not 4: re-adds must not double-count.
}

// ---------------------------------------------------------------------------
// seedAddBondWithinThread / seedExcludeBondWithinThread
// ---------------------------------------------------------------------------

namespace {

__global__ void seedAddBondDriverKernel(Seed<16, 16>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  Seed<16, 16> seed{};
  mcs::fmcs::seedAddBondWithinThread(seed, 2);
  mcs::fmcs::seedAddBondWithinThread(seed, 7);
  *out = seed;
}

__global__ void seedExcludeBondDriverKernel(Seed<16, 16>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  Seed<16, 16> seed{};
  mcs::fmcs::seedExcludeBondWithinThread(seed, 5);
  mcs::fmcs::seedExcludeBondWithinThread(seed, 13);
  *out = seed;
}

}  // namespace

TEST(FMCSPrimitives, SeedAddBondSetsBondsAndExcludedAndCount) {
  AsyncDevicePtr<Seed<16, 16>> d_out;

  seedAddBondDriverKernel<<<1, 1>>>(d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  Seed<16, 16> out{};
  d_out.get(out);
  const std::uint32_t expected = (1u << 2) | (1u << 7);
  EXPECT_EQ(out.bonds[0], expected);
  EXPECT_EQ(out.excludedBonds[0], expected);
  EXPECT_EQ(out.numBonds, 2);
  // Atom side untouched.
  EXPECT_EQ(out.atoms[0], 0u);
  EXPECT_EQ(out.numAtoms, 0);
}

TEST(FMCSPrimitives, SeedExcludeBondDoesNotAddToSeed) {
  AsyncDevicePtr<Seed<16, 16>> d_out;

  seedExcludeBondDriverKernel<<<1, 1>>>(d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  Seed<16, 16> out{};
  d_out.get(out);
  // Excluded bonds are barred from future grows without joining the seed.
  const std::uint32_t expected = (1u << 5) | (1u << 13);
  EXPECT_EQ(out.excludedBonds[0], expected);
  EXPECT_EQ(out.bonds[0], 0u);
  EXPECT_EQ(out.numBonds, 0);
}

// ---------------------------------------------------------------------------
// seedBeginGrowStepWithinThread
// ---------------------------------------------------------------------------

namespace {

__global__ void seedBeginGrowStepDriverKernel(Seed<16, 16>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  Seed<16, 16> seed{};
  mcs::fmcs::seedAddAtomWithinThread(seed, 0);
  mcs::fmcs::seedAddAtomWithinThread(seed, 1);
  mcs::fmcs::seedAddAtomWithinThread(seed, 2);
  // Boundary bitset should reset here.
  mcs::fmcs::seedBeginGrowStepWithinThread(seed);
  // Subsequent adds advance numAtoms and repopulate the boundary.
  mcs::fmcs::seedAddAtomWithinThread(seed, 5);
  mcs::fmcs::seedAddAtomWithinThread(seed, 9);
  *out = seed;
}

}  // namespace

TEST(FMCSPrimitives, SeedBeginGrowStepResetsBoundary) {
  AsyncDevicePtr<Seed<16, 16>> d_out;

  seedBeginGrowStepDriverKernel<<<1, 1>>>(d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  Seed<16, 16> out{};
  d_out.get(out);
  EXPECT_EQ(out.numAtoms, 5);
  EXPECT_EQ(out.atoms[0], (1u << 0) | (1u << 1) | (1u << 2) | (1u << 5) | (1u << 9));
  // Only this step's atoms remain in the boundary set.
  EXPECT_EQ(out.lastAddedAtoms[0], (1u << 5) | (1u << 9));
}

// ---------------------------------------------------------------------------
// seedCanGrowBiggerThanWithinThread
// ---------------------------------------------------------------------------

namespace {

struct CanGrowResults {
  bool moreBonds;           // possible bonds 8 vs best 5: true
  bool fewerBonds;          // possible bonds 5 vs best 8: false
  bool tiedBondsMoreAtoms;  // possible bonds 8 best 8, atoms tie-break: true
  bool tiedBondsFewerAtoms;
  bool tiedBondsTiedAtoms;
};

__global__ void seedCanGrowDriverKernel(CanGrowResults* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  Seed<16, 16> seed{};
  seed.numBonds       = 5;
  seed.remainingBonds = 3;  // possibleBonds = 8
  seed.numAtoms       = 4;
  seed.remainingAtoms = 2;  // possibleAtoms = 6

  out->moreBonds           = mcs::fmcs::seedCanGrowBiggerThanWithinThread(seed, /*bestBonds=*/5, /*bestAtoms=*/0);
  out->fewerBonds          = mcs::fmcs::seedCanGrowBiggerThanWithinThread(seed, /*bestBonds=*/9, /*bestAtoms=*/0);
  out->tiedBondsMoreAtoms  = mcs::fmcs::seedCanGrowBiggerThanWithinThread(seed, /*bestBonds=*/8, /*bestAtoms=*/5);
  out->tiedBondsFewerAtoms = mcs::fmcs::seedCanGrowBiggerThanWithinThread(seed, /*bestBonds=*/8, /*bestAtoms=*/7);
  out->tiedBondsTiedAtoms  = mcs::fmcs::seedCanGrowBiggerThanWithinThread(seed, /*bestBonds=*/8, /*bestAtoms=*/6);
}

struct CanGrowFromEmptyResults {
  bool nonEmptyVsZero;  // possibleBonds=2 vs (0,0): true
  bool emptyVsZero;     // possibleBonds=0 vs (0,0): false (strict >)
};

__global__ void seedCanGrowFromEmptyDriverKernel(CanGrowFromEmptyResults* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  Seed<16, 16> live{};
  live.numBonds       = 1;
  live.remainingBonds = 1;
  out->nonEmptyVsZero = mcs::fmcs::seedCanGrowBiggerThanWithinThread(live, 0, 0);
  Seed<16, 16> empty{};
  out->emptyVsZero = mcs::fmcs::seedCanGrowBiggerThanWithinThread(empty, 0, 0);
}

}  // namespace

TEST(FMCSPrimitives, SeedCanGrowBiggerThanBoundCheck) {
  AsyncDevicePtr<CanGrowResults> d_out;

  seedCanGrowDriverKernel<<<1, 1>>>(d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  CanGrowResults out{};
  d_out.get(out);
  EXPECT_TRUE(out.moreBonds);
  EXPECT_FALSE(out.fewerBonds);
  EXPECT_TRUE(out.tiedBondsMoreAtoms);
  EXPECT_FALSE(out.tiedBondsFewerAtoms);
  EXPECT_FALSE(out.tiedBondsTiedAtoms);  // strictly bigger required
}

TEST(FMCSPrimitives, SeedCanGrowBiggerThanFromEmptyIncumbent) {
  AsyncDevicePtr<CanGrowFromEmptyResults> d_out;

  seedCanGrowFromEmptyDriverKernel<<<1, 1>>>(d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  CanGrowFromEmptyResults out{};
  d_out.get(out);
  EXPECT_TRUE(out.nonEmptyVsZero);
  EXPECT_FALSE(out.emptyVsZero);
}

// ---------------------------------------------------------------------------
// Multi-word and boundary edge cases
// ---------------------------------------------------------------------------

namespace {

__global__ void seedAddAtomMultiWordDriverKernel(Seed<128, 128>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  Seed<128, 128> seed{};
  // Atoms at the four corners of the 2-uint64-word atom bitset:
  //   0 (word 0, bit 0), 63 (word 0, bit 63),
  //  64 (word 1, bit 0), 127 (word 1, bit 63).
  mcs::fmcs::seedAddAtomWithinThread(seed, 0);
  mcs::fmcs::seedAddAtomWithinThread(seed, 63);
  mcs::fmcs::seedAddAtomWithinThread(seed, 64);
  mcs::fmcs::seedAddAtomWithinThread(seed, 127);
  *out = seed;
}

__global__ void seedAddBondMultiWordDriverKernel(Seed<128, 128>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  Seed<128, 128> seed{};
  mcs::fmcs::seedAddBondWithinThread(seed, 0);
  mcs::fmcs::seedAddBondWithinThread(seed, 63);
  mcs::fmcs::seedAddBondWithinThread(seed, 64);
  mcs::fmcs::seedAddBondWithinThread(seed, 127);
  *out = seed;
}

__global__ void seedAddAtomBoundaryDriverKernel(Seed<16, 16>* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  Seed<16, 16> seed{};
  mcs::fmcs::seedAddAtomWithinThread(seed, 0);
  mcs::fmcs::seedAddAtomWithinThread(seed, 15);  // tier-16 max valid atom idx
  *out = seed;
}

}  // namespace

TEST(FMCSPrimitives, SeedAddAtomCrossesWordBoundary) {
  AsyncDevicePtr<Seed<128, 128>> d_out;

  seedAddAtomMultiWordDriverKernel<<<1, 1>>>(d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  Seed<128, 128> out{};
  d_out.get(out);
  const std::uint64_t bit0  = std::uint64_t{1} << 0;
  const std::uint64_t bit63 = std::uint64_t{1} << 63;
  EXPECT_EQ(out.atoms[0], bit0 | bit63);
  EXPECT_EQ(out.atoms[1], bit0 | bit63);
  EXPECT_EQ(out.numAtoms, 4);
}

TEST(FMCSPrimitives, SeedAddBondCrossesWordBoundary) {
  AsyncDevicePtr<Seed<128, 128>> d_out;

  seedAddBondMultiWordDriverKernel<<<1, 1>>>(d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  Seed<128, 128> out{};
  d_out.get(out);
  const std::uint64_t bit0     = std::uint64_t{1} << 0;
  const std::uint64_t bit63    = std::uint64_t{1} << 63;
  const std::uint64_t expected = bit0 | bit63;
  EXPECT_EQ(out.bonds[0], expected);
  EXPECT_EQ(out.bonds[1], expected);
  EXPECT_EQ(out.excludedBonds[0], expected);
  EXPECT_EQ(out.excludedBonds[1], expected);
  EXPECT_EQ(out.numBonds, 4);
}

TEST(FMCSPrimitives, SeedAddAtomBoundaryIndices) {
  AsyncDevicePtr<Seed<16, 16>> d_out;

  seedAddAtomBoundaryDriverKernel<<<1, 1>>>(d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  Seed<16, 16> out{};
  d_out.get(out);
  EXPECT_EQ(out.atoms[0], (1u << 0) | (1u << 15));
  EXPECT_EQ(out.numAtoms, 2);
}

// ---------------------------------------------------------------------------
// SeedQueue: pushWithinThread / popWithinThread / batchReserveCooperative /
// popReserveCooperative
// ---------------------------------------------------------------------------

namespace {

// Single-thread driver: push 5 ints, pop 5, record the popped order so
// the host can verify LIFO.
__global__ void queuePushPopLIFODriver(int* poppedOrder, int* outSizeAfterPushes) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __shared__ int storage[8];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  queue.init(storage, 8);
  for (int v : {10, 20, 30, 40, 50}) {
    bool ok = queue.pushWithinThread(v);
    (void)ok;
  }
  *outSizeAfterPushes = queue.size();
  for (int i = 0; i < 5; ++i) {
    int  popped    = -1;
    bool ok        = queue.popWithinThread(popped);
    poppedOrder[i] = ok ? popped : -1;
  }
}

__global__ void queueOverflowDriver(bool* pushOk, int* finalSize) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __shared__ int storage[4];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  queue.init(storage, 4);
  for (int i = 0; i < 5; ++i) {
    pushOk[i] = queue.pushWithinThread(100 + i);
  }
  *finalSize = queue.size();
}

__global__ void queuePopFromEmptyDriver(bool* popOk, int* outVal) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __shared__ int storage[4];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  queue.init(storage, 4);
  int v   = 12345;
  *popOk  = queue.popWithinThread(v);
  *outVal = v;
}

// Multi-thread (1 warp = 32 threads): each lane pushes its own payload.
// Verifies CAS-based pushes don't lose any concurrent inserts.
__global__ void queueConcurrentPushesDriver(int* outStorage, int* outSize) {
  __shared__ int storage[64];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  if (threadIdx.x == 0)
    queue.init(storage, 64);
  __syncthreads();

  bool ok = queue.pushWithinThread(1000 + static_cast<int>(threadIdx.x));
  __syncthreads();
  (void)ok;

  if (threadIdx.x == 0) {
    *outSize = queue.size();
    for (int i = 0; i < queue.size(); ++i)
      outStorage[i] = storage[i];
  }
}

// Multi-thread: each of 32 lanes pops one element concurrently.
// Verifies CAS-based pops hand out every slot exactly once.
__global__ void queueConcurrentPopsDriver(int* outVals, bool* outOk, int* outSize) {
  __shared__ int storage[32];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  if (threadIdx.x == 0) {
    queue.init(storage, 32);
    for (int i = 0; i < 32; ++i)
      queue.pushWithinThread(3000 + i);
  }
  __syncthreads();

  int v                = -1;
  outOk[threadIdx.x]   = queue.popWithinThread(v);
  outVals[threadIdx.x] = v;
  __syncthreads();

  if (threadIdx.x == 0)
    *outSize = queue.size();
}

// Multi-thread: 32 lanes do a single batchReserveCooperative call,
// each lane writes its own payload into the reserved slot.  Verifies
// the cooperative reservation broadcasts correctly and all 32 slots
// land contiguously.
__global__ void queueBatchReserveDriver(int* outStorage, int* outStart, int* outSize) {
  __shared__ int storage[64];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  if (threadIdx.x == 0)
    queue.init(storage, 64);
  __syncthreads();

  auto      block  = cg::this_thread_block();
  auto      warp   = cg::tiled_partition<32>(block);
  const int laneId = static_cast<int>(warp.thread_rank());
  const int start  = queue.batchReserveCooperative(warp, 32);
  if (start >= 0) {
    queue.slot(start + laneId) = 2000 + laneId;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    *outStart = start;
    *outSize  = queue.size();
    for (int i = 0; i < queue.size(); ++i)
      outStorage[i] = storage[i];
  }
}

__global__ void queueBatchReserveOverflowDriver(int* outStart, int* outSize) {
  __shared__ int storage[10];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  if (threadIdx.x == 0)
    queue.init(storage, 10);
  __syncthreads();

  auto      block = cg::this_thread_block();
  auto      warp  = cg::tiled_partition<32>(block);
  const int start = queue.batchReserveCooperative(warp, 15);
  __syncthreads();

  if (threadIdx.x == 0) {
    *outStart = start;
    *outSize  = queue.size();
  }
}

// Capacity 10, pre-push 7 elements, then request 5 slots: only 3 free.
// Reservation must fail atomically (no partial reservation, no slots
// consumed) -- partial fit must be treated as a hard failure.
__global__ void queueBatchReservePartialFitDriver(int* outStart, int* outSizeBefore, int* outSizeAfter) {
  __shared__ int storage[10];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  if (threadIdx.x == 0) {
    queue.init(storage, 10);
    for (int i = 0; i < 7; ++i)
      queue.pushWithinThread(i);
  }
  __syncthreads();

  if (threadIdx.x == 0)
    *outSizeBefore = queue.size();
  __syncthreads();

  auto      block = cg::this_thread_block();
  auto      warp  = cg::tiled_partition<32>(block);
  const int start = queue.batchReserveCooperative(warp, 5);
  __syncthreads();

  if (threadIdx.x == 0) {
    *outStart     = start;
    *outSizeAfter = queue.size();
  }
}

// Multi-thread: pop reservation broadcasts the old top to every lane,
// callers read slot(oldTop - 1), and an empty queue yields -1 to all
// lanes without consuming anything.
__global__ void queuePopReserveDriver(int* outPerLaneTop, int* outValue, int* outEmptyTop, int* outSizeAfter) {
  __shared__ int storage[8];
  __shared__ SeedQueue<int, ThreadBlockScope> queue;
  if (threadIdx.x == 0) {
    queue.init(storage, 8);
    queue.pushWithinThread(111);
    queue.pushWithinThread(222);
    queue.pushWithinThread(333);
  }
  __syncthreads();

  auto      block  = cg::this_thread_block();
  auto      warp   = cg::tiled_partition<32>(block);
  const int laneId = static_cast<int>(warp.thread_rank());

  const int oldTop      = queue.popReserveCooperative(warp);
  outPerLaneTop[laneId] = oldTop;
  if (laneId == 0 && oldTop > 0)
    *outValue = queue.slot(oldTop - 1);
  __syncthreads();

  // Drain the remaining two entries, then reserve on an empty queue.
  queue.popReserveCooperative(warp);
  queue.popReserveCooperative(warp);
  const int emptyTop = queue.popReserveCooperative(warp);
  __syncthreads();

  if (laneId == 0) {
    *outEmptyTop  = emptyTop;
    *outSizeAfter = queue.size();
  }
}

}  // namespace

TEST(FMCSPrimitives, QueuePushPopLIFO) {
  AsyncDeviceVector<int> d_order(5);
  AsyncDevicePtr<int>    d_size;

  queuePushPopLIFODriver<<<1, 1>>>(d_order.data(), d_size.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int size = 0;
  d_size.get(size);
  std::vector<int> order(5);
  d_order.copyToHost(order);
  EXPECT_EQ(size, 5);
  // LIFO: first popped should be the last pushed.
  EXPECT_EQ(order[0], 50);
  EXPECT_EQ(order[1], 40);
  EXPECT_EQ(order[2], 30);
  EXPECT_EQ(order[3], 20);
  EXPECT_EQ(order[4], 10);
}

TEST(FMCSPrimitives, QueuePushOverflowReturnsFalse) {
  AsyncDeviceVector<bool> d_ok(5);
  AsyncDevicePtr<int>     d_size;

  queueOverflowDriver<<<1, 1>>>(d_ok.data(), d_size.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // std::vector<bool> is bit-packed; copy through a plain buffer instead.
  bool okRaw[5] = {};
  d_ok.copyToHost(okRaw, 5);
  int size = 0;
  d_size.get(size);

  EXPECT_TRUE(okRaw[0]);
  EXPECT_TRUE(okRaw[1]);
  EXPECT_TRUE(okRaw[2]);
  EXPECT_TRUE(okRaw[3]);
  EXPECT_FALSE(okRaw[4]);  // 5th push should fail; capacity == 4.
  EXPECT_EQ(size, 4);
}

TEST(FMCSPrimitives, QueuePopFromEmptyReturnsFalse) {
  AsyncDevicePtr<bool> d_ok;
  AsyncDevicePtr<int>  d_val;

  queuePopFromEmptyDriver<<<1, 1>>>(d_ok.data(), d_val.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  bool ok  = true;
  int  val = 0;
  d_ok.get(ok);
  d_val.get(val);
  EXPECT_FALSE(ok);
  EXPECT_EQ(val, 12345);  // Caller's outElement should be untouched.
}

TEST(FMCSPrimitives, QueueConcurrentPushesAllLand) {
  AsyncDeviceVector<int> d_storage(64);
  AsyncDevicePtr<int>    d_size;

  queueConcurrentPushesDriver<<<1, 32>>>(d_storage.data(), d_size.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int size = 0;
  d_size.get(size);
  ASSERT_EQ(size, 32);
  std::vector<int> storage(64);
  d_storage.copyToHost(storage);
  // All 32 lanes' payloads should be present exactly once.  Order is
  // non-deterministic across CAS races; check by set membership.
  std::set<int> seen;
  for (int i = 0; i < 32; ++i)
    seen.insert(storage[i]);
  EXPECT_EQ(seen.size(), 32u);
  for (int i = 0; i < 32; ++i) {
    EXPECT_TRUE(seen.count(1000 + i)) << "missing lane " << i;
  }
}

TEST(FMCSPrimitives, QueueConcurrentPopsAllDistinct) {
  AsyncDeviceVector<int>  d_vals(32);
  AsyncDeviceVector<bool> d_ok(32);
  AsyncDevicePtr<int>     d_size;

  queueConcurrentPopsDriver<<<1, 32>>>(d_vals.data(), d_ok.data(), d_size.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int size = -1;
  d_size.get(size);
  std::vector<int> vals(32);
  d_vals.copyToHost(vals);
  bool okRaw[32] = {};
  d_ok.copyToHost(okRaw, 32);

  EXPECT_EQ(size, 0);
  std::set<int> seen;
  for (int i = 0; i < 32; ++i) {
    EXPECT_TRUE(okRaw[i]) << "lane " << i << " pop failed";
    seen.insert(vals[i]);
  }
  // Every pushed payload should be handed out exactly once.
  EXPECT_EQ(seen.size(), 32u);
  for (int i = 0; i < 32; ++i) {
    EXPECT_TRUE(seen.count(3000 + i)) << "missing payload " << i;
  }
}

TEST(FMCSPrimitives, QueueBatchReserveCooperative) {
  AsyncDeviceVector<int> d_storage(64);
  AsyncDevicePtr<int>    d_start;
  AsyncDevicePtr<int>    d_size;

  queueBatchReserveDriver<<<1, 32>>>(d_storage.data(), d_start.data(), d_size.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int start = -1;
  int size  = -1;
  d_start.get(start);
  d_size.get(size);
  std::vector<int> storage(64);
  d_storage.copyToHost(storage);
  EXPECT_EQ(start, 0);
  EXPECT_EQ(size, 32);
  // Each lane wrote 2000 + laneId into slot start + laneId; storage[i]
  // should equal 2000 + i for i in [0, 32).
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(storage[i], 2000 + i);
  }
}

TEST(FMCSPrimitives, QueueBatchReserveOverflowReturnsNegativeOne) {
  AsyncDevicePtr<int> d_start;
  AsyncDevicePtr<int> d_size;

  queueBatchReserveOverflowDriver<<<1, 32>>>(d_start.data(), d_size.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int start = 0;
  int size  = -1;
  d_start.get(start);
  d_size.get(size);
  EXPECT_EQ(start, -1);
  EXPECT_EQ(size, 0);  // No slots consumed on overflow.
}

TEST(FMCSPrimitives, QueueBatchReservePartialFitFails) {
  AsyncDevicePtr<int> d_start;
  AsyncDevicePtr<int> d_sizeBefore;
  AsyncDevicePtr<int> d_sizeAfter;

  queueBatchReservePartialFitDriver<<<1, 32>>>(d_start.data(), d_sizeBefore.data(), d_sizeAfter.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int start      = 0;
  int sizeBefore = -1;
  int sizeAfter  = -1;
  d_start.get(start);
  d_sizeBefore.get(sizeBefore);
  d_sizeAfter.get(sizeAfter);
  EXPECT_EQ(sizeBefore, 7);
  EXPECT_EQ(start, -1);     // Reservation must fail atomically.
  EXPECT_EQ(sizeAfter, 7);  // No partial slots consumed; size unchanged.
}

TEST(FMCSPrimitives, QueuePopReserveCooperative) {
  AsyncDeviceVector<int> d_perLaneTop(32);
  AsyncDevicePtr<int>    d_value;
  AsyncDevicePtr<int>    d_emptyTop;
  AsyncDevicePtr<int>    d_sizeAfter;

  queuePopReserveDriver<<<1, 32>>>(d_perLaneTop.data(), d_value.data(), d_emptyTop.data(), d_sizeAfter.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<int> perLaneTop(32);
  d_perLaneTop.copyToHost(perLaneTop);
  int value     = 0;
  int emptyTop  = 0;
  int sizeAfter = -1;
  d_value.get(value);
  d_emptyTop.get(emptyTop);
  d_sizeAfter.get(sizeAfter);

  // The reservation result must be broadcast identically to all lanes.
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(perLaneTop[i], 3) << "lane " << i;
  }
  EXPECT_EQ(value, 333);    // slot(oldTop - 1) is the LIFO top.
  EXPECT_EQ(emptyTop, -1);  // Empty queue: -1 to every lane.
  EXPECT_EQ(sizeAfter, 0);  // The empty-pop consumed nothing.
}

// ---------------------------------------------------------------------------
// warpCopy
// ---------------------------------------------------------------------------

namespace {

__global__ void warpCopyDriver(unsigned char* dst, const unsigned char* src, int nbytes) {
  auto block = cg::this_thread_block();
  auto warp  = cg::tiled_partition<32>(block);
  mcs::warpCopy(warp, dst, src, nbytes);
}

using QueuedSeedT = QueuedSeed<32, 32, 32, 32>;

__global__ void warpCopyQueuedSeedDriver(QueuedSeedT* dst, const QueuedSeedT* src) {
  auto block = cg::this_thread_block();
  auto warp  = cg::tiled_partition<32>(block);
  mcs::warpCopy(warp, dst, src, sizeof(QueuedSeedT));
}

}  // namespace

// Exercises the int4 wide path and every tail size (nbytes % 16 in
// {0, 4, 8, 12}), including sub-16-byte copies that skip the wide path
// entirely.  Also verifies warpCopy never writes past nbytes.
TEST(FMCSPrimitives, WarpCopySizesAndTailsNoOverrun) {
  constexpr int kMax   = 512;
  constexpr int kSlack = 64;  // Sentinel region past the copy end.

  std::vector<unsigned char> pattern(kMax + kSlack);
  for (int i = 0; i < kMax + kSlack; ++i) {
    pattern[i] = static_cast<unsigned char>((i * 13 + 7) & 0xFF);
  }
  const std::vector<unsigned char> sentinel(kMax + kSlack, 0xEE);

  AsyncDeviceVector<unsigned char> d_src(kMax + kSlack);
  AsyncDeviceVector<unsigned char> d_dst(kMax + kSlack);
  d_src.copyFromHost(pattern);

  // Multiples of 16 plus every 4-byte tail case, small and large.
  const int sizes[] = {4, 8, 12, 16, 20, 24, 28, 32, 48, 52, 244, 512};
  for (const int nbytes : sizes) {
    d_dst.copyFromHost(sentinel);
    warpCopyDriver<<<1, 32>>>(d_dst.data(), d_src.data(), nbytes);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "nbytes=" << nbytes;

    std::vector<unsigned char> result(kMax + kSlack);
    d_dst.copyToHost(result);
    for (int i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i]) << "nbytes=" << nbytes << " byte " << i;
    }
    for (int i = nbytes; i < nbytes + kSlack; ++i) {
      ASSERT_EQ(result[i], 0xEE) << "nbytes=" << nbytes << " overrun at byte " << i;
    }
  }
}

// Copies the struct the kernel actually moves with warpCopy and checks
// byte-exact round-trip (alignas(16) guarantees the aligned fast path).
TEST(FMCSPrimitives, WarpCopyQueuedSeedRoundTrip) {
  static_assert(sizeof(QueuedSeedT) % 16 == 0, "QueuedSeed must pad to an int4 multiple");

  QueuedSeedT hostSrc{};
  auto*       srcBytes = reinterpret_cast<unsigned char*>(&hostSrc);
  for (size_t i = 0; i < sizeof(QueuedSeedT); ++i) {
    srcBytes[i] = static_cast<unsigned char>((i * 31 + 3) & 0xFF);
  }

  AsyncDevicePtr<QueuedSeedT> d_src(hostSrc);
  AsyncDevicePtr<QueuedSeedT> d_dst;
  d_dst.memSet(0);

  warpCopyQueuedSeedDriver<<<1, 32>>>(d_dst.data(), d_src.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  QueuedSeedT hostDst{};
  d_dst.get(hostDst);
  EXPECT_EQ(std::memcmp(&hostDst, &hostSrc, sizeof(QueuedSeedT)), 0);
}
