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
//
// Unit tests for the per-helper device functions in fmcs_cuda/.  Each test
// launches a tiny __global__ driver that constructs inputs, calls one
// helper, and copies results back to host memory for assertion.  This
// file is populated incrementally as Steps 1-5 land real implementations.

#include <cooperative_groups.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <set>
#include <utility>
#include <vector>

#include "src/mcs/fmcs_cuda/fmcs_match.cuh"
#include "src/mcs/fmcs_cuda/fmcs_match_tables.cuh"
#include "src/mcs/fmcs_cuda/fmcs_seed.cuh"
#include "src/utils/device_vector.h"

namespace {

using nvMolKit::AsyncDevicePtr;
using nvMolKit::AsyncDeviceVector;

using mcs::fmcs::MatchResult;
using mcs::fmcs::Seed;

}  // namespace

// ---------------------------------------------------------------------------
// matchSingleBondWithinThread / matchIncrementalFastCooperative
// ---------------------------------------------------------------------------

namespace {

using mcs::fmcs::MatchTableDevice;
using mcs::fmcs::PairMatchTablesDevice;
using mcs::fmcs::SingleBondMatch;

using mcs::fmcs::DeviceCsrView;

// Match tables staged on the host and uploaded to device memory on
// demand.  Both atom and bond tables are 32-bit row-packed bitmasks
// (one bit per (q, t) pair).  Tests mutate the host staging vectors
// via the set*Bit helpers; device() uploads (if dirty) and returns the
// device view to pass to kernels.
struct ManagedMatchTables {
  std::vector<std::uint32_t>       atomHost;
  std::vector<std::uint32_t>       bondHost;
  AsyncDeviceVector<std::uint32_t> atomData;
  AsyncDeviceVector<std::uint32_t> bondData;
  int                              qNumAtoms = 0;
  int                              qNumBonds = 0;

  void allocate(int qAtoms, int tAtoms, int qBonds, int tBonds) {
    qNumAtoms                 = qAtoms;
    qNumBonds                 = qBonds;
    const int atomWordsPerRow = (tAtoms + 31) / 32;
    const int bondWordsPerRow = (tBonds + 31) / 32;
    atomHost.assign(qAtoms * atomWordsPerRow, 0);
    bondHost.assign(qBonds * bondWordsPerRow, 0);
    dev_.atoms = MatchTableDevice{nullptr, qAtoms, tAtoms, atomWordsPerRow};
    dev_.bonds = MatchTableDevice{nullptr, qBonds, tBonds, bondWordsPerRow};
    dirty_     = true;
  }

  void setAtomBit(int qAtom, int tAtom) {
    atomHost[qAtom * dev_.atoms.wordsPerRow + tAtom / 32] |= (1u << (tAtom % 32));
    dirty_ = true;
  }
  void setBondBit(int qBond, int tBond) {
    bondHost[qBond * dev_.bonds.wordsPerRow + tBond / 32] |= (1u << (tBond % 32));
    dirty_ = true;
  }
  void setAllAtomBits() {
    for (int q = 0; q < dev_.atoms.nRows; ++q)
      for (int t = 0; t < dev_.atoms.nCols; ++t)
        setAtomBit(q, t);
  }
  void setAllBondBits() {
    for (int q = 0; q < dev_.bonds.nRows; ++q)
      for (int t = 0; t < dev_.bonds.nCols; ++t)
        setBondBit(q, t);
  }

  PairMatchTablesDevice device() {
    if (dirty_) {
      atomData.setFromVector(atomHost);
      bondData.setFromVector(bondHost);
      dev_.atoms.data = atomData.data();
      dev_.bonds.data = bondData.data();
      dirty_          = false;
    }
    return dev_;
  }

 private:
  PairMatchTablesDevice dev_{};
  bool                  dirty_ = true;
};

// Owns the device buffers behind a DeviceCsrView.  Built from an
// undirected edge list: bond i is edges[i], and the CSR is the symmetric
// expansion of that list (each edge contributes one entry to each
// endpoint's row), matching what the host-side graph builder uploads for
// real molecules.
class TestGraph {
 public:
  TestGraph(int numAtoms, const std::vector<std::pair<int, int>>& edges) {
    const int numBonds = static_cast<int>(edges.size());

    std::vector<std::uint32_t> bondEndpointsHost(numBonds);
    for (int i = 0; i < numBonds; ++i) {
      bondEndpointsHost[i] =
        (static_cast<std::uint32_t>(edges[i].first) << 16) | static_cast<std::uint32_t>(edges[i].second);
    }

    // Counting sort into CSR: degree pass, prefix sum, then scatter.
    std::vector<std::uint32_t> rowOffsetsHost(numAtoms + 1, 0);
    for (const auto& edge : edges) {
      ++rowOffsetsHost[edge.first + 1];
      ++rowOffsetsHost[edge.second + 1];
    }
    for (int atom = 0; atom < numAtoms; ++atom) {
      rowOffsetsHost[atom + 1] += rowOffsetsHost[atom];
    }

    std::vector<std::uint32_t> colIndicesHost(2 * numBonds);
    std::vector<std::uint32_t> bondIndicesHost(2 * numBonds);
    std::vector<std::uint32_t> cursor(rowOffsetsHost.begin(), rowOffsetsHost.end() - 1);
    for (int bond = 0; bond < numBonds; ++bond) {
      const int u                = edges[bond].first;
      const int v                = edges[bond].second;
      colIndicesHost[cursor[u]]  = static_cast<std::uint32_t>(v);
      bondIndicesHost[cursor[u]] = static_cast<std::uint32_t>(bond);
      ++cursor[u];
      colIndicesHost[cursor[v]]  = static_cast<std::uint32_t>(u);
      bondIndicesHost[cursor[v]] = static_cast<std::uint32_t>(bond);
      ++cursor[v];
    }

    bondEndpoints_.setFromVector(bondEndpointsHost);
    rowOffsets_.setFromVector(rowOffsetsHost);
    colIndices_.setFromVector(colIndicesHost);
    bondIndices_.setFromVector(bondIndicesHost);

    view_.bondEndpoints = bondEndpoints_.data();
    view_.rowOffsets    = rowOffsets_.data();
    view_.colIndices    = colIndices_.data();
    view_.bondIndices   = bondIndices_.data();
    view_.numAtoms      = numAtoms;
    view_.numBonds      = numBonds;
  }

  DeviceCsrView view() const { return view_; }

 private:
  AsyncDeviceVector<std::uint32_t> bondEndpoints_;
  AsyncDeviceVector<std::uint32_t> rowOffsets_;
  AsyncDeviceVector<std::uint32_t> colIndices_;
  AsyncDeviceVector<std::uint32_t> bondIndices_;
  DeviceCsrView                    view_{};
};

// ---- matchSingleBondWithinThread ----

struct SingleBondTestOut {
  bool            ok;
  SingleBondMatch match;
};

__global__ void matchSingleBondDriver(int                   qBondIdx,
                                      int                   tBondIdx,
                                      bool                  reversed,
                                      DeviceCsrView         qView,
                                      DeviceCsrView         tView,
                                      PairMatchTablesDevice tables,
                                      SingleBondTestOut*    out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  SingleBondMatch sm{};
  out->ok    = mcs::fmcs::matchSingleBondWithinThread(qBondIdx, tBondIdx, reversed, qView, tView, tables, sm);
  out->match = sm;
}

}  // namespace

TEST(FMCSUnit, MatchSingleBondForwardOrientation) {
  // Query bond (0,1), target bond (0,1).  All atoms / bonds compatible.
  TestGraph          query(2,
                           {
                    {0, 1}
  });
  TestGraph          target(2,
                            {
                     {0, 1}
  });
  ManagedMatchTables tables;
  tables.allocate(2, 2, 1, 1);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<SingleBondTestOut> d_out;
  matchSingleBondDriver<<<1, 1>>>(0,
                                  0,
                                  /*reversed=*/false,
                                  query.view(),
                                  target.view(),
                                  tables.device(),
                                  d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SingleBondTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_EQ(out.match.targetAtomU, 0u);  // qU=0 -> tU=0
  EXPECT_EQ(out.match.targetAtomV, 1u);  // qV=1 -> tV=1
}

TEST(FMCSUnit, MatchSingleBondReverseOrientation) {
  TestGraph          query(2,
                           {
                    {0, 1}
  });
  TestGraph          target(2,
                            {
                     {0, 1}
  });
  ManagedMatchTables tables;
  tables.allocate(2, 2, 1, 1);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<SingleBondTestOut> d_out;
  matchSingleBondDriver<<<1, 1>>>(0,
                                  0,
                                  /*reversed=*/true,
                                  query.view(),
                                  target.view(),
                                  tables.device(),
                                  d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SingleBondTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_EQ(out.match.targetAtomU, 1u);  // qU=0 -> tV=1 (reversed)
  EXPECT_EQ(out.match.targetAtomV, 0u);  // qV=1 -> tU=0
}

TEST(FMCSUnit, MatchSingleBondBondTableRejection) {
  TestGraph          query(2,
                           {
                    {0, 1}
  });
  TestGraph          target(2,
                            {
                     {0, 1}
  });
  ManagedMatchTables tables;
  tables.allocate(2, 2, 1, 1);
  tables.setAllAtomBits();
  // Deliberately leave bondData all zero -> bond-table reject.

  AsyncDevicePtr<SingleBondTestOut> d_out;
  matchSingleBondDriver<<<1, 1>>>(0,
                                  0,
                                  /*reversed=*/false,
                                  query.view(),
                                  target.view(),
                                  tables.device(),
                                  d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SingleBondTestOut out{};
  d_out.get(out);

  EXPECT_FALSE(out.ok);
}

TEST(FMCSUnit, MatchSingleBondAtomTableRejection) {
  TestGraph          query(2,
                           {
                    {0, 1}
  });
  TestGraph          target(2,
                            {
                     {0, 1}
  });
  ManagedMatchTables tables;
  tables.allocate(2, 2, 1, 1);
  tables.setAllBondBits();
  // Atom 0 compatible with target atom 0, but atom 1 is incompatible
  // with target atom 1 -> forward orientation rejects on second atom.
  tables.setAtomBit(0, 0);
  tables.setAtomBit(0, 1);
  tables.setAtomBit(1, 0);
  // Note: (1, 1) deliberately left unset.

  AsyncDevicePtr<SingleBondTestOut> d_out;
  matchSingleBondDriver<<<1, 1>>>(0,
                                  0,
                                  /*reversed=*/false,
                                  query.view(),
                                  target.view(),
                                  tables.device(),
                                  d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SingleBondTestOut out{};
  d_out.get(out);

  EXPECT_FALSE(out.ok);
}

// ---- matchIncrementalFastCooperative ----

namespace {

// Helper: build a parent MatchResult that records the mapping
// {qAtom[i] -> tAtom[i]} and {qBond[i] -> tBond[i]} from caller-supplied
// parallel arrays.  Used by the incremental tests to set up "what the
// parent already had matched" before adding new bonds.
template <int maxA, int maxB, int maxTA, int maxTB>
__device__ __forceinline__ void buildParentMatch(mcs::fmcs::MatchResult<maxA, maxB, maxTA, maxTB>& match,
                                                 const int*                                        qAtoms,
                                                 const int*                                        tAtoms,
                                                 int                                               nAtomMaps,
                                                 const int*                                        qBonds,
                                                 const int*                                        tBonds,
                                                 int                                               nBondMaps) {
  using MatchT = mcs::fmcs::MatchResult<maxA, maxB, maxTA, maxTB>;
  mcs::fmcs::matchResultClearWithinThread(match);
  for (int i = 0; i < nAtomMaps; ++i) {
    match.targetAtomIdx[qAtoms[i]] = static_cast<std::uint8_t>(tAtoms[i]);
    const int t                    = tAtoms[i];
    match.visitedTargetAtoms[t / MatchT::kTargetAtomBitsPerWord] |= (typename MatchT::target_atom_word{1})
                                                                 << (t % MatchT::kTargetAtomBitsPerWord);
  }
  for (int i = 0; i < nBondMaps; ++i) {
    match.targetBondIdx[qBonds[i]] = static_cast<std::uint8_t>(tBonds[i]);
    const int t                    = tBonds[i];
    match.visitedTargetBonds[t / MatchT::kTargetBondBitsPerWord] |= (typename MatchT::target_bond_word{1})
                                                                 << (t % MatchT::kTargetBondBitsPerWord);
  }
  match.matchedAtomSize = static_cast<std::uint16_t>(nAtomMaps);
  match.matchedBondSize = static_cast<std::uint16_t>(nBondMaps);
  match.empty           = (nAtomMaps == 0 && nBondMaps == 0);
}

}  // namespace

namespace mcs_fmcs_incremental_test {

using QueuedT16 = mcs::fmcs::QueuedSeed<16, 16, 16, 16>;

struct IncrementalTestOut {
  bool      ok;
  QueuedT16 child;
};

// One-warp driver: builds parent match in shared mem, then constructs
// the child seed (parent + new bonds) and runs
// matchIncrementalFastCooperative on it.
__global__ void matchIncrementalAtomAddingDriver(DeviceCsrView         qView,
                                                 DeviceCsrView         tView,
                                                 PairMatchTablesDevice tables,
                                                 IncrementalTestOut*   out) {
  __shared__ QueuedT16 child;
  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(child.seed);
    // Parent: bond (0,1) mapped 0->0, 1->1, bond 0->target bond 0.
    int qA[] = {0, 1};
    int tA[] = {0, 1};
    int qB[] = {0};
    int tB[] = {0};
    buildParentMatch(child.match, qA, tA, 2, qB, tB, 1);
    // Child seed has bonds {0, 1} and atoms {0, 1, 2}.  Bond 1 is the
    // new atom-adding bond (qU=1 already mapped, qV=2 unmapped).
    mcs::fmcs::seedAddBondWithinThread(child.seed, 0);
    mcs::fmcs::seedAddBondWithinThread(child.seed, 1);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 0);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 1);
    mcs::fmcs::seedBeginGrowStepWithinThread(child.seed);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 2);
  }
  __syncthreads();

  auto block = cooperative_groups::this_thread_block();
  auto warp  = cooperative_groups::tiled_partition<32>(block);
  bool ok    = mcs::fmcs::matchIncrementalFastCooperative(warp, child.seed, qView, tView, tables, child.match);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok    = ok;
    out->child = child;
  }
}

__global__ void matchIncrementalRingClosingDriver(DeviceCsrView         qView,
                                                  DeviceCsrView         tView,
                                                  PairMatchTablesDevice tables,
                                                  IncrementalTestOut*   out) {
  __shared__ QueuedT16 child;
  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(child.seed);
    // 4-atom square query: bonds (0,1), (1,2), (2,3), (0,3).  Parent
    // has all 4 atoms mapped via 3 path bonds; child adds the
    // ring-closing 4th bond (0,3).
    int qA[] = {0, 1, 2, 3};
    int tA[] = {0, 1, 2, 3};
    int qB[] = {0, 1, 2};
    int tB[] = {0, 1, 2};
    buildParentMatch(child.match, qA, tA, 4, qB, tB, 3);
    for (int b : {0, 1, 2, 3})
      mcs::fmcs::seedAddBondWithinThread(child.seed, b);
    for (int a : {0, 1, 2, 3})
      mcs::fmcs::seedAddAtomWithinThread(child.seed, a);
  }
  __syncthreads();

  auto block = cooperative_groups::this_thread_block();
  auto warp  = cooperative_groups::tiled_partition<32>(block);
  bool ok    = mcs::fmcs::matchIncrementalFastCooperative(warp, child.seed, qView, tView, tables, child.match);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok    = ok;
    out->child = child;
  }
}

__global__ void matchIncrementalVisitedConflictDriver(DeviceCsrView         qView,
                                                      DeviceCsrView         tView,
                                                      PairMatchTablesDevice tables,
                                                      IncrementalTestOut*   out) {
  __shared__ QueuedT16 child;
  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(child.seed);
    // Query: atoms 0,1,2; bonds (0,1), (0,2).  Target: atoms 0,1; one
    // bond (0,1).  Parent has bond (0,1) mapped (qU=0->t=0, qV=1->t=1).
    // Now child wants atom-adding bond (0,2), but the only target bond
    // out of t=0 is (0,1) and t=1 is already visited -> must fail.
    int qA[] = {0, 1};
    int tA[] = {0, 1};
    int qB[] = {0};
    int tB[] = {0};
    buildParentMatch(child.match, qA, tA, 2, qB, tB, 1);
    mcs::fmcs::seedAddBondWithinThread(child.seed, 0);
    mcs::fmcs::seedAddBondWithinThread(child.seed, 1);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 0);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 1);
    mcs::fmcs::seedBeginGrowStepWithinThread(child.seed);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 2);
  }
  __syncthreads();

  auto block = cooperative_groups::this_thread_block();
  auto warp  = cooperative_groups::tiled_partition<32>(block);
  bool ok    = mcs::fmcs::matchIncrementalFastCooperative(warp, child.seed, qView, tView, tables, child.match);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok    = ok;
    out->child = child;
  }
}

__global__ void matchIncrementalTwoBondChainDriver(DeviceCsrView         qView,
                                                   DeviceCsrView         tView,
                                                   PairMatchTablesDevice tables,
                                                   IncrementalTestOut*   out) {
  __shared__ QueuedT16 child;
  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(child.seed);
    // 4-atom path 0-1-2-3.  Parent has only bond (0,1) mapped.  Child
    // adds bonds (1,2) and (2,3) in one matchIncrementalFast call.
    int qA[] = {0, 1};
    int tA[] = {0, 1};
    int qB[] = {0};
    int tB[] = {0};
    buildParentMatch(child.match, qA, tA, 2, qB, tB, 1);
    mcs::fmcs::seedAddBondWithinThread(child.seed, 0);
    mcs::fmcs::seedAddBondWithinThread(child.seed, 1);
    mcs::fmcs::seedAddBondWithinThread(child.seed, 2);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 0);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 1);
    mcs::fmcs::seedBeginGrowStepWithinThread(child.seed);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 2);
    mcs::fmcs::seedAddAtomWithinThread(child.seed, 3);
  }
  __syncthreads();

  auto block = cooperative_groups::this_thread_block();
  auto warp  = cooperative_groups::tiled_partition<32>(block);
  bool ok    = mcs::fmcs::matchIncrementalFastCooperative(warp, child.seed, qView, tView, tables, child.match);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok    = ok;
    out->child = child;
  }
}

}  // namespace mcs_fmcs_incremental_test

TEST(FMCSUnit, MatchIncrementalFastAtomAdding) {
  using mcs_fmcs_incremental_test::IncrementalTestOut;
  using mcs_fmcs_incremental_test::matchIncrementalAtomAddingDriver;

  // Query/target are both a 3-atom path 0-1-2 with bonds (0,1), (1,2).
  TestGraph          query(3,
                           {
                    {0, 1},
                    {1, 2}
  });
  TestGraph          target(3,
                            {
                     {0, 1},
                     {1, 2}
  });
  ManagedMatchTables tables;
  tables.allocate(3, 3, 2, 2);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<IncrementalTestOut> d_out;
  matchIncrementalAtomAddingDriver<<<1, 32>>>(query.view(), target.view(), tables.device(), d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  IncrementalTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_EQ(out.child.match.targetBondIdx[1], 1u);
  EXPECT_EQ(out.child.match.targetAtomIdx[2], 2u);
  EXPECT_EQ(out.child.match.matchedBondSize, 2);
  EXPECT_EQ(out.child.match.matchedAtomSize, 3);
}

TEST(FMCSUnit, MatchIncrementalFastRingClosing) {
  using mcs_fmcs_incremental_test::IncrementalTestOut;
  using mcs_fmcs_incremental_test::matchIncrementalRingClosingDriver;

  // 4-atom square with one diagonal-free closure.  Bonds: (0,1) (1,2)
  // (2,3) (0,3).
  TestGraph          query(4,
                           {
                    {0, 1},
                    {1, 2},
                    {2, 3},
                    {0, 3}
  });
  TestGraph          target(4,
                            {
                     {0, 1},
                     {1, 2},
                     {2, 3},
                     {0, 3}
  });
  ManagedMatchTables tables;
  tables.allocate(4, 4, 4, 4);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<IncrementalTestOut> d_out;
  matchIncrementalRingClosingDriver<<<1, 32>>>(query.view(), target.view(), tables.device(), d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  IncrementalTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_EQ(out.child.match.targetBondIdx[3], 3u);
  EXPECT_EQ(out.child.match.matchedBondSize, 4);
  // No new atoms added by the ring-closing bond.
  EXPECT_EQ(out.child.match.matchedAtomSize, 4);
}

TEST(FMCSUnit, MatchIncrementalFastVisitedConflictFails) {
  using mcs_fmcs_incremental_test::IncrementalTestOut;
  using mcs_fmcs_incremental_test::matchIncrementalVisitedConflictDriver;

  // Query: 3 atoms / 2 bonds.  Target: 2 atoms / 1 bond.  Parent has
  // bond (0,1) mapped; trying to extend with bond (0,2) forces atom 2
  // onto target atom 1, which is already visited.
  TestGraph          query(3,
                           {
                    {0, 1},
                    {0, 2}
  });
  TestGraph          target(2,
                            {
                     {0, 1}
  });
  ManagedMatchTables tables;
  tables.allocate(3, 2, 2, 1);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<IncrementalTestOut> d_out;
  matchIncrementalVisitedConflictDriver<<<1, 32>>>(query.view(), target.view(), tables.device(), d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  IncrementalTestOut out{};
  d_out.get(out);

  EXPECT_FALSE(out.ok);
}

TEST(FMCSUnit, MatchIncrementalFastTwoBondChain) {
  using mcs_fmcs_incremental_test::IncrementalTestOut;
  using mcs_fmcs_incremental_test::matchIncrementalTwoBondChainDriver;

  // Both sides are the 4-atom path 0-1-2-3 with bonds 0=(0,1), 1=(1,2),
  // 2=(2,3).
  TestGraph          query(4,
                           {
                    {0, 1},
                    {1, 2},
                    {2, 3}
  });
  TestGraph          target(4,
                            {
                     {0, 1},
                     {1, 2},
                     {2, 3}
  });
  ManagedMatchTables tables;
  tables.allocate(4, 4, 3, 3);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<IncrementalTestOut> d_out;
  matchIncrementalTwoBondChainDriver<<<1, 32>>>(query.view(), target.view(), tables.device(), d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  IncrementalTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_EQ(out.child.match.targetBondIdx[1], 1u);
  EXPECT_EQ(out.child.match.targetBondIdx[2], 2u);
  EXPECT_EQ(out.child.match.targetAtomIdx[2], 2u);
  EXPECT_EQ(out.child.match.targetAtomIdx[3], 3u);
  EXPECT_EQ(out.child.match.matchedBondSize, 3);
  EXPECT_EQ(out.child.match.matchedAtomSize, 4);
}
