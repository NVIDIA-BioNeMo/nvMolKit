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

// Tiny CSR-view used by the match helpers via duck-typing; satisfies the
// QueryTopology / TargetTopology template requirement (bondEndpoints +
// numAtoms / numBonds), with optional CSR adjacency fields.
struct TestCsrView {
  static constexpr bool kHasAdjacencyBondIndices = false;

  const std::uint32_t* bondEndpoints = nullptr;
  int                  numAtoms      = 0;
  int                  numBonds      = 0;
  const std::uint32_t* rowOffsets    = nullptr;
  const std::uint32_t* colIndices    = nullptr;
  const std::uint32_t* bondIndices   = nullptr;
};

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

AsyncDeviceVector<std::uint32_t> makeBondEndpointsDevice(const std::vector<std::pair<int, int>>& edges) {
  std::vector<std::uint32_t> host(edges.size());
  for (size_t i = 0; i < edges.size(); ++i) {
    host[i] = (static_cast<std::uint32_t>(edges[i].first) << 16) | static_cast<std::uint32_t>(edges[i].second);
  }
  AsyncDeviceVector<std::uint32_t> dev(edges.size());
  dev.copyFromHost(host);
  return dev;
}

}  // namespace

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
namespace mcs_fmcs_substructure_test {

using QueuedT16 = mcs::fmcs::QueuedSeed<16, 16, 16, 16>;

struct SubstructureTestOut {
  bool      ok;
  bool      overflowed;
  QueuedT16 child;
};

constexpr int kTestSubstructurePartialCapacity = 64;

__device__ __forceinline__ void addMaskSeed(QueuedT16& child, std::uint32_t atomMask, std::uint32_t bondMask) {
  mcs::fmcs::seedClearWithinThread(child.seed);
  mcs::fmcs::matchResultClearWithinThread(child.match);
  for (int a = 0; a < 16; ++a) {
    if ((atomMask >> a) & 1u) {
      mcs::fmcs::seedAddAtomWithinThread(child.seed, a);
    }
  }
  for (int b = 0; b < 16; ++b) {
    if ((bondMask >> b) & 1u) {
      mcs::fmcs::seedAddBondWithinThread(child.seed, b);
    }
  }
}

__global__ void matchSubstructureMaskDriver(const std::uint32_t*  qBE,
                                            int                   qNumAtoms,
                                            int                   qNumBonds,
                                            const std::uint32_t*  tBE,
                                            int                   tNumAtoms,
                                            int                   tNumBonds,
                                            PairMatchTablesDevice tables,
                                            std::uint32_t         atomMask,
                                            std::uint32_t         bondMask,
                                            std::uint8_t*         partialStorage,
                                            int                   partialCapacity,
                                            SubstructureTestOut*  out) {
  __shared__ QueuedT16 child;
  __shared__ mcs::fmcs::FmcsSubstructureScratch<16, 16> scratch;
  if (threadIdx.x == 0) {
    addMaskSeed(child, atomMask, bondMask);
  }
  __syncthreads();

  auto        block = cooperative_groups::this_thread_block();
  auto        warp  = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBE, qNumAtoms, qNumBonds};
  TestCsrView tView{tBE, tNumAtoms, tNumBonds};
  bool        overflowed = false;
  bool        ok         = mcs::fmcs::matchSeedSubstructureCooperative(warp,
                                                        child.seed,
                                                        qView,
                                                        tView,
                                                        tables,
                                                        child.match,
                                                        scratch,
                                                        partialStorage,
                                                        partialCapacity,
                                                        &overflowed);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok         = ok;
    out->overflowed = overflowed;
    out->child      = child;
  }
}

__global__ void matchFallbackBadParentDriver(const std::uint32_t*  qBE,
                                             int                   qNumAtoms,
                                             int                   qNumBonds,
                                             const std::uint32_t*  tBE,
                                             int                   tNumAtoms,
                                             int                   tNumBonds,
                                             PairMatchTablesDevice tables,
                                             std::uint8_t*         partialStorage,
                                             int                   partialCapacity,
                                             SubstructureTestOut*  out) {
  __shared__ QueuedT16 child;
  __shared__ mcs::fmcs::FmcsSubstructureScratch<16, 16> scratch;
  __shared__ int                                        scratchLock;
  if (threadIdx.x == 0) {
    // Query seed is the 4-edge path inside the triangle-with-leaves
    // repro: query bonds 1,2,3,4 and all five atoms.  The stored parent
    // match maps q bond 1 = (0,2) onto target bond 0 = (0,3), which is
    // locally valid but blocks q bond 2 = (0,4) in the fast extender.
    addMaskSeed(child, /*atomMask=*/0x1Fu, /*bondMask=*/0x1Eu);
    mcs::fmcs::matchResultClearWithinThread(child.match);
    using MatchT                 = decltype(child.match);
    child.match.targetAtomIdx[0] = 0;
    child.match.targetAtomIdx[2] = 3;
    child.match.visitedTargetAtoms[0 / MatchT::kTargetAtomBitsPerWord] |= typename MatchT::target_atom_word{1}
                                                                       << (0 % MatchT::kTargetAtomBitsPerWord);
    child.match.visitedTargetAtoms[3 / MatchT::kTargetAtomBitsPerWord] |= typename MatchT::target_atom_word{1}
                                                                       << (3 % MatchT::kTargetAtomBitsPerWord);
    child.match.targetBondIdx[1] = 0;
    child.match.visitedTargetBonds[0 / MatchT::kTargetBondBitsPerWord] |= typename MatchT::target_bond_word{1}
                                                                       << (0 % MatchT::kTargetBondBitsPerWord);
    child.match.matchedAtomSize = 2;
    child.match.matchedBondSize = 1;
    child.match.empty           = false;
    scratchLock                 = 0;
  }
  __syncthreads();

  auto        block = cooperative_groups::this_thread_block();
  auto        warp  = cooperative_groups::tiled_partition<32>(block);
  TestCsrView qView{qBE, qNumAtoms, qNumBonds};
  TestCsrView tView{tBE, tNumAtoms, tNumBonds};
  bool        overflowed = false;
  bool        ok         = mcs::fmcs::matchSeedWithSubstructureFallbackCooperative(warp,
                                                                    child.seed,
                                                                    qView,
                                                                    tView,
                                                                    tables,
                                                                    child.match,
                                                                    scratch,
                                                                    &scratchLock,
                                                                    partialStorage,
                                                                    partialCapacity,
                                                                    &overflowed);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok         = ok;
    out->overflowed = overflowed;
    out->child      = child;
  }
}

}  // namespace mcs_fmcs_substructure_test

TEST(FMCSUnit, MatchSeedSubstructurePath) {
  using namespace mcs_fmcs_substructure_test;

  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {2, 3}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {2, 3},
    {3, 4}
  });
  ManagedMatchTables tables;
  tables.allocate(4, 5, 3, 4);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<SubstructureTestOut> d_out;
  AsyncDeviceVector<std::uint8_t>     partials(2 * kTestSubstructurePartialCapacity * 16);
  matchSubstructureMaskDriver<<<1, 32>>>(qBE.data(),
                                         4,
                                         3,
                                         tBE.data(),
                                         5,
                                         4,
                                         tables.device(),
                                         /*atomMask=*/0xFu,
                                         /*bondMask=*/0x7u,
                                         partials.data(),
                                         kTestSubstructurePartialCapacity,
                                         d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SubstructureTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_FALSE(out.overflowed);
  EXPECT_EQ(out.child.match.matchedAtomSize, 4);
  EXPECT_EQ(out.child.match.matchedBondSize, 3);
  for (int q = 0; q < 4; ++q) {
    EXPECT_NE(out.child.match.targetAtomIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }
  for (int q = 0; q < 3; ++q) {
    EXPECT_NE(out.child.match.targetBondIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }
}

TEST(FMCSUnit, MatchSeedSubstructureRejectsNoMatch) {
  using namespace mcs_fmcs_substructure_test;

  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {0, 2}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2}
  });
  ManagedMatchTables tables;
  tables.allocate(3, 3, 3, 2);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<SubstructureTestOut> d_out;
  AsyncDeviceVector<std::uint8_t>     partials(2 * kTestSubstructurePartialCapacity * 16);
  matchSubstructureMaskDriver<<<1, 32>>>(qBE.data(),
                                         3,
                                         3,
                                         tBE.data(),
                                         3,
                                         2,
                                         tables.device(),
                                         /*atomMask=*/0x7u,
                                         /*bondMask=*/0x7u,
                                         partials.data(),
                                         kTestSubstructurePartialCapacity,
                                         d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SubstructureTestOut out{};
  d_out.get(out);

  EXPECT_FALSE(out.ok);
  EXPECT_FALSE(out.overflowed);
  EXPECT_TRUE(out.child.match.empty);
}

TEST(FMCSUnit, MatchSeedSubstructureRespectsAtomTable) {
  using namespace mcs_fmcs_substructure_test;

  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2}
  });
  ManagedMatchTables tables;
  tables.allocate(3, 3, 2, 2);
  tables.setAtomBit(0, 0);
  tables.setAtomBit(1, 1);
  tables.setAtomBit(2, 2);
  tables.setAllBondBits();

  AsyncDevicePtr<SubstructureTestOut> d_out;
  AsyncDeviceVector<std::uint8_t>     partials(2 * kTestSubstructurePartialCapacity * 16);
  matchSubstructureMaskDriver<<<1, 32>>>(qBE.data(),
                                         3,
                                         2,
                                         tBE.data(),
                                         3,
                                         2,
                                         tables.device(),
                                         /*atomMask=*/0x7u,
                                         /*bondMask=*/0x3u,
                                         partials.data(),
                                         kTestSubstructurePartialCapacity,
                                         d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SubstructureTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_FALSE(out.overflowed);
  EXPECT_EQ(out.child.match.targetAtomIdx[0], 0u);
  EXPECT_EQ(out.child.match.targetAtomIdx[1], 1u);
  EXPECT_EQ(out.child.match.targetAtomIdx[2], 2u);
  EXPECT_EQ(out.child.match.matchedAtomSize, 3);
  EXPECT_EQ(out.child.match.matchedBondSize, 2);
}

TEST(FMCSUnit, MatchSeedSubstructureRespectsBondTable) {
  using namespace mcs_fmcs_substructure_test;

  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {0, 2}
  });
  ManagedMatchTables tables;
  tables.allocate(3, 3, 2, 3);
  tables.setAllAtomBits();
  tables.setBondBit(0, 0);
  tables.setBondBit(1, 1);

  AsyncDevicePtr<SubstructureTestOut> d_out;
  AsyncDeviceVector<std::uint8_t>     partials(2 * kTestSubstructurePartialCapacity * 16);
  matchSubstructureMaskDriver<<<1, 32>>>(qBE.data(),
                                         3,
                                         2,
                                         tBE.data(),
                                         3,
                                         3,
                                         tables.device(),
                                         /*atomMask=*/0x7u,
                                         /*bondMask=*/0x3u,
                                         partials.data(),
                                         kTestSubstructurePartialCapacity,
                                         d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SubstructureTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_FALSE(out.overflowed);
  EXPECT_EQ(out.child.match.targetBondIdx[0], 0u);
  EXPECT_EQ(out.child.match.targetBondIdx[1], 1u);
  EXPECT_EQ(out.child.match.matchedAtomSize, 3);
  EXPECT_EQ(out.child.match.matchedBondSize, 2);
}

TEST(FMCSUnit, MatchSeedSubstructureFindsPathInsideTriangleWithLeaves) {
  using namespace mcs_fmcs_substructure_test;

  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {0, 2},
    {0, 4},
    {1, 2},
    {1, 3}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 3},
    {1, 2},
    {1, 4},
    {2, 3}
  });
  ManagedMatchTables tables;
  tables.allocate(5, 5, 5, 4);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<SubstructureTestOut> d_out;
  AsyncDeviceVector<std::uint8_t>     partials(2 * kTestSubstructurePartialCapacity * 16);
  matchSubstructureMaskDriver<<<1, 32>>>(qBE.data(),
                                         5,
                                         5,
                                         tBE.data(),
                                         5,
                                         4,
                                         tables.device(),
                                         /*atomMask=*/0x1Fu,
                                         /*bondMask=*/0x1Eu,
                                         partials.data(),
                                         kTestSubstructurePartialCapacity,
                                         d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SubstructureTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_FALSE(out.overflowed);
  EXPECT_EQ(out.child.match.matchedAtomSize, 5);
  EXPECT_EQ(out.child.match.matchedBondSize, 4);
  EXPECT_EQ(out.child.match.targetBondIdx[0], mcs::fmcs::kUnmappedTargetIdx);
  for (int q : {1, 2, 3, 4}) {
    EXPECT_NE(out.child.match.targetBondIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }
}

TEST(FMCSUnit, MatchSeedFallbackRebuildsAfterGreedyFailure) {
  using namespace mcs_fmcs_substructure_test;

  auto               qBE = makeBondEndpointsDevice({
    {0, 1},
    {0, 2},
    {0, 4},
    {1, 2},
    {1, 3}
  });
  auto               tBE = makeBondEndpointsDevice({
    {0, 3},
    {1, 2},
    {1, 4},
    {2, 3}
  });
  ManagedMatchTables tables;
  tables.allocate(5, 5, 5, 4);
  tables.setAllAtomBits();
  tables.setAllBondBits();

  AsyncDevicePtr<SubstructureTestOut> d_out;
  AsyncDeviceVector<std::uint8_t>     partials(2 * kTestSubstructurePartialCapacity * 16);
  matchFallbackBadParentDriver<<<1, 32>>>(qBE.data(),
                                          5,
                                          5,
                                          tBE.data(),
                                          5,
                                          4,
                                          tables.device(),
                                          partials.data(),
                                          kTestSubstructurePartialCapacity,
                                          d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SubstructureTestOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.ok);
  EXPECT_FALSE(out.overflowed);
  EXPECT_EQ(out.child.match.matchedAtomSize, 5);
  EXPECT_EQ(out.child.match.matchedBondSize, 4);
  for (int q = 0; q < 5; ++q) {
    EXPECT_NE(out.child.match.targetAtomIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }
  for (int q : {1, 2, 3, 4}) {
    EXPECT_NE(out.child.match.targetBondIdx[q], mcs::fmcs::kUnmappedTargetIdx);
  }
}
