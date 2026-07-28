// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include "src/mcs/fmcs_cuda/fmcs_grow.cuh"
#include "src/utils/device_vector.h"

namespace {

using nvMolKit::AsyncDevicePtr;
using nvMolKit::AsyncDeviceVector;

AsyncDeviceVector<std::uint32_t> makeBondEndpointsDevice(const std::vector<std::pair<int, int>>& edges) {
  std::vector<std::uint32_t> host(edges.size());
  for (std::size_t i = 0; i < edges.size(); ++i) {
    host[i] = (static_cast<std::uint32_t>(edges[i].first) << 16) | static_cast<std::uint32_t>(edges[i].second);
  }
  AsyncDeviceVector<std::uint32_t> device(edges.size());
  device.copyFromHost(host);
  return device;
}

}  // namespace

// ---------------------------------------------------------------------------
// fillNewBondsCooperative / pruneIndividualBondsCooperative
// ---------------------------------------------------------------------------

namespace mcs_fmcs_grow_test {

using mcs::fmcs::NewBond;
using SeedT   = mcs::fmcs::Seed<16, 16>;
using QueuedT = mcs::fmcs::QueuedSeed<16, 16, 16, 16>;

constexpr int kMaxNewBonds = 8;

struct FillNewBondsOut {
  NewBond bonds[kMaxNewBonds];
  int     count;
  bool    ok;
};

// Driver: caller-provided seed-setup function fills the seed before
// calling fillNewBondsCooperative.  Templated on the setup so each
// test can hand-construct its own scenario.
template <class SeedSetup>
__device__ __forceinline__ void fillNewBondsRun(SeedSetup&&          setup,
                                                const std::uint32_t* qBondEndpoints,
                                                int                  qNumAtoms,
                                                int                  qNumBonds,
                                                int                  maxNewBonds,
                                                FillNewBondsOut*     out) {
  __shared__ SeedT   seed;
  __shared__ NewBond bonds[kMaxNewBonds];
  __shared__ int     count;

  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(seed);
    setup(seed);
  }
  __syncthreads();

  auto                     block = cooperative_groups::this_thread_block();
  auto                     warp  = cooperative_groups::tiled_partition<32>(block);
  // fillNewBondsCooperative reads only the bond-endpoint side of the
  // topology, so the CSR arrays stay null here.
  mcs::fmcs::DeviceCsrView qView{};
  qView.bondEndpoints = qBondEndpoints;
  qView.numAtoms      = qNumAtoms;
  qView.numBonds      = qNumBonds;
  bool ok             = mcs::fmcs::fillNewBondsCooperative(warp, seed, qView, bonds, &count, maxNewBonds);
  __syncthreads();

  if (threadIdx.x == 0) {
    out->ok    = ok;
    out->count = count;
    for (int i = 0; i < kMaxNewBonds; ++i)
      out->bonds[i] = bonds[i];
  }
}

// Test 1: 3-atom path query (atoms 0-1-2, bonds (0,1), (1,2)).  Seed
// holds just atom 1, marked last-added.  Both bonds touch atom 1;
// each should become an atom-adding NewBond.
__global__ void fillNewBondsAtomAddingDriver(const std::uint32_t* qBE,
                                             int                  qNumAtoms,
                                             int                  qNumBonds,
                                             FillNewBondsOut*     out) {
  fillNewBondsRun([] __device__(SeedT & seed) { mcs::fmcs::seedAddAtomWithinThread(seed, 1); },
                  qBE,
                  qNumAtoms,
                  qNumBonds,
                  kMaxNewBonds,
                  out);
}

// Test 2: same query, seed has bond 0 in excludedBonds.  Only bond 1
// should appear in the output.
__global__ void fillNewBondsExcludedSkippedDriver(const std::uint32_t* qBE,
                                                  int                  qNumAtoms,
                                                  int                  qNumBonds,
                                                  FillNewBondsOut*     out) {
  fillNewBondsRun(
    [] __device__(SeedT & seed) {
      mcs::fmcs::seedAddAtomWithinThread(seed, 1);
      // Mark bond 0 as excluded but DO NOT add it to seed.bonds.
      // The exclusion alone is what fillNewBonds checks.
      using BondWord     = SeedT::bond_word_type;
      constexpr int kBPW = SeedT::kBondBitsPerWord;
      seed.excludedBonds[0 / kBPW] |= static_cast<BondWord>(1) << (0 % kBPW);
    },
    qBE,
    qNumAtoms,
    qNumBonds,
    kMaxNewBonds,
    out);
}

// Test 3: 3-atom triangle query (atoms 0,1,2; bonds (0,1), (1,2), (0,2)).
// Seed holds atoms 0 and 1 (both last-added) plus bond (0,1).  Bond 2
// = (0,2) is atom-adding (atom 2 not in seed).  More importantly, this
// test verifies the seed.atoms vs seed.lastAddedAtoms split: if we
// only mark atom 1 as "last added" (atom 0 stays in seed.atoms but
// NOT in lastAddedAtoms), bond (0,2) should NOT be reported because
// neither endpoint is "newly added".  Bond (1,2) SHOULD be reported.
__global__ void fillNewBondsLastAddedFilteringDriver(const std::uint32_t* qBE,
                                                     int                  qNumAtoms,
                                                     int                  qNumBonds,
                                                     FillNewBondsOut*     out) {
  fillNewBondsRun(
    [] __device__(SeedT & seed) {
      // Atom 0 is "old" (in seed but not last-added).
      seed.atoms[0] |= 1u << 0;
      seed.numAtoms = 1;
      // Begin a grow step boundary.
      mcs::fmcs::seedBeginGrowStepWithinThread(seed);
      // Atom 1 is "newly added".
      mcs::fmcs::seedAddAtomWithinThread(seed, 1);
      // Bond 0 already mapped (in seed and excluded).
      mcs::fmcs::seedAddBondWithinThread(seed, 0);
    },
    qBE,
    qNumAtoms,
    qNumBonds,
    kMaxNewBonds,
    out);
}

// Test 4: ring-closing.  Seed has atoms 0, 1, 2 all marked last-added,
// bond 0 = (0,1) and bond 1 = (1,2) already in excludedBonds.  Bond 2
// = (0,2) closes the ring: both endpoints in seed.
__global__ void fillNewBondsRingClosingDriver(const std::uint32_t* qBE,
                                              int                  qNumAtoms,
                                              int                  qNumBonds,
                                              FillNewBondsOut*     out) {
  fillNewBondsRun(
    [] __device__(SeedT & seed) {
      mcs::fmcs::seedAddAtomWithinThread(seed, 0);
      mcs::fmcs::seedAddAtomWithinThread(seed, 1);
      mcs::fmcs::seedAddAtomWithinThread(seed, 2);
      mcs::fmcs::seedAddBondWithinThread(seed, 0);
      mcs::fmcs::seedAddBondWithinThread(seed, 1);
      // Bond 2 = (0,2) is the only candidate; it's ring-closing
      // since both 0 and 2 are in seed.atoms.
    },
    qBE,
    qNumAtoms,
    qNumBonds,
    kMaxNewBonds,
    out);
}

// Test 5: overflow.  4-atom path with seed = {atom 0 newly-added}.
// Query bonds 0..3 all touch atom 0 (star graph: bonds (0,1) (0,2)
// (0,3) (0,4)).  maxNewBonds = 2 -> first 2 win the race, function
// returns false, count clamped at 2.
__global__ void fillNewBondsOverflowDriver(const std::uint32_t* qBE,
                                           int                  qNumAtoms,
                                           int                  qNumBonds,
                                           FillNewBondsOut*     out) {
  fillNewBondsRun([] __device__(SeedT & seed) { mcs::fmcs::seedAddAtomWithinThread(seed, 0); },
                  qBE,
                  qNumAtoms,
                  qNumBonds,
                  /*maxNewBonds=*/2,
                  out);
}

}  // namespace mcs_fmcs_grow_test

TEST(FMCSUnit, FillNewBondsAtomAddingFromBoundary) {
  using namespace mcs_fmcs_grow_test;
  // Path 0-1-2: bonds (0,1), (1,2).
  auto                            qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2}
  });
  AsyncDevicePtr<FillNewBondsOut> d_out;

  fillNewBondsAtomAddingDriver<<<1, 32>>>(qBE.data(), 3, 2, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  FillNewBondsOut out{};
  ASSERT_EQ(cudaMemcpy(&out, d_out.data(), sizeof(out), cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_TRUE(out.ok);
  EXPECT_EQ(out.count, 2);
  // Both bonds appeared; race-order is non-deterministic so collect
  // by bondIdx.
  std::set<int> seenBonds;
  for (int i = 0; i < out.count; ++i) {
    seenBonds.insert(out.bonds[i].bondIdx);
    EXPECT_EQ(out.bonds[i].endAtomSeedIdx, NewBond::kNotInSeed);
    EXPECT_TRUE(out.bonds[i].alive);
  }
  EXPECT_TRUE(seenBonds.count(0));
  EXPECT_TRUE(seenBonds.count(1));
}

TEST(FMCSUnit, FillNewBondsExcludedSkipped) {
  using namespace mcs_fmcs_grow_test;
  auto                            qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2}
  });
  AsyncDevicePtr<FillNewBondsOut> d_out;

  fillNewBondsExcludedSkippedDriver<<<1, 32>>>(qBE.data(), 3, 2, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  FillNewBondsOut out{};
  ASSERT_EQ(cudaMemcpy(&out, d_out.data(), sizeof(out), cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_TRUE(out.ok);
  EXPECT_EQ(out.count, 1);
  EXPECT_EQ(out.bonds[0].bondIdx, 1);  // bond 0 excluded; only bond 1 left
}

TEST(FMCSUnit, FillNewBondsLastAddedFiltering) {
  using namespace mcs_fmcs_grow_test;
  // Triangle: bonds 0=(0,1), 1=(1,2), 2=(0,2).
  auto                            qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {0, 2}
  });
  AsyncDevicePtr<FillNewBondsOut> d_out;

  fillNewBondsLastAddedFilteringDriver<<<1, 32>>>(qBE.data(), 3, 3, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  FillNewBondsOut out{};
  ASSERT_EQ(cudaMemcpy(&out, d_out.data(), sizeof(out), cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_TRUE(out.ok);
  // Bond 0 is excluded.  Bond 1 = (1,2) touches newly-added atom 1.
  // Bond 2 = (0,2) touches atom 0 (NOT newly-added) and atom 2 (also
  // not in seed) -> neither endpoint newly-added, must be skipped.
  EXPECT_EQ(out.count, 1);
  EXPECT_EQ(out.bonds[0].bondIdx, 1);
  EXPECT_EQ(out.bonds[0].endAtomSeedIdx, NewBond::kNotInSeed);
  EXPECT_EQ(out.bonds[0].newAtomIdx, 2);  // atom 2 is the unmapped end
}

TEST(FMCSUnit, FillNewBondsRingClosing) {
  using namespace mcs_fmcs_grow_test;
  // Triangle: bonds 0=(0,1), 1=(1,2), 2=(0,2).
  auto                            qBE = makeBondEndpointsDevice({
    {0, 1},
    {1, 2},
    {0, 2}
  });
  AsyncDevicePtr<FillNewBondsOut> d_out;

  fillNewBondsRingClosingDriver<<<1, 32>>>(qBE.data(), 3, 3, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  FillNewBondsOut out{};
  ASSERT_EQ(cudaMemcpy(&out, d_out.data(), sizeof(out), cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_TRUE(out.ok);
  // Only bond 2 = (0,2) is candidate (others are excluded).  Both
  // endpoints in seed -> ring-closing.
  EXPECT_EQ(out.count, 1);
  EXPECT_EQ(out.bonds[0].bondIdx, 2);
  EXPECT_NE(out.bonds[0].endAtomSeedIdx, NewBond::kNotInSeed);
}

TEST(FMCSUnit, FillNewBondsOverflowReturnsFalse) {
  using namespace mcs_fmcs_grow_test;
  // Star graph: 4 bonds from atom 0 -> 1, 2, 3, 4.  numAtoms=5.
  auto                            qBE = makeBondEndpointsDevice({
    {0, 1},
    {0, 2},
    {0, 3},
    {0, 4}
  });
  AsyncDevicePtr<FillNewBondsOut> d_out;

  fillNewBondsOverflowDriver<<<1, 32>>>(qBE.data(), 5, 4, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  FillNewBondsOut out{};
  ASSERT_EQ(cudaMemcpy(&out, d_out.data(), sizeof(out), cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_FALSE(out.ok);
  // count is clamped to maxNewBonds (=2).
  EXPECT_EQ(out.count, 2);
}

// ---- pruneIndividualBondsCooperative ----

namespace mcs_fmcs_prune_test {

using mcs::fmcs::NewBond;
using mcs_fmcs_grow_test::QueuedT;
using mcs_fmcs_grow_test::SeedT;

constexpr int kMaxNewBondsPrune = 8;

struct PruneOut {
  // Identity of children that survived matchFn and reached childSink.
  // We record the child seed's bondIdx-of-the-newly-added bond (the
  // first set bit beyond the parent's bond bitset).  That uniquely
  // identifies which NewBond the child was built from.
  int  survivorBondIdx[kMaxNewBondsPrune];
  bool bondAlive[kMaxNewBondsPrune];
  int  numSurvivors;
  int  numMatchAttempts;
};

__global__ void pruneStage1Driver(const NewBond* hostBonds, int nBonds, PruneOut* out) {
  __shared__ std::uint64_t parentStorage[(sizeof(QueuedT) + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t)];
  __shared__ std::uint64_t workspaceStorage[(sizeof(QueuedT) + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t)];
  __shared__ NewBond       bonds[kMaxNewBondsPrune];
  __shared__ int           matchAttempts;
  __shared__ int           survivorIdx;
  QueuedT&                 parent    = *reinterpret_cast<QueuedT*>(parentStorage);
  QueuedT&                 workspace = *reinterpret_cast<QueuedT*>(workspaceStorage);

  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(parent.seed);
    mcs::fmcs::matchResultClearWithinThread(parent.match);
    // Parent has bond 0 already in its bitset (so pruneIndividualBonds
    // appears as "extending past bond 0" for tracking).
    mcs::fmcs::seedAddAtomWithinThread(parent.seed, 0);
    mcs::fmcs::seedAddAtomWithinThread(parent.seed, 1);
    mcs::fmcs::seedAddBondWithinThread(parent.seed, 0);
    for (int i = 0; i < nBonds; ++i)
      bonds[i] = hostBonds[i];
    matchAttempts = 0;
    survivorIdx   = 0;
    for (int i = 0; i < kMaxNewBondsPrune; ++i) {
      out->survivorBondIdx[i] = -1;
    }
  }
  __syncthreads();

  auto block = cooperative_groups::this_thread_block();
  auto warp  = cooperative_groups::tiled_partition<32>(block);

  // Stub matchFn: returns true iff the new bond's bondIdx is even.
  // Counts attempts so the test can verify every bond was tried.
  auto matchFn = [&] __device__(QueuedT & child) -> bool {
    if (warp.thread_rank() == 0)
      ++matchAttempts;
    warp.sync();
    // Find which bondIdx the workspace's seed has beyond the parent's.
    // Parent has only bond 0 set; child has one more.
    int childBond = -1;
    for (int b = 0; b < 16; ++b) {
      using BondWord            = SeedT::bond_word_type;
      const BondWord childWord  = child.seed.bonds[0];
      const BondWord parentWord = parent.seed.bonds[0];
      const BondWord newBits    = childWord & ~parentWord;
      if ((newBits >> b) & 1) {
        childBond = b;
        break;
      }
    }
    return (childBond % 2) == 0;
  };
  auto childSink = [&] __device__(QueuedT & child) {
    if (warp.thread_rank() == 0) {
      // Record child's new-bond identity.
      using BondWord           = SeedT::bond_word_type;
      const BondWord newBits   = child.seed.bonds[0] & ~parent.seed.bonds[0];
      int            childBond = -1;
      for (int b = 0; b < 16; ++b) {
        if ((newBits >> b) & 1) {
          childBond = b;
          break;
        }
      }
      out->survivorBondIdx[survivorIdx++] = childBond;
    }
    warp.sync();
  };
  mcs::fmcs::pruneIndividualBondsCooperative(warp, parent, workspace, bonds, nBonds, matchFn, childSink);

  __syncthreads();
  if (threadIdx.x == 0) {
    out->numMatchAttempts = matchAttempts;
    out->numSurvivors     = survivorIdx;
    for (int i = 0; i < nBonds; ++i)
      out->bondAlive[i] = bonds[i].alive;
  }
}

}  // namespace mcs_fmcs_prune_test

TEST(FMCSUnit, PruneIndividualBondsStage1OnlySurvivorsSinked) {
  using namespace mcs_fmcs_prune_test;
  // Five candidate bonds with bondIdx 1..5. Bond 3 is already dead and
  // must be skipped without a match attempt. Stub matchFn passes even
  // bondIdx (-> 2, 4 succeed; 1, 5 fail). The failed live bonds must
  // also be marked dead for Stage 2.
  const std::vector<NewBond> hostBonds = {
    NewBond{1, 5, NewBond::kNotInSeed,  true},
    NewBond{2, 6, NewBond::kNotInSeed,  true},
    NewBond{3, 7, NewBond::kNotInSeed, false},
    NewBond{4, 8, NewBond::kNotInSeed,  true},
    NewBond{5, 9, NewBond::kNotInSeed,  true},
  };
  AsyncDeviceVector<NewBond> d_bonds(hostBonds.size());
  d_bonds.copyFromHost(hostBonds);

  AsyncDevicePtr<PruneOut> d_out;
  pruneStage1Driver<<<1, 32>>>(d_bonds.data(), 5, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  PruneOut out{};
  ASSERT_EQ(cudaMemcpy(&out, d_out.data(), sizeof(out), cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_EQ(out.numMatchAttempts, 4);  // pre-dead bond 3 was skipped
  EXPECT_EQ(out.numSurvivors, 2);
  std::set<int> survivors{out.survivorBondIdx[0], out.survivorBondIdx[1]};
  EXPECT_TRUE(survivors.count(2));
  EXPECT_TRUE(survivors.count(4));
  EXPECT_FALSE(survivors.count(1));
  EXPECT_FALSE(survivors.count(3));
  EXPECT_FALSE(survivors.count(5));
  EXPECT_FALSE(out.bondAlive[0]);  // singleton match failed
  EXPECT_TRUE(out.bondAlive[1]);   // singleton match survived
  EXPECT_FALSE(out.bondAlive[2]);  // already dead and remained skipped
  EXPECT_TRUE(out.bondAlive[3]);   // singleton match survived
  EXPECT_FALSE(out.bondAlive[4]);  // singleton match failed
}

// ---------------------------------------------------------------------------
