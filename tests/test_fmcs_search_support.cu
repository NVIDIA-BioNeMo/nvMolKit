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

#include "src/mcs/fmcs_cuda/fmcs_grow.cuh"
#include "src/mcs/fmcs_cuda/fmcs_match.cuh"
#include "src/mcs/fmcs_cuda/fmcs_match_tables.cuh"
#include "src/mcs/fmcs_cuda/fmcs_search_support.cuh"
#include "src/mcs/fmcs_cuda/fmcs_seed.cuh"
#include "src/mcs/fmcs_cuda/fmcs_seed_queue.cuh"
#include "src/utils/device_vector.h"

namespace {

using nvMolKit::AsyncDevicePtr;
using nvMolKit::AsyncDeviceVector;

using mcs::fmcs::MatchResult;
using mcs::fmcs::Seed;

}  // namespace

namespace mcs_fmcs_worklist_test {

using QueuedT = mcs::fmcs::QueuedSeed<16, 16, 16, 16>;

struct WorklistOut {
  int  poppedBonds[3];
  int  finalSize;
  bool overflowRejected;
};

__global__ void sortedWorklistDriver(WorklistOut* out) {
  __shared__ mcs::fmcs::SeedQueue<QueuedT, mcs::fmcs::ThreadBlockScope> queue;
  __shared__ QueuedT                                                    storage[3];
  __shared__ QueuedT                                                    candidate;
  __shared__ QueuedT                                                    popped;

  auto block = cooperative_groups::this_thread_block();
  auto warp  = cooperative_groups::tiled_partition<32>(block);
  if (threadIdx.x == 0) {
    queue.init(storage, 3);
    mcs::fmcs::seedClearWithinThread(candidate.seed);
  }
  __syncthreads();

  for (int bonds : {1, 3, 2}) {
    if (threadIdx.x == 0)
      candidate.seed.numBonds = bonds;
    warp.sync();
    mcs::fmcs::insertSortedByBondsCooperative(warp, queue, candidate);
  }

  if (threadIdx.x == 0)
    candidate.seed.numBonds = 4;
  warp.sync();
  const bool insertedPastCapacity = mcs::fmcs::insertSortedByBondsCooperative(warp, queue, candidate);

  for (int i = 0; i < 3; ++i) {
    mcs::fmcs::popFrontCooperative(warp, queue, popped);
    if (threadIdx.x == 0)
      out->poppedBonds[i] = popped.seed.numBonds;
    warp.sync();
  }
  if (threadIdx.x == 0) {
    out->finalSize        = queue.size();
    out->overflowRejected = !insertedPastCapacity;
  }
}

struct IncumbentOut {
  unsigned int score;
  int          bonds;
  int          atoms;
  int          lock;
};

__global__ void incumbentDriver(IncumbentOut* out) {
  __shared__ QueuedT      best;
  __shared__ QueuedT      candidate;
  __shared__ unsigned int bestScore;
  __shared__ int          lock;

  auto block = cooperative_groups::this_thread_block();
  auto warp  = cooperative_groups::tiled_partition<32>(block);
  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(best.seed);
    mcs::fmcs::seedClearWithinThread(candidate.seed);
    best.seed.numBonds      = 1;
    best.seed.numAtoms      = 4;
    candidate.seed.numBonds = 2;
    candidate.seed.numAtoms = 3;
    bestScore               = (1u << 16) | 4u;
    lock                    = 0;
  }
  __syncthreads();

  mcs::fmcs::updateIncumbentCooperative(warp, candidate, best, &bestScore, &lock);
  if (threadIdx.x == 0) {
    candidate.seed.numBonds = 1;
    candidate.seed.numAtoms = 15;
  }
  warp.sync();
  mcs::fmcs::updateIncumbentCooperative(warp, candidate, best, &bestScore, &lock);

  if (threadIdx.x == 0) {
    out->score = bestScore;
    out->bonds = best.seed.numBonds;
    out->atoms = best.seed.numAtoms;
    out->lock  = lock;
  }
}

}  // namespace mcs_fmcs_worklist_test

TEST(FMCSUnit, SortedWorklistPopsLargestSeedFirstAndRejectsOverflow) {
  using namespace mcs_fmcs_worklist_test;
  AsyncDevicePtr<WorklistOut> deviceOut;
  sortedWorklistDriver<<<1, 32>>>(deviceOut.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  WorklistOut out{};
  ASSERT_EQ(cudaMemcpy(&out, deviceOut.data(), sizeof(out), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(out.poppedBonds[0], 3);
  EXPECT_EQ(out.poppedBonds[1], 2);
  EXPECT_EQ(out.poppedBonds[2], 1);
  EXPECT_EQ(out.finalSize, 0);
  EXPECT_TRUE(out.overflowRejected);
}

TEST(FMCSUnit, IncumbentUsesBondFirstLexicographicScore) {
  using namespace mcs_fmcs_worklist_test;
  AsyncDevicePtr<IncumbentOut> deviceOut;
  incumbentDriver<<<1, 32>>>(deviceOut.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  IncumbentOut out{};
  ASSERT_EQ(cudaMemcpy(&out, deviceOut.data(), sizeof(out), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(out.score, (2u << 16) | 3u);
  EXPECT_EQ(out.bonds, 2);
  EXPECT_EQ(out.atoms, 3);
  EXPECT_EQ(out.lock, 0);
}
