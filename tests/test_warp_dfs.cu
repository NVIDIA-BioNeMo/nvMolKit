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

// Tests nvMolKit::dfsFromRoots against a host brute-force enumerator on
// synthetic graphs, using a minimal oracle: candidates(d) = compat[d] & ~used
// & AND over back edges of adj[mapping[e]]. This exercises the core's stack
// discipline, terminal verdicts, abort hook, and every TargetMask width --
// including the 128-atom form's lo/hi word boundary -- without any molecule
// plumbing; the substructure integration tests cover the production oracle.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "src/subgraph/target_mask.cuh"
#include "src/subgraph/warp_dfs.cuh"
#include "src/utils/cuda_error_check.h"

using nvMolKit::checkReturnCode;

namespace {

constexpr int kHostMaxTargetAtoms = 128;

/// A (target graph, query constraints) pair small enough to pass as a kernel
/// argument. Target-atom sets are stored as (lo, hi) 64-bit word pairs; hi is
/// unused below 65 atoms. adj[t] is target atom t's neighbour set; compat[d]
/// the target atoms query atom d may map to; backEdges[d] the earlier query
/// atoms d is bonded to. Every edge accepts every bond (bond predicates are
/// the oracle's business, not the core's).
template <std::size_t MaxTargetAtoms, int MaxDepth> struct TestProblem {
  uint64_t      adjLo[MaxTargetAtoms]    = {};
  uint64_t      adjHi[MaxTargetAtoms]    = {};
  uint64_t      compatLo[MaxDepth]       = {};
  uint64_t      compatHi[MaxDepth]       = {};
  unsigned char backEdges[MaxDepth][4]   = {};
  unsigned char backEdgeCounts[MaxDepth] = {};
  int           numQueryAtoms            = 0;
  int           numTargetAtoms           = 0;
};

enum class TestMode {
  CountAll,    ///< Every lane counts all embeddings from its roots.
  FirstOnly,   ///< All roots on lane 0; stop the lane at the first embedding and record it.
  RootExists,  ///< Flag each root that starts at least one embedding, then abandon it.
  AbortAll,    ///< Abort hook returns true from the start; nothing may be found.
  ExistsRace   ///< All lanes race for one embedding via a found flag: the MCS existence pattern.
};

template <std::size_t MaxTargetAtoms>
__device__ nvMolKit::TargetMask<MaxTargetAtoms> maskFromWords(uint64_t lo, uint64_t hi) {
  nvMolKit::TargetMask<MaxTargetAtoms> mask;
  mask.clear();
  mask.setWord32(0, static_cast<uint32_t>(lo));
  if constexpr (MaxTargetAtoms >= 64) {
    mask.setWord32(1, static_cast<uint32_t>(lo >> 32));
  }
  if constexpr (MaxTargetAtoms >= 128) {
    mask.setWord32(2, static_cast<uint32_t>(hi));
    mask.setWord32(3, static_cast<uint32_t>(hi >> 32));
  }
  return mask;
}

template <std::size_t MaxTargetAtoms, int MaxDepth>
__global__ void warpDfsTestKernel(TestProblem<MaxTargetAtoms, MaxDepth> problem,
                                  TestMode                              mode,
                                  int*                                  outCount,
                                  int*                                  outMapping,
                                  int*                                  outRootFlags,
                                  int*                                  foundFlag) {
  const int lane = static_cast<int>(threadIdx.x);
  using Mask     = nvMolKit::TargetMask<MaxTargetAtoms>;

  const Mask compatRoots = maskFromWords<MaxTargetAtoms>(problem.compatLo[0], problem.compatHi[0]);

  Mask roots;
  roots.clear();
  if (mode == TestMode::FirstOnly) {
    if (lane == 0) {
      roots = compatRoots;
    }
  } else {
    // Production distribution: lane L owns target atoms L, L+32, L+64, L+96.
    Mask laneAtoms;
    laneAtoms.clear();
#pragma unroll
    for (int k = 0; k < static_cast<int>(MaxTargetAtoms / 32); ++k) {
      laneAtoms.set(lane + 32 * k);
    }
    roots = compatRoots;
    roots.andEq(laneAtoms);
  }

  auto candidatesAt = [&](int depth, const unsigned char* mapping, const Mask& used, int /*prevTargetAtom*/) -> Mask {
    Mask candidates = maskFromWords<MaxTargetAtoms>(problem.compatLo[depth], problem.compatHi[depth]);
    candidates.andNotEq(used);
    for (int k = 0; k < problem.backEdgeCounts[depth]; ++k) {
      const int mapped = mapping[problem.backEdges[depth][k]];
      candidates.andEq(maskFromWords<MaxTargetAtoms>(problem.adjLo[mapped], problem.adjHi[mapped]));
    }
    return candidates;
  };

  auto onTerminal = [&](Mask terminals, const unsigned char* mapping) -> nvMolKit::DfsTerminalVerdict {
    nvMolKit::DfsTerminalVerdict verdict{false, false};
    if (mode == TestMode::CountAll || mode == TestMode::AbortAll) {
      atomicAdd(outCount, terminals.popcount());
    } else if (mode == TestMode::FirstOnly) {
      if (!terminals.empty()) {
        for (int d = 0; d < problem.numQueryAtoms - 1; ++d) {
          outMapping[d] = mapping[d];
        }
        outMapping[problem.numQueryAtoms - 1] = terminals.lowest();
        atomicAdd(outCount, 1);
        verdict.laneDone = true;
      }
    } else if (mode == TestMode::RootExists) {
      if (!terminals.empty()) {
        outRootFlags[mapping[0]] = 1;
        verdict.rootDone         = true;
      }
    } else {  // ExistsRace
      if (!terminals.empty()) {
        if (atomicCAS(foundFlag, 0, 1) == 0) {
          // This lane won: it alone records the embedding.
          for (int d = 0; d < problem.numQueryAtoms - 1; ++d) {
            outMapping[d] = mapping[d];
          }
          outMapping[problem.numQueryAtoms - 1] = terminals.lowest();
          atomicAdd(outCount, 1);
        }
        verdict.laneDone = true;
      }
    }
    return verdict;
  };

  auto abortRequested = [&]() -> bool {
    if (mode == TestMode::AbortAll) {
      return true;
    }
    if (mode == TestMode::ExistsRace) {
      return atomicAdd(foundFlag, 0) != 0;
    }
    return false;
  };

  nvMolKit::dfsFromRoots<MaxDepth>(roots, problem.numQueryAtoms - 1, candidatesAt, onTerminal, abortRequested);
}

// =============================================================================
// Host-side graph construction and brute force
// =============================================================================

struct WordPair {
  uint64_t lo = 0;
  uint64_t hi = 0;
};

void setBit(WordPair& words, int bit) {
  if (bit < 64) {
    words.lo |= 1ULL << bit;
  } else {
    words.hi |= 1ULL << (bit - 64);
  }
}

bool testBit(const WordPair& words, int bit) {
  return bit < 64 ? ((words.lo >> bit) & 1ULL) != 0 : ((words.hi >> (bit - 64)) & 1ULL) != 0;
}

/// Size-agnostic host mirror of TestProblem, converted per instantiation.
struct HostProblem {
  WordPair                      adj[kHostMaxTargetAtoms] = {};
  std::vector<WordPair>         compat;
  std::vector<std::vector<int>> backEdges;
  int                           numQueryAtoms  = 0;
  int                           numTargetAtoms = 0;

  void init(int numTarget, int numQuery) {
    numTargetAtoms = numTarget;
    numQueryAtoms  = numQuery;
    compat.assign(numQuery, {});
    backEdges.assign(numQuery, {});
  }

  void addUndirectedEdge(int a, int b) {
    setBit(adj[a], b);
    setBit(adj[b], a);
  }

  void allowAllLabels() {
    for (int d = 0; d < numQueryAtoms; ++d) {
      for (int t = 0; t < numTargetAtoms; ++t) {
        setBit(compat[d], t);
      }
    }
  }

  /// Chain query: each atom bonded to the previous one.
  void makePathQuery() {
    for (int d = 1; d < numQueryAtoms; ++d) {
      backEdges[d] = {d - 1};
    }
  }
};

template <std::size_t MaxTargetAtoms, int MaxDepth>
TestProblem<MaxTargetAtoms, MaxDepth> toDeviceProblem(const HostProblem& host) {
  TestProblem<MaxTargetAtoms, MaxDepth> device;
  device.numTargetAtoms = host.numTargetAtoms;
  device.numQueryAtoms  = host.numQueryAtoms;
  for (int t = 0; t < host.numTargetAtoms; ++t) {
    device.adjLo[t] = host.adj[t].lo;
    device.adjHi[t] = host.adj[t].hi;
  }
  for (int d = 0; d < host.numQueryAtoms; ++d) {
    device.compatLo[d] = host.compat[d].lo;
    device.compatHi[d] = host.compat[d].hi;
    for (std::size_t k = 0; k < host.backEdges[d].size(); ++k) {
      device.backEdges[d][k] = static_cast<unsigned char>(host.backEdges[d][k]);
    }
    device.backEdgeCounts[d] = static_cast<unsigned char>(host.backEdges[d].size());
  }
  return device;
}

// Host-side mirror of the kernel's rules, enumerating all injective embeddings.
void bruteForceRecurse(const HostProblem& problem, std::vector<int>& mapping, WordPair used, int depth, long& count) {
  if (depth == problem.numQueryAtoms) {
    ++count;
    return;
  }
  for (int t = 0; t < problem.numTargetAtoms; ++t) {
    if (!testBit(problem.compat[depth], t) || testBit(used, t)) {
      continue;
    }
    bool edgesOk = true;
    for (const int e : problem.backEdges[depth]) {
      if (!testBit(problem.adj[mapping[e]], t)) {
        edgesOk = false;
        break;
      }
    }
    if (!edgesOk) {
      continue;
    }
    mapping[depth]    = t;
    WordPair nextUsed = used;
    setBit(nextUsed, t);
    bruteForceRecurse(problem, mapping, nextUsed, depth + 1, count);
  }
}

long bruteForceCount(const HostProblem& problem) {
  long             count = 0;
  std::vector<int> mapping(problem.numQueryAtoms, -1);
  bruteForceRecurse(problem, mapping, {}, 0, count);
  return count;
}

bool rootCanEmbed(const HostProblem& problem, int root) {
  if (!testBit(problem.compat[0], root)) {
    return false;
  }
  HostProblem restricted = problem;
  restricted.compat[0]   = {};
  setBit(restricted.compat[0], root);
  return bruteForceCount(restricted) > 0;
}

/// Cyclohexane-like ring 0-1-2, closing 2-0, plus tail edge 2-3.
HostProblem triangleWithTailTarget(int numQueryAtoms) {
  HostProblem problem;
  problem.init(4, numQueryAtoms);
  problem.addUndirectedEdge(0, 1);
  problem.addUndirectedEdge(0, 2);
  problem.addUndirectedEdge(1, 2);
  problem.addUndirectedEdge(2, 3);
  problem.allowAllLabels();
  return problem;
}

/// Cycle over target atoms firstAtom..firstAtom+cycleLength-1 inside a
/// numTargetAtoms-atom graph whose other atoms are isolated.
HostProblem cycleTarget(int numTargetAtoms, int firstAtom, int cycleLength, int numQueryAtoms) {
  HostProblem problem;
  problem.init(numTargetAtoms, numQueryAtoms);
  for (int i = 0; i < cycleLength; ++i) {
    problem.addUndirectedEdge(firstAtom + i, firstAtom + (i + 1) % cycleLength);
  }
  problem.allowAllLabels();
  return problem;
}

// =============================================================================
// Test fixture
// =============================================================================

constexpr int kMaxRecordedDepth = 48;

class WarpDfsTest : public ::testing::Test {
 protected:
  template <std::size_t MaxTargetAtoms, int MaxDepth> void run(const HostProblem& host, TestMode mode) {
    ASSERT_LE(host.numTargetAtoms, static_cast<int>(MaxTargetAtoms));
    ASSERT_LE(host.numQueryAtoms, MaxDepth);
    ASSERT_LE(host.numQueryAtoms, kMaxRecordedDepth);
    cudaCheckError(cudaMemset(devCount_, 0, sizeof(int)));
    cudaCheckError(cudaMemset(devMapping_, 0xFF, kMaxRecordedDepth * sizeof(int)));
    cudaCheckError(cudaMemset(devRootFlags_, 0, kHostMaxTargetAtoms * sizeof(int)));
    cudaCheckError(cudaMemset(devFoundFlag_, 0, sizeof(int)));
    const auto problem = toDeviceProblem<MaxTargetAtoms, MaxDepth>(host);
    warpDfsTestKernel<MaxTargetAtoms, MaxDepth>
      <<<1, 32>>>(problem, mode, devCount_, devMapping_, devRootFlags_, devFoundFlag_);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaMemcpy(&count_, devCount_, sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(mapping_, devMapping_, kMaxRecordedDepth * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(rootFlags_, devRootFlags_, kHostMaxTargetAtoms * sizeof(int), cudaMemcpyDeviceToHost));
  }

  void expectValidEmbedding(const HostProblem& problem) {
    for (int d = 0; d < problem.numQueryAtoms; ++d) {
      ASSERT_GE(mapping_[d], 0);
      ASSERT_LT(mapping_[d], problem.numTargetAtoms);
      EXPECT_TRUE(testBit(problem.compat[d], mapping_[d]));
      for (int e = 0; e < d; ++e) {
        EXPECT_NE(mapping_[d], mapping_[e]);
      }
      for (const int k : problem.backEdges[d]) {
        EXPECT_TRUE(testBit(problem.adj[mapping_[k]], mapping_[d]));
      }
    }
  }

  void SetUp() override {
    cudaCheckError(cudaMalloc(&devCount_, sizeof(int)));
    cudaCheckError(cudaMalloc(&devMapping_, kMaxRecordedDepth * sizeof(int)));
    cudaCheckError(cudaMalloc(&devRootFlags_, kHostMaxTargetAtoms * sizeof(int)));
    cudaCheckError(cudaMalloc(&devFoundFlag_, sizeof(int)));
  }

  void TearDown() override {
    cudaFree(devCount_);
    cudaFree(devMapping_);
    cudaFree(devRootFlags_);
    cudaFree(devFoundFlag_);
  }

  int  count_                          = 0;
  int  mapping_[kMaxRecordedDepth]     = {};
  int  rootFlags_[kHostMaxTargetAtoms] = {};
  int* devCount_                       = nullptr;
  int* devMapping_                     = nullptr;
  int* devRootFlags_                   = nullptr;
  int* devFoundFlag_                   = nullptr;
};

// =============================================================================
// 32-atom form
// =============================================================================

TEST_F(WarpDfsTest, CountsPathEmbeddings) {
  // Query: path 0-1-2. In triangle+tail: sum over middle atoms of
  // degree*(degree-1) = 2 + 2 + 6 + 0 = 10 ordered embeddings.
  HostProblem problem = triangleWithTailTarget(3);
  problem.makePathQuery();

  run<32, 8>(problem, TestMode::CountAll);
  EXPECT_EQ(count_, 10);
  EXPECT_EQ(bruteForceCount(problem), 10);
}

TEST_F(WarpDfsTest, CountsTriangleEmbeddings) {
  // Query: triangle. Only the target triangle matches, in all 3! orders.
  HostProblem problem = triangleWithTailTarget(3);
  problem.makePathQuery();
  problem.backEdges[2] = {0, 1};

  run<32, 8>(problem, TestMode::CountAll);
  EXPECT_EQ(count_, 6);
  EXPECT_EQ(bruteForceCount(problem), 6);
}

TEST_F(WarpDfsTest, RespectsLabelCompatibility) {
  // Path query whose first atom may only map to target atoms 2 or 3.
  HostProblem problem = triangleWithTailTarget(3);
  problem.makePathQuery();
  problem.compat[0] = {};
  setBit(problem.compat[0], 2);
  setBit(problem.compat[0], 3);

  run<32, 8>(problem, TestMode::CountAll);
  EXPECT_EQ(static_cast<long>(count_), bruteForceCount(problem));
}

TEST_F(WarpDfsTest, RingClosureMatchesBruteForce) {
  // 4-atom query with a ring closure (cycle 0-1-2-3-0 plus chord 0-2) forces
  // backtracking through several depths; validate against brute force.
  HostProblem problem;
  problem.init(6, 4);
  problem.addUndirectedEdge(0, 1);
  problem.addUndirectedEdge(1, 2);
  problem.addUndirectedEdge(2, 3);
  problem.addUndirectedEdge(3, 0);
  problem.addUndirectedEdge(0, 2);
  problem.addUndirectedEdge(3, 4);
  problem.addUndirectedEdge(4, 5);
  problem.allowAllLabels();
  problem.makePathQuery();
  problem.backEdges[3] = {2, 0};

  run<32, 8>(problem, TestMode::CountAll);
  const long expected = bruteForceCount(problem);
  EXPECT_GT(expected, 0);
  EXPECT_EQ(static_cast<long>(count_), expected);
}

TEST_F(WarpDfsTest, LaneDoneStopsAfterFirstEmbedding) {
  HostProblem problem = triangleWithTailTarget(3);
  problem.makePathQuery();

  run<32, 8>(problem, TestMode::FirstOnly);
  EXPECT_EQ(count_, 1);
  expectValidEmbedding(problem);
}

TEST_F(WarpDfsTest, RootDoneFlagsExactlyTheRootsThatEmbed) {
  // Triangle query: target atoms 0,1,2 can each start an embedding; the tail
  // atom 3 cannot.
  HostProblem problem = triangleWithTailTarget(3);
  problem.makePathQuery();
  problem.backEdges[2] = {0, 1};

  run<32, 8>(problem, TestMode::RootExists);
  EXPECT_EQ(rootFlags_[0], 1);
  EXPECT_EQ(rootFlags_[1], 1);
  EXPECT_EQ(rootFlags_[2], 1);
  EXPECT_EQ(rootFlags_[3], 0);
}

TEST_F(WarpDfsTest, AbortHookStopsBeforeAnyRoot) {
  HostProblem problem = triangleWithTailTarget(3);
  problem.makePathQuery();

  run<32, 8>(problem, TestMode::AbortAll);
  EXPECT_EQ(count_, 0);
}

// =============================================================================
// 64- and 128-atom forms, multi-root lanes, deep stacks
// =============================================================================

TEST_F(WarpDfsTest, Cycle64CountsAcrossBothMaskWords) {
  // 64-cycle, 5-atom path query: every atom starts one path per direction, so
  // 128 ordered embeddings, half of them rooted in the mask's upper word. Two
  // roots per lane exercises the root loop past its first root.
  HostProblem problem = cycleTarget(64, 0, 64, 5);
  problem.makePathQuery();

  run<64, 8>(problem, TestMode::CountAll);
  EXPECT_EQ(count_, 128);
  EXPECT_EQ(bruteForceCount(problem), 128);
}

TEST_F(WarpDfsTest, Cycle64RootDoneAdvancesToSecondRootOfLane) {
  // Every atom of the 64-cycle can root an embedding. RootExists abandons a
  // root at its first hit, so flagging all 64 proves each lane continued to
  // its second root after rootDone on the first.
  HostProblem problem = cycleTarget(64, 0, 64, 5);
  problem.makePathQuery();

  run<64, 8>(problem, TestMode::RootExists);
  for (int t = 0; t < 64; ++t) {
    EXPECT_EQ(rootFlags_[t], 1) << "root " << t;
  }
}

TEST_F(WarpDfsTest, Ring128StraddlesWordBoundary) {
  // 12-cycle on atoms 58..69 of a 128-atom target: the ring's edges, roots,
  // and used-set updates all cross the TargetMask<128> lo/hi boundary. The
  // remaining 116 atoms are isolated, so a 4-path query embeds only in the
  // ring: 12 starts x 2 directions = 24.
  HostProblem problem = cycleTarget(128, 58, 12, 4);
  problem.makePathQuery();

  run<128, 8>(problem, TestMode::CountAll);
  EXPECT_EQ(count_, 24);
  EXPECT_EQ(bruteForceCount(problem), 24);

  run<128, 8>(problem, TestMode::RootExists);
  for (int t = 0; t < 128; ++t) {
    EXPECT_EQ(rootFlags_[t], rootCanEmbed(problem, t) ? 1 : 0) << "root " << t;
  }
}

TEST_F(WarpDfsTest, DeepPathQueryFillsTheStack) {
  // 40-atom path query on a 64-cycle, with MaxDepth == numQueryAtoms so the
  // terminal fires at exactly MaxDepth - 1: 64 starts x 2 directions = 128,
  // found only by walking the stack through all 40 depths.
  HostProblem problem = cycleTarget(64, 0, 64, 40);
  problem.makePathQuery();

  run<64, 40>(problem, TestMode::CountAll);
  EXPECT_EQ(count_, 128);
  EXPECT_EQ(bruteForceCount(problem), 128);
}

TEST_F(WarpDfsTest, ExistsRaceFindsExactlyOneEmbedding) {
  // The MCS existence pattern: lanes race via a found flag polled in the abort
  // hook, and the atomicCAS winner alone records its embedding.
  HostProblem problem = cycleTarget(64, 0, 64, 5);
  problem.makePathQuery();

  run<64, 8>(problem, TestMode::ExistsRace);
  EXPECT_EQ(count_, 1);
  expectValidEmbedding(problem);
}

TEST_F(WarpDfsTest, ExistsRaceFindsNothingWhenNoEmbeddingExists) {
  // Same shape, but the query path is one atom longer than the ring region
  // can hold injectively... use a 6-path against a 5-cycle in a 64 target.
  HostProblem problem = cycleTarget(64, 10, 5, 6);
  problem.makePathQuery();
  // Restrict all depths to the cycle's atoms so isolated atoms cannot help.
  for (int d = 0; d < problem.numQueryAtoms; ++d) {
    problem.compat[d] = {};
    for (int i = 0; i < 5; ++i) {
      setBit(problem.compat[d], 10 + i);
    }
  }

  ASSERT_EQ(bruteForceCount(problem), 0);
  run<64, 8>(problem, TestMode::ExistsRace);
  EXPECT_EQ(count_, 0);
}

}  // namespace
