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
// discipline, terminal verdicts, and abort hook without any molecule plumbing;
// the substructure integration tests cover the production oracle.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "src/subgraph/target_mask.cuh"
#include "src/subgraph/warp_dfs.cuh"
#include "src/utils/cuda_error_check.h"

using nvMolKit::checkReturnCode;

namespace {

constexpr int kMaxQueryAtoms  = 8;
constexpr int kMaxTargetAtoms = 32;

/// A (target graph, query constraints) pair small enough to pass as a kernel
/// argument. adj[t] is target atom t's neighbour mask; compat[d] the target
/// atoms query atom d may map to; backEdges[d] the earlier query atoms d is
/// bonded to. Every edge accepts every bond (bond predicates are the oracle's
/// business, not the core's).
struct TestProblem {
  uint32_t      adj[kMaxTargetAtoms]           = {};
  uint32_t      compat[kMaxQueryAtoms]         = {};
  unsigned char backEdges[kMaxQueryAtoms][4]   = {};
  unsigned char backEdgeCounts[kMaxQueryAtoms] = {};
  int           numQueryAtoms                  = 0;
  int           numTargetAtoms                 = 0;
};

enum class TestMode {
  CountAll,    ///< Every lane counts all embeddings from its own root.
  FirstOnly,   ///< All roots on lane 0; stop the lane at the first embedding and record it.
  RootExists,  ///< Flag each root that starts at least one embedding, then abandon it.
  AbortAll     ///< Abort hook returns true from the start; nothing may be found.
};

__global__ void warpDfsTestKernel(TestProblem problem,
                                  TestMode    mode,
                                  int*        outCount,
                                  int*        outMapping,
                                  int*        outRootFlags) {
  const int lane = static_cast<int>(threadIdx.x);
  using Mask     = nvMolKit::TargetMask<kMaxTargetAtoms>;

  Mask roots;
  roots.clear();
  if (mode == TestMode::FirstOnly) {
    if (lane == 0) {
      roots.bits = problem.compat[0];
    }
  } else {
    roots.bits = problem.compat[0] & (1u << lane);
  }

  auto candidatesAt = [&](int depth, const unsigned char* mapping, const Mask& used, int /*prevTargetAtom*/) -> Mask {
    Mask candidates;
    candidates.bits = problem.compat[depth];
    candidates.andNotEq(used);
    for (int k = 0; k < problem.backEdgeCounts[depth]; ++k) {
      Mask adjacency;
      adjacency.bits = problem.adj[mapping[problem.backEdges[depth][k]]];
      candidates.andEq(adjacency);
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
    } else {  // RootExists
      if (!terminals.empty()) {
        outRootFlags[mapping[0]] = 1;
        verdict.rootDone         = true;
      }
    }
    return verdict;
  };

  auto abortRequested = [&]() -> bool { return mode == TestMode::AbortAll; };

  nvMolKit::dfsFromRoots<kMaxQueryAtoms>(roots, problem.numQueryAtoms - 1, candidatesAt, onTerminal, abortRequested);
}

// Host-side mirror of the kernel's rules, enumerating all injective embeddings.
void bruteForceRecurse(const TestProblem&             problem,
                       std::vector<int>&              mapping,
                       uint32_t                       used,
                       int                            depth,
                       std::vector<std::vector<int>>& embeddings) {
  if (depth == problem.numQueryAtoms) {
    embeddings.push_back(mapping);
    return;
  }
  for (int t = 0; t < problem.numTargetAtoms; ++t) {
    if (!((problem.compat[depth] >> t) & 1u) || ((used >> t) & 1u)) {
      continue;
    }
    bool edgesOk = true;
    for (int k = 0; k < problem.backEdgeCounts[depth]; ++k) {
      const int mapped = mapping[problem.backEdges[depth][k]];
      if (!((problem.adj[mapped] >> t) & 1u)) {
        edgesOk = false;
        break;
      }
    }
    if (!edgesOk) {
      continue;
    }
    mapping[depth] = t;
    bruteForceRecurse(problem, mapping, used | (1u << t), depth + 1, embeddings);
  }
}

std::vector<std::vector<int>> bruteForceEmbeddings(const TestProblem& problem) {
  std::vector<std::vector<int>> embeddings;
  std::vector<int>              mapping(problem.numQueryAtoms, -1);
  bruteForceRecurse(problem, mapping, 0u, 0, embeddings);
  return embeddings;
}

void addUndirectedEdge(TestProblem& problem, int a, int b) {
  problem.adj[a] |= 1u << b;
  problem.adj[b] |= 1u << a;
}

void allowAllLabels(TestProblem& problem) {
  const uint32_t all = (problem.numTargetAtoms == 32) ? 0xFFFFFFFFu : ((1u << problem.numTargetAtoms) - 1u);
  for (int d = 0; d < problem.numQueryAtoms; ++d) {
    problem.compat[d] = all;
  }
}

/// Triangle 0-1-2 with tail edge 2-3.
TestProblem triangleWithTailTarget(int numQueryAtoms) {
  TestProblem problem;
  problem.numTargetAtoms = 4;
  problem.numQueryAtoms  = numQueryAtoms;
  addUndirectedEdge(problem, 0, 1);
  addUndirectedEdge(problem, 0, 2);
  addUndirectedEdge(problem, 1, 2);
  addUndirectedEdge(problem, 2, 3);
  allowAllLabels(problem);
  return problem;
}

void setBackEdges(TestProblem& problem, int depth, std::initializer_list<int> earlier) {
  int count = 0;
  for (const int e : earlier) {
    problem.backEdges[depth][count++] = static_cast<unsigned char>(e);
  }
  problem.backEdgeCounts[depth] = static_cast<unsigned char>(count);
}

class WarpDfsTest : public ::testing::Test {
 protected:
  void run(const TestProblem& problem, TestMode mode) {
    cudaCheckError(cudaMemset(devCount_, 0, sizeof(int)));
    cudaCheckError(cudaMemset(devMapping_, 0xFF, kMaxQueryAtoms * sizeof(int)));
    cudaCheckError(cudaMemset(devRootFlags_, 0, kMaxTargetAtoms * sizeof(int)));
    warpDfsTestKernel<<<1, 32>>>(problem, mode, devCount_, devMapping_, devRootFlags_);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaMemcpy(&count_, devCount_, sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(mapping_, devMapping_, kMaxQueryAtoms * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(rootFlags_, devRootFlags_, kMaxTargetAtoms * sizeof(int), cudaMemcpyDeviceToHost));
  }

  void SetUp() override {
    cudaCheckError(cudaMalloc(&devCount_, sizeof(int)));
    cudaCheckError(cudaMalloc(&devMapping_, kMaxQueryAtoms * sizeof(int)));
    cudaCheckError(cudaMalloc(&devRootFlags_, kMaxTargetAtoms * sizeof(int)));
  }

  void TearDown() override {
    cudaFree(devCount_);
    cudaFree(devMapping_);
    cudaFree(devRootFlags_);
  }

  int  count_                      = 0;
  int  mapping_[kMaxQueryAtoms]    = {};
  int  rootFlags_[kMaxTargetAtoms] = {};
  int* devCount_                   = nullptr;
  int* devMapping_                 = nullptr;
  int* devRootFlags_               = nullptr;
};

TEST_F(WarpDfsTest, CountsPathEmbeddings) {
  // Query: path 0-1-2. In triangle+tail: sum over middle atoms of
  // degree*(degree-1) = 2 + 2 + 6 + 0 = 10 ordered embeddings.
  TestProblem problem = triangleWithTailTarget(3);
  setBackEdges(problem, 1, {0});
  setBackEdges(problem, 2, {1});

  run(problem, TestMode::CountAll);
  EXPECT_EQ(count_, 10);
  EXPECT_EQ(bruteForceEmbeddings(problem).size(), 10u);
}

TEST_F(WarpDfsTest, CountsTriangleEmbeddings) {
  // Query: triangle. Only the target triangle matches, in all 3! orders.
  TestProblem problem = triangleWithTailTarget(3);
  setBackEdges(problem, 1, {0});
  setBackEdges(problem, 2, {0, 1});

  run(problem, TestMode::CountAll);
  EXPECT_EQ(count_, 6);
  EXPECT_EQ(bruteForceEmbeddings(problem).size(), 6u);
}

TEST_F(WarpDfsTest, RespectsLabelCompatibility) {
  // Path query whose first atom may only map to target atoms 2 or 3.
  TestProblem problem = triangleWithTailTarget(3);
  setBackEdges(problem, 1, {0});
  setBackEdges(problem, 2, {1});
  problem.compat[0] = (1u << 2) | (1u << 3);

  run(problem, TestMode::CountAll);
  EXPECT_EQ(static_cast<std::size_t>(count_), bruteForceEmbeddings(problem).size());
}

TEST_F(WarpDfsTest, DeepQueryMatchesBruteForce) {
  // 4-atom query with a ring closure (cycle 0-1-2-3-0) forces backtracking
  // through several depths; validate against brute force on a denser target.
  TestProblem problem;
  problem.numTargetAtoms = 6;
  problem.numQueryAtoms  = 4;
  addUndirectedEdge(problem, 0, 1);
  addUndirectedEdge(problem, 1, 2);
  addUndirectedEdge(problem, 2, 3);
  addUndirectedEdge(problem, 3, 0);
  addUndirectedEdge(problem, 0, 2);
  addUndirectedEdge(problem, 3, 4);
  addUndirectedEdge(problem, 4, 5);
  allowAllLabels(problem);
  setBackEdges(problem, 1, {0});
  setBackEdges(problem, 2, {1});
  setBackEdges(problem, 3, {2, 0});

  run(problem, TestMode::CountAll);
  const auto expected = bruteForceEmbeddings(problem);
  EXPECT_GT(expected.size(), 0u);
  EXPECT_EQ(static_cast<std::size_t>(count_), expected.size());
}

TEST_F(WarpDfsTest, LaneDoneStopsAfterFirstEmbedding) {
  TestProblem problem = triangleWithTailTarget(3);
  setBackEdges(problem, 1, {0});
  setBackEdges(problem, 2, {1});

  run(problem, TestMode::FirstOnly);
  EXPECT_EQ(count_, 1);
  // The recorded mapping must be a genuine embedding: injective, compatible,
  // and edge-consistent.
  for (int d = 0; d < problem.numQueryAtoms; ++d) {
    ASSERT_GE(mapping_[d], 0);
    ASSERT_LT(mapping_[d], problem.numTargetAtoms);
    EXPECT_TRUE((problem.compat[d] >> mapping_[d]) & 1u);
    for (int e = 0; e < d; ++e) {
      EXPECT_NE(mapping_[d], mapping_[e]);
    }
    for (int k = 0; k < problem.backEdgeCounts[d]; ++k) {
      EXPECT_TRUE((problem.adj[mapping_[problem.backEdges[d][k]]] >> mapping_[d]) & 1u);
    }
  }
}

TEST_F(WarpDfsTest, RootDoneFlagsExactlyTheRootsThatEmbed) {
  // Triangle query: target atoms 0,1,2 can each start an embedding; the tail
  // atom 3 cannot.
  TestProblem problem = triangleWithTailTarget(3);
  setBackEdges(problem, 1, {0});
  setBackEdges(problem, 2, {0, 1});

  run(problem, TestMode::RootExists);
  EXPECT_EQ(rootFlags_[0], 1);
  EXPECT_EQ(rootFlags_[1], 1);
  EXPECT_EQ(rootFlags_[2], 1);
  EXPECT_EQ(rootFlags_[3], 0);
}

TEST_F(WarpDfsTest, AbortHookStopsBeforeAnyRoot) {
  TestProblem problem = triangleWithTailTarget(3);
  setBackEdges(problem, 1, {0});
  setBackEdges(problem, 2, {1});

  run(problem, TestMode::AbortAll);
  EXPECT_EQ(count_, 0);
}

}  // namespace
