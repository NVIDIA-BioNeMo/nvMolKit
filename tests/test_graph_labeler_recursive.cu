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

#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "src/substruct/boolean_tree.cuh"
#include "src/substruct/graph_labeler.cuh"
#include "src/substruct/molecules.h"
#include "src/substruct/molecules_device.cuh"
#include "src/substruct/substruct_search.h"
#include "src/substruct/substruct_search_internal.h"
#include "src/substruct/substruct_types.h"
#include "src/testutils/substruct_validation.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"

using nvMolKit::addQueryToBatch;
using nvMolKit::addToBatch;
using nvMolKit::AsyncDeviceVector;
using nvMolKit::AtomDataPacked;
using nvMolKit::AtomQueryTree;
using nvMolKit::BatchedPatternEntry;
using nvMolKit::BitMatrix2DView;
using nvMolKit::BoolInstruction;
using nvMolKit::BoolOp;
using nvMolKit::checkReturnCode;
using nvMolKit::compareLabelMatrices;
using nvMolKit::computeGpuLabelMatrix;
using nvMolKit::extractRecursivePatterns;
using nvMolKit::FlatBitVect;
using nvMolKit::getRDKitSubstructMatches;
using nvMolKit::hasRecursiveSmarts;
using nvMolKit::kMaxQueryAtoms;
using nvMolKit::kMaxSmartsNestingDepth;
using nvMolKit::kMaxTargetAtoms;
using nvMolKit::LeafSubpatterns;
using nvMolKit::MiniBatchResultsDevice;
using nvMolKit::MoleculesDevice;
using nvMolKit::MoleculesHost;
using nvMolKit::preprocessRecursiveSmarts;
using nvMolKit::RecursivePatternInfo;
using nvMolKit::RecursiveScratchBuffers;
using nvMolKit::ScopedStream;
using nvMolKit::SubstructAlgorithm;
using nvMolKit::SubstructSearchResults;
using nvMolKit::SubstructTemplateConfig;

namespace {

std::unique_ptr<RDKit::ROMol> makeMolFromSmiles(const std::string& smiles) {
  return std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
}

std::unique_ptr<RDKit::ROMol> makeMolFromSmarts(const std::string& smarts) {
  return std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
}

using LabelMatrixStorage = FlatBitVect<kMaxTargetAtoms * kMaxQueryAtoms>;
using LabelMatrixView    = BitMatrix2DView<kMaxTargetAtoms, kMaxQueryAtoms>;

}  // namespace

// =============================================================================
// RecursiveMatch Instruction Generation Tests
// =============================================================================

class RecursiveInstructionTest : public ::testing::Test {
 protected:
  void checkRecursiveMatchInstruction(const MoleculesHost& queryHost, int atomIdx, int expectedPatternId) {
    ASSERT_GT(queryHost.atomQueryTrees.size(), static_cast<size_t>(atomIdx));
    ASSERT_GT(queryHost.atomInstrStarts.size(), static_cast<size_t>(atomIdx));

    const AtomQueryTree& tree       = queryHost.atomQueryTrees[atomIdx];
    int                  instrStart = queryHost.atomInstrStarts[atomIdx];
    int                  instrEnd   = instrStart + tree.numInstructions;

    bool foundRecursiveMatch = false;
    int  foundPatternId      = -1;

    for (int i = instrStart; i < instrEnd; ++i) {
      const BoolInstruction& instr = queryHost.queryInstructions[i];
      if (instr.op == BoolOp::RecursiveMatch) {
        foundRecursiveMatch = true;
        foundPatternId      = instr.auxArg;
        break;
      }
    }

    EXPECT_TRUE(foundRecursiveMatch) << "Expected RecursiveMatch instruction for atom " << atomIdx;
    EXPECT_EQ(foundPatternId, expectedPatternId)
      << "Expected pattern ID " << expectedPatternId << " but got " << foundPatternId;
  }
};

TEST_F(RecursiveInstructionTest, SingleRecursivePattern) {
  auto queryMol = makeMolFromSmarts("[$(*-N)]");
  ASSERT_NE(queryMol, nullptr);

  EXPECT_TRUE(hasRecursiveSmarts(queryMol.get()));

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  ASSERT_EQ(info.size(), 1);
  EXPECT_EQ(info.patterns[0].patternId, 0);
  EXPECT_EQ(info.patterns[0].queryAtomIdx, 0);

  MoleculesHost queryHost;
  addQueryToBatch(queryMol.get(), queryHost);

  checkRecursiveMatchInstruction(queryHost, 0, 0);
}

TEST_F(RecursiveInstructionTest, TwoRecursivePatternsOnDifferentAtoms) {
  auto queryMol = makeMolFromSmarts("[$(*-N)][$(*-O)]");
  ASSERT_NE(queryMol, nullptr);

  EXPECT_TRUE(hasRecursiveSmarts(queryMol.get()));

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  ASSERT_EQ(info.size(), 2);
  EXPECT_EQ(info.patterns[0].patternId, 0);
  EXPECT_EQ(info.patterns[0].queryAtomIdx, 0);
  EXPECT_EQ(info.patterns[1].patternId, 1);
  EXPECT_EQ(info.patterns[1].queryAtomIdx, 1);

  MoleculesHost queryHost;
  addQueryToBatch(queryMol.get(), queryHost);

  checkRecursiveMatchInstruction(queryHost, 0, 0);
  checkRecursiveMatchInstruction(queryHost, 1, 1);
}

TEST_F(RecursiveInstructionTest, RecursivePatternWithAndCondition) {
  auto queryMol = makeMolFromSmarts("[C;$(*-N)]");
  ASSERT_NE(queryMol, nullptr);

  EXPECT_TRUE(hasRecursiveSmarts(queryMol.get()));

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  ASSERT_EQ(info.size(), 1);
  EXPECT_EQ(info.patterns[0].patternId, 0);

  MoleculesHost queryHost;
  addQueryToBatch(queryMol.get(), queryHost);

  checkRecursiveMatchInstruction(queryHost, 0, 0);
}

TEST_F(RecursiveInstructionTest, MultipleRecursivePatternsOnSameAtom) {
  auto queryMol = makeMolFromSmarts("[C;$(*-N);$(*-O)]");
  ASSERT_NE(queryMol, nullptr);

  EXPECT_TRUE(hasRecursiveSmarts(queryMol.get()));

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  ASSERT_EQ(info.size(), 2);
  EXPECT_EQ(info.patterns[0].patternId, 0);
  EXPECT_EQ(info.patterns[0].queryAtomIdx, 0);
  EXPECT_EQ(info.patterns[1].patternId, 1);
  EXPECT_EQ(info.patterns[1].queryAtomIdx, 0);

  MoleculesHost queryHost;
  addQueryToBatch(queryMol.get(), queryHost);

  const AtomQueryTree& tree       = queryHost.atomQueryTrees[0];
  int                  instrStart = queryHost.atomInstrStarts[0];
  int                  instrEnd   = instrStart + tree.numInstructions;

  int recursiveMatchCount = 0;
  for (int i = instrStart; i < instrEnd; ++i) {
    if (queryHost.queryInstructions[i].op == BoolOp::RecursiveMatch) {
      recursiveMatchCount++;
    }
  }
  EXPECT_EQ(recursiveMatchCount, 2);
}

TEST_F(RecursiveInstructionTest, RecursivePatternWithOr) {
  auto queryMol = makeMolFromSmarts("[C,$(*-N)]");
  ASSERT_NE(queryMol, nullptr);

  EXPECT_TRUE(hasRecursiveSmarts(queryMol.get()));

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  ASSERT_EQ(info.size(), 1);

  MoleculesHost queryHost;
  addQueryToBatch(queryMol.get(), queryHost);

  checkRecursiveMatchInstruction(queryHost, 0, 0);
}

TEST_F(RecursiveInstructionTest, NegatedRecursivePattern) {
  auto queryMol = makeMolFromSmarts("[C;!$(*-N)]");
  ASSERT_NE(queryMol, nullptr);

  EXPECT_TRUE(hasRecursiveSmarts(queryMol.get()));

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  ASSERT_EQ(info.size(), 1);

  MoleculesHost queryHost;
  addQueryToBatch(queryMol.get(), queryHost);

  const AtomQueryTree& tree       = queryHost.atomQueryTrees[0];
  int                  instrStart = queryHost.atomInstrStarts[0];
  int                  instrEnd   = instrStart + tree.numInstructions;

  bool hasRecursive = false;
  bool hasNot       = false;
  int  recursiveIdx = -1;

  for (int i = instrStart; i < instrEnd; ++i) {
    const BoolInstruction& instr = queryHost.queryInstructions[i];
    if (instr.op == BoolOp::RecursiveMatch) {
      hasRecursive = true;
      recursiveIdx = instr.dst;
    }
    if (instr.op == BoolOp::Not && instr.src1 == recursiveIdx) {
      hasNot = true;
    }
  }

  EXPECT_TRUE(hasRecursive);
  EXPECT_TRUE(hasNot) << "Expected NOT instruction negating the RecursiveMatch";
}

// =============================================================================
// Preprocess -> Paint Workflow Tests
// =============================================================================

class RecursivePaintTest : public ::testing::Test {
 protected:
  ScopedStream                            stream_;
  std::unique_ptr<MiniBatchResultsDevice> results_;
  std::unique_ptr<AsyncDeviceVector<int>> pairMatchStartsDev_;
  int                                     maxTargetAtoms_ = 0;
  int                                     numTargets_     = 0;
  int                                     numQueries_     = 1;

  void setupResults(const MoleculesHost& targetsHost, int numQueries = 1) {
    const int numTargets = static_cast<int>(targetsHost.numMolecules());
    numTargets_          = numTargets;
    numQueries_          = numQueries;

    std::vector<int> queryAtomCounts(numQueries, 1);
    maxTargetAtoms_ = 0;
    for (int t = 0; t < numTargets; ++t) {
      int atomCount   = targetsHost.batchAtomStarts[t + 1] - targetsHost.batchAtomStarts[t];
      maxTargetAtoms_ = std::max(maxTargetAtoms_, atomCount);
    }

    const int miniBatchSize = numTargets * numQueries;
    pairMatchStartsDev_     = std::make_unique<AsyncDeviceVector<int>>(miniBatchSize + 1, stream_.stream());
    pairMatchStartsDev_->zero();

    results_ = std::make_unique<MiniBatchResultsDevice>(stream_.stream());
    results_->allocateMiniBatch(miniBatchSize, pairMatchStartsDev_->data(), 0, numQueries, maxTargetAtoms_, 2);
    results_->setQueryAtomCounts(queryAtomCounts.data(), queryAtomCounts.size());
    results_->zeroRecursiveBits();
  }

  void verifyRecursiveBitSet(int targetMolIdx, int targetAtomIdx, int patternId, bool expectedSet, int queryIdx = 0) {
    ASSERT_NE(results_, nullptr) << "Results not initialized - call setupResults first";

    const int pairIdx   = targetMolIdx * results_->numQueries() + queryIdx;
    const int bufferIdx = pairIdx * maxTargetAtoms_ + targetAtomIdx;

    std::vector<uint32_t> hostBits(1);
    cudaCheckError(cudaMemcpyAsync(hostBits.data(),
                                   results_->recursiveMatchBits() + bufferIdx,
                                   sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost,
                                   stream_.stream()));
    cudaCheckError(cudaStreamSynchronize(stream_.stream()));

    bool isSet = (hostBits[0] >> patternId) & 1u;
    EXPECT_EQ(isSet, expectedSet) << "Target mol " << targetMolIdx << ", atom " << targetAtomIdx << ", pattern "
                                  << patternId << " expected " << (expectedSet ? "set" : "unset");
  }

  void preprocessRecursive(const MoleculesDevice& targetDevice,
                           const MoleculesHost& /* targetHost */,
                           const RDKit::ROMol* queryMol) {
    MoleculesHost queryHost;
    addQueryToBatch(queryMol, queryHost);

    LeafSubpatterns leafSubpatterns;
    leafSubpatterns.buildAllPatterns(queryHost);
    leafSubpatterns.syncToDevice(stream_.stream());

    RecursiveScratchBuffers scratch(stream_.stream());
    scratch.allocateBuffers(256);
    std::vector<BatchedPatternEntry> scratchPatternEntries;
    preprocessRecursiveSmarts(SubstructTemplateConfig::Config_T128_Q64_B8,
                              targetDevice,
                              queryHost,
                              leafSubpatterns,
                              *results_,
                              numQueries_,
                              0,
                              numTargets_ * numQueries_,
                              SubstructAlgorithm::GSI,
                              stream_.stream(),
                              scratch,
                              scratchPatternEntries,
                              nullptr,
                              0);
    cudaCheckError(cudaStreamSynchronize(stream_.stream()));
  }
};

TEST_F(RecursivePaintTest, SimpleCarbonBondedToNitrogen) {
  auto targetMol = makeMolFromSmiles("CN");
  auto queryMol  = makeMolFromSmarts("[$(*-N)]");
  ASSERT_NE(targetMol, nullptr);
  ASSERT_NE(queryMol, nullptr);

  MoleculesHost targetHost;
  addToBatch(targetMol.get(), targetHost);
  setupResults(targetHost);

  MoleculesDevice targetDevice(stream_.stream());
  targetDevice.copyFromHost(targetHost);

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  ASSERT_EQ(info.size(), 1);

  preprocessRecursive(targetDevice, targetHost, queryMol.get());

  // Pattern *-N paints the atom matching * (the anchor), not the N
  verifyRecursiveBitSet(0, 0, 0, true);   // C is bonded to N
  verifyRecursiveBitSet(0, 1, 0, false);  // N is not the anchor
}

TEST_F(RecursivePaintTest, OnlyMatchingAtomsArePainted) {
  // CCN: C(0)-C(1)-N(2)
  auto targetMol = makeMolFromSmiles("CCN");
  auto queryMol  = makeMolFromSmarts("[$(*-N)]");
  ASSERT_NE(targetMol, nullptr);
  ASSERT_NE(queryMol, nullptr);

  MoleculesHost targetHost;
  addToBatch(targetMol.get(), targetHost);
  setupResults(targetHost);

  MoleculesDevice targetDevice(stream_.stream());
  targetDevice.copyFromHost(targetHost);

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());

  preprocessRecursive(targetDevice, targetHost, queryMol.get());

  // Only C(1) is bonded to N, so only C(1) gets painted
  verifyRecursiveBitSet(0, 0, 0, false);  // C(0) not bonded to N
  verifyRecursiveBitSet(0, 1, 0, true);   // C(1) bonded to N
  verifyRecursiveBitSet(0, 2, 0, false);  // N is not the anchor
}

TEST_F(RecursivePaintTest, MultiplePatternsMultipleBits) {
  // CNO: C(0)-N(1)-O(2)
  // Both C and O are bonded to N!
  auto targetMol = makeMolFromSmiles("CNO");
  auto queryMol  = makeMolFromSmarts("[$(*-N)][$(*-O)]");
  ASSERT_NE(targetMol, nullptr);
  ASSERT_NE(queryMol, nullptr);

  MoleculesHost targetHost;
  addToBatch(targetMol.get(), targetHost);
  setupResults(targetHost);

  MoleculesDevice targetDevice(stream_.stream());
  targetDevice.copyFromHost(targetHost);

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  ASSERT_EQ(info.size(), 2);

  preprocessRecursive(targetDevice, targetHost, queryMol.get());

  // Pattern 0 (*-N): C(0) and O(2) are both bonded to N(1)
  verifyRecursiveBitSet(0, 0, 0, true);   // C bonded to N
  verifyRecursiveBitSet(0, 1, 0, false);  // N is not anchor for *-N
  verifyRecursiveBitSet(0, 2, 0, true);   // O bonded to N

  // Pattern 1 (*-O): N(1) is bonded to O(2)
  verifyRecursiveBitSet(0, 0, 1, false);  // C not bonded to O
  verifyRecursiveBitSet(0, 1, 1, true);   // N bonded to O
  verifyRecursiveBitSet(0, 2, 1, false);  // O is not anchor for *-O
}

TEST_F(RecursivePaintTest, NoMatchNoBitsPainted) {
  auto targetMol = makeMolFromSmiles("CCC");
  auto queryMol  = makeMolFromSmarts("[$(*-N)]");
  ASSERT_NE(targetMol, nullptr);
  ASSERT_NE(queryMol, nullptr);

  MoleculesHost targetHost;
  addToBatch(targetMol.get(), targetHost);
  setupResults(targetHost);

  MoleculesDevice targetDevice(stream_.stream());
  targetDevice.copyFromHost(targetHost);

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  preprocessRecursive(targetDevice, targetHost, queryMol.get());

  for (int i = 0; i < 3; ++i) {
    verifyRecursiveBitSet(0, i, 0, false);
  }
}

TEST_F(RecursivePaintTest, MultipleTargetMolecules) {
  // Target 1: CN -> C(0)-N(1)
  // Target 2: CC -> C(0)-C(1), no N
  // Target 3: NCN -> N(0)-C(1)-N(2)
  auto target1  = makeMolFromSmiles("CN");
  auto target2  = makeMolFromSmiles("CC");
  auto target3  = makeMolFromSmiles("NCN");
  auto queryMol = makeMolFromSmarts("[$(*-N)]");
  ASSERT_NE(target1, nullptr);
  ASSERT_NE(target2, nullptr);
  ASSERT_NE(target3, nullptr);
  ASSERT_NE(queryMol, nullptr);

  MoleculesHost targetHost;
  addToBatch(target1.get(), targetHost);
  addToBatch(target2.get(), targetHost);
  addToBatch(target3.get(), targetHost);
  setupResults(targetHost);

  MoleculesDevice targetDevice(stream_.stream());
  targetDevice.copyFromHost(targetHost);

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  preprocessRecursive(targetDevice, targetHost, queryMol.get());

  // Target 1 (CN): C bonded to N
  verifyRecursiveBitSet(0, 0, 0, true);   // C bonded to N
  verifyRecursiveBitSet(0, 1, 0, false);  // N not anchor

  // Target 2 (CC): no N, nothing painted
  verifyRecursiveBitSet(1, 0, 0, false);
  verifyRecursiveBitSet(1, 1, 0, false);

  // Target 3 (NCN): C(1) is bonded to both N(0) and N(2)
  verifyRecursiveBitSet(2, 0, 0, false);  // N(0) not anchor
  verifyRecursiveBitSet(2, 1, 0, true);   // C bonded to N
  verifyRecursiveBitSet(2, 2, 0, false);  // N(2) not anchor
}

TEST_F(RecursivePaintTest, AromaticPattern) {
  // c1ccccc1N: atoms 0-5 are aromatic carbons, atom 6 is N bonded to atom 5
  auto targetMol = makeMolFromSmiles("c1ccccc1N");
  auto queryMol  = makeMolFromSmarts("[$(*-N)]");
  ASSERT_NE(targetMol, nullptr);
  ASSERT_NE(queryMol, nullptr);

  MoleculesHost targetHost;
  addToBatch(targetMol.get(), targetHost);
  setupResults(targetHost);

  MoleculesDevice targetDevice(stream_.stream());
  targetDevice.copyFromHost(targetHost);

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  preprocessRecursive(targetDevice, targetHost, queryMol.get());

  // Only atom 5 (the carbon bonded to N) gets painted
  for (int i = 0; i < 6; ++i) {
    bool shouldMatch = (i == 5);
    verifyRecursiveBitSet(0, i, 0, shouldMatch);
  }
  verifyRecursiveBitSet(0, 6, 0, false);  // N is not anchor
}

TEST_F(RecursivePaintTest, MultipleTargetsMultiplePatterns) {
  // Test batch of targets with a query that has multiple recursive patterns
  // Target 0: CN (C-N)
  // Target 1: CO (C-O)
  // Query: [$(*-N)][$(*-O)] has pattern 0 (*-N) and pattern 1 (*-O)
  auto target0  = makeMolFromSmiles("CN");
  auto target1  = makeMolFromSmiles("CO");
  auto queryMol = makeMolFromSmarts("[$(*-N)][$(*-O)]");
  ASSERT_NE(target0, nullptr);
  ASSERT_NE(target1, nullptr);
  ASSERT_NE(queryMol, nullptr);

  MoleculesHost targetHost;
  addToBatch(target0.get(), targetHost);
  addToBatch(target1.get(), targetHost);
  setupResults(targetHost);

  MoleculesDevice targetDevice(stream_.stream());
  targetDevice.copyFromHost(targetHost);

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  ASSERT_EQ(info.size(), 2);

  preprocessRecursive(targetDevice, targetHost, queryMol.get());

  // Target 0 (CN): C bonded to N, no O
  verifyRecursiveBitSet(0, 0, 0, true);   // C has p0 (bonded to N)
  verifyRecursiveBitSet(0, 1, 0, false);  // N not anchor
  verifyRecursiveBitSet(0, 0, 1, false);  // C not bonded to O
  verifyRecursiveBitSet(0, 1, 1, false);  // N not bonded to O

  // Target 1 (CO): C bonded to O, no N
  verifyRecursiveBitSet(1, 0, 0, false);  // C not bonded to N
  verifyRecursiveBitSet(1, 1, 0, false);  // O not bonded to N
  verifyRecursiveBitSet(1, 0, 1, true);   // C has p1 (bonded to O)
  verifyRecursiveBitSet(1, 1, 1, false);  // O not anchor
}

TEST_F(RecursivePaintTest, MultipleTargetsDifferentQueries) {
  // Test that preprocessing works correctly with multiple targets
  // and a query with distinct recursive patterns that match different targets
  // Target 0: CCN (chain with N)
  // Target 1: CCO (chain with O)
  // Query: [$(*-N);$(*-O)] requires BOTH patterns to match (nothing should match)
  auto target0  = makeMolFromSmiles("CCN");
  auto target1  = makeMolFromSmiles("CCO");
  auto queryMol = makeMolFromSmarts("[$(*-N);$(*-O)]");
  ASSERT_NE(target0, nullptr);
  ASSERT_NE(target1, nullptr);
  ASSERT_NE(queryMol, nullptr);

  MoleculesHost targetHost;
  addToBatch(target0.get(), targetHost);
  addToBatch(target1.get(), targetHost);
  setupResults(targetHost);

  MoleculesDevice targetDevice(stream_.stream());
  targetDevice.copyFromHost(targetHost);

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  ASSERT_EQ(info.size(), 2);

  preprocessRecursive(targetDevice, targetHost, queryMol.get());

  // Target 0 (CCN): C(1) bonded to N gets p0, but no O so no p1
  verifyRecursiveBitSet(0, 0, 0, false);  // C(0) not bonded to N
  verifyRecursiveBitSet(0, 1, 0, true);   // C(1) bonded to N
  verifyRecursiveBitSet(0, 2, 0, false);  // N not anchor
  verifyRecursiveBitSet(0, 0, 1, false);  // no O in molecule
  verifyRecursiveBitSet(0, 1, 1, false);
  verifyRecursiveBitSet(0, 2, 1, false);

  // Target 1 (CCO): C(1) bonded to O gets p1, but no N so no p0
  verifyRecursiveBitSet(1, 0, 0, false);  // no N in molecule
  verifyRecursiveBitSet(1, 1, 0, false);
  verifyRecursiveBitSet(1, 2, 0, false);
  verifyRecursiveBitSet(1, 0, 1, false);  // C(0) not bonded to O
  verifyRecursiveBitSet(1, 1, 1, true);   // C(1) bonded to O
  verifyRecursiveBitSet(1, 2, 1, false);  // O not anchor
}

// =============================================================================
// Label Matrix Tests with Recursive SMARTS
// =============================================================================

class RecursiveLabelingTest : public ::testing::Test {
 protected:
  ScopedStream stream_;

  void runRecursiveLabelingTest(const std::string&                 targetSmiles,
                                const std::string&                 querySmarts,
                                std::vector<std::vector<uint8_t>>& expectedMatrix) {
    auto targetMol = makeMolFromSmiles(targetSmiles);
    auto queryMol  = makeMolFromSmarts(querySmarts);
    ASSERT_NE(targetMol, nullptr) << "Failed to parse target: " << targetSmiles;
    ASSERT_NE(queryMol, nullptr) << "Failed to parse query: " << querySmarts;

    auto gpuMatrix = computeGpuLabelMatrix(*targetMol, *queryMol, stream_.stream());

    const int numTargetAtoms = static_cast<int>(gpuMatrix.size());
    const int numQueryAtoms  = numTargetAtoms > 0 ? static_cast<int>(gpuMatrix[0].size()) : 0;

    ASSERT_EQ(expectedMatrix.size(), numTargetAtoms);
    for (int i = 0; i < numTargetAtoms; ++i) {
      ASSERT_EQ(expectedMatrix[i].size(), numQueryAtoms);
      for (int j = 0; j < numQueryAtoms; ++j) {
        EXPECT_EQ(gpuMatrix[i][j], expectedMatrix[i][j]) << "Mismatch at target atom " << i << ", query atom " << j
                                                         << " for target=" << targetSmiles << ", query=" << querySmarts;
      }
    }
  }

  void runRecursiveLabelingTestVsRDKit(const std::string& targetSmiles, const std::string& querySmarts) {
    auto targetMol = makeMolFromSmiles(targetSmiles);
    auto queryMol  = makeMolFromSmarts(querySmarts);
    ASSERT_NE(targetMol, nullptr) << "Failed to parse target: " << targetSmiles;
    ASSERT_NE(queryMol, nullptr) << "Failed to parse query: " << querySmarts;

    auto result = compareLabelMatrices(*targetMol, *queryMol, stream_.stream());
    EXPECT_TRUE(result.allMatch) << "GPU/RDKit mismatch for target=" << targetSmiles << ", query=" << querySmarts
                                 << " (FP=" << result.falsePositives << ", FN=" << result.falseNegatives << ")";
  }
};

TEST_F(RecursiveLabelingTest, SimpleRecursiveMatch) {
  // CN: C(0) has recursive bit (bonded to N), N(1) does not
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // C has bit set
    {false}  // N does not have bit set
  };
  runRecursiveLabelingTest("CN", "[$(*-N)]", expected);
}

TEST_F(RecursiveLabelingTest, RecursiveNoMatch) {
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}};
  runRecursiveLabelingTest("CCC", "[$(*-N)]", expected);
}

TEST_F(RecursiveLabelingTest, RecursivePartialMatch) {
  // CCN: only C(1) has the recursive bit (bonded to N)
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // C(0) not bonded to N
    {true},   // C(1) bonded to N
    {false}   // N(2) does not have bit set
  };
  runRecursiveLabelingTest("CCN", "[$(*-N)]", expected);
}

TEST_F(RecursiveLabelingTest, RecursiveWithAtomType) {
  std::vector<std::vector<uint8_t>> expected = {{true}, {false}};
  runRecursiveLabelingTest("CN", "[C;$(*-N)]", expected);
}

TEST_F(RecursiveLabelingTest, RecursiveWithAtomTypeNoMatch) {
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}};
  runRecursiveLabelingTest("CC", "[C;$(*-N)]", expected);
}

TEST_F(RecursiveLabelingTest, NegatedRecursive) {
  // CCN: C(0) has no bit, C(1) has bit, N(2) has no bit
  // Query [C;!$(*-N)] = aliphatic C AND NOT bonded-to-N
  std::vector<std::vector<uint8_t>> expected = {
    {true},   // C(0): is C, no bit -> matches
    {false},  // C(1): is C, HAS bit -> fails NOT
    {false}   // N(2): not C -> fails
  };
  runRecursiveLabelingTest("CCN", "[C;!$(*-N)]", expected);
}

TEST_F(RecursiveLabelingTest, TwoAtomQueryWithRecursive) {
  // CN: C(0) has bit, N(1) does not
  // Query: q0=[$(*-N)], q1=N
  std::vector<std::vector<uint8_t>> expected = {
    { true, false}, // C: has bit (matches q0), not N (fails q1)
    {false,  true}  // N: no bit (fails q0), is N (matches q1)
  };
  runRecursiveLabelingTest("CN", "[$(*-N)]N", expected);
}

TEST_F(RecursiveLabelingTest, RecursiveOrAtomType) {
  // CCN: C(0) no bit, C(1) has bit, N(2) no bit
  // Query [C,$(*-N)] = aliphatic C OR has-recursive-bit
  std::vector<std::vector<uint8_t>> expected = {
    {true},  // C(0): is C -> matches
    {true},  // C(1): is C (or has bit) -> matches
    {false}  // N(2): not C, no bit -> fails
  };
  runRecursiveLabelingTest("CCN", "[C,$(*-N)]", expected);
}

TEST_F(RecursiveLabelingTest, ComplexRecursivePattern) {
  // c1ccccc1N: atoms 0-5 are aromatic c, atom 6 is N
  // Only c(5) has the recursive bit (bonded to N)
  // Query [c;$(*-N)] = aromatic c AND has-bit
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c(0)
    {false},  // c(1)
    {false},  // c(2)
    {false},  // c(3)
    {false},  // c(4)
    {true},   // c(5) bonded to N
    {false}   // N(6) not aromatic c
  };
  runRecursiveLabelingTest("c1ccccc1N", "[c;$(*-N)]", expected);
}

TEST_F(RecursiveLabelingTest, MultipleRecursivePatternsInQuery) {
  // CNO: C(0)-N(1)-O(2)
  // Pattern 0 (*-N): C(0) and O(2) both get bit 0 (both bonded to N)
  // Pattern 1 (*-O): N(1) gets bit 1
  // Query: q0=[$(*-N)] (has p0), q1=[$(*-O)] (has p1)
  std::vector<std::vector<uint8_t>> expected = {
    { true, false}, // C: has p0 (matches q0), no p1 (fails q1)
    {false,  true}, // N: no p0 (fails q0), has p1 (matches q1)
    { true, false}  // O: has p0 (matches q0), no p1 (fails q1)
  };
  runRecursiveLabelingTest("CNO", "[$(*-N)][$(*-O)]", expected);
}

TEST_F(RecursiveLabelingTest, RecursiveAndRingQuery) {
  // c1ccccc1N: atoms 0-5 aromatic c (in ring), atom 6 is N (not in ring)
  // Only c(5) has the recursive bit
  // Query [R;$(*-N)] = in-ring AND has-bit
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c(0): in ring but no bit
    {false},  // c(1): in ring but no bit
    {false},  // c(2): in ring but no bit
    {false},  // c(3): in ring but no bit
    {false},  // c(4): in ring but no bit
    {true},   // c(5): in ring AND has bit
    {false}   // N(6): not in ring
  };
  runRecursiveLabelingTest("c1ccccc1N", "[R;$(*-N)]", expected);
}

TEST_F(RecursiveLabelingTest, ChainedRecursiveOnSameAtom) {
  // CNO: C(0)-N(1)-O(2)
  // Pattern 0 (*-N): C(0) and O(2) get p0 (both bonded to N)
  // Pattern 1 (*-O): N(1) gets p1
  // Query [$(*-N);$(*-O)] = has p0 AND has p1
  // No atom has BOTH bits
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // C: has p0 only
    {false},  // N: has p1 only
    {false}   // O: has p0 only
  };
  runRecursiveLabelingTest("CNO", "[$(*-N);$(*-O)]", expected);
}

TEST_F(RecursiveLabelingTest, RecursiveWithAromaticity) {
  std::vector<std::vector<uint8_t>> expected = {{false}, {false}, {false}, {false}, {false}, {false}, {false}};
  runRecursiveLabelingTest("c1ccccc1N", "[c;$(*-n)]", expected);
}

TEST_F(RecursiveLabelingTest, RecursiveAliphaticNitrogen) {
  // c1ccccc1N: atoms 0-5 aromatic c, atom 6 is N
  // Pattern *-N: only c(5) gets the bit (bonded to N)
  // Query [$(*-N)] = has bit
  std::vector<std::vector<uint8_t>> expected = {
    {false},  // c(0)
    {false},  // c(1)
    {false},  // c(2)
    {false},  // c(3)
    {false},  // c(4)
    {true},   // c(5) bonded to N
    {false}   // N(6) is not the anchor
  };
  runRecursiveLabelingTest("c1ccccc1N", "[$(*-N)]", expected);
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

TEST(RecursiveLabelerEdgeCases, NoRecursivePatterns) {
  auto queryMol = makeMolFromSmarts("CC");
  ASSERT_NE(queryMol, nullptr);

  EXPECT_FALSE(hasRecursiveSmarts(queryMol.get()));

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  EXPECT_TRUE(info.empty());
  EXPECT_EQ(info.size(), 0);
}

TEST(RecursiveLabelerEdgeCases, MaxPatternsLimit) {
  std::string smarts = "[C";
  for (int i = 0; i < RecursivePatternInfo::kMaxPatterns; ++i) {
    smarts += ";$(*-N)";
  }
  smarts += "]";

  auto queryMol = makeMolFromSmarts(smarts);
  ASSERT_NE(queryMol, nullptr);

  RecursivePatternInfo info = extractRecursivePatterns(queryMol.get());
  EXPECT_EQ(info.size(), RecursivePatternInfo::kMaxPatterns);

  for (size_t i = 0; i < RecursivePatternInfo::kMaxPatterns; ++i) {
    EXPECT_EQ(info.patterns[i].patternId, i);
  }
}

TEST(RecursiveLabelerEdgeCases, TooManyPatternsThrows) {
  std::string smarts = "[C";
  for (int i = 0; i < RecursivePatternInfo::kMaxPatterns + 1; ++i) {
    smarts += ";$(*-N)";
  }
  smarts += "]";

  auto queryMol = makeMolFromSmarts(smarts);
  ASSERT_NE(queryMol, nullptr);

  EXPECT_THROW(extractRecursivePatterns(queryMol.get()), std::runtime_error);
}

TEST(RecursiveLabelerEdgeCases, MaxRecursionDepthLimit) {
  // kMaxSmartsNestingDepth is 4, so maxDepth of 3 (0,1,2,3) should work
  // maxDepth of 4 (0,1,2,3,4) should throw
  // Create nested patterns: depth 0 inside depth 1 inside depth 2 etc.
  // [$([*;$([*;$([*;$([*;$(*)])])])])] has depth 4 (5 levels: 0,1,2,3,4)

  // This pattern has depth 4: [$([*;$([*;$([*;$([*;$(*-N)])])])])]
  // Outer: depth 4, contains depth 3
  // Next: depth 3, contains depth 2
  // Next: depth 2, contains depth 1
  // Next: depth 1, contains depth 0
  // Inner: depth 0 (leaf)
  const std::string deeplyNested = "[$([*;$([*;$([*;$([*;$(*-N)])])])])]";

  auto queryMol = makeMolFromSmarts(deeplyNested);
  ASSERT_NE(queryMol, nullptr);

  // Extract should succeed (patterns are extracted but depth is recorded)
  auto info = extractRecursivePatterns(queryMol.get());
  EXPECT_GE(info.maxDepth, kMaxSmartsNestingDepth);
}

TEST(RecursiveLabelerEdgeCases, MaxRecursionDepthFallsBackToRDKit) {
  // Create a pattern that exceeds kMaxSmartsNestingDepth and verify it falls back
  // to RDKit instead of throwing
  const std::string deeplyNested = "[$([*;$([*;$([*;$([*;$(*-N)])])])])]";

  auto targetMol = makeMolFromSmiles("CCCCN");
  auto queryMol  = makeMolFromSmarts(deeplyNested);
  ASSERT_NE(targetMol, nullptr);
  ASSERT_NE(queryMol, nullptr);

  // Verify the pattern exceeds depth limit
  auto info = extractRecursivePatterns(queryMol.get());
  ASSERT_GE(info.maxDepth, kMaxSmartsNestingDepth) << "Test pattern does not exceed depth limit";

  ScopedStream                     stream;
  std::vector<const RDKit::ROMol*> targets = {targetMol.get()};
  std::vector<const RDKit::ROMol*> queries = {queryMol.get()};

  nvMolKit::SubstructSearchResults results;
  EXPECT_NO_THROW(nvMolKit::getSubstructMatches(targets, queries, results, SubstructAlgorithm::GSI, stream.stream()))
    << "Deep recursion should fall back to RDKit instead of throwing";

  // Verify we got the correct result from RDKit fallback
  auto rdkitMatches = getRDKitSubstructMatches(*targetMol, *queryMol, false);
  EXPECT_EQ(results.matchCount(0, 0), static_cast<int>(rdkitMatches.size()))
    << "Fallback should produce same results as RDKit";
}

// =============================================================================
// Nested Recursive Labeling Tests
// =============================================================================

TEST_F(RecursiveLabelingTest, NestedRecursiveMatchVsRDKit) {
  runRecursiveLabelingTestVsRDKit("CN", "[$([*;$(*-N)])]");
  runRecursiveLabelingTestVsRDKit("CCN", "[$([C;$(*-N)])]");
  runRecursiveLabelingTestVsRDKit("CNOF", "[$([*;$([*;$(*-F)])])]");
  runRecursiveLabelingTestVsRDKit("c1ccccc1N", "[$([c;$(*-N)])]");
}
