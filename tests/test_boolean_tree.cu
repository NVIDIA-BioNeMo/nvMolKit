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

#include <gtest/gtest.h>

#include <vector>

#include "src/substruct/boolean_tree.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"

using nvMolKit::AsyncDeviceVector;
using nvMolKit::AtomDataPacked;
using nvMolKit::AtomQueryMask;
using nvMolKit::AtomQueryTree;
using nvMolKit::BondTypeCounts;
using nvMolKit::BoolInstruction;
using nvMolKit::BoolOp;
using nvMolKit::checkReturnCode;
using nvMolKit::CompareField;
using nvMolKit::evaluateBoolTree;
using nvMolKit::ScopedStream;

namespace {

// =============================================================================
// Helper functions to create test atoms and masks
// =============================================================================

AtomDataPacked makeCarbon() {
  AtomDataPacked atom;
  atom.setAtomicNum(6);
  atom.setIsAromatic(false);
  return atom;
}

AtomDataPacked makeNitrogen() {
  AtomDataPacked atom;
  atom.setAtomicNum(7);
  atom.setIsAromatic(false);
  return atom;
}

AtomDataPacked makeOxygen() {
  AtomDataPacked atom;
  atom.setAtomicNum(8);
  atom.setIsAromatic(false);
  return atom;
}

AtomDataPacked makeAromaticCarbon() {
  AtomDataPacked atom;
  atom.setAtomicNum(6);
  atom.setIsAromatic(true);
  return atom;
}

AtomQueryMask makeCarbonMask() {
  AtomQueryMask mask;
  mask.maskLo     = 0xFFULL;
  mask.expectedLo = 6ULL;
  mask.maskHi     = 0;
  mask.expectedHi = 0;
  return mask;
}

AtomQueryMask makeNitrogenMask() {
  AtomQueryMask mask;
  mask.maskLo     = 0xFFULL;
  mask.expectedLo = 7ULL;
  mask.maskHi     = 0;
  mask.expectedHi = 0;
  return mask;
}

AtomQueryMask makeOxygenMask() {
  AtomQueryMask mask;
  mask.maskLo     = 0xFFULL;
  mask.expectedLo = 8ULL;
  mask.maskHi     = 0;
  mask.expectedHi = 0;
  return mask;
}

AtomQueryMask makeAromaticMask() {
  AtomQueryMask mask;
  mask.maskLo                  = 0;
  mask.expectedLo              = 0;
  constexpr int aromaticBitPos = AtomDataPacked::kDegreeByte * 8 + AtomDataPacked::kIsAromaticBit;
  mask.maskHi                  = 1ULL << aromaticBitPos;
  mask.expectedHi              = 1ULL << aromaticBitPos;
  return mask;
}

AtomQueryMask makeAliphaticMask() {
  AtomQueryMask mask;
  mask.maskLo                  = 0;
  mask.expectedLo              = 0;
  constexpr int aromaticBitPos = AtomDataPacked::kDegreeByte * 8 + AtomDataPacked::kIsAromaticBit;
  mask.maskHi                  = 1ULL << aromaticBitPos;
  mask.expectedHi              = 0;  // Expect bit NOT set for aliphatic
  return mask;
}

BondTypeCounts makeEmptyBonds() {
  return BondTypeCounts{};
}

}  // namespace

// =============================================================================
// BoolInstruction Static Factory Tests
// =============================================================================

TEST(BoolInstructionTest, MakeLeafSetsCorrectFields) {
  const BoolInstruction instr = BoolInstruction::makeLeaf(3, 5);

  EXPECT_EQ(instr.op, BoolOp::Leaf);
  EXPECT_EQ(instr.dst, 3);
  EXPECT_EQ(instr.auxArg, 5);
}

TEST(BoolInstructionTest, MakeAndSetsCorrectFields) {
  const BoolInstruction instr = BoolInstruction::makeAnd(2, 0, 1);

  EXPECT_EQ(instr.op, BoolOp::And);
  EXPECT_EQ(instr.dst, 2);
  EXPECT_EQ(instr.src1, 0);
  EXPECT_EQ(instr.src2, 1);
}

TEST(BoolInstructionTest, MakeOrSetsCorrectFields) {
  const BoolInstruction instr = BoolInstruction::makeOr(4, 2, 3);

  EXPECT_EQ(instr.op, BoolOp::Or);
  EXPECT_EQ(instr.dst, 4);
  EXPECT_EQ(instr.src1, 2);
  EXPECT_EQ(instr.src2, 3);
}

TEST(BoolInstructionTest, MakeNotSetsCorrectFields) {
  const BoolInstruction instr = BoolInstruction::makeNot(1, 0);

  EXPECT_EQ(instr.op, BoolOp::Not);
  EXPECT_EQ(instr.dst, 1);
  EXPECT_EQ(instr.src1, 0);
}

// =============================================================================
// evaluateBoolTree Host Tests - Single Leaf (Simple AND-only)
// =============================================================================

TEST(EvaluateBoolTreeTest, SingleLeafMatchesCarbon) {
  const AtomDataPacked target      = makeCarbon();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds()};

  BoolInstruction     instructions[] = {BoolInstruction::makeLeaf(0, 0)};
  const AtomQueryTree tree{1, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_TRUE(result);
}

TEST(EvaluateBoolTreeTest, SingleLeafDoesNotMatchWrongElement) {
  const AtomDataPacked target      = makeNitrogen();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0)};
  AtomQueryTree   tree{1, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_FALSE(result);
}

// =============================================================================
// evaluateBoolTree Host Tests - Binary And
// =============================================================================

TEST(EvaluateBoolTreeTest, AndBothTrue) {
  const AtomDataPacked target      = makeCarbon();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask(), makeAliphaticMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeLeaf(1, 1),
                                    BoolInstruction::makeAnd(2, 0, 1)};
  AtomQueryTree   tree{2, 3, 3, 2};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_TRUE(result);
}

TEST(EvaluateBoolTreeTest, AndFirstFalse) {
  const AtomDataPacked target      = makeNitrogen();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask(), makeAliphaticMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeLeaf(1, 1),
                                    BoolInstruction::makeAnd(2, 0, 1)};
  AtomQueryTree   tree{2, 3, 3, 2};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_FALSE(result);
}

TEST(EvaluateBoolTreeTest, AndSecondFalse) {
  const AtomDataPacked target      = makeAromaticCarbon();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask(), makeAliphaticMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeLeaf(1, 1),
                                    BoolInstruction::makeAnd(2, 0, 1)};
  AtomQueryTree   tree{2, 3, 3, 2};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_FALSE(result);
}

TEST(EvaluateBoolTreeTest, AndBothFalse) {
  const AtomDataPacked target      = makeAromaticCarbon();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeNitrogenMask(), makeAliphaticMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeLeaf(1, 1),
                                    BoolInstruction::makeAnd(2, 0, 1)};
  AtomQueryTree   tree{2, 3, 3, 2};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_FALSE(result);
}

// =============================================================================
// evaluateBoolTree Host Tests - Binary Or
// =============================================================================

TEST(EvaluateBoolTreeTest, OrBothTrue) {
  const AtomDataPacked target      = makeCarbon();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask(), makeAliphaticMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeLeaf(1, 1),
                                    BoolInstruction::makeOr(2, 0, 1)};
  AtomQueryTree   tree{2, 3, 3, 2};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_TRUE(result);
}

TEST(EvaluateBoolTreeTest, OrFirstTrueOnly) {
  const AtomDataPacked target      = makeCarbon();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask(), makeNitrogenMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeLeaf(1, 1),
                                    BoolInstruction::makeOr(2, 0, 1)};
  AtomQueryTree   tree{2, 3, 3, 2};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_TRUE(result);
}

TEST(EvaluateBoolTreeTest, OrSecondTrueOnly) {
  const AtomDataPacked target      = makeNitrogen();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask(), makeNitrogenMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeLeaf(1, 1),
                                    BoolInstruction::makeOr(2, 0, 1)};
  AtomQueryTree   tree{2, 3, 3, 2};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_TRUE(result);
}

TEST(EvaluateBoolTreeTest, OrBothFalse) {
  const AtomDataPacked target      = makeOxygen();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask(), makeNitrogenMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeLeaf(1, 1),
                                    BoolInstruction::makeOr(2, 0, 1)};
  AtomQueryTree   tree{2, 3, 3, 2};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_FALSE(result);
}

// =============================================================================
// evaluateBoolTree Host Tests - Unary Not
// =============================================================================

TEST(EvaluateBoolTreeTest, NotTrueBecomeFalse) {
  const AtomDataPacked target      = makeCarbon();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0), BoolInstruction::makeNot(1, 0)};
  AtomQueryTree   tree{1, 2, 2, 1};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_FALSE(result);
}

TEST(EvaluateBoolTreeTest, NotFalseBecomeTrue) {
  const AtomDataPacked target      = makeNitrogen();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0), BoolInstruction::makeNot(1, 0)};
  AtomQueryTree   tree{1, 2, 2, 1};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_TRUE(result);
}

// =============================================================================
// evaluateBoolTree Host Tests - Complex Combinations
// =============================================================================

TEST(EvaluateBoolTreeTest, AndWithNot_CarbonAndNotAromatic) {
  const AtomDataPacked target      = makeCarbon();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask(), makeAromaticMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeLeaf(1, 1),
                                    BoolInstruction::makeNot(2, 1),
                                    BoolInstruction::makeAnd(3, 0, 2)};
  AtomQueryTree   tree{2, 4, 4, 3};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_TRUE(result);
}

TEST(EvaluateBoolTreeTest, AndWithNot_AromaticCarbonFails) {
  const AtomDataPacked target      = makeAromaticCarbon();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask(), makeAromaticMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeLeaf(1, 1),
                                    BoolInstruction::makeNot(2, 1),
                                    BoolInstruction::makeAnd(3, 0, 2)};
  AtomQueryTree   tree{2, 4, 4, 3};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_FALSE(result);
}

TEST(EvaluateBoolTreeTest, OrOfAnd_CarbonOrNitrogen_AndAliphatic) {
  const AtomDataPacked target      = makeCarbon();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask(), makeNitrogenMask(), makeAliphaticMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeLeaf(1, 1),
                                    BoolInstruction::makeOr(2, 0, 1),
                                    BoolInstruction::makeLeaf(3, 2),
                                    BoolInstruction::makeAnd(4, 2, 3)};
  AtomQueryTree   tree{3, 5, 5, 4};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_TRUE(result);
}

TEST(EvaluateBoolTreeTest, OrOfAnd_NitrogenMatchesAlso) {
  const AtomDataPacked target      = makeNitrogen();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask(), makeNitrogenMask(), makeAliphaticMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeLeaf(1, 1),
                                    BoolInstruction::makeOr(2, 0, 1),
                                    BoolInstruction::makeLeaf(3, 2),
                                    BoolInstruction::makeAnd(4, 2, 3)};
  AtomQueryTree   tree{3, 5, 5, 4};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_TRUE(result);
}

TEST(EvaluateBoolTreeTest, OrOfAnd_OxygenDoesNotMatch) {
  const AtomDataPacked target      = makeOxygen();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask(), makeNitrogenMask(), makeAliphaticMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeLeaf(1, 1),
                                    BoolInstruction::makeOr(2, 0, 1),
                                    BoolInstruction::makeLeaf(3, 2),
                                    BoolInstruction::makeAnd(4, 2, 3)};
  AtomQueryTree   tree{3, 5, 5, 4};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_FALSE(result);
}

TEST(EvaluateBoolTreeTest, ChainedOrs_CarbonOrNitrogenOrOxygen) {
  const AtomDataPacked carbon   = makeCarbon();
  const AtomDataPacked nitrogen = makeNitrogen();
  const AtomDataPacked oxygen   = makeOxygen();
  const BondTypeCounts bonds    = makeEmptyBonds();

  const AtomQueryMask  leafMasks[] = {makeCarbonMask(), makeNitrogenMask(), makeOxygenMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeLeaf(1, 1),
                                    BoolInstruction::makeOr(2, 0, 1),
                                    BoolInstruction::makeLeaf(3, 2),
                                    BoolInstruction::makeOr(4, 2, 3)};
  AtomQueryTree   tree{3, 5, 5, 4};

  EXPECT_TRUE(evaluateBoolTree(&carbon, &bonds, leafMasks, leafBonds, instructions, tree));
  EXPECT_TRUE(evaluateBoolTree(&nitrogen, &bonds, leafMasks, leafBonds, instructions, tree));
  EXPECT_TRUE(evaluateBoolTree(&oxygen, &bonds, leafMasks, leafBonds, instructions, tree));

  AtomDataPacked sulfur;
  sulfur.setAtomicNum(16);
  EXPECT_FALSE(evaluateBoolTree(&sulfur, &bonds, leafMasks, leafBonds, instructions, tree));
}

TEST(EvaluateBoolTreeTest, DoubleNegation) {
  const AtomDataPacked target      = makeCarbon();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds()};

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeNot(1, 0),
                                    BoolInstruction::makeNot(2, 1)};
  AtomQueryTree   tree{1, 3, 3, 2};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_TRUE(result);
}

TEST(EvaluateBoolTreeTest, DeMorgans_NotOrEqualsAndOfNots) {
  const AtomDataPacked oxygen      = makeOxygen();
  const BondTypeCounts bonds       = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask(), makeNitrogenMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds(), makeEmptyBonds()};

  BoolInstruction notOrInstructions[] = {BoolInstruction::makeLeaf(0, 0),
                                         BoolInstruction::makeLeaf(1, 1),
                                         BoolInstruction::makeOr(2, 0, 1),
                                         BoolInstruction::makeNot(3, 2)};
  AtomQueryTree   notOrTree{2, 4, 4, 3};

  BoolInstruction andNotInstructions[] = {BoolInstruction::makeLeaf(0, 0),
                                          BoolInstruction::makeNot(1, 0),
                                          BoolInstruction::makeLeaf(2, 1),
                                          BoolInstruction::makeNot(3, 2),
                                          BoolInstruction::makeAnd(4, 1, 3)};
  AtomQueryTree   andNotTree{2, 5, 5, 4};

  const bool notOrResult  = evaluateBoolTree(&oxygen, &bonds, leafMasks, leafBonds, notOrInstructions, notOrTree);
  const bool andNotResult = evaluateBoolTree(&oxygen, &bonds, leafMasks, leafBonds, andNotInstructions, andNotTree);

  EXPECT_EQ(notOrResult, andNotResult);
  EXPECT_TRUE(notOrResult);
}

// =============================================================================
// evaluateBoolTree Host Tests - Bond Matching
// =============================================================================

TEST(EvaluateBoolTreeTest, BondMatchFailsWhenInsufficient) {
  AtomDataPacked target;
  target.setAtomicNum(6);
  BondTypeCounts targetBonds{1, 0, 0, 0, 0};

  const AtomQueryMask leafMasks[] = {makeCarbonMask()};
  BondTypeCounts      leafBonds[] = {
    {2, 0, 0, 0, 0}
  };

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0)};
  AtomQueryTree   tree{1, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_FALSE(result);
}

TEST(EvaluateBoolTreeTest, BondMatchSucceedsWhenSufficient) {
  AtomDataPacked target;
  target.setAtomicNum(6);
  BondTypeCounts targetBonds{2, 1, 0, 0, 0};

  const AtomQueryMask leafMasks[] = {makeCarbonMask()};
  BondTypeCounts      leafBonds[] = {
    {2, 0, 0, 0, 0}
  };

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0)};
  AtomQueryTree   tree{1, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_TRUE(result);
}

TEST(EvaluateBoolTreeTest, AnyBondMatchesAnyType) {
  AtomDataPacked target;
  target.setAtomicNum(6);
  BondTypeCounts targetBonds{0, 0, 1, 0, 0};

  const AtomQueryMask leafMasks[] = {makeCarbonMask()};
  BondTypeCounts      leafBonds[] = {
    {0, 0, 0, 0, 1}
  };

  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0)};
  AtomQueryTree   tree{1, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_TRUE(result);
}

// =============================================================================
// Device Kernel for Testing
// =============================================================================

__global__ void evaluateBoolTreeKernel(const AtomDataPacked*  targets,
                                       const BondTypeCounts*  targetBonds,
                                       const AtomQueryMask*   leafMasks,
                                       const BondTypeCounts*  leafBonds,
                                       const BoolInstruction* instructions,
                                       const AtomQueryTree*   trees,
                                       int                    numTargets,
                                       int*                   results) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numTargets) {
    return;
  }
  results[idx] =
    evaluateBoolTree(&targets[idx], &targetBonds[idx], leafMasks, leafBonds, instructions, trees[0]) ? 1 : 0;
}

__global__ void evaluateBoolTreeWithRecursiveKernel(const AtomDataPacked*  targets,
                                                    const BondTypeCounts*  targetBonds,
                                                    const AtomQueryMask*   leafMasks,
                                                    const BondTypeCounts*  leafBonds,
                                                    const BoolInstruction* instructions,
                                                    const AtomQueryTree*   trees,
                                                    const uint32_t*        recursiveMatchBits,
                                                    int                    numTargets,
                                                    int*                   results) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numTargets) {
    return;
  }
  results[idx] = evaluateBoolTree(&targets[idx],
                                  &targetBonds[idx],
                                  leafMasks,
                                  leafBonds,
                                  instructions,
                                  trees[0],
                                  recursiveMatchBits[idx]) ?
                   1 :
                   0;
}

// =============================================================================
// evaluateBoolTree Device Tests
// =============================================================================

class BoolTreeDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override { stream_ = std::make_unique<ScopedStream>(); }

  std::unique_ptr<ScopedStream> stream_;
};

TEST_F(BoolTreeDeviceTest, SingleLeafOnDevice) {
  std::vector<AtomDataPacked>  targets      = {makeCarbon(), makeNitrogen(), makeOxygen()};
  std::vector<BondTypeCounts>  targetBonds  = {makeEmptyBonds(), makeEmptyBonds(), makeEmptyBonds()};
  std::vector<AtomQueryMask>   leafMasks    = {makeCarbonMask()};
  std::vector<BondTypeCounts>  leafBonds    = {makeEmptyBonds()};
  std::vector<BoolInstruction> instructions = {BoolInstruction::makeLeaf(0, 0)};
  std::vector<AtomQueryTree>   trees        = {
    {1, 1, 1, 0}
  };

  AsyncDeviceVector<AtomDataPacked>  targetsDev(targets.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  targetBondsDev(targetBonds.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryMask>   leafMasksDev(leafMasks.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  leafBondsDev(leafBonds.size(), stream_->stream());
  AsyncDeviceVector<BoolInstruction> instructionsDev(instructions.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryTree>   treesDev(trees.size(), stream_->stream());
  AsyncDeviceVector<int>             resultsDev(targets.size(), stream_->stream());

  targetsDev.copyFromHost(targets);
  targetBondsDev.copyFromHost(targetBonds);
  leafMasksDev.copyFromHost(leafMasks);
  leafBondsDev.copyFromHost(leafBonds);
  instructionsDev.copyFromHost(instructions);
  treesDev.copyFromHost(trees);

  evaluateBoolTreeKernel<<<1, 3, 0, stream_->stream()>>>(targetsDev.data(),
                                                         targetBondsDev.data(),
                                                         leafMasksDev.data(),
                                                         leafBondsDev.data(),
                                                         instructionsDev.data(),
                                                         treesDev.data(),
                                                         static_cast<int>(targets.size()),
                                                         resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(targets.size());
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream_->stream()));

  EXPECT_TRUE(results[0]);
  EXPECT_FALSE(results[1]);
  EXPECT_FALSE(results[2]);
}

TEST_F(BoolTreeDeviceTest, OrOnDevice) {
  std::vector<AtomDataPacked> targets     = {makeCarbon(), makeNitrogen(), makeOxygen()};
  std::vector<BondTypeCounts> targetBonds = {makeEmptyBonds(), makeEmptyBonds(), makeEmptyBonds()};
  std::vector<AtomQueryMask>  leafMasks   = {makeCarbonMask(), makeNitrogenMask()};
  std::vector<BondTypeCounts> leafBonds   = {makeEmptyBonds(), makeEmptyBonds()};

  std::vector<BoolInstruction> instructions = {BoolInstruction::makeLeaf(0, 0),
                                               BoolInstruction::makeLeaf(1, 1),
                                               BoolInstruction::makeOr(2, 0, 1)};
  std::vector<AtomQueryTree>   trees        = {
    {2, 3, 3, 2}
  };

  AsyncDeviceVector<AtomDataPacked>  targetsDev(targets.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  targetBondsDev(targetBonds.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryMask>   leafMasksDev(leafMasks.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  leafBondsDev(leafBonds.size(), stream_->stream());
  AsyncDeviceVector<BoolInstruction> instructionsDev(instructions.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryTree>   treesDev(trees.size(), stream_->stream());
  AsyncDeviceVector<int>             resultsDev(targets.size(), stream_->stream());

  targetsDev.copyFromHost(targets);
  targetBondsDev.copyFromHost(targetBonds);
  leafMasksDev.copyFromHost(leafMasks);
  leafBondsDev.copyFromHost(leafBonds);
  instructionsDev.copyFromHost(instructions);
  treesDev.copyFromHost(trees);

  evaluateBoolTreeKernel<<<1, 3, 0, stream_->stream()>>>(targetsDev.data(),
                                                         targetBondsDev.data(),
                                                         leafMasksDev.data(),
                                                         leafBondsDev.data(),
                                                         instructionsDev.data(),
                                                         treesDev.data(),
                                                         static_cast<int>(targets.size()),
                                                         resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(targets.size());
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream_->stream()));

  EXPECT_TRUE(results[0]);
  EXPECT_TRUE(results[1]);
  EXPECT_FALSE(results[2]);
}

TEST_F(BoolTreeDeviceTest, AndOnDevice) {
  std::vector<AtomDataPacked> targets     = {makeCarbon(), makeAromaticCarbon(), makeNitrogen()};
  std::vector<BondTypeCounts> targetBonds = {makeEmptyBonds(), makeEmptyBonds(), makeEmptyBonds()};
  std::vector<AtomQueryMask>  leafMasks   = {makeCarbonMask(), makeAliphaticMask()};
  std::vector<BondTypeCounts> leafBonds   = {makeEmptyBonds(), makeEmptyBonds()};

  std::vector<BoolInstruction> instructions = {BoolInstruction::makeLeaf(0, 0),
                                               BoolInstruction::makeLeaf(1, 1),
                                               BoolInstruction::makeAnd(2, 0, 1)};
  std::vector<AtomQueryTree>   trees        = {
    {2, 3, 3, 2}
  };

  AsyncDeviceVector<AtomDataPacked>  targetsDev(targets.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  targetBondsDev(targetBonds.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryMask>   leafMasksDev(leafMasks.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  leafBondsDev(leafBonds.size(), stream_->stream());
  AsyncDeviceVector<BoolInstruction> instructionsDev(instructions.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryTree>   treesDev(trees.size(), stream_->stream());
  AsyncDeviceVector<int>             resultsDev(targets.size(), stream_->stream());

  targetsDev.copyFromHost(targets);
  targetBondsDev.copyFromHost(targetBonds);
  leafMasksDev.copyFromHost(leafMasks);
  leafBondsDev.copyFromHost(leafBonds);
  instructionsDev.copyFromHost(instructions);
  treesDev.copyFromHost(trees);

  evaluateBoolTreeKernel<<<1, 3, 0, stream_->stream()>>>(targetsDev.data(),
                                                         targetBondsDev.data(),
                                                         leafMasksDev.data(),
                                                         leafBondsDev.data(),
                                                         instructionsDev.data(),
                                                         treesDev.data(),
                                                         static_cast<int>(targets.size()),
                                                         resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(targets.size());
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream_->stream()));

  EXPECT_TRUE(results[0]);
  EXPECT_FALSE(results[1]);
  EXPECT_FALSE(results[2]);
}

TEST_F(BoolTreeDeviceTest, NotOnDevice) {
  std::vector<AtomDataPacked> targets     = {makeCarbon(), makeNitrogen(), makeOxygen()};
  std::vector<BondTypeCounts> targetBonds = {makeEmptyBonds(), makeEmptyBonds(), makeEmptyBonds()};
  std::vector<AtomQueryMask>  leafMasks   = {makeCarbonMask()};
  std::vector<BondTypeCounts> leafBonds   = {makeEmptyBonds()};

  std::vector<BoolInstruction> instructions = {BoolInstruction::makeLeaf(0, 0), BoolInstruction::makeNot(1, 0)};
  std::vector<AtomQueryTree>   trees        = {
    {1, 2, 2, 1}
  };

  AsyncDeviceVector<AtomDataPacked>  targetsDev(targets.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  targetBondsDev(targetBonds.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryMask>   leafMasksDev(leafMasks.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  leafBondsDev(leafBonds.size(), stream_->stream());
  AsyncDeviceVector<BoolInstruction> instructionsDev(instructions.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryTree>   treesDev(trees.size(), stream_->stream());
  AsyncDeviceVector<int>             resultsDev(targets.size(), stream_->stream());

  targetsDev.copyFromHost(targets);
  targetBondsDev.copyFromHost(targetBonds);
  leafMasksDev.copyFromHost(leafMasks);
  leafBondsDev.copyFromHost(leafBonds);
  instructionsDev.copyFromHost(instructions);
  treesDev.copyFromHost(trees);

  evaluateBoolTreeKernel<<<1, 3, 0, stream_->stream()>>>(targetsDev.data(),
                                                         targetBondsDev.data(),
                                                         leafMasksDev.data(),
                                                         leafBondsDev.data(),
                                                         instructionsDev.data(),
                                                         treesDev.data(),
                                                         static_cast<int>(targets.size()),
                                                         resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(targets.size());
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream_->stream()));

  EXPECT_FALSE(results[0]);
  EXPECT_TRUE(results[1]);
  EXPECT_TRUE(results[2]);
}

TEST_F(BoolTreeDeviceTest, ComplexExpressionOnDevice) {
  std::vector<AtomDataPacked> targets     = {makeCarbon(), makeAromaticCarbon(), makeNitrogen(), makeOxygen()};
  std::vector<BondTypeCounts> targetBonds = {makeEmptyBonds(), makeEmptyBonds(), makeEmptyBonds(), makeEmptyBonds()};
  std::vector<AtomQueryMask>  leafMasks   = {makeCarbonMask(), makeNitrogenMask(), makeAromaticMask()};
  std::vector<BondTypeCounts> leafBonds   = {makeEmptyBonds(), makeEmptyBonds(), makeEmptyBonds()};

  std::vector<BoolInstruction> instructions = {BoolInstruction::makeLeaf(0, 0),
                                               BoolInstruction::makeLeaf(1, 1),
                                               BoolInstruction::makeOr(2, 0, 1),
                                               BoolInstruction::makeLeaf(3, 2),
                                               BoolInstruction::makeNot(4, 3),
                                               BoolInstruction::makeAnd(5, 2, 4)};
  std::vector<AtomQueryTree>   trees        = {
    {3, 6, 6, 5}
  };

  AsyncDeviceVector<AtomDataPacked>  targetsDev(targets.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  targetBondsDev(targetBonds.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryMask>   leafMasksDev(leafMasks.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  leafBondsDev(leafBonds.size(), stream_->stream());
  AsyncDeviceVector<BoolInstruction> instructionsDev(instructions.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryTree>   treesDev(trees.size(), stream_->stream());
  AsyncDeviceVector<int>             resultsDev(targets.size(), stream_->stream());

  targetsDev.copyFromHost(targets);
  targetBondsDev.copyFromHost(targetBonds);
  leafMasksDev.copyFromHost(leafMasks);
  leafBondsDev.copyFromHost(leafBonds);
  instructionsDev.copyFromHost(instructions);
  treesDev.copyFromHost(trees);

  evaluateBoolTreeKernel<<<1, 4, 0, stream_->stream()>>>(targetsDev.data(),
                                                         targetBondsDev.data(),
                                                         leafMasksDev.data(),
                                                         leafBondsDev.data(),
                                                         instructionsDev.data(),
                                                         treesDev.data(),
                                                         static_cast<int>(targets.size()),
                                                         resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(targets.size());
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream_->stream()));

  EXPECT_TRUE(results[0]);
  EXPECT_FALSE(results[1]);
  EXPECT_TRUE(results[2]);
  EXPECT_FALSE(results[3]);
}

// =============================================================================
// BoolInstruction::makeRecursiveMatch Tests
// =============================================================================

TEST(BoolInstructionTest, MakeRecursiveMatchSetsCorrectFields) {
  const BoolInstruction instr = BoolInstruction::makeRecursiveMatch(5, 3);

  EXPECT_EQ(instr.op, BoolOp::RecursiveMatch);
  EXPECT_EQ(instr.dst, 5);
  EXPECT_EQ(instr.auxArg, 3);  // pattern ID stored here
}

// =============================================================================
// RecursiveMatch Evaluation Tests
// =============================================================================

TEST_F(BoolTreeDeviceTest, RecursiveMatchChecksPatternBit) {
  std::vector<AtomDataPacked> targets(4);
  std::vector<BondTypeCounts> targetBonds(4);

  // Recursive match bits are now passed separately, not stored in AtomDataPacked
  std::vector<uint32_t> recursiveMatchBits = {
    1u << 0,                // target 0: pattern 0 bit set
    1u << 1,                // target 1: pattern 1 bit set
    (1u << 0) | (1u << 1),  // target 2: both patterns set
    0                       // target 3: no bits set
  };

  std::vector<AtomQueryMask>  leafMasks;
  std::vector<BondTypeCounts> leafBonds;

  std::vector<BoolInstruction> instructions = {BoolInstruction::makeRecursiveMatch(0, 0)};
  std::vector<AtomQueryTree>   trees        = {
    {0, 1, 1, 0}
  };

  AsyncDeviceVector<AtomDataPacked>  targetsDev(targets.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  targetBondsDev(targetBonds.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryMask>   leafMasksDev(1, stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  leafBondsDev(1, stream_->stream());
  AsyncDeviceVector<BoolInstruction> instructionsDev(instructions.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryTree>   treesDev(trees.size(), stream_->stream());
  AsyncDeviceVector<uint32_t>        recursiveMatchBitsDev(recursiveMatchBits.size(), stream_->stream());
  AsyncDeviceVector<int>             resultsDev(targets.size(), stream_->stream());

  targetsDev.copyFromHost(targets);
  targetBondsDev.copyFromHost(targetBonds);
  instructionsDev.copyFromHost(instructions);
  treesDev.copyFromHost(trees);
  recursiveMatchBitsDev.copyFromHost(recursiveMatchBits);

  evaluateBoolTreeWithRecursiveKernel<<<1, 4, 0, stream_->stream()>>>(targetsDev.data(),
                                                                      targetBondsDev.data(),
                                                                      leafMasksDev.data(),
                                                                      leafBondsDev.data(),
                                                                      instructionsDev.data(),
                                                                      treesDev.data(),
                                                                      recursiveMatchBitsDev.data(),
                                                                      static_cast<int>(targets.size()),
                                                                      resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(targets.size());
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream_->stream()));

  EXPECT_TRUE(results[0]) << "Target with pattern 0 bit set should match pattern 0";
  EXPECT_FALSE(results[1]) << "Target with only pattern 1 bit set should not match pattern 0";
  EXPECT_TRUE(results[2]) << "Target with both patterns should match pattern 0";
  EXPECT_FALSE(results[3]) << "Target with no patterns should not match";
}

TEST_F(BoolTreeDeviceTest, RecursiveMatchWithOr) {
  std::vector<AtomDataPacked> targets(4);
  std::vector<BondTypeCounts> targetBonds(4);

  // Recursive match bits are now passed separately
  std::vector<uint32_t> recursiveMatchBits = {
    1u << 0,                // target 0: pattern 0 bit set
    1u << 1,                // target 1: pattern 1 bit set
    (1u << 0) | (1u << 1),  // target 2: both patterns set
    0                       // target 3: no bits set
  };

  std::vector<AtomQueryMask>  leafMasks;
  std::vector<BondTypeCounts> leafBonds;

  std::vector<BoolInstruction> instructions = {BoolInstruction::makeRecursiveMatch(0, 0),
                                               BoolInstruction::makeRecursiveMatch(1, 1),
                                               BoolInstruction::makeOr(2, 0, 1)};
  std::vector<AtomQueryTree>   trees        = {
    {0, 3, 3, 2}
  };

  AsyncDeviceVector<AtomDataPacked>  targetsDev(targets.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  targetBondsDev(targetBonds.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryMask>   leafMasksDev(1, stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  leafBondsDev(1, stream_->stream());
  AsyncDeviceVector<BoolInstruction> instructionsDev(instructions.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryTree>   treesDev(trees.size(), stream_->stream());
  AsyncDeviceVector<uint32_t>        recursiveMatchBitsDev(recursiveMatchBits.size(), stream_->stream());
  AsyncDeviceVector<int>             resultsDev(targets.size(), stream_->stream());

  targetsDev.copyFromHost(targets);
  targetBondsDev.copyFromHost(targetBonds);
  instructionsDev.copyFromHost(instructions);
  treesDev.copyFromHost(trees);
  recursiveMatchBitsDev.copyFromHost(recursiveMatchBits);

  evaluateBoolTreeWithRecursiveKernel<<<1, 4, 0, stream_->stream()>>>(targetsDev.data(),
                                                                      targetBondsDev.data(),
                                                                      leafMasksDev.data(),
                                                                      leafBondsDev.data(),
                                                                      instructionsDev.data(),
                                                                      treesDev.data(),
                                                                      recursiveMatchBitsDev.data(),
                                                                      static_cast<int>(targets.size()),
                                                                      resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(targets.size());
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream_->stream()));

  EXPECT_TRUE(results[0]) << "Target with pattern 0 should match (pattern 0 OR pattern 1)";
  EXPECT_TRUE(results[1]) << "Target with pattern 1 should match (pattern 0 OR pattern 1)";
  EXPECT_TRUE(results[2]) << "Target with both patterns should match";
  EXPECT_FALSE(results[3]) << "Target with no patterns should not match";
}

TEST_F(BoolTreeDeviceTest, RecursiveMatchNegated) {
  std::vector<AtomDataPacked> targets(2);
  std::vector<BondTypeCounts> targetBonds(2);

  // Recursive match bits are now passed separately
  std::vector<uint32_t> recursiveMatchBits = {
    1u << 0,  // target 0: pattern 0 bit set
    0         // target 1: no bits set
  };

  std::vector<BoolInstruction> instructions = {BoolInstruction::makeRecursiveMatch(0, 0),
                                               BoolInstruction::makeNot(1, 0)};
  std::vector<AtomQueryTree>   trees        = {
    {0, 2, 2, 1}
  };

  AsyncDeviceVector<AtomDataPacked>  targetsDev(targets.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  targetBondsDev(targetBonds.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryMask>   leafMasksDev(1, stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  leafBondsDev(1, stream_->stream());
  AsyncDeviceVector<BoolInstruction> instructionsDev(instructions.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryTree>   treesDev(trees.size(), stream_->stream());
  AsyncDeviceVector<uint32_t>        recursiveMatchBitsDev(recursiveMatchBits.size(), stream_->stream());
  AsyncDeviceVector<int>             resultsDev(targets.size(), stream_->stream());

  targetsDev.copyFromHost(targets);
  targetBondsDev.copyFromHost(targetBonds);
  instructionsDev.copyFromHost(instructions);
  treesDev.copyFromHost(trees);
  recursiveMatchBitsDev.copyFromHost(recursiveMatchBits);

  evaluateBoolTreeWithRecursiveKernel<<<1, 2, 0, stream_->stream()>>>(targetsDev.data(),
                                                                      targetBondsDev.data(),
                                                                      leafMasksDev.data(),
                                                                      leafBondsDev.data(),
                                                                      instructionsDev.data(),
                                                                      treesDev.data(),
                                                                      recursiveMatchBitsDev.data(),
                                                                      static_cast<int>(targets.size()),
                                                                      resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(targets.size());
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream_->stream()));

  EXPECT_FALSE(results[0]) << "Target with pattern should NOT match negated recursive pattern";
  EXPECT_TRUE(results[1]) << "Target without pattern should match negated recursive pattern";
}

// =============================================================================
// BoolInstruction Factory Tests - Comparison Operations
// =============================================================================

TEST(BoolInstructionTest, MakeGreaterThanSetsCorrectFields) {
  const BoolInstruction instr = BoolInstruction::makeGreaterThan(2, CompareField::Degree, 3);

  EXPECT_EQ(instr.op, BoolOp::GreaterThan);
  EXPECT_EQ(instr.dst, 2);
  EXPECT_EQ(instr.src1, static_cast<uint8_t>(CompareField::Degree));
  EXPECT_EQ(instr.src2, 3);
}

TEST(BoolInstructionTest, MakeLessEqualSetsCorrectFields) {
  const BoolInstruction instr = BoolInstruction::makeLessEqual(4, CompareField::NumRings, 5);

  EXPECT_EQ(instr.op, BoolOp::LessEqual);
  EXPECT_EQ(instr.dst, 4);
  EXPECT_EQ(instr.src1, static_cast<uint8_t>(CompareField::NumRings));
  EXPECT_EQ(instr.src2, 5);
}

TEST(BoolInstructionTest, MakeGreaterEqualSetsCorrectFields) {
  const BoolInstruction instr = BoolInstruction::makeGreaterEqual(1, CompareField::TotalValence, 2);

  EXPECT_EQ(instr.op, BoolOp::GreaterEqual);
  EXPECT_EQ(instr.dst, 1);
  EXPECT_EQ(instr.src1, static_cast<uint8_t>(CompareField::TotalValence));
  EXPECT_EQ(instr.src2, 2);
}

TEST(BoolInstructionTest, MakeRangeSetsCorrectFields) {
  const BoolInstruction instr = BoolInstruction::makeRange(3, CompareField::MinRingSize, 5, 7);

  EXPECT_EQ(instr.op, BoolOp::Range);
  EXPECT_EQ(instr.dst, 3);
  EXPECT_EQ(instr.src1, static_cast<uint8_t>(CompareField::MinRingSize));
  EXPECT_EQ(instr.src2, 5);    // minVal
  EXPECT_EQ(instr.auxArg, 7);  // maxVal
}

// =============================================================================
// evaluateBoolTree Host Tests - Empty Tree (Edge Case)
// =============================================================================

TEST(EvaluateBoolTreeTest, EmptyTreeMatchesAnyAtom) {
  const AtomDataPacked target      = makeCarbon();
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds()};

  // Empty tree has no instructions
  AtomQueryTree tree{1, 0, 0, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, nullptr, tree);
  EXPECT_TRUE(result);
}

TEST(EvaluateBoolTreeTest, EmptyTreeWithNoLeavesMatchesAny) {
  const AtomDataPacked target      = makeNitrogen();
  const BondTypeCounts targetBonds = makeEmptyBonds();

  AtomQueryTree tree{0, 0, 0, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, nullptr, tree);
  EXPECT_TRUE(result);
}

TEST(EvaluateBoolTreeTest, EmptyTreeChecksBondsWhenLeafPresent) {
  AtomDataPacked target;
  target.setAtomicNum(6);
  BondTypeCounts targetBonds{1, 0, 0, 0, 0};

  const AtomQueryMask leafMasks[] = {makeCarbonMask()};
  BondTypeCounts      leafBonds[] = {
    {2, 0, 0, 0, 0}
  };

  // Empty tree but has leaf for bond count checking
  AtomQueryTree tree{1, 0, 0, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, nullptr, tree);
  EXPECT_FALSE(result) << "Empty tree should still check bond counts via leaf 0";
}

// =============================================================================
// evaluateBoolTree Host Tests - Comparison Operations
// =============================================================================

namespace {

AtomDataPacked makeAtomWithDegree(uint8_t degree) {
  AtomDataPacked atom;
  atom.setAtomicNum(6);
  atom.setDegree(degree);
  return atom;
}

AtomDataPacked makeAtomWithMinRingSize(uint8_t ringSize) {
  AtomDataPacked atom;
  atom.setAtomicNum(6);
  atom.setMinRingSize(ringSize);
  return atom;
}

}  // namespace

TEST(EvaluateBoolTreeTest, GreaterThanDegreeTrue) {
  const AtomDataPacked target      = makeAtomWithDegree(4);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  BoolInstruction instructions[] = {BoolInstruction::makeGreaterThan(0, CompareField::Degree, 3)};
  AtomQueryTree   tree{0, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_TRUE(result) << "degree 4 > 3 should be true";
}

TEST(EvaluateBoolTreeTest, GreaterThanDegreeFalse) {
  const AtomDataPacked target      = makeAtomWithDegree(3);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  BoolInstruction instructions[] = {BoolInstruction::makeGreaterThan(0, CompareField::Degree, 3)};
  AtomQueryTree   tree{0, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_FALSE(result) << "degree 3 > 3 should be false";
}

TEST(EvaluateBoolTreeTest, GreaterThanDegreeBoundary) {
  const AtomDataPacked target      = makeAtomWithDegree(2);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  BoolInstruction instructions[] = {BoolInstruction::makeGreaterThan(0, CompareField::Degree, 3)};
  AtomQueryTree   tree{0, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_FALSE(result) << "degree 2 > 3 should be false";
}

TEST(EvaluateBoolTreeTest, LessEqualDegreeTrue) {
  const AtomDataPacked target      = makeAtomWithDegree(3);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  BoolInstruction instructions[] = {BoolInstruction::makeLessEqual(0, CompareField::Degree, 3)};
  AtomQueryTree   tree{0, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_TRUE(result) << "degree 3 <= 3 should be true";
}

TEST(EvaluateBoolTreeTest, LessEqualDegreeTrueWhenLess) {
  const AtomDataPacked target      = makeAtomWithDegree(2);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  BoolInstruction instructions[] = {BoolInstruction::makeLessEqual(0, CompareField::Degree, 3)};
  AtomQueryTree   tree{0, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_TRUE(result) << "degree 2 <= 3 should be true";
}

TEST(EvaluateBoolTreeTest, LessEqualDegreeFalse) {
  const AtomDataPacked target      = makeAtomWithDegree(4);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  BoolInstruction instructions[] = {BoolInstruction::makeLessEqual(0, CompareField::Degree, 3)};
  AtomQueryTree   tree{0, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_FALSE(result) << "degree 4 <= 3 should be false";
}

TEST(EvaluateBoolTreeTest, GreaterEqualDegreeTrue) {
  const AtomDataPacked target      = makeAtomWithDegree(3);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  BoolInstruction instructions[] = {BoolInstruction::makeGreaterEqual(0, CompareField::Degree, 3)};
  AtomQueryTree   tree{0, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_TRUE(result) << "degree 3 >= 3 should be true";
}

TEST(EvaluateBoolTreeTest, GreaterEqualDegreeTrueWhenGreater) {
  const AtomDataPacked target      = makeAtomWithDegree(4);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  BoolInstruction instructions[] = {BoolInstruction::makeGreaterEqual(0, CompareField::Degree, 3)};
  AtomQueryTree   tree{0, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_TRUE(result) << "degree 4 >= 3 should be true";
}

TEST(EvaluateBoolTreeTest, GreaterEqualDegreeFalse) {
  const AtomDataPacked target      = makeAtomWithDegree(2);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  BoolInstruction instructions[] = {BoolInstruction::makeGreaterEqual(0, CompareField::Degree, 3)};
  AtomQueryTree   tree{0, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_FALSE(result) << "degree 2 >= 3 should be false";
}

TEST(EvaluateBoolTreeTest, RangeInclusive) {
  const AtomDataPacked target      = makeAtomWithMinRingSize(5);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  BoolInstruction instructions[] = {BoolInstruction::makeRange(0, CompareField::MinRingSize, 5, 7)};
  AtomQueryTree   tree{0, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_TRUE(result) << "minRingSize 5 in [5, 7] should be true";
}

TEST(EvaluateBoolTreeTest, RangeInclusiveUpperBound) {
  const AtomDataPacked target      = makeAtomWithMinRingSize(7);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  BoolInstruction instructions[] = {BoolInstruction::makeRange(0, CompareField::MinRingSize, 5, 7)};
  AtomQueryTree   tree{0, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_TRUE(result) << "minRingSize 7 in [5, 7] should be true";
}

TEST(EvaluateBoolTreeTest, RangeMiddleValue) {
  const AtomDataPacked target      = makeAtomWithMinRingSize(6);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  BoolInstruction instructions[] = {BoolInstruction::makeRange(0, CompareField::MinRingSize, 5, 7)};
  AtomQueryTree   tree{0, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_TRUE(result) << "minRingSize 6 in [5, 7] should be true";
}

TEST(EvaluateBoolTreeTest, RangeBelowMin) {
  const AtomDataPacked target      = makeAtomWithMinRingSize(4);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  BoolInstruction instructions[] = {BoolInstruction::makeRange(0, CompareField::MinRingSize, 5, 7)};
  AtomQueryTree   tree{0, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_FALSE(result) << "minRingSize 4 in [5, 7] should be false";
}

TEST(EvaluateBoolTreeTest, RangeAboveMax) {
  const AtomDataPacked target      = makeAtomWithMinRingSize(8);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  BoolInstruction instructions[] = {BoolInstruction::makeRange(0, CompareField::MinRingSize, 5, 7)};
  AtomQueryTree   tree{0, 1, 1, 0};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_FALSE(result) << "minRingSize 8 in [5, 7] should be false";
}

TEST(EvaluateBoolTreeTest, ComparisonWithDifferentFields) {
  AtomDataPacked target;
  target.setAtomicNum(6);
  target.setNumRings(2);
  target.setTotalValence(4);
  const BondTypeCounts targetBonds = makeEmptyBonds();

  // numRings > 1 AND totalValence >= 4
  BoolInstruction instructions[] = {BoolInstruction::makeGreaterThan(0, CompareField::NumRings, 1),
                                    BoolInstruction::makeGreaterEqual(1, CompareField::TotalValence, 4),
                                    BoolInstruction::makeAnd(2, 0, 1)};
  AtomQueryTree   tree{0, 3, 3, 2};

  const bool result = evaluateBoolTree(&target, &targetBonds, nullptr, nullptr, instructions, tree);
  EXPECT_TRUE(result) << "numRings 2 > 1 AND totalValence 4 >= 4 should be true";
}

TEST(EvaluateBoolTreeTest, ComparisonOrLeaf) {
  AtomDataPacked target;
  target.setAtomicNum(7);  // nitrogen
  target.setDegree(2);
  const BondTypeCounts targetBonds = makeEmptyBonds();
  const AtomQueryMask  leafMasks[] = {makeCarbonMask()};
  const BondTypeCounts leafBonds[] = {makeEmptyBonds()};

  // Carbon OR degree > 1
  BoolInstruction instructions[] = {BoolInstruction::makeLeaf(0, 0),
                                    BoolInstruction::makeGreaterThan(1, CompareField::Degree, 1),
                                    BoolInstruction::makeOr(2, 0, 1)};
  AtomQueryTree   tree{1, 3, 3, 2};

  const bool result = evaluateBoolTree(&target, &targetBonds, leafMasks, leafBonds, instructions, tree);
  EXPECT_TRUE(result) << "Not carbon but degree 2 > 1, so OR should be true";
}

// =============================================================================
// evaluateBoolTree Device Tests - Comparison Operations
// =============================================================================

__global__ void evaluateComparisonKernel(const AtomDataPacked*  targets,
                                         const BondTypeCounts*  targetBonds,
                                         const BoolInstruction* instructions,
                                         const AtomQueryTree*   trees,
                                         int                    numTargets,
                                         int*                   results) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numTargets) {
    return;
  }
  results[idx] =
    evaluateBoolTree<false>(&targets[idx], &targetBonds[idx], nullptr, nullptr, instructions, trees[0]) ? 1 : 0;
}

TEST_F(BoolTreeDeviceTest, GreaterThanOnDevice) {
  std::vector<AtomDataPacked> targets = {makeAtomWithDegree(2), makeAtomWithDegree(3), makeAtomWithDegree(4)};
  std::vector<BondTypeCounts> targetBonds(3);

  std::vector<BoolInstruction> instructions = {BoolInstruction::makeGreaterThan(0, CompareField::Degree, 3)};
  std::vector<AtomQueryTree>   trees        = {
    {0, 1, 1, 0}
  };

  AsyncDeviceVector<AtomDataPacked>  targetsDev(targets.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  targetBondsDev(targetBonds.size(), stream_->stream());
  AsyncDeviceVector<BoolInstruction> instructionsDev(instructions.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryTree>   treesDev(trees.size(), stream_->stream());
  AsyncDeviceVector<int>             resultsDev(targets.size(), stream_->stream());

  targetsDev.copyFromHost(targets);
  targetBondsDev.copyFromHost(targetBonds);
  instructionsDev.copyFromHost(instructions);
  treesDev.copyFromHost(trees);

  evaluateComparisonKernel<<<1, 3, 0, stream_->stream()>>>(targetsDev.data(),
                                                           targetBondsDev.data(),
                                                           instructionsDev.data(),
                                                           treesDev.data(),
                                                           static_cast<int>(targets.size()),
                                                           resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(targets.size());
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream_->stream()));

  EXPECT_FALSE(results[0]) << "degree 2 > 3 should be false";
  EXPECT_FALSE(results[1]) << "degree 3 > 3 should be false";
  EXPECT_TRUE(results[2]) << "degree 4 > 3 should be true";
}

TEST_F(BoolTreeDeviceTest, LessEqualOnDevice) {
  std::vector<AtomDataPacked> targets = {makeAtomWithDegree(2), makeAtomWithDegree(3), makeAtomWithDegree(4)};
  std::vector<BondTypeCounts> targetBonds(3);

  std::vector<BoolInstruction> instructions = {BoolInstruction::makeLessEqual(0, CompareField::Degree, 3)};
  std::vector<AtomQueryTree>   trees        = {
    {0, 1, 1, 0}
  };

  AsyncDeviceVector<AtomDataPacked>  targetsDev(targets.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  targetBondsDev(targetBonds.size(), stream_->stream());
  AsyncDeviceVector<BoolInstruction> instructionsDev(instructions.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryTree>   treesDev(trees.size(), stream_->stream());
  AsyncDeviceVector<int>             resultsDev(targets.size(), stream_->stream());

  targetsDev.copyFromHost(targets);
  targetBondsDev.copyFromHost(targetBonds);
  instructionsDev.copyFromHost(instructions);
  treesDev.copyFromHost(trees);

  evaluateComparisonKernel<<<1, 3, 0, stream_->stream()>>>(targetsDev.data(),
                                                           targetBondsDev.data(),
                                                           instructionsDev.data(),
                                                           treesDev.data(),
                                                           static_cast<int>(targets.size()),
                                                           resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(targets.size());
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream_->stream()));

  EXPECT_TRUE(results[0]) << "degree 2 <= 3 should be true";
  EXPECT_TRUE(results[1]) << "degree 3 <= 3 should be true";
  EXPECT_FALSE(results[2]) << "degree 4 <= 3 should be false";
}

TEST_F(BoolTreeDeviceTest, GreaterEqualOnDevice) {
  std::vector<AtomDataPacked> targets = {makeAtomWithDegree(2), makeAtomWithDegree(3), makeAtomWithDegree(4)};
  std::vector<BondTypeCounts> targetBonds(3);

  std::vector<BoolInstruction> instructions = {BoolInstruction::makeGreaterEqual(0, CompareField::Degree, 3)};
  std::vector<AtomQueryTree>   trees        = {
    {0, 1, 1, 0}
  };

  AsyncDeviceVector<AtomDataPacked>  targetsDev(targets.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  targetBondsDev(targetBonds.size(), stream_->stream());
  AsyncDeviceVector<BoolInstruction> instructionsDev(instructions.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryTree>   treesDev(trees.size(), stream_->stream());
  AsyncDeviceVector<int>             resultsDev(targets.size(), stream_->stream());

  targetsDev.copyFromHost(targets);
  targetBondsDev.copyFromHost(targetBonds);
  instructionsDev.copyFromHost(instructions);
  treesDev.copyFromHost(trees);

  evaluateComparisonKernel<<<1, 3, 0, stream_->stream()>>>(targetsDev.data(),
                                                           targetBondsDev.data(),
                                                           instructionsDev.data(),
                                                           treesDev.data(),
                                                           static_cast<int>(targets.size()),
                                                           resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(targets.size());
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream_->stream()));

  EXPECT_FALSE(results[0]) << "degree 2 >= 3 should be false";
  EXPECT_TRUE(results[1]) << "degree 3 >= 3 should be true";
  EXPECT_TRUE(results[2]) << "degree 4 >= 3 should be true";
}

TEST_F(BoolTreeDeviceTest, RangeOnDevice) {
  std::vector<AtomDataPacked> targets = {makeAtomWithMinRingSize(4),
                                         makeAtomWithMinRingSize(5),
                                         makeAtomWithMinRingSize(6),
                                         makeAtomWithMinRingSize(7),
                                         makeAtomWithMinRingSize(8)};
  std::vector<BondTypeCounts> targetBonds(5);

  std::vector<BoolInstruction> instructions = {BoolInstruction::makeRange(0, CompareField::MinRingSize, 5, 7)};
  std::vector<AtomQueryTree>   trees        = {
    {0, 1, 1, 0}
  };

  AsyncDeviceVector<AtomDataPacked>  targetsDev(targets.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  targetBondsDev(targetBonds.size(), stream_->stream());
  AsyncDeviceVector<BoolInstruction> instructionsDev(instructions.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryTree>   treesDev(trees.size(), stream_->stream());
  AsyncDeviceVector<int>             resultsDev(targets.size(), stream_->stream());

  targetsDev.copyFromHost(targets);
  targetBondsDev.copyFromHost(targetBonds);
  instructionsDev.copyFromHost(instructions);
  treesDev.copyFromHost(trees);

  evaluateComparisonKernel<<<1, 5, 0, stream_->stream()>>>(targetsDev.data(),
                                                           targetBondsDev.data(),
                                                           instructionsDev.data(),
                                                           treesDev.data(),
                                                           static_cast<int>(targets.size()),
                                                           resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(targets.size());
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream_->stream()));

  EXPECT_FALSE(results[0]) << "minRingSize 4 in [5, 7] should be false";
  EXPECT_TRUE(results[1]) << "minRingSize 5 in [5, 7] should be true";
  EXPECT_TRUE(results[2]) << "minRingSize 6 in [5, 7] should be true";
  EXPECT_TRUE(results[3]) << "minRingSize 7 in [5, 7] should be true";
  EXPECT_FALSE(results[4]) << "minRingSize 8 in [5, 7] should be false";
}

TEST_F(BoolTreeDeviceTest, ComparisonCombinedWithLeafOnDevice) {
  std::vector<AtomDataPacked> targets(3);
  targets[0].setAtomicNum(6);  // carbon with degree 2
  targets[0].setDegree(2);
  targets[1].setAtomicNum(7);  // nitrogen with degree 3
  targets[1].setDegree(3);
  targets[2].setAtomicNum(8);  // oxygen with degree 1
  targets[2].setDegree(1);

  std::vector<BondTypeCounts> targetBonds = {makeEmptyBonds(), makeEmptyBonds(), makeEmptyBonds()};
  std::vector<AtomQueryMask>  leafMasks   = {makeCarbonMask()};
  std::vector<BondTypeCounts> leafBonds   = {makeEmptyBonds()};

  // Carbon OR degree > 2
  std::vector<BoolInstruction> instructions = {BoolInstruction::makeLeaf(0, 0),
                                               BoolInstruction::makeGreaterThan(1, CompareField::Degree, 2),
                                               BoolInstruction::makeOr(2, 0, 1)};
  std::vector<AtomQueryTree>   trees        = {
    {1, 3, 3, 2}
  };

  AsyncDeviceVector<AtomDataPacked>  targetsDev(targets.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  targetBondsDev(targetBonds.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryMask>   leafMasksDev(leafMasks.size(), stream_->stream());
  AsyncDeviceVector<BondTypeCounts>  leafBondsDev(leafBonds.size(), stream_->stream());
  AsyncDeviceVector<BoolInstruction> instructionsDev(instructions.size(), stream_->stream());
  AsyncDeviceVector<AtomQueryTree>   treesDev(trees.size(), stream_->stream());
  AsyncDeviceVector<int>             resultsDev(targets.size(), stream_->stream());

  targetsDev.copyFromHost(targets);
  targetBondsDev.copyFromHost(targetBonds);
  leafMasksDev.copyFromHost(leafMasks);
  leafBondsDev.copyFromHost(leafBonds);
  instructionsDev.copyFromHost(instructions);
  treesDev.copyFromHost(trees);

  evaluateBoolTreeKernel<<<1, 3, 0, stream_->stream()>>>(targetsDev.data(),
                                                         targetBondsDev.data(),
                                                         leafMasksDev.data(),
                                                         leafBondsDev.data(),
                                                         instructionsDev.data(),
                                                         treesDev.data(),
                                                         static_cast<int>(targets.size()),
                                                         resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(targets.size());
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream_->stream()));

  EXPECT_TRUE(results[0]) << "Carbon matches leaf, so OR is true";
  EXPECT_TRUE(results[1]) << "Nitrogen with degree 3 > 2, so OR is true";
  EXPECT_FALSE(results[2]) << "Oxygen with degree 1 <= 2, and not carbon, so OR is false";
}
