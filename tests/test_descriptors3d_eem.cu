// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>

#include "src/descriptors3d_eem.cuh"

using namespace nvMolKit::descriptors3d_detail;

TEST(Descriptors3DEemScratch, SharedMemoryNeedsNoBudgetLimit) {
  EXPECT_EQ(eemBlockCount(0, 20, 0), 0);
  EXPECT_EQ(eemBlockCount(1, 20, 0), 1);
  EXPECT_EQ(eemBlockCount(1000, 20, 0), 320);
}

TEST(Descriptors3DEemScratch, RespectsConformerAndOccupancyLimits) {
  EXPECT_EQ(eemBlockCount(0, 20, 1024), 0);
  EXPECT_EQ(eemBlockCount(1, 20, 1024), 1);
  EXPECT_EQ(eemBlockCount(100, 20, 1024), 100);
  EXPECT_EQ(eemBlockCount(1000, 20, 1024), 320);
}

TEST(Descriptors3DEemScratch, BudgetBoundaryAndSingleOversizedWorkspace) {
  EXPECT_EQ(eemBlockCount(1000, 20, kEemScratchBudgetBytes / 2), 2);
  EXPECT_EQ(eemBlockCount(1000, 20, kEemScratchBudgetBytes / 2 + 1), 1);
  EXPECT_EQ(eemBlockCount(1000, 20, kEemScratchBudgetBytes), 1);
  EXPECT_EQ(eemBlockCount(1000, 20, kEemScratchBudgetBytes + 1), 1);
  EXPECT_EQ(eemBlockCount(1000, 20, EemWorkspace<double>::bytes(nvMolKit::kMaxEemAtoms)), 1);
}

TEST(Descriptors3DEemScratch, BothPrecisionsStayWithinBudget) {
  for (const int numAtoms : {120, 256, 512, 1024}) {
    for (const int64_t stride : {EemWorkspace<float>::bytes(numAtoms), EemWorkspace<double>::bytes(numAtoms)}) {
      const int blocks = eemBlockCount(10000, 100, stride);
      EXPECT_GE(blocks, 1);
      EXPECT_LE(blocks, 1600);
      EXPECT_LE(blocks * stride, kEemScratchBudgetBytes);
      if (blocks < 1600) {
        EXPECT_GT((blocks + 1) * stride, kEemScratchBudgetBytes);
      }
    }
  }
}
