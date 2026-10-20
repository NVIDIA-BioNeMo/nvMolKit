// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <limits>

#include "src/aap_threads.h"

TEST(AapThreads, AutoPreservesWorkloadAndHardwareLimits) {
  EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(0, -1, 16), 1);
  EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(1, -1, 16), 1);
  EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(63, -1, 16), 1);
  EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(64, -1, 16), 1);
  EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(127, -1, 16), 1);
  EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(128, -1, 16), 2);
  EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(191, -1, 16), 2);
  EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(192, -1, 16), 3);
  EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(4096, -1, 0), 1);
  EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(4096, -1, 1), 1);
  EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(4096, -1, 16), 16);
  EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(4096, -1, 128), 64);
}

TEST(AapThreads, ExplicitBudgetIncludesCallerAndLimitsLargeInputs) {
  for (const int budget : {1, 2, 3, 8, std::numeric_limits<int>::max()}) {
    EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(0, budget, 128), 1);
    EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(127, budget, 128), 1);
    EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(128, budget, 128), std::min(2, budget));
    EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(std::numeric_limits<std::size_t>::max(), budget, 128), budget);
  }
  EXPECT_EQ(nvMolKit::detail::aapDescriptorThreadCount(4096, 4, 1), 4);
}

TEST(AapThreads, RejectsInvalidBudgets) {
  for (const int budget : {0, -2, -100, std::numeric_limits<int>::min()}) {
    EXPECT_THROW(nvMolKit::detail::aapDescriptorThreadCount(0, budget, 128), std::invalid_argument);
    EXPECT_THROW(nvMolKit::detail::aapDescriptorThreadCount(4096, budget, 128), std::invalid_argument);
  }
}

TEST(AapThreads, SharedValidationAcceptsAutoAndPositiveBudgets) {
  for (const int budget : {-1, 1, 2, std::numeric_limits<int>::max()}) {
    EXPECT_NO_THROW(nvMolKit::detail::validateAapThreadCount(budget));
  }
}

TEST(AapThreads, SharedValidationRejectsInvalidBudgets) {
  for (const int budget : {0, -2, -100, std::numeric_limits<int>::min()}) {
    EXPECT_THROW(nvMolKit::detail::validateAapThreadCount(budget), std::invalid_argument);
  }
}
