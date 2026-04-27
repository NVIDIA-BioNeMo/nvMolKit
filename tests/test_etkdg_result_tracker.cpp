// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <thread>
#include <unordered_set>
#include <vector>

#include "src/etkdg_impl.h"

using namespace nvMolKit::detail;

class SchedulerTest : public ::testing::Test {
 protected:
  int numMols       = 5;
  int confsPerMol   = 3;
  int maxIterations = 2;
};

// Test constructor validation with invalid parameters
TEST_F(SchedulerTest, ConstructorInvalidParameters) {
  EXPECT_THROW(Scheduler(-1, confsPerMol, maxIterations), std::invalid_argument);
  EXPECT_THROW(Scheduler(numMols, -1, maxIterations), std::invalid_argument);
  EXPECT_THROW(Scheduler(numMols, confsPerMol, -1), std::invalid_argument);
  EXPECT_THROW(Scheduler(0, confsPerMol, maxIterations), std::invalid_argument);
  EXPECT_THROW(Scheduler(numMols, 0, maxIterations), std::invalid_argument);
  EXPECT_THROW(Scheduler(numMols, confsPerMol, 0), std::invalid_argument);
}

TEST_F(SchedulerTest, BasicDispatchOversubscribe) {
  Scheduler tracker(numMols, confsPerMol, maxIterations);
  auto      molIds = tracker.dispatch(numMols);
  ASSERT_THAT(molIds, ::testing::ElementsAreArray({0, 0, 0, 1, 1}));
  auto molIds2 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds2, ::testing::ElementsAreArray({1, 2, 2, 2, 3}));
  auto molIds3 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds3, ::testing::ElementsAreArray({3, 3, 4, 4, 4}));
  // oversubscribe in the same round robin format.
  molIds = tracker.dispatch(numMols);
  ASSERT_THAT(molIds, ::testing::ElementsAreArray({0, 0, 0, 1, 1}));
  molIds2 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds2, ::testing::ElementsAreArray({1, 2, 2, 2, 3}));
  molIds3 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds3, ::testing::ElementsAreArray({3, 3, 4, 4, 4}));

  // We've dispatched max attempts.
  auto molIds4 = tracker.dispatch(numMols);
  EXPECT_THAT(molIds4, testing::IsEmpty());
}

TEST_F(SchedulerTest, BasicDispatchFullComplete) {
  Scheduler tracker(numMols, confsPerMol, maxIterations);
  auto      molIds = tracker.dispatch(numMols);
  ASSERT_THAT(molIds, ::testing::ElementsAreArray({0, 0, 0, 1, 1}));
  auto molIds2 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds2, ::testing::ElementsAreArray({1, 2, 2, 2, 3}));
  auto molIds3 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds3, ::testing::ElementsAreArray({3, 3, 4, 4, 4}));

  std::vector<int16_t> goodResults = {1, 1, 1, 1, 1};
  // Out of order should be fine.
  tracker.record(molIds3, goodResults);
  tracker.record(molIds, goodResults);
  tracker.record(molIds2, goodResults);

  auto molIds4 = tracker.dispatch(numMols);
  EXPECT_THAT(molIds4, testing::IsEmpty());
}

// Test basic dispatch functionality
TEST_F(SchedulerTest, BasicDispatchPartialCompleteNoErrors) {
  Scheduler tracker(numMols, confsPerMol, maxIterations);

  // First dispatch should return molecule IDs 0-4 (all unique molecules)
  auto molIds = tracker.dispatch(numMols);
  ASSERT_THAT(molIds, ::testing::ElementsAreArray({0, 0, 0, 1, 1}));
  auto molIds2 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds2, ::testing::ElementsAreArray({1, 2, 2, 2, 3}));

  std::vector<int16_t> goodResults = {1, 1, 1, 1, 1};
  tracker.record(molIds, goodResults);
  // Here we've completed all of 0, 2 of 1. The next dispatch should still be unstarted runs.
  auto molIds3 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds3, ::testing::ElementsAreArray({3, 3, 4, 4, 4}));

  // Second round, we skip 0 since it's finished already.
  auto molIds4 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds4, ::testing::ElementsAreArray({1, 1, 1, 2, 2}));

  tracker.record(molIds2, goodResults);
  tracker.record(molIds3, goodResults);

  // We've recorded passes on everything
  auto molIds5 = tracker.dispatch(numMols);
  EXPECT_THAT(molIds5, testing::IsEmpty());
}

// Test basic dispatch functionality
TEST_F(SchedulerTest, BasicDispatchFullWithSomeFails) {
  Scheduler tracker(numMols, confsPerMol, maxIterations);

  // First dispatch should return molecule IDs 0-4 (all unique molecules)
  auto molIds = tracker.dispatch(numMols);
  ASSERT_THAT(molIds, ::testing::ElementsAreArray({0, 0, 0, 1, 1}));
  auto molIds2 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds2, ::testing::ElementsAreArray({1, 2, 2, 2, 3}));
  auto molIds3 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds3, ::testing::ElementsAreArray({3, 3, 4, 4, 4}));
  std::vector<int16_t>       goodResults  = {3, 2, 1, 4, 0};
  const std::vector<int16_t> mixedResults = {-1, -1, 0, 1, 2};
  tracker.record(molIds, goodResults);
  tracker.record(molIds2, mixedResults);
  tracker.record(molIds3, goodResults);

  // Systems 0, 3 and 4 are done.
  auto molIds4 = tracker.dispatch(numMols);
  EXPECT_THAT(molIds4, ::testing::ElementsAreArray({1, 1, 1, 2, 2}));
  auto molIds5 = tracker.dispatch(numMols);
  EXPECT_THAT(molIds5, ::testing::ElementsAreArray({2}));
  auto molIds6 = tracker.dispatch(numMols);
  EXPECT_THAT(molIds6, ::testing::IsEmpty());
}

// Test dispatch with batch size larger than number of molecules
TEST_F(SchedulerTest, DispatchLargeBatchSize) {
  Scheduler tracker(2, 2, 4);

  constexpr int largeBatchSize = 100;
  auto          molIds         = tracker.dispatch(largeBatchSize);
  EXPECT_THAT(molIds, ::testing::ElementsAreArray({0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1}));
}

// Test record validation - mismatched vector sizes
TEST_F(SchedulerTest, RecordMismatchedSizes) {
  Scheduler tracker(numMols, confsPerMol, maxIterations);

  std::vector<int>     molIds  = {0, 1, 2};
  std::vector<int16_t> results = {1, 1};  // Wrong size

  EXPECT_THROW(tracker.record(molIds, results), std::invalid_argument);
}

// Test record validation - invalid molecule ID
TEST_F(SchedulerTest, RecordInvalidMoleculeId) {
  Scheduler tracker(numMols, confsPerMol, maxIterations);

  std::vector<int>     molIds  = {0, numMols};  // numMols is out of bounds
  std::vector<int16_t> results = {1, 1};

  EXPECT_THROW(tracker.record(molIds, results), std::out_of_range);

  // Test negative molecule ID
  std::vector<int> negativeMolIds = {0, -1};
  EXPECT_THROW(tracker.record(negativeMolIds, results), std::out_of_range);
}

// Test batch size edge cases
TEST_F(SchedulerTest, BatchSizeEdgeCases) {
  Scheduler tracker(numMols, confsPerMol, maxIterations);

  // Zero batch size
  auto emptyBatch = tracker.dispatch(0);
  EXPECT_TRUE(emptyBatch.empty());

  // Batch size of 1
  auto singleBatch = tracker.dispatch(1);
  EXPECT_EQ(singleBatch.size(), 1);
}

// Test thread safety (basic concurrent access)
TEST_F(SchedulerTest, ThreadSafety) {
  Scheduler                     tracker(10, 2, 3);
  std::vector<std::thread>      threads;
  std::vector<std::vector<int>> allMolIds(4);

  // Launch multiple threads to dispatch simultaneously
  for (int i = 0; i < 4; i++) {
    threads.emplace_back([&tracker, &allMolIds, i]() {
      allMolIds[i] = tracker.dispatch(5);
      tracker.record(allMolIds[i], std::vector<int16_t>(allMolIds[i].size(), 1));
    });
  }

  // Wait for all threads
  for (auto& t : threads) {
    t.join();
  }

  // Verify we got reasonable results (no crashes, valid IDs)
  for (const auto& molIds : allMolIds) {
    EXPECT_GT(molIds.size(), 0);
    for (int id : molIds) {
      EXPECT_GE(id, 0);
      EXPECT_LT(id, 10);
    }
  }
}
