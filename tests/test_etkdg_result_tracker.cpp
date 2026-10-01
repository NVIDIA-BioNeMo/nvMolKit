// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <chrono>
#include <future>
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
  EXPECT_THROW(Scheduler(confsPerMol, std::vector<int>{}), std::invalid_argument);
  EXPECT_THROW(Scheduler(-1, std::vector<int>(numMols, maxIterations)), std::invalid_argument);
  EXPECT_THROW(Scheduler(0, std::vector<int>(numMols, maxIterations)), std::invalid_argument);
  EXPECT_THROW(Scheduler(confsPerMol, std::vector<int>(numMols, -1)), std::invalid_argument);
  EXPECT_THROW(Scheduler(confsPerMol, std::vector<int>(numMols, 0)), std::invalid_argument);
  EXPECT_THROW(Scheduler(confsPerMol, std::vector<int>{1, 0}), std::invalid_argument);
}

TEST_F(SchedulerTest, BasicDispatchOversubscribe) {
  Scheduler tracker(confsPerMol, std::vector<int>(numMols, maxIterations));
  auto      molIds = tracker.dispatch(numMols);
  ASSERT_THAT(molIds, ::testing::ElementsAreArray({0, 0, 0, 1, 1}));
  auto molIds2 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds2, ::testing::ElementsAreArray({1, 2, 2, 2, 3}));
  auto molIds3 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds3, ::testing::ElementsAreArray({3, 3, 4, 4, 4}));
  const std::vector<int16_t> failedResults(numMols, -1);
  tracker.record(molIds, failedResults);
  tracker.record(molIds2, failedResults);
  tracker.record(molIds3, failedResults);

  // Failed attempts open replacement slots. Each batch is filled by the normal pass, so no top-up attempts are added.
  molIds = tracker.dispatch(numMols);
  ASSERT_THAT(molIds, ::testing::ElementsAreArray({0, 0, 0, 1, 1}));
  molIds2 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds2, ::testing::ElementsAreArray({1, 2, 2, 2, 3}));
  molIds3 = tracker.dispatch(numMols);
  ASSERT_THAT(molIds3, ::testing::ElementsAreArray({3, 3, 4, 4, 4}));
  tracker.record(molIds, failedResults);
  tracker.record(molIds2, failedResults);
  tracker.record(molIds3, failedResults);

  // We've dispatched max attempts.
  auto molIds4 = tracker.dispatch(numMols);
  EXPECT_THAT(molIds4, testing::IsEmpty());
}

TEST_F(SchedulerTest, BasicDispatchFullComplete) {
  Scheduler tracker(confsPerMol, std::vector<int>(numMols, maxIterations));
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
  Scheduler tracker(confsPerMol, std::vector<int>(numMols, maxIterations));

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

  tracker.record(molIds2, goodResults);
  tracker.record(molIds3, goodResults);

  // We've recorded passes on everything
  auto molIds4 = tracker.dispatch(numMols);
  EXPECT_THAT(molIds4, testing::IsEmpty());
}

// Test basic dispatch functionality
TEST_F(SchedulerTest, BasicDispatchFullWithSomeFails) {
  Scheduler tracker(confsPerMol, std::vector<int>(numMols, maxIterations));

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

  // Systems 0, 3 and 4 are done. Molecules 1 and 2 each need one replacement, and the spare slots are
  // topped up round-robin with extra attempts for those two molecules.
  auto molIds4 = tracker.dispatch(numMols);
  EXPECT_THAT(molIds4, ::testing::ElementsAreArray({1, 2, 1, 2, 1}));
  tracker.record(molIds4, {0, -1, -1, -1, -1});

  // Molecule 1 is done and molecule 2 has one attempt left in its budget.
  auto molIds5 = tracker.dispatch(numMols);
  EXPECT_THAT(molIds5, ::testing::ElementsAreArray({2}));
  tracker.record(molIds5, {-1});

  // Molecule 2 exhausted its budget without finishing.
  EXPECT_THAT(tracker.dispatch(numMols), ::testing::IsEmpty());
  EXPECT_FALSE(tracker.allFinished());
}

// Test dispatch with batch size larger than number of molecules
TEST_F(SchedulerTest, DispatchLargeBatchSize) {
  Scheduler tracker(2, std::vector<int>(2, 4));

  // The normal pass covers the needed conformers, then top-up alternates between molecules until each
  // reaches its budget of 8 attempts.
  constexpr int largeBatchSize = 100;
  auto          molIds         = tracker.dispatch(largeBatchSize);
  EXPECT_THAT(molIds, ::testing::ElementsAreArray({0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}));
}

// Test record validation - mismatched vector sizes
TEST_F(SchedulerTest, RecordMismatchedSizes) {
  Scheduler tracker(confsPerMol, std::vector<int>(numMols, maxIterations));

  std::vector<int>     molIds  = {0, 1, 2};
  std::vector<int16_t> results = {1, 1};  // Wrong size

  EXPECT_THROW(tracker.record(molIds, results), std::invalid_argument);
}

// Test record validation - invalid molecule ID
TEST_F(SchedulerTest, RecordInvalidMoleculeId) {
  Scheduler tracker(confsPerMol, std::vector<int>(numMols, maxIterations));

  std::vector<int>     molIds  = {0, numMols};  // numMols is out of bounds
  std::vector<int16_t> results = {1, 1};

  EXPECT_THROW(tracker.record(molIds, results), std::out_of_range);

  // Test negative molecule ID
  std::vector<int> negativeMolIds = {0, -1};
  EXPECT_THROW(tracker.record(negativeMolIds, results), std::out_of_range);
}

// Test batch size edge cases
TEST_F(SchedulerTest, BatchSizeEdgeCases) {
  Scheduler tracker(confsPerMol, std::vector<int>(numMols, maxIterations));

  // Zero batch size
  auto emptyBatch = tracker.dispatch(0);
  EXPECT_TRUE(emptyBatch.empty());

  // Batch size of 1
  auto singleBatch = tracker.dispatch(1);
  EXPECT_EQ(singleBatch.size(), 1);
}

TEST_F(SchedulerTest, ReportsStablePerMoleculeAttemptIds) {
  Scheduler        tracker(3, std::vector<int>(2, 2));
  std::vector<int> attemptIds;

  auto firstBatch = tracker.dispatch(4, &attemptIds);
  EXPECT_THAT(firstBatch, testing::ElementsAre(0, 0, 0, 1));
  EXPECT_THAT(attemptIds, testing::ElementsAre(0, 1, 2, 0));

  // Remaining initial attempts for molecule 1, then top-up attempts for both molecules.
  auto secondBatch = tracker.dispatch(4, &attemptIds);
  EXPECT_THAT(secondBatch, testing::ElementsAre(1, 1, 0, 1));
  EXPECT_THAT(attemptIds, testing::ElementsAre(1, 2, 3, 3));

  // Molecule 0's failures reopen two normal slots, which use the rest of its budget. Molecule 1 then
  // consumes the rest of its budget through top-up.
  tracker.record(firstBatch, {-1, -1, -1, -1});
  auto retries = tracker.dispatch(4, &attemptIds);
  EXPECT_THAT(retries, testing::ElementsAre(0, 0, 1, 1));
  EXPECT_THAT(attemptIds, testing::ElementsAre(4, 5, 4, 5));
  tracker.record(secondBatch, {-1, -1, -1, -1});
  tracker.record(retries, {-1, -1, -1, -1});
  EXPECT_TRUE(tracker.dispatch(1).empty());
}

TEST_F(SchedulerTest, DispatchWaitsForInFlightRetryResults) {
  Scheduler scheduler(1, std::vector<int>(1, 2));

  auto firstAttempt = scheduler.dispatch(1);
  ASSERT_THAT(firstAttempt, testing::ElementsAre(0));
  // The retry is dispatched by top-up while the first attempt is still in flight.
  auto secondAttempt = scheduler.dispatch(1);
  ASSERT_THAT(secondAttempt, testing::ElementsAre(0));

  // With the budget fully dispatched, an empty result must not be reported until every in-flight
  // attempt has been recorded.
  auto nextDispatch = std::async(std::launch::async, [&scheduler]() { return scheduler.dispatch(1); });
  EXPECT_EQ(nextDispatch.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);

  scheduler.record(firstAttempt, {-1});
  EXPECT_EQ(nextDispatch.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);

  scheduler.record(secondAttempt, {0});
  ASSERT_EQ(nextDispatch.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_TRUE(nextDispatch.get().empty());
  EXPECT_TRUE(scheduler.allFinished());
}

TEST_F(SchedulerTest, ResolvedFailureRetriesWhileUnrelatedAttemptIsInFlight) {
  Scheduler scheduler(1, std::vector<int>(2, 2));

  auto initialAttempts = scheduler.dispatch(2);
  ASSERT_THAT(initialAttempts, testing::ElementsAre(0, 1));
  scheduler.record({0}, {-1});

  auto retry = std::async(std::launch::async, [&scheduler]() { return scheduler.dispatch(1); });
  if (retry.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
    scheduler.cancel();
  }
  ASSERT_EQ(retry.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_THAT(retry.get(), testing::ElementsAre(0));

  scheduler.record({0, 1}, {0, 0});
  EXPECT_TRUE(scheduler.dispatch(1).empty());
}

TEST_F(SchedulerTest, CancelWakesBlockingDispatch) {
  // A budget of one attempt so that the second dispatch has nothing to top up and blocks.
  Scheduler scheduler(1, std::vector<int>(1, 1));
  ASSERT_THAT(scheduler.dispatch(1), testing::ElementsAre(0));

  auto blockedDispatch = std::async(std::launch::async, [&scheduler]() { return scheduler.dispatch(1); });
  EXPECT_EQ(blockedDispatch.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);

  scheduler.cancel();
  ASSERT_EQ(blockedDispatch.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_TRUE(blockedDispatch.get().empty());
}

TEST_F(SchedulerTest, AllFailingMoleculesKeepBatchesFullUntilBudgetExhausted) {
  // Two molecules that never embed, mimicking the tail of a large run with several workers in flight.
  constexpr int    kNumMols        = 2;
  constexpr int    kConfsPerMol    = 50;
  constexpr int    kMaxIterations  = 27;
  constexpr int    kBatchSize      = 500;
  constexpr int    kMaxInFlight    = 4;
  constexpr int    kMaxTriesPerMol = kConfsPerMol * kMaxIterations;
  constexpr int    kTotalBudget    = kNumMols * kMaxTriesPerMol;
  Scheduler        scheduler(kConfsPerMol, std::vector<int>(kNumMols, kMaxIterations));
  std::vector<int> attemptsPerMol(kNumMols, 0);

  std::vector<std::vector<int>> inFlight;
  int                           totalDispatched = 0;
  while (totalDispatched < kTotalBudget) {
    if (static_cast<int>(inFlight.size()) == kMaxInFlight) {
      scheduler.record(inFlight.front(), std::vector<int16_t>(inFlight.front().size(), -1));
      inFlight.erase(inFlight.begin());
    }
    auto molIds = scheduler.dispatch(kBatchSize);
    ASSERT_EQ(static_cast<int>(molIds.size()), std::min(kBatchSize, kTotalBudget - totalDispatched));
    for (const int molId : molIds) {
      attemptsPerMol[molId]++;
    }
    totalDispatched += static_cast<int>(molIds.size());
    inFlight.push_back(std::move(molIds));
  }
  for (const auto& molIds : inFlight) {
    scheduler.record(molIds, std::vector<int16_t>(molIds.size(), -1));
  }

  EXPECT_TRUE(scheduler.dispatch(kBatchSize).empty());
  EXPECT_FALSE(scheduler.allFinished());
  EXPECT_EQ(totalDispatched, kTotalBudget);
  EXPECT_THAT(attemptsPerMol, testing::Each(kMaxTriesPerMol));
}

TEST_F(SchedulerTest, PerMoleculeIterationLimitsBoundEachMoleculeIndependently) {
  constexpr int    kConfsPerMol = 2;
  Scheduler        scheduler(kConfsPerMol, std::vector<int>{1, 3});
  std::vector<int> attemptsPerMol(2, 0);
  for (auto molIds = scheduler.dispatch(100); !molIds.empty(); molIds = scheduler.dispatch(100)) {
    for (const int molId : molIds) {
      attemptsPerMol[molId]++;
    }
    scheduler.record(molIds, std::vector<int16_t>(molIds.size(), -1));
  }
  EXPECT_THAT(attemptsPerMol, testing::ElementsAre(1 * kConfsPerMol, 3 * kConfsPerMol));
  EXPECT_FALSE(scheduler.allFinished());
}

TEST_F(SchedulerTest, TopUpSweepsRoundRobin) {
  Scheduler        scheduler(1, std::vector<int>(3, 10));
  std::vector<int> attemptIds;

  auto initial = scheduler.dispatch(3, &attemptIds);
  ASSERT_THAT(initial, testing::ElementsAre(0, 1, 2));
  scheduler.record({1}, {0});

  // Molecules 0 and 2 still have an attempt in flight, so only top-up can fill the batch, alternating
  // between the unfinished molecules.
  EXPECT_THAT(scheduler.dispatch(5, &attemptIds), testing::ElementsAre(0, 2, 0, 2, 0));
  EXPECT_THAT(attemptIds, testing::ElementsAre(1, 1, 2, 2, 3));
}

// Test thread safety (basic concurrent access)
TEST_F(SchedulerTest, ThreadSafety) {
  Scheduler                     tracker(2, std::vector<int>(10, 3));
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
