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

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "src/utils/gpu_executor_ring.h"
#include "src/utils/thread_safe_queue.h"

using nvMolKit::ThreadSafeQueue;

namespace {

struct FakeExecutor {
  int id;
};

}  // namespace

// =============================================================================
// Basic Operations
// =============================================================================

TEST(ThreadSafeQueueTest, PushPopSingleItem) {
  ThreadSafeQueue<int> queue;
  queue.push(42);
  const auto result = queue.pop();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 42);
}

TEST(ThreadSafeQueueTest, TryPopEmpty) {
  ThreadSafeQueue<int> queue;
  const auto           result = queue.tryPop();
  EXPECT_FALSE(result.has_value());
}

TEST(ThreadSafeQueueTest, TryPopNonEmpty) {
  ThreadSafeQueue<int> queue;
  queue.push(42);
  const auto result = queue.tryPop();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 42);
}

TEST(ThreadSafeQueueTest, FifoOrder) {
  ThreadSafeQueue<int> queue;
  queue.push(1);
  queue.push(2);
  queue.push(3);

  EXPECT_EQ(*queue.pop(), 1);
  EXPECT_EQ(*queue.pop(), 2);
  EXPECT_EQ(*queue.pop(), 3);
}

TEST(ThreadSafeQueueTest, SizeTracking) {
  ThreadSafeQueue<int> queue;
  EXPECT_EQ(queue.size(), 0u);
  EXPECT_TRUE(queue.empty());

  queue.push(1);
  EXPECT_EQ(queue.size(), 1u);
  EXPECT_FALSE(queue.empty());

  queue.push(2);
  EXPECT_EQ(queue.size(), 2u);

  queue.pop();
  EXPECT_EQ(queue.size(), 1u);

  queue.pop();
  EXPECT_EQ(queue.size(), 0u);
  EXPECT_TRUE(queue.empty());
}

TEST(ThreadSafeQueueTest, MoveOnlyType) {
  ThreadSafeQueue<std::unique_ptr<int>> queue;
  queue.push(std::make_unique<int>(42));

  const auto result = queue.pop();
  ASSERT_TRUE(result.has_value());
  ASSERT_NE(*result, nullptr);
  EXPECT_EQ(**result, 42);
}

// =============================================================================
// Queued Executor Ring
// =============================================================================

TEST(GpuExecutorRingTest, RejectsEmptyExecutorList) {
  ThreadSafeQueue<int>       queue;
  std::vector<FakeExecutor*> executors;
  queue.close();

  EXPECT_THROW(nvMolKit::runQueuedExecutorRing(
                 executors,
                 queue,
                 [](FakeExecutor&, int&) {},
                 [](FakeExecutor&, int&) {}),
               std::invalid_argument);
}

TEST(GpuExecutorRingTest, ReusesExecutorsAndDrainsInLaunchOrder) {
  ThreadSafeQueue<int> queue;
  for (int batch = 1; batch <= 5; ++batch) {
    queue.push(batch);
  }
  queue.close();

  FakeExecutor                     executor0{0};
  FakeExecutor                     executor1{1};
  std::vector<FakeExecutor*>       executors{&executor0, &executor1};
  std::vector<std::pair<int, int>> launches;
  std::vector<std::pair<int, int>> drains;

  nvMolKit::runQueuedExecutorRing(
    executors,
    queue,
    [&](FakeExecutor& executor, int& batch) { launches.emplace_back(executor.id, batch); },
    [&](FakeExecutor& executor, int& batch) { drains.emplace_back(executor.id, batch); });

  EXPECT_EQ(launches,
            (std::vector<std::pair<int, int>>{
              {0, 1},
              {1, 2},
              {0, 3},
              {1, 4},
              {0, 5}
  }));
  EXPECT_EQ(drains,
            (std::vector<std::pair<int, int>>{
              {0, 1},
              {1, 2},
              {0, 3},
              {1, 4},
              {0, 5}
  }));
}

TEST(GpuExecutorRingTest, ClosedEmptyQueueDoesNotInvokeCallbacks) {
  ThreadSafeQueue<int> queue;
  queue.close();

  FakeExecutor               executor{0};
  std::vector<FakeExecutor*> executors{&executor};
  int                        launches = 0;
  int                        drains   = 0;

  nvMolKit::runQueuedExecutorRing(
    executors,
    queue,
    [&](FakeExecutor&, int&) { ++launches; },
    [&](FakeExecutor&, int&) { ++drains; });

  EXPECT_EQ(launches, 0);
  EXPECT_EQ(drains, 0);
}

// =============================================================================
// Close Semantics
// =============================================================================

TEST(ThreadSafeQueueTest, CloseWithItems) {
  ThreadSafeQueue<int> queue;
  queue.push(1);
  queue.push(2);
  queue.close();

  EXPECT_EQ(*queue.pop(), 1);
  EXPECT_EQ(*queue.pop(), 2);
  EXPECT_FALSE(queue.pop().has_value());
}

TEST(ThreadSafeQueueTest, PushAfterClose) {
  ThreadSafeQueue<int> queue;
  queue.push(1);
  queue.close();
  queue.push(2);

  EXPECT_EQ(*queue.pop(), 1);
  EXPECT_FALSE(queue.pop().has_value());
}

// =============================================================================
// Batch Operations
// =============================================================================

TEST(ThreadSafeQueueTest, PushBatch) {
  ThreadSafeQueue<int> queue;
  std::vector<int>     items = {1, 2, 3, 4, 5};
  queue.pushBatch(items);

  EXPECT_EQ(queue.size(), 5u);
  for (int i = 1; i <= 5; ++i) {
    EXPECT_EQ(*queue.pop(), i);
  }
}

TEST(ThreadSafeQueueTest, PushBatchMoveOnly) {
  ThreadSafeQueue<std::unique_ptr<int>> queue;
  std::vector<std::unique_ptr<int>>     items;
  items.push_back(std::make_unique<int>(1));
  items.push_back(std::make_unique<int>(2));
  queue.pushBatch(std::move(items));

  EXPECT_EQ(queue.size(), 2u);
  EXPECT_EQ(**queue.pop(), 1);
  EXPECT_EQ(**queue.pop(), 2);
}

TEST(ThreadSafeQueueTest, PushBatchAfterClose) {
  ThreadSafeQueue<int> queue;
  queue.close();
  std::vector<int> items = {1, 2, 3};
  queue.pushBatch(items);

  EXPECT_TRUE(queue.empty());
}

// =============================================================================
// Concurrent Operations
// =============================================================================

TEST(ThreadSafeQueueTest, SingleProducerSingleConsumer) {
  ThreadSafeQueue<int> queue;
  const int            numItems = 1000;
  std::atomic<int>     consumed{0};

  std::thread consumer([&] {
    for (int i = 0; i < numItems; ++i) {
      auto val = queue.pop();
      EXPECT_TRUE(val.has_value());
      ++consumed;
    }
  });

  std::thread producer([&] {
    for (int i = 0; i < numItems; ++i) {
      queue.push(i);
    }
  });

  producer.join();
  consumer.join();

  EXPECT_EQ(consumed.load(), numItems);
  EXPECT_TRUE(queue.empty());
}

TEST(ThreadSafeQueueTest, MultipleProducersSingleConsumer) {
  ThreadSafeQueue<int> queue;
  const int            numProducers     = 4;
  const int            itemsPerProducer = 250;
  const int            totalItems       = numProducers * itemsPerProducer;
  std::atomic<int>     consumed{0};

  std::thread consumer([&] {
    for (int i = 0; i < totalItems; ++i) {
      auto val = queue.pop();
      EXPECT_TRUE(val.has_value());
      ++consumed;
    }
  });

  std::vector<std::thread> producers;
  for (int p = 0; p < numProducers; ++p) {
    producers.emplace_back([&, p] {
      for (int i = 0; i < itemsPerProducer; ++i) {
        queue.push(p * itemsPerProducer + i);
      }
    });
  }

  for (auto& t : producers) {
    t.join();
  }
  consumer.join();

  EXPECT_EQ(consumed.load(), totalItems);
  EXPECT_TRUE(queue.empty());
}

TEST(ThreadSafeQueueTest, SingleProducerMultipleConsumers) {
  ThreadSafeQueue<int> queue;
  const int            numConsumers = 4;
  const int            totalItems   = 1000;
  std::atomic<int>     consumed{0};

  std::vector<std::thread> consumers;
  for (int c = 0; c < numConsumers; ++c) {
    consumers.emplace_back([&] {
      while (true) {
        auto val = queue.pop();
        if (!val.has_value()) {
          break;
        }
        ++consumed;
      }
    });
  }

  std::thread producer([&] {
    for (int i = 0; i < totalItems; ++i) {
      queue.push(i);
    }
    queue.close();
  });

  producer.join();
  for (auto& t : consumers) {
    t.join();
  }

  EXPECT_EQ(consumed.load(), totalItems);
}

TEST(ThreadSafeQueueTest, CloseWhileWaiting) {
  ThreadSafeQueue<int> queue;
  std::atomic<bool>    popReturned{false};
  std::atomic<bool>    gotNullopt{false};

  std::thread consumer([&] {
    auto val = queue.pop();
    popReturned.store(true);
    gotNullopt.store(!val.has_value());
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(popReturned.load());

  queue.close();
  consumer.join();

  EXPECT_TRUE(popReturned.load());
  EXPECT_TRUE(gotNullopt.load());
}

// =============================================================================
// Resource Pool Pattern (items cycle back)
// =============================================================================

TEST(ThreadSafeQueueTest, ResourcePoolPattern) {
  ThreadSafeQueue<int*> pool;
  int                   resources[3] = {1, 2, 3};

  for (int& r : resources) {
    pool.push(&r);
  }
  EXPECT_EQ(pool.size(), 3u);

  auto r1 = pool.pop();
  auto r2 = pool.pop();
  EXPECT_EQ(pool.size(), 1u);

  pool.push(*r1);
  EXPECT_EQ(pool.size(), 2u);

  pool.push(*r2);
  EXPECT_EQ(pool.size(), 3u);

  auto r3 = pool.pop();
  auto r4 = pool.pop();
  auto r5 = pool.pop();
  EXPECT_TRUE(r3.has_value());
  EXPECT_TRUE(r4.has_value());
  EXPECT_TRUE(r5.has_value());
  EXPECT_TRUE(pool.empty());
}

TEST(ThreadSafeQueueTest, ResourcePoolConcurrent) {
  ThreadSafeQueue<int> pool;
  const int            poolSize      = 4;
  const int            numIterations = 100;

  for (int i = 0; i < poolSize; ++i) {
    pool.push(i);
  }

  std::atomic<int>         acquireCount{0};
  std::atomic<int>         releaseCount{0};
  std::vector<std::thread> workers;

  for (int w = 0; w < 8; ++w) {
    workers.emplace_back([&] {
      for (int i = 0; i < numIterations; ++i) {
        auto resource = pool.pop();
        if (!resource.has_value()) {
          break;
        }
        ++acquireCount;
        std::this_thread::yield();
        pool.push(*resource);
        ++releaseCount;
      }
    });
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  pool.close();

  for (auto& t : workers) {
    t.join();
  }

  EXPECT_EQ(acquireCount.load(), releaseCount.load());
}
