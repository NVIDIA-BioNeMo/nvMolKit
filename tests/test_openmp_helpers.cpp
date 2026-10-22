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

#include <gtest/gtest.h>
#include <omp.h>

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "src/utils/openmp_helpers.h"

namespace {

using ::nvMolKit::detail::OpenMPExceptionRegistry;

constexpr int kNumThreads = 4;

class ScopedThreadCount {
 public:
  explicit ScopedThreadCount(const int desiredThreads)
      : previousThreads_(omp_get_max_threads()),
        appliedThreads_(desiredThreads <= previousThreads_ ? desiredThreads : previousThreads_) {
    if (appliedThreads_ > 0) {
      omp_set_num_threads(appliedThreads_);
    }
  }
  ScopedThreadCount(const ScopedThreadCount&)            = delete;
  ScopedThreadCount& operator=(const ScopedThreadCount&) = delete;
  ScopedThreadCount(ScopedThreadCount&&)                 = delete;
  ScopedThreadCount& operator=(ScopedThreadCount&&)      = delete;

  ~ScopedThreadCount() { omp_set_num_threads(previousThreads_); }

 private:
  int previousThreads_;
  int appliedThreads_;
};

}  // namespace

TEST(OpenMPExceptionRegistryTest, NoExceptionNoThrow) {
  ScopedThreadCount const setThreads(kNumThreads);
  OpenMPExceptionRegistry exceptionHandler;
  EXPECT_NO_THROW(exceptionHandler.rethrow());
}

TEST(OpenMPExceptionRegistryTest, SingleExceptionCaptured) {
  ScopedThreadCount const setThreads(kNumThreads);
  OpenMPExceptionRegistry exceptionHandler;

#pragma omp parallel for shared(exceptionHandler) default(none)
  for (int i = 0; i < kNumThreads; ++i) {
    try {
      if (omp_get_thread_num() == 2) {
        throw std::runtime_error("thread 2 exception");
      }
    } catch (...) {
      exceptionHandler.store(std::current_exception());
    }
  }

  try {
    exceptionHandler.rethrow();
    FAIL() << "Expected exception not thrown";
  } catch (const std::runtime_error& err) {
    EXPECT_STREQ("thread 2 exception", err.what());
  }
}

TEST(OpenMPExceptionRegistryTest, OnlyFirstExceptionCapturedUnderRace) {
  ScopedThreadCount const setThreads(kNumThreads);

  // Only run if OMP parallel sections execute simultaneously for all threads
  if (omp_get_max_threads() < kNumThreads) {
    GTEST_SKIP() << "Insufficient OpenMP threads to run concurrency test";
  }
  OpenMPExceptionRegistry exceptionHandler;

  // The other threads wait for thread 3 to store first, then race each other to store. A sleep-based delay is not
  // enough here: OpenMP thread startup can exceed it on many-core hosts. The wait requires a full team, which
  // omp_get_max_threads() does not guarantee (e.g. OMP_THREAD_LIMIT, dynamic adjustment), so check the real team size.
  std::atomic<bool> firstStored{false};
  std::atomic<bool> teamTooSmall{false};
#pragma omp parallel default(none) shared(exceptionHandler, firstStored, teamTooSmall) num_threads(kNumThreads)
  {
    if (omp_get_num_threads() < kNumThreads) {
      teamTooSmall.store(true);
    } else {
      const int threadIdx = omp_get_thread_num();
      try {
        if (threadIdx == 3) {
          throw std::runtime_error("thread 3 immediate exception");
        }
        while (!firstStored.load()) {
          std::this_thread::yield();
        }
        throw std::runtime_error("delayed exception");
      } catch (...) {
        exceptionHandler.store(std::current_exception());
        if (threadIdx == 3) {
          firstStored.store(true);
        }
      }
    }
  }
  if (teamTooSmall.load()) {
    GTEST_SKIP() << "OpenMP team smaller than requested; cannot run concurrency test";
  }

  try {
    exceptionHandler.rethrow();
    FAIL() << "Expected exception not thrown";
  } catch (const std::runtime_error& err) {
    EXPECT_STREQ("thread 3 immediate exception", err.what());
  }
}

TEST(OpenMPExceptionRegistryTest, NoDeadlockWithOtherSynchronization) {
  ScopedThreadCount const setThreads(kNumThreads);
  OpenMPExceptionRegistry exceptionHandler;

  std::mutex mutex;

#pragma omp parallel for default(none) shared(mutex, exceptionHandler)
  for (int i = 0; i < kNumThreads; ++i) {
    try {
      {
        const std::lock_guard<std::mutex> lock(mutex);
        if (i == 3) {
          throw std::runtime_error("Critical section throws");
        }
      }

      if (i == 1) {
        throw std::runtime_error("Noncritical section throws");
      }
    } catch (...) {
      exceptionHandler.store(std::current_exception());
    }
  }

  EXPECT_THROW(exceptionHandler.rethrow(), std::runtime_error);
}
