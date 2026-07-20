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

#ifndef NVMOLKIT_GPU_EXECUTOR_RING_H
#define NVMOLKIT_GPU_EXECUTOR_RING_H

#include <stdexcept>
#include <utility>
#include <vector>

#include "src/utils/thread_safe_queue.h"

namespace nvMolKit {

template <typename ExecutorT, typename BatchT> struct InFlightExecutorBatch {
  ExecutorT* executor = nullptr;
  BatchT     batch{};
};

/**
 * @brief Consume prepared batches with a fixed ring of asynchronous executors.
 *
 * Launch callbacks enqueue work on an executor and return after the completion
 * event/copy has been recorded. Drain callbacks wait for the oldest executor,
 * accumulate results, and release per-batch resources. This keeps up to
 * `executors.size()` batches in flight while preserving oldest-first draining.
 *
 * `BatchT` must be movable and default-constructible.
 */
template <typename ExecutorT, typename BatchT, typename LaunchFunc, typename DrainFunc>
void runQueuedExecutorRing(const std::vector<ExecutorT*>& executors,
                           ThreadSafeQueue<BatchT>&       batchQueue,
                           LaunchFunc&&                   launch,
                           DrainFunc&&                    drain) {
  const int numExecutors = static_cast<int>(executors.size());
  if (numExecutors <= 0) {
    throw std::invalid_argument("runQueuedExecutorRing requires at least one executor");
  }

  std::vector<InFlightExecutorBatch<ExecutorT, BatchT>> pending(static_cast<size_t>(numExecutors));
  int                                                   pendingHead  = 0;
  int                                                   pendingTail  = 0;
  int                                                   pendingCount = 0;

  auto drainOne = [&]() {
    auto& slot = pending[static_cast<size_t>(pendingHead)];
    drain(*slot.executor, slot.batch);
    slot.executor = nullptr;
    slot.batch    = BatchT{};

    pendingHead = (pendingHead + 1) % numExecutors;
    --pendingCount;
  };

  while (true) {
    if (pendingCount == numExecutors) {
      drainOne();
      continue;
    }

    BatchT batch{};
    if (pendingCount > 0) {
      auto optBatch = batchQueue.tryPop();
      if (!optBatch) {
        drainOne();
        continue;
      }
      batch = std::move(*optBatch);
    } else {
      auto optBatch = batchQueue.pop();
      if (!optBatch) {
        break;
      }
      batch = std::move(*optBatch);
    }

    ExecutorT* executor = executors[static_cast<size_t>(pendingTail)];
    launch(*executor, batch);

    auto& slot    = pending[static_cast<size_t>(pendingTail)];
    slot.executor = executor;
    slot.batch    = std::move(batch);

    pendingTail = (pendingTail + 1) % numExecutors;
    ++pendingCount;
  }

  while (pendingCount > 0) {
    drainOne();
  }
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_GPU_EXECUTOR_RING_H
