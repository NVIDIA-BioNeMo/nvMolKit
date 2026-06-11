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

#ifndef NVMOLKIT_GPU_EXECUTOR_H
#define NVMOLKIT_GPU_EXECUTOR_H

#include <array>
#include <utility>
#include <vector>

#include "src/substruct/minibatch_planner.h"
#include "src/substruct/molecules_device.cuh"
#include "src/substruct/pinned_buffer_pool.h"
#include "src/substruct/recursive_preprocessor.h"
#include "src/substruct/substruct_search_internal.h"
#include "src/substruct/substruct_types.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

/**
 * @brief Device-side consolidated buffer mirroring the fixed-width portion of PinnedHostBuffer.
 *
 * All fixed-width buffers (based on maxBatchSize) are stored contiguously for single-copy H2D transfers.
 * Accessors compute offsets using the same layout as the host allocation.
 */
class ConsolidatedDeviceBuffer {
 public:
  void allocate(int maxBatchSize, cudaStream_t stream);
  void copyFromHost(const PinnedHostBuffer& host, cudaStream_t stream);
  void setStream(cudaStream_t stream);

  [[nodiscard]] int*       pairIndices() const;
  [[nodiscard]] int*       miniBatchPairMatchStarts() const;
  [[nodiscard]] int*       matchCounts() const;
  [[nodiscard]] int*       reportedCounts() const;
  [[nodiscard]] uint8_t*   overflowFlags() const;
  [[nodiscard]] const int* matchGlobalPairIndices(int depth) const;
  [[nodiscard]] const int* matchBatchLocalIndices(int depth) const;

  [[nodiscard]] int maxBatchSize() const { return maxBatchSize_; }

 private:
  static constexpr size_t kAlignment = 256;

  [[nodiscard]] size_t alignOffset(size_t offset) const { return (offset + kAlignment - 1) & ~(kAlignment - 1); }

  [[nodiscard]] size_t pairIndicesOffset() const { return 0; }
  [[nodiscard]] size_t miniBatchPairMatchStartsOffset() const;
  [[nodiscard]] size_t matchCountsOffset() const;
  [[nodiscard]] size_t reportedCountsOffset() const;
  [[nodiscard]] size_t overflowFlagsOffset() const;
  [[nodiscard]] size_t matchGlobalPairIndicesOffset(int depth) const;
  [[nodiscard]] size_t matchBatchLocalIndicesOffset(int depth) const;

  AsyncDeviceVector<std::byte> data_;
  int                          maxBatchSize_ = 0;
};

/**
 * @brief Owns CUDA resources and device buffers for a worker executor.
 */
struct GpuExecutor {
  MiniBatchPlan plan;

  // Streams and events (declared first so they're destroyed last)
  ScopedStream    computeStream;
  ScopedCudaEvent copyDoneEvent;
  ScopedCudaEvent allocDoneEvent;
  ScopedCudaEvent targetsReadyEvent;

  // Recursive pipeline
  ScopedStreamWithPriority                            recursiveStream;
  ScopedStreamWithPriority                            postRecursionStream;
  std::array<ScopedCudaEvent, kMaxSmartsNestingDepth> depthEvents;
  ScopedCudaEvent                                     recursiveDoneEvent;
  ScopedCudaEvent                                     postRecursionDoneEvent;
  RecursiveScratchBuffers                             recursiveScratch;
  MiniBatchResultsDevice                              deviceResults;
  ConsolidatedDeviceBuffer                            consolidatedBuffer;
  MoleculesDevice                                     targetsDevice;

  int deviceId = 0;  ///< GPU device ID this executor is assigned to

  GpuExecutor(int executorIdx, int gpuDeviceId);

  void initializeForStream();

  cudaStream_t stream() const { return computeStream.stream(); }

  void applyMiniBatchPlan(MiniBatchPlan&& plan);
};

std::pair<int, int> getStreamPriorityRange();

}  // namespace nvMolKit

#endif  // NVMOLKIT_GPU_EXECUTOR_H
