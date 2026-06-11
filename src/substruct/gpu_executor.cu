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

#include "src/substruct/gpu_executor.h"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit {

std::pair<int, int> getStreamPriorityRange() {
  int leastPriority    = 0;
  int greatestPriority = 0;
  cudaCheckError(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  return {greatestPriority, leastPriority};
}

GpuExecutor::GpuExecutor(int executorIdx, int gpuDeviceId)
    : computeStream(("executor" + std::to_string(executorIdx) + "_mainStream").c_str()),
      recursiveStream(getStreamPriorityRange().first,
                      ("executor" + std::to_string(executorIdx) + "_priorityRecursiveStream").c_str()),
      postRecursionStream(getStreamPriorityRange().second,
                          ("executor" + std::to_string(executorIdx) + "_postRecursionStream").c_str()),
      recursiveScratch(nullptr),
      deviceId(gpuDeviceId) {}

void GpuExecutor::initializeForStream() {
  cudaStream_t s         = computeStream.stream();
  cudaStream_t recStream = recursiveStream.stream();
  deviceResults.setStream(s);
  consolidatedBuffer.setStream(s);
  recursiveScratch.setStream(recStream);
}

void GpuExecutor::applyMiniBatchPlan(MiniBatchPlan&& plan) {
  this->plan = std::move(plan);
}

// =============================================================================
// ConsolidatedDeviceBuffer Implementation
// =============================================================================

void ConsolidatedDeviceBuffer::allocate(int maxBatchSize, cudaStream_t stream) {
  maxBatchSize_           = maxBatchSize;
  const size_t totalBytes = computeConsolidatedBufferBytes(maxBatchSize);
  data_.setStream(stream);
  if (data_.size() < totalBytes) {
    data_.resize(totalBytes);
  }
}

void ConsolidatedDeviceBuffer::copyFromHost(const PinnedHostBuffer& host, cudaStream_t stream) {
  cudaCheckError(cudaMemcpyAsync(data_.data(),
                                 host.consolidated.basePtr,
                                 host.consolidated.totalBytes,
                                 cudaMemcpyHostToDevice,
                                 stream));
}

void ConsolidatedDeviceBuffer::setStream(cudaStream_t stream) {
  data_.setStream(stream);
}

size_t ConsolidatedDeviceBuffer::miniBatchPairMatchStartsOffset() const {
  size_t offset = sizeof(int) * static_cast<size_t>(maxBatchSize_);  // after pairIndices
  return alignOffset(offset);
}

size_t ConsolidatedDeviceBuffer::matchCountsOffset() const {
  size_t offset = miniBatchPairMatchStartsOffset();
  offset += sizeof(int) * static_cast<size_t>(maxBatchSize_ + 1);
  return alignOffset(offset);
}

size_t ConsolidatedDeviceBuffer::reportedCountsOffset() const {
  size_t offset = matchCountsOffset();
  offset += sizeof(int) * static_cast<size_t>(maxBatchSize_);
  return alignOffset(offset);
}

size_t ConsolidatedDeviceBuffer::overflowFlagsOffset() const {
  size_t offset = reportedCountsOffset();
  offset += sizeof(int) * static_cast<size_t>(maxBatchSize_);
  return alignOffset(offset);
}

size_t ConsolidatedDeviceBuffer::matchGlobalPairIndicesOffset(int depth) const {
  size_t offset = overflowFlagsOffset();
  offset += sizeof(uint8_t) * static_cast<size_t>(maxBatchSize_);
  offset = alignOffset(offset);

  for (int i = 0; i < depth; ++i) {
    offset += sizeof(int) * static_cast<size_t>(maxBatchSize_);  // matchGlobalPairIndices[i]
    offset = alignOffset(offset);
    offset += sizeof(int) * static_cast<size_t>(maxBatchSize_);  // matchBatchLocalIndices[i]
    offset = alignOffset(offset);
  }
  return offset;
}

size_t ConsolidatedDeviceBuffer::matchBatchLocalIndicesOffset(int depth) const {
  size_t offset = matchGlobalPairIndicesOffset(depth);
  offset += sizeof(int) * static_cast<size_t>(maxBatchSize_);
  return alignOffset(offset);
}

int* ConsolidatedDeviceBuffer::pairIndices() const {
  return reinterpret_cast<int*>(data_.data() + pairIndicesOffset());
}

int* ConsolidatedDeviceBuffer::miniBatchPairMatchStarts() const {
  return reinterpret_cast<int*>(data_.data() + miniBatchPairMatchStartsOffset());
}

int* ConsolidatedDeviceBuffer::matchCounts() const {
  return reinterpret_cast<int*>(data_.data() + matchCountsOffset());
}

int* ConsolidatedDeviceBuffer::reportedCounts() const {
  return reinterpret_cast<int*>(data_.data() + reportedCountsOffset());
}

uint8_t* ConsolidatedDeviceBuffer::overflowFlags() const {
  return reinterpret_cast<uint8_t*>(data_.data() + overflowFlagsOffset());
}

const int* ConsolidatedDeviceBuffer::matchGlobalPairIndices(int depth) const {
  return reinterpret_cast<const int*>(data_.data() + matchGlobalPairIndicesOffset(depth));
}

const int* ConsolidatedDeviceBuffer::matchBatchLocalIndices(int depth) const {
  return reinterpret_cast<const int*>(data_.data() + matchBatchLocalIndicesOffset(depth));
}

}  // namespace nvMolKit
