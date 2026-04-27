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

#include "src/substruct/pinned_buffer_pool.h"

#include <optional>

namespace nvMolKit {

namespace {

constexpr size_t kPinnedHostAlignment = 256;

size_t alignPinnedOffset(const size_t offset) {
  return (offset + kPinnedHostAlignment - 1) & ~(kPinnedHostAlignment - 1);
}

}  // namespace

size_t computeConsolidatedBufferBytes(int maxBatchSize) {
  size_t offset = 0;

  auto addBlock = [&](const size_t bytes) {
    offset = alignPinnedOffset(offset);
    offset += bytes;
  };

  // Fixed-width buffers (based on maxBatchSize) - consolidated for single memcpy
  addBlock(sizeof(int) * static_cast<size_t>(maxBatchSize));      // pairIndices
  addBlock(sizeof(int) * static_cast<size_t>(maxBatchSize + 1));  // miniBatchPairMatchStarts
  addBlock(sizeof(int) * static_cast<size_t>(maxBatchSize));      // matchCounts
  addBlock(sizeof(int) * static_cast<size_t>(maxBatchSize));      // reportedCounts
  addBlock(sizeof(uint8_t) * static_cast<size_t>(maxBatchSize));  // overflowFlags

  for (int i = 0; i <= kMaxSmartsNestingDepth; ++i) {
    addBlock(sizeof(int) * static_cast<size_t>(maxBatchSize));  // matchGlobalPairIndicesHost[i]
    addBlock(sizeof(int) * static_cast<size_t>(maxBatchSize));  // matchBatchLocalIndicesHost[i]
  }

  return offset;
}

size_t computePinnedHostBufferBytes(int maxBatchSize, int maxMatchIndicesEstimate, int maxPatternsPerDepth) {
  size_t offset = computeConsolidatedBufferBytes(maxBatchSize);

  auto addBlock = [&](const size_t bytes) {
    offset = alignPinnedOffset(offset);
    offset += bytes;
  };

  // Variable-width buffers - copied separately
  addBlock(sizeof(int16_t) * static_cast<size_t>(maxMatchIndicesEstimate));  // matchIndices

  for (int i = 0; i < 2; ++i) {
    addBlock(sizeof(BatchedPatternEntry) * static_cast<size_t>(maxPatternsPerDepth));  // patternsAtDepthHost
  }

  return offset;
}

void PinnedHostBufferPool::initialize(int poolSize,
                                      int maxBatchSize,
                                      int maxMatchIndicesEstimate,
                                      int maxPatternsPerDepth) {
  buffers_.clear();
  available_ = std::make_unique<ThreadSafeQueue<PinnedHostBuffer*>>();

  buffers_.reserve(static_cast<size_t>(poolSize));
  for (int i = 0; i < poolSize; ++i) {
    auto buffer = createBuffer(maxBatchSize, maxMatchIndicesEstimate, maxPatternsPerDepth);
    available_->push(buffer.get());
    buffers_.push_back(std::move(buffer));
  }
}

PinnedHostBuffer* PinnedHostBufferPool::acquire() {
  auto opt = available_->pop();
  return opt.value_or(nullptr);
}

void PinnedHostBufferPool::release(PinnedHostBuffer* buffer) {
  if (buffer != nullptr) {
    available_->push(buffer);
  }
}

void PinnedHostBufferPool::shutdown() {
  available_->close();
}

std::unique_ptr<PinnedHostBuffer> PinnedHostBufferPool::createBuffer(int maxBatchSize,
                                                                     int maxMatchIndicesEstimate,
                                                                     int maxPatternsPerDepth) {
  const size_t bufferBytes = computePinnedHostBufferBytes(maxBatchSize, maxMatchIndicesEstimate, maxPatternsPerDepth);
  PinnedHostAllocator allocator(bufferBytes);
  auto                buffer = std::make_unique<PinnedHostBuffer>();

  // Allocate fixed-width buffers first (consolidated region for single H2D copy)
  buffer->pairIndices               = allocator.allocate<int>(static_cast<size_t>(maxBatchSize));
  buffer->consolidated.basePtr      = reinterpret_cast<std::byte*>(buffer->pairIndices.data());
  buffer->consolidated.maxBatchSize = maxBatchSize;

  buffer->miniBatchPairMatchStarts = allocator.allocate<int>(static_cast<size_t>(maxBatchSize + 1));
  buffer->matchCounts              = allocator.allocate<int>(static_cast<size_t>(maxBatchSize));
  buffer->reportedCounts           = allocator.allocate<int>(static_cast<size_t>(maxBatchSize));
  buffer->overflowFlags            = allocator.allocate<uint8_t>(static_cast<size_t>(maxBatchSize));

  for (int i = 0; i <= kMaxSmartsNestingDepth; ++i) {
    buffer->matchGlobalPairIndicesHost[i] = allocator.allocate<int>(static_cast<size_t>(maxBatchSize));
    buffer->matchBatchLocalIndicesHost[i] = allocator.allocate<int>(static_cast<size_t>(maxBatchSize));
  }

  buffer->consolidated.totalBytes = computeConsolidatedBufferBytes(maxBatchSize);

  // Allocate variable-width buffers (copied separately)
  if (maxMatchIndicesEstimate > 0) {
    buffer->matchIndices = allocator.allocate<int16_t>(static_cast<size_t>(maxMatchIndicesEstimate));
  }

  for (int i = 0; i < 2; ++i) {
    buffer->patternsAtDepthHost[i] = allocator.allocate<BatchedPatternEntry>(static_cast<size_t>(maxPatternsPerDepth));
  }

  return buffer;
}

}  // namespace nvMolKit
