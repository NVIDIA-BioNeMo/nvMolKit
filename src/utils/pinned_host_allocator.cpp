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

#include "src/utils/pinned_host_allocator.h"

#include <stdexcept>

namespace nvMolKit {

namespace {
constexpr size_t kAlignment = 256;

size_t alignUp(const size_t offset) {
  return (offset + kAlignment - 1) & ~(kAlignment - 1);
}
}  // namespace

PinnedHostAllocator::PinnedHostAllocator(const size_t estimatedBytes) {
  if (estimatedBytes > 0) {
    preallocate(estimatedBytes);
  }
}

void PinnedHostAllocator::preallocate(const size_t estimatedBytes) {
  if (preallocated_) {
    throw std::runtime_error("PinnedHostAllocator preallocate called more than once.");
  }
  if (estimatedBytes == 0) {
    throw std::invalid_argument("PinnedHostAllocator preallocate requires non-zero size.");
  }
  bufferBytes_ = estimatedBytes;
  buffers_.push_back(BufferEntry{std::make_shared<PinnedHostVector<std::byte>>(estimatedBytes), 0});
  preallocated_ = true;
}

PinnedHostAllocator::PinnedHostAllocation PinnedHostAllocator::allocateBytes(const size_t bytes) {
  if (bytes == 0) {
    throw std::invalid_argument("PinnedHostAllocator allocate requires non-zero size.");
  }
  if (!preallocated_) {
    throw std::runtime_error("PinnedHostAllocator allocate called before preallocate.");
  }
  if (bytes > bufferBytes_) {
    throw std::runtime_error("PinnedHostAllocator allocation exceeds preallocated buffer size.");
  }

  for (auto& entry : buffers_) {
    const size_t alignedOffset = alignUp(entry.offset);
    if (alignedOffset + bytes <= entry.buffer->size()) {
      entry.offset = alignedOffset + bytes;
      std::shared_ptr<std::byte> owner(entry.buffer, entry.buffer->data());
      return {entry.buffer->data() + alignedOffset, bytes, std::move(owner)};
    }
  }

  auto buffer = std::make_shared<PinnedHostVector<std::byte>>(bufferBytes_);
  buffers_.push_back(BufferEntry{buffer, bytes});
  std::shared_ptr<std::byte> owner(buffer, buffer->data());
  return {buffer->data(), bytes, std::move(owner)};
}

}  // namespace nvMolKit
