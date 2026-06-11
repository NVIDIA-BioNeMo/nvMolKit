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

#ifndef NVMOLKIT_PINNED_HOST_ALLOCATOR_H
#define NVMOLKIT_PINNED_HOST_ALLOCATOR_H

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "src/utils/host_vector.h"

namespace nvMolKit {

/**
 * @brief Arena allocator for CUDA pinned host memory.
 *
 * Provides fast bump-pointer allocation from a preallocated pinned memory pool.
 * The total expected size should be provided exactly once, either via the constructor
 * or by calling preallocate(). Calling both or calling preallocate() multiple times
 * is an error.
 *
 * @note Does not support deallocation or defragmentation. Memory is freed when the
 *       allocator is destroyed and all views are destroyed.
 * @note All allocations are aligned to 256 bytes.
 * @note If allocations exceed the preallocated size, additional buffers are allocated
 *       in chunks equal to the original preallocation size. This incurs synchronous
 *       cudaMallocHost calls and should be avoided for performance.
 */
class PinnedHostAllocator {
 public:
  PinnedHostAllocator() = default;

  /// @brief Construct with preallocated pinned memory pool.
  /// @param estimatedBytes Total bytes to preallocate. Cannot call preallocate() after this.
  explicit PinnedHostAllocator(size_t estimatedBytes);

  PinnedHostAllocator(const PinnedHostAllocator&)                = delete;
  PinnedHostAllocator& operator=(const PinnedHostAllocator&)     = delete;
  PinnedHostAllocator(PinnedHostAllocator&&) noexcept            = default;
  PinnedHostAllocator& operator=(PinnedHostAllocator&&) noexcept = default;

  /// @brief Preallocate the pinned memory pool.
  /// @param estimatedBytes Total bytes to preallocate.
  /// @throws std::logic_error if already preallocated (via constructor or prior call).
  void preallocate(size_t estimatedBytes);

  /// @brief Allocate a view of count elements from the pool.
  template <typename T> PinnedHostView<T> allocate(size_t count) {
    if (count == 0) {
      throw std::invalid_argument("PinnedHostAllocator allocate requires non-zero size.");
    }
    const size_t         bytes = count * sizeof(T);
    PinnedHostAllocation alloc = allocateBytes(bytes);
    return PinnedHostView<T>(std::span<T>(reinterpret_cast<T*>(alloc.data), count), std::move(alloc.owner));
  }

 private:
  struct PinnedHostAllocation {
    std::byte*                 data  = nullptr;
    size_t                     bytes = 0;
    std::shared_ptr<std::byte> owner;
  };

  struct BufferEntry {
    std::shared_ptr<PinnedHostVector<std::byte>> buffer;
    size_t                                       offset = 0;
  };

  PinnedHostAllocation allocateBytes(size_t bytes);

  std::vector<BufferEntry> buffers_;
  size_t                   bufferBytes_  = 0;
  bool                     preallocated_ = false;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_PINNED_HOST_ALLOCATOR_H
