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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "src/utils/pinned_host_allocator.h"

using namespace nvMolKit;

namespace {
constexpr size_t KB = 1024;
constexpr size_t MB = 1024 * KB;
}  // namespace

TEST(PinnedHostAllocator, PreallocateZeroThrows) {
  PinnedHostAllocator allocator;
  EXPECT_THROW(allocator.preallocate(0), std::invalid_argument);
}

TEST(PinnedHostAllocator, PreallocateTwiceThrows) {
  PinnedHostAllocator allocator;
  allocator.preallocate(64 * KB);
  EXPECT_THROW(allocator.preallocate(64 * KB), std::runtime_error);
}

TEST(PinnedHostAllocator, AllocateZeroThrows) {
  PinnedHostAllocator allocator(64 * KB);
  EXPECT_THROW(allocator.allocate<int>(0), std::invalid_argument);
}

TEST(PinnedHostAllocator, AllocateBeforePreallocateThrows) {
  PinnedHostAllocator allocator;
  EXPECT_THROW(allocator.allocate<int>(1), std::runtime_error);
}

TEST(PinnedHostAllocator, AllocateTooLargeThrows) {
  PinnedHostAllocator allocator(64 * KB);
  EXPECT_THROW(allocator.allocate<std::byte>(128 * KB), std::runtime_error);
}

TEST(PinnedHostAllocator, AllocationAlignment) {
  constexpr size_t kAlignment = 256;

  PinnedHostAllocator allocator(1 * MB);
  auto                viewA = allocator.allocate<int>(1);
  auto                viewB = allocator.allocate<int>(7);

  EXPECT_NE(viewA.data(), nullptr);
  EXPECT_NE(viewB.data(), nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(viewA.data()) % kAlignment, 0u);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(viewB.data()) % kAlignment, 0u);
}

TEST(PinnedHostAllocator, AllocationNoOverlap) {
  PinnedHostAllocator allocator(1 * MB);
  auto                viewA = allocator.allocate<int>(16 * KB);
  auto                viewB = allocator.allocate<int>(16 * KB);

  ASSERT_NE(viewA.data(), nullptr);
  ASSERT_NE(viewB.data(), nullptr);
  EXPECT_NE(viewA.data(), viewB.data());

  for (size_t i = 0; i < viewA.size(); ++i) {
    viewA[i] = 11;
  }
  for (size_t i = 0; i < viewB.size(); ++i) {
    viewB[i] = 22;
  }
  for (size_t i = 0; i < viewA.size(); ++i) {
    EXPECT_EQ(viewA[i], 11);
  }
  for (size_t i = 0; i < viewB.size(); ++i) {
    EXPECT_EQ(viewB[i], 22);
  }
}

TEST(PinnedHostAllocator, AllocationsExceedingFirstChunkTriggerNewBlock) {
  constexpr size_t kPreallocBytes = 10 * MB;
  constexpr size_t kAlignment     = 256;
  constexpr size_t kAllocBytes    = 6 * MB;

  PinnedHostAllocator allocator(kPreallocBytes);

  // First allocation: 6 MB
  auto viewA = allocator.allocate<std::byte>(kAllocBytes);
  EXPECT_NE(viewA.data(), nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(viewA.data()) % kAlignment, 0u);

  // Second allocation: 6 MB, would need offset ~6MB + 6MB = 12MB > 10MB
  // This must trigger a new block allocation
  auto viewB = allocator.allocate<std::byte>(kAllocBytes);
  EXPECT_NE(viewB.data(), nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(viewB.data()) % kAlignment, 0u);

  // Views should not overlap (different blocks in this case)
  const auto ptrA = reinterpret_cast<std::uintptr_t>(viewA.data());
  const auto ptrB = reinterpret_cast<std::uintptr_t>(viewB.data());
  EXPECT_TRUE(ptrA + viewA.size() <= ptrB || ptrB + viewB.size() <= ptrA);

  // Both views should be writable
  std::memset(viewA.data(), 0xAA, viewA.size());
  std::memset(viewB.data(), 0xBB, viewB.size());
  EXPECT_EQ(static_cast<unsigned char>(viewA[0]), 0xAAu);
  EXPECT_EQ(static_cast<unsigned char>(viewB[0]), 0xBBu);
}

TEST(PinnedHostView, RetainsOwnership) {
  PinnedHostView<int> view;
  {
    PinnedHostAllocator allocator(1 * MB);
    view = allocator.allocate<int>(8 * KB);
    for (size_t i = 0; i < view.size(); ++i) {
      view[i] = static_cast<int>(i + 5);
    }
  }

  EXPECT_EQ(view.size(), 8u * KB);
  EXPECT_NE(view.data(), nullptr);
  for (size_t i = 0; i < view.size(); ++i) {
    EXPECT_EQ(view[i], static_cast<int>(i + 5));
  }
}
