// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_HOST_VECTOR_H
#define NVMOLKIT_HOST_VECTOR_H

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/utils/cuda_error_check.h"

namespace nvMolKit {

// Forward declaration for compatibility methods
template <typename T> class AsyncDeviceVector;

//! Vector using CUDA pinned host memory for faster transfers to/from device.
//! Compatible with AsyncDeviceVector for efficient data transfers.
//! Does not provide capacity optimizations like std::vector, so preallocate as much as possible.
template <typename T> class PinnedHostVector {
 public:
  explicit PinnedHostVector() = default;

  explicit PinnedHostVector(const size_t size) : size_(size) {
    if (size_ > 0) {
      cudaCheckError(cudaMallocHost(&data_, size_ * sizeof(T)));
    }
  }

  explicit PinnedHostVector(const size_t size, const T& value) : size_(size) {
    if (size_ > 0) {
      cudaCheckError(cudaMallocHost(&data_, size_ * sizeof(T)));
      std::fill(data_, data_ + size_, value);
    }
  }

  // Copy assignment/constructor not supported
  PinnedHostVector(const PinnedHostVector& other)            = delete;
  PinnedHostVector& operator=(const PinnedHostVector& other) = delete;

  // Move constructor
  PinnedHostVector(PinnedHostVector&& other) noexcept : size_(other.size_), data_(other.data_) {
    other.size_ = 0;
    other.data_ = nullptr;
  }

  // Move assignment
  PinnedHostVector& operator=(PinnedHostVector&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (data_ != nullptr) {
      cudaFreeHost(data_);
    }
    size_       = other.size_;
    data_       = other.data_;
    other.size_ = 0;
    other.data_ = nullptr;
    return *this;
  }

  ~PinnedHostVector() {
    if (data_ != nullptr) {
      cudaFreeHost(data_);
    }
  }

  // Standard container interface
  T*       data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }
  size_t   size() const noexcept { return size_; }
  bool     empty() const noexcept { return size_ == 0; }

  // Element access
  T&       operator[](size_t index) noexcept { return data_[index]; }
  const T& operator[](size_t index) const noexcept { return data_[index]; }

  // Begin and end methods
  T*       begin() noexcept { return data_; }
  const T* begin() const noexcept { return data_; }
  T*       end() noexcept { return data_ + size_; }
  const T* end() const noexcept { return data_ + size_; }

  // Capacity operations
  void resize(const size_t newSize) {
    if (newSize <= size_) {
      size_ = newSize;
      return;
    }
    if (newSize == 0) {
      cudaFreeHost(data_);
      data_ = nullptr;
      size_ = 0;
      return;
    }

    T* newData = nullptr;
    cudaCheckError(cudaMallocHost(&newData, newSize * sizeof(T)));

    if (size_ > 0 && data_ != nullptr) {
      // Copy existing data up to min(oldSize, newSize)
      size_t copySize = std::min(size_, newSize);
      std::copy(data_, data_ + copySize, newData);
      cudaFreeHost(data_);
    }

    data_ = newData;
    size_ = newSize;
  }

  void resize(size_t newSize, const T& value) {
    size_t oldSize = size_;
    resize(newSize);
    if (newSize > oldSize) {
      std::fill(data_ + oldSize, data_ + newSize, value);
    }
  }

  void clear() {
    if (data_ != nullptr) {
      cudaFreeHost(data_);
      data_ = nullptr;
    }
    size_ = 0;
  }

  // Utility methods
  void fill(const T& value) { std::fill(data_, data_ + size_, value); }

  void zero() { std::memset(data_, 0, size_ * sizeof(T)); }

  // Compatibility with AsyncDeviceVector
  template <typename U>
  void copyFromDevice(const AsyncDeviceVector<U>& deviceVec, const cudaStream_t stream = nullptr) {
    static_assert(std::is_same_v<T, U>, "Type mismatch between host and device vectors");
    if (size() != deviceVec.size()) {
      throw std::out_of_range("Size mismatch: host size " + std::to_string(size()) + " != device size " +
                              std::to_string(deviceVec.size()));
    }
    cudaCheckError(cudaMemcpyAsync(data_, deviceVec.data(), size_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
  }

  template <typename U> void copyToDevice(AsyncDeviceVector<U>& deviceVec, const cudaStream_t stream = nullptr) const {
    static_assert(std::is_same_v<T, U>, "Type mismatch between host and device vectors");
    if (size() != deviceVec.size()) {
      throw std::out_of_range("Size mismatch: host size " + std::to_string(size()) + " != device size " +
                              std::to_string(deviceVec.size()));
    }
    cudaCheckError(cudaMemcpyAsync(deviceVec.data(), data_, size_ * sizeof(T), cudaMemcpyHostToDevice, stream));
  }

 private:
  size_t size_ = 0;
  T*     data_ = nullptr;
};

template <typename T> class PinnedHostView {
 public:
  PinnedHostView() = default;
  PinnedHostView(std::span<T> view, std::shared_ptr<std::byte> owner) : view_(view), owner_(std::move(owner)) {}

  T*     data() const noexcept { return view_.data(); }
  size_t size() const noexcept { return view_.size(); }
  bool   empty() const noexcept { return view_.empty(); }

  T&       operator[](size_t index) noexcept { return view_[index]; }
  const T& operator[](size_t index) const noexcept { return view_[index]; }

 private:
  std::span<T>               view_{};
  std::shared_ptr<std::byte> owner_{};
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_HOST_VECTOR_H
