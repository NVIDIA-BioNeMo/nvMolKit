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

#ifndef NVMOLKIT_DEVICE_VECTOR_H
#define NVMOLKIT_DEVICE_VECTOR_H

#include <cuda_runtime.h>

#include <cstdio>
#include <cuda/std/span>
#include <stdexcept>
#include <vector>

#include "src/utils/cuda_error_check.h"

namespace nvMolKit {

//! Simple replacement for thrust device vector allocation and storage, with all async operations.
//! Responsibility is 100% on user for syncs and device management at the moment.
//! Useful for when you want to avoid making a CUDA file with thrust.
template <typename T> class AsyncDeviceVector {
 public:
  explicit AsyncDeviceVector() = default;
  explicit AsyncDeviceVector(size_t size, cudaStream_t stream = 0) : size_(size), stream_(stream) {
    if (size > 0) {
      cudaCheckError(cudaMallocAsync(&data_, size * sizeof(T), stream_));
    }
  }
  AsyncDeviceVector(const AsyncDeviceVector& other) = delete;
  AsyncDeviceVector(AsyncDeviceVector&& other) noexcept
      : size_(other.size_),
        data_(other.data_),
        stream_(other.stream_) {
    other.size_   = 0;
    other.data_   = nullptr;
    other.stream_ = nullptr;
  }
  AsyncDeviceVector& operator=(const AsyncDeviceVector& other) = delete;

  AsyncDeviceVector& operator=(AsyncDeviceVector&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (data_ != nullptr) {
      cudaFreeAsync(data_, stream_);
    }
    stream_     = other.stream_;
    size_       = other.size_;
    data_       = other.data_;
    other.size_ = 0;
    other.data_ = nullptr;
    return *this;
  }

  ~AsyncDeviceVector() { cudaFreeAsync(data_, stream_); }
  T*     data() const noexcept { return data_; }
  size_t size() const noexcept { return size_; }

  void         setStream(cudaStream_t stream) noexcept { stream_ = stream; }
  cudaStream_t stream() const noexcept { return stream_; }

  void zero() {
    if (size_ > 0) {
      cudaCheckError(cudaMemsetAsync(data_, 0, size_ * sizeof(T), stream_));
    }
  }

  //! Returns pointer to an element within the array.
  T* at(size_t index) const {
    if (index >= size_) {
      throw std::out_of_range("Index out of range: " + std::to_string(index) + " >= " + std::to_string(size_));
    }
    return data_ + index;
  }

  void resize(size_t newSize) {
    if (newSize == size_) {
      return;
    }
    if (newSize == 0) {
      cudaFreeAsync(data_, stream_);
      data_ = nullptr;
      size_ = 0;
      return;
    }

    T* newData;
    cudaCheckError(cudaMallocAsync(&newData, newSize * sizeof(T), stream_));
    if (size_ > 0) {
      cudaCheckError(
        cudaMemcpyAsync(newData, data_, std::min(size_, newSize) * sizeof(T), cudaMemcpyDeviceToDevice, stream_));
      cudaFreeAsync(data_, stream_);
    }
    data_ = newData;
    size_ = newSize;
  }

  void setFromVector(const std::vector<T>& hostData) {
    resize(hostData.size());
    if (size_ > 0) {
      copyFromHost(hostData);
    }
  }

  void setFromArray(const T* hostData, size_t size) {
    resize(size);
    if (size_ > 0) {
      copyFromHost(hostData, size);
    }
  }

  //! Copies data from host to device, with basic bounds checking on device.
  //! @param hostData Pointer to host memory to copy from.
  //! @param size Number of elements to copy.
  //! @param firstElementHost Index of the first element in host memory to copy from.
  //! @param firstElementDevice Index of the first element in device memory to copy to.
  void copyFromHost(const T* hostData, size_t size, size_t firstElementHost = 0, size_t firstElementDevice = 0) {
    if (size > size_ - firstElementDevice) {
      const std::string errorStr = "Size mismatch: copy size " + std::to_string(size) + " > " + std::to_string(size_) +
                                   " - first element " + std::to_string(firstElementDevice);
      throw std::out_of_range(errorStr);
    }
    cudaCheckError(cudaMemcpyAsync(data_ + firstElementDevice,
                                   hostData + firstElementHost,
                                   size * sizeof(T),
                                   cudaMemcpyHostToDevice,
                                   stream_));
  }

  //! Copies data from host to device, with basic bounds checking on both host and device.
  void copyFromHost(const std::vector<T>& hostData,
                    size_t                size,
                    size_t                firstElementHost   = 0,
                    size_t                firstElementDevice = 0) {
    if (size + firstElementHost > hostData.size()) {
      const std::string errorStr = "Size mismatch in source vector: copy size " + std::to_string(size) +
                                   " + first element " + std::to_string(firstElementHost) + " > " +
                                   std::to_string(hostData.size());
      throw std::out_of_range(errorStr);
    }
    copyFromHost(hostData.data(), size, firstElementHost, firstElementDevice);
  }

  //! Copy alias for full copy between same size arrays
  void copyFromHost(const std::vector<T>& hostData) {
    if (size() != hostData.size()) {
      throw std::out_of_range("Size mismatch: copy size " + std::to_string(hostData.size()) +
                              " != " + std::to_string(size()));
    }
    copyFromHost(hostData, hostData.size());
  }

  //! Copies data from device to host, with basic bounds checking on device.
  //! @param hostData Pointer to host memory to copy to.
  //! @param size Number of elements to copy.
  //! @param firstElementHost Index of the first element in host memory to copy to.
  //! @param firstElementDevice Index of the first element in device memory to copy from.
  void copyToHost(T* hostData, size_t size, size_t firstElementHost = 0, size_t firstElementDevice = 0) const {
    if (size > size_ - firstElementDevice) {
      const std::string errorStr = "Size mismatch: copy size " + std::to_string(size) + " > " + std::to_string(size_) +
                                   " - first element " + std::to_string(firstElementDevice);
      throw std::out_of_range(errorStr);
    }
    cudaCheckError(cudaMemcpyAsync(hostData + firstElementHost,
                                   data_ + firstElementDevice,
                                   size * sizeof(T),
                                   cudaMemcpyDeviceToHost,
                                   stream_));
  }

  //! Copies data from device to host, with basic bounds checking on both host and device.
  void copyToHost(std::vector<T>& hostData,
                  size_t          size,
                  size_t          firstElementHost   = 0,
                  size_t          firstElementDevice = 0) const {
    if (size + firstElementHost > hostData.size()) {
      const std::string errorStr = "Size mismatch in destination vector: copy size " + std::to_string(size) +
                                   " + first element " + std::to_string(firstElementHost) + " > " +
                                   std::to_string(hostData.size());
      throw std::out_of_range(errorStr);
    }
    copyToHost(hostData.data(), size, firstElementHost, firstElementDevice);
  }

  //! Alias for full copy between same size arrays
  void copyToHost(std::vector<T>& hostData) const {
    if (size() != hostData.size()) {
      throw std::out_of_range("Size mismatch: copy size " + std::to_string(hostData.size()) +
                              " != " + std::to_string(size()));
    }
    copyToHost(hostData, hostData.size());
  }

  T* release() noexcept {
    T* result = data_;
    data_     = nullptr;
    return result;
  }

 private:
  size_t       size_   = 0;
  T*           data_   = nullptr;
  cudaStream_t stream_ = nullptr;
};

// TODO stream option
template <typename T> class AsyncDevicePtr {
 public:
  AsyncDevicePtr() : stream_(nullptr) { cudaCheckError(cudaMallocAsync(&data_, sizeof(T), nullptr)); };
  explicit AsyncDevicePtr(const T& data, const cudaStream_t stream = nullptr) : stream_(stream) {
    cudaCheckError(cudaMallocAsync(&data_, sizeof(T), stream_));
    cudaCheckError(cudaMemcpyAsync(data_, &data, sizeof(T), cudaMemcpyHostToDevice, stream_));
  }
  ~AsyncDevicePtr() noexcept { cudaFreeAsync(data_, stream_); }
  AsyncDevicePtr(const AsyncDevicePtr&)            = delete;
  AsyncDevicePtr& operator=(const AsyncDevicePtr&) = delete;
  AsyncDevicePtr(AsyncDevicePtr&& other) noexcept : data_(other.data_), stream_(other.stream_) {
    other.data_   = nullptr;
    other.stream_ = nullptr;
  }
  AsyncDevicePtr& operator=(AsyncDevicePtr&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    cudaFreeAsync(data_, stream_);
    data_         = other.data_;
    stream_       = other.stream_;
    other.data_   = nullptr;
    other.stream_ = nullptr;
    return *this;
  }
  T* release() noexcept {
    T* result = data_;
    data_     = nullptr;
    return result;
  }
  T*   data() const noexcept { return data_; }
  void set(const T& data) { cudaCheckError(cudaMemcpyAsync(data_, &data, sizeof(T), cudaMemcpyHostToDevice, stream_)); }
  void get(T& data) const { cudaCheckError(cudaMemcpyAsync(&data, data_, sizeof(T), cudaMemcpyDeviceToHost, stream_)); }
  void memSet(int value) { cudaCheckError(cudaMemsetAsync(data_, value, sizeof(T), stream_)); }
  void setStream(const cudaStream_t stream) noexcept { stream_ = stream; }
  cudaStream_t stream() const noexcept { return stream_; }

 private:
  T*           data_ = nullptr;
  cudaStream_t stream_;
};

template <typename T> cuda::std::span<T> toSpan(AsyncDeviceVector<T>& vec) {
  assert(vec.size() > 0);
  return cuda::std::span<T>(vec.data(), vec.size());
}

template <typename T> const cuda::std::span<T> toSpan(const AsyncDeviceVector<T>& vec) {
  assert(vec.size() > 0);
  return cuda::std::span<T>(vec.data(), vec.size());
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_DEVICE_VECTOR_H
