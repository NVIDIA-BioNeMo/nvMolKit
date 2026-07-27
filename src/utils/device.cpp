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

#include "src/utils/device.h"

#include <cuda_runtime.h>

#include "src/utils/cuda_error_check.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {

int countCudaDevices() {
  int device_count = 0;
  cudaCheckError(cudaGetDeviceCount(&device_count));
  return device_count;
}

WithDevice::WithDevice(int device_id) {
  cudaCheckError(cudaGetDevice(&original_device_id_));
  cudaCheckError(cudaSetDevice(device_id));
}

WithDevice::~WithDevice() {
  cudaCheckErrorNoThrow(cudaSetDevice(original_device_id_));
}

namespace {

//! Returns true if `stream` belongs to the current device.
//! cudaStreamQuery accepts streams from any device in the process, so it cannot answer this on its
//! own.
bool streamIsOnCurrentDevice(cudaStream_t stream) {
#if CUDART_VERSION >= 12080
  int currentDevice = -1;
  if (cudaGetDevice(&currentDevice) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  int streamDevice = -1;
  if (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return streamDevice == currentDevice;
#else
  // cudaStreamGetDevice needs CUDA 12.8. Below that, probe with an event instead: an event and the
  // stream it is recorded into must share a CUDA context, so recording an event created on the
  // current device fails with cudaErrorInvalidResourceHandle if the stream belongs to another one.
  cudaEvent_t probe = nullptr;
  if (cudaEventCreateWithFlags(&probe, cudaEventDisableTiming) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  const cudaError_t recordErr = cudaEventRecord(probe, stream);
  cudaCheckErrorNoThrow(cudaEventDestroy(probe));
  if (recordErr != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return true;
#endif
}

}  // namespace

std::optional<cudaStream_t> acquireExternalStream(std::uintptr_t streamPtr) {
  auto stream = reinterpret_cast<cudaStream_t>(streamPtr);
  if (streamPtr == 0) {
    return stream;
  }
  cudaError_t err = cudaStreamQuery(stream);
  if (err != cudaSuccess && err != cudaErrorNotReady) {
    // Clear the sticky error state
    cudaGetLastError();
    return std::nullopt;
  }
  if (!streamIsOnCurrentDevice(stream)) {
    return std::nullopt;
  }
  return stream;
}

size_t getDeviceFreeMemory() {
  size_t free  = 0;
  size_t total = 0;
  cudaCheckError(cudaMemGetInfo(&free, &total));
  return free;
}

ScopedStream::ScopedStream(const char* name) {
  cudaCheckError(cudaStreamCreateWithFlags(&original_stream_, cudaStreamNonBlocking));
  if (name != nullptr) {
    nvtxNameCudaStreamA(original_stream_, name);
  }
}

ScopedStream::~ScopedStream() noexcept {
  if (original_stream_ == nullptr) {
    return;
  }
  cudaCheckErrorNoThrow(cudaStreamSynchronize(original_stream_));
  cudaCheckErrorNoThrow(cudaStreamDestroy(original_stream_));
}

ScopedStream::ScopedStream(ScopedStream&& other) noexcept : original_stream_(other.original_stream_) {
  other.original_stream_ = nullptr;
}

ScopedStreamWithPriority::ScopedStreamWithPriority(int priority, const char* name) {
  int leastPriority    = 0;
  int greatestPriority = 0;
  cudaCheckError(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

  const int clampedPriority = std::max(greatestPriority, std::min(leastPriority, priority));
  cudaCheckError(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, clampedPriority));
  if (name != nullptr) {
    nvtxNameCudaStreamA(stream_, name);
  }
}

ScopedStreamWithPriority::~ScopedStreamWithPriority() noexcept {
  if (stream_ == nullptr) {
    return;
  }
  cudaCheckErrorNoThrow(cudaStreamSynchronize(stream_));
  cudaCheckErrorNoThrow(cudaStreamDestroy(stream_));
}

ScopedStreamWithPriority::ScopedStreamWithPriority(ScopedStreamWithPriority&& other) noexcept : stream_(other.stream_) {
  other.stream_ = nullptr;
}

ScopedStreamWithPriority& ScopedStreamWithPriority::operator=(ScopedStreamWithPriority&& other) noexcept {
  if (stream_ != nullptr && stream_ != other.stream_) {
    cudaCheckErrorNoThrow(cudaStreamSynchronize(stream_));
    cudaCheckErrorNoThrow(cudaStreamDestroy(stream_));
  }
  stream_       = other.stream_;
  other.stream_ = nullptr;
  return *this;
}

ScopedCudaEvent::ScopedCudaEvent() {
  cudaCheckError(cudaEventCreateWithFlags(&original_event_, cudaEventDisableTiming));
}

ScopedCudaEvent::~ScopedCudaEvent() noexcept {
  if (original_event_ == nullptr) {
    return;
  }
  cudaCheckErrorNoThrow(cudaEventDestroy(original_event_));
}

ScopedCudaEvent::ScopedCudaEvent(ScopedCudaEvent&& other) noexcept : original_event_(other.original_event_) {
  other.original_event_ = nullptr;
}

ScopedCudaEvent& ScopedCudaEvent::operator=(ScopedCudaEvent&& other) noexcept {
  if (original_event_ != nullptr && original_event_ != other.original_event_) {
    cudaCheckErrorNoThrow(cudaEventDestroy(original_event_));
  }
  original_event_       = other.original_event_;
  other.original_event_ = nullptr;
  return *this;
}

ScopedStream& ScopedStream::operator=(ScopedStream&& other) noexcept {
  if (original_stream_ != nullptr && original_stream_ != other.original_stream_) {
    cudaCheckErrorNoThrow(cudaStreamDestroy(original_stream_));
  }
  original_stream_       = other.original_stream_;
  other.original_stream_ = nullptr;
  return *this;
}

}  // namespace nvMolKit
