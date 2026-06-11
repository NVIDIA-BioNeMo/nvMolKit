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

#ifndef NVMOLKIT_DEVICE_TIMINGS_CUH
#define NVMOLKIT_DEVICE_TIMINGS_CUH

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

#include "src/utils/cuda_error_check.h"

namespace nvMolKit {

#ifndef NVMOLKIT_ENABLE_DEVICE_TIMINGS
// Set NVMOLKIT_ENABLE_DEVICE_TIMINGS=1 per translation unit to enable timings.
#define NVMOLKIT_ENABLE_DEVICE_TIMINGS 0
#endif

constexpr bool enableDeviceTimings = (NVMOLKIT_ENABLE_DEVICE_TIMINGS != 0);

// ============================================================================
// Device-side struct (passed to kernels)
// ============================================================================

struct DeviceTimingsData {
  static constexpr int kMaxSections = 20;
  long long int        totals[kMaxSections];
  long long int        counts[kMaxSections];
};

// ============================================================================
// Device Macros
// ============================================================================

#define DEVICE_TIMING_START(timings_ptr, section, start_var) \
  do {                                                       \
    if constexpr (nvMolKit::enableDeviceTimings) {           \
      if (blockIdx.x == 0 && threadIdx.x == 0) {             \
        start_var = clock64();                               \
      }                                                      \
      __threadfence_block();                                 \
    }                                                        \
  } while (0)

#define DEVICE_TIMING_END(timings_ptr, section, start_var)       \
  do {                                                           \
    if constexpr (nvMolKit::enableDeviceTimings) {               \
      if (blockIdx.x == 0 && threadIdx.x == 0) {                 \
        (timings_ptr)->totals[section] += clock64() - start_var; \
        (timings_ptr)->counts[section] += 1;                     \
      }                                                          \
      __threadfence_block();                                     \
    }                                                            \
  } while (0)

// ============================================================================
// Host RAII Wrapper
// ============================================================================

/**
 * @brief RAII wrapper for device-side kernel timing instrumentation.
 *
 * Manages device memory for timing data and provides labeled output.
 * Enable by setting enableDeviceTimings = true.
 */
class DeviceTimings {
 public:
  DeviceTimings() : labels_(DeviceTimingsData::kMaxSections) {
    if constexpr (enableDeviceTimings) {
      cudaCheckError(cudaMalloc(&d_data_, sizeof(DeviceTimingsData)));
      reset();
    }
  }

  ~DeviceTimings() {
    if constexpr (enableDeviceTimings) {
      if (d_data_ != nullptr) {
        cudaCheckErrorNoThrow(cudaFree(d_data_));
      }
    }
  }

  DeviceTimings(const DeviceTimings&)            = delete;
  DeviceTimings& operator=(const DeviceTimings&) = delete;
  DeviceTimings(DeviceTimings&&)                 = delete;
  DeviceTimings& operator=(DeviceTimings&&)      = delete;

  /// Get device pointer to pass to kernels
  DeviceTimingsData*       data() { return d_data_; }
  const DeviceTimingsData* data() const { return d_data_; }

  /// Set a label for a section index
  void setLabel(int section, const std::string& label) {
    if (section >= 0 && section < DeviceTimingsData::kMaxSections) {
      labels_[section] = label;
    }
  }

  /// Reset all timings to zero
  void reset() {
    if constexpr (enableDeviceTimings) {
      cudaCheckError(cudaMemset(d_data_, 0, sizeof(DeviceTimingsData)));
    }
  }

  /// Copy timings from device to host and print
  void print() const {
    if constexpr (enableDeviceTimings) {
      DeviceTimingsData host_data{};
      cudaCheckError(cudaMemcpy(&host_data, d_data_, sizeof(DeviceTimingsData), cudaMemcpyDeviceToHost));

      printf("Device Timings:\n");
      for (int i = 0; i < DeviceTimingsData::kMaxSections; ++i) {
        if (host_data.counts[i] > 0) {
          const char*   label = labels_[i].empty() ? "(unlabeled)" : labels_[i].c_str();
          long long int avg   = host_data.totals[i] / host_data.counts[i];
          printf("  [%2d] %-20s : %12lld cycles, %6lld hits, avg %12lld cycles/hit\n",
                 i,
                 label,
                 host_data.totals[i],
                 host_data.counts[i],
                 avg);
        }
      }
    }
  }

 private:
  DeviceTimingsData*       d_data_ = nullptr;
  std::vector<std::string> labels_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_DEVICE_TIMINGS_CUH