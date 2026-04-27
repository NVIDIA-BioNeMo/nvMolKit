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

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "src/utils/cuda_error_check.h"
#include "src/utils/device_timings.cuh"

using namespace nvMolKit;

namespace {

__device__ __forceinline__ void spinCycles(long long cycles) {
  const long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

__global__ void timingKernel(DeviceTimingsData* timings, long long short_cycles, long long long_cycles) {
  unsigned long long start = 0;
  DEVICE_TIMING_START(timings, 0, start);
  spinCycles(short_cycles);
  DEVICE_TIMING_END(timings, 0, start);

  DEVICE_TIMING_START(timings, 1, start);
  spinCycles(long_cycles);
  DEVICE_TIMING_END(timings, 1, start);
}

}  // namespace

TEST(DeviceTimings, RecordsSleepDurations) {
  int device_count = 0;
  cudaCheckError(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    GTEST_SKIP();
  }

  if (!enableDeviceTimings) {
    GTEST_SKIP();
  }

  const long long short_cycles = 5'000'000LL;
  const long long long_cycles  = 10'000'000LL;

  DeviceTimings timings;
  ASSERT_NE(timings.data(), nullptr);
  timings.reset();

  timingKernel<<<1, 1>>>(timings.data(), short_cycles, long_cycles);
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());

  DeviceTimingsData host_data{};
  cudaCheckError(cudaMemcpy(&host_data, timings.data(), sizeof(DeviceTimingsData), cudaMemcpyDeviceToHost));

  EXPECT_EQ(host_data.counts[0], 1);
  EXPECT_EQ(host_data.counts[1], 1);

  EXPECT_GE(host_data.totals[0], short_cycles * 9 / 10);
  EXPECT_LE(host_data.totals[0], short_cycles * 11 / 10);

  EXPECT_GE(host_data.totals[1], long_cycles * 9 / 10);
  EXPECT_LE(host_data.totals[1], long_cycles * 11 / 10);
}
