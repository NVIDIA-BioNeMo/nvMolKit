// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/utils/device_convert.cuh"
#include "src/utils/device_convert.h"

namespace nvMolKit::detail {

cudaError_t convertDeviceArray(float* dst, const double* src, const int count, cudaStream_t stream) {
  return convertDeviceArray<float, double>(dst, src, count, stream);
}

cudaError_t convertDeviceArray(double* dst, const float* src, const int count, cudaStream_t stream) {
  return convertDeviceArray<double, float>(dst, src, count, stream);
}

}  // namespace nvMolKit::detail
