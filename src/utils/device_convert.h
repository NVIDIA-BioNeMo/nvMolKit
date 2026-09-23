// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DEVICE_CONVERT_H
#define NVMOLKIT_DEVICE_CONVERT_H

#include <cuda_runtime.h>

namespace nvMolKit::detail {

cudaError_t convertDeviceArray(float* dst, const double* src, int count, cudaStream_t stream);
cudaError_t convertDeviceArray(double* dst, const float* src, int count, cudaStream_t stream);

}  // namespace nvMolKit::detail

#endif  // NVMOLKIT_DEVICE_CONVERT_H
