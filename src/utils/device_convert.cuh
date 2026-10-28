// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DEVICE_CONVERT_CUH
#define NVMOLKIT_DEVICE_CONVERT_CUH

#include <cuda_runtime.h>

namespace nvMolKit::detail {

template <typename To, typename From>
__global__ void convertDeviceArrayKernel(To* dst, const From* src, const int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] = static_cast<To>(src[idx]);
  }
}

template <typename To, typename From>
cudaError_t convertDeviceArray(To* dst, const From* src, const int count, cudaStream_t stream) {
  if (count == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  convertDeviceArrayKernel<<<(count + blockSize - 1) / blockSize, blockSize, 0, stream>>>(dst, src, count);
  return cudaGetLastError();
}

}  // namespace nvMolKit::detail

#endif  // NVMOLKIT_DEVICE_CONVERT_CUH
