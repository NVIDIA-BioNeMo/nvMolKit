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

#ifndef NVMOLKIT_FF_KERNEL_UTILS_H
#define NVMOLKIT_FF_KERNEL_UTILS_H

#include <cuda_runtime.h>

#include <cstdint>

#include "src/utils/device_vector.h"

namespace nvMolKit {
namespace FFKernelUtils {

//! Broadcasts the value from lane 0 of the warp to all lanes in the warp.
//! Helps the compiler understand that all lanes will have the same value after this call.
__device__ __forceinline__ int mark_warp_uniform(const int input) {
  return __shfl_sync(0xffffffff, input, 0);
}

__device__ __forceinline__ double distanceSquared(const double* pos,
                                                  const int     idx1,
                                                  const int     idx2,
                                                  const int     dim = 3) {
  const double dx   = pos[dim * idx1 + 0] - pos[dim * idx2 + 0];
  const double dy   = pos[dim * idx1 + 1] - pos[dim * idx2 + 1];
  const double dz   = pos[dim * idx1 + 2] - pos[dim * idx2 + 2];
  double       dist = dx * dx + dy * dy + dz * dz;
  if (dim == 4) {
    const double dw = pos[dim * idx1 + 3] - pos[dim * idx2 + 3];
    dist += dw * dw;
  }
  return dist;
}

__device__ __forceinline__ double distanceSquaredPosIdx(const double* pos,
                                                        const int     posIdx1,
                                                        const int     posIdx2,
                                                        const int     dim) {
  const double dx   = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  const double dy   = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  const double dz   = pos[posIdx1 + 2] - pos[posIdx2 + 2];
  double       dist = dx * dx + dy * dy + dz * dz;
  if (dim == 4) {
    const double dw = pos[posIdx1 + 3] - pos[posIdx2 + 3];
    dist += dw * dw;
  }
  return dist;
}

template <int fixedDimension, typename floatType = double>
__device__ __forceinline__ floatType distanceSquaredPosIdx(const double* pos, const int posIdx1, const int posIdx2) {
  const floatType dx   = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  const floatType dy   = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  const floatType dz   = pos[posIdx1 + 2] - pos[posIdx2 + 2];
  floatType       dist = dx * dx + dy * dy + dz * dz;
  if constexpr (fixedDimension == 4) {
    const floatType dw = pos[posIdx1 + 3] - pos[posIdx2 + 3];
    dist += dw * dw;
  }
  return dist;
}

template <typename floatTypeIn = double, typename floatTypeOut = double>
__device__ __forceinline__ double distanceSquaredWithComponents(const floatTypeIn* pos,
                                                                const int          idx1,
                                                                const int          idx2,
                                                                floatTypeOut&      dx,
                                                                floatTypeOut&      dy,
                                                                floatTypeOut&      dz) {
  dx = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  dy = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  dz = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];
  return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ double clamp(const double val, const double minVal, const double maxVal) {
  return fmax(minVal, fmin(maxVal, val));
}

__device__ __forceinline__ float clamp(const float val, const float minVal, const float maxVal) {
  return fmaxf(minVal, fminf(maxVal, val));
}

template <typename TIn, typename TOut>
__device__ __forceinline__ void crossProduct(const TIn& x1,
                                             const TIn& y1,
                                             const TIn& z1,
                                             const TIn& x2,
                                             const TIn& y2,
                                             const TIn& z2,
                                             TOut&      x,
                                             TOut&      y,
                                             TOut&      z) {
  x = y1 * z2 - z1 * y2;
  y = z1 * x2 - x1 * z2;
  z = x1 * y2 - y1 * x2;
}

template <typename T>
__device__ __forceinline__ T dotProduct(const T& x1, const T& y1, const T& z1, const T& x2, const T& y2, const T& z2) {
  return x1 * x2 + y1 * y2 + z1 * z2;
}

__device__ __forceinline__ void clipToOne(double& x) {
  x = fmax(-1.0, fmin(1.0, x));
}

__device__ __forceinline__ bool isDoubleZero(const double val) {
  return ((val < 1.0e-10) && (val > -1.0e-10));
}

__device__ __forceinline__ bool isFloatZero(const float val) {
  return ((val < 1.0e-10f) && (val > -1.0e-10f));
}

__device__ __forceinline__ int getEnergyAccumulatorIndex(const int  absoluteIdx,
                                                         const int  batchIdx,
                                                         const int* energyBufferStarts,
                                                         const int* termBatchStarts) {
  const int energyBufferStart = energyBufferStarts[batchIdx];
  const int termStart         = termBatchStarts[batchIdx];
  const int termRelativeIdx   = absoluteIdx - termStart;
  return termRelativeIdx + energyBufferStart;
}

__device__ __forceinline__ void normalizeVector(double& x, double& y, double& z) {
  const double norm = sqrt(x * x + y * y + z * z);
  x /= norm;
  y /= norm;
  z /= norm;
}

__global__ void reduceEnergiesKernel(const double*  energyBuffer,
                                     const int*     energyBufferBlockIdxToBatchIdx,
                                     double*        outs,
                                     const uint8_t* activeThisStage = nullptr);

constexpr int blockSizeEnergyReduction = 128;

template <typename BatchedMolecularSystemHost, typename BatchedMolecularDeviceBuffers>
void allocateIntermediateBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                 BatchedMolecularDeviceBuffers&    molSystemDevice,
                                 const int                         numMols) {
  const int totalNumEnergyTerms = molSystemHost.indices.energyBufferStarts.back();
  molSystemDevice.energyBuffer.resize(totalNumEnergyTerms);
  molSystemDevice.energyBuffer.zero();
  molSystemDevice.energyOuts.resize(numMols);
  molSystemDevice.energyOuts.zero();
}

}  // namespace FFKernelUtils
}  // namespace nvMolKit

#endif  // NVMOLKIT_FF_KERNEL_UTILS_H
