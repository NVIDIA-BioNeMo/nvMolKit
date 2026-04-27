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

#include <cub/cub.cuh>

#include "src/forcefields/dist_geom_kernels.h"
#include "src/forcefields/dist_geom_kernels_device.cuh"
#include "src/forcefields/kernel_utils.cuh"

using namespace nvMolKit::FFKernelUtils;

namespace nvMolKit {
namespace DistGeom {

template <int dimension>
__global__ void DistViolationEnergyKernel(const int      numDist,
                                          const int*     idx1s,
                                          const int*     idx2s,
                                          const double*  lb2s,
                                          const double*  ub2s,
                                          const double*  weights,
                                          const double*  pos,
                                          double*        energyBuffer,
                                          const int*     energyBufferStarts,
                                          const int*     atomIdxToBatchIdx,
                                          const int*     distTermStarts,
                                          const int*     atomStarts,
                                          const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numDist) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2   = idx2s[idx];
      const double lb2    = lb2s[idx];
      const double ub2    = ub2s[idx];
      const double weight = weights[idx];

      const double energy = distViolationEnergy<dimension>(pos, idx1, idx2, lb2, ub2, weight);
      if (energy > 0.0) {
        const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, distTermStarts);
        energyBuffer[outputIdx] += energy;
      }
    }
  }
}

template <int dimension>
__global__ void DistViolationGradientKernel(const int      numDist,
                                            const int*     idx1s,
                                            const int*     idx2s,
                                            const double*  lb2s,
                                            const double*  ub2s,
                                            const double*  weights,
                                            const double*  pos,
                                            double*        grad,
                                            const int*     atomIdxToBatchIdx,
                                            const int*     atomStarts,
                                            const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numDist) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2   = idx2s[idx];
      const double lb2    = lb2s[idx];
      const double ub2    = ub2s[idx];
      const double weight = weights[idx];

      distViolationGrad<dimension>(pos, idx1, idx2, lb2, ub2, weight, grad);
    }
  }
}

template <int dimension>
__global__ void ChiralViolationEnergyKernel(const int      numChiral,
                                            const int*     idx1s,
                                            const int*     idx2s,
                                            const int*     idx3s,
                                            const int*     idx4s,
                                            const double*  volLower,
                                            const double*  volUpper,
                                            const double   weight,
                                            const double*  pos,
                                            double*        energyBuffer,
                                            const int*     energyBufferStarts,
                                            const int*     atomIdxToBatchIdx,
                                            const int*     chiralTermStarts,
                                            const int*     atomStarts,
                                            const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numChiral) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2      = idx2s[idx];
      const int    idx3      = idx3s[idx];
      const int    idx4      = idx4s[idx];
      const double lb        = volLower[idx];
      const double ub        = volUpper[idx];
      const double energy    = chiralViolationEnergy<dimension>(pos, idx1, idx2, idx3, idx4, lb, ub, weight);
      const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, chiralTermStarts);
      energyBuffer[outputIdx] += energy;
    }
  }
}

template <int dimension>
__global__ void ChiralViolationGradientKernel(const int      numChiral,
                                              const int*     idx1s,
                                              const int*     idx2s,
                                              const int*     idx3s,
                                              const int*     idx4s,
                                              const double*  volLower,
                                              const double*  volUpper,
                                              const double   weight,
                                              const double*  pos,
                                              double*        grad,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     atomStarts,
                                              const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numChiral) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2 = idx2s[idx];
      const int    idx3 = idx3s[idx];
      const int    idx4 = idx4s[idx];
      const double lb   = volLower[idx];
      const double ub   = volUpper[idx];
      chiralViolationGrad<dimension>(pos, idx1, idx2, idx3, idx4, lb, ub, weight, grad);
    }
  }
}

template <int dimension>
__global__ void fourthDimEnergyKernel(const int      numFD,
                                      const int*     idxs,
                                      const double   weight,
                                      const double*  pos,
                                      double*        energyBuffer,
                                      const int*     energyBufferStarts,
                                      const int*     atomIdxToBatchIdx,
                                      const int*     fourthTermStarts,
                                      const int*     atomStarts,
                                      const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numFD) {
    const int idx1     = idxs[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      unsigned  pid       = idx1 * dimension + 3;
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, fourthTermStarts);
      energyBuffer[outputIdx] += weight * pos[pid] * pos[pid];
    }
  }
}

template <int dimension>
__global__ void fourthDimGradientKernel(const int      numFD,
                                        const int*     idxs,
                                        const double   weight,
                                        const double*  pos,
                                        double*        grad,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     atomStarts,
                                        const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numFD) {
    const int idx1     = idxs[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      int pid = idx1 * dimension + 3;
      grad[pid] += weight * pos[pid];
    }
  }
}

__global__ void TorsionAngleEnergyKernel(const int      numTorsion,
                                         const int*     idx1s,
                                         const int*     idx2s,
                                         const int*     idx3s,
                                         const int*     idx4s,
                                         const double*  forceConstants,
                                         const int*     signs,
                                         const double*  pos,
                                         double*        energyBuffer,
                                         const int*     energyBufferStarts,
                                         const int*     atomIdxToBatchIdx,
                                         const int*     torsionTermStarts,
                                         const int*     atomStarts,
                                         const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTorsion) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int idx2 = idx2s[idx];
      const int idx3 = idx3s[idx];
      const int idx4 = idx4s[idx];

      const double* fc = &forceConstants[idx * 6];
      const int*    s  = &signs[idx * 6];

      const double energy = torsionAngleEnergy(pos, idx1, idx2, idx3, idx4, fc, s);

      // Accumulate energy in the appropriate buffer
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, torsionTermStarts);
      energyBuffer[outputIdx] += energy;
    }
  }
}

__global__ void TorsionAngleGradientKernel(const int      numTorsion,
                                           const int*     idx1s,
                                           const int*     idx2s,
                                           const int*     idx3s,
                                           const int*     idx4s,
                                           const double*  forceConstants,
                                           const int*     signs,
                                           const double*  pos,
                                           double*        grad,
                                           const int*     atomIdxToBatchIdx,
                                           const int*     atomStarts,
                                           const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTorsion) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      torsionAngleGrad(pos, idx1, idx2s[idx], idx3s[idx], idx4s[idx], &forceConstants[idx * 6], &signs[idx * 6], grad);
    }
  }
}

__global__ void InversionEnergyKernel(const int      numInversion,
                                      const int*     idx1s,
                                      const int*     idx2s,
                                      const int*     idx3s,
                                      const int*     idx4s,
                                      const int*     at2AtomicNum,
                                      const uint8_t* isCBoundToO,
                                      const double*  C0,
                                      const double*  C1,
                                      const double*  C2,
                                      const double*  forceConstants,
                                      const double*  pos,
                                      double*        energyBuffer,
                                      const int*     energyBufferStarts,
                                      const int*     atomIdxToBatchIdx,
                                      const int*     inversionTermStarts,
                                      const int*     atomStarts,
                                      const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numInversion) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int idx2 = idx2s[idx];
      const int idx3 = idx3s[idx];
      const int idx4 = idx4s[idx];

      const double energy =
        inversionEnergy(pos, idx1, idx2, idx3, idx4, C0[idx], C1[idx], C2[idx], forceConstants[idx]);

      // Accumulate energy in the appropriate buffer
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, inversionTermStarts);
      energyBuffer[outputIdx] += energy;
    }
  }
}

__global__ void InversionGradientKernel(const int      numInversion,
                                        const int*     idx1s,
                                        const int*     idx2s,
                                        const int*     idx3s,
                                        const int*     idx4s,
                                        const int*     at2AtomicNum,
                                        const uint8_t* isCBoundToO,
                                        const double*  C0,
                                        const double*  C1,
                                        const double*  C2,
                                        const double*  forceConstants,
                                        const double*  pos,
                                        double*        grad,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     atomStarts,
                                        const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numInversion) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      inversionGrad(pos,
                    idx1,
                    idx2s[idx],
                    idx3s[idx],
                    idx4s[idx],
                    C0[idx],
                    C1[idx],
                    C2[idx],
                    forceConstants[idx],
                    grad);
    }
  }
}

__global__ void DistanceConstraintEnergyKernel(const int      numDist,
                                               const int*     idx1s,
                                               const int*     idx2s,
                                               const double*  minLen,
                                               const double*  maxLen,
                                               const double*  forceConstants,
                                               const double*  pos,
                                               double*        energyBuffer,
                                               const int*     energyBufferStarts,
                                               const int*     atomIdxToBatchIdx,
                                               const int*     distTermStarts,
                                               const int*     atomStarts,
                                               const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numDist) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2          = idx2s[idx];
      const double forceConstant = forceConstants[idx];

      const double energy = distanceConstraintEnergy(pos, idx1, idx2, minLen[idx], maxLen[idx], forceConstant);

      // Accumulate energy in the appropriate buffer
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, distTermStarts);
      energyBuffer[outputIdx] += energy;
    }
  }
}

__global__ void DistanceConstraintGradientKernel(const int      numDist,
                                                 const int*     idx1s,
                                                 const int*     idx2s,
                                                 const double*  minLen,
                                                 const double*  maxLen,
                                                 const double*  forceConstants,
                                                 const double*  pos,
                                                 double*        grad,
                                                 const int*     atomIdxToBatchIdx,
                                                 const int*     atomStarts,
                                                 const uint8_t* activeThisStage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numDist) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      distanceConstraintGrad(pos, idx1, idx2s[idx], minLen[idx], maxLen[idx], forceConstants[idx], grad);
    }
  }
}

__global__ void AngleConstraintEnergyKernel(const int      numAngle,
                                            const int*     idx1s,
                                            const int*     idx2s,
                                            const int*     idx3s,
                                            const double*  minAngle,
                                            const double*  maxAngle,
                                            const double*  pos,
                                            double*        energyBuffer,
                                            const int*     energyBufferStarts,
                                            const int*     atomIdxToBatchIdx,
                                            const int*     angleTermStarts,
                                            const int*     atomStarts,
                                            const uint8_t* activeThisStage,
                                            const double   forceConstant) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numAngle) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      const int    idx2   = idx2s[idx];
      const int    idx3   = idx3s[idx];
      const double minAng = minAngle[idx];
      const double maxAng = maxAngle[idx];

      const double energy = angleConstraintEnergy(pos, idx1, idx2, idx3, minAng, maxAng, forceConstant);

      // Accumulate energy in the appropriate buffer
      const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, angleTermStarts);
      energyBuffer[outputIdx] += energy;
    }
  }
}

__global__ void AngleConstraintGradientKernel(const int      numAngle,
                                              const int*     idx1s,
                                              const int*     idx2s,
                                              const int*     idx3s,
                                              const double*  minAngle,
                                              const double*  maxAngle,
                                              const double*  pos,
                                              double*        grad,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     atomStarts,
                                              const uint8_t* activeThisStage,
                                              const double   forceConstant) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numAngle) {
    const int idx1     = idx1s[idx];
    const int batchIdx = atomIdxToBatchIdx[idx1];

    // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
    if (activeThisStage == nullptr || activeThisStage[batchIdx] == 1) {
      angleConstraintGrad(pos, idx1, idx2s[idx], idx3s[idx], minAngle[idx], maxAngle[idx], forceConstant, grad);
    }
  }
}

cudaError_t launchDistViolationEnergyKernel(const int      numDist,
                                            const int*     idx1,
                                            const int*     idx2,
                                            const double*  lb2,
                                            const double*  ub2,
                                            const double*  weight,
                                            const double*  pos,
                                            double*        energyBuffer,
                                            const int*     energyBufferStarts,
                                            const int*     atomIdxToBatchIdx,
                                            const int*     distTermStarts,
                                            const int*     atomStarts,
                                            const int      dimension,
                                            const uint8_t* activeThisStage,
                                            cudaStream_t   stream) {
  if (numDist == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numDist + blockSize - 1) / blockSize;
  if (dimension == 3) {
    DistViolationEnergyKernel<3><<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                      idx1,
                                                                      idx2,
                                                                      lb2,
                                                                      ub2,
                                                                      weight,
                                                                      pos,
                                                                      energyBuffer,
                                                                      energyBufferStarts,
                                                                      atomIdxToBatchIdx,
                                                                      distTermStarts,
                                                                      atomStarts,
                                                                      activeThisStage);
  } else if (dimension == 4) {
    DistViolationEnergyKernel<4><<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                      idx1,
                                                                      idx2,
                                                                      lb2,
                                                                      ub2,
                                                                      weight,
                                                                      pos,
                                                                      energyBuffer,
                                                                      energyBufferStarts,
                                                                      atomIdxToBatchIdx,
                                                                      distTermStarts,
                                                                      atomStarts,
                                                                      activeThisStage);
  }
  return cudaGetLastError();
}

cudaError_t launchDistViolationGradientKernel(const int      numDist,
                                              const int*     idx1,
                                              const int*     idx2,
                                              const double*  lb2,
                                              const double*  ub2,
                                              const double*  weight,
                                              const double*  pos,
                                              double*        grad,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     atomStarts,
                                              const int      dimension,
                                              const uint8_t* activeThisStage,
                                              cudaStream_t   stream) {
  if (numDist == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numDist + blockSize - 1) / blockSize;
  if (dimension == 3) {
    DistViolationGradientKernel<3><<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                        idx1,
                                                                        idx2,
                                                                        lb2,
                                                                        ub2,
                                                                        weight,
                                                                        pos,
                                                                        grad,
                                                                        atomIdxToBatchIdx,
                                                                        atomStarts,
                                                                        activeThisStage);
  } else if (dimension == 4) {
    DistViolationGradientKernel<4><<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                        idx1,
                                                                        idx2,
                                                                        lb2,
                                                                        ub2,
                                                                        weight,
                                                                        pos,
                                                                        grad,
                                                                        atomIdxToBatchIdx,
                                                                        atomStarts,
                                                                        activeThisStage);
  }
  return cudaGetLastError();
}

cudaError_t launchChiralViolationEnergyKernel(const int      numChiral,
                                              const int*     idx1,
                                              const int*     idx2,
                                              const int*     idx3,
                                              const int*     idx4,
                                              const double*  volLower,
                                              const double*  volUpper,
                                              double         weight,
                                              const double*  pos,
                                              double*        energyBuffer,
                                              const int*     energyBufferStarts,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     chiralTermStarts,
                                              const int*     atomStarts,
                                              const int      dimension,
                                              const uint8_t* activeThisStage,
                                              cudaStream_t   stream) {
  if (numChiral == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numChiral + blockSize - 1) / blockSize;
  if (dimension == 3) {
    ChiralViolationEnergyKernel<3><<<numBlocks, blockSize, 0, stream>>>(numChiral,
                                                                        idx1,
                                                                        idx2,
                                                                        idx3,
                                                                        idx4,
                                                                        volLower,
                                                                        volUpper,
                                                                        weight,
                                                                        pos,
                                                                        energyBuffer,
                                                                        energyBufferStarts,
                                                                        atomIdxToBatchIdx,
                                                                        chiralTermStarts,
                                                                        atomStarts,
                                                                        activeThisStage);
  } else if (dimension == 4) {
    ChiralViolationEnergyKernel<4><<<numBlocks, blockSize, 0, stream>>>(numChiral,
                                                                        idx1,
                                                                        idx2,
                                                                        idx3,
                                                                        idx4,
                                                                        volLower,
                                                                        volUpper,
                                                                        weight,
                                                                        pos,
                                                                        energyBuffer,
                                                                        energyBufferStarts,
                                                                        atomIdxToBatchIdx,
                                                                        chiralTermStarts,
                                                                        atomStarts,
                                                                        activeThisStage);
  }
  return cudaGetLastError();
}

cudaError_t launchChiralViolationGradientKernel(const int      numChiral,
                                                const int*     idx1,
                                                const int*     idx2,
                                                const int*     idx3,
                                                const int*     idx4,
                                                const double*  volLower,
                                                const double*  volUpper,
                                                double         weight,
                                                const double*  pos,
                                                double*        grad,
                                                const int*     atomIdxToBatchIdx,
                                                const int*     atomStarts,
                                                const int      dimension,
                                                const uint8_t* activeThisStage,
                                                cudaStream_t   stream) {
  if (numChiral == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numChiral + blockSize - 1) / blockSize;
  if (dimension == 3) {
    ChiralViolationGradientKernel<3><<<numBlocks, blockSize, 0, stream>>>(numChiral,
                                                                          idx1,
                                                                          idx2,
                                                                          idx3,
                                                                          idx4,
                                                                          volLower,
                                                                          volUpper,
                                                                          weight,
                                                                          pos,
                                                                          grad,
                                                                          atomIdxToBatchIdx,
                                                                          atomStarts,
                                                                          activeThisStage);
  } else if (dimension == 4) {
    ChiralViolationGradientKernel<4><<<numBlocks, blockSize, 0, stream>>>(numChiral,
                                                                          idx1,
                                                                          idx2,
                                                                          idx3,
                                                                          idx4,
                                                                          volLower,
                                                                          volUpper,
                                                                          weight,
                                                                          pos,
                                                                          grad,
                                                                          atomIdxToBatchIdx,
                                                                          atomStarts,
                                                                          activeThisStage);
  }
  return cudaGetLastError();
}

cudaError_t launchFourthDimEnergyKernel(const int      numFD,
                                        const int*     idx,
                                        double         weight,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     fourthTermStarts,
                                        const int*     atomStarts,
                                        const int      dimension,
                                        const uint8_t* activeThisStage,
                                        cudaStream_t   stream) {
  if (numFD == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numFD + blockSize - 1) / blockSize;
  if (dimension == 3) {
    fourthDimEnergyKernel<3><<<numBlocks, blockSize, 0, stream>>>(numFD,
                                                                  idx,
                                                                  weight,
                                                                  pos,
                                                                  energyBuffer,
                                                                  energyBufferStarts,
                                                                  atomIdxToBatchIdx,
                                                                  fourthTermStarts,
                                                                  atomStarts,
                                                                  activeThisStage);
  } else if (dimension == 4) {
    fourthDimEnergyKernel<4><<<numBlocks, blockSize, 0, stream>>>(numFD,
                                                                  idx,
                                                                  weight,
                                                                  pos,
                                                                  energyBuffer,
                                                                  energyBufferStarts,
                                                                  atomIdxToBatchIdx,
                                                                  fourthTermStarts,
                                                                  atomStarts,
                                                                  activeThisStage);
  }
  return cudaGetLastError();
}

cudaError_t launchFourthDimGradientKernel(const int      numFD,
                                          const int*     idx,
                                          double         weight,
                                          const double*  pos,
                                          double*        grad,
                                          const int*     atomIdxToBatchIdx,
                                          const int*     atomStarts,
                                          const int      dimension,
                                          const uint8_t* activeThisStage,
                                          cudaStream_t   stream) {
  if (numFD == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numFD + blockSize - 1) / blockSize;
  if (dimension == 3) {
    fourthDimGradientKernel<3><<<numBlocks, blockSize, 0, stream>>>(numFD,
                                                                    idx,
                                                                    weight,
                                                                    pos,
                                                                    grad,
                                                                    atomIdxToBatchIdx,
                                                                    atomStarts,
                                                                    activeThisStage);
  } else if (dimension == 4) {
    fourthDimGradientKernel<4><<<numBlocks, blockSize, 0, stream>>>(numFD,
                                                                    idx,
                                                                    weight,
                                                                    pos,
                                                                    grad,
                                                                    atomIdxToBatchIdx,
                                                                    atomStarts,
                                                                    activeThisStage);
  }
  return cudaGetLastError();
}

cudaError_t launchTorsionAngleEnergyKernel(const int      numTorsion,
                                           const int*     idx1,
                                           const int*     idx2,
                                           const int*     idx3,
                                           const int*     idx4,
                                           const double*  forceConstant,
                                           const int*     signs,
                                           const double*  pos,
                                           double*        energyBuffer,
                                           const int*     energyBufferStarts,
                                           const int*     atomIdxToBatchIdx,
                                           const int*     torsionTermStarts,
                                           const int*     atomStarts,
                                           const uint8_t* activeThisStage,
                                           cudaStream_t   stream) {
  if (numTorsion == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsion + blockSize - 1) / blockSize;
  TorsionAngleEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsion,
                                                                idx1,
                                                                idx2,
                                                                idx3,
                                                                idx4,
                                                                forceConstant,
                                                                signs,
                                                                pos,
                                                                energyBuffer,
                                                                energyBufferStarts,
                                                                atomIdxToBatchIdx,
                                                                torsionTermStarts,
                                                                atomStarts,
                                                                activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchTorsionAngleGradientKernel(const int      numTorsion,
                                             const int*     idx1,
                                             const int*     idx2,
                                             const int*     idx3,
                                             const int*     idx4,
                                             const double*  forceConstant,
                                             const int*     signs,
                                             const double*  pos,
                                             double*        grad,
                                             const int*     atomIdxToBatchIdx,
                                             const int*     atomStarts,
                                             const uint8_t* activeThisStage,
                                             cudaStream_t   stream) {
  if (numTorsion == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsion + blockSize - 1) / blockSize;
  TorsionAngleGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsion,
                                                                  idx1,
                                                                  idx2,
                                                                  idx3,
                                                                  idx4,
                                                                  forceConstant,
                                                                  signs,
                                                                  pos,
                                                                  grad,
                                                                  atomIdxToBatchIdx,
                                                                  atomStarts,
                                                                  activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchInversionEnergyKernel(const int      numInversion,
                                        const int*     idx1,
                                        const int*     idx2,
                                        const int*     idx3,
                                        const int*     idx4,
                                        const int*     at2AtomicNum,
                                        const uint8_t* isCBoundToO,
                                        const double*  C0,
                                        const double*  C1,
                                        const double*  C2,
                                        const double*  forceConstants,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomIdxToBatchIdx,
                                        const int*     inversionTermStarts,
                                        const int*     atomStarts,
                                        const uint8_t* activeThisStage,
                                        cudaStream_t   stream) {
  if (numInversion == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numInversion + blockSize - 1) / blockSize;
  InversionEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numInversion,
                                                             idx1,
                                                             idx2,
                                                             idx3,
                                                             idx4,
                                                             at2AtomicNum,
                                                             isCBoundToO,
                                                             C0,
                                                             C1,
                                                             C2,
                                                             forceConstants,
                                                             pos,
                                                             energyBuffer,
                                                             energyBufferStarts,
                                                             atomIdxToBatchIdx,
                                                             inversionTermStarts,
                                                             atomStarts,
                                                             activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchInversionGradientKernel(const int      numInversion,
                                          const int*     idx1,
                                          const int*     idx2,
                                          const int*     idx3,
                                          const int*     idx4,
                                          const int*     at2AtomicNum,
                                          const uint8_t* isCBoundToO,
                                          const double*  C0,
                                          const double*  C1,
                                          const double*  C2,
                                          const double*  forceConstants,
                                          const double*  pos,
                                          double*        grad,
                                          const int*     atomIdxToBatchIdx,
                                          const int*     atomStarts,
                                          const uint8_t* activeThisStage,
                                          cudaStream_t   stream) {
  if (numInversion == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numInversion + blockSize - 1) / blockSize;
  InversionGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numInversion,
                                                               idx1,
                                                               idx2,
                                                               idx3,
                                                               idx4,
                                                               at2AtomicNum,
                                                               isCBoundToO,
                                                               C0,
                                                               C1,
                                                               C2,
                                                               forceConstants,
                                                               pos,
                                                               grad,
                                                               atomIdxToBatchIdx,
                                                               atomStarts,
                                                               activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchDistanceConstraintEnergyKernel(const int      numDist,
                                                 const int*     idx1,
                                                 const int*     idx2,
                                                 const double*  minLen,
                                                 const double*  maxLen,
                                                 const double*  forceConstants,
                                                 const double*  pos,
                                                 double*        energyBuffer,
                                                 const int*     energyBufferStarts,
                                                 const int*     atomIdxToBatchIdx,
                                                 const int*     distTermStarts,
                                                 const int*     atomStarts,
                                                 const uint8_t* activeThisStage,
                                                 cudaStream_t   stream) {
  if (numDist == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numDist + blockSize - 1) / blockSize;
  DistanceConstraintEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                      idx1,
                                                                      idx2,
                                                                      minLen,
                                                                      maxLen,
                                                                      forceConstants,
                                                                      pos,
                                                                      energyBuffer,
                                                                      energyBufferStarts,
                                                                      atomIdxToBatchIdx,
                                                                      distTermStarts,
                                                                      atomStarts,
                                                                      activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchDistanceConstraintGradientKernel(const int      numDist,
                                                   const int*     idx1s,
                                                   const int*     idx2s,
                                                   const double*  minLen,
                                                   const double*  maxLen,
                                                   const double*  forceConstants,
                                                   const double*  pos,
                                                   double*        grad,
                                                   const int*     atomIdxToBatchIdx,
                                                   const int*     atomStarts,
                                                   const uint8_t* activeThisStage,
                                                   cudaStream_t   stream) {
  if (numDist == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numDist + blockSize - 1) / blockSize;
  DistanceConstraintGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numDist,
                                                                        idx1s,
                                                                        idx2s,
                                                                        minLen,
                                                                        maxLen,
                                                                        forceConstants,
                                                                        pos,
                                                                        grad,
                                                                        atomIdxToBatchIdx,
                                                                        atomStarts,
                                                                        activeThisStage);
  return cudaGetLastError();
}

cudaError_t launchAngleConstraintEnergyKernel(const int      numAngle,
                                              const int*     idx1,
                                              const int*     idx2,
                                              const int*     idx3,
                                              const double*  minAngle,
                                              const double*  maxAngle,
                                              const double*  pos,
                                              double*        energyBuffer,
                                              const int*     energyBufferStarts,
                                              const int*     atomIdxToBatchIdx,
                                              const int*     angleTermStarts,
                                              const int*     atomStarts,
                                              const uint8_t* activeThisStage,
                                              const double   forceConstant,
                                              cudaStream_t   stream) {
  if (numAngle == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngle + blockSize - 1) / blockSize;
  AngleConstraintEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numAngle,
                                                                   idx1,
                                                                   idx2,
                                                                   idx3,
                                                                   minAngle,
                                                                   maxAngle,
                                                                   pos,
                                                                   energyBuffer,
                                                                   energyBufferStarts,
                                                                   atomIdxToBatchIdx,
                                                                   angleTermStarts,
                                                                   atomStarts,
                                                                   activeThisStage,
                                                                   forceConstant);
  return cudaGetLastError();
}

cudaError_t launchAngleConstraintGradientKernel(const int      numAngle,
                                                const int*     idx1,
                                                const int*     idx2,
                                                const int*     idx3,
                                                const double*  minAngle,
                                                const double*  maxAngle,
                                                const double*  pos,
                                                double*        grad,
                                                const int*     atomIdxToBatchIdx,
                                                const int*     atomStarts,
                                                const uint8_t* activeThisStage,
                                                const double   forceConstant,
                                                cudaStream_t   stream) {
  if (numAngle == 0) {
    return cudaSuccess;
  }
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngle + blockSize - 1) / blockSize;
  AngleConstraintGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numAngle,
                                                                     idx1,
                                                                     idx2,
                                                                     idx3,
                                                                     minAngle,
                                                                     maxAngle,
                                                                     pos,
                                                                     grad,
                                                                     atomIdxToBatchIdx,
                                                                     atomStarts,
                                                                     activeThisStage,
                                                                     forceConstant);
  return cudaGetLastError();
}

cudaError_t launchReduceEnergiesKernel(const int      numBlocks,
                                       const double*  energyBuffer,
                                       const int*     energyBufferBlockIdxToBatchIdx,
                                       double*        outs,
                                       const uint8_t* activeThisStage,
                                       cudaStream_t   stream) {
  reduceEnergiesKernel<<<numBlocks, blockSizeEnergyReduction, 0, stream>>>(energyBuffer,
                                                                           energyBufferBlockIdxToBatchIdx,
                                                                           outs,
                                                                           activeThisStage);
  return cudaGetLastError();
}

constexpr int blockSizePerMol = 128;

template <int dimension>
__global__ void combinedEnergiesKernel(const EnergyForceContribsDevicePtr* terms,
                                       const BatchedIndicesDevicePtr*      systemIndices,
                                       const double*                       coords,
                                       double*                             energies,
                                       const double                        chiralWeight,
                                       const double                        fourthDimWeight,
                                       const uint8_t*                      activeThisStage) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;

  if (activeThisStage != nullptr && activeThisStage[molIdx] == 0) {
    if (tid == 0) {
      energies[molIdx] = 0.0;
    }
    return;
  }

  using BlockReduce = cub::BlockReduce<double, blockSizePerMol>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  const int     atomStart = systemIndices->atomStarts[molIdx];
  const double* molCoords = coords + atomStart * dimension;
  const double  threadEnergy =
    molEnergyDG<dimension>(*terms, *systemIndices, molCoords, molIdx, chiralWeight, fourthDimWeight, tid);
  const double blockEnergy = BlockReduce(tempStorage).Sum(threadEnergy);

  if (tid == 0) {
    energies[molIdx] = blockEnergy;
  }
}

template <int dimension>
__global__ void combinedGradKernel(const EnergyForceContribsDevicePtr* terms,
                                   const BatchedIndicesDevicePtr*      systemIndices,
                                   const double*                       coords,
                                   double*                             grad,
                                   const double                        chiralWeight,
                                   const double                        fourthDimWeight,
                                   const uint8_t*                      activeThisStage) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;

  if (activeThisStage != nullptr && activeThisStage[molIdx] == 0) {
    return;
  }

  const int atomStart = systemIndices->atomStarts[molIdx];
  const int atomEnd   = systemIndices->atomStarts[molIdx + 1];
  const int numAtoms  = atomEnd - atomStart;

  constexpr int     maxAtomSize = 256;
  __shared__ double accumGrad[maxAtomSize * 4];  // Support up to 4D

  const bool useSharedMem = numAtoms * dimension <= maxAtomSize * 4;
  double*    molGradBase  = useSharedMem ? accumGrad : grad + atomStart * dimension;

  for (int i = tid; i < numAtoms * dimension; i += blockSizePerMol) {
    molGradBase[i] = 0.0;
  }
  __syncthreads();

  const double* molCoords = coords + atomStart * dimension;
  molGradDG<dimension>(*terms, *systemIndices, molCoords, molGradBase, molIdx, chiralWeight, fourthDimWeight, tid);
  __syncthreads();

  if (useSharedMem) {
    double* globalGrad = grad + (atomStart * dimension);
    for (int i = tid; i < numAtoms * dimension; i += blockSizePerMol) {
      globalGrad[i] = molGradBase[i];
    }
  }
}

cudaError_t launchBlockPerMolEnergyKernel(int                                 numMols,
                                          const EnergyForceContribsDevicePtr& terms,
                                          const BatchedIndicesDevicePtr&      systemIndices,
                                          const double*                       coords,
                                          double*                             energies,
                                          const int                           dimension,
                                          const double                        chiralWeight,
                                          const double                        fourthDimWeight,
                                          const uint8_t*                      activeThisStage,
                                          cudaStream_t                        stream) {
  const AsyncDevicePtr<EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndicesDevicePtr>      devSysIdx(systemIndices, stream);
  if (dimension == 3) {
    combinedEnergiesKernel<3><<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(),
                                                                       devSysIdx.data(),
                                                                       coords,
                                                                       energies,
                                                                       chiralWeight,
                                                                       fourthDimWeight,
                                                                       activeThisStage);
  } else {
    combinedEnergiesKernel<4><<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(),
                                                                       devSysIdx.data(),
                                                                       coords,
                                                                       energies,
                                                                       chiralWeight,
                                                                       fourthDimWeight,
                                                                       activeThisStage);
  }
  return cudaGetLastError();
}

cudaError_t launchBlockPerMolGradKernel(int                                 numMols,
                                        const EnergyForceContribsDevicePtr& terms,
                                        const BatchedIndicesDevicePtr&      systemIndices,
                                        const double*                       coords,
                                        double*                             grad,
                                        const int                           dimension,
                                        const double                        chiralWeight,
                                        const double                        fourthDimWeight,
                                        const uint8_t*                      activeThisStage,
                                        cudaStream_t                        stream) {
  const AsyncDevicePtr<EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndicesDevicePtr>      devSysIdx(systemIndices, stream);
  if (dimension == 3) {
    combinedGradKernel<3><<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(),
                                                                   devSysIdx.data(),
                                                                   coords,
                                                                   grad,
                                                                   chiralWeight,
                                                                   fourthDimWeight,
                                                                   activeThisStage);
  } else {
    combinedGradKernel<4><<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(),
                                                                   devSysIdx.data(),
                                                                   coords,
                                                                   grad,
                                                                   chiralWeight,
                                                                   fourthDimWeight,
                                                                   activeThisStage);
  }
  return cudaGetLastError();
}

// ETK (3D) combined kernels
__global__ void combinedEnergiesKernelETK(const Energy3DForceContribsDevicePtr* terms,
                                          const BatchedIndices3DDevicePtr*      systemIndices,
                                          const double*                         coords,
                                          double*                               energies,
                                          const uint8_t*                        activeThisStage) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;

  if (activeThisStage != nullptr && activeThisStage[molIdx] == 0) {
    if (tid == 0) {
      energies[molIdx] = 0.0;
    }
    return;
  }

  using BlockReduce = cub::BlockReduce<double, blockSizePerMol>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  const int     atomStart    = systemIndices->atomStarts[molIdx];
  const double* molCoords    = coords + atomStart * 4;  // ETK uses 4D
  const double  threadEnergy = molEnergyETK(*terms, *systemIndices, molCoords, molIdx, tid);
  const double  blockEnergy  = BlockReduce(tempStorage).Sum(threadEnergy);

  if (tid == 0) {
    energies[molIdx] = blockEnergy;
  }
}

cudaError_t launchBlockPerMolEnergyKernelETK(int                                   numMols,
                                             const Energy3DForceContribsDevicePtr& terms,
                                             const BatchedIndices3DDevicePtr&      systemIndices,
                                             const double*                         coords,
                                             double*                               energies,
                                             const uint8_t*                        activeThisStage,
                                             cudaStream_t                          stream) {
  const AsyncDevicePtr<Energy3DForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndices3DDevicePtr>      devSysIdx(systemIndices, stream);
  combinedEnergiesKernelETK<<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(),
                                                                     devSysIdx.data(),
                                                                     coords,
                                                                     energies,
                                                                     activeThisStage);
  return cudaGetLastError();
}

// ETK (3D) combined gradient kernel
__global__ void combinedGradKernelETK(const Energy3DForceContribsDevicePtr* terms,
                                      const BatchedIndices3DDevicePtr*      systemIndices,
                                      const double*                         coords,
                                      double*                               grad,
                                      const uint8_t*                        activeThisStage) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;

  if (activeThisStage != nullptr && activeThisStage[molIdx] == 0) {
    return;
  }

  const int     atomStart = systemIndices->atomStarts[molIdx];
  const double* molCoords = coords + atomStart * 4;  // ETK uses 4D
  double*       molGrad   = grad + atomStart * 4;    // Offset to molecule start for ETK (4D)

  molGradETK(*terms, *systemIndices, molCoords, molGrad, molIdx, tid);
}

cudaError_t launchBlockPerMolGradKernelETK(int                                   numMols,
                                           const Energy3DForceContribsDevicePtr& terms,
                                           const BatchedIndices3DDevicePtr&      systemIndices,
                                           const double*                         coords,
                                           double*                               grad,
                                           const uint8_t*                        activeThisStage,
                                           cudaStream_t                          stream) {
  const AsyncDevicePtr<Energy3DForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndices3DDevicePtr>      devSysIdx(systemIndices, stream);
  combinedGradKernelETK<<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(),
                                                                 devSysIdx.data(),
                                                                 coords,
                                                                 grad,
                                                                 activeThisStage);
  return cudaGetLastError();
}
}  // namespace DistGeom
}  // namespace nvMolKit
