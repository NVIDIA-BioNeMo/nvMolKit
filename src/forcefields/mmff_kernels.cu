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

#include "kernel_utils.cuh"
#include "mmff_kernels.h"
#include "mmff_kernels_device.cuh"

using namespace nvMolKit::FFKernelUtils;

__global__ void bondStretchEnergyKernel(const int     numBonds,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const double* r0,
                                        const double* kb,
                                        const double* pos,
                                        double*       energyBuffer,
                                        const int*    energyBufferStarts,
                                        const int*    atomBatchMap,
                                        const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numBonds) {
    const double energy    = bondStretchEnergy(pos, idx1[idx], idx2[idx], r0[idx], kb[idx]);
    const int    batchIdx  = atomBatchMap[idx1[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void bondStretchGradKernel(const int     numBonds,
                                      const int*    idx1s,
                                      const int*    idx2s,
                                      const double* r0,
                                      const double* kb,
                                      const double* pos,
                                      double*       grad) {
  const int bondIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (bondIdx < numBonds) {
    bondStretchGrad(pos, idx1s[bondIdx], idx2s[bondIdx], r0[bondIdx], kb[bondIdx], grad);
  }
}

__global__ void angleBendEnergyKernel(const int      numAngles,
                                      const int*     idx1s,
                                      const int*     idx2s,
                                      const int*     idx3s,
                                      const double*  theta0,
                                      const double*  ka,
                                      const uint8_t* isLinear,
                                      const double*  pos,
                                      double*        energyBuffer,
                                      const int*     energyBufferStarts,
                                      const int*     atomBatchMap,
                                      const int*     termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numAngles) {
    const double energy = angleBendEnergy(pos, idx1s[idx], idx2s[idx], idx3s[idx], theta0[idx], ka[idx], isLinear[idx]);

    const int batchIdx  = atomBatchMap[idx1s[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void angleBendGradientKernel(const int      numAngles,
                                        const int*     idx1s,
                                        const int*     idx2s,
                                        const int*     idx3s,
                                        const double*  theta0,
                                        const double*  ka,
                                        const uint8_t* isLinear,
                                        const double*  pos,
                                        double*        grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numAngles) {
    angleBendGrad(idx1s[idx], idx2s[idx], idx3s[idx], theta0[idx], ka[idx], isLinear[idx], pos, grad);
  }
}

__global__ void bendStretchEnergyKernel(const int     numAngles,
                                        const int*    idx1s,
                                        const int*    idx2s,
                                        const int*    idx3s,
                                        const double* theta0,
                                        const double* restLen1,
                                        const double* restLen2,
                                        const double* forceConst1,
                                        const double* forceConst2,
                                        const double* pos,
                                        double*       energyBuffer,
                                        const int*    energyBufferStarts,
                                        const int*    atomBatchMap,
                                        const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numAngles) {
    const double energy    = bendStretchEnergy(pos,
                                            idx1s[idx],
                                            idx2s[idx],
                                            idx3s[idx],
                                            theta0[idx],
                                            restLen1[idx],
                                            restLen2[idx],
                                            forceConst1[idx],
                                            forceConst2[idx]);
    const int    batchIdx  = atomBatchMap[idx1s[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}
__global__ void bendStretchGradKernel(const int     numAngles,
                                      const int*    idx1s,
                                      const int*    idx2s,
                                      const int*    idx3s,
                                      const double* theta0,
                                      const double* restLen1,
                                      const double* restLen2,
                                      const double* forceConst1,
                                      const double* forceConst2,
                                      const double* pos,
                                      double*       grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numAngles) {
    bendStretchGrad(pos,
                    idx1s[idx],
                    idx2s[idx],
                    idx3s[idx],
                    theta0[idx],
                    restLen1[idx],
                    restLen2[idx],
                    forceConst1[idx],
                    forceConst2[idx],
                    grad);
  }
}

__global__ void oopBendEnergyKernel(const int     numOopBends,
                                    const int*    idx1s,
                                    const int*    idx2s,
                                    const int*    idx3s,
                                    const int*    idx4s,
                                    const double* koop,
                                    const double* pos,
                                    double*       energyBuffer,
                                    const int*    energyBufferStarts,
                                    const int*    atomBatchMap,
                                    const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numOopBends) {
    // Using I, J, K, L notation

    const double energy    = oopBendEnergy(pos, idx1s[idx], idx2s[idx], idx3s[idx], idx4s[idx], koop[idx]);
    const int    batchIdx  = atomBatchMap[idx1s[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}
__global__ void oopBendGradKernel(const int     numOopBends,
                                  const int*    idx1s,
                                  const int*    idx2s,
                                  const int*    idx3s,
                                  const int*    idx4s,
                                  const double* koop,
                                  const double* pos,
                                  double*       grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numOopBends) {
    rdkit_ports::oopGrad(pos, idx1s[idx], idx2s[idx], idx3s[idx], idx4s[idx], koop[idx], grad);
  }
}

__global__ void torsionEnergyKernel(const int     numTorsions,
                                    const int*    idx1s,
                                    const int*    idx2s,
                                    const int*    idx3s,
                                    const int*    idx4s,
                                    const float*  V1s,
                                    const float*  V2s,
                                    const float*  V3s,
                                    const double* pos,
                                    double*       energyBuffer,
                                    const int*    energyBufferStarts,
                                    const int*    atomBatchMap,
                                    const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numTorsions) {
    const double energy =
      torsionEnergy(pos, idx1s[idx], idx2s[idx], idx3s[idx], idx4s[idx], V1s[idx], V2s[idx], V3s[idx]);
    const int batchIdx  = atomBatchMap[idx1s[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}
__global__ void torsionGradKernel(const int     numTorsions,
                                  const int*    idx1s,
                                  const int*    idx2s,
                                  const int*    idx3s,
                                  const int*    idx4s,
                                  const float*  V1s,
                                  const float*  V2s,
                                  const float*  V3s,
                                  const double* pos,
                                  double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTorsions) {
    rdkit_ports::torsionGrad(pos, idx1s[idx], idx2s[idx], idx3s[idx], idx4s[idx], V1s[idx], V2s[idx], V3s[idx], grad);
  }
}

__global__ void vdwEnergyKernel(const int     numVdws,
                                const int*    idxs1,
                                const int*    idx2s,
                                const double* R_ij_stars,
                                const double* wellDepths,
                                const double* pos,
                                double*       energyBuffer,
                                const int*    energyBufferStarts,
                                const int*    atomBatchMap,
                                const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numVdws) {
    const double energy = vdwEnergy(pos, idxs1[idx], idx2s[idx], R_ij_stars[idx], wellDepths[idx]);

    const int batchIdx  = atomBatchMap[idxs1[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}
__global__ void vdwGradKernel(const int     numVdws,
                              const int*    idx1,
                              const int*    idx2,
                              const double* R_ij_star,
                              const double* wellDepth,
                              const double* pos,
                              double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numVdws) {
    rdkit_ports::vDWGrad(pos, idx1[idx], idx2[idx], R_ij_star[idx], wellDepth[idx], grad);
  }
}

__global__ void eleEnergyKernel(const int      numEles,
                                const int*     idx1s,
                                const int*     idx2s,
                                const double*  chargeTerms,
                                const uint8_t* dielModels,
                                const uint8_t* is1_4s,
                                const double*  pos,
                                double*        energyBuffer,
                                const int*     energyBufferStarts,
                                const int*     atomBatchMap,
                                const int*     termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numEles) {
    const double energy = eleEnergy(pos, idx1s[idx], idx2s[idx], chargeTerms[idx], dielModels[idx], is1_4s[idx]);

    const int batchIdx  = atomBatchMap[idx1s[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}
__global__ void eleGradKernel(const int      numEles,
                              const int*     idx1s,
                              const int*     idx2s,
                              const double*  chargeTerms,
                              const uint8_t* dielModels,
                              const uint8_t* is1_4s,
                              const double*  pos,
                              double*        grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numEles) {
    eleGrad(pos, idx1s[idx], idx2s[idx], chargeTerms[idx], dielModels[idx], is1_4s[idx], grad);
  }
}

__global__ void distanceConstraintEnergyKernel(const int     numConstraints,
                                               const int*    idx1s,
                                               const int*    idx2s,
                                               const double* minLens,
                                               const double* maxLens,
                                               const double* forceConstants,
                                               const double* pos,
                                               double*       energyBuffer,
                                               const int*    energyBufferStarts,
                                               const int*    atomBatchMap,
                                               const int*    termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    const double energy =
      distanceConstraintEnergy(pos, idx1s[idx], idx2s[idx], minLens[idx], maxLens[idx], forceConstants[idx]);
    const int batchIdx  = atomBatchMap[idx1s[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void distanceConstraintGradKernel(const int     numConstraints,
                                             const int*    idx1s,
                                             const int*    idx2s,
                                             const double* minLens,
                                             const double* maxLens,
                                             const double* forceConstants,
                                             const double* pos,
                                             double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    distanceConstraintGrad(pos, idx1s[idx], idx2s[idx], minLens[idx], maxLens[idx], forceConstants[idx], grad);
  }
}

__global__ void positionConstraintEnergyKernel(const int     numConstraints,
                                               const int*    idxs,
                                               const double* refXs,
                                               const double* refYs,
                                               const double* refZs,
                                               const double* maxDispls,
                                               const double* forceConstants,
                                               const double* pos,
                                               double*       energyBuffer,
                                               const int*    energyBufferStarts,
                                               const int*    atomBatchMap,
                                               const int*    termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    const double energy =
      positionConstraintEnergy(pos, idxs[idx], refXs[idx], refYs[idx], refZs[idx], maxDispls[idx], forceConstants[idx]);
    const int batchIdx  = atomBatchMap[idxs[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void positionConstraintGradKernel(const int     numConstraints,
                                             const int*    idxs,
                                             const double* refXs,
                                             const double* refYs,
                                             const double* refZs,
                                             const double* maxDispls,
                                             const double* forceConstants,
                                             const double* pos,
                                             double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    positionConstraintGrad(pos,
                           idxs[idx],
                           refXs[idx],
                           refYs[idx],
                           refZs[idx],
                           maxDispls[idx],
                           forceConstants[idx],
                           grad);
  }
}

__global__ void angleConstraintEnergyKernel(const int     numConstraints,
                                            const int*    idx1s,
                                            const int*    idx2s,
                                            const int*    idx3s,
                                            const double* minAngleDegs,
                                            const double* maxAngleDegs,
                                            const double* forceConstants,
                                            const double* pos,
                                            double*       energyBuffer,
                                            const int*    energyBufferStarts,
                                            const int*    atomBatchMap,
                                            const int*    termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    const double energy    = angleConstraintEnergy(pos,
                                                idx1s[idx],
                                                idx2s[idx],
                                                idx3s[idx],
                                                minAngleDegs[idx],
                                                maxAngleDegs[idx],
                                                forceConstants[idx]);
    const int    batchIdx  = atomBatchMap[idx1s[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void angleConstraintGradKernel(const int     numConstraints,
                                          const int*    idx1s,
                                          const int*    idx2s,
                                          const int*    idx3s,
                                          const double* minAngleDegs,
                                          const double* maxAngleDegs,
                                          const double* forceConstants,
                                          const double* pos,
                                          double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    angleConstraintGrad(pos,
                        idx1s[idx],
                        idx2s[idx],
                        idx3s[idx],
                        minAngleDegs[idx],
                        maxAngleDegs[idx],
                        forceConstants[idx],
                        grad);
  }
}

__global__ void torsionConstraintEnergyKernel(const int     numConstraints,
                                              const int*    idx1s,
                                              const int*    idx2s,
                                              const int*    idx3s,
                                              const int*    idx4s,
                                              const double* minDihedralDegs,
                                              const double* maxDihedralDegs,
                                              const double* forceConstants,
                                              const double* pos,
                                              double*       energyBuffer,
                                              const int*    energyBufferStarts,
                                              const int*    atomBatchMap,
                                              const int*    termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    const double energy    = torsionConstraintEnergy(pos,
                                                  idx1s[idx],
                                                  idx2s[idx],
                                                  idx3s[idx],
                                                  idx4s[idx],
                                                  minDihedralDegs[idx],
                                                  maxDihedralDegs[idx],
                                                  forceConstants[idx]);
    const int    batchIdx  = atomBatchMap[idx1s[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void torsionConstraintGradKernel(const int     numConstraints,
                                            const int*    idx1s,
                                            const int*    idx2s,
                                            const int*    idx3s,
                                            const int*    idx4s,
                                            const double* minDihedralDegs,
                                            const double* maxDihedralDegs,
                                            const double* forceConstants,
                                            const double* pos,
                                            double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numConstraints) {
    torsionConstraintGrad(pos,
                          idx1s[idx],
                          idx2s[idx],
                          idx3s[idx],
                          idx4s[idx],
                          minDihedralDegs[idx],
                          maxDihedralDegs[idx],
                          forceConstants[idx],
                          grad);
  }
}

namespace nvMolKit {
namespace MMFF {

cudaError_t launchBondStretchEnergyKernel(const int     numBonds,
                                          const int*    idx1,
                                          const int*    idx2,
                                          const double* r0,
                                          const double* kb,
                                          const double* pos,
                                          double*       energyBuffer,
                                          const int*    energyBufferStarts,
                                          const int*    atomBatchMap,
                                          const int*    termBatchStarts,
                                          cudaStream_t  stream) {
  assert(numBonds > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numBonds + blockSize - 1) / blockSize;
  bondStretchEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numBonds,
                                                               idx1,
                                                               idx2,
                                                               r0,
                                                               kb,
                                                               pos,
                                                               energyBuffer,
                                                               energyBufferStarts,
                                                               atomBatchMap,
                                                               termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchBondStretchGradientKernel(const int     numBonds,
                                            const int*    idx1,
                                            const int*    idx2,
                                            const double* r0,
                                            const double* kb,
                                            const double* pos,
                                            double*       grad,
                                            cudaStream_t  stream) {
  assert(numBonds > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numBonds + blockSize - 1) / blockSize;
  bondStretchGradKernel<<<numBlocks, blockSize, 0, stream>>>(numBonds, idx1, idx2, r0, kb, pos, grad);

  return cudaGetLastError();
}

cudaError_t launchAngleBendEnergyKernel(const int      numAngles,
                                        const int*     idx1,
                                        const int*     idx2,
                                        const int*     idx3,
                                        const double*  theta0,
                                        const double*  ka,
                                        const uint8_t* isLinear,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomBatchMap,
                                        const int*     termBatchStarts,
                                        cudaStream_t   stream) {
  assert(numAngles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  angleBendEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                             idx1,
                                                             idx2,
                                                             idx3,
                                                             theta0,
                                                             ka,
                                                             isLinear,
                                                             pos,
                                                             energyBuffer,
                                                             energyBufferStarts,
                                                             atomBatchMap,
                                                             termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchAngleBendGradientKernel(const int      numAngles,
                                          const int*     idx1,
                                          const int*     idx2,
                                          const int*     idx3,
                                          const double*  theta0,
                                          const double*  ka,
                                          const uint8_t* isLinear,
                                          const double*  pos,
                                          double*        grad,
                                          cudaStream_t   stream) {
  assert(numAngles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  angleBendGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                               idx1,
                                                               idx2,
                                                               idx3,
                                                               theta0,
                                                               ka,
                                                               isLinear,
                                                               pos,
                                                               grad);

  return cudaGetLastError();
}

cudaError_t launchBendStretchEnergyKernel(const int     numAngles,
                                          const int*    idx1,
                                          const int*    idx2,
                                          const int*    idx3,
                                          const double* theta0,
                                          const double* restLen1,
                                          const double* restLen2,
                                          const double* forceConst1,
                                          const double* forceConst2,
                                          const double* pos,
                                          double*       energyBuffer,
                                          const int*    energyBufferStarts,
                                          const int*    atomBatchMap,
                                          const int*    termBatchStarts,
                                          cudaStream_t  stream) {
  assert(numAngles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  bendStretchEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                               idx1,
                                                               idx2,
                                                               idx3,
                                                               theta0,
                                                               restLen1,
                                                               restLen2,
                                                               forceConst1,
                                                               forceConst2,
                                                               pos,
                                                               energyBuffer,
                                                               energyBufferStarts,
                                                               atomBatchMap,
                                                               termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchBendStretchGradientKernel(const int     numAngles,
                                            const int*    idx1,
                                            const int*    idx2,
                                            const int*    idx3,
                                            const double* theta0,
                                            const double* restLen1,
                                            const double* restLen2,
                                            const double* forceConst1,
                                            const double* forceConst2,
                                            const double* pos,
                                            double*       grad,
                                            cudaStream_t  stream) {
  assert(numAngles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  bendStretchGradKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                             idx1,
                                                             idx2,
                                                             idx3,
                                                             theta0,
                                                             restLen1,
                                                             restLen2,
                                                             forceConst1,
                                                             forceConst2,
                                                             pos,
                                                             grad);
  return cudaGetLastError();
}

cudaError_t launchOopBendEnergyKernel(const int     numOopBends,
                                      const int*    idx1,
                                      const int*    idx2,
                                      const int*    idx3,
                                      const int*    idx4,
                                      const double* koop,
                                      const double* pos,
                                      double*       energyBuffer,
                                      const int*    energyBufferStarts,
                                      const int*    atomBatchMap,
                                      const int*    termBatchStarts,
                                      cudaStream_t  stream) {
  assert(numOopBends > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numOopBends + blockSize - 1) / blockSize;
  oopBendEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numOopBends,
                                                           idx1,
                                                           idx2,
                                                           idx3,
                                                           idx4,
                                                           koop,
                                                           pos,
                                                           energyBuffer,
                                                           energyBufferStarts,
                                                           atomBatchMap,
                                                           termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchOopBendGradientKernel(const int     numOopBends,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const int*    idx3,
                                        const int*    idx4,
                                        const double* koop,
                                        const double* pos,
                                        double*       grad,
                                        cudaStream_t  stream) {
  assert(numOopBends > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numOopBends + blockSize - 1) / blockSize;
  oopBendGradKernel<<<numBlocks, blockSize, 0, stream>>>(numOopBends, idx1, idx2, idx3, idx4, koop, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchTorsionEnergyKernel(const int     numTorsions,
                                      const int*    idx1,
                                      const int*    idx2,
                                      const int*    idx3,
                                      const int*    idx4,
                                      const float*  V1,
                                      const float*  V2,
                                      const float*  V3,
                                      const double* pos,
                                      double*       energyBuffer,
                                      const int*    energyBufferStarts,
                                      const int*    atomBatchMap,
                                      const int*    termBatchStarts,
                                      cudaStream_t  stream) {
  assert(numTorsions > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsions + blockSize - 1) / blockSize;
  torsionEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsions,
                                                           idx1,
                                                           idx2,
                                                           idx3,
                                                           idx4,
                                                           V1,
                                                           V2,
                                                           V3,
                                                           pos,
                                                           energyBuffer,
                                                           energyBufferStarts,
                                                           atomBatchMap,
                                                           termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchTorsionGradientKernel(const int     numTorsions,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const int*    idx3,
                                        const int*    idx4,
                                        const float*  V1,
                                        const float*  V2,
                                        const float*  V3,
                                        const double* pos,
                                        double*       grad,
                                        cudaStream_t  stream) {
  assert(numTorsions > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsions + blockSize - 1) / blockSize;
  torsionGradKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsions, idx1, idx2, idx3, idx4, V1, V2, V3, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchVdwEnergyKernel(const int     numVdws,
                                  const int*    idx1,
                                  const int*    idx2,
                                  const double* R_ij_star,
                                  const double* wellDepth,
                                  const double* pos,
                                  double*       energyBuffer,
                                  const int*    energyBufferStarts,
                                  const int*    atomBatchMap,
                                  const int*    termBatchStarts,
                                  cudaStream_t  stream) {
  assert(numVdws > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numVdws + blockSize - 1) / blockSize;
  vdwEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numVdws,
                                                       idx1,
                                                       idx2,
                                                       R_ij_star,
                                                       wellDepth,
                                                       pos,
                                                       energyBuffer,
                                                       energyBufferStarts,
                                                       atomBatchMap,
                                                       termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchVdwGradientKernel(const int     numVdws,
                                    const int*    idx1,
                                    const int*    idx2,
                                    const double* R_ij_star,
                                    const double* wellDepth,
                                    const double* pos,
                                    double*       grad,
                                    cudaStream_t  stream) {
  assert(numVdws > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numVdws + blockSize - 1) / blockSize;
  vdwGradKernel<<<numBlocks, blockSize, 0, stream>>>(numVdws, idx1, idx2, R_ij_star, wellDepth, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchEleEnergyKernel(const int      numEles,
                                  const int*     idx1,
                                  const int*     idx2,
                                  const double*  chargeTerm,
                                  const uint8_t* dielModel,
                                  const uint8_t* is1_4,
                                  const double*  pos,
                                  double*        energyBuffer,
                                  const int*     energyBufferStarts,
                                  const int*     atomBatchMap,
                                  const int*     termBatchStarts,
                                  cudaStream_t   stream) {
  assert(numEles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numEles + blockSize - 1) / blockSize;
  eleEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numEles,
                                                       idx1,
                                                       idx2,
                                                       chargeTerm,
                                                       dielModel,
                                                       is1_4,
                                                       pos,
                                                       energyBuffer,
                                                       energyBufferStarts,
                                                       atomBatchMap,
                                                       termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchEleGradientKernel(const int      numEles,
                                    const int*     idx1,
                                    const int*     idx2,
                                    const double*  chargeTerm,
                                    const uint8_t* dielModel,
                                    const uint8_t* is1_4,
                                    const double*  pos,
                                    double*        grad,
                                    cudaStream_t   stream) {
  assert(numEles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numEles + blockSize - 1) / blockSize;
  eleGradKernel<<<numBlocks, blockSize, 0, stream>>>(numEles, idx1, idx2, chargeTerm, dielModel, is1_4, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchDistanceConstraintEnergyKernel(const int     numConstraints,
                                                 const int*    idx1,
                                                 const int*    idx2,
                                                 const double* minLen,
                                                 const double* maxLen,
                                                 const double* forceConstant,
                                                 const double* pos,
                                                 double*       energyBuffer,
                                                 const int*    energyBufferStarts,
                                                 const int*    atomBatchMap,
                                                 const int*    termBatchStarts,
                                                 cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  distanceConstraintEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                      idx1,
                                                                      idx2,
                                                                      minLen,
                                                                      maxLen,
                                                                      forceConstant,
                                                                      pos,
                                                                      energyBuffer,
                                                                      energyBufferStarts,
                                                                      atomBatchMap,
                                                                      termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchDistanceConstraintGradientKernel(const int     numConstraints,
                                                   const int*    idx1,
                                                   const int*    idx2,
                                                   const double* minLen,
                                                   const double* maxLen,
                                                   const double* forceConstant,
                                                   const double* pos,
                                                   double*       grad,
                                                   cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  distanceConstraintGradKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                    idx1,
                                                                    idx2,
                                                                    minLen,
                                                                    maxLen,
                                                                    forceConstant,
                                                                    pos,
                                                                    grad);
  return cudaGetLastError();
}

cudaError_t launchPositionConstraintEnergyKernel(const int     numConstraints,
                                                 const int*    idx,
                                                 const double* refX,
                                                 const double* refY,
                                                 const double* refZ,
                                                 const double* maxDispl,
                                                 const double* forceConstant,
                                                 const double* pos,
                                                 double*       energyBuffer,
                                                 const int*    energyBufferStarts,
                                                 const int*    atomBatchMap,
                                                 const int*    termBatchStarts,
                                                 cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  positionConstraintEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                      idx,
                                                                      refX,
                                                                      refY,
                                                                      refZ,
                                                                      maxDispl,
                                                                      forceConstant,
                                                                      pos,
                                                                      energyBuffer,
                                                                      energyBufferStarts,
                                                                      atomBatchMap,
                                                                      termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchPositionConstraintGradientKernel(const int     numConstraints,
                                                   const int*    idx,
                                                   const double* refX,
                                                   const double* refY,
                                                   const double* refZ,
                                                   const double* maxDispl,
                                                   const double* forceConstant,
                                                   const double* pos,
                                                   double*       grad,
                                                   cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  positionConstraintGradKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                    idx,
                                                                    refX,
                                                                    refY,
                                                                    refZ,
                                                                    maxDispl,
                                                                    forceConstant,
                                                                    pos,
                                                                    grad);
  return cudaGetLastError();
}

cudaError_t launchAngleConstraintEnergyKernel(const int     numConstraints,
                                              const int*    idx1,
                                              const int*    idx2,
                                              const int*    idx3,
                                              const double* minAngleDeg,
                                              const double* maxAngleDeg,
                                              const double* forceConstant,
                                              const double* pos,
                                              double*       energyBuffer,
                                              const int*    energyBufferStarts,
                                              const int*    atomBatchMap,
                                              const int*    termBatchStarts,
                                              cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  angleConstraintEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                   idx1,
                                                                   idx2,
                                                                   idx3,
                                                                   minAngleDeg,
                                                                   maxAngleDeg,
                                                                   forceConstant,
                                                                   pos,
                                                                   energyBuffer,
                                                                   energyBufferStarts,
                                                                   atomBatchMap,
                                                                   termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchAngleConstraintGradientKernel(const int     numConstraints,
                                                const int*    idx1,
                                                const int*    idx2,
                                                const int*    idx3,
                                                const double* minAngleDeg,
                                                const double* maxAngleDeg,
                                                const double* forceConstant,
                                                const double* pos,
                                                double*       grad,
                                                cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  angleConstraintGradKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                 idx1,
                                                                 idx2,
                                                                 idx3,
                                                                 minAngleDeg,
                                                                 maxAngleDeg,
                                                                 forceConstant,
                                                                 pos,
                                                                 grad);
  return cudaGetLastError();
}

cudaError_t launchTorsionConstraintEnergyKernel(const int     numConstraints,
                                                const int*    idx1,
                                                const int*    idx2,
                                                const int*    idx3,
                                                const int*    idx4,
                                                const double* minDihedralDeg,
                                                const double* maxDihedralDeg,
                                                const double* forceConstant,
                                                const double* pos,
                                                double*       energyBuffer,
                                                const int*    energyBufferStarts,
                                                const int*    atomBatchMap,
                                                const int*    termBatchStarts,
                                                cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  torsionConstraintEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                     idx1,
                                                                     idx2,
                                                                     idx3,
                                                                     idx4,
                                                                     minDihedralDeg,
                                                                     maxDihedralDeg,
                                                                     forceConstant,
                                                                     pos,
                                                                     energyBuffer,
                                                                     energyBufferStarts,
                                                                     atomBatchMap,
                                                                     termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchTorsionConstraintGradientKernel(const int     numConstraints,
                                                  const int*    idx1,
                                                  const int*    idx2,
                                                  const int*    idx3,
                                                  const int*    idx4,
                                                  const double* minDihedralDeg,
                                                  const double* maxDihedralDeg,
                                                  const double* forceConstant,
                                                  const double* pos,
                                                  double*       grad,
                                                  cudaStream_t  stream) {
  assert(numConstraints > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numConstraints + blockSize - 1) / blockSize;
  torsionConstraintGradKernel<<<numBlocks, blockSize, 0, stream>>>(numConstraints,
                                                                   idx1,
                                                                   idx2,
                                                                   idx3,
                                                                   idx4,
                                                                   minDihedralDeg,
                                                                   maxDihedralDeg,
                                                                   forceConstant,
                                                                   pos,
                                                                   grad);
  return cudaGetLastError();
}

cudaError_t launchReduceEnergiesKernel(const int      numBlocks,
                                       const double*  energyBuffer,
                                       const int*     energyBufferBlockIdxToBatchIdx,
                                       double*        outs,
                                       const uint8_t* activeThisStage,
                                       cudaStream_t   stream) {
  reduceEnergiesKernel<<<numBlocks, nvMolKit::FFKernelUtils::blockSizeEnergyReduction, 0, stream>>>(
    energyBuffer,
    energyBufferBlockIdxToBatchIdx,
    outs,
    activeThisStage);
  return cudaGetLastError();
}

constexpr int blockSizePerMol = 128;

template <bool HasConstraints>
__global__ void combinedEnergiesKernel(const EnergyForceContribsDevicePtr* terms,
                                       const BatchedIndicesDevicePtr*      systemIndices,
                                       const double*                       coords,
                                       double*                             energies) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;

  using BlockReduce = cub::BlockReduce<double, blockSizePerMol>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  const int     atomStart = systemIndices->atomStarts[molIdx];
  const double* molCoords = coords + atomStart * 3;
  const double  threadEnergy =
    molEnergy<blockSizePerMol, HasConstraints>(*terms, *systemIndices, molCoords, molIdx, tid);
  const double blockEnergy = BlockReduce(tempStorage).Sum(threadEnergy);

  if (tid == 0) {
    energies[molIdx] = blockEnergy;
  }
}

template <bool HasConstraints>
__global__ void combinedGradKernel(const EnergyForceContribsDevicePtr* terms,
                                   const BatchedIndicesDevicePtr*      systemIndices,
                                   const double*                       coords,
                                   double*                             grad) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;

  const int atomStart = systemIndices->atomStarts[molIdx];
  const int atomEnd   = systemIndices->atomStarts[molIdx + 1];
  const int numAtoms  = atomEnd - atomStart;

  constexpr int     maxAtomSize = 256;
  __shared__ double accumGrad[maxAtomSize * 3];

  const bool useSharedMem = numAtoms <= maxAtomSize;
  double*    molGradBase  = useSharedMem ? accumGrad : grad + atomStart * 3;

  for (int i = tid; i < numAtoms * 3; i += blockSizePerMol) {
    molGradBase[i] = 0.0;
  }
  __syncthreads();

  const double* molCoords = coords + atomStart * 3;
  molGrad<blockSizePerMol, HasConstraints>(*terms, *systemIndices, molCoords, molGradBase, molIdx, tid);
  __syncthreads();

  if (useSharedMem) {
    double* globalGrad = grad + (atomStart * 3);
    for (int i = tid; i < numAtoms * 3; i += blockSizePerMol) {
      globalGrad[i] = molGradBase[i];
    }
  }
}

cudaError_t launchBlockPerMolEnergyKernel(int                                 numMols,
                                          const EnergyForceContribsDevicePtr& terms,
                                          const BatchedIndicesDevicePtr&      sytemIndices,
                                          const double*                       coords,
                                          double*                             energies,
                                          bool                                hasConstraints,
                                          cudaStream_t                        stream) {
  const AsyncDevicePtr<EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndicesDevicePtr>      devSysIdx(sytemIndices, stream);
  if (hasConstraints) {
    combinedEnergiesKernel<true>
      <<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(), devSysIdx.data(), coords, energies);
  } else {
    combinedEnergiesKernel<false>
      <<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(), devSysIdx.data(), coords, energies);
  }
  return cudaGetLastError();
}

cudaError_t launchBlockPerMolGradKernel(int                                 numMols,
                                        const EnergyForceContribsDevicePtr& terms,
                                        const BatchedIndicesDevicePtr&      sytemIndices,
                                        const double*                       coords,
                                        double*                             grad,
                                        bool                                hasConstraints,
                                        cudaStream_t                        stream) {
  const AsyncDevicePtr<EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndicesDevicePtr>      devSysIdx(sytemIndices, stream);
  if (hasConstraints) {
    combinedGradKernel<true><<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(), devSysIdx.data(), coords, grad);
  } else {
    combinedGradKernel<false><<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(), devSysIdx.data(), coords, grad);
  }
  return cudaGetLastError();
}
}  // namespace MMFF
}  // namespace nvMolKit
