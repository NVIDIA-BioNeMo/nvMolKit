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

#include <cub/cub.cuh>

#include "kernel_utils.cuh"
#include "uff_kernels.h"
#include "uff_kernels_device.cuh"

using namespace nvMolKit::FFKernelUtils;

namespace {

__global__ void bondStretchEnergyKernel(const int     numBonds,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const double* restLen,
                                        const double* forceConstant,
                                        const double* pos,
                                        double*       energyBuffer,
                                        const int*    energyBufferStarts,
                                        const int*    atomBatchMap,
                                        const int*    termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numBonds) {
    const double energy    = uffBondStretchEnergy(pos, idx1[idx], idx2[idx], restLen[idx], forceConstant[idx]);
    const int    batchIdx  = atomBatchMap[idx1[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void bondStretchGradKernel(const int     numBonds,
                                      const int*    idx1,
                                      const int*    idx2,
                                      const double* restLen,
                                      const double* forceConstant,
                                      const double* pos,
                                      double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numBonds) {
    uffBondStretchGrad(pos, idx1[idx], idx2[idx], restLen[idx], forceConstant[idx], grad);
  }
}

__global__ void angleBendEnergyKernel(const int      numAngles,
                                      const int*     idx1,
                                      const int*     idx2,
                                      const int*     idx3,
                                      const double*  theta0,
                                      const double*  forceConstant,
                                      const uint8_t* order,
                                      const double*  C0,
                                      const double*  C1,
                                      const double*  C2,
                                      const double*  pos,
                                      double*        energyBuffer,
                                      const int*     energyBufferStarts,
                                      const int*     atomBatchMap,
                                      const int*     termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numAngles) {
    const double energy    = uffAngleBendEnergy(pos,
                                             idx1[idx],
                                             idx2[idx],
                                             idx3[idx],
                                             theta0[idx],
                                             forceConstant[idx],
                                             order[idx],
                                             C0[idx],
                                             C1[idx],
                                             C2[idx]);
    const int    batchIdx  = atomBatchMap[idx1[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void angleBendGradKernel(const int      numAngles,
                                    const int*     idx1,
                                    const int*     idx2,
                                    const int*     idx3,
                                    const double*  theta0,
                                    const double*  forceConstant,
                                    const uint8_t* order,
                                    const double*  C0,
                                    const double*  C1,
                                    const double*  C2,
                                    const double*  pos,
                                    double*        grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numAngles) {
    uffAngleBendGrad(pos,
                     idx1[idx],
                     idx2[idx],
                     idx3[idx],
                     theta0[idx],
                     forceConstant[idx],
                     order[idx],
                     C0[idx],
                     C1[idx],
                     C2[idx],
                     grad);
  }
}

__global__ void torsionEnergyKernel(const int      numTorsions,
                                    const int*     idx1,
                                    const int*     idx2,
                                    const int*     idx3,
                                    const int*     idx4,
                                    const double*  forceConstant,
                                    const uint8_t* order,
                                    const double*  cosTerm,
                                    const double*  pos,
                                    double*        energyBuffer,
                                    const int*     energyBufferStarts,
                                    const int*     atomBatchMap,
                                    const int*     termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTorsions) {
    const double energy =
      uffTorsionEnergy(pos, idx1[idx], idx2[idx], idx3[idx], idx4[idx], forceConstant[idx], order[idx], cosTerm[idx]);
    const int batchIdx  = atomBatchMap[idx1[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void torsionGradKernel(const int      numTorsions,
                                  const int*     idx1,
                                  const int*     idx2,
                                  const int*     idx3,
                                  const int*     idx4,
                                  const double*  forceConstant,
                                  const uint8_t* order,
                                  const double*  cosTerm,
                                  const double*  pos,
                                  double*        grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTorsions) {
    uffTorsionGrad(pos, idx1[idx], idx2[idx], idx3[idx], idx4[idx], forceConstant[idx], order[idx], cosTerm[idx], grad);
  }
}

__global__ void inversionEnergyKernel(const int     numInversions,
                                      const int*    idx1,
                                      const int*    idx2,
                                      const int*    idx3,
                                      const int*    idx4,
                                      const double* forceConstant,
                                      const double* C0,
                                      const double* C1,
                                      const double* C2,
                                      const double* pos,
                                      double*       energyBuffer,
                                      const int*    energyBufferStarts,
                                      const int*    atomBatchMap,
                                      const int*    termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numInversions) {
    const double energy    = uffInversionEnergy(pos,
                                             idx1[idx],
                                             idx2[idx],
                                             idx3[idx],
                                             idx4[idx],
                                             forceConstant[idx],
                                             C0[idx],
                                             C1[idx],
                                             C2[idx]);
    const int    batchIdx  = atomBatchMap[idx1[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void inversionGradKernel(const int     numInversions,
                                    const int*    idx1,
                                    const int*    idx2,
                                    const int*    idx3,
                                    const int*    idx4,
                                    const double* forceConstant,
                                    const double* C1,
                                    const double* C2,
                                    const double* pos,
                                    double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numInversions) {
    uffInversionGrad(pos, idx1[idx], idx2[idx], idx3[idx], idx4[idx], forceConstant[idx], C1[idx], C2[idx], grad);
  }
}

__global__ void vdwEnergyKernel(const int     numVdws,
                                const int*    idx1,
                                const int*    idx2,
                                const double* x_ij,
                                const double* wellDepth,
                                const double* threshold,
                                const double* pos,
                                double*       energyBuffer,
                                const int*    energyBufferStarts,
                                const int*    atomBatchMap,
                                const int*    termBatchStarts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numVdws) {
    const double energy    = uffVdwEnergy(pos, idx1[idx], idx2[idx], x_ij[idx], wellDepth[idx], threshold[idx]);
    const int    batchIdx  = atomBatchMap[idx1[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}

__global__ void vdwGradKernel(const int     numVdws,
                              const int*    idx1,
                              const int*    idx2,
                              const double* x_ij,
                              const double* wellDepth,
                              const double* threshold,
                              const double* pos,
                              double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numVdws) {
    uffVdwGrad(pos, idx1[idx], idx2[idx], x_ij[idx], wellDepth[idx], threshold[idx], grad);
  }
}

constexpr int blockSizePerMol = 128;

template <bool HasConstraints>
__global__ void combinedEnergiesKernel(const nvMolKit::UFF::EnergyForceContribsDevicePtr* terms,
                                       const nvMolKit::UFF::BatchedIndicesDevicePtr*      systemIndices,
                                       const double*                                      coords,
                                       double*                                            energies) {
  const int molIdx = blockIdx.x;
  const int tid    = threadIdx.x;

  const int     atomStart = systemIndices->atomStarts[molIdx];
  const double* molCoords = coords + atomStart * 3;
  const double  threadEnergy =
    nvMolKit::UFF::molEnergy<blockSizePerMol, HasConstraints>(*terms, *systemIndices, molCoords, molIdx, tid);

  using BlockReduce = cub::BlockReduce<double, blockSizePerMol>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  const double                                 blockEnergy = BlockReduce(tempStorage).Sum(threadEnergy);

  if (tid == 0) {
    energies[molIdx] = blockEnergy;
  }
}

template <bool HasConstraints>
__global__ void combinedGradKernel(const nvMolKit::UFF::EnergyForceContribsDevicePtr* terms,
                                   const nvMolKit::UFF::BatchedIndicesDevicePtr*      systemIndices,
                                   const double*                                      coords,
                                   double*                                            grad) {
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
  nvMolKit::UFF::molGrad<blockSizePerMol, HasConstraints>(*terms, *systemIndices, molCoords, molGradBase, molIdx, tid);
  __syncthreads();

  if (useSharedMem) {
    double* globalGrad = grad + atomStart * 3;
    for (int i = tid; i < numAtoms * 3; i += blockSizePerMol) {
      globalGrad[i] = molGradBase[i];
    }
  }
}

}  // namespace

namespace nvMolKit {
namespace UFF {

cudaError_t launchBondStretchEnergyKernel(int           numBonds,
                                          const int*    idx1,
                                          const int*    idx2,
                                          const double* restLen,
                                          const double* forceConstant,
                                          const double* pos,
                                          double*       energyBuffer,
                                          const int*    energyBufferStarts,
                                          const int*    atomBatchMap,
                                          const int*    termBatchStarts,
                                          cudaStream_t  stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numBonds + blockSize - 1) / blockSize;
  bondStretchEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numBonds,
                                                               idx1,
                                                               idx2,
                                                               restLen,
                                                               forceConstant,
                                                               pos,
                                                               energyBuffer,
                                                               energyBufferStarts,
                                                               atomBatchMap,
                                                               termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchBondStretchGradientKernel(int           numBonds,
                                            const int*    idx1,
                                            const int*    idx2,
                                            const double* restLen,
                                            const double* forceConstant,
                                            const double* pos,
                                            double*       grad,
                                            cudaStream_t  stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numBonds + blockSize - 1) / blockSize;
  bondStretchGradKernel<<<numBlocks, blockSize, 0, stream>>>(numBonds, idx1, idx2, restLen, forceConstant, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchAngleBendEnergyKernel(int            numAngles,
                                        const int*     idx1,
                                        const int*     idx2,
                                        const int*     idx3,
                                        const double*  theta0,
                                        const double*  forceConstant,
                                        const uint8_t* order,
                                        const double*  C0,
                                        const double*  C1,
                                        const double*  C2,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomBatchMap,
                                        const int*     termBatchStarts,
                                        cudaStream_t   stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  angleBendEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                             idx1,
                                                             idx2,
                                                             idx3,
                                                             theta0,
                                                             forceConstant,
                                                             order,
                                                             C0,
                                                             C1,
                                                             C2,
                                                             pos,
                                                             energyBuffer,
                                                             energyBufferStarts,
                                                             atomBatchMap,
                                                             termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchAngleBendGradientKernel(int            numAngles,
                                          const int*     idx1,
                                          const int*     idx2,
                                          const int*     idx3,
                                          const double*  theta0,
                                          const double*  forceConstant,
                                          const uint8_t* order,
                                          const double*  C0,
                                          const double*  C1,
                                          const double*  C2,
                                          const double*  pos,
                                          double*        grad,
                                          cudaStream_t   stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  angleBendGradKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                           idx1,
                                                           idx2,
                                                           idx3,
                                                           theta0,
                                                           forceConstant,
                                                           order,
                                                           C0,
                                                           C1,
                                                           C2,
                                                           pos,
                                                           grad);
  return cudaGetLastError();
}

cudaError_t launchTorsionEnergyKernel(int            numTorsions,
                                      const int*     idx1,
                                      const int*     idx2,
                                      const int*     idx3,
                                      const int*     idx4,
                                      const double*  forceConstant,
                                      const uint8_t* order,
                                      const double*  cosTerm,
                                      const double*  pos,
                                      double*        energyBuffer,
                                      const int*     energyBufferStarts,
                                      const int*     atomBatchMap,
                                      const int*     termBatchStarts,
                                      cudaStream_t   stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsions + blockSize - 1) / blockSize;
  torsionEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsions,
                                                           idx1,
                                                           idx2,
                                                           idx3,
                                                           idx4,
                                                           forceConstant,
                                                           order,
                                                           cosTerm,
                                                           pos,
                                                           energyBuffer,
                                                           energyBufferStarts,
                                                           atomBatchMap,
                                                           termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchTorsionGradientKernel(int            numTorsions,
                                        const int*     idx1,
                                        const int*     idx2,
                                        const int*     idx3,
                                        const int*     idx4,
                                        const double*  forceConstant,
                                        const uint8_t* order,
                                        const double*  cosTerm,
                                        const double*  pos,
                                        double*        grad,
                                        cudaStream_t   stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsions + blockSize - 1) / blockSize;
  torsionGradKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsions,
                                                         idx1,
                                                         idx2,
                                                         idx3,
                                                         idx4,
                                                         forceConstant,
                                                         order,
                                                         cosTerm,
                                                         pos,
                                                         grad);
  return cudaGetLastError();
}

cudaError_t launchInversionEnergyKernel(int           numInversions,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const int*    idx3,
                                        const int*    idx4,
                                        const double* forceConstant,
                                        const double* C0,
                                        const double* C1,
                                        const double* C2,
                                        const double* pos,
                                        double*       energyBuffer,
                                        const int*    energyBufferStarts,
                                        const int*    atomBatchMap,
                                        const int*    termBatchStarts,
                                        cudaStream_t  stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numInversions + blockSize - 1) / blockSize;
  inversionEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numInversions,
                                                             idx1,
                                                             idx2,
                                                             idx3,
                                                             idx4,
                                                             forceConstant,
                                                             C0,
                                                             C1,
                                                             C2,
                                                             pos,
                                                             energyBuffer,
                                                             energyBufferStarts,
                                                             atomBatchMap,
                                                             termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchInversionGradientKernel(int           numInversions,
                                          const int*    idx1,
                                          const int*    idx2,
                                          const int*    idx3,
                                          const int*    idx4,
                                          const double* forceConstant,
                                          const double* C0,
                                          const double* C1,
                                          const double* C2,
                                          const double* pos,
                                          double*       grad,
                                          cudaStream_t  stream) {
  (void)C0;
  constexpr int blockSize = 256;
  const int     numBlocks = (numInversions + blockSize - 1) / blockSize;
  inversionGradKernel<<<numBlocks, blockSize, 0, stream>>>(numInversions,
                                                           idx1,
                                                           idx2,
                                                           idx3,
                                                           idx4,
                                                           forceConstant,
                                                           C1,
                                                           C2,
                                                           pos,
                                                           grad);
  return cudaGetLastError();
}

cudaError_t launchVdwEnergyKernel(int           numVdws,
                                  const int*    idx1,
                                  const int*    idx2,
                                  const double* x_ij,
                                  const double* wellDepth,
                                  const double* threshold,
                                  const double* pos,
                                  double*       energyBuffer,
                                  const int*    energyBufferStarts,
                                  const int*    atomBatchMap,
                                  const int*    termBatchStarts,
                                  cudaStream_t  stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numVdws + blockSize - 1) / blockSize;
  vdwEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numVdws,
                                                       idx1,
                                                       idx2,
                                                       x_ij,
                                                       wellDepth,
                                                       threshold,
                                                       pos,
                                                       energyBuffer,
                                                       energyBufferStarts,
                                                       atomBatchMap,
                                                       termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchVdwGradientKernel(int           numVdws,
                                    const int*    idx1,
                                    const int*    idx2,
                                    const double* x_ij,
                                    const double* wellDepth,
                                    const double* threshold,
                                    const double* pos,
                                    double*       grad,
                                    cudaStream_t  stream) {
  constexpr int blockSize = 256;
  const int     numBlocks = (numVdws + blockSize - 1) / blockSize;
  vdwGradKernel<<<numBlocks, blockSize, 0, stream>>>(numVdws, idx1, idx2, x_ij, wellDepth, threshold, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchBlockPerMolEnergyKernel(int                                 numMols,
                                          const EnergyForceContribsDevicePtr& terms,
                                          const BatchedIndicesDevicePtr&      systemIndices,
                                          const double*                       coords,
                                          double*                             energies,
                                          bool                                hasConstraints,
                                          cudaStream_t                        stream) {
  const AsyncDevicePtr<EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndicesDevicePtr>      devSysIdx(systemIndices, stream);
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
                                        const BatchedIndicesDevicePtr&      systemIndices,
                                        const double*                       coords,
                                        double*                             grad,
                                        bool                                hasConstraints,
                                        cudaStream_t                        stream) {
  const AsyncDevicePtr<EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<BatchedIndicesDevicePtr>      devSysIdx(systemIndices, stream);
  if (hasConstraints) {
    combinedGradKernel<true><<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(), devSysIdx.data(), coords, grad);
  } else {
    combinedGradKernel<false><<<numMols, blockSizePerMol, 0, stream>>>(devTerms.data(), devSysIdx.data(), coords, grad);
  }
  return cudaGetLastError();
}

}  // namespace UFF
}  // namespace nvMolKit
