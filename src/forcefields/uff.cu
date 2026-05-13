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

#include <cassert>
#include <vector>

#include "kernel_utils.cuh"
#include "uff.h"

namespace nvMolKit {
namespace UFF {
namespace {

EnergyForceContribsDevicePtr toPointerStruct(const EnergyForceContribsDevice& src) {
  EnergyForceContribsDevicePtr dst;
  dst.bondTerms.idx1          = src.bondTerms.idx1.data();
  dst.bondTerms.idx2          = src.bondTerms.idx2.data();
  dst.bondTerms.restLen       = src.bondTerms.restLen.data();
  dst.bondTerms.forceConstant = src.bondTerms.forceConstant.data();

  dst.angleTerms.idx1          = src.angleTerms.idx1.data();
  dst.angleTerms.idx2          = src.angleTerms.idx2.data();
  dst.angleTerms.idx3          = src.angleTerms.idx3.data();
  dst.angleTerms.theta0        = src.angleTerms.theta0.data();
  dst.angleTerms.forceConstant = src.angleTerms.forceConstant.data();
  dst.angleTerms.order         = src.angleTerms.order.data();
  dst.angleTerms.C0            = src.angleTerms.C0.data();
  dst.angleTerms.C1            = src.angleTerms.C1.data();
  dst.angleTerms.C2            = src.angleTerms.C2.data();

  dst.torsionTerms.idx1          = src.torsionTerms.idx1.data();
  dst.torsionTerms.idx2          = src.torsionTerms.idx2.data();
  dst.torsionTerms.idx3          = src.torsionTerms.idx3.data();
  dst.torsionTerms.idx4          = src.torsionTerms.idx4.data();
  dst.torsionTerms.forceConstant = src.torsionTerms.forceConstant.data();
  dst.torsionTerms.order         = src.torsionTerms.order.data();
  dst.torsionTerms.cosTerm       = src.torsionTerms.cosTerm.data();

  dst.inversionTerms.idx1          = src.inversionTerms.idx1.data();
  dst.inversionTerms.idx2          = src.inversionTerms.idx2.data();
  dst.inversionTerms.idx3          = src.inversionTerms.idx3.data();
  dst.inversionTerms.idx4          = src.inversionTerms.idx4.data();
  dst.inversionTerms.forceConstant = src.inversionTerms.forceConstant.data();
  dst.inversionTerms.C0            = src.inversionTerms.C0.data();
  dst.inversionTerms.C1            = src.inversionTerms.C1.data();
  dst.inversionTerms.C2            = src.inversionTerms.C2.data();

  dst.vdwTerms.idx1      = src.vdwTerms.idx1.data();
  dst.vdwTerms.idx2      = src.vdwTerms.idx2.data();
  dst.vdwTerms.x_ij      = src.vdwTerms.x_ij.data();
  dst.vdwTerms.wellDepth = src.vdwTerms.wellDepth.data();
  dst.vdwTerms.threshold = src.vdwTerms.threshold.data();

  dst.distanceConstraintTerms.idx1          = src.distanceConstraintTerms.idx1.data();
  dst.distanceConstraintTerms.idx2          = src.distanceConstraintTerms.idx2.data();
  dst.distanceConstraintTerms.minLen        = src.distanceConstraintTerms.minLen.data();
  dst.distanceConstraintTerms.maxLen        = src.distanceConstraintTerms.maxLen.data();
  dst.distanceConstraintTerms.forceConstant = src.distanceConstraintTerms.forceConstant.data();

  dst.positionConstraintTerms.idx           = src.positionConstraintTerms.idx.data();
  dst.positionConstraintTerms.refX          = src.positionConstraintTerms.refX.data();
  dst.positionConstraintTerms.refY          = src.positionConstraintTerms.refY.data();
  dst.positionConstraintTerms.refZ          = src.positionConstraintTerms.refZ.data();
  dst.positionConstraintTerms.maxDispl      = src.positionConstraintTerms.maxDispl.data();
  dst.positionConstraintTerms.forceConstant = src.positionConstraintTerms.forceConstant.data();

  dst.angleConstraintTerms.idx1          = src.angleConstraintTerms.idx1.data();
  dst.angleConstraintTerms.idx2          = src.angleConstraintTerms.idx2.data();
  dst.angleConstraintTerms.idx3          = src.angleConstraintTerms.idx3.data();
  dst.angleConstraintTerms.minAngleDeg   = src.angleConstraintTerms.minAngleDeg.data();
  dst.angleConstraintTerms.maxAngleDeg   = src.angleConstraintTerms.maxAngleDeg.data();
  dst.angleConstraintTerms.forceConstant = src.angleConstraintTerms.forceConstant.data();

  dst.torsionConstraintTerms.idx1           = src.torsionConstraintTerms.idx1.data();
  dst.torsionConstraintTerms.idx2           = src.torsionConstraintTerms.idx2.data();
  dst.torsionConstraintTerms.idx3           = src.torsionConstraintTerms.idx3.data();
  dst.torsionConstraintTerms.idx4           = src.torsionConstraintTerms.idx4.data();
  dst.torsionConstraintTerms.minDihedralDeg = src.torsionConstraintTerms.minDihedralDeg.data();
  dst.torsionConstraintTerms.maxDihedralDeg = src.torsionConstraintTerms.maxDihedralDeg.data();
  dst.torsionConstraintTerms.forceConstant  = src.torsionConstraintTerms.forceConstant.data();
  return dst;
}

BatchedIndicesDevicePtr toPointerStruct(const BatchedIndicesDevice& src) {
  BatchedIndicesDevicePtr dst;
  dst.atomStarts                   = src.atomStarts.data();
  dst.bondTermStarts               = src.bondTermStarts.data();
  dst.angleTermStarts              = src.angleTermStarts.data();
  dst.torsionTermStarts            = src.torsionTermStarts.data();
  dst.inversionTermStarts          = src.inversionTermStarts.data();
  dst.vdwTermStarts                = src.vdwTermStarts.data();
  dst.distanceConstraintTermStarts = src.distanceConstraintTermStarts.data();
  dst.positionConstraintTermStarts = src.positionConstraintTermStarts.data();
  dst.angleConstraintTermStarts    = src.angleConstraintTermStarts.data();
  dst.torsionConstraintTermStarts  = src.torsionConstraintTermStarts.data();
  return dst;
}

__global__ void zeroInactiveGradientEntries(const int*     atomIdxToBatchIdx,
                                            const uint8_t* activeSystemMask,
                                            const int      numAtoms,
                                            const int      dataDim,
                                            double*        grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numAtoms * dataDim) {
    return;
  }
  const int atomIdx  = idx / dataDim;
  const int batchIdx = atomIdxToBatchIdx[atomIdx];
  if (activeSystemMask[batchIdx] == 0) {
    grad[idx] = 0.0;
  }
}

template <typename T> void appendOffsetIndices(std::vector<T>& dst, const std::vector<T>& src, const int offset) {
  dst.reserve(dst.size() + src.size());
  for (const auto value : src) {
    dst.push_back(value + offset);
  }
}

template <typename T> void appendValues(std::vector<T>& dst, const std::vector<T>& src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

}  // namespace

void setStreams(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream) {
  molSystemDevice.positions.setStream(stream);
  molSystemDevice.grad.setStream(stream);
  molSystemDevice.energyOuts.setStream(stream);
  molSystemDevice.energyBuffer.setStream(stream);

  auto& contribs = molSystemDevice.contribs;
  contribs.bondTerms.idx1.setStream(stream);
  contribs.bondTerms.idx2.setStream(stream);
  contribs.bondTerms.restLen.setStream(stream);
  contribs.bondTerms.forceConstant.setStream(stream);

  contribs.angleTerms.idx1.setStream(stream);
  contribs.angleTerms.idx2.setStream(stream);
  contribs.angleTerms.idx3.setStream(stream);
  contribs.angleTerms.theta0.setStream(stream);
  contribs.angleTerms.forceConstant.setStream(stream);
  contribs.angleTerms.order.setStream(stream);
  contribs.angleTerms.C0.setStream(stream);
  contribs.angleTerms.C1.setStream(stream);
  contribs.angleTerms.C2.setStream(stream);

  contribs.torsionTerms.idx1.setStream(stream);
  contribs.torsionTerms.idx2.setStream(stream);
  contribs.torsionTerms.idx3.setStream(stream);
  contribs.torsionTerms.idx4.setStream(stream);
  contribs.torsionTerms.forceConstant.setStream(stream);
  contribs.torsionTerms.order.setStream(stream);
  contribs.torsionTerms.cosTerm.setStream(stream);

  contribs.inversionTerms.idx1.setStream(stream);
  contribs.inversionTerms.idx2.setStream(stream);
  contribs.inversionTerms.idx3.setStream(stream);
  contribs.inversionTerms.idx4.setStream(stream);
  contribs.inversionTerms.forceConstant.setStream(stream);
  contribs.inversionTerms.C0.setStream(stream);
  contribs.inversionTerms.C1.setStream(stream);
  contribs.inversionTerms.C2.setStream(stream);

  contribs.vdwTerms.idx1.setStream(stream);
  contribs.vdwTerms.idx2.setStream(stream);
  contribs.vdwTerms.x_ij.setStream(stream);
  contribs.vdwTerms.wellDepth.setStream(stream);
  contribs.vdwTerms.threshold.setStream(stream);

  contribs.distanceConstraintTerms.idx1.setStream(stream);
  contribs.distanceConstraintTerms.idx2.setStream(stream);
  contribs.distanceConstraintTerms.minLen.setStream(stream);
  contribs.distanceConstraintTerms.maxLen.setStream(stream);
  contribs.distanceConstraintTerms.forceConstant.setStream(stream);

  contribs.positionConstraintTerms.idx.setStream(stream);
  contribs.positionConstraintTerms.refX.setStream(stream);
  contribs.positionConstraintTerms.refY.setStream(stream);
  contribs.positionConstraintTerms.refZ.setStream(stream);
  contribs.positionConstraintTerms.maxDispl.setStream(stream);
  contribs.positionConstraintTerms.forceConstant.setStream(stream);

  contribs.angleConstraintTerms.idx1.setStream(stream);
  contribs.angleConstraintTerms.idx2.setStream(stream);
  contribs.angleConstraintTerms.idx3.setStream(stream);
  contribs.angleConstraintTerms.minAngleDeg.setStream(stream);
  contribs.angleConstraintTerms.maxAngleDeg.setStream(stream);
  contribs.angleConstraintTerms.forceConstant.setStream(stream);

  contribs.torsionConstraintTerms.idx1.setStream(stream);
  contribs.torsionConstraintTerms.idx2.setStream(stream);
  contribs.torsionConstraintTerms.idx3.setStream(stream);
  contribs.torsionConstraintTerms.idx4.setStream(stream);
  contribs.torsionConstraintTerms.minDihedralDeg.setStream(stream);
  contribs.torsionConstraintTerms.maxDihedralDeg.setStream(stream);
  contribs.torsionConstraintTerms.forceConstant.setStream(stream);

  molSystemDevice.indices.atomStarts.setStream(stream);
  molSystemDevice.indices.atomIdxToBatchIdx.setStream(stream);
  molSystemDevice.indices.energyBufferStarts.setStream(stream);
  molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.setStream(stream);
  molSystemDevice.indices.bondTermStarts.setStream(stream);
  molSystemDevice.indices.angleTermStarts.setStream(stream);
  molSystemDevice.indices.torsionTermStarts.setStream(stream);
  molSystemDevice.indices.inversionTermStarts.setStream(stream);
  molSystemDevice.indices.vdwTermStarts.setStream(stream);
  molSystemDevice.indices.distanceConstraintTermStarts.setStream(stream);
  molSystemDevice.indices.positionConstraintTermStarts.setStream(stream);
  molSystemDevice.indices.angleConstraintTermStarts.setStream(stream);
  molSystemDevice.indices.torsionConstraintTermStarts.setStream(stream);
}

void sendContribsAndIndicesToDevice(const BatchedMolecularSystemHost& molSystemHost,
                                    BatchedMolecularDeviceBuffers&    molSystemDevice) {
  const auto& hostContribs = molSystemHost.contribs;
  auto&       contribs     = molSystemDevice.contribs;

  contribs.bondTerms.idx1.setFromVector(hostContribs.bondTerms.idx1);
  contribs.bondTerms.idx2.setFromVector(hostContribs.bondTerms.idx2);
  contribs.bondTerms.restLen.setFromVector(hostContribs.bondTerms.restLen);
  contribs.bondTerms.forceConstant.setFromVector(hostContribs.bondTerms.forceConstant);

  contribs.angleTerms.idx1.setFromVector(hostContribs.angleTerms.idx1);
  contribs.angleTerms.idx2.setFromVector(hostContribs.angleTerms.idx2);
  contribs.angleTerms.idx3.setFromVector(hostContribs.angleTerms.idx3);
  contribs.angleTerms.theta0.setFromVector(hostContribs.angleTerms.theta0);
  contribs.angleTerms.forceConstant.setFromVector(hostContribs.angleTerms.forceConstant);
  contribs.angleTerms.order.setFromVector(hostContribs.angleTerms.order);
  contribs.angleTerms.C0.setFromVector(hostContribs.angleTerms.C0);
  contribs.angleTerms.C1.setFromVector(hostContribs.angleTerms.C1);
  contribs.angleTerms.C2.setFromVector(hostContribs.angleTerms.C2);

  contribs.torsionTerms.idx1.setFromVector(hostContribs.torsionTerms.idx1);
  contribs.torsionTerms.idx2.setFromVector(hostContribs.torsionTerms.idx2);
  contribs.torsionTerms.idx3.setFromVector(hostContribs.torsionTerms.idx3);
  contribs.torsionTerms.idx4.setFromVector(hostContribs.torsionTerms.idx4);
  contribs.torsionTerms.forceConstant.setFromVector(hostContribs.torsionTerms.forceConstant);
  contribs.torsionTerms.order.setFromVector(hostContribs.torsionTerms.order);
  contribs.torsionTerms.cosTerm.setFromVector(hostContribs.torsionTerms.cosTerm);

  contribs.inversionTerms.idx1.setFromVector(hostContribs.inversionTerms.idx1);
  contribs.inversionTerms.idx2.setFromVector(hostContribs.inversionTerms.idx2);
  contribs.inversionTerms.idx3.setFromVector(hostContribs.inversionTerms.idx3);
  contribs.inversionTerms.idx4.setFromVector(hostContribs.inversionTerms.idx4);
  contribs.inversionTerms.forceConstant.setFromVector(hostContribs.inversionTerms.forceConstant);
  contribs.inversionTerms.C0.setFromVector(hostContribs.inversionTerms.C0);
  contribs.inversionTerms.C1.setFromVector(hostContribs.inversionTerms.C1);
  contribs.inversionTerms.C2.setFromVector(hostContribs.inversionTerms.C2);

  contribs.vdwTerms.idx1.setFromVector(hostContribs.vdwTerms.idx1);
  contribs.vdwTerms.idx2.setFromVector(hostContribs.vdwTerms.idx2);
  contribs.vdwTerms.x_ij.setFromVector(hostContribs.vdwTerms.x_ij);
  contribs.vdwTerms.wellDepth.setFromVector(hostContribs.vdwTerms.wellDepth);
  contribs.vdwTerms.threshold.setFromVector(hostContribs.vdwTerms.threshold);

  contribs.distanceConstraintTerms.idx1.setFromVector(hostContribs.distanceConstraintTerms.idx1);
  contribs.distanceConstraintTerms.idx2.setFromVector(hostContribs.distanceConstraintTerms.idx2);
  contribs.distanceConstraintTerms.minLen.setFromVector(hostContribs.distanceConstraintTerms.minLen);
  contribs.distanceConstraintTerms.maxLen.setFromVector(hostContribs.distanceConstraintTerms.maxLen);
  contribs.distanceConstraintTerms.forceConstant.setFromVector(hostContribs.distanceConstraintTerms.forceConstant);

  contribs.positionConstraintTerms.idx.setFromVector(hostContribs.positionConstraintTerms.idx);
  contribs.positionConstraintTerms.refX.setFromVector(hostContribs.positionConstraintTerms.refX);
  contribs.positionConstraintTerms.refY.setFromVector(hostContribs.positionConstraintTerms.refY);
  contribs.positionConstraintTerms.refZ.setFromVector(hostContribs.positionConstraintTerms.refZ);
  contribs.positionConstraintTerms.maxDispl.setFromVector(hostContribs.positionConstraintTerms.maxDispl);
  contribs.positionConstraintTerms.forceConstant.setFromVector(hostContribs.positionConstraintTerms.forceConstant);

  contribs.angleConstraintTerms.idx1.setFromVector(hostContribs.angleConstraintTerms.idx1);
  contribs.angleConstraintTerms.idx2.setFromVector(hostContribs.angleConstraintTerms.idx2);
  contribs.angleConstraintTerms.idx3.setFromVector(hostContribs.angleConstraintTerms.idx3);
  contribs.angleConstraintTerms.minAngleDeg.setFromVector(hostContribs.angleConstraintTerms.minAngleDeg);
  contribs.angleConstraintTerms.maxAngleDeg.setFromVector(hostContribs.angleConstraintTerms.maxAngleDeg);
  contribs.angleConstraintTerms.forceConstant.setFromVector(hostContribs.angleConstraintTerms.forceConstant);

  contribs.torsionConstraintTerms.idx1.setFromVector(hostContribs.torsionConstraintTerms.idx1);
  contribs.torsionConstraintTerms.idx2.setFromVector(hostContribs.torsionConstraintTerms.idx2);
  contribs.torsionConstraintTerms.idx3.setFromVector(hostContribs.torsionConstraintTerms.idx3);
  contribs.torsionConstraintTerms.idx4.setFromVector(hostContribs.torsionConstraintTerms.idx4);
  contribs.torsionConstraintTerms.minDihedralDeg.setFromVector(hostContribs.torsionConstraintTerms.minDihedralDeg);
  contribs.torsionConstraintTerms.maxDihedralDeg.setFromVector(hostContribs.torsionConstraintTerms.maxDihedralDeg);
  contribs.torsionConstraintTerms.forceConstant.setFromVector(hostContribs.torsionConstraintTerms.forceConstant);

  molSystemDevice.indices.atomStarts.setFromVector(molSystemHost.indices.atomStarts);
  molSystemDevice.indices.atomIdxToBatchIdx.setFromVector(molSystemHost.indices.atomIdxToBatchIdx);
  molSystemDevice.indices.energyBufferStarts.setFromVector(molSystemHost.indices.energyBufferStarts);
  molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.setFromVector(
    molSystemHost.indices.energyBufferBlockIdxToBatchIdx);
  molSystemDevice.indices.bondTermStarts.setFromVector(molSystemHost.indices.bondTermStarts);
  molSystemDevice.indices.angleTermStarts.setFromVector(molSystemHost.indices.angleTermStarts);
  molSystemDevice.indices.torsionTermStarts.setFromVector(molSystemHost.indices.torsionTermStarts);
  molSystemDevice.indices.inversionTermStarts.setFromVector(molSystemHost.indices.inversionTermStarts);
  molSystemDevice.indices.vdwTermStarts.setFromVector(molSystemHost.indices.vdwTermStarts);
  molSystemDevice.indices.distanceConstraintTermStarts.setFromVector(
    molSystemHost.indices.distanceConstraintTermStarts);
  molSystemDevice.indices.positionConstraintTermStarts.setFromVector(
    molSystemHost.indices.positionConstraintTermStarts);
  molSystemDevice.indices.angleConstraintTermStarts.setFromVector(molSystemHost.indices.angleConstraintTermStarts);
  molSystemDevice.indices.torsionConstraintTermStarts.setFromVector(molSystemHost.indices.torsionConstraintTermStarts);
}

void addMoleculeToBatch(const EnergyForceContribsHost& contribs,
                        const std::vector<double>&     positions,
                        BatchedMolecularSystemHost&    molSystem) {
  const int atomOffset  = molSystem.indices.atomStarts.back();
  const int batchIdx    = static_cast<int>(molSystem.indices.atomStarts.size()) - 1;
  const int newNumAtoms = static_cast<int>(positions.size()) / 3;
  auto&     indices     = molSystem.indices;
  auto&     dstContribs = molSystem.contribs;

  indices.atomStarts.push_back(atomOffset + newNumAtoms);
  molSystem.maxNumAtoms = std::max(molSystem.maxNumAtoms, newNumAtoms);
  molSystem.positions.insert(molSystem.positions.end(), positions.begin(), positions.end());
  indices.atomIdxToBatchIdx.resize(molSystem.positions.size() / 3, batchIdx);

  indices.bondTermStarts.push_back(indices.bondTermStarts.back() + contribs.bondTerms.idx1.size());
  indices.angleTermStarts.push_back(indices.angleTermStarts.back() + contribs.angleTerms.idx1.size());
  indices.torsionTermStarts.push_back(indices.torsionTermStarts.back() + contribs.torsionTerms.idx1.size());
  indices.inversionTermStarts.push_back(indices.inversionTermStarts.back() + contribs.inversionTerms.idx1.size());
  indices.vdwTermStarts.push_back(indices.vdwTermStarts.back() + contribs.vdwTerms.idx1.size());
  indices.distanceConstraintTermStarts.push_back(indices.distanceConstraintTermStarts.back() +
                                                 contribs.distanceConstraintTerms.idx1.size());
  indices.positionConstraintTermStarts.push_back(indices.positionConstraintTermStarts.back() +
                                                 contribs.positionConstraintTerms.idx.size());
  indices.angleConstraintTermStarts.push_back(indices.angleConstraintTermStarts.back() +
                                              contribs.angleConstraintTerms.idx1.size());
  indices.torsionConstraintTermStarts.push_back(indices.torsionConstraintTermStarts.back() +
                                                contribs.torsionConstraintTerms.idx1.size());

  int maxNumContribs = 0;
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.bondTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.angleTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.torsionTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.inversionTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.vdwTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.distanceConstraintTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.positionConstraintTerms.idx.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.angleConstraintTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.torsionConstraintTerms.idx1.size());

  const int numBlocksNeeded =
    (maxNumContribs + FFKernelUtils::blockSizeEnergyReduction - 1) / FFKernelUtils::blockSizeEnergyReduction;
  const int numThreadsNeeded = numBlocksNeeded * FFKernelUtils::blockSizeEnergyReduction;
  indices.energyBufferStarts.push_back(indices.energyBufferStarts.back() + numThreadsNeeded);
  for (int i = 0; i < numBlocksNeeded; ++i) {
    indices.energyBufferBlockIdxToBatchIdx.push_back(batchIdx);
  }

  appendOffsetIndices(dstContribs.bondTerms.idx1, contribs.bondTerms.idx1, atomOffset);
  appendOffsetIndices(dstContribs.bondTerms.idx2, contribs.bondTerms.idx2, atomOffset);
  appendValues(dstContribs.bondTerms.restLen, contribs.bondTerms.restLen);
  appendValues(dstContribs.bondTerms.forceConstant, contribs.bondTerms.forceConstant);

  appendOffsetIndices(dstContribs.angleTerms.idx1, contribs.angleTerms.idx1, atomOffset);
  appendOffsetIndices(dstContribs.angleTerms.idx2, contribs.angleTerms.idx2, atomOffset);
  appendOffsetIndices(dstContribs.angleTerms.idx3, contribs.angleTerms.idx3, atomOffset);
  appendValues(dstContribs.angleTerms.theta0, contribs.angleTerms.theta0);
  appendValues(dstContribs.angleTerms.forceConstant, contribs.angleTerms.forceConstant);
  appendValues(dstContribs.angleTerms.order, contribs.angleTerms.order);
  appendValues(dstContribs.angleTerms.C0, contribs.angleTerms.C0);
  appendValues(dstContribs.angleTerms.C1, contribs.angleTerms.C1);
  appendValues(dstContribs.angleTerms.C2, contribs.angleTerms.C2);

  appendOffsetIndices(dstContribs.torsionTerms.idx1, contribs.torsionTerms.idx1, atomOffset);
  appendOffsetIndices(dstContribs.torsionTerms.idx2, contribs.torsionTerms.idx2, atomOffset);
  appendOffsetIndices(dstContribs.torsionTerms.idx3, contribs.torsionTerms.idx3, atomOffset);
  appendOffsetIndices(dstContribs.torsionTerms.idx4, contribs.torsionTerms.idx4, atomOffset);
  appendValues(dstContribs.torsionTerms.forceConstant, contribs.torsionTerms.forceConstant);
  appendValues(dstContribs.torsionTerms.order, contribs.torsionTerms.order);
  appendValues(dstContribs.torsionTerms.cosTerm, contribs.torsionTerms.cosTerm);

  appendOffsetIndices(dstContribs.inversionTerms.idx1, contribs.inversionTerms.idx1, atomOffset);
  appendOffsetIndices(dstContribs.inversionTerms.idx2, contribs.inversionTerms.idx2, atomOffset);
  appendOffsetIndices(dstContribs.inversionTerms.idx3, contribs.inversionTerms.idx3, atomOffset);
  appendOffsetIndices(dstContribs.inversionTerms.idx4, contribs.inversionTerms.idx4, atomOffset);
  appendValues(dstContribs.inversionTerms.forceConstant, contribs.inversionTerms.forceConstant);
  appendValues(dstContribs.inversionTerms.C0, contribs.inversionTerms.C0);
  appendValues(dstContribs.inversionTerms.C1, contribs.inversionTerms.C1);
  appendValues(dstContribs.inversionTerms.C2, contribs.inversionTerms.C2);

  appendOffsetIndices(dstContribs.vdwTerms.idx1, contribs.vdwTerms.idx1, atomOffset);
  appendOffsetIndices(dstContribs.vdwTerms.idx2, contribs.vdwTerms.idx2, atomOffset);
  appendValues(dstContribs.vdwTerms.x_ij, contribs.vdwTerms.x_ij);
  appendValues(dstContribs.vdwTerms.wellDepth, contribs.vdwTerms.wellDepth);
  appendValues(dstContribs.vdwTerms.threshold, contribs.vdwTerms.threshold);

  appendOffsetIndices(dstContribs.distanceConstraintTerms.idx1, contribs.distanceConstraintTerms.idx1, atomOffset);
  appendOffsetIndices(dstContribs.distanceConstraintTerms.idx2, contribs.distanceConstraintTerms.idx2, atomOffset);
  appendValues(dstContribs.distanceConstraintTerms.minLen, contribs.distanceConstraintTerms.minLen);
  appendValues(dstContribs.distanceConstraintTerms.maxLen, contribs.distanceConstraintTerms.maxLen);
  appendValues(dstContribs.distanceConstraintTerms.forceConstant, contribs.distanceConstraintTerms.forceConstant);

  appendOffsetIndices(dstContribs.positionConstraintTerms.idx, contribs.positionConstraintTerms.idx, atomOffset);
  appendValues(dstContribs.positionConstraintTerms.refX, contribs.positionConstraintTerms.refX);
  appendValues(dstContribs.positionConstraintTerms.refY, contribs.positionConstraintTerms.refY);
  appendValues(dstContribs.positionConstraintTerms.refZ, contribs.positionConstraintTerms.refZ);
  appendValues(dstContribs.positionConstraintTerms.maxDispl, contribs.positionConstraintTerms.maxDispl);
  appendValues(dstContribs.positionConstraintTerms.forceConstant, contribs.positionConstraintTerms.forceConstant);

  appendOffsetIndices(dstContribs.angleConstraintTerms.idx1, contribs.angleConstraintTerms.idx1, atomOffset);
  appendOffsetIndices(dstContribs.angleConstraintTerms.idx2, contribs.angleConstraintTerms.idx2, atomOffset);
  appendOffsetIndices(dstContribs.angleConstraintTerms.idx3, contribs.angleConstraintTerms.idx3, atomOffset);
  appendValues(dstContribs.angleConstraintTerms.minAngleDeg, contribs.angleConstraintTerms.minAngleDeg);
  appendValues(dstContribs.angleConstraintTerms.maxAngleDeg, contribs.angleConstraintTerms.maxAngleDeg);
  appendValues(dstContribs.angleConstraintTerms.forceConstant, contribs.angleConstraintTerms.forceConstant);

  appendOffsetIndices(dstContribs.torsionConstraintTerms.idx1, contribs.torsionConstraintTerms.idx1, atomOffset);
  appendOffsetIndices(dstContribs.torsionConstraintTerms.idx2, contribs.torsionConstraintTerms.idx2, atomOffset);
  appendOffsetIndices(dstContribs.torsionConstraintTerms.idx3, contribs.torsionConstraintTerms.idx3, atomOffset);
  appendOffsetIndices(dstContribs.torsionConstraintTerms.idx4, contribs.torsionConstraintTerms.idx4, atomOffset);
  appendValues(dstContribs.torsionConstraintTerms.minDihedralDeg, contribs.torsionConstraintTerms.minDihedralDeg);
  appendValues(dstContribs.torsionConstraintTerms.maxDihedralDeg, contribs.torsionConstraintTerms.maxDihedralDeg);
  appendValues(dstContribs.torsionConstraintTerms.forceConstant, contribs.torsionConstraintTerms.forceConstant);
}

void addMoleculeToBatch(const EnergyForceContribsHost& contribs,
                        const std::vector<double>&     positions,
                        BatchedMolecularSystemHost&    molSystem,
                        BatchedForcefieldMetadata&     metadata,
                        const int                      moleculeIdx,
                        const int                      conformerIdx,
                        const HostCustomization&       customization) {
  EnergyForceContribsHost contribsCopy = contribs;
  const BatchedSystemInfo systemInfo   = metadata.recordSystem(moleculeIdx, conformerIdx);
  if (customization) {
    customization(systemInfo, positions, contribsCopy);
  }
  addMoleculeToBatch(contribsCopy, positions, molSystem);
}

void allocateIntermediateBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                 BatchedMolecularDeviceBuffers&    molSystemDevice) {
  FFKernelUtils::allocateIntermediateBuffers(molSystemHost,
                                             molSystemDevice,
                                             molSystemHost.indices.atomStarts.size() - 1);
}

cudaError_t computeEnergy(BatchedMolecularDeviceBuffers& molSystemDevice,
                          double*                        energyOuts,
                          const double*                  positions,
                          const uint8_t*                 activeSystemMask,
                          cudaStream_t                   stream) {
  assert(molSystemDevice.energyBuffer.size() > 0);
  assert(energyOuts != nullptr);
  molSystemDevice.energyBuffer.zero();

  const auto& contribs = molSystemDevice.contribs;
  cudaError_t err      = cudaSuccess;
  if (contribs.bondTerms.idx1.size() > 0) {
    err = launchBondStretchEnergyKernel(contribs.bondTerms.idx1.size(),
                                        contribs.bondTerms.idx1.data(),
                                        contribs.bondTerms.idx2.data(),
                                        contribs.bondTerms.restLen.data(),
                                        contribs.bondTerms.forceConstant.data(),
                                        positions,
                                        molSystemDevice.energyBuffer.data(),
                                        molSystemDevice.indices.energyBufferStarts.data(),
                                        molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                        molSystemDevice.indices.bondTermStarts.data(),
                                        stream);
  }
  if (err == cudaSuccess && contribs.angleTerms.idx1.size() > 0) {
    err = launchAngleBendEnergyKernel(contribs.angleTerms.idx1.size(),
                                      contribs.angleTerms.idx1.data(),
                                      contribs.angleTerms.idx2.data(),
                                      contribs.angleTerms.idx3.data(),
                                      contribs.angleTerms.theta0.data(),
                                      contribs.angleTerms.forceConstant.data(),
                                      contribs.angleTerms.order.data(),
                                      contribs.angleTerms.C0.data(),
                                      contribs.angleTerms.C1.data(),
                                      contribs.angleTerms.C2.data(),
                                      positions,
                                      molSystemDevice.energyBuffer.data(),
                                      molSystemDevice.indices.energyBufferStarts.data(),
                                      molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                      molSystemDevice.indices.angleTermStarts.data(),
                                      stream);
  }
  if (err == cudaSuccess && contribs.torsionTerms.idx1.size() > 0) {
    err = launchTorsionEnergyKernel(contribs.torsionTerms.idx1.size(),
                                    contribs.torsionTerms.idx1.data(),
                                    contribs.torsionTerms.idx2.data(),
                                    contribs.torsionTerms.idx3.data(),
                                    contribs.torsionTerms.idx4.data(),
                                    contribs.torsionTerms.forceConstant.data(),
                                    contribs.torsionTerms.order.data(),
                                    contribs.torsionTerms.cosTerm.data(),
                                    positions,
                                    molSystemDevice.energyBuffer.data(),
                                    molSystemDevice.indices.energyBufferStarts.data(),
                                    molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                    molSystemDevice.indices.torsionTermStarts.data(),
                                    stream);
  }
  if (err == cudaSuccess && contribs.inversionTerms.idx1.size() > 0) {
    err = launchInversionEnergyKernel(contribs.inversionTerms.idx1.size(),
                                      contribs.inversionTerms.idx1.data(),
                                      contribs.inversionTerms.idx2.data(),
                                      contribs.inversionTerms.idx3.data(),
                                      contribs.inversionTerms.idx4.data(),
                                      contribs.inversionTerms.forceConstant.data(),
                                      contribs.inversionTerms.C0.data(),
                                      contribs.inversionTerms.C1.data(),
                                      contribs.inversionTerms.C2.data(),
                                      positions,
                                      molSystemDevice.energyBuffer.data(),
                                      molSystemDevice.indices.energyBufferStarts.data(),
                                      molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                      molSystemDevice.indices.inversionTermStarts.data(),
                                      stream);
  }
  if (err == cudaSuccess && contribs.vdwTerms.idx1.size() > 0) {
    err = launchVdwEnergyKernel(contribs.vdwTerms.idx1.size(),
                                contribs.vdwTerms.idx1.data(),
                                contribs.vdwTerms.idx2.data(),
                                contribs.vdwTerms.x_ij.data(),
                                contribs.vdwTerms.wellDepth.data(),
                                contribs.vdwTerms.threshold.data(),
                                positions,
                                molSystemDevice.energyBuffer.data(),
                                molSystemDevice.indices.energyBufferStarts.data(),
                                molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                molSystemDevice.indices.vdwTermStarts.data(),
                                stream);
  }
  if (err == cudaSuccess && contribs.distanceConstraintTerms.idx1.size() > 0) {
    err = MMFF::launchDistanceConstraintEnergyKernel(contribs.distanceConstraintTerms.idx1.size(),
                                                     contribs.distanceConstraintTerms.idx1.data(),
                                                     contribs.distanceConstraintTerms.idx2.data(),
                                                     contribs.distanceConstraintTerms.minLen.data(),
                                                     contribs.distanceConstraintTerms.maxLen.data(),
                                                     contribs.distanceConstraintTerms.forceConstant.data(),
                                                     positions,
                                                     molSystemDevice.energyBuffer.data(),
                                                     molSystemDevice.indices.energyBufferStarts.data(),
                                                     molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                                     molSystemDevice.indices.distanceConstraintTermStarts.data(),
                                                     stream);
  }
  if (err == cudaSuccess && contribs.positionConstraintTerms.idx.size() > 0) {
    err = MMFF::launchPositionConstraintEnergyKernel(contribs.positionConstraintTerms.idx.size(),
                                                     contribs.positionConstraintTerms.idx.data(),
                                                     contribs.positionConstraintTerms.refX.data(),
                                                     contribs.positionConstraintTerms.refY.data(),
                                                     contribs.positionConstraintTerms.refZ.data(),
                                                     contribs.positionConstraintTerms.maxDispl.data(),
                                                     contribs.positionConstraintTerms.forceConstant.data(),
                                                     positions,
                                                     molSystemDevice.energyBuffer.data(),
                                                     molSystemDevice.indices.energyBufferStarts.data(),
                                                     molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                                     molSystemDevice.indices.positionConstraintTermStarts.data(),
                                                     stream);
  }
  if (err == cudaSuccess && contribs.angleConstraintTerms.idx1.size() > 0) {
    err = MMFF::launchAngleConstraintEnergyKernel(contribs.angleConstraintTerms.idx1.size(),
                                                  contribs.angleConstraintTerms.idx1.data(),
                                                  contribs.angleConstraintTerms.idx2.data(),
                                                  contribs.angleConstraintTerms.idx3.data(),
                                                  contribs.angleConstraintTerms.minAngleDeg.data(),
                                                  contribs.angleConstraintTerms.maxAngleDeg.data(),
                                                  contribs.angleConstraintTerms.forceConstant.data(),
                                                  positions,
                                                  molSystemDevice.energyBuffer.data(),
                                                  molSystemDevice.indices.energyBufferStarts.data(),
                                                  molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                                  molSystemDevice.indices.angleConstraintTermStarts.data(),
                                                  stream);
  }
  if (err == cudaSuccess && contribs.torsionConstraintTerms.idx1.size() > 0) {
    err = MMFF::launchTorsionConstraintEnergyKernel(contribs.torsionConstraintTerms.idx1.size(),
                                                    contribs.torsionConstraintTerms.idx1.data(),
                                                    contribs.torsionConstraintTerms.idx2.data(),
                                                    contribs.torsionConstraintTerms.idx3.data(),
                                                    contribs.torsionConstraintTerms.idx4.data(),
                                                    contribs.torsionConstraintTerms.minDihedralDeg.data(),
                                                    contribs.torsionConstraintTerms.maxDihedralDeg.data(),
                                                    contribs.torsionConstraintTerms.forceConstant.data(),
                                                    positions,
                                                    molSystemDevice.energyBuffer.data(),
                                                    molSystemDevice.indices.energyBufferStarts.data(),
                                                    molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                                    molSystemDevice.indices.torsionConstraintTermStarts.data(),
                                                    stream);
  }
  if (err == cudaSuccess) {
    err = MMFF::launchReduceEnergiesKernel(molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.size(),
                                           molSystemDevice.energyBuffer.data(),
                                           molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.data(),
                                           energyOuts,
                                           activeSystemMask,
                                           stream);
  }
  return err;
}

cudaError_t computeEnergy(BatchedMolecularDeviceBuffers& molSystemDevice, const double* coords, cudaStream_t stream) {
  return computeEnergy(molSystemDevice,
                       molSystemDevice.energyOuts.data(),
                       coords != nullptr ? coords : molSystemDevice.positions.data(),
                       nullptr,
                       stream);
}

cudaError_t computeGradients(BatchedMolecularDeviceBuffers& molSystemDevice,
                             const double*                  positions,
                             double*                        grad,
                             const uint8_t*                 activeSystemMask,
                             cudaStream_t                   stream) {
  const auto& contribs = molSystemDevice.contribs;
  cudaError_t err      = cudaSuccess;
  if (contribs.bondTerms.idx1.size() > 0) {
    err = launchBondStretchGradientKernel(contribs.bondTerms.idx1.size(),
                                          contribs.bondTerms.idx1.data(),
                                          contribs.bondTerms.idx2.data(),
                                          contribs.bondTerms.restLen.data(),
                                          contribs.bondTerms.forceConstant.data(),
                                          positions,
                                          grad,
                                          stream);
  }
  if (err == cudaSuccess && contribs.angleTerms.idx1.size() > 0) {
    err = launchAngleBendGradientKernel(contribs.angleTerms.idx1.size(),
                                        contribs.angleTerms.idx1.data(),
                                        contribs.angleTerms.idx2.data(),
                                        contribs.angleTerms.idx3.data(),
                                        contribs.angleTerms.theta0.data(),
                                        contribs.angleTerms.forceConstant.data(),
                                        contribs.angleTerms.order.data(),
                                        contribs.angleTerms.C0.data(),
                                        contribs.angleTerms.C1.data(),
                                        contribs.angleTerms.C2.data(),
                                        positions,
                                        grad,
                                        stream);
  }
  if (err == cudaSuccess && contribs.torsionTerms.idx1.size() > 0) {
    err = launchTorsionGradientKernel(contribs.torsionTerms.idx1.size(),
                                      contribs.torsionTerms.idx1.data(),
                                      contribs.torsionTerms.idx2.data(),
                                      contribs.torsionTerms.idx3.data(),
                                      contribs.torsionTerms.idx4.data(),
                                      contribs.torsionTerms.forceConstant.data(),
                                      contribs.torsionTerms.order.data(),
                                      contribs.torsionTerms.cosTerm.data(),
                                      positions,
                                      grad,
                                      stream);
  }
  if (err == cudaSuccess && contribs.inversionTerms.idx1.size() > 0) {
    err = launchInversionGradientKernel(contribs.inversionTerms.idx1.size(),
                                        contribs.inversionTerms.idx1.data(),
                                        contribs.inversionTerms.idx2.data(),
                                        contribs.inversionTerms.idx3.data(),
                                        contribs.inversionTerms.idx4.data(),
                                        contribs.inversionTerms.forceConstant.data(),
                                        contribs.inversionTerms.C0.data(),
                                        contribs.inversionTerms.C1.data(),
                                        contribs.inversionTerms.C2.data(),
                                        positions,
                                        grad,
                                        stream);
  }
  if (err == cudaSuccess && contribs.vdwTerms.idx1.size() > 0) {
    err = launchVdwGradientKernel(contribs.vdwTerms.idx1.size(),
                                  contribs.vdwTerms.idx1.data(),
                                  contribs.vdwTerms.idx2.data(),
                                  contribs.vdwTerms.x_ij.data(),
                                  contribs.vdwTerms.wellDepth.data(),
                                  contribs.vdwTerms.threshold.data(),
                                  positions,
                                  grad,
                                  stream);
  }
  if (err == cudaSuccess && contribs.distanceConstraintTerms.idx1.size() > 0) {
    err = MMFF::launchDistanceConstraintGradientKernel(contribs.distanceConstraintTerms.idx1.size(),
                                                       contribs.distanceConstraintTerms.idx1.data(),
                                                       contribs.distanceConstraintTerms.idx2.data(),
                                                       contribs.distanceConstraintTerms.minLen.data(),
                                                       contribs.distanceConstraintTerms.maxLen.data(),
                                                       contribs.distanceConstraintTerms.forceConstant.data(),
                                                       positions,
                                                       grad,
                                                       stream);
  }
  if (err == cudaSuccess && contribs.positionConstraintTerms.idx.size() > 0) {
    err = MMFF::launchPositionConstraintGradientKernel(contribs.positionConstraintTerms.idx.size(),
                                                       contribs.positionConstraintTerms.idx.data(),
                                                       contribs.positionConstraintTerms.refX.data(),
                                                       contribs.positionConstraintTerms.refY.data(),
                                                       contribs.positionConstraintTerms.refZ.data(),
                                                       contribs.positionConstraintTerms.maxDispl.data(),
                                                       contribs.positionConstraintTerms.forceConstant.data(),
                                                       positions,
                                                       grad,
                                                       stream);
  }
  if (err == cudaSuccess && contribs.angleConstraintTerms.idx1.size() > 0) {
    err = MMFF::launchAngleConstraintGradientKernel(contribs.angleConstraintTerms.idx1.size(),
                                                    contribs.angleConstraintTerms.idx1.data(),
                                                    contribs.angleConstraintTerms.idx2.data(),
                                                    contribs.angleConstraintTerms.idx3.data(),
                                                    contribs.angleConstraintTerms.minAngleDeg.data(),
                                                    contribs.angleConstraintTerms.maxAngleDeg.data(),
                                                    contribs.angleConstraintTerms.forceConstant.data(),
                                                    positions,
                                                    grad,
                                                    stream);
  }
  if (err == cudaSuccess && contribs.torsionConstraintTerms.idx1.size() > 0) {
    err = MMFF::launchTorsionConstraintGradientKernel(contribs.torsionConstraintTerms.idx1.size(),
                                                      contribs.torsionConstraintTerms.idx1.data(),
                                                      contribs.torsionConstraintTerms.idx2.data(),
                                                      contribs.torsionConstraintTerms.idx3.data(),
                                                      contribs.torsionConstraintTerms.idx4.data(),
                                                      contribs.torsionConstraintTerms.minDihedralDeg.data(),
                                                      contribs.torsionConstraintTerms.maxDihedralDeg.data(),
                                                      contribs.torsionConstraintTerms.forceConstant.data(),
                                                      positions,
                                                      grad,
                                                      stream);
  }
  if (err == cudaSuccess && activeSystemMask != nullptr && molSystemDevice.indices.atomIdxToBatchIdx.size() > 0) {
    constexpr int blockSize = 256;
    const int     numTerms  = molSystemDevice.indices.atomIdxToBatchIdx.size() * 3;
    const int     numBlocks = (numTerms + blockSize - 1) / blockSize;
    zeroInactiveGradientEntries<<<numBlocks, blockSize, 0, stream>>>(molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                                                     activeSystemMask,
                                                                     molSystemDevice.indices.atomIdxToBatchIdx.size(),
                                                                     3,
                                                                     grad);
    err = cudaGetLastError();
  }
  return err;
}

cudaError_t computeGradients(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream) {
  return computeGradients(molSystemDevice,
                          molSystemDevice.positions.data(),
                          molSystemDevice.grad.data(),
                          nullptr,
                          stream);
}

cudaError_t computeEnergyBlockPerMol(BatchedMolecularDeviceBuffers& molSystemDevice,
                                     const double*                  coords,
                                     cudaStream_t                   stream) {
  return launchBlockPerMolEnergyKernel(molSystemDevice.indices.atomStarts.size() - 1,
                                       toPointerStruct(molSystemDevice.contribs),
                                       toPointerStruct(molSystemDevice.indices),
                                       coords != nullptr ? coords : molSystemDevice.positions.data(),
                                       molSystemDevice.energyOuts.data(),
                                       batchHasConstraints(molSystemDevice.contribs),
                                       stream);
}

cudaError_t computeGradBlockPerMol(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream) {
  return launchBlockPerMolGradKernel(molSystemDevice.indices.atomStarts.size() - 1,
                                     toPointerStruct(molSystemDevice.contribs),
                                     toPointerStruct(molSystemDevice.indices),
                                     molSystemDevice.positions.data(),
                                     molSystemDevice.grad.data(),
                                     batchHasConstraints(molSystemDevice.contribs),
                                     stream);
}

EnergyForceContribsDevicePtr toEnergyForceContribsDevicePtr(const BatchedMolecularDeviceBuffers& molSystemDevice) {
  return toPointerStruct(molSystemDevice.contribs);
}

BatchedIndicesDevicePtr toBatchedIndicesDevicePtr(const BatchedMolecularDeviceBuffers& molSystemDevice) {
  return toPointerStruct(molSystemDevice.indices);
}

bool batchHasConstraints(const EnergyForceContribsDevice& contribs) {
  return contribs.distanceConstraintTerms.idx1.size() > 0 || contribs.positionConstraintTerms.idx.size() > 0 ||
         contribs.angleConstraintTerms.idx1.size() > 0 || contribs.torsionConstraintTerms.idx1.size() > 0;
}

}  // namespace UFF
}  // namespace nvMolKit
