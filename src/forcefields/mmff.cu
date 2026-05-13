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

#include <cassert>
#include <vector>

#include "kernel_utils.cuh"
#include "mmff.h"
#include "mmff_kernels.h"

namespace nvMolKit {
namespace MMFF {
namespace {
EnergyForceContribsDevicePtr toPointerStruct(const EnergyForceContribsDevice& src) {
  EnergyForceContribsDevicePtr dst;
  dst.bondTerms.idx1 = src.bondTerms.idx1.data();
  dst.bondTerms.idx2 = src.bondTerms.idx2.data();
  dst.bondTerms.r0   = src.bondTerms.r0.data();
  dst.bondTerms.kb   = src.bondTerms.kb.data();

  dst.angleTerms.idx1     = src.angleTerms.idx1.data();
  dst.angleTerms.idx2     = src.angleTerms.idx2.data();
  dst.angleTerms.idx3     = src.angleTerms.idx3.data();
  dst.angleTerms.theta0   = src.angleTerms.theta0.data();
  dst.angleTerms.ka       = src.angleTerms.ka.data();
  dst.angleTerms.isLinear = src.angleTerms.isLinear.data();

  dst.bendTerms.idx1        = src.bendTerms.idx1.data();
  dst.bendTerms.idx2        = src.bendTerms.idx2.data();
  dst.bendTerms.idx3        = src.bendTerms.idx3.data();
  dst.bendTerms.theta0      = src.bendTerms.theta0.data();
  dst.bendTerms.restLen1    = src.bendTerms.restLen1.data();
  dst.bendTerms.restLen2    = src.bendTerms.restLen2.data();
  dst.bendTerms.forceConst1 = src.bendTerms.forceConst1.data();
  dst.bendTerms.forceConst2 = src.bendTerms.forceConst2.data();

  dst.oopTerms.idx1 = src.oopTerms.idx1.data();
  dst.oopTerms.idx2 = src.oopTerms.idx2.data();
  dst.oopTerms.idx3 = src.oopTerms.idx3.data();
  dst.oopTerms.idx4 = src.oopTerms.idx4.data();
  dst.oopTerms.koop = src.oopTerms.koop.data();

  dst.torsionTerms.idx1 = src.torsionTerms.idx1.data();
  dst.torsionTerms.idx2 = src.torsionTerms.idx2.data();
  dst.torsionTerms.idx3 = src.torsionTerms.idx3.data();
  dst.torsionTerms.idx4 = src.torsionTerms.idx4.data();
  dst.torsionTerms.V1   = src.torsionTerms.V1.data();
  dst.torsionTerms.V2   = src.torsionTerms.V2.data();
  dst.torsionTerms.V3   = src.torsionTerms.V3.data();

  dst.vdwTerms.idx1      = src.vdwTerms.idx1.data();
  dst.vdwTerms.idx2      = src.vdwTerms.idx2.data();
  dst.vdwTerms.R_ij_star = src.vdwTerms.R_ij_star.data();
  dst.vdwTerms.wellDepth = src.vdwTerms.wellDepth.data();

  dst.eleTerms.idx1       = src.eleTerms.idx1.data();
  dst.eleTerms.idx2       = src.eleTerms.idx2.data();
  dst.eleTerms.chargeTerm = src.eleTerms.chargeTerm.data();
  dst.eleTerms.dielModel  = src.eleTerms.dielModel.data();
  dst.eleTerms.is1_4      = src.eleTerms.is1_4.data();

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
  dst.bendTermStarts               = src.bendTermStarts.data();
  dst.oopTermStarts                = src.oopTermStarts.data();
  dst.torsionTermStarts            = src.torsionTermStarts.data();
  dst.vdwTermStarts                = src.vdwTermStarts.data();
  dst.eleTermStarts                = src.eleTermStarts.data();
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
}  // namespace

void setStreams(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream) {
  molSystemDevice.positions.setStream(stream);
  molSystemDevice.grad.setStream(stream);
  molSystemDevice.energyOuts.setStream(stream);
  molSystemDevice.energyBuffer.setStream(stream);

  auto& deviceContribs = molSystemDevice.contribs;
  // Bond terms
  deviceContribs.bondTerms.idx1.setStream(stream);
  deviceContribs.bondTerms.idx2.setStream(stream);
  deviceContribs.bondTerms.r0.setStream(stream);
  deviceContribs.bondTerms.kb.setStream(stream);
  // Angle terms
  deviceContribs.angleTerms.idx1.setStream(stream);
  deviceContribs.angleTerms.idx2.setStream(stream);
  deviceContribs.angleTerms.idx3.setStream(stream);
  deviceContribs.angleTerms.theta0.setStream(stream);
  deviceContribs.angleTerms.ka.setStream(stream);
  deviceContribs.angleTerms.isLinear.setStream(stream);

  // Bend terms
  deviceContribs.bendTerms.idx1.setStream(stream);
  deviceContribs.bendTerms.idx2.setStream(stream);
  deviceContribs.bendTerms.idx3.setStream(stream);
  deviceContribs.bendTerms.theta0.setStream(stream);
  deviceContribs.bendTerms.restLen1.setStream(stream);
  deviceContribs.bendTerms.restLen2.setStream(stream);
  deviceContribs.bendTerms.forceConst1.setStream(stream);
  deviceContribs.bendTerms.forceConst2.setStream(stream);

  // Oop terms
  deviceContribs.oopTerms.idx1.setStream(stream);
  deviceContribs.oopTerms.idx2.setStream(stream);
  deviceContribs.oopTerms.idx3.setStream(stream);
  deviceContribs.oopTerms.idx4.setStream(stream);
  deviceContribs.oopTerms.koop.setStream(stream);

  // Torsion terms
  deviceContribs.torsionTerms.idx1.setStream(stream);
  deviceContribs.torsionTerms.idx2.setStream(stream);
  deviceContribs.torsionTerms.idx3.setStream(stream);
  deviceContribs.torsionTerms.idx4.setStream(stream);
  deviceContribs.torsionTerms.V1.setStream(stream);
  deviceContribs.torsionTerms.V2.setStream(stream);
  deviceContribs.torsionTerms.V3.setStream(stream);

  // Vdw terms
  deviceContribs.vdwTerms.idx1.setStream(stream);
  deviceContribs.vdwTerms.idx2.setStream(stream);
  deviceContribs.vdwTerms.R_ij_star.setStream(stream);
  deviceContribs.vdwTerms.wellDepth.setStream(stream);

  // Ele terms
  deviceContribs.eleTerms.idx1.setStream(stream);
  deviceContribs.eleTerms.idx2.setStream(stream);
  deviceContribs.eleTerms.chargeTerm.setStream(stream);
  deviceContribs.eleTerms.dielModel.setStream(stream);
  deviceContribs.eleTerms.is1_4.setStream(stream);

  deviceContribs.distanceConstraintTerms.idx1.setStream(stream);
  deviceContribs.distanceConstraintTerms.idx2.setStream(stream);
  deviceContribs.distanceConstraintTerms.minLen.setStream(stream);
  deviceContribs.distanceConstraintTerms.maxLen.setStream(stream);
  deviceContribs.distanceConstraintTerms.forceConstant.setStream(stream);

  deviceContribs.positionConstraintTerms.idx.setStream(stream);
  deviceContribs.positionConstraintTerms.refX.setStream(stream);
  deviceContribs.positionConstraintTerms.refY.setStream(stream);
  deviceContribs.positionConstraintTerms.refZ.setStream(stream);
  deviceContribs.positionConstraintTerms.maxDispl.setStream(stream);
  deviceContribs.positionConstraintTerms.forceConstant.setStream(stream);

  deviceContribs.angleConstraintTerms.idx1.setStream(stream);
  deviceContribs.angleConstraintTerms.idx2.setStream(stream);
  deviceContribs.angleConstraintTerms.idx3.setStream(stream);
  deviceContribs.angleConstraintTerms.minAngleDeg.setStream(stream);
  deviceContribs.angleConstraintTerms.maxAngleDeg.setStream(stream);
  deviceContribs.angleConstraintTerms.forceConstant.setStream(stream);

  deviceContribs.torsionConstraintTerms.idx1.setStream(stream);
  deviceContribs.torsionConstraintTerms.idx2.setStream(stream);
  deviceContribs.torsionConstraintTerms.idx3.setStream(stream);
  deviceContribs.torsionConstraintTerms.idx4.setStream(stream);
  deviceContribs.torsionConstraintTerms.minDihedralDeg.setStream(stream);
  deviceContribs.torsionConstraintTerms.maxDihedralDeg.setStream(stream);
  deviceContribs.torsionConstraintTerms.forceConstant.setStream(stream);

  // Indices
  molSystemDevice.indices.atomStarts.setStream(stream);
  molSystemDevice.indices.energyBufferStarts.setStream(stream);
  molSystemDevice.indices.atomIdxToBatchIdx.setStream(stream);
  molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.setStream(stream);

  molSystemDevice.indices.bondTermStarts.setStream(stream);
  molSystemDevice.indices.angleTermStarts.setStream(stream);
  molSystemDevice.indices.bendTermStarts.setStream(stream);
  molSystemDevice.indices.oopTermStarts.setStream(stream);
  molSystemDevice.indices.torsionTermStarts.setStream(stream);
  molSystemDevice.indices.vdwTermStarts.setStream(stream);
  molSystemDevice.indices.eleTermStarts.setStream(stream);
  molSystemDevice.indices.distanceConstraintTermStarts.setStream(stream);
  molSystemDevice.indices.positionConstraintTermStarts.setStream(stream);
  molSystemDevice.indices.angleConstraintTermStarts.setStream(stream);
  molSystemDevice.indices.torsionConstraintTermStarts.setStream(stream);
}

void sendContribsAndIndicesToDevice(const BatchedMolecularSystemHost& molSystemHost,
                                    BatchedMolecularDeviceBuffers&    molSystemDevice) {
  auto&       deviceContribs = molSystemDevice.contribs;
  const auto& hostContribs   = molSystemHost.contribs;
  // Bond terms
  deviceContribs.bondTerms.idx1.setFromVector(hostContribs.bondTerms.idx1);
  deviceContribs.bondTerms.idx2.setFromVector(hostContribs.bondTerms.idx2);
  deviceContribs.bondTerms.r0.setFromVector(hostContribs.bondTerms.r0);
  deviceContribs.bondTerms.kb.setFromVector(hostContribs.bondTerms.kb);
  // Angle terms
  deviceContribs.angleTerms.idx1.setFromVector(hostContribs.angleTerms.idx1);
  deviceContribs.angleTerms.idx2.setFromVector(hostContribs.angleTerms.idx2);
  deviceContribs.angleTerms.idx3.setFromVector(hostContribs.angleTerms.idx3);
  deviceContribs.angleTerms.theta0.setFromVector(hostContribs.angleTerms.theta0);
  deviceContribs.angleTerms.ka.setFromVector(hostContribs.angleTerms.ka);
  deviceContribs.angleTerms.isLinear.setFromVector(hostContribs.angleTerms.isLinear);

  // Bend terms
  deviceContribs.bendTerms.idx1.setFromVector(hostContribs.bendTerms.idx1);
  deviceContribs.bendTerms.idx2.setFromVector(hostContribs.bendTerms.idx2);
  deviceContribs.bendTerms.idx3.setFromVector(hostContribs.bendTerms.idx3);
  deviceContribs.bendTerms.theta0.setFromVector(hostContribs.bendTerms.theta0);
  deviceContribs.bendTerms.restLen1.setFromVector(hostContribs.bendTerms.restLen1);
  deviceContribs.bendTerms.restLen2.setFromVector(hostContribs.bendTerms.restLen2);
  deviceContribs.bendTerms.forceConst1.setFromVector(hostContribs.bendTerms.forceConst1);
  deviceContribs.bendTerms.forceConst2.setFromVector(hostContribs.bendTerms.forceConst2);

  // Oop terms
  deviceContribs.oopTerms.idx1.setFromVector(hostContribs.oopTerms.idx1);
  deviceContribs.oopTerms.idx2.setFromVector(hostContribs.oopTerms.idx2);
  deviceContribs.oopTerms.idx3.setFromVector(hostContribs.oopTerms.idx3);
  deviceContribs.oopTerms.idx4.setFromVector(hostContribs.oopTerms.idx4);
  deviceContribs.oopTerms.koop.setFromVector(hostContribs.oopTerms.koop);

  // Torsion terms
  deviceContribs.torsionTerms.idx1.setFromVector(hostContribs.torsionTerms.idx1);
  deviceContribs.torsionTerms.idx2.setFromVector(hostContribs.torsionTerms.idx2);
  deviceContribs.torsionTerms.idx3.setFromVector(hostContribs.torsionTerms.idx3);
  deviceContribs.torsionTerms.idx4.setFromVector(hostContribs.torsionTerms.idx4);
  deviceContribs.torsionTerms.V1.setFromVector(hostContribs.torsionTerms.V1);
  deviceContribs.torsionTerms.V2.setFromVector(hostContribs.torsionTerms.V2);
  deviceContribs.torsionTerms.V3.setFromVector(hostContribs.torsionTerms.V3);

  // Vdw terms
  deviceContribs.vdwTerms.idx1.setFromVector(hostContribs.vdwTerms.idx1);
  deviceContribs.vdwTerms.idx2.setFromVector(hostContribs.vdwTerms.idx2);
  deviceContribs.vdwTerms.R_ij_star.setFromVector(hostContribs.vdwTerms.R_ij_star);
  deviceContribs.vdwTerms.wellDepth.setFromVector(hostContribs.vdwTerms.wellDepth);

  // Ele terms
  deviceContribs.eleTerms.idx1.setFromVector(hostContribs.eleTerms.idx1);
  deviceContribs.eleTerms.idx2.setFromVector(hostContribs.eleTerms.idx2);
  deviceContribs.eleTerms.chargeTerm.setFromVector(hostContribs.eleTerms.chargeTerm);
  deviceContribs.eleTerms.dielModel.setFromVector(hostContribs.eleTerms.dielModel);
  deviceContribs.eleTerms.is1_4.setFromVector(hostContribs.eleTerms.is1_4);

  deviceContribs.distanceConstraintTerms.idx1.setFromVector(hostContribs.distanceConstraintTerms.idx1);
  deviceContribs.distanceConstraintTerms.idx2.setFromVector(hostContribs.distanceConstraintTerms.idx2);
  deviceContribs.distanceConstraintTerms.minLen.setFromVector(hostContribs.distanceConstraintTerms.minLen);
  deviceContribs.distanceConstraintTerms.maxLen.setFromVector(hostContribs.distanceConstraintTerms.maxLen);
  deviceContribs.distanceConstraintTerms.forceConstant.setFromVector(
    hostContribs.distanceConstraintTerms.forceConstant);

  deviceContribs.positionConstraintTerms.idx.setFromVector(hostContribs.positionConstraintTerms.idx);
  deviceContribs.positionConstraintTerms.refX.setFromVector(hostContribs.positionConstraintTerms.refX);
  deviceContribs.positionConstraintTerms.refY.setFromVector(hostContribs.positionConstraintTerms.refY);
  deviceContribs.positionConstraintTerms.refZ.setFromVector(hostContribs.positionConstraintTerms.refZ);
  deviceContribs.positionConstraintTerms.maxDispl.setFromVector(hostContribs.positionConstraintTerms.maxDispl);
  deviceContribs.positionConstraintTerms.forceConstant.setFromVector(
    hostContribs.positionConstraintTerms.forceConstant);

  deviceContribs.angleConstraintTerms.idx1.setFromVector(hostContribs.angleConstraintTerms.idx1);
  deviceContribs.angleConstraintTerms.idx2.setFromVector(hostContribs.angleConstraintTerms.idx2);
  deviceContribs.angleConstraintTerms.idx3.setFromVector(hostContribs.angleConstraintTerms.idx3);
  deviceContribs.angleConstraintTerms.minAngleDeg.setFromVector(hostContribs.angleConstraintTerms.minAngleDeg);
  deviceContribs.angleConstraintTerms.maxAngleDeg.setFromVector(hostContribs.angleConstraintTerms.maxAngleDeg);
  deviceContribs.angleConstraintTerms.forceConstant.setFromVector(hostContribs.angleConstraintTerms.forceConstant);

  deviceContribs.torsionConstraintTerms.idx1.setFromVector(hostContribs.torsionConstraintTerms.idx1);
  deviceContribs.torsionConstraintTerms.idx2.setFromVector(hostContribs.torsionConstraintTerms.idx2);
  deviceContribs.torsionConstraintTerms.idx3.setFromVector(hostContribs.torsionConstraintTerms.idx3);
  deviceContribs.torsionConstraintTerms.idx4.setFromVector(hostContribs.torsionConstraintTerms.idx4);
  deviceContribs.torsionConstraintTerms.minDihedralDeg.setFromVector(
    hostContribs.torsionConstraintTerms.minDihedralDeg);
  deviceContribs.torsionConstraintTerms.maxDihedralDeg.setFromVector(
    hostContribs.torsionConstraintTerms.maxDihedralDeg);
  deviceContribs.torsionConstraintTerms.forceConstant.setFromVector(hostContribs.torsionConstraintTerms.forceConstant);

  // Indices
  molSystemDevice.indices.atomStarts.setFromVector(molSystemHost.indices.atomStarts);
  molSystemDevice.indices.energyBufferStarts.setFromVector(molSystemHost.indices.energyBufferStarts);
  molSystemDevice.indices.atomIdxToBatchIdx.setFromVector(molSystemHost.indices.atomIdxToBatchIdx);
  molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.setFromVector(
    molSystemHost.indices.energyBufferBlockIdxToBatchIdx);

  molSystemDevice.indices.bondTermStarts.setFromVector(molSystemHost.indices.bondTermStarts);
  molSystemDevice.indices.angleTermStarts.setFromVector(molSystemHost.indices.angleTermStarts);
  molSystemDevice.indices.bendTermStarts.setFromVector(molSystemHost.indices.bendTermStarts);
  molSystemDevice.indices.oopTermStarts.setFromVector(molSystemHost.indices.oopTermStarts);
  molSystemDevice.indices.torsionTermStarts.setFromVector(molSystemHost.indices.torsionTermStarts);
  molSystemDevice.indices.vdwTermStarts.setFromVector(molSystemHost.indices.vdwTermStarts);
  molSystemDevice.indices.eleTermStarts.setFromVector(molSystemHost.indices.eleTermStarts);
  molSystemDevice.indices.distanceConstraintTermStarts.setFromVector(
    molSystemHost.indices.distanceConstraintTermStarts);
  molSystemDevice.indices.positionConstraintTermStarts.setFromVector(
    molSystemHost.indices.positionConstraintTermStarts);
  molSystemDevice.indices.angleConstraintTermStarts.setFromVector(molSystemHost.indices.angleConstraintTermStarts);
  molSystemDevice.indices.torsionConstraintTermStarts.setFromVector(molSystemHost.indices.torsionConstraintTermStarts);
}

void addMoleculeToBatch(const EnergyForceContribsHost& contribs,
                        const std::vector<double>&     positions,
                        BatchedMolecularSystemHost&    molSystem,
                        BatchedForcefieldMetadata*     metadata,
                        const int                      moleculeIdx,
                        const int                      conformerIdx,
                        const ForcefieldModifier&      customization) {
  if (metadata != nullptr || customization) {
    EnergyForceContribsHost contribsCopy = contribs;
    BatchedSystemInfo       systemInfo;
    if (metadata != nullptr) {
      systemInfo = metadata->recordSystem(moleculeIdx, conformerIdx);
    }
    if (customization) {
      customization(systemInfo, positions, contribsCopy);
    }
    addMoleculeToBatch(contribsCopy, positions, molSystem, nullptr, moleculeIdx, conformerIdx, {});
    return;
  }

  const int previousLastAtomIndex = molSystem.indices.atomStarts.back();
  const int numBatches            = molSystem.indices.atomStarts.size() - 1;
  const int newNumAtoms           = positions.size() / 3;
  molSystem.indices.atomStarts.push_back(molSystem.indices.atomStarts.back() + newNumAtoms);
  molSystem.maxNumAtoms = std::max(molSystem.maxNumAtoms, newNumAtoms);

  auto& indexHolder   = molSystem.indices;
  auto& contribHolder = molSystem.contribs;

  // First append positions
  molSystem.positions.insert(molSystem.positions.end(), positions.begin(), positions.end());

  // Next handle indices
  indexHolder.atomIdxToBatchIdx.resize(molSystem.positions.size() / 3, numBatches);
  indexHolder.bondTermStarts.push_back(indexHolder.bondTermStarts.back() + contribs.bondTerms.idx1.size());
  indexHolder.angleTermStarts.push_back(indexHolder.angleTermStarts.back() + contribs.angleTerms.idx1.size());
  indexHolder.bendTermStarts.push_back(indexHolder.bendTermStarts.back() + contribs.bendTerms.idx1.size());
  indexHolder.oopTermStarts.push_back(indexHolder.oopTermStarts.back() + contribs.oopTerms.idx1.size());
  indexHolder.torsionTermStarts.push_back(indexHolder.torsionTermStarts.back() + contribs.torsionTerms.idx1.size());
  indexHolder.vdwTermStarts.push_back(indexHolder.vdwTermStarts.back() + contribs.vdwTerms.idx1.size());
  indexHolder.eleTermStarts.push_back(indexHolder.eleTermStarts.back() + contribs.eleTerms.idx1.size());
  indexHolder.distanceConstraintTermStarts.push_back(indexHolder.distanceConstraintTermStarts.back() +
                                                     contribs.distanceConstraintTerms.idx1.size());
  indexHolder.positionConstraintTermStarts.push_back(indexHolder.positionConstraintTermStarts.back() +
                                                     contribs.positionConstraintTerms.idx.size());
  indexHolder.angleConstraintTermStarts.push_back(indexHolder.angleConstraintTermStarts.back() +
                                                  contribs.angleConstraintTerms.idx1.size());
  indexHolder.torsionConstraintTermStarts.push_back(indexHolder.torsionConstraintTermStarts.back() +
                                                    contribs.torsionConstraintTerms.idx1.size());

  int maxNumContribs = 0;
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.bondTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.angleTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.bendTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.oopTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.torsionTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.vdwTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.eleTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.distanceConstraintTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.positionConstraintTerms.idx.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.angleConstraintTerms.idx1.size());
  maxNumContribs     = std::max<int>(maxNumContribs, contribs.torsionConstraintTerms.idx1.size());

  const int numBlocksNeeded  = (maxNumContribs + 127) / nvMolKit::FFKernelUtils::blockSizeEnergyReduction;
  const int numThreadsNeeded = numBlocksNeeded * nvMolKit::FFKernelUtils::blockSizeEnergyReduction;

  indexHolder.energyBufferStarts.push_back(numThreadsNeeded + indexHolder.energyBufferStarts.back());
  for (int i = 0; i < numBlocksNeeded; i++) {
    indexHolder.energyBufferBlockIdxToBatchIdx.push_back(numBatches);
  }

  // Now append the contribs, updating indices.
  // Bond terms
  for (size_t i = 0; i < contribs.bondTerms.idx1.size(); i++) {
    contribHolder.bondTerms.idx1.push_back(contribs.bondTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.bondTerms.idx2.push_back(contribs.bondTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.bondTerms.r0.push_back(contribs.bondTerms.r0[i]);
    contribHolder.bondTerms.kb.push_back(contribs.bondTerms.kb[i]);
  }
  // Angle terms
  for (size_t i = 0; i < contribs.angleTerms.idx1.size(); i++) {
    contribHolder.angleTerms.idx1.push_back(contribs.angleTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.angleTerms.idx2.push_back(contribs.angleTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.angleTerms.idx3.push_back(contribs.angleTerms.idx3[i] + previousLastAtomIndex);
    contribHolder.angleTerms.theta0.push_back(contribs.angleTerms.theta0[i]);
    contribHolder.angleTerms.ka.push_back(contribs.angleTerms.ka[i]);
    contribHolder.angleTerms.isLinear.push_back(contribs.angleTerms.isLinear[i]);
  }
  // Bend terms
  for (size_t i = 0; i < contribs.bendTerms.idx1.size(); i++) {
    contribHolder.bendTerms.idx1.push_back(contribs.bendTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.bendTerms.idx2.push_back(contribs.bendTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.bendTerms.idx3.push_back(contribs.bendTerms.idx3[i] + previousLastAtomIndex);
    contribHolder.bendTerms.theta0.push_back(contribs.bendTerms.theta0[i]);
    contribHolder.bendTerms.restLen1.push_back(contribs.bendTerms.restLen1[i]);
    contribHolder.bendTerms.restLen2.push_back(contribs.bendTerms.restLen2[i]);
    contribHolder.bendTerms.forceConst1.push_back(contribs.bendTerms.forceConst1[i]);
    contribHolder.bendTerms.forceConst2.push_back(contribs.bendTerms.forceConst2[i]);
  }
  // Oop terms
  for (size_t i = 0; i < contribs.oopTerms.idx1.size(); i++) {
    contribHolder.oopTerms.idx1.push_back(contribs.oopTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.oopTerms.idx2.push_back(contribs.oopTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.oopTerms.idx3.push_back(contribs.oopTerms.idx3[i] + previousLastAtomIndex);
    contribHolder.oopTerms.idx4.push_back(contribs.oopTerms.idx4[i] + previousLastAtomIndex);
    contribHolder.oopTerms.koop.push_back(contribs.oopTerms.koop[i]);
  }
  // Torsion terms
  for (size_t i = 0; i < contribs.torsionTerms.idx1.size(); i++) {
    contribHolder.torsionTerms.idx1.push_back(contribs.torsionTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.torsionTerms.idx2.push_back(contribs.torsionTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.torsionTerms.idx3.push_back(contribs.torsionTerms.idx3[i] + previousLastAtomIndex);
    contribHolder.torsionTerms.idx4.push_back(contribs.torsionTerms.idx4[i] + previousLastAtomIndex);
    contribHolder.torsionTerms.V1.push_back(contribs.torsionTerms.V1[i]);
    contribHolder.torsionTerms.V2.push_back(contribs.torsionTerms.V2[i]);
    contribHolder.torsionTerms.V3.push_back(contribs.torsionTerms.V3[i]);
  }
  // Vdw terms
  for (size_t i = 0; i < contribs.vdwTerms.idx1.size(); i++) {
    contribHolder.vdwTerms.idx1.push_back(contribs.vdwTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.vdwTerms.idx2.push_back(contribs.vdwTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.vdwTerms.R_ij_star.push_back(contribs.vdwTerms.R_ij_star[i]);
    contribHolder.vdwTerms.wellDepth.push_back(contribs.vdwTerms.wellDepth[i]);
  }
  // Ele terms
  for (size_t i = 0; i < contribs.eleTerms.idx1.size(); i++) {
    contribHolder.eleTerms.idx1.push_back(contribs.eleTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.eleTerms.idx2.push_back(contribs.eleTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.eleTerms.chargeTerm.push_back(contribs.eleTerms.chargeTerm[i]);
    contribHolder.eleTerms.dielModel.push_back(contribs.eleTerms.dielModel[i]);
    contribHolder.eleTerms.is1_4.push_back(contribs.eleTerms.is1_4[i]);
  }
  for (size_t i = 0; i < contribs.distanceConstraintTerms.idx1.size(); i++) {
    contribHolder.distanceConstraintTerms.idx1.push_back(contribs.distanceConstraintTerms.idx1[i] +
                                                         previousLastAtomIndex);
    contribHolder.distanceConstraintTerms.idx2.push_back(contribs.distanceConstraintTerms.idx2[i] +
                                                         previousLastAtomIndex);
    contribHolder.distanceConstraintTerms.minLen.push_back(contribs.distanceConstraintTerms.minLen[i]);
    contribHolder.distanceConstraintTerms.maxLen.push_back(contribs.distanceConstraintTerms.maxLen[i]);
    contribHolder.distanceConstraintTerms.forceConstant.push_back(contribs.distanceConstraintTerms.forceConstant[i]);
  }
  for (size_t i = 0; i < contribs.positionConstraintTerms.idx.size(); i++) {
    contribHolder.positionConstraintTerms.idx.push_back(contribs.positionConstraintTerms.idx[i] +
                                                        previousLastAtomIndex);
    contribHolder.positionConstraintTerms.refX.push_back(contribs.positionConstraintTerms.refX[i]);
    contribHolder.positionConstraintTerms.refY.push_back(contribs.positionConstraintTerms.refY[i]);
    contribHolder.positionConstraintTerms.refZ.push_back(contribs.positionConstraintTerms.refZ[i]);
    contribHolder.positionConstraintTerms.maxDispl.push_back(contribs.positionConstraintTerms.maxDispl[i]);
    contribHolder.positionConstraintTerms.forceConstant.push_back(contribs.positionConstraintTerms.forceConstant[i]);
  }
  for (size_t i = 0; i < contribs.angleConstraintTerms.idx1.size(); i++) {
    contribHolder.angleConstraintTerms.idx1.push_back(contribs.angleConstraintTerms.idx1[i] + previousLastAtomIndex);
    contribHolder.angleConstraintTerms.idx2.push_back(contribs.angleConstraintTerms.idx2[i] + previousLastAtomIndex);
    contribHolder.angleConstraintTerms.idx3.push_back(contribs.angleConstraintTerms.idx3[i] + previousLastAtomIndex);
    contribHolder.angleConstraintTerms.minAngleDeg.push_back(contribs.angleConstraintTerms.minAngleDeg[i]);
    contribHolder.angleConstraintTerms.maxAngleDeg.push_back(contribs.angleConstraintTerms.maxAngleDeg[i]);
    contribHolder.angleConstraintTerms.forceConstant.push_back(contribs.angleConstraintTerms.forceConstant[i]);
  }
  for (size_t i = 0; i < contribs.torsionConstraintTerms.idx1.size(); i++) {
    contribHolder.torsionConstraintTerms.idx1.push_back(contribs.torsionConstraintTerms.idx1[i] +
                                                        previousLastAtomIndex);
    contribHolder.torsionConstraintTerms.idx2.push_back(contribs.torsionConstraintTerms.idx2[i] +
                                                        previousLastAtomIndex);
    contribHolder.torsionConstraintTerms.idx3.push_back(contribs.torsionConstraintTerms.idx3[i] +
                                                        previousLastAtomIndex);
    contribHolder.torsionConstraintTerms.idx4.push_back(contribs.torsionConstraintTerms.idx4[i] +
                                                        previousLastAtomIndex);
    contribHolder.torsionConstraintTerms.minDihedralDeg.push_back(contribs.torsionConstraintTerms.minDihedralDeg[i]);
    contribHolder.torsionConstraintTerms.maxDihedralDeg.push_back(contribs.torsionConstraintTerms.maxDihedralDeg[i]);
    contribHolder.torsionConstraintTerms.forceConstant.push_back(contribs.torsionConstraintTerms.forceConstant[i]);
  }
}

void allocateIntermediateBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                 BatchedMolecularDeviceBuffers&    molSystemDevice) {
  nvMolKit::FFKernelUtils::allocateIntermediateBuffers(molSystemHost,
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

  cudaError_t err = cudaSuccess;
  if (contribs.bondTerms.idx1.size() > 0) {
    err = launchBondStretchEnergyKernel(contribs.bondTerms.idx1.size(),
                                        contribs.bondTerms.idx1.data(),
                                        contribs.bondTerms.idx2.data(),
                                        contribs.bondTerms.r0.data(),
                                        contribs.bondTerms.kb.data(),
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
                                      contribs.angleTerms.ka.data(),
                                      contribs.angleTerms.isLinear.data(),
                                      positions,
                                      molSystemDevice.energyBuffer.data(),
                                      molSystemDevice.indices.energyBufferStarts.data(),
                                      molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                      molSystemDevice.indices.angleTermStarts.data(),
                                      stream);
  }
  if (err == cudaSuccess && contribs.bendTerms.idx1.size() > 0) {
    err = launchBendStretchEnergyKernel(contribs.bendTerms.idx1.size(),
                                        contribs.bendTerms.idx1.data(),
                                        contribs.bendTerms.idx2.data(),
                                        contribs.bendTerms.idx3.data(),
                                        contribs.bendTerms.theta0.data(),
                                        contribs.bendTerms.restLen1.data(),
                                        contribs.bendTerms.restLen2.data(),
                                        contribs.bendTerms.forceConst1.data(),
                                        contribs.bendTerms.forceConst2.data(),
                                        positions,
                                        molSystemDevice.energyBuffer.data(),
                                        molSystemDevice.indices.energyBufferStarts.data(),
                                        molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                        molSystemDevice.indices.bendTermStarts.data(),
                                        stream);
  }
  if (err == cudaSuccess && contribs.oopTerms.idx1.size() > 0) {
    err = launchOopBendEnergyKernel(contribs.oopTerms.idx1.size(),
                                    contribs.oopTerms.idx1.data(),
                                    contribs.oopTerms.idx2.data(),
                                    contribs.oopTerms.idx3.data(),
                                    contribs.oopTerms.idx4.data(),
                                    contribs.oopTerms.koop.data(),
                                    positions,
                                    molSystemDevice.energyBuffer.data(),
                                    molSystemDevice.indices.energyBufferStarts.data(),
                                    molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                    molSystemDevice.indices.oopTermStarts.data(),
                                    stream);
  }
  if (err == cudaSuccess && contribs.torsionTerms.idx1.size() > 0) {
    err = launchTorsionEnergyKernel(contribs.torsionTerms.idx1.size(),
                                    contribs.torsionTerms.idx1.data(),
                                    contribs.torsionTerms.idx2.data(),
                                    contribs.torsionTerms.idx3.data(),
                                    contribs.torsionTerms.idx4.data(),
                                    contribs.torsionTerms.V1.data(),
                                    contribs.torsionTerms.V2.data(),
                                    contribs.torsionTerms.V3.data(),
                                    positions,
                                    molSystemDevice.energyBuffer.data(),
                                    molSystemDevice.indices.energyBufferStarts.data(),
                                    molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                    molSystemDevice.indices.torsionTermStarts.data(),
                                    stream);
  }
  if (err == cudaSuccess && contribs.vdwTerms.idx1.size() > 0) {
    err = launchVdwEnergyKernel(contribs.vdwTerms.idx1.size(),
                                contribs.vdwTerms.idx1.data(),
                                contribs.vdwTerms.idx2.data(),
                                contribs.vdwTerms.R_ij_star.data(),
                                contribs.vdwTerms.wellDepth.data(),
                                positions,
                                molSystemDevice.energyBuffer.data(),
                                molSystemDevice.indices.energyBufferStarts.data(),
                                molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                molSystemDevice.indices.vdwTermStarts.data(),
                                stream);
  }
  if (err == cudaSuccess && contribs.eleTerms.idx1.size() > 0) {
    err = launchEleEnergyKernel(contribs.eleTerms.idx1.size(),
                                contribs.eleTerms.idx1.data(),
                                contribs.eleTerms.idx2.data(),
                                contribs.eleTerms.chargeTerm.data(),
                                contribs.eleTerms.dielModel.data(),
                                contribs.eleTerms.is1_4.data(),
                                positions,
                                molSystemDevice.energyBuffer.data(),
                                molSystemDevice.indices.energyBufferStarts.data(),
                                molSystemDevice.indices.atomIdxToBatchIdx.data(),
                                molSystemDevice.indices.eleTermStarts.data(),
                                stream);
  }
  if (err == cudaSuccess && contribs.distanceConstraintTerms.idx1.size() > 0) {
    err = launchDistanceConstraintEnergyKernel(contribs.distanceConstraintTerms.idx1.size(),
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
    err = launchPositionConstraintEnergyKernel(contribs.positionConstraintTerms.idx.size(),
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
    err = launchAngleConstraintEnergyKernel(contribs.angleConstraintTerms.idx1.size(),
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
    err = launchTorsionConstraintEnergyKernel(contribs.torsionConstraintTerms.idx1.size(),
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
    return launchReduceEnergiesKernel(molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.size(),
                                      molSystemDevice.energyBuffer.data(),
                                      molSystemDevice.indices.energyBufferBlockIdxToBatchIdx.data(),
                                      energyOuts,
                                      activeSystemMask,
                                      stream);
  }
  return err;
}

// TODO: More sophisticated error handling for energy and gradient.
cudaError_t computeEnergy(BatchedMolecularDeviceBuffers& molSystemDevice, const double* coords, cudaStream_t stream) {
  const double* positions = coords != nullptr ? coords : molSystemDevice.positions.data();
  return computeEnergy(molSystemDevice, molSystemDevice.energyOuts.data(), positions, nullptr, stream);
}

cudaError_t computeGradients(BatchedMolecularDeviceBuffers& molSystemDevice,
                             const double*                  positions,
                             double*                        grad,
                             const uint8_t*                 activeSystemMask,
                             cudaStream_t                   stream) {
  const auto& contribs = molSystemDevice.contribs;

  cudaError_t err = cudaSuccess;
  if (contribs.bondTerms.idx1.size() > 0) {
    err = launchBondStretchGradientKernel(contribs.bondTerms.idx1.size(),
                                          contribs.bondTerms.idx1.data(),
                                          contribs.bondTerms.idx2.data(),
                                          contribs.bondTerms.r0.data(),
                                          contribs.bondTerms.kb.data(),
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
                                        contribs.angleTerms.ka.data(),
                                        contribs.angleTerms.isLinear.data(),
                                        positions,
                                        grad,
                                        stream);
  }
  if (err == cudaSuccess && contribs.bendTerms.idx1.size() > 0) {
    err = launchBendStretchGradientKernel(contribs.bendTerms.idx1.size(),
                                          contribs.bendTerms.idx1.data(),
                                          contribs.bendTerms.idx2.data(),
                                          contribs.bendTerms.idx3.data(),
                                          contribs.bendTerms.theta0.data(),
                                          contribs.bendTerms.restLen1.data(),
                                          contribs.bendTerms.restLen2.data(),
                                          contribs.bendTerms.forceConst1.data(),
                                          contribs.bendTerms.forceConst2.data(),
                                          positions,
                                          grad,
                                          stream);
  }
  if (err == cudaSuccess && contribs.oopTerms.idx1.size() > 0) {
    err = launchOopBendGradientKernel(contribs.oopTerms.idx1.size(),
                                      contribs.oopTerms.idx1.data(),
                                      contribs.oopTerms.idx2.data(),
                                      contribs.oopTerms.idx3.data(),
                                      contribs.oopTerms.idx4.data(),
                                      contribs.oopTerms.koop.data(),
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
                                      contribs.torsionTerms.V1.data(),
                                      contribs.torsionTerms.V2.data(),
                                      contribs.torsionTerms.V3.data(),
                                      positions,
                                      grad,
                                      stream);
  }
  if (err == cudaSuccess && contribs.vdwTerms.idx1.size() > 0) {
    err = launchVdwGradientKernel(contribs.vdwTerms.idx1.size(),
                                  contribs.vdwTerms.idx1.data(),
                                  contribs.vdwTerms.idx2.data(),
                                  contribs.vdwTerms.R_ij_star.data(),
                                  contribs.vdwTerms.wellDepth.data(),
                                  positions,
                                  grad,
                                  stream);
  }
  if (err == cudaSuccess && contribs.eleTerms.idx1.size() > 0) {
    err = launchEleGradientKernel(contribs.eleTerms.idx1.size(),
                                  contribs.eleTerms.idx1.data(),
                                  contribs.eleTerms.idx2.data(),
                                  contribs.eleTerms.chargeTerm.data(),
                                  contribs.eleTerms.dielModel.data(),
                                  contribs.eleTerms.is1_4.data(),
                                  positions,
                                  grad,
                                  stream);
  }
  if (err == cudaSuccess && contribs.distanceConstraintTerms.idx1.size() > 0) {
    err = launchDistanceConstraintGradientKernel(contribs.distanceConstraintTerms.idx1.size(),
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
    err = launchPositionConstraintGradientKernel(contribs.positionConstraintTerms.idx.size(),
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
    err = launchAngleConstraintGradientKernel(contribs.angleConstraintTerms.idx1.size(),
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
    err = launchTorsionConstraintGradientKernel(contribs.torsionConstraintTerms.idx1.size(),
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
    // TODO: Thread activeSystemMask through the MMFF term kernels so inactive
    // systems can be skipped before gradient accumulation; energy kernels
    // should follow the same early-filter path instead of relying on late
    // masking.
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
  const auto pointers       = toPointerStruct(molSystemDevice.contribs);
  const auto indices        = toPointerStruct(molSystemDevice.indices);
  const bool hasConstraints = batchHasConstraints(molSystemDevice.contribs);
  return launchBlockPerMolEnergyKernel(molSystemDevice.indices.atomStarts.size() - 1,
                                       pointers,
                                       indices,
                                       coords != nullptr ? coords : molSystemDevice.positions.data(),
                                       molSystemDevice.energyOuts.data(),
                                       hasConstraints,
                                       stream);
}

cudaError_t computeGradBlockPerMol(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream) {
  const auto pointers       = toPointerStruct(molSystemDevice.contribs);
  const auto indices        = toPointerStruct(molSystemDevice.indices);
  const bool hasConstraints = batchHasConstraints(molSystemDevice.contribs);
  return launchBlockPerMolGradKernel(molSystemDevice.indices.atomStarts.size() - 1,
                                     pointers,
                                     indices,
                                     molSystemDevice.positions.data(),
                                     molSystemDevice.grad.data(),
                                     hasConstraints,
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
}  // namespace MMFF
}  // namespace nvMolKit
