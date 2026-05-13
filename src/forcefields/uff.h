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

#ifndef NVMOLKIT_UFF_H
#define NVMOLKIT_UFF_H

#include <cstdint>
#include <functional>
#include <vector>

#include "batched_forcefield.h"
#include "device_vector.h"
// TODO: Constraint types and kernels (DistanceConstraintTerms, launchReduceEnergiesKernel, etc.)
// should be extracted from MMFF into shared forcefield-generic headers so UFF doesn't depend on MMFF.
#include "mmff.h"
#include "uff_kernels.h"

namespace nvMolKit {
namespace UFF {

struct BondStretchTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<double> restLen;
  std::vector<double> forceConstant;
};

struct AngleBendTerms {
  std::vector<int>          idx1;
  std::vector<int>          idx2;
  std::vector<int>          idx3;
  std::vector<double>       theta0;
  std::vector<double>       forceConstant;
  std::vector<std::uint8_t> order;
  std::vector<double>       C0;
  std::vector<double>       C1;
  std::vector<double>       C2;
};

struct TorsionTerms {
  std::vector<int>          idx1;
  std::vector<int>          idx2;
  std::vector<int>          idx3;
  std::vector<int>          idx4;
  std::vector<double>       forceConstant;
  std::vector<std::uint8_t> order;
  std::vector<double>       cosTerm;
};

struct InversionTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<int>    idx3;
  std::vector<int>    idx4;
  std::vector<double> forceConstant;
  std::vector<double> C0;
  std::vector<double> C1;
  std::vector<double> C2;
};

struct VdwTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<double> x_ij;
  std::vector<double> wellDepth;
  std::vector<double> threshold;
};

using DistanceConstraintTerms = MMFF::DistanceConstraintTerms;
using PositionConstraintTerms = MMFF::PositionConstraintTerms;
using AngleConstraintTerms    = MMFF::AngleConstraintTerms;
using TorsionConstraintTerms  = MMFF::TorsionConstraintTerms;

struct EnergyForceContribsHost {
  BondStretchTerms        bondTerms;
  AngleBendTerms          angleTerms;
  TorsionTerms            torsionTerms;
  InversionTerms          inversionTerms;
  VdwTerms                vdwTerms;
  DistanceConstraintTerms distanceConstraintTerms;
  PositionConstraintTerms positionConstraintTerms;
  AngleConstraintTerms    angleConstraintTerms;
  TorsionConstraintTerms  torsionConstraintTerms;
};

using HostCustomization =
  std::function<void(const BatchedSystemInfo&, const std::vector<double>&, EnergyForceContribsHost&)>;

struct BatchedIndicesHost {
  std::vector<int> atomStarts         = {0};
  std::vector<int> energyBufferStarts = {0};
  std::vector<int> atomIdxToBatchIdx;
  std::vector<int> energyBufferBlockIdxToBatchIdx;

  std::vector<int> bondTermStarts               = {0};
  std::vector<int> angleTermStarts              = {0};
  std::vector<int> torsionTermStarts            = {0};
  std::vector<int> inversionTermStarts          = {0};
  std::vector<int> vdwTermStarts                = {0};
  std::vector<int> distanceConstraintTermStarts = {0};
  std::vector<int> positionConstraintTermStarts = {0};
  std::vector<int> angleConstraintTermStarts    = {0};
  std::vector<int> torsionConstraintTermStarts  = {0};
};

struct BatchedMolecularSystemHost {
  EnergyForceContribsHost contribs;
  BatchedIndicesHost      indices;
  std::vector<double>     positions;
  int                     maxNumAtoms = 0;
};

struct BondStretchTermsDevice {
  AsyncDeviceVector<int>    idx1;
  AsyncDeviceVector<int>    idx2;
  AsyncDeviceVector<double> restLen;
  AsyncDeviceVector<double> forceConstant;
};

struct AngleBendTermsDevice {
  AsyncDeviceVector<int>          idx1;
  AsyncDeviceVector<int>          idx2;
  AsyncDeviceVector<int>          idx3;
  AsyncDeviceVector<double>       theta0;
  AsyncDeviceVector<double>       forceConstant;
  AsyncDeviceVector<std::uint8_t> order;
  AsyncDeviceVector<double>       C0;
  AsyncDeviceVector<double>       C1;
  AsyncDeviceVector<double>       C2;
};

struct TorsionTermsDevice {
  AsyncDeviceVector<int>          idx1;
  AsyncDeviceVector<int>          idx2;
  AsyncDeviceVector<int>          idx3;
  AsyncDeviceVector<int>          idx4;
  AsyncDeviceVector<double>       forceConstant;
  AsyncDeviceVector<std::uint8_t> order;
  AsyncDeviceVector<double>       cosTerm;
};

struct InversionTermsDevice {
  AsyncDeviceVector<int>    idx1;
  AsyncDeviceVector<int>    idx2;
  AsyncDeviceVector<int>    idx3;
  AsyncDeviceVector<int>    idx4;
  AsyncDeviceVector<double> forceConstant;
  AsyncDeviceVector<double> C0;
  AsyncDeviceVector<double> C1;
  AsyncDeviceVector<double> C2;
};

struct VdwTermsDevice {
  AsyncDeviceVector<int>    idx1;
  AsyncDeviceVector<int>    idx2;
  AsyncDeviceVector<double> x_ij;
  AsyncDeviceVector<double> wellDepth;
  AsyncDeviceVector<double> threshold;
};

using DistanceConstraintTermsDevice = MMFF::DistanceConstraintTermsDevice;
using PositionConstraintTermsDevice = MMFF::PositionConstraintTermsDevice;
using AngleConstraintTermsDevice    = MMFF::AngleConstraintTermsDevice;
using TorsionConstraintTermsDevice  = MMFF::TorsionConstraintTermsDevice;

struct EnergyForceContribsDevice {
  BondStretchTermsDevice        bondTerms;
  AngleBendTermsDevice          angleTerms;
  TorsionTermsDevice            torsionTerms;
  InversionTermsDevice          inversionTerms;
  VdwTermsDevice                vdwTerms;
  DistanceConstraintTermsDevice distanceConstraintTerms;
  PositionConstraintTermsDevice positionConstraintTerms;
  AngleConstraintTermsDevice    angleConstraintTerms;
  TorsionConstraintTermsDevice  torsionConstraintTerms;
};

struct BatchedIndicesDevice {
  AsyncDeviceVector<int> atomStarts;
  AsyncDeviceVector<int> atomIdxToBatchIdx;
  AsyncDeviceVector<int> energyBufferStarts;
  AsyncDeviceVector<int> energyBufferBlockIdxToBatchIdx;

  AsyncDeviceVector<int> bondTermStarts;
  AsyncDeviceVector<int> angleTermStarts;
  AsyncDeviceVector<int> torsionTermStarts;
  AsyncDeviceVector<int> inversionTermStarts;
  AsyncDeviceVector<int> vdwTermStarts;
  AsyncDeviceVector<int> distanceConstraintTermStarts;
  AsyncDeviceVector<int> positionConstraintTermStarts;
  AsyncDeviceVector<int> angleConstraintTermStarts;
  AsyncDeviceVector<int> torsionConstraintTermStarts;
};

struct BatchedMolecularDeviceBuffers {
  EnergyForceContribsDevice contribs;
  BatchedIndicesDevice      indices;
  AsyncDeviceVector<double> positions;
  AsyncDeviceVector<double> grad;
  AsyncDeviceVector<double> energyBuffer;
  AsyncDeviceVector<double> energyOuts;
};

void addMoleculeToBatch(const EnergyForceContribsHost& contribs,
                        const std::vector<double>&     positions,
                        BatchedMolecularSystemHost&    molSystem);

void addMoleculeToBatch(const EnergyForceContribsHost& contribs,
                        const std::vector<double>&     positions,
                        BatchedMolecularSystemHost&    molSystem,
                        BatchedForcefieldMetadata&     metadata,
                        int                            moleculeIdx,
                        int                            conformerIdx,
                        const HostCustomization&       customization = {});

void setStreams(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream);

void sendContribsAndIndicesToDevice(const BatchedMolecularSystemHost& molSystemHost,
                                    BatchedMolecularDeviceBuffers&    molSystemDevice);

void allocateIntermediateBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                 BatchedMolecularDeviceBuffers&    molSystemDevice);

cudaError_t computeEnergy(BatchedMolecularDeviceBuffers& molSystemDevice,
                          double*                        energyOuts,
                          const double*                  positions,
                          const uint8_t*                 activeSystemMask = nullptr,
                          cudaStream_t                   stream           = nullptr);

cudaError_t computeEnergy(BatchedMolecularDeviceBuffers& molSystemDevice,
                          const double*                  coords = nullptr,
                          cudaStream_t                   stream = nullptr);

cudaError_t computeEnergyBlockPerMol(BatchedMolecularDeviceBuffers& molSystemDevice,
                                     const double*                  coords = nullptr,
                                     cudaStream_t                   stream = nullptr);

cudaError_t computeGradients(BatchedMolecularDeviceBuffers& molSystemDevice,
                             const double*                  positions,
                             double*                        grad,
                             const uint8_t*                 activeSystemMask = nullptr,
                             cudaStream_t                   stream           = nullptr);

cudaError_t computeGradients(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream = nullptr);

cudaError_t computeGradBlockPerMol(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream = nullptr);

EnergyForceContribsDevicePtr toEnergyForceContribsDevicePtr(const BatchedMolecularDeviceBuffers& molSystemDevice);

BatchedIndicesDevicePtr toBatchedIndicesDevicePtr(const BatchedMolecularDeviceBuffers& molSystemDevice);

//! Returns true if any molecule in the batch contributes a distance, position, angle, or torsion
//! constraint term. Used by per-molecule kernels to dispatch to a specialization that compiles out
//! the constraint loops, recovering register pressure when no constraints are active.
bool batchHasConstraints(const EnergyForceContribsDevice& contribs);

}  // namespace UFF
}  // namespace nvMolKit

#endif  // NVMOLKIT_UFF_H
