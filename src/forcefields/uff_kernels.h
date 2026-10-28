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

#ifndef NVMOLKIT_UFF_KERNELS_H
#define NVMOLKIT_UFF_KERNELS_H

#include <cstdint>

#include "src/forcefields/mmff_kernels.h"

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
                                          cudaStream_t  stream = 0);

cudaError_t launchBondStretchGradientKernel(int           numBonds,
                                            const int*    idx1,
                                            const int*    idx2,
                                            const double* restLen,
                                            const double* forceConstant,
                                            const double* pos,
                                            double*       grad,
                                            cudaStream_t  stream = 0);

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
                                        cudaStream_t   stream = 0);

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
                                          cudaStream_t   stream = 0);

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
                                      cudaStream_t   stream = 0);

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
                                        cudaStream_t   stream = 0);

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
                                        cudaStream_t  stream = 0);

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
                                          cudaStream_t  stream = 0);

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
                                  cudaStream_t  stream = 0);

cudaError_t launchVdwGradientKernel(int           numVdws,
                                    const int*    idx1,
                                    const int*    idx2,
                                    const double* x_ij,
                                    const double* wellDepth,
                                    const double* threshold,
                                    const double* pos,
                                    double*       grad,
                                    cudaStream_t  stream = 0);

template <typename Scalar> struct BondStretchTermsDevicePtrT {
  int*    idx1          = nullptr;
  int*    idx2          = nullptr;
  Scalar* restLen       = nullptr;
  Scalar* forceConstant = nullptr;
};

template <typename Scalar> struct AngleBendTermsDevicePtrT {
  int*          idx1          = nullptr;
  int*          idx2          = nullptr;
  int*          idx3          = nullptr;
  Scalar*       theta0        = nullptr;
  Scalar*       forceConstant = nullptr;
  std::uint8_t* order         = nullptr;
  Scalar*       C0            = nullptr;
  Scalar*       C1            = nullptr;
  Scalar*       C2            = nullptr;
};

template <typename Scalar> struct TorsionTermsDevicePtrT {
  int*          idx1          = nullptr;
  int*          idx2          = nullptr;
  int*          idx3          = nullptr;
  int*          idx4          = nullptr;
  Scalar*       forceConstant = nullptr;
  std::uint8_t* order         = nullptr;
  Scalar*       cosTerm       = nullptr;
};

template <typename Scalar> struct InversionTermsDevicePtrT {
  int*    idx1          = nullptr;
  int*    idx2          = nullptr;
  int*    idx3          = nullptr;
  int*    idx4          = nullptr;
  Scalar* forceConstant = nullptr;
  Scalar* C0            = nullptr;
  Scalar* C1            = nullptr;
  Scalar* C2            = nullptr;
};

template <typename Scalar> struct VdwTermsDevicePtrT {
  int*    idx1      = nullptr;
  int*    idx2      = nullptr;
  Scalar* x_ij      = nullptr;
  Scalar* wellDepth = nullptr;
  Scalar* threshold = nullptr;
};

template <typename Scalar> struct EnergyForceContribsDevicePtrT {
  BondStretchTermsDevicePtrT<Scalar>              bondTerms;
  AngleBendTermsDevicePtrT<Scalar>                angleTerms;
  TorsionTermsDevicePtrT<Scalar>                  torsionTerms;
  InversionTermsDevicePtrT<Scalar>                inversionTerms;
  VdwTermsDevicePtrT<Scalar>                      vdwTerms;
  MMFF::DistanceConstraintTermsDevicePtrT<Scalar> distanceConstraintTerms;
  MMFF::PositionConstraintTermsDevicePtrT<Scalar> positionConstraintTerms;
  MMFF::AngleConstraintTermsDevicePtrT<Scalar>    angleConstraintTerms;
  MMFF::TorsionConstraintTermsDevicePtrT<Scalar>  torsionConstraintTerms;
};

using BondStretchTermsDevicePtr          = BondStretchTermsDevicePtrT<double>;
using AngleBendTermsDevicePtr            = AngleBendTermsDevicePtrT<double>;
using TorsionTermsDevicePtr              = TorsionTermsDevicePtrT<double>;
using InversionTermsDevicePtr            = InversionTermsDevicePtrT<double>;
using VdwTermsDevicePtr                  = VdwTermsDevicePtrT<double>;
using EnergyForceContribsDevicePtr       = EnergyForceContribsDevicePtrT<double>;
using EnergyForceContribsDevicePtrSingle = EnergyForceContribsDevicePtrT<float>;

struct BatchedIndicesDevicePtr {
  int* atomStarts                   = nullptr;
  int* bondTermStarts               = nullptr;
  int* angleTermStarts              = nullptr;
  int* torsionTermStarts            = nullptr;
  int* inversionTermStarts          = nullptr;
  int* vdwTermStarts                = nullptr;
  int* distanceConstraintTermStarts = nullptr;
  int* positionConstraintTermStarts = nullptr;
  int* angleConstraintTermStarts    = nullptr;
  int* torsionConstraintTermStarts  = nullptr;
};

cudaError_t launchBlockPerMolEnergyKernel(int                                 numMols,
                                          const EnergyForceContribsDevicePtr& terms,
                                          const BatchedIndicesDevicePtr&      sytemIndices,
                                          const double*                       coords,
                                          double*                             energies,
                                          bool                                hasConstraints,
                                          cudaStream_t                        stream           = nullptr,
                                          const uint8_t*                      activeSystemMask = nullptr);

cudaError_t launchBlockPerMolGradKernel(int                                 numMols,
                                        const EnergyForceContribsDevicePtr& terms,
                                        const BatchedIndicesDevicePtr&      sytemIndices,
                                        const double*                       coords,
                                        double*                             grad,
                                        bool                                hasConstraints,
                                        cudaStream_t                        stream           = nullptr,
                                        const uint8_t*                      activeSystemMask = nullptr);

cudaError_t launchBlockPerMolEnergyKernel(int                                       numMols,
                                          const EnergyForceContribsDevicePtrSingle& terms,
                                          const BatchedIndicesDevicePtr&            sytemIndices,
                                          const float*                              coords,
                                          float*                                    energies,
                                          bool                                      hasConstraints,
                                          cudaStream_t                              stream           = nullptr,
                                          const uint8_t*                            activeSystemMask = nullptr);

cudaError_t launchBlockPerMolGradKernel(int                                       numMols,
                                        const EnergyForceContribsDevicePtrSingle& terms,
                                        const BatchedIndicesDevicePtr&            sytemIndices,
                                        const float*                              coords,
                                        float*                                    grad,
                                        bool                                      hasConstraints,
                                        cudaStream_t                              stream           = nullptr,
                                        const uint8_t*                            activeSystemMask = nullptr);

}  // namespace UFF
}  // namespace nvMolKit

#endif  // NVMOLKIT_UFF_KERNELS_H
