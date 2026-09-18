// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_MMFF_KERNELS_H
#define NVMOLKIT_MMFF_KERNELS_H

#include <cstdint>

namespace nvMolKit {
namespace MMFF {

cudaError_t launchBondStretchEnergyKernel(int           numBonds,
                                          const int*    idx1,
                                          const int*    idx2,
                                          const double* r0,
                                          const double* kb,
                                          const double* pos,
                                          double*       energyBuffer,
                                          const int*    energyBufferStarts,
                                          const int*    atomBatchMap,
                                          const int*    termBatchStarts,
                                          cudaStream_t  stream = 0);

cudaError_t launchBondStretchGradientKernel(int           numBonds,
                                            const int*    idx1,
                                            const int*    idx2,
                                            const double* r0,
                                            const double* kb,
                                            const double* pos,
                                            double*       grad,
                                            cudaStream_t  stream = 0);

cudaError_t launchAngleBendEnergyKernel(int            numAngles,
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
                                        cudaStream_t   stream = 0);

cudaError_t launchAngleBendGradientKernel(int            numAngles,
                                          const int*     idx1,
                                          const int*     idx2,
                                          const int*     idx3,
                                          const double*  theta0,
                                          const double*  ka,
                                          const uint8_t* isLinear,
                                          const double*  pos,
                                          double*        grad,
                                          cudaStream_t   stream = 0);

cudaError_t launchBendStretchEnergyKernel(int           numAngles,
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
                                          cudaStream_t  stream = 0);

cudaError_t launchBendStretchGradientKernel(int           numAngles,
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
                                            cudaStream_t  stream = 0);

cudaError_t launchOopBendEnergyKernel(int           numOopBends,
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
                                      cudaStream_t  stream = 0);

cudaError_t launchOopBendGradientKernel(int           numOopBends,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const int*    idx3,
                                        const int*    idx4,
                                        const double* koop,
                                        const double* pos,
                                        double*       grad,
                                        cudaStream_t  stream = 0);

cudaError_t launchTorsionEnergyKernel(int           numTorsions,
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
                                      cudaStream_t  stream = 0);

cudaError_t launchTorsionGradientKernel(int           numTorsions,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const int*    idx3,
                                        const int*    idx4,
                                        const float*  V1,
                                        const float*  V2,
                                        const float*  V3,
                                        const double* pos,
                                        double*       grad,
                                        cudaStream_t  stream = 0);

cudaError_t launchVdwEnergyKernel(int           numVdws,
                                  const int*    idx1,
                                  const int*    idx2,
                                  const double* R_ij_star,
                                  const double* wellDepth,
                                  const double* pos,
                                  double*       energyBuffer,
                                  const int*    energyBufferStarts,
                                  const int*    atomBatchMap,
                                  const int*    termBatchStarts,
                                  cudaStream_t  stream = 0);

cudaError_t launchVdwGradientKernel(int           numVdws,
                                    const int*    idx1,
                                    const int*    idx2,
                                    const double* R_ij_star,
                                    const double* wellDepth,
                                    const double* pos,
                                    double*       grad,
                                    cudaStream_t  stream = 0);

cudaError_t launchEleEnergyKernel(int            numEles,
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
                                  cudaStream_t   stream = 0);

cudaError_t launchEleGradientKernel(int            numEles,
                                    const int*     idx1,
                                    const int*     idx2,
                                    const double*  chargeTerm,
                                    const uint8_t* dielModel,
                                    const uint8_t* is1_4,
                                    const double*  pos,
                                    double*        grad,
                                    cudaStream_t   stream = 0);

cudaError_t launchDistanceConstraintEnergyKernel(int           numConstraints,
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
                                                 cudaStream_t  stream = 0);

cudaError_t launchDistanceConstraintGradientKernel(int           numConstraints,
                                                   const int*    idx1,
                                                   const int*    idx2,
                                                   const double* minLen,
                                                   const double* maxLen,
                                                   const double* forceConstant,
                                                   const double* pos,
                                                   double*       grad,
                                                   cudaStream_t  stream = 0);

cudaError_t launchPositionConstraintEnergyKernel(int           numConstraints,
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
                                                 cudaStream_t  stream = 0);

cudaError_t launchPositionConstraintGradientKernel(int           numConstraints,
                                                   const int*    idx,
                                                   const double* refX,
                                                   const double* refY,
                                                   const double* refZ,
                                                   const double* maxDispl,
                                                   const double* forceConstant,
                                                   const double* pos,
                                                   double*       grad,
                                                   cudaStream_t  stream = 0);

cudaError_t launchAngleConstraintEnergyKernel(int           numConstraints,
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
                                              cudaStream_t  stream = 0);

cudaError_t launchAngleConstraintGradientKernel(int           numConstraints,
                                                const int*    idx1,
                                                const int*    idx2,
                                                const int*    idx3,
                                                const double* minAngleDeg,
                                                const double* maxAngleDeg,
                                                const double* forceConstant,
                                                const double* pos,
                                                double*       grad,
                                                cudaStream_t  stream = 0);

cudaError_t launchTorsionConstraintEnergyKernel(int           numConstraints,
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
                                                cudaStream_t  stream = 0);

cudaError_t launchTorsionConstraintGradientKernel(int           numConstraints,
                                                  const int*    idx1,
                                                  const int*    idx2,
                                                  const int*    idx3,
                                                  const int*    idx4,
                                                  const double* minDihedralDeg,
                                                  const double* maxDihedralDeg,
                                                  const double* forceConstant,
                                                  const double* pos,
                                                  double*       grad,
                                                  cudaStream_t  stream = 0);

//! Reduce the energy buffer to the output energies.
//!
//! Energies written to energyBuffer are accumulated in outs, with one term in outs corresponding to each molecule in
//! the batch.
//!
//! \param numBlocks Number of blocks
//! \param energyBuffer Energy terms, guaranteed each block is only assigned to one molecule.
//! \param energyBufferBlockIdxToBatchIdx Matching of blocks to output indices.
//! \param outs Output energies
//! \param stream
//! \return
cudaError_t launchReduceEnergiesKernel(int            numBlocks,
                                       const double*  energyBuffer,
                                       const int*     energyBufferBlockIdxToBatchIdx,
                                       double*        outs,
                                       const uint8_t* activeThisStage = nullptr,
                                       cudaStream_t   stream          = 0);

//! Pointer versions of contrib structs for kernel launches
template <typename Scalar> struct BondStretchContribTermsDevicePtrT {
  int*    idx1 = nullptr;
  int*    idx2 = nullptr;
  Scalar* r0   = nullptr;
  Scalar* kb   = nullptr;
};

template <typename Scalar> struct AngleBendTermsDevicePtrT {
  int*          idx1     = nullptr;
  int*          idx2     = nullptr;
  int*          idx3     = nullptr;
  Scalar*       theta0   = nullptr;
  Scalar*       ka       = nullptr;
  std::uint8_t* isLinear = nullptr;
};

template <typename Scalar> struct BendStretchTermsDevicePtrT {
  int*    idx1        = nullptr;
  int*    idx2        = nullptr;
  int*    idx3        = nullptr;
  Scalar* theta0      = nullptr;
  Scalar* restLen1    = nullptr;
  Scalar* restLen2    = nullptr;
  Scalar* forceConst1 = nullptr;
  Scalar* forceConst2 = nullptr;
};

template <typename Scalar> struct OutOfPlaneTermsDevicePtrT {
  int*    idx1 = nullptr;
  int*    idx2 = nullptr;
  int*    idx3 = nullptr;
  int*    idx4 = nullptr;
  Scalar* koop = nullptr;
};

template <typename Scalar> struct TorsionContribTermsDevicePtrT {
  int*    idx1 = nullptr;
  int*    idx2 = nullptr;
  int*    idx3 = nullptr;
  int*    idx4 = nullptr;
  Scalar* V1   = nullptr;
  Scalar* V2   = nullptr;
  Scalar* V3   = nullptr;
};

template <typename Scalar> struct VdwTermsDevicePtrT {
  int*    idx1      = nullptr;
  int*    idx2      = nullptr;
  Scalar* R_ij_star = nullptr;
  Scalar* wellDepth = nullptr;
};

template <typename Scalar> struct EleTermsDevicePtrT {
  int*     idx1       = nullptr;
  int*     idx2       = nullptr;
  Scalar*  chargeTerm = nullptr;
  uint8_t* dielModel  = nullptr;
  uint8_t* is1_4      = nullptr;
};

template <typename Scalar> struct DistanceConstraintTermsDevicePtrT {
  int*    idx1          = nullptr;
  int*    idx2          = nullptr;
  Scalar* minLen        = nullptr;
  Scalar* maxLen        = nullptr;
  Scalar* forceConstant = nullptr;
};

template <typename Scalar> struct PositionConstraintTermsDevicePtrT {
  int*    idx           = nullptr;
  Scalar* refX          = nullptr;
  Scalar* refY          = nullptr;
  Scalar* refZ          = nullptr;
  Scalar* maxDispl      = nullptr;
  Scalar* forceConstant = nullptr;
};

template <typename Scalar> struct AngleConstraintTermsDevicePtrT {
  int*    idx1          = nullptr;
  int*    idx2          = nullptr;
  int*    idx3          = nullptr;
  Scalar* minAngleDeg   = nullptr;
  Scalar* maxAngleDeg   = nullptr;
  Scalar* forceConstant = nullptr;
};

template <typename Scalar> struct TorsionConstraintTermsDevicePtrT {
  int*    idx1           = nullptr;
  int*    idx2           = nullptr;
  int*    idx3           = nullptr;
  int*    idx4           = nullptr;
  Scalar* minDihedralDeg = nullptr;
  Scalar* maxDihedralDeg = nullptr;
  Scalar* forceConstant  = nullptr;
};

template <typename Scalar, typename TorsionScalar = float> struct EnergyForceContribsDevicePtrT {
  BondStretchContribTermsDevicePtrT<Scalar>    bondTerms;
  AngleBendTermsDevicePtrT<Scalar>             angleTerms;
  BendStretchTermsDevicePtrT<Scalar>           bendTerms;
  OutOfPlaneTermsDevicePtrT<Scalar>            oopTerms;
  TorsionContribTermsDevicePtrT<TorsionScalar> torsionTerms;
  VdwTermsDevicePtrT<Scalar>                   vdwTerms;
  EleTermsDevicePtrT<Scalar>                   eleTerms;
  DistanceConstraintTermsDevicePtrT<Scalar>    distanceConstraintTerms;
  PositionConstraintTermsDevicePtrT<Scalar>    positionConstraintTerms;
  AngleConstraintTermsDevicePtrT<Scalar>       angleConstraintTerms;
  TorsionConstraintTermsDevicePtrT<Scalar>     torsionConstraintTerms;
};

using BondStretchContribTermsDevicePtr = BondStretchContribTermsDevicePtrT<double>;
using AngleBendTermsDevicePtr          = AngleBendTermsDevicePtrT<double>;
using BendStretchTermsDevicePtr        = BendStretchTermsDevicePtrT<double>;
using OutOfPlaneTermsDevicePtr         = OutOfPlaneTermsDevicePtrT<double>;
using TorsionContribTermsDevicePtr     = TorsionContribTermsDevicePtrT<float>;
using VdwTermsDevicePtr                = VdwTermsDevicePtrT<double>;
using EleTermsDevicePtr                = EleTermsDevicePtrT<double>;
using DistanceConstraintTermsDevicePtr = DistanceConstraintTermsDevicePtrT<double>;
using PositionConstraintTermsDevicePtr = PositionConstraintTermsDevicePtrT<double>;
using AngleConstraintTermsDevicePtr    = AngleConstraintTermsDevicePtrT<double>;
using TorsionConstraintTermsDevicePtr  = TorsionConstraintTermsDevicePtrT<double>;
using EnergyForceContribsDevicePtr     = EnergyForceContribsDevicePtrT<double, float>;

struct BatchedIndicesDevicePtr {
  int* atomStarts                   = nullptr;
  int* bondTermStarts               = nullptr;
  int* angleTermStarts              = nullptr;
  int* bendTermStarts               = nullptr;
  int* oopTermStarts                = nullptr;
  int* torsionTermStarts            = nullptr;
  int* vdwTermStarts                = nullptr;
  int* eleTermStarts                = nullptr;
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
                                          cudaStream_t                        stream = nullptr);

cudaError_t launchBlockPerMolGradKernel(int                                 numMols,
                                        const EnergyForceContribsDevicePtr& terms,
                                        const BatchedIndicesDevicePtr&      sytemIndices,
                                        const double*                       coords,
                                        double*                             grad,
                                        bool                                hasConstraints,
                                        cudaStream_t                        stream = nullptr);

}  // namespace MMFF
}  // namespace nvMolKit

#endif  // NVMOLKIT_MMFF_KERNELS_H
