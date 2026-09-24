// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_MOMENTS_CUH
#define NVMOLKIT_DESCRIPTORS3D_MOMENTS_CUH

#include <cmath>

#include "src/descriptors3d.h"
#include "src/descriptors3d_kernel.cuh"
#include "src/utils/symmetric_eigenvalues_3x3.cuh"

namespace nvMolKit::descriptors3d_detail {

template <typename Real> struct MomentState {
  Real inertiaXX;
  Real inertiaXY;
  Real inertiaXZ;
  Real inertiaYY;
  Real inertiaYZ;
  Real inertiaZZ;
  Real totalWeight;
  Real smallestMoment;
  Real middleMoment;
  Real largestMoment;
};

template <typename Real>
__device__ __forceinline__ void computeInertiaTensor(const ConformerAtoms& atoms,
                                                     const int             laneInGroup,
                                                     MomentState<Real>&    state) {
  Real weightedX   = 0;
  Real weightedY   = 0;
  Real weightedZ   = 0;
  Real totalWeight = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const Real weight = atoms.weights == nullptr ? Real(1) : static_cast<Real>(atoms.weights[atomIdx]);
    weightedX += weight * static_cast<Real>(atoms.positions[atomIdx * 3 + 0]);
    weightedY += weight * static_cast<Real>(atoms.positions[atomIdx * 3 + 1]);
    weightedZ += weight * static_cast<Real>(atoms.positions[atomIdx * 3 + 2]);
    totalWeight += weight;
  }
  totalWeight          = groupAllReduceSum(totalWeight);
  const Real centroidX = groupAllReduceSum(weightedX) / totalWeight;
  const Real centroidY = groupAllReduceSum(weightedY) / totalWeight;
  const Real centroidZ = groupAllReduceSum(weightedZ) / totalWeight;

  Real inertiaXX = 0;
  Real inertiaXY = 0;
  Real inertiaXZ = 0;
  Real inertiaYY = 0;
  Real inertiaYZ = 0;
  Real inertiaZZ = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const Real weight = atoms.weights == nullptr ? Real(1) : static_cast<Real>(atoms.weights[atomIdx]);
    const Real x      = static_cast<Real>(atoms.positions[atomIdx * 3 + 0]) - centroidX;
    const Real y      = static_cast<Real>(atoms.positions[atomIdx * 3 + 1]) - centroidY;
    const Real z      = static_cast<Real>(atoms.positions[atomIdx * 3 + 2]) - centroidZ;
    inertiaXX += weight * (y * y + z * z);
    inertiaXY -= weight * x * y;
    inertiaXZ -= weight * x * z;
    inertiaYY += weight * (x * x + z * z);
    inertiaYZ -= weight * y * z;
    inertiaZZ += weight * (x * x + y * y);
  }
  state.inertiaXX   = groupAllReduceSum(inertiaXX);
  state.inertiaXY   = groupAllReduceSum(inertiaXY);
  state.inertiaXZ   = groupAllReduceSum(inertiaXZ);
  state.inertiaYY   = groupAllReduceSum(inertiaYY);
  state.inertiaYZ   = groupAllReduceSum(inertiaYZ);
  state.inertiaZZ   = groupAllReduceSum(inertiaZZ);
  state.totalWeight = totalWeight;
}

template <typename Real> __device__ __forceinline__ void computePrincipalMoments(MomentState<Real>& state) {
  symmetricEigenvaluesJacobi3x3(state.inertiaXX,
                                state.inertiaXY,
                                state.inertiaXZ,
                                state.inertiaYY,
                                state.inertiaYZ,
                                state.inertiaZZ,
                                state.largestMoment,
                                state.middleMoment,
                                state.smallestMoment);
}

template <typename Real> __device__ __forceinline__ Real principalMoment1(const MomentState<Real>& state) {
  return fmax(state.smallestMoment, Real(0));
}

template <typename Real> __device__ __forceinline__ Real principalMoment2(const MomentState<Real>& state) {
  return fmax(state.middleMoment, Real(0));
}

template <typename Real> __device__ __forceinline__ Real principalMoment3(const MomentState<Real>& state) {
  return fmax(state.largestMoment, Real(0));
}

template <typename Real>
__device__ __forceinline__ void gyrationMoments(const MomentState<Real>& state,
                                                Real&                    smallest,
                                                Real&                    middle,
                                                Real&                    largest) {
  const Real moment1       = principalMoment1(state);
  const Real moment2       = principalMoment2(state);
  const Real moment3       = principalMoment3(state);
  const Real inverseWeight = Real(0.5) / state.totalWeight;
  smallest                 = fmax((moment1 + moment2 - moment3) * inverseWeight, Real(0));
  middle                   = fmax((moment1 + moment3 - moment2) * inverseWeight, Real(0));
  largest                  = fmax((moment2 + moment3 - moment1) * inverseWeight, Real(0));
}

template <typename Real>
__device__ __forceinline__ Real computeMomentProperty(const Property3D property, const MomentState<Real>& state) {
  switch (property) {
    case Property3D::PMI1:
      return principalMoment1(state);
    case Property3D::PMI2:
      return principalMoment2(state);
    case Property3D::PMI3:
      return principalMoment3(state);
    case Property3D::RadiusOfGyration:
      return sqrt(fmax(Real(0.5) * (state.inertiaXX + state.inertiaYY + state.inertiaZZ) / state.totalWeight, Real(0)));
    case Property3D::NPR1: {
      const Real moment1 = principalMoment1(state);
      const Real moment3 = principalMoment3(state);
      return moment3 < Real(1e-8) ? Real(0) : moment1 / moment3;
    }
    case Property3D::NPR2: {
      const Real moment2 = principalMoment2(state);
      const Real moment3 = principalMoment3(state);
      return moment3 < Real(1e-8) ? Real(0) : moment2 / moment3;
    }
    case Property3D::InertialShapeFactor: {
      const Real moment1 = principalMoment1(state);
      const Real moment2 = principalMoment2(state);
      const Real moment3 = principalMoment3(state);
      return moment1 < Real(1e-4) || moment3 < Real(1e-4) ? Real(0) : moment2 / (moment1 * moment3);
    }
    case Property3D::Eccentricity: {
      const Real moment1           = principalMoment1(state);
      const Real moment3           = principalMoment3(state);
      const Real squaredDifference = moment3 * moment3 - moment1 * moment1;
      return moment3 < Real(1e-4) || squaredDifference < Real(1e-4) ? Real(0) : sqrt(squaredDifference) / moment3;
    }
    case Property3D::Asphericity: {
      Real smallest;
      Real middle;
      Real largest;
      gyrationMoments(state, smallest, middle, largest);
      if (largest < Real(1e-4)) {
        return Real(0);
      }
      const Real sum = smallest + middle + largest;
      return Real(0.5) *
             ((smallest - middle) * (smallest - middle) + (smallest - largest) * (smallest - largest) +
              (middle - largest) * (middle - largest)) /
             (sum * sum);
    }
    case Property3D::SpherocityIndex: {
      Real smallest;
      Real middle;
      Real largest;
      gyrationMoments(state, smallest, middle, largest);
      return largest < Real(1e-4) ? Real(0) : Real(3) * smallest / (smallest + middle + largest);
    }
    case Property3D::PBF:
    case Property3D::WHIM:
      return static_cast<Real>(nan(""));
  }
  return static_cast<Real>(nan(""));
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_MOMENTS_CUH
