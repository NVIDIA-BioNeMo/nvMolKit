// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "src/descriptors3d.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/symmetric_eigenvalues_3x3.cuh"

namespace nvMolKit {

std::string_view property3DName(const Property3D property) {
  switch (property) {
    case Property3D::PMI1:
      return "PMI1";
    case Property3D::PMI2:
      return "PMI2";
    case Property3D::PMI3:
      return "PMI3";
    case Property3D::RadiusOfGyration:
      return "RadiusOfGyration";
  }
  throw std::invalid_argument("Unknown Property3D value " + std::to_string(static_cast<int>(property)));
}

Property3D property3DFromName(const std::string_view name) {
  for (const Property3D property : kAllProperty3D) {
    if (property3DName(property) == name) {
      return property;
    }
  }
  throw std::invalid_argument("Unknown 3D property '" + std::string(name) + "'");
}

namespace {

// A group of kGroupSize lanes works on one conformer, so a warp covers kGroupsPerWarp conformers.
constexpr int      kWarpSize           = 32;
constexpr int      kBlockSize          = 128;
constexpr int      kWarpsPerBlock      = kBlockSize / kWarpSize;
constexpr int      kGroupSize          = 8;
constexpr int      kGroupsPerWarp      = kWarpSize / kGroupSize;
constexpr int      kConformersPerBlock = kWarpsPerBlock * kGroupsPerWarp;
constexpr int      kNumProperty3D      = static_cast<int>(kAllProperty3D.size());
constexpr unsigned kFullWarpMask       = 0xffffffffu;

//! Per-conformer work shared between properties. Enumerators are in dependency order.
enum class SharedStage : int {
  InertiaTensor    = 0,  //!< Total weight and inertia tensor about the weighted centroid.
  PrincipalMoments = 1,  //!< Eigenvalues of the inertia tensor. Requires InertiaTensor.
};
constexpr int kNumSharedStages = 2;

using SharedStageSet = uint32_t;

constexpr SharedStageSet stageBit(const SharedStage stage) {
  return 1u << static_cast<int>(stage);
}

//! Shared stages a property reads directly.
constexpr SharedStageSet directStages(const Property3D property) {
  switch (property) {
    case Property3D::PMI1:
    case Property3D::PMI2:
    case Property3D::PMI3:
      return stageBit(SharedStage::PrincipalMoments);
    case Property3D::RadiusOfGyration:
      return stageBit(SharedStage::InertiaTensor);
  }
  return 0;
}

//! Close a stage set over its prerequisites.
constexpr SharedStageSet withPrerequisites(SharedStageSet stages) {
  if (stages & stageBit(SharedStage::PrincipalMoments)) {
    stages |= stageBit(SharedStage::InertiaTensor);
  }
  return stages;
}

//! Host-precomputed work list: shared stages in dependency order, then requested properties and their
//! output buffers in request order. Trivially constructible so the kernel can stage it in shared memory
//! for its runtime-indexed loops.
template <typename Real> struct Property3DWork {
  SharedStage stages[kNumSharedStages];
  Property3D  properties[kNumProperty3D];
  Real*       values[kNumProperty3D];
  int         numStages;
  int         numProperties;
};

//! Results of the shared stages. Only members of stages in the work list are set.
template <typename Real> struct SharedState {
  // InertiaTensor: upper triangle of the weighted inertia tensor about the weighted centroid.
  Real inertiaXX;
  Real inertiaXY;
  Real inertiaXZ;
  Real inertiaYY;
  Real inertiaYZ;
  Real inertiaZZ;
  Real totalWeight;
  // PrincipalMoments
  Real smallestMoment;
  Real middleMoment;
  Real largestMoment;
};

//! Read-only inputs for one conformer. Out-of-range or inconsistent rows have no atoms and are invalid.
struct ConformerAtoms {
  const double* positions = nullptr;
  const double* weights   = nullptr;
  int           numAtoms  = 0;
  bool          valid     = false;
};

//! Everything a shared stage or property sees for one conformer. Every lane of a group holds the same
//! atoms and shared state.
template <typename Real> struct ConformerContext {
  ConformerAtoms    atoms;
  int               laneInGroup;
  SharedState<Real> shared;
};

__device__ __forceinline__ ConformerAtoms loadConformer(const DeviceCoordView& coordinates,
                                                        const double*          atomWeights,
                                                        const int32_t*         moleculeAtomStarts,
                                                        const int              conformerIdx) {
  ConformerAtoms atoms;
  if (conformerIdx >= coordinates.numConformers) {
    return atoms;
  }
  const int     moleculeIdx = coordinates.molIndices[conformerIdx];
  const int64_t atomStart   = coordinates.atomStarts[conformerIdx];
  const int64_t atomStop    = coordinates.atomStarts[conformerIdx + 1];
  // Caller-supplied offsets are untrusted: the row must lie inside the coordinate buffer.
  if (moleculeIdx < 0 || moleculeIdx >= coordinates.nMols || atomStart < 0 || atomStop <= atomStart ||
      atomStop > coordinates.numAtoms) {
    return atoms;
  }
  const int numAtoms    = static_cast<int>(atomStop - atomStart);
  const int weightStart = moleculeAtomStarts[moleculeIdx];
  if (moleculeAtomStarts[moleculeIdx + 1] - weightStart != numAtoms) {
    return atoms;
  }
  atoms.positions = coordinates.positions + static_cast<size_t>(atomStart) * 3;
  atoms.weights   = atomWeights + weightStart;
  atoms.numAtoms  = numAtoms;
  atoms.valid     = true;
  return atoms;
}

//! Sum across the kGroupSize lanes of a group; every lane of the group receives the total. Must be
//! called by every lane of the warp.
template <typename Real> __device__ __forceinline__ Real groupAllReduceSum(Real value) {
  for (int offset = kGroupSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFullWarpMask, value, offset);
  }
  return value;
}

//! Group-collective. Two passes (centroid, then moments about it) avoid the cancellation of a single
//! raw-moment pass for molecules far from the origin.
template <typename Real> __device__ __forceinline__ void computeInertiaTensor(ConformerContext<Real>& context) {
  const ConformerAtoms& atoms       = context.atoms;
  Real                  weightedX   = 0;
  Real                  weightedY   = 0;
  Real                  weightedZ   = 0;
  Real                  totalWeight = 0;
  for (int atomIdx = context.laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const Real weight = static_cast<Real>(atoms.weights[atomIdx]);
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
  for (int atomIdx = context.laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const Real weight = static_cast<Real>(atoms.weights[atomIdx]);
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
  SharedState<Real>& shared = context.shared;
  shared.inertiaXX          = groupAllReduceSum(inertiaXX);
  shared.inertiaXY          = groupAllReduceSum(inertiaXY);
  shared.inertiaXZ          = groupAllReduceSum(inertiaXZ);
  shared.inertiaYY          = groupAllReduceSum(inertiaYY);
  shared.inertiaYZ          = groupAllReduceSum(inertiaYZ);
  shared.inertiaZZ          = groupAllReduceSum(inertiaZZ);
  shared.totalWeight        = totalWeight;
}

//! Degenerate moments (symmetric and spherical tops) need the backward-stable Jacobi solver to stay
//! accurate to rounding, most visibly in single precision.
template <typename Real> __device__ __forceinline__ void computePrincipalMoments(ConformerContext<Real>& context) {
  SharedState<Real>& shared = context.shared;
  symmetricEigenvaluesJacobi3x3(shared.inertiaXX,
                                shared.inertiaXY,
                                shared.inertiaXZ,
                                shared.inertiaYY,
                                shared.inertiaYZ,
                                shared.inertiaZZ,
                                shared.largestMoment,
                                shared.middleMoment,
                                shared.smallestMoment);
}

//! Must be called by every lane of the warp with the same @p stage.
template <typename Real>
__device__ __forceinline__ void runSharedStage(const SharedStage stage, ConformerContext<Real>& context) {
  switch (stage) {
    case SharedStage::InertiaTensor:
      computeInertiaTensor(context);
      return;
    case SharedStage::PrincipalMoments:
      computePrincipalMoments(context);
      return;
  }
}

template <typename Real> __device__ __forceinline__ Real principalMoment1(const ConformerContext<Real>& context) {
  return fmax(context.shared.smallestMoment, Real(0));
}

template <typename Real> __device__ __forceinline__ Real principalMoment2(const ConformerContext<Real>& context) {
  return fmax(context.shared.middleMoment, Real(0));
}

template <typename Real> __device__ __forceinline__ Real principalMoment3(const ConformerContext<Real>& context) {
  return fmax(context.shared.largestMoment, Real(0));
}

template <typename Real> __device__ __forceinline__ Real radiusOfGyration(const ConformerContext<Real>& context) {
  const SharedState<Real>& shared                = context.shared;
  // trace(I) = 2 * sum(w * r^2) about the centroid.
  const Real               weightedSquaredRadius = Real(0.5) * (shared.inertiaXX + shared.inertiaYY + shared.inertiaZZ);
  return sqrt(fmax(weightedSquaredRadius / shared.totalWeight, Real(0)));
}

//! Must be called by every lane of the warp with the same @p property, so properties may use
//! group-collective work over the conformer's atoms. Every lane of the group returns the value.
template <typename Real>
__device__ __forceinline__ Real computeProperty(const Property3D property, const ConformerContext<Real>& context) {
  switch (property) {
    case Property3D::PMI1:
      return principalMoment1(context);
    case Property3D::PMI2:
      return principalMoment2(context);
    case Property3D::PMI3:
      return principalMoment3(context);
    case Property3D::RadiusOfGyration:
      return radiusOfGyration(context);
  }
  return static_cast<Real>(nan(""));
}

template <typename Real>
__global__ void property3DKernel(const DeviceCoordView coordinates,
                                 const double* __restrict__ atomWeights,
                                 const int32_t* __restrict__ moleculeAtomStarts,
                                 const Property3DWork<Real> work) {
  __shared__ Property3DWork<Real> blockWork;
  if (threadIdx.x == 0) {
    blockWork = work;
  }
  __syncthreads();

  const int lane         = static_cast<int>(threadIdx.x) % kWarpSize;
  const int warpStart    = (blockIdx.x * kWarpsPerBlock + static_cast<int>(threadIdx.x) / kWarpSize) * kGroupsPerWarp;
  const int conformerIdx = warpStart + lane / kGroupSize;
  if (warpStart >= coordinates.numConformers) {
    return;  // Uniform across the warp; partially filled warps keep every lane for the shuffles.
  }

  ConformerContext<Real> context;
  context.atoms       = loadConformer(coordinates, atomWeights, moleculeAtomStarts, conformerIdx);
  context.laneInGroup = lane % kGroupSize;

  for (int i = 0; i < blockWork.numStages; ++i) {
    runSharedStage(blockWork.stages[i], context);
  }
  for (int i = 0; i < blockWork.numProperties; ++i) {
    const Real value = computeProperty(blockWork.properties[i], context);
    if (context.laneInGroup == 0 && conformerIdx < coordinates.numConformers) {
      blockWork.values[i][conformerIdx] = context.atoms.valid ? value : static_cast<Real>(nan(""));
    }
  }
}

}  // namespace

template <typename Real>
Property3DResults<Real> calc3DPropertiesGpu(const DeviceCoordView&         coordinates,
                                            const double*                  atomWeights,
                                            const int32_t*                 moleculeAtomStarts,
                                            const std::vector<Property3D>& properties,
                                            const cudaStream_t             stream) {
  if (properties.empty()) {
    throw std::invalid_argument("At least one 3D property must be requested");
  }
  if (coordinates.numConformers < 0 || coordinates.nMols < 0) {
    throw std::invalid_argument("Batch dimensions must not be negative");
  }

  Property3DResults<Real> results;
  Property3DWork<Real>    work{};
  SharedStageSet          stages = 0;
  for (const Property3D property : properties) {
    // Bounds the work arrays, which hold one slot per known property.
    if (std::find(kAllProperty3D.begin(), kAllProperty3D.end(), property) == kAllProperty3D.end()) {
      throw std::invalid_argument("Unknown Property3D value " + std::to_string(static_cast<int>(property)));
    }
    auto [it, inserted] = results.try_emplace(property, coordinates.numConformers, stream);
    if (!inserted) {
      throw std::invalid_argument("Duplicate 3D property '" + std::string(property3DName(property)) + "'");
    }
    work.properties[work.numProperties] = property;
    work.values[work.numProperties]     = it->second.data();
    ++work.numProperties;
    stages |= directStages(property);
  }
  stages = withPrerequisites(stages);
  for (int stage = 0; stage < kNumSharedStages; ++stage) {
    if (stages & stageBit(static_cast<SharedStage>(stage))) {
      work.stages[work.numStages++] = static_cast<SharedStage>(stage);
    }
  }

  if (coordinates.numConformers == 0) {
    return results;
  }
  if (coordinates.positions == nullptr || coordinates.atomStarts == nullptr || coordinates.molIndices == nullptr ||
      atomWeights == nullptr || moleculeAtomStarts == nullptr) {
    throw std::invalid_argument("3D property input buffers must not be null for a non-empty batch");
  }

  const int numBlocks = (coordinates.numConformers + kConformersPerBlock - 1) / kConformersPerBlock;
  property3DKernel<Real><<<numBlocks, kBlockSize, 0, stream>>>(coordinates, atomWeights, moleculeAtomStarts, work);
  cudaCheckError(cudaGetLastError());
  return results;
}

template Property3DResults<float>  calc3DPropertiesGpu<float>(const DeviceCoordView&,
                                                             const double*,
                                                             const int32_t*,
                                                             const std::vector<Property3D>&,
                                                             cudaStream_t);
template Property3DResults<double> calc3DPropertiesGpu<double>(const DeviceCoordView&,
                                                               const double*,
                                                               const int32_t*,
                                                               const std::vector<Property3D>&,
                                                               cudaStream_t);

}  // namespace nvMolKit
