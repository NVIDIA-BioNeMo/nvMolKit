// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "src/descriptors3d.h"
#include "src/descriptors3d_kernel.cuh"
#include "src/descriptors3d_moments.cuh"
#include "src/utils/cuda_error_check.h"

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
    case Property3D::NPR1:
      return "NPR1";
    case Property3D::NPR2:
      return "NPR2";
    case Property3D::InertialShapeFactor:
      return "InertialShapeFactor";
    case Property3D::Eccentricity:
      return "Eccentricity";
    case Property3D::Asphericity:
      return "Asphericity";
    case Property3D::SpherocityIndex:
      return "SpherocityIndex";
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

using descriptors3d_detail::computeInertiaTensor;
using descriptors3d_detail::computeMomentProperty;
using descriptors3d_detail::computePrincipalMoments;
using descriptors3d_detail::ConformerAtoms;
using descriptors3d_detail::kBlockSize;
using descriptors3d_detail::kConformersPerBlock;
using descriptors3d_detail::kGroupSize;
using descriptors3d_detail::kGroupsPerWarp;
using descriptors3d_detail::kWarpSize;
using descriptors3d_detail::kWarpsPerBlock;
using descriptors3d_detail::loadConformer;
using descriptors3d_detail::MomentState;

constexpr int kNumMomentProperty3D = static_cast<int>(kAllProperty3D.size());

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
    case Property3D::NPR1:
    case Property3D::NPR2:
    case Property3D::InertialShapeFactor:
    case Property3D::Eccentricity:
    case Property3D::Asphericity:
    case Property3D::SpherocityIndex:
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
  Property3D  properties[kNumMomentProperty3D];
  Real*       values[kNumMomentProperty3D];
  int         numStages;
  int         numProperties;
};

//! Everything a shared stage or property sees for one conformer. Every lane of a group holds the same
//! atoms and shared state.
template <typename Real> struct ConformerContext {
  ConformerAtoms    atoms;
  int               laneInGroup;
  MomentState<Real> moments;
};

//! Must be called by every lane of the warp with the same @p stage.
template <typename Real>
__device__ __forceinline__ void runSharedStage(const SharedStage stage, ConformerContext<Real>& context) {
  switch (stage) {
    case SharedStage::InertiaTensor:
      computeInertiaTensor(context.atoms, context.laneInGroup, context.moments);
      return;
    case SharedStage::PrincipalMoments:
      computePrincipalMoments(context.moments);
      return;
  }
}

template <typename Real>
__device__ __forceinline__ Real computeUnitWeightSpherocity(const ConformerContext<Real>& context) {
  ConformerAtoms unitAtoms = context.atoms;
  unitAtoms.weights        = nullptr;
  MomentState<Real> unitMoments;
  computeInertiaTensor(unitAtoms, context.laneInGroup, unitMoments);
  computePrincipalMoments(unitMoments);
  return computeMomentProperty(Property3D::SpherocityIndex, unitMoments);
}

template <typename Real, bool kSeparateSpherocityState>
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

  Real unitWeightSpherocity = 0;
  if constexpr (kSeparateSpherocityState) {
    unitWeightSpherocity = computeUnitWeightSpherocity(context);
  }
  for (int i = 0; i < blockWork.numStages; ++i) {
    runSharedStage(blockWork.stages[i], context);
  }
  for (int i = 0; i < blockWork.numProperties; ++i) {
    Real value;
    if constexpr (kSeparateSpherocityState) {
      value = blockWork.properties[i] == Property3D::SpherocityIndex ?
                unitWeightSpherocity :
                computeMomentProperty(blockWork.properties[i], context.moments);
    } else {
      value = computeMomentProperty(blockWork.properties[i], context.moments);
    }
    if (context.laneInGroup == 0 && conformerIdx < coordinates.numConformers) {
      blockWork.values[i][conformerIdx] = context.atoms.valid ? value : static_cast<Real>(nan(""));
    }
  }
}

template <typename Real> void addMomentProperty(Property3DWork<Real>& work, const Property3D property, Real* values) {
  work.properties[work.numProperties] = property;
  work.values[work.numProperties]     = values;
  ++work.numProperties;
}

template <typename Real> void prepareStages(Property3DWork<Real>& work, const bool separateSpherocityState) {
  SharedStageSet stages = 0;
  for (int propertyIdx = 0; propertyIdx < work.numProperties; ++propertyIdx) {
    if (separateSpherocityState && work.properties[propertyIdx] == Property3D::SpherocityIndex) {
      continue;
    }
    stages |= directStages(work.properties[propertyIdx]);
  }
  stages = withPrerequisites(stages);
  for (int stage = 0; stage < kNumSharedStages; ++stage) {
    if (stages & stageBit(static_cast<SharedStage>(stage))) {
      work.stages[work.numStages++] = static_cast<SharedStage>(stage);
    }
  }
}

template <typename Real>
void launchMomentProperties(const DeviceCoordView& coordinates,
                            const double*          atomWeights,
                            const int32_t*         moleculeAtomStarts,
                            Property3DWork<Real>&  work,
                            const bool             separateSpherocityState,
                            const cudaStream_t     stream) {
  if (work.numProperties == 0 || coordinates.numConformers == 0) {
    return;
  }
  prepareStages(work, separateSpherocityState);
  const int numBlocks = (coordinates.numConformers + kConformersPerBlock - 1) / kConformersPerBlock;
  if (separateSpherocityState) {
    property3DKernel<Real, true>
      <<<numBlocks, kBlockSize, 0, stream>>>(coordinates, atomWeights, moleculeAtomStarts, work);
  } else {
    property3DKernel<Real, false>
      <<<numBlocks, kBlockSize, 0, stream>>>(coordinates, atomWeights, moleculeAtomStarts, work);
  }
  cudaCheckError(cudaGetLastError());
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
  bool                    hasSpherocity = false;
  for (const Property3D property : properties) {
    // Bounds the work arrays, which hold one slot per known property.
    if (std::find(kAllProperty3D.begin(), kAllProperty3D.end(), property) == kAllProperty3D.end()) {
      throw std::invalid_argument("Unknown Property3D value " + std::to_string(static_cast<int>(property)));
    }
    auto [it, inserted] = results.try_emplace(property, coordinates.numConformers, stream);
    if (!inserted) {
      throw std::invalid_argument("Duplicate 3D property '" + std::string(property3DName(property)) + "'");
    }
    addMomentProperty(work, property, it->second.data());
    hasSpherocity |= property == Property3D::SpherocityIndex;
  }

  if (coordinates.numConformers == 0) {
    return results;
  }
  if (coordinates.positions == nullptr || coordinates.atomStarts == nullptr || coordinates.molIndices == nullptr ||
      moleculeAtomStarts == nullptr) {
    throw std::invalid_argument("3D property input buffers must not be null for a non-empty batch");
  }

  const bool    atomWeightsAreUnit      = atomWeights == nullptr;
  const bool    separateSpherocityState = hasSpherocity && properties.size() > 1 && !atomWeightsAreUnit;
  const double* effectiveAtomWeights =
    atomWeightsAreUnit || (hasSpherocity && properties.size() == 1) ? nullptr : atomWeights;
  launchMomentProperties(coordinates, effectiveAtomWeights, moleculeAtomStarts, work, separateSpherocityState, stream);
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
