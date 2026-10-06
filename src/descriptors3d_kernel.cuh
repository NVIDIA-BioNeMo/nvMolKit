// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_KERNEL_CUH
#define NVMOLKIT_DESCRIPTORS3D_KERNEL_CUH

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstddef>
#include <cstdint>

#include "src/conformer/device_coord_result.h"

namespace nvMolKit::descriptors3d_detail {

constexpr int kWarpSize           = 32;
constexpr int kBlockSize          = 128;
constexpr int kWarpsPerBlock      = kBlockSize / kWarpSize;
constexpr int kGroupSize          = 8;
constexpr int kGroupsPerWarp      = kWarpSize / kGroupSize;
constexpr int kConformersPerBlock = kWarpsPerBlock * kGroupsPerWarp;

//! Atom-property channels uploaded for WHIM, RDF and MORSE (Property3DDeviceInputs::atomPropertyWeights),
//! used after their unweighted channel.
constexpr int kNumAtomPropertyChannels = 6;

struct ConformerAtoms {
  const double* positions = nullptr;
  const double* weights   = nullptr;
  int           numAtoms  = 0;
  bool          valid     = false;
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
  atoms.weights   = atomWeights == nullptr ? nullptr : atomWeights + weightStart;
  atoms.numAtoms  = numAtoms;
  atoms.valid     = true;
  return atoms;
}

//! Aligned tile of @p kWidth lanes containing the calling thread.
template <int kWidth> __device__ __forceinline__ cooperative_groups::thread_block_tile<kWidth> laneTile() {
  return cooperative_groups::tiled_partition<kWidth>(cooperative_groups::this_thread_block());
}

//! Sum over aligned groups of @p kWidth lanes (a conformer group by default, or the whole warp), returned to
//! every lane. Every lane of the group must call it.
template <int kWidth = kGroupSize, typename Real> __device__ __forceinline__ Real groupAllReduceSum(Real value) {
  return cooperative_groups::reduce(laneTile<kWidth>(), value, cooperative_groups::plus<Real>());
}

//! Sum over the warp's lanes with equal `lane % kGroupSize`, i.e. across its kGroupsPerWarp groups, returned
//! to each of those lanes. Every lane of the warp must call it. No tile covers these strided lanes, so this
//! combines with shuffles across group offsets.
template <typename Real> __device__ __forceinline__ Real sumAcrossGroups(Real value) {
  for (int offset = kGroupSize; offset < kWarpSize; offset <<= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

/**
 * @brief Coordinate position minus a centroid, converted to @p Real after the subtraction.
 *
 * FP64 required: centroids and centering. Casting coordinates to float32 before centering keeps only
 * 24 bits relative to the distance from the origin, so a conformer translated to 1e4 A loses ~1e-3 A of
 * interatomic detail and translation-invariant descriptors drift by ~1e-4 relative (PBF, PMI and
 * RadiusOfGyration, measured). Centered values are small, so everything after this step is float32-safe.
 */
template <typename Real>
__device__ __forceinline__ void centeredPosition(const double* positions,
                                                 const int     atomIdx,
                                                 const double  centroidX,
                                                 const double  centroidY,
                                                 const double  centroidZ,
                                                 Real&         x,
                                                 Real&         y,
                                                 Real&         z) {
  x = static_cast<Real>(positions[atomIdx * 3 + 0] - centroidX);
  y = static_cast<Real>(positions[atomIdx * 3 + 1] - centroidY);
  z = static_cast<Real>(positions[atomIdx * 3 + 2] - centroidZ);
}

/**
 * @brief Group-collective. Writes the conformer's coordinates, centered on their unweighted centroid in
 *        float64 (see centeredPosition()), to @p centered as @p Real, laid out like the coordinate rows.
 *        Every lane of the group can read all of @p centered on return.
 * @return The float64 centroid.
 */
template <typename Real>
__device__ __forceinline__ double3
writeCenteredConformer(const ConformerAtoms& atoms, const int laneInGroup, const unsigned groupMask, Real* centered) {
  double sumX = 0;
  double sumY = 0;
  double sumZ = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    sumX += atoms.positions[atomIdx * 3 + 0];
    sumY += atoms.positions[atomIdx * 3 + 1];
    sumZ += atoms.positions[atomIdx * 3 + 2];
  }
  const double inverseAtoms = 1.0 / static_cast<double>(atoms.numAtoms);
  const double centroidX    = groupAllReduceSum(sumX) * inverseAtoms;
  const double centroidY    = groupAllReduceSum(sumY) * inverseAtoms;
  const double centroidZ    = groupAllReduceSum(sumZ) * inverseAtoms;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    centeredPosition(atoms.positions,
                     atomIdx,
                     centroidX,
                     centroidY,
                     centroidZ,
                     centered[atomIdx * 3 + 0],
                     centered[atomIdx * 3 + 1],
                     centered[atomIdx * 3 + 2]);
  }
  __syncwarp(groupMask);
  return make_double3(centroidX, centroidY, centroidZ);
}

//! Bond depth of an atom not reached within the search's depth cap.
constexpr uint8_t kUnreachedDepth = 0xFF;

/**
 * @brief Collective over the @p groupMask lanes. Breadth-first search over the molecule's bonds from atom
 *        @p source, writing each atom's bond-count distance to @p depth, or kUnreachedDepth beyond
 *        @p kMaxDepth bonds. Matches RDKit's MolOps::getDistanceMat(mol, false) up to that cap.
 *
 * Each level scans the atoms set at the previous level, O(kMaxDepth * atoms) per source; lanes that reach the
 * same atom write the same value.
 */
template <int kMaxDepth>
__device__ __forceinline__ void searchBondDepths(const int32_t* neighborStarts,
                                                 const int32_t* neighbors,
                                                 const int      numAtoms,
                                                 const int      source,
                                                 const int      laneInGroup,
                                                 const unsigned groupMask,
                                                 uint8_t*       depth) {
  static_assert(kMaxDepth < kUnreachedDepth);
  // Every lane has finished reading the previous source's depths before they are overwritten.
  __syncwarp(groupMask);
  for (int atomIdx = laneInGroup; atomIdx < numAtoms; atomIdx += kGroupSize) {
    depth[atomIdx] = atomIdx == source ? 0 : kUnreachedDepth;
  }
  __syncwarp(groupMask);
  for (int level = 1; level <= kMaxDepth; ++level) {
    for (int atomIdx = laneInGroup; atomIdx < numAtoms; atomIdx += kGroupSize) {
      if (depth[atomIdx] == level - 1) {
        for (int edge = neighborStarts[atomIdx]; edge < neighborStarts[atomIdx + 1]; ++edge) {
          const int neighbor = neighbors[edge];
          if (depth[neighbor] == kUnreachedDepth) {
            depth[neighbor] = static_cast<uint8_t>(level);
          }
        }
      }
    }
    __syncwarp(groupMask);
  }
}

//! Round to three decimals as RDKit's WHIM, RDF and MORSE do: std::round(1000 * x) / 1000.
template <typename Real> __device__ __forceinline__ Real roundThousandths(const Real value) {
  return round(value * Real(1000)) / Real(1000);
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_KERNEL_CUH
