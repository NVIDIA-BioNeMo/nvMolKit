// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_REFERENCE_CUH
#define NVMOLKIT_DESCRIPTORS3D_REFERENCE_CUH

#include <cmath>

#include "src/descriptors3d.h"
#include "src/descriptors3d_kernel.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

namespace nvMolKit::descriptors3d_detail {

constexpr int kNumUsrReferencePoints = 4;
constexpr int kNumUsrMoments         = 3;
//! USRCAT's atom classes (Property3DDeviceInputs::usrcatAtomClasses bits); subset 0 is every atom.
constexpr int kNumUsrcatClasses      = kNumUsrcatProperties / kNumUsrProperties - 1;

//! Closest or farthest atom seen so far; ties keep the lower atom index, as RDKit's scans do.
struct ExtremeAtom {
  double distance;
  int    atomIdx;
};

template <bool kLargest>
__device__ __forceinline__ bool isBetter(const double distance, const int atomIdx, const ExtremeAtom& best) {
  const bool strictlyBetter = kLargest ? distance > best.distance : distance < best.distance;
  return strictlyBetter || (distance == best.distance && atomIdx < best.atomIdx);
}

/**
 * @brief Group-collective. Index of the atom closest to (or, with @p kLargest, farthest from) @p point.
 *
 * FP64 required: the comparisons pick USR's reference atoms. Salt forms with a hydrochloride put an H and a
 * Cl within 1e-8 A of the same distance from the centroid (20 of 484 ChEMBL conformers measured), below
 * float32 resolution; a float32 comparison picked a different closest atom in 8 of them and changed that
 * reference point's moments by up to 1.3. The selection reads the float64 coordinates, O(atoms) per call.
 */
template <bool kLargest>
__device__ __forceinline__ int extremeAtom(const ConformerAtoms& atoms,
                                           const int             laneInGroup,
                                           const unsigned        groupMask,
                                           const double3         point) {
  ExtremeAtom best{kLargest ? -INFINITY : INFINITY, atoms.numAtoms};
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const double dx       = atoms.positions[atomIdx * 3 + 0] - point.x;
    const double dy       = atoms.positions[atomIdx * 3 + 1] - point.y;
    const double dz       = atoms.positions[atomIdx * 3 + 2] - point.z;
    const double distance = sqrt(dx * dx + dy * dy + dz * dz);
    if (isBetter<kLargest>(distance, atomIdx, best)) {
      best = {distance, atomIdx};
    }
  }
  for (int offset = kGroupSize / 2; offset > 0; offset >>= 1) {
    const double otherDistance = __shfl_xor_sync(groupMask, best.distance, offset);
    const int    otherIdx      = __shfl_xor_sync(groupMask, best.atomIdx, offset);
    if (isBetter<kLargest>(otherDistance, otherIdx, best)) {
      best = {otherDistance, otherIdx};
    }
  }
  return best.atomIdx;
}

//! Float64 position of atom @p atomIdx.
__device__ __forceinline__ double3 atomPosition(const ConformerAtoms& atoms, const int atomIdx) {
  return make_double3(atoms.positions[atomIdx * 3 + 0],
                      atoms.positions[atomIdx * 3 + 1],
                      atoms.positions[atomIdx * 3 + 2]);
}

/**
 * @brief Group-collective. RDKit's USR moments of the distances from (@p px, @p py, @p pz) to each atom
 *        subset: mean, standard deviation and the cube root of the standardized third central moment (0 when
 *        the standard deviation is 0; all 0 for an empty subset). Subset 0 is every atom; subset c + 1 holds
 *        the atoms whose @p classes bit c is set.
 */
template <typename Real, int kNumSubsets>
__device__ __forceinline__ void usrMoments(const Real*    centered,
                                           const uint8_t* classes,
                                           const int      numAtoms,
                                           const int      laneInGroup,
                                           const Real     px,
                                           const Real     py,
                                           const Real     pz,
                                           Real (&moments)[kNumSubsets][kNumUsrMoments]) {
  auto inSubset = [&](const int atomIdx, const int subset) {
    return subset == 0 || ((classes[atomIdx] >> (subset - 1)) & 1u) != 0;
  };
  auto distanceTo = [&](const int atomIdx) {
    const Real dx = centered[atomIdx * 3 + 0] - px;
    const Real dy = centered[atomIdx * 3 + 1] - py;
    const Real dz = centered[atomIdx * 3 + 2] - pz;
    return sqrt(dx * dx + dy * dy + dz * dz);
  };

  Real counts[kNumSubsets] = {};
  Real means[kNumSubsets]  = {};
  for (int atomIdx = laneInGroup; atomIdx < numAtoms; atomIdx += kGroupSize) {
    const Real distance = distanceTo(atomIdx);
    for (int subset = 0; subset < kNumSubsets; ++subset) {
      if (inSubset(atomIdx, subset)) {
        counts[subset] += Real(1);
        means[subset] += distance;
      }
    }
  }
  for (int subset = 0; subset < kNumSubsets; ++subset) {
    counts[subset] = groupAllReduceSum(counts[subset]);
    means[subset]  = counts[subset] > Real(0) ? groupAllReduceSum(means[subset]) / counts[subset] : Real(0);
  }

  Real squares[kNumSubsets] = {};
  Real cubes[kNumSubsets]   = {};
  for (int atomIdx = laneInGroup; atomIdx < numAtoms; atomIdx += kGroupSize) {
    const Real distance = distanceTo(atomIdx);
    for (int subset = 0; subset < kNumSubsets; ++subset) {
      if (inSubset(atomIdx, subset)) {
        const Real diff = distance - means[subset];
        squares[subset] += diff * diff;
        cubes[subset] += diff * diff * diff;
      }
    }
  }
  for (int subset = 0; subset < kNumSubsets; ++subset) {
    const Real squareSum = groupAllReduceSum(squares[subset]);
    const Real cubeSum   = groupAllReduceSum(cubes[subset]);
    if (counts[subset] == Real(0)) {
      moments[subset][0] = moments[subset][1] = moments[subset][2] = Real(0);
      continue;
    }
    const Real deviation = sqrt(squareSum / counts[subset]);
    const Real skew      = cubeSum / counts[subset];
    moments[subset][0]   = means[subset];
    moments[subset][1]   = deviation;
    moments[subset][2]   = deviation == Real(0) ? Real(0) : cbrt(skew / (deviation * deviation * deviation));
  }
}

/**
 * @brief USR and/or USRCAT for every conformer; one group of kGroupSize lanes per conformer.
 *
 * Reference points follow RDKit: the centroid, the atom closest to it, the atom farthest from it, and the
 * atom farthest from that one. USR holds the moments of every atom's distances from each point; USRCAT
 * (@p kUsrcat) appends the same moments for each atom class, measured from the same points. Each group first
 * writes its conformer's coordinates, centered in float64, to @p centered; everything after runs in @p Real.
 * Conformers with fewer than three atoms, which RDKit rejects, produce NaN.
 */
template <typename Real, bool kUsrcat>
__global__ void referencePoints3DKernel(const DeviceCoordView        coordinates,
                                        const Property3DDeviceInputs inputs,
                                        Real* __restrict__ centered,
                                        Real* __restrict__ usrOutput,
                                        Real* __restrict__ usrcatOutput) {
  constexpr int kNumSubsets = kUsrcat ? kNumUsrcatClasses + 1 : 1;
  const int     lane        = static_cast<int>(threadIdx.x) % kWarpSize;
  const int     laneInGroup = lane % kGroupSize;
  const int     conformerIdx =
    (blockIdx.x * kWarpsPerBlock + static_cast<int>(threadIdx.x) / kWarpSize) * kGroupsPerWarp + lane / kGroupSize;
  if (conformerIdx >= coordinates.numConformers) {
    return;
  }
  const unsigned       groupMask = 0xffu << (lane - laneInGroup);
  const ConformerAtoms atoms     = loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, conformerIdx);
  const bool           valid     = atoms.valid && atoms.numAtoms >= 3;
  Real* const          usrRow =
    usrOutput != nullptr ? usrOutput + static_cast<size_t>(conformerIdx) * kNumUsrProperties : nullptr;
  Real* const usrcatRow = kUsrcat ? usrcatOutput + static_cast<size_t>(conformerIdx) * kNumUsrcatProperties : nullptr;

  if (!valid) {
    for (int valueIdx = laneInGroup; valueIdx < kNumUsrProperties; valueIdx += kGroupSize) {
      if (usrRow != nullptr) {
        usrRow[valueIdx] = static_cast<Real>(nan(""));
      }
    }
    if constexpr (kUsrcat) {
      for (int valueIdx = laneInGroup; valueIdx < kNumUsrcatProperties; valueIdx += kGroupSize) {
        usrcatRow[valueIdx] = static_cast<Real>(nan(""));
      }
    }
    return;
  }

  Real* const    conformerCentered = centered + (atoms.positions - coordinates.positions);
  const double3  centroid          = writeCenteredConformer(atoms, laneInGroup, groupMask, conformerCentered);
  const uint8_t* classes =
    kUsrcat ? inputs.usrcatAtomClasses + inputs.moleculeAtomStarts[coordinates.molIndices[conformerIdx]] : nullptr;

  const int closest              = extremeAtom<false>(atoms, laneInGroup, groupMask, centroid);
  const int farthest             = extremeAtom<true>(atoms, laneInGroup, groupMask, centroid);
  const int farthestFromFarthest = extremeAtom<true>(atoms, laneInGroup, groupMask, atomPosition(atoms, farthest));
  // -1 is the centroid, the origin of the centered coordinates.
  const int referenceAtoms[kNumUsrReferencePoints] = {-1, closest, farthest, farthestFromFarthest};

  for (int point = 0; point < kNumUsrReferencePoints; ++point) {
    const int  atom = referenceAtoms[point];
    const Real px   = atom < 0 ? Real(0) : conformerCentered[atom * 3 + 0];
    const Real py   = atom < 0 ? Real(0) : conformerCentered[atom * 3 + 1];
    const Real pz   = atom < 0 ? Real(0) : conformerCentered[atom * 3 + 2];
    Real       moments[kNumSubsets][kNumUsrMoments];
    usrMoments<Real, kNumSubsets>(conformerCentered, classes, atoms.numAtoms, laneInGroup, px, py, pz, moments);
    if (laneInGroup == 0) {
      for (int subset = 0; subset < kNumSubsets; ++subset) {
        for (int moment = 0; moment < kNumUsrMoments; ++moment) {
          const int valueIdx = subset * kNumUsrProperties + point * kNumUsrMoments + moment;
          if (subset == 0 && usrRow != nullptr) {
            usrRow[valueIdx] = moments[subset][moment];
          }
          if constexpr (kUsrcat) {
            usrcatRow[valueIdx] = moments[subset][moment];
          }
        }
      }
    }
  }
}

//! Launches the reference-point kernel when USR or USRCAT is requested; USRCAT's first block is USR, so
//! requesting both costs one pass.
template <typename Real>
void launchReferencePointProperties(const DeviceCoordView&        coordinates,
                                    const Property3DDeviceInputs& inputs,
                                    Real*                         usrOutput,
                                    Real*                         usrcatOutput,
                                    const cudaStream_t            stream) {
  const int numConformers = coordinates.numConformers;
  if (numConformers == 0 || (usrOutput == nullptr && usrcatOutput == nullptr)) {
    return;
  }
  const int               numBlocks = (numConformers + kConformersPerBlock - 1) / kConformersPerBlock;
  AsyncDeviceVector<Real> centered(static_cast<size_t>(coordinates.numAtoms) * 3, stream);
  if (usrcatOutput != nullptr) {
    referencePoints3DKernel<Real, true>
      <<<numBlocks, kBlockSize, 0, stream>>>(coordinates, inputs, centered.data(), usrOutput, usrcatOutput);
  } else {
    referencePoints3DKernel<Real, false>
      <<<numBlocks, kBlockSize, 0, stream>>>(coordinates, inputs, centered.data(), usrOutput, nullptr);
  }
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_REFERENCE_CUH
