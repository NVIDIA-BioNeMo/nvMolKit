// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_KERNEL_CUH
#define NVMOLKIT_DESCRIPTORS3D_KERNEL_CUH

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

template <typename Real> __device__ __forceinline__ Real groupAllReduceSum(Real value) {
  const unsigned activeMask = __activemask();
  for (int offset = kGroupSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(activeMask, value, offset);
  }
  return value;
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_KERNEL_CUH
