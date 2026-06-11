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

#include "src/etkdg_stage_stereochem_checks.h"
#include "src/forcefields/kernel_utils.cuh"
#include "versions.h"
namespace nvMolKit {
namespace detail {

namespace {
constexpr double MIN_TETRAHEDRAL_CHIRAL_VOL = 0.50;

//! CUDA implementation of RDKit _sameSide in embedder.cpp
__device__ __forceinline__ bool _sameSide(const double tol,
                                          const double v1x,
                                          const double v1y,
                                          const double v1z,
                                          const double v2x,
                                          const double v2y,
                                          const double v2z,
                                          const double v3x,
                                          const double v3y,
                                          const double v3z,
                                          const double v4x,
                                          const double v4y,
                                          const double v4z,
                                          const double p0x,
                                          const double p0y,
                                          const double p0z) {
  double crossx, crossy, crossz;
  FFKernelUtils::crossProduct(v2x - v1x, v2y - v1y, v2z - v1z, v3x - v1x, v3y - v1y, v3z - v1z, crossx, crossy, crossz);
  const double d1 = FFKernelUtils::dotProduct(crossx, crossy, crossz, v4x - v1x, v4y - v1y, v4z - v1z);
  const double d2 = FFKernelUtils::dotProduct(crossx, crossy, crossz, p0x - v1x, p0y - v1y, p0z - v1z);
  if (fabs(d1) < tol || fabs(d2) < tol) {
    return false;
  }
  return !((d1 < 0.) ^ (d2 < 0.));
}

template <bool doVolumeTest>
__global__ void tetrahedralCheckKernel(const int            numTerms,
                                       const int            positionDimensionality,
                                       const double*        positions,
                                       const int*           idx0s,
                                       const int*           idx1s,
                                       const int*           idx2s,
                                       const int*           idx3s,
                                       const int*           idx4s,
                                       const std::uint64_t* structureFlags,
                                       const int*           sysIdxs,
                                       const uint8_t*       activeThisStage,
                                       uint8_t*             failedThisStage,
                                       const double         tol) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (termIdx >= numTerms) {
    return;
  }

  const int thisTermSysIdx = sysIdxs[termIdx];

  if (!activeThisStage[thisTermSysIdx]) {
    return;
  }

  const int    idx0 = idx0s[termIdx];
  const int    idx1 = idx1s[termIdx];
  const int    idx2 = idx2s[termIdx];
  const int    idx3 = idx3s[termIdx];
  const int    idx4 = idx4s[termIdx];
  // TODO: See if we can float.
  const double p0x  = positions[idx0 * positionDimensionality + 0];
  const double p0y  = positions[idx0 * positionDimensionality + 1];
  const double p0z  = positions[idx0 * positionDimensionality + 2];
  const double p1x  = positions[idx1 * positionDimensionality + 0];
  const double p1y  = positions[idx1 * positionDimensionality + 1];
  const double p1z  = positions[idx1 * positionDimensionality + 2];
  const double p2x  = positions[idx2 * positionDimensionality + 0];
  const double p2y  = positions[idx2 * positionDimensionality + 1];
  const double p2z  = positions[idx2 * positionDimensionality + 2];
  const double p3x  = positions[idx3 * positionDimensionality + 0];
  const double p3y  = positions[idx3 * positionDimensionality + 1];
  const double p3z  = positions[idx3 * positionDimensionality + 2];
  const double p4x  = positions[idx4 * positionDimensionality + 0];
  const double p4y  = positions[idx4 * positionDimensionality + 1];
  const double p4z  = positions[idx4 * positionDimensionality + 2];

  if constexpr (doVolumeTest) {
    double dx1 = p0x - p1x;
    double dy1 = p0y - p1y;
    double dz1 = p0z - p1z;
    double dx2 = p0x - p2x;
    double dy2 = p0y - p2y;
    double dz2 = p0z - p2z;
    double dx3 = p0x - p3x;
    double dy3 = p0y - p3y;
    double dz3 = p0z - p3z;
    double dx4 = p0x - p4x;
    double dy4 = p0y - p4y;
    double dz4 = p0z - p4z;

    const std::uint64_t structureFlag = structureFlags[termIdx];
    FFKernelUtils::normalizeVector(dx1, dy1, dz1);
    FFKernelUtils::normalizeVector(dx2, dy2, dz2);
    FFKernelUtils::normalizeVector(dx3, dy3, dz3);
    FFKernelUtils::normalizeVector(dx4, dy4, dz4);
    constexpr uint64_t IN_FUSED_SMALL_RINGS = 1 << 0;
    const double       volScale             = (structureFlag & IN_FUSED_SMALL_RINGS) ? 0.25 : 1.0;

    double crossx, crossy, crossz;

    // FIrst cross 1/2, dot 3
    FFKernelUtils::crossProduct(dx1, dy1, dz1, dx2, dy2, dz2, crossx, crossy, crossz);
    double vol = FFKernelUtils::dotProduct(crossx, crossy, crossz, dx3, dy3, dz3);
    if (fabs(vol) < volScale * MIN_TETRAHEDRAL_CHIRAL_VOL) {
      failedThisStage[thisTermSysIdx] = 1;
      return;
    }
    // Reuse 1/2 cross, dot 4
    vol = FFKernelUtils::dotProduct(crossx, crossy, crossz, dx4, dy4, dz4);
    if (fabs(vol) < volScale * MIN_TETRAHEDRAL_CHIRAL_VOL) {
      failedThisStage[thisTermSysIdx] = 1;
      return;
    }

    // Now compute 1/3 cross, dot 4
    FFKernelUtils::crossProduct(dx1, dy1, dz1, dx3, dy3, dz3, crossx, crossy, crossz);
    vol = FFKernelUtils::dotProduct(crossx, crossy, crossz, dx4, dy4, dz4);
    if (fabs(vol) < volScale * MIN_TETRAHEDRAL_CHIRAL_VOL) {
      failedThisStage[thisTermSysIdx] = 1;
      return;
    }

    // cross 2, 3, dot 4
    FFKernelUtils::crossProduct(dx2, dy2, dz2, dx3, dy3, dz3, crossx, crossy, crossz);
    vol = FFKernelUtils::dotProduct(crossx, crossy, crossz, dx4, dy4, dz4);
    if (fabs(vol) < volScale * MIN_TETRAHEDRAL_CHIRAL_VOL) {
      failedThisStage[thisTermSysIdx] = 1;
      return;
    }
  }

  // ------------------------
  // center in volume check
  // ------------------------
  // 3 coordinate centers.
  if (idx0 == idx4) {
    return;
  }
  // Same side check
  if (!_sameSide(tol, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, p4x, p4y, p4z, p0x, p0y, p0z)) {
    failedThisStage[thisTermSysIdx] = 1;
    return;
  }

  if (!_sameSide(tol, p2x, p2y, p2z, p3x, p3y, p3z, p4x, p4y, p4z, p1x, p1y, p1z, p0x, p0y, p0z)) {
    failedThisStage[thisTermSysIdx] = 1;
    return;
  }

  if (!_sameSide(tol, p3x, p3y, p3z, p4x, p4y, p4z, p1x, p1y, p1z, p2x, p2y, p2z, p0x, p0y, p0z)) {
    failedThisStage[thisTermSysIdx] = 1;
    return;
  }

  if (!_sameSide(tol, p4x, p4y, p4z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, p0x, p0y, p0z)) {
    failedThisStage[thisTermSysIdx] = 1;
  }
}

//! Direct port of RDKit calcChiralVolume in ChiralViolationsContrib.
__device__ __forceinline__ double calcChiralVolume(const double p1x,
                                                   const double p1y,
                                                   const double p1z,
                                                   const double p2x,
                                                   const double p2y,
                                                   const double p2z,
                                                   const double p3x,
                                                   const double p3y,
                                                   const double p3z,
                                                   const double p4x,
                                                   const double p4y,
                                                   const double p4z) {
  const double v1x = p1x - p4x;
  const double v1y = p1y - p4y;
  const double v1z = p1z - p4z;

  const double v2x = p2x - p4x;
  const double v2y = p2y - p4y;
  const double v2z = p2z - p4z;

  const double v3x = p3x - p4x;
  const double v3y = p3y - p4y;
  const double v3z = p3z - p4z;

  double crossx, crossy, crossz;
  FFKernelUtils::crossProduct(v2x, v2y, v2z, v3x, v3y, v3z, crossx, crossy, crossz);
  return FFKernelUtils::dotProduct(v1x, v1y, v1z, crossx, crossy, crossz);
}

//! Checks for sign equivalency between doubles. Probably undefined for 0/-0
__device__ __forceinline__ bool haveOppositeSign(const double a, const double b) {
  return std::signbit(a) ^ std::signbit(b);
}

//! Direct port of RDKit initial chirality check.
//! Note that idx0 and the central atom coordinates are not used.
__global__ void firstChiralCheckKernel(const int      numTerms,
                                       const int      positionDimensionality,
                                       const double*  positions,
                                       const int*     idx1s,
                                       const int*     idx2s,
                                       const int*     idx3s,
                                       const int*     idx4s,
                                       const int*     sysIdxs,
                                       const double*  volumeLowerBounds,
                                       const double*  volumeUpperBounds,
                                       const uint8_t* activeThisStage,
                                       uint8_t*       failedThisStage) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (termIdx >= numTerms) {
    return;
  }

  const int thisTermSysIdx = sysIdxs[termIdx];

  if (!activeThisStage[thisTermSysIdx]) {
    return;
  }
  const int    idx1 = idx1s[termIdx];
  const int    idx2 = idx2s[termIdx];
  const int    idx3 = idx3s[termIdx];
  const int    idx4 = idx4s[termIdx];
  // TODO: See if we can float.
  const double p1x  = positions[idx1 * positionDimensionality + 0];
  const double p1y  = positions[idx1 * positionDimensionality + 1];
  const double p1z  = positions[idx1 * positionDimensionality + 2];
  const double p2x  = positions[idx2 * positionDimensionality + 0];
  const double p2y  = positions[idx2 * positionDimensionality + 1];
  const double p2z  = positions[idx2 * positionDimensionality + 2];
  const double p3x  = positions[idx3 * positionDimensionality + 0];
  const double p3y  = positions[idx3 * positionDimensionality + 1];
  const double p3z  = positions[idx3 * positionDimensionality + 2];
  const double p4x  = positions[idx4 * positionDimensionality + 0];
  const double p4y  = positions[idx4 * positionDimensionality + 1];
  const double p4z  = positions[idx4 * positionDimensionality + 2];
  const double vol  = calcChiralVolume(p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, p4x, p4y, p4z);

  const double lb = volumeLowerBounds[termIdx];
  const double ub = volumeUpperBounds[termIdx];

  if ((lb > 0 && vol < lb && (vol / lb < .8 || haveOppositeSign(vol, lb))) ||
      (ub < 0 && vol > ub && (vol / ub < .8 || haveOppositeSign(vol, ub)))) {
    failedThisStage[thisTermSysIdx] = true;
  }
}

__global__ void chiralDistMatrixCheck(const int      numTerms,
                                      const int      positionDimensionality,
                                      const double*  positions,
                                      const int*     idx0s,
                                      const int*     idx1s,
                                      const double*  lowerBounds,
                                      const double*  upperBounds,
                                      const int*     sysIdxs,
                                      const uint8_t* activeThisStage,
                                      uint8_t*       failedThisStage) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (termIdx >= numTerms) {
    return;
  }

  const int thisTermSysIdx = sysIdxs[termIdx];

  if (!activeThisStage[thisTermSysIdx]) {
    return;
  }

  const int    idx0       = idx0s[termIdx];
  const int    idx1       = idx1s[termIdx];
  const double lowerBound = lowerBounds[termIdx];
  const double upperBound = upperBounds[termIdx];

  const double p0x = positions[idx0 * positionDimensionality + 0];
  const double p0y = positions[idx0 * positionDimensionality + 1];
  const double p0z = positions[idx0 * positionDimensionality + 2];
  const double p1x = positions[idx1 * positionDimensionality + 0];
  const double p1y = positions[idx1 * positionDimensionality + 1];
  const double p1z = positions[idx1 * positionDimensionality + 2];

  const double dx   = p0x - p1x;
  const double dy   = p0y - p1y;
  const double dz   = p0z - p1z;
  const double dist = sqrt(dx * dx + dy * dy + dz * dz);

  if (((dist < lowerBound) && (fabs(dist - lowerBound) > 0.1 * upperBound)) ||
      ((dist > upperBound) && (fabs(dist - upperBound) > 0.1 * upperBound))) {
    failedThisStage[thisTermSysIdx] = true;
  }
}

__global__ void doubleBondStereoKernel(const int      numTerms,
                                       const int      positionDimensionality,
                                       const double*  positions,
                                       const int*     idx0s,
                                       const int*     idx1s,
                                       const int*     idx2s,
                                       const int*     idx3s,
                                       const int*     signs,
                                       const int*     sysIdxs,
                                       const uint8_t* activeThisStage,
                                       uint8_t*       failedThisStage) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (termIdx >= numTerms) {
    return;
  }

  const int thisTermSysIdx = sysIdxs[termIdx];

  if (!activeThisStage[thisTermSysIdx]) {
    return;
  }

  const double p0x  = positions[idx0s[termIdx] * positionDimensionality + 0];
  const double p0y  = positions[idx0s[termIdx] * positionDimensionality + 1];
  const double p0z  = positions[idx0s[termIdx] * positionDimensionality + 2];
  const double p1x  = positions[idx1s[termIdx] * positionDimensionality + 0];
  const double p1y  = positions[idx1s[termIdx] * positionDimensionality + 1];
  const double p1z  = positions[idx1s[termIdx] * positionDimensionality + 2];
  const double p2x  = positions[idx2s[termIdx] * positionDimensionality + 0];
  const double p2y  = positions[idx2s[termIdx] * positionDimensionality + 1];
  const double p2z  = positions[idx2s[termIdx] * positionDimensionality + 2];
  const double p3x  = positions[idx3s[termIdx] * positionDimensionality + 0];
  const double p3y  = positions[idx3s[termIdx] * positionDimensionality + 1];
  const double p3z  = positions[idx3s[termIdx] * positionDimensionality + 2];
  const int    sign = signs[termIdx];

  const double dx1 = p2x - p1x;
  const double dy1 = p2y - p1y;
  const double dz1 = p2z - p1z;

  const double dx2 = p0x - p1x;
  const double dy2 = p0y - p1y;
  const double dz2 = p0z - p1z;

  const double dx3 = p3x - p2x;
  const double dy3 = p3y - p2y;
  const double dz3 = p3z - p2z;

  double cross1x, cross1y, cross1z;
  FFKernelUtils::crossProduct(dx2, dy2, dz2, dx1, dy1, dz1, cross1x, cross1y, cross1z);

  double cross2x, cross2y, cross2z;
  FFKernelUtils::crossProduct(dx3, dy3, dz3, dx1, dy1, dz1, cross2x, cross2y, cross2z);

  // Compute angle between two crosses.
  double       dot       = FFKernelUtils::dotProduct(cross1x, cross1y, cross1z, cross2x, cross2y, cross2z);
  const double l1squared = cross1x * cross1x + cross1y * cross1y + cross1z * cross1z;
  const double l2squared = cross2x * cross2x + cross2y * cross2y + cross2z * cross2z;
  const double denom     = sqrt(l1squared * l2squared);
  dot /= denom;

  double angle = acos(dot);
  if (dot <= -1.0) {
    angle = M_PI;
  } else if (dot >= 1.0) {
    angle = 0.0;
  }

  if (((angle - M_PI_2) * sign) < 0.0) {
    // Stereo is wrong.
    failedThisStage[thisTermSysIdx] = 1;
  }
}

__global__ void doubleBondGeometryKernel(const int      numTerms,
                                         const int      positionDimensionality,
                                         const double*  positions,
                                         const int*     idx0s,
                                         const int*     idx1s,
                                         const int*     idx2s,
                                         const int*     sysIdxs,
                                         const uint8_t* activeThisStage,
                                         uint8_t*       failedThisStage) {
  constexpr double linearTol = 1e-3;
  const int        termIdx   = blockIdx.x * blockDim.x + threadIdx.x;

  if (termIdx >= numTerms) {
    return;
  }

  const int thisTermSysIdx = sysIdxs[termIdx];

  if (!activeThisStage[thisTermSysIdx]) {
    return;
  }

  const double p0x = positions[idx0s[termIdx] * positionDimensionality + 0];
  const double p0y = positions[idx0s[termIdx] * positionDimensionality + 1];
  const double p0z = positions[idx0s[termIdx] * positionDimensionality + 2];
  const double p1x = positions[idx1s[termIdx] * positionDimensionality + 0];
  const double p1y = positions[idx1s[termIdx] * positionDimensionality + 1];
  const double p1z = positions[idx1s[termIdx] * positionDimensionality + 2];
  const double p2x = positions[idx2s[termIdx] * positionDimensionality + 0];
  const double p2y = positions[idx2s[termIdx] * positionDimensionality + 1];
  const double p2z = positions[idx2s[termIdx] * positionDimensionality + 2];

  double       dx1    = p1x - p0x;
  double       dy1    = p1y - p0y;
  double       dz1    = p1z - p0z;
  const double denom1 = sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);
  dx1 /= denom1;
  dy1 /= denom1;
  dz1 /= denom1;

  double       dx2    = p1x - p2x;
  double       dy2    = p1y - p2y;
  double       dz2    = p1z - p2z;
  const double denom2 = sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);
  dx2 /= denom2;
  dy2 /= denom2;
  dz2 /= denom2;

  const double dot = FFKernelUtils::dotProduct(dx1, dy1, dz1, dx2, dy2, dz2);
  if ((dot + 1.0) < linearTol) {
    failedThisStage[thisTermSysIdx] = 1;
  }
}
}  // namespace

void ETKDGChiralCheckBase::loadChiralDataset(const ETKDGContext&           ctx,
                                             const std::vector<EmbedArgs>& eargs,
                                             ChiralCheckType               checkType) {
  std::vector<int>           idx0Host;
  std::vector<int>           idx1Host;
  std::vector<int>           idx2Host;
  std::vector<int>           idx3Host;
  std::vector<int>           idx4Host;
  std::vector<double>        volLowerHost;
  std::vector<double>        volUpperHost;
  std::vector<std::uint64_t> structureFlagsHost;
  std::vector<int>           sysIdxHost;

  for (int i = 0; i < ctx.nTotalSystems; ++i) {
    const auto& embedArg        = eargs[i];
    const int   atomIndexOffset = ctx.systemHost.atomStarts[i];
    const auto& checklist =
      (checkType == ChiralCheckType::Tetrahedral) ? embedArg.tetrahedralCarbons : embedArg.chiralCenters;
    for (const auto& check : checklist) {
      idx0Host.push_back(check->d_idx0 + atomIndexOffset);
      idx1Host.push_back(check->d_idx1 + atomIndexOffset);
      idx2Host.push_back(check->d_idx2 + atomIndexOffset);
      idx3Host.push_back(check->d_idx3 + atomIndexOffset);
      idx4Host.push_back(check->d_idx4 + atomIndexOffset);
#if RDKIT_NEW_FLAG_API
      structureFlagsHost.push_back(check->d_structureFlags);
#else
      structureFlagsHost.push_back(0);
#endif
      sysIdxHost.push_back(i);

      // Terms only used in one chiral check or another.
      if (checkType == ChiralCheckType::Chiral) {
        volLowerHost.push_back(check->d_volumeLowerBound);
        volUpperHost.push_back(check->d_volumeUpperBound);
      }
    }
  }

  idx0.setFromVector(idx0Host);
  idx1.setFromVector(idx1Host);
  idx2.setFromVector(idx2Host);
  idx3.setFromVector(idx3Host);
  idx4.setFromVector(idx4Host);
  sysIdx.setFromVector(sysIdxHost);

  if (checkType == ChiralCheckType::Chiral) {
    volumeLowerBound.setFromVector(volLowerHost);
    volumeUpperBound.setFromVector(volUpperHost);
  } else {
    structureFlags.setFromVector(structureFlagsHost);
  }
  cudaStreamSynchronize(idx0.stream());  // Sync before local vectors go out of scope
}

void ETKDGChiralCheckBase::setStreams(const cudaStream_t stream) {
  idx0.setStream(stream);
  idx1.setStream(stream);
  idx2.setStream(stream);
  idx3.setStream(stream);
  idx4.setStream(stream);
  sysIdx.setStream(stream);
  structureFlags.setStream(stream);
  volumeLowerBound.setStream(stream);
  volumeUpperBound.setStream(stream);
}

ETKDGTetrahedralCheckStage::ETKDGTetrahedralCheckStage(const ETKDGContext&           ctx,
                                                       const std::vector<EmbedArgs>& eargs,
                                                       const int                     dim,
                                                       cudaStream_t                  stream)
    : dim_(dim),
      stream_(stream) {
  setStreams(stream);
  loadChiralDataset(ctx, eargs, ChiralCheckType::Tetrahedral);
}

void ETKDGTetrahedralCheckStage::execute(ETKDGContext& ctx) {
  const int numSystems = ctx.nTotalSystems;
  if (numSystems == 0 || idx0.size() == 0) {
    return;
  }

  constexpr int blockSize = 128;
  const int     numBlocks = (idx0.size() + blockSize - 1) / blockSize;
  tetrahedralCheckKernel</*doVolumeTest=*/true><<<numBlocks, blockSize, 0, stream_>>>(idx0.size(),
                                                                                      dim_,
                                                                                      ctx.systemDevice.positions.data(),
                                                                                      idx0.data(),
                                                                                      idx1.data(),
                                                                                      idx2.data(),
                                                                                      idx3.data(),
                                                                                      idx4.data(),
                                                                                      structureFlags.data(),
                                                                                      sysIdx.data(),
                                                                                      ctx.activeThisStage.data(),
                                                                                      ctx.failedThisStage.data(),
                                                                                      volCheckTol_);
  cudaCheckError(cudaGetLastError());
}

ETKDGFirstChiralCenterCheckStage::ETKDGFirstChiralCenterCheckStage(const ETKDGContext&           ctx,
                                                                   const std::vector<EmbedArgs>& eargs,
                                                                   const int                     dim,
                                                                   cudaStream_t                  stream)
    : dim_(dim),
      stream_(stream) {
  setStreams(stream);
  loadChiralDataset(ctx, eargs, ChiralCheckType::Chiral);
}

void ETKDGFirstChiralCenterCheckStage::execute(ETKDGContext& ctx) {
  const int numSystems = ctx.nTotalSystems;
  if (numSystems == 0 || idx0.size() == 0) {
    return;
  }

  constexpr int blockSize = 128;
  const int     numBlocks = (idx0.size() + blockSize - 1) / blockSize;
  firstChiralCheckKernel<<<numBlocks, blockSize, 0, stream_>>>(idx0.size(),
                                                               dim_,
                                                               ctx.systemDevice.positions.data(),
                                                               idx1.data(),
                                                               idx2.data(),
                                                               idx3.data(),
                                                               idx4.data(),
                                                               sysIdx.data(),
                                                               volumeLowerBound.data(),
                                                               volumeUpperBound.data(),
                                                               ctx.activeThisStage.data(),
                                                               ctx.failedThisStage.data());
  cudaCheckError(cudaGetLastError());
}

ETKDGChiralCenterVolumeCheckStage::ETKDGChiralCenterVolumeCheckStage(const ETKDGContext&           ctx,
                                                                     const std::vector<EmbedArgs>& eargs,
                                                                     const int                     dim,
                                                                     cudaStream_t                  stream)
    : dim_(dim),
      stream_(stream) {
  // TODO: We're now loading the same dataset multiple times and could figure out a shared reference system.
  setStreams(stream);
  loadChiralDataset(ctx, eargs, ChiralCheckType::Chiral);
}

void ETKDGChiralCenterVolumeCheckStage::execute(ETKDGContext& ctx) {
  const int numSystems = ctx.nTotalSystems;
  if (numSystems == 0 || idx0.size() == 0) {
    return;
  }

  constexpr int blockSize = 128;
  const int     numBlocks = (idx0.size() + blockSize - 1) / blockSize;
  tetrahedralCheckKernel</*doVolumeTest=*/false>
    <<<numBlocks, blockSize, 0, stream_>>>(idx0.size(),
                                           dim_,
                                           ctx.systemDevice.positions.data(),
                                           idx0.data(),
                                           idx1.data(),
                                           idx2.data(),
                                           idx3.data(),
                                           idx4.data(),
                                           structureFlags.data(),
                                           sysIdx.data(),
                                           ctx.activeThisStage.data(),
                                           ctx.failedThisStage.data(),
                                           volCheckTol_);
  cudaCheckError(cudaGetLastError());
}

void ETKDGChiralDistMatrixCheckStage::loadDataset(const ETKDGContext& ctx, const std::vector<EmbedArgs>& eargs) {
  std::vector<int>    idx0Host;
  std::vector<int>    idx1Host;
  std::vector<double> lowerBound;
  std::vector<double> upperBound;
  std::vector<int>    sysIdxHost;

  for (int i = 0; i < ctx.nTotalSystems; ++i) {
    const auto&   mmat            = *eargs[i].mmat;
    const int     atomIndexOffset = ctx.systemHost.atomStarts[i];
    const auto&   checklist       = eargs[i].chiralCenters;
    std::set<int> chiralIdxs;

    for (const auto& check : checklist) {
      if (check->d_idx0 == check->d_idx4) {
        continue;
      }  // Push each interaction pair.
      chiralIdxs.insert(check->d_idx0);
      chiralIdxs.insert(check->d_idx1);
      chiralIdxs.insert(check->d_idx2);
      chiralIdxs.insert(check->d_idx3);
      chiralIdxs.insert(check->d_idx4);
    }
    if (chiralIdxs.size() == 0) {
      continue;
    }
    std::vector<int> chiralIdxsVec(chiralIdxs.begin(), chiralIdxs.end());
    int              count = 0;
    for (size_t j = 0; j < chiralIdxsVec.size() - 1; ++j) {
      for (size_t k = j + 1; k < chiralIdxsVec.size(); ++k) {
        // We need to add all pairs of chiral centers.
        const int idx0 = chiralIdxsVec[j];
        const int idx1 = chiralIdxsVec[k];
        idx0Host.push_back(idx0 + atomIndexOffset);
        idx1Host.push_back(idx1 + atomIndexOffset);
        lowerBound.push_back(mmat.getLowerBound(idx0, idx1));
        upperBound.push_back(mmat.getUpperBound(idx0, idx1));
        count++;
      }
    }
    sysIdxHost.resize(sysIdxHost.size() + count, i);  // Add the system index for each pair.
  }

  idx0.setFromVector(idx0Host);
  idx1.setFromVector(idx1Host);
  sysIdx.setFromVector(sysIdxHost);
  matLowerBound.setFromVector(lowerBound);
  matUpperBound.setFromVector(upperBound);
  cudaStreamSynchronize(idx0.stream());  // Sync before local vectors go out of scope
}

ETKDGChiralDistMatrixCheckStage::ETKDGChiralDistMatrixCheckStage(const ETKDGContext&           ctx,
                                                                 const std::vector<EmbedArgs>& eargs,
                                                                 const int                     dim,
                                                                 cudaStream_t                  stream)
    : dim_(dim),
      stream_(stream) {
  idx0.setStream(stream);
  idx1.setStream(stream);
  sysIdx.setStream(stream);
  matLowerBound.setStream(stream);
  matUpperBound.setStream(stream);
  loadDataset(ctx, eargs);
}

void ETKDGChiralDistMatrixCheckStage::execute(ETKDGContext& ctx) {
  const int numSystems = ctx.nTotalSystems;
  if (numSystems == 0 || idx0.size() == 0) {
    return;
  }

  constexpr int blockSize = 128;
  const int     numBlocks = (idx0.size() + blockSize - 1) / blockSize;
  chiralDistMatrixCheck<<<numBlocks, blockSize, 0, stream_>>>(idx0.size(),
                                                              dim_,
                                                              ctx.systemDevice.positions.data(),
                                                              idx0.data(),
                                                              idx1.data(),
                                                              matLowerBound.data(),
                                                              matUpperBound.data(),
                                                              sysIdx.data(),
                                                              ctx.activeThisStage.data(),
                                                              ctx.failedThisStage.data());
  cudaCheckError(cudaGetLastError());
}

ETKDGDoubleBondStereoCheckStage::ETKDGDoubleBondStereoCheckStage(const ETKDGContext&           ctx,
                                                                 const std::vector<EmbedArgs>& eargs,
                                                                 int                           dim,
                                                                 cudaStream_t                  stream)
    : dim_(dim),
      stream_(stream) {
  idx0.setStream(stream);
  idx1.setStream(stream);
  idx2.setStream(stream);
  idx3.setStream(stream);
  signs.setStream(stream);
  sysIdx.setStream(stream);

  loadDataset(ctx, eargs);
}

void ETKDGDoubleBondStereoCheckStage::execute(ETKDGContext& ctx) {
  const int numSystems = ctx.nTotalSystems;
  if (numSystems == 0 || idx0.size() == 0) {
    return;
  }

  constexpr int blockSize = 128;
  const int     numBlocks = (idx0.size() + blockSize - 1) / blockSize;
  doubleBondStereoKernel<<<numBlocks, blockSize, 0, stream_>>>(idx0.size(),
                                                               dim_,
                                                               ctx.systemDevice.positions.data(),
                                                               idx0.data(),
                                                               idx1.data(),
                                                               idx2.data(),
                                                               idx3.data(),
                                                               signs.data(),
                                                               sysIdx.data(),
                                                               ctx.activeThisStage.data(),
                                                               ctx.failedThisStage.data());
  cudaCheckError(cudaGetLastError());
}

void ETKDGDoubleBondStereoCheckStage::loadDataset(const ETKDGContext& ctx, const std::vector<EmbedArgs>& eargs) {
  std::vector<int> idx0Host;
  std::vector<int> idx1Host;
  std::vector<int> idx2Host;
  std::vector<int> idx3Host;
  std::vector<int> signsHost;
  std::vector<int> sysIdxHost;

  for (size_t i = 0; i < eargs.size(); i++) {
    for (const auto& [indices, sign] : eargs[i].stereoDoubleBonds) {
      if (indices.size() != 4) {
        throw std::runtime_error("Double bond stereo check requires 4 indices, got" + std::to_string(indices.size()));
      }
      idx0Host.push_back(indices[0] + ctx.systemHost.atomStarts[i]);
      idx1Host.push_back(indices[1] + ctx.systemHost.atomStarts[i]);
      idx2Host.push_back(indices[2] + ctx.systemHost.atomStarts[i]);
      idx3Host.push_back(indices[3] + ctx.systemHost.atomStarts[i]);
      signsHost.push_back(sign);
      sysIdxHost.push_back(i);
    }
  }

  idx0.setFromVector(idx0Host);
  idx1.setFromVector(idx1Host);
  idx2.setFromVector(idx2Host);
  idx3.setFromVector(idx3Host);
  signs.setFromVector(signsHost);
  sysIdx.setFromVector(sysIdxHost);
  cudaStreamSynchronize(idx0.stream());  // Sync before local vectors go out of scope
}

ETKDGDoubleBondGeometryCheckStage::ETKDGDoubleBondGeometryCheckStage(const ETKDGContext&           ctx,
                                                                     const std::vector<EmbedArgs>& eargs,
                                                                     int                           dim,
                                                                     cudaStream_t                  stream)
    : dim_(dim),
      stream_(stream) {
  idx0.setStream(stream);
  idx1.setStream(stream);
  idx2.setStream(stream);
  sysIdx.setStream(stream);
  loadDataset(ctx, eargs);
}

void ETKDGDoubleBondGeometryCheckStage::execute(ETKDGContext& ctx) {
  const int numSystems = ctx.nTotalSystems;
  if (numSystems == 0 || idx0.size() == 0) {
    return;
  }

  constexpr int blockSize = 128;
  const int     numBlocks = (idx0.size() + blockSize - 1) / blockSize;
  doubleBondGeometryKernel<<<numBlocks, blockSize, 0, stream_>>>(idx0.size(),
                                                                 dim_,
                                                                 ctx.systemDevice.positions.data(),
                                                                 idx0.data(),
                                                                 idx1.data(),
                                                                 idx2.data(),
                                                                 sysIdx.data(),
                                                                 ctx.activeThisStage.data(),
                                                                 ctx.failedThisStage.data());
  cudaCheckError(cudaGetLastError());
}
void ETKDGDoubleBondGeometryCheckStage::loadDataset(const ETKDGContext& ctx, const std::vector<EmbedArgs>& eargs) {
  std::vector<int> idx0Host;
  std::vector<int> idx1Host;
  std::vector<int> idx2Host;
  std::vector<int> sysIdxHost;

  for (size_t i = 0; i < eargs.size(); i++) {
    for (const auto& [idx0, idx1, idx2] : eargs[i].doubleBondEnds) {
      idx0Host.push_back(idx0 + ctx.systemHost.atomStarts[i]);
      idx1Host.push_back(idx1 + ctx.systemHost.atomStarts[i]);
      idx2Host.push_back(idx2 + ctx.systemHost.atomStarts[i]);
      sysIdxHost.push_back(i);
    }
  }

  idx0.setFromVector(idx0Host);
  idx1.setFromVector(idx1Host);
  idx2.setFromVector(idx2Host);
  sysIdx.setFromVector(sysIdxHost);
  cudaStreamSynchronize(idx0.stream());  // Sync before local vectors go out of scope
}

}  // namespace detail
}  // namespace nvMolKit
