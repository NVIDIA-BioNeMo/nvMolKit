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

#ifndef NVMOLKIT_DISTGEOM_KERNELS_DEVICE_CUH
#define NVMOLKIT_DISTGEOM_KERNELS_DEVICE_CUH

#include <cooperative_groups.h>

#include "src/forcefields/dist_geom_kernels.h"
#include "src/forcefields/kernel_utils.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
constexpr double RAD2DEG = 180.0 / M_PI;

using namespace nvMolKit::FFKernelUtils;

namespace nvMolKit {
namespace DistGeom {

// --------------
// DG terms
// --------------

template <int dimension>
static __device__ __forceinline__ double distViolationEnergy(const double* pos,
                                                             const int     idx1,
                                                             const int     idx2,
                                                             const double  lb2,
                                                             const double  ub2,
                                                             const double  weight) {
  const int    posIdx1   = idx1 * dimension;
  const int    posIdx2   = idx2 * dimension;
  const double distance2 = distanceSquaredPosIdx<dimension>(pos, posIdx1, posIdx2);
  double       val       = 0.0;
  if (distance2 > ub2) {
    val = (distance2 / ub2) - 1.0;
  } else if (distance2 < lb2) {
    val = ((2 * lb2) / (lb2 + distance2)) - 1.0;
  }
  if (val > 0.0) {
    return weight * val * val;
  }
  return 0.0;
}

template <int dimension>
static __device__ __forceinline__ void distViolationGrad(const double* pos,
                                                         const int     idx1,
                                                         const int     idx2,
                                                         const double  lb2,
                                                         const double  ub2,
                                                         const double  weight,
                                                         double*       grad) {
  const int   posIdx1   = idx1 * dimension;
  const int   posIdx2   = idx2 * dimension;
  const float distance2 = distanceSquaredPosIdx<dimension>(pos, posIdx1, posIdx2);
  float       preFactor = 0.0;
  if (distance2 > ub2) {
    preFactor = 4.f * ((distance2 / ub2) - 1.0f) / ub2;
  } else if (distance2 < lb2) {
    const float l2d2 = distance2 + lb2;
    preFactor        = 8.f * lb2 * (1.f - 2.0f * lb2 / l2d2) / (l2d2 * l2d2);
  } else {
    return;
  }
  const float dGradx = weight * preFactor * (pos[posIdx1 + 0] - pos[posIdx2 + 0]);
  const float dGrady = weight * preFactor * (pos[posIdx1 + 1] - pos[posIdx2 + 1]);
  const float dGradz = weight * preFactor * (pos[posIdx1 + 2] - pos[posIdx2 + 2]);

  atomicAdd(&grad[posIdx1 + 0], dGradx);
  atomicAdd(&grad[posIdx1 + 1], dGrady);
  atomicAdd(&grad[posIdx1 + 2], dGradz);
  atomicAdd(&grad[posIdx2 + 0], -dGradx);
  atomicAdd(&grad[posIdx2 + 1], -dGrady);
  atomicAdd(&grad[posIdx2 + 2], -dGradz);

  if constexpr (dimension == 4) {
    const float dGradw = weight * preFactor * (pos[posIdx1 + 3] - pos[posIdx2 + 3]);
    atomicAdd(&grad[posIdx1 + 3], dGradw);
    atomicAdd(&grad[posIdx2 + 3], -dGradw);
  }
}

template <typename T>
static __device__ __forceinline__ T calcChiralVolume(const int&    posIdx1,
                                                     const int&    posIdx2,
                                                     const int&    posIdx3,
                                                     const int&    posIdx4,
                                                     const double* pos,
                                                     T&            v1x,
                                                     T&            v1y,
                                                     T&            v1z,
                                                     T&            v2x,
                                                     T&            v2y,
                                                     T&            v2z,
                                                     T&            v3x,
                                                     T&            v3y,
                                                     T&            v3z) {
  v1x = pos[posIdx1 + 0] - pos[posIdx4 + 0];
  v1y = pos[posIdx1 + 1] - pos[posIdx4 + 1];
  v1z = pos[posIdx1 + 2] - pos[posIdx4 + 2];

  v2x = pos[posIdx2 + 0] - pos[posIdx4 + 0];
  v2y = pos[posIdx2 + 1] - pos[posIdx4 + 1];
  v2z = pos[posIdx2 + 2] - pos[posIdx4 + 2];

  v3x = pos[posIdx3 + 0] - pos[posIdx4 + 0];
  v3y = pos[posIdx3 + 1] - pos[posIdx4 + 1];
  v3z = pos[posIdx3 + 2] - pos[posIdx4 + 2];

  T v2v3x, v2v3y, v2v3z;
  crossProduct(v2x, v2y, v2z, v3x, v3y, v3z, v2v3x, v2v3y, v2v3z);
  T vol = dotProduct(v1x, v1y, v1z, v2v3x, v2v3y, v2v3z);
  return vol;
}

template <int dimension>
static __device__ __forceinline__ double chiralViolationEnergy(const double* pos,
                                                               const int     idx1,
                                                               const int     idx2,
                                                               const int     idx3,
                                                               const int     idx4,
                                                               const double  lb,
                                                               const double  ub,
                                                               const double  weight) {
  const int posIdx1 = idx1 * dimension;
  const int posIdx2 = idx2 * dimension;
  const int posIdx3 = idx3 * dimension;
  const int posIdx4 = idx4 * dimension;

  double v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
  // Using the float version of this causes drift on the order of ~10^-5, so stay in double precision.
  double vol = calcChiralVolume(posIdx1, posIdx2, posIdx3, posIdx4, pos, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z);

  if (vol < lb) {
    return weight * (vol - lb) * (vol - lb);
  }
  if (vol > ub) {
    return weight * (vol - ub) * (vol - ub);
  }
  return 0.0;
}

template <int dimension>
static __device__ __forceinline__ void chiralViolationGrad(const double* pos,
                                                           const int     idx1,
                                                           const int     idx2,
                                                           const int     idx3,
                                                           const int     idx4,
                                                           const double  lb,
                                                           const double  ub,
                                                           const double  weight,
                                                           double*       grad) {
  const int posIdx1 = idx1 * dimension;
  const int posIdx2 = idx2 * dimension;
  const int posIdx3 = idx3 * dimension;
  const int posIdx4 = idx4 * dimension;

  float v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
  float vol = calcChiralVolume(posIdx1, posIdx2, posIdx3, posIdx4, pos, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z);

  if (vol < lb || vol > ub) {
    float preFactor;
    if (vol < lb) {
      preFactor = weight * (vol - lb);
    } else {
      preFactor = weight * (vol - ub);
    }

    atomicAdd(&grad[posIdx1 + 0], preFactor * (v2y * v3z - v2z * v3y));
    atomicAdd(&grad[posIdx1 + 1], preFactor * (v2z * v3x - v2x * v3z));
    atomicAdd(&grad[posIdx1 + 2], preFactor * (v2x * v3y - v2y * v3x));

    atomicAdd(&grad[posIdx2 + 0], preFactor * (v3y * v1z - v3z * v1y));
    atomicAdd(&grad[posIdx2 + 1], preFactor * (v3z * v1x - v3x * v1z));
    atomicAdd(&grad[posIdx2 + 2], preFactor * (v3x * v1y - v3y * v1x));

    atomicAdd(&grad[posIdx3 + 0], preFactor * (v2z * v1y - v2y * v1z));
    atomicAdd(&grad[posIdx3 + 1], preFactor * (v2x * v1z - v2z * v1x));
    atomicAdd(&grad[posIdx3 + 2], preFactor * (v2y * v1x - v2x * v1y));

    float x1 = pos[posIdx1 + 0];
    float y1 = pos[posIdx1 + 1];
    float z1 = pos[posIdx1 + 2];
    float x2 = pos[posIdx2 + 0];
    float y2 = pos[posIdx2 + 1];
    float z2 = pos[posIdx2 + 2];
    float x3 = pos[posIdx3 + 0];
    float y3 = pos[posIdx3 + 1];
    float z3 = pos[posIdx3 + 2];
    atomicAdd(&grad[posIdx4 + 0], preFactor * (z1 * (y2 - y3) + z2 * (y3 - y1) + z3 * (y1 - y2)));
    atomicAdd(&grad[posIdx4 + 1], preFactor * (x1 * (z2 - z3) + x2 * (z3 - z1) + x3 * (z1 - z2)));
    atomicAdd(&grad[posIdx4 + 2], preFactor * (y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)));
  }
}

template <int dimension>
static __device__ __forceinline__ double fourthDimEnergy(const double* pos, const int idx, const double weight) {
  if constexpr (dimension != 4) {
    return 0.0;
  }
  const int    posIdx    = idx * dimension;
  const double fourthVal = pos[posIdx + 3];
  return weight * fourthVal * fourthVal;
}

template <int dimension>
static __device__ __forceinline__ void fourthDimGrad(const double* pos,
                                                     const int     idx,
                                                     const double  weight,
                                                     double*       grad) {
  if constexpr (dimension != 4) {
    return;
  }
  const int    posIdx    = idx * dimension;
  const double fourthVal = pos[posIdx + 3];
  atomicAdd(&grad[posIdx + 3], weight * fourthVal);
}

// ----------------------
// ETK Terms
// ----------------------

static __device__ __forceinline__ float calcTorsionEnergyM6(const double* forceConstants,
                                                            const int*    signs,
                                                            const double  cosPhi) {
  const float cosPhi2 = cosPhi * cosPhi;
  const float cosPhi3 = cosPhi * cosPhi2;
  const float cosPhi4 = cosPhi * cosPhi3;
  const float cosPhi5 = cosPhi * cosPhi4;
  const float cosPhi6 = cosPhi * cosPhi5;

  const float cos2Phi = 2.0f * cosPhi2 - 1.0f;
  const float cos3Phi = 4.0f * cosPhi3 - 3.0f * cosPhi;
  const float cos4Phi = 8.0f * cosPhi4 - 8.0f * cosPhi2 + 1.0f;
  const float cos5Phi = 16.0f * cosPhi5 - 20.0f * cosPhi3 + 5.0f * cosPhi;
  const float cos6Phi = 32.0f * cosPhi6 - 48.0f * cosPhi4 + 18.0f * cosPhi2 - 1.0f;

  return (forceConstants[0] * (1.0f + signs[0] * cosPhi) + forceConstants[1] * (1.0f + signs[1] * cos2Phi) +
          forceConstants[2] * (1.0f + signs[2] * cos3Phi) + forceConstants[3] * (1.0f + signs[3] * cos4Phi) +
          forceConstants[4] * (1.0f + signs[4] * cos5Phi) + forceConstants[5] * (1.0f + signs[5] * cos6Phi));
}

static __device__ __forceinline__ double calcTorsionCosPhi(const double* pos,
                                                           const int     posIdx1,
                                                           const int     posIdx2,
                                                           const int     posIdx3,
                                                           const int     posIdx4) {
  double r1x = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  double r1y = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  double r1z = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  double r2x = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  double r2y = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  double r2z = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  double r3x = pos[posIdx2 + 0] - pos[posIdx3 + 0];
  double r3y = pos[posIdx2 + 1] - pos[posIdx3 + 1];
  double r3z = pos[posIdx2 + 2] - pos[posIdx3 + 2];

  double r4x = pos[posIdx4 + 0] - pos[posIdx3 + 0];
  double r4y = pos[posIdx4 + 1] - pos[posIdx3 + 1];
  double r4z = pos[posIdx4 + 2] - pos[posIdx3 + 2];

  double t1x, t1y, t1z;
  crossProduct(r1x, r1y, r1z, r2x, r2y, r2z, t1x, t1y, t1z);

  double t2x, t2y, t2z;
  crossProduct(r3x, r3y, r3z, r4x, r4y, r4z, t2x, t2y, t2z);

  const double t1_lenSquared      = t1x * t1x + t1y * t1y + t1z * t1z;
  const double t2_lenSquared      = t2x * t2x + t2y * t2y + t2z * t2z;
  const double lenSquaredCombined = t1_lenSquared * t2_lenSquared;

  if (isDoubleZero(lenSquaredCombined)) {
    return 0.0;
  }
  const double invLenComb = rsqrtf(lenSquaredCombined);
  double       cosPhi     = dotProduct(t1x, t1y, t1z, t2x, t2y, t2z) * invLenComb;
  clipToOne(cosPhi);
  return cosPhi;
}

static __device__ __forceinline__ double torsionAngleEnergy(const double* pos,
                                                            const int     idx1,
                                                            const int     idx2,
                                                            const int     idx3,
                                                            const int     idx4,
                                                            const double* forceConstants,
                                                            const int*    signs) {
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;
  const int posIdx4 = idx4 * 4;

  double cosPhi = calcTorsionCosPhi(pos, posIdx1, posIdx2, posIdx3, posIdx4);
  return calcTorsionEnergyM6(forceConstants, signs, cosPhi);
}

static __device__ __forceinline__ float calcInversionCosY(const double* pos,
                                                          const int     posIdx1,
                                                          const int     posIdx2,
                                                          const int     posIdx3,
                                                          const int     posIdx4) {
  constexpr float inversionZeroTol = 1.0e-16f;

  float rJIx = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  float rJIy = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  float rJIz = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  float rJKx = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  float rJKy = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  float rJKz = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  float rJLx = pos[posIdx4 + 0] - pos[posIdx2 + 0];
  float rJLy = pos[posIdx4 + 1] - pos[posIdx2 + 1];
  float rJLz = pos[posIdx4 + 2] - pos[posIdx2 + 2];

  float l2JI = rJIx * rJIx + rJIy * rJIy + rJIz * rJIz;
  float l2JK = rJKx * rJKx + rJKy * rJKy + rJKz * rJKz;
  float l2JL = rJLx * rJLx + rJLy * rJLy + rJLz * rJLz;

  if (l2JI < inversionZeroTol || l2JK < inversionZeroTol || l2JL < inversionZeroTol) {
    return 0.0f;
  }

  float nx, ny, nz;
  crossProduct(rJIx, rJIy, rJIz, rJKx, rJKy, rJKz, nx, ny, nz);

  const float norm_factor = rsqrtf(l2JI * l2JK);
  nx *= norm_factor;
  ny *= norm_factor;
  nz *= norm_factor;

  float l2n = nx * nx + ny * ny + nz * nz;
  if (l2n < inversionZeroTol) {
    return 0.0f;
  }

  return dotProduct(nx, ny, nz, rJLx, rJLy, rJLz) * rsqrtf(l2JL) * rsqrtf(l2n);
}

static __device__ __forceinline__ double inversionEnergy(const double* pos,
                                                         const int     idx1,
                                                         const int     idx2,
                                                         const int     idx3,
                                                         const int     idx4,
                                                         const double  C0,
                                                         const double  C1,
                                                         const double  C2,
                                                         const double  forceConstant) {
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;
  const int posIdx4 = idx4 * 4;

  double cosY = calcInversionCosY(pos, posIdx1, posIdx2, posIdx3, posIdx4);

  const double sinYSq = 1.0 - cosY * cosY;
  const double sinY   = ((sinYSq > 0.0) ? sqrtf(sinYSq) : 0.0);
  const double cos2W  = 2.0 * sinY * sinY - 1.0;

  return forceConstant * (C0 + C1 * sinY + C2 * cos2W);
}

static __device__ __forceinline__ double distanceConstraintEnergy(const double* pos,
                                                                  const int     idx1,
                                                                  const int     idx2,
                                                                  const double  minLen,
                                                                  const double  maxLen,
                                                                  const double  forceConstant) {
  const int    posIdx1   = idx1 * 4;
  const int    posIdx2   = idx2 * 4;
  const double distance2 = distanceSquaredPosIdx(pos, posIdx1, posIdx2, 3);

  const double minLen2 = minLen * minLen;
  const double maxLen2 = maxLen * maxLen;

  double difference = 0.0;
  if (distance2 < minLen2) {
    difference = minLen - sqrtf(distance2);
  } else if (distance2 > maxLen2) {
    difference = sqrtf(distance2) - maxLen;
  } else {
    return 0.0;
  }

  return 0.5 * forceConstant * difference * difference;
}

static __device__ __forceinline__ double computeAngleTerm(const double angle,
                                                          const double minAngle,
                                                          const double maxAngle) {
  double angleTerm = 0.0;
  if (angle < minAngle) {
    angleTerm = angle - minAngle;
  } else if (angle > maxAngle) {
    angleTerm = angle - maxAngle;
  }
  return angleTerm;
}

static __device__ __forceinline__ double angleConstraintEnergy(const double* pos,
                                                               const int     idx1,
                                                               const int     idx2,
                                                               const int     idx3,
                                                               const double  minAngle,
                                                               const double  maxAngle,
                                                               const double  forceConstant) {
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;

  double dx1 = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  double dy1 = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  double dz1 = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  double dx2 = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  double dy2 = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  double dz2 = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  const double dist1Sq  = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
  const double dist2Sq  = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
  const double distTerm = dist1Sq * dist2Sq;
  if (isDoubleZero(distTerm)) {
    return 0.0;
  }

  const double dot      = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  // This double precision sqrt is sensitive, can't downcast.
  const double cosTheta = clamp(dot * rsqrt(distTerm), -1.0, 1.0);
  const double angle    = RAD2DEG * acos(cosTheta);

  const double angleTerm = computeAngleTerm(angle, minAngle, maxAngle);
  return forceConstant * angleTerm * angleTerm;
}

static __device__ __forceinline__ void torsionAngleGrad(const double* pos,
                                                        const int     idx1,
                                                        const int     idx2,
                                                        const int     idx3,
                                                        const int     idx4,
                                                        const double* forceConstants,  // 6 components
                                                        const int*    signs,           // 6 components
                                                        double*       grad) {
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;
  const int posIdx4 = idx4 * 4;

  // Calculate bond vectors r1 = p1-p2, r2 = p3-p2
  const double r1x = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  const double r1y = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  const double r1z = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  const double r2x = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  const double r2y = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  const double r2z = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  // Calculate bond vectors r3 = p2-p3, r4 = p4-p3
  const double r3x = -r2x;
  const double r3y = -r2y;
  const double r3z = -r2z;

  const double r4x = pos[posIdx4 + 0] - pos[posIdx3 + 0];
  const double r4y = pos[posIdx4 + 1] - pos[posIdx3 + 1];
  const double r4z = pos[posIdx4 + 2] - pos[posIdx3 + 2];

  // Calculate plane normals via cross products: t0 = r1 × r2, t1 = r3 × r4
  double t0x, t0y, t0z;
  crossProduct(r1x, r1y, r1z, r2x, r2y, r2z, t0x, t0y, t0z);

  double t1x, t1y, t1z;
  crossProduct(r3x, r3y, r3z, r4x, r4y, r4z, t1x, t1y, t1z);

  // Calculate lengths and check for degeneracy
  const double d02 = t0x * t0x + t0y * t0y + t0z * t0z;
  const double d12 = t1x * t1x + t1y * t1y + t1z * t1z;
  if (isDoubleZero(d02) || isDoubleZero(d12)) {
    return;
  }
  const double inv_d0 = rsqrt(t0x * t0x + t0y * t0y + t0z * t0z);
  const double inv_d1 = rsqrt(t1x * t1x + t1y * t1y + t1z * t1z);

  t0x *= inv_d0;
  t0y *= inv_d0;
  t0z *= inv_d0;
  t1x *= inv_d1;
  t1y *= inv_d1;
  t1z *= inv_d1;

  // Calculate cosine of torsion angle
  double cosPhi = dotProduct(t0x, t0y, t0z, t1x, t1y, t1z);
  clipToOne(cosPhi);

  // Calculate sinPhi
  const double sinPhiSq = 1.0 - cosPhi * cosPhi;
  const double sinPhi   = (sinPhiSq > 0.0) ? sqrtf(sinPhiSq) : 0.0;

  // Calculate derivatives for dE/dPhi
  const double cosPhi2 = cosPhi * cosPhi;
  const double cosPhi3 = cosPhi * cosPhi2;
  const double cosPhi4 = cosPhi * cosPhi3;
  const double cosPhi5 = cosPhi * cosPhi4;

  const double dE_dPhi =
    (-forceConstants[0] * signs[0] * sinPhi - 2.0 * forceConstants[1] * signs[1] * (2.0 * cosPhi * sinPhi) -
     3.0 * forceConstants[2] * signs[2] * (4.0 * cosPhi2 * sinPhi - sinPhi) -
     4.0 * forceConstants[3] * signs[3] * (8.0 * cosPhi3 * sinPhi - 4.0 * cosPhi * sinPhi) -
     5.0 * forceConstants[4] * signs[4] * (16.0 * cosPhi4 * sinPhi - 12.0 * cosPhi2 * sinPhi + sinPhi) -
     6.0 * forceConstants[5] * signs[5] * (32.0 * cosPhi5 * sinPhi - 32.0 * cosPhi3 * sinPhi + 6.0 * cosPhi * sinPhi));

  const double sinTerm = -dE_dPhi * (isDoubleZero(sinPhi) ? (1.0 / cosPhi) : (1.0 / sinPhi));

  // Calculate dCos_dT components on-the-fly and compute gradients directly
  const double dCos_dT0x = inv_d0 * (t1x - cosPhi * t0x);
  const double dCos_dT0y = inv_d0 * (t1y - cosPhi * t0y);
  const double dCos_dT0z = inv_d0 * (t1z - cosPhi * t0z);
  const double dCos_dT1x = inv_d1 * (t0x - cosPhi * t1x);
  const double dCos_dT1y = inv_d1 * (t0y - cosPhi * t1y);
  const double dCos_dT1z = inv_d1 * (t0z - cosPhi * t1z);

  // Atom 1 gradient: grad1 = sinTerm * (dCos_dT0 × r2)
  const double g1x = sinTerm * (dCos_dT0z * r2y - dCos_dT0y * r2z);
  const double g1y = sinTerm * (dCos_dT0x * r2z - dCos_dT0z * r2x);
  const double g1z = sinTerm * (dCos_dT0y * r2x - dCos_dT0x * r2y);

  // Atom 4 gradient: grad4 = sinTerm * (dCos_dT1 × r3)
  const double g4x = sinTerm * (dCos_dT1y * r3z - dCos_dT1z * r3y);
  const double g4y = sinTerm * (dCos_dT1z * r3x - dCos_dT1x * r3z);
  const double g4z = sinTerm * (dCos_dT1x * r3y - dCos_dT1y * r3x);

  // Atom 2 gradient: more complex, involves both cross products
  const double g2x =
    sinTerm * (dCos_dT0y * (r2z - r1z) + dCos_dT0z * (r1y - r2y) + dCos_dT1y * (-r4z) + dCos_dT1z * (r4y));
  const double g2y =
    sinTerm * (dCos_dT0x * (r1z - r2z) + dCos_dT0z * (r2x - r1x) + dCos_dT1x * (r4z) + dCos_dT1z * (-r4x));
  const double g2z =
    sinTerm * (dCos_dT0x * (r2y - r1y) + dCos_dT0y * (r1x - r2x) + dCos_dT1x * (-r4y) + dCos_dT1y * (r4x));

  // Atom 3 gradient: grad3 = -(grad1 + grad2 + grad4) by conservation
  const double g3x =
    sinTerm * (dCos_dT0y * r1z + dCos_dT0z * (-r1y) + dCos_dT1y * (r4z - r3z) + dCos_dT1z * (r3y - r4y));
  const double g3y =
    sinTerm * (dCos_dT0x * (-r1z) + dCos_dT0z * r1x + dCos_dT1x * (r3z - r4z) + dCos_dT1z * (r4x - r3x));
  const double g3z =
    sinTerm * (dCos_dT0x * r1y + dCos_dT0y * (-r1x) + dCos_dT1x * (r4y - r3y) + dCos_dT1y * (r3x - r4x));

  // Add gradients using atomic operations
  atomicAdd(&grad[posIdx1 + 0], g1x);
  atomicAdd(&grad[posIdx1 + 1], g1y);
  atomicAdd(&grad[posIdx1 + 2], g1z);
  atomicAdd(&grad[posIdx2 + 0], g2x);
  atomicAdd(&grad[posIdx2 + 1], g2y);
  atomicAdd(&grad[posIdx2 + 2], g2z);
  atomicAdd(&grad[posIdx3 + 0], g3x);
  atomicAdd(&grad[posIdx3 + 1], g3y);
  atomicAdd(&grad[posIdx3 + 2], g3z);
  atomicAdd(&grad[posIdx4 + 0], g4x);
  atomicAdd(&grad[posIdx4 + 1], g4y);
  atomicAdd(&grad[posIdx4 + 2], g4z);
}

static __device__ __forceinline__ void inversionGrad(const double* pos,
                                                     const int     idx1,
                                                     const int     idx2,
                                                     const int     idx3,
                                                     const int     idx4,
                                                     const double  C0,
                                                     const double  C1,
                                                     const double  C2,
                                                     const double  forceConstant,
                                                     double*       grad) {
  // Get positions for all four atoms
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;
  const int posIdx4 = idx4 * 4;

  // Calculate vectors
  double rJIx = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  double rJIy = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  double rJIz = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  double rJKx = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  double rJKy = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  double rJKz = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  double rJLx = pos[posIdx4 + 0] - pos[posIdx2 + 0];
  double rJLy = pos[posIdx4 + 1] - pos[posIdx2 + 1];
  double rJLz = pos[posIdx4 + 2] - pos[posIdx2 + 2];

  const double dJIsq = rJIx * rJIx + rJIy * rJIy + rJIz * rJIz;
  const double dJKsq = rJKx * rJKx + rJKy * rJKy + rJKz * rJKz;
  const double dJLsq = rJLx * rJLx + rJLy * rJLy + rJLz * rJLz;
  if (isDoubleZero(dJIsq) || isDoubleZero(dJKsq) || isDoubleZero(dJLsq)) {
    return;
  }
  // Calculate lengths
  float invdJI = rsqrtf(dJIsq);
  float invdJK = rsqrtf(dJKsq);
  float invdJL = rsqrtf(dJLsq);

  // Normalize vectors
  rJIx *= invdJI;
  rJIy *= invdJI;
  rJIz *= invdJI;
  rJKx *= invdJK;
  rJKy *= invdJK;
  rJKz *= invdJK;
  rJLx *= invdJL;
  rJLy *= invdJL;
  rJLz *= invdJL;

  // Calculate n = (-rJI) × rJK
  double nx, ny, nz;
  crossProduct(-rJIx, -rJIy, -rJIz, rJKx, rJKy, rJKz, nx, ny, nz);

  // Normalize n
  double inv_n_len = rsqrtf(nx * nx + ny * ny + nz * nz);
  nx *= inv_n_len;
  ny *= inv_n_len;
  nz *= inv_n_len;

  // Calculate cosY and clamp
  double cosY = dotProduct(nx, ny, nz, rJLx, rJLy, rJLz);
  clipToOne(cosY);

  // Calculate sinY
  const double sinYSq = 1.0 - cosY * cosY;
  const double sinY   = fmaxf(sqrtf(sinYSq), 1.0e-8f);

  // Calculate cosTheta and clamp
  double cosTheta = dotProduct(rJIx, rJIy, rJIz, rJKx, rJKy, rJKz);
  clipToOne(cosTheta);

  // Calculate sinTheta
  const double sinThetaSq = 1.0 - cosTheta * cosTheta;
  const double sinTheta   = fmaxf(sqrtf(sinThetaSq), 1.0e-8f);

  // Calculate dE_dW
  const double dE_dW = -forceConstant * (C1 * cosY - 4.0 * C2 * cosY * sinY);

  // Calculate cross products for gradient terms
  double t1x, t1y, t1z;  // rJL × rJK
  crossProduct(rJLx, rJLy, rJLz, rJKx, rJKy, rJKz, t1x, t1y, t1z);

  double t2x, t2y, t2z;  // rJI × rJL
  crossProduct(rJIx, rJIy, rJIz, rJLx, rJLy, rJLz, t2x, t2y, t2z);

  double t3x, t3y, t3z;  // rJK × rJI
  crossProduct(rJKx, rJKy, rJKz, rJIx, rJIy, rJIz, t3x, t3y, t3z);

  // Calculate terms for gradient
  const double inverseTerm1   = 1.0 / (sinY * sinTheta);
  const double term2          = cosY / (sinY * sinThetaSq);
  const double cosY_over_sinY = cosY / sinY;

  // Compute gradient components on-the-fly and apply directly
  // Atom 1 gradient components
  const double tg1x = (t1x * inverseTerm1 - (rJIx - rJKx * cosTheta) * term2) * invdJI;
  const double tg1y = (t1y * inverseTerm1 - (rJIy - rJKy * cosTheta) * term2) * invdJI;
  const double tg1z = (t1z * inverseTerm1 - (rJIz - rJKz * cosTheta) * term2) * invdJI;

  // Atom 3 gradient components
  const double tg3x = (t2x * inverseTerm1 - (rJKx - rJIx * cosTheta) * term2) * invdJK;
  const double tg3y = (t2y * inverseTerm1 - (rJKy - rJIy * cosTheta) * term2) * invdJK;
  const double tg3z = (t2z * inverseTerm1 - (rJKz - rJIz * cosTheta) * term2) * invdJK;

  // Atom 4 gradient components
  const double tg4x = (t3x * inverseTerm1 - rJLx * cosY_over_sinY) * invdJL;
  const double tg4y = (t3y * inverseTerm1 - rJLy * cosY_over_sinY) * invdJL;
  const double tg4z = (t3z * inverseTerm1 - rJLz * cosY_over_sinY) * invdJL;

  // Add gradients using atomic operations
  atomicAdd(&grad[posIdx1 + 0], dE_dW * tg1x);
  atomicAdd(&grad[posIdx1 + 1], dE_dW * tg1y);
  atomicAdd(&grad[posIdx1 + 2], dE_dW * tg1z);

  atomicAdd(&grad[posIdx2 + 0], -dE_dW * (tg1x + tg3x + tg4x));
  atomicAdd(&grad[posIdx2 + 1], -dE_dW * (tg1y + tg3y + tg4y));
  atomicAdd(&grad[posIdx2 + 2], -dE_dW * (tg1z + tg3z + tg4z));

  atomicAdd(&grad[posIdx3 + 0], dE_dW * tg3x);
  atomicAdd(&grad[posIdx3 + 1], dE_dW * tg3y);
  atomicAdd(&grad[posIdx3 + 2], dE_dW * tg3z);

  atomicAdd(&grad[posIdx4 + 0], dE_dW * tg4x);
  atomicAdd(&grad[posIdx4 + 1], dE_dW * tg4y);
  atomicAdd(&grad[posIdx4 + 2], dE_dW * tg4z);
}

static __device__ __forceinline__ void distanceConstraintGrad(const double* pos,
                                                              const int     idx1,
                                                              const int     idx2,
                                                              const double  minLen,
                                                              const double  maxLen,
                                                              const double  forceConstant,
                                                              double*       grad) {
  const double minLen2 = minLen * minLen;
  const double maxLen2 = maxLen * maxLen;
  const int    posIdx1 = idx1 * 4;
  const int    posIdx2 = idx2 * 4;

  // Calculate squared distance
  const double distance2 = distanceSquaredPosIdx(pos, posIdx1, posIdx2, 3);

  // Check if distance is outside bounds and calculate preFactor
  double preFactor;
  if (distance2 < minLen2) {
    const double distance = sqrt(distance2);
    preFactor             = forceConstant * (distance - minLen) / fmax(1.0e-8, distance);
  } else if (distance2 > maxLen2) {
    const double distance = sqrt(distance2);
    preFactor             = forceConstant * (distance - maxLen) / fmax(1.0e-8, distance);
  } else {
    return;  // Distance within bounds, no gradient contribution
  }

  // Calculate and accumulate gradients for each component
  for (int i = 0; i < 3; i++) {
    const double dGrad = preFactor * (pos[posIdx1 + i] - pos[posIdx2 + i]);
    atomicAdd(&grad[posIdx1 + i], dGrad);
    atomicAdd(&grad[posIdx2 + i], -dGrad);
  }
}

// Angle constraint gradient
static __device__ __forceinline__ void angleConstraintGrad(const double* pos,
                                                           const int     idx1,
                                                           const int     idx2,
                                                           const int     idx3,
                                                           const double  minAngle,
                                                           const double  maxAngle,
                                                           const double  forceConstant,
                                                           double*       grad) {
  // Get positions for all three atoms
  const int posIdx1 = idx1 * 4;
  const int posIdx2 = idx2 * 4;
  const int posIdx3 = idx3 * 4;

  // Calculate vectors r1 = p1 - p2 and r2 = p3 - p2
  double r1x = pos[posIdx1 + 0] - pos[posIdx2 + 0];
  double r1y = pos[posIdx1 + 1] - pos[posIdx2 + 1];
  double r1z = pos[posIdx1 + 2] - pos[posIdx2 + 2];

  double r2x = pos[posIdx3 + 0] - pos[posIdx2 + 0];
  double r2y = pos[posIdx3 + 1] - pos[posIdx2 + 1];
  double r2z = pos[posIdx3 + 2] - pos[posIdx2 + 2];

  // Calculate squared lengths and take max with 1.0e-5 as in RDKit
  const double r1LengthSq = fmax(1.0e-5, r1x * r1x + r1y * r1y + r1z * r1z);
  const double r2LengthSq = fmax(1.0e-5, r2x * r2x + r2y * r2y + r2z * r2z);
  const double denom      = rsqrt(r1LengthSq * r2LengthSq);

  // Calculate cosine of angle using dot product
  double cosTheta = dotProduct(r1x, r1y, r1z, r2x, r2y, r2z) * denom;

  // Clamp cosTheta to [-1, 1]
  clipToOne(cosTheta);

  // Convert to degrees using RDKit's RAD2DEG constant
  const double angle = RAD2DEG * acos(cosTheta);

  // Calculate angle term using the separate device function
  const double angleTerm = computeAngleTerm(angle, minAngle, maxAngle);

  // Calculate dE_dTheta
  const double dE_dTheta = 2.0 * RAD2DEG * forceConstant * angleTerm;

  // Calculate cross product rp = r2 × r1
  double rpx, rpy, rpz;
  crossProduct(r2x, r2y, r2z, r1x, r1y, r1z, rpx, rpy, rpz);

  // Calculate length of rp and prefactor
  const double rpLengthSq  = fmax(rpx * rpx + rpy * rpy + rpz * rpz, 1e-10);
  const double rpLengthInv = rsqrt(rpLengthSq);
  const double prefactor   = dE_dTheta * rpLengthInv;

  // Calculate t factors
  const double t1 = -prefactor / r1LengthSq;
  const double t2 = prefactor / r2LengthSq;

  // Calculate cross products for gradients
  double dedp1x, dedp1y, dedp1z;  // r1 × rp
  crossProduct(r1x, r1y, r1z, rpx, rpy, rpz, dedp1x, dedp1y, dedp1z);

  double dedp3x, dedp3y, dedp3z;  // r2 × rp
  crossProduct(r2x, r2y, r2z, rpx, rpy, rpz, dedp3x, dedp3y, dedp3z);

  // Scale the cross products by t factors
  dedp1x *= t1;
  dedp1y *= t1;
  dedp1z *= t1;

  dedp3x *= t2;
  dedp3y *= t2;
  dedp3z *= t2;

  // Calculate middle point gradient as negative sum of other two
  const double dedp2x = -(dedp1x + dedp3x);
  const double dedp2y = -(dedp1y + dedp3y);
  const double dedp2z = -(dedp1z + dedp3z);

  // Accumulate gradients using atomic operations
  atomicAdd(&grad[posIdx1 + 0], dedp1x);
  atomicAdd(&grad[posIdx1 + 1], dedp1y);
  atomicAdd(&grad[posIdx1 + 2], dedp1z);

  atomicAdd(&grad[posIdx2 + 0], dedp2x);
  atomicAdd(&grad[posIdx2 + 1], dedp2y);
  atomicAdd(&grad[posIdx2 + 2], dedp2z);

  atomicAdd(&grad[posIdx3 + 0], dedp3x);
  atomicAdd(&grad[posIdx3 + 1], dedp3y);
  atomicAdd(&grad[posIdx3 + 2], dedp3z);
}

template <int dimension>
static __device__ __inline__ double molEnergyDG(const EnergyForceContribsDevicePtr& terms,
                                                const BatchedIndicesDevicePtr&      systemIndices,
                                                const double*                       molCoords,
                                                const int                           molIdx,
                                                const double                        chiralWeight,
                                                const double                        fourthDimWeight,
                                                const int                           tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];

  double energy = 0.0;

  namespace cg            = cooperative_groups;
  constexpr int WARP_SIZE = 32;
  auto          tile32    = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
  const int     laneId    = tile32.thread_rank();
  const int     warpId    = mark_warp_uniform(tile32.meta_group_rank());
  const int     numWarps  = mark_warp_uniform(tile32.meta_group_size());

  // Get term ranges
  const int distStart   = systemIndices.distTermStarts[molIdx];
  const int distEnd     = systemIndices.distTermStarts[molIdx + 1];
  const int chiralStart = systemIndices.chiralTermStarts[molIdx];
  const int chiralEnd   = systemIndices.chiralTermStarts[molIdx + 1];
  const int fourthStart = systemIndices.fourthTermStarts[molIdx];
  const int fourthEnd   = systemIndices.fourthTermStarts[molIdx + 1];

  // Get term data
  const auto& [d_idx1s, d_idx2s, d_ub2s, d_lb2s, d_weights]                  = terms.distTerms;
  const auto& [c_idx1s, c_idx2s, c_idx3s, c_idx4s, c_volUppers, c_volLowers] = terms.chiralTerms;
  const auto& [f_idxs]                                                       = terms.fourthTerms;

  const int numDist   = distEnd - distStart;
  const int numChiral = chiralEnd - chiralStart;
  const int numFourth = fourthEnd - fourthStart;

  // Calculate number of warps needed for each term type
  const int warpsForDist     = (numDist + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForChiral   = (numChiral + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForFourth   = (numFourth + WARP_SIZE - 1) / WARP_SIZE;
  const int totalWarpsNeeded = warpsForDist + warpsForChiral + warpsForFourth;

  // Each warp processes chunks in round-robin fashion
  for (int chunkIdx = warpId; chunkIdx < totalWarpsNeeded; chunkIdx += numWarps) {
    // Determine which term type this chunk belongs to
    if (chunkIdx < warpsForDist) {
      // Distance terms
      const int baseIdx = chunkIdx * WARP_SIZE;
      const int termIdx = distStart + baseIdx + laneId;
      if (baseIdx + laneId < numDist) {
        const int localIdx1 = d_idx1s[termIdx] - atomStart;
        const int localIdx2 = d_idx2s[termIdx] - atomStart;
        energy += distViolationEnergy<dimension>(molCoords,
                                                 localIdx1,
                                                 localIdx2,
                                                 d_lb2s[termIdx],
                                                 d_ub2s[termIdx],
                                                 d_weights[termIdx]);
      }
    } else if (chunkIdx < warpsForDist + warpsForChiral) {
      // Chiral terms
      const int warpOffset = chunkIdx - warpsForDist;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = chiralStart + baseIdx + laneId;
      if (baseIdx + laneId < numChiral) {
        const int localIdx1 = c_idx1s[termIdx] - atomStart;
        const int localIdx2 = c_idx2s[termIdx] - atomStart;
        const int localIdx3 = c_idx3s[termIdx] - atomStart;
        const int localIdx4 = c_idx4s[termIdx] - atomStart;
        energy += chiralViolationEnergy<dimension>(molCoords,
                                                   localIdx1,
                                                   localIdx2,
                                                   localIdx3,
                                                   localIdx4,
                                                   c_volLowers[termIdx],
                                                   c_volUppers[termIdx],
                                                   chiralWeight);
      }
    } else {
      // Fourth dimension terms
      const int warpOffset = chunkIdx - warpsForDist - warpsForChiral;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = fourthStart + baseIdx + laneId;
      if (baseIdx + laneId < numFourth) {
        const int localIdx = f_idxs[termIdx] - atomStart;
        energy += fourthDimEnergy<dimension>(molCoords, localIdx, fourthDimWeight);
      }
    }
  }

  return energy;
}

// Consolidated per-molecule gradient calculation
template <int dimension>
static __device__ __inline__ void molGradDG(const EnergyForceContribsDevicePtr& terms,
                                            const BatchedIndicesDevicePtr&      systemIndices,
                                            const double*                       molCoords,
                                            double*                             molGrad,
                                            const int                           molIdx,
                                            const double                        chiralWeight,
                                            const double                        fourthDimWeight,
                                            const int                           tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];

  namespace cg            = cooperative_groups;
  constexpr int WARP_SIZE = 32;
  auto          tile32    = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
  const int     laneId    = tile32.thread_rank();
  const int     warpId    = mark_warp_uniform(tile32.meta_group_rank());
  const int     numWarps  = mark_warp_uniform(tile32.meta_group_size());

  // Get term ranges
  const int distStart   = systemIndices.distTermStarts[molIdx];
  const int distEnd     = systemIndices.distTermStarts[molIdx + 1];
  const int chiralStart = systemIndices.chiralTermStarts[molIdx];
  const int chiralEnd   = systemIndices.chiralTermStarts[molIdx + 1];
  const int fourthStart = systemIndices.fourthTermStarts[molIdx];
  const int fourthEnd   = systemIndices.fourthTermStarts[molIdx + 1];

  // Get term data
  const auto& [d_idx1s, d_idx2s, d_ub2s, d_lb2s, d_weights]                  = terms.distTerms;
  const auto& [c_idx1s, c_idx2s, c_idx3s, c_idx4s, c_volUppers, c_volLowers] = terms.chiralTerms;
  const auto& [f_idxs]                                                       = terms.fourthTerms;

  const int numDist   = distEnd - distStart;
  const int numChiral = chiralEnd - chiralStart;
  const int numFourth = fourthEnd - fourthStart;

  // Calculate number of warps needed for each term type
  const int warpsForDist     = (numDist + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForChiral   = (numChiral + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForFourth   = (numFourth + WARP_SIZE - 1) / WARP_SIZE;
  const int totalWarpsNeeded = warpsForDist + warpsForChiral + warpsForFourth;

  // Each warp processes chunks in round-robin fashion
  for (int chunkIdx = warpId; chunkIdx < totalWarpsNeeded; chunkIdx += numWarps) {
    // Determine which term type this chunk belongs to
    if (chunkIdx < warpsForDist) {
      // Distance terms
      const int baseIdx = chunkIdx * WARP_SIZE;
      const int termIdx = distStart + baseIdx + laneId;
      if (baseIdx + laneId < numDist) {
        const int localIdx1 = d_idx1s[termIdx] - atomStart;
        const int localIdx2 = d_idx2s[termIdx] - atomStart;
        distViolationGrad<dimension>(molCoords,
                                     localIdx1,
                                     localIdx2,
                                     d_lb2s[termIdx],
                                     d_ub2s[termIdx],
                                     d_weights[termIdx],
                                     molGrad);
      }
    } else if (chunkIdx < warpsForDist + warpsForChiral) {
      // Chiral terms
      const int warpOffset = chunkIdx - warpsForDist;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = chiralStart + baseIdx + laneId;
      if (baseIdx + laneId < numChiral) {
        const int localIdx1 = c_idx1s[termIdx] - atomStart;
        const int localIdx2 = c_idx2s[termIdx] - atomStart;
        const int localIdx3 = c_idx3s[termIdx] - atomStart;
        const int localIdx4 = c_idx4s[termIdx] - atomStart;
        chiralViolationGrad<dimension>(molCoords,
                                       localIdx1,
                                       localIdx2,
                                       localIdx3,
                                       localIdx4,
                                       c_volLowers[termIdx],
                                       c_volUppers[termIdx],
                                       chiralWeight,
                                       molGrad);
      }
    } else {
      // Fourth dimension terms
      const int warpOffset = chunkIdx - warpsForDist - warpsForChiral;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = fourthStart + baseIdx + laneId;
      if (baseIdx + laneId < numFourth) {
        const int localIdx = f_idxs[termIdx] - atomStart;
        fourthDimGrad<dimension>(molCoords, localIdx, fourthDimWeight, molGrad);
      }
    }
  }
}

static __device__ __inline__ double molEnergyETK(const Energy3DForceContribsDevicePtr& terms,
                                                 const BatchedIndices3DDevicePtr&      systemIndices,
                                                 const double*                         molCoords,
                                                 const int                             molIdx,
                                                 const int                             tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];

  double energy = 0.0;

  namespace cg            = cooperative_groups;
  constexpr int WARP_SIZE = 32;
  auto          tile32    = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

  const int laneId   = tile32.thread_rank();
  const int warpId   = mark_warp_uniform(tile32.meta_group_rank());
  const int numWarps = mark_warp_uniform(tile32.meta_group_size());

  // Get term ranges
  const int torsionStart  = systemIndices.experimentalTorsionTermStarts[molIdx];
  const int torsionEnd    = systemIndices.experimentalTorsionTermStarts[molIdx + 1];
  const int improperStart = systemIndices.improperTorsionTermStarts[molIdx];
  const int improperEnd   = systemIndices.improperTorsionTermStarts[molIdx + 1];
  const int dist12Start   = systemIndices.dist12TermStarts[molIdx];
  const int dist12End     = systemIndices.dist12TermStarts[molIdx + 1];
  const int dist13Start   = systemIndices.dist13TermStarts[molIdx];
  const int dist13End     = systemIndices.dist13TermStarts[molIdx + 1];
  const int angle13Start  = systemIndices.angle13TermStarts[molIdx];
  const int angle13End    = systemIndices.angle13TermStarts[molIdx + 1];
  const int distLRStart   = systemIndices.longRangeDistTermStarts[molIdx];
  const int distLREnd     = systemIndices.longRangeDistTermStarts[molIdx + 1];

  const int numTorsion  = torsionEnd - torsionStart;
  const int numImproper = improperEnd - improperStart;
  const int numDist12   = dist12End - dist12Start;
  const int numDist13   = dist13End - dist13Start;
  const int numAngle13  = angle13End - angle13Start;
  const int numDistLR   = distLREnd - distLRStart;

  // Get term data
  const auto& [t_idx1s, t_idx2s, t_idx3s, t_idx4s, t_forceConstants, t_signs] = terms.experimentalTorsionTerms;
  const auto& [i_idx1s, i_idx2s, i_idx3s, i_idx4s, i_at2AtomicNum, i_isCBoundToO, i_C0, i_C1, i_C2, i_forceConstant] =
    terms.improperTorsionTerms;
  const auto& [d12_idx1s, d12_idx2s, d12_minLen, d12_maxLen, d12_forceConstant] = terms.dist12Terms;
  const auto& [d13_idx1s, d13_idx2s, d13_minLen, d13_maxLen, d13_forceConstant] = terms.dist13Terms;
  const auto& [a13_idx1s, a13_idx2s, a13_idx3s, a13_minAngle, a13_maxAngle]     = terms.angle13Terms;
  const auto& [dlr_idx1s, dlr_idx2s, dlr_minLen, dlr_maxLen, dlr_forceConstant] = terms.longRangeDistTerms;

  constexpr double defaultAngleForceConstant = 1.0;

  // Calculate number of warps needed for each term type
  const int warpsForTorsion  = (numTorsion + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForImproper = (numImproper + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForDist12   = (numDist12 + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForDist13   = (numDist13 + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForAngle13  = (numAngle13 + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForDistLR   = (numDistLR + WARP_SIZE - 1) / WARP_SIZE;
  const int totalWarpsNeeded =
    warpsForTorsion + warpsForImproper + warpsForDist12 + warpsForDist13 + warpsForAngle13 + warpsForDistLR;

  // Each warp processes chunks in round-robin fashion
  for (int chunkIdx = warpId; chunkIdx < totalWarpsNeeded; chunkIdx += numWarps) {
    // Determine which term type this chunk belongs to
    if (chunkIdx < warpsForTorsion) {
      // Torsion terms
      const int baseIdx = chunkIdx * WARP_SIZE;
      const int termIdx = torsionStart + baseIdx + laneId;
      if (baseIdx + laneId < numTorsion) {
        const int localIdx1 = t_idx1s[termIdx] - atomStart;
        const int localIdx2 = t_idx2s[termIdx] - atomStart;
        const int localIdx3 = t_idx3s[termIdx] - atomStart;
        const int localIdx4 = t_idx4s[termIdx] - atomStart;
        energy += torsionAngleEnergy(molCoords,
                                     localIdx1,
                                     localIdx2,
                                     localIdx3,
                                     localIdx4,
                                     &t_forceConstants[termIdx * 6],
                                     &t_signs[termIdx * 6]);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper) {
      // Improper torsion terms
      const int warpOffset = chunkIdx - warpsForTorsion;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = improperStart + baseIdx + laneId;
      if (baseIdx + laneId < numImproper) {
        const int localIdx1 = i_idx1s[termIdx] - atomStart;
        const int localIdx2 = i_idx2s[termIdx] - atomStart;
        const int localIdx3 = i_idx3s[termIdx] - atomStart;
        const int localIdx4 = i_idx4s[termIdx] - atomStart;
        energy += inversionEnergy(molCoords,
                                  localIdx1,
                                  localIdx2,
                                  localIdx3,
                                  localIdx4,
                                  i_C0[termIdx],
                                  i_C1[termIdx],
                                  i_C2[termIdx],
                                  i_forceConstant[termIdx]);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper + warpsForDist12) {
      // 1-2 distance terms
      const int warpOffset = chunkIdx - warpsForTorsion - warpsForImproper;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = dist12Start + baseIdx + laneId;
      if (baseIdx + laneId < numDist12) {
        const int localIdx1 = d12_idx1s[termIdx] - atomStart;
        const int localIdx2 = d12_idx2s[termIdx] - atomStart;
        energy += distanceConstraintEnergy(molCoords,
                                           localIdx1,
                                           localIdx2,
                                           d12_minLen[termIdx],
                                           d12_maxLen[termIdx],
                                           d12_forceConstant[termIdx]);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper + warpsForDist12 + warpsForDist13) {
      // 1-3 distance terms
      const int warpOffset = chunkIdx - warpsForTorsion - warpsForImproper - warpsForDist12;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = dist13Start + baseIdx + laneId;
      if (baseIdx + laneId < numDist13) {
        const int localIdx1 = d13_idx1s[termIdx] - atomStart;
        const int localIdx2 = d13_idx2s[termIdx] - atomStart;
        energy += distanceConstraintEnergy(molCoords,
                                           localIdx1,
                                           localIdx2,
                                           d13_minLen[termIdx],
                                           d13_maxLen[termIdx],
                                           d13_forceConstant[termIdx]);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper + warpsForDist12 + warpsForDist13 + warpsForAngle13) {
      // 1-3 angle terms
      const int warpOffset = chunkIdx - warpsForTorsion - warpsForImproper - warpsForDist12 - warpsForDist13;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = angle13Start + baseIdx + laneId;
      if (baseIdx + laneId < numAngle13) {
        const int localIdx1 = a13_idx1s[termIdx] - atomStart;
        const int localIdx2 = a13_idx2s[termIdx] - atomStart;
        const int localIdx3 = a13_idx3s[termIdx] - atomStart;
        energy += angleConstraintEnergy(molCoords,
                                        localIdx1,
                                        localIdx2,
                                        localIdx3,
                                        a13_minAngle[termIdx],
                                        a13_maxAngle[termIdx],
                                        defaultAngleForceConstant);
      }
    } else {
      // Long-range distance terms
      const int warpOffset =
        chunkIdx - warpsForTorsion - warpsForImproper - warpsForDist12 - warpsForDist13 - warpsForAngle13;
      const int baseIdx = warpOffset * WARP_SIZE;
      const int termIdx = distLRStart + baseIdx + laneId;
      if (baseIdx + laneId < numDistLR) {
        const int localIdx1 = dlr_idx1s[termIdx] - atomStart;
        const int localIdx2 = dlr_idx2s[termIdx] - atomStart;
        energy += distanceConstraintEnergy(molCoords,
                                           localIdx1,
                                           localIdx2,
                                           dlr_minLen[termIdx],
                                           dlr_maxLen[termIdx],
                                           dlr_forceConstant[termIdx]);
      }
    }
  }

  return energy;
}

static __device__ __inline__ void molGradETK(const Energy3DForceContribsDevicePtr& terms,
                                             const BatchedIndices3DDevicePtr&      systemIndices,
                                             const double*                         molCoords,
                                             double*                               grad,
                                             const int                             molIdx,
                                             const int                             tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];

  namespace cg            = cooperative_groups;
  constexpr int WARP_SIZE = 32;
  auto          tile32    = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
  const int     laneId    = tile32.thread_rank();

  const int warpId   = mark_warp_uniform(tile32.meta_group_rank());
  const int numWarps = mark_warp_uniform(tile32.meta_group_size());

  // Get term ranges
  const int torsionStart  = systemIndices.experimentalTorsionTermStarts[molIdx];
  const int torsionEnd    = systemIndices.experimentalTorsionTermStarts[molIdx + 1];
  const int improperStart = systemIndices.improperTorsionTermStarts[molIdx];
  const int improperEnd   = systemIndices.improperTorsionTermStarts[molIdx + 1];
  const int dist12Start   = systemIndices.dist12TermStarts[molIdx];
  const int dist12End     = systemIndices.dist12TermStarts[molIdx + 1];
  const int dist13Start   = systemIndices.dist13TermStarts[molIdx];
  const int dist13End     = systemIndices.dist13TermStarts[molIdx + 1];
  const int angle13Start  = systemIndices.angle13TermStarts[molIdx];
  const int angle13End    = systemIndices.angle13TermStarts[molIdx + 1];
  const int distLRStart   = systemIndices.longRangeDistTermStarts[molIdx];
  const int distLREnd     = systemIndices.longRangeDistTermStarts[molIdx + 1];

  const int numTorsion  = torsionEnd - torsionStart;
  const int numImproper = improperEnd - improperStart;
  const int numDist12   = dist12End - dist12Start;
  const int numDist13   = dist13End - dist13Start;
  const int numAngle13  = angle13End - angle13Start;
  const int numDistLR   = distLREnd - distLRStart;

  // Get term data
  const auto& [t_idx1s, t_idx2s, t_idx3s, t_idx4s, t_forceConstants, t_signs] = terms.experimentalTorsionTerms;
  const auto& [i_idx1s, i_idx2s, i_idx3s, i_idx4s, i_at2AtomicNum, i_isCBoundToO, i_C0, i_C1, i_C2, i_forceConstant] =
    terms.improperTorsionTerms;
  const auto& [d12_idx1s, d12_idx2s, d12_minLen, d12_maxLen, d12_forceConstant] = terms.dist12Terms;
  const auto& [d13_idx1s, d13_idx2s, d13_minLen, d13_maxLen, d13_forceConstant] = terms.dist13Terms;
  const auto& [a13_idx1s, a13_idx2s, a13_idx3s, a13_minAngle, a13_maxAngle]     = terms.angle13Terms;
  const auto& [dlr_idx1s, dlr_idx2s, dlr_minLen, dlr_maxLen, dlr_forceConstant] = terms.longRangeDistTerms;

  constexpr double defaultAngleForceConstant = 1.0;

  // Calculate number of warps needed for each term type
  const int warpsForTorsion  = (numTorsion + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForImproper = (numImproper + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForDist12   = (numDist12 + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForDist13   = (numDist13 + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForAngle13  = (numAngle13 + WARP_SIZE - 1) / WARP_SIZE;
  const int warpsForDistLR   = (numDistLR + WARP_SIZE - 1) / WARP_SIZE;
  const int totalWarpsNeeded =
    warpsForTorsion + warpsForImproper + warpsForDist12 + warpsForDist13 + warpsForAngle13 + warpsForDistLR;

  // Each warp processes chunks in round-robin fashion
  for (int chunkIdx = warpId; chunkIdx < totalWarpsNeeded; chunkIdx += numWarps) {
    // Determine which term type this chunk belongs to
    if (chunkIdx < warpsForTorsion) {
      // Torsion terms
      const int baseIdx = chunkIdx * WARP_SIZE;
      const int termIdx = torsionStart + baseIdx + laneId;
      if (baseIdx + laneId < numTorsion) {
        const int localIdx1 = t_idx1s[termIdx] - atomStart;
        const int localIdx2 = t_idx2s[termIdx] - atomStart;
        const int localIdx3 = t_idx3s[termIdx] - atomStart;
        const int localIdx4 = t_idx4s[termIdx] - atomStart;
        torsionAngleGrad(molCoords,
                         localIdx1,
                         localIdx2,
                         localIdx3,
                         localIdx4,
                         &t_forceConstants[termIdx * 6],
                         &t_signs[termIdx * 6],
                         grad);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper) {
      // Improper torsion terms
      const int warpOffset = chunkIdx - warpsForTorsion;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = improperStart + baseIdx + laneId;
      if (baseIdx + laneId < numImproper) {
        const int localIdx1 = i_idx1s[termIdx] - atomStart;
        const int localIdx2 = i_idx2s[termIdx] - atomStart;
        const int localIdx3 = i_idx3s[termIdx] - atomStart;
        const int localIdx4 = i_idx4s[termIdx] - atomStart;
        inversionGrad(molCoords,
                      localIdx1,
                      localIdx2,
                      localIdx3,
                      localIdx4,
                      i_C0[termIdx],
                      i_C1[termIdx],
                      i_C2[termIdx],
                      i_forceConstant[termIdx],
                      grad);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper + warpsForDist12) {
      // 1-2 distance terms
      const int warpOffset = chunkIdx - warpsForTorsion - warpsForImproper;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = dist12Start + baseIdx + laneId;
      if (baseIdx + laneId < numDist12) {
        const int localIdx1 = d12_idx1s[termIdx] - atomStart;
        const int localIdx2 = d12_idx2s[termIdx] - atomStart;
        distanceConstraintGrad(molCoords,
                               localIdx1,
                               localIdx2,
                               d12_minLen[termIdx],
                               d12_maxLen[termIdx],
                               d12_forceConstant[termIdx],
                               grad);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper + warpsForDist12 + warpsForDist13) {
      // 1-3 distance terms
      const int warpOffset = chunkIdx - warpsForTorsion - warpsForImproper - warpsForDist12;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = dist13Start + baseIdx + laneId;
      if (baseIdx + laneId < numDist13) {
        const int localIdx1 = d13_idx1s[termIdx] - atomStart;
        const int localIdx2 = d13_idx2s[termIdx] - atomStart;
        distanceConstraintGrad(molCoords,
                               localIdx1,
                               localIdx2,
                               d13_minLen[termIdx],
                               d13_maxLen[termIdx],
                               d13_forceConstant[termIdx],
                               grad);
      }
    } else if (chunkIdx < warpsForTorsion + warpsForImproper + warpsForDist12 + warpsForDist13 + warpsForAngle13) {
      // 1-3 angle terms
      const int warpOffset = chunkIdx - warpsForTorsion - warpsForImproper - warpsForDist12 - warpsForDist13;
      const int baseIdx    = warpOffset * WARP_SIZE;
      const int termIdx    = angle13Start + baseIdx + laneId;
      if (baseIdx + laneId < numAngle13) {
        const int localIdx1 = a13_idx1s[termIdx] - atomStart;
        const int localIdx2 = a13_idx2s[termIdx] - atomStart;
        const int localIdx3 = a13_idx3s[termIdx] - atomStart;
        angleConstraintGrad(molCoords,
                            localIdx1,
                            localIdx2,
                            localIdx3,
                            a13_minAngle[termIdx],
                            a13_maxAngle[termIdx],
                            defaultAngleForceConstant,
                            grad);
      }
    } else {
      // Long-range distance terms
      const int warpOffset =
        chunkIdx - warpsForTorsion - warpsForImproper - warpsForDist12 - warpsForDist13 - warpsForAngle13;
      const int baseIdx = warpOffset * WARP_SIZE;
      const int termIdx = distLRStart + baseIdx + laneId;
      if (baseIdx + laneId < numDistLR) {
        const int localIdx1 = dlr_idx1s[termIdx] - atomStart;
        const int localIdx2 = dlr_idx2s[termIdx] - atomStart;
        distanceConstraintGrad(molCoords,
                               localIdx1,
                               localIdx2,
                               dlr_minLen[termIdx],
                               dlr_maxLen[termIdx],
                               dlr_forceConstant[termIdx],
                               grad);
      }
    }
  }
}

}  // namespace DistGeom
}  // namespace nvMolKit

#endif  // NVMOLKIT_DISTGEOM_KERNELS_DEVICE_CUH
