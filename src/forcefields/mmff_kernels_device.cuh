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

#ifndef NVMOLKIT_MMFF_KERNELS_DEVICE_CUH
#define NVMOLKIT_MMFF_KERNELS_DEVICE_CUH

#include "kernel_utils.cuh"

using namespace nvMolKit::FFKernelUtils;

constexpr double degreeToRadian = M_PI / 180.0;
constexpr double radianToDegree = 180.0 / M_PI;

namespace rdkit_ports {

static __device__ __forceinline__ void oopGrad(const double* pos,
                                               const int     idx1,
                                               const int     idx2,
                                               const int     idx3,
                                               const int     idx4,
                                               const double  koop,
                                               double*       grad) {
  constexpr double prefactor = 143.9325 * degreeToRadian;

  float dJIx = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  float dJIy = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  float dJIz = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

  float dJKx = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  float dJKy = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  float dJKz = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

  float dJLx = pos[3 * idx4 + 0] - pos[3 * idx2 + 0];
  float dJLy = pos[3 * idx4 + 1] - pos[3 * idx2 + 1];
  float dJLz = pos[3 * idx4 + 2] - pos[3 * idx2 + 2];

  const float invdJI = rsqrtf(dJIx * dJIx + dJIy * dJIy + dJIz * dJIz);
  const float invdJK = rsqrtf(dJKx * dJKx + dJKy * dJKy + dJKz * dJKz);
  const float invdJL = rsqrtf(dJLx * dJLx + dJLy * dJLy + dJLz * dJLz);

  dJIx *= invdJI;
  dJIy *= invdJI;
  dJIz *= invdJI;
  dJKx *= invdJK;
  dJKy *= invdJK;
  dJKz *= invdJK;
  dJLx *= invdJL;
  dJLy *= invdJL;
  dJLz *= invdJL;

  float normalJIKx, normalJIKy, normalJIKz;
  crossProduct(-dJIx, -dJIy, -dJIz, dJKx, dJKy, dJKz, normalJIKx, normalJIKy, normalJIKz);
  const float invNormLength = rsqrtf(normalJIKx * normalJIKx + normalJIKy * normalJIKy + normalJIKz * normalJIKz);
  normalJIKx *= invNormLength;
  normalJIKy *= invNormLength;
  normalJIKz *= invNormLength;

  const float sinChi    = clamp(dotProduct(dJLx, dJLy, dJLz, normalJIKx, normalJIKy, normalJIKz), -1.0f, 1.0f);
  const float cosChiSq  = 1.0 - sinChi * sinChi;
  const float invCosChi = cosChiSq > 0 ? rsqrtf(cosChiSq) : 1.0e8;
  const float chi       = radianToDegree * asin(sinChi);
  const float cosTheta  = clamp(dotProduct(dJIx, dJIy, dJIz, dJKx, dJKy, dJKz), -1.0f, 1.0f);

  float invSinTheta = rsqrtf(fmax(1.0 - cosTheta * cosTheta, 1.0e-8));

  float dE_dChi = prefactor * koop * chi;
  float t1x, t1y, t1z, t2x, t2y, t2z, t3x, t3y, t3z;
  crossProduct(dJLx, dJLy, dJLz, dJKx, dJKy, dJKz, t1x, t1y, t1z);
  crossProduct(dJIx, dJIy, dJIz, dJLx, dJLy, dJLz, t2x, t2y, t2z);
  crossProduct(dJKx, dJKy, dJKz, dJIx, dJIy, dJIz, t3x, t3y, t3z);

  float term1  = invCosChi * invSinTheta;
  float term2  = sinChi * invCosChi * (invSinTheta * invSinTheta);
  float tg1[3] = {(t1x * term1 - (dJIx - dJKx * cosTheta) * term2) * invdJI,
                  (t1y * term1 - (dJIy - dJKy * cosTheta) * term2) * invdJI,
                  (t1z * term1 - (dJIz - dJKz * cosTheta) * term2) * invdJI};
  float tg3[3] = {(t2x * term1 - (dJKx - dJIx * cosTheta) * term2) * invdJK,
                  (t2y * term1 - (dJKy - dJIy * cosTheta) * term2) * invdJK,
                  (t2z * term1 - (dJKz - dJIz * cosTheta) * term2) * invdJK};
  float tg4[3] = {(t3x * term1 - dJLx * sinChi * invCosChi) * invdJL,
                  (t3y * term1 - dJLy * sinChi * invCosChi) * invdJL,
                  (t3z * term1 - dJLz * sinChi * invCosChi) * invdJL};

  atomicAdd(&grad[3 * idx1 + 0], dE_dChi * tg1[0]);
  atomicAdd(&grad[3 * idx1 + 1], dE_dChi * tg1[1]);
  atomicAdd(&grad[3 * idx1 + 2], dE_dChi * tg1[2]);
  atomicAdd(&grad[3 * idx2 + 0], -dE_dChi * (tg1[0] + tg3[0] + tg4[0]));
  atomicAdd(&grad[3 * idx2 + 1], -dE_dChi * (tg1[1] + tg3[1] + tg4[1]));
  atomicAdd(&grad[3 * idx2 + 2], -dE_dChi * (tg1[2] + tg3[2] + tg4[2]));
  atomicAdd(&grad[3 * idx3 + 0], dE_dChi * tg3[0]);
  atomicAdd(&grad[3 * idx3 + 1], dE_dChi * tg3[1]);
  atomicAdd(&grad[3 * idx3 + 2], dE_dChi * tg3[2]);
  atomicAdd(&grad[3 * idx4 + 0], dE_dChi * tg4[0]);
  atomicAdd(&grad[3 * idx4 + 1], dE_dChi * tg4[1]);
  atomicAdd(&grad[3 * idx4 + 2], dE_dChi * tg4[2]);
}

static __device__ __forceinline__ void torsionGrad(const double* pos,
                                                   const int     idx1,
                                                   const int     idx2,
                                                   const int     idx3,
                                                   const int     idx4,
                                                   const float   V1,
                                                   const float   V2,
                                                   const float   V3,
                                                   double*       grad) {
  // P1 - P2
  const float dx1 = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const float dy1 = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const float dz1 = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

  // P3 - P2
  const float dx2 = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  const float dy2 = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  const float dz2 = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

  // P4 - P3
  const float dx4 = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
  const float dy4 = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
  const float dz4 = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

  double cross1x, cross1y, cross1z;
  crossProduct(dx1, dy1, dz1, dx2, dy2, dz2, cross1x, cross1y, cross1z);
  const double invNorm1 = fmin(rsqrt(cross1x * cross1x + cross1y * cross1y + cross1z * cross1z), 1.0e5);
  cross1x *= invNorm1;
  cross1y *= invNorm1;
  cross1z *= invNorm1;

  double cross2x, cross2y, cross2z;
  // Use -dx2, -dy2, -dz2 directly instead of storing dx3, dy3, dz3
  crossProduct(-dx2, -dy2, -dz2, dx4, dy4, dz4, cross2x, cross2y, cross2z);
  const double invNorm2 = fmin(rsqrt(cross2x * cross2x + cross2y * cross2y + cross2z * cross2z), 1.0e5);
  cross2x *= invNorm2;
  cross2y *= invNorm2;
  cross2z *= invNorm2;

  const double cosPhi = clamp(dotProduct(cross1x, cross1y, cross1z, cross2x, cross2y, cross2z), -1.0, 1.0);

  const float sinPhiSq = 1.0f - cosPhi * cosPhi;
  float       sinTerm  = 0.0;
  if (sinPhiSq > 0.0) {
    const float sin2Phi = 2.0f * cosPhi;
    const float sin3Phi = 3.0f - 4.0f * sinPhiSq;
    sinTerm             = 0.5f * (V1 - 2.0f * V2 * sin2Phi + 3.0f * V3 * sin3Phi);
  }

  float dCos_dT0 = invNorm1 * (cross2x - cosPhi * cross1x);
  float dCos_dT1 = invNorm1 * (cross2y - cosPhi * cross1y);
  float dCos_dT2 = invNorm1 * (cross2z - cosPhi * cross1z);

  atomicAdd(&grad[3 * idx1 + 0], sinTerm * (dCos_dT2 * dy2 - dCos_dT1 * dz2));
  atomicAdd(&grad[3 * idx1 + 1], sinTerm * (dCos_dT0 * dz2 - dCos_dT2 * dx2));
  atomicAdd(&grad[3 * idx1 + 2], sinTerm * (dCos_dT1 * dx2 - dCos_dT0 * dy2));

  // idx3 and idx4 gradients - reuse variables dCos_dT0-2 for dCos_dT3-5
  const float dCos_dT3 = invNorm2 * (cross1x - cosPhi * cross2x);
  const float dCos_dT4 = invNorm2 * (cross1y - cosPhi * cross2y);
  const float dCos_dT5 = invNorm2 * (cross1z - cosPhi * cross2z);

  atomicAdd(&grad[3 * idx2 + 0],
            sinTerm * (dCos_dT1 * (dz2 - dz1) + dCos_dT2 * (dy1 - dy2) + dCos_dT4 * (-dz4) + dCos_dT5 * (dy4)));
  atomicAdd(&grad[3 * idx2 + 1],
            sinTerm * (dCos_dT0 * (dz1 - dz2) + dCos_dT2 * (dx2 - dx1) + dCos_dT3 * (dz4) + dCos_dT5 * (-dx4)));
  atomicAdd(&grad[3 * idx2 + 2],
            sinTerm * (dCos_dT0 * (dy2 - dy1) + dCos_dT1 * (dx1 - dx2) + dCos_dT3 * (-dy4) + dCos_dT4 * (dx4)));

  atomicAdd(&grad[3 * idx3 + 0],
            sinTerm * (dCos_dT1 * (dz1) + dCos_dT2 * (-dy1) + dCos_dT4 * (dz4 + dz2) + dCos_dT5 * (-dy4 - dy2)));
  atomicAdd(&grad[3 * idx3 + 1],
            sinTerm * (dCos_dT0 * (-dz1) + dCos_dT2 * (dx1) + dCos_dT3 * (-dz4 - dz2) + dCos_dT5 * (dx4 + dx2)));
  atomicAdd(&grad[3 * idx3 + 2],
            sinTerm * (dCos_dT0 * (dy1) + dCos_dT1 * (-dx1) + dCos_dT3 * (dy4 + dy2) + dCos_dT4 * (-dx4 - dx2)));

  atomicAdd(&grad[3 * idx4 + 0], sinTerm * (dCos_dT4 * (-dz2) - dCos_dT5 * (-dy2)));
  atomicAdd(&grad[3 * idx4 + 1], sinTerm * (dCos_dT5 * (-dx2) - dCos_dT3 * (-dz2)));
  atomicAdd(&grad[3 * idx4 + 2], sinTerm * (dCos_dT3 * (-dy2) - dCos_dT4 * (-dx2)));
}
static __device__ __forceinline__ void vDWGrad(const double* pos,
                                               const int     idx1,
                                               const int     idx2,
                                               const double  R_ij_star,
                                               const double  wellDepth,
                                               double*       grad) {
  constexpr float vdw1   = 1.07;
  constexpr float vdw1m1 = vdw1 - 1.0;
  constexpr float vdw2   = 1.12;
  constexpr float vdw2m1 = vdw2 - 1.0;
  constexpr float vdw2t7 = vdw2 * 7.0;

  const float invDistance = rsqrtf(distanceSquared(pos, idx1, idx2));
  const float distance    = 1.0f / invDistance;

  const float invRIJStar = 1.0f / R_ij_star;

  const float q         = distance * invRIJStar;
  const float q2        = q * q;
  const float q6        = q2 * q2 * q2;
  const float q7        = q6 * q;
  const float q7pvdw2m1 = q7 + vdw2m1;
  const float invQ7Term = 1.0f / q7pvdw2m1;
  const float t         = vdw1 / (q + vdw1 - 1.0);
  const float t2        = t * t;
  const float t7        = t2 * t2 * t2 * t;
  const float dE_dr     = wellDepth * invRIJStar * t7 *
                      (-vdw2t7 * q6 * invQ7Term * invQ7Term + ((-vdw2t7 * invQ7Term + 14.0) / (q + vdw1m1)));

  float term1x, term1y, term1z;
  if (distance <= 0.0) {
    term1x = R_ij_star * 0.01f;
    term1y = R_ij_star * 0.01f;
    term1z = R_ij_star * 0.01f;
  } else {
    term1x = dE_dr * (pos[3 * idx1 + 0] - pos[3 * idx2 + 0]) * invDistance;
    term1y = dE_dr * (pos[3 * idx1 + 1] - pos[3 * idx2 + 1]) * invDistance;
    term1z = dE_dr * (pos[3 * idx1 + 2] - pos[3 * idx2 + 2]) * invDistance;
  }

  atomicAdd(&grad[3 * idx1 + 0], term1x);
  atomicAdd(&grad[3 * idx1 + 1], term1y);
  atomicAdd(&grad[3 * idx1 + 2], term1z);

  atomicAdd(&grad[3 * idx2 + 0], -term1x);
  atomicAdd(&grad[3 * idx2 + 1], -term1y);
  atomicAdd(&grad[3 * idx2 + 2], -term1z);
}

}  // namespace rdkit_ports

static __device__ __forceinline__ double bondStretchEnergy(const double* pos,
                                                           const int     idx1,
                                                           const int     idx2,
                                                           const double  r0,
                                                           const double  kb) {
  constexpr double prefactor           = 143.9325 / 2.0;
  constexpr double csFactorDist        = -2.0;
  constexpr double csFactorDistSquared = 7.0 / 12.0 * csFactorDist * csFactorDist;

  const double distSquared = distanceSquared(pos, idx1, idx2);
  const float  distance    = sqrtf(static_cast<float>(distSquared));

  const float deltaR  = distance - r0;
  const float deltaR2 = deltaR * deltaR;
  return prefactor * kb * deltaR2 * (1.0 + csFactorDist * deltaR + csFactorDistSquared * deltaR2);
}

static __device__ __forceinline__ void bondStretchGrad(const double* pos,
                                                       const int     idx1,
                                                       const int     idx2,
                                                       const double  r0,
                                                       const double  kb,
                                                       double*       grad) {
  constexpr double c1                          = 143.9325;
  constexpr double cs                          = -2.0;
  constexpr double csFactorTimesSecondConstant = cs * 1.5;
  constexpr double lastFactor                  = 2.0 * 7.0 / 12.0 * cs * cs;

  double       dx, dy, dz;
  const double distanceSquared = distanceSquaredWithComponents(pos, idx1, idx2, dx, dy, dz);
  const double invDist         = rsqrt(distanceSquared);
  const double distance        = 1.0 / invDist;
  const double deltaR          = distance - r0;

  const double de_dr = c1 * kb * deltaR * (1.0 + csFactorTimesSecondConstant * deltaR + lastFactor * deltaR * deltaR);

  double dE_dx, dE_dy, dE_dz;
  if (distance > 0.0) {
    dE_dx = de_dr * dx * invDist;
    dE_dy = de_dr * dy * invDist;
    dE_dz = de_dr * dz * invDist;
  } else {
    dE_dx = kb * 0.01;
    dE_dy = kb * 0.01;
    dE_dz = kb * 0.01;
  }

  atomicAdd(&grad[3 * idx1 + 0], dE_dx);
  atomicAdd(&grad[3 * idx1 + 1], dE_dy);
  atomicAdd(&grad[3 * idx1 + 2], dE_dz);

  atomicAdd(&grad[3 * idx2 + 0], -dE_dx);
  atomicAdd(&grad[3 * idx2 + 1], -dE_dy);
  atomicAdd(&grad[3 * idx2 + 2], -dE_dz);
}

static __device__ __forceinline__ double angleBendEnergy(const double* pos,
                                                         const int     idx1,
                                                         const int     idx2,
                                                         const int     idx3,
                                                         const double  theta0,
                                                         const double  ka,
                                                         const bool    isLinear) {
  constexpr double prefactor = 0.5 * 143.9325 * degreeToRadian * degreeToRadian;
  constexpr double cb        = -0.4 * degreeToRadian;

  float       dx1, dy1, dz1, dx2, dy2, dz2;
  const float dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const float dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const float dist1        = rsqrtf(dist1Squared);
  const float dist2        = rsqrtf(dist2Squared);

  const float  dot         = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const double cosTheta    = clamp(dot * (dist1 * dist2), -1.0f, 1.0f);
  const double theta       = radianToDegree * acos(cosTheta);
  const double deltaTheta  = theta - theta0;
  const double deltaTheta2 = deltaTheta * deltaTheta;

  if (isLinear) {
    constexpr double linearPrefactor = 143.9325;
    return linearPrefactor * ka * (1.0 + cosTheta);
  }
  return prefactor * ka * deltaTheta2 * (1.0 + cb * deltaTheta);
}

static __device__ __forceinline__ void angleBendGrad(const int     idx1,
                                                     const int     idx2,
                                                     const int     idx3,
                                                     const double  theta0,
                                                     const double  ka,
                                                     const bool    isLinear,
                                                     const double* pos,
                                                     double*       grad) {
  constexpr double c1       = 143.9325 * degreeToRadian;
  constexpr double cbFactor = -0.006981317 * 1.5;
  // These values are sensitive to double precision.
  double           dx1, dy1, dz1, dx2, dy2, dz2;
  const double     dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const double     dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const double     invDist1     = rsqrt(dist1Squared);
  const double     invDist2     = rsqrt(dist2Squared);

  const double dot        = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const double cosTheta   = clamp(dot * invDist1 * invDist2, -1.0, 1.0);
  const double sinThetaSq = 1.0 - cosTheta * cosTheta;
  if (isDoubleZero(sinThetaSq) || isDoubleZero(dist1Squared) || isDoubleZero(dist2Squared)) {
    return;
  }

  const double invNegSinTheta = -rsqrt(sinThetaSq);
  const float  theta          = radianToDegree * acos(cosTheta);
  const float  deltaTheta     = theta - theta0;

  float de_dDeltaTheta;

  if (isLinear) {
    constexpr float linearPrefactor = 143.9325;
    de_dDeltaTheta                  = -linearPrefactor * ka * sqrtf(1.0 - (cosTheta * cosTheta));
  } else {
    de_dDeltaTheta = c1 * ka * deltaTheta * (1.0 + cbFactor * deltaTheta);
  }

  const float dxnorm1 = dx1 * invDist1;
  const float dynorm1 = dy1 * invDist1;
  const float dznorm1 = dz1 * invDist1;
  const float dxnorm2 = dx2 * invDist2;
  const float dynorm2 = dy2 * invDist2;
  const float dznorm2 = dz2 * invDist2;

  const float intermediate1 = invDist1 * (dxnorm2 - cosTheta * dxnorm1);
  const float intermediate2 = invDist1 * (dynorm2 - cosTheta * dynorm1);
  const float intermediate3 = invDist1 * (dznorm2 - cosTheta * dznorm1);
  const float intermediate4 = invDist2 * (dxnorm1 - cosTheta * dxnorm2);
  const float intermediate5 = invDist2 * (dynorm1 - cosTheta * dynorm2);
  const float intermediate6 = invDist2 * (dznorm1 - cosTheta * dznorm2);

  const float constantFactor = de_dDeltaTheta * invNegSinTheta;

  atomicAdd(&grad[3 * idx1 + 0], constantFactor * intermediate1);
  atomicAdd(&grad[3 * idx1 + 1], constantFactor * intermediate2);
  atomicAdd(&grad[3 * idx1 + 2], constantFactor * intermediate3);

  atomicAdd(&grad[3 * idx2 + 0], constantFactor * (-intermediate1 - intermediate4));
  atomicAdd(&grad[3 * idx2 + 1], constantFactor * (-intermediate2 - intermediate5));
  atomicAdd(&grad[3 * idx2 + 2], constantFactor * (-intermediate3 - intermediate6));

  atomicAdd(&grad[3 * idx3 + 0], constantFactor * intermediate4);
  atomicAdd(&grad[3 * idx3 + 1], constantFactor * intermediate5);
  atomicAdd(&grad[3 * idx3 + 2], constantFactor * intermediate6);
}

static __device__ __forceinline__ double bendStretchEnergy(const double* pos,
                                                           const int     idx1,
                                                           const int     idx2,
                                                           const int     idx3,
                                                           const double  theta0,
                                                           const double  restLen1,
                                                           const double  restLen2,
                                                           const double  forceConst1,
                                                           const double  forceConst2) {
  constexpr double prefactor = 2.51210;

  float       dx1, dy1, dz1, dx2, dy2, dz2;
  const float dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const float dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const float dist1        = sqrtf(dist1Squared);
  const float dist2        = sqrtf(dist2Squared);

  const float  dot      = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const double cosTheta = clamp(dot / (dist1 * dist2), -1.0f, 1.0f);
  const double theta    = 180 / M_PI * acos(cosTheta);

  const double deltaTheta = theta - theta0;
  const double deltaR1    = dist1 - restLen1;
  const double deltaR2    = dist2 - restLen2;

  return prefactor * deltaTheta * (deltaR1 * forceConst1 + deltaR2 * forceConst2);
}

static __device__ __forceinline__ void bendStretchGrad(const double* pos,
                                                       const int     idx1,
                                                       const int     idx2,
                                                       const int     idx3,
                                                       const double  theta0,
                                                       const double  restLen1,
                                                       const double  restLen2,
                                                       const double  forceConst1,
                                                       const double  forceConst2,
                                                       double*       grad) {
  constexpr float prefactor = 143.9325 * M_PI / 180.0;

  float       dx1, dy1, dz1, dx2, dy2, dz2;
  const float dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const float dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  // Note that doing the inverse sqrt would be better here, but it causes drift in some edge case tests.
  const float dist1        = sqrtf(dist1Squared);
  const float dist2        = sqrtf(dist2Squared);
  const float invDist1     = 1.0f / dist1;
  const float invDist2     = 1.0f / dist2;
  const float dot          = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const float cosTheta     = clamp(dot * invDist1 * invDist2, -1.0f, 1.0f);
  const float invSinTheta  = fmin(rsqrtf(1.0f - cosTheta * cosTheta), 1.0e8f);

  constexpr float bondFactor = 180.f / M_PI;
  const float     theta      = bondFactor * acos(cosTheta);

  const float deltaTheta = theta - theta0;
  const float deltaR1    = dist1 - restLen1;
  const float deltaR2    = dist2 - restLen2;

  const float bondEnergyTerm = bondFactor * (forceConst1 * deltaR1 + forceConst2 * deltaR2);

  const float scaledDx1 = dx1 * invDist1;
  const float scaledDy1 = dy1 * invDist1;
  const float scaledDz1 = dz1 * invDist1;
  const float scaledDx2 = dx2 * invDist2;
  const float scaledDy2 = dy2 * invDist2;
  const float scaledDz2 = dz2 * invDist2;

  const float intermediate1 = invDist1 * (scaledDx2 - cosTheta * scaledDx1);
  const float intermediate2 = invDist1 * (scaledDy2 - cosTheta * scaledDy1);
  const float intermediate3 = invDist1 * (scaledDz2 - cosTheta * scaledDz1);
  const float intermediate4 = invDist2 * (scaledDx1 - cosTheta * scaledDx2);
  const float intermediate5 = invDist2 * (scaledDy1 - cosTheta * scaledDy2);
  const float intermediate6 = invDist2 * (scaledDz1 - cosTheta * scaledDz2);

  const float bondEnergyTimesInvSinTheta = bondEnergyTerm * invSinTheta;

  const float gradx1 = prefactor * (deltaTheta * scaledDx1 * forceConst1 - intermediate1 * bondEnergyTimesInvSinTheta);
  const float grady1 = prefactor * (deltaTheta * scaledDy1 * forceConst1 - intermediate2 * bondEnergyTimesInvSinTheta);
  const float gradz1 = prefactor * (deltaTheta * scaledDz1 * forceConst1 - intermediate3 * bondEnergyTimesInvSinTheta);

  const float gradx2 = prefactor * (-deltaTheta * (scaledDx1 * forceConst1 + scaledDx2 * forceConst2) +
                                    (intermediate1 + intermediate4) * bondEnergyTimesInvSinTheta);
  const float grady2 = prefactor * (-deltaTheta * (scaledDy1 * forceConst1 + scaledDy2 * forceConst2) +
                                    (intermediate2 + intermediate5) * bondEnergyTimesInvSinTheta);
  const float gradz2 = prefactor * (-deltaTheta * (scaledDz1 * forceConst1 + scaledDz2 * forceConst2) +
                                    (intermediate3 + intermediate6) * bondEnergyTimesInvSinTheta);

  const float gradx3 = prefactor * (deltaTheta * scaledDx2 * forceConst2 - intermediate4 * bondEnergyTimesInvSinTheta);
  const float grady3 = prefactor * (deltaTheta * scaledDy2 * forceConst2 - intermediate5 * bondEnergyTimesInvSinTheta);
  const float gradz3 = prefactor * (deltaTheta * scaledDz2 * forceConst2 - intermediate6 * bondEnergyTimesInvSinTheta);

  atomicAdd(&grad[3 * idx1 + 0], gradx1);
  atomicAdd(&grad[3 * idx1 + 1], grady1);
  atomicAdd(&grad[3 * idx1 + 2], gradz1);

  atomicAdd(&grad[3 * idx3 + 0], gradx3);
  atomicAdd(&grad[3 * idx3 + 1], grady3);
  atomicAdd(&grad[3 * idx3 + 2], gradz3);

  atomicAdd(&grad[3 * idx2 + 0], gradx2);
  atomicAdd(&grad[3 * idx2 + 1], grady2);
  atomicAdd(&grad[3 * idx2 + 2], gradz2);
}

static __device__ __forceinline__ double oopBendEnergy(const double* pos,
                                                       const int     idx1,
                                                       const int     idx2,
                                                       const int     idx3,
                                                       const int     idx4,
                                                       const double  koop) {
  constexpr float prefactor = 0.5 * 143.9325 * degreeToRadian * degreeToRadian;

  float       dxji, dyji, dzji, dxjk, dyjk, dzjk, dxjl, dyjl, dzjl;
  const float distSquaredJI = distanceSquaredWithComponents(pos, idx1, idx2, dxji, dyji, dzji);
  const float distSquaredJK = distanceSquaredWithComponents(pos, idx3, idx2, dxjk, dyjk, dzjk);
  const float distSquaredJL = distanceSquaredWithComponents(pos, idx4, idx2, dxjl, dyjl, dzjl);

  const float invDistJI = rsqrtf(distSquaredJI);
  const float invDistJK = rsqrtf(distSquaredJK);
  const float invDistJL = rsqrtf(distSquaredJL);

  const float scaledDxJI = dxji * invDistJI;
  const float scaledDyJI = dyji * invDistJI;
  const float scaledDzJI = dzji * invDistJI;

  const float scaledDxJK = dxjk * invDistJK;
  const float scaledDyJK = dyjk * invDistJK;
  const float scaledDzJK = dzjk * invDistJK;

  const float scaledDxJL = dxjl * invDistJL;
  const float scaledDyJL = dyjl * invDistJL;
  const float scaledDzJL = dzjl * invDistJL;

  float crossX, crossY, crossZ;
  crossProduct(scaledDxJI, scaledDyJI, scaledDzJI, scaledDxJK, scaledDyJK, scaledDzJK, crossX, crossY, crossZ);
  const float invDistCross = rsqrtf(crossX * crossX + crossY * crossY + crossZ * crossZ);

  const float scaledCrossX = crossX * invDistCross;
  const float scaledCrossY = crossY * invDistCross;
  const float scaledCrossZ = crossZ * invDistCross;

  const float dotProduct = scaledCrossX * scaledDxJL + scaledCrossY * scaledDyJL + scaledCrossZ * scaledDzJL;
  const float chi        = radianToDegree * asinf(clamp(dotProduct, -1.0f, 1.0f));

  return prefactor * koop * chi * chi;
}

static __device__ __forceinline__ double torsionEnergy(const double* pos,
                                                       const int     idx1,
                                                       const int     idx2,
                                                       const int     idx3,
                                                       const int     idx4,
                                                       const double  V1,
                                                       const double  V2,
                                                       const double  V3) {
  const float dxIJ = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const float dyIJ = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const float dzIJ = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

  const float dxKJ = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  const float dyKJ = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  const float dzKJ = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

  const float dxLK = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
  const float dyLK = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
  const float dzLK = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

  const float crossIJKJx = dyIJ * dzKJ - dzIJ * dyKJ;
  const float crossIJKJy = dzIJ * dxKJ - dxIJ * dzKJ;
  const float crossIJKJz = dxIJ * dyKJ - dyIJ * dxKJ;

  const float crossJKLKx = -dyKJ * dzLK + dzKJ * dyLK;
  const float crossJKLKy = -dzKJ * dxLK + dxKJ * dzLK;
  const float crossJKLKz = -dxKJ * dyLK + dyKJ * dxLK;

  const float invCross1Norm = rsqrtf(crossIJKJx * crossIJKJx + crossIJKJy * crossIJKJy + crossIJKJz * crossIJKJz);
  const float invCross2Norm = rsqrtf(crossJKLKx * crossJKLKx + crossJKLKy * crossJKLKy + crossJKLKz * crossJKLKz);

  const float  dotProduct = crossIJKJx * crossJKLKx + crossIJKJy * crossJKLKy + crossIJKJz * crossJKLKz;
  const float  cosPhi     = dotProduct * invCross1Norm * invCross2Norm;
  const double phi        = acosf(clamp(cosPhi, -1.0f, 1.0f));

  return 0.5 * (V1 * (1.0 + cosPhi) + V2 * (1.0 - cosf(2.0 * phi)) + V3 * (1.0 + cosf(3.0 * phi)));
}

static __device__ __forceinline__ double vdwEnergy(const double* pos,
                                                   const int     idx1,
                                                   const int     idx2,
                                                   const double  R_ij_star,
                                                   const double  wellDepth) {
  // Note, this kernel is quite sensitive, any downcasting to fp32 causes significant drift.
  double R_ij_star2 = R_ij_star * R_ij_star;
  double R_ij_star7 = R_ij_star2 * R_ij_star2 * R_ij_star2 * R_ij_star;

  const double epsilon = wellDepth;

  const double distSquared = distanceSquared(pos, idx1, idx2);
  const double dist        = sqrtf(distSquared);
  const double dist7       = distSquared * distSquared * distSquared * dist;

  const double term1        = 1.07 * R_ij_star / (dist + 0.07 * R_ij_star);
  const double term1Squared = term1 * term1;
  const double term1_7th    = term1Squared * term1Squared * term1Squared * term1;

  const double term2Fraction = 1.12 * R_ij_star7 / (dist7 + 0.12 * R_ij_star7);

  return epsilon * term1_7th * (term2Fraction - 2.0);
}

static __device__ __forceinline__ double eleEnergy(const double* pos,
                                                   const int     idx1,
                                                   const int     idx2,
                                                   const double  chargeTerm,
                                                   const int     dielModel,
                                                   const bool    is1_4) {
  constexpr float prefactor         = 332.0716;
  constexpr float bufferingConstant = 0.05;
  const float     distSquared       = distanceSquared(pos, idx1, idx2);
  float           distTerm          = sqrtf(distSquared) + bufferingConstant;
  if (dielModel == 2) {
    distTerm *= distTerm;
  }
  float energy = prefactor * chargeTerm / (distTerm);
  if (is1_4) {
    energy *= 0.75f;
  }
  return energy;
}

static __device__ __forceinline__ void eleGrad(const double* pos,
                                               const int     idx1,
                                               const int     idx2,
                                               const double  chargeTerm,
                                               const int     dielModel,
                                               const bool    is1_4,
                                               double*       grad) {
  constexpr float prefactor         = 332.0716;
  constexpr float bufferingConstant = 0.05;

  const float distSquared = distanceSquared(pos, idx1, idx2);
  const float invDistance = rsqrtf(distSquared);
  const float distance    = 1.0f / invDistance;
  const float rBuf        = distance + bufferingConstant;
  float       numerator   = -prefactor * chargeTerm;
  // E_1 = q / (r+b)         -> dE/dr = -q / (r+b)^2
  // E_2 = q / (r+b)^2       -> dE/dr = -2q / (r+b)^3
  float       denominator = rBuf * rBuf;
  if (dielModel == 2) {
    numerator *= 2;
    denominator *= rBuf;
  }

  float dE_dr = numerator / denominator;
  if (is1_4) {
    dE_dr *= 0.75;
  }

  const float dE_dx = dE_dr * (pos[3 * idx1 + 0] - pos[3 * idx2 + 0]) * invDistance;
  const float dE_dy = dE_dr * (pos[3 * idx1 + 1] - pos[3 * idx2 + 1]) * invDistance;
  const float dE_dz = dE_dr * (pos[3 * idx1 + 2] - pos[3 * idx2 + 2]) * invDistance;

  atomicAdd(&grad[3 * idx1 + 0], dE_dx);
  atomicAdd(&grad[3 * idx1 + 1], dE_dy);
  atomicAdd(&grad[3 * idx1 + 2], dE_dz);

  atomicAdd(&grad[3 * idx2 + 0], -dE_dx);
  atomicAdd(&grad[3 * idx2 + 1], -dE_dy);
  atomicAdd(&grad[3 * idx2 + 2], -dE_dz);
}

static __device__ __forceinline__ double normalizeAngleDeg(double angleDeg) {
  angleDeg = fmod(angleDeg, 360.0);
  if (angleDeg < -180.0) {
    angleDeg += 360.0;
  } else if (angleDeg > 180.0) {
    angleDeg -= 360.0;
  }
  return angleDeg;
}

static __device__ __forceinline__ double distanceConstraintEnergy(const double* pos,
                                                                  const int     idx1,
                                                                  const int     idx2,
                                                                  const double  minLen,
                                                                  const double  maxLen,
                                                                  const double  forceConstant) {
  const double distance2Val = distanceSquared(pos, idx1, idx2);
  double       difference   = 0.0;
  if (distance2Val < minLen * minLen) {
    difference = minLen - sqrt(distance2Val);
  } else if (distance2Val > maxLen * maxLen) {
    difference = sqrt(distance2Val) - maxLen;
  } else {
    return 0.0;
  }
  return 0.5 * forceConstant * difference * difference;
}

static __device__ __forceinline__ void distanceConstraintGrad(const double* pos,
                                                              const int     idx1,
                                                              const int     idx2,
                                                              const double  minLen,
                                                              const double  maxLen,
                                                              const double  forceConstant,
                                                              double*       grad) {
  const double distance2Val = distanceSquared(pos, idx1, idx2);
  double       preFactor    = 0.0;
  double       distance     = 0.0;
  if (distance2Val < minLen * minLen) {
    distance  = sqrt(distance2Val);
    preFactor = distance - minLen;
  } else if (distance2Val > maxLen * maxLen) {
    distance  = sqrt(distance2Val);
    preFactor = distance - maxLen;
  } else {
    return;
  }
  preFactor *= forceConstant;
  preFactor /= fmax(1.0e-8, distance);
  for (int i = 0; i < 3; ++i) {
    const double dGrad = preFactor * (pos[3 * idx1 + i] - pos[3 * idx2 + i]);
    atomicAdd(&grad[3 * idx1 + i], dGrad);
    atomicAdd(&grad[3 * idx2 + i], -dGrad);
  }
}

static __device__ __forceinline__ double positionConstraintEnergy(const double* pos,
                                                                  const int     idx,
                                                                  const double  refX,
                                                                  const double  refY,
                                                                  const double  refZ,
                                                                  const double  maxDispl,
                                                                  const double  forceConstant) {
  const double dx       = pos[3 * idx + 0] - refX;
  const double dy       = pos[3 * idx + 1] - refY;
  const double dz       = pos[3 * idx + 2] - refZ;
  const double dist     = sqrt(dx * dx + dy * dy + dz * dz);
  const double distTerm = fmax(dist - maxDispl, 0.0);
  return 0.5 * forceConstant * distTerm * distTerm;
}

static __device__ __forceinline__ void positionConstraintGrad(const double* pos,
                                                              const int     idx,
                                                              const double  refX,
                                                              const double  refY,
                                                              const double  refZ,
                                                              const double  maxDispl,
                                                              const double  forceConstant,
                                                              double*       grad) {
  const double dx   = pos[3 * idx + 0] - refX;
  const double dy   = pos[3 * idx + 1] - refY;
  const double dz   = pos[3 * idx + 2] - refZ;
  const double dist = sqrt(dx * dx + dy * dy + dz * dz);
  if (dist <= maxDispl) {
    return;
  }
  const double preFactor = (dist - maxDispl) * forceConstant / fmax(dist, 1.0e-8);
  atomicAdd(&grad[3 * idx + 0], preFactor * dx);
  atomicAdd(&grad[3 * idx + 1], preFactor * dy);
  atomicAdd(&grad[3 * idx + 2], preFactor * dz);
}

static __device__ __forceinline__ double computeAngleConstraintTerm(const double angle,
                                                                    const double minAngleDeg,
                                                                    const double maxAngleDeg) {
  double angleTerm = 0.0;
  if (angle < minAngleDeg) {
    angleTerm = angle - minAngleDeg;
  } else if (angle > maxAngleDeg) {
    angleTerm = angle - maxAngleDeg;
  }
  return angleTerm;
}

static __device__ __forceinline__ double angleConstraintEnergy(const double* pos,
                                                               const int     idx1,
                                                               const int     idx2,
                                                               const int     idx3,
                                                               const double  minAngleDeg,
                                                               const double  maxAngleDeg,
                                                               const double  forceConstant) {
  const double p1x = pos[3 * idx1 + 0];
  const double p1y = pos[3 * idx1 + 1];
  const double p1z = pos[3 * idx1 + 2];
  const double p2x = pos[3 * idx2 + 0];
  const double p2y = pos[3 * idx2 + 1];
  const double p2z = pos[3 * idx2 + 2];
  const double p3x = pos[3 * idx3 + 0];
  const double p3y = pos[3 * idx3 + 1];
  const double p3z = pos[3 * idx3 + 2];

  const double r1x        = p1x - p2x;
  const double r1y        = p1y - p2y;
  const double r1z        = p1z - p2z;
  const double r2x        = p3x - p2x;
  const double r2y        = p3y - p2y;
  const double r2z        = p3z - p2z;
  const double rLengthSq1 = fmax(1.0e-5, r1x * r1x + r1y * r1y + r1z * r1z);
  const double rLengthSq2 = fmax(1.0e-5, r2x * r2x + r2y * r2y + r2z * r2z);
  double       cosTheta   = (r1x * r2x + r1y * r2y + r1z * r2z) / sqrt(rLengthSq1 * rLengthSq2);
  cosTheta                = clamp(cosTheta, -1.0, 1.0);
  const double angle      = radianToDegree * acos(cosTheta);
  const double angleTerm  = computeAngleConstraintTerm(angle, minAngleDeg, maxAngleDeg);
  return forceConstant * angleTerm * angleTerm;
}

static __device__ __forceinline__ void angleConstraintGrad(const double* pos,
                                                           const int     idx1,
                                                           const int     idx2,
                                                           const int     idx3,
                                                           const double  minAngleDeg,
                                                           const double  maxAngleDeg,
                                                           const double  forceConstant,
                                                           double*       grad) {
  const double p1x = pos[3 * idx1 + 0];
  const double p1y = pos[3 * idx1 + 1];
  const double p1z = pos[3 * idx1 + 2];
  const double p2x = pos[3 * idx2 + 0];
  const double p2y = pos[3 * idx2 + 1];
  const double p2z = pos[3 * idx2 + 2];
  const double p3x = pos[3 * idx3 + 0];
  const double p3y = pos[3 * idx3 + 1];
  const double p3z = pos[3 * idx3 + 2];

  const double r1x        = p1x - p2x;
  const double r1y        = p1y - p2y;
  const double r1z        = p1z - p2z;
  const double r2x        = p3x - p2x;
  const double r2y        = p3y - p2y;
  const double r2z        = p3z - p2z;
  const double rLengthSq1 = fmax(1.0e-5, r1x * r1x + r1y * r1y + r1z * r1z);
  const double rLengthSq2 = fmax(1.0e-5, r2x * r2x + r2y * r2y + r2z * r2z);
  const double invDist1   = rsqrt(rLengthSq1);
  const double invDist2   = rsqrt(rLengthSq2);
  double       cosTheta   = (r1x * r2x + r1y * r2y + r1z * r2z) * invDist1 * invDist2;
  cosTheta                = clamp(cosTheta, -1.0, 1.0);
  const double angle      = radianToDegree * acos(cosTheta);
  const double angleTerm  = computeAngleConstraintTerm(angle, minAngleDeg, maxAngleDeg);
  if (isDoubleZero(angleTerm)) {
    return;
  }

  const double rpX       = r2y * r1z - r2z * r1y;
  const double rpY       = r2z * r1x - r2x * r1z;
  const double rpZ       = r2x * r1y - r2y * r1x;
  const double rpLength  = fmax(1.0e-5, sqrt(rpX * rpX + rpY * rpY + rpZ * rpZ));
  const double dE_dTheta = 2.0 * radianToDegree * forceConstant * angleTerm;
  const double prefactor = dE_dTheta / rpLength;
  const double t0        = -prefactor / rLengthSq1;
  const double t1        = prefactor / rLengthSq2;

  const double c0x = r1y * rpZ - r1z * rpY;
  const double c0y = r1z * rpX - r1x * rpZ;
  const double c0z = r1x * rpY - r1y * rpX;
  const double c1x = r2y * rpZ - r2z * rpY;
  const double c1y = r2z * rpX - r2x * rpZ;
  const double c1z = r2x * rpY - r2y * rpX;

  const double dedp0x = c0x * t0;
  const double dedp0y = c0y * t0;
  const double dedp0z = c0z * t0;
  const double dedp2x = c1x * t1;
  const double dedp2y = c1y * t1;
  const double dedp2z = c1z * t1;
  const double dedp1x = -dedp0x - dedp2x;
  const double dedp1y = -dedp0y - dedp2y;
  const double dedp1z = -dedp0z - dedp2z;

  atomicAdd(&grad[3 * idx1 + 0], dedp0x);
  atomicAdd(&grad[3 * idx1 + 1], dedp0y);
  atomicAdd(&grad[3 * idx1 + 2], dedp0z);
  atomicAdd(&grad[3 * idx2 + 0], dedp1x);
  atomicAdd(&grad[3 * idx2 + 1], dedp1y);
  atomicAdd(&grad[3 * idx2 + 2], dedp1z);
  atomicAdd(&grad[3 * idx3 + 0], dedp2x);
  atomicAdd(&grad[3 * idx3 + 1], dedp2y);
  atomicAdd(&grad[3 * idx3 + 2], dedp2z);
}

static __device__ __forceinline__ double computeDihedralConstraintTerm(double       dihedral,
                                                                       const double minDihedralDeg,
                                                                       const double maxDihedralDeg) {
  double dihedralTarget = dihedral;
  if (!(dihedral > minDihedralDeg && dihedral < maxDihedralDeg) &&
      !(dihedral > minDihedralDeg && minDihedralDeg > maxDihedralDeg) &&
      !(dihedral < maxDihedralDeg && minDihedralDeg > maxDihedralDeg)) {
    double dihedralMinTarget = normalizeAngleDeg(dihedral - minDihedralDeg);
    double dihedralMaxTarget = normalizeAngleDeg(dihedral - maxDihedralDeg);
    if (fabs(dihedralMinTarget) < fabs(dihedralMaxTarget)) {
      dihedralTarget = minDihedralDeg;
    } else {
      dihedralTarget = maxDihedralDeg;
    }
  }
  return normalizeAngleDeg(dihedral - dihedralTarget);
}

static __device__ __forceinline__ double computeSignedDihedral(const double* pos,
                                                               const int     idx1,
                                                               const int     idx2,
                                                               const int     idx3,
                                                               const int     idx4,
                                                               double*       cosPhiOut = nullptr,
                                                               double        r[4][3]   = nullptr,
                                                               double        t[2][3]   = nullptr,
                                                               double        d[2]      = nullptr) {
  double localR[4][3];
  double localT[2][3];
  double localD[2];
  if (r == nullptr) {
    r = localR;
  }
  if (t == nullptr) {
    t = localT;
  }
  if (d == nullptr) {
    d = localD;
  }
  r[0][0] = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  r[0][1] = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  r[0][2] = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];
  r[1][0] = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  r[1][1] = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  r[1][2] = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];
  r[2][0] = -r[1][0];
  r[2][1] = -r[1][1];
  r[2][2] = -r[1][2];
  r[3][0] = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
  r[3][1] = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
  r[3][2] = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

  crossProduct(r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], t[0][0], t[0][1], t[0][2]);
  d[0] = fmax(sqrt(t[0][0] * t[0][0] + t[0][1] * t[0][1] + t[0][2] * t[0][2]), 1.0e-5);
  t[0][0] /= d[0];
  t[0][1] /= d[0];
  t[0][2] /= d[0];
  crossProduct(r[2][0], r[2][1], r[2][2], r[3][0], r[3][1], r[3][2], t[1][0], t[1][1], t[1][2]);
  d[1] = fmax(sqrt(t[1][0] * t[1][0] + t[1][1] * t[1][1] + t[1][2] * t[1][2]), 1.0e-5);
  t[1][0] /= d[1];
  t[1][1] /= d[1];
  t[1][2] /= d[1];
  const double cosPhi = clamp(t[0][0] * t[1][0] + t[0][1] * t[1][1] + t[0][2] * t[1][2], -1.0, 1.0);
  if (cosPhiOut != nullptr) {
    *cosPhiOut = cosPhi;
  }
  double mX, mY, mZ;
  crossProduct(t[0][0], t[0][1], t[0][2], r[1][0], r[1][1], r[1][2], mX, mY, mZ);
  const double mLength = fmax(sqrt(mX * mX + mY * mY + mZ * mZ), 1.0e-5);
  return -atan2((mX * t[1][0] + mY * t[1][1] + mZ * t[1][2]) / mLength, cosPhi);
}

static __device__ __forceinline__ double torsionConstraintEnergy(const double* pos,
                                                                 const int     idx1,
                                                                 const int     idx2,
                                                                 const int     idx3,
                                                                 const int     idx4,
                                                                 const double  minDihedralDeg,
                                                                 const double  maxDihedralDeg,
                                                                 const double  forceConstant) {
  const double dihedral     = radianToDegree * computeSignedDihedral(pos, idx1, idx2, idx3, idx4);
  const double dihedralTerm = computeDihedralConstraintTerm(dihedral, minDihedralDeg, maxDihedralDeg);
  return forceConstant * dihedralTerm * dihedralTerm;
}

static __device__ __forceinline__ void torsionConstraintGrad(const double* pos,
                                                             const int     idx1,
                                                             const int     idx2,
                                                             const int     idx3,
                                                             const int     idx4,
                                                             const double  minDihedralDeg,
                                                             const double  maxDihedralDeg,
                                                             const double  forceConstant,
                                                             double*       grad) {
  double       r[4][3];
  double       t[2][3];
  double       d[2];
  const double dihedral     = radianToDegree * computeSignedDihedral(pos, idx1, idx2, idx3, idx4, nullptr, r, t, d);
  const double dihedralTerm = computeDihedralConstraintTerm(dihedral, minDihedralDeg, maxDihedralDeg);
  if (isDoubleZero(dihedralTerm)) {
    return;
  }
  const double dE_dPhi = 2.0 * radianToDegree * forceConstant * dihedralTerm;

  const double d23       = sqrt(distanceSquared(pos, idx2, idx3));
  const double prefactor = dE_dPhi / fmax(d23, 1.0e-8);

  double tt0[3], tt1[3];
  crossProduct(r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], tt0[0], tt0[1], tt0[2]);
  crossProduct(r[2][0], r[2][1], r[2][2], r[3][0], r[3][1], r[3][2], tt1[0], tt1[1], tt1[2]);
  const double tt0LenSq = fmax(tt0[0] * tt0[0] + tt0[1] * tt0[1] + tt0[2] * tt0[2], 1.0e-8);
  const double tt1LenSq = fmax(tt1[0] * tt1[0] + tt1[1] * tt1[1] + tt1[2] * tt1[2], 1.0e-8);

  double tmp0[3], tmp1[3];
  crossProduct(tt0[0], tt0[1], tt0[2], r[2][0], r[2][1], r[2][2], tmp0[0], tmp0[1], tmp0[2]);
  crossProduct(tt1[0], tt1[1], tt1[2], r[1][0], r[1][1], r[1][2], tmp1[0], tmp1[1], tmp1[2]);
  const double dedt0[3] = {tmp0[0] / tt0LenSq * prefactor,
                           tmp0[1] / tt0LenSq * prefactor,
                           tmp0[2] / tt0LenSq * prefactor};
  const double dedt1[3] = {tmp1[0] / tt1LenSq * prefactor,
                           tmp1[1] / tt1LenSq * prefactor,
                           tmp1[2] / tt1LenSq * prefactor};

  const double r31[3] = {pos[3 * idx3 + 0] - pos[3 * idx1 + 0],
                         pos[3 * idx3 + 1] - pos[3 * idx1 + 1],
                         pos[3 * idx3 + 2] - pos[3 * idx1 + 2]};
  const double r42[3] = {pos[3 * idx4 + 0] - pos[3 * idx2 + 0],
                         pos[3 * idx4 + 1] - pos[3 * idx2 + 1],
                         pos[3 * idx4 + 2] - pos[3 * idx2 + 2]};

  double dedp0[3], dedp1[3], dedp2[3], dedp3[3];
  crossProduct(r[2][0], r[2][1], r[2][2], dedt0[0], dedt0[1], dedt0[2], dedp0[0], dedp0[1], dedp0[2]);

  double r31Cross[3], r3Cross[3];
  crossProduct(r31[0], r31[1], r31[2], dedt0[0], dedt0[1], dedt0[2], r31Cross[0], r31Cross[1], r31Cross[2]);
  crossProduct(r[3][0], r[3][1], r[3][2], dedt1[0], dedt1[1], dedt1[2], r3Cross[0], r3Cross[1], r3Cross[2]);
  dedp1[0] = r31Cross[0] - r3Cross[0];
  dedp1[1] = r31Cross[1] - r3Cross[1];
  dedp1[2] = r31Cross[2] - r3Cross[2];

  double r0Cross[3], r42Cross[3];
  crossProduct(r[0][0], r[0][1], r[0][2], dedt0[0], dedt0[1], dedt0[2], r0Cross[0], r0Cross[1], r0Cross[2]);
  crossProduct(r42[0], r42[1], r42[2], dedt1[0], dedt1[1], dedt1[2], r42Cross[0], r42Cross[1], r42Cross[2]);
  dedp2[0] = r0Cross[0] + r42Cross[0];
  dedp2[1] = r0Cross[1] + r42Cross[1];
  dedp2[2] = r0Cross[2] + r42Cross[2];

  crossProduct(r[2][0], r[2][1], r[2][2], dedt1[0], dedt1[1], dedt1[2], dedp3[0], dedp3[1], dedp3[2]);

  atomicAdd(&grad[3 * idx1 + 0], dedp0[0]);
  atomicAdd(&grad[3 * idx1 + 1], dedp0[1]);
  atomicAdd(&grad[3 * idx1 + 2], dedp0[2]);
  atomicAdd(&grad[3 * idx2 + 0], dedp1[0]);
  atomicAdd(&grad[3 * idx2 + 1], dedp1[1]);
  atomicAdd(&grad[3 * idx2 + 2], dedp1[2]);
  atomicAdd(&grad[3 * idx3 + 0], dedp2[0]);
  atomicAdd(&grad[3 * idx3 + 1], dedp2[1]);
  atomicAdd(&grad[3 * idx3 + 2], dedp2[2]);
  atomicAdd(&grad[3 * idx4 + 0], dedp3[0]);
  atomicAdd(&grad[3 * idx4 + 1], dedp3[1]);
  atomicAdd(&grad[3 * idx4 + 2], dedp3[2]);
}

namespace nvMolKit {
namespace MMFF {

template <int stride, bool HasConstraints>
static __device__ __inline__ double molEnergy(const EnergyForceContribsDevicePtr& terms,
                                              const BatchedIndicesDevicePtr&      systemIndices,
                                              const double*                       molCoords,
                                              const int                           molIdx,
                                              const int                           tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];

  double energy = 0.0;

  const auto& [idx1s, idx2s, r0s, kbs] = terms.bondTerms;
  const int bondStart                  = systemIndices.bondTermStarts[molIdx];
  const int bondEnd                    = systemIndices.bondTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = bondStart + tid; i < bondEnd; i += stride) {
    const int localIdx1 = idx1s[i] - atomStart;
    const int localIdx2 = idx2s[i] - atomStart;
    energy += bondStretchEnergy(molCoords, localIdx1, localIdx2, r0s[i], kbs[i]);
  }

  const auto& [a_idx1s, a_idx2s, a_idx3s, theta0s, kas, isLinears] = terms.angleTerms;
  const int angleStart                                             = systemIndices.angleTermStarts[molIdx];
  const int angleEnd                                               = systemIndices.angleTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = angleStart + tid; i < angleEnd; i += stride) {
    const int  localIdx1 = a_idx1s[i] - atomStart;
    const int  localIdx2 = a_idx2s[i] - atomStart;
    const int  localIdx3 = a_idx3s[i] - atomStart;
    const bool isLinear  = static_cast<bool>(isLinears[i]);
    energy += angleBendEnergy(molCoords, localIdx1, localIdx2, localIdx3, theta0s[i], kas[i], isLinear);
  }

  const auto& [bs_idx1s, bs_idx2s, bs_idx3s, bs_theta0s, restLen1s, restLen2s, forceConst1s, forceConst2s] =
    terms.bendTerms;
  const int bendStart = systemIndices.bendTermStarts[molIdx];
  const int bendEnd   = systemIndices.bendTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = bendStart + tid; i < bendEnd; i += stride) {
    const int localIdx1 = bs_idx1s[i] - atomStart;
    const int localIdx2 = bs_idx2s[i] - atomStart;
    const int localIdx3 = bs_idx3s[i] - atomStart;
    energy += bendStretchEnergy(molCoords,
                                localIdx1,
                                localIdx2,
                                localIdx3,
                                bs_theta0s[i],
                                restLen1s[i],
                                restLen2s[i],
                                forceConst1s[i],
                                forceConst2s[i]);
  }

  const auto& [o_idx1s, o_idx2s, o_idx3s, o_idx4s, koops] = terms.oopTerms;
  const int oopStart                                      = systemIndices.oopTermStarts[molIdx];
  const int oopEnd                                        = systemIndices.oopTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = oopStart + tid; i < oopEnd; i += stride) {
    const int localIdx1 = o_idx1s[i] - atomStart;
    const int localIdx2 = o_idx2s[i] - atomStart;
    const int localIdx3 = o_idx3s[i] - atomStart;
    const int localIdx4 = o_idx4s[i] - atomStart;
    energy += oopBendEnergy(molCoords, localIdx1, localIdx2, localIdx3, localIdx4, koops[i]);
  }

  const auto& [t_idx1s, t_idx2s, t_idx3s, t_idx4s, V1s, V2s, V3s] = terms.torsionTerms;
  const int torsionStart                                          = systemIndices.torsionTermStarts[molIdx];
  const int torsionEnd                                            = systemIndices.torsionTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = torsionStart + tid; i < torsionEnd; i += stride) {
    const int localIdx1 = t_idx1s[i] - atomStart;
    const int localIdx2 = t_idx2s[i] - atomStart;
    const int localIdx3 = t_idx3s[i] - atomStart;
    const int localIdx4 = t_idx4s[i] - atomStart;
    energy += torsionEnergy(molCoords, localIdx1, localIdx2, localIdx3, localIdx4, V1s[i], V2s[i], V3s[i]);
  }

  const auto& [v_idx1s, v_idx2s, R_ij_stars, wellDepths] = terms.vdwTerms;
  const int vdwStart                                     = systemIndices.vdwTermStarts[molIdx];
  const int vdwEnd                                       = systemIndices.vdwTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = vdwStart + tid; i < vdwEnd; i += stride) {
    const int localIdx1 = v_idx1s[i] - atomStart;
    const int localIdx2 = v_idx2s[i] - atomStart;
    energy += vdwEnergy(molCoords, localIdx1, localIdx2, R_ij_stars[i], wellDepths[i]);
  }

  const auto& [e_idx1s, e_idx2s, chargeTerms, dielModels, is1_4s] = terms.eleTerms;
  const int eleStart                                              = systemIndices.eleTermStarts[molIdx];
  const int eleEnd                                                = systemIndices.eleTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = eleStart + tid; i < eleEnd; i += stride) {
    const int  localIdx1 = e_idx1s[i] - atomStart;
    const int  localIdx2 = e_idx2s[i] - atomStart;
    const int  dielModel = static_cast<int>(dielModels[i]);
    const bool is14      = is1_4s[i] > 0;
    energy += eleEnergy(molCoords, localIdx1, localIdx2, chargeTerms[i], dielModel, is14);
  }

  if constexpr (HasConstraints) {
    const auto& [dc_idx1s, dc_idx2s, minLens, maxLens, dcForceConstants] = terms.distanceConstraintTerms;
    const int dcStart = systemIndices.distanceConstraintTermStarts[molIdx];
    const int dcEnd   = systemIndices.distanceConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = dcStart + tid; i < dcEnd; i += stride) {
      const int localIdx1 = dc_idx1s[i] - atomStart;
      const int localIdx2 = dc_idx2s[i] - atomStart;
      energy += distanceConstraintEnergy(molCoords, localIdx1, localIdx2, minLens[i], maxLens[i], dcForceConstants[i]);
    }

    const auto& [pc_idxs, refXs, refYs, refZs, maxDispls, pcForceConstants] = terms.positionConstraintTerms;
    const int pcStart = systemIndices.positionConstraintTermStarts[molIdx];
    const int pcEnd   = systemIndices.positionConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = pcStart + tid; i < pcEnd; i += stride) {
      const int localIdx = pc_idxs[i] - atomStart;
      energy +=
        positionConstraintEnergy(molCoords, localIdx, refXs[i], refYs[i], refZs[i], maxDispls[i], pcForceConstants[i]);
    }

    const auto& [ac_idx1s, ac_idx2s, ac_idx3s, minAngleDegs, maxAngleDegs, acForceConstants] =
      terms.angleConstraintTerms;
    const int acStart = systemIndices.angleConstraintTermStarts[molIdx];
    const int acEnd   = systemIndices.angleConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = acStart + tid; i < acEnd; i += stride) {
      const int localIdx1 = ac_idx1s[i] - atomStart;
      const int localIdx2 = ac_idx2s[i] - atomStart;
      const int localIdx3 = ac_idx3s[i] - atomStart;
      energy += angleConstraintEnergy(molCoords,
                                      localIdx1,
                                      localIdx2,
                                      localIdx3,
                                      minAngleDegs[i],
                                      maxAngleDegs[i],
                                      acForceConstants[i]);
    }

    const auto& [tc_idx1s, tc_idx2s, tc_idx3s, tc_idx4s, minDihedralDegs, maxDihedralDegs, tcForceConstants] =
      terms.torsionConstraintTerms;
    const int tcStart = systemIndices.torsionConstraintTermStarts[molIdx];
    const int tcEnd   = systemIndices.torsionConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = tcStart + tid; i < tcEnd; i += stride) {
      const int localIdx1 = tc_idx1s[i] - atomStart;
      const int localIdx2 = tc_idx2s[i] - atomStart;
      const int localIdx3 = tc_idx3s[i] - atomStart;
      const int localIdx4 = tc_idx4s[i] - atomStart;
      energy += torsionConstraintEnergy(molCoords,
                                        localIdx1,
                                        localIdx2,
                                        localIdx3,
                                        localIdx4,
                                        minDihedralDegs[i],
                                        maxDihedralDegs[i],
                                        tcForceConstants[i]);
    }
  }

  return energy;
}

template <int stride, bool HasConstraints>
static __device__ __inline__ void molGrad(const EnergyForceContribsDevicePtr& terms,
                                          const BatchedIndicesDevicePtr&      systemIndices,
                                          const double*                       molCoords,
                                          double*                             grad,
                                          const int                           molIdx,
                                          const int                           tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];

  const auto& [idx1s, idx2s, r0s, kbs] = terms.bondTerms;
  const int bondStart                  = systemIndices.bondTermStarts[molIdx];
  const int bondEnd                    = systemIndices.bondTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = bondStart + tid; i < bondEnd; i += stride) {
    const int localIdx1 = idx1s[i] - atomStart;
    const int localIdx2 = idx2s[i] - atomStart;
    bondStretchGrad(molCoords, localIdx1, localIdx2, r0s[i], kbs[i], grad);
  }

  const auto& [a_idx1s, a_idx2s, a_idx3s, theta0s, kas, isLinears] = terms.angleTerms;
  const int angleStart                                             = systemIndices.angleTermStarts[molIdx];
  const int angleEnd                                               = systemIndices.angleTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = angleStart + tid; i < angleEnd; i += stride) {
    const int  localIdx1 = a_idx1s[i] - atomStart;
    const int  localIdx2 = a_idx2s[i] - atomStart;
    const int  localIdx3 = a_idx3s[i] - atomStart;
    const bool isLinear  = static_cast<bool>(isLinears[i]);
    angleBendGrad(localIdx1, localIdx2, localIdx3, theta0s[i], kas[i], isLinear, molCoords, grad);
  }

  const auto& [bs_idx1s, bs_idx2s, bs_idx3s, bs_theta0s, restLen1s, restLen2s, forceConst1s, forceConst2s] =
    terms.bendTerms;
  const int bendStart = systemIndices.bendTermStarts[molIdx];
  const int bendEnd   = systemIndices.bendTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = bendStart + tid; i < bendEnd; i += stride) {
    const int localIdx1 = bs_idx1s[i] - atomStart;
    const int localIdx2 = bs_idx2s[i] - atomStart;
    const int localIdx3 = bs_idx3s[i] - atomStart;
    bendStretchGrad(molCoords,
                    localIdx1,
                    localIdx2,
                    localIdx3,
                    bs_theta0s[i],
                    restLen1s[i],
                    restLen2s[i],
                    forceConst1s[i],
                    forceConst2s[i],
                    grad);
  }

  const auto& [o_idx1s, o_idx2s, o_idx3s, o_idx4s, koops] = terms.oopTerms;
  const int oopStart                                      = systemIndices.oopTermStarts[molIdx];
  const int oopEnd                                        = systemIndices.oopTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = oopStart + tid; i < oopEnd; i += stride) {
    const int localIdx1 = o_idx1s[i] - atomStart;
    const int localIdx2 = o_idx2s[i] - atomStart;
    const int localIdx3 = o_idx3s[i] - atomStart;
    const int localIdx4 = o_idx4s[i] - atomStart;
    rdkit_ports::oopGrad(molCoords, localIdx1, localIdx2, localIdx3, localIdx4, koops[i], grad);
  }

  const auto& [t_idx1s, t_idx2s, t_idx3s, t_idx4s, V1s, V2s, V3s] = terms.torsionTerms;
  const int torsionStart                                          = systemIndices.torsionTermStarts[molIdx];
  const int torsionEnd                                            = systemIndices.torsionTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = torsionStart + tid; i < torsionEnd; i += stride) {
    const int localIdx1 = t_idx1s[i] - atomStart;
    const int localIdx2 = t_idx2s[i] - atomStart;
    const int localIdx3 = t_idx3s[i] - atomStart;
    const int localIdx4 = t_idx4s[i] - atomStart;
    rdkit_ports::torsionGrad(molCoords, localIdx1, localIdx2, localIdx3, localIdx4, V1s[i], V2s[i], V3s[i], grad);
  }

  const auto& [v_idx1s, v_idx2s, R_ij_stars, wellDepths] = terms.vdwTerms;
  const int vdwStart                                     = systemIndices.vdwTermStarts[molIdx];
  const int vdwEnd                                       = systemIndices.vdwTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = vdwStart + tid; i < vdwEnd; i += stride) {
    const int localIdx1 = v_idx1s[i] - atomStart;
    const int localIdx2 = v_idx2s[i] - atomStart;
    rdkit_ports::vDWGrad(molCoords, localIdx1, localIdx2, R_ij_stars[i], wellDepths[i], grad);
  }

  const auto& [e_idx1s, e_idx2s, chargeTerms, dielModels, is1_4s] = terms.eleTerms;
  const int eleStart                                              = systemIndices.eleTermStarts[molIdx];
  const int eleEnd                                                = systemIndices.eleTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = eleStart + tid; i < eleEnd; i += stride) {
    const int  localIdx1 = e_idx1s[i] - atomStart;
    const int  localIdx2 = e_idx2s[i] - atomStart;
    const bool is14      = is1_4s[i] > 0;
    eleGrad(molCoords, localIdx1, localIdx2, chargeTerms[i], dielModels[i], is14, grad);
  }

  if constexpr (HasConstraints) {
    const auto& [dc_idx1s, dc_idx2s, minLens, maxLens, dcForceConstants] = terms.distanceConstraintTerms;
    const int dcStart = systemIndices.distanceConstraintTermStarts[molIdx];
    const int dcEnd   = systemIndices.distanceConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = dcStart + tid; i < dcEnd; i += stride) {
      const int localIdx1 = dc_idx1s[i] - atomStart;
      const int localIdx2 = dc_idx2s[i] - atomStart;
      distanceConstraintGrad(molCoords, localIdx1, localIdx2, minLens[i], maxLens[i], dcForceConstants[i], grad);
    }

    const auto& [pc_idxs, refXs, refYs, refZs, maxDispls, pcForceConstants] = terms.positionConstraintTerms;
    const int pcStart = systemIndices.positionConstraintTermStarts[molIdx];
    const int pcEnd   = systemIndices.positionConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = pcStart + tid; i < pcEnd; i += stride) {
      const int localIdx = pc_idxs[i] - atomStart;
      positionConstraintGrad(molCoords,
                             localIdx,
                             refXs[i],
                             refYs[i],
                             refZs[i],
                             maxDispls[i],
                             pcForceConstants[i],
                             grad);
    }

    const auto& [ac_idx1s, ac_idx2s, ac_idx3s, minAngleDegs, maxAngleDegs, acForceConstants] =
      terms.angleConstraintTerms;
    const int acStart = systemIndices.angleConstraintTermStarts[molIdx];
    const int acEnd   = systemIndices.angleConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = acStart + tid; i < acEnd; i += stride) {
      const int localIdx1 = ac_idx1s[i] - atomStart;
      const int localIdx2 = ac_idx2s[i] - atomStart;
      const int localIdx3 = ac_idx3s[i] - atomStart;
      angleConstraintGrad(molCoords,
                          localIdx1,
                          localIdx2,
                          localIdx3,
                          minAngleDegs[i],
                          maxAngleDegs[i],
                          acForceConstants[i],
                          grad);
    }

    const auto& [tc_idx1s, tc_idx2s, tc_idx3s, tc_idx4s, minDihedralDegs, maxDihedralDegs, tcForceConstants] =
      terms.torsionConstraintTerms;
    const int tcStart = systemIndices.torsionConstraintTermStarts[molIdx];
    const int tcEnd   = systemIndices.torsionConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = tcStart + tid; i < tcEnd; i += stride) {
      const int localIdx1 = tc_idx1s[i] - atomStart;
      const int localIdx2 = tc_idx2s[i] - atomStart;
      const int localIdx3 = tc_idx3s[i] - atomStart;
      const int localIdx4 = tc_idx4s[i] - atomStart;
      torsionConstraintGrad(molCoords,
                            localIdx1,
                            localIdx2,
                            localIdx3,
                            localIdx4,
                            minDihedralDegs[i],
                            maxDihedralDegs[i],
                            tcForceConstants[i],
                            grad);
    }
  }
}

}  // namespace MMFF
}  // namespace nvMolKit

#endif  // NVMOLKIT_MMFF_KERNELS_DEVICE_CUH
