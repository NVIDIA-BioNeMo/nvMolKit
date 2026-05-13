// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_UFF_KERNELS_DEVICE_CUH
#define NVMOLKIT_UFF_KERNELS_DEVICE_CUH

#include <cmath>

#include "mmff_kernels_device.cuh"

using namespace nvMolKit::FFKernelUtils;

namespace {
constexpr double kUffAngleCorrectionThreshold = 0.8660;
constexpr double kUffZeroTol                  = 1.0e-16;

__device__ __forceinline__ double squareValue(const double x) {
  return x * x;
}

__device__ __forceinline__ double cubeValue(const double x) {
  return x * x * x;
}

__device__ __forceinline__ double uffBondStretchEnergy(const double* pos,
                                                       const int     idx1,
                                                       const int     idx2,
                                                       const double  restLen,
                                                       const double  forceConstant) {
  const double dist = sqrt(distanceSquared(pos, idx1, idx2));
  const double diff = dist - restLen;
  return 0.5 * forceConstant * diff * diff;
}

__device__ __forceinline__ void uffBondStretchGrad(const double* pos,
                                                   const int     idx1,
                                                   const int     idx2,
                                                   const double  restLen,
                                                   const double  forceConstant,
                                                   double*       grad) {
  double       dx, dy, dz;
  const double distSq = distanceSquaredWithComponents(pos, idx1, idx2, dx, dy, dz);
  const double dist   = sqrt(distSq);
  const double pref   = forceConstant * (dist - restLen);

  double gx = forceConstant * 0.01;
  double gy = gx;
  double gz = gx;
  if (dist > 0.0) {
    const double invDist = 1.0 / dist;
    gx                   = pref * dx * invDist;
    gy                   = pref * dy * invDist;
    gz                   = pref * dz * invDist;
  }

  atomicAdd(&grad[3 * idx1 + 0], gx);
  atomicAdd(&grad[3 * idx1 + 1], gy);
  atomicAdd(&grad[3 * idx1 + 2], gz);
  atomicAdd(&grad[3 * idx2 + 0], -gx);
  atomicAdd(&grad[3 * idx2 + 1], -gy);
  atomicAdd(&grad[3 * idx2 + 2], -gz);
}

__device__ __forceinline__ double uffAngleEnergyTerm(const double  cosTheta,
                                                     const double  sinThetaSq,
                                                     const uint8_t order,
                                                     const double  C0,
                                                     const double  C1,
                                                     const double  C2) {
  const double cos2Theta = cosTheta * cosTheta - sinThetaSq;
  if (order == 0) {
    return C0 + C1 * cosTheta + C2 * cos2Theta;
  }

  double result = 0.0;
  switch (order) {
    case 1:
      result = -cosTheta;
      break;
    case 2:
      result = cos2Theta;
      break;
    case 3:
      result = cosTheta * (cosTheta * cosTheta - 3.0 * sinThetaSq);
      break;
    case 4:
      result = squareValue(squareValue(cosTheta)) - 6.0 * cosTheta * cosTheta * sinThetaSq + squareValue(sinThetaSq);
      break;
    default:
      result = 0.0;
      break;
  }
  return (1.0 - result) / static_cast<double>(order * order);
}

__device__ __forceinline__ double uffAngleThetaDeriv(const double  cosTheta,
                                                     const double  sinTheta,
                                                     const uint8_t order,
                                                     const double  forceConstant,
                                                     const double  C1,
                                                     const double  C2) {
  const double sin2Theta = 2.0 * sinTheta * cosTheta;
  if (order == 0) {
    return -forceConstant * (C1 * sinTheta + 2.0 * C2 * sin2Theta);
  }

  double result = 0.0;
  switch (order) {
    case 1:
      result = -sinTheta;
      break;
    case 2:
      result = sin2Theta;
      break;
    case 3:
      result = sinTheta * (3.0 - 4.0 * sinTheta * sinTheta);
      break;
    case 4:
      result = cosTheta * sinTheta * (4.0 - 8.0 * sinTheta * sinTheta);
      break;
    default:
      return 0.0;
  }
  return result * forceConstant / static_cast<double>(order);
}

__device__ __forceinline__ double uffAngleBendEnergy(const double* pos,
                                                     const int     idx1,
                                                     const int     idx2,
                                                     const int     idx3,
                                                     const double  theta0,
                                                     const double  forceConstant,
                                                     const uint8_t order,
                                                     const double  C0,
                                                     const double  C1,
                                                     const double  C2) {
  double       dx1, dy1, dz1, dx2, dy2, dz2;
  const double dist1Sq = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const double dist2Sq = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const double dist1   = sqrt(dist1Sq);
  const double dist2   = sqrt(dist2Sq);
  if (dist1 <= 0.0 || dist2 <= 0.0) {
    return 0.0;
  }

  double cosTheta = (dx1 * dx2 + dy1 * dy2 + dz1 * dz2) / (dist1 * dist2);
  clipToOne(cosTheta);
  const double sinThetaSq = 1.0 - cosTheta * cosTheta;
  double       energy     = forceConstant * uffAngleEnergyTerm(cosTheta, sinThetaSq, order, C0, C1, C2);

  if (order > 0 && order < 5 && cosTheta > kUffAngleCorrectionThreshold) {
    const double theta = acos(cosTheta);
    energy += exp(-20.0 * (theta - theta0 + 0.25));
  }

  return energy;
}

__device__ __forceinline__ void uffAngleBendGrad(const double* pos,
                                                 const int     idx1,
                                                 const int     idx2,
                                                 const int     idx3,
                                                 const double  theta0,
                                                 const double  forceConstant,
                                                 const uint8_t order,
                                                 const double  C0,
                                                 const double  C1,
                                                 const double  C2,
                                                 double*       grad) {
  double       dx1, dy1, dz1, dx2, dy2, dz2;
  const double dist1Sq = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const double dist2Sq = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  if (dist1Sq <= 0.0 || dist2Sq <= 0.0) {
    return;
  }

  const double dist1    = sqrt(dist1Sq);
  const double dist2    = sqrt(dist2Sq);
  const double invDist1 = 1.0 / dist1;
  const double invDist2 = 1.0 / dist2;
  double       cosTheta = (dx1 * dx2 + dy1 * dy2 + dz1 * dz2) * invDist1 * invDist2;
  clipToOne(cosTheta);
  const double sinThetaSq = 1.0 - cosTheta * cosTheta;
  if (isDoubleZero(sinThetaSq)) {
    return;
  }
  const double sinTheta  = fmax(sqrt(sinThetaSq), 1.0e-8);
  double       dE_dTheta = uffAngleThetaDeriv(cosTheta, sinTheta, order, forceConstant, C1, C2);
  if (order > 0 && order < 5 && cosTheta > kUffAngleCorrectionThreshold) {
    const double theta = acos(cosTheta);
    dE_dTheta += -20.0 * exp(-20.0 * (theta - theta0 + 0.25));
  }

  const double ndx1 = dx1 * invDist1;
  const double ndy1 = dy1 * invDist1;
  const double ndz1 = dz1 * invDist1;
  const double ndx2 = dx2 * invDist2;
  const double ndy2 = dy2 * invDist2;
  const double ndz2 = dz2 * invDist2;

  const double common = dE_dTheta / (-sinTheta);

  const double i1 = invDist1 * (ndx2 - cosTheta * ndx1);
  const double i2 = invDist1 * (ndy2 - cosTheta * ndy1);
  const double i3 = invDist1 * (ndz2 - cosTheta * ndz1);
  const double i4 = invDist2 * (ndx1 - cosTheta * ndx2);
  const double i5 = invDist2 * (ndy1 - cosTheta * ndy2);
  const double i6 = invDist2 * (ndz1 - cosTheta * ndz2);

  atomicAdd(&grad[3 * idx1 + 0], common * i1);
  atomicAdd(&grad[3 * idx1 + 1], common * i2);
  atomicAdd(&grad[3 * idx1 + 2], common * i3);

  atomicAdd(&grad[3 * idx2 + 0], common * (-i1 - i4));
  atomicAdd(&grad[3 * idx2 + 1], common * (-i2 - i5));
  atomicAdd(&grad[3 * idx2 + 2], common * (-i3 - i6));

  atomicAdd(&grad[3 * idx3 + 0], common * i4);
  atomicAdd(&grad[3 * idx3 + 1], common * i5);
  atomicAdd(&grad[3 * idx3 + 2], common * i6);
}

__device__ __forceinline__ double uffCalculateCosTorsion(const double* pos,
                                                         const int     idx1,
                                                         const int     idx2,
                                                         const int     idx3,
                                                         const int     idx4) {
  const double r1x = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const double r1y = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const double r1z = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];
  const double r2x = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  const double r2y = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  const double r2z = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];
  const double r3x = pos[3 * idx2 + 0] - pos[3 * idx3 + 0];
  const double r3y = pos[3 * idx2 + 1] - pos[3 * idx3 + 1];
  const double r3z = pos[3 * idx2 + 2] - pos[3 * idx3 + 2];
  const double r4x = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
  const double r4y = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
  const double r4z = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

  double t1x, t1y, t1z, t2x, t2y, t2z;
  crossProduct(r1x, r1y, r1z, r2x, r2y, r2z, t1x, t1y, t1z);
  crossProduct(r3x, r3y, r3z, r4x, r4y, r4z, t2x, t2y, t2z);
  const double d1 = sqrt(t1x * t1x + t1y * t1y + t1z * t1z);
  const double d2 = sqrt(t2x * t2x + t2y * t2y + t2z * t2z);
  if (isDoubleZero(d1) || isDoubleZero(d2)) {
    return 0.0;
  }
  double cosPhi = (t1x * t2x + t1y * t2y + t1z * t2z) / (d1 * d2);
  clipToOne(cosPhi);
  return cosPhi;
}

__device__ __forceinline__ double uffTorsionThetaDeriv(const double  cosTheta,
                                                       const double  sinTheta,
                                                       const double  forceConstant,
                                                       const uint8_t order,
                                                       const double  cosTerm) {
  const double sinThetaSq = sinTheta * sinTheta;
  double       result     = 0.0;
  switch (order) {
    case 2:
      result = 2.0 * sinTheta * cosTheta;
      break;
    case 3:
      result = sinTheta * (3.0 - 4.0 * sinThetaSq);
      break;
    case 6:
      result = cosTheta * sinTheta * (32.0 * sinThetaSq * (sinThetaSq - 1.0) + 6.0);
      break;
    default:
      return 0.0;
  }
  return result * forceConstant / 2.0 * cosTerm * -1.0 * static_cast<double>(order);
}

__device__ __forceinline__ double uffTorsionEnergy(const double* pos,
                                                   const int     idx1,
                                                   const int     idx2,
                                                   const int     idx3,
                                                   const int     idx4,
                                                   const double  forceConstant,
                                                   const uint8_t order,
                                                   const double  cosTerm) {
  const double cosPhi   = uffCalculateCosTorsion(pos, idx1, idx2, idx3, idx4);
  const double sinPhiSq = 1.0 - cosPhi * cosPhi;
  double       cosNPhi  = 0.0;
  switch (order) {
    case 2:
      cosNPhi = 1.0 - 2.0 * sinPhiSq;
      break;
    case 3:
      cosNPhi = cosPhi * (cosPhi * cosPhi - 3.0 * sinPhiSq);
      break;
    case 6:
      cosNPhi = 1.0 + sinPhiSq * (-32.0 * sinPhiSq * sinPhiSq + 48.0 * sinPhiSq - 18.0);
      break;
    default:
      return 0.0;
  }
  return forceConstant / 2.0 * (1.0 - cosTerm * cosNPhi);
}

__device__ __forceinline__ void uffTorsionGrad(const double* pos,
                                               const int     idx1,
                                               const int     idx2,
                                               const int     idx3,
                                               const int     idx4,
                                               const double  forceConstant,
                                               const uint8_t order,
                                               const double  cosTerm,
                                               double*       grad) {
  const double r0x = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const double r0y = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const double r0z = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];
  const double r1x = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  const double r1y = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  const double r1z = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];
  const double r2x = -r1x;
  const double r2y = -r1y;
  const double r2z = -r1z;
  const double r3x = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
  const double r3y = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
  const double r3z = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

  double t0x, t0y, t0z, t1x, t1y, t1z;
  crossProduct(r0x, r0y, r0z, r1x, r1y, r1z, t0x, t0y, t0z);
  crossProduct(r2x, r2y, r2z, r3x, r3y, r3z, t1x, t1y, t1z);

  const double d0 = sqrt(t0x * t0x + t0y * t0y + t0z * t0z);
  const double d1 = sqrt(t1x * t1x + t1y * t1y + t1z * t1z);
  if (isDoubleZero(d0) || isDoubleZero(d1)) {
    return;
  }
  t0x /= d0;
  t0y /= d0;
  t0z /= d0;
  t1x /= d1;
  t1y /= d1;
  t1z /= d1;

  double cosPhi = t0x * t1x + t0y * t1y + t0z * t1z;
  clipToOne(cosPhi);
  const double sinPhiSq = 1.0 - cosPhi * cosPhi;
  const double sinPhi   = sinPhiSq > 0.0 ? sqrt(sinPhiSq) : 0.0;
  const double dE_dPhi  = uffTorsionThetaDeriv(cosPhi, sinPhi, forceConstant, order, cosTerm);
  const double sinTerm  = dE_dPhi * (isDoubleZero(sinPhi) ? (1.0 / fmax(fabs(cosPhi), 1.0e-8)) : (1.0 / sinPhi));

  const double dCos_dT0 = (t1x - cosPhi * t0x) / d0;
  const double dCos_dT1 = (t1y - cosPhi * t0y) / d0;
  const double dCos_dT2 = (t1z - cosPhi * t0z) / d0;
  const double dCos_dT3 = (t0x - cosPhi * t1x) / d1;
  const double dCos_dT4 = (t0y - cosPhi * t1y) / d1;
  const double dCos_dT5 = (t0z - cosPhi * t1z) / d1;

  atomicAdd(&grad[3 * idx1 + 0], sinTerm * (dCos_dT2 * r1y - dCos_dT1 * r1z));
  atomicAdd(&grad[3 * idx1 + 1], sinTerm * (dCos_dT0 * r1z - dCos_dT2 * r1x));
  atomicAdd(&grad[3 * idx1 + 2], sinTerm * (dCos_dT1 * r1x - dCos_dT0 * r1y));

  atomicAdd(&grad[3 * idx2 + 0],
            sinTerm * (dCos_dT1 * (r1z - r0z) + dCos_dT2 * (r0y - r1y) + dCos_dT4 * (-r3z) + dCos_dT5 * (r3y)));
  atomicAdd(&grad[3 * idx2 + 1],
            sinTerm * (dCos_dT0 * (r0z - r1z) + dCos_dT2 * (r1x - r0x) + dCos_dT3 * (r3z) + dCos_dT5 * (-r3x)));
  atomicAdd(&grad[3 * idx2 + 2],
            sinTerm * (dCos_dT0 * (r1y - r0y) + dCos_dT1 * (r0x - r1x) + dCos_dT3 * (-r3y) + dCos_dT4 * (r3x)));

  atomicAdd(&grad[3 * idx3 + 0],
            sinTerm * (dCos_dT1 * r0z + dCos_dT2 * (-r0y) + dCos_dT4 * (r3z - r2z) + dCos_dT5 * (r2y - r3y)));
  atomicAdd(&grad[3 * idx3 + 1],
            sinTerm * (dCos_dT0 * (-r0z) + dCos_dT2 * r0x + dCos_dT3 * (r2z - r3z) + dCos_dT5 * (r3x - r2x)));
  atomicAdd(&grad[3 * idx3 + 2],
            sinTerm * (dCos_dT0 * r0y + dCos_dT1 * (-r0x) + dCos_dT3 * (r3y - r2y) + dCos_dT4 * (r2x - r3x)));

  atomicAdd(&grad[3 * idx4 + 0], sinTerm * (dCos_dT4 * r2z - dCos_dT5 * r2y));
  atomicAdd(&grad[3 * idx4 + 1], sinTerm * (dCos_dT5 * r2x - dCos_dT3 * r2z));
  atomicAdd(&grad[3 * idx4 + 2], sinTerm * (dCos_dT3 * r2y - dCos_dT4 * r2x));
}

__device__ __forceinline__ double uffCalculateCosY(const double* pos,
                                                   const int     idx1,
                                                   const int     idx2,
                                                   const int     idx3,
                                                   const int     idx4) {
  const double rJIx = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const double rJIy = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const double rJIz = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];
  const double rJKx = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  const double rJKy = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  const double rJKz = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];
  const double rJLx = pos[3 * idx4 + 0] - pos[3 * idx2 + 0];
  const double rJLy = pos[3 * idx4 + 1] - pos[3 * idx2 + 1];
  const double rJLz = pos[3 * idx4 + 2] - pos[3 * idx2 + 2];

  const double l2JI = rJIx * rJIx + rJIy * rJIy + rJIz * rJIz;
  const double l2JK = rJKx * rJKx + rJKy * rJKy + rJKz * rJKz;
  const double l2JL = rJLx * rJLx + rJLy * rJLy + rJLz * rJLz;
  if (l2JI < kUffZeroTol || l2JK < kUffZeroTol || l2JL < kUffZeroTol) {
    return 0.0;
  }

  double nx, ny, nz;
  crossProduct(rJIx, rJIy, rJIz, rJKx, rJKy, rJKz, nx, ny, nz);
  const double normScale = sqrt(l2JI) * sqrt(l2JK);
  nx /= normScale;
  ny /= normScale;
  nz /= normScale;
  const double l2n = nx * nx + ny * ny + nz * nz;
  if (l2n < kUffZeroTol) {
    return 0.0;
  }
  return (nx * rJLx + ny * rJLy + nz * rJLz) / (sqrt(l2JL) * sqrt(l2n));
}

__device__ __forceinline__ double uffInversionEnergy(const double* pos,
                                                     const int     idx1,
                                                     const int     idx2,
                                                     const int     idx3,
                                                     const int     idx4,
                                                     const double  forceConstant,
                                                     const double  C0,
                                                     const double  C1,
                                                     const double  C2) {
  const double cosY   = uffCalculateCosY(pos, idx1, idx2, idx3, idx4);
  const double sinYSq = 1.0 - cosY * cosY;
  const double sinY   = sinYSq > 0.0 ? sqrt(sinYSq) : 0.0;
  const double cos2W  = 2.0 * sinY * sinY - 1.0;
  return forceConstant * (C0 + C1 * sinY + C2 * cos2W);
}

__device__ __forceinline__ void uffInversionGrad(const double* pos,
                                                 const int     idx1,
                                                 const int     idx2,
                                                 const int     idx3,
                                                 const int     idx4,
                                                 const double  forceConstant,
                                                 const double  C1,
                                                 const double  C2,
                                                 double*       grad) {
  double rJIx = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  double rJIy = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  double rJIz = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];
  double rJKx = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  double rJKy = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  double rJKz = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];
  double rJLx = pos[3 * idx4 + 0] - pos[3 * idx2 + 0];
  double rJLy = pos[3 * idx4 + 1] - pos[3 * idx2 + 1];
  double rJLz = pos[3 * idx4 + 2] - pos[3 * idx2 + 2];

  double dJI = sqrt(rJIx * rJIx + rJIy * rJIy + rJIz * rJIz);
  double dJK = sqrt(rJKx * rJKx + rJKy * rJKy + rJKz * rJKz);
  double dJL = sqrt(rJLx * rJLx + rJLy * rJLy + rJLz * rJLz);
  if (isDoubleZero(dJI) || isDoubleZero(dJK) || isDoubleZero(dJL)) {
    return;
  }
  rJIx /= dJI;
  rJIy /= dJI;
  rJIz /= dJI;
  rJKx /= dJK;
  rJKy /= dJK;
  rJKz /= dJK;
  rJLx /= dJL;
  rJLy /= dJL;
  rJLz /= dJL;

  double nx, ny, nz;
  crossProduct(-rJIx, -rJIy, -rJIz, rJKx, rJKy, rJKz, nx, ny, nz);
  const double nNorm = sqrt(nx * nx + ny * ny + nz * nz);
  if (nNorm <= 0.0) {
    return;
  }
  nx /= nNorm;
  ny /= nNorm;
  nz /= nNorm;

  double cosY = nx * rJLx + ny * rJLy + nz * rJLz;
  clipToOne(cosY);
  const double sinYSq   = 1.0 - cosY * cosY;
  const double sinY     = fmax(sqrt(sinYSq), 1.0e-8);
  double       cosTheta = rJIx * rJKx + rJIy * rJKy + rJIz * rJKz;
  clipToOne(cosTheta);
  const double sinThetaSq = 1.0 - cosTheta * cosTheta;
  const double sinTheta   = fmax(sqrt(sinThetaSq), 1.0e-8);

  const double dE_dW = -forceConstant * (C1 * cosY - 4.0 * C2 * cosY * sinY);
  double       t1x, t1y, t1z, t2x, t2y, t2z, t3x, t3y, t3z;
  crossProduct(rJLx, rJLy, rJLz, rJKx, rJKy, rJKz, t1x, t1y, t1z);
  crossProduct(rJIx, rJIy, rJIz, rJLx, rJLy, rJLz, t2x, t2y, t2z);
  crossProduct(rJKx, rJKy, rJKz, rJIx, rJIy, rJIz, t3x, t3y, t3z);
  const double term1 = sinY * sinTheta;
  const double term2 = cosY / (sinY * sinThetaSq);

  const double tg1x = (t1x / term1 - (rJIx - rJKx * cosTheta) * term2) / dJI;
  const double tg1y = (t1y / term1 - (rJIy - rJKy * cosTheta) * term2) / dJI;
  const double tg1z = (t1z / term1 - (rJIz - rJKz * cosTheta) * term2) / dJI;
  const double tg3x = (t2x / term1 - (rJKx - rJIx * cosTheta) * term2) / dJK;
  const double tg3y = (t2y / term1 - (rJKy - rJIy * cosTheta) * term2) / dJK;
  const double tg3z = (t2z / term1 - (rJKz - rJIz * cosTheta) * term2) / dJK;
  const double tg4x = (t3x / term1 - rJLx * cosY / sinY) / dJL;
  const double tg4y = (t3y / term1 - rJLy * cosY / sinY) / dJL;
  const double tg4z = (t3z / term1 - rJLz * cosY / sinY) / dJL;

  atomicAdd(&grad[3 * idx1 + 0], dE_dW * tg1x);
  atomicAdd(&grad[3 * idx1 + 1], dE_dW * tg1y);
  atomicAdd(&grad[3 * idx1 + 2], dE_dW * tg1z);
  atomicAdd(&grad[3 * idx2 + 0], -dE_dW * (tg1x + tg3x + tg4x));
  atomicAdd(&grad[3 * idx2 + 1], -dE_dW * (tg1y + tg3y + tg4y));
  atomicAdd(&grad[3 * idx2 + 2], -dE_dW * (tg1z + tg3z + tg4z));
  atomicAdd(&grad[3 * idx3 + 0], dE_dW * tg3x);
  atomicAdd(&grad[3 * idx3 + 1], dE_dW * tg3y);
  atomicAdd(&grad[3 * idx3 + 2], dE_dW * tg3z);
  atomicAdd(&grad[3 * idx4 + 0], dE_dW * tg4x);
  atomicAdd(&grad[3 * idx4 + 1], dE_dW * tg4y);
  atomicAdd(&grad[3 * idx4 + 2], dE_dW * tg4z);
}

__device__ __forceinline__ double uffVdwEnergy(const double* pos,
                                               const int     idx1,
                                               const int     idx2,
                                               const double  x_ij,
                                               const double  wellDepth,
                                               const double  threshold) {
  const double dist = sqrt(distanceSquared(pos, idx1, idx2));
  if (dist > threshold || dist <= 0.0) {
    return 0.0;
  }
  const double r   = x_ij / dist;
  const double r6  = cubeValue(squareValue(r));
  const double r12 = r6 * r6;
  return wellDepth * (r12 - 2.0 * r6);
}

__device__ __forceinline__ void uffVdwGrad(const double* pos,
                                           const int     idx1,
                                           const int     idx2,
                                           const double  x_ij,
                                           const double  wellDepth,
                                           const double  threshold,
                                           double*       grad) {
  const double dist = sqrt(distanceSquared(pos, idx1, idx2));
  if (dist > threshold) {
    return;
  }
  if (dist <= 0.0) {
    atomicAdd(&grad[3 * idx1 + 0], 100.0);
    atomicAdd(&grad[3 * idx1 + 1], 100.0);
    atomicAdd(&grad[3 * idx1 + 2], 100.0);
    atomicAdd(&grad[3 * idx2 + 0], -100.0);
    atomicAdd(&grad[3 * idx2 + 1], -100.0);
    atomicAdd(&grad[3 * idx2 + 2], -100.0);
    return;
  }

  const double r         = x_ij / dist;
  const double r7        = r * cubeValue(squareValue(r));
  const double r13       = r7 * squareValue(squareValue(r)) * squareValue(r);
  const double preFactor = 12.0 * wellDepth / x_ij * (r7 - r13);

  const double dx = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const double dy = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const double dz = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];
  const double gx = preFactor * dx / dist;
  const double gy = preFactor * dy / dist;
  const double gz = preFactor * dz / dist;

  atomicAdd(&grad[3 * idx1 + 0], gx);
  atomicAdd(&grad[3 * idx1 + 1], gy);
  atomicAdd(&grad[3 * idx1 + 2], gz);
  atomicAdd(&grad[3 * idx2 + 0], -gx);
  atomicAdd(&grad[3 * idx2 + 1], -gy);
  atomicAdd(&grad[3 * idx2 + 2], -gz);
}

}  // namespace

namespace nvMolKit {
namespace UFF {

template <int stride, bool HasConstraints>
__device__ __inline__ double molEnergy(const EnergyForceContribsDevicePtr& terms,
                                       const BatchedIndicesDevicePtr&      systemIndices,
                                       const double*                       molCoords,
                                       const int                           molIdx,
                                       const int                           tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];
  double    energy    = 0.0;

  const int bondStart = systemIndices.bondTermStarts[molIdx];
  const int bondEnd   = systemIndices.bondTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = bondStart + tid; i < bondEnd; i += stride) {
    energy += uffBondStretchEnergy(molCoords,
                                   terms.bondTerms.idx1[i] - atomStart,
                                   terms.bondTerms.idx2[i] - atomStart,
                                   terms.bondTerms.restLen[i],
                                   terms.bondTerms.forceConstant[i]);
  }

  const int angleStart = systemIndices.angleTermStarts[molIdx];
  const int angleEnd   = systemIndices.angleTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = angleStart + tid; i < angleEnd; i += stride) {
    energy += uffAngleBendEnergy(molCoords,
                                 terms.angleTerms.idx1[i] - atomStart,
                                 terms.angleTerms.idx2[i] - atomStart,
                                 terms.angleTerms.idx3[i] - atomStart,
                                 terms.angleTerms.theta0[i],
                                 terms.angleTerms.forceConstant[i],
                                 terms.angleTerms.order[i],
                                 terms.angleTerms.C0[i],
                                 terms.angleTerms.C1[i],
                                 terms.angleTerms.C2[i]);
  }

  const int torsionStart = systemIndices.torsionTermStarts[molIdx];
  const int torsionEnd   = systemIndices.torsionTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = torsionStart + tid; i < torsionEnd; i += stride) {
    energy += uffTorsionEnergy(molCoords,
                               terms.torsionTerms.idx1[i] - atomStart,
                               terms.torsionTerms.idx2[i] - atomStart,
                               terms.torsionTerms.idx3[i] - atomStart,
                               terms.torsionTerms.idx4[i] - atomStart,
                               terms.torsionTerms.forceConstant[i],
                               terms.torsionTerms.order[i],
                               terms.torsionTerms.cosTerm[i]);
  }

  const int inversionStart = systemIndices.inversionTermStarts[molIdx];
  const int inversionEnd   = systemIndices.inversionTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = inversionStart + tid; i < inversionEnd; i += stride) {
    energy += uffInversionEnergy(molCoords,
                                 terms.inversionTerms.idx1[i] - atomStart,
                                 terms.inversionTerms.idx2[i] - atomStart,
                                 terms.inversionTerms.idx3[i] - atomStart,
                                 terms.inversionTerms.idx4[i] - atomStart,
                                 terms.inversionTerms.forceConstant[i],
                                 terms.inversionTerms.C0[i],
                                 terms.inversionTerms.C1[i],
                                 terms.inversionTerms.C2[i]);
  }

  const int vdwStart = systemIndices.vdwTermStarts[molIdx];
  const int vdwEnd   = systemIndices.vdwTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = vdwStart + tid; i < vdwEnd; i += stride) {
    energy += uffVdwEnergy(molCoords,
                           terms.vdwTerms.idx1[i] - atomStart,
                           terms.vdwTerms.idx2[i] - atomStart,
                           terms.vdwTerms.x_ij[i],
                           terms.vdwTerms.wellDepth[i],
                           terms.vdwTerms.threshold[i]);
  }

  if constexpr (HasConstraints) {
    const int dcStart = systemIndices.distanceConstraintTermStarts[molIdx];
    const int dcEnd   = systemIndices.distanceConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = dcStart + tid; i < dcEnd; i += stride) {
      energy += distanceConstraintEnergy(molCoords,
                                         terms.distanceConstraintTerms.idx1[i] - atomStart,
                                         terms.distanceConstraintTerms.idx2[i] - atomStart,
                                         terms.distanceConstraintTerms.minLen[i],
                                         terms.distanceConstraintTerms.maxLen[i],
                                         terms.distanceConstraintTerms.forceConstant[i]);
    }

    const int pcStart = systemIndices.positionConstraintTermStarts[molIdx];
    const int pcEnd   = systemIndices.positionConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = pcStart + tid; i < pcEnd; i += stride) {
      energy += positionConstraintEnergy(molCoords,
                                         terms.positionConstraintTerms.idx[i] - atomStart,
                                         terms.positionConstraintTerms.refX[i],
                                         terms.positionConstraintTerms.refY[i],
                                         terms.positionConstraintTerms.refZ[i],
                                         terms.positionConstraintTerms.maxDispl[i],
                                         terms.positionConstraintTerms.forceConstant[i]);
    }

    const int acStart = systemIndices.angleConstraintTermStarts[molIdx];
    const int acEnd   = systemIndices.angleConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = acStart + tid; i < acEnd; i += stride) {
      energy += angleConstraintEnergy(molCoords,
                                      terms.angleConstraintTerms.idx1[i] - atomStart,
                                      terms.angleConstraintTerms.idx2[i] - atomStart,
                                      terms.angleConstraintTerms.idx3[i] - atomStart,
                                      terms.angleConstraintTerms.minAngleDeg[i],
                                      terms.angleConstraintTerms.maxAngleDeg[i],
                                      terms.angleConstraintTerms.forceConstant[i]);
    }

    const int tcStart = systemIndices.torsionConstraintTermStarts[molIdx];
    const int tcEnd   = systemIndices.torsionConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = tcStart + tid; i < tcEnd; i += stride) {
      energy += torsionConstraintEnergy(molCoords,
                                        terms.torsionConstraintTerms.idx1[i] - atomStart,
                                        terms.torsionConstraintTerms.idx2[i] - atomStart,
                                        terms.torsionConstraintTerms.idx3[i] - atomStart,
                                        terms.torsionConstraintTerms.idx4[i] - atomStart,
                                        terms.torsionConstraintTerms.minDihedralDeg[i],
                                        terms.torsionConstraintTerms.maxDihedralDeg[i],
                                        terms.torsionConstraintTerms.forceConstant[i]);
    }
  }

  return energy;
}

template <int stride, bool HasConstraints>
__device__ __inline__ void molGrad(const EnergyForceContribsDevicePtr& terms,
                                   const BatchedIndicesDevicePtr&      systemIndices,
                                   const double*                       molCoords,
                                   double*                             grad,
                                   const int                           molIdx,
                                   const int                           tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];

  const int bondStart = systemIndices.bondTermStarts[molIdx];
  const int bondEnd   = systemIndices.bondTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = bondStart + tid; i < bondEnd; i += stride) {
    uffBondStretchGrad(molCoords,
                       terms.bondTerms.idx1[i] - atomStart,
                       terms.bondTerms.idx2[i] - atomStart,
                       terms.bondTerms.restLen[i],
                       terms.bondTerms.forceConstant[i],
                       grad);
  }

  const int angleStart = systemIndices.angleTermStarts[molIdx];
  const int angleEnd   = systemIndices.angleTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = angleStart + tid; i < angleEnd; i += stride) {
    uffAngleBendGrad(molCoords,
                     terms.angleTerms.idx1[i] - atomStart,
                     terms.angleTerms.idx2[i] - atomStart,
                     terms.angleTerms.idx3[i] - atomStart,
                     terms.angleTerms.theta0[i],
                     terms.angleTerms.forceConstant[i],
                     terms.angleTerms.order[i],
                     terms.angleTerms.C0[i],
                     terms.angleTerms.C1[i],
                     terms.angleTerms.C2[i],
                     grad);
  }

  const int torsionStart = systemIndices.torsionTermStarts[molIdx];
  const int torsionEnd   = systemIndices.torsionTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = torsionStart + tid; i < torsionEnd; i += stride) {
    uffTorsionGrad(molCoords,
                   terms.torsionTerms.idx1[i] - atomStart,
                   terms.torsionTerms.idx2[i] - atomStart,
                   terms.torsionTerms.idx3[i] - atomStart,
                   terms.torsionTerms.idx4[i] - atomStart,
                   terms.torsionTerms.forceConstant[i],
                   terms.torsionTerms.order[i],
                   terms.torsionTerms.cosTerm[i],
                   grad);
  }

  const int inversionStart = systemIndices.inversionTermStarts[molIdx];
  const int inversionEnd   = systemIndices.inversionTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = inversionStart + tid; i < inversionEnd; i += stride) {
    uffInversionGrad(molCoords,
                     terms.inversionTerms.idx1[i] - atomStart,
                     terms.inversionTerms.idx2[i] - atomStart,
                     terms.inversionTerms.idx3[i] - atomStart,
                     terms.inversionTerms.idx4[i] - atomStart,
                     terms.inversionTerms.forceConstant[i],
                     terms.inversionTerms.C1[i],
                     terms.inversionTerms.C2[i],
                     grad);
  }

  const int vdwStart = systemIndices.vdwTermStarts[molIdx];
  const int vdwEnd   = systemIndices.vdwTermStarts[molIdx + 1];
#pragma unroll 1
  for (int i = vdwStart + tid; i < vdwEnd; i += stride) {
    uffVdwGrad(molCoords,
               terms.vdwTerms.idx1[i] - atomStart,
               terms.vdwTerms.idx2[i] - atomStart,
               terms.vdwTerms.x_ij[i],
               terms.vdwTerms.wellDepth[i],
               terms.vdwTerms.threshold[i],
               grad);
  }

  if constexpr (HasConstraints) {
    const int dcStart = systemIndices.distanceConstraintTermStarts[molIdx];
    const int dcEnd   = systemIndices.distanceConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = dcStart + tid; i < dcEnd; i += stride) {
      distanceConstraintGrad(molCoords,
                             terms.distanceConstraintTerms.idx1[i] - atomStart,
                             terms.distanceConstraintTerms.idx2[i] - atomStart,
                             terms.distanceConstraintTerms.minLen[i],
                             terms.distanceConstraintTerms.maxLen[i],
                             terms.distanceConstraintTerms.forceConstant[i],
                             grad);
    }

    const int pcStart = systemIndices.positionConstraintTermStarts[molIdx];
    const int pcEnd   = systemIndices.positionConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = pcStart + tid; i < pcEnd; i += stride) {
      positionConstraintGrad(molCoords,
                             terms.positionConstraintTerms.idx[i] - atomStart,
                             terms.positionConstraintTerms.refX[i],
                             terms.positionConstraintTerms.refY[i],
                             terms.positionConstraintTerms.refZ[i],
                             terms.positionConstraintTerms.maxDispl[i],
                             terms.positionConstraintTerms.forceConstant[i],
                             grad);
    }

    const int acStart = systemIndices.angleConstraintTermStarts[molIdx];
    const int acEnd   = systemIndices.angleConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = acStart + tid; i < acEnd; i += stride) {
      angleConstraintGrad(molCoords,
                          terms.angleConstraintTerms.idx1[i] - atomStart,
                          terms.angleConstraintTerms.idx2[i] - atomStart,
                          terms.angleConstraintTerms.idx3[i] - atomStart,
                          terms.angleConstraintTerms.minAngleDeg[i],
                          terms.angleConstraintTerms.maxAngleDeg[i],
                          terms.angleConstraintTerms.forceConstant[i],
                          grad);
    }

    const int tcStart = systemIndices.torsionConstraintTermStarts[molIdx];
    const int tcEnd   = systemIndices.torsionConstraintTermStarts[molIdx + 1];
#pragma unroll 1
    for (int i = tcStart + tid; i < tcEnd; i += stride) {
      torsionConstraintGrad(molCoords,
                            terms.torsionConstraintTerms.idx1[i] - atomStart,
                            terms.torsionConstraintTerms.idx2[i] - atomStart,
                            terms.torsionConstraintTerms.idx3[i] - atomStart,
                            terms.torsionConstraintTerms.idx4[i] - atomStart,
                            terms.torsionConstraintTerms.minDihedralDeg[i],
                            terms.torsionConstraintTerms.maxDihedralDeg[i],
                            terms.torsionConstraintTerms.forceConstant[i],
                            grad);
    }
  }
}

}  // namespace UFF
}  // namespace nvMolKit

#endif  // NVMOLKIT_UFF_KERNELS_DEVICE_CUH
