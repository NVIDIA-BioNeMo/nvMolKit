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

// Instantiated by uff_kernels_device_dispatch.cuh for each arithmetic type.

using namespace nvMolKit::FFKernelUtils;

namespace detail {
constexpr real kUffAngleCorrectionThreshold = real{0.8660};
constexpr real kUffZeroTol                  = real{1.0e-16};

__device__ __forceinline__ real squareValue(const real x) {
  return x * x;
}

__device__ __forceinline__ real cubeValue(const real x) {
  return x * x * x;
}

__device__ __forceinline__ real
uffBondStretchEnergy(const auto* pos, const int idx1, const int idx2, const real restLen, const real forceConstant) {
  const real dist = NVMOLKIT_SQRT(distanceSquared(pos, idx1, idx2));
  const real diff = dist - restLen;
  return real{0.5} * forceConstant * diff * diff;
}

__device__ __forceinline__ void uffBondStretchGrad(const auto* pos,
                                                   const int   idx1,
                                                   const int   idx2,
                                                   const real  restLen,
                                                   const real  forceConstant,
                                                   auto*       grad) {
  real       dx, dy, dz;
  const real distSq = distanceSquaredWithComponents(pos, idx1, idx2, dx, dy, dz);
  const real dist   = NVMOLKIT_SQRT(distSq);
  const real pref   = forceConstant * (dist - restLen);

  real gx = forceConstant * real{0.01};
  real gy = gx;
  real gz = gx;
  if (dist > real{0.0}) {
    const real invDist = real{1.0} / dist;
    gx                 = pref * dx * invDist;
    gy                 = pref * dy * invDist;
    gz                 = pref * dz * invDist;
  }

  atomicAdd(&grad[3 * idx1 + 0], gx);
  atomicAdd(&grad[3 * idx1 + 1], gy);
  atomicAdd(&grad[3 * idx1 + 2], gz);
  atomicAdd(&grad[3 * idx2 + 0], -gx);
  atomicAdd(&grad[3 * idx2 + 1], -gy);
  atomicAdd(&grad[3 * idx2 + 2], -gz);
}

__device__ __forceinline__ real uffAngleEnergyTerm(const real    cosTheta,
                                                   const real    sinThetaSq,
                                                   const uint8_t order,
                                                   const real    C0,
                                                   const real    C1,
                                                   const real    C2) {
  const real cos2Theta = cosTheta * cosTheta - sinThetaSq;
  if (order == 0) {
    return C0 + C1 * cosTheta + C2 * cos2Theta;
  }

  real result = real{0.0};
  switch (order) {
    case 1:
      result = -cosTheta;
      break;
    case 2:
      result = cos2Theta;
      break;
    case 3:
      result = cosTheta * (cosTheta * cosTheta - real{3.0} * sinThetaSq);
      break;
    case 4:
      result =
        squareValue(squareValue(cosTheta)) - real{6.0} * cosTheta * cosTheta * sinThetaSq + squareValue(sinThetaSq);
      break;
    default:
      result = real{0.0};
      break;
  }
  return (real{1.0} - result) / static_cast<real>(order * order);
}

__device__ __forceinline__ real uffAngleThetaDeriv(const real    cosTheta,
                                                   const real    sinTheta,
                                                   const uint8_t order,
                                                   const real    forceConstant,
                                                   const real    C1,
                                                   const real    C2) {
  const real sin2Theta = real{2.0} * sinTheta * cosTheta;
  if (order == 0) {
    return -forceConstant * (C1 * sinTheta + real{2.0} * C2 * sin2Theta);
  }

  real result = real{0.0};
  switch (order) {
    case 1:
      result = -sinTheta;
      break;
    case 2:
      result = sin2Theta;
      break;
    case 3:
      result = sinTheta * (real{3.0} - real{4.0} * sinTheta * sinTheta);
      break;
    case 4:
      result = cosTheta * sinTheta * (real{4.0} - real{8.0} * sinTheta * sinTheta);
      break;
    default:
      return real{0.0};
  }
  return result * forceConstant / static_cast<real>(order);
}

__device__ __forceinline__ real uffAngleBendEnergy(const auto*   pos,
                                                   const int     idx1,
                                                   const int     idx2,
                                                   const int     idx3,
                                                   const real    theta0,
                                                   const real    forceConstant,
                                                   const uint8_t order,
                                                   const real    C0,
                                                   const real    C1,
                                                   const real    C2) {
  real       dx1, dy1, dz1, dx2, dy2, dz2;
  const real dist1Sq = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const real dist2Sq = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const real dist1   = NVMOLKIT_SQRT(dist1Sq);
  const real dist2   = NVMOLKIT_SQRT(dist2Sq);
  if (dist1 <= real{0.0} || dist2 <= real{0.0}) {
    return real{0.0};
  }

  real cosTheta = (dx1 * dx2 + dy1 * dy2 + dz1 * dz2) / (dist1 * dist2);
  clipToOne(cosTheta);
  const real sinThetaSq = real{1.0} - cosTheta * cosTheta;
  real       energy     = forceConstant * uffAngleEnergyTerm(cosTheta, sinThetaSq, order, C0, C1, C2);

  if (order > 0 && order < 5 && cosTheta > kUffAngleCorrectionThreshold) {
    const real theta = NVMOLKIT_ACOS(cosTheta);
    energy += NVMOLKIT_EXP(-real{20.0} * (theta - theta0 + real{0.25}));
  }

  return energy;
}

__device__ __forceinline__ void uffAngleBendGrad(const auto*   pos,
                                                 const int     idx1,
                                                 const int     idx2,
                                                 const int     idx3,
                                                 const real    theta0,
                                                 const real    forceConstant,
                                                 const uint8_t order,
                                                 const real    C0,
                                                 const real    C1,
                                                 const real    C2,
                                                 auto*         grad) {
  real       dx1, dy1, dz1, dx2, dy2, dz2;
  const real dist1Sq = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const real dist2Sq = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  if (dist1Sq <= real{0.0} || dist2Sq <= real{0.0}) {
    return;
  }

  const real dist1    = NVMOLKIT_SQRT(dist1Sq);
  const real dist2    = NVMOLKIT_SQRT(dist2Sq);
  const real invDist1 = real{1.0} / dist1;
  const real invDist2 = real{1.0} / dist2;
  real       cosTheta = (dx1 * dx2 + dy1 * dy2 + dz1 * dz2) * invDist1 * invDist2;
  clipToOne(cosTheta);
  const real sinThetaSq = real{1.0} - cosTheta * cosTheta;
  if (NVMOLKIT_IS_ZERO(sinThetaSq)) {
    return;
  }
  const real sinTheta  = NVMOLKIT_FMAX(NVMOLKIT_SQRT(sinThetaSq), real{1.0e-8});
  real       dE_dTheta = uffAngleThetaDeriv(cosTheta, sinTheta, order, forceConstant, C1, C2);
  if (order > 0 && order < 5 && cosTheta > kUffAngleCorrectionThreshold) {
    const real theta = NVMOLKIT_ACOS(cosTheta);
    dE_dTheta += -real{20.0} * NVMOLKIT_EXP(-real{20.0} * (theta - theta0 + real{0.25}));
  }

  const real ndx1 = dx1 * invDist1;
  const real ndy1 = dy1 * invDist1;
  const real ndz1 = dz1 * invDist1;
  const real ndx2 = dx2 * invDist2;
  const real ndy2 = dy2 * invDist2;
  const real ndz2 = dz2 * invDist2;

  const real common = dE_dTheta / (-sinTheta);

  const real i1 = invDist1 * (ndx2 - cosTheta * ndx1);
  const real i2 = invDist1 * (ndy2 - cosTheta * ndy1);
  const real i3 = invDist1 * (ndz2 - cosTheta * ndz1);
  const real i4 = invDist2 * (ndx1 - cosTheta * ndx2);
  const real i5 = invDist2 * (ndy1 - cosTheta * ndy2);
  const real i6 = invDist2 * (ndz1 - cosTheta * ndz2);

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

__device__ __forceinline__ real
uffCalculateCosTorsion(const auto* pos, const int idx1, const int idx2, const int idx3, const int idx4) {
  const real r1x = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const real r1y = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const real r1z = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];
  const real r2x = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  const real r2y = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  const real r2z = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];
  const real r3x = pos[3 * idx2 + 0] - pos[3 * idx3 + 0];
  const real r3y = pos[3 * idx2 + 1] - pos[3 * idx3 + 1];
  const real r3z = pos[3 * idx2 + 2] - pos[3 * idx3 + 2];
  const real r4x = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
  const real r4y = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
  const real r4z = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

  real t1x, t1y, t1z, t2x, t2y, t2z;
  crossProduct(r1x, r1y, r1z, r2x, r2y, r2z, t1x, t1y, t1z);
  crossProduct(r3x, r3y, r3z, r4x, r4y, r4z, t2x, t2y, t2z);
  const real d1 = NVMOLKIT_SQRT(t1x * t1x + t1y * t1y + t1z * t1z);
  const real d2 = NVMOLKIT_SQRT(t2x * t2x + t2y * t2y + t2z * t2z);
  if (NVMOLKIT_IS_ZERO(d1) || NVMOLKIT_IS_ZERO(d2)) {
    return real{0.0};
  }
  real cosPhi = (t1x * t2x + t1y * t2y + t1z * t2z) / (d1 * d2);
  clipToOne(cosPhi);
  return cosPhi;
}

__device__ __forceinline__ real uffTorsionThetaDeriv(const real    cosTheta,
                                                     const real    sinTheta,
                                                     const real    forceConstant,
                                                     const uint8_t order,
                                                     const real    cosTerm) {
  const real sinThetaSq = sinTheta * sinTheta;
  real       result     = real{0.0};
  switch (order) {
    case 2:
      result = real{2.0} * sinTheta * cosTheta;
      break;
    case 3:
      result = sinTheta * (real{3.0} - real{4.0} * sinThetaSq);
      break;
    case 6:
      result = cosTheta * sinTheta * (real{32.0} * sinThetaSq * (sinThetaSq - real{1.0}) + real{6.0});
      break;
    default:
      return real{0.0};
  }
  return result * forceConstant / real{2.0} * cosTerm * -real{1.0} * static_cast<real>(order);
}

__device__ __forceinline__ real uffTorsionEnergy(const auto*   pos,
                                                 const int     idx1,
                                                 const int     idx2,
                                                 const int     idx3,
                                                 const int     idx4,
                                                 const real    forceConstant,
                                                 const uint8_t order,
                                                 const real    cosTerm) {
  const real cosPhi   = uffCalculateCosTorsion(pos, idx1, idx2, idx3, idx4);
  const real sinPhiSq = real{1.0} - cosPhi * cosPhi;
  real       cosNPhi  = real{0.0};
  switch (order) {
    case 2:
      cosNPhi = real{1.0} - real{2.0} * sinPhiSq;
      break;
    case 3:
      cosNPhi = cosPhi * (cosPhi * cosPhi - real{3.0} * sinPhiSq);
      break;
    case 6:
      cosNPhi = real{1.0} + sinPhiSq * (-real{32.0} * sinPhiSq * sinPhiSq + real{48.0} * sinPhiSq - real{18.0});
      break;
    default:
      return real{0.0};
  }
  return forceConstant / real{2.0} * (real{1.0} - cosTerm * cosNPhi);
}

__device__ __forceinline__ void uffTorsionGrad(const auto*   pos,
                                               const int     idx1,
                                               const int     idx2,
                                               const int     idx3,
                                               const int     idx4,
                                               const real    forceConstant,
                                               const uint8_t order,
                                               const real    cosTerm,
                                               auto*         grad) {
  const real r0x = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const real r0y = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const real r0z = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];
  const real r1x = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  const real r1y = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  const real r1z = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];
  const real r2x = -r1x;
  const real r2y = -r1y;
  const real r2z = -r1z;
  const real r3x = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
  const real r3y = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
  const real r3z = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

  real t0x, t0y, t0z, t1x, t1y, t1z;
  crossProduct(r0x, r0y, r0z, r1x, r1y, r1z, t0x, t0y, t0z);
  crossProduct(r2x, r2y, r2z, r3x, r3y, r3z, t1x, t1y, t1z);

  const real d0 = NVMOLKIT_SQRT(t0x * t0x + t0y * t0y + t0z * t0z);
  const real d1 = NVMOLKIT_SQRT(t1x * t1x + t1y * t1y + t1z * t1z);
  if (NVMOLKIT_IS_ZERO(d0) || NVMOLKIT_IS_ZERO(d1)) {
    return;
  }
  t0x /= d0;
  t0y /= d0;
  t0z /= d0;
  t1x /= d1;
  t1y /= d1;
  t1z /= d1;

  real cosPhi = t0x * t1x + t0y * t1y + t0z * t1z;
  clipToOne(cosPhi);
  const real sinPhiSq = real{1.0} - cosPhi * cosPhi;
  const real sinPhi   = sinPhiSq > real{0.0} ? NVMOLKIT_SQRT(sinPhiSq) : real{0.0};
  const real dE_dPhi  = uffTorsionThetaDeriv(cosPhi, sinPhi, forceConstant, order, cosTerm);
  const real sinTerm  = dE_dPhi * (NVMOLKIT_IS_ZERO(sinPhi) ? (real{1.0} / NVMOLKIT_FMAX(fabs(cosPhi), real{1.0e-8})) :
                                                              (real{1.0} / sinPhi));

  const real dCos_dT0 = (t1x - cosPhi * t0x) / d0;
  const real dCos_dT1 = (t1y - cosPhi * t0y) / d0;
  const real dCos_dT2 = (t1z - cosPhi * t0z) / d0;
  const real dCos_dT3 = (t0x - cosPhi * t1x) / d1;
  const real dCos_dT4 = (t0y - cosPhi * t1y) / d1;
  const real dCos_dT5 = (t0z - cosPhi * t1z) / d1;

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

__device__ __forceinline__ real
uffCalculateCosY(const auto* pos, const int idx1, const int idx2, const int idx3, const int idx4) {
  const real rJIx = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const real rJIy = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const real rJIz = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];
  const real rJKx = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  const real rJKy = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  const real rJKz = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];
  const real rJLx = pos[3 * idx4 + 0] - pos[3 * idx2 + 0];
  const real rJLy = pos[3 * idx4 + 1] - pos[3 * idx2 + 1];
  const real rJLz = pos[3 * idx4 + 2] - pos[3 * idx2 + 2];

  const real l2JI = rJIx * rJIx + rJIy * rJIy + rJIz * rJIz;
  const real l2JK = rJKx * rJKx + rJKy * rJKy + rJKz * rJKz;
  const real l2JL = rJLx * rJLx + rJLy * rJLy + rJLz * rJLz;
  if (l2JI < kUffZeroTol || l2JK < kUffZeroTol || l2JL < kUffZeroTol) {
    return real{0.0};
  }

  real nx, ny, nz;
  crossProduct(rJIx, rJIy, rJIz, rJKx, rJKy, rJKz, nx, ny, nz);
  const real normScale = NVMOLKIT_SQRT(l2JI) * NVMOLKIT_SQRT(l2JK);
  nx /= normScale;
  ny /= normScale;
  nz /= normScale;
  const real l2n = nx * nx + ny * ny + nz * nz;
  if (l2n < kUffZeroTol) {
    return real{0.0};
  }
  return (nx * rJLx + ny * rJLy + nz * rJLz) / (NVMOLKIT_SQRT(l2JL) * NVMOLKIT_SQRT(l2n));
}

__device__ __forceinline__ real uffInversionEnergy(const auto* pos,
                                                   const int   idx1,
                                                   const int   idx2,
                                                   const int   idx3,
                                                   const int   idx4,
                                                   const real  forceConstant,
                                                   const real  C0,
                                                   const real  C1,
                                                   const real  C2) {
  const real cosY   = uffCalculateCosY(pos, idx1, idx2, idx3, idx4);
  const real sinYSq = real{1.0} - cosY * cosY;
  const real sinY   = sinYSq > real{0.0} ? NVMOLKIT_SQRT(sinYSq) : real{0.0};
  const real cos2W  = real{2.0} * sinY * sinY - real{1.0};
  return forceConstant * (C0 + C1 * sinY + C2 * cos2W);
}

__device__ __forceinline__ void uffInversionGrad(const auto* pos,
                                                 const int   idx1,
                                                 const int   idx2,
                                                 const int   idx3,
                                                 const int   idx4,
                                                 const real  forceConstant,
                                                 const real  C1,
                                                 const real  C2,
                                                 auto*       grad) {
  real rJIx = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  real rJIy = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  real rJIz = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];
  real rJKx = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  real rJKy = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  real rJKz = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];
  real rJLx = pos[3 * idx4 + 0] - pos[3 * idx2 + 0];
  real rJLy = pos[3 * idx4 + 1] - pos[3 * idx2 + 1];
  real rJLz = pos[3 * idx4 + 2] - pos[3 * idx2 + 2];

  real dJI = NVMOLKIT_SQRT(rJIx * rJIx + rJIy * rJIy + rJIz * rJIz);
  real dJK = NVMOLKIT_SQRT(rJKx * rJKx + rJKy * rJKy + rJKz * rJKz);
  real dJL = NVMOLKIT_SQRT(rJLx * rJLx + rJLy * rJLy + rJLz * rJLz);
  if (NVMOLKIT_IS_ZERO(dJI) || NVMOLKIT_IS_ZERO(dJK) || NVMOLKIT_IS_ZERO(dJL)) {
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

  real nx, ny, nz;
  crossProduct(-rJIx, -rJIy, -rJIz, rJKx, rJKy, rJKz, nx, ny, nz);
  const real nNorm = NVMOLKIT_SQRT(nx * nx + ny * ny + nz * nz);
  if (nNorm <= real{0.0}) {
    return;
  }
  nx /= nNorm;
  ny /= nNorm;
  nz /= nNorm;

  real cosY = nx * rJLx + ny * rJLy + nz * rJLz;
  clipToOne(cosY);
  const real sinYSq   = real{1.0} - cosY * cosY;
  const real sinY     = NVMOLKIT_FMAX(NVMOLKIT_SQRT(sinYSq), real{1.0e-8});
  real       cosTheta = rJIx * rJKx + rJIy * rJKy + rJIz * rJKz;
  clipToOne(cosTheta);
  const real sinThetaSq = real{1.0} - cosTheta * cosTheta;
  const real sinTheta   = NVMOLKIT_FMAX(NVMOLKIT_SQRT(sinThetaSq), real{1.0e-8});

  const real dE_dW = -forceConstant * (C1 * cosY - real{4.0} * C2 * cosY * sinY);
  real       t1x, t1y, t1z, t2x, t2y, t2z, t3x, t3y, t3z;
  crossProduct(rJLx, rJLy, rJLz, rJKx, rJKy, rJKz, t1x, t1y, t1z);
  crossProduct(rJIx, rJIy, rJIz, rJLx, rJLy, rJLz, t2x, t2y, t2z);
  crossProduct(rJKx, rJKy, rJKz, rJIx, rJIy, rJIz, t3x, t3y, t3z);
  const real term1 = sinY * sinTheta;
  const real term2 = cosY / (sinY * sinThetaSq);

  const real tg1x = (t1x / term1 - (rJIx - rJKx * cosTheta) * term2) / dJI;
  const real tg1y = (t1y / term1 - (rJIy - rJKy * cosTheta) * term2) / dJI;
  const real tg1z = (t1z / term1 - (rJIz - rJKz * cosTheta) * term2) / dJI;
  const real tg3x = (t2x / term1 - (rJKx - rJIx * cosTheta) * term2) / dJK;
  const real tg3y = (t2y / term1 - (rJKy - rJIy * cosTheta) * term2) / dJK;
  const real tg3z = (t2z / term1 - (rJKz - rJIz * cosTheta) * term2) / dJK;
  const real tg4x = (t3x / term1 - rJLx * cosY / sinY) / dJL;
  const real tg4y = (t3y / term1 - rJLy * cosY / sinY) / dJL;
  const real tg4z = (t3z / term1 - rJLz * cosY / sinY) / dJL;

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

__device__ __forceinline__ real uffVdwEnergy(const auto* pos,
                                             const int   idx1,
                                             const int   idx2,
                                             const real  x_ij,
                                             const real  wellDepth,
                                             const real  threshold) {
  const real dist = NVMOLKIT_SQRT(distanceSquared(pos, idx1, idx2));
  if (dist > threshold || dist <= real{0.0}) {
    return real{0.0};
  }
  const real r   = x_ij / dist;
  const real r6  = cubeValue(squareValue(r));
  const real r12 = r6 * r6;
  return wellDepth * (r12 - real{2.0} * r6);
}

__device__ __forceinline__ void uffVdwGrad(const auto* pos,
                                           const int   idx1,
                                           const int   idx2,
                                           const real  x_ij,
                                           const real  wellDepth,
                                           const real  threshold,
                                           auto*       grad) {
  const real dist = NVMOLKIT_SQRT(distanceSquared(pos, idx1, idx2));
  if (dist > threshold) {
    return;
  }
  if (dist <= real{0.0}) {
    atomicAdd(&grad[3 * idx1 + 0], real{100.0});
    atomicAdd(&grad[3 * idx1 + 1], real{100.0});
    atomicAdd(&grad[3 * idx1 + 2], real{100.0});
    atomicAdd(&grad[3 * idx2 + 0], -real{100.0});
    atomicAdd(&grad[3 * idx2 + 1], -real{100.0});
    atomicAdd(&grad[3 * idx2 + 2], -real{100.0});
    return;
  }

  const real r         = x_ij / dist;
  const real r7        = r * cubeValue(squareValue(r));
  const real r13       = r7 * squareValue(squareValue(r)) * squareValue(r);
  const real preFactor = real{12.0} * wellDepth / x_ij * (r7 - r13);

  const real dx = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const real dy = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const real dz = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];
  const real gx = preFactor * dx / dist;
  const real gy = preFactor * dy / dist;
  const real gz = preFactor * dz / dist;

  atomicAdd(&grad[3 * idx1 + 0], gx);
  atomicAdd(&grad[3 * idx1 + 1], gy);
  atomicAdd(&grad[3 * idx1 + 2], gz);
  atomicAdd(&grad[3 * idx2 + 0], -gx);
  atomicAdd(&grad[3 * idx2 + 1], -gy);
  atomicAdd(&grad[3 * idx2 + 2], -gz);
}

}  // namespace detail
using namespace detail;

template <int stride, bool HasConstraints, typename Terms>
__device__ __inline__ real molEnergy(const Terms&                   terms,
                                     const BatchedIndicesDevicePtr& systemIndices,
                                     const auto*                    molCoords,
                                     const int                      molIdx,
                                     const int                      tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];
  real      energy    = real{0.0};

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

template <int stride, bool HasConstraints, typename Terms>
__device__ __inline__ void molGrad(const Terms&                   terms,
                                   const BatchedIndicesDevicePtr& systemIndices,
                                   const auto*                    molCoords,
                                   auto*                          grad,
                                   const int                      molIdx,
                                   const int                      tid) {
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
