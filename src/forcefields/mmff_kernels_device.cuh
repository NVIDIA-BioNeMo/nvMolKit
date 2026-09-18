// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Instantiated by mmff_kernels_device_dispatch.cuh for each arithmetic type.

using namespace nvMolKit::FFKernelUtils;

constexpr real degreeToRadian = static_cast<real>(M_PI) / real{180.0};
constexpr real radianToDegree = real{180.0} / static_cast<real>(M_PI);

namespace rdkit_ports {

static __device__ __forceinline__ void
oopGrad(const auto* pos, const int idx1, const int idx2, const int idx3, const int idx4, const real koop, auto* grad) {
  constexpr real prefactor = real{143.9325} * degreeToRadian;

  real dJIx = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  real dJIy = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  real dJIz = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

  real dJKx = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  real dJKy = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  real dJKz = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

  real dJLx = pos[3 * idx4 + 0] - pos[3 * idx2 + 0];
  real dJLy = pos[3 * idx4 + 1] - pos[3 * idx2 + 1];
  real dJLz = pos[3 * idx4 + 2] - pos[3 * idx2 + 2];

  const real invdJI = NVMOLKIT_RSQRT(dJIx * dJIx + dJIy * dJIy + dJIz * dJIz);
  const real invdJK = NVMOLKIT_RSQRT(dJKx * dJKx + dJKy * dJKy + dJKz * dJKz);
  const real invdJL = NVMOLKIT_RSQRT(dJLx * dJLx + dJLy * dJLy + dJLz * dJLz);

  dJIx *= invdJI;
  dJIy *= invdJI;
  dJIz *= invdJI;
  dJKx *= invdJK;
  dJKy *= invdJK;
  dJKz *= invdJK;
  dJLx *= invdJL;
  dJLy *= invdJL;
  dJLz *= invdJL;

  real normalJIKx, normalJIKy, normalJIKz;
  crossProduct(-dJIx, -dJIy, -dJIz, dJKx, dJKy, dJKz, normalJIKx, normalJIKy, normalJIKz);
  const real invNormLength =
    NVMOLKIT_RSQRT(normalJIKx * normalJIKx + normalJIKy * normalJIKy + normalJIKz * normalJIKz);
  normalJIKx *= invNormLength;
  normalJIKy *= invNormLength;
  normalJIKz *= invNormLength;

  const real sinChi    = clamp(dotProduct(dJLx, dJLy, dJLz, normalJIKx, normalJIKy, normalJIKz), real{-1.0}, real{1.0});
  const real cosChiSq  = real{1.0} - sinChi * sinChi;
  const real invCosChi = cosChiSq > 0 ? NVMOLKIT_RSQRT(cosChiSq) : real{1.0e8};
  const real chi       = radianToDegree * NVMOLKIT_ASIN(sinChi);
  const real cosTheta  = clamp(dotProduct(dJIx, dJIy, dJIz, dJKx, dJKy, dJKz), real{-1.0}, real{1.0});

  real invSinTheta = NVMOLKIT_RSQRT(NVMOLKIT_FMAX(real{1.0} - cosTheta * cosTheta, real{1.0e-8}));

  real dE_dChi = prefactor * koop * chi;
  real t1x, t1y, t1z, t2x, t2y, t2z, t3x, t3y, t3z;
  crossProduct(dJLx, dJLy, dJLz, dJKx, dJKy, dJKz, t1x, t1y, t1z);
  crossProduct(dJIx, dJIy, dJIz, dJLx, dJLy, dJLz, t2x, t2y, t2z);
  crossProduct(dJKx, dJKy, dJKz, dJIx, dJIy, dJIz, t3x, t3y, t3z);

  real term1  = invCosChi * invSinTheta;
  real term2  = sinChi * invCosChi * (invSinTheta * invSinTheta);
  real tg1[3] = {(t1x * term1 - (dJIx - dJKx * cosTheta) * term2) * invdJI,
                 (t1y * term1 - (dJIy - dJKy * cosTheta) * term2) * invdJI,
                 (t1z * term1 - (dJIz - dJKz * cosTheta) * term2) * invdJI};
  real tg3[3] = {(t2x * term1 - (dJKx - dJIx * cosTheta) * term2) * invdJK,
                 (t2y * term1 - (dJKy - dJIy * cosTheta) * term2) * invdJK,
                 (t2z * term1 - (dJKz - dJIz * cosTheta) * term2) * invdJK};
  real tg4[3] = {(t3x * term1 - dJLx * sinChi * invCosChi) * invdJL,
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

static __device__ __forceinline__ void torsionGrad(const auto* pos,
                                                   const int   idx1,
                                                   const int   idx2,
                                                   const int   idx3,
                                                   const int   idx4,
                                                   const real  V1,
                                                   const real  V2,
                                                   const real  V3,
                                                   auto*       grad) {
  // P1 - P2
  const real dx1 = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const real dy1 = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const real dz1 = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

  // P3 - P2
  const real dx2 = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  const real dy2 = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  const real dz2 = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

  // P4 - P3
  const real dx4 = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
  const real dy4 = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
  const real dz4 = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

  real cross1x, cross1y, cross1z;
  crossProduct(dx1, dy1, dz1, dx2, dy2, dz2, cross1x, cross1y, cross1z);
  const real invNorm1 =
    NVMOLKIT_FMIN(NVMOLKIT_RSQRT(cross1x * cross1x + cross1y * cross1y + cross1z * cross1z), real{1.0e5});
  cross1x *= invNorm1;
  cross1y *= invNorm1;
  cross1z *= invNorm1;

  real cross2x, cross2y, cross2z;
  // Use -dx2, -dy2, -dz2 directly instead of storing dx3, dy3, dz3
  crossProduct(-dx2, -dy2, -dz2, dx4, dy4, dz4, cross2x, cross2y, cross2z);
  const real invNorm2 =
    NVMOLKIT_FMIN(NVMOLKIT_RSQRT(cross2x * cross2x + cross2y * cross2y + cross2z * cross2z), real{1.0e5});
  cross2x *= invNorm2;
  cross2y *= invNorm2;
  cross2z *= invNorm2;

  const real cosPhi = clamp(dotProduct(cross1x, cross1y, cross1z, cross2x, cross2y, cross2z), -real{1.0}, real{1.0});

  const real sinPhiSq = 1.0f - cosPhi * cosPhi;
  real       sinTerm  = real{0.0};
  if (sinPhiSq > real{0.0}) {
    const real sin2Phi = 2.0f * cosPhi;
    const real sin3Phi = 3.0f - 4.0f * sinPhiSq;
    sinTerm            = 0.5f * (V1 - 2.0f * V2 * sin2Phi + 3.0f * V3 * sin3Phi);
  }

  real dCos_dT0 = invNorm1 * (cross2x - cosPhi * cross1x);
  real dCos_dT1 = invNorm1 * (cross2y - cosPhi * cross1y);
  real dCos_dT2 = invNorm1 * (cross2z - cosPhi * cross1z);

  atomicAdd(&grad[3 * idx1 + 0], sinTerm * (dCos_dT2 * dy2 - dCos_dT1 * dz2));
  atomicAdd(&grad[3 * idx1 + 1], sinTerm * (dCos_dT0 * dz2 - dCos_dT2 * dx2));
  atomicAdd(&grad[3 * idx1 + 2], sinTerm * (dCos_dT1 * dx2 - dCos_dT0 * dy2));

  // idx3 and idx4 gradients - reuse variables dCos_dT0-2 for dCos_dT3-5
  const real dCos_dT3 = invNorm2 * (cross1x - cosPhi * cross2x);
  const real dCos_dT4 = invNorm2 * (cross1y - cosPhi * cross2y);
  const real dCos_dT5 = invNorm2 * (cross1z - cosPhi * cross2z);

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
static __device__ __forceinline__ void vDWGrad(const auto* pos,
                                               const int   idx1,
                                               const int   idx2,
                                               const real  R_ij_star,
                                               const real  wellDepth,
                                               auto*       grad) {
  constexpr real vdw1   = real{1.07};
  constexpr real vdw1m1 = vdw1 - real{1.0};
  constexpr real vdw2   = real{1.12};
  constexpr real vdw2m1 = vdw2 - real{1.0};
  constexpr real vdw2t7 = vdw2 * real{7.0};

  const real invDistance = NVMOLKIT_RSQRT(distanceSquared(pos, idx1, idx2));
  const real distance    = 1.0f / invDistance;

  const real invRIJStar = 1.0f / R_ij_star;

  const real q         = distance * invRIJStar;
  const real q2        = q * q;
  const real q6        = q2 * q2 * q2;
  const real q7        = q6 * q;
  const real q7pvdw2m1 = q7 + vdw2m1;
  const real invQ7Term = 1.0f / q7pvdw2m1;
  const real t         = vdw1 / (q + vdw1 - real{1.0});
  const real t2        = t * t;
  const real t7        = t2 * t2 * t2 * t;
  const real dE_dr     = wellDepth * invRIJStar * t7 *
                     (-vdw2t7 * q6 * invQ7Term * invQ7Term + ((-vdw2t7 * invQ7Term + real{14.0}) / (q + vdw1m1)));

  real term1x, term1y, term1z;
  if (distance <= real{0.0}) {
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

static __device__ __forceinline__ real
bondStretchEnergy(const auto* pos, const int idx1, const int idx2, const real r0, const real kb) {
  constexpr real prefactor           = real{143.9325} / real{2.0};
  constexpr real csFactorDist        = -real{2.0};
  constexpr real csFactorDistSquared = real{7.0} / real{12.0} * csFactorDist * csFactorDist;

  const real distSquared = distanceSquared(pos, idx1, idx2);
  const real distance    = NVMOLKIT_SQRT(static_cast<real>(distSquared));

  const real deltaR  = distance - r0;
  const real deltaR2 = deltaR * deltaR;
  return prefactor * kb * deltaR2 * (real{1.0} + csFactorDist * deltaR + csFactorDistSquared * deltaR2);
}

static __device__ __forceinline__ void bondStretchGrad(const auto* pos,
                                                       const int   idx1,
                                                       const int   idx2,
                                                       const real  r0,
                                                       const real  kb,
                                                       auto*       grad) {
  constexpr real c1                          = real{143.9325};
  constexpr real cs                          = -real{2.0};
  constexpr real csFactorTimesSecondConstant = cs * real{1.5};
  constexpr real lastFactor                  = real{2.0} * real{7.0} / real{12.0} * cs * cs;

  real       dx, dy, dz;
  const real distanceSquared = distanceSquaredWithComponents(pos, idx1, idx2, dx, dy, dz);
  const real invDist         = NVMOLKIT_RSQRT(distanceSquared);
  const real distance        = real{1.0} / invDist;
  const real deltaR          = distance - r0;

  const real de_dr =
    c1 * kb * deltaR * (real{1.0} + csFactorTimesSecondConstant * deltaR + lastFactor * deltaR * deltaR);

  real dE_dx, dE_dy, dE_dz;
  if (distance > real{0.0}) {
    dE_dx = de_dr * dx * invDist;
    dE_dy = de_dr * dy * invDist;
    dE_dz = de_dr * dz * invDist;
  } else {
    dE_dx = kb * real{0.01};
    dE_dy = kb * real{0.01};
    dE_dz = kb * real{0.01};
  }

  atomicAdd(&grad[3 * idx1 + 0], dE_dx);
  atomicAdd(&grad[3 * idx1 + 1], dE_dy);
  atomicAdd(&grad[3 * idx1 + 2], dE_dz);

  atomicAdd(&grad[3 * idx2 + 0], -dE_dx);
  atomicAdd(&grad[3 * idx2 + 1], -dE_dy);
  atomicAdd(&grad[3 * idx2 + 2], -dE_dz);
}

static __device__ __forceinline__ real angleBendEnergy(const auto* pos,
                                                       const int   idx1,
                                                       const int   idx2,
                                                       const int   idx3,
                                                       const real  theta0,
                                                       const real  ka,
                                                       const bool  isLinear) {
  constexpr real prefactor = real{0.5} * real{143.9325} * degreeToRadian * degreeToRadian;
  constexpr real cb        = -real{0.4} * degreeToRadian;

  real       dx1, dy1, dz1, dx2, dy2, dz2;
  const real dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const real dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const real dist1        = NVMOLKIT_RSQRT(dist1Squared);
  const real dist2        = NVMOLKIT_RSQRT(dist2Squared);

  const real dot         = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const real cosTheta    = clamp(dot * (dist1 * dist2), real{-1.0}, real{1.0});
  const real theta       = radianToDegree * NVMOLKIT_ACOS(cosTheta);
  const real deltaTheta  = theta - theta0;
  const real deltaTheta2 = deltaTheta * deltaTheta;

  if (isLinear) {
    constexpr real linearPrefactor = real{143.9325};
    return linearPrefactor * ka * (real{1.0} + cosTheta);
  }
  return prefactor * ka * deltaTheta2 * (real{1.0} + cb * deltaTheta);
}

static __device__ __forceinline__ void angleBendGrad(const int   idx1,
                                                     const int   idx2,
                                                     const int   idx3,
                                                     const real  theta0,
                                                     const real  ka,
                                                     const bool  isLinear,
                                                     const auto* pos,
                                                     auto*       grad) {
  constexpr real c1       = real{143.9325} * degreeToRadian;
  constexpr real cbFactor = -real{0.006981317} * real{1.5};
  real           dx1, dy1, dz1, dx2, dy2, dz2;
  const real     dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const real     dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const real     invDist1     = NVMOLKIT_RSQRT(dist1Squared);
  const real     invDist2     = NVMOLKIT_RSQRT(dist2Squared);

  const real dot        = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const real cosTheta   = clamp(dot * invDist1 * invDist2, -real{1.0}, real{1.0});
  const real sinThetaSq = real{1.0} - cosTheta * cosTheta;
  if (NVMOLKIT_IS_ZERO(sinThetaSq) || NVMOLKIT_IS_ZERO(dist1Squared) || NVMOLKIT_IS_ZERO(dist2Squared)) {
    return;
  }

  const real invNegSinTheta = -NVMOLKIT_RSQRT(sinThetaSq);
  const real theta          = radianToDegree * NVMOLKIT_ACOS(cosTheta);
  const real deltaTheta     = theta - theta0;

  real de_dDeltaTheta;

  if (isLinear) {
    constexpr real linearPrefactor = real{143.9325};
    de_dDeltaTheta                 = -linearPrefactor * ka * NVMOLKIT_SQRT(real{1.0} - (cosTheta * cosTheta));
  } else {
    de_dDeltaTheta = c1 * ka * deltaTheta * (real{1.0} + cbFactor * deltaTheta);
  }

  const real dxnorm1 = dx1 * invDist1;
  const real dynorm1 = dy1 * invDist1;
  const real dznorm1 = dz1 * invDist1;
  const real dxnorm2 = dx2 * invDist2;
  const real dynorm2 = dy2 * invDist2;
  const real dznorm2 = dz2 * invDist2;

  const real intermediate1 = invDist1 * (dxnorm2 - cosTheta * dxnorm1);
  const real intermediate2 = invDist1 * (dynorm2 - cosTheta * dynorm1);
  const real intermediate3 = invDist1 * (dznorm2 - cosTheta * dznorm1);
  const real intermediate4 = invDist2 * (dxnorm1 - cosTheta * dxnorm2);
  const real intermediate5 = invDist2 * (dynorm1 - cosTheta * dynorm2);
  const real intermediate6 = invDist2 * (dznorm1 - cosTheta * dznorm2);

  const real constantFactor = de_dDeltaTheta * invNegSinTheta;

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

static __device__ __forceinline__ real bendStretchEnergy(const auto* pos,
                                                         const int   idx1,
                                                         const int   idx2,
                                                         const int   idx3,
                                                         const real  theta0,
                                                         const real  restLen1,
                                                         const real  restLen2,
                                                         const real  forceConst1,
                                                         const real  forceConst2) {
  constexpr real prefactor = real{2.51210};

  real       dx1, dy1, dz1, dx2, dy2, dz2;
  const real dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const real dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const real dist1        = NVMOLKIT_SQRT(dist1Squared);
  const real dist2        = NVMOLKIT_SQRT(dist2Squared);

  const real dot      = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const real cosTheta = clamp(dot / (dist1 * dist2), real{-1.0}, real{1.0});
  const real theta    = 180 / static_cast<real>(M_PI) * NVMOLKIT_ACOS(cosTheta);

  const real deltaTheta = theta - theta0;
  const real deltaR1    = dist1 - restLen1;
  const real deltaR2    = dist2 - restLen2;

  return prefactor * deltaTheta * (deltaR1 * forceConst1 + deltaR2 * forceConst2);
}

static __device__ __forceinline__ void bendStretchGrad(const auto* pos,
                                                       const int   idx1,
                                                       const int   idx2,
                                                       const int   idx3,
                                                       const real  theta0,
                                                       const real  restLen1,
                                                       const real  restLen2,
                                                       const real  forceConst1,
                                                       const real  forceConst2,
                                                       auto*       grad) {
  constexpr real prefactor = real{143.9325} * static_cast<real>(M_PI) / real{180.0};

  real       dx1, dy1, dz1, dx2, dy2, dz2;
  const real dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const real dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const real dist1        = NVMOLKIT_SQRT(dist1Squared);
  const real dist2        = NVMOLKIT_SQRT(dist2Squared);
  const real invDist1     = real{1.0} / dist1;
  const real invDist2     = real{1.0} / dist2;
  const real dot          = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const real cosTheta     = clamp(dot * invDist1 * invDist2, real{-1.0}, real{1.0});
  const real invSinTheta  = NVMOLKIT_FMIN(NVMOLKIT_RSQRT(real{1.0} - cosTheta * cosTheta), real{1.0e8});

  constexpr real bondFactor = real{180.0} / static_cast<real>(M_PI);
  const real     theta      = bondFactor * NVMOLKIT_ACOS(cosTheta);

  const real deltaTheta = theta - theta0;
  const real deltaR1    = dist1 - restLen1;
  const real deltaR2    = dist2 - restLen2;

  const real bondEnergyTerm = bondFactor * (forceConst1 * deltaR1 + forceConst2 * deltaR2);

  const real scaledDx1 = dx1 * invDist1;
  const real scaledDy1 = dy1 * invDist1;
  const real scaledDz1 = dz1 * invDist1;
  const real scaledDx2 = dx2 * invDist2;
  const real scaledDy2 = dy2 * invDist2;
  const real scaledDz2 = dz2 * invDist2;

  const real intermediate1 = invDist1 * (scaledDx2 - cosTheta * scaledDx1);
  const real intermediate2 = invDist1 * (scaledDy2 - cosTheta * scaledDy1);
  const real intermediate3 = invDist1 * (scaledDz2 - cosTheta * scaledDz1);
  const real intermediate4 = invDist2 * (scaledDx1 - cosTheta * scaledDx2);
  const real intermediate5 = invDist2 * (scaledDy1 - cosTheta * scaledDy2);
  const real intermediate6 = invDist2 * (scaledDz1 - cosTheta * scaledDz2);

  const real bondEnergyTimesInvSinTheta = bondEnergyTerm * invSinTheta;

  const real gradx1 = prefactor * (deltaTheta * scaledDx1 * forceConst1 - intermediate1 * bondEnergyTimesInvSinTheta);
  const real grady1 = prefactor * (deltaTheta * scaledDy1 * forceConst1 - intermediate2 * bondEnergyTimesInvSinTheta);
  const real gradz1 = prefactor * (deltaTheta * scaledDz1 * forceConst1 - intermediate3 * bondEnergyTimesInvSinTheta);

  const real gradx2 = prefactor * (-deltaTheta * (scaledDx1 * forceConst1 + scaledDx2 * forceConst2) +
                                   (intermediate1 + intermediate4) * bondEnergyTimesInvSinTheta);
  const real grady2 = prefactor * (-deltaTheta * (scaledDy1 * forceConst1 + scaledDy2 * forceConst2) +
                                   (intermediate2 + intermediate5) * bondEnergyTimesInvSinTheta);
  const real gradz2 = prefactor * (-deltaTheta * (scaledDz1 * forceConst1 + scaledDz2 * forceConst2) +
                                   (intermediate3 + intermediate6) * bondEnergyTimesInvSinTheta);

  const real gradx3 = prefactor * (deltaTheta * scaledDx2 * forceConst2 - intermediate4 * bondEnergyTimesInvSinTheta);
  const real grady3 = prefactor * (deltaTheta * scaledDy2 * forceConst2 - intermediate5 * bondEnergyTimesInvSinTheta);
  const real gradz3 = prefactor * (deltaTheta * scaledDz2 * forceConst2 - intermediate6 * bondEnergyTimesInvSinTheta);

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

static __device__ __forceinline__ real
oopBendEnergy(const auto* pos, const int idx1, const int idx2, const int idx3, const int idx4, const real koop) {
  constexpr real prefactor = real{0.5} * real{143.9325} * degreeToRadian * degreeToRadian;

  real       dxji, dyji, dzji, dxjk, dyjk, dzjk, dxjl, dyjl, dzjl;
  const real distSquaredJI = distanceSquaredWithComponents(pos, idx1, idx2, dxji, dyji, dzji);
  const real distSquaredJK = distanceSquaredWithComponents(pos, idx3, idx2, dxjk, dyjk, dzjk);
  const real distSquaredJL = distanceSquaredWithComponents(pos, idx4, idx2, dxjl, dyjl, dzjl);

  const real invDistJI = NVMOLKIT_RSQRT(distSquaredJI);
  const real invDistJK = NVMOLKIT_RSQRT(distSquaredJK);
  const real invDistJL = NVMOLKIT_RSQRT(distSquaredJL);

  const real scaledDxJI = dxji * invDistJI;
  const real scaledDyJI = dyji * invDistJI;
  const real scaledDzJI = dzji * invDistJI;

  const real scaledDxJK = dxjk * invDistJK;
  const real scaledDyJK = dyjk * invDistJK;
  const real scaledDzJK = dzjk * invDistJK;

  const real scaledDxJL = dxjl * invDistJL;
  const real scaledDyJL = dyjl * invDistJL;
  const real scaledDzJL = dzjl * invDistJL;

  real crossX, crossY, crossZ;
  crossProduct(scaledDxJI, scaledDyJI, scaledDzJI, scaledDxJK, scaledDyJK, scaledDzJK, crossX, crossY, crossZ);
  const real invDistCross = NVMOLKIT_RSQRT(crossX * crossX + crossY * crossY + crossZ * crossZ);

  const real scaledCrossX = crossX * invDistCross;
  const real scaledCrossY = crossY * invDistCross;
  const real scaledCrossZ = crossZ * invDistCross;

  const real dotProduct = scaledCrossX * scaledDxJL + scaledCrossY * scaledDyJL + scaledCrossZ * scaledDzJL;
  const real chi        = radianToDegree * NVMOLKIT_ASIN(clamp(dotProduct, real{-1.0}, real{1.0}));

  return prefactor * koop * chi * chi;
}

static __device__ __forceinline__ real torsionEnergy(const auto* pos,
                                                     const int   idx1,
                                                     const int   idx2,
                                                     const int   idx3,
                                                     const int   idx4,
                                                     const real  V1,
                                                     const real  V2,
                                                     const real  V3) {
  const real dxIJ = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const real dyIJ = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const real dzIJ = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

  const real dxKJ = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  const real dyKJ = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  const real dzKJ = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

  const real dxLK = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
  const real dyLK = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
  const real dzLK = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

  const real crossIJKJx = dyIJ * dzKJ - dzIJ * dyKJ;
  const real crossIJKJy = dzIJ * dxKJ - dxIJ * dzKJ;
  const real crossIJKJz = dxIJ * dyKJ - dyIJ * dxKJ;

  const real crossJKLKx = -dyKJ * dzLK + dzKJ * dyLK;
  const real crossJKLKy = -dzKJ * dxLK + dxKJ * dzLK;
  const real crossJKLKz = -dxKJ * dyLK + dyKJ * dxLK;

  const real invCross1Norm =
    NVMOLKIT_RSQRT(crossIJKJx * crossIJKJx + crossIJKJy * crossIJKJy + crossIJKJz * crossIJKJz);
  const real invCross2Norm =
    NVMOLKIT_RSQRT(crossJKLKx * crossJKLKx + crossJKLKy * crossJKLKy + crossJKLKz * crossJKLKz);

  const real dotProduct = crossIJKJx * crossJKLKx + crossIJKJy * crossJKLKy + crossIJKJz * crossJKLKz;
  const real cosPhi     = dotProduct * invCross1Norm * invCross2Norm;
  const real phi        = NVMOLKIT_ACOS(clamp(cosPhi, real{-1.0}, real{1.0}));

  return real{0.5} * (V1 * (real{1.0} + cosPhi) + V2 * (real{1.0} - NVMOLKIT_COS(real{2.0} * phi)) +
                      V3 * (real{1.0} + NVMOLKIT_COS(real{3.0} * phi)));
}

static __device__ __forceinline__ real
vdwEnergy(const auto* pos, const int idx1, const int idx2, const real R_ij_star, const real wellDepth) {
  real R_ij_star2 = R_ij_star * R_ij_star;
  real R_ij_star7 = R_ij_star2 * R_ij_star2 * R_ij_star2 * R_ij_star;

  const real epsilon = wellDepth;

  const real distSquared = distanceSquared(pos, idx1, idx2);
  const real dist        = NVMOLKIT_SQRT(distSquared);
  const real dist7       = distSquared * distSquared * distSquared * dist;

  const real term1        = real{1.07} * R_ij_star / (dist + real{0.07} * R_ij_star);
  const real term1Squared = term1 * term1;
  const real term1_7th    = term1Squared * term1Squared * term1Squared * term1;

  const real term2Fraction = real{1.12} * R_ij_star7 / (dist7 + real{0.12} * R_ij_star7);

  return epsilon * term1_7th * (term2Fraction - real{2.0});
}

static __device__ __forceinline__ real eleEnergy(const auto* pos,
                                                 const int   idx1,
                                                 const int   idx2,
                                                 const real  chargeTerm,
                                                 const int   dielModel,
                                                 const bool  is1_4) {
  constexpr real prefactor         = real{332.0716};
  constexpr real bufferingConstant = real{0.05};
  const real     distSquared       = distanceSquared(pos, idx1, idx2);
  real           distTerm          = NVMOLKIT_SQRT(distSquared) + bufferingConstant;
  if (dielModel == 2) {
    distTerm *= distTerm;
  }
  real energy = prefactor * chargeTerm / (distTerm);
  if (is1_4) {
    energy *= real{0.75};
  }
  return energy;
}

static __device__ __forceinline__ void eleGrad(const auto* pos,
                                               const int   idx1,
                                               const int   idx2,
                                               const real  chargeTerm,
                                               const int   dielModel,
                                               const bool  is1_4,
                                               auto*       grad) {
  constexpr real prefactor         = real{332.0716};
  constexpr real bufferingConstant = real{0.05};

  const real distSquared = distanceSquared(pos, idx1, idx2);
  const real invDistance = NVMOLKIT_RSQRT(distSquared);
  const real distance    = real{1.0} / invDistance;
  const real rBuf        = distance + bufferingConstant;
  real       numerator   = -prefactor * chargeTerm;
  // E_1 = q / (r+b)         -> dE/dr = -q / (r+b)^2
  // E_2 = q / (r+b)^2       -> dE/dr = -2q / (r+b)^3
  real       denominator = rBuf * rBuf;
  if (dielModel == 2) {
    numerator *= 2;
    denominator *= rBuf;
  }

  real dE_dr = numerator / denominator;
  if (is1_4) {
    dE_dr *= real{0.75};
  }

  const real dE_dx = dE_dr * (pos[3 * idx1 + 0] - pos[3 * idx2 + 0]) * invDistance;
  const real dE_dy = dE_dr * (pos[3 * idx1 + 1] - pos[3 * idx2 + 1]) * invDistance;
  const real dE_dz = dE_dr * (pos[3 * idx1 + 2] - pos[3 * idx2 + 2]) * invDistance;

  atomicAdd(&grad[3 * idx1 + 0], dE_dx);
  atomicAdd(&grad[3 * idx1 + 1], dE_dy);
  atomicAdd(&grad[3 * idx1 + 2], dE_dz);

  atomicAdd(&grad[3 * idx2 + 0], -dE_dx);
  atomicAdd(&grad[3 * idx2 + 1], -dE_dy);
  atomicAdd(&grad[3 * idx2 + 2], -dE_dz);
}

static __device__ __forceinline__ real normalizeAngleDeg(real angleDeg) {
  angleDeg = NVMOLKIT_FMOD(angleDeg, real{360.0});
  if (angleDeg < real{-180.0}) {
    angleDeg += real{360.0};
  } else if (angleDeg > real{180.0}) {
    angleDeg -= real{360.0};
  }
  return angleDeg;
}

static __device__ __forceinline__ real distanceConstraintEnergy(const auto* pos,
                                                                const int   idx1,
                                                                const int   idx2,
                                                                const real  minLen,
                                                                const real  maxLen,
                                                                const real  forceConstant) {
  const real distance2Val = distanceSquared(pos, idx1, idx2);
  real       difference   = real{0.0};
  if (distance2Val < minLen * minLen) {
    difference = minLen - NVMOLKIT_SQRT(distance2Val);
  } else if (distance2Val > maxLen * maxLen) {
    difference = NVMOLKIT_SQRT(distance2Val) - maxLen;
  } else {
    return real{0.0};
  }
  return real{0.5} * forceConstant * difference * difference;
}

static __device__ __forceinline__ void distanceConstraintGrad(const auto* pos,
                                                              const int   idx1,
                                                              const int   idx2,
                                                              const real  minLen,
                                                              const real  maxLen,
                                                              const real  forceConstant,
                                                              auto*       grad) {
  const real distance2Val = distanceSquared(pos, idx1, idx2);
  real       preFactor    = real{0.0};
  real       distance     = real{0.0};
  if (distance2Val < minLen * minLen) {
    distance  = NVMOLKIT_SQRT(distance2Val);
    preFactor = distance - minLen;
  } else if (distance2Val > maxLen * maxLen) {
    distance  = NVMOLKIT_SQRT(distance2Val);
    preFactor = distance - maxLen;
  } else {
    return;
  }
  preFactor *= forceConstant;
  preFactor /= NVMOLKIT_FMAX(real{1.0e-8}, distance);
  for (int i = 0; i < 3; ++i) {
    const real dGrad = preFactor * (pos[3 * idx1 + i] - pos[3 * idx2 + i]);
    atomicAdd(&grad[3 * idx1 + i], dGrad);
    atomicAdd(&grad[3 * idx2 + i], -dGrad);
  }
}

static __device__ __forceinline__ real positionConstraintEnergy(const auto* pos,
                                                                const int   idx,
                                                                const real  refX,
                                                                const real  refY,
                                                                const real  refZ,
                                                                const real  maxDispl,
                                                                const real  forceConstant) {
  const real dx       = pos[3 * idx + 0] - refX;
  const real dy       = pos[3 * idx + 1] - refY;
  const real dz       = pos[3 * idx + 2] - refZ;
  const real dist     = NVMOLKIT_SQRT(dx * dx + dy * dy + dz * dz);
  const real distTerm = NVMOLKIT_FMAX(dist - maxDispl, real{0.0});
  return real{0.5} * forceConstant * distTerm * distTerm;
}

static __device__ __forceinline__ void positionConstraintGrad(const auto* pos,
                                                              const int   idx,
                                                              const real  refX,
                                                              const real  refY,
                                                              const real  refZ,
                                                              const real  maxDispl,
                                                              const real  forceConstant,
                                                              auto*       grad) {
  const real dx   = pos[3 * idx + 0] - refX;
  const real dy   = pos[3 * idx + 1] - refY;
  const real dz   = pos[3 * idx + 2] - refZ;
  const real dist = NVMOLKIT_SQRT(dx * dx + dy * dy + dz * dz);
  if (dist <= maxDispl) {
    return;
  }
  const real preFactor = (dist - maxDispl) * forceConstant / NVMOLKIT_FMAX(dist, real{1.0e-8});
  atomicAdd(&grad[3 * idx + 0], preFactor * dx);
  atomicAdd(&grad[3 * idx + 1], preFactor * dy);
  atomicAdd(&grad[3 * idx + 2], preFactor * dz);
}

static __device__ __forceinline__ real computeAngleConstraintTerm(const real angle,
                                                                  const real minAngleDeg,
                                                                  const real maxAngleDeg) {
  real angleTerm = real{0.0};
  if (angle < minAngleDeg) {
    angleTerm = angle - minAngleDeg;
  } else if (angle > maxAngleDeg) {
    angleTerm = angle - maxAngleDeg;
  }
  return angleTerm;
}

static __device__ __forceinline__ real angleConstraintEnergy(const auto* pos,
                                                             const int   idx1,
                                                             const int   idx2,
                                                             const int   idx3,
                                                             const real  minAngleDeg,
                                                             const real  maxAngleDeg,
                                                             const real  forceConstant) {
  const real p1x = pos[3 * idx1 + 0];
  const real p1y = pos[3 * idx1 + 1];
  const real p1z = pos[3 * idx1 + 2];
  const real p2x = pos[3 * idx2 + 0];
  const real p2y = pos[3 * idx2 + 1];
  const real p2z = pos[3 * idx2 + 2];
  const real p3x = pos[3 * idx3 + 0];
  const real p3y = pos[3 * idx3 + 1];
  const real p3z = pos[3 * idx3 + 2];

  const real r1x        = p1x - p2x;
  const real r1y        = p1y - p2y;
  const real r1z        = p1z - p2z;
  const real r2x        = p3x - p2x;
  const real r2y        = p3y - p2y;
  const real r2z        = p3z - p2z;
  const real rLengthSq1 = NVMOLKIT_FMAX(real{1.0e-5}, r1x * r1x + r1y * r1y + r1z * r1z);
  const real rLengthSq2 = NVMOLKIT_FMAX(real{1.0e-5}, r2x * r2x + r2y * r2y + r2z * r2z);
  real       cosTheta   = (r1x * r2x + r1y * r2y + r1z * r2z) / NVMOLKIT_SQRT(rLengthSq1 * rLengthSq2);
  cosTheta              = clamp(cosTheta, real{-1.0}, real{1.0});
  const real angle      = radianToDegree * NVMOLKIT_ACOS(cosTheta);
  const real angleTerm  = computeAngleConstraintTerm(angle, minAngleDeg, maxAngleDeg);
  return forceConstant * angleTerm * angleTerm;
}

static __device__ __forceinline__ void angleConstraintGrad(const auto* pos,
                                                           const int   idx1,
                                                           const int   idx2,
                                                           const int   idx3,
                                                           const real  minAngleDeg,
                                                           const real  maxAngleDeg,
                                                           const real  forceConstant,
                                                           auto*       grad) {
  const real p1x = pos[3 * idx1 + 0];
  const real p1y = pos[3 * idx1 + 1];
  const real p1z = pos[3 * idx1 + 2];
  const real p2x = pos[3 * idx2 + 0];
  const real p2y = pos[3 * idx2 + 1];
  const real p2z = pos[3 * idx2 + 2];
  const real p3x = pos[3 * idx3 + 0];
  const real p3y = pos[3 * idx3 + 1];
  const real p3z = pos[3 * idx3 + 2];

  const real r1x        = p1x - p2x;
  const real r1y        = p1y - p2y;
  const real r1z        = p1z - p2z;
  const real r2x        = p3x - p2x;
  const real r2y        = p3y - p2y;
  const real r2z        = p3z - p2z;
  const real rLengthSq1 = NVMOLKIT_FMAX(real{1.0e-5}, r1x * r1x + r1y * r1y + r1z * r1z);
  const real rLengthSq2 = NVMOLKIT_FMAX(real{1.0e-5}, r2x * r2x + r2y * r2y + r2z * r2z);
  const real invDist1   = rsqrt(rLengthSq1);
  const real invDist2   = rsqrt(rLengthSq2);
  real       cosTheta   = (r1x * r2x + r1y * r2y + r1z * r2z) * invDist1 * invDist2;
  cosTheta              = clamp(cosTheta, real{-1.0}, real{1.0});
  const real angle      = radianToDegree * NVMOLKIT_ACOS(cosTheta);
  const real angleTerm  = computeAngleConstraintTerm(angle, minAngleDeg, maxAngleDeg);
  if (NVMOLKIT_IS_ZERO(angleTerm)) {
    return;
  }

  const real rpX       = r2y * r1z - r2z * r1y;
  const real rpY       = r2z * r1x - r2x * r1z;
  const real rpZ       = r2x * r1y - r2y * r1x;
  const real rpLength  = NVMOLKIT_FMAX(real{1.0e-5}, NVMOLKIT_SQRT(rpX * rpX + rpY * rpY + rpZ * rpZ));
  const real dE_dTheta = real{2.0} * radianToDegree * forceConstant * angleTerm;
  const real prefactor = dE_dTheta / rpLength;
  const real t0        = -prefactor / rLengthSq1;
  const real t1        = prefactor / rLengthSq2;

  const real c0x = r1y * rpZ - r1z * rpY;
  const real c0y = r1z * rpX - r1x * rpZ;
  const real c0z = r1x * rpY - r1y * rpX;
  const real c1x = r2y * rpZ - r2z * rpY;
  const real c1y = r2z * rpX - r2x * rpZ;
  const real c1z = r2x * rpY - r2y * rpX;

  const real dedp0x = c0x * t0;
  const real dedp0y = c0y * t0;
  const real dedp0z = c0z * t0;
  const real dedp2x = c1x * t1;
  const real dedp2y = c1y * t1;
  const real dedp2z = c1z * t1;
  const real dedp1x = -dedp0x - dedp2x;
  const real dedp1y = -dedp0y - dedp2y;
  const real dedp1z = -dedp0z - dedp2z;

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

static __device__ __forceinline__ real computeDihedralConstraintTerm(real       dihedral,
                                                                     const real minDihedralDeg,
                                                                     const real maxDihedralDeg) {
  real dihedralTarget = dihedral;
  if (!(dihedral > minDihedralDeg && dihedral < maxDihedralDeg) &&
      !(dihedral > minDihedralDeg && minDihedralDeg > maxDihedralDeg) &&
      !(dihedral < maxDihedralDeg && minDihedralDeg > maxDihedralDeg)) {
    real dihedralMinTarget = normalizeAngleDeg(dihedral - minDihedralDeg);
    real dihedralMaxTarget = normalizeAngleDeg(dihedral - maxDihedralDeg);
    if (fabs(dihedralMinTarget) < fabs(dihedralMaxTarget)) {
      dihedralTarget = minDihedralDeg;
    } else {
      dihedralTarget = maxDihedralDeg;
    }
  }
  return normalizeAngleDeg(dihedral - dihedralTarget);
}

static __device__ __forceinline__ real computeSignedDihedral(const auto* pos,
                                                             const int   idx1,
                                                             const int   idx2,
                                                             const int   idx3,
                                                             const int   idx4,
                                                             real*       cosPhiOut = nullptr,
                                                             real        r[4][3]   = nullptr,
                                                             real        t[2][3]   = nullptr,
                                                             real        d[2]      = nullptr) {
  real localR[4][3];
  real localT[2][3];
  real localD[2];
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
  d[0] = NVMOLKIT_FMAX(NVMOLKIT_SQRT(t[0][0] * t[0][0] + t[0][1] * t[0][1] + t[0][2] * t[0][2]), real{1.0e-5});
  t[0][0] /= d[0];
  t[0][1] /= d[0];
  t[0][2] /= d[0];
  crossProduct(r[2][0], r[2][1], r[2][2], r[3][0], r[3][1], r[3][2], t[1][0], t[1][1], t[1][2]);
  d[1] = NVMOLKIT_FMAX(NVMOLKIT_SQRT(t[1][0] * t[1][0] + t[1][1] * t[1][1] + t[1][2] * t[1][2]), real{1.0e-5});
  t[1][0] /= d[1];
  t[1][1] /= d[1];
  t[1][2] /= d[1];
  const real cosPhi = clamp(t[0][0] * t[1][0] + t[0][1] * t[1][1] + t[0][2] * t[1][2], real{-1.0}, real{1.0});
  if (cosPhiOut != nullptr) {
    *cosPhiOut = cosPhi;
  }
  real mX, mY, mZ;
  crossProduct(t[0][0], t[0][1], t[0][2], r[1][0], r[1][1], r[1][2], mX, mY, mZ);
  const real mLength = NVMOLKIT_FMAX(NVMOLKIT_SQRT(mX * mX + mY * mY + mZ * mZ), real{1.0e-5});
  return -NVMOLKIT_ATAN2((mX * t[1][0] + mY * t[1][1] + mZ * t[1][2]) / mLength, cosPhi);
}

static __device__ __forceinline__ real torsionConstraintEnergy(const auto* pos,
                                                               const int   idx1,
                                                               const int   idx2,
                                                               const int   idx3,
                                                               const int   idx4,
                                                               const real  minDihedralDeg,
                                                               const real  maxDihedralDeg,
                                                               const real  forceConstant) {
  const real dihedral     = radianToDegree * computeSignedDihedral(pos, idx1, idx2, idx3, idx4);
  const real dihedralTerm = computeDihedralConstraintTerm(dihedral, minDihedralDeg, maxDihedralDeg);
  return forceConstant * dihedralTerm * dihedralTerm;
}

static __device__ __forceinline__ void torsionConstraintGrad(const auto* pos,
                                                             const int   idx1,
                                                             const int   idx2,
                                                             const int   idx3,
                                                             const int   idx4,
                                                             const real  minDihedralDeg,
                                                             const real  maxDihedralDeg,
                                                             const real  forceConstant,
                                                             auto*       grad) {
  real       r[4][3];
  real       t[2][3];
  real       d[2];
  const real dihedral     = radianToDegree * computeSignedDihedral(pos, idx1, idx2, idx3, idx4, nullptr, r, t, d);
  const real dihedralTerm = computeDihedralConstraintTerm(dihedral, minDihedralDeg, maxDihedralDeg);
  if (NVMOLKIT_IS_ZERO(dihedralTerm)) {
    return;
  }
  const real dE_dPhi = real{2.0} * radianToDegree * forceConstant * dihedralTerm;

  const real d23       = NVMOLKIT_SQRT(distanceSquared(pos, idx2, idx3));
  const real prefactor = dE_dPhi / NVMOLKIT_FMAX(d23, real{1.0e-8});

  real tt0[3], tt1[3];
  crossProduct(r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], tt0[0], tt0[1], tt0[2]);
  crossProduct(r[2][0], r[2][1], r[2][2], r[3][0], r[3][1], r[3][2], tt1[0], tt1[1], tt1[2]);
  const real tt0LenSq = NVMOLKIT_FMAX(tt0[0] * tt0[0] + tt0[1] * tt0[1] + tt0[2] * tt0[2], real{1.0e-8});
  const real tt1LenSq = NVMOLKIT_FMAX(tt1[0] * tt1[0] + tt1[1] * tt1[1] + tt1[2] * tt1[2], real{1.0e-8});

  real tmp0[3], tmp1[3];
  crossProduct(tt0[0], tt0[1], tt0[2], r[2][0], r[2][1], r[2][2], tmp0[0], tmp0[1], tmp0[2]);
  crossProduct(tt1[0], tt1[1], tt1[2], r[1][0], r[1][1], r[1][2], tmp1[0], tmp1[1], tmp1[2]);
  const real dedt0[3] = {tmp0[0] / tt0LenSq * prefactor,
                         tmp0[1] / tt0LenSq * prefactor,
                         tmp0[2] / tt0LenSq * prefactor};
  const real dedt1[3] = {tmp1[0] / tt1LenSq * prefactor,
                         tmp1[1] / tt1LenSq * prefactor,
                         tmp1[2] / tt1LenSq * prefactor};

  const real r31[3] = {static_cast<real>(pos[3 * idx3 + 0]) - static_cast<real>(pos[3 * idx1 + 0]),
                       static_cast<real>(pos[3 * idx3 + 1]) - static_cast<real>(pos[3 * idx1 + 1]),
                       static_cast<real>(pos[3 * idx3 + 2]) - static_cast<real>(pos[3 * idx1 + 2])};
  const real r42[3] = {static_cast<real>(pos[3 * idx4 + 0]) - static_cast<real>(pos[3 * idx2 + 0]),
                       static_cast<real>(pos[3 * idx4 + 1]) - static_cast<real>(pos[3 * idx2 + 1]),
                       static_cast<real>(pos[3 * idx4 + 2]) - static_cast<real>(pos[3 * idx2 + 2])};

  real dedp0[3], dedp1[3], dedp2[3], dedp3[3];
  crossProduct(r[2][0], r[2][1], r[2][2], dedt0[0], dedt0[1], dedt0[2], dedp0[0], dedp0[1], dedp0[2]);

  real r31Cross[3], r3Cross[3];
  crossProduct(r31[0], r31[1], r31[2], dedt0[0], dedt0[1], dedt0[2], r31Cross[0], r31Cross[1], r31Cross[2]);
  crossProduct(r[3][0], r[3][1], r[3][2], dedt1[0], dedt1[1], dedt1[2], r3Cross[0], r3Cross[1], r3Cross[2]);
  dedp1[0] = r31Cross[0] - r3Cross[0];
  dedp1[1] = r31Cross[1] - r3Cross[1];
  dedp1[2] = r31Cross[2] - r3Cross[2];

  real r0Cross[3], r42Cross[3];
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

template <int stride, bool HasConstraints, typename Terms>
static __device__ __inline__ real molEnergy(const Terms&                   terms,
                                            const BatchedIndicesDevicePtr& systemIndices,
                                            const auto*                    molCoords,
                                            const int                      molIdx,
                                            const int                      tid) {
  const int atomStart = systemIndices.atomStarts[molIdx];

  real energy = real{0.0};

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

template <int stride, bool HasConstraints, typename Terms>
static __device__ __inline__ void molGrad(const Terms&                   terms,
                                          const BatchedIndicesDevicePtr& systemIndices,
                                          const auto*                    molCoords,
                                          auto*                          grad,
                                          const int                      molIdx,
                                          const int                      tid) {
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
