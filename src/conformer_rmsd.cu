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

#include <climits>
#include <cmath>
#include <cub/cub.cuh>
#include <cuda/std/span>

#include "src/conformer/device_conformer_pruning.h"
#include "src/conformer_rmsd.h"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit {

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

/// Compute eigenvalues of a 3x3 symmetric matrix using Cardano's analytical formula.
/// Input: upper triangle {a00, a01, a02, a11, a12, a22}.
/// Output: three eigenvalues in descending order.
__device__ __forceinline__ void symmetricEigenvalues3x3(const double a00,
                                                        const double a01,
                                                        const double a02,
                                                        const double a11,
                                                        const double a12,
                                                        const double a22,
                                                        double&      e0,
                                                        double&      e1,
                                                        double&      e2) {
  // Characteristic polynomial:  λ³ - p λ² + q λ - r = 0
  const double p       = a00 + a11 + a22;  // trace
  // Pre-compute pairwise products reused in both q and r.
  const double a00_a11 = a00 * a11;
  const double a00_a22 = a00 * a22;
  const double a11_a22 = a11 * a22;
  const double q       = a00_a11 + a00_a22 + a11_a22 - a01 * a01 - a02 * a02 - a12 * a12;
  const double r       = a00_a11 * a22 + 2.0 * a01 * a02 * a12 - a00 * a12 * a12 - a11 * a02 * a02 - a22 * a01 * a01;

  // Shift to depressed cubic:  t³ + pt' + q' = 0  where λ = t + p/3
  const double p3        = p / 3.0;
  const double pp        = (p * p - 3.0 * q) / 9.0;  // -p'/3
  const double qq        = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 54.0;
  // Three real roots (guaranteed for real symmetric matrices).  Near-degenerate
  // inputs are handled by the fmax guards on sqrtPP and the acos argument below.
  const double sqrtPP    = sqrt(fmax(pp, 0.0));
  const double theta     = acos(fmin(fmax(qq / fmax(sqrtPP * sqrtPP * sqrtPP, 1e-30), -1.0), 1.0)) / 3.0;
  const double twoSqrtPP = 2.0 * sqrtPP;
  e0                     = twoSqrtPP * cos(theta) + p3;
  e1                     = twoSqrtPP * cos(theta - 2.0 * M_PI / 3.0) + p3;
  e2                     = twoSqrtPP * cos(theta - 4.0 * M_PI / 3.0) + p3;

  // Sort descending
  if (e1 > e0) {
    double t = e0;
    e0       = e1;
    e1       = t;
  }
  if (e2 > e0) {
    double t = e0;
    e0       = e2;
    e2       = t;
  }
  if (e2 > e1) {
    double t = e1;
    e1       = e2;
    e2       = t;
  }
}

/// Compute determinant of a 3x3 matrix stored as scalar elements.
__device__ __forceinline__ double det3x3(const double h00,
                                         const double h01,
                                         const double h02,
                                         const double h10,
                                         const double h11,
                                         const double h12,
                                         const double h20,
                                         const double h21,
                                         const double h22) {
  return h00 * (h11 * h22 - h12 * h21) - h01 * (h10 * h22 - h12 * h20) + h02 * (h10 * h21 - h11 * h20);
}

// This can be extended later to support arbitrary dimensions, but currently supports only 3D due to its specialized
// linear algebra functions.
__device__ __forceinline__ double alignedRmsd(const double Sp, const double Sq, const double* H, const double invN) {
  // Finish the Kabsch calculation from the centered sums and covariance matrix.
  // G = H^T H  (3x3 symmetric positive semi-definite)
  const double g00 = H[0] * H[0] + H[3] * H[3] + H[6] * H[6];
  const double g01 = H[0] * H[1] + H[3] * H[4] + H[6] * H[7];
  const double g02 = H[0] * H[2] + H[3] * H[5] + H[6] * H[8];
  const double g11 = H[1] * H[1] + H[4] * H[4] + H[7] * H[7];
  const double g12 = H[1] * H[2] + H[4] * H[5] + H[7] * H[8];
  const double g22 = H[2] * H[2] + H[5] * H[5] + H[8] * H[8];

  // Eigenvalues of G = squared singular values of H
  double ev0, ev1, ev2;
  symmetricEigenvalues3x3(g00, g01, g02, g11, g12, g22, ev0, ev1, ev2);

  // Singular values (clamp negatives from numerical noise)
  const double s0 = sqrt(fmax(ev0, 0.0));
  const double s1 = sqrt(fmax(ev1, 0.0));
  double       s2 = sqrt(fmax(ev2, 0.0));

  // Handle reflection: if det(H) < 0, negate the smallest singular value
  if (det3x3(H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7], H[8]) < 0.0) {
    s2 = -s2;
  }

  // RMSD^2 = (Sp + Sq - 2*(s0 + s1 + s2)) / N
  return sqrt(fmax((Sp + Sq - 2.0 * (s0 + s1 + s2)) * invN, 0.0));
}

// ---------------------------------------------------------------------------
// Rigid-alignment helpers
//
// Both alignment modes need centroids and H = Pc^T Qc. Pairwise RMSD consumes
// H's singular values; first-conformer alignment consumes its rotation.
// ---------------------------------------------------------------------------

constexpr int kRmsdBlockSize = 128;
constexpr int kRmsdWarps     = kRmsdBlockSize / 32;  // 4 warps per block

using RmsdWarpReduce = cub::WarpReduce<double>;

struct RigidAlignmentMoments {
  double movingCenterX;
  double movingCenterY;
  double movingCenterZ;
  double fixedCenterX;
  double fixedCenterY;
  double fixedCenterZ;
  double movingSquaredNorm;
  double fixedSquaredNorm;
  double h00;
  double h01;
  double h02;
  double h10;
  double h11;
  double h12;
  double h20;
  double h21;
  double h22;
};

/// Accumulate the centroids and cross-covariance shared by both rigid-alignment
/// consumers. Pairwise RMSD additionally requests the two centered norms.
template <bool ComputeSquaredNorms>
__device__ __forceinline__ void accumulateRigidAlignmentMoments(const double* __restrict__ movingCoords,
                                                                const double* __restrict__ fixedCoords,
                                                                const int                             numAtoms,
                                                                typename RmsdWarpReduce::TempStorage* warpReduceTemp,
                                                                double*                               warpBuf,
                                                                RigidAlignmentMoments*                moments) {
  constexpr int kNumFields = ComputeSquaredNorms ? 11 : 9;
  const int     tid        = threadIdx.x;
  const int     warpId     = tid / 32;
  const int     laneId     = tid % 32;

  double sumMx = 0.0, sumMy = 0.0, sumMz = 0.0;
  double sumFx = 0.0, sumFy = 0.0, sumFz = 0.0;
  for (int atom = tid; atom < numAtoms; atom += kRmsdBlockSize) {
    sumMx += movingCoords[atom * 3 + 0];
    sumMy += movingCoords[atom * 3 + 1];
    sumMz += movingCoords[atom * 3 + 2];
    sumFx += fixedCoords[atom * 3 + 0];
    sumFy += fixedCoords[atom * 3 + 1];
    sumFz += fixedCoords[atom * 3 + 2];
  }
  sumMx = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(sumMx);
  sumMy = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(sumMy);
  sumMz = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(sumMz);
  sumFx = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(sumFx);
  sumFy = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(sumFy);
  sumFz = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(sumFz);
  if (laneId == 0) {
    warpBuf[warpId * kNumFields + 0] = sumMx;
    warpBuf[warpId * kNumFields + 1] = sumMy;
    warpBuf[warpId * kNumFields + 2] = sumMz;
    warpBuf[warpId * kNumFields + 3] = sumFx;
    warpBuf[warpId * kNumFields + 4] = sumFy;
    warpBuf[warpId * kNumFields + 5] = sumFz;
  }
  __syncthreads();

  if (tid == 0) {
    double mx = 0.0, my = 0.0, mz = 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int warp = 0; warp < kRmsdWarps; ++warp) {
      mx += warpBuf[warp * kNumFields + 0];
      my += warpBuf[warp * kNumFields + 1];
      mz += warpBuf[warp * kNumFields + 2];
      fx += warpBuf[warp * kNumFields + 3];
      fy += warpBuf[warp * kNumFields + 4];
      fz += warpBuf[warp * kNumFields + 5];
    }
    const double invN      = 1.0 / static_cast<double>(numAtoms);
    moments->movingCenterX = mx * invN;
    moments->movingCenterY = my * invN;
    moments->movingCenterZ = mz * invN;
    moments->fixedCenterX  = fx * invN;
    moments->fixedCenterY  = fy * invN;
    moments->fixedCenterZ  = fz * invN;
  }
  __syncthreads();

  const double cmx               = moments->movingCenterX;
  const double cmy               = moments->movingCenterY;
  const double cmz               = moments->movingCenterZ;
  const double cfx               = moments->fixedCenterX;
  const double cfy               = moments->fixedCenterY;
  const double cfz               = moments->fixedCenterZ;
  double       movingSquaredNorm = 0.0, fixedSquaredNorm = 0.0;
  double       h00 = 0.0, h01 = 0.0, h02 = 0.0;
  double       h10 = 0.0, h11 = 0.0, h12 = 0.0;
  double       h20 = 0.0, h21 = 0.0, h22 = 0.0;
  for (int atom = tid; atom < numAtoms; atom += kRmsdBlockSize) {
    const double mx = movingCoords[atom * 3 + 0] - cmx;
    const double my = movingCoords[atom * 3 + 1] - cmy;
    const double mz = movingCoords[atom * 3 + 2] - cmz;
    const double fx = fixedCoords[atom * 3 + 0] - cfx;
    const double fy = fixedCoords[atom * 3 + 1] - cfy;
    const double fz = fixedCoords[atom * 3 + 2] - cfz;
    if constexpr (ComputeSquaredNorms) {
      movingSquaredNorm += mx * mx + my * my + mz * mz;
      fixedSquaredNorm += fx * fx + fy * fy + fz * fz;
    }
    h00 += mx * fx;
    h01 += mx * fy;
    h02 += mx * fz;
    h10 += my * fx;
    h11 += my * fy;
    h12 += my * fz;
    h20 += mz * fx;
    h21 += mz * fy;
    h22 += mz * fz;
  }
  if constexpr (ComputeSquaredNorms) {
    movingSquaredNorm = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(movingSquaredNorm);
    fixedSquaredNorm  = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(fixedSquaredNorm);
  }
  h00 = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(h00);
  h01 = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(h01);
  h02 = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(h02);
  h10 = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(h10);
  h11 = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(h11);
  h12 = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(h12);
  h20 = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(h20);
  h21 = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(h21);
  h22 = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(h22);
  if (laneId == 0) {
    constexpr int kCovarianceOffset = ComputeSquaredNorms ? 2 : 0;
    if constexpr (ComputeSquaredNorms) {
      warpBuf[warpId * kNumFields + 0] = movingSquaredNorm;
      warpBuf[warpId * kNumFields + 1] = fixedSquaredNorm;
    }
    warpBuf[warpId * kNumFields + kCovarianceOffset + 0] = h00;
    warpBuf[warpId * kNumFields + kCovarianceOffset + 1] = h01;
    warpBuf[warpId * kNumFields + kCovarianceOffset + 2] = h02;
    warpBuf[warpId * kNumFields + kCovarianceOffset + 3] = h10;
    warpBuf[warpId * kNumFields + kCovarianceOffset + 4] = h11;
    warpBuf[warpId * kNumFields + kCovarianceOffset + 5] = h12;
    warpBuf[warpId * kNumFields + kCovarianceOffset + 6] = h20;
    warpBuf[warpId * kNumFields + kCovarianceOffset + 7] = h21;
    warpBuf[warpId * kNumFields + kCovarianceOffset + 8] = h22;
  }
  __syncthreads();

  if (tid == 0) {
    constexpr int kCovarianceOffset = ComputeSquaredNorms ? 2 : 0;
    moments->movingSquaredNorm      = 0.0;
    moments->fixedSquaredNorm       = 0.0;
    moments->h00 = moments->h01 = moments->h02 = 0.0;
    moments->h10 = moments->h11 = moments->h12 = 0.0;
    moments->h20 = moments->h21 = moments->h22 = 0.0;
    for (int warp = 0; warp < kRmsdWarps; ++warp) {
      if constexpr (ComputeSquaredNorms) {
        moments->movingSquaredNorm += warpBuf[warp * kNumFields + 0];
        moments->fixedSquaredNorm += warpBuf[warp * kNumFields + 1];
      }
      moments->h00 += warpBuf[warp * kNumFields + kCovarianceOffset + 0];
      moments->h01 += warpBuf[warp * kNumFields + kCovarianceOffset + 1];
      moments->h02 += warpBuf[warp * kNumFields + kCovarianceOffset + 2];
      moments->h10 += warpBuf[warp * kNumFields + kCovarianceOffset + 3];
      moments->h11 += warpBuf[warp * kNumFields + kCovarianceOffset + 4];
      moments->h12 += warpBuf[warp * kNumFields + kCovarianceOffset + 5];
      moments->h20 += warpBuf[warp * kNumFields + kCovarianceOffset + 6];
      moments->h21 += warpBuf[warp * kNumFields + kCovarianceOffset + 7];
      moments->h22 += warpBuf[warp * kNumFields + kCovarianceOffset + 8];
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void jacobiRotate4x4(double* matrix, double* eigenvectors, const int p, const int q) {
  const double apq = matrix[p * 4 + q];
  if (apq == 0.0)
    return;

  const double app = matrix[p * 4 + p];
  const double aqq = matrix[q * 4 + q];
  const double tau = (aqq - app) / (2.0 * apq);
  const double t   = copysign(1.0 / (fabs(tau) + sqrt(1.0 + tau * tau)), tau);
  const double c   = 1.0 / sqrt(1.0 + t * t);
  const double s   = t * c;

  for (int k = 0; k < 4; ++k) {
    if (k == p || k == q)
      continue;
    const double akp  = matrix[k * 4 + p];
    const double akq  = matrix[k * 4 + q];
    matrix[k * 4 + p] = c * akp - s * akq;
    matrix[p * 4 + k] = matrix[k * 4 + p];
    matrix[k * 4 + q] = s * akp + c * akq;
    matrix[q * 4 + k] = matrix[k * 4 + q];
  }
  matrix[p * 4 + p] = app - t * apq;
  matrix[q * 4 + q] = aqq + t * apq;
  matrix[p * 4 + q] = 0.0;
  matrix[q * 4 + p] = 0.0;

  for (int k = 0; k < 4; ++k) {
    const double vkp        = eigenvectors[k * 4 + p];
    const double vkq        = eigenvectors[k * 4 + q];
    eigenvectors[k * 4 + p] = c * vkp - s * vkq;
    eigenvectors[k * 4 + q] = s * vkp + c * vkq;
  }
}

__device__ __forceinline__ void computeOptimalRotation(const RigidAlignmentMoments& moments,
                                                       double*                      quaternionMatrix,
                                                       double*                      eigenvectors,
                                                       double*                      rotation) {
  quaternionMatrix[0]  = moments.h00 + moments.h11 + moments.h22;
  quaternionMatrix[1]  = moments.h12 - moments.h21;
  quaternionMatrix[2]  = moments.h20 - moments.h02;
  quaternionMatrix[3]  = moments.h01 - moments.h10;
  quaternionMatrix[4]  = quaternionMatrix[1];
  quaternionMatrix[5]  = moments.h00 - moments.h11 - moments.h22;
  quaternionMatrix[6]  = moments.h01 + moments.h10;
  quaternionMatrix[7]  = moments.h20 + moments.h02;
  quaternionMatrix[8]  = quaternionMatrix[2];
  quaternionMatrix[9]  = quaternionMatrix[6];
  quaternionMatrix[10] = -moments.h00 + moments.h11 - moments.h22;
  quaternionMatrix[11] = moments.h12 + moments.h21;
  quaternionMatrix[12] = quaternionMatrix[3];
  quaternionMatrix[13] = quaternionMatrix[7];
  quaternionMatrix[14] = quaternionMatrix[11];
  quaternionMatrix[15] = -moments.h00 - moments.h11 + moments.h22;

  for (int row = 0; row < 4; ++row)
    for (int col = 0; col < 4; ++col)
      eigenvectors[row * 4 + col] = row == col ? 1.0 : 0.0;
  // A fixed 4x4 cyclic Jacobi solve converges to double precision in far
  // fewer than 16 sweeps; the fixed bound keeps execution deterministic.
  constexpr int kJacobiSweeps = 16;
  for (int sweep = 0; sweep < kJacobiSweeps; ++sweep) {
    for (int p = 0; p < 3; ++p)
      for (int q = p + 1; q < 4; ++q)
        jacobiRotate4x4(quaternionMatrix, eigenvectors, p, q);
  }

  int largest = 0;
  for (int i = 1; i < 4; ++i)
    if (quaternionMatrix[i * 4 + i] > quaternionMatrix[largest * 4 + largest])
      largest = i;
  double       qw      = eigenvectors[largest];
  double       qx      = eigenvectors[4 + largest];
  double       qy      = eigenvectors[8 + largest];
  double       qz      = eigenvectors[12 + largest];
  const double invNorm = 1.0 / sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  qw *= invNorm;
  qx *= invNorm;
  qy *= invNorm;
  qz *= invNorm;

  rotation[0] = qw * qw + qx * qx - qy * qy - qz * qz;
  rotation[1] = 2.0 * (qx * qy - qw * qz);
  rotation[2] = 2.0 * (qx * qz + qw * qy);
  rotation[3] = 2.0 * (qx * qy + qw * qz);
  rotation[4] = qw * qw - qx * qx + qy * qy - qz * qz;
  rotation[5] = 2.0 * (qy * qz - qw * qx);
  rotation[6] = 2.0 * (qx * qz - qw * qy);
  rotation[7] = 2.0 * (qy * qz + qw * qx);
  rotation[8] = qw * qw - qx * qx - qy * qy + qz * qz;
}

__device__ __forceinline__ void alignConformerToFirst(const double* __restrict__ movingCoords,
                                                      const double* __restrict__ firstCoords,
                                                      double* __restrict__ alignedCoords,
                                                      const int numAtoms) {
  const int tid = threadIdx.x;

  if (movingCoords == firstCoords) {
    for (int i = tid; i < numAtoms * 3; i += kRmsdBlockSize)
      alignedCoords[i] = movingCoords[i];
    return;
  }

  __shared__ typename RmsdWarpReduce::TempStorage warpReduceTemp[kRmsdWarps];
  __shared__ double                               warpBuf[kRmsdWarps * 9];
  __shared__ RigidAlignmentMoments                moments;
  __shared__ double                               quaternionMatrix[16];
  __shared__ double                               eigenvectors[16];
  __shared__ double                               rotation[9];

  accumulateRigidAlignmentMoments<false>(movingCoords, firstCoords, numAtoms, warpReduceTemp, warpBuf, &moments);
  if (tid == 0)
    computeOptimalRotation(moments, quaternionMatrix, eigenvectors, rotation);
  __syncthreads();

  for (int atom = tid; atom < numAtoms; atom += kRmsdBlockSize) {
    const double x              = movingCoords[atom * 3 + 0] - moments.movingCenterX;
    const double y              = movingCoords[atom * 3 + 1] - moments.movingCenterY;
    const double z              = movingCoords[atom * 3 + 2] - moments.movingCenterZ;
    alignedCoords[atom * 3 + 0] = rotation[0] * x + rotation[1] * y + rotation[2] * z + moments.fixedCenterX;
    alignedCoords[atom * 3 + 1] = rotation[3] * x + rotation[4] * y + rotation[5] * z + moments.fixedCenterY;
    alignedCoords[atom * 3 + 2] = rotation[6] * x + rotation[7] * y + rotation[8] * z + moments.fixedCenterZ;
  }
}

__device__ __forceinline__ double computeOptimalRmsd(const RigidAlignmentMoments& moments, const double invN) {
  // G = H^T H (3x3 symmetric positive semi-definite).
  const double g00 = moments.h00 * moments.h00 + moments.h10 * moments.h10 + moments.h20 * moments.h20;
  const double g01 = moments.h00 * moments.h01 + moments.h10 * moments.h11 + moments.h20 * moments.h21;
  const double g02 = moments.h00 * moments.h02 + moments.h10 * moments.h12 + moments.h20 * moments.h22;
  const double g11 = moments.h01 * moments.h01 + moments.h11 * moments.h11 + moments.h21 * moments.h21;
  const double g12 = moments.h01 * moments.h02 + moments.h11 * moments.h12 + moments.h21 * moments.h22;
  const double g22 = moments.h02 * moments.h02 + moments.h12 * moments.h12 + moments.h22 * moments.h22;

  double ev0, ev1, ev2;
  symmetricEigenvalues3x3(g00, g01, g02, g11, g12, g22, ev0, ev1, ev2);
  const double s0 = sqrt(fmax(ev0, 0.0));
  const double s1 = sqrt(fmax(ev1, 0.0));
  double       s2 = sqrt(fmax(ev2, 0.0));
  if (det3x3(moments.h00,
             moments.h01,
             moments.h02,
             moments.h10,
             moments.h11,
             moments.h12,
             moments.h20,
             moments.h21,
             moments.h22) < 0.0)
    s2 = -s2;

  const double rmsdSq = fmax((moments.movingSquaredNorm + moments.fixedSquaredNorm - 2.0 * (s0 + s1 + s2)) * invN, 0.0);
  return sqrt(rmsdSq);
}

__device__ __forceinline__ void computePairRmsd(const double* __restrict__ coordI,
                                                const double* __restrict__ coordJ,
                                                const int  numAtoms,
                                                const bool prealigned,
                                                double*    outRmsd) {
  const int tid    = threadIdx.x;
  const int warpId = tid / 32;
  const int laneId = tid % 32;

  __shared__ typename RmsdWarpReduce::TempStorage warpReduceTemp[kRmsdWarps];
  __shared__ double                               warpBuf[kRmsdWarps * 11];
  __shared__ RigidAlignmentMoments                moments;

  if (prealigned) {
    // ---- Simple RMSD without alignment (no centering, matches RDKit behavior) ----
    double sumSqDiff = 0.0;
    for (int a = tid; a < numAtoms; a += kRmsdBlockSize) {
      const double dx = coordI[a * 3 + 0] - coordJ[a * 3 + 0];
      const double dy = coordI[a * 3 + 1] - coordJ[a * 3 + 1];
      const double dz = coordI[a * 3 + 2] - coordJ[a * 3 + 2];
      sumSqDiff += dx * dx + dy * dy + dz * dz;
    }
    sumSqDiff = RmsdWarpReduce(warpReduceTemp[warpId]).Sum(sumSqDiff);
    if (laneId == 0)
      warpBuf[warpId * 11] = sumSqDiff;
    __syncthreads();
    if (tid == 0) {
      double total = 0.0;
      for (int w = 0; w < kRmsdWarps; ++w)
        total += warpBuf[w * 11];
      *outRmsd = sqrt(total / static_cast<double>(numAtoms));
    }
    return;
  }

  const double invN = 1.0 / static_cast<double>(numAtoms);
  accumulateRigidAlignmentMoments<true>(coordI, coordJ, numAtoms, warpReduceTemp, warpBuf, &moments);
  if (tid == 0)
    *outRmsd = computeOptimalRmsd(moments, invN);
}

// ---------------------------------------------------------------------------
// Alignment kernels: one thread-block per conformer
// ---------------------------------------------------------------------------

__global__ void alignConformersToFirstKernel(const double* __restrict__ coords,
                                             double* __restrict__ alignedCoords,
                                             const int numConformers,
                                             const int numAtoms) {
  const int conformer = blockIdx.x;
  if (conformer >= numConformers)
    return;

  const int     stride       = numAtoms * 3;
  const double* movingCoords = coords + conformer * stride;
  double*       outputCoords = alignedCoords + conformer * stride;
  alignConformerToFirst(movingCoords, coords, outputCoords, numAtoms);
}

__global__ void alignConformersToFirstBatchKernel(const double* __restrict__ coords,
                                                  double* __restrict__ alignedCoords,
                                                  const int* __restrict__ conformerOffsets,
                                                  const size_t* __restrict__ coordOffsets,
                                                  const int* __restrict__ numConfsPerMol,
                                                  const int* __restrict__ numAtomsPerMol,
                                                  const int numMols) {
  if (numMols <= 0)
    return;

  const int globalConformer = blockIdx.x;
  int       lo = 0, hi = numMols - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (globalConformer >= conformerOffsets[mid])
      lo = mid;
    else
      hi = mid - 1;
  }
  const int mol            = lo;
  const int localConformer = globalConformer - conformerOffsets[mol];
  if (localConformer < 0 || localConformer >= numConfsPerMol[mol])
    return;

  const int     numAtoms     = numAtomsPerMol[mol];
  const int     stride       = numAtoms * 3;
  const size_t  coordOffset  = coordOffsets[mol];
  const double* firstCoords  = coords + static_cast<ptrdiff_t>(coordOffset);
  const double* movingCoords = firstCoords + localConformer * stride;
  double*       outputCoords = alignedCoords + static_cast<ptrdiff_t>(coordOffset) + localConformer * stride;
  alignConformerToFirst(movingCoords, firstCoords, outputCoords, numAtoms);
}

// ---------------------------------------------------------------------------
// RMSD kernel: one thread-block per conformer pair
// ---------------------------------------------------------------------------

__global__ void conformerRmsdKernel(const double* __restrict__ coords,
                                    double* __restrict__ rmsdOut,
                                    const int  numConformers,
                                    const int  numAtoms,
                                    const bool prealigned) {
  // Map blockIdx to pair (ci, cj) with ci > cj using lower-triangle indexing.
  const int pairIdx = blockIdx.x;
  // Inverse of pairIdx = ci*(ci-1)/2 + cj:  ci = floor((1 + sqrt(1 + 8*pairIdx)) / 2)
  // Precision note: double has 53-bit significand; pairIdx is bounded by INT_MAX (~2^31),
  // which fits exactly in double, so the sqrt cannot round to a wrong integer here.
  const int ci      = static_cast<int>(floor((1.0 + sqrt(1.0 + 8.0 * static_cast<double>(pairIdx))) / 2.0));
  const int cj      = pairIdx - ci * (ci - 1) / 2;

  // Safety check (floating point edge cases in the inverse formula)
  if (ci >= numConformers || cj < 0 || cj >= ci)
    return;

  const int     stride = numAtoms * 3;
  const double* coordI = coords + ci * stride;
  const double* coordJ = coords + cj * stride;

  computePairRmsd(coordI, coordJ, numAtoms, prealigned, &rmsdOut[pairIdx]);
}

// ---------------------------------------------------------------------------
// Batch kernel: one thread-block per conformer pair across all molecules.
//
// pairOffsets[m]..pairOffsets[m+1] is the global block range for molecule m.
// coordOffsets[m] is the start index (in doubles) of molecule m in coords[].
// rmsdOutputs[m] is the pre-allocated device output buffer for molecule m.
// ---------------------------------------------------------------------------

__global__ void conformerRmsdBatchKernel(const double* __restrict__ coords,
                                         double** __restrict__ rmsdOutputs,
                                         const int* __restrict__ pairOffsets,
                                         const size_t* __restrict__ coordOffsets,
                                         const int* __restrict__ numConfsPerMol,
                                         const int* __restrict__ numAtomsPerMol,
                                         const int  numMols,
                                         const bool prealigned) {
  if (numMols <= 0)
    return;

  const int globalPairIdx = blockIdx.x;

  // Find which molecule this block belongs to via binary search on pairOffsets.
  int lo = 0, hi = numMols - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (globalPairIdx >= pairOffsets[mid])
      lo = mid;
    else
      hi = mid - 1;
  }
  const int mol = lo;

  const int     localPairIdx = globalPairIdx - pairOffsets[mol];
  const int     numConfs     = numConfsPerMol[mol];
  const int     numAtoms     = numAtomsPerMol[mol];
  const double* molCoords    = coords + static_cast<ptrdiff_t>(coordOffsets[mol]);
  double*       molRmsd      = rmsdOutputs[mol];

  // Map localPairIdx to (ci, cj) with ci > cj.
  // Precision note: double has 53-bit significand; localPairIdx is bounded by INT_MAX (~2^31),
  // which fits exactly in double, so the sqrt cannot round to a wrong integer here.
  const int ci = static_cast<int>(floor((1.0 + sqrt(1.0 + 8.0 * static_cast<double>(localPairIdx))) / 2.0));
  const int cj = localPairIdx - ci * (ci - 1) / 2;

  if (ci >= numConfs || cj < 0 || cj >= ci)
    return;

  const int     stride = numAtoms * 3;
  const double* coordI = molCoords + ci * stride;
  const double* coordJ = molCoords + cj * stride;

  computePairRmsd(coordI, coordJ, numAtoms, prealigned, &molRmsd[localPairIdx]);
}

namespace {

using detail::ConformerPruningMolInfo;
using detail::kNumCoordinateDimensions;

constexpr unsigned int kFullWarpMask    = 0xffffffffU;
constexpr int          kGreedyBlockSize = 64;

__device__ __forceinline__ double warpSum(double value) {
  __shared__ cub::WarpReduce<double>::TempStorage tempStorage[kGreedyBlockSize / 32];
  const int                                       warp = threadIdx.x / warpSize;
  const double                                    sum  = cub::WarpReduce<double>(tempStorage[warp]).Sum(value);

  // CUB returns the reduced value in lane 0, broadcast it to the rest of the warp
  return __shfl_sync(kFullWarpMask, sum, 0);
}

__device__ __forceinline__ double computePairRmsdWarp(const double* __restrict__ coordI,
                                                      const double* __restrict__ coordJ,
                                                      const int* __restrict__ atomMapI,
                                                      const int* __restrict__ atomMapJ,
                                                      const int numAtoms) {
  const int lane = threadIdx.x % warpSize;

  // Each lane handles every warp-stride mapped atom. Warp-wide reductions combine
  // the partial values to find both conformer centroids.
  double sumIx = 0.0, sumIy = 0.0, sumIz = 0.0;
  double sumJx = 0.0, sumJy = 0.0, sumJz = 0.0;
  for (int atom = lane; atom < numAtoms; atom += warpSize) {
    const int offsetI = atomMapI[atom] * kNumCoordinateDimensions;
    const int offsetJ = atomMapJ[atom] * kNumCoordinateDimensions;
    sumIx += coordI[offsetI + 0];
    sumIy += coordI[offsetI + 1];
    sumIz += coordI[offsetI + 2];
    sumJx += coordJ[offsetJ + 0];
    sumJy += coordJ[offsetJ + 1];
    sumJz += coordJ[offsetJ + 2];
  }

  const double invN = 1.0 / static_cast<double>(numAtoms);
  const double cIx  = warpSum(sumIx) * invN;
  const double cIy  = warpSum(sumIy) * invN;
  const double cIz  = warpSum(sumIz) * invN;
  const double cJx  = warpSum(sumJx) * invN;
  const double cJy  = warpSum(sumJy) * invN;
  const double cJz  = warpSum(sumJz) * invN;

  // Center the mapped coordinates and build the values needed by Kabsch.
  double localSp = 0.0, localSq = 0.0;
  double localH[9] = {0.0};
  for (int atom = lane; atom < numAtoms; atom += warpSize) {
    const int    offsetI = atomMapI[atom] * kNumCoordinateDimensions;
    const int    offsetJ = atomMapJ[atom] * kNumCoordinateDimensions;
    const double px      = coordI[offsetI + 0] - cIx;
    const double py      = coordI[offsetI + 1] - cIy;
    const double pz      = coordI[offsetI + 2] - cIz;
    const double qx      = coordJ[offsetJ + 0] - cJx;
    const double qy      = coordJ[offsetJ + 1] - cJy;
    const double qz      = coordJ[offsetJ + 2] - cJz;

    localSp += px * px + py * py + pz * pz;
    localSq += qx * qx + qy * qy + qz * qz;
    localH[0] += px * qx;
    localH[1] += px * qy;
    localH[2] += px * qz;
    localH[3] += py * qx;
    localH[4] += py * qy;
    localH[5] += py * qz;
    localH[6] += pz * qx;
    localH[7] += pz * qy;
    localH[8] += pz * qz;
  }

  const double Sp = warpSum(localSp);
  const double Sq = warpSum(localSq);
  double       H[9];
  for (int i = 0; i < 9; ++i) {
    H[i] = warpSum(localH[i]);
  }

  // Lane 0 finishes the small matrix calculation and shares the RMSD with
  // the other lanes so they all make the same pruning decision.
  double rmsd = 0.0;
  if (lane == 0) {
    rmsd = alignedRmsd(Sp, Sq, H, invN);
  }
  return __shfl_sync(kFullWarpMask, rmsd, 0);
}

__global__ void selectOrderedConformersGreedyKernel(const double* __restrict__ coords,
                                                    const int32_t* __restrict__ atomStarts,
                                                    int32_t* __restrict__ groupedConfIds,
                                                    const ConformerPruningMolInfo* __restrict__ molInfos,
                                                    const int32_t* __restrict__ atomMaps,
                                                    uint8_t* __restrict__ selected,
                                                    const double threshold) {
  const ConformerPruningMolInfo info         = molInfos[blockIdx.x];
  const int*                    referenceMap = info.atomMapCount == 0 ? nullptr : atomMaps + info.atomMapBegin;
  const int                     warp         = threadIdx.x / warpSize;
  const int                     lane         = threadIdx.x % warpSize;
  __shared__ int                isConflict;
  __shared__ int                retainedCount;

  if (threadIdx.x == 0) {
    retainedCount = 0;
    isConflict    = 0;
  }
  __syncthreads();

  // Greedy pruning is order-dependent: a candidate is kept only when it does
  // not match an earlier kept conformer. One block owns one molecule so it can
  // process candidates in order while its warps compare earlier conformers in parallel.
  for (int candidateRank = 0; candidateRank < info.confCount; ++candidateRank) {
    const int     candidateConf   = groupedConfIds[info.confBegin + candidateRank];
    const double* candidateCoords = coords + static_cast<size_t>(atomStarts[candidateConf]) * kNumCoordinateDimensions;
    for (int previousBase = 0; previousBase < retainedCount; previousBase += blockDim.x / warpSize) {
      // Each warp checks one earlier retained conformer. Any warp can mark the shared
      // conflict flag, and the block stops as soon as one match is found.
      const int retainedIdx  = previousBase + warp;
      bool      warpConflict = false;
      if (retainedIdx < retainedCount) {
        const int     previousConf = groupedConfIds[info.confBegin + retainedIdx];
        const double* previousCoords =
          coords + static_cast<size_t>(atomStarts[previousConf]) * kNumCoordinateDimensions;
        for (int mapIdx = 0; mapIdx < info.atomMapCount; ++mapIdx) {
          // Compare the reference atom order with every allowed symmetry map.
          const int*   probeMap = referenceMap + mapIdx * info.mappedAtomCount;
          const double rmsd =
            computePairRmsdWarp(candidateCoords, previousCoords, referenceMap, probeMap, info.mappedAtomCount);
          if (rmsd < threshold) {
            warpConflict = true;
            break;
          }
        }
      }
      if (lane == 0 && warpConflict) {
        atomicExch(&isConflict, 1);
      }
      __syncthreads();
      if (isConflict) {
        break;
      }
    }

    if (threadIdx.x == 0) {
      const bool retainCandidate               = !isConflict;
      selected[info.confBegin + candidateRank] = retainCandidate;
      if (retainCandidate) {
        // Store retained IDs over entries we've already processed.
        groupedConfIds[info.confBegin + retainedCount] = candidateConf;
        ++retainedCount;
      }
      isConflict = 0;
    }
    __syncthreads();
  }
}

__global__ void compactConformerPositionsKernel(const double* __restrict__ positions,
                                                const int32_t* __restrict__ atomStarts,
                                                const int32_t* __restrict__ sourceConformerIds,
                                                const int32_t* __restrict__ compactedAtomStarts,
                                                double* __restrict__ compactedPositions) {
  // One block copies one retained conformer into its new contiguous range.
  const int    sourceConformer = sourceConformerIds[blockIdx.x];
  const size_t sourceBegin     = static_cast<size_t>(atomStarts[sourceConformer]) * kNumCoordinateDimensions;
  const size_t valueCount =
    static_cast<size_t>(atomStarts[sourceConformer + 1] - atomStarts[sourceConformer]) * kNumCoordinateDimensions;
  const size_t destinationBegin = static_cast<size_t>(compactedAtomStarts[blockIdx.x]) * kNumCoordinateDimensions;

  for (size_t value = threadIdx.x; value < valueCount; value += blockDim.x) {
    compactedPositions[destinationBegin + value] = positions[sourceBegin + value];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------

void alignConformersToFirstGpu(cuda::std::span<const double> coords,
                               cuda::std::span<double>       alignedCoords,
                               const int                     numConformers,
                               const int                     numAtoms,
                               cudaStream_t                  stream) {
  if (numConformers <= 0)
    return;
  alignConformersToFirstKernel<<<numConformers, kRmsdBlockSize, 0, stream>>>(coords.data(),
                                                                             alignedCoords.data(),
                                                                             numConformers,
                                                                             numAtoms);
  cudaCheckError(cudaGetLastError());
}

void alignConformersToFirstBatchGpu(cuda::std::span<const double> coords,
                                    cuda::std::span<double>       alignedCoords,
                                    cuda::std::span<const int>    conformerOffsets,
                                    cuda::std::span<const size_t> coordOffsets,
                                    cuda::std::span<const int>    numConfsPerMol,
                                    cuda::std::span<const int>    numAtomsPerMol,
                                    const int                     numMols,
                                    const int                     totalConformers,
                                    cudaStream_t                  stream) {
  if (totalConformers <= 0)
    return;
  alignConformersToFirstBatchKernel<<<totalConformers, kRmsdBlockSize, 0, stream>>>(coords.data(),
                                                                                    alignedCoords.data(),
                                                                                    conformerOffsets.data(),
                                                                                    coordOffsets.data(),
                                                                                    numConfsPerMol.data(),
                                                                                    numAtomsPerMol.data(),
                                                                                    numMols);
  cudaCheckError(cudaGetLastError());
}

void conformerRmsdMatrixGpu(cuda::std::span<const double> coords,
                            cuda::std::span<double>       rmsdOut,
                            const int                     numConformers,
                            const int                     numAtoms,
                            const bool                    prealigned,
                            cudaStream_t                  stream) {
  if (numConformers <= 1)
    return;

  const int64_t numPairs = static_cast<int64_t>(numConformers) * (numConformers - 1) / 2;
  if (numPairs > INT_MAX) {
    throw std::overflow_error("Number of conformer pairs exceeds maximum kernel grid size");
  }
  conformerRmsdKernel<<<static_cast<int>(numPairs), kRmsdBlockSize, 0, stream>>>(coords.data(),
                                                                                 rmsdOut.data(),
                                                                                 numConformers,
                                                                                 numAtoms,
                                                                                 prealigned);
  cudaCheckError(cudaGetLastError());
}

void conformerRmsdBatchMatrixGpu(cuda::std::span<const double> coords,
                                 cuda::std::span<double*>      rmsdOutputs,
                                 cuda::std::span<const int>    pairOffsets,
                                 cuda::std::span<const size_t> coordOffsets,
                                 cuda::std::span<const int>    numConfsPerMol,
                                 cuda::std::span<const int>    numAtomsPerMol,
                                 const int                     numMols,
                                 const int                     totalPairs,
                                 const bool                    prealigned,
                                 cudaStream_t                  stream) {
  if (totalPairs <= 0)
    return;

  conformerRmsdBatchKernel<<<totalPairs, kRmsdBlockSize, 0, stream>>>(coords.data(),
                                                                      rmsdOutputs.data(),
                                                                      pairOffsets.data(),
                                                                      coordOffsets.data(),
                                                                      numConfsPerMol.data(),
                                                                      numAtomsPerMol.data(),
                                                                      numMols,
                                                                      prealigned);
  cudaCheckError(cudaGetLastError());
}

void detail::conformerPruneMaskGpu(cuda::std::span<const double>                  coords,
                                   cuda::std::span<const int32_t>                 atomStarts,
                                   cuda::std::span<int32_t>                       groupedConfIds,
                                   cuda::std::span<const ConformerPruningMolInfo> molInfos,
                                   cuda::std::span<const int32_t>                 atomMaps,
                                   cuda::std::span<uint8_t>                       selected,
                                   const double                                   threshold,
                                   cudaStream_t                                   stream) {
  if (molInfos.empty()) {
    return;
  }

  selectOrderedConformersGreedyKernel<<<static_cast<int>(molInfos.size()), kGreedyBlockSize, 0, stream>>>(
    coords.data(),
    atomStarts.data(),
    groupedConfIds.data(),
    molInfos.data(),
    atomMaps.data(),
    selected.data(),
    threshold);
  cudaCheckError(cudaGetLastError());
}

void detail::compactConformerPositionsGpu(cuda::std::span<const double>  positions,
                                          cuda::std::span<const int32_t> atomStarts,
                                          cuda::std::span<const int32_t> sourceConformerIds,
                                          cuda::std::span<const int32_t> compactedAtomStarts,
                                          cuda::std::span<double>        compactedPositions,
                                          cudaStream_t                   stream) {
  if (sourceConformerIds.empty()) {
    return;
  }

  constexpr int kCompactionBlockSize = 128;
  compactConformerPositionsKernel<<<static_cast<int>(sourceConformerIds.size()), kCompactionBlockSize, 0, stream>>>(
    positions.data(),
    atomStarts.data(),
    sourceConformerIds.data(),
    compactedAtomStarts.data(),
    compactedPositions.data());
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit
