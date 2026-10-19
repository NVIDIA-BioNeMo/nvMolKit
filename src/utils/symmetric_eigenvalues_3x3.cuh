// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_SYMMETRIC_EIGENVALUES_3X3_CUH
#define NVMOLKIT_SYMMETRIC_EIGENVALUES_3X3_CUH

#include <cmath>
#include <cuda/std/limits>

namespace nvMolKit {

/**
 * @brief Eigenvalues of a real symmetric 3x3 matrix in descending order, via Cardano's formula.
 *
 * Fast, but eigenvalues of a (near-)degenerate pair carry an error of order sqrt(eps) * ||A||. Use
 * symmetricEigenvaluesJacobi3x3() when degenerate spectra must be resolved to rounding accuracy.
 */
__device__ __forceinline__ void symmetricEigenvalues3x3(const double a00,
                                                        const double a01,
                                                        const double a02,
                                                        const double a11,
                                                        const double a12,
                                                        const double a22,
                                                        double&      largest,
                                                        double&      middle,
                                                        double&      smallest) {
  const double p       = a00 + a11 + a22;
  const double a00_a11 = a00 * a11;
  const double a00_a22 = a00 * a22;
  const double a11_a22 = a11 * a22;
  const double q       = a00_a11 + a00_a22 + a11_a22 - a01 * a01 - a02 * a02 - a12 * a12;
  const double r       = a00_a11 * a22 + 2.0 * a01 * a02 * a12 - a00 * a12 * a12 - a11 * a02 * a02 - a22 * a01 * a01;

  const double     p3               = p / 3.0;
  const double     pp               = (p * p - 3.0 * q) / 9.0;
  const double     qq               = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 54.0;
  const double     sqrtPP           = sqrt(fmax(pp, 0.0));
  const double     theta            = acos(fmin(fmax(qq / fmax(sqrtPP * sqrtPP * sqrtPP, 1.0e-30), -1.0), 1.0)) / 3.0;
  const double     twoSqrtPP        = 2.0 * sqrtPP;
  constexpr double kTwoPiOverThree  = 2.0943951023931954923;
  constexpr double kFourPiOverThree = 4.1887902047863909846;
  largest                           = twoSqrtPP * cos(theta) + p3;
  middle                            = twoSqrtPP * cos(theta - kTwoPiOverThree) + p3;
  smallest                          = twoSqrtPP * cos(theta - kFourPiOverThree) + p3;

  if (middle > largest) {
    const double tmp = largest;
    largest          = middle;
    middle           = tmp;
  }
  if (smallest > largest) {
    const double tmp = largest;
    largest          = smallest;
    smallest         = tmp;
  }
  if (smallest > middle) {
    const double tmp = middle;
    middle           = smallest;
    smallest         = tmp;
  }
}

namespace detail {

//! Jacobi rotation annihilating apq of a symmetric 3x3 matrix; r is the remaining index. With
//! @p kVectors, columns @p p and @p q of the row-major eigenvector matrix rotate with it.
template <typename Real, bool kVectors>
__device__ __forceinline__ void
jacobiRotate3x3(Real& app, Real& aqq, Real& apq, Real& arp, Real& arq, Real* eigenvectors, const int p, const int q) {
  if (apq == Real(0)) {
    return;
  }
  const Real theta = (aqq - app) / (Real(2) * apq);
  // Smaller-magnitude root of t^2 + 2 theta t - 1 = 0; the large-theta branch avoids overflow.
  const Real t     = fabs(theta) > Real(1e18) ? Real(0.5) / theta :
                                                copysign(Real(1), theta) / (fabs(theta) + sqrt(theta * theta + Real(1)));
  const Real c     = Real(1) / sqrt(t * t + Real(1));
  const Real s     = t * c;
  app -= t * apq;
  aqq += t * apq;
  apq           = Real(0);
  const Real rp = arp;
  const Real rq = arq;
  arp           = c * rp - s * rq;
  arq           = s * rp + c * rq;
  if constexpr (kVectors) {
    for (int row = 0; row < 3; ++row) {
      const Real vp             = eigenvectors[row * 3 + p];
      const Real vq             = eigenvectors[row * 3 + q];
      eigenvectors[row * 3 + p] = c * vp - s * vq;
      eigenvectors[row * 3 + q] = s * vp + c * vq;
    }
  }
}

//! Cyclic Jacobi sweeps until the off-diagonal part is negligible; leaves the eigenvalues on the
//! diagonal (a00, a11, a22), unsorted. With @p kVectors, @p eigenvectors starts as the identity and
//! ends with column i paired to the i-th diagonal entry.
template <typename Real, bool kVectors>
__device__ __forceinline__ void
jacobiDiagonalize3x3(Real& a00, Real& a01, Real& a02, Real& a11, Real& a12, Real& a22, Real* eigenvectors) {
  if constexpr (kVectors) {
    for (int i = 0; i < 9; ++i) {
      eigenvectors[i] = (i % 4 == 0) ? Real(1) : Real(0);
    }
  }
  // Quadratic convergence reaches rounding level within a few sweeps; the cap bounds non-finite input.
  constexpr int  kMaxSweeps = 16;
  constexpr Real kEpsilon   = cuda::std::numeric_limits<Real>::epsilon();
  for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
    const Real offDiagonal = a01 * a01 + a02 * a02 + a12 * a12;
    const Real diagonal    = a00 * a00 + a11 * a11 + a22 * a22;
    if (!(offDiagonal > kEpsilon * kEpsilon * diagonal)) {
      break;
    }
    jacobiRotate3x3<Real, kVectors>(a00, a11, a01, a02, a12, eigenvectors, 0, 1);  // r = 2
    jacobiRotate3x3<Real, kVectors>(a00, a22, a02, a01, a12, eigenvectors, 0, 2);  // r = 1
    jacobiRotate3x3<Real, kVectors>(a11, a22, a12, a01, a02, eigenvectors, 1, 2);  // r = 0
  }
}

}  // namespace detail

/**
 * @brief Eigenvalues of a real symmetric 3x3 matrix in descending order, via cyclic Jacobi rotations.
 *
 * Backward stable: every eigenvalue, including members of a degenerate pair, is accurate to a small
 * multiple of eps * ||A||. @p Real is float or double; all arithmetic is carried out in that type.
 */
template <typename Real>
__device__ __forceinline__ void symmetricEigenvaluesJacobi3x3(Real  a00,
                                                              Real  a01,
                                                              Real  a02,
                                                              Real  a11,
                                                              Real  a12,
                                                              Real  a22,
                                                              Real& largest,
                                                              Real& middle,
                                                              Real& smallest) {
  detail::jacobiDiagonalize3x3<Real, false>(a00, a01, a02, a11, a12, a22, nullptr);
  largest  = fmax(a00, fmax(a11, a22));
  smallest = fmin(a00, fmin(a11, a22));
  middle   = fmax(fmin(a00, a11), fmin(fmax(a00, a11), a22));  // Median of three, exact.
}

/**
 * @brief Eigenvalues and eigenvectors of a real symmetric 3x3 matrix, via cyclic Jacobi rotations.
 *
 * Backward stable like symmetricEigenvaluesJacobi3x3(). The pairs are returned unsorted:
 * @p eigenvalues[i] belongs to column i of the row-major @p eigenvectors (`eigenvectors[row * 3 + i]`),
 * and the columns are orthonormal.
 */
template <typename Real>
__device__ __forceinline__ void symmetricEigensystemJacobi3x3(Real a00,
                                                              Real a01,
                                                              Real a02,
                                                              Real a11,
                                                              Real a12,
                                                              Real a22,
                                                              Real (&eigenvalues)[3],
                                                              Real (&eigenvectors)[9]) {
  detail::jacobiDiagonalize3x3<Real, true>(a00, a01, a02, a11, a12, a22, eigenvectors);
  eigenvalues[0] = a00;
  eigenvalues[1] = a11;
  eigenvalues[2] = a22;
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_SYMMETRIC_EIGENVALUES_3X3_CUH
