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

/// Compute determinant of a 3x3 matrix stored row-major.
__device__ __forceinline__ double det3x3(const double* H) {
  return H[0] * (H[4] * H[8] - H[5] * H[7]) - H[1] * (H[3] * H[8] - H[5] * H[6]) + H[2] * (H[3] * H[7] - H[4] * H[6]);
}

// ---------------------------------------------------------------------------
// Per-pair RMSD helper
//
// Called by both kernels after they resolve coordI, coordJ, and outRmsd from
// their respective index schemes.  All shared memory is managed internally.
//
// Each block computes:
//   1. Centroids of both conformers            (cub::BlockReduce, results broadcast)
//   2. Centered inner products  Sp, Sq         (cub::BlockReduce, thread 0 only)
//   3. Cross-covariance matrix  H = Pc^T Qc    (cub::BlockReduce, thread 0 only)
//   4. Singular values of H via eigenvalues of H^T H  (thread 0, analytical)
//   5. RMSD with optional Kabsch alignment      (thread 0)
// ---------------------------------------------------------------------------

constexpr int kRmsdBlockSize = 128;
using RmsdBlockReduceT       = cub::BlockReduce<double, kRmsdBlockSize>;

__device__ __forceinline__ void computePairRmsd(const double* __restrict__ coordI,
                                                const double* __restrict__ coordJ,
                                                const int  numAtoms,
                                                const bool prealigned,
                                                double*    outRmsd) {
  const int  tid = threadIdx.x;
  __shared__ RmsdBlockReduceT::TempStorage reduceTmp;

  if (prealigned) {
    // ---- Simple RMSD without alignment (no centering, matches RDKit behavior) ----
    double sumSqDiff = 0.0;
    for (int a = tid; a < numAtoms; a += kRmsdBlockSize) {
      const double dx = coordI[a * 3 + 0] - coordJ[a * 3 + 0];
      const double dy = coordI[a * 3 + 1] - coordJ[a * 3 + 1];
      const double dz = coordI[a * 3 + 2] - coordJ[a * 3 + 2];
      sumSqDiff += dx * dx + dy * dy + dz * dz;
    }
    const double total = RmsdBlockReduceT(reduceTmp).Sum(sumSqDiff);
    if (tid == 0)
      *outRmsd = sqrt(total / static_cast<double>(numAtoms));
    return;
  }

  // ---- Kabsch alignment path: compute centroids ----
  __shared__ double sCentI[3];
  __shared__ double sCentJ[3];

  double sumIx = 0.0, sumIy = 0.0, sumIz = 0.0;
  double sumJx = 0.0, sumJy = 0.0, sumJz = 0.0;
  for (int a = tid; a < numAtoms; a += kRmsdBlockSize) {
    sumIx += coordI[a * 3 + 0];
    sumIy += coordI[a * 3 + 1];
    sumIz += coordI[a * 3 + 2];
    sumJx += coordJ[a * 3 + 0];
    sumJy += coordJ[a * 3 + 1];
    sumJz += coordJ[a * 3 + 2];
  }

  // Reduce centroid components; write results to shared memory for broadcast.
  // Each __syncthreads() both allows TempStorage reuse and makes the previous
  // shared-memory write visible to all threads before the next reduction.
  const double invN = 1.0 / static_cast<double>(numAtoms);
  sumIx             = RmsdBlockReduceT(reduceTmp).Sum(sumIx);
  __syncthreads();
  if (tid == 0)
    sCentI[0] = sumIx * invN;
  sumIy = RmsdBlockReduceT(reduceTmp).Sum(sumIy);
  __syncthreads();
  if (tid == 0)
    sCentI[1] = sumIy * invN;
  sumIz = RmsdBlockReduceT(reduceTmp).Sum(sumIz);
  __syncthreads();
  if (tid == 0)
    sCentI[2] = sumIz * invN;
  sumJx = RmsdBlockReduceT(reduceTmp).Sum(sumJx);
  __syncthreads();
  if (tid == 0)
    sCentJ[0] = sumJx * invN;
  sumJy = RmsdBlockReduceT(reduceTmp).Sum(sumJy);
  __syncthreads();
  if (tid == 0)
    sCentJ[1] = sumJy * invN;
  sumJz = RmsdBlockReduceT(reduceTmp).Sum(sumJz);
  __syncthreads();
  if (tid == 0)
    sCentJ[2] = sumJz * invN;
  __syncthreads();  // broadcast sCentJ[2] and ensure all centroid writes are visible

  const double cIx = sCentI[0], cIy = sCentI[1], cIz = sCentI[2];
  const double cJx = sCentJ[0], cJy = sCentJ[1], cJz = sCentJ[2];

  // ---- Compute Sp, Sq, and cross-covariance H (Kabsch alignment) ----
  // Sp = sum ||pi - centI||^2,  Sq = sum ||qj - centJ||^2
  // H[r][c] = sum (pi[r] - centI[r]) * (qj[c] - centJ[c])
  double localSp = 0.0, localSq = 0.0;
  double localH[9] = {0.0};

  for (int a = tid; a < numAtoms; a += kRmsdBlockSize) {
    const double px = coordI[a * 3 + 0] - cIx;
    const double py = coordI[a * 3 + 1] - cIy;
    const double pz = coordI[a * 3 + 2] - cIz;
    const double qx = coordJ[a * 3 + 0] - cJx;
    const double qy = coordJ[a * 3 + 1] - cJy;
    const double qz = coordJ[a * 3 + 2] - cJz;

    localSp += px * px + py * py + pz * pz;
    localSq += qx * qx + qy * qy + qz * qz;

    // H = P^T Q  (sum of outer products)
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

  // Reduce all 11 values into thread 0.  Results in non-zero threads are
  // undefined and unused; only thread 0 performs the RMSD computation below.
  // __syncthreads() between calls allows TempStorage reuse.
  localSp = RmsdBlockReduceT(reduceTmp).Sum(localSp);
  __syncthreads();
  localSq = RmsdBlockReduceT(reduceTmp).Sum(localSq);
  __syncthreads();
  localH[0] = RmsdBlockReduceT(reduceTmp).Sum(localH[0]);
  __syncthreads();
  localH[1] = RmsdBlockReduceT(reduceTmp).Sum(localH[1]);
  __syncthreads();
  localH[2] = RmsdBlockReduceT(reduceTmp).Sum(localH[2]);
  __syncthreads();
  localH[3] = RmsdBlockReduceT(reduceTmp).Sum(localH[3]);
  __syncthreads();
  localH[4] = RmsdBlockReduceT(reduceTmp).Sum(localH[4]);
  __syncthreads();
  localH[5] = RmsdBlockReduceT(reduceTmp).Sum(localH[5]);
  __syncthreads();
  localH[6] = RmsdBlockReduceT(reduceTmp).Sum(localH[6]);
  __syncthreads();
  localH[7] = RmsdBlockReduceT(reduceTmp).Sum(localH[7]);
  __syncthreads();
  localH[8] = RmsdBlockReduceT(reduceTmp).Sum(localH[8]);
  // No final sync: only thread 0 reads localSp, localSq, localH below.

  // ---- Thread 0: compute RMSD from Sp, Sq, singular values of H ----
  if (tid == 0) {
    const double* H = localH;

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
    if (det3x3(H) < 0.0)
      s2 = -s2;

    // RMSD^2 = (Sp + Sq - 2*(s0 + s1 + s2)) / N
    const double rmsdSq = fmax((localSp + localSq - 2.0 * (s0 + s1 + s2)) * invN, 0.0);
    *outRmsd            = sqrt(rmsdSq);
  }
}

// ---------------------------------------------------------------------------
// Kernel: one thread-block per conformer pair
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

// ---------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------

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

}  // namespace nvMolKit
