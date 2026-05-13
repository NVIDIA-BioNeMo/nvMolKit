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

#include <cub/cub.cuh>

#include "bfgs_minimize_permol_kernels.h"
#include "cub_helpers.cuh"
#include "device_vector.h"
#include "dist_geom_kernels_device.cuh"
#include "mmff_kernels.h"
#include "mmff_kernels_device.cuh"

namespace nvMolKit {

namespace {
constexpr int16_t BLOCK_SIZE           = 128;
constexpr int16_t MAX_LINESEARCH_ITERS = 1000;
constexpr double  FUNCTOL              = 1e-4;
constexpr double  MOVETOL              = 1e-7;
constexpr double  TOLX                 = 4. * 3e-8;

__device__ void setMaxStep(const double*                                               pos,
                           const int                                                   numTerms,
                           float*                                                      maxStepOutSquared,
                           typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  float sumSquaredPos = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    float dx2 = pos[i] * pos[i];
    sumSquaredPos += dx2;
  }
  using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;

  const float squaredSum = BlockReduce(tempStorage).Sum(sumSquaredPos);
  if (threadIdx.x == 0) {
    constexpr float maxStepFactorSquared = 100.0 * 100.0;
    *maxStepOutSquared =
      maxStepFactorSquared * max(squaredSum, static_cast<float>(numTerms) * static_cast<float>(numTerms));
  }
}

__device__ void lineSearchSetup(const int                                                   numTerms,
                                const double*                                               posStart,
                                const double*                                               gradStart,
                                const float                                                 maxStepSquared,
                                double*                                                     dirStart,
                                double&                                                     slope,
                                double&                                                     lambdaMin,
                                typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  const int idxInSys = threadIdx.x;
  using BlockReduce  = cub::BlockReduce<double, BLOCK_SIZE>;
  __shared__ float dirSumSquared;

  // ---------------------------------
  //  Scale direction vector if needed
  // ---------------------------------
  float sumSquaredLocal = 0.0;
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    float dx2 = dirStart[i] * dirStart[i];
    sumSquaredLocal += dx2;
  }
  float blockSum = BlockReduce(tempStorage).Sum(sumSquaredLocal);
  if (idxInSys == 0) {
    dirSumSquared = blockSum;
  }
  __syncthreads();
  if (dirSumSquared > maxStepSquared) {
    const float inverseScaleSquared = dirSumSquared / maxStepSquared;
    const float scale               = rsqrtf(inverseScaleSquared);
    for (int i = idxInSys; i < numTerms; i += blockDim.x) {
      dirStart[i] *= scale;
    }
  }
  __syncthreads();

  // -------------------------
  // Set slope, check validity
  // -------------------------
  float localSum     = 0.0;
  float localGradSum = 0.0;
  float localDirSum  = 0.0;
  // Each thread computes its partial sum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    localSum += dirStart[i] * gradStart[i];
    localGradSum += gradStart[i] * gradStart[i];
    localDirSum += dirStart[i] * dirStart[i];
  }

  // Perform block-wide reduction to compute the total sum
  blockSum = BlockReduce(tempStorage).Sum(localSum);

  // The first thread in the block writes the result
  if (idxInSys == 0) {
    slope = blockSum;
  }
  __syncthreads();

  // ----------------------
  // Compute initial lambda
  // ----------------------
  float localMax_numerator   = 0.0;
  float localMax_denominator = 1.0;
  // Each thread computes its local maximum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    float temp_numerator   = fabs(dirStart[i]);
    float temp_denominator = fmax(fabs(posStart[i]), 1.0);
    // temp_numerator / temp_denominator > localMax_numerator / localMax_denominator
    // <=>
    // temp_numerator * localMax_denominator > localMax_numerator * temp_denominator
    if (temp_numerator * localMax_denominator > localMax_numerator * temp_denominator) {
      localMax_numerator   = temp_numerator;
      localMax_denominator = temp_denominator;
    }
  }

  float localInvMax = localMax_denominator / (localMax_numerator > 0.0f ? localMax_numerator : 1.0e-20f);
  // Perform block-wide reduction to find the maximum
  float blockInvMax = BlockReduce(tempStorage).Reduce(static_cast<double>(localInvMax), cubMin());

  // The first thread in the block writes the result
  if (threadIdx.x == 0) {
    lambdaMin = static_cast<float>(MOVETOL) * blockInvMax;
  }
}

__device__ void lineSearchPerturb(const int     numTerms,
                                  const double* refPos,
                                  const double* dirStart,
                                  const float   lambda,
                                  double*       scratchPos) {
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    scratchPos[i] = refPos[i] + lambda * dirStart[i];
  }
  __syncthreads();
}

__device__ bool lineSearchPostEnergy(const bool  isFirstIter,
                                     const float prevE,
                                     const float newE,
                                     const float slope,
                                     const float lambda,
                                     const float lambdaMin,
                                     double&     lambda2,
                                     double&     eScratch,
                                     double&     lambdaOut) {
  bool converged = false;

  if (threadIdx.x == 0) {
    const float eDiff = newE - prevE;
    if (lambda < lambdaMin || eDiff <= FUNCTOL * lambda * slope) {
      converged = true;
    } else {
      float tmpLambda;
      if (isFirstIter) {
        tmpLambda = -slope / (2.0f * (eDiff - slope));
      } else {
        const float rhs1     = eDiff - lambda * slope;
        const float rhs2     = eScratch - prevE - lambda2 * slope;
        const float rLambda  = 1.0f / static_cast<float>(lambda);
        const float rLambda2 = 1.0f / static_cast<float>(lambda2);
        const float rScale   = 1.0f / (lambda - static_cast<float>(lambda2));
        const float a        = (rhs1 * rLambda * rLambda - rhs2 * rLambda2 * rLambda2) * rScale;
        const float b        = (-lambda2 * rhs1 * rLambda * rLambda + lambda * rhs2 * rLambda2 * rLambda2) * rScale;
        if (a == 0.0f) {
          tmpLambda = -slope / (2.0f * b);
        } else {
          const float disc = b * b - 3.0f * a * slope;
          if (disc < 0.0f) {
            tmpLambda = 0.5f * lambda;
          } else {
            const float sqrtDisc = sqrtf(disc);
            tmpLambda            = (b <= 0.0f) ? (-b + sqrtDisc) / (3.0f * a) : -slope / (b + sqrtDisc);
          }
        }
        tmpLambda = fminf(tmpLambda, 0.5f * lambda);
      }
      lambda2   = lambda;
      eScratch  = newE;
      lambdaOut = fmaxf(tmpLambda, 0.1f * lambda);
    }
  }
  __syncthreads();
  return converged;
}

__device__ void setDirection(const int                                                   numTerms,
                             const double*                                               posFromLineSearch,
                             const double*                                               pos,
                             double*                                                     xi,
                             double*                                                     dGrad,
                             const double*                                               grad,
                             bool&                                                       converged,
                             typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  float localMax_numerator   = 0.0;
  float localMax_denominator = 1.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    xi[i]    = posFromLineSearch[i] - pos[i];
    dGrad[i] = grad[i];

    float temp_numerator   = fabs(xi[i]);
    float temp_denominator = fmax(fabs(posFromLineSearch[i]), 1.0);
    // temp_numerator / temp_denominator > localMax_numerator / localMax_denominator
    // <=>
    // temp_numerator * localMax_denominator > localMax_numerator * temp_denominator
    if (temp_numerator * localMax_denominator > localMax_numerator * temp_denominator) {
      localMax_numerator   = temp_numerator;
      localMax_denominator = temp_denominator;
    }
  }

  float localMax = localMax_numerator / localMax_denominator;
  float blockMax = cub::BlockReduce<double, BLOCK_SIZE>(tempStorage).Reduce(localMax, cubMax());

  if (threadIdx.x == 0 && blockMax < TOLX) {
    converged = true;
  }
  __syncthreads();
}

template <bool scaleGrads>
__device__ void scaleGrad(const int                                                   numTerms,
                          double*                                                     grad,
                          double&                                                     gradScale,
                          typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  gradScale = scaleGrads ? 0.1 : 1.0;

  double maxGrad = -1e8;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    if constexpr (scaleGrads) {
      grad[i] *= gradScale;
    }
    if (grad[i] > maxGrad) {
      maxGrad = grad[i];
    }
  }

  double blockMax = cub::BlockReduce<double, BLOCK_SIZE>(tempStorage).Reduce(maxGrad, cubMax());

  __shared__ double distributedMax[1];
  if (threadIdx.x == 0) {
    distributedMax[0] = blockMax;
  }
  __syncthreads();

  maxGrad = distributedMax[0];

  if (scaleGrads && maxGrad > 10.0) {
    while (maxGrad * gradScale > 10.0) {
      gradScale *= 0.5;
    }
    for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
      grad[i] *= gradScale;
    }
  }
  __syncthreads();
}

__device__ void updateDGrad(const int                                                   numTerms,
                            const double                                                gradTol,
                            const double                                                energy,
                            const double                                                gradScale,
                            const double*                                               grad,
                            const double*                                               pos,
                            double*                                                     dGrad,
                            bool&                                                       converged,
                            typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  double localMax = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    dGrad[i]    = grad[i] - dGrad[i];
    double temp = fabs(grad[i]) * fmax(fabs(pos[i]), 1.0);
    if (temp > localMax) {
      localMax = temp;
    }
  }

  float blockMax = cub::BlockReduce<double, BLOCK_SIZE>(tempStorage).Reduce(localMax, cubMax());

  if (threadIdx.x == 0) {
    const float term = max(energy * gradScale, 1.0);
    blockMax /= term;
    if (blockMax < gradTol) {
      converged = true;
    }
  }
  __syncthreads();
}

__device__ void updateInverseHessian(const int                                                   numTerms,
                                     double*                                                     invHessian,
                                     double*                                                     dGrad,
                                     double*                                                     xi,
                                     double*                                                     hessDGrad,
                                     double*                                                     grad,
                                     typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;

  // Compute hessDGrad = invHessian * dGrad
  for (int row = threadIdx.x; row < numTerms; row += blockDim.x) {
    double dotProduct = 0.0;
    for (int col = 0; col < numTerms; col++) {
      dotProduct += invHessian[col * numTerms + row] * dGrad[col];
    }
    hessDGrad[row] = dotProduct;
  }
  __syncthreads();

  // Compute BFGS sums
  __shared__ double fac, fae, fad, sumDGrad, sumXi;
  __shared__ bool   needUpdate;

  double sumFac = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumFac += dGrad[i] * xi[i];
  }
  double facReduced = BlockReduce(tempStorage).Sum(sumFac);
  if (threadIdx.x == 0)
    fac = facReduced;
  __syncthreads();

  double sumFae = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumFae += dGrad[i] * hessDGrad[i];
  }
  double faeReduced = BlockReduce(tempStorage).Sum(sumFae);
  if (threadIdx.x == 0)
    fae = faeReduced;
  __syncthreads();

  double sumDGradSq = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumDGradSq += dGrad[i] * dGrad[i];
  }
  double sumDGradReduced = BlockReduce(tempStorage).Sum(sumDGradSq);
  if (threadIdx.x == 0)
    sumDGrad = sumDGradReduced;
  __syncthreads();

  double sumXiSq = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumXiSq += xi[i] * xi[i];
  }
  double sumXiReduced = BlockReduce(tempStorage).Sum(sumXiSq);
  if (threadIdx.x == 0)
    sumXi = sumXiReduced;
  __syncthreads();

  if (threadIdx.x == 0) {
    constexpr double EPS = 3e-8;
    needUpdate           = (fac > 0) && ((fac * fac) > (EPS * sumDGrad * sumXi));

    if (needUpdate) {
      fac = 1.0 / fac;
      fad = 1.0 / fae;
    }
  }
  __syncthreads();

  if (needUpdate) {
    // Update dGrad for Hessian update
    for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
      dGrad[i] = fac * xi[i] - fad * hessDGrad[i];
    }
    __syncthreads();

    // Update inverse Hessian
    for (int row = threadIdx.x; row < numTerms; row += blockDim.x) {
      double pxi  = fac * xi[row];
      double hdgi = fad * hessDGrad[row];
      double dgi  = fae * dGrad[row];

      for (int col = 0; col < numTerms; col++) {
        double pxj    = xi[col];
        double hdgj   = hessDGrad[col];
        double dgj    = dGrad[col];
        double update = pxi * pxj - hdgi * hdgj + dgi * dgj;
        invHessian[col * numTerms + row] += update;
      }
    }
    __syncthreads();
  }

  // Update xi = -invHessian * grad
  for (int row = threadIdx.x; row < numTerms; row += blockDim.x) {
    double dotProduct = 0.0;
    for (int col = 0; col < numTerms; col++) {
      dotProduct += invHessian[col * numTerms + row] * grad[col];
    }
    xi[row] = -dotProduct;
  }
  __syncthreads();
}

// Helper to get data dimensionality from ForceFieldType at compile time
template <ForceFieldType FFType> struct DataDimTraits;

template <> struct DataDimTraits<ForceFieldType::MMFF> {
  static constexpr int value = 3;
};

template <> struct DataDimTraits<ForceFieldType::ETK> {
  static constexpr int value = 4;
};

template <> struct DataDimTraits<ForceFieldType::DG> {
  static constexpr int value = 4;
};

}  // namespace

template <int            MaxAtoms,
          bool           UseSharedMem,
          ForceFieldType FFType,
          bool           HasConstraints,
          typename TermsType,
          typename IndicesType>
__launch_bounds__(BLOCK_SIZE) __global__ void bfgsMinimizeKernel(const int               numIters,
                                                                 const double            gradTol,
                                                                 const bool              scaleGrads,
                                                                 const TermsType*        terms,
                                                                 const IndicesType*      systemIndices,
                                                                 const int*              molIdList,
                                                                 const int*              atomStarts,
                                                                 const int*              hessianStarts,
                                                                 double*                 positions,
                                                                 double*                 grad,
                                                                 double*                 inverseHessian,
                                                                 double**                scratchBuffers,
                                                                 double*                 energyOuts,
                                                                 int16_t*                statuses,
                                                                 [[maybe_unused]] double chiralWeight,
                                                                 [[maybe_unused]] double fourthDimWeight) {
  const int     molIdx = molIdList[blockIdx.x];
  const int16_t tid    = threadIdx.x;

  const int     atomStart = atomStarts[molIdx];
  const int     atomEnd   = atomStarts[molIdx + 1];
  const int16_t numAtoms  = atomEnd - atomStart;

  // Use compile-time dimension for correctness
  constexpr int16_t dataDim  = DataDimTraits<FFType>::value;
  constexpr int16_t maxTerms = MaxAtoms * dataDim;
  const int16_t     numTerms = dataDim * numAtoms;

  // Pointers to working memory (either shared or global)
  double* localPos;
  double* localGrad;
  double* localDir;
  double* scratchPos;
  double* dGrad;
  double* oldPos;

  if constexpr (UseSharedMem) {
    // Shared memory for small molecules (≤64 atoms)
    __shared__ double sharedLocalPos[maxTerms];
    __shared__ double sharedLocalGrad[maxTerms];
    __shared__ double sharedLocalDir[maxTerms];
    __shared__ double sharedScratchPos[maxTerms];
    __shared__ double sharedDGrad[maxTerms];

    const int termStart = atomStart * dataDim;
    localPos            = sharedLocalPos;
    localGrad           = sharedLocalGrad;
    localDir            = sharedLocalDir;
    scratchPos          = sharedScratchPos;
    dGrad               = sharedDGrad;
    // For small molecules, grad buffer is unused (using sharedLocalGrad), so reuse it for oldPos
    oldPos              = scratchBuffers[0] + termStart;  // Reuse grad buffer for oldPos
  } else {
    // Global memory for large molecules (>64 atoms) - index into pre-allocated buffers
    const int termStart = atomStart * dataDim;
    localPos            = positions + atomStart * dataDim;  // Use main positions array directly (no separate copy)
    localGrad           = grad + termStart;                 // Use main gradient buffer
    localDir            = scratchBuffers[1] + termStart;    // lineSearchDir
    scratchPos          = scratchBuffers[2] + termStart;    // scratchPositions
    dGrad               = scratchBuffers[3] + termStart;    // hessDGrad
    oldPos              = scratchBuffers[4] + termStart;    // scratchGrad (used as oldPos)
  }

  // Shared scalars
  __shared__ float  maxStep;
  __shared__ double prevE;
  __shared__ double currE;
  __shared__ double slope;
  __shared__ double lambda;
  __shared__ double lambdaMin;
  __shared__ double lambda2;
  __shared__ double eScratch;
  __shared__ double gradScale;
  __shared__ bool   converged;
  __shared__ bool   lineSearchConverged;

  // Inverse Hessian in global memory (O(n^2), too large for shared)
  // Indexed by hessianStarts which stores cumulative (numTerms * numTerms) offsets
  double* invHessian = inverseHessian + hessianStarts[molIdx];

  // Initialize positions from global memory
  double* globalPos = positions + atomStart * dataDim;
  // For shared memory case, copy to local shared buffer
  // For non-shared case, localPos already points to globalPos, so no copy needed
  if constexpr (UseSharedMem) {
    for (int i = tid; i < numTerms; i += blockDim.x) {
      localPos[i] = globalPos[i];
    }
    __syncthreads();
  }

  // Initialize inverse Hessian to identity
  const int hessianSize = numTerms * numTerms;
  for (int i = tid; i < hessianSize; i += blockDim.x) {
    invHessian[i] = 0.0;
  }
  __syncthreads();
  for (int i = tid; i < numTerms; i += blockDim.x) {
    invHessian[i * numTerms + i] = 1.0;
  }

  if (tid == 0) {
    converged = false;
  }
  __syncthreads();

  // Shared temp storage for all BlockReduce operations
  using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  // Compute initial energy
  double threadEnergy;
  if constexpr (FFType == ForceFieldType::MMFF) {
    threadEnergy = MMFF::molEnergy<BLOCK_SIZE, HasConstraints>(*terms, *systemIndices, localPos, molIdx, tid);
  } else if constexpr (FFType == ForceFieldType::ETK) {
    threadEnergy = DistGeom::molEnergyETK(*terms, *systemIndices, localPos, molIdx, tid);
  } else {  // DG
    threadEnergy =
      DistGeom::molEnergyDG<dataDim>(*terms, *systemIndices, localPos, molIdx, chiralWeight, fourthDimWeight, tid);
  }
  const double blockEnergy = BlockReduce(tempStorage).Sum(threadEnergy);

  if (tid == 0) {
    prevE              = blockEnergy;
    energyOuts[molIdx] = blockEnergy;
  }
  __syncthreads();

  for (int i = tid; i < numTerms; i += blockDim.x) {
    localGrad[i] = 0.0;
  }
  __syncthreads();

  if constexpr (FFType == ForceFieldType::MMFF) {
    MMFF::molGrad<BLOCK_SIZE, HasConstraints>(*terms, *systemIndices, localPos, localGrad, molIdx, tid);
  } else if constexpr (FFType == ForceFieldType::ETK) {
    DistGeom::molGradETK(*terms, *systemIndices, localPos, localGrad, molIdx, tid);
  } else {  // DG
    DistGeom::molGradDG<dataDim>(*terms,
                                 *systemIndices,
                                 localPos,
                                 localGrad,
                                 molIdx,
                                 chiralWeight,
                                 fourthDimWeight,
                                 tid);
  }
  __syncthreads();

  // Scale gradients
  if (scaleGrads) {
    scaleGrad<true>(numTerms, localGrad, gradScale, tempStorage);
  } else {
    scaleGrad<false>(numTerms, localGrad, gradScale, tempStorage);
  }
  // Set initial direction as negative gradient
  for (int i = tid; i < numTerms; i += blockDim.x) {
    localDir[i] = -localGrad[i];
  }
  __syncthreads();

  // Set max step
  setMaxStep(localPos, numTerms, &maxStep, tempStorage);
  __syncthreads();

  // Main BFGS loop
  __shared__ int currIter;
  if (tid == 0) {
    currIter = 0;
  }
  __syncthreads();

  while (!converged && currIter < numIters) {
    // Save current position before line search
    for (int i = tid; i < numTerms; i += blockDim.x) {
      oldPos[i] = localPos[i];
    }
    __syncthreads();

    // Line search setup
    if (tid == 0) {
      lineSearchConverged = false;
      lambda              = 1.0;
    }
    __syncthreads();

    lineSearchSetup(numTerms, localPos, localGrad, maxStep, localDir, slope, lambdaMin, tempStorage);
    __syncthreads();

    // Line search loop
    __shared__ int16_t lineSearchIter;
    if (tid == 0) {
      lineSearchIter = 0;
    }
    __syncthreads();

    while (!lineSearchConverged && lineSearchIter < MAX_LINESEARCH_ITERS) {
      // Perturb positions from saved oldPos (not localPos, which may have been modified)
      lineSearchPerturb(numTerms, oldPos, localDir, lambda, scratchPos);

      // Compute energy at perturbed position (use scratchPos which has the perturbed coordinates)
      double lsThreadEnergy;
      if constexpr (FFType == ForceFieldType::MMFF) {
        lsThreadEnergy = MMFF::molEnergy<BLOCK_SIZE, HasConstraints>(*terms, *systemIndices, scratchPos, molIdx, tid);
      } else if constexpr (FFType == ForceFieldType::ETK) {
        lsThreadEnergy = DistGeom::molEnergyETK(*terms, *systemIndices, scratchPos, molIdx, tid);
      } else {  // DG
        lsThreadEnergy = DistGeom::molEnergyDG<dataDim>(*terms,
                                                        *systemIndices,
                                                        scratchPos,
                                                        molIdx,
                                                        chiralWeight,
                                                        fourthDimWeight,
                                                        tid);
      }
      const double lsBlockEnergy = BlockReduce(tempStorage).Sum(lsThreadEnergy);

      if (tid == 0) {
        currE = lsBlockEnergy;
      }
      __syncthreads();

      // Check convergence and update lambda
      lineSearchConverged =
        lineSearchPostEnergy(lineSearchIter == 0, prevE, currE, slope, lambda, lambdaMin, lambda2, eScratch, lambda);
      __syncthreads();

      if (tid == 0) {
        lineSearchIter++;
      }
      __syncthreads();
    }

    // Update positions with final line search result and compute direction
    for (int i = tid; i < numTerms; i += blockDim.x) {
      localPos[i] = scratchPos[i];
    }
    __syncthreads();

    // Set direction (compute xi = new - old)
    setDirection(numTerms, scratchPos, oldPos, localDir, dGrad, localGrad, converged, tempStorage);
    if (converged) {
      break;
    }

    // Update stored energy for next iteration
    if (tid == 0) {
      prevE = currE;
    }
    __syncthreads();

    // Compute gradients at new position
    for (int i = tid; i < numTerms; i += blockDim.x) {
      localGrad[i] = 0.0;
    }
    __syncthreads();

    if constexpr (FFType == ForceFieldType::MMFF) {
      MMFF::molGrad<BLOCK_SIZE, HasConstraints>(*terms, *systemIndices, localPos, localGrad, molIdx, tid);
    } else if constexpr (FFType == ForceFieldType::ETK) {
      DistGeom::molGradETK(*terms, *systemIndices, localPos, localGrad, molIdx, tid);
    } else {  // DG
      DistGeom::molGradDG<dataDim>(*terms,
                                   *systemIndices,
                                   localPos,
                                   localGrad,
                                   molIdx,
                                   chiralWeight,
                                   fourthDimWeight,
                                   tid);
    }
    __syncthreads();

    // Scale gradients
    if (scaleGrads) {
      scaleGrad<true>(numTerms, localGrad, gradScale, tempStorage);
    } else {
      scaleGrad<false>(numTerms, localGrad, gradScale, tempStorage);
    }

    // Update dGrad and check convergence
    updateDGrad(numTerms, gradTol, currE, gradScale, localGrad, localPos, dGrad, converged, tempStorage);
    if (converged) {
      break;
    }

    // Update Hessian and compute new direction (reuses scratchPos as hessDGrad)
    updateInverseHessian(numTerms, invHessian, dGrad, localDir, scratchPos, localGrad, tempStorage);

    if (tid == 0) {
      currIter++;
    }
    __syncthreads();
  }

  // If in shared mem mode, we've been updating positions in shared memory. Copy back to global memory
  // If not in shared memory mode, it's already in global memory
  if constexpr (UseSharedMem) {
    for (int i = tid; i < numTerms; i += blockDim.x) {
      globalPos[i] = localPos[i];
    }
  }

  // Write final energy and status
  if (tid == 0) {
    energyOuts[molIdx] = prevE;
    // Write status to match batched kernel behavior (0 = converged, 1 = not converged)
    if (statuses != nullptr) {
      statuses[molIdx] = converged ? 0 : 1;
    }
  }
}

namespace {

template <int            MaxAtoms,
          bool           UseSharedMem,
          ForceFieldType FFType,
          bool           HasConstraints,
          typename TermsType,
          typename IndicesType>
cudaError_t launchKernelForSize(int                numMols,
                                const int*         molIdList,
                                int                numIters,
                                double             gradTol,
                                bool               scaleGrads,
                                const TermsType*   devTerms,
                                const IndicesType* devSysIdx,
                                const int*         atomStarts,
                                const int*         hessianStarts,
                                double*            positions,
                                double*            grad,
                                double*            inverseHessian,
                                double**           scratchBuffers,
                                double*            energyOuts,
                                int16_t*           statuses,
                                cudaStream_t       stream,
                                double             chiralWeight,
                                double             fourthDimWeight) {
  if (numMols == 0) {
    return cudaSuccess;
  }

  bfgsMinimizeKernel<MaxAtoms, UseSharedMem, FFType, HasConstraints, TermsType, IndicesType>
    <<<numMols, BLOCK_SIZE, 0, stream>>>(numIters,
                                         gradTol,
                                         scaleGrads,
                                         devTerms,
                                         devSysIdx,
                                         molIdList,
                                         atomStarts,
                                         hessianStarts,
                                         positions,
                                         grad,
                                         inverseHessian,
                                         scratchBuffers,
                                         energyOuts,
                                         statuses,
                                         chiralWeight,
                                         fourthDimWeight);

  return cudaGetLastError();
}

template <ForceFieldType FFType, bool HasConstraints, typename TermsType, typename IndicesType>
cudaError_t dispatchByMaxAtoms(int                numMols,
                               const int*         molIdList,
                               int                maxAtoms,
                               int                numIters,
                               double             gradTol,
                               bool               scaleGrads,
                               const TermsType*   devTerms,
                               const IndicesType* devSysIdx,
                               const int*         atomStarts,
                               const int*         hessianStarts,
                               double*            positions,
                               double*            grad,
                               double*            inverseHessian,
                               double**           scratchBuffers,
                               double*            energyOuts,
                               int16_t*           statuses,
                               cudaStream_t       stream,
                               double             chiralWeight,
                               double             fourthDimWeight) {
  // Use shared memory for <=128 atoms (in increments of 32), global memory for larger
  if (maxAtoms <= 32) {
    return launchKernelForSize<32, true, FFType, HasConstraints>(numMols,
                                                                 molIdList,
                                                                 numIters,
                                                                 gradTol,
                                                                 scaleGrads,
                                                                 devTerms,
                                                                 devSysIdx,
                                                                 atomStarts,
                                                                 hessianStarts,
                                                                 positions,
                                                                 grad,
                                                                 inverseHessian,
                                                                 scratchBuffers,
                                                                 energyOuts,
                                                                 statuses,
                                                                 stream,
                                                                 chiralWeight,
                                                                 fourthDimWeight);
  } else if (maxAtoms <= 64) {
    return launchKernelForSize<64, true, FFType, HasConstraints>(numMols,
                                                                 molIdList,
                                                                 numIters,
                                                                 gradTol,
                                                                 scaleGrads,
                                                                 devTerms,
                                                                 devSysIdx,
                                                                 atomStarts,
                                                                 hessianStarts,
                                                                 positions,
                                                                 grad,
                                                                 inverseHessian,
                                                                 scratchBuffers,
                                                                 energyOuts,
                                                                 statuses,
                                                                 stream,
                                                                 chiralWeight,
                                                                 fourthDimWeight);
  } else if (maxAtoms <= 96) {
    return launchKernelForSize<96, true, FFType, HasConstraints>(numMols,
                                                                 molIdList,
                                                                 numIters,
                                                                 gradTol,
                                                                 scaleGrads,
                                                                 devTerms,
                                                                 devSysIdx,
                                                                 atomStarts,
                                                                 hessianStarts,
                                                                 positions,
                                                                 grad,
                                                                 inverseHessian,
                                                                 scratchBuffers,
                                                                 energyOuts,
                                                                 statuses,
                                                                 stream,
                                                                 chiralWeight,
                                                                 fourthDimWeight);
  } else if (maxAtoms <= 128) {
    return launchKernelForSize<128, true, FFType, HasConstraints>(numMols,
                                                                  molIdList,
                                                                  numIters,
                                                                  gradTol,
                                                                  scaleGrads,
                                                                  devTerms,
                                                                  devSysIdx,
                                                                  atomStarts,
                                                                  hessianStarts,
                                                                  positions,
                                                                  grad,
                                                                  inverseHessian,
                                                                  scratchBuffers,
                                                                  energyOuts,
                                                                  statuses,
                                                                  stream,
                                                                  chiralWeight,
                                                                  fourthDimWeight);
  } else if (maxAtoms <= 256) {
    return launchKernelForSize<256, false, FFType, HasConstraints>(numMols,
                                                                   molIdList,
                                                                   numIters,
                                                                   gradTol,
                                                                   scaleGrads,
                                                                   devTerms,
                                                                   devSysIdx,
                                                                   atomStarts,
                                                                   hessianStarts,
                                                                   positions,
                                                                   grad,
                                                                   inverseHessian,
                                                                   scratchBuffers,
                                                                   energyOuts,
                                                                   statuses,
                                                                   stream,
                                                                   chiralWeight,
                                                                   fourthDimWeight);
  } else {
    return launchKernelForSize<2048, false, FFType, HasConstraints>(numMols,
                                                                    molIdList,
                                                                    numIters,
                                                                    gradTol,
                                                                    scaleGrads,
                                                                    devTerms,
                                                                    devSysIdx,
                                                                    atomStarts,
                                                                    hessianStarts,
                                                                    positions,
                                                                    grad,
                                                                    inverseHessian,
                                                                    scratchBuffers,
                                                                    energyOuts,
                                                                    statuses,
                                                                    stream,
                                                                    chiralWeight,
                                                                    fourthDimWeight);
  }
}

}  // namespace

cudaError_t launchBfgsMinimizePerMolKernel(int                                       numMols,
                                           const int*                                molIds,
                                           int                                       maxAtoms,
                                           const int*                                atomStarts,
                                           const int*                                hessianStarts,
                                           int                                       numIters,
                                           double                                    gradTol,
                                           bool                                      scaleGrads,
                                           const MMFF::EnergyForceContribsDevicePtr& terms,
                                           const MMFF::BatchedIndicesDevicePtr&      systemIndices,
                                           double*                                   positions,
                                           double*                                   grad,
                                           double*                                   inverseHessian,
                                           double**                                  scratchBuffers,
                                           double*                                   energyOuts,
                                           bool                                      hasConstraints,
                                           int16_t*                                  statuses,
                                           cudaStream_t                              stream) {
  if (numMols == 0) {
    return cudaSuccess;
  }

  const AsyncDevicePtr<MMFF::EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<MMFF::BatchedIndicesDevicePtr>      devSysIdx(systemIndices, stream);

  if (hasConstraints) {
    return dispatchByMaxAtoms<ForceFieldType::MMFF, true>(numMols,
                                                          molIds,
                                                          maxAtoms,
                                                          numIters,
                                                          gradTol,
                                                          scaleGrads,
                                                          devTerms.data(),
                                                          devSysIdx.data(),
                                                          atomStarts,
                                                          hessianStarts,
                                                          positions,
                                                          grad,
                                                          inverseHessian,
                                                          scratchBuffers,
                                                          energyOuts,
                                                          statuses,
                                                          stream,
                                                          1.0,
                                                          1.0);
  }
  return dispatchByMaxAtoms<ForceFieldType::MMFF, false>(numMols,
                                                         molIds,
                                                         maxAtoms,
                                                         numIters,
                                                         gradTol,
                                                         scaleGrads,
                                                         devTerms.data(),
                                                         devSysIdx.data(),
                                                         atomStarts,
                                                         hessianStarts,
                                                         positions,
                                                         grad,
                                                         inverseHessian,
                                                         scratchBuffers,
                                                         energyOuts,
                                                         statuses,
                                                         stream,
                                                         1.0,
                                                         1.0);
}
cudaError_t launchBfgsMinimizePerMolKernelETK(int                                             numMols,
                                              const int*                                      molIds,
                                              int                                             maxAtoms,
                                              const int*                                      atomStarts,
                                              const int*                                      hessianStarts,
                                              int                                             numIters,
                                              double                                          gradTol,
                                              bool                                            scaleGrads,
                                              const DistGeom::Energy3DForceContribsDevicePtr& terms,
                                              const DistGeom::BatchedIndices3DDevicePtr&      systemIndices,
                                              double*                                         positions,
                                              double*                                         grad,
                                              double*                                         inverseHessian,
                                              double**                                        scratchBuffers,
                                              double*                                         energyOuts,
                                              int16_t*                                        statuses,
                                              cudaStream_t                                    stream) {
  if (numMols == 0) {
    return cudaSuccess;
  }

  const AsyncDevicePtr<DistGeom::Energy3DForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<DistGeom::BatchedIndices3DDevicePtr>      devSysIdx(systemIndices, stream);

  return dispatchByMaxAtoms<ForceFieldType::ETK, false>(numMols,
                                                        molIds,
                                                        maxAtoms,
                                                        numIters,
                                                        gradTol,
                                                        scaleGrads,
                                                        devTerms.data(),
                                                        devSysIdx.data(),
                                                        atomStarts,
                                                        hessianStarts,
                                                        positions,
                                                        grad,
                                                        inverseHessian,
                                                        scratchBuffers,
                                                        energyOuts,
                                                        statuses,
                                                        stream,
                                                        1.0,
                                                        1.0);
}

cudaError_t launchBfgsMinimizePerMolKernelDG(int                                           numMols,
                                             const int*                                    molIds,
                                             int                                           maxAtoms,
                                             const int*                                    atomStarts,
                                             const int*                                    hessianStarts,
                                             int                                           numIters,
                                             double                                        gradTol,
                                             bool                                          scaleGrads,
                                             const DistGeom::EnergyForceContribsDevicePtr& terms,
                                             const DistGeom::BatchedIndicesDevicePtr&      systemIndices,
                                             double*                                       positions,
                                             double*                                       grad,
                                             double*                                       inverseHessian,
                                             double**                                      scratchBuffers,
                                             double*                                       energyOuts,
                                             double                                        chiralWeight,
                                             double                                        fourthDimWeight,
                                             int16_t*                                      statuses,
                                             cudaStream_t                                  stream) {
  if (numMols == 0) {
    return cudaSuccess;
  }

  const AsyncDevicePtr<DistGeom::EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<DistGeom::BatchedIndicesDevicePtr>      devSysIdx(systemIndices, stream);

  return dispatchByMaxAtoms<ForceFieldType::DG, false>(numMols,
                                                       molIds,
                                                       maxAtoms,
                                                       numIters,
                                                       gradTol,
                                                       scaleGrads,
                                                       devTerms.data(),
                                                       devSysIdx.data(),
                                                       atomStarts,
                                                       hessianStarts,
                                                       positions,
                                                       grad,
                                                       inverseHessian,
                                                       scratchBuffers,
                                                       energyOuts,
                                                       statuses,
                                                       stream,
                                                       chiralWeight,
                                                       fourthDimWeight);
}

}  // namespace nvMolKit