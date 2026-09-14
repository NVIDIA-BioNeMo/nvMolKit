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

#include <cub/cub.cuh>

#include "src/forcefields/dist_geom_kernels_device_dispatch.cuh"
#include "src/forcefields/mmff_kernels.h"
#include "src/forcefields/mmff_kernels_device_dispatch.cuh"
#include "src/minimizer/bfgs_minimize_permol_kernels.h"
#include "src/utils/cub_helpers.cuh"
#include "src/utils/device_vector.h"
#include "versions.h"

namespace nvMolKit {

namespace {
constexpr int16_t BLOCK_SIZE           = 128;
constexpr int16_t MAX_LINESEARCH_ITERS = 1000;
constexpr double  FUNCTOL              = 1e-4;
constexpr double  MOVETOL              = 1e-7;
constexpr double  TOLX                 = 4. * 3e-8;

template <typename storageT>
__device__ void setMaxStep(const storageT*                                               pos,
                           const int                                                     numTerms,
                           storageT*                                                     maxStepOutSquared,
                           typename cub::BlockReduce<storageT, BLOCK_SIZE>::TempStorage& tempStorage) {
  storageT sumSquaredPos = 0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    const storageT dx2 = pos[i] * pos[i];
    sumSquaredPos += dx2;
  }
  using BlockReduce = cub::BlockReduce<storageT, BLOCK_SIZE>;

  const storageT squaredSum = BlockReduce(tempStorage).Sum(sumSquaredPos);
  if (threadIdx.x == 0) {
    constexpr storageT maxStepFactorSquared = static_cast<storageT>(100.0 * 100.0);
    const storageT     terms                = static_cast<storageT>(numTerms);
    *maxStepOutSquared                      = maxStepFactorSquared * max(squaredSum, terms * terms);
  }
}

template <typename storageT>
__device__ void lineSearchSetup(const int                                                     numTerms,
                                const storageT*                                               posStart,
                                const storageT*                                               gradStart,
                                const storageT                                                maxStepSquared,
                                storageT*                                                     dirStart,
                                storageT&                                                     slope,
                                storageT&                                                     lambdaMin,
                                typename cub::BlockReduce<storageT, BLOCK_SIZE>::TempStorage& tempStorage) {
  const int idxInSys = threadIdx.x;
  using BlockReduce  = cub::BlockReduce<storageT, BLOCK_SIZE>;
  __shared__ storageT dirSumSquared;

  // ---------------------------------
  //  Scale direction vector if needed
  // ---------------------------------
  storageT sumSquaredLocal = 0;
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    const storageT dx2 = dirStart[i] * dirStart[i];
    sumSquaredLocal += dx2;
  }
  storageT blockSum = BlockReduce(tempStorage).Sum(sumSquaredLocal);
  if (idxInSys == 0) {
    dirSumSquared = blockSum;
  }
  __syncthreads();
  if (dirSumSquared > maxStepSquared) {
    const storageT inverseScaleSquared = dirSumSquared / maxStepSquared;
    const storageT scale               = static_cast<storageT>(1) / sqrt(inverseScaleSquared);
    for (int i = idxInSys; i < numTerms; i += blockDim.x) {
      dirStart[i] *= scale;
    }
  }
  __syncthreads();

  // -------------------------
  // Set slope, check validity
  // -------------------------
  storageT localSum     = 0;
  storageT localGradSum = 0;
  storageT localDirSum  = 0;
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
  storageT localMax_numerator   = 0;
  storageT localMax_denominator = 1;
  // Each thread computes its local maximum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    const storageT temp_numerator   = abs(dirStart[i]);
    const storageT temp_denominator = max(abs(posStart[i]), static_cast<storageT>(1));
    // temp_numerator / temp_denominator > localMax_numerator / localMax_denominator
    // <=>
    // temp_numerator * localMax_denominator > localMax_numerator * temp_denominator
    if (temp_numerator * localMax_denominator > localMax_numerator * temp_denominator) {
      localMax_numerator   = temp_numerator;
      localMax_denominator = temp_denominator;
    }
  }

  const storageT localInvMax =
    localMax_denominator /
    (localMax_numerator > static_cast<storageT>(0) ? localMax_numerator : static_cast<storageT>(1.0e-20));
  // Perform block-wide reduction to find the maximum
  const storageT blockInvMax = BlockReduce(tempStorage).Reduce(localInvMax, cubMin());

  // The first thread in the block writes the result
  if (threadIdx.x == 0) {
    lambdaMin = static_cast<storageT>(MOVETOL) * blockInvMax;
  }
}

template <typename storageT>
__device__ void lineSearchPerturb(const int       numTerms,
                                  const storageT* refPos,
                                  const storageT* dirStart,
                                  const storageT  lambda,
                                  storageT*       scratchPos) {
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    scratchPos[i] = refPos[i] + lambda * dirStart[i];
  }
  __syncthreads();
}

template <typename storageT>
__device__ bool lineSearchPostEnergy(const bool     isFirstIter,
                                     const storageT prevE,
                                     const storageT newE,
                                     const storageT slope,
                                     const storageT lambda,
                                     const storageT lambdaMin,
                                     storageT&      lambda2,
                                     storageT&      eScratch,
                                     storageT&      lambdaOut) {
  bool converged = false;

  if (threadIdx.x == 0) {
    const storageT eDiff = newE - prevE;
    if (lambda < lambdaMin || eDiff <= FUNCTOL * lambda * slope) {
      converged = true;
    } else {
      storageT tmpLambda;
      if (isFirstIter) {
        tmpLambda = -slope / (2.0f * (eDiff - slope));
      } else {
        const storageT rhs1     = eDiff - lambda * slope;
        const storageT rhs2     = eScratch - prevE - lambda2 * slope;
        const storageT rLambda  = static_cast<storageT>(1) / lambda;
        const storageT rLambda2 = static_cast<storageT>(1) / lambda2;
        const storageT rScale   = static_cast<storageT>(1) / (lambda - lambda2);
        const storageT a        = (rhs1 * rLambda * rLambda - rhs2 * rLambda2 * rLambda2) * rScale;
        const storageT b        = (-lambda2 * rhs1 * rLambda * rLambda + lambda * rhs2 * rLambda2 * rLambda2) * rScale;
        if (a == 0.0f) {
          tmpLambda = -slope / (2.0f * b);
        } else {
          const storageT disc = b * b - static_cast<storageT>(3) * a * slope;
          if (disc < 0.0f) {
            tmpLambda = 0.5f * lambda;
          } else {
            const storageT sqrtDisc = sqrt(disc);
            tmpLambda = (b <= static_cast<storageT>(0)) ? (-b + sqrtDisc) / (static_cast<storageT>(3) * a) :
                                                          -slope / (b + sqrtDisc);
          }
        }
        tmpLambda = min(tmpLambda, static_cast<storageT>(0.5) * lambda);
      }
      lambda2   = lambda;
      eScratch  = newE;
      lambdaOut = max(tmpLambda, static_cast<storageT>(0.1) * lambda);
    }
  }
  __syncthreads();
  return converged;
}

template <typename storageT>
__device__ void setDirection(const int                                                     numTerms,
                             const storageT*                                               posFromLineSearch,
                             const storageT*                                               pos,
                             storageT*                                                     xi,
                             storageT*                                                     dGrad,
                             const storageT*                                               grad,
                             bool&                                                         converged,
                             typename cub::BlockReduce<storageT, BLOCK_SIZE>::TempStorage& tempStorage) {
  storageT localMax_numerator   = 0;
  storageT localMax_denominator = 1;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    xi[i]    = posFromLineSearch[i] - pos[i];
    dGrad[i] = grad[i];

    const storageT temp_numerator   = abs(xi[i]);
    const storageT temp_denominator = max(abs(posFromLineSearch[i]), static_cast<storageT>(1));
    // temp_numerator / temp_denominator > localMax_numerator / localMax_denominator
    // <=>
    // temp_numerator * localMax_denominator > localMax_numerator * temp_denominator
    if (temp_numerator * localMax_denominator > localMax_numerator * temp_denominator) {
      localMax_numerator   = temp_numerator;
      localMax_denominator = temp_denominator;
    }
  }

  const storageT localMax = localMax_numerator / localMax_denominator;
  const storageT blockMax = cub::BlockReduce<storageT, BLOCK_SIZE>(tempStorage).Reduce(localMax, cubMax());

  if (threadIdx.x == 0 && blockMax < TOLX) {
    converged = true;
  }
  __syncthreads();
}

template <bool scaleGrads, typename storageT>
__device__ void scaleGrad(const int                                                     numTerms,
                          storageT*                                                     grad,
                          storageT&                                                     gradScale,
                          typename cub::BlockReduce<storageT, BLOCK_SIZE>::TempStorage& tempStorage) {
  // See scaleGradKernel in bfgs_minimize.cu for the RDKit 5b1d04d23 (2025.09) rationale.
  constexpr bool kRdkitHasGradScaleFix =
    RDKIT_VERSION_MAJOR > 2025 || (RDKIT_VERSION_MAJOR == 2025 && RDKIT_VERSION_MINOR >= 9);
  gradScale = scaleGrads ? static_cast<storageT>(0.1) : static_cast<storageT>(1);

  storageT maxGrad = kRdkitHasGradScaleFix ? static_cast<storageT>(0) : static_cast<storageT>(-1e8);
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    if constexpr (scaleGrads) {
      grad[i] *= gradScale;
    }
    const storageT cmp = kRdkitHasGradScaleFix ? abs(grad[i]) : grad[i];
    if (cmp > maxGrad) {
      maxGrad = cmp;
    }
  }

  const storageT blockMax = cub::BlockReduce<storageT, BLOCK_SIZE>(tempStorage).Reduce(maxGrad, cubMax());

  __shared__ storageT distributedMax[1];
  if (threadIdx.x == 0) {
    distributedMax[0] = blockMax;
  }
  __syncthreads();

  maxGrad = distributedMax[0];

  if (scaleGrads && maxGrad > static_cast<storageT>(10)) {
    while (maxGrad * gradScale > static_cast<storageT>(10)) {
      gradScale *= static_cast<storageT>(0.5);
    }
    for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
      grad[i] *= gradScale;
    }
  }
  __syncthreads();
}

template <typename storageT>
__device__ void updateDGrad(const int                                                     numTerms,
                            const storageT                                                gradTol,
                            const storageT                                                energy,
                            const storageT                                                gradScale,
                            const storageT*                                               grad,
                            const storageT*                                               pos,
                            storageT*                                                     dGrad,
                            bool&                                                         converged,
                            typename cub::BlockReduce<storageT, BLOCK_SIZE>::TempStorage& tempStorage) {
  storageT localMax = 0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    dGrad[i]            = grad[i] - dGrad[i];
    const storageT temp = abs(grad[i]) * max(abs(pos[i]), static_cast<storageT>(1));
    if (temp > localMax) {
      localMax = temp;
    }
  }

  storageT blockMax = cub::BlockReduce<storageT, BLOCK_SIZE>(tempStorage).Reduce(localMax, cubMax());

  if (threadIdx.x == 0) {
    // rdkit/rdkit#9298 (merged RDKit 2026.03): use |energy| to avoid clamping the
    // denominator to 1 when energy is negative; match signed behavior on older RDKit.
    constexpr bool kRdkitHasGradDenomFix =
      RDKIT_VERSION_MAJOR > 2026 || (RDKIT_VERSION_MAJOR == 2026 && RDKIT_VERSION_MINOR >= 3);
    const storageT energyMag = kRdkitHasGradDenomFix ? abs(energy) : energy;
    const storageT term      = max(energyMag * gradScale, static_cast<storageT>(1));
    blockMax /= term;
    if (blockMax < gradTol) {
      converged = true;
    }
  }
  __syncthreads();
}

template <typename storageT>
__device__ void updateInverseHessian(const int                                                     numTerms,
                                     storageT*                                                     invHessian,
                                     storageT*                                                     dGrad,
                                     storageT*                                                     xi,
                                     storageT*                                                     hessDGrad,
                                     storageT*                                                     grad,
                                     typename cub::BlockReduce<storageT, BLOCK_SIZE>::TempStorage& tempStorage) {
  using BlockReduce = cub::BlockReduce<storageT, BLOCK_SIZE>;

  // Compute hessDGrad = invHessian * dGrad
  for (int row = threadIdx.x; row < numTerms; row += blockDim.x) {
    storageT dotProduct = 0;
    for (int col = 0; col < numTerms; col++) {
      dotProduct += invHessian[col * numTerms + row] * dGrad[col];
    }
    hessDGrad[row] = dotProduct;
  }
  __syncthreads();

  // Compute BFGS sums
  __shared__ storageT fac, fae, fad, sumDGrad, sumXi;
  __shared__ bool     needUpdate;

  storageT sumFac = 0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumFac += dGrad[i] * xi[i];
  }
  const storageT facReduced = BlockReduce(tempStorage).Sum(sumFac);
  if (threadIdx.x == 0)
    fac = facReduced;
  __syncthreads();

  storageT sumFae = 0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumFae += dGrad[i] * hessDGrad[i];
  }
  const storageT faeReduced = BlockReduce(tempStorage).Sum(sumFae);
  if (threadIdx.x == 0)
    fae = faeReduced;
  __syncthreads();

  storageT sumDGradSq = 0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumDGradSq += dGrad[i] * dGrad[i];
  }
  const storageT sumDGradReduced = BlockReduce(tempStorage).Sum(sumDGradSq);
  if (threadIdx.x == 0)
    sumDGrad = sumDGradReduced;
  __syncthreads();

  storageT sumXiSq = 0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumXiSq += xi[i] * xi[i];
  }
  const storageT sumXiReduced = BlockReduce(tempStorage).Sum(sumXiSq);
  if (threadIdx.x == 0)
    sumXi = sumXiReduced;
  __syncthreads();

  if (threadIdx.x == 0) {
    constexpr storageT EPS = static_cast<storageT>(3e-8);
    needUpdate             = (fac > 0) && ((fac * fac) > (EPS * sumDGrad * sumXi));

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
      const storageT pxi  = fac * xi[row];
      const storageT hdgi = fad * hessDGrad[row];
      const storageT dgi  = fae * dGrad[row];

      for (int col = 0; col < numTerms; col++) {
        const storageT pxj    = xi[col];
        const storageT hdgj   = hessDGrad[col];
        const storageT dgj    = dGrad[col];
        const storageT update = pxi * pxj - hdgi * hdgj + dgi * dgj;
        invHessian[col * numTerms + row] += update;
      }
    }
    __syncthreads();
  }

  // Update xi = -invHessian * grad
  for (int row = threadIdx.x; row < numTerms; row += blockDim.x) {
    storageT dotProduct = 0;
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
          typename IndicesType,
          typename storageT>
__launch_bounds__(BLOCK_SIZE) __global__ void bfgsMinimizeKernel(const int               numIters,
                                                                 const double            gradTol,
                                                                 const bool              scaleGrads,
                                                                 const TermsType*        terms,
                                                                 const IndicesType*      systemIndices,
                                                                 const int*              molIdList,
                                                                 const int*              atomStarts,
                                                                 const int*              hessianStarts,
                                                                 storageT*               positions,
                                                                 storageT*               grad,
                                                                 storageT*               inverseHessian,
                                                                 storageT**              scratchBuffers,
                                                                 storageT*               energyOuts,
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
  storageT* localPos;
  storageT* localGrad;
  storageT* localDir;
  storageT* scratchPos;
  storageT* dGrad;
  storageT* oldPos;

  if constexpr (UseSharedMem) {
    // Shared memory for small molecules (≤64 atoms)
    __shared__ storageT sharedLocalPos[maxTerms];
    __shared__ storageT sharedLocalGrad[maxTerms];
    __shared__ storageT sharedLocalDir[maxTerms];
    __shared__ storageT sharedScratchPos[maxTerms];
    __shared__ storageT sharedDGrad[maxTerms];

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
  __shared__ storageT maxStep;
  __shared__ storageT prevE;
  __shared__ storageT currE;
  __shared__ storageT slope;
  __shared__ storageT lambda;
  __shared__ storageT lambdaMin;
  __shared__ storageT lambda2;
  __shared__ storageT eScratch;
  __shared__ storageT gradScale;
  __shared__ bool     converged;
  __shared__ bool     lineSearchConverged;

  // Inverse Hessian in global memory (O(n^2), too large for shared)
  // Indexed by hessianStarts which stores cumulative (numTerms * numTerms) offsets
  storageT* invHessian = inverseHessian + hessianStarts[molIdx];

  // Initialize positions from global memory
  storageT* globalPos = positions + atomStart * dataDim;
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
    invHessian[i] = static_cast<storageT>(0);
  }
  __syncthreads();
  for (int i = tid; i < numTerms; i += blockDim.x) {
    invHessian[i * numTerms + i] = static_cast<storageT>(1);
  }

  if (tid == 0) {
    converged = false;
  }
  __syncthreads();

  // Shared temp storage for all BlockReduce operations
  using BlockReduce = cub::BlockReduce<storageT, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  // Compute initial energy
  storageT threadEnergy;
  if constexpr (FFType == ForceFieldType::MMFF) {
    threadEnergy = MMFF::molEnergy<BLOCK_SIZE, HasConstraints>(*terms, *systemIndices, localPos, molIdx, tid);
  } else if constexpr (FFType == ForceFieldType::ETK) {
    threadEnergy = DistGeom::molEnergyETK(*terms, *systemIndices, localPos, molIdx, tid);
  } else {  // DG
    threadEnergy =
      DistGeom::molEnergyDG<dataDim>(*terms, *systemIndices, localPos, molIdx, chiralWeight, fourthDimWeight, tid);
  }
  const storageT blockEnergy = BlockReduce(tempStorage).Sum(threadEnergy);

  if (tid == 0) {
    prevE              = blockEnergy;
    energyOuts[molIdx] = blockEnergy;
  }
  __syncthreads();

  for (int i = tid; i < numTerms; i += blockDim.x) {
    localGrad[i] = static_cast<storageT>(0);
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
      lambda              = static_cast<storageT>(1);
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
      storageT lsThreadEnergy;
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
      const storageT lsBlockEnergy = BlockReduce(tempStorage).Sum(lsThreadEnergy);

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
      localGrad[i] = static_cast<storageT>(0);
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
    updateDGrad(numTerms,
                static_cast<storageT>(gradTol),
                currE,
                gradScale,
                localGrad,
                localPos,
                dGrad,
                converged,
                tempStorage);
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
          typename IndicesType,
          typename storageT>
cudaError_t launchKernelForSize(int                numMols,
                                const int*         molIdList,
                                int                numIters,
                                double             gradTol,
                                bool               scaleGrads,
                                const TermsType*   devTerms,
                                const IndicesType* devSysIdx,
                                const int*         atomStarts,
                                const int*         hessianStarts,
                                storageT*          positions,
                                storageT*          grad,
                                storageT*          inverseHessian,
                                storageT**         scratchBuffers,
                                storageT*          energyOuts,
                                int16_t*           statuses,
                                cudaStream_t       stream,
                                double             chiralWeight,
                                double             fourthDimWeight) {
  if (numMols == 0) {
    return cudaSuccess;
  }

  bfgsMinimizeKernel<MaxAtoms, UseSharedMem, FFType, HasConstraints, TermsType, IndicesType, storageT>
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

template <ForceFieldType FFType, bool HasConstraints, typename TermsType, typename IndicesType, typename storageT>
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
                               storageT*          positions,
                               storageT*          grad,
                               storageT*          inverseHessian,
                               storageT**         scratchBuffers,
                               storageT*          energyOuts,
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

template <typename Terms, typename storageT>
cudaError_t launchBfgsMinimizePerMolKernelImpl(int                                  numMols,
                                               const int*                           molIds,
                                               int                                  maxAtoms,
                                               const int*                           atomStarts,
                                               const int*                           hessianStarts,
                                               int                                  numIters,
                                               double                               gradTol,
                                               bool                                 scaleGrads,
                                               const Terms&                         terms,
                                               const MMFF::BatchedIndicesDevicePtr& systemIndices,
                                               storageT*                            positions,
                                               storageT*                            grad,
                                               storageT*                            inverseHessian,
                                               storageT**                           scratchBuffers,
                                               storageT*                            energyOuts,
                                               bool                                 hasConstraints,
                                               int16_t*                             statuses,
                                               cudaStream_t                         stream) {
  if (numMols == 0) {
    return cudaSuccess;
  }

  const AsyncDevicePtr<Terms>                         devTerms(terms, stream);
  const AsyncDevicePtr<MMFF::BatchedIndicesDevicePtr> devSysIdx(systemIndices, stream);

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

#define NVMOLKIT_DEFINE_MMFF_PER_MOL_LAUNCHER(TERMS_TYPE, STORAGE_TYPE)                           \
  cudaError_t launchBfgsMinimizePerMolKernel(int                                  numMols,        \
                                             const int*                           molIds,         \
                                             int                                  maxAtoms,       \
                                             const int*                           atomStarts,     \
                                             const int*                           hessianStarts,  \
                                             int                                  numIters,       \
                                             double                               gradTol,        \
                                             bool                                 scaleGrads,     \
                                             const TERMS_TYPE&                    terms,          \
                                             const MMFF::BatchedIndicesDevicePtr& systemIndices,  \
                                             STORAGE_TYPE*                        positions,      \
                                             STORAGE_TYPE*                        grad,           \
                                             STORAGE_TYPE*                        inverseHessian, \
                                             STORAGE_TYPE**                       scratchBuffers, \
                                             STORAGE_TYPE*                        energyOuts,     \
                                             bool                                 hasConstraints, \
                                             int16_t*                             statuses,       \
                                             cudaStream_t                         stream) {                               \
    return launchBfgsMinimizePerMolKernelImpl(numMols,                                            \
                                              molIds,                                             \
                                              maxAtoms,                                           \
                                              atomStarts,                                         \
                                              hessianStarts,                                      \
                                              numIters,                                           \
                                              gradTol,                                            \
                                              scaleGrads,                                         \
                                              terms,                                              \
                                              systemIndices,                                      \
                                              positions,                                          \
                                              grad,                                               \
                                              inverseHessian,                                     \
                                              scratchBuffers,                                     \
                                              energyOuts,                                         \
                                              hasConstraints,                                     \
                                              statuses,                                           \
                                              stream);                                            \
  }

NVMOLKIT_DEFINE_MMFF_PER_MOL_LAUNCHER(MMFF::EnergyForceContribsDevicePtr, double)
NVMOLKIT_DEFINE_MMFF_PER_MOL_LAUNCHER(MMFF::EnergyForceContribsDevicePtrSingle, float)

#undef NVMOLKIT_DEFINE_MMFF_PER_MOL_LAUNCHER
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
