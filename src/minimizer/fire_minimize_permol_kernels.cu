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

#include <cub/cub.cuh>

#include "src/forcefields/mmff_kernels.h"
#include "src/forcefields/mmff_kernels_device_dispatch.cuh"
#include "src/minimizer/fire_minimize_permol_kernels.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

namespace {

constexpr int kFirePerMolBlockSize = 128;
constexpr int kDataDim             = 3;

//! Acceleration conversion factor: 1 kcal/mol/Å applied to 1 amu produces 4.184 * 100 Å/ps^2.
//! Must remain identical to the batched FIRE conversion factor.
constexpr double kForceKcalMolPerAng_PerAmu_to_AngPerPs2 = 4.184 * 100.0;

template <typename storageT> struct FirePerMolKernelParams {
  storageT dtIncrementFactor;
  storageT dtDecrementFactor;
  storageT minDt;
  storageT maxDt;
  storageT dMax;
  storageT alphaStart;
  storageT alphaDecrementFactor;
  storageT gradTol;
  int      nMinForIncrease;
};

template <typename storageT>
FirePerMolKernelParams<storageT> buildKernelParams(const FireOptions& opts, const double gradTol) {
  FirePerMolKernelParams<storageT> params{};
  params.dtIncrementFactor    = static_cast<storageT>(opts.timeStepIncrement);
  params.dtDecrementFactor    = static_cast<storageT>(opts.timeStepDecrement);
  params.minDt                = static_cast<storageT>(opts.dtInit * opts.dtMinFactor);
  params.maxDt                = static_cast<storageT>(opts.dtInit * opts.dtMaxFactor);
  params.dMax                 = static_cast<storageT>(opts.dMax);
  params.alphaStart           = static_cast<storageT>(opts.alphaInit);
  params.alphaDecrementFactor = static_cast<storageT>(opts.alphaDecrement);
  params.gradTol              = static_cast<storageT>(gradTol);
  params.nMinForIncrease      = opts.nMinForIncrease;
  return params;
}

template <typename Terms, typename storageT>
__launch_bounds__(kFirePerMolBlockSize)
  __global__ void firePerMolMmffKernel(const int                              numIters,
                                       const FirePerMolKernelParams<storageT> params,
                                       const bool                             takeHalfStepBack,
                                       const bool                             useAbc,
                                       const bool                             useMass,
                                       const Terms*                           terms,
                                       const MMFF::BatchedIndicesDevicePtr*   systemIndices,
                                       const int*                             molIdList,
                                       const int*                             atomStarts,
                                       storageT*                              positions,
                                       storageT*                              grad,
                                       storageT*                              velocities,
                                       storageT*                              alphas,
                                       storageT*                              dts,
                                       int*                                   nStepsPositive,
                                       const storageT*                        masses,
                                       storageT*                              energyOuts,
                                       uint8_t*                               statuses) {
  const int molIdx = molIdList[blockIdx.x];
  const int tid    = threadIdx.x;

  if (statuses[molIdx] == 0) {
    return;
  }

  const int atomStart = atomStarts[molIdx];
  const int atomEnd   = atomStarts[molIdx + 1];
  const int numTerms  = (atomEnd - atomStart) * kDataDim;

  storageT* const       molCoords = positions + atomStart * kDataDim;
  storageT* const       molGrad   = grad + atomStart * kDataDim;
  storageT* const       molVel    = velocities + atomStart * kDataDim;
  const storageT* const massSys   = useMass ? (masses + atomStart) : nullptr;

  using BlockReduce = cub::BlockReduce<storageT, kFirePerMolBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  __shared__ storageT sharedDt;
  __shared__ storageT sharedAlpha;
  __shared__ int      sharedNsteps;
  __shared__ storageT sharedScalar0;
  __shared__ storageT sharedScalar1;
  __shared__ bool     sharedConverged;

  if (tid == 0) {
    sharedDt        = dts[molIdx];
    sharedAlpha     = alphas[molIdx];
    sharedNsteps    = nStepsPositive[molIdx];
    sharedConverged = false;
  }
  __syncthreads();

  for (int iter = 0; iter < numIters; ++iter) {
    const bool isFirstStep = (iter == 0);

    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      molGrad[i] = storageT{0};
    }
    __syncthreads();
    MMFF::molGrad<kFirePerMolBlockSize, false>(*terms, *systemIndices, molCoords, molGrad, molIdx, tid);
    __syncthreads();

    storageT power  = 0;
    storageT gradSq = 0;
    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      const double fi = molGrad[i];
      if (!isFirstStep) {
        power += molVel[i] * -fi;
      }
      gradSq += fi * fi;
    }
    storageT powerSum = 0;
    if (!isFirstStep) {
      powerSum = BlockReduce(tempStorage).Sum(power);
      __syncthreads();
    }
    const storageT gradSqSum = BlockReduce(tempStorage).Sum(gradSq);
    if (tid == 0) {
      sharedScalar0 = powerSum;
      sharedScalar1 = gradSqSum;
    }
    __syncthreads();
    const storageT powerShared  = sharedScalar0;
    const storageT gradSqShared = sharedScalar1;

    if (tid == 0 && sqrt(gradSqShared) <= params.gradTol) {
      sharedConverged  = true;
      statuses[molIdx] = 0;
    }
    __syncthreads();
    if (sharedConverged) {
      break;
    }

    if (tid == 0 && !isFirstStep) {
      storageT  newDt     = sharedDt;
      storageT  newAlpha  = sharedAlpha;
      const int newNsteps = powerShared >= storageT{0} ? sharedNsteps + 1 : 0;
      if (powerShared >= storageT{0}) {
        if (newNsteps > params.nMinForIncrease) {
          newDt    = fmin(sharedDt * params.dtIncrementFactor, params.maxDt);
          newAlpha = sharedAlpha * params.alphaDecrementFactor;
        }
      } else {
        newAlpha = params.alphaStart;
        newDt    = fmax(sharedDt * params.dtDecrementFactor, params.minDt);
      }
      sharedDt     = newDt;
      sharedAlpha  = newAlpha;
      sharedNsteps = newNsteps;
    }
    __syncthreads();

    const bool negative = !isFirstStep && (powerShared < storageT{0});
    if (negative) {
      const storageT dtNow = sharedDt;
      for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
        if (takeHalfStepBack) {
          molCoords[i] -= storageT{0.5} * dtNow * molVel[i];
        }
        molVel[i]  = storageT{0};
        molGrad[i] = storageT{0};
      }
      __syncthreads();
      MMFF::molGrad<kFirePerMolBlockSize, false>(*terms, *systemIndices, molCoords, molGrad, molIdx, tid);
      __syncthreads();
    }

    const storageT dt     = sharedDt;
    const storageT alpha  = sharedAlpha;
    const int      nsteps = sharedNsteps;

    storageT vSqAccum    = 0;
    storageT gradSqAccum = 0;
    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      storageT accel;
      if (useMass) {
        const storageT accelMag = -molGrad[i] * static_cast<storageT>(kForceKcalMolPerAng_PerAmu_to_AngPerPs2);
        const int      coordIdx = i / kDataDim;
        accel                   = accelMag / massSys[coordIdx];
      } else {
        accel = -molGrad[i];
      }
      const storageT newV = molVel[i] + dt * accel;
      molVel[i]           = newV;
      vSqAccum += newV * newV;
      gradSqAccum += molGrad[i] * molGrad[i];
    }
    const storageT vSqReduced = BlockReduce(tempStorage).Sum(vSqAccum);
    if (tid == 0) {
      sharedScalar0 = vSqReduced;
    }
    __syncthreads();
    const storageT vSqSum        = sharedScalar0;
    const storageT gradSqReduced = BlockReduce(tempStorage).Sum(gradSqAccum);
    if (tid == 0) {
      sharedScalar0 = gradSqReduced;
    }
    __syncthreads();
    const storageT gradSqSum2 = sharedScalar0;

    const storageT mixCoef1 = storageT{1} - alpha;
    const storageT mixCoef2 =
      (gradSqSum2 > static_cast<storageT>(1e-30)) ? (alpha * sqrt(vSqSum) / sqrt(gradSqSum2)) : storageT{0};
    storageT abcMult = 1;
    if (useAbc) {
      const storageT oneMinusA = storageT{1} - max(alpha, static_cast<storageT>(1e-10));
      const storageT powTerm   = pow(oneMinusA, static_cast<storageT>(nsteps + 1));
      const storageT denom     = storageT{1} - powTerm;
      abcMult                  = (denom > static_cast<storageT>(1e-30)) ? (storageT{1} / denom) : storageT{1};
    }

    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      const storageT vMix = mixCoef1 * molVel[i] + mixCoef2 * (-molGrad[i]);
      molVel[i]           = abcMult * vMix;
    }
    __syncthreads();

    storageT drScale = 1;
    if (useAbc) {
      if (params.dMax > storageT{0}) {
        const storageT maxV = params.dMax / dt;
        for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
          molVel[i] = max(-maxV, min(maxV, molVel[i]));
        }
        __syncthreads();
      }
    } else if (params.dMax > storageT{0}) {
      storageT drSqAccum = 0;
      for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
        const storageT dri = dt * molVel[i];
        drSqAccum += dri * dri;
      }
      const storageT drSqReduced = BlockReduce(tempStorage).Sum(drSqAccum);
      if (tid == 0) {
        sharedScalar0 = drSqReduced;
      }
      __syncthreads();
      const storageT drNorm = sqrt(sharedScalar0);
      if (drNorm > params.dMax) {
        drScale = params.dMax / drNorm;
      }
    }

    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      molCoords[i] += drScale * dt * molVel[i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    dts[molIdx]            = sharedDt;
    alphas[molIdx]         = sharedAlpha;
    nStepsPositive[molIdx] = sharedNsteps;
  }
  __syncthreads();

  const storageT finalThreadEnergy =
    MMFF::molEnergy<kFirePerMolBlockSize, false>(*terms, *systemIndices, molCoords, molIdx, tid);
  const storageT finalEnergy = BlockReduce(tempStorage).Sum(finalThreadEnergy);
  if (tid == 0) {
    energyOuts[molIdx] = finalEnergy;
  }
}

}  // namespace

template <typename Terms, typename storageT>
cudaError_t launchFirePerMolKernelImpl(const int                            numMols,
                                       const int*                           molIds,
                                       [[maybe_unused]] const int           maxAtoms,
                                       const int*                           atomStarts,
                                       const FireOptions&                   fireOptions,
                                       const int                            numIters,
                                       const double                         gradTol,
                                       const Terms&                         terms,
                                       const MMFF::BatchedIndicesDevicePtr& systemIndices,
                                       const bool                           hasConstraints,
                                       storageT*                            positions,
                                       storageT*                            grad,
                                       storageT*                            velocities,
                                       storageT*                            alphas,
                                       storageT*                            dts,
                                       int*                                 nStepsPositive,
                                       const storageT*                      masses,
                                       storageT*                            energyOuts,
                                       uint8_t*                             statuses,
                                       const cudaStream_t                   stream) {
  if (numMols == 0) {
    return cudaSuccess;
  }
  if (hasConstraints) {
    return cudaErrorNotSupported;
  }

  const AsyncDevicePtr<Terms>                         devTerms(terms, stream);
  const AsyncDevicePtr<MMFF::BatchedIndicesDevicePtr> devSysIdx(systemIndices, stream);
  const FirePerMolKernelParams<storageT>              params  = buildKernelParams<storageT>(fireOptions, gradTol);
  const bool                                          useMass = fireOptions.useMass && masses != nullptr;
  firePerMolMmffKernel<Terms, storageT><<<numMols, kFirePerMolBlockSize, 0, stream>>>(numIters,
                                                                                      params,
                                                                                      fireOptions.takeHalfStepBack,
                                                                                      fireOptions.abcCorrection,
                                                                                      useMass,
                                                                                      devTerms.data(),
                                                                                      devSysIdx.data(),
                                                                                      molIds,
                                                                                      atomStarts,
                                                                                      positions,
                                                                                      grad,
                                                                                      velocities,
                                                                                      alphas,
                                                                                      dts,
                                                                                      nStepsPositive,
                                                                                      masses,
                                                                                      energyOuts,
                                                                                      statuses);
  return cudaGetLastError();
}

cudaError_t launchFirePerMolKernel(const int                                 numMols,
                                   const int*                                molIds,
                                   const int                                 maxAtoms,
                                   const int*                                atomStarts,
                                   const FireOptions&                        fireOptions,
                                   const int                                 numIters,
                                   const double                              gradTol,
                                   const MMFF::EnergyForceContribsDevicePtr& terms,
                                   const MMFF::BatchedIndicesDevicePtr&      systemIndices,
                                   const bool                                hasConstraints,
                                   double*                                   positions,
                                   double*                                   grad,
                                   double*                                   velocities,
                                   double*                                   alphas,
                                   double*                                   dts,
                                   int*                                      nStepsPositive,
                                   const double*                             masses,
                                   double*                                   energyOuts,
                                   uint8_t*                                  statuses,
                                   const cudaStream_t                        stream) {
  return launchFirePerMolKernelImpl(numMols,
                                    molIds,
                                    maxAtoms,
                                    atomStarts,
                                    fireOptions,
                                    numIters,
                                    gradTol,
                                    terms,
                                    systemIndices,
                                    hasConstraints,
                                    positions,
                                    grad,
                                    velocities,
                                    alphas,
                                    dts,
                                    nStepsPositive,
                                    masses,
                                    energyOuts,
                                    statuses,
                                    stream);
}

cudaError_t launchFirePerMolKernel(const int                                       numMols,
                                   const int*                                      molIds,
                                   const int                                       maxAtoms,
                                   const int*                                      atomStarts,
                                   const FireOptions&                              fireOptions,
                                   const int                                       numIters,
                                   const double                                    gradTol,
                                   const MMFF::EnergyForceContribsDevicePtrSingle& terms,
                                   const MMFF::BatchedIndicesDevicePtr&            systemIndices,
                                   const bool                                      hasConstraints,
                                   float*                                          positions,
                                   float*                                          grad,
                                   float*                                          velocities,
                                   float*                                          alphas,
                                   float*                                          dts,
                                   int*                                            nStepsPositive,
                                   const float*                                    masses,
                                   float*                                          energyOuts,
                                   uint8_t*                                        statuses,
                                   const cudaStream_t                              stream) {
  return launchFirePerMolKernelImpl(numMols,
                                    molIds,
                                    maxAtoms,
                                    atomStarts,
                                    fireOptions,
                                    numIters,
                                    gradTol,
                                    terms,
                                    systemIndices,
                                    hasConstraints,
                                    positions,
                                    grad,
                                    velocities,
                                    alphas,
                                    dts,
                                    nStepsPositive,
                                    masses,
                                    energyOuts,
                                    statuses,
                                    stream);
}

}  // namespace nvMolKit
