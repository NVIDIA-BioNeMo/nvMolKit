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
#include "src/forcefields/mmff_kernels_device.cuh"
#include "src/minimizer/fire_minimize_permol_kernels.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

namespace {

constexpr int kFirePerMolBlockSize = 128;
constexpr int kDataDim             = 3;

//! Acceleration conversion factor: 1 kcal/mol/Å applied to 1 amu produces 4.184 * 100 Å/ps^2.
//! Must remain identical to the batched FIRE conversion factor.
constexpr double kForceKcalMolPerAng_PerAmu_to_AngPerPs2 = 4.184 * 100.0;

struct FirePerMolKernelParams {
  double dtIncrementFactor;
  double dtDecrementFactor;
  double minDt;
  double maxDt;
  double dMax;
  double alphaStart;
  double alphaDecrementFactor;
  double gradTol;
  int    nMinForIncrease;
};

FirePerMolKernelParams buildKernelParams(const FireOptions& opts, const double gradTol) {
  FirePerMolKernelParams params{};
  params.dtIncrementFactor    = opts.timeStepIncrement;
  params.dtDecrementFactor    = opts.timeStepDecrement;
  params.minDt                = opts.dtInit * opts.dtMinFactor;
  params.maxDt                = opts.dtInit * opts.dtMaxFactor;
  params.dMax                 = opts.dMax;
  params.alphaStart           = opts.alphaInit;
  params.alphaDecrementFactor = opts.alphaDecrement;
  params.gradTol              = gradTol;
  params.nMinForIncrease      = opts.nMinForIncrease;
  return params;
}

__launch_bounds__(kFirePerMolBlockSize)
  __global__ void firePerMolMmffKernel(const int                                 numIters,
                                       const FirePerMolKernelParams              params,
                                       const bool                                takeHalfStepBack,
                                       const bool                                useAbc,
                                       const bool                                useMass,
                                       const MMFF::EnergyForceContribsDevicePtr* terms,
                                       const MMFF::BatchedIndicesDevicePtr*      systemIndices,
                                       const int*                                molIdList,
                                       const int*                                atomStarts,
                                       double*                                   positions,
                                       double*                                   grad,
                                       double*                                   velocities,
                                       double*                                   alphas,
                                       double*                                   dts,
                                       int*                                      nStepsPositive,
                                       const double*                             masses,
                                       double*                                   energyOuts,
                                       uint8_t*                                  statuses) {
  const int molIdx = molIdList[blockIdx.x];
  const int tid    = threadIdx.x;

  if (statuses[molIdx] == 0) {
    return;
  }

  const int atomStart = atomStarts[molIdx];
  const int atomEnd   = atomStarts[molIdx + 1];
  const int numTerms  = (atomEnd - atomStart) * kDataDim;

  double* const       molCoords = positions + atomStart * kDataDim;
  double* const       molGrad   = grad + atomStart * kDataDim;
  double* const       molVel    = velocities + atomStart * kDataDim;
  const double* const massSys   = useMass ? (masses + atomStart) : nullptr;

  using BlockReduce = cub::BlockReduce<double, kFirePerMolBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  __shared__ double sharedDt;
  __shared__ double sharedAlpha;
  __shared__ int    sharedNsteps;
  __shared__ double sharedScalar0;
  __shared__ double sharedScalar1;
  __shared__ bool   sharedConverged;

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
      molGrad[i] = 0.0;
    }
    __syncthreads();
    MMFF::molGrad<kFirePerMolBlockSize, false>(*terms, *systemIndices, molCoords, molGrad, molIdx, tid);
    __syncthreads();

    double power  = 0.0;
    double gradSq = 0.0;
    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      const double fi = molGrad[i];
      if (!isFirstStep) {
        power += molVel[i] * -fi;
      }
      gradSq += fi * fi;
    }
    double powerSum = 0.0;
    if (!isFirstStep) {
      powerSum = BlockReduce(tempStorage).Sum(power);
      __syncthreads();
    }
    const double gradSqSum = BlockReduce(tempStorage).Sum(gradSq);
    if (tid == 0) {
      sharedScalar0 = powerSum;
      sharedScalar1 = gradSqSum;
    }
    __syncthreads();
    const double powerShared  = sharedScalar0;
    const double gradSqShared = sharedScalar1;

    if (tid == 0 && sqrt(gradSqShared) <= params.gradTol) {
      sharedConverged  = true;
      statuses[molIdx] = 0;
    }
    __syncthreads();
    if (sharedConverged) {
      break;
    }

    if (tid == 0 && !isFirstStep) {
      double newDt     = sharedDt;
      double newAlpha  = sharedAlpha;
      int    newNsteps = sharedNsteps;
      if (powerShared >= 0.0) {
        newNsteps = sharedNsteps + 1;
        if (newNsteps > params.nMinForIncrease) {
          newDt    = fmin(sharedDt * params.dtIncrementFactor, params.maxDt);
          newAlpha = sharedAlpha * params.alphaDecrementFactor;
        }
      } else {
        newNsteps = 0;
        newAlpha  = params.alphaStart;
        newDt     = fmax(sharedDt * params.dtDecrementFactor, params.minDt);
      }
      sharedDt     = newDt;
      sharedAlpha  = newAlpha;
      sharedNsteps = newNsteps;
    }
    __syncthreads();

    const bool negative = !isFirstStep && (powerShared < 0.0);
    if (negative) {
      const double dtNow = sharedDt;
      for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
        if (takeHalfStepBack) {
          molCoords[i] -= 0.5 * dtNow * molVel[i];
        }
        molVel[i]  = 0.0;
        molGrad[i] = 0.0;
      }
      __syncthreads();
      MMFF::molGrad<kFirePerMolBlockSize, false>(*terms, *systemIndices, molCoords, molGrad, molIdx, tid);
      __syncthreads();
    }

    const double dt     = sharedDt;
    const double alpha  = sharedAlpha;
    const int    nsteps = sharedNsteps;

    double vSqAccum    = 0.0;
    double gradSqAccum = 0.0;
    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      double accel;
      if (useMass) {
        const double accelMag = -molGrad[i] * kForceKcalMolPerAng_PerAmu_to_AngPerPs2;
        const int    coordIdx = i / kDataDim;
        accel                 = accelMag / massSys[coordIdx];
      } else {
        accel = -molGrad[i];
      }
      const double newV = molVel[i] + dt * accel;
      molVel[i]         = newV;
      vSqAccum += newV * newV;
      gradSqAccum += molGrad[i] * molGrad[i];
    }
    const double vSqReduced = BlockReduce(tempStorage).Sum(vSqAccum);
    if (tid == 0) {
      sharedScalar0 = vSqReduced;
    }
    __syncthreads();
    const double vSqSum        = sharedScalar0;
    const double gradSqReduced = BlockReduce(tempStorage).Sum(gradSqAccum);
    if (tid == 0) {
      sharedScalar0 = gradSqReduced;
    }
    __syncthreads();
    const double gradSqSum2 = sharedScalar0;

    const double mixCoef1 = 1.0 - alpha;
    const double mixCoef2 = (gradSqSum2 > 1e-30) ? (alpha * sqrt(vSqSum) / sqrt(gradSqSum2)) : 0.0;
    double       abcMult  = 1.0;
    if (useAbc) {
      const double oneMinusA = 1.0 - fmax(alpha, 1e-10);
      const double powTerm   = pow(oneMinusA, static_cast<double>(nsteps + 1));
      const double denom     = 1.0 - powTerm;
      abcMult                = (denom > 1e-30) ? (1.0 / denom) : 1.0;
    }

    for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
      const double vMix = mixCoef1 * molVel[i] + mixCoef2 * (-molGrad[i]);
      molVel[i]         = abcMult * vMix;
    }
    __syncthreads();

    double drScale = 1.0;
    if (useAbc) {
      if (params.dMax > 0.0) {
        const double maxV = params.dMax / dt;
        for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
          molVel[i] = fmax(-maxV, fmin(maxV, molVel[i]));
        }
        __syncthreads();
      }
    } else if (params.dMax > 0.0) {
      double drSqAccum = 0.0;
      for (int i = tid; i < numTerms; i += kFirePerMolBlockSize) {
        const double dri = dt * molVel[i];
        drSqAccum += dri * dri;
      }
      const double drSqReduced = BlockReduce(tempStorage).Sum(drSqAccum);
      if (tid == 0) {
        sharedScalar0 = drSqReduced;
      }
      __syncthreads();
      const double drNorm = sqrt(sharedScalar0);
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

  const double finalThreadEnergy =
    MMFF::molEnergy<kFirePerMolBlockSize, false>(*terms, *systemIndices, molCoords, molIdx, tid);
  const double finalEnergy = BlockReduce(tempStorage).Sum(finalThreadEnergy);
  if (tid == 0) {
    energyOuts[molIdx] = finalEnergy;
  }
}

}  // namespace

cudaError_t launchFirePerMolKernel(const int                                 numMols,
                                   const int*                                molIds,
                                   [[maybe_unused]] const int                maxAtoms,
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
  if (numMols == 0) {
    return cudaSuccess;
  }
  if (hasConstraints) {
    return cudaErrorNotSupported;
  }

  const AsyncDevicePtr<MMFF::EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<MMFF::BatchedIndicesDevicePtr>      devSysIdx(systemIndices, stream);
  const FirePerMolKernelParams                             params  = buildKernelParams(fireOptions, gradTol);
  const bool                                               useMass = fireOptions.useMass && masses != nullptr;
  firePerMolMmffKernel<<<numMols, kFirePerMolBlockSize, 0, stream>>>(numIters,
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

}  // namespace nvMolKit
