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

#include <algorithm>
#include <cstdlib>
#include <cub/cub.cuh>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "src/forcefields/batched_forcefield.h"
#include "src/minimizer/fire_minimizer.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {

namespace {

constexpr int kFireBlockSize = 256;

// Acceleration conversion factor:
//   1 kcal/mol/Å applied to 1 amu produces 4.184 * 100 Å/ps^2
// Derivation: 1 kcal/mol = 4184 J/mol -> 6.9477e-21 J/molecule.
//             F = 6.9477e-21 / 1e-10 m  = 6.9477e-11 N.
//             a = F / m = 6.9477e-11 N / 1.66054e-27 kg = 4.184e16 m/s^2.
//             Convert m/s^2 -> Å/ps^2: multiply by 1e10/1e24 = 1e-14.
//             a = 418.4 Å/ps^2.
constexpr double kForceKcalMolPerAng_PerAmu_to_AngPerPs2 = 4.184 * 100.0;

template <typename T> __global__ void setAllKernel(const int numElements, const T value, T* dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = value;
  }
}

template <typename T> void setAll(AsyncDeviceVector<T>& vec, const T& value) {
  const int numElements = static_cast<int>(vec.size());
  if (numElements == 0) {
    return;
  }
  constexpr int blockSize = 128;
  const int     numBlocks = (numElements + blockSize - 1) / blockSize;
  setAllKernel<<<numBlocks, blockSize, 0, vec.stream()>>>(numElements, value, vec.data());
  cudaCheckError(cudaGetLastError());
}

template <typename T>
__device__ __forceinline__ cuda::std::span<T> getSystemSpan(const cuda::std::span<T>         data,
                                                            const cuda::std::span<const int> atomStarts,
                                                            const int                        sysIdx,
                                                            const int                        dataDim) {
  return data.subspan(atomStarts[sysIdx] * dataDim, (atomStarts[sysIdx + 1] - atomStarts[sysIdx]) * dataDim);
}

//! Packed read-only kernel parameters.
struct FireKernelParams {
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

//! \brief Pre-kick stage of FIRE 2.0. One block per system.
//!
//! Reads the gradient at the *current* position, runs the convergence check, runs the
//! dt/alpha/nsteps state machine, and applies the half-step-back-with-velocity-zero
//! when power < 0. Does not touch positions otherwise. After this kernel runs the
//! caller must re-evaluate the gradient at the new positions before invoking
//! ::firePostKickKernel. This split mirrors ASE FIRE2 which evaluates forces twice per
//! iteration (once before the state machine and again after any half-step-back).
__global__ void firePreKickKernel(const cuda::std::span<const int>    atomStarts,
                                  const cuda::std::span<const int>    activeIndices,
                                  const cuda::std::span<double>       x,
                                  const cuda::std::span<double>       v,
                                  const cuda::std::span<const double> f,
                                  const int                           dataDim,
                                  const cuda::std::span<double>       alphas,
                                  const cuda::std::span<double>       dts,
                                  const cuda::std::span<int>          nStepsPositive,
                                  const FireKernelParams              params,
                                  const bool                          takeHalfStepBack,
                                  const bool                          isFirstStep,
                                  uint8_t*                            statuses,
                                  uint8_t*                            convergeReason,
                                  const cuda::std::span<double>       debugPowers) {
  using BlockReduce = cub::BlockReduce<double, kFireBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ double                            sharedScalar0;
  __shared__ double                            sharedScalar1;
  __shared__ double                            sharedDt;
  __shared__ bool                              sharedConverged;

  const int sysIdx = activeIndices[blockIdx.x];
  if (statuses[sysIdx] == 0) {
    return;
  }

  const auto vSys = getSystemSpan(v, atomStarts, sysIdx, dataDim);
  const auto fSys = getSystemSpan(f, atomStarts, sysIdx, dataDim);
  const auto xSys = getSystemSpan(x, atomStarts, sysIdx, dataDim);

  if (threadIdx.x == 0) {
    sharedDt        = dts[sysIdx];
    sharedConverged = false;
  }
  __syncthreads();

  const double dtIn    = sharedDt;
  const double alphaIn = alphas[sysIdx];
  const int    nstepIn = nStepsPositive[sysIdx];

  double power  = 0.0;
  double gradSq = 0.0;
  for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
    const double fi = fSys[i];
    if (!isFirstStep) {
      power += vSys[i] * -fi;
    }
    gradSq += fi * fi;
  }
  double powerSum = 0.0;
  if (!isFirstStep) {
    powerSum = BlockReduce(tempStorage).Sum(power);
    __syncthreads();
  }
  const double gradSqSum = BlockReduce(tempStorage).Sum(gradSq);
  if (threadIdx.x == 0) {
    sharedScalar0 = powerSum;
    sharedScalar1 = gradSqSum;
  }
  __syncthreads();
  const double powerShared  = sharedScalar0;
  const double gradSqShared = sharedScalar1;

  if (threadIdx.x == 0) {
    if (sqrt(gradSqShared) <= params.gradTol) {
      sharedConverged        = true;
      statuses[sysIdx]       = 0;
      convergeReason[sysIdx] = 1;
    }
  }
  __syncthreads();
  if (sharedConverged) {
    return;
  }

  if (threadIdx.x == 0 && !isFirstStep) {
    if (!debugPowers.empty()) {
      debugPowers[sysIdx] = powerShared;
    }

    double newDt     = dtIn;
    double newAlpha  = alphaIn;
    int    newNsteps = nstepIn;

    if (powerShared >= 0.0) {
      newNsteps = nstepIn + 1;
      if (newNsteps > params.nMinForIncrease) {
        newDt    = fmin(dtIn * params.dtIncrementFactor, params.maxDt);
        newAlpha = alphaIn * params.alphaDecrementFactor;
      }
    } else {
      newNsteps = 0;
      newAlpha  = params.alphaStart;
      newDt     = fmax(dtIn * params.dtDecrementFactor, params.minDt);
    }

    sharedDt               = newDt;
    dts[sysIdx]            = newDt;
    alphas[sysIdx]         = newAlpha;
    nStepsPositive[sysIdx] = newNsteps;
  }
  __syncthreads();

  const double dt = sharedDt;

  // Negative-power half-step-back must read the freshly-updated nstepsPositive=0 marker
  // to decide if it fires. Equivalently: it fires iff power < 0 on a non-first step.
  const bool negative = !isFirstStep && (powerShared < 0.0);
  if (negative) {
    for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
      if (takeHalfStepBack) {
        xSys[i] -= 0.5 * dt * vSys[i];
      }
      vSys[i] = 0.0;
    }
    __syncthreads();
  }
}

//! \brief Post-kick stage of FIRE 2.0. One block per system.
//!
//! Reads the *new* gradient at positions produced by ::firePreKickKernel and applies the
//! semi-implicit Euler kick (v += dt*F), the FIRE mixer, and the displacement clip /
//! position update. Per-system dt/alpha/nstepsPositive are read back from the device
//! buffers populated by the pre-kick.
__global__ void firePostKickKernel(const cuda::std::span<const int>    atomStarts,
                                   const cuda::std::span<const int>    activeIndices,
                                   const cuda::std::span<double>       x,
                                   const cuda::std::span<double>       v,
                                   const cuda::std::span<const double> f,
                                   const cuda::std::span<const double> masses,
                                   const int                           dataDim,
                                   const cuda::std::span<const double> alphas,
                                   const cuda::std::span<const double> dts,
                                   const cuda::std::span<const int>    nStepsPositive,
                                   const FireKernelParams              params,
                                   const bool                          useAbc,
                                   const uint8_t*                      statuses) {
  using BlockReduce = cub::BlockReduce<double, kFireBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ double                            sharedScalar0;

  const int sysIdx = activeIndices[blockIdx.x];
  if (statuses[sysIdx] == 0) {
    return;
  }

  const auto vSys = getSystemSpan(v, atomStarts, sysIdx, dataDim);
  const auto fSys = getSystemSpan(f, atomStarts, sysIdx, dataDim);
  const auto xSys = getSystemSpan(x, atomStarts, sysIdx, dataDim);

  cuda::std::span<const double> massSys;
  const bool                    massEnabled = !masses.empty();
  if (massEnabled) {
    const int atomStart = atomStarts[sysIdx];
    const int atomCount = atomStarts[sysIdx + 1] - atomStart;
    massSys             = masses.subspan(atomStart, atomCount);
  }

  const double dt     = dts[sysIdx];
  const double alpha  = alphas[sysIdx];
  const int    nsteps = nStepsPositive[sysIdx];

  // Kick: v += dt * F. With mass weighting (real MD-style integration), F is converted
  // to physical Å/ps^2 via kForceKcalMolPerAng_PerAmu_to_AngPerPs2 and divided by
  // per-atom mass. Without mass weighting we mirror ASE FIRE2: v += dt * F where F is
  // the raw force in the calculator's native units (no implicit unit conversion).
  double vSqAccum    = 0.0;
  double gradSqAccum = 0.0;
  for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
    double accel;
    if (massEnabled) {
      const double accelMag = -fSys[i] * kForceKcalMolPerAng_PerAmu_to_AngPerPs2;
      const int    coordIdx = i / dataDim;
      accel                 = accelMag / massSys[coordIdx];
    } else {
      accel = -fSys[i];
    }
    const double newV = vSys[i] + dt * accel;
    vSys[i]           = newV;
    vSqAccum += newV * newV;
    gradSqAccum += fSys[i] * fSys[i];
  }
  const double vSqReduced = BlockReduce(tempStorage).Sum(vSqAccum);
  if (threadIdx.x == 0) {
    sharedScalar0 = vSqReduced;
  }
  __syncthreads();
  const double vSqSum        = sharedScalar0;
  const double gradSqReduced = BlockReduce(tempStorage).Sum(gradSqAccum);
  if (threadIdx.x == 0) {
    sharedScalar0 = gradSqReduced;
  }
  __syncthreads();
  const double gradSqSum = sharedScalar0;

  // Mixer: v = (1 - alpha) * v + alpha * |v|/|grad| * (-grad). With ABC,
  // multiply by 1 / (1 - (1 - alpha)^(N+1)).
  const double mixCoef1 = 1.0 - alpha;
  const double mixCoef2 = (gradSqSum > 1e-30) ? (alpha * sqrt(vSqSum) / sqrt(gradSqSum)) : 0.0;
  double       abcMult  = 1.0;
  if (useAbc) {
    const double oneMinusA = 1.0 - fmax(alpha, 1e-10);
    const double pow_term  = pow(oneMinusA, static_cast<double>(nsteps + 1));
    const double denom     = 1.0 - pow_term;
    abcMult                = (denom > 1e-30) ? (1.0 / denom) : 1.0;
  }

  for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
    const double vMix = mixCoef1 * vSys[i] + mixCoef2 * (-fSys[i]);
    vSys[i]           = abcMult * vMix;
  }
  __syncthreads();

  // Displacement handling: ABC clips per-component v to ±dMax/dt, non-ABC
  // norm-clips dr (without modifying v) before the position update.
  double drScale = 1.0;
  if (useAbc) {
    if (params.dMax > 0.0) {
      const double maxV = params.dMax / dt;
      for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
        const double clamped = fmax(-maxV, fmin(maxV, vSys[i]));
        vSys[i]              = clamped;
      }
      __syncthreads();
    }
  } else {
    if (params.dMax > 0.0) {
      double drSqAccum = 0.0;
      for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
        const double dri = dt * vSys[i];
        drSqAccum += dri * dri;
      }
      const double drSqReduced = BlockReduce(tempStorage).Sum(drSqAccum);
      if (threadIdx.x == 0) {
        sharedScalar0 = drSqReduced;
      }
      __syncthreads();
      const double drNorm = sqrt(sharedScalar0);
      if (drNorm > params.dMax) {
        drScale = params.dMax / drNorm;
      }
    }
  }

  for (int i = threadIdx.x; i < static_cast<int>(vSys.size()); i += kFireBlockSize) {
    xSys[i] += drScale * dt * vSys[i];
  }
}

//! \brief Energy-plateau stuck detection.
//!
//! Launched with one block per active system; each block reads its own scalar energy and
//! per-system streak state. When the windowed extrema relative spread falls below
//! @p relTol, the streak counter increments; otherwise the window resets to the current
//! sample. Reaching @p streakLimit declares the system converged (status = 0).
__global__ void fireStuckCheckKernel(cuda::std::span<const int>    activeSystemIndices,
                                     cuda::std::span<const double> energies,
                                     cuda::std::span<double>       energyMinStreak,
                                     cuda::std::span<double>       energyMaxStreak,
                                     cuda::std::span<int32_t>      stuckStreak,
                                     uint8_t*                      statuses,
                                     uint8_t*                      convergeReason,
                                     const double                  relTol,
                                     const int                     streakLimit) {
  if (threadIdx.x != 0) {
    return;
  }
  const int sysIdx = activeSystemIndices[blockIdx.x];
  if (statuses[sysIdx] == 0) {
    return;
  }
  const double energy = energies[sysIdx];
  double       newMin = fmin(energyMinStreak[sysIdx], energy);
  double       newMax = fmax(energyMaxStreak[sysIdx], energy);
  const double denom  = fmax(fabs(energy), 1.0);
  if ((newMax - newMin) <= relTol * denom) {
    const int32_t streak = stuckStreak[sysIdx] + 1;
    stuckStreak[sysIdx]  = streak;
    if (streak >= streakLimit) {
      statuses[sysIdx]       = 0;
      convergeReason[sysIdx] = 2;
    }
    energyMinStreak[sysIdx] = newMin;
    energyMaxStreak[sysIdx] = newMax;
  } else {
    stuckStreak[sysIdx]     = 1;
    energyMinStreak[sysIdx] = energy;
    energyMaxStreak[sysIdx] = energy;
  }
}

}  // namespace

FireBatchMinimizer::FireBatchMinimizer(const int          dataDim,
                                       const FireOptions& options,
                                       cudaStream_t       stream,
                                       const bool         debugMode)
    : dataDim_(dataDim),
      fireOptions_(options),
      stream_(stream),
      debugMode_(debugMode) {
  velocities_.setStream(stream_);
  statuses_.setStream(stream_);
  dt_.setStream(stream_);
  alpha_.setStream(stream_);
  allSystemIndices_.setStream(stream_);
  activeSystemIndices_.setStream(stream_);
  numStepsWithPositivePower_.setStream(stream_);
  countUnfinished_.setStream(stream_);
  countTempStorage_.setStream(stream_);
  masses_.setStream(stream_);
  debugPowers_.setStream(stream_);
  energyMinStreak_.setStream(stream_);
  energyMaxStreak_.setStream(stream_);
  stuckStreak_.setStream(stream_);
  convergeReason_.setStream(stream_);
  loopStatusHost_.resize(1);
  loopStatusHost_[0] = 0;
}

void FireBatchMinimizer::setMasses(const std::vector<double>& masses) {
  hostMasses_ = masses;
}

void FireBatchMinimizer::resetContinuationCache() {
  hasInitializedBatch_   = false;
  cachedNumSystems_      = -1;
  cachedTotalAtoms_      = -1;
  cachedActiveThisStage_ = nullptr;
  cachedMasses_          = nullptr;
}

void FireBatchMinimizer::setConvergencePollInterval(const int interval) {
  if (interval < 1) {
    throw std::invalid_argument("FireBatchMinimizer poll interval must be >= 1");
  }
  convergencePollInterval_ = interval;
}

void FireBatchMinimizer::initialize(const std::vector<int>& atomStartsHost,
                                    const double*           masses,
                                    const uint8_t*          activeThisStage) {
  step_                = 0;
  const int totalAtoms = atomStartsHost.back();
  const int numSystems = static_cast<int>(atomStartsHost.size()) - 1;

  const bool isContinuation = hasInitializedBatch_ && cachedNumSystems_ == numSystems &&
                              cachedTotalAtoms_ == totalAtoms && cachedActiveThisStage_ == activeThisStage &&
                              cachedMasses_ == masses;

  numSystems_ = numSystems;

  velocities_.resize(static_cast<size_t>(totalAtoms) * dataDim_);
  velocities_.zero();

  if (fireOptions_.useMass && masses != nullptr) {
    masses_.resize(totalAtoms);
    cudaCheckError(cudaMemcpyAsync(masses_.data(), masses, totalAtoms * sizeof(double), cudaMemcpyDefault, stream_));
  } else if (fireOptions_.useMass && !hostMasses_.empty()) {
    if (hostMasses_.size() != static_cast<size_t>(totalAtoms)) {
      throw std::runtime_error("Stored masses size does not match atom count");
    }
    masses_.setFromVector(hostMasses_);
  } else {
    masses_.resize(0);
  }

  statuses_.resize(numSystems);
  if (!isContinuation) {
    if (activeThisStage != nullptr) {
      cudaCheckError(
        cudaMemcpyAsync(statuses_.data(), activeThisStage, numSystems * sizeof(uint8_t), cudaMemcpyDefault, stream_));
    } else {
      setAll(statuses_, static_cast<uint8_t>(1));
    }
  }

  numStepsWithPositivePower_.resize(numSystems);
  numStepsWithPositivePower_.zero();
  alpha_.resize(numSystems);
  setAll(alpha_, fireOptions_.alphaInit);
  dt_.resize(numSystems);
  setAll(dt_, fireOptions_.dtInit);

  activeSystemIndices_.resize(numSystems);
  allSystemIndices_.resize(numSystems);
  std::vector<int> indicesHost(numSystems);
  std::iota(indicesHost.begin(), indicesHost.end(), 0);
  allSystemIndices_.setFromVector(indicesHost);
  activeSystemIndices_.setFromVector(indicesHost);

  if (fireOptions_.stuckDetectionEnabled) {
    energyMinStreak_.resize(numSystems);
    energyMaxStreak_.resize(numSystems);
    stuckStreak_.resize(numSystems);
    if (!isContinuation) {
      setAll(energyMinStreak_, std::numeric_limits<double>::infinity());
      setAll(energyMaxStreak_, -std::numeric_limits<double>::infinity());
      stuckStreak_.zero();
    }
  } else {
    energyMinStreak_.resize(0);
    energyMaxStreak_.resize(0);
    stuckStreak_.resize(0);
  }
  pollsSinceLastEnergyEval_ = 0;

  convergeReason_.resize(numSystems);
  if (!isContinuation) {
    convergeReason_.zero();
  }

  hasInitializedBatch_   = true;
  cachedNumSystems_      = numSystems;
  cachedTotalAtoms_      = totalAtoms;
  cachedActiveThisStage_ = activeThisStage;
  cachedMasses_          = masses;

  size_t tempStorageBytes = 0;
  cudaCheckError(cub::DeviceSelect::Flagged(nullptr,
                                            tempStorageBytes,
                                            allSystemIndices_.data(),
                                            statuses_.data(),
                                            activeSystemIndices_.data(),
                                            countUnfinished_.data(),
                                            allSystemIndices_.size(),
                                            stream_));
  if (tempStorageBytes > countTempStorage_.size()) {
    countTempStorage_.resize(tempStorageBytes);
  }

  // Filter activeSystemIndices_ to the initial mask so the first kernel launch
  // already skips any caller-marked-inactive systems and uses the right block count.
  compactActiveAsync();
  lastKnownNumUnfinished_ = readbackNumUnfinished();

  if (debugMode_) {
    debugPowers_.resize(numSystems);
    debugPowers_.zero();
    debugOutputs_.assign(numSystems, FireDebugOutput{});
  } else {
    debugPowers_.resize(0);
    debugOutputs_.clear();
  }
}

void FireBatchMinimizer::compactActiveAsync() {
  size_t tempStorageBytes = countTempStorage_.size();
  cudaCheckError(cub::DeviceSelect::Flagged(countTempStorage_.data(),
                                            tempStorageBytes,
                                            allSystemIndices_.data(),
                                            statuses_.data(),
                                            activeSystemIndices_.data(),
                                            countUnfinished_.data(),
                                            allSystemIndices_.size(),
                                            stream_));
}

int FireBatchMinimizer::readbackNumUnfinished() {
  int& host = loopStatusHost_[0];
  countUnfinished_.get(host);
  cudaCheckError(cudaStreamSynchronize(stream_));
  return host;
}

namespace {
FireKernelParams buildKernelParams(const FireOptions& opts, const double gradTol) {
  FireKernelParams params{};
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
}  // namespace

void FireBatchMinimizer::launchPreKick(const double                  gradTol,
                                       const AsyncDeviceVector<int>& atomStarts,
                                       AsyncDeviceVector<double>&    positions,
                                       AsyncDeviceVector<double>&    grad,
                                       const int                     launchBlocks,
                                       const bool                    isFirstStep) {
  if (launchBlocks <= 0) {
    return;
  }

  cuda::std::span<double> debugPowersSpan;
  if (debugMode_ && debugPowers_.size() > 0) {
    debugPowersSpan = cuda::std::span<double>(debugPowers_.data(), debugPowers_.size());
  }

  const FireKernelParams params = buildKernelParams(fireOptions_, gradTol);

  firePreKickKernel<<<launchBlocks, kFireBlockSize, 0, stream_>>>(
    cuda::std::span<const int>(atomStarts.data(), atomStarts.size()),
    cuda::std::span<const int>(activeSystemIndices_.data(), activeSystemIndices_.size()),
    cuda::std::span<double>(positions.data(), positions.size()),
    cuda::std::span<double>(velocities_.data(), velocities_.size()),
    cuda::std::span<const double>(grad.data(), grad.size()),
    dataDim_,
    cuda::std::span<double>(alpha_.data(), alpha_.size()),
    cuda::std::span<double>(dt_.data(), dt_.size()),
    cuda::std::span<int>(numStepsWithPositivePower_.data(), numStepsWithPositivePower_.size()),
    params,
    fireOptions_.takeHalfStepBack,
    isFirstStep,
    statuses_.data(),
    convergeReason_.data(),
    debugPowersSpan);
  cudaCheckError(cudaGetLastError());
}

void FireBatchMinimizer::launchPostKick(const double                  gradTol,
                                        const AsyncDeviceVector<int>& atomStarts,
                                        AsyncDeviceVector<double>&    positions,
                                        AsyncDeviceVector<double>&    grad,
                                        const int                     launchBlocks) {
  if (launchBlocks <= 0) {
    return;
  }

  cuda::std::span<const double> massesSpan;
  if (masses_.size() > 0) {
    massesSpan = cuda::std::span<const double>(masses_.data(), masses_.size());
  }

  const FireKernelParams params = buildKernelParams(fireOptions_, gradTol);

  firePostKickKernel<<<launchBlocks, kFireBlockSize, 0, stream_>>>(
    cuda::std::span<const int>(atomStarts.data(), atomStarts.size()),
    cuda::std::span<const int>(activeSystemIndices_.data(), activeSystemIndices_.size()),
    cuda::std::span<double>(positions.data(), positions.size()),
    cuda::std::span<double>(velocities_.data(), velocities_.size()),
    cuda::std::span<const double>(grad.data(), grad.size()),
    massesSpan,
    dataDim_,
    cuda::std::span<const double>(alpha_.data(), alpha_.size()),
    cuda::std::span<const double>(dt_.data(), dt_.size()),
    cuda::std::span<const int>(numStepsWithPositivePower_.data(), numStepsWithPositivePower_.size()),
    params,
    fireOptions_.abcCorrection,
    statuses_.data());
  cudaCheckError(cudaGetLastError());
}

bool FireBatchMinimizer::step(const double                  gradTol,
                              const AsyncDeviceVector<int>& atomStarts,
                              AsyncDeviceVector<double>&    positions,
                              AsyncDeviceVector<double>&    grad,
                              const GradFunctor&            gFunc) {
  const ScopedNvtxRange stepRange("FireBatchMinimizer::step");
  {
    const ScopedNvtxRange gradRange("FIRE pre-kick gradient");
    grad.zero();
    gFunc();
  }
  const bool isFirstStep = (step_ == 0);
  {
    const ScopedNvtxRange preKickRange("FIRE preKick");
    launchPreKick(gradTol, atomStarts, positions, grad, lastKnownNumUnfinished_, isFirstStep);
  }
  {
    const ScopedNvtxRange gradRange("FIRE post-kick gradient");
    grad.zero();
    gFunc();
  }
  {
    const ScopedNvtxRange postKickRange("FIRE postKick");
    launchPostKick(gradTol, atomStarts, positions, grad, lastKnownNumUnfinished_);
  }
  compactActiveAsync();
  lastKnownNumUnfinished_ = readbackNumUnfinished();
  step_++;
  return lastKnownNumUnfinished_ == 0;
}

namespace {

std::vector<double> debugDump(const AsyncDeviceVector<double>& vec) {
  std::vector<double> result(vec.size());
  if (vec.size() == 0) {
    return result;
  }
  vec.copyToHost(result);
  cudaCheckError(cudaStreamSynchronize(vec.stream()));
  return result;
}

}  // namespace

bool FireBatchMinimizer::minimize(const int                                   numIters,
                                  const double                                gradTol,
                                  const std::vector<int>&                     atomStartsHost,
                                  const AsyncDeviceVector<int>&               atomStarts,
                                  AsyncDeviceVector<double>&                  positions,
                                  AsyncDeviceVector<double>&                  grad,
                                  [[maybe_unused]] AsyncDeviceVector<double>& energyOuts,
                                  [[maybe_unused]] AsyncDeviceVector<double>& energyBuffer,
                                  EnergyFunctor                               eFunc,
                                  const GradFunctor                           gFunc,
                                  const uint8_t*                              activeThisStage) {
  const ScopedNvtxRange minimizeRange("FireBatchMinimizer::minimize (batched)");
  initialize(atomStartsHost, nullptr, activeThisStage);

  for (int iter = 0; iter < numIters; ++iter) {
    if (debugMode_) {
      energyBuffer.zero();
      energyOuts.zero();
      eFunc(positions.data());

      const std::vector<double> energies = debugDump(energyOuts);
      const std::vector<double> powers   = debugDump(debugPowers_);
      const std::vector<double> alphas   = debugDump(alpha_);
      const std::vector<double> dts      = debugDump(dt_);

      for (size_t sysIdx = 0; sysIdx < energies.size(); ++sysIdx) {
        debugOutputs_[sysIdx].energies.push_back(energies[sysIdx]);
        debugOutputs_[sysIdx].powers.push_back(powers[sysIdx]);
        debugOutputs_[sysIdx].alphas.push_back(alphas[sysIdx]);
        debugOutputs_[sysIdx].dt.push_back(dts[sysIdx]);
      }
    }

    if (lastKnownNumUnfinished_ == 0) {
      return true;
    }

    grad.zero();
    gFunc();

    const bool isFirstStep = (step_ == 0);
    launchPreKick(gradTol, atomStarts, positions, grad, lastKnownNumUnfinished_, isFirstStep);
    grad.zero();
    gFunc();
    launchPostKick(gradTol, atomStarts, positions, grad, lastKnownNumUnfinished_);
    compactActiveAsync();

    const bool poll = debugMode_ || ((iter + 1) % convergencePollInterval_ == 0) || (iter + 1 == numIters);
    if (poll) {
      const int activeBeforeStuckCheck = lastKnownNumUnfinished_;
      if (fireOptions_.stuckDetectionEnabled && activeBeforeStuckCheck > 0) {
        ++pollsSinceLastEnergyEval_;
        if (pollsSinceLastEnergyEval_ >= fireOptions_.stuckEvalEveryNPolls) {
          pollsSinceLastEnergyEval_ = 0;
          energyOuts.zero();
          eFunc(nullptr);
          fireStuckCheckKernel<<<activeBeforeStuckCheck, 1, 0, stream_>>>(
            cuda::std::span<const int>(activeSystemIndices_.data(), activeSystemIndices_.size()),
            cuda::std::span<const double>(energyOuts.data(), energyOuts.size()),
            cuda::std::span<double>(energyMinStreak_.data(), energyMinStreak_.size()),
            cuda::std::span<double>(energyMaxStreak_.data(), energyMaxStreak_.size()),
            cuda::std::span<int32_t>(stuckStreak_.data(), stuckStreak_.size()),
            statuses_.data(),
            convergeReason_.data(),
            fireOptions_.stuckEnergyRelTol,
            fireOptions_.stuckStreakLength);
          cudaCheckError(cudaGetLastError());
          compactActiveAsync();
        }
      }
      lastKnownNumUnfinished_ = readbackNumUnfinished();
    }

    step_++;
  }

  if (lastKnownNumUnfinished_ != 0) {
    lastKnownNumUnfinished_ = readbackNumUnfinished();
  }

  static const bool diagVerbose = []() {
    const char* env = std::getenv("NVMOLKIT_FIRE_DIAG");
    return env != nullptr && env[0] != '0';
  }();
  if (diagVerbose) {
    std::vector<uint8_t> reasons(numSystems_);
    convergeReason_.copyToHost(reasons.data(), numSystems_);
    cudaCheckError(cudaStreamSynchronize(stream_));
    int byGrad  = 0;
    int byStuck = 0;
    for (uint8_t reason : reasons) {
      if (reason == 1) {
        ++byGrad;
      } else if (reason == 2) {
        ++byStuck;
      }
    }
    std::cerr << "[FIRE-diag] systems=" << numSystems_ << " iters=" << step_ << " converged_grad=" << byGrad
              << " converged_stuck=" << byStuck << " unfinished=" << lastKnownNumUnfinished_ << '\n';
  }

  return lastKnownNumUnfinished_ == 0;
}

bool FireBatchMinimizer::minimize(const int                  numIters,
                                  const double               gradTol,
                                  BatchedForcefield&         ff,
                                  AsyncDeviceVector<double>& positions,
                                  AsyncDeviceVector<double>& grad,
                                  AsyncDeviceVector<double>& energyOuts,
                                  const uint8_t*             activeSystemMask) {
  const ScopedNvtxRange minimizeRange("FireBatchMinimizer::minimize (BatchedForcefield)");
  const auto&           atomStartsHost = ff.atomStartsHost();

  AsyncDeviceVector<double> energyBuffer;
  energyBuffer.setStream(stream_);
  energyBuffer.resize(energyOuts.size());
  energyBuffer.zero();

  auto eFunc = [&](const double* evalPositions) {
    const double* positionsToEvaluate = evalPositions != nullptr ? evalPositions : positions.data();
    ff.computeEnergy(energyOuts.data(), positionsToEvaluate, activeSystemMask, stream_);
  };
  auto gFunc = [&]() { ff.computeGradients(grad.data(), positions.data(), activeSystemMask, stream_); };

  // The BatchMinimizer interface requires owning storage for device atom offsets.
  AsyncDeviceVector<int> atomStartsDeviceMirror;
  atomStartsDeviceMirror.setStream(stream_);
  atomStartsDeviceMirror.setFromArray(ff.atomStartsHost().data(), atomStartsHost.size());

  return minimize(numIters,
                  gradTol,
                  atomStartsHost,
                  atomStartsDeviceMirror,
                  positions,
                  grad,
                  energyOuts,
                  energyBuffer,
                  eFunc,
                  gFunc,
                  activeSystemMask);
}

FireInternalState FireBatchMinimizer::snapshotInternalState() const {
  FireInternalState snap;
  snap.velocities.resize(velocities_.size());
  if (velocities_.size() > 0) {
    velocities_.copyToHost(snap.velocities);
  }
  snap.dt.resize(dt_.size());
  if (dt_.size() > 0) {
    dt_.copyToHost(snap.dt);
  }
  snap.alpha.resize(alpha_.size());
  if (alpha_.size() > 0) {
    alpha_.copyToHost(snap.alpha);
  }
  snap.nStepsPositive.resize(numStepsWithPositivePower_.size());
  if (numStepsWithPositivePower_.size() > 0) {
    numStepsWithPositivePower_.copyToHost(snap.nStepsPositive);
  }
  snap.statuses.resize(statuses_.size());
  if (statuses_.size() > 0) {
    statuses_.copyToHost(snap.statuses);
  }
  cudaCheckError(cudaStreamSynchronize(stream_));
  return snap;
}

}  // namespace nvMolKit
