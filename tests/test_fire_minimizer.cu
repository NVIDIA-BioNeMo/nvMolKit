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

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "src/minimizer/fire_minimizer.h"
#include "src/utils/cuda_error_check.h"

using nvMolKit::checkReturnCode;

namespace {

constexpr int    kDim                                          = 3;
constexpr double kForceKcalMolPerAng_PerAmu_to_AngPerPs2_Local = 4.184 * 100.0;
constexpr double kReferenceTrajectoryTol                       = 1e-3;

//! Reference single-system FIRE 2.0 step that mirrors the device kernel
//! algorithm bit-for-bit (apart from FP reduction order). Used to validate
//! batched device behavior step-by-step.
struct ReferenceConfig {
  double dtInit            = 0.001;
  double dtMinFactor       = 0.002;
  double dtMaxFactor       = 10.0;
  double dMax              = 0.0;
  double timeStepIncrement = 1.1;
  double timeStepDecrement = 0.5;
  int    nMinForIncrease   = 20;
  double alphaInit         = 0.25;
  double alphaDecrement    = 0.99;
  double gradTol           = 1e-4;
  bool   takeHalfStepBack  = true;
  bool   abcCorrection     = false;
  bool   useMass           = false;
  int    dataDim           = kDim;
};

struct ReferenceSystem {
  std::vector<double> positions;
  std::vector<double> velocities;
  std::vector<double> masses;
  double              dt        = 0.0;
  double              alpha     = 0.0;
  int                 nstep     = 0;
  bool                converged = false;
};

void initReferenceSystem(ReferenceSystem&           sys,
                         const std::vector<double>& startingPositions,
                         const std::vector<double>& masses,
                         const ReferenceConfig&     cfg) {
  sys.positions = startingPositions;
  sys.velocities.assign(startingPositions.size(), 0.0);
  sys.masses    = masses;
  sys.dt        = cfg.dtInit;
  sys.alpha     = cfg.alphaInit;
  sys.nstep     = 0;
  sys.converged = false;
}

void referenceStep(ReferenceSystem&           sys,
                   const std::vector<double>& grad,
                   const ReferenceConfig&     cfg,
                   const bool                 isFirstStep) {
  if (sys.converged) {
    return;
  }
  double gradSq = 0.0;
  for (const double gi : grad) {
    gradSq += gi * gi;
  }
  if (std::sqrt(gradSq) <= cfg.gradTol) {
    sys.converged = true;
    return;
  }

  bool   negative = false;
  double power    = 0.0;
  if (!isFirstStep) {
    for (size_t i = 0; i < sys.velocities.size(); ++i) {
      power += sys.velocities[i] * -grad[i];
    }
    if (power >= 0.0) {
      sys.nstep += 1;
      if (sys.nstep > cfg.nMinForIncrease) {
        sys.dt    = std::min(sys.dt * cfg.timeStepIncrement, cfg.dtInit * cfg.dtMaxFactor);
        sys.alpha = sys.alpha * cfg.alphaDecrement;
      }
    } else {
      negative  = true;
      sys.nstep = 0;
      sys.alpha = cfg.alphaInit;
      sys.dt    = std::max(sys.dt * cfg.timeStepDecrement, cfg.dtInit * cfg.dtMinFactor);
    }
  }

  if (negative) {
    for (size_t i = 0; i < sys.velocities.size(); ++i) {
      if (cfg.takeHalfStepBack) {
        sys.positions[i] -= 0.5 * sys.dt * sys.velocities[i];
      }
      sys.velocities[i] = 0.0;
    }
  }

  for (size_t i = 0; i < sys.velocities.size(); ++i) {
    double accel;
    if (cfg.useMass && !sys.masses.empty()) {
      const double accelMag = -grad[i] * kForceKcalMolPerAng_PerAmu_to_AngPerPs2_Local;
      const int    atomIdx  = static_cast<int>(i / cfg.dataDim);
      accel                 = accelMag / sys.masses[atomIdx];
    } else {
      accel = -grad[i];
    }
    sys.velocities[i] += sys.dt * accel;
  }

  double vSq = 0.0;
  for (const double vi : sys.velocities) {
    vSq += vi * vi;
  }
  const double mixCoef1 = 1.0 - sys.alpha;
  const double mixCoef2 = (gradSq > 1e-30) ? (sys.alpha * std::sqrt(vSq) / std::sqrt(gradSq)) : 0.0;
  double       abcMult  = 1.0;
  if (cfg.abcCorrection) {
    const double oneMinusA = 1.0 - std::max(sys.alpha, 1e-10);
    const double powTerm   = std::pow(oneMinusA, static_cast<double>(sys.nstep + 1));
    const double denom     = 1.0 - powTerm;
    abcMult                = (denom > 1e-30) ? (1.0 / denom) : 1.0;
  }

  for (size_t i = 0; i < sys.velocities.size(); ++i) {
    const double vMix = mixCoef1 * sys.velocities[i] + mixCoef2 * (-grad[i]);
    sys.velocities[i] = abcMult * vMix;
  }

  if (cfg.abcCorrection) {
    if (cfg.dMax > 0.0) {
      const double maxV = cfg.dMax / sys.dt;
      for (double& vi : sys.velocities) {
        vi = std::clamp(vi, -maxV, maxV);
      }
    }
    for (size_t i = 0; i < sys.velocities.size(); ++i) {
      sys.positions[i] += sys.dt * sys.velocities[i];
    }
  } else {
    double drScale = 1.0;
    if (cfg.dMax > 0.0) {
      double drSq = 0.0;
      for (const double vi : sys.velocities) {
        const double dri = sys.dt * vi;
        drSq += dri * dri;
      }
      const double drNorm = std::sqrt(drSq);
      if (drNorm > cfg.dMax) {
        drScale = cfg.dMax / drNorm;
      }
    }
    for (size_t i = 0; i < sys.velocities.size(); ++i) {
      sys.positions[i] += drScale * sys.dt * sys.velocities[i];
    }
  }
}

ReferenceConfig referenceConfigFromOptions(const nvMolKit::FireOptions& opts) {
  ReferenceConfig cfg;
  cfg.dtInit            = opts.dtInit;
  cfg.dtMinFactor       = opts.dtMinFactor;
  cfg.dtMaxFactor       = opts.dtMaxFactor;
  cfg.dMax              = opts.dMax;
  cfg.timeStepIncrement = opts.timeStepIncrement;
  cfg.timeStepDecrement = opts.timeStepDecrement;
  cfg.nMinForIncrease   = opts.nMinForIncrease;
  cfg.alphaInit         = opts.alphaInit;
  cfg.alphaDecrement    = opts.alphaDecrement;
  cfg.gradTol           = opts.gradTol;
  cfg.takeHalfStepBack  = opts.takeHalfStepBack;
  cfg.abcCorrection     = opts.abcCorrection;
  cfg.useMass           = opts.useMass;
  cfg.dataDim           = kDim;
  return cfg;
}

//! Per-system harmonic potential gradient: grad_i = k_s * (x_i - target_s_i).
__global__ void harmonicGradKernel(const int     numSystems,
                                   const int*    atomStarts,
                                   const double* kPerSystem,
                                   const double* targetPerCoord,
                                   const int     dataDim,
                                   const double* positions,
                                   double*       grad) {
  const int sysIdx = blockIdx.x;
  if (sysIdx >= numSystems) {
    return;
  }
  const int    atomBegin  = atomStarts[sysIdx];
  const int    atomEnd    = atomStarts[sysIdx + 1];
  const int    coordCount = (atomEnd - atomBegin) * dataDim;
  const double k          = kPerSystem[sysIdx];
  for (int i = threadIdx.x; i < coordCount; i += blockDim.x) {
    const int globalCoord = atomBegin * dataDim + i;
    const int coordInSys  = i % dataDim;
    grad[globalCoord]     = k * (positions[globalCoord] - targetPerCoord[sysIdx * dataDim + coordInSys]);
  }
}

class HarmonicSystems {
 public:
  HarmonicSystems(const std::vector<int>&    atomCounts,
                  const std::vector<double>& kPerSystem,
                  const std::vector<double>& startingPositions,
                  const std::vector<double>& targetPositions) {
    numSystems_ = static_cast<int>(atomCounts.size());
    atomStarts_.resize(numSystems_ + 1);
    atomStarts_[0] = 0;
    for (int sysIdx = 0; sysIdx < numSystems_; ++sysIdx) {
      atomStarts_[sysIdx + 1] = atomStarts_[sysIdx] + atomCounts[sysIdx];
    }
    totalAtoms_  = atomStarts_.back();
    totalCoords_ = totalAtoms_ * kDim;

    if (static_cast<int>(startingPositions.size()) != totalCoords_) {
      throw std::runtime_error("Starting positions size mismatch");
    }
    if (static_cast<int>(targetPositions.size()) != numSystems_ * kDim) {
      throw std::runtime_error("Target positions size mismatch");
    }
    positionsHost_ = startingPositions;
    targetHost_    = targetPositions;
    kHost_         = kPerSystem;

    atomStartsDevice_.setFromVector(atomStarts_);
    positionsDevice_.setFromVector(positionsHost_);
    gradDevice_.resize(totalCoords_);
    gradDevice_.zero();
    energyOuts_.resize(numSystems_);
    energyOuts_.zero();
    energyBuffer_.resize(totalCoords_);
    energyBuffer_.zero();
    kDevice_.setFromVector(kPerSystem);
    targetDevice_.setFromVector(targetHost_);
  }

  std::function<void()> gradFunctor() {
    return [this]() {
      const int blockSize = 64;
      harmonicGradKernel<<<numSystems_, blockSize>>>(numSystems_,
                                                     atomStartsDevice_.data(),
                                                     kDevice_.data(),
                                                     targetDevice_.data(),
                                                     kDim,
                                                     positionsDevice_.data(),
                                                     gradDevice_.data());
      cudaCheckError(cudaGetLastError());
    };
  }

  std::vector<double> systemPositions(const std::vector<double>& positions, const int sysIdx) const {
    const int begin = atomStarts_[sysIdx] * kDim;
    const int end   = atomStarts_[sysIdx + 1] * kDim;
    return std::vector<double>(positions.begin() + begin, positions.begin() + end);
  }

  std::vector<double> readbackPositions() {
    std::vector<double> result(totalCoords_);
    positionsDevice_.copyToHost(result);
    cudaCheckError(cudaDeviceSynchronize());
    return result;
  }

  int                                  numSystems() const { return numSystems_; }
  const std::vector<int>&              atomStartsHost() const { return atomStarts_; }
  const std::vector<double>&           startingPositionsHost() const { return positionsHost_; }
  const std::vector<double>&           targetsHost() const { return targetHost_; }
  const std::vector<double>&           kHost() const { return kHost_; }
  nvMolKit::AsyncDeviceVector<int>&    atomStartsDevice() { return atomStartsDevice_; }
  nvMolKit::AsyncDeviceVector<double>& positionsDevice() { return positionsDevice_; }
  nvMolKit::AsyncDeviceVector<double>& gradDevice() { return gradDevice_; }
  nvMolKit::AsyncDeviceVector<double>& energyOutsDevice() { return energyOuts_; }
  nvMolKit::AsyncDeviceVector<double>& energyBufferDevice() { return energyBuffer_; }

 private:
  int                                 numSystems_  = 0;
  int                                 totalAtoms_  = 0;
  int                                 totalCoords_ = 0;
  std::vector<int>                    atomStarts_;
  std::vector<double>                 positionsHost_;
  std::vector<double>                 targetHost_;
  std::vector<double>                 kHost_;
  nvMolKit::AsyncDeviceVector<int>    atomStartsDevice_;
  nvMolKit::AsyncDeviceVector<double> positionsDevice_;
  nvMolKit::AsyncDeviceVector<double> gradDevice_;
  nvMolKit::AsyncDeviceVector<double> energyOuts_;
  nvMolKit::AsyncDeviceVector<double> energyBuffer_;
  nvMolKit::AsyncDeviceVector<double> kDevice_;
  nvMolKit::AsyncDeviceVector<double> targetDevice_;
};

std::vector<ReferenceSystem> initializeReferenceSystems(const HarmonicSystems& systems, const ReferenceConfig& cfg) {
  std::vector<ReferenceSystem> refs(systems.numSystems());
  for (int sysIdx = 0; sysIdx < systems.numSystems(); ++sysIdx) {
    const auto          sysPos = systems.systemPositions(systems.startingPositionsHost(), sysIdx);
    std::vector<double> empty;
    initReferenceSystem(refs[sysIdx], sysPos, empty, cfg);
  }
  return refs;
}

void runReferenceStep(std::vector<ReferenceSystem>& refs,
                      const HarmonicSystems&        systems,
                      const ReferenceConfig&        cfg,
                      const bool                    isFirstStep) {
  // Use the reference's CURRENT positions for the gradient (matching the
  // device path where gFunc is called on the device positions before each
  // kernel iteration).
  for (int sysIdx = 0; sysIdx < systems.numSystems(); ++sysIdx) {
    if (refs[sysIdx].converged) {
      continue;
    }
    std::vector<double> grad(refs[sysIdx].positions.size());
    const double        k = systems.kHost()[sysIdx];
    for (size_t i = 0; i < refs[sysIdx].positions.size(); ++i) {
      const int    coordInSys = static_cast<int>(i) % cfg.dataDim;
      const double target     = systems.targetsHost()[sysIdx * cfg.dataDim + coordInSys];
      grad[i]                 = k * (refs[sysIdx].positions[i] - target);
    }
    referenceStep(refs[sysIdx], grad, cfg, isFirstStep);
  }
}

}  // namespace

// ---------- Tests ----------

TEST(FireMinimizer, BatchedReferenceTrajectoryMatchesAseFire2) {
  const std::vector<int>    atomCounts = {1, 1, 1, 1, 1, 1};
  const std::vector<double> kPerSys    = {2.5, 5.0, 7.5, 10.0, 12.5, 15.0};
  std::vector<double>       startingPositions(atomCounts.size() * kDim, 0.0);
  std::vector<double>       targets(atomCounts.size() * kDim, 0.0);
  for (size_t sysIdx = 0; sysIdx < atomCounts.size(); ++sysIdx) {
    startingPositions[sysIdx * kDim + 0] = 1.0;
  }
  HarmonicSystems systems(atomCounts, kPerSys, startingPositions, targets);

  nvMolKit::FireOptions options;
  options.stuckDetectionEnabled = false;  // ASE FIRE2 reference has no stuck-plateau exit; keep parity.
  options.dtInit                = 0.001;
  options.dMax                  = 0.0;
  options.gradTol               = 1e-3;
  options.useMass               = false;
  options.takeHalfStepBack      = true;
  options.abcCorrection         = false;
  options.nMinForIncrease       = 5;

  nvMolKit::FireBatchMinimizer minimizer(kDim, options);
  minimizer.setConvergencePollInterval(1);
  minimizer.initialize(systems.atomStartsHost());

  ReferenceConfig              refCfg = referenceConfigFromOptions(options);
  std::vector<ReferenceSystem> refs   = initializeReferenceSystems(systems, refCfg);
  std::vector<bool>            sawConvergedReference(refs.size(), false);
  std::vector<int>             refConvergedAtIter(refs.size(), -1);
  std::vector<int>             deviceConvergedAtIter(refs.size(), -1);

  const int maxIters = 4000;
  bool      done     = false;
  for (int iter = 0; iter < maxIters && !done; ++iter) {
    const bool isFirstStep = (iter == 0);
    done                   = minimizer.step(options.gradTol,
                          systems.atomStartsDevice(),
                          systems.positionsDevice(),
                          systems.gradDevice(),
                          systems.gradFunctor());
    runReferenceStep(refs, systems, refCfg, isFirstStep);

    const auto positions = systems.readbackPositions();
    const auto state     = minimizer.snapshotInternalState();

    for (int sysIdx = 0; sysIdx < systems.numSystems(); ++sysIdx) {
      const auto             devPos = systems.systemPositions(positions, sysIdx);
      const ReferenceSystem& ref    = refs[sysIdx];
      if (ref.converged && !sawConvergedReference[sysIdx]) {
        sawConvergedReference[sysIdx] = true;
        refConvergedAtIter[sysIdx]    = iter;
      }
      if (state.statuses[sysIdx] == 0 && deviceConvergedAtIter[sysIdx] < 0) {
        deviceConvergedAtIter[sysIdx] = iter;
      }
      for (size_t coord = 0; coord < ref.positions.size(); ++coord) {
        const double diff = std::abs(devPos[coord] - ref.positions[coord]);
        ASSERT_LT(diff, kReferenceTrajectoryTol) << "iter=" << iter << " sysIdx=" << sysIdx << " coord=" << coord
                                                 << " device=" << devPos[coord] << " ref=" << ref.positions[coord];
      }
      if (!ref.converged && iter < 50) {
        ASSERT_NEAR(state.dt[sysIdx], ref.dt, 1e-12) << "dt mismatch iter=" << iter << " sys=" << sysIdx;
        ASSERT_NEAR(state.alpha[sysIdx], ref.alpha, 1e-12) << "alpha mismatch iter=" << iter << " sys=" << sysIdx;
        ASSERT_EQ(state.nStepsPositive[sysIdx], ref.nstep) << "nstep mismatch iter=" << iter << " sys=" << sysIdx;
      }
    }
  }

  for (int sysIdx = 0; sysIdx < systems.numSystems(); ++sysIdx) {
    EXPECT_TRUE(refs[sysIdx].converged) << "Reference system " << sysIdx << " never converged";
    EXPECT_GE(refConvergedAtIter[sysIdx], 0) << "Reference convergence not recorded for sys " << sysIdx;
    EXPECT_GE(deviceConvergedAtIter[sysIdx], 0) << "Device convergence not recorded for sys " << sysIdx;
    EXPECT_LE(std::abs(deviceConvergedAtIter[sysIdx] - refConvergedAtIter[sysIdx]), 50)
      << "System " << sysIdx << " converged at different iterations on device vs reference";
  }
  std::vector<int> uniqueConvIters(refConvergedAtIter.begin(), refConvergedAtIter.end());
  std::sort(uniqueConvIters.begin(), uniqueConvIters.end());
  uniqueConvIters.erase(std::unique(uniqueConvIters.begin(), uniqueConvIters.end()), uniqueConvIters.end());
  EXPECT_GE(uniqueConvIters.size(), 2u) << "Test should produce at least two distinct convergence iterations";
}

TEST(FireMinimizer, AbcModeMatchesReference) {
  const std::vector<int>    atomCounts = {1, 1, 1, 1};
  const std::vector<double> kPerSys    = {3.0, 6.0, 9.0, 12.0};
  std::vector<double>       startingPositions(atomCounts.size() * kDim, 0.0);
  std::vector<double>       targets(atomCounts.size() * kDim, 0.0);
  for (size_t sysIdx = 0; sysIdx < atomCounts.size(); ++sysIdx) {
    startingPositions[sysIdx * kDim + 0] = 0.8;
    startingPositions[sysIdx * kDim + 1] = 0.3;
  }
  HarmonicSystems systems(atomCounts, kPerSys, startingPositions, targets);

  nvMolKit::FireOptions options;
  options.stuckDetectionEnabled = false;  // ASE FIRE2 reference has no stuck-plateau exit; keep parity.
  options.dtInit                = 0.001;
  options.dMax                  = 0.0;
  options.gradTol               = 1e-3;
  options.useMass               = false;
  options.takeHalfStepBack      = true;
  options.abcCorrection         = true;
  options.nMinForIncrease       = 5;

  nvMolKit::FireBatchMinimizer minimizer(kDim, options);
  minimizer.setConvergencePollInterval(1);
  minimizer.initialize(systems.atomStartsHost());

  ReferenceConfig              refCfg = referenceConfigFromOptions(options);
  std::vector<ReferenceSystem> refs   = initializeReferenceSystems(systems, refCfg);

  const int maxIters = 4000;
  bool      done     = false;
  for (int iter = 0; iter < maxIters && !done; ++iter) {
    const bool isFirstStep = (iter == 0);
    done                   = minimizer.step(options.gradTol,
                          systems.atomStartsDevice(),
                          systems.positionsDevice(),
                          systems.gradDevice(),
                          systems.gradFunctor());
    runReferenceStep(refs, systems, refCfg, isFirstStep);

    if (iter < 50) {
      const auto positions = systems.readbackPositions();
      for (int sysIdx = 0; sysIdx < systems.numSystems(); ++sysIdx) {
        const auto devPos = systems.systemPositions(positions, sysIdx);
        for (size_t coord = 0; coord < refs[sysIdx].positions.size(); ++coord) {
          const double diff = std::abs(devPos[coord] - refs[sysIdx].positions[coord]);
          ASSERT_LT(diff, kReferenceTrajectoryTol)
            << "iter=" << iter << " sys=" << sysIdx << " coord=" << coord << " device=" << devPos[coord]
            << " ref=" << refs[sysIdx].positions[coord];
        }
      }
    }
  }
  for (const auto& ref : refs) {
    EXPECT_TRUE(ref.converged);
  }
}

TEST(FireMinimizer, MaxStepNormClipping) {
  const std::vector<int>    atomCounts = {1};
  const std::vector<double> kPerSys    = {100.0};
  std::vector<double>       startingPositions(kDim, 0.0);
  startingPositions[0] = 5.0;
  std::vector<double> targets(kDim, 0.0);

  HarmonicSystems systems(atomCounts, kPerSys, startingPositions, targets);

  nvMolKit::FireOptions options;
  options.stuckDetectionEnabled = false;  // ASE FIRE2 reference has no stuck-plateau exit; keep parity.
  options.dtInit                = 0.001;
  options.dMax                  = 0.05;
  options.gradTol               = 1e-3;
  options.useMass               = false;
  options.abcCorrection         = false;
  options.takeHalfStepBack      = false;  // isolate the clipped post-kick displacement in this test.

  nvMolKit::FireBatchMinimizer minimizer(kDim, options);
  minimizer.setConvergencePollInterval(1);
  minimizer.initialize(systems.atomStartsHost());

  std::vector<double> previous = systems.readbackPositions();
  for (int iter = 0; iter < 200; ++iter) {
    const bool done    = minimizer.step(options.gradTol,
                                     systems.atomStartsDevice(),
                                     systems.positionsDevice(),
                                     systems.gradDevice(),
                                     systems.gradFunctor());
    const auto current = systems.readbackPositions();
    double     drNorm  = 0.0;
    for (size_t i = 0; i < current.size(); ++i) {
      const double d = current[i] - previous[i];
      drNorm += d * d;
    }
    drNorm = std::sqrt(drNorm);
    EXPECT_LE(drNorm, options.dMax + 1e-9) << "iter=" << iter << " unbounded displacement " << drNorm;
    previous = current;
    if (done) {
      break;
    }
  }

  // Repeat without clipping: drNorm exceeds dMax for at least one step
  HarmonicSystems       systems2(atomCounts, kPerSys, startingPositions, targets);
  nvMolKit::FireOptions options2 = options;
  options2.dMax                  = 0.0;
  nvMolKit::FireBatchMinimizer minimizer2(kDim, options2);
  minimizer2.setConvergencePollInterval(1);
  minimizer2.initialize(systems2.atomStartsHost());

  bool                exceeded = false;
  std::vector<double> prev2    = systems2.readbackPositions();
  for (int iter = 0; iter < 50; ++iter) {
    minimizer2.step(options.gradTol,
                    systems2.atomStartsDevice(),
                    systems2.positionsDevice(),
                    systems2.gradDevice(),
                    systems2.gradFunctor());
    const auto current = systems2.readbackPositions();
    double     drNorm  = 0.0;
    for (size_t i = 0; i < current.size(); ++i) {
      const double d = current[i] - prev2[i];
      drNorm += d * d;
    }
    drNorm = std::sqrt(drNorm);
    if (drNorm > options.dMax + 1e-9) {
      exceeded = true;
      break;
    }
    prev2 = current;
  }
  EXPECT_TRUE(exceeded) << "Without clipping, the unconstrained dynamics should overshoot dMax in at least one step";
}

TEST(FireMinimizer, NegativePowerHalfStepBack) {
  const std::vector<int>    atomCounts = {1};
  const std::vector<double> kPerSys    = {2000.0};  // chosen so raw-force dynamics overshoot quickly
  std::vector<double>       startingPositions(kDim, 0.0);
  startingPositions[0] = 1.5;
  std::vector<double> targets(kDim, 0.0);
  HarmonicSystems     systems(atomCounts, kPerSys, startingPositions, targets);

  nvMolKit::FireOptions options;
  options.stuckDetectionEnabled = false;  // ASE FIRE2 reference has no stuck-plateau exit; keep parity.
  options.dtInit                = 0.05;   // intentionally large -> overshoot -> negative power soon
  options.dtMinFactor           = 0.001;
  options.dtMaxFactor           = 10.0;
  options.dMax                  = 0.0;
  options.gradTol               = 1e-3;
  options.useMass               = false;
  options.takeHalfStepBack      = true;
  options.abcCorrection         = false;
  options.nMinForIncrease       = 5;

  nvMolKit::FireBatchMinimizer minimizer(kDim, options);
  minimizer.setConvergencePollInterval(1);
  minimizer.initialize(systems.atomStartsHost());

  ReferenceConfig              refCfg = referenceConfigFromOptions(options);
  std::vector<ReferenceSystem> refs   = initializeReferenceSystems(systems, refCfg);

  bool   sawReferenceNegative = false;
  bool   sawDeviceNegative    = false;
  double previousDeviceDt     = options.dtInit;
  for (int iter = 0; iter < 60; ++iter) {
    const bool isFirstStep = (iter == 0);
    minimizer.step(options.gradTol,
                   systems.atomStartsDevice(),
                   systems.positionsDevice(),
                   systems.gradDevice(),
                   systems.gradFunctor());
    runReferenceStep(refs, systems, refCfg, isFirstStep);

    const auto state = minimizer.snapshotInternalState();
    if (refs[0].nstep == 0 && refs[0].dt < refCfg.dtInit) {
      sawReferenceNegative = true;
    }
    if (state.nStepsPositive[0] == 0 && state.dt[0] < previousDeviceDt) {
      sawDeviceNegative = true;
      EXPECT_NEAR(state.alpha[0], options.alphaInit, 1e-12);
    }
    previousDeviceDt = state.dt[0];
  }
  EXPECT_TRUE(sawReferenceNegative) << "Test setup should trigger at least one reference negative-power step";
  EXPECT_TRUE(sawDeviceNegative) << "Test setup should trigger at least one device negative-power step";
}

TEST(FireMinimizer, MmffPhysicalUnits) {
  const std::vector<int>    atomCounts = {2};
  const std::vector<double> kPerSys    = {300.0};  // ~kcal/mol/Å^2 spring
  std::vector<double>       startingPositions(2 * kDim, 0.0);
  startingPositions[0] = 0.4;
  startingPositions[3] = -0.3;
  std::vector<double> targets(kDim, 0.0);
  HarmonicSystems     systems(atomCounts, kPerSys, startingPositions, targets);

  nvMolKit::FireOptions options;
  options.stuckDetectionEnabled = false;  // ASE FIRE2 reference has no stuck-plateau exit; keep parity.
  options.dtInit                = 0.001;
  options.dMax                  = 0.1;
  options.gradTol               = 1e-4;
  options.useMass               = true;
  options.abcCorrection         = false;
  options.nMinForIncrease       = 5;

  nvMolKit::FireBatchMinimizer minimizer(kDim, options);
  std::vector<double>          masses(2, 12.0);  // carbon mass
  minimizer.setMasses(masses);
  minimizer.setConvergencePollInterval(1);

  bool converged = minimizer.minimize(/*numIters=*/200,
                                      options.gradTol,
                                      systems.atomStartsHost(),
                                      systems.atomStartsDevice(),
                                      systems.positionsDevice(),
                                      systems.gradDevice(),
                                      systems.energyOutsDevice(),
                                      systems.energyBufferDevice(),
                                      [](const double*) {},
                                      systems.gradFunctor());
  EXPECT_TRUE(converged) << "Realistic MMFF-scale system should converge in <=200 FIRE steps";
  const auto positions = systems.readbackPositions();
  for (size_t i = 0; i < positions.size(); ++i) {
    EXPECT_NEAR(positions[i], 0.0, 1e-3) << "coord " << i << " did not relax to target";
  }
}

TEST(FireMinimizer, MassWeightingScalesAcceleration) {
  const std::vector<int>    atomCounts = {1};
  const std::vector<double> kPerSys    = {10.0};
  std::vector<double>       startingPositions(kDim, 0.0);
  startingPositions[0] = 1.0;
  std::vector<double> targets(kDim, 0.0);

  auto runOneKick = [&](double mass) {
    HarmonicSystems       systems(atomCounts, kPerSys, startingPositions, targets);
    nvMolKit::FireOptions options;
    options.stuckDetectionEnabled = false;  // ASE FIRE2 reference has no stuck-plateau exit; keep parity.
    options.dtInit                = 0.001;
    options.dMax                  = 0.0;
    options.gradTol               = 0.0;
    options.useMass               = true;
    options.takeHalfStepBack      = true;
    options.abcCorrection         = false;
    nvMolKit::FireBatchMinimizer minimizer(kDim, options);
    minimizer.setMasses({mass});
    minimizer.setConvergencePollInterval(1);
    minimizer.initialize(systems.atomStartsHost());
    minimizer.step(options.gradTol,
                   systems.atomStartsDevice(),
                   systems.positionsDevice(),
                   systems.gradDevice(),
                   systems.gradFunctor());
    return systems.readbackPositions();
  };

  const auto pos1  = runOneKick(1.0);
  const auto pos5  = runOneKick(5.0);
  const auto pos10 = runOneKick(10.0);

  const double disp1  = std::abs(pos1[0] - startingPositions[0]);
  const double disp5  = std::abs(pos5[0] - startingPositions[0]);
  const double disp10 = std::abs(pos10[0] - startingPositions[0]);

  // Heavier mass -> smaller acceleration -> smaller first-step displacement.
  EXPECT_GT(disp1, disp5);
  EXPECT_GT(disp5, disp10);
  // Linear inverse scaling for the very first step (no mixer history).
  EXPECT_NEAR(disp5 * 5.0, disp1 * 1.0, 1e-9);
  EXPECT_NEAR(disp10 * 10.0, disp1 * 1.0, 1e-9);
}

TEST(FireMinimizer, UseMassFalseEqualsEquivalentPhysicalMass) {
  const std::vector<int>    atomCounts = {2};
  const std::vector<double> kPerSys    = {3.0};
  std::vector<double>       startingPositions(2 * kDim, 0.0);
  startingPositions[0] = 0.5;
  startingPositions[3] = -0.5;
  std::vector<double> targets(kDim, 0.0);

  auto runFifty = [&](bool useMass, std::vector<double> masses) {
    HarmonicSystems       systems(atomCounts, kPerSys, startingPositions, targets);
    nvMolKit::FireOptions options;
    options.stuckDetectionEnabled = false;  // ASE FIRE2 reference has no stuck-plateau exit; keep parity.
    options.dtInit                = 0.001;
    options.dMax                  = 0.0;
    options.gradTol               = 0.0;
    options.useMass               = useMass;
    options.abcCorrection         = false;
    options.nMinForIncrease       = 5;
    nvMolKit::FireBatchMinimizer minimizer(kDim, options);
    if (useMass) {
      minimizer.setMasses(masses);
    }
    minimizer.setConvergencePollInterval(1);
    minimizer.initialize(systems.atomStartsHost());
    for (int iter = 0; iter < 50; ++iter) {
      minimizer.step(options.gradTol,
                     systems.atomStartsDevice(),
                     systems.positionsDevice(),
                     systems.gradDevice(),
                     systems.gradFunctor());
    }
    return systems.readbackPositions();
  };

  const auto noMass = runFifty(false, {});
  const auto equivalentPhysicalMass =
    runFifty(true, {kForceKcalMolPerAng_PerAmu_to_AngPerPs2_Local, kForceKcalMolPerAng_PerAmu_to_AngPerPs2_Local});
  ASSERT_EQ(noMass.size(), equivalentPhysicalMass.size());
  for (size_t i = 0; i < noMass.size(); ++i) {
    EXPECT_NEAR(noMass[i], equivalentPhysicalMass[i], 1e-12) << "coord " << i;
  }
}

TEST(FireMinimizer, ParameterPropagation) {
  const std::vector<int>    atomCounts = {1};
  const std::vector<double> kPerSys    = {1.0};
  std::vector<double>       startingPositions(kDim, 0.0);
  startingPositions[0] = 0.5;
  std::vector<double> targets(kDim, 0.0);
  HarmonicSystems     systems(atomCounts, kPerSys, startingPositions, targets);

  nvMolKit::FireOptions options;
  options.stuckDetectionEnabled = false;  // ASE FIRE2 reference has no stuck-plateau exit; keep parity.
  options.dtInit                = 0.002;
  options.dtMinFactor           = 0.01;
  options.dtMaxFactor           = 4.0;
  options.alphaInit             = 0.4;
  options.alphaDecrement        = 0.5;
  options.timeStepIncrement     = 1.5;
  options.timeStepDecrement     = 0.25;
  options.nMinForIncrease       = 2;
  options.dMax                  = 0.0;
  options.gradTol               = 1e-9;  // never converge during this test
  options.useMass               = false;
  options.takeHalfStepBack      = true;
  options.abcCorrection         = false;

  nvMolKit::FireBatchMinimizer minimizer(kDim, options);
  minimizer.setConvergencePollInterval(1);
  minimizer.initialize(systems.atomStartsHost());

  auto state0 = minimizer.snapshotInternalState();
  EXPECT_NEAR(state0.dt[0], options.dtInit, 1e-15);
  EXPECT_NEAR(state0.alpha[0], options.alphaInit, 1e-15);
  EXPECT_EQ(state0.nStepsPositive[0], 0);

  ReferenceConfig              refCfg = referenceConfigFromOptions(options);
  std::vector<ReferenceSystem> refs   = initializeReferenceSystems(systems, refCfg);
  for (int iter = 0; iter < 30; ++iter) {
    const bool isFirstStep = (iter == 0);
    minimizer.step(options.gradTol,
                   systems.atomStartsDevice(),
                   systems.positionsDevice(),
                   systems.gradDevice(),
                   systems.gradFunctor());
    runReferenceStep(refs, systems, refCfg, isFirstStep);
    const auto state = minimizer.snapshotInternalState();
    EXPECT_NEAR(state.dt[0], refs[0].dt, 1e-12) << "iter=" << iter;
    EXPECT_NEAR(state.alpha[0], refs[0].alpha, 1e-12) << "iter=" << iter;
    EXPECT_EQ(state.nStepsPositive[0], refs[0].nstep) << "iter=" << iter;
    EXPECT_LE(state.dt[0], options.dtInit * options.dtMaxFactor + 1e-12);
    EXPECT_GE(state.dt[0], options.dtInit * options.dtMinFactor - 1e-12);
  }
}

TEST(FireMinimizer, ActiveSystemMaskRespected) {
  const std::vector<int>    atomCounts = {1, 1, 1};
  const std::vector<double> kPerSys    = {2.0, 2.0, 2.0};
  std::vector<double>       startingPositions(3 * kDim, 0.0);
  for (int sysIdx = 0; sysIdx < 3; ++sysIdx) {
    startingPositions[sysIdx * kDim + 0] = 1.0;
  }
  std::vector<double> targets(3 * kDim, 0.0);
  HarmonicSystems     systems(atomCounts, kPerSys, startingPositions, targets);

  nvMolKit::FireOptions options;
  options.stuckDetectionEnabled = false;  // ASE FIRE2 reference has no stuck-plateau exit; keep parity.
  options.gradTol               = 1e-4;
  options.dMax                  = 0.0;
  options.useMass               = false;
  options.takeHalfStepBack      = true;
  options.abcCorrection         = false;
  options.nMinForIncrease       = 5;

  nvMolKit::FireBatchMinimizer minimizer(kDim, options);
  std::vector<uint8_t>         mask = {1, 0, 1};
  minimizer.setConvergencePollInterval(1);
  minimizer.initialize(systems.atomStartsHost(), nullptr, mask.data());

  bool done = false;
  for (int iter = 0; iter < 2000 && !done; ++iter) {
    done = minimizer.step(options.gradTol,
                          systems.atomStartsDevice(),
                          systems.positionsDevice(),
                          systems.gradDevice(),
                          systems.gradFunctor());
  }
  EXPECT_TRUE(done);
  const auto positions = systems.readbackPositions();
  for (int dim = 0; dim < kDim; ++dim) {
    const int coord = 1 * kDim + dim;
    EXPECT_DOUBLE_EQ(positions[coord], startingPositions[coord]) << "dim=" << dim;
  }
  for (int sysIdx : {0, 2}) {
    for (int dim = 0; dim < kDim; ++dim) {
      const int coord = sysIdx * kDim + dim;
      EXPECT_NEAR(positions[coord], 0.0, 1e-2) << "sys=" << sysIdx << " dim=" << dim;
    }
  }
}

TEST(FireMinimizer, ActiveMaskMatchesBfgsContract) {
  // Pin down the contract that ETKDG's bfgs_distgeom-style call sites depend
  // on: with activeThisStage = {1, 0, 1, 1, 0, 1}, the FIRE minimizer must
  // not touch inactive systems' positions, velocities, dt, alpha, or
  // nStepsPositive, regardless of what the gradient functor writes.
  const std::vector<int>    atomCounts = {1, 1, 1, 1, 1, 1};
  const std::vector<double> kPerSys    = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<double>       startingPositions(atomCounts.size() * kDim, 0.0);
  for (size_t sysIdx = 0; sysIdx < atomCounts.size(); ++sysIdx) {
    startingPositions[sysIdx * kDim + 0] = 1.0;
  }
  std::vector<double> targets(atomCounts.size() * kDim, 0.0);
  HarmonicSystems     systems(atomCounts, kPerSys, startingPositions, targets);

  std::vector<uint8_t> mask = {1, 0, 1, 1, 0, 1};

  nvMolKit::FireOptions options;
  options.stuckDetectionEnabled = false;  // ASE FIRE2 reference has no stuck-plateau exit; keep parity.
  options.gradTol               = 1e-4;
  options.dMax                  = 0.0;
  options.useMass               = false;
  options.takeHalfStepBack      = true;
  options.abcCorrection         = false;
  options.nMinForIncrease       = 5;
  nvMolKit::FireBatchMinimizer minimizer(kDim, options);
  minimizer.setConvergencePollInterval(1);
  minimizer.initialize(systems.atomStartsHost(), nullptr, mask.data());

  const auto initialState = minimizer.snapshotInternalState();
  bool       done         = false;
  for (int iter = 0; iter < 2000 && !done; ++iter) {
    done = minimizer.step(options.gradTol,
                          systems.atomStartsDevice(),
                          systems.positionsDevice(),
                          systems.gradDevice(),
                          systems.gradFunctor());
  }
  EXPECT_TRUE(done);

  const auto positions  = systems.readbackPositions();
  const auto finalState = minimizer.snapshotInternalState();

  for (size_t sysIdx = 0; sysIdx < mask.size(); ++sysIdx) {
    if (mask[sysIdx] == 0) {
      for (int dim = 0; dim < kDim; ++dim) {
        const int coord = sysIdx * kDim + dim;
        EXPECT_DOUBLE_EQ(positions[coord], startingPositions[coord])
          << "Inactive system " << sysIdx << " dim " << dim << " was modified";
      }
      EXPECT_EQ(finalState.dt[sysIdx], initialState.dt[sysIdx]) << "Inactive system " << sysIdx << " dt was modified";
      EXPECT_EQ(finalState.alpha[sysIdx], initialState.alpha[sysIdx])
        << "Inactive system " << sysIdx << " alpha was modified";
      EXPECT_EQ(finalState.nStepsPositive[sysIdx], initialState.nStepsPositive[sysIdx])
        << "Inactive system " << sysIdx << " nStepsPositive was modified";
      EXPECT_EQ(finalState.statuses[sysIdx], 0) << "Inactive system " << sysIdx << " status flipped to active";
      for (int dim = 0; dim < kDim; ++dim) {
        const int coord = sysIdx * kDim + dim;
        EXPECT_DOUBLE_EQ(finalState.velocities[coord], 0.0)
          << "Inactive system " << sysIdx << " velocity coord " << dim << " was modified";
      }
    } else {
      for (int dim = 0; dim < kDim; ++dim) {
        const int coord = sysIdx * kDim + dim;
        EXPECT_NEAR(positions[coord], 0.0, 1e-2) << "Active system " << sysIdx << " dim " << dim << " did not relax";
      }
    }
  }
}

TEST(FireMinimizer, StaggeredConvergenceCount) {
  const std::vector<int>    atomCounts = {1, 1, 1, 1, 1, 1, 1, 1};
  const std::vector<double> kPerSys    = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  std::vector<double>       startingPositions(atomCounts.size() * kDim, 0.0);
  for (size_t sysIdx = 0; sysIdx < atomCounts.size(); ++sysIdx) {
    startingPositions[sysIdx * kDim + 0] = 1.0;
  }
  std::vector<double> targets(atomCounts.size() * kDim, 0.0);
  HarmonicSystems     systems(atomCounts, kPerSys, startingPositions, targets);

  nvMolKit::FireOptions options;
  options.stuckDetectionEnabled = false;  // ASE FIRE2 reference has no stuck-plateau exit; keep parity.
  options.dtInit                = 0.001;
  options.gradTol               = 1e-3;
  options.dMax                  = 0.0;
  options.useMass               = false;
  options.abcCorrection         = false;
  options.nMinForIncrease       = 5;

  nvMolKit::FireBatchMinimizer minimizer(kDim, options);
  minimizer.setConvergencePollInterval(1);
  minimizer.initialize(systems.atomStartsHost());

  std::vector<int> deviceConvergedAtIter(systems.numSystems(), -1);

  bool done = false;
  for (int iter = 0; iter < 4000 && !done; ++iter) {
    done             = minimizer.step(options.gradTol,
                          systems.atomStartsDevice(),
                          systems.positionsDevice(),
                          systems.gradDevice(),
                          systems.gradFunctor());
    const auto state = minimizer.snapshotInternalState();

    int activeFromStatuses = 0;
    for (int sysIdx = 0; sysIdx < systems.numSystems(); ++sysIdx) {
      if (state.statuses[sysIdx] != 0) {
        ++activeFromStatuses;
      } else if (deviceConvergedAtIter[sysIdx] < 0) {
        deviceConvergedAtIter[sysIdx] = iter;
      }
    }
    EXPECT_EQ(minimizer.numActiveSystemsHost(), activeFromStatuses)
      << "iter=" << iter << " host active count diverged from device statuses";
  }
  EXPECT_EQ(minimizer.numActiveSystemsHost(), 0);

  for (int sysIdx = 0; sysIdx < systems.numSystems(); ++sysIdx) {
    EXPECT_GE(deviceConvergedAtIter[sysIdx], 0) << "Device convergence not recorded for sys " << sysIdx;
  }
  std::vector<int> uniqueConvIters(deviceConvergedAtIter.begin(), deviceConvergedAtIter.end());
  std::sort(uniqueConvIters.begin(), uniqueConvIters.end());
  uniqueConvIters.erase(std::unique(uniqueConvIters.begin(), uniqueConvIters.end()), uniqueConvIters.end());
  EXPECT_GE(uniqueConvIters.size(), 2u) << "Test should produce at least two distinct convergence iterations";
}

TEST(FireMinimizer, HybridBackendSelectionAndPerMolInitialization) {
  nvMolKit::FireOptions        options;
  nvMolKit::FireBatchMinimizer minimizer(kDim,
                                         options,
                                         /*stream=*/nullptr,
                                         /*debugMode=*/false,
                                         nvMolKit::FireBackend::HYBRID);

  EXPECT_EQ(minimizer.resolveBackend({0, 5, 10, 30}), nvMolKit::FireBackend::PER_MOLECULE);
  EXPECT_EQ(minimizer.resolveBackend({0, 5, 10, 200}), nvMolKit::FireBackend::BATCHED);

  const std::vector<int> atomStarts = {0, 4, 10, 18};
  minimizer.initialize(atomStarts, nullptr, nullptr, nvMolKit::FireBackend::PER_MOLECULE);

  EXPECT_EQ(minimizer.numActiveSystemsHost(), 3);
  const auto state = minimizer.snapshotInternalState();
  ASSERT_EQ(state.statuses.size(), 3u);
  ASSERT_EQ(state.dt.size(), 3u);
  ASSERT_EQ(state.alpha.size(), 3u);
  ASSERT_EQ(state.nStepsPositive.size(), 3u);
  for (size_t i = 0; i < state.statuses.size(); ++i) {
    EXPECT_EQ(state.statuses[i], 1);
    EXPECT_EQ(state.nStepsPositive[i], 0);
    EXPECT_NEAR(state.dt[i], options.dtInit, 0.0);
    EXPECT_NEAR(state.alpha[i], options.alphaInit, 0.0);
  }
}
