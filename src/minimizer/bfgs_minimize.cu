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

#include <algorithm>
#include <cub/cub.cuh>
#include <numeric>
#include <type_traits>

#include "src/forcefields/batched_forcefield.h"
#include "src/forcefields/dist_geom.h"
#include "src/forcefields/dist_geom_kernels.h"
#include "src/forcefields/mmff.h"
#include "src/forcefields/mmff_kernels.h"
#include "src/minimizer/bfgs_hessian.h"
#include "src/minimizer/bfgs_minimize.h"
#include "src/minimizer/bfgs_minimize_permol_kernels.h"
#include "src/utils/cub_helpers.cuh"
#include "src/utils/device_convert.cuh"
#include "src/utils/device_vector.h"
#include "src/utils/nvtx.h"
#include "versions.h"

namespace nvMolKit {
constexpr double FUNCTOL = 1e-4;  //!< Default tolerance for function convergence in the minimizer
constexpr double MOVETOL = 1e-7;  //!< Default tolerance for x changes in the minimizer

namespace {
template <typename real> __device__ __forceinline__ real bfgsSqrt(real value) {
  if constexpr (cuda::std::is_same_v<real, float>)
    return sqrtf(value);
  else
    return sqrt(value);
}
template <typename real> __device__ __forceinline__ real bfgsAbs(real value) {
  if constexpr (cuda::std::is_same_v<real, float>)
    return fabsf(value);
  else
    return fabs(value);
}
BfgsBackend resolveBackend(BfgsBackend backend, const std::vector<int>& atomStartsHost) {
  if (backend != BfgsBackend::HYBRID) {
    return backend;
  }
  for (size_t i = 0; i + 1 < atomStartsHost.size(); ++i) {
    if (atomStartsHost[i + 1] - atomStartsHost[i] > kHybridBackendAtomThreshold) {
      return BfgsBackend::BATCHED;
    }
  }
  return BfgsBackend::PER_MOLECULE;
}

}  // namespace

// TODO - consolidate this to device vector code. We don't want CUDA in the device vector
// header so we'll need to specialize for a few types and instantiate them in the cu file.
template <typename T> __global__ void setAllKernel(const int numElements, T value, T* dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = value;
  }
}

// Specialized version for copying from uint8_t to int16_t
__global__ void copyActiveToStatusKernel(const int numElements, const uint8_t* src, int16_t* dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = static_cast<int16_t>(src[idx]);
  }
}

template <typename T> void setAll(AsyncDeviceVector<T>& vec, const T& value) {
  const int          numElements = vec.size();
  const cudaStream_t stream      = vec.stream();
  if (numElements == 0) {
    return;
  }
  const int blockSize = 128;
  const int numBlocks = (numElements + blockSize - 1) / blockSize;
  setAllKernel<<<numBlocks, blockSize, 0, stream>>>(numElements, value, vec.data());
  cudaCheckError(cudaGetLastError());
}

// Scale direction vector, get slope and test values.
template <typename real, typename reduceT, typename storageT>
__global__ void initializeLineSearchKernel(const int16_t*  statuses,
                                           const storageT* oldPositions,
                                           const storageT* grads,
                                           const int*      atomStarts,
                                           const storageT* maxSteps,
                                           storageT*       dirs,
                                           storageT*       slopes,
                                           storageT*       lambdaMins,
                                           const int*      activeSystemIndices,
                                           const int       DIM) {
  const int  sysIdx        = activeSystemIndices[blockIdx.x];
  const int  idxInSys      = threadIdx.x;
  const bool isFirstThread = threadIdx.x == 0;

  if (statuses[sysIdx] == 0) {
    return;
  }

  const int       numTerms  = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const storageT* posStart  = &oldPositions[atomStarts[sysIdx] * DIM];
  const storageT* gradStart = &grads[atomStarts[sysIdx] * DIM];
  storageT*       dirStart  = &dirs[atomStarts[sysIdx] * DIM];

  using BlockReduce = cub::BlockReduce<reduceT, 128>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ reduceT                           dirSum[1];

  // ---------------------------------
  //  Scale direction vector if needed
  // ---------------------------------
  reduceT sumSquaredLocal = 0;
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    const real dx = static_cast<real>(dirStart[i]);
    sumSquaredLocal += static_cast<reduceT>(dx * dx);
  }
  reduceT blockSum = BlockReduce(tempStorage).Sum(sumSquaredLocal);
  if (isFirstThread) {
    dirSum[0] = bfgsSqrt(blockSum);
  }
  __syncthreads();
  if (dirSum[0] > maxSteps[sysIdx]) {
    for (int i = idxInSys; i < numTerms; i += blockDim.x) {
      const real scaled =
        static_cast<real>(dirStart[i]) * (static_cast<real>(maxSteps[sysIdx]) / static_cast<real>(dirSum[0]));
      dirStart[i] = static_cast<storageT>(scaled);
    }
  }

  // -------------------------
  // Set slope, check validity
  // -------------------------
  reduceT localSum = 0;
  // Each thread computes its partial sum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    localSum += static_cast<reduceT>(static_cast<real>(dirStart[i]) * static_cast<real>(gradStart[i]));
  }

  // Perform block-wide reduction to compute the total sum
  blockSum = BlockReduce(tempStorage).Sum(localSum);
  __syncthreads();
  // The first thread in the block writes the result

  if (isFirstThread) {
    slopes[sysIdx] = static_cast<storageT>(blockSum);
  }

  // ----------------------
  // Compute initial lambda
  // ----------------------
  reduceT localMax = 0;
  // Each thread computes its local maximum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    const real temp = bfgsAbs(static_cast<real>(dirStart[i])) / max(bfgsAbs(static_cast<real>(posStart[i])), real{1});
    if (temp > localMax) {
      localMax = temp;
    }
  }
  // Perform block-wide reduction to find the maximum
  reduceT blockMax = BlockReduce(tempStorage).Reduce(localMax, cubMax());

  // The first thread in the block writes the result
  if (isFirstThread) {
    lambdaMins[sysIdx] = static_cast<storageT>(static_cast<reduceT>(MOVETOL) / blockMax);
  }
}

template <typename storageT>
__global__ void setLineStatusAndEnergyFromGlobalKernel(const int numSystems,
                                                       const int16_t* __restrict__ statuses,
                                                       int16_t* __restrict__ lineSearchStatus,
                                                       const storageT* __restrict__ srcEnergies,
                                                       storageT* __restrict__ destEnergies,
                                                       storageT* __restrict__ lineSearchLambdas) {
  const int sysIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (sysIdx < numSystems) {
    const int16_t status      = statuses[sysIdx];
    lineSearchStatus[sysIdx]  = status == 0 ? 0 : -2;
    destEnergies[sysIdx]      = static_cast<storageT>(srcEnergies[sysIdx]);
    lineSearchLambdas[sysIdx] = storageT{1};
  }
}

template <typename Workspace, typename storageT>
void BfgsBatchMinimizer::doLineSearchSetupImpl(const storageT* srcEnergies,
                                               const storageT* positions,
                                               const storageT* grads,
                                               Workspace&      workspace) {
  using real                   = typename Workspace::ComputeScalar;
  using reduceT                = typename Workspace::ReductionScalar;
  const int     numblocks      = (numSystems_ + 128 - 1) / 128;
  constexpr int blockSizeSetup = 128;
  setLineStatusAndEnergyFromGlobalKernel<storageT>
    <<<numblocks, 128, 0, stream_>>>(numSystems_,
                                     statuses_.data(),
                                     lineSearchStatus_.data(),
                                     srcEnergies,
                                     workspace.lineSearchStoredEnergy.data(),
                                     workspace.lineSearchLambdas.data());
  initializeLineSearchKernel<real, reduceT, storageT>
    <<<numUnfinishedSystems_, blockSizeSetup, 0, stream_>>>(statuses_.data(),
                                                            positions,
                                                            grads,
                                                            atomStartsDevice,
                                                            workspace.lineSearchMaxSteps.data(),
                                                            workspace.lineSearchDir.data(),
                                                            workspace.lineSearchSlope.data(),
                                                            workspace.lineSearchLambdaMins.data(),
                                                            activeSystemIndices_.data(),
                                                            dataDim_);
  cudaCheckError(cudaGetLastError());
}

void BfgsBatchMinimizer::doLineSearchSetup(const double* srcEnergies) {
  if (usesSinglePrecision(precision_)) {
    doLineSearchSetupImpl(singleWorkspace().energy.data(),
                          singleWorkspace().positions.data(),
                          singleWorkspace().grad.data(),
                          singleWorkspace());
    return;
  }
  doLineSearchSetupImpl(srcEnergies, positionsDevice, gradDevice, fullWorkspace());
}

template <typename real, typename storageT>
__global__ void lineSearchPerturbKernel(const int*      atomStarts,
                                        const storageT* refPos,
                                        const storageT* dirs,
                                        const storageT* lambdas,
                                        const storageT* lambdaMins,
                                        storageT*       statePositions,
                                        storageT*       evalPositions,
                                        int16_t*        statuses,
                                        const int*      activeSystemIndices,
                                        const int       DIM) {
  const int       sysIdx        = activeSystemIndices[blockIdx.x];
  const int       idxInSys      = threadIdx.x;
  const int       numTerms      = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const storageT* dirStart      = &dirs[atomStarts[sysIdx] * DIM];
  const storageT* oldPosStart   = &refPos[atomStarts[sysIdx] * DIM];
  storageT*       statePosStart = &statePositions[atomStarts[sysIdx] * DIM];
  storageT*       evalPosStart  = &evalPositions[atomStarts[sysIdx] * DIM];
  const bool      isFirstThread = threadIdx.x == 0;

  const int16_t status = statuses[sysIdx];
  if (status != -2) {
    // Case where we've already converged or failed.
    return;
  }
  const real lambda    = static_cast<real>(lambdas[sysIdx]);
  const real lambdaMin = static_cast<real>(lambdaMins[sysIdx]);

  if (lambda < lambdaMin) {
    if (isFirstThread) {
      statuses[sysIdx] = 1;
    }
    return;
  }

  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    const real value = static_cast<real>(oldPosStart[i]) + lambda * static_cast<real>(dirStart[i]);
    statePosStart[i] = static_cast<storageT>(value);
    evalPosStart[i]  = static_cast<storageT>(value);
  }
}

template <typename Workspace, typename storageT>
void BfgsBatchMinimizer::doLineSearchPerturbImpl(const storageT* positions, Workspace& workspace) {
  using real = typename Workspace::ComputeScalar;
  lineSearchPerturbKernel<real, storageT>
    <<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                 positions,
                                                 workspace.lineSearchDir.data(),
                                                 workspace.lineSearchLambdas.data(),
                                                 workspace.lineSearchLambdaMins.data(),
                                                 workspace.scratchPositions.data(),
                                                 workspace.scratchPositions.data(),
                                                 lineSearchStatus_.data(),
                                                 activeSystemIndices_.data(),
                                                 dataDim_);
  cudaCheckError(cudaGetLastError());
}

void BfgsBatchMinimizer::doLineSearchPerturb() {
  if (usesSinglePrecision(precision_)) {
    doLineSearchPerturbImpl(singleWorkspace().positions.data(), singleWorkspace());
    return;
  }
  doLineSearchPerturbImpl(positionsDevice, fullWorkspace());
}

template <typename real, typename storageT>
__global__ void lineSearchPostEnergyKernel(const int       numSystems,
                                           const bool      isFirstIter,
                                           const storageT* prevE,  // oldval
                                           const storageT* newE,   // newval
                                           const storageT* slopes,
                                           storageT*       eScratch,  // val2
                                           storageT*       lambdas,
                                           storageT*       lambda2s,
                                           int16_t*        statuses) {
  const int sysIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (sysIdx >= numSystems) {
    return;
  }
  // Finished run.
  if (statuses[sysIdx] != -2) {
    return;
  }

  const real slope  = static_cast<real>(slopes[sysIdx]);
  const real newVal = static_cast<real>(newE[sysIdx]);
  const real oldVal = static_cast<real>(prevE[sysIdx]);
  const real lambda = static_cast<real>(lambdas[sysIdx]);
  if (newVal - oldVal <= static_cast<real>(FUNCTOL) * lambda * slope) {
    // we're converged on the function:
    statuses[sysIdx] = 0;
    return;
  }
  // if we made it this far, we need to backtrack:
  real tmpLambda;
  if (isFirstIter) {
    // it's the first step:
    tmpLambda = -slope / (real{2} * (newVal - oldVal - slope));
  } else {
    const real val2    = static_cast<real>(eScratch[sysIdx]);
    const real lambda2 = static_cast<real>(lambda2s[sysIdx]);
    real       rhs1    = newVal - oldVal - lambda * slope;
    real       rhs2    = val2 - oldVal - lambda2 * slope;
    real       a       = (rhs1 / (lambda * lambda) - rhs2 / (lambda2 * lambda2)) / (lambda - lambda2);
    real       b = (-lambda2 * rhs1 / (lambda * lambda) + lambda * rhs2 / (lambda2 * lambda2)) / (lambda - lambda2);
    if (a == real{0}) {
      tmpLambda = -slope / (real{2} * b);
    } else {
      real disc = b * b - real{3} * a * slope;
      if (disc < real{0}) {
        tmpLambda = real{0.5} * lambda;
      } else if (b <= real{0}) {
        tmpLambda = (-b + bfgsSqrt(disc)) / (real{3} * a);
      } else {
        tmpLambda = -slope / (b + bfgsSqrt(disc));
      }
    }
    if (tmpLambda > real{0.5} * lambda) {
      tmpLambda = real{0.5} * lambda;
    }
  }
  lambda2s[sysIdx] = static_cast<storageT>(lambda);
  eScratch[sysIdx] = static_cast<storageT>(newVal);
  lambdas[sysIdx]  = static_cast<storageT>(max(tmpLambda, real{0.1} * lambda));
}

template <typename Workspace, typename storageT>
void BfgsBatchMinimizer::doLineSearchPostEnergyImpl(const int iter, const storageT* energies, Workspace& workspace) {
  using real          = typename Workspace::ComputeScalar;
  const int numBlocks = (numSystems_ + 127) / 128;
  lineSearchPostEnergyKernel<real, storageT><<<numBlocks, 128, 0, stream_>>>(numSystems_,
                                                                             iter == 0,
                                                                             workspace.lineSearchStoredEnergy.data(),
                                                                             energies,
                                                                             workspace.lineSearchSlope.data(),
                                                                             workspace.lineSearchEnergyScratch.data(),
                                                                             workspace.lineSearchLambdas.data(),
                                                                             workspace.lineSearchLambdas2.data(),
                                                                             lineSearchStatus_.data());
  cudaCheckError(cudaGetLastError());
}

void BfgsBatchMinimizer::doLineSearchPostEnergy(const int iter) {
  if (usesSinglePrecision(precision_)) {
    doLineSearchPostEnergyImpl(iter, singleWorkspace().energy.data(), singleWorkspace());
    return;
  }
  doLineSearchPostEnergyImpl(iter, energyOutsDevice, fullWorkspace());
}

template <typename storageT>
__global__ void lineSearchPostLoopKernel(const int*      atomStarts,
                                         int16_t*        statuses,
                                         const storageT* oldPos,
                                         storageT*       statePos,
                                         storageT*       evalPos,
                                         const int*      activeSystemIndices,
                                         const int       DIM) {
  const int       sysIdx      = activeSystemIndices[blockIdx.x];
  const int       idxInSys    = threadIdx.x;
  const int       numTerms    = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const storageT* oldPosStart = &oldPos[atomStarts[sysIdx] * DIM];
  storageT*       stateStart  = &statePos[atomStarts[sysIdx] * DIM];
  storageT*       evalStart   = &evalPos[atomStarts[sysIdx] * DIM];

  // Special handling of statuses needed here, to reproduce RDKit behavior which has either early returns
  // or loop breaks depending on the status. Note that "-2" is not a status in the RDKit code, but for us
  // it means reached the end of the loop, and should be a -1.
  const int16_t status              = statuses[sysIdx];
  // These are the two cases in the RDKit loop where this end section triggers. A -1 or 0 exits the function.
  const bool    needUpdatePositions = status == -2 || status == 1;
  // Match RDKit for end of loop case.
  if (status == -2) {
    if (threadIdx.x == 0) {
      statuses[sysIdx] = -1;
    }
  }
  if (needUpdatePositions) {
    for (int i = idxInSys; i < numTerms; i += blockDim.x) {
      stateStart[i] = static_cast<storageT>(oldPosStart[i]);
      evalStart[i]  = oldPosStart[i];
    }
  }
}

template <typename Workspace, typename storageT>
void BfgsBatchMinimizer::doLineSearchPostLoopImpl(const storageT* positions, Workspace& workspace) {
  lineSearchPostLoopKernel<storageT><<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                                                 lineSearchStatus_.data(),
                                                                                 positions,
                                                                                 workspace.scratchPositions.data(),
                                                                                 workspace.scratchPositions.data(),
                                                                                 activeSystemIndices_.data(),
                                                                                 dataDim_);
  cudaCheckError(cudaGetLastError());
}

void BfgsBatchMinimizer::doLineSearchPostLoop() {
  if (usesSinglePrecision(precision_)) {
    doLineSearchPostLoopImpl(singleWorkspace().positions.data(), singleWorkspace());
    return;
  }
  doLineSearchPostLoopImpl(positionsDevice, fullWorkspace());
}

struct NotEqualToMinusTwoFunctor {
  __host__ __device__ int operator()(const int16_t& x) const { return x != -2 ? 1 : 0; }
};

struct EqualsZeroFunctor {
  __host__ __device__ int operator()(const int16_t& x) const { return x == 0; }
};

BfgsBatchMinimizer::BfgsBatchMinimizer(const int     dataDim,
                                       DebugLevel    debugLevel,
                                       bool          scaleGrads,
                                       cudaStream_t  stream,
                                       BfgsBackend   backend,
                                       PrecisionMode precision)
    : countFinished_(0, stream) {
  debugLevel_ = debugLevel;
  dataDim_    = dataDim;
  scaleGrads_ = scaleGrads;
  stream_     = stream;
  backend_    = backend;
  precision_  = precision;
  if (usesSinglePrecision(precision_)) {
    singleWorkspace_ = std::make_unique<SingleBfgsWorkspace>();
    singleWorkspace_->setStream(stream_);
  } else {
    fullWorkspace_ = std::make_unique<FullBfgsWorkspace>();
    fullWorkspace_->setStream(stream_);
  }
  // For HYBRID, we need to support both paths, so initialize for both
  if (backend_ == BfgsBackend::BATCHED || backend_ == BfgsBackend::HYBRID) {
    loopStatusHost_.resize(1);
  }

  if (stream_ != nullptr) {
    activeSystemIndices_.setStream(stream_);
    allSystemIndices_.setStream(stream_);

    statuses_.setStream(stream_);

    lineSearchStatus_.setStream(stream_);
    countTempStorage_.setStream(stream_);
    countFinished_.setStream(stream_);
    hessianStarts_.setStream(stream_);
    activeMolIdsDevice_.setStream(stream_);
  }
}
BfgsBatchMinimizer::~BfgsBatchMinimizer() = default;

BfgsBackend BfgsBatchMinimizer::resolveBackend(const std::vector<int>& atomStartsHost) const {
  return nvMolKit::resolveBackend(backend_, atomStartsHost);
}

template <typename Workspace>
void BfgsBatchMinimizer::initializeImpl(const std::vector<int>& atomStartsHost,
                                        const int*              atomStarts,
                                        double*                 positions,
                                        double*                 grad,
                                        double*                 energyOuts,
                                        BfgsBackend             effectiveBackend,
                                        Workspace&              workspace,
                                        const uint8_t*          activeThisStage) {
  atomStartsDevice = atomStarts;
  positionsDevice  = positions;
  gradDevice       = grad;
  energyOutsDevice = energyOuts;

  const int numSystems = atomStartsHost.size() - 1;
  activeHost_.resize(numSystems);
  convergenceHost_.resize(numSystems);

  statuses_.resize(numSystems);
  if (activeThisStage) {
    // Copy activeThisStage to statuses_ with type conversion
    const int blockSize = 128;
    const int numBlocks = (numSystems + blockSize - 1) / blockSize;
    copyActiveToStatusKernel<<<numBlocks, blockSize, 0, stream_>>>(numSystems, activeThisStage, statuses_.data());
  } else {
    // Default initialization to all 1s
    setAll(statuses_, static_cast<int16_t>(1));
  }
  cudaCheckError(cudaGetLastError());

  numSystems_     = numSystems;
  numAtomsTotal_  = atomStartsHost.back();
  hasLargeSystem_ = false;

  if (effectiveBackend == BfgsBackend::PER_MOLECULE) {
    // Copy activeThisStage to host for CPU-side filtering (using pinned memory)
    std::fill_n(activeHost_.begin(), numSystems, 1);
    if (activeThisStage) {
      cudaCheckError(cudaMemcpyAsync(activeHost_.data(),
                                     activeThisStage,
                                     numSystems * sizeof(uint8_t),
                                     cudaMemcpyDeviceToHost,
                                     stream_));
      cudaCheckError(cudaStreamSynchronize(stream_));
    }

    activeMolIds_.clear();
    maxAtomsInBatch_ = 0;

    for (int i = 0; i < numSystems_; ++i) {
      if (activeHost_[i] == 0) {
        continue;
      }

      const int numAtoms = atomStartsHost[i + 1] - atomStartsHost[i];
      activeMolIds_.push_back(i);

      if (numAtoms > maxAtomsInBatch_) {
        maxAtomsInBatch_ = numAtoms;
      }
      if (numAtoms > 256) {
        hasLargeSystem_ = true;
      }
    }

    // Transfer active molecule list to device
    if (!activeMolIds_.empty()) {
      activeMolIdsDevice_.resize(activeMolIds_.size());
      activeMolIdsDevice_.setFromVector(activeMolIds_);
    }
  } else {
    // Original logic for batched backend
    for (int i = 0; i < numSystems_; ++i) {
      const int numAtoms = atomStartsHost[i + 1] - atomStartsHost[i];
      if (numAtoms > 256) {
        hasLargeSystem_ = true;
        break;
      }
    }
  }

  hessianStartsHost_.clear();
  hessianStartsHost_.reserve(numSystems + 1);
  hessianStartsHost_.push_back(0);
  for (int i = 0; i < numSystems; ++i) {
    const int numAtoms = atomStartsHost[i + 1] - atomStartsHost[i];
    // Note - hessian starts is total term based, not atom based.
    const int numTerms = (dataDim_ * numAtoms) * (dataDim_ * numAtoms);
    hessianStartsHost_.push_back(hessianStartsHost_.back() + numTerms);
  }
  hessianStarts_.resize(numSystems + 1);
  hessianStarts_.setFromVector(hessianStartsHost_);
  workspace.inverseHessian.resize(hessianStartsHost_.back());
  workspace.inverseHessian.zero();

  const int numStateTerms = atomStartsHost.back() * dataDim_;
  workspace.scratchPositions.resize(numStateTerms);
  workspace.scratchPositions.zero();
  workspace.lineSearchDir.resize(numStateTerms);
  workspace.scratchGrad.resize(numStateTerms);
  workspace.hessDGrad.resize(numStateTerms);
  workspace.hessDGrad.zero();

  if (effectiveBackend == BfgsBackend::PER_MOLECULE) {
    return;
  }

  activeSystemIndices_.resize(numSystems_);
  allSystemIndices_.resize(numSystems_);
  systemIndicesHost_.resize(numSystems_);
  std::iota(systemIndicesHost_.begin(), systemIndicesHost_.end(), 0);
  allSystemIndices_.setFromVector(systemIndicesHost_);
  activeSystemIndices_.setFromVector(systemIndicesHost_);

  lineSearchStatus_.resize(numSystems);
  if constexpr (std::is_same_v<typename Workspace::StorageScalar, float>) {
    workspace.energy.resize(numSystems);
    workspace.positions.resize(numStateTerms);
    workspace.grad.resize(numStateTerms);
    if (positions != nullptr) {
      cudaCheckError(detail::convertDeviceArray(workspace.positions.data(), positions, numStateTerms, stream_));
    } else {
      workspace.positions.zero();
    }
    workspace.grad.zero();
  }
  workspace.gradScales.resize(numSystems);
  workspace.lineSearchLambdaMins.resize(numSystems);
  workspace.lineSearchLambdas.resize(numSystems);
  workspace.lineSearchLambdas2.resize(numSystems);
  workspace.lineSearchSlope.resize(numSystems);
  workspace.lineSearchMaxSteps.resize(numSystems);
  workspace.lineSearchStoredEnergy.resize(numSystems);
  workspace.lineSearchEnergyScratch.resize(numSystems);

  // Compute needed reduction storage.
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::TransformReduce(nullptr,
                                     temp_storage_bytes,
                                     lineSearchStatus_.data(),
                                     countFinished_.data(),
                                     lineSearchStatus_.size(),
                                     cubSum(),
                                     NotEqualToMinusTwoFunctor(),
                                     0,
                                     stream_);
  countTempStorage_.resize(temp_storage_bytes);

  cub::DeviceSelect::Flagged(nullptr,
                             temp_storage_bytes,
                             allSystemIndices_.data(),
                             statuses_.data(),
                             activeSystemIndices_.data(),
                             countFinished_.data(),
                             statuses_.size(),
                             stream_);

  if (temp_storage_bytes > countTempStorage_.size()) {
    countTempStorage_.zero();
    countTempStorage_.resize(temp_storage_bytes);
  }
}

void BfgsBatchMinimizer::initialize(const std::vector<int>& atomStartsHost,
                                    const int*              atomStarts,
                                    double*                 positions,
                                    double*                 grad,
                                    double*                 energyOuts,
                                    BfgsBackend             effectiveBackend,
                                    const uint8_t*          activeThisStage) {
  if (usesSinglePrecision(precision_)) {
    initializeImpl(atomStartsHost,
                   atomStarts,
                   positions,
                   grad,
                   energyOuts,
                   effectiveBackend,
                   singleWorkspace(),
                   activeThisStage);
    return;
  }
  initializeImpl(atomStartsHost,
                 atomStarts,
                 positions,
                 grad,
                 energyOuts,
                 effectiveBackend,
                 fullWorkspace(),
                 activeThisStage);
}

template <typename storageT>
__global__ void populateHessianIdentityKernel(const int* hessianStarts,
                                              const int* atomStarts,
                                              storageT*  inverseHessian,
                                              const int  DIM) {
  const int sysIdx        = blockIdx.x;
  const int idxInSys      = threadIdx.x;
  const int writeStartIdx = hessianStarts[sysIdx];
  const int numTerms      = hessianStarts[sysIdx + 1] - hessianStarts[sysIdx];
  const int numAtoms      = atomStarts[sysIdx + 1] - atomStarts[sysIdx];
  const int rowLength     = DIM * numAtoms;

  for (int i = idxInSys; i < rowLength; i += blockDim.x) {
    if (i < numTerms) {
      inverseHessian[writeStartIdx + i * rowLength + i] = static_cast<storageT>(1.0);
    }
  }
}

template <typename Workspace> void BfgsBatchMinimizer::setHessianToIdentityImpl(Workspace& workspace) {
  constexpr int blockDim  = 128;
  const int     numBlocks = hessianStarts_.size() - 1;
  workspace.inverseHessian.zero();
  populateHessianIdentityKernel<<<numBlocks, blockDim, 0, stream_>>>(hessianStarts_.data(),
                                                                     atomStartsDevice,
                                                                     workspace.inverseHessian.data(),
                                                                     dataDim_);
  cudaCheckError(cudaGetLastError());
}

void BfgsBatchMinimizer::setHessianToIdentity() {
  if (usesSinglePrecision(precision_)) {
    setHessianToIdentityImpl(singleWorkspace());
    return;
  }
  setHessianToIdentityImpl(fullWorkspace());
}

template <typename real, typename reduceT, typename storageT>
__global__ void setMaxStepKernel(const int* atomStarts, const storageT* positions, storageT* maxSteps, const int DIM) {
  const int  sysIdx        = blockIdx.x;
  const int  idxInSys      = threadIdx.x;
  const bool isFirstThread = threadIdx.x == 0;

  const int       numTerms = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const storageT* posStart = &positions[atomStarts[sysIdx] * DIM];

  reduceT sumSquaredPos = 0;
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    const real x = static_cast<real>(posStart[i]);
    sumSquaredPos += static_cast<reduceT>(x * x);
  }

  using BlockReduce = cub::BlockReduce<reduceT, 128>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  const reduceT squaredSum = BlockReduce(tempStorage).Sum(sumSquaredPos);
  if (isFirstThread) {
    constexpr real maxStepFactor = real{100};
    maxSteps[sysIdx] =
      static_cast<storageT>(maxStepFactor * max(static_cast<real>(bfgsSqrt(squaredSum)), static_cast<real>(numTerms)));
  }
}

template <typename Workspace, typename storageT>
void BfgsBatchMinimizer::setMaxStepImpl(const storageT* positions, Workspace& workspace) {
  using real    = typename Workspace::ComputeScalar;
  using reduceT = typename Workspace::ReductionScalar;
  setMaxStepKernel<real, reduceT, storageT>
    <<<numSystems_, 128, 0, stream_>>>(atomStartsDevice, positions, workspace.lineSearchMaxSteps.data(), dataDim_);
  cudaCheckError(cudaGetLastError());
}

void BfgsBatchMinimizer::setMaxStep() {
  if (usesSinglePrecision(precision_)) {
    setMaxStepImpl(singleWorkspace().positions.data(), singleWorkspace());
    return;
  }
  setMaxStepImpl(positionsDevice, fullWorkspace());
}

namespace {

template <typename sourceT, typename storageT>
__global__ void copyAndNegate(const int numElements, const sourceT* src, storageT* dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = static_cast<storageT>(-src[idx]);
  }
}

template <typename storageT>
void prepareScratchBuffers(AsyncDeviceVector<storageT>&  grad,
                           AsyncDeviceVector<storageT>&  lineSearchDir,
                           AsyncDeviceVector<storageT>&  scratchPositions,
                           AsyncDeviceVector<storageT>&  hessDGrad,
                           AsyncDeviceVector<storageT>&  scratchGrad,
                           AsyncDeviceVector<storageT*>& scratchBuffersDevice,
                           PinnedHostVector<storageT*>&  scratchBufferPointersHost,
                           cudaStream_t                  stream) {
  scratchBuffersDevice.resize(5);
  scratchBufferPointersHost.resize(5);
  scratchBufferPointersHost[0] = grad.data();
  scratchBufferPointersHost[1] = lineSearchDir.data();
  scratchBufferPointersHost[2] = scratchPositions.data();
  scratchBufferPointersHost[3] = hessDGrad.data();
  scratchBufferPointersHost[4] = scratchGrad.data();

  cudaCheckError(cudaMemcpyAsync(scratchBuffersDevice.data(),
                                 scratchBufferPointersHost.data(),
                                 5 * sizeof(storageT*),
                                 cudaMemcpyHostToDevice,
                                 stream));
}

bool checkConvergence(const std::vector<int>&     activeMolIds,
                      AsyncDeviceVector<int16_t>& statuses,
                      PinnedHostVector<int16_t>&  convergenceHost,
                      const int                   numSystems,
                      cudaStream_t                stream) {
  statuses.copyToHost(convergenceHost.data(), numSystems);
  cudaCheckError(cudaStreamSynchronize(stream));

  for (const int molIdx : activeMolIds) {
    if (convergenceHost[molIdx] != 0) {
      return true;
    }
  }
  return false;
}

}  // namespace

int BfgsBatchMinimizer::lineSearchCountFinished() const {
  size_t temp_storage_bytes = countTempStorage_.size();
  cub::DeviceReduce::TransformReduce(countTempStorage_.data(),
                                     temp_storage_bytes,
                                     lineSearchStatus_.data(),
                                     countFinished_.data(),
                                     lineSearchStatus_.size(),
                                     cubSum(),
                                     NotEqualToMinusTwoFunctor(),
                                     0,
                                     stream_);
  int& finishedHost = loopStatusHost_[0];
  countFinished_.get(finishedHost);
  cudaStreamSynchronize(stream_);
  return finishedHost;
}

int BfgsBatchMinimizer::compactAndCountConverged() const {
  const ScopedNvtxRange bfgsCompactAndCountConverged("BfgsBatchMinimizer::compactAndCountConverged");
  size_t                temp_storage_bytes = countTempStorage_.size();

  cudaCheckError(cub::DeviceSelect::Flagged(countTempStorage_.data(),
                                            temp_storage_bytes,
                                            allSystemIndices_.data(),
                                            statuses_.data(),
                                            activeSystemIndices_.data(),
                                            countFinished_.data(),
                                            statuses_.size(),
                                            stream_));
  // std::vector<int> allHost(numSystems_);
  // std::vector<int> allCompact(numSystems_);
  // std::vector<int16_t> statusHost(numSystems_);
  // statuses_.copyToHost(statusHost);
  // allSystemIndices_.copyToHost(allHost);
  // activeSystemIndices_.copyToHost(allCompact);
  int& unfinishedHost = loopStatusHost_[0];
  countFinished_.get(unfinishedHost);
  cudaStreamSynchronize(stream_);
  numUnfinishedSystems_ = unfinishedHost;
  return numSystems_ - unfinishedHost;
}

template <typename real, typename reduceT, typename storageT>
__global__ void setDirectionKernel(const int*      atomStarts,
                                   const storageT* positionsFromLineSearch,
                                   const storageT* grads,
                                   storageT*       xis,
                                   storageT*       positions,
                                   storageT*       dGrads,
                                   int16_t*        statuses,
                                   const int*      activeSystemIndices,
                                   const int       DIM) {
  const int sysIdx          = activeSystemIndices[blockIdx.x];
  const int idxWithinSystem = threadIdx.x;
  const int numTerms        = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const int startIdx        = atomStarts[sysIdx] * DIM;

  if (statuses[sysIdx] == 0) {
    return;
  }

  storageT*       localXi            = &xis[startIdx];
  storageT*       localPos           = &positions[startIdx];
  const storageT* localPosLineSearch = &positionsFromLineSearch[startIdx];
  const storageT* localGrad          = &grads[startIdx];
  storageT*       localDGrad         = &dGrads[startIdx];

  reduceT localMax = 0;
  for (int i = idxWithinSystem; i < numTerms; i += blockDim.x) {
    const real xi = static_cast<real>(localPosLineSearch[i]) - static_cast<real>(localPos[i]);
    localXi[i]    = static_cast<storageT>(xi);
    localPos[i]   = localPosLineSearch[i];
    localDGrad[i] = static_cast<storageT>(localGrad[i]);

    const real temp = bfgsAbs(xi) / max(bfgsAbs(static_cast<real>(localPos[i])), real{1});
    // TODO we could have a better thread distribution pattern for the local Max.
    if (temp > localMax) {
      localMax = temp;
    }
  }

  __shared__ typename cub::BlockReduce<reduceT, 128>::TempStorage tempStorage;
  const reduceT     blockMax = cub::BlockReduce<reduceT, 128>(tempStorage).Reduce(localMax, cubMax());
  constexpr reduceT TOLX     = static_cast<reduceT>(4. * 3e-8);
  if (idxWithinSystem == 0 && blockMax < TOLX) {
    // Converged
    statuses[sysIdx] = 0;
  }
}

template <typename Workspace, typename storageT>
void BfgsBatchMinimizer::setDirectionImpl(storageT* positions, storageT* grads, Workspace& workspace) {
  const ScopedNvtxRange bfgsSetDirection("BfgsBatchMinimizer::setDirection");
  using real    = typename Workspace::ComputeScalar;
  using reduceT = typename Workspace::ReductionScalar;
  setDirectionKernel<real, reduceT, storageT>
    <<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                 workspace.scratchPositions.data(),
                                                 grads,
                                                 workspace.lineSearchDir.data(),
                                                 positions,
                                                 workspace.scratchGrad.data(),
                                                 statuses_.data(),
                                                 activeSystemIndices_.data(),
                                                 dataDim_);
  cudaCheckError(cudaGetLastError());
}

void BfgsBatchMinimizer::setDirection() {
  if (usesSinglePrecision(precision_)) {
    setDirectionImpl(singleWorkspace().positions.data(), singleWorkspace().grad.data(), singleWorkspace());
    return;
  }
  setDirectionImpl(positionsDevice, gradDevice, fullWorkspace());
}

// Mirrors RDKit's ForceField::minimize gradient cap (calcGradient in
// Code/ForceField/ForceField.cpp). RDKit historically tracked the signed max of
// gradient components; commit 5b1d04d23 (RDKit 2025.09) switched to |grad|.
// Follow whichever rule the linked RDKit uses so weighted MMFF/UFF minimization
// trajectories agree with the host reference.
template <bool scaleGrads, typename real, typename reduceT, typename storageT>
__global__ void scaleGradKernel(const int16_t* statuses,
                                const int*     atomStarts,
                                storageT*      grads,
                                storageT*      gradScales,
                                const int*     activeSystemIndices,
                                const int      DIM) {
  constexpr bool kRdkitHasGradScaleFix =
    RDKIT_VERSION_MAJOR > 2025 || (RDKIT_VERSION_MAJOR == 2025 && RDKIT_VERSION_MINOR >= 9);
  const int sysIdx          = activeSystemIndices == nullptr ? blockIdx.x : activeSystemIndices[blockIdx.x];
  const int idxWithinSystem = threadIdx.x;
  const int numTerms        = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);

  if (statuses[sysIdx] == 0) {
    return;
  }

  storageT* localGrad = &grads[atomStarts[sysIdx] * DIM];

  reduceT            maxGrad   = kRdkitHasGradScaleFix ? reduceT{0} : reduceT{-1e8};
  real               gradScale = scaleGrads ? real{0.1} : real{1};
  __shared__ reduceT distributedMax[1];
  if (idxWithinSystem == 0) {
    distributedMax[0] = -1.0;  // See note at start at function, this will work for now.
  }

  for (int i = idxWithinSystem; i < numTerms; i += blockDim.x) {
    const real scaled = static_cast<real>(localGrad[i]) * gradScale;
    localGrad[i]      = scaled;
    const real cmp    = kRdkitHasGradScaleFix ? bfgsAbs(scaled) : scaled;
    if (cmp > maxGrad) {
      maxGrad = cmp;
    }
  }

  __shared__ typename cub::BlockReduce<reduceT, 128>::TempStorage tempStorage;
  const reduceT blockMax = cub::BlockReduce<reduceT, 128>(tempStorage).Reduce(maxGrad, cubMax());

  if (idxWithinSystem == 0) {
    distributedMax[0] = blockMax;
  }
  __syncthreads();
  maxGrad = distributedMax[0];

  if (scaleGrads && maxGrad > 10.0) {
    while (maxGrad * gradScale > 10.0) {
      gradScale *= .5;
    }
    for (int i = idxWithinSystem; i < numTerms; i += blockDim.x) {
      localGrad[i] = static_cast<real>(localGrad[i]) * gradScale;
    }
  }
  if (idxWithinSystem == 0) {
    gradScales[sysIdx] = static_cast<storageT>(gradScale);
  }
}

template <typename Workspace, typename storageT>
void BfgsBatchMinimizer::scaleGradImpl(const bool preLoop, storageT* grads, Workspace& workspace) {
  const int  numSystems          = preLoop ? numSystems_ : numUnfinishedSystems_;
  const int* activeSystemIndices = preLoop ? nullptr : activeSystemIndices_.data();
  using real                     = typename Workspace::ComputeScalar;
  using reduceT                  = typename Workspace::ReductionScalar;
  if (scaleGrads_) {
    scaleGradKernel<true, real, reduceT, storageT><<<numSystems, 128, 0, stream_>>>(statuses_.data(),
                                                                                    atomStartsDevice,
                                                                                    grads,
                                                                                    workspace.gradScales.data(),
                                                                                    activeSystemIndices,
                                                                                    dataDim_);
  } else {
    scaleGradKernel<false, real, reduceT, storageT><<<numSystems, 128, 0, stream_>>>(statuses_.data(),
                                                                                     atomStartsDevice,
                                                                                     grads,
                                                                                     workspace.gradScales.data(),
                                                                                     activeSystemIndices,
                                                                                     dataDim_);
  }
}

void BfgsBatchMinimizer::scaleGrad(const bool preLoop) {
  if (usesSinglePrecision(precision_)) {
    scaleGradImpl(preLoop, singleWorkspace().grad.data(), singleWorkspace());
    return;
  }
  scaleGradImpl(preLoop, gradDevice, fullWorkspace());
}

template <typename real, typename reduceT, typename storageT>
__global__ void updateDGradKernel(const storageT  gradTol,
                                  const int*      atomStarts,
                                  const storageT* energies,
                                  const storageT* gradScales,
                                  const storageT* grads,
                                  const storageT* positions,
                                  storageT*       dGrads,
                                  int16_t*        statuses,
                                  const int*      activeSystemIndices,
                                  const int       DIM) {
  const int sysIdx          = activeSystemIndices[blockIdx.x];
  const int idxWithinSystem = threadIdx.x;
  const int numTerms        = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const int startIdx        = atomStarts[sysIdx] * DIM;

  if (statuses[sysIdx] == 0) {
    return;
  }

  const storageT* localGrad = &grads[startIdx];

  const storageT* localPos   = &positions[startIdx];
  storageT*       localDGrad = &dGrads[startIdx];

  reduceT localMax = 0;

  for (int i = idxWithinSystem; i < numTerms; i += blockDim.x) {
    const real gradValue = static_cast<real>(localGrad[i]);
    localDGrad[i]        = static_cast<storageT>(gradValue - static_cast<real>(localDGrad[i]));
    const real temp      = bfgsAbs(gradValue) * max(bfgsAbs(static_cast<real>(localPos[i])), real{1});
    // TODO we could have a better thread distribution pattern for the local Max.
    if (temp > localMax) {
      localMax = temp;
    }
  }
  __shared__ typename cub::BlockReduce<reduceT, 128>::TempStorage tempStorage;
  reduceT blockMax = cub::BlockReduce<reduceT, 128>(tempStorage).Reduce(localMax, cubMax());

  if (idxWithinSystem == 0) {
    // rdkit/rdkit#9298 (merged RDKit 2026.03) fixed the signed-energy denominator bug:
    // raw negative energy clamped the denominator to 1, artificially tightening gradTol.
    // Use |energy| when linked against a fixed RDKit; keep signed otherwise for parity.
    constexpr bool kRdkitHasGradDenomFix =
      RDKIT_VERSION_MAJOR > 2026 || (RDKIT_VERSION_MAJOR == 2026 && RDKIT_VERSION_MINOR >= 3);
    const real energyValue = static_cast<real>(energies[sysIdx]);
    const real energyMag   = kRdkitHasGradDenomFix ? bfgsAbs(energyValue) : energyValue;
    const real term        = max(energyMag * static_cast<real>(gradScales[sysIdx]), real{1});
    blockMax /= term;
    if (blockMax < static_cast<reduceT>(gradTol)) {
      // Converged
      statuses[sysIdx] = 0;
    }
  }
}

template <typename Workspace, typename storageT>
void BfgsBatchMinimizer::updateDGradImpl(const storageT* energies,
                                         const storageT* grads,
                                         const storageT* positions,
                                         Workspace&      workspace) {
  const ScopedNvtxRange bfgsUpdateDGrad("BfgsBatchMinimizer::updateDGrad");
  using real    = typename Workspace::ComputeScalar;
  using reduceT = typename Workspace::ReductionScalar;
  updateDGradKernel<real, reduceT, storageT>
    <<<numUnfinishedSystems_, 128, 0, stream_>>>(static_cast<storageT>(gradTol_),
                                                 atomStartsDevice,
                                                 energies,
                                                 workspace.gradScales.data(),
                                                 grads,
                                                 positions,
                                                 workspace.scratchGrad.data(),
                                                 statuses_.data(),
                                                 activeSystemIndices_.data(),
                                                 dataDim_);
  cudaCheckError(cudaGetLastError());
}

void BfgsBatchMinimizer::updateDGrad() {
  if (usesSinglePrecision(precision_)) {
    updateDGradImpl(singleWorkspace().energy.data(),
                    singleWorkspace().grad.data(),
                    singleWorkspace().positions.data(),
                    singleWorkspace());
    return;
  }
  updateDGradImpl(energyOutsDevice, gradDevice, positionsDevice, fullWorkspace());
}

template <typename storageT>
void updateHessianState(int             numUnfinishedSystems,
                        const int16_t*  statuses,
                        const int*      hessianStarts,
                        const int*      atomStarts,
                        storageT*       inverseHessian,
                        storageT*       dGrads,
                        storageT*       dirs,
                        storageT*       hessDGrads,
                        const storageT* grads,
                        int             dataDim,
                        bool            largeMol,
                        const int*      activeSystemIndices,
                        cudaStream_t    stream) {
  nvMolKit::updateInverseHessianBFGSBatch(numUnfinishedSystems,
                                          statuses,
                                          hessianStarts,
                                          atomStarts,
                                          inverseHessian,
                                          dGrads,
                                          dirs,
                                          hessDGrads,
                                          grads,
                                          dataDim,
                                          largeMol,
                                          activeSystemIndices,
                                          stream);
}

template <typename Workspace, typename storageT>
void BfgsBatchMinimizer::updateHessianImpl(const storageT* grads, Workspace& workspace) {
  const ScopedNvtxRange bfgsUpdateHessian("BfgsBatchMinimizer::updateHessian");
  updateHessianState(numUnfinishedSystems_,
                     statuses_.data(),
                     hessianStarts_.data(),
                     atomStartsDevice,
                     workspace.inverseHessian.data(),
                     workspace.scratchGrad.data(),
                     workspace.lineSearchDir.data(),
                     workspace.hessDGrad.data(),
                     grads,
                     dataDim_,
                     hasLargeSystem_,
                     activeSystemIndices_.data(),
                     stream_);
}

void BfgsBatchMinimizer::updateHessian() {
  if (usesSinglePrecision(precision_)) {
    updateHessianImpl(singleWorkspace().grad.data(), singleWorkspace());
    return;
  }
  updateHessianImpl(gradDevice, fullWorkspace());
}

template <typename storageT> void BfgsBatchMinimizer::collectDebugDataImpl(const storageT* energies) {
  if (debugLevel_ != DebugLevel::STEPWISE) {
    return;
  }

  std::vector<int16_t> statusesHost(numSystems_);
  std::vector<double>  energiesHost(numSystems_);
  cudaCheckError(cudaMemcpyAsync(statusesHost.data(),
                                 statuses_.data(),
                                 numSystems_ * sizeof(int16_t),
                                 cudaMemcpyDeviceToHost,
                                 stream_));
  if constexpr (std::is_same_v<storageT, float>) {
    std::vector<float> singleEnergiesHost(numSystems_);
    cudaCheckError(cudaMemcpyAsync(singleEnergiesHost.data(),
                                   energies,
                                   numSystems_ * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   stream_));
    cudaCheckError(cudaStreamSynchronize(stream_));
    std::transform(singleEnergiesHost.begin(), singleEnergiesHost.end(), energiesHost.begin(), [](const float value) {
      return static_cast<double>(value);
    });
  } else {
    cudaCheckError(
      cudaMemcpyAsync(energiesHost.data(), energies, numSystems_ * sizeof(double), cudaMemcpyDeviceToHost, stream_));
    cudaCheckError(cudaStreamSynchronize(stream_));
  }

  stepwiseStatuses.push_back(std::move(statusesHost));
  stepwiseEnergies.push_back(std::move(energiesHost));
}

void BfgsBatchMinimizer::collectDebugData() {
  if (usesSinglePrecision(precision_)) {
    collectDebugDataImpl(singleWorkspace().energy.data());
    return;
  }
  collectDebugDataImpl(energyOutsDevice);
}

template <typename Workspace, typename storageT, typename EnergyEvaluator, typename GradientEvaluator>
bool BfgsBatchMinimizer::minimizeImpl(const int                  numIters,
                                      const double               gradTol,
                                      AsyncDeviceVector<double>& publicPositions,
                                      AsyncDeviceVector<double>& publicGrad,
                                      AsyncDeviceVector<double>& publicEnergies,
                                      storageT*                  positions,
                                      storageT*                  grad,
                                      storageT*                  energies,
                                      Workspace&                 workspace,
                                      EnergyEvaluator            evaluateEnergy,
                                      GradientEvaluator          evaluateGradient) {
  gradTol_             = gradTol;
  const int numSystems = numSystems_;

  {
    const ScopedNvtxRange bfgsFullInitialize("BfgsBatchMinimizer::fullInitialize");
    setHessianToIdentityImpl(workspace);

    cudaCheckError(cudaMemsetAsync(energies, 0, numSystems * sizeof(storageT), stream_));
    evaluateEnergy(positions);
    cudaCheckError(cudaMemsetAsync(grad, 0, publicGrad.size() * sizeof(storageT), stream_));
    evaluateGradient();
    scaleGradImpl(/*preLoop=*/true, grad, workspace);

    collectDebugDataImpl(energies);
    constexpr int copyBlockSize = 128;
    const int     copyBlocks    = (publicGrad.size() + copyBlockSize - 1) / copyBlockSize;
    copyAndNegate<<<copyBlocks, copyBlockSize, 0, stream_>>>(publicGrad.size(), grad, workspace.lineSearchDir.data());
    cudaCheckError(cudaGetLastError());
    setMaxStepImpl(positions, workspace);
  }

  for (int currIter = 0; currIter < numIters && compactAndCountConverged() < numSystems; currIter++) {
    {
      const ScopedNvtxRange bfgsLineSearch("BfgsBatchMinimizer::lineSearch");
      doLineSearchSetupImpl(energies, positions, grad, workspace);

      int              lineSearchIter         = 0;
      constexpr double MAX_ITER_LINEAR_SEARCH = 1000;
      while (lineSearchIter < MAX_ITER_LINEAR_SEARCH && lineSearchCountFinished() < numSystems) {
        doLineSearchPerturbImpl(positions, workspace);
        cudaCheckError(cudaMemsetAsync(energies, 0, numSystems * sizeof(storageT), stream_));
        evaluateEnergy(workspace.scratchPositions.data());
        doLineSearchPostEnergyImpl(lineSearchIter, energies, workspace);
        lineSearchIter++;
      }
      doLineSearchPostLoopImpl(positions, workspace);
    }
    setDirectionImpl(positions, grad, workspace);

    {
      const ScopedNvtxRange bfgsGetAndScaleGrad("BfgsBatchMinimizer::getAndScaleGrad");
      cudaCheckError(cudaMemsetAsync(grad, 0, publicGrad.size() * sizeof(storageT), stream_));
      evaluateGradient();
      scaleGradImpl(/*preLoop=*/false, grad, workspace);
    }

    updateDGradImpl(energies, grad, positions, workspace);
    updateHessianImpl(grad, workspace);
    collectDebugDataImpl(energies);
  }

  cudaCheckError(cudaMemsetAsync(energies, 0, numSystems * sizeof(storageT), stream_));
  evaluateEnergy(positions);
  if constexpr (std::is_same_v<storageT, float>) {
    cudaCheckError(detail::convertDeviceArray(publicEnergies.data(), energies, publicEnergies.size(), stream_));
    cudaCheckError(detail::convertDeviceArray(publicPositions.data(), positions, publicPositions.size(), stream_));
    cudaCheckError(detail::convertDeviceArray(publicGrad.data(), grad, publicGrad.size(), stream_));
  }
  return compactAndCountConverged() == numSystems ? 0 : 1;
}

bool BfgsBatchMinimizer::minimize(const int                  numIters,
                                  const double               gradTol,
                                  BatchedForcefield&         ff,
                                  AsyncDeviceVector<double>& positions,
                                  AsyncDeviceVector<double>& grad,
                                  AsyncDeviceVector<double>& energyOuts,
                                  const uint8_t*             activeSystemMask) {
  const auto& atomStartsHost = ff.atomStartsHost();

  if (resolveBackend(atomStartsHost) != BfgsBackend::BATCHED) {
    throw std::runtime_error("BatchedForcefield minimization is only supported on the BATCHED backend");
  }

  if (usesSinglePrecision(precision_)) {
    initializeImpl(atomStartsHost,
                   ff.atomStartsDevice(),
                   positions.data(),
                   grad.data(),
                   energyOuts.data(),
                   BfgsBackend::BATCHED,
                   singleWorkspace(),
                   activeSystemMask);

    auto* singlePrecisionForcefield = dynamic_cast<SinglePrecisionBatchedForcefield*>(&ff);
    if (singlePrecisionForcefield == nullptr) {
      auto evaluateEnergy = [&](const float* evalPositions) {
        cudaCheckError(detail::convertDeviceArray(positions.data(), evalPositions, positions.size(), stream_));
        energyOuts.zero();
        cudaCheckError(ff.computeEnergy(energyOuts.data(), positions.data(), activeSystemMask, stream_));
        cudaCheckError(
          detail::convertDeviceArray(singleWorkspace().energy.data(), energyOuts.data(), energyOuts.size(), stream_));
      };
      auto evaluateGradient = [&]() {
        cudaCheckError(
          detail::convertDeviceArray(positions.data(), singleWorkspace().positions.data(), positions.size(), stream_));
        grad.zero();
        cudaCheckError(ff.computeGradients(grad.data(), positions.data(), activeSystemMask, stream_));
        cudaCheckError(detail::convertDeviceArray(singleWorkspace().grad.data(), grad.data(), grad.size(), stream_));
      };
      return minimizeImpl(numIters,
                          gradTol,
                          positions,
                          grad,
                          energyOuts,
                          singleWorkspace().positions.data(),
                          singleWorkspace().grad.data(),
                          singleWorkspace().energy.data(),
                          singleWorkspace(),
                          evaluateEnergy,
                          evaluateGradient);
    }

    auto evaluateEnergy = [&](const float* evalPositions) {
      cudaCheckError(singlePrecisionForcefield->computeEnergy(singleWorkspace().energy.data(),
                                                              evalPositions,
                                                              activeSystemMask,
                                                              stream_));
    };
    auto evaluateGradient = [&]() {
      cudaCheckError(singlePrecisionForcefield->computeGradients(singleWorkspace().grad.data(),
                                                                 singleWorkspace().positions.data(),
                                                                 activeSystemMask,
                                                                 stream_));
    };
    return minimizeImpl(numIters,
                        gradTol,
                        positions,
                        grad,
                        energyOuts,
                        singleWorkspace().positions.data(),
                        singleWorkspace().grad.data(),
                        singleWorkspace().energy.data(),
                        singleWorkspace(),
                        evaluateEnergy,
                        evaluateGradient);
  }

  initializeImpl(atomStartsHost,
                 ff.atomStartsDevice(),
                 positions.data(),
                 grad.data(),
                 energyOuts.data(),
                 BfgsBackend::BATCHED,
                 fullWorkspace(),
                 activeSystemMask);
  auto evaluateEnergy = [&](const double* evalPositions) {
    cudaCheckError(ff.computeEnergy(energyOuts.data(), evalPositions, activeSystemMask, stream_));
  };
  auto evaluateGradient = [&]() {
    cudaCheckError(ff.computeGradients(grad.data(), positions.data(), activeSystemMask, stream_));
  };
  return minimizeImpl(numIters,
                      gradTol,
                      positions,
                      grad,
                      energyOuts,
                      positions.data(),
                      grad.data(),
                      energyOuts.data(),
                      fullWorkspace(),
                      evaluateEnergy,
                      evaluateGradient);
}

template <typename DeviceBuffers, typename Workspace>
bool BfgsBatchMinimizer::minimizeWithMMFFImpl(const int               numIters,
                                              const double            gradTol,
                                              const std::vector<int>& atomStartsHost,
                                              DeviceBuffers&          systemDevice,
                                              Workspace&              workspace,
                                              const uint8_t*          activeThisStage) {
  const int         numSystems       = atomStartsHost.size() - 1;
  const BfgsBackend effectiveBackend = resolveBackend(atomStartsHost);

  if (effectiveBackend == BfgsBackend::BATCHED) {
    throw std::runtime_error("Use minimize(..., BatchedForcefield&) for batched MMFF minimization");
  }

  initializeImpl(atomStartsHost,
                 systemDevice.indices.atomStarts.data(),
                 nullptr,
                 nullptr,
                 nullptr,
                 effectiveBackend,
                 workspace,
                 activeThisStage);

  const ScopedNvtxRange bfgsPerMolecule("BfgsBatchMinimizer::perMoleculeMinimize");

  prepareScratchBuffers(systemDevice.grad,
                        workspace.lineSearchDir,
                        workspace.scratchPositions,
                        workspace.hessDGrad,
                        workspace.scratchGrad,
                        workspace.scratchBufferPointers,
                        workspace.scratchBufferPointersHost,
                        stream_);

  auto terms         = MMFF::toEnergyForceContribsDevicePtr(systemDevice);
  auto systemIndices = MMFF::toBatchedIndicesDevicePtr(systemDevice);

  const cudaError_t err = launchBfgsMinimizePerMolKernel(static_cast<int>(activeMolIds_.size()),
                                                         activeMolIdsDevice_.data(),
                                                         maxAtomsInBatch_,
                                                         systemDevice.indices.atomStarts.data(),
                                                         hessianStarts_.data(),
                                                         numIters,
                                                         gradTol,
                                                         scaleGrads_,
                                                         terms,
                                                         systemIndices,
                                                         systemDevice.positions.data(),
                                                         systemDevice.grad.data(),
                                                         workspace.inverseHessian.data(),
                                                         workspace.scratchBufferPointers.data(),
                                                         systemDevice.energyOuts.data(),
                                                         MMFF::batchHasConstraints(systemDevice.contribs),
                                                         statuses_.data(),
                                                         stream_);

  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Per-molecule BFGS kernel failed: ") + cudaGetErrorString(err));
  }

  return checkConvergence(activeMolIds_, statuses_, convergenceHost_, numSystems, stream_);
}

bool BfgsBatchMinimizer::minimizeWithMMFF(const int                            numIters,
                                          const double                         gradTol,
                                          const std::vector<int>&              atomStartsHost,
                                          MMFF::BatchedMolecularDeviceBuffers& systemDevice,
                                          const uint8_t*                       activeThisStage) {
  if (usesSinglePrecision(precision_)) {
    throw std::invalid_argument("Double MMFF buffers require PrecisionMode::FULL");
  }
  return minimizeWithMMFFImpl(numIters, gradTol, atomStartsHost, systemDevice, fullWorkspace(), activeThisStage);
}

bool BfgsBatchMinimizer::minimizeWithMMFF(const int                                  numIters,
                                          const double                               gradTol,
                                          const std::vector<int>&                    atomStartsHost,
                                          MMFF::BatchedMolecularDeviceBuffersSingle& systemDevice,
                                          const uint8_t*                             activeThisStage) {
  if (!usesSinglePrecision(precision_)) {
    throw std::invalid_argument("Single MMFF buffers require PrecisionMode::SINGLE");
  }
  return minimizeWithMMFFImpl(numIters, gradTol, atomStartsHost, systemDevice, singleWorkspace(), activeThisStage);
}

bool BfgsBatchMinimizer::minimizeWithETK(const int                                  numIters,
                                         const double                               gradTol,
                                         const std::vector<int>&                    atomStartsHost,
                                         const AsyncDeviceVector<int>&              atomStarts,
                                         AsyncDeviceVector<double>&                 positions,
                                         DistGeom::BatchedMolecular3DDeviceBuffers& systemDevice,
                                         const uint8_t*                             activeThisStage) {
  const int         numSystems       = atomStartsHost.size() - 1;
  const BfgsBackend effectiveBackend = resolveBackend(atomStartsHost);

  if (effectiveBackend == BfgsBackend::BATCHED) {
    throw std::runtime_error("Use minimize(..., BatchedForcefield&) for batched ETK minimization");
  }

  initialize(atomStartsHost,
             atomStarts.data(),
             positions.data(),
             systemDevice.grad.data(),
             systemDevice.energyOuts.data(),
             effectiveBackend,
             activeThisStage);

  setHessianToIdentity();

  const ScopedNvtxRange bfgsPerMoleculeETK("BfgsBatchMinimizer::perMoleculeMinimizeETK");

  prepareScratchBuffers(systemDevice.grad,
                        fullWorkspace().lineSearchDir,
                        fullWorkspace().scratchPositions,
                        fullWorkspace().hessDGrad,
                        fullWorkspace().scratchGrad,
                        fullWorkspace().scratchBufferPointers,
                        fullWorkspace().scratchBufferPointersHost,
                        stream_);

  auto terms         = DistGeom::toEnergy3DForceContribsDevicePtr(systemDevice);
  auto systemIndices = DistGeom::toBatchedIndices3DDevicePtr(systemDevice, atomStarts.data());

  const cudaError_t err = launchBfgsMinimizePerMolKernelETK(static_cast<int>(activeMolIds_.size()),
                                                            activeMolIdsDevice_.data(),
                                                            maxAtomsInBatch_,
                                                            atomStarts.data(),
                                                            hessianStarts_.data(),
                                                            numIters,
                                                            gradTol,
                                                            scaleGrads_,
                                                            terms,
                                                            systemIndices,
                                                            positions.data(),
                                                            systemDevice.grad.data(),
                                                            fullWorkspace().inverseHessian.data(),
                                                            fullWorkspace().scratchBufferPointers.data(),
                                                            systemDevice.energyOuts.data(),
                                                            statuses_.data(),
                                                            stream_);

  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Per-molecule BFGS ETK kernel failed: ") + cudaGetErrorString(err));
  }

  return checkConvergence(activeMolIds_, statuses_, convergenceHost_, numSystems, stream_);
}

bool BfgsBatchMinimizer::minimizeWithDG(const int                                numIters,
                                        const double                             gradTol,
                                        const std::vector<int>&                  atomStartsHost,
                                        const AsyncDeviceVector<int>&            atomStarts,
                                        AsyncDeviceVector<double>&               positions,
                                        DistGeom::BatchedMolecularDeviceBuffers& systemDevice,
                                        double                                   chiralWeight,
                                        double                                   fourthDimWeight,
                                        const uint8_t*                           activeThisStage) {
  const int numSystems = atomStartsHost.size() - 1;

  if (dataDim_ != 4) {
    throw std::runtime_error("minimizeWithDG requires BfgsBatchMinimizer to be constructed with dataDim=4");
  }

  const BfgsBackend effectiveBackend = resolveBackend(atomStartsHost);

  if (effectiveBackend == BfgsBackend::BATCHED) {
    throw std::runtime_error("Use minimize(..., BatchedForcefield&) for batched DG minimization");
  }

  initialize(atomStartsHost,
             atomStarts.data(),
             positions.data(),
             systemDevice.grad.data(),
             systemDevice.energyOuts.data(),
             effectiveBackend,
             activeThisStage);

  setHessianToIdentity();

  const ScopedNvtxRange bfgsPerMoleculeDG("BfgsBatchMinimizer::perMoleculeMinimizeDG");

  prepareScratchBuffers(systemDevice.grad,
                        fullWorkspace().lineSearchDir,
                        fullWorkspace().scratchPositions,
                        fullWorkspace().hessDGrad,
                        fullWorkspace().scratchGrad,
                        fullWorkspace().scratchBufferPointers,
                        fullWorkspace().scratchBufferPointersHost,
                        stream_);

  auto terms         = DistGeom::toEnergyForceContribsDevicePtr(systemDevice);
  auto systemIndices = DistGeom::toBatchedIndicesDevicePtr(systemDevice, atomStarts.data());

  const cudaError_t err = launchBfgsMinimizePerMolKernelDG(static_cast<int>(activeMolIds_.size()),
                                                           activeMolIdsDevice_.data(),
                                                           maxAtomsInBatch_,
                                                           atomStarts.data(),
                                                           hessianStarts_.data(),
                                                           numIters,
                                                           gradTol,
                                                           scaleGrads_,
                                                           terms,
                                                           systemIndices,
                                                           positions.data(),
                                                           systemDevice.grad.data(),
                                                           fullWorkspace().inverseHessian.data(),
                                                           fullWorkspace().scratchBufferPointers.data(),
                                                           systemDevice.energyOuts.data(),
                                                           chiralWeight,
                                                           fourthDimWeight,
                                                           statuses_.data(),
                                                           stream_);

  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Per-molecule BFGS DG kernel failed: ") + cudaGetErrorString(err));
  }

  return checkConvergence(activeMolIds_, statuses_, convergenceHost_, numSystems, stream_);
}

template <typename sourceT, typename storageT>
void copyAndInvertImpl(const AsyncDeviceVector<sourceT>& src, AsyncDeviceVector<storageT>& dst) {
  const size_t numElements = src.size();
  cudaStream_t stream      = dst.stream();
  if (numElements == 0) {
    return;
  }
  if (dst.size() != numElements) {
    throw std::runtime_error("Destination vector size does not match source vector size:" +
                             std::to_string(numElements) + " vs " + std::to_string(dst.size()));
  }
  const int blockSize = 128;
  const int numBlocks = (numElements + blockSize - 1) / blockSize;
  copyAndNegate<<<numBlocks, blockSize, 0, stream>>>(numElements, src.data(), dst.data());
  cudaCheckError(cudaGetLastError());
}
void copyAndInvert(const AsyncDeviceVector<double>& src, AsyncDeviceVector<double>& dst) {
  copyAndInvertImpl(src, dst);
}
void copyAndInvert(const AsyncDeviceVector<float>& src, AsyncDeviceVector<float>& dst) {
  copyAndInvertImpl(src, dst);
}
}  // namespace nvMolKit
