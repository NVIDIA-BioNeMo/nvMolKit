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

template <typename real> void BfgsBatchMinimizerT<real>::doLineSearchSetup(const real* srcEnergies) {
  const int     numblocks      = (numSystems_ + 128 - 1) / 128;
  constexpr int blockSizeSetup = 128;
  setLineStatusAndEnergyFromGlobalKernel<real><<<numblocks, 128, 0, stream_>>>(numSystems_,
                                                                               statuses_.data(),
                                                                               lineSearchStatus_.data(),
                                                                               srcEnergies,
                                                                               lineSearchStoredEnergy_.data(),
                                                                               lineSearchLambdas_.data());
  initializeLineSearchKernel<real, real, real>
    <<<numUnfinishedSystems_, blockSizeSetup, 0, stream_>>>(statuses_.data(),
                                                            positionsDevice,
                                                            gradDevice,
                                                            atomStartsDevice,
                                                            lineSearchMaxSteps_.data(),
                                                            lineSearchDir_.data(),
                                                            lineSearchSlope_.data(),
                                                            lineSearchLambdaMins_.data(),
                                                            activeSystemIndices_.data(),
                                                            dataDim_);
  cudaCheckError(cudaGetLastError());
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

template <typename real> void BfgsBatchMinimizerT<real>::doLineSearchPerturb() {
  lineSearchPerturbKernel<real, real><<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                                                  positionsDevice,
                                                                                  lineSearchDir_.data(),
                                                                                  lineSearchLambdas_.data(),
                                                                                  lineSearchLambdaMins_.data(),
                                                                                  scratchPositions_.data(),
                                                                                  scratchPositions_.data(),
                                                                                  lineSearchStatus_.data(),
                                                                                  activeSystemIndices_.data(),
                                                                                  dataDim_);
  cudaCheckError(cudaGetLastError());
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

template <typename real> void BfgsBatchMinimizerT<real>::doLineSearchPostEnergy(const int iter) {
  const int numBlocks = (numSystems_ + 127) / 128;
  lineSearchPostEnergyKernel<real, real><<<numBlocks, 128, 0, stream_>>>(numSystems_,
                                                                         iter == 0,
                                                                         lineSearchStoredEnergy_.data(),
                                                                         energyOutsDevice,
                                                                         lineSearchSlope_.data(),
                                                                         lineSearchEnergyScratch_.data(),
                                                                         lineSearchLambdas_.data(),
                                                                         lineSearchLambdas2_.data(),
                                                                         lineSearchStatus_.data());
  cudaCheckError(cudaGetLastError());
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

template <typename real> void BfgsBatchMinimizerT<real>::doLineSearchPostLoop() {
  lineSearchPostLoopKernel<real><<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                                             lineSearchStatus_.data(),
                                                                             positionsDevice,
                                                                             scratchPositions_.data(),
                                                                             scratchPositions_.data(),
                                                                             activeSystemIndices_.data(),
                                                                             dataDim_);
  cudaCheckError(cudaGetLastError());
}

struct NotEqualToMinusTwoFunctor {
  __host__ __device__ int operator()(const int16_t& x) const { return x != -2 ? 1 : 0; }
};

struct EqualsZeroFunctor {
  __host__ __device__ int operator()(const int16_t& x) const { return x == 0; }
};

template <typename real>
BfgsBatchMinimizerT<real>::BfgsBatchMinimizerT(const int    dataDim,
                                               DebugLevel   debugLevel,
                                               bool         scaleGrads,
                                               cudaStream_t stream,
                                               BfgsBackend  backend)
    : countFinished_(0, stream) {
  debugLevel_ = debugLevel;
  dataDim_    = dataDim;
  scaleGrads_ = scaleGrads;
  stream_     = stream;
  backend_    = backend;

  // For HYBRID, we need to support both paths, so initialize for both
  if (backend_ == BfgsBackend::BATCHED || backend_ == BfgsBackend::HYBRID) {
    loopStatusHost_.resize(1);
  }

  if (stream_ != nullptr) {
    activeSystemIndices_.setStream(stream_);
    allSystemIndices_.setStream(stream_);

    scratchPositions_.setStream(stream_);
    statuses_.setStream(stream_);

    lineSearchDir_.setStream(stream_);
    lineSearchStatus_.setStream(stream_);
    lineSearchLambdaMins_.setStream(stream_);
    lineSearchLambdas_.setStream(stream_);
    lineSearchLambdas2_.setStream(stream_);
    lineSearchSlope_.setStream(stream_);
    lineSearchMaxSteps_.setStream(stream_);
    countTempStorage_.setStream(stream_);
    countFinished_.setStream(stream_);
    lineSearchStoredEnergy_.setStream(stream_);
    lineSearchEnergyScratch_.setStream(stream_);

    hessianStarts_.setStream(stream_);
    scratchGrad_.setStream(stream_);
    gradScales_.setStream(stream_);
    inverseHessian_.setStream(stream_);
    hessDGrad_.setStream(stream_);
    ownedPositions_.setStream(stream_);
    ownedGrad_.setStream(stream_);
    ownedEnergies_.setStream(stream_);
    scratchBuffersDevice_.setStream(stream_);
    activeMolIdsDevice_.setStream(stream_);
  }
}
template <typename real> BfgsBatchMinimizerT<real>::~BfgsBatchMinimizerT() = default;

template <typename real>
BfgsBackend BfgsBatchMinimizerT<real>::resolveBackend(const std::vector<int>& atomStartsHost) const {
  return nvMolKit::resolveBackend(backend_, atomStartsHost);
}

template <typename real>
void BfgsBatchMinimizerT<real>::initialize(const std::vector<int>& atomStartsHost,
                                           const int*              atomStarts,
                                           double*                 positions,
                                           double*                 grad,
                                           double*                 energyOuts,
                                           BfgsBackend             effectiveBackend,
                                           const uint8_t*          activeThisStage) {
  atomStartsDevice = atomStarts;
  if constexpr (std::is_same_v<real, double>) {
    positionsDevice  = positions;
    gradDevice       = grad;
    energyOutsDevice = energyOuts;
  }

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
  inverseHessian_.resize(hessianStartsHost_.back());
  inverseHessian_.zero();

  const int numStateTerms = atomStartsHost.back() * dataDim_;
  scratchPositions_.resize(numStateTerms);
  scratchPositions_.zero();
  lineSearchDir_.resize(numStateTerms);
  scratchGrad_.resize(numStateTerms);
  hessDGrad_.resize(numStateTerms);
  hessDGrad_.zero();

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
  if constexpr (!std::is_same_v<real, double>) {
    ownedEnergies_.resize(numSystems);
    ownedPositions_.resize(numStateTerms);
    ownedGrad_.resize(numStateTerms);
    if (positions != nullptr) {
      cudaCheckError(detail::convertDeviceArray(ownedPositions_.data(), positions, numStateTerms, stream_));
    } else {
      ownedPositions_.zero();
    }
    ownedGrad_.zero();
    positionsDevice  = ownedPositions_.data();
    gradDevice       = ownedGrad_.data();
    energyOutsDevice = ownedEnergies_.data();
  }
  gradScales_.resize(numSystems);
  lineSearchLambdaMins_.resize(numSystems);
  lineSearchLambdas_.resize(numSystems);
  lineSearchLambdas2_.resize(numSystems);
  lineSearchSlope_.resize(numSystems);
  lineSearchMaxSteps_.resize(numSystems);
  lineSearchStoredEnergy_.resize(numSystems);
  lineSearchEnergyScratch_.resize(numSystems);

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

template <typename real> void BfgsBatchMinimizerT<real>::setHessianToIdentity() {
  constexpr int blockDim  = 128;
  const int     numBlocks = hessianStarts_.size() - 1;
  inverseHessian_.zero();
  populateHessianIdentityKernel<<<numBlocks, blockDim, 0, stream_>>>(hessianStarts_.data(),
                                                                     atomStartsDevice,
                                                                     inverseHessian_.data(),
                                                                     dataDim_);
  cudaCheckError(cudaGetLastError());
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

template <typename real> void BfgsBatchMinimizerT<real>::setMaxStep() {
  setMaxStepKernel<real, real, real>
    <<<numSystems_, 128, 0, stream_>>>(atomStartsDevice, positionsDevice, lineSearchMaxSteps_.data(), dataDim_);
  cudaCheckError(cudaGetLastError());
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

template <typename real> int BfgsBatchMinimizerT<real>::lineSearchCountFinished() const {
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

template <typename real> int BfgsBatchMinimizerT<real>::compactAndCountConverged() const {
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

template <typename real> void BfgsBatchMinimizerT<real>::setDirection() {
  const ScopedNvtxRange bfgsSetDirection("BfgsBatchMinimizer::setDirection");
  setDirectionKernel<real, real, real><<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                                                   scratchPositions_.data(),
                                                                                   gradDevice,
                                                                                   lineSearchDir_.data(),
                                                                                   positionsDevice,
                                                                                   scratchGrad_.data(),
                                                                                   statuses_.data(),
                                                                                   activeSystemIndices_.data(),
                                                                                   dataDim_);
  cudaCheckError(cudaGetLastError());
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

template <typename real> void BfgsBatchMinimizerT<real>::scaleGrad(const bool preLoop) {
  const int  numSystems          = preLoop ? numSystems_ : numUnfinishedSystems_;
  const int* activeSystemIndices = preLoop ? nullptr : activeSystemIndices_.data();
  if (scaleGrads_) {
    scaleGradKernel<true, real, real, real><<<numSystems, 128, 0, stream_>>>(statuses_.data(),
                                                                             atomStartsDevice,
                                                                             gradDevice,
                                                                             gradScales_.data(),
                                                                             activeSystemIndices,
                                                                             dataDim_);
  } else {
    scaleGradKernel<false, real, real, real><<<numSystems, 128, 0, stream_>>>(statuses_.data(),
                                                                              atomStartsDevice,
                                                                              gradDevice,
                                                                              gradScales_.data(),
                                                                              activeSystemIndices,
                                                                              dataDim_);
  }
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

template <typename real> void BfgsBatchMinimizerT<real>::updateDGrad() {
  const ScopedNvtxRange bfgsUpdateDGrad("BfgsBatchMinimizer::updateDGrad");
  updateDGradKernel<real, real, real><<<numUnfinishedSystems_, 128, 0, stream_>>>(static_cast<real>(gradTol_),
                                                                                  atomStartsDevice,
                                                                                  energyOutsDevice,
                                                                                  gradScales_.data(),
                                                                                  gradDevice,
                                                                                  positionsDevice,
                                                                                  scratchGrad_.data(),
                                                                                  statuses_.data(),
                                                                                  activeSystemIndices_.data(),
                                                                                  dataDim_);
  cudaCheckError(cudaGetLastError());
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

template <typename real> void BfgsBatchMinimizerT<real>::updateHessian() {
  const ScopedNvtxRange bfgsUpdateHessian("BfgsBatchMinimizer::updateHessian");
  updateHessianState(numUnfinishedSystems_,
                     statuses_.data(),
                     hessianStarts_.data(),
                     atomStartsDevice,
                     inverseHessian_.data(),
                     scratchGrad_.data(),
                     lineSearchDir_.data(),
                     hessDGrad_.data(),
                     gradDevice,
                     dataDim_,
                     hasLargeSystem_,
                     activeSystemIndices_.data(),
                     stream_);
}

template <typename real> void BfgsBatchMinimizerT<real>::collectDebugData() {
  if (debugLevel_ != DebugLevel::STEPWISE) {
    return;
  }

  // Copy energies and statuses to host for debugging.
  std::vector<int16_t> statusesHost(numSystems_);
  std::vector<real>    energiesHost(numSystems_);
  cudaCheckError(cudaMemcpyAsync(statusesHost.data(),
                                 statuses_.data(),
                                 numSystems_ * sizeof(int16_t),
                                 cudaMemcpyDeviceToHost,
                                 stream_));
  cudaCheckError(cudaMemcpyAsync(energiesHost.data(),
                                 energyOutsDevice,
                                 numSystems_ * sizeof(real),
                                 cudaMemcpyDeviceToHost,
                                 stream_));
  cudaCheckError(cudaStreamSynchronize(stream_));

  stepwiseStatuses.push_back(std::move(statusesHost));
  stepwiseEnergies.emplace_back(energiesHost.begin(), energiesHost.end());
}

template <typename real>
template <typename EnergyEvaluator, typename GradientEvaluator>
bool BfgsBatchMinimizerT<real>::minimizeBatched(const int         numIters,
                                                const double      gradTol,
                                                EnergyEvaluator   evaluateEnergy,
                                                GradientEvaluator evaluateGradient) {
  gradTol_                = gradTol;
  const int numSystems    = numSystems_;
  const int numStateTerms = numAtomsTotal_ * dataDim_;

  {
    const ScopedNvtxRange bfgsFullInitialize("BfgsBatchMinimizer::fullInitialize");
    setHessianToIdentity();

    cudaCheckError(cudaMemsetAsync(energyOutsDevice, 0, numSystems * sizeof(real), stream_));
    evaluateEnergy(positionsDevice);
    cudaCheckError(cudaMemsetAsync(gradDevice, 0, numStateTerms * sizeof(real), stream_));
    evaluateGradient();
    scaleGrad(/*preLoop=*/true);

    collectDebugData();
    constexpr int copyBlockSize = 128;
    const int     copyBlocks    = (numStateTerms + copyBlockSize - 1) / copyBlockSize;
    copyAndNegate<<<copyBlocks, copyBlockSize, 0, stream_>>>(numStateTerms, gradDevice, lineSearchDir_.data());
    cudaCheckError(cudaGetLastError());
    setMaxStep();
  }

  for (int currIter = 0; currIter < numIters && compactAndCountConverged() < numSystems; currIter++) {
    {
      const ScopedNvtxRange bfgsLineSearch("BfgsBatchMinimizer::lineSearch");
      doLineSearchSetup(energyOutsDevice);

      int              lineSearchIter         = 0;
      constexpr double MAX_ITER_LINEAR_SEARCH = 1000;
      while (lineSearchIter < MAX_ITER_LINEAR_SEARCH && lineSearchCountFinished() < numSystems) {
        doLineSearchPerturb();
        cudaCheckError(cudaMemsetAsync(energyOutsDevice, 0, numSystems * sizeof(real), stream_));
        evaluateEnergy(scratchPositions_.data());
        doLineSearchPostEnergy(lineSearchIter);
        lineSearchIter++;
      }
      doLineSearchPostLoop();
    }
    setDirection();

    {
      const ScopedNvtxRange bfgsGetAndScaleGrad("BfgsBatchMinimizer::getAndScaleGrad");
      cudaCheckError(cudaMemsetAsync(gradDevice, 0, numStateTerms * sizeof(real), stream_));
      evaluateGradient();
      scaleGrad(/*preLoop=*/false);
    }

    updateDGrad();
    updateHessian();
    collectDebugData();
  }

  cudaCheckError(cudaMemsetAsync(energyOutsDevice, 0, numSystems * sizeof(real), stream_));
  evaluateEnergy(positionsDevice);
  return compactAndCountConverged() == numSystems ? 0 : 1;
}

template <typename real>
bool BfgsBatchMinimizerT<real>::minimize(const int                  numIters,
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

  initialize(atomStartsHost,
             ff.atomStartsDevice(),
             positions.data(),
             grad.data(),
             energyOuts.data(),
             BfgsBackend::BATCHED,
             activeSystemMask);

  if constexpr (std::is_same_v<real, double>) {
    auto evaluateEnergy = [&](const double* evalPositions) {
      cudaCheckError(ff.computeEnergy(energyOutsDevice, evalPositions, activeSystemMask, stream_));
    };
    auto evaluateGradient = [&]() {
      cudaCheckError(ff.computeGradients(gradDevice, positionsDevice, activeSystemMask, stream_));
    };
    return minimizeBatched(numIters, gradTol, evaluateEnergy, evaluateGradient);
  } else {
    bool  needsAnotherCycle         = false;
    auto* singlePrecisionForcefield = dynamic_cast<SinglePrecisionBatchedForcefield*>(&ff);
    if (singlePrecisionForcefield != nullptr && usesSinglePrecision(singlePrecisionForcefield->precision())) {
      auto evaluateEnergy = [&](const real* evalPositions) {
        cudaCheckError(
          singlePrecisionForcefield->computeEnergy(energyOutsDevice, evalPositions, activeSystemMask, stream_));
      };
      auto evaluateGradient = [&]() {
        cudaCheckError(
          singlePrecisionForcefield->computeGradients(gradDevice, positionsDevice, activeSystemMask, stream_));
      };
      needsAnotherCycle = minimizeBatched(numIters, gradTol, evaluateEnergy, evaluateGradient);
    } else {
      // Forcefield only evaluates in double precision: round-trip through the caller's buffers.
      auto evaluateEnergy = [&](const real* evalPositions) {
        cudaCheckError(detail::convertDeviceArray(positions.data(), evalPositions, positions.size(), stream_));
        energyOuts.zero();
        cudaCheckError(ff.computeEnergy(energyOuts.data(), positions.data(), activeSystemMask, stream_));
        cudaCheckError(detail::convertDeviceArray(energyOutsDevice, energyOuts.data(), energyOuts.size(), stream_));
      };
      auto evaluateGradient = [&]() {
        cudaCheckError(detail::convertDeviceArray(positions.data(), positionsDevice, positions.size(), stream_));
        grad.zero();
        cudaCheckError(ff.computeGradients(grad.data(), positions.data(), activeSystemMask, stream_));
        cudaCheckError(detail::convertDeviceArray(gradDevice, grad.data(), grad.size(), stream_));
      };
      needsAnotherCycle = minimizeBatched(numIters, gradTol, evaluateEnergy, evaluateGradient);
    }
    cudaCheckError(detail::convertDeviceArray(energyOuts.data(), energyOutsDevice, energyOuts.size(), stream_));
    cudaCheckError(detail::convertDeviceArray(positions.data(), positionsDevice, positions.size(), stream_));
    cudaCheckError(detail::convertDeviceArray(grad.data(), gradDevice, grad.size(), stream_));
    return needsAnotherCycle;
  }
}

template <typename real>
bool BfgsBatchMinimizerT<real>::minimizeWithMMFF(const int               numIters,
                                                 const double            gradTol,
                                                 const std::vector<int>& atomStartsHost,
                                                 MMFFDeviceBuffers&      systemDevice,
                                                 const uint8_t*          activeThisStage) {
  const int         numSystems       = atomStartsHost.size() - 1;
  const BfgsBackend effectiveBackend = resolveBackend(atomStartsHost);

  if (effectiveBackend == BfgsBackend::BATCHED) {
    throw std::runtime_error("Use minimize(..., BatchedForcefield&) for batched MMFF minimization");
  }

  initialize(atomStartsHost,
             systemDevice.indices.atomStarts.data(),
             nullptr,
             nullptr,
             nullptr,
             effectiveBackend,
             activeThisStage);

  const ScopedNvtxRange bfgsPerMolecule("BfgsBatchMinimizer::perMoleculeMinimize");

  prepareScratchBuffers(systemDevice.grad,
                        lineSearchDir_,
                        scratchPositions_,
                        hessDGrad_,
                        scratchGrad_,
                        scratchBuffersDevice_,
                        scratchBufferPointersHost_,
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
                                                         inverseHessian_.data(),
                                                         scratchBuffersDevice_.data(),
                                                         systemDevice.energyOuts.data(),
                                                         MMFF::batchHasConstraints(systemDevice.contribs),
                                                         statuses_.data(),
                                                         stream_);

  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Per-molecule BFGS kernel failed: ") + cudaGetErrorString(err));
  }

  return checkConvergence(activeMolIds_, statuses_, convergenceHost_, numSystems, stream_);
}

template <typename real>
bool BfgsBatchMinimizerT<real>::minimizeWithETK(const int                                  numIters,
                                                const double                               gradTol,
                                                const std::vector<int>&                    atomStartsHost,
                                                const AsyncDeviceVector<int>&              atomStarts,
                                                AsyncDeviceVector<double>&                 positions,
                                                DistGeom::BatchedMolecular3DDeviceBuffers& systemDevice,
                                                const uint8_t*                             activeThisStage)
  requires std::is_same_v<real, double>
{
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
                        lineSearchDir_,
                        scratchPositions_,
                        hessDGrad_,
                        scratchGrad_,
                        scratchBuffersDevice_,
                        scratchBufferPointersHost_,
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
                                                            inverseHessian_.data(),
                                                            scratchBuffersDevice_.data(),
                                                            systemDevice.energyOuts.data(),
                                                            statuses_.data(),
                                                            stream_);

  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Per-molecule BFGS ETK kernel failed: ") + cudaGetErrorString(err));
  }

  return checkConvergence(activeMolIds_, statuses_, convergenceHost_, numSystems, stream_);
}

template <typename real>
bool BfgsBatchMinimizerT<real>::minimizeWithDG(const int                                numIters,
                                               const double                             gradTol,
                                               const std::vector<int>&                  atomStartsHost,
                                               const AsyncDeviceVector<int>&            atomStarts,
                                               AsyncDeviceVector<double>&               positions,
                                               DistGeom::BatchedMolecularDeviceBuffers& systemDevice,
                                               double                                   chiralWeight,
                                               double                                   fourthDimWeight,
                                               const uint8_t*                           activeThisStage)
  requires std::is_same_v<real, double>
{
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
                        lineSearchDir_,
                        scratchPositions_,
                        hessDGrad_,
                        scratchGrad_,
                        scratchBuffersDevice_,
                        scratchBufferPointersHost_,
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
                                                           inverseHessian_.data(),
                                                           scratchBuffersDevice_.data(),
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

template struct BfgsBatchMinimizerT<double>;
template struct BfgsBatchMinimizerT<float>;

}  // namespace nvMolKit
