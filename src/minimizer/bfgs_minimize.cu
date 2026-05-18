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
#include <numeric>

#include "batched_forcefield.h"
#include "bfgs_hessian.h"
#include "bfgs_minimize.h"
#include "bfgs_minimize_permol_kernels.h"
#include "cub_helpers.cuh"
#include "device_vector.h"
#include "dist_geom.h"
#include "dist_geom_kernels.h"
#include "mmff.h"
#include "mmff_kernels.h"
#include "nvtx.h"
#include "versions.h"

namespace nvMolKit {
constexpr double FUNCTOL = 1e-4;  //!< Default tolerance for function convergence in the minimizer
constexpr double MOVETOL = 1e-7;  //!< Default tolerance for x changes in the minimizer

namespace {
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
__global__ void initializeLineSearchKernel(const int16_t* statuses,
                                           const double*  oldPositions,
                                           const double*  grads,
                                           const int*     atomStarts,
                                           const double*  maxSteps,
                                           double*        dirs,
                                           double*        slopes,
                                           double*        lambdaMins,
                                           const int*     activeSystemIndices,
                                           const int      DIM) {
  const int  sysIdx        = activeSystemIndices[blockIdx.x];
  const int  idxInSys      = threadIdx.x;
  const bool isFirstThread = threadIdx.x == 0;

  if (statuses[sysIdx] == 0) {
    return;
  }

  const int     numTerms  = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const double* posStart  = &oldPositions[atomStarts[sysIdx] * DIM];
  const double* gradStart = &grads[atomStarts[sysIdx] * DIM];
  double*       dirStart  = &dirs[atomStarts[sysIdx] * DIM];

  using BlockReduce = cub::BlockReduce<double, 128>;
  __shared__ BlockReduce::TempStorage tempStorage;
  __shared__ double                   dirSum[1];

  // ---------------------------------
  //  Scale direction vector if needed
  // ---------------------------------
  double sumSquaredLocal = 0.0;
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    double dx2 = dirStart[i] * dirStart[i];
    sumSquaredLocal += dx2;
  }
  double blockSum = BlockReduce(tempStorage).Sum(sumSquaredLocal);
  if (isFirstThread) {
    dirSum[0] = sqrt(blockSum);
  }
  __syncthreads();
  if (dirSum[0] > maxSteps[sysIdx]) {
    for (int i = idxInSys; i < numTerms; i += blockDim.x) {
      dirStart[i] *= maxSteps[sysIdx] / dirSum[0];
    }
  }

  // -------------------------
  // Set slope, check validity
  // -------------------------
  double localSum = 0.0;
  // Each thread computes its partial sum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    localSum += dirStart[i] * gradStart[i];
  }

  // Perform block-wide reduction to compute the total sum
  blockSum = BlockReduce(tempStorage).Sum(localSum);
  __syncthreads();
  // The first thread in the block writes the result

  if (isFirstThread) {
    slopes[sysIdx] = blockSum;
  }

  // ----------------------
  // Compute initial lambda
  // ----------------------
  double localMax = 0.0;
  // Each thread computes its local maximum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    double temp = fabs(dirStart[i]) / fmax(fabs(posStart[i]), 1.0);
    if (temp > localMax) {
      localMax = temp;
    }
  }
  // Perform block-wide reduction to find the maximum
  double blockMax = BlockReduce(tempStorage).Reduce(localMax, cubMax());

  // The first thread in the block writes the result
  if (isFirstThread) {
    lambdaMins[sysIdx] = MOVETOL / blockMax;
  }
}

__global__ void setLineStatusAndEnergyFromGlobalKernel(const int numSystems,
                                                       const int16_t* __restrict__ statuses,
                                                       int16_t* __restrict__ lineSearchStatus,
                                                       const double* __restrict__ srcEnergies,
                                                       double* __restrict__ destEnergies,
                                                       double* __restrict__ lineSearchLambdas) {
  const int sysIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (sysIdx < numSystems) {
    const int16_t status      = statuses[sysIdx];
    lineSearchStatus[sysIdx]  = status == 0 ? 0 : -2;
    destEnergies[sysIdx]      = srcEnergies[sysIdx];
    lineSearchLambdas[sysIdx] = 1.0;
  }
}

void BfgsBatchMinimizer::doLineSearchSetup(const double* srcEnergies) {
  const int numblocks = (numSystems_ + 128 - 1) / 128;
  setLineStatusAndEnergyFromGlobalKernel<<<numblocks, 128, 0, stream_>>>(numSystems_,
                                                                         statuses_.data(),
                                                                         lineSearchStatus_.data(),
                                                                         srcEnergies,
                                                                         lineSearchStoredEnergy_.data(),
                                                                         lineSearchLambdas_.data());

  constexpr int blockSizeSetup = 128;
  initializeLineSearchKernel<<<numUnfinishedSystems_, blockSizeSetup, 0, stream_>>>(statuses_.data(),
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

__global__ void lineSearchPerturbKernel(const int*    atomStarts,
                                        const double* refPos,
                                        const double* dirs,
                                        const double* lambdas,
                                        const double* lambdaMins,
                                        double*       positions,
                                        int16_t*      statuses,
                                        const int*    activeSystemIndices,
                                        const int     DIM) {
  const int     sysIdx        = activeSystemIndices[blockIdx.x];
  const int     idxInSys      = threadIdx.x;
  const int     numTerms      = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const double* dirStart      = &dirs[atomStarts[sysIdx] * DIM];
  const double* oldPosStart   = &refPos[atomStarts[sysIdx] * DIM];
  double*       posStart      = &positions[atomStarts[sysIdx] * DIM];
  const bool    isFirstThread = threadIdx.x == 0;

  const int16_t status = statuses[sysIdx];
  if (status != -2) {
    // Case where we've already converged or failed.
    return;
  }
  const double lambda    = lambdas[sysIdx];
  const double lambdaMin = lambdaMins[sysIdx];

  if (lambda < lambdaMin) {
    if (isFirstThread) {
      statuses[sysIdx] = 1;
    }
    return;
  }

  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    posStart[i] = oldPosStart[i] + lambda * dirStart[i];
  }
}

void BfgsBatchMinimizer::doLineSearchPerturb() {
  lineSearchPerturbKernel<<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                                      positionsDevice,
                                                                      lineSearchDir_.data(),
                                                                      lineSearchLambdas_.data(),
                                                                      lineSearchLambdaMins_.data(),
                                                                      scratchPositions_.data(),
                                                                      lineSearchStatus_.data(),
                                                                      activeSystemIndices_.data(),
                                                                      dataDim_);
  cudaCheckError(cudaGetLastError());
}

__global__ void lineSearchPostEnergyKernel(const int     numSystems,
                                           const bool    isFirstIter,
                                           const double* prevE,  // oldval
                                           const double* newE,   // newval
                                           const double* slopes,
                                           double*       eScratch,  // val2
                                           double*       lambdas,
                                           double*       lambda2s,
                                           int16_t*      statuses) {
  const int sysIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (sysIdx >= numSystems) {
    return;
  }
  // Finished run.
  if (statuses[sysIdx] != -2) {
    return;
  }

  const double slope  = slopes[sysIdx];
  const double newVal = newE[sysIdx];
  const double oldVal = prevE[sysIdx];
  const double lambda = lambdas[sysIdx];
  if (newVal - oldVal <= FUNCTOL * lambda * slope) {
    // we're converged on the function:
    statuses[sysIdx] = 0;
    return;
  }
  // if we made it this far, we need to backtrack:
  double tmpLambda;
  if (isFirstIter) {
    // it's the first step:
    tmpLambda = -slope / (2.0 * (newVal - oldVal - slope));
  } else {
    const double val2    = eScratch[sysIdx];
    const double lambda2 = lambda2s[sysIdx];
    double       rhs1    = newVal - oldVal - lambda * slope;
    double       rhs2    = val2 - oldVal - lambda2 * slope;
    double       a       = (rhs1 / (lambda * lambda) - rhs2 / (lambda2 * lambda2)) / (lambda - lambda2);
    double       b = (-lambda2 * rhs1 / (lambda * lambda) + lambda * rhs2 / (lambda2 * lambda2)) / (lambda - lambda2);
    if (a == 0.0) {
      tmpLambda = -slope / (2.0 * b);
    } else {
      double disc = b * b - 3 * a * slope;
      if (disc < 0.0) {
        tmpLambda = 0.5 * lambda;
      } else if (b <= 0.0) {
        tmpLambda = (-b + sqrt(disc)) / (3.0 * a);
      } else {
        tmpLambda = -slope / (b + sqrt(disc));
      }
    }
    if (tmpLambda > 0.5 * lambda) {
      tmpLambda = 0.5 * lambda;
    }
  }
  lambda2s[sysIdx] = lambda;
  eScratch[sysIdx] = newVal;
  lambdas[sysIdx]  = max(tmpLambda, 0.1 * lambda);
}

void BfgsBatchMinimizer::doLineSearchPostEnergy(const int iter) {
  const int numBlocks = (numSystems_ + 127) / 128;
  lineSearchPostEnergyKernel<<<numBlocks, 128, 0, stream_>>>(numSystems_,
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

__global__ void lineSearchPostLoopKernel(const int*    atomStarts,
                                         int16_t*      statuses,
                                         const double* oldPos,
                                         double*       pos,
                                         const int*    activeSystemIndices,
                                         const int     DIM) {
  const int     sysIdx      = activeSystemIndices[blockIdx.x];
  const int     idxInSys    = threadIdx.x;
  const int     numTerms    = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const double* oldPosStart = &oldPos[atomStarts[sysIdx] * DIM];
  double*       posStart    = &pos[atomStarts[sysIdx] * DIM];

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
      posStart[i] = oldPosStart[i];
    }
  }
}

void BfgsBatchMinimizer::doLineSearchPostLoop() {
  lineSearchPostLoopKernel<<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                                       lineSearchStatus_.data(),
                                                                       positionsDevice,
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

BfgsBatchMinimizer::BfgsBatchMinimizer(const int    dataDim,
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
    lineSearchEnergyOut_.setStream(stream_);

    finalEnergies_.setStream(stream_);

    hessianStarts_.setStream(stream_);
    scratchGrad_.setStream(stream_);
    gradScales_.setStream(stream_);
    inverseHessian_.setStream(stream_);
    hessDGrad_.setStream(stream_);
    scratchBuffersDevice_.setStream(stream_);
    activeMolIdsDevice_.setStream(stream_);
  }
  // Allocate device array for per-molecule backend (also needed for HYBRID which might use it)
  if (backend_ == BfgsBackend::PER_MOLECULE || backend_ == BfgsBackend::HYBRID) {
    scratchBuffersDevice_.resize(5);
  }
}
BfgsBatchMinimizer::~BfgsBatchMinimizer() = default;

BfgsBackend BfgsBatchMinimizer::resolveBackend(const std::vector<int>& atomStartsHost) const {
  return nvMolKit::resolveBackend(backend_, atomStartsHost);
}

void BfgsBatchMinimizer::initialize(const std::vector<int>& atomStartsHost,
                                    const int*              atomStarts,
                                    double*                 positions,
                                    double*                 grad,
                                    double*                 energyOuts,
                                    BfgsBackend             effectiveBackend,
                                    const uint8_t*          activeThisStage) {
  atomStartsDevice = atomStarts;
  positionsDevice  = positions;
  gradDevice       = grad;
  energyOutsDevice = energyOuts;

  const int numSystems = atomStartsHost.size() - 1;
  activeHost_.resize(numSystems);
  convergenceHost_.resize(numSystems);
  scratchBufferPointersHost_.resize(5);

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

  activeSystemIndices_.resize(numSystems_);
  allSystemIndices_.resize(numSystems_);
  systemIndicesHost_.resize(numSystems_);
  std::iota(systemIndicesHost_.begin(), systemIndicesHost_.end(), 0);
  allSystemIndices_.setFromVector(systemIndicesHost_);
  activeSystemIndices_.setFromVector(systemIndicesHost_);

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

  hessDGrad_.resize(atomStartsHost.back() * dataDim_);
  hessDGrad_.zero();

  scratchPositions_.resize(atomStartsHost.back() * dataDim_);
  scratchPositions_.zero();
  scratchGrad_.resize(atomStartsHost.back() * dataDim_);
  gradScales_.resize(numSystems);

  lineSearchDir_.resize(atomStartsHost.back() * dataDim_);
  lineSearchStatus_.resize(numSystems);
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

__global__ void populateHessianIdentityKernel(const int* hessianStarts,
                                              const int* atomStarts,
                                              double*    inverseHessian,
                                              const int  DIM) {
  const int sysIdx        = blockIdx.x;
  const int idxInSys      = threadIdx.x;
  const int writeStartIdx = hessianStarts[sysIdx];
  const int numTerms      = hessianStarts[sysIdx + 1] - hessianStarts[sysIdx];
  const int numAtoms      = atomStarts[sysIdx + 1] - atomStarts[sysIdx];
  const int rowLength     = DIM * numAtoms;

  for (int i = idxInSys; i < rowLength; i += blockDim.x) {
    if (i < numTerms) {
      inverseHessian[writeStartIdx + i * rowLength + i] = 1.0;
    }
  }
}

void BfgsBatchMinimizer::setHessianToIdentity() {
  constexpr int blockDim  = 128;
  const int     numBlocks = hessianStarts_.size() - 1;
  inverseHessian_.zero();
  populateHessianIdentityKernel<<<numBlocks, blockDim, 0, stream_>>>(hessianStarts_.data(),
                                                                     atomStartsDevice,
                                                                     inverseHessian_.data(),
                                                                     dataDim_);
  cudaCheckError(cudaGetLastError());
}

__global__ void setMaxStepKernel(const int* atomStarts, const double* positions, double* maxSteps, const int DIM) {
  const int  sysIdx        = blockIdx.x;
  const int  idxInSys      = threadIdx.x;
  const bool isFirstThread = threadIdx.x == 0;

  const int     numTerms = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const double* posStart = &positions[atomStarts[sysIdx] * DIM];

  double sumSquaredPos = 0.0;
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    double dx2 = posStart[i] * posStart[i];
    sumSquaredPos += dx2;
  }

  using BlockReduce = cub::BlockReduce<double, 128>;
  __shared__ BlockReduce::TempStorage tempStorage;

  const double squaredSum = BlockReduce(tempStorage).Sum(sumSquaredPos);
  if (isFirstThread) {
    constexpr double maxStepFactor = 100.0;
    maxSteps[sysIdx]               = maxStepFactor * max(sqrt(squaredSum), static_cast<double>(numTerms));
  }
}

void BfgsBatchMinimizer::setMaxStep() {
  setMaxStepKernel<<<numSystems_, 128, 0, stream_>>>(atomStartsDevice,
                                                     positionsDevice,
                                                     lineSearchMaxSteps_.data(),
                                                     dataDim_);
  cudaCheckError(cudaGetLastError());
}

namespace {

__global__ void copyAndNegate(const int numElements, const double* src, double* dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = -src[idx];
  }
}

void prepareScratchBuffers(AsyncDeviceVector<double>&  grad,
                           AsyncDeviceVector<double>&  lineSearchDir,
                           AsyncDeviceVector<double>&  scratchPositions,
                           AsyncDeviceVector<double>&  hessDGrad,
                           AsyncDeviceVector<double>&  scratchGrad,
                           AsyncDeviceVector<double*>& scratchBuffersDevice,
                           PinnedHostVector<double*>&  scratchBufferPointersHost,
                           cudaStream_t                stream) {
  scratchBufferPointersHost[0] = grad.data();
  scratchBufferPointersHost[1] = lineSearchDir.data();
  scratchBufferPointersHost[2] = scratchPositions.data();
  scratchBufferPointersHost[3] = hessDGrad.data();
  scratchBufferPointersHost[4] = scratchGrad.data();

  cudaCheckError(cudaMemcpyAsync(scratchBuffersDevice.data(),
                                 scratchBufferPointersHost.data(),
                                 5 * sizeof(double*),
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

__global__ void setDirectionKernel(const int*    atomStarts,
                                   const double* positionsFromLineSearch,
                                   const double* grads,
                                   double*       xis,
                                   double*       positions,
                                   double*       dGrads,
                                   int16_t*      statuses,
                                   const int*    activeSystemIndices,
                                   const int     DIM) {
  const int sysIdx          = activeSystemIndices[blockIdx.x];
  const int idxWithinSystem = threadIdx.x;
  const int numTerms        = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const int startIdx        = atomStarts[sysIdx] * DIM;

  if (statuses[sysIdx] == 0) {
    return;
  }

  double*       localXi            = &xis[startIdx];
  double*       localPos           = &positions[startIdx];
  const double* localPosLineSearch = &positionsFromLineSearch[startIdx];
  const double* localGrad          = &grads[startIdx];
  double*       localDGrad         = &dGrads[startIdx];

  double localMax = 0.0;
  for (int i = idxWithinSystem; i < numTerms; i += blockDim.x) {
    localXi[i]    = localPosLineSearch[i] - localPos[i];
    localPos[i]   = localPosLineSearch[i];
    localDGrad[i] = localGrad[i];

    double temp = fabs(localXi[i]) / fmax(fabs(localPos[i]), 1.0);
    // TODO we could have a better thread distribution pattern for the local Max.
    if (temp > localMax) {
      localMax = temp;
    }
  }

  __shared__ cub::BlockReduce<double, 128>::TempStorage tempStorage;
  double           blockMax = cub::BlockReduce<double, 128>(tempStorage).Reduce(localMax, cubMax());
  constexpr double TOLX     = 4. * 3e-8;
  if (idxWithinSystem == 0 && blockMax < TOLX) {
    // Converged
    statuses[sysIdx] = 0;
  }
}

void BfgsBatchMinimizer::setDirection() {
  const ScopedNvtxRange bfgsSetDirection("BfgsBatchMinimizer::setDirection");
  setDirectionKernel<<<numUnfinishedSystems_, 128, 0, stream_>>>(atomStartsDevice,
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
template <bool scaleGrads>
__global__ void scaleGradKernel(const int16_t* statuses,
                                const int*     atomStarts,
                                double*        grads,
                                double*        gradScales,
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

  double* localGrad = &grads[atomStarts[sysIdx] * DIM];

  double            maxGrad   = kRdkitHasGradScaleFix ? 0.0 : -1e8;
  double            gradScale = scaleGrads ? 0.1 : 1.0;
  __shared__ double distributedMax[1];
  if (idxWithinSystem == 0) {
    distributedMax[0] = -1.0;  // See note at start at function, this will work for now.
  }

  for (int i = idxWithinSystem; i < numTerms; i += blockDim.x) {
    localGrad[i] *= gradScale;
    const double cmp = kRdkitHasGradScaleFix ? fabs(localGrad[i]) : localGrad[i];
    if (cmp > maxGrad) {
      maxGrad = cmp;
    }
  }

  __shared__ cub::BlockReduce<double, 128>::TempStorage tempStorage;
  double blockMax = cub::BlockReduce<double, 128>(tempStorage).Reduce(maxGrad, cubMax());

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
      localGrad[i] *= gradScale;
    }
  }
  if (idxWithinSystem == 0) {
    gradScales[sysIdx] = gradScale;
  }
}

void BfgsBatchMinimizer::scaleGrad(const bool preLoop) {
  const int  numSystems          = preLoop ? numSystems_ : numUnfinishedSystems_;
  const int* activeSystemIndices = preLoop ? nullptr : activeSystemIndices_.data();
  if (scaleGrads_) {
    scaleGradKernel<true><<<numSystems, 128, 0, stream_>>>(statuses_.data(),
                                                           atomStartsDevice,
                                                           gradDevice,
                                                           gradScales_.data(),
                                                           activeSystemIndices,
                                                           dataDim_);
  } else {
    scaleGradKernel<false><<<numSystems, 128, 0, stream_>>>(statuses_.data(),
                                                            atomStartsDevice,
                                                            gradDevice,
                                                            gradScales_.data(),
                                                            activeSystemIndices,
                                                            dataDim_);
  }
}

__global__ void updateDGradKernel(const double  gradTol,
                                  const int*    atomStarts,
                                  const double* energies,
                                  const double* gradScales,
                                  const double* grads,
                                  const double* positions,
                                  double*       dGrads,
                                  int16_t*      statuses,
                                  const int*    activeSystemIndices,
                                  const int     DIM) {
  const int sysIdx          = activeSystemIndices[blockIdx.x];
  const int idxWithinSystem = threadIdx.x;
  const int numTerms        = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const int startIdx        = atomStarts[sysIdx] * DIM;

  if (statuses[sysIdx] == 0) {
    return;
  }

  const double* localGrad = &grads[startIdx];

  const double* localPos   = &positions[startIdx];
  double*       localDGrad = &dGrads[startIdx];

  double localMax = 0.0;

  for (int i = idxWithinSystem; i < numTerms; i += blockDim.x) {
    localDGrad[i] = localGrad[i] - localDGrad[i];
    double temp   = fabs(localGrad[i]) * fmax(fabs(localPos[i]), 1.0);
    // TODO we could have a better thread distribution pattern for the local Max.
    if (temp > localMax) {
      localMax = temp;
    }
  }
  __shared__ cub::BlockReduce<double, 128>::TempStorage tempStorage;
  double blockMax = cub::BlockReduce<double, 128>(tempStorage).Reduce(localMax, cubMax());

  if (idxWithinSystem == 0) {
    const double term = max(energies[sysIdx] * gradScales[sysIdx], 1.0);
    blockMax /= term;
    if (blockMax < gradTol) {
      // Converged
      statuses[sysIdx] = 0;
    }
  }
}

void BfgsBatchMinimizer::updateDGrad() {
  const ScopedNvtxRange bfgsUpdateDGrad("BfgsBatchMinimizer::updateDGrad");
  updateDGradKernel<<<numUnfinishedSystems_, 128, 0, stream_>>>(gradTol_,
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

void BfgsBatchMinimizer::updateHessian() {
  const ScopedNvtxRange bfgsUpdateHessian("BfgsBatchMinimizer::updateHessian");
  // Determine if any active system exceeds the shared-memory-optimized limit
  bool                  largeMol = hasLargeSystem_;
  nvMolKit::updateInverseHessianBFGSBatch(numUnfinishedSystems_,
                                          statuses_.data(),
                                          hessianStarts_.data(),
                                          atomStartsDevice,
                                          inverseHessian_.data(),
                                          scratchGrad_.data(),
                                          lineSearchDir_.data(),
                                          hessDGrad_.data(),
                                          gradDevice,
                                          dataDim_,
                                          largeMol,
                                          activeSystemIndices_.data(),
                                          stream_);
}

void BfgsBatchMinimizer::collectDebugData() {
  if (debugLevel_ != DebugLevel::STEPWISE) {
    return;
  }

  // Copy energies and statuses to host for debugging.
  std::vector<int16_t> statusesHost(numSystems_);
  std::vector<double>  energiesHost(numSystems_);
  cudaCheckError(cudaMemcpyAsync(statusesHost.data(),
                                 statuses_.data(),
                                 numSystems_ * sizeof(int16_t),
                                 cudaMemcpyDeviceToHost,
                                 stream_));
  cudaCheckError(cudaMemcpyAsync(energiesHost.data(),
                                 energyOutsDevice,
                                 numSystems_ * sizeof(double),
                                 cudaMemcpyDeviceToHost,
                                 stream_));
  cudaCheckError(cudaStreamSynchronize(stream_));

  stepwiseStatuses.push_back(std::move(statusesHost));
  stepwiseEnergies.push_back(std::move(energiesHost));
}

bool BfgsBatchMinimizer::minimize(const int                  numIters,
                                  const double               gradTol,
                                  const std::vector<int>&    atomStartsHost,
                                  const int*                 atomStarts,
                                  AsyncDeviceVector<double>& positions,
                                  AsyncDeviceVector<double>& grad,
                                  AsyncDeviceVector<double>& energyOuts,
                                  EnergyFunctor              eFunc,
                                  GradFunctor                gFunc,
                                  const uint8_t*             activeThisStage) {
  gradTol_             = gradTol;
  const int numSystems = atomStartsHost.size() - 1;

  if (backend_ == BfgsBackend::PER_MOLECULE) {
    throw std::runtime_error(
      "PER_MOLECULE backend is only supported through the forcefield-specific entry points. "
      "Use minimizeWithMMFF(), minimizeWithETK(), minimizeWithDG(), or switch to BATCHED backend.");
  }

  {
    const ScopedNvtxRange bfgsFullInitialize("BfgsBatchMinimizer::fullInitialize");
    initialize(atomStartsHost,
               atomStarts,
               positions.data(),
               grad.data(),
               energyOuts.data(),
               BfgsBackend::BATCHED,
               activeThisStage);

    setHessianToIdentity();

    energyOuts.zero();
    eFunc(nullptr);
    grad.zero();
    gFunc();
    scaleGrad(/*preLoop=*/true);

    collectDebugData();
    copyAndInvert(grad, lineSearchDir_);
    setMaxStep();
  }

  for (int currIter = 0; currIter < numIters && compactAndCountConverged() < numSystems; currIter++) {
    {
      const ScopedNvtxRange bfgsLineSearch("BfgsBatchMinimizer::lineSearch");
      doLineSearchSetup(energyOuts.data());

      int              lineSearchIter         = 0;
      constexpr double MAX_ITER_LINEAR_SEARCH = 1000;
      while (lineSearchIter < MAX_ITER_LINEAR_SEARCH && lineSearchCountFinished() < numSystems) {
        doLineSearchPerturb();
        energyOuts.zero();
        eFunc(scratchPositions_.data());
        doLineSearchPostEnergy(lineSearchIter);
        lineSearchIter++;
      }
      doLineSearchPostLoop();
    }
    setDirection();

    {
      const ScopedNvtxRange bfgsGetAndScaleGrad("BfgsBatchMinimizer::getAndScaleGrad");
      grad.zero();
      gFunc();
      scaleGrad(/*preLoop=*/false);
    }

    updateDGrad();
    updateHessian();
    collectDebugData();
  }

  energyOuts.zero();
  eFunc(nullptr);
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

  auto eFunc = [&](const double* evalPositions) {
    const double* positionsToEvaluate = evalPositions != nullptr ? evalPositions : positions.data();
    ff.computeEnergy(energyOuts.data(), positionsToEvaluate, activeSystemMask, stream_);
  };
  auto gFunc = [&]() { ff.computeGradients(grad.data(), positions.data(), activeSystemMask, stream_); };

  return minimize(numIters,
                  gradTol,
                  atomStartsHost,
                  ff.atomStartsDevice(),
                  positions,
                  grad,
                  energyOuts,
                  eFunc,
                  gFunc,
                  activeSystemMask);
}

bool BfgsBatchMinimizer::minimizeWithMMFF(const int                            numIters,
                                          const double                         gradTol,
                                          const std::vector<int>&              atomStartsHost,
                                          MMFF::BatchedMolecularDeviceBuffers& systemDevice,
                                          const uint8_t*                       activeThisStage) {
  const int         numSystems       = atomStartsHost.size() - 1;
  const BfgsBackend effectiveBackend = resolveBackend(atomStartsHost);

  if (effectiveBackend == BfgsBackend::BATCHED) {
    throw std::runtime_error("Use minimize(..., BatchedForcefield&) for batched MMFF minimization");
  }

  initialize(atomStartsHost,
             systemDevice.indices.atomStarts.data(),
             systemDevice.positions.data(),
             systemDevice.grad.data(),
             systemDevice.energyOuts.data(),
             effectiveBackend,
             activeThisStage);

  setHessianToIdentity();

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

  cudaError_t err = launchBfgsMinimizePerMolKernel(static_cast<int>(activeMolIds_.size()),
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
                        lineSearchDir_,
                        scratchPositions_,
                        hessDGrad_,
                        scratchGrad_,
                        scratchBuffersDevice_,
                        scratchBufferPointersHost_,
                        stream_);

  auto terms         = DistGeom::toEnergy3DForceContribsDevicePtr(systemDevice);
  auto systemIndices = DistGeom::toBatchedIndices3DDevicePtr(systemDevice, atomStarts.data());

  cudaError_t err = launchBfgsMinimizePerMolKernelETK(static_cast<int>(activeMolIds_.size()),
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
                        lineSearchDir_,
                        scratchPositions_,
                        hessDGrad_,
                        scratchGrad_,
                        scratchBuffersDevice_,
                        scratchBufferPointersHost_,
                        stream_);

  auto terms         = DistGeom::toEnergyForceContribsDevicePtr(systemDevice);
  auto systemIndices = DistGeom::toBatchedIndicesDevicePtr(systemDevice, atomStarts.data());

  cudaError_t err = launchBfgsMinimizePerMolKernelDG(static_cast<int>(activeMolIds_.size()),
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

void copyAndInvert(const AsyncDeviceVector<double>& src, AsyncDeviceVector<double>& dst) {
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
}  // namespace nvMolKit
