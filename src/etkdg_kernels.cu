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

#include <cub/device/device_reduce.cuh>

#include "src/etkdg_impl.h"

__global__ void collectAndFilterFailuresKernel(const int      numTerms,
                                               const uint8_t* stageFailed,
                                               uint8_t*       runFilter,
                                               int16_t*       failSum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTerms) {
    const bool systemWasActive = runFilter[idx];
    if (systemWasActive && stageFailed[idx]) {
      // Deactivate for future stages in the current iteration.
      runFilter[idx] = 0;
      failSum[idx]++;
    }
  }
}

__global__ void getFinishedKernels(const int      numTerms,
                                   const int      iteration,
                                   const uint8_t* runFilter,
                                   int16_t*       runFinished,
                                   int*           newlyFinishedSum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  using WarpReduce = cub::WarpReduce<int16_t>;
  __shared__ typename WarpReduce::TempStorage temp_storage;

  if (idx < numTerms) {
    // Check if the system was active at any point in this iteration. Note that runFinished is set below, so has not
    // been updated at this point and reflects the finished status at the start of the iteration. A molecule that
    // started the iteration, and has not had its activity turned off by a failed stage is finished.
    const bool    wasActive     = runFinished[idx] == -1;
    const bool    failed        = !runFilter[idx];
    const int16_t newlyFinished = !failed && wasActive;
    if (newlyFinished) {
      runFinished[idx] = iteration;
    }
    // Sum new finishes at warp level to reduce atomicAdd clashes. A blockreduce may be even more efficient.
    int aggregate = WarpReduce(temp_storage).Sum(newlyFinished);
    __syncwarp();
    if (threadIdx.x % warpSize == 0) {
      atomicAdd(newlyFinishedSum, aggregate);
    }
  }
}

__global__ void setRunFilterForIterationKernel(const int numTerms, uint8_t* runFilter, int16_t* runFinished) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTerms) {
    // At every iteration, any unfinished conformer is tried.
    runFilter[idx] = runFinished[idx] < 0;  // set to -1 if not finished.
  }
}

namespace nvMolKit::detail {

namespace {
constexpr int kBlockSize = 256;
}  // namespace

void launchCollectAndFilterFailuresKernel(ETKDGContext& context, const int stageId, cudaStream_t stream) {
  const int numTerms = context.nTotalSystems;
  int       gridSize = (numTerms + kBlockSize - 1) / kBlockSize;
  collectAndFilterFailuresKernel<<<gridSize, kBlockSize, 0, stream>>>(numTerms,
                                                                      context.failedThisStage.data(),
                                                                      context.activeThisStage.data(),
                                                                      context.totalFailures[stageId].data());
  cudaCheckError(cudaGetLastError());
}

int launchGetFinishedKernels(ETKDGContext& context, const int iteration, int* finishedCountHost, cudaStream_t stream) {
  const int numTerms = context.nTotalSystems;
  int       gridSize = (numTerms + kBlockSize - 1) / kBlockSize;
  context.countFinishedThisIteration.memSet(0);
  getFinishedKernels<<<gridSize, kBlockSize, 0, stream>>>(numTerms,
                                                          iteration,
                                                          context.activeThisStage.data(),
                                                          context.finishedOnIteration.data(),
                                                          context.countFinishedThisIteration.data());
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaMemcpyAsync(finishedCountHost,
                                 context.countFinishedThisIteration.data(),
                                 sizeof(int),
                                 cudaMemcpyDeviceToHost,
                                 stream));
  cudaStreamSynchronize(stream);
  return *finishedCountHost;
}

void launchSetRunFilterForIterationKernel(ETKDGContext& context, cudaStream_t stream) {
  const int numTerms = context.nTotalSystems;
  int       gridSize = (numTerms + kBlockSize - 1) / kBlockSize;
  setRunFilterForIterationKernel<<<gridSize, kBlockSize, 0, stream>>>(numTerms,
                                                                      context.activeThisStage.data(),
                                                                      context.finishedOnIteration.data());
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit::detail
