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

#include "src/forcefields/kernel_utils.cuh"

namespace nvMolKit {
namespace FFKernelUtils {
__global__ void reduceEnergiesKernel(const double*  energyBuffer,
                                     const int*     energyBufferBlockIdxToBatchIdx,
                                     double*        outs,
                                     const uint8_t* activeThisStage) {
  assert(blockDim.x == nvMolKit::FFKernelUtils::blockSizeEnergyReduction);

  const int outIdx = energyBufferBlockIdxToBatchIdx[blockIdx.x];

  using BlockReduce = cub::BlockReduce<double, nvMolKit::FFKernelUtils::blockSizeEnergyReduction>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // Check if activeThisStage is nullptr or if this molecule/conformer is active in this stage
  if (activeThisStage == nullptr || activeThisStage[outIdx] == 1) {
    const int    termIdx    = threadIdx.x + blockIdx.x * blockDim.x;
    const double energyTerm = energyBuffer[termIdx];
    const double blockSum   = BlockReduce(temp_storage).Sum(energyTerm);

    if (threadIdx.x == 0) {
      atomicAdd(&outs[outIdx], blockSum);
    }
  }
}

}  // namespace FFKernelUtils
}  // namespace nvMolKit
