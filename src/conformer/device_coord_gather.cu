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

#include <cstdint>

#include "src/conformer/device_coord_gather.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {
namespace detail {

namespace {

__global__ void gatherPositionsKernel(const unsigned long long* __restrict__ srcPositions,
                                      const int* __restrict__ dstAtomStarts,
                                      const int* __restrict__ atomCounts,
                                      double* __restrict__ dst) {
  const int           conformerIdx = blockIdx.x;
  const double* const src          = reinterpret_cast<const double*>(srcPositions[conformerIdx]);
  double* const       out          = dst + static_cast<size_t>(dstAtomStarts[conformerIdx]) * 3;
  const int           numValues    = atomCounts[conformerIdx] * 3;
  for (int i = threadIdx.x; i < numValues; i += blockDim.x) {
    out[i] = src[i];
  }
}

}  // namespace

void gatherConformerPositions(const std::vector<const double*>& srcPositions,
                              const std::vector<int>&           dstAtomStarts,
                              const std::vector<int>&           atomCounts,
                              double* const                     dst,
                              const cudaStream_t                stream) {
  const size_t numConformers = srcPositions.size();
  if (numConformers == 0) {
    return;
  }
  std::vector<unsigned long long> srcHost(numConformers);
  for (size_t i = 0; i < numConformers; ++i) {
    srcHost[i] = reinterpret_cast<uintptr_t>(srcPositions[i]);
  }
  AsyncDeviceVector<unsigned long long> srcDev(numConformers, stream);
  AsyncDeviceVector<int>                startsDev(numConformers, stream);
  AsyncDeviceVector<int>                countsDev(numConformers, stream);
  srcDev.copyFromHost(srcHost);
  startsDev.copyFromHost(dstAtomStarts);
  countsDev.copyFromHost(atomCounts);

  constexpr int kThreadsPerBlock = 128;
  gatherPositionsKernel<<<static_cast<unsigned>(numConformers), kThreadsPerBlock, 0, stream>>>(srcDev.data(),
                                                                                               startsDev.data(),
                                                                                               countsDev.data(),
                                                                                               dst);
  cudaCheckError(cudaGetLastError());
  // The index vectors above are pageable host memory and the device scratch is freed on return.
  cudaCheckError(cudaStreamSynchronize(stream));
}

}  // namespace detail
}  // namespace nvMolKit
