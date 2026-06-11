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

#include <algorithm>
#include <mutex>
#include <stdexcept>

#include "src/conformer/etkdg_device_collect.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"

namespace nvMolKit {
namespace detail {

namespace {

/**
 * Pack survivors' positions from a 4D source buffer into a contiguous 3D destination region.
 *
 * Each conformer i (0 <= i < numConformers) has @c atomCounts[i] atoms. Source data lives at
 * @c srcPositions starting at byte offset `srcStartsAtoms[i] * dim * sizeof(double)`. Destination
 * region for conformer i begins at `dstStartsAtoms[i] * 3` doubles into @p dst3D. Atoms are
 * dispatched along x, conformers along y.
 */
__global__ void packKernel4DTo3D(const double* __restrict__ srcPositions,
                                 const int* __restrict__ srcStartsAtoms,
                                 const int* __restrict__ dstStartsAtoms,
                                 const int* __restrict__ atomCounts,
                                 int dim,
                                 int numConformers,
                                 double* __restrict__ dst3D) {
  const int conformerIdx = blockIdx.y;
  if (conformerIdx >= numConformers) {
    return;
  }
  const int atomIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int natoms  = atomCounts[conformerIdx];
  if (atomIdx >= natoms) {
    return;
  }
  const int           srcAtom = srcStartsAtoms[conformerIdx] + atomIdx;
  const int           dstAtom = dstStartsAtoms[conformerIdx] + atomIdx;
  const double* const srcPtr  = srcPositions + srcAtom * dim;
  double* const       dstPtr  = dst3D + dstAtom * 3;
  dstPtr[0]                   = srcPtr[0];
  dstPtr[1]                   = srcPtr[1];
  dstPtr[2]                   = srcPtr[2];
}

}  // namespace

void appendActive(const AsyncDeviceVector<double>&  srcPositions,
                  const std::vector<int>&           srcAtomStarts,
                  const AsyncDeviceVector<uint8_t>& active,
                  const int                         dim,
                  const std::vector<int>&           batchGlobalMolIds,
                  DeviceCoordCollectorCap&          cap,
                  DeviceCoordCollector&             collector) {
  const int batchSize = static_cast<int>(srcAtomStarts.size()) - 1;
  if (batchSize <= 0) {
    return;
  }
  if (static_cast<int>(batchGlobalMolIds.size()) != batchSize) {
    throw std::invalid_argument("batchGlobalMolIds size does not match batch size");
  }

  std::vector<uint8_t> activeHost(static_cast<size_t>(batchSize));
  active.copyToHost(activeHost.data(), static_cast<size_t>(batchSize));
  cudaCheckError(cudaStreamSynchronize(active.stream()));

  std::vector<int> srcStartsHost;
  std::vector<int> dstStartsHost;
  std::vector<int> atomCountsHost;
  srcStartsHost.reserve(batchSize);
  dstStartsHost.reserve(batchSize);
  atomCountsHost.reserve(batchSize);

  const int oldAtomTotal = static_cast<int>(collector.positions.size() / 3);
  int       runningAtoms = oldAtomTotal;
  int       maxAtoms     = 0;

  {
    const std::lock_guard<std::mutex> lock(cap.mutex);
    for (int batchSlot = 0; batchSlot < batchSize; ++batchSlot) {
      if (activeHost[static_cast<size_t>(batchSlot)] != 1) {
        continue;
      }
      const int molId = batchGlobalMolIds[batchSlot];
      if (cap.maxConformersPerMol > 0 && cap.keptPerMol[molId] >= cap.maxConformersPerMol) {
        continue;
      }
      const int srcAtomStart = srcAtomStarts[batchSlot];
      const int srcAtomEnd   = srcAtomStarts[batchSlot + 1];
      const int natoms       = srcAtomEnd - srcAtomStart;
      srcStartsHost.push_back(srcAtomStart);
      dstStartsHost.push_back(runningAtoms);
      atomCountsHost.push_back(natoms);
      collector.atomCounts.push_back(natoms);
      collector.molIds.push_back(molId);
      runningAtoms += natoms;
      maxAtoms = std::max(maxAtoms, natoms);
      cap.keptPerMol[molId]++;
    }
  }

  const int numActive = static_cast<int>(srcStartsHost.size());
  if (numActive == 0) {
    return;
  }

  const WithDevice withDevice(collector.gpuId);
  const size_t     newPositionsSize = static_cast<size_t>(runningAtoms) * 3;
  collector.positions.resize(newPositionsSize);

  AsyncDeviceVector<int> srcStartsDev(static_cast<size_t>(numActive), collector.stream);
  AsyncDeviceVector<int> dstStartsDev(static_cast<size_t>(numActive), collector.stream);
  AsyncDeviceVector<int> atomCountsDev(static_cast<size_t>(numActive), collector.stream);
  srcStartsDev.copyFromHost(srcStartsHost);
  dstStartsDev.copyFromHost(dstStartsHost);
  atomCountsDev.copyFromHost(atomCountsHost);

  constexpr int kThreadsPerBlock = 64;
  const int     blocksPerConf    = (maxAtoms + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const dim3    blocks(static_cast<unsigned>(blocksPerConf), static_cast<unsigned>(numActive));
  const dim3    threads(kThreadsPerBlock);
  packKernel4DTo3D<<<blocks, threads, 0, collector.stream>>>(srcPositions.data(),
                                                             srcStartsDev.data(),
                                                             dstStartsDev.data(),
                                                             atomCountsDev.data(),
                                                             dim,
                                                             numActive,
                                                             collector.positions.data());
  cudaCheckError(cudaGetLastError());
  // Sync to ensure the pageable host index vectors used by the H2D copies above are no longer
  // needed once we return; the scratch device buffers themselves get freed in stream order via
  // their AsyncDeviceVector destructors.
  cudaCheckError(cudaStreamSynchronize(collector.stream));
}

}  // namespace detail
}  // namespace nvMolKit
