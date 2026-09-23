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

#include "src/conformer/conformer_coord_upload.h"

#include <GraphMol/Conformer.h>
#include <GraphMol/ROMol.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/utils/cuda_error_check.h"

namespace nvMolKit {

DeviceCoordResult uploadConformerCoordinates(const std::vector<const RDKit::ROMol*>& mols, cudaStream_t stream) {
  const int numMols = static_cast<int>(mols.size());

  // Serial prefix sums over molecules give each molecule a fixed output window, so the coordinate
  // fill below can run in parallel without synchronization.
  std::vector<int64_t> molConformerStarts(numMols + 1, 0);
  std::vector<int64_t> molAtomStarts(numMols + 1, 0);
  for (int molIdx = 0; molIdx < numMols; ++molIdx) {
    if (mols[molIdx] == nullptr) {
      throw std::invalid_argument("Null molecule at index " + std::to_string(molIdx));
    }
    const int64_t numConformers    = mols[molIdx]->getNumConformers();
    const int64_t numAtoms         = mols[molIdx]->getNumAtoms();
    molConformerStarts[molIdx + 1] = molConformerStarts[molIdx] + numConformers;
    molAtomStarts[molIdx + 1]      = molAtomStarts[molIdx] + numConformers * numAtoms;
  }
  if (molAtomStarts[numMols] > std::numeric_limits<int32_t>::max()) {
    throw std::overflow_error("Total conformer atom count exceeds int32 range");
  }
  const size_t numConformers = static_cast<size_t>(molConformerStarts[numMols]);
  const size_t totalAtoms    = static_cast<size_t>(molAtomStarts[numMols]);

  int gpuId = 0;
  cudaCheckError(cudaGetDevice(&gpuId));
  DeviceCoordResult result;
  result.gpuId       = gpuId;
  result.nMols       = numMols;
  // Start the async device allocations before the host fill so they overlap.
  result.positions   = AsyncDeviceVector<double>(totalAtoms * 3, stream);
  result.atomStarts  = AsyncDeviceVector<int32_t>(numConformers + 1, stream);
  result.molIndices  = AsyncDeviceVector<int32_t>(numConformers, stream);
  result.confIndices = AsyncDeviceVector<int32_t>(numConformers, stream);

  std::vector<double>  hostPositions(totalAtoms * 3);
  std::vector<int32_t> hostAtomStarts(numConformers + 1);
  std::vector<int32_t> hostMolIndices(numConformers);
  std::vector<int32_t> hostConfIndices(numConformers);
  hostAtomStarts[numConformers] = static_cast<int32_t>(totalAtoms);

#pragma omp parallel for schedule(dynamic)
  for (int molIdx = 0; molIdx < numMols; ++molIdx) {
    const RDKit::ROMol& mol        = *mols[molIdx];
    const int           numAtoms   = static_cast<int>(mol.getNumAtoms());
    size_t              rowIdx     = static_cast<size_t>(molConformerStarts[molIdx]);
    size_t              atomOffset = static_cast<size_t>(molAtomStarts[molIdx]);
    int                 confIdx    = 0;
    for (auto confIt = mol.beginConformers(); confIt != mol.endConformers(); ++confIt, ++confIdx, ++rowIdx) {
      hostAtomStarts[rowIdx]       = static_cast<int32_t>(atomOffset);
      hostMolIndices[rowIdx]       = molIdx;
      hostConfIndices[rowIdx]      = confIdx;
      const RDKit::Conformer& conf = **confIt;
      for (int atomIdx = 0; atomIdx < numAtoms; ++atomIdx, ++atomOffset) {
        const auto& pos                   = conf.getAtomPos(atomIdx);
        hostPositions[atomOffset * 3 + 0] = pos.x;
        hostPositions[atomOffset * 3 + 1] = pos.y;
        hostPositions[atomOffset * 3 + 2] = pos.z;
      }
    }
  }

  // Pageable cudaMemcpyAsync returns once the source is staged, so the vectors may go out of scope.
  if (totalAtoms > 0) {
    result.positions.copyFromHost(hostPositions);
  }
  result.atomStarts.copyFromHost(hostAtomStarts);
  if (numConformers > 0) {
    result.molIndices.copyFromHost(hostMolIndices);
    result.confIndices.copyFromHost(hostConfIndices);
  }
  return result;
}

}  // namespace nvMolKit
