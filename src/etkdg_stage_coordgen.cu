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

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/DistGeomUtils.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>

#include "src/etkdg_stage_coordgen.h"

namespace nvMolKit {

namespace detail {

ETKDGCoordGenStage::ETKDGCoordGenStage(const RDKit::DGeomHelpers::EmbedParameters& params,
                                       const std::vector<const RDKit::ROMol*>&     mols)
    : params_(params),
      mols_(mols) {}

__global__ void updateFailedStageKernel(const int      nSystems,
                                        uint8_t*       failedThisStageGlobal,
                                        const uint8_t* passedThisStageLocal,
                                        const uint8_t* activeThisStage) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nSystems) {
    if (activeThisStage[idx]) {
      failedThisStageGlobal[idx] = !passedThisStageLocal[idx];
    } else {
      failedThisStageGlobal[idx] = 0;
    }
  }
}

void ETKDGCoordGenStage::execute(ETKDGContext& ctx) {
  const int numSystems = ctx.nTotalSystems;
  if (numSystems == 0) {
    return;
  }
  if (numSystems != coordGenerator_.numSystemsPrepared()) {
    std::vector<ForceFields::CrystalFF::CrystalFFDetails> details(numSystems);
    coordGenerator_.computeBoundsMatrices(mols_, params_, details);
  }

  ctx.systemDevice.positions.zero();
  double*    deviceCoords     = ctx.systemDevice.positions.data();
  const int* deviceAtomStarts = ctx.systemDevice.atomStarts.data();

  coordGenerator_.computeInitialCoordinates(deviceCoords, deviceAtomStarts, ctx.activeThisStage.data());
  const auto*   passedThisStageLocal = coordGenerator_.getPassFail();
  constexpr int blockSize            = 128;
  const int     numBlocks            = (numSystems + blockSize - 1) / blockSize;
  updateFailedStageKernel<<<numBlocks, blockSize>>>(numSystems,
                                                    ctx.failedThisStage.data(),
                                                    passedThisStageLocal,
                                                    ctx.activeThisStage.data());
  cudaCheckError(cudaGetLastError());
}

ETKDGCoordGenRDKitStage::ETKDGCoordGenRDKitStage(const RDKit::DGeomHelpers::EmbedParameters& params,
                                                 const std::vector<const RDKit::ROMol*>&     mols,
                                                 const std::vector<EmbedArgs>&               eargs,
                                                 PinnedHostVector<double>&                   positionsScratch,
                                                 PinnedHostVector<uint8_t>&                  activeScratch,
                                                 cudaStream_t                                stream)
    : params_(params),
      mols_(mols),
      eargs_(eargs),
      positionsScratch_(positionsScratch),
      activeScratch_(activeScratch),
      stream_(stream) {}

void ETKDGCoordGenRDKitStage::execute(ETKDGContext& ctx) {
  const int numSystems = ctx.nTotalSystems;
  if (numSystems == 0) {
    return;
  }
  const size_t requiredActiveSize    = ctx.activeThisStage.size();
  const size_t requiredPositionsSize = ctx.systemHost.atomStarts.back() * eargs_[0].dim;

  if (activeScratch_.size() < requiredActiveSize) {
    activeScratch_.resize(requiredActiveSize);
  }
  if (positionsScratch_.size() < requiredPositionsSize) {
    positionsScratch_.resize(requiredPositionsSize);
  }

  ctx.activeThisStage.copyToHost(activeScratch_.data(), requiredActiveSize);

  auto& rng = getDoubleRandomSource();

  double boxSize;
  if (params_.boxSizeMult > 0) {
    boxSize = 5. * params_.boxSizeMult;
  } else {
    boxSize = -1 * params_.boxSizeMult;
  }
  cudaStreamSynchronize(stream_);  // for status check
  // First pass: Generate coordinates for each active molecule
  for (size_t molIdx = 0; molIdx < ctx.systemHost.atomStarts.size() - 1; ++molIdx) {
    if (!activeScratch_[molIdx]) {
      continue;
    }
    for (size_t atomIdx = 0; atomIdx < mols_[molIdx]->getNumAtoms(); ++atomIdx) {
      for (int dim = 0; dim < eargs_[molIdx].dim; ++dim) {
        // Generate random position
        double    pos          = (rng() - 0.5) * boxSize;
        const int idx          = (ctx.systemHost.atomStarts[molIdx] + atomIdx) * eargs_[molIdx].dim + dim;
        positionsScratch_[idx] = pos;
      }
    }
  }

  // Update device positions in one go
  ctx.systemDevice.positions.resize(requiredPositionsSize);
  ctx.systemDevice.positions.copyFromHost(positionsScratch_.data(), requiredPositionsSize);
}

}  // namespace detail

}  // namespace nvMolKit
