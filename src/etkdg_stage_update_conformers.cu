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

#include <GraphMol/ROMol.h>

#include "src/etkdg_stage_update_conformers.h"

namespace nvMolKit {
namespace detail {

ETKDGUpdateConformersStage::ETKDGUpdateConformersStage(
  const std::vector<RDKit::ROMol*>&                                                        mols,
  const std::vector<EmbedArgs>&                                                            eargs,
  std::unordered_map<const RDKit::ROMol*, std::vector<std::unique_ptr<RDKit::Conformer>>>& conformers,
  PinnedHostVector<double>&                                                                positionsScratch,
  PinnedHostVector<uint8_t>&                                                               activeScratch,
  cudaStream_t                                                                             stream,
  std::mutex*                                                                              conformer_mutex,
  const int                                                                                maxConformersPerMol)
    : mols_(mols),
      eargs_(eargs),
      conformers_(conformers),
      positionsScratch_(positionsScratch),
      activeScratch_(activeScratch),
      stream_(stream),
      conformer_mutex_(conformer_mutex),
      maxConformersPerMol_(maxConformersPerMol) {}

void ETKDGUpdateConformersStage::execute(ETKDGContext& ctx) {
  // Copy positions from device to host
  const size_t requiredPositionsSize = ctx.systemDevice.positions.size();
  const size_t requiredActiveSize    = ctx.activeThisStage.size();

  if (positionsScratch_.size() < requiredPositionsSize) {
    positionsScratch_.resize(requiredPositionsSize);
  }
  if (activeScratch_.size() < requiredActiveSize) {
    activeScratch_.resize(requiredActiveSize);
  }

  ctx.systemDevice.positions.copyToHost(positionsScratch_.data(), requiredPositionsSize);
  ctx.activeThisStage.copyToHost(activeScratch_.data(), requiredActiveSize);
  cudaStreamSynchronize(stream_);

  for (size_t i = 0; i < mols_.size(); ++i) {
    // Skip if not active this stage
    if (activeScratch_[i] != 1) {
      continue;
    }

    const auto& mol         = mols_[i];
    const int   dim         = eargs_[i].dim;
    const int   startPosIdx = ctx.systemHost.atomStarts[i] * dim;
    const int   nAtoms      = mol->getNumAtoms();

    auto newConf = std::make_unique<RDKit::Conformer>(mol->getNumAtoms());

    for (int j = 0; j < nAtoms; ++j) {
      const int       posIdx = startPosIdx + j * dim;
      RDGeom::Point3D pos(positionsScratch_[posIdx], positionsScratch_[posIdx + 1], positionsScratch_[posIdx + 2]);
      newConf->setAtomPos(j, pos);
    }

    // Thread-safe conformer addition with count checking
    if (conformer_mutex_) {
      std::lock_guard<std::mutex> lock(*conformer_mutex_);
      auto&                       confVec = conformers_[mol];
      if (maxConformersPerMol_ <= 0 || static_cast<int>(confVec.size()) < maxConformersPerMol_) {
        confVec.push_back(std::move(newConf));
      }
    } else {
      // Without mutex, assume single-threaded. Since we're in a batch, we could still be oversubscribing.
      auto& confVec = conformers_[mol];
      if (maxConformersPerMol_ <= 0 || static_cast<int>(confVec.size()) < maxConformersPerMol_) {
        confVec.push_back(std::move(newConf));
      }
    }
    // If conformer wasn't added, it's still a unique_ptr, and will destruct out of scope.
  }
}

}  // namespace detail
}  // namespace nvMolKit
