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

#include <GraphMol/ROMol.h>

#include <algorithm>

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
  const int                                                                                maxConformersPerMol,
  std::vector<int>                                                                         attemptIds)
    : mols_(mols),
      eargs_(eargs),
      conformers_(conformers),
      positionsScratch_(positionsScratch),
      activeScratch_(activeScratch),
      stream_(stream),
      conformer_mutex_(conformer_mutex),
      maxConformersPerMol_(maxConformersPerMol),
      attemptIds_(std::move(attemptIds)) {}

void ETKDGUpdateConformersStage::addOrReplace(std::vector<std::unique_ptr<RDKit::Conformer>>& confVec,
                                              std::unique_ptr<RDKit::Conformer>               newConf) const {
  if (maxConformersPerMol_ <= 0 || static_cast<int>(confVec.size()) < maxConformersPerMol_) {
    confVec.push_back(std::move(newConf));
    return;
  }
  if (attemptIds_.empty()) {
    return;
  }
  // Full: keep the lowest attempt IDs so the retained set does not depend on which worker finished first.
  const auto worst = std::max_element(confVec.begin(), confVec.end(), [](const auto& a, const auto& b) {
    return a->getId() < b->getId();
  });
  if (newConf->getId() < (*worst)->getId()) {
    *worst = std::move(newConf);
  }
}

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

    // Tag the conformer with its attempt ID so the lowest-ID successes can be kept regardless of arrival order.
    if (!attemptIds_.empty()) {
      newConf->setId(static_cast<unsigned int>(attemptIds_[i]));
    }

    // Thread-safe conformer addition with count checking
    if (conformer_mutex_) {
      std::lock_guard<std::mutex> lock(*conformer_mutex_);
      addOrReplace(conformers_[mol], std::move(newConf));
    } else {
      // Without mutex, assume single-threaded. Since we're in a batch, we could still be oversubscribing.
      addOrReplace(conformers_[mol], std::move(newConf));
    }
    // If conformer wasn't kept, it's still a unique_ptr, and will destruct out of scope.
  }
}

}  // namespace detail
}  // namespace nvMolKit
