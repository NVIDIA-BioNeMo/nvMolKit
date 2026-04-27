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

#ifndef NVMOLKIT_ETKDG_STAGE_UPDATE_CONFORMERS_H
#define NVMOLKIT_ETKDG_STAGE_UPDATE_CONFORMERS_H

#include <mutex>
#include <unordered_map>
#include <vector>

#include "src/etkdg_impl.h"
#include "src/utils/host_vector.h"

namespace nvMolKit {
namespace detail {

class ETKDGUpdateConformersStage final : public ETKDGStage {
 public:
  ETKDGUpdateConformersStage(
    const std::vector<RDKit::ROMol*>&                                                        mols,
    const std::vector<EmbedArgs>&                                                            eargs,
    std::unordered_map<const RDKit::ROMol*, std::vector<std::unique_ptr<RDKit::Conformer>>>& conformers,
    PinnedHostVector<double>&                                                                positionsScratch,
    PinnedHostVector<uint8_t>&                                                               activeScratch,
    cudaStream_t                                                                             stream          = nullptr,
    std::mutex*                                                                              conformer_mutex = nullptr,
    int                                                                                      maxConformersPerMol = -1);

  void        execute(ETKDGContext& ctx) override;
  std::string name() const override { return "Update Conformers"; }

 private:
  const std::vector<RDKit::ROMol*>&                                                        mols_;
  const std::vector<EmbedArgs>&                                                            eargs_;
  std::unordered_map<const RDKit::ROMol*, std::vector<std::unique_ptr<RDKit::Conformer>>>& conformers_;
  PinnedHostVector<double>&                                                                positionsScratch_;
  PinnedHostVector<uint8_t>&                                                               activeScratch_;
  cudaStream_t                                                                             stream_;
  std::mutex*                                                                              conformer_mutex_;
  int                                                                                      maxConformersPerMol_;
};

}  // namespace detail
}  // namespace nvMolKit

#endif
