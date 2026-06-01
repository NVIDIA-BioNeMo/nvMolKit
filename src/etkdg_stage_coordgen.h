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

#ifndef NVMOLKIT_ETKDG_STAGE_COORDGEN_H
#define NVMOLKIT_ETKDG_STAGE_COORDGEN_H

#include <DistGeom/DistGeomUtils.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>

#include "src/etkdg_impl.h"
#include "src/forcefields/coord_gen.h"
#include "src/utils/host_vector.h"

namespace nvMolKit {

namespace detail {

class ETKDGCoordGenStage : public ETKDGStage {
 public:
  ETKDGCoordGenStage(const RDKit::DGeomHelpers::EmbedParameters& params, const std::vector<const RDKit::ROMol*>& mols);
  ~ETKDGCoordGenStage() override = default;

  void        execute(ETKDGContext& ctx) override final;
  std::string name() const override { return "Coordinate Generation (CUDA)"; }

 private:
  const RDKit::DGeomHelpers::EmbedParameters& params_;
  const std::vector<const RDKit::ROMol*>&     mols_;
  InitialCoordinateGenerator                  coordGenerator_;
};

class ETKDGCoordGenRDKitStage final : public ETKDGStage {
 public:
  ETKDGCoordGenRDKitStage(const RDKit::DGeomHelpers::EmbedParameters& params,
                          const std::vector<const RDKit::ROMol*>&     mols,
                          const std::vector<EmbedArgs>&               eargs,
                          PinnedHostVector<double>&                   positionsScratch,
                          PinnedHostVector<uint8_t>&                  activeScratch,
                          cudaStream_t                                stream = nullptr);
  ~ETKDGCoordGenRDKitStage() override = default;

  void        execute(ETKDGContext& ctx) override;
  std::string name() const override { return "Coordinate Generation (RDKit)"; }

 private:
  const RDKit::DGeomHelpers::EmbedParameters& params_;
  const std::vector<const RDKit::ROMol*>&     mols_;
  const std::vector<EmbedArgs>&               eargs_;
  PinnedHostVector<double>&                   positionsScratch_;
  PinnedHostVector<uint8_t>&                  activeScratch_;
  cudaStream_t                                stream_;
};

}  // namespace detail

}  // namespace nvMolKit

#endif  // NVMOLKIT_ETKDG_STAGE_COORDGEN_H
