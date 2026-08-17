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

#ifndef NVMOLKIT_DEVICE_CONFORMER_PRUNING_H
#define NVMOLKIT_DEVICE_CONFORMER_PRUNING_H

#include <cstdint>
#include <vector>

#include "src/conformer/device_coord_result.h"

namespace RDKit {
class ROMol;
namespace DGeomHelpers {
struct EmbedParameters;
}  // namespace DGeomHelpers
}  // namespace RDKit

namespace nvMolKit {
namespace detail {

struct ConformerPruningMolInfo {
  int confBegin       = 0;
  int confCount       = 0;
  int atomMapBegin    = 0;
  int atomMapCount    = 0;
  int mappedAtomCount = 0;
  int pairBegin       = 0;
};

//! Find RMSD conflicts and select conformers with RDKit's ordered greedy rule.
void conformerPruneMaskGpu(cuda::std::span<const double>                  coords,
                           cuda::std::span<const int32_t>                 atomStarts,
                           cuda::std::span<const int32_t>                 groupedConfIds,
                           cuda::std::span<const ConformerPruningMolInfo> molInfos,
                           cuda::std::span<const int32_t>                 atomMaps,
                           cuda::std::span<uint8_t>                       conflicts,
                           cuda::std::span<uint8_t>                       selected,
                           int                                            totalPairs,
                           double                                         threshold,
                           cudaStream_t                                   stream);

//! Prune a finalized ETKDG result without copying coordinates to the host.
DeviceCoordResult pruneDeviceConformers(DeviceCoordResult                           result,
                                        const std::vector<RDKit::ROMol*>&           mols,
                                        const RDKit::DGeomHelpers::EmbedParameters& params);

}  // namespace detail
}  // namespace nvMolKit

#endif  // NVMOLKIT_DEVICE_CONFORMER_PRUNING_H
