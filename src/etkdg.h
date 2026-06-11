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

#ifndef NVMOLKIT_ETKDG_H
#define NVMOLKIT_ETKDG_H

#include <optional>
#include <vector>

#include "src/conformer/device_coord_result.h"
#include "src/hardware_options.h"
#include "src/minimizer/bfgs_minimize.h"

namespace RDKit {
class ROMol;

namespace DGeomHelpers {
struct EmbedParameters;
}  // namespace DGeomHelpers
}  // namespace RDKit

namespace nvMolKit {

/**
 * @brief Embed molecules using ETKDG, optionally returning coordinates on the GPU.
 *
 * In @c CoordinateOutput::RDKIT_CONFORMERS mode (default), optimized coordinates are written
 * back to each input molecule's RDKit conformer list and the function returns @c std::nullopt.
 *
 * In @c CoordinateOutput::DEVICE mode, coordinates remain on the GPU and are returned as a
 * @c DeviceCoordResult collected onto @p targetGpu (defaults to the first id in
 * @c hardwareOptions.gpuIds, or device 0 when no ids are specified). RDKit conformer lists
 * are left untouched in this mode. ETKDG conformer pruning (@c params.pruneRmsThresh) must be
 * disabled when using DEVICE mode; an exception is thrown otherwise.
 */
std::optional<DeviceCoordResult> embedMolecules(const std::vector<RDKit::ROMol*>&           mols,
                                                const RDKit::DGeomHelpers::EmbedParameters& params,
                                                int                                         confsPerMolecule = 1,
                                                int                                         maxIterations    = -1,
                                                bool                                        debugMode        = false,
                                                std::vector<std::vector<int16_t>>*          failures         = nullptr,
                                                const BatchHardwareOptions&                 hardwareOptions  = {},
                                                BfgsBackend      backend   = BfgsBackend::HYBRID,
                                                CoordinateOutput output    = CoordinateOutput::RDKIT_CONFORMERS,
                                                int              targetGpu = -1);

}  // namespace nvMolKit

#endif
