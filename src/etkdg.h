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

#include <vector>

#include "src/minimizer/bfgs_minimize.h"
#include "src/etkdg_impl.h"
#include "src/hardware_options.h"

namespace RDKit {
class ROMol;

namespace DGeomHelpers {
struct EmbedParameters;
}  // namespace DGeomHelpers
}  // namespace RDKit

namespace nvMolKit {

void embedMolecules(const std::vector<RDKit::ROMol*>&           mols,
                    const RDKit::DGeomHelpers::EmbedParameters& params,
                    int                                         confsPerMolecule = 1,
                    int                                         maxIterations    = -1,
                    bool                                        debugMode        = false,
                    std::vector<std::vector<int16_t>>*          failures         = nullptr,
                    const BatchHardwareOptions&                 hardwareOptions  = {},
                    BfgsBackend                                 backend          = BfgsBackend::HYBRID);

}  // namespace nvMolKit

#endif
