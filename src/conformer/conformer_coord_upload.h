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

#ifndef NVMOLKIT_CONFORMER_COORD_UPLOAD_H
#define NVMOLKIT_CONFORMER_COORD_UPLOAD_H

#include <cuda_runtime.h>

#include <vector>

#include "src/conformer/device_coord_result.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

/**
 * @brief Pack every RDKit conformer of a molecule batch into a device-resident DeviceCoordResult.
 *
 * Conformers are emitted in input-molecule order, then RDKit conformer iteration order. Molecules
 * without conformers contribute no rows but still count toward @c nMols. @c confIndices holds the
 * per-molecule conformer position (0, 1, ...). @c energies and @c converged are left empty.
 *
 * Coordinates are extracted into pageable host memory in parallel across molecules and copied on
 * @p stream. The result is bound to @p stream and to the current CUDA device.
 *
 * @throws std::invalid_argument if any molecule pointer is null.
 * @throws std::overflow_error   if the total atom count across all conformers exceeds int32 range.
 */
DeviceCoordResult uploadConformerCoordinates(const std::vector<const RDKit::ROMol*>& mols, cudaStream_t stream);

}  // namespace nvMolKit

#endif  // NVMOLKIT_CONFORMER_COORD_UPLOAD_H
