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

#ifndef NVMOLKIT_DEVICE_COORD_RESULT_H
#define NVMOLKIT_DEVICE_COORD_RESULT_H

#include <cstdint>

#include "src/utils/device_vector.h"

namespace nvMolKit {

/**
 * @brief Selector for the coordinate output mode of conformer-producing APIs.
 *
 * RDKIT_CONFORMERS writes optimized coordinates back into each input molecule's RDKit conformer
 * list and returns host-side energies (where applicable).
 *
 * DEVICE retains coordinates and (where applicable) energies on the GPU and returns them as a
 * DeviceCoordResult. Use this when chaining multiple GPU passes to avoid host round-trips.
 */
enum class CoordinateOutput : int {
  RDKIT_CONFORMERS = 0,
  DEVICE           = 1,
};

/**
 * @brief Flat CSR-style on-device representation of a batch of conformer coordinates.
 *
 * All buffers live on the GPU identified by @ref gpuId. Sizes are linked as follows:
 *  - @ref positions has length `total_atoms * 3` (3D, contiguous, conformer-major).
 *  - @ref atomStarts has length `n_conformers + 1`. `atomStarts[i+1] - atomStarts[i]` is the
 *    atom count of the i-th conformer; `atomStarts[i] * 3` is the offset into `positions`.
 *  - @ref molIndices has length `n_conformers`. `molIndices[i]` is the input-molecule index
 *    that produced conformer i.
 *  - @ref confIndices has length `n_conformers`. `confIndices[i]` is the per-molecule conformer
 *    index assigned to conformer i (stable ordering matching the host-side output).
 *  - @ref energies has length `n_conformers` for MMFF/UFF results, or 0 for ETKDG.
 *  - @ref converged has length `n_conformers` for MMFF/UFF results (1 = converged), or 0 for ETKDG.
 *  - @ref nMols is the number of molecules in the original input batch, including those that
 *    produced zero conformers. This is the authoritative outer-list size for per-molecule views.
 *
 * The streams of the contained AsyncDeviceVector members may differ during accumulation; callers
 * are responsible for synchronizing on the appropriate stream(s) before consuming the data.
 */
struct DeviceCoordResult {
  AsyncDeviceVector<double>  positions;
  AsyncDeviceVector<int32_t> atomStarts;
  AsyncDeviceVector<int32_t> molIndices;
  AsyncDeviceVector<int32_t> confIndices;
  AsyncDeviceVector<double>  energies;
  AsyncDeviceVector<int8_t>  converged;
  int                        gpuId = -1;
  int                        nMols = 0;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_DEVICE_COORD_RESULT_H
