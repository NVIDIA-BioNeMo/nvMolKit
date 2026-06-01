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

#ifndef NVMOLKIT_ETKDG_DEVICE_COLLECT_H
#define NVMOLKIT_ETKDG_DEVICE_COLLECT_H

#include <cstdint>
#include <vector>

#include "src/conformer/device_coord_collector.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {
namespace detail {

/**
 * @brief Append the active subset of an ETKDG batch's positions to @p collector.
 *
 * Reads the per-conformer @p active flags from device, compacts the surviving conformers'
 * positions in 4D->3D layout up to the per-molecule cap encoded in @p cap, and appends them
 * to @p collector.positions via a packing kernel. Updates @p collector.atomCounts and
 * @p collector.molIds for each accepted conformer.
 *
 * @param srcPositions      Device buffer of source positions in 4D layout (length =
 *                          sum(srcAtomStarts back-differences) * @p dim).
 * @param srcAtomStarts     Host-side CSR starts for the batch (length = batch_size + 1).
 * @param active            Per-conformer active flags on device (length = batch_size). A value
 *                          of 1 means the conformer should be collected.
 * @param dim               Source dimensionality (4 for ETKDG).
 * @param batchGlobalMolIds Length must equal @c srcAtomStarts.size() - 1; entry i is the
 *                          global molecule index of batch slot i.
 * @param cap               Shared cap state across all collectors; updated atomically.
 * @param collector         Thread-local accumulator to append into.
 *
 * Postcondition: @p collector buffers are extended and ready for downstream collection. No host
 * synchronization beyond the small `active` D2H copy; the actual position pack is async on
 * @p collector.stream.
 */
void appendActive(const AsyncDeviceVector<double>&  srcPositions,
                  const std::vector<int>&           srcAtomStarts,
                  const AsyncDeviceVector<uint8_t>& active,
                  int                               dim,
                  const std::vector<int>&           batchGlobalMolIds,
                  DeviceCoordCollectorCap&          cap,
                  DeviceCoordCollector&             collector);

}  // namespace detail
}  // namespace nvMolKit

#endif  // NVMOLKIT_ETKDG_DEVICE_COLLECT_H
