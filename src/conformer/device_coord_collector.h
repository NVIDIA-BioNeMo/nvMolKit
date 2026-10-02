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

#ifndef NVMOLKIT_DEVICE_COORD_COLLECTOR_H
#define NVMOLKIT_DEVICE_COORD_COLLECTOR_H

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

#include "src/conformer/device_coord_result.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {
namespace detail {

/**
 * @brief Per-thread / per-GPU accumulator of conformer batch outputs in device-output mode.
 *
 * Each OpenMP worker writes into its own DeviceCoordCollector. Producers (ETKDG, FF, ...) append
 * conformers to @ref positions in packed 3D layout (x,y,z per atom, conformer-major) and record
 * per-conformer metadata on the host (cheap) for later stitching in @ref finalizeOnTarget.
 *
 * The accumulator and its buffers live on the GPU identified by @ref gpuId. All device operations
 * execute on @ref stream.
 *
 * Optional per-conformer fields:
 * - @ref confIds: when populated by the producer (e.g. FF, which receives explicit (molIdx, confIdx)
 *   pairs from the host), @ref finalizeOnTarget uses these directly. When left empty (e.g. ETKDG),
 *   @ref finalizeOnTarget assigns per-molecule conformer indices deterministically by walking
 *   partials in the supplied order and counting per molecule.
 * - @ref attemptIds: per-molecule attempt number of each conformer (ETKDG). When populated,
 *   @ref finalizeOnTarget keeps only the lowest @c maxConformersPerMol IDs per molecule and uses each
 *   conformer's rank among them as its conformer index, independent of collection order.
 * - @ref energies and @ref converged: populated by FF; left empty by ETKDG. @ref finalizeOnTarget
 *   only allocates the matching result fields when at least one collector populates them.
 */
struct DeviceCoordCollector {
  int                       gpuId  = -1;
  cudaStream_t              stream = nullptr;
  AsyncDeviceVector<double> positions;   //!< Packed 3D, length = sum(atomCounts)*3
  AsyncDeviceVector<double> energies;    //!< Optional; length = atomCounts.size() when populated
  AsyncDeviceVector<int8_t> converged;   //!< Optional; length = atomCounts.size() when populated
  std::vector<int>          atomCounts;  //!< One per accumulated conformer
  std::vector<int>          molIds;      //!< Global molecule index per accumulated conformer
  std::vector<int>          confIds;     //!< Optional; length = atomCounts.size() when populated
  std::vector<int>          attemptIds;  //!< Optional; length = atomCounts.size() when populated
};

/**
 * @brief Shared cap-tracking state across DeviceCoordCollectors.
 *
 * Producers can dispatch many parallel attempts for a single molecule and they may all
 * succeed in the same iteration; only @c maxConformersPerMol of them should appear in the final
 * output. This struct provides shared bookkeeping so that worker threads collectively keep at
 * most that many conformers per molecule. Conformers are identified by attempt ID and the lowest IDs win,
 * so the retained set does not depend on which worker finished first. The mutex guards reads/writes to
 * @ref keptAttemptIds; an @c -1 value of @ref maxConformersPerMol disables the cap.
 *
 * A conformer displaced by a lower ID stays in its collector until @ref finalizeOnTarget drops it.
 */
struct DeviceCoordCollectorCap {
  std::mutex                                        mutex;
  //! Max-heap of the lowest attempt IDs accepted so far, per molecule.
  std::unordered_map<int, std::priority_queue<int>> keptAttemptIds;
  int                                               maxConformersPerMol = -1;
};

/**
 * @brief Stitch all per-thread DeviceCoordCollectors into a single DeviceCoordResult on @p targetGpu.
 *
 * Concatenates per-thread positions onto @p targetGpu using @ref copyDeviceToDeviceAsync and
 * computes CSR `atomStarts`. Per-molecule `confIndices` are taken from @ref DeviceCoordCollector::confIds
 * when populated and otherwise assigned deterministically by walking partials in the supplied order
 * and counting per molecule. The result's `energies` / `converged` buffers are allocated and
 * concatenated only when at least one collector populates them. All resulting buffers are
 * allocated on @p targetGpu and bound to the default stream of that GPU before returning.
 *
 * When collectors carry @ref DeviceCoordCollector::attemptIds, at most @p maxConformersPerMol conformers with the
 * lowest attempt IDs are kept per molecule (no limit if <= 0).
 *
 * @note This call is synchronous on the target stream by the time it returns: every contributing
 *       partial stream has been waited on via cross-stream events, and a final
 *       `cudaStreamSynchronize` ensures the result is visible.
 */
DeviceCoordResult finalizeOnTarget(std::vector<DeviceCoordCollector>& collectors,
                                   int                                targetGpu,
                                   int                                nMols,
                                   int                                maxConformersPerMol = -1);

}  // namespace detail
}  // namespace nvMolKit

#endif  // NVMOLKIT_DEVICE_COORD_COLLECTOR_H
