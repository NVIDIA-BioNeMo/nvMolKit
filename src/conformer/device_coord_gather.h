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

#ifndef NVMOLKIT_DEVICE_COORD_GATHER_H
#define NVMOLKIT_DEVICE_COORD_GATHER_H

#include <cuda_runtime.h>

#include <vector>

namespace nvMolKit {
namespace detail {

/**
 * @brief Gather scattered conformers into one contiguous destination with a single kernel launch.
 *
 * Conformer i has @p atomCounts[i] atoms, read from @p srcPositions[i] (packed 3D device pointer on the current device)
 * and written at atom offset @p dstAtomStarts[i] of @p dst. Must be called with the destination GPU current. Returns
 * after the kernel has completed on @p stream.
 */
void gatherConformerPositions(const std::vector<const double*>& srcPositions,
                              const std::vector<int>&           dstAtomStarts,
                              const std::vector<int>&           atomCounts,
                              double*                           dst,
                              cudaStream_t                      stream);

}  // namespace detail
}  // namespace nvMolKit

#endif  // NVMOLKIT_DEVICE_COORD_GATHER_H
