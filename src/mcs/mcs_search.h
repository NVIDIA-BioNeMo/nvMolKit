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

#ifndef NVMOLKIT_MCS_SEARCH_H
#define NVMOLKIT_MCS_SEARCH_H

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "src/mcs/mcs_types.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

using MCSPair = std::pair<std::size_t, std::size_t>;

/// Find an exact, connected, bond-maximizing MCS for every requested pair.
///
/// Each pair indexes into @p mols and results preserve pair order. With
/// MCSParameters::requireGpu=false, only pair-specific GPU limits (128 or more
/// atoms or bonds, atom degree above 8, or GPU search overflow) cause that pair
/// to be recomputed with RDKit. With requireGpu=true, those cases throw
/// std::runtime_error. Unsupported batch-wide parameters always throw
/// std::invalid_argument. Timeouts return canceled partial results and never
/// fall back.
///
/// @throws std::invalid_argument for unsupported search parameters or invalid
///         execution configuration.
/// @throws std::runtime_error for invalid molecule references, required-GPU
///         eligibility/overflow failures, or CUDA/runtime errors.
std::vector<MCSResult> findMCSBatch(const std::vector<const RDKit::ROMol*>& mols,
                                    const std::vector<MCSPair>&             pairs,
                                    cudaStream_t                            stream = nullptr,
                                    const MCSParameters&                    params = MCSParameters{});

/// Convenience wrapper for two same-sized lists, paired by index.
/// @throws std::runtime_error if the lists have different sizes.
std::vector<MCSResult> findMCSBatch(const std::vector<const RDKit::ROMol*>& molsA,
                                    const std::vector<const RDKit::ROMol*>& molsB,
                                    cudaStream_t                            stream = nullptr,
                                    const MCSParameters&                    params = MCSParameters{});

struct MCSAllPairsOptions {
  bool upperTriangle   = true;
  bool includeDiagonal = true;
};

/// Convenience wrapper for square all-pairs dispatch over one molecule table.
///
/// Results are returned in generated-pair order. With the default options this
/// is upper-triangular row-major order: (0,0), (0,1), ..., (1,1), ...
std::vector<MCSResult> findMCSAllPairs(const std::vector<const RDKit::ROMol*>& mols,
                                       MCSAllPairsOptions                      options = {},
                                       cudaStream_t                            stream  = nullptr,
                                       const MCSParameters&                    params  = MCSParameters{});

}  // namespace nvMolKit

#endif  // NVMOLKIT_MCS_SEARCH_H
