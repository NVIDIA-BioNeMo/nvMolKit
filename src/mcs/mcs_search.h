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

/// Primary batch API. Each pair indexes into the single molecule table.
std::vector<MCSResult> findMCSBatch(const std::vector<const RDKit::ROMol*>& mols,
                                    const std::vector<MCSPair>&             pairs,
                                    cudaStream_t                            stream = nullptr,
                                    const MCSParameters&                    params = MCSParameters{});

/// Convenience wrapper for two same-sized lists, paired by index.
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
