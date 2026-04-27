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

#ifndef NVMOLKIT_SUBSTRUCTURE_SEARCH_H
#define NVMOLKIT_SUBSTRUCTURE_SEARCH_H

#include <cuda_runtime.h>

#include <vector>

#include "src/substruct/substruct_types.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

/**
 * @brief Perform batch substructure matching on GPU.
 *
 * Targets are processed in input order and results are returned in the same order.
 *
 * @param targets Vector of target molecule pointers
 * @param queries Vector of query molecule pointers (typically from SMARTS)
 * @param results Output: matches[target][query][match] = vector of target atom indices
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param config Execution configuration (threading, batching). Defaults to single-threaded.
 */
void getSubstructMatches(const std::vector<const RDKit::ROMol*>& targets,
                         const std::vector<const RDKit::ROMol*>& queries,
                         SubstructSearchResults&                 results,
                         SubstructAlgorithm                      algorithm,
                         cudaStream_t                            stream,
                         const SubstructSearchConfig&            config = SubstructSearchConfig{});

/**
 * @brief Count substructure matches per (target, query) pair.
 *
 * @param targets Vector of target molecule pointers
 * @param queries Vector of query molecule pointers (typically from SMARTS)
 * @param counts Output: flattened [target * numQueries + query] match counts
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param config Execution configuration (threading, batching). Defaults to single-threaded.
 */
void countSubstructMatches(const std::vector<const RDKit::ROMol*>& targets,
                           const std::vector<const RDKit::ROMol*>& queries,
                           std::vector<int>&                       counts,
                           SubstructAlgorithm                      algorithm,
                           cudaStream_t                            stream,
                           const SubstructSearchConfig&            config = SubstructSearchConfig{});

/**
 * @brief Check if targets contain queries as substructures
 *
 * @param targets Vector of target molecule pointers
 * @param queries Vector of query molecule pointers (typically from SMARTS)
 * @param results Output: boolean for each (target, query) pair
 * @param algorithm Algorithm to use for matching
 * @param stream CUDA stream for async operations
 * @param config Execution configuration (threading, batching). Defaults to single-threaded.
 */
void hasSubstructMatch(const std::vector<const RDKit::ROMol*>& targets,
                       const std::vector<const RDKit::ROMol*>& queries,
                       HasSubstructMatchResults&               results,
                       SubstructAlgorithm                      algorithm,
                       cudaStream_t                            stream,
                       const SubstructSearchConfig&            config = SubstructSearchConfig{});

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCTURE_SEARCH_H
