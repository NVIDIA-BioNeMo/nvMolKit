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

#ifndef NVMOLKIT_THREAD_WORKER_CONTEXT_H
#define NVMOLKIT_THREAD_WORKER_CONTEXT_H

#include <cstdint>
#include <vector>

#include "src/substruct/substruct_template_config.h"

namespace nvMolKit {

// =============================================================================
// Thread Worker Context
// =============================================================================

/**
 * @brief Per-worker context for substructure search threads.
 *
 * Contains cached data about queries and targets that's reused across mini-batches.
 */
struct ThreadWorkerContext {
  int const*              queryAtomCounts       = nullptr;
  int const*              queryPipelineDepths   = nullptr;  ///< Pipeline stage depth (maxRecursiveDepth + 1, 0 if none)
  int const*              queryMaxDepths        = nullptr;
  int8_t const*           queryHasPatterns      = nullptr;
  std::vector<int> const* targetAtomCounts      = nullptr;
  std::vector<int> const* targetOriginalIndices = nullptr;
  int                     numTargets            = 0;
  int                     numQueries            = 0;
  int                     maxTargetAtoms        = 0;
  int                     maxQueryAtoms         = 0;
  int                     maxBondsPerAtom       = 0;
  int                     maxMatches            = 0;
  bool                    countOnly             = false;  ///< If true, count matches only (for hasSubstructMatch)
  SubstructTemplateConfig templateConfig        = SubstructTemplateConfig::Config_T128_Q64_B8;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_THREAD_WORKER_CONTEXT_H
