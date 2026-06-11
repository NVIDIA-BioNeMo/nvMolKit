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

#ifndef NVMOLKIT_GRAPH_LABELER_CUH
#define NVMOLKIT_GRAPH_LABELER_CUH

#include <cooperative_groups.h>

#include "src/data_structures/flat_bit_vect.h"
#include "src/substruct/atom_data_packed.h"
#include "src/substruct/boolean_tree.cuh"
#include "src/substruct/molecules.h"
#include "src/substruct/molecules_device.cuh"
#include "src/substruct/substruct_debug.h"

namespace nvMolKit {

/**
 * @brief Match using boolean expression tree for compound queries (OR/NOT).
 *
 * @param target Target molecule view
 * @param targetAtomIdx Index of target atom
 * @param query Query molecule view (must have query trees populated)
 * @param queryAtomIdx Index of query atom
 * @param recursiveMatchBits Per-pair recursive match bits for this target atom (32 bits for patterns 0-31)
 * @return true if target atom's properties match the compound query expression
 */
__device__ __forceinline__ bool atomPairMatches(const TargetMoleculeView& target,
                                                const int                 targetAtomIdx,
                                                const QueryMoleculeView&  query,
                                                const int                 queryAtomIdx,
                                                const uint32_t            recursiveMatchBits = 0) {
  const AtomDataPacked&  targetPacked = target.getAtomPacked(targetAtomIdx);
  const AtomQueryTree&   tree         = query.getQueryTree(queryAtomIdx);
  const BoolInstruction* instructions = query.getQueryInstructions(queryAtomIdx);
  const AtomQueryMask*   leafMasks    = query.getQueryLeafMasks(queryAtomIdx);

  // For label matrix, only check atom properties (not bond counts)
  // Bond connectivity is verified during actual substructure search
  return evaluateBoolTree</*checkBonds=*/false>(&targetPacked,
                                                nullptr,
                                                leafMasks,
                                                nullptr,
                                                instructions,
                                                tree,
                                                recursiveMatchBits);
}

/**
 * @brief Populate label matrix with fast-path warp parallelism or query trees.
 *
 * For compound queries with OR/NOT, this evaluates the boolean tree per pair.
 * For simple AND-only queries, it uses warp-level parallelism and shared query
 * data to populate the label matrix efficiently.
 *
 * @tparam MaxTargetAtoms Maximum number of atoms in target graph
 * @tparam MaxQueryAtoms Maximum number of atoms in query graph
 * @param target Target molecule view with packed data
 * @param query Query molecule view with query masks and packed data
 * @param labelMatrix Output 2D bit matrix view
 * @param pairRecursiveBits Per-pair recursive match bits indexed by [targetAtomIdx], or nullptr if none
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ void populateLabelMatrix(const TargetMoleculeView&                       target,
                                    const QueryMoleculeView&                        query,
                                    BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>& labelMatrix,
                                    const uint32_t*                                 pairRecursiveBits = nullptr) {
  namespace cg = cooperative_groups;

  auto      block      = cg::this_thread_block();
  auto      tile32     = cg::tiled_partition<32>(block);
  const int tid        = block.thread_rank();
  const int numThreads = block.size();
  const int laneId     = tile32.thread_rank();
  const int warpId     = tile32.meta_group_rank();
  const int numWarps   = tile32.meta_group_size();

  const int numQueryAtoms  = query.numAtoms;
  const int numTargetAtoms = target.numAtoms;
  const int numPairs       = numTargetAtoms * numQueryAtoms;

  labelMatrix.clearParallel(tid, numThreads);
  block.sync();

  // Check if query has boolean trees (compound queries with OR/NOT)
  if (query.hasQueryTrees()) {
    // Use boolean tree evaluation for compound queries

    for (int pairIdx = tid; pairIdx < numPairs; pairIdx += numThreads) {
      const int targetIdx = pairIdx / numQueryAtoms;
      const int queryIdx  = pairIdx % numQueryAtoms;

      const uint32_t recursiveBits = pairRecursiveBits ? pairRecursiveBits[targetIdx] : 0;
      const bool     matches       = atomPairMatches(target, targetIdx, query, queryIdx, recursiveBits);
      if constexpr (kDebugLabelMatrix) {
        if (pairRecursiveBits != nullptr) {
          printf("[LabelMatrixTree] targetAtom=%d, queryAtom=%d, recursiveBits=0x%x, matches=%d\n",
                 targetIdx,
                 queryIdx,
                 recursiveBits,
                 matches ? 1 : 0);
        }
      }
      if (matches) {
        labelMatrix.setAtomic(targetIdx, queryIdx);
      }
    }
  } else {
    // Use fast path for simple AND-only queries
    __shared__ AtomDataPacked sharedQueryPacked[MaxQueryAtoms];
    __shared__ AtomQueryMask  sharedQueryMasks[MaxQueryAtoms];

    for (int q = tid; q < numQueryAtoms; q += numThreads) {
      sharedQueryPacked[q] = query.getAtomPacked(q);
      sharedQueryMasks[q]  = query.getQueryMask(q);
    }
    block.sync();

    const int warpsNeeded = (numPairs + 31) / 32;

    for (int chunkIdx = warpId; chunkIdx < warpsNeeded; chunkIdx += numWarps) {
      const int pairIdx = chunkIdx * 32 + laneId;

      const bool validPair = (pairIdx < numPairs);

      const int targetIdx = validPair ? (pairIdx / numQueryAtoms) : 0;
      const int queryIdx  = validPair ? (pairIdx % numQueryAtoms) : 0;

      const AtomDataPacked targetPacked = validPair ? target.getAtomPacked(targetIdx) : AtomDataPacked{};
      const AtomQueryMask  queryMask    = sharedQueryMasks[queryIdx];

      const bool atomMatch = atomMatchesPacked(targetPacked, queryMask);
      const bool matches   = validPair && atomMatch;

      // Write result atomically
      if (matches) {
        labelMatrix.setAtomic(targetIdx, queryIdx);
      }
    }
  }
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_GRAPH_LABELER_CUH
