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

// Lane-local depth-first subgraph embedding search.
//
// Finds injective maps from query atoms to target atoms. Query atoms are
// matched in depth order: depth d means "choose the target atom for query atom
// d", and every query atom below d is already mapped. The caller owns what
// "query atom d" means -- the substructure backend matches query atoms in index
// order, while a seed-matching frontend (MCS) can search in any
// most-constrained-first permutation and apply the inverse permutation to the
// resulting mapping.
//
// All constraint knowledge lives behind the candidates oracle: at each descent
// the core asks it for the bitset of target atoms query atom `depth` may still
// map to given the mapping so far. The canonical oracle shape is one bitset
// intersection over target atoms:
//
//   candidates(d) = labelCompatible(d) & ~used
//                   & AND over each back edge d--e (e < d) of
//                       { neighbours of mapping[e] reachable over a bond that
//                         edge's bond predicate accepts }
//
// Forward edges need no check; they are back edges of the deeper atom. How the
// per-edge neighbour sets are produced is the oracle's business: the
// substructure backend precomputes them per bond-mask class (see
// substruct_dfs.cuh), an MCS backend can walk a CSR row testing a
// bond-compatibility bit matrix. The core only ever sees the resulting masks.
//
// Each calling thread searches the subtrees rooted at the target atoms in its
// @p roots mask; those subtrees are disjoint, so all search state is
// lane-local registers and no synchronisation happens inside the search. The
// intended deployment is warp-per-pair with lane L owning roots L, L+32, ...,
// but the core is agnostic to how roots are distributed.
//
// An empty candidate bitset pops the stack; a non-empty one at the last depth
// means each of its bits completes an embedding, and the terminal handler
// decides what happens next (count, store, paint, stop -- see
// DfsTerminalVerdict). The abort hook is polled between roots so a frontend
// that only needs existence can stop every lane once one lane has found an
// embedding (e.g. by polling a warp-shared flag; a finer-grained frontend can
// additionally poll inside its oracle, which runs on every descent).

#ifndef NVMOLKIT_SUBGRAPH_WARP_DFS_CUH
#define NVMOLKIT_SUBGRAPH_WARP_DFS_CUH

#include "src/subgraph/target_mask.cuh"

namespace nvMolKit {

/// What the terminal handler tells the search to do after a terminal-depth
/// visit.
struct DfsTerminalVerdict {
  bool rootDone;  ///< Abandon the rest of the current root's subtree.
  bool laneDone;  ///< Stop this thread's search entirely; no further roots.
};

/**
 * @brief Depth-first search over injective query->target atom maps, run
 *        independently by each calling thread over its @p roots.
 *
 * @tparam MaxDepth     Capacity of the per-thread stack: the maximum number of
 *                      query atoms (depths). Target atom indices are stored as
 *                      bytes, so target capacity is bounded by the Mask width
 *                      (<= 128 everywhere in nvMolKit).
 * @tparam Mask         Target-atom bitset, TargetMask<32/64/128> or anything
 *                      matching its interface.
 * @tparam CandidatesFn Mask candidatesAt(int depth, const unsigned char* mapping,
 *                      const Mask& used, int prevTargetAtom). Must already
 *                      exclude atoms in @p used. @p mapping[0..depth-1] are the
 *                      committed target atoms; @p prevTargetAtom == mapping[depth-1],
 *                      passed separately so chain-shaped oracles skip the load.
 * @tparam TerminalFn   DfsTerminalVerdict onTerminal(Mask terminals,
 *                      const unsigned char* mapping). Called at @p lastDepth
 *                      with the candidate set for the final query atom; every
 *                      bit of @p terminals completes a distinct embedding with
 *                      @p mapping[0..lastDepth-1]. mapping[0] is the root.
 * @tparam AbortFn      bool abortRequested(), polled before each root.
 *
 * @param roots     Target atoms to anchor query atom 0 on, already filtered to
 *                  label-compatible ones.
 * @param lastDepth Index of the final query atom; must be >= 1 and < MaxDepth.
 *                  Single-atom queries (lastDepth == 0) have no bond
 *                  constraints and should be handled by the caller directly
 *                  from its root mask.
 */
template <int MaxDepth, class Mask, class CandidatesFn, class TerminalFn, class AbortFn>
__device__ __forceinline__ void dfsFromRoots(Mask           roots,
                                             const int      lastDepth,
                                             CandidatesFn&& candidatesAt,
                                             TerminalFn&&   onTerminal,
                                             AbortFn&&      abortRequested) {
  // The lane-local stack: mapping[d] is the target atom assigned to query atom
  // d, remaining[d] the candidates at depth d not yet tried, and used the
  // target atoms taken by depths 0..depth-1, which keeps the mapping injective.
  Mask          remaining[MaxDepth];
  unsigned char mapping[MaxDepth];

  while (!roots.empty()) {
    if (abortRequested()) {
      return;
    }
    // Restart with query atom 0 anchored on the thread's next root.
    const int rootAtom = roots.lowest();
    roots.clearLowest();

    Mask used;
    used.clear();
    used.set(rootAtom);
    mapping[0] = static_cast<unsigned char>(rootAtom);

    int depth     = 1;
    remaining[1]  = candidatesAt(1, mapping, used, rootAtom);
    bool laneDone = false;

    while (depth >= 1) {
      if (depth == lastDepth) {
        // Nothing deeper constrains the last query atom, so every remaining
        // candidate completes a distinct embedding. The handler consumes the
        // whole bitset; fall through to the pop below with this depth emptied.
        const DfsTerminalVerdict verdict = onTerminal(remaining[depth], mapping);
        remaining[depth].clear();
        if (verdict.rootDone || verdict.laneDone) {
          laneDone = verdict.laneDone;
          break;
        }
      }

      if (remaining[depth].empty()) {
        // Exhausted: pop, releasing the atom the depth above had taken. Depth
        // 0 is the root, which the outer loop advances, so the stack bottoms
        // out at depth 1.
        --depth;
        if (depth >= 1) {
          used.reset(mapping[depth]);
        }
        continue;
      }

      // Otherwise commit the lowest untried candidate and descend, building
      // the next depth's candidate set under the extended mapping.
      const int candidate = remaining[depth].lowest();
      remaining[depth].clearLowest();
      mapping[depth] = static_cast<unsigned char>(candidate);
      used.set(candidate);
      ++depth;
      remaining[depth] = candidatesAt(depth, mapping, used, candidate);
    }

    if (laneDone) {
      return;
    }
  }
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBGRAPH_WARP_DFS_CUH
