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

// Per-pair, per-predicate-class target adjacency tables for the warp DFS.
//
// This generalises the Uniform/Dual precompute in candidate_tables.cuh to K
// predicate classes.  A "class" is an equivalence class of query edges whose
// predicates accept exactly the same target bonds -- for a labelled
// substructure query that is the edge label; for an MCS pair it is the query
// bond's row of the (query bond, target bond) compatibility matrix.  Once
// classes are known, the candidate intersection for a back edge in class k
// anchored on mapped target atom m is a single table read:
//
//   neighbors[k][m] = target atoms reachable from m over a target bond the
//                     class-k predicate accepts
//
// which replaces the General-plan per-descent recompute (a CSR row walk plus a
// predicate test per adjacency slot) with one shared-memory load and an AND.
//
// Scope is the caller's choice and is the point of this component being
// separate from WarpSharedState: the tables depend only on the (query, target)
// pair, so a warp-per-pair kernel (substructure search) owns one instance per
// warp, while a block-per-pair kernel (MCS) owns a single instance per block,
// fills it once, and shares it across every searching warp group.
//
// Classification itself stays with the frontend -- it is the only part that
// differs between problems -- as does the decision of how many classes to
// support before falling back to a General-style recompute.  Storage is
// K * MaxTargetAtoms masks, so K is a real shared-memory/occupancy trade-off:
// warp-per-pair callers multiply it by warps per block, and should measure
// before raising K above the Uniform/Dual equivalent of 2.

#ifndef NVMOLKIT_SUBGRAPH_CLASS_ADJACENCY_CUH
#define NVMOLKIT_SUBGRAPH_CLASS_ADJACENCY_CUH

#include <cstddef>

#include "src/subgraph/target_mask.cuh"

namespace nvMolKit {
namespace subgraph {

/// Per-class target adjacency masks.  neighbors[k][a] is the set of target
/// atoms reachable from target atom a over a target bond accepted by the
/// class-k edge predicate.  Rows at or above the pair's class count are
/// never written or read.
template <std::size_t MaxTargetAtoms, int MaxClasses> struct ClassAdjacency {
  using Mask                       = TargetMask<MaxTargetAtoms>;
  static constexpr int kMaxClasses = MaxClasses;

  Mask neighbors[MaxClasses][MaxTargetAtoms];
};

/**
 * @brief Cooperatively fill @p tables for @p numClasses classes.
 *
 * Work is striped (class, target atom)-flat across @p stride threads of rank
 * @p rank; any thread grouping works as long as every rank in [0, stride)
 * participates and the caller synchronises that grouping before reading the
 * tables.
 *
 * @tparam NeighborsFn Mask neighborsMatchingClass(int targetAtom, int cls):
 *         the target atoms reachable from @p targetAtom over a bond the
 *         class-@p cls predicate accepts.  How the predicate is evaluated
 *         (packed rows, CSR walk, match-table probe) is the frontend's
 *         business; it runs once per (class, atom) here instead of once per
 *         DFS descent.
 */
template <std::size_t MaxTargetAtoms, int MaxClasses, class NeighborsFn>
__device__ __forceinline__ void fillClassAdjacency(ClassAdjacency<MaxTargetAtoms, MaxClasses>& tables,
                                                   int                                         numClasses,
                                                   int                                         numTargetAtoms,
                                                   int                                         rank,
                                                   int                                         stride,
                                                   NeighborsFn&&                               neighborsMatchingClass) {
  const int total = numClasses * numTargetAtoms;
  for (int idx = rank; idx < total; idx += stride) {
    const int cls               = idx / numTargetAtoms;
    const int atom              = idx - cls * numTargetAtoms;
    tables.neighbors[cls][atom] = neighborsMatchingClass(atom, cls);
  }
}

}  // namespace subgraph
}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBGRAPH_CLASS_ADJACENCY_CUH
