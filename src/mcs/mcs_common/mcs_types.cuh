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

#ifndef MCS_COMMON_MCS_TYPES_CUH
#define MCS_COMMON_MCS_TYPES_CUH

#include <cstddef>
#include <utility>
#include <vector>

namespace mcs {

/**
 * @brief CSR (Compressed Sparse Row) graph representation.
 *
 * Stores an undirected graph where each undirected edge (u,v) appears twice
 * in the adjacency structure: once under u and once under v.
 */
struct Graph {
  int                 numVertices = 0;
  int                 numEdges    = 0;  ///< Count of undirected edges.
  std::vector<size_t> rowOffsets;       ///< Size = numVertices + 1.
  std::vector<size_t> colIndices;       ///< Size = 2 * numEdges (symmetric).
};

/**
 * @brief Build a CSR graph from a vertex count and undirected edge list.
 */
Graph buildGraphFromEdges(size_t numVertices, const std::vector<std::pair<size_t, size_t>>& edges);

/**
 * @brief Result of a maximum common substructure computation.
 */
struct MCSResult {
  int  numCommonVertices = 0;
  int  numCommonEdges    = 0;
  bool timedOut          = false;
  bool killed            = false;
  bool overflowed        = false;

  /// Vertex mappings: mappingA[i] <-> mappingB[i] in the common subgraph.
  std::vector<size_t> mappingA;
  std::vector<size_t> mappingB;

  /// Edge mappings (populated for MCES mode):
  /// edgeMappingA[i] = (u, v) in graphA, edgeMappingB[i] = (u, v) in graphB.
  std::vector<std::pair<size_t, size_t>> edgeMappingA;
  std::vector<std::pair<size_t, size_t>> edgeMappingB;
};

}  // namespace mcs

#endif  // MCS_COMMON_MCS_TYPES_CUH
