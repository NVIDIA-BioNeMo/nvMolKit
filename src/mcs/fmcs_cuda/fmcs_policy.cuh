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

#ifndef FMCS_CUDA_FMCS_POLICY_CUH
#define FMCS_CUDA_FMCS_POLICY_CUH

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "src/mcs/fmcs_cuda/fmcs_match_tables.cuh"
#include "src/mcs/labeled_graph.h"
#include "src/mcs/mcs_common/mcs_types.cuh"

namespace mcs {
namespace fmcs {

inline void fillAllCompatible(MatchTableHost& out, int nRows, int nCols) {
  out.resize(nRows, nCols);
  for (int i = 0; i < nRows; ++i) {
    for (int j = 0; j < nCols; ++j) {
      out.setBit(i, j);
    }
    // Trailing bits past nCols must stay zero so row popcounts do not
    // overcount.
    if (out.wordsPerRow > 0) {
      const int tailBits = out.wordsPerRow * 32 - nCols;
      if (tailBits > 0) {
        const uint32_t mask = (tailBits == 32) ? 0u : (0xFFFFFFFFu >> tailBits);
        out.data[static_cast<std::size_t>(i) * out.wordsPerRow + out.wordsPerRow - 1] &= mask;
      }
    }
  }
}

/// Deterministic bond enumeration for a CSR @ref Graph: each undirected
/// edge (u, v) with u < v gets a bondIdx in [0, numEdges) by first-
/// appearance order in colIndices.  @ref Graph carries no explicit bond
/// identifiers, so this function defines the ordering the fMCS code uses
/// everywhere it needs to index bonds.
inline std::vector<std::pair<int, int>> enumerateBonds(const Graph& g) {
  std::vector<std::pair<int, int>> out;
  out.reserve(static_cast<std::size_t>(g.numEdges));
  for (int u = 0; u < g.numVertices; ++u) {
    const std::size_t begin = g.rowOffsets[u];
    const std::size_t end   = g.rowOffsets[u + 1];
    for (std::size_t k = begin; k < end; ++k) {
      const int v = static_cast<int>(g.colIndices[k]);
      if (u < v)
        out.emplace_back(u, v);
    }
  }
  return out;
}

/// Unlabeled policy.  Every atom and every bond is compatible; real
/// topology and connectivity enforcement happens in the device-side match
/// walk, not in these tables.
struct NullFMCSPolicy {
  using graph_type = Graph;

  static void buildAtomMatchTable(const Graph&    query,
                                  const Graph&    target,
                                  MatchTableHost& out,
                                  bool /*matchVertexLabels*/) {
    fillAllCompatible(out, query.numVertices, target.numVertices);
  }

  static void buildBondMatchTable(const Graph&    query,
                                  const Graph&    target,
                                  MatchTableHost& out,
                                  bool /*matchEdgeLabels*/) {
    fillAllCompatible(out, query.numEdges, target.numEdges);
  }
};

/// Exact uint16_t vertex- and edge-label matching on labeled graphs;
/// 0 in @c edgeLabels means "no edge".  Bond indices follow
/// @ref enumerateBonds of the underlying topology.
struct LabeledFMCSPolicy {
  using graph_type = LabeledGraph;

  static uint16_t edgeLabel(const LabeledGraph& labeledGraph, int u, int v) {
    const std::size_t index = static_cast<std::size_t>(u) * labeledGraph.graph.numVertices + v;
    return index < labeledGraph.edgeLabels.size() ? labeledGraph.edgeLabels[index] : 0;
  }

  static void buildAtomMatchTable(const LabeledGraph& query,
                                  const LabeledGraph& target,
                                  MatchTableHost&     out,
                                  bool                matchVertexLabels) {
    const int nQ = query.graph.numVertices;
    const int nT = target.graph.numVertices;
    if (!matchVertexLabels) {
      fillAllCompatible(out, nQ, nT);
      return;
    }

    out.resize(nQ, nT);
    for (int i = 0; i < nQ; ++i) {
      const uint16_t lq = (i < static_cast<int>(query.vertexLabels.size())) ? query.vertexLabels[i] : 0;
      for (int j = 0; j < nT; ++j) {
        const uint16_t lt = (j < static_cast<int>(target.vertexLabels.size())) ? target.vertexLabels[j] : 0;
        if (lq == lt)
          out.setBit(i, j);
      }
    }
  }

  static void buildBondMatchTable(const LabeledGraph& query,
                                  const LabeledGraph& target,
                                  MatchTableHost&     out,
                                  bool                matchEdgeLabels) {
    const auto qBonds = enumerateBonds(query.graph);
    const auto tBonds = enumerateBonds(target.graph);
    out.resize(static_cast<int>(qBonds.size()), static_cast<int>(tBonds.size()));
    if (!matchEdgeLabels) {
      fillAllCompatible(out, static_cast<int>(qBonds.size()), static_cast<int>(tBonds.size()));
      return;
    }

    for (std::size_t i = 0; i < qBonds.size(); ++i) {
      const int      u  = qBonds[i].first;
      const int      v  = qBonds[i].second;
      const uint16_t lq = edgeLabel(query, u, v);
      if (lq == 0)
        continue;
      for (std::size_t j = 0; j < tBonds.size(); ++j) {
        const int      x  = tBonds[j].first;
        const int      y  = tBonds[j].second;
        const uint16_t lt = edgeLabel(target, x, y);
        if (lt != 0 && lq == lt) {
          out.setBit(static_cast<int>(i), static_cast<int>(j));
        }
      }
    }
  }
};

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_POLICY_CUH
