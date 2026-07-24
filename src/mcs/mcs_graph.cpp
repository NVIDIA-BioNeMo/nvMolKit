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

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mcs_common/mcs_types.cuh"

namespace mcs {

Graph buildGraphFromEdges(size_t numVertices, const std::vector<std::pair<size_t, size_t>>& edges) {
  Graph graph;
  graph.numVertices = static_cast<int>(numVertices);
  graph.numEdges    = static_cast<int>(edges.size());
  graph.rowOffsets.assign(numVertices + 1, 0);

  for (const auto& [u, v] : edges) {
    if (u >= numVertices || v >= numVertices) {
      throw std::runtime_error("MCS graph edge endpoint out of range");
    }
    if (u == v) {
      throw std::runtime_error("MCS graph self-loops are not supported");
    }
    ++graph.rowOffsets[u + 1];
    ++graph.rowOffsets[v + 1];
  }

  for (size_t i = 1; i < graph.rowOffsets.size(); ++i) {
    graph.rowOffsets[i] += graph.rowOffsets[i - 1];
  }

  graph.colIndices.assign(graph.rowOffsets.back(), 0);
  std::vector<size_t> cursor = graph.rowOffsets;
  for (const auto& [u, v] : edges) {
    graph.colIndices[cursor[u]++] = v;
    graph.colIndices[cursor[v]++] = u;
  }

  for (size_t atomIdx = 0; atomIdx < numVertices; ++atomIdx) {
    const auto begin = graph.colIndices.begin() + static_cast<std::ptrdiff_t>(graph.rowOffsets[atomIdx]);
    const auto end   = graph.colIndices.begin() + static_cast<std::ptrdiff_t>(graph.rowOffsets[atomIdx + 1]);
    std::sort(begin, end);
  }

  return graph;
}

}  // namespace mcs
