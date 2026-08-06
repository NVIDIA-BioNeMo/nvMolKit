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

#ifndef FMCS_CUDA_FMCS_CUH
#define FMCS_CUDA_FMCS_CUH

// Public API for the fMCS (seed-grow connected-MCES) CUDA solver.
//
// Algorithm reference: RDKit's rdFMCS (Novartis, 2014), re-implemented
// here as a GPU-native grow-and-verify search that enumerates the
// connected-subgraph lattice of the smaller input and matches each
// candidate against the larger input on the device.
//
// Feature surface: unlabeled topology, optional exact vertex/edge-label
// matching, MaximizeBonds objective, Threshold=1.0, connected-only result.
// RDKit atomCompare/bondCompare modes map to caller-provided labels: CompareAny
// disables the corresponding table, while CompareElements/CompareIsotopes
// and CompareOrder/CompareOrderExact require the caller to encode those
// semantics as uint16 labels. RingMatchesRingOnly is likewise supported when
// ring membership is encoded in atom/bond labels. Chirality, fused-ring
// strictness, and Threshold < 1.0 are out of scope.

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "src/mcs/mcs_common/mcs_types.cuh"

namespace mcs {

namespace fmcs {

/// Maximum CSR degree supported by the device substructure matcher.
inline constexpr int kMaxNeighborsPerAtom = 8;

/// Algorithm-level parameters for the fMCS solver.
///
/// The objective (connected MCES, MaximizeBonds, Threshold=1) is baked
/// into the algorithm itself and is not exposed as a knob.
struct Parameters {
  /// CUDA block size for the per-pair kernel. Supported: 64, 128, 256, 512.
  /// Tier-128 searches at 512 threads place substructure scratch in global
  /// memory; smaller tiers retain shared-memory scratch.
  int   blockSize          = 128;
  /// Per-pair wall timeout in milliseconds.  0 = no timeout.
  float timeoutMs          = 0;
  /// Max pairs per kernel launch in the batch API.  0 = no explicit limit;
  /// process each nonempty size tier as one chunk.
  int   batchSize          = 0;
  /// Number of asynchronous executor streams for tier sub-batches. 1 = serial.
  int   executorsPerRunner = 1;
  /// For labeled inputs, require exact vertex-label equality.  When false,
  /// atom compatibility is CompareAny-style.
  bool  matchVertexLabels  = true;
  /// For labeled inputs, require exact edge-label equality.  When false,
  /// bond compatibility is CompareAny-style.
  bool  matchEdgeLabels    = true;
};

/// Find the connected MCES for a batch of unlabeled graph pairs.
///
/// `graphsA` and `graphsB` must have equal length. For unlabeled input,
/// atom/bond compatibility is topology-only. When a graph
/// exceeds the maximum supported maxSize, that pair's result has `overflowed`
/// set and all counts are zero. Graphs containing a vertex with more than
/// @ref kMaxNeighborsPerAtom neighbors are rejected.
std::vector<MCSResult> findMCESfMCSBatch(const std::vector<Graph>& graphsA,
                                         const std::vector<Graph>& graphsB,
                                         Parameters                params = {},
                                         cudaStream_t              stream = nullptr);

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_CUH
