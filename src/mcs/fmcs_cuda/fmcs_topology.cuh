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

#ifndef FMCS_CUDA_FMCS_TOPOLOGY_CUH
#define FMCS_CUDA_FMCS_TOPOLOGY_CUH

#include <cstdint>

namespace mcs {
namespace fmcs {

/// Bond endpoints are stored across the kernel as a single uint32 with
/// the u-endpoint atom index in the high 16 bits and the v-endpoint
/// atom index in the low 16 bits.  Tiers cap maxAtoms at 128 so 16 bits
/// per index is plenty.
constexpr int           kBondEndpointShift = 16;
constexpr std::uint32_t kBondEndpointMask  = 0xFFFFu;

/// Non-owning view over one side's CSR + bond-endpoint arrays.  Passed to
/// the matcher and grow helpers as the query- or target-side topology.
///
/// All four arrays are required; helpers index them unconditionally.
/// @c rowOffsets has @c numAtoms + 1 entries; @c colIndices and
/// @c bondIndices are parallel and hold one entry per directed CSR edge
/// (the neighbour atom and the undirected bond id respectively).
/// @c bondEndpoints has one packed (u << 16 | v) entry per undirected
/// bond, ordered to match the bond dimension of the match tables.
struct DeviceCsrView {
  const std::uint32_t* rowOffsets    = nullptr;
  const std::uint32_t* colIndices    = nullptr;
  const std::uint32_t* bondIndices   = nullptr;
  const std::uint32_t* bondEndpoints = nullptr;
  int                  numAtoms      = 0;
  int                  numBonds      = 0;
};

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_TOPOLOGY_CUH
