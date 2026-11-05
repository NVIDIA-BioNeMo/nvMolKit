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

// Full-warp reductions returning the aggregate to every lane.
//
// cooperative_groups::reduce over a 32-thread tile lowers to __reduce_*_sync
// (redux.sync) on sm_80+, including bitwise OR, and to a butterfly shuffle on
// sm_70/sm_75, returning the aggregate to every lane in both cases.
// cub::WarpReduce returns the aggregate only to lane 0 and has no redux-backed
// OR reduction, so it is not used here.

#ifndef NVMOLKIT_SUBGRAPH_WARP_REDUCE_CUH
#define NVMOLKIT_SUBGRAPH_WARP_REDUCE_CUH

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstdint>

namespace nvMolKit {

constexpr uint32_t kFullWarpMask = 0xFFFFFFFFu;

namespace detail {
__device__ __forceinline__ cooperative_groups::thread_block_tile<32> fullWarp() {
  return cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
}
}  // namespace detail

__device__ __forceinline__ uint32_t warpReduceAdd(uint32_t value) {
  return cooperative_groups::reduce(detail::fullWarp(), value, cooperative_groups::plus<uint32_t>());
}

__device__ __forceinline__ uint32_t warpReduceOr(uint32_t value) {
  return cooperative_groups::reduce(detail::fullWarp(), value, cooperative_groups::bit_or<uint32_t>());
}

__device__ __forceinline__ uint32_t warpReduceMin(uint32_t value) {
  return cooperative_groups::reduce(detail::fullWarp(), value, cooperative_groups::less<uint32_t>());
}

__device__ __forceinline__ uint32_t warpReduceMax(uint32_t value) {
  return cooperative_groups::reduce(detail::fullWarp(), value, cooperative_groups::greater<uint32_t>());
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBGRAPH_WARP_REDUCE_CUH
