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
// __reduce_*_sync requires sm_80. nvMolKit also builds for sm_70/sm_75, which
// fall back to a butterfly shuffle producing the same values.
//
// cub::WarpReduce is deliberately not used here. Its sm_80+ fast path lowers to
// these same __reduce_*_sync intrinsics, so there is nothing to gain, and it is
// worse in two ways: its contract returns the aggregate only to lane 0, while
// the min/max/or call sites need the all-lane broadcast that redux.sync
// provides for free (CUB would need an extra __shfl_sync); and CUB has no
// redux-backed OR reduction at all, so warpReduceOr would regress to a
// five-step shuffle loop even on sm_80+.

#ifndef NVMOLKIT_SUBGRAPH_WARP_REDUCE_CUH
#define NVMOLKIT_SUBGRAPH_WARP_REDUCE_CUH

#include <cstdint>

namespace nvMolKit {

constexpr uint32_t kFullWarpMask = 0xFFFFFFFFu;

__device__ __forceinline__ uint32_t warpReduceAdd(uint32_t value) {
#if __CUDA_ARCH__ >= 800
  return __reduce_add_sync(kFullWarpMask, value);
#else
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFullWarpMask, value, offset);
  }
  return value;
#endif
}

__device__ __forceinline__ uint32_t warpReduceOr(uint32_t value) {
#if __CUDA_ARCH__ >= 800
  return __reduce_or_sync(kFullWarpMask, value);
#else
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value |= __shfl_xor_sync(kFullWarpMask, value, offset);
  }
  return value;
#endif
}

__device__ __forceinline__ uint32_t warpReduceMin(uint32_t value) {
#if __CUDA_ARCH__ >= 800
  return __reduce_min_sync(kFullWarpMask, value);
#else
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = min(value, __shfl_xor_sync(kFullWarpMask, value, offset));
  }
  return value;
#endif
}

__device__ __forceinline__ uint32_t warpReduceMax(uint32_t value) {
#if __CUDA_ARCH__ >= 800
  return __reduce_max_sync(kFullWarpMask, value);
#else
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = max(value, __shfl_xor_sync(kFullWarpMask, value, offset));
  }
  return value;
#endif
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBGRAPH_WARP_REDUCE_CUH
