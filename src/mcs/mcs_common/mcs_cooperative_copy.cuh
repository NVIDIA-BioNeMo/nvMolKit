// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef MCS_COMMON_MCS_COOPERATIVE_COPY_CUH
#define MCS_COMMON_MCS_COOPERATIVE_COPY_CUH

namespace mcs {

/// Warp-cooperative coalesced copy via int4 (16 B) loads/stores.
/// Both dst and src should be 16-byte aligned for full throughput;
/// any tail bytes (nbytes % 16) are handled at int (4 B) granularity.
template <typename WarpT>
__forceinline__ __device__ void warpCopy(const WarpT& warp,
                                         void* __restrict__ dst,
                                         const void* __restrict__ src,
                                         int nbytes) {
  const int threadRank = static_cast<int>(warp.thread_rank());
  const int numThreads = static_cast<int>(warp.num_threads());

  auto*       dst16         = static_cast<int4*>(dst);
  const auto* src16         = static_cast<const int4*>(src);
  const int   numWideChunks = nbytes / 16;
  for (int i = threadRank; i < numWideChunks; i += numThreads) {
    dst16[i] = src16[i];
  }

  const int tailBytes = nbytes - numWideChunks * 16;
  if (tailBytes > 0) {
    auto*       dstTail       = reinterpret_cast<int*>(dst16 + numWideChunks);
    const auto* srcTail       = reinterpret_cast<const int*>(src16 + numWideChunks);
    const int   numTailChunks = tailBytes / 4;
    for (int i = threadRank; i < numTailChunks; i += numThreads) {
      dstTail[i] = srcTail[i];
    }
  }
}

}  // namespace mcs

#endif  // MCS_COMMON_MCS_COOPERATIVE_COPY_CUH
