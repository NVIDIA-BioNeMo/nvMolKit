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

#ifndef NVMOLKIT_P2P_H
#define NVMOLKIT_P2P_H

#include <cuda_runtime.h>

#include <cstddef>

namespace nvMolKit {

/**
 * @brief Asynchronously copy a contiguous buffer from one GPU to another.
 *
 * The copy is ordered after any work previously enqueued on @p srcStream and before any
 * work subsequently enqueued on @p dstStream that depends on the destination.
 *
 * Caller responsibilities:
 *   - Allocations on both GPUs must be valid and large enough for the requested transfer.
 *   - When @p srcGpu == @p dstGpu the helper degenerates to a single in-stream
 *     `cudaMemcpyAsync`; in that case @p srcStream and @p dstStream should be the same.
 *
 * Synchronization model (cross-GPU case):
 *   1. Record an event on @p srcStream so any pending writes on the source buffer complete
 *      before the copy starts.
 *   2. Make @p dstStream wait on that event, then enqueue `cudaMemcpyPeerAsync` on
 *      @p dstStream so all downstream consumers ordered against @p dstStream observe the
 *      result.
 *
 * `cudaMemcpyPeerAsync` itself succeeds regardless of whether explicit peer access has been
 * enabled, but performance is materially better when it has been; calling
 * @ref enablePeerAccess once during driver setup is recommended.
 *
 * @param dstDevice  Pointer to the destination buffer (resides on @p dstGpu).
 * @param srcDevice  Pointer to the source buffer (resides on @p srcGpu).
 * @param byteCount  Number of bytes to copy. A zero count is a no-op.
 * @param srcGpu     CUDA device id where @p srcDevice was allocated.
 * @param srcStream  Stream that produced the source data. Must belong to @p srcGpu.
 * @param dstGpu     CUDA device id where @p dstDevice was allocated.
 * @param dstStream  Stream on which the copy is enqueued. Must belong to @p dstGpu.
 */
void copyDeviceToDeviceAsync(void*        dstDevice,
                             const void*  srcDevice,
                             std::size_t  byteCount,
                             int          srcGpu,
                             cudaStream_t srcStream,
                             int          dstGpu,
                             cudaStream_t dstStream);

/**
 * @brief Enable bidirectional P2P access between two GPUs.
 *
 * Calls `cudaDeviceEnablePeerAccess` in both directions. The
 * `cudaErrorPeerAccessAlreadyEnabled` status is treated as success so this function is
 * idempotent. Any other CUDA failure (including the link not being supported per
 * `cudaDeviceCanAccessPeer`) is reported as a thrown exception.
 *
 * @throws std::runtime_error If either direction cannot be enabled.
 */
void enablePeerAccess(int gpuA, int gpuB);

}  // namespace nvMolKit

#endif  // NVMOLKIT_P2P_H
