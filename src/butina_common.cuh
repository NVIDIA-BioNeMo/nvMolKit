// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_BUTINA_COMMON_CUH
#define NVMOLKIT_BUTINA_COMMON_CUH

#include <cuda_runtime.h>

#include <cstdint>
#include <cuda/std/span>
#include <utility>

#include "src/utils/cuda_error_check.h"

namespace nvMolKit {

// Pack a neighbor count and point index so an unsigned maximum prefers higher counts, then higher indices. (Later we
// will make the reordering = true path use this and the next helper)
__device__ __forceinline__ std::uint64_t makeButinaCandidate(const int value, const int index) {
  if (value < 0) {
    return 0;
  }
  const auto encodedValue = static_cast<std::uint32_t>(value) + 1U;
  return (static_cast<std::uint64_t>(encodedValue) << 32) | static_cast<std::uint32_t>(index);
}

// Decode a candidate produced by makeButinaCandidate().
__device__ __forceinline__ int storeButinaCandidate(const std::uint64_t candidate, int* value, int* index) {
  const auto encodedValue = static_cast<std::uint32_t>(candidate >> 32);
  *value                  = encodedValue == 0 ? -1 : static_cast<int>(encodedValue) - 1;
  *index                  = encodedValue == 0 ? -1 : static_cast<int>(static_cast<std::uint32_t>(candidate));
  return *value;
}

// Update a conditional WHILE graph from a device-side scalar threshold.
static __global__ void setConditionalLoopGraphCondition(cudaGraphConditionalHandle handle,
                                                        const int*                 value,
                                                        const int                  threshold) {
  cudaGraphSetConditional(handle, *value >= threshold ? 1 : 0);
}

/**
 * @brief Own a CUDA graph containing one conditional WHILE node.
 *
 * The callback receives the capture stream and conditional handle used to
 * populate the loop body. The handle starts enabled, giving the graph
 * do-while semantics; the captured body is responsible for updating it.
 */
class ConditionalLoopGraph {
 public:
  template <typename CaptureBody> explicit ConditionalLoopGraph(CaptureBody&& captureBody) {
    cudaCheckError(cudaGraphCreate(&graph_, 0));
    cudaCheckError(cudaGraphConditionalHandleCreate(&handle_, graph_, 1, cudaGraphCondAssignDefault));

    cudaGraphNodeParams params = {};
    params.type                = cudaGraphNodeTypeConditional;
    params.conditional.handle  = handle_;
    params.conditional.type    = cudaGraphCondTypeWhile;
    params.conditional.size    = 1;
    cudaGraphNode_t conditionalNode;
#if CUDART_VERSION >= 13000
    cudaCheckError(cudaGraphAddNode(&conditionalNode, graph_, nullptr, nullptr, 0, &params));
#else
    cudaCheckError(cudaGraphAddNode(&conditionalNode, graph_, nullptr, 0, &params));
#endif

    cudaStream_t captureStream;
    cudaCheckError(cudaStreamCreate(&captureStream));
    cudaCheckError(cudaStreamBeginCaptureToGraph(captureStream,
                                                 params.conditional.phGraph_out[0],
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 cudaStreamCaptureModeRelaxed));
    std::forward<CaptureBody>(captureBody)(captureStream, handle_);
    cudaCheckError(cudaStreamEndCapture(captureStream, nullptr));
    cudaCheckError(cudaStreamDestroy(captureStream));
    cudaCheckError(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
  }

  ~ConditionalLoopGraph() {
    if (graphExec_) {
      cudaGraphExecDestroy(graphExec_);
    }
    if (graph_) {
      cudaGraphDestroy(graph_);
    }
  }

  ConditionalLoopGraph(const ConditionalLoopGraph&)            = delete;
  ConditionalLoopGraph& operator=(const ConditionalLoopGraph&) = delete;

  void launch(cudaStream_t stream) const { cudaCheckError(cudaGraphLaunch(graphExec_, stream)); }

 private:
  cudaGraph_t                graph_     = nullptr;
  cudaGraphExec_t            graphExec_ = nullptr;
  cudaGraphConditionalHandle handle_    = {};
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_BUTINA_COMMON_CUH
