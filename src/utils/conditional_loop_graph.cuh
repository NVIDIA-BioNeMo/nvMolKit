// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_CONDITIONAL_LOOP_GRAPH_CUH
#define NVMOLKIT_CONDITIONAL_LOOP_GRAPH_CUH

#include <cuda_runtime.h>

#include <type_traits>
#include <utility>

#include "src/utils/cuda_error_check.h"

namespace nvMolKit {

/**
 * @brief Own a CUDA graph containing one conditional WHILE node.
 *
 * The callback receives the capture stream and conditional handle used to
 * populate the loop body. The handle starts enabled, giving the graph
 * do-while semantics; the captured body is responsible for updating it.
 */
class ConditionalLoopGraph {
 public:
  template <typename CaptureBody>
    requires(!std::is_same_v<std::remove_cvref_t<CaptureBody>, ConditionalLoopGraph>)
  explicit ConditionalLoopGraph(CaptureBody&& captureBody) {
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

#endif  // NVMOLKIT_CONDITIONAL_LOOP_GRAPH_CUH
