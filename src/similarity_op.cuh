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

#ifndef NVMOLKIT_SIMILARITY_OP_H
#define NVMOLKIT_SIMILARITY_OP_H
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "src/fingerprint_similarity_device.cuh"
#include "src/utils/macros_ptx.cuh"

namespace nvMolKit {

template <bool _IsTensorOp, FingerprintSimilarityMetric Metric> struct SimilarityOp {
  __device__ void operator()(uint32_t d_and[4], uint32_t d_xor[4], uint32_t d_and_not[4]) {}
};

template <> struct SimilarityOp<true, FingerprintSimilarityMetric::Tanimoto> {
  __forceinline__ __device__ void operator()(uint32_t d_and[4],
                                             uint32_t d_xor[4],
                                             uint32_t d_and_not[4],
                                             uint32_t a_regs[4],
                                             uint32_t b_regs[2]) {
    bmma_and_m16n8k256(d_and[0],
                       d_and[1],
                       d_and[2],
                       d_and[3],
                       a_regs[0],
                       a_regs[1],
                       a_regs[2],
                       a_regs[3],
                       b_regs[0],
                       b_regs[1],
                       d_and[0],
                       d_and[1],
                       d_and[2],
                       d_and[3]);

    bmma_xor_m16n8k256(d_xor[0],
                       d_xor[1],
                       d_xor[2],
                       d_xor[3],
                       a_regs[0],
                       a_regs[1],
                       a_regs[2],
                       a_regs[3],
                       b_regs[0],
                       b_regs[1],
                       d_xor[0],
                       d_xor[1],
                       d_xor[2],
                       d_xor[3]);
  }
};

template <> struct SimilarityOp<true, FingerprintSimilarityMetric::Cosine> {
  __device__ void operator()(uint32_t d_and[4],
                             uint32_t d_xor[4],
                             uint32_t d_and_not[4],
                             uint32_t a_regs[4],
                             uint32_t b_regs[2]) {
    bmma_and_m16n8k256(d_and[0],
                       d_and[1],
                       d_and[2],
                       d_and[3],
                       a_regs[0],
                       a_regs[1],
                       a_regs[2],
                       a_regs[3],
                       b_regs[0],
                       b_regs[1],
                       d_and[0],
                       d_and[1],
                       d_and[2],
                       d_and[3]);

    bmma_and_m16n8k256(d_and_not[0],
                       d_and_not[1],
                       d_and_not[2],
                       d_and_not[3],
                       ~a_regs[0],
                       ~a_regs[1],
                       ~a_regs[2],
                       ~a_regs[3],
                       b_regs[0],
                       b_regs[1],
                       d_and_not[0],
                       d_and_not[1],
                       d_and_not[2],
                       d_and_not[3]);

    bmma_xor_m16n8k256(d_xor[0],
                       d_xor[1],
                       d_xor[2],
                       d_xor[3],
                       a_regs[0],
                       a_regs[1],
                       a_regs[2],
                       a_regs[3],
                       b_regs[0],
                       b_regs[1],
                       d_xor[0],
                       d_xor[1],
                       d_xor[2],
                       d_xor[3]);
  }
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_SIMILARITY_OP_H
