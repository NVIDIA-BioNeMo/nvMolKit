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

#ifndef LOAD_STORE_H
#define LOAD_STORE_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "src/fingerprint_similarity_device.cuh"
#include "src/utils/macros_ptx.cuh"

namespace nvMolKit {

/**
 * @file load_store.cuh
 * @brief Load Global to Shared (lgs) memory for matrix multiplication for TensorOp implementation.
 *
 * This file contains the implementation of loading matrices A and B into shared memory,
 * performing necessary transpositions, and computing k reductions on the loaded data.
 *
 * @tparam T Data type of the matrix elements (e.g., unint32).
 * @tparam BLOCK_TILE_SIZE_X Block tile size in the X dimension.
 * @tparam BLOCK_TILE_SIZE_Y Block tile size in the Y dimension.
 * @tparam BLOCK_TILE_SIZE_K Block tile size in the K dimension (common dimension for matrix multiplication).
 * @tparam NUM_THREADS Number of threads per block.
 *
 * @param A Input matrix A as a span of constant elements.
 * @param lda Leading dimension of matrix A.
 * @param B Input matrix B as a span of constant elements.
 * @param ldb Leading dimension of matrix B.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix B.
 * @param k Common dimension between matrices A and B.
 * @param load_k Current offset in the K dimension for loading.
 * @param A_smem_tile Shared memory tile for matrix A.
 * @param B_smem_tile Shared memory tile for matrix B.
 *
 * ## Input and Output Parameter Ranges
 * - `A`: Non-null, read-only span of size at least `m * k`.
 * - `lda`: Must be at least `k`.
 * - `B`: Non-null, read-only span of size at least `n * k`.
 * - `ldb`: Must be at least `k`.
 * - `m`, `n`, `k`: Non-negative integers.
 * - `load_k`: Non-negative integer, typically less than `k`.
 * - `A_smem_tile`, `B_smem_tile`: Non-null writable arrays.
 * - `A_smem_k_reduced`, `B_smem_k_reduced`: Non-null writable integer arrays.
 *
 * ## Hardware-Software Interfaces
 * - Requires CUDA capable GPU.
 * - Utilizes CUDA atomic operations and warp shuffle functions.
 **/
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS>
__device__ void load_gmem_to_smem_tensorop(
  const cuda::std::span<const T> A,
  const size_t                   lda,
  const cuda::std::span<const T> B,
  const size_t                   ldb,
  const size_t                   m,
  const size_t                   n,
  const size_t                   k,
  const size_t                   load_k,
  T                              A_smem_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K + 1],
  T                              B_smem_tile[BLOCK_TILE_SIZE_X][BLOCK_TILE_SIZE_K + 1]  // take care of bank conflict
) {
  const size_t warpid = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  int          lane_id;
  get_lane(lane_id);

  constexpr size_t NUM_WARPS = NUM_THREADS / 32;
  constexpr size_t A_ITERS   = (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K) / NUM_WARPS;

  NVMOLKIT_UNROLL
  for (int i = lane_id; i < A_ITERS; i += 32) {
    int       linear_idx = i + warpid * A_ITERS;
    const int smem_row   = linear_idx / BLOCK_TILE_SIZE_K;
    const int smem_col   = linear_idx % BLOCK_TILE_SIZE_K;
    const int gmem_row   = smem_row + BLOCK_TILE_SIZE_Y * blockIdx.y;
    const int gmem_col   = smem_col + BLOCK_TILE_SIZE_K * load_k;

    if (gmem_row < m && gmem_col < k) {
      A_smem_tile[smem_row][smem_col] = A[gmem_row * k + gmem_col];
    } else {
      A_smem_tile[smem_row][smem_col] = T(0);
    }
  }

  constexpr size_t B_ITERS = (BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K) / NUM_WARPS;

  NVMOLKIT_UNROLL
  for (int i = lane_id; i < B_ITERS; i += 32) {
    int       linear_idx = i + warpid * B_ITERS;
    const int smem_row   = linear_idx / BLOCK_TILE_SIZE_K;
    const int smem_col   = linear_idx % BLOCK_TILE_SIZE_K;
    const int gmem_row   = smem_row + BLOCK_TILE_SIZE_X * blockIdx.x;
    const int gmem_col   = smem_col + BLOCK_TILE_SIZE_K * load_k;

    if (gmem_row < n && gmem_col < k) {
      B_smem_tile[smem_row][smem_col] = B[gmem_row * k + gmem_col];
    } else {
      B_smem_tile[smem_row][smem_col] = T(0);
    }
  }
}

/**
 * @file load_store.cuh
 * @brief Template function to load data from shared memory into registers.
 *
 * This function loads data from shared memory into registers for kmajor layout
 *
 * @tparam TILE_Y The tile size in the Y dimension.
 * @tparam TILE_K The tile size in the K dimension.
 *
 * @param smem_ptr Pointer to the shared memory.
 * @param groupID The group ID of the thread.
 * @param threadID_in_group The thread ID within the group.
 * @param load_xy The load factor in the XY plane.
 * @param load_k The load factor in the K dimension.
 * @param regs Array to store the loaded data.
 **/
template <size_t TILE_Y, size_t TILE_K>
__forceinline__ __device__ void ld_m16n8k256_x4(const uint32_t* smem_ptr,
                                                int             groupID,
                                                int             threadID_in_group,
                                                int             load_xy,
                                                int             load_k,
                                                unsigned        regs[4]) {
  /*
  smem shape [M * K] assumed KMAJOR  [TILE_K, 1]
  */
  int row, col;
  row     = groupID;
  col     = threadID_in_group;
  regs[0] = *(smem_ptr + (row + load_xy * 16) * TILE_K + (col + 8 * load_k));

  row     = groupID + 8;
  col     = threadID_in_group;
  regs[1] = *(smem_ptr + (row + load_xy * 16) * TILE_K + (col + 8 * load_k));

  row     = groupID;
  col     = threadID_in_group + 4;
  regs[2] = *(smem_ptr + (row + load_xy * 16) * TILE_K + (col + 8 * load_k));

  row     = groupID + 8;
  col     = threadID_in_group + 4;
  regs[3] = *(smem_ptr + (row + load_xy * 16) * TILE_K + (col + 8 * load_k));
}

/**
 * @brief Load function for matrix data from shared memory with configurable layout.
 *
 * This function loads 2 unsigned integers from shared memory into registers. The layout of the data
 * in shared memory is kmajor.
 *
 * @tparam TILE_X The tile size in the X dimension.
 * @tparam TILE_K The tile size in the K dimension.
 *
 * @param smem_ptr Pointer to the shared memory.
 * @param groupID The group ID in the grid.
 * @param threadID_in_group The thread ID within the group.
 * @param load_xy The load index in the X or Y dimension.
 * @param load_k The load index in the K dimension.
 * @param regs Output array where the loaded data will be stored.
 *
 * ## Input and Output Parameter Ranges
 * - `smem_ptr` should point to a valid memory location accessible by the device.
 * - `groupID` should be a non-negative integer less than the number of groups.
 * - `threadID_in_group` should be a non-negative integer less than the number of threads per group.
 * - `load_xy` and `load_k` should be non-negative integers.
 * - `regs` should be an array of size 2 to store the loaded 16-bit values.
 *
 * ## Hardware-Software Interfaces
 * - Requires CUDA capable device with appropriate shared memory and compute capability.
 * - This function is designed to be called from within a CUDA kernel.
 **/
template <size_t TILE_X, size_t TILE_K>
__forceinline__ __device__ void ld_m16n8k256_x2(const uint32_t* smem_ptr,
                                                int             groupID,
                                                int             threadID_in_group,
                                                int             load_xy,
                                                int             load_k,
                                                unsigned        regs[2]) {
  /*
  smem shape [N * K] assumed KMAJOR : [TILE_K, 1]
  */
  int row, col;

  row     = threadID_in_group;
  col     = groupID;
  regs[0] = *(smem_ptr + (row + 8 * load_k) + (col + 8 * load_xy) * TILE_K);

  row     = threadID_in_group + 4;
  regs[1] = *(smem_ptr + (row + 8 * load_k) + (col + 8 * load_xy) * TILE_K);
}

/**
 * @file load_store.cuh
 * @brief Template function for storing computed values in shared memory.
 *
 * This function is designed to handle the storage of computed values into shared memory
 * for both Tanimoto and Cosine calculations based on the template parameter Metric.
 *
 * @tparam TILE_X The tile width in shared memory.
 * @tparam TILE_Y The tile height in shared memory.
 * @tparam Metric Fingerprint similarity metric.
 *
 * @param regs_and Array of unsigned integers containing AND results.
 * @param regs_xor Array of unsigned integers containing XOR results.
 * @param regs_and_not Array of unsigned integers containing AND NOT results (used only in Cosine mode).
 * @param groupID The ID of the group in the grid.
 * @param threadID_in_group The ID of the thread within its group.
 * @param load_y Y-coordinate offset for loading data.
 * @param load_x X-coordinate offset for loading data.
 * @param m The height of the global memory matrix.
 * @param n The width of the global memory matrix.
 * @param smem_ptr Pointer to the shared memory where results will be stored.
 *
 * ## Input and Output Parameter Ranges
 * - `regs_and`, `regs_xor`, `regs_and_not` should be arrays of size 4 containing non-negative integers.
 * - `groupID`, `threadID_in_group` should be non-negative integers.
 * - `load_y`, `load_x` should be non-negative integers and should be within the bounds of the data being processed.
 * - `m`, `n` should be positive integers representing matrix dimensions.
 * - `smem_ptr` should point to a valid memory location with enough space to accommodate the results.
 *
 * ## Hardware-Software Interfaces
 * - This function is expected to be called within a CUDA kernel, hence it interfaces directly with GPU hardware.
 * - Assumes the presence of CUDA-capable hardware and appropriate CUDA drivers.
 **/
template <size_t TILE_X, size_t TILE_Y, FingerprintSimilarityMetric Metric>
__forceinline__ __device__ void st_m16n8k256_x4(unsigned regs_and[4],
                                                unsigned regs_xor[4],
                                                unsigned regs_and_not[4],
                                                int      groupID,
                                                int      threadID_in_group,
                                                int      load_y,
                                                int      load_x,
                                                int      m,
                                                int      n,
                                                float*   smem_ptr) {
  /*
  gmem shape [M * N] assumed NMAJOR : [N, 1]
  */
  int row, col;

  NVMOLKIT_UNROLL
  for (int r = 0; r < 2; r++) {
    NVMOLKIT_UNROLL
    for (int c = 0; c < 2; c++) {
      row = groupID + r * 8 + load_y * 16;
      col = threadID_in_group * 2 + c + 8 * load_x;

      if constexpr (Metric == FingerprintSimilarityMetric::Tanimoto) {
        smem_ptr[row * (TILE_Y + 1) + col] = detail::fingerprintSimilarityFromDenominator<Metric>(
          static_cast<int>(regs_and[2 * r + c]),
          static_cast<float>(regs_and[2 * r + c] + regs_xor[2 * r + c]));
      } else {
        const int popcB = regs_and[2 * r + c] + regs_and_not[2 * r + c];
        const int popcA = regs_and[2 * r + c] + regs_xor[2 * r + c] - regs_and_not[2 * r + c];
        smem_ptr[row * (TILE_Y + 1) + col] =
          detail::fingerprintSimilarity<Metric, float>(static_cast<int>(regs_and[2 * r + c]), popcA, popcB);
      }
    }
  }
}

/**
 * @file load_store.cuh
 * @brief Load Global to Shared (lgs) memory for matrix multiplication for SIMT implementation.
 *
 * This file contains the implementation of loading matrices A and B into shared memory,
 * performing necessary transpositions, and computing k reductions on the loaded data.
 *
 * @tparam T Data type of the matrix elements (e.g., unint32).
 * @tparam BLOCK_TILE_SIZE_X Block tile size in the X dimension.
 * @tparam BLOCK_TILE_SIZE_Y Block tile size in the Y dimension.
 * @tparam BLOCK_TILE_SIZE_K Block tile size in the K dimension (common dimension for matrix multiplication).
 * @tparam NUM_THREADS Number of threads per block.
 *
 * @param A Input matrix A as a span of constant elements.
 * @param lda Leading dimension of matrix A.
 * @param B Input matrix B as a span of constant elements.
 * @param ldb Leading dimension of matrix B.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix B.
 * @param k Common dimension between matrices A and B.
 * @param load_k Current offset in the K dimension for loading.
 * @param A_smem_tile Shared memory tile for matrix A.
 * @param B_smem_tile Shared memory tile for matrix B.
 * @param A_smem_k_reduced Array to store bit reductions of matrix A.
 * @param B_smem_k_reduced Array to store bit reductions of matrix B.
 *
 * ## Input and Output Parameter Ranges
 * - `A`: Non-null, read-only span of size at least `m * k`.
 * - `lda`: Must be at least `k`.
 * - `B`: Non-null, read-only span of size at least `n * k`.
 * - `ldb`: Must be at least `k`.
 * - `m`, `n`, `k`: Non-negative integers.
 * - `load_k`: Non-negative integer, typically less than `k`.
 * - `A_smem_tile`, `B_smem_tile`: Non-null writable arrays.
 * - `A_smem_k_reduced`, `B_smem_k_reduced`: Non-null writable integer arrays.
 *
 * ## Hardware-Software Interfaces
 * - Requires CUDA capable GPU.
 * - Utilizes CUDA atomic operations and warp shuffle functions.
 **/
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS>
__device__ void load_gmem_to_smem_simt(const cuda::std::span<const T> A,
                                       const size_t                   lda,
                                       const cuda::std::span<const T> B,
                                       const size_t                   ldb,
                                       const size_t                   m,
                                       const size_t                   n,
                                       const size_t                   k,
                                       const size_t                   load_k,
                                       T   A_smem_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + 1],
                                       T   B_smem_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + 1],
                                       int A_smem_k_reduced[BLOCK_TILE_SIZE_Y],
                                       int B_smem_k_reduced[BLOCK_TILE_SIZE_X]) {
  // little to no effect on perf
  const size_t warpid = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  int          lane_id;
  get_lane(lane_id);

  constexpr int NUM_WARPS = NUM_THREADS / 32;

  NVMOLKIT_UNROLL
  for (int i = lane_id; i < ((BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K) / NUM_WARPS); i += 32) {
    int linear_idx = i + warpid * ((BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K) / NUM_WARPS);

    const int smem_row = linear_idx / BLOCK_TILE_SIZE_K;
    const int smem_col = linear_idx % BLOCK_TILE_SIZE_K;

    const int gmem_row = smem_row + BLOCK_TILE_SIZE_Y * blockIdx.y;
    const int gmem_col = smem_col + BLOCK_TILE_SIZE_K * load_k;

    T tmp = {0};
    if (gmem_row < m && gmem_col < k) {
      tmp = A[gmem_row * lda + gmem_col];
    }

    // transpose
    A_smem_tile[smem_col][smem_row] = tmp;
    atomicAdd(&A_smem_k_reduced[smem_row], __popc(tmp));
  }

  NVMOLKIT_UNROLL
  for (int i = lane_id; i < ((BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K) / NUM_WARPS); i += 32) {
    int       linear_idx = i + warpid * ((BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K) / NUM_WARPS);
    const int smem_row   = linear_idx / BLOCK_TILE_SIZE_K;
    const int smem_col   = linear_idx % BLOCK_TILE_SIZE_K;
    const int gmem_row   = smem_row + BLOCK_TILE_SIZE_X * blockIdx.x;
    const int gmem_col   = smem_col + BLOCK_TILE_SIZE_K * load_k;

    T tmp = {0};

    if (gmem_row < n && gmem_col < k) {
      tmp = B[gmem_row * ldb + gmem_col];
    }

    // transpose
    B_smem_tile[smem_col][smem_row] = tmp;
    atomicAdd(&B_smem_k_reduced[smem_row], __popc(tmp));
  }
}

}  // namespace nvMolKit

#endif
