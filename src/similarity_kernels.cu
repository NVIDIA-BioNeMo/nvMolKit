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

#include <cub/cub.cuh>
#include <cuda/std/span>

#include "device_vector.h"
#include "load_store.cuh"
#include "similarity_kernels.h"
#include "similarity_op.cuh"

namespace nvMolKit {

using internal::kNumThreadsPerFingerPrintTemporaryFixed;

namespace {

//! Detects if we can use BMMA tensor operations for the given device compute capability,
//! taking into account compile-time targets.
bool supportsTensorOps(const int major, const int minor) {
  // BMMA m16n8k256 .b1 {.and,.xor}.popc is supported on sm_80+ per PTX ISA, including Blackwell.
  // We explicitly support Ampere/Ada (8.x), Hopper (9.0), and Blackwell sm_100 / sm_120.
  if (major != 8 && major != 9 && major != 10 && major != 12) {
    return false;
  }

  // Now we do compile time checks. Account for forward compatibilty,
  // so if we're built for 80, we can run on 80, 86, 89
  if constexpr (NVMOLKIT_CUDA_CC_80) {
    return true;
  }
  // If built or 86, we can run on 86 and 89.
  if (NVMOLKIT_CUDA_CC_86 && minor >= 6) {
    return true;
  }
  // If built for 89, we can run on 89.
  if (NVMOLKIT_CUDA_CC_89 && minor == 9) {
    return true;
  }
  if (NVMOLKIT_CUDA_CC_90 && major == 9) {
    return true;
  }
  // Blackwell builds are per-arch (-real); each macro only matches its exact SM.
  if (NVMOLKIT_CUDA_CC_100 && major == 10 && minor == 0) {
    return true;
  }
  if (NVMOLKIT_CUDA_CC_103 && major == 10 && minor == 3) {
    return true;
  }
  if (NVMOLKIT_CUDA_CC_120 && major == 12 && minor == 0) {
    return true;
  }
  return false;
}

//! Cast a device vector to a span of a type smaller or equal to self.
//! Does some basic checking of compatibility between types.
template <typename T, typename OtherT>
[[maybe_unused]] cuda::std::span<OtherT> castAsSpanOfSmallerType(AsyncDeviceVector<T>& vec) {
  static_assert(sizeof(OtherT) <= sizeof(T), "Size of smaller type must be less than or equal to the larger type.");
  static_assert(sizeof(T) % sizeof(OtherT) == 0, "Size of smaller type must be a divisor of the larger type.");
  constexpr int sizeMultiplier = sizeof(T) / sizeof(OtherT);
  assert(vec.size() > 0);
  return cuda::std::span<OtherT>(reinterpret_cast<OtherT*>(vec.data()), vec.size() * sizeMultiplier);
}
template <typename T, typename OtherT>
cuda::std::span<const OtherT> castAsSpanOfSmallerType(const AsyncDeviceVector<T>& vec) {
  static_assert(sizeof(OtherT) <= sizeof(T), "Size of smaller type must be less than or equal to the larger type.");
  static_assert(sizeof(T) % sizeof(OtherT) == 0, "Size of smaller type must be a divisor of the larger type.");
  constexpr int sizeMultiplier = sizeof(T) / sizeof(OtherT);
  assert(vec.size() > 0);
  return cuda::std::span<OtherT>(reinterpret_cast<OtherT*>(vec.data()), vec.size() * sizeMultiplier);
}

enum class SimilarityType {
  Tanimoto = 0,
  Cosine
};

// --------------------------------
// SIMT & TensorOp Kernel Template
// --------------------------------

template <SimilarityType similarityType,
          typename kThreadReductionType,
          typename T_out,
          size_t BLOCK_TILE_SIZE_X,
          size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K,
          size_t NUM_WARP_X,
          size_t NUM_WARP_Y>
__device__ void crossSimilarityKernelTensorOp(const cuda::std::span<const kThreadReductionType> bitsOne,
                                              const cuda::std::span<const kThreadReductionType> bitsTwo,
                                              const size_t                                      elementsPerMolecule,
                                              cuda::std::span<T_out>                            similarities,
                                              const size_t                                      offset) {
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 8;
  constexpr int WMMA_K = 256;

  const size_t m = bitsOne.size() / elementsPerMolecule;
  const size_t n = bitsTwo.size() / elementsPerMolecule;
  const size_t k = elementsPerMolecule;

  constexpr int NUM_THREADS = NUM_WARP_X * 32 * NUM_WARP_Y;

  constexpr bool IS_TANIMOTO = (similarityType == SimilarityType::Tanimoto);

  constexpr int WARP_TILE_X = BLOCK_TILE_SIZE_X / NUM_WARP_X;  // BLOCK_TILE_SIZE_X % NUM_WARP_X == 0
  constexpr int WARP_TILE_Y = BLOCK_TILE_SIZE_Y / NUM_WARP_Y;  // BLOCK_TILE_SIZE_Y % NUM_WARP_Y == 0

  constexpr int NUM_M_PER_WARP =
    (BLOCK_TILE_SIZE_Y / (WMMA_M * NUM_WARP_Y));  // BLOCK_TILE_SIZE_Y % (WMMA_M * NUM_WARP_Y) == 0
  constexpr int NUM_N_PER_WARP =
    (BLOCK_TILE_SIZE_X / (WMMA_N * NUM_WARP_X));  // BLOCK_TILE_SIZE_X / (WMMA_N * NUM_WARP_X) == 0

  constexpr int D_SEGMENTS = NUM_M_PER_WARP * NUM_N_PER_WARP;

  unsigned d_and[D_SEGMENTS][4]{0};
  unsigned d_xor[D_SEGMENTS][4]{0};
  unsigned d_and_not[D_SEGMENTS][4]{0};

  unsigned a_regs[4];
  unsigned b_regs[2];

  SimilarityOp<true, IS_TANIMOTO> similarityOp;

  int lane_id;
  get_lane(lane_id);

  int groupID = lane_id >> 2;

  int threadID_in_group = lane_id % 4;

  const int warpid = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

  int warp_id_x = __shfl_sync(0xffffffff, (threadIdx.x / 32) % NUM_WARP_X, 0);
  int warp_id_y = __shfl_sync(0xffffffff, (threadIdx.x / 32) / NUM_WARP_X, 0);

  __shared__ kThreadReductionType AB_smem[2 * (BLOCK_TILE_SIZE_X + BLOCK_TILE_SIZE_Y)][BLOCK_TILE_SIZE_K + 1];

  // Create pointers to split the shared memory
  kThreadReductionType(*A_smem)[BLOCK_TILE_SIZE_K + 1] = (kThreadReductionType(*)[BLOCK_TILE_SIZE_K + 1]) AB_smem;
  kThreadReductionType(*B_smem)[BLOCK_TILE_SIZE_K + 1] =
    (kThreadReductionType(*)[BLOCK_TILE_SIZE_K + 1]) & AB_smem[BLOCK_TILE_SIZE_Y];

  constexpr int NUM_WMMA_K_LOADS_PER_BLOCK = (BLOCK_TILE_SIZE_K + (WMMA_K / 32) - 1) / (WMMA_K / 32);
  const int     NUM_K_LOADS_PER_BLOCK      = (k + BLOCK_TILE_SIZE_K - 1) / (BLOCK_TILE_SIZE_K);

  NVMOLKIT_UNROLL
  for (int load_k = 0; load_k < NUM_K_LOADS_PER_BLOCK; load_k++) {
    load_gmem_to_smem_tensorop<kThreadReductionType,
                               BLOCK_TILE_SIZE_X,
                               BLOCK_TILE_SIZE_Y,
                               BLOCK_TILE_SIZE_K,
                               NUM_THREADS>(bitsOne, k, bitsTwo, k, m, n, k, load_k, A_smem, B_smem);

    __syncthreads();

    NVMOLKIT_UNROLL
    for (int idk = 0; idk < NUM_WMMA_K_LOADS_PER_BLOCK; idk++) {
      uint32_t* A_smem_ptr = reinterpret_cast<uint32_t*>(&A_smem[warp_id_y * WARP_TILE_Y][idk * (WMMA_K / 32)]);
      uint32_t* B_smem_ptr = reinterpret_cast<uint32_t*>(&B_smem[warp_id_x * WARP_TILE_X][idk * (WMMA_K / 32)]);

      NVMOLKIT_UNROLL
      for (int m_idx = 0; m_idx < NUM_M_PER_WARP; m_idx++) {
        ld_m16n8k256_x4<BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K + 1>(A_smem_ptr,
                                                                  groupID,
                                                                  threadID_in_group,
                                                                  m_idx,
                                                                  0,
                                                                  a_regs);

        NVMOLKIT_UNROLL
        for (int n_idx = 0; n_idx < NUM_N_PER_WARP; n_idx++) {
          ld_m16n8k256_x2<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_K + 1>(B_smem_ptr,
                                                                    groupID,
                                                                    threadID_in_group,
                                                                    n_idx,
                                                                    0,
                                                                    b_regs);

          similarityOp(d_and[m_idx * NUM_N_PER_WARP + n_idx],
                       d_xor[m_idx * NUM_N_PER_WARP + n_idx],
                       d_and_not[m_idx * NUM_N_PER_WARP + n_idx],
                       a_regs,
                       b_regs);
        }
      }
    }
    __syncthreads();
  }

  float* C_smem = reinterpret_cast<float*>(AB_smem);

  NVMOLKIT_UNROLL
  for (int m_idx = 0; m_idx < NUM_M_PER_WARP; m_idx++) {
    NVMOLKIT_UNROLL
    for (int n_idx = 0; n_idx < NUM_N_PER_WARP; n_idx++) {
      st_m16n8k256_x4<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, IS_TANIMOTO>(d_and[m_idx * NUM_N_PER_WARP + n_idx],
                                                                         d_xor[m_idx * NUM_N_PER_WARP + n_idx],
                                                                         d_and_not[m_idx * NUM_N_PER_WARP + n_idx],
                                                                         groupID,
                                                                         threadID_in_group,
                                                                         m_idx + warp_id_y * (WARP_TILE_Y / WMMA_M),
                                                                         n_idx + warp_id_x * (WARP_TILE_X / WMMA_N),
                                                                         m,
                                                                         n,
                                                                         C_smem);
    }
  }
  __syncthreads();

  constexpr size_t C_ITERS = (BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y) / (NUM_WARP_X * NUM_WARP_Y);

  NVMOLKIT_UNROLL
  for (int i = lane_id; i < C_ITERS; i += 32) {
    int       linear_idx = i + warpid * C_ITERS;
    const int smem_row   = linear_idx / BLOCK_TILE_SIZE_X;
    const int smem_col   = linear_idx % BLOCK_TILE_SIZE_X;
    const int gmem_row   = smem_row + BLOCK_TILE_SIZE_Y * blockIdx.y;
    const int gmem_col   = smem_col + BLOCK_TILE_SIZE_X * blockIdx.x;

    if (gmem_row < m && gmem_col < n) {
      similarities[gmem_row * n + gmem_col] = (C_smem[smem_row * (BLOCK_TILE_SIZE_X + 1) + smem_col]);
    }
  }
}

template <SimilarityType similarityType,
          typename kThreadReductionType,
          typename T_out,
          size_t BLOCK_TILE_SIZE_X,
          size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K,
          size_t THREAD_TILE_X,
          size_t THREAD_TILE_Y>
__device__ void crossSimilarityKernelSIMT(const cuda::std::span<const kThreadReductionType> bitsOne,
                                          const cuda::std::span<const kThreadReductionType> bitsTwo,
                                          const size_t                                      elementsPerMolecule,
                                          cuda::std::span<T_out>                            similarities,
                                          const size_t                                      offset) {
  const size_t numMoleculesTotalOne = bitsOne.size() / elementsPerMolecule;
  const size_t numMoleculesTotalTwo = bitsTwo.size() / elementsPerMolecule;

  constexpr size_t NUM_THREADS{(BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_X) / (THREAD_TILE_Y * THREAD_TILE_X)};

  __shared__ kThreadReductionType A_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + 1];
  __shared__ kThreadReductionType B_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + 1];

  __shared__ int A_k_reduced[BLOCK_TILE_SIZE_Y];
  __shared__ int B_k_reduced[BLOCK_TILE_SIZE_X];
  const size_t   tid = threadIdx.x;

  NVMOLKIT_UNROLL
  for (int load_a = 0U; load_a < (NUM_THREADS + BLOCK_TILE_SIZE_Y - 1) / (NUM_THREADS); load_a++) {
    const int row_As = tid + load_a * NUM_THREADS;
    if (row_As < BLOCK_TILE_SIZE_Y) {
      A_k_reduced[row_As] = kThreadReductionType(0);
    }
  }
  NVMOLKIT_UNROLL
  for (int load_b = 0U; load_b < (NUM_THREADS + BLOCK_TILE_SIZE_X - 1) / (NUM_THREADS); load_b++) {
    const int row_Bs = tid + load_b * NUM_THREADS;
    if (row_Bs < BLOCK_TILE_SIZE_X) {
      B_k_reduced[row_Bs] = kThreadReductionType(0);
    }
  }

  __syncthreads();

  const int NUM_K_LOADS_PER_BLOCK = (elementsPerMolecule + BLOCK_TILE_SIZE_K - 1) / (BLOCK_TILE_SIZE_K);

  int C_thread_results[THREAD_TILE_X][THREAD_TILE_Y] = {0};

  kThreadReductionType A_regs[THREAD_TILE_Y];
  kThreadReductionType B_regs[THREAD_TILE_X];

  NVMOLKIT_UNROLL
  for (int load_k = 0; load_k < NUM_K_LOADS_PER_BLOCK; load_k++) {
    load_gmem_to_smem_simt<kThreadReductionType, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, NUM_THREADS>(
      bitsOne,
      elementsPerMolecule,
      bitsTwo,
      elementsPerMolecule,
      numMoleculesTotalOne,
      numMoleculesTotalTwo,
      elementsPerMolecule,
      load_k,
      A_tile,
      B_tile,
      A_k_reduced,
      B_k_reduced);

    __syncthreads();

    NVMOLKIT_UNROLL
    for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i) {
      NVMOLKIT_UNROLL
      for (size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_Y; thread_tile_row_idx++) {
        A_regs[thread_tile_row_idx] =
          A_tile[k_i][threadIdx.x / (BLOCK_TILE_SIZE_X / THREAD_TILE_X) * THREAD_TILE_Y + thread_tile_row_idx];
      }

      NVMOLKIT_UNROLL
      for (size_t thread_tile_col_idx{0U}; thread_tile_col_idx < THREAD_TILE_X; thread_tile_col_idx++) {
        B_regs[thread_tile_col_idx] =
          B_tile[k_i][threadIdx.x % (BLOCK_TILE_SIZE_X / THREAD_TILE_X) * THREAD_TILE_X + thread_tile_col_idx];
      }

      NVMOLKIT_UNROLL
      for (size_t thread_tile_col_idx{0U}; thread_tile_col_idx < THREAD_TILE_X; thread_tile_col_idx++) {
        NVMOLKIT_UNROLL
        for (size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_Y; thread_tile_row_idx++) {
          C_thread_results[thread_tile_col_idx][thread_tile_row_idx] +=
            __popc(A_regs[thread_tile_row_idx] & B_regs[thread_tile_col_idx]);
        }
      }
    }
    __syncthreads();
  }

  // store to global mem
  NVMOLKIT_UNROLL
  for (size_t thread_tile_row_idx{0U}; thread_tile_row_idx < THREAD_TILE_Y; ++thread_tile_row_idx) {
    const int A_row_oneBits =
      A_k_reduced[threadIdx.x / (BLOCK_TILE_SIZE_X / THREAD_TILE_X) * THREAD_TILE_Y + thread_tile_row_idx];
    NVMOLKIT_UNROLL
    for (size_t thread_tile_col_idx{0U}; thread_tile_col_idx < THREAD_TILE_X; ++thread_tile_col_idx) {
      size_t const C_row_idx{threadIdx.x / (BLOCK_TILE_SIZE_X / THREAD_TILE_X) * THREAD_TILE_Y + thread_tile_row_idx +
                             blockIdx.y * BLOCK_TILE_SIZE_Y};
      size_t const C_col_idx{threadIdx.x % (BLOCK_TILE_SIZE_X / THREAD_TILE_X) * THREAD_TILE_X + thread_tile_col_idx +
                             blockIdx.x * BLOCK_TILE_SIZE_X};

      const int B_row_oneBits =
        B_k_reduced[threadIdx.x % (BLOCK_TILE_SIZE_X / THREAD_TILE_X) * THREAD_TILE_X + thread_tile_col_idx];

      T_out tmp;
      if constexpr (similarityType == SimilarityType::Tanimoto) {
        tmp = static_cast<T_out>(C_thread_results[thread_tile_col_idx][thread_tile_row_idx]) /
              static_cast<T_out>(max(1,
                                     A_row_oneBits + B_row_oneBits -
                                       static_cast<int>(C_thread_results[thread_tile_col_idx][thread_tile_row_idx])));
      } else {
        T_out denom = sqrt(static_cast<T_out>(A_row_oneBits) * static_cast<T_out>(B_row_oneBits));

        tmp = (C_thread_results[thread_tile_col_idx][thread_tile_row_idx] == 0 || denom == 0.0f) ?
                0.0f :
                static_cast<T_out>(C_thread_results[thread_tile_col_idx][thread_tile_row_idx]) / denom;
      }
      if (C_row_idx < numMoleculesTotalOne && C_col_idx < numMoleculesTotalTwo) {
        similarities[C_row_idx * numMoleculesTotalTwo + C_col_idx] = tmp;
      }
    }
  }
}

// --------------------------------
// Tanimoto similarity kernels
// --------------------------------

template <typename kThreadReductionType,
          size_t BLOCK_TILE_SIZE_X,
          size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K,
          size_t THREAD_TILE_X_OR_NUM_WARP_X,
          size_t THREAD_TILE_Y_OR_NUM_WARP_Y>
__global__ void tanimotoCrossSimilarityKernel(const cuda::std::span<const kThreadReductionType> bitsOne,
                                              const cuda::std::span<const kThreadReductionType> bitsTwo,
                                              const size_t                                      elementsPerMolecule,
                                              cuda::std::span<double>                           similarities,
                                              const size_t                                      offset) {
#if __CUDA_ARCH__ >= 800
  crossSimilarityKernelTensorOp<SimilarityType::Tanimoto,
                                kThreadReductionType,
                                double,
                                BLOCK_TILE_SIZE_X,
                                BLOCK_TILE_SIZE_Y,
                                BLOCK_TILE_SIZE_K,
                                THREAD_TILE_X_OR_NUM_WARP_X,
                                THREAD_TILE_Y_OR_NUM_WARP_Y>(bitsOne,
                                                             bitsTwo,
                                                             elementsPerMolecule,
                                                             similarities,
                                                             offset);
#else
  crossSimilarityKernelSIMT<SimilarityType::Tanimoto,
                            kThreadReductionType,
                            double,
                            BLOCK_TILE_SIZE_X,
                            BLOCK_TILE_SIZE_Y,
                            BLOCK_TILE_SIZE_K,
                            THREAD_TILE_X_OR_NUM_WARP_X,
                            THREAD_TILE_Y_OR_NUM_WARP_Y>(bitsOne, bitsTwo, elementsPerMolecule, similarities, offset);

#endif
}

}  // namespace

// --------------------------------
// Tanimoto similarity launch functions
// --------------------------------

//! Launches a kernel to compute the all-to-all Tanimoto similarity between a list of fingerprints.
//! \param bits The list of fingerprints
//! \param results The output similarity, not safe to use until a sync.
template <typename kThreadReductionType>
void launchCrossTanimotoSimilarity(const AsyncDeviceVector<internal::kBlockType>& bitsOne,
                                   const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
                                   const size_t                                   numBitsPerMolecule,
                                   AsyncDeviceVector<double>&                     results,
                                   const size_t                                   offset) {
  int device;
  cudaGetDevice(&device);
  cudaCheckError(cudaGetLastError());

  cudaDeviceProp deviceProp;
  cudaCheckError(cudaGetDeviceProperties(&deviceProp, device));

  size_t m = bitsOne.size();
  size_t n = bitsTwo.size();

  if (supportsTensorOps(deviceProp.major, deviceProp.minor)) {
    constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{64U};

    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};

    constexpr unsigned int NUM_WARP_X{4U};
    constexpr unsigned int NUM_WARP_Y{2U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{32 * NUM_WARP_X * NUM_WARP_Y};

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{(static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) / (BLOCK_TILE_SIZE_X),
                        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) / (BLOCK_TILE_SIZE_Y),
                        1U};

    tanimotoCrossSimilarityKernel<kThreadReductionType,
                                  BLOCK_TILE_SIZE_X,
                                  BLOCK_TILE_SIZE_Y,
                                  BLOCK_TILE_SIZE_K,
                                  NUM_WARP_X,
                                  NUM_WARP_Y>
      <<<grid_dim, block_dim>>>(castAsSpanOfSmallerType<internal::kBlockType, kThreadReductionType>(bitsOne),
                                castAsSpanOfSmallerType<internal::kBlockType, kThreadReductionType>(bitsTwo),
                                numBitsPerMolecule,
                                toSpan(results),
                                offset);
  } else {
    constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};

    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

    constexpr unsigned int THREAD_TILE_X{4U};
    constexpr unsigned int THREAD_TILE_Y{8U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
                                                 (THREAD_TILE_X * THREAD_TILE_Y)};
    static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_Y == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK == 0U);
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS_PER_BLOCK == 0U);

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{(static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) / (BLOCK_TILE_SIZE_X),
                        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) / (BLOCK_TILE_SIZE_Y),
                        1U};

    tanimotoCrossSimilarityKernel<kThreadReductionType,
                                  BLOCK_TILE_SIZE_X,
                                  BLOCK_TILE_SIZE_Y,
                                  BLOCK_TILE_SIZE_K,
                                  THREAD_TILE_X,
                                  THREAD_TILE_Y>
      <<<grid_dim, block_dim>>>(castAsSpanOfSmallerType<internal::kBlockType, kThreadReductionType>(bitsOne),
                                castAsSpanOfSmallerType<internal::kBlockType, kThreadReductionType>(bitsTwo),
                                numBitsPerMolecule,
                                toSpan(results),
                                offset);
  }

  cudaCheckError(cudaGetLastError());
}

//! Launches a kernel to compute the all-to-all Tanimoto similarity between a list of fingerprints.
//! \param bits The list of fingerprints
//! \param results The output similarity, not safe to use until a sync.
void launchCrossTanimotoSimilarity(const cuda::std::span<const std::uint32_t> bitsOne,
                                   const cuda::std::span<const std::uint32_t> bitsTwo,
                                   const size_t                               numBitsPerMolecule,
                                   const cuda::std::span<double>              results,
                                   const size_t                               offset,
                                   cudaStream_t                               stream) {
  int device;
  cudaGetDevice(&device);
  cudaCheckError(cudaGetLastError());

  cudaDeviceProp deviceProp;
  cudaCheckError(cudaGetDeviceProperties(&deviceProp, device));

  size_t m = bitsOne.size() / numBitsPerMolecule;
  size_t n = bitsTwo.size() / numBitsPerMolecule;

  if (supportsTensorOps(deviceProp.major, deviceProp.minor)) {
    constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{64U};

    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};

    constexpr int SMEM_IS_ENOUGH = (2 * (BLOCK_TILE_SIZE_X + BLOCK_TILE_SIZE_Y) * (BLOCK_TILE_SIZE_K + 1)) /
                                   ((BLOCK_TILE_SIZE_Y) * (BLOCK_TILE_SIZE_X + 1));

    static_assert(SMEM_IS_ENOUGH > 0);

    constexpr unsigned int NUM_WARP_X{4U};
    constexpr unsigned int NUM_WARP_Y{2U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{32 * NUM_WARP_X * NUM_WARP_Y};

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{(static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) / (BLOCK_TILE_SIZE_X),
                        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) / (BLOCK_TILE_SIZE_Y),
                        1U};

    tanimotoCrossSimilarityKernel<std::uint32_t,
                                  BLOCK_TILE_SIZE_X,
                                  BLOCK_TILE_SIZE_Y,
                                  BLOCK_TILE_SIZE_K,
                                  NUM_WARP_X,
                                  NUM_WARP_Y>
      <<<grid_dim, block_dim, 0, stream>>>(bitsOne, bitsTwo, numBitsPerMolecule, results, offset);
  } else {
    constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};

    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

    constexpr unsigned int THREAD_TILE_X{4U};
    constexpr unsigned int THREAD_TILE_Y{8U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
                                                 (THREAD_TILE_X * THREAD_TILE_Y)};
    static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_Y == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK == 0U);
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS_PER_BLOCK == 0U);

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{(static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) / (BLOCK_TILE_SIZE_X),
                        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) / (BLOCK_TILE_SIZE_Y),
                        1U};

    tanimotoCrossSimilarityKernel<std::uint32_t,
                                  BLOCK_TILE_SIZE_X,
                                  BLOCK_TILE_SIZE_Y,
                                  BLOCK_TILE_SIZE_K,
                                  THREAD_TILE_X,
                                  THREAD_TILE_Y>
      <<<grid_dim, block_dim, 0, stream>>>(bitsOne, bitsTwo, numBitsPerMolecule, results, offset);
  }

  cudaCheckError(cudaGetLastError());
}

template void launchCrossTanimotoSimilarity<typename std::uint32_t>(
  const AsyncDeviceVector<internal::kBlockType>& bitsOne,
  const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
  const size_t                                   numBitsPerMolecule,
  AsyncDeviceVector<double>&                     results,
  const size_t                                   offset);

namespace {
// --------------------------------
// Cosine similarity kernels
// --------------------------------

template <typename kThreadReductionType,
          size_t BLOCK_TILE_SIZE_X,
          size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K,
          size_t THREAD_TILE_X_OR_NUM_WARP_X,
          size_t THREAD_TILE_Y_OR_NUM_WARP_Y>
__global__ void cosineCrossSimilarityKernel(const cuda::std::span<const kThreadReductionType> bitsOne,
                                            const cuda::std::span<const kThreadReductionType> bitsTwo,
                                            const size_t                                      elementsPerMolecule,
                                            cuda::std::span<double>                           similarities,
                                            const size_t                                      offset) {
#if __CUDA_ARCH__ >= 800
  crossSimilarityKernelTensorOp<SimilarityType::Cosine,
                                kThreadReductionType,
                                double,
                                BLOCK_TILE_SIZE_X,
                                BLOCK_TILE_SIZE_Y,
                                BLOCK_TILE_SIZE_K,
                                THREAD_TILE_X_OR_NUM_WARP_X,
                                THREAD_TILE_Y_OR_NUM_WARP_Y>(bitsOne,
                                                             bitsTwo,
                                                             elementsPerMolecule,
                                                             similarities,
                                                             offset);
#else
  crossSimilarityKernelSIMT<SimilarityType::Cosine,
                            kThreadReductionType,
                            double,
                            BLOCK_TILE_SIZE_X,
                            BLOCK_TILE_SIZE_Y,
                            BLOCK_TILE_SIZE_K,
                            THREAD_TILE_X_OR_NUM_WARP_X,
                            THREAD_TILE_Y_OR_NUM_WARP_Y>(bitsOne, bitsTwo, elementsPerMolecule, similarities, offset);

#endif
}

}  // namespace

// --------------------------------
// Cosine similarity launch functions
// --------------------------------

//! Launches a kernel to compute the all-to-all Tanimoto similarity between a list of fingerprints.
//! \param bits The list of fingerprints
//! \param results The output similarity, not safe to use until a sync.
template <typename kThreadReductionType>
void launchCrossCosineSimilarity(const AsyncDeviceVector<internal::kBlockType>& bitsOne,
                                 const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
                                 const size_t                                   numBitsPerMolecule,
                                 AsyncDeviceVector<double>&                     results,
                                 const size_t                                   offset) {
  int device;
  cudaGetDevice(&device);
  cudaCheckError(cudaGetLastError());

  cudaDeviceProp deviceProp;
  cudaCheckError(cudaGetDeviceProperties(&deviceProp, device));

  size_t m = bitsOne.size();
  size_t n = bitsTwo.size();

  if (supportsTensorOps(deviceProp.major, deviceProp.minor)) {
    constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{64U};

    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};

    constexpr unsigned int NUM_WARP_X{4U};
    constexpr unsigned int NUM_WARP_Y{2U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{32 * NUM_WARP_X * NUM_WARP_Y};

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{(static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) / (BLOCK_TILE_SIZE_X),
                        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) / (BLOCK_TILE_SIZE_Y),
                        1U};

    cosineCrossSimilarityKernel<kThreadReductionType,
                                BLOCK_TILE_SIZE_X,
                                BLOCK_TILE_SIZE_Y,
                                BLOCK_TILE_SIZE_K,
                                NUM_WARP_X,
                                NUM_WARP_Y>
      <<<grid_dim, block_dim>>>(castAsSpanOfSmallerType<internal::kBlockType, kThreadReductionType>(bitsOne),
                                castAsSpanOfSmallerType<internal::kBlockType, kThreadReductionType>(bitsTwo),
                                numBitsPerMolecule,
                                toSpan(results),
                                offset);
  } else {
    constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};

    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

    constexpr unsigned int THREAD_TILE_X{4U};
    constexpr unsigned int THREAD_TILE_Y{8U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
                                                 (THREAD_TILE_X * THREAD_TILE_Y)};
    static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_Y == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK == 0U);
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS_PER_BLOCK == 0U);

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{(static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) / (BLOCK_TILE_SIZE_X),
                        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) / (BLOCK_TILE_SIZE_Y),
                        1U};

    cosineCrossSimilarityKernel<kThreadReductionType,
                                BLOCK_TILE_SIZE_X,
                                BLOCK_TILE_SIZE_Y,
                                BLOCK_TILE_SIZE_K,
                                THREAD_TILE_X,
                                THREAD_TILE_Y>
      <<<grid_dim, block_dim>>>(castAsSpanOfSmallerType<internal::kBlockType, kThreadReductionType>(bitsOne),
                                castAsSpanOfSmallerType<internal::kBlockType, kThreadReductionType>(bitsTwo),
                                numBitsPerMolecule,
                                toSpan(results),
                                offset);
  }

  cudaCheckError(cudaGetLastError());
}

//! Launches a kernel to compute the all-to-all Tanimoto similarity between a list of fingerprints.
//! \param bits The list of fingerprints
//! \param results The output similarity, not safe to use until a sync.
void launchCrossCosineSimilarity(const cuda::std::span<const std::uint32_t> bitsOne,
                                 const cuda::std::span<const std::uint32_t> bitsTwo,
                                 const size_t                               numBitsPerMolecule,
                                 const cuda::std::span<double>              results,
                                 const size_t                               offset,
                                 cudaStream_t                               stream) {
  int device;
  cudaGetDevice(&device);
  cudaCheckError(cudaGetLastError());

  cudaDeviceProp deviceProp;
  cudaCheckError(cudaGetDeviceProperties(&deviceProp, device));

  size_t m = bitsOne.size() / numBitsPerMolecule;
  size_t n = bitsTwo.size() / numBitsPerMolecule;

  if (supportsTensorOps(deviceProp.major, deviceProp.minor)) {
    constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{64U};

    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};

    constexpr unsigned int NUM_WARP_X{4U};
    constexpr unsigned int NUM_WARP_Y{2U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{32 * NUM_WARP_X * NUM_WARP_Y};

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{(static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) / (BLOCK_TILE_SIZE_X),
                        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) / (BLOCK_TILE_SIZE_Y),
                        1U};

    cosineCrossSimilarityKernel<std::uint32_t,
                                BLOCK_TILE_SIZE_X,
                                BLOCK_TILE_SIZE_Y,
                                BLOCK_TILE_SIZE_K,
                                NUM_WARP_X,
                                NUM_WARP_Y>
      <<<grid_dim, block_dim, 0, stream>>>(bitsOne, bitsTwo, numBitsPerMolecule, results, offset);
  } else {
    constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};

    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

    constexpr unsigned int THREAD_TILE_X{4U};
    constexpr unsigned int THREAD_TILE_Y{8U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
                                                 (THREAD_TILE_X * THREAD_TILE_Y)};
    static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_Y == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK == 0U);
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS_PER_BLOCK == 0U);

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{(static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) / (BLOCK_TILE_SIZE_X),
                        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) / (BLOCK_TILE_SIZE_Y),
                        1U};

    cosineCrossSimilarityKernel<std::uint32_t,
                                BLOCK_TILE_SIZE_X,
                                BLOCK_TILE_SIZE_Y,
                                BLOCK_TILE_SIZE_K,
                                THREAD_TILE_X,
                                THREAD_TILE_Y>
      <<<grid_dim, block_dim, 0, stream>>>(bitsOne, bitsTwo, numBitsPerMolecule, results, offset);
  }

  cudaCheckError(cudaGetLastError());
}

template void launchCrossCosineSimilarity<typename std::uint32_t>(
  const AsyncDeviceVector<internal::kBlockType>& bitsOne,
  const AsyncDeviceVector<internal::kBlockType>& bitsTwo,
  const size_t                                   numBitsPerMolecule,
  AsyncDeviceVector<double>&                     results,
  const size_t                                   offset);

}  // namespace nvMolKit
