// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tensor-core assignment: rows x centers^T as an fp16 GEMM (mma.sync m16n8k16, fp32 accumulation) whose A operand is
 * expanded from the packed bits in registers, so neither the unpacked rows nor the rows x centers product is stored.
 *
 * A block owns kMmaRows rows, keeps their packed words in shared memory, and streams the fp16 centers through a
 * cp.async pipeline of 2 to 4 (kMmaBlockCenters centers) x (kMmaStageBits bits) stages, as many as the device's shared
 * memory allows. Each warp computes a 64 x 64 tile; after the last stage of a center tile the epilogue folds the metric
 * and a running argmax into registers. Needs sm_80 (cp.async, mma.sync m16n8k16).
 */

#include <cuda_pipeline.h>

#include <cub/thread/thread_operators.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <stdexcept>

#include "src/kmeans_kernels.cuh"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit::detail {
namespace {

constexpr int kMmaRows       = 128;
constexpr int kMmaWarpsM     = 2;
constexpr int kMmaWarpsN     = 4;
constexpr int kMmaThreads    = kMmaWarpsM * kMmaWarpsN * 32;
constexpr int kStageRowBytes = kMmaStageBits * 2;  // 128
constexpr int kStageBytes    = kMmaBlockCenters * kStageRowBytes;
constexpr int kMmaMaxStages  = 4;
static_assert(kStageRowBytes == 128, "the swizzle assumes 128-byte center rows per stage");

// Device code needs sm_80 (cp.async through the pipeline primitives, ldmatrix, mma.sync m16n8k16); older targets
// compile the kernel to a stub.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

constexpr int kWarpRows    = kMmaRows / kMmaWarpsM;          // 64
constexpr int kWarpCenters = kMmaBlockCenters / kMmaWarpsN;  // 64 for 256 centers
constexpr int kMTiles      = kWarpRows / 16;                 // 4
constexpr int kNTiles      = kWarpCenters / 8;               // 8

//! Argmax over the four lanes of a quad: higher score, then lower center index.
using QuadArgMax = cub::WarpReduce<cub::KeyValuePair<int, float>, 4>;

__device__ __forceinline__ void ldmatrixX4(unsigned (&out)[4], const void* shared) {
  const auto address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
               : "r"(address));
}

__device__ __forceinline__ void mma16816(float (&acc)[4],
                                         const unsigned (&a)[4],
                                         const unsigned b0,
                                         const unsigned b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
    "{%0, %1, %2, %3};\n"
    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

//! Bits 0 and 1 of x as an fp16x2 register of zeros and ones.
__device__ __forceinline__ unsigned bitsToHalf2(const unsigned x) {
  return ((x & 1U) | ((x << 15) & 0x10000U)) * 0x3C00U;
}

//! Running best of one row. Tanimoto keeps the fraction dot / denominator to avoid a division per candidate.
template <KMeansMetric Metric> struct RunningBest {
  float value       = -INFINITY;
  float denominator = 1.0F;
  int   index       = 0x7FFFFFFF;

  __device__ __forceinline__ void offer(const RowScorer<Metric>& scorer,
                                        const float              dot,
                                        const float              norm,
                                        const int                col) {
    if constexpr (Metric == KMeansMetric::Tanimoto) {
      float numerator   = dot;
      float denominator = scorer.count + norm - dot;
      if (!(denominator > 0.0F)) {
        numerator   = 1.0F;
        denominator = 1.0F;
      }
      if (numerator * this->denominator > value * denominator) {
        value             = numerator;
        this->denominator = denominator;
        index             = col;
      }
    } else {
      const float score = scorer.score(dot, norm);
      if (score > value) {
        value = score;
        index = col;
      }
    }
  }

  __device__ __forceinline__ float score() const {
    if constexpr (Metric == KMeansMetric::Tanimoto) {
      return index == 0x7FFFFFFF ? -INFINITY : __fdiv_rn(value, denominator);
    } else {
      return value;
    }
  }
};

#endif

template <KMeansMetric Metric, int Stages>
__global__ void __launch_bounds__(kMmaThreads, 1) mmaAssignKernel(const AssignArgs args, const int paddedBits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ __align__(128) unsigned char shared[];
  unsigned char*                                 stages = shared;
  auto*          rowWords    = reinterpret_cast<std::uint32_t*>(shared + Stages * kStageBytes);
  const int      paddedWords = paddedBits / 32;
  const int      rowStride   = paddedWords + 1;
  // The per-warp results of the final reduction reuse the stage buffers, which are idle by then.
  auto*          bestScores  = reinterpret_cast<float*>(stages);
  auto*          bestIndices = reinterpret_cast<int*>(bestScores + kMmaWarpsN * kMmaRows);
  __shared__ int changedCount;
  __shared__ typename QuadArgMax::TempStorage quadStorage[kMmaThreads / 4];

  const int warp     = static_cast<int>(threadIdx.x / 32);
  const int lane     = static_cast<int>(threadIdx.x % 32);
  const int warpM    = warp / kMmaWarpsN;
  const int warpN    = warp % kMmaWarpsN;
  const int group    = lane / 4;
  const int quad     = lane % 4;
  const int blockRow = static_cast<int>(blockIdx.x) * kMmaRows;

  for (int index = static_cast<int>(threadIdx.x); index < kMmaRows * paddedWords; index += kMmaThreads) {
    const int row                    = index / paddedWords;
    const int word                   = index % paddedWords;
    const int g                      = blockRow + row;
    rowWords[row * rowStride + word] = g < args.numRows && word < args.numWords ?
                                         args.fingerprints[static_cast<std::size_t>(g) * args.numWords + word] :
                                         0U;
  }
  if (threadIdx.x == 0) {
    changedCount = 0;
  }

  const int stagesPerTile = paddedBits / kMmaStageBits;
  const int numTiles      = (args.numClusters + kMmaBlockCenters - 1) / kMmaBlockCenters;
  const int numStages     = numTiles * stagesPerTile;

  auto loadStage = [&](const int stage) {
    if (stage < numStages) {
      const int      tile   = stage / stagesPerTile;
      const int      kStage = stage % stagesPerTile;
      unsigned char* target = stages + (stage % Stages) * kStageBytes;
      const __half*  source =
        args.centers + static_cast<std::size_t>(tile) * kMmaBlockCenters * paddedBits + kStage * kMmaStageBits;
      for (int chunk = static_cast<int>(threadIdx.x); chunk < kMmaBlockCenters * 8; chunk += kMmaThreads) {
        const int center = chunk / 8;
        const int part   = chunk % 8;
        __pipeline_memcpy_async(target + center * kStageRowBytes + ((part ^ (center & 7)) * 16),
                                source + static_cast<std::size_t>(center) * paddedBits + part * 8,
                                16);
      }
    }
    __pipeline_commit();
  };

  for (int stage = 0; stage < Stages - 1; ++stage) {
    loadStage(stage);
  }
  __syncthreads();

  RowScorer<Metric> scorers[kMTiles][2] = {
    {RowScorer<Metric>(0), RowScorer<Metric>(0)},
    {RowScorer<Metric>(0), RowScorer<Metric>(0)},
    {RowScorer<Metric>(0), RowScorer<Metric>(0)},
    {RowScorer<Metric>(0), RowScorer<Metric>(0)}
  };
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row     = blockRow + warpM * kWarpRows + mt * 16 + half * 8 + group;
      scorers[mt][half] = RowScorer<Metric>(row < args.numRows ? args.rowCounts[row] : 0);
    }
  }
  RunningBest<Metric> best[kMTiles][2];
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      best[mt][half] = RunningBest<Metric>{};
    }
  }
  float acc[kMTiles][kNTiles][4];
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
    for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        acc[mt][nt][e] = 0.0F;
      }
    }
  }

  for (int stage = 0; stage < numStages; ++stage) {
    __pipeline_wait_prior(Stages - 2);
    __syncthreads();
    loadStage(stage + Stages - 1);

    const unsigned char* stageData = stages + (stage % Stages) * kStageBytes;
    const int            kStage    = stage % stagesPerTile;
    std::uint32_t        words[kMTiles][2][2];
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = warpM * kWarpRows + mt * 16 + half * 8 + group;
#pragma unroll
        for (int w = 0; w < 2; ++w) {
          words[mt][half][w] = rowWords[row * rowStride + kStage * 2 + w] >> (2 * quad);
        }
      }
    }
#pragma unroll
    for (int kk = 0; kk < kMmaStageBits / 16; ++kk) {
      unsigned b[kNTiles][2];
#pragma unroll
      for (int j = 0; j < kNTiles / 2; ++j) {
        const int center = warpN * kWarpCenters + j * 16 + (lane / 16) * 8 + (lane % 8);
        const int part   = kk * 2 + ((lane / 8) & 1);
        unsigned  fragments[4];
        ldmatrixX4(fragments, stageData + center * kStageRowBytes + ((part ^ (center & 7)) * 16));
        b[2 * j][0]     = fragments[0];
        b[2 * j][1]     = fragments[1];
        b[2 * j + 1][0] = fragments[2];
        b[2 * j + 1][1] = fragments[3];
      }
#pragma unroll
      for (int mt = 0; mt < kMTiles; ++mt) {
        const int      shift = (kk % 2) * 16;
        const unsigned low   = words[mt][0][kk / 2] >> shift;
        const unsigned high  = words[mt][1][kk / 2] >> shift;
        const unsigned a[4]  = {bitsToHalf2(low), bitsToHalf2(high), bitsToHalf2(low >> 8), bitsToHalf2(high >> 8)};
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
          mma16816(acc[mt][nt], a, b[nt][0], b[nt][1]);
        }
      }
    }

    if (kStage == stagesPerTile - 1) {
      const int tile = stage / stagesPerTile;
#pragma unroll
      for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int   col  = tile * kMmaBlockCenters + warpN * kWarpCenters + nt * 8 + 2 * quad + e;
          const bool  live = col < args.numClusters;
          const float norm = live ? args.norms[col] : 0.0F;
#pragma unroll
          for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
              if (live) {
                best[mt][half].offer(scorers[mt][half], acc[mt][nt][half * 2 + e], norm, col);
              }
            }
          }
        }
      }
#pragma unroll
      for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            acc[mt][nt][e] = 0.0F;
          }
        }
      }
    }
  }
  __pipeline_wait_prior(0);
  __syncthreads();

  // Reduce across the four lanes of a quad, then across the warps that share rows.
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const cub::KeyValuePair<int, float> pair(best[mt][half].index, best[mt][half].score());
      const auto                          winner = QuadArgMax(quadStorage[threadIdx.x / 4]).Reduce(pair, cub::ArgMax());
      if (quad == 0) {
        const int row                       = warpM * kWarpRows + mt * 16 + half * 8 + group;
        bestScores[warpN * kMmaRows + row]  = winner.value;
        bestIndices[warpN * kMmaRows + row] = winner.key;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < kMmaRows) {
    const int row       = static_cast<int>(threadIdx.x);
    const int globalRow = blockRow + row;
    if (globalRow < args.numRows) {
      float score = bestScores[row];
      int   index = bestIndices[row];
      for (int n = 1; n < kMmaWarpsN; ++n) {
        const float otherScore = bestScores[n * kMmaRows + row];
        const int   otherIndex = bestIndices[n * kMmaRows + row];
        if (otherScore > score || (otherScore == score && otherIndex < index)) {
          score = otherScore;
          index = otherIndex;
        }
      }
      const RowScorer<Metric> scorer(args.rowCounts[globalRow]);
      if (storeAssignment(args, globalRow, scorer, score, index)) {
        atomicAdd(&changedCount, 1);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && changedCount > 0) {
    atomicAdd(args.changed, changedCount);
  }
#else
  (void)args;
  (void)paddedBits;
  __trap();
#endif
}

std::size_t mmaSharedBytes(const int paddedBits, const int stages) {
  static_assert((sizeof(float) + sizeof(int)) * kMmaWarpsN * kMmaRows <= 2 * kStageBytes,
                "the final reduction must fit in the stage buffers");
  return static_cast<std::size_t>(stages) * kStageBytes + sizeof(std::uint32_t) * kMmaRows * (paddedBits / 32 + 1);
}

//! Most pipeline stages whose shared memory fits the current device, or 0 when not even two fit or it is pre-sm_80.
int mmaStages(const int paddedBits) {
  int device = 0;
  int major  = 0;
  int limit  = 0;
  cudaCheckError(cudaGetDevice(&device));
  cudaCheckError(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  cudaCheckError(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  if (major < 8) {
    return 0;
  }
  // Leave room for the kernel's static shared memory.
  const std::size_t available = static_cast<std::size_t>(limit) - 1024;
  for (int stages = kMmaMaxStages; stages >= 2; --stages) {
    if (mmaSharedBytes(paddedBits, stages) <= available) {
      return stages;
    }
  }
  return 0;
}

}  // namespace

bool mmaSupported(const int paddedBits) {
  return mmaStages(paddedBits) > 0;
}

template <KMeansMetric Metric> void launchMmaAssign(const AssignArgs& args, const int paddedBits, cudaStream_t stream) {
  if (args.numRows == 0) {
    return;
  }
  const int stages = mmaStages(paddedBits);
  if (stages == 0) {
    throw std::invalid_argument(
      "k-means needs a GPU of compute capability 8.0 or newer with enough shared memory for the fingerprint width");
  }
  const std::size_t sharedBytes = mmaSharedBytes(paddedBits, stages);
  auto              kernel      = stages == 4 ? mmaAssignKernel<Metric, 4> :
                                  stages == 3 ? mmaAssignKernel<Metric, 3> :
                                                mmaAssignKernel<Metric, 2>;
  cudaCheckError(
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes)));
  const int blocks = (args.numRows + kMmaRows - 1) / kMmaRows;
  kernel<<<blocks, kMmaThreads, sharedBytes, stream>>>(args, paddedBits);
  cudaCheckError(cudaGetLastError());
}

template void launchMmaAssign<KMeansMetric::Cosine>(const AssignArgs&, int, cudaStream_t);
template void launchMmaAssign<KMeansMetric::Tanimoto>(const AssignArgs&, int, cudaStream_t);

}  // namespace nvMolKit::detail
