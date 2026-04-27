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

#include "src/data_structures/flat_bit_vect.h"
#include "src/substruct/graph_labeler.cuh"
#include "src/substruct/molecules_device.cuh"
#include "src/substruct/sm_shared_mem_config.cuh"
#include "src/substruct/substruct_algos.cuh"
#include "src/substruct/substruct_debug.h"
#include "src/substruct/substruct_kernels.h"
#include "src/substruct/substruct_launch_config.h"
#include "src/substruct/substruct_search_internal.h"
#include "src/substruct/substruct_types.h"
namespace nvMolKit {

namespace {

// =============================================================================
// Internal View Struct (passed to kernels by value)
// =============================================================================

template <std::size_t MaxQueryAtoms = kMaxQueryAtoms> struct SubstructMatchResultsDeviceViewT {
  int*                          matchCounts;
  int*                          reportedCounts;
  const int*                    pairMatchStarts;
  int16_t*                      matchIndices;
  int                           numQueries;
  const int*                    queryAtomCounts;
  PartialMatchT<MaxQueryAtoms>* overflowBuffer;
  int                           overflowEntriesPerBuffer;
  int                           overflowBuffersPerBlock;
  uint32_t*                     recursiveMatchBits;
  int                           maxTargetAtoms;
  uint32_t*                     labelMatrixBuffer;
  std::size_t                   labelMatrixWords;
  int                           maxMatchesToFind;
  bool                          countOnly;
  uint8_t*                      overflowFlags;  ///< Per-pair overflow detection (set if buffers exhausted)

  __device__ __forceinline__ uint32_t* getLabelMatrixPtr(int miniBatchIdx) const {
    return labelMatrixBuffer + miniBatchIdx * labelMatrixWords;
  }

  __device__ __forceinline__ PartialMatchT<MaxQueryAtoms>* getOverflowBuffer(int bufferIdx = 0) const {
    return overflowBuffer + (blockIdx.x * overflowBuffersPerBlock + bufferIdx) * overflowEntriesPerBuffer;
  }

  __device__ __forceinline__ int getOverflowCapacity() const { return overflowEntriesPerBuffer; }

  __device__ __forceinline__ uint32_t getRecursiveMatchBits(int miniBatchIdx, int atomIdx) const {
    return recursiveMatchBits[miniBatchIdx * maxTargetAtoms + atomIdx];
  }

  __device__ __forceinline__ void setRecursiveMatchBit(int miniBatchIdx, int atomIdx, int patternId) const {
    if (patternId < 32) {
      atomicOr(&recursiveMatchBits[miniBatchIdx * maxTargetAtoms + atomIdx], 1u << patternId);
    }
  }
};

// =============================================================================
// Architecture-Specific Thread Configuration
// =============================================================================

/// Returns SM version for compile-time configuration (device code uses __CUDA_ARCH__, host defaults to 86)
__host__ __device__ constexpr int getComputeCapability() {
#ifdef __CUDA_ARCH__
  return __CUDA_ARCH__ / 10;
#else
  return 86;
#endif
}

/// Max threads per SM for each compute capability
__host__ __device__ constexpr int getMaxThreadsPerSM(int sm) {
  if (sm == 120)
    return 1536;  // Consumer Blackwell (sm_120)
  if (sm == 100)
    return 2048;  // Datacenter Blackwell (sm_100)
  if (sm >= 90)
    return 2048;  // Hopper
  if (sm == 80)
    return 2048;  // A100
  if (sm >= 86)
    return 1536;  // Ada/consumer Ampere
  return 1536;    // Default
}

/// Compute max blocks per SM given block size
__host__ __device__ constexpr int getMaxBlocksPerSM(int sm, int blockSize) {
  return getMaxThreadsPerSM(sm) / blockSize;
}

/// Compute max partials that fit in shared memory budget
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__host__ __device__ constexpr int computeMaxPartials(int sharedPerSM_KB, int blocksPerSM) {
  constexpr int kLabelMatrixBytes = (MaxTargetAtoms * MaxQueryAtoms) / 8;
  constexpr int kControlVarsBytes = 32;
  constexpr int kPartialMatchSize = sizeof(PartialMatchT<MaxQueryAtoms>);
  constexpr int kTargetBondsBytes = MaxTargetAtoms * sizeof(TargetAtomBonds);
  constexpr int kQueryBondsBytes  = MaxQueryAtoms * sizeof(QueryAtomBonds);

  const int budgetBytes    = (sharedPerSM_KB * 1024) / blocksPerSM;
  const int fixedOverhead  = kLabelMatrixBytes + kControlVarsBytes + kTargetBondsBytes + kQueryBondsBytes;
  const int availableBytes = (budgetBytes * 9 / 10) - fixedOverhead;
  if (availableBytes < kPartialMatchSize * 2) {
    return 0;
  }
  const int rawPartials = availableBytes / (kPartialMatchSize * 2);  // ping-pong
  const int rounded     = (rawPartials / 10) * 10;
  return rounded > 0 ? rounded : rawPartials;
}

/// Compute partials for a given SM architecture
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__host__ __device__ constexpr int getMaxPartialsForSM(int sm, int blockSize) {
  return computeMaxPartials<MaxTargetAtoms, MaxQueryAtoms>(getMaxSharedMemoryPerSM_KB(sm),
                                                           getMaxBlocksPerSM(sm, blockSize));
}

// Compute at compile time based on __CUDA_ARCH__
constexpr int kDefaultBlockSize = getBlockSizeForConfig<kMaxTargetAtoms>();

static_assert(getMaxThreadsPerSM(86) % kDefaultBlockSize == 0, "block size must evenly divide max threads/SM");
constexpr int kWarpsPerBlock = kDefaultBlockSize / 32;

// =============================================================================
// Shared Memory Carveout Configuration
// =============================================================================

template <typename KernelFunc> void configureSharedMemCarveout(KernelFunc kernel) {
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
}

inline bool& sharedMemCarveoutConfigured() {
  static bool configured = false;
  return configured;
}

// =============================================================================
// Device Helper Functions
// =============================================================================

template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ void writeLabelMatrixToGlobal(const TargetMoleculeView&                    target,
                                         const QueryMoleculeView&                     query,
                                         FlatBitVect<MaxTargetAtoms * MaxQueryAtoms>& sharedLabelMatrix,
                                         const uint32_t*                              pairRecursiveBits,
                                         uint32_t*                                    globalOut) {
  constexpr std::size_t kLabelMatrixWordsT = (MaxTargetAtoms * MaxQueryAtoms) / 32;
  using LabelMatrixViewT                   = BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>;
  LabelMatrixViewT labelMatrix(&sharedLabelMatrix);

  populateLabelMatrix<MaxTargetAtoms, MaxQueryAtoms>(target, query, labelMatrix, pairRecursiveBits);
  __syncthreads();

  const uint32_t* sharedIn   = sharedLabelMatrix.cbegin();
  const int       tid        = threadIdx.x;
  const int       numThreads = blockDim.x;
  for (std::size_t i = tid; i < kLabelMatrixWordsT; i += numThreads) {
    globalOut[i] = sharedIn[i];
  }
}

template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ void loadLabelMatrixToShared(FlatBitVect<MaxTargetAtoms * MaxQueryAtoms>& sharedLabelMatrix,
                                        const uint32_t*                              globalIn) {
  constexpr std::size_t kLabelMatrixWordsT = (MaxTargetAtoms * MaxQueryAtoms) / 32;
  uint32_t*             sharedOut          = sharedLabelMatrix.begin();
  const int             tid                = threadIdx.x;
  const int             numThreads         = blockDim.x;
  for (std::size_t i = tid; i < kLabelMatrixWordsT; i += numThreads) {
    sharedOut[i] = globalIn[i];
  }
}

// =============================================================================
// Kernel Definitions
// =============================================================================

/**
 * @brief Templated kernel for batch label matrix computation.
 *
 * One block per (target, query) pair. Computes label matrix and writes to global buffer.
 *
 * @tparam MaxTargetAtoms Maximum target atoms for label matrix sizing
 * @tparam MaxQueryAtoms Maximum query atoms for label matrix sizing
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__global__ void labelMatrixKernelT(TargetMoleculesDeviceView targets,
                                   QueryMoleculesDeviceView  queries,
                                   const int*                pairIndices,
                                   int                       numQueries,
                                   uint32_t*                 labelMatrixBuffer,
                                   const uint32_t*           recursiveMatchBits,
                                   int                       maxTargetAtoms,
                                   const int*                batchLocalIndices) {
  constexpr std::size_t kLabelMatrixBitsT = MaxTargetAtoms * MaxQueryAtoms;
  using LabelMatrixStorageT               = FlatBitVect<kLabelMatrixBitsT>;

  const int launchIdx     = blockIdx.x;
  const int miniBatchIdx  = batchLocalIndices ? batchLocalIndices[launchIdx] : launchIdx;
  const int globalPairIdx = pairIndices[launchIdx];
  const int targetIdx     = globalPairIdx / numQueries;
  const int queryIdx      = globalPairIdx % numQueries;

  if (targetIdx >= targets.numMolecules || queryIdx >= queries.numMolecules) {
    return;
  }

  const TargetMoleculeView target = getMolecule(targets, targetIdx);
  const QueryMoleculeView  query  = getMolecule(queries, queryIdx);

  const uint32_t* pairRecursiveBits = recursiveMatchBits ? &recursiveMatchBits[miniBatchIdx * maxTargetAtoms] : nullptr;

  __shared__ LabelMatrixStorageT sharedLabelMatrix;
  uint32_t*                      globalOut = labelMatrixBuffer + miniBatchIdx * (kLabelMatrixBitsT / 32);
  writeLabelMatrixToGlobal<MaxTargetAtoms, MaxQueryAtoms>(target,
                                                          query,
                                                          sharedLabelMatrix,
                                                          pairRecursiveBits,
                                                          globalOut);
}

/**
 * @brief Templated kernel for label matrix computation for recursive pattern preprocessing.
 *
 * Block indexing: blockIdx.x = localTargetIdx * numPatterns + localPatternIdx
 *
 * @tparam MaxTargetAtoms Maximum target atoms for label matrix sizing
 * @tparam MaxQueryAtoms Maximum query atoms for label matrix sizing
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__global__ void labelMatrixPaintKernelT(TargetMoleculesDeviceView  targets,
                                        QueryMoleculesDeviceView   patterns,
                                        const BatchedPatternEntry* patternEntries,
                                        int                        numPatterns,
                                        int                        numQueries,
                                        int                        miniBatchPairOffset,
                                        int                        miniBatchSize,
                                        uint32_t*                  labelMatrixBuffer,
                                        int                        firstTargetIdx,
                                        const uint32_t*            recursiveMatchBits,
                                        int                        maxTargetAtoms,
                                        ZeroBuffersSpec            zeroBuffers) {
  // Zero buffers using grid-stride loop (only on first label kernel launch)
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  if (zeroBuffers.recursiveBitsSize > 0 && zeroBuffers.recursiveBits != nullptr) {
    for (int i = tid; i < zeroBuffers.recursiveBitsSize; i += stride) {
      zeroBuffers.recursiveBits[i] = 0;
    }
  }

  if (zeroBuffers.overflowFlagsSize > 0 && zeroBuffers.overflowFlags != nullptr) {
    for (int i = tid; i < zeroBuffers.overflowFlagsSize; i += stride) {
      zeroBuffers.overflowFlags[i] = 0;
    }
  }

  constexpr std::size_t kLabelMatrixBitsT = MaxTargetAtoms * MaxQueryAtoms;
  using LabelMatrixStorageT               = FlatBitVect<kLabelMatrixBitsT>;

  const int localTargetIdx  = blockIdx.x / numPatterns;
  const int targetIdx       = firstTargetIdx + localTargetIdx;
  const int localPatternIdx = blockIdx.x % numPatterns;

  if (targetIdx >= targets.numMolecules || localPatternIdx >= numPatterns) {
    return;
  }

  const int mainQueryIdx  = patternEntries[localPatternIdx].mainQueryIdx;
  const int patternMolIdx = patternEntries[localPatternIdx].patternMolIdx;

  const int globalPairIdx = targetIdx * numQueries + mainQueryIdx;

  if (globalPairIdx < miniBatchPairOffset || globalPairIdx >= miniBatchPairOffset + miniBatchSize) {
    return;
  }

  const int batchLocalPairIdx = globalPairIdx - miniBatchPairOffset;

  const TargetMoleculeView target  = getMolecule(targets, targetIdx);
  const QueryMoleculeView  pattern = getMolecule(patterns, patternMolIdx);

  const uint32_t* pairBits =
    (recursiveMatchBits != nullptr) ? recursiveMatchBits + batchLocalPairIdx * maxTargetAtoms : nullptr;

  __shared__ LabelMatrixStorageT sharedLabelMatrix;
  uint32_t*                      globalOut = labelMatrixBuffer + blockIdx.x * (kLabelMatrixBitsT / 32);
  writeLabelMatrixToGlobal<MaxTargetAtoms, MaxQueryAtoms>(target, pattern, sharedLabelMatrix, pairBits, globalOut);
}

/**
 * @brief Templated kernel for batch substructure matching.
 *
 * One block per (target, query) pair. Loads pre-computed label matrix from global
 * memory, then dispatches to algorithm-specific search based on template parameters.
 *
 * @tparam MaxTargetAtoms Maximum target atoms for label matrix sizing
 * @tparam MaxQueryAtoms Maximum query atoms for label matrix and partial match sizing
 * @tparam MaxBondsPerAtom Maximum bonds per atom for edge consistency loop unrolling
 * @tparam Algo Algorithm to use (VF2 or GSI)
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms, int MaxBondsPerAtom, SubstructAlgorithm Algo>
__global__ void substructMatchKernelT(TargetMoleculesDeviceView                       targets,
                                      QueryMoleculesDeviceView                        queries,
                                      SubstructMatchResultsDeviceViewT<MaxQueryAtoms> results,
                                      const int*                                      pairIndices,
                                      int                                             numQueries,
                                      const int*                                      batchLocalIndices,
                                      DeviceTimingsData*                              timings) {
  const int launchIdx     = blockIdx.x;
  const int miniBatchIdx  = batchLocalIndices ? batchLocalIndices[launchIdx] : launchIdx;
  const int globalPairIdx = pairIndices[launchIdx];
  const int targetIdx     = globalPairIdx / numQueries;
  const int queryIdx      = globalPairIdx % numQueries;

  if (targetIdx >= targets.numMolecules || queryIdx >= queries.numMolecules) {
    return;
  }

  TargetMoleculeView target = getMolecule(targets, targetIdx);
  QueryMoleculeView  query  = getMolecule(queries, queryIdx);

  constexpr std::size_t kLabelMatrixBitsT = MaxTargetAtoms * MaxQueryAtoms;
  using LabelMatrixStorageT               = FlatBitVect<kLabelMatrixBitsT>;
  using LabelMatrixViewT                  = BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>;

  __shared__ LabelMatrixStorageT sharedLabelMatrix;
  LabelMatrixViewT               labelMatrix(&sharedLabelMatrix);

  __shared__ TargetAtomBonds sharedTargetBonds[MaxTargetAtoms];
  __shared__ QueryAtomBonds  sharedQueryBonds[MaxQueryAtoms];

  const uint32_t* globalIn   = results.getLabelMatrixPtr(miniBatchIdx);
  const int       tid        = threadIdx.x;
  const int       numThreads = blockDim.x;

  loadLabelMatrixToShared<MaxTargetAtoms, MaxQueryAtoms>(sharedLabelMatrix, globalIn);

  const int numTargetAtoms = target.numAtoms;
  const int numQueryAtoms  = query.numAtoms;

  const int       totalTargetBytes = numTargetAtoms * sizeof(TargetAtomBonds);
  const int       totalTargetWords = (totalTargetBytes + 3) / 4;
  const uint32_t* targetBondsSrc   = reinterpret_cast<const uint32_t*>(target.targetAtomBonds);
  uint32_t*       targetBondsDst   = reinterpret_cast<uint32_t*>(sharedTargetBonds);
  for (int i = tid; i < totalTargetWords; i += numThreads) {
    targetBondsDst[i] = targetBondsSrc[i];
  }

  constexpr int kQueryBondWords = sizeof(QueryAtomBonds) / sizeof(uint32_t);
  static_assert(sizeof(QueryAtomBonds) % sizeof(uint32_t) == 0, "QueryAtomBonds must be word-aligned");
  const uint32_t* queryBondsSrc   = reinterpret_cast<const uint32_t*>(query.queryAtomBonds);
  uint32_t*       queryBondsDst   = reinterpret_cast<uint32_t*>(sharedQueryBonds);
  const int       totalQueryWords = numQueryAtoms * kQueryBondWords;
  for (int i = tid; i < totalQueryWords; i += numThreads) {
    queryBondsDst[i] = queryBondsSrc[i];
  }
  __syncthreads();

  target.targetAtomBonds = sharedTargetBonds;
  query.queryAtomBonds   = sharedQueryBonds;

  if constexpr (kDebugDumpLabelMatrix) {
    if (threadIdx.x == 0) {
      printf("[LabelDump] pair=%d (target=%d, query=%d): targetAtoms=%d, queryAtoms=%d\n",
             globalPairIdx,
             targetIdx,
             queryIdx,
             target.numAtoms,
             query.numAtoms);
      printf("[LabelDump] Label matrix (row=target, col=query, 1=compatible):\n");
      printf("[LabelDump]     ");
      for (int q = 0; q < query.numAtoms; ++q) {
        printf("q%d ", q);
      }
      printf("\n");
      for (int t = 0; t < target.numAtoms; ++t) {
        printf("[LabelDump] t%2d: ", t);
        for (int q = 0; q < query.numAtoms; ++q) {
          printf("%d  ", labelMatrix.get(t, q) ? 1 : 0);
        }
        printf("\n");
      }
    }
    __syncthreads();
  }

  const int  maxMatchesToFind = results.maxMatchesToFind;
  const bool countOnly        = results.countOnly;

  const int matchOffset = countOnly ? 0 : results.pairMatchStarts[miniBatchIdx];
  const int maxMatches  = countOnly ? 0 : (results.pairMatchStarts[miniBatchIdx + 1] - matchOffset) / query.numAtoms;

  __shared__ int sharedMatchCount;
  __shared__ int sharedReportedCount;

  if (threadIdx.x == 0) {
    sharedMatchCount    = 0;
    sharedReportedCount = 0;
  }
  __syncthreads();

  if constexpr (Algo == SubstructAlgorithm::VF2) {
    namespace cg       = cooperative_groups;
    auto      tile32   = cg::tiled_partition<32>(cg::this_thread_block());
    const int warpId   = tile32.meta_group_rank();
    const int numWarps = tile32.meta_group_size();

    __shared__ VF2StateT<MaxQueryAtoms> vf2States[kWarpsPerBlock];

    if (tile32.thread_rank() == 0) {
      vf2States[warpId].init();
    }
    __syncthreads();

    for (int startT = warpId; startT < target.numAtoms; startT += numWarps) {
      vf2SearchGPU<MaxTargetAtoms, MaxQueryAtoms, MaxBondsPerAtom>(target,
                                                                   query,
                                                                   labelMatrix,
                                                                   vf2States[warpId],
                                                                   startT,
                                                                   &sharedMatchCount,
                                                                   &sharedReportedCount,
                                                                   results.matchIndices,
                                                                   maxMatches,
                                                                   matchOffset,
                                                                   maxMatchesToFind,
                                                                   countOnly);
    }

  } else if constexpr (Algo == SubstructAlgorithm::GSI) {
    constexpr int kBlockSizeT = getBlockSizeForConfig<MaxTargetAtoms>();
    constexpr int kMaxPartialsT =
      getMaxPartialsForSM<MaxTargetAtoms, MaxQueryAtoms>(getComputeCapability(), kBlockSizeT);
    static_assert(kMaxPartialsT > 0,
                  "Insufficient shared memory for GSI partials - check block size for MaxTargetAtoms/MaxQueryAtoms");
    __shared__ PartialMatchT<MaxQueryAtoms> gsiPartials[kMaxPartialsT * 2];

    uint8_t* pairOverflowFlag = results.overflowFlags ? &results.overflowFlags[miniBatchIdx] : nullptr;
    if (pairOverflowFlag != nullptr && threadIdx.x == 0) {
      *pairOverflowFlag = 0;
    }
    gsiBFSSearchGPU<MaxTargetAtoms, MaxQueryAtoms, MaxBondsPerAtom>(target,
                                                                    query,
                                                                    labelMatrix,
                                                                    gsiPartials,
                                                                    kMaxPartialsT,
                                                                    results.getOverflowBuffer(0),
                                                                    results.getOverflowBuffer(1),
                                                                    results.getOverflowCapacity(),
                                                                    &sharedMatchCount,
                                                                    &sharedReportedCount,
                                                                    results.matchIndices,
                                                                    maxMatches,
                                                                    matchOffset,
                                                                    {},
                                                                    maxMatchesToFind,
                                                                    countOnly,
                                                                    timings,
                                                                    pairOverflowFlag);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    results.matchCounts[miniBatchIdx] = sharedMatchCount;
    if (!countOnly) {
      results.reportedCounts[miniBatchIdx] = sharedReportedCount;
    }
  }
}

/**
 * @brief Paint mode kernel for recursive SMARTS preprocessing.
 *
 * Instead of storing match mappings, directly paints recursive match bits
 * into the output buffer.
 *
 * @tparam MaxTargetAtoms Maximum target atoms for label matrix sizing
 * @tparam MaxQueryAtoms Maximum query atoms for label matrix and partial match sizing
 * @tparam MaxBondsPerAtom Maximum bonds per atom for edge consistency loop unrolling
 * @tparam Algo Algorithm to use (VF2 or GSI)
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms, int MaxBondsPerAtom, SubstructAlgorithm Algo>
__global__ void substructPaintKernelT(TargetMoleculesDeviceView     targets,
                                      QueryMoleculesDeviceView      patterns,
                                      const BatchedPatternEntry*    patternEntries,
                                      int                           numPatterns,
                                      uint32_t*                     outputRecursiveBits,
                                      int                           maxTargetAtoms,
                                      int                           outputNumQueries,
                                      int                           defaultPatternId,
                                      int                           defaultMainQueryIdx,
                                      int                           miniBatchPairOffset,
                                      int                           miniBatchSize,
                                      PartialMatchT<MaxQueryAtoms>* overflowA,
                                      PartialMatchT<MaxQueryAtoms>* overflowB,
                                      int                           overflowCapacity,
                                      const uint32_t*               labelMatrixBuffer,
                                      int                           firstTargetIdx) {
  const int localTargetIdx  = blockIdx.x / numPatterns;
  const int targetIdx       = firstTargetIdx + localTargetIdx;
  const int localPatternIdx = blockIdx.x % numPatterns;

  if (targetIdx >= targets.numMolecules || localPatternIdx >= numPatterns) {
    return;
  }

  const int mainQueryIdx  = patternEntries ? patternEntries[localPatternIdx].mainQueryIdx : defaultMainQueryIdx;
  const int patternId     = patternEntries ? patternEntries[localPatternIdx].patternId : defaultPatternId;
  const int patternMolIdx = patternEntries ? patternEntries[localPatternIdx].patternMolIdx : localPatternIdx;

  const int globalPairIdx = targetIdx * outputNumQueries + mainQueryIdx;

  if (globalPairIdx < miniBatchPairOffset || globalPairIdx >= miniBatchPairOffset + miniBatchSize) {
    return;
  }

  const int batchLocalPairIdx = globalPairIdx - miniBatchPairOffset;

  TargetMoleculeView target  = getMolecule(targets, targetIdx);
  QueryMoleculeView  pattern = getMolecule(patterns, patternMolIdx);

  constexpr std::size_t kLabelMatrixBitsT  = MaxTargetAtoms * MaxQueryAtoms;
  constexpr std::size_t kLabelMatrixWordsT = kLabelMatrixBitsT / 32;
  using LabelMatrixStorageT                = FlatBitVect<kLabelMatrixBitsT>;
  using LabelMatrixViewT                   = BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>;

  __shared__ LabelMatrixStorageT sharedLabelMatrix;
  LabelMatrixViewT               labelMatrix(&sharedLabelMatrix);

  __shared__ TargetAtomBonds sharedTargetBonds[MaxTargetAtoms];
  __shared__ QueryAtomBonds  sharedPatternBonds[MaxQueryAtoms];

  const uint32_t* globalIn   = labelMatrixBuffer + blockIdx.x * kLabelMatrixWordsT;
  const int       tid        = threadIdx.x;
  const int       numThreads = blockDim.x;

  loadLabelMatrixToShared<MaxTargetAtoms, MaxQueryAtoms>(sharedLabelMatrix, globalIn);

  const int numTargetAtoms  = target.numAtoms;
  const int numPatternAtoms = pattern.numAtoms;

  const int       totalTargetBytes = numTargetAtoms * sizeof(TargetAtomBonds);
  const int       totalTargetWords = (totalTargetBytes + 3) / 4;
  const uint32_t* targetBondsSrc   = reinterpret_cast<const uint32_t*>(target.targetAtomBonds);
  uint32_t*       targetBondsDst   = reinterpret_cast<uint32_t*>(sharedTargetBonds);
  for (int i = tid; i < totalTargetWords; i += numThreads) {
    targetBondsDst[i] = targetBondsSrc[i];
  }

  constexpr int kQueryBondWords = sizeof(QueryAtomBonds) / sizeof(uint32_t);
  static_assert(sizeof(QueryAtomBonds) % sizeof(uint32_t) == 0, "QueryAtomBonds must be word-aligned");
  const uint32_t* patternBondsSrc   = reinterpret_cast<const uint32_t*>(pattern.queryAtomBonds);
  uint32_t*       patternBondsDst   = reinterpret_cast<uint32_t*>(sharedPatternBonds);
  const int       totalPatternWords = numPatternAtoms * kQueryBondWords;
  for (int i = tid; i < totalPatternWords; i += numThreads) {
    patternBondsDst[i] = patternBondsSrc[i];
  }
  __syncthreads();

  target.targetAtomBonds = sharedTargetBonds;
  pattern.queryAtomBonds = sharedPatternBonds;

  __shared__ int sharedMatchCount;
  __shared__ int sharedReportedCount;

  if (threadIdx.x == 0) {
    sharedMatchCount    = 0;
    sharedReportedCount = 0;
  }
  __syncthreads();

  PaintModeParams paintParams;
  paintParams.recursiveBits  = outputRecursiveBits;
  paintParams.patternId      = patternId;
  paintParams.maxTargetAtoms = maxTargetAtoms;
  paintParams.outputPairIdx  = batchLocalPairIdx;

  constexpr int gsiBuffersPerBlock = 2;

  if constexpr (Algo == SubstructAlgorithm::GSI) {
    constexpr int kBlockSizeT = getBlockSizeForConfig<MaxTargetAtoms>();
    constexpr int kMaxPartialsT =
      getMaxPartialsForSM<MaxTargetAtoms, MaxQueryAtoms>(getComputeCapability(), kBlockSizeT);
    static_assert(kMaxPartialsT > 0,
                  "Insufficient shared memory for GSI partials - check block size for MaxTargetAtoms/MaxQueryAtoms");
    __shared__ PartialMatchT<MaxQueryAtoms> gsiPartials[kMaxPartialsT * 2];

    PartialMatchT<MaxQueryAtoms>* blockOverflowA = overflowA + blockIdx.x * gsiBuffersPerBlock * overflowCapacity;
    PartialMatchT<MaxQueryAtoms>* blockOverflowB = blockOverflowA + overflowCapacity;

    gsiBFSSearchGPU<MaxTargetAtoms, MaxQueryAtoms, MaxBondsPerAtom, SubstructOutputMode::PaintBits>(
      target,
      pattern,
      labelMatrix,
      gsiPartials,
      kMaxPartialsT,
      blockOverflowA,
      blockOverflowB,
      overflowCapacity,
      &sharedMatchCount,
      &sharedReportedCount,
      nullptr,
      0,
      0,
      paintParams);
  }
}

// =============================================================================
// Explicit Template Instantiations for All 24 Valid Configurations
// =============================================================================

// Helper macro to instantiate both VF2 and GSI for a given configuration
#define INSTANTIATE_SUBSTRUCT_KERNELS(MaxT, MaxQ, MaxB)                                      \
  template __global__ void substructMatchKernelT<MaxT, MaxQ, MaxB, SubstructAlgorithm::VF2>( \
    TargetMoleculesDeviceView,                                                               \
    QueryMoleculesDeviceView,                                                                \
    SubstructMatchResultsDeviceViewT<MaxQ>,                                                  \
    const int*,                                                                              \
    int,                                                                                     \
    const int*,                                                                              \
    DeviceTimingsData*);                                                                     \
  template __global__ void substructMatchKernelT<MaxT, MaxQ, MaxB, SubstructAlgorithm::GSI>( \
    TargetMoleculesDeviceView,                                                               \
    QueryMoleculesDeviceView,                                                                \
    SubstructMatchResultsDeviceViewT<MaxQ>,                                                  \
    const int*,                                                                              \
    int,                                                                                     \
    const int*,                                                                              \
    DeviceTimingsData*);                                                                     \
  template __global__ void substructPaintKernelT<MaxT, MaxQ, MaxB, SubstructAlgorithm::GSI>( \
    TargetMoleculesDeviceView,                                                               \
    QueryMoleculesDeviceView,                                                                \
    const BatchedPatternEntry*,                                                              \
    int,                                                                                     \
    uint32_t*,                                                                               \
    int,                                                                                     \
    int,                                                                                     \
    int,                                                                                     \
    int,                                                                                     \
    int,                                                                                     \
    int,                                                                                     \
    PartialMatchT<MaxQ>*,                                                                    \
    PartialMatchT<MaxQ>*,                                                                    \
    int,                                                                                     \
    const uint32_t*,                                                                         \
    int);

// Target 32, Query 16
INSTANTIATE_SUBSTRUCT_KERNELS(32, 16, 4)
INSTANTIATE_SUBSTRUCT_KERNELS(32, 16, 6)
INSTANTIATE_SUBSTRUCT_KERNELS(32, 16, 8)

// Target 32, Query 32
INSTANTIATE_SUBSTRUCT_KERNELS(32, 32, 4)
INSTANTIATE_SUBSTRUCT_KERNELS(32, 32, 6)
INSTANTIATE_SUBSTRUCT_KERNELS(32, 32, 8)

// Target 64, Query 16
INSTANTIATE_SUBSTRUCT_KERNELS(64, 16, 4)
INSTANTIATE_SUBSTRUCT_KERNELS(64, 16, 6)
INSTANTIATE_SUBSTRUCT_KERNELS(64, 16, 8)

// Target 64, Query 32
INSTANTIATE_SUBSTRUCT_KERNELS(64, 32, 4)
INSTANTIATE_SUBSTRUCT_KERNELS(64, 32, 6)
INSTANTIATE_SUBSTRUCT_KERNELS(64, 32, 8)

// Target 64, Query 64
INSTANTIATE_SUBSTRUCT_KERNELS(64, 64, 4)
INSTANTIATE_SUBSTRUCT_KERNELS(64, 64, 6)
INSTANTIATE_SUBSTRUCT_KERNELS(64, 64, 8)

// Target 128, Query 16
INSTANTIATE_SUBSTRUCT_KERNELS(128, 16, 4)
INSTANTIATE_SUBSTRUCT_KERNELS(128, 16, 6)
INSTANTIATE_SUBSTRUCT_KERNELS(128, 16, 8)

// Target 128, Query 32
INSTANTIATE_SUBSTRUCT_KERNELS(128, 32, 4)
INSTANTIATE_SUBSTRUCT_KERNELS(128, 32, 6)
INSTANTIATE_SUBSTRUCT_KERNELS(128, 32, 8)

// Target 128, Query 64
INSTANTIATE_SUBSTRUCT_KERNELS(128, 64, 4)
INSTANTIATE_SUBSTRUCT_KERNELS(128, 64, 6)
INSTANTIATE_SUBSTRUCT_KERNELS(128, 64, 8)

#undef INSTANTIATE_SUBSTRUCT_KERNELS

// Label matrix kernel instantiations (one per target/query combo, no MaxBonds needed)
#define INSTANTIATE_LABEL_MATRIX_KERNEL(MaxT, MaxQ)                                        \
  template __global__ void labelMatrixKernelT<MaxT, MaxQ>(TargetMoleculesDeviceView,       \
                                                          QueryMoleculesDeviceView,        \
                                                          const int*,                      \
                                                          int,                             \
                                                          uint32_t*,                       \
                                                          const uint32_t*,                 \
                                                          int,                             \
                                                          const int*);                     \
  template __global__ void labelMatrixPaintKernelT<MaxT, MaxQ>(TargetMoleculesDeviceView,  \
                                                               QueryMoleculesDeviceView,   \
                                                               const BatchedPatternEntry*, \
                                                               int,                        \
                                                               int,                        \
                                                               int,                        \
                                                               int,                        \
                                                               uint32_t*,                  \
                                                               int,                        \
                                                               const uint32_t*,            \
                                                               int,                        \
                                                               ZeroBuffersSpec);

INSTANTIATE_LABEL_MATRIX_KERNEL(32, 16)
INSTANTIATE_LABEL_MATRIX_KERNEL(32, 32)
INSTANTIATE_LABEL_MATRIX_KERNEL(64, 16)
INSTANTIATE_LABEL_MATRIX_KERNEL(64, 32)
INSTANTIATE_LABEL_MATRIX_KERNEL(64, 64)
INSTANTIATE_LABEL_MATRIX_KERNEL(128, 16)
INSTANTIATE_LABEL_MATRIX_KERNEL(128, 32)
INSTANTIATE_LABEL_MATRIX_KERNEL(128, 64)

#undef INSTANTIATE_LABEL_MATRIX_KERNEL

}  // anonymous namespace

// =============================================================================
// Template Config Dispatch Macros
// =============================================================================

// clang-format off

/**
 * @brief Dispatch macro for kernels that only need (MaxTargetAtoms, MaxQueryAtoms).
 *
 * Used by label matrix kernels where MaxBondsPerAtom doesn't affect the kernel.
 * Groups B4/B6/B8 variants together with case fallthrough.
 */
#define DISPATCH_BY_TQ_CONFIG(config, func, ...)                                                          \
  do {                                                                                                    \
    switch (config) {                                                                                     \
      case SubstructTemplateConfig::Config_T32_Q16_B4:                                                    \
      case SubstructTemplateConfig::Config_T32_Q16_B6:                                                    \
      case SubstructTemplateConfig::Config_T32_Q16_B8:   func<32, 16>(__VA_ARGS__);   break;              \
      case SubstructTemplateConfig::Config_T32_Q32_B4:                                                    \
      case SubstructTemplateConfig::Config_T32_Q32_B6:                                                    \
      case SubstructTemplateConfig::Config_T32_Q32_B8:   func<32, 32>(__VA_ARGS__);   break;              \
      case SubstructTemplateConfig::Config_T64_Q16_B4:                                                    \
      case SubstructTemplateConfig::Config_T64_Q16_B6:                                                    \
      case SubstructTemplateConfig::Config_T64_Q16_B8:   func<64, 16>(__VA_ARGS__);   break;              \
      case SubstructTemplateConfig::Config_T64_Q32_B4:                                                    \
      case SubstructTemplateConfig::Config_T64_Q32_B6:                                                    \
      case SubstructTemplateConfig::Config_T64_Q32_B8:   func<64, 32>(__VA_ARGS__);   break;              \
      case SubstructTemplateConfig::Config_T64_Q64_B4:                                                    \
      case SubstructTemplateConfig::Config_T64_Q64_B6:                                                    \
      case SubstructTemplateConfig::Config_T64_Q64_B8:   func<64, 64>(__VA_ARGS__);   break;              \
      case SubstructTemplateConfig::Config_T128_Q16_B4:                                                   \
      case SubstructTemplateConfig::Config_T128_Q16_B6:                                                   \
      case SubstructTemplateConfig::Config_T128_Q16_B8:  func<128, 16>(__VA_ARGS__);  break;              \
      case SubstructTemplateConfig::Config_T128_Q32_B4:                                                   \
      case SubstructTemplateConfig::Config_T128_Q32_B6:                                                   \
      case SubstructTemplateConfig::Config_T128_Q32_B8:  func<128, 32>(__VA_ARGS__);  break;              \
      case SubstructTemplateConfig::Config_T128_Q64_B4:                                                   \
      case SubstructTemplateConfig::Config_T128_Q64_B6:                                                   \
      case SubstructTemplateConfig::Config_T128_Q64_B8:                                                   \
      default:                                           func<128, 64>(__VA_ARGS__);  break;              \
    }                                                                                                     \
  } while (0)

/**
 * @brief Dispatch macro for kernels that need (MaxTargetAtoms, MaxQueryAtoms, MaxBondsPerAtom).
 *
 * Used by match and paint kernels where all three template parameters affect behavior.
 */
#define DISPATCH_BY_TQB_CONFIG(config, func, ...)                                                         \
  do {                                                                                                    \
    switch (config) {                                                                                     \
      case SubstructTemplateConfig::Config_T32_Q16_B4:   func<32, 16, 4>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T32_Q16_B6:   func<32, 16, 6>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T32_Q16_B8:   func<32, 16, 8>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T32_Q32_B4:   func<32, 32, 4>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T32_Q32_B6:   func<32, 32, 6>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T32_Q32_B8:   func<32, 32, 8>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T64_Q16_B4:   func<64, 16, 4>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T64_Q16_B6:   func<64, 16, 6>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T64_Q16_B8:   func<64, 16, 8>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T64_Q32_B4:   func<64, 32, 4>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T64_Q32_B6:   func<64, 32, 6>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T64_Q32_B8:   func<64, 32, 8>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T64_Q64_B4:   func<64, 64, 4>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T64_Q64_B6:   func<64, 64, 6>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T64_Q64_B8:   func<64, 64, 8>(__VA_ARGS__);   break;           \
      case SubstructTemplateConfig::Config_T128_Q16_B4:  func<128, 16, 4>(__VA_ARGS__);  break;           \
      case SubstructTemplateConfig::Config_T128_Q16_B6:  func<128, 16, 6>(__VA_ARGS__);  break;           \
      case SubstructTemplateConfig::Config_T128_Q16_B8:  func<128, 16, 8>(__VA_ARGS__);  break;           \
      case SubstructTemplateConfig::Config_T128_Q32_B4:  func<128, 32, 4>(__VA_ARGS__);  break;           \
      case SubstructTemplateConfig::Config_T128_Q32_B6:  func<128, 32, 6>(__VA_ARGS__);  break;           \
      case SubstructTemplateConfig::Config_T128_Q32_B8:  func<128, 32, 8>(__VA_ARGS__);  break;           \
      case SubstructTemplateConfig::Config_T128_Q64_B4:  func<128, 64, 4>(__VA_ARGS__);  break;           \
      case SubstructTemplateConfig::Config_T128_Q64_B6:  func<128, 64, 6>(__VA_ARGS__);  break;           \
      case SubstructTemplateConfig::Config_T128_Q64_B8:                                                   \
      default:                                           func<128, 64, 8>(__VA_ARGS__);  break;           \
    }                                                                                                     \
  } while (0)

// clang-format on

// =============================================================================
// Public Launch Wrapper Implementations
// =============================================================================

namespace {

template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
void launchLabelMatrixKernelForConfig(TargetMoleculesDeviceView targets,
                                      QueryMoleculesDeviceView  queries,
                                      const int*                pairIndices,
                                      int                       numPairs,
                                      int                       numQueries,
                                      uint32_t*                 labelMatrixBuffer,
                                      const uint32_t*           recursiveMatchBits,
                                      int                       maxTargetAtoms,
                                      const int*                batchLocalIndices,
                                      cudaStream_t              stream) {
  constexpr int kBlockSize = getBlockSizeForConfig<MaxTargetAtoms>();
  labelMatrixKernelT<MaxTargetAtoms, MaxQueryAtoms><<<numPairs, kBlockSize, 0, stream>>>(targets,
                                                                                         queries,
                                                                                         pairIndices,
                                                                                         numQueries,
                                                                                         labelMatrixBuffer,
                                                                                         recursiveMatchBits,
                                                                                         maxTargetAtoms,
                                                                                         batchLocalIndices);
}

}  // namespace

void launchLabelMatrixKernel(SubstructTemplateConfig   config,
                             TargetMoleculesDeviceView targets,
                             QueryMoleculesDeviceView  queries,
                             const int*                pairIndices,
                             int                       numPairs,
                             int                       numQueries,
                             uint32_t*                 labelMatrixBuffer,
                             const uint32_t*           recursiveMatchBits,
                             int                       maxTargetAtoms,
                             const int*                batchLocalIndices,
                             cudaStream_t              stream) {
  DISPATCH_BY_TQ_CONFIG(config,
                        launchLabelMatrixKernelForConfig,
                        targets,
                        queries,
                        pairIndices,
                        numPairs,
                        numQueries,
                        labelMatrixBuffer,
                        recursiveMatchBits,
                        maxTargetAtoms,
                        batchLocalIndices,
                        stream);
}

namespace {

template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
void launchLabelMatrixPaintKernelForConfig(TargetMoleculesDeviceView  targets,
                                           QueryMoleculesDeviceView   patterns,
                                           const BatchedPatternEntry* patternEntries,
                                           int                        numPatterns,
                                           int                        numBlocks,
                                           int                        numQueries,
                                           int                        miniBatchPairOffset,
                                           int                        miniBatchSize,
                                           uint32_t*                  labelMatrixBuffer,
                                           int                        firstTargetIdx,
                                           const uint32_t*            recursiveMatchBits,
                                           int                        maxTargetAtoms,
                                           ZeroBuffersSpec            zeroBuffers,
                                           cudaStream_t               stream) {
  constexpr int kBlockSize = getBlockSizeForConfig<MaxTargetAtoms>();
  labelMatrixPaintKernelT<MaxTargetAtoms, MaxQueryAtoms><<<numBlocks, kBlockSize, 0, stream>>>(targets,
                                                                                               patterns,
                                                                                               patternEntries,
                                                                                               numPatterns,
                                                                                               numQueries,
                                                                                               miniBatchPairOffset,
                                                                                               miniBatchSize,
                                                                                               labelMatrixBuffer,
                                                                                               firstTargetIdx,
                                                                                               recursiveMatchBits,
                                                                                               maxTargetAtoms,
                                                                                               zeroBuffers);
}

}  // namespace

void launchLabelMatrixPaintKernel(SubstructTemplateConfig        config,
                                  TargetMoleculesDeviceView      targets,
                                  QueryMoleculesDeviceView       patterns,
                                  const BatchedPatternEntry*     patternEntries,
                                  int                            numPatterns,
                                  int                            numBlocks,
                                  int                            numQueries,
                                  int                            miniBatchPairOffset,
                                  int                            miniBatchSize,
                                  uint32_t*                      labelMatrixBuffer,
                                  int                            firstTargetIdx,
                                  const uint32_t*                recursiveMatchBits,
                                  int                            maxTargetAtoms,
                                  std::optional<ZeroBuffersSpec> zeroBuffers,
                                  cudaStream_t                   stream) {
  DISPATCH_BY_TQ_CONFIG(config,
                        launchLabelMatrixPaintKernelForConfig,
                        targets,
                        patterns,
                        patternEntries,
                        numPatterns,
                        numBlocks,
                        numQueries,
                        miniBatchPairOffset,
                        miniBatchSize,
                        labelMatrixBuffer,
                        firstTargetIdx,
                        recursiveMatchBits,
                        maxTargetAtoms,
                        zeroBuffers.value_or(ZeroBuffersSpec{}),
                        stream);
}

namespace {

template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms, int MaxBondsPerAtom>
void launchSubstructPaintKernelForConfig(SubstructAlgorithm         algorithm,
                                         TargetMoleculesDeviceView  targets,
                                         QueryMoleculesDeviceView   patterns,
                                         const BatchedPatternEntry* patternEntries,
                                         int                        numPatterns,
                                         int                        numBlocks,
                                         uint32_t*                  outputRecursiveBits,
                                         int                        maxTargetAtoms,
                                         int                        outputNumQueries,
                                         int                        defaultPatternId,
                                         int                        defaultMainQueryIdx,
                                         int                        miniBatchPairOffset,
                                         int                        miniBatchSize,
                                         PartialMatch*              overflowA,
                                         PartialMatch*              overflowB,
                                         int                        overflowCapacity,
                                         const uint32_t*            labelMatrixBuffer,
                                         int                        firstTargetIdx,
                                         cudaStream_t               stream) {
  constexpr int kBlockSize = getBlockSizeForConfig<MaxTargetAtoms>();
  switch (algorithm) {
    case SubstructAlgorithm::VF2:
      break;
    case SubstructAlgorithm::GSI:
      substructPaintKernelT<MaxTargetAtoms, MaxQueryAtoms, MaxBondsPerAtom, SubstructAlgorithm::GSI>
        <<<numBlocks, kBlockSize, 0, stream>>>(targets,
                                               patterns,
                                               patternEntries,
                                               numPatterns,
                                               outputRecursiveBits,
                                               maxTargetAtoms,
                                               outputNumQueries,
                                               defaultPatternId,
                                               defaultMainQueryIdx,
                                               miniBatchPairOffset,
                                               miniBatchSize,
                                               reinterpret_cast<PartialMatchT<MaxQueryAtoms>*>(overflowA),
                                               reinterpret_cast<PartialMatchT<MaxQueryAtoms>*>(overflowB),
                                               overflowCapacity,
                                               labelMatrixBuffer,
                                               firstTargetIdx);
      break;
  }
}

}  // namespace

void launchSubstructPaintKernel(SubstructTemplateConfig    config,
                                SubstructAlgorithm         algorithm,
                                TargetMoleculesDeviceView  targets,
                                QueryMoleculesDeviceView   patterns,
                                const BatchedPatternEntry* patternEntries,
                                int                        numPatterns,
                                int                        numBlocks,
                                uint32_t*                  outputRecursiveBits,
                                int                        maxTargetAtoms,
                                int                        outputNumQueries,
                                int                        defaultPatternId,
                                int                        defaultMainQueryIdx,
                                int                        miniBatchPairOffset,
                                int                        miniBatchSize,
                                PartialMatch*              overflowA,
                                PartialMatch*              overflowB,
                                int                        overflowCapacity,
                                const uint32_t*            labelMatrixBuffer,
                                int                        firstTargetIdx,
                                cudaStream_t               stream) {
  DISPATCH_BY_TQB_CONFIG(config,
                         launchSubstructPaintKernelForConfig,
                         algorithm,
                         targets,
                         patterns,
                         patternEntries,
                         numPatterns,
                         numBlocks,
                         outputRecursiveBits,
                         maxTargetAtoms,
                         outputNumQueries,
                         defaultPatternId,
                         defaultMainQueryIdx,
                         miniBatchPairOffset,
                         miniBatchSize,
                         overflowA,
                         overflowB,
                         overflowCapacity,
                         labelMatrixBuffer,
                         firstTargetIdx,
                         stream);
}

void configureSubstructKernelsSharedMem() {
  if (sharedMemCarveoutConfigured())
    return;

  configureSharedMemCarveout(
    substructMatchKernelT<kMaxTargetAtoms, kMaxQueryAtoms, kMaxBondsPerAtom, SubstructAlgorithm::GSI>);
  configureSharedMemCarveout(
    substructPaintKernelT<kMaxTargetAtoms, kMaxQueryAtoms, kMaxBondsPerAtom, SubstructAlgorithm::GSI>);

  sharedMemCarveoutConfigured() = true;
}

namespace {

template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms, int MaxBondsPerAtom>
void launchMatchKernelForConfig(SubstructAlgorithm            algorithm,
                                TargetMoleculesDeviceView     targets,
                                QueryMoleculesDeviceView      queries,
                                const MiniBatchResultsDevice& miniBatchResults,
                                const int*                    pairIndices,
                                int                           numPairs,
                                int                           numQueries,
                                const int*                    batchLocalIndices,
                                DeviceTimingsData*            timings,
                                cudaStream_t                  stream) {
  constexpr std::size_t labelMatrixWordsT = MaxTargetAtoms * MaxQueryAtoms / 32;
  constexpr int         kBlockSize        = getBlockSizeForConfig<MaxTargetAtoms>();

  SubstructMatchResultsDeviceViewT<MaxQueryAtoms> results;
  results.matchCounts              = miniBatchResults.matchCounts();
  results.reportedCounts           = miniBatchResults.reportedCounts();
  results.pairMatchStarts          = miniBatchResults.pairMatchStarts();
  results.matchIndices             = miniBatchResults.matchIndices();
  results.numQueries               = miniBatchResults.numQueries();
  results.queryAtomCounts          = miniBatchResults.queryAtomCounts();
  results.overflowBuffer           = reinterpret_cast<PartialMatchT<MaxQueryAtoms>*>(miniBatchResults.overflowBuffer());
  results.overflowEntriesPerBuffer = kOverflowEntriesPerBuffer;
  results.overflowBuffersPerBlock  = miniBatchResults.overflowBuffersPerBlock();
  results.recursiveMatchBits       = miniBatchResults.recursiveMatchBits();
  results.maxTargetAtoms           = miniBatchResults.maxTargetAtoms();
  results.labelMatrixBuffer        = miniBatchResults.labelMatrixBuffer();
  results.labelMatrixWords         = labelMatrixWordsT;
  results.maxMatchesToFind         = miniBatchResults.maxMatchesToFind();
  results.countOnly                = miniBatchResults.countOnly();
  results.overflowFlags            = miniBatchResults.overflowFlags();

  switch (algorithm) {
    case SubstructAlgorithm::VF2:
      substructMatchKernelT<MaxTargetAtoms, MaxQueryAtoms, MaxBondsPerAtom, SubstructAlgorithm::VF2>
        <<<numPairs, kBlockSize, 0, stream>>>(targets,
                                              queries,
                                              results,
                                              pairIndices,
                                              numQueries,
                                              batchLocalIndices,
                                              timings);
      break;
    case SubstructAlgorithm::GSI:
      substructMatchKernelT<MaxTargetAtoms, MaxQueryAtoms, MaxBondsPerAtom, SubstructAlgorithm::GSI>
        <<<numPairs, kBlockSize, 0, stream>>>(targets,
                                              queries,
                                              results,
                                              pairIndices,
                                              numQueries,
                                              batchLocalIndices,
                                              timings);
      break;
  }
}

}  // namespace

void launchSubstructMatchKernel(SubstructTemplateConfig       config,
                                SubstructAlgorithm            algorithm,
                                TargetMoleculesDeviceView     targets,
                                QueryMoleculesDeviceView      queries,
                                const MiniBatchResultsDevice& miniBatchResults,
                                const int*                    pairIndices,
                                int                           numPairs,
                                int                           numQueries,
                                const int*                    batchLocalIndices,
                                DeviceTimingsData*            timings,
                                cudaStream_t                  stream) {
  DISPATCH_BY_TQB_CONFIG(config,
                         launchMatchKernelForConfig,
                         algorithm,
                         targets,
                         queries,
                         miniBatchResults,
                         pairIndices,
                         numPairs,
                         numQueries,
                         batchLocalIndices,
                         timings,
                         stream);
}

}  // namespace nvMolKit
