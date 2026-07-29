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

#ifndef NVMOLKIT_SUBSTRUCT_ALGOS_CUH
#define NVMOLKIT_SUBSTRUCT_ALGOS_CUH

#include <cooperative_groups.h>

#include <cstdint>

#include "src/data_structures/flat_bit_vect.h"
#include "src/substruct/molecules_device.cuh"
#include "src/substruct/packed_bonds_device.cuh"
#include "src/substruct/substruct_debug.h"
#include "src/substruct/substruct_types.h"
#include "src/utils/device_timings.cuh"

namespace nvMolKit {

// =============================================================================
// Shared Constants
// =============================================================================

constexpr int kWarpSize = 32;

// =============================================================================
// Output Mode for Substructure Search
// =============================================================================

/**
 * @brief Determines how matches are recorded.
 */
enum class SubstructOutputMode {
  StoreMatches,  ///< Store full match mappings to output buffer
  PaintBits      ///< Paint recursive match bit for first matched atom (no storage)
};

/**
 * @brief Parameters for paint mode output.
 */
struct PaintModeParams {
  uint32_t* recursiveBits;   ///< Buffer to paint bits into [maxTargetAtoms per pair]
  int       patternId;       ///< Bit position (0-31) to set
  int       maxTargetAtoms;  ///< Stride for indexing recursiveBits
  int       outputPairIdx;   ///< Which pair's buffer to write to
};

// =============================================================================
// Helper function for checking if target atom is used in mapping
// =============================================================================

/**
 * @brief Check if a target atom is already used in a partial mapping.
 *
 * Only checks mapping[0..depth-1] since query atoms are processed in order.
 * This avoids the need for -1 sentinel initialization.
 *
 * @param mapping The partial mapping array
 * @param depth Current search depth (number of assigned query atoms)
 * @param targetAtom Target atom to check
 * @return true if targetAtom is already used in the mapping
 */
__device__ __forceinline__ bool isTargetUsedInMapping(const int8_t* mapping, int depth, int targetAtom) {
  for (int q = 0; q < depth; ++q) {
    if (mapping[q] == targetAtom) {
      return true;
    }
  }
  return false;
}

// =============================================================================
// VF2 Data Structures (templated)
// =============================================================================

/**
 * @brief State for VF2 iterative search (per-warp in shared memory).
 *
 * @tparam MaxQueryAtoms Maximum query atoms for array sizing
 *
 * Maintains partial match and exploration stack for DFS backtracking.
 */
template <std::size_t MaxQueryAtoms = kMaxQueryAtoms> struct VF2StateT {
  static constexpr std::size_t kMaxQueryAtomsValue = MaxQueryAtoms;
  int8_t                       mapping[MaxQueryAtoms];       ///< mapping[q] = target atom idx (only [0..depth-1] valid)
  int8_t                       candidateIdx[MaxQueryAtoms];  ///< Current candidate index at each stack level
  int                          depth;                        ///< Current recursion depth (0 to numQueryAtoms-1)
  int                          matchCount;                   ///< Number of complete matches found

  __device__ __forceinline__ void init() {
#pragma unroll
    for (std::size_t i = 0; i < MaxQueryAtoms; ++i) {
      candidateIdx[i] = 0;
    }
    depth      = 0;
    matchCount = 0;
  }

  __device__ __forceinline__ bool isTargetUsed(int targetIdx) const {
    return isTargetUsedInMapping(mapping, depth, targetIdx);
  }
};

// =============================================================================
// VF2 Algorithm Implementation
// =============================================================================

/**
 * @brief VF2 iterative DFS search for subgraph isomorphism.
 *
 * Each warp explores from a different starting target atom for query atom 0.
 * Uses explicit stack to avoid recursion and maintain uniform control flow.
 *
 * @tparam MaxTargetAtoms Maximum target atoms (for label matrix sizing)
 * @tparam MaxQueryAtoms Maximum query atoms (for label matrix sizing)
 * @tparam MaxBondsPerAtom Maximum bonds per atom (for edge consistency loop unrolling)
 * @param target Target molecule view
 * @param query Query molecule view
 * @param labelMatrix Precomputed label compatibility matrix
 * @param state VF2 state in shared memory (per warp)
 * @param startingTargetAtom Starting target atom for this warp (query atom 0)
 * @param matchCount Output: number of matches found
 * @param reportedCount Output: number of matches written (capped)
 * @param matchIndices Output: match index buffer
 * @param totalMatchStorageCapacity Output buffer capacity (matches beyond this are counted but not stored)
 * @param matchOffset Offset into matchIndices for this pair
 * @param maxMatchesToFind Stop searching after finding this many matches (-1 = no limit)
 * @param countOnly If true, count matches but don't store them
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms, int MaxBondsPerAtom = kMaxBondsPerAtom>
__device__ void vf2SearchGPU(const TargetMoleculeView&                             target,
                             const QueryMoleculeView&                              query,
                             const BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>& labelMatrix,
                             VF2StateT<MaxQueryAtoms>&                             state,
                             int                                                   startingTargetAtom,
                             int*                                                  matchCount,
                             int*                                                  reportedCount,
                             int16_t*                                              matchIndices,
                             int                                                   totalMatchStorageCapacity,
                             int                                                   matchOffset,
                             int                                                   maxMatchesToFind = -1,
                             bool                                                  countOnly        = false) {
  namespace cg     = cooperative_groups;
  auto      tile32 = cg::tiled_partition<32>(cg::this_thread_block());
  const int laneId = tile32.thread_rank();

  const int numQueryAtoms  = query.numAtoms;
  const int numTargetAtoms = target.numAtoms;

  // Only lane 0 does the DFS logic; other lanes assist with parallel candidate evaluation
  if (laneId != 0) {
    return;
  }

  // Check if starting atom is valid
  if (startingTargetAtom >= numTargetAtoms || !labelMatrix.get(startingTargetAtom, 0)) {
    return;
  }

  state.init();
  state.mapping[0] = static_cast<int8_t>(startingTargetAtom);
  state.depth      = 1;

  const bool hasEarlyExitLimit = (maxMatchesToFind >= 0);

  // Iterative DFS
  while (state.depth > 0) {
    // Check early exit
    if (hasEarlyExitLimit && *matchCount >= maxMatchesToFind) {
      break;
    }

    if (state.depth == numQueryAtoms) {
      // Complete match found
      const int currentMatchCount = atomicAdd(matchCount, 1);

      if (!countOnly && currentMatchCount < totalMatchStorageCapacity) {
        // Write match to output
        const int writeOffset = matchOffset + currentMatchCount * numQueryAtoms;
        for (int q = 0; q < numQueryAtoms; ++q) {
          matchIndices[writeOffset + q] = state.mapping[q];
        }
        atomicAdd(reportedCount, 1);
      }

      // Backtrack to find more matches
      --state.depth;
      if (state.depth > 0) {
        ++state.candidateIdx[state.depth];
      }
      continue;
    }

    const int currentQueryAtom = state.depth;
    int8_t&   candIdx          = state.candidateIdx[state.depth];

    // Find next valid candidate
    bool foundCandidate = false;
    while (candIdx < numTargetAtoms) {
      const int candidateTarget = candIdx;

      // Check feasibility
      const bool labelOk = labelMatrix.get(candidateTarget, currentQueryAtom);
      const bool notUsed = !state.isTargetUsed(candidateTarget);
      const bool edgeOk  = labelOk && notUsed &&
                          checkEdgeConsistencyPacked<MaxBondsPerAtom>(target.targetAtomBonds,
                                                                      query.getQueryBonds(currentQueryAtom),
                                                                      state.mapping,
                                                                      currentQueryAtom,
                                                                      candidateTarget);

      if (edgeOk) {
        // Extend match
        state.mapping[currentQueryAtom]     = static_cast<int8_t>(candidateTarget);
        state.candidateIdx[state.depth + 1] = 0;
        ++state.depth;
        foundCandidate = true;
        break;
      }

      ++candIdx;
    }

    if (!foundCandidate) {
      // Backtrack
      state.candidateIdx[state.depth] = 0;
      --state.depth;
      if (state.depth > 0) {
        ++state.candidateIdx[state.depth];
      } else {
        // Exhausted this starting point
        break;
      }
    }
  }
}

// =============================================================================
// GSI BFS Algorithm Implementation
// =============================================================================

/**
 * @brief GSI-inspired BFS level-by-level search.
 *
 * Processes query atoms in order, extending all partial matches at each level.
 * Uses shared memory with pre-allocated per-block overflow buffers.
 *
 * @tparam MaxTargetAtoms Maximum target atoms
 * @tparam MaxQueryAtoms Maximum query atoms
 * @tparam MaxBondsPerAtom Maximum bonds per atom (for edge consistency loop unrolling)
 * @tparam OutputMode How to record matches (StoreMatches or PaintBits)
 * @param target Target molecule view
 * @param query Query molecule view
 * @param labelMatrix Precomputed label compatibility matrix
 * @param sharedPartials Shared memory for partial matches (ping-pong buffers)
 * @param maxPartials Max partials in shared memory
 * @param overflowA Pre-allocated overflow buffer A (ping)
 * @param overflowB Pre-allocated overflow buffer B (pong)
 * @param maxOverflow Entries per overflow buffer
 * @param matchCount Output: number of matches found
 * @param reportedCount Output: number of matches written (StoreMatches) or painted (PaintBits)
 * @param matchIndices Output buffer (only used in StoreMatches mode)
 * @param totalMatchStorageCapacity Output buffer capacity (only used in StoreMatches mode)
 * @param matchOffset Offset into output (only used in StoreMatches mode)
 * @param paintParams Paint mode parameters (only used in PaintBits mode)
 * @param maxMatchesToFind Stop searching after finding this many matches (-1 = no limit)
 * @param countOnly If true, count matches but don't store them
 * @param timings Optional timing data collection
 * @param overflowFlag Output: set to 1 if partial or output buffers overflow (nullptr to skip)

 // TODO: Separate out match-writing logic from main BFS logic, as it can only happen on the last step.
 */
template <std::size_t         MaxTargetAtoms,
          std::size_t         MaxQueryAtoms,
          int                 MaxBondsPerAtom = kMaxBondsPerAtom,
          SubstructOutputMode OutputMode      = SubstructOutputMode::StoreMatches>
__device__ void gsiBFSSearchGPU(const TargetMoleculeView&                             target,
                                const QueryMoleculeView&                              query,
                                const BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>& labelMatrix,
                                PartialMatchT<MaxQueryAtoms>*                         sharedPartials,
                                int                                                   maxPartials,
                                PartialMatchT<MaxQueryAtoms>*                         overflowA,
                                PartialMatchT<MaxQueryAtoms>*                         overflowB,
                                int                                                   maxOverflow,
                                int*                                                  matchCount,
                                int*                                                  reportedCount,
                                int16_t*                                              matchIndices,
                                int                                                   totalMatchStorageCapacity,
                                int                                                   matchOffset,
                                PaintModeParams                                       paintParams      = {},
                                int                                                   maxMatchesToFind = -1,
                                bool                                                  countOnly        = false,
                                DeviceTimingsData*                                    timings          = nullptr,
                                uint8_t*                                              overflowFlag     = nullptr) {
  long long int t_start;
  DEVICE_TIMING_START(timings, 0, t_start);

  const int tid      = threadIdx.x;
  const int laneId   = tid % 32;
  const int warpId   = __shfl_sync(0xFFFFFFFF, tid / 32, 0);
  const int numWarps = blockDim.x / 32;

  const int numQueryAtoms  = query.numAtoms;
  const int numTargetAtoms = target.numAtoms;

  // Runtime-sized partial matches: stride = numQueryAtoms bytes per partial.
  // This allows fitting more partials when queries have fewer atoms.
  const int stride = numQueryAtoms;

  // Treat buffers as raw bytes for runtime-sized access
  int8_t* sharedBytes    = reinterpret_cast<int8_t*>(sharedPartials);
  int8_t* overflowABytes = reinterpret_cast<int8_t*>(overflowA);
  int8_t* overflowBBytes = reinterpret_cast<int8_t*>(overflowB);

  // Compute effective capacities based on runtime stride
  const int sharedBytesPerHalf   = maxPartials * sizeof(PartialMatchT<MaxQueryAtoms>);
  const int effectiveMaxPartials = sharedBytesPerHalf / stride;
  const int overflowBytes        = maxOverflow * sizeof(PartialMatchT<MaxQueryAtoms>);
  const int effectiveMaxOverflow = overflowBytes / stride;
  const int maxTotal             = effectiveMaxPartials + effectiveMaxOverflow;

  // Ping-pong byte pointers for shared memory halves
  int8_t* currentShared = sharedBytes;
  int8_t* nextShared    = sharedBytes + sharedBytesPerHalf;

  // Ping-pong overflow byte pointers
  int8_t* currentOverflow = overflowABytes;
  int8_t* nextOverflow    = overflowBBytes;

  __shared__ int currentCount;
  __shared__ int nextCount;
  __shared__ int partialOverflowFlag;

  if (tid == 0) {
    currentCount        = 0;
    nextCount           = 0;
    partialOverflowFlag = 0;
  }
  __syncthreads();

  DEVICE_TIMING_END(timings, 0, t_start);
  DEVICE_TIMING_START(timings, 1, t_start);

  if constexpr (kDebugGSI) {
    if (tid == 0) {
      printf(
        "[GSI] numQueryAtoms=%d, numTargetAtoms=%d, effectiveMaxPartials=%d, effectiveMaxOverflow=%d, maxTotal=%d\n",
        numQueryAtoms,
        numTargetAtoms,
        effectiveMaxPartials,
        effectiveMaxOverflow,
        maxTotal);
      printf("[GSI] query.hasQueryTrees()=%d\n", query.hasQueryTrees() ? 1 : 0);
    }
    __syncthreads();
  }

  // Initialize level 0: all candidates for query atom 0
  const bool singleAtomQuery   = (numQueryAtoms == 1);
  const bool hasEarlyExitLimit = (maxMatchesToFind >= 0);

  for (int t = tid; t < numTargetAtoms; t += blockDim.x) {
    // Check early exit before processing
    if (hasEarlyExitLimit && *matchCount >= maxMatchesToFind) {
      break;
    }
    if (labelMatrix.get(t, 0)) {
      if (singleAtomQuery) {
        const int matchIdx = atomicAdd(matchCount, 1);
        if constexpr (OutputMode == SubstructOutputMode::StoreMatches) {
          if (!countOnly && matchIdx < totalMatchStorageCapacity) {
            matchIndices[matchOffset + matchIdx] = static_cast<int16_t>(t);
            atomicAdd(reportedCount, 1);
          }
        } else {
          // Paint mode: set bit for this target atom
          atomicOr(&paintParams.recursiveBits[paintParams.outputPairIdx * paintParams.maxTargetAtoms + t],
                   1u << paintParams.patternId);
          atomicAdd(reportedCount, 1);
          if constexpr (kDebugGSI) {
            printf("[GSI Paint] pairIdx=%d, targetAtom=%d, patternId=%d, bit=0x%x\n",
                   paintParams.outputPairIdx,
                   t,
                   paintParams.patternId,
                   1u << paintParams.patternId);
          }
        }
      } else {
        const int slot = atomicAdd(&currentCount, 1);
        if (slot < effectiveMaxPartials) {
          int8_t* p     = currentShared + slot * stride;
          p[0]          = static_cast<int8_t>(t);
          p[stride - 1] = 1;  // nextQueryAtom
        } else if (slot < maxTotal) {
          if constexpr (kDebugGSI) {
            if (slot == effectiveMaxPartials) {
              printf("[GSI] Level 0: spilling to overflow buffer (slot=%d)\n", slot);
            }
          }
          const int overflowSlot = slot - effectiveMaxPartials;
          int8_t*   p            = currentOverflow + overflowSlot * stride;
          p[0]                   = static_cast<int8_t>(t);
          p[stride - 1]          = 1;  // nextQueryAtom
        } else {
          if constexpr (kDebugGSI) {
            if (slot == maxTotal) {
              printf("[GSI] WARNING: Level 0 OVERFLOW EXHAUSTED! slot=%d >= maxTotal=%d\n", slot, maxTotal);
            }
          }
          partialOverflowFlag = 1;
        }
      }
    }
  }
  __syncthreads();

  if constexpr (kDebugGSI) {
    if (tid == 0) {
      printf("[GSI] Level 0: %d candidates for query atom 0", currentCount);
      if (currentCount > effectiveMaxPartials) {
        printf(" (%d in shared, %d in overflow)", effectiveMaxPartials, currentCount - effectiveMaxPartials);
      }
      printf("\n");
      printf("[GSI] Level 0 candidates (shared): ");
      for (int i = 0; i < min(currentCount, effectiveMaxPartials) && i < 10; ++i) {
        printf("%d ", (int)currentShared[i * stride]);
      }
      if (currentCount > 10)
        printf("...");
      printf("\n");
    }
    __syncthreads();
  }

  DEVICE_TIMING_END(timings, 1, t_start);

  if (singleAtomQuery) {
    return;
  }

  DEVICE_TIMING_START(timings, 2, t_start);

  // BFS levels
  for (int level = 1; level < numQueryAtoms; ++level) {
    // Check early exit at start of level
    if (hasEarlyExitLimit && *matchCount >= maxMatchesToFind) {
      break;
    }

    const int queryAtom   = level;
    const int numPartials = min(currentCount, maxTotal);

    if constexpr (kDebugGSI) {
      if (tid == 0) {
        printf("[GSI] === Level %d: processing %d partials for queryAtom %d ===\n", level, numPartials, queryAtom);
      }
      __syncthreads();
    }

    if (tid == 0) {
      nextCount = 0;
    }
    __syncthreads();

    __shared__ int debugValidTotal;
    __shared__ int debugCheckedTotal;
    if constexpr (kDebugGSI) {
      if (tid == 0) {
        debugValidTotal   = 0;
        debugCheckedTotal = 0;
      }
      __syncthreads();
    }

    // Each warp processes partial matches in round-robin
    for (int pIdx = warpId; pIdx < numPartials; pIdx += numWarps) {
      // Get pointer to partial's mapping (raw byte access)
      const int8_t* partial = (pIdx < effectiveMaxPartials) ?
                                (currentShared + pIdx * stride) :
                                (currentOverflow + (pIdx - effectiveMaxPartials) * stride);

      if constexpr (kDebugGSI) {
        if (pIdx == 0 && laneId == 0 && level <= 5) {
          printf("[GSI] Level %d partial 0 mapping: ", level);
          for (int q = 0; q < numQueryAtoms; ++q) {
            printf("%d ", (int)partial[q]);
          }
          printf("\n");
        }
      }

      for (int tBase = 0; tBase < numTargetAtoms; tBase += kWarpSize) {
        const int t = tBase + laneId;

        bool valid   = false;
        bool labelOk = false;
        bool notUsed = false;
        bool edgeOk  = false;

        if (t < numTargetAtoms) {
          labelOk = labelMatrix.get(t, queryAtom);
          notUsed = !isTargetUsedInMapping(partial, queryAtom, t);
          edgeOk  = checkEdgeConsistencyPacked<MaxBondsPerAtom>(target.targetAtomBonds,
                                                               query.getQueryBonds(queryAtom),
                                                               partial,
                                                               queryAtom,
                                                               t);
          valid   = labelOk && notUsed && edgeOk;

          if constexpr (kDebugGSI) {
            atomicAdd(&debugCheckedTotal, 1);
            if (valid)
              atomicAdd(&debugValidTotal, 1);
          }
        }

        if constexpr (kDebugGSI) {
          if (pIdx == 0 && t < 5 && level <= 3) {
            printf("[GSI] Level %d partial 0 cand %d: labelOk=%d, notUsed=%d, edgeOk=%d, valid=%d\n",
                   level,
                   t,
                   labelOk ? 1 : 0,
                   notUsed ? 1 : 0,
                   edgeOk ? 1 : 0,
                   valid ? 1 : 0);
          }
        }

        if (valid) {
          if (level == numQueryAtoms - 1) {
            const int matchIdx = atomicAdd(matchCount, 1);
            if constexpr (kDebugGSI) {
              printf("[GSI] MATCH FOUND! matchIdx=%d, mapping: ", matchIdx);
              for (int q = 0; q < numQueryAtoms; ++q) {
                printf("%d ", (q == queryAtom) ? t : (int)partial[q]);
              }
              printf("\n");
            }
            if constexpr (OutputMode == SubstructOutputMode::StoreMatches) {
              if (!countOnly && matchIdx < totalMatchStorageCapacity) {
                const int writeOffset = matchOffset + matchIdx * numQueryAtoms;
                for (int q = 0; q < numQueryAtoms; ++q) {
                  matchIndices[writeOffset + q] = (q == queryAtom) ? static_cast<int16_t>(t) : partial[q];
                }
                atomicAdd(reportedCount, 1);
              }
            } else {
              // Paint mode: set bit for first target atom in this match (mapping[0])
              const int firstTargetAtom = partial[0];
              atomicOr(
                &paintParams.recursiveBits[paintParams.outputPairIdx * paintParams.maxTargetAtoms + firstTargetAtom],
                1u << paintParams.patternId);
              atomicAdd(reportedCount, 1);
            }
          } else {
            const int slot = atomicAdd(&nextCount, 1);
            if constexpr (kDebugGSI) {
              if (slot < 3 && level <= 3) {
                printf("[GSI] Level %d: enqueueing slot %d, queryAtom %d -> target %d\n", level, slot, queryAtom, t);
              }
            }
            // Write to shared memory ping-pong or global overflow
            // Only copy [0..queryAtom-1] from parent, then set [queryAtom]
            if (slot < effectiveMaxPartials) {
              int8_t* next = nextShared + slot * stride;
              for (int q = 0; q < queryAtom; ++q) {
                next[q] = partial[q];
              }
              next[queryAtom]  = static_cast<int8_t>(t);
              next[stride - 1] = static_cast<int8_t>(queryAtom + 1);  // nextQueryAtom
            } else if (slot < maxTotal) {
              if constexpr (kDebugGSI) {
                if (slot == effectiveMaxPartials) {
                  printf("[GSI] Level %d: spilling to overflow buffer (slot=%d)\n", level, slot);
                }
              }
              const int overflowSlot = slot - effectiveMaxPartials;
              int8_t*   next         = nextOverflow + overflowSlot * stride;
              for (int q = 0; q < queryAtom; ++q) {
                next[q] = partial[q];
              }
              next[queryAtom]  = static_cast<int8_t>(t);
              next[stride - 1] = static_cast<int8_t>(queryAtom + 1);  // nextQueryAtom
            } else {
              if constexpr (kDebugGSI) {
                if (slot == maxTotal) {
                  printf("[GSI] WARNING: Level %d OVERFLOW EXHAUSTED! slot=%d >= maxTotal=%d\n", level, slot, maxTotal);
                }
              }
              partialOverflowFlag = 1;
            }
          }
        }
      }
    }
    __syncthreads();

    if constexpr (kDebugGSI) {
      if (tid == 0) {
        printf("[GSI] Level %d summary: checked=%d, valid=%d, nextCount=%d",
               level,
               debugCheckedTotal,
               debugValidTotal,
               nextCount);
        if (nextCount > effectiveMaxPartials) {
          printf(" (%d in shared, %d in overflow)", effectiveMaxPartials, nextCount - effectiveMaxPartials);
        }
        if (nextCount > maxTotal) {
          printf(" [OVERFLOW LOST %d]", nextCount - maxTotal);
        }
        printf("\n");
      }
      __syncthreads();
    }

    // Swap buffers (just swap pointers, no copy needed)
    if (tid == 0) {
      currentCount = nextCount;
    }
    // Swap shared pointers (each thread has its own copy)
    int8_t* tmpShared   = currentShared;
    currentShared       = nextShared;
    nextShared          = tmpShared;
    // Swap overflow pointers
    int8_t* tmpOverflow = currentOverflow;
    currentOverflow     = nextOverflow;
    nextOverflow        = tmpOverflow;

    __syncthreads();

    if constexpr (kDebugGSI) {
      if (tid == 0 && currentCount == 0) {
        printf("[GSI] Level %d: no more partials, stopping early\n", level);
      }
    }
    if (currentCount == 0) {
      break;
    }
  }

  DEVICE_TIMING_END(timings, 2, t_start);

  if constexpr (kDebugGSI) {
    if (tid == 0) {
      printf("[GSI] DONE: final matchCount=%d, reportedCount=%d\n", *matchCount, *reportedCount);
    }
  }

  // Write overflow flag if requested
  if (tid == 0 && overflowFlag != nullptr && partialOverflowFlag) {
    *overflowFlag = 1;
  }
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_ALGOS_CUH
