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

#ifndef NVMOLKIT_SUBSTRUCT_KERNELS_H
#define NVMOLKIT_SUBSTRUCT_KERNELS_H

#include <cuda_runtime.h>

#include <optional>

#include "src/substruct/molecules.h"
#include "src/substruct/substruct_types.h"

namespace nvMolKit {

// Forward declarations
class MiniBatchResultsDevice;
struct DeviceTimingsData;

// =============================================================================
// Shared Memory Configuration Constants (host-side)
// =============================================================================

// =============================================================================
// Kernel Launch Wrappers
// =============================================================================

/**
 * @brief Launch label matrix computation kernel.
 *
 * One block per pair. Computes label matrix and writes to global buffer.
 *
 * @param targets Target molecules device view
 * @param queries Query molecules device view
 * @param pairIndices Global pair indices for this launch
 * @param numPairs Number of pairs to process
 * @param numQueries Total number of queries (for decoding pair indices)
 * @param labelMatrixBuffer Output label matrices [numPairs * kLabelMatrixWords]
 * @param recursiveMatchBits Per-pair recursive bits, or nullptr
 * @param maxTargetAtoms Stride for recursiveMatchBits indexing
 * @param batchLocalIndices Optional remapping for split launches (nullptr = identity)
 * @param stream CUDA stream
 */
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
                             cudaStream_t              stream);

/**
 * @brief Launch label matrix kernel for recursive pattern preprocessing.
 *
 * Block indexing: blockIdx.x = localTargetIdx * numPatterns + localPatternIdx
 *
 * @param targets Target molecules
 * @param patterns Pattern molecules (recursive subpatterns)
 * @param patternEntries Per-pattern metadata
 * @param numPatterns Number of patterns
 * @param numBlocks Total blocks to launch
 * @param numQueries Number of main queries
 * @param miniBatchPairOffset Global pair index where the current mini-batch starts
 * @param miniBatchSize Number of pairs in the current mini-batch
 * @param labelMatrixBuffer Output label matrices
 * @param firstTargetIdx First target index for block offset calculation
 * @param recursiveMatchBits Per-pair recursive bits (for nested patterns)
 * @param maxTargetAtoms Stride for recursiveMatchBits indexing
 * @param zeroBuffers Buffers to zero on first launch (nullopt to skip)
 * @param stream CUDA stream
 */
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
                                  cudaStream_t                   stream);

/**
 * @brief Launch substructure matching kernel with template configuration dispatch.
 *
 * Selects the appropriate templated kernel based on the config parameter.
 * Uses smaller label matrices and data structures for improved performance
 * when actual molecule sizes are smaller than maximums.
 *
 * @param config Template configuration specifying MaxTargetAtoms, MaxQueryAtoms, MaxBondsPerAtom
 * @param algorithm Algorithm to use (VF2 or GSI)
 * @param targets Target molecules
 * @param queries Query molecules
 * @param miniBatchResults Device view for results
 * @param pairIndices Global pair indices
 * @param numPairs Number of pairs to process
 * @param numQueries Total number of queries
 * @param batchLocalIndices Optional remapping for split launches (nullptr = identity)
 * @param timings Optional device timings (nullptr if not collecting)
 * @param stream CUDA stream
 */
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
                                cudaStream_t                  stream);

/**
 * @brief Launch paint mode kernel for recursive SMARTS preprocessing.
 *
 * Instead of storing match mappings, directly paints recursive match bits
 * into the output buffer.
 *
 * @param algorithm Algorithm to use (GSI only currently)
 * @param targets Target molecules
 * @param patterns Pattern molecules (recursive subpatterns)
 * @param patternEntries Per-pattern metadata array (null for single-pattern mode)
 * @param numPatterns Number of patterns
 * @param numBlocks Total blocks to launch
 * @param outputRecursiveBits Buffer to paint bits into
 * @param maxTargetAtoms Stride for recursiveBits indexing
 * @param outputNumQueries Number of queries in the output (main query) results
 * @param defaultPatternId Bit position (used when patternEntries is null)
 * @param defaultMainQueryIdx Main query index (used when patternEntries is null)
 * @param miniBatchPairOffset Global pair index where the current mini-batch starts
 * @param miniBatchSize Number of pairs in the current mini-batch
 * @param overflowA First overflow buffer
 * @param overflowB Second overflow buffer (for ping-pong)
 * @param overflowCapacity Entries per overflow buffer
 * @param labelMatrixBuffer Pre-computed label matrices
 * @param firstTargetIdx First target index for block offset calculation
 * @param stream CUDA stream
 */
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
                                cudaStream_t               stream);

/**
 * @brief Configure all substruct kernels for maximum shared memory carveout.
 *
 * Should be called once before first kernel launch. Safe to call multiple times
 * (subsequent calls are no-ops).
 */
void configureSubstructKernelsSharedMem();

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_KERNELS_H
