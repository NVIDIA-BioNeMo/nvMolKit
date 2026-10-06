// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_TOPOLOGY_CUH
#define NVMOLKIT_DESCRIPTORS3D_TOPOLOGY_CUH

#include <algorithm>
#include <cstdint>
#include <cub/device/device_scan.cuh>

#include "src/descriptors3d.h"
#include "src/descriptors3d_kernel.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

namespace nvMolKit::descriptors3d_detail {

//! Largest bond-count distance the table records; AUTOCORR3D reads up to 10, GETAWAY up to 8.
constexpr int    kBondTableDepth       = 10;
//! Upper bound on the bond-distance table; molecules past it fall back to per-row searches.
constexpr size_t kBondTableBudgetBytes = size_t{256} << 20;
//! Largest molecule whose per-row search arrays (one per group of a block) fit the default shared memory.
constexpr int    kBondTableMaxAtoms    = (48 << 10) / (kWarpsPerBlock * kGroupsPerWarp);

/**
 * @brief Per-molecule bond-count distances, shared by every conformer of a molecule and by AUTOCORR3D and
 *        GETAWAY. Molecule m's pairs (j < k) are stored row-major from `starts[m]`, one byte each: the
 *        distance up to kBondTableDepth, or kUnreachedDepth beyond it or between fragments. Molecules whose
 *        range ends past @ref capacity have no table entries; their kernels search per row instead.
 */
struct BondDistanceTable {
  const uint8_t* distances = nullptr;
  const int64_t* starts    = nullptr;
  int64_t        capacity  = 0;
};

//! Table entries of molecule @p moleculeIdx, or null when it has none.
__device__ __forceinline__ const uint8_t* moleculeBondDistances(const BondDistanceTable& table, const int moleculeIdx) {
  if (table.distances == nullptr || table.starts[moleculeIdx + 1] > table.capacity) {
    return nullptr;
  }
  return table.distances + table.starts[moleculeIdx];
}

//! Offset of pair (@p j, @p j + 1) within a molecule's table entries for @p numAtoms atoms.
__device__ __forceinline__ int64_t bondTableRowStart(const int j, const int numAtoms) {
  return static_cast<int64_t>(j) * numAtoms - static_cast<int64_t>(j) * (j + 1) / 2;
}

//! Pair count n (n - 1) / 2 of each molecule into `pairCounts[m + 1]`, and 0 into `pairCounts[0]`.
__global__ void moleculePairCountsKernel(const int32_t* moleculeAtomStarts, const int nMols, int64_t* pairCounts) {
  const int moleculeIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (moleculeIdx == 0) {
    pairCounts[0] = 0;
  }
  if (moleculeIdx < nMols) {
    const int64_t numAtoms      = moleculeAtomStarts[moleculeIdx + 1] - moleculeAtomStarts[moleculeIdx];
    pairCounts[moleculeIdx + 1] = numAtoms * (numAtoms - 1) / 2;
  }
}

/**
 * @brief Fills the table entries of every molecule that fits @p table's capacity; one warp per molecule.
 *        Each group searches interleaved rows j with searchBondDepths() into its own shared-memory row of
 *        @p maxAtoms bytes and writes the row's pairs (j, k > j).
 */
__global__ void bondDistanceTableKernel(const int32_t* moleculeAtomStarts,
                                        const int32_t* bondNeighborStarts,
                                        const int32_t* bondNeighbors,
                                        const int      nMols,
                                        const int      maxAtoms,
                                        const int64_t* starts,
                                        const int64_t  capacity,
                                        uint8_t* __restrict__ distances) {
  extern __shared__ uint8_t searchRows[];
  const int                 lane        = static_cast<int>(threadIdx.x) % kWarpSize;
  const int                 laneInGroup = lane % kGroupSize;
  const int                 group       = lane / kGroupSize;
  const int                 warpInBlock = static_cast<int>(threadIdx.x) / kWarpSize;
  const int                 moleculeIdx = blockIdx.x * kWarpsPerBlock + warpInBlock;
  if (moleculeIdx >= nMols || starts[moleculeIdx + 1] > capacity) {
    return;  // Uniform across the warp.
  }
  const int      atomStart = moleculeAtomStarts[moleculeIdx];
  const int      numAtoms  = moleculeAtomStarts[moleculeIdx + 1] - atomStart;
  uint8_t* const depth     = searchRows + (warpInBlock * kGroupsPerWarp + group) * maxAtoms;
  uint8_t* const entries   = distances + starts[moleculeIdx];
  const unsigned groupMask = 0xffu << (group * kGroupSize);
  for (int j = group; j < numAtoms - 1; j += kGroupsPerWarp) {
    searchBondDepths<kBondTableDepth>(bondNeighborStarts + atomStart,
                                      bondNeighbors,
                                      numAtoms,
                                      j,
                                      laneInGroup,
                                      groupMask,
                                      depth);
    uint8_t* const row = entries + bondTableRowStart(j, numAtoms);
    for (int k = j + 1 + laneInGroup; k < numAtoms; k += kGroupSize) {
      row[k - j - 1] = depth[k];
    }
  }
}

//! Owns a BondDistanceTable's device storage.
struct BondDistanceTableStorage {
  AsyncDeviceVector<int64_t> starts;
  AsyncDeviceVector<uint8_t> distances;
  BondDistanceTable          table;
};

/**
 * @brief Builds the bond-distance table for the batch's molecules on @p stream, without host
 *        synchronization. Capacity is the smaller of kBondTableBudgetBytes and the worst case for
 *        `inputs.maxMoleculeAtoms`; batches whose largest molecule exceeds kBondTableMaxAtoms (or whose
 *        maxMoleculeAtoms is unset) get an empty table, so every conformer searches per row.
 */
inline void buildBondDistanceTable(const DeviceCoordView&        coordinates,
                                   const Property3DDeviceInputs& inputs,
                                   BondDistanceTableStorage&     storage,
                                   const cudaStream_t            stream) {
  const int nMols    = coordinates.nMols;
  const int maxAtoms = inputs.maxMoleculeAtoms;
  if (nMols == 0 || maxAtoms < 2 || maxAtoms > kBondTableMaxAtoms) {
    return;
  }
  const int64_t maxPairs = static_cast<int64_t>(maxAtoms) * (maxAtoms - 1) / 2;
  const int64_t capacity = std::min<int64_t>(kBondTableBudgetBytes, maxPairs * nMols);

  storage.starts = AsyncDeviceVector<int64_t>(static_cast<size_t>(nMols) + 1, stream);
  moleculePairCountsKernel<<<(nMols + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(inputs.moleculeAtomStarts,
                                                                                             nMols,
                                                                                             storage.starts.data());
  cudaCheckError(cudaGetLastError());
  size_t tempBytes = 0;
  cudaCheckError(cub::DeviceScan::InclusiveSum(nullptr,
                                               tempBytes,
                                               storage.starts.data() + 1,
                                               storage.starts.data() + 1,
                                               nMols,
                                               stream));
  AsyncDeviceVector<uint8_t> temp(tempBytes, stream);
  cudaCheckError(cub::DeviceScan::InclusiveSum(temp.data(),
                                               tempBytes,
                                               storage.starts.data() + 1,
                                               storage.starts.data() + 1,
                                               nMols,
                                               stream));

  storage.distances        = AsyncDeviceVector<uint8_t>(static_cast<size_t>(capacity), stream);
  const int    numBlocks   = (nMols + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const size_t sharedBytes = static_cast<size_t>(kWarpsPerBlock) * kGroupsPerWarp * maxAtoms;
  bondDistanceTableKernel<<<numBlocks, kBlockSize, sharedBytes, stream>>>(inputs.moleculeAtomStarts,
                                                                          inputs.bondNeighborStarts,
                                                                          inputs.bondNeighbors,
                                                                          nMols,
                                                                          maxAtoms,
                                                                          storage.starts.data(),
                                                                          capacity,
                                                                          storage.distances.data());
  cudaCheckError(cudaGetLastError());
  storage.table = {storage.distances.data(), storage.starts.data(), capacity};
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_TOPOLOGY_CUH
