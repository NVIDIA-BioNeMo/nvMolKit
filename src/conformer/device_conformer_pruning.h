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

#ifndef NVMOLKIT_DEVICE_CONFORMER_PRUNING_H
#define NVMOLKIT_DEVICE_CONFORMER_PRUNING_H

#include <cstdint>
#include <vector>

#include "src/conformer/device_coord_result.h"

namespace RDKit {
class ROMol;
namespace DGeomHelpers {
struct EmbedParameters;
}  // namespace DGeomHelpers
}  // namespace RDKit

namespace nvMolKit {
namespace detail {

inline constexpr int kNumCoordinateDimensions = 3;

/**
 * @brief Locations of one molecule's conformers and atom maps.
 */
struct ConformerPruningMolInfo {
  int confBegin       = 0;  // Where this molecule begins in the grouped conformer IDs.
  int confCount       = 0;  // Number of conformers for this molecule.
  int atomMapBegin    = 0;  // Where this molecule begins in the atom maps.
  int atomMapCount    = 0;  // Number of atom maps for this molecule.
  int mappedAtomCount = 0;  // Number of atoms in each map.
};

/**
 * @brief Select conformers on the GPU using RDKit's ordered greedy RMSD rule.
 *
 * One CUDA block handles each molecule. It considers conformers in input order and
 * compares each one with earlier retained conformers in parallel.
 *
 * @param coords         Coordinates for every input conformer, stored on the GPU.
 * @param atomStarts     Where each conformer's atoms begin in @p coords.
 * @param groupedConfIds Conformer IDs grouped by molecule; compacted in place as pruning scratch space.
 * @param molInfos       Locations of each molecule's conformers and atom maps.
 * @param atomMaps       Atom orders RDKit uses for heavy-atom and symmetry comparisons.
 * @param selected       One result per grouped conformer; 1 means keep it and 0 means discard it.
 * @param threshold      RMSD values below this threshold are considered duplicates.
 * @param stream         CUDA stream used for the pruning work.
 */
void conformerPruneMaskGpu(cuda::std::span<const double>                  coords,
                           cuda::std::span<const int32_t>                 atomStarts,
                           cuda::std::span<int32_t>                       groupedConfIds,
                           cuda::std::span<const ConformerPruningMolInfo> molInfos,
                           cuda::std::span<const int32_t>                 atomMaps,
                           cuda::std::span<uint8_t>                       selected,
                           double                                         threshold,
                           cudaStream_t                                   stream);

/**
 * @brief Copy retained conformers into a contiguous device buffer.
 *
 * @param positions           Coordinates before pruning, stored on the GPU.
 * @param atomStarts          Where each source conformer's atoms begin in @p positions.
 * @param sourceConformerIds  Original ID of each retained conformer.
 * @param compactedAtomStarts Where each retained conformer's atoms begin in @p compactedPositions.
 * @param compactedPositions  Output coordinates containing only retained conformers.
 * @param stream              CUDA stream used for the copy.
 */
void compactConformerPositionsGpu(cuda::std::span<const double>  positions,
                                  cuda::std::span<const int32_t> atomStarts,
                                  cuda::std::span<const int32_t> sourceConformerIds,
                                  cuda::std::span<const int32_t> compactedAtomStarts,
                                  cuda::std::span<double>        compactedPositions,
                                  cudaStream_t                   stream);

/**
 * @brief Prune a finalized ETKDG device result using RDKit-compatible RMSD comparisons.
 *
 * Coordinates remain on the GPU identified by @c result.gpuId. Small index arrays and pruning
 * decisions are copied to the CPU to group conformers and allocate the output.
 *
 * @param result Coordinates and molecule IDs produced by ETKDG.
 * @param mols   Input molecules in the same order used for embedding.
 * @param params RDKit settings for the RMSD threshold, heavy atoms, and symmetry.
 * @return Retained conformers in input order, renumbered within each molecule.
 *
 * @note This function synchronizes the default stream before returning.
 */
DeviceCoordResult pruneDeviceConformers(DeviceCoordResult                           result,
                                        const std::vector<RDKit::ROMol*>&           mols,
                                        const RDKit::DGeomHelpers::EmbedParameters& params);

}  // namespace detail
}  // namespace nvMolKit

#endif  // NVMOLKIT_DEVICE_CONFORMER_PRUNING_H
