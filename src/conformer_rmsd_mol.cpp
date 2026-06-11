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

#include "src/conformer_rmsd_mol.h"

#include <GraphMol/Conformer.h>

#include <climits>
#include <limits>
#include <stdexcept>
#include <string>

#include "src/utils/host_vector.h"

namespace nvMolKit {

AsyncDeviceVector<double> conformerRmsdMatrixMol(const RDKit::ROMol& mol, const bool prealigned, cudaStream_t stream) {
  const int numConfs = mol.getNumConformers();
  if (numConfs <= 1) {
    return AsyncDeviceVector<double>(0);
  }

  const int numAtoms = mol.getNumAtoms();
  if (numAtoms == 0) {
    // Intentional divergence from RDKit, which returns [nan] for exactly 2
    // zero-atom conformers and raises ZeroDivisionError for 3+. We fail fast
    // with a consistent error for all degenerate zero-atom inputs.
    throw std::invalid_argument("Molecule has no atoms");
  }

  const size_t  numCoords  = static_cast<size_t>(numConfs) * numAtoms * 3;
  const int64_t numPairs64 = static_cast<int64_t>(numConfs) * (numConfs - 1) / 2;
  if (numPairs64 > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Number of conformer pairs exceeds maximum kernel grid size");
  }
  const int numPairs = static_cast<int>(numPairs64);

  // Allocate device buffers before filling the host buffer so that the async
  // GPU memory allocation can proceed while the CPU extracts coordinates.
  AsyncDeviceVector<double> devCoords(numCoords, stream);
  AsyncDeviceVector<double> devRmsd(numPairs, stream);

  // Extract coordinates into a flat pinned host buffer.
  // Layout: coords[conf * numAtoms * 3 + atom * 3 + xyz]
  // Pinned memory allows the DMA engine to transfer directly without a staging
  // copy; the destructor handles cleanup safely after all stream work is submitted.
  PinnedHostVector<double> hostCoords(numCoords);
  int                      confIdx = 0;
  for (auto it = mol.beginConformers(); it != mol.endConformers(); ++it, ++confIdx) {
    const RDKit::Conformer& conf = **it;
    for (int a = 0; a < numAtoms; ++a) {
      const auto& pos                                = conf.getAtomPos(a);
      hostCoords[confIdx * numAtoms * 3 + a * 3 + 0] = pos.x;
      hostCoords[confIdx * numAtoms * 3 + a * 3 + 1] = pos.y;
      hostCoords[confIdx * numAtoms * 3 + a * 3 + 2] = pos.z;
    }
  }

  hostCoords.copyToDevice(devCoords, stream);
  conformerRmsdMatrixGpu(toSpan(devCoords), toSpan(devRmsd), numConfs, numAtoms, prealigned, stream);
  return devRmsd;
}

std::vector<AsyncDeviceVector<double>> conformerRmsdBatchMatrixMol(const std::vector<const RDKit::ROMol*>& mols,
                                                                   const bool                              prealigned,
                                                                   cudaStream_t                            stream) {
  const int numMols = static_cast<int>(mols.size());
  if (numMols == 0)
    return {};

  // --- Validate inputs and compute per-molecule metadata ---
  // pairOffsets and totalPairs are intentionally 32-bit: the kernel launches one
  // block per pair on the x-dimension, whose hardware limit is 2^31-1 (INT_MAX).
  // The int64_t accumulation below detects prefix-sum overflow before the value
  // materializes in that type; an unchecked overflow would silently route blocks
  // to the wrong molecule or write out of bounds.
  std::vector<int>    numConfsVec(numMols), numAtomsVec(numMols);
  std::vector<int>    pairOffsetsVec(numMols + 1);
  std::vector<size_t> coordOffsetsVec(numMols);

  pairOffsetsVec[0]  = 0;
  size_t totalCoords = 0;
  for (int m = 0; m < numMols; ++m) {
    if (!mols[m]) {
      throw std::invalid_argument("Null molecule at index " + std::to_string(m));
    }
    const int nc = mols[m]->getNumConformers();
    const int na = mols[m]->getNumAtoms();
    if (na == 0 && nc >= 2) {
      throw std::invalid_argument("Molecule at index " + std::to_string(m) + " has no atoms");
    }
    numConfsVec[m]     = nc;
    numAtomsVec[m]     = na;
    coordOffsetsVec[m] = totalCoords;
    totalCoords += static_cast<size_t>(nc) * na * 3;

    const int64_t numPairs64 = nc >= 2 ? static_cast<int64_t>(nc) * (nc - 1) / 2 : 0;
    const int64_t newOffset  = static_cast<int64_t>(pairOffsetsVec[m]) + numPairs64;
    if (newOffset > static_cast<int64_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("Cumulative conformer pairs exceed int range by molecule index " + std::to_string(m));
    }
    pairOffsetsVec[m + 1] = static_cast<int>(newOffset);
  }
  const int totalPairs = pairOffsetsVec[numMols];

  // --- Allocate device buffers first so GPU allocation overlaps CPU work below ---
  AsyncDeviceVector<double> devCoords(totalCoords > 0 ? totalCoords : 1, stream);
  AsyncDeviceVector<int>    devNumConfs(numMols, stream);
  AsyncDeviceVector<int>    devNumAtoms(numMols, stream);
  AsyncDeviceVector<int>    devPairOffsets(numMols + 1, stream);
  AsyncDeviceVector<size_t> devCoordOffsets(numMols, stream);

  // Per-molecule output buffers.  Always allocate at least 1 element so that
  // devRmsdPtrs never contains a null — zero-pair molecules dispatch 0 blocks
  // and their slot is never written, but a null would be a latent hazard.
  std::vector<AsyncDeviceVector<double>> devRmsdVecs;
  devRmsdVecs.reserve(numMols);
  for (int m = 0; m < numMols; ++m) {
    const int numPairs = pairOffsetsVec[m + 1] - pairOffsetsVec[m];
    devRmsdVecs.emplace_back(numPairs > 0 ? numPairs : 1, stream);
  }
  AsyncDeviceVector<double*> devRmsdPtrs(numMols, stream);

  // --- Fill pinned host buffers (CPU work, overlaps with async device allocs) ---
  PinnedHostVector<double>  hostCoords(totalCoords > 0 ? totalCoords : 1);
  PinnedHostVector<int>     numConfsArr(numMols);
  PinnedHostVector<int>     numAtomsArr(numMols);
  PinnedHostVector<int>     pairOffsetsArr(numMols + 1);
  PinnedHostVector<size_t>  coordOffsetsArr(numMols);
  PinnedHostVector<double*> hostRmsdPtrs(numMols);

  for (int m = 0; m < numMols; ++m) {
    numConfsArr[m]     = numConfsVec[m];
    numAtomsArr[m]     = numAtomsVec[m];
    pairOffsetsArr[m]  = pairOffsetsVec[m];
    coordOffsetsArr[m] = coordOffsetsVec[m];
    hostRmsdPtrs[m]    = devRmsdVecs[m].data();

    const int na      = numAtomsVec[m];
    int       confIdx = 0;
    for (auto it = mols[m]->beginConformers(); it != mols[m]->endConformers(); ++it, ++confIdx) {
      const RDKit::Conformer& conf = **it;
      for (int a = 0; a < na; ++a) {
        const auto&  pos     = conf.getAtomPos(a);
        const size_t base    = coordOffsetsVec[m] + static_cast<size_t>(confIdx) * na * 3 + static_cast<size_t>(a) * 3;
        hostCoords[base + 0] = pos.x;
        hostCoords[base + 1] = pos.y;
        hostCoords[base + 2] = pos.z;
      }
    }
  }
  pairOffsetsArr[numMols] = pairOffsetsVec[numMols];

  // --- Transfer to device and launch ---
  if (totalCoords > 0)
    hostCoords.copyToDevice(devCoords, stream);
  numConfsArr.copyToDevice(devNumConfs, stream);
  numAtomsArr.copyToDevice(devNumAtoms, stream);
  pairOffsetsArr.copyToDevice(devPairOffsets, stream);
  coordOffsetsArr.copyToDevice(devCoordOffsets, stream);
  hostRmsdPtrs.copyToDevice(devRmsdPtrs, stream);

  if (totalPairs > 0) {
    conformerRmsdBatchMatrixGpu(toSpan(devCoords),
                                toSpan(devRmsdPtrs),
                                toSpan(devPairOffsets),
                                toSpan(devCoordOffsets),
                                toSpan(devNumConfs),
                                toSpan(devNumAtoms),
                                numMols,
                                totalPairs,
                                prealigned,
                                stream);
  }

  return devRmsdVecs;
}

}  // namespace nvMolKit
