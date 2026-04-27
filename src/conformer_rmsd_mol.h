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

#ifndef NVMOLKIT_CONFORMER_RMSD_MOL_H
#define NVMOLKIT_CONFORMER_RMSD_MOL_H

#include <GraphMol/GraphMol.h>

#include <vector>

#include "src/conformer_rmsd.h"

namespace nvMolKit {

/**
 * @brief Compute the pairwise RMSD matrix for all conformers of a molecule.
 *
 * Extracts atom coordinates from the RDKit molecule, transfers them to the GPU,
 * and dispatches conformerRmsdMatrixGpu.  Device allocation is initiated before
 * the host coordinate buffer is filled so that GPU memory allocation overlaps
 * with the CPU extraction work.
 *
 * @param mol       RDKit molecule with two or more conformers.
 * @param prealigned If true, skip Kabsch alignment.
 * @param stream    CUDA stream.
 * @return Device buffer of N*(N-1)/2 doubles in lower-triangle condensed order.
 *         Returns an empty (size 0) buffer if mol has fewer than 2 conformers.
 * @throws std::invalid_argument if the molecule has conformers but no atoms.
 * @throws std::overflow_error   if the number of pairs exceeds INT_MAX.
 */
AsyncDeviceVector<double> conformerRmsdMatrixMol(const RDKit::ROMol& mol,
                                                 bool                prealigned = false,
                                                 cudaStream_t        stream     = nullptr);

/**
 * @brief Compute pairwise RMSD matrices for a batch of molecules on GPU.
 *
 * All molecules are processed in a single kernel launch.  Device allocation for
 * all buffers is initiated before host coordinate packing so that GPU memory
 * allocation overlaps with CPU work.
 *
 * @param mols      Non-null RDKit molecule pointers.
 * @param prealigned If true, skip Kabsch alignment.
 * @param stream    CUDA stream.
 * @return Per-molecule device buffers in the same order as mols.
 *         Buffer m holds N_m*(N_m-1)/2 doubles; size 0 if mol m has < 2 conformers.
 * @throws std::invalid_argument if any pointer is null or any molecule has conformers
 *                               but no atoms.
 * @throws std::overflow_error   if cumulative pair count exceeds INT_MAX.
 */
std::vector<AsyncDeviceVector<double>> conformerRmsdBatchMatrixMol(const std::vector<const RDKit::ROMol*>& mols,
                                                                   bool         prealigned = false,
                                                                   cudaStream_t stream     = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_CONFORMER_RMSD_MOL_H
