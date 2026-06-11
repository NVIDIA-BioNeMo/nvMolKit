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

#ifndef NVMOLKIT_CONFORMER_RMSD_H
#define NVMOLKIT_CONFORMER_RMSD_H

#include "src/utils/device_vector.h"

namespace nvMolKit {

/**
 * @brief Compute pairwise RMSD between all conformers, returning a condensed lower-triangle matrix.
 *
 * For N conformers with M atoms each, computes N*(N-1)/2 pairwise RMSD values.
 * When prealigned is false, each pair is optimally aligned using the Kabsch algorithm
 * (closed-form SVD of the 3x3 cross-covariance matrix) before computing RMSD.
 *
 * The output format matches RDKit's GetConformerRMSMatrix: a flat array of N*(N-1)/2
 * doubles in lower-triangle order, where element index for pair (i, j) with i > j is:
 *   index = i*(i-1)/2 + j
 *
 * @param coords Flattened coordinate array of shape (numConformers * numAtoms * 3).
 *               Layout: coords[conf * numAtoms * 3 + atom * 3 + xyz].
 * @param rmsdOut Output array of size numConformers*(numConformers-1)/2.
 * @param numConformers Number of conformers.
 * @param numAtoms Number of atoms per conformer.
 * @param prealigned If true, skip Kabsch alignment and compute RMSD directly.
 *                   If false (default), optimally align each pair before RMSD.
 * @param stream CUDA stream to execute operations on.
 */
void conformerRmsdMatrixGpu(cuda::std::span<const double> coords,
                            cuda::std::span<double>       rmsdOut,
                            int                           numConformers,
                            int                           numAtoms,
                            bool                          prealigned = false,
                            cudaStream_t                  stream     = nullptr);

/**
 * @brief Compute pairwise RMSD matrices for a batch of molecules on GPU.
 *
 * All molecules are processed in a single kernel launch, so their pairs execute
 * concurrently.  Each molecule writes to its own pre-allocated output buffer.
 * Molecules with fewer than 2 conformers contribute no blocks and their output
 * buffers should have size 0.
 *
 * Coordinate layout for molecule m:
 *   coords[coordOffsets[m] + conf * numAtomsPerMol[m] * 3 + atom * 3 + xyz]
 *
 * @param coords          Flat coordinate array for all molecules.
 * @param rmsdOutputs     Device pointers to per-molecule output buffers.
 *                        Buffer m must hold pairOffsets[m+1]-pairOffsets[m] doubles.
 * @param pairOffsets     Prefix-sum of per-molecule pair counts, size numMols+1.
 * @param coordOffsets    Start of each molecule's data in coords[], in units of double.
 * @param numConfsPerMol  Number of conformers per molecule.
 * @param numAtomsPerMol  Number of atoms per molecule.
 * @param numMols         Number of molecules.
 * @param prealigned      If true, skip Kabsch alignment.
 * @param stream          CUDA stream.
 */
void conformerRmsdBatchMatrixGpu(cuda::std::span<const double> coords,
                                 cuda::std::span<double*>      rmsdOutputs,
                                 cuda::std::span<const int>    pairOffsets,
                                 cuda::std::span<const size_t> coordOffsets,
                                 cuda::std::span<const int>    numConfsPerMol,
                                 cuda::std::span<const int>    numAtomsPerMol,
                                 int                           numMols,
                                 int                           totalPairs,
                                 bool                          prealigned,
                                 cudaStream_t                  stream);

}  // namespace nvMolKit

#endif  // NVMOLKIT_CONFORMER_RMSD_H
