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

#ifndef NVMOLKIT_TFD_KERNELS_H
#define NVMOLKIT_TFD_KERNELS_H

#include <cuda_runtime.h>

#include <cstdint>

#include "src/tfd/tfd_types.h"

namespace nvMolKit {

//! Block size for TFD kernels
constexpr int kTFDBlockSize = 256;

//! Launch kernel to compute dihedral angles for all conformers.
//! One thread per (conformer, quartet) work item; uses binary search on
//! dihedralWorkStarts to find the molecule, then computes indices arithmetically.
void launchDihedralKernel(int                  totalWorkItems,
                          const float*         positions,
                          const int*           confPositionStarts,
                          const int*           torsionAtoms,
                          const MolDescriptor* molDescriptors,
                          const int*           dihedralWorkStarts,
                          int                  numMolecules,
                          float*               dihedralAngles,
                          cudaStream_t         stream);

//! Launch kernel to compute TFD matrix for all conformer pairs.
//! One block per molecule; threads within a block cooperatively process pairs.
//! Uses shared memory for torsion metadata to reduce global reads.
void launchTFDMatrixKernel(int                  numMolecules,
                           const float*         dihedralAngles,
                           const float*         torsionWeights,
                           const float*         torsionMaxDevs,
                           const int*           quartetStarts,
                           const uint8_t*       torsionTypes,
                           const MolDescriptor* molDescriptors,
                           float*               tfdOutput,
                           cudaStream_t         stream);

}  // namespace nvMolKit

#endif  // NVMOLKIT_TFD_KERNELS_H
