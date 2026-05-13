// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_BFGS_MINIMIZE_PERMOL_KERNELS_H
#define NVMOLKIT_BFGS_MINIMIZE_PERMOL_KERNELS_H

#include <cuda_runtime.h>

#include "bfgs_types.h"
#include "dist_geom_kernels.h"
#include "mmff_kernels.h"

namespace nvMolKit {

/// Launch per-molecule BFGS minimization kernel - MMFF specialization.
/// `hasConstraints` selects between two specializations of the kernel: when false, the
/// distance/position/angle/torsion constraint loops are compiled out, which lowers register
/// pressure and improves occupancy on the common no-constraint path.
cudaError_t launchBfgsMinimizePerMolKernel(int                                       numMols,
                                           const int*                                molIds,
                                           int                                       maxAtoms,
                                           const int*                                atomStarts,
                                           const int*                                hessianStarts,
                                           int                                       numIters,
                                           double                                    gradTol,
                                           bool                                      scaleGrads,
                                           const MMFF::EnergyForceContribsDevicePtr& terms,
                                           const MMFF::BatchedIndicesDevicePtr&      systemIndices,
                                           double*                                   positions,
                                           double*                                   grad,
                                           double*                                   inverseHessian,
                                           double**                                  scratchBuffers,
                                           double*                                   energyOuts,
                                           bool                                      hasConstraints,
                                           int16_t*                                  statuses = nullptr,
                                           cudaStream_t                              stream   = nullptr);

/// Launch per-molecule BFGS minimization kernel - ETK specialization
cudaError_t launchBfgsMinimizePerMolKernelETK(int                                             numMols,
                                              const int*                                      molIds,
                                              int                                             maxAtoms,
                                              const int*                                      atomStarts,
                                              const int*                                      hessianStarts,
                                              int                                             numIters,
                                              double                                          gradTol,
                                              bool                                            scaleGrads,
                                              const DistGeom::Energy3DForceContribsDevicePtr& terms,
                                              const DistGeom::BatchedIndices3DDevicePtr&      systemIndices,
                                              double*                                         positions,
                                              double*                                         grad,
                                              double*                                         inverseHessian,
                                              double**                                        scratchBuffers,
                                              double*                                         energyOuts,
                                              int16_t*                                        statuses = nullptr,
                                              cudaStream_t                                    stream   = nullptr);

/// Launch per-molecule BFGS minimization kernel - DG  specialization
cudaError_t launchBfgsMinimizePerMolKernelDG(int                                           numMols,
                                             const int*                                    molIds,
                                             int                                           maxAtoms,
                                             const int*                                    atomStarts,
                                             const int*                                    hessianStarts,
                                             int                                           numIters,
                                             double                                        gradTol,
                                             bool                                          scaleGrads,
                                             const DistGeom::EnergyForceContribsDevicePtr& terms,
                                             const DistGeom::BatchedIndicesDevicePtr&      systemIndices,
                                             double*                                       positions,
                                             double*                                       grad,
                                             double*                                       inverseHessian,
                                             double**                                      scratchBuffers,
                                             double*                                       energyOuts,
                                             double                                        chiralWeight,
                                             double                                        fourthDimWeight,
                                             int16_t*                                      statuses = nullptr,
                                             cudaStream_t                                  stream   = nullptr);
}  // namespace nvMolKit

#endif  // NVMOLKIT_BFGS_MINIMIZE_PERMOL_KERNELS_H
