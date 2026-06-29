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

#ifndef NVMOLKIT_FIRE_MINIMIZE_PERMOL_KERNELS_H
#define NVMOLKIT_FIRE_MINIMIZE_PERMOL_KERNELS_H

#include <cuda_runtime.h>

#include <cstdint>

#include "src/forcefields/mmff_kernels.h"
#include "src/minimizer/fire_minimizer.h"

namespace nvMolKit {

//! \brief Per-block launch configuration for the per-molecule FIRE kernels.
//!
//! The caller supplies per-system state buffers and force-field term descriptors. The
//! kernel iterates the full FIRE 2.0 loop internally and writes a per-system
//! status (0 = converged, 1 = active) into @p statuses.
struct FirePerMolLaunchParams {
  int    numIters         = 0;      //!< Maximum FIRE iterations to run inside the kernel.
  double gradTol          = 0.0;    //!< sqrt(sum(grad^2)) per-system convergence tolerance.
  bool   takeHalfStepBack = true;   //!< When true and power<0, take a half step back and zero v.
  bool   useAbc           = false;  //!< Apply ABC-FIRE mixer correction.
  bool   useMass          = false;  //!< Mass-weight the force kick (requires non-null masses).
};

//! \brief Launch per-molecule FIRE 2.0 minimization - MMFF specialization.
//! \note ::FireOptions stuck-detection fields are not supported on the per-mol path.
//! \note Returns cudaErrorNotSupported when @p hasConstraints is true.
cudaError_t launchFirePerMolKernel(int                                       numMols,
                                   const int*                                molIds,
                                   int                                       maxAtoms,
                                   const int*                                atomStarts,
                                   const FireOptions&                        fireOptions,
                                   int                                       numIters,
                                   double                                    gradTol,
                                   const MMFF::EnergyForceContribsDevicePtr& terms,
                                   const MMFF::BatchedIndicesDevicePtr&      systemIndices,
                                   bool                                      hasConstraints,
                                   double*                                   positions,
                                   double*                                   grad,
                                   double*                                   velocities,
                                   double*                                   alphas,
                                   double*                                   dts,
                                   int*                                      nStepsPositive,
                                   const double*                             masses,
                                   double*                                   energyOuts,
                                   uint8_t*                                  statuses,
                                   cudaStream_t                              stream = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_FIRE_MINIMIZE_PERMOL_KERNELS_H
