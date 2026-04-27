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

#ifndef NVMOLKIT_BFGS_HESSIAN_H
#define NVMOLKIT_BFGS_HESSIAN_H

#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"

namespace nvMolKit {

// Update the inverse Hessian matrix using BFGS formula for a batch of systems
void updateInverseHessianBFGSBatch(int            numActiveSystems,
                                   const int16_t* statuses,
                                   const int*     hessianStarts,
                                   const int*     atomStarts,
                                   double*        invHessians,
                                   double*        dGrads,
                                   double*        xis,
                                   double*        hessDGrads,
                                   const double*  grads,
                                   int            dataDim,
                                   bool           hasLargeMolecule,
                                   const int*     activeSystemIndices,
                                   cudaStream_t   stream = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_BFGS_HESSIAN_H
