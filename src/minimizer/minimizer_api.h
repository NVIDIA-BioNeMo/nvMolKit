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

#ifndef NVMOLKIT_MINIMIZER_API_H
#define NVMOLKIT_MINIMIZER_API_H

#include <functional>
#include <vector>

#include "src/utils/device_vector.h"

namespace nvMolKit {

//! Callback that computes system energies. When positions is nullptr, implementations should use internal buffers.
using EnergyFunctor = std::function<void(const double*)>;
//! Callback that computes gradients and writes them into the gradient buffer managed by the minimizer.
using GradFunctor   = std::function<void()>;

//! Interface for batched minimizers operating on device-managed coordinate arrays.
class BatchMinimizer {
 public:
  virtual ~BatchMinimizer() = default;

  //! Perform a minimization pass on the current batch.
  //! @param numIters Maximum number of iterations to run.
  //! @param gradTol Convergence threshold for gradient magnitudes.
  //! @param atomStartsHost Offsets for atom indices per system on the host.
  //! @param atomStarts Device-side offsets matching @p atomStartsHost.
  //! @param positions Device positions buffer laid out as contiguous systems.
  //! @param grad Device gradient buffer associated with @p positions.
  //! @param energyOuts Device buffer receiving per-system energies.
  //! @param energyBuffer Scratch buffer for temporary energy storage.
  //! @param eFunc Energy functor invoked with the positions to evaluate.
  //! @param gFunc Gradient functor invoked after energy evaluations.
  //! @param activeThisStage Optional mask indicating which systems participate this iteration.
  //! @return True if all systems converged, false otherwise.
  virtual bool minimize(int                           numIters,
                        double                        gradTol,
                        const std::vector<int>&       atomStartsHost,
                        const AsyncDeviceVector<int>& atomStarts,
                        AsyncDeviceVector<double>&    positions,
                        AsyncDeviceVector<double>&    grad,
                        AsyncDeviceVector<double>&    energyOuts,
                        AsyncDeviceVector<double>&    energyBuffer,
                        EnergyFunctor                 eFunc,
                        GradFunctor                   gFunc,
                        const uint8_t*                activeThisStage = nullptr) = 0;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_MINIMIZER_API_H
