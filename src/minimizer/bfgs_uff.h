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

#ifndef NVMOLKIT_BFGS_UFF_H
#define NVMOLKIT_BFGS_UFF_H

#include <cstdint>
#include <vector>

#include "src/hardware_options.h"
#include "src/minimizer/bfgs_minimize.h"
#include "src/forcefields/forcefield_constraints.h"

namespace RDKit {
class ROMol;
}

namespace nvMolKit::UFF {

std::vector<std::vector<double>> UFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>& mols,
                                                               int                         maxIters,
                                                               const std::vector<double>&  vdwThresholds,
                                                               const std::vector<bool>&    ignoreInterfragInteractions,
                                                               const BatchHardwareOptions& perfOptions = {});

//! \brief Result from constraint-aware UFF minimization.
struct UFFMinimizeResult {
  std::vector<std::vector<double>> energies;   //!< Per-molecule, per-conformer final energies.
  std::vector<std::vector<int8_t>> converged;  //!< Per-molecule, per-conformer convergence flags (1 = converged).
};

//! \brief Optimize with per-molecule constraints and return convergence status.
//! \param mols The molecules to optimize (positions written back in-place).
//! \param maxIters Maximum BFGS iterations.
//! \param gradTol Gradient convergence tolerance.
//! \param vdwThresholds Per-molecule VDW cutoff distances.
//! \param ignoreInterfragInteractions Per-molecule interfragment interaction flags.
//! \param constraints Per-molecule constraint specifications (empty = no constraints).
//! \param perfOptions Hardware and batching configuration.
//! \return Energies and per-system convergence flags.
UFFMinimizeResult UFFMinimizeMoleculesConfs(
  std::vector<RDKit::ROMol*>&                                  mols,
  int                                                          maxIters                    = 200,
  double                                                       gradTol                     = 1e-4,
  const std::vector<double>&                                   vdwThresholds               = {},
  const std::vector<bool>&                                     ignoreInterfragInteractions = {},
  const std::vector<ForceFieldConstraints::PerMolConstraints>& constraints                 = {},
  const BatchHardwareOptions&                                  perfOptions                 = {});

}  // namespace nvMolKit::UFF

#endif  // NVMOLKIT_BFGS_UFF_H
