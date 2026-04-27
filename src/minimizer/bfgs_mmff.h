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

#ifndef NVMOLKIT_BFGS_MMFF_H
#define NVMOLKIT_BFGS_MMFF_H

#include <vector>

#include "src/hardware_options.h"
#include "src/minimizer/bfgs_minimize.h"
#include "src/forcefields/forcefield_constraints.h"
#include "src/forcefields/mmff_properties.h"

namespace RDKit {
class ROMol;
}

namespace nvMolKit::MMFF {

//! \brief Optimize conformers for multiple molecules using MMFF force field
//! \param mols The molecules to optimize
//! \param maxIters The maximum number of iterations to perform
//! \param nonBondedThreshold The radius threshold for non-bonded interactions
//! \param perfOptions Performance tuning options (threading, batching)
//! \param backend The BFGS backend to use (BATCHED, PER_MOLECULE, or HYBRID which auto-selects)
//! \return A vector of vectors of energies, where each inner vector contains energies for conformers of one molecule
std::vector<std::vector<double>> MMFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>& mols,
                                                                int                         maxIters    = 200,
                                                                const MMFFProperties&       properties  = {},
                                                                const BatchHardwareOptions& perfOptions = {},
                                                                BfgsBackend backend = BfgsBackend::HYBRID);

std::vector<std::vector<double>> MMFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>&        mols,
                                                                int                                maxIters,
                                                                const std::vector<MMFFProperties>& properties,
                                                                const BatchHardwareOptions&        perfOptions = {},
                                                                BfgsBackend backend = BfgsBackend::HYBRID);

//! \brief Result from constraint-aware MMFF minimization.
struct MMFFMinimizeResult {
  std::vector<std::vector<double>> energies;   //!< Per-molecule, per-conformer final energies.
  std::vector<std::vector<int8_t>> converged;  //!< Per-molecule, per-conformer convergence flags (1 = converged).
};

//! \brief Optimize with per-molecule constraints and return convergence status.
//! \param mols The molecules to optimize (positions written back in-place).
//! \param maxIters Maximum BFGS iterations.
//! \param gradTol Gradient convergence tolerance.
//! \param properties Per-molecule MMFF settings.
//! \param constraints Per-molecule constraint specifications.
//! \param perfOptions Hardware and batching configuration.
//! \param backend BFGS backend selection.
//! \return Energies and per-system convergence flags.
MMFFMinimizeResult MMFFMinimizeMoleculesConfs(
  std::vector<RDKit::ROMol*>&                                  mols,
  int                                                          maxIters    = 200,
  double                                                       gradTol     = 1e-4,
  const std::vector<MMFFProperties>&                           properties  = {},
  const std::vector<ForceFieldConstraints::PerMolConstraints>& constraints = {},
  const BatchHardwareOptions&                                  perfOptions = {},
  BfgsBackend                                                  backend     = BfgsBackend::HYBRID);

}  // namespace nvMolKit::MMFF
#endif  // NVMOLKIT_BFGS_MMFF_H
