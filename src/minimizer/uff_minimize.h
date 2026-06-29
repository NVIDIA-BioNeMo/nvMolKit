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

#ifndef NVMOLKIT_UFF_MINIMIZE_H
#define NVMOLKIT_UFF_MINIMIZE_H

#include <cstdint>
#include <optional>
#include <vector>

#include "src/conformer/device_coord_result.h"
#include "src/forcefields/forcefield_constraints.h"
#include "src/hardware_options.h"
#include "src/minimizer/bfgs_minimize.h"
#include "src/minimizer/fire_minimizer.h"

namespace RDKit {
class ROMol;
}

namespace nvMolKit::UFF {

std::vector<std::vector<double>> UFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>& mols,
                                                               int                         maxIters,
                                                               const std::vector<double>&  vdwThresholds,
                                                               const std::vector<bool>&    ignoreInterfragInteractions,
                                                               const BatchHardwareOptions& perfOptions = {});

std::vector<std::vector<double>> UFFOptimizeMoleculesConfsFire(std::vector<RDKit::ROMol*>& mols,
                                                               int                         maxIters,
                                                               const FireOptions&          fireOptions,
                                                               const std::vector<double>&  vdwThresholds,
                                                               const std::vector<bool>&    ignoreInterfragInteractions,
                                                               const BatchHardwareOptions& perfOptions = {});

//! \brief Result from constraint-aware UFF minimization.
//!
//! In CoordinateOutput::RDKIT_CONFORMERS mode, @ref energies and @ref converged are populated and
//! @ref device is empty; coordinates are written back into each input molecule's RDKit conformer
//! list. In CoordinateOutput::DEVICE mode, @ref device holds the on-GPU coordinates / energies /
//! convergence flags collected onto the chosen target GPU and the host-side vectors are empty.
struct UFFMinimizeResult {
  std::vector<std::vector<double>> energies;   //!< Per-molecule, per-conformer final energies (RDKIT mode only).
  std::vector<std::vector<int8_t>> converged;  //!< Per-molecule, per-conformer convergence flags (RDKIT mode only).
  std::optional<DeviceCoordResult> device;     //!< Populated when output==DEVICE.
};

//! \brief Optimize with per-molecule constraints and return convergence status.
//! \param mols The molecules to optimize (positions written back in-place in RDKIT_CONFORMERS mode).
//!             When @p deviceInput is provided, each mol still needs at least one RDKit
//!             conformer; that conformer is only consulted for force-field construction (e.g.
//!             special-case hybridization checks) - the actual starting coordinates come from
//!             @p deviceInput.
//! \param maxIters Maximum BFGS iterations.
//! \param gradTol Gradient convergence tolerance.
//! \param vdwThresholds Per-molecule VDW cutoff distances.
//! \param ignoreInterfragInteractions Per-molecule interfragment interaction flags.
//! \param constraints Per-molecule constraint specifications (empty = no constraints).
//!                    Constraints read anchor positions from each mol's RDKit conformer at
//!                    force-field construction time, so they are incompatible with @p deviceInput
//!                    (whose positions never reach the host before contribs are built). To apply
//!                    constraints, use the RDKit Mol + Conformer path (omit @p deviceInput) so
//!                    anchor positions match the optimizer's starting coordinates.
//! \param perfOptions Hardware and batching configuration.
//! \param output Whether to write coordinates back into RDKit conformers (default) or return
//!               them on-device as a DeviceCoordResult.
//! \param targetGpu In DEVICE mode, the GPU to consolidate the result onto. -1 selects the first
//!                  configured execution GPU (or device 0).
//! \param deviceInput Optional on-device starting coordinates. When supplied, the conformer count
//!                    and (molIdx, confIdx) labels must exactly match the host-side flattening of
//!                    @p mols (one entry per RDKit conformer, in input-mol order, then conformer
//!                    order). Positions are broadcast from @p deviceInput onto each executing GPU.
//! \return Either host-side energies/convergence (RDKIT mode) or a populated `device` field (DEVICE mode).
UFFMinimizeResult UFFMinimizeMoleculesConfs(
  std::vector<RDKit::ROMol*>&                                  mols,
  int                                                          maxIters                    = 200,
  double                                                       gradTol                     = 1e-4,
  const std::vector<double>&                                   vdwThresholds               = {},
  const std::vector<bool>&                                     ignoreInterfragInteractions = {},
  const std::vector<ForceFieldConstraints::PerMolConstraints>& constraints                 = {},
  const BatchHardwareOptions&                                  perfOptions                 = {},
  CoordinateOutput                                             output      = CoordinateOutput::RDKIT_CONFORMERS,
  int                                                          targetGpu   = -1,
  const DeviceCoordResult*                                     deviceInput = nullptr);

//! \brief Minimize UFF energies with FIRE 2.0 and report per-conformer convergence.
//! \param mols Molecules whose conformers provide the initial coordinates. In
//!             RDKIT_CONFORMERS mode, optimized coordinates are written back in place.
//! \param maxIters Maximum FIRE iterations.
//! \param fireOptions FIRE integration and convergence settings. A conformer converges when
//!                    its gradient norm satisfies @ref FireOptions::gradTol. The UFF wrapper
//!                    uses the batched backend and can additionally use energy-plateau detection.
//! \param vdwThresholds Per-molecule VDW cutoff distances.
//! \param ignoreInterfragInteractions Per-molecule interfragment interaction flags.
//! \param constraints Per-molecule constraint specifications (empty = no constraints).
//! \param perfOptions Hardware and batching configuration.
//! \param output Whether to write coordinates back into RDKit conformers (default) or return
//!               them on-device as a DeviceCoordResult.
//! \param targetGpu In DEVICE mode, the GPU to consolidate the result onto. -1 selects the first
//!                  configured execution GPU (or device 0).
//! \return Final energies and convergence flags in RDKIT_CONFORMERS mode, or a device-resident
//!         result in DEVICE mode. A convergence flag is false when @p maxIters is reached first.
UFFMinimizeResult UFFMinimizeMoleculesConfsFire(
  std::vector<RDKit::ROMol*>&                                  mols,
  int                                                          maxIters,
  const FireOptions&                                           fireOptions,
  const std::vector<double>&                                   vdwThresholds,
  const std::vector<bool>&                                     ignoreInterfragInteractions,
  const std::vector<ForceFieldConstraints::PerMolConstraints>& constraints = {},
  const BatchHardwareOptions&                                  perfOptions = {},
  CoordinateOutput                                             output      = CoordinateOutput::RDKIT_CONFORMERS,
  int                                                          targetGpu   = -1);

}  // namespace nvMolKit::UFF

#endif  // NVMOLKIT_UFF_MINIMIZE_H
