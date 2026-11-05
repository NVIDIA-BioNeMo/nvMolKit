// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_TOPOLOGY_BOUNDS_COMPAT_H
#define NVMOLKIT_TOPOLOGY_BOUNDS_COMPAT_H

#include <DistGeom/BoundsMatrix.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <GraphMol/ROMol.h>

#include "versions.h"

namespace nvMolKit::detail {

//! Set topology bounds the way RDKit's embedder does for the RDKit version being built against.
//! When etkdgDetails is given, the bonds and angles used by the ETKDG force fields are collected, and on
//! RDKit >= 2026.09 so are the 1-4 path configurations that RDKit turns into forced amide/ester torsions.
inline void setEmbedderTopolBounds(const RDKit::ROMol&                         mol,
                                   const ::DistGeom::BoundsMatPtr&             mmat,
                                   const RDKit::DGeomHelpers::EmbedParameters& params,
                                   ForceFields::CrystalFF::CrystalFFDetails*   etkdgDetails,
                                   const bool                                  scaleVDW,
                                   const bool                                  set15bounds) {
#if RDKIT_ETKDG_2026_09_API
  // Embedding rejects embedForceField=MMFF (validateSupportedEmbedParameters), so UFF matches RDKit there.
  if (etkdgDetails != nullptr) {
    RDKit::DGeomHelpers::setTopolBounds(mol,
                                        mmat,
                                        etkdgDetails->bonds,
                                        etkdgDetails->angles,
                                        params,
                                        scaleVDW,
                                        set15bounds,
                                        true,
                                        true,
                                        &etkdgDetails->path14Configs,
                                        RDKit::DGeomHelpers::EmbedFF::UFF);
  } else {
    RDKit::DGeomHelpers::setTopolBounds(mol,
                                        mmat,
                                        params,
                                        scaleVDW,
                                        set15bounds,
                                        true,
                                        true,
                                        nullptr,
                                        RDKit::DGeomHelpers::EmbedFF::UFF);
  }
#else
  if (etkdgDetails != nullptr) {
    RDKit::DGeomHelpers::setTopolBounds(mol,
                                        mmat,
                                        etkdgDetails->bonds,
                                        etkdgDetails->angles,
                                        set15bounds,
                                        scaleVDW,
                                        params.useMacrocycle14config,
                                        params.forceTransAmides);
  } else {
    RDKit::DGeomHelpers::setTopolBounds(mol,
                                        mmat,
                                        set15bounds,
                                        scaleVDW,
                                        params.useMacrocycle14config,
                                        params.forceTransAmides);
  }
#endif
}

}  // namespace nvMolKit::detail

#endif  // NVMOLKIT_TOPOLOGY_BOUNDS_COMPAT_H
