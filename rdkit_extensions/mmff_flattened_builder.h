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

#ifndef NVMOLKIT_MMFF_FLATTENED_BUILDER_H
#define NVMOLKIT_MMFF_FLATTENED_BUILDER_H

#include <mutex>

#include "src/forcefields/mmff.h"
#include "src/forcefields/mmff_properties.h"

namespace RDKit {
class ROMol;
namespace MMFF {
class MMFFMolProperties;

}  // namespace MMFF
}  // namespace RDKit

namespace nvMolKit::MMFF {

//! Construct flattened MMFF forcefield contribs for a molecule.
//! Uses RDKit parametrization
/*!

  \param mol       the molecule to use
  \param nonBondedThresh  the threshold to be used in adding non-bonded terms
                          to the force field. Any non-bonded contact whose current
                          distance is greater than \c nonBondedThresh * the minimum
                          value for that contact will not be included.
  \param confId    the optional conformer id, if this isn't provided, the
                   molecule's default confId will be used.
  \param ignoreInterfragInteractions if true, nonbonded terms will not be added between fragments

  \return the flattened force field
*/
EnergyForceContribsHost constructForcefieldContribs(RDKit::ROMol& mol,
                                                    double        nonBondedThresh             = 100.0,
                                                    int           confId                      = -1,
                                                    bool          ignoreInterfragInteractions = true);
//! \overload
EnergyForceContribsHost constructForcefieldContribs(const RDKit::ROMol&             mol,
                                                    RDKit::MMFF::MMFFMolProperties* mmffMolProperties,
                                                    double                          nonBondedThresh             = 100.0,
                                                    int                             confId                      = -1,
                                                    bool                            ignoreInterfragInteractions = true);

EnergyForceContribsHost constructForcefieldContribs(RDKit::ROMol&               mol,
                                                    const nvMolKit::MMFFProperties& props,
                                                    int                             confId = -1);

}  // namespace nvMolKit::MMFF

#endif  // NVMOLKIT_MMFF_FLATTENED_BUILDER_H
