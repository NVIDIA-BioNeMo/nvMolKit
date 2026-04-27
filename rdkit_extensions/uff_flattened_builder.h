#ifndef NVMOLKIT_UFF_FLATTENED_BUILDER_H
#define NVMOLKIT_UFF_FLATTENED_BUILDER_H

#include "src/forcefields/uff.h"

namespace RDKit {
class ROMol;
}

namespace nvMolKit::UFF {

//! Construct flattened UFF forcefield contribs for a molecule.
EnergyForceContribsHost constructForcefieldContribs(RDKit::ROMol& mol,
                                                    double        vdwThresh                  = 100.0,
                                                    int           confId                     = -1,
                                                    bool          ignoreInterfragInteractions = true);

}  // namespace nvMolKit::UFF

#endif  // NVMOLKIT_UFF_FLATTENED_BUILDER_H
