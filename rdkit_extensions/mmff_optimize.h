#ifndef NVMOLKIT_MMFF_OPTIMIZE_H
#define NVMOLKIT_MMFF_OPTIMIZE_H

#include <ForceField/ForceField.h>
#include <GraphMol/ForceFieldHelpers/FFConvenience.h>
#include <GraphMol/ROMol.h>

#include "rdkit_extensions/mmff_builder.h"

namespace nvMolKit {
namespace MMFF {
inline std::pair<int, double> MMFFOptimizeMolecule(RDKit::ROMol& mol,
                                                   int           maxIters                    = 1000,
                                                   double        nonBondedThresh             = 10.0,
                                                   int           confId                      = -1,
                                                   bool          ignoreInterfragInteractions = true) {
  std::pair<int, double>                   res = std::make_pair(-1, -1);
  std::unique_ptr<ForceFields::ForceField> ff(
    nvMolKit::MMFF::constructForceField(mol, nonBondedThresh, confId, ignoreInterfragInteractions));
  res = RDKit::ForceFieldsHelper::OptimizeMolecule(*ff, maxIters);
  return res;
}

inline void MMFFOptimizeMoleculeConfs(RDKit::ROMol&                        mol,
                                      std::vector<std::pair<int, double>>& res,
                                      int                                  numThreads                  = 1,
                                      int                                  maxIters                    = 1000,
                                      double                               nonBondedThresh             = 10.0,
                                      bool                                 ignoreInterfragInteractions = true) {
  std::unique_ptr<ForceFields::ForceField> ff(
    nvMolKit::MMFF::constructForceField(mol, nonBondedThresh, -1, ignoreInterfragInteractions));
  RDKit::ForceFieldsHelper::OptimizeMoleculeConfs(mol, *ff, res, numThreads, maxIters);
}
}  // namespace MMFF
}  // namespace nvMolKit
#endif  // NVMOLKIT_MMFF_OPTIMIZE_H