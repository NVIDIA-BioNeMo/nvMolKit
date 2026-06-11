#include "rdkit_extensions/mmff_builder.h"

#include <ForceField/ForceField.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>
#include <GraphMol/ROMol.h>

#include "src/forcefields/ff_utils.h"
#include "rdkit_extensions/mmff_contribs.h"
#include "rdkit_extensions/mmff_flattened_builder.h"

namespace nvMolKit {
namespace MMFF {
ForceFields::ForceField* constructForceField(RDKit::ROMol& mol,
                                             double        nonBondedThresh,
                                             int           confId,
                                             bool          ignoreInterfragInteractions) {
  std::unique_ptr<ForceFields::ForceField> res(new ForceFields::ForceField());
  setFFPosFromConf(mol, res.get(), confId);

  res->initialize();

  auto contrib = boost::make_shared<nvMolKit::MMFF::MMFFGPUContrib>(res.get(),
                                                                    mol,
                                                                    nonBondedThresh,
                                                                    confId,
                                                                    ignoreInterfragInteractions);
  res->contribs().emplace_back(std::move(contrib));

  return res.release();
}
}  // namespace MMFF
}  // namespace nvMolKit
