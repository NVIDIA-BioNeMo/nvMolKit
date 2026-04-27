#ifndef __NVMOLKIT_MMFFCONTRIBS_H__
#define __NVMOLKIT_MMFFCONTRIBS_H__

#include <ForceField/Contrib.h>
#include <ForceField/ForceField.h>
#include <GraphMol/ROMol.h>
#include <RDGeneral/export.h>

#include "src/forcefields/mmff.h"

namespace nvMolKit {
namespace MMFF {

class RDKIT_FORCEFIELD_EXPORT MMFFGPUContrib : public ForceFields::ForceFieldContrib {
 public:
  MMFFGPUContrib() = default;

  MMFFGPUContrib(ForceFields::ForceField* owner,
                 RDKit::ROMol&            mol,
                 double                   nonBondedThresh             = 100.0,
                 int                      confId                      = -1,
                 bool                     ignoreInterfragInteractions = true);
  ~MMFFGPUContrib() override = default;
  double getEnergy(double* pos) const override;

  void            getGrad(double* pos, double* grad) const override;
  MMFFGPUContrib* copy() const override { return new MMFFGPUContrib(*this); }

 private:
  nvMolKit::MMFF::BatchedMolecularSystemHost systemHost_;
};
}  // namespace MMFF
}  // namespace nvMolKit
#endif
