#include "rdkit_extensions/mmff_contribs.h"

#include <ForceField/ForceField.h>
#include <GraphMol/ROMol.h>
#include <RDGeneral/Invariant.h>

#include <vector>

#include "src/forcefields/ff_utils.h"
#include "src/forcefields/mmff.h"
#include "rdkit_extensions/mmff_flattened_builder.h"

namespace nvMolKit {
namespace MMFF {
MMFFGPUContrib::MMFFGPUContrib(ForceFields::ForceField* owner,
                               RDKit::ROMol&            mol,
                               double                   nonBondedThresh,
                               int                      confId,
                               bool                     ignoreInterfragInteractions) {
  PRECONDITION(owner, "bad owner");

  dp_forceField = owner;

  std::vector<double> positions;
  confPosToVect(mol, positions, confId);

  auto ffParams =
    nvMolKit::MMFF::constructForcefieldContribs(mol, nonBondedThresh, confId, ignoreInterfragInteractions);
  nvMolKit::MMFF::addMoleculeToBatch(ffParams, positions, systemHost_);
}

double MMFFGPUContrib::getEnergy(double* pos) const {
  PRECONDITION(dp_forceField, "no owner");
  PRECONDITION(pos, "bad vector");

  // TODO: Consider adding this as a class member.
  nvMolKit::MMFF::BatchedMolecularDeviceBuffers systemDevice;

  const unsigned int dim  = dp_forceField->dimension();
  const unsigned int num  = dp_forceField->positions().size();
  const unsigned int size = dim * num;

  nvMolKit::MMFF::sendContribsAndIndicesToDevice(systemHost_, systemDevice);

  systemDevice.positions.setFromArray(pos, size);

  nvMolKit::MMFF::allocateIntermediateBuffers(systemHost_, systemDevice);
  nvMolKit::MMFF::computeEnergy(systemDevice);

  double hostEnergy = 0.0;
  systemDevice.energyOuts.copyToHost(&hostEnergy, 1);
  cudaDeviceSynchronize();

  return hostEnergy;
}

void MMFFGPUContrib::getGrad(double* pos, double* grad) const {
  PRECONDITION(dp_forceField, "no owner");
  PRECONDITION(pos, "bad vector");
  PRECONDITION(grad, "bad vector");

  nvMolKit::MMFF::BatchedMolecularDeviceBuffers systemDevice;

  const unsigned int dim  = dp_forceField->dimension();
  const unsigned int num  = dp_forceField->positions().size();
  const unsigned int size = dim * num;

  nvMolKit::MMFF::sendContribsAndIndicesToDevice(systemHost_, systemDevice);

  systemDevice.positions.setFromArray(pos, size);

  systemDevice.grad.resize(systemHost_.positions.size());
  systemDevice.grad.zero();
  nvMolKit::MMFF::computeGradients(systemDevice);

  systemDevice.grad.copyToHost(grad, systemHost_.positions.size());
  cudaDeviceSynchronize();
}
}  // namespace MMFF
}  // namespace nvMolKit
