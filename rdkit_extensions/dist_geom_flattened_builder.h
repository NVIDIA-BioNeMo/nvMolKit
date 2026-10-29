#ifndef NVMOLKIT_DISTGEOM_FLATTENED_BUILDER_H
#define NVMOLKIT_DISTGEOM_FLATTENED_BUILDER_H

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/ChiralSet.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>

#include <map>

#include "src/forcefields/dist_geom.h"
#include "versions.h"
namespace nvMolKit {
namespace DistGeom {

nvMolKit::DistGeom::EnergyForceContribsHost constructForceFieldContribs(
  const int                              dim,
  const ::DistGeom::BoundsMatrix&        mmat,
  const ::DistGeom::VECT_CHIRALSET&      csets,
  double                                 weightChiral    = 1.0,
  double                                 weightFourthDim = 0.1,
  std::map<std::pair<int, int>, double>* extraWeights    = nullptr,
  double                                 basinSizeTol    = 5.0);

nvMolKit::DistGeom::Energy3DForceContribsHost construct3DForceFieldContribs(
  const ::DistGeom::BoundsMatrix&                   mmat,
  const ::ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
  const std::vector<double>&                        positions,
  int                                               dim,
  bool                                              useBasicKnowledge = true);

//! Builds the terms of RDKit's DistGeom::construct3DImproperForceField: improper torsion terms with force scaling 10
//! and 179-180 degree angle terms for linear centers. The angle force constant, also 10, is left to the caller.
nvMolKit::DistGeom::Energy3DForceContribsHost constructImproperCheckContribs(
  const ::ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
  unsigned int                                      numAtoms);

#if RDKIT_AIO_ETKDG_API
//! Builds one molecule's terms of RDKit's all-in-one ETKDG force field, as DistGeom::constructAllInOneForceField
//! followed by DistGeom::addTorsionTerms. Force constants come from etkdgDetails.forceConsts, the 1-2/1-3 pairs from
//! etkdgDetails.distMat, and the experimental torsion constants in etkdgDetails must already carry RDKit's all-in-one
//! scaling.
nvMolKit::DistGeom::AllInOneForceContribsHost constructAllInOneForceFieldContribs(
  int                                               dim,
  const ::DistGeom::BoundsMatrix&                   mmat,
  const ::DistGeom::VECT_CHIRALSET&                 csets,
  const ::ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
  bool                                              useExpTorsionAnglePrefs,
  bool                                              useBasicKnowledge);

//! The batch-wide force constants that go with constructAllInOneForceFieldContribs for the same etkdgDetails.
nvMolKit::DistGeom::AllInOneForceWeights allInOneForceWeights(
  const ::ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails);
#endif

}  // namespace DistGeom
}  // namespace nvMolKit

#endif  // NVMOLKIT_DISTGEOM_FLATTENED_BUILDER_H