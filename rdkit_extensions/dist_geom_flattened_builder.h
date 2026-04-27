#ifndef NVMOLKIT_DISTGEOM_FLATTENED_BUILDER_H
#define NVMOLKIT_DISTGEOM_FLATTENED_BUILDER_H

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/ChiralSet.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>

#include <map>

#include "src/forcefields/dist_geom.h"
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

}  // namespace DistGeom
}  // namespace nvMolKit

#endif  // NVMOLKIT_DISTGEOM_FLATTENED_BUILDER_H