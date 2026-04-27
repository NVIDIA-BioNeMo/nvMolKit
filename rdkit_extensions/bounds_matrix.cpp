#include "rdkit_extensions/bounds_matrix.h"

#include <DistGeom/TriangleSmooth.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <GraphMol/ROMol.h>
#include <Numerics/SymmMatrix.h>

namespace RDKit::DGeomHelpers {

// TODO: Coordmap support.
bool setupInitialBoundsMatrix(const ROMol*                              mol,
                              const DistGeom::BoundsMatPtr&             mmat,
                              const EmbedParameters&                    params,
                              ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails) {
  PRECONDITION(mol, "bad molecule");
  if (params.useExpTorsionAnglePrefs || params.useBasicKnowledge) {
    setTopolBounds(*mol,
                   mmat,
                   etkdgDetails.bonds,
                   etkdgDetails.angles,
                   true,
                   false,
                   params.useMacrocycle14config,
                   params.forceTransAmides);
  } else {
    setTopolBounds(*mol, mmat, true, false, params.useMacrocycle14config, params.forceTransAmides);
  }

  if (!DistGeom::triangleSmoothBounds(mmat)) {
    // ok this bound matrix failed to triangle smooth - re-compute the
    // bounds matrix without 15 bounds and with VDW scaling
    initBoundsMat(mmat);
    setTopolBounds(*mol, mmat, false, true, params.useMacrocycle14config, params.forceTransAmides);

    // try triangle smoothing again
    if (!DistGeom::triangleSmoothBounds(mmat)) {
      // ok, we're not going to be able to smooth this,
      if (params.ignoreSmoothingFailures) {
        // proceed anyway with the more relaxed bounds matrix
        initBoundsMat(mmat);
        setTopolBounds(*mol, mmat, false, true, params.useMacrocycle14config, params.forceTransAmides);
      } else {
        BOOST_LOG(rdWarningLog) << "Could not triangle bounds smooth molecule.\n";
        return false;
      }
    }
  }
  return true;
}

void initETKDG(ROMol* mol, const EmbedParameters& params, ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails) {
  PRECONDITION(mol, "bad molecule");
  const unsigned int nAtoms = mol->getNumAtoms();
  if (params.useExpTorsionAnglePrefs || params.useBasicKnowledge) {
    ForceFields::CrystalFF::getExperimentalTorsions(*mol,
                                                    etkdgDetails,
                                                    params.useExpTorsionAnglePrefs,
                                                    params.useSmallRingTorsions,
                                                    params.useMacrocycleTorsions,
                                                    params.useBasicKnowledge,
                                                    params.ETversion,
                                                    params.verbose);
    etkdgDetails.atomNums.resize(nAtoms);
    for (unsigned int i = 0; i < nAtoms; ++i) {
      etkdgDetails.atomNums[i] = mol->getAtomWithIdx(i)->getAtomicNum();
    }
  }
  etkdgDetails.boundsMatForceScaling = params.boundsMatForceScaling;
}

// turn off linting for RDKit port.
// NOLINTBEGIN
RDNumeric::SymmMatrix<double> initialCoordsNormDistances(const RDNumeric::SymmMatrix<double>& initialDistMat) {
  constexpr double EIGVAL_TOL = 0.001;
  const int        N          = initialDistMat.numRows();

  RDNumeric::SymmMatrix<double> sqMat(N), T(N, 0.0);

  double* sqDat = sqMat.getData();

  int           dSize   = initialDistMat.getDataSize();
  const double* data    = initialDistMat.getData();
  double        sumSqD2 = 0.0;
  for (int i = 0; i < dSize; i++) {
    sqDat[i] = data[i] * data[i];
    sumSqD2 += sqDat[i];
  }
  sumSqD2 /= (N * N);

  RDNumeric::DoubleVector sqD0i(N, 0.0);
  double*                 sqD0iData = sqD0i.getData();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sqD0iData[i] += sqMat.getVal(i, j);
    }
    sqD0iData[i] /= N;
    sqD0iData[i] -= sumSqD2;

    if ((sqD0iData[i] < EIGVAL_TOL) && (N > 3)) {
      sqD0iData[i] = 10 * EIGVAL_TOL;
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      double val = 0.5 * (sqD0iData[i] + sqD0iData[j] - sqMat.getVal(i, j));
      T.setVal(i, j, val);
    }
  }
  return T;
}
// NOLINTEND

}  // namespace RDKit::DGeomHelpers

namespace nvMolKit {

std::vector<DistGeom::BoundsMatPtr> getBoundsMatrices(
  const std::vector<const RDKit::ROMol*>&                mols,
  const RDKit::DGeomHelpers::EmbedParameters&            params,
  std::vector<ForceFields::CrystalFF::CrystalFFDetails>& etkdgDetails) {
  std::vector<DistGeom::BoundsMatPtr> boundsMatrices;
  for (size_t idx = 0; idx < mols.size(); ++idx) {
    const auto*                                     mol = mols[idx];
    const boost::shared_ptr<DistGeom::BoundsMatrix> boundsMatrix(new DistGeom::BoundsMatrix(mol->getNumAtoms()));
    // TODO: configure min/max
    RDKit::DGeomHelpers::initBoundsMat(boundsMatrix.get());
    if (!RDKit::DGeomHelpers::setupInitialBoundsMatrix(mol, boundsMatrix, params, etkdgDetails[idx])) {
      throw std::runtime_error("Could not setup the bounds matrix");
    }
    boundsMatrices.push_back(boundsMatrix);
  }
  return boundsMatrices;
}

}  // namespace nvMolKit