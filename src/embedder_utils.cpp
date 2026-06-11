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

/**
 * This code is adapted from RDKit's embedder.cpp to facilitate:
 * 1. Generating reference ETKDG force fields for unit testing
 * 2. Initializing and configuring the ETKDG pipeline
 * 3. Managing force field parameters and variables for the embedding process
 */
#include "src/embedder_utils.h"

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/ChiralSet.h>
#include <DistGeom/DistGeomUtils.h>
#include <DistGeom/TriangleSmooth.h>
#include <ForceField/ForceField.h>
#include <Geometry/Transform3D.h>
#include <GraphMol/Atom.h>
#include <GraphMol/AtomIterators.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <Numerics/Alignment/AlignPoints.h>
#include <RDGeneral/Exceptions.h>
#include <RDGeneral/RDLog.h>
#include <RDGeneral/RDThreads.h>
#include <RDGeneral/types.h>

#include <boost/dynamic_bitset.hpp>
#include <iomanip>
#include <typeinfo>

#include "versions.h"

using namespace RDKit;

namespace {
constexpr int    DEFAULT_ITERATIONS_PER_ATOM        = 10;  // Number of iterations per atom for embedding
constexpr double ERROR_TOL                          = 0.00001;
constexpr double MAX_MINIMIZED_E_PER_ATOM           = 0.05;
constexpr double MAX_MINIMIZED_E_CONTRIB            = 0.20;
constexpr int    DEFAULT_FIRST_MINIMIZATION_STEPS   = 400;  // Default number of steps for first minimization
constexpr int DEFAULT_FOURTH_DIM_MINIMIZATION_STEPS = 200;  // Default number of steps for fourth dimension minimization
constexpr double FIRST_MINIMIZE_CHIRAL_WEIGHT       = 1.0;  // Weight for chiral constraints in first minimization
constexpr double FIRST_MINIMIZE_FOURTH_DIM_WEIGHT   = 0.1;  // Weight for fourth dimension in first minimization
constexpr double FOURTH_DIM_MINIMIZE_CHIRAL_WEIGHT =
  0.2;  // Weight for chiral constraints in fourth dimension minimization
constexpr double FOURTH_DIM_MINIMIZE_FOURTH_DIM_WEIGHT =
  1.0;  // Weight for fourth dimension in fourth dimension minimization
}  // namespace

namespace nvMolKit {
namespace DGeomHelpers {
namespace EmbeddingOps {
bool generateInitialCoords(const std::vector<std::unique_ptr<RDGeom::Point>>& positions,
                           const detail::EmbedArgs&                           eargs,
                           const RDKit::DGeomHelpers::EmbedParameters&        embedParams,
                           RDNumeric::DoubleSymmMatrix&                       distMat,
                           RDKit::double_source_type*                         rng) {
  bool gotCoords = false;

  // Convert unique_ptr vector to PointPtrVect for RDKit functions
  RDGeom::PointPtrVect tempPositions;
  tempPositions.reserve(positions.size());
  for (const auto& pos : positions) {
    tempPositions.push_back(pos.get());
  }

  if (!embedParams.useRandomCoords) {
    const double largestDistance = DistGeom::pickRandomDistMat(*eargs.mmat, distMat, *rng);
    RDUNUSED_PARAM(largestDistance);
    gotCoords =
      DistGeom::computeInitialCoords(distMat, tempPositions, *rng, embedParams.randNegEig, embedParams.numZeroFail);
  } else {
    double boxSize = 0.0;
    if (embedParams.boxSizeMult > 0) {
      constexpr double kDefaultBoxSizeMultiplier = 5.0;
      boxSize                                    = kDefaultBoxSizeMultiplier * embedParams.boxSizeMult;
    } else {
      boxSize = -1 * embedParams.boxSizeMult;
    }
    gotCoords = DistGeom::computeRandomCoords(tempPositions, boxSize, *rng);
    if (embedParams.useRandomCoords && embedParams.coordMap != nullptr) {
      for (const auto& [idx, coordPoint] : *embedParams.coordMap) {
        auto* point = positions[idx].get();
        for (unsigned int ci = 0; ci < coordPoint.dimension(); ++ci) {
          (*point)[ci] = coordPoint[ci];
        }
        // zero out any higher dimensional components:
        for (unsigned int ci = coordPoint.dimension(); ci < point->dimension(); ++ci) {
          (*point)[ci] = 0.0;
        }
      }
    }
  }
  return gotCoords;
}

// NOLINTBEGIN
void findChiralSets(const ROMol&                          mol,
                    DistGeom::VECT_CHIRALSET&             chiralCenters,
                    DistGeom::VECT_CHIRALSET&             tetrahedralCenters,
                    const std::map<int, RDGeom::Point3D>* coordMap) {
  for (const auto& atom : mol.atoms()) {
    if (atom->getAtomicNum() != 1) {  // skip hydrogens
      const Atom::ChiralType chiralType         = atom->getChiralTag();
      constexpr int          kCarbonAtomicNum   = 6;
      constexpr int          kNitrogenAtomicNum = 7;
      if ((chiralType == Atom::CHI_TETRAHEDRAL_CW || chiralType == Atom::CHI_TETRAHEDRAL_CCW) ||
          ((atom->getAtomicNum() == kCarbonAtomicNum || atom->getAtomicNum() == kNitrogenAtomicNum) &&
           atom->getDegree() == 4)) {
        // make a chiral set from the neighbors
        INT_VECT nbrs;
        nbrs.reserve(4);
        // find the neighbors of this atom and enter them into the
        // nbr list
        ROMol::OEDGE_ITER beg;
        ROMol::OEDGE_ITER end;
        boost::tie(beg, end) = mol.getAtomBonds(atom);
        while (beg != end) {
          nbrs.push_back(static_cast<int>(mol[*beg]->getOtherAtom(atom)->getIdx()));
          ++beg;
        }
        // if we have less than 4 heavy atoms as neighbors,
        // we need to include the chiral center into the mix
        // we should at least have 3 though
        CHECK_INVARIANT(nbrs.size() >= 3, "Cannot be a chiral center");

        constexpr double kDefaultVolLowerBound       = 5.0;  // Default lower bound for 4 neighbors
        constexpr double kThreeNeighborVolLowerBound = 2.0;  // Lower bound for 3 neighbors
        constexpr double volUpperBound               = 100.0;
        double           volLowerBound               = kDefaultVolLowerBound;

        if (nbrs.size() < 4) {
          // we get lower volumes if there are three neighbors,
          //  this was github #5883
          volLowerBound = kThreeNeighborVolLowerBound;
          nbrs.insert(nbrs.end(), static_cast<int>(atom->getIdx()));
        }
        // Account for API break adding d_structureFlags. Our GPU codepath always includes it, but
        // for an earlier version we pass in 1.0 to not change results. The CPU API must be adapted here and in
        // chiralty ETKDG code.
#if RDKIT_NEW_FLAG_API
        // set a flag for tetrahedral centers that are in multiple small rings
        auto          numSmallRings = 0;
        constexpr int smallRingSize = 5;
        for (const auto sz : mol.getRingInfo()->atomRingSizes(atom->getIdx())) {
          if (sz < smallRingSize) {
            ++numSmallRings;
          }
        }
        std::uint64_t structureFlags = 0;
        if (numSmallRings > 1) {
          structureFlags = static_cast<std::uint64_t>(DistGeom::ChiralSetStructureFlags::IN_FUSED_SMALL_RINGS);
        }
#endif
        // now create a chiral set and set the upper and lower bound on the
        // volume
        if (chiralType == Atom::CHI_TETRAHEDRAL_CCW) {
          // positive chiral volume
          // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
          auto* cset =
            new DistGeom::ChiralSet(atom->getIdx(), nbrs[0], nbrs[1], nbrs[2], nbrs[3], volLowerBound, volUpperBound);
          const DistGeom::ChiralSetPtr cptr(cset);
          chiralCenters.push_back(cptr);
        } else if (chiralType == Atom::CHI_TETRAHEDRAL_CW) {
          // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
          auto* cset =
            new DistGeom::ChiralSet(atom->getIdx(), nbrs[0], nbrs[1], nbrs[2], nbrs[3], -volUpperBound, -volLowerBound);
          const DistGeom::ChiralSetPtr cptr(cset);
          chiralCenters.push_back(cptr);
        } else {
          if ((coordMap != nullptr && coordMap->find(static_cast<int>(atom->getIdx())) != coordMap->end()) ||
              (mol.getRingInfo()->isInitialized() && (mol.getRingInfo()->numAtomRings(atom->getIdx()) < 2 ||
                                                      mol.getRingInfo()->isAtomInRingOfSize(atom->getIdx(), 3)))) {
            // we only want to these tests for ring atoms that are not part of
            // the coordMap
            // there's no sense doing 3-rings because those are a nightmare
          } else {
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
#if RDKIT_NEW_FLAG_API
            auto* cset =
              new DistGeom::ChiralSet(atom->getIdx(), nbrs[0], nbrs[1], nbrs[2], nbrs[3], 0.0, 0.0, structureFlags);
#else
            auto* cset = new DistGeom::ChiralSet(atom->getIdx(), nbrs[0], nbrs[1], nbrs[2], nbrs[3], 0.0, 0.0);
#endif
            const DistGeom::ChiralSetPtr cptr(cset);
            tetrahedralCenters.push_back(cptr);
          }
        }
      }
    }
  }
}
// NOLINTEND

void adjustBoundsMatFromCoordMap(const DistGeom::BoundsMatPtr& mmat, const std::map<int, RDGeom::Point3D>* coordMap) {
  for (auto iIt = coordMap->begin(); iIt != coordMap->end(); ++iIt) {
    const unsigned int     iIdx   = iIt->first;
    const RDGeom::Point3D& iPoint = iIt->second;
    auto                   jIt    = iIt;
    while (++jIt != coordMap->end()) {
      const unsigned int     jIdx   = jIt->first;
      const RDGeom::Point3D& jPoint = jIt->second;
      const double           dist   = (iPoint - jPoint).length();
      mmat->setUpperBound(iIdx, jIdx, dist);
      mmat->setLowerBound(iIdx, jIdx, dist);
    }
  }
}

void setupTopologyBounds(const ROMol*                                mol,
                         const ::DistGeom::BoundsMatPtr&             mmat,
                         const RDKit::DGeomHelpers::EmbedParameters& params,
                         ForceFields::CrystalFF::CrystalFFDetails&   etkdgDetails) {
  PRECONDITION(mol, "bad molecule");
  if (params.useExpTorsionAnglePrefs || params.useBasicKnowledge) {
    RDKit::DGeomHelpers::setTopolBounds(*mol,
                                        mmat,
                                        etkdgDetails.bonds,
                                        etkdgDetails.angles,
                                        true,
                                        false,
                                        params.useMacrocycle14config,
                                        params.forceTransAmides);
  } else {
    RDKit::DGeomHelpers::setTopolBounds(*mol, mmat, true, false, params.useMacrocycle14config, params.forceTransAmides);
  }
  // Note: coordMap handling removed as per user request
}

void setupRelaxedBounds(const ROMol*                                mol,
                        const ::DistGeom::BoundsMatPtr&             mmat,
                        const RDKit::DGeomHelpers::EmbedParameters& params) {
  // Re-compute the bounds matrix without 15 bounds and with VDW scaling
  RDKit::DGeomHelpers::initBoundsMat(mmat);
  RDKit::DGeomHelpers::setTopolBounds(*mol, mmat, false, true, params.useMacrocycle14config, params.forceTransAmides);
  // Note: coordMap handling removed as per user request
}

void setupIgnoredSmoothingBounds(const ROMol*                                mol,
                                 const ::DistGeom::BoundsMatPtr&             mmat,
                                 const RDKit::DGeomHelpers::EmbedParameters& params) {
  // Proceed with the more relaxed bounds matrix when ignoring smoothing failures
  RDKit::DGeomHelpers::initBoundsMat(mmat);
  RDKit::DGeomHelpers::setTopolBounds(*mol, mmat, false, true, params.useMacrocycle14config, params.forceTransAmides);
  // Note: coordMap handling removed as per user request
}

void initETKDG(const ROMol*                                mol,
               const RDKit::DGeomHelpers::EmbedParameters& params,
               ForceFields::CrystalFF::CrystalFFDetails&   etkdgDetails) {
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

bool setupInitialBoundsMatrix(const ROMol*                                mol,
                              const DistGeom::BoundsMatPtr&               mmat,
                              const std::map<int, RDGeom::Point3D>*       coordMap,
                              const RDKit::DGeomHelpers::EmbedParameters& params,
                              ForceFields::CrystalFF::CrystalFFDetails&   etkdgDetails) {
  PRECONDITION(mol, "bad molecule");
  if (params.useExpTorsionAnglePrefs || params.useBasicKnowledge) {
    RDKit::DGeomHelpers::setTopolBounds(*mol,
                                        mmat,
                                        etkdgDetails.bonds,
                                        etkdgDetails.angles,
                                        true,
                                        false,
                                        params.useMacrocycle14config,
                                        params.forceTransAmides);
  } else {
    RDKit::DGeomHelpers::setTopolBounds(*mol, mmat, true, false, params.useMacrocycle14config, params.forceTransAmides);
  }
  constexpr double kCoordMapTolerance = 0.05;
  double           tol                = 0.0;
  if (coordMap != nullptr) {
    adjustBoundsMatFromCoordMap(mmat, coordMap);
    tol = kCoordMapTolerance;
  }
  if (!DistGeom::triangleSmoothBounds(mmat, tol)) {
    // ok this bound matrix failed to triangle smooth - re-compute the
    // bounds matrix without 15 bounds and with VDW scaling
    RDKit::DGeomHelpers::initBoundsMat(mmat);
    RDKit::DGeomHelpers::setTopolBounds(*mol, mmat, false, true, params.useMacrocycle14config, params.forceTransAmides);

    if (coordMap != nullptr) {
      adjustBoundsMatFromCoordMap(mmat, coordMap);
    }

    // try triangle smoothing again
    if (!DistGeom::triangleSmoothBounds(mmat, tol)) {
      // ok, we're not going to be able to smooth this,
      if (params.ignoreSmoothingFailures) {
        // proceed anyway with the more relaxed bounds matrix
        RDKit::DGeomHelpers::initBoundsMat(mmat);
        RDKit::DGeomHelpers::setTopolBounds(*mol,
                                            mmat,
                                            false,
                                            true,
                                            params.useMacrocycle14config,
                                            params.forceTransAmides);

        if (coordMap != nullptr) {
          adjustBoundsMatFromCoordMap(mmat, coordMap);
        }
      } else {
        BOOST_LOG(rdWarningLog) << "Could not triangle bounds smooth molecule.\n";
        return false;
      }
    }
  }
  return true;
}

std::unique_ptr<ForceFields::ForceField> constructForceField(std::vector<std::unique_ptr<RDGeom::Point>>& positions,
                                                             const detail::EmbedArgs&                     eargs,
                                                             RDKit::DGeomHelpers::EmbedParameters&        embedParams) {
  const unsigned int nAtoms = eargs.mmat->numRows();

  if (positions.empty()) {
    positions.resize(nAtoms);

    for (unsigned int i = 0; i < nAtoms; ++i) {
      if (eargs.dim == 4) {
        positions[i] = std::make_unique<RDGeom::PointND>(4);
      } else {
        positions[i] = std::make_unique<RDGeom::Point3D>();
      }
    }

    RDNumeric::DoubleSymmMatrix distMat(positions.size(), 0.0);

    // The basin threshold just gets us into trouble when we're using
    // random coordinates since it ends up ignoring 1-4 (and higher)
    // interactions. This causes us to get folded-up (and self-penetrating)
    // conformations for large flexible molecules
    constexpr double kBasinThreshold = 1e8;
    if (embedParams.useRandomCoords) {
      embedParams.basinThresh = kBasinThreshold;
    }

    RDKit::double_source_type* rng = nullptr;
    rng                            = &RDKit::getDoubleRandomSource();

    EmbeddingOps::generateInitialCoords(positions, eargs, embedParams, distMat, rng);
  } else if (eargs.dim == 4 && positions[0]->dimension() == 3) {
    // Convert 3D positions to 4D while preserving the original 3D coordinates
    std::vector<RDGeom::Point3D*> temp;
    temp.reserve(nAtoms);
    for (unsigned int i = 0; i < nAtoms; ++i) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
      temp.push_back(static_cast<RDGeom::Point3D*>(positions[i].get()));
    }

    for (unsigned int i = 0; i < nAtoms; ++i) {
      auto newPos = std::make_unique<RDGeom::PointND>(4);
      // Copy first 3 dimensions
      for (unsigned int j = 0; j < 3; ++j) {
        (*newPos)[j] = (*temp[i])[j];
      }
      // Initialize 4th dimension to 0
      (*newPos)[3] = 0.0;

      positions[i] = std::move(newPos);
    }
  } else {
    CHECK_INVARIANT(positions.size() == nAtoms, "positions vector must be the same size as the number of atoms");
  }

  boost::dynamic_bitset<> fixedPts(positions.size());
  if (embedParams.useRandomCoords && embedParams.coordMap != nullptr) {
    for (const auto& coordPair : *embedParams.coordMap) {
      fixedPts.set(coordPair.first);
    }
  }

  double           weightChiral            = 1.0;  // Default value for first minimize
  constexpr double kDefaultFourthDimWeight = 0.1;
  double           weightFourthDim         = kDefaultFourthDimWeight;  // Default value for first minimize

  if (eargs.stage == detail::MinimizeStage::FirstMinimize) {
    weightChiral    = FIRST_MINIMIZE_CHIRAL_WEIGHT;
    weightFourthDim = FIRST_MINIMIZE_FOURTH_DIM_WEIGHT;
  } else {
    weightChiral    = FOURTH_DIM_MINIMIZE_CHIRAL_WEIGHT;
    weightFourthDim = FOURTH_DIM_MINIMIZE_FOURTH_DIM_WEIGHT;
  }

  // Convert unique_ptr vector to PointPtrVect for RDKit functions
  RDGeom::PointPtrVect tempPositions;
  tempPositions.reserve(positions.size());
  for (const auto& pos : positions) {
    tempPositions.push_back(pos.get());
  }

  std::unique_ptr<ForceFields::ForceField> field(DistGeom::constructForceField(*eargs.mmat,
                                                                               tempPositions,
                                                                               eargs.chiralCenters,
                                                                               weightChiral,
                                                                               weightFourthDim,
                                                                               nullptr,
                                                                               embedParams.basinThresh,
                                                                               &fixedPts));
  field->initialize();
  return field;
}

bool firstMinimization(const std::vector<std::unique_ptr<RDGeom::Point>>& positions,
                       const detail::EmbedArgs&                           eargs,
                       const RDKit::DGeomHelpers::EmbedParameters&        embedParams) {
  bool                    gotCoords = true;
  boost::dynamic_bitset<> fixedPts(positions.size());
  if (embedParams.useRandomCoords && embedParams.coordMap != nullptr) {
    for (const auto& coord_pair : *embedParams.coordMap) {
      fixedPts.set(coord_pair.first);
    }
  }

  // Convert unique_ptr vector to PointPtrVect for RDKit functions
  RDGeom::PointPtrVect tempPositions;
  tempPositions.reserve(positions.size());
  for (const auto& pos : positions) {
    tempPositions.push_back(pos.get());
  }

  std::unique_ptr<ForceFields::ForceField> field(DistGeom::constructForceField(*eargs.mmat,
                                                                               tempPositions,
                                                                               eargs.chiralCenters,
                                                                               FIRST_MINIMIZE_CHIRAL_WEIGHT,
                                                                               FIRST_MINIMIZE_FOURTH_DIM_WEIGHT,
                                                                               nullptr,
                                                                               embedParams.basinThresh,
                                                                               &fixedPts));
  if (embedParams.useRandomCoords && embedParams.coordMap != nullptr) {
    for (const auto& coord_pair : *embedParams.coordMap) {
      field->fixedPoints().push_back(coord_pair.first);
    }
  }
  field->initialize();
  if (field->calcEnergy() > ERROR_TOL) {
    int needMore = 1;
    while (needMore != 0) {
      needMore = field->minimize(DEFAULT_FIRST_MINIMIZATION_STEPS, embedParams.optimizerForceTol);
    }
  }
  std::vector<double> e_contribs;
  const double        local_e = field->calcEnergy(&e_contribs);

  // check that neither the energy nor any of the contributions to it are
  // too high (this is part of github #971)
  if (local_e / static_cast<double>(positions.size()) >= MAX_MINIMIZED_E_PER_ATOM ||
      (!e_contribs.empty() && *(std::max_element(e_contribs.begin(), e_contribs.end())) > MAX_MINIMIZED_E_CONTRIB)) {
    gotCoords = false;
  }
  return gotCoords;
}

bool minimizeFourthDimension(const std::vector<std::unique_ptr<RDGeom::Point>>& positions,
                             const detail::EmbedArgs&                           eargs,
                             const RDKit::DGeomHelpers::EmbedParameters&        embedParams) {
  // Convert unique_ptr vector to PointPtrVect for RDKit functions
  RDGeom::PointPtrVect tempPositions;
  tempPositions.reserve(positions.size());
  for (const auto& pos : positions) {
    tempPositions.push_back(pos.get());
  }

  std::unique_ptr<ForceFields::ForceField> field2(DistGeom::constructForceField(*eargs.mmat,
                                                                                tempPositions,
                                                                                eargs.chiralCenters,
                                                                                FOURTH_DIM_MINIMIZE_CHIRAL_WEIGHT,
                                                                                FOURTH_DIM_MINIMIZE_FOURTH_DIM_WEIGHT,
                                                                                nullptr,
                                                                                embedParams.basinThresh));
  if (embedParams.useRandomCoords && embedParams.coordMap != nullptr) {
    for (const auto& coord_pair : *embedParams.coordMap) {
      field2->fixedPoints().push_back(coord_pair.first);
    }
  }

  field2->initialize();
  // std::cerr<<"FIELD2 E: "<<field2->calcEnergy()<<std::endl;
  if (field2->calcEnergy() > ERROR_TOL) {
    int needMore = 1;
    while (needMore != 0) {
      needMore = field2->minimize(DEFAULT_FOURTH_DIM_MINIMIZATION_STEPS, embedParams.optimizerForceTol);
    }
  }
  return true;
}

bool embedPoints(const std::vector<std::unique_ptr<RDGeom::Point>>& positions,
                 const detail::EmbedArgs&                           eargs,
                 RDKit::DGeomHelpers::EmbedParameters&              embedParams) {
  if (embedParams.maxIterations == 0) {
    embedParams.maxIterations = DEFAULT_ITERATIONS_PER_ATOM * positions.size();
  }
  RDNumeric::DoubleSymmMatrix distMat(positions.size(), 0.0);

  // The basin threshold just gets us into trouble when we're using
  // random coordinates since it ends up ignoring 1-4 (and higher)
  // interactions. This causes us to get folded-up (and self-penetrating)
  // conformations for large flexible molecules
  constexpr double kBasinThreshold = 1e8;
  if (embedParams.useRandomCoords) {
    embedParams.basinThresh = kBasinThreshold;
  }

  RDKit::double_source_type* rng = nullptr;
  rng                            = &RDKit::getDoubleRandomSource();

  bool         gotCoords = false;
  unsigned int iter      = 0;
  while (!gotCoords && iter < embedParams.maxIterations) {
    ++iter;
    gotCoords = EmbeddingOps::generateInitialCoords(positions, eargs, embedParams, distMat, rng);
    if (gotCoords) {
      gotCoords = EmbeddingOps::firstMinimization(positions, eargs, embedParams);

      // redo the minimization if we have a chiral center
      // or have started from random coords.
      if (gotCoords && (!eargs.chiralCenters.empty() || embedParams.useRandomCoords)) {
        gotCoords = EmbeddingOps::minimizeFourthDimension(positions, eargs, embedParams);
      }
    }
  }  // while
  return gotCoords;
}

bool processConformers(ROMol&                                             mol,
                       detail::ConformerData&                             conformerData,
                       const std::vector<std::unique_ptr<RDGeom::Point>>& positions) {
  bool               gotCoords = false;
  const unsigned int nAtoms    = positions.size();

  for (unsigned int ci = 0; ci < conformerData.confs.size(); ++ci) {
    if (conformerData.confsOk[ci]) {
      auto& conf = conformerData.confs[ci];
      // Set atom positions
      for (unsigned int i = 0; i < nAtoms; ++i) {
        conf->setAtomPos(i, RDGeom::Point3D((*positions[i])[0], (*positions[i])[1], (*positions[i])[2]));
      }
      // Add conformer to molecule
      mol.addConformer(conf.release(), true);
      gotCoords = true;
    }
  }
  return gotCoords;
}

void embedHelper(std::vector<std::unique_ptr<RDGeom::Point>>& positions,
                 detail::EmbedArgs&                           eargs,
                 RDKit::DGeomHelpers::EmbedParameters&        embedParams,
                 detail::ConformerData&                       conformerData) {
  PRECONDITION(&eargs, "bogus eargs");
  PRECONDITION(&embedParams, "bogus params");
  const unsigned int nAtoms = eargs.mmat->numRows();

  // Always reset and initialize positions
  positions.clear();
  positions.resize(nAtoms);

  for (unsigned int i = 0; i < nAtoms; ++i) {
    if (eargs.dim == 4) {
      positions[i] = std::make_unique<RDGeom::PointND>(4);
    } else {
      positions[i] = std::make_unique<RDGeom::Point3D>();
    }
  }

  // Generate multiple conformers
  for (size_t ci = 0; ci < conformerData.confs.size(); ++ci) {
    // Embed molecule
    const bool gotCoords = embedPoints(positions, eargs, embedParams);
    if (!gotCoords) {
      conformerData.confsOk[ci] = false;
    }
  }
}
// Turn off linter for RDKit ports
// NOLINTBEGIN

//! Populates double-bond info for later stereochem checks.
void findDoubleBonds(const ROMol&                                                       mol,
                     std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>& doubleBondEnds,
                     std::vector<std::pair<std::vector<unsigned int>, int>>&            stereoDoubleBonds,
                     const std::map<int, RDGeom::Point3D>*                              coordMap) {
  doubleBondEnds.clear();
  stereoDoubleBonds.clear();
  for (const auto bnd : mol.bonds()) {
    if (bnd->getBondType() == Bond::BondType::DOUBLE) {
      for (const auto atm : {bnd->getBeginAtom(), bnd->getEndAtom()}) {
        if (atm->getDegree() < 2) {
          continue;
        }
        auto oatm = bnd->getOtherAtom(atm);
        for (const auto nbr : mol.atomNeighbors(atm)) {
          if (nbr == oatm) {
            continue;
          }
          const auto obnd = mol.getBondBetweenAtoms(atm->getIdx(), nbr->getIdx());
          if (!obnd || (obnd->getBondType() != Bond::BondType::SINGLE && atm->getDegree() == 2)) {
            continue;
          }
          doubleBondEnds.emplace_back(nbr->getIdx(), atm->getIdx(), oatm->getIdx());
        }
      }
      // if there's stereo, handle that too:
      if (bnd->getStereo() > Bond::BondStereo::STEREOANY) {
        // only do this if the controlling atoms aren't in the coord map
        if (coordMap && coordMap->find(bnd->getStereoAtoms()[0]) != coordMap->end() &&
            coordMap->find(bnd->getStereoAtoms()[1]) != coordMap->end()) {
          continue;
        }
        int sign = 1;
        if (bnd->getStereo() == Bond::BondStereo::STEREOCIS || bnd->getStereo() == Bond::BondStereo::STEREOZ) {
          sign = -1;
        }
        std::pair<std::vector<unsigned int>, int> elem{
          {static_cast<unsigned>(bnd->getStereoAtoms()[0]),
           bnd->getBeginAtomIdx(),
           bnd->getEndAtomIdx(),
           static_cast<unsigned>(bnd->getStereoAtoms()[1])},
          sign
        };
        stereoDoubleBonds.push_back(elem);
      }
    }
  }
}

// NOLINTEND

}  // namespace EmbeddingOps

// Prepares ETKDG parameters and initializes necessary data structures
// Returns true if successful, false if bounds matrix setup failed
bool prepareEmbedderArgs(ROMol&                                      mol,
                         const RDKit::DGeomHelpers::EmbedParameters& params,
                         detail::EmbedArgs&                          eargs,
                         bool                                        setupBoundsMatrix) {
  if (mol.getNumAtoms() == 0) {
    throw ValueErrorException("molecule has no atoms");
  }

  if (params.ETversion < 1 || params.ETversion > 2) {
    throw ValueErrorException(
      "Only version 1 and 2 of the experimental "
      "torsion-angle preferences (ETversion) supported");
  }

  if (MolOps::needsHs(mol)) {
    BOOST_LOG(rdWarningLog) << "Molecule does not have explicit Hs. Consider calling AddHs()\n";
  }

  const std::map<int, RDGeom::Point3D>* coordMap = params.coordMap;
  const unsigned int                    nAtoms   = mol.getNumAtoms();

  // Initialize ETKDG details
  EmbeddingOps::initETKDG(&mol, params, eargs.etkdgDetails);

  // Create and initialize the distance bounds matrix
  eargs.mmat = std::make_unique<DistGeom::BoundsMatrix>(nAtoms);
  RDKit::DGeomHelpers::initBoundsMat(eargs.mmat);

  if (setupBoundsMatrix) {
    // Set up topology bounds and triangle smoothing
    if (!EmbeddingOps::setupInitialBoundsMatrix(&mol, eargs.mmat, coordMap, params, eargs.etkdgDetails)) {
      return false;
    }
  }

  // Find chiral centers
  MolOps::assignStereochemistry(mol);
  EmbeddingOps::findChiralSets(mol, eargs.chiralCenters, eargs.tetrahedralCarbons, coordMap);
  EmbeddingOps::findDoubleBonds(mol, eargs.doubleBondEnds, eargs.stereoDoubleBonds, coordMap);

  return true;
}

std::unique_ptr<ForceFields::ForceField> generateRDKitFF(ROMol&                                       mol,
                                                         RDKit::DGeomHelpers::EmbedParameters&        params,
                                                         detail::EmbedArgs&                           eargs,
                                                         std::vector<std::unique_ptr<RDGeom::Point>>& positions,
                                                         const Dimensionality                         dimensionality) {
  if (params.ETversion < 1 || params.ETversion > 2) {
    throw ValueErrorException(
      "Only version 1 and 2 of the experimental "
      "torsion-angle preferences (ETversion) supported");
  }

  if (!prepareEmbedderArgs(mol, params, eargs, true)) {
    return nullptr;
  }

  // Determine dimensionality
  switch (dimensionality) {
    case Dimensionality::DIM_4D:
      eargs.dim = 4;
      break;
    case Dimensionality::DIM_3D:
    default:
      throw std::runtime_error("3D ETKDG should not be active at this time.");
  }

  // Construct and return the force field
  return EmbeddingOps::constructForceField(positions, eargs, params);
}

void setupRDKitFFWithPos(RDKit::ROMol*                                mol,
                         RDKit::DGeomHelpers::EmbedParameters&        params,
                         std::unique_ptr<ForceFields::ForceField>&    field,
                         detail::EmbedArgs&                           eargs,
                         std::vector<std::unique_ptr<RDGeom::Point>>& positions,
                         const int                                    confId,
                         const Dimensionality                         dimensionality) {
  const unsigned int nAtoms = mol->getNumAtoms();

  // Check if molecule has conformers and copy positions if it does
  if (mol->getNumConformers() > 0) {
    const auto& conf = confId >= 0 ? mol->getConformer(confId) : mol->getConformer();
    positions.clear();
    positions.reserve(nAtoms);
    for (unsigned int i = 0; i < nAtoms; ++i) {
      const RDGeom::Point3D& pos = conf.getAtomPos(i);
      positions.push_back(std::make_unique<RDGeom::Point3D>(pos.x, pos.y, pos.z));
    }
  }

  field = generateRDKitFF(*mol, params, eargs, positions, dimensionality);

  eargs.posVec.resize(nAtoms * eargs.dim);
  for (unsigned int i = 0; i < nAtoms; i++) {
    for (int j = 0; j < eargs.dim; j++) {
      eargs.posVec[i * eargs.dim + j] = (*positions[i])[j];
    }
  }
}
}  // namespace DGeomHelpers
}  // namespace nvMolKit
