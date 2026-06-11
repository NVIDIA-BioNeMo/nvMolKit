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

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/ChiralSet.h>
#include <ForceField/ForceField.h>
#include <gmock/gmock.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <random>

#include "src/embedder_utils.h"
#include "src/etkdg_impl.h"
#include "src/etkdg_stage_stereochem_checks.h"
#include "src/forcefields/dist_geom.h"
#include "tests/test_utils.h"

using namespace ::nvMolKit::detail;

// -----------------------
// Setup and test helpers
// -----------------------
// Helper function for common initialization logic
void initTestComponentsCommon(const std::vector<const ROMol*>&           mols,
                              const std::vector<std::unique_ptr<ROMol>>& molsPtrs,
                              ETKDGContext&                              context,
                              std::vector<EmbedArgs>&                    eargsVec,
                              DGeomHelpers::EmbedParameters&             embedParam,
                              ETKDGOption                                option,
                              const int                                  dim) {
  // Get ETKDG options and store for later use
  embedParam = getETKDGOption(option);
  const auto dimensionalityOption =
    dim == 3 ? nvMolKit::DGeomHelpers::Dimensionality::DIM_3D : nvMolKit::DGeomHelpers::Dimensionality::DIM_4D;
  if (dim == 3) {
    throw std::runtime_error("3D ETKDG should not be active at this time.");
  }

  // Initialize context
  context.nTotalSystems         = mols.size();
  context.systemHost.atomStarts = {0};
  context.systemHost.positions.clear();

  for (size_t i = 0; i < mols.size(); ++i) {
    EmbedArgs eargs;
    eargs.dim = dim;
    std::vector<std::unique_ptr<RDGeom::Point>> positions;
    std::unique_ptr<ForceFields::ForceField>    field;
    auto                                        params = DGeomHelpers::ETKDGv3;

    nvMolKit::DGeomHelpers::setupRDKitFFWithPos(molsPtrs[i].get(),
                                                params,
                                                field,
                                                eargs,
                                                positions,
                                                -1,  // Use default conformer
                                                dimensionalityOption);

    nvMolKit::DistGeom::addMoleculeToContextWithPositions(eargs.posVec,
                                                          eargs.dim,
                                                          context.systemHost.atomStarts,
                                                          context.systemHost.positions);

    // Store embed args for later use
    eargsVec.push_back(std::move(eargs));
  }

  // Send context to device
  nvMolKit::DistGeom::sendContextToDevice(context.systemHost.positions,
                                          context.systemDevice.positions,
                                          context.systemHost.atomStarts,
                                          context.systemDevice.atomStarts);
}

std::pair<std::vector<std::unique_ptr<ROMol>>, std::vector<const ROMol*>> getMolsWithView() {
  const std::string                   fileName = getTestDataFolderPath() + "/MMFF94_dative.sdf";
  std::vector<std::unique_ptr<ROMol>> mols;
  getMols(fileName, mols);
  std::vector<const ROMol*> molViews;
  for (const auto& mol : mols) {
    molViews.push_back(mol.get());
  }
  return {std::move(mols), std::move(molViews)};
}

constexpr int wantNumConfsParsedMMFF = 761;

// Check that all molecules in the set have succeeded, with noted exception counts.
void allPassChecks(const ETKDGDriver& driver, int expectedExceptions = 0) {
  EXPECT_EQ(driver.numConfsFinished(), wantNumConfsParsedMMFF - expectedExceptions);
  EXPECT_EQ(driver.iterationsComplete(), 1);

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  const auto                          failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 1);  // One stage

  const auto totalFailures =
    std::accumulate(failureCounts[0].begin(), failureCounts[0].end(), static_cast<std::int16_t>(0));
  EXPECT_EQ(totalFailures, expectedExceptions);

  const auto completed      = driver.completedConformers();
  const int  completedCount = std::accumulate(completed.begin(), completed.end(), static_cast<int16_t>(0));
  const int  totalConfs     = failureCounts[0].size();
  EXPECT_EQ(completedCount, totalConfs - expectedExceptions);
}

// ------------------------------------
// Reference implementations from RDKit,
// when they are not exposed as headers.
// ------------------------------------

namespace RDKit {
namespace DGeomHelpers {
constexpr double MIN_TETRAHEDRAL_CHIRAL_VOL     = 0.50;
constexpr double TETRAHEDRAL_CENTERINVOLUME_TOL = 0.30;

//! Direct port of RDKit _volumeTest for CPU reference.
bool _volumeTest(const DistGeom::ChiralSetPtr& chiralSet, const RDGeom::PointPtrVect& positions) {
  RDGeom::Point3D p0((*positions[chiralSet->d_idx0])[0],
                     (*positions[chiralSet->d_idx0])[1],
                     (*positions[chiralSet->d_idx0])[2]);
  RDGeom::Point3D p1((*positions[chiralSet->d_idx1])[0],
                     (*positions[chiralSet->d_idx1])[1],
                     (*positions[chiralSet->d_idx1])[2]);
  RDGeom::Point3D p2((*positions[chiralSet->d_idx2])[0],
                     (*positions[chiralSet->d_idx2])[1],
                     (*positions[chiralSet->d_idx2])[2]);
  RDGeom::Point3D p3((*positions[chiralSet->d_idx3])[0],
                     (*positions[chiralSet->d_idx3])[1],
                     (*positions[chiralSet->d_idx3])[2]);
  RDGeom::Point3D p4((*positions[chiralSet->d_idx4])[0],
                     (*positions[chiralSet->d_idx4])[1],
                     (*positions[chiralSet->d_idx4])[2]);

  // even if we are minimizing in higher dimension the chiral volume is
  // calculated using only the first 3 dimensions
  RDGeom::Point3D v1 = p0 - p1;
  v1.normalize();
  RDGeom::Point3D v2 = p0 - p2;
  v2.normalize();
  RDGeom::Point3D v3 = p0 - p3;
  v3.normalize();
  RDGeom::Point3D v4 = p0 - p4;
  v4.normalize();

  // be more tolerant of tethrahedral centers that are involved in multiple
  // small rings
  double volScale = 1;
#if (RDKIT_VERSION_MAJOR > 2024 || (RDKIT_VERSION_MAJOR == 2024 && RDKIT_VERSION_MINOR >= 9))
  if (chiralSet->d_structureFlags &
      static_cast<std::uint64_t>(DistGeom::ChiralSetStructureFlags::IN_FUSED_SMALL_RINGS)) {
    volScale = 0.25;
  }
#endif
  RDGeom::Point3D crossp = v1.crossProduct(v2);
  double          vol    = crossp.dotProduct(v3);
  if (fabs(vol) < volScale * MIN_TETRAHEDRAL_CHIRAL_VOL) {
    return false;
  }
  crossp = v1.crossProduct(v2);
  vol    = crossp.dotProduct(v4);
  if (fabs(vol) < volScale * MIN_TETRAHEDRAL_CHIRAL_VOL) {
    return false;
  }
  crossp = v1.crossProduct(v3);
  vol    = crossp.dotProduct(v4);
  if (fabs(vol) < volScale * MIN_TETRAHEDRAL_CHIRAL_VOL) {
    return false;
  }
  crossp = v2.crossProduct(v3);
  vol    = crossp.dotProduct(v4);
  return fabs(vol) >= volScale * MIN_TETRAHEDRAL_CHIRAL_VOL;
}

//! Direct port of RDKit _sameSIde for CPU reference.
bool _sameSide(const RDGeom::Point3D& v1,
               const RDGeom::Point3D& v2,
               const RDGeom::Point3D& v3,
               const RDGeom::Point3D& v4,
               const RDGeom::Point3D& p0,
               double                 tol = 0.1) {
  RDGeom::Point3D normal = (v2 - v1).crossProduct(v3 - v1);
  double          d1     = normal.dotProduct(v4 - v1);
  double          d2     = normal.dotProduct(p0 - v1);
  if (fabs(d1) < tol || fabs(d2) < tol) {
    return false;
  }
  return !((d1 < 0.) ^ (d2 < 0.));
}

//! Direct port of RDKit _centerInVolume for CPU reference.
bool _centerInVolume(unsigned int                idx0,
                     unsigned int                idx1,
                     unsigned int                idx2,
                     unsigned int                idx3,
                     unsigned int                idx4,
                     const RDGeom::PointPtrVect& positions,
                     double                      tol) {
  RDGeom::Point3D p0((*positions[idx0])[0], (*positions[idx0])[1], (*positions[idx0])[2]);
  RDGeom::Point3D p1((*positions[idx1])[0], (*positions[idx1])[1], (*positions[idx1])[2]);
  RDGeom::Point3D p2((*positions[idx2])[0], (*positions[idx2])[1], (*positions[idx2])[2]);
  RDGeom::Point3D p3((*positions[idx3])[0], (*positions[idx3])[1], (*positions[idx3])[2]);
  RDGeom::Point3D p4((*positions[idx4])[0], (*positions[idx4])[1], (*positions[idx4])[2]);
  bool            res = _sameSide(p1, p2, p3, p4, p0, tol) && _sameSide(p2, p3, p4, p1, p0, tol) &&
             _sameSide(p3, p4, p1, p2, p0, tol) && _sameSide(p4, p1, p2, p3, p0, tol);
  return res;
}

bool _centerInVolume(const DistGeom::ChiralSetPtr& chiralSet, const RDGeom::PointPtrVect& positions, double tol = 0.1) {
  if (chiralSet->d_idx0 == chiralSet->d_idx4) {  // this happens for three-coordinate centers
    return true;
  }
  return _centerInVolume(chiralSet->d_idx0,
                         chiralSet->d_idx1,
                         chiralSet->d_idx2,
                         chiralSet->d_idx3,
                         chiralSet->d_idx4,
                         positions,
                         tol);
}

//! Direct port of RDKit checkTetrahedralCenters for CPU reference.
bool checkTetrahedralCenters(const RDGeom::PointPtrVect*     positions,
                             const DistGeom::VECT_CHIRALSET& tetrahedralCenters) {
  for (const auto& tetSet : tetrahedralCenters) {
    if (!_volumeTest(tetSet, *positions) || !_centerInVolume(tetSet, *positions, TETRAHEDRAL_CENTERINVOLUME_TOL)) {
      return false;
    }
  }
  return true;
}
}  // namespace DGeomHelpers
}  // namespace RDKit

namespace {

void perturbConformer(RDKit::Conformer& conf, const float delta = 0.1, const int seed = 0) {
  std::mt19937                          gen(seed);  // Mersenne Twister engine
  std::uniform_real_distribution<float> dist(-delta, delta);
  for (unsigned int i = 0; i < conf.getNumAtoms(); ++i) {
    RDGeom::Point3D pos = conf.getAtomPos(i);
    pos.x += delta * dist(gen);
    pos.y += delta * dist(gen);
    pos.z += delta * dist(gen);
    conf.setAtomPos(i, pos);
  }
}

}  // namespace

namespace {

//! Direct port of calcChiralVolume from RDKit. Note that old versions have this
//! exposed in the API but it's location is recently moving so just reproducing instead.
double calcChiralVolume(const unsigned int          idx1,
                        const unsigned int          idx2,
                        const unsigned int          idx3,
                        const unsigned int          idx4,
                        const RDGeom::PointPtrVect& pts) {
  // even if we are minimizing in higher dimension the chiral volume is
  // calculated using only the first 3 dimensions
  RDGeom::Point3D v1((*pts[idx1])[0] - (*pts[idx4])[0],
                     (*pts[idx1])[1] - (*pts[idx4])[1],
                     (*pts[idx1])[2] - (*pts[idx4])[2]);

  RDGeom::Point3D v2((*pts[idx2])[0] - (*pts[idx4])[0],
                     (*pts[idx2])[1] - (*pts[idx4])[1],
                     (*pts[idx2])[2] - (*pts[idx4])[2]);

  RDGeom::Point3D v3((*pts[idx3])[0] - (*pts[idx4])[0],
                     (*pts[idx3])[1] - (*pts[idx4])[1],
                     (*pts[idx3])[2] - (*pts[idx4])[2]);

  RDGeom::Point3D v2xv3 = v2.crossProduct(v3);

  double vol = v1.dotProduct(v2xv3);
  return vol;
}

inline bool haveOppositeSign(double a, double b) {
  return std::signbit(a) ^ std::signbit(b);
}

//! CPU reproduction of RDKit chirality check for GPU comparison.
bool checkChiralCenters(const RDGeom::PointPtrVect* positions, const DistGeom::VECT_CHIRALSET& chiralSets) {
  // check the chiral volume:
  for (const auto& chiralSet : chiralSets) {
    double vol =
      calcChiralVolume(chiralSet->d_idx1, chiralSet->d_idx2, chiralSet->d_idx3, chiralSet->d_idx4, *positions);
    double lb = chiralSet->getLowerVolumeBound();
    double ub = chiralSet->getUpperVolumeBound();
    if ((lb > 0 && vol < lb && (vol / lb < .8 || haveOppositeSign(vol, lb))) ||
        (ub < 0 && vol > ub && (vol / ub < .8 || haveOppositeSign(vol, ub)))) {
      return false;
    }
  }
  return true;
}

bool _boundsFulfilled(const std::vector<int>&       atoms,
                      const DistGeom::BoundsMatrix& mmat,
                      const RDGeom::PointPtrVect&   positions) {
  // unsigned int N = mmat.numRows();
  // std::cerr << N << " " << atoms.size() << std::endl;
  // loop over all pair of atoms
  for (unsigned int i = 0; i < atoms.size() - 1; ++i) {
    for (unsigned int j = i + 1; j < atoms.size(); ++j) {
      int             a1 = atoms[i];
      int             a2 = atoms[j];
      RDGeom::Point3D p0((*positions[a1])[0], (*positions[a1])[1], (*positions[a1])[2]);
      RDGeom::Point3D p1((*positions[a2])[0], (*positions[a2])[1], (*positions[a2])[2]);
      double          d2 = (p0 - p1).length();  // distance
      double          lb = mmat.getLowerBound(a1, a2);
      double          ub = mmat.getUpperBound(a1, a2);  // bounds
      if (((d2 < lb) && (fabs(d2 - lb) > 0.1 * ub)) || ((d2 > ub) && (fabs(d2 - ub) > 0.1 * ub))) {
        return false;
      }
    }
  }
  return true;
}

bool doubleBondStereoChecks(const RDGeom::PointPtrVect&                                   positions,
                            const std::vector<std::pair<std::vector<unsigned int>, int>>& stereoDoubleBonds) {
  for (const auto& itm : stereoDoubleBonds) {
    // itm is a pair with [controlling_atoms], sign
    // where the sign tells us about cis/trans

    const auto&     a0 = *positions[itm.first[0]];
    const auto&     a1 = *positions[itm.first[1]];
    const auto&     a2 = *positions[itm.first[2]];
    const auto&     a3 = *positions[itm.first[3]];
    RDGeom::Point3D p0(a0[0], a0[1], a0[2]);
    RDGeom::Point3D p1(a1[0], a1[1], a1[2]);
    RDGeom::Point3D p2(a2[0], a2[1], a2[2]);
    RDGeom::Point3D p3(a3[0], a3[1], a3[2]);

    // check the dihedral and be super permissive. Here's the logic of the
    // check:
    // The second element of the dihedralBond item contains 1 for trans
    //   bonds and -1 for cis bonds.
    // The dihedral is between 0 and 180. subtracting 90 from that gives:
    //   positive values for dihedrals > 90 (closer to trans than cis)
    //   negative values for dihedrals < 90 (closer to cis than trans)
    // So multiplying the result of the subtracion from the second element of
    //   the dihedralBond element will give a positive value if the dihedral is
    //   closer to correct than it is to incorrect and a negative value
    //   otherwise.
    auto dihedral = RDGeom::computeDihedralAngle(p0, p1, p2, p3);
    if ((dihedral - M_PI_2) * itm.second < 0) {
      // closer to incorrect than correct... it's a bad geometry
      return false;
    }
  }
  return true;
}

bool doubleBondGeometryChecks(const RDGeom::PointPtrVect&                                              positions,
                              const std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>& doubleBondEnds,
                              double linearTol = 1e-3) {
  for (const auto& itm : doubleBondEnds) {
    const auto&     a0 = *positions[std::get<0>(itm)];
    const auto&     a1 = *positions[std::get<1>(itm)];
    const auto&     a2 = *positions[std::get<2>(itm)];
    RDGeom::Point3D p0(a0[0], a0[1], a0[2]);
    RDGeom::Point3D p1(a1[0], a1[1], a1[2]);
    RDGeom::Point3D p2(a2[0], a2[1], a2[2]);

    // check for a linear arrangement

    auto v1 = p1 - p0;
    v1.normalize();
    auto v2 = p1 - p2;
    v2.normalize();
    // this is the arrangement:
    //     a0
    //       \       [intentionally left blank]
    //        a1 = a2
    // we want to be sure it's not actually:
    //   ao - a1 = a2
    const double dotProd = v1.dotProduct(v2);
    if (dotProd + 1.0 < linearTol) {
      return false;
    }
  }
  return true;
}
}  // namespace

// Enum class for different check types
enum class ETKDGCheckType {
  Tetrahedral,
  Chirality,
  ChiralVolumeCenter,
  ChiralDistMat,
  DoubleBondGeometry,
  DoubleBondStereo
};

// Unified test fixture for both tetrahedral and chirality checks
class ETKDGUnifiedCheckTest : public ::testing::TestWithParam<std::tuple<int, ETKDGCheckType>> {
 protected:
  ETKDGContext                  context_;
  std::vector<EmbedArgs>        eargs_;
  DGeomHelpers::EmbedParameters embedParam_;

  // Helper method to create appropriate stage based on check type
  std::unique_ptr<ETKDGStage> createStage(const ETKDGCheckType checkType, int dim) {
    switch (checkType) {
      case ETKDGCheckType::Tetrahedral:
        return std::make_unique<nvMolKit::detail::ETKDGTetrahedralCheckStage>(context_, eargs_, dim);
      case ETKDGCheckType::Chirality:
        return std::make_unique<nvMolKit::detail::ETKDGFirstChiralCenterCheckStage>(context_, eargs_, dim);
      case ETKDGCheckType::ChiralVolumeCenter:
        return std::make_unique<nvMolKit::detail::ETKDGChiralCenterVolumeCheckStage>(context_, eargs_, dim);
      case ETKDGCheckType::ChiralDistMat:
        return std::make_unique<nvMolKit::detail::ETKDGChiralDistMatrixCheckStage>(context_, eargs_, dim);
      case ETKDGCheckType::DoubleBondStereo:
        return std::make_unique<nvMolKit::detail::ETKDGDoubleBondStereoCheckStage>(context_, eargs_, dim);
      case ETKDGCheckType::DoubleBondGeometry:
        return std::make_unique<nvMolKit::detail::ETKDGDoubleBondGeometryCheckStage>(context_, eargs_, dim);
      default:
        throw std::invalid_argument("Unknown check type");
    }
  }

  // Helper method to get perturbation factor based on check type. Tuned by hand.
  static float getPerturbationFactor(const ETKDGCheckType checkType) {
    switch (checkType) {
      case ETKDGCheckType::Tetrahedral:
        return 0.4f;
      case ETKDGCheckType::Chirality:
        return 1.0f;
      case ETKDGCheckType::ChiralVolumeCenter:
        return 0.7f;
      case ETKDGCheckType::ChiralDistMat:
        return 0.3f;
      case ETKDGCheckType::DoubleBondStereo:
        return 1.5f;
      case ETKDGCheckType::DoubleBondGeometry:
        return 0.0f;  // This is a check for nonlinearity, so scrambling does not induce many failures.
      default:
        throw std::invalid_argument("Unknown check type");
    }
  }

  static void linearizeSomeDoubleBonds(const std::vector<std::unique_ptr<ROMol>>& mols) {
    for (size_t i = 0; i < mols.size(); ++i) {
      // Only fail half of the molecules
      if (i % 2 == 0) {
        continue;
      }
      auto& mol = mols[i];

      std::vector<std::pair<std::vector<unsigned int>, int>>            placeholder;
      std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> doubleBonds;
      nvMolKit::DGeomHelpers::EmbeddingOps::findDoubleBonds(*mol, doubleBonds, placeholder, nullptr);

      if (doubleBonds.empty()) {
        continue;  // No double bonds to linearize
      }

      for (size_t j = 0; j < doubleBonds.size(); ++j) {
        // set half the double bonds to linear.
        // Note that this could delinearize if we're tweaking the same bond, but overall the effect should work.
        if (j % 2 != 0) {
          continue;
        }
        const auto&        db   = doubleBonds[j];
        const unsigned int idx0 = std::get<0>(db);
        const unsigned int idx1 = std::get<1>(db);
        const unsigned int idx2 = std::get<2>(db);

        RDGeom::Point3D& p0 = mol->getConformer().getAtomPos(idx0);
        RDGeom::Point3D& p2 = mol->getConformer().getAtomPos(idx2);

        // Linearize the double bond
        RDGeom::Point3D newPos = (p0 + p2) / 2.0;
        mol->getConformer().setAtomPos(idx1, newPos);
      }
    }
  }

  // Helper method to compute expected results based on check type
  static std::vector<int16_t> computeExpectedResults(const std::vector<std::unique_ptr<ROMol>>& mols,
                                                     const ETKDGCheckType                       checkType,
                                                     int&                                       numMolsWithFeatures,
                                                     const std::vector<EmbedArgs>&              eargs_) {
    std::vector<int16_t> wantCompleted;
    numMolsWithFeatures = 0;

    for (size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
      const auto&          mol = mols[molIdx];
      RDGeom::PointPtrVect positions;
      auto&                conf = mol->getConformer();
      for (unsigned int i = 0; i < conf.getNumAtoms(); ++i) {
        auto* pos = &conf.getAtomPos(i);
        positions.push_back(pos);
      }

      bool passed = false;
      switch (checkType) {
        case ETKDGCheckType::Tetrahedral: {
          DistGeom::VECT_CHIRALSET rdkitTetrahedralChecks;
          DistGeom::VECT_CHIRALSET placeHolder;
          nvMolKit::DGeomHelpers::EmbeddingOps::findChiralSets(*mol, placeHolder, rdkitTetrahedralChecks, nullptr);
          if (rdkitTetrahedralChecks.size() > 0) {
            numMolsWithFeatures++;
          }
          passed = RDKit::DGeomHelpers::checkTetrahedralCenters(&positions, rdkitTetrahedralChecks);
          break;
        }
        case ETKDGCheckType::Chirality: {
          DistGeom::VECT_CHIRALSET chiralSets;
          DistGeom::VECT_CHIRALSET placeHolder;
          nvMolKit::DGeomHelpers::EmbeddingOps::findChiralSets(*mol, chiralSets, placeHolder, nullptr);
          if (chiralSets.size() > 0) {
            numMolsWithFeatures++;
          }
          passed = checkChiralCenters(&positions, chiralSets);
          break;
        }
        case ETKDGCheckType::ChiralVolumeCenter: {
          DistGeom::VECT_CHIRALSET chiralSets;
          DistGeom::VECT_CHIRALSET placeHolder;
          nvMolKit::DGeomHelpers::EmbeddingOps::findChiralSets(*mol, chiralSets, placeHolder, nullptr);
          if (chiralSets.size() > 0) {
            numMolsWithFeatures++;
          }
          passed = true;
          for (const auto& chiralSet : chiralSets) {
            if (!RDKit::DGeomHelpers::_centerInVolume(chiralSet, positions)) {
              passed = false;
              break;
            }
          }
          break;
        }
        case ETKDGCheckType::ChiralDistMat: {
          DistGeom::VECT_CHIRALSET chiralSets;
          DistGeom::VECT_CHIRALSET placeHolder;
          nvMolKit::DGeomHelpers::EmbeddingOps::findChiralSets(*mol, chiralSets, placeHolder, nullptr);
          if (chiralSets.size() > 0) {
            numMolsWithFeatures++;
          }
          std::set<int> atoms;
          passed = true;

          for (const auto& chiralSet : eargs_[molIdx].chiralCenters) {
            if (chiralSet->d_idx0 != chiralSet->d_idx4) {
              atoms.insert(chiralSet->d_idx0);
              atoms.insert(chiralSet->d_idx1);
              atoms.insert(chiralSet->d_idx2);
              atoms.insert(chiralSet->d_idx3);
              atoms.insert(chiralSet->d_idx4);
            }
          }
          if (std::vector<int> atomsToCheck(atoms.begin(), atoms.end());
              atomsToCheck.size() > 0 && !_boundsFulfilled(atomsToCheck, *eargs_[molIdx].mmat, positions)) {
            passed = false;
            break;
          }
          break;
        }
        case ETKDGCheckType::DoubleBondStereo: {
          std::vector<std::pair<std::vector<unsigned int>, int>>            stereoDoubleBonds;
          std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> placeHolder;
          nvMolKit::DGeomHelpers::EmbeddingOps::findDoubleBonds(*mol, placeHolder, stereoDoubleBonds, nullptr);
          if (stereoDoubleBonds.size() > 0) {
            numMolsWithFeatures++;
          }
          passed = doubleBondStereoChecks(positions, stereoDoubleBonds);
          break;
        }
        case ETKDGCheckType::DoubleBondGeometry: {
          std::vector<std::pair<std::vector<unsigned int>, int>>            placeholder;
          std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> doubleBondGeoms;
          nvMolKit::DGeomHelpers::EmbeddingOps::findDoubleBonds(*mol, doubleBondGeoms, placeholder, nullptr);
          if (doubleBondGeoms.size() > 0) {
            numMolsWithFeatures++;
          }
          passed = doubleBondGeometryChecks(positions, doubleBondGeoms);
          break;
        }
        default:
          throw std::invalid_argument("Unknown check type");
      }
      wantCompleted.push_back(passed);
    }
    return wantCompleted;
  }
};

TEST_P(ETKDGUnifiedCheckTest, MMFFConformersAllPassByDefault) {
  // -----
  // Setup
  // -----
  const int            dim       = std::get<0>(GetParam());
  const ETKDGCheckType checkType = std::get<1>(GetParam());
  auto [mols, molViews]          = getMolsWithView();
  ASSERT_EQ(mols.size(), wantNumConfsParsedMMFF);

  initTestComponentsCommon(molViews, mols, context_, eargs_, embedParam_, ETKDGOption::ETKDGv3, dim);
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  stages.push_back(createStage(checkType, dim));
  ETKDGDriver driver(std::make_unique<ETKDGContext>(std::move(context_)), std::move(stages));

  // -----------
  // Execution
  // -----------
  driver.run(1);

  // ------------
  // Verification
  // ------------
  // Several MMFF conformers fail some of the chirality checks. Verified this is real outside of our implementation,
  // it identifies some critical differences between MMFF and DGeom forcefields that we could look into.
  int expectedFailures = 0;
  switch (checkType) {
    case ETKDGCheckType::ChiralVolumeCenter:
      expectedFailures = 3;
      break;
    case ETKDGCheckType::ChiralDistMat:
      expectedFailures = 23;
      break;
    default:
      expectedFailures = 0;
      break;
  }
  allPassChecks(driver, expectedFailures);
}

TEST_P(ETKDGUnifiedCheckTest, MMFFConformersScrambledMatchesRDKitResult) {
  // -------
  // Setup
  // -------
  const int            dim       = std::get<0>(GetParam());
  const ETKDGCheckType checkType = std::get<1>(GetParam());
  auto [mols, molViews]          = getMolsWithView();

  // Perturb the coordinates of each molecule with check-type specific factor
  const float perturbationFactor = getPerturbationFactor(checkType);
  for (auto& mol : mols) {
    auto& conf = mol->getConformer();
    perturbConformer(conf, perturbationFactor);
  }
  if (checkType == ETKDGCheckType::DoubleBondGeometry) {
    // For double bond geometry, we need to linearize some double bonds, that won't start failing with coordinate
    // randomization.
    linearizeSomeDoubleBonds(mols);
  }
  initTestComponentsCommon(molViews, mols, context_, eargs_, embedParam_, ETKDGOption::ETKDGv3, dim);
  std::vector<std::unique_ptr<ETKDGStage>> stages;
  stages.push_back(createStage(checkType, dim));
  ETKDGDriver driver(std::make_unique<ETKDGContext>(std::move(context_)), std::move(stages));
  // Compute expected results using CPU reference implementation.
  // This **must** be done after the GPU component initialization, since eargs is populated there.
  int         numMolsWithFeatures = 0;

  std::vector<int16_t> wantCompleted = computeExpectedResults(mols, checkType, numMolsWithFeatures, eargs_);

  // Make sure our scrambling has induced some failures but not in every single molecule
  const int totalWantFailures = mols.size() - std::accumulate(wantCompleted.begin(), wantCompleted.end(), 0);
  ASSERT_GT(totalWantFailures, 0);
  ASSERT_LT(totalWantFailures, numMolsWithFeatures);

  // ---------
  // Execution
  // ---------
  driver.run(1);

  // ------------
  // Verification
  // ------------
  EXPECT_EQ(driver.iterationsComplete(), 1);
  const auto completed = driver.completedConformers();
  EXPECT_THAT(completed, ::testing::Pointwise(::testing::Eq(), wantCompleted));
}

INSTANTIATE_TEST_SUITE_P(ETKDGUnifiedCheckTests,
                         ETKDGUnifiedCheckTest,
                         ::testing::Combine(::testing::Values(4),
                                            ::testing::Values(ETKDGCheckType::Tetrahedral,
                                                              ETKDGCheckType::Chirality,
                                                              ETKDGCheckType::ChiralVolumeCenter,
                                                              ETKDGCheckType::ChiralDistMat,
                                                              ETKDGCheckType::DoubleBondGeometry,
                                                              ETKDGCheckType::DoubleBondStereo)),

                         [](const ::testing::TestParamInfo<std::tuple<int, ETKDGCheckType>>& info) {
                           std::string checkTypeName;
                           switch (std::get<1>(info.param)) {
                             case ETKDGCheckType::Tetrahedral:
                               checkTypeName = "Tetrahedral";
                               break;
                             case ETKDGCheckType::Chirality:
                               checkTypeName = "Chirality";
                               break;
                             case ETKDGCheckType::ChiralVolumeCenter:
                               checkTypeName = "ChiralVolumeCenter";
                               break;
                             case ETKDGCheckType::ChiralDistMat:
                               checkTypeName = "ChiralDistMat";
                               break;
                             case ETKDGCheckType::DoubleBondStereo:
                               checkTypeName = "DoubleBondStereo";
                               break;
                             case ETKDGCheckType::DoubleBondGeometry:
                               checkTypeName = "DoubleBondGeometry";
                               break;
                             default:
                               checkTypeName = "Unknown";
                               break;
                           }
                           return "Dim" + std::to_string(std::get<0>(info.param)) + "_" + checkTypeName;
                         });
