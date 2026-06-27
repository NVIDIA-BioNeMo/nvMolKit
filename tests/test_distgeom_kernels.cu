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

#include <DistGeom/ChiralViolationContribs.h>
#include <DistGeom/DistViolationContribs.h>
#include <DistGeom/FourthDimContribs.h>
#include <ForceField/AngleConstraints.h>
#include <ForceField/DistanceConstraints.h>
#include <ForceField/UFF/Inversions.h>
#include <gmock/gmock.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionAngleContribs.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <random>

#include "rdkit_extensions/bounds_matrix.h"
#include "rdkit_extensions/dist_geom_flattened_builder.h"
#include "src/embedder_utils.h"
#include "src/forcefields/dist_geom.h"
#include "src/forcefields/dist_geom_kernels.h"
#include "tests/test_utils.h"

namespace {

// Test tolerances
constexpr double E_TOL_FUNCTION = 1e-5;
constexpr double E_TOL_COMBINED = 1e-4;
constexpr double G_TOL          = 1e-4;

// Copy of term additions for 3D forcefield.
constexpr double KNOWN_DIST_FORCE_CONSTANT = 100.0;  // Force constant for known distances
constexpr double KNOWN_DIST_TOL            = 0.01;   // Tolerance for known distances

void addImproperTorsionTerms(ForceFields::ForceField*             ff,
                             double                               forceScalingFactor,
                             const std::vector<std::vector<int>>& improperAtoms,
                             boost::dynamic_bitset<>&             isImproperConstrained) {
  PRECONDITION(ff, "bad force field");
  auto inversionContribs = std::make_unique<ForceFields::UFF::InversionContribs>(ff);
  for (const auto& improperAtom : improperAtoms) {
    std::vector<int> n(4);
    for (unsigned int i = 0; i < 3; ++i) {
      n[1] = 1;
      switch (i) {
        case 0:
          n[0] = 0;
          n[2] = 2;
          n[3] = 3;
          break;

        case 1:
          n[0] = 0;
          n[2] = 3;
          n[3] = 2;
          break;

        case 2:
          n[0] = 2;
          n[2] = 3;
          n[3] = 0;
          break;
        default:
          throw std::runtime_error("Bad improper atom index");
      }

      inversionContribs->addContrib(improperAtom[n[0]],
                                    improperAtom[n[1]],
                                    improperAtom[n[2]],
                                    improperAtom[n[3]],
                                    improperAtom[4],
                                    static_cast<bool>(improperAtom[5]),
                                    forceScalingFactor);
      isImproperConstrained[improperAtom[n[1]]] = true;
    }
  }
  if (!inversionContribs->empty()) {
    ff->contribs().push_back(std::move(inversionContribs));
  }
}

//! Add experimental torsion angle contributions to a force field
/*!

  \param ff Force field to add contributions to
  \param etkdgDetails Contains information about the ETKDG force field
  \param atomPairs bit set for every atom pair in the molecule where
  a bit is set to one when the atom pair are the end atoms of a torsion
  angle contribution
  \param numAtoms number of atoms in the molecule

 */
void addExperimentalTorsionTerms(ForceFields::ForceField*                        ff,
                                 const ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
                                 boost::dynamic_bitset<>&                        atomPairs,
                                 unsigned int                                    numAtoms) {
  PRECONDITION(ff, "bad force field");
  auto torsionContribs = std::make_unique<ForceFields::CrystalFF::TorsionAngleContribs>(ff);
  for (unsigned int t = 0; t < etkdgDetails.expTorsionAtoms.size(); ++t) {
    int i = etkdgDetails.expTorsionAtoms[t][0];
    int j = etkdgDetails.expTorsionAtoms[t][1];
    int k = etkdgDetails.expTorsionAtoms[t][2];
    int l = etkdgDetails.expTorsionAtoms[t][3];
    if (i < l) {
      atomPairs[i * numAtoms + l] = true;
    } else {
      atomPairs[l * numAtoms + i] = true;
    }
    torsionContribs
      ->addContrib(i, j, k, l, etkdgDetails.expTorsionAngles[t].second, etkdgDetails.expTorsionAngles[t].first);
  }
  if (!torsionContribs->empty()) {
    ff->contribs().push_back(std::move(torsionContribs));
  }
}

//! Add bond constraints with padding at current positions to force field
/*!

  \param ff Force field to add contributions to
  \param etkdgDetails Contains information about the ETKDG force field
  \param atomPairs bit set for every atom pair in the molecule where
  a bit is set to one when the atom pair is a bond that is constrained here
  \param positions A vector of pointers to 3D Points to write out the
  resulting coordinates
  \param forceConstant force constant with which to constrain bond distances
  \param numAtoms number of atoms in molecule

*/
void add12Terms(ForceFields::ForceField*                        ff,
                const ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
                boost::dynamic_bitset<>&                        atomPairs,
                RDGeom::Point3DPtrVect&                         positions,
                double                                          forceConstant,
                unsigned int                                    numAtoms) {
  PRECONDITION(ff, "bad force field");
  auto distContribs = std::make_unique<ForceFields::DistanceConstraintContribs>(ff);
  for (const auto& bond : etkdgDetails.bonds) {
    unsigned int i = bond.first;
    unsigned int j = bond.second;
    if (i < j) {
      atomPairs[i * numAtoms + j] = 1;
    } else {
      atomPairs[j * numAtoms + i] = 1;
    }
    double d = ((*positions[i]) - (*positions[j])).length();
    distContribs->addContrib(i, j, d - KNOWN_DIST_TOL, d + KNOWN_DIST_TOL, forceConstant);
  }
  if (!distContribs->empty()) {
    ff->contribs().push_back(std::move(distContribs));
  }
}
//! Add 1-3 distance constraints with padding at current positions to force
/// field
/*!

  \param ff Force field to add contributions to
  \param etkdgDetails Contains information about the ETKDG force field
  \param atomPairs bit set for every atom pair in the molecule where
  a bit is set to one when the atom pair is the both end atoms of a 13
  contribution that is constrained here
  \param positions A vector of pointers to 3D Points to write out the resulting
  coordinates \param forceConstant force constant with which to constrain bond
  distances \param isImproperConstrained bit vector with length of total num
  atoms of the molecule where index of every central atom of improper torsion is
  set to one \param useBasicKnowledge whether to use basic knowledge terms
  \param mmat Bounds matrix from which 13 distances are used in case an angle
  is part of an improper torsion
  \param numAtoms number of atoms in molecule

*/
void add13Terms(ForceFields::ForceField*                        ff,
                const ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
                boost::dynamic_bitset<>&                        atomPairs,
                RDGeom::Point3DPtrVect&                         positions,
                double                                          forceConstant,
                const boost::dynamic_bitset<>&                  isImproperConstrained,
                bool                                            useBasicKnowledge,
                const DistGeom::BoundsMatrix&                   mmat,
                unsigned int                                    numAtoms) {
  PRECONDITION(ff, "bad force field");
  auto distContribs  = std::make_unique<ForceFields::DistanceConstraintContribs>(ff);
  auto angleContribs = std::make_unique<ForceFields::AngleConstraintContribs>(ff);
  for (const auto& angle : etkdgDetails.angles) {
    unsigned int i = angle[0];
    unsigned int j = angle[1];
    unsigned int k = angle[2];
    if (i < k) {
      atomPairs[i * numAtoms + k] = 1;
    } else {
      atomPairs[k * numAtoms + i] = 1;
    }
    // check for triple bonds
    if (useBasicKnowledge && angle[3]) {
      angleContribs->addContrib(i, j, k, 179.0, 180.0, 1);
    } else if (isImproperConstrained[j]) {
      distContribs->addContrib(i, k, mmat.getLowerBound(i, k), mmat.getUpperBound(i, k), forceConstant);
    } else {
      double d = ((*positions[i]) - (*positions[k])).length();
      distContribs->addContrib(i, k, d - KNOWN_DIST_TOL, d + KNOWN_DIST_TOL, forceConstant);
    }
  }
  if (!angleContribs->empty()) {
    ff->contribs().push_back(std::move(angleContribs));
  }
  if (!distContribs->empty()) {
    ff->contribs().push_back(std::move(distContribs));
  }
}

//! Add long distance constraints to bounds matrix borders or constrained atoms
/// when provideds
/*!

  \param ff Force field to add contributions to
  \param etkdgDetails Contains information about the ETKDG force field
  \param atomPairs bit set for every atom pair in the molecule where
  a bit is set to one when the two atoms in the pair are distance constrained
  with respect to each other
  \param positions A vector of pointers to 3D Points to write out the
  resulting coordinates
  \param knownDistanceForceConstant force constant with which to constrain bond
  distances
  \param mmat  Bounds matrix to use bounds from for constraints
  \param numAtoms number of atoms in molecule

*/
void addLongRangeDistanceConstraints(ForceFields::ForceField*                        ff,
                                     const ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
                                     const boost::dynamic_bitset<>&                  atomPairs,
                                     RDGeom::Point3DPtrVect&                         positions,
                                     double                                          knownDistanceForceConstant,
                                     const DistGeom::BoundsMatrix&                   mmat,
                                     unsigned int                                    numAtoms) {
  PRECONDITION(ff, "bad force field");
  auto   distContribs = std::make_unique<ForceFields::DistanceConstraintContribs>(ff);
  double fdist        = knownDistanceForceConstant;
  for (unsigned int i = 1; i < numAtoms; ++i) {
    for (unsigned int j = 0; j < i; ++j) {
      if (!atomPairs[j * numAtoms + i]) {
        fdist    = etkdgDetails.boundsMatForceScaling * 10.0;
        double l = mmat.getLowerBound(i, j);
        double u = mmat.getUpperBound(i, j);
        if (!etkdgDetails.constrainedAtoms.empty() && etkdgDetails.constrainedAtoms[i] &&
            etkdgDetails.constrainedAtoms[j]) {
          // we're constrained, so use very tight bounds
          l = u = ((*positions[i]) - (*positions[j])).length();
          l -= KNOWN_DIST_TOL;
          u += KNOWN_DIST_TOL;
          fdist = knownDistanceForceConstant;
        }
        // printf("CPU LR Term %d indices %d %d minLen %f maxlen %f const %f\n", distContribs->size(), i, j, l, u,
        // fdist);
        distContribs->addContrib(i, j, l, u, fdist);
      }
    }
  }
  if (!distContribs->empty()) {
    ff->contribs().push_back(std::move(distContribs));
  }
}

//! Mark atom pairs that would be constrained by other terms without adding force field contributions
/*!
  This function marks atom pairs exactly as the other 3D ETKDG terms would mark them,
  but without adding any actual force field contributions. This allows the long range
  distance term to be tested in true isolation by only constraining the pairs it should.

  \param etkdgDetails Contains information about the ETKDG force field
  \param positions A vector of pointers to 3D Points
  \param mmat Bounds matrix for 1-3 distance constraints
  \param numAtoms number of atoms in molecule
  \param atomPairs [out] bit set that will be marked for constrained atom pairs
  \param isImproperConstrained [out] bit vector marking improper torsion central atoms
*/
void markAtomPairsForIsolatedLongRange(const ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
                                       unsigned int                                    numAtoms,
                                       boost::dynamic_bitset<>&                        atomPairs,
                                       boost::dynamic_bitset<>&                        isImproperConstrained) {
  // Mark experimental torsion pairs (end atoms of torsions)
  for (unsigned int t = 0; t < etkdgDetails.expTorsionAtoms.size(); ++t) {
    int i = etkdgDetails.expTorsionAtoms[t][0];
    int l = etkdgDetails.expTorsionAtoms[t][3];
    if (i < l) {
      atomPairs[i * numAtoms + l] = 1;
    } else {
      atomPairs[l * numAtoms + i] = 1;
    }
  }

  // Mark improper torsion central atoms
  for (const auto& improperAtom : etkdgDetails.improperAtoms) {
    isImproperConstrained[improperAtom[1]] = 1;  // Central atom
  }

  // Mark 1-2 bond pairs
  for (const auto& bond : etkdgDetails.bonds) {
    unsigned int i = bond.first;
    unsigned int j = bond.second;
    if (i < j) {
      atomPairs[i * numAtoms + j] = 1;
    } else {
      atomPairs[j * numAtoms + i] = 1;
    }
  }

  // Mark 1-3 pairs (both distance and angle constraints)
  for (const auto& angle : etkdgDetails.angles) {
    unsigned int i = angle[0];
    unsigned int k = angle[2];
    if (i < k) {
      atomPairs[i * numAtoms + k] = 1;
    } else {
      atomPairs[k * numAtoms + i] = 1;
    }
  }
}
}  // namespace

// Enum for standard distance geometry terms
enum class FFTerm {
  DistanceViolation,
  ChiralViolation,
  FourthDim
};

// Enum for 3D ETKDG terms
enum class ETK3DTerm {
  ExperimentalTorsion,
  ImproperTorsion,
  Distance12,
  Distance13,
  Angle13,
  LongRangeDistance,
};

constexpr std::array<ETK3DTerm, 6> allETK3DTerms = {ETK3DTerm::ExperimentalTorsion,
                                                    ETK3DTerm::ImproperTorsion,
                                                    ETK3DTerm::Distance12,
                                                    ETK3DTerm::Distance13,
                                                    ETK3DTerm::Angle13,
                                                    ETK3DTerm::LongRangeDistance};

// Plain 3D terms(ETDG variant) - excludes improper torsion terms(useBasicKnowledge = false)
constexpr std::array<ETK3DTerm, 5> plainETK3DTerms = {ETK3DTerm::ExperimentalTorsion,
                                                      ETK3DTerm::Distance12,
                                                      ETK3DTerm::Distance13,
                                                      ETK3DTerm::Angle13,
                                                      ETK3DTerm::LongRangeDistance};

// Helper function to get individual 3D ETKDG term energies
std::vector<double> getETK3DEnergyTerms(nvMolKit::DistGeom::BatchedMolecular3DDeviceBuffers& deviceFF,
                                        const nvMolKit::AsyncDeviceVector<int>&              atomStartsDevice,
                                        const nvMolKit::AsyncDeviceVector<double>&           positionsDevice,
                                        const ETK3DTerm&                                     term) {
  // Zero out energy buffer first
  deviceFF.energyBuffer.zero();
  deviceFF.energyOuts.zero();

  nvMolKit::DistGeom::ETKTerm etkTerm;
  switch (term) {
    case ETK3DTerm::ExperimentalTorsion:
      etkTerm = nvMolKit::DistGeom::ETKTerm::EXPERIMANTAL_TORSION;
      break;
    case ETK3DTerm::ImproperTorsion:
      etkTerm = nvMolKit::DistGeom::ETKTerm::IMPPROPER_TORSION;
      break;
    case ETK3DTerm::Distance12:
      etkTerm = nvMolKit::DistGeom::ETKTerm::DISTANCE_12;
      break;
    case ETK3DTerm::Distance13:
      etkTerm = nvMolKit::DistGeom::ETKTerm::DISTANCE_13;
      break;
    case ETK3DTerm::Angle13:
      etkTerm = nvMolKit::DistGeom::ETKTerm::ANGLE_13;
      break;
    case ETK3DTerm::LongRangeDistance:
      etkTerm = nvMolKit::DistGeom::ETKTerm::LONGDISTANCE;
      break;
    default:
      throw std::invalid_argument("Unknown ETK3DTerm");
  }

  CHECK_CUDA_RETURN(
    nvMolKit::DistGeom::computeEnergyETK(deviceFF, atomStartsDevice, positionsDevice, nullptr, nullptr, etkTerm));

  std::vector<double> energy(deviceFF.energyOuts.size(), 0.0);
  deviceFF.energyOuts.copyToHost(energy);
  return energy;
}

// Helper function to get individual 3D ETKDG term gradients
std::vector<double> getETK3DGradientTerm(nvMolKit::DistGeom::BatchedMolecular3DDeviceBuffers& deviceFF,
                                         const nvMolKit::AsyncDeviceVector<int>&              atomStartsDevice,
                                         const nvMolKit::AsyncDeviceVector<double>&           positionsDevice,
                                         const ETK3DTerm&                                     term) {
  // Zero out gradients first
  deviceFF.grad.zero();

  nvMolKit::DistGeom::ETKTerm etkTerm;
  switch (term) {
    case ETK3DTerm::ExperimentalTorsion:
      etkTerm = nvMolKit::DistGeom::ETKTerm::EXPERIMANTAL_TORSION;
      break;
    case ETK3DTerm::ImproperTorsion:
      etkTerm = nvMolKit::DistGeom::ETKTerm::IMPPROPER_TORSION;
      break;
    case ETK3DTerm::Distance12:
      etkTerm = nvMolKit::DistGeom::ETKTerm::DISTANCE_12;
      break;
    case ETK3DTerm::Distance13:
      etkTerm = nvMolKit::DistGeom::ETKTerm::DISTANCE_13;
      break;
    case ETK3DTerm::Angle13:
      etkTerm = nvMolKit::DistGeom::ETKTerm::ANGLE_13;
      break;
    case ETK3DTerm::LongRangeDistance:
      etkTerm = nvMolKit::DistGeom::ETKTerm::LONGDISTANCE;
      break;
    default:
      throw std::invalid_argument("Unknown ETK3DTerm");
  }

  CHECK_CUDA_RETURN(
    nvMolKit::DistGeom::computeGradientsETK(deviceFF, atomStartsDevice, positionsDevice, nullptr, etkTerm));

  std::vector<double> grad(deviceFF.grad.size(), 0.0);
  deviceFF.grad.copyToHost(grad);
  cudaDeviceSynchronize();
  return grad;
}

std::vector<std::vector<double>> splitCombinedGrads(const std::vector<double>& combinedGrads,
                                                    const std::vector<int>&    atomStarts) {
  std::vector<std::vector<double>> splitGrads;
  splitGrads.reserve(atomStarts.size() - 1);
  for (size_t i = 0; i < atomStarts.size() - 1; ++i) {
    int start = atomStarts[i];
    int end   = atomStarts[i + 1];
    splitGrads.emplace_back(combinedGrads.begin() + start * 4, combinedGrads.begin() + end * 4);
  }
  return splitGrads;
}

std::vector<double> pad3Dto4D(const std::vector<double>& positions, double value);

// Reference implementation for 3D ETKDG terms
std::unique_ptr<ForceFields::ForceField> setup3DReferenceFF(
  RDKit::ROMol*                        mol,
  const ETK3DTerm&                     term,
  const DGeomHelpers::EmbedParameters& params = DGeomHelpers::ETKDGv3) {
  // Get 3D positions from conformer
  auto& conf = mol->getConformer();

  // Create reference force field
  auto referenceForceField = std::make_unique<ForceFields::ForceField>(3);

  // Add positions as Point3D
  for (unsigned int i = 0; i < conf.getNumAtoms(); ++i) {
    auto* pos = &conf.getAtomPos(i);
    referenceForceField->positions().push_back(pos);
  }
  referenceForceField->initialize();

  // Set up 3D force field parameters
  nvMolKit::detail::EmbedArgs eargs;
  nvMolKit::DGeomHelpers::prepareEmbedderArgs(*mol, params, eargs);
  const auto&            etkdgDetails = eargs.etkdgDetails;
  // Build reference force field for combined energy/grad
  RDGeom::Point3DPtrVect point3DVec;
  for (unsigned int j = 0; j < conf.getNumAtoms(); ++j) {
    point3DVec.push_back(&conf.getAtomPos(j));
  }

  // Add specific term contributions based on term type
  unsigned int            N = mol->getNumAtoms();
  boost::dynamic_bitset<> atomPairs(N * N);
  boost::dynamic_bitset<> isImproperConstrained(N);

  switch (term) {
    case ETK3DTerm::ExperimentalTorsion:
      addExperimentalTorsionTerms(referenceForceField.get(), etkdgDetails, atomPairs, N);
      break;
    case ETK3DTerm::ImproperTorsion:
      addImproperTorsionTerms(referenceForceField.get(), 10.0, etkdgDetails.improperAtoms, isImproperConstrained);
      break;
    case ETK3DTerm::Distance12:
      add12Terms(referenceForceField.get(), etkdgDetails, atomPairs, point3DVec, KNOWN_DIST_FORCE_CONSTANT, N);
      break;
    case ETK3DTerm::Distance13: {
      // Need to set up improper constraints first for 1-3 terms
      addImproperTorsionTerms(referenceForceField.get(), 10.0, etkdgDetails.improperAtoms, isImproperConstrained);
      // Clear existing contribs to only test 1-3 terms
      referenceForceField->contribs().clear();
      add13Terms(referenceForceField.get(),
                 etkdgDetails,
                 atomPairs,
                 point3DVec,
                 KNOWN_DIST_FORCE_CONSTANT,
                 isImproperConstrained,
                 params.useBasicKnowledge,
                 *eargs.mmat,
                 N);
      std::vector<int> removeIndices;
      for (auto& contrib : referenceForceField->contribs()) {
        auto distanceContrib = dynamic_cast<const ForceFields::DistanceConstraintContribs*>(contrib.get());
        if (!distanceContrib) {
          removeIndices.push_back(std::distance(
            referenceForceField->contribs().begin(),
            std::find(referenceForceField->contribs().begin(), referenceForceField->contribs().end(), contrib)));
        }
      }
      for (int idx = removeIndices.size() - 1; idx >= 0; --idx) {
        referenceForceField->contribs().erase(referenceForceField->contribs().begin() + removeIndices[idx]);
      }
      break;
    }
    case ETK3DTerm::Angle13: {
      // Similar to Distance13 but focusing on angle terms
      addImproperTorsionTerms(referenceForceField.get(), 10.0, etkdgDetails.improperAtoms, isImproperConstrained);
      add13Terms(referenceForceField.get(),
                 etkdgDetails,
                 atomPairs,
                 point3DVec,
                 KNOWN_DIST_FORCE_CONSTANT,
                 isImproperConstrained,
                 params.useBasicKnowledge,
                 *eargs.mmat,
                 N);
      std::vector<int> removeIndices;
      for (auto& contrib : referenceForceField->contribs()) {
        auto angleContrib = dynamic_cast<const ForceFields::AngleConstraintContribs*>(contrib.get());
        if (!angleContrib) {
          removeIndices.push_back(std::distance(
            referenceForceField->contribs().begin(),
            std::find(referenceForceField->contribs().begin(), referenceForceField->contribs().end(), contrib)));
        }
      }
      for (int idx = removeIndices.size() - 1; idx >= 0; --idx) {
        referenceForceField->contribs().erase(referenceForceField->contribs().begin() + removeIndices[idx]);
      }
      break;
    }
    case ETK3DTerm::LongRangeDistance:
      // Use the isolated marking function to properly mark atom pairs without adding contributions
      markAtomPairsForIsolatedLongRange(etkdgDetails, N, atomPairs, isImproperConstrained);
      addLongRangeDistanceConstraints(referenceForceField.get(),
                                      etkdgDetails,
                                      atomPairs,
                                      point3DVec,
                                      KNOWN_DIST_FORCE_CONSTANT,
                                      *eargs.mmat,
                                      N);
      break;
  }

  return referenceForceField;
}

std::unique_ptr<ForceFields::ForceField> setupPlain3DReferenceFF(
  RDKit::ROMol*                        mol,
  const DGeomHelpers::EmbedParameters& params = DGeomHelpers::ETDG) {
  auto& conf = mol->getConformer();

  auto referenceForceField = std::make_unique<ForceFields::ForceField>(3);
  for (unsigned int i = 0; i < conf.getNumAtoms(); ++i) {
    auto* pos = &conf.getAtomPos(i);
    referenceForceField->positions().push_back(pos);
  }
  referenceForceField->initialize();

  auto modifiedParams            = params;
  modifiedParams.useRandomCoords = true;

  nvMolKit::detail::EmbedArgs eargs;
  nvMolKit::DGeomHelpers::prepareEmbedderArgs(*mol, modifiedParams, eargs);
  const auto& etkdgDetails = eargs.etkdgDetails;

  RDGeom::Point3DPtrVect point3DVec;
  for (unsigned int j = 0; j < conf.getNumAtoms(); ++j) {
    point3DVec.push_back(&conf.getAtomPos(j));
  }

  const unsigned int      numAtoms = mol->getNumAtoms();
  boost::dynamic_bitset<> atomPairs(numAtoms * numAtoms);
  boost::dynamic_bitset<> isImproperConstrained(numAtoms);

  addExperimentalTorsionTerms(referenceForceField.get(), etkdgDetails, atomPairs, numAtoms);
  add12Terms(referenceForceField.get(), etkdgDetails, atomPairs, point3DVec, KNOWN_DIST_FORCE_CONSTANT, numAtoms);
  add13Terms(referenceForceField.get(),
             etkdgDetails,
             atomPairs,
             point3DVec,
             KNOWN_DIST_FORCE_CONSTANT,
             isImproperConstrained,
             modifiedParams.useBasicKnowledge,
             *eargs.mmat,
             numAtoms);
  addLongRangeDistanceConstraints(referenceForceField.get(),
                                  etkdgDetails,
                                  atomPairs,
                                  point3DVec,
                                  KNOWN_DIST_FORCE_CONSTANT,
                                  *eargs.mmat,
                                  numAtoms);

  return referenceForceField;
}

double getReferencePlain3DEnergy(RDKit::ROMol* mol, double* ePos) {
  auto FF = setupPlain3DReferenceFF(mol);
  return FF->calcEnergy(ePos);
}

std::vector<double> getReferencePlain3DGradient(RDKit::ROMol* mol, double* fPos) {
  auto                FF = setupPlain3DReferenceFF(mol);
  std::vector<double> gradients(3 * mol->getNumAtoms(), 0.0);
  FF->calcGrad(fPos, gradients.data());
  return pad3Dto4D(gradients, 0.0);
}

std::vector<double> pad3Dto4D(const std::vector<double>& positions, const double value = 0.0) {
  std::vector<double> paddedPositions(positions.size() * 4 / 3);
  int                 atomIdx = 0;
  for (size_t i = 0; i < positions.size() / 3; i++) {
    paddedPositions[i * 4]     = positions[i * 3];      // x
    paddedPositions[i * 4 + 1] = positions[i * 3 + 1];  // y
    paddedPositions[i * 4 + 2] = positions[i * 3 + 2];  // z
    paddedPositions[i * 4 + 3] = value;                 // w, set to a constant value
    atomIdx++;
  }
  return paddedPositions;
}

double getReferenceETK3DEnergyTerm(RDKit::ROMol*                        mol,
                                   const ETK3DTerm&                     term,
                                   double*                              ePos,
                                   const DGeomHelpers::EmbedParameters& params = DGeomHelpers::ETKDGv3) {
  auto FF = setup3DReferenceFF(mol, term, params);
  return FF->calcEnergy(ePos);
}

std::vector<double> getReferenceETK3DGradientTerm(RDKit::ROMol*                        mol,
                                                  const ETK3DTerm&                     term,
                                                  double*                              fPos,
                                                  const DGeomHelpers::EmbedParameters& params = DGeomHelpers::ETKDGv3) {
  auto                FF = setup3DReferenceFF(mol, term, params);
  std::vector<double> gradients(3 * mol->getNumAtoms(), 0.0);
  FF->calcGrad(fPos, gradients.data());
  return pad3Dto4D(gradients);
}

std::vector<double> getReferenceETK3DEnergyTerms(const std::vector<RDKit::ROMol*>     mols,
                                                 const ETK3DTerm&                     term,
                                                 double*                              ePos,
                                                 const std::vector<int>&              atomStarts,
                                                 const DGeomHelpers::EmbedParameters& params = DGeomHelpers::ETKDGv3) {
  std::vector<double> results;
  for (size_t i = 0; i < mols.size(); i++) {
    const auto FF = setup3DReferenceFF(mols[i], term, params);
    results.push_back(FF->calcEnergy(ePos + 3 * atomStarts[i]));
  }
  return results;
}

std::vector<std::vector<double>> getReferenceETK3DGradientTerms(
  const std::vector<RDKit::ROMol*>     mols,
  const ETK3DTerm&                     term,
  double*                              fPos,
  const std::vector<int>&              atomStarts,
  const DGeomHelpers::EmbedParameters& params = DGeomHelpers::ETKDGv3) {
  std::vector<std::vector<double>> results;
  for (size_t i = 0; i < mols.size(); i++) {
    const auto          FF = setup3DReferenceFF(mols[i], term, params);
    std::vector<double> gradients(3 * mols[i]->getNumAtoms(), 0.0);
    FF->calcGrad(fPos + 3 * atomStarts[i], gradients.data());
    results.push_back(pad3Dto4D(gradients));
  }
  return results;
}

std::vector<double> getEnergyTerm(nvMolKit::DistGeom::BatchedMolecularDeviceBuffers& deviceFF,
                                  const double*                                      pos,
                                  const int*                                         atomStarts,
                                  const FFTerm&                                      term,
                                  const int                                          dim) {
  switch (term) {
    case FFTerm::DistanceViolation:
      if (deviceFF.contribs.distTerms.idx1.size() == 0) {
        return {0.0};
      }
      CHECK_CUDA_RETURN(nvMolKit::DistGeom::launchDistViolationEnergyKernel(deviceFF.contribs.distTerms.idx1.size(),
                                                                            deviceFF.contribs.distTerms.idx1.data(),
                                                                            deviceFF.contribs.distTerms.idx2.data(),
                                                                            deviceFF.contribs.distTerms.lb2.data(),
                                                                            deviceFF.contribs.distTerms.ub2.data(),
                                                                            deviceFF.contribs.distTerms.weight.data(),
                                                                            pos,
                                                                            deviceFF.energyBuffer.data(),
                                                                            deviceFF.indices.energyBufferStarts.data(),
                                                                            deviceFF.indices.atomIdxToBatchIdx.data(),
                                                                            deviceFF.indices.distTermStarts.data(),
                                                                            atomStarts,
                                                                            dim));
      break;
    case FFTerm::ChiralViolation:
      if (deviceFF.contribs.chiralTerms.idx1.size() == 0) {
        return {0.0};
      }
      CHECK_CUDA_RETURN(
        nvMolKit::DistGeom::launchChiralViolationEnergyKernel(deviceFF.contribs.chiralTerms.idx1.size(),
                                                              deviceFF.contribs.chiralTerms.idx1.data(),
                                                              deviceFF.contribs.chiralTerms.idx2.data(),
                                                              deviceFF.contribs.chiralTerms.idx3.data(),
                                                              deviceFF.contribs.chiralTerms.idx4.data(),
                                                              deviceFF.contribs.chiralTerms.volLower.data(),
                                                              deviceFF.contribs.chiralTerms.volUpper.data(),
                                                              1.0,
                                                              pos,
                                                              deviceFF.energyBuffer.data(),
                                                              deviceFF.indices.energyBufferStarts.data(),
                                                              deviceFF.indices.atomIdxToBatchIdx.data(),
                                                              deviceFF.indices.chiralTermStarts.data(),
                                                              atomStarts,
                                                              dim));
      break;
    case FFTerm::FourthDim:
      if (deviceFF.contribs.fourthTerms.idx.size() == 0) {
        return {0.0};
      }
      CHECK_CUDA_RETURN(nvMolKit::DistGeom::launchFourthDimEnergyKernel(deviceFF.contribs.fourthTerms.idx.size(),
                                                                        deviceFF.contribs.fourthTerms.idx.data(),
                                                                        0.1,
                                                                        pos,
                                                                        deviceFF.energyBuffer.data(),
                                                                        deviceFF.indices.energyBufferStarts.data(),
                                                                        deviceFF.indices.atomIdxToBatchIdx.data(),
                                                                        deviceFF.indices.fourthTermStarts.data(),
                                                                        atomStarts,
                                                                        dim));
      break;
  }
  CHECK_CUDA_RETURN(
    nvMolKit::DistGeom::launchReduceEnergiesKernel(deviceFF.indices.energyBufferBlockIdxToBatchIdx.size(),
                                                   deviceFF.energyBuffer.data(),
                                                   deviceFF.indices.energyBufferBlockIdxToBatchIdx.data(),
                                                   deviceFF.energyOuts.data()));
  std::vector<double> energy(deviceFF.energyOuts.size());
  deviceFF.energyOuts.copyToHost(energy);
  return energy;
}

std::vector<double> getGradientTerm(nvMolKit::DistGeom::BatchedMolecularDeviceBuffers& deviceFF,
                                    const double*                                      pos,
                                    const int*                                         atomStarts,
                                    const FFTerm&                                      term,
                                    const int                                          dim) {
  switch (term) {
    case FFTerm::DistanceViolation:
      if (deviceFF.contribs.distTerms.idx1.size() == 0) {
        break;
      }
      CHECK_CUDA_RETURN(nvMolKit::DistGeom::launchDistViolationGradientKernel(deviceFF.contribs.distTerms.idx1.size(),
                                                                              deviceFF.contribs.distTerms.idx1.data(),
                                                                              deviceFF.contribs.distTerms.idx2.data(),
                                                                              deviceFF.contribs.distTerms.lb2.data(),
                                                                              deviceFF.contribs.distTerms.ub2.data(),
                                                                              deviceFF.contribs.distTerms.weight.data(),
                                                                              pos,
                                                                              deviceFF.grad.data(),
                                                                              deviceFF.indices.atomIdxToBatchIdx.data(),
                                                                              atomStarts,
                                                                              dim));
      break;
    case FFTerm::ChiralViolation:
      if (deviceFF.contribs.chiralTerms.idx1.size() == 0) {
        break;
      }
      CHECK_CUDA_RETURN(
        nvMolKit::DistGeom::launchChiralViolationGradientKernel(deviceFF.contribs.chiralTerms.idx1.size(),
                                                                deviceFF.contribs.chiralTerms.idx1.data(),
                                                                deviceFF.contribs.chiralTerms.idx2.data(),
                                                                deviceFF.contribs.chiralTerms.idx3.data(),
                                                                deviceFF.contribs.chiralTerms.idx4.data(),
                                                                deviceFF.contribs.chiralTerms.volLower.data(),
                                                                deviceFF.contribs.chiralTerms.volUpper.data(),
                                                                1.0,
                                                                pos,
                                                                deviceFF.grad.data(),
                                                                deviceFF.indices.atomIdxToBatchIdx.data(),
                                                                atomStarts,
                                                                dim));
      break;
    case FFTerm::FourthDim:
      if (deviceFF.contribs.fourthTerms.idx.size() == 0) {
        break;
      }
      CHECK_CUDA_RETURN(nvMolKit::DistGeom::launchFourthDimGradientKernel(deviceFF.contribs.fourthTerms.idx.size(),
                                                                          deviceFF.contribs.fourthTerms.idx.data(),
                                                                          0.1,
                                                                          pos,
                                                                          deviceFF.grad.data(),
                                                                          deviceFF.indices.atomIdxToBatchIdx.data(),
                                                                          atomStarts,
                                                                          dim));
      break;
  }
  std::vector<double> grad(deviceFF.grad.size(), 0.0);
  deviceFF.grad.copyToHost(grad);
  cudaDeviceSynchronize();
  return grad;
}

void addDistViolationContribs(ForceFields::ForceField&      field,
                              const DistGeom::BoundsMatrix& mmat,
                              const double                  basinSizeTol) {
  auto               contrib = std::make_unique<DistGeom::DistViolationContribs>(&field);
  const unsigned int N       = mmat.numRows();
  for (unsigned int i = 1; i < N; i++) {
    for (unsigned int j = 0; j < i; j++) {
      const double l         = mmat.getLowerBound(i, j);
      const double u         = mmat.getUpperBound(i, j);
      bool         includeIt = false;
      if (u - l <= basinSizeTol) {
        includeIt = true;
      }
      if (includeIt) {
        constexpr double w = 1.0;
        contrib->addContrib(i, j, u, l, w);
      }
    }
  }
  field.contribs().push_back(ForceFields::ContribPtr(contrib.release()));
}

void addChiralViolationContribs(ForceFields::ForceField&        field,
                                const DistGeom::VECT_CHIRALSET& csets,
                                const double                    weight) {
  auto contrib = std::make_unique<DistGeom::ChiralViolationContribs>(&field);
  for (const auto& cset : csets) {
    contrib->addContrib(cset.get(), weight);
  }
  field.contribs().push_back(ForceFields::ContribPtr(contrib.release()));
}

double getReferenceEnergyTerm(const nvMolKit::detail::EmbedArgs& eargs,
                              RDGeom::PointPtrVect&              pos,
                              const FFTerm&                      term,
                              const double                       basinTol) {
  const auto referenceForceField = std::make_unique<ForceFields::ForceField>(4);
  referenceForceField->positions().insert(referenceForceField->positions().begin(), pos.begin(), pos.end());

  referenceForceField->initialize();
  if (term == FFTerm::DistanceViolation) {
    addDistViolationContribs(*referenceForceField, *eargs.mmat, basinTol);
  } else if (term == FFTerm::ChiralViolation) {
    addChiralViolationContribs(*referenceForceField, eargs.chiralCenters, 1.0);
  } else if (term == FFTerm::FourthDim) {
    auto contrib = std::make_unique<DistGeom::FourthDimContribs>(referenceForceField.get());
    for (size_t i = 0; i < pos.size(); ++i) {
      contrib->addContrib(i, 0.1);  // Use z-coordinate as the 4th dimension
    }
    referenceForceField->contribs().push_back(ForceFields::ContribPtr(contrib.release()));
  }
  return referenceForceField->calcEnergy();
}

std::vector<double> getReferenceGradientTerm(const nvMolKit::detail::EmbedArgs& eargs,
                                             RDGeom::PointPtrVect&              pos,
                                             const FFTerm&                      term,
                                             const double                       basinTol) {
  auto referenceForceField = std::make_unique<ForceFields::ForceField>(4);
  referenceForceField->positions().insert(referenceForceField->positions().begin(), pos.begin(), pos.end());
  referenceForceField->initialize();

  if (term == FFTerm::DistanceViolation) {
    addDistViolationContribs(*referenceForceField, *eargs.mmat, basinTol);
  } else if (term == FFTerm::ChiralViolation) {
    addChiralViolationContribs(*referenceForceField, eargs.chiralCenters, 1.0);
  } else if (term == FFTerm::FourthDim) {
    auto contrib = std::make_unique<DistGeom::FourthDimContribs>(referenceForceField.get());
    for (size_t i = 0; i < pos.size(); ++i) {
      contrib->addContrib(i, 0.1);  // Use z-coordinate as the 4th dimension
    }
    referenceForceField->contribs().push_back(ForceFields::ContribPtr(contrib.release()));
  }
  std::vector<double> gradients(referenceForceField->dimension() * eargs.mmat->numRows(), 0.0);
  referenceForceField->calcGrad(gradients.data());
  return gradients;
}

// Multi-molecule helper functions for 4D DG terms
std::vector<double> getReferenceEnergyTerms(const std::vector<nvMolKit::detail::EmbedArgs>& allEargs,
                                            const std::vector<RDGeom::PointPtrVect>&        allPointVecHolders,
                                            const FFTerm&                                   term,
                                            const double                                    basinTol) {
  std::vector<double> results;
  results.reserve(allEargs.size());

  for (size_t i = 0; i < allEargs.size(); ++i) {
    const auto& eargs          = allEargs[i];
    auto        pointVecHolder = allPointVecHolders[i];  // Make a copy to avoid const issues
    double      energy         = getReferenceEnergyTerm(eargs, pointVecHolder, term, basinTol);
    results.push_back(energy);
  }

  return results;
}

std::vector<std::vector<double>> getReferenceGradientTerms(const std::vector<nvMolKit::detail::EmbedArgs>& allEargs,
                                                           const std::vector<RDGeom::PointPtrVect>& allPointVecHolders,
                                                           const FFTerm&                            term,
                                                           const double                             basinTol) {
  std::vector<std::vector<double>> results;
  results.reserve(allEargs.size());

  for (size_t i = 0; i < allEargs.size(); ++i) {
    const auto&         eargs          = allEargs[i];
    auto                pointVecHolder = allPointVecHolders[i];  // Make a copy to avoid const issues
    std::vector<double> gradients      = getReferenceGradientTerm(eargs, pointVecHolder, term, basinTol);
    results.push_back(gradients);
  }

  return results;
}

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

void perturbConformer(std::vector<double>& conf, const float delta = 0.1, const int seed = 0) {
  std::mt19937                          gen(seed);  // Mersenne Twister engine
  std::uniform_real_distribution<float> dist(-delta, delta);
  for (double& val : conf) {
    val += delta * dist(gen);
  }
}

void perturbConformerCpuGpuPair(std::vector<double>&  positions,
                                RDGeom::POINT3D_VECT& molPos,
                                const double          delta = 0.1,
                                const double          seed  = 0) {
  const int dim = positions.size() / molPos.size();

  std::mt19937                          gen(seed);  // Mersenne Twister engine
  std::uniform_real_distribution<float> dist(-delta, delta);
  for (unsigned int i = 0; i < molPos.size(); ++i) {
    auto& pos = molPos[i];

    double& xPos = positions[dim * i + 0];
    double& yPos = positions[dim * i + 1];
    double& zPos = positions[dim * i + 2];

    if (pos.x != xPos || pos.y != yPos || pos.z != zPos) {
      throw std::invalid_argument("Positions do not match molPos");
    }

    const double dx = dist(gen);
    const double dy = dist(gen);
    const double dz = dist(gen);
    pos.x += delta * dx;
    pos.y += delta * dy;
    pos.z += delta * dz;
    xPos += delta * dx;
    yPos += delta * dy;
    zPos += delta * dz;
  }
}

class DistGeomDGKernelTestFixture : public ::testing::Test {
 public:
  DistGeomDGKernelTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

  void SetUp() override {
    // Minimal setup - just initialize options
    options                 = RDKit::DGeomHelpers::ETKDGv3;
    options.useRandomCoords = true;
  }

 protected:
  void loadSingleMol() {
    const std::string mol2FilePath = testDataFolderPath_ + "/50_atom_mol.sdf";
    ASSERT_TRUE(std::filesystem::exists(mol2FilePath)) << "Could not find " << mol2FilePath;
    mol_ = std::unique_ptr<RDKit::RWMol>(RDKit::MolFileToMol(mol2FilePath, false));
    ASSERT_NE(mol_, nullptr);
    RDKit::MolOps::sanitizeMol(*mol_);
    perturbConformer(mol_->getConformer(), 1.0);
    std::vector<RDKit::ROMol*> molBatch = {mol_.get()};
    setUpSystems(molBatch);
  }

  void loadMMFFMols(int count) {
    getMols(testDataFolderPath_ + "/MMFF94_dative.sdf", mols_, /*count=*/count);

    for (auto& mol : mols_) {
      molsPtrs_.push_back(mol.get());
    }
    setUpSystems(molsPtrs_);
  }

  void setUpSystems(const std::vector<RDKit::ROMol*>& molBatch);

  std::string                                testDataFolderPath_;
  std::unique_ptr<RDKit::RWMol>              mol_;
  std::vector<std::unique_ptr<RDKit::ROMol>> mols_;
  std::vector<RDKit::ROMol*>                 molsPtrs_;  // For passing to setUpSystems

  nvMolKit::AsyncDeviceVector<double>      positionsDevice;
  nvMolKit::AsyncDeviceVector<int>         atomStartsDevice;
  std::vector<nvMolKit::detail::EmbedArgs> allEargs_;

  nvMolKit::DistGeom::BatchedMolecularSystemHost    systemHost;
  nvMolKit::DistGeom::BatchedMolecularDeviceBuffers systemDevice;
  std::vector<std::unique_ptr<RDGeom::PointND>>     pointVec;
  std::vector<RDGeom::PointPtrVect>                 allPointVecHolders;
  DGeomHelpers::EmbedParameters                     options;

  std::vector<double> positionsHost;
  std::vector<int>    atomStartsHost;
};

void DistGeomDGKernelTestFixture::setUpSystems(const std::vector<RDKit::ROMol*>& molBatch) {
  atomStartsHost = {0};
  allEargs_.clear();
  allPointVecHolders.clear();

  for (auto* mol : molBatch) {
    // For each molecule, generate random coordinates and set up the force field
    auto rwMol = dynamic_cast<RDKit::RWMol*>(mol);
    ASSERT_NE(rwMol, nullptr);
    if (rwMol->getNumConformers() == 0) {
      // Add a random conformer if none exists
      DGeomHelpers::EmbedParameters tempParams = DGeomHelpers::ETKDGv3;
      tempParams.useRandomCoords               = true;
      DGeomHelpers::EmbedMolecule(*rwMol, tempParams);
    }

    perturbConformer(rwMol->getConformer(), 1.0);

    std::vector<std::unique_ptr<RDGeom::Point>> positions;
    std::unique_ptr<ForceFields::ForceField>    field;

    // Setup force field using ETKDGv3
    options                 = RDKit::DGeomHelpers::ETKDGv3;
    options.useRandomCoords = true;

    nvMolKit::detail::EmbedArgs& eargs = allEargs_.emplace_back();
    eargs.dim                          = 4;
    nvMolKit::DGeomHelpers::setupRDKitFFWithPos(rwMol,
                                                options,
                                                field,
                                                eargs,
                                                positions,
                                                -1,
                                                nvMolKit::DGeomHelpers::Dimensionality::DIM_4D);

    // Set up 4th dimension values first
    std::vector<double> scrambled4thDimValues;
    for (size_t i = 0; i < rwMol->getNumAtoms(); ++i) {
      scrambled4thDimValues.push_back(rwMol->getConformer().getAtomPos(i).z);
    }

    // Update the 4th dimension in the position vectors
    for (size_t i = 0; i < positions.size(); ++i) {
      (*positions[i])[3] = scrambled4thDimValues[i];
    }

    // Update the 4th dimension in eargs.posVec
    for (size_t i = 0; i < scrambled4thDimValues.size(); i++) {
      eargs.posVec[i * 4 + 3] = scrambled4thDimValues[i];
    }

    // Construct force field contributions and add to batch
    const auto ffParams = nvMolKit::DistGeom::constructForceFieldContribs(eargs.dim,
                                                                          *eargs.mmat,
                                                                          eargs.chiralCenters,
                                                                          1.0,
                                                                          0.1,
                                                                          nullptr,
                                                                          options.basinThresh);
    addMoleculeToBatch(ffParams, eargs.posVec, systemHost, eargs.dim, atomStartsHost, positionsHost);

    // Set up point vectors for this molecule
    RDGeom::PointPtrVect molPointVecHolder;
    int                  atomIdx = 0;
    for (const auto& pos : positions) {
      pointVec.push_back(std::make_unique<RDGeom::PointND>(4));
      molPointVecHolder.push_back(pointVec.back().get());
      for (int i = 0; i < eargs.dim; ++i) {
        (*pointVec.back())[i] = (*pos)[i];
      }
      (*pointVec.back())[eargs.dim - 1] = scrambled4thDimValues[atomIdx];
      atomIdx++;
    }

    allPointVecHolders.push_back(molPointVecHolder);
  }

  // Set up device buffers
  sendContribsAndIndicesToDevice(systemHost, systemDevice);
  nvMolKit::DistGeom::sendContextToDevice(positionsHost, positionsDevice, atomStartsHost, atomStartsDevice);
  setupDeviceBuffers(systemHost, systemDevice, positionsHost, atomStartsHost.size() - 1);

  // Set up gradients buffer
  int totalAtoms = atomStartsHost.back();
  systemDevice.grad.resize(totalAtoms * 4);
  systemDevice.grad.zero();
}

TEST_F(DistGeomDGKernelTestFixture, DistViolationEnergySingleMolecule) {
  loadSingleMol();
  const double wantEnergy =
    getReferenceEnergyTerm(allEargs_[0], allPointVecHolders[0], FFTerm::DistanceViolation, options.basinThresh);
  const double gotEnergy =
    getEnergyTerm(systemDevice, positionsDevice.data(), atomStartsDevice.data(), FFTerm::DistanceViolation, 4)[0];

  std::vector<double> hostPos;
  hostPos.resize(allPointVecHolders[0].size() * 4);
  positionsDevice.copyToHost(hostPos);
  cudaDeviceSynchronize();

  EXPECT_NEAR(gotEnergy, wantEnergy, E_TOL_FUNCTION);
}

TEST_F(DistGeomDGKernelTestFixture, DistViolationSingleMolecule) {
  loadSingleMol();
  const std::vector<double> wantGradients =
    getReferenceGradientTerm(allEargs_[0], allPointVecHolders[0], FFTerm::DistanceViolation, options.basinThresh);
  const std::vector<double> gotGrad =
    getGradientTerm(systemDevice, positionsDevice.data(), atomStartsDevice.data(), FFTerm::DistanceViolation, 4);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients));
}

TEST_F(DistGeomDGKernelTestFixture, ChiralViolationEnergySingleMolecule) {
  loadSingleMol();
  const double wantEnergy =
    getReferenceEnergyTerm(allEargs_[0], allPointVecHolders[0], FFTerm::ChiralViolation, options.basinThresh);
  const double gotEnergy =
    getEnergyTerm(systemDevice, positionsDevice.data(), atomStartsDevice.data(), FFTerm::ChiralViolation, 4)[0];

  std::vector<double> hostPos;
  hostPos.resize(allPointVecHolders[0].size() * 4);
  positionsDevice.copyToHost(hostPos);
  cudaDeviceSynchronize();

  EXPECT_NEAR(gotEnergy, wantEnergy, E_TOL_FUNCTION);
}

TEST_F(DistGeomDGKernelTestFixture, ChiralViolationGradSingleMolecule) {
  loadSingleMol();
  const std::vector<double> wantGradients =
    getReferenceGradientTerm(allEargs_[0], allPointVecHolders[0], FFTerm::ChiralViolation, options.basinThresh);
  const std::vector<double> gotGrad =
    getGradientTerm(systemDevice, positionsDevice.data(), atomStartsDevice.data(), FFTerm::ChiralViolation, 4);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients));
}

TEST_F(DistGeomDGKernelTestFixture, FourthDimEnergySingleMolecule) {
  loadSingleMol();
  const double wantEnergy =
    getReferenceEnergyTerm(allEargs_[0], allPointVecHolders[0], FFTerm::FourthDim, options.basinThresh);
  const double gotEnergy =
    getEnergyTerm(systemDevice, positionsDevice.data(), atomStartsDevice.data(), FFTerm::FourthDim, 4)[0];

  std::vector<double> hostPos;
  hostPos.resize(allPointVecHolders[0].size() * 4);
  positionsDevice.copyToHost(hostPos);
  cudaDeviceSynchronize();

  EXPECT_NEAR(gotEnergy, wantEnergy, E_TOL_FUNCTION);
}

TEST_F(DistGeomDGKernelTestFixture, FourthDimGradSingleMolecule) {
  loadSingleMol();
  const std::vector<double> wantGradients =
    getReferenceGradientTerm(allEargs_[0], allPointVecHolders[0], FFTerm::FourthDim, options.basinThresh);
  const std::vector<double> gotGrad =
    getGradientTerm(systemDevice, positionsDevice.data(), atomStartsDevice.data(), FFTerm::FourthDim, 4);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients));
}

// -------------------------
// Multi-molecule 4D DG tests
// -------------------------

TEST_F(DistGeomDGKernelTestFixture, DistViolationEnergyMultiMol) {
  constexpr auto term = FFTerm::DistanceViolation;
  loadMMFFMols(100);
  std::vector<double> wantEnergy = getReferenceEnergyTerms(allEargs_, allPointVecHolders, term, options.basinThresh);
  std::vector<double> gotEnergy = getEnergyTerm(systemDevice, positionsDevice.data(), atomStartsDevice.data(), term, 4);
  EXPECT_THAT(gotEnergy, ::testing::Pointwise(::testing::DoubleNear(E_TOL_FUNCTION), wantEnergy));
}

TEST_F(DistGeomDGKernelTestFixture, DistViolationGradMultiMol) {
  constexpr auto term = FFTerm::DistanceViolation;
  loadMMFFMols(100);
  std::vector<std::vector<double>> wantGrad =
    getReferenceGradientTerms(allEargs_, allPointVecHolders, term, options.basinThresh);
  std::vector<std::vector<double>> gotGrad =
    splitCombinedGrads(getGradientTerm(systemDevice, positionsDevice.data(), atomStartsDevice.data(), term, 4),
                       atomStartsHost);
  for (size_t i = 0; i < wantGrad.size(); ++i) {
    EXPECT_THAT(gotGrad[i], ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGrad[i])) << "For system " << i;
  }
}

TEST_F(DistGeomDGKernelTestFixture, ChiralViolationEnergyMultiMol) {
  constexpr auto term = FFTerm::ChiralViolation;
  loadMMFFMols(100);
  std::vector<double> wantEnergy = getReferenceEnergyTerms(allEargs_, allPointVecHolders, term, options.basinThresh);
  std::vector<double> gotEnergy = getEnergyTerm(systemDevice, positionsDevice.data(), atomStartsDevice.data(), term, 4);
  EXPECT_THAT(gotEnergy, ::testing::Pointwise(::testing::DoubleNear(E_TOL_FUNCTION), wantEnergy));
}

TEST_F(DistGeomDGKernelTestFixture, ChiralViolationGradMultiMol) {
  constexpr auto term = FFTerm::ChiralViolation;
  loadMMFFMols(100);
  std::vector<std::vector<double>> wantGrad =
    getReferenceGradientTerms(allEargs_, allPointVecHolders, term, options.basinThresh);
  std::vector<std::vector<double>> gotGrad =
    splitCombinedGrads(getGradientTerm(systemDevice, positionsDevice.data(), atomStartsDevice.data(), term, 4),
                       atomStartsHost);
  for (size_t i = 0; i < wantGrad.size(); ++i) {
    EXPECT_THAT(gotGrad[i], ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGrad[i])) << "For system " << i;
  }
}

TEST_F(DistGeomDGKernelTestFixture, FourthDimEnergyMultiMol) {
  constexpr auto term = FFTerm::FourthDim;
  loadMMFFMols(100);
  std::vector<double> wantEnergy = getReferenceEnergyTerms(allEargs_, allPointVecHolders, term, options.basinThresh);
  std::vector<double> gotEnergy = getEnergyTerm(systemDevice, positionsDevice.data(), atomStartsDevice.data(), term, 4);
  EXPECT_THAT(gotEnergy, ::testing::Pointwise(::testing::DoubleNear(E_TOL_FUNCTION), wantEnergy));
}

TEST_F(DistGeomDGKernelTestFixture, FourthDimGradMultiMol) {
  constexpr auto term = FFTerm::FourthDim;
  loadMMFFMols(100);
  std::vector<std::vector<double>> wantGrad =
    getReferenceGradientTerms(allEargs_, allPointVecHolders, term, options.basinThresh);
  std::vector<std::vector<double>> gotGrad =
    splitCombinedGrads(getGradientTerm(systemDevice, positionsDevice.data(), atomStartsDevice.data(), term, 4),
                       atomStartsHost);
  for (size_t i = 0; i < wantGrad.size(); ++i) {
    EXPECT_THAT(gotGrad[i], ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGrad[i])) << "For system " << i;
  }
}

// =============================================================================
// 3D ETKDG TERM TESTS - Similar structure to MMFF tests
// =============================================================================

class ETK3DGpuTestFixture : public ::testing::Test {
 public:
  ETK3DGpuTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

 protected:
  void loadSingleMol(const DGeomHelpers::EmbedParameters& params = DGeomHelpers::ETKDGv3) {
    const std::string mol2FilePath = testDataFolderPath_ + "/rdkit_smallmol_1.mol2";
    ASSERT_TRUE(std::filesystem::exists(mol2FilePath)) << "Could not find " << mol2FilePath;
    mol_ = std::unique_ptr<RDKit::RWMol>(RDKit::MolFileToMol(mol2FilePath, false));
    ASSERT_NE(mol_, nullptr);
    RDKit::MolOps::sanitizeMol(*mol_);
    std::vector<RDKit::ROMol*> molBatch = {mol_.get()};
    setUpSystems(molBatch, params);
  }

  void loadMMFFMols(int count, const DGeomHelpers::EmbedParameters& params = DGeomHelpers::ETKDGv3) {
    getMols(testDataFolderPath_ + "/MMFF94_dative.sdf", mols_, /*count=*/count);

    for (auto& mol : mols_) {
      molsPtrs_.push_back(mol.get());
    }
    setUpSystems(molsPtrs_, params);
  }

  void                                       setUpSystems(const std::vector<RDKit::ROMol*>&    molBatch,
                                                          const DGeomHelpers::EmbedParameters& params = DGeomHelpers::ETKDGv3);
  std::string                                testDataFolderPath_;
  std::unique_ptr<RDKit::RWMol>              mol_;
  std::vector<std::unique_ptr<RDKit::ROMol>> mols_;
  std::vector<RDKit::ROMol*>                 molsPtrs_;  // For passing to setUpSystems

  nvMolKit::DistGeom::BatchedMolecularSystem3DHost    systemHost;
  nvMolKit::DistGeom::BatchedMolecular3DDeviceBuffers systemDevice;
  nvMolKit::AsyncDeviceVector<double>                 positionsDevice;
  nvMolKit::AsyncDeviceVector<int>                    atomStartsDevice;
  std::vector<double>                                 positionsHost;
  std::vector<int>                                    atomStartsHost;
};

void ETK3DGpuTestFixture::setUpSystems(const std::vector<RDKit::ROMol*>&    molBatch,
                                       const DGeomHelpers::EmbedParameters& params) {
  atomStartsHost = {0};
  for (auto* mol : molBatch) {
    auto&               conf = mol->getConformer();
    std::vector<double> molPositions;
    for (unsigned int i = 0; i < conf.getNumAtoms(); ++i) {
      auto pos = conf.getAtomPos(i);
      molPositions.push_back(pos.x);
      molPositions.push_back(pos.y);
      molPositions.push_back(pos.z);
    }

    // Set up 3D force field parameters
    auto modifiedParams            = params;
    modifiedParams.useRandomCoords = true;
    auto args                      = nvMolKit::detail::EmbedArgs();
    nvMolKit::DGeomHelpers::prepareEmbedderArgs(*mol, modifiedParams, args);
    const auto& etkdgDetails = args.etkdgDetails;
    const auto& mmat         = args.mmat;

    // Build reference force field for combined energy/grad
    RDGeom::Point3DPtrVect point3DVec;

    for (unsigned int j = 0; j < conf.getNumAtoms(); ++j) {
      point3DVec.push_back(&conf.getAtomPos(j));
    }

    // Set up GPU system. NOTE: Regardless of 3 or 4D system, the setup uses 3D
    auto ffParams = nvMolKit::DistGeom::construct3DForceFieldContribs(*mmat,
                                                                      etkdgDetails,
                                                                      molPositions,
                                                                      /*dim=*/3,
                                                                      modifiedParams.useBasicKnowledge);
    addMoleculeToBatch3D(ffParams, molPositions, systemHost, atomStartsHost, positionsHost);
  }
  sendContribsAndIndicesToDevice3D(systemHost, systemDevice);

  // Use a nonzero dummy value to make sure the 4th dimension is not used. If it was 0 and we were accidentally
  // using it, test might not fail depending on the nature of the computation.
  perturbConformer(positionsHost, 0.3, 42);
  std::vector<double> paddedPositions = pad3Dto4D(positionsHost, 100.0);

  nvMolKit::DistGeom::sendContextToDevice(paddedPositions, positionsDevice, atomStartsHost, atomStartsDevice);
  setupDeviceBuffers3D(systemHost, systemDevice, paddedPositions, atomStartsHost.size() - 1);
}

// Individual term tests for 3D ETKDG
TEST_F(ETK3DGpuTestFixture, ExperimentalTorsionEnergySingleMolecule) {
  loadSingleMol();
  double wantEnergy = getReferenceETK3DEnergyTerm(mol_.get(), ETK3DTerm::ExperimentalTorsion, positionsHost.data());
  double gotEnergy =
    getETK3DEnergyTerms(systemDevice, atomStartsDevice, positionsDevice, ETK3DTerm::ExperimentalTorsion)[0];
  ASSERT_NE(wantEnergy, 0.0);
  EXPECT_NEAR(gotEnergy, wantEnergy, E_TOL_FUNCTION);
}

TEST_F(ETK3DGpuTestFixture, ExperimentalTorsionGradientSingleMolecule) {
  loadSingleMol();
  std::vector<double> wantGradients =
    getReferenceETK3DGradientTerm(mol_.get(), ETK3DTerm::ExperimentalTorsion, positionsHost.data());
  std::vector<double> gotGrad =
    getETK3DGradientTerm(systemDevice, atomStartsDevice, positionsDevice, ETK3DTerm::ExperimentalTorsion);
  ASSERT_THAT(wantGradients, ::testing::Not(::testing::Each(0.0)));
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients));
}

TEST_F(ETK3DGpuTestFixture, ImproperTorsionEnergySingleMolecule) {
  loadSingleMol();
  double wantEnergy = getReferenceETK3DEnergyTerm(mol_.get(), ETK3DTerm::ImproperTorsion, positionsHost.data());
  double gotEnergy =
    getETK3DEnergyTerms(systemDevice, atomStartsDevice, positionsDevice, ETK3DTerm::ImproperTorsion)[0];
  ASSERT_NE(wantEnergy, 0.0);
  EXPECT_NEAR(gotEnergy, wantEnergy, E_TOL_FUNCTION);
}

TEST_F(ETK3DGpuTestFixture, ImproperTorsionGradientSingleMolecule) {
  loadSingleMol();
  std::vector<double> wantGradients =
    getReferenceETK3DGradientTerm(mol_.get(), ETK3DTerm::ImproperTorsion, positionsHost.data());
  std::vector<double> gotGrad =
    getETK3DGradientTerm(systemDevice, atomStartsDevice, positionsDevice, ETK3DTerm::ImproperTorsion);
  ASSERT_THAT(wantGradients, ::testing::Not(::testing::Each(0.0)));
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients));
}

TEST_F(ETK3DGpuTestFixture, Distance12EnergySingleMolecule) {
  loadSingleMol();
  double wantEnergy = getReferenceETK3DEnergyTerm(mol_.get(), ETK3DTerm::Distance12, positionsHost.data());
  double gotEnergy  = getETK3DEnergyTerms(systemDevice, atomStartsDevice, positionsDevice, ETK3DTerm::Distance12)[0];
  ASSERT_NE(wantEnergy, 0.0);
  EXPECT_NEAR(gotEnergy, wantEnergy, E_TOL_FUNCTION);
}

TEST_F(ETK3DGpuTestFixture, Distance12GradientSingleMolecule) {
  loadSingleMol();
  std::vector<double> wantGradients =
    getReferenceETK3DGradientTerm(mol_.get(), ETK3DTerm::Distance12, positionsHost.data());
  std::vector<double> gotGrad =
    getETK3DGradientTerm(systemDevice, atomStartsDevice, positionsDevice, ETK3DTerm::Distance12);
  ASSERT_THAT(wantGradients, ::testing::Not(::testing::Each(0.0)));
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients));
}

TEST_F(ETK3DGpuTestFixture, Distance13EnergySingleMolecule) {
  loadSingleMol();

  double wantEnergy = getReferenceETK3DEnergyTerm(mol_.get(), ETK3DTerm::Distance13, positionsHost.data());

  double gotEnergy = getETK3DEnergyTerms(systemDevice, atomStartsDevice, positionsDevice, ETK3DTerm::Distance13)[0];
  ASSERT_NE(wantEnergy, 0.0);
  EXPECT_NEAR(gotEnergy, wantEnergy, E_TOL_FUNCTION);
}

TEST_F(ETK3DGpuTestFixture, Distance13GradientSingleMolecule) {
  loadSingleMol();
  std::vector<double> wantGradients =
    getReferenceETK3DGradientTerm(mol_.get(), ETK3DTerm::Distance13, positionsHost.data());
  std::vector<double> gotGrad =
    getETK3DGradientTerm(systemDevice, atomStartsDevice, positionsDevice, ETK3DTerm::Distance13);
  ASSERT_THAT(wantGradients, ::testing::Not(::testing::Each(0.0)));
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients));
}

// FIXME: The test molecule does not have any angle13 terms, so the reference energy is 0.0 and grads are empty.
TEST_F(ETK3DGpuTestFixture, Angle13EnergySingleMolecule) {
  loadSingleMol();
  double wantEnergy = getReferenceETK3DEnergyTerm(mol_.get(), ETK3DTerm::Angle13, positionsHost.data());
  double gotEnergy  = getETK3DEnergyTerms(systemDevice, atomStartsDevice, positionsDevice, ETK3DTerm::Angle13)[0];
  EXPECT_NEAR(gotEnergy, wantEnergy, E_TOL_FUNCTION);
}

TEST_F(ETK3DGpuTestFixture, Angle13GradientSingleMolecule) {
  loadSingleMol();
  std::vector<double> wantGradients =
    getReferenceETK3DGradientTerm(mol_.get(), ETK3DTerm::Angle13, positionsHost.data());
  std::vector<double> gotGrad =
    getETK3DGradientTerm(systemDevice, atomStartsDevice, positionsDevice, ETK3DTerm::Angle13);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients));
}

TEST_F(ETK3DGpuTestFixture, LongRangeDistanceEnergySingleMolecule) {
  loadSingleMol();

  double wantEnergy = getReferenceETK3DEnergyTerm(mol_.get(), ETK3DTerm::LongRangeDistance, positionsHost.data());
  double gotEnergy =
    getETK3DEnergyTerms(systemDevice, atomStartsDevice, positionsDevice, ETK3DTerm::LongRangeDistance)[0];
  ASSERT_NE(wantEnergy, 0.0);
  EXPECT_NEAR(gotEnergy, wantEnergy, E_TOL_FUNCTION);
}

TEST_F(ETK3DGpuTestFixture, LongRangeDistanceGradientSingleMolecule) {
  loadSingleMol();

  std::vector<double> wantGradients =
    getReferenceETK3DGradientTerm(mol_.get(), ETK3DTerm::LongRangeDistance, positionsHost.data());
  std::vector<double> gotGrad =
    getETK3DGradientTerm(systemDevice, atomStartsDevice, positionsDevice, ETK3DTerm::LongRangeDistance);
  ASSERT_THAT(wantGradients, ::testing::Not(::testing::Each(0.0)));
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients));
}

// Combined energy and gradient tests
TEST_F(ETK3DGpuTestFixture, CombinedEnergiesSingleMolecule) {
  loadSingleMol();

  // Test combined energy calculation with all 3D terms
  CHECK_CUDA_RETURN(nvMolKit::DistGeom::computeEnergyETK(systemDevice, atomStartsDevice, positionsDevice));
  double gotEnergy;
  CHECK_CUDA_RETURN(cudaMemcpy(&gotEnergy, systemDevice.energyOuts.data() + 0, sizeof(double), cudaMemcpyDeviceToHost));

  // Calculate reference combined energy by summing individual terms
  double wantEnergy = 0.0;
  for (const auto& term : allETK3DTerms) {
    const double e = getReferenceETK3DEnergyTerm(mol_.get(), term, positionsHost.data());
    wantEnergy += e;
  }

  EXPECT_NEAR(gotEnergy, wantEnergy, E_TOL_FUNCTION);
}

TEST_F(ETK3DGpuTestFixture, CombinedGradientsSingleMolecue) {
  loadSingleMol();

  // Test combined gradient calculation with all 3D terms
  CHECK_CUDA_RETURN(nvMolKit::DistGeom::computeGradientsETK(systemDevice, atomStartsDevice, positionsDevice));
  std::vector<double> gotGrad(systemDevice.grad.size(), 0.0);
  systemDevice.grad.copyToHost(gotGrad);
  cudaDeviceSynchronize();

  // Calculate reference combined gradients by summing individual terms
  std::vector<double> wantGradients(4 * mol_->getNumAtoms(), 0.0);
  for (const auto& term : allETK3DTerms) {
    auto termGrad = getReferenceETK3DGradientTerm(mol_.get(), term, positionsHost.data());
    for (size_t i = 0; i < wantGradients.size(); ++i) {
      wantGradients[i] += termGrad[i];
    }
  }

  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients));
}

TEST_F(ETK3DGpuTestFixture, PlainCombinedEnergiesSingleMolecule) {
  // Use ETDG parameters (useBasicKnowledge=false) for plain 3D testing
  loadSingleMol(DGeomHelpers::ETDG);

  // Test combined energy calculation with plain 3D terms (ETDG variant - no improper torsions)
  CHECK_CUDA_RETURN(nvMolKit::DistGeom::computeEnergyETK(systemDevice,
                                                         atomStartsDevice,
                                                         positionsDevice,
                                                         nullptr,
                                                         nullptr,
                                                         nvMolKit::DistGeom::ETKTerm::PLAIN));
  double gotEnergy;
  CHECK_CUDA_RETURN(cudaMemcpy(&gotEnergy, systemDevice.energyOuts.data() + 0, sizeof(double), cudaMemcpyDeviceToHost));

  const double wantEnergy = getReferencePlain3DEnergy(mol_.get(), positionsHost.data());

  EXPECT_NEAR(gotEnergy, wantEnergy, E_TOL_FUNCTION);
}

TEST_F(ETK3DGpuTestFixture, PlainCombinedGradientsSingleMolecule) {
  // Use ETDG parameters (useBasicKnowledge=false) for plain 3D testing
  loadSingleMol(DGeomHelpers::ETDG);

  // Test combined gradient calculation with plain 3D terms (ETDG variant - no improper torsions)
  CHECK_CUDA_RETURN(nvMolKit::DistGeom::computeGradientsETK(systemDevice,
                                                            atomStartsDevice,
                                                            positionsDevice,
                                                            nullptr,
                                                            nvMolKit::DistGeom::ETKTerm::PLAIN));
  std::vector<double> gotGrad(systemDevice.grad.size(), 0.0);
  systemDevice.grad.copyToHost(gotGrad);
  cudaDeviceSynchronize();

  const std::vector<double> wantGradients = getReferencePlain3DGradient(mol_.get(), positionsHost.data());

  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients));
}

// -------------------------
// Multi-mol ETK tests
// -------------------------

TEST_F(ETK3DGpuTestFixture, ExperimentalTorsionEnergyMultiMol) {
  constexpr auto term = ETK3DTerm::ExperimentalTorsion;
  loadMMFFMols(10);
  std::vector<double> wantEnergy = getReferenceETK3DEnergyTerms(molsPtrs_, term, positionsHost.data(), atomStartsHost);
  std::vector<double> gotEnergy  = getETK3DEnergyTerms(systemDevice, atomStartsDevice, positionsDevice, term);
  EXPECT_THAT(gotEnergy, ::testing::Pointwise(::testing::DoubleNear(E_TOL_COMBINED), wantEnergy));
}

TEST_F(ETK3DGpuTestFixture, ExperimentalTorsionGradMultiMol) {
  constexpr auto term = ETK3DTerm::ExperimentalTorsion;
  loadMMFFMols(10);
  std::vector<std::vector<double>> wantGrad =
    getReferenceETK3DGradientTerms(molsPtrs_, term, positionsHost.data(), atomStartsHost);
  std::vector<std::vector<double>> gotGrad =
    splitCombinedGrads(getETK3DGradientTerm(systemDevice, atomStartsDevice, positionsDevice, term), atomStartsHost);
  for (size_t i = 0; i < wantGrad.size(); ++i) {
    EXPECT_THAT(gotGrad[i], ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGrad[i])) << "For system " << i;
  }
}

TEST_F(ETK3DGpuTestFixture, ImproperTorsionEnergyMultiMol) {
  constexpr auto term = ETK3DTerm::ImproperTorsion;
  loadMMFFMols(10);
  std::vector<double> wantEnergy = getReferenceETK3DEnergyTerms(molsPtrs_, term, positionsHost.data(), atomStartsHost);
  std::vector<double> gotEnergy  = getETK3DEnergyTerms(systemDevice, atomStartsDevice, positionsDevice, term);
  EXPECT_THAT(gotEnergy, ::testing::Pointwise(::testing::DoubleNear(E_TOL_COMBINED), wantEnergy));
}

TEST_F(ETK3DGpuTestFixture, ImproperTorsionGradMultiMol) {
  constexpr auto term = ETK3DTerm::ImproperTorsion;
  loadMMFFMols(10);
  std::vector<std::vector<double>> wantGrad =
    getReferenceETK3DGradientTerms(molsPtrs_, term, positionsHost.data(), atomStartsHost);
  std::vector<std::vector<double>> gotGrad =
    splitCombinedGrads(getETK3DGradientTerm(systemDevice, atomStartsDevice, positionsDevice, term), atomStartsHost);
  for (size_t i = 0; i < wantGrad.size(); ++i) {
    EXPECT_THAT(gotGrad[i], ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGrad[i])) << "For system " << i;
  }
}

TEST_F(ETK3DGpuTestFixture, Distance12EnergyMultiMol) {
  constexpr auto term = ETK3DTerm::Distance12;
  loadMMFFMols(100);
  std::vector<double> wantEnergy = getReferenceETK3DEnergyTerms(molsPtrs_, term, positionsHost.data(), atomStartsHost);
  std::vector<double> gotEnergy  = getETK3DEnergyTerms(systemDevice, atomStartsDevice, positionsDevice, term);
  EXPECT_THAT(gotEnergy, ::testing::Pointwise(::testing::DoubleNear(E_TOL_COMBINED), wantEnergy));
}

TEST_F(ETK3DGpuTestFixture, Distance12GradMultiMol) {
  constexpr auto term = ETK3DTerm::Distance12;
  loadMMFFMols(10);
  std::vector<std::vector<double>> wantGrad =
    getReferenceETK3DGradientTerms(molsPtrs_, term, positionsHost.data(), atomStartsHost);
  std::vector<std::vector<double>> gotGrad =
    splitCombinedGrads(getETK3DGradientTerm(systemDevice, atomStartsDevice, positionsDevice, term), atomStartsHost);
  for (size_t i = 0; i < wantGrad.size(); ++i) {
    EXPECT_THAT(gotGrad[i], ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGrad[i])) << "For system " << i;
  }
}

TEST_F(ETK3DGpuTestFixture, Distance13EnergyMultiMol) {
  constexpr auto term = ETK3DTerm::Distance13;
  loadMMFFMols(100);
  std::vector<double> wantEnergy = getReferenceETK3DEnergyTerms(molsPtrs_, term, positionsHost.data(), atomStartsHost);
  std::vector<double> gotEnergy  = getETK3DEnergyTerms(systemDevice, atomStartsDevice, positionsDevice, term);
  EXPECT_THAT(gotEnergy, ::testing::Pointwise(::testing::DoubleNear(E_TOL_COMBINED), wantEnergy));
}

TEST_F(ETK3DGpuTestFixture, Distance13GradMultiMol) {
  constexpr auto term = ETK3DTerm::Distance13;
  loadMMFFMols(100);
  std::vector<std::vector<double>> wantGrad =
    getReferenceETK3DGradientTerms(molsPtrs_, term, positionsHost.data(), atomStartsHost);
  std::vector<std::vector<double>> gotGrad =
    splitCombinedGrads(getETK3DGradientTerm(systemDevice, atomStartsDevice, positionsDevice, term), atomStartsHost);
  for (size_t i = 0; i < wantGrad.size(); ++i) {
    EXPECT_THAT(gotGrad[i], ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGrad[i])) << "For system " << i;
  }
}

TEST_F(ETK3DGpuTestFixture, Angle13EnergyMultiMol) {
  constexpr auto term = ETK3DTerm::Angle13;
  loadMMFFMols(10);
  std::vector<double> wantEnergy = getReferenceETK3DEnergyTerms(molsPtrs_, term, positionsHost.data(), atomStartsHost);
  std::vector<double> gotEnergy  = getETK3DEnergyTerms(systemDevice, atomStartsDevice, positionsDevice, term);
  EXPECT_THAT(gotEnergy, ::testing::Pointwise(::testing::DoubleNear(E_TOL_COMBINED), wantEnergy));
}

TEST_F(ETK3DGpuTestFixture, Angle13GradMultiMol) {
  constexpr auto term = ETK3DTerm::Angle13;
  loadMMFFMols(100);
  std::vector<std::vector<double>> wantGrad =
    getReferenceETK3DGradientTerms(molsPtrs_, term, positionsHost.data(), atomStartsHost);
  std::vector<std::vector<double>> gotGrad =
    splitCombinedGrads(getETK3DGradientTerm(systemDevice, atomStartsDevice, positionsDevice, term), atomStartsHost);
  for (size_t i = 0; i < wantGrad.size(); ++i) {
    EXPECT_THAT(gotGrad[i], ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGrad[i])) << "For system " << i;
  }
}

TEST_F(ETK3DGpuTestFixture, LongRangeEnergyMultiMol) {
  constexpr auto term = ETK3DTerm::LongRangeDistance;
  loadMMFFMols(100);
  std::vector<double> wantEnergy = getReferenceETK3DEnergyTerms(molsPtrs_, term, positionsHost.data(), atomStartsHost);
  std::vector<double> gotEnergy  = getETK3DEnergyTerms(systemDevice, atomStartsDevice, positionsDevice, term);
  EXPECT_THAT(gotEnergy, ::testing::Pointwise(::testing::DoubleNear(E_TOL_COMBINED), wantEnergy));
}

TEST_F(ETK3DGpuTestFixture, LongRangeGradMultiMol) {
  constexpr auto term = ETK3DTerm::LongRangeDistance;
  loadMMFFMols(100);
  std::vector<std::vector<double>> wantGrad =
    getReferenceETK3DGradientTerms(molsPtrs_, term, positionsHost.data(), atomStartsHost);
  std::vector<std::vector<double>> gotGrad =
    splitCombinedGrads(getETK3DGradientTerm(systemDevice, atomStartsDevice, positionsDevice, term), atomStartsHost);
  for (size_t i = 0; i < wantGrad.size(); ++i) {
    EXPECT_THAT(gotGrad[i], ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGrad[i])) << "For system " << i;
  }
}

TEST_F(ETK3DGpuTestFixture, CombinedEnergiesMultiMol) {
  loadMMFFMols(10);

  // Test combined energy calculation with all 3D terms (ETKDGv3 - includes improper torsions)
  CHECK_CUDA_RETURN(nvMolKit::DistGeom::computeEnergyETK(systemDevice, atomStartsDevice, positionsDevice));
  std::vector<double> gotEnergy(systemDevice.energyOuts.size(), 0.0);
  systemDevice.energyOuts.copyToHost(gotEnergy);
  cudaDeviceSynchronize();

  // Calculate reference combined energy by summing individual terms for each molecule
  std::vector<double> wantEnergy(molsPtrs_.size(), 0.0);
  for (size_t molIdx = 0; molIdx < molsPtrs_.size(); ++molIdx) {
    for (const auto& term : allETK3DTerms) {
      const double e =
        getReferenceETK3DEnergyTerm(molsPtrs_[molIdx], term, positionsHost.data() + 3 * atomStartsHost[molIdx]);
      wantEnergy[molIdx] += e;
    }
  }

  EXPECT_THAT(gotEnergy, ::testing::Pointwise(::testing::DoubleNear(E_TOL_COMBINED), wantEnergy));
}

TEST_F(ETK3DGpuTestFixture, CombinedGradientsMultiMol) {
  loadMMFFMols(10);

  // Test combined gradient calculation with all 3D terms (ETKDGv3 - includes improper torsions)
  CHECK_CUDA_RETURN(nvMolKit::DistGeom::computeGradientsETK(systemDevice, atomStartsDevice, positionsDevice));
  std::vector<double> gotGrad(systemDevice.grad.size(), 0.0);
  systemDevice.grad.copyToHost(gotGrad);
  cudaDeviceSynchronize();

  // Calculate reference combined gradients by summing individual terms for each molecule
  std::vector<std::vector<double>> wantGradients;
  for (size_t molIdx = 0; molIdx < molsPtrs_.size(); ++molIdx) {
    std::vector<double> molGradients(4 * molsPtrs_[molIdx]->getNumAtoms(), 0.0);
    for (const auto& term : allETK3DTerms) {
      auto termGrad =
        getReferenceETK3DGradientTerm(molsPtrs_[molIdx], term, positionsHost.data() + 3 * atomStartsHost[molIdx]);
      for (size_t i = 0; i < molGradients.size(); ++i) {
        molGradients[i] += termGrad[i];
      }
    }
    wantGradients.push_back(molGradients);
  }

  std::vector<std::vector<double>> gotGradSplit = splitCombinedGrads(gotGrad, atomStartsHost);
  for (size_t i = 0; i < wantGradients.size(); ++i) {
    EXPECT_THAT(gotGradSplit[i], ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients[i]))
      << "For system " << i;
  }
}

// Block-per-mol kernel tests
TEST_F(ETK3DGpuTestFixture, BlockPerMolEnergiesSingleMolecule) {
  loadSingleMol();

  // Test block-per-mol energy calculation with all 3D terms
  CHECK_CUDA_RETURN(nvMolKit::DistGeom::computeEnergyBlockPerMolETK(systemDevice, atomStartsDevice, positionsDevice));
  double gotEnergy;
  CHECK_CUDA_RETURN(cudaMemcpy(&gotEnergy, systemDevice.energyOuts.data() + 0, sizeof(double), cudaMemcpyDeviceToHost));

  // Calculate reference combined energy by summing individual terms
  double wantEnergy = 0.0;
  for (const auto& term : allETK3DTerms) {
    const double e = getReferenceETK3DEnergyTerm(mol_.get(), term, positionsHost.data());
    wantEnergy += e;
  }

  EXPECT_NEAR(gotEnergy, wantEnergy, E_TOL_COMBINED);
}

TEST_F(ETK3DGpuTestFixture, BlockPerMolEnergiesMultiMolecule) {
  loadMMFFMols(10);

  // Test block-per-mol energy calculation with all 3D terms
  CHECK_CUDA_RETURN(nvMolKit::DistGeom::computeEnergyBlockPerMolETK(systemDevice, atomStartsDevice, positionsDevice));
  std::vector<double> gotEnergy(systemDevice.energyOuts.size(), 0.0);
  systemDevice.energyOuts.copyToHost(gotEnergy);
  cudaDeviceSynchronize();

  // Calculate reference combined energies by summing individual terms for each molecule
  std::vector<double> wantEnergy(molsPtrs_.size(), 0.0);
  for (const auto& term : allETK3DTerms) {
    auto termEnergies = getReferenceETK3DEnergyTerms(molsPtrs_, term, positionsHost.data(), atomStartsHost);
    for (size_t i = 0; i < wantEnergy.size(); ++i) {
      wantEnergy[i] += termEnergies[i];
    }
  }

  for (size_t i = 0; i < wantEnergy.size(); ++i) {
    EXPECT_NEAR(gotEnergy[i], wantEnergy[i], E_TOL_COMBINED) << "Mismatch at molecule " << i;
  }
}

TEST_F(ETK3DGpuTestFixture, BlockPerMolGradientsSingleMolecule) {
  loadSingleMol();

  // Test block-per-mol gradient calculation with all 3D terms
  CHECK_CUDA_RETURN(nvMolKit::DistGeom::computeGradBlockPerMolETK(systemDevice, atomStartsDevice, positionsDevice));
  std::vector<double> gotGrad(systemDevice.grad.size(), 0.0);
  systemDevice.grad.copyToHost(gotGrad);
  cudaDeviceSynchronize();

  // Calculate reference combined gradients by summing individual terms
  std::vector<double> wantGradients(4 * mol_->getNumAtoms(), 0.0);
  for (const auto& term : allETK3DTerms) {
    auto termGrad = getReferenceETK3DGradientTerm(mol_.get(), term, positionsHost.data());
    for (size_t i = 0; i < wantGradients.size(); ++i) {
      wantGradients[i] += termGrad[i];
    }
  }

  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients));
}

TEST_F(ETK3DGpuTestFixture, BlockPerMolGradientsMultiMolecule) {
  loadMMFFMols(10);

  // Test block-per-mol gradient calculation with all 3D terms
  CHECK_CUDA_RETURN(nvMolKit::DistGeom::computeGradBlockPerMolETK(systemDevice, atomStartsDevice, positionsDevice));
  std::vector<double> gotGrad(systemDevice.grad.size(), 0.0);
  systemDevice.grad.copyToHost(gotGrad);
  cudaDeviceSynchronize();

  // Calculate reference combined gradients by summing individual terms for each molecule
  std::vector<std::vector<double>> wantGradients;
  for (size_t molIdx = 0; molIdx < molsPtrs_.size(); ++molIdx) {
    std::vector<double> molGradients(4 * molsPtrs_[molIdx]->getNumAtoms(), 0.0);
    for (const auto& term : allETK3DTerms) {
      auto termGrad =
        getReferenceETK3DGradientTerm(molsPtrs_[molIdx], term, positionsHost.data() + 3 * atomStartsHost[molIdx]);
      for (size_t i = 0; i < molGradients.size(); ++i) {
        molGradients[i] += termGrad[i];
      }
    }
    wantGradients.push_back(molGradients);
  }

  std::vector<std::vector<double>> gotGradSplit = splitCombinedGrads(gotGrad, atomStartsHost);
  for (size_t i = 0; i < wantGradients.size(); ++i) {
    EXPECT_THAT(gotGradSplit[i], ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients[i]))
      << "For system " << i;
  }
}

TEST_F(ETK3DGpuTestFixture, PlainCombinedEnergiesMultiMol) {
  loadMMFFMols(10, DGeomHelpers::ETDG);  // Use ETDG parameters (useBasicKnowledge=false)

  // Test combined energy calculation with plain 3D terms (ETDG variant - no improper torsions)
  CHECK_CUDA_RETURN(nvMolKit::DistGeom::computeEnergyETK(systemDevice,
                                                         atomStartsDevice,
                                                         positionsDevice,
                                                         nullptr,
                                                         nullptr,
                                                         nvMolKit::DistGeom::ETKTerm::PLAIN));
  std::vector<double> gotEnergy(systemDevice.energyOuts.size(), 0.0);
  systemDevice.energyOuts.copyToHost(gotEnergy);
  cudaDeviceSynchronize();

  // Calculate reference combined energy using the same shared atom-pair exclusions as the device setup.
  std::vector<double> wantEnergy(molsPtrs_.size(), 0.0);
  for (size_t molIdx = 0; molIdx < molsPtrs_.size(); ++molIdx) {
    wantEnergy[molIdx] =
      getReferencePlain3DEnergy(molsPtrs_[molIdx], positionsHost.data() + 3 * atomStartsHost[molIdx]);
  }

  EXPECT_THAT(gotEnergy, ::testing::Pointwise(::testing::DoubleNear(E_TOL_COMBINED), wantEnergy));
}

TEST_F(ETK3DGpuTestFixture, PlainCombinedGradientsMultiMol) {
  loadMMFFMols(10, DGeomHelpers::ETDG);  // Use ETDG parameters (useBasicKnowledge=false)

  // Test combined gradient calculation with plain 3D terms (ETDG variant - no improper torsions)
  systemDevice.grad.zero();
  CHECK_CUDA_RETURN(nvMolKit::DistGeom::computeGradientsETK(systemDevice,
                                                            atomStartsDevice,
                                                            positionsDevice,
                                                            nullptr,
                                                            nvMolKit::DistGeom::ETKTerm::PLAIN));
  std::vector<double> gotGrad(systemDevice.grad.size(), 0.0);
  systemDevice.grad.copyToHost(gotGrad);
  cudaDeviceSynchronize();

  // Validate the combined PLAIN path against the sum of isolated device term kernels.
  // The device torsion gradient intentionally carries fixes that can differ from RDKit's
  // scalar CrystalFF gradient for affected torsions.
  std::vector<double> wantCombinedGrad(systemDevice.grad.size(), 0.0);
  for (const auto& term : plainETK3DTerms) {
    auto termGrad = getETK3DGradientTerm(systemDevice, atomStartsDevice, positionsDevice, term);
    for (size_t i = 0; i < wantCombinedGrad.size(); ++i) {
      wantCombinedGrad[i] += termGrad[i];
    }
  }
  std::vector<std::vector<double>> wantGradients = splitCombinedGrads(wantCombinedGrad, atomStartsHost);

  std::vector<std::vector<double>> gotGradSplit = splitCombinedGrads(gotGrad, atomStartsHost);
  for (size_t i = 0; i < wantGradients.size(); ++i) {
    EXPECT_THAT(gotGradSplit[i], ::testing::Pointwise(::testing::FloatNear(G_TOL), wantGradients[i]))
      << "For system " << i;
  }
}
