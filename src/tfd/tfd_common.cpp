// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tfd_common.h"

#include <Geometry/point.h>
#include <GraphMol/Fingerprints/FingerprintGenerator.h>
#include <GraphMol/Fingerprints/MorganGenerator.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/RingInfo.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "nvtx.h"
#include "versions.h"

namespace nvMolKit {

namespace {

//! Get heavy atom neighbors of an atom, optionally excluding one atom
std::vector<const RDKit::Atom*> getHeavyAtomNeighbors(const RDKit::Atom* atom, int excludeIdx = -1) {
  std::vector<const RDKit::Atom*> neighbors;
  for (const auto* neighbor : atom->getOwningMol().atomNeighbors(atom)) {
    if (neighbor->getAtomicNum() != 1 && neighbor->getIdx() != static_cast<unsigned int>(excludeIdx)) {
      neighbors.push_back(neighbor);
    }
  }
  return neighbors;
}

//! Check if all atoms have the same invariant
bool doMatch(const std::vector<std::uint32_t>& inv, const std::vector<const RDKit::Atom*>& atoms) {
  if (atoms.size() < 2) {
    return true;
  }
  auto firstInv = inv[atoms[0]->getIdx()];
  for (size_t i = 1; i < atoms.size(); ++i) {
    if (inv[atoms[i]->getIdx()] != firstInv) {
      return false;
    }
  }
  return true;
}

//! Find the atom that is different when two atoms match (for 3 atoms)
const RDKit::Atom* doMatchExcept1(const std::vector<std::uint32_t>& inv, const std::vector<const RDKit::Atom*>& atoms) {
  if (atoms.size() != 3) {
    return nullptr;
  }
  int a1 = atoms[0]->getIdx();
  int a2 = atoms[1]->getIdx();
  int a3 = atoms[2]->getIdx();

  if (inv[a1] == inv[a2] && inv[a1] != inv[a3] && inv[a2] != inv[a3]) {
    return atoms[2];
  } else if (inv[a1] != inv[a2] && inv[a1] == inv[a3] && inv[a2] != inv[a3]) {
    return atoms[1];
  } else if (inv[a1] != inv[a2] && inv[a1] != inv[a3] && inv[a2] == inv[a3]) {
    return atoms[0];
  }
  return nullptr;
}

//! Get atom invariants using Morgan fingerprints at given radius
std::vector<std::uint32_t> getAtomInvariantsWithRadius(const RDKit::ROMol& mol, int radius) {
  std::vector<std::uint32_t> inv(mol.getNumAtoms(), 0);

  auto fpGen = RDKit::MorganFingerprint::getMorganGenerator<std::uint32_t>(radius,
                                                                           false /* countSimulation */,
                                                                           false /* includeChirality */,
                                                                           true /* useBondTypes */,
                                                                           false /* onlyNonzeroInvariants */,
                                                                           true /* includeRedundantEnvironments */);

  RDKit::AdditionalOutput ao;
  ao.allocateBitInfoMap();

  // Call to populate ao.bitInfoMap; result is unused
  (void)fpGen->getSparseCountFingerprint(mol, nullptr, nullptr, -1, &ao);
  const auto& bitInfo = ao.bitInfoMap;

  if (!bitInfo) {
    throw std::runtime_error("Morgan fingerprint bitInfoMap was not populated");
  }
  for (const auto& [bitId, atomRadiusPairs] : *bitInfo) {
    for (const auto& [atomIdx, r] : atomRadiusPairs) {
      if (r == static_cast<unsigned int>(radius)) {
        inv[atomIdx] = bitId;
      }
    }
  }

  return inv;
}

//! Get reference atoms for torsion based on neighbor symmetry
std::vector<const RDKit::Atom*> getIndexForTorsion(const std::vector<const RDKit::Atom*>& neighbors,
                                                   const std::vector<std::uint32_t>&      inv) {
  if (neighbors.size() == 1) {
    return neighbors;
  } else if (doMatch(inv, neighbors)) {
    // All symmetric neighbors - return all
    return neighbors;
  } else if (neighbors.size() == 3) {
    const RDKit::Atom* different = doMatchExcept1(inv, neighbors);
    if (different) {
      return {different};
    }
  }
  // Fallback: take the atom with the smallest invariant
  auto it = std::min_element(neighbors.begin(), neighbors.end(), [&inv](const RDKit::Atom* a, const RDKit::Atom* b) {
    return inv[a->getIdx()] < inv[b->getIdx()];
  });
  return {*it};
}

//! Information about a rotatable bond for torsion calculation
struct BondInfo {
  int                             a1;   // First central atom
  int                             a2;   // Second central atom
  std::vector<const RDKit::Atom*> nb1;  // Heavy atom neighbors of a1 (excluding a2)
  std::vector<const RDKit::Atom*> nb2;  // Heavy atom neighbors of a2 (excluding a1)
};

//! Get bonds for which torsions should be calculated
std::vector<BondInfo> getBondsForTorsions(const RDKit::ROMol& mol, bool ignoreColinearBonds) {
  // Flag atoms that cannot be middle atoms of torsion (triple bonds, allenes)
  std::vector<int> atomFlags(mol.getNumAtoms(), 0);

  // Flag atoms adjacent to triple bonds
  for (const auto* bond : mol.bonds()) {
    if (bond->getBondTypeAsDouble() == 3.0) {
      atomFlags[bond->getBeginAtomIdx()] = 1;
      atomFlags[bond->getEndAtomIdx()]   = 1;
    }
  }

  // Flag allene centers: carbon atoms with exactly two double bonds
  for (const auto* atom : mol.atoms()) {
    if (atom->getAtomicNum() != 6)
      continue;
    int doubleBondCount = 0;
    for (const auto* bond : mol.atomBonds(atom)) {
      if (bond->getBondTypeAsDouble() == 2.0) {
        doubleBondCount++;
      }
    }
    if (doubleBondCount == 2) {
      atomFlags[atom->getIdx()] = 1;
    }
  }

  std::vector<BondInfo> bonds;
  std::vector<int>      doneBonds(mol.getNumBonds(), 0);

  const auto* ringInfo = mol.getRingInfo();
  for (const auto* bond : mol.bonds()) {
    if (ringInfo->numBondRings(bond->getIdx()) > 0) {
      continue;
    }

    int a1 = bond->getBeginAtomIdx();
    int a2 = bond->getEndAtomIdx();

    auto nb1 = getHeavyAtomNeighbors(bond->getBeginAtom(), a2);
    auto nb2 = getHeavyAtomNeighbors(bond->getEndAtom(), a1);

    if (!doneBonds[bond->getIdx()] && !nb1.empty() && !nb2.empty()) {
      doneBonds[bond->getIdx()] = 1;

      // Check if atoms cannot be middle atoms
      if (atomFlags[a1] || atomFlags[a2]) {
        if (!ignoreColinearBonds) {
          // Search for alternative atoms (following the Python logic)
          while (nb1.size() == 1 && atomFlags[a1]) {
            int a1old = a1;
            a1        = nb1[0]->getIdx();
            auto* b   = mol.getBondBetweenAtoms(a1old, a1);
            if (b) {
              if (b->getEndAtomIdx() == static_cast<unsigned int>(a1old)) {
                nb1 = getHeavyAtomNeighbors(b->getBeginAtom(), a1old);
              } else {
                nb1 = getHeavyAtomNeighbors(b->getEndAtom(), a1old);
              }
              doneBonds[b->getIdx()] = 1;
            } else {
              break;
            }
          }
          while (nb2.size() == 1 && atomFlags[a2]) {
            int a2old = a2;
            a2        = nb2[0]->getIdx();
            auto* b   = mol.getBondBetweenAtoms(a2old, a2);
            if (b) {
              if (b->getBeginAtomIdx() == static_cast<unsigned int>(a2old)) {
                nb2 = getHeavyAtomNeighbors(b->getEndAtom(), a2old);
              } else {
                nb2 = getHeavyAtomNeighbors(b->getBeginAtom(), a2old);
              }
              doneBonds[b->getIdx()] = 1;
            } else {
              break;
            }
          }
          if (!nb1.empty() && !nb2.empty()) {
            bonds.push_back({a1, a2, nb1, nb2});
          }
        }
        // If ignoreColinearBonds is true, we skip this bond
      } else {
        bonds.push_back({a1, a2, nb1, nb2});
      }
    }
  }

  return bonds;
}

//! Find the most central bond in a molecule.
//! Returns {-1, -1} if no central bond can be found (e.g., methane, linear molecules).
//!
//! NOTE: For molecules with near-perfect C2 topological symmetry (e.g. macrocyclic
//! peptides), two atoms may have STDs that differ only at machine-epsilon (~4e-16).
//! The sort tiebreak can therefore pick either atom as "most central", and the choice
//! may differ from RDKit Python (which uses numpy.std with a different accumulation
//! order).  This causes the weighted TFD to diverge by up to ~0.1 for affected
//! molecules (~0.8% of ChEMBL), while unweighted TFD matches exactly.  Both weight
//! schemes are equally valid — the TFD paper does not prescribe a tiebreak rule.
std::pair<int, int> findCentralBond(const RDKit::ROMol& mol, const double* distMat) {
  int numAtoms = mol.getNumAtoms();

  // Calculate STD of distances for each non-terminal atom
  std::vector<std::pair<double, int>> stds;
  for (int i = 0; i < numAtoms; ++i) {
    auto neighbors = getHeavyAtomNeighbors(mol.getAtomWithIdx(i));
    if (neighbors.size() < 2) {
      continue;  // Skip terminal atoms
    }

    // Calculate STD of distances
    double sum   = 0.0;
    double sumSq = 0.0;
    int    count = 0;
    for (int j = 0; j < numAtoms; ++j) {
      if (j != i) {
        double d = distMat[i * numAtoms + j];
        sum += d;
        sumSq += d * d;
        count++;
      }
    }
    double mean     = sum / count;
    double variance = (sumSq / count) - (mean * mean);
    double stdDev   = std::sqrt(std::max(0.0, variance));
    stds.emplace_back(stdDev, i);
  }

  if (stds.empty()) {
    return {-1, -1};  // No non-terminal atoms found
  }

  std::sort(stds.begin(), stds.end());
  int aid1 = stds[0].second;

  // Find second most central atom that is bonded to aid1
  for (size_t i = 1; i < stds.size(); ++i) {
    if (mol.getBondBetweenAtoms(aid1, stds[i].second) != nullptr) {
      return {aid1, stds[i].second};
    }
  }

  return {-1, -1};  // Could not find central bond
}

//! Calculate beta parameter for weight calculation
double calculateBeta(const RDKit::ROMol& mol, const double* distMat, int aid1) {
  int numAtoms = mol.getNumAtoms();

  // Match RDKit's _calculateBeta (TorsionFingerprints.py) version-for-version.
  // Pre-2026.03.1 RDKit had a typo that checked nb2 twice, inflating dmax by
  // including bonds where only the end atom was non-terminal. Commit b56f3dc68
  // (RDKit 2026.03.1) fixed it to check both endpoints. We match the RDKit version installed against.
  constexpr bool kRdkitHasBetaTypoFix =
    RDKIT_VERSION_MAJOR > 2026 || (RDKIT_VERSION_MAJOR == 2026 && RDKIT_VERSION_MINOR >= 3);
  double dmax = 0.0;
  for (const auto* bond : mol.bonds()) {
    auto       nb1                = getHeavyAtomNeighbors(bond->getBeginAtom());
    auto       nb2                = getHeavyAtomNeighbors(bond->getEndAtom());
    const bool beginIsNonTerminal = kRdkitHasBetaTypoFix ? (nb1.size() > 1) : (nb2.size() > 1);
    if (beginIsNonTerminal && nb2.size() > 1) {
      int    bid1 = bond->getBeginAtomIdx();
      int    bid2 = bond->getEndAtomIdx();
      double d    = std::max(distMat[aid1 * numAtoms + bid1], distMat[aid1 * numAtoms + bid2]);
      dmax        = std::max(dmax, d);
    }
  }

  double dmax2 = dmax / 2.0;
  if (dmax2 < 1e-6) {
    dmax2 = 1.0;  // Avoid division by zero
  }
  double beta = -std::log(0.1) / (dmax2 * dmax2);
  return beta;
}

}  // namespace

// Internal: build torsion list from precomputed bonds (used when caller already has bonds)
static TorsionList extractTorsionListImpl(const RDKit::ROMol&          mol,
                                          TFDMaxDevMode                maxDevMode,
                                          int                          symmRadius,
                                          const std::vector<BondInfo>& bonds) {
  TorsionList result;

  // Get atom invariants
  std::vector<std::uint32_t> inv;
  if (symmRadius > 0) {
    inv = getAtomInvariantsWithRadius(mol, symmRadius);
  } else {
    // Use connectivity invariants as fallback
    inv.resize(mol.getNumAtoms());
    for (unsigned int i = 0; i < mol.getNumAtoms(); ++i) {
      inv[i] = mol.getAtomWithIdx(i)->getDegree();
    }
  }

  // Process each bond to create torsions
  for (const auto& bond : bonds) {
    auto d1 = getIndexForTorsion(bond.nb1, inv);
    auto d2 = getIndexForTorsion(bond.nb2, inv);

    TorsionDef torsion;

    if (maxDevMode == TFDMaxDevMode::Equal) {
      // Equal mode: all combinations (d1 x d2), maxDev 180 (default path)
      for (const auto* n1 : d1) {
        for (const auto* n2 : d2) {
          torsion.atomQuartets.push_back(
            {static_cast<int>(n1->getIdx()), bond.a1, bond.a2, static_cast<int>(n2->getIdx())});
        }
      }
      torsion.maxDev = 180.0f;
    } else {
      // Spec mode: build quartets and set torsion-specific maxDev
      if (d1.size() == 1 && d2.size() == 1) {
        // Case 1, 2, 4, 5, 7, 10, 16, 12, 17, 19 - single torsion
        torsion.atomQuartets.push_back(
          {static_cast<int>(d1[0]->getIdx()), bond.a1, bond.a2, static_cast<int>(d2[0]->getIdx())});
        torsion.maxDev = 180.0f;
      } else if (d1.size() == 1) {
        // Case 3, 6, 8, 13, 20 - multiple torsions from d2
        for (const auto* nb : d2) {
          torsion.atomQuartets.push_back(
            {static_cast<int>(d1[0]->getIdx()), bond.a1, bond.a2, static_cast<int>(nb->getIdx())});
        }
        torsion.maxDev = (bond.nb2.size() == 2) ? 90.0f : 60.0f;
      } else if (d2.size() == 1) {
        // Case 3, 6, 8, 13, 20 - multiple torsions from d1
        for (const auto* nb : d1) {
          torsion.atomQuartets.push_back(
            {static_cast<int>(nb->getIdx()), bond.a1, bond.a2, static_cast<int>(d2[0]->getIdx())});
        }
        torsion.maxDev = (bond.nb1.size() == 2) ? 90.0f : 60.0f;
      } else {
        // Both symmetric - all combinations
        for (const auto* n1 : d1) {
          for (const auto* n2 : d2) {
            torsion.atomQuartets.push_back(
              {static_cast<int>(n1->getIdx()), bond.a1, bond.a2, static_cast<int>(n2->getIdx())});
          }
        }
        if (bond.nb1.size() == 2 && bond.nb2.size() == 2) {
          torsion.maxDev = 90.0f;
        } else if (bond.nb1.size() == 3 && bond.nb2.size() == 3) {
          torsion.maxDev = 60.0f;
        } else {
          torsion.maxDev = 30.0f;
        }
      }
    }

    result.nonRingTorsions.push_back(std::move(torsion));
  }

  // Process rings
  auto rings = mol.getRingInfo()->atomRings();
  for (const auto& ring : rings) {
    TorsionDef torsion;
    int        num = static_cast<int>(ring.size());

    // Calculate max deviation for ring
    float maxdev;
    if (num >= 14) {
      maxdev = 180.0f;
    } else {
      maxdev = 180.0f * std::exp(-0.025f * (num - 14) * (num - 14));
    }

    // Create torsions for ring (consecutive 4 atoms)
    for (int i = 0; i < num; ++i) {
      torsion.atomQuartets.push_back({ring[i], ring[(i + 1) % num], ring[(i + 2) % num], ring[(i + 3) % num]});
    }
    torsion.maxDev = maxdev;

    result.ringTorsions.push_back(std::move(torsion));
  }

  return result;
}

TorsionList extractTorsionList(const RDKit::ROMol& mol,
                               TFDMaxDevMode       maxDevMode,
                               int                 symmRadius,
                               bool                ignoreColinearBonds) {
  auto bonds = getBondsForTorsions(mol, ignoreColinearBonds);
  return extractTorsionListImpl(mol, maxDevMode, symmRadius, bonds);
}

// Internal: compute weights using precomputed bonds (used when caller already has bonds)
static std::vector<float> computeTorsionWeightsImpl(const RDKit::ROMol&          mol,
                                                    const TorsionList&           torsionList,
                                                    const std::vector<BondInfo>& bonds) {
  std::vector<float> weights;

  // If no torsions, return empty weights
  size_t totalTorsions = torsionList.totalCount();
  if (totalTorsions == 0) {
    return weights;
  }

  // Get distance matrix (returns raw pointer, stored in mol's dictionary)
  const double* distMat  = RDKit::MolOps::getDistanceMat(mol);
  int           numAtoms = mol.getNumAtoms();

  // Find central bond
  auto [aid1, aid2] = findCentralBond(mol, distMat);

  // If no central bond found, return uniform weights
  if (aid1 < 0 || aid2 < 0) {
    weights.resize(totalTorsions, 1.0f);
    return weights;
  }

  // Calculate beta
  double beta = calculateBeta(mol, distMat, aid1);

  // Calculate weights for non-ring torsions (bonds provided by caller)
  for (size_t i = 0; i < bonds.size(); ++i) {
    const auto& bond = bonds[i];
    double      d;

    if ((bond.a1 == aid1 && bond.a2 == aid2) || (bond.a1 == aid2 && bond.a2 == aid1)) {
      d = 0.0;  // Central bond itself
    } else {
      // Shortest distance to central bond atoms + 1
      d = std::min({distMat[aid1 * numAtoms + bond.a1],
                    distMat[aid1 * numAtoms + bond.a2],
                    distMat[aid2 * numAtoms + bond.a1],
                    distMat[aid2 * numAtoms + bond.a2]}) +
          1.0;
    }

    float w = static_cast<float>(std::exp(-beta * d * d));
    weights.push_back(w);
  }

  // Calculate weights for ring torsions
  auto ringInfo  = mol.getRingInfo();
  auto bondRings = ringInfo->bondRings();

  for (const auto& bondRing : bondRings) {
    int    num  = static_cast<int>(bondRing.size());
    double sumD = 0.0;

    for (int bidx : bondRing) {
      const auto* bond = mol.getBondWithIdx(bidx);
      int         bid1 = bond->getBeginAtomIdx();
      int         bid2 = bond->getEndAtomIdx();
      double      d    = std::min({distMat[aid1 * numAtoms + bid1],
                                   distMat[aid1 * numAtoms + bid2],
                                   distMat[aid2 * numAtoms + bid1],
                                   distMat[aid2 * numAtoms + bid2]}) +
                 1.0;
      sumD += d;
    }

    double avgD = sumD / num;
    float  w    = static_cast<float>(std::exp(-beta * avgD * avgD) * (num / 2.0));
    weights.push_back(w);
  }

  return weights;
}

std::vector<float> computeTorsionWeights(const RDKit::ROMol& mol,
                                         const TorsionList&  torsionList,
                                         bool                ignoreColinearBonds) {
  auto bonds = getBondsForTorsions(mol, ignoreColinearBonds);
  return computeTorsionWeightsImpl(mol, torsionList, bonds);
}

// Sequential single-molecule builder (used as the building block for parallel batch builds)
static TFDSystemHost buildTFDSystemImpl(const RDKit::ROMol& mol, const TFDComputeOptions& options) {
  TFDSystemHost system;

  int numConformers = mol.getNumConformers();
  int numAtoms      = mol.getNumAtoms();

  if (numConformers == 0) {
    throw std::runtime_error("Molecule has no conformers");
  }

  // Get bonds once and reuse for torsion list and (optionally) weights
  auto        bonds       = getBondsForTorsions(mol, options.ignoreColinearBonds);
  TorsionList torsionList = extractTorsionListImpl(mol, options.maxDevMode, options.symmRadius, bonds);

  // Extract weights if needed (reuse same bonds)
  std::vector<float> weights;
  if (options.useWeights) {
    weights = computeTorsionWeightsImpl(mol, torsionList, bonds);
  }

  int confStart          = static_cast<int>(system.confPositionStarts.size());
  int torsStart          = system.totalTorsions();
  int quartetStartForMol = system.totalQuartets();

  // Extract coordinates (tightly packed, no padding)
  for (auto confIt = mol.beginConformers(); confIt != mol.endConformers(); ++confIt) {
    system.confPositionStarts.push_back(static_cast<int>(system.positions.size()));
    const auto& conf = **confIt;
    for (int atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
      const auto& pos = conf.getAtomPos(atomIdx);
      system.positions.push_back(static_cast<float>(pos.x));
      system.positions.push_back(static_cast<float>(pos.y));
      system.positions.push_back(static_cast<float>(pos.z));
    }
  }

  // Add torsion definitions (store ALL quartets, classify type)
  int torsionIdx = 0;

  for (const auto& torsion : torsionList.nonRingTorsions) {
    if (torsion.atomQuartets.empty()) {
      torsionIdx++;
      continue;
    }
    for (const auto& q : torsion.atomQuartets)
      system.torsionAtoms.push_back(q);
    system.quartetStarts.push_back(static_cast<int>(system.torsionAtoms.size()));
    system.torsionTypes.push_back(torsion.atomQuartets.size() > 1 ? TorsionType::Symmetric : TorsionType::Single);
    system.torsionMaxDevs.push_back(torsion.maxDev);
    system.torsionWeights.push_back(
      (options.useWeights && torsionIdx < static_cast<int>(weights.size())) ? weights[torsionIdx] : 1.0f);
    torsionIdx++;
  }

  for (const auto& torsion : torsionList.ringTorsions) {
    if (torsion.atomQuartets.empty()) {
      torsionIdx++;
      continue;
    }
    for (const auto& q : torsion.atomQuartets)
      system.torsionAtoms.push_back(q);
    system.quartetStarts.push_back(static_cast<int>(system.torsionAtoms.size()));
    system.torsionTypes.push_back(torsion.atomQuartets.size() > 1 ? TorsionType::Ring : TorsionType::Single);
    system.torsionMaxDevs.push_back(torsion.maxDev);
    system.torsionWeights.push_back(
      (options.useWeights && torsionIdx < static_cast<int>(weights.size())) ? weights[torsionIdx] : 1.0f);
    torsionIdx++;
  }

  // Build per-molecule descriptor
  int numTorsions         = system.totalTorsions() - torsStart;
  int totalQuartetsForMol = system.totalQuartets() - quartetStartForMol;
  int dihedStart          = system.totalDihedrals();
  int numDihedrals        = numConformers * totalQuartetsForMol;
  int numTFDOutputs       = numConformers * (numConformers - 1) / 2;
  int tfdOutStart         = system.totalTFDOutputs();

  system.totalDihedrals_ = dihedStart + numDihedrals;

  MolDescriptor desc;
  desc.confStart     = confStart;
  desc.numConformers = numConformers;
  desc.quartetStart  = quartetStartForMol;
  desc.numQuartets   = totalQuartetsForMol;
  desc.dihedStart    = dihedStart;
  desc.torsStart     = torsStart;
  desc.numTorsions   = numTorsions;
  desc.tfdOutStart   = tfdOutStart;
  system.molDescriptors.push_back(std::move(desc));

  system.dihedralWorkStarts.push_back(system.dihedralWorkStarts.back() + numDihedrals);
  system.tfdWorkStarts.push_back(system.tfdWorkStarts.back() + numTFDOutputs);

  return system;
}

//! Lightweight per-molecule extraction result (used in two-pass batch build).
//! Contains RDKit-derived torsion data, coordinates, and per-molecule sizes needed
//! for computing global offsets. Coordinates are extracted during Pass 1 (parallel)
//! so that Pass 2 only needs memcpy, not RDKit conformer access.
struct MolExtraction {
  TorsionList                     torsionList;
  std::vector<float>              weights;
  std::vector<std::array<int, 4>> atoms;      //!< Flattened torsion atom quartets
  std::vector<float>              wts;        //!< Weight per torsion
  std::vector<float>              maxDevs;    //!< MaxDev per torsion
  std::vector<TorsionType>        types;      //!< Type per torsion
  std::vector<int>                qStarts;    //!< CSR for quartets (including leading 0)
  std::vector<float>              positions;  //!< Flat xyz coords for all conformers
  int                             numConformers;
  int                             numAtoms;
  int                             numTorsions;   //!< = wts.size()
  int                             numQuartets;   //!< = atoms.size()
  int                             numPositions;  //!< = numConformers * numAtoms * 3
};

//! Extract torsion data and coordinates from a single molecule (Pass 1 of two-pass build).
//! Coordinates are extracted here (in parallel) so Pass 2 only needs memcpy.
static MolExtraction extractMolData(const RDKit::ROMol& mol, const TFDComputeOptions& options) {
  MolExtraction ext;
  ext.numConformers = mol.getNumConformers();
  ext.numAtoms      = mol.getNumAtoms();
  ext.numPositions  = ext.numConformers * ext.numAtoms * 3;

  if (ext.numConformers == 0) {
    throw std::runtime_error("Molecule has no conformers");
  }

  auto bonds      = getBondsForTorsions(mol, options.ignoreColinearBonds);
  ext.torsionList = extractTorsionListImpl(mol, options.maxDevMode, options.symmRadius, bonds);
  if (options.useWeights) {
    ext.weights = computeTorsionWeightsImpl(mol, ext.torsionList, bonds);
  }

  // Extract coordinates (tightly packed, no padding)
  ext.positions.resize(ext.numPositions);
  int posIdx = 0;
  for (auto confIt = mol.beginConformers(); confIt != mol.endConformers(); ++confIt) {
    const auto& conf = **confIt;
    for (int a = 0; a < ext.numAtoms; ++a) {
      const auto& pos           = conf.getAtomPos(a);
      ext.positions[posIdx]     = static_cast<float>(pos.x);
      ext.positions[posIdx + 1] = static_cast<float>(pos.y);
      ext.positions[posIdx + 2] = static_cast<float>(pos.z);
      posIdx += 3;
    }
  }

  // Flatten torsion definitions into GPU-ready arrays
  ext.qStarts.push_back(0);
  int torsionIdx = 0;

  auto addTorsion = [&](const TorsionDef& torsion, bool isRing) {
    if (torsion.atomQuartets.empty()) {
      torsionIdx++;
      return;
    }
    for (const auto& q : torsion.atomQuartets)
      ext.atoms.push_back(q);
    ext.qStarts.push_back(static_cast<int>(ext.atoms.size()));
    if (isRing) {
      ext.types.push_back(torsion.atomQuartets.size() > 1 ? TorsionType::Ring : TorsionType::Single);
    } else {
      ext.types.push_back(torsion.atomQuartets.size() > 1 ? TorsionType::Symmetric : TorsionType::Single);
    }
    ext.maxDevs.push_back(torsion.maxDev);
    ext.wts.push_back(
      (options.useWeights && torsionIdx < static_cast<int>(ext.weights.size())) ? ext.weights[torsionIdx] : 1.0f);
    torsionIdx++;
  };

  for (const auto& t : ext.torsionList.nonRingTorsions)
    addTorsion(t, false);
  for (const auto& t : ext.torsionList.ringTorsions)
    addTorsion(t, true);

  ext.numTorsions = static_cast<int>(ext.wts.size());
  ext.numQuartets = static_cast<int>(ext.atoms.size());
  return ext;
}

TFDSystemHost buildTFDSystem(const std::vector<const RDKit::ROMol*>& mols, const TFDComputeOptions& options) {
  ScopedNvtxRange range("buildTFDSystem (" + std::to_string(mols.size()) + " mols)", NvtxColor::kCyan);

  if (mols.empty()) {
    return {};
  }
  if (mols.size() == 1) {
    return buildTFDSystemImpl(*mols[0], options);
  }

  int N = static_cast<int>(mols.size());

  // ---- Pass 1: extract torsion data per-molecule in parallel ----
  // This is the expensive part (RDKit fingerprinting, SMARTS matching, etc.)
  std::vector<MolExtraction> extractions(N);
  {
    ScopedNvtxRange buildRange("Parallel RDKit extraction", NvtxColor::kCyan);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
      extractions[i] = extractMolData(*mols[i], options);
    }
  }

  // ---- Compute global offsets (sequential, ~O(N) with tiny per-element cost) ----
  ScopedNvtxRange mergeRange("Compute offsets & fill system", NvtxColor::kCyan);

  std::vector<int> posOffset(N);
  std::vector<int> confOffset(N);
  std::vector<int> quartetOffset(N);
  std::vector<int> torsOffset(N);
  std::vector<int> dihedOffset(N);
  std::vector<int> tfdOutOffset(N);

  int totalPos      = 0;
  int totalConfs    = 0;
  int totalQuartets = 0;
  int totalTors     = 0;
  int totalDiheds   = 0;
  int totalTfdOuts  = 0;

  for (int i = 0; i < N; ++i) {
    const auto& ext  = extractions[i];
    posOffset[i]     = totalPos;
    confOffset[i]    = totalConfs;
    quartetOffset[i] = totalQuartets;
    torsOffset[i]    = totalTors;
    dihedOffset[i]   = totalDiheds;
    tfdOutOffset[i]  = totalTfdOuts;

    totalPos += ext.numPositions;
    totalConfs += ext.numConformers;
    totalQuartets += ext.numQuartets;
    totalTors += ext.numTorsions;
    totalDiheds += ext.numConformers * ext.numQuartets;
    totalTfdOuts += ext.numConformers * (ext.numConformers - 1) / 2;
  }

  // ---- Allocate the final system (single allocation per array) ----
  // For large arrays (positions, torsionAtoms), we use reserve() + resize()
  // to avoid double-writing: reserve() allocates without init, then the parallel
  // fill writes every element exactly once before the vector is used.
  TFDSystemHost system;
  // Positions is the largest array (~29MB for 1900 mols x 100 confs).
  // Use reserve+resize pattern: reserve avoids reallocation during parallel fill.
  system.positions.resize(totalPos);
  system.confPositionStarts.resize(totalConfs);
  system.torsionAtoms.resize(totalQuartets);
  system.torsionWeights.resize(totalTors);
  system.torsionMaxDevs.resize(totalTors);
  system.torsionTypes.resize(totalTors);
  system.quartetStarts.resize(1 + totalTors);
  system.quartetStarts[0] = 0;
  system.molDescriptors.resize(N);
  system.dihedralWorkStarts.resize(1 + N);
  system.dihedralWorkStarts[0] = 0;
  system.tfdWorkStarts.resize(1 + N);
  system.tfdWorkStarts[0] = 0;
  system.totalDihedrals_  = totalDiheds;

  // ---- Pass 2: fill the system in parallel ----
  // Each thread writes to its own contiguous block of molecules, avoiding false sharing
  // on adjacent cache lines. static schedule with a chunk size ensures spatial locality.
  const int chunkSize = std::max(1, N / (omp_get_max_threads() * 4));
#pragma omp parallel for schedule(static, chunkSize)
  for (int i = 0; i < N; ++i) {
    const auto& ext = extractions[i];

    // Positions: bulk memcpy from pre-extracted coordinates
    if (ext.numPositions > 0) {
      std::memcpy(system.positions.data() + posOffset[i], ext.positions.data(), ext.numPositions * sizeof(float));
    }
    // confPositionStarts: compute directly
    for (int c = 0; c < ext.numConformers; ++c) {
      system.confPositionStarts[confOffset[i] + c] = posOffset[i] + c * ext.numAtoms * 3;
    }

    // Torsion atoms: bulk copy (no offset adjustment needed)
    if (ext.numQuartets > 0) {
      std::memcpy(&system.torsionAtoms[quartetOffset[i]],
                  ext.atoms.data(),
                  ext.numQuartets * sizeof(std::array<int, 4>));
    }
    if (ext.numTorsions > 0) {
      std::memcpy(&system.torsionWeights[torsOffset[i]], ext.wts.data(), ext.numTorsions * sizeof(float));
      std::memcpy(&system.torsionMaxDevs[torsOffset[i]], ext.maxDevs.data(), ext.numTorsions * sizeof(float));
      std::memcpy(&system.torsionTypes[torsOffset[i]], ext.types.data(), ext.numTorsions * sizeof(TorsionType));
    }

    // QuartetStarts CSR: skip leading 0, add global quartet offset
    for (int j = 0; j < ext.numTorsions; ++j) {
      system.quartetStarts[1 + torsOffset[i] + j] = quartetOffset[i] + ext.qStarts[j + 1];
    }

    // MolDescriptor
    MolDescriptor desc;
    desc.confStart           = confOffset[i];
    desc.numConformers       = ext.numConformers;
    desc.quartetStart        = quartetOffset[i];
    desc.numQuartets         = ext.numQuartets;
    desc.dihedStart          = dihedOffset[i];
    desc.torsStart           = torsOffset[i];
    desc.numTorsions         = ext.numTorsions;
    desc.tfdOutStart         = tfdOutOffset[i];
    system.molDescriptors[i] = desc;

    // Work CSRs
    system.dihedralWorkStarts[1 + i] = dihedOffset[i] + ext.numConformers * ext.numQuartets;
    system.tfdWorkStarts[1 + i]      = tfdOutOffset[i] + ext.numConformers * (ext.numConformers - 1) / 2;
  }

  return system;
}

TFDSystemHost buildTFDSystem(const RDKit::ROMol& mol, const TFDComputeOptions& options) {
  return buildTFDSystemImpl(mol, options);
}

}  // namespace nvMolKit
