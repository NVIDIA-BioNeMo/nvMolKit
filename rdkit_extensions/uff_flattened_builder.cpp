#include "rdkit_extensions/uff_flattened_builder.h"

#include <ForceField/UFF/AngleBend.h>
#include <ForceField/UFF/BondStretch.h>
#include <ForceField/UFF/Nonbonded.h>
#include <ForceField/UFF/Params.h>
#include <ForceField/UFF/TorsionAngle.h>
#include <ForceField/UFF/Utils.h>
#include <GraphMol/Atom.h>
#include <GraphMol/ForceFieldHelpers/UFF/AtomTyper.h>
#include <GraphMol/ForceFieldHelpers/UFF/Builder.h>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <RDGeneral/Invariant.h>

#include <cmath>
#include <tuple>
#include <vector>

namespace RDKit {
namespace UFF {
using namespace ForceFields::UFF;

namespace Tools {

namespace {
void addAngleTerm(nvMolKit::UFF::EnergyForceContribsHost& contribs,
                  const unsigned int                      idx1,
                  const unsigned int                      idx2,
                  const unsigned int                      idx3,
                  const double                            bondOrder12,
                  const double                            bondOrder23,
                  const AtomicParams*                     at1Params,
                  const AtomicParams*                     at2Params,
                  const AtomicParams*                     at3Params,
                  unsigned int                            order) {
  double theta0 = at2Params->theta0;
  if (order >= 30) {
    switch (order) {
      case 30:
        theta0 = 150.0 / 180.0 * M_PI;
        break;
      case 35:
        theta0 = 60.0 / 180.0 * M_PI;
        break;
      case 40:
        theta0 = 135.0 / 180.0 * M_PI;
        break;
      case 45:
        theta0 = 90.0 / 180.0 * M_PI;
        break;
      default:
        break;
    }
    order = 0;
  }

  const double forceConstant =
    ForceFields::UFF::Utils::calcAngleForceConstant(theta0, bondOrder12, bondOrder23, at1Params, at2Params, at3Params);
  double C0 = 0.0;
  double C1 = 0.0;
  double C2 = 0.0;
  if (order == 0) {
    const double sinTheta0 = std::sin(theta0);
    const double cosTheta0 = std::cos(theta0);
    C2                    = 1.0 / (4.0 * std::max(sinTheta0 * sinTheta0, 1.0e-8));
    C1                    = -4.0 * C2 * cosTheta0;
    C0                    = C2 * (2.0 * cosTheta0 * cosTheta0 + 1.0);
  }

  auto& terms = contribs.angleTerms;
  terms.idx1.push_back(static_cast<int>(idx1));
  terms.idx2.push_back(static_cast<int>(idx2));
  terms.idx3.push_back(static_cast<int>(idx3));
  terms.theta0.push_back(theta0);
  terms.forceConstant.push_back(forceConstant);
  terms.order.push_back(static_cast<std::uint8_t>(order));
  terms.C0.push_back(C0);
  terms.C1.push_back(C1);
  terms.C2.push_back(C2);
}

std::tuple<double, std::uint8_t, double> calcTorsionParams(const double              bondOrder23,
                                                           const int                 atNum2,
                                                           const int                 atNum3,
                                                           RDKit::Atom::HybridizationType hyb2,
                                                           RDKit::Atom::HybridizationType hyb3,
                                                           const AtomicParams*       at2Params,
                                                           const AtomicParams*       at3Params,
                                                           const bool                endAtomIsSP2) {
  PRECONDITION((hyb2 == RDKit::Atom::SP2 || hyb2 == RDKit::Atom::SP3) &&
                   (hyb3 == RDKit::Atom::SP2 || hyb3 == RDKit::Atom::SP3),
               "bad hybridizations");

  if (hyb2 == RDKit::Atom::SP3 && hyb3 == RDKit::Atom::SP3) {
    double       forceConstant = std::sqrt(at2Params->V1 * at3Params->V1);
    std::uint8_t order         = 3;
    double       cosTerm       = -1.0;

    if (bondOrder23 == 1.0 && ForceFields::UFF::Utils::isInGroup6(atNum2) &&
        ForceFields::UFF::Utils::isInGroup6(atNum3)) {
      double V2 = 6.8;
      double V3 = 6.8;
      if (atNum2 == 8) {
        V2 = 2.0;
      }
      if (atNum3 == 8) {
        V3 = 2.0;
      }
      forceConstant = std::sqrt(V2 * V3);
      order         = 2;
      cosTerm       = -1.0;
    }
    return {forceConstant, order, cosTerm};
  }

  if (hyb2 == RDKit::Atom::SP2 && hyb3 == RDKit::Atom::SP2) {
    return {ForceFields::UFF::Utils::equation17(bondOrder23, at2Params, at3Params), 2, 1.0};
  }

  double       forceConstant = 1.0;
  std::uint8_t order         = 6;
  double       cosTerm       = 1.0;
  if (bondOrder23 == 1.0) {
    if ((hyb2 == RDKit::Atom::SP3 && ForceFields::UFF::Utils::isInGroup6(atNum2) &&
         !ForceFields::UFF::Utils::isInGroup6(atNum3)) ||
        (hyb3 == RDKit::Atom::SP3 && ForceFields::UFF::Utils::isInGroup6(atNum3) &&
         !ForceFields::UFF::Utils::isInGroup6(atNum2))) {
      forceConstant = ForceFields::UFF::Utils::equation17(bondOrder23, at2Params, at3Params);
      order         = 2;
      cosTerm       = -1.0;
    } else if (endAtomIsSP2) {
      forceConstant = 2.0;
      order         = 3;
      cosTerm       = -1.0;
    }
  }
  return {forceConstant, order, cosTerm};
}
}  // namespace

void addBonds(const ROMol& mol, const AtomicParamVect& params, nvMolKit::UFF::EnergyForceContribsHost& contribs) {
  PRECONDITION(mol.getNumAtoms() == params.size(), "bad parameters");
  for (ROMol::ConstBondIterator bi = mol.beginBonds(); bi != mol.endBonds(); ++bi) {
    const unsigned int idx1 = (*bi)->getBeginAtomIdx();
    const unsigned int idx2 = (*bi)->getEndAtomIdx();
    if (!params[idx1] || !params[idx2]) {
      continue;
    }
    const double bondOrder = (*bi)->getBondTypeAsDouble();
    auto&        terms     = contribs.bondTerms;
    terms.idx1.push_back(static_cast<int>(idx1));
    terms.idx2.push_back(static_cast<int>(idx2));
    terms.restLen.push_back(ForceFields::UFF::Utils::calcBondRestLength(bondOrder, params[idx1], params[idx2]));
    terms.forceConstant.push_back(
      ForceFields::UFF::Utils::calcBondForceConstant(terms.restLen.back(), params[idx1], params[idx2]));
  }
}

void addAngles(const ROMol& mol, const AtomicParamVect& params, nvMolKit::UFF::EnergyForceContribsHost& contribs) {
  PRECONDITION(mol.getNumAtoms() == params.size(), "bad parameters");
  RingInfo* ringInfo = mol.getRingInfo();
  for (unsigned int j = 0; j < mol.getNumAtoms(); ++j) {
    if (!params[j]) {
      continue;
    }
    const Atom* atomJ = mol.getAtomWithIdx(j);
    if (atomJ->getDegree() == 1) {
      continue;
    }
    ROMol::ADJ_ITER nbr1Idx, end1Nbrs, nbr2Idx, end2Nbrs;
    boost::tie(nbr1Idx, end1Nbrs) = mol.getAtomNeighbors(atomJ);
    for (; nbr1Idx != end1Nbrs; ++nbr1Idx) {
      const Atom* atomI = mol[*nbr1Idx];
      const auto  i     = atomI->getIdx();
      if (!params[i]) {
        continue;
      }
      boost::tie(nbr2Idx, end2Nbrs) = mol.getAtomNeighbors(atomJ);
      for (; nbr2Idx != end2Nbrs; ++nbr2Idx) {
        if (nbr2Idx < (nbr1Idx + 1)) {
          continue;
        }
        const Atom* atomK = mol[*nbr2Idx];
        const auto  k     = atomK->getIdx();
        if (!params[k]) {
          continue;
        }
        if (atomJ->getHybridization() == Atom::SP3D && atomJ->getDegree() == 5) {
          continue;
        }

        unsigned int order = 0;
        switch (atomJ->getHybridization()) {
          case Atom::SP:
            order = 1;
            break;
          case Atom::SP2:
            order = 3;
            if (ringInfo->isAtomInRingOfSize(j, 3)) {
              if (ringInfo->isAtomInRingOfSize(i, 3) != ringInfo->isAtomInRingOfSize(k, 3)) {
                order = 30;
              } else if (ringInfo->isAtomInRingOfSize(i, 3) && ringInfo->isAtomInRingOfSize(k, 3)) {
                order = 35;
              }
            } else if (ringInfo->isAtomInRingOfSize(j, 4)) {
              if (ringInfo->isAtomInRingOfSize(i, 4) != ringInfo->isAtomInRingOfSize(k, 4)) {
                order = 40;
              } else if (ringInfo->isAtomInRingOfSize(i, 4) && ringInfo->isAtomInRingOfSize(k, 4)) {
                order = 45;
              }
            }
            break;
          case Atom::SP3D2:
            order = 4;
            break;
          default:
            break;
        }

        const Bond* b1 = mol.getBondBetweenAtoms(i, j);
        const Bond* b2 = mol.getBondBetweenAtoms(k, j);
        addAngleTerm(
          contribs, i, j, k, b1->getBondTypeAsDouble(), b2->getBondTypeAsDouble(), params[i], params[j], params[k], order);
      }
    }
  }
}

void addTrigonalBipyramidAngles(const Atom* atom,
                                const ROMol& mol,
                                const int confId,
                                const AtomicParamVect& params,
                                nvMolKit::UFF::EnergyForceContribsHost& contribs) {
  PRECONDITION(atom, "bad atom");
  PRECONDITION(atom->getHybridization() == Atom::SP3D, "bad hybridization");
  PRECONDITION(atom->getDegree() == 5, "bad degree");

  const Conformer& conf = mol.getConformer(confId);
  const Bond *ax1 = nullptr, *ax2 = nullptr, *eq1 = nullptr, *eq2 = nullptr, *eq3 = nullptr;
  double mostNeg = 100.0;
  const unsigned int atomIdx = atom->getIdx();

  ROMol::OEDGE_ITER beg1, end1;
  boost::tie(beg1, end1) = mol.getAtomBonds(atom);
  while (beg1 != end1) {
    const Bond* bond1 = mol[*beg1];
    const unsigned int other1 = bond1->getOtherAtomIdx(atomIdx);
    const auto v1 = conf.getAtomPos(atomIdx).directionVector(conf.getAtomPos(other1));
    ROMol::OEDGE_ITER beg2, end2;
    boost::tie(beg2, end2) = mol.getAtomBonds(atom);
    while (beg2 != end2) {
      const Bond* bond2 = mol[*beg2];
      if (bond2->getIdx() > bond1->getIdx()) {
        const unsigned int other2 = bond2->getOtherAtomIdx(atomIdx);
        const auto v2 = conf.getAtomPos(atomIdx).directionVector(conf.getAtomPos(other2));
        const double dot = v1.dotProduct(v2);
        if (dot < mostNeg) {
          mostNeg = dot;
          ax1     = bond1;
          ax2     = bond2;
        }
      }
      ++beg2;
    }
    ++beg1;
  }
  CHECK_INVARIANT(ax1, "axial bond not found");
  CHECK_INVARIANT(ax2, "axial bond not found");

  boost::tie(beg1, end1) = mol.getAtomBonds(atom);
  while (beg1 != end1) {
    const Bond* bond = mol[*beg1];
    ++beg1;
    if (bond == ax1 || bond == ax2) {
      continue;
    }
    if (!eq1) {
      eq1 = bond;
    } else if (!eq2) {
      eq2 = bond;
    } else {
      eq3 = bond;
    }
  }

  CHECK_INVARIANT(eq1, "equatorial bond not found");
  CHECK_INVARIANT(eq2, "equatorial bond not found");
  CHECK_INVARIANT(eq3, "equatorial bond not found");

  auto maybeAdd = [&](const Bond* lhs, const Bond* rhs, unsigned int order) {
    const auto i = lhs->getOtherAtomIdx(atomIdx);
    const auto j = rhs->getOtherAtomIdx(atomIdx);
    if (!params[i] || !params[j]) {
      return;
    }
    addAngleTerm(contribs,
                 i,
                 atomIdx,
                 j,
                 lhs->getBondTypeAsDouble(),
                 rhs->getBondTypeAsDouble(),
                 params[i],
                 params[atomIdx],
                 params[j],
                 order);
  };

  maybeAdd(ax1, ax2, 2);
  maybeAdd(eq1, eq2, 3);
  maybeAdd(eq1, eq3, 3);
  maybeAdd(eq2, eq3, 3);
  maybeAdd(ax1, eq1, 0);
  maybeAdd(ax1, eq2, 0);
  maybeAdd(ax1, eq3, 0);
  maybeAdd(ax2, eq1, 0);
  maybeAdd(ax2, eq2, 0);
  maybeAdd(ax2, eq3, 0);
}

void addAngleSpecialCases(const ROMol& mol,
                          const int confId,
                          const AtomicParamVect& params,
                          nvMolKit::UFF::EnergyForceContribsHost& contribs) {
  PRECONDITION(mol.getNumAtoms() == params.size(), "bad parameters");
  for (unsigned int i = 0; i < mol.getNumAtoms(); ++i) {
    const Atom* atom = mol.getAtomWithIdx(i);
    if (atom->getHybridization() == Atom::SP3D && atom->getDegree() == 5) {
      addTrigonalBipyramidAngles(atom, mol, confId, params, contribs);
    }
  }
}

void addNonbonded(const ROMol& mol,
                  const int confId,
                  const AtomicParamVect& params,
                  nvMolKit::UFF::EnergyForceContribsHost& contribs,
                  boost::shared_array<std::uint8_t> neighborMatrix,
                  const double vdwThresh,
                  const bool ignoreInterfragInteractions) {
  PRECONDITION(mol.getNumAtoms() == params.size(), "bad parameters");
  INT_VECT fragMapping;
  if (ignoreInterfragInteractions) {
    std::vector<ROMOL_SPTR> molFrags = MolOps::getMolFrags(mol, true, &fragMapping);
    (void)molFrags;
  }

  const Conformer& conf = mol.getConformer(confId);
  for (unsigned int i = 0; i < mol.getNumAtoms(); ++i) {
    if (!params[i]) {
      continue;
    }
    for (unsigned int j = i + 1; j < mol.getNumAtoms(); ++j) {
      if (!params[j] || (ignoreInterfragInteractions && fragMapping[i] != fragMapping[j])) {
        continue;
      }
      if (RDKit::UFF::Tools::getTwoBitCell(neighborMatrix, RDKit::UFF::Tools::twoBitCellPos(mol.getNumAtoms(), i, j)) >=
          RDKit::UFF::Tools::RELATION_1_4) {
        const double xij = ForceFields::UFF::Utils::calcNonbondedMinimum(params[i], params[j]);
        const double threshold = vdwThresh * xij;
        const double dist = (conf.getAtomPos(i) - conf.getAtomPos(j)).length();
        if (dist < threshold) {
          auto& terms = contribs.vdwTerms;
          terms.idx1.push_back(static_cast<int>(i));
          terms.idx2.push_back(static_cast<int>(j));
          terms.x_ij.push_back(xij);
          terms.wellDepth.push_back(ForceFields::UFF::Utils::calcNonbondedDepth(params[i], params[j]));
          terms.threshold.push_back(threshold);
        }
      }
    }
  }
}

void addTorsions(const ROMol& mol, const AtomicParamVect& params, nvMolKit::UFF::EnergyForceContribsHost& contribs) {
  PRECONDITION(mol.getNumAtoms() == params.size(), "bad parameters");

  std::vector<MatchVectType> matchVect;
  const ROMol* query = RDKit::UFF::Tools::DefaultTorsionBondSmarts::query();
  PRECONDITION(query, "missing default torsion SMARTS");
  const unsigned int nHits = SubstructMatch(mol, *query, matchVect);
  for (unsigned int hitIdx = 0; hitIdx < nHits; ++hitIdx) {
    const MatchVectType& match = matchVect[hitIdx];
    PRECONDITION(match.size() == 2, "unexpected torsion match size");
    const int idx1 = match[0].second;
    const int idx2 = match[1].second;
    if (!params[idx1] || !params[idx2]) {
      continue;
    }

    const Bond* bond = mol.getBondBetweenAtoms(idx1, idx2);
    PRECONDITION(bond, "missing torsion bond");
    const Atom* atom1 = mol.getAtomWithIdx(idx1);
    const Atom* atom2 = mol.getAtomWithIdx(idx2);
    std::vector<size_t> contribIndices;

    if (!((atom1->getHybridization() == Atom::SP2 || atom1->getHybridization() == Atom::SP3) &&
          (atom2->getHybridization() == Atom::SP2 || atom2->getHybridization() == Atom::SP3))) {
      continue;
    }

    ROMol::OEDGE_ITER beg1, end1;
    boost::tie(beg1, end1) = mol.getAtomBonds(atom1);
    while (beg1 != end1) {
      const Bond* tBond1 = mol[*beg1];
      ++beg1;
      if (tBond1 == bond) {
        continue;
      }
      const int bIdx = tBond1->getOtherAtomIdx(idx1);
      ROMol::OEDGE_ITER beg2, end2;
      boost::tie(beg2, end2) = mol.getAtomBonds(atom2);
      while (beg2 != end2) {
        const Bond* tBond2 = mol[*beg2];
        ++beg2;
        if (tBond2 == bond || tBond2 == tBond1) {
          continue;
        }
        const int eIdx = tBond2->getOtherAtomIdx(idx2);
        if (eIdx == bIdx) {
          continue;
        }

        bool hasSP2 = false;
        if (mol.getAtomWithIdx(bIdx)->getHybridization() == Atom::SP2 ||
            mol.getAtomWithIdx(eIdx)->getHybridization() == Atom::SP2) {
          hasSP2 = true;
        }

        const auto [forceConstant, order, cosTerm] =
          calcTorsionParams(bond->getBondTypeAsDouble(),
                            atom1->getAtomicNum(),
                            atom2->getAtomicNum(),
                            atom1->getHybridization(),
                            atom2->getHybridization(),
                            params[idx1],
                            params[idx2],
                            hasSP2);
        auto& terms = contribs.torsionTerms;
        terms.idx1.push_back(bIdx);
        terms.idx2.push_back(idx1);
        terms.idx3.push_back(idx2);
        terms.idx4.push_back(eIdx);
        terms.forceConstant.push_back(forceConstant);
        terms.order.push_back(order);
        terms.cosTerm.push_back(cosTerm);
        contribIndices.push_back(terms.forceConstant.size() - 1);
      }
    }

    if (!contribIndices.empty()) {
      const double scale = static_cast<double>(contribIndices.size());
      for (const auto contribIdx : contribIndices) {
        contribs.torsionTerms.forceConstant[contribIdx] /= scale;
      }
    }
  }
}

void addInversions(const ROMol& mol, const AtomicParamVect& params, nvMolKit::UFF::EnergyForceContribsHost& contribs) {
  PRECONDITION(mol.getNumAtoms() == params.size(), "bad parameters");

  unsigned int idx[4];
  unsigned int perm[4];
  const Atom*  atom[4];
  ROMol::ADJ_ITER nbrIdx, endNbrs;
  for (idx[1] = 0; idx[1] < mol.getNumAtoms(); ++idx[1]) {
    atom[1] = mol.getAtomWithIdx(idx[1]);
    const int at2AtomicNum = atom[1]->getAtomicNum();
    if (((at2AtomicNum != 6) && (at2AtomicNum != 7) && (at2AtomicNum != 8) && (at2AtomicNum != 15) &&
         (at2AtomicNum != 33) && (at2AtomicNum != 51) && (at2AtomicNum != 83)) ||
        atom[1]->getDegree() != 3) {
      continue;
    }
    if (((at2AtomicNum == 6) || (at2AtomicNum == 7) || (at2AtomicNum == 8)) &&
        atom[1]->getHybridization() != Atom::SP2) {
      continue;
    }

    boost::tie(nbrIdx, endNbrs) = mol.getAtomNeighbors(atom[1]);
    unsigned int neighborSlot = 0;
    bool         isBoundToSP2O = false;
    for (; nbrIdx != endNbrs; ++nbrIdx) {
      atom[neighborSlot] = mol[*nbrIdx];
      idx[neighborSlot]  = atom[neighborSlot]->getIdx();
      if (!isBoundToSP2O) {
        isBoundToSP2O = (at2AtomicNum == 6) && (atom[neighborSlot]->getAtomicNum() == 8) &&
                        (atom[neighborSlot]->getHybridization() == Atom::SP2);
      }
      if (!neighborSlot) {
        ++neighborSlot;
      }
      ++neighborSlot;
    }

    const auto [forceConstant, C0, C1, C2] =
      ForceFields::UFF::Utils::calcInversionCoefficientsAndForceConstant(at2AtomicNum, isBoundToSP2O);
    for (unsigned int i = 0; i < 3; ++i) {
      perm[1] = 1;
      switch (i) {
        case 0:
          perm[0] = 0;
          perm[2] = 2;
          perm[3] = 3;
          break;
        case 1:
          perm[0] = 0;
          perm[2] = 3;
          perm[3] = 2;
          break;
        default:
          perm[0] = 2;
          perm[2] = 3;
          perm[3] = 0;
          break;
      }
      auto& terms = contribs.inversionTerms;
      terms.idx1.push_back(idx[perm[0]]);
      terms.idx2.push_back(idx[perm[1]]);
      terms.idx3.push_back(idx[perm[2]]);
      terms.idx4.push_back(idx[perm[3]]);
      terms.forceConstant.push_back(forceConstant);
      terms.C0.push_back(C0);
      terms.C1.push_back(C1);
      terms.C2.push_back(C2);
    }
  }
}

}  // namespace Tools

}  // namespace UFF
}  // namespace RDKit

namespace nvMolKit {
namespace UFF {

EnergyForceContribsHost constructForcefieldContribs(RDKit::ROMol& mol,
                                                    const double   vdwThresh,
                                                    const int      confId,
                                                    const bool     ignoreInterfragInteractions) {
  bool foundAll = false;
  RDKit::UFF::AtomicParamVect params;
  std::tie(params, foundAll) = RDKit::UFF::getAtomTypes(mol);
  PRECONDITION(foundAll, "missing atom types - invalid force-field");

  EnergyForceContribsHost contribs;
  RDKit::UFF::Tools::addBonds(mol, params, contribs);
  RDKit::UFF::Tools::addAngles(mol, params, contribs);
  RDKit::UFF::Tools::addAngleSpecialCases(mol, confId, params, contribs);
  auto neighborMat = RDKit::UFF::Tools::buildNeighborMatrix(mol);
  RDKit::UFF::Tools::addNonbonded(mol, confId, params, contribs, neighborMat, vdwThresh, ignoreInterfragInteractions);
  RDKit::UFF::Tools::addTorsions(mol, params, contribs);
  RDKit::UFF::Tools::addInversions(mol, params, contribs);
  return contribs;
}

}  // namespace UFF
}  // namespace nvMolKit
