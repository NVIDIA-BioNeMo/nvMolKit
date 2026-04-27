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

#include "rdkit_extensions/mmff_flattened_builder.h"

#include <ForceField/MMFF/Contribs.h>
#include <ForceField/MMFF/Params.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>
#include <GraphMol/ForceFieldHelpers/MMFF/Builder.h>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <RDGeneral/Invariant.h>

#include <iostream>
#include <vector>

// Don't lint adapted RDKit code
// NOLINTBEGIN
namespace RDKit {
namespace MMFF {
using namespace ForceFields::MMFF;

namespace Tools {
// ------------------------------------------------------------------------
// The following functions in this namespace are adapted from RDKit code
// ------------------------------------------------------------------------
void addBonds(const ROMol&                             mol,
              MMFFMolProperties*                       mmffMolProperties,
              nvMolKit::MMFF::EnergyForceContribsHost& contribs) {
  PRECONDITION(mmffMolProperties, "bad MMFFMolProperties");
  PRECONDITION(mmffMolProperties->isValid(), "missing atom types - invalid force-field");

  for (ROMol::ConstBondIterator bi = mol.beginBonds(); bi != mol.endBonds(); ++bi) {
    unsigned int idx1 = (*bi)->getBeginAtomIdx();
    unsigned int idx2 = (*bi)->getEndAtomIdx();
    unsigned int bondType;
    MMFFBond     mmffBondParams;
    if (mmffMolProperties->getMMFFBondStretchParams(mol, idx1, idx2, bondType, mmffBondParams)) {
      contribs.bondTerms.idx1.push_back(idx1);
      contribs.bondTerms.idx2.push_back(idx2);
      contribs.bondTerms.kb.push_back(mmffBondParams.kb);
      contribs.bondTerms.r0.push_back(mmffBondParams.r0);
    }
  }
}

unsigned int twoBitCellPos(unsigned int nAtoms, int i, int j) {
  if (j < i) {
    std::swap(i, j);
  }
  return i * (nAtoms - 1) + i * (1 - i) / 2 + j;
}

void setTwoBitCell(std::vector<std::uint8_t>& res, unsigned int pos, std::uint8_t value) {
  unsigned int twoBitPos  = pos / 4;
  unsigned int shift      = 2 * (pos % 4);
  std::uint8_t twoBitMask = 3 << shift;
  res[twoBitPos]          = ((res[twoBitPos] & (~twoBitMask)) | (value << shift));
}

std::uint8_t getTwoBitCell(const std::vector<std::uint8_t>& res, unsigned int pos) {
  unsigned int twoBitPos  = pos / 4;
  unsigned int shift      = 2 * (pos % 4);
  std::uint8_t twoBitMask = 3 << shift;
  return ((res[twoBitPos] & twoBitMask) >> shift);
}

// ------------------------------------------------------------------------
//
// the two-bit matrix returned by this contains:
//   0: if atoms i and j are directly connected
//   1: if atoms i and j are connected via an atom
//   2: if atoms i and j are in a 1,4 relationship
//   3: otherwise
//
//  NOTE: the caller is responsible for calling delete []
//  on the result
//
// ------------------------------------------------------------------------
std::vector<std::uint8_t> buildNeighborMatrixInternal(const ROMol& mol) {
  const std::uint8_t RELATION_1_X_INIT = RELATION_1_X | (RELATION_1_X << 2) | (RELATION_1_X << 4) | (RELATION_1_X << 6);
  unsigned int       nAtoms            = mol.getNumAtoms();
  unsigned           nTwoBitCells      = (nAtoms * (nAtoms + 1) - 1) / 8 + 1;
  std::vector<std::uint8_t> res(nTwoBitCells, RELATION_1_X_INIT);
  for (ROMol::ConstBondIterator bondi = mol.beginBonds(); bondi != mol.endBonds(); ++bondi) {
    setTwoBitCell(res, twoBitCellPos(nAtoms, (*bondi)->getBeginAtomIdx(), (*bondi)->getEndAtomIdx()), RELATION_1_2);
    unsigned int bondiBeginAtomIdx = (*bondi)->getBeginAtomIdx();
    unsigned int bondiEndAtomIdx   = (*bondi)->getEndAtomIdx();
    for (ROMol::ConstBondIterator bondj = bondi; ++bondj != mol.endBonds();) {
      int          idx1              = -1;
      int          idx3              = -1;
      unsigned int bondjBeginAtomIdx = (*bondj)->getBeginAtomIdx();
      unsigned int bondjEndAtomIdx   = (*bondj)->getEndAtomIdx();
      if (bondiBeginAtomIdx == bondjBeginAtomIdx) {
        idx1 = bondiEndAtomIdx;
        idx3 = bondjEndAtomIdx;
      } else if (bondiBeginAtomIdx == bondjEndAtomIdx) {
        idx1 = bondiEndAtomIdx;
        idx3 = bondjBeginAtomIdx;
      } else if (bondiEndAtomIdx == bondjBeginAtomIdx) {
        idx1 = bondiBeginAtomIdx;
        idx3 = bondjEndAtomIdx;
      } else if (bondiEndAtomIdx == bondjEndAtomIdx) {
        idx1 = bondiBeginAtomIdx;
        idx3 = bondjBeginAtomIdx;
      } else {
        // check if atoms i and j are in a 1,4-relationship
        if ((mol.getBondBetweenAtoms(bondiBeginAtomIdx, bondjBeginAtomIdx)) &&
            (getTwoBitCell(res, twoBitCellPos(nAtoms, bondiEndAtomIdx, bondjEndAtomIdx)) == RELATION_1_X)) {
          setTwoBitCell(res, twoBitCellPos(nAtoms, bondiEndAtomIdx, bondjEndAtomIdx), RELATION_1_4);
        } else if ((mol.getBondBetweenAtoms(bondiBeginAtomIdx, bondjEndAtomIdx)) &&
                   (getTwoBitCell(res, twoBitCellPos(nAtoms, bondiEndAtomIdx, bondjBeginAtomIdx)) == RELATION_1_X)) {
          setTwoBitCell(res, twoBitCellPos(nAtoms, bondiEndAtomIdx, bondjBeginAtomIdx), RELATION_1_4);
        } else if ((mol.getBondBetweenAtoms(bondiEndAtomIdx, bondjBeginAtomIdx)) &&
                   (getTwoBitCell(res, twoBitCellPos(nAtoms, bondiBeginAtomIdx, bondjEndAtomIdx)) == RELATION_1_X)) {
          setTwoBitCell(res, twoBitCellPos(nAtoms, bondiBeginAtomIdx, bondjEndAtomIdx), RELATION_1_4);
        } else if ((mol.getBondBetweenAtoms(bondiEndAtomIdx, bondjEndAtomIdx)) &&
                   (getTwoBitCell(res, twoBitCellPos(nAtoms, bondiBeginAtomIdx, bondjBeginAtomIdx)) == RELATION_1_X)) {
          setTwoBitCell(res, twoBitCellPos(nAtoms, bondiBeginAtomIdx, bondjBeginAtomIdx), RELATION_1_4);
        }
      }
      if (idx1 > -1) {
        setTwoBitCell(res, twoBitCellPos(nAtoms, idx1, idx3), RELATION_1_3);
      }
    }
  }
  return res;
}

void addAngles(const ROMol&                             mol,
               MMFFMolProperties*                       mmffMolProperties,
               nvMolKit::MMFF::EnergyForceContribsHost& contribs) {
  PRECONDITION(mmffMolProperties, "bad MMFFMolProperties");
  PRECONDITION(mmffMolProperties->isValid(), "missing atom types - invalid force-field");

  unsigned int              idx[3];
  const MMFFPropCollection* mmffProp = DefaultParameters::getMMFFProp();
  ROMol::ADJ_ITER           nbr1Idx;
  ROMol::ADJ_ITER           end1Nbrs;
  ROMol::ADJ_ITER           nbr2Idx;
  ROMol::ADJ_ITER           end2Nbrs;

  unsigned int nAtoms = mol.getNumAtoms();

  for (idx[1] = 0; idx[1] < nAtoms; ++idx[1]) {
    const Atom* jAtom = mol.getAtomWithIdx(idx[1]);
    if (jAtom->getDegree() == 1) {
      continue;
    }
    unsigned int    jAtomType                 = mmffMolProperties->getMMFFAtomType(idx[1]);
    const MMFFProp* mmffPropParamsCentralAtom = (*mmffProp)(jAtomType);
    boost::tie(nbr1Idx, end1Nbrs)             = mol.getAtomNeighbors(jAtom);
    for (; nbr1Idx != end1Nbrs; ++nbr1Idx) {
      const Atom* iAtom             = mol[*nbr1Idx];
      idx[0]                        = iAtom->getIdx();
      boost::tie(nbr2Idx, end2Nbrs) = mol.getAtomNeighbors(jAtom);
      for (; nbr2Idx != end2Nbrs; ++nbr2Idx) {
        if (nbr2Idx < (nbr1Idx + 1)) {
          continue;
        }
        const Atom* kAtom = mol[*nbr2Idx];
        idx[2]            = kAtom->getIdx();
        unsigned int angleType;
        MMFFAngle    mmffAngleParams;
        if (mmffMolProperties->getMMFFAngleBendParams(mol, idx[0], idx[1], idx[2], angleType, mmffAngleParams)) {
          nvMolKit::MMFF::AngleBendTerms& contrib = contribs.angleTerms;
          contrib.idx1.push_back(idx[0]);
          contrib.idx2.push_back(idx[1]);
          contrib.idx3.push_back(idx[2]);
          contrib.ka.push_back(mmffAngleParams.ka);
          contrib.theta0.push_back(mmffAngleParams.theta0);
          contrib.isLinear.push_back(mmffPropParamsCentralAtom->linh > 0u);
        }
      }
    }
  }
}

void addStretchBend(const ROMol&                             mol,
                    MMFFMolProperties*                       mmffMolProperties,
                    nvMolKit::MMFF::EnergyForceContribsHost& contribs) {
  PRECONDITION(mmffMolProperties, "bad MMFFMolProperties");
  PRECONDITION(mmffMolProperties->isValid(), "missing atom types - invalid force-field");

  unsigned int              idx[3];
  const MMFFPropCollection* mmffProp = DefaultParameters::getMMFFProp();
  ROMol::ADJ_ITER           nbr1Idx;
  ROMol::ADJ_ITER           end1Nbrs;
  ROMol::ADJ_ITER           nbr2Idx;
  ROMol::ADJ_ITER           end2Nbrs;

  unsigned int nAtoms = mol.getNumAtoms();

  for (idx[1] = 0; idx[1] < nAtoms; ++idx[1]) {
    const Atom* jAtom = mol.getAtomWithIdx(idx[1]);
    if (jAtom->getDegree() == 1) {
      continue;
    }
    unsigned int    jAtomType                 = mmffMolProperties->getMMFFAtomType(idx[1]);
    const MMFFProp* mmffPropParamsCentralAtom = (*mmffProp)(jAtomType);
    if (mmffPropParamsCentralAtom->linh) {
      continue;
    }
    boost::tie(nbr1Idx, end1Nbrs) = mol.getAtomNeighbors(jAtom);
    unsigned int i                = 0;
    for (; nbr1Idx != end1Nbrs; ++nbr1Idx) {
      const Atom* iAtom             = mol[*nbr1Idx];
      boost::tie(nbr2Idx, end2Nbrs) = mol.getAtomNeighbors(jAtom);
      unsigned int j                = 0;
      for (; nbr2Idx != end2Nbrs; ++nbr2Idx) {
        const Atom* kAtom = mol[*nbr2Idx];
        if (j < (i + 1)) {
          ++j;
          continue;
        }
        idx[0] = iAtom->getIdx();
        idx[2] = kAtom->getIdx();
        unsigned int stretchBendType;
        MMFFStbn     mmffStbnParams;
        MMFFBond     mmffBondParams[2];
        MMFFAngle    mmffAngleParams;
        if (mmffMolProperties->getMMFFStretchBendParams(mol,
                                                        idx[0],
                                                        idx[1],
                                                        idx[2],
                                                        stretchBendType,
                                                        mmffStbnParams,
                                                        mmffBondParams,
                                                        mmffAngleParams)) {
          auto& contrib = contribs.bendTerms;
          contrib.idx1.push_back(idx[0]);
          contrib.idx2.push_back(idx[1]);
          contrib.idx3.push_back(idx[2]);
          contrib.restLen1.push_back(ForceFields::MMFF::Utils::calcBondRestLength(&mmffBondParams[0]));
          contrib.restLen2.push_back(ForceFields::MMFF::Utils::calcBondRestLength(&mmffBondParams[1]));
          contrib.theta0.push_back(mmffAngleParams.theta0);
          contrib.forceConst1.push_back(mmffStbnParams.kbaIJK);
          contrib.forceConst2.push_back(mmffStbnParams.kbaKJI);
        }
        ++j;
      }
      ++i;
    }
  }
}

void addOop(const ROMol& mol, MMFFMolProperties* mmffMolProperties, nvMolKit::MMFF::EnergyForceContribsHost& contribs) {
  PRECONDITION(mmffMolProperties, "bad MMFFMolProperties");
  PRECONDITION(mmffMolProperties->isValid(), "missing atom types - invalid force-field");

  unsigned int    idx[4];
  unsigned int    n[4];
  const Atom*     atom[4];
  ROMol::ADJ_ITER nbrIdx;
  ROMol::ADJ_ITER endNbrs;

  for (idx[1] = 0; idx[1] < mol.getNumAtoms(); ++idx[1]) {
    atom[1] = mol.getAtomWithIdx(idx[1]);
    if (atom[1]->getDegree() != 3) {
      continue;
    }
    boost::tie(nbrIdx, endNbrs) = mol.getAtomNeighbors(atom[1]);
    {
      unsigned int i = 0;
      for (; nbrIdx != endNbrs; ++nbrIdx) {
        atom[i] = mol[*nbrIdx];
        idx[i]  = atom[i]->getIdx();
        if (!i) {
          ++i;
        }
        ++i;
      }
    }

    MMFFOop mmffOopParams;
    // if no parameters could be found, we exclude this term (SURDOX02)
    if (!(mmffMolProperties->getMMFFOopBendParams(mol, idx[0], idx[1], idx[2], idx[3], mmffOopParams))) {
      continue;
    }
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
      }
      auto& contrib = contribs.oopTerms;
      contrib.idx1.push_back(idx[n[0]]);
      contrib.idx2.push_back(idx[n[1]]);
      contrib.idx3.push_back(idx[n[2]]);
      contrib.idx4.push_back(idx[n[3]]);
      contrib.koop.push_back(mmffOopParams.koop);
    }
  }
}

void addTorsions(const ROMol&                             mol,
                 MMFFMolProperties*                       mmffMolProperties,
                 nvMolKit::MMFF::EnergyForceContribsHost& contribs) {
  PRECONDITION(mmffMolProperties, "bad MMFFMolProperties");
  PRECONDITION(mmffMolProperties->isValid(), "missing atom types - invalid force-field");
  const std::string    torsionBondSmarts = DefaultTorsionBondSmarts::string();
  ROMol::ADJ_ITER      nbr1Idx;
  ROMol::ADJ_ITER      end1Nbrs;
  ROMol::ADJ_ITER      nbr2Idx;
  ROMol::ADJ_ITER      end2Nbrs;
  RDGeom::PointPtrVect points;

  std::vector<MatchVectType> matchVect;
  const ROMol*               defaultQuery = DefaultTorsionBondSmarts::query();
  const ROMol*               query =
    (torsionBondSmarts == DefaultTorsionBondSmarts::string()) ? defaultQuery : SmartsToMol(torsionBondSmarts);
  TEST_ASSERT(query);
  unsigned int nHits = SubstructMatch(mol, *query, matchVect);
  if (query != defaultQuery) {
    delete query;
  }

  for (unsigned int i = 0; i < nHits; ++i) {
    MatchVectType match = matchVect[i];
    TEST_ASSERT(match.size() == 2);
    int         idx2 = match[0].second;
    int         idx3 = match[1].second;
    const Bond* bond = mol.getBondBetweenAtoms(idx2, idx3);
    TEST_ASSERT(bond);
    const Atom* jAtom = mol.getAtomWithIdx(idx2);
    const Atom* kAtom = mol.getAtomWithIdx(idx3);
    if (((jAtom->getHybridization() == Atom::SP2) || (jAtom->getHybridization() == Atom::SP3)) &&
        ((kAtom->getHybridization() == Atom::SP2) || (kAtom->getHybridization() == Atom::SP3))) {
      ROMol::OEDGE_ITER beg1, end1;
      boost::tie(beg1, end1) = mol.getAtomBonds(jAtom);
      while (beg1 != end1) {
        const Bond* tBond1 = mol[*beg1];
        if (tBond1 != bond) {
          int               idx1 = tBond1->getOtherAtomIdx(idx2);
          ROMol::OEDGE_ITER beg2, end2;
          boost::tie(beg2, end2) = mol.getAtomBonds(kAtom);
          while (beg2 != end2) {
            const Bond* tBond2 = mol[*beg2];
            if ((tBond2 != bond) && (tBond2 != tBond1)) {
              int idx4 = tBond2->getOtherAtomIdx(idx3);
              // make sure this isn't a three-membered ring:
              if (idx4 != idx1) {
                // we now have a torsion involving atoms (bonds):
                //  bIdx - (tBond1) - idx1 - (bond) - idx2 - (tBond2) - eIdx
                unsigned int torType;
                MMFFTor      mmffTorParams;
                if (mmffMolProperties->getMMFFTorsionParams(mol, idx1, idx2, idx3, idx4, torType, mmffTorParams)) {
                  auto& contrib = contribs.torsionTerms;
                  contrib.idx1.push_back(idx1);
                  contrib.idx2.push_back(idx2);
                  contrib.idx3.push_back(idx3);
                  contrib.idx4.push_back(idx4);
                  contrib.V1.push_back(mmffTorParams.V1);
                  contrib.V2.push_back(mmffTorParams.V2);
                  contrib.V3.push_back(mmffTorParams.V3);
                }
              }
            }
            beg2++;
          }
        }
        beg1++;
      }
    }
  }
}

void addVdW(const ROMol&                             mol,
            int                                      confId,
            MMFFMolProperties*                       mmffMolProperties,
            nvMolKit::MMFF::EnergyForceContribsHost& contribs,
            std::vector<std::uint8_t>                neighborMatrix,
            double                                   nonBondedThresh,
            bool                                     ignoreInterfragInteractions) {
  PRECONDITION(mmffMolProperties, "bad MMFFMolProperties");
  PRECONDITION(mmffMolProperties->isValid(), "missing atom types - invalid force-field");

  INT_VECT fragMapping;
  if (ignoreInterfragInteractions) {
    std::ignore = MolOps::getMolFrags(mol, true, &fragMapping);
  }

  unsigned int     nAtoms = mol.getNumAtoms();
  const Conformer& conf   = mol.getConformer(confId);
  for (unsigned int i = 0; i < nAtoms; ++i) {
    for (unsigned int j = i + 1; j < nAtoms; ++j) {
      if (ignoreInterfragInteractions && (fragMapping[i] != fragMapping[j])) {
        continue;
      }
      if (getTwoBitCell(neighborMatrix, twoBitCellPos(nAtoms, i, j)) >= RELATION_1_4) {
        double dist = (conf.getAtomPos(i) - conf.getAtomPos(j)).length();
        if (dist > nonBondedThresh) {
          continue;
        }
        MMFFVdWRijstarEps mmffVdWConstants;
        if (mmffMolProperties->getMMFFVdWParams(i, j, mmffVdWConstants)) {
          auto& contrib = contribs.vdwTerms;
          contrib.idx1.push_back(i);
          contrib.idx2.push_back(j);
          contrib.R_ij_star.push_back(mmffVdWConstants.R_ij_star);
          contrib.wellDepth.push_back(mmffVdWConstants.epsilon);
        }
      }
    }
  }
}

void addEle(const ROMol&                             mol,
            int                                      confId,
            MMFFMolProperties*                       mmffMolProperties,
            nvMolKit::MMFF::EnergyForceContribsHost& contribs,
            std::vector<std::uint8_t>                neighborMatrix,
            double                                   nonBondedThresh,
            bool                                     ignoreInterfragInteractions) {
  PRECONDITION(mmffMolProperties, "bad MMFFMolProperties");
  PRECONDITION(mmffMolProperties->isValid(), "missing atom types - invalid force-field");

  INT_VECT fragMapping;
  if (ignoreInterfragInteractions) {
    std::ignore = MolOps::getMolFrags(mol, true, &fragMapping);
  }
  unsigned int nAtoms = mol.getNumAtoms();

  const Conformer& conf      = mol.getConformer(confId);
  double           dielConst = mmffMolProperties->getMMFFDielectricConstant();
  std::uint8_t     dielModel = mmffMolProperties->getMMFFDielectricModel();
  for (unsigned int i = 0; i < nAtoms; ++i) {
    for (unsigned int j = i + 1; j < nAtoms; ++j) {
      if (ignoreInterfragInteractions && (fragMapping[i] != fragMapping[j])) {
        continue;
      }
      std::uint8_t cell  = getTwoBitCell(neighborMatrix, twoBitCellPos(nAtoms, i, j));
      bool         is1_4 = (cell == RELATION_1_4);
      if (cell >= RELATION_1_4) {
        if (isDoubleZero(mmffMolProperties->getMMFFPartialCharge(i)) ||
            isDoubleZero(mmffMolProperties->getMMFFPartialCharge(j))) {
          continue;
        }
        double dist = (conf.getAtomPos(i) - conf.getAtomPos(j)).length();
        if (dist > nonBondedThresh) {
          continue;
        }
        double chargeTerm =
          mmffMolProperties->getMMFFPartialCharge(i) * mmffMolProperties->getMMFFPartialCharge(j) / dielConst;
        auto& contrib = contribs.eleTerms;
        contrib.idx1.push_back(i);
        contrib.idx2.push_back(j);
        contrib.chargeTerm.push_back(chargeTerm);
        contrib.dielModel.push_back(dielModel);
        contrib.is1_4.push_back(is1_4);
      }
    }
  }
}

}  // namespace Tools

}  // namespace MMFF
}  // namespace RDKit
// NOLINTEND

namespace nvMolKit {
namespace MMFF {

MMFF::EnergyForceContribsHost constructForcefieldContribs(const RDKit::ROMol&             mol,
                                                          RDKit::MMFF::MMFFMolProperties* mmffMolProperties,
                                                          double                          nonBondedThresh,
                                                          int                             confId,
                                                          bool                            ignoreInterfragInteractions) {
  PRECONDITION(mmffMolProperties, "bad MMFFMolProperties");
  PRECONDITION(mmffMolProperties->isValid(), "missing atom types - invalid force-field");
  MMFF::EnergyForceContribsHost contribs;

  if (mmffMolProperties->getMMFFBondTerm()) {
    RDKit::MMFF::Tools::addBonds(mol, mmffMolProperties, contribs);
  }
  if (mmffMolProperties->getMMFFAngleTerm()) {
    RDKit::MMFF::Tools::addAngles(mol, mmffMolProperties, contribs);
  }
  if (mmffMolProperties->getMMFFStretchBendTerm()) {
    RDKit::MMFF::Tools::addStretchBend(mol, mmffMolProperties, contribs);
  }
  if (mmffMolProperties->getMMFFOopTerm()) {
    RDKit::MMFF::Tools::addOop(mol, mmffMolProperties, contribs);
  }
  if (mmffMolProperties->getMMFFTorsionTerm()) {
    RDKit::MMFF::Tools::addTorsions(mol, mmffMolProperties, contribs);
  }
  if (mmffMolProperties->getMMFFVdWTerm() || mmffMolProperties->getMMFFEleTerm()) {
    const std::vector<std::uint8_t> neighborMat = RDKit::MMFF::Tools::buildNeighborMatrixInternal(mol);
    if (mmffMolProperties->getMMFFVdWTerm()) {
      RDKit::MMFF::Tools::addVdW(mol,
                                 confId,
                                 mmffMolProperties,
                                 contribs,
                                 neighborMat,
                                 nonBondedThresh,
                                 ignoreInterfragInteractions);
    }
    if (mmffMolProperties->getMMFFEleTerm()) {
      RDKit::MMFF::Tools::addEle(mol,
                                 confId,
                                 mmffMolProperties,
                                 contribs,
                                 neighborMat,
                                 nonBondedThresh,
                                 ignoreInterfragInteractions);
    }
  }

  return contribs;
}

MMFF::EnergyForceContribsHost constructForcefieldContribs(RDKit::ROMol& mol,
                                                          double        nonBondedThresh,
                                                          int           confId,
                                                          bool          ignoreInterfragInteractions) {
  RDKit::MMFF::MMFFMolProperties mmffMolProperties(mol);
  PRECONDITION(mmffMolProperties.isValid(), "missing atom types - invalid force-field");
  return constructForcefieldContribs(mol, &mmffMolProperties, nonBondedThresh, confId, ignoreInterfragInteractions);
}

MMFF::EnergyForceContribsHost constructForcefieldContribs(RDKit::ROMol&                mol,
                                                          const nvMolKit::MMFFProperties& props,
                                                          int                             confId) {
  RDKit::MMFF::MMFFMolProperties mmffMolProperties(mol, props.variant);
  PRECONDITION(mmffMolProperties.isValid(), "missing atom types - invalid force-field");
  mmffMolProperties.setMMFFVariant(props.variant);
  mmffMolProperties.setMMFFDielectricConstant(props.dielectricConstant);
  mmffMolProperties.setMMFFDielectricModel(props.dielectricModel);
  mmffMolProperties.setMMFFBondTerm(props.bondTerm);
  mmffMolProperties.setMMFFAngleTerm(props.angleTerm);
  mmffMolProperties.setMMFFStretchBendTerm(props.stretchBendTerm);
  mmffMolProperties.setMMFFOopTerm(props.oopTerm);
  mmffMolProperties.setMMFFTorsionTerm(props.torsionTerm);
  mmffMolProperties.setMMFFVdWTerm(props.vdwTerm);
  mmffMolProperties.setMMFFEleTerm(props.eleTerm);
  return constructForcefieldContribs(
    mol, &mmffMolProperties, props.nonBondedThreshold, confId, props.ignoreInterfragInteractions);
}

}  // namespace MMFF
}  // namespace nvMolKit
