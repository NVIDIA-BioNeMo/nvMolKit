#include "rdkit_extensions/conformer_pruning.h"

#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/MolAlign/AlignMolecules.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <Numerics/Alignment/AlignPoints.h>

#include <vector>

// Don't lint RDKit ports
// NOLINTBEGIN

namespace RDKit {
class Conformer;
} // namespace RDKit



namespace RDKit {

namespace DGeomHelpers {


std::vector<std::vector<unsigned int>> getMolSelfMatches(
    const ROMol &mol, const EmbedParameters &params) {
  std::vector<std::vector<unsigned int>> res;
  if (params.pruneRmsThresh && params.useSymmetryForPruning) {
    RWMol tmol(mol);
    MolOps::RemoveHsParameters ps;
    bool sanitize = false;
    MolOps::removeHs(tmol, ps, sanitize);

    std::unique_ptr<RWMol> prbMolSymm;
    if (params.symmetrizeConjugatedTerminalGroupsForPruning) {
      prbMolSymm.reset(new RWMol(tmol));
      MolAlign::details::symmetrizeTerminalAtoms(*prbMolSymm);
    }
    const auto &prbMolForMatch = prbMolSymm ? *prbMolSymm : tmol;

    SubstructMatchParameters sssps;
    sssps.maxMatches = 1;
    // provides the atom indices in the molecule corresponding
    // to the indices in the H-stripped version
    auto strippedMatch = SubstructMatch(mol, prbMolForMatch, sssps);
    CHECK_INVARIANT(strippedMatch.size() == 1, "expected match not found");

    sssps.maxMatches = 1000;
    sssps.uniquify = false;
    auto heavyAtomMatches = SubstructMatch(tmol, prbMolForMatch, sssps);
    for (const auto &match : heavyAtomMatches) {
      res.emplace_back(0);
      res.back().reserve(match.size());
      for (auto midx : match) {
        res.back().push_back(strippedMatch[0][midx.second].second);
      }
    }
  } else if (params.onlyHeavyAtomsForRMS) {
    res.emplace_back(0);
    for (const auto &at : mol.atoms()) {
      if (at->getAtomicNum() != 1) {
        res.back().push_back(at->getIdx());
      }
    }
  } else {
    res.emplace_back(0);
    res.back().reserve(mol.getNumAtoms());
    for (unsigned int i = 0; i < mol.getNumAtoms(); ++i) {
      res.back().push_back(i);
    }
  }
  return res;
}

void _fillAtomPositions(RDGeom::Point3DConstPtrVect &pts, const Conformer &conf,
                        const ROMol &, const std::vector<unsigned int> &match) {
  PRECONDITION(pts.size() == match.size(), "bad pts size");
  for (unsigned int i = 0; i < match.size(); i++) {
    pts[i] = &conf.getAtomPos(match[i]);
  }
}

bool _isConfFarFromRest(
    const ROMol &mol, const Conformer &conf, double threshold,
    const std::vector<std::vector<unsigned int>> &selfMatches) {
  // NOTE: it is tempting to use some triangle inequality to prune
  // conformations here but some basic testing has shown very
  // little advantage and given that the time for pruning fades in
  // comparison to embedding - we will use a simple for loop below
  // over all conformation until we find a match
  RDGeom::Point3DConstPtrVect refPoints(selfMatches[0].size());
  RDGeom::Point3DConstPtrVect prbPoints(selfMatches[0].size());
  _fillAtomPositions(refPoints, conf, mol, selfMatches[0]);

  double ssrThres = selfMatches[0].size() * threshold * threshold;
  for (const auto &match : selfMatches) {
    for (auto confi = mol.beginConformers(); confi != mol.endConformers();
         ++confi) {
      _fillAtomPositions(prbPoints, *(*confi), mol, match);
      RDGeom::Transform3D trans;
      auto ssr =
          RDNumeric::Alignments::AlignPoints(refPoints, prbPoints, trans);
      if (ssr < ssrThres) {
        return false;
      }
         }
  }
  return true;
}

}// namespace DGeomHelpers


// NOLINTEND

} // namespace RDKit

namespace nvmolkit {
//! Adds existing conformers to the molecule, performing RMS based pruning if that option is set in parameters.
void addConformersToMoleculeWithPruning(RDKit::ROMol& mol, std::vector<std::unique_ptr<RDKit::Conformer>>& confs,
  const RDKit::DGeomHelpers::EmbedParameters& params) {

  std::vector<std::vector<unsigned int>> selfMatches;
  if (params.pruneRmsThresh > 0.0) {
    selfMatches = RDKit::DGeomHelpers::getMolSelfMatches(mol, params);
  }

  for (auto& conf: confs) {
     // check if we are pruning away conformations and
      // a close-by conformation has already been chosen :
      if (params.pruneRmsThresh <= 0.0 ||
          RDKit::DGeomHelpers::_isConfFarFromRest(mol, *conf, params.pruneRmsThresh, selfMatches)) {
        mol.addConformer(conf.release(), true);
      }
  }
}
} // namespace nvmolkit

