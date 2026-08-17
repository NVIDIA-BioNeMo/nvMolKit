#ifndef NVMOLKIT_CONFORMER_PRUNING_H
#define NVMOLKIT_CONFORMER_PRUNING_H

#include <memory>
#include <vector>

namespace RDKit {
class ROMol;
namespace DGeomHelpers {
struct EmbedParameters;
//! Atom orders used for symmetry-aware and heavy-atom conformer pruning.
std::vector<std::vector<unsigned int>> getMolSelfMatches(
    const ROMol &mol, const EmbedParameters &params);
}
class Conformer;
} // namespace RDKit

namespace nvmolkit {

//! Removes conformers from the vector that fail to meet RMS uniqueness thresholds.
//! Wraps RDKit code which takes a greedy approach, treating the first molecule as a reference
//! and then iteratively building up.
void addConformersToMoleculeWithPruning(RDKit::ROMol& mol, std::vector<std::unique_ptr<RDKit::Conformer>>& confs,
  const RDKit::DGeomHelpers::EmbedParameters& params);


}

#endif // NVMOLKIT_CONFORMER_PRUNING_H
