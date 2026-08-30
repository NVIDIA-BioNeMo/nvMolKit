#ifndef NVMOLKIT_CONFORMER_PRUNING_H
#define NVMOLKIT_CONFORMER_PRUNING_H

#include <memory>
#include <vector>

namespace RDKit {
class ROMol;
namespace DGeomHelpers {
struct EmbedParameters;
/**
 * @brief Build the atom mappings used by RDKit for conformer pruning.
 *
 * @param mol Molecule whose conformers will be compared.
 * @param params RDKit settings that control heavy-atom and symmetry comparisons.
 * @return Atom orders to test during each RMSD comparison.
 */
std::vector<std::vector<unsigned int>> getMolSelfMatches(
    const ROMol &mol, const EmbedParameters &params);
}
class Conformer;
} // namespace RDKit

namespace nvmolkit {

/**
 * @brief Add conformers to a molecule using RDKit's ordered RMSD pruning rule.
 *
 * @param mol Molecule that receives the retained conformers.
 * @param confs Candidate conformers in pruning order. Retained conformers move into @p mol.
 * @param params RDKit settings that control pruning.
 */
void addConformersToMoleculeWithPruning(RDKit::ROMol& mol, std::vector<std::unique_ptr<RDKit::Conformer>>& confs,
  const RDKit::DGeomHelpers::EmbedParameters& params);


}

#endif // NVMOLKIT_CONFORMER_PRUNING_H
