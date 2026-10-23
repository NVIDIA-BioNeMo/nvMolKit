// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/descriptors3d_mol.h"

#include <GraphMol/Conformer.h>
#include <GraphMol/Descriptors/MolData3Ddescriptors.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/SanitException.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/conformer/conformer_coord_upload.h"
#include "src/utils/openmp_helpers.h"

namespace nvMolKit {

namespace {

struct DeviceDescriptorInputs {
  AsyncDeviceVector<double>  momentWeights;
  AsyncDeviceVector<double>  atomPropertyWeights;
  AsyncDeviceVector<double>  iStateDragWeights;
  AsyncDeviceVector<double>  covalentRadiusWeights;
  AsyncDeviceVector<int32_t> bondNeighborStarts;
  AsyncDeviceVector<int32_t> bondNeighbors;
  AsyncDeviceVector<uint8_t> usrcatAtomClasses;
  AsyncDeviceVector<uint8_t> heavyAtomFlags;
  AsyncDeviceVector<double>  vdwRadii;
  AsyncDeviceVector<uint8_t> dclvPolarClasses;
  AsyncDeviceVector<int32_t> defaultConformerRows;
  AsyncDeviceVector<int8_t>  conformerIs3D;
  AsyncDeviceVector<double>  eemElectronegativity;
  AsyncDeviceVector<double>  eemHardness;
  AsyncDeviceVector<double>  moleculeFormalCharges;
  AsyncDeviceVector<int32_t> moleculeAtomStarts;
};

//! Per-molecule inputs the requested properties read; see Property3DDeviceInputs.
struct DescriptorInputNeeds {
  bool momentWeights         = false;
  bool atomPropertyWeights   = false;
  bool iStateDragWeights     = false;
  bool covalentRadiusWeights = false;
  bool bondAdjacency         = false;
  bool usrcatAtomClasses     = false;
  bool heavyAtomFlags        = false;
  bool dclvAtomInputs        = false;
  bool defaultConformerRows  = false;
  bool conformerFlags        = false;
  bool eemParameters         = false;
};

//! RDKit's EEM parameters, indexed by atomic number up to bromine, for atoms whose highest Kekulé bond order
//! is 1, 2 and 3 (Code/GraphMol/Descriptors/EEM.cpp, from the NEEMP B3LYP/6-311G NPA set). A 0 entry marks an
//! element and type the set has no parameters for.
constexpr int kNumEemElements = 36;
// clang-format off
constexpr std::array<std::array<double, kNumEemElements>, 3> kEemElectronegativity = {{
  {0.0, 2.5473, 0.0, 0.0, 0.0, 0.0, 2.7221, 2.9750, 3.1503, 2.9976, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   2.6511, 2.7026, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.6263},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.7667, 2.8895, 3.0486, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2933,
   2.6471, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.6944, 3.0240, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
}};
constexpr std::array<std::array<double, kNumEemElements>, 3> kEemHardness = {{
  {0.0, 1.1641, 0.0, 0.0, 0.0, 0.0, 0.6403, 0.9083, 1.0577, 0.9983, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.4897, 1.1537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1105},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6513, 0.6647, 0.8410, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5759,
   0.4512, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6776, 1.4240, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
}};
// clang-format on

/**
 * @brief Writes each atom's EEM electronegativity and hardness, and returns the molecule's formal charge,
 *        as RDKit's EEM types them: on a Kekulé copy, by element and highest bond order (aromatic counted
 *        as 2).
 *
 * Atoms the parameter set does not cover get a NaN electronegativity so the kernel reports NaN, as do molecules
 * that fail to kekulize (which RDKit rejects). For elements beyond bromine RDKit reads past its tables, and for
 * bond orders above 3 it leaves the right-hand side uninitialized. Unlike RDKit, which silently solves with zero
 * parameters for uncovered atoms within its tables (Si, single-bonded P, B, ...) and returns charges of -10 or
 * beyond, those atoms are NaN too.
 */
double writeEemParameters(const RDKit::ROMol& mol, double* electronegativity, double* hardness) {
  const double missing = std::numeric_limits<double>::quiet_NaN();
  RDKit::RWMol kekule(mol);
  try {
    RDKit::MolOps::Kekulize(kekule, true);
  } catch (const RDKit::MolSanitizeException&) {
    std::fill(electronegativity, electronegativity + mol.getNumAtoms(), missing);
    std::fill(hardness, hardness + mol.getNumAtoms(), missing);
    return 0.0;
  }
  for (const auto* atom : kekule.atoms()) {
    unsigned int type = 1;
    for (const auto* bond : kekule.atomBonds(atom)) {
      double order = bond->getBondTypeAsDouble();
      if (order == 1.5) {
        order = 2.0;
      }
      type = std::max(type, static_cast<unsigned int>(order));
    }
    const unsigned int element = atom->getAtomicNum();
    const unsigned int idx     = atom->getIdx();
    if (element >= kNumEemElements || type > 3 || kEemHardness[type - 1][element] == 0.0) {
      electronegativity[idx] = missing;
      hardness[idx]          = missing;
    } else {
      electronegativity[idx] = kEemElectronegativity[type - 1][element];
      hardness[idx]          = kEemHardness[type - 1][element];
    }
  }
  return RDKit::MolOps::getFormalCharge(kekule);
}

//! Writes molecule @p mol's bond adjacency into the global CSR arrays: atom i's neighbors go to
//! @p neighbors from `neighborStarts[i]`, as atom indices within the molecule. @p neighborStarts points at the
//! molecule's first atom; its entries are global offsets beginning at @p neighborOffset.
void writeBondAdjacency(const RDKit::ROMol& mol,
                        const int32_t       neighborOffset,
                        int32_t*            neighborStarts,
                        int32_t*            neighbors) {
  const int numAtoms = static_cast<int>(mol.getNumAtoms());
  int32_t   offset   = neighborOffset;
  for (int atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
    neighborStarts[atomIdx] = offset;
    offset += static_cast<int32_t>(mol.getAtomWithIdx(atomIdx)->getDegree());
  }
  for (int atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
    int32_t cursor = neighborStarts[atomIdx];
    for (const auto* neighbor : mol.atomNeighbors(mol.getAtomWithIdx(atomIdx))) {
      neighbors[cursor++] = static_cast<int32_t>(neighbor->getIdx());
    }
  }
}

//! Property3DDeviceInputs::dclvPolarClasses bits of @p atom, from RDKit's includeAsPolar (DCLV.cpp).
uint8_t dclvPolarClasses(const RDKit::ROMol& mol, const RDKit::Atom& atom) {
  auto heavyClass = [](const int atomicNum) -> uint8_t {
    switch (atomicNum) {
      case 7:
      case 8:
        return kDclvPolarNitrogenOxygen;
      case 15:
      case 16:
        return kDclvPolarSulfurPhosphorus;
      default:
        return 0;
    }
  };
  if (atom.getAtomicNum() != 1) {
    return heavyClass(atom.getAtomicNum());
  }
  // A hydrogen is polar when a neighbor is. RDKit recurses into hydrogen neighbors, which never terminates for
  // bonded hydrogens (H2); such hydrogens are not polar here.
  uint8_t classes = 0;
  for (const auto* neighbor : mol.atomNeighbors(&atom)) {
    const uint8_t neighborClass = heavyClass(neighbor->getAtomicNum());
    if (neighborClass == kDclvPolarNitrogenOxygen) {
      classes |= kDclvPolarHydrogenOnNO;
    } else if (neighborClass == kDclvPolarSulfurPhosphorus) {
      classes |= kDclvPolarHydrogenOnSulfurPhos;
    }
  }
  return classes;
}

//! RDKit's USRCAT atom classes (hydrophobic, aromatic, acceptor, donor), from
//! Code/GraphMol/Descriptors/USRDescriptor.cpp.
const std::array<std::unique_ptr<RDKit::RWMol>, 4>& usrcatClassPatterns() {
  static const std::array<std::unique_ptr<RDKit::RWMol>, 4> patterns = {
    std::unique_ptr<RDKit::RWMol>(RDKit::SmartsToMol("[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]")),
    std::unique_ptr<RDKit::RWMol>(RDKit::SmartsToMol("[a]")),
    std::unique_ptr<RDKit::RWMol>(RDKit::SmartsToMol(
      "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N&v3;H1,H2]-[!$(*=[O,N,P,S])]),"
      "$([N;v3;H0]),$([n,o,s;+0]),F]")),
    std::unique_ptr<RDKit::RWMol>(RDKit::SmartsToMol("[N!H0v3,N!H0+v4,OH+0,SH+0,nH+0]")),
  };
  return patterns;
}

//! This thread's copies of the USRCAT class patterns: matching the shared patterns from several threads
//! would contend on their recursive-query state, and copying them per molecule is costly.
const std::array<std::unique_ptr<RDKit::ROMol>, 4>& threadUsrcatClassPatterns() {
  thread_local const std::array<std::unique_ptr<RDKit::ROMol>, 4> copies = [] {
    std::array<std::unique_ptr<RDKit::ROMol>, 4> result;
    const auto&                                  patterns = usrcatClassPatterns();
    for (size_t classIdx = 0; classIdx < patterns.size(); ++classIdx) {
      result[classIdx] = std::make_unique<RDKit::ROMol>(*patterns[classIdx], true);
    }
    return result;
  }();
  return copies;
}

DeviceDescriptorInputs uploadDescriptorInputs(const std::vector<const RDKit::ROMol*>& mols,
                                              const DescriptorInputNeeds&             needs,
                                              const int                               numThreads,
                                              cudaStream_t                            stream) {
  const int            numMols = static_cast<int>(mols.size());
  std::vector<int32_t> atomStarts(numMols + 1, 0);
  int64_t              totalAtoms = 0;
  for (int molIdx = 0; molIdx < numMols; ++molIdx) {
    atomStarts[molIdx] = static_cast<int32_t>(totalAtoms);
    totalAtoms += mols[molIdx]->getNumAtoms();
    if (totalAtoms > std::numeric_limits<int32_t>::max()) {
      throw std::overflow_error("Total molecule atom count exceeds int32 range");
    }
  }
  atomStarts[numMols] = static_cast<int32_t>(totalAtoms);

  // Offset of each molecule's first neighbor entry in the bond adjacency (two entries per bond).
  std::vector<int32_t> neighborOffsets;
  if (needs.bondAdjacency) {
    neighborOffsets.assign(numMols + 1, 0);
    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      const int64_t next = int64_t{neighborOffsets[molIdx]} + 2 * int64_t{mols[molIdx]->getNumBonds()};
      if (next > std::numeric_limits<int32_t>::max()) {
        throw std::overflow_error("Total bond count exceeds int32 range");
      }
      neighborOffsets[molIdx + 1] = static_cast<int32_t>(next);
    }
  }
  std::vector<double>  weights(needs.momentWeights ? static_cast<size_t>(totalAtoms) : 0);
  std::vector<double>  atomPropertyWeights(needs.atomPropertyWeights ? static_cast<size_t>(totalAtoms) * 6 : 0);
  std::vector<double>  iStateDragWeights(needs.iStateDragWeights ? static_cast<size_t>(totalAtoms) : 0);
  std::vector<double>  covalentRadiusWeights(needs.covalentRadiusWeights ? static_cast<size_t>(totalAtoms) : 0);
  std::vector<int32_t> bondNeighborStarts(needs.bondAdjacency ? static_cast<size_t>(totalAtoms) + 1 : 0);
  // At least one entry, so a batch without bonds still uploads a non-null (unread) neighbor buffer.
  std::vector<int32_t> bondNeighbors(needs.bondAdjacency ? std::max<size_t>(neighborOffsets.back(), 1) : 0);
  if (needs.bondAdjacency) {
    bondNeighborStarts[totalAtoms] = neighborOffsets.back();
  }
  std::vector<uint8_t> usrcatAtomClasses(needs.usrcatAtomClasses ? static_cast<size_t>(totalAtoms) : 0);
  std::vector<uint8_t> heavyAtomFlags(needs.heavyAtomFlags ? static_cast<size_t>(totalAtoms) : 0);
  std::vector<double>  vdwRadii(needs.dclvAtomInputs ? static_cast<size_t>(totalAtoms) : 0);
  std::vector<uint8_t> polarClasses(needs.dclvAtomInputs ? static_cast<size_t>(totalAtoms) : 0);
  // Coordinate rows follow each molecule's conformers in order, so a molecule's first row is its default
  // conformer (RDKit's getConformer(-1)).
  std::vector<int32_t> defaultConformerRows;
  if (needs.defaultConformerRows) {
    defaultConformerRows.resize(numMols);
    int32_t row = 0;
    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      const int32_t numConformers  = static_cast<int32_t>(mols[molIdx]->getNumConformers());
      defaultConformerRows[molIdx] = numConformers > 0 ? row : -1;
      row += numConformers;
    }
  }
  std::vector<double> eemElectronegativity(needs.eemParameters ? static_cast<size_t>(totalAtoms) : 0);
  std::vector<double> eemHardness(needs.eemParameters ? static_cast<size_t>(totalAtoms) : 0);
  std::vector<double> moleculeFormalCharges(needs.eemParameters ? numMols : 0);
  std::vector<int8_t> conformerIs3D;
  if (needs.conformerFlags) {
    for (const RDKit::ROMol* mol : mols) {
      for (auto conformer = mol->beginConformers(); conformer != mol->endConformers(); ++conformer) {
        conformerIs3D.push_back((*conformer)->is3D());
      }
    }
  }
  detail::OpenMPExceptionRegistry exceptionRegistry;
  if (needs.momentWeights) {
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) default(none) \
  shared(numMols, mols, atomStarts, weights, exceptionRegistry)
    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      try {
        size_t atomOffset = static_cast<size_t>(atomStarts[molIdx]);
        for (const auto* atom : mols[molIdx]->atoms()) {
          weights[atomOffset++] = atom->getMass();
        }
      } catch (...) {
        exceptionRegistry.store(std::current_exception());
      }
    }
    exceptionRegistry.rethrow();
  }
  if (needs.atomPropertyWeights || needs.iStateDragWeights || needs.covalentRadiusWeights || needs.bondAdjacency ||
      needs.usrcatAtomClasses || needs.heavyAtomFlags || needs.dclvAtomInputs || needs.eemParameters) {
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) default(none) shared(numMols,                 \
                                                                                          mols,                  \
                                                                                          atomStarts,            \
                                                                                          totalAtoms,            \
                                                                                          needs,                 \
                                                                                          neighborOffsets,       \
                                                                                          atomPropertyWeights,   \
                                                                                          iStateDragWeights,     \
                                                                                          covalentRadiusWeights, \
                                                                                          bondNeighborStarts,    \
                                                                                          bondNeighbors,         \
                                                                                          usrcatAtomClasses,     \
                                                                                          heavyAtomFlags,        \
                                                                                          vdwRadii,              \
                                                                                          polarClasses,          \
                                                                                          eemElectronegativity,  \
                                                                                          eemHardness,           \
                                                                                          moleculeFormalCharges, \
                                                                                          exceptionRegistry)
    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      try {
        const RDKit::ROMol&  mol = *mols[molIdx];
        MolData3Ddescriptors descriptorData;
        const size_t         atomStart = static_cast<size_t>(atomStarts[molIdx]);
        if (needs.atomPropertyWeights) {
          const std::array<std::vector<double>, 6> moleculeWeights = {
            descriptorData.GetRelativeMW(mol),
            descriptorData.GetRelativeVdW(mol),
            descriptorData.GetRelativeENeg(mol),
            descriptorData.GetRelativePol(mol),
            descriptorData.GetRelativeIonPol(mol),
            descriptorData.GetIState(mol),
          };
          for (size_t channel = 0; channel < moleculeWeights.size(); ++channel) {
            std::copy(moleculeWeights[channel].begin(),
                      moleculeWeights[channel].end(),
                      atomPropertyWeights.begin() + static_cast<size_t>(totalAtoms) * channel + atomStart);
          }
        }
        if (needs.iStateDragWeights) {
          const std::vector<double> iStateDrag = descriptorData.GetIStateDrag(mol);
          std::copy(iStateDrag.begin(), iStateDrag.end(), iStateDragWeights.begin() + atomStart);
        }
        if (needs.covalentRadiusWeights) {
          const std::vector<double> radii = descriptorData.GetRelativeRcov(mol);
          std::copy(radii.begin(), radii.end(), covalentRadiusWeights.begin() + atomStart);
        }
        if (needs.heavyAtomFlags) {
          for (const auto* atom : mol.atoms()) {
            heavyAtomFlags[atomStart + atom->getIdx()] = atom->getAtomicNum() > 1 ? 1 : 0;
          }
        }
        if (needs.dclvAtomInputs) {
          const RDKit::PeriodicTable* table = RDKit::PeriodicTable::getTable();
          for (const auto* atom : mol.atoms()) {
            vdwRadii[atomStart + atom->getIdx()]     = table->getRvdw(atom->getAtomicNum());
            polarClasses[atomStart + atom->getIdx()] = dclvPolarClasses(mol, *atom);
          }
        }
        if (needs.bondAdjacency) {
          writeBondAdjacency(mol, neighborOffsets[molIdx], bondNeighborStarts.data() + atomStart, bondNeighbors.data());
        }
        if (needs.eemParameters) {
          moleculeFormalCharges[molIdx] =
            writeEemParameters(mol, eemElectronegativity.data() + atomStart, eemHardness.data() + atomStart);
        }
        if (needs.usrcatAtomClasses) {
          const auto& patterns = threadUsrcatClassPatterns();
          for (size_t classIdx = 0; classIdx < patterns.size(); ++classIdx) {
            // Same call as RDKit's USRCAT, including its default maxMatches.
            std::vector<RDKit::MatchVectType> matches;
            RDKit::SubstructMatch(mol, *patterns[classIdx], matches);
            for (const auto& match : matches) {
              for (const auto& [queryIdx, atomIdx] : match) {
                usrcatAtomClasses[atomStart + atomIdx] |= static_cast<uint8_t>(1u << classIdx);
              }
            }
          }
        }
      } catch (...) {
        exceptionRegistry.store(std::current_exception());
      }
    }
    exceptionRegistry.rethrow();
  }

  DeviceDescriptorInputs result{AsyncDeviceVector<double>(weights.size(), stream),
                                AsyncDeviceVector<double>(atomPropertyWeights.size(), stream),
                                AsyncDeviceVector<double>(iStateDragWeights.size(), stream),
                                AsyncDeviceVector<double>(covalentRadiusWeights.size(), stream),
                                AsyncDeviceVector<int32_t>(bondNeighborStarts.size(), stream),
                                AsyncDeviceVector<int32_t>(bondNeighbors.size(), stream),
                                AsyncDeviceVector<uint8_t>(usrcatAtomClasses.size(), stream),
                                AsyncDeviceVector<uint8_t>(heavyAtomFlags.size(), stream),
                                AsyncDeviceVector<double>(vdwRadii.size(), stream),
                                AsyncDeviceVector<uint8_t>(polarClasses.size(), stream),
                                AsyncDeviceVector<int32_t>(defaultConformerRows.size(), stream),
                                AsyncDeviceVector<int8_t>(conformerIs3D.size(), stream),
                                AsyncDeviceVector<double>(eemElectronegativity.size(), stream),
                                AsyncDeviceVector<double>(eemHardness.size(), stream),
                                AsyncDeviceVector<double>(moleculeFormalCharges.size(), stream),
                                AsyncDeviceVector<int32_t>(atomStarts.size(), stream)};
  if (!weights.empty()) {
    result.momentWeights.copyFromHost(weights);
  }
  if (!atomPropertyWeights.empty()) {
    result.atomPropertyWeights.copyFromHost(atomPropertyWeights);
  }
  if (!iStateDragWeights.empty()) {
    result.iStateDragWeights.copyFromHost(iStateDragWeights);
  }
  if (!covalentRadiusWeights.empty()) {
    result.covalentRadiusWeights.copyFromHost(covalentRadiusWeights);
  }
  if (!bondNeighborStarts.empty()) {
    result.bondNeighborStarts.copyFromHost(bondNeighborStarts);
  }
  if (!bondNeighbors.empty()) {
    result.bondNeighbors.copyFromHost(bondNeighbors);
  }
  if (!usrcatAtomClasses.empty()) {
    result.usrcatAtomClasses.copyFromHost(usrcatAtomClasses);
  }
  if (!heavyAtomFlags.empty()) {
    result.heavyAtomFlags.copyFromHost(heavyAtomFlags);
  }
  if (!vdwRadii.empty()) {
    result.vdwRadii.copyFromHost(vdwRadii);
    result.dclvPolarClasses.copyFromHost(polarClasses);
  }
  if (!defaultConformerRows.empty()) {
    result.defaultConformerRows.copyFromHost(defaultConformerRows);
  }
  if (!conformerIs3D.empty()) {
    result.conformerIs3D.copyFromHost(conformerIs3D);
  }
  if (!eemElectronegativity.empty()) {
    result.eemElectronegativity.copyFromHost(eemElectronegativity);
    result.eemHardness.copyFromHost(eemHardness);
  }
  if (!moleculeFormalCharges.empty()) {
    result.moleculeFormalCharges.copyFromHost(moleculeFormalCharges);
  }
  result.moleculeAtomStarts.copyFromHost(atomStarts);
  return result;
}

}  // namespace

template <typename Real>
Property3DBatchResult<Real> calc3DProperties(const std::vector<const RDKit::ROMol*>& mols,
                                             const std::vector<Property3D>&          properties,
                                             const Property3DOptions&                options,
                                             cudaStream_t                            stream,
                                             const DeviceCoordView*                  coordinates,
                                             const int                               preprocessingThreads) {
  const int numThreads = detail::resolveNumThreads(preprocessingThreads);
  for (size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    if (mols[molIdx] == nullptr) {
      throw std::invalid_argument("Null molecule at index " + std::to_string(molIdx));
    }
  }
  if (coordinates != nullptr && coordinates->nMols != static_cast<int>(mols.size())) {
    throw std::invalid_argument("Device coordinates describe " + std::to_string(coordinates->nMols) +
                                " molecules, but " + std::to_string(mols.size()) + " molecules were provided");
  }

  // Uploaded buffers outlive the kernel launch below; their stream-ordered frees run after it.
  DeviceCoordResult uploaded;
  DeviceCoordView   view;
  if (coordinates != nullptr) {
    view = *coordinates;
  } else {
    uploaded = uploadConformerCoordinates(mols, stream, numThreads);
    view     = makeDeviceCoordView(uploaded);
  }
  DescriptorInputNeeds needs;
  for (const Property3D property : properties) {
    const Property3DFamily family = property3DFamily(property);
    needs.momentWeights |=
      family == Property3DFamily::Moments && options.moments.useAtomicMasses && property != Property3D::SpherocityIndex;
    needs.atomPropertyWeights |=
      property == Property3D::WHIM || family == Property3DFamily::Pairwise || property == Property3D::GETAWAY;
    needs.iStateDragWeights |= property == Property3D::RDF;
    needs.covalentRadiusWeights |= property == Property3D::AUTOCORR3D;
    needs.bondAdjacency |= property == Property3D::AUTOCORR3D || property == Property3D::GETAWAY;
    needs.heavyAtomFlags |= property == Property3D::GETAWAY;
    needs.dclvAtomInputs |= family == Property3DFamily::Dclv;
    // Device coordinate rows are not tied to molecule conformers, so GETAWAY then uses each row's own.
    needs.defaultConformerRows |= property == Property3D::GETAWAY && coordinates == nullptr;
    needs.usrcatAtomClasses |= property == Property3D::USRCAT;
    // Device coordinate rows carry no is3D flag and are treated as three-dimensional.
    needs.conformerFlags |= (property == Property3D::PBF || property == Property3D::GETAWAY) && coordinates == nullptr;
    needs.eemParameters |= property == Property3D::EEMcharges;
  }
  const DeviceDescriptorInputs uploadedInputs = uploadDescriptorInputs(mols, needs, numThreads, stream);

  Property3DDeviceInputs inputs;
  inputs.moleculeAtomStarts    = uploadedInputs.moleculeAtomStarts.data();
  inputs.momentWeights         = uploadedInputs.momentWeights.data();
  inputs.atomPropertyWeights   = uploadedInputs.atomPropertyWeights.data();
  inputs.iStateDragWeights     = uploadedInputs.iStateDragWeights.data();
  inputs.covalentRadiusWeights = uploadedInputs.covalentRadiusWeights.data();
  inputs.bondNeighborStarts    = uploadedInputs.bondNeighborStarts.data();
  inputs.bondNeighbors         = uploadedInputs.bondNeighbors.data();
  inputs.usrcatAtomClasses     = uploadedInputs.usrcatAtomClasses.data();
  inputs.heavyAtomFlags        = uploadedInputs.heavyAtomFlags.data();
  inputs.vdwRadii              = uploadedInputs.vdwRadii.data();
  inputs.dclvPolarClasses      = uploadedInputs.dclvPolarClasses.data();
  inputs.defaultConformerRows  = uploadedInputs.defaultConformerRows.data();
  inputs.conformerIs3D         = uploadedInputs.conformerIs3D.data();
  inputs.eemElectronegativity  = uploadedInputs.eemElectronegativity.data();
  inputs.eemHardness           = uploadedInputs.eemHardness.data();
  inputs.moleculeFormalCharges = uploadedInputs.moleculeFormalCharges.data();
  for (const RDKit::ROMol* mol : mols) {
    const int64_t numAtoms  = mol->getNumAtoms();
    inputs.maxMoleculeAtoms = std::max(inputs.maxMoleculeAtoms, static_cast<int32_t>(numAtoms));
    inputs.moleculeAtomPairs += numAtoms * (numAtoms - 1) / 2;
  }

  Property3DBatchResult<Real> result;
  result.properties  = calc3DPropertiesGpu<Real>(view, inputs, properties, options, stream);
  result.molIndices  = std::move(uploaded.molIndices);
  result.confIndices = std::move(uploaded.confIndices);
  result.atomStarts  = std::move(uploaded.atomStarts);
  return result;
}

template Property3DBatchResult<float>  calc3DProperties<float>(const std::vector<const RDKit::ROMol*>&,
                                                              const std::vector<Property3D>&,
                                                              const Property3DOptions&,
                                                              cudaStream_t,
                                                              const DeviceCoordView*,
                                                              int);
template Property3DBatchResult<double> calc3DProperties<double>(const std::vector<const RDKit::ROMol*>&,
                                                                const std::vector<Property3D>&,
                                                                const Property3DOptions&,
                                                                cudaStream_t,
                                                                const DeviceCoordView*,
                                                                int);

}  // namespace nvMolKit
