// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "src/morgan_fingerprint_common.h"

#include <GraphMol/PeriodicTable.h>

#include <array>
#include <RDGeneral/hash/hash.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace nvMolKit {

constexpr int kNumAtomInvariantMaxFeatures = 6;
constexpr int kMaxMorganGpuAtoms           = 128;

void MorganInvariantsGenerator::ComputeInvariants(const std::vector<const RDKit::ROMol*>& mols, size_t maxAtoms) {
  const size_t nMols = mols.size();
  invariantsInfo_.atomInvariants.resize(nMols * maxAtoms);
  invariantsInfo_.bondInvariants.resize(nMols * maxAtoms);
  invariantsInfo_.bondAtomIndices.clear();
  invariantsInfo_.bondOtherAtomIndices.clear();
  invariantsInfo_.bondAtomIndices.resize(nMols * maxAtoms * kMaxBondsPerAtom, -1);
  invariantsInfo_.bondOtherAtomIndices.resize(nMols * maxAtoms * kMaxBondsPerAtom, -1);

  ComputeInvariantsInto(mols,
                        maxAtoms,
                        invariantsInfo_.atomInvariants.data(),
                        invariantsInfo_.bondInvariants.data(),
                        invariantsInfo_.bondAtomIndices.data(),
                        invariantsInfo_.bondOtherAtomIndices.data());
}

void MorganInvariantsGenerator::ComputeInvariantsInto(const std::vector<const RDKit::ROMol*>& mols,
                                                      size_t                                  maxAtoms,
                                                      std::uint32_t*                          atomInvariantsOut,
                                                      std::uint32_t*                          bondInvariantsOut,
                                                      std::int16_t*                           bondAtomIndicesOut,
                                                      std::int16_t*                           bondOtherAtomIndicesOut) {
  const size_t nMols = mols.size();
  if (nMols == 0 || maxAtoms == 0) {
    return;
  }
  const size_t                                       molBondStride = maxAtoms * kMaxBondsPerAtom;
  const size_t                                       molAtomStride = maxAtoms;
  std::array<std::uint8_t, kMaxMorganGpuAtoms>       bondCountsStorage;
  std::array<std::uint8_t, kMaxMorganGpuAtoms>       neighboringHydrogenCountsStorage;
  std::array<const RDKit::Atom*, kMaxMorganGpuAtoms> atomsStorage;
  std::array<std::uint8_t, kMaxMorganGpuAtoms>       isHydrogenStorage;
  std::vector<std::uint8_t>                          countOverflowStorage;
  std::vector<const RDKit::Atom*>                    atomOverflowStorage;
  std::vector<std::uint8_t>                          hydrogenOverflowStorage;
  auto*                                              bondCounts = bondCountsStorage.data();
  auto* neighboringHydrogenCounts                               = neighboringHydrogenCountsStorage.data();
  auto* atoms                                                   = atomsStorage.data();
  auto* isHydrogen                                              = isHydrogenStorage.data();
  if (maxAtoms > kMaxMorganGpuAtoms) {
    countOverflowStorage.resize(2 * maxAtoms);
    atomOverflowStorage.resize(maxAtoms);
    hydrogenOverflowStorage.resize(maxAtoms);
    bondCounts                = countOverflowStorage.data();
    neighboringHydrogenCounts = countOverflowStorage.data() + maxAtoms;
    atoms                     = atomOverflowStorage.data();
    isHydrogen                = hydrogenOverflowStorage.data();
  }
  const RDKit::PeriodicTable* periodicTable = RDKit::PeriodicTable::getTable();

  // Initialize outputs
  std::fill(atomInvariantsOut, atomInvariantsOut + nMols * maxAtoms, 0U);
  std::fill(bondInvariantsOut, bondInvariantsOut + nMols * maxAtoms, 0U);
  std::fill(bondAtomIndicesOut, bondAtomIndicesOut + nMols * molBondStride, static_cast<int16_t>(-1));
  std::fill(bondOtherAtomIndicesOut, bondOtherAtomIndicesOut + nMols * molBondStride, static_cast<int16_t>(-1));

  for (size_t molIdx = 0; molIdx < nMols; ++molIdx) {
    const RDKit::ROMol& mol = *mols[molIdx];
    if (mol.getNumAtoms() >= maxAtoms || mol.getNumBonds() >= maxAtoms) {
      continue;
    }

    const size_t numAtoms = mol.getNumAtoms();
    std::fill_n(bondCounts, numAtoms, std::uint8_t{0});
    std::fill_n(neighboringHydrogenCounts, numAtoms, std::uint8_t{0});

    bool hasGraphHydrogens = false;
    for (const RDKit::Atom* atom : mol.atoms()) {
      const size_t atomIdx = atom->getIdx();
      atoms[atomIdx]       = atom;
      isHydrogen[atomIdx]  = atom->getAtomicNum() == 1;
      hasGraphHydrogens |= isHydrogen[atomIdx];
    }

    // Visit each bond once and populate both endpoint adjacency lists. The old
    // atom-centric traversal fetched every bond twice (once from each endpoint)
    // and recomputed its invariant twice.
    const size_t molBondOffset = molIdx * molBondStride;
    const auto   recordBond    = [&](const RDKit::Bond* bond) {
      const auto bondIdx   = static_cast<std::uint32_t>(bond->getIdx());
      const auto beginIdx  = static_cast<size_t>(bond->getBeginAtomIdx());
      const auto endIdx    = static_cast<size_t>(bond->getEndAtomIdx());
      const auto beginSlot = bondCounts[beginIdx]++;
      const auto endSlot   = bondCounts[endIdx]++;

      if (beginSlot >= kMaxBondsPerAtom || endSlot >= kMaxBondsPerAtom) {
        throw std::runtime_error("Morgan fingerprint supports at most " + std::to_string(kMaxBondsPerAtom) +
                                 " bonds per atom");
      }

      const size_t beginOffset                            = molBondOffset + beginIdx * kMaxBondsPerAtom + beginSlot;
      const size_t endOffset                              = molBondOffset + endIdx * kMaxBondsPerAtom + endSlot;
      bondAtomIndicesOut[beginOffset]                     = static_cast<std::int16_t>(bondIdx);
      bondOtherAtomIndicesOut[beginOffset]                = static_cast<std::int16_t>(endIdx);
      bondAtomIndicesOut[endOffset]                       = static_cast<std::int16_t>(bondIdx);
      bondOtherAtomIndicesOut[endOffset]                  = static_cast<std::int16_t>(beginIdx);
      bondInvariantsOut[molAtomStride * molIdx + bondIdx] = static_cast<std::uint32_t>(bond->getBondType());
    };

    if (hasGraphHydrogens) {
      for (const RDKit::Bond* bond : mol.bonds()) {
        recordBond(bond);
        const auto beginIdx = static_cast<size_t>(bond->getBeginAtomIdx());
        const auto endIdx   = static_cast<size_t>(bond->getEndAtomIdx());
        neighboringHydrogenCounts[beginIdx] += isHydrogen[endIdx];
        neighboringHydrogenCounts[endIdx] += isHydrogen[beginIdx];
      }
    } else {
      for (const RDKit::Bond* bond : mol.bonds()) {
        recordBond(bond);
      }
    }

    const RDKit::RingInfo* ringInfo = mol.getRingInfo();
    for (size_t atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
      const RDKit::Atom* tAtom = atoms[atomIdx];

      int deltaMass = 0;
      if (tAtom->getIsotope() != 0) {
        deltaMass = static_cast<int>(tAtom->getMass() - periodicTable->getAtomicWeight(tAtom->getAtomicNum()));
      }

      const auto explicitImplicitHs  = static_cast<unsigned int>(tAtom->getNumExplicitHs() + tAtom->getNumImplicitHs());
      const unsigned int totalDegree = explicitImplicitHs + bondCounts[atomIdx];
      const unsigned int totalHsIncludingNeighbors = explicitImplicitHs + neighboringHydrogenCounts[atomIdx];

      const bool isInRing = ringInfo->numAtomRings(tAtom->getIdx()) > 0;
      const std::array<std::uint32_t, kNumAtomInvariantMaxFeatures> atomInvariantComponents = {
        static_cast<std::uint32_t>(tAtom->getAtomicNum()),
        totalDegree,
        totalHsIncludingNeighbors,
        static_cast<std::uint32_t>(tAtom->getFormalCharge()),
        static_cast<std::uint32_t>(deltaMass),
        1U};
      const auto numComponents = isInRing ? atomInvariantComponents.size() : atomInvariantComponents.size() - 1;
      atomInvariantsOut[molAtomStride * molIdx + atomIdx] =
        gboost::hash_range(atomInvariantComponents.begin(), atomInvariantComponents.begin() + numComponents);
    }
  }
}

}  // namespace nvMolKit
