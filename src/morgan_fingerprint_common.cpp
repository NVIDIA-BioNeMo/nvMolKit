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

#include "src/morgan_fingerprint_common.h"

#include <GraphMol/PeriodicTable.h>

#include <RDGeneral/hash/hash.hpp>

namespace nvMolKit {

constexpr int kNumAtomInvariantMaxFeatures = 6;

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

  const gboost::hash<std::vector<uint32_t>> vectHasher;

  const size_t                molBondStride = maxAtoms * kMaxBondsPerAtom;
  const size_t                molAtomStride = maxAtoms;
  std::vector<std::uint32_t>  atomInvariantComponents(kNumAtomInvariantMaxFeatures);
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

    // bondInvariantsOut will be filled during atom neighbor iteration below to avoid getBondWithIdx calls

    const size_t           numAtoms = std::min<size_t>(mol.getNumAtoms(), maxAtoms);
    const RDKit::RingInfo* ringInfo = mol.getRingInfo();
    for (size_t atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
      const RDKit::Atom* tAtom = mol.getAtomWithIdx(atomIdx);

      const int deltaMass = static_cast<int>(tAtom->getMass() - periodicTable->getAtomicWeight(tAtom->getAtomicNum()));

      atomInvariantComponents.resize(kNumAtomInvariantMaxFeatures - 1);
      const bool               isInRing = ringInfo->numAtomRings(tAtom->getIdx()) > 0;
      // Compute degree and neighbor Hs while recording bond indices; also fill bondInvariantsOut.
      RDKit::ROMol::OEDGE_ITER beg;
      RDKit::ROMol::OEDGE_ITER end;
      boost::tie(beg, end)     = mol.getAtomBonds(tAtom);
      size_t       startIdx    = molIdx * molBondStride + kMaxBondsPerAtom * atomIdx;
      unsigned int degreeCount = 0;
      unsigned int neighborHs  = 0;
      while (beg != end) {
        const RDKit::Bond* bond           = mol[*beg];
        const auto         bondIdxLocal   = static_cast<std::uint32_t>(bond->getIdx());
        bondAtomIndicesOut[startIdx]      = static_cast<int16_t>(bondIdxLocal);
        const unsigned otherIdx           = bond->getOtherAtomIdx(atomIdx);
        bondOtherAtomIndicesOut[startIdx] = static_cast<int16_t>(otherIdx);
        if (bondIdxLocal < maxAtoms) {
          bondInvariantsOut[molAtomStride * molIdx + bondIdxLocal] = static_cast<uint32_t>(bond->getBondType());
        }
        const RDKit::Atom* otherAtom = bond->getOtherAtom(tAtom);
        if (otherAtom->getAtomicNum() == 1) {
          ++neighborHs;
        }
        ++degreeCount;
        ++startIdx;
        ++beg;
      }

      const auto explicitImplicitHs  = static_cast<unsigned int>(tAtom->getNumExplicitHs() + tAtom->getNumImplicitHs());
      const unsigned int totalDegree = explicitImplicitHs + degreeCount;
      const unsigned int totalHsIncludingNeighbors = explicitImplicitHs + neighborHs;

      atomInvariantComponents[0] = tAtom->getAtomicNum();
      atomInvariantComponents[1] = totalDegree;
      atomInvariantComponents[2] = totalHsIncludingNeighbors;
      atomInvariantComponents[3] = tAtom->getFormalCharge();
      atomInvariantComponents[4] = deltaMass;
      if (isInRing) {
        atomInvariantComponents.push_back(1);
      }
      atomInvariantsOut[molAtomStride * molIdx + atomIdx] = vectHasher(atomInvariantComponents);
    }
  }
}

}  // namespace nvMolKit
