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

#include "src/substruct/molecules.h"

// RDConfig defines RDK_BUILD_THREADSAFE_SSS, which changes RecursiveStructureQuery layout.
// It must be visible before QueryOps is included through QueryAtom.h.
// clang-format off
#include <RDGeneral/RDConfig.h>
// clang-format on

#include <GraphMol/MolOps.h>
#include <GraphMol/QueryAtom.h>
#include <GraphMol/QueryOps.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmartsWrite.h>
#include <omp.h>
#include <RDGeneral/versions.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>

#include "src/substruct/packed_bonds.h"
#include "src/substruct/substruct_debug.h"
#include "src/substruct/substruct_types.h"
#include "src/utils/nvtx.h"
#include "src/utils/openmp_helpers.h"
#include "src/utils/rdkit_compat.h"

namespace nvMolKit {

namespace {

/**
 * @brief Populate non-bond-related atom properties into packed format.
 *
 * Extracts scalar atom properties from RDKit atom. Bond-related properties
 * (ring bond count, heteroatom neighbors, bond type counts) are populated
 * separately during the fused bond iteration.
 */
void populateAtomScalars(const RDKit::Atom* atom, AtomDataPacked& packed, const RDKit::RingInfo* ringInfo) {
  packed.setAtomicNum(atom->getAtomicNum());
  packed.setChiralTag(atom->getChiralTag());
  packed.setNumExplicitHs(atom->getTotalNumHs());
  packed.setExplicitValence(compat::getExplicitValence(atom));
  packed.setImplicitValence(compat::getImplicitValence(atom));
  packed.setTotalValence(atom->getTotalValence());
  packed.setFormalCharge(atom->getFormalCharge());
  packed.setHybridization(atom->getHybridization());
  packed.setIsAromatic(atom->getIsAromatic());
  packed.setNumRadicalElectrons(atom->getNumRadicalElectrons());

  const int idx      = atom->getIdx();
  const int numRings = ringInfo->numAtomRings(idx);
  if (numRings > AtomDataPacked::kMax4BitValue) {
    throw std::runtime_error("Atom ring count " + std::to_string(numRings) + " exceeds maximum storable value of " +
                             std::to_string(AtomDataPacked::kMax4BitValue));
  }
  packed.setNumRings(numRings);
  packed.setMinRingSize(ringInfo->minAtomRingSize(idx));
  packed.setIsInRing(numRings > 0);

  const unsigned int numImplicitHs = atom->getNumImplicitHs();
  if (numImplicitHs > AtomDataPacked::kMax4BitValue) {
    throw std::runtime_error("Implicit H count " + std::to_string(numImplicitHs) +
                             " exceeds maximum storable value of " + std::to_string(AtomDataPacked::kMax4BitValue));
  }
  packed.setNumImplicitHs(numImplicitHs);

  const unsigned int isotope = atom->getIsotope();
  if (isotope > 255) {
    throw std::runtime_error("Atom isotope " + std::to_string(isotope) + " exceeds maximum supported value of 255");
  }
  packed.setIsotope(static_cast<uint8_t>(isotope));

  packed.setDegree(atom->getDegree());
  packed.setTotalConnectivity(atom->getTotalDegree());
}

/**
 * @brief Increment bond type count based on RDKit bond type.
 */
void incrementBondTypeCount(BondTypeCounts& counts, int bondType) {
  switch (bondType) {
    case 1:
      ++counts.single;
      break;
    case 2:
      ++counts.double_;
      break;
    case 3:
      ++counts.triple;
      break;
    case 7:
    case 12:
      ++counts.aromatic;
      break;
    default:
      throw std::runtime_error("Unsupported bond type " + std::to_string(bondType) +
                               " in target molecule. Only single, double, triple, and aromatic bonds are supported.");
  }
}

/**
 * @brief Process all atoms and bonds for a target molecule in a single fused pass.
 */
void populateTargetMolecule(const RDKit::ROMol* mol, MoleculesHost& batch, const RDKit::RingInfo* ringInfo) {
  auto& atomDataPackedVec  = batch.atomDataPacked;
  auto& bondTypeCountsVec  = batch.bondTypeCounts;
  auto& targetAtomBondsVec = batch.targetAtomBonds;

  for (const RDKit::Atom* atom : mol->atoms()) {
    auto& packed     = atomDataPackedVec.emplace_back();
    auto& bondCounts = bondTypeCountsVec.emplace_back();
    auto& tab        = targetAtomBondsVec.emplace_back();

    populateAtomScalars(atom, packed, ringInfo);

    const unsigned int atomIdx            = atom->getIdx();
    int                ringBondCount      = 0;
    int                numHeteroNeighbors = 0;
    int                totalBonds         = 0;
    tab.degree                            = 0;

    auto [beg, bondEnd] = mol->getAtomBonds(atom);
    while (beg != bondEnd) {
      const auto*        bond        = (*mol)[*beg];
      const unsigned int bondIdx     = bond->getIdx();
      const int          bondType    = bond->getBondType();
      const int          otherAtomId = bond->getOtherAtomIdx(atomIdx);
      const bool         isInRing    = ringInfo->numBondRings(bondIdx) > 0;

      incrementBondTypeCount(bondCounts, bondType);

      ringBondCount += isInRing;

      const int neighborAtomicNum = mol->getAtomWithIdx(otherAtomId)->getAtomicNum();
      numHeteroNeighbors += (neighborAtomicNum != 6 && neighborAtomicNum != 1);

      if (tab.degree < kMaxBondsPerAtom) {
        tab.neighborIdx[tab.degree] = static_cast<uint8_t>(otherAtomId);
        tab.bondInfo[tab.degree]    = packTargetBondInfo(bondType, isInRing);
        ++tab.degree;
      }
      ++totalBonds;
      ++beg;
    }

    if (totalBonds > kMaxBondsPerAtom) {
      throw std::runtime_error("Atom has more than " + std::to_string(kMaxBondsPerAtom) + " bonds");
    }

    if (ringBondCount > AtomDataPacked::kMax4BitValue) {
      throw std::runtime_error("Ring bond count " + std::to_string(ringBondCount) +
                               " exceeds maximum storable value of " + std::to_string(AtomDataPacked::kMax4BitValue));
    }
    packed.setRingBondCount(ringBondCount);

    if (numHeteroNeighbors > AtomDataPacked::kMax4BitValue) {
      throw std::runtime_error("Heteroatom neighbor count " + std::to_string(numHeteroNeighbors) +
                               " exceeds maximum storable value of " + std::to_string(AtomDataPacked::kMax4BitValue));
    }
    packed.setNumHeteroatomNeighbors(numHeteroNeighbors);
  }
}

/**
 * @brief Get the effective bond type for a query bond.
 *
 * For SMARTS queries, implicit bonds (no explicit bond symbol) are represented
 * with SingleOrAromaticBond query which should match any bond type. This function
 * checks for such queries and returns 0 (any) instead of the nominal bond type.
 *
 * Negated bond queries like !- (NOT single) are treated as "any" since they can
 * match multiple bond types.
 */
int getQueryBondEffectiveType(const RDKit::Bond* bond) {
  int bondType = bond->getBondType();

  if (bond->hasQuery()) {
    const auto* query = bond->getQuery();
    if (query != nullptr) {
      const std::string desc      = query->getDescription();
      const bool        isNegated = query->getNegation();

      if (desc == "SingleOrAromaticBond" || desc == "DoubleOrAromaticBond" || desc == "BondNull") {
        return 0;  // Any bond (flexible match)
      }
      if (desc == "BondIsAromatic") {
        return isNegated ? 0 : 12;  // Negated aromatic = any; aromatic = 12
      }
      if (desc == "BondOrder") {
        // Negated bond order (e.g., !-) can match multiple types, treat as "any"
        if (isNegated) {
          return 0;
        }
      }
      // For BondAnd/BondOr queries, check for flexible bond patterns
      if (desc == "BondAnd" || desc == "BondOr") {
        bool hasSingle   = false;
        bool hasDouble   = false;
        bool hasAromatic = false;
        bool hasNegated  = false;
        for (auto it = query->beginChildren(); it != query->endChildren(); ++it) {
          const std::string childDesc = (*it)->getDescription();
          if ((*it)->getNegation()) {
            hasNegated = true;
          }
          if (childDesc == "SingleOrAromaticBond" || childDesc == "DoubleOrAromaticBond" || childDesc == "BondNull") {
            return 0;
          }
          if (childDesc == "BondOrder") {
            const auto* eqQuery   = static_cast<const RDKit::BOND_EQUALS_QUERY*>((*it).get());
            int         childType = eqQuery->getVal();
            if (childType == 1)
              hasSingle = true;
            else if (childType == 2)
              hasDouble = true;
            else if (childType == 7 || childType == 12)
              hasAromatic = true;
          } else if (childDesc == "BondIsAromatic") {
            hasAromatic = true;
          }
        }
        // If any child is negated, treat as flexible
        if (hasNegated) {
          return 0;
        }
        // If this is a "single or aromatic" or "double or aromatic" BondOr pattern
        if ((hasSingle && hasAromatic) || (hasDouble && hasAromatic)) {
          return 0;  // Treat as flexible bond for counting
        }
      }
    }
  }

  return bondType;
}

void populateQueryBondTypeCounts(const RDKit::ROMol* mol, const RDKit::Atom* atom, BondTypeCounts& counts) {
  auto [beg, bondEnd] = mol->getAtomBonds(atom);
  while (beg != bondEnd) {
    const auto* bond     = (*mol)[*beg];
    int         bondType = getQueryBondEffectiveType(bond);
    switch (bondType) {
      case 0:
        ++counts.any;
        break;  // UNSPECIFIED or any bond query
      case 1:
        ++counts.single;
        break;  // SINGLE
      case 2:
        ++counts.double_;
        break;  // DOUBLE
      case 3:
        ++counts.triple;
        break;  // TRIPLE
      case 7:   // ONEANDAHALF (aromatic)
      case 12:
        ++counts.aromatic;
        break;  // AROMATIC
      default:
        throw std::runtime_error("Unsupported bond type " + std::to_string(bondType) + " in query molecule.");
    }
    ++beg;
  }
}

}  // namespace

MoleculesHost::MoleculesHost() {
  batchAtomStarts.push_back(0);
}

void MoleculesHost::reserve(size_t numMols, size_t numAtoms) {
  batchAtomStarts.reserve(numMols + 1);

  atomDataPacked.reserve(numAtoms);
  bondTypeCounts.reserve(numAtoms);
  targetAtomBonds.reserve(numAtoms);
  queryAtomBonds.reserve(numAtoms);

  atomQueryMasks.reserve(numAtoms);
  atomQueryTrees.reserve(numAtoms);
  atomInstrStarts.reserve(numAtoms);
  atomLeafMaskStarts.reserve(numAtoms);
  recursivePatterns.reserve(numMols);
}

void MoleculesHost::clear() {
  batchAtomStarts.clear();
  batchAtomStarts.push_back(0);

  atomDataPacked.clear();
  bondTypeCounts.clear();
  targetAtomBonds.clear();
  queryAtomBonds.clear();

  atomQueryMasks.clear();
  atomQueryTrees.clear();
  queryInstructions.clear();
  queryLeafMasks.clear();
  queryLeafBondCounts.clear();
  atomInstrStarts.clear();
  atomLeafMaskStarts.clear();
  recursivePatterns.clear();
}

void MoleculesDevice::setStream(cudaStream_t stream) {
  stream_ = stream;
  batchAtomStarts_.setStream(stream);
  atomDataPacked_.setStream(stream);
  atomQueryMasks_.setStream(stream);
  bondTypeCounts_.setStream(stream);
  targetAtomBonds_.setStream(stream);
  queryAtomBonds_.setStream(stream);
  atomQueryTrees_.setStream(stream);
  queryInstructions_.setStream(stream);
  queryLeafMasks_.setStream(stream);
  queryLeafBondCounts_.setStream(stream);
  atomInstrStarts_.setStream(stream);
  atomLeafMaskStarts_.setStream(stream);
}

namespace {

template <typename T> void setFromVectorGrowOnly(AsyncDeviceVector<T>& dest, const std::vector<T>& src) {
  if (src.empty()) {
    return;
  }
  if (src.size() > dest.size()) {
    dest.resize(static_cast<size_t>(src.size() * 1.5));
  }
  dest.copyFromHost(src, src.size());
}

}  // namespace

void MoleculesDevice::copyFromHost(const MoleculesHost& host, cudaStream_t stream) {
  if (host.numMolecules() == 0) {
    throw std::invalid_argument("Cannot copy empty MoleculesHost to device");
  }

  setStream(stream);
  numMolecules_ = static_cast<int>(host.numMolecules());

  setFromVectorGrowOnly(batchAtomStarts_, host.batchAtomStarts);

  // Copy GPU-optimized packed data
  setFromVectorGrowOnly(atomDataPacked_, host.atomDataPacked);
  if (!host.atomQueryMasks.empty()) {
    setFromVectorGrowOnly(atomQueryMasks_, host.atomQueryMasks);
  }
  if (!host.bondTypeCounts.empty()) {
    setFromVectorGrowOnly(bondTypeCounts_, host.bondTypeCounts);
  }
  if (!host.targetAtomBonds.empty()) {
    setFromVectorGrowOnly(targetAtomBonds_, host.targetAtomBonds);
  }
  if (!host.queryAtomBonds.empty()) {
    setFromVectorGrowOnly(queryAtomBonds_, host.queryAtomBonds);
  }

  // Copy boolean expression tree data for compound queries
  if (!host.atomQueryTrees.empty()) {
    setFromVectorGrowOnly(atomQueryTrees_, host.atomQueryTrees);
    setFromVectorGrowOnly(queryInstructions_, host.queryInstructions);
    setFromVectorGrowOnly(queryLeafMasks_, host.queryLeafMasks);
    setFromVectorGrowOnly(queryLeafBondCounts_, host.queryLeafBondCounts);
    setFromVectorGrowOnly(atomInstrStarts_, host.atomInstrStarts);
    setFromVectorGrowOnly(atomLeafMaskStarts_, host.atomLeafMaskStarts);
  }
}

template <> TargetMoleculesDeviceView MoleculesDevice::view<MoleculeType::Target>() const {
  TargetMoleculesDeviceView v;
  v.batchAtomStarts = batchAtomStarts_.data();
  v.numMolecules    = numMolecules_;
  v.atomDataPacked  = atomDataPacked_.data();
  v.bondTypeCounts  = bondTypeCounts_.data();
  v.targetAtomBonds = targetAtomBonds_.data();
  return v;
}

template <> QueryMoleculesDeviceView MoleculesDevice::view<MoleculeType::Query>() const {
  QueryMoleculesDeviceView v;
  v.batchAtomStarts     = batchAtomStarts_.data();
  v.numMolecules        = numMolecules_;
  v.atomDataPacked      = atomDataPacked_.data();
  v.atomQueryMasks      = atomQueryMasks_.data();
  v.bondTypeCounts      = bondTypeCounts_.data();
  v.queryAtomBonds      = queryAtomBonds_.data();
  v.atomQueryTrees      = atomQueryTrees_.data();
  v.queryInstructions   = queryInstructions_.data();
  v.queryLeafMasks      = queryLeafMasks_.data();
  v.queryLeafBondCounts = queryLeafBondCounts_.data();
  v.atomInstrStarts     = atomInstrStarts_.data();
  v.atomLeafMaskStarts  = atomLeafMaskStarts_.data();
  return v;
}

AtomQuery atomQueryFromDescription(const std::string& description) {
  if (description == "AtomAtomicNum") {
    return AtomQueryAtomicNum;
  }
  if (description == "AtomHCount") {
    return AtomQueryNumExplicitHs;
  }
  if (description == "AtomExplicitValence") {
    return AtomQueryExplicitValence;
  }
  if (description == "AtomImplicitValence") {
    return AtomQueryImplicitValence;
  }
  if (description == "AtomFormalCharge") {
    return AtomQueryFormalCharge;
  }
  if (description == "AtomHybridization") {
    return AtomQueryHybridization;
  }
  if (description == "AtomIsAromatic") {
    return AtomQueryIsAromatic;
  }
  if (description == "AtomIsAliphatic") {
    return AtomQueryIsAliphatic;
  }
  if (description == "AtomMinRingSize") {
    return AtomQueryMinRingSize;
  }
  if (description == "AtomInNRings") {
    return AtomQueryNumRings;
  }
  if (description == "AtomNumRadicalElectrons") {
    return AtomQueryNumRadicalElectrons;
  }
  if (description == "AtomNull") {
    return AtomQueryNone;
  }

  // Degree and connectivity queries
  if (description == "AtomExplicitDegree") {
    return AtomQueryDegree;
  }
  if (description == "AtomTotalDegree") {
    return AtomQueryTotalConnectivity;
  }

  // Ring bond count [x] queries
  if (description == "AtomRingBondCount") {
    return AtomQueryRingBondCount;
  }
  if (description == "AtomTotalValence") {
    return AtomQueryTotalValence;
  }
  // Implicit H count [h] queries
  if (description == "AtomImplicitHCount") {
    return AtomQueryNumImplicitHs;
  }
  if (description == "AtomHasImplicitH") {
    return AtomQueryHasImplicitH;
  }
  // Heteroatom neighbor count queries
  if (description == "AtomNumHeteroatomNeighbors") {
    return AtomQueryNumHeteroNeighbors;
  }
  if (description == "AtomMass" || description == "AtomIsotope") {
    return AtomQueryIsotope;
  }
  if (description == "AtomHasRingBond") {
    throw std::runtime_error("SMARTS ring bond query (@) is not supported");
  }
  if (description == "AtomUnsaturated") {
    throw std::runtime_error("SMARTS unsaturation query is not supported");
  }
  if (description == "AtomChiralTag") {
    throw std::runtime_error("SMARTS chirality query (@/@@ ) is not supported");
  }
  if (description == "AtomInRing") {
    return AtomQueryIsInRing;  // [r] any ring query
  }

  throw std::runtime_error("Unsupported SMARTS atom query: " + description);
}

AtomQueryMask buildQueryMask(const AtomDataPacked& queryAtom, AtomQuery queryFlags) {
  AtomQueryMask m = {0, 0, 0, 0};

  // Impossible constraint (e.g., [C;a] aromatic aliphatic) - create unmatchable mask
  if (queryFlags & AtomQueryNeverMatches) {
    m.maskLo     = 0xFFULL;  // Check atomic number byte
    m.expectedLo = 0xFFULL;  // Require atomic number 255 (impossible, max is ~118)
    return m;
  }

  // Helper lambda to set mask and expected for a byte in the lower 64 bits
  auto setLoField = [&](int byteOffset, uint8_t value) {
    m.maskLo |= 0xFFULL << (byteOffset * 8);
    m.expectedLo |= static_cast<uint64_t>(value) << (byteOffset * 8);
  };

  // Helper lambda to set mask and expected for a byte in the upper 64 bits
  auto setHiField = [&](int byteOffset, uint8_t value) {
    m.maskHi |= 0xFFULL << (byteOffset * 8);
    m.expectedHi |= static_cast<uint64_t>(value) << (byteOffset * 8);
  };

  // Helper lambda to set mask and expected for a single bit in the upper 64 bits
  auto setHiBit = [&](int bitOffset, bool value) {
    m.maskHi |= 1ULL << bitOffset;
    if (value) {
      m.expectedHi |= 1ULL << bitOffset;
    }
  };

  // Helper lambda to set mask and expected for a partial byte in the upper 64 bits
  auto setHiPartialField = [&](int byteOffset, uint8_t mask, uint8_t value) {
    m.maskHi |= static_cast<uint64_t>(mask) << (byteOffset * 8);
    m.expectedHi |= static_cast<uint64_t>(value & mask) << (byteOffset * 8);
  };

  // Lower 64-bit fields
  if (queryFlags & AtomQueryAtomicNum) {
    setLoField(AtomDataPacked::kAtomicNumByte, queryAtom.atomicNum());
  }
  if (queryFlags & AtomQueryNumExplicitHs) {
    setLoField(AtomDataPacked::kNumExplicitHsByte, queryAtom.numExplicitHs());
  }
  if (queryFlags & AtomQueryExplicitValence) {
    setLoField(AtomDataPacked::kExplicitValenceByte, queryAtom.explicitValence());
  }
  if (queryFlags & AtomQueryImplicitValence) {
    setLoField(AtomDataPacked::kImplicitValenceByte, queryAtom.implicitValence());
  }
  if (queryFlags & AtomQueryFormalCharge) {
    setLoField(AtomDataPacked::kFormalChargeByte, static_cast<uint8_t>(queryAtom.formalCharge()));
  }
  if (queryFlags & AtomQueryChiralTag) {
    setLoField(AtomDataPacked::kChiralTagByte, queryAtom.chiralTag());
  }
  if (queryFlags & AtomQueryNumRadicalElectrons) {
    setLoField(AtomDataPacked::kNumRadicalElectronsByte, queryAtom.numRadicalElectrons());
  }
  if (queryFlags & AtomQueryHybridization) {
    setLoField(AtomDataPacked::kHybridizationByte, queryAtom.hybridization());
  }

  // Helper lambda to set mask and expected for a 4-bit field in the upper 64 bits
  auto setHi4BitField = [&](int byteOffset, int bitOffset, uint8_t value) {
    const uint64_t shift = byteOffset * 8 + bitOffset;
    m.maskHi |= static_cast<uint64_t>(AtomDataPacked::k4BitMask) << shift;
    m.expectedHi |= static_cast<uint64_t>(value & AtomDataPacked::k4BitMask) << shift;
  };

  // Upper 64-bit fields
  if (queryFlags & AtomQueryMinRingSize) {
    setHiField(AtomDataPacked::kMinRingSizeByte, queryAtom.minRingSize());
  }
  if (queryFlags & AtomQueryNumRings) {
    setHi4BitField(AtomDataPacked::kNumRingsRingBondsByte, AtomDataPacked::kNumRingsBits, queryAtom.numRings());
  }
  if (queryFlags & AtomQueryRingBondCount) {
    setHi4BitField(AtomDataPacked::kNumRingsRingBondsByte,
                   AtomDataPacked::kRingBondCountBits,
                   queryAtom.ringBondCount());
  }
  if (queryFlags & AtomQueryNumImplicitHs) {
    setHi4BitField(AtomDataPacked::kImplicitHsHeterosByte,
                   AtomDataPacked::kNumImplicitHsBits,
                   queryAtom.numImplicitHs());
  }
  if (queryFlags & AtomQueryNumHeteroNeighbors) {
    setHi4BitField(AtomDataPacked::kImplicitHsHeterosByte,
                   AtomDataPacked::kNumHeteroNeighborBits,
                   queryAtom.numHeteroatomNeighbors());
  }
  if (queryFlags & AtomQueryTotalValence) {
    setHiField(AtomDataPacked::kTotalValenceByte, queryAtom.totalValence());
  }

  // Special handling for aromaticity: uses single bit within degree byte
  if (queryFlags & AtomQueryIsAromatic) {
    setHiBit(AtomDataPacked::kDegreeByte * 8 + AtomDataPacked::kIsAromaticBit, true);
  }
  if (queryFlags & AtomQueryIsAliphatic) {
    setHiBit(AtomDataPacked::kDegreeByte * 8 + AtomDataPacked::kIsAromaticBit, false);
  }

  // [R] and [r] any-ring queries check isInRing bit within degree byte
  if (queryFlags & AtomQueryIsInRing) {
    setHiBit(AtomDataPacked::kDegreeByte * 8 + AtomDataPacked::kIsInRingBit, true);
  }

  // Isotope queries like [13C]
  if (queryFlags & AtomQueryIsotope) {
    setHiField(AtomDataPacked::kIsotopeByte, queryAtom.isotope());
  }

  // Degree queries like [D3] - uses 6 bits of degree byte
  if (queryFlags & AtomQueryDegree) {
    setHiPartialField(AtomDataPacked::kDegreeByte, AtomDataPacked::kDegreeMask, queryAtom.degree());
  }

  // Total connectivity queries like [X4]
  if (queryFlags & AtomQueryTotalConnectivity) {
    setHiField(AtomDataPacked::kTotalConnectivityByte, queryAtom.totalConnectivity());
  }

  return m;
}

namespace {

/**
 * @brief Builder for constructing boolean expression trees from RDKit queries.
 *
 * Recursively processes RDKit query atoms (supporting AND, OR, NOT) and generates
 * a sequence of BoolInstructions for evaluation on the GPU.
 */
struct QueryTreeBuilder {
  std::vector<AtomQueryMask>   leafMasks;
  std::vector<BondTypeCounts>  leafBondCounts;
  std::vector<BoolInstruction> instructions;
  int                          nextScratchIdx = 0;  // Use int to allow counting beyond uint8_t max

  /**
   * @brief Allocate the next scratch slot.
   * @return Scratch index where the result will be stored
   */
  uint8_t allocateScratch() { return static_cast<uint8_t>(nextScratchIdx++); }

  /**
   * @brief Process a leaf query (primitive comparison) and add to the tree.
   * @return Scratch index where the result will be stored
   */
  uint8_t addLeaf(const AtomDataPacked& packed, AtomQuery flags, const BondTypeCounts& bondCounts) {
    const uint8_t maskIdx = static_cast<uint8_t>(leafMasks.size());
    leafMasks.push_back(buildQueryMask(packed, flags));
    leafBondCounts.push_back(bondCounts);

    const uint8_t dst = allocateScratch();
    instructions.push_back(BoolInstruction::makeLeaf(dst, maskIdx));
    return dst;
  }

  /**
   * @brief Add an AND instruction combining two operands.
   */
  uint8_t addAnd(uint8_t left, uint8_t right) {
    const uint8_t dst = allocateScratch();
    instructions.push_back(BoolInstruction::makeAnd(dst, left, right));
    return dst;
  }

  /**
   * @brief Add an OR instruction combining two operands.
   */
  uint8_t addOr(uint8_t left, uint8_t right) {
    const uint8_t dst = allocateScratch();
    instructions.push_back(BoolInstruction::makeOr(dst, left, right));
    return dst;
  }

  /**
   * @brief Add a NOT instruction.
   */
  uint8_t addNot(uint8_t src) {
    const uint8_t dst = allocateScratch();
    instructions.push_back(BoolInstruction::makeNot(dst, src));
    return dst;
  }

  /**
   * @brief Add a RecursiveMatch instruction.
   * @param patternId The pattern ID to check (0-15)
   * @return Scratch index where the result will be stored
   */
  uint8_t addRecursiveMatch(uint8_t patternId) {
    const uint8_t dst = allocateScratch();
    instructions.push_back(BoolInstruction::makeRecursiveMatch(dst, patternId));
    return dst;
  }

  /**
   * @brief Add a comparison instruction (for GreaterThan, LessEqual, GreaterEqual).
   * @param op The comparison operation
   * @param field The field to compare
   * @param value The value to compare against
   * @return Scratch index where the result will be stored
   */
  uint8_t addCompare(BoolOp op, CompareField field, uint8_t value) {
    const uint8_t dst = allocateScratch();
    switch (op) {
      case BoolOp::GreaterThan:
        instructions.push_back(BoolInstruction::makeGreaterThan(dst, field, value));
        break;
      case BoolOp::LessEqual:
        instructions.push_back(BoolInstruction::makeLessEqual(dst, field, value));
        break;
      case BoolOp::GreaterEqual:
        instructions.push_back(BoolInstruction::makeGreaterEqual(dst, field, value));
        break;
      default:
        break;
    }
    return dst;
  }

  /**
   * @brief Add a range comparison instruction.
   * @param op Should be BoolOp::Range
   * @param field The field to compare
   * @param minVal Minimum value (inclusive)
   * @param maxVal Maximum value (inclusive)
   * @return Scratch index where the result will be stored
   */
  uint8_t addCompare(BoolOp op, CompareField field, uint8_t minVal, uint8_t maxVal) {
    const uint8_t dst = allocateScratch();
    if (op == BoolOp::Range) {
      instructions.push_back(BoolInstruction::makeRange(dst, field, minVal, maxVal));
    }
    return dst;
  }

  /**
   * @brief Check if the tree exceeds scratch limits.
   */
  bool exceedsScratchLimit() const { return nextScratchIdx > kMaxBoolScratchSize; }

  /**
   * @brief Get the required scratch size.
   */
  int requiredScratchSize() const { return nextScratchIdx; }

  /**
   * @brief Build the final AtomQueryTree metadata.
   */
  AtomQueryTree buildTree() const {
    AtomQueryTree tree;
    tree.numLeaves       = static_cast<uint8_t>(leafMasks.size());
    tree.numInstructions = static_cast<uint8_t>(instructions.size());
    tree.scratchSize     = static_cast<uint8_t>(std::min(nextScratchIdx, 255));
    tree.resultIdx       = nextScratchIdx > 0 ? static_cast<uint8_t>(nextScratchIdx - 1) : 0;
    return tree;
  }
};

/**
 * @brief Check if a query description is a comparison (range) query type.
 *
 * RDKit encodes range queries with prefixes: less_, greater_, range_
 * These require comparison instructions rather than mask/expected matching.
 */
bool isComparisonQuery(const std::string& desc) {
  return desc.rfind("less_", 0) == 0 || desc.rfind("greater_", 0) == 0 || desc.rfind("range_", 0) == 0;
}

/**
 * @brief Map a base query description to its CompareField enum.
 */
CompareField getCompareField(const std::string& baseDesc) {
  if (baseDesc == "AtomMinRingSize") {
    return CompareField::MinRingSize;
  } else if (baseDesc == "AtomInNRings") {
    return CompareField::NumRings;
  } else if (baseDesc == "AtomRingBondCount") {
    return CompareField::RingBondCount;
  } else if (baseDesc == "AtomImplicitHCount") {
    return CompareField::NumImplicitHs;
  } else if (baseDesc == "AtomNumHeteroatomNeighbors") {
    return CompareField::NumHeteroatomNeighbors;
  } else if (baseDesc == "AtomTotalValence") {
    return CompareField::TotalValence;
  } else if (baseDesc == "AtomExplicitDegree") {
    return CompareField::Degree;
  } else if (baseDesc == "AtomHCount") {
    return CompareField::NumExplicitHs;
  }
  throw std::runtime_error("Unsupported comparison field: " + baseDesc);
}

/**
 * @brief Extract the base query name from a comparison query description.
 *
 * E.g., "range_AtomMinRingSize" -> "AtomMinRingSize"
 */
std::string getBaseQueryName(const std::string& desc) {
  if (desc.rfind("less_", 0) == 0) {
    return desc.substr(5);
  } else if (desc.rfind("greater_", 0) == 0) {
    return desc.substr(8);
  } else if (desc.rfind("range_", 0) == 0) {
    return desc.substr(6);
  }
  return desc;
}

/**
 * @brief Check if a query subtree contains only AND operations (no OR/NOT/Recursive/Comparison).
 */
bool isAndOnlyQuery(const RDKit::Atom::QUERYATOM_QUERY* query) {
  if (query->getNegation()) {
    return false;
  }

  const std::string desc = query->getDescription();
  if (desc == "AtomOr" || desc == "AtomXor" || desc == "RecursiveStructure") {
    return false;
  }

  // Comparison queries require special handling
  if (isComparisonQuery(desc) || desc == "AtomHasImplicitH") {
    return false;
  }

  if (desc == "AtomAnd") {
    for (auto it = query->beginChildren(); it != query->endChildren(); ++it) {
      if (!isAndOnlyQuery((*it).get())) {
        return false;
      }
    }
  }

  return true;
}

/**
 * @brief Collect flags and packed data from an AND-only query subtree.
 *
 * This optimized path merges all AND conditions into a single leaf mask.
 * Detects contradictory constraints (same property with different values) and
 * sets AtomQueryNeverMatches when found.
 */
void collectAndOnlyFlags(const RDKit::Atom::QUERYATOM_QUERY* query, AtomQuery& flags, AtomDataPacked& packed) {
  const std::string desc = query->getDescription();

  if constexpr (kDebugBoolTreeBuild) {
    printf("[collectAndOnlyFlags] desc=\"%s\" negated=%d\n", desc.c_str(), query->getNegation());
  }

  // If already marked as never-matching, skip processing
  if (flags & AtomQueryNeverMatches) {
    return;
  }

  if (desc == "AtomAnd") {
    for (auto it = query->beginChildren(); it != query->endChildren(); ++it) {
      collectAndOnlyFlags((*it).get(), flags, packed);
    }
    return;
  }

  if (desc == "AtomType") {
    const auto* eqQuery = static_cast<const RDKit::ATOM_EQUALS_QUERY*>(query);
    int         typeVal = eqQuery->getVal();
    if (typeVal >= 1000) {
      flags |= AtomQueryAtomicNum | AtomQueryIsAromatic;
      packed.setAtomicNum(typeVal - 1000);
      packed.setIsAromatic(true);
    } else {
      flags |= AtomQueryAtomicNum | AtomQueryIsAliphatic;
      packed.setAtomicNum(typeVal);
      packed.setIsAromatic(false);
    }
    return;
  }

  if (desc == "AtomIsAromatic") {
    flags |= AtomQueryIsAromatic;
    packed.setIsAromatic(true);
    return;
  }
  if (desc == "AtomIsAliphatic") {
    flags |= AtomQueryIsAliphatic;
    packed.setIsAromatic(false);
    return;
  }

  if (desc == "AtomNull") {
    return;
  }

  const auto* eqQuery = static_cast<const RDKit::ATOM_EQUALS_QUERY*>(query);

  // Helper to detect conflicting values for the same property
  auto checkConflict = [&](AtomQuery flag, auto currentVal, auto newVal) {
    if ((flags & flag) && currentVal != newVal) {
      flags |= AtomQueryNeverMatches;
      return true;
    }
    return false;
  };

  if (desc == "AtomAtomicNum") {
    if (!checkConflict(AtomQueryAtomicNum, packed.atomicNum(), static_cast<uint8_t>(eqQuery->getVal()))) {
      flags |= AtomQueryAtomicNum;
      packed.setAtomicNum(eqQuery->getVal());
    }
  } else if (desc == "AtomHCount") {
    if (!checkConflict(AtomQueryNumExplicitHs, packed.numExplicitHs(), static_cast<uint8_t>(eqQuery->getVal()))) {
      flags |= AtomQueryNumExplicitHs;
      packed.setNumExplicitHs(eqQuery->getVal());
    }
  } else if (desc == "AtomFormalCharge") {
    if (!checkConflict(AtomQueryFormalCharge, packed.formalCharge(), static_cast<int8_t>(eqQuery->getVal()))) {
      flags |= AtomQueryFormalCharge;
      packed.setFormalCharge(eqQuery->getVal());
    }
  } else if (desc == "AtomHybridization") {
    if (!checkConflict(AtomQueryHybridization, packed.hybridization(), static_cast<uint8_t>(eqQuery->getVal()))) {
      flags |= AtomQueryHybridization;
      packed.setHybridization(eqQuery->getVal());
    }
  } else if (desc == "AtomInNRings") {
    int val = eqQuery->getVal();
    if (val < 0) {
      // [R] any ring query - just check isInRing
      flags |= AtomQueryIsInRing;
      packed.setIsInRing(true);
    } else {
      if (!checkConflict(AtomQueryNumRings, packed.numRings(), static_cast<uint8_t>(val))) {
        flags |= AtomQueryNumRings;
        packed.setNumRings(val);
      }
    }
  } else if (desc == "AtomInRing") {
    // [r] any ring query - just check isInRing
    flags |= AtomQueryIsInRing;
    packed.setIsInRing(true);
  } else if (desc == "AtomMinRingSize") {
    if (!checkConflict(AtomQueryMinRingSize, packed.minRingSize(), static_cast<uint8_t>(eqQuery->getVal()))) {
      flags |= AtomQueryMinRingSize;
      packed.setMinRingSize(eqQuery->getVal());
    }
  } else if (desc == "AtomNumRadicalElectrons") {
    if (!checkConflict(AtomQueryNumRadicalElectrons,
                       packed.numRadicalElectrons(),
                       static_cast<uint8_t>(eqQuery->getVal()))) {
      flags |= AtomQueryNumRadicalElectrons;
      packed.setNumRadicalElectrons(eqQuery->getVal());
    }
  } else if (desc == "AtomTotalValence") {
    if (!checkConflict(AtomQueryTotalValence, packed.totalValence(), static_cast<uint8_t>(eqQuery->getVal()))) {
      flags |= AtomQueryTotalValence;
      packed.setTotalValence(eqQuery->getVal());
    }
  } else if (desc == "AtomRingBondCount") {
    int val = eqQuery->getVal();
    if (val > AtomDataPacked::kMax4BitValue) {
      throw std::runtime_error("Ring bond count query value " + std::to_string(val) +
                               " exceeds maximum storable value of " + std::to_string(AtomDataPacked::kMax4BitValue));
    }
    if (!checkConflict(AtomQueryRingBondCount, packed.ringBondCount(), static_cast<uint8_t>(val))) {
      flags |= AtomQueryRingBondCount;
      packed.setRingBondCount(val);
    }
  } else if (desc == "AtomImplicitHCount") {
    int val = eqQuery->getVal();
    if (val > AtomDataPacked::kMax4BitValue) {
      throw std::runtime_error("Implicit H count query value " + std::to_string(val) +
                               " exceeds maximum storable value of " + std::to_string(AtomDataPacked::kMax4BitValue));
    }
    if (!checkConflict(AtomQueryNumImplicitHs, packed.numImplicitHs(), static_cast<uint8_t>(val))) {
      flags |= AtomQueryNumImplicitHs;
      packed.setNumImplicitHs(val);
    }
  } else if (desc == "AtomHasImplicitH") {
    // [h] without a number - check if numImplicitHs > 0
    // This is handled specially during matching via comparison
    flags |= AtomQueryHasImplicitH;
  } else if (desc == "AtomNumHeteroatomNeighbors") {
    int val = eqQuery->getVal();
    if (val > AtomDataPacked::kMax4BitValue) {
      throw std::runtime_error("Heteroatom neighbor count query value " + std::to_string(val) +
                               " exceeds maximum storable value of " + std::to_string(AtomDataPacked::kMax4BitValue));
    }
    if (!checkConflict(AtomQueryNumHeteroNeighbors, packed.numHeteroatomNeighbors(), static_cast<uint8_t>(val))) {
      flags |= AtomQueryNumHeteroNeighbors;
      packed.setNumHeteroatomNeighbors(val);
    }
  } else if (desc == "AtomMass" || desc == "AtomIsotope") {
    int isotope = eqQuery->getVal();
    if (isotope > 255) {
      throw std::runtime_error("Isotope mass " + std::to_string(isotope) + " exceeds maximum supported value of 255");
    }
    if (!checkConflict(AtomQueryIsotope, packed.isotope(), static_cast<uint8_t>(isotope))) {
      flags |= AtomQueryIsotope;
      packed.setIsotope(static_cast<uint8_t>(isotope));
    }
  } else if (desc == "AtomExplicitDegree") {
    if (!checkConflict(AtomQueryDegree, packed.degree(), static_cast<uint8_t>(eqQuery->getVal()))) {
      flags |= AtomQueryDegree;
      packed.setDegree(eqQuery->getVal());
    }
  } else if (desc == "AtomTotalDegree") {
    if (!checkConflict(AtomQueryTotalConnectivity,
                       packed.totalConnectivity(),
                       static_cast<uint8_t>(eqQuery->getVal()))) {
      flags |= AtomQueryTotalConnectivity;
      packed.setTotalConnectivity(eqQuery->getVal());
    }
  } else {
    AtomQuery flag = atomQueryFromDescription(desc);
    if constexpr (kDebugBoolTreeBuild) {
      printf("[collectAndOnlyFlags] unhandled desc=\"%s\" -> flag=0x%x\n", desc.c_str(), flag);
    }
    flags |= flag;
  }
}

/**
 * @brief Recursively process a query tree and build boolean instructions.
 *
 * @param query The RDKit query to process
 * @param builder The builder accumulating leaves and instructions
 * @param bondCounts Bond type counts for the atom (used for leaf nodes)
 * @param nextPatternId Reference to pattern ID counter for recursive SMARTS
 * @param childPatternIds Optional: explicit pattern IDs for RecursiveMatch (for cached patterns)
 * @return Scratch index where this subtree's result will be stored
 */
uint8_t processQueryTree(const RDKit::Atom::QUERYATOM_QUERY* query,
                         QueryTreeBuilder&                   builder,
                         const BondTypeCounts&               bondCounts,
                         int&                                nextPatternId,
                         const std::vector<int>*             childPatternIds = nullptr) {
  const std::string desc      = query->getDescription();
  const bool        isNegated = query->getNegation();

  // Handle AND-only subtrees efficiently by merging into a single leaf
  if (!isNegated && isAndOnlyQuery(query)) {
    AtomQuery      flags  = AtomQueryNone;
    AtomDataPacked packed = {};
    collectAndOnlyFlags(query, flags, packed);

    // Detect contradictory aromaticity constraints (e.g., [C;a] or [c;A])
    if ((flags & AtomQueryIsAromatic) && (flags & AtomQueryIsAliphatic)) {
      flags |= AtomQueryNeverMatches;
    }

    return builder.addLeaf(packed, flags, bondCounts);
  }

  // Handle OR: process children and combine with OR instructions
  if (desc == "AtomOr") {
    std::vector<uint8_t> childResults;
    for (auto it = query->beginChildren(); it != query->endChildren(); ++it) {
      childResults.push_back(processQueryTree((*it).get(), builder, bondCounts, nextPatternId, childPatternIds));
    }

    if (childResults.empty()) {
      throw std::runtime_error("Empty AtomOr query");
    }

    uint8_t result = childResults[0];
    for (size_t i = 1; i < childResults.size(); ++i) {
      result = builder.addOr(result, childResults[i]);
    }

    if (isNegated) {
      result = builder.addNot(result);
    }
    return result;
  }

  // Handle AND with complex children (some may have OR or NOT)
  if (desc == "AtomAnd") {
    std::vector<uint8_t> childResults;
    for (auto it = query->beginChildren(); it != query->endChildren(); ++it) {
      childResults.push_back(processQueryTree((*it).get(), builder, bondCounts, nextPatternId, childPatternIds));
    }

    if (childResults.empty()) {
      throw std::runtime_error("Empty AtomAnd query");
    }

    uint8_t result = childResults[0];
    for (size_t i = 1; i < childResults.size(); ++i) {
      result = builder.addAnd(result, childResults[i]);
    }

    if (isNegated) {
      result = builder.addNot(result);
    }
    return result;
  }

  if (desc == "AtomXor") {
    throw std::runtime_error("SMARTS XOR queries are not supported");
  }

  if (desc == "RecursiveStructure") {
    int patternIdToUse;
    if (childPatternIds != nullptr) {
      if (nextPatternId >= static_cast<int>(childPatternIds->size())) {
        throw std::runtime_error("Child pattern ID index out of bounds");
      }
      patternIdToUse = (*childPatternIds)[nextPatternId++];
    } else {
      if (nextPatternId >= RecursivePatternInfo::kMaxPatterns) {
        throw std::runtime_error("Too many recursive SMARTS patterns (maximum " +
                                 std::to_string(RecursivePatternInfo::kMaxPatterns) + " supported)");
      }
      patternIdToUse = nextPatternId++;
    }
    if constexpr (kDebugBoolTreeBuild) {
      printf("[BoolTreeBuild] Adding RecursiveMatch for patternId=%d\n", patternIdToUse);
    }
    uint8_t result = builder.addRecursiveMatch(static_cast<uint8_t>(patternIdToUse));
    if (isNegated) {
      result = builder.addNot(result);
    }
    return result;
  }

  // Handle [h] (has implicit H) query - check if numImplicitHs > 0
  if (desc == "AtomHasImplicitH") {
    uint8_t result = builder.addCompare(BoolOp::GreaterThan, CompareField::NumImplicitHs, 0);
    if (isNegated) {
      result = builder.addNot(result);
    }
    return result;
  }

  // Handle comparison queries (less_, greater_, range_ prefixes)
  if (isComparisonQuery(desc)) {
    const std::string  baseDesc = getBaseQueryName(desc);
    const CompareField field    = getCompareField(baseDesc);
    // Cast to ATOM_EQUALS_QUERY to access getVal()/getTol() - all numeric query types derive from this
    const auto*        eqQuery  = static_cast<const RDKit::ATOM_EQUALS_QUERY*>(query);
    const int          queryVal = eqQuery->getVal();

    uint8_t result;
    if (desc.rfind("less_", 0) == 0) {
      // RDKit's "less_" means the value is a LOWER bound (field >= val)
      // This is counterintuitive but matches RDKit's actual behavior
      result = builder.addCompare(BoolOp::GreaterEqual, field, static_cast<uint8_t>(queryVal));
    } else if (desc.rfind("greater_", 0) == 0) {
      // RDKit's "greater_" means the value is an UPPER bound (field <= val)
      result = builder.addCompare(BoolOp::LessEqual, field, static_cast<uint8_t>(queryVal));
    } else {
      // range_ - RDKit stores bounds in getLower()/getUpper(), not getVal()/getTol()
      const auto* rangeQuery = static_cast<const RDKit::ATOM_RANGE_QUERY*>(query);
      int         minVal     = rangeQuery->getLower();
      int         maxVal     = rangeQuery->getUpper();
      if (minVal < 0)
        minVal = 0;
      if (maxVal > 255)
        maxVal = 255;
      result = builder.addCompare(BoolOp::Range, field, static_cast<uint8_t>(minVal), static_cast<uint8_t>(maxVal));
    }

    if (isNegated) {
      result = builder.addNot(result);
    }
    return result;
  }

  // Leaf node - create a single leaf mask
  AtomQuery      flags  = AtomQueryNone;
  AtomDataPacked packed = {};
  collectAndOnlyFlags(query, flags, packed);
  uint8_t result = builder.addLeaf(packed, flags, bondCounts);

  if (isNegated) {
    result = builder.addNot(result);
  }
  return result;
}

/**
 * @brief Build a complete query tree for an atom.
 *
 * @param atom The RDKit atom to process
 * @param bondCounts Bond type counts for the atom
 * @param builder Output: the populated QueryTreeBuilder
 * @param nextPatternId Reference to pattern ID counter for recursive SMARTS
 * @param childPatternIds Optional: explicit pattern IDs for RecursiveMatch (for cached patterns)
 */
void buildQueryTreeForAtom(const RDKit::Atom*      atom,
                           const BondTypeCounts&   bondCounts,
                           QueryTreeBuilder&       builder,
                           int&                    nextPatternId,
                           const std::vector<int>* childPatternIds = nullptr) {
  // Check for chirality specified on the atom (SMARTS @/@@ notation)
  if (atom->getChiralTag() != RDKit::Atom::ChiralType::CHI_UNSPECIFIED) {
    throw std::runtime_error("SMARTS chirality query (@/@@) is not supported");
  }

  if (!atom->hasQuery()) {
    builder.addLeaf(AtomDataPacked{}, AtomQueryNone, bondCounts);
    return;
  }

  const auto* query = atom->getQuery();
  if (query == nullptr) {
    builder.addLeaf(AtomDataPacked{}, AtomQueryNone, bondCounts);
    return;
  }

  processQueryTree(query, builder, bondCounts, nextPatternId, childPatternIds);
}

}  // namespace

void addToBatch(const RDKit::ROMol* mol, MoleculesHost& batch) {
  ScopedNvtxRange range("addToBatch");
  if (mol->getNumAtoms() > kMaxTargetAtoms) {
    throw std::runtime_error("Target molecule has " + std::to_string(mol->getNumAtoms()) +
                             " atoms, which exceeds the maximum of " + std::to_string(kMaxTargetAtoms));
  }

  const auto* ringInfo = mol->getRingInfo();
  populateTargetMolecule(mol, batch, ringInfo);

  batch.batchAtomStarts.push_back(static_cast<int>(batch.atomDataPacked.size()));
}

namespace {

void populateQueryAtomDataPacked(const RDKit::Atom* atom, AtomDataPacked& packed) {
  if (!atom->hasQuery()) {
    return;
  }

  const auto* query = atom->getQuery();
  if (query == nullptr) {
    return;
  }

  // Use the existing populateFromQuery logic to extract values, but store in packed format
  // We'll duplicate the logic here to avoid converting back and forth
  std::function<void(const RDKit::Atom::QUERYATOM_QUERY*)> populatePacked;
  populatePacked = [&](const RDKit::Atom::QUERYATOM_QUERY* q) {
    const std::string desc = q->getDescription();

    if (desc == "AtomAnd") {
      for (auto it = q->beginChildren(); it != q->endChildren(); ++it) {
        populatePacked((*it).get());
      }
      return;
    }

    if (desc == "AtomType") {
      const auto* eqQuery = static_cast<const RDKit::ATOM_EQUALS_QUERY*>(q);
      int         typeVal = eqQuery->getVal();
      if (typeVal >= 1000) {
        packed.setAtomicNum(typeVal - 1000);
        packed.setIsAromatic(true);
      } else {
        packed.setAtomicNum(typeVal);
        packed.setIsAromatic(false);
      }
      return;
    }

    if (desc == "AtomIsAromatic") {
      packed.setIsAromatic(true);
      return;
    }
    if (desc == "AtomIsAliphatic") {
      packed.setIsAromatic(false);
      return;
    }

    const auto* eqQuery = static_cast<const RDKit::ATOM_EQUALS_QUERY*>(q);

    if (desc == "AtomAtomicNum") {
      packed.setAtomicNum(eqQuery->getVal());
    } else if (desc == "AtomHCount") {
      packed.setNumExplicitHs(eqQuery->getVal());
    } else if (desc == "AtomFormalCharge") {
      packed.setFormalCharge(eqQuery->getVal());
    } else if (desc == "AtomHybridization") {
      packed.setHybridization(eqQuery->getVal());
    } else if (desc == "AtomInNRings") {
      packed.setNumRings(eqQuery->getVal());
    } else if (desc == "AtomMinRingSize") {
      packed.setMinRingSize(eqQuery->getVal());
    } else if (desc == "AtomNumRadicalElectrons") {
      packed.setNumRadicalElectrons(eqQuery->getVal());
    } else if (desc == "AtomTotalValence") {
      packed.setTotalValence(eqQuery->getVal());
    } else if (desc == "AtomMass" || desc == "AtomIsotope") {
      int isotope = eqQuery->getVal();
      if (isotope > 255) {
        throw std::runtime_error("Isotope mass " + std::to_string(isotope) + " exceeds maximum supported value of 255");
      }
      packed.setIsotope(static_cast<uint8_t>(isotope));
    } else if (desc == "AtomExplicitDegree") {
      packed.setDegree(eqQuery->getVal());
    } else if (desc == "AtomTotalDegree") {
      packed.setTotalConnectivity(eqQuery->getVal());
    } else if (desc == "AtomRingBondCount") {
      packed.setRingBondCount(eqQuery->getVal());
    } else if (desc == "AtomImplicitHCount") {
      packed.setNumImplicitHs(eqQuery->getVal());
    } else if (desc == "AtomNumHeteroatomNeighbors") {
      packed.setNumHeteroatomNeighbors(eqQuery->getVal());
    }
  };

  populatePacked(query);
}

/**
 * @brief Extract bond query flags from an RDKit bond query.
 *
 * Handles ring bond constraints (!@ and @) by examining the bond's query.
 */
void extractBondQueryFlags(const RDKit::Bond* bond, BondQueryData& queryData) {
  queryData.bondType   = bond->getBondType();
  queryData.queryFlags = BondQueryNone;

  if (!bond->hasQuery()) {
    return;
  }

  const auto* query = bond->getQuery();
  if (query == nullptr) {
    return;
  }

  // Recursive function to process bond query tree
  std::function<void(const RDKit::Bond::QUERYBOND_QUERY*)> processQuery;
  processQuery = [&](const RDKit::Bond::QUERYBOND_QUERY* q) {
    const std::string desc      = q->getDescription();
    const bool        isNegated = q->getNegation();

    if (desc == "BondAnd") {
      // Check for impossible constraints like single AND aromatic
      // Must track both positive and negated constraints:
      //   -:  (single AND aromatic) -> impossible
      //   -!: (single AND NOT aromatic) -> valid
      //   !-: (NOT single AND aromatic) -> valid
      bool hasPositiveSingle   = false;
      bool hasPositiveDouble   = false;
      bool hasPositiveTriple   = false;
      bool hasPositiveAromatic = false;
      for (auto it = q->beginChildren(); it != q->endChildren(); ++it) {
        const std::string childDesc    = (*it)->getDescription();
        const bool        childNegated = (*it)->getNegation();
        if (childDesc == "BondOrder") {
          const auto* eqQuery  = static_cast<const RDKit::BOND_EQUALS_QUERY*>((*it).get());
          int         bondType = eqQuery->getVal();
          // Only count positive (non-negated) bond order constraints
          if (!childNegated) {
            if (bondType == 1) {
              hasPositiveSingle = true;
            } else if (bondType == 2) {
              hasPositiveDouble = true;
            } else if (bondType == 3) {
              hasPositiveTriple = true;
            } else if (bondType == 7 || bondType == 12) {
              hasPositiveAromatic = true;
            }
          }
        } else if (childDesc == "BondIsAromatic") {
          // Only count positive (non-negated) aromatic constraint
          // !: (NOT aromatic) is valid with single/double/triple
          if (!childNegated) {
            hasPositiveAromatic = true;
          }
        }
      }
      // Conflicting constraints: positive non-aromatic bond type AND positive aromatic requirement
      // Examples that ARE impossible: -: (single AND aromatic)
      // Examples that are valid: -!: (single AND NOT aromatic), !-: (NOT single AND aromatic)
      if (hasPositiveAromatic && (hasPositiveSingle || hasPositiveDouble || hasPositiveTriple)) {
        queryData.queryFlags |= BondQueryNeverMatches;
        return;
      }
      // Process children normally for other BondAnd patterns (e.g., ring constraints)
      for (auto it = q->beginChildren(); it != q->endChildren(); ++it) {
        processQuery((*it).get());
      }
      return;
    }

    if (desc == "BondOr") {
      // Collect all allowed bond types from the OR pattern (recursive for nested BondOr)
      std::function<uint16_t(const RDKit::Bond::QUERYBOND_QUERY*)> collectBondMask;
      collectBondMask = [&](const RDKit::Bond::QUERYBOND_QUERY* orQuery) -> uint16_t {
        uint16_t mask = 0;
        for (auto it = orQuery->beginChildren(); it != orQuery->endChildren(); ++it) {
          const std::string childDesc = (*it)->getDescription();
          if (childDesc == "BondOr") {
            mask |= collectBondMask((*it).get());
          } else if (childDesc == "BondOrder") {
            const auto* eqQuery  = static_cast<const RDKit::BOND_EQUALS_QUERY*>((*it).get());
            int         bondType = eqQuery->getVal();
            if (bondType >= 0 && bondType < 16) {
              mask |= (1u << bondType);
              if (bondType == 7 || bondType == 12) {
                mask |= (1u << 7) | (1u << 12);
              }
            }
          } else if (childDesc == "BondIsAromatic") {
            mask |= (1u << 7) | (1u << 12);
          } else if (childDesc == "DoubleOrAromaticBond") {
            mask |= (1u << 2) | (1u << 7) | (1u << 12);
          } else if (childDesc == "SingleOrAromaticBond") {
            mask |= (1u << 1) | (1u << 7) | (1u << 12);
          } else if (childDesc == "TripleBond") {
            mask |= (1u << 3);
          } else if (childDesc == "DoubleBond") {
            mask |= (1u << 2);
          } else if (childDesc == "SingleBond") {
            mask |= (1u << 1);
          }
        }
        return mask;
      };

      uint16_t bondMask = collectBondMask(q);

      if (bondMask != 0) {
        queryData.queryFlags |= BondQueryUseBondMask;
        queryData.allowedBondTypes = bondMask;
        return;
      }

      // Fall through to process children normally for other BondOr patterns
      for (auto it = q->beginChildren(); it != q->endChildren(); ++it) {
        processQuery((*it).get());
      }
      return;
    }

    if (desc == "BondIsInRing" || desc == "BondInRing") {
      if (isNegated) {
        queryData.queryFlags |= BondQueryNotRingBond;
      } else {
        queryData.queryFlags |= BondQueryIsRingBond;
      }
    } else if (desc == "SingleOrAromaticBond") {
      queryData.queryFlags |= BondQueryUseBondMask;
      queryData.allowedBondTypes = (1u << 1) | (1u << 7) | (1u << 12);  // single, oneandahalf, aromatic
    } else if (desc == "DoubleOrAromaticBond") {
      queryData.queryFlags |= BondQueryUseBondMask;
      queryData.allowedBondTypes = (1u << 2) | (1u << 7) | (1u << 12);  // double, oneandahalf, aromatic
    } else if (desc == "BondIsAromatic") {
      queryData.queryFlags |= BondQueryUseBondMask;
      queryData.allowedBondTypes = (1u << 7) | (1u << 12);  // oneandahalf, aromatic
    } else if (desc == "BondOrder") {
      const auto* eqQuery  = static_cast<const RDKit::BOND_EQUALS_QUERY*>(q);
      int         bondType = eqQuery->getVal();
      queryData.bondType   = bondType;
      // Also set up mask for consistency with edge consistency checker
      if (bondType >= 0 && bondType < 16) {
        queryData.queryFlags |= BondQueryUseBondMask;
        if (isNegated) {
          // NOT this bond type - allow all common types except this one
          // Common bond types: single(1), double(2), triple(3), aromatic(7,12)
          uint16_t allCommonTypes = (1u << 1) | (1u << 2) | (1u << 3) | (1u << 7) | (1u << 12);
          uint16_t excludeMask    = (1u << bondType);
          if (bondType == 7 || bondType == 12) {
            excludeMask = (1u << 7) | (1u << 12);  // Exclude both aromatic representations
          }
          queryData.allowedBondTypes = allCommonTypes & ~excludeMask;
        } else {
          queryData.allowedBondTypes = (1u << bondType);
          // Aromatic bonds can be stored as type 7 or 12
          if (bondType == 7 || bondType == 12) {
            queryData.allowedBondTypes |= (1u << 7) | (1u << 12);
          }
        }
      }
    } else if (desc == "BondNull") {
      queryData.bondType = 0;  // Any bond - don't set mask, bondTypeMatches handles type 0
    }
  };

  processQuery(query);
}

/**
 * @brief Add bonds and connectivity for a query molecule (SMARTS).
 *
 * Similar to addBondsAndConnectivity but also extracts bond query information
 * including ring bond constraints.
 */
void populateQueryAtomBonds(const RDKit::ROMol* mol, MoleculesHost& batch) {
  auto& queryAtomBondsVec = batch.queryAtomBonds;

  for (const RDKit::Atom* atom : mol->atoms()) {
    auto& qab  = queryAtomBondsVec.emplace_back();
    qab.degree = 0;

    const unsigned int atomIdx = atom->getIdx();
    auto [beg, bondEnd]        = mol->getAtomBonds(atom);

    while (beg != bondEnd && qab.degree < kMaxBondsPerAtom) {
      const auto* bond        = (*mol)[*beg];
      const int   otherAtomId = bond->getOtherAtomIdx(atomIdx);

      BondQueryData bqd;
      extractBondQueryFlags(bond, bqd);

      if constexpr (kDebugBoolTreeBuild) {
        printf("[BondQuery] atom %d bond %d (%d-%d): queryType=%d, flags=0x%x, allowedTypes=0x%x\n",
               atomIdx,
               qab.degree,
               bond->getBeginAtomIdx(),
               bond->getEndAtomIdx(),
               bqd.bondType,
               bqd.queryFlags,
               bqd.allowedBondTypes);
      }

      qab.neighborIdx[qab.degree] = static_cast<uint8_t>(otherAtomId);
      qab.matchMask[qab.degree]   = buildQueryBondMatchMask(bqd.bondType, bqd.queryFlags, bqd.allowedBondTypes);
      ++qab.degree;
      ++beg;
    }

    if (beg != bondEnd) {
      throw std::runtime_error("Query atom has more than " + std::to_string(kMaxBondsPerAtom) + " bonds");
    }
  }
}

}  // namespace

void addQueryToBatch(const RDKit::ROMol* mol, MoleculesHost& batch) {
  ScopedNvtxRange range("addQueryToBatch");
  if (mol->getNumAtoms() > kMaxTargetAtoms) {
    throw std::runtime_error("Query molecule has " + std::to_string(mol->getNumAtoms()) +
                             " atoms, which exceeds the maximum of " + std::to_string(kMaxTargetAtoms));
  }

  std::vector<int> fragMapping;
  const unsigned   numFrags = RDKit::MolOps::getMolFrags(*mol, fragMapping);
  if (numFrags > 1) {
    throw std::runtime_error(
      "Fragment queries (disconnected SMARTS patterns) are not supported. "
      "Query has " +
      std::to_string(numFrags) + " disconnected components: " + RDKit::MolToSmarts(*mol));
  }

  auto& atomDataPackedVec = batch.atomDataPacked;
  auto& atomQueryMasksVec = batch.atomQueryMasks;
  auto& bondTypeCountsVec = batch.bondTypeCounts;

  // Boolean tree data
  auto& atomQueryTreesVec      = batch.atomQueryTrees;
  auto& queryInstructionsVec   = batch.queryInstructions;
  auto& queryLeafMasksVec      = batch.queryLeafMasks;
  auto& queryLeafBondCountsVec = batch.queryLeafBondCounts;
  auto& atomInstrStartsVec     = batch.atomInstrStarts;
  auto& atomLeafMaskStartsVec  = batch.atomLeafMaskStarts;

  populateQueryAtomBonds(mol, batch);

  int nextPatternId = 0;

  if constexpr (kDebugBoolTreeBuild) {
    printf("[BoolTreeBuild] addQueryToBatch: starting with nextPatternId=0, smarts=%s\n",
           RDKit::MolToSmarts(*mol).c_str());
  }

  for (const RDKit::Atom* atom : mol->atoms()) {
    auto& thisAtomPacked = atomDataPackedVec.emplace_back();
    populateQueryAtomDataPacked(atom, thisAtomPacked);

    // Compute bond type counts first (needed for query tree building)
    auto& thisBondCounts = bondTypeCountsVec.emplace_back();
    populateQueryBondTypeCounts(mol, atom, thisBondCounts);

    // Build the boolean expression tree for this atom
    QueryTreeBuilder builder;
    buildQueryTreeForAtom(atom, thisBondCounts, builder, nextPatternId);

    if (builder.exceedsScratchLimit()) {
      throw std::runtime_error("SMARTS query too complex: boolean expression requires " +
                               std::to_string(builder.requiredScratchSize()) + " scratch slots, but maximum is " +
                               std::to_string(kMaxBoolScratchSize) + ". Query: " + RDKit::MolToSmarts(*mol));
    }

    // Store offsets into global instruction/leaf arrays
    atomInstrStartsVec.push_back(static_cast<int>(queryInstructionsVec.size()));
    atomLeafMaskStartsVec.push_back(static_cast<int>(queryLeafMasksVec.size()));

    // Append this atom's data to global arrays
    queryInstructionsVec.insert(queryInstructionsVec.end(), builder.instructions.begin(), builder.instructions.end());
    queryLeafMasksVec.insert(queryLeafMasksVec.end(), builder.leafMasks.begin(), builder.leafMasks.end());
    queryLeafBondCountsVec.insert(queryLeafBondCountsVec.end(),
                                  builder.leafBondCounts.begin(),
                                  builder.leafBondCounts.end());
    atomQueryTreesVec.push_back(builder.buildTree());

    // For simple queries, use the first leaf mask
    if (!builder.leafMasks.empty()) {
      atomQueryMasksVec.push_back(builder.leafMasks[0]);
    } else {
      atomQueryMasksVec.push_back(AtomQueryMask{});
    }
  }

  batch.batchAtomStarts.push_back(static_cast<int>(atomDataPackedVec.size()));

  // Extract recursive SMARTS patterns for preprocessing
  batch.recursivePatterns.push_back(extractRecursivePatterns(mol));
}

void addQueryToBatch(const RDKit::ROMol* mol, MoleculesHost& batch, const std::vector<int>& childPatternIds) {
  if (mol->getNumAtoms() > kMaxTargetAtoms) {
    throw std::runtime_error("Query molecule has " + std::to_string(mol->getNumAtoms()) +
                             " atoms, which exceeds the maximum of " + std::to_string(kMaxTargetAtoms));
  }

  std::vector<int> fragMapping;
  const unsigned   numFrags = RDKit::MolOps::getMolFrags(*mol, fragMapping);
  if (numFrags > 1) {
    throw std::runtime_error(
      "Fragment queries (disconnected SMARTS patterns) are not supported. "
      "Query has " +
      std::to_string(numFrags) + " disconnected components: " + RDKit::MolToSmarts(*mol));
  }

  auto& atomDataPackedVec      = batch.atomDataPacked;
  auto& atomQueryMasksVec      = batch.atomQueryMasks;
  auto& bondTypeCountsVec      = batch.bondTypeCounts;
  auto& atomQueryTreesVec      = batch.atomQueryTrees;
  auto& queryInstructionsVec   = batch.queryInstructions;
  auto& queryLeafMasksVec      = batch.queryLeafMasks;
  auto& queryLeafBondCountsVec = batch.queryLeafBondCounts;
  auto& atomInstrStartsVec     = batch.atomInstrStarts;
  auto& atomLeafMaskStartsVec  = batch.atomLeafMaskStarts;

  populateQueryAtomBonds(mol, batch);

  int nextPatternId = 0;

  if constexpr (kDebugBoolTreeBuild) {
    printf("[BoolTreeBuild] addQueryToBatch (with childPatternIds): smarts=%s, childIds=[",
           RDKit::MolToSmarts(*mol).c_str());
    for (size_t i = 0; i < childPatternIds.size(); ++i) {
      printf("%d%s", childPatternIds[i], i + 1 < childPatternIds.size() ? "," : "");
    }
    printf("]\n");
  }

  for (const RDKit::Atom* atom : mol->atoms()) {
    auto& thisAtomPacked = atomDataPackedVec.emplace_back();
    populateQueryAtomDataPacked(atom, thisAtomPacked);

    auto& thisBondCounts = bondTypeCountsVec.emplace_back();
    populateQueryBondTypeCounts(mol, atom, thisBondCounts);

    QueryTreeBuilder builder;
    buildQueryTreeForAtom(atom, thisBondCounts, builder, nextPatternId, &childPatternIds);

    if (builder.exceedsScratchLimit()) {
      throw std::runtime_error("SMARTS query too complex: boolean expression requires " +
                               std::to_string(builder.requiredScratchSize()) + " scratch slots, but maximum is " +
                               std::to_string(kMaxBoolScratchSize) + ". Query: " + RDKit::MolToSmarts(*mol));
    }

    atomInstrStartsVec.push_back(static_cast<int>(queryInstructionsVec.size()));
    atomLeafMaskStartsVec.push_back(static_cast<int>(queryLeafMasksVec.size()));

    queryInstructionsVec.insert(queryInstructionsVec.end(), builder.instructions.begin(), builder.instructions.end());
    queryLeafMasksVec.insert(queryLeafMasksVec.end(), builder.leafMasks.begin(), builder.leafMasks.end());
    queryLeafBondCountsVec.insert(queryLeafBondCountsVec.end(),
                                  builder.leafBondCounts.begin(),
                                  builder.leafBondCounts.end());
    atomQueryTreesVec.push_back(builder.buildTree());

    if (!builder.leafMasks.empty()) {
      atomQueryMasksVec.push_back(builder.leafMasks[0]);
    } else {
      atomQueryMasksVec.push_back(AtomQueryMask{});
    }
  }

  batch.batchAtomStarts.push_back(static_cast<int>(atomDataPackedVec.size()));

  // No recursive patterns extraction for cached patterns - they're pre-extracted
  batch.recursivePatterns.emplace_back();
}

namespace {

/**
 * @brief Check if a query contains any RecursiveStructure queries.
 *
 * @param query The query to check
 * @return true if the query contains any RecursiveStructure queries
 */
bool containsRecursiveSmarts(const RDKit::Atom::QUERYATOM_QUERY* query) {
  if (query == nullptr) {
    return false;
  }

  const std::string desc = query->getDescription();
  if (desc == "RecursiveStructure") {
    return true;
  }

  for (auto it = query->beginChildren(); it != query->endChildren(); ++it) {
    if (containsRecursiveSmarts((*it).get())) {
      return true;
    }
  }
  return false;
}

/**
 * @brief Recursively collect RecursiveStructure patterns from a query tree.
 *
 * Extracts patterns depth-first: child patterns are collected before parents.
 * Each pattern gets a unique patternId and tracks its parent relationship.
 *
 * @param query The query to search
 * @param atomIdx The atom index containing this query (in the parent molecule)
 * @param patterns Output vector of patterns found
 * @param nextPatternId Next available pattern ID
 * @param parentIdx Index of parent pattern in patterns array, -1 for root
 * @param currentDepth Current nesting depth
 * @param nextLocalId Counter for localIdInParent assignment (passed to children)
 * @return Maximum depth encountered in this subtree
 */
struct PendingRecursiveChild {
  int                 patternIdx = -1;
  const RDKit::ROMol* queryMol   = nullptr;
  int                 atomIdx    = -1;
  int                 depth      = 0;
};

int collectRecursivePatterns(const RDKit::Atom::QUERYATOM_QUERY* query,
                             int                                 atomIdx,
                             std::vector<RecursivePatternEntry>& patterns,
                             int&                                nextPatternId,
                             int                                 parentIdx,
                             int                                 currentDepth,
                             int&                                nextLocalId,
                             std::vector<PendingRecursiveChild>* pendingChildren) {
  if (query == nullptr) {
    return 0;
  }

  const std::string desc     = query->getDescription();
  int               maxDepth = 0;

  if (desc == "RecursiveStructure") {
    if (nextPatternId >= RecursivePatternInfo::kMaxPatterns) {
      throw std::runtime_error("Too many recursive SMARTS patterns (maximum " +
                               std::to_string(RecursivePatternInfo::kMaxPatterns) + " supported)");
    }

    const auto* recursiveQuery = static_cast<const RDKit::RecursiveStructureQuery*>(query);
    auto        queryMol       = recursiveQuery->getQueryMol();

    if (queryMol != nullptr) {
      const int thisPatternIdx = static_cast<int>(patterns.size());
      const int thisLocalId    = nextLocalId++;

      RecursivePatternEntry entry;
      entry.queryMol           = queryMol;
      entry.queryAtomIdx       = atomIdx;
      entry.patternId          = nextPatternId++;
      entry.parentPatternIdx   = parentIdx;
      entry.parentPatternId    = (parentIdx >= 0) ? patterns[parentIdx].patternId : -1;
      entry.parentQueryAtomIdx = atomIdx;
      entry.localIdInParent    = thisLocalId;
      patterns.push_back(entry);

      if constexpr (kDebugBoolTreeBuild) {
        printf(
          "[ExtractPatterns] Added pattern: patternId=%d, depth=%d, parentIdx=%d, parentPatternId=%d, localIdInParent=%d, smarts=%s\n",
          entry.patternId,
          currentDepth,
          parentIdx,
          entry.parentPatternId,
          thisLocalId,
          RDKit::MolToSmarts(*queryMol).c_str());
      }

      // Defer descending into the recursive query molecule until we've assigned IDs
      // for all RecursiveStructure nodes in this molecule. This keeps top-level
      // recursive IDs consistent with the order used when building query trees.
      if (pendingChildren != nullptr) {
        pendingChildren->push_back(PendingRecursiveChild{thisPatternIdx, queryMol, atomIdx, currentDepth});
      }
      // This RecursiveStructure contributes at least one level of depth.
      maxDepth = std::max(maxDepth, 1);
    }
    return maxDepth;
  }

  for (auto it = query->beginChildren(); it != query->endChildren(); ++it) {
    int childDepth = collectRecursivePatterns((*it).get(),
                                              atomIdx,
                                              patterns,
                                              nextPatternId,
                                              parentIdx,
                                              currentDepth,
                                              nextLocalId,
                                              pendingChildren);
    maxDepth       = std::max(maxDepth, childDepth);
  }

  return maxDepth;
}

/**
 * @brief Recursively collect patterns from a molecule's atoms.
 *
 * Helper for extracting patterns from inner query molecules.
 *
 * @param mol The molecule to extract patterns from
 * @param patterns Output vector of patterns found
 * @param nextPatternId Next available pattern ID
 * @param parentIdx Index of parent pattern in patterns array, -1 for root
 * @param parentAtomIdx Atom index in the parent pattern that contains this pattern
 * @param currentDepth Current nesting depth
 * @param nextLocalId Counter for localIdInParent assignment (reset per parent)
 * @return Maximum depth encountered in this subtree
 */
int collectPatternsFromMolecule(const RDKit::ROMol*                 mol,
                                std::vector<RecursivePatternEntry>& patterns,
                                int&                                nextPatternId,
                                int                                 parentIdx,
                                int /* parentAtomIdx */,
                                int  currentDepth,
                                int& nextLocalId) {
  int                                maxDepth = 0;
  std::vector<PendingRecursiveChild> pendingChildren;
  for (const auto* atom : mol->atoms()) {
    if (!atom->hasQuery()) {
      continue;
    }

    const auto* query = atom->getQuery();
    if (query != nullptr) {
      int childDepth = collectRecursivePatterns(query,
                                                atom->getIdx(),
                                                patterns,
                                                nextPatternId,
                                                parentIdx,
                                                currentDepth,
                                                nextLocalId,
                                                &pendingChildren);
      maxDepth       = std::max(maxDepth, childDepth);
    }
  }

  // Now that all recursive patterns in this molecule have stable patternIds, walk
  // into each recursive query molecule to collect nested patterns and finalize depths.
  for (const auto& pending : pendingChildren) {
    int childLocalId  = 0;
    int childMaxDepth = 0;
    if (pending.queryMol != nullptr) {
      childMaxDepth = collectPatternsFromMolecule(pending.queryMol,
                                                  patterns,
                                                  nextPatternId,
                                                  pending.patternIdx,
                                                  pending.atomIdx,
                                                  currentDepth + 1,
                                                  childLocalId);
    }
    patterns[pending.patternIdx].depth = childMaxDepth;
    maxDepth                           = std::max(maxDepth, childMaxDepth + 1);
  }

  return maxDepth;
}

}  // namespace

bool hasRecursiveSmarts(const RDKit::ROMol* mol) {
  if (mol == nullptr) {
    return false;
  }

  for (const auto* atom : mol->atoms()) {
    if (!atom->hasQuery()) {
      continue;
    }

    const auto* query = atom->getQuery();
    if (query != nullptr && containsRecursiveSmarts(query)) {
      return true;
    }
  }
  return false;
}

RecursivePatternInfo extractRecursivePatterns(const RDKit::ROMol* mol) {
  RecursivePatternInfo info;

  if (mol == nullptr) {
    return info;
  }

  int nextPatternId = 0;
  int nextLocalId   = 0;

  // Collect patterns molecule-wide so that patternIds are assigned in the same
  // order that query trees encounter RecursiveStructure nodes (outer level first).
  int maxDepth = collectPatternsFromMolecule(mol, info.patterns, nextPatternId, -1, -1, 0, nextLocalId);

  info.hasRecursivePatterns = !info.patterns.empty();
  info.maxDepth             = maxDepth;

  if (info.size() > RecursivePatternInfo::kMaxPatterns) {
    throw std::runtime_error("Query contains " + std::to_string(info.size()) + " recursive SMARTS patterns, but only " +
                             std::to_string(RecursivePatternInfo::kMaxPatterns) + " are supported");
  }

  std::stable_sort(info.patterns.begin(),
                   info.patterns.end(),
                   [](const RecursivePatternEntry& a, const RecursivePatternEntry& b) { return a.depth < b.depth; });

  if constexpr (kDebugBoolTreeBuild) {
    printf("[ExtractPatterns] After sorting by depth:\n");
    for (const auto& p : info.patterns) {
      printf("[ExtractPatterns]   patternId=%d, depth=%d, parentPatternId=%d, localIdInParent=%d\n",
             p.patternId,
             p.depth,
             p.parentPatternId,
             p.localIdInParent);
    }
  }

  return info;
}

/**
 * @brief Merge a source batch into destination, adjusting all offsets.
 *
 * Works for both target and query batches. Query-specific fields are only
 * copied if non-empty in the source.
 */
void mergeBatch(MoleculesHost& dest, const MoleculesHost& src) {
  ScopedNvtxRange range("mergeBatch");
  if (src.numMolecules() == 0)
    return;

  const int atomOffset     = static_cast<int>(dest.atomDataPacked.size());
  const int instrOffset    = static_cast<int>(dest.queryInstructions.size());
  const int leafMaskOffset = static_cast<int>(dest.queryLeafMasks.size());

  dest.atomDataPacked.insert(dest.atomDataPacked.end(), src.atomDataPacked.begin(), src.atomDataPacked.end());
  dest.bondTypeCounts.insert(dest.bondTypeCounts.end(), src.bondTypeCounts.begin(), src.bondTypeCounts.end());
  dest.targetAtomBonds.insert(dest.targetAtomBonds.end(), src.targetAtomBonds.begin(), src.targetAtomBonds.end());
  dest.queryAtomBonds.insert(dest.queryAtomBonds.end(), src.queryAtomBonds.begin(), src.queryAtomBonds.end());

  dest.atomQueryMasks.insert(dest.atomQueryMasks.end(), src.atomQueryMasks.begin(), src.atomQueryMasks.end());
  dest.atomQueryTrees.insert(dest.atomQueryTrees.end(), src.atomQueryTrees.begin(), src.atomQueryTrees.end());
  dest.queryInstructions.insert(dest.queryInstructions.end(),
                                src.queryInstructions.begin(),
                                src.queryInstructions.end());
  dest.queryLeafMasks.insert(dest.queryLeafMasks.end(), src.queryLeafMasks.begin(), src.queryLeafMasks.end());
  dest.queryLeafBondCounts.insert(dest.queryLeafBondCounts.end(),
                                  src.queryLeafBondCounts.begin(),
                                  src.queryLeafBondCounts.end());

  for (int start : src.atomInstrStarts) {
    dest.atomInstrStarts.push_back(start + instrOffset);
  }
  for (int start : src.atomLeafMaskStarts) {
    dest.atomLeafMaskStarts.push_back(start + leafMaskOffset);
  }

  for (size_t i = 1; i < src.batchAtomStarts.size(); ++i) {
    dest.batchAtomStarts.push_back(src.batchAtomStarts[i] + atomOffset);
  }

  dest.recursivePatterns.insert(dest.recursivePatterns.end(),
                                src.recursivePatterns.begin(),
                                src.recursivePatterns.end());
}

void buildTargetBatchParallelInto(MoleculesHost&                          result,
                                  int                                     numThreads,
                                  const std::vector<const RDKit::ROMol*>& molecules,
                                  const std::vector<int>&                 sortOrder) {
  ScopedNvtxRange range("buildTargetBatchParallelInto");

  const int numMols = static_cast<int>(molecules.size());

  if (numMols == 0) {
    result.clear();
    return;
  }

  const bool useSortOrder = !sortOrder.empty();

  if (numThreads <= 1) {
    result.clear();
    for (int i = 0; i < numMols; ++i) {
      const int molIdx = useSortOrder ? sortOrder[i] : i;
      addToBatch(molecules[molIdx], result);
    }
    return;
  }

  // Compute per-molecule atom offsets for direct writing
  std::vector<int>& atomStarts = result.batchAtomStarts;
  atomStarts.resize(numMols + 1);
  atomStarts[0] = 0;
  for (int i = 0; i < numMols; ++i) {
    const int molIdx  = useSortOrder ? sortOrder[i] : i;
    atomStarts[i + 1] = atomStarts[i] + static_cast<int>(molecules[molIdx]->getNumAtoms());
  }
  const size_t totalAtoms = atomStarts[numMols];

  // Resize result vectors (reuses capacity if sufficient)
  result.atomDataPacked.resize(totalAtoms);
  result.bondTypeCounts.resize(totalAtoms);
  result.targetAtomBonds.resize(totalAtoms);

  // Direct parallel write - each thread writes to its molecules' positions in result
#pragma omp parallel num_threads(numThreads)
  {
    ScopedNvtxRange threadRange("Preprocess direct write");

#pragma omp for schedule(static)
    for (int i = 0; i < numMols; ++i) {
      const int           molIdx     = useSortOrder ? sortOrder[i] : i;
      const RDKit::ROMol* mol        = molecules[molIdx];
      const int           atomOffset = atomStarts[i];
      const auto*         ringInfo   = mol->getRingInfo();

      int localAtomIdx = 0;
      for (const RDKit::Atom* atom : mol->atoms()) {
        const int destIdx = atomOffset + localAtomIdx;

        AtomDataPacked&  packed     = result.atomDataPacked[destIdx];
        BondTypeCounts&  bondCounts = result.bondTypeCounts[destIdx];
        TargetAtomBonds& tab        = result.targetAtomBonds[destIdx];

        packed     = AtomDataPacked{};
        bondCounts = BondTypeCounts{};
        tab        = TargetAtomBonds{};

        populateAtomScalars(atom, packed, ringInfo);

        const unsigned int atomIdx            = atom->getIdx();
        int                ringBondCount      = 0;
        int                numHeteroNeighbors = 0;
        int                totalBonds         = 0;
        tab.degree                            = 0;

        auto [beg, bondEnd] = mol->getAtomBonds(atom);
        while (beg != bondEnd) {
          const auto*        bond        = (*mol)[*beg];
          const unsigned int bondIdx     = bond->getIdx();
          const int          bondType    = bond->getBondType();
          const int          otherAtomId = bond->getOtherAtomIdx(atomIdx);
          const bool         isInRing    = ringInfo->numBondRings(bondIdx) > 0;

          incrementBondTypeCount(bondCounts, bondType);
          ringBondCount += isInRing;

          const int neighborAtomicNum = mol->getAtomWithIdx(otherAtomId)->getAtomicNum();
          numHeteroNeighbors += (neighborAtomicNum != 6 && neighborAtomicNum != 1);

          if (tab.degree < kMaxBondsPerAtom) {
            tab.neighborIdx[tab.degree] = static_cast<uint8_t>(otherAtomId);
            tab.bondInfo[tab.degree]    = packTargetBondInfo(bondType, isInRing);
            ++tab.degree;
          }
          ++totalBonds;
          ++beg;
        }

        if (totalBonds > kMaxBondsPerAtom) {
          throw std::runtime_error("Atom has more than " + std::to_string(kMaxBondsPerAtom) + " bonds");
        }
        if (ringBondCount > AtomDataPacked::kMax4BitValue) {
          throw std::runtime_error("Ring bond count exceeds maximum");
        }
        if (numHeteroNeighbors > AtomDataPacked::kMax4BitValue) {
          throw std::runtime_error("Heteroatom neighbor count exceeds maximum");
        }

        packed.setRingBondCount(ringBondCount);
        packed.setNumHeteroatomNeighbors(numHeteroNeighbors);

        ++localAtomIdx;
      }
    }
  }
}

MoleculesHost buildQueryBatchParallel(const std::vector<const RDKit::ROMol*>& molecules,
                                      const std::vector<int>&                 sortOrder,
                                      int                                     numThreads) {
  ScopedNvtxRange range("buildQueryBatchParallel");

  const int numMols = static_cast<int>(molecules.size());
  if (numMols == 0) {
    return MoleculesHost();
  }

  const bool useSortOrder = !sortOrder.empty();

  if (numThreads <= 1) {
    MoleculesHost batch;
    for (int i = 0; i < numMols; ++i) {
      const int molIdx = useSortOrder ? sortOrder[i] : i;
      addQueryToBatch(molecules[molIdx], batch);
    }
    return batch;
  }

  // Compute total atoms to estimate per-thread capacity
  size_t totalAtoms = 0;
  for (int i = 0; i < numMols; ++i) {
    totalAtoms += molecules[i]->getNumAtoms();
  }
  const size_t atomsPerThread = (totalAtoms + numThreads - 1) / numThreads;
  const size_t molsPerThread  = (numMols + numThreads - 1) / numThreads;

  std::vector<MoleculesHost> threadBatches(numThreads);

  // Pre-reserve to avoid allocator contention during parallel phase
  for (int t = 0; t < numThreads; ++t) {
    threadBatches[t].reserve(molsPerThread, atomsPerThread);
  }

  // addQueryToBatch validates each query and throws on unsupported inputs (e.g.
  // disconnected SMARTS). An exception escaping the OpenMP region would call
  // std::terminate, so capture the first failure and rethrow after the region
  // joins.
  detail::OpenMPExceptionRegistry exceptionRegistry;

#pragma omp parallel num_threads(numThreads)
  {
    const int       tid = omp_get_thread_num();
    ScopedNvtxRange threadRange("Query preprocess thread " + std::to_string(tid));
    MoleculesHost&  localBatch = threadBatches[tid];

#pragma omp for schedule(static)
    for (int i = 0; i < numMols; ++i) {
      try {
        const int molIdx = useSortOrder ? sortOrder[i] : i;
        addQueryToBatch(molecules[molIdx], localBatch);
      } catch (...) {
        exceptionRegistry.store(std::current_exception());
      }
    }
  }
  exceptionRegistry.rethrow();

  MoleculesHost result;
  {
    ScopedNvtxRange mergeRange("Merge thread batches");
    for (int t = 0; t < numThreads; ++t) {
      mergeBatch(result, threadBatches[t]);
    }
  }

  return result;
}

int getQueryPipelineDepth(const MoleculesHost& queriesHost, int queryIdx) {
  if (queryIdx >= static_cast<int>(queriesHost.recursivePatterns.size())) {
    return 0;
  }
  const auto& recursiveInfo = queriesHost.recursivePatterns[queryIdx];
  if (recursiveInfo.empty()) {
    return 0;
  }
  return recursiveInfo.maxDepth + 1;
}

bool requiresRDKitFallback(const RDKit::ROMol* mol) {
  const auto* ringInfo = mol->getRingInfo();
  if (!ringInfo->isSymmSssr()) {
    throw std::runtime_error("Molecule ring info not initialized - call RDKit::MolOps::symmetrizeSSSR first");
  }

  for (const RDKit::Atom* atom : mol->atoms()) {
    const int idx = atom->getIdx();

    if (atom->getDegree() > kMaxBondsPerAtom) {
      return true;
    }

    if (ringInfo->numAtomRings(idx) > AtomDataPacked::kMax4BitValue) {
      return true;
    }

    if (atom->getNumImplicitHs() > AtomDataPacked::kMax4BitValue) {
      return true;
    }

    int ringBondCount      = 0;
    int numHeteroNeighbors = 0;
    auto [beg, bondEnd]    = mol->getAtomBonds(atom);
    while (beg != bondEnd) {
      const auto* bond = (*mol)[*beg];
      if (ringInfo->numBondRings(bond->getIdx()) > 0) {
        ++ringBondCount;
      }
      const int otherAtomIdx      = bond->getOtherAtomIdx(idx);
      const int neighborAtomicNum = mol->getAtomWithIdx(otherAtomIdx)->getAtomicNum();
      if (neighborAtomicNum != 6 && neighborAtomicNum != 1) {
        ++numHeteroNeighbors;
      }
      ++beg;
    }
    if (ringBondCount > AtomDataPacked::kMax4BitValue) {
      return true;
    }
    if (numHeteroNeighbors > AtomDataPacked::kMax4BitValue) {
      return true;
    }
  }

  return false;
}

}  // namespace nvMolKit
