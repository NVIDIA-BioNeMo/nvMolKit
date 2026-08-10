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

#include <GraphMol/FMCS/FMCS.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "src/mcs/mcs_search.h"
#include "src/testutils/mol_data.h"
#include "tests/test_utils.h"

namespace {

using nvMolKit::MCSAtomCompare;
using nvMolKit::MCSBondCompare;
using nvMolKit::MCSPair;
using nvMolKit::MCSParameters;
using nvMolKit::MCSResultSource;
using nvMolKit::testing::readSmilesFileWithStrings;

constexpr size_t       kMaxAtoms          = 24;
constexpr size_t       kMaxBonds          = 128;
constexpr unsigned int kMaxDegree         = 8;
constexpr size_t       kDatasetCandidates = 1000;
constexpr size_t       kNumMolecules      = 160;
constexpr size_t       kNumPairs          = 300;
constexpr size_t       kPairingStrides[]  = {17, 53};
constexpr size_t       kNumPairingStrides = sizeof(kPairingStrides) / sizeof(kPairingStrides[0]);

struct RingConfig {
  bool        atomRingMatchesRingOnly = false;
  bool        bondRingMatchesRingOnly = false;
  const char* name                    = "NoRing";
};

using FmcsIntegrationParams = std::tuple<MCSAtomCompare, MCSBondCompare, RingConfig>;

const char* atomCompareName(MCSAtomCompare compare) {
  switch (compare) {
    case MCSAtomCompare::Any:
      return "AtomAny";
    case MCSAtomCompare::Elements:
      return "AtomElements";
    case MCSAtomCompare::Isotopes:
      return "AtomIsotopes";
    case MCSAtomCompare::AnyHeavyAtom:
      return "AtomAnyHeavy";
  }
  return "AtomUnknown";
}

const char* bondCompareName(MCSBondCompare compare) {
  switch (compare) {
    case MCSBondCompare::Any:
      return "BondAny";
    case MCSBondCompare::Order:
      return "BondOrder";
    case MCSBondCompare::OrderExact:
      return "BondOrderExact";
  }
  return "BondUnknown";
}

MCSParameters makeParams(MCSAtomCompare atomCompare, MCSBondCompare bondCompare, RingConfig ringConfig) {
  MCSParameters params;
  params.requireGpu                                = true;
  params.atomCompare                               = atomCompare;
  params.bondCompare                               = bondCompare;
  params.atomCompareParameters.ringMatchesRingOnly = ringConfig.atomRingMatchesRingOnly;
  params.bondCompareParameters.ringMatchesRingOnly = ringConfig.bondRingMatchesRingOnly;
  return params;
}

RDKit::MCSParameters makeRdkitParams(const MCSParameters& params) {
  RDKit::MCSParameters rdParams;
  rdParams.MaximizeBonds                             = params.maximizeBonds;
  rdParams.Timeout                                   = params.timeoutSeconds;
  rdParams.AtomCompareParameters.MatchValences       = params.atomCompareParameters.matchValences;
  rdParams.AtomCompareParameters.MatchFormalCharge   = params.atomCompareParameters.matchFormalCharge;
  rdParams.AtomCompareParameters.RingMatchesRingOnly = params.atomCompareParameters.ringMatchesRingOnly;
  rdParams.AtomCompareParameters.MatchIsotope        = params.atomCompareParameters.matchIsotope;
  rdParams.BondCompareParameters.RingMatchesRingOnly = params.bondCompareParameters.ringMatchesRingOnly;

  switch (params.atomCompare) {
    case MCSAtomCompare::Any:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareAny);
      break;
    case MCSAtomCompare::Elements:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareElements);
      break;
    case MCSAtomCompare::Isotopes:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareIsotopes);
      break;
    case MCSAtomCompare::AnyHeavyAtom:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareAnyHeavyAtom);
      break;
  }

  switch (params.bondCompare) {
    case MCSBondCompare::Any:
      rdParams.setMCSBondTyperFromEnum(RDKit::BondCompareAny);
      break;
    case MCSBondCompare::Order:
      rdParams.setMCSBondTyperFromEnum(RDKit::BondCompareOrder);
      break;
    case MCSBondCompare::OrderExact:
      rdParams.setMCSBondTyperFromEnum(RDKit::BondCompareOrderExact);
      break;
  }

  return rdParams;
}

RDKit::MCSResult findRdkitMCS(const RDKit::ROMol& molA, const RDKit::ROMol& molB, const MCSParameters& params) {
  std::vector<RDKit::ROMOL_SPTR> mols;
  mols.emplace_back(new RDKit::ROMol(molA));
  mols.emplace_back(new RDKit::ROMol(molB));
  auto rdParams = makeRdkitParams(params);
  return RDKit::findMCS(mols, &rdParams);
}

bool mappingHasUniqueAtoms(const std::vector<std::pair<int, int>>& mapping) {
  std::set<int> atomsA;
  std::set<int> atomsB;
  for (const auto& [a, b] : mapping) {
    if (!atomsA.insert(a).second || !atomsB.insert(b).second) {
      return false;
    }
  }
  return true;
}

bool mappingMatchesRdkitMCS(const std::vector<std::pair<int, int>>& gpuMapping,
                            const RDKit::ROMol&                     molA,
                            const RDKit::ROMol&                     molB,
                            const RDKit::MCSResult&                 rdkitMCS) {
  if (gpuMapping.empty()) {
    return rdkitMCS.NumAtoms == 0;
  }
  if (rdkitMCS.QueryMol == nullptr || gpuMapping.size() != rdkitMCS.NumAtoms || !mappingHasUniqueAtoms(gpuMapping)) {
    return false;
  }

  RDKit::SubstructMatchParameters params;
  params.uniquify     = false;
  params.maxMatches   = 0;
  const auto matchesA = RDKit::SubstructMatch(molA, *rdkitMCS.QueryMol, params);
  const auto matchesB = RDKit::SubstructMatch(molB, *rdkitMCS.QueryMol, params);

  for (const auto& matchA : matchesA) {
    std::vector<int> atomAToQuery(molA.getNumAtoms(), -1);
    for (const auto& [queryAtomIdx, targetAtomIdx] : matchA) {
      atomAToQuery[static_cast<size_t>(targetAtomIdx)] = queryAtomIdx;
    }

    std::vector<int> requiredB(rdkitMCS.QueryMol->getNumAtoms(), -1);
    bool             compatibleWithA = true;
    for (const auto& [gpuAtomA, gpuAtomB] : gpuMapping) {
      if (gpuAtomA < 0 || gpuAtomA >= static_cast<int>(atomAToQuery.size())) {
        compatibleWithA = false;
        break;
      }
      const int queryIdx = atomAToQuery[static_cast<size_t>(gpuAtomA)];
      if (queryIdx < 0) {
        compatibleWithA = false;
        break;
      }
      if (requiredB[static_cast<size_t>(queryIdx)] >= 0 && requiredB[static_cast<size_t>(queryIdx)] != gpuAtomB) {
        compatibleWithA = false;
        break;
      }
      requiredB[static_cast<size_t>(queryIdx)] = gpuAtomB;
    }
    if (!compatibleWithA) {
      continue;
    }

    for (const auto& matchB : matchesB) {
      bool compatibleWithB = true;
      for (const auto& [queryAtomIdx, targetAtomIdx] : matchB) {
        if (requiredB[static_cast<size_t>(queryAtomIdx)] >= 0 &&
            requiredB[static_cast<size_t>(queryAtomIdx)] != targetAtomIdx) {
          compatibleWithB = false;
          break;
        }
      }
      if (compatibleWithB) {
        return true;
      }
    }
  }

  return false;
}

bool atomsCompatible(const RDKit::ROMol&  molA,
                     const RDKit::Atom&   atomA,
                     const RDKit::ROMol&  molB,
                     const RDKit::Atom&   atomB,
                     const MCSParameters& params) {
  switch (params.atomCompare) {
    case MCSAtomCompare::Any:
      break;
    case MCSAtomCompare::Elements:
      if (atomA.getAtomicNum() != atomB.getAtomicNum())
        return false;
      break;
    case MCSAtomCompare::Isotopes:
      if (atomA.getIsotope() != atomB.getIsotope())
        return false;
      break;
    case MCSAtomCompare::AnyHeavyAtom: {
      const bool heavyA = atomA.getAtomicNum() != 1;
      const bool heavyB = atomB.getAtomicNum() != 1;
      if (heavyA != heavyB)
        return false;
      break;
    }
  }

  if (params.atomCompareParameters.matchIsotope && atomA.getIsotope() != atomB.getIsotope())
    return false;
  if (params.atomCompareParameters.matchValences && atomA.getTotalValence() != atomB.getTotalValence())
    return false;
  if (params.atomCompareParameters.matchFormalCharge && atomA.getFormalCharge() != atomB.getFormalCharge()) {
    return false;
  }
  if (params.atomCompareParameters.ringMatchesRingOnly) {
    const bool ringA = molA.getRingInfo()->numAtomRings(atomA.getIdx()) > 0;
    const bool ringB = molB.getRingInfo()->numAtomRings(atomB.getIdx()) > 0;
    if (ringA != ringB)
      return false;
  }
  return true;
}

int mappedBondOrderClass(const RDKit::Bond& bond, const MCSParameters& params) {
  switch (params.bondCompare) {
    case MCSBondCompare::Any:
      return 0;
    case MCSBondCompare::Order: {
      const auto bondType = static_cast<int>(bond.getBondType());
      if (bondType == static_cast<int>(RDKit::Bond::SINGLE) || bondType == static_cast<int>(RDKit::Bond::AROMATIC)) {
        return 1;
      }
      return bondType;
    }
    case MCSBondCompare::OrderExact: {
      const auto bondType = static_cast<int>(bond.getBondType());
      if (bondType == static_cast<int>(RDKit::Bond::ONEANDAHALF) ||
          bondType == static_cast<int>(RDKit::Bond::AROMATIC)) {
        return static_cast<int>(RDKit::Bond::AROMATIC);
      }
      return bondType;
    }
  }
  return 0;
}

bool bondsCompatible(const RDKit::ROMol&  molA,
                     const RDKit::Bond&   bondA,
                     const RDKit::ROMol&  molB,
                     const RDKit::Bond&   bondB,
                     const MCSParameters& params) {
  if (mappedBondOrderClass(bondA, params) != mappedBondOrderClass(bondB, params))
    return false;
  if (params.bondCompareParameters.ringMatchesRingOnly) {
    const bool ringA = molA.getRingInfo()->numBondRings(bondA.getIdx()) > 0;
    const bool ringB = molB.getRingInfo()->numBondRings(bondB.getIdx()) > 0;
    if (ringA != ringB)
      return false;
  }
  return true;
}

bool mappingIsValidCommonSubgraph(const nvMolKit::MCSResult& gpu,
                                  const RDKit::ROMol&        molA,
                                  const RDKit::ROMol&        molB,
                                  const MCSParameters&       params) {
  if (gpu.atomMapping.size() != gpu.numAtoms || gpu.bondMapping.size() != gpu.numBonds)
    return false;
  if (!mappingHasUniqueAtoms(gpu.atomMapping))
    return false;

  std::vector<int> atomAToB(molA.getNumAtoms(), -1);
  std::vector<int> atomAToMappingPos(molA.getNumAtoms(), -1);
  for (size_t pos = 0; pos < gpu.atomMapping.size(); ++pos) {
    const auto [atomAIdx, atomBIdx] = gpu.atomMapping[pos];
    if (atomAIdx < 0 || atomAIdx >= static_cast<int>(molA.getNumAtoms()) || atomBIdx < 0 ||
        atomBIdx >= static_cast<int>(molB.getNumAtoms())) {
      return false;
    }
    const auto* atomA = molA.getAtomWithIdx(static_cast<unsigned int>(atomAIdx));
    const auto* atomB = molB.getAtomWithIdx(static_cast<unsigned int>(atomBIdx));
    if (!atomsCompatible(molA, *atomA, molB, *atomB, params))
      return false;
    atomAToB[static_cast<size_t>(atomAIdx)]          = atomBIdx;
    atomAToMappingPos[static_cast<size_t>(atomAIdx)] = static_cast<int>(pos);
  }

  std::set<int>                 bondsA;
  std::set<int>                 bondsB;
  std::vector<std::vector<int>> adjacency(gpu.atomMapping.size());
  for (const auto& [bondAIdx, bondBIdx] : gpu.bondMapping) {
    if (bondAIdx < 0 || bondAIdx >= static_cast<int>(molA.getNumBonds()) || bondBIdx < 0 ||
        bondBIdx >= static_cast<int>(molB.getNumBonds())) {
      return false;
    }
    if (!bondsA.insert(bondAIdx).second || !bondsB.insert(bondBIdx).second)
      return false;

    const auto* bondA = molA.getBondWithIdx(static_cast<unsigned int>(bondAIdx));
    const auto* bondB = molB.getBondWithIdx(static_cast<unsigned int>(bondBIdx));
    if (!bondsCompatible(molA, *bondA, molB, *bondB, params))
      return false;

    const int aBegin = static_cast<int>(bondA->getBeginAtomIdx());
    const int aEnd   = static_cast<int>(bondA->getEndAtomIdx());
    const int bBegin = static_cast<int>(bondB->getBeginAtomIdx());
    const int bEnd   = static_cast<int>(bondB->getEndAtomIdx());
    if (aBegin < 0 || aBegin >= static_cast<int>(atomAToB.size()) || aEnd < 0 ||
        aEnd >= static_cast<int>(atomAToB.size())) {
      return false;
    }
    const bool sameOrientation =
      atomAToB[static_cast<size_t>(aBegin)] == bBegin && atomAToB[static_cast<size_t>(aEnd)] == bEnd;
    const bool oppositeOrientation =
      atomAToB[static_cast<size_t>(aBegin)] == bEnd && atomAToB[static_cast<size_t>(aEnd)] == bBegin;
    if (!sameOrientation && !oppositeOrientation)
      return false;

    const int beginPos = atomAToMappingPos[static_cast<size_t>(aBegin)];
    const int endPos   = atomAToMappingPos[static_cast<size_t>(aEnd)];
    if (beginPos < 0 || endPos < 0)
      return false;
    adjacency[static_cast<size_t>(beginPos)].push_back(endPos);
    adjacency[static_cast<size_t>(endPos)].push_back(beginPos);
  }

  if (gpu.numAtoms <= 1)
    return gpu.numBonds == 0;
  if (gpu.numBonds == 0)
    return false;

  std::vector<char> visited(gpu.atomMapping.size(), 0);
  std::vector<int>  stack = {0};
  visited[0]              = 1;
  size_t seen             = 0;
  while (!stack.empty()) {
    const int cur = stack.back();
    stack.pop_back();
    ++seen;
    for (const int next : adjacency[static_cast<size_t>(cur)]) {
      if (!visited[static_cast<size_t>(next)]) {
        visited[static_cast<size_t>(next)] = 1;
        stack.push_back(next);
      }
    }
  }
  return seen == gpu.atomMapping.size();
}

struct Dataset {
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  std::vector<std::string>                   smiles;
};

Dataset loadDataset() {
  const std::string smilesPath = getTestDataFolderPath() + "/chembl_1k.smi";
  if (!std::filesystem::exists(smilesPath)) {
    throw std::runtime_error("SMILES file not found: " + smilesPath);
  }
  auto [candidateMols, candidateSmiles] = readSmilesFileWithStrings(smilesPath, kDatasetCandidates, kMaxAtoms);

  Dataset data;
  data.mols.reserve(kNumMolecules);
  data.smiles.reserve(kNumMolecules);
  for (size_t i = 0; i < candidateMols.size() && data.mols.size() < kNumMolecules; ++i) {
    const auto& mol             = candidateMols[i];
    bool        degreeSupported = true;
    for (const auto* atom : mol->atoms()) {
      if (atom->getDegree() > kMaxDegree) {
        degreeSupported = false;
        break;
      }
    }
    if (mol->getNumBonds() <= kMaxBonds && degreeSupported) {
      data.mols.push_back(std::move(candidateMols[i]));
      data.smiles.push_back(std::move(candidateSmiles[i]));
    }
  }
  if (data.mols.empty()) {
    throw std::runtime_error("No molecules loaded from " + smilesPath);
  }
  return data;
}

const Dataset& dataset() {
  static const Dataset data = loadDataset();
  return data;
}

std::vector<const RDKit::ROMol*> moleculeTable(const Dataset& data, size_t maxMols = 0) {
  const size_t                     count = maxMols == 0 ? data.mols.size() : std::min(maxMols, data.mols.size());
  std::vector<const RDKit::ROMol*> mols;
  mols.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    mols.push_back(data.mols[i].get());
  }
  return mols;
}

void expectSameResultShape(const std::vector<nvMolKit::MCSResult>& expected,
                           const std::vector<nvMolKit::MCSResult>& actual) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i].numAtoms, expected[i].numAtoms) << "result " << i;
    EXPECT_EQ(actual[i].numBonds, expected[i].numBonds) << "result " << i;
    EXPECT_EQ(actual[i].source, expected[i].source) << "result " << i;
  }
}

class FMCSIntegrationTest : public ::testing::TestWithParam<FmcsIntegrationParams> {};

TEST_P(FMCSIntegrationTest, ChemblPairsMatchRDKitExactly) {
  const auto& data = dataset();

  const auto atomCompare = std::get<0>(GetParam());
  const auto bondCompare = std::get<1>(GetParam());
  const auto ringConfig  = std::get<2>(GetParam());
  const auto params      = makeParams(atomCompare, bondCompare, ringConfig);

  ASSERT_EQ(data.mols.size(), kNumMolecules);

  const auto           mols = moleculeTable(data);
  std::vector<MCSPair> pairs;
  pairs.reserve(kNumPairs);
  for (size_t i = 0; i < kNumPairs; ++i) {
    const size_t idxA   = i % data.mols.size();
    const size_t cycle  = i / data.mols.size();
    const size_t stride = kPairingStrides[cycle % kNumPairingStrides];
    const size_t idxB   = (idxA + stride) % data.mols.size();
    pairs.emplace_back(idxA, idxB);
  }

  auto gpuResults = nvMolKit::findMCSBatch(mols, pairs, nullptr, params);
  ASSERT_EQ(gpuResults.size(), pairs.size());

  for (size_t i = 0; i < pairs.size(); ++i) {
    const auto [idxA, idxB] = pairs[i];
    SCOPED_TRACE("pair=" + std::to_string(i) + " a=" + std::to_string(idxA) + " b=" + std::to_string(idxB) + " A=" +
                 data.smiles[idxA] + " B=" + data.smiles[idxB] + " atomCompare=" + atomCompareName(atomCompare) +
                 " bondCompare=" + bondCompareName(bondCompare) + " ringConfig=" + ringConfig.name);
    const auto& molA = *data.mols[idxA];
    const auto& molB = *data.mols[idxB];
    const auto  rd   = findRdkitMCS(molA, molB, params);
    const auto& gpu  = gpuResults[i];

    EXPECT_EQ(gpu.source, MCSResultSource::GPU);
    EXPECT_FALSE(gpu.canceled);
    EXPECT_EQ(gpu.numAtoms, rd.NumAtoms);
    EXPECT_EQ(gpu.numBonds, rd.NumBonds);
    const bool sizeMatches    = gpu.numAtoms == rd.NumAtoms && gpu.numBonds == rd.NumBonds;
    const bool mappingMatches = sizeMatches ? mappingMatchesRdkitMCS(gpu.atomMapping, molA, molB, rd) : false;
    const bool mappingValid   = sizeMatches ? mappingIsValidCommonSubgraph(gpu, molA, molB, params) : false;
    EXPECT_TRUE(mappingMatches || mappingValid);
  }
}

TEST(FMCSIntegration, NoCompatibleBondReturnsSingleAtomLikeRDKit) {
  auto molA = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol("CC"));
  auto molB = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol("C=C"));
  ASSERT_NE(molA, nullptr);
  ASSERT_NE(molB, nullptr);

  const auto params = makeParams(MCSAtomCompare::Elements, MCSBondCompare::Order, RingConfig{});
  const std::vector<const RDKit::ROMol*> mols{molA.get(), molB.get()};
  const std::vector<MCSPair>             pairs{
                {0, 1}
  };

  const auto gpuResults = nvMolKit::findMCSBatch(mols, pairs, nullptr, params);
  const auto rd         = findRdkitMCS(*molA, *molB, params);
  ASSERT_EQ(gpuResults.size(), 1u);
  const auto& gpu = gpuResults.front();
  EXPECT_EQ(gpu.source, MCSResultSource::GPU);
  EXPECT_EQ(gpu.numAtoms, rd.NumAtoms);
  EXPECT_EQ(gpu.numBonds, rd.NumBonds);
  EXPECT_TRUE(mappingMatchesRdkitMCS(gpu.atomMapping, *molA, *molB, rd));
}

TEST(FMCSIntegration, CuratedHigherTierChemblPairsMatchRDKitExactly) {
  struct TierCase {
    const char* largeSmiles;
    const char* partnerSmiles;
    size_t      lowerTierBound;
    size_t      upperTierBound;
    bool        largeFirst;
  };

  // Sparse ChEMBL molecules exercise the higher dispatch tiers without making
  // exact RDKit comparison prohibitively expensive for a PR integration test.
  // clang-format off
  const TierCase cases[] = {
    {"CCCCCCCCCCCCCC(=O)NCc1ccc(C(=O)N[C@H](C(=O)O)[C@@H](C)CC)cc1",
     "CCO", 32, 64, true},
    {"C#C/C=C\\CCCCCCCCCCCCCC/C=C\\CCCCC(O)/C=C/CCCC#C[C@H](O)C#CCCCCCC/C=C/[C@@H](O)C#C",
     "CCO", 32, 64, false},
    {"NCCCC[C@@H](C=O)NC(=O)CC(CCCN)NC(=O)CC(CCCN)NC(=O)CC(CCCN)NC(=O)CC(CCCN)NC(=O)CC(CCCN)NC(=O)CC(CCCN)"
     "NC(=O)CC(CCCN)NC(=O)CC(N)CCCN",
     "CCN", 64, 128, true},
    {"CCCCCCCCCCCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO",
     "CCO", 64, 128, false},
  };
  // clang-format on

  std::vector<std::unique_ptr<RDKit::ROMol>> ownedMols;
  std::vector<const RDKit::ROMol*>           mols;
  std::vector<MCSPair>                       pairs;
  ownedMols.reserve(2 * std::size(cases));
  mols.reserve(2 * std::size(cases));
  pairs.reserve(std::size(cases));

  for (const auto& testCase : cases) {
    auto large   = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(testCase.largeSmiles));
    auto partner = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(testCase.partnerSmiles));
    ASSERT_NE(large, nullptr) << testCase.largeSmiles;
    ASSERT_NE(partner, nullptr) << testCase.partnerSmiles;

    const size_t graphSize = std::max<size_t>(large->getNumAtoms(), large->getNumBonds());
    EXPECT_GT(graphSize, testCase.lowerTierBound);
    EXPECT_LE(graphSize, testCase.upperTierBound);

    const size_t largeIdx   = mols.size();
    const size_t partnerIdx = largeIdx + 1;
    mols.push_back(large.get());
    mols.push_back(partner.get());
    ownedMols.push_back(std::move(large));
    ownedMols.push_back(std::move(partner));
    pairs.emplace_back(testCase.largeFirst ? largeIdx : partnerIdx, testCase.largeFirst ? partnerIdx : largeIdx);
  }

  const auto params     = makeParams(MCSAtomCompare::Elements, MCSBondCompare::Order, RingConfig{});
  const auto gpuResults = nvMolKit::findMCSBatch(mols, pairs, nullptr, params);
  ASSERT_EQ(gpuResults.size(), pairs.size());

  for (size_t i = 0; i < pairs.size(); ++i) {
    const auto [idxA, idxB] = pairs[i];
    SCOPED_TRACE("higher-tier case=" + std::to_string(i));
    const auto& molA = *mols[idxA];
    const auto& molB = *mols[idxB];
    const auto  rd   = findRdkitMCS(molA, molB, params);
    const auto& gpu  = gpuResults[i];

    EXPECT_EQ(gpu.source, MCSResultSource::GPU);
    EXPECT_FALSE(gpu.canceled);
    EXPECT_EQ(gpu.numAtoms, rd.NumAtoms);
    EXPECT_EQ(gpu.numBonds, rd.NumBonds);
    const bool sizeMatches    = gpu.numAtoms == rd.NumAtoms && gpu.numBonds == rd.NumBonds;
    const bool mappingMatches = sizeMatches ? mappingMatchesRdkitMCS(gpu.atomMapping, molA, molB, rd) : false;
    const bool mappingValid   = sizeMatches ? mappingIsValidCommonSubgraph(gpu, molA, molB, params) : false;
    EXPECT_TRUE(mappingMatches || mappingValid);
  }
}

TEST(FMCSIntegration, TimeoutIsIsolatedPerPairWithinBatch) {
  // Both pairs dispatch together through tier 128. The easy pair must finish
  // even though the repetitive peptide pair exhausts its one-second budget.
  auto easyA = std::unique_ptr<RDKit::ROMol>(
    RDKit::SmilesToMol("CCCCCCCCCCCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO"));
  auto easyB = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol("CCO"));
  auto hardA = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(
    "C[C@H](NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](N)[C@@H](C)O)C(=O)N"
    "[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)"
    "C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](Cc1ccccc1)C(=O)O"));
  auto hardB = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(
    "CSCC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)CNC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](Cc1ccccc1)NC(=O)[C@@H](CCC(N)=O)NC(=O)"
    "[C@@H](CCC(N)=O)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](CCCCN)NC(=O)[C@H]1CCCN1C(=O)[C@H](N)CCCN=C(N)N)C(N)=O"));
  ASSERT_NE(easyA, nullptr);
  ASSERT_NE(easyB, nullptr);
  ASSERT_NE(hardA, nullptr);
  ASSERT_NE(hardB, nullptr);

  const auto graphSize = [](const RDKit::ROMol& mol) { return std::max<size_t>(mol.getNumAtoms(), mol.getNumBonds()); };
  const size_t easyPairSize = std::max(graphSize(*easyA), graphSize(*easyB));
  const size_t hardPairSize = std::max(graphSize(*hardA), graphSize(*hardB));
  ASSERT_GT(easyPairSize, 64u);
  ASSERT_LE(easyPairSize, 128u);
  ASSERT_GT(hardPairSize, 64u);
  ASSERT_LE(hardPairSize, 128u);

  const std::vector<const RDKit::ROMol*> mols{easyA.get(), easyB.get(), hardA.get(), hardB.get()};
  const std::vector<MCSPair>             pairs{
                {0, 1},
                {2, 3}
  };
  MCSParameters params;
  params.timeoutSeconds       = 1;
  params.requireGpu           = true;
  params.batchSize            = 2;
  params.workerThreads        = 1;
  params.preprocessingThreads = 1;
  params.executorsPerRunner   = 1;

  const auto results = nvMolKit::findMCSBatch(mols, pairs, nullptr, params);
  const auto rdEasy  = findRdkitMCS(*easyA, *easyB, params);
  ASSERT_EQ(results.size(), 2u);

  const auto& easy = results[0];
  EXPECT_EQ(easy.source, MCSResultSource::GPU);
  EXPECT_FALSE(easy.canceled);
  EXPECT_EQ(easy.numAtoms, rdEasy.NumAtoms);
  EXPECT_EQ(easy.numBonds, rdEasy.NumBonds);
  EXPECT_TRUE(mappingMatchesRdkitMCS(easy.atomMapping, *easyA, *easyB, rdEasy));

  const auto& hard = results[1];
  EXPECT_EQ(hard.source, MCSResultSource::GPU);
  EXPECT_TRUE(hard.canceled);
  EXPECT_GT(hard.numAtoms, 0u);
  EXPECT_GT(hard.numBonds, 0u);
}

TEST(FMCSIntegration, SpecialMoleculeSemanticsMatchRDKitExactly) {
  struct SpecialCase {
    const char*   name;
    const char*   smilesA;
    const char*   smilesB;
    MCSParameters params;
    unsigned int  expectedAtoms;
    unsigned int  expectedBonds;
  };

  const auto gpuParams = [] {
    MCSParameters params;
    params.requireGpu = true;
    return params;
  };

  std::vector<SpecialCase> cases;
  {
    auto params        = gpuParams();
    params.atomCompare = MCSAtomCompare::Isotopes;
    cases.push_back({"isotope-labelled atoms", "[13CH3]CO", "CCO", params, 2, 1});
  }
  {
    auto params                                    = gpuParams();
    params.atomCompareParameters.matchFormalCharge = true;
    cases.push_back({"formal charge", "C[NH2+]C", "CNC", params, 1, 0});
  }
  {
    auto params                                = gpuParams();
    params.atomCompareParameters.matchValences = true;
    cases.push_back({"different phosphorus valences", "P(C)(C)C", "P(C)(C)(C)(C)C", params, 1, 0});
  }
  cases.push_back({"disconnected salts", "CC.[Na+]", "CCC.[K+]", gpuParams(), 2, 1});
  cases.push_back({"aromatic and aliphatic rings with CompareOrder", "c1ccccc1", "C1CCCCC1", gpuParams(), 6, 6});
  {
    auto params        = gpuParams();
    params.bondCompare = MCSBondCompare::OrderExact;
    cases.push_back({"aromatic and aliphatic rings with CompareOrderExact", "c1ccccc1", "C1CCCCC1", params, 1, 0});
  }
  {
    auto params                                      = gpuParams();
    params.atomCompareParameters.ringMatchesRingOnly = true;
    cases.push_back({"ring atoms cannot match chain atoms", "C1CCCCC1", "CCCCCC", params, 0, 0});
  }
  {
    auto params                                      = gpuParams();
    params.bondCompareParameters.ringMatchesRingOnly = true;
    cases.push_back({"ring bonds cannot match chain bonds", "C1CCCCC1", "CCCCCC", params, 1, 0});
  }
  cases.push_back({"no compatible atom", "[Na+]", "[Cl-]", gpuParams(), 0, 0});
  cases.push_back({"empty molecule", "", "CC", gpuParams(), 0, 0});
  cases.push_back(
    {"opposite tetrahedral stereochemistry is ignored", "F[C@H](Cl)Br", "F[C@@H](Cl)Br", gpuParams(), 4, 3});
  for (const auto& testCase : cases) {
    SCOPED_TRACE(testCase.name);
    auto molA = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(testCase.smilesA));
    auto molB = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(testCase.smilesB));
    ASSERT_NE(molA, nullptr) << testCase.smilesA;
    ASSERT_NE(molB, nullptr) << testCase.smilesB;

    const std::vector<const RDKit::ROMol*> mols{molA.get(), molB.get()};
    const std::vector<MCSPair>             pairs{
                  {0, 1}
    };
    const auto rd      = findRdkitMCS(*molA, *molB, testCase.params);
    const auto results = nvMolKit::findMCSBatch(mols, pairs, nullptr, testCase.params);
    ASSERT_EQ(results.size(), 1u);
    const auto& result = results.front();

    EXPECT_EQ(rd.NumAtoms, testCase.expectedAtoms);
    EXPECT_EQ(rd.NumBonds, testCase.expectedBonds);
    EXPECT_EQ(result.numAtoms, rd.NumAtoms);
    EXPECT_EQ(result.numBonds, rd.NumBonds);
    EXPECT_EQ(result.source, MCSResultSource::GPU);
    EXPECT_FALSE(result.canceled);

    const bool sizeMatches    = result.numAtoms == rd.NumAtoms && result.numBonds == rd.NumBonds;
    const bool mappingMatches = sizeMatches ? mappingMatchesRdkitMCS(result.atomMapping, *molA, *molB, rd) : false;
    const bool mappingValid = sizeMatches ? mappingIsValidCommonSubgraph(result, *molA, *molB, testCase.params) : false;
    EXPECT_TRUE(mappingMatches || mappingValid);
  }
}

TEST(FMCSIntegration, UnsupportedBatchWideParametersThrowInsteadOfFallingBack) {
  struct UnsupportedCase {
    const char*   name;
    MCSParameters params;
  };

  std::vector<UnsupportedCase> cases;
  {
    MCSParameters params;
    params.storeAll = true;
    cases.push_back({"StoreAll", params});
  }
  {
    MCSParameters params;
    params.connectedOnly = false;
    cases.push_back({"disconnected MCS", params});
  }
  {
    MCSParameters params;
    params.maximizeBonds = false;
    cases.push_back({"maximize atoms", params});
  }
  {
    MCSParameters params;
    params.threshold = 0.5;
    cases.push_back({"Threshold", params});
  }
  {
    MCSParameters params;
    params.verbose = true;
    cases.push_back({"Verbose", params});
  }
  {
    MCSParameters params;
    params.initialSeed = "CC";
    cases.push_back({"initial SMARTS seed", params});
  }
  {
    MCSParameters params;
    params.atomCompare = MCSAtomCompare::AnyHeavyAtom;
    cases.push_back({"AnyHeavyAtom", params});
  }
  {
    MCSParameters params;
    params.atomCompareParameters.matchChiralTag = true;
    cases.push_back({"atom MatchChiralTag", params});
  }
  {
    MCSParameters params;
    params.atomCompareParameters.completeRingsOnly = true;
    cases.push_back({"atom CompleteRingsOnly", params});
  }
  {
    MCSParameters params;
    params.atomCompareParameters.matchIsotope = true;
    cases.push_back({"MatchIsotope flag", params});
  }
  {
    MCSParameters params;
    params.atomCompareParameters.maxDistance = 1.0;
    cases.push_back({"atom MaxDistance", params});
  }
  {
    MCSParameters params;
    params.bondCompareParameters.completeRingsOnly = true;
    cases.push_back({"bond CompleteRingsOnly", params});
  }
  {
    MCSParameters params;
    params.bondCompareParameters.matchFusedRings = true;
    cases.push_back({"MatchFusedRings", params});
  }
  {
    MCSParameters params;
    params.bondCompareParameters.matchFusedRingsStrict = true;
    cases.push_back({"MatchFusedRingsStrict", params});
  }
  {
    MCSParameters params;
    params.bondCompareParameters.matchStereo = true;
    cases.push_back({"bond MatchStereo", params});
  }
  auto molA = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol("CN"));
  auto molB = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol("CO"));
  ASSERT_NE(molA, nullptr);
  ASSERT_NE(molB, nullptr);
  const std::vector<const RDKit::ROMol*> mols{molA.get(), molB.get()};
  const std::vector<MCSPair>             pairs{
                {0, 1}
  };

  for (const auto& testCase : cases) {
    SCOPED_TRACE(testCase.name);
    EXPECT_THROW((void)nvMolKit::findMCSBatch(mols, pairs, nullptr, testCase.params), std::invalid_argument);
  }
}

TEST(FMCSDispatchRoutes, EqualListsWrapperMatchesExplicitPairs) {
  const auto& data = dataset();
  ASSERT_GE(data.mols.size(), 3);

  std::vector<const RDKit::ROMol*> molsA = {data.mols[0].get(), data.mols[1].get(), data.mols[2].get()};
  std::vector<const RDKit::ROMol*> molsB = {data.mols[1].get(), data.mols[2].get(), data.mols[0].get()};
  std::vector<const RDKit::ROMol*> combined;
  combined.reserve(molsA.size() + molsB.size());
  combined.insert(combined.end(), molsA.begin(), molsA.end());
  combined.insert(combined.end(), molsB.begin(), molsB.end());

  std::vector<MCSPair> pairs;
  pairs.reserve(molsA.size());
  for (size_t i = 0; i < molsA.size(); ++i) {
    pairs.emplace_back(i, molsA.size() + i);
  }

  MCSParameters params;
  const auto    explicitResults = nvMolKit::findMCSBatch(combined, pairs, nullptr, params);
  const auto    wrapperResults  = nvMolKit::findMCSBatch(molsA, molsB, nullptr, params);
  expectSameResultShape(explicitResults, wrapperResults);
}

TEST(FMCSDispatchRoutes, AllPairsUpperTriangleWrapperMatchesExplicitPairs) {
  const auto& data = dataset();
  ASSERT_GE(data.mols.size(), 3);

  const auto                 mols  = moleculeTable(data, 3);
  const std::vector<MCSPair> pairs = {
    {0, 0},
    {0, 1},
    {0, 2},
    {1, 1},
    {1, 2},
    {2, 2}
  };

  MCSParameters params;
  const auto    explicitResults = nvMolKit::findMCSBatch(mols, pairs, nullptr, params);
  const auto    wrapperResults  = nvMolKit::findMCSAllPairs(mols, nvMolKit::MCSAllPairsOptions{}, nullptr, params);
  expectSameResultShape(explicitResults, wrapperResults);
}

TEST(FMCSDispatchRoutes, ExternalStreamUsesConfiguredDevice) {
  int deviceCount = 0;
  ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
  if (deviceCount < 2) {
    GTEST_SKIP() << "Need at least 2 GPUs to test an external stream on a configured device";
  }

  constexpr int callerDevice     = 0;
  constexpr int configuredDevice = 1;
  ASSERT_EQ(cudaSetDevice(configuredDevice), cudaSuccess);
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaSetDevice(callerDevice), cudaSuccess);

  auto molA = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol("CCO"));
  auto molB = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol("CCN"));
  ASSERT_NE(molA, nullptr);
  ASSERT_NE(molB, nullptr);
  const std::vector<const RDKit::ROMol*> mols{molA.get(), molB.get()};
  const std::vector<MCSPair>             pairs{
                {0, 1}
  };

  MCSParameters params;
  params.requireGpu           = true;
  params.preprocessingThreads = 1;
  params.workerThreads        = 1;
  params.executorsPerRunner   = 1;
  params.gpuIds               = {configuredDevice};

  const auto results = nvMolKit::findMCSBatch(mols, pairs, stream, params);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].source, MCSResultSource::GPU);

  int currentDevice = -1;
  ASSERT_EQ(cudaGetDevice(&currentDevice), cudaSuccess);
  EXPECT_EQ(currentDevice, callerDevice);

  ASSERT_EQ(cudaSetDevice(configuredDevice), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  ASSERT_EQ(cudaSetDevice(callerDevice), cudaSuccess);
}

TEST(FMCSFallback, DegreeAboveEightUsesRDKit) {
  RDKit::SmilesParserParams parserParams;
  parserParams.sanitize = false;
  auto hypervalent = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol("[Fe](C)(C)(C)(C)(C)(C)(C)(C)C", parserParams));
  ASSERT_NE(hypervalent, nullptr);
  ASSERT_EQ(hypervalent->getAtomWithIdx(0)->getDegree(), 9u);

  const std::vector<const RDKit::ROMol*> mols{hypervalent.get()};
  const std::vector<MCSPair>             pairs{
                {0, 0}
  };
  MCSParameters params;
  params.preprocessingThreads = 1;
  params.workerThreads        = 1;
  params.executorsPerRunner   = 1;

  const auto results = nvMolKit::findMCSBatch(mols, pairs, nullptr, params);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].source, MCSResultSource::RDKitFallback);
}

TEST(FMCSFallback, DegreeAboveEightRejectedWhenGpuRequired) {
  RDKit::SmilesParserParams parserParams;
  parserParams.sanitize = false;
  auto hypervalent = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol("[Fe](C)(C)(C)(C)(C)(C)(C)(C)C", parserParams));
  ASSERT_NE(hypervalent, nullptr);

  const std::vector<const RDKit::ROMol*> mols{hypervalent.get()};
  const std::vector<MCSPair>             pairs{
                {0, 0}
  };
  MCSParameters params;
  params.requireGpu           = true;
  params.preprocessingThreads = 1;
  params.workerThreads        = 1;
  params.executorsPerRunner   = 1;

  EXPECT_THROW((void)nvMolKit::findMCSBatch(mols, pairs, nullptr, params), std::runtime_error);
}

constexpr RingConfig kNoRing{false, false, "NoRing"};

std::string integrationParamName(const ::testing::TestParamInfo<FmcsIntegrationParams>& info) {
  return std::string(atomCompareName(std::get<0>(info.param))) + "_" + bondCompareName(std::get<1>(info.param)) + "_" +
         std::get<2>(info.param).name;
}

INSTANTIATE_TEST_SUITE_P(
  AtomBondCompareIntegration,
  FMCSIntegrationTest,
  ::testing::Combine(::testing::Values(MCSAtomCompare::Any, MCSAtomCompare::Elements, MCSAtomCompare::Isotopes),
                     ::testing::Values(MCSBondCompare::Any, MCSBondCompare::Order, MCSBondCompare::OrderExact),
                     ::testing::Values(kNoRing)),
  integrationParamName);

INSTANTIATE_TEST_SUITE_P(
  RingCompareIntegration,
  FMCSIntegrationTest,
  ::testing::Values(
    FmcsIntegrationParams{
      MCSAtomCompare::Elements,
      MCSBondCompare::Order,
      RingConfig{true, false, "AtomRing"}
},
    FmcsIntegrationParams{MCSAtomCompare::Elements, MCSBondCompare::Order, RingConfig{false, true, "BondRing"}},
    FmcsIntegrationParams{MCSAtomCompare::Elements, MCSBondCompare::Order, RingConfig{true, true, "AtomBondRing"}}),
  integrationParamName);

}  // namespace
