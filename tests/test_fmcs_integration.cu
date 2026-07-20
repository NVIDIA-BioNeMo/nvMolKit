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
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
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
using nvMolKit::testing::readSmilesFileWithStrings;

constexpr size_t       kMaxAtoms                = 24;
constexpr size_t       kDefaultDatasetMolecules = 48;
constexpr size_t       kDefaultRandomPairs      = 1;
constexpr unsigned int kDefaultSeed             = 1337;

struct RingConfig {
  bool        atomRingMatchesRingOnly = false;
  bool        bondRingMatchesRingOnly = false;
  const char* name                    = "NoRing";
};

using FmcsIntegrationParams = std::tuple<MCSAtomCompare, MCSBondCompare, RingConfig>;

unsigned int integrationSeed() {
  if (const char* seedStr = std::getenv("NVMOLKIT_MCS_TEST_SEED")) {
    try {
      return static_cast<unsigned int>(std::stoul(seedStr));
    } catch (const std::exception&) {
    }
  }
  return kDefaultSeed;
}

size_t envSize(const char* name, size_t defaultValue) {
  if (const char* value = std::getenv(name)) {
    try {
      return std::max<size_t>(1, std::stoull(value));
    } catch (const std::exception&) {
    }
  }
  return defaultValue;
}

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
  auto [mols, smiles] =
    readSmilesFileWithStrings(smilesPath, envSize("NVMOLKIT_MCS_TEST_MOLECULES", kDefaultDatasetMolecules), kMaxAtoms);
  if (mols.empty()) {
    throw std::runtime_error("No molecules loaded from " + smilesPath);
  }
  return Dataset{std::move(mols), std::move(smiles)};
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
    EXPECT_EQ(actual[i].usedGpu, expected[i].usedGpu) << "result " << i;
    EXPECT_EQ(actual[i].usedFallback, expected[i].usedFallback) << "result " << i;
    EXPECT_EQ(actual[i].overflowed, expected[i].overflowed) << "result " << i;
  }
}

class FMCSIntegrationTest : public ::testing::TestWithParam<FmcsIntegrationParams> {};

TEST_P(FMCSIntegrationTest, SeededChemblPairsMatchRDKit) {
  const auto& data = dataset();

  const auto atomCompare = std::get<0>(GetParam());
  const auto bondCompare = std::get<1>(GetParam());
  const auto ringConfig  = std::get<2>(GetParam());
  const auto params      = makeParams(atomCompare, bondCompare, ringConfig);

  const unsigned int seed =
    integrationSeed() ^ (static_cast<unsigned int>(atomCompare) << 8) ^ (static_cast<unsigned int>(bondCompare) << 16) ^
    (ringConfig.atomRingMatchesRingOnly ? 0x10000u : 0u) ^ (ringConfig.bondRingMatchesRingOnly ? 0x20000u : 0u);
  const size_t                       numPairs = envSize("NVMOLKIT_MCS_TEST_PAIRS", kDefaultRandomPairs);
  std::mt19937                       rng(seed);
  std::uniform_int_distribution<int> dist(0, static_cast<int>(data.mols.size() - 1));

  const auto           mols = moleculeTable(data);
  std::vector<int>     indicesA;
  std::vector<int>     indicesB;
  std::vector<MCSPair> pairs;
  indicesA.reserve(numPairs);
  indicesB.reserve(numPairs);
  pairs.reserve(numPairs);

  for (size_t i = 0; i < numPairs; ++i) {
    const int idxA = dist(rng);
    const int idxB = dist(rng);
    indicesA.push_back(idxA);
    indicesB.push_back(idxB);
    pairs.emplace_back(static_cast<size_t>(idxA), static_cast<size_t>(idxB));
  }

  auto gpuResults = nvMolKit::findMCSBatch(mols, pairs, nullptr, params);
  ASSERT_EQ(gpuResults.size(), numPairs);

  for (size_t i = 0; i < numPairs; ++i) {
    SCOPED_TRACE("pair=" + std::to_string(i) + " a=" + std::to_string(indicesA[i]) +
                 " b=" + std::to_string(indicesB[i]) + " A=" + data.smiles[static_cast<size_t>(indicesA[i])] +
                 " B=" + data.smiles[static_cast<size_t>(indicesB[i])] + " seed=" + std::to_string(seed) +
                 " atomCompare=" + atomCompareName(atomCompare) + " bondCompare=" + bondCompareName(bondCompare) +
                 " ringConfig=" + ringConfig.name);
    const auto& molA = *data.mols[static_cast<size_t>(indicesA[i])];
    const auto& molB = *data.mols[static_cast<size_t>(indicesB[i])];
    const auto  rd   = findRdkitMCS(molA, molB, params);
    const auto& gpu  = gpuResults[i];

    const bool sizeMatches    = gpu.numAtoms == rd.NumAtoms && gpu.numBonds == rd.NumBonds;
    const bool mappingMatches = sizeMatches ? mappingMatchesRdkitMCS(gpu.atomMapping, molA, molB, rd) : false;
    const bool mappingValid   = sizeMatches ? mappingIsValidCommonSubgraph(gpu, molA, molB, params) : false;
    EXPECT_TRUE(sizeMatches);
    EXPECT_TRUE(mappingMatches || mappingValid);
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
  EXPECT_TRUE(results[0].usedFallback);
  EXPECT_FALSE(results[0].usedGpu);
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
  AtomBondCompareSmoke,
  FMCSIntegrationTest,
  ::testing::Combine(::testing::Values(MCSAtomCompare::Any, MCSAtomCompare::Elements, MCSAtomCompare::Isotopes),
                     ::testing::Values(MCSBondCompare::Any, MCSBondCompare::Order, MCSBondCompare::OrderExact),
                     ::testing::Values(kNoRing)),
  integrationParamName);

INSTANTIATE_TEST_SUITE_P(
  RingCompareSmoke,
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
