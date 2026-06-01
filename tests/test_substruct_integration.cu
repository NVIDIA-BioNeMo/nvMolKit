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

#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <gtest/gtest.h>

#include <climits>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "src/substruct/graph_labeler.cuh"
#include "src/substruct/substruct_search.h"
#include "src/testutils/mol_data.h"
#include "src/testutils/substruct_validation.h"
#include "src/utils/device.h"
#include "tests/test_utils.h"

using nvMolKit::countCudaDevices;

using nvMolKit::algorithmName;
using nvMolKit::countSubstructMatches;
using nvMolKit::getSubstructMatches;
using nvMolKit::hasSubstructMatch;
using nvMolKit::HasSubstructMatchResults;
using nvMolKit::matchSetsEqual;
using nvMolKit::printValidationResultDetailed;
using nvMolKit::ScopedStream;
using nvMolKit::SubstructAlgorithm;
using nvMolKit::SubstructSearchConfig;
using nvMolKit::SubstructSearchResults;
using nvMolKit::validateAgainstRDKit;
using nvMolKit::testing::makeMolsView;
using nvMolKit::testing::readSmartsFileWithStrings;
using nvMolKit::testing::readSmilesFileWithStrings;

namespace {

constexpr size_t kMaxAtoms  = 128;
constexpr size_t kNumSmiles = 300;

std::unique_ptr<RDKit::ROMol> makeSmartsQuery(const std::string& smarts) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
  EXPECT_NE(mol, nullptr) << "Failed to parse SMARTS: " << smarts;
  return mol;
}

struct DatasetConfig {
  const char* smartsFile;
  const char* name;
  bool        uniquify = false;
};

struct ThreadingConfig {
  nvMolKit::SubstructSearchConfig config;
  const char*                     name;
};

std::vector<int> getAllGpuIds() {
  const int        numDevices = countCudaDevices();
  std::vector<int> ids;
  ids.reserve(numDevices);
  for (int i = 0; i < numDevices; ++i) {
    ids.push_back(i);
  }
  return ids;
}

// clang-format off
const ThreadingConfig kThreadingConfigs[] = {
  {nvMolKit::SubstructSearchConfig{.batchSize = 1024, .workerThreads = 1, .preprocessingThreads = 1, .executorsPerRunner = -1, .gpuIds = {}}, "SingleThreaded"},
  {nvMolKit::SubstructSearchConfig{.batchSize = 256,  .workerThreads = 2, .preprocessingThreads = 4, .executorsPerRunner = -1, .gpuIds = {}}, "MultiThreaded"},
  {nvMolKit::SubstructSearchConfig{.batchSize = 256,                                                  .executorsPerRunner = -1, .gpuIds = {}}, "Autoselect"},
};
// clang-format on

constexpr DatasetConfig kDatasets[] = {
  {"pwalters_alert_collection_supported.txt", "PwaltersAlertCollection"},
  {"openbabel_functional_groups_supported.txt", "OpenBabelFunctionalGroups"},
  {"BMS_2006_filter_supported.txt", "BMS2006Filter"},
  {"rdkit_fragment_descriptors_supported.txt", "RDKitFragmentDescriptors"},
  {"rdkit_tautomer_transforms_supported.txt", "RDKitTautomerTransforms"},
  {"rdkit_tautomer_transforms_supported.txt", "RDKitTautomerTransformsUniquified", /*uniquify=*/true},
  {"rdkit_torsionPreferences_v2_supported.txt", "RDKitTorsionPreferencesV2"},
  {"rdkit_torsionPreferences_smallrings_supported.txt", "RDKitTorsionPreferencesSmallRings"},
  {"rdkit_pattern_fingerprint_supported.txt", "RDKitPatternFingerprints"},
  {"rdkit_torsionPreferences_macrocycles_supported.txt", "RDKitTorsionPreferencesMacrocycles"},
  {"RLewis_smarts_supported.txt", "RLewisSMARTS"},
  {"wehi_pains_supported.txt", "WEHIPAINS"},
};

struct SmallestRepro {
  int t = -1;
  int q = -1;
};

struct SmallestRepros {
  SmallestRepro smallestSum;
  SmallestRepro smallestQ;
  SmallestRepro smallestT;

  bool allSame() const {
    return smallestSum.t == smallestQ.t && smallestSum.q == smallestQ.q && smallestSum.t == smallestT.t &&
           smallestSum.q == smallestT.q;
  }

  bool hasAny() const { return smallestSum.t >= 0; }
};

template <typename PairContainer, typename GetT, typename GetQ>
SmallestRepros findSmallestRepros(const PairContainer& pairs, GetT getT, GetQ getQ) {
  SmallestRepros result;
  int            minSum = INT_MAX;
  int            minQ   = INT_MAX;
  int            minT   = INT_MAX;

  for (const auto& pair : pairs) {
    const int t   = getT(pair);
    const int q   = getQ(pair);
    const int sum = t + q;

    if (sum < minSum) {
      minSum             = sum;
      result.smallestSum = {t, q};
    }
    if (q < minQ) {
      minQ             = q;
      result.smallestQ = {t, q};
    }
    if (t < minT) {
      minT             = t;
      result.smallestT = {t, q};
    }
  }
  return result;
}

void printMatches(const std::string& label, const std::vector<std::vector<int>>& matches) {
  std::cout << "    " << label << ": ";
  if (matches.empty()) {
    std::cout << "(none)\n";
    return;
  }
  std::cout << matches.size() << " match(es)\n";
  for (size_t i = 0; i < matches.size(); ++i) {
    std::cout << "      [" << i << "]: {";
    for (size_t j = 0; j < matches[i].size(); ++j) {
      if (j > 0)
        std::cout << ", ";
      std::cout << matches[i][j];
    }
    std::cout << "}\n";
  }
}

std::vector<std::vector<int>> extractGpuMatches(const SubstructSearchResults& results,
                                                int                           targetIdx,
                                                int                           queryIdx,
                                                int /* numQueryAtoms */) {
  return results.getMatches(targetIdx, queryIdx);
}

void printSmallestRepro(const char*                                       label,
                        const SmallestRepro&                              r,
                        const std::vector<std::string>&                   targetSmiles,
                        const std::vector<std::string>&                   querySmarts,
                        const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                        const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                        const SubstructSearchResults&                     gpuResults,
                        bool                                              uniquify) {
  std::cout << "  --- " << label << " (t=" << r.t << " q=" << r.q << " sum=" << (r.t + r.q) << ") ---\n";
  std::cout << "  Target[" << r.t << "]: " << targetSmiles[r.t] << "\n";
  std::cout << "  Query[" << r.q << "]:  " << querySmarts[r.q] << "\n";

  auto rdkitMatches = nvMolKit::getRDKitSubstructMatches(*targetMols[r.t], *queryMols[r.q], uniquify);
  printMatches("Expected (RDKit)", rdkitMatches);

  const int numQueryAtoms = static_cast<int>(queryMols[r.q]->getNumAtoms());
  auto      gpuMatches    = extractGpuMatches(gpuResults, r.t, r.q, numQueryAtoms);
  printMatches("Actual (GPU)", gpuMatches);

  std::cout << "\n";
}

void printSmallestRepros(const SmallestRepros&                             repros,
                         const std::vector<std::string>&                   targetSmiles,
                         const std::vector<std::string>&                   querySmarts,
                         const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                         const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                         const SubstructSearchResults&                     gpuResults,
                         const std::string&                                category,
                         bool                                              uniquify) {
  if (!repros.hasAny())
    return;

  std::cout << "\n=== Smallest " << category << " repros ===\n";

  if (repros.allSame()) {
    printSmallestRepro("smallest (all criteria)",
                       repros.smallestSum,
                       targetSmiles,
                       querySmarts,
                       targetMols,
                       queryMols,
                       gpuResults,
                       uniquify);
  } else {
    printSmallestRepro("smallest sum (t+q)",
                       repros.smallestSum,
                       targetSmiles,
                       querySmarts,
                       targetMols,
                       queryMols,
                       gpuResults,
                       uniquify);
    if (repros.smallestQ.t != repros.smallestSum.t || repros.smallestQ.q != repros.smallestSum.q) {
      printSmallestRepro("smallest q",
                         repros.smallestQ,
                         targetSmiles,
                         querySmarts,
                         targetMols,
                         queryMols,
                         gpuResults,
                         uniquify);
    }
    if (repros.smallestT.t != repros.smallestSum.t || repros.smallestT.q != repros.smallestSum.q) {
      if (repros.smallestT.t != repros.smallestQ.t || repros.smallestT.q != repros.smallestQ.q) {
        printSmallestRepro("smallest t",
                           repros.smallestT,
                           targetSmiles,
                           querySmarts,
                           targetMols,
                           queryMols,
                           gpuResults,
                           uniquify);
      }
    }
  }
}

}  // namespace

enum class SubstructMode {
  Matches,
  HasMatch,
  CountMatches
};

using SubstructParams = std::tuple<SubstructAlgorithm, DatasetConfig, ThreadingConfig, SubstructMode>;

class SubstructureIntegrationTest : public ::testing::TestWithParam<SubstructParams> {
 protected:
  ScopedStream stream_;
  std::string  testDataPath_;

  void SetUp() override { testDataPath_ = getTestDataFolderPath(); }

  SubstructAlgorithm     algorithm() const { return std::get<0>(GetParam()); }
  const DatasetConfig&   dataset() const { return std::get<1>(GetParam()); }
  const ThreadingConfig& threading() const { return std::get<2>(GetParam()); }
  SubstructMode          mode() const { return std::get<3>(GetParam()); }
};

const ThreadingConfig kMainTestThreadingConfigs[] = {
  kThreadingConfigs[0],  // SingleThreaded
  kThreadingConfigs[1],  // MultiThreaded
};

INSTANTIATE_TEST_SUITE_P(AllCombinations,
                         SubstructureIntegrationTest,
                         ::testing::Combine(::testing::Values(SubstructAlgorithm::GSI),
                                            ::testing::ValuesIn(kDatasets),
                                            ::testing::ValuesIn(kMainTestThreadingConfigs),
                                            ::testing::Values(SubstructMode::Matches)),
                         [](const ::testing::TestParamInfo<SubstructParams>& info) {
                           return std::string(algorithmName(std::get<0>(info.param))) + "_" +
                                  std::get<1>(info.param).name + "_" + std::get<2>(info.param).name;
                         });

INSTANTIATE_TEST_SUITE_P(ConfigOptionTests,
                         SubstructureIntegrationTest,
                         ::testing::Values(SubstructParams{SubstructAlgorithm::GSI,
                                                           kDatasets[0],
                                                           kThreadingConfigs[2],
                                                           SubstructMode::Matches},  // Autoselect
                                           SubstructParams{SubstructAlgorithm::GSI,
                                                           kDatasets[0],
                                                           kThreadingConfigs[0],
                                                           SubstructMode::HasMatch},  // SingleThreaded, bool
                                           SubstructParams{SubstructAlgorithm::GSI,
                                                           kDatasets[3],
                                                           kThreadingConfigs[2],
                                                           SubstructMode::CountMatches}),  // Autoselect, counts
                         [](const ::testing::TestParamInfo<SubstructParams>& info) {
                           const SubstructMode mode = std::get<3>(info.param);
                           const char*         modeSuffix =
                             (mode == SubstructMode::HasMatch) ?
                                       "HasSubstructMatch" :
                                       (mode == SubstructMode::CountMatches ? "CountSubstructMatches" : "Autoselect");
                           return std::string(algorithmName(std::get<0>(info.param))) + "_" +
                                  std::get<1>(info.param).name + "_" + std::get<2>(info.param).name + "_" + modeSuffix;
                         });

TEST_P(SubstructureIntegrationTest, ChemblVsSmarts) {
  const std::string smilesPath = testDataPath_ + "/chembl_1k.smi";
  const std::string smartsPath = testDataPath_ + "/SMARTS/" + dataset().smartsFile;

  ASSERT_TRUE(std::filesystem::exists(smilesPath)) << "SMILES file not found: " << smilesPath;
  ASSERT_TRUE(std::filesystem::exists(smartsPath)) << "SMARTS file not found: " << smartsPath;

  auto [targetMols, targetSmiles] = readSmilesFileWithStrings(smilesPath, kNumSmiles, kMaxAtoms);
  auto [queryMols, querySmarts]   = readSmartsFileWithStrings(smartsPath);

  ASSERT_FALSE(targetMols.empty()) << "No target molecules loaded";
  ASSERT_FALSE(queryMols.empty()) << "No query patterns loaded";

  ASSERT_LE(targetMols.size(), kNumSmiles) << "Loaded more targets than requested";

  const int numTargets = static_cast<int>(targetMols.size());
  const int numQueries = static_cast<int>(queryMols.size());
  const int numGpus    = threading().config.gpuIds.empty() ? 1 : static_cast<int>(threading().config.gpuIds.size());

  if (mode() == SubstructMode::HasMatch) {
    HasSubstructMatchResults boolResults;
    hasSubstructMatch(makeMolsView(targetMols),
                      makeMolsView(queryMols),
                      boolResults,
                      algorithm(),
                      stream_.stream(),
                      threading().config);

    EXPECT_EQ(boolResults.numTargets, numTargets);
    EXPECT_EQ(boolResults.numQueries, numQueries);

    int                              mismatches   = 0;
    int                              totalMatches = 0;
    std::vector<std::pair<int, int>> mismatchPairs;

    for (int t = 0; t < numTargets; ++t) {
      for (int q = 0; q < numQueries; ++q) {
        const bool           gpuHasMatch = boolResults.matches(t, q);
        RDKit::MatchVectType matchVect;
        const bool           rdkitHasMatch = RDKit::SubstructMatch(*targetMols[t], *queryMols[q], matchVect);

        if (gpuHasMatch)
          ++totalMatches;

        if (gpuHasMatch != rdkitHasMatch) {
          ++mismatches;
          if (mismatchPairs.size() < 10) {
            mismatchPairs.emplace_back(t, q);
          }
        }
      }
    }

    std::cout << "[" << algorithmName(algorithm()) << ", " << threading().name << ", HasSubstructMatch] Statistics:\n"
              << "  Threading: workerThreads=" << threading().config.workerThreads
              << ", preprocessingThreads=" << threading().config.preprocessingThreads << ", " << numGpus << " GPU(s)\n"
              << "  Total queries: " << numQueries << "\n"
              << "  Total targets: " << numTargets << "\n"
              << "  Total pairs with matches: " << totalMatches << "\n"
              << "  Mismatches: " << mismatches << "\n";

    if (!mismatchPairs.empty()) {
      std::cout << "  First mismatches:\n";
      for (const auto& [t, q] : mismatchPairs) {
        RDKit::MatchVectType matchVect;
        const bool           rdkitHasMatch = RDKit::SubstructMatch(*targetMols[t], *queryMols[q], matchVect);
        std::cout << "    T[" << t << "]=" << targetSmiles[t] << " Q[" << q << "]=" << querySmarts[q]
                  << " GPU=" << boolResults.matches(t, q) << " RDKit=" << rdkitHasMatch << "\n";
      }
    }

    EXPECT_EQ(mismatches, 0) << "HasSubstructMatch results do not match RDKit for algorithm "
                             << algorithmName(algorithm());
  } else if (mode() == SubstructMode::CountMatches) {
    auto config     = threading().config;
    config.uniquify = dataset().uniquify;

    std::vector<int> counts;
    countSubstructMatches(makeMolsView(targetMols),
                          makeMolsView(queryMols),
                          counts,
                          algorithm(),
                          stream_.stream(),
                          config);

    EXPECT_EQ(counts.size(), static_cast<size_t>(numTargets * numQueries));

    RDKit::SubstructMatchParameters params;
    params.uniquify   = dataset().uniquify;
    params.maxMatches = 0;

    int                              mismatches = 0;
    std::vector<std::pair<int, int>> mismatchPairs;
    for (int t = 0; t < numTargets; ++t) {
      for (int q = 0; q < numQueries; ++q) {
        const int  gpuCount     = counts[t * numQueries + q];
        const auto rdkitMatches = RDKit::SubstructMatch(*targetMols[t], *queryMols[q], params);
        const int  rdkitCount   = static_cast<int>(rdkitMatches.size());
        if (gpuCount != rdkitCount) {
          ++mismatches;
          if (mismatchPairs.size() < 10) {
            mismatchPairs.emplace_back(t, q);
          }
        }
      }
    }

    std::cout << "[" << algorithmName(algorithm()) << ", " << threading().name
              << ", CountSubstructMatches] Statistics:\n"
              << "  Threading: workerThreads=" << threading().config.workerThreads
              << ", preprocessingThreads=" << threading().config.preprocessingThreads << ", " << numGpus << " GPU(s)\n"
              << "  Total queries: " << numQueries << "\n"
              << "  Total targets: " << numTargets << "\n"
              << "  Mismatches: " << mismatches << "\n";

    if (!mismatchPairs.empty()) {
      std::cout << "  First mismatches:\n";
      for (const auto& [t, q] : mismatchPairs) {
        const auto rdkitMatches = RDKit::SubstructMatch(*targetMols[t], *queryMols[q], params);
        std::cout << "    T[" << t << "]=" << targetSmiles[t] << " Q[" << q << "]=" << querySmarts[q]
                  << " GPU=" << counts[t * numQueries + q] << " RDKit=" << rdkitMatches.size() << "\n";
      }
    }

    EXPECT_EQ(mismatches, 0) << "CountSubstructMatches results do not match RDKit for algorithm "
                             << algorithmName(algorithm());
  } else {
    auto config     = threading().config;
    config.uniquify = dataset().uniquify;

    SubstructSearchResults results;
    getSubstructMatches(makeMolsView(targetMols),
                        makeMolsView(queryMols),
                        results,
                        algorithm(),
                        stream_.stream(),
                        config);

    EXPECT_EQ(results.numTargets, numTargets);
    EXPECT_EQ(results.numQueries, numQueries);

    std::vector<int64_t> totalMatchesPerQuery(numQueries, 0);
    int64_t              grandTotalMatches = 0;

    for (int q = 0; q < numQueries; ++q) {
      for (int t = 0; t < numTargets; ++t) {
        totalMatchesPerQuery[q] += results.matchCount(t, q);
      }
      grandTotalMatches += totalMatchesPerQuery[q];
    }

    std::vector<int> zeroMatchQueries;
    for (int q = 0; q < numQueries; ++q) {
      if (totalMatchesPerQuery[q] == 0) {
        zeroMatchQueries.push_back(q);
      }
    }

    std::cout << "[" << algorithmName(algorithm()) << ", " << threading().name << "] Query statistics:\n"
              << "  Threading: workerThreads=" << threading().config.workerThreads
              << ", preprocessingThreads=" << threading().config.preprocessingThreads << ", " << numGpus << " GPU(s)\n"
              << "  Total queries: " << numQueries << "\n"
              << "  Total targets: " << numTargets << "\n"
              << "  Grand total matches: " << grandTotalMatches << "\n"
              << "  Queries with 0 matches: " << zeroMatchQueries.size() << "\n";

    if (!zeroMatchQueries.empty()) {
      std::cout << "  Zero-match queries:\n";
      const size_t maxToShow = 20;
      for (size_t i = 0; i < std::min(zeroMatchQueries.size(), maxToShow); ++i) {
        const int q = zeroMatchQueries[i];
        std::cout << "    [" << q << "]: " << querySmarts[q] << "\n";
      }
      if (zeroMatchQueries.size() > maxToShow) {
        std::cout << "    ... and " << (zeroMatchQueries.size() - maxToShow) << " more\n";
      }
    }

    auto validationResult = validateAgainstRDKit(results, targetMols, queryMols, dataset().uniquify);

    if (!validationResult.allMatch) {
      printValidationResultDetailed(validationResult,
                                    results,
                                    targetMols,
                                    queryMols,
                                    targetSmiles,
                                    querySmarts,
                                    algorithmName(algorithm()),
                                    5,
                                    dataset().uniquify);

      if (!validationResult.mismatches.empty()) {
        auto repros = findSmallestRepros(
          validationResult.mismatches,
          [](const auto& m) { return std::get<0>(m); },
          [](const auto& m) { return std::get<1>(m); });
        printSmallestRepros(repros,
                            targetSmiles,
                            querySmarts,
                            targetMols,
                            queryMols,
                            results,
                            "count mismatch",
                            dataset().uniquify);
      }

      if (!validationResult.mappingMismatches.empty()) {
        auto repros = findSmallestRepros(
          validationResult.mappingMismatches,
          [](const auto& m) { return m.first; },
          [](const auto& m) { return m.second; });
        printSmallestRepros(repros,
                            targetSmiles,
                            querySmarts,
                            targetMols,
                            queryMols,
                            results,
                            "mapping mismatch",
                            dataset().uniquify);
      }
    }

    EXPECT_TRUE(validationResult.allMatch)
      << "GPU results do not match RDKit for algorithm " << algorithmName(algorithm())
      << ". Count mismatches: " << validationResult.mismatchedPairs
      << ", Mapping mismatches: " << validationResult.wrongMappingPairs << " / " << validationResult.totalPairs
      << " total pairs";
  }
}

// =============================================================================
// Multi-GPU Tests
// =============================================================================

class MultiGpuSubstructTest : public ::testing::Test {
 protected:
  ScopedStream stream_;
  std::string  testDataPath_;

  void SetUp() override {
    testDataPath_        = getTestDataFolderPath();
    const int numDevices = countCudaDevices();
    if (numDevices < 2) {
      GTEST_SKIP() << "Multi-GPU test requires at least 2 GPUs, found " << numDevices;
    }
  }
};

TEST_F(MultiGpuSubstructTest, MultiGpuMatchesSingleGpu) {
  const std::string smilesPath = testDataPath_ + "/chembl_1k.smi";
  const std::string smartsPath = testDataPath_ + "/SMARTS/rdkit_fragment_descriptors_supported.txt";

  ASSERT_TRUE(std::filesystem::exists(smilesPath)) << "SMILES file not found: " << smilesPath;
  ASSERT_TRUE(std::filesystem::exists(smartsPath)) << "SMARTS file not found: " << smartsPath;

  auto [targetMols, targetSmiles] = readSmilesFileWithStrings(smilesPath, kNumSmiles, kMaxAtoms);
  auto [queryMols, querySmarts]   = readSmartsFileWithStrings(smartsPath);

  ASSERT_FALSE(targetMols.empty()) << "No target molecules loaded";
  ASSERT_FALSE(queryMols.empty()) << "No query patterns loaded";

  auto targetPtrs = makeMolsView(targetMols);
  auto queryPtrs  = makeMolsView(queryMols);

  // Run single-GPU
  SubstructSearchConfig singleGpuConfig;
  singleGpuConfig.batchSize     = 1024;
  singleGpuConfig.workerThreads = 2;

  SubstructSearchResults singleGpuResults;
  getSubstructMatches(targetPtrs,
                      queryPtrs,
                      singleGpuResults,
                      SubstructAlgorithm::GSI,
                      stream_.stream(),
                      singleGpuConfig);

  // Run multi-GPU
  SubstructSearchConfig multiGpuConfig;
  multiGpuConfig.batchSize     = 1024;
  multiGpuConfig.workerThreads = 2;
  multiGpuConfig.gpuIds        = getAllGpuIds();

  SubstructSearchResults multiGpuResults;
  getSubstructMatches(targetPtrs,
                      queryPtrs,
                      multiGpuResults,
                      SubstructAlgorithm::GSI,
                      stream_.stream(),
                      multiGpuConfig);

  const int numGpus = static_cast<int>(multiGpuConfig.gpuIds.size());
  std::cout << "[MultiGPU] Using " << numGpus << " GPUs with " << multiGpuConfig.workerThreads << " workers each\n";

  // Compare results
  EXPECT_EQ(singleGpuResults.numTargets, multiGpuResults.numTargets);
  EXPECT_EQ(singleGpuResults.numQueries, multiGpuResults.numQueries);

  int64_t singleGpuTotal    = 0;
  int64_t multiGpuTotal     = 0;
  int     mismatches        = 0;
  int     mappingMismatches = 0;

  for (int t = 0; t < singleGpuResults.numTargets; ++t) {
    for (int q = 0; q < singleGpuResults.numQueries; ++q) {
      const int singleCount = singleGpuResults.matchCount(t, q);
      const int multiCount  = multiGpuResults.matchCount(t, q);
      singleGpuTotal += singleCount;
      multiGpuTotal += multiCount;
      if (singleCount != multiCount) {
        ++mismatches;
      } else if (singleCount > 0) {
        const int  numQueryAtoms = static_cast<int>(queryMols[q]->getNumAtoms());
        const auto singleMatches = extractGpuMatches(singleGpuResults, t, q, numQueryAtoms);
        const auto multiMatches  = extractGpuMatches(multiGpuResults, t, q, numQueryAtoms);
        if (!matchSetsEqual(singleMatches, multiMatches)) {
          ++mappingMismatches;
        }
      }
    }
  }

  std::cout << "[MultiGPU] Single-GPU total matches: " << singleGpuTotal << "\n";
  std::cout << "[MultiGPU] Multi-GPU total matches: " << multiGpuTotal << "\n";
  std::cout << "[MultiGPU] Mismatched pairs: " << mismatches << "\n";
  std::cout << "[MultiGPU] Mapping mismatched pairs: " << mappingMismatches << "\n";

  EXPECT_EQ(singleGpuTotal, multiGpuTotal) << "Total match counts differ between single and multi-GPU";
  EXPECT_EQ(mismatches, 0) << "Some pairs have different match counts";
  EXPECT_EQ(mappingMismatches, 0) << "Some pairs have different match mappings";
}

// =============================================================================
// Recursive SMARTS Tests
// =============================================================================

TEST(RecursiveSmartsTest, HasRecursiveSmartsDetection) {
  auto nonRecursive = makeSmartsQuery("[CH3]");
  EXPECT_FALSE(nvMolKit::hasRecursiveSmarts(nonRecursive.get()));

  auto recursive = makeSmartsQuery("[$([OH])]");
  EXPECT_TRUE(nvMolKit::hasRecursiveSmarts(recursive.get()));
}

TEST(RecursiveSmartsTest, ExtractSimplePattern) {
  auto query = makeSmartsQuery("[$([OH])]");
  auto info  = nvMolKit::extractRecursivePatterns(query.get());

  EXPECT_EQ(info.size(), 1);
  EXPECT_TRUE(info.hasRecursivePatterns);
}

TEST(RecursiveSmartsTest, ExtractMultiplePatterns) {
  auto query = makeSmartsQuery("[$([OH]),$([NH2])]");
  auto info  = nvMolKit::extractRecursivePatterns(query.get());

  EXPECT_EQ(info.size(), 2);
}

TEST(RecursiveSmartsTest, PatternIdsAreSequential) {
  auto query = makeSmartsQuery("[$([C]),$([N]),$([O])]");
  auto info  = nvMolKit::extractRecursivePatterns(query.get());

  EXPECT_EQ(info.size(), 3);
  EXPECT_EQ(info.patterns[0].patternId, 0);
  EXPECT_EQ(info.patterns[1].patternId, 1);
  EXPECT_EQ(info.patterns[2].patternId, 2);
}

TEST(RecursiveSmartsTest, TooManyPatternsThrows) {
  std::string smarts = "[C";
  for (int i = 0; i < nvMolKit::RecursivePatternInfo::kMaxPatterns + 1; ++i) {
    smarts += ";$(*-N)";
  }
  smarts += "]";

  auto query = makeSmartsQuery(smarts);
  EXPECT_THROW(nvMolKit::extractRecursivePatterns(query.get()), std::runtime_error);
}
