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

#include <GraphMol/MolOps.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <future>
#include <latch>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/substruct/substruct_library.h"
#include "src/substruct/substruct_search.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"

namespace {

using MoleculeId = unsigned int;

std::unique_ptr<RDKit::ROMol> molFromSmiles(const std::string& smiles) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
  EXPECT_NE(mol, nullptr) << "Failed to parse SMILES: " << smiles;
  return mol;
}

std::unique_ptr<RDKit::ROMol> queryFromSmarts(const std::string& smarts) {
  auto query = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
  EXPECT_NE(query, nullptr) << "Failed to parse SMARTS: " << smarts;
  return query;
}

std::vector<MoleculeId> rdkitMatchingIds(const std::vector<std::unique_ptr<RDKit::ROMol>>& targets,
                                         const RDKit::ROMol&                               query) {
  std::vector<MoleculeId> expected;
  for (std::size_t targetIdx = 0; targetIdx < targets.size(); ++targetIdx) {
    RDKit::MatchVectType match;
    if (RDKit::SubstructMatch(*targets[targetIdx], query, match)) {
      expected.push_back(static_cast<MoleculeId>(targetIdx));
    }
  }
  return expected;
}

std::string linearAlkane(std::size_t numAtoms) {
  return std::string(numAtoms, 'C');
}

//! While alive, the current device's stream-ordered allocations come from a private pool that is capped at its
//! smallest size and already full, so they fail without taking memory from other processes sharing the GPU.
class FullMemoryPool {
 public:
  FullMemoryPool() {
    EXPECT_EQ(cudaGetDevice(&device_), cudaSuccess);
    EXPECT_EQ(cudaDeviceGetMemPool(&originalPool_, device_), cudaSuccess);
    cudaMemPoolProps props = {};
    props.allocType        = cudaMemAllocationTypePinned;
    props.location.type    = cudaMemLocationTypeDevice;
    props.location.id      = device_;
    props.maxSize          = 1;
    EXPECT_EQ(cudaMemPoolCreate(&pool_, &props), cudaSuccess);
    for (std::size_t chunk = std::size_t{1} << 20; chunk >= 256;) {
      void* allocation = nullptr;
      if (cudaMallocFromPoolAsync(&allocation, chunk, pool_, nullptr) == cudaSuccess) {
        filler_.push_back(allocation);
      } else {
        chunk /= 2;
      }
    }
    static_cast<void>(cudaGetLastError());
    EXPECT_EQ(cudaDeviceSetMemPool(device_, pool_), cudaSuccess);
  }
  FullMemoryPool(const FullMemoryPool&)            = delete;
  FullMemoryPool& operator=(const FullMemoryPool&) = delete;
  ~FullMemoryPool() {
    EXPECT_EQ(cudaDeviceSetMemPool(device_, originalPool_), cudaSuccess);
    for (void* allocation : filler_) {
      EXPECT_EQ(cudaFreeAsync(allocation, nullptr), cudaSuccess);
    }
    EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    EXPECT_EQ(cudaMemPoolDestroy(pool_), cudaSuccess);
  }

 private:
  int                device_       = 0;
  cudaMemPool_t      originalPool_ = nullptr;
  cudaMemPool_t      pool_         = nullptr;
  std::vector<void*> filler_;
};

std::unique_ptr<RDKit::ROMol> highDegreeMolecule() {
  RDKit::SmilesParserParams params;
  params.sanitize = false;
  auto mol        = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol("[Fe](C)(C)(C)(C)(C)(C)(C)(C)C", params));
  if (mol) {
    RDKit::MolOps::symmetrizeSSSR(*mol);
  }
  return mol;
}

TEST(SubstructLibraryState, RejectsQueriesBeforeFirstFinalize) {
  nvMolKit::SubstructLibrary library;
  auto                       target = molFromSmiles("CCO");
  auto                       query  = queryFromSmarts("CO");

  ASSERT_NE(target, nullptr);
  ASSERT_NE(query, nullptr);
  EXPECT_EQ(library.addMol(*target), 0U);
  EXPECT_EQ(library.size(), 0U);
  EXPECT_EQ(library.pendingSize(), 1U);

  EXPECT_THROW(static_cast<void>(library.getMatches(*query)), std::logic_error);
  EXPECT_THROW(static_cast<void>(library.countMatches(*query)), std::logic_error);
  EXPECT_THROW(static_cast<void>(library.hasMatch(*query)), std::logic_error);
}

TEST(SubstructLibraryState, FinalizedEmptyLibraryHasEmptyResults) {
  nvMolKit::SubstructLibrary library;
  auto                       query = queryFromSmarts("C");

  ASSERT_NE(query, nullptr);
  EXPECT_NO_THROW(library.finalize());
  EXPECT_EQ(library.size(), 0U);
  EXPECT_EQ(library.pendingSize(), 0U);
  EXPECT_TRUE(library.getMatches(*query).empty());
  EXPECT_EQ(library.countMatches(*query), 0U);
  EXPECT_FALSE(library.hasMatch(*query));

  EXPECT_NO_THROW(library.finalize());
  EXPECT_TRUE(library.getMatches(*query).empty());
}

TEST(SubstructLibraryState, AddedMoleculesBecomeSearchableAtFinalize) {
  nvMolKit::SubstructLibrary library;
  auto                       ethanol        = molFromSmiles("CCO");
  auto                       benzene        = molFromSmiles("c1ccccc1");
  auto                       phenol         = molFromSmiles("Oc1ccccc1");
  auto                       aromaticCarbon = queryFromSmarts("c");

  ASSERT_NE(ethanol, nullptr);
  ASSERT_NE(benzene, nullptr);
  ASSERT_NE(phenol, nullptr);
  ASSERT_NE(aromaticCarbon, nullptr);

  EXPECT_EQ(library.addMol(*ethanol), 0U);
  EXPECT_EQ(library.addMol(*benzene), 1U);
  library.finalize();
  EXPECT_EQ(library.size(), 2U);
  EXPECT_EQ(library.pendingSize(), 0U);
  EXPECT_EQ(library.getMatches(*aromaticCarbon), std::vector<MoleculeId>({1U}));

  // Bulk adds continue the ID sequence after single adds and vice versa.
  auto pyridine = molFromSmiles("c1ccncc1");
  auto methane  = molFromSmiles("C");
  ASSERT_NE(pyridine, nullptr);
  ASSERT_NE(methane, nullptr);
  EXPECT_EQ(library.addMol(*phenol), 2U);
  EXPECT_EQ(library.addMols({pyridine.get(), methane.get()}), std::vector<MoleculeId>({3U, 4U}));
  EXPECT_EQ(library.addMol(*benzene), 5U);
  EXPECT_EQ(library.size(), 2U);
  EXPECT_EQ(library.pendingSize(), 4U);

  EXPECT_EQ(library.getMatches(*aromaticCarbon), std::vector<MoleculeId>({1U}));
  EXPECT_EQ(library.countMatches(*aromaticCarbon), 1U);

  library.finalize();
  EXPECT_EQ(library.size(), 6U);
  EXPECT_EQ(library.pendingSize(), 0U);
  EXPECT_EQ(library.getMatches(*aromaticCarbon), std::vector<MoleculeId>({1U, 2U, 3U, 5U}));

  library.finalize();
  EXPECT_EQ(library.size(), 6U);
  EXPECT_EQ(library.getMatches(*aromaticCarbon), std::vector<MoleculeId>({1U, 2U, 3U, 5U}));
}

TEST(SubstructLibraryState, FailedIncrementalFinalizeKeepsPreviousMoleculesAndCanBeRetried) {
  nvMolKit::SubstructLibrary library;
  auto                       benzene        = molFromSmiles("c1ccccc1");
  auto                       phenol         = molFromSmiles("Oc1ccccc1");
  auto                       aromaticCarbon = queryFromSmarts("c");
  ASSERT_NE(benzene, nullptr);
  ASSERT_NE(phenol, nullptr);
  ASSERT_NE(aromaticCarbon, nullptr);

  library.addMol(*benzene);
  library.finalize();
  const std::size_t querySlots = library.maxConcurrentQueries();
  library.addMol(*phenol);

  {
    const FullMemoryPool fullPool;
    EXPECT_THROW(library.finalize(), nvMolKit::CudaBadReturnCode);
  }

  EXPECT_EQ(library.size(), 1U);
  EXPECT_EQ(library.pendingSize(), 1U);
  EXPECT_EQ(library.maxConcurrentQueries(), querySlots);
  EXPECT_EQ(library.getMatches(*aromaticCarbon), std::vector<MoleculeId>({0U}));

  library.finalize();
  EXPECT_EQ(library.size(), 2U);
  EXPECT_EQ(library.pendingSize(), 0U);
  EXPECT_EQ(library.getMatches(*aromaticCarbon), std::vector<MoleculeId>({0U, 1U}));
}

TEST(SubstructLibraryResults, PreservesInsertionOrderAndAppliesExactLimits) {
  nvMolKit::SubstructLibrary                 library;
  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  for (const auto& smiles : {"CC", "O", "CCC", "N", "c1ccccc1", "CO"}) {
    targets.push_back(molFromSmiles(smiles));
    ASSERT_NE(targets.back(), nullptr);
    EXPECT_EQ(library.addMol(*targets.back()), targets.size() - 1);
  }
  auto query = queryFromSmarts("[#6]");
  ASSERT_NE(query, nullptr);
  library.finalize();

  const std::vector<MoleculeId> allExpected{0U, 2U, 4U, 5U};
  EXPECT_EQ(library.getMatches(*query), allExpected);
  EXPECT_EQ(library.getMatches(*query, 0), std::vector<MoleculeId>{});
  EXPECT_EQ(library.getMatches(*query, 1), std::vector<MoleculeId>({0U}));
  EXPECT_EQ(library.getMatches(*query, 3), std::vector<MoleculeId>({0U, 2U, 4U}));
  EXPECT_EQ(library.getMatches(*query, 20), allExpected);
  EXPECT_EQ(library.countMatches(*query), allExpected.size());
  EXPECT_TRUE(library.hasMatch(*query));

  auto noMatch = queryFromSmarts("[Si]");
  ASSERT_NE(noMatch, nullptr);
  EXPECT_TRUE(library.getMatches(*noMatch).empty());
  EXPECT_EQ(library.countMatches(*noMatch), 0U);
  EXPECT_FALSE(library.hasMatch(*noMatch));
}

TEST(SubstructLibraryResults, MatchesRDKitAcrossRepresentativeQuerySemantics) {
  nvMolKit::SubstructLibrary                 library;
  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  for (const auto& smiles :
       {"CCO", "CC(=O)C", "c1ccccc1", "C1CCCCC1", "C[N+](C)(C)C", "CC(=O)[O-]", "[Na+].[Cl-]", "CCOC(=O)c1ccccc1O"}) {
    targets.push_back(molFromSmiles(smiles));
    ASSERT_NE(targets.back(), nullptr);
    EXPECT_EQ(library.addMol(*targets.back()), targets.size() - 1);
  }
  library.finalize();

  for (const auto& smarts :
       {"[#6]", "C=O", "c1ccccc1", "[R]", "[N+]", "[#8;H1]", "[$([CX3]=[OX1])]", "[Cl-]", "[Si]"}) {
    auto query = queryFromSmarts(smarts);
    ASSERT_NE(query, nullptr);
    const auto expected = rdkitMatchingIds(targets, *query);

    EXPECT_EQ(library.getMatches(*query), expected) << "SMARTS: " << smarts;
    EXPECT_EQ(library.countMatches(*query), expected.size()) << "SMARTS: " << smarts;
    EXPECT_EQ(library.hasMatch(*query), !expected.empty()) << "SMARTS: " << smarts;
  }
}

TEST(SubstructLibraryResults, ReusedWorkspaceDoesNotCarryMatchesBetweenQueries) {
  nvMolKit::SubstructLibrary                 library;
  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  for (const auto& smiles : {"CCO", "c1ccccc1", "[Na+].[Cl-]", "CCN", "ClCCl", "CC(=O)O"}) {
    targets.push_back(molFromSmiles(smiles));
    ASSERT_NE(targets.back(), nullptr);
    library.addMol(*targets.back());
  }
  library.finalize();

  // Alternate recursive and plain queries to check sequential workspace reuse.
  auto broadRecursive = queryFromSmarts("[$([#6])]");
  auto narrowPlain    = queryFromSmarts("[Cl-]");
  ASSERT_NE(broadRecursive, nullptr);
  ASSERT_NE(narrowPlain, nullptr);
  const auto expectedBroad  = rdkitMatchingIds(targets, *broadRecursive);
  const auto expectedNarrow = rdkitMatchingIds(targets, *narrowPlain);
  ASSERT_EQ(expectedNarrow, std::vector<MoleculeId>({2U}));
  for (std::size_t round = 0; round < 3; ++round) {
    EXPECT_EQ(library.getMatches(*broadRecursive), expectedBroad) << "round " << round;
    EXPECT_EQ(library.getMatches(*narrowPlain), expectedNarrow) << "round " << round;
  }
}

TEST(SubstructLibraryOwnership, DoesNotDependOnInputMoleculeLifetime) {
  nvMolKit::SubstructLibrary library;
  {
    auto temporary = molFromSmiles("CC(=O)Oc1ccccc1C(=O)O");
    ASSERT_NE(temporary, nullptr);
    EXPECT_EQ(library.addMol(*temporary), 0U);
  }

  auto query = queryFromSmarts("C(=O)O");
  ASSERT_NE(query, nullptr);
  library.finalize();
  EXPECT_EQ(library.getMatches(*query), std::vector<MoleculeId>({0U}));
}

std::vector<std::unique_ptr<RDKit::ROMol>> unpackableAndPackableTargets() {
  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  targets.push_back(molFromSmiles(linearAlkane(128)));  // Largest packable target.
  targets.push_back(molFromSmiles(linearAlkane(129)));  // Too many atoms.
  targets.push_back(highDegreeMolecule());              // Atom degree above the packed limit.
  targets.push_back(molFromSmiles("CCN"));
  targets.push_back(molFromSmiles("[NH3]->[Cu]"));  // Dative bond.
  targets.push_back(molFromSmiles("[300C]CN"));     // Isotope above 255.
  targets.push_back(molFromSmiles("c1ccccc1N"));
  return targets;
}

TEST(SubstructLibraryResults, RingQueriesMatchRDKitForTargetsWithoutRingInfo) {
  // Norbornane's bridgehead atoms are in two SSSR rings. Without ring info, it should match as RDKit's
  // SubstructLibrary matches it, after symmetrizing its rings.
  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  targets.push_back(molFromSmiles("C1CC2CCC1C2"));
  targets.push_back(molFromSmiles("c1ccccc1CC"));
  for (const auto& target : targets) {
    ASSERT_NE(target, nullptr);
  }
  targets[0]->getRingInfo()->reset();

  nvMolKit::SubstructLibrary library;
  for (const auto& target : targets) {
    library.addMol(*target);
  }
  library.finalize();

  std::vector<std::unique_ptr<RDKit::ROMol>> symmetrized;
  for (const auto& target : targets) {
    symmetrized.push_back(std::make_unique<RDKit::ROMol>(*target));
    RDKit::MolOps::symmetrizeSSSR(*symmetrized.back());
  }
  for (const char* smarts : {"[R2]", "[r5]", "[R1]", "[x3]", "C1CCCC1", "[!R]"}) {
    auto query = queryFromSmarts(smarts);
    ASSERT_NE(query, nullptr);
    EXPECT_EQ(library.getMatches(*query), rdkitMatchingIds(symmetrized, *query)) << smarts;
  }
}

TEST(SubstructLibraryFallback, MatchesUnpackableTargetsFromEitherAddPathWithRDKit) {
  const auto                       targets = unpackableAndPackableTargets();
  std::vector<const RDKit::ROMol*> targetPtrs;
  for (const auto& target : targets) {
    ASSERT_NE(target, nullptr);
    targetPtrs.push_back(target.get());
  }
  ASSERT_GT(targets[2]->getAtomWithIdx(0)->getDegree(), 8U);

  nvMolKit::SubstructLibrary bulkLibrary;
  EXPECT_EQ(bulkLibrary.addMols(targetPtrs), std::vector<MoleculeId>({0U, 1U, 2U, 3U, 4U, 5U, 6U}));
  bulkLibrary.finalize();

  nvMolKit::SubstructLibrary singleLibrary;
  for (const auto* target : targetPtrs) {
    singleLibrary.addMol(*target);
  }
  singleLibrary.finalize();

  for (const char* smarts : {"CCCC", "[Fe]", "[#7]", "[Cu]", "[#6]-[#7]", "c"}) {
    auto query = queryFromSmarts(smarts);
    ASSERT_NE(query, nullptr);
    const auto expected = rdkitMatchingIds(targets, *query);
    for (auto* library : {&bulkLibrary, &singleLibrary}) {
      EXPECT_EQ(library->getMatches(*query), expected) << smarts;
      EXPECT_EQ(library->countMatches(*query), expected.size()) << smarts;
      EXPECT_EQ(library->hasMatch(*query), !expected.empty()) << smarts;
    }
  }
}

TEST(SubstructLibraryFallback, ResultLimitsMergeGpuAndFallbackMatchesInIdOrder) {
  // GPU and RDKit-fallback matches alternate by ID.
  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  for (const char* smiles : {"[NH3]->[Cu]", "CCN", "[300C]CN", "CC", "NCCN", "[NH3]->[Zn]", "c1ccncc1"}) {
    targets.push_back(molFromSmiles(smiles));
    ASSERT_NE(targets.back(), nullptr);
  }
  nvMolKit::SubstructLibrary library;
  for (const auto& target : targets) {
    library.addMol(*target);
  }
  library.finalize();

  auto nitrogen = queryFromSmarts("[#7]");
  ASSERT_NE(nitrogen, nullptr);
  const auto expected = rdkitMatchingIds(targets, *nitrogen);
  ASSERT_EQ(expected, std::vector<MoleculeId>({0U, 1U, 2U, 4U, 5U, 6U}));
  for (int limit = 1; limit <= static_cast<int>(expected.size()) + 1; ++limit) {
    const auto prefix =
      std::vector<MoleculeId>(expected.begin(), expected.begin() + std::min<std::size_t>(limit, expected.size()));
    EXPECT_EQ(library.getMatches(*nitrogen, limit), prefix) << "limit " << limit;
  }
  EXPECT_EQ(library.countMatches(*nitrogen), expected.size());
}

TEST(SubstructLibraryFallback, LibraryOfOnlyFallbackTargetsAnswersQueries) {
  nvMolKit::SubstructLibrary                 library;
  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  for (const char* smiles : {"[NH3]->[Cu]", "[300C]CN", "[NH3]->[Zn]"}) {
    targets.push_back(molFromSmiles(smiles));
    ASSERT_NE(targets.back(), nullptr);
    library.addMol(*targets.back());
  }
  library.finalize();

  for (const char* smarts : {"[#7]", "[Cu]", "[Si]"}) {
    auto query = queryFromSmarts(smarts);
    ASSERT_NE(query, nullptr);
    const auto expected = rdkitMatchingIds(targets, *query);
    EXPECT_EQ(library.getMatches(*query), expected) << smarts;
    EXPECT_EQ(library.getMatches(*query, 1),
              std::vector<MoleculeId>(expected.begin(), expected.begin() + std::min<std::size_t>(1, expected.size())))
      << smarts;
    EXPECT_EQ(library.countMatches(*query), expected.size()) << smarts;
    EXPECT_EQ(library.hasMatch(*query), !expected.empty()) << smarts;
  }
}

TEST(SubstructLibraryConfiguration, SupportsProductionBackendsAndRejectsInvalidConstruction) {
  nvMolKit::SubstructSearchConfig vf2Config;
  vf2Config.algorithm = nvMolKit::SubstructAlgorithm::VF2;
  EXPECT_THROW(nvMolKit::SubstructLibrary{vf2Config}, std::invalid_argument);

  nvMolKit::SubstructSearchConfig duplicateGpuConfig;
  duplicateGpuConfig.gpuIds = {0, 0};
  EXPECT_THROW(nvMolKit::SubstructLibrary{duplicateGpuConfig}, std::invalid_argument);

  nvMolKit::SubstructSearchConfig invalidGpuConfig;
  invalidGpuConfig.gpuIds = {std::numeric_limits<int>::max()};
  nvMolKit::SubstructLibrary invalidGpuLibrary(invalidGpuConfig);
  EXPECT_THROW(invalidGpuLibrary.finalize(), std::invalid_argument);

  auto target = molFromSmiles("CCOC(=O)C");
  auto query  = queryFromSmarts("C=O");
  ASSERT_NE(target, nullptr);
  ASSERT_NE(query, nullptr);

  for (const auto algorithm : {nvMolKit::SubstructAlgorithm::GSI, nvMolKit::SubstructAlgorithm::DFS}) {
    nvMolKit::SubstructSearchConfig config;
    config.algorithm            = algorithm;
    config.workerThreads        = 1;
    config.preprocessingThreads = 1;
    nvMolKit::SubstructLibrary library(config);
    EXPECT_EQ(library.addMol(*target), 0U);
    library.finalize();
    EXPECT_EQ(library.getMatches(*query), std::vector<MoleculeId>({0U}));
  }
}

TEST(SubstructLibraryMultiGpu, ShardsTargetsAndMergesEveryOperationInInsertionOrder) {
  int deviceCount = 0;
  ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
  if (deviceCount < 2) {
    GTEST_SKIP() << "Multi-GPU library test requires at least two CUDA devices";
  }

  nvMolKit::SubstructSearchConfig config;
  config.gpuIds               = {0, 1};
  config.workerThreads        = 1;
  config.preprocessingThreads = 4;
  nvMolKit::SubstructLibrary                 library(config);
  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  std::vector<const RDKit::ROMol*>           pointers;
  for (const auto& smiles : {"CC", "O", "CCC", "N", "c1ccccc1", "CO", "C=O", "[Si]"}) {
    targets.push_back(molFromSmiles(smiles));
    ASSERT_NE(targets.back(), nullptr);
    pointers.push_back(targets.back().get());
  }
  EXPECT_EQ(library.addMols(pointers), std::vector<MoleculeId>({0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U}));
  library.finalize();

  auto carbon = queryFromSmarts("[#6]");
  ASSERT_NE(carbon, nullptr);
  const std::vector<MoleculeId> expected{0U, 2U, 4U, 5U, 6U};
  EXPECT_EQ(library.getMatches(*carbon), expected);
  EXPECT_EQ(library.getMatches(*carbon, 3), std::vector<MoleculeId>({0U, 2U, 4U}));
  EXPECT_EQ(library.countMatches(*carbon), expected.size());
  EXPECT_TRUE(library.hasMatch(*carbon));

  auto phosphorus = queryFromSmarts("[P]");
  ASSERT_NE(phosphorus, nullptr);
  EXPECT_TRUE(library.getMatches(*phosphorus).empty());
  EXPECT_EQ(library.countMatches(*phosphorus), 0U);
  EXPECT_FALSE(library.hasMatch(*phosphorus));
}

TEST(SubstructLibraryConcurrency, MoreQueriesThanSlotsAllComplete) {
  nvMolKit::SubstructLibrary                 library;
  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  for (const auto& smiles : {"CCO", "c1ccccc1", "CC(=O)O", "N", "OCCN"}) {
    targets.push_back(molFromSmiles(smiles));
    ASSERT_NE(targets.back(), nullptr);
    library.addMol(*targets.back());
  }
  library.finalize();

  auto recursive = queryFromSmarts("[$([#6]O)]");
  auto plain     = queryFromSmarts("N");
  ASSERT_NE(recursive, nullptr);
  ASSERT_NE(plain, nullptr);
  const auto expectedRecursive = rdkitMatchingIds(targets, *recursive);
  const auto expectedPlain     = rdkitMatchingIds(targets, *plain);

  // More callers than query slots, half of them needing recursive scratch.
  std::vector<std::future<std::vector<MoleculeId>>> recursiveResults;
  std::vector<std::future<std::vector<MoleculeId>>> plainResults;
  for (std::size_t caller = 0; caller < 2 * library.maxConcurrentQueries() + 2; ++caller) {
    recursiveResults.push_back(std::async(std::launch::async, [&] { return library.getMatches(*recursive); }));
    plainResults.push_back(std::async(std::launch::async, [&] { return library.getMatches(*plain); }));
  }
  for (auto& result : recursiveResults) {
    EXPECT_EQ(result.get(), expectedRecursive);
  }
  for (auto& result : plainResults) {
    EXPECT_EQ(result.get(), expectedPlain);
  }
}

TEST(SubstructLibraryConcurrency, QueriesSeeOnlyFinalizedMoleculesWhileMoreAreAdded) {
  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  for (const char* smiles : {"CCO", "c1ccccc1", "CCN", "OCCO", "c1ccncc1", "CC(=O)O", "CCCl", "Oc1ccccc1"}) {
    targets.push_back(molFromSmiles(smiles));
    ASSERT_NE(targets.back(), nullptr);
  }
  auto oxygen = queryFromSmarts("[#8]");
  ASSERT_NE(oxygen, nullptr);
  // A query sees the molecules of some finalize() call, never a partial set.
  const std::vector<std::size_t>       finalizedSizes{2, 4, 6, 8};
  std::vector<std::vector<MoleculeId>> finalizedResults;
  for (const std::size_t size : finalizedSizes) {
    std::vector<std::unique_ptr<RDKit::ROMol>> prefix;
    for (std::size_t index = 0; index < size; ++index) {
      prefix.push_back(std::make_unique<RDKit::ROMol>(*targets[index]));
    }
    finalizedResults.push_back(rdkitMatchingIds(prefix, *oxygen));
  }

  nvMolKit::SubstructLibrary library;
  library.addMols({targets[0].get(), targets[1].get()});
  library.finalize();

  std::atomic<bool>             done{false};
  std::latch                    readersReady{3};
  std::vector<std::future<int>> readers;
  for (int reader = 0; reader < 3; ++reader) {
    readers.push_back(std::async(std::launch::async, [&] {
      int queries = 0;
      try {
        while (!done.load()) {
          const auto matches = library.getMatches(*oxygen);
          EXPECT_NE(std::find(finalizedResults.begin(), finalizedResults.end(), matches), finalizedResults.end());
          if (++queries == 1) {
            readersReady.count_down();
          }
        }
      } catch (...) {
        // Let the writer finish so reader.get() can report a failed first query.
        if (queries == 0) {
          readersReady.count_down();
        }
        throw;
      }
      return queries;
    }));
  }
  readersReady.wait();
  for (std::size_t round = 1; round < finalizedSizes.size(); ++round) {
    for (std::size_t index = finalizedSizes[round - 1]; index < finalizedSizes[round]; ++index) {
      library.addMol(*targets[index]);
    }
    library.finalize();
  }
  done.store(true);
  for (auto& reader : readers) {
    EXPECT_GT(reader.get(), 0);
  }
  EXPECT_EQ(library.getMatches(*oxygen), finalizedResults.back());
}

TEST(SubstructLibraryStreams, SupportsFinalizeAndQueriesOnANondefaultStream) {
  nvMolKit::ScopedStream     stream("SubstructLibraryStreams");
  nvMolKit::SubstructLibrary library;
  auto                       target = molFromSmiles("CCOC(=O)C");
  auto                       query  = queryFromSmarts("C=O");
  ASSERT_NE(target, nullptr);
  ASSERT_NE(query, nullptr);

  EXPECT_EQ(library.addMol(*target), 0U);
  library.finalize(stream.stream());
  EXPECT_EQ(library.getMatches(*query, -1, stream.stream()), std::vector<MoleculeId>({0U}));
  EXPECT_EQ(library.countMatches(*query, stream.stream()), 1U);
  EXPECT_TRUE(library.hasMatch(*query, stream.stream()));
}

}  // namespace
