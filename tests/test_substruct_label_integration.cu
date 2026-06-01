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

#include <GraphMol/QueryAtom.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <climits>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "src/data_structures/flat_bit_vect.h"
#include "src/substruct/graph_labeler.cuh"
#include "src/substruct/molecules.h"
#include "src/substruct/molecules_device.cuh"
#include "src/substruct/substruct_search.h"
#include "src/testutils/mol_data.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "tests/test_utils.h"

using nvMolKit::addQueryToBatch;
using nvMolKit::addToBatch;
using nvMolKit::AsyncDeviceVector;
using nvMolKit::BitMatrix2DView;
using nvMolKit::checkReturnCode;
using nvMolKit::FlatBitVect;
using nvMolKit::kMaxQueryAtoms;
using nvMolKit::kMaxTargetAtoms;
using nvMolKit::MoleculesDevice;
using nvMolKit::MoleculesHost;
using nvMolKit::ScopedStream;
using nvMolKit::testing::readSmartsFileWithStrings;
using nvMolKit::testing::readSmilesFileWithStrings;

namespace {

constexpr size_t kMaxAtoms  = 128;
constexpr size_t kNumSmiles = 300;

struct DatasetConfig {
  const char* smartsFile;
  const char* name;
};

constexpr DatasetConfig kDatasets[] = {
  {           "pwalters_alert_collection_supported.txt",            "PwaltersAlertCollection"},
  {         "openbabel_functional_groups_supported.txt",          "OpenBabelFunctionalGroups"},
  {                     "BMS_2006_filter_supported.txt",                      "BMS2006Filter"},
  {          "rdkit_fragment_descriptors_supported.txt",           "RDKitFragmentDescriptors"},
  {           "rdkit_tautomer_transforms_supported.txt",            "RDKitTautomerTransforms"},
  {         "rdkit_torsionPreferences_v2_supported.txt",          "RDKitTorsionPreferencesV2"},
  { "rdkit_torsionPreferences_smallrings_supported.txt",  "RDKitTorsionPreferencesSmallRings"},
  {           "rdkit_pattern_fingerprint_supported.txt",           "RDKitPatternFingerprints"},
  {"rdkit_torsionPreferences_macrocycles_supported.txt", "RDKitTorsionPreferencesMacrocycles"},
  {                       "RLewis_smarts_supported.txt",                       "RLewisSMARTS"},
  {                          "wehi_pains_supported.txt",                          "WEHIPAINS"},
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

void printSmallestReproSimple(const char*                     label,
                              const SmallestRepro&            r,
                              const std::vector<std::string>& targetSmiles,
                              const std::vector<std::string>& querySmarts,
                              int                             fp,
                              int                             fn) {
  std::cout << "  --- " << label << " (t=" << r.t << " q=" << r.q << " sum=" << (r.t + r.q) << ") ---\n";
  std::cout << "  Target[" << r.t << "]: " << targetSmiles[r.t] << "\n";
  std::cout << "  Query[" << r.q << "]:  " << querySmarts[r.q] << "\n";
  std::cout << "    FP=" << fp << " FN=" << fn << "\n\n";
}

template <typename GetFPFN>
void printSmallestReprosSimple(const SmallestRepros&           repros,
                               const std::vector<std::string>& targetSmiles,
                               const std::vector<std::string>& querySmarts,
                               const std::string&              category,
                               GetFPFN                         getFPFN) {
  if (!repros.hasAny())
    return;

  std::cout << "\n=== Smallest " << category << " repros ===\n";

  auto printOne = [&](const char* label, const SmallestRepro& r) {
    auto [fp, fn] = getFPFN(r.t, r.q);
    printSmallestReproSimple(label, r, targetSmiles, querySmarts, fp, fn);
  };

  if (repros.allSame()) {
    printOne("smallest (all criteria)", repros.smallestSum);
  } else {
    printOne("smallest sum (t+q)", repros.smallestSum);
    if (repros.smallestQ.t != repros.smallestSum.t || repros.smallestQ.q != repros.smallestSum.q) {
      printOne("smallest q", repros.smallestQ);
    }
    if (repros.smallestT.t != repros.smallestSum.t || repros.smallestT.q != repros.smallestSum.q) {
      if (repros.smallestT.t != repros.smallestQ.t || repros.smallestT.q != repros.smallestQ.q) {
        printOne("smallest t", repros.smallestT);
      }
    }
  }
}

using LabelMatrixStorage = FlatBitVect<kMaxTargetAtoms * kMaxQueryAtoms>;
using LabelMatrixView    = BitMatrix2DView<kMaxTargetAtoms, kMaxQueryAtoms>;

template <std::size_t MaxTarget, std::size_t MaxQuery>
__global__ void populateLabelMatrixKernelForIntegration(nvMolKit::TargetMoleculesDeviceView targetsView,
                                                        int                                 targetIdx,
                                                        nvMolKit::QueryMoleculesDeviceView  queriesView,
                                                        int                                 queryIdx,
                                                        LabelMatrixStorage*                 output) {
  nvMolKit::TargetMoleculeView target = nvMolKit::getMolecule(targetsView, targetIdx);
  nvMolKit::QueryMoleculeView  query  = nvMolKit::getMolecule(queriesView, queryIdx);

  BitMatrix2DView<MaxTarget, MaxQuery> view(*output);
  nvMolKit::populateLabelMatrix<MaxTarget, MaxQuery>(target, query, view);
}

}  // namespace

class LabelMatrixIntegrationTest : public ::testing::TestWithParam<DatasetConfig> {
 protected:
  ScopedStream stream_;
  std::string  testDataPath_;

  void SetUp() override { testDataPath_ = getTestDataFolderPath(); }

  const DatasetConfig& dataset() const { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(AllDatasets,
                         LabelMatrixIntegrationTest,
                         ::testing::ValuesIn(kDatasets),
                         [](const ::testing::TestParamInfo<DatasetConfig>& info) {
                           return std::string(info.param.name);
                         });

TEST_P(LabelMatrixIntegrationTest, ChemblVsSmartsLabelMatrix) {
  const std::string smilesPath = testDataPath_ + "/chembl_1k.smi";
  const std::string smartsPath = testDataPath_ + "/SMARTS/" + dataset().smartsFile;

  ASSERT_TRUE(std::filesystem::exists(smilesPath)) << "SMILES file not found: " << smilesPath;
  ASSERT_TRUE(std::filesystem::exists(smartsPath)) << "SMARTS file not found: " << smartsPath;

  auto [targetMols, targetSmiles] = readSmilesFileWithStrings(smilesPath, kNumSmiles, kMaxAtoms);
  auto [queryMols, querySmarts]   = readSmartsFileWithStrings(smartsPath);

  ASSERT_FALSE(targetMols.empty()) << "No target molecules loaded";
  ASSERT_FALSE(queryMols.empty()) << "No query patterns loaded";

  MoleculesHost targetsHost;
  MoleculesHost queriesHost;

  for (const auto& mol : targetMols) {
    addToBatch(mol.get(), targetsHost);
  }
  for (const auto& mol : queryMols) {
    addQueryToBatch(mol.get(), queriesHost);
  }

  MoleculesDevice targetsDevice(stream_.stream());
  MoleculesDevice queriesDevice(stream_.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  AsyncDeviceVector<LabelMatrixStorage> matrixDev(1, stream_.stream());

  const int numTargets = static_cast<int>(targetMols.size());
  const int numQueries = static_cast<int>(queryMols.size());

  std::vector<bool> queryHasRecursive(static_cast<size_t>(numQueries), false);
  for (int q = 0; q < numQueries; ++q) {
    queryHasRecursive[static_cast<size_t>(q)] = nvMolKit::hasRecursiveSmarts(queryMols[q].get());
  }

  int totalPairs          = 0;
  int totalMismatches     = 0;
  int totalFalsePositives = 0;
  int totalFalseNegatives = 0;

  std::vector<std::tuple<int, int, int, int>>                       fpPairs;
  std::vector<std::tuple<int, int, int, int>>                       fnPairs;
  std::vector<std::tuple<int, int, int, int, int, int, bool, bool>> fpDetails;
  std::vector<std::tuple<int, int, int, int, int, int, bool, bool>> fnDetails;

  for (int t = 0; t < numTargets; ++t) {
    for (int q = 0; q < numQueries; ++q) {
      ++totalPairs;
      const bool isRecursive = queryHasRecursive[static_cast<size_t>(q)];

      LabelMatrixStorage hostMatrix(false);
      matrixDev.setFromVector(std::vector<LabelMatrixStorage>{hostMatrix});

      populateLabelMatrixKernelForIntegration<kMaxTargetAtoms, kMaxQueryAtoms>
        <<<1, 128, 0, stream_.stream()>>>(targetsDevice.view<nvMolKit::MoleculeType::Target>(),
                                          t,
                                          queriesDevice.view<nvMolKit::MoleculeType::Query>(),
                                          q,
                                          matrixDev.data());
      cudaCheckError(cudaGetLastError());

      std::vector<LabelMatrixStorage> resultMatrix(1);
      matrixDev.copyToHost(resultMatrix);
      cudaCheckError(cudaStreamSynchronize(stream_.stream()));

      LabelMatrixView view(resultMatrix[0]);

      const int numTargetAtoms = static_cast<int>(targetMols[t]->getNumAtoms());
      const int numQueryAtoms  = static_cast<int>(queryMols[q]->getNumAtoms());

      int pairMismatches     = 0;
      int pairFalsePositives = 0;
      int pairFalseNegatives = 0;

      for (int ta = 0; ta < numTargetAtoms; ++ta) {
        const auto* targetAtom = targetMols[t]->getAtomWithIdx(ta);
        for (int qa = 0; qa < numQueryAtoms; ++qa) {
          const auto* queryAtom = queryMols[q]->getAtomWithIdx(qa);

          bool rdkitResult = false;
          if (queryAtom->hasQuery()) {
            rdkitResult = queryAtom->Match(targetAtom);
          } else {
            rdkitResult = (targetAtom->getAtomicNum() == queryAtom->getAtomicNum());
          }

          bool gpuResult = view.get(ta, qa);

          if (gpuResult && !rdkitResult) {
            ++pairFalsePositives;
            ++pairMismatches;
            if (fpDetails.size() < 20) {
              fpDetails.push_back({t,
                                   q,
                                   ta,
                                   qa,
                                   targetAtom->getAtomicNum(),
                                   queryAtom->getAtomicNum(),
                                   targetAtom->getIsAromatic(),
                                   queryAtom->getIsAromatic()});
            }
          } else if (!gpuResult && rdkitResult && !isRecursive) {
            ++pairFalseNegatives;
            ++pairMismatches;
            if (fnDetails.size() < 20) {
              fnDetails.push_back({t,
                                   q,
                                   ta,
                                   qa,
                                   targetAtom->getAtomicNum(),
                                   queryAtom->getAtomicNum(),
                                   targetAtom->getIsAromatic(),
                                   queryAtom->getIsAromatic()});
            }
          }
        }
      }

      totalMismatches += pairMismatches;
      totalFalsePositives += pairFalsePositives;
      totalFalseNegatives += pairFalseNegatives;

      if (pairFalsePositives > 0 && fpPairs.size() < 10) {
        fpPairs.emplace_back(t, q, pairFalsePositives, pairFalseNegatives);
      }
      if (pairFalseNegatives > 0 && fnPairs.size() < 10) {
        fnPairs.emplace_back(t, q, pairFalsePositives, pairFalseNegatives);
      }
    }
  }

  std::cout << "Label matrix integration test:\n"
            << "  Total pairs tested: " << totalPairs << "\n"
            << "  Total atom-level mismatches: " << totalMismatches << "\n"
            << "  False positives (GPU yes, RDKit no): " << totalFalsePositives << "\n"
            << "  False negatives (GPU no, RDKit yes): " << totalFalseNegatives << "\n";

  if (!fpPairs.empty()) {
    std::cout << "  Pairs with FALSE POSITIVES:\n";
    for (const auto& [t, q, fp, fn] : fpPairs) {
      std::cout << "    Target[" << t << "] x Query[" << q << "]: " << fp << " false positives\n"
                << "      Target: " << targetSmiles[t].substr(0, 80) << "...\n"
                << "      Query: " << querySmarts[q] << "\n";
    }
  }

  if (!fnPairs.empty()) {
    std::cout << "  Pairs with FALSE NEGATIVES:\n";
    for (const auto& [t, q, fp, fn] : fnPairs) {
      std::cout << "    Target[" << t << "] x Query[" << q << "]: " << fn << " false negatives\n"
                << "      Target: " << targetSmiles[t].substr(0, 80) << "...\n"
                << "      Query: " << querySmarts[q] << "\n";
    }
  }

  if (!fpDetails.empty()) {
    std::cout << "  False positive atom details:\n";
    for (const auto& [t, q, ta, qa, tAtomNum, qAtomNum, tArom, qArom] : fpDetails) {
      std::cout << "    T[" << t << "].atom" << ta << " (Z=" << tAtomNum << ",arom=" << tArom << ")"
                << " vs Q[" << q << "].atom" << qa << " (Z=" << qAtomNum << ",arom=" << qArom << ")\n";
    }
  }

  if (!fnDetails.empty()) {
    std::cout << "  False negative atom details:\n";
    for (const auto& [t, q, ta, qa, tAtomNum, qAtomNum, tArom, qArom] : fnDetails) {
      std::cout << "    T[" << t << "].atom" << ta << " (Z=" << tAtomNum << ",arom=" << tArom << ")"
                << " vs Q[" << q << "].atom" << qa << " (Z=" << qAtomNum << ",arom=" << qArom << ")\n";
    }
  }

  if (!fpPairs.empty()) {
    std::map<std::pair<int, int>, std::pair<int, int>> fpInfo;
    for (const auto& [t, q, fp, fn] : fpPairs) {
      fpInfo[{t, q}] = {fp, fn};
    }
    auto repros = findSmallestRepros(
      fpPairs,
      [](const auto& p) { return std::get<0>(p); },
      [](const auto& p) { return std::get<1>(p); });
    printSmallestReprosSimple(repros,
                              targetSmiles,
                              querySmarts,
                              "false positive",
                              [&](int t, int q) -> std::pair<int, int> {
                                auto it = fpInfo.find({t, q});
                                return it != fpInfo.end() ? it->second : std::pair{0, 0};
                              });
  }

  if (!fnPairs.empty()) {
    std::map<std::pair<int, int>, std::pair<int, int>> fnInfo;
    for (const auto& [t, q, fp, fn] : fnPairs) {
      fnInfo[{t, q}] = {fp, fn};
    }
    auto repros = findSmallestRepros(
      fnPairs,
      [](const auto& p) { return std::get<0>(p); },
      [](const auto& p) { return std::get<1>(p); });
    printSmallestReprosSimple(repros,
                              targetSmiles,
                              querySmarts,
                              "false negative",
                              [&](int t, int q) -> std::pair<int, int> {
                                auto it = fnInfo.find({t, q});
                                return it != fnInfo.end() ? it->second : std::pair{0, 0};
                              });
  }

  EXPECT_EQ(totalFalsePositives, 0)
    << "GPU label matrix has false positives (marks atoms as compatible when RDKit says no)";
  EXPECT_EQ(totalFalseNegatives, 0)
    << "GPU label matrix has false negatives for non-recursive queries (misses atoms that RDKit marks compatible)";
}
