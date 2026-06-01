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
#include <GraphMol/Substruct/SubstructMatch.h>

#include <algorithm>
#include <iostream>
#include <set>

#include "src/substruct/graph_labeler.cuh"
#include "src/substruct/molecules_device.cuh"
#include "src/substruct/substruct_search_internal.h"
#include "src/testutils/substruct_validation.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

std::vector<std::vector<int>> getRDKitSubstructMatches(const RDKit::ROMol& target,
                                                       const RDKit::ROMol& query,
                                                       bool                uniquify) {
  RDKit::SubstructMatchParameters params;
  params.uniquify = uniquify;

  std::vector<RDKit::MatchVectType> matches = RDKit::SubstructMatch(target, query, params);

  std::vector<std::vector<int>> result;
  result.reserve(matches.size());

  for (const auto& match : matches) {
    std::vector<int> mapping(match.size());
    for (size_t i = 0; i < match.size(); ++i) {
      mapping[match[i].first] = match[i].second;
    }
    result.push_back(std::move(mapping));
  }

  return result;
}

bool matchSetsEqual(const std::vector<std::vector<int>>& gpuMatches,
                    const std::vector<std::vector<int>>& rdkitMatches) {
  if (gpuMatches.size() != rdkitMatches.size()) {
    return false;
  }

  std::set<std::vector<int>> gpuSet(gpuMatches.begin(), gpuMatches.end());
  std::set<std::vector<int>> rdkitSet(rdkitMatches.begin(), rdkitMatches.end());

  return gpuSet == rdkitSet;
}

std::string algorithmName(SubstructAlgorithm algo) {
  switch (algo) {
    case SubstructAlgorithm::VF2:
      return "VF2";
    case SubstructAlgorithm::GSI:
      return "GSI";
  }
  return "Unknown";
}

namespace {

/**
 * @brief Extract GPU matches for a (target, query) pair from results.
 */
std::vector<std::vector<int>> extractGpuMatches(const SubstructSearchResults& results,
                                                int                           targetIdx,
                                                int                           queryIdx,
                                                int /* numQueryAtoms */) {
  return results.getMatches(targetIdx, queryIdx);
}

}  // namespace

SubstructValidationResult validateAgainstRDKit(const SubstructSearchResults&                     results,
                                               const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                                               const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                                               bool                                              uniquify) {
  SubstructValidationResult validation;
  validation.totalPairs = results.numTargets * results.numQueries;

  for (int t = 0; t < results.numTargets; ++t) {
    for (int q = 0; q < results.numQueries; ++q) {
      const auto rdkitMatches    = getRDKitSubstructMatches(*targetMols[t], *queryMols[q], uniquify);
      const int  gpuMatchCount   = results.matchCount(t, q);
      const int  rdkitMatchCount = static_cast<int>(rdkitMatches.size());

      if (gpuMatchCount != rdkitMatchCount) {
        validation.mismatchedPairs++;
        validation.mismatches.emplace_back(t, q, gpuMatchCount, rdkitMatchCount);
      } else if (gpuMatchCount > 0) {
        const int  numQueryAtoms = static_cast<int>(queryMols[q]->getNumAtoms());
        const auto gpuMatches    = extractGpuMatches(results, t, q, numQueryAtoms);

        if (matchSetsEqual(gpuMatches, rdkitMatches)) {
          validation.matchingPairs++;
        } else {
          validation.wrongMappingPairs++;
          validation.mappingMismatches.emplace_back(t, q);
        }
      } else {
        validation.matchingPairs++;
      }
    }
  }

  validation.allMatch = (validation.mismatchedPairs == 0 && validation.wrongMappingPairs == 0);
  return validation;
}

void printValidationResult(const SubstructValidationResult& result, const std::string& algoName) {
  std::string prefix = algoName.empty() ? "" : "[" + algoName + "] ";

  std::cout << prefix << "Validation: " << result.matchingPairs << "/" << result.totalPairs << " pairs match RDKit";

  if (result.allMatch) {
    std::cout << " - PASS" << std::endl;
  } else {
    const int totalFailures = result.mismatchedPairs + result.wrongMappingPairs;
    std::cout << " - FAIL (" << totalFailures << " failures)" << std::endl;

    const int maxPrint = 10;
    int       printed  = 0;

    if (result.mismatchedPairs > 0) {
      std::cout << "  Count mismatches:" << std::endl;
      for (const auto& [t, q, gpu, rdkit] : result.mismatches) {
        if (printed++ >= maxPrint) {
          std::cout << "    ... and " << (result.mismatchedPairs - maxPrint) << " more" << std::endl;
          break;
        }
        std::cout << "    target=" << t << " query=" << q << ": GPU=" << gpu << " RDKit=" << rdkit << std::endl;
      }
    }

    if (result.wrongMappingPairs > 0) {
      std::cout << "  Mapping mismatches (count correct but indices differ):" << std::endl;
      printed = 0;
      for (const auto& [t, q] : result.mappingMismatches) {
        if (printed++ >= maxPrint) {
          std::cout << "    ... and " << (result.wrongMappingPairs - maxPrint) << " more" << std::endl;
          break;
        }
        std::cout << "    target=" << t << " query=" << q << std::endl;
      }
    }
  }
}

namespace {

void printMatches(const std::string& label, const std::vector<std::vector<int>>& matches, size_t maxToPrint = 20) {
  std::cout << "    " << label << ": ";
  if (matches.empty()) {
    std::cout << "(none)" << std::endl;
    return;
  }
  std::cout << matches.size() << " match(es)" << std::endl;
  const size_t toPrint = std::min(matches.size(), maxToPrint);
  for (size_t i = 0; i < toPrint; ++i) {
    std::cout << "      [" << i << "]: {";
    for (size_t j = 0; j < matches[i].size(); ++j) {
      if (j > 0)
        std::cout << ", ";
      std::cout << matches[i][j];
    }
    std::cout << "}" << std::endl;
  }
  if (matches.size() > maxToPrint) {
    std::cout << "      ... and " << (matches.size() - maxToPrint) << " more" << std::endl;
  }
}

std::vector<std::vector<int>> extractGpuMatchesForPrint(const SubstructSearchResults& results,
                                                        int                           targetIdx,
                                                        int                           queryIdx,
                                                        int /* numQueryAtoms */) {
  return results.getMatches(targetIdx, queryIdx);
}

using LabelMatrixStorage = FlatBitVect<kMaxTargetAtoms * kMaxQueryAtoms>;
using LabelMatrixView    = BitMatrix2DView<kMaxTargetAtoms, kMaxQueryAtoms>;

template <std::size_t MaxTarget, std::size_t MaxQuery>
__global__ void populateLabelMatrixKernel(TargetMoleculesDeviceView          targetBatch,
                                          int                                targetMolIdx,
                                          QueryMoleculesDeviceView           queryBatch,
                                          int                                queryMolIdx,
                                          FlatBitVect<MaxTarget * MaxQuery>* matrix,
                                          const uint32_t*                    pairRecursiveBits) {
  TargetMoleculeView                   target = getMolecule(targetBatch, targetMolIdx);
  QueryMoleculeView                    query  = getMolecule(queryBatch, queryMolIdx);
  BitMatrix2DView<MaxTarget, MaxQuery> view(matrix);
  populateLabelMatrix<MaxTarget, MaxQuery>(target, query, view, pairRecursiveBits);
}

}  // namespace

void printValidationResultDetailed(const SubstructValidationResult&                  result,
                                   const SubstructSearchResults&                     gpuResults,
                                   const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                                   const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                                   const std::vector<std::string>&                   targetSmiles,
                                   const std::vector<std::string>&                   querySmarts,
                                   const std::string&                                algoName,
                                   int                                               maxDetails,
                                   bool                                              uniquify) {
  std::string prefix = algoName.empty() ? "" : "[" + algoName + "] ";

  std::cout << prefix << "Validation: " << result.matchingPairs << "/" << result.totalPairs << " pairs match RDKit";

  if (result.allMatch) {
    std::cout << " - PASS" << std::endl;
    return;
  }

  const int totalFailures = result.mismatchedPairs + result.wrongMappingPairs;
  std::cout << " - FAIL (" << totalFailures << " failures)" << std::endl;

  int printed = 0;

  if (result.mismatchedPairs > 0) {
    std::cout << "  Count mismatches:" << std::endl;
    for (const auto& [t, q, gpuCount, rdkitCount] : result.mismatches) {
      if (printed++ >= maxDetails) {
        std::cout << "  ... and " << (result.mismatchedPairs - maxDetails) << " more count mismatches" << std::endl;
        break;
      }
      std::cout << "  Target[" << t << "]: " << targetSmiles[t] << std::endl;
      std::cout << "  Query[" << q << "]:  " << querySmarts[q] << std::endl;

      auto rdkitMatches = getRDKitSubstructMatches(*targetMols[t], *queryMols[q], uniquify);
      printMatches("Expected (RDKit)", rdkitMatches);

      const int numQueryAtoms = static_cast<int>(queryMols[q]->getNumAtoms());
      auto      gpuMatches    = extractGpuMatchesForPrint(gpuResults, t, q, numQueryAtoms);
      printMatches("Actual (GPU)", gpuMatches);

      std::cout << std::endl;
    }
  }

  if (result.wrongMappingPairs > 0) {
    std::cout << "  Mapping mismatches (count correct but indices differ):" << std::endl;
    printed = 0;
    for (const auto& [t, q] : result.mappingMismatches) {
      if (printed++ >= maxDetails) {
        std::cout << "  ... and " << (result.wrongMappingPairs - maxDetails) << " more mapping mismatches" << std::endl;
        break;
      }
      std::cout << "  Target[" << t << "]: " << targetSmiles[t] << std::endl;
      std::cout << "  Query[" << q << "]:  " << querySmarts[q] << std::endl;

      auto rdkitMatches = getRDKitSubstructMatches(*targetMols[t], *queryMols[q], uniquify);
      printMatches("Expected (RDKit)", rdkitMatches);

      // Get GPU matches
      const int numQueryAtoms = static_cast<int>(queryMols[q]->getNumAtoms());
      auto      gpuMatches    = extractGpuMatchesForPrint(gpuResults, t, q, numQueryAtoms);
      printMatches("Actual (GPU)", gpuMatches);

      std::cout << std::endl;
    }
  }
}

std::vector<std::vector<uint8_t>> computeRDKitLabelMatrix(const RDKit::ROMol& targetMol, const RDKit::ROMol& queryMol) {
  const int numTargetAtoms = static_cast<int>(targetMol.getNumAtoms());
  const int numQueryAtoms  = static_cast<int>(queryMol.getNumAtoms());

  std::vector<std::vector<uint8_t>> result(numTargetAtoms, std::vector<uint8_t>(numQueryAtoms, 0));

  // For recursive SMARTS, atom-level Match() doesn't work correctly.
  // Use full substructure matching to get valid (targetAtom, queryAtom) pairs.
  RDKit::MatchVectType            match;
  RDKit::SubstructMatchParameters params;
  params.uniquify   = false;
  params.maxMatches = 0;  // Find all matches

  auto matches = RDKit::SubstructMatch(targetMol, queryMol, params);

  for (const auto& matchVec : matches) {
    for (const auto& pair : matchVec) {
      int qa = pair.first;
      int ta = pair.second;
      if (ta >= 0 && ta < numTargetAtoms && qa >= 0 && qa < numQueryAtoms) {
        result[ta][qa] = 1;
      }
    }
  }

  return result;
}

std::vector<std::vector<uint8_t>> computeGpuLabelMatrix(const RDKit::ROMol& targetMol,
                                                        const RDKit::ROMol& queryMol,
                                                        cudaStream_t        stream) {
  MoleculesHost targetHost;
  MoleculesHost queryHost;
  addToBatch(&targetMol, targetHost);
  addQueryToBatch(&queryMol, queryHost);

  MoleculesDevice targetDevice(stream);
  MoleculesDevice queryDevice(stream);
  targetDevice.copyFromHost(targetHost);
  queryDevice.copyFromHost(queryHost);

  const int numTargetAtoms = static_cast<int>(targetHost.totalAtoms());
  const int numQueryAtoms  = static_cast<int>(queryHost.totalAtoms());

  std::vector<int>       queryAtomCounts = {numQueryAtoms};
  AsyncDeviceVector<int> pairMatchStartsDev(2, stream);
  pairMatchStartsDev.zero();

  MiniBatchResultsDevice miniBatchResults(stream);
  miniBatchResults.allocateMiniBatch(1, pairMatchStartsDev.data(), 0, 1, numTargetAtoms, 2);
  miniBatchResults.setQueryAtomCounts(queryAtomCounts.data(), queryAtomCounts.size());
  miniBatchResults.zeroRecursiveBits();

  if (!queryHost.recursivePatterns.empty() && !queryHost.recursivePatterns[0].empty()) {
    LeafSubpatterns leafSubpatterns;
    leafSubpatterns.buildAllPatterns(queryHost);
    leafSubpatterns.syncToDevice(stream);

    RecursiveScratchBuffers scratch(stream);
    scratch.allocateBuffers(256);
    std::vector<BatchedPatternEntry> scratchPatternEntries;
    preprocessRecursiveSmarts(SubstructTemplateConfig::Config_T128_Q64_B8,
                              targetDevice,
                              queryHost,
                              leafSubpatterns,
                              miniBatchResults,
                              1,
                              0,
                              1,
                              SubstructAlgorithm::GSI,
                              stream,
                              scratch,
                              scratchPatternEntries,
                              nullptr,
                              0);
  }
  RecursivePatternInfo info = extractRecursivePatterns(&queryMol);

  AsyncDeviceVector<LabelMatrixStorage> matrixDev(1, stream);
  const LabelMatrixStorage              hostMatrix(false);
  matrixDev.setFromVector(std::vector<LabelMatrixStorage>{hostMatrix});

  const uint32_t* pairRecursiveBits = info.empty() ? nullptr : miniBatchResults.recursiveMatchBits();

  populateLabelMatrixKernel<kMaxTargetAtoms, kMaxQueryAtoms>
    <<<1, 128, 0, stream>>>(targetDevice.view<MoleculeType::Target>(),
                            0,
                            queryDevice.view<MoleculeType::Query>(),
                            0,
                            matrixDev.data(),
                            pairRecursiveBits);
  cudaCheckError(cudaGetLastError());

  std::vector<LabelMatrixStorage> resultMatrix(1);
  matrixDev.copyToHost(resultMatrix);
  cudaCheckError(cudaStreamSynchronize(stream));

  const LabelMatrixView view(resultMatrix[0]);

  std::vector<std::vector<uint8_t>> result(numTargetAtoms, std::vector<uint8_t>(numQueryAtoms));
  for (int ta = 0; ta < numTargetAtoms; ++ta) {
    for (int qa = 0; qa < numQueryAtoms; ++qa) {
      result[ta][qa] = view.get(ta, qa) ? 1 : 0;
    }
  }

  return result;
}

LabelMatrixComparisonResult compareLabelMatrices(const RDKit::ROMol& targetMol,
                                                 const RDKit::ROMol& queryMol,
                                                 cudaStream_t        stream) {
  auto gpuMatrix   = computeGpuLabelMatrix(targetMol, queryMol, stream);
  auto rdkitMatrix = computeRDKitLabelMatrix(targetMol, queryMol);

  LabelMatrixComparisonResult result;
  result.numTargetAtoms   = static_cast<int>(gpuMatrix.size());
  result.numQueryAtoms    = result.numTargetAtoms > 0 ? static_cast<int>(gpuMatrix[0].size()) : 0;
  result.totalComparisons = result.numTargetAtoms * result.numQueryAtoms;

  for (int ta = 0; ta < result.numTargetAtoms; ++ta) {
    for (int qa = 0; qa < result.numQueryAtoms; ++qa) {
      bool gpuResult   = gpuMatrix[ta][qa] != 0;
      bool rdkitResult = rdkitMatrix[ta][qa] != 0;

      if (gpuResult != rdkitResult) {
        if (gpuResult && !rdkitResult) {
          ++result.falsePositives;
        } else {
          ++result.falseNegatives;
        }
        result.mismatches.emplace_back(ta, qa, gpuResult, rdkitResult);
      }
    }
  }

  result.allMatch = (result.falsePositives == 0 && result.falseNegatives == 0);
  return result;
}

}  // namespace nvMolKit
