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
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "src/substruct/molecules.h"
#include "src/substruct/molecules_device.cuh"
#include "src/substruct/recursive_preprocessor.h"
#include "src/substruct/substruct_search_internal.h"
#include "src/substruct/substruct_types.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"

using nvMolKit::AsyncDeviceVector;
using nvMolKit::BatchedPatternEntry;
using nvMolKit::checkReturnCode;
using nvMolKit::kMaxSmartsNestingDepth;
using nvMolKit::LeafSubpatterns;
using nvMolKit::MiniBatchResultsDevice;
using nvMolKit::MoleculesDevice;
using nvMolKit::MoleculesHost;
using nvMolKit::RecursivePatternPreprocessor;
using nvMolKit::RecursiveScratchBuffers;
using nvMolKit::ScopedStream;
using nvMolKit::SubstructAlgorithm;
using nvMolKit::SubstructTemplateConfig;

namespace {

std::unique_ptr<RDKit::ROMol> makeMolFromSmiles(const std::string& smiles) {
  return std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
}

std::unique_ptr<RDKit::ROMol> makeMolFromSmarts(const std::string& smarts) {
  return std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
}

int maxAtomsPerTarget(const MoleculesHost& targetsHost) {
  int       maxAtoms   = 0;
  const int numTargets = static_cast<int>(targetsHost.numMolecules());
  for (int i = 0; i < numTargets; ++i) {
    const int atomStart = targetsHost.batchAtomStarts[i];
    const int atomEnd   = targetsHost.batchAtomStarts[i + 1];
    maxAtoms            = std::max(maxAtoms, atomEnd - atomStart);
  }
  return maxAtoms;
}

std::vector<int> queryAtomCounts(const MoleculesHost& queriesHost) {
  const int        numQueries = static_cast<int>(queriesHost.numMolecules());
  std::vector<int> counts(static_cast<size_t>(numQueries));
  for (int i = 0; i < numQueries; ++i) {
    const int atomStart = queriesHost.batchAtomStarts[i];
    const int atomEnd   = queriesHost.batchAtomStarts[i + 1];
    counts[i]           = atomEnd - atomStart;
  }
  return counts;
}

}  // namespace

TEST(RecursivePreprocessorTest, PaintsBitsForSimpleRecursivePattern) {
  ScopedStream stream;

  auto target0 = makeMolFromSmiles("CN");
  auto target1 = makeMolFromSmiles("CC");
  auto query   = makeMolFromSmarts("[$(*-N)]");

  ASSERT_NE(target0, nullptr);
  ASSERT_NE(target1, nullptr);
  ASSERT_NE(query, nullptr);

  std::vector<const RDKit::ROMol*> targets = {target0.get(), target1.get()};
  std::vector<const RDKit::ROMol*> queries = {query.get()};
  std::vector<int>                 emptySortOrder;

  MoleculesHost targetsHost;
  nvMolKit::buildTargetBatchParallelInto(targetsHost, 1, targets, emptySortOrder);
  MoleculesHost queriesHost = nvMolKit::buildQueryBatchParallel(queries, emptySortOrder, 1);

  MoleculesDevice targetsDevice(stream.stream());
  targetsDevice.copyFromHost(targetsHost);

  RecursivePatternPreprocessor preprocessor;
  preprocessor.buildPatterns(queriesHost);
  preprocessor.syncToDevice(stream.stream());

  const int numTargets     = static_cast<int>(targets.size());
  const int numQueries     = static_cast<int>(queries.size());
  const int miniBatchSize  = numTargets * numQueries;
  const int maxTargetAtoms = maxAtomsPerTarget(targetsHost);
  ASSERT_GT(maxTargetAtoms, 0);

  AsyncDeviceVector<int> pairMatchStartsDev(static_cast<size_t>(miniBatchSize + 1), stream.stream());
  pairMatchStartsDev.zero();
  MiniBatchResultsDevice miniBatchResults(stream.stream());
  miniBatchResults.allocateMiniBatch(miniBatchSize, pairMatchStartsDev.data(), 0, numQueries, maxTargetAtoms, 2);
  const std::vector<int> atomCounts = queryAtomCounts(queriesHost);
  miniBatchResults.setQueryAtomCounts(atomCounts.data(), atomCounts.size());
  miniBatchResults.zeroRecursiveBits();

  RecursiveScratchBuffers scratch(stream.stream());
  scratch.allocateBuffers(256);

  const LeafSubpatterns& leafSubpatterns = preprocessor.leafSubpatterns();
  ASSERT_FALSE(leafSubpatterns.perQueryPatterns.empty());

  std::array<std::vector<BatchedPatternEntry>, kMaxSmartsNestingDepth + 1> patternsAtDepth;
  for (auto& vec : patternsAtDepth) {
    vec.clear();
  }

  const int queryMaxDepth = leafSubpatterns.perQueryMaxDepth.empty() ? 0 : leafSubpatterns.perQueryMaxDepth[0];
  for (int depth = 0; depth <= queryMaxDepth; ++depth) {
    const auto& src = leafSubpatterns.perQueryPatterns[0][depth];
    patternsAtDepth[depth].insert(patternsAtDepth[depth].end(), src.begin(), src.end());
  }

  const int miniBatchPairOffset    = 0;
  const int firstTargetInMiniBatch = miniBatchPairOffset / numQueries;
  const int lastTargetInMiniBatch  = (miniBatchPairOffset + miniBatchSize - 1) / numQueries;
  const int numTargetsInMiniBatch  = lastTargetInMiniBatch - firstTargetInMiniBatch + 1;

  preprocessor.preprocessMiniBatch(SubstructTemplateConfig::Config_T32_Q16_B4,
                                   targetsDevice,
                                   miniBatchResults,
                                   numQueries,
                                   miniBatchPairOffset,
                                   miniBatchSize,
                                   SubstructAlgorithm::GSI,
                                   stream.stream(),
                                   scratch,
                                   patternsAtDepth,
                                   queryMaxDepth,
                                   firstTargetInMiniBatch,
                                   numTargetsInMiniBatch,
                                   nullptr,
                                   0);

  std::vector<uint32_t> hostBits(static_cast<size_t>(miniBatchSize) * maxTargetAtoms);
  cudaCheckError(cudaMemcpyAsync(hostBits.data(),
                                 miniBatchResults.recursiveMatchBits(),
                                 hostBits.size() * sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost,
                                 stream.stream()));
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  auto hasRecursiveBit = [&](int targetIdx, int atomIdx) {
    const size_t offset = static_cast<size_t>(targetIdx) * maxTargetAtoms + atomIdx;
    return (hostBits[offset] & 0x1u) != 0;
  };

  EXPECT_TRUE(hasRecursiveBit(0, 0));
  EXPECT_FALSE(hasRecursiveBit(0, 1));

  EXPECT_FALSE(hasRecursiveBit(1, 0));
  EXPECT_FALSE(hasRecursiveBit(1, 1));
}

/**
 * @brief Leaf subpattern with more atoms than the caller's MaxQueryAtoms
 *        template tier should not overflow the shared memory label matrix.
 */
TEST(RecursivePreprocessorTest, LeafPatternLargerThanConfigMaxQueryAtoms) {
  ScopedStream stream;

  auto target = makeMolFromSmiles("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC");
  auto query  = makeMolFromSmarts("[$(*~C~C~C~C~C~C~C~C~C~C~C~C~C~C~C~C~C)]");

  ASSERT_NE(target, nullptr);
  ASSERT_NE(query, nullptr);

  std::vector<const RDKit::ROMol*> targets = {target.get()};
  std::vector<const RDKit::ROMol*> queries = {query.get()};
  std::vector<int>                 emptySortOrder;

  MoleculesHost targetsHost;
  nvMolKit::buildTargetBatchParallelInto(targetsHost, 1, targets, emptySortOrder);
  MoleculesHost queriesHost = nvMolKit::buildQueryBatchParallel(queries, emptySortOrder, 1);

  const int maxTargetAtoms = maxAtomsPerTarget(targetsHost);
  ASSERT_GE(maxTargetAtoms, 32);

  MoleculesDevice targetsDevice(stream.stream());
  targetsDevice.copyFromHost(targetsHost);

  RecursivePatternPreprocessor preprocessor;
  preprocessor.buildPatterns(queriesHost);
  preprocessor.syncToDevice(stream.stream());

  const LeafSubpatterns& leafSubpatterns = preprocessor.leafSubpatterns();
  ASSERT_FALSE(leafSubpatterns.empty());
  ASSERT_GT(leafSubpatterns.maxPatternAtoms(), 16);

  const int numTargets    = 1;
  const int numQueries    = 1;
  const int miniBatchSize = numTargets * numQueries;

  AsyncDeviceVector<int> pairMatchStartsDev(static_cast<size_t>(miniBatchSize + 1), stream.stream());
  pairMatchStartsDev.zero();
  MiniBatchResultsDevice miniBatchResults(stream.stream());
  miniBatchResults.allocateMiniBatch(miniBatchSize, pairMatchStartsDev.data(), 0, numQueries, maxTargetAtoms, 2);
  const std::vector<int> atomCounts = queryAtomCounts(queriesHost);
  miniBatchResults.setQueryAtomCounts(atomCounts.data(), atomCounts.size());
  miniBatchResults.zeroRecursiveBits();

  RecursiveScratchBuffers scratch(stream.stream());
  scratch.allocateBuffers(256);

  std::array<std::vector<BatchedPatternEntry>, kMaxSmartsNestingDepth + 1> patternsAtDepth;
  for (auto& vec : patternsAtDepth) {
    vec.clear();
  }

  const int queryMaxDepth = leafSubpatterns.perQueryMaxDepth.empty() ? 0 : leafSubpatterns.perQueryMaxDepth[0];
  for (int depth = 0; depth <= queryMaxDepth; ++depth) {
    const auto& src = leafSubpatterns.perQueryPatterns[0][depth];
    patternsAtDepth[depth].insert(patternsAtDepth[depth].end(), src.begin(), src.end());
  }

  preprocessor.preprocessMiniBatch(SubstructTemplateConfig::Config_T32_Q16_B4,
                                   targetsDevice,
                                   miniBatchResults,
                                   numQueries,
                                   0,
                                   miniBatchSize,
                                   SubstructAlgorithm::GSI,
                                   stream.stream(),
                                   scratch,
                                   patternsAtDepth,
                                   queryMaxDepth,
                                   0,
                                   numTargets,
                                   nullptr,
                                   0);

  cudaCheckError(cudaStreamSynchronize(stream.stream()));
  cudaCheckError(cudaGetLastError());

  std::vector<uint32_t> hostBits(static_cast<size_t>(miniBatchSize) * maxTargetAtoms);
  cudaCheckError(cudaMemcpyAsync(hostBits.data(),
                                 miniBatchResults.recursiveMatchBits(),
                                 hostBits.size() * sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost,
                                 stream.stream()));
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  bool anyBitSet = false;
  for (size_t i = 0; i < hostBits.size(); ++i) {
    if (hostBits[i] != 0) {
      anyBitSet = true;
      break;
    }
  }
  EXPECT_TRUE(anyBitSet);
}
