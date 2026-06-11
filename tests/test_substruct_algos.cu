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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "src/data_structures/flat_bit_vect.h"
#include "src/substruct/atom_data_packed.h"
#include "src/substruct/graph_labeler.cuh"
#include "src/substruct/molecules.h"
#include "src/substruct/molecules_device.cuh"
#include "src/substruct/packed_bonds.h"
#include "src/substruct/substruct_algos.cuh"
#include "src/substruct/substruct_types.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"

using nvMolKit::AsyncDeviceVector;
using nvMolKit::AtomDataPacked;
using nvMolKit::AtomQueryMask;
using nvMolKit::AtomQueryTree;
using nvMolKit::BitMatrix2DView;
using nvMolKit::BondTypeCounts;
using nvMolKit::BoolInstruction;
using nvMolKit::checkReturnCode;
using nvMolKit::FlatBitVect;
using nvMolKit::gsiBFSSearchGPU;
using nvMolKit::kMaxBondsPerAtom;
using nvMolKit::PartialMatchT;
using nvMolKit::QueryAtomBonds;
using nvMolKit::QueryMoleculeView;
using nvMolKit::ScopedStream;
using nvMolKit::SubstructOutputMode;
using nvMolKit::TargetAtomBonds;
using nvMolKit::TargetMoleculeView;
using nvMolKit::vf2SearchGPU;
using nvMolKit::VF2StateT;

namespace {

// =============================================================================
// Test Template Parameters (small sizes for overflow testing)
// =============================================================================

constexpr std::size_t kTestMaxTargetAtoms  = 32;
constexpr std::size_t kTestMaxQueryAtoms   = 16;
constexpr int         kTestMaxBondsPerAtom = 4;

// =============================================================================
// Synthetic Molecule Builder
// =============================================================================

/**
 * @brief Iteratively builds synthetic molecule data for testing.
 *
 * Avoids recursion by using simple loops to construct atom and bond arrays.
 */
class SyntheticMoleculeBuilder {
 public:
  explicit SyntheticMoleculeBuilder(int numAtoms) : numAtoms_(numAtoms) {
    atomDataPacked_.resize(numAtoms);
    bondTypeCounts_.resize(numAtoms);
    targetAtomBonds_.resize(numAtoms);
    queryAtomBonds_.resize(numAtoms);
    atomQueryMasks_.resize(numAtoms);
    atomQueryTrees_.resize(numAtoms);

    for (int i = 0; i < numAtoms; ++i) {
      atomDataPacked_[i]  = AtomDataPacked{};
      bondTypeCounts_[i]  = BondTypeCounts{};
      targetAtomBonds_[i] = TargetAtomBonds{};
      queryAtomBonds_[i]  = QueryAtomBonds{};
      atomQueryMasks_[i]  = AtomQueryMask{};
      atomQueryTrees_[i]  = AtomQueryTree{};
    }
  }

  void setAtomicNum(int atomIdx, uint8_t atomicNum) {
    atomDataPacked_[atomIdx].setAtomicNum(atomicNum);
    atomQueryMasks_[atomIdx].maskLo |= 0xFFULL;
    atomQueryMasks_[atomIdx].expectedLo |= static_cast<uint64_t>(atomicNum);
  }

  void setDegree(int atomIdx, uint8_t degree) { atomDataPacked_[atomIdx].setDegree(degree); }

  void addTargetBond(int atomIdx, int neighborIdx, uint8_t bondType = 1, bool isInRing = false) {
    auto& bonds            = targetAtomBonds_[atomIdx];
    int   idx              = bonds.degree;
    bonds.neighborIdx[idx] = static_cast<uint8_t>(neighborIdx);
    bonds.bondInfo[idx]    = nvMolKit::packTargetBondInfo(bondType, isInRing);
    bonds.degree           = idx + 1;
    bondTypeCounts_[atomIdx].single++;
  }

  void addQueryBond(int atomIdx, int neighborIdx, uint32_t matchMask = 0xFFFFFFFF) {
    auto& bonds            = queryAtomBonds_[atomIdx];
    int   idx              = bonds.degree;
    bonds.neighborIdx[idx] = static_cast<uint8_t>(neighborIdx);
    bonds.matchMask[idx]   = matchMask;
    bonds.degree           = idx + 1;
    bondTypeCounts_[atomIdx].single++;
  }

  void buildLinearChain(uint8_t atomicNum = 6) {
    for (int i = 0; i < numAtoms_; ++i) {
      setAtomicNum(i, atomicNum);
      int degree = 0;
      if (i > 0)
        degree++;
      if (i < numAtoms_ - 1)
        degree++;
      setDegree(i, degree);
    }
    for (int i = 0; i < numAtoms_ - 1; ++i) {
      addTargetBond(i, i + 1);
      addTargetBond(i + 1, i);
      addQueryBond(i, i + 1);
      addQueryBond(i + 1, i);
    }
  }

  TargetMoleculeView buildTargetView(cudaStream_t stream) {
    d_atomDataPacked_  = AsyncDeviceVector<AtomDataPacked>(numAtoms_, stream);
    d_bondTypeCounts_  = AsyncDeviceVector<BondTypeCounts>(numAtoms_, stream);
    d_targetAtomBonds_ = AsyncDeviceVector<TargetAtomBonds>(numAtoms_, stream);

    d_atomDataPacked_.copyFromHost(atomDataPacked_);
    d_bondTypeCounts_.copyFromHost(bondTypeCounts_);
    d_targetAtomBonds_.copyFromHost(targetAtomBonds_);

    TargetMoleculeView view;
    view.numAtoms        = numAtoms_;
    view.atomDataPacked  = d_atomDataPacked_.data();
    view.bondTypeCounts  = d_bondTypeCounts_.data();
    view.targetAtomBonds = d_targetAtomBonds_.data();
    return view;
  }

  QueryMoleculeView buildQueryView(cudaStream_t stream) {
    d_atomDataPacked_ = AsyncDeviceVector<AtomDataPacked>(numAtoms_, stream);
    d_atomQueryMasks_ = AsyncDeviceVector<AtomQueryMask>(numAtoms_, stream);
    d_bondTypeCounts_ = AsyncDeviceVector<BondTypeCounts>(numAtoms_, stream);
    d_queryAtomBonds_ = AsyncDeviceVector<QueryAtomBonds>(numAtoms_, stream);
    d_atomQueryTrees_ = AsyncDeviceVector<AtomQueryTree>(numAtoms_, stream);

    d_atomDataPacked_.copyFromHost(atomDataPacked_);
    d_atomQueryMasks_.copyFromHost(atomQueryMasks_);
    d_bondTypeCounts_.copyFromHost(bondTypeCounts_);
    d_queryAtomBonds_.copyFromHost(queryAtomBonds_);
    d_atomQueryTrees_.copyFromHost(atomQueryTrees_);

    QueryMoleculeView view;
    view.numAtoms            = numAtoms_;
    view.atomDataPacked      = d_atomDataPacked_.data();
    view.atomQueryMasks      = d_atomQueryMasks_.data();
    view.bondTypeCounts      = d_bondTypeCounts_.data();
    view.queryAtomBonds      = d_queryAtomBonds_.data();
    view.atomQueryTrees      = d_atomQueryTrees_.data();
    view.queryInstructions   = nullptr;
    view.queryLeafMasks      = nullptr;
    view.queryLeafBondCounts = nullptr;
    view.atomInstrStarts     = nullptr;
    view.atomLeafMaskStarts  = nullptr;
    return view;
  }

 private:
  int                          numAtoms_;
  std::vector<AtomDataPacked>  atomDataPacked_;
  std::vector<BondTypeCounts>  bondTypeCounts_;
  std::vector<TargetAtomBonds> targetAtomBonds_;
  std::vector<QueryAtomBonds>  queryAtomBonds_;
  std::vector<AtomQueryMask>   atomQueryMasks_;
  std::vector<AtomQueryTree>   atomQueryTrees_;

  AsyncDeviceVector<AtomDataPacked>  d_atomDataPacked_;
  AsyncDeviceVector<BondTypeCounts>  d_bondTypeCounts_;
  AsyncDeviceVector<TargetAtomBonds> d_targetAtomBonds_;
  AsyncDeviceVector<QueryAtomBonds>  d_queryAtomBonds_;
  AsyncDeviceVector<AtomQueryMask>   d_atomQueryMasks_;
  AsyncDeviceVector<AtomQueryTree>   d_atomQueryTrees_;
};

// =============================================================================
// Test Parameter Structures
// =============================================================================

struct AlgoTestParams {
  bool countOnly;
  int  maxMatches;
  int  maxMatchesToFind;

  friend std::ostream& operator<<(std::ostream& os, const AlgoTestParams& p) {
    os << "countOnly=" << p.countOnly << ",maxMatches=" << p.maxMatches << ",maxMatchesToFind=" << p.maxMatchesToFind;
    return os;
  }
};

// =============================================================================
// Kernel Wrappers for Testing
// =============================================================================

template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms, int MaxBondsPerAtom>
__global__ void testVF2SearchKernel(
  TargetMoleculeView                                                    target,
  QueryMoleculeView                                                     query,
  typename BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>::StorageType* labelMatrixStorage,
  int                                                                   startingTargetAtom,
  int*                                                                  matchCount,
  int*                                                                  reportedCount,
  int16_t*                                                              matchIndices,
  int                                                                   maxMatches,
  int                                                                   matchOffset,
  int                                                                   maxMatchesToFind,
  bool                                                                  countOnly) {
  __shared__ VF2StateT<MaxQueryAtoms>            state;
  BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms> labelMatrix(labelMatrixStorage);

  vf2SearchGPU<MaxTargetAtoms, MaxQueryAtoms, MaxBondsPerAtom>(target,
                                                               query,
                                                               labelMatrix,
                                                               state,
                                                               startingTargetAtom,
                                                               matchCount,
                                                               reportedCount,
                                                               matchIndices,
                                                               maxMatches,
                                                               matchOffset,
                                                               maxMatchesToFind,
                                                               countOnly);
}

template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms, int MaxBondsPerAtom>
__global__ void testPopulateLabelMatrixKernel(
  TargetMoleculeView                                                    target,
  QueryMoleculeView                                                     query,
  typename BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>::StorageType* labelMatrixStorage) {
  BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms> labelMatrix(labelMatrixStorage);
  nvMolKit::populateLabelMatrix<MaxTargetAtoms, MaxQueryAtoms>(target, query, labelMatrix);
}

template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms, int MaxBondsPerAtom>
__global__ void testGSISearchKernel(
  TargetMoleculeView                                                    target,
  QueryMoleculeView                                                     query,
  typename BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>::StorageType* labelMatrixStorage,
  PartialMatchT<MaxQueryAtoms>*                                         sharedPartials,
  int                                                                   maxPartials,
  PartialMatchT<MaxQueryAtoms>*                                         overflowA,
  PartialMatchT<MaxQueryAtoms>*                                         overflowB,
  int                                                                   maxOverflow,
  int*                                                                  matchCount,
  int*                                                                  reportedCount,
  int16_t*                                                              matchIndices,
  int                                                                   maxMatches,
  int                                                                   matchOffset,
  int                                                                   maxMatchesToFind,
  bool                                                                  countOnly,
  uint8_t*                                                              overflowFlag = nullptr) {
  BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms> labelMatrix(labelMatrixStorage);

  gsiBFSSearchGPU<MaxTargetAtoms, MaxQueryAtoms, MaxBondsPerAtom, SubstructOutputMode::StoreMatches>(target,
                                                                                                     query,
                                                                                                     labelMatrix,
                                                                                                     sharedPartials,
                                                                                                     maxPartials,
                                                                                                     overflowA,
                                                                                                     overflowB,
                                                                                                     maxOverflow,
                                                                                                     matchCount,
                                                                                                     reportedCount,
                                                                                                     matchIndices,
                                                                                                     maxMatches,
                                                                                                     matchOffset,
                                                                                                     {},
                                                                                                     maxMatchesToFind,
                                                                                                     countOnly,
                                                                                                     nullptr,
                                                                                                     overflowFlag);
}

template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms, int MaxBondsPerAtom>
__global__ void testGSIPaintKernel(
  TargetMoleculeView                                                    target,
  QueryMoleculeView                                                     query,
  typename BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>::StorageType* labelMatrixStorage,
  PartialMatchT<MaxQueryAtoms>*                                         sharedPartials,
  int                                                                   maxPartials,
  PartialMatchT<MaxQueryAtoms>*                                         overflowA,
  PartialMatchT<MaxQueryAtoms>*                                         overflowB,
  int                                                                   maxOverflow,
  int*                                                                  matchCount,
  int*                                                                  reportedCount,
  uint32_t*                                                             recursiveBits,
  int                                                                   patternId,
  int                                                                   maxTargetAtoms,
  int                                                                   outputPairIdx) {
  BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms> labelMatrix(labelMatrixStorage);
  nvMolKit::PaintModeParams                      paintParams{recursiveBits, patternId, maxTargetAtoms, outputPairIdx};

  gsiBFSSearchGPU<MaxTargetAtoms, MaxQueryAtoms, MaxBondsPerAtom, SubstructOutputMode::PaintBits>(target,
                                                                                                  query,
                                                                                                  labelMatrix,
                                                                                                  sharedPartials,
                                                                                                  maxPartials,
                                                                                                  overflowA,
                                                                                                  overflowB,
                                                                                                  maxOverflow,
                                                                                                  matchCount,
                                                                                                  reportedCount,
                                                                                                  nullptr,
                                                                                                  0,
                                                                                                  0,
                                                                                                  paintParams,
                                                                                                  -1,
                                                                                                  false);
}

// =============================================================================
// Test Fixtures
// =============================================================================

class SubstructAlgosTestBase : public ::testing::Test {
 protected:
  ScopedStream stream_;

  void SetUp() override {}

  template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
  void populateLabelMatrixOnDevice(const TargetMoleculeView&                                    target,
                                   const QueryMoleculeView&                                     query,
                                   BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>::StorageType* d_labelMatrix) {
    testPopulateLabelMatrixKernel<MaxTargetAtoms, MaxQueryAtoms, kTestMaxBondsPerAtom>
      <<<1, 128, 0, stream_.stream()>>>(target, query, d_labelMatrix);
    cudaCheckError(cudaStreamSynchronize(stream_.stream()));
  }
};

// =============================================================================
// VF2 Algorithm Tests
// =============================================================================

class VF2AlgoTest : public SubstructAlgosTestBase, public ::testing::WithParamInterface<AlgoTestParams> {};

TEST_P(VF2AlgoTest, SingleAtomMatch) {
  auto params = GetParam();

  SyntheticMoleculeBuilder targetBuilder(1);
  targetBuilder.setAtomicNum(0, 6);
  targetBuilder.setDegree(0, 0);

  SyntheticMoleculeBuilder queryBuilder(1);
  queryBuilder.setAtomicNum(0, 6);
  queryBuilder.setDegree(0, 0);

  auto target = targetBuilder.buildTargetView(stream_.stream());
  auto query  = queryBuilder.buildQueryView(stream_.stream());

  using LabelStorage = typename BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms>::StorageType;
  AsyncDeviceVector<LabelStorage> d_labelMatrix(1, stream_.stream());
  AsyncDeviceVector<int>          d_matchCount(1, stream_.stream());
  AsyncDeviceVector<int>          d_reportedCount(1, stream_.stream());
  AsyncDeviceVector<int16_t>      d_matchIndices(16, stream_.stream());

  d_matchCount.zero();
  d_reportedCount.zero();
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  LabelStorage h_labelMatrix;
  h_labelMatrix.clear();
  BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms> labelView(&h_labelMatrix);
  labelView.set(0, 0, true);

  cudaCheckError(cudaMemcpyAsync(d_labelMatrix.data(),
                                 &h_labelMatrix,
                                 sizeof(LabelStorage),
                                 cudaMemcpyHostToDevice,
                                 stream_.stream()));

  testVF2SearchKernel<kTestMaxTargetAtoms, kTestMaxQueryAtoms, kTestMaxBondsPerAtom>
    <<<1, 32, 0, stream_.stream()>>>(target,
                                     query,
                                     d_labelMatrix.data(),
                                     0,
                                     d_matchCount.data(),
                                     d_reportedCount.data(),
                                     d_matchIndices.data(),
                                     params.maxMatches,
                                     0,
                                     params.maxMatchesToFind,
                                     params.countOnly);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  int matchCount    = 0;
  int reportedCount = 0;
  cudaCheckError(cudaMemcpy(&matchCount, d_matchCount.data(), sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(&reportedCount, d_reportedCount.data(), sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(matchCount, 1);
  if (!params.countOnly && params.maxMatches > 0) {
    EXPECT_EQ(reportedCount, 1);
  }
}

TEST_P(VF2AlgoTest, SingleAtomNoMatch) {
  auto params = GetParam();

  SyntheticMoleculeBuilder targetBuilder(1);
  targetBuilder.setAtomicNum(0, 6);

  SyntheticMoleculeBuilder queryBuilder(1);
  queryBuilder.setAtomicNum(0, 7);

  auto target = targetBuilder.buildTargetView(stream_.stream());
  auto query  = queryBuilder.buildQueryView(stream_.stream());

  using LabelStorage = typename BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms>::StorageType;
  AsyncDeviceVector<LabelStorage> d_labelMatrix(1, stream_.stream());
  AsyncDeviceVector<int>          d_matchCount(1, stream_.stream());
  AsyncDeviceVector<int>          d_reportedCount(1, stream_.stream());
  AsyncDeviceVector<int16_t>      d_matchIndices(16, stream_.stream());

  d_matchCount.zero();
  d_reportedCount.zero();

  LabelStorage h_labelMatrix;
  h_labelMatrix.clear();

  cudaCheckError(cudaMemcpyAsync(d_labelMatrix.data(),
                                 &h_labelMatrix,
                                 sizeof(LabelStorage),
                                 cudaMemcpyHostToDevice,
                                 stream_.stream()));

  testVF2SearchKernel<kTestMaxTargetAtoms, kTestMaxQueryAtoms, kTestMaxBondsPerAtom>
    <<<1, 32, 0, stream_.stream()>>>(target,
                                     query,
                                     d_labelMatrix.data(),
                                     0,
                                     d_matchCount.data(),
                                     d_reportedCount.data(),
                                     d_matchIndices.data(),
                                     params.maxMatches,
                                     0,
                                     params.maxMatchesToFind,
                                     params.countOnly);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  int matchCount = 0;
  cudaCheckError(cudaMemcpy(&matchCount, d_matchCount.data(), sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(matchCount, 0);
}

TEST_P(VF2AlgoTest, TwoAtomChainMultipleMatches) {
  auto params = GetParam();

  SyntheticMoleculeBuilder targetBuilder(3);
  targetBuilder.buildLinearChain(6);

  SyntheticMoleculeBuilder queryBuilder(2);
  queryBuilder.buildLinearChain(6);

  auto target = targetBuilder.buildTargetView(stream_.stream());
  auto query  = queryBuilder.buildQueryView(stream_.stream());

  using LabelStorage = typename BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms>::StorageType;
  AsyncDeviceVector<LabelStorage> d_labelMatrix(1, stream_.stream());
  AsyncDeviceVector<int>          d_matchCount(1, stream_.stream());
  AsyncDeviceVector<int>          d_reportedCount(1, stream_.stream());
  AsyncDeviceVector<int16_t>      d_matchIndices(100, stream_.stream());

  d_matchCount.zero();
  d_reportedCount.zero();

  LabelStorage h_labelMatrix;
  h_labelMatrix.clear();
  BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms> labelView(&h_labelMatrix);
  for (int t = 0; t < 3; ++t) {
    for (int q = 0; q < 2; ++q) {
      labelView.set(t, q, true);
    }
  }

  cudaCheckError(cudaMemcpyAsync(d_labelMatrix.data(),
                                 &h_labelMatrix,
                                 sizeof(LabelStorage),
                                 cudaMemcpyHostToDevice,
                                 stream_.stream()));

  for (int startAtom = 0; startAtom < 3; ++startAtom) {
    testVF2SearchKernel<kTestMaxTargetAtoms, kTestMaxQueryAtoms, kTestMaxBondsPerAtom>
      <<<1, 32, 0, stream_.stream()>>>(target,
                                       query,
                                       d_labelMatrix.data(),
                                       startAtom,
                                       d_matchCount.data(),
                                       d_reportedCount.data(),
                                       d_matchIndices.data(),
                                       params.maxMatches,
                                       0,
                                       params.maxMatchesToFind,
                                       params.countOnly);
  }
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  int matchCount = 0;
  cudaCheckError(cudaMemcpy(&matchCount, d_matchCount.data(), sizeof(int), cudaMemcpyDeviceToHost));

  if (params.maxMatchesToFind == 1) {
    EXPECT_GE(matchCount, 1);
  } else {
    EXPECT_EQ(matchCount, 4);
  }
}

INSTANTIATE_TEST_SUITE_P(Modes,
                         VF2AlgoTest,
                         ::testing::Values(AlgoTestParams{false, 100, -1},
                                           AlgoTestParams{true, 0, -1},
                                           AlgoTestParams{false, 100, 1}));

TEST_F(SubstructAlgosTestBase, VF2ParamOutputBufferOverflow) {
  SyntheticMoleculeBuilder targetBuilder(4);
  targetBuilder.buildLinearChain(6);

  SyntheticMoleculeBuilder queryBuilder(2);
  queryBuilder.buildLinearChain(6);

  auto target = targetBuilder.buildTargetView(stream_.stream());
  auto query  = queryBuilder.buildQueryView(stream_.stream());

  using LabelStorage = typename BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms>::StorageType;
  AsyncDeviceVector<LabelStorage> d_labelMatrix(1, stream_.stream());
  AsyncDeviceVector<int>          d_matchCount(1, stream_.stream());
  AsyncDeviceVector<int>          d_reportedCount(1, stream_.stream());
  AsyncDeviceVector<int16_t>      d_matchIndices(4, stream_.stream());

  d_matchCount.zero();
  d_reportedCount.zero();

  LabelStorage h_labelMatrix;
  h_labelMatrix.clear();
  BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms> labelView(&h_labelMatrix);
  for (int t = 0; t < 4; ++t) {
    for (int q = 0; q < 2; ++q) {
      labelView.set(t, q, true);
    }
  }

  cudaCheckError(cudaMemcpyAsync(d_labelMatrix.data(),
                                 &h_labelMatrix,
                                 sizeof(LabelStorage),
                                 cudaMemcpyHostToDevice,
                                 stream_.stream()));

  int smallMaxMatches = 2;
  for (int startAtom = 0; startAtom < 4; ++startAtom) {
    testVF2SearchKernel<kTestMaxTargetAtoms, kTestMaxQueryAtoms, kTestMaxBondsPerAtom>
      <<<1, 32, 0, stream_.stream()>>>(target,
                                       query,
                                       d_labelMatrix.data(),
                                       startAtom,
                                       d_matchCount.data(),
                                       d_reportedCount.data(),
                                       d_matchIndices.data(),
                                       smallMaxMatches,
                                       0,
                                       -1,
                                       false);
  }
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  int matchCount    = 0;
  int reportedCount = 0;
  cudaCheckError(cudaMemcpy(&matchCount, d_matchCount.data(), sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(&reportedCount, d_reportedCount.data(), sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_GT(matchCount, reportedCount);
  EXPECT_EQ(reportedCount, smallMaxMatches);
}

// =============================================================================
// GSI BFS Algorithm Tests (Parameterized)
// =============================================================================

class GSIAlgoTest : public SubstructAlgosTestBase, public ::testing::WithParamInterface<AlgoTestParams> {};

TEST_P(GSIAlgoTest, SingleAtomQuery) {
  auto params = GetParam();

  SyntheticMoleculeBuilder targetBuilder(3);
  for (int i = 0; i < 3; ++i) {
    targetBuilder.setAtomicNum(i, 6);
    targetBuilder.setDegree(i, 0);
  }

  SyntheticMoleculeBuilder queryBuilder(1);
  queryBuilder.setAtomicNum(0, 6);
  queryBuilder.setDegree(0, 0);

  auto target = targetBuilder.buildTargetView(stream_.stream());
  auto query  = queryBuilder.buildQueryView(stream_.stream());

  using LabelStorage = typename BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms>::StorageType;
  AsyncDeviceVector<LabelStorage>                      d_labelMatrix(1, stream_.stream());
  AsyncDeviceVector<int>                               d_matchCount(1, stream_.stream());
  AsyncDeviceVector<int>                               d_reportedCount(1, stream_.stream());
  AsyncDeviceVector<int16_t>                           d_matchIndices(100, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_sharedPartials(64, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowA(32, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowB(32, stream_.stream());

  d_matchCount.zero();
  d_reportedCount.zero();

  LabelStorage h_labelMatrix;
  h_labelMatrix.clear();
  BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms> labelView(&h_labelMatrix);
  for (int t = 0; t < 3; ++t) {
    labelView.set(t, 0, true);
  }

  cudaCheckError(cudaMemcpyAsync(d_labelMatrix.data(),
                                 &h_labelMatrix,
                                 sizeof(LabelStorage),
                                 cudaMemcpyHostToDevice,
                                 stream_.stream()));

  testGSISearchKernel<kTestMaxTargetAtoms, kTestMaxQueryAtoms, kTestMaxBondsPerAtom>
    <<<1, 128, 0, stream_.stream()>>>(target,
                                      query,
                                      d_labelMatrix.data(),
                                      d_sharedPartials.data(),
                                      32,
                                      d_overflowA.data(),
                                      d_overflowB.data(),
                                      16,
                                      d_matchCount.data(),
                                      d_reportedCount.data(),
                                      d_matchIndices.data(),
                                      params.maxMatches,
                                      0,
                                      params.maxMatchesToFind,
                                      params.countOnly);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  int matchCount = 0;
  cudaCheckError(cudaMemcpy(&matchCount, d_matchCount.data(), sizeof(int), cudaMemcpyDeviceToHost));

  if (params.maxMatchesToFind == 1) {
    EXPECT_GE(matchCount, 1);
  } else {
    EXPECT_EQ(matchCount, 3);
  }
}

TEST_P(GSIAlgoTest, TwoAtomChain) {
  auto params = GetParam();

  SyntheticMoleculeBuilder targetBuilder(3);
  targetBuilder.buildLinearChain(6);

  SyntheticMoleculeBuilder queryBuilder(2);
  queryBuilder.buildLinearChain(6);

  auto target = targetBuilder.buildTargetView(stream_.stream());
  auto query  = queryBuilder.buildQueryView(stream_.stream());

  using LabelStorage = typename BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms>::StorageType;
  AsyncDeviceVector<LabelStorage>                      d_labelMatrix(1, stream_.stream());
  AsyncDeviceVector<int>                               d_matchCount(1, stream_.stream());
  AsyncDeviceVector<int>                               d_reportedCount(1, stream_.stream());
  AsyncDeviceVector<int16_t>                           d_matchIndices(100, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_sharedPartials(64, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowA(32, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowB(32, stream_.stream());

  d_matchCount.zero();
  d_reportedCount.zero();

  LabelStorage h_labelMatrix;
  h_labelMatrix.clear();
  BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms> labelView(&h_labelMatrix);
  for (int t = 0; t < 3; ++t) {
    for (int q = 0; q < 2; ++q) {
      labelView.set(t, q, true);
    }
  }

  cudaCheckError(cudaMemcpyAsync(d_labelMatrix.data(),
                                 &h_labelMatrix,
                                 sizeof(LabelStorage),
                                 cudaMemcpyHostToDevice,
                                 stream_.stream()));

  testGSISearchKernel<kTestMaxTargetAtoms, kTestMaxQueryAtoms, kTestMaxBondsPerAtom>
    <<<1, 128, 0, stream_.stream()>>>(target,
                                      query,
                                      d_labelMatrix.data(),
                                      d_sharedPartials.data(),
                                      32,
                                      d_overflowA.data(),
                                      d_overflowB.data(),
                                      16,
                                      d_matchCount.data(),
                                      d_reportedCount.data(),
                                      d_matchIndices.data(),
                                      params.maxMatches,
                                      0,
                                      params.maxMatchesToFind,
                                      params.countOnly);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  int matchCount = 0;
  cudaCheckError(cudaMemcpy(&matchCount, d_matchCount.data(), sizeof(int), cudaMemcpyDeviceToHost));

  if (params.maxMatchesToFind == 1) {
    EXPECT_GE(matchCount, 1);
  } else {
    EXPECT_EQ(matchCount, 4);
  }
}

TEST_P(GSIAlgoTest, NoMatch) {
  auto params = GetParam();

  SyntheticMoleculeBuilder targetBuilder(3);
  for (int i = 0; i < 3; ++i) {
    targetBuilder.setAtomicNum(i, 6);
  }

  SyntheticMoleculeBuilder queryBuilder(1);
  queryBuilder.setAtomicNum(0, 7);

  auto target = targetBuilder.buildTargetView(stream_.stream());
  auto query  = queryBuilder.buildQueryView(stream_.stream());

  using LabelStorage = typename BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms>::StorageType;
  AsyncDeviceVector<LabelStorage>                      d_labelMatrix(1, stream_.stream());
  AsyncDeviceVector<int>                               d_matchCount(1, stream_.stream());
  AsyncDeviceVector<int>                               d_reportedCount(1, stream_.stream());
  AsyncDeviceVector<int16_t>                           d_matchIndices(16, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_sharedPartials(64, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowA(32, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowB(32, stream_.stream());

  d_matchCount.zero();
  d_reportedCount.zero();

  LabelStorage h_labelMatrix;
  h_labelMatrix.clear();

  cudaCheckError(cudaMemcpyAsync(d_labelMatrix.data(),
                                 &h_labelMatrix,
                                 sizeof(LabelStorage),
                                 cudaMemcpyHostToDevice,
                                 stream_.stream()));

  testGSISearchKernel<kTestMaxTargetAtoms, kTestMaxQueryAtoms, kTestMaxBondsPerAtom>
    <<<1, 128, 0, stream_.stream()>>>(target,
                                      query,
                                      d_labelMatrix.data(),
                                      d_sharedPartials.data(),
                                      32,
                                      d_overflowA.data(),
                                      d_overflowB.data(),
                                      16,
                                      d_matchCount.data(),
                                      d_reportedCount.data(),
                                      d_matchIndices.data(),
                                      params.maxMatches,
                                      0,
                                      params.maxMatchesToFind,
                                      params.countOnly);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  int matchCount = 0;
  cudaCheckError(cudaMemcpy(&matchCount, d_matchCount.data(), sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(matchCount, 0);
}

INSTANTIATE_TEST_SUITE_P(Modes,
                         GSIAlgoTest,
                         ::testing::Values(AlgoTestParams{false, 100, -1},
                                           AlgoTestParams{true, 0, -1},
                                           AlgoTestParams{false, 100, 1}));

// =============================================================================
// GSI Paint Mode Test
// =============================================================================

TEST_F(SubstructAlgosTestBase, GSIPaintMode) {
  SyntheticMoleculeBuilder targetBuilder(3);
  for (int i = 0; i < 3; ++i) {
    targetBuilder.setAtomicNum(i, 6);
    targetBuilder.setDegree(i, 0);
  }

  SyntheticMoleculeBuilder queryBuilder(1);
  queryBuilder.setAtomicNum(0, 6);
  queryBuilder.setDegree(0, 0);

  auto target = targetBuilder.buildTargetView(stream_.stream());
  auto query  = queryBuilder.buildQueryView(stream_.stream());

  using LabelStorage = typename BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms>::StorageType;
  AsyncDeviceVector<LabelStorage>                      d_labelMatrix(1, stream_.stream());
  AsyncDeviceVector<int>                               d_matchCount(1, stream_.stream());
  AsyncDeviceVector<int>                               d_reportedCount(1, stream_.stream());
  AsyncDeviceVector<uint32_t>                          d_recursiveBits(kTestMaxTargetAtoms, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_sharedPartials(64, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowA(32, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowB(32, stream_.stream());

  d_matchCount.zero();
  d_reportedCount.zero();
  d_recursiveBits.zero();

  LabelStorage h_labelMatrix;
  h_labelMatrix.clear();
  BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms> labelView(&h_labelMatrix);
  for (int t = 0; t < 3; ++t) {
    labelView.set(t, 0, true);
  }

  cudaCheckError(cudaMemcpyAsync(d_labelMatrix.data(),
                                 &h_labelMatrix,
                                 sizeof(LabelStorage),
                                 cudaMemcpyHostToDevice,
                                 stream_.stream()));

  int patternId = 5;
  testGSIPaintKernel<kTestMaxTargetAtoms, kTestMaxQueryAtoms, kTestMaxBondsPerAtom>
    <<<1, 128, 0, stream_.stream()>>>(target,
                                      query,
                                      d_labelMatrix.data(),
                                      d_sharedPartials.data(),
                                      32,
                                      d_overflowA.data(),
                                      d_overflowB.data(),
                                      16,
                                      d_matchCount.data(),
                                      d_reportedCount.data(),
                                      d_recursiveBits.data(),
                                      patternId,
                                      kTestMaxTargetAtoms,
                                      0);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  int matchCount    = 0;
  int reportedCount = 0;
  cudaCheckError(cudaMemcpy(&matchCount, d_matchCount.data(), sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(&reportedCount, d_reportedCount.data(), sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(matchCount, 3);
  EXPECT_EQ(reportedCount, 3);

  std::vector<uint32_t> h_recursiveBits(kTestMaxTargetAtoms);
  cudaCheckError(cudaMemcpy(h_recursiveBits.data(),
                            d_recursiveBits.data(),
                            kTestMaxTargetAtoms * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));

  uint32_t expectedBit = 1u << patternId;
  for (int t = 0; t < 3; ++t) {
    EXPECT_EQ(h_recursiveBits[t] & expectedBit, expectedBit)
      << "Target atom " << t << " should have bit " << patternId << " set";
  }
}

// =============================================================================
// Realistic Molecule Tests
// =============================================================================

/**
 * @brief Test finding nitro groups in a nitrobenzene-like structure.
 *
 * Target: Simplified nitrobenzene (C6H5NO2)
 *   - Benzene ring: C0-C1-C2-C3-C4-C5 (aromatic carbons)
 *   - Nitro group attached to C2: N6 bonded to O7 and O8
 *
 * Query: Nitro pattern (NO2)
 *   - N0 bonded to O1 and O2
 *
 * Expected: 1 match (the single nitro group)
 */
TEST_F(SubstructAlgosTestBase, NitroGroupInNitrobenzene) {
  // Build nitrobenzene-like target: 6 carbons + N + 2 oxygens = 9 atoms
  SyntheticMoleculeBuilder targetBuilder(9);

  // Carbons in benzene ring (atomic number 6)
  for (int i = 0; i < 6; ++i) {
    targetBuilder.setAtomicNum(i, 6);
  }
  // Nitrogen (atomic number 7)
  targetBuilder.setAtomicNum(6, 7);
  // Oxygens (atomic number 8)
  targetBuilder.setAtomicNum(7, 8);
  targetBuilder.setAtomicNum(8, 8);

  // Set degrees: ring carbons have 2-3 bonds, N has 3 bonds, O has 1 bond
  targetBuilder.setDegree(0, 2);  // C0
  targetBuilder.setDegree(1, 2);  // C1
  targetBuilder.setDegree(2, 3);  // C2 (attached to nitro)
  targetBuilder.setDegree(3, 2);  // C3
  targetBuilder.setDegree(4, 2);  // C4
  targetBuilder.setDegree(5, 2);  // C5
  targetBuilder.setDegree(6, 3);  // N (bonded to C2, O7, O8)
  targetBuilder.setDegree(7, 1);  // O7
  targetBuilder.setDegree(8, 1);  // O8

  // Benzene ring bonds (simplified as single bonds for this test)
  targetBuilder.addTargetBond(0, 1);
  targetBuilder.addTargetBond(1, 0);
  targetBuilder.addTargetBond(1, 2);
  targetBuilder.addTargetBond(2, 1);
  targetBuilder.addTargetBond(2, 3);
  targetBuilder.addTargetBond(3, 2);
  targetBuilder.addTargetBond(3, 4);
  targetBuilder.addTargetBond(4, 3);
  targetBuilder.addTargetBond(4, 5);
  targetBuilder.addTargetBond(5, 4);
  targetBuilder.addTargetBond(5, 0);
  targetBuilder.addTargetBond(0, 5);

  // Nitro group bonds: C2-N6, N6-O7, N6-O8
  targetBuilder.addTargetBond(2, 6);
  targetBuilder.addTargetBond(6, 2);
  targetBuilder.addTargetBond(6, 7);
  targetBuilder.addTargetBond(7, 6);
  targetBuilder.addTargetBond(6, 8);
  targetBuilder.addTargetBond(8, 6);

  // Build nitro query: N bonded to 2 oxygens (3 atoms)
  SyntheticMoleculeBuilder queryBuilder(3);
  queryBuilder.setAtomicNum(0, 7);  // N
  queryBuilder.setAtomicNum(1, 8);  // O
  queryBuilder.setAtomicNum(2, 8);  // O

  queryBuilder.setDegree(0, 2);  // N bonded to 2 oxygens (in query context)
  queryBuilder.setDegree(1, 1);  // O
  queryBuilder.setDegree(2, 1);  // O

  queryBuilder.addQueryBond(0, 1);
  queryBuilder.addQueryBond(1, 0);
  queryBuilder.addQueryBond(0, 2);
  queryBuilder.addQueryBond(2, 0);

  auto target = targetBuilder.buildTargetView(stream_.stream());
  auto query  = queryBuilder.buildQueryView(stream_.stream());

  using LabelStorage = typename BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms>::StorageType;
  AsyncDeviceVector<LabelStorage>                      d_labelMatrix(1, stream_.stream());
  AsyncDeviceVector<int>                               d_matchCount(1, stream_.stream());
  AsyncDeviceVector<int>                               d_reportedCount(1, stream_.stream());
  AsyncDeviceVector<int16_t>                           d_matchIndices(100, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_sharedPartials(64, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowA(32, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowB(32, stream_.stream());

  d_matchCount.zero();
  d_reportedCount.zero();

  // Build label matrix: N matches N, O matches O, C doesn't match N or O
  LabelStorage h_labelMatrix;
  h_labelMatrix.clear();
  BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms> labelView(&h_labelMatrix);

  // Query atom 0 is N (atomic num 7) - only target atom 6 (N) matches
  labelView.set(6, 0, true);
  // Query atoms 1,2 are O (atomic num 8) - target atoms 7,8 (O) match both
  labelView.set(7, 1, true);
  labelView.set(8, 1, true);
  labelView.set(7, 2, true);
  labelView.set(8, 2, true);

  cudaCheckError(cudaMemcpyAsync(d_labelMatrix.data(),
                                 &h_labelMatrix,
                                 sizeof(LabelStorage),
                                 cudaMemcpyHostToDevice,
                                 stream_.stream()));

  testGSISearchKernel<kTestMaxTargetAtoms, kTestMaxQueryAtoms, kTestMaxBondsPerAtom>
    <<<1, 128, 0, stream_.stream()>>>(target,
                                      query,
                                      d_labelMatrix.data(),
                                      d_sharedPartials.data(),
                                      32,
                                      d_overflowA.data(),
                                      d_overflowB.data(),
                                      16,
                                      d_matchCount.data(),
                                      d_reportedCount.data(),
                                      d_matchIndices.data(),
                                      100,
                                      0,
                                      -1,
                                      false);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  int matchCount = 0;
  cudaCheckError(cudaMemcpy(&matchCount, d_matchCount.data(), sizeof(int), cudaMemcpyDeviceToHost));

  // Should find 2 matches: N6->O7,O8 and N6->O8,O7 (the two oxygen permutations)
  EXPECT_EQ(matchCount, 2);
}

/**
 * @brief Test finding multiple nitro groups (like in TNT).
 *
 * Target: Simplified TNT-like structure with 3 nitro groups
 *   - Central structure with 3 nitrogens, each bonded to 2 oxygens
 *
 * Query: Nitro pattern (NO2)
 *
 * Expected: 6 matches (2 permutations per nitro group x 3 groups)
 */
TEST_F(SubstructAlgosTestBase, MultipleNitroGroupsLikeTNT) {
  // Build simplified structure: 3 nitrogens, each bonded to 2 oxygens = 9 atoms
  // N0-O1,O2  N3-O4,O5  N6-O7,O8
  SyntheticMoleculeBuilder targetBuilder(9);

  // Three nitrogens
  targetBuilder.setAtomicNum(0, 7);
  targetBuilder.setAtomicNum(3, 7);
  targetBuilder.setAtomicNum(6, 7);

  // Six oxygens (2 per nitrogen)
  targetBuilder.setAtomicNum(1, 8);
  targetBuilder.setAtomicNum(2, 8);
  targetBuilder.setAtomicNum(4, 8);
  targetBuilder.setAtomicNum(5, 8);
  targetBuilder.setAtomicNum(7, 8);
  targetBuilder.setAtomicNum(8, 8);

  // Set degrees
  targetBuilder.setDegree(0, 2);  // N0
  targetBuilder.setDegree(1, 1);  // O1
  targetBuilder.setDegree(2, 1);  // O2
  targetBuilder.setDegree(3, 2);  // N3
  targetBuilder.setDegree(4, 1);  // O4
  targetBuilder.setDegree(5, 1);  // O5
  targetBuilder.setDegree(6, 2);  // N6
  targetBuilder.setDegree(7, 1);  // O7
  targetBuilder.setDegree(8, 1);  // O8

  // Nitro group bonds
  targetBuilder.addTargetBond(0, 1);
  targetBuilder.addTargetBond(1, 0);
  targetBuilder.addTargetBond(0, 2);
  targetBuilder.addTargetBond(2, 0);

  targetBuilder.addTargetBond(3, 4);
  targetBuilder.addTargetBond(4, 3);
  targetBuilder.addTargetBond(3, 5);
  targetBuilder.addTargetBond(5, 3);

  targetBuilder.addTargetBond(6, 7);
  targetBuilder.addTargetBond(7, 6);
  targetBuilder.addTargetBond(6, 8);
  targetBuilder.addTargetBond(8, 6);

  // Build nitro query: N bonded to 2 oxygens
  SyntheticMoleculeBuilder queryBuilder(3);
  queryBuilder.setAtomicNum(0, 7);  // N
  queryBuilder.setAtomicNum(1, 8);  // O
  queryBuilder.setAtomicNum(2, 8);  // O

  queryBuilder.setDegree(0, 2);
  queryBuilder.setDegree(1, 1);
  queryBuilder.setDegree(2, 1);

  queryBuilder.addQueryBond(0, 1);
  queryBuilder.addQueryBond(1, 0);
  queryBuilder.addQueryBond(0, 2);
  queryBuilder.addQueryBond(2, 0);

  auto target = targetBuilder.buildTargetView(stream_.stream());
  auto query  = queryBuilder.buildQueryView(stream_.stream());

  using LabelStorage = typename BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms>::StorageType;
  AsyncDeviceVector<LabelStorage>                      d_labelMatrix(1, stream_.stream());
  AsyncDeviceVector<int>                               d_matchCount(1, stream_.stream());
  AsyncDeviceVector<int>                               d_reportedCount(1, stream_.stream());
  AsyncDeviceVector<int16_t>                           d_matchIndices(100, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_sharedPartials(64, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowA(32, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowB(32, stream_.stream());

  d_matchCount.zero();
  d_reportedCount.zero();

  // Build label matrix
  LabelStorage h_labelMatrix;
  h_labelMatrix.clear();
  BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms> labelView(&h_labelMatrix);

  // Query atom 0 is N - target atoms 0,3,6 are N
  labelView.set(0, 0, true);
  labelView.set(3, 0, true);
  labelView.set(6, 0, true);

  // Query atoms 1,2 are O - target atoms 1,2,4,5,7,8 are O
  for (int o : {1, 2, 4, 5, 7, 8}) {
    labelView.set(o, 1, true);
    labelView.set(o, 2, true);
  }

  cudaCheckError(cudaMemcpyAsync(d_labelMatrix.data(),
                                 &h_labelMatrix,
                                 sizeof(LabelStorage),
                                 cudaMemcpyHostToDevice,
                                 stream_.stream()));

  testGSISearchKernel<kTestMaxTargetAtoms, kTestMaxQueryAtoms, kTestMaxBondsPerAtom>
    <<<1, 128, 0, stream_.stream()>>>(target,
                                      query,
                                      d_labelMatrix.data(),
                                      d_sharedPartials.data(),
                                      32,
                                      d_overflowA.data(),
                                      d_overflowB.data(),
                                      16,
                                      d_matchCount.data(),
                                      d_reportedCount.data(),
                                      d_matchIndices.data(),
                                      100,
                                      0,
                                      -1,
                                      false);
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  int matchCount = 0;
  cudaCheckError(cudaMemcpy(&matchCount, d_matchCount.data(), sizeof(int), cudaMemcpyDeviceToHost));

  // 3 nitro groups x 2 oxygen permutations = 6 matches
  EXPECT_EQ(matchCount, 6);
}

// =============================================================================
// Overflow Stress Tests
// =============================================================================

/**
 * @brief Test partials spilling from shared to global overflow buffer.
 *
 * Uses a 20-atom target with a 16-atom query (stride = 16 = sizeof(PartialMatchT)).
 * With maxPartials=4 for shared, effective capacity is exactly 4 slots.
 * At level 0, all 20 atoms match, exceeding 4 slots and spilling to overflow.
 * With maxOverflow=128, total capacity is 132, enough for BFS branching.
 * No overflow flag should be set.
 */
TEST_F(SubstructAlgosTestBase, GSIPartialsSpillToOverflow) {
  constexpr int numTargetAtoms = 20;
  constexpr int numQueryAtoms  = 16;

  SyntheticMoleculeBuilder targetBuilder(numTargetAtoms);
  targetBuilder.buildLinearChain(6);

  SyntheticMoleculeBuilder queryBuilder(numQueryAtoms);
  queryBuilder.buildLinearChain(6);

  auto target = targetBuilder.buildTargetView(stream_.stream());
  auto query  = queryBuilder.buildQueryView(stream_.stream());

  using LabelStorage = typename BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms>::StorageType;
  AsyncDeviceVector<LabelStorage>                      d_labelMatrix(1, stream_.stream());
  AsyncDeviceVector<int>                               d_matchCount(1, stream_.stream());
  AsyncDeviceVector<int>                               d_reportedCount(1, stream_.stream());
  AsyncDeviceVector<int16_t>                           d_matchIndices(500, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_sharedPartials(8, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowA(256, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowB(256, stream_.stream());
  AsyncDeviceVector<uint8_t>                           d_overflowFlag(1, stream_.stream());

  d_matchCount.zero();
  d_reportedCount.zero();
  d_overflowFlag.zero();

  LabelStorage h_labelMatrix;
  h_labelMatrix.clear();
  BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms> labelView(&h_labelMatrix);
  for (int t = 0; t < numTargetAtoms; ++t) {
    for (int q = 0; q < numQueryAtoms; ++q) {
      labelView.set(t, q, true);
    }
  }

  cudaCheckError(cudaMemcpyAsync(d_labelMatrix.data(),
                                 &h_labelMatrix,
                                 sizeof(LabelStorage),
                                 cudaMemcpyHostToDevice,
                                 stream_.stream()));

  // Small shared (4 effective slots), large overflow (128 slots each)
  // Total capacity = 4 + 128 = 132, enough for BFS branching at all levels
  testGSISearchKernel<kTestMaxTargetAtoms, kTestMaxQueryAtoms, kTestMaxBondsPerAtom>
    <<<1, 128, 0, stream_.stream()>>>(target,
                                      query,
                                      d_labelMatrix.data(),
                                      d_sharedPartials.data(),
                                      4,
                                      d_overflowA.data(),
                                      d_overflowB.data(),
                                      128,
                                      d_matchCount.data(),
                                      d_reportedCount.data(),
                                      d_matchIndices.data(),
                                      500,
                                      0,
                                      -1,
                                      false,
                                      d_overflowFlag.data());
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  int     matchCount   = 0;
  uint8_t overflowFlag = 0;
  cudaCheckError(cudaMemcpy(&matchCount, d_matchCount.data(), sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(&overflowFlag, d_overflowFlag.data(), sizeof(uint8_t), cudaMemcpyDeviceToHost));

  // 20-atom chain with 16-atom query: 10 matches (5 positions x 2 directions)
  EXPECT_EQ(matchCount, 10);
  // No overflow because we had enough capacity
  EXPECT_EQ(overflowFlag, 0);
}

/**
 * @brief Test partials overflowing past all buffers - overflow flag must be set.
 *
 * Uses a 32-atom target with a 16-atom query (stride = 16 = sizeof(PartialMatchT)).
 * With maxPartials=4 and maxOverflow=4, total capacity is only 8 slots.
 * At level 0, all 32 atoms match, vastly exceeding 8 slots.
 * Overflow flag MUST be set.
 */
TEST_F(SubstructAlgosTestBase, GSIPartialsExhausted) {
  constexpr int numTargetAtoms = 32;
  constexpr int numQueryAtoms  = 16;

  SyntheticMoleculeBuilder targetBuilder(numTargetAtoms);
  targetBuilder.buildLinearChain(6);

  SyntheticMoleculeBuilder queryBuilder(numQueryAtoms);
  queryBuilder.buildLinearChain(6);

  auto target = targetBuilder.buildTargetView(stream_.stream());
  auto query  = queryBuilder.buildQueryView(stream_.stream());

  using LabelStorage = typename BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms>::StorageType;
  AsyncDeviceVector<LabelStorage>                      d_labelMatrix(1, stream_.stream());
  AsyncDeviceVector<int>                               d_matchCount(1, stream_.stream());
  AsyncDeviceVector<int>                               d_reportedCount(1, stream_.stream());
  AsyncDeviceVector<int16_t>                           d_matchIndices(500, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_sharedPartials(8, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowA(8, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowB(8, stream_.stream());
  AsyncDeviceVector<uint8_t>                           d_overflowFlag(1, stream_.stream());

  d_matchCount.zero();
  d_reportedCount.zero();
  d_overflowFlag.zero();

  LabelStorage h_labelMatrix;
  h_labelMatrix.clear();
  BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms> labelView(&h_labelMatrix);
  for (int t = 0; t < numTargetAtoms; ++t) {
    for (int q = 0; q < numQueryAtoms; ++q) {
      labelView.set(t, q, true);
    }
  }

  cudaCheckError(cudaMemcpyAsync(d_labelMatrix.data(),
                                 &h_labelMatrix,
                                 sizeof(LabelStorage),
                                 cudaMemcpyHostToDevice,
                                 stream_.stream()));

  // Very small buffers: 4 shared + 4 overflow = 8 total
  // 30 candidates at level 0 will overflow
  testGSISearchKernel<kTestMaxTargetAtoms, kTestMaxQueryAtoms, kTestMaxBondsPerAtom>
    <<<1, 128, 0, stream_.stream()>>>(target,
                                      query,
                                      d_labelMatrix.data(),
                                      d_sharedPartials.data(),
                                      4,
                                      d_overflowA.data(),
                                      d_overflowB.data(),
                                      4,
                                      d_matchCount.data(),
                                      d_reportedCount.data(),
                                      d_matchIndices.data(),
                                      500,
                                      0,
                                      -1,
                                      false,
                                      d_overflowFlag.data());
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  int     matchCount   = 0;
  uint8_t overflowFlag = 0;
  cudaCheckError(cudaMemcpy(&matchCount, d_matchCount.data(), sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(&overflowFlag, d_overflowFlag.data(), sizeof(uint8_t), cudaMemcpyDeviceToHost));

  // May find some matches from the slots that fit
  EXPECT_GE(matchCount, 0);
  // Overflow flag MUST be set - buffers were exhausted
  EXPECT_EQ(overflowFlag, 1);
}

/**
 * @brief Test VF2 output buffer overflow detection.
 *
 * Uses a 20-atom target with a 10-atom query. With small maxMatches, the algorithm
 * finds more matches than it can store, demonstrating output buffer overflow.
 * VF2 doesn't have partial buffer overflow (uses stack-based DFS), only output overflow.
 */
TEST_F(SubstructAlgosTestBase, VF2OutputBufferOverflow) {
  constexpr int numTargetAtoms = 20;
  constexpr int numQueryAtoms  = 10;

  SyntheticMoleculeBuilder targetBuilder(numTargetAtoms);
  targetBuilder.buildLinearChain(6);

  SyntheticMoleculeBuilder queryBuilder(numQueryAtoms);
  queryBuilder.buildLinearChain(6);

  auto target = targetBuilder.buildTargetView(stream_.stream());
  auto query  = queryBuilder.buildQueryView(stream_.stream());

  using LabelStorage = typename BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms>::StorageType;
  AsyncDeviceVector<LabelStorage> d_labelMatrix(1, stream_.stream());
  AsyncDeviceVector<int>          d_matchCount(1, stream_.stream());
  AsyncDeviceVector<int>          d_reportedCount(1, stream_.stream());
  AsyncDeviceVector<int16_t>      d_matchIndices(100, stream_.stream());

  d_matchCount.zero();
  d_reportedCount.zero();

  LabelStorage h_labelMatrix;
  h_labelMatrix.clear();
  BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms> labelView(&h_labelMatrix);
  for (int t = 0; t < numTargetAtoms; ++t) {
    for (int q = 0; q < numQueryAtoms; ++q) {
      labelView.set(t, q, true);
    }
  }

  cudaCheckError(cudaMemcpyAsync(d_labelMatrix.data(),
                                 &h_labelMatrix,
                                 sizeof(LabelStorage),
                                 cudaMemcpyHostToDevice,
                                 stream_.stream()));

  // Limit output to only 3 matches
  int smallMaxMatches = 3;
  for (int startAtom = 0; startAtom < numTargetAtoms; ++startAtom) {
    testVF2SearchKernel<kTestMaxTargetAtoms, kTestMaxQueryAtoms, kTestMaxBondsPerAtom>
      <<<1, 32, 0, stream_.stream()>>>(target,
                                       query,
                                       d_labelMatrix.data(),
                                       startAtom,
                                       d_matchCount.data(),
                                       d_reportedCount.data(),
                                       d_matchIndices.data(),
                                       smallMaxMatches,
                                       0,
                                       -1,
                                       false);
  }
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  int matchCount    = 0;
  int reportedCount = 0;
  cudaCheckError(cudaMemcpy(&matchCount, d_matchCount.data(), sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(&reportedCount, d_reportedCount.data(), sizeof(int), cudaMemcpyDeviceToHost));

  // 20-atom chain with 10-atom query: 22 matches (11 positions x 2 directions)
  EXPECT_EQ(matchCount, 22);
  EXPECT_EQ(reportedCount, smallMaxMatches);
  EXPECT_GT(matchCount, reportedCount);
}

/**
 * @brief Test GSI output buffer overflow (more matches found than can be stored).
 *
 * Uses a 24-atom target with a 16-atom query. With ample partial buffers but
 * limited maxMatches, the algorithm finds more matches than it can store.
 * The overflow flag should NOT be set (partial buffers are fine), but
 * matchCount > reportedCount indicates output overflow.
 */
TEST_F(SubstructAlgosTestBase, GSIOutputBufferOverflow) {
  constexpr int numTargetAtoms = 24;
  constexpr int numQueryAtoms  = 16;

  SyntheticMoleculeBuilder targetBuilder(numTargetAtoms);
  targetBuilder.buildLinearChain(6);

  SyntheticMoleculeBuilder queryBuilder(numQueryAtoms);
  queryBuilder.buildLinearChain(6);

  auto target = targetBuilder.buildTargetView(stream_.stream());
  auto query  = queryBuilder.buildQueryView(stream_.stream());

  using LabelStorage = typename BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms>::StorageType;
  AsyncDeviceVector<LabelStorage>                      d_labelMatrix(1, stream_.stream());
  AsyncDeviceVector<int>                               d_matchCount(1, stream_.stream());
  AsyncDeviceVector<int>                               d_reportedCount(1, stream_.stream());
  AsyncDeviceVector<int16_t>                           d_matchIndices(100, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_sharedPartials(64, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowA(64, stream_.stream());
  AsyncDeviceVector<PartialMatchT<kTestMaxQueryAtoms>> d_overflowB(64, stream_.stream());
  AsyncDeviceVector<uint8_t>                           d_overflowFlag(1, stream_.stream());

  d_matchCount.zero();
  d_reportedCount.zero();
  d_overflowFlag.zero();

  LabelStorage h_labelMatrix;
  h_labelMatrix.clear();
  BitMatrix2DView<kTestMaxTargetAtoms, kTestMaxQueryAtoms> labelView(&h_labelMatrix);
  for (int t = 0; t < numTargetAtoms; ++t) {
    for (int q = 0; q < numQueryAtoms; ++q) {
      labelView.set(t, q, true);
    }
  }

  cudaCheckError(cudaMemcpyAsync(d_labelMatrix.data(),
                                 &h_labelMatrix,
                                 sizeof(LabelStorage),
                                 cudaMemcpyHostToDevice,
                                 stream_.stream()));

  // Ample partial buffers (32 each), but limit output to 3 matches
  int smallMaxMatches = 3;
  testGSISearchKernel<kTestMaxTargetAtoms, kTestMaxQueryAtoms, kTestMaxBondsPerAtom>
    <<<1, 128, 0, stream_.stream()>>>(target,
                                      query,
                                      d_labelMatrix.data(),
                                      d_sharedPartials.data(),
                                      32,
                                      d_overflowA.data(),
                                      d_overflowB.data(),
                                      32,
                                      d_matchCount.data(),
                                      d_reportedCount.data(),
                                      d_matchIndices.data(),
                                      smallMaxMatches,
                                      0,
                                      -1,
                                      false,
                                      d_overflowFlag.data());
  cudaCheckError(cudaStreamSynchronize(stream_.stream()));

  int     matchCount    = 0;
  int     reportedCount = 0;
  uint8_t overflowFlag  = 0;
  cudaCheckError(cudaMemcpy(&matchCount, d_matchCount.data(), sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(&reportedCount, d_reportedCount.data(), sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(&overflowFlag, d_overflowFlag.data(), sizeof(uint8_t), cudaMemcpyDeviceToHost));

  // 24-atom chain with 16-atom query: 18 matches (9 positions x 2 directions)
  EXPECT_EQ(matchCount, 18);
  EXPECT_EQ(reportedCount, smallMaxMatches);
  EXPECT_GT(matchCount, reportedCount);
  // No partial buffer overflow - flag should be clear
  EXPECT_EQ(overflowFlag, 0);
}

}  // namespace
