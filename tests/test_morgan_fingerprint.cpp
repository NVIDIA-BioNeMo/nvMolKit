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

#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <tuple>

#include "src/morgan_fingerprint.h"
#include "src/testutils/mol_data.h"
#include "src/utils/rdkit_ownership_wrap.h"

namespace {

using ::nvMolKit::testing::loadNChemblMolecules;
using ::nvMolKit::testing::makeMolsView;

const std::array<std::string, 10> rdkitTestSmiles = {
  "C[C@@H]1CCC[C@H](C)[C@H]1C",
  "N[C@@]1(C[C@H]([18F])C1)C(=O)O",
  "CC(C)CCCC[C@@H]1C[C@H](/C=C/[C@]2(C)CC[C@H](O)CC2)[C@@H](O)[C@H]1O",
  "COC(=O)/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(\\C)/C=C/C(=O)[O-]",
  "[O:1]=[C:2]([CH2:3][C:4]1=[CH:5][CH:6]=[CH:7][CH:8]=[CH:9]1)[NH2:10]",
  "Cl[C@H]1[C@@H](Cl)[C@H](Cl)[C@@H](Cl)[C@H](Cl)[C@@H]1Cl",
  "O=S(=O)(NC[C@H]1CC[C@H](CNCc2ccc3ccccc3c2)CC1)c1ccc2ccccc2c1",
  "CCn1c2ccc3cc2c2cc(ccc21)C(=O)c1ccc(cc1)Cn1cc[n+](c1)Cc1ccc(cc1)-c1cccc(-"
  "c2ccc(cc2)C[n+]2ccn(c2)Cc2ccc(cc2)C3=O)c1C(=O)O.[Br-].[Br-]",
  "CCCCCCC1C23C4=c5c6c7c8c9c%10c%11c%12c%13c%14c%15c%16c%17c%18c%19c%20c%21c%"
  "22c%23c(c5c5c6c6c8c%11c8c%11c%12c%15c%12c(c%20%16)c%21c%15c%23c5c(c68)c%"
  "15c%12%11)C2(C[N+]1(C)C)C%22C%19c1c-%18c2c5c(c13)C4C7C9=C5C1(C2C%17%14)C("
  "CCCCCC)[N+](C)(C)CC%10%131.[I-].[I-]",
  "C12C3C4C5C6C7C8C1C1C9C5C5C%10C2C2C%11C%12C%13C3C3C7C%10C7C4C%11C1C3C(C5C8%12)C(C62)C7C9%13"};

}  // namespace

std::string serializeBv(const ExplicitBitVect& bv) {
  std::vector<int> onBits;
  bv.getOnBits(onBits);
  std::string serialized = "{";
  for (const auto& bit : onBits) {
    serialized += std::to_string(bit) + " ";
  }
  serialized += "}";
  return serialized;
}

class MorganFingerprintTestFixture : public ::testing::TestWithParam<std::tuple<int, int>> {};

// Stream operator for explicitbitvect
std::ostream& operator<<(std::ostream& os, const ExplicitBitVect& bv) {
  os << serializeBv(bv);
  return os;
}

TEST(MorganFingerprintTest, CpuImplWorks) {
  const unsigned int radius       = 5;
  const unsigned int fpSize       = 2048;
  auto               refGenerator = std::unique_ptr<RDKit::FingerprintGenerator<std::uint32_t>>(
    RDKit::MorganFingerprint::getMorganGenerator<
      std::uint32_t>(radius, false, false, true, false, nullptr, nullptr, fpSize, {1, 2, 4, 8}, false, false));
  auto generator = nvMolKit::MorganFingerprintGenerator(radius, fpSize);

  auto [mols, smiles] = loadNChemblMolecules(100);
  for (size_t i = 0; i < mols.size(); i++) {
    auto refFingerprint = std::unique_ptr<ExplicitBitVect>(refGenerator->getFingerprint(*mols[i]));
    auto fingerprint    = std::unique_ptr<ExplicitBitVect>(generator.GetFingerprint(*mols[i]));
    ASSERT_NE(fingerprint, nullptr);
    ASSERT_NE(refFingerprint, nullptr);
    EXPECT_EQ(*fingerprint, *refFingerprint) << "With smiles " << smiles[i];
  }
}

void PrintFPDiff(const ExplicitBitVect* refFingerprint, const ExplicitBitVect* fingerprint) {
  std::vector<int> onBitsRef;
  std::vector<int> onBitsGot;
  refFingerprint->getOnBits(onBitsRef);
  fingerprint->getOnBits(onBitsGot);
  std::vector<int> diff;
  std::set_difference(onBitsRef.begin(), onBitsRef.end(), onBitsGot.begin(), onBitsGot.end(), std::back_inserter(diff));
  std::vector<int> diff2;
  std::set_difference(onBitsGot.begin(),
                      onBitsGot.end(),
                      onBitsRef.begin(),
                      onBitsRef.end(),
                      std::back_inserter(diff2));
  std::cout << "Mol with " << diff.size() << " different bits in ref not in gen:\n";
  for (const auto& bit : diff) {
    std::cout << bit << " ";
  }
  std::cout << "\n";
  std::cout << "Mol with " << diff2.size() << " different bits in gen not in ref\n";
  for (const auto& bit : diff2) {
    std::cout << bit << " ";
  }
  std::cout << "\n";
}

class MorganFingerprintGpuTestFixture : public testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(MorganFingerprintGpuTestFixture, CorrectParallelGpuMF) {
  const unsigned int radius    = std::get<0>(GetParam());
  const unsigned int fpSize    = std::get<1>(GetParam());
  const unsigned int batchSize = std::get<2>(GetParam());
  SCOPED_TRACE("Radius: " + std::to_string(radius) + " FpSize: " + std::to_string(fpSize) +
               " BatchSize: " + std::to_string(batchSize));
  auto refGenerator = std::unique_ptr<RDKit::FingerprintGenerator<std::uint32_t>>(
    RDKit::MorganFingerprint::getMorganGenerator<
      std::uint32_t>(radius, false, false, true, false, nullptr, nullptr, fpSize, {1, 2, 4, 8}, false, false));

  auto [mols, smiles] = loadNChemblMolecules(100, 128);
  auto molsView       = makeMolsView(mols);

  std::vector<std::unique_ptr<ExplicitBitVect>> refResults;

  for (size_t i = 0; i < mols.size(); i++) {
    const auto& mol            = mols[i];
    auto        refFingerprint = std::unique_ptr<ExplicitBitVect>(refGenerator->getFingerprint(*mol));
    ASSERT_NE(refFingerprint, nullptr);
    refResults.push_back(std::move(refFingerprint));
  }

  auto                                generator = nvMolKit::MorganFingerprintGenerator(radius, fpSize);
  nvMolKit::FingerprintComputeOptions options;
  options.backend      = nvMolKit::FingerprintComputeBackend::GPU;
  options.gpuBatchSize = batchSize;
  auto newResults      = generator.GetFingerprints(molsView, options);

  ASSERT_EQ(newResults.size(), mols.size());
  for (size_t i = 0; i < mols.size(); i++) {
    ASSERT_NE(newResults[i], nullptr);
    ASSERT_NE(refResults[i], nullptr);
    ASSERT_EQ(*newResults[i], *refResults[i]) << "on element " << i << " with smiles " << smiles[i];
  }
}

std::string PrintToStringParamName(const ::testing::TestParamInfo<std::tuple<int, int, int>>& info) {
  int radius    = std::get<0>(info.param);
  int fpsize    = std::get<1>(info.param);
  int batchsize = std::get<2>(info.param);
  return "radius_" + std::to_string(radius) + "__fpsize_" + std::to_string(fpsize) + "__batchsize_" +
         std::to_string(batchsize);
}

INSTANTIATE_TEST_SUITE_P(MorganFingerprintGpuTest,
                         MorganFingerprintGpuTestFixture,
                         ::testing::Combine(::testing::Values(0, 1, 4),
                                            ::testing::Values(128, 256, 512, 1024, 2048, 4096),
                                            ::testing::Values(5, 64, 2048)),
                         PrintToStringParamName);

TEST(MorganFingerprintCpuEdgeTest, SizeZero) {
  const unsigned int radius       = 4;
  const unsigned int fpSize       = 1024;
  auto               refGenerator = std::unique_ptr<RDKit::FingerprintGenerator<std::uint32_t>>(
    RDKit::MorganFingerprint::getMorganGenerator<
      std::uint32_t>(radius, false, false, true, false, nullptr, nullptr, fpSize, {1, 2, 4, 8}, false, false));
  auto generator = nvMolKit::MorganFingerprintGenerator(radius, fpSize);

  auto res = generator.GetFingerprints({});
  ASSERT_EQ(res.size(), 0);
}

TEST(MorganFingerprintCpuEdgeTest, SizeOne) {
  const unsigned int maxRadius   = 5;
  const unsigned int fpSize      = 128;
  std::string        caseFailing = "Cc1cc(-c2csc(N=C(N)N)n2)cn1C";
  auto               Singlemol   = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(caseFailing));
  for (size_t radius = 0; radius < maxRadius; radius++) {
    auto refGenerator = std::unique_ptr<RDKit::FingerprintGenerator<std::uint32_t>>(
      RDKit::MorganFingerprint::getMorganGenerator<
        std::uint32_t>(radius, false, false, true, false, nullptr, nullptr, fpSize, {1, 2, 4, 8}, false, false));
    auto refFingerprint = std::unique_ptr<ExplicitBitVect>(refGenerator->getFingerprint(*Singlemol));

    auto generator = nvMolKit::MorganFingerprintGenerator(radius, fpSize);
    auto res       = generator.GetFingerprints({Singlemol.get()});

    ASSERT_EQ(res.size(), 1);
    ASSERT_NE(res[0], nullptr);
    ASSERT_NE(refFingerprint, nullptr);
    if (*res[0] != *refFingerprint) {
      PrintFPDiff(refFingerprint.get(), res[0].get());
    }
    ASSERT_EQ(*res[0], *refFingerprint) << "With Radius " << radius;
  }
}

TEST(MorganFingerprintTest, GpuWithLargeMolecules) {
  const unsigned int radius       = 3;
  const unsigned int fpSize       = 2048;
  auto               refGenerator = std::unique_ptr<RDKit::FingerprintGenerator<std::uint32_t>>(
    RDKit::MorganFingerprint::getMorganGenerator<
      std::uint32_t>(radius, false, false, true, false, nullptr, nullptr, fpSize, {1, 2, 4, 8}, false, false));

  auto [mols, smiles] = loadNChemblMolecules(100);
  int largeCount      = 0;
  for (const auto& mol : mols) {
    if (mol->getNumAtoms() > 128 || mol->getNumBonds() > 128) {
      largeCount++;
    }
  }
  ASSERT_GT(largeCount, 0) << "Test case should have large molecules";
  auto molsView = makeMolsView(mols);

  std::vector<std::unique_ptr<ExplicitBitVect>> refResults;

  for (size_t i = 0; i < mols.size(); i++) {
    const auto& mol            = mols[i];
    auto        refFingerprint = std::unique_ptr<ExplicitBitVect>(refGenerator->getFingerprint(*mol));
    ASSERT_NE(refFingerprint, nullptr);
    refResults.push_back(std::move(refFingerprint));
  }

  auto                                generator = nvMolKit::MorganFingerprintGenerator(radius, fpSize);
  nvMolKit::FingerprintComputeOptions options;
  options.backend = nvMolKit::FingerprintComputeBackend::GPU;
  auto newResults = generator.GetFingerprints(molsView);

  ASSERT_EQ(newResults.size(), mols.size());
  for (size_t i = 0; i < mols.size(); i++) {
    ASSERT_NE(newResults[i], nullptr);
    ASSERT_NE(refResults[i], nullptr);
    ASSERT_EQ(*newResults[i], *refResults[i]) << "on element " << i << " with smiles " << smiles[i];
  }
}

TEST(MorganFingerprintGpuTest, ThrowsRequestingCpuBackendGpuBuffer) {
  const unsigned int                  radius    = 3;
  const unsigned int                  fpSize    = 1024;
  std::string                         smi       = "C";
  auto                                Singlemol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smi));
  auto                                generator = nvMolKit::MorganFingerprintGenerator(radius, fpSize);
  nvMolKit::FingerprintComputeOptions options;
  options.backend = nvMolKit::FingerprintComputeBackend::CPU;
  ASSERT_THROW(generator.GetFingerprintsGpuBuffer<1024>({Singlemol.get()}, nullptr, options), std::runtime_error);
}

TEST(MorganFingerprintGpuTest, GpuBufferSameResult) {
  constexpr unsigned int radius    = 3;
  constexpr unsigned int fpSize    = 1024;
  auto                   generator = nvMolKit::MorganFingerprintGenerator(radius, fpSize);
  auto [mols, smiles]              = loadNChemblMolecules(100, 128);
  auto molsView                    = makeMolsView(mols);

  nvMolKit::FingerprintComputeOptions options;
  options.backend = nvMolKit::FingerprintComputeBackend::GPU;

  auto gpuResults = generator.GetFingerprintsGpuBuffer<fpSize>(molsView, nullptr, options);
  cudaDeviceSynchronize();

  std::vector<nvMolKit::FlatBitVect<fpSize>> gpuResultsHost(gpuResults.size());
  gpuResults.copyToHost(gpuResultsHost);
  cudaDeviceSynchronize();

  auto refGenerator = std::unique_ptr<RDKit::FingerprintGenerator<std::uint32_t>>(
    RDKit::MorganFingerprint::getMorganGenerator<
      std::uint32_t>(radius, false, false, true, false, nullptr, nullptr, fpSize, {1, 2, 4, 8}, false, false));

  std::vector<std::unique_ptr<ExplicitBitVect>> refResults;
  for (size_t i = 0; i < mols.size(); i++) {
    const auto& mol            = mols[i];
    auto        refFingerprint = std::unique_ptr<ExplicitBitVect>(refGenerator->getFingerprint(*mol));
    ASSERT_NE(refFingerprint, nullptr);
    refResults.push_back(std::move(refFingerprint));
  }
  ASSERT_EQ(refResults.size(), gpuResultsHost.size());
  for (size_t i = 0; i < refResults.size(); i++) {
    ASSERT_NE(refResults[i], nullptr);
    for (size_t bitId = 0; bitId < fpSize; bitId++) {
      ASSERT_EQ(refResults[i]->getBit(bitId), gpuResultsHost[i][bitId])
        << "on element " << i << " with smiles " << smiles[i] << ", bit " << bitId;
    }
  }
}

class MorganFingerprintParametrizedTest
    : public testing::TestWithParam<std::tuple<int, nvMolKit::FingerprintComputeBackend>> {};

TEST_P(MorganFingerprintParametrizedTest, MultipleCallsWithDifferenThreads) {
  const nvMolKit::FingerprintComputeBackend backend      = std::get<1>(GetParam());
  const unsigned int                        nThreads     = std::get<0>(GetParam());
  constexpr unsigned int                    radius       = 3;
  constexpr unsigned int                    fpSize       = 1024;
  auto                                      generator    = nvMolKit::MorganFingerprintGenerator(radius, fpSize);
  auto                                      refGenerator = std::unique_ptr<RDKit::FingerprintGenerator<std::uint32_t>>(
    RDKit::MorganFingerprint::getMorganGenerator<
      std::uint32_t>(radius, false, false, true, false, nullptr, nullptr, fpSize, {1, 2, 4, 8}, false, false));
  auto [mols, smiles] = loadNChemblMolecules(100, 128);
  auto molsView       = makeMolsView(mols);

  std::vector<std::unique_ptr<ExplicitBitVect>> refResults;
  for (size_t i = 0; i < mols.size(); i++) {
    const auto& mol            = mols[i];
    auto        refFingerprint = std::unique_ptr<ExplicitBitVect>(refGenerator->getFingerprint(*mol));
    ASSERT_NE(refFingerprint, nullptr);
    refResults.push_back(std::move(refFingerprint));
  }

  nvMolKit::FingerprintComputeOptions options;
  options.backend       = backend;
  options.numCpuThreads = nThreads;
  auto resultsTry1      = generator.GetFingerprints(molsView, options);
  auto resultsTry2      = generator.GetFingerprints(molsView, options);

  ASSERT_EQ(resultsTry2.size(), mols.size());
  for (size_t i = 0; i < mols.size(); i++) {
    ASSERT_NE(resultsTry2[i], nullptr);
    ASSERT_NE(refResults[i], nullptr);
    ASSERT_EQ(*resultsTry2[i], *refResults[i]) << "on element " << i << " with smiles " << smiles[i];
  }

  if (backend == nvMolKit::FingerprintComputeBackend::GPU) {
    auto gpuResultsTry1 = generator.GetFingerprintsGpuBuffer<1024>(molsView, nullptr, options);
    auto gpuResultsTry2 = generator.GetFingerprintsGpuBuffer<1024>(molsView, nullptr, options);
    cudaDeviceSynchronize();
    std::vector<nvMolKit::FlatBitVect<fpSize>> gpuResultsHost(gpuResultsTry2.size());
    gpuResultsTry2.copyToHost(gpuResultsHost);
    cudaDeviceSynchronize();
    ASSERT_EQ(refResults.size(), gpuResultsHost.size());
    for (size_t i = 0; i < refResults.size(); i++) {
      ASSERT_NE(refResults[i], nullptr);
      for (size_t bitId = 0; bitId < fpSize; bitId++) {
        ASSERT_EQ(refResults[i]->getBit(bitId), gpuResultsHost[i][bitId])
          << "on element " << i << " with smiles " << smiles[i] << ", bit " << bitId;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
  MorganFingerprintTest,
  MorganFingerprintParametrizedTest,
  ::testing::Combine(::testing::Values(0, 1, 2, 5, 32),
                     ::testing::Values(nvMolKit::FingerprintComputeBackend::CPU,
                                       nvMolKit::FingerprintComputeBackend::GPU)),
  [](const ::testing::TestParamInfo<std::tuple<int, nvMolKit::FingerprintComputeBackend>>& info) {
    int         numThreads = std::get<0>(info.param);
    auto        backend    = std::get<1>(info.param);
    std::string backendStr = (backend == nvMolKit::FingerprintComputeBackend::CPU) ? "CPU" : "GPU";
    return "numThreads_" + std::to_string(numThreads) + "__backend_" + backendStr;
  });
TEST(MorganFingerprintGpuEdgeTest, SizeZero) {
  const unsigned int radius       = 4;
  const unsigned int fpSize       = 1024;
  auto               refGenerator = std::unique_ptr<RDKit::FingerprintGenerator<std::uint32_t>>(
    RDKit::MorganFingerprint::getMorganGenerator<
      std::uint32_t>(radius, false, false, true, false, nullptr, nullptr, fpSize, {1, 2, 4, 8}, false, false));
  auto                                generator = nvMolKit::MorganFingerprintGenerator(radius, fpSize);
  nvMolKit::FingerprintComputeOptions options;
  options.backend = nvMolKit::FingerprintComputeBackend::GPU;
  auto res        = generator.GetFingerprints({}, options);
  ASSERT_EQ(res.size(), 0);
}

TEST(MorganFingerprintGpuEdgeTest, SizeOne) {
  const unsigned int maxRadius   = 5;
  const unsigned int fpSize      = 128;
  std::string        caseFailing = "Cc1cc(-c2csc(N=C(N)N)n2)cn1C";
  auto               Singlemol   = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(caseFailing));
  for (size_t radius = 0; radius < maxRadius; radius++) {
    auto refGenerator = std::unique_ptr<RDKit::FingerprintGenerator<std::uint32_t>>(
      RDKit::MorganFingerprint::getMorganGenerator<
        std::uint32_t>(radius, false, false, true, false, nullptr, nullptr, fpSize, {1, 2, 4, 8}, false, false));
    auto refFingerprint = std::unique_ptr<ExplicitBitVect>(refGenerator->getFingerprint(*Singlemol));

    auto                                generator = nvMolKit::MorganFingerprintGenerator(radius, fpSize);
    nvMolKit::FingerprintComputeOptions options;
    options.backend = nvMolKit::FingerprintComputeBackend::GPU;
    auto res        = generator.GetFingerprints({Singlemol.get()}, options);

    ASSERT_EQ(res.size(), 1);
    ASSERT_NE(res[0], nullptr);
    ASSERT_NE(refFingerprint, nullptr);
    if (*res[0] != *refFingerprint) {
      PrintFPDiff(refFingerprint.get(), res[0].get());
    }
    ASSERT_EQ(*res[0], *refFingerprint) << "With Radius " << radius;
  }
}
