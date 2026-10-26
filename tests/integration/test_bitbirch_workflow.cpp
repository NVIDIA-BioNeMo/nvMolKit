// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/ROMol.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "src/bitbirch.h"
#include "src/morgan_fingerprint.h"
#include "src/testutils/mol_data.h"

namespace {

std::vector<const RDKit::ROMol*> makeMoleculeView(const std::vector<std::unique_ptr<RDKit::ROMol>>& molecules) {
  std::vector<const RDKit::ROMol*> view;
  view.reserve(molecules.size());
  for (const auto& molecule : molecules) {
    view.push_back(molecule.get());
  }
  return view;
}

TEST(BitBirchIntegration, MorganFingerprintsRemainDeviceResidentThroughClustering) {
  constexpr int fingerprintSize = 1024;
  constexpr int numWords        = fingerprintSize / 32;
  auto [molecules, smiles]      = nvMolKit::testing::loadNChemblMolecules(100, 128);
  const auto moleculeView       = makeMoleculeView(molecules);

  nvMolKit::MorganFingerprintGenerator generator(3, fingerprintSize);
  nvMolKit::FingerprintComputeOptions  options;
  options.backend                = nvMolKit::FingerprintComputeBackend::GPU;
  auto        deviceFingerprints = generator.GetFingerprintsGpuBuffer<fingerprintSize>(moleculeView, nullptr, options);
  const auto* packed             = reinterpret_cast<const std::uint32_t*>(deviceFingerprints.data());
  const cuda::std::span<const std::uint32_t> packedSpan{packed, molecules.size() * numWords};

  nvMolKit::BitBirchOptions treeOptions;
  treeOptions.branchingFactor = 7;
  treeOptions.batchSize       = 32;
  treeOptions.returnCentroids = true;
  auto result = nvMolKit::bitBirchGpu(packedSpan, static_cast<int>(molecules.size()), numWords, 0.0, treeOptions);
  std::vector<int>           labels(molecules.size());
  std::vector<std::uint32_t> centroid(numWords);
  std::vector<std::uint32_t> hostPacked(molecules.size() * numWords);
  result.clusterIds.copyToHost(labels);
  result.centroids.copyToHost(centroid, numWords);
  ASSERT_EQ(cudaMemcpy(hostPacked.data(), packed, hostPacked.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  ASSERT_EQ(result.numClusters, 1);
  EXPECT_EQ(labels, std::vector<int>(molecules.size(), 0));
  for (int word = 0; word < numWords; ++word) {
    std::uint32_t expected = 0;
    for (int bit = 0; bit < 32; ++bit) {
      int linearSum = 0;
      for (std::size_t molecule = 0; molecule < molecules.size(); ++molecule) {
        linearSum += (hostPacked[molecule * numWords + word] >> bit) & 1U;
      }
      if (2 * linearSum >= static_cast<int>(molecules.size())) {
        expected |= std::uint32_t{1} << bit;
      }
    }
    EXPECT_EQ(centroid[word], expected);
  }
}

TEST(BitBirchIntegration, MorganClusteringIsDeterministicAndLabelsAreDense) {
  constexpr int fingerprintSize = 512;
  constexpr int numWords        = fingerprintSize / 32;
  auto [molecules, smiles]      = nvMolKit::testing::loadNChemblMolecules(100, 128);
  const auto moleculeView       = makeMoleculeView(molecules);

  nvMolKit::MorganFingerprintGenerator generator(2, fingerprintSize);
  nvMolKit::FingerprintComputeOptions  options;
  options.backend                = nvMolKit::FingerprintComputeBackend::GPU;
  auto        deviceFingerprints = generator.GetFingerprintsGpuBuffer<fingerprintSize>(moleculeView, nullptr, options);
  const auto* packed             = reinterpret_cast<const std::uint32_t*>(deviceFingerprints.data());
  const cuda::std::span<const std::uint32_t> packedSpan{packed, molecules.size() * numWords};

  nvMolKit::BitBirchOptions treeOptions;
  treeOptions.branchingFactor = 7;
  treeOptions.batchSize       = 32;
  auto first  = nvMolKit::bitBirchGpu(packedSpan, static_cast<int>(molecules.size()), numWords, 0.55, treeOptions);
  auto second = nvMolKit::bitBirchGpu(packedSpan, static_cast<int>(molecules.size()), numWords, 0.55, treeOptions);
  std::vector<int> firstLabels(molecules.size());
  std::vector<int> secondLabels(molecules.size());
  first.clusterIds.copyToHost(firstLabels);
  second.clusterIds.copyToHost(secondLabels);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(first.numClusters, second.numClusters);
  EXPECT_EQ(firstLabels, secondLabels);
  std::vector<bool> seen(first.numClusters, false);
  for (const int label : firstLabels) {
    ASSERT_GE(label, 0);
    ASSERT_LT(label, first.numClusters);
    seen[label] = true;
  }
  EXPECT_EQ(seen, std::vector<bool>(first.numClusters, true));
}

}  // namespace
