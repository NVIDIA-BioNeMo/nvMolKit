// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "src/descriptors3d_topology.cuh"
#include "src/utils/device_vector.h"

using namespace nvMolKit::descriptors3d_detail;

namespace {

//! Linear chains with the given atom counts: CSR neighbor lists (molecule-local indices) and atom starts.
struct ChainBatch {
  std::vector<int32_t> atomStarts{0};
  std::vector<int32_t> neighborStarts{0};
  std::vector<int32_t> neighbors;

  explicit ChainBatch(const std::vector<int>& sizes) {
    for (const int size : sizes) {
      for (int atom = 0; atom < size; ++atom) {
        if (atom > 0) {
          neighbors.push_back(atom - 1);
        }
        if (atom + 1 < size) {
          neighbors.push_back(atom + 1);
        }
        neighborStarts.push_back(static_cast<int32_t>(neighbors.size()));
      }
      atomStarts.push_back(atomStarts.back() + size);
    }
  }
};

std::vector<uint8_t> buildTable(const std::vector<int>& sizes, const int64_t moleculeAtomPairs) {
  const ChainBatch                     batch(sizes);
  nvMolKit::AsyncDeviceVector<int32_t> atomStarts(batch.atomStarts.size());
  nvMolKit::AsyncDeviceVector<int32_t> neighborStarts(batch.neighborStarts.size());
  nvMolKit::AsyncDeviceVector<int32_t> neighbors(batch.neighbors.size());
  atomStarts.copyFromHost(batch.atomStarts);
  neighborStarts.copyFromHost(batch.neighborStarts);
  neighbors.copyFromHost(batch.neighbors);

  nvMolKit::DeviceCoordView coordinates;
  coordinates.nMols = static_cast<int>(sizes.size());
  nvMolKit::Property3DDeviceInputs inputs;
  inputs.moleculeAtomStarts = atomStarts.data();
  inputs.bondNeighborStarts = neighborStarts.data();
  inputs.bondNeighbors      = neighbors.data();
  for (const int size : sizes) {
    inputs.maxMoleculeAtoms = std::max(inputs.maxMoleculeAtoms, size);
  }
  inputs.moleculeAtomPairs = moleculeAtomPairs;

  BondDistanceTableStorage storage;
  buildBondDistanceTable(coordinates, inputs, storage, nullptr);
  std::vector<uint8_t> table(storage.distances.size());
  storage.distances.copyToHost(table);
  EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
  return table;
}

//! Pairs (j < k) of each chain in row-major order: bond count k - j up to kBondTableDepth, else unreached.
std::vector<uint8_t> expectedChainDistances(const std::vector<int>& sizes) {
  std::vector<uint8_t> expected;
  for (const int size : sizes) {
    for (int j = 0; j < size; ++j) {
      for (int k = j + 1; k < size; ++k) {
        expected.push_back(k - j <= kBondTableDepth ? static_cast<uint8_t>(k - j) : kUnreachedDepth);
      }
    }
  }
  return expected;
}

}  // namespace

TEST(BondDistanceTable, SizedByMoleculeAtomPairs) {
  // One large molecule among small ones: the table holds exactly the batch's pairs, not
  // nMols * maxPairs.
  const std::vector<int> sizes    = {3, 40, 1, 12, 2};
  const auto             expected = expectedChainDistances(sizes);
  EXPECT_EQ(buildTable(sizes, static_cast<int64_t>(expected.size())), expected);
}

TEST(BondDistanceTable, WithoutPairCountSizesForLargestMolecule) {
  const std::vector<int> sizes    = {3, 40, 1, 12, 2};
  const auto             expected = expectedChainDistances(sizes);
  auto                   table    = buildTable(sizes, 0);
  ASSERT_EQ(table.size(), sizes.size() * (40 * 39 / 2));
  table.resize(expected.size());  // Entries past the batch's pairs are unused.
  EXPECT_EQ(table, expected);
}
