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

#include <gmock/gmock.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include "src/substruct/molecules_device.cuh"
#include "src/substruct/packed_bonds_device.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/rdkit_compat.h"

using nvMolKit::AsyncDeviceVector;
using nvMolKit::AtomDataPacked;
using nvMolKit::checkReturnCode;
using nvMolKit::getMolecule;
using nvMolKit::MoleculesDevice;
using nvMolKit::MoleculesDeviceViewT;
using nvMolKit::MoleculesHost;
using nvMolKit::MoleculeType;
using nvMolKit::ScopedStream;
using nvMolKit::TargetAtomBonds;
using nvMolKit::TargetMoleculesDeviceView;
using nvMolKit::TargetMoleculeView;
using nvMolKit::unpackBondType;

namespace {

std::unique_ptr<RDKit::ROMol> makeMol(const std::string& smiles) {
  auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
  EXPECT_NE(mol, nullptr) << "Failed to parse SMILES: " << smiles;
  return mol;
}

//! Enum for all testable atom properties
enum class AtomProperty {
  AtomicNum,
  NumExplicitHs,
  ExplicitValence,
  ImplicitValence,
  FormalCharge,
  ChiralTag,
  NumRadicalElectrons,
  Hybridization,
  MinRingSize,
  NumRings,
  IsAromatic
};

//! Get string name for AtomProperty (for test output)
const char* atomPropertyName(AtomProperty prop) {
  switch (prop) {
    case AtomProperty::AtomicNum:
      return "AtomicNum";
    case AtomProperty::NumExplicitHs:
      return "NumExplicitHs";
    case AtomProperty::ExplicitValence:
      return "ExplicitValence";
    case AtomProperty::ImplicitValence:
      return "ImplicitValence";
    case AtomProperty::FormalCharge:
      return "FormalCharge";
    case AtomProperty::ChiralTag:
      return "ChiralTag";
    case AtomProperty::NumRadicalElectrons:
      return "NumRadicalElectrons";
    case AtomProperty::Hybridization:
      return "Hybridization";
    case AtomProperty::MinRingSize:
      return "MinRingSize";
    case AtomProperty::NumRings:
      return "NumRings";
    case AtomProperty::IsAromatic:
      return "IsAromatic";
  }
  return "Unknown";
}

//! Get expected value from RDKit for a given atom property
int getExpectedAtomProperty(const RDKit::ROMol* mol, int atomIdx, AtomProperty prop) {
  const auto* atom     = mol->getAtomWithIdx(atomIdx);
  const auto* ringInfo = mol->getRingInfo();
  switch (prop) {
    case AtomProperty::AtomicNum:
      return atom->getAtomicNum();
    case AtomProperty::NumExplicitHs:
      return atom->getTotalNumHs();
    case AtomProperty::ExplicitValence:
      return nvMolKit::compat::getExplicitValence(atom);
    case AtomProperty::ImplicitValence:
      return nvMolKit::compat::getImplicitValence(atom);
    case AtomProperty::FormalCharge:
      return atom->getFormalCharge();
    case AtomProperty::ChiralTag:
      return static_cast<int>(atom->getChiralTag());
    case AtomProperty::NumRadicalElectrons:
      return atom->getNumRadicalElectrons();
    case AtomProperty::Hybridization:
      return static_cast<int>(atom->getHybridization());
    case AtomProperty::MinRingSize:
      return ringInfo->minAtomRingSize(atomIdx);
    case AtomProperty::NumRings:
      return ringInfo->numAtomRings(atomIdx);
    case AtomProperty::IsAromatic:
      return atom->getIsAromatic() ? 1 : 0;
  }
  return -1;
}

//! Kernel to read any atom property for all atoms of a specific molecule
template <MoleculeType Type>
__global__ void readAtomPropertyKernel(MoleculesDeviceViewT<Type> view, int molIdx, AtomProperty prop, int* results) {
  const auto mol     = getMolecule(view, molIdx);
  const int  atomIdx = threadIdx.x;
  if (atomIdx >= mol.numAtoms) {
    return;
  }

  const AtomDataPacked& atom = mol.getAtomPacked(atomIdx);
  switch (prop) {
    case AtomProperty::AtomicNum:
      results[atomIdx] = atom.atomicNum();
      break;
    case AtomProperty::NumExplicitHs:
      results[atomIdx] = atom.numExplicitHs();
      break;
    case AtomProperty::ExplicitValence:
      results[atomIdx] = atom.explicitValence();
      break;
    case AtomProperty::ImplicitValence:
      results[atomIdx] = atom.implicitValence();
      break;
    case AtomProperty::FormalCharge:
      results[atomIdx] = atom.formalCharge();
      break;
    case AtomProperty::ChiralTag:
      results[atomIdx] = atom.chiralTag();
      break;
    case AtomProperty::NumRadicalElectrons:
      results[atomIdx] = atom.numRadicalElectrons();
      break;
    case AtomProperty::Hybridization:
      results[atomIdx] = atom.hybridization();
      break;
    case AtomProperty::MinRingSize:
      results[atomIdx] = atom.minRingSize();
      break;
    case AtomProperty::NumRings:
      results[atomIdx] = atom.numRings();
      break;
    case AtomProperty::IsAromatic:
      results[atomIdx] = atom.isAromatic() ? 1 : 0;
      break;
  }
}

//! Kernel to read number of atoms per molecule
__global__ void readNumAtomsKernel(TargetMoleculesDeviceView view, int* results, int numMols) {
  const int molIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (molIdx >= numMols) {
    return;
  }
  const TargetMoleculeView mol = getMolecule(view, molIdx);
  results[molIdx]              = mol.numAtoms;
}

//! Kernel to read bond type of first atom's first bond
__global__ void readBondTypeKernel(TargetMoleculesDeviceView view, int* results, int numMols) {
  const int molIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (molIdx >= numMols) {
    return;
  }
  const TargetMoleculeView mol = getMolecule(view, molIdx);
  if (mol.numAtoms > 0 && mol.getAtomDegree(0) > 0) {
    const TargetAtomBonds& bonds = mol.getTargetBonds(0);
    results[molIdx]              = unpackBondType(bonds.bondInfo[0]);
  } else {
    results[molIdx] = -1;
  }
}

//! Kernel to read degree (number of bonds) of first atom
__global__ void readAtomDegreeKernel(TargetMoleculesDeviceView view, int* results, int numMols) {
  const int molIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (molIdx >= numMols) {
    return;
  }
  const TargetMoleculeView mol = getMolecule(view, molIdx);
  if (mol.numAtoms > 0) {
    results[molIdx] = mol.getAtomDegree(0);
  } else {
    results[molIdx] = -1;
  }
}

//! Kernel to read neighbor atom index of first atom's first neighbor
__global__ void readNeighborAtomKernel(TargetMoleculesDeviceView view, int* results, int numMols) {
  const int molIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (molIdx >= numMols) {
    return;
  }
  const TargetMoleculeView mol = getMolecule(view, molIdx);
  if (mol.numAtoms > 0 && mol.getAtomDegree(0) > 0) {
    const TargetAtomBonds& bonds = mol.getTargetBonds(0);
    results[molIdx]              = bonds.neighborIdx[0];
  } else {
    results[molIdx] = -1;
  }
}

//! Kernel to read degree for all atoms of a specific molecule
__global__ void readAllAtomDegreesKernel(TargetMoleculesDeviceView view, int molIdx, int* results) {
  const TargetMoleculeView mol     = getMolecule(view, molIdx);
  const int                atomIdx = threadIdx.x;
  if (atomIdx < mol.numAtoms) {
    results[atomIdx] = mol.getAtomDegree(atomIdx);
  }
}

//! Kernel to read all neighbor atom indices for a specific atom in a specific molecule
__global__ void readAllNeighborsKernel(TargetMoleculesDeviceView view, int molIdx, int atomIdx, int* results) {
  const TargetMoleculeView mol         = getMolecule(view, molIdx);
  const int                neighborIdx = threadIdx.x;
  const TargetAtomBonds&   bonds       = mol.getTargetBonds(atomIdx);
  if (neighborIdx < bonds.degree) {
    results[neighborIdx] = bonds.neighborIdx[neighborIdx];
  }
}

//! Kernel to read all neighbor bond types for a specific atom in a specific molecule
__global__ void readAllNeighborBondTypesKernel(TargetMoleculesDeviceView view, int molIdx, int atomIdx, int* results) {
  const TargetMoleculeView mol         = getMolecule(view, molIdx);
  const int                neighborIdx = threadIdx.x;
  const TargetAtomBonds&   bonds       = mol.getTargetBonds(atomIdx);
  if (neighborIdx < bonds.degree) {
    results[neighborIdx] = unpackBondType(bonds.bondInfo[neighborIdx]);
  }
}

}  // namespace

// =============================================================================
// Batch Structure Tests
// =============================================================================

class BatchStructureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mols_.push_back(makeMol("CCO"));       // ethanol: 3 atoms, 2 bonds
    mols_.push_back(makeMol("c1ccccc1"));  // benzene: 6 atoms, 6 bonds (aromatic)
    mols_.push_back(makeMol("CC(=O)O"));   // acetic acid: 4 atoms, 3 bonds
    mols_.push_back(makeMol("[NH4+]"));    // ammonium: 1 atom (charged)
    mols_.push_back(makeMol("C#N"));       // hydrogen cyanide: 2 atoms, 1 triple bond

    for (const auto& mol : mols_) {
      nvMolKit::addToBatch(mol.get(), batch_);
    }
  }

  std::vector<std::unique_ptr<RDKit::ROMol>> mols_;
  MoleculesHost                              batch_;
};

TEST_F(BatchStructureTest, HostBatchStructure) {
  EXPECT_EQ(batch_.numMolecules(), 5);
  EXPECT_EQ(batch_.totalAtoms(), 16);  // 3 + 6 + 4 + 1 + 2
}

TEST_F(BatchStructureTest, NumAtomsMatchRDKit) {
  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch_);

  const int              numMols = static_cast<int>(batch_.numMolecules());
  AsyncDeviceVector<int> resultsDev(numMols, stream.stream());

  readNumAtomsKernel<<<1, 32, 0, stream.stream()>>>(device.view<MoleculeType::Target>(), resultsDev.data(), numMols);
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(numMols);
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < numMols; ++i) {
    EXPECT_EQ(results[i], static_cast<int>(mols_[i]->getNumAtoms())) << "Mismatch at molecule " << i;
  }
}

TEST_F(BatchStructureTest, AtomDegreeMatchesRDKit) {
  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch_);

  const int              numMols = static_cast<int>(batch_.numMolecules());
  AsyncDeviceVector<int> resultsDev(numMols, stream.stream());

  readAtomDegreeKernel<<<1, 32, 0, stream.stream()>>>(device.view<MoleculeType::Target>(), resultsDev.data(), numMols);
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(numMols);
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < numMols; ++i) {
    if (mols_[i]->getNumAtoms() > 0) {
      EXPECT_EQ(results[i], static_cast<int>(mols_[i]->getAtomWithIdx(0)->getDegree())) << "Mismatch at molecule " << i;
    }
  }
}

TEST_F(BatchStructureTest, BondTypeMatchesRDKit) {
  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch_);

  const int              numMols = static_cast<int>(batch_.numMolecules());
  AsyncDeviceVector<int> resultsDev(numMols, stream.stream());

  readBondTypeKernel<<<1, 32, 0, stream.stream()>>>(device.view<MoleculeType::Target>(), resultsDev.data(), numMols);
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(numMols);
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < numMols; ++i) {
    const auto& mol = mols_[i];
    if (mol->getNumAtoms() > 0 && mol->getAtomWithIdx(0)->getDegree() > 0) {
      auto [beg, end]      = mol->getAtomBonds(mol->getAtomWithIdx(0));
      const auto* bond     = (*mol)[*beg];
      const int   expected = static_cast<int>(bond->getBondType());
      EXPECT_EQ(results[i], expected) << "Mismatch at molecule " << i;
    }
  }
}

// =============================================================================
// Parametrized Atom Property Tests
// =============================================================================

class AtomPropertyTest : public ::testing::TestWithParam<AtomProperty> {
 protected:
  void SetUp() override {
    // Diverse molecules to test various properties:
    // - ethanol: sp3 carbons, hydroxyl
    // - benzene: aromatic, sp2 hybridization
    // - acetic acid: sp2 carbonyl carbon
    // - ammonium: positive charge
    // - hydrogen cyanide: sp hybridization, triple bond
    // - indole: fused rings, nitrogen in ring
    // - methyl radical: radical electron
    mols_.push_back(makeMol("CCO"));               // ethanol
    mols_.push_back(makeMol("c1ccccc1"));          // benzene
    mols_.push_back(makeMol("CC(=O)O"));           // acetic acid
    mols_.push_back(makeMol("[NH4+]"));            // ammonium
    mols_.push_back(makeMol("C#N"));               // hydrogen cyanide
    mols_.push_back(makeMol("c1ccc2[nH]ccc2c1"));  // indole
    mols_.push_back(makeMol("[CH3]"));             // methyl radical

    for (const auto& mol : mols_) {
      nvMolKit::addToBatch(mol.get(), batch_);
    }
  }

  std::vector<std::unique_ptr<RDKit::ROMol>> mols_;
  MoleculesHost                              batch_;
};

TEST_P(AtomPropertyTest, PropertyMatchesRDKit) {
  const AtomProperty prop = GetParam();

  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch_);

  // Test each molecule exhaustively
  for (size_t molIdx = 0; molIdx < mols_.size(); ++molIdx) {
    const auto& mol      = mols_[molIdx];
    const int   numAtoms = mol->getNumAtoms();

    AsyncDeviceVector<int> resultsDev(numAtoms, stream.stream());
    readAtomPropertyKernel<<<1, numAtoms, 0, stream.stream()>>>(device.view<MoleculeType::Target>(),
                                                                static_cast<int>(molIdx),
                                                                prop,
                                                                resultsDev.data());
    cudaCheckError(cudaGetLastError());

    std::vector<int> results(numAtoms);
    resultsDev.copyToHost(results);
    cudaCheckError(cudaStreamSynchronize(stream.stream()));

    for (int atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
      const int expected = getExpectedAtomProperty(mol.get(), atomIdx, prop);
      EXPECT_EQ(results[atomIdx], expected)
        << "Property " << atomPropertyName(prop) << " mismatch at mol " << molIdx << " atom " << atomIdx;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(AllAtomProperties,
                         AtomPropertyTest,
                         ::testing::Values(AtomProperty::AtomicNum,
                                           AtomProperty::NumExplicitHs,
                                           AtomProperty::ExplicitValence,
                                           AtomProperty::ImplicitValence,
                                           AtomProperty::FormalCharge,
                                           AtomProperty::ChiralTag,
                                           AtomProperty::NumRadicalElectrons,
                                           AtomProperty::Hybridization,
                                           AtomProperty::MinRingSize,
                                           AtomProperty::NumRings,
                                           AtomProperty::IsAromatic),
                         [](const ::testing::TestParamInfo<AtomProperty>& info) {
                           return atomPropertyName(info.param);
                         });

// =============================================================================
// Connectivity Tests (neighbors, bonds between atoms)
// =============================================================================

class ConnectivityTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    mols_.push_back(makeMol("CCO"));       // ethanol: 3 atoms, 2 bonds
    mols_.push_back(makeMol("c1ccccc1"));  // benzene: 6 atoms, 6 bonds (aromatic)
    mols_.push_back(makeMol("CC(=O)O"));   // acetic acid: 4 atoms, 3 bonds
    mols_.push_back(makeMol("[NH4+]"));    // ammonium: 1 atom (charged)
    mols_.push_back(makeMol("C#N"));       // hydrogen cyanide: 2 atoms, 1 triple bond

    for (const auto& mol : mols_) {
      nvMolKit::addToBatch(mol.get(), batch_);
    }
  }

  std::vector<std::unique_ptr<RDKit::ROMol>> mols_;
  MoleculesHost                              batch_;
};

TEST_F(ConnectivityTestFixture, AtomDegreeMatchesRDKit) {
  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch_);

  const int              numMols = static_cast<int>(batch_.numMolecules());
  AsyncDeviceVector<int> resultsDev(numMols, stream.stream());

  readAtomDegreeKernel<<<1, 32, 0, stream.stream()>>>(device.view<MoleculeType::Target>(), resultsDev.data(), numMols);
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(numMols);
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < numMols; ++i) {
    if (mols_[i]->getNumAtoms() > 0) {
      const int expected = mols_[i]->getAtomWithIdx(0)->getDegree();
      EXPECT_EQ(results[i], expected) << "Mismatch at molecule " << i;
    }
  }
}

TEST_F(ConnectivityTestFixture, NeighborAtomIndexMatchesRDKit) {
  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch_);

  const int              numMols = static_cast<int>(batch_.numMolecules());
  AsyncDeviceVector<int> resultsDev(numMols, stream.stream());

  readNeighborAtomKernel<<<1, 32, 0, stream.stream()>>>(device.view<MoleculeType::Target>(),
                                                        resultsDev.data(),
                                                        numMols);
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(numMols);
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < numMols; ++i) {
    const auto& mol = mols_[i];
    if (mol->getNumAtoms() > 0 && mol->getAtomWithIdx(0)->getDegree() > 0) {
      auto [beg, end]      = mol->getAtomBonds(mol->getAtomWithIdx(0));
      const auto* bond     = (*mol)[*beg];
      const int   expected = bond->getOtherAtomIdx(0);
      EXPECT_EQ(results[i], expected) << "Mismatch at molecule " << i;
    }
  }
}

TEST(MoleculesEmptyBatchTest, EmptyBatchHasZeroMolecules) {
  MoleculesHost batch;
  EXPECT_EQ(batch.numMolecules(), 0);
  EXPECT_EQ(batch.totalAtoms(), 0);
}

TEST(MoleculesTotalHCountTest, StoresTotalNotExplicitHCount) {
  auto          mol = makeMol("C=N");
  MoleculesHost batch;
  nvMolKit::addToBatch(mol.get(), batch);

  ASSERT_EQ(batch.atomDataPacked.size(), 2);
  const int nitrogenIdx = 1;
  EXPECT_EQ(mol->getAtomWithIdx(nitrogenIdx)->getNumExplicitHs(), 0);
  EXPECT_EQ(mol->getAtomWithIdx(nitrogenIdx)->getTotalNumHs(), 1);
  EXPECT_EQ(batch.atomDataPacked[nitrogenIdx].numExplicitHs(), 1);
}

TEST(MoleculesEmptyBatchTest, CopyFromHostThrowsOnEmptyBatch) {
  MoleculesHost   batch;
  ScopedStream    stream;
  MoleculesDevice device(stream.stream());

  EXPECT_THROW(device.copyFromHost(batch), std::invalid_argument);
}

TEST(MoleculesSingleMolTest, SingleMoleculeWorks) {
  auto          mol = makeMol("C");  // methane
  MoleculesHost batch;
  nvMolKit::addToBatch(mol.get(), batch);

  EXPECT_EQ(batch.numMolecules(), 1);
  EXPECT_EQ(batch.totalAtoms(), 1);

  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch);

  AsyncDeviceVector<int> resultsDev(1, stream.stream());
  readAtomPropertyKernel<<<1, 1, 0, stream.stream()>>>(device.view<MoleculeType::Target>(),
                                                       0,
                                                       AtomProperty::AtomicNum,
                                                       resultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(1);
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_EQ(results[0], 6);  // Carbon
}

TEST(MoleculesLargeBatchTest, ManyMoleculesWork) {
  constexpr int                              numMols = 100;
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  MoleculesHost                              batch;

  for (int i = 0; i < numMols; ++i) {
    mols.push_back(makeMol("CCCCCC"));  // hexane
    nvMolKit::addToBatch(mols.back().get(), batch);
  }

  EXPECT_EQ(batch.numMolecules(), numMols);
  EXPECT_EQ(batch.totalAtoms(), numMols * 6);

  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch);

  AsyncDeviceVector<int> resultsDev(numMols, stream.stream());
  const int              numBlocks = (numMols + 255) / 256;
  readNumAtomsKernel<<<numBlocks, 256, 0, stream.stream()>>>(device.view<MoleculeType::Target>(),
                                                             resultsDev.data(),
                                                             numMols);
  cudaCheckError(cudaGetLastError());

  std::vector<int> results(numMols);
  resultsDev.copyToHost(results);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < numMols; ++i) {
    EXPECT_EQ(results[i], 6) << "Mismatch at molecule " << i;
  }
}

// =============================================================================
// Ring Membership Tests
// =============================================================================

class RingMembershipTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mols_.push_back(makeMol("CCCCCC"));            // hexane: no rings
    mols_.push_back(makeMol("c1ccccc1"));          // benzene: 6-membered ring
    mols_.push_back(makeMol("c1ccc2[nH]ccc2c1"));  // indole: fused 5+6 rings

    for (const auto& mol : mols_) {
      nvMolKit::addToBatch(mol.get(), batch_);
    }
  }

  std::vector<std::unique_ptr<RDKit::ROMol>> mols_;
  MoleculesHost                              batch_;
};

TEST_F(RingMembershipTest, IndoleRingMembershipExhaustive) {
  const auto& indole   = mols_[2];
  const int   numAtoms = indole->getNumAtoms();
  const auto* ringInfo = indole->getRingInfo();

  ASSERT_NE(ringInfo, nullptr) << "Indole ringInfo should not be null";
  ASSERT_TRUE(ringInfo->isInitialized()) << "Indole ringInfo should be initialized";
  ASSERT_EQ(ringInfo->numRings(), 2) << "Indole should have 2 rings (5+6 fused)";

  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch_);

  // Test min ring size
  AsyncDeviceVector<int> minRingSizeDev(numAtoms, stream.stream());
  readAtomPropertyKernel<<<1, numAtoms, 0, stream.stream()>>>(device.view<MoleculeType::Target>(),
                                                              2,
                                                              AtomProperty::MinRingSize,
                                                              minRingSizeDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> minRingSizes(numAtoms);
  minRingSizeDev.copyToHost(minRingSizes);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < numAtoms; ++i) {
    const int expected = ringInfo->minAtomRingSize(i);
    EXPECT_EQ(minRingSizes[i], expected) << "minRingSize mismatch at atom " << i;
  }

  // Test num rings
  AsyncDeviceVector<int> numRingsDev(numAtoms, stream.stream());
  readAtomPropertyKernel<<<1, numAtoms, 0, stream.stream()>>>(device.view<MoleculeType::Target>(),
                                                              2,
                                                              AtomProperty::NumRings,
                                                              numRingsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> numRings(numAtoms);
  numRingsDev.copyToHost(numRings);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < numAtoms; ++i) {
    const int expected = ringInfo->numAtomRings(i);
    EXPECT_EQ(numRings[i], expected) << "numRings mismatch at atom " << i;
  }
}

TEST_F(RingMembershipTest, HexaneHasNoRings) {
  const auto& hexane   = mols_[0];
  const int   numAtoms = hexane->getNumAtoms();
  const auto* ringInfo = hexane->getRingInfo();

  ASSERT_NE(ringInfo, nullptr) << "Hexane ringInfo should not be null";
  ASSERT_TRUE(ringInfo->isInitialized()) << "Hexane ringInfo should be initialized";
  ASSERT_EQ(ringInfo->numRings(), 0) << "Hexane should have 0 rings";

  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch_);

  AsyncDeviceVector<int> numRingsDev(numAtoms, stream.stream());
  readAtomPropertyKernel<<<1, numAtoms, 0, stream.stream()>>>(device.view<MoleculeType::Target>(),
                                                              0,
                                                              AtomProperty::NumRings,
                                                              numRingsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> numRings(numAtoms);
  numRingsDev.copyToHost(numRings);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < numAtoms; ++i) {
    EXPECT_EQ(numRings[i], 0) << "Hexane atom " << i << " should have no rings";
  }
}

TEST_F(RingMembershipTest, BenzeneHasOneRing) {
  const auto& benzene  = mols_[1];
  const int   numAtoms = benzene->getNumAtoms();
  const auto* ringInfo = benzene->getRingInfo();

  ASSERT_NE(ringInfo, nullptr) << "Benzene ringInfo should not be null";
  ASSERT_TRUE(ringInfo->isInitialized()) << "Benzene ringInfo should be initialized";
  ASSERT_EQ(ringInfo->numRings(), 1) << "Benzene should have 1 ring";

  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch_);

  AsyncDeviceVector<int> numRingsDev(numAtoms, stream.stream());
  readAtomPropertyKernel<<<1, numAtoms, 0, stream.stream()>>>(device.view<MoleculeType::Target>(),
                                                              1,
                                                              AtomProperty::NumRings,
                                                              numRingsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> numRings(numAtoms);
  numRingsDev.copyToHost(numRings);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < numAtoms; ++i) {
    EXPECT_EQ(numRings[i], 1) << "Benzene atom " << i << " should be in 1 ring";
  }

  AsyncDeviceVector<int> minRingSizeDev(numAtoms, stream.stream());
  readAtomPropertyKernel<<<1, numAtoms, 0, stream.stream()>>>(device.view<MoleculeType::Target>(),
                                                              1,
                                                              AtomProperty::MinRingSize,
                                                              minRingSizeDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> minRingSizes(numAtoms);
  minRingSizeDev.copyToHost(minRingSizes);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < numAtoms; ++i) {
    EXPECT_EQ(minRingSizes[i], 6) << "Benzene atom " << i << " should have minRingSize 6";
  }
}

// =============================================================================
// Exhaustive Connectivity Tests (methane, neopentane)
// =============================================================================

class ExhaustiveConnectivityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mols_.push_back(makeMol("C"));          // methane: 1 carbon, no bonds
    mols_.push_back(makeMol("CC(C)(C)C"));  // neopentane: central C with 4 neighbors

    for (const auto& mol : mols_) {
      nvMolKit::addToBatch(mol.get(), batch_);
    }
  }

  std::vector<std::unique_ptr<RDKit::ROMol>> mols_;
  MoleculesHost                              batch_;
};

TEST_F(ExhaustiveConnectivityTest, MethaneHasNoNeighbors) {
  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch_);

  AsyncDeviceVector<int> degreeDev(1, stream.stream());
  readAllAtomDegreesKernel<<<1, 1, 0, stream.stream()>>>(device.view<MoleculeType::Target>(), 0, degreeDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> degrees(1);
  degreeDev.copyToHost(degrees);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_EQ(degrees[0], 0) << "Methane carbon should have 0 neighbors (no explicit H)";
}

TEST_F(ExhaustiveConnectivityTest, NeopentaneDegreesExhaustive) {
  // Neopentane: CC(C)(C)C
  // Atom 0: CH3 (peripheral) - degree 1
  // Atom 1: C (central) - degree 4
  // Atom 2: CH3 (peripheral) - degree 1
  // Atom 3: CH3 (peripheral) - degree 1
  // Atom 4: CH3 (peripheral) - degree 1

  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch_);

  const auto& neopentane = mols_[1];
  const int   numAtoms   = neopentane->getNumAtoms();

  AsyncDeviceVector<int> degreeDev(numAtoms, stream.stream());
  readAllAtomDegreesKernel<<<1, numAtoms, 0, stream.stream()>>>(device.view<MoleculeType::Target>(),
                                                                1,
                                                                degreeDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> degrees(numAtoms);
  degreeDev.copyToHost(degrees);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < numAtoms; ++i) {
    const int expected = neopentane->getAtomWithIdx(i)->getDegree();
    EXPECT_EQ(degrees[i], expected) << "Degree mismatch at atom " << i;
  }
}

TEST_F(ExhaustiveConnectivityTest, NeopentaneNeighborsExhaustive) {
  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch_);

  const auto& neopentane = mols_[1];
  const int   numAtoms   = neopentane->getNumAtoms();

  for (int atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
    const auto* atom   = neopentane->getAtomWithIdx(atomIdx);
    const int   degree = atom->getDegree();

    if (degree == 0) {
      continue;
    }

    AsyncDeviceVector<int> neighborsDev(degree, stream.stream());
    readAllNeighborsKernel<<<1, degree, 0, stream.stream()>>>(device.view<MoleculeType::Target>(),
                                                              1,
                                                              atomIdx,
                                                              neighborsDev.data());
    cudaCheckError(cudaGetLastError());

    std::vector<int> neighbors(degree);
    neighborsDev.copyToHost(neighbors);
    cudaCheckError(cudaStreamSynchronize(stream.stream()));

    // Get expected neighbors from RDKit
    std::set<int> expectedNeighbors;
    auto [beg, end] = neopentane->getAtomBonds(atom);
    while (beg != end) {
      const auto* bond = (*neopentane)[*beg];
      expectedNeighbors.insert(bond->getOtherAtomIdx(atomIdx));
      ++beg;
    }

    std::set<int> actualNeighbors(neighbors.begin(), neighbors.end());
    EXPECT_EQ(actualNeighbors, expectedNeighbors) << "Neighbor mismatch at atom " << atomIdx;
  }
}

TEST_F(ExhaustiveConnectivityTest, NeopentaneNeighborBondTypesExhaustive) {
  ScopedStream    stream;
  MoleculesDevice device(stream.stream());
  device.copyFromHost(batch_);

  const auto& neopentane = mols_[1];
  const int   numAtoms   = neopentane->getNumAtoms();

  for (int atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
    const auto* atom   = neopentane->getAtomWithIdx(atomIdx);
    const int   degree = atom->getDegree();

    if (degree == 0) {
      continue;
    }

    AsyncDeviceVector<int> bondTypeDev(degree, stream.stream());
    readAllNeighborBondTypesKernel<<<1, degree, 0, stream.stream()>>>(device.view<MoleculeType::Target>(),
                                                                      1,
                                                                      atomIdx,
                                                                      bondTypeDev.data());
    cudaCheckError(cudaGetLastError());

    std::vector<int> bondTypes(degree);
    bondTypeDev.copyToHost(bondTypes);
    cudaCheckError(cudaStreamSynchronize(stream.stream()));

    // Get expected bond types from RDKit
    std::multiset<int> expectedBondTypes;
    auto [beg, end] = neopentane->getAtomBonds(atom);
    while (beg != end) {
      const auto* bond = (*neopentane)[*beg];
      expectedBondTypes.insert(static_cast<int>(bond->getBondType()));
      ++beg;
    }

    std::multiset<int> actualBondTypes(bondTypes.begin(), bondTypes.end());
    EXPECT_EQ(actualBondTypes, expectedBondTypes) << "Bond type mismatch at atom " << atomIdx;
  }
}

// =============================================================================
// Query Batch Structure Tests (addQueryToBatch produces same structure)
// =============================================================================

struct SmilesSmartsPair {
  std::string smiles;
  std::string smarts;  // Equivalent SMARTS using atomic number syntax
};

class QueryBatchStructureTest : public ::testing::TestWithParam<SmilesSmartsPair> {
 protected:
  std::unique_ptr<RDKit::ROMol> makeMolFromSmiles(const std::string& smiles) {
    auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
    EXPECT_NE(mol, nullptr) << "Failed to parse SMILES: " << smiles;
    return mol;
  }

  std::unique_ptr<RDKit::ROMol> makeMolFromSmarts(const std::string& smarts) {
    auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
    EXPECT_NE(mol, nullptr) << "Failed to parse SMARTS: " << smarts;
    return mol;
  }
};

TEST_P(QueryBatchStructureTest, StructureMatchesBetweenSmilesAndSmarts) {
  const auto& pair = GetParam();

  auto smilesMol = makeMolFromSmiles(pair.smiles);
  auto smartsMol = makeMolFromSmarts(pair.smarts);
  ASSERT_NE(smilesMol, nullptr);
  ASSERT_NE(smartsMol, nullptr);

  MoleculesHost smilesBatch;
  MoleculesHost smartsBatch;

  nvMolKit::addToBatch(smilesMol.get(), smilesBatch);
  nvMolKit::addQueryToBatch(smartsMol.get(), smartsBatch);

  // Verify same structure
  EXPECT_EQ(smilesBatch.numMolecules(), smartsBatch.numMolecules());
  EXPECT_EQ(smilesBatch.totalAtoms(), smartsBatch.totalAtoms());

  // Verify atom data matches (atomic numbers should match)
  ASSERT_EQ(smilesBatch.atomDataPacked.size(), smartsBatch.atomDataPacked.size());
  for (size_t i = 0; i < smilesBatch.atomDataPacked.size(); ++i) {
    EXPECT_EQ(smilesBatch.atomDataPacked[i].atomicNum(), smartsBatch.atomDataPacked[i].atomicNum())
      << "Atomic number mismatch at atom " << i;
  }

  // Verify packed bond data matches (same degree and neighbor structure)
  ASSERT_EQ(smilesBatch.targetAtomBonds.size(), smartsBatch.queryAtomBonds.size());
  for (size_t i = 0; i < smilesBatch.targetAtomBonds.size(); ++i) {
    EXPECT_EQ(smilesBatch.targetAtomBonds[i].degree, smartsBatch.queryAtomBonds[i].degree)
      << "Degree mismatch at atom " << i;
  }
}

TEST_P(QueryBatchStructureTest, DeviceStructureMatchesBetweenSmilesAndSmarts) {
  const auto& pair = GetParam();

  auto smilesMol = makeMolFromSmiles(pair.smiles);
  auto smartsMol = makeMolFromSmarts(pair.smarts);
  ASSERT_NE(smilesMol, nullptr);
  ASSERT_NE(smartsMol, nullptr);

  MoleculesHost smilesBatch;
  MoleculesHost smartsBatch;

  nvMolKit::addToBatch(smilesMol.get(), smilesBatch);
  nvMolKit::addQueryToBatch(smartsMol.get(), smartsBatch);

  ScopedStream    stream;
  MoleculesDevice smilesDevice(stream.stream());
  MoleculesDevice smartsDevice(stream.stream());

  smilesDevice.copyFromHost(smilesBatch);
  smartsDevice.copyFromHost(smartsBatch);

  const int numAtoms = static_cast<int>(smilesBatch.totalAtoms());

  // Compare atomic numbers on device
  AsyncDeviceVector<int> smilesResultsDev(numAtoms, stream.stream());
  AsyncDeviceVector<int> smartsResultsDev(numAtoms, stream.stream());

  readAtomPropertyKernel<<<1, numAtoms, 0, stream.stream()>>>(smilesDevice.view<MoleculeType::Target>(),
                                                              0,
                                                              AtomProperty::AtomicNum,
                                                              smilesResultsDev.data());
  readAtomPropertyKernel<<<1, numAtoms, 0, stream.stream()>>>(smartsDevice.view<MoleculeType::Query>(),
                                                              0,
                                                              AtomProperty::AtomicNum,
                                                              smartsResultsDev.data());
  cudaCheckError(cudaGetLastError());

  std::vector<int> smilesResults(numAtoms);
  std::vector<int> smartsResults(numAtoms);
  smilesResultsDev.copyToHost(smilesResults);
  smartsResultsDev.copyToHost(smartsResults);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  for (int i = 0; i < numAtoms; ++i) {
    EXPECT_EQ(smilesResults[i], smartsResults[i]) << "Device atomic number mismatch at atom " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(SmilesVsSmarts,
                         QueryBatchStructureTest,
                         ::testing::Values(
                           // Same strings work as both SMILES and SMARTS
                           SmilesSmartsPair{"CCO", "CCO"},            // ethanol
                           SmilesSmartsPair{"c1ccccc1", "c1ccccc1"},  // benzene
                           SmilesSmartsPair{"CC(=O)O", "CC(=O)O"},    // acetic acid
                           SmilesSmartsPair{"C#N", "C#N"},            // hydrogen cyanide
                           SmilesSmartsPair{"CCCCCC", "CCCCCC"}       // hexane
                           ),
                         [](const ::testing::TestParamInfo<SmilesSmartsPair>& info) {
                           std::string name = info.param.smiles;
                           for (char& c : name) {
                             if (!std::isalnum(c)) {
                               c = '_';
                             }
                           }
                           return name;
                         });

// =============================================================================
// Re-copy to Device Test
// =============================================================================

TEST(MoleculesReCopyTest, CopyFromHostTwiceWorks) {
  // First batch: small molecules
  std::vector<std::unique_ptr<RDKit::ROMol>> mols1;
  MoleculesHost                              batch1;
  mols1.push_back(makeMol("C"));   // methane
  mols1.push_back(makeMol("CC"));  // ethane
  for (const auto& mol : mols1) {
    nvMolKit::addToBatch(mol.get(), batch1);
  }

  // Second batch: different molecules
  std::vector<std::unique_ptr<RDKit::ROMol>> mols2;
  MoleculesHost                              batch2;
  mols2.push_back(makeMol("c1ccccc1"));  // benzene
  mols2.push_back(makeMol("CCO"));       // ethanol
  mols2.push_back(makeMol("CCCC"));      // butane
  for (const auto& mol : mols2) {
    nvMolKit::addToBatch(mol.get(), batch2);
  }

  ScopedStream    stream;
  MoleculesDevice device(stream.stream());

  // First copy
  device.copyFromHost(batch1);

  AsyncDeviceVector<int> resultsDev(2, stream.stream());
  readNumAtomsKernel<<<1, 2, 0, stream.stream()>>>(device.view<MoleculeType::Target>(), resultsDev.data(), 2);
  cudaCheckError(cudaGetLastError());

  std::vector<int> results1(2);
  resultsDev.copyToHost(results1);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_EQ(results1[0], 1);  // methane: 1 atom
  EXPECT_EQ(results1[1], 2);  // ethane: 2 atoms

  // Second copy (re-copy with different data)
  device.copyFromHost(batch2);

  AsyncDeviceVector<int> resultsDev2(3, stream.stream());
  readNumAtomsKernel<<<1, 3, 0, stream.stream()>>>(device.view<MoleculeType::Target>(), resultsDev2.data(), 3);
  cudaCheckError(cudaGetLastError());

  std::vector<int> results2(3);
  resultsDev2.copyToHost(results2);
  cudaCheckError(cudaStreamSynchronize(stream.stream()));

  EXPECT_EQ(results2[0], 6);  // benzene: 6 atoms
  EXPECT_EQ(results2[1], 3);  // ethanol: 3 atoms
  EXPECT_EQ(results2[2], 4);  // butane: 4 atoms
}

// =============================================================================
// Molecule Size Limit Tests
// =============================================================================

TEST(MoleculesSizeLimitTest, TargetMoleculeExceeding128AtomsThrows) {
  // 129-carbon chain: exceeds 128 atom limit
  std::string longChain(129, 'C');
  auto        mol = makeMol(longChain);
  ASSERT_NE(mol, nullptr);
  ASSERT_GT(mol->getNumAtoms(), 128u);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addToBatch(mol.get(), batch), std::runtime_error);
}

TEST(MoleculesSizeLimitTest, QueryMoleculeExceeding128AtomsThrows) {
  // 129-carbon chain as SMARTS: exceeds 128 atom limit
  std::string longChain(129, 'C');
  auto        mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(longChain));
  ASSERT_NE(mol, nullptr);
  ASSERT_GT(mol->getNumAtoms(), 128u);

  MoleculesHost batch;
  EXPECT_THROW(nvMolKit::addQueryToBatch(mol.get(), batch), std::runtime_error);
}

TEST(MoleculesSizeLimitTest, TargetMoleculeAt128AtomsSucceeds) {
  // Exactly 128 carbons: should succeed
  std::string chain128(128, 'C');
  auto        mol = makeMol(chain128);
  ASSERT_NE(mol, nullptr);
  ASSERT_EQ(mol->getNumAtoms(), 128u);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
  EXPECT_EQ(batch.totalAtoms(), 128);
}

TEST(MoleculesSizeLimitTest, QueryMoleculeAt128AtomsSucceeds) {
  // Exactly 128 carbons as SMARTS: should succeed
  std::string chain128(128, 'C');
  auto        mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(chain128));
  ASSERT_NE(mol, nullptr);
  ASSERT_EQ(mol->getNumAtoms(), 128u);

  MoleculesHost batch;
  EXPECT_NO_THROW(nvMolKit::addQueryToBatch(mol.get(), batch));
  EXPECT_EQ(batch.numMolecules(), 1);
  EXPECT_EQ(batch.totalAtoms(), 128);
}
