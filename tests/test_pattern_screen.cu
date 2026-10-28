// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <DataStructs/BitOps.h>
#include <DataStructs/ExplicitBitVect.h>
#include <GraphMol/Fingerprints/Fingerprints.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/substruct/molecules.h"
#include "src/substruct/pattern_screen.h"
#include "src/testutils/mol_data.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"
#include "tests/test_utils.h"

namespace {

using nvMolKit::checkReturnCode;
using nvMolKit::kPatternFingerprintBits;
using nvMolKit::kPatternFingerprintWords;

std::vector<std::unique_ptr<RDKit::ROMol>> chemblTargets(std::size_t count) {
  return nvMolKit::testing::readSmilesFile(getTestDataFolderPath() + "/chembl_1k.smi", count, 128);
}

std::unique_ptr<ExplicitBitVect> patternFingerprint(const RDKit::ROMol& mol) {
  return std::unique_ptr<ExplicitBitVect>(RDKit::PatternFingerprintMol(mol));
}

TEST(PatternScreenHost, PacksFingerprintsWordMajorAndTransposesThemToBitSlices) {
  // 70 targets span three 32-target slice words, the last one partial.
  const auto targets = chemblTargets(70);
  ASSERT_EQ(targets.size(), 70U);
  std::vector<std::unique_ptr<ExplicitBitVect>> fingerprints;
  std::vector<const ExplicitBitVect*>           fingerprintPtrs;
  for (const auto& target : targets) {
    fingerprints.push_back(patternFingerprint(*target));
    fingerprintPtrs.push_back(fingerprints.back().get());
  }

  const std::size_t count       = targets.size();
  const auto        words       = nvMolKit::packPatternFingerprintsWordMajor(fingerprintPtrs);
  const auto        slices      = nvMolKit::buildPatternBitSlices(words, count);
  const auto        frequencies = nvMolKit::patternBitFrequencies(slices, count);
  const std::size_t sliceWords  = nvMolKit::patternSliceWords(count);
  ASSERT_EQ(words.size(), count * kPatternFingerprintWords);
  ASSERT_EQ(sliceWords, 3U);
  ASSERT_EQ(slices.size(), kPatternFingerprintBits * sliceWords);
  ASSERT_EQ(frequencies.size(), static_cast<std::size_t>(kPatternFingerprintBits));

  for (int bit = 0; bit < kPatternFingerprintBits; ++bit) {
    std::uint32_t carriers = 0;
    for (std::size_t target = 0; target < count; ++target) {
      const bool expected = fingerprints[target]->getBit(static_cast<unsigned int>(bit));
      const auto word     = words[static_cast<std::size_t>(bit / 64) * count + target];
      const auto slice    = slices[static_cast<std::size_t>(bit) * sliceWords + target / 32];
      ASSERT_EQ(((word >> (bit % 64)) & 1U) != 0, expected) << "bit " << bit << ", target " << target;
      ASSERT_EQ(((slice >> (target % 32)) & 1U) != 0, expected) << "bit " << bit << ", target " << target;
      carriers += expected ? 1U : 0U;
    }
    ASSERT_EQ(frequencies[static_cast<std::size_t>(bit)], carriers) << "bit " << bit;
  }
}

TEST(PatternScreenHost, QueryBitsAreTheFingerprintBitsOrderedRarestFirst) {
  const auto query       = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol("c1ccccc1C(=O)N"));
  const auto fingerprint = patternFingerprint(*query);
  auto       screenQuery = nvMolKit::makePatternScreenQuery(fingerprint.get(), static_cast<int>(query->getNumAtoms()));

  std::vector<std::uint16_t> onBits;
  for (unsigned int bit = 0; bit < fingerprint->getNumBits(); ++bit) {
    if (fingerprint->getBit(bit)) {
      onBits.push_back(static_cast<std::uint16_t>(bit));
    }
  }
  ASSERT_FALSE(onBits.empty());
  EXPECT_EQ(screenQuery.bits, onBits);
  EXPECT_EQ(screenQuery.numAtoms, 9);

  // Frequencies cycle so some bits tie; ties keep ascending bit order.
  std::vector<std::uint32_t> frequencies(kPatternFingerprintBits);
  for (int bit = 0; bit < kPatternFingerprintBits; ++bit) {
    frequencies[static_cast<std::size_t>(bit)] = static_cast<std::uint32_t>((bit * 7) % 5);
  }
  nvMolKit::orderPatternScreenBits(screenQuery, frequencies);
  ASSERT_EQ(screenQuery.bits.size(), onBits.size());
  EXPECT_TRUE(std::is_permutation(screenQuery.bits.begin(), screenQuery.bits.end(), onBits.begin()));
  for (std::size_t index = 1; index < screenQuery.bits.size(); ++index) {
    const auto previous = screenQuery.bits[index - 1];
    const auto current  = screenQuery.bits[index];
    ASSERT_TRUE(frequencies[previous] < frequencies[current] ||
                (frequencies[previous] == frequencies[current] && previous < current));
  }

  EXPECT_TRUE(nvMolKit::makePatternScreenQuery(nullptr, 4).bits.empty());
}

// Packed targets and their fingerprint bit slices on the current device.
struct ScreenTargets {
  std::vector<const RDKit::ROMol*>              molecules;
  std::vector<std::unique_ptr<ExplicitBitVect>> fingerprints;
  std::vector<std::uint32_t>                    bitFrequencies;
  nvMolKit::MoleculesHost                       host;
  std::unique_ptr<nvMolKit::MoleculesDevice>    device;
  nvMolKit::AsyncDeviceVector<std::uint32_t>    slices;

  ScreenTargets(std::vector<const RDKit::ROMol*> targets, cudaStream_t stream) : molecules(std::move(targets)) {
    std::vector<const ExplicitBitVect*> fingerprintPtrs;
    for (const auto* molecule : molecules) {
      fingerprints.push_back(patternFingerprint(*molecule));
      fingerprintPtrs.push_back(fingerprints.back().get());
    }
    const auto hostSlices =
      nvMolKit::buildPatternBitSlices(nvMolKit::packPatternFingerprintsWordMajor(fingerprintPtrs), molecules.size());
    bitFrequencies = nvMolKit::patternBitFrequencies(hostSlices, molecules.size());
    nvMolKit::buildTargetBatchParallelInto(host, 2, molecules, {});
    device = std::make_unique<nvMolKit::MoleculesDevice>(stream);
    device->copyFromHost(host, stream);
    slices = nvMolKit::AsyncDeviceVector<std::uint32_t>(hostSlices.size(), stream);
    slices.copyFromHost(hostSlices);
    cudaCheckError(cudaStreamSynchronize(stream));
  }

  [[nodiscard]] const int* batchAtomStarts() const {
    return device->view<nvMolKit::MoleculeType::Target>().batchAtomStarts;
  }
};

TEST(PatternScreenDevice, KeepsExactlyTheTargetsThatCanContainTheQuery) {
  // 90 and 200 targets span partial 32-target words; screening the smaller set first makes the workspace grow.
  const auto targets = chemblTargets(200);
  ASSERT_EQ(targets.size(), 200U);
  std::vector<const RDKit::ROMol*> pointers;
  for (const auto& target : targets) {
    pointers.push_back(target.get());
  }
  nvMolKit::ScopedStream stream("PatternScreenDevice");
  const ScreenTargets small(std::vector<const RDKit::ROMol*>(pointers.begin(), pointers.begin() + 90), stream.stream());
  const ScreenTargets all(pointers, stream.stream());

  int deviceId = 0;
  ASSERT_EQ(cudaGetDevice(&deviceId), cudaSuccess);
  nvMolKit::PatternScreenWorkspace workspace(deviceId);
  const auto                       ring = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol("c1ccccc1"));
  workspace.screen(small.slices.data(),
                   small.batchAtomStarts(),
                   static_cast<int>(small.molecules.size()),
                   nvMolKit::makePatternScreenQuery(patternFingerprint(*ring).get(), 6));

  const int numTargets = static_cast<int>(all.molecules.size());
  for (const char* smarts :
       {"c1ccccc1", "[OX2H]", "C(=O)N", "[#7;R]", "S(=O)(=O)", "[Cl,Br]", "CCCCCCCCCCCC", "[$(C=O)]", "[Si]"}) {
    const auto query       = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
    const auto fingerprint = patternFingerprint(*query);
    const int  queryAtoms  = static_cast<int>(query->getNumAtoms());
    auto       screenQuery = nvMolKit::makePatternScreenQuery(fingerprint.get(), queryAtoms);
    nvMolKit::orderPatternScreenBits(screenQuery, all.bitFrequencies);
    workspace.screen(all.slices.data(), all.batchAtomStarts(), numTargets, screenQuery);
    const std::vector<int> survivors(workspace.indices(), workspace.indices() + workspace.count());

    std::vector<int> expected;
    for (int target = 0; target < numTargets; ++target) {
      const auto& mol = *all.molecules[static_cast<std::size_t>(target)];
      if (static_cast<int>(mol.getNumAtoms()) >= queryAtoms &&
          AllProbeBitsMatch(*fingerprint, *all.fingerprints[static_cast<std::size_t>(target)])) {
        expected.push_back(target);
      }
      RDKit::MatchVectType match;
      if (RDKit::SubstructMatch(mol, *query, match)) {
        EXPECT_TRUE(std::binary_search(survivors.begin(), survivors.end(), target))
          << smarts << " rejected matching target " << target;
      }
    }
    EXPECT_EQ(survivors, expected) << smarts;
  }

  // A query larger than every target leaves no survivors.
  const auto chain = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(std::string(129, 'C')));
  workspace.screen(all.slices.data(),
                   all.batchAtomStarts(),
                   numTargets,
                   nvMolKit::makePatternScreenQuery(patternFingerprint(*chain).get(), 129));
  EXPECT_EQ(workspace.count(), 0);

  // Without fingerprints the screen keeps every target with enough atoms.
  workspace.screen(nullptr, all.batchAtomStarts(), numTargets, nvMolKit::makePatternScreenQuery(nullptr, 30));
  std::vector<int> largeEnough;
  for (int target = 0; target < numTargets; ++target) {
    if (all.molecules[static_cast<std::size_t>(target)]->getNumAtoms() >= 30U) {
      largeEnough.push_back(target);
    }
  }
  EXPECT_EQ(std::vector<int>(workspace.indices(), workspace.indices() + workspace.count()), largeEnough);
}

}  // namespace
