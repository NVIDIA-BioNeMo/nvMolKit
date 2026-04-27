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

#include "src/testutils/mol_data.h"

#include <GraphMol/SmilesParse/SmilesParse.h>

#include <filesystem>
#include <fstream>

namespace nvMolKit::testing {

std::vector<std::string> loadLines(const std::string& filePath, const size_t maxLines) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file " + filePath);
  }

  std::vector<std::string> lines;
  std::string              line;
  while (std::getline(file, line) && lines.size() < maxLines) {
    lines.push_back(line);
  }

  file.close();
  return lines;
}

std::string extractFirstToken(const std::string& line) {
  const char* ws    = " \t\n\r";
  size_t      start = line.find_first_not_of(ws);
  if (start == std::string::npos) {
    return "";
  }
  if (line[start] == '#') {
    return "";
  }
  size_t end = line.find_first_of(ws, start);
  if (end == std::string::npos) {
    return line.substr(start);
  }
  return line.substr(start, end - start);
}

std::pair<std::vector<std::unique_ptr<RDKit::ROMol>>, std::vector<std::string>>
loadNMols(const std::vector<std::string>& smiles, size_t n, size_t atomBondSizeCutoff) {
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  std::vector<std::string>                   outSmiles;
  while (mols.size() < n) {
    for (const auto& smi : smiles) {
      std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(smi));
      assert(mol != nullptr);
      if (mol->getNumAtoms() <= atomBondSizeCutoff && mol->getNumBonds() <= atomBondSizeCutoff) {
        mols.push_back(std::move(mol));
        outSmiles.push_back(smi);
      }
      if (mols.size() == n) {
        break;
      }
    }
  }
  assert(mols.size() == n);
  return {std::move(mols), std::move(outSmiles)};
}

std::vector<const RDKit::ROMol*> makeMolsView(const std::vector<std::unique_ptr<RDKit::ROMol>>& mols) {
  std::vector<const RDKit::ROMol*> molsView;
  molsView.reserve(mols.size());
  for (const auto& mol : mols) {
    molsView.push_back(mol.get());
  }
  return molsView;
}

std::string getFilePathInSameDirectory(const std::string& fileName) {
  // Get the current file's path
  const std::string currentFilePath = __FILE__;

  // Extract the directory path
  const std::filesystem::path dirPath = std::filesystem::path(currentFilePath).parent_path();

  // Append the target file name to the directory path
  const std::filesystem::path targetFilePath = dirPath / fileName;

  return targetFilePath.string();
}

std::string getCheml29SmilesPath() {
  const char* path = std::getenv("NVMOLKIT_CHEMBL29_SMILES_PATH");
  if (path == nullptr) {
    return getFilePathInSameDirectory("chembl_1k_smis.txt");
    // Use the chembl_1k_smis.txt path
  }
  return path;
}

std::pair<std::vector<std::unique_ptr<RDKit::ROMol>>, std::vector<std::string>> loadNChemblMolecules(
  size_t                n,
  std::optional<size_t> atomBondSizeCutoff) {
  const std::string              filePath   = getCheml29SmilesPath();
  const std::vector<std::string> testSmiles = loadLines(filePath, n);
  return loadNMols(testSmiles, n, atomBondSizeCutoff.value_or(std::numeric_limits<size_t>::max()));
}

namespace {

template <typename ParseFunc>
std::pair<std::vector<std::unique_ptr<RDKit::ROMol>>, std::vector<std::string>> readMolFileWithStrings(
  const std::string& filePath,
  ParseFunc          parseFunc,
  size_t             maxCount = std::numeric_limits<size_t>::max(),
  size_t             maxAtoms = std::numeric_limits<size_t>::max()) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filePath);
  }

  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  std::vector<std::string>                   molStrings;
  std::string                                line;

  while (std::getline(file, line) && mols.size() < maxCount) {
    std::string molStr = extractFirstToken(line);
    if (molStr.empty()) {
      continue;
    }

    try {
      auto mol = std::unique_ptr<RDKit::ROMol>(parseFunc(molStr));
      if (mol && mol->getNumAtoms() <= maxAtoms) {
        mols.push_back(std::move(mol));
        molStrings.push_back(molStr);
      }
    } catch (const std::exception&) {
    }
  }

  return {std::move(mols), std::move(molStrings)};
}

}  // namespace

std::pair<std::vector<std::unique_ptr<RDKit::ROMol>>, std::vector<std::string>>
readSmilesFileWithStrings(const std::string& filePath, size_t maxCount, size_t maxAtoms) {
  return readMolFileWithStrings(
    filePath,
    [](const std::string& s) { return RDKit::SmilesToMol(s); },
    maxCount,
    maxAtoms);
}

std::pair<std::vector<std::unique_ptr<RDKit::ROMol>>, std::vector<std::string>> readSmartsFileWithStrings(
  const std::string& filePath) {
  return readMolFileWithStrings(filePath, [](const std::string& s) { return RDKit::SmartsToMol(s); });
}

std::vector<std::unique_ptr<RDKit::ROMol>> readSmilesFile(const std::string& filePath,
                                                          size_t             maxCount,
                                                          size_t             maxAtoms) {
  return readSmilesFileWithStrings(filePath, maxCount, maxAtoms).first;
}

std::vector<std::unique_ptr<RDKit::ROMol>> readSmartsFile(const std::string& filePath) {
  return readSmartsFileWithStrings(filePath).first;
}

}  // namespace nvMolKit::testing
