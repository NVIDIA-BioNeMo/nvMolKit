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

/**
 * @file smarts_filter.cpp
 * @brief Tool to filter SMARTS patterns into supported/unsupported/invalid lists.
 *
 * Usage: smarts_filter <input_file> <supported_output> <unsupported_output> <invalid_output>
 *
 * Reads SMARTS patterns (one per line) from input_file and categorizes each:
 *   - supported_output:   Valid SMARTS that nvMolKit can process
 *   - unsupported_output: Valid SMARTS that nvMolKit cannot process (with error message)
 *   - invalid_output:     Invalid SMARTS that RDKit cannot parse
 */

#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "src/substruct/molecules.h"
#include "src/testutils/mol_data.h"

namespace {

enum class SmartsStatus {
  Supported,    // Valid SMARTS, nvMolKit can process
  Unsupported,  // Valid SMARTS, nvMolKit cannot process
  Invalid       // RDKit cannot parse
};

/**
 * @brief Test if a SMARTS pattern is supported by nvMolKit.
 *
 * @param smarts The SMARTS pattern to test
 * @param errorMsg Output parameter for error message if not supported
 * @return SmartsStatus indicating whether pattern is supported, unsupported, or invalid
 */
SmartsStatus classifySmarts(const std::string& smarts, std::string& errorMsg) {
  // Try to parse with RDKit
  std::unique_ptr<RDKit::ROMol> mol(RDKit::SmartsToMol(smarts));
  if (!mol) {
    errorMsg = "RDKit parse error";
    return SmartsStatus::Invalid;
  }

  // Try to add to nvMolKit batch
  try {
    nvMolKit::MoleculesHost batch;
    nvMolKit::addQueryToBatch(mol.get(), batch);

    // Also validate inner patterns of recursive SMARTS ($(...))
    auto recursiveInfo = nvMolKit::extractRecursivePatterns(mol.get());
    for (const auto& entry : recursiveInfo.patterns) {
      if (entry.queryMol != nullptr) {
        nvMolKit::MoleculesHost innerBatch;
        nvMolKit::addQueryToBatch(entry.queryMol, innerBatch);
      }
    }

    return SmartsStatus::Supported;
  } catch (const std::exception& e) {
    errorMsg = e.what();
    return SmartsStatus::Unsupported;
  }
}

void printUsage(const char* progName) {
  std::cerr << "Usage: " << progName << " <input_file> <supported_output> <unsupported_output> <invalid_output>\n"
            << "\n"
            << "Filters SMARTS patterns into supported/unsupported/invalid lists.\n"
            << "\n"
            << "Arguments:\n"
            << "  input_file         File containing SMARTS patterns (one per line)\n"
            << "  supported_output   Output file for patterns nvMolKit can process\n"
            << "  unsupported_output Output file for valid SMARTS nvMolKit cannot process\n"
            << "  invalid_output     Output file for invalid SMARTS (RDKit parse errors)\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printUsage(argv[0]);
    return 1;
  }

  const std::string inputFile       = argv[1];
  const std::string supportedFile   = argv[2];
  const std::string unsupportedFile = argv[3];
  const std::string invalidFile     = argv[4];

  std::ifstream input(inputFile);
  if (!input) {
    std::cerr << "Error: Cannot open input file: " << inputFile << "\n";
    return 1;
  }

  std::ofstream supported(supportedFile);
  if (!supported) {
    std::cerr << "Error: Cannot open supported output file: " << supportedFile << "\n";
    return 1;
  }

  std::ofstream unsupported(unsupportedFile);
  if (!unsupported) {
    std::cerr << "Error: Cannot open unsupported output file: " << unsupportedFile << "\n";
    return 1;
  }

  std::ofstream invalid(invalidFile);
  if (!invalid) {
    std::cerr << "Error: Cannot open invalid output file: " << invalidFile << "\n";
    return 1;
  }

  int                        totalCount       = 0;
  int                        supportedCount   = 0;
  int                        unsupportedCount = 0;
  int                        invalidCount     = 0;
  int                        emptyCount       = 0;
  std::map<std::string, int> reasonCounts;

  std::string line;
  while (std::getline(input, line)) {
    std::string smarts = nvMolKit::testing::extractFirstToken(line);

    // Skip empty lines and comments
    if (smarts.empty() || smarts[0] == '#') {
      emptyCount++;
      continue;
    }

    totalCount++;
    std::string errorMsg;

    switch (classifySmarts(smarts, errorMsg)) {
      case SmartsStatus::Supported:
        supported << smarts << "\n";
        supportedCount++;
        break;
      case SmartsStatus::Unsupported:
        unsupported << "# " << errorMsg << "\n" << smarts << "\n";
        unsupportedCount++;
        reasonCounts[errorMsg]++;
        break;
      case SmartsStatus::Invalid:
        invalid << smarts << "\n";
        invalidCount++;
        break;
    }
  }

  std::cout << "Processed " << totalCount << " SMARTS patterns:\n"
            << "  Supported:   " << supportedCount << "\n"
            << "  Unsupported: " << unsupportedCount << "\n"
            << "  Invalid:     " << invalidCount << "\n";
  if (emptyCount > 0) {
    std::cout << "  (Skipped " << emptyCount << " empty/comment lines)\n";
  }

  if (!reasonCounts.empty()) {
    std::cout << "\nUnsupported reasons:\n";
    for (const auto& [reason, count] : reasonCounts) {
      std::cout << "  " << count << "x " << reason << "\n";
    }
  }

  return 0;
}
