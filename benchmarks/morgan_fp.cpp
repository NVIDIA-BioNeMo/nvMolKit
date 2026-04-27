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

#include <getopt.h>
#include <GraphMol/Fingerprints/MorganGenerator.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <nanobench.h>
#include <omp.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/morgan_fingerprint.h"

using namespace RDKit;

namespace {

void printHelp(const char* progName) {
  std::cout << "Usage: " << progName << " [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  -f, --file_path <path>     Input file (.sdf, .smi, .smiles, .cxsmiles)\n";
  std::cout << "                                [default: benchmarks/data/chembl_10k.smi]\n";
  std::cout << "  -n, --num_mols <int>       Number of molecules to process [default: 10000]\n";
  std::cout << "  -t, --num_threads <int>    CPU threads for RDKit and nvMolKit preprocessing [default: OMP max]\n";
  std::cout << "  -g, --num_gpus <int>       Number of GPUs to use (only 1 supported) [default: 1]\n";
  std::cout << "      --radius <int>         Morgan fingerprint radius [default: 2]\n";
  std::cout << "  -s, --fp_size <int>        Fingerprint size {128,256,512,1024,2048,4096} [default: 1024]\n";
  std::cout << "  -r, --do_rdkit <bool>     Run RDKit comparison (true/false) [default: true]\n";
  std::cout << "  -b, --use_nanobench <bool> Use nanobench timing (true/false) [default: true]\n";
  std::cout << "  -B, --batch_size <int>     GPU batch size (molecules per dispatch) [default: auto]\n";
  std::cout << "  -h, --help                 Show this help message\n";
}

int resolveThreads(int requested) {
  if (requested > 0) {
    return requested;
  }
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

bool parseBoolArg(const std::string& arg) {
  std::string s = arg;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  return (s == "1" || s == "true" || s == "yes" || s == "on");
}

template <typename Func> void runOrOnce(bool useBench, const std::string& title, Func&& func) {
  if (useBench) {
    ankerl::nanobench::Bench().warmup(1).epochs(3).run(title, std::forward<Func>(func));
  } else {
    func();
  }
}

}  // namespace

int main(const int argc, char* argv[]) {
  std::string filePath     = "benchmarks/data/chembl_10k.smi";
  int         numMols      = 10000;
  int         numThreads   = -1;  // If <0, use OMP max
  int         numGpus      = 1;   // Only 1 supported
  int         radius       = 2;
  int         fpSize       = 1024;
  bool        doRdkit      = true;
  bool        useNanobench = true;
  int         batchSize    = -1;  // If <0, leave as auto

  static struct option long_options[] = {
    {    "file_path", required_argument, 0, 'f'},
    {     "num_mols", required_argument, 0, 'n'},
    {  "num_threads", required_argument, 0, 't'},
    {     "num_gpus", required_argument, 0, 'g'},
    {       "radius", required_argument, 0,   0},
    {      "fp_size", required_argument, 0, 's'},
    {     "do_rdkit", required_argument, 0, 'r'},
    {"use_nanobench", required_argument, 0, 'b'},
    {   "batch_size", required_argument, 0, 'B'},
    {         "help",       no_argument, 0, 'h'},
    {              0,                 0, 0,   0}
  };

  int option_index = 0;
  int c;
  while ((c = getopt_long(argc, argv, "f:n:t:g:s:r:b:B:h", long_options, &option_index)) != -1) {
    switch (c) {
      case 'f':
        filePath = optarg;
        break;
      case 'n':
        try {
          numMols = std::stoi(optarg);
          if (numMols <= 0) {
            std::cerr << "Error: num_mols must be positive\n";
            return 1;
          }
        } catch (...) {
          std::cerr << "Error: Invalid value for num_mols: " << optarg << "\n";
          return 1;
        }
        break;
      case 't':
        try {
          numThreads = std::stoi(optarg);
        } catch (...) {
          std::cerr << "Error: Invalid value for num_threads: " << optarg << "\n";
          return 1;
        }
        break;
      case 'g':
        try {
          numGpus = std::stoi(optarg);
        } catch (...) {
          std::cerr << "Error: Invalid value for num_gpus: " << optarg << "\n";
          return 1;
        }
        break;
      case 0:  // long option without short name
        if (std::string(long_options[option_index].name) == "radius") {
          try {
            radius = std::stoi(optarg);
            if (radius <= 0) {
              std::cerr << "Error: radius must be positive\n";
              return 1;
            }
          } catch (...) {
            std::cerr << "Error: Invalid value for radius: " << optarg << "\n";
            return 1;
          }
        }
        break;
      case 's':
        try {
          fpSize = std::stoi(optarg);
        } catch (...) {
          std::cerr << "Error: Invalid value for fp_size: " << optarg << "\n";
          return 1;
        }
        break;
      case 'r':
        doRdkit = parseBoolArg(optarg);
        break;
      case 'b':
        useNanobench = parseBoolArg(optarg);
        break;
      case 'B':
        try {
          batchSize = std::stoi(optarg);
          if (batchSize <= 0) {
            std::cerr << "Error: batch_size must be positive\n";
            return 1;
          }
        } catch (...) {
          std::cerr << "Error: Invalid value for batch_size: " << optarg << "\n";
          return 1;
        }
        break;
      case 'h':
        printHelp(argv[0]);
        return 0;
      case '?':
      default:
        std::cerr << "\nUse --help for usage information.\n";
        return 1;
    }
  }

  if (optind < argc) {
    std::cerr << "Error: Unexpected non-option arguments: ";
    while (optind < argc) {
      std::cerr << argv[optind++] << " ";
    }
    std::cerr << "\nUse --help for usage information.\n";
    return 1;
  }

  if (!std::filesystem::exists(filePath)) {
    std::cerr << "Error: File does not exist: " << filePath << std::endl;
    return 1;
  }

  if (numGpus != 1) {
    throw std::runtime_error("Multi-GPU not supported yet. Please set --num_gpus=1");
  }

  const int threadsResolved = resolveThreads(numThreads);

  std::cout << "Configuration:\n";
  std::cout << "  File path: " << filePath << "\n";
  std::cout << "  Number of molecules: " << numMols << "\n";
  std::cout << "  Radius: " << radius << "\n";
  std::cout << "  Fingerprint size: " << fpSize << "\n";
  std::cout << "  RDKit/nvMolKit CPU threads: " << threadsResolved << "\n";
  std::cout << "  Number of GPUs: 1 (only 1 supported)\n\n";
  std::cout << "  Run RDKit comparison: " << (doRdkit ? "yes" : "no") << "\n";
  std::cout << "  Use nanobench: " << (useNanobench ? "yes" : "no") << "\n\n";
  if (batchSize > 0) {
    std::cout << "  GPU batch size: " << batchSize << "\n\n";
  }

  // Read molecules: support SMILES and CXSMILES; ignore SDF to keep it simple
  const std::string extension = std::filesystem::path(filePath).extension().string();
  auto              toLower   = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
  };
  const std::string extLower = toLower(extension);

  if (!(extLower == ".smi" || extLower == ".smiles" || extLower == ".cxsmiles")) {
    std::cerr << "Error: Only SMILES-like inputs are supported (.smi, .smiles, .cxsmiles)" << std::endl;
    return 1;
  }

  std::ifstream infile(filePath);
  if (!infile.is_open()) {
    std::cerr << "Error: Could not open input file: " << filePath << std::endl;
    return 1;
  }

  std::vector<std::unique_ptr<RDKit::ROMol>> owningMols;
  owningMols.reserve(static_cast<size_t>(numMols));
  std::string line;
  while (static_cast<int>(owningMols.size()) < numMols && std::getline(infile, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    // Take first token up to whitespace or tab
    const auto        endPos = line.find_first_of(" \t");
    const std::string smiles = endPos == std::string::npos ? line : line.substr(0, endPos);
    try {
      std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(smiles));
      if (mol) {
        owningMols.push_back(std::move(mol));
      }
    } catch (const std::exception& e) {
      // Skip invalid lines
    }
  }
  infile.close();

  if (owningMols.empty()) {
    std::cerr << "Error: No valid molecules parsed from SMILES file." << std::endl;
    return 1;
  }
  // If fewer than requested, duplicate to reach numMols
  while (static_cast<int>(owningMols.size()) < numMols) {
    const size_t idx = owningMols.size() % owningMols.size();
    owningMols.push_back(std::make_unique<RDKit::ROMol>(*owningMols[idx]));
  }

  // Build pointer view
  std::vector<const RDKit::ROMol*> mols;
  mols.reserve(owningMols.size());
  for (auto& m : owningMols) {
    mols.push_back(static_cast<const RDKit::ROMol*>(m.get()));
  }

  auto rdkitGen =
    std::unique_ptr<FingerprintGenerator<std::uint32_t>>(MorganFingerprint::getMorganGenerator<std::uint32_t>(radius));
  nvMolKit::MorganFingerprintGenerator nvmolkitGen(radius, static_cast<std::uint32_t>(fpSize));

  // RDKit benchmark (batch API with numThreads)
  if (doRdkit) {
    const std::string title = "RDKit Morgan fingerprint, radius=" + std::to_string(radius) +
                              ", num_mols=" + std::to_string(numMols) + ", threads=" + std::to_string(threadsResolved);
    runOrOnce(useNanobench, title, [&] {
      ankerl::nanobench::doNotOptimizeAway(rdkitGen->getFingerprints(mols, threadsResolved));
    });
  }

  // nvMolKit GPU benchmark (templated on fp size)
  {
    nvMolKit::FingerprintComputeOptions options;
    options.backend       = nvMolKit::FingerprintComputeBackend::GPU;
    options.numCpuThreads = threadsResolved;
    if (batchSize > 0) {
      options.gpuBatchSize = batchSize;
    }
    const std::string title = "nvMolKit Morgan fingerprint (GPU), radius=" + std::to_string(radius) +
                              ", num_mols=" + std::to_string(numMols) + ", threads=" + std::to_string(threadsResolved);
    runOrOnce(useNanobench, title, [&] {
      switch (fpSize) {
        case 128:
          ankerl::nanobench::doNotOptimizeAway(nvmolkitGen.GetFingerprintsGpuBuffer<128>(mols, nullptr, options));
          break;
        case 256:
          ankerl::nanobench::doNotOptimizeAway(nvmolkitGen.GetFingerprintsGpuBuffer<256>(mols, nullptr, options));
          break;
        case 512:
          ankerl::nanobench::doNotOptimizeAway(nvmolkitGen.GetFingerprintsGpuBuffer<512>(mols, nullptr, options));
          break;
        case 1024:
          ankerl::nanobench::doNotOptimizeAway(nvmolkitGen.GetFingerprintsGpuBuffer<1024>(mols, nullptr, options));
          break;
        case 2048:
          ankerl::nanobench::doNotOptimizeAway(nvmolkitGen.GetFingerprintsGpuBuffer<2048>(mols, nullptr, options));
          break;
        case 4096:
          ankerl::nanobench::doNotOptimizeAway(nvmolkitGen.GetFingerprintsGpuBuffer<4096>(mols, nullptr, options));
          break;
        default:
          throw std::runtime_error("Unsupported fp_size. Must be one of {128,256,512,1024,2048,4096}");
      }
      // FIXME: MultiGPU needs different handling, probably in the function and we'll take the tiny perf hit.
      cudaDeviceSynchronize();
    });
  }

  return 0;
}
