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
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <nanobench.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/embedder_utils.h"
#include "src/etkdg.h"
#include "src/minimizer/bfgs_minimize.h"
#include "src/testutils/conformer_checkers.h"
#include "tests/test_utils.h"

constexpr int maxAtoms = 256;

// Helper function to read molecules from SDF or SMI file
std::vector<std::unique_ptr<RDKit::RWMol>> readMolecules(const std::string& filePath, unsigned int count) {
  std::vector<std::unique_ptr<RDKit::ROMol>> tempMols;

  // Determine file type based on extension
  std::string extension = std::filesystem::path(filePath).extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

  if (extension == ".sdf") {
    // Handle SDF file
    getMols(filePath, tempMols, count);
  } else if (extension == ".smi" || extension == ".smiles" || extension == ".cxsmiles") {
    // Handle SMI/SMILES file
    std::ifstream file(filePath);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open SMILES file: " + filePath);
    }

    std::string                                line;
    std::vector<std::unique_ptr<RDKit::ROMol>> allMols;

    while (std::getline(file, line) && allMols.size() < count) {
      // Skip empty lines and comments
      if (line.empty() || line[0] == '#') {
        continue;
      }

      // Extract SMILES (first part before any whitespace)
      std::string smiles = line.substr(0, line.find_first_of(" \t"));

      try {
        auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
        if (mol && mol->getNumAtoms() <= maxAtoms) {
          allMols.push_back(std::move(mol));
        } else if (mol) {
          std::cerr << "Warning: Molecule with SMILES " << smiles << " has more than " << maxAtoms
                    << " atoms and will be skipped." << std::endl;
        }
      } catch (const std::exception& e) {
        // Skip invalid SMILES
        std::cerr << "Warning: Failed to parse SMILES: " << smiles << " - " << e.what() << std::endl;
      }
    }

    if (allMols.empty()) {
      throw std::runtime_error("No valid molecules found in SMILES file: " + filePath);
    }

    // If we don't have enough molecules, duplicate from the beginning
    if (allMols.size() < count) {
      size_t originalSize = allMols.size();
      for (size_t i = 0; i < count - originalSize; ++i) {
        allMols.push_back(std::make_unique<RDKit::ROMol>(*allMols[i % originalSize]));
      }
    }

    tempMols = std::move(allMols);
  } else {
    throw std::runtime_error(
      "Unsupported file format. Only .sdf, .smi, .smiles, and .cxsmiles files are supported. Got " + extension);
  }

  std::vector<std::unique_ptr<RDKit::RWMol>> mols;
  for (auto& tempMol : tempMols) {
    std::unique_ptr<RDKit::ROMol> mol2(RDKit::MolOps::addHs(*tempMol));
    mols.push_back(std::make_unique<RDKit::RWMol>(*mol2));
    mols.back()->clearConformers();
    RDKit::MolOps::sanitizeMol(*mols.back());
  }
  return mols;
}

std::vector<double> getMMFFEnergies(const std::vector<RDKit::ROMol*>& mols, const int numConfs) {
  std::vector<double> energies;
  energies.reserve(mols.size() * numConfs);

  for (auto* mol : mols) {
    RDKit::MMFF::MMFFMolProperties mmffProps(*mol);

    for (unsigned int i = 0; i < numConfs; ++i) {
      if (i >= mol->getNumConformers()) {
        energies.push_back(0.0);
        continue;
      }

      const int confId = mol->getConformer(i).getId();
      try {
        std::unique_ptr<ForceFields::ForceField> ff(RDKit::MMFF::constructForceField(*mol, &mmffProps, confId));
        if (ff) {
          std::vector<double> flatPos;
          const auto&         conf = mol->getConformer(i);
          for (unsigned int j = 0; j < mol->getNumAtoms(); ++j) {
            const auto& pos = conf.getAtomPos(j);
            flatPos.push_back(pos.x);
            flatPos.push_back(pos.y);
            flatPos.push_back(pos.z);
          }
          energies.push_back(ff->calcEnergy(flatPos.data()));
        } else {
          energies.push_back(0);
        }
      } catch (const Invar::Invariant&) {
      }
    }
  }
  return energies;
}

// Benchmark nvMolKit's embedMolecules
void benchNvMolKit(const std::vector<RDKit::ROMol*>& mols,
                   const int                         numConfs,
                   std::vector<double>&              energies,
                   const bool                        doEnergyCheck,
                   std::vector<int>&                 failCounts,
                   const int                         numThreads,
                   const int                         batchSize,
                   const int                         batchesPerGpu,
                   const int                         maxIterations,
                   const int                         numGpus,
                   const nvMolKit::BfgsBackend       backend) {
  std::string backendName = (backend == nvMolKit::BfgsBackend::BATCHED) ? "BATCHED" :
                            (backend == nvMolKit::BfgsBackend::HYBRID)  ? "HYBRID" :
                                                                          "PER_MOLECULE";
  std::string benchName   = "nvMolKit EmbedMultipleConfs, num_mols=" + std::to_string(mols.size()) +
                          ", num_confs=" + std::to_string(numConfs) + ", batch_size=" + std::to_string(batchSize) +
                          ", num_concurrent_batches=" + std::to_string(batchesPerGpu) +
                          ", num_threads=" + std::to_string(numThreads) +
                          ", max_iterations=" + std::to_string(maxIterations) + ", backend=" + backendName;

  // Create copies for nvMolKit benchmark
  std::vector<std::unique_ptr<RDKit::RWMol>> molCopies;
  std::vector<RDKit::ROMol*>                 molCopyPtrs;
  for (const auto& mol : mols) {
    molCopies.push_back(std::make_unique<RDKit::RWMol>(*mol));
    molCopyPtrs.push_back(molCopies.back().get());
  }

  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.maxIterations                        = maxIterations;
  params.useRandomCoords                      = true;
  params.basinThresh                          = 1e8;
  std::vector<std::vector<int16_t>> fails;

  // Create hardware options struct
  nvMolKit::BatchHardwareOptions hardwareOptions;
  hardwareOptions.preprocessingThreads = numThreads;
  hardwareOptions.batchSize            = batchSize;
  hardwareOptions.batchesPerGpu        = batchesPerGpu;
  if (numGpus > 0) {
    hardwareOptions.gpuIds.clear();
    hardwareOptions.gpuIds.reserve(static_cast<size_t>(numGpus));
    for (int i = 0; i < numGpus; ++i) {
      hardwareOptions.gpuIds.push_back(i);
    }
  }

  ankerl::nanobench::Bench().epochIterations(1).epochs(1).run(benchName, [&]() {
    nvMolKit::embedMolecules(molCopyPtrs, params, numConfs, maxIterations, false, &fails, hardwareOptions, backend);
  });

  for (const auto& fail : fails) {
    int& sum = failCounts.emplace_back(0);
    for (const auto& f : fail) {
      sum += f;
    }
  }

  if (doEnergyCheck) {
    nvMolKit::checkForCompletedConformers(molCopyPtrs, numConfs, 1000, std::nullopt, true, true);
    energies = getMMFFEnergies(molCopyPtrs, numConfs);
  }
}

// Benchmark RDKit
void benchRDKit(const std::vector<RDKit::ROMol*>& mols,
                int                               numConfs,
                std::vector<double>&              energies,
                const bool                        doEnergyCheck,
                std::vector<int>&                 failCounts,
                const int                         numThreads,
                const int                         maxIterations) {
  std::string benchName = "RDKit EmbedMultipleConfs, num_mols=" + std::to_string(mols.size()) +
                          ", num_confs=" + std::to_string(numConfs) + ", numThreads=" + std::to_string(numThreads) +
                          ", maxIterations=" + std::to_string(maxIterations);

  // Create copies for RDKit benchmark
  std::vector<std::unique_ptr<RDKit::RWMol>> molCopies;
  std::vector<RDKit::ROMol*>                 molCopyPtrs;
  for (const auto& mol : mols) {
    molCopies.push_back(std::make_unique<RDKit::RWMol>(*mol));
    molCopyPtrs.push_back(molCopies.back().get());
  }

  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.maxIterations                        = maxIterations;
  params.useRandomCoords                      = true;
  params.basinThresh                          = 1e8;
  params.trackFailures                        = true;
  params.numThreads                           = numThreads;
  failCounts.resize(12, 0);
  ankerl::nanobench::Bench().epochIterations(1).epochs(1).run(benchName, [&]() {
    for (auto* mol : molCopyPtrs) {
      std::vector<int> res;
      RDKit::DGeomHelpers::EmbedMultipleConfs(*mol, res, numConfs, params);
      for (int i = 0; i < 12; i++) {
        failCounts[i] += params.failures[i];
      }
      params.failures.clear();
    }
  });

  if (doEnergyCheck) {
    nvMolKit::checkForCompletedConformers(molCopyPtrs, numConfs, 1000, std::nullopt, true, true);
    energies = getMMFFEnergies(molCopyPtrs, numConfs);
  }
}

void printEnergyDiffs(const std::vector<double>& rdkitEnergies,
                      const std::vector<double>& nvmolkitEnergies,
                      const int                  numConfsPerMol) {
  const int numMols = static_cast<int>(rdkitEnergies.size() / numConfsPerMol);
  if (rdkitEnergies.size() != nvmolkitEnergies.size()) {
    std::cerr << "Error: Energy vectors have different sizes: "
              << "RDKit: " << rdkitEnergies.size() << ", nvMolKit: " << nvmolkitEnergies.size() << std::endl;
    return;
  }
  int    denom    = 0;
  double sum      = 0.0;
  int    medDenom = numMols;
  double medSum   = 0.0;
  double magSum   = 0.0;
  for (int i = 0; i < numMols; ++i) {
    std::vector<double> rdkitVals;
    std::vector<double> nvmolkitVals;
    for (int j = 0; j < numConfsPerMol; ++j) {
      const int idx = i * numConfsPerMol + j;
      if (rdkitEnergies[idx] == 0.0 || nvmolkitEnergies[idx] == 0.0) {
        continue;
      }
      if (rdkitEnergies[idx] != 0.0) {
        rdkitVals.push_back(rdkitEnergies[idx]);
      }
      if (nvmolkitEnergies[idx] != 0.0) {
        nvmolkitVals.push_back(nvmolkitEnergies[idx]);
      }

      sum += rdkitEnergies[idx] - nvmolkitEnergies[idx];
      magSum += (rdkitEnergies[idx] - nvmolkitEnergies[idx]) / std::abs(rdkitEnergies[idx]);
      denom++;
    }
    std::sort(rdkitVals.begin(), rdkitVals.end());
    std::sort(nvmolkitVals.begin(), nvmolkitVals.end());
    if (rdkitVals.size() > 0 && nvmolkitVals.size() > 0) {
      double medRDKit    = rdkitVals[rdkitVals.size() / 2];
      double medNvMolKit = nvmolkitVals[nvmolkitVals.size() / 2];
      medSum += medRDKit - medNvMolKit;
    } else {
      medDenom--;
    }
  }
  std::cout << "Average energy difference (RDKit - nvMolKit): " << (denom > 0 ? sum / denom : 0.0) << " kcal/mol over "
            << denom << " conformers." << std::endl;
  std::cout << "Average relative energy difference (RDKit - nvMolKit) / RDKit: "
            << (denom > 0 ? magSum / denom : 0.0) * 100 << "% over " << denom << " conformers." << std::endl;
  std::cout << "Average median energy difference [median within mol, average across] (RDKit - nvMolKit): "
            << (medDenom > 0 ? medSum / medDenom : 0.0) << " kcal/mol over " << medDenom << " molecules." << std::endl;
}

// Run benchmarks with different combinations of molecule counts and conformer counts
void runBench(const std::string&          filePath,
              const int                   numMols,
              const int                   confsPerMol,
              const bool                  doRdkit,
              const bool                  doWarmup,
              const bool                  doEnergyCheck,
              const int                   numThreads,
              const int                   batchSize,
              const int                   batchesPerGpu,
              const int                   maxIterations,
              const int                   numGpus,
              const nvMolKit::BfgsBackend backend) {
  if (doWarmup) {
    printf("Warming up...\n");
    // Warm up with a small test
    auto                       warmupMols = readMolecules(filePath, 1);
    std::vector<RDKit::ROMol*> warmupMolsPtrs;
    for (const auto& mol : warmupMols) {
      warmupMolsPtrs.push_back(mol.get());
    }
    std::vector<double> dummy;
    std::vector<int>    failCountsDummy;
    benchNvMolKit(warmupMolsPtrs,
                  1,
                  dummy,
                  false,
                  failCountsDummy,
                  numThreads,
                  batchSize,
                  batchesPerGpu,
                  maxIterations,
                  numGpus,
                  backend);
    if (doRdkit) {
      benchRDKit(warmupMolsPtrs, 1, dummy, false, failCountsDummy, 1, maxIterations);
    }
    printf("Warmed up\n");
  } else {
    printf("Skipping  warmup\n");
  }

  // Run benchmark
  auto                       mols = readMolecules(filePath, numMols);
  std::vector<RDKit::ROMol*> molsPtrs;
  for (const auto& mol : mols) {
    molsPtrs.push_back(mol.get());
  }

  printf("\nBenchmarking with %u molecules and %u conformers per molecule\n", numMols, confsPerMol);
  std::vector<double> nvmolkitEnergies;
  std::vector<int>    failCountsNvMolKit;
  benchNvMolKit(molsPtrs,
                confsPerMol,
                nvmolkitEnergies,
                doEnergyCheck,
                failCountsNvMolKit,
                numThreads,
                batchSize,
                batchesPerGpu,
                maxIterations,
                numGpus,
                backend);
  if (doRdkit) {
    std::vector<double> rdkitEnergies;
    std::vector<int>    failCountsRDKit;
    benchRDKit(molsPtrs, confsPerMol, rdkitEnergies, doEnergyCheck, failCountsRDKit, numThreads, maxIterations);
    printEnergyDiffs(rdkitEnergies, nvmolkitEnergies, confsPerMol);
    std::cout << "\nFailure counts, RDKit , nvMolKit\n";

    for (int i = 0; i < 11; i++) {
      std::cout << "Stage " << i << ": " << failCountsRDKit[i] << " , " << failCountsNvMolKit[i] << std::endl;
    }
  } else {
    std::cout << "\nFailure counts,nvMolKit\n";

    for (int i = 0; i < 11; i++) {
      std::cout << "Stage " << i << ": " << failCountsNvMolKit[i] << std::endl;
    }
  }
}

bool parseBoolArg(const std::string& arg) {
  std::string s = arg;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  return (s == "1" || s == "true" || s == "yes" || s == "on");
}

nvMolKit::BfgsBackend parseBackendArg(const std::string& arg) {
  std::string s = arg;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  if (s == "batched" || s == "0") {
    return nvMolKit::BfgsBackend::BATCHED;
  } else if (s == "per_molecule" || s == "per-molecule" || s == "permolecule" || s == "1") {
    return nvMolKit::BfgsBackend::PER_MOLECULE;
  } else if (s == "hybrid" || s == "2") {
    return nvMolKit::BfgsBackend::HYBRID;
  } else {
    throw std::runtime_error("Invalid backend value. Use 'batched', 'per_molecule', or 'hybrid'");
  }
}

void printHelp(const char* progName) {
  std::cout << "Usage: " << progName << " [options]\n\n";
  std::cout << "Options:\n";
  std::cout
    << "  -f, --file_path <path>              Path to input file (.sdf, .smi, or .smiles) [default: benchmarks/data/MPCONF196.sdf]\n";
  std::cout << "  -n, --num_mols <int>                Number of molecules to process [default: 20]\n";
  std::cout << "  -c, --confs_per_mol <int>           Number of conformers per molecule [default: 20]\n";
  std::cout << "  -r, --do_rdkit <bool>               Run RDKit benchmark comparison [default: true]\n";
  std::cout << "  -w, --do_warmup <bool>              Run warmup before benchmarking [default: true]\n";
  std::cout << "  -e, --do_energy_check <bool>        Check energies after embedding [default: false]\n";
  std::cout << "  -t, --num_threads <int>             Number of threads to use [default: 10]\n";
  std::cout << "  -b, --batch_size <int>              Batch size for processing [default: 100]\n";
  std::cout << "  -B, --num_concurrent_batches <int>  Number of concurrent batches [default: 10]\n";
  std::cout << "  -g, --num_gpus <int>                Number of GPUs to use (IDs 0..n-1). If omitted, use all GPUs.\n";
  std::cout
    << "  -k, --backend <string>              BFGS backend: batched, per_molecule, or hybrid [default: hybrid]\n";
  std::cout << "  -h, --help                          Show this help message\n\n";
  std::cout << "Boolean values can be: true/false, 1/0, yes/no, on/off (case insensitive)\n";
  std::cout << "\nExamples:\n";
  std::cout << "  " << progName << " --file_path data.sdf --num_mols 50 --confs_per_mol 10 --do_rdkit true\n";
  std::cout << "  " << progName << " -f data.sdf -n 50 -c 10 -r false -t 8 -b 50 -B 4 -g 2 -k batched\n";
}

int main(int argc, char* argv[]) {
  // Set defaults
  std::string           filePath      = "benchmarks/data/MPCONF196.sdf";
  int                   numMols       = 20;
  int                   confsPerMol   = 20;
  bool                  doRdkit       = true;
  bool                  doWarmup      = true;
  bool                  doEnergyCheck = false;
  int                   numThreads    = 10;
  int                   batchSize     = 100;
  int                   batchesPerGpu = 10;
  int                   maxIterations = 10;
  int                   numGpus       = -1;  // If <0, use all GPUs by default
  nvMolKit::BfgsBackend backend       = nvMolKit::BfgsBackend::HYBRID;

  // Define long options for getopt_long
  static struct option long_options[] = {
    {             "file_path", required_argument, 0, 'f'},
    {              "num_mols", required_argument, 0, 'n'},
    {         "confs_per_mol", required_argument, 0, 'c'},
    {              "do_rdkit", required_argument, 0, 'r'},
    {             "do_warmup", required_argument, 0, 'w'},
    {       "do_energy_check", required_argument, 0, 'e'},
    {           "num_threads", required_argument, 0, 't'},
    {            "batch_size", required_argument, 0, 'b'},
    {"num_concurrent_batches", required_argument, 0, 'B'},
    {        "max_iterations", required_argument, 0, 'i'},
    {              "num_gpus", required_argument, 0, 'g'},
    {               "backend", required_argument, 0, 'k'},
    {                  "help",       no_argument, 0, 'h'},
    {                       0,                 0, 0,   0}
  };

  int option_index = 0;
  int c;

  // Parse command line arguments using getopt_long
  while ((c = getopt_long(argc, argv, "f:n:c:r:w:e:t:b:B:i:g:k:h", long_options, &option_index)) != -1) {
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
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for num_mols: " << optarg << "\n";
          return 1;
        }
        break;
      case 'c':
        try {
          confsPerMol = std::stoi(optarg);
          if (confsPerMol <= 0) {
            std::cerr << "Error: confs_per_mol must be positive\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for confs_per_mol: " << optarg << "\n";
          return 1;
        }
        break;
      case 'r':
        doRdkit = parseBoolArg(optarg);
        break;
      case 'w':
        doWarmup = parseBoolArg(optarg);
        break;
      case 'e':
        doEnergyCheck = parseBoolArg(optarg);
        break;
      case 't':
        try {
          numThreads = std::stoi(optarg);
          if (numThreads <= 0) {
            std::cerr << "Error: num_threads must be positive\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for num_threads: " << optarg << "\n";
          return 1;
        }
        break;
      case 'b':
        try {
          batchSize = std::stoi(optarg);
          if (batchSize <= 0) {
            std::cerr << "Error: batch_size must be positive\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for batch_size: " << optarg << "\n";
          return 1;
        }
        break;
      case 'B':
        try {
          batchesPerGpu = std::stoi(optarg);
          if (batchesPerGpu <= 0) {
            std::cerr << "Error: num_concurrent_batches must be positive\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for num_concurrent_batches: " << optarg << "\n";
          return 1;
        }
        break;
      case 'i':
        try {
          maxIterations = std::stoi(optarg);
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for max_iterations: " << optarg << "\n";
          return 1;
        }
        break;
      case 'g':
        try {
          numGpus = std::stoi(optarg);
          if (numGpus <= 0) {
            std::cerr << "Error: num_gpus must be positive\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for num_gpus: " << optarg << "\n";
          return 1;
        }
        break;
      case 'k':
        try {
          backend = parseBackendArg(optarg);
        } catch (const std::exception& e) {
          std::cerr << "Error: " << e.what() << "\n";
          return 1;
        }
        break;
      case 'h':
        printHelp(argv[0]);
        return 0;
      case '?':
        // getopt_long already printed an error message
        std::cerr << "\nUse --help for usage information.\n";
        return 1;
      default:
        std::cerr << "Unknown option\n";
        return 1;
    }
  }

  // Handle non-option arguments (if any)
  if (optind < argc) {
    std::cerr << "Error: Unexpected non-option arguments: ";
    while (optind < argc) {
      std::cerr << argv[optind++] << " ";
    }
    std::cerr << "\nUse --help for usage information.\n";
    return 1;
  }

  // Check if file exists
  if (!std::filesystem::exists(filePath)) {
    std::cerr << "Error: File does not exist: " << filePath << std::endl;
    return 1;
  }

  // Print configuration
  std::string backendName = (backend == nvMolKit::BfgsBackend::BATCHED) ? "batched" :
                            (backend == nvMolKit::BfgsBackend::HYBRID)  ? "hybrid" :
                                                                          "per_molecule";
  std::cout << "Configuration:\n";
  std::cout << "  File path: " << filePath << "\n";
  std::cout << "  Number of molecules: " << numMols << "\n";
  std::cout << "  Conformers per molecule: " << confsPerMol << "\n";
  std::cout << "  Run RDKit comparison: " << (doRdkit ? "yes" : "no") << "\n";
  std::cout << "  Run warmup: " << (doWarmup ? "yes" : "no") << "\n";
  std::cout << "  Check energies: " << (doEnergyCheck ? "yes" : "no") << "\n";
  std::cout << "  Number of threads: " << numThreads << "\n";
  std::cout << "  Batch size: " << batchSize << "\n";
  std::cout << "  Number of concurrent batches: " << batchesPerGpu << "\n";
  std::cout << "  Number of GPUs: " << (numGpus > 0 ? std::to_string(numGpus) : std::string("all")) << "\n";
  std::cout << "  BFGS backend: " << backendName << "\n\n";

  // Run the benchmark
  runBench(filePath,
           numMols,
           confsPerMol,
           doRdkit,
           doWarmup,
           doEnergyCheck,
           numThreads,
           batchSize,
           batchesPerGpu,
           maxIterations,
           numGpus,
           backend);
  return 0;
}
