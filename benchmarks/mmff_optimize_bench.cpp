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

#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <nanobench.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <random>

#include "src/minimizer/mmff_minimize.h"
#include "tests/test_utils.h"

void perturbConformer(RDKit::Conformer& conf, const float delta = 0.1, const int seed = 0) {
  std::mt19937                          gen(seed);  // Mersenne Twister engine
  std::uniform_real_distribution<float> dist(-delta, delta);
  for (unsigned int i = 0; i < conf.getNumAtoms(); ++i) {
    RDGeom::Point3D pos = conf.getAtomPos(i);
    pos.x += delta * dist(gen);
    pos.y += delta * dist(gen);
    pos.z += delta * dist(gen);
    conf.setAtomPos(i, pos);
  }
}

void genNConformers(RDKit::ROMol& mol, const int numConfs) {
  const int confIdExisting = mol.getConformer().getId();
  for (int i = 1; i < numConfs; i++) {
    auto conf = new RDKit::Conformer(mol.getConformer());
    perturbConformer(*conf, 0.5, i + 5);
    mol.addConformer(conf, true);
  }
  perturbConformer(mol.getConformer(confIdExisting), 0.5, 0);
  if (static_cast<int>(mol.getNumConformers()) != numConfs) {
    throw std::runtime_error("Failed to generate the expected number of conformers, got " +
                             std::to_string(mol.getNumConformers()) + " expected " + std::to_string(numConfs));
  }
}

std::vector<double> benchRDKit(const std::vector<RDKit::ROMol*>& mols,
                               const int                         size,
                               const int                         numConfs,
                               const int                         maxIters) {
  std::vector<RDKit::ROMol> molsToBench;
  for (const auto& mol : mols) {
    if (static_cast<int>(mol->getNumAtoms()) == size) {
      molsToBench.push_back(*mol);
      break;
    };
  }

  if (molsToBench.empty()) {
    throw std::runtime_error("No molecules found with the specified size");
  }
  genNConformers(molsToBench[0], numConfs);

  std::vector<std::pair<int, double>> res(numConfs);
  std::string benchName = "RDKit, mol_size=" + std::to_string(size) + ", num_confs=" + std::to_string(numConfs);

  ankerl::nanobench::Bench().epochIterations(1).epochs(1).run(benchName, [&]() {
    RDKit::MMFF::MMFFOptimizeMoleculeConfs(molsToBench[0], res, 1, maxIters, "MMFF94", 100.0);
  });

  std::vector<double> energies;
  for (const auto& r : res) {
    energies.push_back(r.second);
  }
  return energies;
}

std::vector<double> benchNvMolKit(const std::vector<RDKit::ROMol*>& mols,
                                  const int                         size,
                                  const int                         numConfs,
                                  const int                         maxIters) {
  std::vector<RDKit::ROMol> molsToBench;
  for (const auto& mol : mols) {
    if (static_cast<int>(mol->getNumAtoms()) == size) {
      molsToBench.push_back(*mol);
      break;
    };
  }

  if (molsToBench.empty()) {
    throw std::runtime_error("No molecules found with the specified size");
  }

  std::string benchName = "nvMolKit, mol_size=" + std::to_string(size) + ", num_confs=" + std::to_string(numConfs);

  genNConformers(molsToBench[0], numConfs);
  std::vector<double>        energies;
  std::vector<RDKit::ROMol*> molsToBenchPtrs = {&molsToBench[0]};
  ankerl::nanobench::Bench().epochIterations(1).epochs(1).run(benchName, [&]() {
    energies = nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molsToBenchPtrs, maxIters)[0];
  });
  return energies;
}

enum class datasets {
  mmff_validation,
  chembl
};

void runMMFFBench(int s, int n) {
  const std::string                          fileName = getTestDataFolderPath() + "/MMFF94_hypervalent.sdf";
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(fileName, mols);
  std::vector<RDKit::ROMol*> molsPtrs;
  for (const auto& mol : mols) {
    molsPtrs.push_back(mol.get());
  }
  constexpr int maxIters = 1000;
  printf("Warming up\n");
  benchNvMolKit(molsPtrs, 10, 100, maxIters);
  benchRDKit(molsPtrs, 10, 100, maxIters);
  printf("Warmed up\n");
  if (s != -1 && n != -1) {
    std::vector<double>                    nvmolkitRes = benchNvMolKit(molsPtrs, s, n, maxIters);
    std::vector<double>                    rdkitRes    = benchRDKit(molsPtrs, s, n, maxIters);
    // Compare the two, count number of differences greater than 1e-3
    int                                    numDiffs    = 0;
    std::vector<std::pair<double, double>> values;
    for (size_t i = 0; i < nvmolkitRes.size(); i++) {
      if (std::abs(nvmolkitRes[i] - rdkitRes[i]) > 1e-2) {
        numDiffs++;
        values.push_back({nvmolkitRes[i], rdkitRes[i]});
      }
    }
    if (numDiffs > 0) {
      std::cout << "Differences found for size " << s << " and numConfs " << n << ": " << numDiffs << " differences"
                << std::endl;
      std::cout << "Deltas: ";
      for (const auto& delta : values) {
        std::cout << "Conf " << delta.first << ": " << delta.second << ", " << std::abs(delta.first - delta.second)
                  << "\n";
      }
    }
    return;
  }

  const std::vector<int> molSizes = {10, 20, 30, 40, 50};
  const std::vector<int> numConfs = {220, 240, 260, 280, 300};
  for (const auto size : molSizes) {
    for (const auto numConf : numConfs) {
      std::vector<double> nvmolkitRes = benchNvMolKit(molsPtrs, size, numConf, maxIters);
      std::vector<double> rdkitRes    = benchRDKit(molsPtrs, size, numConf, maxIters);

      // Compare the two, count number of differences greater than 1e-3
      int numDiffs = 0;
      for (size_t i = 0; i < nvmolkitRes.size(); i++) {
        if (std::abs(nvmolkitRes[i] - rdkitRes[i]) > 1e-2) {
          numDiffs++;
        }
      }
      if (numDiffs > 0) {
        std::cout << "Differences found for size " << size << " and numConfs " << numConf << ": " << numDiffs
                  << " differences" << std::endl;
      }
    }
  }
}

void benchChembl(int size, int numConfs, const std::string& path, const int maxIters) {
  // Open a csv file, parse the first line, first entry
  std::ifstream file(path + "/molecules_" + std::to_string(size) + "_atoms.csv");
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path + "/chembl_" + std::to_string(size) + ".csv");
  }
  // skip first line
  std::string line;
  std::getline(file, line);
  std::getline(file, line);
  std::stringstream ss(line);
  std::string       smi;
  std::getline(ss, smi, ',');
  file.close();

  // Parse the smiles string
  RDKit::RWMol* mol = RDKit::SmilesToMol(smi);
  if (!mol) {
    throw std::runtime_error("Failed to parse SMILES: " + smi);
  }
  auto molHolder = std::unique_ptr<RDKit::RWMol>(mol);

  // Generate initial conformer using RDKit etkdg, and relax before generic perturbation.
  int attempt = 0;
  int confRes = -1;
  while (attempt < 10 && confRes == -1) {
    confRes = RDKit::DGeomHelpers::EmbedMolecule(*mol);
    attempt++;
  }
  if (confRes == -1) {
    throw std::runtime_error("Failed to generate initial conformer for SMILES: " + smi);
  }

  RDKit::MMFF::MMFFOptimizeMolecule(*mol);

  RDKit::ROMol nvmolkitMol = *mol;
  RDKit::ROMol rdkitMol    = *mol;

  genNConformers(nvmolkitMol, numConfs);
  genNConformers(rdkitMol, numConfs);

  std::string benchName      = "nvMolKit, mol_size=" + std::to_string(size) + ", num_confs=" + std::to_string(numConfs);
  std::string benchNamerdkit = "RDKit, mol_size=" + std::to_string(size) + ", num_confs=" + std::to_string(numConfs);

  std::vector<RDKit::ROMol*> molsToBenchPtrs = {&nvmolkitMol};
  ankerl::nanobench::Bench().epochIterations(1).epochs(1).run(benchName, [&]() {
    nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molsToBenchPtrs, maxIters);
  });

  std::vector<std::pair<int, double>> res;
  ankerl::nanobench::Bench().epochIterations(1).epochs(1).run(benchNamerdkit, [&]() {
    RDKit::MMFF::MMFFOptimizeMoleculeConfs(rdkitMol, res, 1, maxIters);
  });

  // Validate:
}

void runChemblBench(int s, int n) {
  const std::string       folder   = "/home/kevin/omg/datasets/chembl/discrete_sizes";
  constexpr int           maxIters = 200;
  static std::vector<int> molSizes = {60, 70, 80, 90, 100};
  static std::vector<int> numConfs = {20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300};
  printf("Warming up\n");

  benchChembl(60, 20, folder, maxIters);
  printf("Warmed up\n");
  if (s != -1 && n != -1) {
    benchChembl(s, n, folder, maxIters);
    return;
  }

  for (const auto size : molSizes) {
    for (const auto numConf : numConfs) {
      benchChembl(size, numConf, folder, maxIters);
    }
  }
}

int main(int argc, char* argv[]) {
  std::string dataset = "mmff_validation";
  if (argc >= 2) {
    dataset = argv[1];
  }
  auto dataset_selected = dataset == "chembl" ? datasets::chembl : datasets::mmff_validation;

  int size     = -1;
  int numConfs = -1;
  if (argc == 4) {
    printf("Using size %s and numConfs %s\n", argv[2], argv[3]);
    size     = std::stoi(argv[2]);
    numConfs = std::stoi(argv[3]);
  }

  if (dataset_selected == datasets::mmff_validation) {
    std::cout << "Running MMFF validation benchmarks" << std::endl;
    runMMFFBench(size, numConfs);
  } else {
    std::cout << "Running Chembl benchmarks" << std::endl;
    runChemblBench(size, numConfs);
  }

  return 0;
}
