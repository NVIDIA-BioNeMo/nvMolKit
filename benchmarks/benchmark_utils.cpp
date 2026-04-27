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

#include "benchmarks/benchmark_utils.h"

#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <omp.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include "tests/test_utils.h"

namespace BenchUtils {

std::string getFileExtensionLower(const std::string& filePath) {
  std::string ext = std::filesystem::path(filePath).extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return ext;
}

static std::vector<std::unique_ptr<RDKit::ROMol>> readRawMolecules(const std::string& filePath,
                                                                   unsigned int       count,
                                                                   unsigned int       maxAtoms) {
  std::vector<std::unique_ptr<RDKit::ROMol>> tempMols;
  const std::string                          extension = getFileExtensionLower(filePath);

  if (extension == ".sdf") {
    getMols(filePath, tempMols, static_cast<int>(count));
  } else if (extension == ".smi" || extension == ".smiles" || extension == ".cxsmiles") {
    std::ifstream file(filePath);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open SMILES file: " + filePath);
    }
    std::string                                smilesLine;
    std::vector<std::unique_ptr<RDKit::ROMol>> allMols;
    while (std::getline(file, smilesLine) && allMols.size() < count) {
      if (smilesLine.empty() || smilesLine[0] == '#') {
        continue;
      }
      const std::string smiles = smilesLine.substr(0, smilesLine.find_first_of(" \t"));
      try {
        auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
        if (mol && mol->getNumAtoms() <= maxAtoms) {
          allMols.push_back(std::move(mol));
        } else if (mol) {
          std::cerr << "Warning: Molecule with SMILES " << smiles << " has more than " << maxAtoms
                    << " atoms and will be skipped." << std::endl;
        }
      } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to parse SMILES: " << smiles << " - " << e.what() << std::endl;
      }
    }
    if (allMols.empty()) {
      throw std::runtime_error("No valid molecules found in SMILES file: " + filePath);
    }
    if (allMols.size() < count) {
      const size_t originalSize = allMols.size();
      for (size_t i = 0; i < count - originalSize; ++i) {
        allMols.push_back(std::make_unique<RDKit::ROMol>(*allMols[i % originalSize]));
      }
    }
    tempMols = std::move(allMols);
  } else {
    throw std::runtime_error(
      "Unsupported file format. Only .sdf, .smi, .smiles, and .cxsmiles files are supported. Got " + extension);
  }
  return tempMols;
}

std::vector<std::unique_ptr<RDKit::RWMol>> readMoleculesForEmbedding(const std::string& filePath,
                                                                     unsigned int       count,
                                                                     unsigned int       maxAtoms) {
  auto                                       raw = readRawMolecules(filePath, count, maxAtoms);
  std::vector<std::unique_ptr<RDKit::RWMol>> mols;
  mols.reserve(raw.size());
  for (auto& r : raw) {
    try {
      std::unique_ptr<RDKit::ROMol> molH(RDKit::MolOps::addHs(*r));
      auto                          rw = std::make_unique<RDKit::RWMol>(*molH);
      rw->clearConformers();
      RDKit::MolOps::sanitizeMol(*rw);
      mols.push_back(std::move(rw));
    } catch (const std::exception& e) {
      std::cerr << "Warning: sanitize failed and molecule will be skipped: " << e.what() << std::endl;
    }
  }
  if (mols.empty()) {
    throw std::runtime_error("No valid molecules available after sanitization for embedding from: " + filePath);
  }
  if (mols.size() < count) {
    const size_t originalSize = mols.size();
    for (size_t i = 0; i < count - originalSize; ++i) {
      mols.push_back(std::make_unique<RDKit::RWMol>(*mols[i % originalSize]));
    }
  }
  return mols;
}

std::vector<std::unique_ptr<RDKit::RWMol>> readMoleculesKeepConfs(const std::string& filePath,
                                                                  unsigned int       count,
                                                                  unsigned int       maxAtoms) {
  auto                                       raw = readRawMolecules(filePath, count, maxAtoms);
  std::vector<std::unique_ptr<RDKit::RWMol>> mols;
  mols.reserve(raw.size());
  for (auto& r : raw) {
    try {
      // Preserve conformers for SDF inputs; for SMILES there are none.
      std::unique_ptr<RDKit::ROMol> molH(RDKit::MolOps::addHs(*r));
      auto                          rw = std::make_unique<RDKit::RWMol>(*molH);
      // Avoid re-sanitizing SDF-derived molecules; supplier likely sanitized already.
      mols.push_back(std::move(rw));
    } catch (const std::exception& e) {
      std::cerr << "Warning: sanitize failed and molecule will be skipped: " << e.what() << std::endl;
    }
  }
  if (mols.empty()) {
    throw std::runtime_error("No valid molecules available after sanitization from: " + filePath);
  }
  if (mols.size() < count) {
    const size_t originalSize = mols.size();
    for (size_t i = 0; i < count - originalSize; ++i) {
      mols.push_back(std::make_unique<RDKit::RWMol>(*mols[i % originalSize]));
    }
  }
  return mols;
}

static void perturbConformer(RDKit::Conformer& conf, float delta, int seed) {
  std::mt19937                          gen(seed);
  std::uniform_real_distribution<float> dist(-delta, delta);
  for (unsigned int i = 0; i < conf.getNumAtoms(); ++i) {
    RDGeom::Point3D pos = conf.getAtomPos(i);
    pos.x += delta * dist(gen);
    pos.y += delta * dist(gen);
    pos.z += delta * dist(gen);
    conf.setAtomPos(i, pos);
  }
}

void ensureNumConformersByCopying(RDKit::ROMol& mol, int numConfs) {
  const int existing = static_cast<int>(mol.getNumConformers());
  if (existing <= 0) {
    return;
  }
  for (int i = existing; i < numConfs; ++i) {
    auto conf = new RDKit::Conformer(mol.getConformer(0));
    mol.addConformer(conf, true);
  }
}

void perturbAllConformers(std::vector<RDKit::ROMol*>& mols, float delta, int seedBase) {
  int molIdx = 0;
  for (auto* m : mols) {
    for (unsigned int c = 0; c < m->getNumConformers(); ++c) {
      perturbConformer(m->getConformer(c), delta, seedBase + molIdx * 97 + static_cast<int>(c));
    }
    molIdx++;
  }
}

void embedConformersRDKit(std::vector<RDKit::ROMol*>& mols, int numConfs, int maxIterations, bool clearExisting) {
  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.maxIterations                        = maxIterations;
  params.useRandomCoords                      = true;
  params.trackFailures                        = false;

  for (auto* mol : mols) {
    if (clearExisting) {
      auto rw = dynamic_cast<RDKit::RWMol*>(mol);
      if (rw) {
        rw->clearConformers();
      }
    }
    std::vector<int> res;
    RDKit::DGeomHelpers::EmbedMultipleConfs(*mol, res, numConfs, params);
  }
}

void embedOneConfThenDuplicate(std::vector<RDKit::ROMol*>& mols,
                               int                         numConfs,
                               int                         rdkitNumThreads,
                               int /*maxIterationsUnused*/) {
  RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
  params.maxIterations                        = 10;  // Force to 10 per request
  params.useRandomCoords                      = true;
  params.trackFailures                        = false;
  params.numThreads                           = rdkitNumThreads;

  std::vector<char> success(mols.size(), 0);

#pragma omp parallel for schedule(dynamic)
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(mols.size()); ++i) {
    RDKit::ROMol* mol = mols[static_cast<size_t>(i)];
    auto          rw  = dynamic_cast<RDKit::RWMol*>(mol);
    if (rw) {
      rw->clearConformers();
    }
    try {
      std::vector<int> res;
      RDKit::DGeomHelpers::EmbedMultipleConfs(*mol, res, 1, params);
      if (!res.empty()) {
        success[static_cast<size_t>(i)] = 1;
      }
    } catch (...) {
      // Leave success as 0 and continue
    }
  }

  // Duplicate only successful molecules
  for (size_t i = 0; i < mols.size(); ++i) {
    if (success[i]) {
      ensureNumConformersByCopying(*mols[i], numConfs);
    }
  }
}

}  // namespace BenchUtils