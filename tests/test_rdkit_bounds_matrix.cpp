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
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include "rdkit_extensions/bounds_matrix.h"

namespace nvMolKit {
namespace test {

TEST(BoundsMatrixTest, Hexane) {
  static const std::string                smiles   = "C1CCCCC1";
  static std::vector<std::vector<double>> expected = {
    {      0,   1.524, 2.51279, 3.81072, 2.51279,   1.524},
    {  1.504,       0,   1.524, 2.51279, 3.81072, 2.51279},
    {2.43279,   1.504,       0,   1.524, 2.51279, 3.81072},
    {2.52477, 2.43279,   1.504,       0,   1.524, 2.51279},
    {2.43279, 2.52477, 2.43279,   1.504,       0,   1.524},
    {  1.504, 2.43279, 2.52477, 2.43279,   1.504,       0}
  };

  // Get ROMol
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  mols.emplace_back(RDKit::SmilesToMol(smiles));
  std::vector<const RDKit::ROMol*> molsView{mols[0].get()};

  // Match what EmbedMultipleConfs sets by default.
  RDKit::DGeomHelpers::EmbedParameters params(30,       // maxIterations
                                              1,        // numThreads
                                              -1,       // seed
                                              true,     // clearConfs
                                              false,    // useRandomCoords
                                              2.0,      // boxSizeMult
                                              true,     // randNegEig
                                              1,        // numZeroFail
                                              nullptr,  // coordMap
                                              1e-3,     // optimizerForceTol
                                              false,    // ignoreSmoothingFailures
                                              true,     // enforceChirality
                                              false,    // useExpTorsionAnglePrefs
                                              false,    // useBasicKnowledge
                                              false,    // verbose
                                              5.0,      // basinThresh
                                              -1.0,     // pruneRmsThresh
                                              false,    // onlyHeavyAtomsForRMS
                                              2,        // ETversion
                                              nullptr,  // extraParams
                                              true,     // useSmallRingTorsions
                                              true,     // useMacrocycleTorsions
                                              true,     // useMacrocycle14config
                                              0         // timeout
  );

  std::vector<ForceFields::CrystalFF::CrystalFFDetails> details(mols.size());
  RDKit::DGeomHelpers::initETKDG(mols[0].get(), params, details[0]);

  auto        bounds_matrices = nvMolKit::getBoundsMatrices(molsView, params, details);
  const auto& result          = bounds_matrices[0];
  for (size_t i = 0; i < expected.size(); ++i) {
    for (size_t j = 0; j < expected[i].size(); ++j) {
      EXPECT_NEAR(result->getVal(i, j), expected[i][j], 1e-5) << "i: " << i << " j: " << j;
    }
  }
}

}  // namespace test
}  // namespace nvMolKit
