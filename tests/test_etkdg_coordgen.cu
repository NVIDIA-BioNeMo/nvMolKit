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

#include <gmock/gmock.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <gtest/gtest.h>

#include "src/etkdg_impl.h"
#include "src/etkdg_stage_coordgen.h"
#include "src/forcefields/coord_gen.h"
#include "src/utils/device_vector.h"
#include "tests/test_utils.h"

using ::nvMolKit::detail::ETKDGContext;
using ::nvMolKit::detail::ETKDGCoordGenStage;
using ::nvMolKit::detail::ETKDGDriver;
using ::nvMolKit::detail::ETKDGStage;

class ProgrammableStep : public ETKDGStage {
 public:
  void execute(ETKDGContext& ctx) override {
    ASSERT_LT(iteration, failedPerIteration.size());
    ctx.failedThisStage.copyFromHost(failedPerIteration[iteration]);
    iteration++;
  }
  std::string                       name() const override { return "Programmable Test Stage"; }
  int                               iteration = 0;
  std::vector<std::vector<uint8_t>> failedPerIteration;
};

void createContext(ETKDGContext& context, const std::vector<const RDKit::ROMol*>& mols) {
  context.nTotalSystems = mols.size();

  // TODO: 4D support.
  constexpr unsigned int dim = 3;

  std::vector<int> atomStarts = {0};
  for (const auto& mol : mols) {
    atomStarts.push_back(atomStarts.back() + mol->getNumAtoms());
  }
  ASSERT_GT(atomStarts.back(), 0);
  context.systemDevice.positions.resize(atomStarts.back() * dim);
  context.systemDevice.positions.zero();
  context.systemDevice.atomStarts.setFromVector(atomStarts);
}

TEST(EtkdgCoordGenTest, TestGenerateInitialCoords) {
  // Set up test parameters
  auto                                       defaultParams      = RDKit::DGeomHelpers::ETKDGv3;
  // Set up test data
  std::string                                testDataFolderPath = getTestDataFolderPath();
  std::vector<std::unique_ptr<RDKit::ROMol>> molsPtrs;
  getMols(testDataFolderPath + "/MMFF94_dative.sdf", molsPtrs, /*count=*/10);
  std::vector<const RDKit::ROMol*> mols;
  for (auto& molPtr : molsPtrs) {
    molPtr->clearConformers();
    mols.push_back(molPtr.get());
  }
  auto params  = RDKit::DGeomHelpers::ETKDGv3;
  auto context = std::make_unique<nvMolKit::detail::ETKDGContext>();
  createContext(*context, mols);

  std::vector<std::unique_ptr<ETKDGStage>> stages;
  auto                                     prevStage = std::make_unique<ProgrammableStep>();
  prevStage->failedPerIteration                      = {
    {0, 0, 1, 1, 0, 0, 0, 0, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1}
  };

  auto stage = std::make_unique<ETKDGCoordGenStage>(params, mols);
  stages.push_back(std::move(prevStage));
  stages.push_back(std::move(stage));
  ETKDGDriver driver(std::move(context), std::move(stages));

  driver.run(2);

  ASSERT_EQ(driver.iterationsComplete(), 2);
  EXPECT_GT(driver.numConfsFinished(), 0);  // Statistically likely at least one conformer finished.
  EXPECT_LE(driver.numConfsFinished(), 8);  // At least two failed first stage both iterations.

  nvMolKit::PinnedHostVector<int16_t> failuresScratch;
  auto                                failureCounts = driver.getFailures(failuresScratch);
  EXPECT_EQ(failureCounts.size(), 2);

  auto stage0Failures = failureCounts[0];
  EXPECT_THAT(stage0Failures, ::testing::ElementsAre(0, 0, 1, 1, 0, 0, 0, 0, 2, 2));

  auto stage1Failures = failureCounts[1];
  // Check that no new failures on already failed run.
  EXPECT_LE(stage1Failures[2], 1);
  EXPECT_LE(stage1Failures[3], 1);
  EXPECT_EQ(stage1Failures[8], 0);
  EXPECT_EQ(stage1Failures[9], 0);

  // Check that positions are nonzero for cases that finished on the last iteration.
  std::vector<double> gotPos;
  const auto&         ctx = driver.context();
  gotPos.resize(ctx.systemDevice.positions.size());
  ctx.systemDevice.positions.copyToHost(gotPos);

  std::vector<int> atomStarts;
  atomStarts.resize(ctx.systemDevice.atomStarts.size());
  ctx.systemDevice.atomStarts.copyToHost(atomStarts);
  cudaDeviceSynchronize();
  auto finishedOnIterations = driver.getFinishedOnIterations();
  for (size_t i = 0; i < finishedOnIterations.size(); i++) {
    if (finishedOnIterations[i] == 1) {
      const int atomStart = atomStarts[i];
      const int atomEnd   = atomStarts[i + 1];
      for (int j = atomStart; j < atomEnd; j++) {
        for (int k = 0; k < 3; k++) {
          ASSERT_NE(gotPos[j * 3 + k], 0.0);
        }
      }
    }
  }
}
