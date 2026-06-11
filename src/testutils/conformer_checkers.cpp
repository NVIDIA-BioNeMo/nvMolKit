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

#include "src/testutils/conformer_checkers.h"

#include <GraphMol/ROMol.h>

#include <iostream>
namespace nvMolKit {

bool checkForCompletedConformers(const std::vector<const RDKit::ROMol*>& mols,
                                 const int                               numConfsExpected,
                                 const std::optional<int>                totalFailuresTolerance,
                                 const std::optional<int>                failsPerMoleculeTolerance,
                                 const bool                              acceptEitherMetricAsPass,
                                 const bool                              printResults) {
  if (!totalFailuresTolerance && !failsPerMoleculeTolerance) {
    throw std::invalid_argument(
      "At least one of totalFailuresTolerance or failsPerMoleculeTolerance must be provided.");
  }

  size_t totalFailures           = 0;
  bool   foundPerMoleculeFailure = false;
  for (size_t i = 0; i < mols.size(); ++i) {
    const auto& mol = mols[i];

    if (const int numConfs = static_cast<int>(mol->getNumConformers()); numConfs < numConfsExpected) {
      const size_t numConfsFailed = numConfsExpected - numConfs;
      totalFailures += numConfsFailed;
      if (failsPerMoleculeTolerance && numConfsFailed > *failsPerMoleculeTolerance) {
        foundPerMoleculeFailure = true;
        if (printResults) {
          std::cout << "Molecule " << i << " failed with " << (numConfsExpected - numConfs)
                    << " failures, which is below the per-molecule tolerance of " << *failsPerMoleculeTolerance
                    << ".\n";
        }
      }
    }
  }

  const bool foundTotalFailure = totalFailuresTolerance.has_value() && totalFailures > *totalFailuresTolerance;
  if (foundTotalFailure && printResults) {
    std::cout << "Total failures across all molecules: " << totalFailures
              << ", which exceeds the total failures tolerance of " << *totalFailuresTolerance << ".\n";
  }
  if (!foundTotalFailure && !foundPerMoleculeFailure) {
    return true;
  }
  if (foundPerMoleculeFailure && foundTotalFailure) {
    return false;
  }
  // If we made it here, it's 1 fail, but we might still pass based on one of the criteria..
  if (acceptEitherMetricAsPass) {
    return (totalFailuresTolerance.has_value() && !foundTotalFailure) ||
           (failsPerMoleculeTolerance.has_value() && !foundPerMoleculeFailure);
  }
  return false;  //(totalFailuresTolerance.has_value() && !foundTotalFailure) &&
                 //(failsPerMoleculeTolerance.has_value() && !foundPerMoleculeFailure)
}

bool checkForCompletedConformers(const std::vector<RDKit::ROMol*>& mols,
                                 const int                         numConfsExpected,
                                 const std::optional<int>          totalFailuresTolerance,
                                 const std::optional<int>          failsPerMoleculeTolerance,
                                 const bool                        acceptEitherMetricAsPass,
                                 const bool                        printResults) {
  std::vector<const RDKit::ROMol*> molsView;
  molsView.reserve(mols.size());
  for (const auto& mol : mols) {
    molsView.push_back(mol);
  }
  return checkForCompletedConformers(molsView,
                                     numConfsExpected,
                                     totalFailuresTolerance,
                                     failsPerMoleculeTolerance,
                                     acceptEitherMetricAsPass,
                                     printResults);
}
}  // namespace nvMolKit
