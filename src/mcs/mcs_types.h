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

#ifndef NVMOLKIT_MCS_TYPES_H
#define NVMOLKIT_MCS_TYPES_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace nvMolKit {

enum class MCSAtomCompare : std::uint8_t {
  Any,
  Elements,
  Isotopes,
  AnyHeavyAtom
};

enum class MCSBondCompare : std::uint8_t {
  Any,
  Order,
  OrderExact
};

struct MCSAtomCompareParameters {
  bool matchValences       = false;
  bool matchFormalCharge   = false;
  bool ringMatchesRingOnly = false;
  bool matchIsotope        = false;
};

struct MCSBondCompareParameters {
  bool ringMatchesRingOnly = false;
};

struct MCSParameters {
  bool                     maximizeBonds        = true;
  bool                     connectedOnly        = true;
  bool                     requireGpu           = false;
  unsigned int             timeoutSeconds       = 0;
  int                      batchSize            = 0;
  int                      blockSize            = 128;
  int                      workerThreads        = -1;
  int                      preprocessingThreads = -1;
  int                      executorsPerRunner   = -1;
  std::vector<int>         gpuIds;
  MCSAtomCompare           atomCompare = MCSAtomCompare::Elements;
  MCSBondCompare           bondCompare = MCSBondCompare::Order;
  MCSAtomCompareParameters atomCompareParameters;
  MCSBondCompareParameters bondCompareParameters;
};

struct MCSResult {
  unsigned int numAtoms     = 0;
  unsigned int numBonds     = 0;
  bool         canceled     = false;
  bool         overflowed   = false;
  bool         usedGpu      = false;
  bool         usedFallback = false;
  std::string  smartsString;

  std::vector<std::pair<int, int>> atomMapping;
  std::vector<std::pair<int, int>> bondMapping;

  [[nodiscard]] bool isCompleted() const { return !canceled; }
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_MCS_TYPES_H
