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

#include "src/substruct/packed_bonds.h"

namespace nvMolKit {

uint32_t buildQueryBondMatchMask(uint8_t queryBondType, uint8_t queryFlags, uint16_t allowedBondTypes) {
  uint16_t typeMask;

  if (queryFlags & BondQueryNeverMatches) {
    typeMask = 0;
  } else if (queryFlags & BondQueryUseBondMask) {
    typeMask = allowedBondTypes;
  } else if (queryBondType == 0) {
    typeMask = 0xFFFF;
  } else {
    typeMask = static_cast<uint16_t>(1u << queryBondType);
  }

  const bool mustBeRing    = (queryFlags & BondQueryIsRingBond) != 0;
  const bool mustNotBeRing = (queryFlags & BondQueryNotRingBond) != 0;

  uint16_t maskNotInRing = mustBeRing ? 0 : typeMask;
  uint16_t maskInRing    = mustNotBeRing ? 0 : typeMask;

  return static_cast<uint32_t>(maskNotInRing) | (static_cast<uint32_t>(maskInRing) << 16);
}

}  // namespace nvMolKit
