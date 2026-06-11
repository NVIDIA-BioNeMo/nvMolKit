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

#include <cuda/std/cstddef>

#include "src/substruct/substruct_types.h"

namespace nvMolKit {
TemplateConfigProperties getTemplateConfigProperties(SubstructTemplateConfig config) {
  switch (config) {
    case SubstructTemplateConfig::Config_T32_Q16_B4:
      return {32, 16, 4, 32 * 16, 32 * 16 / 32};
    case SubstructTemplateConfig::Config_T32_Q16_B6:
      return {32, 16, 6, 32 * 16, 32 * 16 / 32};
    case SubstructTemplateConfig::Config_T32_Q16_B8:
      return {32, 16, 8, 32 * 16, 32 * 16 / 32};
    case SubstructTemplateConfig::Config_T32_Q32_B4:
      return {32, 32, 4, 32 * 32, 32 * 32 / 32};
    case SubstructTemplateConfig::Config_T32_Q32_B6:
      return {32, 32, 6, 32 * 32, 32 * 32 / 32};
    case SubstructTemplateConfig::Config_T32_Q32_B8:
      return {32, 32, 8, 32 * 32, 32 * 32 / 32};
    case SubstructTemplateConfig::Config_T64_Q16_B4:
      return {64, 16, 4, 64 * 16, 64 * 16 / 32};
    case SubstructTemplateConfig::Config_T64_Q16_B6:
      return {64, 16, 6, 64 * 16, 64 * 16 / 32};
    case SubstructTemplateConfig::Config_T64_Q16_B8:
      return {64, 16, 8, 64 * 16, 64 * 16 / 32};
    case SubstructTemplateConfig::Config_T64_Q32_B4:
      return {64, 32, 4, 64 * 32, 64 * 32 / 32};
    case SubstructTemplateConfig::Config_T64_Q32_B6:
      return {64, 32, 6, 64 * 32, 64 * 32 / 32};
    case SubstructTemplateConfig::Config_T64_Q32_B8:
      return {64, 32, 8, 64 * 32, 64 * 32 / 32};
    case SubstructTemplateConfig::Config_T64_Q64_B4:
      return {64, 64, 4, 64 * 64, 64 * 64 / 32};
    case SubstructTemplateConfig::Config_T64_Q64_B6:
      return {64, 64, 6, 64 * 64, 64 * 64 / 32};
    case SubstructTemplateConfig::Config_T64_Q64_B8:
      return {64, 64, 8, 64 * 64, 64 * 64 / 32};
    case SubstructTemplateConfig::Config_T128_Q16_B4:
      return {128, 16, 4, 128 * 16, 128 * 16 / 32};
    case SubstructTemplateConfig::Config_T128_Q16_B6:
      return {128, 16, 6, 128 * 16, 128 * 16 / 32};
    case SubstructTemplateConfig::Config_T128_Q16_B8:
      return {128, 16, 8, 128 * 16, 128 * 16 / 32};
    case SubstructTemplateConfig::Config_T128_Q32_B4:
      return {128, 32, 4, 128 * 32, 128 * 32 / 32};
    case SubstructTemplateConfig::Config_T128_Q32_B6:
      return {128, 32, 6, 128 * 32, 128 * 32 / 32};
    case SubstructTemplateConfig::Config_T128_Q32_B8:
      return {128, 32, 8, 128 * 32, 128 * 32 / 32};
    case SubstructTemplateConfig::Config_T128_Q64_B4:
      return {128, 64, 4, 128 * 64, 128 * 64 / 32};
    case SubstructTemplateConfig::Config_T128_Q64_B6:
      return {128, 64, 6, 128 * 64, 128 * 64 / 32};
    case SubstructTemplateConfig::Config_T128_Q64_B8:
      return {128, 64, 8, 128 * 64, 128 * 64 / 32};
    default:
      return {128, 64, 8, 128 * 64, 128 * 64 / 32};  // fallback to max
  }
}

std::size_t computeLabelMatrixWords(int maxTargetAtoms, int maxQueryAtoms) {
  return static_cast<std::size_t>(maxTargetAtoms) * maxQueryAtoms / 32;
}

SubstructTemplateConfig selectTemplateConfig(int maxTargetAtoms, int maxQueryAtoms, int maxBondsPerAtom) {
  // Clamp maxBondsPerAtom to valid template values: 4, 6, or 8
  int bondConfig;
  if (maxBondsPerAtom <= 4) {
    bondConfig = 4;
  } else if (maxBondsPerAtom <= 6) {
    bondConfig = 6;
  } else {
    bondConfig = 8;
  }

  // Select target size tier
  int targetTier;
  if (maxTargetAtoms <= 32) {
    targetTier = 32;
  } else if (maxTargetAtoms <= 64) {
    targetTier = 64;
  } else {
    targetTier = 128;
  }

  // Select query size tier (must be <= target tier)
  int queryTier;
  if (maxQueryAtoms <= 16) {
    queryTier = 16;
  } else if (maxQueryAtoms <= 32) {
    queryTier = 32;
  } else {
    queryTier = 64;
  }

  // Ensure query <= target
  if (queryTier > targetTier) {
    queryTier = targetTier;
  }

  // Map to config enum
  if (targetTier == 32) {
    if (queryTier == 16) {
      if (bondConfig == 4)
        return SubstructTemplateConfig::Config_T32_Q16_B4;
      if (bondConfig == 6)
        return SubstructTemplateConfig::Config_T32_Q16_B6;
      return SubstructTemplateConfig::Config_T32_Q16_B8;
    } else {  // queryTier == 32
      if (bondConfig == 4)
        return SubstructTemplateConfig::Config_T32_Q32_B4;
      if (bondConfig == 6)
        return SubstructTemplateConfig::Config_T32_Q32_B6;
      return SubstructTemplateConfig::Config_T32_Q32_B8;
    }
  } else if (targetTier == 64) {
    if (queryTier == 16) {
      if (bondConfig == 4)
        return SubstructTemplateConfig::Config_T64_Q16_B4;
      if (bondConfig == 6)
        return SubstructTemplateConfig::Config_T64_Q16_B6;
      return SubstructTemplateConfig::Config_T64_Q16_B8;
    } else if (queryTier == 32) {
      if (bondConfig == 4)
        return SubstructTemplateConfig::Config_T64_Q32_B4;
      if (bondConfig == 6)
        return SubstructTemplateConfig::Config_T64_Q32_B6;
      return SubstructTemplateConfig::Config_T64_Q32_B8;
    } else {  // queryTier == 64
      if (bondConfig == 4)
        return SubstructTemplateConfig::Config_T64_Q64_B4;
      if (bondConfig == 6)
        return SubstructTemplateConfig::Config_T64_Q64_B6;
      return SubstructTemplateConfig::Config_T64_Q64_B8;
    }
  } else {  // targetTier == 128
    if (queryTier == 16) {
      if (bondConfig == 4)
        return SubstructTemplateConfig::Config_T128_Q16_B4;
      if (bondConfig == 6)
        return SubstructTemplateConfig::Config_T128_Q16_B6;
      return SubstructTemplateConfig::Config_T128_Q16_B8;
    } else if (queryTier == 32) {
      if (bondConfig == 4)
        return SubstructTemplateConfig::Config_T128_Q32_B4;
      if (bondConfig == 6)
        return SubstructTemplateConfig::Config_T128_Q32_B6;
      return SubstructTemplateConfig::Config_T128_Q32_B8;
    } else {  // queryTier == 64
      if (bondConfig == 4)
        return SubstructTemplateConfig::Config_T128_Q64_B4;
      if (bondConfig == 6)
        return SubstructTemplateConfig::Config_T128_Q64_B6;
      return SubstructTemplateConfig::Config_T128_Q64_B8;
    }
  }
}
}  // namespace nvMolKit