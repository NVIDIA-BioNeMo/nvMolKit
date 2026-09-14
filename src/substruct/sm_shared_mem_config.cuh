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

#ifndef NVMOLKIT_SM_SHARED_MEM_CONFIG_CUH
#define NVMOLKIT_SM_SHARED_MEM_CONFIG_CUH

#include <cstddef>

#ifdef __CUDACC__
#define NVMOLKIT_HOST_DEVICE __host__ __device__
#else
#define NVMOLKIT_HOST_DEVICE
#endif

namespace nvMolKit {

/**
 * @brief Maximum shared memory per SM in KB for each compute capability.
 *
 * Values represent the maximum configurable shared memory per SM.
 * Only specific known architectures get their actual limits; others default to 100 KB.
 *
 * @param sm Combined SM version (e.g., 86 for SM 8.6, 90 for SM 9.0)
 */
NVMOLKIT_HOST_DEVICE constexpr int getMaxSharedMemoryPerSM_KB(const int sm) {
  if (sm >= 120)
    return 128;  // SM 12.0+ (Blackwell RTX)
  if (sm == 107)
    return 328;  // SM 10.7 (Rubin)
  if (sm >= 100)
    return 228;  // SM 10.0+ (Blackwell)
  if (sm >= 90)
    return 228;  // SM 9.0+ (Hopper)
  if (sm == 80)
    return 160;  // SM 8.0 (Ampere A100)

  return 100;  // Default: SM 8.6/8.9 (Ada/consumer Ampere), older, or unknown
}

/**
 * @brief Maximum shared memory per SM in KB for each compute capability.
 *
 * @param smMajor SM major version
 * @param smMinor SM minor version (default 0)
 */
NVMOLKIT_HOST_DEVICE constexpr int getMaxSharedMemoryPerSM_KB(const int smMajor, const int smMinor) {
  return getMaxSharedMemoryPerSM_KB(smMajor * 10 + smMinor);
}

/**
 * @brief Maximum shared memory per SM in bytes.
 */
NVMOLKIT_HOST_DEVICE constexpr std::size_t getMaxSharedMemoryPerSM(const int sm) {
  return static_cast<std::size_t>(getMaxSharedMemoryPerSM_KB(sm)) * 1024;
}

NVMOLKIT_HOST_DEVICE constexpr std::size_t getMaxSharedMemoryPerSM(const int smMajor, const int smMinor) {
  return static_cast<std::size_t>(getMaxSharedMemoryPerSM_KB(smMajor, smMinor)) * 1024;
}

/**
 * @brief Calculate shared memory per block for a target number of blocks per SM.
 *
 * @param sm Combined SM version (e.g., 86 for SM 8.6)
 * @param blocksPerSM Target blocks per SM for occupancy
 * @return Shared memory budget per block in bytes
 */
NVMOLKIT_HOST_DEVICE constexpr std::size_t getSharedMemoryPerBlock(const int sm, int blocksPerSM) {
  return getMaxSharedMemoryPerSM(sm) / blocksPerSM;
}

NVMOLKIT_HOST_DEVICE constexpr std::size_t getSharedMemoryPerBlock(const int smMajor,
                                                                   const int smMinor,
                                                                   int       blocksPerSM) {
  return getMaxSharedMemoryPerSM(smMajor, smMinor) / blocksPerSM;
}

/**
 * @brief Calculate maximum PartialMatch entries that fit in a shared memory budget.
 *
 * PartialMatch is 65 bytes (64-byte mapping + 1-byte nextQueryAtom).
 * We need space for ping-pong (2x) OR single buffer with copy.
 *
 * Applies a 10% safety buffer and rounds down to nearest multiple of 10.
 *
 * @param sharedMemBudget Total shared memory budget in bytes
 * @param labelMatrixBytes Space needed for label matrix (typically 1024)
 * @param controlVarsBytes Space for counters and control variables (typically 32)
 * @param usePingPong If true, budget covers 2 buffers; if false, single buffer
 * @return Number of PartialMatch entries per buffer (multiple of 10)
 */
NVMOLKIT_HOST_DEVICE constexpr int calculateMaxPartials(const std::size_t sharedMemBudget,
                                                        const std::size_t labelMatrixBytes = 1024,
                                                        const std::size_t controlVarsBytes = 32,
                                                        const bool        usePingPong      = true) {
  constexpr std::size_t kPartialMatchSize = 65;  // sizeof(PartialMatch)

  const std::size_t available        = (sharedMemBudget * 9 / 10) - labelMatrixBytes - controlVarsBytes;
  const int         bufferMultiplier = usePingPong ? 2 : 1;

  const int raw = static_cast<int>(available / (kPartialMatchSize * bufferMultiplier));
  return (raw / 10) * 10;  // Round down to nearest 10
}

/**
 * @brief Shared memory configuration for substructure search.
 *
 * Provides architecture-aware buffer sizing for GSI algorithm.
 */
template <int SmMajor, int SmMinor = 0, int BlocksPerSM = 4> struct SubstructSharedMemConfig {
  static constexpr int kSmMajor     = SmMajor;
  static constexpr int kSmMinor     = SmMinor;
  static constexpr int kBlocksPerSM = BlocksPerSM;

  static constexpr std::size_t kMaxSharedPerSM = getMaxSharedMemoryPerSM(SmMajor, SmMinor);
  static constexpr std::size_t kSharedPerBlock = kMaxSharedPerSM / BlocksPerSM;

  static constexpr std::size_t kLabelMatrixBytes = 1024;  // 128x64 bits = 8192 bits = 1KB
  static constexpr std::size_t kControlVarsBytes = 32;    // counters, offsets, etc.

  static constexpr int kMaxPartialsPerBlock =
    calculateMaxPartials(kSharedPerBlock, kLabelMatrixBytes, kControlVarsBytes, true);

  static constexpr int kMaxPartialsSingleBuffer =
    calculateMaxPartials(kSharedPerBlock, kLabelMatrixBytes, kControlVarsBytes, false);

  static constexpr std::size_t kActualSharedUsagePingPong =
    kLabelMatrixBytes + kControlVarsBytes + (kMaxPartialsPerBlock * 2 * 65);

  static constexpr std::size_t kActualSharedUsageSingle =
    kLabelMatrixBytes + kControlVarsBytes + (kMaxPartialsSingleBuffer * 65);
};

// Pre-defined configurations for common architectures
using SharedMemConfigSM80_4Blocks = SubstructSharedMemConfig<8, 0, 4>;  // Ampere, 4 blocks/SM
using SharedMemConfigSM80_8Blocks = SubstructSharedMemConfig<8, 0, 8>;  // Ampere, 8 blocks/SM
using SharedMemConfigSM89_4Blocks = SubstructSharedMemConfig<8, 9, 4>;  // Ada, 4 blocks/SM
using SharedMemConfigSM89_6Blocks = SubstructSharedMemConfig<8, 9, 6>;  // Ada, 6 blocks/SM
using SharedMemConfigSM90_4Blocks = SubstructSharedMemConfig<9, 0, 4>;  // Hopper, 4 blocks/SM
using SharedMemConfigSM90_8Blocks = SubstructSharedMemConfig<9, 0, 8>;  // Hopper, 8 blocks/SM

/**
 * @brief Runtime query for shared memory configuration.
 *
 * Use when SM version is only known at runtime.
 */
struct RuntimeSharedMemConfig {
  int maxSharedPerSM_KB;
  int sharedPerBlock;
  int maxPartialsPerBlock;
  int maxPartialsSingleBuffer;

  RuntimeSharedMemConfig(int smMajor, int smMinor, int blocksPerSM) {
    maxSharedPerSM_KB       = getMaxSharedMemoryPerSM_KB(smMajor, smMinor);
    sharedPerBlock          = (maxSharedPerSM_KB * 1024) / blocksPerSM;
    maxPartialsPerBlock     = calculateMaxPartials(sharedPerBlock, 1024, 32, true);
    maxPartialsSingleBuffer = calculateMaxPartials(sharedPerBlock, 1024, 32, false);
  }
};

/**
 * @brief Print shared memory configuration summary.
 */
template <typename Config> void printSharedMemConfig() {
  printf("SM %d.%d with %d blocks/SM:\n", Config::kSmMajor, Config::kSmMinor, Config::kBlocksPerSM);
  printf("  Max shared/SM:        %zu KB\n", Config::kMaxSharedPerSM / 1024);
  printf("  Budget/block:         %zu bytes (%.1f KB)\n", Config::kSharedPerBlock, Config::kSharedPerBlock / 1024.0);
  printf("  Max partials (ping-pong): %d (uses %zu bytes)\n",
         Config::kMaxPartialsPerBlock,
         Config::kActualSharedUsagePingPong);
  printf("  Max partials (single):    %d (uses %zu bytes)\n",
         Config::kMaxPartialsSingleBuffer,
         Config::kActualSharedUsageSingle);
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_SM_SHARED_MEM_CONFIG_CUH
