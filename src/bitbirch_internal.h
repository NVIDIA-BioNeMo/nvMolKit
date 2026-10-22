// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_BITBIRCH_INTERNAL_H
#define NVMOLKIT_BITBIRCH_INTERNAL_H

#include "src/bitbirch.h"

namespace nvMolKit::bitbirch {

//! Largest split tile whose static and dynamic shared memory fit both kernels.
int selectSplitTileSize(int numWords, int sharedMemoryBudget);

//! Inputs must be validated and nonempty. The budget includes static and dynamic shared memory per block.
BitBirchResult clusterWithSharedMemoryBudget(cuda::std::span<const std::uint32_t> fingerprints,
                                             int                                  numFingerprints,
                                             int                                  numWords,
                                             double                               threshold,
                                             const BitBirchOptions&               options,
                                             int                                  sharedMemoryBudget,
                                             cudaStream_t                         stream);

}  // namespace nvMolKit::bitbirch

#endif  // NVMOLKIT_BITBIRCH_INTERNAL_H
