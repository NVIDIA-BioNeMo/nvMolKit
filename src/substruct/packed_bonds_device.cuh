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

#ifndef NVMOLKIT_PACKED_BONDS_DEVICE_CUH
#define NVMOLKIT_PACKED_BONDS_DEVICE_CUH

#include "src/substruct/packed_bonds.h"

namespace nvMolKit {

/**
 * @brief Extract bond type from packed target bond info.
 */
__device__ __forceinline__ int unpackBondType(uint8_t bondInfo) {
  return bondInfo & 0x0F;
}

/**
 * @brief Extract isInRing from packed target bond info.
 */
__device__ __forceinline__ int unpackIsInRing(uint8_t bondInfo) {
  return (bondInfo >> 4) & 1;
}

/**
 * @brief Check if query bond matches target bond. Fully branchless.
 *
 * @param queryMask Precomputed 32-bit match mask from QueryAtomBonds
 * @param targetBondInfo Packed bond info from TargetAtomBonds
 * @return true if the bonds are compatible
 */
__device__ __forceinline__ bool packedBondMatches(uint32_t queryMask, uint8_t targetBondInfo) {
  const int bondType = unpackBondType(targetBondInfo);
  const int isInRing = unpackIsInRing(targetBondInfo);
  return (queryMask >> (isInRing * 16 + bondType)) & 1;
}

/**
 * @brief Check edge consistency using packed bond data with templated loop unrolling.
 *
 * Storage is always max-sized (kMaxBondsPerAtom=8), but templates on the
 * expected maximum bonds for tighter loop unrolling.
 *
 * For each already-mapped neighbor of queryAtom, verifies that the corresponding
 * edge exists in the target graph with compatible bond properties.
 *
 * @tparam MaxBonds Maximum bonds per atom for loop unrolling (4, 6, or 8)
 * @param targetBonds Array of packed target atom bonds (indexed by atom)
 * @param queryBonds Packed bonds for the current query atom
 * @param mapping Current partial mapping (query atom -> target atom)
 * @param queryAtom Current query atom index (equals current search depth)
 * @param targetAtom Candidate target atom to extend mapping with
 * @return true if edge consistency is satisfied
 */
template <int MaxBonds = kMaxBondsPerAtom>
__device__ __forceinline__ bool checkEdgeConsistencyPacked(const TargetAtomBonds* targetBonds,
                                                           const QueryAtomBonds&  queryBonds,
                                                           const int8_t*          mapping,
                                                           int                    queryAtom,
                                                           int                    targetAtom) {
  const int              depth        = queryAtom;
  const TargetAtomBonds& tb           = targetBonds[targetAtom];
  const int              queryDegree  = queryBonds.degree;
  const int              targetDegree = tb.degree;

  for (int i = 0; i < queryDegree; ++i) {
    const int neighborQueryAtom = queryBonds.neighborIdx[i];

    if (neighborQueryAtom >= depth) {
      continue;
    }

    const int      neighborTargetAtom = mapping[neighborQueryAtom];
    const uint32_t queryMask          = queryBonds.matchMask[i];

    bool foundMatch = false;

#pragma unroll
    for (int j = 0; j < MaxBonds; ++j) {
      if (j >= targetDegree) {
        break;
      }
      if (tb.neighborIdx[j] == neighborTargetAtom) {
        foundMatch = packedBondMatches(queryMask, tb.bondInfo[j]);
        break;
      }
    }

    if (!foundMatch) {
      return false;
    }
  }

  return true;
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_PACKED_BONDS_DEVICE_CUH
