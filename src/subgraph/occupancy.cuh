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

// Compile-time occupancy modelling for kernels whose residency is decided by
// one dominant __shared__ allocation, used to pick the second
// __launch_bounds__ argument.

#ifndef NVMOLKIT_SUBGRAPH_OCCUPANCY_CUH
#define NVMOLKIT_SUBGRAPH_OCCUPANCY_CUH

#include <cstddef>

namespace nvMolKit {

/**
 * @brief Per-SM shared memory budget assumed when choosing minBlocksPerSM.
 *
 * Hopper and datacenter Blackwell (sm_90/100/103) have 228 KB per SM; everything
 * else is modelled as A100's 164 KB, deliberately including the architectures
 * with less (Turing 64 KB, Volta 96 KB, consumer Ampere/Ada/Blackwell 100 KB).
 * ptxas does not validate .minnctapersm against shared memory, so on those an
 * unreachable ask costs nothing at runtime -- occupancy just lands where shared
 * memory puts it -- but it keeps the register cap as tight as the tuned
 * configuration allows. Modelling the 228 KB parts does matter: the smaller
 * assumed budget under-asks for the largest shapes there, and the relaxed
 * register cap can drop real occupancy below what shared memory allows.
 */
constexpr std::size_t sharedBudgetPerSMBytes() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200)
  return 228 * 1024;
#else
  return 164 * 1024;
#endif
}

/**
 * @brief Max resident threads per SM, from the CUDA occupancy tables.
 *
 * Unlike shared memory, ptxas validates .minnctapersm against this limit and
 * fails the build when minBlocks * blockSize exceeds it, so minBlocksPerSM
 * must clamp by it.
 */
constexpr int maxThreadsPerSM() {
#if !defined(__CUDA_ARCH__)
  return 2048;  // Host pass; the value is never used.
#elif __CUDA_ARCH__ == 750
  return 1024;  // Turing
#elif (__CUDA_ARCH__ >= 860 && __CUDA_ARCH__ < 900) || __CUDA_ARCH__ >= 1200
  return 1536;  // Consumer Ampere/Ada, Orin, consumer Blackwell
#else
  return 2048;  // Volta, A100, Hopper, datacenter Blackwell
#endif
}

/**
 * @brief Second __launch_bounds__ argument: CTAs the kernel should fit per SM.
 *
 * @tparam SharedStateBytes   Size of the kernel's dominant (assumed only)
 *                            __shared__ allocation per CTA.
 * @tparam BlockSize          Threads per CTA.
 * @tparam TargetResidentCTAs The residency the kernel is tuned at; asking for
 *                            it also caps the compiler's register budget
 *                            accordingly, so the two go together. The result is
 *                            this value when shared memory and the SM thread
 *                            limit allow it, otherwise whatever does fit
 *                            (at least 1).
 */
template <std::size_t SharedStateBytes, int BlockSize, int TargetResidentCTAs = 8> constexpr int minBlocksPerSM() {
  constexpr std::size_t kBySharedMem = sharedBudgetPerSMBytes() / SharedStateBytes;
  constexpr std::size_t kByThreads   = static_cast<std::size_t>(maxThreadsPerSM() / BlockSize);
  constexpr std::size_t kResident    = kBySharedMem < kByThreads ? kBySharedMem : kByThreads;
  return kResident >= static_cast<std::size_t>(TargetResidentCTAs) ? TargetResidentCTAs :
                                                                     (kResident < 1 ? 1 : static_cast<int>(kResident));
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBGRAPH_OCCUPANCY_CUH
