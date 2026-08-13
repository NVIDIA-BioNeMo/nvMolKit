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

#ifndef NVMOLKIT_SUBGRAPH_TARGET_MASK_CUH
#define NVMOLKIT_SUBGRAPH_TARGET_MASK_CUH

#include <cstddef>
#include <cstdint>

namespace nvMolKit {

/**
 * @brief Set of target atoms, bit t meaning target atom t.
 *
 * Specialised on the atom capacity rather than looped over an array so that each
 * form is exactly as wide as it needs to be -- a bare uint32_t at 32 atoms, a
 * uint64_t at 64 -- and the 128-atom form never dynamically indexes a
 * register-resident array, which would spill it to local memory. The 32-atom
 * form matters because these masks are the DFS inner loop: on 32-bit ALUs every
 * 64-bit operation is two instructions, and popcount/ffs are two-instruction
 * emulations, so the narrow form roughly halves the work per mask operation and
 * halves the local-memory traffic of the per-depth candidate stack.
 *
 * setWord32(k, v) writes the k-th 32-bit lane group, which is how a label
 * matrix transpose assembles a mask out of per-root ballots.
 */
template <std::size_t MaxAtoms> struct TargetMask;

template <> struct TargetMask<32> {
  uint32_t bits;

  __device__ __forceinline__ void clear() { bits = 0; }
  __device__ __forceinline__ bool empty() const { return bits == 0; }
  __device__ __forceinline__ bool test(int bit) const { return ((bits >> bit) & 1u) != 0; }
  __device__ __forceinline__ void set(int bit) { bits |= 1u << bit; }
  __device__ __forceinline__ void reset(int bit) { bits &= ~(1u << bit); }
  /// Branch-free conditional set: @p value must be 0 or 1.
  __device__ __forceinline__ void setIf(int bit, uint64_t value) { bits |= static_cast<uint32_t>(value) << bit; }
  __device__ __forceinline__ void setWord32(int, uint32_t value) { bits |= value; }
  __device__ __forceinline__ int  popcount() const { return __popc(bits); }
  /// Lowest set atom index; undefined if empty.
  __device__ __forceinline__ int  lowest() const { return __ffs(static_cast<int>(bits)) - 1; }
  __device__ __forceinline__ void clearLowest() { bits &= bits - 1; }
  __device__ __forceinline__ void andEq(const TargetMask& other) { bits &= other.bits; }
  __device__ __forceinline__ void andNotEq(const TargetMask& other) { bits &= ~other.bits; }
};

template <> struct TargetMask<64> {
  uint64_t lo;

  __device__ __forceinline__ void clear() { lo = 0; }
  __device__ __forceinline__ bool empty() const { return lo == 0; }
  __device__ __forceinline__ bool test(int bit) const { return ((lo >> bit) & 1ULL) != 0; }
  __device__ __forceinline__ void set(int bit) { lo |= 1ULL << bit; }
  __device__ __forceinline__ void reset(int bit) { lo &= ~(1ULL << bit); }
  /// Branch-free conditional set: @p value must be 0 or 1.
  __device__ __forceinline__ void setIf(int bit, uint64_t value) { lo |= value << bit; }
  __device__ __forceinline__ void setWord32(int word, uint32_t value) {
    lo |= static_cast<uint64_t>(value) << (32 * word);
  }
  __device__ __forceinline__ int  popcount() const { return __popcll(lo); }
  /// Lowest set atom index; undefined if empty.
  __device__ __forceinline__ int  lowest() const { return __ffsll(static_cast<long long>(lo)) - 1; }
  __device__ __forceinline__ void clearLowest() { lo &= lo - 1; }
  __device__ __forceinline__ void andEq(const TargetMask& other) { lo &= other.lo; }
  __device__ __forceinline__ void andNotEq(const TargetMask& other) { lo &= ~other.lo; }
};

template <> struct TargetMask<128> {
  uint64_t lo;
  uint64_t hi;

  __device__ __forceinline__ void clear() { lo = hi = 0; }
  __device__ __forceinline__ bool empty() const { return (lo | hi) == 0; }
  __device__ __forceinline__ bool test(int bit) const { return (((bit < 64 ? lo : hi) >> (bit & 63)) & 1ULL) != 0; }
  __device__ __forceinline__ void set(int bit) {
    if (bit < 64) {
      lo |= 1ULL << bit;
    } else {
      hi |= 1ULL << (bit - 64);
    }
  }
  __device__ __forceinline__ void reset(int bit) {
    if (bit < 64) {
      lo &= ~(1ULL << bit);
    } else {
      hi &= ~(1ULL << (bit - 64));
    }
  }
  /// Branch-free conditional set: @p value must be 0 or 1.
  __device__ __forceinline__ void setIf(int bit, uint64_t value) {
    if (bit < 64) {
      lo |= value << bit;
    } else {
      hi |= value << (bit - 64);
    }
  }
  __device__ __forceinline__ void setWord32(int word, uint32_t value) {
    if (word < 2) {
      lo |= static_cast<uint64_t>(value) << (32 * word);
    } else {
      hi |= static_cast<uint64_t>(value) << (32 * (word - 2));
    }
  }
  __device__ __forceinline__ int popcount() const { return __popcll(lo) + __popcll(hi); }
  /// Lowest set atom index; undefined if empty.
  __device__ __forceinline__ int lowest() const {
    return lo != 0 ? __ffsll(static_cast<long long>(lo)) - 1 : __ffsll(static_cast<long long>(hi)) + 63;
  }
  __device__ __forceinline__ void clearLowest() {
    if (lo != 0) {
      lo &= lo - 1;
    } else {
      hi &= hi - 1;
    }
  }
  __device__ __forceinline__ void andEq(const TargetMask& other) {
    lo &= other.lo;
    hi &= other.hi;
  }
  __device__ __forceinline__ void andNotEq(const TargetMask& other) {
    lo &= ~other.lo;
    hi &= ~other.hi;
  }
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBGRAPH_TARGET_MASK_CUH
