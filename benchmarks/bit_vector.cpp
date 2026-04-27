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

#include <DataStructs/ExplicitBitVect.h>
#include <nanobench.h>

#include <array>
#include <random>

#include "src/data_structures/flat_bit_vect.h"

constexpr std::array<std::size_t, 5> kNBits           = {1, 10, 100, 1000, 10000};
constexpr std::array<std::size_t, 5> kNumBitsInVector = {1, 10, 100, 1000, 10000};

#define BENCHMARK_FLAT_BIT_VECT_ALLOCATION(nBits)                                                                  \
  ankerl::nanobench::Bench().run("Allocation: FlatBitVect<" #nBits ">",                                            \
                                 [&]() { ankerl::nanobench::doNotOptimizeAway(nvMolKit::FlatBitVect<nBits>()); }); \
  ankerl::nanobench::Bench().run("Allocation + initialization: FlatBitVect<" #nBits ">",                           \
                                 [&]() { ankerl::nanobench::doNotOptimizeAway(nvMolKit::FlatBitVect<nBits>(true)); });

#define BENCHMARK_FLAT_BIT_MOVE(nBits)                            \
  nvMolKit::FlatBitVect<nBits> bitsOneMove##nBits;                \
  nvMolKit::FlatBitVect<nBits> bitsTwoMove##nBits;                \
  ankerl::nanobench::Bench().run("Move: FlatBitVect<" #nBits ">", \
                                 [&]() { bitsOneMove##nBits = std::move(bitsTwoMove##nBits); });

#define BENCHMARK_FLAT_BIT_SWAP(nBits)         \
  nvMolKit::FlatBitVect<nBits> bitsOne##nBits; \
  nvMolKit::FlatBitVect<nBits> bitsTwo##nBits; \
  ankerl::nanobench::Bench().run("Swap: FlatBitVect<" #nBits ">", [&]() { std::swap(bitsOne##nBits, bitsTwo##nBits); });

// TODO consolidate definitions
#define BENCHMARK_FLAT_BIT_EQUALITY(nBits)                                                                            \
  nvMolKit::FlatBitVect<nBits> bitsOneEq##nBits(false);                                                               \
  nvMolKit::FlatBitVect<nBits> bitsTwoEq##nBits(false);                                                               \
  for (size_t i = 0; i < nBits; i++) {                                                                                \
    bool toSet = gen();                                                                                               \
    if (toSet) {                                                                                                      \
      bitsOneEq##nBits.setBit(i, true);                                                                               \
      bitsTwoEq##nBits.setBit(i, true);                                                                               \
    }                                                                                                                 \
  }                                                                                                                   \
  ankerl::nanobench::Bench().run("Equality (==True): FlatBitVect(" + std::to_string(nBits) + ")", [&]() {             \
    ankerl::nanobench::doNotOptimizeAway(bitsOneEq##nBits == bitsTwoEq##nBits);                                       \
  });                                                                                                                 \
  bitsOneEq##nBits.setBit(0, true);                                                                                   \
  bitsTwoEq##nBits.setBit(0, false);                                                                                  \
  ankerl::nanobench::Bench().run("Equality (==False, early bit): FlatBitVect(" + std::to_string(nBits) + ")", [&]() { \
    ankerl::nanobench::doNotOptimizeAway(bitsOneEq##nBits == bitsTwoEq##nBits);                                       \
  });

#define BENCHMARK_FLAT_BIT_VECT_STD_VECTOR_ALLOCATION(nBits, nElems)                  \
  ankerl::nanobench::Bench().run(                                                     \
    "Vector Allocation of " + std::to_string(vecSize) + "+: FlatBitVect<" #nBits ">", \
    [&]() { ankerl::nanobench::doNotOptimizeAway(std::vector<nvMolKit::FlatBitVect<nBits>>(nElems)); });

int main() {
  auto gen = std::bind(std::uniform_int_distribution<>(0, 1), std::default_random_engine());

  //-----------------------
  // Allocation time test
  //-----------------------

  BENCHMARK_FLAT_BIT_VECT_ALLOCATION(1);
  BENCHMARK_FLAT_BIT_VECT_ALLOCATION(10);
  BENCHMARK_FLAT_BIT_VECT_ALLOCATION(100);
  BENCHMARK_FLAT_BIT_VECT_ALLOCATION(1000);
  BENCHMARK_FLAT_BIT_VECT_ALLOCATION(10000);
  for (const auto& nBits : kNBits) {
    ankerl::nanobench::Bench().run("Allocation: ExplicitBitVect(" + std::to_string(nBits) + ")",
                                   [&]() { ankerl::nanobench::doNotOptimizeAway(ExplicitBitVect(nBits)); });
  }

  // ----------------
  // Move check
  // ----------------
  BENCHMARK_FLAT_BIT_MOVE(1);
  BENCHMARK_FLAT_BIT_MOVE(10);
  BENCHMARK_FLAT_BIT_MOVE(100);
  BENCHMARK_FLAT_BIT_MOVE(1000);
  BENCHMARK_FLAT_BIT_MOVE(10000);

  for (const auto& nBits : kNBits) {
    ExplicitBitVect bitsOneMove(nBits);
    ExplicitBitVect bitsTwoMove(nBits);
    ankerl::nanobench::Bench().run("Move: ExplicitBitVect(" + std::to_string(nBits) + ")",
                                   [&]() { bitsOneMove = std::move(bitsTwoMove); });
  }

  // ---------------
  // Swap speed test
  // ---------------
  BENCHMARK_FLAT_BIT_SWAP(1);
  BENCHMARK_FLAT_BIT_SWAP(10);
  BENCHMARK_FLAT_BIT_SWAP(100);
  BENCHMARK_FLAT_BIT_SWAP(1000);
  BENCHMARK_FLAT_BIT_SWAP(10000);

  for (const auto& nBits : kNBits) {
    ExplicitBitVect bitsOne(nBits);
    ExplicitBitVect bitsTwo(nBits);
    ankerl::nanobench::Bench().run("Swap: ExplicitBitVect(" + std::to_string(nBits) + ")",
                                   [&]() { std::swap(bitsOne, bitsTwo); });
  }

  // ------------------------
  // Equality check
  // ------------------------
  BENCHMARK_FLAT_BIT_EQUALITY(1);
  BENCHMARK_FLAT_BIT_EQUALITY(10);
  BENCHMARK_FLAT_BIT_EQUALITY(100);
  BENCHMARK_FLAT_BIT_EQUALITY(1000);
  BENCHMARK_FLAT_BIT_EQUALITY(10000);

  for (const auto& nBits : kNBits) {
    ExplicitBitVect bitsOne(nBits);
    ExplicitBitVect bitsTwo(nBits);
    for (size_t i = 0; i < nBits; i++) {
      bool toSet = gen();
      if (toSet) {
        bitsOne.setBit(i);
        bitsTwo.setBit(i);
      }
    }
    ankerl::nanobench::Bench().run("Equality (==True): ExplicitBitVect(" + std::to_string(nBits) + ")",
                                   [&]() { ankerl::nanobench::doNotOptimizeAway(bitsOne == bitsTwo); });

    bitsOne.setBit(0);
    bitsTwo.unsetBit(0);

    ankerl::nanobench::Bench().run("Equality (==False, early bit): ExplicitBitVect(" + std::to_string(nBits) + ")",
                                   [&]() { ankerl::nanobench::doNotOptimizeAway(bitsOne == bitsTwo); });
  }

  // --------------------------------------------
  // Allocation time test - putting into a vector
  //  -------------------------------------------
  for (const auto& vecSize : kNumBitsInVector) {
    BENCHMARK_FLAT_BIT_VECT_STD_VECTOR_ALLOCATION(1, vecSize);
    BENCHMARK_FLAT_BIT_VECT_STD_VECTOR_ALLOCATION(10, vecSize);
    BENCHMARK_FLAT_BIT_VECT_STD_VECTOR_ALLOCATION(100, vecSize);
    BENCHMARK_FLAT_BIT_VECT_STD_VECTOR_ALLOCATION(1000, vecSize);
    BENCHMARK_FLAT_BIT_VECT_STD_VECTOR_ALLOCATION(10000, vecSize);

    for (const auto& nBits : kNBits) {
      ankerl::nanobench::Bench().run(
        "Vector Allocation of " + std::to_string(vecSize) + "+: ExplicitBitVect(" + std::to_string(nBits) + ")",
        [&]() { ankerl::nanobench::doNotOptimizeAway(std::vector<ExplicitBitVect>(vecSize, ExplicitBitVect(nBits))); });
    }
  }
}
