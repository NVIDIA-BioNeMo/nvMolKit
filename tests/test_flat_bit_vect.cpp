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

#include <gtest/gtest.h>

#include <boost/dynamic_bitset.hpp>
#include <random>
#include <unordered_set>

#include "src/data_structures/flat_bit_vect.h"

constexpr std::array<std::size_t, 5> nBitsTest = {3, 32, 33, 1024, 1025};
constexpr std::array<std::size_t, 5> wantSizes = {1, 1, 2, 32, 33};

template <std::size_t index> void runSizeTest() {
  constexpr int nBits = nBitsTest[index];
  SCOPED_TRACE("Size test for FlatBitVect<" + std::to_string(nBits) + ">");
  nvMolKit::FlatBitVect<nBits> fbv;
  ASSERT_EQ(fbv.kStorageCount, wantSizes[index]);
}

template <std::size_t index> void runInitAndClearTest() {
  constexpr int nBits = nBitsTest[index];

  auto fbv = nvMolKit::FlatBitVect<nBits>(false);
  for (std::size_t i = 0; i < nBits; ++i) {
    ASSERT_EQ(fbv[i], false) << "i = " << i;
  }

  fbv = nvMolKit::FlatBitVect<nBits>(true);
  for (std::size_t i = 0; i < nBits; ++i) {
    ASSERT_EQ(fbv[i], true) << "i = " << i;
  }

  fbv.clear();
  for (std::size_t i = 0; i < nBits; ++i) {
    ASSERT_EQ(fbv[i], false) << "i = " << i;
  }

  fbv.setBit(1, true);
  EXPECT_EQ(fbv[0], false);
  EXPECT_EQ(fbv[1], true);
  ASSERT_EQ(fbv[2], false);

  fbv.setBit(1, false);
  EXPECT_EQ(fbv[1], false);
}

TEST(FlatBitVect, CopyConstructor) {
  nvMolKit::FlatBitVect<64> fbv1(true);
  nvMolKit::FlatBitVect<64> fbv2(fbv1);
  for (std::size_t i = 0; i < 64; ++i) {
    ASSERT_EQ(fbv1[i], true) << "i = " << i;
    ASSERT_EQ(fbv2[i], true) << "i = " << i;
  }
}

TEST(FlatBitVect, CopyAssignmentOperator) {
  nvMolKit::FlatBitVect<64> fbv1(true);
  nvMolKit::FlatBitVect<64> fbv2 = fbv1;
  for (std::size_t i = 0; i < 64; ++i) {
    ASSERT_EQ(fbv1[i], true) << "i = " << i;
    ASSERT_EQ(fbv2[i], true) << "i = " << i;
  }
}

TEST(FlatBitVect, MoveConstructor) {
  nvMolKit::FlatBitVect<64> fbv1(true);
  nvMolKit::FlatBitVect<64> fbv2(std::move(fbv1));
  for (std::size_t i = 0; i < 64; ++i) {
    ASSERT_EQ(fbv1[i], true) << "i = " << i;
    ASSERT_EQ(fbv2[i], true) << "i = " << i;
  }
}

TEST(FlatBitVect, MoveAssignment) {
  nvMolKit::FlatBitVect<64> fbv1(true);
  nvMolKit::FlatBitVect<64> fbv2 = std::move(fbv1);
  for (std::size_t i = 0; i < 64; ++i) {
    ASSERT_EQ(fbv2[i], true) << "i = " << i;
  }
}

// TODO get typed tests working....
TEST(FlatBitVect, TestFlatBitVectSize) {
  runSizeTest<0>();
  runSizeTest<1>();
  runSizeTest<2>();
  runSizeTest<3>();
  runSizeTest<4>();
}

TEST(FlatBitVect, TestInitAndClear3) {
  runInitAndClearTest<0>();
}

TEST(FlatBitVect, TestInitAndClear32) {
  runInitAndClearTest<1>();
}

TEST(FlatBitVect, TestInitAndClear33) {
  runInitAndClearTest<2>();
}

TEST(FlatBitVect, TestInitAndClear1024) {
  runInitAndClearTest<3>();
}
TEST(FlatBitVect, TestInitAndClear1025) {
  runInitAndClearTest<4>();
}

TEST(FlatBitVect, Equality) {
  nvMolKit::FlatBitVect<64> fbv1(true);
  nvMolKit::FlatBitVect<64> fbv2(true);
  ASSERT_EQ(fbv1, fbv2);
  fbv2.setBit(37, false);
  ASSERT_NE(fbv1, fbv2);
}

// Lexicographical less than, bit order dependent.
TEST(FlatBitVect, OperatorLessThan) {
  nvMolKit::FlatBitVect<64> fbv1(false);
  nvMolKit::FlatBitVect<64> fbv2(false);
  ASSERT_FALSE(fbv1 < fbv2);
  fbv2.setBit(37, true);
  ASSERT_TRUE(fbv1 < fbv2);
  fbv1 = nvMolKit::FlatBitVect<64>(true);
  fbv2 = nvMolKit::FlatBitVect<64>(true);
  ASSERT_FALSE(fbv1 < fbv2);
  fbv2.setBit(37, false);
  ASSERT_FALSE(fbv1 < fbv2);
  fbv1.setBit(38, false);
  ASSERT_TRUE(fbv1 < fbv2);
}

// Lexicographical less than, bit order dependent.
TEST(FlatBitVect, OperatorLessThanBoostCompare) {
  constexpr int                   compSize = 128;
  nvMolKit::FlatBitVect<compSize> fbv1(false);
  nvMolKit::FlatBitVect<compSize> fbv2(false);
  boost::dynamic_bitset<>         boost1(compSize);
  boost::dynamic_bitset<>         boost2(compSize);
  auto                            gen = std::bind(std::uniform_int_distribution<>(0, 1), std::default_random_engine());
  for (int trial = 0; trial < 100; trial++) {
    fbv1.clear();
    fbv2.clear();
    boost1.reset();
    boost2.reset();
    for (std::size_t i = 0; i < compSize; i++) {
      bool toSet = gen();
      if (toSet) {
        fbv1.setBit(i, true);
        boost1.set(i);
      }
    }
    for (std::size_t i = 0; i < compSize; i++) {
      bool toSet = gen();
      if (toSet) {
        fbv2.setBit(i, true);
        boost2.set(i);
      }
    }
    EXPECT_EQ(fbv1 < fbv2, boost1 < boost2);
  }
}

TEST(FlatBitVect, SwapWorks) {
  nvMolKit::FlatBitVect<64> fbv1(true);
  nvMolKit::FlatBitVect<64> fbv2(false);
  std::swap(fbv1, fbv2);
  for (std::size_t i = 0; i < 64; ++i) {
    ASSERT_EQ(fbv1[i], false) << "i = " << i;
    ASSERT_EQ(fbv2[i], true) << "i = " << i;
  }
}

TEST(FlatBitVect, OperatorOrEquals) {
  nvMolKit::FlatBitVect<64> fbv1(false);
  fbv1.setBit(33, true);
  nvMolKit::FlatBitVect<64> fbv2(false);
  fbv2.setBit(33, true);
  fbv2.setBit(24, true);
  fbv1 |= fbv2;
  for (std::size_t i = 0; i < 64; ++i) {
    if (i == 24 || i == 33) {
      ASSERT_EQ(fbv1[i], true) << "i = " << i;
    } else {
      ASSERT_EQ(fbv1[i], false) << "i = " << i;
    }
  }
}

TEST(FlatBitVect, HashesAllDifferent) {
  std::unordered_set<size_t> fbvSet;
  const auto                 hasher = std::hash<nvMolKit::FlatBitVect<64>>();
  for (int i = 0; i < 64; i++) {
    nvMolKit::FlatBitVect<64> fbv(false);
    fbv.setBit(i, true);
    auto hash = hasher(fbv);

    auto result = fbvSet.insert(hash);
    ASSERT_EQ(result.second, true) << "i = " << i;
  }
}

TEST(FlatBitVect, SetInjection) {
  std::unordered_set<nvMolKit::FlatBitVect<64>> fbvSet;

  for (int i = 0; i < 64; i++) {
    nvMolKit::FlatBitVect<64> fbv(false);
    fbv.setBit(i, true);
    auto result = fbvSet.insert(fbv);
    ASSERT_EQ(result.second, true) << "i = " << i;
  }
  for (int i = 0; i < 64; i++) {
    nvMolKit::FlatBitVect<64> fbv(false);
    fbv.setBit(i, true);
    ASSERT_EQ(fbvSet.count(fbv), 1);
  }
}

TEST(FlatBitVect, FillSmallerNoCheck) {
  nvMolKit::FlatBitVect<64> fbv1(false);
  fbv1.setBit(33, true);
  fbv1.setBit(31, true);
  auto smaller = fbv1.resize<32>();
  for (std::size_t i = 0; i < 32; ++i) {
    if (i == 31) {
      ASSERT_EQ(smaller[i], true) << "i = " << i;
    } else {
      ASSERT_EQ(smaller[i], false) << "i = " << i;
    }
  }
}

TEST(FlatBitVect, FillSmallerCheck) {
  nvMolKit::FlatBitVect<64> fbv1(false);
  fbv1.setBit(33, true);
  fbv1.setBit(31, true);
  ASSERT_THROW(fbv1.resize<32>(true), std::runtime_error);
}

TEST(FlatBitVect, FillLarger) {
  nvMolKit::FlatBitVect<64> fbv1(false);
  fbv1.setBit(33, true);
  fbv1.setBit(31, true);
  auto larger = fbv1.resize<128>(false, false);
  for (std::size_t i = 0; i < 64; ++i) {
    if (i == 31 || i == 33) {
      ASSERT_EQ(larger[i], true) << "i = " << i;
    } else {
      ASSERT_EQ(larger[i], false) << "i = " << i;
    }
  }
}

TEST(FlatBitVect, FillLargerValueFalse) {
  nvMolKit::FlatBitVect<64> fbv1(false);
  fbv1.setBit(33, true);
  fbv1.setBit(31, true);
  auto larger = fbv1.resize<128>(false, true, false);
  for (std::size_t i = 0; i < 64; ++i) {
    if (i == 31 || i == 33) {
      ASSERT_EQ(larger[i], true) << "i = " << i;
    } else {
      ASSERT_EQ(larger[i], false) << "i = " << i;
    }
  }
  for (std::size_t i = 64; i < 128; ++i) {
    ASSERT_EQ(larger[i], false) << "i = " << i;
  }
}

TEST(FlatBitVect, FillLargerValueTrue) {
  nvMolKit::FlatBitVect<64> fbv1(false);
  fbv1.setBit(33, true);
  fbv1.setBit(31, true);
  auto larger = fbv1.resize<128>(true, true, true);
  for (std::size_t i = 0; i < 64; ++i) {
    if (i == 31 || i == 33) {
      ASSERT_EQ(larger[i], true) << "i = " << i;
    } else {
      ASSERT_EQ(larger[i], false) << "i = " << i;
    }
  }
  for (std::size_t i = 64; i < 128; ++i) {
    ASSERT_EQ(larger[i], true) << "i = " << i;
  }
}

TEST(FlatBitVect, ConstBitsRange) {
  nvMolKit::FlatBitVect<128> fbv1(false);
  fbv1.setBit(33, true);
  auto begin = fbv1.cbegin();
  auto end   = fbv1.cend();
  ASSERT_EQ(*begin, 0u);
  begin++;
  ASSERT_NE(*begin, 0u);
  begin++;
  ASSERT_EQ(*begin, 0u);
  begin++;
  begin++;

  ASSERT_EQ(end, begin);
}

TEST(FlatBitVect, NonConstBitsRange) {
  nvMolKit::FlatBitVect<128> fbv1(false);
  auto                       begin = fbv1.begin();
  ASSERT_EQ(*begin, 0u);
  begin++;
  *begin = 0xFFFFFFFF;
  for (int i = 0; i < 32; i++) {
    ASSERT_EQ(fbv1[i], false);
  }
  for (int i = 32; i < 64; i++) {
    ASSERT_EQ(fbv1[i], true);
  }
  for (int i = 64; i < 128; i++) {
    ASSERT_EQ(fbv1[i], false);
  }
}

TEST(FlatBitVect, NumOnBits) {
  nvMolKit::FlatBitVect<115> fbv1(false);
  ASSERT_EQ(fbv1.numOnBits(), 0u);
  fbv1.setBit(33, true);
  fbv1.setBit(31, true);
  ASSERT_EQ(fbv1.numOnBits(), 2u);
}

TEST(BitMatrix2DView, LinearIndexRowMajor) {
  using View = nvMolKit::BitMatrix2DView<3, 5>;
  EXPECT_EQ(View::linearIndex(0, 0), 0u);
  EXPECT_EQ(View::linearIndex(0, 4), 4u);
  EXPECT_EQ(View::linearIndex(1, 0), 5u);
  EXPECT_EQ(View::linearIndex(2, 3), 13u);
}

TEST(BitMatrix2DView, GetSetRoundTrip) {
  constexpr std::size_t                 rows = 3;
  constexpr std::size_t                 cols = 4;
  nvMolKit::FlatBitVect<rows * cols>    storage(false);
  nvMolKit::BitMatrix2DView<rows, cols> view(storage);

  view.set(0, 0, true);
  view.set(1, 2, true);
  view.set(2, 3, true);

  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      const bool expected = (row == 0 && col == 0) || (row == 1 && col == 2) || (row == 2 && col == 3);
      EXPECT_EQ(view.get(row, col), expected) << "row = " << row << ", col = " << col;
      EXPECT_EQ(storage[view.linearIndex(row, col)], expected) << "row = " << row << ", col = " << col;
    }
  }
}

TEST(BitMatrix2DView, ClearResetsBits) {
  constexpr std::size_t                 rows = 2;
  constexpr std::size_t                 cols = 6;
  nvMolKit::FlatBitVect<rows * cols>    storage(true);
  nvMolKit::BitMatrix2DView<rows, cols> view(storage);

  view.clear();
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      EXPECT_FALSE(view.get(row, col)) << "row = " << row << ", col = " << col;
    }
  }
}

TEST(BitMatrix2DView, RowHasAnySet) {
  constexpr std::size_t                 rows = 4;
  constexpr std::size_t                 cols = 3;
  nvMolKit::FlatBitVect<rows * cols>    storage(false);
  nvMolKit::BitMatrix2DView<rows, cols> view(storage);

  view.set(1, 0, true);
  view.set(3, 2, true);

  EXPECT_FALSE(view.rowHasAnySet(0));
  EXPECT_TRUE(view.rowHasAnySet(1));
  EXPECT_FALSE(view.rowHasAnySet(2));
  EXPECT_TRUE(view.rowHasAnySet(3));

  view.set(1, 0, false);
  EXPECT_FALSE(view.rowHasAnySet(1));
  EXPECT_TRUE(view.rowHasAnySet(3));
}

TEST(BitMatrix2DView, StoragePointer) {
  constexpr std::size_t              rows = 2;
  constexpr std::size_t              cols = 2;
  nvMolKit::FlatBitVect<rows * cols> storage(false);

  nvMolKit::BitMatrix2DView<rows, cols> view_from_ref(storage);
  EXPECT_EQ(view_from_ref.storage(), &storage);

  nvMolKit::BitMatrix2DView<rows, cols> view_from_ptr(&storage);
  EXPECT_EQ(view_from_ptr.storage(), &storage);
}
