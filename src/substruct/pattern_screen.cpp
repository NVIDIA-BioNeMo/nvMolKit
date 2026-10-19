// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/substruct/pattern_screen.h"

#include <DataStructs/ExplicitBitVect.h>

#include <algorithm>
#include <bit>
#include <boost/dynamic_bitset.hpp>
#include <iterator>
#include <stdexcept>

namespace nvMolKit {

namespace {

void fingerprintWords(const ExplicitBitVect& fingerprint, std::uint64_t (&words)[kPatternFingerprintWords]) {
  static_assert(sizeof(boost::dynamic_bitset<>::block_type) == sizeof(std::uint64_t));
  if (fingerprint.getNumBits() != kPatternFingerprintBits) {
    throw std::invalid_argument("Pattern fingerprint screening requires 2048-bit fingerprints");
  }
  boost::to_block_range(*fingerprint.dp_bits, std::begin(words));
}

}  // namespace

std::vector<std::uint64_t> packPatternFingerprintsWordMajor(const std::vector<const ExplicitBitVect*>& fingerprints) {
  const std::size_t          count = fingerprints.size();
  std::vector<std::uint64_t> packed(count * kPatternFingerprintWords);
  std::uint64_t              words[kPatternFingerprintWords];
  for (std::size_t index = 0; index < count; ++index) {
    if (fingerprints[index] == nullptr) {
      throw std::invalid_argument("Missing pattern fingerprint for a packed target");
    }
    fingerprintWords(*fingerprints[index], words);
    for (int word = 0; word < kPatternFingerprintWords; ++word) {
      packed[static_cast<std::size_t>(word) * count + index] = words[word];
    }
  }
  return packed;
}

std::vector<std::uint32_t> buildPatternBitSlices(const std::vector<std::uint64_t>& wordMajor, std::size_t count) {
  if (wordMajor.size() != count * kPatternFingerprintWords) {
    throw std::invalid_argument("Word-major fingerprints do not match the target count");
  }
  const std::size_t          sliceWords = patternSliceWords(count);
  std::vector<std::uint32_t> slices(static_cast<std::size_t>(kPatternFingerprintBits) * sliceWords, 0);
  // Each 32-target block owns one word per slice, so blocks fill independently.
#pragma omp parallel for schedule(static)
  for (std::int64_t block = 0; block < static_cast<std::int64_t>(sliceWords); ++block) {
    const std::size_t first = static_cast<std::size_t>(block) * 32;
    const std::size_t last  = std::min(count, first + 32);
    for (int word = 0; word < kPatternFingerprintWords; ++word) {
      const std::uint64_t* column = wordMajor.data() + static_cast<std::size_t>(word) * count;
      for (std::size_t target = first; target < last; ++target) {
        std::uint64_t bits = column[target];
        while (bits != 0) {
          const int         bit   = word * 64 + std::countr_zero(bits);
          const std::size_t index = static_cast<std::size_t>(bit) * sliceWords + static_cast<std::size_t>(block);
          slices[index] |= std::uint32_t{1} << (target - first);
          bits &= bits - 1;
        }
      }
    }
  }
  return slices;
}

std::vector<std::uint32_t> patternBitFrequencies(const std::vector<std::uint32_t>& slices, std::size_t count) {
  const std::size_t          sliceWords = patternSliceWords(count);
  std::vector<std::uint32_t> frequencies(kPatternFingerprintBits, 0);
#pragma omp parallel for schedule(static)
  for (int bit = 0; bit < kPatternFingerprintBits; ++bit) {
    std::uint32_t total = 0;
    for (std::size_t word = 0; word < sliceWords; ++word) {
      total += static_cast<std::uint32_t>(std::popcount(slices[static_cast<std::size_t>(bit) * sliceWords + word]));
    }
    frequencies[static_cast<std::size_t>(bit)] = total;
  }
  return frequencies;
}

PatternScreenQuery makePatternScreenQuery(const ExplicitBitVect* fingerprint, int numQueryAtoms) {
  PatternScreenQuery query;
  query.numAtoms = numQueryAtoms;
  if (fingerprint == nullptr) {
    return query;
  }
  std::uint64_t words[kPatternFingerprintWords];
  fingerprintWords(*fingerprint, words);
  for (int word = 0; word < kPatternFingerprintWords; ++word) {
    std::uint64_t bits = words[word];
    while (bits != 0) {
      query.bits.push_back(static_cast<std::uint16_t>(word * 64 + std::countr_zero(bits)));
      bits &= bits - 1;
    }
  }
  return query;
}

void orderPatternScreenBits(PatternScreenQuery& query, const std::vector<std::uint32_t>& bitFrequencies) {
  std::stable_sort(query.bits.begin(), query.bits.end(), [&](std::uint16_t lhs, std::uint16_t rhs) {
    return bitFrequencies[lhs] < bitFrequencies[rhs];
  });
}

}  // namespace nvMolKit
