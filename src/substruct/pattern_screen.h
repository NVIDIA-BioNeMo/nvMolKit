// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_PATTERN_SCREEN_H
#define NVMOLKIT_PATTERN_SCREEN_H

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/utils/device.h"
#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"

class ExplicitBitVect;

namespace nvMolKit {

//! RDKit's default pattern fingerprint is 2048 bits.
constexpr int kPatternFingerprintBits  = 2048;
constexpr int kPatternFingerprintWords = kPatternFingerprintBits / 64;

/**
 * Pack pattern fingerprints word-major: word w of fingerprint i is stored at
 * [w * fingerprints.size() + i].
 */
std::vector<std::uint64_t> packPatternFingerprintsWordMajor(const std::vector<const ExplicitBitVect*>& fingerprints);

//! Number of 32-target words in each bit slice of a set of count targets.
inline std::size_t patternSliceWords(std::size_t count) {
  return (count + 31) / 32;
}

/**
 * Transpose word-major fingerprints of count targets into bit slices: slice b
 * is a bitmap over targets, stored at [b * patternSliceWords(count)], with
 * target t at bit t % 32 of word t / 32.
 */
std::vector<std::uint32_t> buildPatternBitSlices(const std::vector<std::uint64_t>& wordMajor, std::size_t count);

//! Number of targets carrying each fingerprint bit.
std::vector<std::uint32_t> patternBitFrequencies(const std::vector<std::uint32_t>& slices, std::size_t count);

//! Query-side screen parameters.
struct PatternScreenQuery {
  std::vector<std::uint16_t> bits;  //!< Query fingerprint bits, tested in this order.
  int                        numAtoms = 0;
};

/**
 * Build screen parameters for a query. A null fingerprint screens by atom count only.
 */
PatternScreenQuery makePatternScreenQuery(const ExplicitBitVect* fingerprint, int numQueryAtoms);

/** Order query bits rarest first so most targets are rejected after a few slices. */
void orderPatternScreenBits(PatternScreenQuery& query, const std::vector<std::uint32_t>& bitFrequencies);

/**
 * Per-query device and pinned host storage for screening a persistent target
 * set on one GPU. Screening is substructure-safe: a target is rejected only
 * when it has fewer atoms than the query or lacks a query pattern-fingerprint bit.
 */
class PatternScreenWorkspace {
 public:
  /** Create a workspace on deviceId, which must be the current device. */
  explicit PatternScreenWorkspace(int deviceId);

  PatternScreenWorkspace(const PatternScreenWorkspace&)            = delete;
  PatternScreenWorkspace& operator=(const PatternScreenWorkspace&) = delete;

  /**
   * Screen numTargets targets and wait for the result. bitSlices may be null
   * to screen by atom count only. Afterwards count() and indices() hold the
   * selected target indices in ascending order.
   */
  void screen(const std::uint32_t*      bitSlices,
              const int*                batchAtomStarts,
              int                       numTargets,
              const PatternScreenQuery& query);

  /** Device bytes one workspace allocates to screen numTargets targets. */
  [[nodiscard]] static std::size_t estimateDeviceBytes(std::size_t numTargets);

  [[nodiscard]] int        count() const { return hostCount_[0]; }
  [[nodiscard]] const int* indices() const { return hostIndices_.data(); }
  [[nodiscard]] int        deviceId() const noexcept { return deviceId_; }

 private:
  void reserve(std::size_t numTargets, std::size_t numQueryBits);

  int                              deviceId_;
  ScopedStream                     stream_;
  AsyncDeviceVector<int>           indices_;
  AsyncDeviceVector<int>           count_;
  AsyncDeviceVector<std::uint32_t> survivors_;
  AsyncDeviceVector<std::uint16_t> queryBits_;
  AsyncDeviceVector<std::uint8_t>  tempStorage_;
  std::size_t                      tempTargets_ = 0;
  PinnedHostVector<int>            hostCount_;
  PinnedHostVector<int>            hostIndices_;
  PinnedHostVector<std::uint16_t>  hostQueryBits_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_PATTERN_SCREEN_H
