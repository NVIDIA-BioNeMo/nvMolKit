// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NVMOLKIT_PRECISION_PRECISION_MODE_H
#define NVMOLKIT_PRECISION_PRECISION_MODE_H

namespace nvMolKit {

//! Precision modes supported by minimizers and force fields.
enum class PrecisionMode {
  FULL = 0,  //!< Double-precision coordinates, arithmetic, state, and reductions.
  SINGLE     //!< Single-precision device storage, arithmetic, and reductions.
};

inline const char* precisionModeName(const PrecisionMode precision) {
  return precision == PrecisionMode::SINGLE ? "SINGLE" : "FULL";
}

inline bool usesSinglePrecision(const PrecisionMode precision) {
  return precision == PrecisionMode::SINGLE;
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_PRECISION_PRECISION_MODE_H
