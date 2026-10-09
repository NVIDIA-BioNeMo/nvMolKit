// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_TORSION_COMPAT_H
#define NVMOLKIT_TORSION_COMPAT_H

#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace nvMolKit::detail {

using CosineTorsionParameters = std::pair<std::vector<int>, std::vector<double>>;

inline const CosineTorsionParameters& getCosineTorsionParameters(const CosineTorsionParameters& parameters) {
  return parameters;
}

// RDKit 2026.09 wraps the existing cosine parameters and new Gaussian
// parameters in a variant. nvMolKit supports the cosine form (ETversion 1/2).
template <typename... Alternatives>
const CosineTorsionParameters& getCosineTorsionParameters(const std::variant<Alternatives...>& parameters) {
  const auto* cosine = std::get_if<CosineTorsionParameters>(&parameters);
  if (cosine == nullptr) {
    throw std::invalid_argument("Gaussian experimental torsion parameters are not supported by nvMolKit");
  }
  return *cosine;
}

}  // namespace nvMolKit::detail

#endif  // NVMOLKIT_TORSION_COMPAT_H
