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

#ifndef NVMOLKIT_CCCL_COMPAT_H
#define NVMOLKIT_CCCL_COMPAT_H

// Shims for CCCL APIs newer than the oldest supported CUDA toolkit (12.6, CCCL 2.5). Each shim forwards to the
// CCCL implementation when it is available and otherwise provides an equivalent with the same contract, so call
// sites can switch to the cuda:: name once the minimum toolkit provides it.

#include <type_traits>

#if __has_include(<cuda/std/version>)
#include <cuda/std/version>
#endif

// cuda::ceil_div exists from CCCL 2.7, but only CCCL 3.0+ accepts mixed operand types and returns their common type;
// 2.7 requires identical types and 2.8 returns the type of the dividend. Forward only to the 3.0+ contract.
#if defined(CCCL_VERSION) && CCCL_VERSION >= 3000000 && __has_include(<cuda/cmath>)
#include <cuda/cmath>
#define NVMOLKIT_HAS_CUDA_CEIL_DIV 1
#else
#define NVMOLKIT_HAS_CUDA_CEIL_DIV 0
#endif

#ifdef __CUDACC__
#define NVMOLKIT_COMPAT_HD __host__ __device__
#else
#define NVMOLKIT_COMPAT_HD
#endif

namespace nvMolKit {
namespace compat {

namespace detail {

//! Fallback for cuda::ceil_div with its CCCL 3.x contract: @p a >= 0, @p b > 0, result in the common type, no
//! intermediate overflow.
template <typename T, typename U>
NVMOLKIT_COMPAT_HD constexpr std::common_type_t<T, U> ceilDivFallback(const T a, const U b) noexcept {
  static_assert(std::is_integral_v<T> && std::is_integral_v<U>, "ceil_div requires integral operands");
  using Unsigned          = std::make_unsigned_t<decltype(T{} / U{})>;
  const Unsigned dividend = static_cast<Unsigned>(a);
  const Unsigned divisor  = static_cast<Unsigned>(b);
  return static_cast<std::common_type_t<T, U>>(dividend / divisor + (dividend % divisor != 0 ? 1 : 0));
}

}  // namespace detail

#if NVMOLKIT_HAS_CUDA_CEIL_DIV
using ::cuda::ceil_div;
#else
//! Smallest integer not less than @p a / @p b. Requires @p a >= 0 and @p b > 0.
template <typename T, typename U>
NVMOLKIT_COMPAT_HD constexpr std::common_type_t<T, U> ceil_div(const T a, const U b) noexcept {
  return detail::ceilDivFallback(a, b);
}
#endif

}  // namespace compat
}  // namespace nvMolKit

#undef NVMOLKIT_COMPAT_HD

#endif  // NVMOLKIT_CCCL_COMPAT_H
