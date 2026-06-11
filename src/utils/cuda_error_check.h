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

#ifndef NVMOLKIT_CUDA_ERROR_CHECK_H
#define NVMOLKIT_CUDA_ERROR_CHECK_H

#include <cstdio>

#include "cuda_runtime.h"
#include "src/utils/exceptions.h"

//! Checks a CUDA return code, throwing a CudaBadReturnCode error
//! if nonzero, and printing out line info if in debug mode.
//! Example usage:
//!     cudaCheckError(cudaMemcpyAsync(dest, src, sizeof(double), cudaMemcpyDefault));
#define cudaCheckError(ans) \
  { checkReturnCode<true>((ans), __FILE__, __LINE__); }

//! Checks a CUDA return code, but does not throw an error if nonzero.
//! Useful for debugging where we can't or don't want to throw exceptions - fast OpenMP regions and destructors.
#define cudaCheckErrorNoThrow(ans) \
  { checkReturnCode<false>((ans), __FILE__, __LINE__); }

namespace nvMolKit {

template <bool throw_on_error>
inline void checkReturnCode(cudaError_t code, [[maybe_unused]] const char* file, [[maybe_unused]] int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "Bad CUDA return code: %s %s %d\n", cudaGetErrorString(code), file, line);
    if constexpr (throw_on_error) {
      throw CudaBadReturnCode(code);
    }
  }
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_CUDA_ERROR_CHECK_H
