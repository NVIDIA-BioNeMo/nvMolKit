// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "src/utils/openmp_helpers.h"

#include <omp.h>

#include <stdexcept>
#include <string>

namespace nvMolKit {
namespace detail {

void OpenMPExceptionRegistry::store(std::exception_ptr exceptionPtr) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!exception_) {
    exception_ = std::move(exceptionPtr);
  }
}

void OpenMPExceptionRegistry::rethrow() {
  std::exception_ptr toThrow;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    toThrow    = exception_;
    exception_ = nullptr;
  }

  if (toThrow) {
    std::rethrow_exception(toThrow);
  }
}

int resolveNumThreads(const int requested) {
  if (requested == -1) {
    return omp_get_max_threads();
  }
  if (requested < 1) {
    throw std::invalid_argument("Thread count must be positive or -1 for all threads, got " +
                                std::to_string(requested));
  }
  return requested;
}

}  // namespace detail
}  // namespace nvMolKit