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

#ifndef NVMOLKIT_OPENMP_HELPERS_H
#define NVMOLKIT_OPENMP_HELPERS_H

#include <exception>
#include <mutex>

namespace nvMolKit {
namespace detail {

//! A thread-safe registry to store and rethrow the first encountered exception from OpenMP threads
//! Usage:
//!   OpenMPExceptionRegistry exceptionRegistry;
//!   #pragma omp parallel
//!   {
//!     try {
//!       // Do work that may throw
//!     } catch (...) {
//!       exceptionRegistry.store(std::current_exception());
//!     }
//!   }
//!   exceptionRegistry.rethrow(); // Rethrows the first stored exception, if any
class OpenMPExceptionRegistry {
 public:
  void store(std::exception_ptr exceptionPtr);
  void rethrow();

 private:
  std::mutex         mutex_;
  std::exception_ptr exception_;
};

//! Resolve a caller's CPU thread count: -1 selects every available OpenMP thread; other values must be
//! positive. @throws std::invalid_argument for 0 or values below -1.
int resolveNumThreads(int requested);

}  // namespace detail
}  // namespace nvMolKit

#endif  // NVMOLKIT_OPENMP_HELPERS_H