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

#ifndef NVMOLKIT_SIMILARITY_H
#define NVMOLKIT_SIMILARITY_H

#include <DataStructs/ExplicitBitVect.h>

#include <cuda/std/span>
#include <optional>

#include "src/utils/device_vector.h"

namespace nvMolKit {

//! Compute options for cross-similarity computation
struct CrossSimilarityOptions {
  //! Overrides autodetect of maximum memory on device for chunking compute.
  std::optional<std::size_t> maxDeviceMemoryBytes = std::nullopt;
};

// --------------------------------
// Tanimoto similarity wrapper functions
// --------------------------------

AsyncDeviceVector<double> crossTanimotoSimilarityGpuResult(cuda::std::span<const std::uint32_t> bits,
                                                           int                                  fpSize,
                                                           cudaStream_t                         stream = nullptr);

AsyncDeviceVector<double> crossTanimotoSimilarityGpuResult(cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                           cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                           int                                  fpSize,
                                                           cudaStream_t                         stream = nullptr);

std::vector<double> crossTanimotoSimilarityCPUResult(cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                     cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                     int                                  fpSize,
                                                     const CrossSimilarityOptions& options = CrossSimilarityOptions(),
                                                     cudaStream_t                  stream  = nullptr);
// --------------------------------
// Cosine similarity wrapper functions
// --------------------------------

AsyncDeviceVector<double> crossCosineSimilarityGpuResult(cuda::std::span<const std::uint32_t> bits,
                                                         int                                  fpSize,
                                                         cudaStream_t                         stream = nullptr);

AsyncDeviceVector<double> crossCosineSimilarityGpuResult(cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                         cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                         int                                  fpSize,
                                                         cudaStream_t                         stream = nullptr);

std::vector<double> crossCosineSimilarityCPUResult(cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                   cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                   int                                  fpSize,
                                                   const CrossSimilarityOptions& options = CrossSimilarityOptions(),
                                                   cudaStream_t                  stream  = nullptr);

}  // namespace nvMolKit

#endif  // NVMOLKIT_SIMILARITY_H
