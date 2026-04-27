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

#include "src/morgan_fingerprint.h"

#include <DataStructs/ExplicitBitVect.h>

#include <boost/dynamic_bitset.hpp>
#include <set>

#include "src/utils/device.h"

namespace nvMolKit {

MorganFingerprintGenerator::MorganFingerprintGenerator(const std::uint32_t radius, const std::uint32_t fpSize) {
  options_.radius = radius;
  options_.fpSize = fpSize;
}

void MorganFingerprintGenerator::initializeBackendIfNeeded(FingerprintComputeBackend backend) {
  if (backend == FingerprintComputeBackend::GPU) {
    if (gpuGenerator_ == nullptr) {
      gpuGenerator_ = std::make_unique<MorganFingerprintGpuGenerator>(options_.radius, options_.fpSize);
    }
  } else {
    if (cpuGenerator_ == nullptr) {
      cpuGenerator_ = std::make_unique<MorganFingerprintCpuGenerator>(options_.radius, options_.fpSize);
    }
  }
}

std::unique_ptr<ExplicitBitVect> MorganFingerprintGenerator::GetFingerprint(
  const RDKit::ROMol&                      mol,
  std::optional<FingerprintComputeOptions> computeOptions) {
  const FingerprintComputeOptions options = computeOptions.value_or(FingerprintComputeOptions());
  initializeBackendIfNeeded(options.backend);
  if (options.backend == FingerprintComputeBackend::GPU) {
    return gpuGenerator_->GetFingerprint(mol);
  }
  return cpuGenerator_->GetFingerprint(mol);
}

std::vector<std::unique_ptr<ExplicitBitVect>> MorganFingerprintGenerator::GetFingerprints(
  const std::vector<const RDKit::ROMol*>&  mols,
  std::optional<FingerprintComputeOptions> computeOptions) {
  const FingerprintComputeOptions options = computeOptions.value_or(FingerprintComputeOptions());
  initializeBackendIfNeeded(options.backend);
  if (options.backend == FingerprintComputeBackend::GPU) {
    return gpuGenerator_->GetFingerprints(mols, options);
  }
  return cpuGenerator_->GetFingerprints(mols, options);
}

template <int nBits>
AsyncDeviceVector<FlatBitVect<nBits>> MorganFingerprintGenerator::GetFingerprintsGpuBuffer(
  const std::vector<const RDKit::ROMol*>&  mols,
  cudaStream_t                             stream,
  std::optional<FingerprintComputeOptions> options) {
  const FingerprintComputeOptions computeOptions = options.value_or(FingerprintComputeOptions());
  initializeBackendIfNeeded(computeOptions.backend);
  return gpuGenerator_->GetFingerprintsGpuBuffer<nBits>(mols, stream, options);
}

#define DEFINE_TEMPLATE(fpSize)                                                                                   \
  template AsyncDeviceVector<FlatBitVect<(fpSize)>> MorganFingerprintGenerator::GetFingerprintsGpuBuffer<fpSize>( \
    const std::vector<const RDKit::ROMol*>&  mols,                                                                \
    cudaStream_t                             stream,                                                              \
    std::optional<FingerprintComputeOptions> options);
DEFINE_TEMPLATE(128)
DEFINE_TEMPLATE(256)
DEFINE_TEMPLATE(512)
DEFINE_TEMPLATE(1024)
DEFINE_TEMPLATE(2048)
DEFINE_TEMPLATE(4096)
#undef DEFINE_TEMPLATE

}  // namespace nvMolKit
