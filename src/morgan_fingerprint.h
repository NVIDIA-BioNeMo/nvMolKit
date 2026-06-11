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

#ifndef NVMOLKIT_MORGAN_FINGERPRINT_H
#define NVMOLKIT_MORGAN_FINGERPRINT_H

#include <DataStructs/ExplicitBitVect.h>
#include <GraphMol/Fingerprints/FingerprintGenerator.h>
#include <GraphMol/Fingerprints/MorganGenerator.h>

#include "src/data_structures/flat_bit_vect.h"
#include "src/morgan_fingerprint_common.h"
#include "src/morgan_fingerprint_cpu.h"
#include "src/morgan_fingerprint_gpu.h"

namespace nvMolKit {

//! Computes Morgan fingerprints
class MorganFingerprintGenerator {
 public:
  MorganFingerprintGenerator(std::uint32_t radius, std::uint32_t fpSize = 2048);
  //! Compute a single fingerprint and convert to RDKit structure
  std::unique_ptr<ExplicitBitVect> GetFingerprint(
    const RDKit::ROMol&                      mol,
    std::optional<FingerprintComputeOptions> computeOptions = std::nullopt);
  //! Compute a batch of fingerprints and convert to RDKit structures
  std::vector<std::unique_ptr<ExplicitBitVect>> GetFingerprints(
    const std::vector<const RDKit::ROMol*>&  mols,
    std::optional<FingerprintComputeOptions> computeOptions = std::nullopt);
  //! Compute a batch of fingerprints and keep results on GPU.
  //! Note that this method overrides fpSize.
  //! @param stream CUDA stream to use for output allocation and ordering.
  template <int nBits>
  AsyncDeviceVector<FlatBitVect<nBits>> GetFingerprintsGpuBuffer(
    const std::vector<const RDKit::ROMol*>&  mols,
    cudaStream_t                             stream         = nullptr,
    std::optional<FingerprintComputeOptions> computeOptions = std::nullopt);

  //! Return the current options
  const MorganFingerprintOptions& GetOptions() const { return options_; }

 private:
  void initializeBackendIfNeeded(FingerprintComputeBackend backend);

  std::unique_ptr<MorganFingerprintGpuGenerator> gpuGenerator_ = nullptr;
  std::unique_ptr<MorganFingerprintCpuGenerator> cpuGenerator_ = nullptr;
  MorganFingerprintOptions                       options_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_MORGAN_FINGERPRINT_H
