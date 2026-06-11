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

#ifndef NVMOLKIT_MORGAN_FINGERPRINT_KERNELS_H
#define NVMOLKIT_MORGAN_FINGERPRINT_KERNELS_H

#include <memory>

#include "src/data_structures/flat_bit_vect.h"
#include "src/morgan_fingerprint_common.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"

namespace nvMolKit {

//! All GPU buffers for Morgan fingerprint computation
struct MorganGPUBuffersBatch {
  AsyncDeviceVector<std::uint32_t> atomInvariants;        // Size = nMolecules * maxAtoms
  AsyncDeviceVector<std::uint32_t> bondInvariants;        // Size = nMolecules *  maxAtoms
  AsyncDeviceVector<std::int16_t>  bondIndices;           // Size = nMolecules *  maxAtoms * maxNumBonds
  AsyncDeviceVector<std::int16_t>  bondOtherAtomIndices;  // Size =nMolecules *  maxAtoms * maxNumBonds
  AsyncDeviceVector<std::int16_t>  nAtomsPerMol;          // Size = nMolecules

  AsyncDeviceVector<int> outputIndices;  // Size = nMolecules

  AsyncDeviceVector<FlatBitVect<32>>  allSeenNeighborhoods32;   // Size = nMolecules * 32 * (maxRadius + 1)
  AsyncDeviceVector<FlatBitVect<64>>  allSeenNeighborhoods64;   // Size = nMolecules * 32 * (maxRadius + 1)
  AsyncDeviceVector<FlatBitVect<128>> allSeenNeighborhoods128;  // Size = nMolecules * 32 * (maxRadius + 1)
};

// Per CPU Thread buffers, including GPU buffers and synchronization structures.
struct MorganPerThreadBuffers {
  PinnedHostVector<std::int16_t>         nAtomsPerMol;
  ScopedStream                           stream;
  std::unique_ptr<MorganGPUBuffersBatch> gpuBuffers32;
  std::unique_ptr<MorganGPUBuffersBatch> gpuBuffers64;
  std::unique_ptr<MorganGPUBuffersBatch> gpuBuffers128;
  ScopedCudaEvent                        prevMemcpyDoneEvent;

  // Pre-allocated pinned host buffers for CPU->GPU transfers (avoid reallocations)
  PinnedHostVector<std::uint32_t> h_atomInvariants32;
  PinnedHostVector<std::uint32_t> h_bondInvariants32;
  PinnedHostVector<std::int16_t>  h_bondIndices32;
  PinnedHostVector<std::int16_t>  h_bondOtherAtomIndices32;

  PinnedHostVector<std::uint32_t> h_atomInvariants64;
  PinnedHostVector<std::uint32_t> h_bondInvariants64;
  PinnedHostVector<std::int16_t>  h_bondIndices64;
  PinnedHostVector<std::int16_t>  h_bondOtherAtomIndices64;

  PinnedHostVector<std::uint32_t> h_atomInvariants128;
  PinnedHostVector<std::uint32_t> h_bondInvariants128;
  PinnedHostVector<std::int16_t>  h_bondIndices128;
  PinnedHostVector<std::int16_t>  h_bondOtherAtomIndices128;

  // Output indices for kernel results routing
  PinnedHostVector<int> h_outputIndices;
  ~MorganPerThreadBuffers() noexcept {
    // Reset GPU buffers first, because they may depend on the stream
    gpuBuffers32.reset();
    gpuBuffers64.reset();
    gpuBuffers128.reset();
  }
  MorganPerThreadBuffers() {
    gpuBuffers32  = std::make_unique<MorganGPUBuffersBatch>();
    gpuBuffers64  = std::make_unique<MorganGPUBuffersBatch>();
    gpuBuffers128 = std::make_unique<MorganGPUBuffersBatch>();
  }

  MorganPerThreadBuffers(const MorganPerThreadBuffers&)             = delete;
  MorganPerThreadBuffers& operator=(const MorganPerThreadBuffers&)  = delete;
  MorganPerThreadBuffers(MorganPerThreadBuffers&& other)            = default;
  MorganPerThreadBuffers& operator=(MorganPerThreadBuffers&& other) = default;
};

template <int fpSize>
void launchMorganFingerprintKernelBatch(const MorganGPUBuffersBatch&            buffers,
                                        AsyncDeviceVector<FlatBitVect<fpSize>>& outputAccumulator,
                                        size_t                                  maxRadius,
                                        int                                     maxAtoms,
                                        int                                     nMolecules = 0,
                                        cudaStream_t                            stream     = nullptr);

}  // namespace nvMolKit

#define DEFINE_EXTERN_TEMPLATE(fpSize)                                       \
  extern template void nvMolKit::launchMorganFingerprintKernelBatch<fpSize>( \
    const nvMolKit::MorganGPUBuffersBatch&  buffers,                         \
    AsyncDeviceVector<FlatBitVect<fpSize>>& outputAccumulator,               \
    size_t                                  maxRadius,                       \
    int                                     maxAtoms,                        \
    int                                     nMolecules,                      \
    cudaStream_t                            stream);
DEFINE_EXTERN_TEMPLATE(128)
DEFINE_EXTERN_TEMPLATE(256)
DEFINE_EXTERN_TEMPLATE(512)
DEFINE_EXTERN_TEMPLATE(1024)
DEFINE_EXTERN_TEMPLATE(2048)
DEFINE_EXTERN_TEMPLATE(4096)

#endif  // NVMOLKIT_MORGAN_FINGERPRINT_KERNELS_H
