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

#ifndef NVMOLKIT_BFGS_COMMON_H
#define NVMOLKIT_BFGS_COMMON_H

#include <cstdint>
#include <vector>

#include "src/conformer/conformer_info.h"
#include "src/hardware_options.h"
#include "src/utils/device.h"
#include "src/utils/host_vector.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

//! Thread-local pinned memory buffers for async host-device transfers during BFGS minimization.
struct ThreadLocalBuffers {
  PinnedHostVector<double> positions;
  PinnedHostVector<double> energies;
  PinnedHostVector<double> initialPositions;

  void ensureCapacity(size_t positionsSize, size_t energiesSize);
};

struct BatchExecutionContext {
  size_t                    batchSize;
  int                       numThreads;
  std::vector<ScopedStream> streamPool;
  std::vector<int>          devicesPerThread;
};

/// Set up GPU streams, thread counts, and batch sizing from hardware options.
BatchExecutionContext setupBatchExecution(const BatchHardwareOptions& perfOptions);

/// Flatten all conformers from all molecules into a single list, validating molecule pointers
/// and initializing the per-molecule energy output vectors.
std::vector<ConformerInfo> flattenConformers(const std::vector<RDKit::ROMol*>& mols,
                                             std::vector<std::vector<double>>& moleculeEnergies);

/// Write optimized positions and energies back from host buffers into the RDKit conformers.
void writeBackResults(const std::vector<ConformerInfo>& batchConformers,
                      const std::vector<uint32_t>&      conformerAtomStarts,
                      const ThreadLocalBuffers&         buffers,
                      std::vector<std::vector<double>>& moleculeEnergies);

}  // namespace nvMolKit

#endif  // NVMOLKIT_BFGS_COMMON_H
