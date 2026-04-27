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

#ifndef NVMOLKIT_BATCHED_FORCEFIELD_H
#define NVMOLKIT_BATCHED_FORCEFIELD_H

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "src/minimizer/bfgs_types.h"

namespace nvMolKit {

//! Identifies a concrete batched system and the logical molecule/conformer it belongs to.
struct BatchedSystemInfo {
  //! Index of this system in the flattened batched arrays.
  int systemIdx    = -1;
  //! Index of the logical molecule shared by one or more systems.
  int moleculeIdx  = -1;
  //! Index of this conformer within its logical molecule.
  int conformerIdx = -1;
};

//! Stores mappings between concrete batched systems and logical molecules.
struct BatchedForcefieldMetadata {
  //! Maps each system index to the logical molecule it belongs to.
  std::vector<int>              systemToMoleculeIdx;
  //! Maps each system index to its conformer index within the logical molecule.
  std::vector<int>              systemToConformerIdx;
  //! Lists all system indices associated with each logical molecule.
  std::vector<std::vector<int>> moleculeToSystemIndices;

  //! Reserves storage for per-system metadata.
  //! \param numSystems Number of concrete systems expected in the batch.
  void reserveSystems(const int numSystems);

  //! Ensures metadata storage exists for the requested logical molecule.
  //! \param moleculeIdx Logical molecule index that will receive system entries.
  void ensureMolecule(const int moleculeIdx);

  //! Records a concrete system for a logical molecule/conformer pair.
  //! \param moleculeIdx Logical molecule index for the system.
  //! \param conformerIdx Conformer index within the logical molecule.
  //! \return The fully populated system metadata for the newly appended system.
  BatchedSystemInfo recordSystem(const int moleculeIdx, const int conformerIdx);

  //! Returns the number of concrete systems currently recorded.
  int numSystems() const;

  //! Returns the number of logical molecules currently tracked.
  int numLogicalMolecules() const;
};

//! Builds metadata for the one-system-per-molecule case.
//! \param numSystems Number of concrete systems and logical molecules.
//! \return Identity metadata where each system maps to its own molecule.
BatchedForcefieldMetadata makeIdentityBatchedForcefieldMetadata(const int numSystems);

//! Abstract base class for forcefields evaluated over a batch of molecular systems.
class BatchedForcefield {
 public:
  virtual ~BatchedForcefield();

  //! Computes energies for each concrete system in the batch.
  //! \param energyOuts Output buffer with one energy value per system.
  //! \param positions Flattened position buffer for the full batch.
  //! \param activeSystemMask Optional mask selecting which systems participate.
  //! \param stream CUDA stream used for the computation.
  //! \return CUDA status for the launch and any immediate setup work.
  virtual cudaError_t computeEnergy(double*        energyOuts,
                                    const double*  positions,
                                    const uint8_t* activeSystemMask = nullptr,
                                    cudaStream_t   stream           = nullptr) = 0;

  //! Computes gradients for each concrete system in the batch.
  //! \param grad Output gradient buffer matching the flattened position layout.
  //! \param positions Flattened position buffer for the full batch.
  //! \param activeSystemMask Optional mask selecting which systems participate.
  //! \param stream CUDA stream used for the computation.
  //! \return CUDA status for the launch and any immediate setup work.
  virtual cudaError_t computeGradients(double*        grad,
                                       const double*  positions,
                                       const uint8_t* activeSystemMask = nullptr,
                                       cudaStream_t   stream           = nullptr) = 0;

  //! Returns the number of concrete systems represented by this batch.
  int                              numMolecules() const;
  //! Returns the coordinate dimensionality stored per atom.
  int                              dataDim() const;
  //! Returns the total number of scalar coordinates in the flattened position buffer.
  int                              totalPositions() const;
  //! Returns host-side atom start offsets for each concrete system.
  const std::vector<int>&          atomStartsHost() const;
  //! Returns device-side atom start offsets for each concrete system.
  const int*                       atomStartsDevice() const;
  //! Returns the concrete forcefield implementation type.
  ForceFieldType                   type() const;
  //! Returns metadata relating concrete systems to logical molecules.
  const BatchedForcefieldMetadata& metadata() const;
  //! Returns the number of logical molecules represented in the batch.
  int                              numLogicalMolecules() const;
  //! Returns the logical molecule index for each concrete system.
  const std::vector<int>&          systemToMoleculeIdx() const;
  //! Returns the conformer index for each concrete system.
  const std::vector<int>&          systemToConformerIdx() const;
  //! Returns all concrete systems that belong to a logical molecule.
  //! \param moleculeIdx Logical molecule index to query.
  const std::vector<int>&          systemsForMolecule(const int moleculeIdx) const;

 protected:
  //! Constructs a batched forcefield view over host/device atom ranges.
  //! \param type Forcefield implementation type.
  //! \param dataDim Coordinate dimensionality per atom.
  //! \param atomStartsHost Host-side atom start offsets for each concrete system.
  //! \param atomStartsDevice Device-side atom start offsets for each concrete system.
  //! \param metadata Optional mapping from systems to logical molecules.
  BatchedForcefield(ForceFieldType            type,
                    int                       dataDim,
                    std::vector<int>          atomStartsHost,
                    const int*                atomStartsDevice,
                    BatchedForcefieldMetadata metadata = {});

  //! Updates the device atom-start pointer after device buffers are initialized.
  //! \param atomStartsDevice Device pointer to system atom-start offsets.
  void setAtomStartsDevice(const int* atomStartsDevice);

 private:
  int                       numMolecules_   = 0;
  int                       dataDim_        = 0;
  int                       totalPositions_ = 0;
  std::vector<int>          atomStartsHost_;
  const int*                atomStartsDevice_ = nullptr;
  BatchedForcefieldMetadata metadata_;
  ForceFieldType            type_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_BATCHED_FORCEFIELD_H
