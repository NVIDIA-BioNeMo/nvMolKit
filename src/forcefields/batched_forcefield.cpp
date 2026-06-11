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

#include "src/forcefields/batched_forcefield.h"

#include <utility>

namespace nvMolKit {

void BatchedForcefieldMetadata::reserveSystems(const int numSystems) {
  systemToMoleculeIdx.reserve(numSystems);
  systemToConformerIdx.reserve(numSystems);
}

void BatchedForcefieldMetadata::ensureMolecule(const int moleculeIdx) {
  if (moleculeIdx >= static_cast<int>(moleculeToSystemIndices.size())) {
    moleculeToSystemIndices.resize(moleculeIdx + 1);
  }
}

BatchedSystemInfo BatchedForcefieldMetadata::recordSystem(const int moleculeIdx, const int conformerIdx) {
  const int systemIdx = static_cast<int>(systemToMoleculeIdx.size());
  ensureMolecule(moleculeIdx);
  systemToMoleculeIdx.push_back(moleculeIdx);
  systemToConformerIdx.push_back(conformerIdx);
  moleculeToSystemIndices[moleculeIdx].push_back(systemIdx);

  BatchedSystemInfo info;
  info.systemIdx    = systemIdx;
  info.moleculeIdx  = moleculeIdx;
  info.conformerIdx = conformerIdx;
  return info;
}

int BatchedForcefieldMetadata::numSystems() const {
  return static_cast<int>(systemToMoleculeIdx.size());
}

int BatchedForcefieldMetadata::numLogicalMolecules() const {
  return static_cast<int>(moleculeToSystemIndices.size());
}

BatchedForcefieldMetadata makeIdentityBatchedForcefieldMetadata(const int numSystems) {
  BatchedForcefieldMetadata metadata;
  metadata.reserveSystems(numSystems);
  metadata.moleculeToSystemIndices.resize(numSystems);
  for (int systemIdx = 0; systemIdx < numSystems; ++systemIdx) {
    metadata.systemToMoleculeIdx.push_back(systemIdx);
    metadata.systemToConformerIdx.push_back(0);
    metadata.moleculeToSystemIndices[systemIdx].push_back(systemIdx);
  }
  return metadata;
}

BatchedForcefield::~BatchedForcefield() = default;

BatchedForcefield::BatchedForcefield(ForceFieldType            type,
                                     int                       dataDim,
                                     std::vector<int>          atomStartsHost,
                                     const int*                atomStartsDevice,
                                     BatchedForcefieldMetadata metadata)
    : numMolecules_(static_cast<int>(atomStartsHost.size()) - 1),
      dataDim_(dataDim),
      totalPositions_(atomStartsHost.empty() ? 0 : atomStartsHost.back() * dataDim),
      atomStartsHost_(std::move(atomStartsHost)),
      atomStartsDevice_(atomStartsDevice),
      metadata_(metadata.numSystems() == 0 ? makeIdentityBatchedForcefieldMetadata(numMolecules_) :
                                             std::move(metadata)),
      type_(type) {}

int BatchedForcefield::numMolecules() const {
  return numMolecules_;
}

int BatchedForcefield::dataDim() const {
  return dataDim_;
}

int BatchedForcefield::totalPositions() const {
  return totalPositions_;
}

const std::vector<int>& BatchedForcefield::atomStartsHost() const {
  return atomStartsHost_;
}

const int* BatchedForcefield::atomStartsDevice() const {
  return atomStartsDevice_;
}

ForceFieldType BatchedForcefield::type() const {
  return type_;
}

const BatchedForcefieldMetadata& BatchedForcefield::metadata() const {
  return metadata_;
}

int BatchedForcefield::numLogicalMolecules() const {
  return metadata_.numLogicalMolecules();
}

const std::vector<int>& BatchedForcefield::systemToMoleculeIdx() const {
  return metadata_.systemToMoleculeIdx;
}

const std::vector<int>& BatchedForcefield::systemToConformerIdx() const {
  return metadata_.systemToConformerIdx;
}

const std::vector<int>& BatchedForcefield::systemsForMolecule(const int moleculeIdx) const {
  return metadata_.moleculeToSystemIndices[moleculeIdx];
}

void BatchedForcefield::setAtomStartsDevice(const int* atomStartsDevice) {
  atomStartsDevice_ = atomStartsDevice;
}

}  // namespace nvMolKit
