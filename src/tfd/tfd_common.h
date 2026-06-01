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

#ifndef NVMOLKIT_TFD_COMMON_H
#define NVMOLKIT_TFD_COMMON_H

#include <GraphMol/ROMol.h>

#include <cstdint>
#include <vector>

#include "src/tfd/tfd_types.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

//! Maximum deviation mode for torsion normalization
enum class TFDMaxDevMode {
  Equal,  //!< All torsions normalized by 180.0
  Spec,   //!< Each torsion normalized by its specific max deviation
};

//! A single torsion definition: four atom indices and normalization factor
struct TorsionDef {
  //! Atom indices defining the torsion (a-b-c-d dihedral)
  //! For symmetric cases, multiple quartets may define equivalent torsions
  std::vector<std::array<int, 4>> atomQuartets;
  //! Maximum deviation for normalization (180.0 for equal mode, or specific value)
  float                           maxDev = 180.0f;
};

//! Torsion list for a single molecule (non-ring and ring torsions)
struct TorsionList {
  std::vector<TorsionDef> nonRingTorsions;
  std::vector<TorsionDef> ringTorsions;

  //! Total number of torsions (non-ring + ring)
  size_t totalCount() const { return nonRingTorsions.size() + ringTorsions.size(); }
};

//! Performance and algorithm options for TFD computation
struct TFDComputeOptions {
  //! Whether to use distance-based weights for torsions
  bool          useWeights          = true;
  //! Maximum deviation mode
  TFDMaxDevMode maxDevMode          = TFDMaxDevMode::Equal;
  //! Radius for Morgan fingerprint atom invariants (for symmetry detection)
  int           symmRadius          = 2;
  //! Whether to ignore single bonds adjacent to triple bonds
  bool          ignoreColinearBonds = true;
};

//! Flattened system data on host for a batch of molecules
struct TFDSystemHost {
  //! Flattened 3D coordinates, tightly packed (no padding)
  //! Stored as: conf0_atom0_xyz, conf0_atom1_xyz, ..., conf1_atom0_xyz, ...
  std::vector<float> positions;

  //! Position start offset per conformer [totalConformers]
  //! confPositionStarts[i] = float offset into positions for conformer i
  std::vector<int> confPositionStarts;

  //! Flattened torsion atom indices: [totalQuartets][4]
  //! Multiple quartets per torsion (when torsionTypes[t] is Ring or Symmetric) are stored contiguously.
  std::vector<std::array<int, 4>> torsionAtoms;

  //! Weight per torsion [totalTorsions]
  std::vector<float> torsionWeights;

  //! Maximum deviation per torsion [totalTorsions]
  std::vector<float> torsionMaxDevs;

  //! Type per torsion [totalTorsions] - Single, Ring, or Symmetric
  std::vector<TorsionType> torsionTypes;

  //! CSR index: quartet boundaries per torsion [totalTorsions + 1]
  //! quartetStarts[t] to quartetStarts[t+1] are indices into torsionAtoms for torsion t
  std::vector<int> quartetStarts = {0};

  // ========== Per-molecule descriptors for GPU kernel dispatch ==========

  //! One descriptor per molecule with all offsets needed by both kernels
  std::vector<MolDescriptor> molDescriptors;

  //! CSR: cumulative dihedral work items per molecule [nMols + 1]
  //! dihedralWorkStarts[m+1] - dihedralWorkStarts[m] = numConformers[m] * numQuartets[m]
  std::vector<int> dihedralWorkStarts = {0};

  //! CSR: cumulative TFD pair work items per molecule [nMols + 1]
  //! tfdWorkStarts[m+1] - tfdWorkStarts[m] = C[m] * (C[m]-1) / 2
  std::vector<int> tfdWorkStarts = {0};

  //! Total number of dihedral angle values (numConformers * totalQuartets across all molecules)
  int totalDihedrals_ = 0;

  //! Total number of molecules
  int numMolecules() const { return static_cast<int>(molDescriptors.size()); }

  //! Total number of torsions across all molecules
  int totalTorsions() const { return quartetStarts.empty() ? 0 : static_cast<int>(quartetStarts.size()) - 1; }

  //! Total number of quartets across all molecules
  int totalQuartets() const { return quartetStarts.empty() ? 0 : quartetStarts.back(); }

  //! Total number of TFD output values
  int totalTFDOutputs() const { return tfdWorkStarts.empty() ? 0 : tfdWorkStarts.back(); }

  //! Total number of dihedral angle values to store
  int totalDihedrals() const { return totalDihedrals_; }

  //! Total number of dihedral work items (for dihedral kernel launch)
  int totalDihedralWorkItems() const { return dihedralWorkStarts.empty() ? 0 : dihedralWorkStarts.back(); }
};

//! Flattened system data on GPU device
//! Mirrors TFDSystemHost for device-side storage.
struct TFDSystemDevice {
  // ========== Shared kernel inputs ==========

  //! Flattened 3D coordinates (tightly packed, no padding)
  AsyncDeviceVector<float> positions;
  //! Position start offset per conformer [totalConformers]
  AsyncDeviceVector<int>   confPositionStarts;
  //! Flattened torsion atom indices [totalQuartets * 4]
  AsyncDeviceVector<int>   torsionAtoms;

  //! Weight per torsion
  AsyncDeviceVector<float>   torsionWeights;
  //! Maximum deviation per torsion
  AsyncDeviceVector<float>   torsionMaxDevs;
  //! CSR index: quartet boundaries per torsion [totalTorsions + 1]
  AsyncDeviceVector<int>     quartetStarts;
  //! Type per torsion [totalTorsions]
  AsyncDeviceVector<uint8_t> torsionTypes;

  // ========== Per-molecule descriptors ==========

  //! One MolDescriptor per molecule [nMols]
  AsyncDeviceVector<MolDescriptor> molDescriptors;
  //! CSR: cumulative dihedral work items [nMols + 1]
  AsyncDeviceVector<int>           dihedralWorkStarts;
  //! CSR: cumulative TFD pair work items [nMols + 1]
  AsyncDeviceVector<int>           tfdWorkStarts;

  // ========== Output buffers ==========

  //! Output: computed dihedral angles [totalDihedrals]
  AsyncDeviceVector<float> dihedralAngles;
  //! Output: TFD matrix values
  AsyncDeviceVector<float> tfdOutput;

  //! Set CUDA stream for all buffers
  void setStream(cudaStream_t stream);
};

//! Extract torsion list from an RDKit molecule
//! @param mol The molecule to analyze
//! @param maxDevMode How to determine max deviation for normalization
//! @param symmRadius Radius for Morgan fingerprint (symmetry detection)
//! @param ignoreColinearBonds Whether to skip bonds adjacent to triple bonds
//! @return TorsionList containing non-ring and ring torsions
TorsionList extractTorsionList(const RDKit::ROMol& mol,
                               TFDMaxDevMode       maxDevMode          = TFDMaxDevMode::Equal,
                               int                 symmRadius          = 2,
                               bool                ignoreColinearBonds = true);

//! Calculate distance-based weights for torsions
//! @param mol The molecule
//! @param torsionList Previously extracted torsion list
//! @param ignoreColinearBonds Must match the value used in extractTorsionList
//! @return Vector of weights (size = total torsion count)
std::vector<float> computeTorsionWeights(const RDKit::ROMol& mol,
                                         const TorsionList&  torsionList,
                                         bool                ignoreColinearBonds = true);

//! Transfer host system data to device, resizing and allocating output buffers
//! @param host Source host data
//! @param device Destination device data (resized as needed)
//! @param stream CUDA stream for async transfers
void transferToDevice(const TFDSystemHost& host, TFDSystemDevice& device, cudaStream_t stream);

//! Build TFDSystemHost from a batch of molecules
//! @param mols Vector of molecules (each may have multiple conformers)
//! @param options Computation options
//! @return Populated TFDSystemHost ready for GPU transfer
TFDSystemHost buildTFDSystem(const std::vector<const RDKit::ROMol*>& mols, const TFDComputeOptions& options);

//! Build TFDSystemHost from a single molecule (convenience wrapper)
TFDSystemHost buildTFDSystem(const RDKit::ROMol& mol, const TFDComputeOptions& options);

}  // namespace nvMolKit

#endif  // NVMOLKIT_TFD_COMMON_H
