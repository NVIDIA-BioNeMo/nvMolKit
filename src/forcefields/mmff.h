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

#ifndef NVMOLKIT_MMFF_H
#define NVMOLKIT_MMFF_H

#include <cstdint>
#include <functional>
#include <vector>

#include "batched_forcefield.h"
#include "device_vector.h"
#include "mmff_kernels.h"
namespace nvMolKit {
namespace MMFF {

// -----------------------------------------------
// Device and host structs for MMFF contrib terms.
//
// For references on the MMFF forcefield equations, see
// https://www.charmm-gui.org/charmmdoc/mmff.html
// or
// https://docs.eyesopen.com/toolkits/python/oefftk/fftheory.html#mmff
// -----------------------------------------------
struct BondStretchContribTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<double> r0;
  std::vector<double> kb;
};

struct AngleBendTerms {
  std::vector<int>          idx1;
  std::vector<int>          idx2;
  std::vector<int>          idx3;
  std::vector<double>       theta0;
  std::vector<double>       ka;
  std::vector<std::uint8_t> isLinear;
};

struct BendStretchTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<int>    idx3;
  std::vector<double> theta0;
  std::vector<double> restLen1;
  std::vector<double> restLen2;
  std::vector<double> forceConst1;
  std::vector<double> forceConst2;
};

struct OutOfPlaneTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<int>    idx3;
  std::vector<int>    idx4;
  std::vector<double> koop;
};

struct TorsionContribTerms {
  std::vector<int>   idx1;
  std::vector<int>   idx2;
  std::vector<int>   idx3;
  std::vector<int>   idx4;
  std::vector<float> V1;
  std::vector<float> V2;
  std::vector<float> V3;
};

struct VdwTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<double> R_ij_star;
  std::vector<double> wellDepth;
};

struct EleTerms {
  std::vector<int>     idx1;
  std::vector<int>     idx2;
  std::vector<double>  chargeTerm;
  std::vector<uint8_t> dielModel;
  std::vector<uint8_t> is1_4;
};

struct DistanceConstraintTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<double> minLen;
  std::vector<double> maxLen;
  std::vector<double> forceConstant;
};

struct PositionConstraintTerms {
  std::vector<int>    idx;
  std::vector<double> refX;
  std::vector<double> refY;
  std::vector<double> refZ;
  std::vector<double> maxDispl;
  std::vector<double> forceConstant;
};

struct AngleConstraintTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<int>    idx3;
  std::vector<double> minAngleDeg;
  std::vector<double> maxAngleDeg;
  std::vector<double> forceConstant;
};

struct TorsionConstraintTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<int>    idx3;
  std::vector<int>    idx4;
  std::vector<double> minDihedralDeg;
  std::vector<double> maxDihedralDeg;
  std::vector<double> forceConstant;
};

struct EnergyForceContribsHost {
  BondStretchContribTerms bondTerms;
  AngleBendTerms          angleTerms;
  BendStretchTerms        bendTerms;
  OutOfPlaneTerms         oopTerms;
  TorsionContribTerms     torsionTerms;
  VdwTerms                vdwTerms;
  EleTerms                eleTerms;
  DistanceConstraintTerms distanceConstraintTerms;
  PositionConstraintTerms positionConstraintTerms;
  AngleConstraintTerms    angleConstraintTerms;
  TorsionConstraintTerms  torsionConstraintTerms;
};

//! Modifies a single MMFF system's contribs before they are flattened into the batched buffers.
//! The callback receives the recorded system metadata, the source coordinates for that system,
//! and a mutable copy of the per-system MMFF contribs.
using ForcefieldModifier =
  std::function<void(const BatchedSystemInfo&, const std::vector<double>&, EnergyForceContribsHost&)>;

struct BatchedIndicesHost {
  //! Size n_molecules + 1, defines the start and end of each molecule in the batch.
  //! The last element contains the number of atoms in the system.
  std::vector<int> atomStarts         = {0};
  //! Defines the start of each molecule's energy buffer region that will be added to then reduced.
  std::vector<int> energyBufferStarts = {0};
  //! Size total atoms, maps atom index to batch index.
  std::vector<int> atomIdxToBatchIdx;
  //! Size total energy buffer blocks, maps energy buffer block index to batch index.
  std::vector<int> energyBufferBlockIdxToBatchIdx;
  //! Size n_molecules, defines the start and end of each molecule's bond term count
  std::vector<int> bondTermStarts               = {0};
  //! Size n_molecules, defines the start and end of each molecule's angle term count
  std::vector<int> angleTermStarts              = {0};
  //! Size n_molecules, defines the start and end of each molecule's bend term count
  std::vector<int> bendTermStarts               = {0};
  //! Size n_molecules, defines the start and end of each molecule's oop term count
  std::vector<int> oopTermStarts                = {0};
  //! Size n_molecules, defines the start and end of each molecule's torsion term count
  std::vector<int> torsionTermStarts            = {0};
  //! Size n_molecules, defines the start and end of each molecule's vdw term count
  std::vector<int> vdwTermStarts                = {0};
  //! Size n_molecules, defines the start and end of each molecule's ele term count
  std::vector<int> eleTermStarts                = {0};
  //! Size n_molecules, defines the start and end of each molecule's distance constraint term count
  std::vector<int> distanceConstraintTermStarts = {0};
  //! Size n_molecules, defines the start and end of each molecule's position constraint term count
  std::vector<int> positionConstraintTermStarts = {0};
  //! Size n_molecules, defines the start and end of each molecule's angle constraint term count
  std::vector<int> angleConstraintTermStarts    = {0};
  //! Size n_molecules, defines the start and end of each molecule's torsion constraint term count
  std::vector<int> torsionConstraintTermStarts  = {0};
};

struct BatchedMolecularSystemHost {
  EnergyForceContribsHost contribs;
  BatchedIndicesHost      indices;
  //! Size total num atoms * 3
  std::vector<double>     positions;

  //! Largest system size in the batch
  int maxNumAtoms = 0;
};

struct BondStretchContribTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;
  nvMolKit::AsyncDeviceVector<int>    idx2;
  nvMolKit::AsyncDeviceVector<double> r0;
  nvMolKit::AsyncDeviceVector<double> kb;
};

struct AngleBendTermsDevice {
  nvMolKit::AsyncDeviceVector<int>          idx1;
  nvMolKit::AsyncDeviceVector<int>          idx2;
  nvMolKit::AsyncDeviceVector<int>          idx3;
  nvMolKit::AsyncDeviceVector<double>       theta0;
  nvMolKit::AsyncDeviceVector<double>       ka;
  nvMolKit::AsyncDeviceVector<std::uint8_t> isLinear;
};

struct BendStretchTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;
  nvMolKit::AsyncDeviceVector<int>    idx2;
  nvMolKit::AsyncDeviceVector<int>    idx3;
  nvMolKit::AsyncDeviceVector<double> theta0;
  nvMolKit::AsyncDeviceVector<double> restLen1;
  nvMolKit::AsyncDeviceVector<double> restLen2;
  nvMolKit::AsyncDeviceVector<double> forceConst1;
  nvMolKit::AsyncDeviceVector<double> forceConst2;
};

struct OutOfPlaneTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;
  nvMolKit::AsyncDeviceVector<int>    idx2;
  nvMolKit::AsyncDeviceVector<int>    idx3;
  nvMolKit::AsyncDeviceVector<int>    idx4;
  nvMolKit::AsyncDeviceVector<double> koop;
};

struct TorsionContribTermsDevice {
  nvMolKit::AsyncDeviceVector<int>   idx1;
  nvMolKit::AsyncDeviceVector<int>   idx2;
  nvMolKit::AsyncDeviceVector<int>   idx3;
  nvMolKit::AsyncDeviceVector<int>   idx4;
  nvMolKit::AsyncDeviceVector<float> V1;
  nvMolKit::AsyncDeviceVector<float> V2;
  nvMolKit::AsyncDeviceVector<float> V3;
};

struct VdwTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;
  nvMolKit::AsyncDeviceVector<int>    idx2;
  nvMolKit::AsyncDeviceVector<double> R_ij_star;
  nvMolKit::AsyncDeviceVector<double> wellDepth;
};

struct EleTermsDevice {
  nvMolKit::AsyncDeviceVector<int>     idx1;
  nvMolKit::AsyncDeviceVector<int>     idx2;
  nvMolKit::AsyncDeviceVector<double>  chargeTerm;
  nvMolKit::AsyncDeviceVector<uint8_t> dielModel;
  nvMolKit::AsyncDeviceVector<uint8_t> is1_4;
};

struct DistanceConstraintTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;
  nvMolKit::AsyncDeviceVector<int>    idx2;
  nvMolKit::AsyncDeviceVector<double> minLen;
  nvMolKit::AsyncDeviceVector<double> maxLen;
  nvMolKit::AsyncDeviceVector<double> forceConstant;
};

struct PositionConstraintTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx;
  nvMolKit::AsyncDeviceVector<double> refX;
  nvMolKit::AsyncDeviceVector<double> refY;
  nvMolKit::AsyncDeviceVector<double> refZ;
  nvMolKit::AsyncDeviceVector<double> maxDispl;
  nvMolKit::AsyncDeviceVector<double> forceConstant;
};

struct AngleConstraintTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;
  nvMolKit::AsyncDeviceVector<int>    idx2;
  nvMolKit::AsyncDeviceVector<int>    idx3;
  nvMolKit::AsyncDeviceVector<double> minAngleDeg;
  nvMolKit::AsyncDeviceVector<double> maxAngleDeg;
  nvMolKit::AsyncDeviceVector<double> forceConstant;
};

struct TorsionConstraintTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;
  nvMolKit::AsyncDeviceVector<int>    idx2;
  nvMolKit::AsyncDeviceVector<int>    idx3;
  nvMolKit::AsyncDeviceVector<int>    idx4;
  nvMolKit::AsyncDeviceVector<double> minDihedralDeg;
  nvMolKit::AsyncDeviceVector<double> maxDihedralDeg;
  nvMolKit::AsyncDeviceVector<double> forceConstant;
};

struct EnergyForceContribsDevice {
  BondStretchContribTermsDevice bondTerms;
  AngleBendTermsDevice          angleTerms;
  BendStretchTermsDevice        bendTerms;
  OutOfPlaneTermsDevice         oopTerms;
  TorsionContribTermsDevice     torsionTerms;
  VdwTermsDevice                vdwTerms;
  EleTermsDevice                eleTerms;
  DistanceConstraintTermsDevice distanceConstraintTerms;
  PositionConstraintTermsDevice positionConstraintTerms;
  AngleConstraintTermsDevice    angleConstraintTerms;
  TorsionConstraintTermsDevice  torsionConstraintTerms;
};

//! See BatchedIndices for more information on each field.
struct BatchedIndicesDevice {
  nvMolKit::AsyncDeviceVector<int> atomStarts;
  nvMolKit::AsyncDeviceVector<int> atomIdxToBatchIdx;
  nvMolKit::AsyncDeviceVector<int> energyBufferStarts;
  nvMolKit::AsyncDeviceVector<int> energyBufferBlockIdxToBatchIdx;

  nvMolKit::AsyncDeviceVector<int> bondTermStarts;
  nvMolKit::AsyncDeviceVector<int> angleTermStarts;
  nvMolKit::AsyncDeviceVector<int> bendTermStarts;
  nvMolKit::AsyncDeviceVector<int> oopTermStarts;
  nvMolKit::AsyncDeviceVector<int> torsionTermStarts;
  nvMolKit::AsyncDeviceVector<int> vdwTermStarts;
  nvMolKit::AsyncDeviceVector<int> eleTermStarts;
  nvMolKit::AsyncDeviceVector<int> distanceConstraintTermStarts;
  nvMolKit::AsyncDeviceVector<int> positionConstraintTermStarts;
  nvMolKit::AsyncDeviceVector<int> angleConstraintTermStarts;
  nvMolKit::AsyncDeviceVector<int> torsionConstraintTermStarts;
};

//! Device buffers for the batched molecular system.
//! Most of the terms are either 1 per molecule or CSR-like format with the BatchedIndicesDevice terms used for
//! indexing.
//!
//! The only nonstandard term is the energyBuffer and associated indices, which is allocated and used the following way.
//! - First, the maximum energy term size for each molecule is calculated. For example, a molecule with 10 bond terms,
//!   20 angle terms, and 5 bend terms would have a max term size of 20.
//! - Next, the energy buffer per molecule is rounded up to the energy reduction block size. See
//!   blockSizeEnergyReduction in kernel_utils.cuh. So, a molecule with a max term of 150 will be allocated
//!   256 energy buffer positions.
//! - Indices are built off of this data. The energyBufferStarts field in BatchedIndicesDevice defines the start of each
//!   buffer region. It is guaranteed to be an offset of blockSizeEnergyReduction but this is not critical to the
//!   algorithm. The energyBufferBlockIdxToBatchIdx maps each block to the molecule it belongs to.
//! - When computing energies, each term adds into term index + energyBufferStarts[moleculeIdx] in the energy buffer.
//!   This means that for smaller terms, the energy buffer will have some unused space, and unless the largest term is
//!   a multiple of the block size, there will be some zero elements. This is fine and expected.
//!   Finally, on reduction, each block does a local summation, then atomically adds to the output energy for the
//!   molecule, using the energyBufferBlockIdxToBatchIdx to map the block to the molecule output index.
struct BatchedMolecularDeviceBuffers {
  EnergyForceContribsDevice           contribs;
  //! Size n_molecules
  BatchedIndicesDevice                indices;
  //! Size total num atoms * 3
  nvMolKit::AsyncDeviceVector<double> positions;
  //! Size total num atoms * 3
  nvMolKit::AsyncDeviceVector<double> grad;
  //! Variable size - max terms in each molecule concatenated.
  //! Each molecule has an energy buffer to add to and reduce to energyOuts.
  nvMolKit::AsyncDeviceVector<double> energyBuffer;
  //! Size n_molecules
  nvMolKit::AsyncDeviceVector<double> energyOuts;
};

//! \brief Adds a molecule to the batched MMFF system.
//! \param contribs MMFF terms for the molecule before flattening into the batch.
//! \param positions Source coordinates for the molecule.
//! \param molSystem Output batched molecular system.
//! \param metadata Optional mapping from concrete systems back to logical molecules.
//!        When null, no logical molecule metadata is recorded.
//! \param moleculeIdx Logical molecule index to associate with the new system when metadata is provided.
//! \param conformerIdx Conformer index within the logical molecule when metadata is provided.
//! \param customization Optional callback that runs after any metadata is recorded and before
//!        the contribs are flattened into the batch.
void addMoleculeToBatch(const EnergyForceContribsHost& contribs,
                        const std::vector<double>&     positions,
                        BatchedMolecularSystemHost&    molSystem,
                        BatchedForcefieldMetadata*     metadata      = nullptr,
                        int                            moleculeIdx   = -1,
                        int                            conformerIdx  = -1,
                        const ForcefieldModifier&      customization = {});

//! Send the batched molecular system to the device.
void sendContribsAndIndicesToDevice(const BatchedMolecularSystemHost& molSystemHost,
                                    BatchedMolecularDeviceBuffers&    molSystemDevice);

//! Sets all device vector streams
void setStreams(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream);

//! Allocate intermediate buffers on the device for the batched molecular system.
//! These include the gradients, energy buffer, and energy outs.
void allocateIntermediateBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                 BatchedMolecularDeviceBuffers&    molSystemDevice);

//! Compute energies into caller-provided output buffer.
//! energyOuts and molSystemDevice.energyBuffer must be zeroed before calling.
cudaError_t computeEnergy(BatchedMolecularDeviceBuffers& molSystemDevice,
                          double*                        energyOuts,
                          const double*                  positions,
                          const uint8_t*                 activeSystemMask = nullptr,
                          cudaStream_t                   stream           = nullptr);

//! Compute the energy of the batched molecular system. This will populate the energyOuts buffer on device.
//! energyOuts and energyBuffer must be zeroed before calling this function.
//! Optionally computes on user-provided coordinates rather than those in molSystemDevice.
//! If not null, coords must be GPU resident. The molSystemDevice intermediate system description,
//! energy accumulator and output buffers are always used, only the coordinates are swappable.
cudaError_t computeEnergy(BatchedMolecularDeviceBuffers& molSystemDevice,
                          const double*                  coords = nullptr,
                          cudaStream_t                   stream = nullptr);

cudaError_t computeEnergyBlockPerMol(BatchedMolecularDeviceBuffers& molSystemDevice,
                                     const double*                  coords = nullptr,
                                     cudaStream_t                   stream = nullptr);
//! Compute the gradients of the batched molecular system. This will populate the grad buffer on device.
//! grad must be zeroed before calling this function.
cudaError_t computeGradients(BatchedMolecularDeviceBuffers& molSystemDevice,
                             const double*                  positions,
                             double*                        grad,
                             const uint8_t*                 activeSystemMask = nullptr,
                             cudaStream_t                   stream           = nullptr);

cudaError_t computeGradients(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream = nullptr);

cudaError_t computeGradBlockPerMol(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream = nullptr);

//! Create pointer struct from device buffers for use in per-molecule kernels
EnergyForceContribsDevicePtr toEnergyForceContribsDevicePtr(const BatchedMolecularDeviceBuffers& molSystemDevice);

//! Create pointer struct from device indices for use in per-molecule kernels
BatchedIndicesDevicePtr toBatchedIndicesDevicePtr(const BatchedMolecularDeviceBuffers& molSystemDevice);

//! Returns true if any molecule in the batch contributes a distance, position, angle, or torsion
//! constraint term. Used by per-molecule kernels to dispatch to a specialization that compiles out
//! the constraint loops, recovering register pressure when no constraints are active.
bool batchHasConstraints(const EnergyForceContribsDevice& contribs);

}  // namespace MMFF
}  // namespace nvMolKit

#endif  // NVMOLKIT_MMFF_H
