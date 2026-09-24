// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_H
#define NVMOLKIT_DESCRIPTORS3D_H

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/conformer/device_coord_result.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

//! Per-conformer 3D properties. Names match the corresponding RDKit descriptor names.
enum class Property3D : int {
  PMI1                = 0,
  PMI2                = 1,
  PMI3                = 2,
  RadiusOfGyration    = 3,
  NPR1                = 4,
  NPR2                = 5,
  InertialShapeFactor = 6,
  Eccentricity        = 7,
  Asphericity         = 8,
  SpherocityIndex     = 9,
};

inline constexpr std::array<Property3D, 10> kAllProperty3D = {
  Property3D::PMI1,
  Property3D::PMI2,
  Property3D::PMI3,
  Property3D::RadiusOfGyration,
  Property3D::NPR1,
  Property3D::NPR2,
  Property3D::InertialShapeFactor,
  Property3D::Eccentricity,
  Property3D::Asphericity,
  Property3D::SpherocityIndex,
};

//! Canonical name of @p property, e.g. "PMI1" or "RadiusOfGyration".
std::string_view property3DName(Property3D property);

//! Parse a canonical property name. @throws std::invalid_argument for unknown names.
Property3D property3DFromName(std::string_view name);

//! One device vector of length numConformers per requested property. @p Real is float
//! (PrecisionMode::SINGLE) or double (PrecisionMode::FULL).
template <typename Real> using Property3DResults = std::unordered_map<Property3D, AsyncDeviceVector<Real>>;

/**
 * @brief Calculate the requested 3D properties for every conformer in a coordinate batch.
 *
 * All arithmetic, reductions, and outputs use @p Real (float or double); double-precision inputs
 * are converted on load. Optional atom weights used by mass-sensitive properties are stored once
 * per molecule in CSR form (@p moleculeAtomStarts, length `coordinates.nMols + 1`) and resolved per
 * conformer through `coordinates.molIndices`. A null @p atomWeights pointer gives every atom unit
 * weight. SpherocityIndex follows RDKit and always uses unit weights. Conformers whose molecule
 * index is out of range, whose atom range lies
 * outside `coordinates.numAtoms`, or whose atom count disagrees with the molecule's weight range
 * produce NaN for every requested property.
 *
 * @throws std::invalid_argument if @p properties is empty, contains duplicates, or contains a value
 *                               outside kAllProperty3D.
 */
template <typename Real>
Property3DResults<Real> calc3DPropertiesGpu(const DeviceCoordView&         coordinates,
                                            const double*                  atomWeights,
                                            const int32_t*                 moleculeAtomStarts,
                                            const std::vector<Property3D>& properties,
                                            cudaStream_t                   stream);

}  // namespace nvMolKit

#endif  // NVMOLKIT_DESCRIPTORS3D_H
