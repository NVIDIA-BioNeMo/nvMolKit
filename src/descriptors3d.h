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
  PBF                 = 10,
  WHIM                = 11,
};

inline constexpr std::array<Property3D, 12> kAllProperty3D = {
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
  Property3D::PBF,
  Property3D::WHIM,
};

inline constexpr int kNumWhimProperties = 114;

//! Canonical name of @p property, e.g. "PMI1" or "RadiusOfGyration".
std::string_view property3DName(Property3D property);

//! Parse a canonical property name. @throws std::invalid_argument for unknown names.
Property3D property3DFromName(std::string_view name);

//! Number of values emitted per conformer. Scalar properties have width one.
constexpr int property3DWidth(const Property3D property) {
  return property == Property3D::WHIM ? kNumWhimProperties : 1;
}

//! Properties computed together because they share per-conformer work; each family has its own kernel,
//! device inputs and options.
enum class Property3DFamily : int {
  Moments,     //!< Inertia/gyration tensor eigenvalues: PMI, NPR, RadiusOfGyration and derived shape indices.
  Projection,  //!< Coordinate PCA and projections onto its axes: PBF and WHIM.
};

constexpr Property3DFamily property3DFamily(const Property3D property) {
  switch (property) {
    case Property3D::PMI1:
    case Property3D::PMI2:
    case Property3D::PMI3:
    case Property3D::RadiusOfGyration:
    case Property3D::NPR1:
    case Property3D::NPR2:
    case Property3D::InertialShapeFactor:
    case Property3D::Eccentricity:
    case Property3D::Asphericity:
    case Property3D::SpherocityIndex:
      return Property3DFamily::Moments;
    case Property3D::PBF:
    case Property3D::WHIM:
      return Property3DFamily::Projection;
  }
  return Property3DFamily::Moments;
}

//! Options for the Moments family.
struct MomentOptions {
  //! Weight atoms by mass (RDKit's default) instead of unit weights. SpherocityIndex is always unweighted.
  bool useAtomicMasses = true;
};

//! Options for WHIM.
struct WhimOptions {
  //! Maximum projected-coordinate difference counted as symmetric; RDKit's default. Must be finite and
  //! non-negative.
  double threshold = 0.001;
};

//! Per-family options; each family reads only its own member. PBF has no options.
struct Property3DOptions {
  MomentOptions moments;
  WhimOptions   whim;
};

/**
 * @brief Device inputs for calc3DPropertiesGpu(). Per-atom arrays are stored once per molecule, indexed
 *        through @ref moleculeAtomStarts, and resolved per conformer via DeviceCoordView::molIndices.
 *
 * Each member is read only when a property of the family noted beside it is requested.
 */
struct Property3DDeviceInputs {
  //! All families: CSR offsets of each molecule's atoms, length `nMols + 1`. Required for a non-empty batch.
  const int32_t* moleculeAtomStarts = nullptr;
  //! Moments: one weight per atom; null gives every atom unit weight.
  const double*  momentWeights      = nullptr;
  //! WHIM: six atom-property channels (mass, van der Waals volume, electronegativity, polarizability,
  //! ionization potential, I-state), channel-major with one value per atom. Required when WHIM is requested.
  const double*  whimWeights        = nullptr;
  //! PBF: per-conformer RDKit is3D flags (one per coordinate row); null treats every row as 3D.
  const int8_t*  conformerIs3D      = nullptr;
  //! WHIM: largest molecule atom count in the batch (host value); sizes the per-conformer symmetry-search
  //! scratch. Rows with more atoms produce NaN.
  int32_t        maxMoleculeAtoms   = 0;
};

//! One row-major device vector of length numConformers * property3DWidth(property) per property.
//! @p Real is float (PrecisionMode::SINGLE) or double (PrecisionMode::FULL).
template <typename Real> using Property3DResults = std::unordered_map<Property3D, AsyncDeviceVector<Real>>;

/**
 * @brief Calculate the requested 3D properties for every conformer in a coordinate batch.
 *
 * Each requested family runs as one kernel launch. Outputs use @p Real (float or double); Moments and a
 * PBF requested without WHIM compute in @p Real, while WHIM (and a PBF requested with it) computes in
 * float64 because WHIM's three-decimal rounding and symmetry matching are unstable in float32.
 * `options.moments` is expressed through `inputs.momentWeights` at this level. Conformers whose molecule
 * index is out of range, whose atom range lies outside `coordinates.numAtoms`, or whose atom count
 * disagrees with the molecule's atom range produce NaN for every requested property.
 *
 * @throws std::invalid_argument if @p properties is empty, contains duplicates, or contains a value
 *                               outside kAllProperty3D; if WHIM is requested and
 *                               `options.whim.threshold` is negative or not finite; or if a required
 *                               input is null.
 */
template <typename Real>
Property3DResults<Real> calc3DPropertiesGpu(const DeviceCoordView&         coordinates,
                                            const Property3DDeviceInputs&  inputs,
                                            const std::vector<Property3D>& properties,
                                            const Property3DOptions&       options,
                                            cudaStream_t                   stream);

}  // namespace nvMolKit

#endif  // NVMOLKIT_DESCRIPTORS3D_H
