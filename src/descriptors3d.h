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
  PMI1                 = 0,
  PMI2                 = 1,
  PMI3                 = 2,
  RadiusOfGyration     = 3,
  NPR1                 = 4,
  NPR2                 = 5,
  InertialShapeFactor  = 6,
  Eccentricity         = 7,
  Asphericity          = 8,
  SpherocityIndex      = 9,
  PBF                  = 10,
  WHIM                 = 11,
  RDF                  = 12,
  MORSE                = 13,
  AUTOCORR3D           = 14,
  USR                  = 15,
  USRCAT               = 16,
  GETAWAY              = 17,
  //! RDKit's DoubleCubicLatticeVolume getters (GetSurfaceArea, ...).
  DCLVSurfaceArea      = 18,
  DCLVPolarSurfaceArea = 19,
  DCLVVolume           = 20,
  DCLVVDWVolume        = 21,
  DCLVPolarVolume      = 22,
  DCLVCompactness      = 23,
  DCLVPackingDensity   = 24,
};

inline constexpr std::array<Property3D, 25> kAllProperty3D = {
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
  Property3D::RDF,
  Property3D::MORSE,
  Property3D::AUTOCORR3D,
  Property3D::USR,
  Property3D::USRCAT,
  Property3D::GETAWAY,
  Property3D::DCLVSurfaceArea,
  Property3D::DCLVPolarSurfaceArea,
  Property3D::DCLVVolume,
  Property3D::DCLVVDWVolume,
  Property3D::DCLVPolarVolume,
  Property3D::DCLVCompactness,
  Property3D::DCLVPackingDensity,
};

inline constexpr int kNumWhimProperties       = 114;
inline constexpr int kNumRdfProperties        = 210;  //!< 7 atom-property channels x 30 radii.
inline constexpr int kNumMorseProperties      = 224;  //!< 7 atom-property channels x 32 scattering values.
inline constexpr int kNumAutocorr3DProperties = 80;   //!< 8 atom-property channels x 10 topological lags.
inline constexpr int kNumUsrProperties        = 12;   //!< 3 distance moments x 4 reference points.
inline constexpr int kNumUsrcatProperties     = 60;   //!< USR of all atoms, then of each of 4 atom classes.
inline constexpr int kNumGetawayProperties    = 273;  //!< RDKit's GETAWAY vector (CalcGETAWAY order).

//! Canonical name of @p property, e.g. "PMI1" or "RadiusOfGyration".
std::string_view property3DName(Property3D property);

//! Parse a canonical property name. @throws std::invalid_argument for unknown names.
Property3D property3DFromName(std::string_view name);

//! Number of values emitted per conformer. Scalar properties have width one.
constexpr int property3DWidth(const Property3D property) {
  switch (property) {
    case Property3D::WHIM:
      return kNumWhimProperties;
    case Property3D::RDF:
      return kNumRdfProperties;
    case Property3D::MORSE:
      return kNumMorseProperties;
    case Property3D::AUTOCORR3D:
      return kNumAutocorr3DProperties;
    case Property3D::USR:
      return kNumUsrProperties;
    case Property3D::USRCAT:
      return kNumUsrcatProperties;
    case Property3D::GETAWAY:
      return kNumGetawayProperties;
    default:
      return 1;
  }
}

//! Properties computed together because they share per-conformer work; each family has its own kernel,
//! device inputs and options.
enum class Property3DFamily : int {
  Moments,     //!< Inertia/gyration tensor eigenvalues: PMI, NPR, RadiusOfGyration and derived shape indices.
  Projection,  //!< Coordinate PCA and projections onto its axes: PBF and WHIM.
  Pairwise,    //!< Sums over atom pairs weighted by atom-property pairs: RDF, MORSE and AUTOCORR3D.
  Usr,         //!< Distance moments from four reference points: USR and USRCAT.
  Getaway,     //!< Leverage (molecular influence) matrix descriptors: GETAWAY.
  Dclv,        //!< Dot-surface areas and volumes of the atom spheres: RDKit's DoubleCubicLatticeVolume (DCLV*).
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
    case Property3D::RDF:
    case Property3D::MORSE:
    case Property3D::AUTOCORR3D:
      return Property3DFamily::Pairwise;
    case Property3D::USR:
    case Property3D::USRCAT:
      return Property3DFamily::Usr;
    case Property3D::GETAWAY:
      return Property3DFamily::Getaway;
    case Property3D::DCLVSurfaceArea:
    case Property3D::DCLVPolarSurfaceArea:
    case Property3D::DCLVVolume:
    case Property3D::DCLVVDWVolume:
    case Property3D::DCLVPolarVolume:
    case Property3D::DCLVCompactness:
    case Property3D::DCLVPackingDensity:
      return Property3DFamily::Dclv;
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

//! Options for GETAWAY.
struct GetawayOptions {
  //! Significant digits the heavy-atom leverages are rounded to before ITH and ISH cluster them; RDKit's
  //! default. Must be between 1 and 6.
  unsigned int precision = 2;
};

//! Property3DDeviceInputs::dclvPolarClasses bits: the atom kinds RDKit's DoubleCubicLatticeVolume counts as polar.
enum DclvPolarClass : uint8_t {
  kDclvPolarNitrogenOxygen       = 1u << 0,  //!< Nitrogen or oxygen; always polar.
  kDclvPolarSulfurPhosphorus     = 1u << 1,  //!< Sulfur or phosphorus; polar with DclvOptions::includeSandP.
  kDclvPolarHydrogenOnNO         = 1u << 2,  //!< Hydrogen bonded to nitrogen or oxygen; with includeHs.
  kDclvPolarHydrogenOnSulfurPhos = 1u << 3,  //!< Hydrogen bonded to sulfur or phosphorus; with both options.
};

//! Options for the DCLV properties (RDKit's DoubleCubicLatticeVolume), with RDKit's defaults.
struct DclvOptions {
  //! Radius of the solvent probe sphere in Angstrom. Must be finite and non-negative.
  double probeRadius  = 1.4;
  //! DCLVPolarSurfaceArea and DCLVPolarVolume also count sulfur and phosphorus (RDKit's includeSandP).
  bool   includeSandP = false;
  //! The polar DCLV properties also count hydrogens bonded to a polar atom (RDKit's includeHs).
  bool   includeHs    = false;
};

//! Per-family options; each family reads only its own member. PBF, RDF, MORSE, AUTOCORR3D, USR and
//! USRCAT have no options.
struct Property3DOptions {
  MomentOptions  moments;
  WhimOptions    whim;
  GetawayOptions getaway;
  DclvOptions    dclv;
};

/**
 * @brief Device inputs for calc3DPropertiesGpu(). Per-atom arrays are stored once per molecule, indexed
 *        through @ref moleculeAtomStarts, and resolved per conformer via DeviceCoordView::molIndices.
 *
 * Each member is read only when a property of the family noted beside it is requested.
 */
struct Property3DDeviceInputs {
  //! All families: CSR offsets of each molecule's atoms, length `nMols + 1`. Required for a non-empty batch.
  const int32_t* moleculeAtomStarts    = nullptr;
  //! Moments: one weight per atom; null gives every atom unit weight.
  const double*  momentWeights         = nullptr;
  //! WHIM, RDF, MORSE, AUTOCORR3D, GETAWAY: six atom-property channels (RDKit's relative mass, van der Waals volume,
  //! electronegativity, polarizability and ionization potential, then I-state), channel-major with one
  //! value per atom. Required when any of them is requested.
  const double*  atomPropertyWeights   = nullptr;
  //! RDF: RDKit's I-state (GetIStateDrag), one value per atom, used in place of the I-state channel.
  //! Required when RDF is requested.
  const double*  iStateDragWeights     = nullptr;
  //! AUTOCORR3D: RDKit's relative covalent radius (GetRelativeRcov), one value per atom.
  const double*  covalentRadiusWeights = nullptr;
  //! AUTOCORR3D, GETAWAY: bond adjacency in CSR form, both required (with `bondNeighbors` non-null even when no
  //! molecule has a bond). Atom g (indexed like @ref moleculeAtomStarts, `totalAtoms + 1` starts) has
  //! neighbors `bondNeighbors[bondNeighborStarts[g] .. bondNeighborStarts[g + 1])`, stored as atom indices
  //! within its molecule.
  const int32_t* bondNeighborStarts    = nullptr;
  const int32_t* bondNeighbors         = nullptr;
  //! USRCAT: per atom, bit c set when the atom is in RDKit's USRCAT class c (hydrophobic, aromatic,
  //! acceptor, donor).
  const uint8_t* usrcatAtomClasses     = nullptr;
  //! GETAWAY: per atom, 1 for heavy atoms (atomic number above 1), else 0.
  const uint8_t* heavyAtomFlags        = nullptr;
  //! GETAWAY: per molecule, the coordinate row of its default (first) conformer, whose PBF decides HIC's
  //! dimension as in RDKit. Null, or a negative entry, uses each row's own conformer.
  const int32_t* defaultConformerRows  = nullptr;
  //! DCLV: per atom, RDKit's van der Waals radius (PeriodicTable::getRvdw). Conformers with a radius-0 atom (dummy
  //! atoms, on which RDKit's DoubleCubicLatticeVolume crashes) produce NaN for every DCLV property.
  const double*  vdwRadii              = nullptr;
  //! DCLV: per atom, its DclvPolarClass bits. DclvOptions selects the classes DCLVPolarSurfaceArea and
  //! DCLVPolarVolume count.
  const uint8_t* dclvPolarClasses      = nullptr;
  //! PBF, GETAWAY: per-conformer RDKit is3D flags (one per coordinate row); null treats every row as 3D.
  const int8_t*  conformerIs3D         = nullptr;
  //! WHIM, AUTOCORR3D, GETAWAY: largest molecule atom count in the batch (host value). Sizes WHIM's
  //! per-conformer symmetry-search scratch (rows with more atoms produce NaN) and, with 0 or above 3072 atoms,
  //! leaves AUTOCORR3D and GETAWAY without the shared bond-distance table.
  int32_t        maxMoleculeAtoms      = 0;
  //! AUTOCORR3D, GETAWAY: atom pairs n (n - 1) / 2 summed over the batch's molecules (host value); sizes the
  //! shared bond-distance table. 0 sizes it as if every molecule had @ref maxMoleculeAtoms atoms.
  int64_t        moleculeAtomPairs     = 0;
};

//! One row-major device vector of length numConformers * property3DWidth(property) per property.
//! @p Real is float (PrecisionMode::SINGLE) or double (PrecisionMode::FULL).
template <typename Real> using Property3DResults = std::unordered_map<Property3D, AsyncDeviceVector<Real>>;

/**
 * @brief Calculate the requested 3D properties for every conformer in a coordinate batch.
 *
 * Each requested family runs as one kernel launch and returns @p Real (float or double), computing in
 * @p Real except for steps where float32 measurably loses accuracy, which are always float64: coordinate
 * centering (descriptors3d_detail::centeredPosition), WHIM's PCA (descriptors3d_detail::WhimReal),
 * MORSE's zero-scattering bin (descriptors3d_detail::computeMorseZeroBin), USR's reference-atom selection
 * (descriptors3d_detail::extremeAtom) and GETAWAY's leverages (descriptors3d_detail::computeLeverages).
 * `options.moments` is expressed through `inputs.momentWeights` at this level. Conformers whose molecule
 * index is out of range, whose atom range lies outside `coordinates.numAtoms`, or whose atom count
 * disagrees with the molecule's atom range produce NaN for every requested property.
 *
 * @throws std::invalid_argument if @p properties is empty, contains duplicates, or contains a value
 *                               outside kAllProperty3D; if WHIM is requested and
 *                               `options.whim.threshold` is negative or not finite; if GETAWAY is
 *                               requested and `options.getaway.precision` is outside [1, 6]; if
 *                               a DCLV property is requested and `options.dclv.probeRadius` is
 *                               negative or not finite; or if a required
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
